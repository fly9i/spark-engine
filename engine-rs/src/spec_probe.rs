//! End-to-end greedy speculation. Serial verifier is a correctness baseline.
use std::{path::Path,time::Instant};
use tch::{Tensor,Device,Kind};
use serde_json::json;
use crate::{forward::{Engine,DecodeStates,snapshot},speculative::{Target,Node},dflash::{Drafter,Candidates}};
use crate::dsa::TreeSelection;
use crate::verifier_state::VerifierStates;
use crate::forward::LayerState;
use crate::feature_prefix::AcceptedFeatures;
// Batched-only result; the generic serial verifier keeps Vec<Tensor> aux.
struct BatchedVerified {tokens:Vec<i64>,state:DecodeStates,next:i64,aux:AcceptedFeatures,evaluated:usize}
fn correction_requested()->bool {std::env::var("GLM53_KDA_CORRECTION_REPLAY").as_deref()==Ok("1")}
fn conv_deferred_requested()->bool {std::env::var("GLM53_KDA_CONV_DEFERRED").as_deref()==Ok("1")}
fn half_input_requested()->bool {crate::moe::input_half_reuse_enabled()}
fn dsa_index_requested()->bool {crate::dsa_index::enabled()}

// These switches must be admitted by ABBA and recompute each arm's prefix.
const PREFILL_ABBA_FLAGS:&[&str]=&["GLM53_FP8_BIG","GLM53_PREFILL_SP","GLM53_MLA_PREFILL_F16","GLM53_PREFILL_MOE_SUM1","GLM53_FP8_PREFILL_HALF","GLM53_DSA_PREFILL_SCORE_F16","GLM53_DSA_PREFILL_SCORE_TILED","GLM53_DSA_PREFILL_SCORE_FUSED","GLM53_KV_FP8","GLM53_MLA_CHAIN_SHARED","GLM53_CHAIN_SHARED_BASE","GLM53_MHC_POST_PRE_FUSED","GLM53_HALF_INPUT_CACHE","GLM53_PREFILL_FAT_MOE","GLM53_MHC_PRE_LARGE","GLM53_MLA_PREFILL_TC","GLM53_KDA_PREFILL_FUSED","GLM53_PREFILL_EXPERT_STREAMS","GLM53_RDMA_AR","GLM53_DRAFT_FUSED_NORM","GLM53_MHC_PRE_TC","GLM53_HALF_SKINNY_K1536","GLM53_KDA_CORR_WARP","GLM53_NORM_FUSED","GLM53_SPEC_CONF_TAU_PCT","GLM53_DRAFT_APPEND_GRAPH","GLM53_DRAFT_CONF_TRUNC","GLM53_HALF_SKINNY","GLM53_DRAFT_GRAPH","GLM53_STATE_INPLACE","GLM53_MLA_HALF_BMM","GLM53_ROUTER_FUSED","GLM53_DSA_NODE_FUSED","GLM53_KDA_GATE_FUSED","GLM53_MHC_PRE_FUSED","GLM53_FP8_QKV_FUSED","GLM53_FP8_SKINNY","GLM53_SHARED_GU_FUSED","GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD","GLM53_DRAFT_FINAL_NORM_SELECT","GLM53_DRAFT_CONV_FUSED","GLM53_DRAFT_SELECTOR_FUSED","GLM53_ACCEPTED_FEATURE_VIEW","GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED","GLM53_KDA_CONV_DEFERRED","GLM53_MOE_INPUT_HALF_REUSE","GLM53_KDA_CORRECTION_REPLAY","GLM53_MHC_POST_FOUR_STREAMS","GLM53_MHC_POST_PACKED","GLM53_COOP_PREFETCH","GLM53_MLA_WEIGHT_CACHE","GLM53_TARGET_TOP1_TP","GLM53_COOP_SHARED_INPUT","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT","GLM53_COOP_LOCAL_GU","GLM53_KDA_CHAIN_RECURRENT","GLM53_KDA_CHAIN_NORM","GLM53_COOP_TRANSPOSE","GLM53_DRAFT_NORM_CACHE","GLM53_COOP_CANDIDATE","GLM53_MOE_NO_COPY","GLM53_KDA_CONV_CHAIN","GLM53_FP8_SHARED","GLM53_FP8_MLA","GLM53_FP8_HEAD","GLM53_FP8_SPLITS","GLM53_DRAFT_FP8_MLP","GLM53_DRAFT_FP8_ATTN","GLM53_DRAFT_FP8_CONV","GLM53_DRAFT_FP8_FC","GLM53_DRAFT_FP8_HEAD","GLM53_DRAFT_FP8_SPLITS","GLM53_GROUPED_SWIGLU","GLM53_FP8_LARGE","GLM53_DSA_SCORE_FUSED","GLM53_DSA_PREFILL_LIMIT","GLM53_GROUPED_REDUCE","GLM53_PREFILL_MIN_ROWS","GLM53_DENSE_FP8","GLM53_TF32","GLM53_KDA_SEQUENCE","GLM53_MLA_PREFILL_BATCHED","GLM53_MLA_SCORE_2D","GLM53_PREFILL_COOP"];

struct ModelTarget<'a>(&'a mut Engine);
/// Exclude rank-to-rank warmup skew from paired wall-clock measurements.
fn timed_start(dev:Device)->Instant {
    tch::Cuda::synchronize(0);
    crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));
    tch::Cuda::synchronize(0);Instant::now()
}
impl Target for ModelTarget<'_> {
    type State=DecodeStates;type Aux=Tensor;
    fn fork(&self,s:&DecodeStates)->DecodeStates{snapshot(s)}
    fn step(&mut self,s:&mut DecodeStates,token:i64)->(i64,Tensor) {
        for layer in &s.0 {if let crate::forward::LayerState::MlaLatent(m)=layer{m.ensure_room(1);break;}}
        let (logits,features)=self.0.step_record(&Tensor::from_slice(&[token]).to_device(self.0.w.device),s,true);
        assert_eq!(features.len(),5);(logits.argmax(-1,false).int64_value(&[]),Tensor::cat(&features,1))
    }
}

// This head mode is fixed for a capture and included in every cache match.
// Quality diagnostics keep calling Engine's raw-logits APIs directly.
fn verifier_forward(eng:&mut Engine,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],
    selection:TreeSelection,check_room:bool,head_ids:bool)->(Tensor,VerifierStates,Vec<Tensor>) {
    eng.verifier_forward_selected(ids,base,parents,true,selection,check_room,head_ids)
}
fn full_verifier_forward(eng:&mut Engine,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],
    selection:TreeSelection,check_room:bool,head_ids:bool)->(Tensor,VerifierStates,Vec<Tensor>) {
    let (output,states,features)=if head_ids {eng.tree_forward_predictions_selected(ids,base,parents,true,selection,check_room)}
        else {eng.tree_forward_selected(ids,base,parents,true,selection,check_room)};
    (output,VerifierStates::Full(states),features)
}

fn batched(eng:&mut Engine,base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64])
    ->BatchedVerified {
    if nodes.is_empty(){return BatchedVerified{tokens:Vec::new(),state:snapshot(base),next,aux:AcceptedFeatures::empty(),evaluated:0};}
    let selection=ChainGraph::selection(base,nodes);
    batched_selected(eng,base,next,nodes,budget,stop,selection)
}
// Selection includes a capacity preflight; avoid another device-len read in every layer.
fn batched_selected(eng:&mut Engine,base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64],selection:TreeSelection)
    ->BatchedVerified {
    let ids:Vec<_>=nodes.iter().map(|n|n.token).collect();let parents:Vec<_>=nodes.iter().map(|n|n.parent).collect();
    let input=Tensor::from_slice(&ids).to_device(eng.w.device);
    let (logits,states,features)=verifier_forward(eng,&input,base,&parents,selection,false,crate::head_select::enabled());
    select(base,next,nodes,budget,stop,&logits,&states,&Tensor::cat(&features,1))
}

fn select_with(base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64],logits:&Tensor,states:&VerifierStates,features:&Tensor,
    commit:Option<&dyn Fn(Option<usize>)->DecodeStates>)->BatchedVerified {
    let device_predictions=if logits.kind()==Kind::Int64 {
        assert_eq!(logits.size(),[nodes.len() as i64]);logits.shallow_clone()
    }else{
        assert_eq!(logits.kind(),Kind::Float);assert_eq!(logits.dim(),2);assert_eq!(logits.size()[0],nodes.len() as i64);
        logits.argmax(-1,false)
    };
    let predictions:Vec<i64>=Vec::try_from(device_predictions.to_device(Device::Cpu)).unwrap();
    let prefix_view=crate::feature_prefix::eligible(nodes,features);
    let mut tokens=Vec::new();let mut aux=Vec::new();
    let (parent,prediction)=crate::feature_prefix::walk_selected(nodes,&predictions,next,budget,stop,|i,position|{
        if prefix_view {assert_eq!(i,position,"fixed-chain accepted features must be a prefix");}
        else {aux.push(features.narrow(0,i as i64,1));}
        tokens.push(nodes[i].token);
    });
    let aux=if prefix_view && !tokens.is_empty() {AcceptedFeatures::prefix(features,tokens.len())}
        else {AcceptedFeatures::Rows(aux)};
    BatchedVerified{tokens,state:match commit {Some(commit)=>commit(parent),None=>states.materialize(base,parent)},next:prediction,aux,evaluated:nodes.len()}
}
fn select(base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64],logits:&Tensor,states:&VerifierStates,features:&Tensor)
    ->BatchedVerified {
    select_with(base,next,nodes,budget,stop,logits,states,features,None)
}
/// GLM53_COMMIT_SIDE=1: the accepted node's in-place state commit (KDA replay + conv window, MLA latent commits) runs
/// on a forked side stream, overlapping the drafter (which never reads target state); settle_side_commit() joins it
/// before anything reads the state again (next verifier replay, snapshots/restores, every serve op but the round).
/// Same kernels, same values (L0).
thread_local!{static SIDE_COMMIT:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
    /// Verifier outputs the side-stream commits read (main-stream allocations): kept alive until the join, so the
    /// caching allocator cannot hand their memory to main-stream work that runs concurrently with the commits.
    static SIDE_KEEP:std::cell::RefCell<Vec<VerifierStates>>=const{std::cell::RefCell::new(Vec::new())};}
pub(crate) fn settle_side_commit() {
    if SIDE_COMMIT.with(|c|c.replace(false)) {extern "C"{fn rs_stream_join(n:i32)->i32;}assert_eq!(unsafe{rs_stream_join(1)},0);}
    SIDE_KEEP.with(|k|k.borrow_mut().clear());
}
fn commit_into_base(states:&VerifierStates,base:&DecodeStates,node:usize) {
    let side=std::env::var("GLM53_COMMIT_SIDE").as_deref()==Ok("1") && !crate::tp::graph::capturing();
    settle_side_commit();
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;}
    if side {assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);}
    commit_into_base_now(states,base,node);
    if side {assert_eq!(unsafe{rs_stream_set(-1)},0);SIDE_COMMIT.with(|c|c.set(true));}
}
/// GLM53_MULTI_COMMIT_SIDE=1 (L0, multi-stream rounds): the in-place commits of every sequence of a batched round go on
/// one forked side stream (no join between sequences: distinct stores), so they overlap the drafter append and the next
/// round's proposals; the join happens before the next round first reads target state (selection) or any other op.
fn multi_commit_side()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_MULTI_COMMIT_SIDE").as_deref()==Ok("1"))}
fn commit_into_base_now(states:&VerifierStates,base:&DecodeStates,node:usize) {
    let VerifierStates::Deferred{layers,..}=states else {unreachable!()};
    assert_eq!(layers.len(),base.0.len());
    let mut records=Vec::new();let mut convs=Vec::new();
    // D8 (GLM53_MLA_COMMIT_FUSED=1): chain-shared latent commits of every MLA layer in one launch.
    let fused=std::env::var("GLM53_MLA_COMMIT_FUSED").as_deref()==Ok("1");let mut batch=Vec::new();
    for (layer,b) in layers.iter().zip(&base.0) {
        match (layer,b) {
            (crate::verifier_state::Layer::Kda(r),LayerState::Kda(k))=>{
                assert_eq!(r.base_h.data_ptr(),k.h.data_ptr(),"KDA record must read the graph base");
                records.push(r);convs.push(&k.conv);}
            (crate::verifier_state::Layer::Full(v),LayerState::MlaLatent(m))=>{
                let LayerState::MlaLatent(n)=&v[node] else {panic!("latent node state expected")};
                match if fused {m.commit_entry(n,node as i64+1)} else {None} {Some(e)=>batch.push(e),None=>m.commit_from(n,node as i64+1)}}
            _=>panic!("unsupported in-place commit layer"),
        }
    }
    if !batch.is_empty() {crate::mla_latent::commit_many(&batch);}
    if !records.is_empty(){crate::kda_correction::commit_inplace(&records,node,&convs);}
}
pub(crate) fn chain_shared_base_enabled()->bool {std::env::var("GLM53_CHAIN_SHARED_BASE").as_deref()==Ok("1")}
pub(crate) fn state_inplace_enabled()->bool {std::env::var("GLM53_STATE_INPLACE").as_deref()==Ok("1")}
/// Shallow alias of a verifier base (KDA + latent MLA layers only).
fn alias_states(ds:&DecodeStates)->Option<DecodeStates> {
    ds.0.iter().map(|l|match l {
        LayerState::Kda(k)=>Some(LayerState::Kda(crate::kda::KdaState{h:k.h.shallow_clone(),conv:k.conv.shallow_clone()})),
        LayerState::MlaLatent(m)=>Some(LayerState::MlaLatent(m.alias())),
        _=>None}).collect::<Option<Vec<_>>>().map(DecodeStates)
}
/// Same layer kinds and latent/KV storage format (dtype and row width), so one can restore into the other.
fn same_format(a:&DecodeStates,b:&DecodeStates)->bool {
    a.0.len()==b.0.len() && a.0.iter().zip(&b.0).all(|(x,y)|match (x,y) {
        (LayerState::Kda(_),LayerState::Kda(_))=>true,
        (LayerState::MlaLatent(x),LayerState::MlaLatent(y))=>x.latent.kind()==y.latent.kind()&&x.latent.size()==y.latent.size(),
        _=>false})
}
fn same_storage(a:&DecodeStates,b:&DecodeStates)->bool {
    a.0.len()==b.0.len() && a.0.iter().zip(&b.0).all(|(x,y)|match (x,y) {
        (LayerState::Kda(x),LayerState::Kda(y))=>x.h.data_ptr()==y.h.data_ptr()&&x.conv.data_ptr()==y.conv.data_ptr(),
        (LayerState::MlaLatent(x),LayerState::MlaLatent(y))=>x.latent.data_ptr()==y.latent.data_ptr()&&x.len.data_ptr()==y.len.data_ptr()&&x.index.pools.data_ptr()==y.index.pools.data_ptr(),
        _=>false})
}

/// Bounded diagnostic of the actual selector with real target features and
/// states. Forced Int64 predictions isolate acceptance boundaries; they are
/// explicitly NOT a natural acceptance-rate or target-quality measurement.
pub(crate) fn feature_prefix_selection_check(eng:&mut Engine,drafter:&Drafter,out:&Path) {
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {fn drop(&mut self) {match &self.0 {
        Some(v)=>std::env::set_var("GLM53_ACCEPTED_FEATURE_VIEW",v),
        None=>std::env::remove_var("GLM53_ACCEPTED_FEATURE_VIEW"),
    }}}
    let _restore=Restore(std::env::var_os("GLM53_ACCEPTED_FEATURE_VIEW"));
    let flag=|on:bool|std::env::set_var("GLM53_ACCEPTED_FEATURE_VIEW",if on{"1"}else{"0"});
    let raw=|a:&Tensor,b:&Tensor,label:&str| {
        assert_eq!(a.kind(),Kind::Float);assert_eq!(a.size(),b.size());
        assert!(a.contiguous().view_dtype(Kind::Int).equal(&b.contiguous().view_dtype(Kind::Int)),"{label}");
    };
    let rank=crate::tp::world().rank;std::fs::create_dir_all(out).unwrap();
    let path=out.join(format!("feature-prefix-selection-rank{rank}.json"));
    let mut report=json!({"gate":false,"rank":rank,"cases":[],
        "scope":"Real full-target feature/state producer and real drafter append/proposal; forced prediction boundaries, not natural acceptance or performance"});
    let save=|report:&serde_json::Value| {
        let temp=path.with_extension("json.tmp");std::fs::write(&temp,serde_json::to_string_pretty(report).unwrap()).unwrap();
        std::fs::rename(temp,&path).unwrap();
    };save(&report);
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let prefix:Vec<i64>=refs["hello"]["prompt_ids"].as_array().unwrap().iter().take(12).map(|x|x.as_i64().unwrap()).collect();
    let (_,base,base_features)=eng.prefill_record(&Tensor::from_slice(&prefix).to_device(eng.w.device),None,true);
    let untouched=snapshot(&base);let mut context=drafter.empty_context();drafter.append(&mut context,&Tensor::cat(&base_features,1));
    let untouched_context=context.snapshot();
    let mut topologies:Vec<(String,Vec<Option<usize>>)>=[1usize,2,4,8].into_iter()
        .map(|n|(format!("chain{n}"),(0..n).map(|i|i.checked_sub(1)).collect())).collect();
    topologies.push(("siblings".into(),vec![None,Some(0),Some(0)]));
    topologies.push(("multiple_roots".into(),vec![None,None,Some(1)]));
    for (name,parents) in topologies {
        let nodes:Vec<_>=parents.iter().enumerate().map(|(i,&parent)|Node{parent,token:13041+i as i64}).collect();
        let input=Tensor::from_slice(&nodes.iter().map(|n|n.token).collect::<Vec<_>>()).to_device(eng.w.device);
        let selection=ChainGraph::selection(&base,&nodes);
        let (_,states,features)=verifier_forward(eng,&input,&base,&parents,selection,false,crate::head_select::enabled());
        let features=Tensor::cat(&features,1);let saved_features=features.copy();
        for selected in std::iter::once(None).chain((0..nodes.len()).map(Some)) {
            let mut route=Vec::new();let mut cursor=selected;
            while let Some(i)=cursor {route.push(i);cursor=parents[i];}route.reverse();
            // a: reject immediately after selected prefix; b: budget truncates;
            // c: EOS at its last token. Empty selection also exercises budget0.
            for boundary in ["rejection","budget","eos"] {
                let budget=if boundary=="budget"{route.len()}else{nodes.len()};
                let stop=if boundary=="eos" {route.last().map(|&i|vec![nodes[i].token]).unwrap_or_default()}else{vec![]};
                let next=route.first().map_or_else(||if boundary=="budget"{nodes[0].token}else{-1},|&i|nodes[i].token);
                let mut predictions=vec![-1i64;nodes.len()];
                for pair in route.windows(2){predictions[pair[0]]=nodes[pair[1]].token;}
                // When possible, leave a matching child after the selected
                // node so budget/EOS must actively prevent accepting it.
                if boundary!="rejection" {if let Some(&last)=route.last() {
                    if let Some(child)=nodes.iter().find(|n|n.parent==Some(last)) {predictions[last]=child.token;}
                }}
                let predictions=Tensor::from_slice(&predictions).to_device(eng.w.device);
                flag(false);let old=select(&base,next,&nodes,budget,&stop,&predictions,&states,&features);
                flag(true);let new=select(&base,next,&nodes,budget,&stop,&predictions,&states,&features);
                assert_eq!(old.tokens,new.tokens);assert_eq!(old.tokens,route.iter().map(|&i|nodes[i].token).collect::<Vec<_>>());
                assert_eq!(old.next,new.next);assert_eq!(old.evaluated,new.evaluated);
                assert_eq!(crate::forward::states_max_diff(&old.state,&new.state),0.);
                for (a,b) in old.state.0.iter().zip(&new.state.0) {
                    if let (crate::forward::LayerState::Kda(a),crate::forward::LayerState::Kda(b))=(a,b) {
                        raw(&a.h,&b.h,"selected KDA H bits");raw(&a.conv,&b.conv,"selected KDA conv bits");
                    }
                }
                let expected_prefix=!route.is_empty()&&crate::feature_prefix::fixed_chain(&nodes);
                assert!(matches!(&old.aux,AcceptedFeatures::Rows(_)));
                assert_eq!(matches!(&new.aux,AcceptedFeatures::Prefix(_)),expected_prefix,"{name}/{boundary} missed actual view selection");
                if route.is_empty() {
                    assert!(matches!(&new.aux,AcceptedFeatures::Rows(rows) if rows.is_empty()));
                }else {
                    let a=old.aux.join();let b=new.aux.join();raw(&a,&b,"selected feature bits");
                    if expected_prefix {assert_eq!(b.data_ptr(),features.data_ptr());assert_ne!(a.data_ptr(),features.data_ptr());}
                    // One real consumer run per selected node, with the other
                    // boundary variants checking identical selector/state bits.
                    if boundary=="rejection" {
                        let mut ca=context.snapshot();let mut cb=context.snapshot();
                        drafter.append(&mut ca,&a);drafter.append(&mut cb,&b);assert!(ca.equal(&cb));
                        let pa=drafter.propose(&ca,13041,&eng.w);let pb=drafter.propose(&cb,13041,&eng.w);
                        assert!(pa.ids.equal(&pb.ids));assert_eq!(pa.path,pb.path);
                        for (a,b) in [(&pa.hidden,&pb.hidden),(&pa.edges,&pb.edges),(&pa.unary,&pb.unary)] {
                            assert_eq!(a.kind(),b.kind());
                            let bits=match a.kind(){Kind::Float=>Kind::Int,Kind::Half|Kind::BFloat16=>Kind::Int16,k=>k};
                            assert!(a.contiguous().view_dtype(bits).equal(&b.contiguous().view_dtype(bits)));
                        }
                    }
                }
                report["cases"].as_array_mut().unwrap().push(json!({"topology":name,"selected_node":selected,
                    "boundary":boundary,"budget":budget,"stop_tokens":stop,"accepted_nodes":route,"prefix_variant":expected_prefix,
                    "tokens_next_state_features_exact":true,"real_drafter_checked":!route.is_empty()&&boundary=="rejection"}));
            }
        }
        raw(&features,&saved_features,"selected features mutated producer");
        assert_eq!(crate::forward::states_max_diff(&base,&untouched),0.);assert!(context.equal(&untouched_context));save(&report);
    }
    report["gate"]=json!(true);report["producer_and_base_unchanged"]=json!(true);save(&report);
}

/// A graph for one fixed topology (1..8 nodes). Tokens and prefix change on replay.
/// An owned handle preserves this capture when another topology is captured.
struct ChainGraph {graph:crate::tp::graph::Owned,parents:Vec<Option<usize>>,selection:TreeSelection,head_ids:bool,correction:bool,conv_deferred:bool,half_input:bool,dsa_index:bool,dsa_topk:bool,dsa_position:bool,shared_gu:bool,fp8_small:((bool,bool),u64),routes:Vec<(usize,Tensor)>,base:DecodeStates,input:Tensor,logits:Tensor,states:VerifierStates,features:Tensor}
impl ChainGraph {
    fn assert_storage_gate(&self) {
        if crate::kda::deferred_eligible(&self.parents) {
            let expected=self.base.0.iter().filter(|s|matches!(s,crate::forward::LayerState::Kda(_))).count();
            match &self.states {
                VerifierStates::Deferred{layers,..}=>{
                    assert_eq!(layers.iter().filter(|s|matches!(s,crate::verifier_state::Layer::Kda(_))).count(),expected,
                        "graph qualification silently missed deferred KDA layers");
                    let deferred_conv=layers.iter().filter(|s|matches!(s,crate::verifier_state::Layer::Kda(r) if matches!(&r.conv,crate::kda_correction::ConvRecord::Deferred{..}))).count();
                    assert_eq!(deferred_conv,if conv_deferred_requested(){expected}else{0},"graph qualification silently missed conv storage mode");
                },
                _=>panic!("graph qualification silently used full KDA states"),
            }
        }else{assert!(matches!(&self.states,VerifierStates::Full(_)),"unsupported topology must use full states");}
    }
    fn selection(base:&DecodeStates,nodes:&[Node])->TreeSelection {
        settle_side_commit();   // reads the committed length
        assert!(!nodes.is_empty());let mut depths=Vec::with_capacity(nodes.len());
        for (i,n) in nodes.iter().enumerate(){assert!(n.parent.map_or(true,|p|p<i));depths.push(n.parent.map_or(1i64,|p|depths[p]+1));}
        let depth=*depths.iter().max().unwrap();
        for state in &base.0 {if let crate::forward::LayerState::MlaLatent(s)=state {
            let len=s.len.int64_value(&[0]);assert!(len>=0&&len+depth<=s.capacity,"tree exceeds latent capacity");
            return crate::dsa::tree_selection(len,depth);
        }}
        panic!("tree verifier requires latent MLA state")
    }
    fn new(eng:&mut Engine,base:&DecodeStates,nodes:&[Node])->Self {
        let selection=Self::selection(base,nodes);Self::new_selected(eng,base,nodes,selection)
    }
    /// `selection` comes from a room/regime check for this exact base/topology.
    fn new_selected(eng:&mut Engine,base:&DecodeStates,nodes:&[Node],selection:TreeSelection)->Self {
        Self::new_selected_shared(eng,base,nodes,selection,None)
    }
    /// `shared`: an existing graph base of the same cache. With GLM53_CHAIN_SHARED_BASE=1
    /// every depth's graph reads the same committed-state storage, so switching depth
    /// (confidence truncation) never forces a full-state restore before replay.
    fn new_selected_shared(eng:&mut Engine,base:&DecodeStates,nodes:&[Node],selection:TreeSelection,shared:Option<&DecodeStates>)->Self {
        assert!((1..=16).contains(&nodes.len()));
        let head_ids=crate::head_select::enabled();
        let fp8_small=crate::dense_fp8::graph_signature();
        let base=match shared.filter(|b|chain_shared_base_enabled()&&state_inplace_enabled()&&same_format(b,base)).and_then(alias_states) {
            Some(mut b)=>{if !same_storage(&b,base){crate::forward::restore(&mut b,base);}b}
            None=>snapshot(base),
        };let ids:Vec<_>=nodes.iter().map(|n|n.token).collect();
        let input=Tensor::from_slice(&ids).to_device(eng.w.device);let parents:Vec<_>=nodes.iter().map(|n|n.parent).collect();
        let _=verifier_forward(eng,&input,&base,&parents,selection,true,head_ids);tch::Cuda::synchronize(0);
        if std::env::var("GLM53_SPEC_ROUTE_TRACE").is_ok(){crate::route_trace::begin();}
        crate::tp::graph::begin().unwrap();
        let (logits,states,features)=verifier_forward(eng,&input,&base,&parents,selection,false,head_ids);
        let features=Tensor::cat(&features,1);crate::tp::graph::end().unwrap();
        Self{graph:crate::tp::graph::Owned::take(),parents,selection,head_ids,correction:correction_requested(),conv_deferred:conv_deferred_requested(),half_input:half_input_requested(),dsa_index:dsa_index_requested(),dsa_topk:crate::dsa_topk::enabled(),dsa_position:crate::dsa_position::enabled(),shared_gu:crate::shared_gu::enabled(),fp8_small,routes:crate::route_trace::take(),base,input,logits,states,features}
    }
    fn replay(&mut self,base:&DecodeStates,nodes:&[Node]) {
        let selection=Self::selection(base,nodes);self.replay_checked(base,nodes,selection)
    }
    /// Private callers must perform selection(base,nodes) before cache lookup.
    /// Reuse that result rather than synchronizing the same device len again.
    fn replay_checked(&mut self,base:&DecodeStates,nodes:&[Node],selection:TreeSelection) {
        settle_side_commit();
        assert_eq!(correction_requested(),self.correction,"graph KDA storage mode changed");
        assert_eq!(conv_deferred_requested(),self.conv_deferred,"graph KDA convolution storage mode changed");
        assert_eq!(half_input_requested(),self.half_input,"graph shared Half input mode changed");
        assert_eq!(dsa_index_requested(),self.dsa_index,"graph DSA index mode changed");
        assert_eq!(crate::dsa_topk::enabled(),self.dsa_topk,"graph DSA TopK batch mode changed");
        assert_eq!(self.dsa_position,crate::dsa_position::enabled(),"graph DSA position capture mode changed");
        assert_eq!(crate::shared_gu::enabled(),self.shared_gu,"graph shared GU mode changed");
        assert_eq!(crate::dense_fp8::graph_signature(),self.fp8_small,"graph FP8 small kernel flags changed");
        assert_eq!(crate::head_select::enabled(),self.head_ids,"graph target head mode changed; select/recapture before replay");
        assert_eq!(selection,self.selection,"graph DSA regime mismatch; select/recapture the correct graph before replay");
        assert_eq!(nodes.len(),self.parents.len());
        assert!(nodes.iter().zip(&self.parents).all(|(n,p)|n.parent==*p),"graph topology mismatch");
        if !(state_inplace_enabled()&&same_storage(&self.base,base)){crate::forward::restore(&mut self.base,base);}
        crate::tp::upload(&self.input,&nodes.iter().map(|n|n.token).collect::<Vec<_>>());
        self.graph.replay();
    }
    fn verify(&mut self,base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64])->BatchedVerified {
        self.replay(base,nodes);select(base,next,nodes,budget,stop,&self.logits,&self.states,&self.features)
    }
    fn verify_checked(&mut self,base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64],selection:TreeSelection)->BatchedVerified {
        self.replay_checked(base,nodes,selection);
        if state_inplace_enabled() && alias_states(&self.base).is_some() && matches!(self.states,VerifierStates::Deferred{..})
            && self.parents.iter().enumerate().all(|(i,p)|*p==i.checked_sub(1)) {
            // W07: commit the selected node directly into the graph base and return an alias;
            // the next replay then skips restore. None (nothing selected) keeps base as-is.
            let graph_base=&self.base;let states=&self.states;
            let commit=|parent:Option<usize>|->DecodeStates {
                if let Some(node)=parent {commit_into_base(states,graph_base,node);}
                alias_states(graph_base).unwrap()
            };
            // The caller's base may be an older alias or a separate state; after the commit
            // its content equals graph base only when it already aliased it.
            return select_with(base,next,nodes,budget,stop,&self.logits,&self.states,&self.features,Some(&commit));
        }
        select(base,next,nodes,budget,stop,&self.logits,&self.states,&self.features)
    }
    fn validate(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node]) {
        let mut changed:Vec<_>=nodes.iter().map(|n|Node{parent:n.parent,token:n.token}).collect();let last=changed.len()-1;changed[last].token=0;
        let (_,mut advanced)=eng.prefill_with(&Tensor::from_slice(&[nodes[0].token]).to_device(eng.w.device),Some(snapshot(base)));
        for (base,nodes) in [(base,nodes),(base,changed.as_slice()),(&mut advanced,changed.as_slice())] {
            // A boundary-crossing prefix needs the other graph. This graph's
            // exact replay/eager test must compare its own checked regime.
            let selection=Self::selection(base,nodes);if selection!=self.selection{continue;}
            self.replay_checked(base,nodes,selection);let ids=Tensor::from_slice(&nodes.iter().map(|n|n.token).collect::<Vec<_>>()).to_device(eng.w.device);
            let (lg,states,features)=full_verifier_forward(eng,&ids,base,&nodes.iter().map(|n|n.parent).collect::<Vec<_>>(),self.selection,true,self.head_ids);
            assert!(self.logits.equal(&lg),"chain graph same-mode target output");assert!(self.features.equal(&Tensor::cat(&features,1)),"chain graph features");
            self.states.assert_exact(&states,base,"chain graph versus full-state eager");
        }
    }
}

/// A fixed topology needs at most one graph per DSA regime. A request restored
/// to a shorter prefix can reuse AllVisible after a previous long request.
struct ChainCache {entries:Vec<ChainGraph>,last_used:usize,captures:usize,lookup_hits:usize,
    replays:usize,creation_wall_ms:f64}
impl ChainCache {
    /// Serving store: the first graph's base is the store's own state storage (no copy), so
    /// every later graph (CHAIN_SHARED_BASE) and the store itself share one set of addresses.
    fn new_on(eng:&mut Engine,base:&DecodeStates,nodes:&[Node])->Self {
        let started=Instant::now();let selection=ChainGraph::selection(base,nodes);
        let own=alias_states(base);
        let graph=ChainGraph::new_selected_shared(eng,base,nodes,selection,own.as_ref());
        Self{entries:vec![graph],last_used:0,captures:1,lookup_hits:0,replays:0,
            creation_wall_ms:started.elapsed().as_secs_f64()*1000.}
    }
    fn new(eng:&mut Engine,base:&DecodeStates,nodes:&[Node])->Self {
        let started=Instant::now();let graph=ChainGraph::new(eng,base,nodes);
        Self{entries:vec![graph],last_used:0,captures:1,lookup_hits:0,replays:0,
            creation_wall_ms:started.elapsed().as_secs_f64()*1000.}
    }
    fn validate(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node]) {self.entries[self.last_used].validate(eng,base,nodes);}
    fn verify(&mut self,eng:&mut Engine,base:&DecodeStates,next:i64,nodes:&[Node],budget:usize,stop:&[i64])->BatchedVerified {
        let selection=ChainGraph::selection(base,nodes);
        let head_ids=crate::head_select::enabled();
        let fp8_small=crate::dense_fp8::graph_signature();
        let found=self.entries.iter().position(|g|g.head_ids==head_ids&&g.correction==correction_requested()&&g.conv_deferred==conv_deferred_requested()&&g.half_input==half_input_requested()&&g.dsa_index==dsa_index_requested()&&g.dsa_topk==crate::dsa_topk::enabled()&&g.dsa_position==crate::dsa_position::enabled()&&g.shared_gu==crate::shared_gu::enabled()&&g.fp8_small==fp8_small&&g.selection==selection&&nodes.len()==g.parents.len()&&nodes.iter().zip(&g.parents).all(|(n,p)|n.parent==*p));
        self.last_used=if let Some(i)=found{self.lookup_hits+=1;i}else{
            // One more slot than the draft depths need: copy drafts may add a 16-row chain (GLM53_COPY_MAX=15).
            if self.entries.len()==if crate::dflash::conf_tau().is_some(){9}else{3}{self.entries.remove(0);}
            let started=Instant::now();
            let shared=self.entries.first().map(|g|alias_states(&g.base)).flatten();
            self.entries.push(ChainGraph::new_selected_shared(eng,base,nodes,selection,shared.as_ref()));
            self.creation_wall_ms+=started.elapsed().as_secs_f64()*1000.;self.captures+=1;
            self.entries.len()-1
        };
        self.replays+=1;
        self.entries[self.last_used].verify_checked(base,next,nodes,budget,stop,selection)
    }
    fn routes(&self)->&[(usize,Tensor)] {&self.entries[self.last_used].routes}
    fn stats(&self)->(usize,usize,usize,f64) {
        (self.captures,self.lookup_hits,self.replays,self.creation_wall_ms)
    }
}


struct GraphPool {entries:Vec<ChainGraph>,limit:usize,pub captures:usize,pub hits:usize,
    seen:Vec<(Vec<Option<usize>>,TreeSelection,bool,bool,usize)>,capture_after:usize,min_remaining:usize,pub eager_misses:usize}
impl GraphPool {
    fn new()->Self {
        let limit=std::env::var("GLM53_SPEC_GRAPH_SLOTS").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(4);
        let capture_after=std::env::var("GLM53_SPEC_TREE_CAPTURE_AFTER").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(3);
        let min_remaining=std::env::var("GLM53_SPEC_TREE_CAPTURE_MIN_REMAINING").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(128);
        assert!((1..=8).contains(&limit));assert!((1..=64).contains(&capture_after));
        Self{entries:Vec::new(),limit,captures:0,hits:0,seen:Vec::new(),capture_after,min_remaining,eager_misses:0}
    }
    fn try_get(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node],remaining:usize)->Option<&mut ChainGraph> {
        let selection=ChainGraph::selection(base,nodes);
        self.try_get_selected(eng,base,nodes,remaining,selection)
    }
    fn try_get_selected(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node],remaining:usize,selection:TreeSelection)->Option<&mut ChainGraph> {
        let parents:Vec<_>=nodes.iter().map(|n|n.parent).collect();
        let head_ids=crate::head_select::enabled();
        let fp8_small=crate::dense_fp8::graph_signature();
        if self.entries.iter().any(|g|g.parents==parents&&g.selection==selection&&g.head_ids==head_ids&&g.correction==correction_requested()&&g.conv_deferred==conv_deferred_requested()&&g.half_input==half_input_requested()&&g.dsa_index==dsa_index_requested()&&g.dsa_topk==crate::dsa_topk::enabled()&&g.dsa_position==crate::dsa_position::enabled()&&g.shared_gu==crate::shared_gu::enabled()&&g.fp8_small==fp8_small){return Some(self.get_selected(eng,base,nodes,selection));}
        if self.should_capture(parents,selection,remaining){Some(self.get_selected(eng,base,nodes,selection))}
        else{self.eager_misses+=1;None}
    }
    fn should_capture(&mut self,parents:Vec<Option<usize>>,selection:TreeSelection,remaining:usize)->bool {
        let topk=crate::dsa_topk::enabled();let position=crate::dsa_position::enabled();
        let count=if let Some(i)=self.seen.iter().position(|(p,s,b,c,_)|*p==parents&&*s==selection&&*b==topk&&*c==position) {
            let (_,_,_,_,n)=self.seen.remove(i);n+1
        }else{1};
        if self.seen.len()==64{self.seen.remove(0);}
        let capture=count>=self.capture_after && remaining>=self.min_remaining;
        // An evicted topology must demonstrate reuse again; don't thrash four
        // graph slots simply because many old shapes have appeared before.
        self.seen.push((parents,selection,topk,position,if capture{0}else{count}));
        capture
    }
    fn get(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node])->&mut ChainGraph {
        let selection=ChainGraph::selection(base,nodes);
        self.get_selected(eng,base,nodes,selection)
    }
    fn get_selected(&mut self,eng:&mut Engine,base:&DecodeStates,nodes:&[Node],selection:TreeSelection)->&mut ChainGraph {
        let head_ids=crate::head_select::enabled();
        let fp8_small=crate::dense_fp8::graph_signature();
        if let Some(i)=self.entries.iter().position(|g|g.head_ids==head_ids&&g.correction==correction_requested()&&g.conv_deferred==conv_deferred_requested()&&g.half_input==half_input_requested()&&g.dsa_index==dsa_index_requested()&&g.dsa_topk==crate::dsa_topk::enabled()&&g.dsa_position==crate::dsa_position::enabled()&&g.shared_gu==crate::shared_gu::enabled()&&g.fp8_small==fp8_small&&g.selection==selection&&nodes.len()==g.parents.len() && nodes.iter().zip(&g.parents).all(|(n,p)|n.parent==*p)) {
            let entry=self.entries.remove(i);self.entries.push(entry);self.hits+=1;
        } else {
            if self.entries.len()==self.limit {self.entries.remove(0);}
            self.entries.push(ChainGraph::new_selected(eng,base,nodes,selection));self.captures+=1;
        }
        self.entries.last_mut().unwrap()
    }
}
fn calibrate(eng:&mut Engine,drafter:&Drafter,context:&crate::dflash::Context,base:&DecodeStates,
             anchor:i64,nodes:&[Node],pool:&mut GraphPool)->crate::spec_policy::Policy {
    assert_eq!(nodes.len(),8);assert!(pool.limit>=3,"adaptive mode requires at least three graph slots");
    let dev=eng.w.device;let mut costs=[0.;3];
    let start=timed_start(dev);for _ in 0..3{let _=drafter.propose(context,anchor,&eng.w);}tch::Cuda::synchronize(0);
    let draft_ms=start.elapsed().as_secs_f64()*1000./3.;
    for (i,depth) in [3usize,5,7].into_iter().enumerate() {
        let nodes=&nodes[..depth+1];let graph=pool.get(eng,base,nodes);
        let _=graph.verify(base,anchor,nodes,depth+1,&[]);
        let mut samples=Vec::new();
        for _ in 0..3 {
            let mut checkout=context.snapshot();
            let start=timed_start(dev);let verified=graph.verify(base,anchor,nodes,depth+1,&[]);
            drafter.append(&mut checkout,&verified.aux.join());tch::Cuda::synchronize(0);
            samples.push(start.elapsed().as_secs_f64()*1000.);
        }
        let local=draft_ms+samples.iter().sum::<f64>()/samples.len() as f64;
        // Exchange both timings via disjoint lanes. Both ranks take the same
        // max and therefore issue identical topology/collective sequences.
        let mut times=[0f32;2];times[crate::tp::world().rank]=local as f32;
        let times=Tensor::from_slice(&times).to_device(dev);crate::tp::allreduce(&times);
        costs[i]=times.max().double_value(&[]);
    }
    crate::spec_policy::Policy::new(costs)
}

fn draft_depth()->usize {
    let n=std::env::var("GLM53_SPEC_MAX_DRAFT").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(7);
    assert!((1..=7).contains(&n));n
}

/// Copy (prompt-lookup) drafts, GLM53_COPY_DRAFTS=1: when the context's last
/// GLM53_COPY_MATCH tokens (default 8, the pending anchor included) occurred earlier in prompt + reply, the round
/// verifies what followed them instead of DFlash2's chain. GLM53_COPY_MAX (default 7) up to 15: a copied chain is
/// either at most 7 drafts (the existing <= 8-row graphs) or exactly GLM53_COPY_MAX when that many follow, so at most
/// one more chain graph per store. Drafts only propose (L2); both ranks hold the same prompt and reply, so no exchange.
fn copy_settings()->Option<(usize,usize)> {
    static S:std::sync::OnceLock<Option<(usize,usize)>>=std::sync::OnceLock::new();
    *S.get_or_init(||{
        if std::env::var("GLM53_COPY_DRAFTS").as_deref()!=Ok("1") {return None;}
        let num=|k:&str,d:usize|std::env::var(k).ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(d);
        let (m,k)=(num("GLM53_COPY_MATCH",8),num("GLM53_COPY_MAX",7));
        assert!((2..=64).contains(&m)&&(1..=15).contains(&k),"GLM53_COPY_MATCH 2..=64, GLM53_COPY_MAX 1..=15");Some((m,k))
    })
}
/// Up to `room` tokens after the latest earlier occurrence of the context's last `m` tokens that has `room` tokens
/// after it, else after the earliest occurrence (the most tokens); empty when the suffix never occurred before.
/// The context is prompt ++ generated ++ [anchor].
fn copy_proposal(prompt:&[i64],generated:&[i64],anchor:i64,m:usize,room:usize)->Vec<i64> {
    let (p,g)=(prompt.len(),generated.len());let l=p+g+1;
    if room==0 || l<=m {return Vec::new();}
    let at=|i:usize|->i64 {if i<p {prompt[i]} else if i<p+g {generated[i-p]} else {anchor}};
    let q=l-m;
    let mut earliest=None;
    // Match ends e = s+m-1 for starts s in [0, q-1], latest first: test the last token, then the rest.
    for s in (0..q).rev() {
        if at(s+m-1)!=anchor || !(0..m-1).all(|k|at(s+k)==at(q+k)) {continue;}
        if s+m+room<=l {return (s+m..s+m+room).map(at).collect();}
        earliest=Some(s);
    }
    earliest.map(|s|(s+m..l.min(s+m+room)).map(at).collect()).unwrap_or_default()
}

/// Exercise multiple live topology captures, including non-chain siblings and a
/// short tail. Revisit after other captures and force LRU eviction/recreation.
pub fn graph_check(eng:&mut Engine,out:&Path) {
    let ids=Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(eng.w.device);
    let (_,base)=eng.prefill(&ids);
    let layouts=vec![vec![None,Some(0),Some(1),Some(2)],
        vec![None,Some(0),Some(0),Some(1),Some(1),Some(2),Some(2),Some(6)],
        vec![None,Some(0)],vec![None,Some(0),Some(0),Some(0),Some(1),Some(2)]];
    let mut pool=GraphPool::new();pool.limit=3;
    for i in [0usize,1,2,0,3,1,2,3] {
        let nodes:Vec<_>=layouts[i].iter().enumerate().map(|(j,&parent)|Node{parent,token:13041+j as i64}).collect();
        let graph=pool.get(eng,&base,&nodes);graph.assert_storage_gate();graph.validate(eng,&base,&nodes);
    }
    assert!(pool.hits>0 && pool.captures>pool.limit);
    let mut admission=GraphPool::new();admission.capture_after=2;admission.min_remaining=8;
    let nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:0}];
    assert!(admission.try_get(eng,&base,&nodes,192).is_none());
    admission.try_get(eng,&base,&nodes,192).unwrap().validate(eng,&base,&nodes);
    assert!(admission.try_get(eng,&base,&nodes,1).is_some(),"existing graph remains useful for a short request");
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join(format!("graphs-rank{}.json",crate::tp::world().rank)),json!({"exact":true,
        "layouts":layouts,"visits":8,"captures":pool.captures,"hits":pool.hits,"live_limit":pool.limit,"admission_reuse_exact":true,
        "checks":"same-mode target output (logits or Int64 predictions)/features/all branch states after changed token and advanced prefix; independent graph lifetime and LRU"}).to_string()).unwrap();
}

/// Full-engine graph identity gate for the optional verifier regime. Uses real
/// prefills at both sides of the boundary and restores a real short request;
/// unlike a quality evaluation, every graph is compared to its same-regime eager path.
pub fn all_visible_graph_check(eng:&mut Engine,out:&Path) {
    assert_eq!(std::env::var("GLM53_DSA_ALL_VISIBLE").as_deref(),Ok("1"));
    assert!(crate::mla_latent::capacity()>=2052);
    std::fs::create_dir_all(out).unwrap();let dev=eng.w.device;let mut records=Vec::new();
    let layouts=vec![(0..8).map(|i:usize|i.checked_sub(1)).collect::<Vec<_>>(),
        vec![None,Some(0),Some(0),Some(1),Some(2),Some(1),Some(2),Some(6)]];
    for (layout,parents) in layouts.iter().enumerate() {
        let mut depths=Vec::new();for &p in parents{depths.push(p.map_or(1i64,|p|depths[p]+1));}
        let max_depth=*depths.iter().max().unwrap();let boundary=2051-max_depth;
        let mut prompt=vec![13041i64;boundary as usize];prompt[..4].copy_from_slice(&[154822,154824,154826,13041]);
        let (_,base)=eng.prefill(&Tensor::from_slice(&prompt).to_device(dev));
        let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13041i64]).to_device(dev),Some(snapshot(&base)));
        let (_,short)=eng.prefill(&Tensor::from_slice(&prompt[..4]).to_device(dev));
        let nodes:Vec<_>=parents.iter().enumerate().map(|(i,&parent)|Node{parent,token:13041+i as i64}).collect();
        let input=Tensor::from_slice(&nodes.iter().map(|n|n.token).collect::<Vec<_>>()).to_device(dev);
        let check=|eng:&mut Engine,g:&ChainGraph,base:&DecodeStates|{
            let (logits,states,features)=full_verifier_forward(eng,&input,base,parents,g.selection,true,g.head_ids);
            assert!(logits.equal(&g.logits)&&Tensor::cat(&features,1).equal(&g.features),"DSA regime graph/eager outputs mismatch");
            states.assert_exact(&g.states,base,"DSA regime graph branch state mismatch");
        };
        if layout==0 {
            let mut cache=ChainCache::new(eng,&base,&nodes);
            for (name,state,wanted) in [("last_all",&base,TreeSelection::AllVisible),("first_ranked",&advanced,TreeSelection::Ranked),
                ("short_restore",&short,TreeSelection::AllVisible),("last_all_reuse",&base,TreeSelection::AllVisible),("ranked_reuse",&advanced,TreeSelection::Ranked)] {
                assert_eq!(ChainGraph::selection(state,&nodes),wanted);
                let _=cache.verify(eng,state,nodes[0].token,&nodes,nodes.len(),&[]);
                let graph=&cache.entries[cache.last_used];graph.assert_storage_gate();assert_eq!(graph.selection,wanted);check(eng,graph,state);
                records.push(json!({"cache":"fixed_chain","name":name,"selection":format!("{wanted:?}"),"max_depth":max_depth,"live_graphs":cache.entries.len(),"exact_same_regime":true}));
            }
            assert_eq!(cache.entries.len(),2);
        }else {
            let mut pool=GraphPool::new();pool.limit=2;
            for (name,state,wanted) in [("last_all",&base,TreeSelection::AllVisible),("first_ranked",&advanced,TreeSelection::Ranked),
                ("short_restore",&short,TreeSelection::AllVisible),("last_all_reuse",&base,TreeSelection::AllVisible),("ranked_reuse",&advanced,TreeSelection::Ranked)] {
                let graph=pool.get(eng,state,&nodes);graph.assert_storage_gate();assert_eq!(graph.selection,wanted);
                let _=graph.verify_checked(state,nodes[0].token,&nodes,nodes.len(),&[],wanted);check(eng,graph,state);
                records.push(json!({"cache":"tree_pool","name":name,"selection":format!("{wanted:?}"),"max_depth":max_depth,"live_graphs":pool.entries.len(),"exact_same_regime":true}));
            }
            assert_eq!(pool.captures,2);assert_eq!(pool.hits,3);
        }
    }
    std::fs::write(out.join(format!("dsa-regime-graphs-rank{}.json",crate::tp::world().rank)),serde_json::to_string_pretty(&json!({"gate":true,"cases":records,
        "scope":"same-regime graph/eager correctness only; Ranked versus AllVisible quality remains separate"})).unwrap()).unwrap();
}

#[cfg(test)] mod graph_admission_tests {
    use super::{GraphPool,TreeSelection};
    #[test] fn capture_requires_reuse_and_remaining_work() {
        let mut p=GraphPool::new();p.capture_after=3;p.min_remaining=128;
        let shape=vec![None,Some(0)];
        for _ in 0..3{assert!(!p.should_capture(shape.clone(),TreeSelection::Ranked,64));}
        assert!(p.should_capture(shape.clone(),TreeSelection::Ranked,128));
        // Eviction followed by a revisit must earn admission again.
        assert!(!p.should_capture(shape.clone(),TreeSelection::Ranked,192));assert!(!p.should_capture(shape.clone(),TreeSelection::Ranked,192));
        assert!(p.should_capture(shape,TreeSelection::Ranked,192));
    }
    #[test] fn topology_history_is_bounded() {
        let mut p=GraphPool::new();p.min_remaining=128;
        for value in 0..128 {let mut code=value;let mut shape=vec![None];
            for node in 1..8{shape.push(Some(code%node));code/=node;}
            assert!(!p.should_capture(shape,TreeSelection::Ranked,0));assert!(p.seen.len()<=64);
        }
        assert_eq!(p.seen.len(),64);
    }
    #[test] fn selection_regimes_have_independent_admission() {
        let mut p=GraphPool::new();p.capture_after=2;p.min_remaining=1;
        let shape=vec![None,Some(0)];
        assert!(!p.should_capture(shape.clone(),TreeSelection::AllVisible,8));
        assert!(p.should_capture(shape.clone(),TreeSelection::AllVisible,8));
        assert!(!p.should_capture(shape.clone(),TreeSelection::Ranked,8));
        assert!(p.should_capture(shape,TreeSelection::Ranked,8));
    }
}

fn with_anchor(anchor:i64,nodes:Vec<Node>)->Vec<Node> {
    let mut all=vec![Node{parent:None,token:anchor}];
    all.extend(nodes.into_iter().map(|n|Node{parent:Some(n.parent.map_or(0,|i|i+1)),token:n.token}));all
}

fn tree(c:&Candidates,budget:usize,depth:usize)->Vec<Node> {
    let ids:Vec<Vec<i64>>=Vec::try_from(c.ids.to_device(Device::Cpu)).unwrap();
    let probabilities=c.edges.log_softmax(-1,Kind::Float).to_device(Device::Cpu);
    // Cumulative conditional log probabilities, never raw logits across depth.
    let mut frontier:Vec<(f64,Option<usize>,usize,usize)>=Vec::new();
    for j in 0..16 {frontier.push((probabilities.double_value(&[0,0,j as i64]),None,0,j));}
    let mut nodes=Vec::new();
    while nodes.len()<budget && !frontier.is_empty() {
        let best=frontier.iter().enumerate().max_by(|a,b|a.1.0.total_cmp(&b.1.0)).unwrap().0;
        let (score,parent,d,j)=frontier.remove(best);let idx=nodes.len();
        nodes.push(Node{parent,token:ids[d][j]});
        if d+1<depth {for k in 0..16 {frontier.push((score+probabilities.double_value(&[(d+1) as i64,j as i64,k as i64]),Some(idx),d+1,k));}}
    }
    nodes
}

/// User-facing token-ID CLI, matching the existing engine CLI conventions.
/// Explicit mode selection: serial speculation is not silently enabled for speed.
pub fn decode(model:&Path,draft:&Path,ids:&[i64],mode:&str,n:usize,out:&Path) {
    assert!(["chain","tree","batch-chain","batch-tree","batch-graph-chain","batch-graph-tree","batch-adaptive-chain"].contains(&mode),"unsupported speculation mode");assert!(!ids.is_empty());
    assert!(ids.len()+n<=crate::mla_latent::capacity() as usize,"request exceeds context capacity");
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"] {std::env::set_var(flag,"1");}
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let draft=Drafter::load_target(draft,&w);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let (logits,mut state,features)=eng.prefill_record_last(&Tensor::from_slice(ids).to_device(dev),None,true);
    let mut context=draft.empty_context();draft.append(&mut context,&Tensor::cat(&features,1));
    let mut anchor=logits.get(logits.size()[0]-1).argmax(-1,false).int64_value(&[]);
    let result=generate(&mut eng,&draft,state,context,anchor,mode,n);
    if tp.rank==0 {std::fs::write(out,serde_json::to_string_pretty(&result).unwrap()).unwrap();}
}

// Private to the resident CLI: graphs never outlive its engine/drafter.
// Bind identity and execution flags so reuse cannot silently cross models.
struct DecodeRuntime {chain:Option<ChainCache>,trees:GraphPool,costs:Option<[f64;3]>,
    engine:usize,draft:usize,mode:String,depth:usize,signature:Vec<Option<String>>}
impl DecodeRuntime {
    fn new(eng:&Engine,draft:&Drafter,mode:&str)->Self {
        Self{chain:None,trees:GraphPool::new(),costs:None,engine:eng as *const Engine as usize,
            draft:draft as *const Drafter as usize,mode:mode.into(),depth:draft_depth(),signature:crate::spec_session::signature()}
    }
    fn check(&self,eng:&Engine,draft:&Drafter,mode:&str) {
        assert_eq!(self.engine,eng as *const Engine as usize);assert_eq!(self.draft,draft as *const Drafter as usize);
        assert_eq!(self.mode,mode);assert_eq!(self.depth,draft_depth());assert_eq!(self.signature,crate::spec_session::signature());
    }
}
pub(crate) fn generate(eng:&mut Engine,draft:&Drafter,state:DecodeStates,context:crate::dflash::Context,
                       anchor:i64,mode:&str,n:usize)->serde_json::Value {
    let mut runtime=DecodeRuntime::new(eng,draft,mode);
    generate_reuse(eng,draft,state,context,anchor,mode,n,&mut runtime)
}
fn generate_reuse(eng:&mut Engine,draft:&Drafter,state:DecodeStates,context:crate::dflash::Context,
                  anchor:i64,mode:&str,n:usize,runtime:&mut DecodeRuntime)->serde_json::Value {
    generate_reuse_hooked(eng,draft,state,context,anchor,mode,n,runtime,None)
}
/// `hook` is called on every rank after each verification round with the newly emitted
/// tokens; it must return the same stop decision on all ranks (serve broadcasts rank0's).
#[allow(clippy::too_many_arguments)]
fn generate_reuse_hooked(eng:&mut Engine,draft:&Drafter,mut state:DecodeStates,mut context:crate::dflash::Context,
                  mut anchor:i64,mode:&str,n:usize,runtime:&mut DecodeRuntime,mut hook:Option<&mut dyn FnMut(&[i64])->bool>)->serde_json::Value {
    let mut emitted=0usize;let mut stopped=false;
    let mut notify=|generated:&Vec<i64>,emitted:&mut usize|->bool {
        match hook.as_mut() {Some(h)=>{let stop=h(&generated[*emitted..]);*emitted=generated.len();stop},None=>false}};
    runtime.check(eng,draft,mode);
    let old_captures=runtime.trees.captures;let old_hits=runtime.trees.hits;let old_misses=runtime.trees.eager_misses;
    let old_chain=runtime.chain.is_some();let had_costs=runtime.costs.is_some();
    let old_chain_stats=runtime.chain.as_ref().map_or((0,0,0,0.),ChainCache::stats);
    assert!(["chain","tree","batch-chain","batch-tree","batch-graph-chain","batch-graph-tree","batch-adaptive-chain"].contains(&mode));
    assert!(context.len+n as i64<=crate::mla_latent::capacity(),"request exceeds context capacity");
    let eos=[154820,154827,154829];let mut generated=Vec::new();let mut cycles=Vec::new();
    let prepare_graph_start=Instant::now();
    let mut adaptive=if mode=="batch-adaptive-chain" && n>1 && !eos.contains(&anchor) && crate::mla_latent::capacity()-context.len>=8 {
        if runtime.costs.is_none() {
            let c=draft.propose(&context,anchor,&eng.w);let nodes=with_anchor(anchor,crate::speculative::chain(&c.path));
            runtime.costs=Some(calibrate(eng,draft,&context,&state,anchor,&nodes,&mut runtime.trees).costs);
        }
        Some(crate::spec_policy::Policy::new(runtime.costs.unwrap()))
    }else{None};
    tch::Cuda::synchronize(0);let graph_prepare_ms=prepare_graph_start.elapsed().as_secs_f64()*1000.;let started=Instant::now();
    while generated.len()<n {
        // This API returns tokens, not a continuation state. The final known
        // token needs neither target evaluation nor a drafter context update.
        if generated.len()+1==n || eos.contains(&anchor) {
            generated.push(anchor);cycles.push(json!({"accepted":0,"evaluated":0,
                "anchor_batched":false,"terminal_emit_only":true,"accepted_feature_prefix_view":false}));
            let _=notify(&generated,&mut emitted);break;
        }
        let remaining=n-generated.len();let mut depth=(remaining-1).min(adaptive.as_ref().map_or(draft_depth(),|p|p.depth()));
        let mut selector_fused=None;let mut conv_calls=None;let mut conv_fused_calls=None;let mut final_norm_selected=None;
        let nodes=if depth==0||eos.contains(&anchor){Vec::new()}else{
            let c=draft.propose(&context,anchor,&eng.w);
            selector_fused=Some(c.selector_fused);conv_calls=Some(c.conv_calls);conv_fused_calls=Some(c.conv_fused_calls);final_norm_selected=Some(c.final_norm_selected);
            // Same L2 depth policy as the resident harness: stop the chain at the first path
            // position whose drafter confidence is below tau (keep at least one draft).
            if let (Some(tau),Some(conf),false)=(crate::dflash::conf_tau(),c.conf.as_ref(),mode.ends_with("tree")) {
                depth=crate::spec_policy::chain_depth(conf,depth,tau);
            }
            if mode.ends_with("tree"){tree(&c,depth,depth)}else{crate::speculative::chain(&c.path[..depth])}
        };
        if mode.starts_with("batch-") {
            let all=with_anchor(anchor,nodes);
            let verified=if mode=="batch-graph-chain"&&(all.len()==draft_depth()+1||(crate::dflash::conf_tau().is_some()&&all.len()>1)) {
                if runtime.chain.is_none(){runtime.chain=Some(ChainCache::new(eng,&state,&all));}
                runtime.chain.as_mut().unwrap().verify(eng,&state,anchor,&all,depth+1,&eos)
            } else if mode=="batch-graph-tree" {
                let selection=ChainGraph::selection(&state,&all);
                match runtime.trees.try_get_selected(eng,&state,&all,remaining,selection) {
                    Some(graph)=>graph.verify_checked(&state,anchor,&all,depth+1,&eos,selection),
                    None=>batched_selected(eng,&state,anchor,&all,depth+1,&eos,selection),
                }
            } else if adaptive.is_some() && [4,6,8].contains(&all.len()) {
                {let graph=runtime.trees.get(eng,&state,&all);graph.verify_checked(&state,anchor,&all,depth+1,&eos,graph.selection)}
            } else {batched(eng,&state,anchor,&all,depth+1,&eos)};
            let prefix_used=verified.aux.is_prefix();
            let accepted=verified.tokens.len()-1;if let Some(policy)=&mut adaptive{policy.observe(accepted,depth);}
            generated.extend(verified.tokens);state=verified.state;anchor=verified.next;
            let mut finished=generated.len()==n || generated.last().map_or(false,|t|eos.contains(t));
            if notify(&generated,&mut emitted) {finished=true;stopped=true;}
            if !finished{draft.append(&mut context,&verified.aux.join());}
            cycles.push(json!({"accepted":accepted,"evaluated":verified.evaluated-1,"anchor_batched":true,"accepted_feature_prefix_view":prefix_used,
                "draft_selector_fused":selector_fused,"draft_final_norm_selected":final_norm_selected,"draft_conv_calls":conv_calls,"draft_conv_fused_calls":conv_fused_calls}));
            if finished{break;}continue;
        }
        let (next,feature)=ModelTarget(&mut *eng).step(&mut state,anchor);
        generated.push(anchor);if eos.contains(&anchor){let _=notify(&generated,&mut emitted);break;}
        let verified=crate::speculative::verify_until(&mut ModelTarget(&mut *eng),&state,next,&nodes,depth,&eos);
        let accepted=verified.tokens.len();generated.extend(verified.tokens);state=verified.state;anchor=verified.next;
        let mut finished=generated.len()==n || generated.last().map_or(false,|t|eos.contains(t));
        if notify(&generated,&mut emitted) {finished=true;stopped=true;}
        if !finished{let mut features=vec![feature];features.extend(verified.aux);draft.append(&mut context,&Tensor::cat(&features,0));}
        cycles.push(json!({"accepted":accepted,"evaluated":verified.evaluated,"accepted_feature_prefix_view":false,"draft_selector_fused":selector_fused,"draft_final_norm_selected":final_norm_selected,"draft_conv_calls":conv_calls,"draft_conv_fused_calls":conv_fused_calls}));
        if finished{break;}
    }
    tch::Cuda::synchronize(0);let elapsed_ms=started.elapsed().as_secs_f64()*1000.;
    let chain_stats=runtime.chain.as_ref().map_or((0,0,0,0.),ChainCache::stats);
    json!({"mode":mode,"token_ids":generated,"stopped_by_hook":stopped,"decode_ms":elapsed_ms,"graph_prepare_ms":graph_prepare_ms,
        "cycles":cycles,"batched":mode.starts_with("batch-"),"tree_graph_captures":runtime.trees.captures-old_captures,"tree_graph_hits":runtime.trees.hits-old_hits,
        "chain_graph_reused":old_chain,"chain_graph_captured":!old_chain&&runtime.chain.is_some(),"adaptive_costs_reused":had_costs&&adaptive.is_some(),
        "chain_graph_captures":chain_stats.0-old_chain_stats.0,
        "chain_graph_lookup_hits":chain_stats.1-old_chain_stats.1,
        "chain_graph_replays":chain_stats.2-old_chain_stats.2,
        "chain_graph_creation_wall_ms":chain_stats.3-old_chain_stats.3,
        "chain_graph_creation_scope":"Host wall for eager initialization, synchronization and graph creation; included in decode_ms. No extra synchronization for telemetry.",
        "tree_eager_misses":runtime.trees.eager_misses-old_misses,
        "adaptive_calibrated_ms":adaptive.as_ref().map(|p|p.costs),
        "numerics":"Batched GEMM changes floating-point rounding; serial modes preserve the sequential target path."})
}

pub fn decode_many(model:&Path,draft:&Path,suite:&Path,mode:&str,out:&Path) {
    let requests:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite).unwrap()).unwrap();
    let cases=requests["cases"].as_array().expect("cases array");assert!(!cases.is_empty());
    assert!(["chain","tree","batch-chain","batch-tree","batch-graph-chain","batch-graph-tree","batch-adaptive-chain"].contains(&mode));
    let config:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(model.join("config.json")).unwrap()).unwrap();
    let vocab=config["text_config"]["vocab_size"].as_i64().or_else(||config["vocab_size"].as_i64()).expect("vocabulary size");
    for c in cases {let ids=c["prompt_ids"].as_array().expect("prompt_ids array");
        let n=c["max_new"].as_u64().expect("max_new unsigned integer") as usize;
        assert!(ids.iter().all(|v|v.as_i64().map_or(false,|id|id>=0&&id<vocab)),"invalid prompt token");
        assert!(!ids.is_empty() && ids.len()+n<=crate::mla_latent::capacity() as usize);}
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);let draft=Drafter::load_target(draft,&w);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let chunk_size=std::env::var("GLM53_PREFILL_CHUNK").ok().map(|s|s.parse::<usize>().unwrap()).unwrap_or(64);assert!(chunk_size>0 && chunk_size<=crate::mla_latent::capacity() as usize);
    let mut session=crate::spec_session::Session::new(&mut eng,&draft,2,chunk_size);let mut results=Vec::new();
    let mut runtime=DecodeRuntime::new(session.engine,session.drafter,mode);
    for c in cases {
        let ids:Vec<i64>=c["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
        let request_started=Instant::now();let started=timed_start(dev);let (logits,state,context,hit)=session.prepare(&ids);tch::Cuda::synchronize(0);
        let prepare_ms=started.elapsed().as_secs_f64()*1000.;
        let anchor=logits.argmax(-1,false).int64_value(&[]);
        let mut result=generate_reuse(session.engine,session.drafter,state,context,anchor,mode,c["max_new"].as_u64().unwrap() as usize,&mut runtime);
        result["total_ms"]=json!(request_started.elapsed().as_secs_f64()*1000.);
        result["name"]=c["name"].clone();result["prefix_hit_tokens"]=json!(hit);result["prepare_ms"]=json!(prepare_ms);results.push(result);
    }
    std::fs::write(out,serde_json::to_string_pretty(&json!({"rank":tp.rank,"cases":results,"prefix_hits":session.hits,
        "prefix_misses":session.misses,"prefix_evictions":session.evictions,"prefix_slots":2,"chunk_size":chunk_size})).unwrap()).unwrap();
}

pub fn run(model:&Path,draft:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"] {std::env::set_var(flag,"1");}
    // Keep the same prefill arithmetic in target-only and speculative arms.
    if std::env::var("GLM53_SPEC_SUITE").is_ok(){std::env::set_var("GLM53_PREFILL_BATCH","1");}
    else {std::env::remove_var("GLM53_PREFILL_BATCH");}
    std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let drafter=Drafter::load_target(draft,&w);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,
        cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;let misses=fast.misses;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    check(&mut eng,&drafter,out);
    drop(drafter);
    if std::env::var("GLM53_POST_GEMV_BENCH").as_deref()==Ok("1") {
        std::env::set_var("GLM53_BENCH_GRAPH_ONLY","1");
        crate::kernel_bench::full_check(&mut eng,&out.join("gemv-perf"),"GLM53_DENSE_GEMV");
    }
}

pub fn check(eng:&mut Engine,drafter:&Drafter,out:&Path) {
    let tp=crate::tp::world();let dev=eng.w.device;let misses=eng.fast.as_ref().unwrap().misses;
    std::fs::create_dir_all(out).unwrap();
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let n=std::env::var("GLM53_SPEC_TOKENS").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(24);assert!(n>=8);
    let repeats=std::env::var("GLM53_SPEC_ROUNDS").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(1);assert!(repeats>0);
    let abba_flag=std::env::var("GLM53_SPEC_ABBA_FLAG").ok().or_else(||
        if std::env::var("GLM53_SPEC_LAYOUT_ABBA").as_deref()==Ok("1"){Some("GLM53_MLA_SCORE_2D".into())}else{None});
    if let Some(flag)=&abba_flag {assert!(PREFILL_ABBA_FLAGS.contains(&flag.as_str()) || ["GLM53_MOE_SHARED_STREAM","GLM53_MLA_COMMIT_FUSED","GLM53_SHARED_GU_ROWS","GLM53_DSA_INDEX_BF16_KERNEL","GLM53_DSA_SCORE_MULTI","GLM53_MHC_POST_PRE_DECODE","GLM53_SPEC_MAX_DRAFT","GLM53_DRAFT_GQA","GLM53_MLA_ACTIVE_COPY","GLM53_DENSE_FP8","GLM53_COOP_GEOMETRY","GLM53_DENSE_LT","GLM53_MLA_SPARSE_FUSED","GLM53_TP_MOE_PACK","GLM53_TP_SMALL_COMM_ACTIVE","GLM53_DRAFT_HEAD_TP","GLM53_DRAFT_KV_BUFFER","GLM53_MOE_COOP_PERSISTENT","GLM53_KDA_FORK_FUSED","GLM53_DENSE_SMALL","GLM53_MHC_POST_FUSED","GLM53_STATIC_TENSORS","GLM53_DRAFT_ROPE_CACHE","GLM53_DRAFT_MLP_TP","GLM53_DRAFT_TOPK_TP"].contains(&flag.as_str()));assert_eq!(repeats,4);}
    let skip_serial=std::env::var("GLM53_SPEC_SKIP_SERIAL_BASELINE").as_deref()==Ok("1");
    let saved_flag=abba_flag.as_ref().and_then(|flag|std::env::var(flag).ok());
    let extra_flags:Vec<(String,Option<String>)>=std::env::var("GLM53_SPEC_ABBA_EXTRA").unwrap_or_default()
        .split(',').filter(|s|!s.is_empty()).map(|s|{
            assert!(abba_flag.is_some() && PREFILL_ABBA_FLAGS.contains(&s),"unsupported ABBA extra flag");
            (s.to_owned(),std::env::var(s).ok())}).collect();
    let off_value=std::env::var("GLM53_SPEC_ABBA_OFF").unwrap_or_else(|_|"0".into());
    let on_value=std::env::var("GLM53_SPEC_ABBA_ON").unwrap_or_else(|_|"1".into());
    let mut cases=Vec::new();
    let ints=|v:&serde_json::Value|v.as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect::<Vec<_>>();
    let (inputs,stop):(Vec<(String,Vec<i64>,usize)>,Vec<i64>)=if let Ok(path)=std::env::var("GLM53_SPEC_SUITE") {
        let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        (suite["cases"].as_array().unwrap().iter().map(|c|(c["name"].as_str().unwrap().to_owned(),ints(&c["prompt_ids"]),c["max_new"].as_u64().unwrap() as usize)).collect(),ints(&suite["eos_token_ids"]))
    }else{(["hello","count","hashmap"].into_iter().map(|name|(name.to_owned(),ints(&refs[name]["prompt_ids"]),n)).collect(),Vec::new())};
    for (name,ids,limit) in inputs {
        if let Some(flag)=&abba_flag {std::env::set_var(flag,&off_value);for (extra,_) in &extra_flags{std::env::set_var(extra,&off_value);}}
        crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
        let prefix_start=timed_start(dev);
        let (logits,initial,features)=eng.prefill_record_last(&Tensor::from_slice(&ids).to_device(dev),None,true);
        tch::Cuda::synchronize(0);let baseline_prefill_ms=prefix_start.elapsed().as_secs_f64()*1000.;
        assert_eq!(features.len(),5);let features=Tensor::cat(&features,1);
        let first=logits.get(logits.size()[0]-1).argmax(-1,false).int64_value(&[]);
        let mut baseline=Vec::new();let mut baseline_state=(!skip_serial).then(||snapshot(&initial));let mut next=first;
        let started=timed_start(dev);
        for i in 0..if skip_serial{0}else{limit} {let token=next;baseline.push(token);
            if i+1==limit || stop.contains(&token){break;}
            next=eng.step(token,baseline_state.as_mut().unwrap()).argmax(-1,false).int64_value(&[]);}
        let n=if skip_serial{limit}else{baseline.len()};
        tch::Cuda::synchronize(0);let baseline_ms=started.elapsed().as_secs_f64()*1000.;
        // Commit the final token outside timing solely for state diagnostics.
        if let Some(&token)=baseline.last(){let _=eng.step(token,baseline_state.as_mut().unwrap());}
        // Performance-only paired runs have no serial reference. Avoid two
        // unused full-capacity snapshots (about 640 MiB at TP2 / 20K).
        let baseline_end=baseline_state.as_ref().map(snapshot);
        let mut graph_rounds=Vec::new();
        if !skip_serial {
        let mut graph_state=snapshot(&initial);
        let mut graph_input=Tensor::from_slice(&[first]).to_device(dev);
        let _=eng.step_buf(&graph_input,&mut graph_state);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();
        let graph_output=eng.step_buf(&graph_input,&mut graph_state);
        crate::tp::graph::end().unwrap();
        for _ in 0..3 {
            crate::forward::restore(&mut graph_state,&initial);let mut next=first;let mut tokens=Vec::new();
            let started=timed_start(dev);
            for i in 0..n {tokens.push(next);if i+1==n{break;}
                graph_input.copy_(&Tensor::from_slice(&[next]).to_device(dev));
                crate::tp::graph::replay().unwrap();next=graph_output.argmax(-1,false).int64_value(&[]);}
            tch::Cuda::synchronize(0);graph_rounds.push(started.elapsed().as_secs_f64()*1000.);
            assert_eq!(tokens,baseline,"graph baseline differs for {name}");
        }
        crate::tp::graph::destroy();
        }
        let requested=std::env::var("GLM53_SPEC_MODES").unwrap_or_else(|_|"chain,tree,batch-chain,batch-tree".into());
        assert!(!skip_serial || (abba_flag.is_some() && requested.split(',').all(|m|m.starts_with("batch-"))),"serial baseline can only be skipped in paired batched performance tests");
        let mut modes=Vec::new();
        // A single verifier warmup misses the drafter's growing-context shapes.
        // For paired flag experiments, run one complete request per arm first.
        let full_warmups=std::env::var("GLM53_SPEC_FULL_WARMUPS").ok().map(|v|v.parse::<usize>().unwrap())
            .unwrap_or(if abba_flag.is_some(){2}else{0});
        assert!(full_warmups<=4);
        let schedule=(0..full_warmups).map(|r|(r,true)).chain((0..repeats).map(|r|(r,false)));
        for (round,warmup,mode) in schedule.flat_map(|(r,w)|requested.split(',').map(move|m|(r,w,m))) {
            if let Some(flag)=&abba_flag {let value=if round==1||round==2{&on_value}else{&off_value};std::env::set_var(flag,value);for (extra,_) in &extra_flags{std::env::set_var(extra,value);}}
            crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
            assert!(["chain","tree","batch-chain","batch-tree","batch-graph-chain","batch-graph-tree","batch-adaptive-chain"].contains(&mode));
            // A prefill-only candidate must actually recompute the prefix in
            // each arm. Keep this time separate from generation throughput.
            let arm_prefix=if !extra_flags.is_empty() || abba_flag.as_deref().map_or(false,|flag|PREFILL_ABBA_FLAGS.contains(&flag)) {
                let started=timed_start(dev);
                let (l,s,f)=eng.prefill_record_last(&Tensor::from_slice(&ids).to_device(dev),None,true);
                tch::Cuda::synchronize(0);
                Some((l.get(l.size()[0]-1).argmax(-1,false).int64_value(&[]),s,Tensor::cat(&f,1),started.elapsed().as_secs_f64()*1000.))
            }else{None};
            let (first,initial,features,prefill_ms)=match &arm_prefix {
                Some((a,s,f,ms))=>(*a,s,f,*ms),None=>(first,&initial,&features,baseline_prefill_ms)};
            let mut state=snapshot(&initial);let mut context=drafter.empty_context();drafter.append(&mut context,&features);
            let mut anchor=first;let mut generated=Vec::new();let mut cycles=Vec::new();
            let mut deferred_terminal=None;let mut deferred_append:Option<AcceptedFeatures>=None;
            // Exclude backend initialization, allocator warmup and graph capture.
            // The complete generation below starts again from the untouched prefix.
            let warm_candidates=drafter.propose(&context,anchor,&eng.w);
            let warm_depth=if mode=="batch-adaptive-chain"{7}else{draft_depth()};
            let warm_nodes=if mode.ends_with("tree"){tree(&warm_candidates,warm_depth,warm_depth)}else{crate::speculative::chain(&warm_candidates.path[..warm_depth])};
            let warm_nodes=with_anchor(anchor,warm_nodes);
            let mut chain_graph=if mode=="batch-graph-chain" {
                let mut graph=ChainCache::new(eng,&initial,&warm_nodes);graph.validate(eng,&initial,&warm_nodes);
                if crate::dflash::conf_tau().is_some() {
                    // Capture every truncated depth before timing; results are discarded.
                    for d in 1..=warm_depth {let nodes=with_anchor(anchor,crate::speculative::chain(&warm_candidates.path[..d]));let _=graph.verify(eng,&initial,anchor,&nodes,d+1,&[]);}
                }
                Some(graph)
            } else {None};
            let mut tree_graphs=GraphPool::new();
            let mut adaptive=if mode=="batch-adaptive-chain" {
                Some(calibrate(eng,drafter,&context,&initial,anchor,&warm_nodes,&mut tree_graphs))
            }else{None};
            if mode=="batch-graph-tree" {tree_graphs.get(eng,&initial,&warm_nodes).validate(eng,&initial,&warm_nodes);}
            if mode.starts_with("batch-") {
                let warm=if let Some(graph)=&mut chain_graph {graph.verify(eng,&initial,anchor,&warm_nodes,warm_depth+1,&[])}
                    else if mode=="batch-graph-tree"{{let graph=tree_graphs.get(eng,&initial,&warm_nodes);graph.verify_checked(&initial,anchor,&warm_nodes,warm_depth+1,&[],graph.selection)}}
                    else {batched(eng,&initial,anchor,&warm_nodes,warm_depth+1,&[])};
                let mut warm_context=drafter.empty_context();drafter.append(&mut warm_context,&features);
                drafter.append(&mut warm_context,&warm.aux.join());
            }
            tch::Cuda::synchronize(0);
            let capture_profile=!warmup&&std::env::var("GLM53_SPEC_NSYS_CASE").ok().as_deref()==Some(name.as_str())&&round==0&&mode=="batch-graph-chain";
            extern "C" {fn cudaProfilerStart()->i32;fn cudaProfilerStop()->i32;}
            if capture_profile{assert_eq!(unsafe{cudaProfilerStart()},0);}
            let memory_before=crate::host_memory::cuda_snapshot(true);
            let host_before=crate::host_memory::snapshot();
            let started=timed_start(dev);
            while generated.len()<limit {
                if generated.len()+1==limit || stop.contains(&anchor) {
                    generated.push(anchor);deferred_terminal=Some(anchor);
                    let stages=if std::env::var("GLM53_SPEC_STAGE_PROFILE").as_deref()==Ok("1") {
                        Some(json!({"draft_ms":0.,"verify_commit_ms":0.,"append_ms":0.}))
                    }else{None};
                    cycles.push(json!({"accepted_drafts":0,"emitted":1,"evaluated_nodes":0,"draft_budget":0,
                        "anchor_batched":false,"graph_used":false,"terminal_emit_only":true,"accepted_feature_prefix_view":false,"synchronized_stages":stages}));break;
                }
                let stage_profile=std::env::var("GLM53_SPEC_STAGE_PROFILE").as_deref()==Ok("1");
                let cycle_start=Instant::now();
                let remaining=limit-generated.len();let depth=(remaining-1).min(adaptive.as_ref().map_or(draft_depth(),|p|p.depth()));
                // Terminal known tokens were emitted above without GPU work.
                let candidates=if depth>0 && !stop.contains(&anchor) {Some(drafter.propose(&context,anchor,&eng.w))}else{None};
                let selector_fused=candidates.as_ref().map(|c|c.selector_fused);
                let final_norm_selected=candidates.as_ref().map(|c|c.final_norm_selected);
                let conv_calls=candidates.as_ref().map(|c|c.conv_calls);let conv_fused_calls=candidates.as_ref().map(|c|c.conv_fused_calls);
                // Diagnostic: per-position softmax confidence of the chosen path token over the top-16.
                let cand_ids:Option<Vec<i64>>=if std::env::var("GLM53_SPEC_LOG_CAND").as_deref()==Ok("1") {candidates.as_ref().map(|c|Vec::<i64>::try_from(c.ids.reshape([-1]).to_device(Device::Cpu)).unwrap())}else{None};
                let draft_conf:Option<Vec<f64>>=if std::env::var("GLM53_SPEC_LOG_CONF").as_deref()==Ok("1") {candidates.as_ref().map(|c|{
                    let p=c.unary.softmax(-1,Kind::Float);let path=Tensor::from_slice(&c.path).to_device(p.device()).view([-1,1]);
                    let hit=c.ids.eq_tensor(&path).to_kind(Kind::Float);Vec::<f64>::try_from((p*hit).sum_dim_intlist(&[-1i64][..],false,Kind::Float).to_kind(Kind::Double).to_device(Device::Cpu)).unwrap()})}else{None};
                // L2 depth policy: stop the chain at the first path position whose drafter
                // confidence is below tau (keep at least one draft); target output is unchanged.
                let depth=match (crate::dflash::conf_tau(),candidates.as_ref().and_then(|c|c.conf.as_ref())) {
                    (Some(tau),Some(conf)) if !mode.ends_with("tree") => crate::spec_policy::chain_depth(conf,depth,tau),
                    _=>depth};
                let nodes=match candidates {
                    Some(c) if mode.ends_with("tree")=>tree(&c,depth,depth),
                    Some(c)=>crate::speculative::chain(&c.path[..depth]),None=>Vec::new(),
                };
                if stage_profile{tch::Cuda::synchronize(0);}
                let draft_ms=cycle_start.elapsed().as_secs_f64()*1000.;let verify_start=Instant::now();
                if mode.starts_with("batch-") {
                    let all=with_anchor(anchor,nodes);
                    let adaptive_graph=adaptive.is_some() && [4,6,8].contains(&all.len());
                    let mut graph_used=((all.len()==draft_depth()+1||crate::dflash::conf_tau().is_some())&&chain_graph.is_some())||mode=="batch-graph-tree"||adaptive_graph;
                    let verified=if mode=="batch-graph-tree" {
                        let selection=ChainGraph::selection(&state,&all);
                        match tree_graphs.try_get_selected(eng,&state,&all,remaining,selection) {
                            Some(graph)=>graph.verify_checked(&state,anchor,&all,depth+1,&stop,selection),
                            None=>{graph_used=false;batched_selected(eng,&state,anchor,&all,depth+1,&stop,selection)},
                        }
                    }else if adaptive_graph{{let graph=tree_graphs.get(eng,&state,&all);graph.verify_checked(&state,anchor,&all,depth+1,&stop,graph.selection)}}
                        else if graph_used{chain_graph.as_mut().unwrap().verify(eng,&state,anchor,&all,depth+1,&stop)}
                        else {batched(eng,&state,anchor,&all,depth+1,&stop)};
                    if !warmup && round==0 && graph_used && chain_graph.is_some() && cycles.len()<8 {
                        if let Ok(dir)=std::env::var("GLM53_SPEC_ROUTE_TRACE") {
                            let dir=Path::new(&dir);std::fs::create_dir_all(dir).unwrap();
                            crate::route_trace::save(chain_graph.as_ref().unwrap().routes(),
                                &dir.join(format!("{name}-rank{}-cycle{}.pt",tp.rank,cycles.len())));
                        }
                    }
                    let prefix_used=verified.aux.is_prefix();
                    let accepted=verified.tokens.len()-1;if let Some(policy)=&mut adaptive{policy.observe(accepted,depth);}
                    generated.extend(verified.tokens);state=verified.state;anchor=verified.next;
                    if stage_profile{tch::Cuda::synchronize(0);}
                    let verify_ms=verify_start.elapsed().as_secs_f64()*1000.;let append_start=Instant::now();
                    let finished=generated.len()==limit || generated.last().map_or(false,|t|stop.contains(t));
                    if finished{deferred_append=Some(verified.aux);}
                    else{drafter.append(&mut context,&verified.aux.join());}
                    if stage_profile{tch::Cuda::synchronize(0);}
                    let stages=if stage_profile{Some(json!({"draft_ms":draft_ms,"verify_commit_ms":verify_ms,"append_ms":append_start.elapsed().as_secs_f64()*1000.}))}else{None};
                    cycles.push(json!({"accepted_drafts":accepted,"emitted":accepted+1,"evaluated_nodes":verified.evaluated-1,"draft_budget":depth,"anchor_batched":true,"graph_used":graph_used,"accepted_feature_prefix_view":prefix_used,"synchronized_stages":stages,
                        "draft_selector_fused":selector_fused,"draft_final_norm_selected":final_norm_selected,"draft_conv_calls":conv_calls,"draft_conv_fused_calls":conv_fused_calls,
                        "draft_conf":draft_conf,"cand_ids":cand_ids,"host_cycle_ms":cycle_start.elapsed().as_secs_f64()*1000.}));
                    if finished{break;}
                    continue;
                }
                let (next,anchor_feature)=ModelTarget(&mut *eng).step(&mut state,anchor);generated.push(anchor);
                if stop.contains(&anchor){drafter.append(&mut context,&anchor_feature);cycles.push(json!({"accepted_drafts":0,"emitted":1,"evaluated_nodes":0,"draft_budget":depth,"accepted_feature_prefix_view":false,"host_cycle_ms":cycle_start.elapsed().as_secs_f64()*1000.}));break;}
                let verified=crate::speculative::verify_until(&mut ModelTarget(&mut *eng),&state,next,&nodes,depth,&stop);
                let accepted=verified.tokens.len();generated.extend(verified.tokens);state=verified.state;anchor=verified.next;
                let mut appended=vec![anchor_feature];appended.extend(verified.aux);
                let finished=generated.len()==limit || generated.last().map_or(false,|t|stop.contains(t));
                if finished{deferred_append=Some(AcceptedFeatures::Rows(appended));}else{drafter.append(&mut context,&Tensor::cat(&appended,0));}
                cycles.push(json!({"accepted_drafts":accepted,"emitted":accepted+1,"evaluated_nodes":verified.evaluated,"draft_budget":depth,"accepted_feature_prefix_view":false,
                    "draft_selector_fused":selector_fused,"draft_final_norm_selected":final_norm_selected,"draft_conv_calls":conv_calls,"draft_conv_fused_calls":conv_fused_calls,"host_cycle_ms":cycle_start.elapsed().as_secs_f64()*1000.}));
                if finished{break;}
            }
            tch::Cuda::synchronize(0);let elapsed_ms=started.elapsed().as_secs_f64()*1000.;
            if state_inplace_enabled(){state=crate::forward::snapshot(&state);}
            let host_after=crate::host_memory::snapshot();
            let memory_after=crate::host_memory::cuda_snapshot(false);
            if capture_profile{assert_eq!(unsafe{cudaProfilerStop()},0);}
            // Public decoding discards these states. The probe completes them
            // after timing to retain its stronger state/context parity checks.
            if let Some(token)=deferred_terminal {
                let (_,feature)=ModelTarget(&mut *eng).step(&mut state,token);
                deferred_append=Some(AcceptedFeatures::Rows(vec![feature]));
            }
            if let Some(features)=deferred_append{drafter.append(&mut context,&features.join());}
            let equal=generated==baseline;
            let state_error=if equal&&!skip_serial{crate::forward::states_max_diff(&state,baseline_end.as_ref().expect("serial reference state missing"))}else{f64::NAN};
            if !mode.starts_with("batch-") {
                assert!(equal,"{mode} target greedy differs for {name}");
                assert_eq!(state_error,0.,"committed state differs {name}/{mode}");
            }
            assert_eq!(context.len,(ids.len()+generated.len()) as i64);
            let tok_s=generated.len() as f64*1000./elapsed_ms;
            if warmup {
                eprintln!("[spec-probe-warmup] rank{} {name} {mode} arm={round} full request {} tokens",tp.rank,generated.len());
                continue;
            }
            let selector_proposals=cycles.iter().filter(|c|c["draft_selector_fused"].is_boolean()).count();
            let selector_fused_proposals=cycles.iter().filter(|c|c["draft_selector_fused"].as_bool()==Some(true)).count();
            let conv_calls:usize=cycles.iter().filter_map(|c|c["draft_conv_calls"].as_u64()).map(|n|n as usize).sum();
            let conv_fused_calls:usize=cycles.iter().filter_map(|c|c["draft_conv_fused_calls"].as_u64()).map(|n|n as usize).sum();
            let conv_proposals=cycles.iter().filter(|c|c["draft_conv_calls"].is_number()).count();
            let mut mode_record=json!({"mode":mode,"round":round,"tokens":generated,"exact_target_greedy":if skip_serial{None}else{Some(equal)},
                "host_before":host_before,"host_after":host_after,"target_top1_tp":crate::head_select::enabled(),"accepted_feature_view":crate::feature_prefix::enabled(),
                "host_cycle_scope":"CPU elapsed without added synchronization; asynchronous append can be attributed across cycles; not GPU stage timing",
                "full_request_warmups":full_warmups,"max_draft":warm_depth,"requested_max_new":limit,"prefill_ms":prefill_ms,
                "adaptive_calibrated_ms":adaptive.as_ref().map(|p|p.costs),"tree_graph_captures":tree_graphs.captures,"tree_graph_hits":tree_graphs.hits,
                "tree_eager_misses":tree_graphs.eager_misses,
                "abba_extra_flags":extra_flags.iter().map(|(s,_)|s).collect::<Vec<_>>(),"abba_flag":abba_flag,"abba_enabled":abba_flag.as_ref().map(|f|std::env::var(f).ok().as_deref()==Some(on_value.as_str())),"abba_off":off_value,"abba_on":on_value,
                "abba_arm_index":abba_flag.as_ref().map(|_|round%4),"abba_arm_name":abba_flag.as_ref().map(|_|if matches!(round%4,1|2){"B"}else{"A"}),
                "mla_score_scope":std::env::var("GLM53_MLA_SCORE_2D_SCOPE").ok(),
                "mla_score_min_rows":std::env::var("GLM53_MLA_SCORE_2D_MIN_ROWS").ok(),
                "draft_mlp_tp":std::env::var("GLM53_DRAFT_MLP_TP").ok(),"draft_rope_cache":std::env::var("GLM53_DRAFT_ROPE_CACHE").ok(),
                "tp_moe_pack":crate::moe::tp_pack_enabled(),
                "small_comm":crate::tp::small_comm_enabled(),
                "mla_score_2d":std::env::var("GLM53_MLA_SCORE_2D").as_deref()==Ok("1"),
                "draft_head_tp":std::env::var("GLM53_DRAFT_HEAD_TP").ok(),"draft_kv_buffer":std::env::var("GLM53_DRAFT_KV_BUFFER").ok(),
                "draft_topk_tp":std::env::var("GLM53_DRAFT_TOPK_TP").ok(),
                "draft_kv_buffer_min_context":std::env::var("GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT").ok(),
                "cuda_memory_allocated_reserved":crate::tp::graph::memory(),
                "same_tokens_state_max_abs":if equal{Some(state_error)}else{None},
                "route_trace":std::env::var("GLM53_SPEC_ROUTE_TRACE").ok(),"elapsed_ms":elapsed_ms,"tok_s":tok_s,"cycles":cycles});
            let record=mode_record.as_object_mut().unwrap();
            record.insert("shared_gu_fused".into(),json!(crate::shared_gu::enabled()));
            record.insert("dsa_topk_batch_enabled".into(),json!(crate::dsa_topk::enabled()));
                record.insert("dsa_position_capture_enabled".into(),json!(crate::dsa_position::enabled()));
            record.insert("cuda_allocator_before".into(),memory_before);
            record.insert("cuda_allocator_after".into(),memory_after);
            record.insert("draft_selector_enabled".into(),json!(crate::draft_selector::enabled()));
            record.insert("draft_selector_proposals".into(),json!(selector_proposals));
            record.insert("draft_selector_fused_proposals".into(),json!(selector_fused_proposals));
            record.insert("draft_conv_enabled".into(),json!(crate::draft_conv::enabled()));
            record.insert("draft_conv_proposals".into(),json!(conv_proposals));
            record.insert("draft_conv_calls".into(),json!(conv_calls));
            record.insert("draft_conv_fused_calls".into(),json!(conv_fused_calls));
            record.insert("draft_final_norm_enabled".into(),json!(crate::draft_final_norm::enabled()));
            record.insert("draft_final_norm_proposals".into(),json!(cycles.iter().filter(|c|c["draft_final_norm_selected"].is_boolean()).count()));
            record.insert("draft_final_norm_selected_proposals".into(),json!(cycles.iter().filter(|c|c["draft_final_norm_selected"].as_bool()==Some(true)).count()));
            modes.push(mode_record);
            eprintln!("[spec-probe] rank{} {name} {mode} target_equal={equal}; {tok_s:.2} tok/s",tp.rank);
        }
        let abba_exact=if abba_flag.is_some() {
            Some(modes.iter().all(|m|{let reference=modes.iter().find(|r|r["mode"]==m["mode"]).unwrap();m["tokens"]==reference["tokens"] &&
                m["cycles"].as_array().unwrap().iter().map(|c|(&c["accepted_drafts"],&c["emitted"]))
                .eq(reference["cycles"].as_array().unwrap().iter().map(|c|(&c["accepted_drafts"],&c["emitted"]))) }))
        }else{None};
        if std::env::var("GLM53_SPEC_REQUIRE_ABBA_EXACT").as_deref()==Ok("1") {
            assert_eq!(abba_exact,Some(true),"ABBA changed emitted tokens or accepted drafts for {name}");
        }
        assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
        cases.push(json!({"name":name,"prompt_tokens":ids.len(),"baseline_ids":baseline,"serial_baseline_omitted":skip_serial,"baseline_eager_ms":if skip_serial{None}else{Some(baseline_ms)},"baseline_graph_ms":graph_rounds,"abba_tokens_and_acceptance_exact":abba_exact,"modes":modes}));
        std::fs::write(out.join(format!("full-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"tokens":n,"cold_loads":0,
            "moe_batch":std::env::var("GLM53_MOE_BATCH").unwrap_or_default(),"moe_coop":std::env::var("GLM53_MOE_COOP").unwrap_or_default(),
            "verifier":"serial and batched projection variants, branch-local snapshots; batching changes GEMM rounding","cases":cases})).unwrap()).unwrap();
    }
    if let Some(flag)=abba_flag {match saved_flag {Some(v)=>std::env::set_var(flag,v),None=>std::env::remove_var(flag)}}
    for (flag,saved) in extra_flags {match saved{Some(v)=>std::env::set_var(flag,v),None=>std::env::remove_var(flag)}}
    crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
}

/// Isolate the actual two-node target path, where the scoped small-N kernel runs.
pub fn small_tree_check(eng:&mut Engine,out:&Path) {
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let mut cases=Vec::new();let dev=eng.w.device;std::fs::create_dir_all(out).unwrap();
    for name in ["hello","count","hashmap"] {
        let ids:Vec<i64>=refs[name]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
        std::env::set_var("GLM53_DENSE_SMALL","0");
        let (logits,base)=eng.prefill(&Tensor::from_slice(&ids).to_device(dev));
        let anchor=logits.get(logits.size()[0]-1).argmax(-1,false).int64_value(&[]);
        let nodes=vec![Node{parent:None,token:anchor},Node{parent:Some(0),token:13041}];
        let input=Tensor::from_slice(&[anchor,13041]).to_device(dev);
        let (a,_,_)=eng.tree_forward(&input,&base,&[None,Some(0)],true);
        std::env::set_var("GLM53_DENSE_SMALL","1");let (b,_,_)=eng.tree_forward(&input,&base,&[None,Some(0)],true);
        let labels:Vec<i64>=Vec::try_from(a.argmax(-1,false).to_device(Device::Cpu)).unwrap();
        let metrics=crate::evaluation::distribution(&a.to_device(Device::Cpu),&b.to_device(Device::Cpu),&labels);
        let mut rounds=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_DENSE_SMALL",if on{"1"}else{"0"});
            let mut graph=ChainGraph::new(eng,&base,&nodes);graph.validate(eng,&base,&nodes);
            for _ in 0..3{graph.replay(&base,&nodes);}let start=timed_start(dev);
            for _ in 0..32{graph.replay(&base,&nodes);}tch::Cuda::synchronize(0);
            rounds.push(json!({"small":on,"ms":start.elapsed().as_secs_f64()*1000./32.}));
        }
        cases.push(json!({"name":name,"distribution":metrics,"rounds":rounds}));
        std::fs::write(out.join(format!("rank{}.json",crate::tp::world().rank)),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
    }
    std::env::set_var("GLM53_DENSE_SMALL","0");
}

// Append inside spec_probe.rs after applying runtime-integration.patch.
// Same GraphPool/ChainCache must distinguish INDEX_FUSED without changing shape.
pub fn dsa_index_graph_check(eng:&mut Engine,out:&Path) {
    struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
    impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
        if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
    }}}
    let _env=Restore(["GLM53_DSA_INDEX_FUSED","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"]
        .into_iter().map(|k|(k,std::env::var_os(k))).collect());
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    std::env::set_var("GLM53_DSA_INDEX_FUSED","0");
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("dsa-index-graphs-rank{}.json",crate::tp::world().rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let dev=eng.w.device;let (_,base)=eng.prefill(&Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(dev));
    let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13042i64]).to_device(dev),Some(snapshot(&base)));
    let mut nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:13042}];
    let mut pool=GraphPool::new();pool.limit=2;let mut records=Vec::new();
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_INDEX_FUSED",if mode{"1"}else{"0"});
        nodes[1].token=13042+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let graph=pool.get(eng,current,&nodes);assert_eq!(graph.selection,TreeSelection::Ranked);
        assert_eq!(graph.dsa_index,mode);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"GraphPool","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(pool.captures,2,"index flag must produce two captures");assert_eq!(pool.hits,2,"return to old index flag must reuse its own graph");
    drop(pool);
    std::env::set_var("GLM53_DSA_INDEX_FUSED","0");let mut cache=ChainCache::new(eng,&base,&nodes);
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_INDEX_FUSED",if mode{"1"}else{"0"});
        nodes[1].token=13046+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let _=cache.verify(eng,current,nodes[0].token,&nodes,nodes.len(),&[]);
        let last=cache.last_used;let graph=&mut cache.entries[last];assert_eq!(graph.dsa_index,mode);
        assert_eq!(graph.selection,TreeSelection::Ranked);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"ChainCache","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(cache.entries.len(),2,"fixed-chain cache must distinguish both modes");
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"records":records,
        "graph_pool_captures":2,"graph_pool_hits":2,"chain_cache_entries":2,
        "scope":"same-cache mode toggle plus changed tokens/advanced prefix; kernel dispatch requires local seam gate or separate trace"})).unwrap()).unwrap();
}

/// Three bounded identity comparisons prove each flag separately and together.
/// This is a diagnostic, never called by production dispatch.
pub(crate) fn fp8_small_graph_check(eng:&mut Engine,out:&Path) {
    struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
    impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
        if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
    }}}
    let _env=Restore(["GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"]
        .into_iter().map(|k|(k,std::env::var_os(k))).collect());
    let set=|(transpose,pad):(bool,bool)| {
        std::env::set_var("GLM53_FP8_SMALL_TRANSPOSE",if transpose{"1"}else{"0"});
        std::env::set_var("GLM53_FP8_SMALL_PAD",if pad{"1"}else{"0"});
    };
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    set((false,false));std::fs::create_dir_all(out).unwrap();
    let path=out.join(format!("fp8-small-graphs-rank{}.json",crate::tp::world().rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let dev=eng.w.device;let (_,base)=eng.prefill(&Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(dev));
    let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13042i64]).to_device(dev),Some(snapshot(&base)));
    let mut nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:13042}];
    let mut records=Vec::new();
    for (label,a,b) in [("transpose",(false,false),(true,false)),("pad",(false,false),(false,true)),
                       ("transpose_with_pad",(false,true),(true,true))] {
        let mut pool=GraphPool::new();pool.limit=2;
        for (turn,mode) in [a,b,a,b].into_iter().enumerate() {
            set(mode);nodes[1].token=13042+turn as i64;let current=if turn<2{&base}else{&advanced};
            let graph=pool.get(eng,current,&nodes);assert_eq!(graph.fp8_small.0,mode);
            assert_eq!(graph.selection,TreeSelection::Ranked);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
            records.push(json!({"axis":label,"cache":"GraphPool","turn":turn,"flags":mode,"same_mode_graph_eager_exact":true}));
        }
        assert_eq!(pool.captures,2,"FP8 small flags must cause separate captures");assert_eq!(pool.hits,2);
        drop(pool);set(a);let mut cache=ChainCache::new(eng,&base,&nodes);
        for (turn,mode) in [a,b,a,b].into_iter().enumerate() {
            set(mode);nodes[1].token=13046+turn as i64;let current=if turn<2{&base}else{&advanced};
            let _=cache.verify(eng,current,nodes[0].token,&nodes,nodes.len(),&[]);
            let last=cache.last_used;let graph=&mut cache.entries[last];assert_eq!(graph.fp8_small.0,mode);
            graph.assert_storage_gate();graph.validate(eng,current,&nodes);
            records.push(json!({"axis":label,"cache":"ChainCache","turn":turn,"flags":mode,"same_mode_graph_eager_exact":true}));
        }
        assert_eq!(cache.entries.len(),2);
    }
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"records":records,
        "graph_pool_captures_per_axis":2,"graph_pool_hits_per_axis":2,"chain_cache_entries_per_axis":2,
        "scope":"cache identity and changed-token/advanced-prefix same-arm replay; numerical off/on is the separate verifier gate; kernel hit requires local seam evidence/trace"})).unwrap()).unwrap();
}

/// Candidate identity must cause a cache miss, then reuse its own capture.
/// Eager off/on numerical proof lives in verifier_probe, not this identity gate.
pub(crate) fn shared_gu_graph_check(eng:&mut Engine,out:&Path) {
    struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
    impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
        if let Some(value)=value{std::env::set_var(key,value);}else{std::env::remove_var(key);}
    }}}
    let _restore=Restore(["GLM53_SHARED_GU_FUSED","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"]
        .into_iter().map(|k|(k,std::env::var_os(k))).collect());
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    let set=|mode:bool|std::env::set_var("GLM53_SHARED_GU_FUSED",if mode{"1"}else{"0"});
    set(true);
    let x=Tensor::zeros([2,4096],(Kind::Float,eng.w.device));
    let mut eligible=0;
    for layer in &eng.w.layers {if let Some(m)=&layer.moe {
        assert!(crate::shared_gu::metadata_eligible(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd),"shared GU graph gate must exercise all sparse layers");eligible+=1;
    }}
    assert_eq!(eligible,42,"expected actual TP2 shared layers");drop(x);
    set(false);std::fs::create_dir_all(out).unwrap();let path=out.join(format!("shared-gu-graphs-rank{}.json",crate::tp::world().rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let dev=eng.w.device;let (_,base)=eng.prefill(&Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(dev));
    let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13042i64]).to_device(dev),Some(snapshot(&base)));
    let mut nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:13042}];let mut records=Vec::new();
    let mut pool=GraphPool::new();pool.limit=2;
    for (turn,mode) in [false,true,false,true].into_iter().enumerate(){
        set(mode);nodes[1].token=13042+turn as i64;let current=if turn<2{&base}else{&advanced};
        let graph=pool.get(eng,current,&nodes);assert_eq!(graph.shared_gu,mode);
        graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"GraphPool","turn":turn,"mode":mode,"same_arm_graph_eager_exact":true}));
    }
    assert_eq!(pool.captures,2);assert_eq!(pool.hits,2);drop(pool);
    set(false);let mut cache=ChainCache::new(eng,&base,&nodes);
    for (turn,mode) in [false,true,false,true].into_iter().enumerate(){
        set(mode);nodes[1].token=13046+turn as i64;let current=if turn<2{&base}else{&advanced};
        let _=cache.verify(eng,current,nodes[0].token,&nodes,nodes.len(),&[]);
        let last=cache.last_used;let graph=&mut cache.entries[last];assert_eq!(graph.shared_gu,mode);
        graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"ChainCache","turn":turn,"mode":mode,"same_arm_graph_eager_exact":true}));
    }
    assert_eq!(cache.entries.len(),2);
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"eligible_layers":eligible,"records":records,
        "captures":2,"hits":2,"chain_entries":2,"scope":"runtime identity; exact off/on is separate verifier gate; trace must confirm 42 shared_gu production launches per full graph"})).unwrap()).unwrap();
}

/// Fixed Ranked mode: isolate batch TopK identity from index bookkeeping.
pub fn dsa_topk_graph_check(eng:&mut Engine,out:&Path) {
    let before=crate::session::signature();let before_spec=crate::spec_session::signature();
    struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
    impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
        if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
    }}}
    let _env=Restore(["GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"]
        .into_iter().map(|k|(k,std::env::var_os(k))).collect());
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    std::env::set_var("GLM53_DSA_INDEX_FUSED","1");
    std::env::set_var("GLM53_DSA_TOPK_BATCH","0");
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("dsa-topk-graphs-rank{}.json",crate::tp::world().rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let dev=eng.w.device;let (_,base)=eng.prefill(&Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(dev));
    let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13042i64]).to_device(dev),Some(snapshot(&base)));
    // Metadata admission against the actual model/state. This does not claim
    // dynamic kernel hits; the candidate trace is the separate dispatch gate.
    std::env::set_var("GLM53_DSA_TOPK_BATCH","1");
    let x=Tensor::empty([2,eng.w.embed.size()[1]],(Kind::Float,dev));
    let mut eligible_layers=0;
    for state in &base.0 {if let crate::forward::LayerState::MlaLatent(st)=state {
        assert!(crate::dsa_topk::eligible(TreeSelection::Ranked,&x,&st.index.pools,&st.len));eligible_layers+=1;
    }}
    assert!(eligible_layers>0);drop(x);std::env::set_var("GLM53_DSA_TOPK_BATCH","0");
    let mut nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:13042}];
    let mut pool=GraphPool::new();pool.limit=2;let mut records=Vec::new();
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_TOPK_BATCH",if mode{"1"}else{"0"});
        nodes[1].token=13042+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let graph=pool.get(eng,current,&nodes);assert_eq!(graph.selection,TreeSelection::Ranked);
        assert_eq!(graph.dsa_topk,mode);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"GraphPool","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(pool.captures,2,"TopK batch flag must produce two captures");assert_eq!(pool.hits,2,"return to old TopK batch flag must reuse its own graph");
    drop(pool);
    std::env::set_var("GLM53_DSA_TOPK_BATCH","0");let mut cache=ChainCache::new(eng,&base,&nodes);
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_TOPK_BATCH",if mode{"1"}else{"0"});
        nodes[1].token=13046+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let _=cache.verify(eng,current,nodes[0].token,&nodes,nodes.len(),&[]);
        let last=cache.last_used;let graph=&mut cache.entries[last];assert_eq!(graph.dsa_topk,mode);
        assert_eq!(graph.selection,TreeSelection::Ranked);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"ChainCache","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(cache.entries.len(),2,"fixed-chain cache must distinguish both TopK batch modes");
    drop(_env);assert_eq!(before,crate::session::signature());assert_eq!(before_spec,crate::spec_session::signature());
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"environment_restored":true,"records":records,
        "graph_pool_captures":2,"graph_pool_hits":2,"chain_cache_entries":2,"metadata_eligible_layers":eligible_layers,
        "scope":"same-cache mode toggle plus changed tokens/advanced prefix; kernel dispatch requires local seam gate or separate trace"})).unwrap()).unwrap();
}

/// Fixed Ranked mode: isolate mask position capture identity with TopK batching fixed on.
pub fn dsa_position_graph_check(eng:&mut Engine,out:&Path) {
    let before=crate::session::signature();let before_spec=crate::spec_session::signature();
    struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
    impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
        if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
    }}}
    let _env=Restore(["GLM53_DSA_TOPK_BATCH","GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_INDEX_FUSED","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"]
        .into_iter().map(|k|(k,std::env::var_os(k))).collect());
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    std::env::set_var("GLM53_DSA_INDEX_FUSED","1");
    std::env::set_var("GLM53_DSA_TOPK_BATCH","1");
    std::env::set_var("GLM53_DSA_POSITION_CAPTURE","0");
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("dsa-position-graphs-rank{}.json",crate::tp::world().rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let dev=eng.w.device;let (_,base)=eng.prefill(&Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(dev));
    let (_,advanced)=eng.prefill_with(&Tensor::from_slice(&[13042i64]).to_device(dev),Some(snapshot(&base)));
    // Metadata admission against the actual model/state. This does not claim
    // dynamic kernel hits; the candidate trace is the separate dispatch gate.
    std::env::set_var("GLM53_DSA_POSITION_CAPTURE","1");
    let x=Tensor::empty([2,eng.w.embed.size()[1]],(Kind::Float,dev));
    let mut eligible_layers=0;
    for state in &base.0 {if let crate::forward::LayerState::MlaLatent(st)=state {
        assert!(crate::dsa_topk::eligible(TreeSelection::Ranked,&x,&st.index.pools,&st.len));eligible_layers+=1;
    }}
    assert!(eligible_layers>0);drop(x);std::env::set_var("GLM53_DSA_POSITION_CAPTURE","0");
    let mut nodes=vec![Node{parent:None,token:13041},Node{parent:Some(0),token:13042}];
    let mut pool=GraphPool::new();pool.limit=2;let mut records=Vec::new();
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_POSITION_CAPTURE",if mode{"1"}else{"0"});
        nodes[1].token=13042+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let graph=pool.get(eng,current,&nodes);assert_eq!(graph.selection,TreeSelection::Ranked);
        assert_eq!(graph.dsa_position,mode);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"GraphPool","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(pool.captures,2,"position capture flag must produce two captures");assert_eq!(pool.hits,2,"return to old position capture flag must reuse its own graph");
    drop(pool);
    std::env::set_var("GLM53_DSA_POSITION_CAPTURE","0");let mut cache=ChainCache::new(eng,&base,&nodes);
    for (turn,mode) in [false,true,false,true].into_iter().enumerate() {
        std::env::set_var("GLM53_DSA_POSITION_CAPTURE",if mode{"1"}else{"0"});
        nodes[1].token=13046+turn as i64;let current=if turn<2 {&base}else{&advanced};
        let _=cache.verify(eng,current,nodes[0].token,&nodes,nodes.len(),&[]);
        let last=cache.last_used;let graph=&mut cache.entries[last];assert_eq!(graph.dsa_position,mode);
        assert_eq!(graph.selection,TreeSelection::Ranked);graph.assert_storage_gate();graph.validate(eng,current,&nodes);
        records.push(json!({"cache":"ChainCache","turn":turn,"mode":mode,"selection":"Ranked","same_mode_graph_eager_exact":true}));
    }
    assert_eq!(cache.entries.len(),2,"fixed-chain cache must distinguish both position capture modes");
    drop(_env);assert_eq!(before,crate::session::signature());assert_eq!(before_spec,crate::spec_session::signature());
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"environment_restored":true,"records":records,
        "graph_pool_captures":2,"graph_pool_hits":2,"chain_cache_entries":2,"metadata_eligible_layers":eligible_layers,
        "scope":"same-cache mode toggle plus changed tokens/advanced prefix; kernel dispatch requires local seam gate or separate trace"})).unwrap()).unwrap();
}

/// One serving store: fixed-address decode state, drafter context and the chain graphs bound
/// to that storage. `history` = tokens committed into `state` (prefix reuse); `used` = LRU clock.
struct Store {generation:u64,state:DecodeStates,context:Option<crate::dflash::Context>,history:Vec<i64>,capacity:i64,chain:Option<ChainCache>,busy:bool,used:u64,ckpt:[Option<Ckpt>;2],
    /// Prefix cache (crate::pcache): ids of the persisted segments that hold exactly this store's rows 0..segs.len()*seg.
    segs:Vec<u64>,
    /// KV pool rows of this store (GLM53_KV_POOL): returned to the pool when the store is dropped. Declared last so
    /// the state views (and graphs holding them) go first.
    lease:Option<crate::kv_pool::Lease>}
/// A fresh store of `capacity` rows: carved from the KV pool when one is installed (the admission planner has
/// checked that it fits), otherwise allocated.
fn new_store(eng:&Engine,draft:&Drafter,capacity:i64)->Store {
    let (state,lease)=match crate::kv_pool::installed() {
        Some(pool)=>{let lease=pool.alloc(capacity).unwrap_or_else(||panic!("KV pool has no {capacity}-row range (largest free {})",pool.largest_free()));
            (eng.fresh_states_pooled(&pool,&lease),Some(lease))}
        None=>(eng.fresh_states(capacity),None)};
    Store{generation:next_generation(),state,context:Some(draft.empty_context()),history:Vec::new(),capacity,chain:None,busy:false,used:0,ckpt:[None,None],segs:Vec::new(),lease}
}   // ckpt[0]: end of prompt, ckpt[1]: before the last user turn
/// Prompt checkpoint: the state right after the last prefill. Latent rows and completed DSA pools below
/// the prompt length are never rewritten by later decoding, so only the recurrent KDA state, each MLA
/// layer's length/tails/partial pool row and the drafter context are saved (plus the last-row anchor).
struct Ckpt {ids:Vec<i64>,kda:Vec<(Tensor,Tensor)>,mla:Vec<(Tensor,Tensor,Tensor,Tensor)>,draft:crate::dflash::Context,last:Tensor}
fn ckpt_save(st:&Store,ids:&[i64],last:&Tensor)->Ckpt {
    let plen=ids.len() as i64;let row=((plen-1)/4).max(0);
    let mut kda=Vec::new();let mut mla=Vec::new();
    for l in &st.state.0 {match l {
        LayerState::Kda(k)=>kda.push((k.h.copy(),k.conv.copy())),
        LayerState::MlaLatent(m)=>mla.push((m.len.copy(),m.index.tail_k.copy(),m.index.tail_gate.copy(),m.index.pools.narrow(0,row,1).copy())),
        _=>panic!("checkpoint supports KDA and latent MLA")}}
    Ckpt{ids:ids.to_vec(),kda,mla,draft:st.context.as_ref().unwrap().snapshot(),last:last.copy()}
}
fn ckpt_restore(st:&mut Store,which:usize) {
    let c=st.ckpt[which].as_ref().unwrap();let row=((c.ids.len() as i64-1)/4).max(0);
    let (mut ki,mut mi)=(0,0);
    for l in &st.state.0 {match l {
        LayerState::Kda(k)=>{k.h.shallow_clone().copy_(&c.kda[ki].0);k.conv.shallow_clone().copy_(&c.kda[ki].1);ki+=1;}
        LayerState::MlaLatent(m)=>{let (len,tk,tg,pr)=&c.mla[mi];m.len.shallow_clone().copy_(len);m.index.tail_k.shallow_clone().copy_(tk);
            m.index.tail_gate.shallow_clone().copy_(tg);m.index.pools.narrow(0,row,1).copy_(pr);mi+=1;}
        _=>unreachable!()}}
    st.context=Some(c.draft.snapshot());st.history=c.ids.clone();
}
/// M2: copy store `src`'s prompt checkpoint into store `dst` (already fresh or reset). Latent rows and
/// complete DSA pool rows below the prompt length are never rewritten by `src`'s later decoding (it
/// may be busy), so they are copied from its live storage; the partial pool row, tails, length, KDA
/// state and drafter context come from the checkpoint. The result is bitwise the state `src` had right
/// after its prompt prefill. `dst` also gets the checkpoint (shared read-only tensors).
fn ckpt_clone(stores:&mut Vec<Option<Store>>,src:usize,which:usize,dst:usize) {
    assert_ne!(src,dst);
    let (ids,kda,mla,draft,last,rows,segs)={
        let s=stores[src].as_ref().unwrap();let c=s.ckpt[which].as_ref().expect("clone source checkpoint");
        let rows:Vec<(Tensor,Tensor)>=s.state.0.iter().filter_map(|l|match l{LayerState::MlaLatent(m)=>Some((m.latent.shallow_clone(),m.index.pools.shallow_clone())),_=>None}).collect();
        (c.ids.clone(),c.kda.iter().map(|(a,b)|(a.shallow_clone(),b.shallow_clone())).collect::<Vec<_>>(),
         c.mla.iter().map(|(a,b,d,e)|(a.shallow_clone(),b.shallow_clone(),d.shallow_clone(),e.shallow_clone())).collect::<Vec<_>>(),
         c.draft.snapshot(),c.last.shallow_clone(),rows,s.segs.clone())};
    let plen=ids.len() as i64;let complete=(plen+3)/4;let row=((plen-1)/4).max(0);
    let d=stores[dst].as_mut().unwrap();assert!(d.capacity>=plen,"clone target too small");
    let (mut ki,mut mi)=(0,0);
    for l in &d.state.0 {match l {
        LayerState::Kda(k)=>{k.h.shallow_clone().copy_(&kda[ki].0);k.conv.shallow_clone().copy_(&kda[ki].1);ki+=1;}
        LayerState::MlaLatent(m)=>{
            let (sl,sp)=&rows[mi];
            m.latent.narrow(0,0,plen).copy_(&sl.narrow(0,0,plen));
            m.index.pools.narrow(0,0,complete).copy_(&sp.narrow(0,0,complete));
            let (len,tk,tg,pr)=&mla[mi];
            m.index.pools.narrow(0,row,1).copy_(pr);m.len.shallow_clone().copy_(len);
            m.index.tail_k.shallow_clone().copy_(tk);m.index.tail_gate.shallow_clone().copy_(tg);mi+=1;}
        _=>unreachable!()}}
    d.context=Some(draft.snapshot());d.history=ids.clone();d.segs=segs;segs_truncate(&mut d.segs,plen as usize);
    d.ckpt[which]=Some(Ckpt{ids,kda,mla,draft,last});
}
/// Keep only the persisted segments that lie entirely below `len` rows (the rows past it are rewritten).
fn segs_truncate(segs:&mut Vec<u64>,len:usize) {let s=crate::pcache::seg_tokens();if s>0 {segs.truncate(len/s as usize);} else {segs.clear();}}
/// The store's per-MLA-layer latent and DSA pool storage (rows 0..capacity).
fn store_rows(st:&Store)->(Vec<Tensor>,Vec<Tensor>) {
    st.state.0.iter().filter_map(|l|match l{LayerState::MlaLatent(m)=>Some((m.latent.shallow_clone(),m.index.pools.shallow_clone())),_=>None}).unzip()
}
/// Persist checkpoint `which` of `st` (crate::pcache; both ranks, right after it is taken).
fn pcache_save(st:&mut Store,which:usize) {
    if !crate::pcache::enabled() {return;}
    let (latent,pools)=store_rows(st);
    let c=st.ckpt[which].as_ref().unwrap();
    let id=crate::pcache::save(&c.ids,which,crate::pcache::Parts{kda:&c.kda,mla:&c.mla,draft:&c.draft,last:&c.last,latent:latent.iter().map(|t|t.shallow_clone()).collect(),pools:pools.iter().map(|t|t.shallow_clone()).collect()},st.generation,&mut st.segs);
    // GLM53_PCACHE_SELFTEST=1 (test only): read the entry back right away into scratch tensors and compare bitwise
    // with the live checkpoint and store rows (every row below the prefix; the partial DSA pool row via the checkpoint).
    if let (Some(id),Ok("1"))=(id,std::env::var("GLM53_PCACHE_SELFTEST").as_deref()) {
        let plen=c.ids.len() as i64;let complete=(plen+3)/4;
        let tl:Vec<Tensor>=latent.iter().map(|t|Tensor::empty([plen,t.size()[1]],(t.kind(),t.device()))).collect();
        let tp:Vec<Tensor>=pools.iter().map(|t|Tensor::empty([complete,t.size()[1]],(t.kind(),t.device()))).collect();
        let bits=|a:&Tensor,b:&Tensor|a.size()==b.size()&&a.kind()==b.kind()&&a.contiguous().view_dtype(Kind::Uint8).equal(&b.contiguous().view_dtype(Kind::Uint8));
        let ok=match crate::pcache::restore(id,&c.ids,&tl,&tp,latent[0].device(),false) {
            None=>false,
            Some(r)=>r.ids==c.ids && r.which==which && bits(&r.last,&c.last)
                && r.kda.iter().zip(&c.kda).all(|(a,b)|bits(&a.0,&b.0)&&bits(&a.1,&b.1)) && r.kda.len()==c.kda.len()
                && r.mla.iter().zip(&c.mla).all(|(a,b)|bits(&a.0,&b.0)&&bits(&a.1,&b.1)&&bits(&a.2,&b.2)&&bits(&a.3,&b.3)) && r.mla.len()==c.mla.len()
                && r.draft.len==c.draft.len && r.draft.start==c.draft.start && r.draft.kv().len()==c.draft.kv().len()
                && r.draft.kv().iter().zip(c.draft.kv()).all(|(a,b)|bits(&a.0,&b.0)&&bits(&a.1,&b.1))
                && tl.iter().zip(&latent).all(|(a,b)|bits(a,&b.narrow(0,0,plen)))
                && tp.iter().zip(&pools).all(|(a,b)|bits(&a.narrow(0,0,complete-1),&b.narrow(0,0,complete-1)))};
        eprintln!("[pcache] selftest rank{} {} tokens which {which}: {}",crate::tp::world().rank,plen,if ok {"bitwise equal"} else {"MISMATCH"});
        assert!(ok||std::env::var("GLM53_PCACHE_SELFTEST_SOFT").as_deref()==Ok("1"),"prefix cache self-test failed");
    }
}
struct Seq {id:serde_json::Value,store:usize,prompt:Vec<i64>,consumed:usize,max_new:usize,anchor:i64,decoding:bool,
    generated:Vec<i64>,emitted:usize,deferred:Option<Tensor>,hit:usize,rounds:usize,accepted:usize,
    queued_ms:f64,prefill_ms:f64,started:Instant,decode_started:Option<Instant>,stop_ids:Vec<i64>,cancel:bool,temp:f32,seed:u64,drafted:usize,boundary:usize,
    /// Prefix cache restore from disk: admission to installed (ms, 0 otherwise), and the background load while it runs.
    disk_ms:f64,load:Option<(u64,crate::pcache::Loading)>,
    /// rank0: media items of this prompt and their encoded segments (salted placeholders, crate::vision).
    mm:Option<crate::vision::MmState>,
    /// Copy drafts (GLM53_COPY_DRAFTS): rounds that verified a copied chain, its drafts, and how many were accepted.
    copy:(usize,usize,usize)}
const SERVE_EOS:[i64;3]=[154820,154827,154829];
// Broadcast opcodes (header = 8 f32: [op, a..g]).
const OP_BEAT:i32=0;const OP_EVICT:i32=1;const OP_ADMIT:i32=2;const OP_PREFILL:i32=3;const OP_ROUND:i32=4;const OP_SHUTDOWN:i32=5;const OP_CANCEL:i32=6;const OP_MULTI:i32=7;const OP_LOADED:i32=8;

/// M2 checkpoint library (GLM53_SERVE_CKPT_LIB_MB, e.g. 4096; default 0 = off until qualified): prompt/boundary checkpoints of
/// stores that are evicted or reset are kept, together with the latent rows and complete DSA pool rows below
/// their length, in an LRU keyed by bytes. A later request whose prompt extends a library prefix restores it
/// into a fresh or reset store (action 5) and prefills only the suffix. Mutations happen only inside the
/// replicated state machine (both ranks, same order); entries are addressed by a stable id.
struct LibEntry {id:u64,ids:Vec<i64>,which:usize,ck:Ckpt,latent:Vec<Tensor>,pools:Vec<Tensor>,bytes:i64,used:u64}
#[derive(Default)] struct CkptLib {entries:Vec<LibEntry>,next:u64,clock:u64}
thread_local!{static CKPT_LIB:std::cell::RefCell<CkptLib>=std::cell::RefCell::new(CkptLib::default());}
fn ckpt_lib_limit()->i64 {std::env::var("GLM53_SERVE_CKPT_LIB_MB").ok().and_then(|v|v.parse::<i64>().ok()).unwrap_or(0).max(0)<<20}
fn tensor_bytes(t:&Tensor)->i64 {t.numel() as i64*t.kind().elt_size_in_bytes() as i64}
/// Copy a store's live checkpoints into the library before the store is dropped or reset.
fn lib_save(st:&Store,protect:Option<u64>) {
    let limit=ckpt_lib_limit();if limit==0 {return;}
    CKPT_LIB.with(|l|{let mut l=l.borrow_mut();
        for which in [1usize,0] {
            let Some(c)=st.ckpt[which].as_ref() else {continue};
            l.clock+=1;let clock=l.clock;
            if let Some(e)=l.entries.iter_mut().find(|e|e.ids==c.ids&&e.which==which) {e.used=clock;continue;}
            let plen=c.ids.len() as i64;let complete=(plen+3)/4;
            let (mut latent,mut pools)=(Vec::new(),Vec::new());
            for layer in &st.state.0 {if let LayerState::MlaLatent(m)=layer {
                latent.push(m.latent.narrow(0,0,plen).copy());pools.push(m.index.pools.narrow(0,0,complete).copy());}}
            let ck=Ckpt{ids:c.ids.clone(),kda:c.kda.iter().map(|(a,b)|(a.shallow_clone(),b.shallow_clone())).collect(),
                mla:c.mla.iter().map(|(a,b,d,e)|(a.shallow_clone(),b.shallow_clone(),d.shallow_clone(),e.shallow_clone())).collect(),
                draft:c.draft.snapshot(),last:c.last.shallow_clone()};
            let bytes=latent.iter().chain(&pools).map(tensor_bytes).sum::<i64>()+ck.kda.iter().map(|(a,b)|tensor_bytes(a)+tensor_bytes(b)).sum::<i64>()
                +ck.mla.iter().map(|(a,b,d,e)|tensor_bytes(a)+tensor_bytes(b)+tensor_bytes(d)+tensor_bytes(e)).sum::<i64>()+ck.draft.kv_bytes();
            if bytes>limit {continue;}
            let id=l.next;l.next+=1;
            l.entries.push(LibEntry{id,ids:c.ids.clone(),which,ck,latent,pools,bytes,used:clock});
            while l.entries.iter().map(|e|e.bytes).sum::<i64>()>limit {
                let Some(v)=l.entries.iter().enumerate().filter(|(_,e)|Some(e.id)!=protect&&e.id!=id).min_by_key(|(_,e)|e.used).map(|(i,_)|i) else {break};
                l.entries.remove(v);
            }
        }
    });
}
/// Restore library entry `id` into store `dst` (fresh or reset): bitwise the state the source store had right
/// after that prefill (same data movement as ckpt_clone, from library-owned copies).
fn lib_restore(stores:&mut Vec<Option<Store>>,id:u64,dst:usize)->usize {
    CKPT_LIB.with(|l|{let mut l=l.borrow_mut();l.clock+=1;let clock=l.clock;
        let e=l.entries.iter_mut().find(|e|e.id==id).expect("checkpoint library entry vanished");e.used=clock;
        let plen=e.ids.len() as i64;let complete=(plen+3)/4;let row=((plen-1)/4).max(0);
        let d=stores[dst].as_mut().unwrap();assert!(d.capacity>=plen,"library restore target too small");
        let (mut ki,mut mi)=(0,0);
        for layer in &d.state.0 {match layer {
            LayerState::Kda(k)=>{k.h.shallow_clone().copy_(&e.ck.kda[ki].0);k.conv.shallow_clone().copy_(&e.ck.kda[ki].1);ki+=1;}
            LayerState::MlaLatent(m)=>{
                m.latent.narrow(0,0,plen).copy_(&e.latent[mi]);m.index.pools.narrow(0,0,complete).copy_(&e.pools[mi]);
                let (len,tk,tg,pr)=&e.ck.mla[mi];
                m.index.pools.narrow(0,row,1).copy_(pr);m.len.shallow_clone().copy_(len);
                m.index.tail_k.shallow_clone().copy_(tk);m.index.tail_gate.shallow_clone().copy_(tg);mi+=1;}
            _=>unreachable!()}}
        d.context=Some(e.ck.draft.snapshot());d.history=e.ids.clone();
        d.ckpt[e.which]=Some(Ckpt{ids:e.ids.clone(),kda:e.ck.kda.iter().map(|(a,b)|(a.shallow_clone(),b.shallow_clone())).collect(),
            mla:e.ck.mla.iter().map(|(a,b,c,f)|(a.shallow_clone(),b.shallow_clone(),c.shallow_clone(),f.shallow_clone())).collect(),
            draft:e.ck.draft.snapshot(),last:e.ck.last.shallow_clone()});
        e.which
    })
}
fn serve_admit(eng:&Engine,draft:&Drafter,stores:&mut Vec<Option<Store>>,store:usize,action:i32,capacity:i64,temp:f32,seed:u64,src:i64,which:usize)->(usize,Option<i64>,Option<(u64,crate::pcache::Loading)>) {
    // action 0: reuse (history prefix), 1: reset in place, 2: allocate new at index `store`, 3: restore prompt checkpoint,
    // 4: clone store `src`'s prompt checkpoint into `store` (allocated new when capacity>0, else reset in place),
    // 5: checkpoint library entry, 6: prefix cache entry `src` on disk (crate::pcache)
    let mut anchor=None;let mut load=None;
    // Every action but 0 may overwrite the target store's rows: background writes still reading them finish first.
    if action!=0 {if let Some(Some(st))=stores.get(store) {crate::pcache::fence(st.generation);}}
    // In-memory reuse keeps the matching disk entries recent (replicated LRU).
    if crate::pcache::enabled() {
        let ck=|i:usize,w:usize|stores.get(i).and_then(|s|s.as_ref()).and_then(|s|s.ckpt[w].as_ref()).map(|c|crate::pcache::touch(&c.ids,w));
        match action {0=>{ck(store,1);ck(store,0);},3=>{ck(store,which);},4=>{ck(src as usize,which);},_=>{}}
    }
    match action {
        0=>{}
        1=>{let st=stores[store].as_mut().unwrap();lib_save(st,None);crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();}
        2=>{multi_purge(store);if stores.len()<=store{stores.resize_with(store+1,||None);}
            stores[store]=Some(new_store(eng,draft,capacity));}
        3=>{let st=stores[store].as_mut().unwrap();ckpt_restore(st,which);let n=st.history.len();segs_truncate(&mut st.segs,n);
            // Restoring the boundary checkpoint rewrites rows past it, so the longer prompt checkpoint of the
            // same store is stale from here on (a busy store is still a clone source).
            if which==1 {st.ckpt[0]=None;}
            if which==0 {anchor=Some(crate::sampling::sample_full(&st.ckpt[0].as_ref().unwrap().last,temp,crate::sampling::key(seed,0)));}}
        4=>{
            if capacity>0 {
                multi_purge(store);if stores.len()<=store{stores.resize_with(store+1,||None);}
                stores[store]=Some(new_store(eng,draft,capacity));
            } else {
                let st=stores[store].as_mut().unwrap();lib_save(st,None);crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();
            }
            ckpt_clone(stores,src as usize,which,store);
            if which==0 {anchor=Some(crate::sampling::sample_full(&stores[store].as_ref().unwrap().ckpt[0].as_ref().unwrap().last,temp,crate::sampling::key(seed,0)));}
        }
        5=>{
            // 5: restore checkpoint-library entry `src` into `store` (allocated new when capacity>0, else reset).
            if capacity>0 {
                multi_purge(store);if stores.len()<=store{stores.resize_with(store+1,||None);}
                stores[store]=Some(new_store(eng,draft,capacity));
            } else {
                let st=stores[store].as_mut().unwrap();lib_save(st,Some(src as u64));crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();
            }
            let which=lib_restore(stores,src as u64,store);
            if which==0 {anchor=Some(crate::sampling::sample_full(&stores[store].as_ref().unwrap().ckpt[0].as_ref().unwrap().last,temp,crate::sampling::key(seed,0)));}
        }
        6=>{
            // 6: prefix cache entry `src` (crate::pcache) into `store` (allocated new when capacity>0, else reset). The files
            // are read in the background (P2); OP_LOADED later installs the checkpoint once both ranks agree (serve_loaded).
            if capacity>0 {
                multi_purge(store);if stores.len()<=store{stores.resize_with(store+1,||None);}
                stores[store]=Some(new_store(eng,draft,capacity));
            } else {
                let st=stores[store].as_mut().unwrap();lib_save(st,None);crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();
            }
            let st=stores[store].as_mut().unwrap();
            let (latent,pools)=store_rows(st);
            // GLM53_PCACHE_CANARY=1 (test only): poison the whole target store first (all rows 0xFF = NaN / -1, KDA
            // state NaN), so any state the restore does not rewrite shows up in the output.
            if std::env::var("GLM53_PCACHE_CANARY").as_deref()==Ok("1") {
                for t in latent.iter().chain(&pools) {let _=t.view_dtype(Kind::Uint8).fill_(255);}
                for l in &st.state.0 {match l {
                    LayerState::Kda(k)=>{let _=k.h.shallow_clone().fill_(f64::NAN);let _=k.conv.shallow_clone().fill_(f64::NAN);}
                    LayerState::MlaLatent(m)=>{let _=m.len.shallow_clone().fill_(-1);let _=m.index.tail_k.shallow_clone().fill_(f64::NAN);let _=m.index.tail_gate.shallow_clone().fill_(f64::NAN);}
                    _=>{}}}
            }
            load=Some((src as u64,crate::pcache::start(src as u64,&latent,&pools,eng.w.device,true)));
        }
        _=>panic!("bad admit action"),
    }
    let st=stores[store].as_mut().unwrap();st.busy=true;(st.history.len(),anchor,load)
}
/// Install a restored prefix cache checkpoint into the (reset) store: the rows are already in place; the recurrent
/// state, lengths, tails and partial pool row come from the checkpoint (as lib_restore).
fn install_restored(st:&mut Store,r:crate::pcache::Restored) {
    let plen=r.ids.len() as i64;let row=((plen-1)/4).max(0);
    let (mut ki,mut mi)=(0,0);
    for l in &st.state.0 {match l {
        LayerState::Kda(k)=>{k.h.shallow_clone().copy_(&r.kda[ki].0);k.conv.shallow_clone().copy_(&r.kda[ki].1);ki+=1;}
        LayerState::MlaLatent(m)=>{let (len,tk,tg,pr)=&r.mla[mi];
            m.index.pools.narrow(0,row,1).copy_(pr);m.len.shallow_clone().copy_(len);
            m.index.tail_k.shallow_clone().copy_(tk);m.index.tail_gate.shallow_clone().copy_(tg);mi+=1;}
        _=>unreachable!()}}
    st.context=Some(r.draft.snapshot());st.history=r.ids.clone();st.segs=r.segs;
    st.ckpt[r.which]=Some(Ckpt{ids:r.ids,kda:r.kda,mla:r.mla,draft:r.draft,last:r.last});
}
/// Boundary checkpoint position: before the last <|user|> turn marker, when long enough and past the reused prefix.
fn boundary_of(prompt:&[i64],hit:usize,min:usize)->usize {
    if min==0 {return 0;}
    prompt.iter().rposition(|&t|t==154827).filter(|&b|b>=min&&b>hit&&b<prompt.len()).unwrap_or(0)
}
/// OP_LOADED (both ranks): finish a background prefix cache restore, agree on the outcome, install it or fall back to
/// a cold prefill (entry dropped on both ranks).
fn serve_loaded(eng:&Engine,draft:&Drafter,st:&mut Store,seq:&mut Seq,boundary_min:usize) {
    let (id,ld)=seq.load.take().unwrap();
    let r=ld.finish(&seq.prompt);
    let mut hit=0;
    if crate::pcache::agree(r.is_some(),eng.w.device) {
        let r=r.unwrap();hit=r.ids.len();
        if r.which==0 && hit==seq.prompt.len() {
            seq.anchor=crate::sampling::sample_full(&r.last,seq.temp,crate::sampling::key(seq.seed,0));seq.decoding=true;seq.decode_started=Some(Instant::now());}
        install_restored(st,r);
    } else {
        crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();
        crate::pcache::drop_entry(id);
    }
    seq.hit=hit;seq.consumed=hit;seq.boundary=boundary_of(&seq.prompt,hit,boundary_min);
    seq.disk_ms=seq.started.elapsed().as_secs_f64()*1000.;
}
/// Drafter-training export (GLM53_DRAFT_EXPORT=<dir>, rank0): per prefill chunk, the chunk's token ids, the
/// 20480-wide target features the drafter consumes (BF16), and the target's top-64 log-probabilities per row
/// (row j predicts token begin+j+1) with the full-vocabulary logsumexp. Teacher forcing: send prompt+response as
/// the prompt with max_tokens=1. Prefix reuse skips rows, so run with GLM53_SERVE_BOUNDARY_CKPT=0 and distinct prompts.
fn export_chunk(dir:&str,seq:&Seq,begin:usize,input:&Tensor,features:&Tensor,logits:&Tensor) {
    if crate::tp::world().rank!=0 {return;}
    assert_eq!(logits.size()[0],input.size()[0],"export needs all-row prefill logits");
    let lf=logits.to_kind(Kind::Float);let lse=lf.logsumexp(&[-1i64][..],false);
    let (v,i)=lf.topk(64,-1,true,true);let lp=v-lse.unsqueeze(1);
    let id:String=seq.id.to_string().chars().filter(|c|c.is_ascii_alphanumeric()||*c=='-'||*c=='_').collect();
    let cpu=|t:Tensor|t.to_device(Device::Cpu);
    std::fs::create_dir_all(dir).unwrap();
    // GLM53_DRAFT_EXPORT_NOFEAT=1: distributions only (precision comparisons), no 20480-wide features.
    let features=if std::env::var("GLM53_DRAFT_EXPORT_NOFEAT").as_deref()==Ok("1") {Tensor::zeros([1],(Kind::BFloat16,features.device()))} else {features.shallow_clone()};
    Tensor::save_multi(&[("ids",&cpu(input.shallow_clone())),("features",&cpu(features.to_kind(Kind::BFloat16))),("top_logprobs",&cpu(lp)),
        ("top_ids",&cpu(i.to_kind(Kind::Int))),("logsumexp",&cpu(lse)),("begin",&Tensor::from(begin as i64)),("prompt_len",&Tensor::from(seq.prompt.len() as i64))],
        std::path::Path::new(dir).join(format!("{id}-{begin:08}.pt"))).unwrap();
}
/// One prefill chunk; returns true when the prompt is fully consumed (anchor ready).
fn serve_prefill(eng:&mut Engine,draft:&Drafter,st:&mut Store,seq:&mut Seq,chunk:usize,vision:Option<&crate::vision::Vision>)->bool {
    let t0=Instant::now();let begin=seq.consumed;
    let mut end=(seq.consumed+chunk).min(seq.prompt.len());
    if seq.boundary>seq.consumed && seq.boundary<end {end=seq.boundary;}   // stop at the boundary checkpoint
    let input=Tensor::from_slice(&seq.prompt[seq.consumed..end]).to_device(eng.w.device);
    // Salted media placeholders in this chunk: every rank zeroes those embedding rows, rank0 fills them with the
    // vision rows (encoded now, only the segments this chunk touches) before the embedding allreduce.
    let rows=crate::vision::placeholder_rows(&seq.prompt[begin..end]);
    if !rows.is_empty() {
        let values=seq.mm.as_mut().map(|m|{let v=vision.expect("media prompt without a vision tower");
            let y=m.chunk_rows(v,begin,end).expect("placeholders without encoded media");
            assert_eq!(y.size()[0],rows.len() as i64,"vision rows do not match the placeholders of rows {begin}..{end}");y});
        crate::weights::set_prefill_mm(Some(crate::weights::MmInject{rows:Tensor::from_slice(&rows).to_device(eng.w.device),values}));
    }
    let state=std::mem::replace(&mut st.state,DecodeStates(Vec::new()));
    let export=std::env::var("GLM53_DRAFT_EXPORT").ok().filter(|v|!v.is_empty());
    // Export mode needs every row's logits; the last-row-only head is kept for normal serving.
    let (l,s,f)=if export.is_some() {eng.prefill_record(&input,Some(state),true)} else {eng.prefill_record_last(&input,Some(state),true)};st.state=s;
    let features=Tensor::cat(&f,1);
    draft.append(st.context.as_mut().unwrap(),&features);
    if let Some(dir)=export {export_chunk(&dir,seq,begin,&input,&features,&l);}
    seq.consumed=end;
    let done=seq.consumed==seq.prompt.len();
    if done {let last=l.get(l.size()[0]-1);seq.anchor=crate::sampling::sample_full(&last,seq.temp,crate::sampling::key(seq.seed,0));
        seq.decoding=true;seq.decode_started=Some(Instant::now());st.ckpt[0]=Some(ckpt_save(st,&seq.prompt,&last));pcache_save(st,0);}
    else if seq.consumed==seq.boundary {st.ckpt[1]=Some(ckpt_save(st,&seq.prompt[..seq.boundary],&Tensor::zeros([1],(Kind::Float,eng.w.device))));pcache_save(st,1);}
    else {tch::Cuda::synchronize(0);}
    seq.prefill_ms+=t0.elapsed().as_secs_f64()*1000.;
    if std::env::var("GLM53_SERVE_LOG").as_deref()==Ok("1") {
        tch::Cuda::synchronize(0);
        eprintln!("[serve-prefill] rank{} rows {}..{} of {} chunk_ms {:.1}",crate::tp::world().rank,begin,end,seq.prompt.len(),t0.elapsed().as_secs_f64()*1000.);
    }
    done
}
/// One speculative round (batch-graph-chain with confidence truncation). Returns finished.
/// GLM53_SERVE_CTRL_TCP=1: rank0 -> rank1 control messages ride the doorbell TCP stream (tag 2, 8 x f32 header,
/// u32 count, count x i64 payload) instead of per-op GPU broadcasts (blocking upload + allreduce + readback, which
/// also drained the stream before every round). Stop decisions are computed on both ranks from the replicated stop
/// ids (sent with ADMIT); cancellations travel as OP_CANCEL.
fn ctrl_write(d:&mut std::net::TcpStream,h:&[f32],payload:&[i64]) {
    use std::io::Write;assert_eq!(h.len(),8);
    let mut b=Vec::with_capacity(1+32+4+payload.len()*8);b.push(2u8);
    for v in h {b.extend_from_slice(&v.to_le_bytes());}
    b.extend_from_slice(&(payload.len() as u32).to_le_bytes());for v in payload {b.extend_from_slice(&v.to_le_bytes());}
    d.write_all(&b).expect("control write");
}
fn ctrl_read(d:&mut std::net::TcpStream)->Option<(Vec<f32>,Vec<i64>)> {
    use std::io::Read;let mut tag=[0u8;1];d.read_exact(&mut tag).ok()?;assert_eq!(tag[0],2,"control stream out of sync");
    let mut hb=[0u8;32];d.read_exact(&mut hb).ok()?;let h=(0..8).map(|i|f32::from_le_bytes(hb[i*4..i*4+4].try_into().unwrap())).collect();
    let mut nb=[0u8;4];d.read_exact(&mut nb).ok()?;let n=u32::from_le_bytes(nb) as usize;
    let mut pb=vec![0u8;n*8];d.read_exact(&mut pb).ok()?;
    Some((h,(0..n).map(|i|i64::from_le_bytes(pb[i*8..i*8+8].try_into().unwrap())).collect()))
}
fn serve_round(eng:&mut Engine,draft:&Drafter,st:&mut Store,seq:&mut Seq)->bool {
    let n=seq.max_new;let anchor=seq.anchor;
    let log=std::env::var("GLM53_SERVE_LOG").as_deref()==Ok("1");let t0=Instant::now();
    if seq.generated.len()+1==n || SERVE_EOS.contains(&anchor) {seq.generated.push(anchor);return true;}
    let remaining=n-seq.generated.len();let mut depth=(remaining-1).min(draft_depth());
    if crate::sampling::coupled() {crate::sampling::set_draft_rows(vec![(seq.temp,crate::sampling::draft_keys(seq.seed,seq.generated.len() as u64))]);}
    if crate::sampling::early_enabled() {let g=seq.generated.len() as u64;
        crate::sampling::prefill_rows(&(0..depth as u64+1).map(|i|(seq.temp,crate::sampling::key(seq.seed,g+1+i))).collect::<Vec<_>>());}
    let copied=copy_settings().map_or(Vec::new(),|(m,k)|{
        // Long chains only at exactly k rows past the anchor (one extra graph); otherwise the usual <= depth drafts.
        let long=k>depth && remaining>k && !crate::sampling::early_enabled();
        let mut c=copy_proposal(&seq.prompt,&seq.generated,anchor,m,if long{k}else{k.min(depth)});
        if c.len()>depth && c.len()<k {c.truncate(depth);}
        c});
    let c=copied.is_empty().then(||draft.propose(st.context.as_ref().unwrap(),anchor,&eng.w));
    crate::sampling::clear_draft_rows();
    let path=if let Some(c)=c.as_ref() {
        if let (Some(tau),Some(conf))=(crate::dflash::conf_tau(),c.conf.as_ref()) {depth=crate::spec_policy::chain_depth(conf,depth,tau);}
        c.path[..depth].to_vec()
    } else {depth=copied.len();copied};
    let all=with_anchor(anchor,crate::speculative::chain(&path));
    seq.drafted+=depth;
    let g=seq.generated.len() as u64;
    crate::sampling::set_rows(&(0..all.len() as u64).map(|i|(seq.temp,crate::sampling::key(seq.seed,g+1+i))).collect::<Vec<_>>());
    let base=std::mem::replace(&mut st.state,DecodeStates(Vec::new()));
    if st.chain.is_none(){st.chain=Some(ChainCache::new_on(eng,&base,&all));}
    let verified=st.chain.as_mut().unwrap().verify(eng,&base,anchor,&all,depth+1,&SERVE_EOS);
    drop(base);
    // Diagnostic (W13/W14 offline estimate): rank0 appends this round's drafter candidates as one JSON line.
    if let (Ok(path),Some(c))=(std::env::var("GLM53_DRAFT_TRACE"),c.as_ref()) {if crate::tp::world().rank==0 {
        let f=|t:&Tensor|Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).reshape([-1])).unwrap();
        let ids=Vec::<i64>::try_from(c.ids.to_device(Device::Cpu).reshape([-1])).unwrap();
        // With GLM53_SPEC_ROUTE_TRACE the captured graph keeps its route taps: this round's experts per layer, [rows, 8].
        let routes:Vec<(usize,Vec<i64>)>=st.chain.as_ref().map(|c|c.routes().iter().map(|(l,t)|(*l,Vec::<i64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Int64).reshape([-1])).unwrap())).collect()).unwrap_or_default();
        let line=json!({"g":seq.generated.len(),"temp":seq.temp,"seed":seq.seed,"anchor":anchor,"depth":depth,"accepted":verified.tokens.len()-1,"path":c.path,
            "ids":ids,"ids_shape":c.ids.size(),"unary":f(&c.unary),"edges":f(&c.edges),"edges_shape":c.edges.size(),"conf":c.conf,"rows":all.len(),"routes":routes});
        use std::io::Write;let mut o=std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();writeln!(o,"{line}").unwrap();}}
    seq.rounds+=1;seq.accepted+=verified.tokens.len()-1;
    if c.is_none() {seq.copy.0+=1;seq.copy.1+=depth;seq.copy.2+=verified.tokens.len()-1;}
    seq.generated.extend(verified.tokens);st.state=verified.state;seq.anchor=verified.next;
    let finished=seq.generated.len()>=n || seq.generated.last().map_or(false,|t|SERVE_EOS.contains(t));
    let aux=verified.aux.join();
    if finished {seq.deferred=Some(aux);} else {draft.append(st.context.as_mut().unwrap(),&aux);}
    if log {tch::Cuda::synchronize(0);eprintln!("[serve-round] rows {} accepted {} round_ms {:.1}",all.len(),seq.accepted,t0.elapsed().as_secs_f64()*1000.);}
    finished
}
/// One batched round over several decoding sequences (GLM53_SERVE_BATCH=1): each proposes its chain
/// (depth <= GLM53_SERVE_BATCH_DEPTH, confidence truncation applies), all chains are verified in one
/// multi-sequence forward (one weight read for MHC/dense/MoE/head), then every sequence selects and
/// commits into its own storage. Returns the per-slot finished flags (before stop hooks).
/// Captured multi-sequence verifier (GLM53_SERVE_BATCH=1). Bases alias each store's own state storage
/// (fixed for the store's lifetime); commits go into that storage in place, so replays need no restore.
struct MultiGraph {graph:crate::tp::graph::Owned,key:Vec<(usize,u64,usize,TreeSelection)>,input:Tensor,preds:Tensor,
    states:Vec<VerifierStates>,features:Tensor,bases:Vec<DecodeStates>,used:u64}
thread_local!{
    static MULTI_GRAPHS:std::cell::RefCell<Vec<MultiGraph>>=std::cell::RefCell::new(Vec::new());
    static GENERATION:std::cell::Cell<u64>=std::cell::Cell::new(0);
    static MULTI_CLOCK:std::cell::Cell<u64>=std::cell::Cell::new(0);
}
fn next_generation()->u64 {GENERATION.with(|g|{g.set(g.get()+1);g.get()})}
/// Drop cached multi graphs that reference store `i` (eviction / reallocation / reset of that index).
fn multi_purge(i:usize) {MULTI_GRAPHS.with(|m|m.borrow_mut().retain(|g|g.key.iter().all(|k|k.0!=i)));}
fn batch_rows_allowed()->Vec<usize> {
    std::env::var("GLM53_SERVE_BATCH_ROWS").ok().map(|v|v.split(',').filter_map(|x|x.trim().parse::<usize>().ok()).collect::<Vec<_>>())
        .filter(|v:&Vec<usize>|!v.is_empty()).unwrap_or_else(||vec![2,4])
}
// Off by default: after KDA/MLA row batching the multi forward is GPU-bound (graph arm measured slower,
// 53.6/57.9/59.9 vs eager 57.5/59.2/65.9 tok/s, code x4), and LRU eviction of captured multi graphs
// reproducibly leads to an illegal address (not with an unbounded cache; cause not yet isolated).
fn multi_graphs_enabled()->bool {std::env::var("GLM53_SERVE_MULTI_GRAPH").as_deref()==Ok("1")}

fn serve_round_multi(eng:&mut Engine,draft:&Drafter,stores:&mut Vec<Option<Store>>,active:&mut Vec<Option<Seq>>,slots:&[usize])->Vec<bool> {
    // Draft depth cap per sequence. Each verified row costs ~5 ms in the batched forward while the per-round
    // base is shared, so small batches favour deep chains and 4+ sequences shallow ones (code x2: depth 7
    // 54.6 vs 3 51.1 tok/s; code x4: depth 3 61.0 vs 7 57.8). GLM53_SERVE_BATCH_DEPTH forces a fixed cap.
    // GLM53_SERVE_BATCH_CUMCONF=θ (draft side, L2; product stop rule): each chain stops where the running
    // product of its drafts' confidences drops below θ, so the cap defaults to 7 for every batch size.
    let cumconf=std::env::var("GLM53_SERVE_BATCH_CUMCONF").ok().map(|v|v.parse::<f64>().unwrap());
    let cap=std::env::var("GLM53_SERVE_BATCH_DEPTH").ok().and_then(|v|v.parse::<usize>().ok())
        .unwrap_or(if cumconf.is_some(){7}else{match slots.len() {0..=2=>7,3=>5,_=>3}}).clamp(1,7);
    let log=std::env::var("GLM53_SERVE_LOG").as_deref()==Ok("1");let t0=Instant::now();
    let lap=|t:&Instant|{if log{tch::Cuda::synchronize(0);}t.elapsed().as_secs_f64()*1000.};
    let mut finished=vec![false;slots.len()];
    let graphs=multi_graphs_enabled();let allowed=batch_rows_allowed();
    // (slot index in `slots`, nodes, depth, selection, store index)
    let mut plan:Vec<(usize,Vec<Node>,usize,crate::dsa::TreeSelection,usize)>=Vec::new();
    // C2 (GLM53_SERVE_DRAFT_BATCH=1): proposals of up to four sequences per drafter forward.
    let mut proposals:Vec<Option<crate::dflash::Candidates>>=(0..slots.len()).map(|_|None).collect();
    let tm=std::env::var("GLM53_ROUND_TIMING").as_deref()==Ok("1");
    let tmark=|label:&str,t:&mut Instant|{if tm{tch::Cuda::synchronize(0);eprintln!("[round-timing] {label} {:.2} ms",t.elapsed().as_secs_f64()*1000.);*t=Instant::now();}};
    let mut tt=Instant::now();
    if std::env::var("GLM53_SERVE_DRAFT_BATCH").as_deref()==Ok("1") {
        let need:Vec<usize>=(0..slots.len()).filter(|&k|{let seq=active[slots[k]].as_ref().unwrap();
            !(seq.generated.len()+1==seq.max_new || SERVE_EOS.contains(&seq.anchor))}).collect();
        for group in need.chunks(4) {
            if group.len()<2 {continue;}
            let ctxs:Vec<&crate::dflash::Context>=group.iter().map(|&k|stores[active[slots[k]].as_ref().unwrap().store].as_ref().unwrap().context.as_ref().unwrap()).collect();
            let anchors:Vec<i64>=group.iter().map(|&k|active[slots[k]].as_ref().unwrap().anchor).collect();
            if crate::sampling::coupled() {crate::sampling::set_draft_rows(group.iter().map(|&k|{let q=active[slots[k]].as_ref().unwrap();
                (q.temp,crate::sampling::draft_keys(q.seed,q.generated.len() as u64))}).collect());}
            if let Some(cs)=draft.propose_many(&ctxs,&anchors,&eng.w) {for (&k,c) in group.iter().zip(cs) {proposals[k]=Some(c);}}
            crate::sampling::clear_draft_rows();
        }
    }
    tmark("draft_batch",&mut tt);
    settle_side_commit();   // previous round's side-stream commits (GLM53_MULTI_COMMIT_SIDE) before reading target state
    for (k,&slot) in slots.iter().enumerate() {
        let seq=active[slot].as_mut().unwrap();let n=seq.max_new;let anchor=seq.anchor;
        if seq.generated.len()+1==n || SERVE_EOS.contains(&anchor) {seq.generated.push(anchor);finished[k]=true;continue;}
        let st=stores[seq.store].as_ref().unwrap();
        let remaining=n-seq.generated.len();let mut depth=(remaining-1).min(cap);
        let c=match proposals[k].take() {Some(c)=>c,None=>{
            if crate::sampling::coupled() {crate::sampling::set_draft_rows(vec![(seq.temp,crate::sampling::draft_keys(seq.seed,seq.generated.len() as u64))]);}
            let c=draft.propose(st.context.as_ref().unwrap(),anchor,&eng.w);crate::sampling::clear_draft_rows();c}};
        if let (Some(th),Some(conf))=(cumconf,c.conf.as_ref()) {let (mut d,mut p)=(0,1f64);while d<depth {p*=conf[d] as f64;if p<th {break;}d+=1;}depth=d.max(1).min(depth);}
        else if let (Some(tau),Some(conf))=(crate::dflash::conf_tau(),c.conf.as_ref()) {let mut d=0;while d<depth && (conf[d] as f64)>=tau {d+=1;}depth=d.max(1).min(depth);}
        // Quantize the chain length to the allowed row counts (fewer distinct graphs); round up.
        if graphs {let max_depth=(remaining-1).min(cap);if let Some(r)=allowed.iter().copied().filter(|&r|r>=depth+1&&r-1<=max_depth).min(){depth=r-1;}}
        let nodes=with_anchor(anchor,crate::speculative::chain(&c.path[..depth]));
        seq.drafted+=depth;
        let selection=ChainGraph::selection(&st.state,&nodes);
        plan.push((k,nodes,depth,selection,seq.store));
    }
    tmark("plan",&mut tt);
    if plan.is_empty() {return finished;}
    let mut ids=Vec::new();let mut segs=Vec::new();let mut parents=Vec::new();let mut selections=Vec::new();
    let mut noise=Vec::new();
    for (k,nodes,_,sel,_) in &plan {segs.push((ids.len(),nodes.len()));ids.extend(nodes.iter().map(|n|n.token));
        let seq=active[slots[*k]].as_ref().unwrap();let g=seq.generated.len() as u64;
        noise.extend((0..nodes.len() as u64).map(|i|(seq.temp,crate::sampling::key(seq.seed,g+1+i))));
        parents.push(nodes.iter().map(|n|n.parent).collect::<Vec<_>>());selections.push(*sel);}
    // Each sequence's base is an alias of its store's own state storage (distinct stores).
    let mut bases:Vec<DecodeStates>=plan.iter().map(|p|alias_states(&stores[p.4].as_ref().unwrap().state).expect("serving state aliases")).collect();
    let key:Vec<(usize,u64,usize,TreeSelection)>=plan.iter().map(|p|(p.4,stores[p.4].as_ref().unwrap().generation,p.1.len(),p.3)).collect();
    let propose_ms=lap(&t0);let t1=Instant::now();
    crate::sampling::set_rows(&noise);
    let input=Tensor::from_slice(&ids).to_device(eng.w.device);
    let mut captured=false;
    let (preds,owned_states,features,graph_index)=if graphs {
        let clock=MULTI_CLOCK.with(|c|{c.set(c.get()+1);c.get()});
        let hit=MULTI_GRAPHS.with(|m|m.borrow().iter().position(|g|g.key==key));
        let index=match hit {Some(i)=>i,None=>{
            captured=true;
            let limit=std::env::var("GLM53_SERVE_MULTI_GRAPHS").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(16).max(1);
            MULTI_GRAPHS.with(|m|{let mut m=m.borrow_mut();while m.len()>=limit {let lru=(0..m.len()).min_by_key(|&i|m[i].used).unwrap();m.remove(lru);}});
            let refs:Vec<&DecodeStates>=bases.iter().collect();
            let _=eng.tree_forward_multi(&input,&refs,&segs,&parents,&selections,true,true,true);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();
            let (p,s,f)=eng.tree_forward_multi(&input,&refs,&segs,&parents,&selections,true,true,false);
            let f=Tensor::cat(&f,1);crate::tp::graph::end().unwrap();
            let g=MultiGraph{graph:crate::tp::graph::Owned::take(),key:key.clone(),input:input.shallow_clone(),preds:p,states:s,features:f,
                bases:bases.iter().map(|b|alias_states(b).unwrap()).collect(),used:clock};
            MULTI_GRAPHS.with(|m|{let mut m=m.borrow_mut();m.push(g);m.len()-1})
        }};
        MULTI_GRAPHS.with(|m|{let mut m=m.borrow_mut();let g=&mut m[index];g.used=clock;
            for (a,b) in g.bases.iter().zip(&bases) {assert!(same_storage(a,b),"multi graph base storage changed");}
            g.input.copy_(&input);g.graph.replay();
            (g.preds.shallow_clone(),None,g.features.shallow_clone(),Some(index))})
    } else {
        // GLM53_MULTI_HOST_ROOM=1 (L0): the latent capacity check from host-known lengths (prompt + generated tokens,
        // an upper bound of the committed length) once per round, instead of one device->host read of each base length
        // per latent layer and sequence (each a stream synchronize that drains the GPU queue). Rounds with an
        // AllVisible selection (short contexts) keep the exact device check.
        let check_room=if std::env::var("GLM53_MULTI_HOST_ROOM").as_deref()==Ok("1") && selections.iter().all(|s|*s!=TreeSelection::AllVisible) {
            for (k,_,depth,_,store) in &plan {
                let seq=active[slots[*k]].as_ref().unwrap();let len=(seq.prompt.len()+seq.generated.len()) as i64;
                let cap=stores[*store].as_ref().unwrap().state.0.iter().find_map(|l|match l {crate::forward::LayerState::MlaLatent(s)=>Some(s.capacity),_=>None}).expect("latent layer");
                assert!(len+*depth as i64+1<=cap,"latent capacity exceeded (host check: len {len} depth {depth} capacity {cap})");
            }
            false
        } else {true};
        let (p,s,f)={let refs:Vec<&DecodeStates>=bases.iter().collect();eng.tree_forward_multi(&input,&refs,&segs,&parents,&selections,true,true,check_room)};
        (p,Some(s),Tensor::cat(&f,1),None)
    };
    let enqueue_ms=t1.elapsed().as_secs_f64()*1000.;
    let forward_ms=lap(&t1);let t2=Instant::now();
    // One device->host copy of all predictions (select_with's own copy is then a no-op per sequence).
    tmark("forward",&mut tt);
    let preds=preds.to_device(Device::Cpu);
    tmark("preds_d2h",&mut tt);
    // C2: the drafter appends of all continuing sequences in one forward (after selection).
    let batch_append=std::env::var("GLM53_SERVE_DRAFT_BATCH").as_deref()==Ok("1");
    let mut pending:Vec<(usize,Tensor)>=Vec::new();
    let side_multi=multi_commit_side() && !crate::tp::graph::capturing();
    let mut select_all=|all_states:&[VerifierStates]| {
    for (pi,((k,nodes,depth,_,store),base)) in plan.iter().zip(bases.drain(..)).enumerate() {
        let (k,depth,store)=(*k,*depth,*store);let states=&all_states[pi];
        let slot=slots[k];let seq=active[slot].as_mut().unwrap();let anchor=seq.anchor;
        let (first,len)=segs[pi];
        let p=preds.narrow(0,first as i64,len as i64);let f=features.narrow(0,first as i64,len as i64);
        let commit=|parent:Option<usize>|->DecodeStates {if let Some(node)=parent {
            if side_multi {commit_into_base_now(states,&base,node);} else {commit_into_base(states,&base,node);}}alias_states(&base).unwrap()};
        let verified=select_with(&base,anchor,nodes,depth+1,&SERVE_EOS,&p,states,&f,Some(&commit));
        seq.rounds+=1;seq.accepted+=verified.tokens.len()-1;
        seq.generated.extend(verified.tokens);seq.anchor=verified.next;
        let st=stores[store].as_mut().unwrap();st.state=verified.state;
        let fin=seq.generated.len()>=seq.max_new || seq.generated.last().map_or(false,|t|SERVE_EOS.contains(t));
        let aux=verified.aux.join();
        if fin {seq.deferred=Some(aux);} else if batch_append {pending.push((store,aux));} else {draft.append(st.context.as_mut().unwrap(),&aux);}
        finished[k]=fin;
    }
    };
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;}
    if side_multi {settle_side_commit();assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);}
    match (owned_states,graph_index) {
        (Some(s),_)=>{select_all(&s);if side_multi {SIDE_KEEP.with(|k|k.borrow_mut().extend(s));}},
        (None,Some(i))=>MULTI_GRAPHS.with(|m|select_all(&m.borrow()[i].states)),
        _=>unreachable!(),
    }
    if side_multi {assert_eq!(unsafe{rs_stream_set(-1)},0);SIDE_COMMIT.with(|c|c.set(true));}
    tmark("select_commit",&mut tt);
    if !pending.is_empty() {
        // Distinct stores per sequence: take each context out, append together, put back.
        let mut taken:Vec<(usize,crate::dflash::Context,Tensor)>=pending.into_iter().map(|(store,aux)|{
            let c=stores[store].as_mut().unwrap().context.take().unwrap();(store,c,aux)}).collect();
        {let mut items:Vec<(&mut crate::dflash::Context,Tensor)>=taken.iter_mut().map(|(_,c,a)|(c,a.shallow_clone())).collect();draft.append_many(&mut items);}
        for (store,c,_) in taken {stores[store].as_mut().unwrap().context=Some(c);}
    }
    tmark("draft_append",&mut tt);
    if log {eprintln!("[serve-multi] seqs {} rows {} graph {} captured {} propose_ms {:.1} enqueue_ms {:.1} forward_ms {:.1} select_ms {:.1}",segs.len(),ids.len(),graphs,captured,propose_ms,enqueue_ms,forward_ms,lap(&t2));}
    finished
}
/// Bring the drafter context level with the target state and record the reusable history.
fn serve_finish(draft:&Drafter,st:&mut Store,seq:&mut Seq,clock:u64) {
    settle_side_commit();
    if let Ok(path)=std::env::var("GLM53_DRAFT_TRACE") {if crate::tp::world().rank==0 {
        use std::io::Write;let mut o=std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        writeln!(o,"{}",json!({"final":seq.generated,"prompt_len":seq.prompt.len()})).unwrap();}}
    if let Some(aux)=seq.deferred.take() {draft.append(st.context.as_mut().unwrap(),&aux);}
    let len=crate::forward::states_len(&st.state) as usize;
    let mut all=seq.prompt.clone();all.extend(&seq.generated);
    let ctx=st.context.as_ref().map_or(-1,|c|c.len);
    st.history=if len<=all.len() && ctx==len as i64 {all[..len].to_vec()}else{Vec::new()};
    st.busy=false;st.used=clock;
}

/// Online TP2 serving with up to GLM53_SERVE_MAX_SEQS concurrent sequences (default 4) sharing a
/// KV budget of GLM53_SERVE_KV_TOKENS tokens (default 1048576). Rank0 accepts newline-delimited JSON
/// on a Unix socket from one front-end connection and multiplexes by request id:
///   request  {"id":..,"prompt_ids":[..],"max_new":N,"stop_token_ids":[..]}   cancel {"cancel":id}
///   shutdown {"shutdown":true}
/// Replies: {"id","queued":true} once admitted to the queue, {"id","delta":[ids]} after every round,
/// then {"id","done":true,"token_ids",...}. Sequences advance round-robin (one prefill chunk or one
/// verification round per turn); every scheduling decision is broadcast to rank1, which mirrors it.
/// Storage per sequence is sized to prompt+max_new (rounded to GLM53_SERVE_KV_GRANULE) and kept after
/// completion for exact-prefix reuse until the budget needs it. Greedy decoding only.
pub fn serve(model:&Path,draft_dir:&Path,socket:&Path) {
    use std::io::{BufRead,Write};
    use std::os::unix::net::UnixStream;
    crate::host_memory::stage("serve start");
    // Warm-up telemetry: caching allocator, MemAvailable, owned graphs, cuBLAS workspace per (handle, stream).
    let mem_note=||{extern "C"{fn rs_cuda_memory(a:*mut i64,r:*mut i64);fn rs_graph_count()->i64;fn rs_cublas_workspace_bytes()->i64;}
        tch::Cuda::synchronize(0);let (mut a,mut r)=(0i64,0i64);unsafe{rs_cuda_memory(&mut a,&mut r)};let g=(1u64<<30) as f64;
        format!("allocated {:.2} GiB reserved {:.2} GiB avail {:.2} GiB graphs {} cublas workspace {:.1} MiB",a as f64/g,r as f64/g,
            crate::kv_pool::meminfo().1 as f64/g,unsafe{rs_graph_count()},unsafe{rs_cublas_workspace_bytes()} as f64/(1u64<<20) as f64)};
    // Expert disk reads and device uploads start now, overlapping the process group and the non-expert load.
    let expert_reader={let c=crate::config::load(&model.join("config.json")).unwrap();
        let weights_read=crate::weights::start_early_prefetch(model,c.num_hidden_layers,draft_dir);
        crate::expert_load::ExpertReader::start(model,c.num_hidden_layers,c.n_routed_experts,crate::tp::env_tp(),Device::Cuda(0),weights_read)};
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    crate::host_memory::stage("tp init");
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    std::env::set_var("GLM53_PREFILL_BATCH","1");
    let env_num=|k:&str,d:i64|std::env::var(k).ok().and_then(|v|v.parse::<i64>().ok()).unwrap_or(d);
    let max_seqs=env_num("GLM53_SERVE_MAX_SEQS",4).clamp(1,8) as usize;
    let mut budget=env_num("GLM53_SERVE_KV_TOKENS",1<<20);let granule=env_num("GLM53_SERVE_KV_GRANULE",16384).max(1024);
    let chunk=env_num("GLM53_PREFILL_CHUNK",2048) as usize;
    let chunk_solo=env_num("GLM53_SERVE_CHUNK_SOLO",0) as usize;
    // Live stores (each holds KDA state, a drafter slot and its own verifier graphs besides its KV).
    let max_stores=env_num("GLM53_SERVE_MAX_STORES",8).max(max_seqs as i64) as usize;
    let clone_on=std::env::var("GLM53_SERVE_CLONE").as_deref()!=Ok("0");
    let clone_min=env_num("GLM53_SERVE_CLONE_MIN",64).max(1) as usize;
    // Boundary checkpoint before the last user turn (GLM53_SERVE_BOUNDARY_CKPT = min boundary length in
    // tokens; 0 disables). It costs one extra prefill chunk split, so only long shared prefixes qualify.
    let boundary_min=env_num("GLM53_SERVE_BOUNDARY_CKPT",1024).max(0) as usize;
    // Test-only (GLM53_SERVE_CKPT_REUSE=0): never restore/clone prompt checkpoints (boundary splits still
    // happen), so a run gives the cold reference with the very same chunk boundaries.
    let ckpt_reuse=std::env::var("GLM53_SERVE_CKPT_REUSE").as_deref()!=Ok("0");
    assert!(crate::mla_latent::capacity()>=budget,"GLM53_MAX_CONTEXT must be >= GLM53_SERVE_KV_TOKENS");
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    // Return the load's freed temporaries (FP32 widening copies, FP8/C12 sources) to the driver before the expert
    // arenas grow, so they do not add to the load's peak (twice: after the target weights and after the drafter).
    extern "C"{fn rs_empty_cache();}
    if crate::weights::fast_load_enabled() {unsafe{rs_empty_cache();}}
    crate::host_memory::stage_mem("model weights");
    let draft=Drafter::load_target(draft_dir,&w);crate::weights::finish_early_prefetch();
    // The non-expert and drafter bytes are on the device now: drop their page cache before the expert arenas grow.
    crate::host_memory::release_file_cache(&[model,draft_dir]);
    if crate::weights::fast_load_enabled() {unsafe{rs_empty_cache();}}
    crate::host_memory::stage_mem("drafter");
    // Vision tower (BF16 as stored, rank0 only: it encodes, the embedding allreduce distributes the rows). Loaded
    // before the KV pool is sized, so the pool accounts for it.
    let vision=(tp.rank==0 && crate::vision::enabled()).then(||{let v=crate::vision::Vision::load(model,dev);crate::host_memory::stage_mem("vision");v});
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all_with(expert_reader,cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;crate::host_memory::stage("experts");
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};crate::host_memory::stage_mem("expert pool");
    // The load's freed temporaries (FP32 widening copies, FP8/C12 sources) stay reserved in the caching allocator;
    // with the expert arenas allocated during the load they are not reused by the experts. Return them.
    if crate::weights::fast_load_enabled() {unsafe{rs_empty_cache();}crate::host_memory::stage_mem("load cache released");}
    // GLM53_KV_POOL: size the KV pool from memory (GLM53_MEM_UTIL of MemTotal as the whole machine's hard use, minus
    // what is in use now and GLM53_MEM_RESERVE_GIB for what serving adds outside the pool: every store's verifier
    // graphs, KDA state and drafter window, the prefill workspaces), allocate it now, and let it set the KV budget.
    // Both ranks take the smaller size so they carve identical ranges.
    if crate::kv_pool::enabled() {
        let layers=eng.kv_pool_layers();let per=crate::kv_pool::KvPool::bytes_per_token(&layers);
        let util=std::env::var("GLM53_MEM_UTIL").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(0.90);
        assert!((0.5..=0.98).contains(&util),"GLM53_MEM_UTIL must be within 0.5..0.98");
        let reserve=(std::env::var("GLM53_MEM_RESERVE_GIB").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(7.0)*(1u64<<30) as f64) as i64;
        let (total,avail)=crate::kv_pool::meminfo();
        let bytes=(util*total as f64) as i64-(total-avail)-reserve;
        let cap_rows=budget.min(crate::mla_latent::capacity())/granule*granule;
        let mine=((bytes.max(0)/per)/granule*granule).min(cap_rows);
        let t=Tensor::zeros([2],(Kind::Float,dev));let _=t.get(tp.rank as i64).fill_((mine/granule) as f64);crate::tp::allreduce(&t);
        let units=Vec::<f32>::try_from(t.to_device(Device::Cpu)).unwrap();
        let rows=units.iter().fold(f32::MAX,|a,&b|a.min(b)) as i64*granule;
        let need=max_seqs as i64*granule.max(chunk as i64+128);
        assert!(rows>=need,"KV pool: GLM53_MEM_UTIL {util} leaves {} tokens on the smaller rank ({:?} granules), below the {need} the warm-up stores need; lower GLM53_MEM_RESERVE_GIB or raise GLM53_MEM_UTIL",rows,units);
        let pool=crate::kv_pool::KvPool::new(dev,&layers,rows,granule);crate::kv_pool::KvPool::install(pool);budget=rows;
        eprintln!("[kv-pool] rank{} util {util:.2}: MemTotal {:.1} GiB, in use {:.1} GiB, reserve {:.1} GiB -> {} tokens ({:.2} GiB, {per} B/token, {} MLA layers; ranks {:?} granules of {granule})",
            tp.rank,total as f64/(1u64<<30) as f64,(total-avail) as f64/(1u64<<30) as f64,reserve as f64/(1u64<<30) as f64,rows,(rows*per) as f64/(1u64<<30) as f64,layers.len(),units);
        crate::host_memory::stage_mem("kv pool");
    }
    let rank=tp.rank;
    let bcast=move |values:Vec<f32>,len:usize|->Vec<f32> {
        // rank0 already holds the values: upload without a stream drain and do not read its own values back
        // (GLM53_ASYNC_UPLOAD=0: old blocking upload + readback). rank1 must read the result.
        if rank==0 && std::env::var("GLM53_ASYNC_UPLOAD").as_deref()==Ok("1") {
            assert_eq!(values.len(),len);let t=Tensor::empty([len as i64],(Kind::Float,dev));crate::tp::upload(&t,&values);
            crate::tp::allreduce(&t);return values;
        }
        let t=if rank==0{assert_eq!(values.len(),len);Tensor::from_slice(&values).to_device(dev)}else{Tensor::zeros([len as i64],(Kind::Float,dev))};
        crate::tp::allreduce(&t);Vec::<f32>::try_from(t.to_device(Device::Cpu)).unwrap()};
    let mut stores:Vec<Option<Store>>=Vec::new();let mut active:Vec<Option<Seq>>=(0..max_seqs).map(|_|None).collect();
    let mut clock:u64=0;
    // ---- shared state machine (both ranks)
    let apply=|op:&[f32],payload:Vec<i64>,eng:&mut Engine,stores:&mut Vec<Option<Store>>,active:&mut Vec<Option<Seq>>,clock:&mut u64,
               stop_flag:&mut dyn FnMut(&Seq,&[i64])->bool| -> Option<(usize,bool)> {
        *clock+=1;
        if op[0] as i32!=OP_ROUND {settle_side_commit();}
        match op[0] as i32 {
            OP_EVICT=>{let i=op[1] as usize;multi_purge(i);if let Some(st)=stores[i].as_ref(){lib_save(st,None);crate::pcache::fence(st.generation);}stores[i]=None;None}
            OP_ADMIT=>{
                let (slot,store,action,cap,max_new,qms)=(op[1] as usize,op[2] as usize,op[3] as i32,op[4] as i64,op[5] as usize,op[6] as f64);
                let mut payload=payload;let which=payload.pop().unwrap() as usize;let src=payload.pop().unwrap();let seed=payload.pop().unwrap() as u64;let temp=payload.pop().unwrap() as f32/1e4;
                let n_stop=payload.pop().unwrap() as usize;let stop_ids=payload.split_off(payload.len()-n_stop);
                let (hit,anchor,load)=serve_admit(eng,&draft,stores,store,action,cap,temp,seed,src,which);
                // Boundary checkpoint: prefix up to (excluding) the last <|user|> turn marker, so later
                // conversations sharing the system prompt/history can restore or clone it (both ranks compute it).
                let boundary=boundary_of(&payload,hit,boundary_min);
                // A restored prompt checkpoint carries the next token only for an EXACT prompt match. When the
                // checkpoint is a strict prefix (agent tool rounds, multi-turn with re-rendered history), the
                // remaining suffix must be prefilled from `hit`; starting to decode from the checkpoint anchor
                // silently drops the suffix (the model continues the older prompt).
                let anchor=anchor.filter(|_|hit==payload.len());
                active[slot]=Some(Seq{id:serde_json::Value::Null,store,prompt:payload,consumed:hit,max_new,anchor:anchor.unwrap_or(0),decoding:anchor.is_some(),generated:Vec::new(),emitted:0,
                    deferred:None,hit,rounds:0,accepted:0,queued_ms:qms,prefill_ms:0.,started:Instant::now(),decode_started:if anchor.is_some(){Some(Instant::now())}else{None},stop_ids,cancel:false,temp,seed,drafted:0,boundary,disk_ms:0.,load,mm:None,copy:(0,0,0)});
                None}
            OP_PREFILL=>{let slot=op[1] as usize;let seq=active[slot].as_mut().unwrap();let st=stores[seq.store].as_mut().unwrap();
                // op[2]: chunk chosen by rank0 for this unit (0 = default), so both ranks split identically.
                let c=if op[2]>0. {op[2] as usize} else {chunk};
                serve_prefill(eng,&draft,st,seq,c,vision.as_ref());None}
            OP_ROUND=>{let slot=op[1] as usize;let seq=active[slot].as_mut().unwrap();let st=stores[seq.store].as_mut().unwrap();
                let mut finished=serve_round(eng,&draft,st,seq);
                let delta=seq.generated[seq.emitted..].to_vec();seq.emitted=seq.generated.len();
                if stop_flag(seq,&delta) && !finished {finished=true;}
                if finished {let c=*clock;serve_finish(&draft,st,seq,c);}
                Some((slot,finished))}
            OP_CANCEL=>{let slot=op[1] as usize;let seq=active[slot].as_mut().unwrap();let st=stores[seq.store].as_mut().unwrap();
                // A background restore still writing into the store: let it stop, then start the store over (both ranks).
                if let Some((_,ld))=seq.load.take() {ld.abandon();crate::forward::reset_states(&st.state);st.context=Some(draft.empty_context());st.history.clear();st.ckpt=[None,None];st.segs.clear();}
                let c=*clock;serve_finish(&draft,st,seq,c);Some((slot,true))}
            OP_LOADED=>{let slot=op[1] as usize;let seq=active[slot].as_mut().unwrap();let st=stores[seq.store].as_mut().unwrap();
                serve_loaded(eng,&draft,st,seq,boundary_min);None}
            _=>None,
        }
    };
    let batch=std::env::var("GLM53_SERVE_BATCH").as_deref()==Ok("1");
    // Batched round over `slots` (both ranks). Returns (slot, finished) for every slot in the batch.
    let apply_multi=|slots:&[usize],eng:&mut Engine,stores:&mut Vec<Option<Store>>,active:&mut Vec<Option<Seq>>,clock:&mut u64,
               stop_flag:&mut dyn FnMut(&Seq,&[i64])->bool| -> Vec<(usize,bool,Vec<i64>)> {
        *clock+=1;
        let fin=serve_round_multi(eng,&draft,stores,active,slots);
        let mut out=Vec::new();
        for (k,&slot) in slots.iter().enumerate() {
            let seq=active[slot].as_mut().unwrap();
            let delta=seq.generated[seq.emitted..].to_vec();seq.emitted=seq.generated.len();
            let mut finished=fin[k];
            if stop_flag(seq,&delta) && !finished {finished=true;}
            if finished {let c=*clock;let st=stores[seq.store].as_mut().unwrap();serve_finish(&draft,st,seq,c);}
            out.push((slot,finished,delta));
        }
        out
    };
    if std::env::var("GLM53_SERVE_SAMPLING").as_deref()!=Ok("0") {
        let (start,width)=match eng.w.vocab_shard {Some((start,_)) if crate::head_select::enabled()=>(start,eng.w.lm_head.size()[0]),Some((_,total))=>(0,total),None=>(0,eng.w.lm_head.size()[0])};
        crate::sampling::enable(start,width,dev);
    }
    // Warm-up (both ranks, identical op sequence), before serving:
    //  - store 0: a short prompt, then (M3) a prompt of chunk+64 tokens so the sequence-parallel prefill
    //    path (NCCL reduce-scatter/all-gather, large-row MHC, chunk-sized GEMM algorithms) and the partial
    //    tail chunk are initialized at start instead of on the first real long request;
    //  - (M3) stores 1..max_seqs-1 prebuilt and decoded briefly so each has its common chain-depth graphs
    //    captured: concurrent requests 2..N no longer capture on first use.
    // GLM53_SERVE_PREWARM=0 keeps only the original single short warm-up.
    {
        let prewarm=std::env::var("GLM53_SERVE_PREWARM").as_deref()!=Ok("0");
        let t0=Instant::now();
        let short:Vec<i64>=(0..96).map(|i|(1000+i*37) as i64).collect();
        let long:Vec<i64>=(0..(chunk as i64+64)).map(|i|(1000+(i*37)%150000) as i64).collect();
        let cap=granule.max(chunk as i64+64+64);
        let mut runs:Vec<(usize,f32,&Vec<i64>)>=vec![(0,2.,&short),(0,1.,&short)];
        if prewarm {
            runs[1]=(0,1.,&long);
            for st in 1..max_seqs {runs.push((st,2.,&short));}
        }
        // GLM53_SERVE_PREWARM_NEW: tokens decoded by every warm-up run after the first (default 4; was 48). The first
        // run initializes the decode path; the graph prewarm below captures every store's chain/append graphs itself.
        // bench/load1 warmtest: first, 4-way concurrent and steady requests after a restart unchanged with 4 (~3 s saved).
        let prewarm_new=env_num("GLM53_SERVE_PREWARM_NEW",4).clamp(1,48) as f32;
        for (i,(store,action,prompt)) in runs.into_iter().enumerate() {
            let tr=Instant::now();
            // ADMIT payload layout: prompt ids, stop ids, stop-id count, temp*1e4, seed, src, which (no stop ids here).
            let hdr=[OP_ADMIT as f32,0.,store as f32,action,cap as f32,if i==0 {48.} else {prewarm_new},0.,(prompt.len()+5) as f32];
            let _=apply(&hdr,[prompt.clone(),vec![0,0,0,-1,0]].concat(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
            loop {
                let prefill=!active[0].as_ref().unwrap().decoding;
                let h=[if prefill{OP_PREFILL}else{OP_ROUND} as f32,0.,0.,0.,0.,0.,0.,0.];
                if let Some((_,true))=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false){active[0]=None;break;}
            }
            tch::Cuda::synchronize(0);
            eprintln!("[serve] rank{} warm-up run store {store} action {action} prompt {} tokens: {:.2}s | {}",tp.rank,prompt.len(),tr.elapsed().as_secs_f64(),mem_note());
        }
        tch::Cuda::synchronize(0);
        eprintln!("[serve] rank{} warm-up {:.1}s (prewarm {prewarm}, stores {})",tp.rank,t0.elapsed().as_secs_f64(),stores.iter().flatten().count());
        // GLM53_SERVE_PREWARM_GRAPHS (default on with prewarm): capture every prebuilt store's drafter append graphs
        // (1..8 rows) and verifier chain graphs (lengths 2..8) now. Otherwise the first requests of each lane
        // capture them mid-decode (measured: first two prose runs after a restart 37-38 vs 43.5-44 tok/s).
        // Synthetic content: history/checkpoints are cleared, so these stores are only ever reset for reuse.
        if prewarm && std::env::var("GLM53_SERVE_PREWARM_GRAPHS").as_deref()!=Ok("0") {
            let t1=Instant::now();let dev=eng.w.device;
            // X4 sizing: allocator reserved bytes before/after capturing every store's graphs (private pools).
            extern "C"{fn rs_cuda_memory(allocated:*mut i64,reserved:*mut i64);}
            let mem=||{tch::Cuda::synchronize(0);let (mut a,mut r)=(0i64,0i64);unsafe{rs_cuda_memory(&mut a,&mut r)};(a,r)};
            let m0=mem();
            for s in 0..stores.len() {
                let Some(st)=stores[s].as_mut() else {continue};
                for n in 1..=8i64 {draft.append(st.context.as_mut().unwrap(),&Tensor::zeros([n,20480],(Kind::Float,dev)));}
                for len in 2..=8usize {
                    let all=with_anchor(1000,crate::speculative::chain(&vec![1001i64;len-1]));
                    crate::sampling::set_rows(&(0..len as u64).map(|i|(0f32,crate::sampling::key(0,i+1))).collect::<Vec<_>>());
                    let base=std::mem::replace(&mut st.state,DecodeStates(Vec::new()));
                    if st.chain.is_none(){st.chain=Some(ChainCache::new_on(&mut eng,&base,&all));}
                    let v=st.chain.as_mut().unwrap().verify(&mut eng,&base,1000,&all,len,&SERVE_EOS);
                    st.state=v.state;
                }
                st.history.clear();st.ckpt=[None,None];
                eprintln!("[serve] rank{} graph prewarm store {s} | {}",tp.rank,mem_note());
            }
            let m1=mem();
            eprintln!("[serve] rank{} graph prewarm {:.1}s allocated {:+.0} MiB reserved {:+.0} MiB ({} stores)",tp.rank,t1.elapsed().as_secs_f64(),
                (m1.0-m0.0) as f64/1048576.,(m1.1-m0.1) as f64/1048576.,stores.iter().flatten().count());
        }
    }
    // Vision warm-up (rank0): initializes the BF16 GEMM/attention kernels on a small image and a 2-group video.
    if let Some(v)=vision.as_ref() {
        let t0=Instant::now();
        for (f,h,w) in [(1i64,448i64,448i64),(4,224,336)] {
            let (px,g,gh,gw)=v.pixels(&Tensor::zeros([f,h,w,3],(Kind::Uint8,dev)));let _=v.forward(&px,g,gh,gw);
        }
        tch::Cuda::synchronize(0);
        eprintln!("[serve] rank0 vision warm-up {:.2}s (SDPA backend for 1024 patches: {}) | {}",t0.elapsed().as_secs_f64(),v.sdp_choice(1024),mem_note());
    }
    crate::host_memory::stage_mem("warm-up done");
    eprintln!("[serve] rank{} ready (max seqs {max_seqs}, KV budget {budget} tokens, granule {granule}, chunk {chunk})",tp.rank);
    // Doorbell (GLM53_SERVE_DOORBELL, default on): rank0 writes one byte over TCP before it starts each
    // broadcast op; rank1 blocks on the host for it and only then enters the op's first collective.
    // Without it rank1 waits for the next op header inside the GPU allreduce, whose kernel busy-polls
    // the peer flag: an idle worker shows ~100% GPU utilization and never drops its clocks.
    let doorbell_on=std::env::var("GLM53_SERVE_DOORBELL").as_deref()!=Ok("0");
    let doorbell_port:u16=std::env::var("GLM53_SERVE_DOORBELL_PORT").ok().and_then(|v|v.parse().ok())
        .unwrap_or_else(||std::env::var("GLM53_MASTER_PORT").ok().and_then(|v|v.parse::<u16>().ok()).unwrap_or(29931)+10);
    let mut doorbell:Option<std::net::TcpStream>=None;
    if doorbell_on {
        if tp.rank==0 {
            let l=std::net::TcpListener::bind(("0.0.0.0",doorbell_port)).expect("bind doorbell port");
            let (s,_)=l.accept().expect("doorbell accept");s.set_nodelay(true).unwrap();doorbell=Some(s);
        } else {
            let addr=std::env::var("GLM53_MASTER_ADDR").unwrap_or_else(|_|"127.0.0.1".into());
            let t0=Instant::now();
            let s=loop {match std::net::TcpStream::connect((addr.as_str(),doorbell_port)) {
                Ok(s)=>break s,
                Err(e)=>{assert!(t0.elapsed().as_secs()<600,"doorbell connect {addr}:{doorbell_port}: {e}");std::thread::sleep(std::time::Duration::from_millis(200));}}};
            s.set_nodelay(true).unwrap();doorbell=Some(s);
        }
        eprintln!("[serve] rank{} doorbell on port {doorbell_port}",tp.rank);
    }
    // Persistent prefix cache (GLM53_PCACHE=1): after the warm-up (whose synthetic checkpoints are not saved), both ranks
    // agree on the cached files over the doorbell stream before any control message.
    if crate::pcache::requested() {
        match doorbell.as_mut() {Some(d)=>crate::pcache::init(tp.rank,tp.world,model,draft_dir,d),None=>eprintln!("[pcache] needs the doorbell stream (TP2); disabled")}
    }
    let ctrl_tcp=std::env::var("GLM53_SERVE_CTRL_TCP").as_deref()==Ok("1");
    assert!(!ctrl_tcp||doorbell.is_some(),"GLM53_SERVE_CTRL_TCP=1 needs the doorbell (GLM53_SERVE_DOORBELL != 0, TP2)");
    if ctrl_tcp {eprintln!("[serve] rank{} control messages over the doorbell TCP stream",tp.rank);
    }
    if tp.rank!=0 {
        loop {
            crate::pcache::poll();
            let mut ctrl_payload:Vec<i64>=Vec::new();
            let h=if ctrl_tcp {
                match ctrl_read(doorbell.as_mut().unwrap()) {Some((h,p))=>{ctrl_payload=p;h},None=>{eprintln!("[serve] rank1 control stream closed; exiting");break;}}
            } else {
                if let Some(d)=doorbell.as_mut() {
                    use std::io::Read;let mut b=[0u8;1];
                    if d.read_exact(&mut b).is_err() {eprintln!("[serve] rank1 doorbell closed; exiting");break;}
                }
                bcast(Vec::new(),8)
            };
            match h[0] as i32 {
                OP_BEAT=>continue,
                OP_SHUTDOWN=>break,
                OP_ADMIT=>{let len=h[7] as usize;let ids:Vec<i64>=if ctrl_tcp {assert_eq!(ctrl_payload.len(),len);std::mem::take(&mut ctrl_payload)} else {bcast(Vec::new(),len).into_iter().map(|v|v as i64).collect()};
                    let _=apply(&h,ids,&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);}
                OP_ROUND=>{
                    let mut flag=|seq:&Seq,delta:&[i64]|->bool{if ctrl_tcp {delta.iter().any(|t|seq.stop_ids.contains(t))} else {bcast(Vec::new(),1)[0]!=0.}};
                    if let Some((slot,true))=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut flag){active[slot]=None;}}
                OP_CANCEL=>{if let Some((slot,true))=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false){active[slot]=None;}}
                OP_MULTI=>{let k=h[1] as usize;let slots:Vec<usize>=if ctrl_tcp {assert_eq!(ctrl_payload.len(),k);ctrl_payload.iter().map(|&v|v as usize).collect()} else {bcast(Vec::new(),k).into_iter().map(|v|v as usize).collect()};
                    let mut flag=|seq:&Seq,delta:&[i64]|->bool{if ctrl_tcp {delta.iter().any(|t|seq.stop_ids.contains(t))} else {bcast(Vec::new(),1)[0]!=0.}};
                    if !multi_commit_side() {settle_side_commit();}
                    for (slot,fin,_) in apply_multi(&slots,&mut eng,&mut stores,&mut active,&mut clock,&mut flag){if fin{active[slot]=None;}}}
                _=>{let _=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);}
            }
        }
        crate::pcache::shutdown(120);
        eprintln!("[serve] rank1 shutdown");return;
    }
    // ---- rank0: socket I/O, queue, admission and round-robin scheduling
    let _=std::fs::remove_file(socket);
    let listener=std::os::unix::net::UnixListener::bind(socket).expect("bind serve socket");listener.set_nonblocking(true).unwrap();
    let mut conn:Option<(std::io::BufReader<UnixStream>,UnixStream)>=None;let mut pending=String::new();
    struct Queued {id:serde_json::Value,ids:Vec<i64>,max_new:usize,stop_ids:Vec<i64>,at:Instant,temp:f32,seed:u64,mm:Vec<crate::vision::MmItem>}
    let vocab=match eng.w.vocab_shard {Some((_,total))=>total,None=>eng.w.embed.size()[0]};
    let mut queue:std::collections::VecDeque<Queued>=Default::default();
    let mut ids_of:Vec<serde_json::Value>=(0..max_seqs).map(|_|serde_json::Value::Null).collect();
    let mut last_beat=Instant::now();let mut rr=0usize;let mut shutdown=false;let mut last_multi=false;
    let write_line=|conn:&mut Option<(std::io::BufReader<UnixStream>,UnixStream)>,v:&serde_json::Value|{
        // The socket is non-blocking (shared with the reader): a full send buffer is WouldBlock, not a
        // lost client. Retry until written; only a real error drops the connection.
        if let Some((_,s))=conn.as_mut(){
            let bytes=format!("{}\n",v).into_bytes();let mut off=0;let mut ok=true;
            while off<bytes.len() {match s.write(&bytes[off..]) {
                Ok(0)=>{ok=false;break;}
                Ok(n)=>off+=n,
                Err(e) if e.kind()==std::io::ErrorKind::WouldBlock||e.kind()==std::io::ErrorKind::Interrupted=>std::thread::sleep(std::time::Duration::from_micros(200)),
                Err(e)=>{eprintln!("[serve] client write failed: {e}");ok=false;break;}}}
            if !ok{eprintln!("[serve] client connection dropped (write)");*conn=None;}}};
    let ring=std::cell::RefCell::new(doorbell);
    let ring_bell=||{if let Some(d)=ring.borrow_mut().as_mut(){d.write_all(&[1u8]).expect("doorbell write");}};
    let send=|h:[f32;8],payload:&[i64]|{
        if ctrl_tcp {ctrl_write(ring.borrow_mut().as_mut().unwrap(),&h,payload);return;}
        ring_bell();let _=bcast(h.to_vec(),8);if !payload.is_empty(){let _=bcast(payload.iter().map(|&t|t as f32).collect(),payload.len());}};
    loop {
        crate::pcache::poll();
        // 1. socket: accept, read every complete line without blocking the decode loop
        if conn.is_none() {
            if let Ok((s,_))=listener.accept(){s.set_nonblocking(true).unwrap();conn=Some((std::io::BufReader::new(s.try_clone().unwrap()),s));pending.clear();}
        }
        let mut lines=Vec::new();let mut gone=false;
        if let Some((reader,_))=conn.as_mut() {
            loop {match reader.read_line(&mut pending) {
                Ok(0)=>{eprintln!("[serve] client closed the connection");gone=true;break;}
                Ok(_) if pending.ends_with('\n')=>lines.push(std::mem::take(&mut pending)),
                _=>break,
            }}
        }
        if gone {
            // Front end went away: cancel everything it owned.
            conn=None;pending.clear();queue.clear();
            for slot in 0..max_seqs {if active[slot].is_some(){active[slot].as_mut().unwrap().cancel=true;}}
        }
        for line in lines {
            let v:serde_json::Value=match serde_json::from_str(line.trim()){Ok(v)=>v,Err(e)=>{write_line(&mut conn,&json!({"done":true,"error":format!("bad json: {e}")}));continue;}};
            if v["shutdown"].as_bool()==Some(true){shutdown=true;continue;}
            // Monitoring snapshot (front end /metrics): engine-side truth for queue and KV occupancy.
            if v["stats"].as_bool()==Some(true) {
                let live=stores.iter().flatten();
                write_line(&mut conn,&json!({"id":v["id"],"stats":{
                    "running":active.iter().filter(|a|a.is_some()).count(),
                    "decoding":active.iter().filter(|a|a.as_ref().map_or(false,|s|s.decoding)).count(),
                    "waiting":queue.len(),"max_seqs":max_seqs,"stores":live.clone().count(),"max_stores":max_stores,
                    "kv_allocated_tokens":live.clone().map(|s|s.capacity).sum::<i64>(),
                    "kv_active_tokens":live.filter(|s|s.busy).map(|s|s.capacity).sum::<i64>(),
                    "kv_budget_tokens":budget,"batch":batch,"pcache":crate::pcache::stats()}}));
                continue;
            }
            if let Some(cid)=v.get("cancel") {
                queue.retain(|q|{if &q.id==cid{false}else{true}});
                for slot in 0..max_seqs {if ids_of[slot]==*cid && active[slot].is_some(){active[slot].as_mut().unwrap().cancel=true;}}
                continue;
            }
            let ids:Vec<i64>=v["prompt_ids"].as_array().map(|a|a.iter().filter_map(|x|x.as_i64()).collect()).unwrap_or_default();
            let max_new=v["max_new"].as_u64().unwrap_or(256) as usize;
            if ids.is_empty()||max_new==0||(ids.len()+max_new) as i64>budget {
                write_line(&mut conn,&json!({"id":v["id"],"done":true,"error":format!("invalid request: {} prompt tokens + {max_new} new must be within the KV budget {budget}",ids.len())}));continue;}
            // Media items (salted placeholders): every id is a vocabulary id or covered by exactly one segment.
            let mm=match crate::vision::parse_items(&v["mm"]).and_then(|mm|{
                if !mm.is_empty()&&vision.is_none() {return Err("this server has no vision tower (GLM53_VISION is not 1)".to_string());}
                crate::vision::validate(&ids,vocab,&mm).map(|_|mm)}) {
                Ok(mm)=>mm,
                Err(e)=>{write_line(&mut conn,&json!({"id":v["id"],"done":true,"error":format!("invalid request: {e}")}));continue;}};
            let stop_ids=v["stop_token_ids"].as_array().map(|a|a.iter().filter_map(|x|x.as_i64()).collect()).unwrap_or_default();
            let temp=(v["temperature"].as_f64().unwrap_or(0.) as f32).clamp(0.,100.);
            let seed=v["seed"].as_u64().unwrap_or_else(||std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64)&0xffffff;
            queue.push_back(Queued{id:v["id"].clone(),ids,max_new,stop_ids,at:Instant::now(),temp,seed,mm});
            write_line(&mut conn,&json!({"id":v["id"],"queued":true,"position":queue.len()}));
        }
        if shutdown && active.iter().all(|a|a.is_none()) {send([OP_SHUTDOWN as f32,0.,0.,0.,0.,0.,0.,0.],&[]);write_line(&mut conn,&json!({"ok":true}));break;}
        // 2. cancellations of running sequences
        for slot in 0..max_seqs {
            if active[slot].as_ref().map_or(false,|s|s.cancel) {
                let h=[OP_CANCEL as f32,slot as f32,0.,0.,0.,0.,0.,0.];send(h,&[]);
                let _=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
                let seq=active[slot].take().unwrap();
                write_line(&mut conn,&json!({"id":ids_of[slot],"done":true,"cancelled":true,"token_ids":seq.generated,"prefill_ms":seq.prefill_ms,"prefix_hit_tokens":seq.hit}));
            }
        }
        // 3. admission (FIFO), bounded by free sequence slots and the KV budget
        while !shutdown {
            let Some(slot)=(0..max_seqs).find(|&i|active[i].is_none()) else {break};
            let Some(q)=queue.front() else {break};
            let need=(q.ids.len()+q.max_new) as i64;
            // (a) exact-prefix reuse, (b) reset a free store that is large enough, (c) allocate
            // Candidates: continue a store's full history, or restore its prompt checkpoint (prefix or exact).
            let mut best:Option<(usize,i32,usize)>=None;let mut best_which=0usize;
            for (i,st) in stores.iter().enumerate() {let Some(st)=st else{continue};if st.busy||st.capacity<need{continue;}
                if !st.history.is_empty() && q.ids.len()>st.history.len() && q.ids.starts_with(&st.history) && best.map_or(true,|b|st.history.len()>b.2) {best=Some((i,0,st.history.len()));}
                for (w,c) in st.ckpt.iter().enumerate() {if let Some(c)=c {if ckpt_reuse && (q.ids.len()>c.ids.len()||w==0) && q.ids.starts_with(&c.ids) && best.map_or(true,|b|c.ids.len()>b.2) {best=Some((i,3,c.ids.len()));best_which=w;}}}
            }
            // No reusable prefix: keep cached stores (history/checkpoints) as long as the budget allows.
            // Allocate a new store while it fits; otherwise take idle stores in LRU order: reuse the LRU one
            // in place if it is large enough, else evict it and retry. (Resetting the smallest fitting store
            // first destroyed a checkpoint written moments earlier by another lane.)
            // M2: a longer prefix may exist as the prompt checkpoint of a store that is busy or too small;
            // clone it into a fresh/reset target instead of re-prefilling (GLM53_SERVE_CLONE=0 disables).
            let mut clone:Option<(usize,usize,usize)>=None;   // (src store, prefix length, checkpoint slot)
            if clone_on && ckpt_reuse {for (i,st) in stores.iter().enumerate() {let Some(st)=st else{continue};
                for (w,c) in st.ckpt.iter().enumerate() {if let Some(c)=c {let l=c.ids.len();
                    if l>=clone_min && (q.ids.len()>l||w==0) && q.ids.starts_with(&c.ids) && l>best.map_or(0,|b|b.2) && clone.map_or(true,|x|l>x.1) {clone=Some((i,l,w));}}}}}
            // Restoring a store's boundary checkpoint in place rewrites rows past it and drops that store's
            // longer prompt checkpoint (an exact-hit candidate for a repeat of the older prompt). Clone the
            // boundary into another store instead; the planner below still falls back to the in-place restore
            // when no store can be allocated or reused.
            if clone.is_none() && clone_on {
                if let Some((i,3,l))=best {if best_which==1 && stores[i].as_ref().is_some_and(|s|s.ckpt[0].is_some()) {clone=Some((i,l,1));}}
            }
            // Checkpoint library: a longer prefix than any live store's candidate.
            let mut libc:Option<(u64,usize)>=None;
            if clone_on && ckpt_reuse {CKPT_LIB.with(|l|{for e in &l.borrow().entries {let len=e.ids.len();
                if len>=clone_min && (q.ids.len()>len||e.which==0) && q.ids.starts_with(&e.ids) && len>best.map_or(0,|b|b.2)
                    && len>clone.map_or(0,|c|c.1) && libc.map_or(true,|x|len>x.1) {libc=Some((e.id,len));}}});}
            // Persistent prefix cache (disk): a longer prefix than every in-memory candidate.
            let disk=if ckpt_reuse && crate::pcache::enabled() {
                crate::pcache::lookup(&q.ids,best.map_or(0,|b|b.2).max(clone.map_or(0,|c|c.1)).max(libc.map_or(0,|x|x.1)),clone_min)} else {None};
            let libc=if disk.is_some(){None}else{libc};
            let clone=if libc.is_some()||disk.is_some(){None}else{clone};
            let protect=clone.map(|c|c.0);
            let plan=if let (Some((i,a,_)),None,None,None)=(best,clone,libc,disk) {Some((i,a,0))} else {
                let cap=((need+granule-1)/granule*granule).min(budget);
                let used=|stores:&Vec<Option<Store>>|stores.iter().flatten().map(|s|s.capacity).sum::<i64>();
                let mut plan=None;
                // A clean idle store (no history, no checkpoints: e.g. a prewarmed one) already has its verifier and
                // drafter graphs captured; reuse it before allocating a new store that would capture them mid-decode.
                if let Some(i)=stores.iter().enumerate().filter_map(|(i,s)|s.as_ref().map(|s|(i,s)))
                    .filter(|(i,s)|!s.busy&&Some(*i)!=protect&&s.history.is_empty()&&s.ckpt.iter().all(|c|c.is_none())&&s.capacity>=need)
                    .min_by_key(|(_,s)|s.capacity).map(|(i,_)|i) {plan=Some((i,1,0));}
                if plan.is_none() {loop {
                    // KV pool: the new store needs a contiguous free range (idle stores are evicted below until one exists).
                    let fits=match crate::kv_pool::installed() {Some(p)=>p.fits(cap),None=>used(&stores)+cap<=budget};
                    if fits && stores.iter().flatten().count()<max_stores {
                        let idx=stores.iter().position(|s|s.is_none()).unwrap_or(stores.len());plan=Some((idx,2,cap));break;
                    }
                    let victim=stores.iter().enumerate().filter_map(|(i,s)|s.as_ref().map(|s|(i,s))).filter(|(i,s)|!s.busy&&Some(*i)!=protect).min_by_key(|(_,s)|s.used).map(|(i,s)|(i,s.capacity));
                    let Some((i,capacity))=victim else {break};
                    if capacity>=need {plan=Some((i,1,0));break;}
                    let h=[OP_EVICT as f32,i as f32,0.,0.,0.,0.,0.,0.];send(h,&[]);
                    let _=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
                }}
                plan
            };
            // No target for the clone: fall back to the (shorter) idle-store reuse if there is one.
            let (plan,clone,libc,disk)=match (plan,best,clone.is_some()||libc.is_some()||disk.is_some()) {(None,Some((i,a,_)),true)=>(Some((i,a,0)),None,None,None),(p,_,_)=>(p,clone,libc,disk)};
            // Victim evictions above may have pushed the chosen library entry out (both ranks apply the same
            // evictions, so this check sees the rank-1 library too): then admit cold.
            let libc=libc.filter(|(id,_)|CKPT_LIB.with(|l|l.borrow().entries.iter().any(|e|e.id==*id)));
            let Some((store,action,cap))=plan else {break};
            let (action,src,which)=match (clone,libc,disk) {
                (Some((src,_,w)),_,_) if action==1||action==2=>(4,src as i64,w),
                (None,Some((id,_)),_) if action==1||action==2=>(5,id as i64,0),
                (None,None,Some((id,_))) if action==1||action==2=>(6,id as i64,0),
                _=>(action,-1,if action==3{best_which}else{0})};
            if std::env::var("GLM53_SERVE_LOG").as_deref()==Ok("1") {
                eprintln!("[serve-admit] prompt {} need {} -> store {} action {} cap {} disk {:?} | stores {:?}",q.ids.len(),need,store,action,cap,disk,
                    stores.iter().enumerate().filter_map(|(i,s)|s.as_ref().map(|s|(i,s.capacity,s.busy,s.history.len(),s.ckpt.iter().flatten().map(|c|(c.ids.len(),q.ids.starts_with(&c.ids))).collect::<Vec<_>>()))).collect::<Vec<_>>());
            }
            let q=queue.pop_front().unwrap();
            let qms=q.at.elapsed().as_secs_f64()*1000.;
            let payload=[q.ids.clone(),q.stop_ids.clone(),vec![q.stop_ids.len() as i64,(q.temp*1e4).round() as i64,q.seed as i64,src,which as i64]].concat();
            let h=[OP_ADMIT as f32,slot as f32,store as f32,action as f32,cap as f32,q.max_new as f32,qms as f32,payload.len() as f32];
            send(h,&payload);
            let _=apply(&h,payload,&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
            let seq=active[slot].as_mut().unwrap();seq.stop_ids=q.stop_ids;seq.id=q.id.clone();ids_of[slot]=q.id;
            if !q.mm.is_empty() {seq.mm=Some(crate::vision::MmState::new(q.mm));}
        }
        // 3b. background prefix cache restores whose reads are done on this rank: install them (both ranks agree).
        for slot in 0..max_seqs {
            if active[slot].as_ref().is_some_and(|s|s.load.as_ref().is_some_and(|l|l.1.ready())) {
                let h=[OP_LOADED as f32,slot as f32,0.,0.,0.,0.,0.,0.];send(h,&[]);
                let _=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
            }
        }
        // 4a. batched verification over every decoding sequence (alternates with prefill chunks)
        let decoding:Vec<usize>=(0..max_seqs).filter(|&i|active[i].as_ref().map_or(false,|s|s.decoding)).collect();
        let prefilling=(0..max_seqs).any(|i|active[i].as_ref().map_or(false,|s|!s.decoding&&s.load.is_none()));
        if batch && decoding.len()>=2 && !(prefilling && last_multi) {
            last_multi=true;last_beat=Instant::now();
            send([OP_MULTI as f32,decoding.len() as f32,0.,0.,0.,0.,0.,0.],&decoding.iter().map(|&x|x as i64).collect::<Vec<_>>());
            let mut flag=|seq:&Seq,delta:&[i64]|->bool {
                // Control over TCP: both ranks decide from the replicated stop ids; a cancel is sent as OP_CANCEL next.
                if ctrl_tcp {return delta.iter().any(|t|seq.stop_ids.contains(t));}
                let stop=seq.cancel||delta.iter().any(|t|seq.stop_ids.contains(t));
                bcast(vec![if stop{1.}else{0.}],1)[0]!=0.
            };
            let results=apply_multi(&decoding,&mut eng,&mut stores,&mut active,&mut clock,&mut flag);
            for (slot,fin,delta) in results {
                if !delta.is_empty() {write_line(&mut conn,&json!({"id":ids_of[slot],"delta":delta}));}
                if fin {
                    let seq=active[slot].take().unwrap();
                    let decode_ms=seq.decode_started.map_or(0.,|t|t.elapsed().as_secs_f64()*1000.);
                    write_line(&mut conn,&json!({"id":ids_of[slot],"done":true,"token_ids":seq.generated,"prefill_ms":seq.prefill_ms,"decode_ms":decode_ms,
                        "total_ms":seq.started.elapsed().as_secs_f64()*1000.,"queue_ms":seq.queued_ms,"prefix_hit_tokens":seq.hit,"prompt_tokens":seq.prompt.len(),"disk_restore_ms":seq.disk_ms,
                        "rounds":seq.rounds,"accepted_drafts":seq.accepted,"drafted_tokens":seq.drafted,"store":seq.store,"batched":true,
                        "mm_encode_ms":seq.mm.as_ref().map_or(0.,|m|m.encode_ms),"mm_encoded_tokens":seq.mm.as_ref().map_or(0,|m|m.encoded_tokens),
                        "kv_active_tokens":stores.iter().flatten().filter(|s|s.busy).map(|s|s.capacity).sum::<i64>(),
                        "kv_used_tokens":stores.iter().flatten().map(|s|s.capacity).sum::<i64>(),"kv_budget_tokens":budget}));
                    ids_of[slot]=serde_json::Value::Null;
                }
            }
            continue;
        }
        last_multi=false;
        // 4. one unit of work for the next active sequence (round-robin)
        // Sequences still loading from the prefix cache wait for OP_LOADED.
        let next=if batch && decoding.len()>=2 {(0..max_seqs).map(|k|(rr+k)%max_seqs).find(|&i|active[i].as_ref().map_or(false,|s|!s.decoding&&s.load.is_none()))}
            else {(0..max_seqs).map(|k|(rr+k)%max_seqs).find(|&i|active[i].as_ref().is_some_and(|s|s.load.is_none()))};
        let Some(slot)=next else {
            // While a prefix cache restore is in flight, beat every idle turn: rank1 moves its staged chunks to the
            // device between control messages.
            if last_beat.elapsed().as_secs_f64()>=1.0 || crate::pcache::loading() {send([OP_BEAT as f32,0.,0.,0.,0.,0.,0.,0.],&[]);last_beat=Instant::now();}
            std::thread::sleep(std::time::Duration::from_millis(2));continue;
        };
        rr=slot+1;last_beat=Instant::now();
        if !active[slot].as_ref().unwrap().decoding {
            // GLM53_SERVE_CHUNK_SOLO (default 0 = off): with no other active sequence nothing waits behind this
            // prefill unit, so use a larger chunk (fewer full weight passes: 4096 measured -3..-5% on 4K-32K
            // prompts, bench/m1-chunk). Re-decided per unit: once another request is admitted, the rest of the
            // prompt continues at GLM53_PREFILL_CHUNK. Chunk splits change GEMM shapes (L1).
            let solo=chunk_solo>0 && active.iter().filter(|a|a.is_some()).count()==1;
            let h=[OP_PREFILL as f32,slot as f32,if solo{chunk_solo as f32}else{0.},0.,0.,0.,0.,0.];send(h,&[]);
            let _=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut |_,_|false);
            continue;
        }
        let h=[OP_ROUND as f32,slot as f32,0.,0.,0.,0.,0.,0.];send(h,&[]);
        let mut out_delta:Vec<i64>=Vec::new();
        let mut flag=|seq:&Seq,delta:&[i64]|->bool {
            out_delta=delta.to_vec();
            if ctrl_tcp {return delta.iter().any(|t|seq.stop_ids.contains(t));}
            let stop=seq.cancel||delta.iter().any(|t|seq.stop_ids.contains(t));
            bcast(vec![if stop{1.}else{0.}],1)[0]!=0.
        };
        let r=apply(&h,Vec::new(),&mut eng,&mut stores,&mut active,&mut clock,&mut flag);
        drop(flag);
        if !out_delta.is_empty() {write_line(&mut conn,&json!({"id":ids_of[slot],"delta":out_delta}));}
        if let Some((_,true))=r {
            let seq=active[slot].take().unwrap();
            let decode_ms=seq.decode_started.map_or(0.,|t|t.elapsed().as_secs_f64()*1000.);
            write_line(&mut conn,&json!({"id":ids_of[slot],"done":true,"token_ids":seq.generated,"prefill_ms":seq.prefill_ms,"decode_ms":decode_ms,
                "total_ms":seq.started.elapsed().as_secs_f64()*1000.,"queue_ms":seq.queued_ms,"prefix_hit_tokens":seq.hit,"prompt_tokens":seq.prompt.len(),"disk_restore_ms":seq.disk_ms,
                "rounds":seq.rounds,"accepted_drafts":seq.accepted,"drafted_tokens":seq.drafted,"store":seq.store,
                "copy_rounds":seq.copy.0,"copy_drafted":seq.copy.1,"copy_accepted":seq.copy.2,
                "mm_encode_ms":seq.mm.as_ref().map_or(0.,|m|m.encode_ms),"mm_encoded_tokens":seq.mm.as_ref().map_or(0,|m|m.encoded_tokens),
                "kv_active_tokens":stores.iter().flatten().filter(|s|s.busy).map(|s|s.capacity).sum::<i64>(),
                "kv_used_tokens":stores.iter().flatten().map(|s|s.capacity).sum::<i64>(),"kv_budget_tokens":budget}));
            ids_of[slot]=serde_json::Value::Null;
        }
    }
    crate::pcache::shutdown(120);
    eprintln!("[serve] rank0 shutdown");
}

/// Proposal 3 audit (TP2, service stopped): is the verify forward row-count invariant? One 512-token prefill, then
/// eager chain verifies of k rows (same token prefix) for every k in GLM53_INV_ROWS (default 2..8,12,16). For each k,
/// rows 0..min(k,k_ref) of every per-layer probe point are compared bitwise with the k_ref (first listed) run; the
/// first differing point per k localises the row-count-dependent kernel.
/// I1 (row-timing): verification-window time per row count on the serving chain graphs. Both ranks run the same
/// sequence (the graphs contain the RDMA allreduce). For each context length (GLM53_I1_CTX, default 4096,16384,32768) a real
/// long code context is prefilled; for rows 1..=8 (anchor + r-1 chain nodes taken from the prompt's continuation) a chain graph
/// is captured exactly as serving does (T=1 sampling rows), then replayed: N back-to-back replays (amortised, the GPU
/// cost of the window) and M single replays with a device sync on each side (adds launch + sync latency). The drafter,
/// host selection and commit are not included: compare with GLM53_SERVE_LOG round_ms for the rest of a round.
pub fn row_timing(model:&std::path::Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    std::env::set_var("GLM53_PREFILL_BATCH","1");
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    if std::env::var("GLM53_SERVE_SAMPLING").as_deref()!=Ok("0") {
        let (start,width)=match eng.w.vocab_shard {Some((start,_)) if crate::head_select::enabled()=>(start,eng.w.lm_head.size()[0]),Some((_,total))=>(0,total),None=>(0,eng.w.lm_head.size()[0])};
        crate::sampling::enable(start,width,dev);
    }
    // Real long context: the longest prompt+answer of the long-context drafter data (32K-100K tokens of source code and
    // the target's own thinking), GLM53_I1_TOKENS overrides the jsonl path.
    let src=std::env::var("GLM53_I1_TOKENS").unwrap_or_else(|_|"bench/opd/think3/gen.jsonl".into());
    let all:Vec<i64>=std::fs::read_to_string(&src).unwrap().lines().map(|l|{let v:serde_json::Value=serde_json::from_str(l).unwrap();
        v["prompt_ids"].as_array().unwrap().iter().chain(v["output_ids"].as_array().unwrap().iter()).map(|x|x.as_i64().unwrap()).collect::<Vec<i64>>()})
        .max_by_key(|t|t.len()).unwrap();
    let env_n=|k:&str,d:usize|std::env::var(k).ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(d);
    let (n_back,n_single)=(env_n("GLM53_I1_N",40),env_n("GLM53_I1_M",20));
    let ctxs:Vec<usize>=std::env::var("GLM53_I1_CTX").unwrap_or_else(|_|"4096,16384,32768".into()).split(',').map(|v|v.parse().unwrap()).collect();
    // GLM53_I1_AB="KEY=VAL;KEY2=VAL2": in-process A/B (the loaded profile is A, these overrides are B).
    let ab:Option<Vec<(String,String)>>=std::env::var("GLM53_I1_AB").ok().filter(|v|!v.is_empty()).map(|v|v.split(';').map(|kv|{
        let (k,v)=kv.split_once('=').expect("GLM53_I1_AB: KEY=VAL;...");(k.trim().to_string(),v.trim().to_string())}).collect());
    for ctx in ctxs {
        assert!(all.len()>ctx+9,"token source shorter than the requested context");
        // chunked like serving (2048-token units continuing the same state)
        let mut st:Option<DecodeStates>=None;
        for c in (0..ctx).step_by(2048) {let e=(c+2048).min(ctx);let (_,s2)=eng.prefill_with(&Tensor::from_slice(&all[c..e]).to_device(dev),st.take());st=Some(s2);}
        let base=st.unwrap();
        // Routed-expert cost depends on which experts the chain's tokens hit: average K chains per row count
        // (drafts taken from different offsets of the real continuation), N back-to-back replays each.
        let k_chains=env_n("GLM53_I1_K",6);
        if let Some(ab)=&ab {row_timing_ab(&mut eng,&base,&all,ctx,k_chains,n_back,n_single,ab,tp.rank);continue;}
        for rows in 1..=8usize {
            let (mut backs,mut singles)=(Vec::new(),Vec::new());
            for k in 0..k_chains {
                let o=ctx+1+k*11;let nodes=with_anchor(all[ctx],crate::speculative::chain(&all[o..o+rows-1]));
                crate::sampling::set_rows(&(0..rows as u64).map(|i|(1.0f32,crate::sampling::key(7,i+1))).collect::<Vec<_>>());
                let mut cache=ChainCache::new(&mut eng,&base,&nodes);
                let g=&mut cache.entries[0];g.replay(&base,&nodes);tch::Cuda::synchronize(0);
                tp_barrier(dev);let t0=Instant::now();for _ in 0..n_back {g.graph.replay();} tch::Cuda::synchronize(0);
                backs.push(t0.elapsed().as_secs_f64()*1000./n_back as f64);
                for _ in 0..n_single {tp_barrier(dev);let t=Instant::now();g.graph.replay();tch::Cuda::synchronize(0);singles.push(t.elapsed().as_secs_f64()*1000.);}
                drop(cache);
            }
            singles.sort_by(|a,b|a.partial_cmp(b).unwrap());
            let mean=backs.iter().sum::<f64>()/backs.len() as f64;
            println!("[i1] rank{} ctx {ctx} rows {rows}: back-to-back mean {mean:.3} ms/replay over {} chains (min {:.3} max {:.3}) | single min {:.3} median {:.3} ms",
                tp.rank,backs.len(),backs.iter().cloned().fold(f64::MAX,f64::min),backs.iter().cloned().fold(0.,f64::max),singles[0],singles[singles.len()/2]);
        }
    }
    crate::sampling::clear_draft_rows();
}
/// Both ranks leave together (a 1-element allreduce, then a device sync): single-replay timings then exclude the other
/// host's launch lateness, which otherwise lands in the first allreduce's wait.
fn tp_barrier(dev:Device) {
    tch::Cuda::synchronize(0);crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
}
/// Sets `vars`, returning the previous values (None: unset) for `env_restore`; resets cached switches.
fn env_apply(vars:&[(String,String)])->Vec<(String,Option<String>)> {
    let old=vars.iter().map(|(k,v)|{let o=std::env::var(k).ok();std::env::set_var(k,v);(k.clone(),o)}).collect();
    crate::tp::reset_flag_cache();old
}
fn env_restore(old:Vec<(String,Option<String>)>) {
    for (k,o) in old {match o {Some(v)=>std::env::set_var(&k,v),None=>std::env::remove_var(&k)}}
    crate::tp::reset_flag_cache();
}
/// In-process A/B on the serving chain graphs (one model load, both ranks in lockstep): per chain, graph A (profile) and
/// graph B (profile + overrides) are captured side by side, each on its own copy of the base state. L0: logits and
/// feature taps of B equal A's after every timed pair. Timing: single replays in A,B,B,A order (paired B-A difference
/// per quadruple) and back-to-back blocks A,B,B,A. Replays are idempotent (no commit), so both graphs see the same input.
fn row_timing_ab(eng:&mut Engine,base:&DecodeStates,all:&[i64],ctx:usize,k_chains:usize,n_back:usize,n_single:usize,ab:&[(String,String)],rank:usize) {
    let dev=eng.w.device;let mut mismatches=0usize;let mut checks=0usize;
    for rows in 1..=8usize {
        let (mut sa,mut sb,mut diffs,mut ba,mut bb)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
        for k in 0..k_chains {
            let o=ctx+1+k*11;let nodes=with_anchor(all[ctx],crate::speculative::chain(&all[o..o+rows-1]));
            crate::sampling::set_rows(&(0..rows as u64).map(|i|(1.0f32,crate::sampling::key(7,i+1))).collect::<Vec<_>>());
            let mut ca=ChainCache::new(eng,base,&nodes);ca.entries[0].replay(base,&nodes);
            let old=env_apply(ab);
            let mut cb=ChainCache::new(eng,base,&nodes);cb.entries[0].replay(base,&nodes);
            env_restore(old);
            let (ga,gb)=(&ca.entries[0],&cb.entries[0]);
            let mut check=|ga:&ChainGraph,gb:&ChainGraph|{tch::Cuda::synchronize(0);checks+=1;
                if !(ga.logits.equal(&gb.logits)&&ga.features.equal(&gb.features)){mismatches+=1;}};
            tch::Cuda::synchronize(0);check(ga,gb);
            let time=|g:&ChainGraph|->f64{tp_barrier(dev);let t=Instant::now();g.graph.replay();tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1000.};
            for _ in 0..n_single.div_ceil(2) {
                let (a1,b1,b2,a2)=(time(ga),time(gb),time(gb),time(ga));
                sa.extend([a1,a2]);sb.extend([b1,b2]);diffs.push((b1+b2-a1-a2)/2.);
            }
            check(ga,gb);
            let block=|g:&ChainGraph|->f64{tp_barrier(dev);let t=Instant::now();for _ in 0..n_back {g.graph.replay();}tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1000./n_back as f64};
            let (a1,b1,b2,a2)=(block(ga),block(gb),block(gb),block(ga));ba.extend([a1,a2]);bb.extend([b1,b2]);
            check(ga,gb);
            // every node's materialized state (KDA h/conv by bit pattern); panics on the first difference
            // GLM53_I1_INEXACT=1: timing of a rounding-level (L1) change; the logits/feature mismatches are only counted
            if std::env::var("GLM53_I1_INEXACT").as_deref()!=Ok("1") {ga.states.assert_exact(&gb.states,base,"in-process A/B verifier states");}
            drop(ca);drop(cb);
        }
        let med=|v:&mut Vec<f64>|{v.sort_by(|a,b|a.partial_cmp(b).unwrap());v[v.len()/2]};
        let mean=|v:&Vec<f64>|v.iter().sum::<f64>()/v.len() as f64;
        println!("[i1ab] rank{rank} ctx {ctx} rows {rows}: single median A {:.3} B {:.3} | paired B-A median {:+.3} mean {:+.3} ms (n {}) | back-to-back A {:.3} B {:.3} ms",
            med(&mut sa),med(&mut sb),med(&mut diffs.clone()),mean(&diffs),diffs.len(),mean(&ba),mean(&bb));
    }
    println!("[i1ab] rank{rank} ctx {ctx} L0: B logits+features equal A in {}/{} checks{}; states equal in all {} chains",checks-mismatches,checks,if mismatches>0{" MISMATCH"}else{""},8*k_chains);
}
pub fn verify_invariance(model:&std::path::Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/p1-prefill/the-suite.json").unwrap()).unwrap();
    let all:Vec<i64>=suite["cases"][0]["prompt_ids"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect();
    let (_,base)=eng.prefill(&Tensor::from_slice(&all[..512]).to_device(dev));
    let ks:Vec<usize>=std::env::var("GLM53_INV_ROWS").unwrap_or_else(|_|"8,2,3,4,5,6,7,12,16".into()).split(',').map(|v|v.parse().unwrap()).collect();
    let head_ids=std::env::var("GLM53_INV_HEAD_IDS").as_deref()==Ok("1");
    // GLM53_INV_MOE=1: routed-expert cooperative kernel alone (layer 3, real resident experts). 16 rows, routes drawn
    // from 24 experts (heavy overlap), random positive weights; each subset's rows vs the same rows in the full batch.
    if std::env::var("GLM53_INV_MOE").as_deref()==Ok("1") {
        tch::manual_seed(99);
        let x=(Tensor::randn([16,4096],(Kind::Float,dev))*0.5).to_kind(Kind::Half);
        let pool=Tensor::randperm(288,(Kind::Int64,dev)).narrow(0,0,24);
        let ids=pool.index_select(0,&Tensor::randint(24,[16*8],(Kind::Int64,dev))).view([16,8]);
        // distinct experts within a row (duplicates would be a different routing contract)
        let ids=Tensor::stack(&(0..16).map(|r|{let row=Vec::<i64>::try_from(ids.get(r).to_device(Device::Cpu)).unwrap();
            let mut seen=Vec::new();let mut extra=Vec::<i64>::try_from(pool.to_device(Device::Cpu)).unwrap().into_iter();
            for e in row {if !seen.contains(&e){seen.push(e);}} while seen.len()<8 {let e=extra.next().unwrap();if !seen.contains(&e){seen.push(e);}}
            Tensor::from_slice(&seen).to_device(dev)}).collect::<Vec<_>>(),0);
        let w=Tensor::rand([16,8],(Kind::Float,dev));let w=(&w/w.sum_dim_intlist(&[1i64][..],true,Kind::Float)).contiguous();
        let fast=eng.fast.as_mut().unwrap();
        let full=fast.expert_cooperative(3,&x,&ids.contiguous(),&w);
        for (a,b) in [(0i64,1i64),(0,2),(0,4),(0,8),(4,12),(8,16),(0,16)] {
            let part=fast.expert_cooperative(3,&x.narrow(0,a,b-a).contiguous(),&ids.narrow(0,a,b-a).contiguous(),&w.narrow(0,a,b-a).contiguous());
            let f=full.narrow(0,a,b-a);
            println!("[inv-moe] rank{} rows {a}..{b} of 16: equal {} max |diff| {:.3e}",tp.rank,f.equal(&part),(f.to_kind(Kind::Float)-part.to_kind(Kind::Float)).abs().max().double_value(&[]));
        }
    }
    // Batch invariance (GLM53_INV_MULTI=1): sequence A's rows verified alone (single path) vs inside a two-sequence
    // tree_forward_multi batch with B (serving batch path). Predictions and the five feature taps, bitwise.
    if std::env::var("GLM53_INV_MULTI").as_deref()==Ok("1") {
        let (_,base_b)=eng.prefill(&Tensor::from_slice(&all[1024..1536]).to_device(dev));
        // A after other sequences (row offset != 0; 8-row pieces then split A): B then A, and C,B,A with a third base.
        let (_,base_c)=eng.prefill(&Tensor::from_slice(&all[2048..2560]).to_device(dev));
        for (kc,kb,ka) in [(0usize,3usize,4usize),(0,5,4),(0,7,6),(3,2,4),(5,3,4),(2,6,7)] {
            let ia=&all[512..512+ka];let ib=&all[1536..1536+kb];let ic=&all[2560..2560+kc];
            let ch=|n:usize|->Vec<Option<usize>>{(0..n).map(|j|if j==0{None}else{Some(j-1)}).collect()};
            let (pa,pb,pc)=(ch(ka),ch(kb),ch(kc));
            let (sa,sb,sc)=(crate::dsa::tree_selection(512,ka as i64),crate::dsa::tree_selection(512,kb as i64),crate::dsa::tree_selection(512,kc.max(1) as i64));
            crate::forward::probe_begin();
            let (l1,_,_)=verifier_forward(&mut eng,&Tensor::from_slice(ia).to_device(dev),&base,&pa,sa,false,false);
            let t1=crate::forward::probe_take();
            let mut ids=Vec::new();let mut segs=Vec::new();let mut bases:Vec<&DecodeStates>=Vec::new();let mut par=Vec::new();let mut sels=Vec::new();
            if kc>0 {segs.push((ids.len(),kc));ids.extend_from_slice(ic);bases.push(&base_c);par.push(pc.clone());sels.push(sc);}
            segs.push((ids.len(),kb));ids.extend_from_slice(ib);bases.push(&base_b);par.push(pb.clone());sels.push(sb);
            let off=ids.len() as i64;segs.push((ids.len(),ka));ids.extend_from_slice(ia);bases.push(&base);par.push(pa.clone());sels.push(sa);
            crate::forward::probe_begin();
            let (l2,_,_)=eng.tree_forward_multi(&Tensor::from_slice(&ids).to_device(dev),&bases,&segs,&par,&sels,true,false,true);
            let t2=crate::forward::probe_take();
            let l2=l2.narrow(0,off,ka as i64);
            let single:std::collections::HashMap<(usize,&str),&Tensor>=t1.iter().filter(|(_,g,_)|!g.starts_with("kda_")).map(|(l,g,t)|((*l,*g),t)).collect();
            let mut first=None;
            for (l,g,t) in &t2 {if let Some(a)=single.get(&(*l,*g)) {
                let total=ids.len() as i64;let b=if t.size()[0]==total {t.narrow(0,off,ka as i64)} else {continue};
                if a.size()!=b.size()||!a.equal(&b) {first=Some((*l,*g,if a.size()==b.size(){(a.to_kind(Kind::Float)-b.to_kind(Kind::Float)).abs().max().double_value(&[])}else{f64::NAN}));break;}
            }}
            println!("[inv-multi-off] rank{} C {kc} + B {kb} + A {ka}: first differing tap {first:?}",tp.rank);
            println!("[inv-multi-off] rank{} C {kc} + B {kb} + A {ka} (A at row {off}): logits equal {} (max |diff| {:.3e})",tp.rank,l1.equal(&l2),(l1.to_kind(Kind::Float)-l2.to_kind(Kind::Float)).abs().max().double_value(&[]));
        }
        for (ka,kb) in [(2usize,3usize),(4,4),(4,8),(8,8),(8,2)] {
            let ia=&all[512..512+ka];let ib=&all[1536..1536+kb];
            let pa:Vec<Option<usize>>=(0..ka).map(|j|if j==0{None}else{Some(j-1)}).collect();
            let pb:Vec<Option<usize>>=(0..kb).map(|j|if j==0{None}else{Some(j-1)}).collect();
            let (sa,sb)=(crate::dsa::tree_selection(512,ka as i64),crate::dsa::tree_selection(512,kb as i64));
            crate::forward::probe_begin();
            let (p1,_,f1)=verifier_forward(&mut eng,&Tensor::from_slice(ia).to_device(dev),&base,&pa,sa,false,true);
            let t1=crate::forward::probe_take();
            let f1=Tensor::cat(&f1,1);
            let mut ids=ia.to_vec();ids.extend_from_slice(ib);
            let input=Tensor::from_slice(&ids).to_device(dev);
            crate::forward::probe_begin();
            let (p2,_,f2)=eng.tree_forward_multi(&input,&[&base,&base_b],&[(0,ka),(ka,kb)],&[pa.clone(),pb.clone()],&[sa,sb],true,true,true);
            let t2=crate::forward::probe_take();
            // first (layer, tag) where A's rows differ between the single and the batched forward
            let single:std::collections::HashMap<(usize,&str),&Tensor>=t1.iter().filter(|(_,g,_)|!g.starts_with("kda_")).map(|(l,g,t)|((*l,*g),t)).collect();
            let mut first=None;
            for (l,g,t) in &t2 {if let Some(a)=single.get(&(*l,*g)) {
                let b=t.narrow(0,0,ka as i64);
                if a.size()!=b.size()||!a.equal(&b) {first=Some((*l,*g,if a.size()==b.size(){(a.to_kind(Kind::Float)-b.to_kind(Kind::Float)).abs().max().double_value(&[])}else{f64::NAN}));break;}
            }}
            println!("[inv-multi] rank{} A {ka} + B {kb}: first differing tap {first:?}",tp.rank);
            // logits (not only argmax): T>0 sampling sees them
            let (l1,_,_)=verifier_forward(&mut eng,&Tensor::from_slice(ia).to_device(dev),&base,&pa,sa,false,false);
            let (l2,_,_)=eng.tree_forward_multi(&input,&[&base,&base_b],&[(0,ka),(ka,kb)],&[pa.clone(),pb.clone()],&[sa,sb],true,false,true);
            let l2=l2.narrow(0,0,ka as i64);
            println!("[inv-multi] rank{} A {ka} + B {kb}: logits equal {} (max |diff| {:.3e})",tp.rank,l1.equal(&l2),(l1.to_kind(Kind::Float)-l2.to_kind(Kind::Float)).abs().max().double_value(&[]));
            let f2=Tensor::cat(&f2,1).narrow(0,0,ka as i64);
            let (pe,fe)=(p1.to_kind(Kind::Int64).equal(&p2.narrow(0,0,ka as i64).to_kind(Kind::Int64)),f1.equal(&f2));
            let d=(f1.to_kind(Kind::Float)-f2.to_kind(Kind::Float)).abs().max().double_value(&[]);
            println!("[inv-multi] rank{} A {ka} rows + B {kb} rows: predictions equal {pe}, features equal {fe} (max |diff| {d:.3e})",tp.rank);
        }
    }
    let mut reference:Option<(usize,Vec<(usize,&'static str,Tensor)>)>=None;
    for &k in &ks {
        let ids=Tensor::from_slice(&all[512..512+k]).to_device(dev);
        let parents:Vec<Option<usize>>=(0..k).map(|j|if j==0{None}else{Some(j-1)}).collect();
        let sel=crate::dsa::tree_selection(512,k as i64);   // as ChainGraph::selection in serving
        let _=verifier_forward(&mut eng,&ids,&base,&parents,sel,false,head_ids);tch::Cuda::synchronize(0);   // warm
        crate::forward::probe_begin();
        let _=verifier_forward(&mut eng,&ids,&base,&parents,sel,false,head_ids);tch::Cuda::synchronize(0);
        let pts=crate::forward::probe_take();
        // repeatability at the same k first (nondeterminism would invalidate the comparison)
        crate::forward::probe_begin();
        let _=verifier_forward(&mut eng,&ids,&base,&parents,sel,false,head_ids);tch::Cuda::synchronize(0);
        let again=crate::forward::probe_take();
        let repeat=pts.iter().zip(&again).all(|(a,b)|a.2.equal(&b.2));
        // GLM53_INV_GRAPH=1: the same forward captured into a CUDA graph (as ChainGraph serves it) and replayed;
        // taps are captured copies. Serving runs single sequences as graph replays but batches eagerly.
        if std::env::var("GLM53_INV_GRAPH").as_deref()==Ok("1") {
            crate::forward::probe_begin();
            crate::tp::graph::begin().unwrap();
            let _=verifier_forward(&mut eng,&ids,&base,&parents,sel,false,head_ids);
            crate::tp::graph::end().unwrap();
            let g=crate::tp::graph::Owned::take();let gp=crate::forward::probe_take();
            g.replay();tch::Cuda::synchronize(0);
            let mut first=None;let mut n=0;let mut layer=0usize;let mut tags=std::collections::BTreeMap::<&str,usize>::new();
            for ((li,tag,a),(_,_,b)) in pts.iter().zip(&gp) {
                if !tag.starts_with("kda_") {layer=*li;}
                if a.size()!=b.size()||!a.equal(b) {n+=1;*tags.entry(tag).or_default()+=1;if first.is_none(){first=Some((layer,*tag,(a.to_kind(Kind::Float)-b.to_kind(Kind::Float)).abs().max().double_value(&[])));}}
            }
            println!("[inv-graph] rank{} k {k}: graph replay vs eager differing taps {n}/{} first {first:?} by tag {tags:?}",tp.rank,pts.len());
        }
        match &reference {
            None=>{println!("[inv] rank{} k {k} sel {sel:?} reference ({} points), repeatable {repeat}",tp.rank,pts.len());reference=Some((k,pts));}
            Some((kr,rp))=>{
                let m=k.min(*kr) as i64;let mut first=None;let mut ndiff=0;let mut tags=std::collections::BTreeMap::<&str,usize>::new();
                let mut layer=0usize;
                for ((li,tag,a),(_,_,b)) in pts.iter().zip(rp) {
                    if !tag.starts_with("kda_") {layer=*li;}   // KDA-internal taps carry no layer index
                    let li=&layer;
                    let rows=|t:&Tensor|if *tag=="route_w"||t.size()[0]>=m {t.narrow(0,0,m.min(t.size()[0]))} else {t.shallow_clone()};
                    let (x,y)=(rows(a),rows(b));
                    if x.size()!=y.size() || !x.equal(&y) {ndiff+=1;*tags.entry(tag).or_default()+=1;
                        if first.is_none(){let d=if x.size()==y.size(){(x.to_kind(Kind::Float)-y.to_kind(Kind::Float)).abs().max().double_value(&[])}else{f64::NAN};first=Some((*li,*tag,d));}}
                }
                println!("[inv] rank{} k {k} sel {sel:?} vs {kr}: repeatable {repeat}, differing points {ndiff}/{} first {:?} by tag {:?}",tp.rank,pts.len(),first,tags);
            }
        }
    }
}
