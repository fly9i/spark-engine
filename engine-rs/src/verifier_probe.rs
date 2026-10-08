// SPDX-License-Identifier: MIT
//! Full-model qualification of deferred KDA verifier records. This gate keeps
//! one topology/mode at a time and materializes selected nodes sequentially.
use std::{ffi::OsString,path::Path};
use serde_json::{json,Value};
use tch::{Kind,Tensor};
use crate::{
    dflash::{Candidates,Context,Drafter},
    dsa::TreeSelection,
    forward::{DecodeStates,Engine,LayerState,snapshot},
    verifier_state::{Layer,VerifierStates},
};

const FLAGS:&[&str]=&[
    "GLM53_KDA_FUSED","GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN",
    "GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT","GLM53_KDA_CORRECTION_REPLAY",
    "GLM53_TARGET_TOP1_TP",
];
struct Restore(Vec<(String,Option<OsString>)>);
impl Restore {
    fn new()->Self {Self(FLAGS.iter().map(|&key|(key.to_owned(),std::env::var_os(key))).collect())}
}
impl Drop for Restore {fn drop(&mut self){for (key,value) in &self.0 {
    if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
}}}

// Reference always uses legacy DSA bookkeeping and old small-M FP8 kernel.
// The same quantized weights, scales and precision flags remain in force.
// OsString restoration includes panic unwind and both candidate flags.
struct IndexReference(Vec<(&'static str,Option<OsString>)>);
impl IndexReference {
    fn enter()->Self {
        let flags=["GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_SHARED_GU_FUSED","GLM53_DSA_INDEX_FUSED","GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD"];
        let saved=Self(flags.into_iter().map(|k|(k,std::env::var_os(k))).collect());
        for key in flags {std::env::set_var(key,"0");}saved
    }
}
impl Drop for IndexReference {fn drop(&mut self) {for (key,value) in &self.0 {
    if let Some(value)=value {std::env::set_var(key,value);}else{std::env::remove_var(key);}
}}}

fn save(out:&Path,rank:usize,report:&Value) {
    let dst=out.join(format!("correction-rank{rank}.json"));
    let tmp=out.join(format!("correction-rank{rank}.json.tmp"));
    std::fs::write(&tmp,serde_json::to_string_pretty(report).unwrap()).unwrap();
    std::fs::rename(tmp,dst).unwrap();
}

fn exact(a:&Tensor,b:&Tensor,context:&str) {
    assert_eq!(a.size(),b.size(),"{context} shape");assert_eq!(a.kind(),b.kind(),"{context} dtype");
    let bit_kind=match a.kind() {
        Kind::Float=>Some(Kind::Int),Kind::Half|Kind::BFloat16=>Some(Kind::Int16),Kind::Double=>Some(Kind::Int64),_=>None,
    };
    if let Some(kind)=bit_kind {
        assert!(a.isfinite().all().int64_value(&[])!=0 && b.isfinite().all().int64_value(&[])!=0,"{context} nonfinite");
        assert!(a.contiguous().view_dtype(kind).equal(&b.contiguous().view_dtype(kind)),"{context} raw bits");
    }else{assert!(a.equal(b),"{context} values");}
}

fn features(a:&[Tensor],b:&[Tensor],context:&str)->(Tensor,Tensor) {
    assert_eq!(a.len(),5,"{context} five target feature sites");assert_eq!(b.len(),a.len());
    for (i,(a,b)) in a.iter().zip(b).enumerate() {exact(a,b,&format!("{context} feature{i}"));}
    (Tensor::cat(a,1),Tensor::cat(b,1))
}

fn state_report(a:&DecodeStates,b:&DecodeStates,context:&str)->Value {
    assert_eq!(a.0.len(),b.0.len(),"{context} layer count");
    let (mut kda,mut mla)=(0usize,0usize);let mut max_mla=0.0f64;
    for (layer,(a,b)) in a.0.iter().zip(&b.0).enumerate() {
        match (a,b) {
            (LayerState::Kda(a),LayerState::Kda(b))=>{
                assert_eq!(a.h.kind(),Kind::Float);assert_eq!(a.conv.kind(),Kind::Float);
                exact(&a.h,&b.h,&format!("{context} layer{layer} H"));
                exact(&a.conv,&b.conv,&format!("{context} layer{layer} conv"));kda+=1;
            },
            (LayerState::MlaLatent(a),LayerState::MlaLatent(b))=>{
                let error=a.max_diff(b);assert_eq!(error,0.,"{context} layer{layer} MLA state");
                max_mla=max_mla.max(error);mla+=1;
            },
            _=>panic!("{context}: unexpected state kind at layer{layer}; latent verifier gate required"),
        }
    }
    json!({"kda_layers":kda,"kda_h_conv_raw_bits":true,"mla_layers":mla,"mla_max_abs":max_mla})
}

fn room(state:&DecodeStates,count:i64) {
    for layer in &state.0 {if let LayerState::MlaLatent(s)=layer {s.ensure_room(count);return;}}
    panic!("correction gate requires latent MLA state");
}

fn prediction(output:&Tensor,row:usize)->i64 {
    if output.kind()==Kind::Int64 {output.int64_value(&[row as i64])}
    else {output.get(row as i64).argmax(-1,false).int64_value(&[])}
}

fn path_to(parents:&[Option<usize>],node:usize)->Vec<usize> {
    let mut path=Vec::new();let mut current=Some(node);
    while let Some(node)=current {path.push(node);current=parents[node];}
    path.reverse();path
}

fn candidates(a:&Candidates,b:&Candidates,context:&str)->Value {
    exact(&a.ids,&b.ids,&format!("{context} drafter ids"));
    exact(&a.hidden,&b.hidden,&format!("{context} drafter hidden"));
    exact(&a.edges,&b.edges,&format!("{context} drafter edges"));
    exact(&a.unary,&b.unary,&format!("{context} drafter unary"));
    assert_eq!(a.path,b.path,"{context} drafter path");
    json!({"ids_equal":true,"hidden_raw_bits":true,"edges_raw_bits":true,"unary_raw_bits":true,"path":a.path})
}

fn proposal(eng:&Engine,drafter:&Drafter,a:&Context,b:&Context,anchor:i64,context:&str)->Value {
    assert!(a.equal(b),"{context} drafter contexts before proposal");
    let before_a=a.snapshot();let before_b=b.snapshot();
    let x=drafter.propose(a,anchor,&eng.w);let y=drafter.propose(b,anchor,&eng.w);
    let result=candidates(&x,&y,context);
    assert!(a.equal(&before_a) && b.equal(&before_b),"{context} proposal mutated context");result
}

/// Own only two selected states here, never every reconstructed node. Features
/// for a selected node include its complete ancestor path, including anchor.
fn continue_selected(eng:&mut Engine,drafter:&Drafter,base_context:&Context,
    mut a:DecodeStates,mut b:DecodeStates,fa:&Tensor,fb:&Tensor,mut anchor:i64,label:&str)->Value {
    exact(fa,fb,&format!("{label} accepted features"));
    let mut ca=base_context.snapshot();let mut cb=base_context.snapshot();
    let base_context_copy=base_context.snapshot();
    if fa.size()[0]>0 {drafter.append(&mut ca,fa);drafter.append(&mut cb,fb);}
    assert!(ca.equal(&cb),"{label} selected append contexts");
    let selected_proposal=proposal(eng,drafter,&ca,&cb,anchor,&format!("{label} selected"));
    let mut steps=Vec::new();
    for step in 0..2 {
        room(&a,1);room(&b,1);let ids=Tensor::from_slice(&[anchor]).to_device(eng.w.device);
        let (oa,sa)={let _reference=IndexReference::enter();eng.step_record(&ids,&mut a,true)};
        let (ob,sb)=eng.step_record(&ids,&mut b,true);
        exact(&oa,&ob,&format!("{label} continuation step{step} logits"));
        let (sa,sb)=features(&sa,&sb,&format!("{label} continuation step{step}"));
        let state=state_report(&a,&b,&format!("{label} continuation step{step}"));
        drafter.append(&mut ca,&sa);drafter.append(&mut cb,&sb);
        assert!(ca.equal(&cb),"{label} continuation step{step} context");
        let next=oa.argmax(-1,false).view([-1]).int64_value(&[0]);
        steps.push(json!({"step":step,"token":anchor,"next":next,"logits_raw_bits":true,"features_raw_bits":true,
            "state":state,"drafter_context_equal":true}));anchor=next;
    }
    let final_proposal=proposal(eng,drafter,&ca,&cb,anchor,&format!("{label} continued"));
    assert!(base_context.equal(&base_context_copy),"{label} append changed base context");
    json!({"accepted_feature_rows":fa.size()[0],"selected_proposal":selected_proposal,"steps":steps,
        "final_proposal":final_proposal,"base_context_unchanged":true,"final_context_len":ca.len})
}

// Keep all Full/Deferred tree outputs scoped inside one case, especially T9.
fn run_case(eng:&mut Engine,drafter:&Drafter,base:&DecodeStates,base_copy:&DecodeStates,base_context:&Context,
    prefix_next:i64,ids:&[i64],parents:&[Option<usize>],name:&str,selection:TreeSelection,head_ids:bool,
    out:&Path,rank:usize,report:&mut Value) {
    let case_index=report["cases"].as_array().unwrap().len();
    let label=format!("{name}/{selection:?}/{}",if head_ids{"top1"}else{"logits"});
    let expected_deferred=parents.len()<=8 && parents.iter().enumerate().all(|(i,p)|*p==i.checked_sub(1));
    report["cases"].as_array_mut().unwrap().push(json!({"name":name,"selection":format!("{selection:?}"),
        "head":if head_ids{"top1"}else{"logits"},"ids":ids,"parents":parents,"gate":false,
        "expected_deferred":expected_deferred,"nodes":[],"continuations":[]}));save(out,rank,report);
    let max_depth=(0..parents.len()).map(|node|path_to(parents,node).len()).max().unwrap();
    room(base,max_depth as i64+2);
    let input=Tensor::from_slice(ids).to_device(eng.w.device);
    // The legacy API is deliberately called with CORRECTION_REPLAY still on:
    // its contract MUST return full states even under the candidate setting.
    let candidate_shared=crate::shared_gu::enabled();
    report["cases"][case_index]["shared_gu_reference"]=json!(false);
    report["cases"][case_index]["shared_gu_candidate"]=json!(candidate_shared);
    let candidate_index=crate::dsa_index::enabled();
    let candidate_topk=crate::dsa_topk::enabled();
    let candidate_position=crate::dsa_position::enabled();
    report["cases"][case_index]["dsa_position_reference"]=json!(false);
    report["cases"][case_index]["dsa_position_candidate"]=json!(candidate_position);
    report["cases"][case_index]["dsa_topk_reference"]=json!(false);
    report["cases"][case_index]["dsa_topk_candidate"]=json!(candidate_topk);
    let candidate_small=crate::dense_fp8::small_kernel_signature();
    report["cases"][case_index]["fp8_small_reference"]=json!([false,false]);
    report["cases"][case_index]["fp8_small_candidate"]=json!(candidate_small);
    report["cases"][case_index]["dsa_index_reference"]=json!(false);
    report["cases"][case_index]["dsa_index_candidate"]=json!(candidate_index);
    let (gold,full,full_features)={
        let _reference=IndexReference::enter();
        if head_ids {eng.tree_forward_predictions_selected(&input,base,parents,true,selection,true)}
        else {eng.tree_forward_selected(&input,base,parents,true,selection,true)}
    };
    assert_eq!(crate::dsa_position::enabled(),candidate_position,"reference did not restore position capture mode");
    assert_eq!(crate::dsa_topk::enabled(),candidate_topk,"reference did not restore candidate TopK batch mode");
    assert_eq!(crate::dsa_index::enabled(),candidate_index,"reference did not restore candidate DSA index mode");
    assert_eq!(crate::dense_fp8::small_kernel_signature(),candidate_small,"reference did not restore FP8 small candidate flags");
    assert_eq!(crate::shared_gu::enabled(),candidate_shared,"reference did not restore shared GU candidate flag");
    let reference=VerifierStates::Full(full);
    let (actual,candidate,candidate_features)=eng.verifier_forward_selected(&input,base,parents,true,selection,true,head_ids);
    exact(&actual,&gold,&format!("{label} output"));
    let (full_features,candidate_features)=features(&full_features,&candidate_features,&label);
    assert_eq!(candidate.len(),parents.len());assert_eq!(reference.len(),parents.len());
    let kda_layers=base.0.iter().filter(|s|matches!(s,LayerState::Kda(_))).count();
    let (deferred,records)=match &candidate {
        VerifierStates::Full(_)=>(false,0),
        VerifierStates::Deferred{layers,..}=>(true,layers.iter().filter(|s|matches!(s,Layer::Kda(_))).count()),
    };
    assert_eq!(deferred,expected_deferred,"{label} silently selected wrong state mode");
    if deferred {assert_eq!(records,kda_layers,"{label} KDA layer unexpectedly fell back");}
    report["cases"][case_index]["output_raw_bits"]=json!(true);
    report["cases"][case_index]["features_raw_bits"]=json!(true);
    report["cases"][case_index]["deferred_records"]=json!(records);
    let deferred_conv=match &candidate {
        VerifierStates::Full(_)=>0,
        VerifierStates::Deferred{layers,..}=>layers.iter().filter(|s|matches!(s,Layer::Kda(r) if matches!(&r.conv,crate::kda_correction::ConvRecord::Deferred{..}))).count(),
    };
    let expected_conv=if deferred && std::env::var("GLM53_KDA_CONV_DEFERRED").as_deref()==Ok("1"){kda_layers}else{0};
    assert_eq!(deferred_conv,expected_conv,"{label} silently missed conv storage mode");
    report["cases"][case_index]["deferred_conv_records"]=json!(deferred_conv);
    report["cases"][case_index]["legacy_full_api_under_candidate_flag"]=json!(true);
    // None means no selected target node, distinct from anchor-only Some(0).
    {
        let a=reference.materialize(base,None);let b=candidate.materialize(base,None);
        let none=state_report(&a,&b,&format!("{label} None parity"));
        state_report(&a,base,&format!("{label} None base"));state_report(&b,base,&format!("{label} None candidate base"));
        report["cases"][case_index]["none_snapshot_base"]=none;
    }
    save(out,rank,report);
    let mut selected=vec![0,parents.len().saturating_sub(1)/2];selected.sort_unstable();selected.dedup();
    for node in 0..parents.len() {
        let a=reference.materialize(base,Some(node));let b=candidate.materialize(base,Some(node));
        let state=state_report(&a,&b,&format!("{label} node{node}"));
        report["cases"][case_index]["nodes"].as_array_mut().unwrap().push(json!({"node":node,"state":state}));
        save(out,rank,report);
        if selected.contains(&node) {
            let path=path_to(parents,node);
            let index=Tensor::from_slice(&path.iter().map(|&i|i as i64).collect::<Vec<_>>()).to_device(eng.w.device);
            let fa=full_features.index_select(0,&index);let fb=candidate_features.index_select(0,&index);
            let continuation=continue_selected(eng,drafter,base_context,a,b,&fa,&fb,prediction(&gold,node),&format!("{label} node{node}"));
            // In-place real steps may not modify the base or saved records.
            let restored_a=reference.materialize(base,Some(node));let restored_b=candidate.materialize(base,Some(node));
            state_report(&restored_a,&restored_b,&format!("{label} node{node} rematerialize after continuation"));
            state_report(base,base_copy,&format!("{label} base after continuation"));
            report["cases"][case_index]["continuations"].as_array_mut().unwrap().push(json!({"selected_node":node,
                "ancestor_nodes":path,"result":continuation,"record_rematerialization_exact":true}));save(out,rank,report);
        }
    }
    // One real None continuation per same-mode pair also proves snapshot
    // ownership rather than merely equal initial bytes.
    if name=="chain1" {
        let empty=Tensor::empty([0,20480],(Kind::Float,eng.w.device));
        let result=continue_selected(eng,drafter,base_context,reference.materialize(base,None),candidate.materialize(base,None),
            &empty,&empty,prefix_next,&format!("{label} None"));
        state_report(base,base_copy,&format!("{label} base after None continuation"));
        report["cases"][case_index]["none_continuation"]=result;
    }
    state_report(base,base_copy,&format!("{label} immutable base"));
    report["cases"][case_index]["base_unchanged"]=json!(true);report["cases"][case_index]["gate"]=json!(true);
    save(out,rank,report);eprintln!("[correction-check] rank{rank} {label} PASS");
}

pub fn correction_check(eng:&mut Engine,drafter:&Drafter,out:&Path) {
    let _env=Restore::new();let _guard=tch::no_grad_guard();let tp=crate::tp::world();
    std::fs::create_dir_all(out).unwrap();
    let mut report=json!({"gate":false,"rank":tp.rank,"world":tp.world,
        "scope":"real full model; legacy DSA index and FP8 small flags off + Full versus incoming candidate flags + verifier Deferred; same quantized weights/scales/precision, head/selection modes matched; correctness only",
        "execution":"eager; graph replay requires the separate ChainGraph qualification gate",
        "dataflow_flags":(["GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_SHARED_GU_FUSED","GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD","GLM53_DSA_INDEX_FUSED","GLM53_KDA_CONV_DEFERRED","GLM53_MOE_INPUT_HALF_REUSE","GLM53_MHC_POST_FOUR_STREAMS","GLM53_MHC_POST_PACKED"].into_iter().map(|k|(k,std::env::var(k).ok())).collect::<std::collections::BTreeMap<_,_>>()),
        "flags":FLAGS.iter().map(|&k|(k,"1")).collect::<std::collections::BTreeMap<_,_>>(),
        "cases":[]});save(out,tp.rank,&report);
    assert_eq!(tp.world,2,"correction full-model gate is TP2");assert!(eng.w.device.is_cuda());
    assert!(crate::mla_latent::enabled());
    for &flag in FLAGS {std::env::set_var(flag,"1");}
    let misses=eng.fast.as_ref().expect("correction gate requires resident experts").misses;
    let refs:Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let prompt:Vec<i64>=refs["hello"]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let tokens:Vec<i64>=refs["hello"]["text_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();assert!(tokens.len()>=9);
    let (prefix_logits,base,prefix_features)=eng.prefill_record(&Tensor::from_slice(&prompt).to_device(eng.w.device),None,true);
    assert_eq!(prefix_features.len(),5);let prefix_features=Tensor::cat(&prefix_features,1);
    assert_eq!(prefix_features.size(),[prompt.len() as i64,20480]);
    let prefix_next=prediction(&prefix_logits,prefix_logits.size()[0] as usize-1);
    let base_copy=snapshot(&base);let mut base_context=drafter.empty_context();drafter.append(&mut base_context,&prefix_features);
    let mut nonzero=0usize;for state in &base.0 {if let LayerState::Kda(k)=state {if k.h.abs().max().double_value(&[])>0. {nonzero+=1;}}}
    assert!(nonzero>0,"prefix must establish nonzero KDA state");
    report["prefix"]=json!({"name":"hello","tokens":prompt,"next":prefix_next,"nonzero_kda_layers":nonzero,
        "drafter_context_len":base_context.len});save(out,tp.rank,&report);
    let chain=|n:usize|(0..n).map(|i|i.checked_sub(1)).collect::<Vec<_>>();
    let topologies=vec![("chain1",chain(1)),("chain2",chain(2)),("chain4",chain(4)),("chain8",chain(8)),
        ("siblings",vec![None,Some(0),Some(0),Some(1)]),("multiroot",vec![None,Some(0),None,Some(2)]),("chain9_fallback",chain(9))];
    for (name,parents) in topologies {
        let mut ids=tokens[..parents.len()].to_vec();ids[0]=prefix_next;
        for selection in [TreeSelection::Ranked,TreeSelection::AllVisible] {for head_ids in [false,true] {
            run_case(eng,drafter,&base,&base_copy,&base_context,prefix_next,&ids,&parents,name,selection,head_ids,out,tp.rank,&mut report);
            // Synchronize only this diagnostic gate so the previous large tree
            // is retired before the next case allocates its full-state oracle.
            tch::Cuda::synchronize(0);
        }}
    }
    assert_eq!(report["cases"].as_array().unwrap().len(),28);
    let cases=report["cases"].as_array().unwrap();
    assert!(cases.iter().all(|c|c["gate"].as_bool()==Some(true)));
    let nodes:usize=cases.iter().map(|c|c["nodes"].as_array().unwrap().len()).sum();
    let selected:usize=cases.iter().map(|c|c["continuations"].as_array().unwrap().len()).sum();
    let none=cases.iter().filter(|c|c.get("none_continuation").is_some()).count();
    assert_eq!((nodes,selected,none),(128,48,4));
    report["coverage"]=json!({"mode_topology_cases":28,"selected_node_states":nodes,"selected_continuations":selected,
        "none_continuations":none,"real_steps_per_continuation":2});
    assert_eq!(misses,eng.fast.as_ref().unwrap().misses,"qualification evicted/loaded expert weights");
    report["resident_misses_unchanged"]=json!(true);report["gate"]=json!(true);save(out,tp.rank,&report);
    eprintln!("[correction-check] rank{} PASS all 28 mode/topology gates; every node + selected continuation + drafter parity",tp.rank);
}
