//! Diagnostic only. Local helper gate and resident real-layer producer gate.
use super::{MaskedRows,RankedRows};
use crate::dsa::TreeSelection;
use serde_json::{json,Value};
use std::{path::Path,ffi::OsString};
use tch::{Tensor,Kind,Device};

struct Environment(Vec<(&'static str,Option<OsString>)>);
impl Environment {fn new()->Self {Self(["GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED",
    "GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT"].into_iter().map(|k|(k,std::env::var_os(k))).collect())}}
impl Drop for Environment {fn drop(&mut self){for (k,v) in &self.0 {
    if let Some(v)=v {std::env::set_var(k,v);}else{std::env::remove_var(k);}
}}}
fn candidate(on:bool){std::env::set_var("GLM53_DSA_TOPK_BATCH",if on{"1"}else{"0"});}
fn bits(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.size(),b.size(),"{label} shape");assert_eq!(a.kind(),b.kind(),"{label} kind");
    let kind=match a.kind(){Kind::Float=>Kind::Int,Kind::Half|Kind::BFloat16=>Kind::Int16,Kind::Double=>Kind::Int64,_=>a.kind()};
    assert!(a.contiguous().view_dtype(kind).equal(&b.contiguous().view_dtype(kind)),"DSA batch raw bits: {label}");
}
fn save(path:&Path,value:Value){std::fs::write(path,serde_json::to_string_pretty(&value).unwrap()).unwrap();}

struct BatchOutput {masked:Tensor,values:Tensor,ranked:RankedRows,tokens:Vec<Tensor>}
/// Production helper: copied oldpos or captured sidecar, followed by mutable
/// len advance before delayed expand. Debug outputs are not timing inputs.
fn batch(scores:&Tensor,lengths:&Tensor)->BatchOutput {
    let t=scores.size()[0];let mut b=MaskedRows::new(t,scores.size()[1],scores.device());
    for row in 0..t {
        let mut len=lengths.get(row);let oldpos=b.position_input(&len);
        b.push(&scores.get(row),oldpos);len.copy_(&(&len+1));
    }
    let masked=b.masked.shallow_clone();let (values,ranked)=b.finish_outputs();
    let tokens=(0..t).map(|i|ranked.tokens(i)).collect();BatchOutput{masked,values,ranked,tokens}
}
/// Independent original sequence: mask -> one-row sorted topk -> expand.
fn old(scores:&Tensor,positions:&Tensor)->Vec<(Tensor,Tensor,Tensor,Tensor)> {
    (0..scores.size()[0]).map(|i| {
        let x=scores.get(i);let pos=positions.get(i);let masked=Tensor::empty_like(&x);
        crate::dsa_index::mask_into(&x,&pos,&masked);
        let (values,ids)=masked.topk(512.min(x.size()[0]),0,true,true);
        let tokens=Tensor::empty([4*ids.size()[0]+3],(Kind::Int64,x.device()));
        crate::dsa_index::expand_into(&ids,&pos,&tokens);(masked,values,ids,tokens)
    }).collect()
}
fn compare(out:&BatchOutput,gold:&[(Tensor,Tensor,Tensor,Tensor)],context:&str) {
    assert_eq!(out.tokens.len(),gold.len());
    for (i,(mask,values,ids,tokens)) in gold.iter().enumerate() {
        bits(&out.masked.get(i as i64),mask,&format!("{context} row{i} masked rows"));bits(&out.values.get(i as i64),values,&format!("{context} row{i} sorted values including NaN payload and zero sign"));
        bits(&out.ranked.selected.get(i as i64),ids,&format!("{context} row{i} sorted indices including tie order"));bits(&out.tokens[i],tokens,&format!("{context} row{i} expanded slots and partial tail"));
    }
}
fn input(rows:i64,width:i64,positions:&[i64],mode:usize,turn:usize)->Tensor {
    let special=[0x7f800000u32,0xff800000,0x7fc12345,0xffc45678,0,0x80000000,
        0x7f7fffff,0xff7fffff,0x00800000,1,0x80000001];
    let raw:Vec<i32>=(0..rows).flat_map(|r|(0..width).map(move |c| {
        let complete=(positions[r as usize]+1)/4;
        let value=match mode {
            0=>(((c as usize*37+r as usize*17+turn*13)%1009) as f32-504.)/257.,
            1=>f32::from_bits(if (c+r+turn as i64)%2==0{0}else{0x80000000}),
            2=>f32::MIN,
            3=>f32::from_bits(special[(c as usize+r as usize+turn)%special.len()]),
            4=>if c>=complete {f32::from_bits(0x7fc54321)}else{((c+turn as i64)%7) as f32},
            _=>unreachable!(),
        };value.to_bits() as i32
    })).collect();Tensor::from_slice(&raw).view([rows,width]).view_dtype(Kind::Float).to_device(Device::Cuda(0))
}
fn starts(width:i64)->Vec<i64> {
    let mut values=vec![0,1,2,3,4,127,2043,2044,2046,2047,2048,2050,2051,2052];
    values.retain(|p|*p<width*4);values.extend([width*4-1,0,2,0]);values
}
fn local_case(rows:i64,width:i64)->Value {
    let dev=Device::Cuda(0);let guard=31;
    let source=Tensor::empty([rows,width+2*guard],(Kind::Float,dev));let mut scores=source.narrow(1,guard,width);
    let mut positions=Tensor::zeros([rows,1],(Kind::Int64,dev));
    let _=source.shallow_clone().view_dtype(Kind::Int).fill_(0x7fc65432i64);
    scores.copy_(&input(rows,width,&vec![0;rows as usize],0,0));
    let _=batch(&scores,&positions);let _=positions.fill_(0);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let captured=batch(&scores,&positions);crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    let mut records=Vec::new();let mut previous:Option<(Vec<Tensor>,Vec<Tensor>)>=None;
    for (turn,start) in starts(width).into_iter().enumerate() {for mode in 0..5 {
        let host:Vec<i64>=(0..rows).map(|r|(start+r%3).min(width*4-1)).collect();
        let dynamic=Tensor::from_slice(&host).view([rows,1]).to_device(dev);
        positions.copy_(&dynamic);scores.copy_(&input(rows,width,&host,mode,turn));let before=source.copy();
        let gold=old(&scores,&dynamic);
        let _=captured.masked.shallow_clone().view_dtype(Kind::Int).fill_(0x7fa7c0dei64);
        for out in &captured.tokens{let _=out.shallow_clone().fill_(i64::MIN);}
        graph.replay();let context=format!("T{rows}/C{width}/pos{start}/mode{mode}/turn{turn}");compare(&captured,&gold,&context);bits(&positions,&(&dynamic+1),"only mutable length advances");bits(&source,&before,"score backing/input guards unchanged");
        positions.copy_(&dynamic);let eager=batch(&scores,&positions);compare(&eager,&gold,&context);bits(&positions,&(&dynamic+1),"eager length advance");
        if let Some((a,b))=&previous {for (x,y) in a.iter().zip(b){bits(x,y,"independent previous output survives graph replay");}}
        previous=Some((eager.tokens,gold.iter().map(|g|g.3.copy()).collect()));
        records.push(json!({"start":start,"positions":host,"pattern":mode,"helper_rows":rows,"mask_values_indices_tokens_exact":true,
            "input_guards_unchanged":true,"changed_input_graph_exact":true,"oldpos_independent_of_advanced_len":true}));
    }}
    // A second graph and different input owners must not overwrite the first.
    if rows==8 && width==5120 {
        let other_scores=Tensor::ones([2,width],(Kind::Float,dev));let other_pos=Tensor::zeros([2,1],(Kind::Int64,dev));
        let _=batch(&other_scores,&other_pos);let _=other_pos.shallow_clone().fill_(0);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let other=batch(&other_scores,&other_pos);crate::tp::graph::end().unwrap();let other_graph=crate::tp::graph::Owned::take();
        let original:Vec<_>=captured.tokens.iter().map(Tensor::copy).collect();other_graph.replay();
        for (a,b) in captured.tokens.iter().zip(&original){bits(a,b,"second graph owns separate workspace");}
        drop(other_graph);drop(other);drop(other_pos);drop(other_scores);
    }
    let expected:Vec<_>=captured.tokens.iter().map(Tensor::copy).collect();
    tch::Cuda::synchronize(0);drop(graph);drop(source);drop(scores);drop(positions);
    // RankedRows explicitly owns selected and oldpos even after source/graph
    // owners disappear. An expand after drop must still produce the last row.
    for (row,gold) in expected.iter().enumerate(){bits(&captured.ranked.tokens(row as i64),gold,"pending expand after graph/source drop");}
    json!({"rows":rows,"width":width,"records":records,"owner_after_drop":true,"same_graph_long_to_short":true})
}
fn eligibility()->Value {
    let d=Device::Cuda(0);let x=Tensor::zeros([8,16],(Kind::Float,d));let pools=Tensor::zeros([513,128],(Kind::Float,d));let pos=Tensor::zeros([1],(Kind::Int64,d));
    std::env::remove_var("GLM53_DSA_TOPK_BATCH");assert!(!super::enabled());candidate(true);
    for t in 2..=8 {assert!(super::eligible(TreeSelection::Ranked,&x.narrow(0,0,t),&pools,&pos));}
    assert!(!super::eligible(TreeSelection::Ranked,&x.narrow(0,0,1),&pools,&pos));
    assert!(!super::eligible(TreeSelection::Ranked,&Tensor::zeros([9,16],(Kind::Float,d)),&pools,&pos));
    assert!(!super::eligible(TreeSelection::AllVisible,&x,&pools,&pos));
    assert!(!super::eligible(TreeSelection::Ranked,&x.to_kind(Kind::Half),&pools,&pos));
    let strided=Tensor::zeros([8,32],(Kind::Float,d)).slice(1,0,32,2);
    assert_eq!(strided.size(),x.size());assert!(!strided.is_contiguous());
    assert!(!super::eligible(TreeSelection::Ranked,&strided,&pools,&pos));
    assert!(!super::eligible(TreeSelection::Ranked,&x,&pools,&pos.to_kind(Kind::Int)));
    assert!(!super::eligible(TreeSelection::Ranked,&x.to_device(Device::Cpu),&pools.to_device(Device::Cpu),&pos.to_device(Device::Cpu)));
    std::env::set_var("GLM53_DSA_INDEX_FUSED","0");assert!(!super::eligible(TreeSelection::Ranked,&x,&pools,&pos));
    std::env::set_var("GLM53_DSA_INDEX_FUSED","1");candidate(false);assert!(!super::eligible(TreeSelection::Ranked,&x,&pools,&pos));candidate(true);
    json!({"default_off":true,"rows2_through8_eligible":true,"T1_T9_allvisible_indexoff_dtype_stride_device_fallback":true})
}
pub(crate) fn run(out:&Path) {
    assert!(!crate::tp::is_tp());let before=crate::session::signature();let saved=Environment::new();let _guard=tch::no_grad_guard();tch::set_num_threads(4);
    std::fs::create_dir_all(out).unwrap();let path=out.join("dsa-topk-local.json");save(&path,json!({"gate":false,"complete":false}));
    std::env::set_var("GLM53_DSA_INDEX_FUSED","1");candidate(true);let eligibility=eligibility();let mut cases=Vec::new();
    for rows in 2..=8 {for width in [511i64,512,513,5120] {cases.push(local_case(rows,width));
        save(&path,json!({"gate":false,"complete":false,"cases":cases}));}}
    let records:usize=cases.iter().map(|c|c["records"].as_array().unwrap().len()).sum();
    drop(saved);assert_eq!(before,crate::session::signature(),"local TopK probe leaked configuration");
    save(&path,json!({"gate":true,"complete":true,"shapes":cases.len(),"records":records,"cases":cases,"eligibility":eligibility,
        "environment_restored":true,"scope":"actual Rust MaskedRows helper and old CUDA mask/one-row sorted topk/expand oracle; no model, TP or timing claim"}));
}

fn index_bits(a:&crate::dsa::State,b:&crate::dsa::State,len:i64) {
    let pools=(len+3)/4;bits(&a.pools.narrow(0,0,pools),&b.pools.narrow(0,0,pools),"real logical pools");
    bits(&a.tail_k,&b.tail_k,"real tail keys");bits(&a.tail_gate,&b.tail_gate,"real tail gates");
}
fn mla_bits(a:&crate::mla_latent::State,b:&crate::mla_latent::State) {
    bits(&a.len,&b.len,"real state length");let len=a.len.int64_value(&[0]);
    bits(&a.latent.narrow(0,0,len),&b.latent.narrow(0,0,len),"real logical latent");index_bits(&a.index,&b.index,len);
}
/// Called solely from Engine's const diagnostic monomorphization, at the actual
/// MLA layer z/base. Uses checkpoint weights and original append_projected as
/// independent producer+selection oracle; no synthetic x replaces model z.
pub(crate) fn real_layer(layer:usize,w:&crate::weights::MlaWeights,x:&Tensor,base:&crate::mla_latent::State,parents:&[Option<usize>])->Value {
    let _env=Environment::new();candidate(true);assert!(crate::dsa_index::enabled());
    let capture_candidate=crate::dsa_position::enabled();
    assert!(super::eligible(TreeSelection::Ranked,x,&base.index.pools,&base.len));
    let p=crate::mla_latent::dsa_topk_project(w,x);let weights=w.indexer.as_ref().unwrap();let t=x.size()[0];
    // Poison actual future pool storage, not just precomputed score values.
    // Keep the current provisional pool: append_keys still needs its tail.
    let index_base=base.index.snapshot();let live=(base.len.int64_value(&[0])+3)/4;
    let _=index_base.pools.narrow(0,live,index_base.pools.size()[0]-live).fill_(f64::NAN);
    let mut old_states:Vec<crate::dsa::State>=Vec::new();let mut new_states:Vec<crate::dsa::State>=Vec::new();
    let mut lengths:Vec<Tensor>=Vec::new();let mut gold=Vec::new();let mut old_tokens=Vec::new();
    let mut b=MaskedRows::new(t,base.index.pools.size()[0],x.device());let original=base.snapshot();
    for (row,&parent) in parents.iter().enumerate() {
        let mut a=parent.map_or(&index_base,|i|&old_states[i]).snapshot();
        let mut c=parent.map_or(&index_base,|i|&new_states[i]).snapshot();
        let mut len=parent.map_or(&base.len,|i|&lengths[i]).copy();let pos=len.copy();
        let ids=a.append_projected(weights,&p,row as i64,&pos);
        let scores=c.append_score(weights,&p,row as i64,&pos);
        let masked=Tensor::empty_like(&scores);crate::dsa_index::mask_into(&scores,&pos,&masked);
        let (values,selected)=masked.topk(512.min(scores.size()[0]),0,true,true);
        let tokens=Tensor::empty([4*selected.size()[0]+3],(Kind::Int64,x.device()));crate::dsa_index::expand_into(&selected,&pos,&tokens);
        bits(&ids,&tokens,"real original append_projected versus copied score seam");
        let candidate_pos=b.position_input(&len);b.push(&scores,candidate_pos);len.copy_(&(&len+1));index_bits(&a,&c,len.int64_value(&[0]));
        let active=(len.int64_value(&[0])+3)/4;
        for state in [&a,&c] {assert!(state.pools.narrow(0,active,state.pools.size()[0]-active).isnan().all().int64_value(&[])!=0,"future pool suffix poison unexpectedly overwritten");}
        old_states.push(a);new_states.push(c);lengths.push(len);old_tokens.push(ids);gold.push((masked,values,selected,tokens));
    }
    let masked=b.masked.shallow_clone();let (values,ranked)=b.finish_outputs();
    let tokens=(0..t).map(|i|ranked.tokens(i)).collect();let actual=BatchOutput{masked,values,ranked,tokens};compare(&actual,&gold,&format!("actual layer{layer}/T{t}"));
    for (a,b) in actual.tokens.iter().zip(&old_tokens){bits(a,b,"real batch ids versus original producer+ranked");}
    std::env::set_var("GLM53_DSA_POSITION_CAPTURE","0");
    candidate(false);let (old_output,old)=crate::mla_latent::tree_impl_selected(w,x,base,parents,TreeSelection::Ranked,true);
    candidate(true);
    // Isolate new capture from L4 TopK: keep ranking enabled on both arms.
    let l4_reference=capture_candidate.then(||crate::mla_latent::tree_impl_selected(w,x,base,parents,TreeSelection::Ranked,true));
    std::env::set_var("GLM53_DSA_POSITION_CAPTURE",if capture_candidate{"1"}else{"0"});
    let (new_output,new)=crate::mla_latent::tree_impl_selected(w,x,base,parents,TreeSelection::Ranked,true);
    if let Some((l4_output,l4_states))=l4_reference {
        bits(&l4_output,&new_output,"actual MLA with TopK fixed on: copy versus sidecar capture");
        for (a,b) in l4_states.iter().zip(&new){mla_bits(a,b);}
    }
    bits(&old_output,&new_output,"real complete MLA output off/on");
    for (a,b) in old.iter().zip(&new){mla_bits(a,b);assert_ne!(b.len.data_ptr(),base.len.data_ptr());}
    for (i,a) in new.iter().enumerate(){for b in new.iter().skip(i+1){assert_ne!(a.len.data_ptr(),b.len.data_ptr());assert_ne!(a.index.pools.data_ptr(),b.index.pools.data_ptr());}}
    mla_bits(base,&original);
    json!({"layer":layer,"rows":t,"parents":parents,"base_len":base.len.int64_value(&[0]),"pools":base.index.pools.size()[0],
        "real_model_activation":true,"checkpoint_index_weights":true,"helper_rows":t,"metadata_eligible":true,
        "sorted_values_indices_exact":true,"original_append_ids_exact":true,"pool_tail_len_exact":true,"future_pool_nan_poison":true,
        "complete_mla_output_state_exact":true,"base_and_branch_isolation":true,
        "position_capture_reference":false,"position_capture_candidate":capture_candidate,"position_capture_isolated_exact":capture_candidate})
}
pub(crate) fn real_check(eng:&mut crate::forward::Engine,suite:&Path,out:&Path) {
    let before=crate::session::signature();let before_spec=crate::spec_session::signature();let saved=Environment::new();let _guard=tch::no_grad_guard();
    assert!(super::enabled(),"real TopK gate requires explicit candidate1");assert!(crate::dsa_index::enabled());
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
    std::fs::create_dir_all(out).unwrap();let rank=crate::tp::world().rank;let path=out.join(format!("dsa-topk-real-rank{rank}.json"));
    save(&path,json!({"gate":false,"complete":false}));
    let suite_path=suite;let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite_path).unwrap()).unwrap();
    let case=suite["cases"].as_array().unwrap().iter().find(|c|c["reference_ids"].as_array().is_some_and(|v|v.len()>=8) && c["prompt_ids"].as_array().is_some_and(|v|!v.is_empty()&&v.len()<=512)).expect("real TopK qualification needs a bounded (1..512) prompt with >=8 reference tokens");
    let prefix:Vec<i64>=case["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let ids:Vec<i64>=case["reference_ids"].as_array().unwrap().iter().take(8).map(|v|v.as_i64().unwrap()).collect();assert!(!prefix.is_empty());
    candidate(false);let (_,base)=eng.prefill(&Tensor::from_slice(&prefix).to_device(eng.w.device));candidate(true);
    let expected_layers=base.0.iter().filter(|s|matches!(s,crate::forward::LayerState::MlaLatent(_))).count();assert!(expected_layers>0);
    let mut cases=Vec::new();let mut topologies:Vec<Vec<Option<usize>>>=(2..=8).map(|t|(0..t).map(|i:usize|i.checked_sub(1)).collect()).collect();
    topologies.push(vec![None,Some(0),Some(0),Some(1),Some(2),Some(1),Some(2),Some(6)]);topologies.push(vec![None,None,Some(0),Some(1),None,Some(2),Some(3),Some(4)]);
    for parents in topologies {
        let mut records=Vec::new();let input=Tensor::from_slice(&ids[..parents.len()]).to_device(eng.w.device);
        eng.tree_dsa_topk_probe(&input,&base,&parents,&mut records);
        assert_eq!(records.len(),expected_layers,"real producer gate missed MLA layers");
        assert!(records.iter().all(|r|r["helper_rows"].as_u64()==Some(parents.len() as u64)));
        cases.push(json!({"parents":parents,"layers":records,"all_expected_mla_layers":expected_layers}));
        save(&path,json!({"gate":false,"complete":false,"cases":cases}));
    }
    drop(saved);assert_eq!(before,crate::session::signature());assert_eq!(before_spec,crate::spec_session::signature());
    save(&path,json!({"gate":true,"complete":true,"rank":rank,"suite_path":suite_path,"prompt_case":case["name"],"prompt_tokens":prefix.len(),"prompt_ids":prefix,"teacher_ids":ids,
        "cases":cases,"environment_restored":true,"expected_mla_layers":expected_layers,
        "scope":"resident actual per-layer model z/base and checkpoint weights, T2..8 plus sibling/multiroot; exact local MLA score/rank/state/output, not full-model acceptance/performance or all long-prefix regimes"}));
}
