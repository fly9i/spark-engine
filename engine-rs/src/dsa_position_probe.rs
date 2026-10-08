//! Diagnostic only: old per-node copies versus same-stream mask capture.
//! Both arms retain identical batched TopK and integer expand operations.
use super::{MaskedRows,RankedRows};
use serde_json::{json,Value};
use std::{path::Path,ffi::OsString};
use tch::{Tensor,Kind,Device};
const GUARD:i64=17;
const MASK_CANARY:i64=0x7fa65432;
const POS_CANARY:i64=-0x123456789;
struct Environment(Vec<(&'static str,Option<OsString>)>);
impl Environment {fn new()->Self {Self(["GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED"]
    .into_iter().map(|k|(k,std::env::var_os(k))).collect())}}
impl Drop for Environment {fn drop(&mut self){for (k,v) in &self.0 {
    if let Some(v)=v {std::env::set_var(k,v);}else{std::env::remove_var(k);}
}}}
fn mode(on:bool){std::env::set_var("GLM53_DSA_POSITION_CAPTURE",if on{"1"}else{"0"});}
fn bits(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.size(),b.size(),"{label} shape");assert_eq!(a.kind(),b.kind(),"{label} dtype");
    let kind=if a.kind()==Kind::Float {Kind::Int}else{a.kind()};
    assert!(a.contiguous().view_dtype(kind).equal(&b.contiguous().view_dtype(kind)),"DSA position raw bits: {label}");
}
fn save(path:&Path,value:Value){std::fs::write(path,serde_json::to_string_pretty(&value).unwrap()).unwrap();}
struct Output {
    masked:Tensor,values:Tensor,ranked:RankedRows,tokens:Vec<Tensor>,
    mask_storage:Tensor,side_storage:Option<Tensor>,capture_used:bool,
}
/// Uses the production allocation choice, position_input, push and finish.
/// Output allocations are replaced by contiguous guarded views, with the same
/// actual shape/stride as production; the independent storage owners survive.
fn batch(scores:&Tensor,lengths:&Tensor)->Output {
    let t=scores.size()[0];let width=scores.size()[1];let dev=scores.device();
    let mut b=MaskedRows::new(t,width,dev);let capture_used=b.sidecar.is_some();
    let mask_storage=Tensor::empty([t*width+2*GUARD],(Kind::Float,dev));
    let _=mask_storage.view_dtype(Kind::Int).fill_(MASK_CANARY);
    b.masked=mask_storage.narrow(0,GUARD,t*width).view([t,width]);
    let side_storage=if capture_used {
        let storage=Tensor::full([t+2*GUARD],POS_CANARY,(Kind::Int64,dev));
        b.sidecar=Some(storage.narrow(0,GUARD,t));Some(storage)
    }else{None};
    for row in 0..t {
        let mut len=lengths.get(row);let pos=b.position_input(&len);
        assert_eq!(pos.data_ptr()==len.data_ptr(),capture_used,"actual position input branch");
        b.push(&scores.get(row),pos);
        assert_ne!(b.positions[row as usize].data_ptr(),len.data_ptr(),"deferred owner must never alias mutable len");
        // Regression trigger: mask capture MUST precede this state mutation.
        len.copy_(&(&len+1));
    }
    let masked=b.masked.shallow_clone();let (values,ranked)=b.finish_outputs();
    let tokens=(0..t).map(|i|ranked.tokens(i)).collect();
    Output{masked,values,ranked,tokens,mask_storage,side_storage,capture_used}
}
fn guards(out:&Output) {
    let size=out.mask_storage.size()[0];
    for offset in [0,size-GUARD] {
        assert!(out.mask_storage.narrow(0,offset,GUARD).view_dtype(Kind::Int).eq(MASK_CANARY).all().int64_value(&[])!=0,"mask guard modified");
    }
    if let Some(storage)=&out.side_storage {for offset in [0,storage.size()[0]-GUARD] {
        assert!(storage.narrow(0,offset,GUARD).eq(POS_CANARY).all().int64_value(&[])!=0,"sidecar guard modified");
    }}
}
fn compare(a:&Output,b:&Output,positions:&Tensor,label:&str) {
    assert!(!a.capture_used&&b.capture_used);
    bits(&a.masked,&b.masked,&format!("{label} mask"));bits(&a.values,&b.values,&format!("{label} values"));
    bits(&a.ranked.selected,&b.ranked.selected,&format!("{label} topk indices"));
    assert_eq!(a.tokens.len(),b.tokens.len());
    for (row,(x,y)) in a.tokens.iter().zip(&b.tokens).enumerate() {
        bits(x,y,&format!("{label} tokens row{row}"));
        bits(&a.ranked.positions[row],&positions.get(row as i64),"old independent position");
        bits(&b.ranked.positions[row],&positions.get(row as i64),"sidecar captured pre-append position");
    }
    bits(b.ranked._sidecar.as_ref().unwrap(),&positions.view([-1]),"sidecar full valid extent");
    guards(a);guards(b);
}
fn poison(out:&Output) {
    let _=out.masked.shallow_clone().view_dtype(Kind::Int).fill_(0x7fa7c0dei64);
    for t in &out.tokens {let _=t.shallow_clone().fill_(i64::MIN);}
    if let Some(sidecar)=&out.ranked._sidecar {let _=sidecar.shallow_clone().fill_(i64::MIN);}
}
fn input(rows:i64,width:i64,positions:&[i64],pattern:usize,turn:usize)->Tensor {
    let special=[0x7f800000u32,0xff800000,0x7fc12345,0xffc45678,0,0x80000000,
        0x7f7fffff,0xff7fffff,0x00800000,1,0x80000001];
    let raw:Vec<i32>=(0..rows).flat_map(|r|(0..width).map(move |c| {
        let complete=(positions[r as usize]+1)/4;
        let value=match pattern {
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
    let dev=Device::Cuda(0);
    let source=Tensor::empty([rows,width+2*GUARD],(Kind::Float,dev));
    let _=source.view_dtype(Kind::Int).fill_(MASK_CANARY);
    let mut scores=source.narrow(1,GUARD,width);
    let mut lengths=Tensor::zeros([rows,1],(Kind::Int64,dev));
    scores.copy_(&input(rows,width,&vec![0;rows as usize],0,0));
    // Two graphs fix the actual producer at capture; env on replay cannot
    // silently make both arms use the same path.
    mode(false);let _=batch(&scores,&lengths);let _=lengths.fill_(0);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let old=batch(&scores,&lengths);crate::tp::graph::end().unwrap();let old_graph=crate::tp::graph::Owned::take();
    mode(true);let _=lengths.fill_(0);let _=batch(&scores,&lengths);let _=lengths.fill_(0);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let captured=batch(&scores,&lengths);crate::tp::graph::end().unwrap();let new_graph=crate::tp::graph::Owned::take();
    let mut records=Vec::new();let mut previous:Option<(Vec<Tensor>,Vec<Tensor>)>=None;
    for (turn,start) in starts(width).into_iter().enumerate() {for pattern in 0..5 {
        let host:Vec<i64>=(0..rows).map(|r|(start+r%3).min(width*4-1)).collect();
        let dynamic=Tensor::from_slice(&host).view([rows,1]).to_device(dev);
        scores.copy_(&input(rows,width,&host,pattern,turn));let before=source.copy();
        poison(&old);poison(&captured);
        lengths.copy_(&dynamic);old_graph.replay();bits(&lengths,&(&dynamic+1),"old graph length advance");
        lengths.copy_(&dynamic);new_graph.replay();bits(&lengths,&(&dynamic+1),"capture graph length advance");
        let label=format!("T{rows}/C{width}/turn{turn}/pattern{pattern}");compare(&old,&captured,&dynamic,&label);
        bits(&source,&before,"source including NaNs and guards remains immutable");
        mode(false);lengths.copy_(&dynamic);let eager_old=batch(&scores,&lengths);
        mode(true);lengths.copy_(&dynamic);let eager_new=batch(&scores,&lengths);
        compare(&eager_old,&eager_new,&dynamic,"eager old/capture");
        bits(&eager_new.masked,&captured.masked,"eager versus replay mask");
        for (a,b) in eager_new.tokens.iter().zip(&captured.tokens){bits(a,b,"eager versus replay tokens");}
        if let Some((a,b))=&previous {for (x,y) in a.iter().zip(b){bits(x,y,"previous eager output remains independent");}}
        previous=Some((eager_new.tokens,eager_old.tokens));
        records.push(json!({"turn":turn,"positions":host,"pattern":pattern,"old_copy":true,"capture_used":true,
            "raw_mask_values_indices_tokens_exact":true,"mutable_len_advanced":true,"sidecar_pre_append_exact":true,
            "input_and_output_guards":true,"changed_graph_exact":true}));
    }}
    // A third graph uses distinct owners and a different token shape.
    let mut multiple_graphs=false;
    if rows==8 && width==5120 {
        let other_scores=Tensor::ones([2,width],(Kind::Float,dev));let other_len=Tensor::zeros([2,1],(Kind::Int64,dev));
        mode(true);let _=batch(&other_scores,&other_len);let _=other_len.shallow_clone().fill_(0);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let other=batch(&other_scores,&other_len);crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        assert_ne!(other.ranked._sidecar.as_ref().unwrap().data_ptr(),captured.ranked._sidecar.as_ref().unwrap().data_ptr());
        let saved:Vec<_>=captured.tokens.iter().map(Tensor::copy).collect();graph.replay();guards(&other);
        for (a,b) in captured.tokens.iter().zip(&saved){bits(a,b,"third graph cannot overwrite prior captured tokens");}
        tch::Cuda::synchronize(0);drop(graph);drop(other);drop(other_len);drop(other_scores);multiple_graphs=true;
    }
    let expected:Vec<_>=captured.tokens.iter().map(Tensor::copy).collect();
    // Only RankedRows retains sidecar and selected allocations at this point.
    // No graph, mutable input or probe guard-storage owner rescues its lifetime.
    let ranked=captured.ranked;
    drop(captured.masked);drop(captured.values);drop(captured.tokens);drop(captured.mask_storage);drop(captured.side_storage);
    tch::Cuda::synchronize(0);drop(old_graph);drop(new_graph);drop(source);drop(scores);drop(lengths);drop(old);
    for (row,gold) in expected.iter().enumerate(){bits(&ranked.tokens(row as i64),gold,"delayed expand after all producer owners drop");}
    json!({"rows":rows,"width":width,"records":records,"multiple_graphs_checked":multiple_graphs,
        "sidecar_owner_after_producer_drop":true,"long_to_short_replay":true})
}
pub(crate) fn run(out:&Path) {
    assert!(!crate::tp::is_tp());let before=crate::session::signature();let before_spec=crate::spec_session::signature();
    let env=Environment::new();let _guard=tch::no_grad_guard();tch::set_num_threads(4);
    std::fs::create_dir_all(out).unwrap();let path=out.join("dsa-position-local.json");save(&path,json!({"gate":false,"complete":false}));
    std::env::remove_var("GLM53_DSA_POSITION_CAPTURE");assert!(!crate::dsa_position::enabled());
    std::env::set_var("GLM53_DSA_TOPK_BATCH","1");std::env::set_var("GLM53_DSA_INDEX_FUSED","1");
    let mut cases=Vec::new();
    let shapes:Vec<_>=(2..=8).flat_map(|rows|[511i64,512,513,5120].map(move |width|(rows,width)))
        .chain([2i64,8].into_iter().flat_map(|rows|[1i64,255,256,257].map(move |width|(rows,width)))).collect();
    for (rows,width) in shapes {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(||local_case(rows,width))) {
            Ok(case)=>cases.push(case),
            Err(failure)=>{
                let message=failure.downcast_ref::<String>().cloned().or_else(||failure.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_else(||"non-string panic".to_owned());
                save(&path,json!({"gate":false,"complete":false,"cases":cases,"failed_rows":rows,"failed_width":width,"failure":message}));
                std::panic::resume_unwind(failure);
            }
        }
        save(&path,json!({"gate":false,"complete":false,"cases":cases}));
    }
    let records:usize=cases.iter().map(|c|c["records"].as_array().unwrap().len()).sum();
    drop(env);assert_eq!(before,crate::session::signature());assert_eq!(before_spec,crate::spec_session::signature());
    save(&path,json!({"gate":true,"complete":true,"shapes":cases.len(),"records":records,"cases":cases,
        "environment_restored":true,"default_off":true,"graph_arms":"separate old-copy and new-capture graphs",
        "scope":"same batched TopK/expand, mutable length update after mask, raw-bit mask and sidecar/lifetime gate; no model/TP/performance claim"}));
}
