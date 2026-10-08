//! Local CUDA gate for mask/index bookkeeping. No model load or score rewrite.
use super::{mask_into,expand_into,try_ranked};
use std::{path::Path,time::Instant};
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};

struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
impl Restore {fn new()->Self {Self(["GLM53_DSA_INDEX_FUSED","GLM53_DSA_SCORE_FUSED","GLM53_STATIC_TENSORS"]
    .into_iter().map(|k|(k,std::env::var_os(k))).collect())}}
impl Drop for Restore {fn drop(&mut self){for (k,v) in &self.0 {
    if let Some(v)=v {std::env::set_var(k,v);}else {std::env::remove_var(k);}
}}}
fn flag(on:bool){std::env::set_var("GLM53_DSA_INDEX_FUSED",if on{"1"}else{"0"});}
fn bits(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let same=if a.kind()==Kind::Float {a.contiguous().view_dtype(Kind::Int).equal(&b.contiguous().view_dtype(Kind::Int))}
        else {a.equal(b)};
    assert!(same,"DSA Ranked index bits/order mismatch: {label}");
}
fn positions(capacity:i64)->Vec<i64> {
    let mut p=vec![0,1,2,3,4,30,31,32,33,34,35,63,64,2046,2047,2048,2049,2050,2051,2052,20479,capacity-1];
    p.retain(|&x|x>=0&&x<capacity);p.sort_unstable();p.dedup();
    p.extend([capacity-1,0,2.min(capacity-1),1.min(capacity-1),0]);p
}
fn scores(pools:i64,pos:i64,mode:usize,turn:usize)->Tensor {
    let complete=(pos+1)/4;
    let extremes=[0x7f800000u32,0xff800000,0x7fc12345,0xffc45678,0,0x80000000,
        0x7f7fffff,0xff7fffff,0x00800000,1,0x80000001];
    let v:Vec<i32>=(0..pools).map(|i| {
        let x=match mode {
            0=>(((i as usize*37+turn*13)%1009) as f32-504.)/257.,
            1=>f32::from_bits(if (i as usize+turn)%2==0 {0}else{0x80000000}),
            2=>f32::MIN,
            3=>f32::from_bits(extremes[(i as usize+turn)%extremes.len()]),
            4=>if i>=complete {f32::from_bits(0x7fc54321)}else{((i+turn as i64)%7) as f32},
            _=>unreachable!(),
        };x.to_bits() as i32
    }).collect();
    Tensor::from_slice(&v).view_dtype(Kind::Float).to_device(Device::Cuda(0))
}

struct Constants {pools:Tensor,slots:Tensor,tail:Tensor}
impl Constants {fn new(n:i64)->Self {let d=(Kind::Int64,Device::Cuda(0));Self {
    pools:Tensor::arange(n,d),slots:Tensor::arange(4,d),tail:Tensor::arange(3,d)}}}
// Preserve the production baseline including its exact sorted ATen topk call.
fn old(scores:&Tensor,pos:&Tensor,c:&Constants)->(Tensor,Tensor,Tensor) {
    let len=pos+1;let complete=len.floor_divide_scalar(4);
    let masked=scores.masked_fill(&c.pools.lt_tensor(&complete).logical_not(),f32::MIN as f64);
    let ids=masked.topk(512.min(scores.size()[0]),0,true,true).1;
    let valid=ids.lt_tensor(&complete);
    let full=(&ids.unsqueeze(1)*4+&c.slots).masked_fill(&valid.logical_not().unsqueeze(1),-1).reshape([-1]);
    let tail=&complete*4+&c.tail;
    let tail=tail.masked_fill(&c.tail.ge_tensor(&len.remainder(4)),-1);
    (masked,ids,Tensor::cat(&[full,tail],0))
}
fn mathematical_indices(ids:&Tensor,pos:i64)->Tensor {
    let ids:Vec<i64>=Vec::try_from(ids.to_device(Device::Cpu)).unwrap();let len=pos+1;let complete=len/4;
    let mut tokens=Vec::new();for id in ids {for slot in 0..4 {tokens.push(if id<complete {id*4+slot}else{-1});}}
    for slot in 0..3 {tokens.push(if slot<len%4 {complete*4+slot}else{-1});}
    Tensor::from_slice(&tokens).to_device(Device::Cuda(0))
}

fn index_case(capacity:i64)->Value {
    let dev=Device::Cuda(0);let pools=(capacity+3)/4;let k=512.min(pools);let slots=k*4+3;let guard=128;
    let source=Tensor::empty([pools+2*guard],(Kind::Float,dev));
    let mut x=source.narrow(0,guard,pools);let mut pos=Tensor::zeros([1],(Kind::Int64,dev));
    let mask_storage=Tensor::empty([pools+2*guard],(Kind::Float,dev));let masked=mask_storage.narrow(0,guard,pools);
    let out_storage=Tensor::empty([slots+2*guard],(Kind::Int64,dev));let out=out_storage.narrow(0,guard,slots);
    let constants=Constants::new(pools);let _=source.shallow_clone().fill_(f64::NAN);
    x.copy_(&scores(pools,0,0,0));mask_into(&x,&pos,&masked);
    let warm_ids=masked.topk(k,0,true,true).1;expand_into(&warm_ids,&pos,&out);let _=old(&x,&pos,&constants);
    tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();mask_into(&x,&pos,&masked);
    let captured_ids=masked.topk(k,0,true,true).1;expand_into(&captured_ids,&pos,&out);
    crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    let mut records=Vec::new();let mut previous:Option<(Tensor,Tensor)>=None;
    for (turn,position) in positions(capacity).into_iter().enumerate() {for mode in 0..5 {
        let _=pos.fill_(position);x.copy_(&scores(pools,position,mode,turn));
        let source_before=source.copy();let pos_before=pos.copy();
        let _=mask_storage.shallow_clone().fill_(731.25);let _=masked.view_dtype(Kind::Int).fill_(0x7fa7c0dei64);
        let _=out_storage.shallow_clone().fill_(-9876543i64);let _=out.shallow_clone().fill_(i64::MIN);
        graph.replay();let (gold_mask,gold_ids,gold_out)=old(&x,&pos,&constants);
        bits(&masked,&gold_mask,"masked scores include raw NaN payload/signed-zero bits");
        bits(&captured_ids,&gold_ids,"sorted topk IDs including tie order");
        bits(&out,&gold_out,"expanded pool IDs and three tail slots");
        bits(&out,&mathematical_indices(&gold_ids,position),"independent CPU index formula");
        for slice in [mask_storage.narrow(0,0,guard),mask_storage.narrow(0,guard+pools,guard)] {
            assert!(slice.eq(731.25).all().int64_value(&[])!=0,"mask output guard overwritten");
        }
        for slice in [out_storage.narrow(0,0,guard),out_storage.narrow(0,guard+slots,guard)] {
            assert!(slice.eq(-9876543i64).all().int64_value(&[])!=0,"index output guard overwritten");
        }
        let ids_before=captured_ids.copy();expand_into(&captured_ids,&pos,&out);
        bits(&captured_ids,&ids_before,"expand must not change selected input IDs");
        bits(&source,&source_before,"score input and both padding guards unchanged");bits(&pos,&pos_before,"device position unchanged");
        flag(true);let public=try_ranked(&x,&pos).expect("eligible fused Ranked path was not used");bits(&public,&gold_out,"public new allocation");
        flag(false);assert!(try_ranked(&x,&pos).is_none(),"default disabled must fall back");
        if let Some((a,b))=&previous {bits(a,b,"previous output survives graph replay and new allocation");}
        previous=Some((public,gold_out));
        records.push(json!({"pos":position,"score_mode":mode,"graph_exact":true,"mask_raw_bits":true,
            "sorted_topk_tie_order":true,"cpu_index_oracle":true,"input_unchanged":true,"both_output_guards":true}));
    }}
    tch::Cuda::synchronize(0);drop(graph);drop(captured_ids);
    if let Some((a,b))=&previous {bits(a,b,"public output survives graph destruction");}
    flag(true);
    let strided=Tensor::zeros([pools*2],(Kind::Float,dev)).slice(0,0,pools*2,2);
    if pools>1 {assert!(try_ranked(&strided,&pos).is_none());}
    assert!(try_ranked(&x.to_kind(Kind::Double),&pos).is_none());
    assert!(try_ranked(&x,&pos.to_kind(Kind::Int)).is_none());
    assert!(try_ranked(&x.to_device(Device::Cpu),&pos.to_device(Device::Cpu)).is_none());
    json!({"capacity":capacity,"pools":pools,"k":k,"slots":slots,"records":records,
        "long_then_short_same_graph":true,"owner_lifetime":true,"unsupported_layout_dtype_device_fallback":true})
}

fn state_bits(a:&crate::dsa::State,b:&crate::dsa::State) {
    bits(&a.pools,&b.pools,"append pools");bits(&a.tail_k,&b.tail_k,"append tail keys");bits(&a.tail_gate,&b.tail_gate,"append tail gates");
}
// This gate requires applying the supplied dsa.rs early-return patch. It
// exercises the actual producer -> Ranked consumer seam, including unchanged
// score_fused off/on. No floating score or append implementation is copied.
fn append_cases()->Vec<Value> {
    let dev=Device::Cuda(0);let mut cases=Vec::new();tch::manual_seed(923901);
    let w=crate::dsa::Weights {q:Tensor::zeros([4096,16],(Kind::Float,dev)),k:Tensor::zeros([128,16],(Kind::Float,dev)),
        norm_w:Tensor::ones([128],(Kind::Float,dev)),norm_b:Tensor::zeros([128],(Kind::Float,dev)),
        score:Tensor::zeros([32,16],(Kind::Float,dev)),ape:Tensor::randn([4,128],(Kind::Float,dev)),
        gate:Tensor::zeros([128,16],(Kind::Float,dev))};
    let mut p=crate::dsa::Projected {k:Tensor::randn([2,128],(Kind::Float,dev)),gate:Tensor::randn([2,128],(Kind::Float,dev)),
        q:Tensor::randn([2,32,128],(Kind::Float,dev))*0.05,mixing:Tensor::randn([2,32,1],(Kind::Float,dev))*0.1};
    let original_k=p.k.copy();let original_q=p.q.copy();let original_gate=p.gate.copy();let original_mix=p.mixing.copy();
    for capacity in [36i64,20480] {for score_mode in ["0","1"] {for constants in ["0","1"] {
        std::env::set_var("GLM53_DSA_SCORE_FUSED",score_mode);std::env::set_var("GLM53_STATIC_TENSORS",constants);
        let original=crate::dsa::State::new(capacity,128,dev);let _=original.pools.shallow_clone().normal_(0.,0.02);
        let _=original.tail_k.shallow_clone().normal_(0.,0.1);let _=original.tail_gate.shallow_clone().normal_(0.,0.1);
        let mut a=original.snapshot();let mut b=original.snapshot();let mut pos=Tensor::zeros([1],(Kind::Int64,dev));
        flag(false);let _=a.append_projected(&w,&p,1,&pos);flag(true);let _=b.append_projected(&w,&p,1,&pos);
        tch::Cuda::synchronize(0);a.restore(&original);b.restore(&original);flag(false);
        crate::tp::graph::begin().unwrap();let out_a=a.append_projected(&w,&p,1,&pos);crate::tp::graph::end().unwrap();let graph_a=crate::tp::graph::Owned::take();
        flag(true);crate::tp::graph::begin().unwrap();let out_b=b.append_projected(&w,&p,1,&pos);crate::tp::graph::end().unwrap();let graph_b=crate::tp::graph::Owned::take();
        let mut checked=0;
        for (turn,position) in positions(capacity).into_iter().enumerate() {
            a.restore(&original);b.restore(&original);let _=pos.fill_(position);
            let scale=if turn%2==0 {1.}else{-0.5};p.k.copy_(&(&original_k*scale));p.q.copy_(&(&original_q*scale));
            p.gate.copy_(&(&original_gate*scale));p.mixing.copy_(&(&original_mix*scale));
            let first_invisible=(position+1)/4;
            let _=a.pools.narrow(0,first_invisible,a.pools.size()[0]-first_invisible).fill_(f64::NAN);
            let _=b.pools.narrow(0,first_invisible,b.pools.size()[0]-first_invisible).fill_(f64::NAN);
            let _=out_a.shallow_clone().fill_(i64::MIN);let _=out_b.shallow_clone().fill_(i64::MIN);
            graph_b.replay();graph_a.replay();bits(&out_a,&out_b,"append_projected old/new changed-input graph");state_bits(&a,&b);
            checked+=1;
        }
        tch::Cuda::synchronize(0);drop(graph_b);drop(graph_a);
        cases.push(json!({"capacity":capacity,"score_fused_mode":score_mode,"static_tensors":constants,
            "changed_position_and_projected_inputs":checked,"index_order_exact":true,"pool_tail_state_bits":true,"inactive_pool_nan":true}));
    }}}
    cases
}

fn timing(pools:i64,position:i64)->Value {
    let c=Constants::new(pools);let dev=Device::Cuda(0);let sets=16;
    let inputs:Vec<_>=(0..sets).map(|i|(scores(pools,position,0,i),Tensor::from_slice(&[position]).to_device(dev))).collect();
    let gold:Vec<_>=inputs.iter().map(|(x,p)|old(x,p,&c).2).collect();let mut rounds=Vec::new();
    for block in 0..2 {for (arm,on) in [false,true,true,false].into_iter().enumerate() {
        flag(on);let call=|(x,p):&(Tensor,Tensor)|if on {try_ranked(x,p).unwrap()}else{old(x,p,&c).2};
        for input in &inputs {let _=call(input);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let out:Vec<_>=inputs.iter().map(call).collect();crate::tp::graph::end().unwrap();
        let graph=crate::tp::graph::Owned::take();graph.replay();for (a,b) in out.iter().zip(&gold){bits(a,b,"timing graph");}
        for _ in 0..3 {graph.replay();}tch::Cuda::synchronize(0);let mut samples=Vec::new();
        for _ in 0..5 {let start=Instant::now();for _ in 0..4 {graph.replay();}tch::Cuda::synchronize(0);
            samples.push(start.elapsed().as_secs_f64()*1e6/(4*sets) as f64);}
        rounds.push(json!({"abba_block":block,"arm_index":arm,"fused":on,"us_per_node":samples}));drop(graph);drop(out);
    }}
    json!({"pools":pools,"pos":position,"sets":sets,"score_workset_bytes":sets*pools as usize*4,"rounds":rounds,
        "scope":"index path including unchanged topk; synthetic precomputed scores, no floating score/append/attention/model/TP"})
}

pub(super) fn run(out:&Path) {
    assert!(!crate::tp::is_tp());let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::set_num_threads(4);
    std::fs::create_dir_all(out).unwrap();let path=out.join("dsa-index-local.json");
    std::fs::write(&path,r#"{"complete":false,"gate":false}"#).unwrap();
    std::fs::remove_file(out.join("dsa-index-timing.json")).ok();
    std::env::remove_var("GLM53_DSA_INDEX_FUSED");assert!(!super::enabled());
    let mut cases=Vec::new();
    for capacity in [1i64,3,4,5,36,65,2047,2053,20480] {
        cases.push(index_case(capacity));
        std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":false,"gate":false,"index_cases":cases})).unwrap()).unwrap();
    }
    let append=append_cases();
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":true,"gate":true,"index_cases":cases,"append_cases":append,
        "scope":"local mask/index/state/changed-input graph gate only; real-model TP2 still required"})).unwrap()).unwrap();
    let mut records=Vec::new();for (pools,pos) in [(9,35),(5120,7),(5120,2052),(5120,20479)] {records.push(timing(pools,pos));
        std::fs::write(out.join("dsa-index-timing.json"),serde_json::to_string_pretty(&json!({"complete":records.len()==4,"cases":records})).unwrap()).unwrap();}
}
