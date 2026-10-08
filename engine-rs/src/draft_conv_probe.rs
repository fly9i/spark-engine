//! Real ten base kernels + synthetic activations/delta; no target model load.
use super::{into,old,try_convolve};
use tch::{Kind,Tensor,Device};
use serde_json::{json,Value};
use std::{path::Path,time::Instant};
struct Restore(Option<std::ffi::OsString>);
impl Drop for Restore{fn drop(&mut self){match &self.0{Some(v)=>std::env::set_var("GLM53_DRAFT_CONV_FUSED",v),None=>std::env::remove_var("GLM53_DRAFT_CONV_FUSED")}}}
fn flag(on:bool){std::env::set_var("GLM53_DRAFT_CONV_FUSED",if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.kind(),Kind::BFloat16);assert_eq!(b.kind(),Kind::BFloat16);assert_eq!(a.size(),b.size());
    assert!(a.contiguous().view_dtype(Kind::Int16).equal(&b.contiguous().view_dtype(Kind::Int16)),"{label}: raw BF16 bits, including NaN payloads");
}
fn patterns(shape:&[i64],turn:usize)->Tensor {
    let bits=[0u16,0x8000,0x7f80,0xff80,0x7fc1,0xffc3,1,0x8001,0x7f7f,0xff7f,0x3f80,0xbf80,0x3f81,0x3f7f];
    let count=shape.iter().product::<i64>() as usize;
    let v:Vec<i16>=(0..count).map(|i|bits[(i+turn)%bits.len()] as i16).collect();
    Tensor::from_slice(&v).view_dtype(Kind::BFloat16).view(shape).to_device(Device::Cuda(0))
}
fn delta_view(n:i64,layout:usize,side:i64)->Tensor {
    let opt=(Kind::BFloat16,Device::Cuda(0));
    match layout {0=>Tensor::zeros([n,2,256],opt),
        1=>Tensor::full([8,2,2,256],f64::NAN,opt).narrow(0,0,n).select(1,side),
        2=>Tensor::full([8,2,512],f64::NAN,opt).narrow(0,0,n).slice(2,0,512,2),_=>unreachable!()}
}
fn fill_finite(x:&mut Tensor,d:&mut Tensor,turn:usize) {
    x.copy_(&(Tensor::randn(x.size().as_slice(),(Kind::Float,x.device()))*(0.03+turn as f64*0.02)).to_kind(Kind::BFloat16));
    d.copy_(&(Tensor::randn(d.size().as_slice(),(Kind::Float,d.device()))*(0.02+turn as f64*0.01)).to_kind(Kind::BFloat16));
}
fn changed_graph(base:&Tensor,n:i64,side:i64,layout:usize)->Value {
    let opt=(Kind::BFloat16,base.device());let guard=64;
    let x_owner=Tensor::full([8*4096+2*guard],f64::NAN,opt);
    let mut x=x_owner.narrow(0,guard,n*4096).view([n,4096]);let mut delta=delta_view(n,layout,side);
    let out_owner=Tensor::full([8*4096+2*guard],23.,opt);let mut out=out_owner.narrow(0,guard,n*4096).view([n,4096]);
    let mut dynamic_base=base.copy();fill_finite(&mut x,&mut delta,0);
    into(&x,&delta,&dynamic_base,side,&out);let _=old(&x,&delta,&dynamic_base,side);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();into(&x,&delta,&dynamic_base,side,&out);crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    let mut turns=Vec::new();let mut previous:Option<(Tensor,Tensor)>=None;
    for turn in 0..5 {
        dynamic_base.copy_(base);
        if turn==1 {x.copy_(&patterns(&[n,4096],turn));delta.copy_(&patterns(&[n,2,256],turn+3));}
        else if turn==2 {
            // All row0 second-tap coefficients become +Inf. +0*Inf must NOT
            // disappear even though shifted row0 has no preceding token.
            let _=dynamic_base.fill_(0.);let _=dynamic_base.get(side).get(1).fill_(f64::INFINITY);
            let _=x.fill_(1.);let _=delta.fill_(0.);
        }else{fill_finite(&mut x,&mut delta,turn);}
        let xb=x_owner.copy();let db=delta.copy();let bb=dynamic_base.copy();let _=out.fill_(f64::NAN);
        graph.replay();let gold=old(&x,&delta,&dynamic_base,side);exact(&out,&gold,"changed graph");
        if turn==2{assert!(out.get(0).isnan().all().int64_value(&[])!=0,"row0 0*Inf lost");}
        exact(&x_owner,&xb,"x and invalid capacity unchanged");exact(&delta,&db,"delta unchanged");exact(&dynamic_base,&bb,"base unchanged");
        for g in [out_owner.narrow(0,0,guard),out_owner.narrow(0,guard+n*4096,(8-n)*4096+guard)]{assert!(g.eq(23.).all().int64_value(&[])!=0);}
        flag(true);let fresh=try_convolve(&x,&delta,&dynamic_base,side).unwrap();exact(&fresh,&gold,"public result");
        if let Some((a,b))=&previous{exact(a,b,"previous result isolated from replay");}previous=Some((fresh,gold));
        turns.push(json!({"turn":turn,"raw_bits_exact":true,"guards_and_inputs_unchanged":true}));
    }
    tch::Cuda::synchronize(0);drop(graph);drop(x);drop(x_owner);drop(delta);drop(dynamic_base);drop(out);drop(out_owner);
    if let Some((a,b))=&previous{exact(a,b,"fresh result owns storage after graph/input drop");}
    json!({"rows":n,"side":side,"delta_layout":layout,"turns":turns,"fresh_result_owns_storage":true})
}
fn timing(bases:&[(String,Tensor)])->Value {
    let work:Vec<_>=bases.iter().flat_map(|(_,b)|(0..2).map(move |s|(Tensor::randn([8,4096],(Kind::Float,b.device())).to_kind(Kind::BFloat16),
        Tensor::randn([8,2,2,256],(Kind::Float,b.device())).to_kind(Kind::BFloat16).select(1,s),b.shallow_clone(),s))).collect();
    let gold:Vec<_>=work.iter().map(|(x,d,b,s)|old(x,d,b,*s)).collect();let mut rounds=Vec::new();
    for execution in ["graph","eager"] {for block in 0..2 {for (arm,on) in [false,true,true,false].into_iter().enumerate() {
        flag(on);let call=|(x,d,b,s):&(Tensor,Tensor,Tensor,i64)|if on{try_convolve(x,d,b,*s).unwrap()}else{old(x,d,b,*s)};
        for input in &work{let _=call(input);}tch::Cuda::synchronize(0);
        let (graph,outputs)=if execution=="graph" {crate::tp::graph::begin().unwrap();let outputs:Vec<_>=work.iter().map(call).collect();
            crate::tp::graph::end().unwrap();(Some(crate::tp::graph::Owned::take()),outputs)}else{(None,work.iter().map(call).collect())};
        if let Some(g)=&graph{g.replay();}for (a,b) in outputs.iter().zip(&gold){exact(a,b,"timing result");}
        let mut samples=Vec::new();for _ in 0..5 {tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..4{
            if let Some(g)=&graph{g.replay();}else{for input in &work{let _=call(input);}}
        }tch::Cuda::synchronize(0);samples.push(start.elapsed().as_secs_f64()*1e6/(4*20) as f64);}
        rounds.push(json!({"execution":execution,"block":block,"arm":arm,"fused":on,"us_per_convolve":samples}));drop(graph);drop(outputs);
    }}}
    json!({"real_base_tensors":10,"operations_per_rotation":20,"rows":8,"rounds":rounds,
        "scope":"20 independent synthetic x/delta + real base operations; excludes unchanged projection/norm/residual/attention/TP/D2H; do not multiply local ratios into model throughput"})
}
pub(super) fn run(draft:&Path,out:&Path) {
    assert!(!crate::tp::is_tp());let _guard=tch::no_grad_guard();tch::set_num_threads(4);tch::manual_seed(2026092321);
    let _restore=Restore(std::env::var_os("GLM53_DRAFT_CONV_FUSED"));std::env::remove_var("GLM53_DRAFT_CONV_FUSED");assert!(!super::enabled());
    std::fs::create_dir_all(out).unwrap();let path=out.join("draft-conv-local.json");
    let mut report=json!({"complete":false,"gate":false,"scope":"real base kernels + synthetic BF16 x/delta; no full-model qualification","eager":[],"graphs":[]});
    let save=|r:&Value|std::fs::write(&path,serde_json::to_string_pretty(r).unwrap()).unwrap();save(&report);
    let idx=crate::safetensors::ShardIndex::scan(draft).unwrap();let mut bases=Vec::new();
    for layer in 0..5 {for name in ["attention_conv","mlp_conv"] {let key=format!("layers.{layer}.{name}.base_kernel");
        let (values,shape)=idx.get_f32(&key).unwrap();assert_eq!(shape,vec![2,2,4096]);
        bases.push((key,Tensor::from_slice(&values).view([2,2,4096]).to_kind(Kind::BFloat16).to_device(Device::Cuda(0))));
    }}drop(idx);
    flag(true);
    for (key,base) in &bases {for n in 1..=8 {for side in 0..2 {for layout in 0..3 {
        let mut x=Tensor::zeros([n,4096],(Kind::BFloat16,base.device()));let mut d=delta_view(n,layout,side);fill_finite(&mut x,&mut d,0);
        let actual=try_convolve(&x,&d,base,side).unwrap();exact(&actual,&old(&x,&d,base,side),"real base eager");
        report["eager"].as_array_mut().unwrap().push(json!({"weight":key,"rows":n,"side":side,"delta_stride":d.stride(),"raw_bits_exact":true}));
    }}}save(&report);}
    // Include long->short capture order; never read NaN invalid capacity.
    for n in [8,1,2] {for side in 0..2 {for layout in 0..3 {
        report["graphs"].as_array_mut().unwrap().push(changed_graph(&bases[0].1,n,side,layout));save(&report);
    }}}
    let base=&bases[0].1;let x=Tensor::zeros([8,4096],(Kind::BFloat16,base.device()));let d=delta_view(8,1,0);
    flag(false);assert!(try_convolve(&x,&d,base,0).is_none());flag(true);
    assert!(try_convolve(&x.to_kind(Kind::Float),&d,base,0).is_none());assert!(try_convolve(&x,&d.to_kind(Kind::Float),base,0).is_none());
    assert!(try_convolve(&x,&d,&base.to_kind(Kind::Float),0).is_none());
    assert!(try_convolve(&x,&d,&base.get(0),0).is_none());
    assert!(try_convolve(&x,&d.narrow(2,0,255),base,0).is_none());
    assert!(try_convolve(&x,&d,base,2).is_none());assert!(try_convolve(&x,&d,base,-1).is_none());
    let padded=Tensor::zeros([8,8192],(Kind::BFloat16,base.device())).slice(1,0,8192,2);assert!(try_convolve(&padded,&d,base,0).is_none());
    for n in [0,9] {let a=Tensor::zeros([n,4096],(Kind::BFloat16,base.device()));let b=Tensor::zeros([n,2,256],(Kind::BFloat16,base.device()));assert!(try_convolve(&a,&b,base,0).is_none());}
    assert!(try_convolve(&x.to_device(Device::Cpu),&d.to_device(Device::Cpu),&base.to_device(Device::Cpu),0).is_none());
    assert_eq!(report["eager"].as_array().unwrap().len(),480);assert_eq!(report["graphs"].as_array().unwrap().len(),18);
    report["complete"]=json!(true);report["gate"]=json!(true);report["eager_cases"]=json!(480);report["graph_captures"]=json!(18);report["graph_replays"]=json!(90);report["fallback_checked"]=json!(true);save(&report);
    std::fs::write(out.join("draft-conv-timing.json"),serde_json::to_string_pretty(&timing(&bases)).unwrap()).unwrap();
}
