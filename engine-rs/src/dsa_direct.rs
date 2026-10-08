//! All-visible attention without materialized DSA indices. Slot order and
//! FP32 reductions match the existing canonical all-visible path exactly.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;

pub(crate) fn attention(q:&Tensor,latent:&Tensor,pos:&Tensor)->Tensor {
    let (heads,queries)=(q.size()[0],q.size()[1]);
    assert_eq!(q.size()[2],512);assert_eq!(q.kind(),Kind::Float);
    assert_eq!(q.stride()[2],1);assert_eq!(q.stride()[1],512);
    assert_eq!(latent.kind(),Kind::Half);assert!(latent.is_contiguous());
    assert_eq!(latent.size()[1],512);assert_eq!(pos.kind(),Kind::Int64);
    assert_eq!(pos.numel() as i64,queries);assert!(pos.is_contiguous());
    assert_eq!(pos.device(),q.device());assert_eq!(latent.device(),q.device());
    // The checked verifier regime proves every visible length fits the full
    // pool budget. The device scalar remains dynamic across graph replay.
    let slots=4*512.min((latent.size()[0]+3)/4)+3;
    let splits=if heads*queries<192{8}else{1};
    let out=Tensor::empty([heads,queries,512],(Kind::Float,q.device()));
    let scratch=Tensor::empty([if splits>1{splits*heads*queries*514}else{0}],(Kind::Float,q.device()));
    extern "C" {fn rs_visible_latent(q:*const f32,c:*const std::ffi::c_void,pos:*const i64,out:*mut f32,scratch:*mut f32,h:i32,t:i32,s:i32,stride:i32,splits:i32)->i32;}
    assert_eq!(unsafe{rs_visible_latent(q.data_ptr().cast(),latent.data_ptr(),pos.data_ptr().cast(),out.data_ptr().cast(),scratch.data_ptr().cast(),heads as i32,queries as i32,slots as i32,q.stride()[0] as i32,splits as i32)},0);
    out
}

fn indices(pos:&Tensor,slots:i64)->Tensor {
    let ids=Tensor::arange(slots-3,(Kind::Int64,pos.device()));
    let offsets=Tensor::arange(3,(Kind::Int64,pos.device()));
    indices_from_offsets(pos,&ids,&offsets)
}
fn indices_from_offsets(pos:&Tensor,ids:&Tensor,offsets:&Tensor)->Tensor {
    let len=pos+1;let full=len.floor_divide_scalar(4)*4;
    let ids=ids.unsqueeze(0).expand([pos.numel() as i64,ids.numel() as i64],false);
    let tokens=ids.masked_fill(&ids.ge_tensor(&full.unsqueeze(1)),-1);
    let offsets=offsets.unsqueeze(0);
    let tail=(&full.unsqueeze(1)+&offsets).masked_fill(&offsets.ge_tensor(&len.remainder(4).unsqueeze(1)),-1);
    Tensor::cat(&[tokens,tail],1)
}

pub fn probe(out:&Path) {
    assert!(!crate::tp::is_tp());tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    std::fs::create_dir_all(out).unwrap();tch::manual_seed(923604);
    std::fs::write(out.join("numeric.json"),r#"{"gate":false,"complete":false}"#).unwrap();
    let dev=Device::Cuda(0);let mut records=Vec::new();
    for heads in [1i64,32,64] {for queries in [1i64,8] {
        // Single-query slices keep the real tree's gap between adjacent heads.
        let original=Tensor::randn([heads,8,512],(Kind::Float,dev))*0.3;
        let mut q=original.copy();let mut pos=Tensor::zeros([queries],(Kind::Int64,dev));
        let values=Tensor::randn([2112,512],(Kind::Half,dev))*0.1;let mut latent=values.copy();
        let _=attention(&q.narrow(1,0,queries),&latent,&pos);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let graph_output=attention(&q.narrow(1,0,queries),&latent,&pos);
        crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        for len in [1i64,2,3,4,5,127,128,2047,2048,2049,2050,2051,3] {
            let positions:Vec<_>=(0..queries).map(|i|0.max(len-1-i)).collect();
            pos.copy_(&Tensor::from_slice(&positions).to_device(dev));
            q.copy_(&(&original*if len%2==0{1.}else{-1.}));
            latent.copy_(&values);let _=latent.narrow(0,len,2112-len).fill_(f64::NAN);
            let selected=indices(&pos,2051);
            let expected=crate::mla_latent::sparse_attention(&q.narrow(1,0,queries),&latent,&selected);
            let actual=attention(&q.narrow(1,0,queries),&latent,&pos);graph.replay();
            assert!(actual.equal(&expected)&&graph_output.equal(&expected),"direct visible order/graph mismatch h={heads} t={queries} len={len}");
            assert!(actual.isfinite().all().int64_value(&[])!=0,"direct visible read inactive NaN suffix");
            records.push(json!({"heads":heads,"queries":queries,"length":len,"explicit_indices_exact":true,"changed_input_length_graph_exact":true,"nan_suffix_safe":true}));
        }
    }}
    // The old padding jump could duplicate tail entries when full_slots was
    // not a multiple of 32. Explicit/direct parity alone cannot detect their
    // shared mistake: use a FP64 causal-prefix oracle as an independent check.
    for (capacity,lengths) in [(36i64,vec![1i64,31,32,33,35,36,3]),
        (65,vec![33,63,64,65,3]),(2047,vec![33,2043,2044,2045,2047,3])] {
        for queries in [1i64,8] {
            let heads=32;let slots=4*512.min((capacity+3)/4)+3;
            let original=Tensor::randn([heads,8,512],(Kind::Float,dev))*0.3;
            let mut q=original.copy();let mut pos=Tensor::zeros([queries],(Kind::Int64,dev));
            let values=Tensor::randn([capacity,512],(Kind::Half,dev))*0.1;let mut latent=values.copy();
            let _=attention(&q.narrow(1,0,queries),&latent,&pos);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let captured=attention(&q.narrow(1,0,queries),&latent,&pos);
            crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
            for &len in &lengths {
                let positions:Vec<_>=(0..queries).map(|i|0.max(len-1-i)).collect();
                pos.copy_(&Tensor::from_slice(&positions).to_device(dev));
                q.copy_(&(&original*if len%2==0{1.}else{-1.}));
                latent.copy_(&values);let _=latent.narrow(0,len,capacity-len).fill_(f64::NAN);
                let actual=attention(&q.narrow(1,0,queries),&latent,&pos);
                let expected=crate::mla_latent::sparse_attention(&q.narrow(1,0,queries),&latent,&indices(&pos,slots));graph.replay();
                assert!(actual.equal(&expected)&&captured.equal(&expected)&&actual.isfinite().all().int64_value(&[])!=0,"small-capacity visible/graph gate");
                let mut oracle_max=0f64;
                for query in [0i64,queries-1] {
                    let visible=positions[query as usize]+1;let c=latent.narrow(0,0,visible).to_kind(Kind::Double);
                    let q=q.narrow(1,query,1).to_kind(Kind::Double);
                    let oracle=(q.matmul(&c.transpose(0,1))/16.).softmax(-1,Kind::Double).matmul(&c);
                    let delta=(actual.narrow(1,query,1).to_kind(Kind::Double)-&oracle).norm()/oracle.norm().clamp_min(1e-30);
                    let relative=delta.double_value(&[]);assert!(relative<2e-5,"small-capacity FP64 oracle mismatch: {relative}");oracle_max=oracle_max.max(relative);
                }
                records.push(json!({"capacity":capacity,"heads":heads,"queries":queries,"length":len,"slots":slots,
                    "explicit_indices_exact":true,"changed_input_length_graph_exact":true,"nan_suffix_safe":true,
                    "fp64_first_last_query_relative_max":oracle_max}));
            }
        }
    }
    std::fs::write(out.join("numeric.json"),serde_json::to_string_pretty(&json!({"gate":true,"complete":true,"cases":records})).unwrap()).unwrap();
    let mut timing=Vec::new();
    for len in [64i64,512,2048] {
        let pos=Tensor::from_slice(&[len-1]).to_device(dev);
        let q=Tensor::randn([32,8,512],(Kind::Float,dev)).narrow(1,0,1)*0.3;
        let latent=Tensor::randn([20480,512],(Kind::Half,dev))*0.1;
        // Production State::new owns these immutable constants. Keep their
        // allocation/arange outside capture in this baseline as well.
        let ids=Tensor::arange(2048,(Kind::Int64,dev));let offsets=Tensor::arange(3,(Kind::Int64,dev));
        for direct in [false,true,true,false] {
            let call=||if direct{attention(&q,&latent,&pos)}else{crate::mla_latent::sparse_attention(&q,&latent,&indices_from_offsets(&pos,&ids,&offsets))};
            let expected=call();tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let result=call();crate::tp::graph::end().unwrap();
            let graph=crate::tp::graph::Owned::take();for _ in 0..5{graph.replay();}
            assert!(result.equal(&expected));tch::Cuda::synchronize(0);let start=Instant::now();
            for _ in 0..64{graph.replay();}tch::Cuda::synchronize(0);
            timing.push(json!({"length":len,"direct":direct,"us":start.elapsed().as_secs_f64()*1e6/64.,"scope":"local synthetic query/cache; baseline includes explicit index generation"}));
        }
    }
    std::fs::write(out.join("timing.json"),serde_json::to_string_pretty(&timing).unwrap()).unwrap();
    eprintln!("[dsa-direct] explicit-index parity, strided heads, changing graph lengths and NaN suffix gates passed");
}
