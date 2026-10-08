//! Real-weight numeric/shape screening for independently gated FP8 extensions.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;
fn rel(a:&Tensor,b:&Tensor)->f64{((a-b).to_kind(Kind::Float).norm()/b.to_kind(Kind::Float).norm().clamp_min(1e-20)).double_value(&[])}
fn load(idx:&mut crate::safetensors::ShardIndex,name:&str,kind:Kind,axis:Option<i64>)->Tensor {
    let (values,dims)=idx.get_f32(name).unwrap();let shape:Vec<_>=dims.iter().map(|v|*v as i64).collect();
    let w=Tensor::from_slice(&values).view(shape.as_slice());
    let w=if let Some(d)=axis{w.narrow(d,0,w.size()[d as usize]/2)}else{w};
    w.to_kind(kind).contiguous().to_device(Device::Cuda(0))
}
pub fn run(target:&Path,draft:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    std::env::set_var("GLM53_TF32","0");crate::tp::init_from_env();crate::tp::set_tf32(false);tch::manual_seed(923701);
    std::env::set_var("GLM53_FP8_WMMA","3");std::env::set_var("GLM53_FP8_LARGE","0");
    let mut ti=crate::safetensors::ShardIndex::scan(target).unwrap();
    let mut di=crate::safetensors::ShardIndex::scan(draft).unwrap();
    let mut weights=Vec::new();
    for (suffix,axis) in [("mlp.shared_experts.gate_proj.weight",0),("mlp.shared_experts.down_proj.weight",1),
        ("self_attn.q_a_proj.weight",-1),("self_attn.q_b_proj.weight",0),("self_attn.o_proj.weight",1)] {
        let name=format!("model.language_model.layers.3.{suffix}");
        weights.push((suffix.to_string(),load(&mut ti,&name,Kind::Half,if axis<0{None}else{Some(axis)})));
    }
    weights.push(("target_head".into(),load(&mut ti,"lm_head.weight",Kind::Half,Some(0))));
    for (name,axis) in [("layers.0.mlp.gate_proj.weight",Some(0)),("layers.0.mlp.down_proj.weight",Some(1)),
        ("layers.0.self_attn.q_proj.weight",None),("layers.0.self_attn.k_proj.weight",None),("fc.weight",None)] {
        weights.push((name.to_string(),load(&mut di,name,Kind::BFloat16,axis)));
    }
    weights.push(("draft_head".into(),weights[5].1.to_kind(Kind::BFloat16)));
    let mut records=Vec::new();
    for (name,w) in &weights {
        let bf=w.kind()==Kind::BFloat16;let (q,s)=crate::dense_fp8::quantize(w);
        let run=|x:&Tensor|if bf{crate::dense_fp8::run_bf16(x,&q,&s,true)}else{crate::dense_fp8::run(x,&q,&s,true)};
        let qf=q.to_kind(Kind::Float);let wf=w.to_kind(Kind::Float);
        for m in [1i64,2,5,8,16,17,64] {
            let original=(Tensor::randn([m,w.size()[1]],(Kind::Float,w.device()))*0.03).to_kind(w.kind()).to_kind(Kind::Float);
            let mut input=original.copy();let reference=input.matmul(&qf.transpose(0,1))*s.unsqueeze(0);
            let native=input.matmul(&wf.transpose(0,1));
            let mut numbers=Vec::new();
            for split in [1,2,4,8,16] {
                std::env::set_var(if bf{"GLM53_DRAFT_FP8_SPLITS"}else{"GLM53_FP8_SPLITS"},split.to_string());
                let actual=run(&input);let e=rel(&actual,&reference);
                assert!(actual.isfinite().all().int64_value(&[])!=0 && e<3e-5,"FP8 oracle {name}/{m}/{split}: {e}");
                numbers.push(json!({"split":split,"oracle_relative":e,"quantized_vs_native_relative":rel(&actual,&native)}));
            }
            crate::tp::graph::begin().unwrap();let graph_y=run(&input);crate::tp::graph::end().unwrap();
            for scale in [0.,-1.,if bf{1e6}else{2.}] {
                input.copy_(&(&original*scale).to_kind(w.kind()).to_kind(Kind::Float));crate::tp::graph::replay().unwrap();
                let eager=run(&input);assert!(graph_y.equal(&eager),"FP8 graph changed input {name}/{m}/{scale}");
                let expected=input.matmul(&qf.transpose(0,1))*s.unsqueeze(0);let e=rel(&graph_y,&expected);
                assert!(graph_y.isfinite().all().int64_value(&[])!=0 && e<3e-5,"FP8 range {name}/{m}/{scale}: {e}");
            }
            crate::tp::graph::destroy();
            records.push(json!({"name":name,"shape":w.size(),"bf16":bf,"m":m,"numeric":numbers,"changed_input_graph":true}));
        }
        // Registry must reject a smaller rank-zero view sharing the base pointer.
        crate::dense_fp8::register_weight(w,"GLM53_FP8_PROBE");std::env::set_var("GLM53_FP8_PROBE","1");
        let x=Tensor::zeros([1,w.size()[1]],(Kind::Float,w.device()));let view=w.narrow(0,0,w.size()[0]/2);
        assert!(if bf{crate::dense_fp8::try_bf16(&x,&view,true)}else{crate::dense_fp8::try_run(&x,&view,true)}.is_none());
        std::fs::write(out.join("numeric.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
        eprintln!("[fp8-extension] numeric {name} passed");
    }
    let mut timing=Vec::new();
    for (name,w) in &weights {
        let bf=w.kind()==Kind::BFloat16;let(q,s)=crate::dense_fp8::quantize(w);
        let rotated:Vec<_>=(0..8).map(|_|q.copy()).collect();
        let native:Vec<_>=(0..8).map(|_|w.copy()).collect();
        for m in [1i64,5,8] {
            let input=(Tensor::randn([m,w.size()[1]],(Kind::Float,w.device()))*0.03).to_kind(w.kind());
            for split in [0,1,2,4,8,16] {
                std::env::set_var(if bf{"GLM53_DRAFT_FP8_SPLITS"}else{"GLM53_FP8_SPLITS"},split.max(1).to_string());
                let call=|i:usize|{if split==0{let y=input.matmul(&native[i].transpose(0,1));drop(y);}else{
                    let y=if bf{crate::dense_fp8::run_bf16(&input,&rotated[i],&s,false)}else{crate::dense_fp8::run(&input,&rotated[i],&s,false)};drop(y);}};
                for i in 0..8{call(i);}tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();for i in 0..8{call(i);}crate::tp::graph::end().unwrap();
                for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                let mut rounds=Vec::new();for _ in 0..3{let started=Instant::now();for _ in 0..8{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);rounds.push(started.elapsed().as_secs_f64()*1e6/64.);}
                crate::tp::graph::destroy();
                timing.push(json!({"name":name,"shape":w.size(),"bf16":bf,"m":m,"split":split,"weight_copies":8,"us":rounds}));
                std::fs::write(out.join("timing.json"),serde_json::to_string_pretty(&timing).unwrap()).unwrap();
            }
        }
    }
}
