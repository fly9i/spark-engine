//! Real resident decode profiling window, excluding model load/prefill/capture.
use std::path::Path;
use tch::{Tensor,Device};
pub fn steady(dir:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    let dev=Device::Cuda(0);let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;let misses=fast.misses;
    let mut eng=crate::forward::Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let ints=|k:&str|refs["count"][k].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect::<Vec<_>>();
    let ids=ints("prompt_ids");let tokens=ints("text_ids");
    let (_,s)=eng.prefill(&Tensor::from_slice(&ids).to_device(dev));
    let mut state=crate::forward::DecodeStates(s.0.into_iter().map(|s|match s {
        crate::forward::LayerState::Mla(m)=>crate::forward::LayerState::MlaG(crate::mla::MlaStateG::from_state(&m,512)),other=>other,
    }).collect());
    let initial=crate::forward::snapshot(&state);let inputs:Vec<_>=tokens.iter().map(|&t|Tensor::from_slice(&[t]).to_device(dev)).collect();
    let mut input=inputs[0].copy();let _=eng.step_buf(&input,&mut state);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let _output=eng.step_buf(&input,&mut state);crate::tp::graph::end().unwrap();
    for _ in 0..3 {crate::forward::restore(&mut state,&initial);for t in &inputs{input.copy_(t);crate::tp::graph::replay().unwrap();}}
    crate::forward::restore(&mut state,&initial);tch::Cuda::synchronize(0);
    extern "C" {fn cudaProfilerStart()->i32;fn cudaProfilerStop()->i32;}
    assert_eq!(unsafe{cudaProfilerStart()},0);
    for t in &inputs {input.copy_(t);crate::tp::graph::replay().unwrap();}
    tch::Cuda::synchronize(0);assert_eq!(unsafe{cudaProfilerStop()},0);
    crate::tp::graph::destroy();assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
    eprintln!("[profile-steady] rank{} captured {} real-input resident graph steps; zero cold loads",tp.rank,inputs.len());
}
