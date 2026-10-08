//! Full-model chunk/prefix/capacity/long-context replay acceptance.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;
use crate::forward::{Engine,snapshot,states_max_diff,restore};

pub fn run(dir:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT","GLM53_PREFILL_BATCH"] {std::env::set_var(flag,"1");}
    std::env::set_var("GLM53_MAX_CONTEXT","2112");
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,
        cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let misses=fast.misses;
    let eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    let mut session=crate::session::Session::new(eng,4,128);
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let base:Vec<i64>=refs["count"]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let continuation:Vec<i64>=refs["count"]["text_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let mut ids=base.clone();ids.extend(&continuation[..8]);
    let input=Tensor::from_slice(&ids).to_device(dev);
    let (logits,st)=session.engine.prefill(&input);let gold=logits.get(logits.size()[0]-1);
    let (cached,state,hit)=session.prepare(&ids);assert_eq!(hit,0);
    assert!(cached.equal(&gold));assert_eq!(states_max_diff(&st,&state),0.);
    let (again,mut reused,hit)=session.prepare(&ids);assert_eq!(hit,ids.len());
    assert!(again.equal(&gold));assert_eq!(states_max_diff(&st,&reused),0.);
    let _=session.engine.step(continuation[8],&mut reused);
    let (_,unmodified,_)=session.prepare(&ids);assert_eq!(states_max_diff(&st,&unmodified),0.);
    let mut extended=ids.clone();extended.extend(&continuation[8..16]);
    let (partial,partial_state,hit)=session.prepare(&extended);assert_eq!(hit,ids.len());
    let (expected,expected_state)=session.engine.prefill_with(&Tensor::from_slice(&continuation[8..16]).to_device(dev),Some(snapshot(&st)));
    assert!(partial.equal(&expected.get(7)));assert_eq!(states_max_diff(&partial_state,&expected_state),0.);
    // Stress prompt, explicitly not a natural-language quality benchmark.
    let long:Vec<i64>=(0..2044).map(|i|base[i%base.len()]).collect();
    tch::Cuda::synchronize(0);let started=Instant::now();
    let (long_logits,mut state,_) = session.prepare(&long);
    tch::Cuda::synchronize(0);let prefill_ms=started.elapsed().as_secs_f64()*1000.;
    assert!(long_logits.isfinite().all().int64_value(&[])!=0);
    let initial=snapshot(&state);let tokens=&continuation[..16];
    let inputs:Vec<_>=tokens.iter().map(|&t|Tensor::from_slice(&[t]).to_device(dev)).collect();
    let mut expected=Vec::new();
    for t in &inputs {expected.push(session.engine.step_buf(t,&mut state));}
    let end=snapshot(&state);restore(&mut state,&initial);
    let mut input=inputs[0].copy();let _=session.engine.step_buf(&input,&mut state);
    tch::Cuda::synchronize(0);restore(&mut state,&initial);
    crate::tp::graph::begin().unwrap();let output=session.engine.step_buf(&input,&mut state);crate::tp::graph::end().unwrap();
    let mut max_abs=0f64;
    for _ in 0..2 {
        restore(&mut state,&initial);
        for (i,t) in inputs.iter().enumerate() {input.copy_(t);crate::tp::graph::replay().unwrap();
            max_abs=max_abs.max(f64::try_from((&output-&expected[i]).abs().max()).unwrap());}
        assert_eq!(states_max_diff(&state,&end),0.,"full-model graph state at 2048 boundary");
    }
    crate::tp::graph::destroy();assert_eq!(max_abs,0.,"full-model graph logits at 2048 boundary");
    assert_eq!(misses,session.engine.fast.as_ref().unwrap().misses);
    Tensor::save_multi(&[("long_logits",long_logits.to_device(Device::Cpu)),("trace",Tensor::stack(&expected,0).to_device(Device::Cpu))],out.join(format!("long-rank{}.pt",tp.rank))).unwrap();
    std::fs::write(out.join(format!("session-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({
        "rank":tp.rank,"chunk_size":128,"prompt_tokens":2044,"prefill_ms":prefill_ms,"decode_steps":16,
        "graph_rounds":2,"graph_max_abs":max_abs,"graph_state_max_abs":0,"cold_loads":0,
        "exact_prefix_hit":true,"partial_prefix_hit":true,"checkout_independent":true,
        "pool_slots":session.prefixes.len(),"evictions":session.prefixes.evictions})).unwrap()).unwrap();
    eprintln!("[session-probe] rank{} PASS 2044+16, prefix/partial/graph; prefill {prefill_ms:.1} ms",tp.rank);
}
