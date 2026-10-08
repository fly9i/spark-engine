//! Batch tree vs independent serial branch execution, same checkpoint and TP layout.
use std::{path::Path,time::Instant};
use tch::{Tensor,Device};
use serde_json::json;
use crate::forward::{Engine,DecodeStates,snapshot,states_max_diff};

fn serial(eng:&mut Engine,base:&DecodeStates,ids:&[i64],parents:&[Option<usize>])->(Tensor,Vec<DecodeStates>) {
    let mut states=Vec::new();let mut logits=Vec::new();
    for (i,(&id,&parent)) in ids.iter().zip(parents).enumerate() {
        assert!(parent.map_or(true,|p|p<i));let mut state=snapshot(parent.map_or(base,|p|&states[p]));
        logits.push(eng.step(id,&mut state));states.push(state);
    }
    (Tensor::stack(&logits,0),states)
}
pub fn run(dir:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;let misses=fast.misses;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    check(&mut eng,out);
}

pub fn check(eng:&mut Engine,out:&Path) {
    let tp=crate::tp::world();let dev=eng.w.device;let misses=eng.fast.as_ref().unwrap().misses;
    std::fs::create_dir_all(out).unwrap();
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let parents=[None,Some(0),Some(0),Some(1),Some(2),Some(1),None];let mut cases=Vec::new();
    for name in ["hello","count","hashmap"] {
        let ints=|k:&str|refs[name][k].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect::<Vec<_>>();
        let prompt=ints("prompt_ids");let tokens=ints("text_ids");
        let ids=[tokens[0],tokens[1],tokens[8],tokens[2],tokens[9],tokens[11],tokens[17]];
        let input=Tensor::from_slice(&ids).to_device(dev);
        let (_,base)=eng.prefill(&Tensor::from_slice(&prompt).to_device(dev));let base_copy=snapshot(&base);
        let (baseline,serial_states)=serial(eng,&base,&ids,&parents);
        let (candidate,batch_states,_)=eng.tree_forward(&input,&base,&parents,true);
        assert_eq!(states_max_diff(&base,&base_copy),0.,"tree mutated parent request");
        let references:Vec<i64>=Vec::try_from(baseline.argmax(-1,false).to_device(Device::Cpu)).unwrap();
        let quality=crate::evaluation::distribution(&baseline.to_device(Device::Cpu),&candidate.to_device(Device::Cpu),&references);
        let state_error=serial_states.iter().zip(&batch_states).map(|(a,b)|states_max_diff(a,b)).fold(0.,f64::max);
        assert!(state_error.is_finite());
        let mut rounds=Vec::new();
        for batch in [false,true,true,false] {
            tch::Cuda::synchronize(0);crate::tp::allreduce(&Tensor::zeros([1],(tch::Kind::Float,dev)));tch::Cuda::synchronize(0);
            let start=Instant::now();
            let (last,states)=if batch {let (l,s,_)=eng.tree_forward(&input,&base,&parents,false);(l,s)}else{serial(eng,&base,&ids,&parents)};
            tch::Cuda::synchronize(0);let ms=start.elapsed().as_secs_f64()*1000.;
            let expected=if batch{&candidate}else{&baseline};let expected_states=if batch{&batch_states}else{&serial_states};
            assert!(last.equal(expected));
            for (a,b) in states.iter().zip(expected_states){assert_eq!(states_max_diff(a,b),0.);}
            rounds.push(json!({"batch":batch,"ms_per_tree":ms}));
        }
        Tensor::save_multi(&[("baseline",baseline.to_device(Device::Cpu)),("candidate",candidate.to_device(Device::Cpu))],out.join(format!("{name}-rank{}.pt",tp.rank))).unwrap();
        cases.push(json!({"name":name,"nodes":ids,"parents":parents,"quality":quality,"state_max_abs":state_error,"rounds":rounds}));
        std::fs::write(out.join(format!("tree-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"cases":cases})).unwrap()).unwrap();
        eprintln!("[tree-probe] rank{} {name} complete",tp.rank);
    }
    assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
}
