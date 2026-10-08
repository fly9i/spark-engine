//! Native fusion validation and resident-model comparisons.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;

pub fn kda_local(out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    tch::manual_seed(20260922);let mut cases=Vec::new();
    for heads in [1,32,64] {for decay_scale in [0.,0.1,1.] {
        let random=||Tensor::randn([heads,128],(Kind::Float,dev));
        let norm=|x:Tensor|{let scale=(&x*&x).sum_dim_intlist(&[-1i64][..],true,Kind::Float).sqrt();x/scale};
        let q=norm(random())/128f64.sqrt();let k=norm(random());
        let v=random();let beta=Tensor::rand([heads],(Kind::Float,dev));
        let decay=Tensor::ones([heads,128],(Kind::Float,dev))*decay_scale;
        let initial=Tensor::randn([heads,128,128],(Kind::Float,dev))*0.1;
        let mut reference=initial.copy();let mut state=initial.copy();let mut max_error=0f64;
        for _ in 0..256 {
            let expected=crate::kda::recurrent(&mut reference,&q,&k,&v,&beta,&decay,false);
            let actual=crate::kda::recurrent(&mut state,&q,&k,&v,&beta,&decay,true);
            max_error=max_error.max(f64::try_from((&expected-&actual).abs().max()).unwrap());
        }
        let state_error=f64::try_from((&state-&reference).abs().max()).unwrap();
        assert!(max_error<2e-5 && state_error<2e-5,"KDA drift output={max_error} state={state_error}");
        let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            let mut input=q.copy();state.copy_(&initial);
            for _ in 0..3{let _=crate::kda::recurrent(&mut state,&input,&k,&v,&beta,&decay,fused);}
            tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();
            let actual=crate::kda::recurrent(&mut state,&input,&k,&v,&beta,&decay,fused);
            crate::tp::graph::end().unwrap();
            state.copy_(&initial);reference.copy_(&initial);
            for x in [&q,&(-&q),&q] {
                input.copy_(x);crate::tp::graph::replay().unwrap();
                let expected=crate::kda::recurrent(&mut reference,x,&k,&v,&beta,&decay,fused);
                assert!(actual.equal(&expected));assert!(state.equal(&reference));
            }
            tch::Cuda::synchronize(0);let start=Instant::now();
            for _ in 0..256{crate::tp::graph::replay().unwrap();}
            tch::Cuda::synchronize(0);let us=start.elapsed().as_secs_f64()*1e6/256.;
            crate::tp::graph::destroy();rounds.push(json!({"fused":fused,"graph_us":us}));
        }
        cases.push(json!({"heads":heads,"decay":decay_scale,"steps":256,"output_max_abs":max_error,"state_max_abs":state_error,"rounds":rounds}));
    }}
    std::fs::write(out.join("local.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
    println!("[kda-local] PASS nine shapes/decays, 256 steps, changed-input graph replay");
}

pub fn mhc_local(out: &Path) {
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    std::fs::create_dir_all(out).unwrap();
    let dev = Device::Cuda(0);
    tch::manual_seed(327);
    let mut cases = Vec::new();
    for rows in [1,2,7,8,9,32,256,2100] {
        for scale in [0.,1.,10.,100.] {
            let input = (Tensor::randn([rows,4,4],(Kind::Float,dev))*scale).softmax(-1,Kind::Float)+1e-6;
            let gold = crate::mhc::sinkhorn(input.copy(),false);
            let actual = crate::mhc::sinkhorn(input.copy(),true);
            let error = f64::try_from((&gold-&actual).abs().max()).unwrap();
            assert!(error < 1e-6,"Sinkhorn error {error}");
            let cpu = crate::mhc::sinkhorn(input.to_device(Device::Cpu),false).to_device(dev);
            assert!(f64::try_from((&cpu-&actual).abs().max()).unwrap()<1e-6);
            assert!(actual.isfinite().all().int64_value(&[])!=0);
            assert!(actual.min().double_value(&[])>0.);
            let col = actual.sum_dim_intlist(&[1i64][..],false,Kind::Float);
            assert!(f64::try_from((col-1.).abs().max()).unwrap()<2e-6);
            let mut rounds=Vec::new();
            let mut x=input.copy();
            for fused in [false,true,true,false] {
                for _ in 0..3 { let _=crate::mhc::sinkhorn(x.shallow_clone(),fused); }
                tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();
                let result=crate::mhc::sinkhorn(x.shallow_clone(),fused);
                crate::tp::graph::end().unwrap();
                for source in [&input,&input.flip([0,1]),&input] {
                    x.copy_(source);crate::tp::graph::replay().unwrap();
                    let expected=crate::mhc::sinkhorn(source.shallow_clone(),false);
                    assert!(f64::try_from((&result-&expected).abs().max()).unwrap()<1e-6);
                }
                tch::Cuda::synchronize(0);
                let start=Instant::now();
                for _ in 0..128 { crate::tp::graph::replay().unwrap(); }
                tch::Cuda::synchronize(0);
                let us=start.elapsed().as_secs_f64()*1e6/128.;
                rounds.push(json!({"fused":fused,"graph_us":us}));
                crate::tp::graph::destroy();
            }
            cases.push(json!({"rows":rows,"scale":scale,"max_abs":error,"exact":gold.equal(&actual),"rounds":rounds}));
        }
    }
    std::fs::write(out.join("local.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
    println!("[mhc-local] PASS {} shapes/scales, CPU/GPU references, graph input changes",cases.len());
}

pub fn mhc_full(dir: &Path, out: &Path) { full_compare(dir,out,"GLM53_MHC_FUSED"); }
pub fn latent_full(dir: &Path, out: &Path) { full_compare(dir,out,"GLM53_MLA_LATENT"); }

pub fn kda_full(dir:&Path,out:&Path) {full_compare(dir,out,"GLM53_KDA_FUSED");}
pub fn tp_pack_full(dir:&Path,out:&Path) {full_compare(dir,out,"GLM53_TP_MOE_PACK");}
pub fn perf_target(dir:&Path,flag:&str,out:&Path) {
    assert!(["GLM53_DENSE_FP8","GLM53_TF32","GLM53_MLA_SPARSE_FUSED","GLM53_DENSE_LT"].contains(&flag));
    full_compare(dir,out,flag);
}

fn full_compare(dir: &Path, out: &Path, flag:&str) {
    use crate::forward::{Engine,DecodeStates,LayerState};
    tch::set_num_threads(4);
    let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    let dev=Device::Cuda(0);
    std::fs::create_dir_all(out).unwrap();
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(
        "bench/m0-refs.json").unwrap()).unwrap();
    if flag!="GLM53_MHC_FUSED" {std::env::set_var("GLM53_MLA_LATENT","1");std::env::set_var("GLM53_MHC_FUSED","1");}
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,
        cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let misses=fast.misses;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    full_check(&mut eng,out,flag);
}

fn set_flag(eng:&mut crate::forward::Engine,flag:&str,on:bool) {
    std::env::set_var(flag,if on{"1"}else{"0"});
    if flag=="GLM53_TF32" {crate::tp::set_tf32(on);}
    if flag=="GLM53_SCRATCH_EMPTY" {
        eng.fast.as_mut().unwrap().scratch_mode=if on{crate::moefast::ScratchMode::Empty}else{crate::moefast::ScratchMode::Zeros};
        std::env::set_var("GLM53_MOE_SCRATCH",if on{"empty"}else{"zeros"});
    }
}

pub fn full_check(eng:&mut crate::forward::Engine,out:&Path,flag:&str) {
    use crate::forward::{DecodeStates,LayerState};
    let tp=crate::tp::world();let dev=eng.w.device;let misses=eng.fast.as_ref().unwrap().misses;
    std::fs::create_dir_all(out).unwrap();
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let mut cases=Vec::new();
    for name in ["hello","count","hashmap"] {
        if let Ok(selected)=std::env::var("GLM53_BENCH_CASES") {
            let names:Vec<_>=selected.split(',').collect();
            assert!(names.iter().all(|n|["hello","count","hashmap"].contains(n)),"invalid benchmark case");
            if !names.contains(&name){continue;}
        }
        let ints=|key:&str|refs[name][key].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect::<Vec<_>>();
        let ids=ints("prompt_ids");let tokens=ints("text_ids");
        let ids_t=Tensor::from_slice(&ids).to_device(dev);
        let mut quality=Vec::new();let mut greedy=Vec::new();
        for fused in [false,true] {
            set_flag(eng,flag,fused);
            let (lg,mut st)=eng.prefill(&ids_t);
            let mut next=lg.get(lg.size()[0]-1);let mut trace=Vec::new();
            for (p,&tok) in tokens.iter().enumerate() {
                trace.push(next.to_device(Device::Cpu));
                if p+1<tokens.len(){next=eng.step(tok,&mut st);}
            }
            quality.push(Tensor::stack(&trace,0));
            greedy.push(eng.greedy_incremental_dbg(&ids,tokens.len(),false).0);
        }
        let distribution=crate::evaluation::distribution(&quality[0],&quality[1],&tokens);
        let mut archives=vec![("baseline_logits".to_owned(),quality.remove(0)),("fused_logits".to_owned(),quality.remove(0))];
        archives.push(("baseline_ids".into(),Tensor::from_slice(&greedy[0])));
        archives.push(("fused_ids".into(),Tensor::from_slice(&greedy[1])));
        Tensor::save_multi(&archives,out.join(format!("{name}-rank{}.pt",tp.rank))).unwrap();
        let steps=std::env::var("GLM53_BENCH_STEPS").ok().map(|s|s.parse::<usize>().unwrap()).unwrap_or(tokens.len());
        assert!(steps>0 && steps<=1024,"benchmark step count outside 1..1024");
        let inputs:Vec<_>=tokens.iter().cycle().take(steps).map(|&t|Tensor::from_slice(&[t]).to_device(dev)).collect();
        let mut input=inputs[0].copy();let mut rounds=Vec::new();
        for graph in [false,true] {
            if !graph && std::env::var("GLM53_BENCH_GRAPH_ONLY").as_deref()==Ok("1"){continue;}
            for fused in [false,true,true,false] {
                set_flag(eng,flag,fused);
                let (_,st)=eng.prefill(&ids_t);
                let mut state=DecodeStates(st.0.into_iter().map(|s|match s {
                    LayerState::Mla(m)=>LayerState::MlaG(crate::mla::MlaStateG::from_state(&m,512)),other=>other,
                }).collect());
                let initial=crate::forward::snapshot(&state);
                crate::benchmark::restore(&mut state,&initial);
                let _=eng.step_buf(&input,&mut state);tch::Cuda::synchronize(0);
                let captured=if graph {
                    crate::tp::graph::begin().unwrap();let y=eng.step_buf(&input,&mut state);
                    crate::tp::graph::end().unwrap();Some(y)
                } else {None};
                let mut warm=Tensor::new();
                for _ in 0..3 {
                    crate::benchmark::restore(&mut state,&initial);
                    for tok in &inputs {
                        input.copy_(tok);
                        if graph {crate::tp::graph::replay().unwrap();}
                        else {warm=eng.step_buf(&input,&mut state);}
                    }
                }
                let expected=captured.as_ref().unwrap_or(&warm).copy();
                let expected_state=crate::forward::snapshot(&state);
                crate::benchmark::restore(&mut state,&initial);
                tch::Cuda::synchronize(0);
                crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
                let start=Instant::now();let mut last=Tensor::new();
                for tok in &inputs {
                    input.copy_(tok);
                    if graph {crate::tp::graph::replay().unwrap();}else{last=eng.step_buf(&input,&mut state);}
                }
                tch::Cuda::synchronize(0);let ms=start.elapsed().as_secs_f64()*1000./steps as f64;
                assert!(expected.equal(captured.as_ref().unwrap_or(&last)),"repeat logits differ");
                assert_eq!(crate::forward::states_max_diff(&state,&expected_state),0.,"repeat state differs");
                rounds.push(json!({"fused":fused,"graph":graph,"steps":steps,"ms_per_step":ms}));
                if graph {crate::tp::graph::destroy();}
                eprintln!("[mhc-full] rank={} {name} fused={fused} graph={graph} {ms:.3} ms/step",tp.rank);
            }
        }
        assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
        cases.push(json!({"name":name,"quality":distribution,"greedy_equal":greedy[0]==greedy[1],
            "baseline_ids":greedy[0],"fused_ids":greedy[1],"rounds":rounds}));
        std::fs::write(out.join(format!("full-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({
            "candidate_flag":flag,"rank":tp.rank,"dense_tp":crate::tp::dense_enabled(),"cold_loads":0,"cases":cases,
            "timing":"device input copy + forward/NCCL + final sync; three warm traces; excludes prefill/load/capture/diagnostics"})).unwrap()).unwrap();
    }
}
