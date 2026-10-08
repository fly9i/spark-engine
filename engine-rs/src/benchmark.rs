//! Paired eager/graph measurements: same resident weights, inputs and graph-compatible states.
use std::path::Path;
use tch::{Device, Kind, Tensor};
use crate::forward::{DecodeStates, Engine, LayerState};

pub(crate) use crate::forward::restore;

pub fn paired(dir: &Path, prompt: &str, n: usize) {
    let tp = crate::tp::init_from_env();
    assert_eq!(tp.world,2,"paired benchmark requires TP2 full residency");
    assert!(n > 0);
    for key in ["GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"] {
        assert!(std::env::var(key).is_err(), "paired benchmark incompatible with {key}");
    }
    let refs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        "bench/m0-refs.json").expect("refs")).unwrap();
    let ids: Vec<i64> = refs[prompt]["prompt_ids"].as_array().expect("prompt_ids").iter().map(|x| x.as_i64().unwrap()).collect();
    let tokens: Vec<i64> = refs[prompt]["text_ids"].as_array().expect("text_ids").iter().take(n).map(|x| x.as_i64().unwrap()).collect();
    assert_eq!(tokens.len(),n,"reference trace too short");
    let max_t: i64 = std::env::var("GLM53_GRAPH_WIN").ok().and_then(|s|s.parse().ok()).unwrap_or(512);
    assert!(ids.len() + n <= max_t as usize && max_t > 0);
    let dev = Device::Cuda(0);
    let cfg = crate::config::load(&dir.join("config.json")).expect("config");
    let w = crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast = crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,
        cfg.num_hidden_layers * cfg.n_routed_experts + 16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);
    fast.assume_hot = true;
    let mut eng = Engine { w, fast: Some(fast), native: None, pool: crate::moe::ExpertPool::new(dir,4) };
    let (_, state) = eng.prefill(&Tensor::from_slice(&ids).to_device(dev));
    let mut state = DecodeStates(state.0.into_iter().map(|s|match s {
        LayerState::Mla(s) => LayerState::MlaG(crate::mla::MlaStateG::from_state(&s,max_t)),
        other => other,
    }).collect());
    let initial = crate::forward::snapshot(&state);
    let mut input = Tensor::zeros([1],(Kind::Int64,dev));
    // Warm all exact input positions before either timed arm.
    for &tok in &tokens { input.copy_(&Tensor::from_slice(&[tok]).to_device(dev)); let _ = eng.step_buf(&input,&mut state); }
    restore(&mut state,&initial);
    input.copy_(&Tensor::from_slice(&[tokens[0]]).to_device(dev));
    tch::Cuda::synchronize(0);
    crate::tp::graph::begin().expect("capture begin");
    let graph_logits = eng.step_buf(&input,&mut state);
    crate::tp::graph::end().expect("capture end");
    crate::tp::graph::replay().expect("warm replay");
    tch::Cuda::synchronize(0);
    let misses_before = eng.fast.as_ref().unwrap().misses;
    let mut rounds = Vec::new();
    let mut reference: Option<(Tensor,DecodeStates)> = None;
    // ABBA reduces ordering bias. Each arm starts from identical state contents and addresses.
    for graph in [false,true,true,false] {
        restore(&mut state,&initial);
        tch::Cuda::synchronize(0);
        // Diagnostics between rounds can finish at different times on the two ranks.
        // Rendezvous outside the timer so one arm does not measure the peer's diagnostics.
        let ready = Tensor::zeros([1], (Kind::Float, dev));
        crate::tp::allreduce(&ready);
        tch::Cuda::synchronize(0);
        let start = std::time::Instant::now();
        let mut last = Tensor::new();
        for &tok in &tokens {
            input.copy_(&Tensor::from_slice(&[tok]).to_device(dev));
            if graph { crate::tp::graph::replay().expect("replay"); }
            else { last = eng.step_buf(&input,&mut state); }
        }
        tch::Cuda::synchronize(0);
        let ms = start.elapsed().as_secs_f64()*1000. / n as f64;
        let last = if graph { graph_logits.copy() } else { last.copy() };
        let (logit_error,state_error) = if let Some((lg,st)) = &reference {
            (f64::try_from((&last-lg).abs().max()).unwrap(),crate::forward::states_max_diff(&state,st))
        } else { reference=Some((last.copy(),crate::forward::snapshot(&state))); (0.,0.) };
        assert!(logit_error < 1e-3 && state_error < 1e-3,"paired output/state mismatch: {logit_error}/{state_error}");
        rounds.push(serde_json::json!({"mode":if graph {"graph"} else {"eager"},"ms_per_step":ms,
            "last_logit_max_error":logit_error,"state_max_error":state_error}));
    }
    assert_eq!(eng.fast.as_ref().unwrap().misses,misses_before,"timed window loaded experts");
    println!("[paired] {}",serde_json::json!({"rank":tp.rank,"dense_tp":crate::tp::dense_enabled(),
        "w_fp16":crate::weights::w_fp16(),"prompt":prompt,"tokens":tokens,"window":max_t,
        "state_layout":"MlaG fp16 + KDA fp32, both arms","residency":"full","cold_loads":0,
        "timing":"input copy + forward + final sync; excludes prefill/load/capture/diagnostic copies",
        "rounds":rounds}));
    if std::env::var("GLM53_QUALIFY").as_deref() == Ok("1") { qualification(&mut eng,&refs,n); }
}

fn qualification(eng: &mut Engine, refs: &serde_json::Value, n: usize) {
    let mut archive: Vec<(String,Tensor)> = Vec::new();
    for name in ["hello","count","hashmap"] {
        let ids: Vec<i64> = refs[name]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
        let reference: Vec<i64> = refs[name]["text_ids"].as_array().unwrap().iter().take(n).map(|v|v.as_i64().unwrap()).collect();
        let eager = eng.greedy_incremental_dbg(&ids,reference.len(),false).0;
        let graph = eng.greedy_incremental_graph(&ids,reference.len());
        let (first,matched,pass) = crate::acceptance::compare(&eager,&reference,reference.len());
        println!("[qualify] {}",serde_json::json!({"rank":crate::tp::world().rank,"prompt":name,
            "reference_pass":pass,"first_difference":first,"matched":matched,
            "graph_equals_eager":graph==eager,"eager_ids":eager,"graph_ids":graph}));
        assert_eq!(eager,graph,"graph/eager token mismatch for {name}");
        archive.push((format!("{name}_eager_ids"),Tensor::from_slice(&eager)));
        archive.push((format!("{name}_graph_ids"),Tensor::from_slice(&graph)));
        // Fixed reference prefix at every position, even after a greedy divergence.
        let (lg,mut state) = eng.prefill(&Tensor::from_slice(&ids).to_device(eng.w.device));
        let mut next = lg.get(lg.size()[0]-1);
        let mut logits = Vec::new();
        for (p,&tok) in reference.iter().enumerate() {
            logits.push(next.to_device(Device::Cpu));
            if p+1 < reference.len() { next=eng.step(tok,&mut state); }
        }
        archive.push((format!("{name}_logits"),Tensor::stack(&logits,0)));
    }
    let file = format!("/tmp/glm53-m1-qualify-{}-rank{}.pt", if crate::tp::dense_enabled(){"dense"}else{"replicated"},crate::tp::world().rank);
    Tensor::save_multi(&archive,&file).expect("save qualification");
    println!("[qualify] archive={file}");
}

pub fn compare_qualification(gold: &Path, candidate: &Path) {
    let gold = Tensor::load_multi_with_device(gold,Device::Cpu).expect("gold archive");
    let candidate: std::collections::HashMap<_,_> = Tensor::load_multi_with_device(candidate,Device::Cpu).expect("candidate archive").into_iter().collect();
    assert_eq!(gold.len(),candidate.len());
    let mut pass = true;
    for (name,a) in gold {
        let b = &candidate[&name]; assert_eq!(a.size(),b.size());
        if name.ends_with("_ids") {
            let equal = a.equal(b); pass &= equal;
            println!("[compare] {name} equal={equal}");
        } else {
            let relative = f64::try_from((&a-b).norm()/a.norm().clamp_min(1e-12)).unwrap();
            let max_error = f64::try_from((&a-b).abs().max()).unwrap();
            let per_position = (&a-b).norm_scalaropt_dim(2.0,[-1],false)
                / a.norm_scalaropt_dim(2.0,[-1],false).clamp_min(1e-12);
            let worst = f64::try_from(per_position.max()).unwrap();
            println!("[compare] {name} rel={relative:.6e} worst_position_rel={worst:.6e} max_error={max_error:.6e}");
            let centered_a = &a - a.mean_dim(&[-1i64][..],true,Kind::Float);
            let centered_b = b - b.mean_dim(&[-1i64][..],true,Kind::Float);
            let centered_rel = f64::try_from((&centered_a-&centered_b).norm()/centered_a.norm()).unwrap();
            let top1_equal = f64::try_from(a.argmax(-1,false).eq_tensor(&b.argmax(-1,false)).to_kind(Kind::Float).mean(None)).unwrap();
            let per: Vec<f32> = per_position.try_into().unwrap();
            println!("[compare] {name} centered_rel={centered_rel:.6e} teacher_top1_equal={top1_equal:.4} per_position={per:?}");
            pass &= relative < 3e-3 && worst < 3e-3;
        }
    }
    assert!(pass,"strict raw-logit comparison failed; assess probability distributions and task quality separately");
    println!("[compare] PASS");
}

pub fn dense_math_smoke() {
    let dev = Device::Cuda(0);
    let mut x = Tensor::ones([1,16],(Kind::Half,dev));
    let mut row = vec![0f32;16]; row[0]=2048.; row[1]=0.75; row[8]=-2048.; row[9]=0.75;
    let w = Tensor::from_slice(&row.repeat(16)).view([16,16]).to_kind(Kind::Half).to_device(dev);
    let a = crate::weights::mm16_partial(&x.narrow(1,0,8),&w.narrow(1,0,8));
    let b = crate::weights::mm16_partial(&x.narrow(1,8,8),&w.narrow(1,8,8));
    let sum = (&a+&b).to_kind(Kind::Half).to_kind(Kind::Float);
    assert!(sum.equal(&crate::weights::mm16(&x,&w)),"partial accumulation mismatch");
    let _ = crate::weights::mm16_partial(&x,&w);
    tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();
    let output = crate::weights::mm16_partial(&x,&w);
    crate::tp::graph::end().unwrap();
    let _ = x.fill_(2.);
    crate::tp::graph::replay().unwrap();
    tch::Cuda::synchronize(0);
    assert!(output.equal(&(&sum*2.)),"captured partial GEMM mismatch");
    println!("[dense-math-smoke] fp32 partial accumulation + graph replay PASS");
}

/// Isolate real-weight dense prefixes without routed-expert kernels.
pub fn dense_prefix_probe(dir: &Path, prompt: &str) {
    let tp = crate::tp::init_from_env();
    assert_eq!(tp.world,2);
    std::env::set_var("GLM53_DENSE_TP","0");
    let dev = Device::Cuda(0);
    let cfg = crate::config::load(&dir.join("config.json")).unwrap();
    let w = crate::weights::ModelWeights::load(dir,&cfg,3,dev);
    assert!(w.layers.iter().all(|l| l.dense.is_some()));
    let mut eng = Engine { w, fast: None, native: None, pool: crate::moe::ExpertPool::new(dir,4) };
    let refs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        "bench/m0-refs.json").unwrap()).unwrap();
    let ids: Vec<i64> = refs[prompt]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let input = Tensor::from_slice(&ids).to_device(dev);
    let mut baseline = Vec::new();
    for n in 1..=3 {
        let mut tail = eng.w.layers.split_off(n);
        let (lg,_) = eng.prefill(&input);
        baseline.push(lg.to_device(Device::Cpu));
        eng.w.layers.append(&mut tail);
    }
    for layer in &mut eng.w.layers { crate::weights::shard_dense_layer(layer,tp.rank,tp.world); }
    std::env::set_var("GLM53_DENSE_TP","1");
    for n in 1..=3 {
        let mut tail = eng.w.layers.split_off(n);
        let (lg,_) = eng.prefill(&input);
        let lg = lg.to_device(Device::Cpu);
        let gold = &baseline[n-1];
        let rel = f64::try_from((&lg-gold).norm()/gold.norm()).unwrap();
        let max = f64::try_from((&lg-gold).abs().max()).unwrap();
        println!("[dense-prefix-probe] rank={} layers={n} rel={rel:.6e} max_abs={max:.6e}",tp.rank);
        eng.w.layers.append(&mut tail);
    }
}
