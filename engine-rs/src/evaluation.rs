//! Expanded TP diagnostics, with one resident expert pool and two dense-weight layouts.
//! No numerical acceptance threshold: persist evidence for probability/task assessment.
use std::{cell::RefCell, collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};
use crate::forward::{Engine, snapshot};

thread_local! {
    static ROUTES: RefCell<Option<Vec<(Tensor,Tensor)>>> = const { RefCell::new(None) };
    static REPLAY: RefCell<Option<(Vec<(Tensor,Tensor)>,usize,bool)>> = const { RefCell::new(None) };
}

/// Diagnostic intervention only: substitute the recorded selection, optionally coefficients.
pub(crate) fn replay_active()->bool {REPLAY.with(|r|r.borrow().is_some())}
pub fn replay_route(natural: Tensor) -> (Tensor,Option<Tensor>) {
    REPLAY.with(|r| {
        let mut r=r.borrow_mut();
        if let Some((plan,index,coefficients))=r.as_mut() {
            let (ids,w)=plan.get(*index).expect("route replay exhausted");
            assert_eq!(natural.size(),ids.size(),"route replay shape mismatch");
            *index+=1;
            (ids.shallow_clone(),if *coefficients {Some(w.shallow_clone())} else {None})
        } else {(natural,None)}
    })
}

/// Inactive outside this diagnostic. Keep device tensors until the trace ends.
pub fn record_route(ids: &Tensor, weights: &Tensor) {
    ROUTES.with(|r| {
        if let Some(trace) = r.borrow_mut().as_mut() {
            trace.push((ids.shallow_clone(),weights.shallow_clone()));
        }
    });
}

fn scalar(t: Tensor) -> f64 { f64::try_from(t).unwrap() }
fn stats(t: &Tensor) -> serde_json::Value {
    let x: Vec<f64> = t.to_kind(Kind::Double).view([-1]).try_into().unwrap();
    assert!(!x.is_empty() && x.iter().all(|v|v.is_finite()));
    let mut sorted=x.clone(); sorted.sort_by(f64::total_cmp);
    serde_json::json!({"mean":x.iter().sum::<f64>()/x.len() as f64,
        "p95":sorted[((sorted.len() as f64*0.95).ceil() as usize-1).min(sorted.len()-1)],
        "max":sorted[sorted.len()-1],"per_position":x})
}

pub fn distribution(a: &Tensor, b: &Tensor, reference: &[i64]) -> serde_json::Value {
    assert_eq!(a.size(),b.size()); assert_eq!(a.size().len(),2);
    assert_eq!(a.size()[0] as usize,reference.len());
    let a=a.to_kind(Kind::Double); let b=b.to_kind(Kind::Double);
    assert_eq!(scalar(a.isfinite().all().to_kind(Kind::Double)),1.);
    assert_eq!(scalar(b.isfinite().all().to_kind(Kind::Double)),1.);
    let la=a.log_softmax(-1,Kind::Double); let lb=b.log_softmax(-1,Kind::Double);
    let p=la.exp(); let q=lb.exp(); let abs=(&p-&q).abs();
    let kl=(&p*(&la-&lb)).sum_dim_intlist(&[-1i64][..],false,Kind::Double);
    let tv=abs.sum_dim_intlist(&[-1i64][..],false,Kind::Double)*0.5;
    let ids=Tensor::from_slice(reference).unsqueeze(-1);
    let na=la.gather(-1,&ids,false).neg().squeeze_dim(-1);
    let nb=lb.gather(-1,&ids,false).neg().squeeze_dim(-1);
    let diff=(&a-&b).norm_scalaropt_dim(2.0,[-1],false)/a.norm_scalaropt_dim(2.0,[-1],false).clamp_min(1e-12);
    let agree=a.argmax(-1,false).eq_tensor(&b.argmax(-1,false)).to_kind(Kind::Double);
    serde_json::json!({"positions":reference.len(),"top1_equal_fraction":scalar(agree.mean(Kind::Double)),
        "top1_equal_per_position":Vec::<f64>::try_from(agree).unwrap(),
        "raw_logits_relative_l2":stats(&diff),"kl_baseline_to_candidate_nats":stats(&kl),
        "total_variation":stats(&tv),"max_probability_absolute_difference":stats(&abs.max_dim(-1,false).0),
        "reference_nll_baseline":stats(&na),"reference_nll_candidate":stats(&nb),
        "reference_nll_delta_candidate_minus_baseline":stats(&(&nb-&na))})
}

fn ints(v: &serde_json::Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect()
}

fn evaluate_case(eng: &mut Engine, case: &serde_json::Value, eos: &[i64]) -> Vec<(String,Tensor)> {
    let ids=ints(&case["prompt_ids"]); let reference=ints(&case["reference_ids"]);
    assert!(!ids.is_empty() && !reference.is_empty());
    ROUTES.with(|r| *r.borrow_mut()=Some(Vec::new()));
    let (lg,mut state,_)=eng.prefill_record_last(&Tensor::from_slice(&ids).to_device(eng.w.device),None,false);
    let initial=snapshot(&state);
    let first=lg.get(lg.size()[0]-1);
    let mut next=first.shallow_clone(); let mut logits=Vec::new();
    for (p,&tok) in reference.iter().enumerate() {
        logits.push(next.to_device(Device::Cpu));
        if p+1<reference.len() { next=eng.step(tok,&mut state); }
    }
    let routes=ROUTES.with(|r| r.borrow_mut().take().unwrap());
    drop(state);
    let mut state=initial; next=first;
    let mut generated=Vec::new(); let mut stopped=false;
    for p in 0..case["max_new"].as_u64().unwrap() {
        let tok=next.argmax(-1,false).int64_value(&[]);
        if eos.contains(&tok) { stopped=true; break; }
        generated.push(tok);
        if p+1<case["max_new"].as_u64().unwrap() { next=eng.step(tok,&mut state); }
    }
    println!("[eval-case] {}",serde_json::json!({"rank":crate::tp::world().rank,
        "dense_tp":crate::tp::dense_enabled(),"name":case["name"],"generated_ids":generated,
        "stopped_eos":stopped,"teacher_positions":reference.len(),"route_calls":routes.len()}));
    let mut archive=vec![("logits".into(),Tensor::stack(&logits,0)),
        ("generated_ids".into(),Tensor::from_slice(&generated)),
        ("stopped_eos".into(),Tensor::from(i64::from(stopped)))];
    for (i,(ids,w)) in routes.into_iter().enumerate() {
        archive.push((format!("route_ids_{i:05}"),ids.to_device(Device::Cpu)));
        archive.push((format!("route_weights_{i:05}"),w.to_device(Device::Cpu)));
    }
    archive
}

fn compare_case(gold: &Path, candidate: &[(String,Tensor)], case: &serde_json::Value, layers: &[usize]) -> serde_json::Value {
    let gold: HashMap<_,_>=Tensor::load_multi_with_device(gold,Device::Cpu).unwrap().into_iter().collect();
    let cand: HashMap<_,_>=candidate.iter().map(|(k,v)|(k.as_str(),v)).collect();
    let reference=ints(&case["reference_ids"]);
    let mut metrics=distribution(&gold["logits"],cand["logits"],&reference);
    metrics["logits_exact"]=serde_json::json!(gold["logits"].equal(cand["logits"]));
    metrics["archive_exact"]=serde_json::json!(gold.len()==candidate.len() &&
        candidate.iter().all(|(k,v)|gold.get(k).map_or(false,|g|g.equal(v))));
    let a: Vec<i64>=(&gold["generated_ids"]).try_into().unwrap();
    let b: Vec<i64>=cand["generated_ids"].try_into().unwrap();
    let first=a.iter().zip(&b).position(|(x,y)|x!=y).or_else(||if a.len()!=b.len(){Some(a.len().min(b.len()))}else{None});
    let mut routing=Vec::new();
    for (k,v) in candidate {
        if let Some(index)=k.strip_prefix("route_ids_") {
            let i: usize=index.parse().unwrap(); let g=&gold[k];
            assert_eq!(g.size(),v.size());
            let same=g.sort(-1,false).0.eq_tensor(&v.sort(-1,false).0).all_dim(-1,false).to_kind(Kind::Double);
            let changed=g.size()[0] as f64-scalar(same.sum(Kind::Double));
            let shared=g.unsqueeze(-1).eq_tensor(&v.unsqueeze(-2)).any_dim(-1,false).to_kind(Kind::Double).sum(Kind::Double);
            routing.push(serde_json::json!({"layer":layers[i%layers.len()],"phase":if i<layers.len(){"prefill"}else{"teacher_step"},
                "teacher_input_position":if i<layers.len(){None}else{Some(i/layers.len()-1)},
                "tokens":g.size()[0],"changed_expert_sets":changed,"shared_expert_slots":scalar(shared),"expert_slots":g.numel()}));
        }
    }
    metrics["name"]=case["name"].clone(); metrics["greedy_first_difference"]=serde_json::json!(first);
    metrics["greedy_equal"]=serde_json::json!(a==b);
    metrics["baseline_generated_ids"]=serde_json::json!(a); metrics["candidate_generated_ids"]=serde_json::json!(b);
    metrics["baseline_stopped_eos"]=serde_json::json!(gold["stopped_eos"].int64_value(&[])!=0);
    metrics["candidate_stopped_eos"]=serde_json::json!(cand["stopped_eos"].int64_value(&[])!=0);
    metrics["routing"]=serde_json::json!(routing);
    metrics
}

pub fn suite(dir: &Path, suite_file: &Path, output: &Path) {
    let tp=crate::tp::init_from_env(); assert_eq!(tp.world,2);
    for key in ["GLM53_GRAPH","GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"] {
        assert!(std::env::var(key).is_err(),"evaluation incompatible with {key}");
    }
    tch::set_num_threads(4);
    let suite: serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite_file).unwrap()).unwrap();
    let cases=suite["cases"].as_array().unwrap(); let eos=ints(&suite["eos_token_ids"]);
    std::env::set_var("GLM53_DENSE_TP","0");
    let cfg=crate::config::load(&dir.join("config.json")).unwrap(); let dev=Device::Cuda(0);
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let layers: Vec<usize>=w.layers.iter().enumerate().filter_map(|(i,l)|l.moe.as_ref().map(|_|i)).collect();
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev); fast.assume_hot=true;
    let mut eng=Engine {w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    let mut results=Vec::new(); let misses=eng.fast.as_ref().unwrap().misses;
    for mode in ["replicated","dense"] {
        if mode=="dense" {
            tch::Cuda::synchronize(0);
            for layer in &mut eng.w.layers { crate::weights::shard_dense_layer(layer,tp.rank,tp.world); }
            std::env::set_var("GLM53_DENSE_TP","1");
        }
        if tp.rank==0 { std::fs::create_dir_all(output.join(mode)).unwrap(); }
        for case in cases {
            let name=case["name"].as_str().unwrap();
            assert!(name.chars().all(|c|c.is_ascii_alphanumeric()||c=='_'));
            let archive=evaluate_case(&mut eng,case,&eos);
            if tp.rank==0 {
                Tensor::save_multi(&archive,output.join(mode).join(format!("{name}.pt"))).unwrap();
                if mode=="dense" {
                    let result=compare_case(&output.join("replicated").join(format!("{name}.pt")),&archive,case,&layers);
                    println!("[eval-metrics] name={name} top1={} KL_mean={} TV_max={} NLL_delta_mean={}",result["top1_equal_fraction"],result["kl_baseline_to_candidate_nats"]["mean"],result["total_variation"]["max"],result["reference_nll_delta_candidate_minus_baseline"]["mean"]);
                    results.push(result);
                    std::fs::write(output.join("metrics.json"),serde_json::to_string_pretty(&serde_json::json!({
                        "suite":suite_file,"temperature":1.0,"state_layout":"eager fp32 MLA cache / fp32 KDA; identical between arms",
                        "reference_note":"Production continuations are not ground truth. Local diagnostic suite, not a general quality benchmark.",
                        "results":results})).unwrap()).unwrap();
                }
            }
        }
    }
    assert_eq!(eng.fast.as_ref().unwrap().misses,misses,"unexpected expert cold loads");
    println!("[eval-suite] completed rank={} cases={} cold_loads=0",tp.rank,cases.len());
}

/// Same TP layout and quantized weights; qualify the combined optimization flags.
pub fn combined(dir:&Path,suite_file:&Path,output:&Path) {
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    assert!(crate::tp::dense_enabled(),"combined qualification requires dense TP2");
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite_file).unwrap()).unwrap();
    let eos=ints(&suite["eos_token_ids"]);let cases=suite["cases"].as_array().unwrap();
    std::env::set_var("GLM53_MLA_LATENT","1");
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();let dev=Device::Cuda(0);
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let layers:Vec<_>=w.layers.iter().enumerate().filter_map(|(i,l)|l.moe.as_ref().map(|_|i)).collect();
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;let misses=fast.misses;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    combined_check(&mut eng,suite_file,output);
}

pub fn combined_check(eng:&mut Engine,suite_file:&Path,output:&Path) {
    flags_check(eng,suite_file,output,&["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT","GLM53_PREFILL_BATCH"]);
}

pub fn flags_check(eng:&mut Engine,suite_file:&Path,output:&Path,flags:&[&str]) {
    let settings:Vec<_>=flags.iter().map(|&f|(f,"0","1")).collect();
    settings_check(eng,suite_file,output,&settings);
}
pub fn settings_check(eng:&mut Engine,suite_file:&Path,output:&Path,settings:&[(&str,&str,&str)]) {
    let tp=crate::tp::world();let misses=eng.fast.as_ref().unwrap().misses;
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite_file).unwrap()).unwrap();
    let eos=ints(&suite["eos_token_ids"]);let cases=suite["cases"].as_array().unwrap();
    let layers:Vec<_>=eng.w.layers.iter().enumerate().filter_map(|(i,l)|l.moe.as_ref().map(|_|i)).collect();
    let mut results=Vec::new();
    for on in [false,true] {
        for &(flag,off,value) in settings {std::env::set_var(flag,if on{value}else{off});}
        crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
        let mode=if on{"candidate"}else{"baseline"};
        if tp.rank==0{std::fs::create_dir_all(output.join(mode)).unwrap();}
        for case in cases {
            let name=case["name"].as_str().unwrap();let archive=evaluate_case(eng,case,&eos);
            if tp.rank==0 {
                Tensor::save_multi(&archive,output.join(mode).join(format!("{name}.pt"))).unwrap();
                if on {
                    results.push(compare_case(&output.join("baseline").join(format!("{name}.pt")),&archive,case,&layers));
                    std::fs::write(output.join("metrics.json"),serde_json::to_string_pretty(&serde_json::json!({
                        "suite":suite_file,"dense_tp":true,"candidate_settings":settings,"cold_loads":0,
                        "reference_note":"Identical 4bpw weights. Short diagnostic tasks, not a general benchmark; archived continuations are not ground truth.",
                        "results":results})).unwrap()).unwrap();
                }
            }
        }
    }
    assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
    eprintln!("[combined-qualification] rank{} PASS {} cases",tp.rank,cases.len());
}

/// Natural routing vs frozen IDs vs frozen IDs+weights, on the same dense-TP weights.
pub fn replay_probe(dir: &Path, suite_file: &Path, gold_dir: &Path, output: &Path, names: &str) {
    let tp=crate::tp::init_from_env(); assert_eq!(tp.world,2);
    for key in ["GLM53_GRAPH","GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"] {
        assert!(std::env::var(key).is_err(),"replay probe incompatible with {key}");
    }
    tch::set_num_threads(4);
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite_file).unwrap()).unwrap();
    let selected:Vec<_>=names.split(',').collect();
    std::env::set_var("GLM53_DENSE_TP","1");
    let cfg=crate::config::load(&dir.join("config.json")).unwrap(); let dev=Device::Cuda(0);
    let w=crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let routed=w.layers.iter().filter(|l|l.moe.is_some()).count();
    let mut fast=crate::moefast::MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};
    let mut results=Vec::new();
    if tp.rank==0 {std::fs::create_dir_all(output).unwrap();}
    for name in selected {
        let case=suite["cases"].as_array().unwrap().iter().find(|c|c["name"]==name).expect("unknown case");
        let gold:HashMap<_,_>=Tensor::load_multi_with_device(gold_dir.join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();
        let ids=ints(&case["prompt_ids"]); let reference=ints(&case["reference_ids"]);
        let mut archive=Vec::new();
        for (mode,replay,coefficients) in [("natural",false,false),("fixed_ids",true,false),("fixed_ids_weights",true,true)] {
            if replay {
                let plan:Vec<_>=(0..routed*reference.len()).map(|i| (
                    gold[&format!("route_ids_{i:05}")].to_device(dev),
                    gold[&format!("route_weights_{i:05}")].to_device(dev))).collect();
                REPLAY.with(|r| *r.borrow_mut()=Some((plan,0,coefficients)));
            }
            let (lg,mut state)=eng.prefill(&Tensor::from_slice(&ids).to_device(dev));
            let mut next=lg.get(lg.size()[0]-1); let mut logits=Vec::new();
            for (p,&tok) in reference.iter().enumerate() {
                logits.push(next.to_device(Device::Cpu));
                if p+1<reference.len() {next=eng.step(tok,&mut state);}
            }
            if replay {
                REPLAY.with(|r| {
                    let (plan,used,_)=r.borrow_mut().take().unwrap();
                    assert_eq!(used,plan.len(),"unused replay routes");
                });
            }
            let logits=Tensor::stack(&logits,0);
            if tp.rank==0 {
                let mut metrics=distribution(&gold["logits"],&logits,&reference);
                metrics["name"]=serde_json::json!(name); metrics["mode"]=serde_json::json!(mode);
                println!("[replay-metrics] name={name} mode={mode} KL_mean={} TV_max={} raw_L2_max={}",metrics["kl_baseline_to_candidate_nats"]["mean"],metrics["total_variation"]["max"],metrics["raw_logits_relative_l2"]["max"]);
                results.push(metrics); archive.push((mode.to_string(),logits));
                std::fs::write(output.join("metrics.json"),serde_json::to_string_pretty(&serde_json::json!({
                    "note":"Counterfactual diagnostic; forced routes/coefficients are not a valid replacement inference mode. Same dense-TP weights for all three arms.","results":results})).unwrap()).unwrap();
            }
        }
        if tp.rank==0 {Tensor::save_multi(&archive,output.join(format!("{name}.pt"))).unwrap();}
    }
    println!("[replay-probe] completed rank={}",tp.rank);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probabilities_ignore_common_logit_shift() {
        let a=Tensor::from_slice(&[1f64,2.,3.,-1.,4.,0.]).view([2,3]);
        let m=distribution(&a,&(&a+100.),&[2,1]);
        for key in ["kl_baseline_to_candidate_nats","total_variation","reference_nll_delta_candidate_minus_baseline"] {
            assert!(m[key]["max"].as_f64().unwrap().abs()<1e-12);
        }
        assert_eq!(m["top1_equal_fraction"],1.0);
    }
    #[test]
    fn probability_change_is_visible_without_top1_change() {
        let a=Tensor::from_slice(&[2f64,0.]).view([1,2]); let b=&a*2.;
        let m=distribution(&a,&b,&[0]);
        assert_eq!(m["top1_equal_fraction"],1.0);
        assert!(m["kl_baseline_to_candidate_nats"]["mean"].as_f64().unwrap()>0.1);
        assert!(m["reference_nll_delta_candidate_minus_baseline"]["mean"].as_f64().unwrap()<0.);
    }

    #[test]
    fn route_replay_distinguishes_ids_and_coefficients() {
        for freeze_weights in [false,true] {
            REPLAY.with(|r| *r.borrow_mut()=Some((vec![(Tensor::from_slice(&[2i64,1]).view([1,2]),
                Tensor::from_slice(&[0.7f32,0.3]).view([1,2]))],0,freeze_weights)));
            let (ids,w)=replay_route(Tensor::from_slice(&[0i64,1]).view([1,2]));
            assert_eq!(ids.int64_value(&[0,0]),2); assert_eq!(w.is_some(),freeze_weights);
            REPLAY.with(|r|assert_eq!(r.borrow_mut().take().unwrap().1,1));
        }
        let (ids,w)=replay_route(Tensor::from_slice(&[0i64,1]));
        assert_eq!(ids.int64_value(&[0]),0);assert!(w.is_none());
    }
}
