//! Performance experiment for EXL3 scratch initialization, with exact-output guards.
use std::{collections::HashMap, path::Path, time::Instant};
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};
use crate::{moefast::{MoeFast, ScratchMode}, tp::{self, Tp}};

const ORDER: [ScratchMode; 6] = [ScratchMode::Zeros, ScratchMode::Empty, ScratchMode::Reuse,
    ScratchMode::Reuse, ScratchMode::Empty, ScratchMode::Zeros];

fn save(out: &Path, name: &str, value: &Value) {
    std::fs::write(out.join(name), serde_json::to_string_pretty(value).unwrap()).unwrap();
}

pub fn local(dir: &Path, archives: &Path, out: &Path) {
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    std::fs::create_dir_all(out).unwrap();
    let idx = crate::safetensors::ShardIndex::scan(dir).unwrap();
    let mut results = Vec::new();
    let mut checks = 0;
    for name in ["hello", "structured_count"] {
        let load = |arm: &str| -> HashMap<String, Tensor> {
            Tensor::load_multi_with_device(archives.join(arm).join(format!("{name}.pt")), Device::Cpu)
                .unwrap().into_iter().collect()
        };
        let base = load("replicated");
        let pert = load("dense");
        let samples = if name == "hello" { vec![(0,3),(0,4),(0,28),(0,44)] }
            else { vec![(0,3),(0,15),(22,15),(22,44)] };
        for (step, layer) in samples {
            let call = step * 42 + layer - 3;
            let key = format!("input_{call:05}");
            let token = base[&key].size()[0] - 1;
            let inputs = [&base[&key], &pert[&key]].map(|t| t.get(token).unsqueeze(0).to_device(dev).to_kind(Kind::Half));
            let ids = base.get(&format!("selected_{call:05}"))
                .map(|t|t.to_device(dev))
                .unwrap_or_else(||base[&format!("score_{call:05}")].to_device(dev).topk(8,-1,true,true).1)
                .get(token);
            let selected: Vec<usize> = Vec::<i64>::try_from(ids.to_device(Device::Cpu)).unwrap()
                .into_iter().map(|v| v as usize).collect();
            let (wv, shape) = idx.get_f32(&format!("model.language_model.layers.{layer}.mlp.gate.weight")).unwrap();
            let wg = Tensor::from_slice(&wv).view([shape[0] as i64, shape[1] as i64]).to_device(dev);
            let scores = base[&key].to_device(dev).matmul(&wg.transpose(0,1)).sigmoid();
            let weights = scores.get(token).gather(0, &ids, false);
            let weights = &weights / weights.sum(Kind::Float) * 2.5;
            // Also exercise unsharded geometry; performance numbers below use TP2 only.
            for spec in [Tp { rank:0, world:2 }, Tp { rank:1, world:2 }, Tp { rank:0, world:1 }] {
                let mut pool = MoeFast::new_with_tp(dir, 45, 288, 16, dev, spec);
                let selected = pool.probe_selection(layer, &selected, dev);
                let misses = pool.misses;
                for k in [8,2,4,8] {
                    let sel = selected.narrow(1,0,k).contiguous();
                    let w = weights.narrow(0,0,k).contiguous();
                    for precision in [0,1,2,0] {
                        pool.projection_precision = precision;
                        pool.scratch_mode = ScratchMode::Zeros;
                        let gold = inputs.each_ref().map(|x| pool.expert_batch_sel(x,&sel,&w,dev));
                        for mode in [ScratchMode::Empty, ScratchMode::Reuse] {
                            pool.scratch_mode = mode;
                            for _ in 0..3 {
                                pool.poison_scratch();
                                let first = pool.expert_batch_sel(&inputs[0],&sel,&w,dev);
                                pool.poison_scratch();
                                let second = pool.expert_batch_sel(&inputs[1],&sel,&w,dev);
                                // Compare first AFTER the next call to catch output/scratch aliasing.
                                assert!(first.equal(&gold[0]) && second.equal(&gold[1]),
                                    "scratch mismatch {name} L{layer} k{k} precision{precision} {mode:?}");
                                checks += 2;
                            }
                        }
                    }
                }
                pool.projection_precision = 0;
                pool.scratch_mode = ScratchMode::Zeros;
                let gold = inputs.each_ref().map(|x| pool.expert_batch_sel(x,&selected,&weights,dev));
                if spec.world == 2 {
                    let mut rounds = Vec::new();
                    let mut x = inputs[0].copy();
                    for graph in [false,true] {
                        for mode in ORDER {
                            pool.scratch_mode = mode;
                            x.copy_(&inputs[0]);
                            for _ in 0..8 { let _ = pool.expert_batch_sel(&x,&selected,&weights,dev); }
                            tch::Cuda::synchronize(0);
                            let captured = if graph {
                                tp::graph::begin().unwrap();
                                let y = pool.expert_batch_sel(&x,&selected,&weights,dev);
                                tp::graph::end().unwrap();
                                // Change the input and poison persistent buffers between replays.
                                for j in [1,0,1,0] {
                                    x.copy_(&inputs[j]); pool.poison_scratch();
                                    tp::graph::replay().unwrap();
                                    assert!(y.equal(&gold[j]), "graph stale scratch/input {mode:?}");
                                    checks += 1;
                                }
                                Some(y)
                            } else { None };
                            tch::Cuda::synchronize(0);
                            let start = Instant::now();
                            let mut last = Tensor::new();
                            for _ in 0..128 {
                                if graph { tp::graph::replay().unwrap(); }
                                else { last = pool.expert_batch_sel(&x,&selected,&weights,dev); }
                            }
                            tch::Cuda::synchronize(0);
                            let us = start.elapsed().as_secs_f64() * 1e6 / 128.;
                            let y = captured.as_ref().unwrap_or(&last);
                            assert!(y.equal(&gold[0])); checks += 1;
                            rounds.push(json!({"mode":mode.label(),"graph":graph,"us_per_call":us}));
                            if graph { tp::graph::destroy(); }
                        }
                    }
                    results.push(json!({"case":name,"step":step,"layer":layer,"rank_slice":spec.rank,"rounds":rounds}));
                }
                assert_eq!(misses,pool.misses);
                eprintln!("[scratch-local] {name} step={step} L{layer} TP{}/{} checks={checks}",spec.rank,spec.world);
                save(out,"local.json",&json!({"exact_output_checks":checks,"cold_loads_during_calls":0,
                    "timing":"warm fixed real activation/routing, 128 calls, wall clock + final CUDA sync; no NCCL",
                    "samples":results}));
            }
        }
    }
}

pub fn full(dir: &Path, out: &Path, n: usize) {
    use crate::forward::{DecodeStates, Engine, LayerState};
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let spec = tp::init_from_env();
    assert_eq!(spec.world,2);
    assert!(n>0);
    for key in ["GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"] {
        assert!(std::env::var(key).is_err(),"incompatible with {key}");
    }
    let dev = Device::Cuda(0);
    std::fs::create_dir_all(out).unwrap();
    let refs: Value = serde_json::from_str(&std::fs::read_to_string(
        "bench/m0-refs.json").unwrap()).unwrap();
    let cfg = crate::config::load(&dir.join("config.json")).unwrap();
    let w = crate::weights::ModelWeights::load(dir,&cfg,cfg.num_hidden_layers,dev);
    let mut fast = MoeFast::new(dir,cfg.num_hidden_layers,cfg.n_routed_experts,
        cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);
    fast.assume_hot = true;
    let misses = fast.misses;
    let mut eng = Engine { w, fast:Some(fast), native:None, pool:crate::moe::ExpertPool::new(dir,4) };
    let mut results = Vec::new();
    let confirm = std::env::var("GLM53_SCRATCH_CONFIRM").as_deref() == Ok("1");
    // Confirmation reverses the initial arm and gives each candidate four rounds.
    let order: &[ScratchMode] = if confirm {
        &[ScratchMode::Empty,ScratchMode::Zeros,ScratchMode::Zeros,ScratchMode::Empty,
          ScratchMode::Zeros,ScratchMode::Empty,ScratchMode::Empty,ScratchMode::Zeros]
    } else { &ORDER };
    let names: &[&str] = if confirm { &["count"] } else { &["hello","count","hashmap"] };
    for &name in names {
        let ids: Vec<i64> = refs[name]["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
        let tokens: Vec<i64> = refs[name]["text_ids"].as_array().unwrap().iter().take(n).map(|v|v.as_i64().unwrap()).collect();
        assert_eq!(tokens.len(),n);
        assert!(ids.len()+n<=512);
        eng.fast.as_mut().unwrap().scratch_mode = ScratchMode::Zeros;
        let input_ids = Tensor::from_slice(&ids).to_device(dev);
        let (lg, initial) = eng.prefill(&input_ids);
        let mut prefill = Vec::new();
        // Warm prefill and measure both directions; validate all logits and states.
        for &mode in order {
            eng.fast.as_mut().unwrap().scratch_mode = mode;
            // Exercise the allocator with the reference state still resident.
            // The first timing must not pay for a new peak allocation footprint.
            let _ = eng.prefill(&input_ids);
            rendezvous(dev);
            let start = Instant::now();
            let (candidate,st) = eng.prefill(&input_ids);
            tch::Cuda::synchronize(0);
            let ms = start.elapsed().as_secs_f64()*1000.;
            assert!(candidate.equal(&lg),"prefill logits differ {name} {mode:?}");
            assert_eq!(crate::forward::states_max_diff(&initial,&st),0.,"prefill state differs");
            prefill.push(json!({"mode":mode.label(),"ms":ms}));
        }
        let mut state = DecodeStates(initial.0.into_iter().map(|s| match s {
            LayerState::Mla(s) => LayerState::MlaG(crate::mla::MlaStateG::from_state(&s,512)), other=>other,
        }).collect());
        let initial = crate::forward::snapshot(&state);
        let inputs: Vec<Tensor> = tokens.iter().map(|&tok|Tensor::from_slice(&[tok]).to_device(dev)).collect();
        let mut input = inputs[0].copy();
        eng.fast.as_mut().unwrap().scratch_mode = ScratchMode::Zeros;
        let mut gold = Vec::new();
        for tok in &inputs { input.copy_(tok); gold.push(eng.step_buf(&input,&mut state)); }
        let final_state = crate::forward::snapshot(&state);
        let mut rounds = Vec::new();
        for graph in [false,true] {
            for &mode in order {
                eng.fast.as_mut().unwrap().scratch_mode = mode;
                crate::benchmark::restore(&mut state,&initial);
                input.copy_(&inputs[0]);
                let _ = eng.step_buf(&input,&mut state);
                tch::Cuda::synchronize(0);
                let captured = if graph {
                    tp::graph::begin().unwrap();
                    let y = eng.step_buf(&input,&mut state);
                    tp::graph::end().unwrap();
                    Some(y)
                } else { None };
                if confirm {
                    for _ in 0..2 {
                        crate::benchmark::restore(&mut state,&initial);
                        for tok in &inputs {
                            input.copy_(tok);
                            if graph { tp::graph::replay().unwrap(); }
                            else { let _ = eng.step_buf(&input,&mut state); }
                        }
                    }
                }
                // Untimed all-position comparison also warms the complete trace.
                crate::benchmark::restore(&mut state,&initial);
                let mut trace = Vec::new();
                for tok in &inputs {
                    input.copy_(tok);
                    let y = if graph { tp::graph::replay().unwrap(); captured.as_ref().unwrap().copy() }
                        else { eng.step_buf(&input,&mut state) };
                    trace.push(y);
                }
                tch::Cuda::synchronize(0);
                assert!(trace.iter().zip(&gold).all(|(a,b)|a.equal(b)),"trace differs {name} {mode:?} graph={graph}");
                assert_eq!(crate::forward::states_max_diff(&state,&final_state),0.,"trace state differs");
                crate::benchmark::restore(&mut state,&initial);
                rendezvous(dev);
                let start = Instant::now();
                let mut last = Tensor::new();
                for tok in &inputs {
                    input.copy_(tok);
                    if graph { tp::graph::replay().unwrap(); }
                    else { last = eng.step_buf(&input,&mut state); }
                }
                tch::Cuda::synchronize(0);
                let ms = start.elapsed().as_secs_f64()*1000./n as f64;
                assert!(captured.as_ref().unwrap_or(&last).equal(gold.last().unwrap()),"timed logits differ");
                assert_eq!(crate::forward::states_max_diff(&state,&final_state),0.,"timed state differs");
                rounds.push(json!({"mode":mode.label(),"graph":graph,"ms_per_step":ms,
                    "all_step_logits_exact":true,"final_state_exact":true}));
                if graph { tp::graph::destroy(); }
                eprintln!("[scratch-full] rank={} {name} {mode:?} graph={graph} {ms:.3} ms/step",spec.rank);
            }
        }
        assert_eq!(misses,eng.fast.as_ref().unwrap().misses);
        results.push(json!({"case":name,"prompt_tokens":ids.len(),"tokens":tokens,"prefill":prefill,"rounds":rounds}));
        save(out,&format!("full-rank{}.json",spec.rank),&json!({"rank":spec.rank,"dense_tp":tp::dense_enabled(),
            "confirmation":confirm,
            "w_fp16":crate::weights::w_fp16(),"cold_loads_after_preload":0,"window":512,
            "timing":"resident model; device input copy + forward/NCCL + final sync; excludes load/capture/diagnostics; teacher forced",
            "cases":results}));
    }
}

fn rendezvous(dev: Device) {
    tch::Cuda::synchronize(0);
    tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));
    tch::Cuda::synchronize(0);
}
