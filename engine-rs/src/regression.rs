//! CPU regressions: no checkpoint, GPU, network or Python required.
use tch::{Device, Kind, Tensor};
use crate::weights::*;

fn tensor(shape: &[i64], phase: f64) -> Tensor {
    // Deterministic independent elements, without process-global RNG state or sinusoidal cancellation.
    let mut seed = (phase as u64).wrapping_add(1);
    let data: Vec<f32> = (0..shape.iter().product::<i64>()).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((seed >> 40) as f32 / 16777216.0) - 0.5) * 0.06
    }).collect();
    Tensor::from_slice(&data).view(shape)
}
fn close(a: &Tensor, b: &Tensor, tol: f64) {
    let error = f64::try_from((a - b).abs().max()).unwrap();
    let relative = f64::try_from((a-b).norm() / b.norm().clamp_min(1e-12)).unwrap();
    assert!(error < tol && relative < 3e-3, "max error {error} / relative {relative}");
}
fn layer() -> LayerWeights {
    let z = || Tensor::zeros([1], (Kind::Float, Device::Cpu));
    LayerWeights {
        plan: LayerPlanOwned { attn: "test".into(), mlp: "test".into() },
        hc: HcParams { attn_fn: z(), attn_scale: z(), attn_base: z(), ffn_fn: z(),
            ffn_scale: z(), ffn_base: z(), in_ln: z(), post_ln: z() },
        kda: None, mla: None, dense: None, moe: None,
    }
}
fn kda_layer() -> LayerWeights {
    let mut l = layer();
    l.kda = Some(KdaWeights {
        constants:None,
        wq: tensor(&[512, 8], 1.), wk: tensor(&[512, 8], 2.), wv: tensor(&[512, 8], 3.),
        wo: tensor(&[8, 512], 4.), wb: tensor(&[4, 8], 5.),
        fa: tensor(&[3, 8], 6.), fb: tensor(&[512, 3], 7.),
        ga: tensor(&[3, 8], 8.), gb: tensor(&[512, 3], 9.),
        conv_q: tensor(&[512, 4], 10.), conv_k: tensor(&[512, 4], 11.),
        conv_v: tensor(&[512, 4], 12.), dt_bias: tensor(&[4, 128], 13.),
        a_log: tensor(&[4], 14.), o_norm: Tensor::ones([128], (Kind::Float, Device::Cpu)),
    }); l
}
fn mla_layer() -> LayerWeights {
    let mut l = layer();
    l.mla = Some(MlaWeights {
        latent_cache:std::cell::RefCell::new(None),
        indexer: None,
        q_a: tensor(&[3, 8], 1.), q_b: tensor(&[1024, 3], 2.),
        kv_a: tensor(&[3, 8], 3.), kv_b: tensor(&[2048, 3], 4.),
        wo: tensor(&[8, 1024], 5.), q_a_ln: Tensor::ones([3], (Kind::Float, Device::Cpu)),
        kv_a_ln: Tensor::ones([3], (Kind::Float, Device::Cpu)),
    }); l
}

#[test]
fn batched_kda_prefill_preserves_chunk_boundaries() {
    tch::set_num_threads(1);
    let l=kda_layer();let w=l.kda.as_ref().unwrap();let x=tensor(&[11,8],43.)*10.;
    let fresh=||crate::kda::KdaState::with_heads(Device::Cpu,4);
    std::env::remove_var("GLM53_PREFILL_BATCH");
    let (expected,gold)=crate::kda::kda_forward_state(w,&x,fresh());
    std::env::set_var("GLM53_PREFILL_BATCH","1");
    let (whole,state)=crate::kda::kda_forward_state(w,&x,fresh());
    let mut st=fresh();let mut outs=Vec::new();let mut pos=0;
    for n in [1,3,7] {let (o,s)=crate::kda::kda_forward_state(w,&x.narrow(0,pos,n),st);st=s;outs.push(o);pos+=n;}
    std::env::remove_var("GLM53_PREFILL_BATCH");
    close(&whole,&expected,1e-5);close(&Tensor::cat(&outs,0),&expected,1e-5);
    close(&state.h,&gold.h,1e-5);close(&st.h,&gold.h,1e-5);
    close(&st.conv,&gold.conv,1e-5);
}

#[test]
fn batched_latent_prefill_preserves_partial_pools() {
    tch::set_num_threads(1);let w=latent_weights();
    let x=tensor(&[11,8],43.)*10.;let mut gold=crate::mla_latent::State::new(&w,16);
    std::env::remove_var("GLM53_PREFILL_BATCH");
    let expected=crate::mla_latent::chunk(&w,&x,&mut gold);
    std::env::set_var("GLM53_PREFILL_BATCH","1");
    let mut state=crate::mla_latent::State::new(&w,16);let mut outs=Vec::new();let mut pos=0;
    for n in [1,3,7] {outs.push(crate::mla_latent::chunk(&w,&x.narrow(0,pos,n),&mut state));pos+=n;}
    std::env::remove_var("GLM53_PREFILL_BATCH");
    close(&Tensor::cat(&outs,0),&expected,2e-4);
    assert!(state.max_diff(&gold)<0.002,"chunk state differs {}",state.max_diff(&gold));
    assert_eq!(state.len.int64_value(&[0]),11);
}

fn latent_weights()->MlaWeights {
    let dev=Device::Cpu;
    MlaWeights{latent_cache:std::cell::RefCell::new(None),q_a:tensor(&[3,8],1.),q_b:tensor(&[256,3],2.),
        kv_a:tensor(&[512,8],3.),kv_b:tensor(&[512,512],4.),wo:tensor(&[8,256],5.),
        q_a_ln:Tensor::ones([3],(Kind::Float,dev)),kv_a_ln:Tensor::ones([512],(Kind::Float,dev)),
        indexer:Some(crate::dsa::Weights{q:tensor(&[8,3],6.),k:tensor(&[4,8],7.),
            norm_w:Tensor::ones([4],(Kind::Float,dev)),norm_b:Tensor::zeros([4],(Kind::Float,dev)),
            score:tensor(&[2,8],8.),ape:tensor(&[4,4],9.),gate:tensor(&[4,8],10.)})}
}

#[test]
fn batched_tree_inherits_ancestors_not_adjacent_rows() {
    tch::set_num_threads(1);let dev=Device::Cpu;
    let parents=[None,Some(0),Some(0),Some(1),Some(2),Some(1),None];
    let x=tensor(&[7,8],92.)*10.;let prefix=tensor(&[3,8],91.)*10.;
    let l=kda_layer();let w=l.kda.as_ref().unwrap();
    let (_,base)=crate::kda::kda_forward_state(w,&prefix,crate::kda::KdaState::with_heads(dev,4));
    let (actual,states)=crate::kda::tree(w,&x,&base,&parents);
    let mut expected_states:Vec<crate::kda::KdaState>=Vec::new();
    for (i,&p) in parents.iter().enumerate() {
        let old=p.map_or(&base,|p|&expected_states[p]);let mut s=crate::kda::KdaState{h:old.h.copy(),conv:old.conv.copy()};
        let expected=crate::kda::kda_step(w,&x.narrow(0,i as i64,1),&mut s);
        close(&actual.narrow(0,i as i64,1),&expected,1e-5);
        close(&states[i].h,&s.h,1e-5);close(&states[i].conv,&s.conv,1e-5);expected_states.push(s);
    }
    let w=latent_weights();let mut base=crate::mla_latent::State::new(&w,16);
    let _=crate::mla_latent::chunk(&w,&prefix,&mut base);
    let original=base.snapshot();let (actual,states)=crate::mla_latent::tree(&w,&x,&base,&parents);
    let mut expected_states:Vec<crate::mla_latent::State>=Vec::new();
    for (i,&p) in parents.iter().enumerate() {
        let mut s=p.map_or(&base,|p|&expected_states[p]).snapshot();
        let expected=crate::mla_latent::step(&w,&x.narrow(0,i as i64,1),&mut s);
        close(&actual.narrow(0,i as i64,1),&expected,2e-4);
        assert!(states[i].max_diff(&s)<0.002);expected_states.push(s);
    }
    assert_eq!(base.max_diff(&original),0.,"tree forward mutated base cache");
}

#[test]
fn incremental_dsa_pool_boundaries_and_restore() {
    use crate::dsa::{Weights,State};
    let dev=Device::Cpu;
    let w=Weights{q:tensor(&[8,3],31.),k:tensor(&[4,8],32.),
        norm_w:Tensor::ones([4],(Kind::Float,dev)),norm_b:Tensor::zeros([4],(Kind::Float,dev)),
        score:tensor(&[2,8],33.),ape:tensor(&[4,4],34.),gate:tensor(&[4,8],35.)};
    let x=tensor(&[2104,8],36.);let qr=tensor(&[2104,3],37.);
    let mut state=State::new(2112,4,dev);
    let mut saved=None;
    for t in 0..2104 {
        let selected=state.append_select(&w,&x.get(t).unsqueeze(0),&qr.get(t).unsqueeze(0),&Tensor::from_slice(&[t]));
        if t==2046 {saved=Some(state.snapshot());}
        if ![0,2,3,4,510,511,512,2046,2047,2048,2050,2051,2103].contains(&t){continue;}
        let actual:std::collections::BTreeSet<i64>=Vec::<i64>::try_from(selected).unwrap().into_iter().filter(|&i|i>=0).collect();
        assert!(actual.iter().all(|&i|i<=t));
        if t<2048 {assert_eq!(actual,(0..=t).collect());continue;}
        // Independent whole-prefix construction: no incremental tail or pool storage.
        let prefix=x.narrow(0,0,t+1);let raw=prefix.matmul(&w.k.transpose(0,1));
        let c=&raw-raw.mean_dim(&[-1i64][..],true,Kind::Float);
        let variance=(&c*&c).mean_dim(&[-1i64][..],true,Kind::Float);
        let keys=c*(variance+1e-6).rsqrt()*&w.norm_w+&w.norm_b;
        let gates=prefix.matmul(&w.gate.transpose(0,1));let p=(t+1)/4;
        let keys=keys.narrow(0,0,p*4).view([p,4,4]);
        let probs=(gates.narrow(0,0,p*4).view([p,4,4])+w.ape.unsqueeze(0)).softmax(1,Kind::Float);
        let pools=(keys*probs).sum_dim_intlist(&[1i64][..],false,Kind::Float);
        close(&state.pools.narrow(0,0,p),&pools,1e-5);
        let query=qr.get(t).unsqueeze(0).matmul(&w.q.transpose(0,1)).view([2,4]);
        let weights=x.get(t).unsqueeze(0).matmul(&w.score.transpose(0,1)).view([2,1])*2f64.powf(-0.5);
        let scores=((query.matmul(&pools.transpose(0,1))*0.5).relu()*weights).sum_dim_intlist(&[0i64][..],false,Kind::Float);
        let best:Vec<i64>=scores.topk(512.min(p),0,true,true).1.try_into().unwrap();
        let mut expected:std::collections::BTreeSet<i64>=best.into_iter().flat_map(|p|p*4..p*4+4).collect();
        expected.extend(p*4..=t);
        if actual!=expected {
            let a:Vec<_>=actual.difference(&expected).copied().collect();
            let b:Vec<_>=expected.difference(&actual).copied().collect();
            let values:Vec<_>=a.iter().chain(&b).map(|i|(*i,scores.double_value(&[i/4]))).collect();
            // torch.topk does not define stable indices for exact ties. Padding
            // the graph-capacity input changes tie choice, not top-k validity.
            // Permit ONLY an exact cutoff tie; never a missing higher score.
            let cutoff=scores.topk(512.min(p),0,true,true).0.min().double_value(&[]);
            assert_eq!(actual.len(),expected.len());
            assert!(values.iter().all(|(_,v)|*v==cutoff),
                "DSA ranking differs at {t}: extra={a:?} missing={b:?} scores={values:?} cutoff={cutoff}");
        }
    }
    let gold=state.snapshot();state.restore(&saved.unwrap());
    for t in 2047..2104 {let _=state.append_select(&w,&x.get(t).unsqueeze(0),&qr.get(t).unsqueeze(0),&Tensor::from_slice(&[t]));}
    assert!(state.pools.equal(&gold.pools)&&state.tail_k.equal(&gold.tail_k)&&state.tail_gate.equal(&gold.tail_gate));
}

fn small_latent_state()->crate::mla_latent::State {
    let dev=Device::Cpu;
    crate::mla_latent::State{latent:Tensor::zeros([2,512],(Kind::Half,dev)),
        index:crate::dsa::State::new(2,4,dev),len:Tensor::from_slice(&[2i64]),capacity:2,pool_row:None,
        wk:Tensor::ones([1],(Kind::Float,dev)),wv:Tensor::ones([1],(Kind::Float,dev))}
}

#[test]
#[should_panic(expected="latent capacity exceeded")]
fn latent_capacity_fails_before_write() {small_latent_state().ensure_room(1);}

#[test]
fn latent_snapshot_is_independent_and_restore_keeps_addresses() {
    let mut a=small_latent_state();let b=a.snapshot();
    let pointers=[a.latent.data_ptr(),a.index.pools.data_ptr(),a.len.data_ptr()];
    let _=a.latent.fill_(3.);let _=a.index.pools.fill_(4.);let _=a.len.fill_(0);
    assert!(a.max_diff(&b)>0.);assert_eq!(b.latent.sum(Kind::Float).double_value(&[]),0.);
    a.restore(&b);assert_eq!(a.max_diff(&b),0.);
    assert_eq!(pointers,[a.latent.data_ptr(),a.index.pools.data_ptr(),a.len.data_ptr()]);
    assert_eq!(a.wk.data_ptr(),b.wk.data_ptr(),"immutable projections should be shared");
}

#[test]
fn dense_tp_kda_prefill_and_recurrent_states() {
    tch::set_num_threads(1);
    let x = tensor(&[5, 8], 50.) * 10.;
    let full = kda_layer(); let w = full.kda.as_ref().unwrap();
    let expected = crate::kda::kda_forward(w, &x);
    let mut sum = Tensor::zeros_like(&expected);
    let (_, full_state) = crate::kda::kda_forward_state(w, &x, crate::kda::KdaState::with_heads(Device::Cpu, 4));
    for rank in 0..2 {
        let mut l = kda_layer(); shard_dense_layer(&mut l, rank, 2);
        let w = l.kda.as_ref().unwrap();
        let (actual, state) = crate::kda::kda_forward_state(w, &x, crate::kda::KdaState::with_heads(Device::Cpu, 2));
        close(&actual, &crate::kda::kda_forward(w, &x), 1e-6);
        close(&state.h, &full_state.h.narrow(0, rank as i64 * 2, 2), 1e-6);
        assert_eq!(state.conv.size(), [3, 768]);
        for segment in 0..3 {
            close(&state.conv.narrow(1, segment*256, 256),
                &full_state.conv.narrow(1, segment*512 + rank as i64*256, 256), 1e-6);
        }
        sum += actual;
    }
    close(&sum, &expected, 1e-6);
}

#[test]
fn dense_tp_mla_interleaved_heads_and_graph_cache() {
    tch::set_num_threads(1);
    let x = tensor(&[5, 8], 30.);
    let full = mla_layer(); let w = full.mla.as_ref().unwrap();
    let expected = crate::mla::mla_forward(w, &x);
    let mut sum = Tensor::zeros_like(&expected);
    for rank in 0..2 {
        let mut l = mla_layer(); shard_dense_layer(&mut l, rank, 2);
        let w = l.mla.as_ref().unwrap();
        let (actual, mut state) = crate::mla::mla_forward_state(w, &x, crate::mla::MlaState::with_heads(Device::Cpu, 2));
        close(&actual, &crate::mla::mla_forward(w, &x), 1e-6);
        let mut gs = crate::mla::MlaStateG::from_state(&state, 8);
        assert_eq!(gs.k.size(), [8, 2, 256]);
        let y = crate::mla::mla_step(w, &x.get(0).unsqueeze(0), &mut state);
        let yg = crate::mla::mla_step_g(w, &x.get(0).unsqueeze(0), &mut gs);
        close(&y, &yg, 1e-4);
        sum += actual;
    }
    close(&sum, &expected, 1e-6);
}

#[test]
fn dense_tp_mlp_and_shared_sum() {
    let x = tensor(&[3, 8], 30.);
    let make = || {
        let mut l = layer();
        l.dense = Some(DenseMlp { wg: tensor(&[12,8],1.), wu: tensor(&[12,8],2.), wd: tensor(&[8,12],3.) });
        l.moe = Some(MoeMeta { w_gate: tensor(&[16,8],4.), bias: tensor(&[16],5.),
            sh_wg: tensor(&[12,8],1.), sh_wu: tensor(&[12,8],2.), sh_wd: tensor(&[8,12],3.) }); l
    };
    let run = |l: &LayerWeights| { let w = l.dense.as_ref().unwrap(); mm16(&(mm16(&x,&w.wg).silu()*mm16(&x,&w.wu)), &w.wd) };
    let expected = run(&make()); let mut sum = Tensor::zeros_like(&expected);
    let mut shared = Tensor::zeros_like(&expected);
    for rank in 0..2 { let mut l = make(); shard_dense_layer(&mut l,rank,2);
        sum += run(&l); shared += crate::moe::shared_forward(l.moe.as_ref().unwrap(), &x); }
    close(&sum, &expected, 1e-7);
    close(&shared, &crate::moe::shared_forward(make().moe.as_ref().unwrap(), &x), 1e-7);
}

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("m1-regression-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut hdr = serde_json::Map::new(); let mut data = Vec::<u8>::new();
        // Include a later layer to test expected_total's layer bound.
        for layer in [3, 45] { for e in 0..23 { for proj in ["gate_proj", "up_proj", "down_proj"] {
            for (suffix, dtype, shape, bytes) in [
                ("trellis", "I16", vec![1,1,64], vec![0u8;128]),
                ("suh", "F32", vec![16], vec![0u8;64]),
                ("svh", "F32", vec![16], vec![0u8;64]),
            ] {
                let start = data.len(); data.extend(bytes);
                hdr.insert(format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.{suffix}"),
                    serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}));
            }
        } } }
        let header = serde_json::to_vec(&hdr).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec(); bytes.extend(header); bytes.extend(data);
        std::fs::write(dir.join("model-test.safetensors"), bytes).unwrap(); Self(dir)
    }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

#[test]
fn full_pool_keeps_selected_hits_and_deduplicates_loads() {
    let f = Fixture::new("pool");
    let mut p = crate::moefast::MoeFast::new(&f.0, 45, 23, 16, Device::Cpu);
    p.ensure_many(3, &(0..16).collect::<Vec<_>>(), Device::Cpu);
    p.ensure_many(3, &[0,16,17,18,19,20,21,22,22], Device::Cpu);
    for e in [0,16,17,18,19,20,21,22] { assert!(p.slot_of(3,e) >= 0); }
    assert_eq!(p.misses,23); assert_eq!(p.expected_total(),23);
    assert!(!p.is_fully_loaded());
    let misses = p.misses;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.preload_all(45,23,Device::Cpu))).is_err());
    assert_eq!(p.misses,misses, "容量校验必须发生在装载前");
    let mut all = crate::moefast::MoeFast::new(&f.0,45,23,23,Device::Cpu);
    all.preload_all(45,23,Device::Cpu);
    assert!(all.is_fully_loaded());
    assert_eq!(all.resident_count(),23);
}

#[test]
fn graph_short_output_matches_requested_budget() {
    let f = Fixture::new("budget");
    let mut eng = crate::forward::Engine {
        w: ModelWeights { embed: tensor(&[16,8],1.), lm_head: tensor(&[16,8],2.),
            final_norm: Tensor::ones([8],(Kind::Float,Device::Cpu)), layers: vec![], device: Device::Cpu, vocab_shard: None },
        pool: crate::moe::ExpertPool::new(&f.0,4), native: None, fast: None,
    };
    for n in 0..=3 {
        let (out, logs) = eng.greedy_incremental_graph_dbg(&[1,2],n,true);
        assert_eq!(out.len(),n); assert_eq!(logs.len(),n);
        assert_eq!(out,eng.greedy_incremental_dbg(&[1,2],n,false).0);
    }
}

#[test]
fn state_comparison_rejects_missing_layers_and_nan() {
    use crate::forward::{DecodeStates,LayerState,states_max_diff};
    let state = || DecodeStates(vec![LayerState::Kda(crate::kda::KdaState::with_heads(Device::Cpu,2))]);
    let a = state(); let mut b = state();
    assert_eq!(states_max_diff(&a,&b),0.);
    assert!(states_max_diff(&a,&DecodeStates(vec![])).is_infinite());
    if let LayerState::Kda(k) = &mut b.0[0] { let _ = k.h.fill_(f64::NAN); }
    assert!(states_max_diff(&a,&b).is_infinite());
}

#[test]
fn row_parallel_rounds_once_after_sum() {
    // Opposite-sign partials: rounding each rank to fp16 loses 0.5 in the final sum.
    let x = Tensor::ones([1,4],(Kind::Float,Device::Cpu));
    let w = Tensor::from_slice(&[2048f32,0.75,-2048.,0.75]).view([1,4]).to_kind(Kind::Half);
    let expected = mm16(&x,&w);
    let old = mm16(&x.narrow(1,0,2),&w.narrow(1,0,2)) + mm16(&x.narrow(1,2,2),&w.narrow(1,2,2));
    assert!(!old.equal(&expected));
    let sum = mm16_partial(&x.narrow(1,0,2),&w.narrow(1,0,2))
        + mm16_partial(&x.narrow(1,2,2),&w.narrow(1,2,2));
    assert!(sum.to_kind(Kind::Half).to_kind(Kind::Float).equal(&expected));
}
