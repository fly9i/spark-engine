// SPDX-License-Identifier: MIT
//! spark-engine CLI:census / exl3-hash / bench-decode / cuda-smoke /
//! parity / kda-probe / shim-smoke / greedy。
//! (main.rs 于 2026-09-21 从补丁堆积中整体重写;核心模块未动。)

mod host_memory;
mod qwen;
mod pcache;
mod expert_load;
mod kv_pool;
mod acceptance;
mod sampling;
mod benchmark;
mod evaluation;
mod root_probe;
mod deep_probe;
mod scratch_bench;
mod kernel_bench;
mod dsa;
mod dsa_index;
mod dsa_topk;
mod dsa_position;
mod feature_prefix;
mod dsa_direct;
mod dsa_quality;
mod mla_latent;
mod latent_probe;
mod session;
mod session_probe;
mod dflash;
mod draft_selector;
mod draft_conv;
mod shared_gu;
mod draft_final_norm;
mod spec_session;
mod route_trace;
mod spec_policy;
mod speculative;
mod spec_probe;
mod tree_probe;
mod qualification;
mod profile;
mod perf_probe;
mod prefill_gate;
mod dense_lt;
mod dense_fp8;
mod c12;
mod fp8_probe;
mod gemv;
mod vocab_probe;
mod head_select;
mod config;
mod exl3;
mod forward;
mod kda;
mod kda_correction;
mod verifier_state;
mod verifier_probe;
mod shared_input_probe;
mod resident_plan;
mod mhc;
mod mla;
mod moe;
mod moefast;
mod moe_own;
mod ablate;
mod nibmap;
mod safetensors;
mod tp;
mod weights;
mod vision;
#[cfg(test)]
mod regression;

use std::path::Path;

fn main() {
    tp::set_fp32_accum(); // fp16 GEMM 一律 fp32 累加(精度门)
    let args: Vec<String> = std::env::args().collect();
    let mut exit_code = 0;
    match args.get(1).map(String::as_str) {
        Some("census") => census(Path::new(&args[2])),
        Some("qwen-serve") => qwen::serve::serve(Path::new(&args[2]), Path::new(&args[3])),
        Some("qwen-vision-probe") => qwen::vision::probe(Path::new(&args[2]), &args[3], args[4].parse().unwrap(), args[5].parse().unwrap(),
                                                         args[6].parse().unwrap(), &args[7]),
        Some("qwen-spec") => qwen::probe::spec(Path::new(&args[2]), Path::new(&args[3]),
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(128), args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3)),
        Some("glm-refusal-dir") => ablate::glm_refusal_dir(Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4]), Path::new(&args[5])),
        Some("qwen-pcache-selftest") => qwen::probe::pcache_selftest(Path::new(&args[2])),
        Some("qwen-refusal-dir") => qwen::probe::refusal_dir(Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4]), Path::new(&args[5])),
        Some("qwen-batch") => qwen::probe::batch(Path::new(&args[2]), &args[3],
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(60), args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3)),
        Some("qwen-gen") => qwen::probe::gen(Path::new(&args[2]), Path::new(&args[3]),
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32), args.get(5).map(Path::new)),
        Some("exl3-hash") => exl3_hash(Path::new(&args[2]), &args[3], true),
        Some("exl3-hash-serial") => exl3_hash(Path::new(&args[2]), &args[3], false),
        Some("bench-decode") => bench_decode(Path::new(&args[2])),
        Some("cuda-smoke") => cuda_smoke(),
        Some("parity") => parity(
            Path::new(&args[2]),
            &args[3],
            args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(3),
        ),
        Some("kda-probe") => kda_probe(Path::new(&args[2]), &args[3]),
        Some("shim-smoke") => shim_smoke(Path::new(&args[2])),
        Some("fast-smoke") => fast_smoke(Path::new(&args[2])),
        Some("fast-probe") => fast_probe(Path::new(&args[2])),
        Some("dense-math-smoke") => benchmark::dense_math_smoke(),
        Some("dense-prefix-probe") => benchmark::dense_prefix_probe(Path::new(&args[2]), &args[3]),
        Some("deep-recompare") => deep_probe::recompare(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("deep-input-replay") => deep_probe::input_replay_run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5])),
        Some("deep-probe") => deep_probe::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("scratch-local") => scratch_bench::local(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("mhc-probe") => kernel_bench::mhc_local(Path::new(&args[2])),
        Some("kda-fusion-probe") => kernel_bench::kda_local(Path::new(&args[2])),
        Some("kda-fusion-bench") => kernel_bench::kda_full(Path::new(&args[2]),Path::new(&args[3])),
        Some("dsa-visible-probe") => dsa::all_visible_probe(Path::new(&args[2])),
        Some("feature-prefix-probe") => feature_prefix::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("draft-final-norm-probe") => dflash::final_norm_select_probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("draft-selector-probe") => draft_selector::probe(Path::new(&args[2])),
        Some("shared-gu-probe") => shared_gu::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("draft-conv-probe") => draft_conv::probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("dsa-topk-probe") => dsa_topk::probe(Path::new(&args[2])),
        Some("dsa-position-probe")=>dsa_topk::position_probe(Path::new(&args[2])),
        Some("dsa-index-probe") => dsa_index::probe(Path::new(&args[2])),
        Some("dsa-direct-probe") => dsa_direct::probe(Path::new(&args[2])),
        Some("kda-correction-probe") => kda_correction::run_local_probe(Path::new(&args[2])),
        Some("head-select-probe") => head_select::probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("latent-probe") => latent_probe::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("latent-bench") => kernel_bench::latent_full(Path::new(&args[2]),Path::new(&args[3])),
        Some("session-probe") => session_probe::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("glm-moe-ab") => moe_own::ab(Path::new(&args[2])),
        Some("mhc-post-probe") => mhc::post_probe(Path::new(&args[2])),
        Some("kda-conv-chain-probe") => kda::conv_chain_probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("kda-fork-probe") => kda::fork_probe(Path::new(&args[2])),
        Some("owned-graph-probe") => tp::graph::owned_probe(),
        Some("graph-workspace-probe") => tp::graph::workspace_probe(),
        Some("prefill-gate") => prefill_gate::run(Path::new(&args[2])),
        Some("verify-invariance") => spec_probe::verify_invariance(Path::new(&args[2])),
        Some("row-timing") => spec_probe::row_timing(Path::new(&args[2])),
        Some("mhc-rows-probe") => prefill_gate::mhc_rows_probe(),
        Some("rdma-big-probe") => prefill_gate::rdma_big_probe(),
        Some("pinned-bw-probe") => {extern "C"{fn glm53_pinned_bw_probe()->i32;} let _g=tch::Tensor::zeros([1],(tch::Kind::Float,tch::Device::Cuda(0))); assert_eq!(unsafe{glm53_pinned_bw_probe()},0);}
        Some("kda-seq-probe") => kda::seq_probe(),
        Some("pdl-probe") => prefill_gate::pdl_probe(),
        Some("l2pf-probe") => prefill_gate::l2pf_probe(),
        Some("fp8-big-probe") => prefill_gate::fp8_big_probe(),
        Some("index-bf16-probe") => prefill_gate::index_bf16_probe(Path::new(&args[2])),
        Some("draft-norm-cache-probe") => dflash::norm_cache_probe(Path::new(&args[2])),
        Some("dflash-dataflow-probe") => dflash::dataflow_probe(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("dflash-opt-probe") => dflash::opt_probe(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("dflash-probe") => dflash::probe(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        // GLM53_DRAFT_DIR overrides the drafter directory.
        Some("vision-probe") => {let a:Vec<String>=args.to_vec();vision::probe(&a)},
        Some("serve") => {let d=std::env::var("GLM53_DRAFT_DIR").unwrap_or_else(|_|args[3].clone());
            if d!=args[3] {eprintln!("[serve] drafter {d} (GLM53_DRAFT_DIR)");}
            spec_probe::serve(Path::new(&args[2]),Path::new(&d),Path::new(&args[4]))},
        Some("spec-decode-many") => spec_probe::decode_many(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),&args[5],Path::new(&args[6])),
        Some("shared-input-probe") => shared_input_probe::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("spec-probe") => spec_probe::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("tree-probe") => tree_probe::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("m2-checks") => qualification::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5])),
        Some("perf-sweep") => perf_probe::sweep(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("perf-fp8") => perf_probe::fp8(Path::new(&args[2]),Path::new(&args[3])),
        Some("half-bmm-probe") => {tp::init_from_env();mla_latent::half_bmm_probe(Path::new(&args[2]));},
        Some("kda-gate-probe") => {tp::init_from_env();kda::gate_probe(Path::new(&args[2]),Path::new(&args[3]));},
        Some("dsa-topk-fast-probe") => {dsa_topk::fast_probe(Path::new(&args[2]));},
        Some("mhc-pre-probe") => {tp::init_from_env();mhc::pre_probe(Path::new(&args[2]),Path::new(&args[3]));},
        Some("fp8-extension-probe") => fp8_probe::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("perf-target") => kernel_bench::perf_target(Path::new(&args[2]),&args[3],Path::new(&args[4])),
        Some("perf-grouped") => perf_probe::grouped(Path::new(&args[2]),Path::new(&args[3])),
        Some("perf-dense") => perf_probe::dense(Path::new(&args[2]),Path::new(&args[3])),
        Some("perf-cross-stage") => perf_probe::cross_stage(Path::new(&args[2]),Path::new(&args[3])),
        Some("perf-swiglu") => perf_probe::swiglu(Path::new(&args[2])),
        Some("perf-dataflow") => perf_probe::dataflow(Path::new(&args[2])),
        Some("perf-cache") => perf_probe::cache(Path::new(&args[2]),Path::new(&args[3])),
        Some("kda-prefill-probe") => kda::prefill_probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("dsa-prefill-score-probe") => dsa::prefill_score_probe(),
        Some("sampling-probe") => sampling::probe(),
        Some("gumbel-fused-probe") => sampling::fused_probe(),
        Some("perf-kernels") => perf_probe::kernels(Path::new(&args[2]),Path::new(&args[3])),
        Some("prefill-perf") => perf_probe::prefill(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("profile-steady") => profile::steady(Path::new(&args[2])),
        Some("small-probe") => {tp::init_from_env();gemv::small_probe(Path::new(&args[2]),Path::new(&args[3]));},
        Some("gemv-probe") => gemv::probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("tp-pack-probe") => moe::tp_pack_probe(Path::new(&args[2]),Path::new(&args[3])),
        Some("tp-pack-full") => kernel_bench::tp_pack_full(Path::new(&args[2]),Path::new(&args[3])),
        Some("allreduce-latency") => tp::allreduce_latency_probe(Path::new(&args[2])),
        Some("ar-fused-probe") => mhc::ar_fused_probe(Path::new(&args[2])),
        Some("draft-head-probe") => dflash::head_probe(),
        Some("tp-network-probe") => tp::network_probe(Path::new(&args[2])),
        Some("vocab-probe") => vocab_probe::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("spec-decode") => {
            let ids:Vec<i64>=args[4].split(',').map(|x|x.parse().expect("comma-separated token IDs")).collect();
            spec_probe::decode(Path::new(&args[2]),Path::new(&args[3]),&ids,&args[5],args[6].parse().unwrap(),Path::new(&args[7]));
        },
        Some("mhc-bench") => kernel_bench::mhc_full(Path::new(&args[2]),Path::new(&args[3])),
        Some("scratch-full") => scratch_bench::full(Path::new(&args[2]),Path::new(&args[3]),args.get(4).and_then(|s|s.parse().ok()).unwrap_or(24)),
        Some("expert-deep-probe") => deep_probe::expert_precision_run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5])),
        Some("root-probe") => root_probe::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),&args[5],args.get(6).map(String::as_str).unwrap_or("native")),
        Some("eval-suite") => evaluation::suite(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("m2-qualify") => evaluation::combined(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("eval-replay") => evaluation::replay_probe(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5]),&args[6]),
        Some("compare-qualification") => benchmark::compare_qualification(Path::new(&args[2]),Path::new(&args[3])),
        Some("bench-paired") => benchmark::paired(Path::new(&args[2]), &args[3],
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(24)),
        Some("bench-step") => bench_step(
            Path::new(&args[2]),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20),
        ),
        Some("state-eq") => state_eq(Path::new(&args[2])),
        Some("graph-smoke") => graph_smoke(),
        Some("graph-churn-probe") => tp::graph::churn_probe(),
        Some("tp-ping") => tp_ping(),
        Some("fast-bench") => fast_bench(Path::new(&args[2])),
        Some("accept") => { if !accept(
            Path::new(&args[2]),
            &args[3],
            args.get(4).map(|s| s.parse().unwrap()).unwrap_or(24),
        ) { exit_code = 1; } },
        Some("greedy-inc") => greedy_inc(
            Path::new(&args[2]),
            &args[3],
            args.get(4).map(|s| s.parse::<usize>().unwrap()).unwrap_or(12),
        ),
        Some("greedy") => greedy(
            Path::new(&args[2]),
            &args[3],
            args.get(4).map(|s| s.parse::<usize>().unwrap()).unwrap_or(12),
        ),
        _ => eprintln!("用法: spark-engine <census|exl3-hash|bench-decode|cuda-smoke|parity|kda-probe|shim-smoke|greedy> ..."),
    }
    tp::shutdown();
    if exit_code != 0 { std::process::exit(exit_code); }
}

// ───────────────────────── census(旧版保留) ─────────────────────────

fn census(dir: &Path) {
    let idx = safetensors::ShardIndex::scan(dir).expect("scan");
    let mut total = 0u64;
    for (k, e) in idx.entries.iter() {
        total += e.nbytes as u64;
        let _ = k;
    }
    println!("张量 {} 个,共 {:.2} GiB", idx.entries.len(), total as f64 / 2f64.powi(30));
}

// ───────────────────────── exl3-hash ─────────────────────────

fn exl3_hash(dir: &Path, prefix: &str, par: bool) {
    use sha2::{Digest, Sha256};
    let mut idx = safetensors::ShardIndex::scan(dir).expect("scan");
    let trellis = idx.get_i16(&format!("{prefix}.trellis")).expect("trellis");
    let shape = idx.entries[&format!("{prefix}.trellis")].shape.clone();
    assert!(shape.len() == 3 && shape[2] == 64, "trellis 形状 {:?}", shape);
    let (kt, nt) = (shape[0], shape[1]);
    let f = exl3::mcg_table();
    let inner = if par {
        exl3::decode_inner_par(&trellis, kt, nt, &f)
    } else {
        exl3::decode_inner(&trellis, kt, nt, &f)
    };
    let mut h = Sha256::new();
    for w in &inner {
        h.update(w.to_le_bytes());
    }
    println!(
        "{}.trellis [{}x{}] → inner [{}x{}] sha256 = {}",
        prefix,
        kt,
        nt,
        kt * 16,
        nt * 16,
        hex(&h.finalize())
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ───────────────────────── bench-decode ─────────────────────────

fn bench_decode(dir: &Path) {
    let t0 = std::time::Instant::now();
    let mut idx = safetensors::ShardIndex::scan(dir).expect("scan");
    println!("scan: {:.1}s ({} 张量)", t0.elapsed().as_secs_f32(), idx.entries.len());
    let f = exl3::mcg_table();
    for (name, par) in [("par", true), ("serial", false)] {
        let t = std::time::Instant::now();
        let mut n = 0usize;
        for e in 0..16u32 {
            for w in ["gate_proj", "up_proj", "down_proj"] {
                let p = format!("model.language_model.layers.4.mlp.experts.{}.{}", e, w);
                let tr = idx.get_i16(&format!("{p}.trellis")).expect("tr");
                let sh = &idx.entries[&format!("{p}.trellis")].shape;
                let (kt, nt) = (sh[0], sh[1]);
                let out = if par {
                    exl3::decode_inner_par(&tr, kt, nt, &f)
                } else {
                    exl3::decode_inner(&tr, kt, nt, &f)
                };
                n += out.len();
            }
        }
        println!(
            "{}: 48 专家 inner 解码 {:.0} ms(校验 {} 字,累计 {:.2} GB)",
            name,
            t.elapsed().as_millis(),
            n,
            n as f64 * 2.0 / 1e9
        );
    }
}

// ───────────────────────── cuda-smoke ─────────────────────────

fn cuda_smoke() {
    let a = tch::Tensor::randn(&[64, 128], (tch::Kind::Float, tch::Device::Cpu));
    let b = tch::Tensor::randn(&[128, 32], (tch::Kind::Float, tch::Device::Cpu));
    let _ = a.matmul(&b);
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    println!("libtorch_cuda loaded: {}", maps.contains("libtorch_cuda.so"));
    let dev = tch::Device::Cuda(0);
    let ag = a.to_device(dev).matmul(&b.to_device(dev));
    println!(
        "CUDA matmul ok: {:?} mean={:.4} device={:?}",
        ag.size(),
        f64::try_from(ag.mean(None)).unwrap(),
        ag.device()
    );
}

// ───────────────────────── parity(Rust vs Python 导出) ─────────────────────────

fn parity(model_dir: &Path, bin_path: &str, n: usize) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    let w = weights::ModelWeights::load(model_dir, &cfg, n, dev);
    let mut pool = moe::ExpertPool::new(model_dir, 384);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bin_path.replace(".bin", ".json")).unwrap())
            .unwrap();
    let blob = std::fs::read(bin_path).unwrap();

    let ids: Vec<i64> = vec![154822, 154824, 154826, 25062, 287, 29905];
    let ids_t = Tensor::from_slice(&ids).to_device(dev);
    let x = w.embed_tokens(&ids_t);
    let mut residual = mhc::hc_expand(&x);
    let mut deferred: Option<(mhc::PreOut, Tensor)> = None;

    let get_ref = |key: &str| -> Option<Tensor> {
        let spec = manifest.get(key)?;
        let off = spec["offset"].as_u64().unwrap() as usize;
        let cnt = spec["count"].as_u64().unwrap() as usize;
        let v: Vec<f32> = blob[off..off + cnt * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shp: Vec<i64> = spec["shape"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        Some(Tensor::from_slice(&v).view(shp.as_slice()).to_device(dev))
    };
    let rel_of = |a: &Tensor, b: &Tensor| -> f64 {
        let bn = b.pow(&Tensor::from(2.0)).sum(None).sqrt();
        let d = a.to_kind(Kind::Float) - b;
        f64::try_from(d.pow(&Tensor::from(2.0)).sum(None).sqrt() / bn).unwrap()
    };

    for (i, layer) in w.layers.iter().enumerate() {
        if let Some((pre, m)) = deferred.take() {
            residual = mhc::mhc_post(&m, &residual, &pre);
        }
        let (pre, z) = mhc::mhc_pre(&residual, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base, &layer.hc.in_ln);
        let a = if let Some(kw) = &layer.kda {
            kda::kda_forward(kw, &z)
        } else {
            mla::mla_forward(layer.mla.as_ref().unwrap(), &z)
        };
        residual = mhc::mhc_post(&a, &residual, &pre);
        if i == 0 {
            for probe in ["z0", "a0"] {
                if let Some(r) = get_ref(probe) {
                    let mine = if probe == "z0" { &z } else { &a };
                    println!("{probe}: rel = {:.3e}", rel_of(mine, &r));
                }
            }
        }
        let (pre2, z2) = mhc::mhc_pre(&residual, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base, &layer.hc.post_ln);
        let m = if let Some(dm) = &layer.dense {
            let g = weights::mm16(&z2, &dm.wg).silu();
            weights::mm16(&(g * weights::mm16(&z2, &dm.wu)), &dm.wd)
        } else {
            let mm = layer.moe.as_ref().unwrap();
            let (topi, wts) = moe::route(&z2, &mm.w_gate, &mm.bias, 8);
            let mut y = Tensor::zeros([z2.size()[0], z2.size()[1]], (Kind::Float, z2.device()));
            for t in 0..z2.size()[0] {
                for ki in 0..topi.size()[1] {
                    let e = topi.get(t).get(ki).int64_value(&[]) as usize;
                    let (wg, wu, wd) = pool.expert(i, e, z2.device());
                    let one = z2.get(t).unsqueeze(0);
                    let eo = moe::expert_forward(&one, wg, wu, wd);
                    let contrib = wts.get(t).get(ki) * eo;
                    let old = y.narrow(0, t, 1);
                    let _ = y.narrow(0, t, 1).copy_(&(old + contrib));
                }
            }
            y + moe::shared_forward(mm, &z2)
        };
        if i == 0 {
            for probe in ["r0", "z2", "m0"] {
                if let Some(r) = get_ref(probe) {
                    let mine = match probe {
                        "r0" => &residual,
                        "z2" => &z2,
                        _ => &m,
                    };
                    println!("{probe}: rel = {:.3e}", rel_of(mine, &r));
                }
            }
        }
        if i == 3 {
            for probe in ["z3", "m3"] {
                if let Some(r) = get_ref(probe) {
                    let mine = if probe == "z3" { &z2 } else { &m };
                    println!("{probe}: rel = {:.3e}", rel_of(mine, &r));
                }
            }
            if let Some(r) = get_ref("topi3") {
                let mm = layer.moe.as_ref().unwrap();
                let (topi, _) = moe::route(&z2, &mm.w_gate, &mm.bias, 8);
                println!("topi3: rel = {:.3e}", rel_of(&topi.to_kind(Kind::Float), &r));
            }
            if let Some(r) = get_ref("exp0_wg") {
                let mm = layer.moe.as_ref().unwrap();
                let (topi, _) = moe::route(&z2, &mm.w_gate, &mm.bias, 8);
                let e0 = topi.get(0).get(0).int64_value(&[]) as usize;
                let (wg, _, _) = pool.expert(3, e0, z2.device());
                println!("exp0({e0})_wg: rel = {:.3e}", rel_of(wg, &r));
            }
        }
        if i == n - 1 {
            residual = mhc::mhc_post(&m, &residual, &pre2);
        } else {
            deferred = Some((pre2, m));
        }
        if let Some(r) = get_ref(&format!("L{i}")) {
            println!("L{i}: rel = {:.4e}", rel_of(&residual, &r));
        }
    }

    // 收尾:contract → final_norm → lm_head
    if let Some(r) = get_ref("final_hidden") {
        let s = mhc::hc_contract(&residual);
        let sq = &s * &s;
        let ms = sq.mean_dim(&[-1i64][..], true, Kind::Float);
        let z = &w.final_norm * (s * (ms + 1e-5).rsqrt());
        let logits = w.logits(&z);
        println!("final_hidden: rel = {:.4e}", rel_of(&z, &r));
        let am = logits.get(logits.size()[0] - 1).argmax(-1, false).int64_value(&[]);
        if let Some(py) = get_ref("argmax") {
            let py_am = f64::try_from(py.mean(None)).unwrap() as i64;
            println!("argmax: rs = {am} py = {py_am}");
        }
    }
}

// ───────────────────────── kda-probe ─────────────────────────

fn kda_probe(model_dir: &Path, bin_path: &str) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    let w = weights::ModelWeights::load(model_dir, &cfg, 1, dev);
    let kw = w.layers[0].kda.as_ref().expect("L0 KDA");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bin_path.replace(".bin", ".json")).unwrap()).unwrap();
    let blob = std::fs::read(bin_path).unwrap();
    let get_ref = |key: &str| -> Tensor {
        let spec = &manifest[key];
        let off = spec["offset"].as_u64().unwrap() as usize;
        let cnt = spec["count"].as_u64().unwrap() as usize;
        let v: Vec<f32> = blob[off..off + cnt * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let shp: Vec<i64> = spec["shape"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        Tensor::from_slice(&v).view(shp.as_slice()).to_device(dev)
    };
    let rel = |a: &Tensor, b: &Tensor| f64::try_from(
        (a.to_kind(Kind::Float) - b).pow(&Tensor::from(2.0)).sum(None).sqrt()
            / b.pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap();

    let ids: Vec<i64> = vec![154822, 154824, 154826, 25062, 287, 29905];
    let ids_t = Tensor::from_slice(&ids).to_device(dev);
    let x = w.embed_tokens(&ids_t);
    let residual = mhc::hc_expand(&x);
    let (_, z) = mhc::mhc_pre(&residual, &w.layers[0].hc.attn_fn, &w.layers[0].hc.attn_scale, &w.layers[0].hc.attn_base, &w.layers[0].hc.in_ln);

    let t = z.size()[0];
    let (hh, dd) = (64i64, 128i64);
    let sc = 128f64.powf(-0.5);
    let l2 = |x: &Tensor| { let sq = x * x; x / (sq.sum_dim_intlist(&[-1i64][..], true, Kind::Float) + 1e-6).sqrt() };
    let q = l2(&kda::causal_conv1d_pub(&weights::mm16(&z, &kw.wq), &kw.conv_q).silu().view([t, hh, dd])) * sc;
    let k = l2(&kda::causal_conv1d_pub(&weights::mm16(&z, &kw.wk), &kw.conv_k).silu().view([t, hh, dd]));
    let v = kda::causal_conv1d_pub(&weights::mm16(&z, &kw.wv), &kw.conv_v).silu().view([t, hh, dd]);
    println!("q: rel = {:.3e}", rel(&q, &get_ref("qkda")));
    println!("k: rel = {:.3e}", rel(&k, &get_ref("kkda")));
    println!("v: rel = {:.3e}", rel(&v, &get_ref("vkda")));
    let beta = z.matmul(&kw.wb.transpose(0, 1)).sigmoid();
    println!("beta: rel = {:.3e}", rel(&beta, &get_ref("beta")));
    let g1 = weights::mm16(&weights::mm16(&z, &kw.fa), &kw.fb).view([t, hh, dd]);
    let a = kw.a_log.exp().unsqueeze(0).unsqueeze(-1);
    let lam = -5.0f64 / ((a * (g1 + kw.dt_bias.unsqueeze(0))).neg().exp() + 1.0);
    println!("lam: rel = {:.3e}", rel(&lam, &get_ref("lam")));

    let mut hstate = Tensor::zeros([hh, dd, dd], (Kind::Float, dev));
    let mut hs: Vec<Tensor> = Vec::new();
    for ti in 0..t {
        hstate = &hstate * lam.get(ti).unsqueeze(1).exp();
        let kt = k.get(ti);
        let vt = v.get(ti);
        let hk = hstate.matmul(&kt.unsqueeze(-1)).squeeze_dim(-1);
        hstate = &hstate + (beta.get(ti).unsqueeze(-1) * (vt - &hk)).unsqueeze(-1) * kt.unsqueeze(1);
        hs.push(hstate.shallow_clone());
    }
    for (name, idx) in [("h1", 0usize), ("h3", 2), ("h5", 4)] {
        println!("{name}: rel = {:.3e}", rel(&hs[idx], &get_ref(name)));
    }
    let o5 = hs[4].matmul(&q.get(5).unsqueeze(-1)).squeeze_dim(-1);
    println!("o5: rel = {:.3e}", rel(&o5, &get_ref("o5")));
}

// ───────────────────────── shim-smoke(exl3_gemm 直算) ─────────────────────────

fn shim_smoke(model_dir: &Path) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let mut idx = safetensors::ShardIndex::scan(model_dir).expect("scan");
    let p = "model.language_model.layers.4.mlp.experts.0.gate_proj";
    let (suh, _) = idx.get_f32(&format!("{p}.suh")).expect("suh");
    let (svh, _) = idx.get_f32(&format!("{p}.svh")).expect("svh");
    let trellis_raw = idx.get_i16(&format!("{p}.trellis")).expect("tr");
    let shp = idx.entries[&format!("{p}.trellis")].shape.clone();
    let (kt, nt) = (shp[0] as i64, shp[1] as i64);

    let trellis_t = Tensor::from_slice(&trellis_raw).view([kt, nt, 64]).to_device(dev);
    let suh_t = Tensor::from_slice(&suh).to_kind(Kind::Half).to_device(dev);
    let svh_t = Tensor::from_slice(&svh).to_kind(Kind::Half).to_device(dev);
    let (k, n) = (kt * 16, nt * 16);
    let rows: i64 = 8;
    let x = Tensor::randn([rows, k], (Kind::Half, dev));
    let y = Tensor::zeros([rows, n], (Kind::Half, dev));
    let xh = Tensor::empty_like(&x);

    extern "C" {
        fn rs_exl3_gemm(x_p: *const u8, y_p: *mut u8, xh_p: *mut u8,
                        tr_p: *const u8, suh_p: *const u8, svh_p: *const u8,
                        rows: i64, k: i64, n: i64, kt: i64, nt: i64) -> i32;
    }
    let rc = unsafe {
        rs_exl3_gemm(x.data_ptr() as *const u8, y.data_ptr() as *mut u8, xh.data_ptr() as *mut u8,
                     trellis_t.data_ptr() as *const u8,
                     suh_t.data_ptr() as *const u8, svh_t.data_ptr() as *const u8,
                     rows, k, n, kt, nt)
    };
    let y2 = y.to_kind(Kind::Float);
    println!("rs_exl3_gemm rc = {rc}, y {:?} norm = {:.4}", y2.size(),
        f64::try_from(y2.pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap_or(0.0));

    let mut pool = moe::ExpertPool::new(model_dir, 4);
    let (wg, _, _) = pool.expert(4, 0, dev);
    let ref_y = x.to_kind(Kind::Float).matmul(wg);
    let d = &y2 - &ref_y;
    let rel = f64::try_from(
        d.pow(&Tensor::from(2.0)).sum(None).sqrt()
            / ref_y.pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap();
    println!("trellis-GEMM vs 解码路径: rel = {rel:.4e}");
}

// ───────────────────────── greedy(端到端) ─────────────────────────

fn greedy(model_dir: &Path, ids_str: &str, max_new: usize) {
    use tch::Device;
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    println!("[greedy] 加载 {} 层权重...", cfg.num_hidden_layers);
    let t0 = std::time::Instant::now();
    let w = weights::ModelWeights::load(model_dir, &cfg, cfg.num_hidden_layers, dev);
    println!("[greedy] 权重就绪 {:.1}s", t0.elapsed().as_secs_f32());
    let mut pool = moe::ExpertPool::new(model_dir, 384);
    let native = moe::NativeExpertPool::new(model_dir, 2048); // greedy 无 fast,native 是主路径
    let mut eng = forward::Engine { w, pool, native: Some(native), fast: None };
    let ids: Vec<i64> = ids_str.split(',').map(|s| s.trim().parse().expect("id")).collect();
    let t1 = std::time::Instant::now();
    let out = eng.greedy(&ids, max_new);
    println!("[greedy] {:.1}s:out = {:?}", t1.elapsed().as_secs_f32(), out);
}


/// M1.3 增量贪心:prefill 一次 + N 步 O(1)。
fn greedy_inc(model_dir: &Path, ids_str: &str, max_new: usize) {
    use tch::Device;
    tp::init_from_env();
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    println!("[greedy-inc] 加载 {} 层权重...", cfg.num_hidden_layers);
    let t0 = std::time::Instant::now();
    let w = weights::ModelWeights::load(model_dir, &cfg, cfg.num_hidden_layers, dev);
    println!("[greedy-inc] 权重就绪 {:.1}s", t0.elapsed().as_secs_f32());
    // fast 路径接管全部专家执行;native 池置空,避免双池并存的足迹陷阱(审阅意见#2)
    let native: Option<moe::NativeExpertPool> = None;
    let pool = moe::ExpertPool::new(model_dir, 4);
    let fast = moefast::MoeFast::new(model_dir, cfg.num_hidden_layers, cfg.n_routed_experts, fast_cap(&cfg), dev);
    let mut fast = fast;
    if std::env::var("GLM53_GRAPH").is_ok() {
        assert!(crate::tp::is_tp(), "GLM53_GRAPH 仅支持 TP2(单机装不下全部专家,逐出会静默错算)");
        let nl = cfg.num_hidden_layers;
        let ne = cfg.n_routed_experts;
        fast.preload_all(nl, ne, dev); // 图模式前提:全驻留
    }
    let mut eng = forward::Engine { w, pool, native, fast: Some(fast) };
    let ids: Vec<i64> = ids_str.split(',').map(|s| s.trim().parse().expect("id")).collect();
    let t1 = std::time::Instant::now();
    let out = eng.greedy_incremental(&ids, max_new);
    println!("[greedy-inc] {:.1}s:out = {:?}", t1.elapsed().as_secs_f32(), out);
}


/// 稳态步时基准:prefill 短 prompt + warm N 步(热专家) → 计时 N 步。
/// fast 路径(mgemm 合批)开关由环境变量 GLM53_NO_FAST 控制。
fn bench_step(model_dir: &Path, n: usize) {
    use tch::Device;
    tp::init_from_env();
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    let t0 = std::time::Instant::now();
    let w = weights::ModelWeights::load(model_dir, &cfg, cfg.num_hidden_layers, dev);
    println!("[bench] 权重就绪 {:.1}s", t0.elapsed().as_secs_f32());
    // fast 路径接管全部专家执行;native 池置空,避免双池并存的足迹陷阱(审阅意见#2)
    let native: Option<moe::NativeExpertPool> = None;
    let pool = moe::ExpertPool::new(model_dir, 4);
    let fast = if std::env::var("GLM53_NO_FAST").is_ok() { None }
        else { Some(moefast::MoeFast::new(model_dir, cfg.num_hidden_layers, cfg.n_routed_experts, fast_cap(&cfg), dev)) };
    let mut fast = fast;
    if std::env::var("GLM53_GRAPH").is_ok() {
        assert!(crate::tp::is_tp(), "GLM53_GRAPH 仅支持 TP2(单机装不下全部专家,逐出会静默错算)");
        if let Some(f) = fast.as_mut() { f.preload_all(cfg.num_hidden_layers, cfg.n_routed_experts, dev); }
    }
    let mut eng = forward::Engine { w, pool, native, fast };
    let ids: Vec<i64> = vec![154822, 154824, 154826, 25062, 287, 29905];
    if std::env::var("GLM53_GRAPH").is_ok() {
        // 图模式:预热+捕获在驱动内完成;计时 = 2n 步总时长
        let t1 = std::time::Instant::now();
        let _ = eng.greedy_incremental(&ids, 2 * n);
        println!("[bench] 图模式总时长 {:.1}s / {} tok", t1.elapsed().as_secs_f64(), 2 * n);
        return;
    }
    let ids_t = tch::Tensor::from_slice(&ids).to_device(dev);
    let (lg, mut ds) = eng.prefill(&ids_t);
    let mut nxt = lg.get(lg.size()[0] - 1).argmax(-1, false).int64_value(&[]);
    // warm:热专家 + cudnn/autotune 稳定
    let fixed: Option<i64> = std::env::var("GLM53_FIXED_TOK").ok().and_then(|s| s.parse().ok());
    for _ in 0..n {
        let lg = eng.step(nxt, &mut ds);
        nxt = fixed.unwrap_or_else(|| lg.argmax(-1, false).int64_value(&[]));
    }
    let m0 = eng.fast.as_ref().map(|f| f.misses).unwrap_or(0);
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        let lg = eng.step(nxt, &mut ds);
        nxt = fixed.unwrap_or_else(|| lg.argmax(-1, false).int64_value(&[]));
    }
    let dt = t1.elapsed().as_secs_f64() / n as f64;
    let misses = eng.fast.as_ref().map(|f| f.misses).unwrap_or(0) - m0;
    println!("[bench] 稳态 {:.1} ms/token ({:.2} tok/s),n={},计时窗冷装载 {}", dt * 1e3, 1.0 / dt, n, misses);
}

/// 状态等价性测试(审阅 §3.4 契约):
/// ① 分块 prefill ≡ 整段 prefill(logits + KDA h/conv + MLA k/v 状态)
/// ② 快照回滚:prefill → 快照 → 3 步 → 恢复 → 同 3 步,输出逐位一致
/// 用 4 层(含 KDA×3 + DSA×1 + MoE),小规模快验证。
fn state_eq(model_dir: &Path) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    let n_layers = 4usize;
    let w = weights::ModelWeights::load(model_dir, &cfg, n_layers, dev);
    let native = moe::NativeExpertPool::new(model_dir, 64);
    let pool = moe::ExpertPool::new(model_dir, 4);
    let mut eng = forward::Engine { w, pool, native: Some(native), fast: None };
    let ids: Vec<i64> = vec![154822, 154824, 154826, 25062, 287, 29905, 151645, 198];
    let rel = |a: &Tensor, b: &Tensor| -> f64 {
        let d = a - b;
        f64::try_from(d.pow(&Tensor::from(2.0)).sum(None).sqrt()
            / b.pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap()
    };

    // ① 分块 ≡ 整段
    let ids_t = Tensor::from_slice(&ids).to_device(dev);
    let (lg_full, ds_full) = eng.prefill(&ids_t);
    let (_lg_a, ds_a) = eng.prefill(&ids_t.narrow(0, 0, 3));
    let (lg_b, ds_b) = eng.prefill_with(&ids_t.narrow(0, 3, 5), Some(ds_a));
    let r1 = rel(&lg_full.narrow(0, 3, 5), &lg_b);
    let d1 = forward::states_max_diff(&ds_full, &ds_b);
    println!("分块 prefill: logits rel = {r1:.3e}, 状态 max|Δ| = {d1:.3e}");
    // fp16 GEMM 形状相关舍入(3/5/8 行不同 tile 拆分):容差 3e-3/5e-3;
    // 语义等价性由同形状逐位回滚测试兜底。
    assert!(r1 < 3e-3 && d1 < 5e-3, "分块 prefill 与整段不等价");

    // ② 回滚:快照 → 3 步 → 恢复 → 重走同 3 步
    let (_lg0, ds0) = eng.prefill(&ids_t);
    let snap = forward::snapshot(&ds0);
    let mut ds_run = forward::snapshot(&ds0);
    let toks = [100i64, 200, 300];
    let mut run1 = Vec::new();
    for &t in &toks { run1.push(eng.step(t, &mut ds_run)); }
    let mut ds_run2 = forward::snapshot(&snap);
    let mut run2 = Vec::new();
    for &t in &toks { run2.push(eng.step(t, &mut ds_run2)); }
    let mut worst = 0.0f64;
    for (a, b) in run1.iter().zip(run2.iter()) { worst = worst.max(rel(a, b)); }
    let d2 = forward::states_max_diff(&ds_run, &ds_run2);
    println!("快照回滚: 3 步 logits 最差 rel = {worst:.3e}, 状态 max|Δ| = {d2:.3e}");
    assert!(worst == 0.0 && d2 == 0.0, "回滚后重跑不逐位一致(状态污染?)");

    println!("state-eq PASS");
}

/// CUDA Graph 冒烟:捕获 x@W→(allreduce)→×2,改输入回放,验证输出跟随。
fn graph_smoke() {
    use tch::{Device, Kind, Tensor};
    tp::init_from_env();
    let dev = Device::cuda_if_available();
    let mut x = Tensor::ones([1, 4096], (Kind::Float, dev));
    tch::manual_seed(42); // 双 rank 同一 w(TP 正确性要求)
    let w = Tensor::randn([4096, 4096], (Kind::Float, dev)) * 0.001;
    // 暖场(建 cublas handle/autotune)
    let _ = x.matmul(&w);
    tch::Cuda::synchronize(0);
    tp::graph::begin().expect("begin");
    let y = if std::env::var("GSM_ADDONLY").is_ok() { &x + 1.0 } else { x.matmul(&w) };
    if tp::is_tp() {
        tp::allreduce(&y);
    }
    let z = &y * 2.0;
    let z = z.shallow_clone(); // 固定输出张量地址
    tp::graph::end().expect("end");
    let _ = z.print();
    // 回放:改 x → z 必须跟着变
    let _ = x.fill_(3.0);
    tp::graph::replay().expect("replay");
    tch::Cuda::synchronize(0);
    let got = f64::try_from(z.mean(None)).unwrap();
    let addonly = std::env::var("GSM_ADDONLY").is_ok();
    let expect = if addonly {
        (3.0 + 1.0) * 2.0
    } else {
        // ones[1,4096] @ w → 每列和;mean 过 4096 列;allreduce 再乘 world
        3.0 * (f64::try_from(w.sum(None)).unwrap() / 4096.0) * 2.0 * tp::world().world as f64
    };
    let rel = (got - expect).abs() / expect.abs().max(1e-12);
    println!("[graph-smoke] rank{} z.mean={got:.6} expect={expect:.6} rel={rel:.2e}", tp::world().rank);
    assert!(rel < 1e-3, "图回放输出不跟随输入");
    println!("[graph-smoke] PASS(allreduce 入图:{})", tp::is_tp());
}

/// TP 通信微基准:1000 次 [1,4096] fp32 allreduce,报平均时延。
fn tp_ping() {
    use tch::{Device, Kind, Tensor};
    let world = tp::init_from_env();
    let dev = Device::cuda_if_available();
    let probe = Tensor::full([1,4096],(world.rank+1) as f64,(Kind::Float,dev));
    tp::allreduce(&probe);
    let expected = (world.world*(world.world+1)/2) as f64;
    assert!(probe.eq(expected).all().int64_value(&[])!=0,"TP SUM mismatch");
    // Repeatedly summing ones overflows to Inf. Zero stays finite without
    // introducing a reset kernel into the measured collective loop.
    let t = Tensor::zeros([1, 4096], (Kind::Float, dev));
    for _ in 0..10 { tp::allreduce(&t); }  // 暖场
    tch::Cuda::synchronize(0);
    tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));
    tch::Cuda::synchronize(0);
    let t0 = std::time::Instant::now();
    let n = 1000;
    for _ in 0..n { tp::allreduce(&t); }
    tch::Cuda::synchronize(0);
    let dt = t0.elapsed().as_secs_f64() / n as f64;
    assert!(t.eq(0.).all().int64_value(&[])!=0,"TP ping became nonzero or nonfinite");
    println!("[tp-ping] rank{} allreduce 16KB × {}: {:.1} μs/次", tp::world().rank, n, dt * 1e6);
}

/// mgemm 合批 vs 逐专家串行的微基准(热专家,隔离 MoE 内核吞吐)。
fn fast_bench(model_dir: &Path) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("cfg");
    let mut fast = moefast::MoeFast::new(model_dir, cfg.num_hidden_layers, cfg.n_routed_experts, 64, dev);
    let mut native = moe::NativeExpertPool::new(model_dir, 64);
    let x16 = Tensor::randn([1, 4096], (Kind::Half, dev));
    let wts = Tensor::full([8], 0.125f64, (Kind::Float, dev));
    let selected: Vec<usize> = (0..8).collect();
    let topi = Tensor::from_slice(&(0..8i64).collect::<Vec<_>>()).to_device(dev);
    // 预热(含 autotune + 专家装载)
    let _ = fast.expert_batch(3, &x16, &selected, &wts, dev);
    for &e in &selected { let _ = native.expert(3, e, dev); }
    let _ = fast.sel_device(3, &topi, dev);
    tch::Cuda::synchronize(0);

    let n = 50;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = fast.expert_batch(3, &x16, &selected, &wts, dev);
    }
    tch::Cuda::synchronize(0);
    let mg = t.elapsed().as_secs_f64() / n as f64;

    let t = std::time::Instant::now();
    for _ in 0..n {
        for &e in &selected {
            let pv = native.expert(3, e, dev);
            let projs = [(pv[0].0.shallow_clone(), pv[0].1.shallow_clone(), pv[0].2.shallow_clone()),
                         (pv[1].0.shallow_clone(), pv[1].1.shallow_clone(), pv[1].2.shallow_clone()),
                         (pv[2].0.shallow_clone(), pv[2].1.shallow_clone(), pv[2].2.shallow_clone())];
            let _ = moe::NativeExpertPool::expert_forward(&x16, &projs, dev);
        }
    }
    tch::Cuda::synchronize(0);
    let nat = t.elapsed().as_secs_f64() / n as f64;
    println!("[fast-bench] 8 专家一层: mgemm 合批 {:.2}ms | 24×单 GEMM {:.2}ms | 加速 {:.2}×",
             mg * 1e3, nat * 1e3, nat / mg);
    // 带宽参考:8 专家 × 12MiB = 96MiB;mgemm 耗时对应的有效带宽
    println!("[fast-bench] mgemm 有效带宽 ≈ {:.0} GB/s", 8.0 * 12.0 / 1024.0 / mg);
}

/// 行为验收:greedy-inc(fast 路径) vs bench/m0-refs.json 的 token 序列。
/// 判据同 engine/tools/acceptance.py:重叠区间逐 token 一致且 ≥ min(12, ref_len)。
fn accept(model_dir: &Path, prompt_name: &str, max_new: usize) -> bool {
    use tch::Device;
    tp::init_from_env();
    let refs: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("../../../bench/m0-refs.json"))
            .or_else(|_| std::fs::read_to_string("bench/m0-refs.json"))
            .expect("m0-refs.json")).expect("json");
    let r = &refs[prompt_name];
    let ids: Vec<i64> = r["prompt_ids"].as_array().expect("prompt_ids(先跑 acceptance.py --write-ids)")
        .iter().map(|v| v.as_i64().unwrap()).collect();
    let ref_ids: Vec<i64> = r["text_ids"].as_array().expect("text_ids")
        .iter().map(|v| v.as_i64().unwrap()).collect();

    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("config");
    let t0 = std::time::Instant::now();
    let w = weights::ModelWeights::load(model_dir, &cfg, cfg.num_hidden_layers, dev);
    println!("[accept] 权重就绪 {:.1}s", t0.elapsed().as_secs_f32());
    // fast 路径接管全部专家执行;native 池置空,避免双池并存的足迹陷阱(审阅意见#2)
    let native: Option<moe::NativeExpertPool> = None;
    let pool = moe::ExpertPool::new(model_dir, 4);
    let fast = moefast::MoeFast::new(model_dir, cfg.num_hidden_layers, cfg.n_routed_experts, fast_cap(&cfg), dev);
    let mut fast = fast;
    if std::env::var("GLM53_GRAPH").is_ok() {
        assert!(crate::tp::is_tp(), "GLM53_GRAPH 仅支持 TP2(单机装不下全部专家,逐出会静默错算)");
        fast.preload_all(cfg.num_hidden_layers, cfg.n_routed_experts, dev);
    }
    let mut eng = forward::Engine { w, pool, native, fast: Some(fast) };
    let t1 = std::time::Instant::now();
    let (out, logs) = if std::env::var("GLM53_GRAPH").is_ok() {
        eng.greedy_incremental_graph_dbg(&ids, max_new, true)
    } else {
        eng.greedy_incremental_dbg(&ids, max_new, true)
    };
    let el = t1.elapsed().as_secs_f32();

    let (first, matched, pass) = acceptance::compare(&out, &ref_ids, max_new);
    let first_diff = first.map(|p| p as i64).unwrap_or(-1);
    let threshold = ref_ids.len().min(12);
    println!("[accept] {prompt_name}: {:.1}s / {} tok | 重叠一致 {}/{} | w_fp16={}",
             el, out.len(), matched, ref_ids.len(), weights::w_fp16());
    println!("[accept] out_ids = {:?}", out);
    if first_diff >= 0 {
        let p = first_diff as usize;
        if logs.is_empty() { println!("[accept] 图模式未保留 logits，分歧原因未分类"); }
        // 分歧分类:边际翻转(数值精度)vs 逻辑错误。
        // mgemm autotune 按计时选内核,跨进程非确定 → logits 抖动 ~0.05 量级,
        // top5/gap 仅提示数值敏感性，不改变严格验收结果。
        if p < logs.len() {
            let lg = logs[p].to_kind(tch::Kind::Float);
            let top = lg.topk(5, -1, true, true);
            let tv: Vec<f32> = top.0.to_device(tch::Device::Cpu).try_into().expect("v");
            let ti: Vec<i64> = top.1.to_device(tch::Device::Cpu).try_into().expect("i");
            println!("[accept] 分歧处 top5: {:?}", ti.iter().zip(tv.iter()).collect::<Vec<_>>());
            let eg = out[p]; let rf = ref_ids[p];
            let le = f64::try_from(lg.get(eg)).unwrap();
            let lr = f64::try_from(lg.get(rf)).unwrap();
            let gap = (le - lr).abs();
            let in_top5 = ti.contains(&rf);
            println!("[accept] logit(engine={eg})={le:.4} logit(ref={rf})={lr:.4} gap={gap:.4}");
            if in_top5 && gap < 0.2 && matched >= threshold {
                println!("[accept] 首个分歧 @ token {first_diff} 疑似边际翻转，仅作诊断；严格验收仍 FAIL");
            }
        }
        println!("[accept] 首个分歧 @ token {first_diff}: engine={} ref={} → FAIL(token 不一致，原因未确证)",
                 out[p], ref_ids[p]);
        return false;
    }
    println!("[accept] {}", if pass { "PASS" } else { "FAIL(重叠不足)" });
    pass
}

/// 逐投影探针:gate/up(广播 A)/down(逐行 A)分别对拍解码路径。
fn fast_probe(model_dir: &Path) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("cfg");
    let mut fast = moefast::MoeFast::new(model_dir, cfg.num_hidden_layers, cfg.n_routed_experts, 64, dev);
    let mut pool = moe::ExpertPool::new(model_dir, 16);
    let selected: Vec<usize> = (0..8).collect();
    let rel_of = |a: &Tensor, b: &Tensor| -> f64 {
        let d = a.to_kind(Kind::Float) - b.to_kind(Kind::Float);
        f64::try_from(d.pow(&Tensor::from(2.0)).sum(None).sqrt()
            / b.to_kind(Kind::Float).pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap()
    };
    // gate / up:A [1,1,4096] 广播
    let x = Tensor::randn([1, 1, 4096], (Kind::Half, dev));
    for (proj, name) in [(0usize, "gate"), (1, "up")] {
        let c = fast.proj_probe(3, &selected, proj, &x, dev); // [8,1,2048]
        for (j, &e) in selected.iter().enumerate() {
            let (wg, wu, _) = pool.expert(3, e, dev);
            let w = if proj == 0 { wg } else { wu };
            let r = x.get(0).to_kind(Kind::Float).matmul(w); // [1,2048]
            println!("{name} e{e}: rel = {:.4e}", rel_of(&c.get(j as i64).get(0), &r.get(0)));
        }
    }
    // down:A [8,1,2048] 逐专家行
    let a = Tensor::randn([8, 1, 2048], (Kind::Half, dev));
    let c = fast.proj_probe(3, &selected, 2, &a, dev); // [8,1,4096]
    for (j, &e) in selected.iter().enumerate() {
        let (_, _, wd) = pool.expert(3, e, dev);
        let r = a.get(j as i64).to_kind(Kind::Float).matmul(wd); // [1,4096]
        println!("down e{e}: rel = {:.4e}", rel_of(&c.get(j as i64).get(0), &r.get(0)));
    }
}

/// TP 下全量专家驻留(每 rank 半份 ≈6MiB);单机受 UMA 容量限制给 6000。
fn fast_cap(cfg: &config::Config) -> usize {
    if crate::tp::is_tp() { cfg.num_hidden_layers * cfg.n_routed_experts + 16 } else { 6000 }
}

/// mgemm 指针表隔离测试:层 3 专家 0..8(生产形状 k=8,单选路径有内核怪癖不用),
/// 全专家加权输出 vs 解码路径;硬断言 rel < 2e-3(fp16 原生 GEMM 容差,参考 shim-smoke 4.6e-4)。
fn fast_smoke(model_dir: &Path) {
    use tch::{Device, Kind, Tensor};
    let dev = Device::cuda_if_available();
    let cfg = config::load(&model_dir.join("config.json")).expect("cfg");
    let n_layers = cfg.num_hidden_layers;
    let mut fast = moefast::MoeFast::new(model_dir, n_layers, cfg.n_routed_experts, 64, dev);
    let x16 = Tensor::randn([1, 4096], (Kind::Half, dev));
    let selected: Vec<usize> = (0..8).collect();
    let wts = Tensor::full([8], 0.125f64, (Kind::Float, dev)); // 均匀权重(已过 renorm 语义)
    let y = fast.expert_batch(3, &x16, &selected, &wts, dev);

    // 参考:逐专家解码路径 + 加权求和
    let mut pool = moe::ExpertPool::new(model_dir, 16);
    let xf = x16.to_kind(tch::Kind::Float);
    let mut ref_y = Tensor::zeros([1, 4096], (Kind::Float, dev));
    for (ki, &e) in selected.iter().enumerate() {
        let (wg, wu, wd) = pool.expert(3, e, dev);
        let eo = moe::expert_forward(&xf, wg, wu, wd);
        ref_y = ref_y + wts.get(ki as i64) * eo;
    }
    let d = &y - &ref_y;
    let rel = f64::try_from(d.pow(&Tensor::from(2.0)).sum(None).sqrt()
        / ref_y.pow(&Tensor::from(2.0)).sum(None).sqrt()).unwrap();
    println!("mgemm(k=8) vs 解码: rel = {rel:.4e}");
    assert!(rel < 2e-3, "mgemm 输出偏差过大 rel={rel:.3e}(指针表/语义回归)");
    println!("fast-smoke PASS");
}
