// SPDX-License-Identifier: MIT
//! SPX1 calibration (`spx-hessian`): GPTQ-style input Hessians H = sum x^T x (FP32, accumulated on the GPU) for every
//! non-expert linear, collected from a teacher-forced prefill over calibration texts.
//!
//! The taps sit at module level where the projection input exists as a tensor (whatever GEMM backend then consumes
//! it), keyed by (layer, kind). They are no-ops unless `begin` was called on this thread: one thread-local check per
//! tap, no tensor work, so the default paths are unchanged.
//!
//! Kinds (TP: "shard" = this rank's contiguous slice of the input dimension, the same slice `shard_dense_layer` keeps):
//!   kda_in      KDA q/k/v, f_a, g_a, b input (4096)                     kda_o       KDA o_proj input (shard)
//!   mla_in      q_a, kv_a_proj_with_mqa, indexer wk/weights_proj/gate (4096)
//!   mla_qb      q_b and indexer wq_b input (q_a_layernorm output, 1536)
//!   mla_kvb     kv_b input: the kv_a_layernorm latent c_kv (FP32, before the Half/FP8 cache rounding; kv_b is only
//!               used absorbed, this is the latent it multiplies) (512)
//!   mla_o       MLA o_proj input (shard)
//!   shared_in   shared expert gate/up input (4096)                        shared_down shared down_proj input (shard)
//!   dense_in    dense MLP (layers 0-2) gate/up input (4096)               dense_down  dense down_proj input (shard)
//!   head_in     lm_head input (final norm output, all rows) (4096); layer index = num_hidden_layers
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use tch::{Kind, Tensor};

pub const KDA_IN: &str = "kda_in";
pub const KDA_O: &str = "kda_o";
pub const MLA_IN: &str = "mla_in";
pub const MLA_QB: &str = "mla_qb";
pub const MLA_KVB: &str = "mla_kvb";
pub const MLA_O: &str = "mla_o";
pub const SHARED_IN: &str = "shared_in";
pub const SHARED_DOWN: &str = "shared_down";
pub const DENSE_IN: &str = "dense_in";
pub const DENSE_DOWN: &str = "dense_down";
pub const HEAD_IN: &str = "head_in";
const ALL: [&str; 11] = [KDA_IN, KDA_O, MLA_IN, MLA_QB, MLA_KVB, MLA_O, SHARED_IN, SHARED_DOWN, DENSE_IN, DENSE_DOWN, HEAD_IN];
/// Kinds whose input is this rank's shard of the full input dimension under dense TP.
const SHARDED: [&str; 4] = [KDA_O, MLA_O, SHARED_DOWN, DENSE_DOWN];

struct Collector {
    layer: usize,
    kinds: Option<HashSet<&'static str>>,
    tf32: bool,
    acc: BTreeMap<(usize, &'static str), (Tensor, i64)>,
}
thread_local! {static HESS: RefCell<Option<Collector>> = const {RefCell::new(None)};}

pub fn begin(kinds: Option<HashSet<&'static str>>) {
    let tf32 = std::env::var("GLM53_TF32").as_deref() == Ok("1");
    HESS.with(|h| *h.borrow_mut() = Some(Collector { layer: 0, kinds, tf32, acc: BTreeMap::new() }));
}
fn take() -> BTreeMap<(usize, &'static str), (Tensor, i64)> {
    HESS.with(|h| h.borrow_mut().take().map(|c| c.acc).unwrap_or_default())
}
/// Current decoder layer for the taps below (prefill loop).
#[inline]
pub fn set_layer(i: usize) {
    HESS.with(|h| if let Some(c) = h.borrow_mut().as_mut() { c.layer = i; });
}
#[inline]
pub fn add(kind: &'static str, x: &Tensor) { accumulate(None, kind, x); }
#[inline]
pub fn add_at(layer: usize, kind: &'static str, x: &Tensor) { accumulate(Some(layer), kind, x); }
/// `add`, returning its argument (for taps inside a call expression).
#[inline]
pub fn tap<'a>(kind: &'static str, x: &'a Tensor) -> &'a Tensor { accumulate(None, kind, x); x }
#[inline]
pub fn tap_owned(kind: &'static str, x: Tensor) -> Tensor { accumulate(None, kind, &x); x }

fn accumulate(layer: Option<usize>, kind: &'static str, x: &Tensor) {
    HESS.with(|h| {
        let mut h = h.borrow_mut();
        let Some(c) = h.as_mut() else { return };
        if c.kinds.as_ref().is_some_and(|k| !k.contains(kind)) { return; }
        let layer = layer.unwrap_or(c.layer);
        let d = *x.size().last().unwrap();
        let x2 = x.reshape([-1, d]).to_kind(Kind::Float);
        let e = c.acc.entry((layer, kind)).or_insert_with(|| (Tensor::zeros([d, d], (Kind::Float, x2.device())), 0));
        assert_eq!(e.0.size(), [d, d], "spx-hessian: L{layer} {kind} input width changed");
        // Exact FP32 products (no TF32) for the Hessian; restore the run's setting for the model's own GEMMs.
        crate::tp::set_tf32(false);
        let _ = e.0.addmm_(&x2.transpose(0, 1), &x2);
        crate::tp::set_tf32(c.tf32);
        e.1 += x2.size()[0];
    });
}

/// Lowest MemAvailable (MiB) seen by the watchdog thread (200 ms sampling).
static MIN_AVAIL_MIB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
fn mem_available_gib() -> f64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find_map(|l| l.strip_prefix("MemAvailable:").map(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())).flatten())
        .map(|kb| kb / (1024. * 1024.)).unwrap_or(f64::NAN)
}

/// Checkpoint tensors (without the `model.language_model.` prefix for layers) a (layer, kind) Hessian applies to.
fn weight_names(layer: usize, kind: &str) -> Vec<String> {
    let l = format!("model.language_model.layers.{layer}");
    let names: Vec<String> = match kind {
        KDA_IN => ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"].iter().map(|n| format!("{l}.self_attn.{n}.weight")).collect(),
        KDA_O | MLA_O => vec![format!("{l}.self_attn.o_proj.weight")],
        MLA_IN => ["q_a_proj.weight", "kv_a_proj_with_mqa.weight", "indexer.wk.weight", "indexer.weights_proj.weight", "indexer.index_kpool_compress_gate"]
            .iter().map(|n| format!("{l}.self_attn.{n}")).collect(),
        MLA_QB => vec![format!("{l}.self_attn.q_b_proj.weight"), format!("{l}.self_attn.indexer.wq_b.weight")],
        MLA_KVB => vec![format!("{l}.self_attn.kv_b_proj.weight")],
        SHARED_IN => vec![format!("{l}.mlp.shared_experts.gate_proj.weight"), format!("{l}.mlp.shared_experts.up_proj.weight")],
        SHARED_DOWN => vec![format!("{l}.mlp.shared_experts.down_proj.weight")],
        DENSE_IN => vec![format!("{l}.mlp.gate_proj.weight"), format!("{l}.mlp.up_proj.weight")],
        DENSE_DOWN => vec![format!("{l}.mlp.down_proj.weight")],
        HEAD_IN => vec!["lm_head.weight".into()],
        _ => unreachable!(),
    };
    names
}

/// Flags this subcommand forces (set before anything reads them). The hooks see the true projection inputs with any
/// GEMM backend, so the production fusions stay on; these only select the hooked code paths:
///   GLM53_DENSE_TP=1            head/intermediate-sharded attention and MLPs (defines the o/down shards recorded)
///   GLM53_MLA_LATENT=1          MLA prefill through mla_latent::chunk (taps live there)
///   GLM53_PREFILL_BATCH=1       batched KDA/MLA prefill (kda_chunk / mla_latent::chunk; not the per-token step path)
///   GLM53_PREFILL_SP=0          no sequence-parallel prefill (every rank sees every row)
///   GLM53_PREFILL_LAST_LOGITS=1 lm_head GEMM only for the last row (head_in is tapped on all rows before that)
const FORCED: [(&str, &str); 5] = [("GLM53_DENSE_TP", "1"), ("GLM53_MLA_LATENT", "1"), ("GLM53_PREFILL_BATCH", "1"),
    ("GLM53_PREFILL_SP", "0"), ("GLM53_PREFILL_LAST_LOGITS", "1")];

/// `spx-hessian <MODEL_DIR>` (TP2: run on both ranks, e.g. `./tp2-run.sh <rank> spx-hessian <MDIR>`).
/// SPX_HESS_GEN: gen.jsonl (bench/opd/export.py); rows with "split":"train", tokens prompt_ids + output_ids.
/// SPX_HESS_TOKENS (default 200000): token budget; the last sequence is truncated to fit exactly.
/// SPX_HESS_KINDS: optional comma list of kinds (collect in several passes to bound memory).
/// SPX_HESS_OUT (default ./spx-hess): writes rank{r}/L{layer:02}_{kind}.pt (H fp32 [d,d], n int64) + manifest.json.
/// SPX_HESS_MIN_AVAIL_GIB (default 5): abort if MemAvailable drops below this (the node memguard only watches serve).
/// Prefill chunk: GLM53_PREFILL_CHUNK (default 4096) rows; state is reset per sequence.
pub fn run(model: &Path) {
    for (k, v) in FORCED { std::env::set_var(k, v); }
    let min_avail: f64 = std::env::var("SPX_HESS_MIN_AVAIL_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    std::thread::spawn(move || loop {
        let a = mem_available_gib();
        MIN_AVAIL_MIB.fetch_min((a * 1024.) as u64, std::sync::atomic::Ordering::Relaxed);
        if a < min_avail {
            eprintln!("[spx-hessian] MemAvailable {a:.2} GiB < {min_avail} GiB: aborting");
            std::process::abort();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    });
    let t_start = std::time::Instant::now();
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    assert!(crate::tp::dense_enabled(), "spx-hessian needs TP (GLM53_TP_WORLD>1) with dense TP");

    // Calibration sequences first (cheap; sizes the MLA latent capacity).
    let gen = std::env::var("SPX_HESS_GEN").expect("SPX_HESS_GEN (gen.jsonl)");
    let budget: usize = std::env::var("SPX_HESS_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let kinds: Option<HashSet<&'static str>> = std::env::var("SPX_HESS_KINDS").ok().filter(|s| !s.is_empty()).map(|s| {
        s.split(',').map(|k| *ALL.iter().find(|a| **a == k.trim()).unwrap_or_else(|| panic!("unknown kind {k}; valid: {ALL:?}"))).collect()
    });
    let mut seqs: Vec<(i64, Vec<i64>)> = Vec::new();
    let mut used = 0usize;
    for line in std::fs::read_to_string(&gen).expect("read SPX_HESS_GEN").lines() {
        if used >= budget { break; }
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(line).expect("gen.jsonl line");
        if v["split"] != "train" { continue; }
        let mut ids: Vec<i64> = v["prompt_ids"].as_array().expect("prompt_ids").iter()
            .chain(v["output_ids"].as_array().expect("output_ids")).map(|x| x.as_i64().unwrap()).collect();
        ids.truncate(budget - used);
        used += ids.len();
        seqs.push((v["idx"].as_i64().unwrap_or(-1), ids));
    }
    assert!(!seqs.is_empty(), "no train rows in {gen}");
    let max_len = seqs.iter().map(|s| s.1.len()).max().unwrap() as i64;
    let chunk = std::env::var("GLM53_PREFILL_CHUNK").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(4096);
    // Small context budget: the latent/DSA state only needs the longest calibration sequence.
    if crate::mla_latent::capacity() < max_len { std::env::set_var("GLM53_MAX_CONTEXT", ((max_len + 1023) / 1024 * 1024).to_string()); }
    eprintln!("[spx-hessian] rank{} {} sequences, {used} tokens (budget {budget}), longest {max_len}, chunk {chunk}, capacity {}, kinds {:?}",
        tp.rank, seqs.len(), crate::mla_latent::capacity(), kinds.as_ref().map(|k| { let mut v: Vec<_> = k.iter().collect(); v.sort(); v }));

    let dev = tch::Device::Cuda(0);
    let cfg = crate::config::load(&model.join("config.json")).unwrap();
    let w = crate::weights::ModelWeights::load(model, &cfg, cfg.num_hidden_layers, dev);
    let mut fast = crate::moefast::MoeFast::new(model, cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_hidden_layers * cfg.n_routed_experts + 16, dev);
    fast.preload_all(cfg.num_hidden_layers, cfg.n_routed_experts, dev);
    fast.assume_hot = true;
    let mut eng = crate::forward::Engine { w, fast: Some(fast), native: None, pool: crate::moe::ExpertPool::new(model, 4) };
    let plans: Vec<(String, String)> = eng.w.layers.iter().map(|l| (l.plan.attn.clone(), l.plan.mlp.clone())).collect();
    let t_loaded = t_start.elapsed().as_secs_f64();
    let mut min_seen = mem_available_gib();
    eprintln!("[spx-hessian] rank{} loaded in {t_loaded:.1}s, MemAvailable {min_seen:.2} GiB", tp.rank);

    begin(kinds);
    let t_run = std::time::Instant::now();
    let mut done = 0usize;
    for (si, (idx, ids)) in seqs.iter().enumerate() {
        let all = Tensor::from_slice(ids).to_device(dev);
        let mut ds = None;
        let mut pos = 0i64;
        let len = ids.len() as i64;
        while pos < len {
            let n = chunk.min(len - pos);
            let (_logits, st, _) = eng.prefill_record_last(&all.narrow(0, pos, n), ds.take(), false);
            ds = Some(st);
            pos += n;
        }
        drop(ds);
        tch::Cuda::synchronize(0);
        done += ids.len();
        let a = mem_available_gib();
        min_seen = min_seen.min(a);
        eprintln!("[spx-hessian] rank{} seq {}/{} idx {idx} len {len} tokens {done} {:.1}s MemAvailable {a:.2} GiB",
            tp.rank, si + 1, seqs.len(), t_run.elapsed().as_secs_f64());
    }
    let t_prefill = t_run.elapsed().as_secs_f64();
    let acc = take();
    let n_layers = eng.w.layers.len();
    drop(eng);
    extern "C" { fn rs_empty_cache(); }
    unsafe { rs_empty_cache() };

    let out = std::path::PathBuf::from(std::env::var("SPX_HESS_OUT").unwrap_or_else(|_| "spx-hess".into())).join(format!("rank{}", tp.rank));
    std::fs::create_dir_all(&out).unwrap();
    let mut entries = Vec::new();
    let mut bytes = 0i64;
    let mut worst_asym = 0f64;
    for ((layer, kind), (h, n)) in acc {
        assert_eq!(n as usize, done, "L{layer} {kind}: {n} rows, {done} tokens");
        let d = h.size()[0];
        let amax = h.abs().max().double_value(&[]);
        let asym = (&h - h.transpose(0, 1)).abs().max().double_value(&[]) / amax.max(f64::MIN_POSITIVE);
        let diag = h.diagonal(0, 0, 1);
        let (dmin, dmean) = (diag.min().double_value(&[]), diag.mean(Kind::Float).double_value(&[]));
        let finite = h.isfinite().all().int64_value(&[]) == 1;
        worst_asym = worst_asym.max(asym);
        let file = format!("L{layer:02}_{kind}.pt");
        Tensor::save_multi(&[("H", &h.to_device(tch::Device::Cpu)), ("n", &Tensor::from(n))], out.join(&file)).unwrap();
        bytes += d * d * 4;
        let sharded = SHARDED.contains(&kind);
        let (full, offset) = if sharded { (d * tp.world as i64, d * tp.rank as i64) } else { (d, 0) };
        let (attn, mlp) = if kind == HEAD_IN { ("head".to_string(), "head".to_string()) } else { plans.get(layer).cloned().unwrap_or_default() };
        let note = if sharded { "input columns [offset, offset+size) of the weights below (contiguous head/intermediate shard)" } else { "full input; identical on every rank" };
        entries.push(serde_json::json!({
            "layer": layer, "kind": kind, "file": file, "dim": d, "n": n,
            "layer_attn": attn, "layer_mlp": mlp,
            "shard": {"sharded": sharded, "rank": tp.rank, "world": tp.world, "full_dim": full, "offset": offset, "size": d, "note": note},
            "weights": weight_names(layer, kind),
            "check": {"asym_rel": asym, "diag_min": dmin, "diag_mean": dmean, "finite": finite},
        }));
        if !(dmin > 0.0) || !finite { eprintln!("[spx-hessian] WARNING L{layer} {kind}: diag_min {dmin} finite {finite}"); }
    }
    let manifest = serde_json::json!({
        "format": "spx1-hessian-v1", "model": model.display().to_string(), "gen": gen, "rank": tp.rank, "world": tp.world,
        "num_hidden_layers": n_layers, "head_layer_index": n_layers, "tokens": done, "sequences": seqs.len(), "token_budget": budget,
        "prefill_chunk": chunk, "forced_env": FORCED.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
        "tf32_in_model": std::env::var("GLM53_TF32").as_deref() == Ok("1"), "hessian_math": "fp32 addmm, TF32 off",
        "mtp_layer": "not run in prefill; no Hessians", "entries": entries,
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    eprintln!("[spx-hessian] rank{} {} Hessians ({:.2} GiB) -> {}; tokens {done}; load {t_loaded:.1}s prefill {t_prefill:.1}s total {:.1}s; min MemAvailable {min_seen:.2} GiB after a sequence, {:.2} GiB sampled; worst asym {worst_asym:.2e}",
        tp.rank, manifest["entries"].as_array().unwrap().len(), bytes as f64 / (1u64 << 30) as f64, out.display(), t_start.elapsed().as_secs_f64(),
        MIN_AVAIL_MIB.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.);
}
