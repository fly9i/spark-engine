// SPDX-License-Identifier: MIT
//! SPX1 drafter calibration (SPX_DHESS_OUT=<dir>): GPTQ input Hessians H = sum x^T x (FP32, TF32 off) for every
//! DFlash2 drafter projection, accumulated from the real drafter inputs while serving (proposal and append paths).
//!
//! Weights are named once at load (`name`, after TP sharding and FP8 compaction, keyed by the tensor's data pointer);
//! the drafter's three GEMM entries (`linear`, `o_proj`, the TP MLP down) call `tap(x, w)`. Unnamed weights and the
//! aliases that always see the same input (v = k, up = gate) are skipped. With SPX_DHESS_OUT unset every tap is one
//! cached env check. The drafter graphs must be off (GLM53_DRAFT_GRAPH=0 GLM53_DRAFT_MANY_GRAPH=0
//! GLM53_DRAFT_APPEND_GRAPH=0): a replay never reaches the taps, and a tap during capture is counted and skipped.
//!
//! Output (rewritten every SPX_DHESS_EVERY proposals, default 100, via tmp + rename): <dir>/rank{r}/<name>.pt
//! {H [d,d] fp32, n int64} + manifest.json (dims, TP input shard of o_proj/down, aliases, counts).
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use tch::{Kind, Tensor};

struct Collector {
    names: HashMap<usize, (String, Vec<i64>)>,
    acc: BTreeMap<String, (Tensor, i64)>,
    proposals: u64,
    skipped_capture: u64,
    every: u64,
    out: std::path::PathBuf,
}
// SAFETY: tch tensors are reference-counted handles; all access is serialized by the mutex.
unsafe impl Send for Collector {}
static COLLECTOR: Mutex<Option<Collector>> = Mutex::new(None);

pub fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("SPX_DHESS_OUT").is_ok_and(|v| !v.is_empty()))
}

/// Record the checkpoint name of a drafter weight (the tensor the GEMM entries will receive).
pub fn name(w: &Tensor, name: &str) {
    if !enabled() { return; }
    let mut g = COLLECTOR.lock().unwrap();
    let c = g.get_or_insert_with(|| {
        let tp = crate::tp::world();
        let out = std::path::PathBuf::from(std::env::var("SPX_DHESS_OUT").unwrap()).join(format!("rank{}", tp.rank));
        std::fs::create_dir_all(&out).unwrap();
        let every = std::env::var("SPX_DHESS_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        Collector { names: HashMap::new(), acc: BTreeMap::new(), proposals: 0, skipped_capture: 0, every, out }
    });
    c.names.insert(w.data_ptr() as usize, (name.to_string(), w.size()));
}

/// Accumulate x^T x for the weight `w` (no-op unless named and collecting).
pub fn tap(x: &Tensor, w: &Tensor) {
    if !enabled() { return; }
    let mut g = COLLECTOR.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    let Some((name, _)) = c.names.get(&(w.data_ptr() as usize)).cloned() else { return };
    if crate::tp::graph::capturing() { c.skipped_capture += 1; return; }
    let d = *x.size().last().unwrap();
    let x2 = x.reshape([-1, d]).to_kind(Kind::Float);
    let e = c.acc.entry(name.clone()).or_insert_with(|| (Tensor::zeros([d, d], (Kind::Float, x2.device())), 0));
    assert_eq!(e.0.size(), [d, d], "spx-dhess: {name} input width changed");
    crate::tp::set_tf32(false);
    let _ = e.0.addmm_(&x2.transpose(0, 1), &x2);
    crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref() == Ok("1"));
    e.1 += x2.size()[0];
    if name == "layers.0.self_attn.q_proj.weight" {
        c.proposals += 1;
        if c.proposals % c.every == 0 { dump(c); }
    }
}

fn dump(c: &Collector) {
    let tp = crate::tp::world();
    let mut entries = Vec::new();
    for (name, (h, n)) in &c.acc {
        let file = format!("{name}.pt");
        let tmp = c.out.join(format!("{file}.tmp"));
        Tensor::save_multi(&[("H", &h.to_device(tch::Device::Cpu)), ("n", &Tensor::from(*n))], &tmp).unwrap();
        std::fs::rename(&tmp, c.out.join(&file)).unwrap();
        let d = h.size()[0];
        let w = c.names.values().find(|(nm, _)| nm == name).map(|(_, s)| s.clone()).unwrap_or_default();
        // Input-dimension shard of this rank: o_proj (head-parallel attention) and down_proj (MLP TP) see only their
        // contiguous slice of the input columns; every other projection sees the full input on both ranks.
        let sharded = name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight");
        let aliases: Vec<String> = if name.ends_with("k_proj.weight") { vec![name.replace("k_proj", "v_proj")] }
            else if name.ends_with("gate_proj.weight") { vec![name.replace("gate_proj", "up_proj")] } else { vec![] };
        entries.push(serde_json::json!({"name": name, "file": file, "dim": d, "n": n, "weight_shape_local": w,
            "shard": {"sharded": sharded, "rank": tp.rank, "world": tp.world, "offset": if sharded { d * tp.rank as i64 } else { 0 }},
            "aliases": aliases}));
    }
    let m = serde_json::json!({"format": "spx1-draft-hessian-v1", "rank": tp.rank, "world": tp.world, "proposals": c.proposals,
        "skipped_capture": c.skipped_capture, "hessian_math": "fp32 addmm, TF32 off", "entries": entries});
    let tmp = c.out.join("manifest.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&m).unwrap()).unwrap();
    std::fs::rename(&tmp, c.out.join("manifest.json")).unwrap();
    eprintln!("[spx-dhess] rank{} {} proposals, {} Hessians, skipped during capture {}", tp.rank, c.proposals, c.acc.len(), c.skipped_capture);
}
