//! DeepSeek-V4.1-Flash configuration (`inference/config.json` of the checkpoint: the official ModelArgs field names) and
//! the TP layout derived from it (docs/07-DeepSeek-V4.1接入与TP3.md §2).
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Cfg {
    pub vocab: i64,
    pub dim: i64,
    pub inter: i64,
    pub n_layers: usize,
    pub n_mtp: usize,
    pub n_heads: i64,
    pub head_dim: i64,
    pub rope_dim: i64,
    pub q_lora: i64,
    pub o_groups: i64,
    pub o_lora: i64,
    pub window: i64,
    pub compress: Vec<i64>,          // per layer, MTP included: 0 window only, r = r-to-1 compressed KV
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub candidate_source: i64,
    pub candidate_topk_blocks: i64,
    pub candidate_block: i64,
    pub index_heads: i64,
    pub index_dim: i64,
    pub index_topk: i64,
    pub n_experts: i64,
    pub n_active: i64,
    pub mtp_experts: i64,
    pub mtp_active: i64,
    pub route_scale: f64,
    pub swiglu_limit: f64,
    pub norm_eps: f64,
    pub hc: i64,
    pub hc_iters: i64,
    pub hc_eps: f64,
    pub rope_theta: f64,
    pub compress_rope_theta: f64,
    pub rope_factor: f64,
    pub original_seq_len: i64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub engram_layers: Vec<usize>,
    pub dspark_block: i64,
    pub dspark_noise: i64,
    pub dspark_targets: Vec<usize>,
    pub dspark_markov_rank: i64,
    pub image_token: i64,
}

fn f(v: &Value, k: &str) -> f64 { v[k].as_f64().unwrap_or_else(|| panic!("config {k}")) }
fn i(v: &Value, k: &str) -> i64 { v[k].as_i64().unwrap_or_else(|| panic!("config {k}")) }
fn list(v: &Value, k: &str) -> Vec<i64> { v[k].as_array().unwrap_or_else(|| panic!("config {k}")).iter().map(|x| x.as_i64().unwrap()).collect() }

impl Cfg {
    pub fn load(ckpt: &Path) -> Cfg {
        let v: Value = serde_json::from_reader(std::fs::File::open(ckpt.join("inference/config.json")).expect("inference/config.json")).unwrap();
        let u = |k| list(&v, k).into_iter().map(|x| x as usize).collect::<Vec<_>>();
        Cfg {
            vocab: i(&v, "vocab_size"), dim: i(&v, "dim"), inter: i(&v, "moe_inter_dim"), n_layers: i(&v, "n_layers") as usize,
            n_mtp: i(&v, "n_mtp_layers") as usize, n_heads: i(&v, "n_heads"), head_dim: i(&v, "head_dim"), rope_dim: i(&v, "rope_head_dim"),
            q_lora: i(&v, "q_lora_rank"), o_groups: i(&v, "o_groups"), o_lora: i(&v, "o_lora_rank"), window: i(&v, "window_size"),
            compress: list(&v, "compress_ratios"), kv_sources: u("kv_source_layers"), index_sources: u("index_source_layers"),
            candidate_source: i(&v, "candidate_source_layer"), candidate_topk_blocks: i(&v, "candidate_topk_blocks"),
            candidate_block: i(&v, "candidate_block_size"), index_heads: i(&v, "index_n_heads"), index_dim: i(&v, "index_head_dim"),
            index_topk: i(&v, "index_topk"), n_experts: i(&v, "n_routed_experts"), n_active: i(&v, "n_activated_experts"),
            mtp_experts: i(&v, "dspark_n_routed_experts"), mtp_active: i(&v, "dspark_n_activated_experts"),
            route_scale: f(&v, "route_scale"), swiglu_limit: f(&v, "swiglu_limit"), norm_eps: f(&v, "norm_eps"),
            hc: i(&v, "hc_mult"), hc_iters: i(&v, "hc_sinkhorn_iters"), hc_eps: f(&v, "hc_eps"), rope_theta: f(&v, "rope_theta"),
            compress_rope_theta: f(&v, "compress_rope_theta"), rope_factor: f(&v, "rope_factor"),
            original_seq_len: i(&v, "original_seq_len"), beta_fast: f(&v, "beta_fast"), beta_slow: f(&v, "beta_slow"),
            engram_layers: u("engram_layer_ids"), dspark_block: i(&v, "dspark_block_size"), dspark_noise: i(&v, "dspark_noise_token_id"),
            dspark_targets: u("dspark_target_layer_ids"), dspark_markov_rank: i(&v, "dspark_markov_rank"), image_token: i(&v, "image_token_id"),
        }
    }
    /// The KV source a layer reads its compressed KV from: the latest source at or below it (backbone layers only).
    pub fn kv_source_of(&self, layer: usize) -> Option<usize> {
        if self.compress[layer] == 0 || layer >= self.n_layers { return None; }
        self.kv_sources.iter().copied().filter(|&s| s <= layer).max()
    }
    pub fn is_kv_source(&self, layer: usize) -> bool { layer < self.n_layers && self.kv_sources.contains(&layer) }
    pub fn is_index_source(&self, layer: usize) -> bool { layer < self.n_layers && self.index_sources.contains(&layer) }
}

/// This rank's share of the TP-split dimensions (docs/07 §2): attention by output group, expert and shared-expert
/// intermediate dimension in equal 128-aligned parts, vocabulary in 128-row blocks.
#[derive(Clone, Copy, Debug)]
pub struct Split { pub rank: usize, pub world: usize, pub g0: i64, pub g1: i64, pub i0: i64, pub i1: i64, pub v0: i64, pub v1: i64 }

/// Contiguous share [start, end) of `n` units over `world` ranks, the first `n % world` ranks one larger.
pub fn share(n: i64, world: usize, rank: usize) -> (i64, i64) {
    let (w, r) = (world as i64, rank as i64);
    let base = n / w;
    let extra = n % w;
    let start = r * base + r.min(extra);
    (start, start + base + i64::from(r < extra))
}

impl Split {
    pub fn new(c: &Cfg, rank: usize, world: usize) -> Split {
        let (g0, g1) = share(c.o_groups, world, rank);
        assert_eq!(c.inter % (128 * world as i64), 0, "intermediate {} not a multiple of 128 x world", c.inter);
        let (i0, i1) = share(c.inter, world, rank);
        let blocks = (c.vocab + 127) / 128;
        let (b0, b1) = share(blocks, world, rank);
        Split { rank, world, g0, g1, i0, i1, v0: b0 * 128, v1: (b1 * 128).min(c.vocab) }
    }
    pub fn heads_per_group(&self, c: &Cfg) -> i64 { c.n_heads / c.o_groups }
    pub fn h0(&self, c: &Cfg) -> i64 { self.g0 * self.heads_per_group(c) }
    pub fn h1(&self, c: &Cfg) -> i64 { self.g1 * self.heads_per_group(c) }
    pub fn local_heads(&self, c: &Cfg) -> i64 { self.h1(c) - self.h0(c) }
}
