//! DSpark speculative decoding for DeepSeek-V4.1-Flash (docs/07 D6). The official repository implements only the draft
//! forward (`Transformer.forward_spec`, `DSparkBlock`, `DSparkAttention`); the verification loop is ours:
//!
//! - Draft: a sequence whose next token `tok` (position P = seq.pos) is known runs rows [tok, noise x (block - 1)] at
//!   positions P .. P + block - 1 through the three DSpark stages (ordinary blocks: window-only attention, 128-expert
//!   top-3 MoE). Their attention sees the stage's own window KV of the main model's positions [P - window, P) (seeded
//!   from the main model's target hidden states, `mtp_seed`) and all rows of the block (no causal mask inside the
//!   block). The shared head plus the Markov bias (low-rank embedding of the previous token, sequential) give the draft
//!   tokens d1 .. d_block for positions P + 1 .. P + block, and a confidence per draft.
//! - Verify: the main model runs [tok, d1 .. d_block] as one chunk (committing it), samples its own token at every row
//!   with the same per-position Gumbel key as plain decoding, and accepts the drafts while they equal those samples:
//!   with k accepted, tok and d1 .. dk stay committed, the state rolls back to P + k + 1 (`spec_rollback`: window
//!   ring slots, the compressor's open group, n-gram history, position), and the main sample at row k is the next tok.
//!   The emitted tokens are exactly the ones plain decoding with the same keys samples (the drafts only decide how many
//!   rows are verified at once); drafts sample with the same keys too, which couples them to the target.
//! - The accepted rows' target hidden states seed the stages' window KV for the next draft.
//! - DS41_SPEC_RS=1 (default off) for sampled sequences (T > 0): standard speculative sampling (Leviathan et al. 2023,
//!   Chen et al. 2023) instead of the key coupling. Drafts sample from q = softmax(draft logits / T) with their own
//!   keys (salted by seed and the round's position); the target accepts draft d at a row with probability
//!   min(1, p(d) / q(d)) (p = softmax(target logits / T), uniform of key(seed ^ SALT_U, position)), else samples
//!   max(p - q, 0) (Gumbel keys of seed ^ SALT_R) and stops; all accepted: the next row samples p as plain decoding
//!   does. Every emitted token is distributed exactly as plain sampling's (exact in exact arithmetic; the q used is the
//!   very logits the draft sampled from), not the same tokens for a seed. Its acceptance is the optimal 1 - TV(p, q) per
//!   draft, against the coupling's P(both Gumbel argmaxes agree). Greedy (T = 0) is unchanged.
use super::cfg::{Cfg, Split};
use super::load::Checkpoint;
use super::model::{attn_fused, graph_on, rmsn, rot, Fp8W, Layer, Model, Seq};
use super::ops;
use tch::{Device, Kind, Tensor};

pub struct Mtp {
    pub layers: Vec<Layer>,
    main_proj: Fp8W,
    main_norm: Tensor,
    norm: Tensor,
    markov_embed: Tensor,   // [vocab, rank] BF16, every row (looked up by token)
    markov_head: Tensor,    // [v1 - v0, rank] BF16, this rank's vocabulary rows
    conf: Tensor,           // [dim + rank] FP32
}

impl Mtp {
    pub(super) fn markov_head(&self) -> &Tensor { &self.markov_head }
    pub fn load(ck: &Checkpoint, c: &Cfg, sp: &Split, fp4: &Tensor, dev: Device) -> Mtp {
        let layers = (0..c.n_mtp).map(|i| {
            let mut l = Model::load_block(ck, c, sp, &format!("mtp.{i}"), c.n_layers + i, c.mtp_experts, c.mtp_active, fp4);
            Model::pack_experts(&mut l, dev);
            l
        }).collect();
        let last = c.n_mtp - 1;
        Mtp {
            layers,
            main_proj: Model::fp8(ck, "mtp.0.main_proj"),
            main_norm: ck.get("mtp.0.main_norm.weight", Kind::BFloat16),
            norm: ck.get(&format!("mtp.{last}.norm.weight"), Kind::BFloat16),
            markov_embed: ck.get(&format!("mtp.{last}.markov_head.embed.weight"), Kind::BFloat16),
            markov_head: ck.rows(&format!("mtp.{last}.markov_head.head.weight"), sp.v0, sp.v1 - sp.v0, Kind::BFloat16),
            conf: ck.get(&format!("mtp.{last}.confidence_head.proj.weight"), Kind::Float).view(-1),
        }
    }
}

/// The draft body (stages + head) as CUDA graphs for B sequences: per stage S1 (mHC mix/pre, norm, Q / KV) and S3
/// (output projection, mHC, MoE, mHC), the window attention per sequence in between (eager), the final mHC pre + norm
/// + this rank's head rows.
pub struct MtpGraph {
    pos: Tensor, h0: Tensor,
    s1: Vec<crate::tp::graph::Owned>, s3: Vec<crate::tp::graph::Owned>, fin: crate::tp::graph::Owned,
    mid: Vec<[Tensor; 2]>, obuf: Vec<Tensor>, xf: Tensor, logits: Tensor, _keep: Vec<Tensor>,
    mk: Option<MarkovGraph>,
}

/// The sequential part of a draft (DS41_DRAFT_GRAPH, default on) as one graph per B on top of the body graph's head
/// logits: for each of the `block` drafts the Markov bias of the previous token, the Gumbel top-1 of this rank's
/// vocabulary slice (rs_gumbel_argmax_packet, bitwise the eager noise + max_dim), the cross-rank choice
/// (head_select::choose) and the confidence. The tokens stay on the device from one step to the next, so a draft costs
/// one parameter upload and one read-back instead of a host round trip (allreduce read + confidence read) per draft.
static DRAFT_CHECK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DRAFT_CHECK_BAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub struct MarkovGraph { prev: Tensor, temps: Tensor, keys: Tensor, out: Tensor, qloc: Option<Tensor>, g: crate::tp::graph::Owned, _keep: Vec<Tensor> }

/// DS41_SPEC_RS=1: rejection-sampling verification for sampled sequences (module doc). Default off: measured 2026-10-28
/// on the service (open texts and thinking at T = 1, DS41_RS_STATS on the same rows) its expected acceptance 1 - TV is
/// only 0.615 against the key coupling's 0.603 (DSpark's q is close to p in shape), and the service speed was flat
/// (45.84 vs 45.78 tok/s, the q gather and verdict eat the +1.2% tokens per round); off keeps T > 0 output a function
/// of the seed alone (the same as plain decoding), bitwise as before.
pub fn rs_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_SPEC_RS").is_ok_and(|v| v == "1")) }
const SALT_D: u64 = 0x6a09_e667_f3bc_c908;
const SALT_R: u64 = 0xbb67_ae85_84ca_a73b;
const SALT_U: u64 = 0x3c6e_f372_fe94_f82b;
/// The seed of a draft's Gumbel keys: the plain one (coupled to the target) unless rejection sampling verifies it, then
/// salted and fresh for every round (p0: the round's pending position).
fn draft_seed(seed: u64, temp: f32, p0: i64) -> u64 {
    if rs_on() && temp > 0.0 { seed ^ SALT_D ^ (p0 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) } else { seed }
}
/// DS41_DRAFT_TSCALE (key coupling only, T > 0): the drafts sample at temperature T x this. With the shared keys the
/// draft is argmax(draft logits / (T s) + G) against the target's argmax(logits / T + G): a sharper draft distribution
/// matches the target's sample more often. Only which drafts are proposed changes; every emitted token is the target's
/// own keyed sample, bitwise the same for a seed. Measured 2026-10-28 (DS41_RS_STATS_X, open texts + thinking at T = 1,
/// 3455 rows): expected acceptance 0.609 at 1.0, 0.619 at 0.6 / 0.7, 0.618 at 0.8, 0.601 at 1.1 (1 - TV: 0.622). Service
/// A/B with fixed seeds (same 15 texts byte for byte in every arm): tokens per round 1.972 / 1.984 -> 2.013 / 2.022,
/// 44.3 -> 45.5 tok/s. Default 0.7 (1 = the earlier behaviour).
fn draft_tscale() -> f32 {
    static E: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("DS41_DRAFT_TSCALE").ok().and_then(|v| v.parse::<f32>().ok()).filter(|&x| x > 0.0).unwrap_or(0.7))
}
/// A uniform in [0, 1) from a key (splitmix64 finalizer, 53 bits).
fn unif(k: u64) -> f64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn draft_graph_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_DRAFT_GRAPH").map_or(true, |v| v != "0")) }

/// How many drafts to verify: the longest prefix whose estimated probability of being accepted entirely (product of
/// sigmoid(confidence)) stays at or above DS41_SPEC_MIN_P (0.15), at most DS41_SPEC_D (all).
pub fn spec_len(conf: &[f32], max: usize) -> usize {
    let min_p = std::env::var("DS41_SPEC_MIN_P").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.15);
    let cap = std::env::var("DS41_SPEC_D").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(max).min(max);
    let mut p = 1f32;
    let mut d = 0;
    while d < cap {
        p *= 1.0 / (1.0 + (-conf[d]).exp());
        if p < min_p { break; }
        d += 1;
    }
    d
}

/// What a verify changed, to roll back to any accepted length.
pub struct Rollback { pos0: i64, n: i64, ring: Vec<Tensor>, saved: Option<Tensor> }
/// DS41_RING_FUSED (default on; 0: an index_select / index_copy per layer): the window rings' checkpoint and rollback of
/// a speculative round in one launch each (kern::ring_save / ring_restore; the same bytes).
fn ring_fused_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_RING_FUSED").map_or(true, |v| v != "0")) }

/// FP32 rows [m, n] = x [m, k] . W^T for BF16 W (the BF16 GEMV in chunks of 16 rows).
fn head_rows(x: &Tensor, w: &Tensor) -> Tensor {
    let m = x.size()[0];
    let parts: Vec<Tensor> = (0..m).step_by(16).map(|r| super::kern::bf16_gemv(&x.narrow(0, r, 16.min(m - r)), w)).collect();
    Tensor::cat(&parts, 0)
}
impl Model {
    /// The draft rows' head logits: the FP8 copy when present (act-quantized input, 16 rows per GEMV), else BF16.
    fn draft_head(&self, x: &Tensor) -> Tensor {
        let Some(h8) = &self.head8 else { return head_rows(x, &self.head) };
        let m = x.size()[0];
        let parts: Vec<Tensor> = (0..m).step_by(16).map(|r| super::kern::fp8_gemv_mx(&x.narrow(0, r, 16.min(m - r)), &h8.w, &h8.s)).collect();
        Tensor::cat(&parts, 0)
    }
}

impl Model {
    /// Samples one token per row from this rank's vocabulary slice of the logits [B, v1 - v0] (the same rule as the
    /// serve loop: temperature 0 = argmax, else argmax of logits + temperature-scaled Gumbel noise of key(seed, pos)),
    /// combined across ranks (largest value, lowest vocabulary id on ties: the global argmax).
    /// One token per row of `local` (this rank's vocabulary slice [R, v1 - v0], e.g. Model::forward_rows_local), the
    /// row's temperature and key: the same token as serve::sample on the full row (argmax of logits + T*G(key, column),
    /// the noise is per global column; ties to the lowest index), from (value, index) packets of every rank instead of
    /// the full rows. Every rank calls it (a collective); one copy to the host.
    pub fn sample_rows(&self, local: &Tensor, temps: &[f32], keys: &[u64]) -> Vec<i64> {
        let r = local.size()[0] as usize;
        assert!(temps.len() == r && keys.len() == r);
        (0..r).step_by(64).flat_map(|a| { let n = 64.min(r - a); self.sample_split(&local.narrow(0, a as i64, n as i64), &temps[a..a + n], &keys[a..a + n]) }).collect()
    }
    fn sample_split(&self, lg: &Tensor, temps: &[f32], keys: &[u64]) -> Vec<i64> {
        extern "C" { fn rs_gumbel_noise(out: *mut f32, stride: i64, rows: i32, cols: i32, col_offset: i64, temps: *const f32, keys: *const u64) -> i32; }
        let b = lg.size()[0];
        let mut l = lg.to_kind(Kind::Float).contiguous();
        if temps.iter().any(|&t| t > 0.0) {
            let noise = Tensor::zeros_like(&l);
            assert_eq!(unsafe { rs_gumbel_noise(noise.data_ptr().cast(), noise.stride()[0], b as i32, l.size()[1] as i32, self.sp.v0, temps.as_ptr(), keys.as_ptr()) }, 0);
            l = l + noise;
        }
        let (v, i) = l.max_dim(-1, false);
        let world = self.sp.world as i64;
        let buf = Tensor::zeros([world, b, 2], (Kind::Float, self.dev));
        let _ = buf.get(self.sp.rank as i64).copy_(&Tensor::stack(&[v, (i + self.sp.v0).to_kind(Kind::Float)], 1));
        if world > 1 { crate::tp::allreduce(&buf); }
        let host: Vec<f32> = Vec::<f32>::try_from(buf.to_device(Device::Cpu).view(-1)).unwrap();
        (0..b as usize).map(|r| {
            let mut best = (f32::NEG_INFINITY, 0i64);
            for k in 0..world as usize {
                let (val, id) = (host[(k * b as usize + r) * 2], host[(k * b as usize + r) * 2 + 1] as i64);
                if val > best.0 { best = (val, id); }
            }
            best.1
        }).collect()
    }

    /// main_proj.mm(hidden) with the output rows split over the ranks (DS41_MTP_SEED_SPLIT, default on; 0: every rank
    /// all of them): rank r computes 32-row blocks [r B / w, (r + 1) B / w) (each output row's GEMV is the same alone:
    /// bitwise) and one all-gather assembles them -- a third of the 79 MB weight read per round at TP3.
    fn mtp_main_proj(&self, w: &super::model::Fp8W, hidden: &Tensor) -> Tensor {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let (world, rank) = (self.sp.world as i64, self.sp.rank as i64);
        let n = hidden.size()[0];
        if !*ON.get_or_init(|| std::env::var("DS41_MTP_SEED_SPLIT").map_or(true, |v| v != "0")) || world < 2 || n > 16 || crate::tp::graph::capturing() {
            return w.mm(hidden);
        }
        let blocks = w.n() / 32;
        let b0 = |r: i64| blocks * r / world;
        let bmax = (0..world).map(|r| b0(r + 1) - b0(r)).max().unwrap();
        let mine = w.rows(b0(rank) * 32, (b0(rank + 1) - b0(rank)) * 32).mm(hidden);           // [n, my rows] BF16
        let pad = Tensor::zeros([n, bmax * 32], (Kind::BFloat16, self.dev));
        let _ = pad.narrow(1, 0, mine.size()[1]).copy_(&mine);
        let all = super::model::sp_ag_pub(&pad, world * n).view([world, n, bmax * 32]);
        Tensor::cat(&(0..world).map(|r| all.select(0, r).narrow(1, 0, (b0(r + 1) - b0(r)) * 32)).collect::<Vec<_>>(), 1)
    }
    /// The DSpark stages' window KV for main positions [pos0, pos0 + n) from their target hidden rows [n, 3 dim].
    pub fn mtp_seed(&self, seq: &mut Seq, hidden: &Tensor, pos0: i64) {
        let Some(mt) = &self.mtp else { return };
        let c = &self.c;
        let n = hidden.size()[0];
        let keep = n.min(c.window);
        let (hidden, pos0) = (hidden.narrow(0, n - keep, keep), pos0 + n - keep);
        let pos = Tensor::arange_start(pos0, pos0 + keep, (Kind::Int64, self.dev));
        let main_x = rmsn(&self.mtp_main_proj(&mt.main_proj, &hidden), &mt.main_norm, c.norm_eps);
        for (i, layer) in mt.layers.iter().enumerate() {
            let a = &layer.attn;
            let kv = super::kern::act_quant(&rot(&rmsn(&a.wkv.mm(&main_x), &a.kv_norm, c.norm_eps), &a.rope, &pos, c.rope_dim, false));
            let _ = seq.mtp_win[i].index_copy_(0, &pos.remainder(c.window), &kv);
        }
    }

    /// Drafts for each sequence (toks[b] at position seqs[b].pos): (draft tokens, confidences), `block` each.
    pub fn mtp_draft(&self, seqs: &[&mut Seq], toks: &[i64], temps: &[f32], seeds: &[u64]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
        let _g = tch::no_grad_guard();
        let scaled: Vec<f32> = temps.iter().map(|&t| if rs_on() { t } else { t * draft_tscale() }).collect();
        let temps = &scaled[..];
        assert!(self.mtp.is_some(), "DSpark stages not loaded (DS41_MTP=1)");
        let c = &self.c;
        let dev = self.dev;
        let blk = c.dspark_block;
        let mut ids = Vec::new();
        let mut posv = Vec::new();
        for (s, &t) in seqs.iter().zip(toks) {
            ids.push(t);
            ids.extend(std::iter::repeat(c.dspark_noise).take(blk as usize - 1));
            posv.extend(s.pos..s.pos + blk);
        }
        let pos = super::kern::upload_i64(&posv, dev);
        let e = self.embed(&super::kern::upload_i64(&ids, dev));                        // [R, dim]
        if graph_on() && draft_graph_on() && self.draft_debug.borrow().is_none() {
            let out = self.mtp_draft_graph(seqs, &e, &pos, toks, temps, seeds);
            if std::env::var("DS41_DRAFT_GRAPH_CHECK").as_deref() == Ok("1") {
                // the eager loop on the same head logits (the body graph's outputs, untouched by the Markov graph)
                let (xf, logits) = {
                    let g = self.mtp_graphs.borrow();
                    let g = &g[&seqs.len()];
                    (g.xf.copy(), g.logits.copy())
                };
                let (d, cf) = self.markov_eager(seqs, &xf, &logits, toks, temps, seeds);
                let bad_d = d.iter().zip(&out.0).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count()).sum::<usize>();
                let bad_c = cf.iter().zip(&out.1).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()).sum::<usize>();
                let n = DRAFT_CHECK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let bad = DRAFT_CHECK_BAD.fetch_add((bad_d + bad_c > 0) as usize, std::sync::atomic::Ordering::Relaxed) + (bad_d + bad_c > 0) as usize;
                if bad_d + bad_c > 0 || n % 200 == 0 {
                    eprintln!("[ds41-draft-check] round {n} B {}: {bad_d} draft / {bad_c} confidence mismatches (rounds with any: {bad})", seqs.len());
                }
            }
            return out;
        }
        let (xf, logits) = if graph_on() { self.mtp_body_graph(seqs, &e, &pos, true) } else { self.mtp_body(seqs, &e, &pos, None) };
        self.markov_eager(seqs, &xf, &logits, toks, temps, seeds)
    }

    /// The Markov / sampling / confidence steps of a draft with a host round trip per draft (DS41_DRAFT_GRAPH=0, the
    /// draft probe's per-step logits, DS41_DRAFT_GRAPH_CHECK's reference).
    fn markov_eager(&self, seqs: &[&mut Seq], xf: &Tensor, logits: &Tensor, toks: &[i64], temps: &[f32], seeds: &[u64]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
        let mt = self.mtp.as_ref().unwrap();
        let c = &self.c;
        let dev = self.dev;
        let (bsz, blk) = (seqs.len(), c.dspark_block);
        let mut drafts = vec![Vec::new(); bsz];
        let mut confs = vec![Vec::new(); bsz];
        let mut prev: Vec<i64> = toks.to_vec();
        let conf_x = mt.conf.narrow(0, 0, c.dim);
        let conf_m = mt.conf.narrow(0, c.dim, c.dspark_markov_rank);
        let mut qs = Vec::new();
        for j in 0..blk {
            let emb = mt.markov_embed.index_select(0, &super::kern::upload_i64(&prev, dev));     // [B, rank]
            let bias = super::kern::bf16_gemv(&emb, &mt.markov_head);                                 // [B, v1 - v0]
            let rows = super::kern::upload_i64(&(0..bsz as i64).map(|b| b * blk + j).collect::<Vec<_>>(), dev);
            let lg = logits.index_select(0, &rows) + bias;
            if let Some(v) = self.draft_debug.borrow_mut().as_mut() {
                let full = Tensor::zeros([bsz as i64, c.vocab], (Kind::Float, dev));
                full.narrow(1, self.sp.v0, self.sp.v1 - self.sp.v0).copy_(&lg);
                if self.sp.world > 1 { crate::tp::allreduce(&full); }
                v.push(full);
            }
            let keys: Vec<u64> = seqs.iter().enumerate().map(|(b, s)| crate::sampling::key(draft_seed(seeds[b], temps[b], s.pos), (s.pos + j + 1) as u64)).collect();
            let next = self.sample_split(&lg, temps, &keys);
            qs.push(lg.to_kind(Kind::Float));
            let cf = (xf.index_select(0, &rows).to_kind(Kind::Float).matmul(&conf_x) + emb.to_kind(Kind::Float).matmul(&conf_m)).to_device(Device::Cpu);
            let cf: Vec<f32> = Vec::<f32>::try_from(cf).unwrap();
            for b in 0..bsz { drafts[b].push(next[b]); confs[b].push(cf[b]); }
            prev = next;
        }
        self.keep_draft_q(&Tensor::stack(&qs, 0), temps);
        (drafts, confs)
    }

    /// With rejection sampling for any sampled sequence: every rank's draft logits [block, B, v1 - v0] into the whole
    /// vocabulary (draft_q, for rs_tokens); else draft_q is cleared.
    fn keep_draft_q(&self, qloc: &Tensor, temps: &[f32]) {
        if !(((rs_on() || stats_on()) && temps.iter().any(|&t| t > 0.0)) || ranks_on()) { *self.draft_q.borrow_mut() = None; return; }
        let (blk, b) = (qloc.size()[0], qloc.size()[1]);
        let q = if self.sp.world == 1 { qloc.copy() } else {
            let full = Tensor::zeros([blk, b, self.c.vocab], (Kind::Float, self.dev));
            full.narrow(2, self.sp.v0, self.sp.v1 - self.sp.v0).copy_(qloc);
            crate::tp::allreduce(&full);
            full
        };
        *self.draft_q.borrow_mut() = Some(q);
    }

    /// The verify rows' tokens of one sequence under rejection sampling: lg [d + 1, vocab] its target logits (rows of
    /// the pending token and the d drafts), q [d, vocab] the drafts' logits (draft_q), p0 the pending position. Row j < d
    /// gives drafts[j] if accepted, else a sample of max(p - q, 0); row d the plain sample. The caller accepts drafts
    /// while they equal these (an accepted draft is returned as itself, a rejected row never returns its draft).
    pub fn rs_tokens(&self, lg: &Tensor, q: &Tensor, drafts: &[i64], temp: f32, seed: u64, p0: i64) -> Vec<i64> { rs_tokens(lg, q, drafts, temp, seed, p0) }
}
/// Model::rs_tokens (any device; ds41-kernel-test DS41_KT_ONLY=rs checks its output distribution)
pub fn rs_tokens(lg: &Tensor, q: &Tensor, drafts: &[i64], temp: f32, seed: u64, p0: i64) -> Vec<i64> {
        extern "C" { fn rs_gumbel_noise(out: *mut f32, stride: i64, rows: i32, cols: i32, col_offset: i64, temps: *const f32, keys: *const u64) -> i32; }
        let d = drafts.len() as i64;
        let plain_row = |j: i64| super::serve::sample(&lg.narrow(0, j, 1), temp, seed, p0 + j + 1);
        if d == 0 { return vec![plain_row(0)]; }
        let t = temp as f64;
        let lp = (lg.narrow(0, 0, d).to_kind(Kind::Float) / t).log_softmax(-1, Kind::Float);
        let lq = (q.narrow(0, 0, d).to_kind(Kind::Float) / t).log_softmax(-1, Kind::Float);
        let di = super::kern::upload_i64(drafts, lg.device()).view([d, 1]);
        let lr = (lp.gather(1, &di, false) - lq.gather(1, &di, false)).view([d]);   // log p(d) / q(d)
        let r = (lp.exp() - lq.exp()).clamp_min(0.0);
        let noise = Tensor::zeros_like(&r);
        let temps = vec![1f32; d as usize];
        let keys: Vec<u64> = (0..d).map(|j| crate::sampling::key(seed ^ SALT_R, (p0 + j + 1) as u64)).collect();
        assert_eq!(unsafe { rs_gumbel_noise(noise.data_ptr().cast(), noise.stride()[0], d as i32, r.size()[1] as i32, 0, temps.as_ptr(), keys.as_ptr()) }, 0);
        let (_, res) = (r.log() + noise).max_dim(-1, false);
        let host: Vec<f32> = Vec::<f32>::try_from(Tensor::stack(&[lr, res.to_kind(Kind::Float), r.amax(&[-1i64][..], false)], 0).to_device(Device::Cpu).view(-1)).unwrap();
        let mut ys = Vec::with_capacity(d as usize + 1);
        for j in 0..d as usize {
            let u = unif(crate::sampling::key(seed ^ SALT_U, p0 as u64 + j as u64 + 1));
            let (lrj, resj, rmax) = (host[j] as f64, host[d as usize + j] as i64, host[2 * d as usize + j]);
            // (u = 0 accepts; a rejection with max(p - q, 0) all zero -- rounding when p ~ q -- samples p instead)
            let y = if u.ln() < lrj { drafts[j] } else if rmax > 0.0 { resj } else { plain_row(j as i64) };
            ys.push(y);
            if y != drafts[j] { break; }
        }
        if ys.len() == d as usize && *ys.last().unwrap() == drafts[d as usize - 1] { ys.push(plain_row(d)); }
        ys
}
/// DS41_SPEC_RANKS=<file> (measurement, rank 0): per verified draft position j (1-based), one line
/// "ts temp j acc rank p_y": ts unix ms, acc 1 / 0 / -1 (accepted / the first rejected / after it), rank the place of the
/// target's token y_j among the draft's scores at that position under the same keys (draft logits as sampled + T * G;
/// T = 0: the draft logits; 1 = the draft's own token, capped at 64), p_y the target probability of y_j at temperature
/// max(T, 1). With it the drafts' logits are kept at every temperature and the full verify logits assembled (slower).
/// What a position with K draft candidates instead of one would accept: rank <= K.
pub fn ranks_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_SPEC_RANKS").is_ok()) }
/// (DS41_SPEC_RANKS) lg [d + 1, vocab] the target logits of one sequence's rows, q [>= d, vocab] its drafts' logits,
/// ys the target tokens of positions 1..=d (ys[j - 1] from row j - 1), k accepted.
pub fn spec_ranks(lg: &Tensor, q: &Tensor, drafts: &[i64], ys: &[i64], temp: f32, dseed: u64, p0: i64, k: usize) {
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(f) = F.get_or_init(|| std::env::var("DS41_SPEC_RANKS").ok().map(|p| std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).expect("DS41_SPEC_RANKS")))) else { return };
    extern "C" { fn rs_gumbel_noise(out: *mut f32, stride: i64, rows: i32, cols: i32, col_offset: i64, temps: *const f32, keys: *const u64) -> i32; }
    let d = drafts.len() as i64;
    if d == 0 { return; }
    let qd = q.narrow(0, 0, d).to_kind(Kind::Float).contiguous();
    let noise = Tensor::zeros_like(&qd);
    if temp > 0.0 {
        let temps = vec![temp; d as usize];
        let keys: Vec<u64> = (0..d).map(|j| crate::sampling::key(dseed, (p0 + j + 1) as u64)).collect();
        assert_eq!(unsafe { rs_gumbel_noise(noise.data_ptr().cast(), noise.stride()[0], d as i32, qd.size()[1] as i32, 0, temps.as_ptr(), keys.as_ptr()) }, 0);
    }
    let sc = &qd + &noise;
    let yt = super::kern::upload_i64(&ys[..d as usize], qd.device()).view([d, 1]);
    let sy = sc.gather(1, &yt, false);
    let rank = (sc.gt_tensor(&sy).sum_dim_intlist(&[1i64][..], false, Kind::Int64) + 1).clamp_max(64);
    let py = (lg.narrow(0, 0, d).to_kind(Kind::Float) / (temp.max(1.0) as f64)).softmax(-1, Kind::Float).gather(1, &yt, false).view([d]);
    let h: Vec<f64> = Vec::<f64>::try_from(Tensor::stack(&[rank.to_kind(Kind::Double), py.to_kind(Kind::Double)], 0).to_device(Device::Cpu).view(-1)).unwrap();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |x| x.as_millis());
    let mut f = f.lock().unwrap();
    use std::io::Write;
    for j in 0..d as usize {
        let acc = if j < k { 1 } else if j == k { 0 } else { -1 };
        let _ = writeln!(f, "{ts} {temp} {} {acc} {} {:.6}", j + 1, h[j] as i64, h[d as usize + j]);
    }
}
pub fn stats_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_RS_STATS").is_ok()) }
/// DS41_RS_STATS=<file> (with or without DS41_SPEC_RS; T > 0 rounds): per draft row of a verify, one line
/// "rs j 1-TV coupling acc check": rs whether rejection sampling verified it; 1-TV = sum min(p, q) the expected acceptance
/// of rejection sampling, coupling = P(argmax(log p + G) = argmax(log q + G)) = sum_i 1 / sum_j max(p_j / p_i, q_j / q_i)
/// that of the key coupling (both on this row's target p = softmax(lg / T) and draft q = softmax(draft logits / T); the
/// coupling over the 512 tokens of the largest p + q); acc 1 / 0 the row's draft accepted / rejected, -1 not reached;
/// check 1 if the draft token is what the gathered q samples with the draft's own keys (q is the distribution the draft
/// sampled from), else 0. lg [d + 1, vocab], q [d, vocab], k the accepted count, dseed the draft keys' seed.
pub fn spec_stats(lg: &Tensor, q: &Tensor, drafts: &[i64], temp: f32, dseed: u64, p0: i64, k: usize) {
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(f) = F.get_or_init(|| std::env::var("DS41_RS_STATS").ok().map(|p| std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).expect("DS41_RS_STATS")))) else { return };
    let d = drafts.len() as i64;
    if d == 0 || temp <= 0.0 { return; }
    let t = temp as f64;
    let p = (lg.narrow(0, 0, d).to_kind(Kind::Float) / t).softmax(-1, Kind::Double);
    let qq = (q.narrow(0, 0, d).to_kind(Kind::Float) / t).softmax(-1, Kind::Double);
    let ovl = p.minimum(&qq).sum_dim_intlist(&[-1i64][..], false, Kind::Double);
    let (_, idx) = (&p + &qq).topk(512, -1, true, false);
    let (pk, qk) = (p.gather(1, &idx, false).clamp_min(1e-300), qq.gather(1, &idx, false).clamp_min(1e-300));
    // m[r, i, j] = max(p_j / p_i, q_j / q_i)
    let m = (pk.unsqueeze(1) / pk.unsqueeze(2)).maximum(&(qk.unsqueeze(1) / qk.unsqueeze(2)));
    let cpl = m.sum_dim_intlist(&[2i64][..], false, Kind::Double).reciprocal().sum_dim_intlist(&[1i64][..], false, Kind::Double);
    // DS41_RS_STATS_X=1: also the coupling with a tempered draft q^(1 / a) (renormalized over the 512 tokens) for a in
    // DS41_RS_ALPHAS (0.6,0.7,0.8,0.9,1.1), and the list coupling: P(the target's Gumbel argmax is among the draft's
    // top 2 / top 3 under the same noise), Monte Carlo over 256 shared Gumbel draws on the 512 tokens
    let mut cols = vec![ovl, cpl];
    if std::env::var("DS41_RS_STATS_X").as_deref() == Ok("1") {
        let alphas: Vec<f64> = std::env::var("DS41_RS_ALPHAS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![0.6, 0.7, 0.8, 0.9, 1.1]);
        let pkn = &pk / pk.sum_dim_intlist(&[-1i64][..], true, Kind::Double);
        for &a in &alphas {
            let qa = qk.log() / a;
            let qa = (&qa - qa.logsumexp(&[-1i64][..], true)).exp().clamp_min(1e-300);
            let m = (pkn.unsqueeze(1) / pkn.unsqueeze(2)).maximum(&(qa.unsqueeze(1) / qa.unsqueeze(2)));
            cols.push(m.sum_dim_intlist(&[2i64][..], false, Kind::Double).reciprocal().sum_dim_intlist(&[1i64][..], false, Kind::Double));
        }
        let g = -(-Tensor::rand([d, 256, 512], (Kind::Double, lg.device())).clamp_min(1e-300).log()).log();   // Gumbel
        let tp = (pk.log().unsqueeze(1) + &g).argmax(-1, false);                                             // [d, 256]
        let (_, tq) = (qk.log().unsqueeze(1) + &g).topk(3, -1, true, true);                                   // [d, 256, 3]
        let hit = |k: i64| tq.narrow(2, 0, k).eq_tensor(&tp.unsqueeze(2)).any_dim(2, false).to_kind(Kind::Double).mean_dim(&[1i64][..], false, Kind::Double);
        cols.push(hit(1)); cols.push(hit(2)); cols.push(hit(3));
    }
    let nc = cols.len();
    let h: Vec<f64> = Vec::<f64>::try_from(Tensor::stack(&cols, 1).to_device(Device::Cpu).view(-1)).unwrap();
    let mut f = f.lock().unwrap();
    use std::io::Write;
    for j in 0..d as usize {
        let chk = super::serve::sample(&q.narrow(0, j as i64, 1), temp, dseed, p0 + j as i64 + 1) == drafts[j];
        let acc = if j < k { 1 } else if j == k { 0 } else { -1 };
        let extra: String = (2..nc).map(|c| format!(" {:.6}", h[nc * j + c])).collect();
        let _ = writeln!(f, "{} {j} {:.6} {:.6} {acc} {}{extra}", rs_on() as i32, h[nc * j], h[nc * j + 1], chk as i32);
    }
}
/// (the draft key seed under rejection sampling, for ds41-kernel-test)
pub fn rs_draft_seed(seed: u64, temp: f32, p0: i64) -> u64 { draft_seed(seed, temp, p0) }
impl Model {

    /// The drafts through the body graph and the Markov graph (MarkovGraph): same tokens and confidences as the eager
    /// loop of mtp_draft.
    fn mtp_draft_graph(&self, seqs: &[&mut Seq], e: &Tensor, pos: &Tensor, toks: &[i64], temps: &[f32], seeds: &[u64]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
        use crate::tp::graph::{begin_pool, end, Owned};
        let (bsz, blk, dev) = (seqs.len(), self.c.dspark_block as usize, self.dev);
        assert!(bsz <= 64, "the Gumbel parameter rows hold 64 sequences");
        let mut tv = vec![0f32; 64];
        tv[..bsz].copy_from_slice(temps);
        let mut kv = vec![0i64; blk * 64];
        for j in 0..blk {
            for (b, s) in seqs.iter().enumerate() { kv[j * 64 + b] = crate::sampling::key(draft_seed(seeds[b], temps[b], s.pos), (s.pos + j as i64 + 1) as u64) as i64; }
        }
        let _ = self.mtp_body_graph(seqs, e, pos, false);
        let mut graphs = self.mtp_graphs.borrow_mut();
        let g = graphs.get_mut(&bsz).expect("body graph");
        if g.mk.is_none() {
            super::mem::release();
            let prev = Tensor::zeros([bsz as i64], (Kind::Int64, dev));
            let tbuf = Tensor::zeros([64], (Kind::Float, dev));
            let kbuf = Tensor::zeros([blk as i64 * 64], (Kind::Int64, dev));
            let rows: Vec<Tensor> = (0..blk as i64).map(|j| Tensor::from_slice(&(0..bsz as i64).map(|b| b * blk as i64 + j).collect::<Vec<_>>()).to_device(dev)).collect();
            let qloc = (rs_on() || stats_on() || ranks_on()).then(|| Tensor::zeros([blk as i64, bsz as i64, self.sp.v1 - self.sp.v0], (Kind::Float, dev)));
            tch::Cuda::synchronize(0);
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let out = self.markov_steps(&g.xf, &g.logits, &prev, &tbuf, &kbuf, &rows, qloc.as_ref());
            end().expect("capture end");
            let mut keep = rows;
            keep.push(g.xf.shallow_clone());
            keep.push(g.logits.shallow_clone());
            g.mk = Some(MarkovGraph { prev, temps: tbuf, keys: kbuf, out, qloc, g: Owned::take(), _keep: keep });
            super::mem::occupy();
        }
        let mk = g.mk.as_mut().unwrap();
        super::kern::upload_into(&mk.prev, toks);
        super::kern::upload_into(&mk.temps, &tv);
        super::kern::upload_into(&mk.keys, &kv);
        mk.g.replay();
        match &mk.qloc { Some(ql) => self.keep_draft_q(ql, temps), None => *self.draft_q.borrow_mut() = None }
        let host: Vec<f32> = Vec::<f32>::try_from(mk.out.to_device(Device::Cpu).view(-1)).unwrap();   // [blk, B, 2]
        let mut drafts = vec![Vec::with_capacity(blk); bsz];
        let mut confs = vec![Vec::with_capacity(blk); bsz];
        for j in 0..blk {
            for b in 0..bsz {
                drafts[b].push(host[(j * bsz + b) * 2] as i64);
                confs[b].push(host[(j * bsz + b) * 2 + 1]);
            }
        }
        (drafts, confs)
    }

    /// The Markov / sampling / confidence steps of a draft on the device (captured by mtp_draft_graph): prev [B] the
    /// pending tokens, temps [64] and keys [block * 64] the Gumbel rows per draft, rows[j] the block rows of draft j.
    /// -> [block, B, 2] FP32 (token id, confidence); ids below 2^24 are exact.
    /// qloc [block, B, v1 - v0]: also keeps every draft's logits (DS41_SPEC_RS)
    fn markov_steps(&self, xf: &Tensor, logits: &Tensor, prev: &Tensor, temps: &Tensor, keys: &Tensor, rows: &[Tensor], qloc: Option<&Tensor>) -> Tensor {
        extern "C" { fn rs_gumbel_argmax_packet(values: *const f32, rows: i32, width: i32, start: i64, temps: *const f32, keys: *const u64,
                                                part_v: *mut f32, part_i: *mut i32, packet: *mut f32, world: i32, rank: i32) -> i32; }
        let mt = self.mtp.as_ref().unwrap();
        let c = &self.c;
        let b = prev.size()[0];
        let (world, rank) = (self.sp.world as i64, self.sp.rank as i64);
        let conf_x = mt.conf.narrow(0, 0, c.dim);
        let conf_m = mt.conf.narrow(0, c.dim, c.dspark_markov_rank);
        let mut prev = prev.shallow_clone();
        let mut outs = Vec::with_capacity(rows.len());
        for (j, r) in rows.iter().enumerate() {
            let emb = mt.markov_embed.index_select(0, &prev);                                         // [B, rank]
            let bias = super::kern::bf16_gemv(&emb, &mt.markov_head);                                 // [B, v1 - v0]
            let lg = (logits.index_select(0, r) + bias).to_kind(Kind::Float).contiguous();
            let width = lg.size()[1];
            if let Some(ql) = qloc { let _ = ql.get(j as i64).copy_(&lg); }
            let part_v = Tensor::empty([b * 32], (Kind::Float, self.dev));
            let part_i = Tensor::empty([b * 32], (Kind::Int, self.dev));
            let packet = Tensor::empty([world, b, 2], (Kind::Float, self.dev));
            let kj = keys.narrow(0, j as i64 * 64, 64);
            assert_eq!(unsafe { rs_gumbel_argmax_packet(lg.data_ptr().cast(), b as i32, width as i32, self.sp.v0, temps.data_ptr().cast(), kj.data_ptr().cast(),
                                                        part_v.data_ptr().cast(), part_i.data_ptr().cast(), packet.data_ptr().cast(), world as i32, rank as i32) }, 0, "draft gumbel top-1");
            let next = if world > 1 {
                crate::tp::allreduce(&packet);
                crate::head_select::choose(&packet)
            } else {
                packet.select(2, 1).get(0).to_kind(Kind::Int64)
            };
            let cf = xf.index_select(0, r).to_kind(Kind::Float).matmul(&conf_x) + emb.to_kind(Kind::Float).matmul(&conf_m);
            outs.push(Tensor::stack(&[next.to_kind(Kind::Float), cf], 1));
            prev = next;
        }
        Tensor::stack(&outs, 0)
    }

    /// What a verify of `n` rows from the current position changes (window ring slots), to roll back later.
    pub fn spec_checkpoint(&self, seq: &Seq, n: i64) -> Rollback {
        let uniform = !seq.win.is_empty() && seq.win.len() <= 64
            && seq.win.iter().all(|r| r.is_contiguous() && r.size() == seq.win[0].size() && r.kind() == seq.win[0].kind());
        if ring_fused_on() && n <= self.c.window && uniform {
            return Rollback { pos0: seq.pos, n, ring: Vec::new(), saved: Some(super::kern::ring_save(&seq.win, seq.pos, n)) };
        }
        let slots = Tensor::arange_start(seq.pos, seq.pos + n, (Kind::Int64, self.dev)).remainder(self.c.window);
        Rollback { pos0: seq.pos, n, ring: seq.win.iter().map(|r| r.index_select(0, &slots)).collect(), saved: None }
    }

    /// The window attention of stage i for every sequence's block rows: o [R, heads, head_dim].
    fn mtp_mid(&self, i: usize, layer: &Layer, q: &Tensor, kv: &Tensor, seqs: &[&mut Seq]) -> Tensor {
        let c = &self.c;
        let (blk, w, dev) = (c.dspark_block, c.window, self.dev);
        let outs: Vec<Tensor> = seqs.iter().enumerate().map(|(b, s)| {
            let (q_b, kv_b) = (q.narrow(0, b as i64 * blk, blk), kv.narrow(0, b as i64 * blk, blk));
            let nwin = s.pos.min(w);
            let row = Tensor::cat(&[Tensor::arange(nwin, (Kind::Int64, dev)), Tensor::arange(blk, (Kind::Int64, dev)) + w], 0);
            let idx = row.unsqueeze(0).expand([blk, nwin + blk], false).contiguous();
            let scale = (c.head_dim as f64).powf(-0.5);
            attn_fused().then(|| super::kern::sparse_attn_pools(&q_b, [Some(&s.mtp_win[i]), Some(&kv_b), None], None, &idx, &layer.attn.sink, scale)).flatten()
                .unwrap_or_else(|| ops::sparse_attn(&q_b, &Tensor::cat(&[s.mtp_win[i].shallow_clone(), kv_b.shallow_clone()], 0), &idx, &layer.attn.sink, scale, 256))
        }).collect();
        Tensor::cat(&outs, 0)
    }

    /// Eager draft body from the embedded rows e [R, dim]: (hc_pre output [R, dim], this rank's head logits [R, v]).
    /// `cap`: capture mode (the graph segments are recorded around the calls; see mtp_body_graph).
    fn mtp_body(&self, seqs: &[&mut Seq], e: &Tensor, pos: &Tensor, _cap: Option<()>) -> (Tensor, Tensor) {
        let mt = self.mtp.as_ref().unwrap();
        let c = &self.c;
        let r = e.size()[0];
        let mut h = e.unsqueeze(1).repeat([1, c.hc, 1]);
        let mut pre_mix = Tensor::zeros([r, c.hc], (Kind::Float, self.dev));
        let _ = pre_mix.narrow(1, 0, 1).fill_(1.0);
        for (i, layer) in mt.layers.iter().enumerate() {
            let residual = h.shallow_clone();
            let (a_pre, a_post, a_comb) = self.hc_mixes(&h, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base);
            let x = rmsn(&Self::hc_pre(&h, &pre_mix), &layer.attn_norm, c.norm_eps);
            let (_qr, q, kv) = self.attn_in(layer, &x, pos);
            let o = self.mtp_mid(i, layer, &q, &kv, seqs);
            let y = self.attn_out(layer, &o, pos);
            h = Self::hc_post(&y, &residual, &a_post, &a_comb);
            let residual = h.shallow_clone();
            let (f_pre, f_post, f_comb) = self.hc_mixes(&h, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base);
            let x2 = rmsn(&Self::hc_pre(&h, &a_pre), &layer.ffn_norm, c.norm_eps);
            let m = self.moe(&layer.moe, &x2);
            h = Self::hc_post(&m, &residual, &f_post, &f_comb);
            pre_mix = f_pre;
        }
        let xf = Self::hc_pre(&h, &pre_mix);
        let logits = self.draft_head(&rmsn(&xf, &mt.norm, c.norm_eps));
        (xf, logits)
    }

    /// The draft body through the per-B graphs (captured on first use).
    /// `copy_out` false: the graph's own output tensors (valid until its next replay).
    fn mtp_body_graph(&self, seqs: &[&mut Seq], e: &Tensor, pos_now: &Tensor, copy_out: bool) -> (Tensor, Tensor) {
        use crate::tp::graph::{begin_pool, end, Owned};
        let mt = self.mtp.as_ref().unwrap();
        let c = &self.c;
        let bsz = seqs.len();
        let r = e.size()[0];
        let mut graphs = self.mtp_graphs.borrow_mut();
        if let Some(g) = graphs.get_mut(&bsz) {
            let _ = g.pos.copy_(pos_now);
            let _ = g.h0.copy_(&e.unsqueeze(1).expand([r, c.hc, c.dim], false));
            for (i, layer) in mt.layers.iter().enumerate() {
                g.s1[i].replay();
                let [q, kv] = &g.mid[i];
                let o = self.mtp_mid(i, layer, q, kv, seqs);
                let _ = g.obuf[i].copy_(&o);
                g.s3[i].replay();
            }
            g.fin.replay();
            return if copy_out { (g.xf.copy(), g.logits.copy()) } else { (g.xf.shallow_clone(), g.logits.shallow_clone()) };
        }
        super::mem::release();
        let pos = pos_now.copy();
        let h0 = e.unsqueeze(1).repeat([1, c.hc, 1]).contiguous();
        let pre0 = Tensor::zeros([r, c.hc], (Kind::Float, self.dev));
        let _ = pre0.narrow(1, 0, 1).fill_(1.0);
        let mut keep = vec![pre0.shallow_clone()];
        let (mut s1, mut s3, mut mid, mut obuf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut h, mut pre_mix) = (h0.shallow_clone(), pre0.shallow_clone());
        for (i, layer) in mt.layers.iter().enumerate() {
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let (a_pre, a_post, a_comb) = self.hc_mixes(&h, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base);
            let x = rmsn(&Self::hc_pre(&h, &pre_mix), &layer.attn_norm, c.norm_eps);
            let (qr, q, kv) = self.attn_in(layer, &x, &pos);
            end().expect("capture end");
            let g1 = Owned::take();
            g1.replay();
            s1.push(g1);
            let o = self.mtp_mid(i, layer, &q, &kv, seqs);
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let y = self.attn_out(layer, &o, &pos);
            let h2 = Self::hc_post(&y, &h, &a_post, &a_comb);
            let (f_pre, f_post, f_comb) = self.hc_mixes(&h2, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base);
            let x2 = rmsn(&Self::hc_pre(&h2, &a_pre), &layer.ffn_norm, c.norm_eps);
            let m = self.moe(&layer.moe, &x2);
            let h3 = Self::hc_post(&m, &h2, &f_post, &f_comb);
            end().expect("capture end");
            let g3 = Owned::take();
            g3.replay();
            s3.push(g3);
            keep.extend([h.shallow_clone(), a_pre, a_post, a_comb, x, qr, y, h2, f_post, f_comb, x2, m]);
            mid.push([q, kv]);
            obuf.push(o);
            h = h3;
            pre_mix = f_pre;
            keep.extend([h.shallow_clone(), pre_mix.shallow_clone()]);
        }
        begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
        let xf = Self::hc_pre(&h, &pre_mix);
        let logits = self.draft_head(&rmsn(&xf, &mt.norm, c.norm_eps));
        end().expect("capture end");
        let fin = Owned::take();
        fin.replay();
        let out = if copy_out { (xf.copy(), logits.copy()) } else { (xf.shallow_clone(), logits.shallow_clone()) };
        graphs.insert(bsz, MtpGraph { pos, h0, s1, s3, fin, mid, obuf, xf, logits, _keep: keep, mk: None });
        super::mem::occupy();
        out
    }

    /// Runs ids (the pending token and the drafts) as one chunk of `seq`; returns (logits [n, vocab], target hidden
    /// [n, 3 dim], what to roll back).
    pub fn spec_verify(&self, seq: &mut Seq, ids: &[i64]) -> (Tensor, Tensor, Rollback) {
        let rb = self.spec_checkpoint(seq, ids.len() as i64);
        let (lg, hid) = self.forward_rows(&mut [seq], &[ids.to_vec()]);
        (lg, hid.expect("no target hidden (DSpark stages not loaded)"), rb)
    }

    /// Keeps the first `keep` rows of the verified chunk and undoes the rest.
    pub fn spec_rollback(&self, seq: &mut Seq, rb: Rollback, keep: i64) {
        if keep >= rb.n { return; }
        let w = self.c.window;
        let new_pos = rb.pos0 + keep;
        // window ring: the rejected positions' slots get their earlier keys back
        if let Some(saved) = &rb.saved {
            super::kern::ring_restore(&seq.win, saved, rb.pos0, keep);
        } else {
            let rej = Tensor::arange_start(new_pos, rb.pos0 + rb.n, (Kind::Int64, self.dev));
            let rows = &rej - rb.pos0;
            for (r, saved) in seq.win.iter_mut().zip(&rb.ring) { let _ = r.index_copy_(0, &rej.remainder(w), &saved.index_select(0, &rows)); }
        }
        // compressor open groups: the rows of the accepted positions of the (now) open group
        for (&src, (ks, ss)) in seq.cstate.iter_mut() {
            let ratio = self.c.compress[src];
            let rem = new_pos % ratio;
            if rem == 0 { continue; }
            let (kv_all, sc_all, base) = seq.comp_log.get(&src).expect("no compressor log");
            let off = new_pos - rem - base;
            let _ = ks.narrow(0, 0, rem).copy_(&kv_all.narrow(0, off, rem));
            let _ = ss.narrow(0, 0, rem).copy_(&sc_all.narrow(0, off, rem));
        }
        seq.hist.truncate(new_pos as usize);
        seq.pos = new_pos;
        seq.dec_pos = seq.dec_pos.min(new_pos);
    }

    /// One speculative round for one sequence with pending token `tok`: draft, verify, accept, roll back, seed.
    /// Returns (emitted tokens: accepted drafts + the target's next token (the new pending one), drafts proposed).
    pub fn spec_round(&self, seq: &mut Seq, tok: i64, temp: f32, seed: u64) -> (Vec<i64>, usize) {
        use super::model::prof;
        let p = seq.pos;
        let (d, conf) = prof("spec.draft", || self.mtp_draft(&[&mut *seq], &[tok], &[temp], &[seed]));
        let n_d = spec_len(&conf[0], d[0].len());
        let d = &d[0][..n_d];
        let mut ids = vec![tok];
        ids.extend(d);
        let (lg, hid, rb) = prof("spec.verify", || self.spec_verify(seq, &ids));
        let n = ids.len();
        let q = self.draft_q.borrow().as_ref().map(|q| q.select(1, 0));
        let ys: Vec<i64> = prof("spec.sample", || match &q {
            Some(q) if temp > 0.0 && rs_on() => self.rs_tokens(&lg, q, d, temp, seed, p),
            _ => (0..n).map(|j| super::serve::sample(&lg.narrow(0, j as i64, 1), temp, seed, p + j as i64 + 1)).collect(),
        });
        let mut k = 0;
        while k < d.len() && d[k] == ys[k] { k += 1; }
        if let Some(q) = &q { spec_stats(&lg, q, d, temp, draft_seed(seed, temp, p), p, k); }
        prof("spec.commit", || { self.spec_rollback(seq, rb, k as i64 + 1); self.mtp_seed(seq, &hid.narrow(0, 0, k as i64 + 1), p); });
        let mut out: Vec<i64> = d[..k].to_vec();
        out.push(ys[k]);
        (out, d.len())
    }
}
