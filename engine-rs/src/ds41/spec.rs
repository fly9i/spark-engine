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
pub struct MarkovGraph { prev: Tensor, temps: Tensor, keys: Tensor, out: Tensor, g: crate::tp::graph::Owned, _keep: Vec<Tensor> }

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
pub struct Rollback { pos0: i64, n: i64, ring: Vec<Tensor> }

/// FP32 rows [m, n] = x [m, k] . W^T for BF16 W (the BF16 GEMV in chunks of 8 rows).
fn head_rows(x: &Tensor, w: &Tensor) -> Tensor {
    let m = x.size()[0];
    let parts: Vec<Tensor> = (0..m).step_by(8).map(|r| super::kern::bf16_gemv(&x.narrow(0, r, 8.min(m - r)), w)).collect();
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

    /// The DSpark stages' window KV for main positions [pos0, pos0 + n) from their target hidden rows [n, 3 dim].
    pub fn mtp_seed(&self, seq: &mut Seq, hidden: &Tensor, pos0: i64) {
        let Some(mt) = &self.mtp else { return };
        let c = &self.c;
        let n = hidden.size()[0];
        let keep = n.min(c.window);
        let (hidden, pos0) = (hidden.narrow(0, n - keep, keep), pos0 + n - keep);
        let pos = Tensor::arange_start(pos0, pos0 + keep, (Kind::Int64, self.dev));
        let main_x = rmsn(&mt.main_proj.mm(&hidden), &mt.main_norm, c.norm_eps);
        for (i, layer) in mt.layers.iter().enumerate() {
            let a = &layer.attn;
            let kv = super::kern::act_quant(&rot(&rmsn(&a.wkv.mm(&main_x), &a.kv_norm, c.norm_eps), &a.rope, &pos, c.rope_dim, false));
            let _ = seq.mtp_win[i].index_copy_(0, &pos.remainder(c.window), &kv);
        }
    }

    /// Drafts for each sequence (toks[b] at position seqs[b].pos): (draft tokens, confidences), `block` each.
    pub fn mtp_draft(&self, seqs: &[&mut Seq], toks: &[i64], temps: &[f32], seeds: &[u64]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
        let _g = tch::no_grad_guard();
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
        let pos = Tensor::from_slice(&posv).to_device(dev);
        let e = self.embed(&Tensor::from_slice(&ids).to_device(dev));                        // [R, dim]
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
        for j in 0..blk {
            let emb = mt.markov_embed.index_select(0, &Tensor::from_slice(&prev).to_device(dev));     // [B, rank]
            let bias = super::kern::bf16_gemv(&emb, &mt.markov_head);                                 // [B, v1 - v0]
            let rows = Tensor::from_slice(&(0..bsz as i64).map(|b| b * blk + j).collect::<Vec<_>>()).to_device(dev);
            let lg = logits.index_select(0, &rows) + bias;
            if let Some(v) = self.draft_debug.borrow_mut().as_mut() {
                let full = Tensor::zeros([bsz as i64, c.vocab], (Kind::Float, dev));
                full.narrow(1, self.sp.v0, self.sp.v1 - self.sp.v0).copy_(&lg);
                if self.sp.world > 1 { crate::tp::allreduce(&full); }
                v.push(full);
            }
            let keys: Vec<u64> = seqs.iter().enumerate().map(|(b, s)| crate::sampling::key(seeds[b], (s.pos + j + 1) as u64)).collect();
            let next = self.sample_split(&lg, temps, &keys);
            let cf = (xf.index_select(0, &rows).to_kind(Kind::Float).matmul(&conf_x) + emb.to_kind(Kind::Float).matmul(&conf_m)).to_device(Device::Cpu);
            let cf: Vec<f32> = Vec::<f32>::try_from(cf).unwrap();
            for b in 0..bsz { drafts[b].push(next[b]); confs[b].push(cf[b]); }
            prev = next;
        }
        (drafts, confs)
    }

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
            for (b, s) in seqs.iter().enumerate() { kv[j * 64 + b] = crate::sampling::key(seeds[b], (s.pos + j as i64 + 1) as u64) as i64; }
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
            tch::Cuda::synchronize(0);
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let out = self.markov_steps(&g.xf, &g.logits, &prev, &tbuf, &kbuf, &rows);
            end().expect("capture end");
            let mut keep = rows;
            keep.push(g.xf.shallow_clone());
            keep.push(g.logits.shallow_clone());
            g.mk = Some(MarkovGraph { prev, temps: tbuf, keys: kbuf, out, g: Owned::take(), _keep: keep });
            super::mem::occupy();
        }
        let mk = g.mk.as_mut().unwrap();
        let _ = mk.prev.copy_(&Tensor::from_slice(toks));
        let _ = mk.temps.copy_(&Tensor::from_slice(&tv));
        let _ = mk.keys.copy_(&Tensor::from_slice(&kv));
        mk.g.replay();
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
    fn markov_steps(&self, xf: &Tensor, logits: &Tensor, prev: &Tensor, temps: &Tensor, keys: &Tensor, rows: &[Tensor]) -> Tensor {
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
        let slots = Tensor::arange_start(seq.pos, seq.pos + n, (Kind::Int64, self.dev)).remainder(self.c.window);
        Rollback { pos0: seq.pos, n, ring: seq.win.iter().map(|r| r.index_select(0, &slots)).collect() }
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
        let rej = Tensor::arange_start(new_pos, rb.pos0 + rb.n, (Kind::Int64, self.dev));
        let rows = &rej - rb.pos0;
        for (r, saved) in seq.win.iter_mut().zip(&rb.ring) { let _ = r.index_copy_(0, &rej.remainder(w), &saved.index_select(0, &rows)); }
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
        let ys: Vec<i64> = prof("spec.sample", || (0..n).map(|j| super::serve::sample(&lg.narrow(0, j as i64, 1), temp, seed, p + j as i64 + 1)).collect());
        let mut k = 0;
        while k < d.len() && d[k] == ys[k] { k += 1; }
        prof("spec.commit", || { self.spec_rollback(seq, rb, k as i64 + 1); self.mtp_seed(seq, &hid.narrow(0, 0, k as i64 + 1), p); });
        let mut out: Vec<i64> = d[..k].to_vec();
        out.push(ys[k]);
        (out, d.len())
    }
}
