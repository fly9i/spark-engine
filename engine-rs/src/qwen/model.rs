//! Qwen3.8-Flash-Next (qwen4_exp) forward for decode / verify chains of <= 64 rows (one sequence, or up to 8
//! sequences batched: linears / HC / MoE run on all rows at once, the recurrent and attention state per segment).
//! Data flow per layer: [PLE] -> attn HC mix -> GDN | QSA -> apply -> MLP HC mix -> MoE -> apply.
//! Speculative chains: `step(.., commit = false)` evaluates rows without writing GDN state / conv windows;
//! `commit(.., n)` replays the first n rows of the last chain into the state (attention caches are
//! position-addressed and need no commit).
use super::ffi;
use super::load::Checkpoint;
use std::path::Path;
use tch::{Device, Kind, Tensor};

pub const H: i64 = 4;
pub const D: i64 = 2560;
pub const MAX_ROWS: i64 = 16;
/// Rows of one decode / verify forward (several sequences batched) that the decode kernels take in one weight pass.
pub const DEC_ROWS: i64 = 64;
thread_local! {
    /// Inside a prompt prefill: chunks above MAX_ROWS rows take the prefill kernels.
    static PREFILLING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// Forwards of up to this many rows use the decode kernels: DEC_ROWS for decode / verify (batched sequences give the
/// same rows as one sequence), MAX_ROWS inside a prompt prefill (for 17..64-row chunks the prefill kernels are closer
/// to the fp32 reference: last-48-row KL 0.00091 vs 0.00144).
pub fn dec_rows() -> i64 { if PREFILLING.with(|p| p.get()) { MAX_ROWS } else { DEC_ROWS } }
/// Marks a prompt prefill for its lifetime.
struct PrefillScope(bool);
impl PrefillScope {
    fn new() -> Self { PrefillScope(PREFILLING.with(|p| p.replace(true))) }
}
impl Drop for PrefillScope { fn drop(&mut self) { PREFILLING.with(|p| p.set(self.0)); } }
/// Row count above which linears switch to decode-once + GEMM (prefill)
pub const BIG_ROWS: i64 = 64;
/// Attention sub-chunk rows of a layer-major prefill (QWEN_PREFILL_CHUNK, read per pass; default 2048: 1% faster than 1024).
/// The tail-512 KL of 4 prompts against the fp32 reference moves by up to 10x per prompt with this size (0.0024..0.0141 for
/// one, 0.008..0.052 for another): the FP8 KV cache turns rounding-level differences into flipped fp8 codes (6% steps) that
/// grow over the layers, so any rounding change acts like a new random seed there; compare L1 changes over many prompts.
pub fn prefill_chunk() -> i64 {
    std::env::var("QWEN_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048)
}
const SEL_LD: i64 = 2052;
const P: &str = "model.language_model.";

fn f32(t: &Tensor) -> Tensor { t.to_kind(Kind::Float) }

thread_local! {
    /// Decoded EXL3 weights (keyed by trellis address and form: inner, or folded W^T) while a layer-major prefill works
    /// on one layer.
    static INNER: std::cell::RefCell<Option<std::collections::HashMap<(usize, bool), Tensor>>> = const { std::cell::RefCell::new(None) };
}

/// Rows per MoE call in a layer-major prefill (QWEN_PREFILL_MOE_ROWS, read per call; default 16384).
fn prefill_moe_rows() -> i64 {
    std::env::var("QWEN_PREFILL_MOE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)
}
/// Tokens per layer-major pass (QWEN_PREFILL_MACRO, read per call; default 16384): the whole pass's stacks stay resident.
pub fn prefill_macro() -> i64 {
    std::env::var("QWEN_PREFILL_MACRO").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)
}
fn emp(shape: &[i64], kind: Kind) -> Tensor { Tensor::empty(shape, (kind, Device::Cuda(0))) }

/// QWEN_EXL3_FOLD=1 (read per call; L1, off): many-row EXL3 linears use the effective weight W = diag(suh) H inner H
/// diag(svh) in fp16 with a cuBLASLt GEMM instead of transforming the activations (had_in, GEMM with the decoded inner
/// weight, finish). 2026-10-07: 12K prefill -3.8%, but tail-512 KL vs the fp32 reference 0.0032 -> 0.0108 (folding
/// even only the shared experts: 0.0041): the transformed weights do not fit fp16 as well as the codebook values.
fn fold_on() -> bool { std::env::var("QWEN_EXL3_FOLD").as_deref() == Ok("1") }

/// EXL3 linear: y = x @ W, W = ((H inner) * suh) H * svh.
pub struct Exl3 { tr: Tensor, suh: Tensor, svh: Tensor, pub k: i64, pub n: i64, bits: i32, cnt: Tensor }
impl Exl3 {
    pub fn load(ck: &Checkpoint, key: &str) -> Self {
        let tr = ck.get(&format!("{key}.trellis"));
        let s = tr.size();
        Exl3 { suh: ck.get(&format!("{key}.suh")), svh: ck.get(&format!("{key}.svh")), k: s[0] * 16, n: s[1] * 16,
               bits: (s[2] / 16) as i32, cnt: Tensor::zeros([s[1] / 8], (Kind::Int, Device::Cuda(0))), tr }
    }
    /// K slices: about 512 blocks per launch (8+ waves on 48 SMs) while keeping >= 8 k-tiles per slice; 10 for K = 6144 and 6
    /// for N >= 6144 (QWEN_EXL3_SLICES_TUNED=0, read per call: the general rule). 2026-10-07: the tuned slices are 1..2% faster
    /// per round; their round-3 rejection (one prompt's last 48 rows, KL 0.00144 -> 0.00186) was within that test's noise:
    /// over 4 prompts x 512 rows the paired KL difference is -0.0042 +- 0.0029.
    fn slices(&self) -> i32 {
        let kt = self.k / 16;
        if std::env::var("QWEN_EXL3_SLICES_TUNED").as_deref() != Ok("0") {
            if self.k == 6144 { return 10; }
            if self.n >= 6144 { return 6; }
        }
        (512 / (self.n / 128).max(1)).min(16).min(kt / 8).max(1) as i32
    }
    /// The decoded inner weight transposed, [N, K] fp16 (W = diag(suh) H inner H diag(svh), H the 128-block Hadamard), and
    /// suh / svh: for products that do the Hadamards themselves (the vision tower's exact linears).
    pub fn inner_t(&self) -> Tensor {
        let inner = emp(&[self.k, self.n], Kind::Half);
        ffi::exl3_reconstruct(&self.tr, &inner, self.k, self.n, self.bits);
        inner.tr().contiguous()
    }
    pub fn scales(&self) -> (&Tensor, &Tensor) { (&self.suh, &self.svh) }
    /// x [M, K] (fp32, row-contiguous) -> [M, N] fp32
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let m = x.size()[0];
        if m > BIG_ROWS { return self.fwd_big(x); }
        // QWEN_EXL3_FUSED=1 (read per call; off): up to 8 rows in one launch (had_in + gemv + finish, bitwise equal).
        // 2026-10-07 same-process A/B at one sequence: 5% slower per round (each block's input transform and barrier
        // delay its weight stream), although it removes two launches per linear.
        if m <= 8 && x.kind() == Kind::Float && std::env::var("QWEN_EXL3_FUSED").as_deref() == Ok("1") {
            let s = self.slices();
            let y = emp(&[m, self.n], Kind::Float);
            let part = emp(&[s as i64, m, self.n], Kind::Float);
            ffi::exl3_linear(&x.contiguous(), &self.suh, &self.tr, &self.svh, &part, &y, &self.cnt, m, self.k, self.n, self.bits, s);
            return y;
        }
        let y = emp(&[m, self.n], Kind::Float);
        let s = self.slices();
        let mut r0 = 0;
        while r0 < m {
            let mm = (m - r0).min(dec_rows());
            let xs = x.narrow(0, r0, mm);
            let xh = emp(&[mm, self.k], Kind::Half);
            let part = emp(&[s as i64, mm, self.n], Kind::Float);
            ffi::exl3_had_in(&xs, &self.suh, &xh, mm, self.k);
            ffi::exl3_gemv(&xh, &self.tr, &part, mm, self.k, self.n, self.bits, s);
            ffi::exl3_finish(&part, s, &self.svh, &y.narrow(0, r0, mm), mm, self.n);
            r0 += mm;
        }
        y
    }
    /// Many rows: decode the inner weight to fp16 once, tensor-core GEMM, Hadamards on both sides. Inside a
    /// layer-major prefill (INNER cache on) the decoded weight is kept for the layer's other sub-chunks.
    fn inner_w(&self, fold: bool) -> Tensor {
        INNER.with(|c| {
            let mut c = c.borrow_mut();
            let key = (self.tr.data_ptr() as usize, fold);
            if let Some(map) = c.as_mut() {
                if let Some(t) = map.get(&key) { return t.shallow_clone(); }
            }
            let w = if fold {
                let wt = emp(&[self.n, self.k], Kind::Half);
                ffi::exl3_fold(&self.tr, &self.suh, &self.svh, self.k, self.n, self.bits, &wt);
                wt
            } else {
                let inner = emp(&[self.k, self.n], Kind::Half);
                ffi::exl3_reconstruct(&self.tr, &inner, self.k, self.n, self.bits);
                inner
            };
            if let Some(map) = c.as_mut() { map.insert(key, w.shallow_clone()); }
            w
        })
    }
    fn fwd_big(&self, x: &Tensor) -> Tensor {
        let m = x.size()[0];
        if fold_on() {
            // y = x @ W: fp16 x (as given, or converted once here), fp32 out, no Hadamard passes
            let inner = self.inner_w(true);
            let x16 = if x.kind() == Kind::Half { x.contiguous() } else { x.to_kind(Kind::Half) };
            let y = emp(&[m, self.n], Kind::Float);
            ffi::lt_mm16(&x16, &inner, &y);
            return y;
        }
        self.finish_big(&self.big_part(x, None))
    }
    /// The unfinished many-row product had128(x * suh) @ inner (fp16 [M, N]); `xh`: the input transform if done already.
    fn big_part(&self, x: &Tensor, xh: Option<Tensor>) -> Tensor {
        let m = x.size()[0];
        let xh = xh.unwrap_or_else(|| { let xh = emp(&[m, self.k], Kind::Half); ffi::exl3_had_in(x, &self.suh, &xh, m, self.k); xh });
        xh.matmul(&self.inner_w(false))
    }
    /// y = had128(part) * svh (fp32)
    fn finish_big(&self, part: &Tensor) -> Tensor {
        let m = part.size()[0];
        let y = emp(&[m, self.n], Kind::Float);
        if std::env::var("QWEN_EXL3_FIN_F32").as_deref() == Ok("1") {   // A/B: convert the GEMM output first
            ffi::exl3_finish(&part.to_kind(Kind::Float), 1, &self.svh, &y, m, self.n);
        } else {
            ffi::exl3_finish_h(part, &self.svh, &y, m, self.n);
        }
        y
    }
    /// Output columns [c0, c0 + nc) (multiples of 128) of the many-row form: x [M, K] fp32 -> [M, nc] fp32.
    pub fn fwd_cols(&self, x: &Tensor, c0: i64, nc: i64) -> Tensor {
        let m = x.size()[0];
        let tr = self.tr.narrow(1, c0 / 16, nc / 16).contiguous();
        let inner = emp(&[self.k, nc], Kind::Half);
        ffi::exl3_reconstruct(&tr, &inner, self.k, nc, self.bits);
        let xh = emp(&[m, self.k], Kind::Half);
        ffi::exl3_had_in(x, &self.suh, &xh, m, self.k);
        let part = xh.matmul(&inner);
        let y = emp(&[m, nc], Kind::Float);
        ffi::exl3_finish_h(&part, &self.svh.narrow(0, c0, nc).contiguous(), &y, m, nc);
        y
    }
    pub fn tensors(&self) -> [&Tensor; 3] { [&self.tr, &self.suh, &self.svh] }
}

/// Decode linears with the same input in three launches (QWEN_EXL3_MULTI=0, read per call: one at a time; bitwise equal
/// either way). Returns (the input, the outputs); flags as ffi::exl3_multi (silu: the first output is silu(a) * b).
fn exl3_multi(lins: &[&Exl3], inp: &MixIn, flags: i32) -> (Tensor, Vec<Tensor>) {
    let (x, mix, mixed) = match inp {
        MixIn::Ready(x) => (Some(x), None, x.shallow_clone()),
        MixIn::Lazy { g, n, mixed } => (None, Some((g, n)), mixed.shallow_clone()),
    };
    let (m, k) = (mixed.size()[0], lins[0].k);
    let ml: Vec<ffi::MultiLin> = lins.iter().map(|l| {
        let s = l.slices();
        ffi::MultiLin { suh: &l.suh, tr: &l.tr, svh: &l.svh, bits: l.bits, s, xh: emp(&[m, k], Kind::Half), part: emp(&[s as i64, m, l.n], Kind::Float),
                        y: emp(&[m, l.n], Kind::Float) }
    }).collect();
    ffi::exl3_multi(x, mix, mix.is_some().then_some(&mixed), m, k, &ml, flags);
    (mixed, ml.into_iter().map(|l| l.y).collect())
}
/// Many-row linears on one fp32 input: one input-transform launch for all (QWEN_PREFILL_HAD_MULTI=0, read per call: one
/// each; bitwise equal), then the GEMMs; returns the unfinished products (see Exl3::big_part).
fn big_parts(lins: &[&Exl3], x: &Tensor) -> Vec<Tensor> {
    if lins.len() == 1 || std::env::var("QWEN_PREFILL_HAD_MULTI").as_deref() == Ok("0") { return lins.iter().map(|l| l.big_part(x, None)).collect(); }
    let m = x.size()[0];
    let xh: Vec<Tensor> = lins.iter().map(|l| emp(&[m, l.k], Kind::Half)).collect();
    ffi::exl3_had_multi(x, &lins.iter().map(|l| &l.suh).collect::<Vec<_>>(), &xh.iter().collect::<Vec<_>>());
    lins.iter().zip(xh).map(|(l, h)| l.big_part(x, Some(h))).collect()
}
/// Prefill chunks whose projections' finishes are fused into their consumers (QWEN_PREFILL_FIN_SPLIT=1, read per call: off)
fn fin_fuse(r: i64) -> bool { r > BIG_ROWS && !fold_on() && std::env::var("QWEN_PREFILL_FIN_SPLIT").as_deref() != Ok("1") }

/// Whether rows r take exl3_multi (decode rows; QWEN_EXL3_MULTI=0 read per call)
/// QWEN_MOE_ROUTE_SIDE=1 / QWEN_GDN_AB_SIDE=1 (off; read per call, i.e. at graph capture; L0, schedule only): in the
/// decode multi path the small latency-bound F16 gemvs (MoE router + top-k, GDN a/b) run on a pool side stream forked
/// right after the shared input transform, next to the bandwidth-bound EXL3 gemvs on the main stream (GLM r26's idea).
/// Only engine kernels run on the side stream (no cuBLAS: F16Lin takes its gemv for <= dec_rows rows).
/// 2026-10-08 same-process A/B: outputs identical, round time unchanged within noise (bench/qwen/README.md), so off.
fn side_on(var: &str) -> bool { std::env::var(var).as_deref() == Ok("1") }
extern "C" { fn rs_stream_set(i: i32) -> i32; fn rs_stream_join(n: i32) -> i32; }
fn on_side(side: bool) { if side { assert_eq!(unsafe { rs_stream_set(0) }, 0, "side stream"); } }
fn on_main(side: bool) { if side { assert_eq!(unsafe { rs_stream_set(-1) }, 0, "main stream"); } }
fn join_side(side: bool) { if side { assert_eq!(unsafe { rs_stream_join(1) }, 0, "side join"); } }
fn multi_on(r: i64) -> bool { r <= dec_rows() && std::env::var("QWEN_EXL3_MULTI").as_deref() != Ok("0") }

/// A block input: the mixed rows, or (decode rows) the HC mix operands g, n [R, 10240] whose mix the first consumer's input
/// transform computes into `mixed` (QWEN_MIX_LAZY=0, read per call: always mixed first).
pub enum MixIn { Ready(Tensor), Lazy { g: Tensor, n: Tensor, mixed: Tensor } }
impl MixIn {
    /// The mixed rows (mixing now if lazy).
    fn ready(self) -> Tensor {
        match self {
            MixIn::Ready(x) => x,
            MixIn::Lazy { g, n, mixed } => { ffi::hc_mix2(&g, &n, &mixed, mixed.size()[0]); mixed }
        }
    }
    fn rows(&self) -> i64 { match self { MixIn::Ready(x) => x.size()[0], MixIn::Lazy { mixed, .. } => mixed.size()[0] } }
}

/// QWEN_Q8_DENSE=1 (L3, lossy): Q8 copies of the F16 dense weights (HC, router, PLE / GDN a,b projections) for
/// decode / verify rows; QWEN_Q8_OFF=1 (read per call, i.e. at graph capture) falls back to F16 for A/B.
fn q8_dense() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("QWEN_Q8_DENSE").as_deref() == Ok("1"))
}

/// Plain F16 linear (checkpoint orientation W [N, K]): skinny kernel for <= 16 rows, tensor-core GEMM above.
pub struct F16Lin { w: Tensor, n: i64, k: i64, q8: Option<(Tensor, Tensor)>, cnt: Tensor }
impl F16Lin {
    /// `tag` (hc_down, hc_up, router, gdn_ab, ple) selects the Q8 copy with QWEN_Q8_SCOPE (comma list, default all)
    fn tagged(w: Tensor, tag: &str) -> Self {
        let s = w.size();
        let w = w.to_kind(Kind::Half).contiguous();
        let scope = std::env::var("QWEN_Q8_SCOPE").ok();
        let in_scope = scope.as_deref().map_or(true, |v| v.split(',').any(|t| t == tag));
        let q8 = (q8_dense() && in_scope && s[1] % 64 == 0).then(|| {
            let q = emp(&[s[0], s[1]], Kind::Int8);
            let sc = emp(&[s[0], s[1] / 64], Kind::Float);
            ffi::q8_encode(&w, s[0], s[1], &q, &sc);
            (q, sc)
        });
        F16Lin { w, n: s[0], k: s[1], q8, cnt: Tensor::zeros([(s[0] + 3) / 4 + 1], (Kind::Int, Device::Cuda(0))) }
    }
    /// x [M, K] fp32 (row-contiguous) -> [M, N] fp32
    fn fwd(&self, x: &Tensor) -> Tensor {
        let m = x.size()[0];
        if m > dec_rows() { return x.to_kind(Kind::Half).matmul(&self.w.tr()).to_kind(Kind::Float); }
        if let (Some((q, sc)), true) = (&self.q8, m <= MAX_ROWS) {
            if std::env::var("QWEN_Q8_OFF").as_deref() != Ok("1") {
                let y = emp(&[m, self.n], Kind::Float);
                let ws = emp(&[ffi::q8_ws_bytes(m, self.n, self.k)], Kind::Uint8);
                ffi::q8_gemv(x, m, q, sc, self.k, self.n, &y, &ws);
                return y;
            }
        }
        let y = emp(&[m, self.n], Kind::Float);
        let ws = emp(&[ffi::f16_ws_bytes(m, self.n, self.k)], Kind::Uint8);
        // QWEN_F16_FUSED=1 (read per call; off, no faster): the slice sum in the gemv's last blocks
        if std::env::var("QWEN_F16_FUSED").as_deref() == Ok("1") { ffi::f16_gemv_c(x, m, &self.w, self.k, self.n, &y, &ws, &self.cnt); }
        else { ffi::f16_gemv(x, m, &self.w, self.k, self.n, &y, &ws); }
        y
    }
}

pub struct Hc { w1: Tensor, w: Tensor, up: Tensor, inject: bool, down_l: F16Lin, up_l: F16Lin }
impl Hc {
    fn load(ck: &Checkpoint, key: &str, inject: bool) -> Self {
        let w1 = f32(&ck.get(&format!("{key}.hc_norm.weight"))) + 1.0;
        let mut w = ck.get(&format!("{key}.input_mix_weight_down.weight"));
        if inject { w = Tensor::cat(&[w, ck.get(&format!("{key}.block_inject_weight.weight"))], 0); }
        let w = w.to_kind(Kind::Half).contiguous();
        let up = ck.get(&format!("{key}.input_mix_weight_up.weight")).to_kind(Kind::Half).contiguous();
        Hc { w1, down_l: F16Lin::tagged(w.shallow_clone(), "hc_down"), up_l: F16Lin::tagged(up.shallow_clone(), "hc_up"), w, up, inject }
    }
    /// X [R, 4, 2560] -> (post [R, 4] if inject, mixed [R, 2560])
    fn mix(&self, x: &Tensor) -> (Option<Tensor>, Tensor) { self.mix_after(x, None) }

    /// `mix`, optionally preceded by the residual update X += post * y of the previous site (prefill rows: fused with
    /// this site's norm, bitwise equal to hc_apply + mix)
    fn mix_after(&self, x: &Tensor, apply: Option<(&Tensor, &Tensor)>) -> (Option<Tensor>, Tensor) {
        let r = x.size()[0];
        if r <= dec_rows() && std::env::var("QWEN_HC_V1").is_err() && apply.is_some() {
            let post = self.inject.then(|| emp(&[r, H], Kind::Float));
            let mixed = emp(&[r, D], Kind::Float);
            let (post, g, n) = self.mix_parts(x, apply, post);
            ffi::hc_mix2(&g, &n, &mixed, r);
            return (post, mixed);
        }
        if let Some((post, y)) = apply {
            if r <= dec_rows() || std::env::var("QWEN_PREFILL_HC_TORCH").as_deref() == Ok("1")
               || std::env::var("QWEN_HC_APPLY_SPLIT").as_deref() == Ok("1") { ffi::hc_apply(x, post, y, r); }
            else {
                let mixed = emp(&[r, D], Kind::Float);
                let post2 = self.inject.then(|| emp(&[r, H], Kind::Float));
                let nh = emp(&[r, H * D], Kind::Half);
                let sc = emp(&[r * H], Kind::Float);
                ffi::hc_apply_norm_h(x, post, y, &self.w1, &nh, &sc, r);
                return self.mix_tail(x, nh, sc, post2, mixed);
            }
        }
        // prefill rows go through the v2 kernels too (L1: rounding only; 11% faster prefill and closer to the fp32
        // reference); QWEN_PREFILL_HC_TORCH=1: the former torch-op path
        if r > dec_rows() && std::env::var("QWEN_PREFILL_HC_TORCH").as_deref() == Ok("1") {
            let n = x * (x.square().mean_dim(-1, true, Kind::Float) + 1e-6).rsqrt() * self.w1.view([H, D]);
            let d = n.flatten(1, 2).to_kind(Kind::Half).matmul(&self.w.tr()).to_kind(Kind::Float);
            let t = (d.narrow(1, 0, 320) / 4.0).silu().to_kind(Kind::Half);
            let g = t.matmul(&self.up.tr()).to_kind(Kind::Float).sigmoid().view([r, H, D]);
            let mixed = (g * &n).mean_dim(1, false, Kind::Float).contiguous();
            let post = self.inject.then(|| ((d.narrow(1, 320, 4) / 4.0).sigmoid() * 2.0).contiguous());
            return (post, mixed);
        }
        let mixed = emp(&[r, D], Kind::Float);
        let post = self.inject.then(|| emp(&[r, H], Kind::Float));
        if r > dec_rows() {
            // prefill rows: fp16 matmul operands straight from the kernels (no fp32 n, no dtype conversions)
            let nh = emp(&[r, H * D], Kind::Half);
            let sc = emp(&[r * H], Kind::Float);
            ffi::hc_norm_h(x, r, &self.w1, &nh, &sc);
            return self.mix_tail(x, nh, sc, post, mixed);
        }
        if r <= MAX_ROWS && std::env::var("QWEN_HC_V1").is_ok() {
            let ws = emp(&[ffi::hc_ws_bytes(r)], Kind::Uint8);
            ffi::hc_mix(x, r, &self.w1, &self.w, self.inject, &self.up, &mixed, post.as_ref(), &ws);
            return (post, mixed);
        }
        let (post, g, n) = self.mix_parts(x, None, post);
        ffi::hc_mix2(&g, &n, &mixed, r);
        (post, mixed)
    }

    /// Decode rows: norm, down, mid2, up; returns (post, g, n) (the mix itself left to the caller). `apply` (the previous
    /// site's residual update) is fused with the norm (QWEN_HC_APPLY_SPLIT=1, read per call: apply, then norm).
    fn mix_parts(&self, x: &Tensor, apply: Option<(&Tensor, &Tensor)>, post: Option<Tensor>) -> (Option<Tensor>, Tensor, Tensor) {
        let r = x.size()[0];
        let n = emp(&[r, H * D], Kind::Float);
        match apply {
            Some((po, y)) if std::env::var("QWEN_HC_APPLY_SPLIT").as_deref() != Ok("1") => ffi::hc_apply_norm(x, po, y, &self.w1, &n, r),
            Some((po, y)) => { ffi::hc_apply(x, po, y, r); ffi::hc_norm(x, r, &self.w1, &n); }
            None => ffi::hc_norm(x, r, &self.w1, &n),
        }
        let d = self.down_l.fwd(&n);
        let t = emp(&[r, 320], Kind::Float);
        ffi::hc_mid2(&d, &t, post.as_ref(), r, self.inject);
        let g = self.up_l.fwd(&t);
        (post, g, n)
    }

    /// `mix_after` with the result as a block input: decode rows (v2 path) leave the mix to the consumer (MixIn::Lazy).
    fn mix_after_in(&self, x: &Tensor, apply: Option<(&Tensor, &Tensor)>) -> (Option<Tensor>, MixIn) {
        let r = x.size()[0];
        if r > dec_rows() || std::env::var("QWEN_HC_V1").is_ok() {
            let (post, m) = self.mix_after(x, apply);
            return (post, MixIn::Ready(m));
        }
        let post = self.inject.then(|| emp(&[r, H], Kind::Float));
        let (post, g, n) = self.mix_parts(x, apply, post);
        if std::env::var("QWEN_MIX_LAZY").as_deref() == Ok("0") {
            let mixed = emp(&[r, D], Kind::Float);
            ffi::hc_mix2(&g, &n, &mixed, r);
            return (post, MixIn::Ready(mixed));
        }
        (post, MixIn::Lazy { g, n, mixed: emp(&[r, D], Kind::Float) })
    }
}

impl Hc {
    /// Prefill rows after the norm: fp16 matmuls, mid2h, mix2h.
    fn mix_tail(&self, x: &Tensor, nh: Tensor, sc: Tensor, post: Option<Tensor>, mixed: Tensor) -> (Option<Tensor>, Tensor) {
        let r = x.size()[0];
        let d = nh.matmul(&self.w.tr());
        let t = emp(&[r, 320], Kind::Half);
        ffi::hc_mid2h(&d, &t, post.as_ref(), r, self.inject);
        // up GEMM fused with the mix (g stays fp32 in registers: L1); QWEN_HC_UPMIX_SPLIT=1: cuBLAS g (fp16) + mix2h
        if std::env::var("QWEN_HC_UPMIX_SPLIT").as_deref() == Ok("1") {
            let g = t.matmul(&self.up.tr());
            ffi::hc_mix2h(&g, x, &sc, &self.w1, &mixed, r);
        } else {
            ffi::hc_upmix(&t, &self.up, x, &sc, &self.w1, &mixed, r);
        }
        (post, mixed)
    }
}

pub struct Gdn { qkv: Exl3, z: Exl3, o: Exl3, ab: F16Lin, conv: Tensor, a_log: Tensor, dt_bias: Tensor, norm: Tensor }
impl Gdn {
    fn load(ck: &Checkpoint, key: &str) -> Self {
        let ab = Tensor::cat(&[ck.get(&format!("{key}.in_proj_a.weight")), ck.get(&format!("{key}.in_proj_b.weight"))], 0);
        Gdn { qkv: Exl3::load(ck, &format!("{key}.in_proj_qkv")), z: Exl3::load(ck, &format!("{key}.in_proj_z")),
              o: Exl3::load(ck, &format!("{key}.out_proj")), ab: F16Lin::tagged(ab, "gdn_ab"),
              conv: ck.get(&format!("{key}.conv1d.weight")).contiguous(), a_log: ck.get(&format!("{key}.A_log")),
              dt_bias: ck.get(&format!("{key}.dt_bias")), norm: ck.get(&format!("{key}.norm.weight")) }
    }
}

pub struct Qsa { q: Exl3, k: Exl3, v: Exl3, o: Exl3, idx: Exl3, qn: Tensor, kn: Tensor, iqn: Tensor, ikn: Tensor }
impl Qsa {
    fn load(ck: &Checkpoint, key: &str) -> Self {
        let e = |n: &str| Exl3::load(ck, &format!("{key}.{n}"));
        let g = |n: &str| ck.get(&format!("{key}.{n}.weight"));
        Qsa { q: e("q_proj"), k: e("k_proj"), v: e("v_proj"), o: e("o_proj"), idx: e("indexer.index_qk_proj"),
              qn: g("q_norm"), kn: g("k_norm"), iqn: g("indexer.q_layernorm"), ikn: g("indexer.k_layernorm") }
    }
}

pub struct Moe { gate: F16Lin, sh: [Exl3; 3], tab: Tensor, _experts: Vec<Exl3> }
impl Moe {
    fn load(ck: &Checkpoint, key: &str) -> Self {
        let mut experts = Vec::with_capacity(512 * 3);
        let mut ptrs = vec![0i64; 9 * 512];
        for e in 0..512 {
            for (t, n) in ["gate", "up", "down"].iter().enumerate() {
                let x = Exl3::load(ck, &format!("{key}.experts.{e}.{n}_proj"));
                for (j, ten) in x.tensors().iter().enumerate() { ptrs[(j * 3 + t) * 512 + e] = ten.data_ptr() as i64; }
                experts.push(x);
            }
        }
        let sh = [Exl3::load(ck, &format!("{key}.shared_expert.gate_proj")), Exl3::load(ck, &format!("{key}.shared_expert.up_proj")),
                  Exl3::load(ck, &format!("{key}.shared_expert.down_proj"))];
        // router rows 0..512, shared-expert gate row 512, zero rows to a multiple of 4
        let g = ck.get(&format!("{key}.gate.weight"));
        let sg = ck.get(&format!("{key}.shared_expert_gate.weight"));
        let gate = Tensor::cat(&[g.to_kind(Kind::Half), sg.to_kind(Kind::Half), Tensor::zeros([3, D], (Kind::Half, Device::Cuda(0)))], 0);
        Moe { gate: F16Lin::tagged(gate, "router"),
              sh, tab: Tensor::from_slice(&ptrs).view([9, 512]).to_device(Device::Cuda(0)), _experts: experts }
    }
    /// MoE on a block input: decode rows run the shared expert's gate / up first (one multi launch that also mixes the
    /// input), then the router on the mixed rows.
    fn fwd_in(&self, inp: MixIn) -> Tensor {
        let r = inp.rows();
        if !(multi_on(r) && std::env::var("QWEN_SHARED_UNFUSED").as_deref() != Ok("1")) { return self.fwd(&inp.ready()); }
        let [g, u, d] = &self.sh;
        let side = r <= dec_rows() && side_on("QWEN_MOE_ROUTE_SIDE");
        let idx = emp(&[r, 10], Kind::Int);
        let w = emp(&[r, 10], Kind::Float);
        let (x, o) = exl3_multi(&[g, u], &inp, 1 | if side { 4 } else { 0 });
        // side stream: router logits + top-k on the mixed rows || main: shared gate/up gemv, finish, shared down
        on_side(side);
        let gl = self.gate.fwd(&x);
        ffi::moe_route_ld(&gl, r, &idx, &w);
        on_main(side);
        let dd = d.fwd(&o[0]);
        let sh = emp(&[r, D], Kind::Float);
        join_side(side);
        ffi::moe_gate_scale(&gl, 512, &dd, &sh);
        let ws = emp(&[ffi::moe_ws_bytes(r, 4)], Kind::Uint8);
        let out = emp(&[r, D], Kind::Float);
        ffi::moe_experts(&x, r, &idx, &w, &self.tab, Some(&sh), &out, &ws, 4);
        out
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        let r = x.size()[0];
        // QWEN_MOE_DUMP=n: write the input rows of the n-th prefill MoE call to /tmp/moe_x_<n>.f32 (kernel benchmarks)
        if r > MAX_ROWS {
            static CALLS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            let c = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if std::env::var("QWEN_MOE_DUMP").ok().and_then(|v| v.parse::<i64>().ok()) == Some(c) {
                let v: Vec<f32> = Vec::try_from(x.to_device(Device::Cpu).contiguous().view([-1])).unwrap();
                let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
                std::fs::write(format!("/tmp/moe_x_{c}.f32"), b).unwrap();
                eprintln!("[qwen] dumped MoE input {} x 2560 to /tmp/moe_x_{c}.f32", r);
            }
        }
        let [g, u, d] = &self.sh;
        let gl = self.gate.fwd(x);
        let idx = emp(&[r, 10], Kind::Int);
        let w = emp(&[r, 10], Kind::Float);
        // decode rows: the shared expert's elementwise ops in two kernels and the router read in place (QWEN_SHARED_UNFUSED=1,
        // read at graph capture: the torch ops and a contiguous copy of the logits)
        let shared = if r <= dec_rows() && std::env::var("QWEN_SHARED_UNFUSED").as_deref() != Ok("1") {
            let (gg, uu) = (g.fwd(x), u.fwd(x));
            let h = emp(&[r, gg.size()[1]], Kind::Float);
            ffi::moe_silu_mul(&gg, &uu, &h);
            let dd = d.fwd(&h);
            let sh = emp(&[r, D], Kind::Float);
            ffi::moe_gate_scale(&gl, 512, &dd, &sh);
            ffi::moe_route_ld(&gl, r, &idx, &w);
            sh
        } else if fin_fuse(r) && std::env::var("QWEN_PREFILL_SHARED_TORCH").as_deref() != Ok("1") {
            // prefill: one input transform for gate / up, the elementwise ops in the decode kernels (the torch ops' expressions:
            // bitwise equal; QWEN_PREFILL_SHARED_TORCH=1: torch ops), the router read in place
            let p = big_parts(&[g, u], x);
            let (gg, uu) = (g.finish_big(&p[0]), u.finish_big(&p[1]));
            let h = emp(&[r, gg.size()[1]], Kind::Float);
            ffi::moe_silu_mul(&gg, &uu, &h);
            let dd = d.fwd(&h);
            let sh = emp(&[r, D], Kind::Float);
            ffi::moe_gate_scale(&gl, 512, &dd, &sh);
            ffi::moe_route_ld(&gl, r, &idx, &w);
            sh
        } else {
            let sh = (gl.narrow(1, 512, 1).sigmoid() * d.fwd(&(g.fwd(x).silu() * u.fwd(x)))).contiguous();
            ffi::moe_route(&gl.narrow(1, 0, 512).contiguous(), r, &idx, &w);
            sh
        };
        let s = if r > dec_rows() { 1 } else { 4 };
        let ws = emp(&[ffi::moe_ws_bytes(r, s)], Kind::Uint8);
        let out = emp(&[r, D], Kind::Float);
        ffi::moe_experts(x, r, &idx, &w, &self.tab, Some(&shared), &out, &ws, s);
        out
    }
}

/// RAM tiers in front of the n-gram table (39 GB, mmapped: page cache / NVMe). Rows are 122 bytes and random, so a
/// 4 KiB page holds ~33 unrelated rows: per byte, a row-level cache keeps ~16x more useful rows than the page cache.
///   hot  : QWEN_NGRAM_HOT=<file from ngram_hot.py, e.g. models/ngram_hot_2m.bin> (off by default): the most frequent
///          rows on a corpus (2M rows, 244 MB: 65% of the accesses on unseen text), read at start-up
///   cache: QWEN_NGRAM_CACHE_MB (default 0 = off): rows read from the table at run time, 2-way set associative
/// Off by default: with the parallel table reads in `pack` a fully cold 12K-token prefill spends 0.21 s packing rows
/// (2.7% of it) and the tiers did not shorten that (0.25 s), at 0.75 GB of RAM. They cut SSD reads by 34-72%, which
/// helps only when the page cache is squeezed.
struct NgramTiers { hot_ids: Vec<u32>, hot: Vec<u8>, sets: usize, tags: Vec<[u32; 2]>, age: Vec<u8>, data: Vec<u8>, rb: usize,
                    stats: [u64; 4] }
impl NgramTiers {
    fn open(rb: usize) -> Self {
        let mut t = NgramTiers { hot_ids: vec![], hot: vec![], sets: 0, tags: vec![], age: vec![], data: vec![], rb, stats: [0; 4] };
        let path = std::env::var("QWEN_NGRAM_HOT").unwrap_or_default();
        if !path.is_empty() && path != "0" {
            if let Ok(b) = std::fs::read(&path) {
                assert!(&b[..4] == b"QNGH", "n-gram hot file magic");
                let n = u64::from_le_bytes(b[4..12].try_into().unwrap()) as usize;
                assert_eq!(u64::from_le_bytes(b[12..20].try_into().unwrap()) as usize, rb, "n-gram hot file row size");
                t.hot_ids = b[20..20 + 4 * n].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                t.hot = b[20 + 4 * n..20 + 4 * n + n * rb].to_vec();
                eprintln!("[qwen] n-gram hot rows: {} ({:.0} MB) from {}", n, (n * rb) as f64 / 1e6, path);
            }
        }
        let mb: usize = std::env::var("QWEN_NGRAM_CACHE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        t.sets = mb * 1_000_000 / rb / 2;
        if t.sets > 0 {
            t.tags = vec![[u32::MAX; 2]; t.sets];
            t.age = vec![0; t.sets];
            t.data = vec![0; t.sets * 2 * rb];
        }
        t
    }
    fn hot_row(&self, row: u32) -> Option<&[u8]> {
        self.hot_ids.binary_search(&row).ok().map(|i| &self.hot[i * self.rb..(i + 1) * self.rb])
    }
    fn set_of(&self, row: u32) -> usize { ((row as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 32) as usize % self.sets }
    fn cached(&mut self, row: u32) -> Option<&[u8]> {
        if self.sets == 0 { return None; }
        let s = self.set_of(row);
        let w = self.tags[s].iter().position(|&t| t == row)?;
        self.age[s] = w as u8 ^ 1;   // the other way is the eviction candidate
        Some(&self.data[(s * 2 + w) * self.rb..(s * 2 + w + 1) * self.rb])
    }
    fn insert(&mut self, row: u32, src: &[u8]) {
        if self.sets == 0 { return; }
        let s = self.set_of(row);
        if self.tags[s].contains(&row) { return; }
        let w = self.age[s] as usize;
        self.tags[s][w] = row;
        self.age[s] = w as u8 ^ 1;
        self.data[(s * 2 + w) * self.rb..(s * 2 + w + 1) * self.rb].copy_from_slice(src);
    }
}

pub struct Ple {
    kp: F16Lin, vp: F16Lin, nk1: Tensor, nq1: Tensor, nc1: Tensor, conv: Tensor, bias: Tensor,
    mult: [i64; 3], off: [i64; 16], size: [i64; 16], eos: i64, bits: i32, words: i64, table: (*const u8, usize),
    tiers: std::sync::Mutex<NgramTiers>,
}
unsafe impl Send for Ple {}
unsafe impl Sync for Ple {}
impl Ple {
    fn load(ck: &Checkpoint, key: &str, eos: i64) -> Self {
        let pre = format!("{key}.ple_embedding.ngram_embedding.");
        let i64s = |n: &str| -> Vec<i64> { Vec::<i64>::try_from(ck.get_on(&format!("{pre}{n}"), Device::Cpu)).unwrap() };
        let (m, o, s) = (i64s("layer_multipliers"), i64s("head_offsets"), i64s("head_vocab_sizes"));
        let tk = format!("{pre}trellis");
        ck.advise_random(&tk);
        let words = ck.tensors[&tk].shape[1];
        let g1 = |n: &str| (f32(&ck.get(&format!("{key}.{n}.weight"))) + 1.0).contiguous();
        Ple { kp: F16Lin::tagged(ck.get(&format!("{key}.key_proj.weight")), "ple"),
              vp: F16Lin::tagged(ck.get(&format!("{key}.value_proj.weight")), "ple"),
              nk1: g1("norm_key"), nq1: g1("norm_query"), nc1: g1("norm_conv"),
              conv: ck.get(&format!("{key}.conv1d.weight")).contiguous(), bias: ck.get(&format!("{pre}head_bias")).contiguous(),
              mult: [m[0], m[1], m[2]], off: o.try_into().unwrap(), size: s.try_into().unwrap(), eos,
              bits: 6, words, table: ck.raw_ptr(&tk), tiers: std::sync::Mutex::new(NgramTiers::open((words * 2) as usize)) }
    }
    /// Global table rows [rows, 16] for positions [p0, p0 + n), the token at position i given by `id(i)`: only positions
    /// t - 2..t are read (token t - s belongs to t's segment unless an EOS lies in t - s..t - 1).
    fn rows_at(&self, id: impl Fn(usize) -> i64, p0: usize, n: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity(n * 16);
        for t in p0..p0 + n {
            let sh = |s: usize| if t >= s && (t - s..t).all(|i| id(i) != self.eos) { id(t - s) } else { self.eos };
            let (t0, t1, t2) = (sh(0), sh(1), sh(2));
            let bi = t0.wrapping_mul(self.mult[0]) ^ t1.wrapping_mul(self.mult[1]);
            let tri = bi ^ t2.wrapping_mul(self.mult[2]);
            for h in 0..16 {
                let v = if h < 8 { bi } else { tri };
                out.push(v.rem_euclid(self.size[h]) + self.off[h]);
            }
        }
        out
    }
    /// Host side: the 16 n-gram ring rows of the given table rows, packed. The table (39 GB) stays in the page cache / on
    /// disk and the rows are random: all their pages are requested at once (MADV_WILLNEED queues the reads) and copied in
    /// parallel, so missing pages load concurrently instead of one fault at a time.
    fn pack_rows(&self, rows: Vec<i64>) -> Vec<u8> {
        use rayon::prelude::*;
        extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
        let rb = (self.words * 2) as usize;
        let base = self.table.0 as usize;
        static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SERIAL.get_or_init(|| std::env::var("QWEN_PLE_SERIAL").is_ok()) {   // A/B: one fault at a time
            return rows.iter().flat_map(|&r| unsafe { std::slice::from_raw_parts((base + r as usize * rb) as *const u8, rb) }.to_vec()).collect();
        }
        // RAM tiers first (hot rows, run-time cache), then the table for the rest
        let t_start = std::time::Instant::now();
        let mut host = vec![0u8; rows.len() * rb];
        let mut miss = Vec::new();
        {
            let mut t = self.tiers.lock().unwrap();
            for (i, &row) in rows.iter().enumerate() {
                let r = row as u32;
                if let Some(src) = t.hot_row(r) { host[i * rb..(i + 1) * rb].copy_from_slice(src); t.stats[0] += 1; continue; }
                if let Some(src) = t.cached(r) { host[i * rb..(i + 1) * rb].copy_from_slice(src); t.stats[1] += 1; continue; }
                miss.push(i);
            }
            t.stats[2] += miss.len() as u64;
        }
        if miss.is_empty() { self.tiers.lock().unwrap().stats[3] += t_start.elapsed().as_nanos() as u64; return host; }
        let mut mpages: Vec<usize> = miss.iter().flat_map(|&i| {
            let a = base + rows[i] as usize * rb;
            [a & !4095, (a + rb - 1) & !4095]
        }).collect();
        mpages.sort_unstable();
        mpages.dedup();
        for &pg in &mpages { unsafe { madvise(pg as *mut _, 4096, 3 /* MADV_WILLNEED */); } }
        let got: Vec<(usize, Vec<u8>)> = miss.par_iter().with_min_len(4).map(|&i| {
            let src = unsafe { std::slice::from_raw_parts((base + rows[i] as usize * rb) as *const u8, rb) };
            (i, src.to_vec())
        }).collect();
        let mut t = self.tiers.lock().unwrap();
        for (i, v) in &got {
            host[i * rb..(i + 1) * rb].copy_from_slice(v);
            t.insert(rows[*i] as u32, v);
        }
        t.stats[3] += t_start.elapsed().as_nanos() as u64;
        host
    }

    /// (hot, cache, table) row lookups and pack nanoseconds so far
    pub fn tier_stats(&self) -> [u64; 4] { self.tiers.lock().unwrap().stats }
    /// X += PLE(X) for the rows whose packed rings are in `packed` (device), segment g against window wins[g];
    /// returns the normed rows (window commit).
    fn apply(&self, x: &Tensor, packed: &Tensor, rows: &[(i64, i64)], wins: &[&Tensor]) -> Tensor {
        let r = x.size()[0];
        let emb = emp(&[r, D], Kind::Float);
        ffi::ple_decode(&packed, self.words, self.bits, &self.bias, &emb, r);
        let key = self.kp.fwd(&emb);
        let value = self.vp.fwd(&emb);
        let gated = emp(&[r, H * D], Kind::Float);
        let nrm = emp(&[r, H * D], Kind::Float);
        for (&(r0, t), win) in rows.iter().zip(wins) {
            let n = |a: &Tensor| a.narrow(0, r0, t);
            ffi::ple_apply(&n(x), &n(&key), &n(&value), &self.nk1, &self.nq1, &self.nc1, &self.conv, win, &n(&gated), &n(&nrm), t);
        }
        nrm
    }
}

pub enum Attn { Gdn(Gdn), Qsa(Qsa) }

type GdnPending = Option<(Tensor, Tensor, Tensor, Tensor)>;

/// One decoder block on the stream stack x [R, 4, 2560] (updated in place): attn HC mix, GDN | QSA,
/// apply, MLP HC mix, MoE, apply. Segment g = rows [r0, r0 + t) of sequence state sts[g] at position pos[g]
/// (device int [segments]). Returns the GDN inputs to replay when `commit` is false.
fn block(l: &Layer, sts: &[&LayerState], x: &Tensor, rows: &[(i64, i64)], pos: &Tensor, commit: bool) -> GdnPending {
    let (pending, upd) = block_pend(l, sts, x, rows, pos, commit, None);
    flush_apply(x, upd);
    pending
}

/// `block` with the previous block's residual update `apply` fused into this block's attn HC norm, and this block's MoE
/// update returned (not yet added to x). Decode rows: the HC mixes are left to their consumers (MixIn).
fn block_pend(l: &Layer, sts: &[&LayerState], x: &Tensor, rows: &[(i64, i64)], pos: &Tensor, commit: bool,
              apply: Option<(&Tensor, &Tensor)>) -> (GdnPending, PendApply) {
    let (pending, post, m_in) = block_attn_in(l, sts, x, rows, pos, commit, apply);
    let m = l.moe.fwd_in(m_in);
    if let Some(i) = l.idx { crate::ablate::apply(i, &m); }
    (pending, Some((post, m)))
}

/// `block_attn` preceded by the previous block's residual update X += post * m (fused into the attn HC norm).
fn block_attn_after(l: &Layer, sts: &[&LayerState], x: &Tensor, rows: &[(i64, i64)], pos: &Tensor, commit: bool,
                    apply: Option<(&Tensor, &Tensor)>) -> (GdnPending, Tensor, Tensor) {
    let (pending, post, m_in) = block_attn_in(l, sts, x, rows, pos, commit, apply);
    (pending, post, m_in.ready())
}

/// `block_attn_after` with the MoE input as a MixIn (decode rows: mixed by the MoE's first launch).
fn block_attn_in(l: &Layer, sts: &[&LayerState], x: &Tensor, rows: &[(i64, i64)], pos: &Tensor, commit: bool,
                 apply: Option<(&Tensor, &Tensor)>) -> (GdnPending, Tensor, MixIn) {
    let r = x.size()[0];
    let (post, a_in) = l.hca.mix_after_in(x, apply);
    // refusal-direction extraction (crate::ablate): the block's mixed input, last row
    let a_in = match (crate::ablate::capturing(), l.idx) {
        (true, Some(i)) => { let m = a_in.ready(); crate::ablate::capture(i, &m); MixIn::Ready(m) }
        _ => a_in,
    };
    let multi = multi_on(r);
    let mut pending = None;
    let a = match &l.attn {
        Attn::Gdn(g) => {
            let (ss, cs): (Vec<&Tensor>, Vec<&Tensor>) =
                sts.iter().map(|st| match st { LayerState::Gdn { s, conv } => (s, conv), _ => unreachable!() }).unzip();
            let (ss, cs) = (ffi::Segs::new(rows, &ss), ffi::Segs::new(rows, &cs));
            if !multi && commit && fin_fuse(r) {
                // prefill: qkv's finish inside the conv and its commit, z's inside the output norm (bitwise equal)
                let a_in = a_in.ready();
                let parts = big_parts(&[&g.qkv, &g.z], &a_in);
                let ab = g.ab.fwd(&a_in);
                let y = emp(&[r, 10240], Kind::Float);
                let out = emp(&[r, 6144], Kind::Float);
                // the recurrence's prep inside the conv too (QWEN_PREFILL_PREP_SPLIT=1, read per call: separate prep)
                if std::env::var("QWEN_PREFILL_PREP_SPLIT").as_deref() == Ok("1") {
                    ffi::gdn_conv_h(&parts[0], &g.qkv.svh, &cs, &g.conv, &y);
                    ffi::gdn_recur_segs_z(&y, &ab, &parts[1], &g.z.svh, &g.a_log, &g.dt_bias, &g.norm, &ss, &out, true);
                } else {
                    let ws = emp(&[ffi::gdn_ws_bytes(r)], Kind::Uint8);
                    ffi::gdn_conv_hq(&parts[0], &g.qkv.svh, &cs, &g.conv, &y, &ab, &g.a_log, &g.dt_bias, &ws);
                    ffi::gdn_recur_segs_zq(&y, &ab, &parts[1], &g.z.svh, &g.a_log, &g.dt_bias, &g.norm, &ss, &out, true, &ws);
                }
                ffi::gdn_conv_commit_h(&parts[0], &g.qkv.svh, &cs);
                let o = g.o.fwd(&out);
                if let Some(i) = l.idx { crate::ablate::apply(i, &o); }
                let (post2, m_in) = l.hcm.mix_after_in(x, Some((post.as_ref().unwrap(), &o)));
                return (None, post2.unwrap(), m_in);
            }
            // decode rows: the input transforms (and the mix) shared, one gemv launch each (2026-10-07: one combined gemv
            // launch was no faster for these two large linears)
            // side stream (QWEN_GDN_AB_SIDE): a/b gemv on the mixed rows || main: qkv / z gemvs, finish, conv
            let side = multi && r <= dec_rows() && side_on("QWEN_GDN_AB_SIDE");
            // (the mixed rows a_in stay alive until the join: the side stream reads them)
            let (a_in, qkv, z, ab) = if multi {
                let (a, mut o) = exl3_multi(&[&g.qkv, &g.z], &a_in, 2 | if side { 4 } else { 0 });
                on_side(side);
                let ab = g.ab.fwd(&a);
                on_main(side);
                let z = o.pop().unwrap();
                (a, o.pop().unwrap(), z, ab)
            } else {
                let a_in = a_in.ready();
                // many rows with folded EXL3 weights: the projections share one fp16 copy of their input
                let a_in = if r > BIG_ROWS && fold_on() { a_in.to_kind(Kind::Half) } else { a_in };
                let ab = g.ab.fwd(&a_in);
                (a_in.shallow_clone(), g.qkv.fwd(&a_in), g.z.fwd(&a_in), ab)
            };
            let y = emp(&[r, 10240], Kind::Float);
            let out = emp(&[r, 6144], Kind::Float);
            ffi::gdn_conv_segs(&qkv, &cs, &g.conv, &y);
            join_side(side);
            drop(a_in);
            ffi::gdn_recur_segs(&y, &ab, &z, &g.a_log, &g.dt_bias, &g.norm, &ss, None, Some(&out), commit);
            if commit { ffi::gdn_conv_commit_segs(&qkv, &cs, None); } else { pending = Some((qkv, y, ab, z)); }
            let o = g.o.fwd(&out);
            if let Some(i) = l.idx { crate::ablate::apply(i, &o); }
            o
        }
        Attn::Qsa(q) => {
            // prefill: one input transform for the four, q_proj's finish inside qsa_prep (bitwise equal)
            let mut qpart = None;
            let (qp, kp, vp, ip) = if multi {
                let (_, o) = exl3_multi(&[&q.q, &q.k, &q.v, &q.idx], &a_in, 0);
                (o[0].shallow_clone(), o[1].shallow_clone(), o[2].shallow_clone(), o[3].shallow_clone())
            } else if fin_fuse(r) {
                let a_in = a_in.ready();
                let p = big_parts(&[&q.q, &q.k, &q.v, &q.idx], &a_in);
                qpart = Some(p[0].shallow_clone());
                (p[0].shallow_clone(), q.k.finish_big(&p[1]), q.v.finish_big(&p[2]), q.idx.finish_big(&p[3]))
            } else {
                let a_in = a_in.ready();
                let a_in = if r > BIG_ROWS && fold_on() { a_in.to_kind(Kind::Half) } else { a_in };
                (q.q.fwd(&a_in), q.k.fwd(&a_in), q.v.fwd(&a_in), q.idx.fwd(&a_in))
            };
            let out = emp(&[r, 6144], Kind::Float);
            for (g, (&(r0, t), st)) in rows.iter().zip(sts).enumerate() {
                let LayerState::Qsa { kc, ks, vc, vs, pooled, ring, yarn, mtab, mctl } = st else { unreachable!() };
                let yarn = *yarn;
                let n = |a: &Tensor| a.narrow(0, r0, t);
                let p0 = pos.narrow(0, g as i64, 1);
                let qq = emp(&[t, 24, 256], Kind::Float);
                let gate = emp(&[t, 24, 256], Kind::Float);
                let qi = emp(&[t, 4, 128], Kind::Float);
                // pool before prep: a group straddling the chunk start reads its earlier raw keys from the ring, which prep
                // overwrites with the chunk's last 32 rows (prefill chunks of >= 29 rows starting off a multiple of 4: after a
                // reused prefix, the MTP catch-up)
                ffi::qsa_pool(ring, &n(&ip), t, &p0, &q.ikn, pooled, yarn, mtab, mctl);
                match &qpart {
                    Some(qh) => ffi::qsa_prep_h(&n(qh), &q.q.svh, &n(&kp), &n(&vp), &n(&ip), t, &p0, &q.qn, &q.kn, &q.iqn, &qq, &gate, kc, ks, vc, vs, &qi, ring, yarn, mtab, mctl),
                    None => ffi::qsa_prep(&n(&qp), &n(&kp), &n(&vp), &n(&ip), t, &p0, &q.qn, &q.kn, &q.iqn, &qq, &gate, kc, ks, vc, vs, &qi, ring, yarn, mtab, mctl),
                }
                let ld = pooled.size()[0];
                let sel = emp(&[t, SEL_LD], Kind::Int);
                let cnt = emp(&[t], Kind::Int);
                // indexer scores of at most 32M floats at a time (a long context's 2048-row prefill chunk would need GBs)
                let rb = ((32i64 << 20) / ld.max(1)).clamp(1, t);
                let scores = emp(&[rb, ld], Kind::Float);
                let mut a = 0;
                while a < t {
                    let m = rb.min(t - a);
                    let pa = if a == 0 { p0.shallow_clone() } else { &p0 + a };
                    ffi::qsa_select(&qi.narrow(0, a, m), pooled, m, &pa, &scores, ld, &sel.narrow(0, a, m), &cnt.narrow(0, a, m), SEL_LD);
                    a += m;
                }
                let splits = (2051 + 127) / 128;
                let ml = emp(&[t * 24 * splits * 2], Kind::Float);
                let acc = emp(&[t * 24 * splits * 256], Kind::Float);
                ffi::qsa_attn(&qq, t, &p0, Some((&sel, &cnt, SEL_LD)), kc, ks, vc, vs, &gate, &ml, &acc, splits, &n(&out));
            }
            let o = q.o.fwd(&out);
            if let Some(i) = l.idx { crate::ablate::apply(i, &o); }
            o
        }
    };
    let (post, m_in) = l.hcm.mix_after_in(x, Some((post.as_ref().unwrap(), &a)));
    (pending, post.unwrap(), m_in)
}

/// The previous layer's MoE residual update (post [T, 4], m [T, 2560]) not yet added to the stream stack.
pub type PendApply = Option<(Tensor, Tensor)>;

/// Add a pending update to x now.
fn flush_apply(x: &Tensor, pend: PendApply) {
    if let Some((post, m)) = pend { ffi::hc_apply(x, &post, &m, x.size()[0]); }
}

/// QWEN_PREFILL_APPLY_SPLIT=1: apply each layer's MoE update right away (no fusion into the next layer's norm).
fn apply_split() -> bool { std::env::var("QWEN_PREFILL_APPLY_SPLIT").as_deref() == Ok("1") }

/// Layer-major prefill of one layer for x [T, 4, 2560] (one sequence, positions p0..p0 + T, state committed):
/// the attention half runs in sub-chunks of `sub` rows in order (recurrent state, KV cache, PLE window), the
/// layer's decoded dense weights are reused across them, then the MoE runs on all T rows at once (each expert's
/// weights read once for the pass, its rows fill the GEMM row tiles). Per row the same operations as `block`.
/// `pend` (the previous layer's MoE update) is fused into this layer's attn HC norm (bitwise equal to applying it
/// first; a PLE layer reads x before that and applies it first); returns this layer's MoE update, pending.
fn layer_prefill(l: &Layer, st: &LayerState, ple: Option<(&Tensor, i64, &Tensor)>, x: &Tensor, pos: &Tensor, sub: i64, pend: PendApply) -> PendApply {
    let t = x.size()[0];
    INNER.with(|c| *c.borrow_mut() = Some(Default::default()));
    let (post_all, min_all) = layer_attn(l, st, ple, x, pos, sub, pend);
    INNER.with(|c| *c.borrow_mut() = None);
    let mr = prefill_moe_rows();
    let m_all = if mr >= t { l.moe.fwd(&min_all) } else {
        let m_all = emp(&[t, D], Kind::Float);
        let mut a = 0;
        while a < t {
            let n = mr.min(t - a);
            m_all.narrow(0, a, n).shallow_clone().copy_(&l.moe.fwd(&min_all.narrow(0, a, n)));
            a += n;
        }
        m_all
    };
    if let Some(i) = l.idx { crate::ablate::apply(i, &m_all); }
    let out = Some((post_all, m_all));
    if apply_split() { flush_apply(x, out); None } else { out }
}

/// The attention half of `layer_prefill` (PLE, attn HC, GDN | QSA, MLP HC) for rows x (positions p0..), the pending
/// update of the previous layer applied first; returns (MLP post [T, 4], MoE input [T, 2560]). The caller keeps the
/// layer's INNER cache.
/// `pos`: device int positions of the sub-chunks (p0, p0 + sub, ..: made on the device once per pass, no host copies).
fn layer_attn(l: &Layer, st: &LayerState, ple: Option<(&Tensor, i64, &Tensor)>, x: &Tensor, pos: &Tensor, sub: i64, pend: PendApply) -> (Tensor, Tensor) {
    let t = x.size()[0];
    let pend = if l.ple.is_some() || apply_split() { flush_apply(x, pend); None } else { pend };
    if let (Some(p), Some((packed, rb, win))) = (&l.ple, ple) {
        let mut a = 0;
        while a < t {
            let n = sub.min(t - a);
            let nrm = p.apply(&x.narrow(0, a, n), &packed.narrow(0, a * rb, n * rb), &[(0, n)], &[win]);
            // the PLE kernel adds into the residual streams directly: keep them orthogonal to an ablated direction
            if let Some(i) = l.idx { crate::ablate::apply(i, &x.narrow(0, a, n)); }
            ffi::ple_commit(win, &nrm, n, None);
            a += n;
        }
    }
    let post_all = emp(&[t, H], Kind::Float);
    let min_all = emp(&[t, D], Kind::Float);
    let mut a = 0;
    while a < t {
        let n = sub.min(t - a);
        let pos = pos.narrow(0, a / sub, 1);
        let ap = pend.as_ref().map(|(po, m)| (po.narrow(0, a, n), m.narrow(0, a, n)));
        let (_, post, m_in) = block_attn_after(l, &[st], &x.narrow(0, a, n), &[(0, n)], &pos, true, ap.as_ref().map(|(p, m)| (p, m)));
        post_all.narrow(0, a, n).shallow_clone().copy_(&post);
        min_all.narrow(0, a, n).shallow_clone().copy_(&m_in);
        a += n;
    }
    (post_all, min_all)
}

fn qsa_state(cap: i64, mtab: &Tensor, mctl: &Tensor) -> LayerState {
    let d = Device::Cuda(0);
    LayerState::Qsa {
        kc: Tensor::zeros([cap, 2, 256], (Kind::Uint8, d)), ks: Tensor::ones([cap, 2], (Kind::Float, d)),
        vc: Tensor::zeros([cap, 2, 256], (Kind::Uint8, d)), vs: Tensor::ones([cap, 2], (Kind::Float, d)),
        pooled: Tensor::zeros([cap / 4, 128], (Kind::BFloat16, d)), ring: Tensor::zeros([32, 128], (Kind::Float, d)), yarn: yarn_for(cap),
        mtab: mtab.shallow_clone(), mctl: mctl.shallow_clone() }
}

/// Native context of the checkpoint (max_position_embeddings).
pub const NATIVE_CTX: i64 = 262144;
/// RoPE mode of a sequence with room for `cap` positions: YaRN (factor 4, as the model card recommends up to 1M tokens) above the
/// native 262144, so shorter sequences keep the plain RoPE (static YaRN costs some quality on short texts). QWEN_YARN=1 / 0:
/// always / never.
pub fn yarn_for(cap: i64) -> i32 {
    match std::env::var("QWEN_YARN").as_deref() { Ok("1") => 1, Ok("0") => 0, _ => (cap > NATIVE_CTX) as i32 }
}

/// Attention caches of all sequence stores (as GLM's KV pool): one allocation per QSA layer (and the MTP layer) of `tokens`
/// positions; a store takes a contiguous range, so one sequence can use any part of the budget up to all of it.
pub struct KvPool { pub tokens: i64, layers: Vec<Option<[Tensor; 5]>>, mtp: Option<[Tensor; 5]>, mtab: Tensor }
fn pool_layer(tokens: i64) -> [Tensor; 5] {
    let d = Device::Cuda(0);
    [Tensor::zeros([tokens, 2, 256], (Kind::Uint8, d)), Tensor::ones([tokens, 2], (Kind::Float, d)),
     Tensor::zeros([tokens, 2, 256], (Kind::Uint8, d)), Tensor::ones([tokens, 2], (Kind::Float, d)),
     Tensor::zeros([tokens / 4, 128], (Kind::BFloat16, d))]
}
fn pool_view(p: &[Tensor; 5], base: i64, len: i64, mtab: &Tensor, mctl: &Tensor) -> LayerState {
    LayerState::Qsa { kc: p[0].narrow(0, base, len), ks: p[1].narrow(0, base, len), vc: p[2].narrow(0, base, len), vs: p[3].narrow(0, base, len),
                      pooled: p[4].narrow(0, base / 4, len / 4), ring: Tensor::zeros([32, 128], (Kind::Float, Device::Cuda(0))),
                      yarn: yarn_for(len - 16), mtab: mtab.shallow_clone(), mctl: mctl.shallow_clone() }
}

/// MTP draft head: one qwen4_exp block (QSA + MoE) on e' + h'_s, own KV cache, own collapse mixer;
/// row i sits at the target position i and reads (target stack at i, token i + 1).
pub struct Mtp { fc_e: Exl3, fc_h: Exl3, ne1: Tensor, nh1: Tensor, layer: Layer, mixer: Hc }
impl Mtp {
    fn load(ck: &Checkpoint) -> Self {
        let g1 = |k: &str| (f32(&ck.get(k)) + 1.0).contiguous();
        let key = "mtp.layers.0";
        Mtp { fc_e: Exl3::load(ck, "mtp.fc_embedding"), fc_h: Exl3::load(ck, "mtp.fc_hidden"),
              ne1: g1("mtp.pre_fc_norm_embedding.weight"), nh1: g1("mtp.pre_fc_norm_hidden.weight"),
              layer: Layer { idx: None, ple: None, hca: Hc::load(ck, &format!("{key}.attn_hyper_connection"), true),
                             attn: Attn::Qsa(Qsa::load(ck, &format!("{key}.self_attn"))),
                             hcm: Hc::load(ck, &format!("{key}.mlp_hyper_connection"), true), moe: Moe::load(ck, &format!("{key}.mlp")) },
              mixer: Hc::load(ck, "mtp.hyper_connection_mixer", false) }
    }
}
pub struct Layer { ple: Option<Ple>, hca: Hc, attn: Attn, hcm: Hc, moe: Moe, idx: Option<usize> }

/// Draft-side output head (L2: only the MTP drafts use it): the lm_head columns of a frequent sub-vocabulary
/// (sorted token ids) as int8 + FP32 scale per 128 (GLM's Q8 GEMV) or, with q4, affine 4-bit + (scale, minimum)
/// per 128 (GLM's INT4 draft GEMM).
/// QWEN_DRAFT_Q4_TILED (default on, read at load; L0): the Q4 copy in the tiled layout (shim/draft_q4.cu: contiguous 512 B
/// warp loads, bitwise equal logits).
pub struct DraftHead { pub keep: Vec<i64>, keep_d: Tensor, q: Tensor, s: Tensor, q4: bool, tiled: bool }
extern "C" {
    fn rs_q8_encode(w: *const std::ffi::c_void, n: i32, k: i32, mse: i32, s: *mut f32, q: *mut std::ffi::c_void, err: *mut f64) -> i32;
    fn rs_q8_gemm(x: *const f32, q: *const u8, s: *const f32, n: i32, k: i32, y: *mut std::ffi::c_void, m: i32, out: i32, ks: i32) -> i32;
    fn glm53_draft_q4_encode_cuda(w: *const std::ffi::c_void, n: i32, k: i32, mse: i32, sm: *mut std::ffi::c_void, q: *mut std::ffi::c_void, s: *mut std::ffi::c_void) -> i32;
    fn glm53_draft_q4_gemm2_cuda(x: *const std::ffi::c_void, xf: i32, q: *const std::ffi::c_void, sm: *const std::ffi::c_void, y: *mut std::ffi::c_void,
                                 yb: i32, m: i32, n: i32, k: i32, s: *mut std::ffi::c_void) -> i32;
    fn glm53_draft_q4t_gemm2_cuda(x: *const std::ffi::c_void, xf: i32, q: *const std::ffi::c_void, sm: *const std::ffi::c_void, y: *mut std::ffi::c_void,
                                  yb: i32, m: i32, n: i32, k: i32, s: *mut std::ffi::c_void) -> i32;
    fn glm53_draft_q4t_tile_cuda(q: *const std::ffi::c_void, sm: *const std::ffi::c_void, n: i32, k: i32, qt: *mut std::ffi::c_void,
                                 smt: *mut std::ffi::c_void, s: *mut std::ffi::c_void) -> i32;
    fn rs_current_stream() -> *mut std::ffi::c_void;
}
impl DraftHead {
    fn build(lm: &Exl3, mut keep: Vec<i64>, q4: bool, tiled: bool) -> Self {
        let _g = tch::no_grad_guard();
        // the encoders take multiples of 16 rows: pad with the lowest unused token ids
        if keep.len() % 16 != 0 {
            let have: std::collections::HashSet<i64> = keep.iter().copied().collect();
            let pad: Vec<i64> = (0..lm.n).filter(|t| !have.contains(t)).take(16 - keep.len() % 16).collect();
            keep.extend(pad);
            keep.sort_unstable();
        }
        let dev = Device::Cuda(0);
        // effective weight columns of the kept vocabulary, lm_head(I) = W [2560, V], in blocks of 16384 columns
        let eye = Tensor::eye(D, (Kind::Float, dev));
        let n = keep.len() as i64;
        let sub = Tensor::empty([n, D], (Kind::BFloat16, dev));
        let (mut ki, mut c0) = (0usize, 0i64);
        while c0 < lm.n {
            let nc = 16384.min(lm.n - c0);
            let end = ki + keep[ki..].partition_point(|&t| t < c0 + nc);
            if end > ki {
                let y = lm.fwd_cols(&eye, c0, nc);
                let local = Tensor::from_slice(&keep[ki..end].iter().map(|&t| t - c0).collect::<Vec<_>>()).to_device(dev);
                sub.narrow(0, ki as i64, (end - ki) as i64).shallow_clone().copy_(&y.index_select(1, &local).tr());
                ki = end;
            }
            c0 += nc;
        }
        assert_eq!(ki, keep.len(), "draft vocab ids beyond lm_head");
        let (q, s) = if q4 {
            let sm = Tensor::empty([n, D / 128, 2], (Kind::Float, dev));
            let q = Tensor::empty([n, D / 2], (Kind::Uint8, dev));
            assert_eq!(unsafe { glm53_draft_q4_encode_cuda(sub.data_ptr(), n as i32, D as i32, 1, sm.data_ptr(), q.data_ptr(), rs_current_stream()) }, 0, "draft head Q4 encode");
            eprintln!("[qwen] draft head: {} tokens, Q4 ({} MB)", n, n * D / 2 / 1_000_000);
            (q, sm)
        } else {
            let s = Tensor::empty([n, D / 128], (Kind::Float, dev));
            let q = Tensor::empty([n, D], (Kind::Uint8, dev));
            let err = Tensor::zeros([2], (Kind::Double, dev));
            assert_eq!(unsafe { rs_q8_encode(sub.data_ptr(), n as i32, D as i32, 1, s.data_ptr().cast(), q.data_ptr(), err.data_ptr().cast()) }, 0, "draft head Q8 encode");
            eprintln!("[qwen] draft head: {} tokens, Q8 relative rms error {:.4}", n, (err.double_value(&[0]) / err.double_value(&[1])).sqrt());
            (q, s)
        };
        let tiled = q4 && tiled;
        let (q, s) = if tiled {
            let (qt, st) = (q.empty_like(), s.empty_like());
            assert_eq!(unsafe { glm53_draft_q4t_tile_cuda(q.data_ptr(), s.data_ptr(), n as i32, D as i32, qt.data_ptr(), st.data_ptr(), rs_current_stream()) }, 0,
                       "draft head Q4 tile");
            (qt, st)
        } else { (q, s) };
        let keep_d = Tensor::from_slice(&keep).to_device(dev);
        DraftHead { keep, keep_d, q, s, q4, tiled }
    }
    /// x [M, 2560] fp32 -> logits over the kept vocabulary [M, Vd]
    fn fwd(&self, x: &Tensor) -> Tensor {
        let m = x.size()[0];
        let n = self.keep.len() as i64;
        let y = emp(&[m, n], Kind::Float);
        let x = x.contiguous();
        if self.q4 {
            let gemm = if self.tiled { glm53_draft_q4t_gemm2_cuda } else { glm53_draft_q4_gemm2_cuda };
            assert_eq!(unsafe { gemm(x.data_ptr(), 1, self.q.data_ptr(), self.s.data_ptr(), y.data_ptr(), 0, m as i32, n as i32, D as i32,
                                     rs_current_stream()) }, 0, "draft head Q4 GEMM");
        } else {
            assert_eq!(unsafe { rs_q8_gemm(x.data_ptr().cast(), self.q.data_ptr().cast(), self.s.data_ptr().cast(), n as i32, D as i32, y.data_ptr(), m as i32, 0, 4) }, 0, "draft head Q8 GEMM");
        }
        y
    }
}

/// Rows of an MTP forward that get logits: all, the last, or device int64 row indices.
pub enum Pick<'a> { All, Last, Rows(&'a Tensor) }

pub struct Model { pub layers: Vec<Layer>, embed: Tensor, mixer: Hc, lm_head: Exl3, pub vocab: i64, pub mtp: Option<Mtp>, pub draft_head: Option<DraftHead>,
                   draft_alts: Vec<DraftHead>, pub vision: Option<super::vision::Vision> }

/// Embedding / n-gram ids of prompt ids (salted multimodal placeholders -> the pad tokens).
fn vocab_ids(ids: &[i64]) -> Vec<i64> { ids.iter().map(|&t| super::vision::vocab_id(t)).collect() }

pub enum LayerState {
    Gdn { s: Tensor, conv: Tensor },
    /// yarn: the sequence's RoPE mode (see yarn_for)
    /// mtab / mctl: the sequence's mRoPE table and (table length, text position delta) (see Model::set_mrope)
    Qsa { kc: Tensor, ks: Tensor, vc: Tensor, vs: Tensor, pooled: Tensor, ring: Tensor, yarn: i32, mtab: Tensor, mctl: Tensor },
}
pub struct Seq {
    /// mRoPE: positions [rows, 3] of a multimodal prompt and (its length, text delta) (`set_mrope`)
    pub mtab: Tensor,
    pub mctl: Tensor,
    pub layers: Vec<LayerState>,
    pub ple_win: Tensor,
    pub ids: Vec<i64>,
    pub pos: i64,
    pub cap: i64,
    pub pending: Option<Pending>,
    pub mtp: Option<LayerState>,
}
/// A prompt checkpoint (`Model::ckpt_save`): the prompt ids and copies of the state that is not per position.
pub struct Ckpt { pub ids: Vec<i64>, t: Vec<Tensor> }
impl Ckpt {
    /// The checkpoint's state tensors, in the order ckpt_save writes and ckpt_restore reads them (per layer GDN s/conv
    /// or QSA ring, then ple_win, then the pending MTP row). For the persistent cache (qwen::pcache).
    pub fn tensors(&self) -> &[Tensor] { &self.t }
    /// Rebuild a checkpoint from a persisted (ids, tensors) payload; the tensors must match this model's state shapes.
    pub fn from_parts(ids: Vec<i64>, t: Vec<Tensor>) -> Self { Ckpt { ids, t } }
}
/// Inputs of the last uncommitted chain, replayed by `commit`.
pub struct Pending { gdn: Vec<Option<(Tensor, Tensor, Tensor, Tensor)>>, ple_nrm: Option<Tensor>, ids: Vec<i64> }

impl Model {
    pub fn load(dir: &Path) -> Self {
        let vis = super::vision::enabled();
        let extra: &[&str] = if vis { &["ngram_embedding.safetensors", "vision_k6.safetensors"] } else { &["ngram_embedding.safetensors"] };
        let ck = Checkpoint::open(dir, extra, Device::Cuda(0));
        let c = ck.text_cfg();
        let n = c["num_hidden_layers"].as_i64().unwrap();
        let types: Vec<String> = c["layer_types"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
        let ple_ids: Vec<i64> = c["ple_layer_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let eos = c["eos_token_id"].as_i64().unwrap();
        let mut layers = Vec::new();
        for i in 0..n {
            let key = format!("{P}layers.{i}");
            ck.preload(&format!("{key}."));
            let attn = if types[i as usize] == "linear_attention" { Attn::Gdn(Gdn::load(&ck, &format!("{key}.linear_attn"))) }
                       else { Attn::Qsa(Qsa::load(&ck, &format!("{key}.self_attn"))) };
            let ple = ple_ids.contains(&(i + 1)).then(|| Ple::load(&ck, &format!("{key}.ple"), eos));
            layers.push(Layer { idx: Some(i as usize), ple, hca: Hc::load(&ck, &format!("{key}.attn_hyper_connection"), true), attn,
                                hcm: Hc::load(&ck, &format!("{key}.mlp_hyper_connection"), true), moe: Moe::load(&ck, &format!("{key}.mlp")) });
            ck.clear_cache();
            if i % 8 == 7 { eprintln!("[qwen] loaded {} layers", i + 1); }
        }
        let embed = ck.get(&format!("{P}embed_tokens.weight"));
        let embed = match crate::ablate::get() { Some(a) => a.ortho_rows(0, &embed), None => embed };
        let vocab = embed.size()[0];
        let mixer = Hc::load(&ck, &format!("{P}hyper_connection_mixer"), false);
        let lm_head = Exl3::load(&ck, "lm_head");
        let mtp = (std::env::var("QWEN_MTP").map_or(true, |v| v != "0") && ck.has("mtp.fc_hidden.trellis")).then(|| {
            ck.preload("mtp.");
            let m = Mtp::load(&ck);
            ck.clear_cache();
            m
        });
        let vision = vis.then(|| super::vision::Vision::load(&ck, &ck.cfg));
        // QWEN_KEEP_SHARD_CACHE=1: leave the loaded shards in the page cache
        if std::env::var("QWEN_KEEP_SHARD_CACHE").as_deref() != Ok("1") { ck.release_shards(&["ngram_embedding.safetensors"]); }
        std::mem::forget(ck);   // keep the mmaps (n-gram table rows are read from it)
        let t_dh = std::time::Instant::now();
        let vocab_file = |v: &str| if v.chars().all(|c| c.is_ascii_digit()) { format!("{}/draft_vocab_{v}.json", std::env::var("QWEN_ASSETS").unwrap_or_else(|_| dir.display().to_string())) } else { v.to_string() };
        let load_keep = |f: &str| -> Vec<i64> { serde_json::from_reader(std::fs::File::open(f).unwrap()).expect("draft vocab json") };
        let dv = vocab_file(&std::env::var("QWEN_DRAFT_VOCAB").unwrap_or_else(|_| "65536".into()));
        // L2 (drafts only), 2026-10-07 same-process A/B at one sequence: Q4 +3..7% tok/s over Q8, same acceptance
        let q4 = std::env::var("QWEN_DRAFT_Q4").as_deref() != Ok("0");
        let draft_head = (mtp.is_some() && !dv.is_empty() && !dv.ends_with("_0.json") && Path::new(&dv).exists())
            .then(|| DraftHead::build(&lm_head, load_keep(&dv), q4, std::env::var("QWEN_DRAFT_Q4_TILED").as_deref() != Ok("0")));
        // QWEN_DRAFT_ALTS="q4:65536;q8:49152": extra heads for same-process A/B (QWEN_DRAFT_PICK=i at graph capture);
        // q4 is tiled, q4r the row layout
        let draft_alts: Vec<DraftHead> = std::env::var("QWEN_DRAFT_ALTS").unwrap_or_default().split(';').filter(|a| a.contains(':')).map(|a| {
            let (kind, v) = a.split_once(':').unwrap();
            DraftHead::build(&lm_head, load_keep(&vocab_file(v)), kind.starts_with("q4"), kind == "q4")
        }).collect();
        if draft_head.is_some() { eprintln!("[qwen] draft heads built in {:.1}s", t_dh.elapsed().as_secs_f64()); }
        Model { layers, embed, mixer, lm_head, vocab, mtp, draft_head, draft_alts, vision }
    }

    /// The KV pool for sequence stores (`tokens` a multiple of 4).
    pub fn new_kv_pool(&self, tokens: i64) -> KvPool {
        KvPool { tokens, layers: self.layers.iter().map(|l| matches!(l.attn, Attn::Qsa(_)).then(|| pool_layer(tokens))).collect(),
                 mtp: self.mtp.as_ref().map(|_| pool_layer(tokens)), mtab: Tensor::zeros([tokens, 3], (Kind::Int, Device::Cuda(0))) }
    }

    /// A sequence whose attention caches are positions [base, base + len) of the pool (base, len multiples of 4); it holds
    /// len - 16 positions (the MTP draft steps write up to 16 past them).
    pub fn new_seq_in(&self, pool: &KvPool, base: i64, len: i64) -> Seq {
        assert!(base % 4 == 0 && len % 4 == 0 && base + len <= pool.tokens && len > 16);
        let (mtab, mctl) = (pool.mtab.narrow(0, base, len), Tensor::zeros([2], (Kind::Int, Device::Cuda(0))));
        let layers = self.layers.iter().zip(&pool.layers).map(|(l, p)| match l.attn {
            Attn::Gdn(_) => LayerState::Gdn { s: Tensor::zeros([48, 128, 128], (Kind::Float, Device::Cuda(0))),
                                               conv: Tensor::zeros([10240, 3], (Kind::Float, Device::Cuda(0))) },
            Attn::Qsa(_) => pool_view(p.as_ref().unwrap(), base, len, &mtab, &mctl),
        }).collect();
        Seq { layers, ple_win: Tensor::zeros([9, H * D], (Kind::Float, Device::Cuda(0))), ids: vec![], pos: 0, cap: len - 16, pending: None,
              mtp: pool.mtp.as_ref().map(|p| pool_view(p, base, len, &mtab, &mctl)), mtab, mctl }
    }

    /// Move or extend a pool sequence to positions [base, base + len) (its RoPE mode stays): with `copy` its positional caches
    /// (attention KV, pooled index keys, mRoPE rows) are copied over from the old range first, which must not overlap the new
    /// one; without it the new range starts where the old one does and only grows. The state that is not per position
    /// (GDN states, index-key rings, PLE window, ids) stays as it is. Graphs captured on the old views must be dropped.
    pub fn reseat(&self, seq: &mut Seq, pool: &KvPool, base: i64, len: i64, copy: bool) {
        assert!(base % 4 == 0 && len % 4 == 0 && base + len <= pool.tokens && len > 16);
        let mtab = pool.mtab.narrow(0, base, len);
        let old = seq.mtab.size()[0];
        if copy { mtab.narrow(0, 0, old.min(len)).copy_(&seq.mtab.narrow(0, 0, old.min(len))); }
        let move_layer = |st: &mut LayerState, p: &[Tensor; 5], mtab: &Tensor| {
            let LayerState::Qsa { kc, ks, vc, vs, pooled, ring, yarn, mctl, .. } = st else { unreachable!() };
            let LayerState::Qsa { kc: k2, ks: s2, vc: v2, vs: w2, pooled: p2, .. } = pool_view(p, base, len, mtab, mctl) else { unreachable!() };
            if copy {
                let n = kc.size()[0].min(len);
                for (a, b) in [(&k2, &*kc), (&s2, &*ks), (&v2, &*vc), (&w2, &*vs)] { a.narrow(0, 0, n).copy_(&b.narrow(0, 0, n)); }
                p2.narrow(0, 0, n / 4).copy_(&pooled.narrow(0, 0, n / 4));
            }
            *st = LayerState::Qsa { kc: k2, ks: s2, vc: v2, vs: w2, pooled: p2, ring: ring.shallow_clone(), yarn: *yarn,
                                    mtab: mtab.shallow_clone(), mctl: mctl.shallow_clone() };
        };
        for (st, p) in seq.layers.iter_mut().zip(&pool.layers) {
            if let Some(p) = p { move_layer(st, p, &mtab); }
        }
        if let (Some(st), Some(p)) = (seq.mtp.as_mut(), pool.mtp.as_ref()) { move_layer(st, p, &mtab); }
        seq.mtab = mtab;
        seq.cap = len - 16;
    }

    /// Save the state of `seq` that is not per position (GDN states and conv windows, index-key rings, PLE window, ids) and the
    /// MTP head's pending row `pend` into `slot` (buffers reused): a prompt checkpoint. The positional caches stay in the
    /// sequence's range, valid as long as nothing rewrites the positions before the checkpoint.
    pub fn ckpt_save(&self, seq: &Seq, pend: &Tensor, slot: &mut Option<Ckpt>) {
        assert!(seq.pending.is_none(), "checkpoint with an uncommitted chain");
        let mut src: Vec<&Tensor> = Vec::new();
        for st in seq.layers.iter().chain(seq.mtp.iter()) {
            match st { LayerState::Gdn { s, conv } => { src.push(s); src.push(conv); } LayerState::Qsa { ring, .. } => src.push(ring) }
        }
        src.push(&seq.ple_win);
        src.push(pend);
        match slot {
            Some(c) if c.t.len() == src.len() && c.t.iter().zip(&src).all(|(a, b)| a.size() == b.size()) => {
                for (a, b) in c.t.iter().zip(&src) { a.shallow_clone().copy_(b); }
                c.ids.clone_from(&seq.ids);
            }
            _ => *slot = Some(Ckpt { ids: seq.ids.clone(), t: src.iter().map(|t| t.copy()).collect() }),
        }
    }

    /// Return `seq` to the checkpoint (position ckpt.ids.len()); returns the MTP head's pending row there.
    pub fn ckpt_restore(&self, seq: &mut Seq, c: &Ckpt) -> Tensor {
        let mut it = c.t.iter();
        for st in seq.layers.iter_mut().chain(seq.mtp.iter_mut()) {
            match st {
                LayerState::Gdn { s, conv } => { s.copy_(it.next().unwrap()); conv.copy_(it.next().unwrap()); }
                LayerState::Qsa { ring, .. } => { ring.copy_(it.next().unwrap()); }
            }
        }
        seq.ple_win.copy_(it.next().unwrap());
        seq.ids.clone_from(&c.ids);
        seq.pos = c.ids.len() as i64;
        seq.pending = None;
        it.next().unwrap().copy()
    }

    pub fn new_seq(&self, cap: i64) -> Seq {
        let cap = (cap + 3) / 4 * 4;
        let (mtab, mctl) = (Tensor::zeros([cap + 16, 3], (Kind::Int, Device::Cuda(0))), Tensor::zeros([2], (Kind::Int, Device::Cuda(0))));
        let layers = self.layers.iter().map(|l| match l.attn {
            Attn::Gdn(_) => LayerState::Gdn { s: Tensor::zeros([48, 128, 128], (Kind::Float, Device::Cuda(0))),
                                               conv: Tensor::zeros([10240, 3], (Kind::Float, Device::Cuda(0))) },
            Attn::Qsa(_) => qsa_state(cap, &mtab, &mctl),
        }).collect();
        let mtp = self.mtp.as_ref().map(|_| { let mut st = qsa_state(cap + 16, &mtab, &mctl); if let LayerState::Qsa { yarn, .. } = &mut st { *yarn = yarn_for(cap); } st });
        Seq { layers, ple_win: Tensor::zeros([9, H * D], (Kind::Float, Device::Cuda(0))), ids: vec![], pos: 0, cap, pending: None, mtp, mtab, mctl }
    }

    /// The sequence's mRoPE positions: `pos` [n][3] for its first n tokens (a prompt with images / videos) and the delta of
    /// the tokens after them (position = index - delta); an empty table for text (position = index).
    pub fn set_mrope(&self, seq: &Seq, pos: &[[i32; 3]], delta: i64) {
        let n = pos.len() as i64;
        assert!(n <= seq.mtab.size()[0], "mRoPE table");
        if n > 0 {
            let flat: Vec<i32> = pos.iter().flat_map(|p| p.iter().copied()).collect();
            let _ = seq.mtab.narrow(0, 0, n).copy_(&Tensor::from_slice(&flat).view([n, 3]).to_device(Device::Cuda(0)));
        }
        let _ = seq.mctl.shallow_clone().copy_(&Tensor::from_slice(&[n as i32, delta as i32]).to_device(Device::Cuda(0)));
    }

    /// Reuse a sequence for a new prompt: zero the recurrent state (attention caches are position-addressed
    /// and every read is preceded by a write at the new positions).
    pub fn reset(&self, seq: &mut Seq) {
        for st in seq.layers.iter_mut() {
            if let LayerState::Gdn { s, conv } = st { let _ = s.zero_(); let _ = conv.zero_(); }
        }
        let _ = seq.ple_win.zero_();
        let _ = seq.mctl.zero_();
        seq.ids.clear();
        seq.pos = 0;
        seq.pending = None;
    }

    /// Rows `ids` (<= 16) at the sequence's position; returns logits [R, vocab]. With commit = false the
    /// recurrent state is left untouched (call `commit` with the accepted row count afterwards).
    pub fn step(&self, seq: &mut Seq, ids: &[i64], commit: bool) -> Tensor {
        assert!(ids.len() as i64 <= MAX_ROWS);
        self.run(seq, ids, commit, false).0
    }

    /// Prompt rows (layer-major passes of prefill_macro() tokens); returns the logits of the last row [1, vocab].
    pub fn prefill(&self, seq: &mut Seq, ids: &[i64]) -> Tensor {
        let mut last = None;
        if std::env::var("QWEN_PREFILL_CM").as_deref() == Ok("1") {   // A/B: chunk-major (a whole forward per chunk)
            let _p = PrefillScope::new();
            for c in ids.chunks(prefill_chunk() as usize) { last = Some(self.run(seq, c, true, true).0); }
            return last.expect("empty prompt");
        }
        for c in ids.chunks(prefill_macro() as usize) { last = Some(self.prefill_lm(seq, c).0); }
        last.expect("empty prompt")
    }

    /// One layer-major prefill pass of `ids` at the sequence position (state committed): every layer runs on all
    /// rows (`layer_prefill`, attention sub-chunks of prefill_chunk()). Returns (last-row logits [1, vocab],
    /// stacks of all rows [T, 4, 2560]).
    pub fn prefill_lm(&self, seq: &mut Seq, ids: &[i64]) -> (Tensor, Tensor) { self.prefill_lm_mm(seq, ids, None) }

    /// `prefill_lm` with multimodal rows: vrows = (rows of the chunk, their vision embeddings [k, 2560]) replace the
    /// placeholders' token embeddings (in every stream, as a token embedding).
    pub fn prefill_lm_mm(&self, seq: &mut Seq, ids: &[i64], vrows: Option<(Vec<i64>, Tensor)>) -> (Tensor, Tensor) {
        let _g = tch::no_grad_guard();
        let _p = PrefillScope::new();
        let t = ids.len() as i64;
        assert!(t > 0 && seq.pending.is_none());
        assert!(seq.pos + t <= seq.cap, "sequence capacity");
        let idt = Tensor::from_slice(&vocab_ids(ids)).to_device(Device::Cuda(0));
        let mut x = f32(&self.embed.index_select(0, &idt)).unsqueeze(1).repeat([1, H, 1]).contiguous();
        if let Some((rows, emb)) = vrows {
            let k = rows.len() as i64;
            let src = emb.to_kind(Kind::Float).unsqueeze(1).expand([k, H, D], false).contiguous();
            let _ = x.index_copy_(0, &Tensor::from_slice(&rows).to_device(Device::Cuda(0)), &src);
        }
        let rb = self.ple_row_bytes().unwrap_or(0);
        let sub = prefill_chunk();
        let pos = Tensor::arange_start_step(seq.pos, seq.pos + t, sub, (Kind::Int, Device::Cuda(0)));
        let mut pend = None;
        // the n-gram rows are packed on another thread while the layers before the first PLE layer are queued (QWEN_PLE_PACK_SYNC=1:
        // packed first)
        let first_ple = self.layers.iter().position(|l| l.ple.is_some()).unwrap_or(self.layers.len());
        let ple_ref = self.layers.iter().find_map(|l| l.ple.as_ref());
        let (hist, pos0) = (&seq.ids, seq.ids.len());
        let pack = |ple: &Ple| {
            let v = super::vision::vocab_id;
            let rows = ple.rows_at(|i| v(if i < pos0 { hist[i] } else { ids[i - pos0] }), pos0, ids.len());
            ple.pack_rows(rows)
        };
        let overlap = std::env::var("QWEN_PLE_PACK_SYNC").as_deref() != Ok("1");
        let packed: Option<Tensor> = std::thread::scope(|sc| {
            let job = ple_ref.filter(|_| overlap).map(|p| sc.spawn(move || pack(p)));
            let mut packed = ple_ref.filter(|_| !overlap).map(|p| Tensor::from_slice(&pack(p)).to_device(Device::Cuda(0)));
            let mut job = job;
            for (li, (l, st)) in self.layers.iter().zip(seq.layers.iter()).enumerate() {
                if li == first_ple { if let Some(j) = job.take() { packed = Some(Tensor::from_slice(&j.join().unwrap()).to_device(Device::Cuda(0))); } }
                let ple = packed.as_ref().map(|p| (p, rb, &seq.ple_win));
                pend = layer_prefill(l, st, ple, &x, &pos, sub, pend.take());
            }
            packed
        });
        drop(packed);
        flush_apply(&x, pend);
        self.advance(seq, ids, true);
        let (_, s) = self.mixer.mix(&x.narrow(0, t - 1, 1).contiguous());
        (self.lm_head.fwd(&s), x)
    }

    /// MTP catch-up over many prompt rows, layer-major: hidden [n, 4, 2560] (target stacks), ids [n] (next tokens) at
    /// positions pos0..; returns (logits of the last row [1, Vd], output stacks [n, 4, 2560]).
    pub fn mtp_prefill(&self, seq: &mut Seq, hidden: &Tensor, ids: &[i64], pos0: i64) -> (Tensor, Tensor) {
        let m = self.mtp.as_ref().expect("no MTP head");
        let n = ids.len() as i64;
        let _p = PrefillScope::new();
        if n <= MAX_ROWS { return self.mtp_run(seq, hidden, ids, pos0, true); }
        let idt = Tensor::from_slice(&vocab_ids(ids)).to_device(Device::Cuda(0));
        let e = f32(&self.embed.index_select(0, &idt));
        let e = m.fc_e.fwd(&((&e * (e.square().mean_dim(-1, true, Kind::Float) + 1e-6).rsqrt()) * &m.ne1).contiguous());
        let hf = hidden.reshape([n, H * D]);
        let hn = (&hf * (hf.square().mean_dim(-1, true, Kind::Float) + 1e-6).rsqrt()) * &m.nh1;
        let h = m.fc_h.fwd(&hn.reshape([n * H, D]).contiguous()).view([n, H, D]);
        let x = (h + e.unsqueeze(1)).contiguous();
        let sub = prefill_chunk();
        let pos = Tensor::arange_start_step(pos0, pos0 + n, sub, (Kind::Int, Device::Cuda(0)));
        let pend = layer_prefill(&m.layer, seq.mtp.as_ref().unwrap(), None, &x, &pos, sub, None);
        flush_apply(&x, pend);
        let (_, s) = m.mixer.mix(&x.narrow(0, n - 1, 1).contiguous());
        let lg = match self.head() { Some(h) => h.fwd(&s), None => self.lm_head.fwd(&s) };
        (lg, x)
    }

    /// QWEN_MTP_PREFILL_TAIL=n (L2, drafts only; default 0 = off): a prompt's MTP catch-up covers only its last n rows. Returns
    /// the first position to catch up for a prompt of `len` tokens (positions before it are zeroed in the MTP cache).
    pub fn mtp_tail_from(&self, len: usize) -> i64 {
        let n: usize = std::env::var("QWEN_MTP_PREFILL_TAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if n == 0 || len <= n { 0 } else { (len - n) as i64 }
    }
    /// Debug: write the sequence state (fp32) to <prefix>_<layer>_<name>.f32 (layer 99: PLE window).
    pub fn dump_state(&self, seq: &Seq, prefix: &str) {
        let w = |t: &Tensor, name: String| {
            let v: Vec<f32> = Vec::try_from(t.to_kind(Kind::Float).to_device(Device::Cpu).flatten(0, -1)).unwrap();
            std::fs::write(format!("{prefix}_{name}.f32"), unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }).unwrap();
        };
        for (li, st) in seq.layers.iter().enumerate() {
            match st {
                LayerState::Gdn { s, conv } => { w(s, format!("{li}_s")); w(conv, format!("{li}_conv")); }
                LayerState::Qsa { kc, ks, vc, pooled, ring, .. } => {
                    let n = seq.pos;
                    w(&kc.narrow(0, 0, n), format!("{li}_kc")); w(&ks.narrow(0, 0, n), format!("{li}_ks")); w(&vc.narrow(0, 0, n), format!("{li}_vc"));
                    w(&pooled.narrow(0, 0, n / 4), format!("{li}_pooled")); w(ring, format!("{li}_ring"));
                }
            }
        }
        w(&seq.ple_win, "99_win".into());
    }

    /// Zero the MTP cache rows of positions [from, to) (skipped by a tail-only catch-up: no stale keys from an earlier
    /// sequence of the store).
    pub fn mtp_clear(&self, seq: &Seq, from: i64, to: i64) {
        if to <= from { return; }
        if let Some(LayerState::Qsa { kc, vc, pooled, .. }) = &seq.mtp {
            let _ = kc.narrow(0, from, to - from).zero_();
            let _ = vc.narrow(0, from, to - from).zero_();
            let (g0, g1) = (from / 4, (to + 3) / 4);
            let _ = pooled.narrow(0, g0, g1 - g0).zero_();
        }
    }

    /// The draft head in use: QWEN_DRAFT_PICK=i (read per call, i.e. at graph capture) selects QWEN_DRAFT_ALTS[i - 1].
    fn head(&self) -> Option<&DraftHead> {
        match std::env::var("QWEN_DRAFT_PICK").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(i) if i >= 1 && i <= self.draft_alts.len() => Some(&self.draft_alts[i - 1]),
            _ => self.draft_head.as_ref(),
        }
    }

    /// Rows `ids` at the sequence position; returns (logits [R or 1, vocab], stream stacks [R, 4, 2560]).
    pub fn run(&self, seq: &mut Seq, ids: &[i64], commit: bool, last_only: bool) -> (Tensor, Tensor) {
        let r = ids.len() as i64;
        assert!(r > 0 && seq.pending.is_none());
        assert!(seq.pos + r <= seq.cap, "sequence capacity");
        let idt = Tensor::from_slice(&vocab_ids(ids)).to_device(Device::Cuda(0));
        let pos = Tensor::from_slice(&[seq.pos as i32]).to_device(Device::Cuda(0));
        let packed = self.ple_pack(seq, ids).map(|h| Tensor::from_slice(&h).to_device(Device::Cuda(0)));
        let out = self.run_dev(seq, &idt, &pos, packed.as_ref(), r, commit, last_only);
        self.advance(seq, ids, commit);
        out
    }

    /// Host bookkeeping after a forward: committed rows extend the history, pending rows wait for `commit`.
    pub fn advance(&self, seq: &mut Seq, ids: &[i64], commit: bool) {
        if commit { seq.ids.extend_from_slice(ids); seq.pos += ids.len() as i64; }
        else if let Some(p) = seq.pending.as_mut() { p.ids = ids.to_vec(); }
    }

    /// n-gram row lookups served by (hot rows, run-time cache, table) so far
    pub fn ngram_stats(&self) -> Option<[u64; 4]> { self.layers.iter().find_map(|l| l.ple.as_ref()).map(|p| p.tier_stats()) }

    /// Packed n-gram rings for rows `ids` appended at the sequence position (None without a PLE layer).
    pub fn ple_row_bytes(&self) -> Option<i64> { self.layers.iter().find_map(|l| l.ple.as_ref()).map(|p| p.words * 2 * 16) }

    pub fn ple_pack(&self, seq: &Seq, ids: &[i64]) -> Option<Vec<u8>> {
        let ple = self.layers.iter().find_map(|l| l.ple.as_ref())?;
        let h = seq.ids.len();
        assert_eq!(h as i64, seq.pos);
        let v = super::vision::vocab_id;
        let rows = ple.rows_at(|i| v(if i < h { seq.ids[i] } else { ids[i - h] }), h, ids.len());
        Some(ple.pack_rows(rows))
    }

    /// Device part of a forward: ids [R] int64, pos [1] int32 and the packed PLE rings live on the device, so the
    /// whole call is graph-capturable. Leaves `seq.pending` (commit = false) without its host ids.
    #[allow(clippy::too_many_arguments)]
    pub fn run_dev(&self, seq: &mut Seq, idt: &Tensor, pos: &Tensor, packed: Option<&Tensor>, r: i64, commit: bool, last_only: bool) -> (Tensor, Tensor) {
        let (lg, x, pend) = self.run_batch_dev(&[&*seq], idt, pos, packed, &[(0, r)], commit, last_only);
        if !commit { seq.pending = Some(pend); }
        (lg, x)
    }

    /// Batched forward of several sequences (segment g = rows [r0, r0 + t) of `seqs[g]` at position pos[g], device int
    /// [segments]; packed PLE rings concatenated in row order). Returns (logits [R or 1, vocab], stacks [R, 4, 2560],
    /// the pending chain of all rows when commit = false).
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch_dev(&self, seqs: &[&Seq], idt: &Tensor, pos: &Tensor, packed: Option<&Tensor>, rows: &[(i64, i64)], commit: bool,
                         last_only: bool) -> (Tensor, Tensor, Pending) {
        let r = idt.size()[0];
        let x = f32(&self.embed.index_select(0, idt)).unsqueeze(1).repeat([1, H, 1]).contiguous();
        let mut pend = Pending { gdn: vec![], ple_nrm: None, ids: vec![] };
        // each layer's MoE update is added inside the next layer's attn HC norm (QWEN_DEC_APPLY_SPLIT=1, read per call:
        // added right away; bitwise equal)
        let split = std::env::var("QWEN_DEC_APPLY_SPLIT").as_deref() == Ok("1");
        let mut upd: PendApply = None;
        for (li, l) in self.layers.iter().enumerate() {
            if let Some(ple) = &l.ple {
                flush_apply(&x, upd.take());
                let wins: Vec<&Tensor> = seqs.iter().map(|s| &s.ple_win).collect();
                let nrm = ple.apply(&x, packed.expect("PLE rings"), rows, &wins);
                if let Some(i) = l.idx { crate::ablate::apply(i, &x); }
                if commit { for (&(r0, t), w) in rows.iter().zip(&wins) { ffi::ple_commit(w, &nrm.narrow(0, r0, t), t, None); } }
                else { pend.ple_nrm = Some(nrm); }
            }
            let sts: Vec<&LayerState> = seqs.iter().map(|s| &s.layers[li]).collect();
            let prev = upd.take();
            let (p, u) = block_pend(l, &sts, &x, rows, pos, commit, prev.as_ref().map(|(a, b)| (a, b)));
            pend.gdn.push(p);
            upd = u;
            if split { flush_apply(&x, upd.take()); }
        }
        flush_apply(&x, upd);
        let xl = if last_only { x.narrow(0, r - 1, 1).contiguous() } else { x.shallow_clone() };
        let (_, s) = self.mixer.mix(&xl);
        (self.lm_head.fwd(&s), x, pend)
    }

    /// Token id of an MTP logits column (draft head columns are a sub-vocabulary).
    pub fn draft_token(&self, col: i64) -> i64 { self.head().map_or(col, |h| h.keep[col as usize]) }

    /// MTP rows: hidden [n, 4, 2560] (target or previous MTP stacks), ids [n] (the next tokens), positions
    /// pos0..; returns (logits [n or 1, vocab], output stacks [n, 4, 2560]).
    pub fn mtp_run(&self, seq: &mut Seq, hidden: &Tensor, ids: &[i64], pos0: i64, last_only: bool) -> (Tensor, Tensor) {
        let idt = Tensor::from_slice(&vocab_ids(ids)).to_device(Device::Cuda(0));
        let pos = Tensor::from_slice(&[pos0 as i32]).to_device(Device::Cuda(0));
        self.mtp_dev(seq, hidden, &idt, &pos, last_only)
    }

    pub fn mtp_dev(&self, seq: &mut Seq, hidden: &Tensor, idt: &Tensor, pos: &Tensor, last_only: bool) -> (Tensor, Tensor) {
        let n = idt.size()[0];
        self.mtp_batch_dev(&[&*seq], hidden, idt, pos, &[(0, n)], if last_only { Pick::Last } else { Pick::All })
    }

    /// Batched MTP rows: segment g = rows [r0, r0 + t) of seqs[g] (its MTP cache) at position pos[g] (device int
    /// [segments]); logits of the picked rows and the output stacks of all rows.
    pub fn mtp_batch_dev(&self, seqs: &[&Seq], hidden: &Tensor, idt: &Tensor, pos: &Tensor, rows: &[(i64, i64)], pick: Pick) -> (Tensor, Tensor) {
        let m = self.mtp.as_ref().expect("no MTP head");
        let n = idt.size()[0];
        let e = f32(&self.embed.index_select(0, idt));
        let e = m.fc_e.fwd(&((&e * (e.square().mean_dim(-1, true, Kind::Float) + 1e-6).rsqrt()) * &m.ne1).contiguous());
        let hf = hidden.reshape([n, H * D]);
        let hn = (&hf * (hf.square().mean_dim(-1, true, Kind::Float) + 1e-6).rsqrt()) * &m.nh1;
        let h = m.fc_h.fwd(&hn.reshape([n * H, D]).contiguous()).view([n, H, D]);
        let x = (h + e.unsqueeze(1)).contiguous();
        let sts: Vec<&LayerState> = seqs.iter().map(|s| s.mtp.as_ref().unwrap()).collect();
        block(&m.layer, &sts, &x, rows, pos, true);
        let xl = match pick { Pick::All => x.shallow_clone(), Pick::Last => x.narrow(0, n - 1, 1).contiguous(), Pick::Rows(i) => x.index_select(0, i) };
        let (_, s) = m.mixer.mix(&xl);
        let lg = match self.head() { Some(h) => h.fwd(&s), None => self.lm_head.fwd(&s) };
        (lg, x)
    }

    /// Greedy draft tokens of MTP logits rows and their confidences (softmax max), on the device
    /// (int64 [rows], fp32 [rows]).
    pub fn draft_pick(&self, lg: &Tensor) -> (Tensor, Tensor) {
        let (mx, col) = lg.max_dim(-1, false);
        let conf = (mx - lg.logsumexp([-1], false)).exp();
        let tok = match self.head() { Some(h) => h.keep_d.index_select(0, &col), None => col };
        (tok, conf)
    }

    /// Accept the first n rows of the pending chain: replay them into the GDN states / conv windows.
    pub fn commit(&self, seq: &mut Seq, n: i64) {
        self.commit_dev(seq, n);
        let pend = seq.pending.take().expect("no pending chain");
        seq.ids.extend_from_slice(&pend.ids[..n as usize]);
        seq.pos += n;
    }

    /// Device part of `commit` (graph-capturable): replay the first n pending rows into the recurrent state.
    pub fn commit_dev(&self, seq: &mut Seq, n: i64) {
        let pend = seq.pending.take().expect("no pending chain");
        self.commit_with(seq, &pend, n);
        seq.pending = Some(pend);
    }

    /// Replay the first n rows of `pend` (a verify forward's GDN inputs and PLE normed rows) into `seq`.
    pub fn commit_with(&self, seq: &mut Seq, pend: &Pending, n: i64) {
        if n > 0 { self.commit_segs(&[&*seq], pend, &[(0, n)], None); }
    }

    /// Replay a batched pending chain: segment g = rows [r0, r0 + t) into seqs[g]; with `nd` (device int [segments])
    /// only the first min(nd[g], t) rows of each segment.
    pub fn commit_segs(&self, seqs: &[&Seq], pend: &Pending, rows: &[(i64, i64)], nd: Option<&Tensor>) {
        for (li, (l, p)) in self.layers.iter().zip(pend.gdn.iter()).enumerate() {
            if let (Attn::Gdn(g), Some((qkv, y, ab, z))) = (&l.attn, p) {
                let (ss, cs): (Vec<&Tensor>, Vec<&Tensor>) = seqs.iter().map(|q| match &q.layers[li] {
                    LayerState::Gdn { s, conv } => (s, conv), _ => unreachable!() }).unzip();
                ffi::gdn_recur_segs(y, ab, z, &g.a_log, &g.dt_bias, &g.norm, &ffi::Segs::new(rows, &ss), nd, None, true);
                ffi::gdn_conv_commit_segs(qkv, &ffi::Segs::new(rows, &cs), nd);
            }
        }
        if let Some(nrm) = &pend.ple_nrm {
            for (g, (&(r0, t), q)) in rows.iter().zip(seqs).enumerate() {
                ffi::ple_commit(&q.ple_win, &nrm.narrow(0, r0, t), t, nd.map(|d| d.narrow(0, g as i64, 1)).as_ref());
            }
        }
    }
}

