//! DeepSeek-V4.1-Flash forward, v0 (docs/07 D3): official `inference/model.py` semantics on libtorch ops, TP over
//! `world` ranks (docs/07 §2): attention by output group, routed/shared experts by intermediate dimension, vocabulary
//! by 128-row blocks; everything else replicated. One code path serves prefill, chunked prefill and decode: a chunk of T
//! tokens at absolute positions [seq.pos, seq.pos + T) reads the committed state (sliding-window rings, compressed KV
//! and index keys, compressor partial groups, n-gram history) and appends to it. Exact mode only (no SWA bounded replay).
//!
//! Numerics follow the official kernels: activations FP8-quantized (UE8M0 per 32) before FP8/FP4 GEMMs, whose weights
//! are dequantized exactly to BF16 (FP4 experts per call); window KV FP8, compressed KV FP4 (E4M3 per 16), indexer
//! q/k FP4 (UE8M0 per 32). The TP partial sums are summed in FP32 across ranks.
use super::cfg::{Cfg, Split};
use super::engram;
use super::load::{dequant_fp8, dequant_mxfp4, fp4_table, Checkpoint};
use super::ops::{self, act_quant, fp4_quant, rms_norm, rope, Rope};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// FP8 E4M3 linear weight as stored: bytes [n, k] (Uint8) and E8M0 scales per 32x32 block. `mm` quantizes the
/// activation (UE8M0 per 32) and multiplies: up to 8 rows with the decode GEMV (ds41.cu), more rows with the BF16 MMA
/// GEMM (ds41_fp8_mm); DS41_GEMV=0 forces the v0 exact BF16 dequantization + cuBLAS for every size.
pub struct Fp8W { pub(super) w: Tensor, pub(super) s: Tensor }
/// DS41_PARITY=1 (single rank, golden checks): round every expert's output and the shared expert's output to BF16 before
/// they are summed, like the official MoE (expert-parallel there, so each expert output is complete). With TP the
/// partial sums are kept in FP32 instead (rounding a partial is not the official operation either).
fn parity() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_PARITY").as_deref() == Ok("1")) }
/// The sum over ranks of a layer output y [T, dim] FP32 (in place; the caller rounds it to BF16 next). Prefill-size
/// buffers (> 64 rows) go over the BF16 transport (DS41_AR_BF16, default on): each rank's partial and the sum are
/// rounded to BF16 on the wire (half the bytes on links that are at their limit; golden p2: prefill KL 0.0146 vs
/// 0.0179 with the FP32 transport, both at the reference's own noise floor 0.0157).
thread_local! { static DEFER_AR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// The TP sum of a layer output, rounded to BF16; inside the two-micro-batch encoder (DEFER_AR) the FP32 partial sum
/// itself, reduced by the caller asynchronously.
fn ar_bf16(y: Tensor) -> Tensor {
    if DEFER_AR.with(|d| d.get()) { return y; }
    ar_prefill(&y);
    bf16(y)
}
fn ar_prefill(y: &Tensor) {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    extern "C" { fn rs_allreduce_bf16t(p: *mut std::ffi::c_void, numel: i64) -> i32; }
    if *E.get_or_init(|| std::env::var("DS41_AR_BF16").map_or(true, |v| v != "0")) && y.size()[0] > 64 && !crate::tp::graph::capturing() {
        let rc = prof("allreduce", || unsafe { rs_allreduce_bf16t(y.data_ptr(), y.numel() as i64) });
        if rc == 0 { return; }
    }
    prof("allreduce", || crate::tp::allreduce(y));
}
/// The block-scaled MMA GEMV (any 1..16 rows, cost independent of the row count) instead of the CUDA-core one: for 2..16
/// rows always (the CUDA-core GEMV's cost grows with the rows), for 1 row with DS41_GEMV_MX1 (default on).
fn mx_gemv_ok(m: i64, k: i64, n: i64) -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let one = *E.get_or_init(|| std::env::var("DS41_GEMV_MX1").map_or(true, |v| v != "0"));
    moe_mx_on() && m <= 16 && (m >= 2 || one) && k % 64 == 0 && n % 8 == 0
}
/// DS41_MOE_MX=0: the BF16-MMA routed-expert kernels (shim/ds41_moe.cu) instead of the block-scaled MMA ones (ds41_mx.cu).
pub(super) fn moe_mx_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_MOE_MX").map_or(true, |v| v != "0")) }
/// DS41_MOE=0: the v0 per-expert loop instead of the fused routed-expert kernels (shim/ds41_moe.cu).
fn moe_fused() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_MOE").map_or(true, |v| v != "0")) }
/// DS41_OPS=0: the v0 libtorch RMSNorm / RoPE instead of shim/ds41_ops.cu.
fn ops_fused() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_OPS").map_or(true, |v| v != "0")) }
pub(super) fn rmsn(x: &Tensor, w: &Tensor, eps: f64) -> Tensor {
    if ops_fused() && x.kind() == Kind::BFloat16 && w.kind() == Kind::BFloat16 { super::kern::rms_norm(x, w, eps) } else { rms_norm(x, w, eps) }
}
/// SPARK_ABLATE_CLAMP on a sublayer input z (docs/05-消融.md §3): the clamp runs on fp32 rows while the DS41 stream is
/// BF16, so an upcast copy is clamped and rounded back. `ln` is the site's norm weight (registered at load with its
/// layer and site); returns z itself when SPARK_ABLATE_CLAMP is unset (zero overhead for unablated runs).
pub(super) fn clamp_z(ln: &Tensor, z: Tensor) -> Tensor {
    let Some(c) = crate::ablate::clamp() else { return z };
    let z32 = z.to_kind(Kind::Float);
    c.apply_input(ln, &z32);
    z32.to_kind(z.kind())
}
/// The sublayer input z = rmsn(x, ln) (docs/05-消融.md §3a): with the clamp fused into the ds41 rms_norm vec kernel's
/// fp32 registers (GLM FUSE=2 style: no fp32 round-trip, no extra kernel) when available, else the fp32-upcast +
/// row-kernel fallback above (STATS mode, FUSE < 2, non-vec shapes). `off` is the row base of these z's rows in the
/// pass (the encoder two-micro-batch path's count gate).
pub(super) fn site_z_off(ln: &Tensor, x: &Tensor, eps: f64, off: i64) -> Tensor {
    if let Some(c) = crate::ablate::clamp() {
        if let Some(a) = c.norm_args_off(ln, x.size()[0], off) {
            if ops_fused() && x.kind() == Kind::BFloat16 && ln.kind() == Kind::BFloat16 {
                if let Some(y) = super::kern::rms_norm_cl(x, ln, eps, &a) { return y; }
            }
        }
        return clamp_z(ln, rmsn(x, ln, eps));
    }
    rmsn(x, ln, eps)
}
pub(super) fn site_z(ln: &Tensor, x: &Tensor, eps: f64) -> Tensor { site_z_off(ln, x, eps, 0) }
/// FP4 fake quantization: the fused kernel (DS41_OPS) for BF16 input, else ops::fp4_quant.
fn fp4q(x: &Tensor, block: i64, e4m3_scale: bool) -> Tensor {
    if ops_fused() && x.kind() == Kind::BFloat16 { super::kern::fp4_quant(x, block, e4m3_scale) } else { fp4_quant(x, block, e4m3_scale) }
}
/// `rot` of a tensor the caller owns (a fresh result, or a buffer rewritten every step): in place when fused.
pub(super) fn rot_mut(x: Tensor, r: &Rope, pos: &Tensor, rd: i64, inverse: bool) -> Tensor {
    if ops_fused() && x.kind() == Kind::BFloat16 && x.is_contiguous() { super::kern::rope_inplace(&x, pos, r.freqs(), rd, inverse); x } else { rot(&x, r, pos, rd, inverse) }
}
pub(super) fn rot(x: &Tensor, r: &Rope, pos: &Tensor, rd: i64, inverse: bool) -> Tensor {
    if ops_fused() && x.kind() == Kind::BFloat16 { super::kern::rope(x, pos, r.freqs(), rd, inverse) } else { rope(x, &r.at(pos), rd, inverse) }
}
/// DS41_HC=0: the v0 libtorch mHC ops instead of shim/ds41_hc.cu.
fn hc_fused() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_HC").map_or(true, |v| v != "0")) }
/// DS41_ATTN=0: the v0 libtorch sparse attention instead of shim/ds41_attn.cu.
pub(super) fn attn_fused() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_ATTN").map_or(true, |v| v != "0")) }
fn gemv_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_GEMV").map_or(true, |v| v != "0")) }
/// DS41_WO_A_ACT=0: feed wo_a the plain BF16 attention output (the official model.py before HF commit 4dd75a8b,
/// 2026-10-12, which the old golden was made with) instead of act_quant(o) (FP8 per 32, UE8M0), as official now does.
fn wo_a_act() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_WO_A_ACT").map_or(true, |v| v != "0")) }
/// A BF16 matrix [n, k] (n, k multiples of 32) as FP8 E4M3 with one UE8M0 scale per 32 x 32 block (the checkpoint's
/// FP8 layout): scale 2^ceil(log2(amax / 448)) per block.
pub(super) fn quant_fp8_blocks(w: &Tensor) -> Fp8W {
    let _g = tch::no_grad_guard();
    let (n, k) = (w.size()[0], w.size()[1]);
    assert!(n % 32 == 0 && k % 32 == 0, "quant_fp8_blocks [{n} x {k}]");
    let wf = w.to_kind(Kind::Float).view([n / 32, 32, k / 32, 32]);
    let amax = wf.abs().amax([1i64, 3].as_slice(), true).clamp_min(1e-30);
    let e = ops::log2_ceil(&(amax * (1.0 / 448.0)));
    let q = (&wf / ops::pow2(&e)).clamp(-448.0, 448.0).to_kind(Kind::Float8e4m3fn).view_dtype(Kind::Uint8).view([n, k]).contiguous();
    let s = (e + 127).clamp(0, 254).to_kind(Kind::Uint8).view([n / 32, k / 32]).contiguous();
    Fp8W { w: q, s }
}
impl Fp8W {
    pub fn n(&self) -> i64 { self.w.size()[0] }
    fn dequant(&self) -> Tensor { dequant_fp8(&self.w.view_dtype(Kind::Float8e4m3fn), &self.s) }
    /// x [.., k] -> BF16 [.., n] (FP32 accumulation, one rounding), like the official FP8 Linear.
    pub fn mm(&self, x: &Tensor) -> Tensor {
        let mut shape = x.size();
        let k = *shape.last().unwrap();
        let x2 = x.reshape([-1, k]).to_kind(Kind::BFloat16);
        let m = x2.size()[0];
        // the MMA paths quantize inside (E4M3 bytes); the others take the BF16 act_quant
        let xq = || super::kern::act_quant(&x2);
        let y = if !gemv_on() { xq().matmul(&self.dequant().tr()) }
                else if mx_gemv_ok(m, k, self.n()) { super::kern::fp8_gemv_mx_bf16(&x2, &self.w, &self.s) }
                else if m <= 8 { bf16(super::kern::fp8_gemv(&xq(), &self.w, &self.s)) }
                else if m > 64 && moe_mx_on() && k % 64 == 0 { super::kern::fp8_gemm_mx_bf16(&x2, &self.w, &self.s) }
                else { bf16(super::kern::fp8_mm(&xq(), &self.w, &self.s)) };
        *shape.last_mut().unwrap() = self.n();
        y.reshape(shape.as_slice())
    }
    /// Rows [r0, r0 + n) (multiples of 32) as a view.
    pub fn rows(&self, r0: i64, n: i64) -> Fp8W { Fp8W { w: self.w.narrow(0, r0, n), s: self.s.narrow(0, r0 / 32, n / 32) } }
    /// x [m, k] BF16 (not quantized: products with the FP8 weight are exact either way) -> FP32 [m, n].
    pub fn mm_raw(&self, x: &Tensor) -> Tensor {
        let x = x.to_kind(Kind::BFloat16).contiguous();
        if gemv_on() && x.size()[0] <= 8 { super::kern::fp8_gemv(&x, &self.w, &self.s) }
        else if gemv_on() { super::kern::fp8_mm(&x, &self.w, &self.s) }
        else { x.to_kind(Kind::Float).matmul(&self.dequant().to_kind(Kind::Float).tr()) }
    }
    /// `mm_f32` of the act_quant operands (xq E4M3 bytes [m, k], xs UE8M0 bytes [m, k / 32]) when mm_f32 would take the
    /// block-scaled MMA GEMM for this shape (which quantizes its BF16 input into exactly these), else None.
    pub fn mm_f32_q(&self, xq: &Tensor, xs: &Tensor) -> Option<Tensor> {
        let (m, k) = (xq.size()[0], xq.size()[1]);
        (gemv_on() && !mx_gemv_ok(m, k, self.n()) && m > 64 && moe_mx_on() && k % 64 == 0)
            .then(|| super::kern::fp8_gemm_mx_q(xq, xs, &self.w, &self.s))
    }
    /// x [m, k] BF16 -> act_quant(x) . W^T FP32 [m, n] (no output rounding: TP partial sums). The MMA paths quantize
    /// inside; act_quant is idempotent, so an already quantized x gives the same result.
    pub fn mm_f32(&self, x: &Tensor) -> Tensor {
        let (m, k) = (x.size()[0], x.size()[1]);
        let xq = || super::kern::act_quant(x);
        if gemv_on() && mx_gemv_ok(m, k, self.n()) { super::kern::fp8_gemv_mx(x, &self.w, &self.s) }
        else if gemv_on() && m <= 8 { super::kern::fp8_gemv(&xq(), &self.w, &self.s) }
        else if gemv_on() && m > 64 && moe_mx_on() && k % 64 == 0 { super::kern::fp8_gemm_mx(x, &self.w, &self.s) }
        else if gemv_on() { super::kern::fp8_mm(&xq(), &self.w, &self.s) }
        else { xq().to_kind(Kind::Float).matmul(&self.dequant().to_kind(Kind::Float).tr()) }
    }
}

/// wkv / wgate of a ratio > 1 compressor are FP32 in the checkpoint but BF16 values (checked at load): then w2 holds
/// both as one BF16 [2 head_dim, dim] weight (wkv / wgate are views of it), read by the BF16 GEMV for decode rows (x
/// is BF16, so BF16 x BF16 products with FP32 sums are the official FP32 matmul's products; a quarter of the bytes).
pub struct Compressor { wkv: Tensor, wgate: Option<Tensor>, norm: Tensor, w2: Option<Tensor> }
pub struct Indexer { wq_b: Fp8W, weights_proj: Tensor, wk: Option<Tensor>, k_norm: Option<Tensor> }
pub struct Attn {
    /// wq_a and wkv stacked (they share the input; one GEMV), and views of the two
    pub(super) qakv: Fp8W, pub(super) wq_a: Fp8W, pub(super) q_norm: Tensor, pub(super) wq_b: Fp8W, pub(super) wkv: Fp8W, pub(super) kv_norm: Tensor, pub(super) sink: Tensor,
    wo_a: Fp8W,    // [local groups * o_lora, heads_per_group * head_dim] FP8 (dequantized: the official BF16 weight)
    wo_b: Fp8W,    // [dim, local groups * o_lora]
    comp: Option<Compressor>, idx: Option<Indexer>, pub(super) rope: Rope,
}
/// Routed experts of one layer, this rank's intermediate slice, MXFP4 (Uint8) in the checkpoint layout (v0 path), and
/// the fused kernel's tile layout (`packed`, made at load; the checkpoint copy is then dropped unless DS41_MOE=0).
pub struct Experts { w1: Tensor, w1s: Tensor, w3: Tensor, w3s: Tensor, w2: Tensor, w2s: Tensor, packed: Option<[Tensor; 6]> }
/// Shared expert: w1 and w3 stacked (one GEMV over the same input) and views of them.
pub struct Mlp { w13: Fp8W, w1: Fp8W, w3: Fp8W, w2: Fp8W }
pub struct Moe { pub(super) gate: Tensor, bias: Tensor, bias_vl: Option<Tensor>, experts: Experts, shared: Mlp, topk: i64 }
pub struct Engram { wkv: Fp8W, row0: i64, qk: Tensor }
pub struct Hc { pub(super) attn_fn: Tensor, pub(super) attn_base: Tensor, pub(super) attn_scale: Tensor, pub(super) ffn_fn: Tensor, pub(super) ffn_base: Tensor, pub(super) ffn_scale: Tensor }
pub struct Layer { pub id: usize, pub(super) attn_norm: Tensor, pub(super) ffn_norm: Tensor, pub(super) hc: Hc, pub(super) attn: Attn, pub(super) moe: Moe, pub(super) engram: Option<Engram> }

pub struct Model {
    pub c: Cfg,
    pub sp: Split,
    pub ck: Checkpoint,
    /// The original checkpoint (engram tables are read from it row by row).
    pub tables: Checkpoint,
    pub dev: Device,
    pub(super) embed: Tensor,          // [v1-v0, dim] BF16, this rank's vocabulary rows
    pub(super) head: Tensor,           // [v1-v0, dim] BF16 (used in FP32)
    pub(super) norm: Tensor,
    pub layers: Vec<Layer>,
    pub hash: Option<engram::Hash>,
    fp4: Tensor,
    /// Captured single-token decode step (DS41_GRAPH, default on), built on the first decode.
    graph: std::cell::RefCell<HashMap<Vec<usize>, DecodeGraph>>,
    /// Use counter of the row graphs (least recently used ones are dropped beyond DS41_GRAPH_MAX).
    graph_clock: std::cell::Cell<u64>,
    /// DSpark draft stages (DS41_MTP=1).
    pub mtp: Option<super::spec::Mtp>,
    /// The DSpark drafts' copy of this rank's head rows in FP8 (32 x 32 blocks, UE8M0 scales; DS41_DRAFT_HEAD_FP8,
    /// default on): half the bytes per draft call; the verify keeps the BF16 head, so the output is unchanged
    pub(super) head8: Option<Fp8W>,
    /// The last forward's DSpark target hidden rows (mean over the hc copies of the attention input of the target
    /// layers, concatenated) for its last min(window, T) positions, when the MTP stages are loaded.
    pub last_hidden: std::cell::RefCell<Option<Tensor>>,
    /// Debug (ds41-dspark-golden): when Some, mtp_draft appends the full draft logits of every draft step.
    pub draft_debug: std::cell::RefCell<Option<Vec<Tensor>>>,
    pub(super) mtp_graphs: std::cell::RefCell<HashMap<usize, super::spec::MtpGraph>>,
}

/// Decode of one token for each of B sequences as CUDA graphs (one set per B): per layer S1 (Engram mix, mHC mix/pre, norm, Q / window KV projections) and
/// S3 (output projection + rank sum, mHC post/mix/pre, MoE + rank sum, mHC post), and the final norm + logits. The
/// shape-dynamic middle (compressor, indexer, pool, sparse attention; attn_mid) runs eagerly between S1 and S3 and
/// writes its output into a fixed buffer; it runs per sequence on that sequence's row. The graphs touch no sequence
/// state, so they serve any sequences.
thread_local! {
    /// Engram rows of later engram layers being gathered in the background for the current prefill chunk (by layer).
    static ENGRAM_PENDING: std::cell::RefCell<HashMap<usize, std::thread::JoinHandle<(Vec<u8>, Vec<u8>)>>> = std::cell::RefCell::new(HashMap::new());
}
thread_local! {
    /// The image rows among the rows the backbone layers currently run (bool [rows] on the device; None: no image row):
    /// their MoE gates take `bias_vl` and the engram leaves them unchanged (official `token_types >= 0`).
    static VL: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}
/// Sets the image-row mask for the layers about to run; cleared when the guard drops.
struct VlGuard;
impl Drop for VlGuard { fn drop(&mut self) { VL.with(|v| *v.borrow_mut() = None); } }
fn vl_set(mask: Option<Tensor>) -> VlGuard { VL.with(|v| *v.borrow_mut() = mask); VlGuard }
/// Rows [off, off + n) of the image-row mask, None without image rows. `exact`: only when the mask has n rows.
fn vl_rows(off: i64, n: i64, exact: bool) -> Option<Tensor> {
    VL.with(|v| v.borrow().as_ref().filter(|m| if exact { m.size()[0] == n } else { m.size()[0] >= off + n }).map(|m| m.narrow(0, off, n)))
}
/// Engram rows of a chunk: BF16, or the act_quant operands of its projection (E4M3 bytes [T, K], UE8M0 bytes [T, K / 32]).
pub(super) enum ERows { Bf16(Tensor), Q(Tensor, Tensor) }
impl ERows {
    fn narrow(&self, r0: i64, n: i64) -> ERows {
        match self { ERows::Bf16(r) => ERows::Bf16(r.narrow(0, r0, n)), ERows::Q(q, s) => ERows::Q(q.narrow(0, r0, n), s.narrow(0, r0, n)) }
    }
}
/// DS41_ENGRAM_Q=0: prefill dequantizes the engram rows to BF16 first (the GEMM quantizes them again).
fn engram_q_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_ENGRAM_Q").map_or(true, |v| v != "0")) }

/// The engram leaves image rows unchanged (its gate is shut there): `out` with those rows of `h` put back.
fn engram_skip_images(h: &Tensor, out: Tensor, off: i64) -> Tensor {
    match vl_rows(off, h.size()[0], false) { Some(m) => h.where_self(&m.view([-1, 1, 1]), &out), None => out }
}
/// DS41_SHARED_PAR (default on): in decode / verify the shared expert runs on a parallel stream beside the routed ones.
fn shared_par() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_SHARED_PAR").map_or(true, |v| v != "0")) }
/// DS41_OVERLAP_ROWS (512; 0 off): prefill chunks of at least twice this many rows run the encoder as two
/// micro-batches with the all-reduces overlapped (Model::encoder_two_batch).
fn overlap_rows() -> i64 { static E: std::sync::OnceLock<i64> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_OVERLAP_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512)) }
/// DS41_CED (default on): prompts prefill through Model::prefill_ced (decoder over the last window only).
pub fn ced_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_CED").map_or(true, |v| v != "0")) }
/// DS41_DEBUG_ROWS=1: the row graphs record row 0 of every layer's S1 outputs, middle output and layer output
/// (take_rows_debug), to find where two row signatures diverge.
fn debug_rows() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_DEBUG_ROWS").is_ok_and(|v| v == "1")) }
thread_local! { static ROWS_DBG: std::cell::RefCell<Vec<(String, Tensor)>> = const { std::cell::RefCell::new(Vec::new()) }; }
fn rows_dbg(i: usize, mid: &[Tensor; 4], o: &Tensor, h: &[Tensor; 5]) {
    if !debug_rows() { return; }
    ROWS_DBG.with(|d| {
        let mut d = d.borrow_mut();
        for (nm, t) in [("x", &mid[0]), ("qr", &mid[1]), ("q", &mid[2]), ("kv", &mid[3]), ("o", o), ("y", &h[0]), ("h2", &h[1]), ("x2", &h[2]), ("moe", &h[3]), ("h", &h[4])] {
            d.push((format!("L{i}.{nm}"), t.narrow(0, 0, 1).to_kind(Kind::Float).to_device(Device::Cpu)));
        }
    });
}
pub fn take_rows_debug() -> Vec<(String, Tensor)> { ROWS_DBG.with(|d| std::mem::take(&mut *d.borrow_mut())) }
pub struct DecodeGraph {
    pos: Tensor, h0: Tensor, rows: HashMap<usize, Tensor>,
    s1: Vec<crate::tp::graph::Owned>, s3: Vec<crate::tp::graph::Owned>, fin: crate::tp::graph::Owned,
    mid: Vec<[Tensor; 4]>,   // S1 outputs read by the middle: x, qr, q, kv
    obuf: Vec<Tensor>,
    hs: Vec<[Tensor; 5]>,    // each layer's attention output, h2, FFN input, MoE output, output (DS41_DEBUG_ROWS)
    logits: Tensor,
    hid: Option<Tensor>,
    used: u64,
    _keep: Vec<Tensor>,
}
/// DS41_TOPK=0: the indexer's selection by libtorch topk + sort instead of the fused kernel.
fn topk_fused() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_TOPK").map_or(true, |v| v != "0")) }
/// DS41_ATTN_MULTI=0: decode / verify batches run the sparse attention sequence by sequence (else one launch for all).
fn attn_multi_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_ATTN_MULTI").map_or(true, |v| v != "0")) }
pub(super) fn graph_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_GRAPH").map_or(true, |v| v != "0")) }

/// Per-sequence state.
pub struct Seq {
    pub pos: i64,
    pub(super) win: Vec<Tensor>,            // per layer [window, head_dim] BF16 ring: slot = position % window
    pub(super) ckv: HashMap<usize, Tensor>, // KV source layer -> [cap / ratio, head_dim] compressed KV (FP4 values, BF16)
    pub(super) ikv: HashMap<usize, Tensor>, // KV source layer -> [cap / ratio, index_dim] index keys (FP4 values, BF16)
    pub(super) cstate: HashMap<usize, (Tensor, Tensor)>, // ratio > 1 sources: kv and score of the open group [ratio, head_dim] FP32
    /// ratio > 1 sources: the compressor rows of the last chunk from its open group's start (kv, score, first
    /// position), so a speculative verify can roll the open group back to any accepted length
    pub(super) comp_log: HashMap<usize, (Tensor, Tensor, i64)>,
    /// DSpark stages' window KV [window, head_dim] BF16 by slot = position % window (main-model positions)
    pub(super) mtp_win: Vec<Tensor>,
    pub hist: Vec<i64>,                     // compressed token ids of every committed position (engram)
    pub cap: i64,
    /// tokens covered by compressed rows ever written (also rolled-back speculative ones): reset_seq zeroes up to it
    pub(super) hw: i64,
    /// CED prefill (docs/07): the decoder layers have run every position before dec_pos (their window rings hold
    /// them); dec_pos < pos after encoder-only prefill chunks, until the closing decoder replay
    pub dec_pos: i64,
    /// CED prefill: the encoder output (residual h [n, hc, dim] BF16, pre-mix [n, hc] FP32) of the last n <= window
    /// positions ending at the given position, for the decoder replay of the prompt's last window
    pub(super) ced_tail: Option<(Tensor, Tensor, i64)>,
    /// Prefix cache checkpoints inside a prefill chunk: positions strictly inside the next chunk (set by the serve loop
    /// before it) whose state is captured while the chunk runs (`ck_cap`), as if the chunk had been cut there.
    pub ck_req: Vec<i64>,
    pub ck_cap: std::collections::BTreeMap<i64, Capture>,
    /// Image input (rank 0, set by the serve loop before a prefill chunk): the chunk rows of its image positions
    /// (offsets in the chunk, rows [k, dim] FP32), written into the embedding (super::vision, crate::mm_canvas).
    pub vis: Option<(Vec<i64>, Tensor)>,
}

/// The state of a sequence at a position inside a prefill chunk that a chunk end would have: the window rings of the
/// layers that ran (the others are unchanged), the compressor's open groups, the CED encoder tail.
#[derive(Default)]
pub struct Capture { pub(super) win: HashMap<usize, Tensor>, pub(super) cs: HashMap<usize, (Tensor, Tensor)>, pub(super) ced: Option<(Tensor, Tensor)> }

fn bf16(t: Tensor) -> Tensor { t.to_kind(Kind::BFloat16) }

impl Model {
    /// Loads this rank's share of the backbone (layers [0, n_layers); MTP not yet). `engram_hash`: the exported hash
    /// constants (None disables engram, debug only).
    pub fn load(ckpt: &Path, rank: usize, world: usize, n_layers: usize, engram_hash: Option<&Path>, dev: Device) -> Model {
        let c = Cfg::load(ckpt);
        let sp = Split::new(&c, rank, world);
        let ck = Checkpoint::open(ckpt, dev);
        let fp4 = fp4_table(dev);
        let t0 = std::time::Instant::now();
        let embed = ck.rows("embed.weight", sp.v0, sp.v1 - sp.v0, Kind::BFloat16);
        let head = ck.rows("head.weight", sp.v0, sp.v1 - sp.v0, Kind::BFloat16);
        let norm = ck.get("norm.weight", Kind::BFloat16);
        let hash = engram_hash.map(engram::Hash::load);
        let mut layers = Vec::new();
        for l in 0..n_layers.min(c.n_layers) {
            let mut layer = Self::load_layer(&ck, &c, &sp, l, &fp4);
            Self::pack_experts(&mut layer, dev);
            layers.push(layer);
            if l % 8 == 7 || l + 1 == n_layers.min(c.n_layers) {
                let (a, r) = crate::tp::graph::memory();
                eprintln!("[ds41] rank{rank} loaded layers ..{} in {:.0}s (allocated {:.1} GiB, reserved {:.1} GiB)", l + 1,
                    t0.elapsed().as_secs_f64(), a as f64 / (1u64 << 30) as f64, r as f64 / (1u64 << 30) as f64);
            }
        }
        let tables = Checkpoint::open(ck.source.as_deref().unwrap_or(ckpt), dev);
        if hash.is_some() { for l in &c.engram_layers { tables.advise_random(&format!("layers.{l}.engram.embed.weight")); } }
        let mtp = (std::env::var("DS41_MTP").as_deref() == Ok("1") && n_layers >= c.n_layers).then(|| super::spec::Mtp::load(&ck, &c, &sp, &fp4, dev));
        if mtp.is_some() {
            let (a, r) = crate::tp::graph::memory();
            eprintln!("[ds41] rank{rank} DSpark stages loaded in {:.0}s (allocated {:.1} GiB, reserved {:.1} GiB)", t0.elapsed().as_secs_f64(),
                a as f64 / (1u64 << 30) as f64, r as f64 / (1u64 << 30) as f64);
        }
        if dev.is_cuda() { super::kern::gemv_mx_init(); }
        let head8 = (mtp.is_some() && std::env::var("DS41_DRAFT_HEAD_FP8").map_or(true, |v| v != "0")).then(|| quant_fp8_blocks(&head));
        Model { c, sp, ck, tables, dev, embed, head, norm, layers, hash, fp4, graph: std::cell::RefCell::new(HashMap::new()), graph_clock: std::cell::Cell::new(0), mtp, head8,
                last_hidden: std::cell::RefCell::new(None), draft_debug: std::cell::RefCell::new(None),
                mtp_graphs: std::cell::RefCell::new(HashMap::new()) }
    }

    /// The fused kernel's tile layout of the routed experts (the checkpoint layout dropped).
    pub(super) fn pack_experts(layer: &mut Layer, dev: Device) {
        Self::pack_experts_tiles(layer, dev);
        if moe_mx_on() {
            if let Some(p) = layer.moe.experts.packed.as_mut() {
                for i in [0usize, 2, 4] { p[i] = super::kern::moe_pack_mx(&p[i]); }
            }
        }
    }
    fn pack_experts_tiles(layer: &mut Layer, dev: Device) {
        if !moe_fused() || parity() || layer.moe.experts.packed.is_some() { return; }
        let ex = &mut layer.moe.experts;
        let (a, b) = super::kern::moe_pack(&ex.w1, &ex.w1s);
        let (c3, d) = super::kern::moe_pack(&ex.w3, &ex.w3s);
        let (e2, f) = super::kern::moe_pack(&ex.w2, &ex.w2s);
        ex.packed = Some([a, b, c3, d, e2, f]);
        let empty = Tensor::empty([0], (Kind::Uint8, dev));
        for t in [&mut ex.w1, &mut ex.w1s, &mut ex.w3, &mut ex.w3s, &mut ex.w2, &mut ex.w2s] { *t = empty.shallow_clone(); }
    }

    pub(super) fn fp8(ck: &Checkpoint, name: &str) -> Fp8W {
        Fp8W { w: ck.get(&format!("{name}.weight"), Kind::Uint8), s: ck.get(&format!("{name}.scale"), Kind::Uint8) }
    }
    /// FP8 rows [r0, r0+n) (multiples of 32).
    fn fp8_rows(ck: &Checkpoint, name: &str, r0: i64, n: i64) -> Fp8W {
        assert!(r0 % 32 == 0 && n % 32 == 0);
        Fp8W { w: ck.rows(&format!("{name}.weight"), r0, n, Kind::Uint8), s: ck.rows(&format!("{name}.scale"), r0 / 32, n / 32, Kind::Uint8) }
    }
    /// FP8 columns [c0, c0+n) (multiples of 32).
    fn fp8_cols(ck: &Checkpoint, name: &str, c0: i64, n: i64) -> Fp8W {
        assert!(c0 % 32 == 0 && n % 32 == 0);
        Fp8W { w: ck.cols(&format!("{name}.weight"), c0, n, Kind::Uint8), s: ck.cols(&format!("{name}.scale"), c0 / 32, n / 32, Kind::Uint8) }
    }

    pub(crate) fn load_layer(ck: &Checkpoint, c: &Cfg, sp: &Split, l: usize, fp4: &Tensor) -> Layer {
        Self::load_block(ck, c, sp, &format!("layers.{l}"), l, c.n_experts, c.n_active, fp4)
    }
    /// One block under `p` (`layers.N` or `mtp.N`), configured as layer `l` of the config (compression, engram),
    /// with `ne` routed experts of which `topk` are active.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_block(ck: &Checkpoint, c: &Cfg, sp: &Split, p: &str, l: usize, ne: i64, topk: i64, _fp4: &Tensor) -> Layer {
        let hd = c.head_dim;
        let (h0, h1) = (sp.h0(c), sp.h1(c));
        let gl = sp.g1 - sp.g0;
        let a = format!("{p}.attn");
        let ratio = c.compress[l];
        let rope = if ratio > 0 {
            Rope::new(c.rope_dim, c.original_seq_len, c.compress_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        } else {
            Rope::new(c.rope_dim, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        }.to(ck.dev);
        let comp = c.is_kv_source(l).then(|| {
            let wkv = ck.get(&format!("{a}.compressor.wkv.weight"), if ratio > 1 { Kind::Float } else { Kind::BFloat16 });
            let wgate = (ratio > 1).then(|| ck.get(&format!("{a}.compressor.wgate.weight"), Kind::Float));
            let norm = ck.get(&format!("{a}.compressor.norm.weight"), Kind::BFloat16);
            let exact = |w: &Tensor| w.to_kind(Kind::BFloat16).to_kind(Kind::Float).equal(w);
            match wgate {
                Some(wg) if exact(&wkv) && exact(&wg) => {
                    let hd = wkv.size()[0];
                    let w2 = Tensor::cat(&[wkv.to_kind(Kind::BFloat16), wg.to_kind(Kind::BFloat16)], 0);
                    Compressor { wkv: w2.narrow(0, 0, hd), wgate: Some(w2.narrow(0, hd, hd)), norm, w2: Some(w2) }
                }
                wgate => Compressor { wkv, wgate, norm, w2: None },
            }
        });
        let idx = c.is_index_source(l).then(|| Indexer {
            wq_b: Self::fp8(ck, &format!("{a}.indexer.wq_b")),
            weights_proj: ck.get(&format!("{a}.indexer.weights_proj.weight"), Kind::BFloat16),
            wk: c.is_kv_source(l).then(|| ck.get(&format!("{a}.indexer.wk.weight"), Kind::BFloat16)),
            k_norm: c.is_kv_source(l).then(|| ck.get(&format!("{a}.indexer.k_norm.weight"), Kind::BFloat16)),
        });
        let wo_a_rows = Self::fp8_rows(ck, &format!("{a}.wo_a"), sp.g0 * c.o_lora, gl * c.o_lora);
        let (wq_a0, wkv0) = (Self::fp8(ck, &format!("{a}.wq_a")), Self::fp8(ck, &format!("{a}.wkv")));
        let qakv = Fp8W { w: Tensor::cat(&[&wq_a0.w, &wkv0.w], 0), s: Tensor::cat(&[&wq_a0.s, &wkv0.s], 0) };
        drop((wq_a0, wkv0));
        let attn = Attn {
            wq_a: qakv.rows(0, c.q_lora),
            wkv: qakv.rows(c.q_lora, c.head_dim),
            qakv,
            q_norm: ck.get(&format!("{a}.q_norm.weight"), Kind::BFloat16),
            wq_b: Self::fp8_rows(ck, &format!("{a}.wq_b"), h0 * hd, (h1 - h0) * hd),
            kv_norm: ck.get(&format!("{a}.kv_norm.weight"), Kind::BFloat16),
            sink: ck.get(&format!("{a}.attn_sink"), Kind::Float).narrow(0, h0, h1 - h0).contiguous(),
            wo_a: wo_a_rows,
            wo_b: Self::fp8_cols(ck, &format!("{a}.wo_b"), sp.g0 * c.o_lora, gl * c.o_lora),
            comp, idx, rope,
        };
        let f = format!("{p}.ffn");
        let (i0, il) = (sp.i0, sp.i1 - sp.i0);
        let experts = if ck.has(&format!("{f}.experts.packed.w1")) {
            // a packed shard (ds41-shard-pack): the fused kernel's stacks as stored
            let g = |n: &str| ck.get(&format!("{f}.experts.packed.{n}"), Kind::Uint8);
            let empty = Tensor::empty([0], (Kind::Uint8, ck.dev));
            Experts { w1: empty.shallow_clone(), w1s: empty.shallow_clone(), w3: empty.shallow_clone(), w3s: empty.shallow_clone(),
                      w2: empty.shallow_clone(), w2s: empty, packed: Some([g("w1"), g("w1s"), g("w3"), g("w3s"), g("w2"), g("w2s")]) }
        } else {
            let mut ex: Vec<Vec<Tensor>> = (0..6).map(|_| Vec::new()).collect();
            for e in 0..ne {
                let x = format!("{f}.experts.{e}");
                ex[0].push(ck.rows(&format!("{x}.w1.weight"), i0, il, Kind::Uint8));
                ex[1].push(ck.rows(&format!("{x}.w1.scale"), i0, il, Kind::Uint8));
                ex[2].push(ck.rows(&format!("{x}.w3.weight"), i0, il, Kind::Uint8));
                ex[3].push(ck.rows(&format!("{x}.w3.scale"), i0, il, Kind::Uint8));
                ex[4].push(ck.cols(&format!("{x}.w2.weight"), i0 / 2, il / 2, Kind::Uint8));
                ex[5].push(ck.cols(&format!("{x}.w2.scale"), i0 / 32, il / 32, Kind::Uint8));
            }
            let st = |v: &Vec<Tensor>| Tensor::stack(v, 0);
            let experts = Experts { w1: st(&ex[0]), w1s: st(&ex[1]), w3: st(&ex[2]), w3s: st(&ex[3]), w2: st(&ex[4]), w2s: st(&ex[5]), packed: None };
            drop(ex);
            experts
        };
        let (w1, w3) = (Self::fp8_rows(ck, &format!("{f}.shared_experts.w1"), i0, il), Self::fp8_rows(ck, &format!("{f}.shared_experts.w3"), i0, il));
        let w13 = Fp8W { w: Tensor::cat(&[&w1.w, &w3.w], 0), s: Tensor::cat(&[&w1.s, &w3.s], 0) };
        drop((w1, w3));
        let shared = Mlp {
            w1: w13.rows(0, il),
            w3: w13.rows(il, il),
            w13,
            w2: Self::fp8_cols(ck, &format!("{f}.shared_experts.w2"), i0, il),
        };
        let moe = Moe {
            gate: ck.get(&format!("{f}.gate.weight"), Kind::BFloat16),
            bias: ck.get(&format!("{f}.gate.bias"), Kind::Float),
            bias_vl: ck.has(&format!("{f}.gate.bias_vl")).then(|| ck.get(&format!("{f}.gate.bias_vl"), Kind::Float)),
            experts, shared, topk,
        };
        let engram = (l < c.n_layers && c.engram_layers.contains(&l)).then(|| {
            let q = ck.get(&format!("{p}.engram.q_weight"), Kind::BFloat16).to_kind(Kind::Float);
            let k = ck.get(&format!("{p}.engram.k_weight"), Kind::BFloat16).to_kind(Kind::Float);
            // wkv's output rows split over the ranks in 32-row blocks (each rank's rows complete; a zero-padded sum
            // assembles them): the 157 MB matrix is read once per token in total instead of once per rank
            let full = Self::fp8(ck, &format!("{p}.engram.wkv"));
            let blocks = full.w.size()[0] / 32;
            let (b0, b1) = super::cfg::share(blocks, sp.world, sp.rank);
            let wkv = Fp8W { w: full.w.narrow(0, b0 * 32, (b1 - b0) * 32).copy(), s: full.s.narrow(0, b0, b1 - b0).copy() };
            Engram { wkv, row0: b0 * 32, qk: q * k }
        });
        let hc = Hc {
            attn_fn: ck.get(&format!("{p}.hc_attn_fn"), Kind::Float), attn_base: ck.get(&format!("{p}.hc_attn_base"), Kind::Float),
            attn_scale: ck.get(&format!("{p}.hc_attn_scale"), Kind::Float), ffn_fn: ck.get(&format!("{p}.hc_ffn_fn"), Kind::Float),
            ffn_base: ck.get(&format!("{p}.hc_ffn_base"), Kind::Float), ffn_scale: ck.get(&format!("{p}.hc_ffn_scale"), Kind::Float),
        };
        let layer = Layer { id: l, attn_norm: ck.get(&format!("{p}.attn_norm.weight"), Kind::BFloat16), ffn_norm: ck.get(&format!("{p}.ffn_norm.weight"), Kind::BFloat16), hc, attn, moe, engram };
        // SPARK_ABLATE_CLAMP sites: 0 = the attention input (attn_norm), 1 = the MoE input (ffn_norm); the MTP blocks
        // sit past tau's layer count and stay unregistered (site() filters them anyway).
        if l < c.n_layers { if let Some(cl) = crate::ablate::clamp() { cl.register(&layer.attn_norm, l, 0); cl.register(&layer.ffn_norm, l, 1); } }
        layer
    }

    pub fn new_seq(&self, cap: i64) -> Seq {
        let mut seq = self.new_seq_fixed();
        let views: Vec<Tensor> = self.kv_arrays().iter().map(|a| Tensor::zeros([cap / a.div + 1, a.width], (a.kind, self.dev))).collect();
        self.attach_kv(&mut seq, &views, cap);
        seq
    }

    /// The KV sources present on this model, ascending: the context-proportional state of a sequence.
    fn kv_srcs(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.c.kv_sources.iter().copied().filter(|&s| s < self.layers.len()).collect();
        v.sort();
        v
    }
    /// The context-proportional state as KV pool arrays (crate::kv_pool): per KV source (ascending) its compressed KV
    /// rows and index keys, one row per `ratio` tokens.
    pub fn kv_arrays(&self) -> Vec<crate::kv_pool::Arr> {
        let c = &self.c;
        self.kv_srcs().iter().flat_map(|&s| {
            let r = c.compress[s];
            [crate::kv_pool::Arr { div: r, width: c.head_dim, kind: Kind::BFloat16 }, crate::kv_pool::Arr { div: r, width: c.index_dim, kind: Kind::BFloat16 }]
        }).collect()
    }
    /// A fresh sequence's fixed-size state (window rings, compressor groups, DSpark windows) without its compressed
    /// KV / index keys (attach_kv) and capacity 0.
    pub fn new_seq_fixed(&self) -> Seq {
        let c = &self.c;
        let opts = (Kind::BFloat16, self.dev);
        let win = (0..self.layers.len()).map(|_| Tensor::zeros([c.window, c.head_dim], opts)).collect();
        let mut cstate = HashMap::new();
        for s in self.kv_srcs() {
            let r = c.compress[s];
            if r > 1 {
                cstate.insert(s, (Tensor::zeros([r, c.head_dim], (Kind::Float, self.dev)), Tensor::full([r, c.head_dim], f64::NEG_INFINITY, (Kind::Float, self.dev))));
            }
        }
        let mtp_win = self.mtp.as_ref().map_or(vec![], |m| (0..m.layers.len()).map(|_| Tensor::zeros([c.window, c.head_dim], opts)).collect());
        Seq { pos: 0, win, ckv: HashMap::new(), ikv: HashMap::new(), cstate, comp_log: HashMap::new(), mtp_win, hist: Vec::new(), cap: 0, hw: 0, dec_pos: 0, ced_tail: None, vis: None,
              ck_req: Vec::new(), ck_cap: Default::default() }
    }
    /// The sequence's compressed KV / index keys are `views` (kv_arrays order; e.g. its KV pool lease), its capacity
    /// `cap` tokens. Rows already written stay (a grown lease keeps them).
    pub fn attach_kv(&self, seq: &mut Seq, views: &[Tensor], cap: i64) {
        for (i, s) in self.kv_srcs().into_iter().enumerate() {
            seq.ckv.insert(s, views[2 * i].shallow_clone());
            seq.ikv.insert(s, views[2 * i + 1].shallow_clone());
        }
        seq.cap = cap;
    }
    /// Back to a fresh sequence's state in place (same tensors: KV pool views stay): the fixed-size state reset and
    /// the written compressed rows zeroed, so the sequence equals a new_seq of its capacity.
    pub fn reset_seq(&self, seq: &mut Seq) {
        let c = &self.c;
        for w in seq.win.iter_mut() { let _ = w.zero_(); }
        for w in seq.mtp_win.iter_mut() { let _ = w.zero_(); }
        for (k, sc) in seq.cstate.values_mut() { let _ = k.zero_(); let _ = sc.fill_(f64::NEG_INFINITY); }
        for (&s, t) in seq.ckv.iter_mut() {
            let n = (seq.pos.max(seq.hw) / c.compress[s] + 1).min(t.size()[0]);
            let _ = t.narrow(0, 0, n).zero_();
            let _ = seq.ikv.get_mut(&s).unwrap().narrow(0, 0, n).zero_();
        }
        seq.pos = 0;
        seq.hw = 0;
        seq.dec_pos = 0;
        seq.hist.clear();
        seq.comp_log.clear();
        seq.ced_tail = None;
        seq.ck_req.clear();
        seq.ck_cap.clear();
    }

    /// The prefix cache's row arrays of `seq` (ds41/pcache.rs): per KV source (ascending) its compressed KV and index
    /// keys, one row per ratio tokens, complete groups only (the open group is in the fixed part).
    pub fn pcache_rows(&self, seq: &Seq) -> Vec<crate::pcache_core::Rows> {
        let c = &self.c;
        self.kv_srcs().into_iter().flat_map(|s| {
            let r = c.compress[s];
            [crate::pcache_core::Rows { name: format!("ckv.{s}"), t: seq.ckv[&s].shallow_clone(), div: r, ceil: false },
             crate::pcache_core::Rows { name: format!("ikv.{s}"), t: seq.ikv[&s].shallow_clone(), div: r, ceil: false }]
        }).collect()
    }
    /// The prefix cache's fixed regions and header fields of `seq`: window rings, compressor groups, DSpark windows,
    /// CED tail, n-gram history (host); pos, dec_pos, ced_end.
    pub fn pcache_fixed(&self, seq: &Seq) -> (Vec<(String, crate::pcache_core::Part)>, serde_json::Map<String, serde_json::Value>) {
        use crate::pcache_core::Part;
        let wins: Vec<Tensor> = seq.win.iter().map(|t| t.shallow_clone()).collect();
        let mut f = vec![("win".to_string(), Part::Dev(self.pcache_win(&wins, seq.dec_pos, seq.pos)))];
        for s in self.kv_srcs() {
            if let Some((k, sc)) = seq.cstate.get(&s) { f.push((format!("csk.{s}"), Part::Dev(k.shallow_clone()))); f.push((format!("css.{s}"), Part::Dev(sc.shallow_clone()))); }
        }
        if !seq.mtp_win.is_empty() { f.push(("mtp_win".into(), Part::Dev(Tensor::stack(&seq.mtp_win, 0)))); }
        let mut meta = serde_json::Map::new();
        meta.insert("pos".into(), serde_json::json!(seq.pos));
        meta.insert("dec_pos".into(), serde_json::json!(seq.dec_pos));
        if let Some((th, tp, end)) = &seq.ced_tail {
            f.push(("ced_h".into(), Part::Dev(th.shallow_clone())));
            f.push(("ced_pm".into(), Part::Dev(tp.shallow_clone())));
            meta.insert("ced_end".into(), serde_json::json!(end));
        }
        let hb: Vec<u8> = seq.hist.iter().flat_map(|t| t.to_le_bytes()).collect();
        f.push(("hist".into(), Part::Host { kind: Kind::Int64, shape: vec![seq.hist.len() as i64], bytes: std::sync::Arc::new(hb) }));
        (f, meta)
    }
    /// The fixed regions and header fields of a checkpoint at position `p` inside the prefill chunk `seq` just ran
    /// (captured state `cap`): None unless it is complete (a CED encoder-only chunk with every encoder layer captured).
    pub fn pcache_fixed_at(&self, seq: &Seq, p: i64, cap: &Capture) -> Option<(Vec<(String, crate::pcache_core::Part)>, serde_json::Map<String, serde_json::Value>)> {
        use crate::pcache_core::Part;
        let enc = self.c.n_layers / 2;
        let (ch, cp) = cap.ced.as_ref()?;
        if p >= seq.pos || seq.dec_pos > p || (0..enc.min(self.layers.len())).any(|l| !cap.win.contains_key(&l)) { return None; }
        let win: Vec<Tensor> = (0..seq.win.len()).map(|l| cap.win.get(&l).map_or_else(|| seq.win[l].shallow_clone(), |t| t.shallow_clone())).collect();
        let mut f = vec![("win".to_string(), Part::Dev(self.pcache_win(&win, seq.dec_pos, p)))];
        for s in self.kv_srcs() {
            if seq.cstate.contains_key(&s) { let (k, sc) = cap.cs.get(&s)?; f.push((format!("csk.{s}"), Part::Dev(k.shallow_clone()))); f.push((format!("css.{s}"), Part::Dev(sc.shallow_clone()))); }
        }
        if !seq.mtp_win.is_empty() { f.push(("mtp_win".into(), Part::Dev(Tensor::stack(&seq.mtp_win, 0)))); }
        f.push(("ced_h".into(), Part::Dev(ch.shallow_clone())));
        f.push(("ced_pm".into(), Part::Dev(cp.shallow_clone())));
        let mut meta = serde_json::Map::new();
        meta.insert("pos".into(), serde_json::json!(p));
        meta.insert("dec_pos".into(), serde_json::json!(seq.dec_pos));
        meta.insert("ced_end".into(), serde_json::json!(p));
        let hb: Vec<u8> = seq.hist[..p as usize].iter().flat_map(|t| t.to_le_bytes()).collect();
        f.push(("hist".into(), Part::Host { kind: Kind::Int64, shape: vec![p], bytes: std::sync::Arc::new(hb) }));
        Some((f, meta))
    }
    /// The window rings a checkpoint at `p` stores (stacked): the encoder layers' only when the decoder last ran a full
    /// window before p (dec_pos + window <= p). Its rings are then never read again: the next prompt's closing replay
    /// starts at most a window before its end, i.e. after dec_pos, so it is the bounded one (window attention truncated
    /// to the replayed rows) and rewrites them (prefill_ced). Saves the 20 decoder rings (2.6 MB) of most fork
    /// checkpoints; a prompt's own end (dec_pos == pos) keeps all.
    fn pcache_win(&self, win: &[Tensor], dec_pos: i64, p: i64) -> Tensor {
        let enc = (self.c.n_layers / 2) as usize;
        if dec_pos + self.c.window <= p && win.len() > enc { Tensor::stack(&win[..enc], 0) } else { Tensor::stack(win, 0) }
    }
    /// Installs a restored checkpoint into `seq`, whose rows (pcache_rows) already hold the restored rows and were reset
    /// before (reset_seq).
    pub fn pcache_install(&self, seq: &mut Seq, l: crate::pcache_core::Loaded) -> Result<(), String> {
        let mut o = l.owned;
        let pos = l.meta["pos"].as_i64().ok_or("pos")?;
        if pos != l.ids.len() as i64 || pos >= seq.cap { return Err(format!("position {pos} for {} ids, capacity {}", l.ids.len(), seq.cap)); }
        let win = o.remove("win").ok_or("win")?;
        let dec_pos = l.meta["dec_pos"].as_i64().ok_or("dec_pos")?;
        // all rings, or the encoder's (pcache_win) with the decoder's left as reset (never read before rewritten)
        let enc = self.c.n_layers / 2;
        let all = win.size()[0] == seq.win.len() as i64;
        if !all && !(win.size()[0] == enc as i64 && dec_pos + self.c.window <= pos) { return Err("window layers".into()); }
        for (i, w) in seq.win.iter_mut().enumerate().take(win.size()[0] as usize) { let _ = w.copy_(&win.get(i as i64)); }
        for (&s, (k, sc)) in seq.cstate.iter_mut() {
            let _ = k.copy_(&o.remove(&format!("csk.{s}")).ok_or("csk")?);
            let _ = sc.copy_(&o.remove(&format!("css.{s}")).ok_or("css")?);
        }
        if !seq.mtp_win.is_empty() {
            // entries written without the DSpark stages do not seed them: not usable
            let mw = o.remove("mtp_win").ok_or("mtp_win")?;
            for (i, w) in seq.mtp_win.iter_mut().enumerate() { let _ = w.copy_(&mw.get(i as i64)); }
        }
        seq.hist = Vec::<i64>::try_from(o.remove("hist").ok_or("hist")?).map_err(|e| e.to_string())?;
        seq.pos = pos;
        seq.hw = pos;
        seq.dec_pos = dec_pos;
        seq.ced_tail = match (o.remove("ced_h"), o.remove("ced_pm"), l.meta["ced_end"].as_i64()) {
            (Some(th), Some(tp), Some(end)) => Some((th, tp, end)),
            _ => None,
        };
        Ok(())
    }

    /// Embedding of `ids` [T] (vocabulary rows of this rank, summed across ranks).
    pub(super) fn embed(&self, ids: &Tensor) -> Tensor { self.embed_vis(ids, None) }
    /// `embed` with image rows (rank 0: chunk offsets, rows [k, dim] FP32) written before the sum: a salted placeholder
    /// is outside every rank's vocabulary slice, so every rank holds zeros there and the sum is the image row.
    pub(super) fn embed_vis(&self, ids: &Tensor, vis: Option<(Vec<i64>, Tensor)>) -> Tensor {
        let local = ids - self.sp.v0;
        let ok = local.ge(0).logical_and(&local.lt(self.sp.v1 - self.sp.v0));
        let rows = self.embed.index_select(0, &local.clamp(0, self.sp.v1 - self.sp.v0 - 1));
        let e = rows.to_kind(Kind::Float) * ok.unsqueeze(-1).to_kind(Kind::Float);
        let mut e = e.contiguous();
        if let Some((offs, r)) = vis { let _ = e.index_copy_(0, &Tensor::from_slice(&offs).to_device(self.dev), &r); }
        crate::tp::allreduce(&e);
        bf16(e)
    }

    /// Logits [T, vocab] FP32 of the final hidden states x [T, dim] (rank slices assembled by a zero-padded sum).
    pub fn logits(&self, x: &Tensor) -> Tensor {
        let m = x.size()[0];
        let local = if gemv_on() && m <= 64 {
            let parts: Vec<Tensor> = (0..m).step_by(8).map(|r| super::kern::bf16_gemv(&x.narrow(0, r, 8.min(m - r)), &self.head)).collect();
            if parts.len() == 1 { parts.into_iter().next().unwrap() } else { Tensor::cat(&parts, 0) }
        } else { x.to_kind(Kind::Float).matmul(&self.head.to_kind(Kind::Float).tr()) };
        if self.sp.world == 1 { return local; }
        let full = Tensor::zeros([x.size()[0], self.c.vocab], (Kind::Float, self.dev));
        full.narrow(1, self.sp.v0, self.sp.v1 - self.sp.v0).copy_(&local);
        crate::tp::allreduce(&full);
        full
    }

    pub(super) fn hc_mixes(&self, h: &Tensor, f: &Tensor, scale: &Tensor, base: &Tensor) -> (Tensor, Tensor, Tensor) {
        if hc_fused() { return super::kern::hc_mix(h, f, scale, base, self.c.norm_eps, self.c.hc_eps, self.c.hc_iters); }
        let t = h.size()[0];
        let x = h.reshape([t, -1]).to_kind(Kind::Float);
        let rs = (x.square().mean_dim([-1].as_slice(), true, Kind::Float) + self.c.norm_eps).rsqrt();
        let mixes = x.matmul(&f.tr()) * rs;
        ops::hc_split_sinkhorn(&mixes, scale, base, self.c.hc, self.c.hc_iters, self.c.hc_eps)
    }
    pub(super) fn hc_pre(h: &Tensor, pre: &Tensor) -> Tensor {
        if hc_fused() { return super::kern::hc_pre(h, pre); }
        bf16((pre.unsqueeze(-1) * h.to_kind(Kind::Float)).sum_dim_intlist([1].as_slice(), false, Kind::Float))
    }
    /// y[k] = post[k] x + sum_j comb[j][k] res[j]
    pub(super) fn hc_post(x: &Tensor, res: &Tensor, post: &Tensor, comb: &Tensor) -> Tensor {
        if hc_fused() { return super::kern::hc_post(x, res, post, comb); }
        let a = post.unsqueeze(-1) * x.to_kind(Kind::Float).unsqueeze(1);
        let b = (comb.unsqueeze(-1) * res.to_kind(Kind::Float).unsqueeze(2)).sum_dim_intlist([1].as_slice(), false, Kind::Float);
        bf16(a + b)
    }

    /// The image-row mask of positions [a, b) (bool on the device), None when none of them is an image position: image
    /// positions are the DEAD entries of the n-gram history (salted placeholders, engram::Hash::compress).
    fn image_rows(&self, seq: &Seq, a: i64, b: i64) -> Option<Tensor> {
        let h = seq.hist.get(a as usize..b as usize)?;
        if !h.contains(&engram::DEAD) { return None; }
        let m: Vec<bool> = h.iter().map(|&t| t == engram::DEAD).collect();
        Some(Tensor::from_slice(&m).to_device(self.dev))
    }

    /// Appends `ids` to the sequence's n-gram history; the engram hash ids [engram layer][T * cols].
    pub(super) fn push_hashes(&self, seq: &mut Seq, ids: &[i64]) -> Vec<Vec<i64>> {
        let Some(hs) = &self.hash else { return Vec::new() };
        let mut hashes = vec![Vec::with_capacity(ids.len() * hs.cols()); hs.layer_ids.len()];
        for &id in ids {
            seq.hist.push(hs.compress(id));
            let n = seq.hist.len();
            hs.at_into(&seq.hist[n.saturating_sub(4)..], &mut hashes);
        }
        hashes
    }

    /// Starts loading the engram table rows of `next` (the tokens that will follow `prev`, the raw ids of up to three
    /// tokens before them) into the page cache in the background (prefill: the next chunk while this one runs).
    pub fn engram_prefetch(&self, prev: &[i64], next: &[i64]) {
        let Some(hs) = &self.hash else { return };
        let mut hist: Vec<i64> = prev[prev.len().saturating_sub(3)..].iter().map(|&t| hs.compress(t)).collect();
        let mut per: Vec<Vec<i64>> = vec![Vec::with_capacity(next.len() * hs.cols()); hs.layer_ids.len()];
        for &id in next {
            hist.push(hs.compress(id));
            let n = hist.len();
            hs.at_into(&hist[n.saturating_sub(4)..], &mut per);
        }
        let rows: Vec<(usize, Vec<i64>)> = hs.layer_ids.iter().copied().zip(per).filter(|(l, _)| self.layers.iter().any(|x| x.id == *l)).collect();
        engram::prefetch(&self.tables, rows, hs.head_dim);
    }

    /// Logits [1, vocab] FP32 of the last of `ids` appended to `seq`: one token through the decode graphs, otherwise
    /// (several tokens, DS41_GRAPH=0, DS41_PARITY=1, DS41_DUMP_LAYERS) the eager forward.
    pub fn step_logits(&self, seq: &mut Seq, ids: &[i64]) -> Tensor {
        if ids.len() == 1 { return self.decode_logits(&mut [seq], ids); }
        let last = ids.len() as i64 - 1;
        let x = self.forward(seq, ids, Some(&[last]));
        self.logits(&x)
    }

    /// One token for each sequence (ids[i] appended to seqs[i]) in one batch: logits [B, vocab] FP32.
    pub fn decode_logits(&self, seqs: &mut [&mut Seq], ids: &[i64]) -> Tensor {
        assert_eq!(seqs.len(), ids.len());
        if graph_on() && !parity() && dump_layers().is_empty() { return self.decode_graph(seqs, ids); }
        let rows: Vec<Tensor> = seqs.iter_mut().zip(ids).map(|(s, &id)| { let x = self.forward(s, &[id], None); self.logits(&x) }).collect();
        Tensor::cat(&rows, 0)
    }

    fn decode_graph(&self, seqs: &mut [&mut Seq], ids: &[i64]) -> Tensor {
        let per: Vec<Vec<i64>> = ids.iter().map(|&i| vec![i]).collect();
        self.rows_graph(seqs, &per).0
    }

    /// Several rows for each sequence (ids[b] appended to seqs[b], committed) through CUDA graphs keyed by the rows per
    /// sequence: logits [R, vocab] FP32 of every row (sequences in order), and with the DSpark stages loaded the target
    /// hidden rows [R, 3 dim]. Single-token decode (rows 1 each) and speculative verification (1 + drafts) both use it;
    /// the middle runs per sequence on its rows (a chunk of attn_mid).
    pub fn forward_rows(&self, seqs: &mut [&mut Seq], ids: &[Vec<i64>]) -> (Tensor, Option<Tensor>) {
        if graph_on() && !parity() && dump_layers().is_empty() { return self.rows_graph(seqs, ids); }
        let mut lg = Vec::new();
        let mut hid = Vec::new();
        for (s, r) in seqs.iter_mut().zip(ids) {
            let x = self.forward(s, r, None);
            lg.push(self.logits(&x));
            if let Some(h) = self.last_hidden.borrow_mut().take() { hid.push(h); }
        }
        (Tensor::cat(&lg, 0), (!hid.is_empty()).then(|| Tensor::cat(&hid, 0)))
    }

    fn rows_graph(&self, seqs: &mut [&mut Seq], ids: &[Vec<i64>]) -> (Tensor, Option<Tensor>) {
        use crate::tp::graph::{begin_pool, end, Owned};
        let _g = tch::no_grad_guard();
        let c = &self.c;
        let dev = self.dev;
        assert_eq!(seqs.len(), ids.len());
        let key: Vec<usize> = ids.iter().map(|r| r.len()).collect();
        let offs: Vec<i64> = key.iter().scan(0i64, |a, &n| { let o = *a; *a += n as i64; Some(o) }).collect();
        let total: i64 = key.iter().sum::<usize>() as i64;
        let starts: Vec<i64> = seqs.iter().zip(&key).map(|(s, &n)| { assert!(s.pos + n as i64 <= s.cap, "sequence capacity {} exceeded", s.cap); s.pos }).collect();
        let hashes: Vec<Vec<Vec<i64>>> = seqs.iter_mut().zip(ids).map(|(s, r)| self.push_hashes(s, r)).collect();
        let flat: Vec<i64> = ids.iter().flatten().copied().collect();
        let e = self.embed(&Tensor::from_slice(&flat).to_device(dev));                  // [R, dim]
        let engram_layers: Vec<usize> = self.hash.as_ref().map_or(vec![], |h| h.layer_ids.iter().copied().filter(|l| self.layers.iter().any(|x| x.id == *l)).collect());
        let rows_now = |li: usize, l: usize| -> Tensor {
            let all: Vec<i64> = hashes.iter().flat_map(|h| h[li].iter().copied()).collect();
            self.engram_rows(l, &all, total)
        };
        let posv: Vec<i64> = starts.iter().zip(&key).flat_map(|(&s, &n)| s..s + n as i64).collect();
        let pos_now = Tensor::from_slice(&posv).to_device(dev);
        let want_hid = self.mtp.is_some();
        let mut graphs = self.graph.borrow_mut();
        let mut run_mid = |layer: &Layer, mid: &[Tensor; 4], pos: &Tensor, obuf: Option<&Tensor>, sh: &mut Vec<Shared>, seqs: &mut [&mut Seq]| -> Option<Tensor> {
            let [x, qr, q, kv] = mid;
            // several sequences: each one's compressor / indexer, then the sparse attention of all rows in one launch
            if let (Some(ob), true) = (obuf, seqs.len() > 1 && attn_fused() && attn_multi_on()) {
                let mut preps = Vec::new();
                let iq = c.is_index_source(layer.id).then(|| self.index_q(layer, x, qr, pos));
                let cp = (c.is_kv_source(layer.id) && c.compress[layer.id] > 0 && !sh.iter().any(|s| s.skip_compress)).then(|| self.comp_proj(layer, x));
                for (b, seq) in seqs.iter_mut().enumerate() {
                    let r = |t: &Tensor| t.narrow(0, offs[b], key[b] as i64);
                    if let Some((q, w)) = &iq { sh[b].index_q = Some((r(q), r(w))); }
                    if let Some(p) = &cp { sh[b].comp_in = Some(r(p)); }
                    preps.push(self.attn_prep(layer, &r(x), &r(qr), seq, &r(pos), starts[b], &mut sh[b]));
                }
                let kvs: Vec<Tensor> = (0..seqs.len()).map(|b| kv.narrow(0, offs[b], key[b] as i64)).collect();
                let descs: Vec<_> = preps.iter().zip(&kvs).enumerate().map(|(b, ((ring, ckv, ci), kb))| (ring, kb, ckv.as_ref(), ci, starts[b], sh[b].win_lo)).collect();
                let scale = (c.head_dim as f64).powf(-0.5);
                let done = prof("attn.sparse", || super::kern::sparse_attn_multi(q, &descs, c.window, pos, &layer.attn.sink, scale, ob));
                if !done {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("[ds41] batched sparse attention not applicable here (rows {total}); per sequence"));
                }
                for (b, seq) in seqs.iter_mut().enumerate() {
                    let r = |t: &Tensor| t.narrow(0, offs[b], key[b] as i64);
                    if !done {
                        let (ring, ckv, ci) = &preps[b];
                        let _ = self.attn_core(layer, &r(q), &r(kv), ring, ckv.as_ref(), ci, &r(pos), starts[b], sh[b].win_lo, Some(&r(ob)));
                    }
                    self.attn_commit(layer, seq, &r(kv), &r(pos), starts[b]);
                }
                return None;
            }
            let mut outs = Vec::new();
            for (b, seq) in seqs.iter_mut().enumerate() {
                let n = key[b] as i64;
                let r = |t: &Tensor| t.narrow(0, offs[b], n);
                let into = obuf.map(|ob| ob.narrow(0, offs[b], n));
                let o = self.attn_mid(layer, &r(x), &r(qr), &r(q), &r(kv), seq, &r(pos), starts[b], &mut sh[b], into.as_ref());
                if obuf.is_none() { outs.push(o) }
            }
            obuf.is_none().then(|| Tensor::cat(&outs, 0))
        };
        self.graph_clock.set(self.graph_clock.get() + 1);
        // the graphs depend on the total row count only: their segments work row by row, the per-sequence middle
        // runs eagerly between them with this call's split (so any split of R rows over the sequences shares a graph)
        let gkey: Vec<usize> = vec![total as usize];
        if let Some(g) = graphs.get_mut(&gkey) {
            g.used = self.graph_clock.get();
            let _ = g.pos.copy_(&pos_now);
            let _ = g.h0.copy_(&e.unsqueeze(1).expand([total, c.hc, c.dim], false));
            for (li, l) in engram_layers.iter().enumerate() {
                if let Some(r) = g.rows.get_mut(l) { let _ = r.copy_(&prof("engram", || rows_now(li, *l))); }
            }
            let mut sh: Vec<Shared> = (0..seqs.len()).map(|_| Shared::default()).collect();
            for (i, layer) in self.layers.iter().enumerate() {
                prof("graph.s1", || g.s1[i].replay());
                let _ = run_mid(layer, &g.mid[i], &g.pos, Some(&g.obuf[i]), &mut sh, seqs);
                prof("graph.s3", || g.s3[i].replay());
                rows_dbg(i, &g.mid[i], &g.obuf[i], &g.hs[i]);
            }
            prof("graph.fin", || g.fin.replay());
            if debug_rows() { ROWS_DBG.with(|d| d.borrow_mut().push(("logits".into(), g.logits.narrow(0, 0, 1).to_device(Device::Cpu)))); }
            for (s, &n) in seqs.iter_mut().zip(&key) { s.advance(n as i64); }
            return (g.logits.copy(), g.hid.as_ref().map(|h| h.copy()));
        }
        // first use of this row signature: capture each static segment, replay it at once (a capture computes
        // nothing), run the middle (the memory budget's occupation is returned first and retaken after). Every
        // signature holds its own memory pool: beyond DS41_GRAPH_MAX (48; warm-up captures totals 1..24) the least recently used one is dropped.
        let max = std::env::var("DS41_GRAPH_MAX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(48).max(1);
        while graphs.len() >= max {
            let old = graphs.iter().min_by_key(|(_, g)| g.used).map(|(k, _)| k.clone()).unwrap();
            graphs.remove(&old);
        }
        super::mem::release();
        let pos = pos_now;
        let h0 = e.unsqueeze(1).repeat([1, c.hc, 1]).contiguous();
        let mut rows = HashMap::new();
        for (li, l) in engram_layers.iter().enumerate() { rows.insert(*l, rows_now(li, *l).contiguous()); }
        let pre0 = Tensor::zeros([total, c.hc], (Kind::Float, dev));
        let _ = pre0.narrow(1, 0, 1).fill_(1.0);
        let mut keep = vec![pre0.shallow_clone()];
        let (mut s1, mut s3, mut mid, mut obuf, mut hs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut h, mut pre_mix) = (h0.shallow_clone(), pre0.shallow_clone());
        let mut sh: Vec<Shared> = (0..seqs.len()).map(|_| Shared::default()).collect();
        let mut hid_parts = Vec::new();
        for layer in &self.layers {
            let l = layer.id;
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let hh = match (&layer.engram, rows.get(&l)) { (Some(en), Some(r)) => self.engram(en, &h, r), _ => h.shallow_clone() };
            if want_hid && c.dspark_targets.contains(&l) { hid_parts.push(hh.mean_dim([1].as_slice(), false, Kind::BFloat16)); }
            let (a_pre, a_post, a_comb) = self.hc_mixes(&hh, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base);
            let x = site_z(&layer.attn_norm, &Self::hc_pre(&hh, &pre_mix), c.norm_eps);
            let (qr, q, kv) = self.attn_in(layer, &x, &pos);
            end().expect("capture end");
            let g1 = Owned::take();
            g1.replay();
            s1.push(g1);
            let m4 = [x, qr, q, kv];
            let o = run_mid(layer, &m4, &pos, None, &mut sh, seqs).unwrap();
            begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
            let y = self.attn_out(layer, &o, &pos);
            let h2 = Self::hc_post(&y, &hh, &a_post, &a_comb);
            let (f_pre, f_post, f_comb) = self.hc_mixes(&h2, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base);
            let x2 = site_z(&layer.ffn_norm, &Self::hc_pre(&h2, &a_pre), c.norm_eps);
            let m = self.moe(&layer.moe, &x2);
            let h3 = Self::hc_post(&m, &h2, &f_post, &f_comb);
            end().expect("capture end");
            let g3 = Owned::take();
            g3.replay();
            s3.push(g3);
            let hd = [y.shallow_clone(), h2.shallow_clone(), x2.shallow_clone(), m.shallow_clone(), h3.shallow_clone()];
            keep.extend([hh, a_pre, a_post, a_comb, y, h2, f_post, f_comb, x2, m]);
            rows_dbg(mid.len(), &m4, &o, &hd);
            mid.push(m4);
            obuf.push(o);
            hs.push(hd);
            h = h3;
            pre_mix = f_pre;
            keep.extend([h.shallow_clone(), pre_mix.shallow_clone()]);
        }
        begin_pool(super::mem::GRAPH_POOL).expect("capture begin");
        let x = crate::ablate::clamp_final(rmsn(&Self::hc_pre(&h, &pre_mix), &self.norm, c.norm_eps));
        let logits = self.logits(&x);
        let hid = (!hid_parts.is_empty()).then(|| Tensor::cat(&hid_parts, 1));
        end().expect("capture end");
        let fin = Owned::take();
        fin.replay();
        if debug_rows() { ROWS_DBG.with(|d| d.borrow_mut().push(("logits".into(), logits.narrow(0, 0, 1).to_device(Device::Cpu)))); }
        keep.push(x);
        keep.extend(hid_parts);
        for (s, &n) in seqs.iter_mut().zip(&key) { s.advance(n as i64); }
        let out = (logits.copy(), hid.as_ref().map(|h| h.copy()));
        graphs.insert(gkey, DecodeGraph { pos, h0, rows, s1, s3, fin, mid, obuf, hs, logits, hid, used: self.graph_clock.get(), _keep: keep });
        super::mem::occupy();
        out
    }

    /// One chunk: tokens `ids` at positions [seq.pos, seq.pos + T). Returns the final hidden states [T, dim] BF16 of the
    /// rows in `keep` (all rows when None), and commits the chunk to `seq`.
    pub fn forward(&self, seq: &mut Seq, ids: &[i64], keep: Option<&[i64]>) -> Tensor {
        let _g = tch::no_grad_guard();
        let c = &self.c;
        let t = ids.len() as i64;
        let start = seq.pos;
        assert!(start + t <= seq.cap, "sequence capacity {} exceeded", seq.cap);
        let dev = self.dev;
        let ids_t = Tensor::from_slice(ids).to_device(dev);
        let pos = Tensor::arange_start(start, start + t, (Kind::Int64, dev));
        let hashes = self.push_hashes(seq, ids);   // engram hash ids [engram layer][T * cols]
        let mut h = self.embed_vis(&ids_t, seq.vis.take()).unsqueeze(1).repeat([1, c.hc, 1]);    // [T, hc, dim]
        let _vl = vl_set(self.image_rows(seq, start, start + t));
        let mut pre_mix = Tensor::zeros([t, c.hc], (Kind::Float, dev));
        let _ = pre_mix.narrow(1, 0, 1).fill_(1.0);
        let mut shared = Shared::default();
        let mut hid: Vec<Tensor> = Vec::new();
        for layer in &self.layers {
            self.layer_forward(layer, &mut h, &mut pre_mix, seq, &pos, start, &mut shared, &hashes, &mut hid);
        }
        seq.pos += t;
        if seq.dec_pos == start { seq.dec_pos = seq.pos; }
        seq.ced_tail = None;
        if !hid.is_empty() { *self.last_hidden.borrow_mut() = Some(Tensor::cat(&hid, 1)); }
        let x = Self::hc_pre(&h, &pre_mix);
        let x = match keep { Some(rows) => x.index_select(0, &Tensor::from_slice(rows).to_device(dev)), None => x };
        crate::ablate::clamp_final(rmsn(&x, &self.norm, c.norm_eps))
    }

    /// CED prefill of one prompt chunk (DS41_CED, docs/07; tech report 2.2 and 3.2.2): the encoder layers (the
    /// bottom half) over every token, the decoder's KV source (its compressor and index keys, projected from the
    /// encoder output) over every token, and the encoder output of the last window kept in `seq.ced_tail`. With
    /// `last`, the decoder layers then run over the prompt's tail only: exactly from `seq.dec_pos` when the tail
    /// reaches back to it (the decoder windows are complete), else over the last window with the window attention
    /// truncated to it (Decoder SWA Bounded Replay: approximate, as the model was trained to expect). Returns the final
    /// normed hidden state of the last position [1, dim] (and sets `last_hidden` from the replayed rows) for `last`.
    pub fn prefill_ced(&self, seq: &mut Seq, ids: &[i64], last: bool) -> Option<Tensor> {
        let _g = tch::no_grad_guard();
        let c = &self.c;
        let t = ids.len() as i64;
        let start = seq.pos;
        assert!(start + t <= seq.cap, "sequence capacity {} exceeded", seq.cap);
        let dev = self.dev;
        let enc = (c.n_layers / 2) as usize;
        let pos = Tensor::arange_start(start, start + t, (Kind::Int64, dev));
        let hashes = self.push_hashes(seq, ids);
        // the later engram layers' rows (layer 14) load on background threads while the layers before them compute
        // (tech report 2.4: deterministic addressing lets them be prefetched; the tables are on NVMe here)
        if let Some(hs) = &self.hash {
            if t >= 256 {
                for (li, &l) in hs.layer_ids.iter().enumerate().skip(1) {
                    if self.layers.iter().any(|x| x.id == l) {
                        let hd = engram::rows_async(&self.tables, l, hashes[li].clone(), hs.head_dim);
                        ENGRAM_PENDING.with(|p| p.borrow_mut().insert(l, hd));
                    }
                }
            }
        }
        let mut h = self.embed_vis(&Tensor::from_slice(ids).to_device(dev), seq.vis.take()).unsqueeze(1).repeat([1, c.hc, 1]);
        let vl = vl_set(self.image_rows(seq, start, start + t));
        let mut pre_mix = Tensor::zeros([t, c.hc], (Kind::Float, dev));
        let _ = pre_mix.narrow(1, 0, 1).fill_(1.0);
        let mut sh = Shared::default();
        let mut hid = Vec::new();
        // The fused clamp carries each micro-batch's row base in the offset count pointer (norm_args_off), so the
        // two-micro-batch overlap stays on; only an UNFUSED active clamp (STATS mode / FUSE < 2) must disable it
        // (per-batch z rows would then be locally indexed in the row kernel).
        if overlap_rows() > 0 && t >= 2 * overlap_rows() && self.sp.world > 1 && !crate::ablate::unfused_active() {
            self.encoder_two_batch(&mut h, &mut pre_mix, seq, &pos, start, &hashes, enc);
        } else {
            for layer in self.layers.iter().filter(|l| l.id < enc) {
                self.layer_forward(layer, &mut h, &mut pre_mix, seq, &pos, start, &mut sh, &hashes, &mut hid);
            }
        }
        // the decoder's global KV: its KV source compresses the encoder output (the source's attention input)
        for layer in self.layers.iter().filter(|l| l.id >= enc && c.is_kv_source(l.id)) {
            let x = rmsn(&Self::hc_pre(&h, &pre_mix), &layer.attn_norm, c.norm_eps);
            prof("attn.compress", || self.compress(layer, &x, seq, start, &x, &mut sh));
        }
        let w = c.window;
        let (th, tp) = match seq.ced_tail.take() {
            Some((th, tp, end)) if end == start => (Tensor::cat(&[th, h.shallow_clone()], 0), Tensor::cat(&[tp, pre_mix.shallow_clone()], 0)),
            _ => (h.shallow_clone(), pre_mix.shallow_clone()),
        };
        let n = th.size()[0];
        // checkpoint positions inside the chunk: the encoder tail a chunk ending there leaves
        for p in seq.ck_req.clone() {
            if p > start && p < start + t {
                let e = n - (start + t - p);
                let k = e.min(w);
                seq.ck_cap.entry(p).or_default().ced = Some((th.narrow(0, e - k, k).copy(), tp.narrow(0, e - k, k).copy()));
            }
        }
        let keep = n.min(w);
        seq.ced_tail = Some((th.narrow(0, n - keep, keep).copy(), tp.narrow(0, n - keep, keep).copy(), start + t));
        seq.pos += t;
        drop(vl);
        if !last { return None; }
        let end = seq.pos;
        let (th, tp, _) = seq.ced_tail.as_ref().unwrap();
        let have = th.size()[0];
        let (r0, lo) = if seq.dec_pos >= end - have { (seq.dec_pos, 0) } else { (end - have, end - have) };
        let r = end - r0;
        let mut hr = th.narrow(0, have - r, r).copy();
        let mut pr = tp.narrow(0, have - r, r).copy();
        let rpos = Tensor::arange_start(r0, end, (Kind::Int64, dev));
        let mut sh = Shared { win_lo: lo, skip_compress: true, ..Default::default() };
        let mut hid = Vec::new();
        let _vl = vl_set(self.image_rows(seq, r0, end));
        for layer in self.layers.iter().filter(|l| l.id >= enc) {
            self.layer_forward(layer, &mut hr, &mut pr, seq, &rpos, r0, &mut sh, &[], &mut hid);
        }
        seq.dec_pos = end;
        if !hid.is_empty() { *self.last_hidden.borrow_mut() = Some(Tensor::cat(&hid, 1)); }
        let x = Self::hc_pre(&hr.narrow(0, r - 1, 1), &pr.narrow(0, r - 1, 1));
        Some(crate::ablate::clamp_final(rmsn(&x, &self.norm, c.norm_eps)))
    }

    /// The encoder layers of a prefill chunk as two micro-batches (rows [0, T/2) and [T/2, T)) interleaved per layer, so
    /// each all-reduce runs on the communication stream while the other micro-batch computes (tech report 3.2:
    /// communication-computation overlap): A attention, (A reduce || B attention), (B reduce || A FFN), (A reduce || B
    /// FFN), ... Causal order holds (B's attention follows A's window and KV commits); the result is that of two
    /// chunks of T/2.
    #[allow(clippy::too_many_arguments)]
    fn encoder_two_batch(&self, h: &mut Tensor, pre_mix: &mut Tensor, seq: &mut Seq, pos: &Tensor, start: i64, hashes: &[Vec<i64>], enc: usize) {
        extern "C" {
            fn rs_comm_mode(on: i32) -> i32;
            fn rs_allreduce_bf16t_async(p: *mut std::ffi::c_void, numel: i64) -> i32;
            fn rs_ar_wait(h: i32) -> i32;
        }
        let c = &self.c;
        let t = h.size()[0];
        let ta = t / 2;
        let rows = [(0, ta), (ta, t - ta)];
        let mut hs: Vec<Tensor> = rows.iter().map(|&(a, n)| h.narrow(0, a, n).contiguous()).collect();
        let mut pms: Vec<Tensor> = rows.iter().map(|&(a, n)| pre_mix.narrow(0, a, n).contiguous()).collect();
        let poss: Vec<Tensor> = rows.iter().map(|&(a, n)| pos.narrow(0, a, n)).collect();
        let starts = [start, start + ta];
        let mut shs = [Shared::default(), Shared::default()];
        let launch = |y: &Tensor| -> Option<i32> {
            let hd = unsafe { rs_allreduce_bf16t_async(y.data_ptr(), y.numel() as i64) };
            if hd >= 0 { Some(hd) } else { ar_prefill(y); None }
        };
        let wait = |hd: Option<i32>| { if let Some(x) = hd { assert_eq!(unsafe { rs_ar_wait(x) }, 0, "rs_ar_wait"); } };
        assert_eq!(unsafe { rs_comm_mode(1) }, 0, "rs_comm_mode");
        DEFER_AR.with(|d| d.set(true));
        // the FFN sum of the previous layer for B, still in flight while A's next attention runs
        let mut pending_b: Option<(Tensor, Option<i32>, Tensor, Tensor, Tensor, Tensor)> = None;
        for layer in self.layers.iter().filter(|l| l.id < enc) {
            let l = layer.id;
            // attention of A then B, each sum started as soon as it exists (A's runs under B's attention)
            let mut att: Vec<(Tensor, Option<i32>, Tensor, Tensor, Tensor, Tensor)> = Vec::new();
            // the engram rows of the whole chunk, gathered once
            let erows = match (&layer.engram, &self.hash) {
                (Some(_), Some(hsh)) => { let li = hsh.layer_ids.iter().position(|&x| x == l).unwrap(); Some(self.engram_rows_q(l, &hashes[li], t)) }
                _ => None,
            };
            for b in 0..2 {
                if b == 1 {
                    // B's input: the previous layer's FFN sum (it ran under A's attention)
                    if let Some((y2, hd2, h2, f_pre, f_post, f_comb)) = pending_b.take() {
                        wait(hd2);
                        hs[1] = prof("hc", || Self::hc_post(&bf16(y2), &h2, &f_post, &f_comb));
                        pms[1] = f_pre;
                    }
                }
                let n = rows[b].1;
                if let (Some(e), Some(er)) = (&layer.engram, &erows) {
                    DEFER_AR.with(|d| d.set(false));
                    hs[b] = prof("engram", || engram_skip_images(&hs[b], self.engram_q(e, &hs[b], &er.narrow(rows[b].0, n)), rows[b].0));
                    DEFER_AR.with(|d| d.set(true));
                }
                let residual = hs[b].shallow_clone();
                let (a_pre, a_post, a_comb) = prof("hc", || self.hc_mixes(&hs[b], &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base));
                let x = site_z_off(&layer.attn_norm, &Self::hc_pre(&hs[b], &pms[b]), c.norm_eps, rows[b].0);
                let y = self.attention(layer, &x, seq, &poss[b], starts[b], &mut shs[b]);
                let hd = launch(&y);
                att.push((y, hd, residual, a_pre, a_post, a_comb));
            }
            // the FFN of A and B in one call (the routed experts' weights are read once per chunk)
            let mut x2s = Vec::new();
            let mut ctx = Vec::new();
            for (b, (y, hd, residual, a_pre, a_post, a_comb)) in att.into_iter().enumerate() {
                wait(hd);
                let h2 = prof("hc", || Self::hc_post(&bf16(y), &residual, &a_post, &a_comb));
                let (f_pre, f_post, f_comb) = prof("hc", || self.hc_mixes(&h2, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base));
                x2s.push(site_z_off(&layer.ffn_norm, &Self::hc_pre(&h2, &a_pre), c.norm_eps, [0, ta][b]));
                ctx.push((h2, f_pre, f_post, f_comb));
            }
            let y2 = self.moe(&layer.moe, &Tensor::cat(&x2s, 0));
            let (ya, yb) = (y2.narrow(0, 0, ta), y2.narrow(0, ta, t - ta));
            let hda = launch(&ya);
            let hdb = launch(&yb);
            let mut ctx = ctx.into_iter();
            let (h2, f_pre, f_post, f_comb) = ctx.next().unwrap();
            wait(hda);
            hs[0] = prof("hc", || Self::hc_post(&bf16(ya), &h2, &f_post, &f_comb));
            pms[0] = f_pre;
            let (h2, f_pre, f_post, f_comb) = ctx.next().unwrap();
            pending_b = Some((yb, hdb, h2, f_pre, f_post, f_comb));
        }
        if let Some((y2, hd2, h2, f_pre, f_post, f_comb)) = pending_b.take() {
            wait(hd2);
            hs[1] = prof("hc", || Self::hc_post(&bf16(y2), &h2, &f_post, &f_comb));
            pms[1] = f_pre;
        }
        DEFER_AR.with(|d| d.set(false));
        assert_eq!(unsafe { rs_comm_mode(0) }, 0, "rs_comm_mode");
        *h = Tensor::cat(&hs, 0);
        *pre_mix = Tensor::cat(&pms, 0);
    }

    /// One layer of a chunk (eager): h / pre_mix [T, hc, dim] / [T, hc] in and out; engram rows from `hashes` (the
    /// chunk's own tokens); the DSpark target hidden states (last window rows) appended to `hid`.
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(&self, layer: &Layer, h: &mut Tensor, pre_mix: &mut Tensor, seq: &mut Seq, pos: &Tensor, start: i64, shared: &mut Shared,
                     hashes: &[Vec<i64>], hid: &mut Vec<Tensor>) {
        let c = &self.c;
        let t = h.size()[0];
        let n_hid = t.min(c.window);
        let pos = pos.shallow_clone();
        let mut hl = h.shallow_clone();
        {
            let l = layer.id;
            if let (Some(e), Some(hs)) = (&layer.engram, &self.hash) {
                let li = hs.layer_ids.iter().position(|&x| x == l).unwrap();
                hl = prof("engram", || engram_skip_images(&hl, self.engram(e, &hl, &self.engram_rows(l, &hashes[li], t)), 0));
                if dump_layers().contains(&(1000 + l)) { DUMP.with(|d| d.borrow_mut().push((1000 + l, hl.to_kind(Kind::Float).to_device(Device::Cpu)))); }
            }
            if self.mtp.is_some() && c.dspark_targets.contains(&l) {
                hid.push(hl.narrow(0, t - n_hid, n_hid).mean_dim([1].as_slice(), false, Kind::BFloat16));
            }
            let residual = hl.shallow_clone();
            let (a_pre, a_post, a_comb) = prof("hc", || self.hc_mixes(&hl, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base));
            // site_z takes the pre-norm input (it normalizes itself); x stays the plain normalized value for the
            // capture / dump hooks and, without a clamp, for the attention (as before 6595e5e, which passed the
            // already normalized x to site_z and so normalized twice).
            let xin = Self::hc_pre(&hl, &*pre_mix);
            let x = prof("hc", || rmsn(&xin, &layer.attn_norm, c.norm_eps));
            if crate::ablate::capturing() { crate::ablate::capture(l, &x); }   // ds41-refusal-dir: the raw sublayer input
            if dump_layers().contains(&(4000 + l)) { DUMP.with(|d| d.borrow_mut().push((4000 + l, x.to_kind(Kind::Float).to_device(Device::Cpu)))); }
            let x = if crate::ablate::clamp().is_some() { site_z(&layer.attn_norm, &xin, c.norm_eps) } else { x };
            let x = self.attention(layer, &x, seq, &pos, start, shared);
            if dump_layers().contains(&(5000 + l)) { DUMP.with(|d| d.borrow_mut().push((5000 + l, x.to_kind(Kind::Float).to_device(Device::Cpu)))); }
            hl = prof("hc", || Self::hc_post(&x, &residual, &a_post, &a_comb));
            if dump_layers().contains(&(2000 + l)) { DUMP.with(|d| d.borrow_mut().push((2000 + l, hl.to_kind(Kind::Float).to_device(Device::Cpu)))); }
            let residual = hl.shallow_clone();
            let (f_pre, f_post, f_comb) = prof("hc", || self.hc_mixes(&hl, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base));
            let x = prof("hc", || site_z(&layer.ffn_norm, &Self::hc_pre(&hl, &a_pre), c.norm_eps));
            let x = self.moe(&layer.moe, &x);
            hl = prof("hc", || Self::hc_post(&x, &residual, &f_post, &f_comb));
            *pre_mix = f_pre;
            if dump_layers().contains(&l) { DUMP.with(|d| d.borrow_mut().push((l, hl.to_kind(Kind::Float).to_device(Device::Cpu)))); }
            if dump_layers().contains(&(3000 + l)) { DUMP.with(|d| d.borrow_mut().push((3000 + l, hl.to_kind(Kind::Float).to_device(Device::Cpu)))); }
        }
        *h = hl;
    }

    /// Engram rows of the chunk's hashes for layer l: [T, cols * head_dim] BF16 (host gather from the tables).
    pub(super) fn engram_rows(&self, l: usize, hashes: &[i64], t: i64) -> Tensor {
        let hs = self.hash.as_ref().unwrap();
        // a gather started at the chunk's start (prefill_ced) for exactly these rows
        if let Some(h) = ENGRAM_PENDING.with(|p| p.borrow_mut().remove(&l)) {
            let (w, s) = h.join().expect("engram gather thread");
            if w.len() as i64 == t * hs.cols() as i64 * hs.head_dim {
                return engram::upload(w, s, t * hs.cols() as i64, hs.head_dim, self.dev).view([t, hs.cols() as i64 * hs.head_dim]);
            }
        }
        engram::rows(&self.tables, l, hashes, hs.head_dim).view([t, hs.cols() as i64 * hs.head_dim])
    }
    /// The chunk's engram rows for prefill: the act_quant operands straight from the table bytes (DS41_ENGRAM_Q, default
    /// on; engram::upload_q), else BF16 as engram_rows.
    pub(super) fn engram_rows_q(&self, l: usize, hashes: &[i64], t: i64) -> ERows {
        if !engram_q_on() { return ERows::Bf16(self.engram_rows(l, hashes, t)); }
        let hs = self.hash.as_ref().unwrap();
        let (n, w) = (t * hs.cols() as i64, hs.cols() as i64 * hs.head_dim);
        let pending = ENGRAM_PENDING.with(|p| p.borrow_mut().remove(&l)).map(|h| h.join().expect("engram gather thread"));
        let (q, s) = match pending {
            Some((wb, sb)) if wb.len() as i64 == n * hs.head_dim => engram::upload_q(wb, sb, self.dev),
            _ => engram::rows_q(&self.tables, l, hashes, hs.head_dim),
        };
        ERows::Q(q.view([t, w]), s.view([t, w / 32]))
    }
    pub(super) fn engram(&self, e: &Engram, h: &Tensor, rows: &Tensor) -> Tensor { self.engram_part(e, h, e.wkv.mm_f32(rows)) }
    /// `engram` of the act_quant operands of the rows (ERows::Q) where the projection takes them directly.
    pub(super) fn engram_q(&self, e: &Engram, h: &Tensor, rows: &ERows) -> Tensor {
        match rows {
            ERows::Bf16(r) => self.engram(e, h, r),
            ERows::Q(q, s) => {
                let part = e.wkv.mm_f32_q(q, s).unwrap_or_else(|| e.wkv.mm_f32(&engram::dequant(q, s)));
                self.engram_part(e, h, part)
            }
        }
    }
    /// The engram after the projection: `part` this rank's wkv output rows (FP32).
    fn engram_part(&self, e: &Engram, h: &Tensor, part: Tensor) -> Tensor {
        let c = &self.c;
        let t = h.size()[0];
        // this rank's output rows, zero-padded and summed over the ranks (exact), then rounded like the full Linear
        let full = Tensor::zeros([t, (c.hc + 1) * c.dim], (Kind::Float, self.dev));
        let _ = full.narrow(1, e.row0, part.size()[1]).copy_(&part);
        // (prefill: BF16 transport, the same bits here: every element has one nonzero part and is rounded to BF16)
        if self.sp.world > 1 { ar_prefill(&full); }
        let kv = bf16(full);                                             // [T, (hc+1) dim]
        if ops_fused() && h.kind() == Kind::BFloat16 { return super::kern::engram_gate(h, &kv, &e.qk, c.norm_eps); }
        let key = kv.narrow(1, 0, c.hc * c.dim).to_kind(Kind::Float).view([t, c.hc, c.dim]);
        let value = kv.narrow(1, c.hc * c.dim, c.dim);
        let hf = h.to_kind(Kind::Float);
        let eps = c.norm_eps;
        let rstd = (hf.square().mean_dim([-1].as_slice(), false, Kind::Float) + eps).rsqrt()
            * (key.square().mean_dim([-1].as_slice(), false, Kind::Float) + eps).rsqrt();
        let dot = (&hf * e.qk.unsqueeze(0) * &key).sum_dim_intlist([-1].as_slice(), false, Kind::Float) * rstd * (c.dim as f64).powf(-0.5);
        let gate = dot.abs().clamp_min(1e-6).sqrt().copysign(&dot).sigmoid();
        bf16(hf + gate.unsqueeze(-1) * value.to_kind(Kind::Float).unsqueeze(1))
    }

    fn attention(&self, layer: &Layer, x: &Tensor, seq: &mut Seq, pos: &Tensor, start: i64, sh: &mut Shared) -> Tensor {
        let (qr, q, kv) = prof("attn.qkv", || self.attn_in(layer, x, pos));
        let o = self.attn_mid(layer, x, &qr, &q, &kv, seq, pos, start, sh, None);
        self.attn_out(layer, &o, pos)
    }

    /// Query (low-rank, normed, rotated) and the window key/value of the chunk (rotated, FP8-quantized).
    pub(super) fn attn_in(&self, layer: &Layer, x: &Tensor, pos: &Tensor) -> (Tensor, Tensor, Tensor) {
        let c = &self.c;
        let a = &layer.attn;
        let t = x.size()[0];
        let (hd, rd) = (c.head_dim, c.rope_dim);
        let hl = self.sp.local_heads(c);
        let qakv = a.qakv.mm(x);                                                      // [T, q_lora + head_dim]
        let qr = rmsn(&qakv.narrow(1, 0, c.q_lora), &a.q_norm, c.norm_eps);
        let q = rot_mut(a.wq_b.mm(&qr).view([t, hl, hd]), &a.rope, pos, rd, false);
        let kv = super::kern::act_quant(&rot_mut(rmsn(&qakv.narrow(1, c.q_lora, hd), &a.kv_norm, c.norm_eps), &a.rope, pos, rd, false));
        (qr, q, kv)
    }

    /// The shape-dynamic middle: compressor and indexer state, the attention pool (window ring, the chunk, compressed
    /// KV) and its indices, sparse attention; commits the chunk's window keys. Returns o [T, heads, head_dim] (still
    /// rotated).
    #[allow(clippy::too_many_arguments)]
    /// `into`: where the output goes (the row graphs' middle buffer), else a new tensor.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_mid(&self, layer: &Layer, x: &Tensor, qr: &Tensor, q: &Tensor, kv: &Tensor, seq: &mut Seq, pos: &Tensor, start: i64, sh: &mut Shared, into: Option<&Tensor>) -> Tensor {
        let (ring, ckv, ci) = self.attn_prep(layer, x, qr, seq, pos, start, sh);
        let o = self.attn_core(layer, q, kv, &ring, ckv.as_ref(), &ci, pos, start, sh.win_lo, into);
        self.attn_commit(layer, seq, kv, pos, start);
        o
    }

    /// attn_mid up to the sparse attention: compress / index this chunk as the layer does, -> (the ring of window
    /// keys, the compressed pool, the selected compressed indices [T, K - W] offset past ring + chunk).
    pub(super) fn attn_prep(&self, layer: &Layer, x: &Tensor, qr: &Tensor, seq: &mut Seq, pos: &Tensor, start: i64, sh: &mut Shared) -> (Tensor, Option<Tensor>, Tensor) {
        let c = &self.c;
        let l = layer.id;
        let t = x.size()[0];
        let w = c.window;
        // pool = [ring (w) | chunk (t) | compressed KV]; query at p sees window positions max(0, p-w+1)..=p (the kernel
        // derives those indices from pos / start) and the indexer's top-k compressed positions
        let ring = seq.win[l].shallow_clone();
        let mut ckv = None;
        let mut ci = Tensor::empty([t, 0], (Kind::Int64, self.dev));
        let ratio = c.compress[l];
        if ratio > 0 {
            let src = c.kv_source_of(l).expect("compressed layer without a KV source");
            if c.is_kv_source(l) && !sh.skip_compress { prof("attn.compress", || self.compress(layer, x, seq, start, qr, sh)); }
            let clen = (start + t) / ratio;
            if c.is_index_source(l) {
                sh.topk = Some(prof("attn.indexer", || self.indexer(layer, x, qr, seq, pos, src, ratio, clen, w + t, sh)));
            }
            ci = sh.topk.as_ref().expect("no index source before a reuse layer").shallow_clone();
            ckv = Some(seq.ckv[&src].narrow(0, 0, clen.max(1)));
        }
        (ring, ckv, ci)
    }

    /// The sparse attention of one sequence's chunk over its pools (attn_prep's output).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_core(&self, layer: &Layer, q: &Tensor, kv: &Tensor, ring: &Tensor, ckv: Option<&Tensor>, ci: &Tensor, pos: &Tensor, start: i64, win_lo: i64, into: Option<&Tensor>) -> Tensor {
        let c = &self.c;
        let a = &layer.attn;
        let (hd, w) = (c.head_dim, c.window);
        let o = prof("attn.sparse", || {
            let scale = (hd as f64).powf(-0.5);
            attn_fused().then(|| super::kern::sparse_attn_into(q, [Some(ring), Some(kv), ckv], Some((w, start, pos)), win_lo, ci, &a.sink, scale, into)).flatten()
                .unwrap_or_else(|| {
                    let off = Tensor::arange(w, (Kind::Int64, self.dev));                        // 0..w-1
                    let kpos = pos.unsqueeze(-1) - (w - 1) + off.unsqueeze(0);                    // [T, w] key positions
                    let widx = (&kpos - start + w).where_self(&kpos.ge(start), &kpos.remainder(w)).masked_fill(&kpos.lt(win_lo.max(0)), -1);
                    let mut full = vec![ring.shallow_clone(), kv.shallow_clone()];
                    if let Some(k) = ckv { full.push(k.shallow_clone()); }
                    ops::sparse_attn(q, &Tensor::cat(&full, 0), &Tensor::cat(&[widx, ci.shallow_clone()], 1), &a.sink, scale, 256)
                })
        });
        match into {
            Some(b) if !b.is_set_to(&o) => { let mut b = b.shallow_clone(); let _ = b.copy_(&o); b }
            _ => o,
        }
    }

    /// Commit the chunk's window keys into the sequence's ring.
    pub(super) fn attn_commit(&self, layer: &Layer, seq: &mut Seq, kv: &Tensor, pos: &Tensor, start: i64) {
        let l = layer.id;
        let w = self.c.window;
        let t = kv.size()[0];
        // checkpoint positions inside these rows: the ring as a chunk ending there leaves it (the same commit)
        for p in seq.ck_req.clone() {
            if p > start && p <= start + t && seq.win[l].is_contiguous() && ops_fused() {
                let ring = seq.win[l].copy();
                super::kern::win_commit(&ring, &kv.narrow(0, 0, p - start), &pos.narrow(0, 0, p - start));
                seq.ck_cap.entry(p).or_default().win.insert(l, ring);
            }
        }
        if ops_fused() && seq.win[l].is_contiguous() { super::kern::win_commit(&seq.win[l], kv, pos); } else {
            let keep = t.min(w);
            let tail = kv.narrow(0, t - keep, keep);
            let slots = pos.narrow(0, t - keep, keep).remainder(w);
            let _ = seq.win[l].index_copy_(0, &slots, &tail);
        }
    }

    /// Inverse rotation, grouped low-rank output projection, sum over ranks -> BF16 [T, dim].
    pub(super) fn attn_out(&self, layer: &Layer, o: &Tensor, pos: &Tensor) -> Tensor {
        let c = &self.c;
        let a = &layer.attn;
        let t = o.size()[0];
        let o = rot(o, &a.rope, pos, c.rope_dim, true);
        // official: act_quant(o.view(.., groups, hpg*hd), 32, inplace) before the einsum; hpg*hd is a multiple of 32, so
        // per-32 blocks of the flat [T, gl*hpg*hd] rows are the same blocks
        let o = if !wo_a_act() { o } else if ops_fused() { super::kern::act_quant(&o) } else { act_quant(&o.to_kind(Kind::BFloat16), 32) };
        // grouped low-rank output: per group [T, hpg*hd] x wo_a[g]^T -> [T, o_lora]
        let gl = self.sp.g1 - self.sp.g0;
        let y = prof("attn.out", || {
            // per group: o[:, g] [T, hpg*hd] x wo_a[g]^T -> BF16 [T, o_lora] (the official BF16 einsum; FP8 weight exact)
            let o3 = o.view([t, gl, -1]);
            let oa = if gemv_on() && t <= super::kern::grouped_max_rows() {
                bf16(super::kern::fp8_gemv_grouped(&o3.reshape([t, -1]), &a.wo_a.w, &a.wo_a.s, c.o_lora))
            } else {
                let parts: Vec<Tensor> = (0..gl).map(|g| bf16(a.wo_a.rows(g * c.o_lora, c.o_lora).mm_raw(&o3.select(1, g)))).collect();
                Tensor::cat(&parts, 1)
            };
            a.wo_b.mm(&oa).to_kind(Kind::Float).contiguous()
        });
        ar_bf16(y)
    }

    /// KV source: compress this chunk's tokens into completed groups (with the open group carried in `seq.cstate`),
    /// write their index keys and compressed KV.
    fn compress(&self, layer: &Layer, x: &Tensor, seq: &mut Seq, start: i64, _qr: &Tensor, sh: &mut Shared) {
        let c = &self.c;
        let l = layer.id;
        let a = &layer.attn;
        let cm = a.comp.as_ref().unwrap();
        let ratio = c.compress[l];
        let t = x.size()[0];
        let proj = match sh.comp_in.take() { Some(p) => p, None => self.comp_proj(layer, x) };
        let (latent, g0) = if ratio == 1 {
            (rmsn(&proj, &cm.norm, c.norm_eps), start)
        } else {
            let hd = c.head_dim;
            let (kv, sc) = (proj.narrow(1, 0, hd), proj.narrow(1, hd, hd));
            let (ks, ss) = seq.cstate.get_mut(&l).unwrap();
            // rows of positions [gs*ratio, start+t): the open group's earlier slots from the state, then the chunk
            let gs = start / ratio;
            let pre = start - gs * ratio;
            let kv_all = Tensor::cat(&[ks.narrow(0, 0, pre), kv], 0);
            let sc_all = Tensor::cat(&[ss.narrow(0, 0, pre), sc], 0);
            seq.comp_log.insert(l, (kv_all.shallow_clone(), sc_all.shallow_clone(), gs * ratio));
            // checkpoint positions inside the chunk: the open group a chunk ending there leaves
            for p in seq.ck_req.clone() {
                if p > start && p <= start + t {
                    let (ks, ss) = seq.cstate.get(&l).unwrap();
                    let (kp, sp) = (ks.copy(), ss.copy());
                    let r = p % ratio;
                    if r > 0 {
                        let _ = kp.narrow(0, 0, r).copy_(&kv_all.narrow(0, p - r - gs * ratio, r));
                        let _ = sp.narrow(0, 0, r).copy_(&sc_all.narrow(0, p - r - gs * ratio, r));
                    }
                    seq.ck_cap.entry(p).or_default().cs.insert(l, (kp, sp));
                }
            }
            let (ks, ss) = seq.cstate.get_mut(&l).unwrap();
            let ng = (pre + t) / ratio;
            let rem = pre + t - ng * ratio;
            if rem > 0 {
                let _ = ks.narrow(0, 0, rem).copy_(&kv_all.narrow(0, ng * ratio, rem));
                let _ = ss.narrow(0, 0, rem).copy_(&sc_all.narrow(0, ng * ratio, rem));
            }
            if ng == 0 { return; }
            let kvg = kv_all.narrow(0, 0, ng * ratio).view([ng, ratio, c.head_dim]);
            let scg = sc_all.narrow(0, 0, ng * ratio).view([ng, ratio, c.head_dim]).softmax(1, Kind::Float);
            let lat = (kvg * scg).sum_dim_intlist([1].as_slice(), false, Kind::Float);
            (rmsn(&bf16(lat), &cm.norm, c.norm_eps), gs)
        };
        let ng = latent.size()[0];
        if ng == 0 { return; }
        let gpos = Tensor::arange_start(g0, g0 + ng, (Kind::Int64, self.dev)) * ratio;
        if let Some(ix) = &a.idx {
            if let (Some(wk), Some(kn)) = (&ix.wk, &ix.k_norm) {
                let k = rmsn(&latent.matmul(&wk.tr()), kn, c.norm_eps);
                let k = fp4q(&rot_mut(k, &a.rope, &gpos, c.rope_dim, false), 32, false);
                let _ = seq.ikv.get_mut(&l).unwrap().narrow(0, g0, ng).copy_(&k);
            }
        }
        let lat = fp4q(&rot_mut(latent, &a.rope, &gpos, c.rope_dim, false), 16, true);
        let _ = seq.ckv.get_mut(&l).unwrap().narrow(0, g0, ng).copy_(&lat);
        seq.hw = seq.hw.max((g0 + ng) * ratio);
    }

    /// The compressor's projection of the chunk rows (row by row, so a batch computes it once for all its sequences):
    /// ratio 1 the BF16 KV rows [T, head_dim] before the norm; else [T, 2 head_dim] FP32, KV then gate scores.
    pub(super) fn comp_proj(&self, layer: &Layer, x: &Tensor) -> Tensor {
        let cm = layer.attn.comp.as_ref().unwrap();
        let t = x.size()[0];
        let gemv = gemv_on() && t <= 8;
        if self.c.compress[layer.id] == 1 {
            return if gemv { bf16(super::kern::bf16_gemv(x, &cm.wkv)) } else { x.matmul(&cm.wkv.tr()) };
        }
        if let (Some(w2), true) = (&cm.w2, gemv) { return super::kern::bf16_gemv(x, w2); }
        // (prefill: the two FP32 matmuls of the checkpoint's shapes)
        let xf = x.to_kind(Kind::Float);
        let f = |w: &Tensor| xf.matmul(&w.to_kind(Kind::Float).tr());
        Tensor::cat(&[f(&cm.wkv), f(cm.wgate.as_ref().unwrap())], 1)
    }

    /// The indexer's per-row inputs: (queries [T, nh, id] (rotated, FP4-rounded), head weights [T, nh] BF16); row by
    /// row, so a batch computes them once for all its sequences.
    pub(super) fn index_q(&self, layer: &Layer, x: &Tensor, qr: &Tensor, pos: &Tensor) -> (Tensor, Tensor) {
        let c = &self.c;
        let ix = layer.attn.idx.as_ref().unwrap();
        let t = x.size()[0];
        let (nh, id, rd) = (c.index_heads, c.index_dim, c.rope_dim);
        let q = fp4q(&rot_mut(ix.wq_b.mm(qr).view([t, nh, id]), &layer.attn.rope, pos, rd, false), 32, false);   // [T, nh, id]
        let wp = if gemv_on() && t <= 8 { bf16(super::kern::bf16_gemv(x, &ix.weights_proj)) } else { x.matmul(&ix.weights_proj.tr()) };
        let wts = wp * ((id as f64).powf(-0.5) * (nh as f64).powf(-0.5));                                         // [T, nh] BF16
        (q, wts)
    }

    /// Index source: top-k compressed positions per query, as indices into the attention pool (offset by `offset`;
    /// -1 = none). The candidate source also records the block candidates for later index layers of the chunk.
    #[allow(clippy::too_many_arguments)]
    fn indexer(&self, layer: &Layer, x: &Tensor, qr: &Tensor, seq: &Seq, pos: &Tensor, src: usize, ratio: i64, clen: i64, offset: i64, sh: &mut Shared) -> Tensor {
        let c = &self.c;
        let t = x.size()[0];
        let k = seq.ikv[&src].narrow(0, 0, clen.max(1));                                                          // [Tc, id]
        let k_top = c.index_topk.min(clen);
        let l = layer.id as i64;
        let is_cand = l == c.candidate_source;
        let use_cand = !is_cand && c.candidate_source >= 0 && c.candidate_source < l;
        // prefill chunks: the ranks split the rows (the indexer is replicated, every row computed alone) and gather the
        // selections with one exact FP32 sum (indices < 2^24, every element from one rank); not the candidate source,
        // whose candidate blocks every rank needs whole (DS41_INDEX_SPLIT_ROWS: least rows, 256; 0 off)
        static SPLIT: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
        let split_rows = *SPLIT.get_or_init(|| std::env::var("DS41_INDEX_SPLIT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(256));
        let (rank, world) = (self.sp.rank as i64, self.sp.world as i64);
        if world > 1 && split_rows > 0 && t >= split_rows && !is_cand && k_top > 0 && !crate::tp::graph::capturing() {
            let (r0, r1) = (t * rank / world, t * (rank + 1) / world);
            let n = r1 - r0;
            let r = |v: &Tensor| v.narrow(0, r0, n);
            let (q, wts) = match sh.index_q.take() { Some((q, w)) => (r(&q), r(&w)), None => self.index_q(layer, &r(x), &r(qr), &r(pos)) };
            let cand = use_cand.then(|| r(sh.candidates.as_ref().expect("candidate source not run")));
            let (mine, _) = self.index_blocks(&q, &wts, &k, &r(pos), ratio, clen, k_top, offset, false, cand.as_ref());
            let all = Tensor::zeros([t, k_top], (Kind::Float, self.dev));
            let _ = all.narrow(0, r0, n).copy_(&mine);
            crate::tp::allreduce(&all);
            return all.to_kind(Kind::Int64);
        }
        let (q, wts) = match sh.index_q.take() { Some(v) => v, None => self.index_q(layer, x, qr, pos) };
        let cand = use_cand.then(|| sh.candidates.as_ref().expect("candidate source not run").shallow_clone());
        let (ids, cand_out) = self.index_blocks(&q, &wts, &k, pos, ratio, clen, k_top, offset, is_cand, cand.as_ref());
        if is_cand { sh.candidates = cand_out; }
        ids
    }

    /// The selection of rows q / wts / pos (with their candidate blocks `cand` for later index layers): long contexts in
    /// row blocks whose scores (and the candidate selection's FP32 copies) stay within DS41_INDEX_MB (512) MiB; every
    /// row is computed alone, so blocks change nothing (prefill at 1M tokens: one encoder index layer's [4096, 512k]
    /// scores were 4 GiB).
    #[allow(clippy::too_many_arguments)]
    fn index_blocks(&self, q: &Tensor, wts: &Tensor, k: &Tensor, pos: &Tensor, ratio: i64, clen: i64, k_top: i64, offset: i64, is_cand: bool, cand: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let t = q.size()[0];
        let per_row = clen.max(1) * if is_cand { 14 } else { 3 };
        let budget = std::env::var("DS41_INDEX_MB").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(512) << 20;
        let rb = (budget / per_row).max(1);
        if rb >= t || k_top == 0 { return self.index_rows(q, wts, k, pos, ratio, clen, k_top, offset, is_cand, cand); }
        let mut ids = Vec::new();
        let mut cands = Vec::new();
        for r0 in (0..t).step_by(rb as usize) {
            let n = rb.min(t - r0);
            let r = |x: &Tensor| x.narrow(0, r0, n);
            let (i, co) = self.index_rows(&r(q), &r(wts), k, &r(pos), ratio, clen, k_top, offset, is_cand, cand.map(r).as_ref());
            ids.push(i);
            if let Some(co) = co { cands.push(co); }
        }
        (Tensor::cat(&ids, 0), is_cand.then(|| Tensor::cat(&cands, 0)))
    }

    /// The indexer's selection for rows q [T, nh, id] / wts [T, nh] / pos [T] against index keys k [Tc, id]: (pool
    /// indices [T, k_top], the candidate blocks [T, Tc] when `is_cand`), with `cand` the candidate blocks of these rows
    /// for later index layers.
    #[allow(clippy::too_many_arguments)]
    fn index_rows(&self, q: &Tensor, wts: &Tensor, k: &Tensor, pos: &Tensor, ratio: i64, clen: i64, k_top: i64, offset: i64, is_cand: bool, cand: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let c = &self.c;
        let t = q.size()[0];
        let (nh, id) = (c.index_heads, c.index_dim);
        // score[t, c] = sum_h relu(q[t,h] . k[c]) w[t,h]   (BF16 like the official einsum); -inf where not reachable
        let lens = (pos + 1).floor_divide_scalar(ratio);                                                          // [T]
        let mut score = if attn_fused() && nh == 32 && id == 128 { super::kern::index_score(q, k, wts, pos, ratio) } else {
            let s = q.matmul(&k.tr().unsqueeze(0)).relu();                                                      // [T, nh, Tc]
            let score = (s * wts.unsqueeze(-1)).sum_dim_intlist([1].as_slice(), false, Kind::BFloat16);         // [T, Tc]
            let cpos = Tensor::arange(clen.max(1), (Kind::Int64, self.dev));
            score.masked_fill(&cpos.unsqueeze(0).ge_tensor(&lens.unsqueeze(-1)), f64::NEG_INFINITY)
        };
        let mut cand_out = None;
        if is_cand {
            cand_out = Some(select_candidate_blocks(&score, &lens, c.candidate_topk_blocks, c.candidate_block));
        } else if let Some(cand) = cand {
            score = score.masked_fill(&cand.logical_not(), f64::NEG_INFINITY);
        }
        if k_top == 0 { return (Tensor::full([t, 0], -1, (Kind::Int64, self.dev)), cand_out); }
        if ops_fused() && topk_fused() { return (super::kern::topk_idx(&score, pos, ratio, k_top, offset), cand_out); }
        let (_, ids) = score.topk(k_top, -1, true, false);
        let (ids, _) = ids.sort(-1, false);
        // (no host scalar here: a pageable host-to-device copy would drain the queue every index layer)
        ((&ids + offset).masked_fill(&ids.ge_tensor(&lens.unsqueeze(-1)), -1), cand_out)
    }

    pub(super) fn moe(&self, m: &Moe, x: &Tensor) -> Tensor {
        let c = &self.c;
        let t = x.size()[0];
        let route = || {
            // FP32 router logits: the BF16 GEMV in chunks of 8 rows up to 64 rows (no cuBLAS inside the decode / verify
            // graphs: every capture stream would get its own cuBLAS workspace), FP32 matmul for prefill chunks
            let logits = if gemv_on() && t <= 64 {
                let parts: Vec<Tensor> = (0..t).step_by(8).map(|r| super::kern::bf16_gemv(&x.narrow(0, r, 8.min(t - r)), &m.gate)).collect();
                if parts.len() == 1 { parts.into_iter().next().unwrap() } else { Tensor::cat(&parts, 0) }
            } else { x.to_kind(Kind::Float).matmul(&m.gate.to_kind(Kind::Float).tr()) };
            let pick = |bias: &Tensor| {
                if ops_fused() { return super::kern::route(&logits, bias, m.topk, c.route_scale); }
                let scores = logits.softplus().sqrt();
                let (_, top) = (&scores + bias.unsqueeze(0)).topk(m.topk, -1, true, true);
                let mut wts = scores.gather(1, &top, false);
                if m.topk > 1 { wts = &wts / (wts.sum_dim_intlist([-1].as_slice(), true, Kind::Float) + 1e-20); }
                (top, wts * c.route_scale)
            };
            let (top, wts) = pick(&m.bias);
            // image rows: routed with the VL bias (each row's routing is its own, so routing all rows twice and taking
            // the image rows from the second is the per-row choice)
            match (vl_rows(0, t, true), &m.bias_vl) {
                (Some(mk), Some(bv)) => {
                    let (tv, wv) = pick(bv);
                    let mk = mk.unsqueeze(-1);
                    (tv.where_self(&mk, &top), wv.where_self(&mk, &wts))
                }
                _ => (top, wts),
            }
        };
        let lim = c.swiglu_limit;
        let ex = &m.experts;
        // shared expert, this rank's intermediate slice
        let sm = &m.shared;
        let shared_fused = || { let gu = sm.w13.mm_f32(x); sm.w2.mm_f32(&super::kern::swiglu_quant(&gu, lim)) };
        // decode / verify: the shared expert on a parallel stream beside the router and routed experts
        if let (true, false, Some(p)) = (t <= 16 && ops_fused() && moe_mx_on() && shared_par(), parity(), ex.packed.as_ref()) {
            let w = [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]];
            let y = prof("moe.experts", || super::kern::moe_mx_par(x, route, w, shared_fused));
            return ar_bf16(y);
        }
        let (top, wts) = prof("moe.route", route);
        let ys = prof("moe.shared", || {
            if ops_fused() { return shared_fused(); }
            let gu = sm.w13.mm(x).to_kind(Kind::Float);
            let il = gu.size()[1] / 2;
            let (gate, up) = (gu.narrow(1, 0, il), gu.narrow(1, il, il));
            let (gate, up) = if lim > 0.0 { (gate.clamp_max(lim), up.clamp(-lim, lim)) } else { (gate, up) };
            let hmid = bf16(gate.silu() * up);
            sm.w2.mm_f32(&hmid)
        });
        if let (false, Some(p)) = (parity(), ex.packed.as_ref()) {
            let w = [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]];
            let y = prof("moe.experts", || if moe_mx_on() { super::kern::moe_mx(x, &top, &wts, w, Some(&ys)) }
                                          else { super::kern::moe(&super::kern::act_quant(x), &top, &wts, w, Some(&ys)) });
            return ar_bf16(y);
        }
        let mut y = Tensor::zeros([t, c.dim], (Kind::Float, self.dev));
        let topv: Vec<i64> = Vec::<i64>::try_from(top.to_device(Device::Cpu).view(-1)).unwrap();
        let mut by_e: HashMap<i64, Vec<i64>> = HashMap::new();
        for (i, &e) in topv.iter().enumerate() { by_e.entry(e).or_default().push(i as i64); }
        let mut es: Vec<_> = by_e.into_iter().collect();
        es.sort_by_key(|(e, _)| *e);
        for (e, flat) in es {
            let flat_t = Tensor::from_slice(&flat).to_device(self.dev);
            let rows = flat_t.floor_divide_scalar(m.topk);
            let xs = x.index_select(0, &rows);
            let we = wts.view(-1).index_select(0, &flat_t).unsqueeze(-1);
            let a = super::kern::act_quant(&xs);
            let fast = gemv_on() && xs.size()[0] <= 8;
            let (gate, up) = if fast {
                (bf16(super::kern::mxfp4_gemv(&a, &ex.w1.get(e), &ex.w1s.get(e))).to_kind(Kind::Float),
                 bf16(super::kern::mxfp4_gemv(&a, &ex.w3.get(e), &ex.w3s.get(e))).to_kind(Kind::Float))
            } else {
                let af = a.to_kind(Kind::Float);
                (bf16(af.matmul(&dequant_mxfp4(&ex.w1.get(e), &ex.w1s.get(e), &self.fp4, Kind::Float).tr())).to_kind(Kind::Float),
                 bf16(af.matmul(&dequant_mxfp4(&ex.w3.get(e), &ex.w3s.get(e), &self.fp4, Kind::Float).tr())).to_kind(Kind::Float))
            };
            let (gate, up) = if lim > 0.0 { (gate.clamp_max(lim), up.clamp(-lim, lim)) } else { (gate, up) };
            let hq = super::kern::act_quant(&bf16(gate.silu() * up * we));
            // FP32 partial over this rank's slice of the intermediate dimension
            let out = if fast { super::kern::mxfp4_gemv(&hq, &ex.w2.get(e), &ex.w2s.get(e)) }
                      else { hq.to_kind(Kind::Float).matmul(&dequant_mxfp4(&ex.w2.get(e), &ex.w2s.get(e), &self.fp4, Kind::Float).tr()) };
            let out = if parity() { bf16(out).to_kind(Kind::Float) } else { out };
            y = y.index_add(0, &rows, &out);
        }
        let ys = if parity() { bf16(ys).to_kind(Kind::Float) } else { ys };
        let y = (y + ys).contiguous();
        ar_bf16(y)
    }
}

thread_local! {
    /// DS41_PROFILE=1: per-section wall time (ms, the device synchronized around each section), see `prof`.
    pub static PROF: std::cell::RefCell<Vec<(&'static str, f64)>> = const { std::cell::RefCell::new(Vec::new()) };
}
pub fn profiling() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_PROFILE").as_deref() == Ok("1")) }
/// Runs `f`, adding its time to section `name` when profiling.
pub fn prof<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !profiling() || crate::tp::graph::capturing() { return f(); }
    tch::Cuda::synchronize(0);
    let t = std::time::Instant::now();
    let r = f();
    tch::Cuda::synchronize(0);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    PROF.with(|p| { let mut p = p.borrow_mut(); match p.iter_mut().find(|e| e.0 == name) { Some(e) => e.1 += ms, None => p.push((name, ms)) } });
    r
}
/// Takes the accumulated profile.
pub fn take_prof() -> Vec<(&'static str, f64)> { PROF.with(|p| std::mem::take(&mut *p.borrow_mut())) }

thread_local! {
    /// DS41_DUMP_LAYERS=l0,l1,..: the residual stream [T, hc, dim] after those layers, kept for `ds41-golden`.
    pub static DUMP: std::cell::RefCell<Vec<(usize, Tensor)>> = const { std::cell::RefCell::new(Vec::new()) };
}
fn dump_layers() -> &'static [usize] {
    static L: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::env::var("DS41_DUMP_LAYERS").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default())
}

impl Seq {
    /// n positions run through every layer (row graphs): the decoder keeps up when it had, the CED tail is stale.
    pub(super) fn advance(&mut self, n: i64) {
        if self.dec_pos == self.pos { self.dec_pos += n; }
        self.pos += n;
        self.ced_tail = None;
    }
}

/// What layers of one chunk hand down: the latest top-k indices and the decoder's candidate blocks.
#[derive(Default)]
pub struct Shared {
    pub(super) topk: Option<Tensor>, pub(super) candidates: Option<Tensor>,
    /// SWA bounded replay: window keys before this position are not attended; the KV source does not compress again
    pub(super) win_lo: i64, pub(super) skip_compress: bool,
    /// this sequence's rows of the indexer inputs computed for the whole batch (index_q), taken by the next indexer
    pub(super) index_q: Option<(Tensor, Tensor)>,
    /// likewise the compressor's projection (comp_proj)
    pub(super) comp_in: Option<Tensor>,
}

/// Level one of the hierarchical indexer (model.py select_candidate_blocks): keep the `topk_blocks` best blocks of
/// `block` positions per query (block score = its best position; the query's newest block always kept).
fn select_candidate_blocks(score: &Tensor, lens: &Tensor, topk_blocks: i64, block: i64) -> Tensor {
    let (t, width) = (score.size()[0], score.size()[1]);
    let pad = (block - width % block) % block;
    let s = score.to_kind(Kind::Float);
    let s = if pad > 0 { Tensor::cat(&[s, Tensor::full([t, pad], f64::NEG_INFINITY, (Kind::Float, score.device()))], 1) } else { s };
    let nb = (width + pad) / block;
    let bs = s.view([t, nb, block]).amax([-1].as_slice(), false);
    let last = (lens - 1).floor_divide_scalar(block);
    let blocks = Tensor::arange(nb, (Kind::Int64, score.device()));
    let bs = bs.masked_fill(&blocks.unsqueeze(0).eq_tensor(&last.unsqueeze(-1)), f64::INFINITY);
    let (vals, ids) = bs.topk(topk_blocks.min(nb), -1, true, false);
    let keep = Tensor::zeros([t, nb], (Kind::Bool, score.device())).scatter(1, &ids, &vals.gt(f64::NEG_INFINITY));
    keep.repeat_interleave_self_int(block, 1, None).narrow(1, 0, width)
}
