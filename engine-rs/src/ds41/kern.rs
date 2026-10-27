//! FFI to the DeepSeek-V4.1-Flash decode kernels (shim/ds41.cu), each checked against the libtorch reference ops of
//! ops.rs by `ds41-kernel-test`.
use tch::{Kind, Tensor};

extern "C" {
    fn ds41_fp8_gemv(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_mxfp4_gemv(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_act_quant32(x: *const std::ffi::c_void, out: *mut std::ffi::c_void, numel: i64, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemv_ex(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, xs: i32, rpg: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemv_bx(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, xs: i32, rpg: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_route(logits: *const f32, bias: *const f32, idx: *mut i64, w: *mut f32, t: i32, e: i32, k: i32, scale: f32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_engram_gate(h: *const std::ffi::c_void, kv: *const std::ffi::c_void, qk: *const f32, qk_stride: i32, out: *mut std::ffi::c_void,
                        t: i32, hc: i32, d: i32, eps: f32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_win_commit(ring: *mut std::ffi::c_void, kv: *const std::ffi::c_void, pos: *const i64, t: i32, w: i32, d: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp4_quant(x: *const std::ffi::c_void, y: *mut std::ffi::c_void, n: i64, block: i32, e4m3: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_swiglu_quant(gu: *const f32, out: *mut std::ffi::c_void, t: i32, i: i32, lim: f32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_act_quant_fp8(x: *const std::ffi::c_void, q: *mut std::ffi::c_void, s: *mut std::ffi::c_void, numel: i64, st: *mut std::ffi::c_void) -> i32;
    fn ds41_requant_fp8(wq: *const std::ffi::c_void, ws: *const std::ffi::c_void, q: *mut std::ffi::c_void, s: *mut std::ffi::c_void, numel: i64, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemm_mx_ex(aq: *const std::ffi::c_void, as_: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut std::ffi::c_void,
                           m: i32, n: i32, k: i32, bf16out: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemm_mx(aq: *const std::ffi::c_void, as_: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32,
                        m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemv_mx_q(x: *const std::ffi::c_void, aq: *const std::ffi::c_void, as_: *const std::ffi::c_void, w: *const std::ffi::c_void,
                          ws: *const std::ffi::c_void, y: *mut std::ffi::c_void, m: i32, n: i32, k: i32, ob16: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_gemv_mx_init() -> i32;
    fn ds41_fp8_gemv_mx_smem_max() -> usize;
    fn ds41_bf16_gemv_mx(x: *const std::ffi::c_void, w: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_moe_mx_ws_bytes(r: i32, i: i32, topk: i32, e: i32) -> usize;
    fn rs_lt_rows_find(w: *const f32, full: i32, n: i32, k: i32, verbose: i32) -> i32;
    fn rs_lt_rows_mm(x: *const f32, w: *const f32, y: *mut f32, full: i32, m: i32, n: i32, k: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn ds41_moe_mx_ex(xq: *const std::ffi::c_void, xs: *const std::ffi::c_void, r: i32, idx: *const i64, rw: *const f32, w1: *const std::ffi::c_void,
                      s1: *const std::ffi::c_void, w3: *const std::ffi::c_void, s3: *const std::ffi::c_void, w2: *const std::ffi::c_void,
                      s2: *const std::ffi::c_void, i: i32, add: *const f32, y: *mut f32, ws: *mut std::ffi::c_void, topk: i32, e: i32, skip: i32,
                      st: *mut std::ffi::c_void) -> i32;
    fn ds41_moe_mx_combine_rows(ws: *mut std::ffi::c_void, r: i32, i: i32, topk: i32, e: i32, add: *const f32, y: *mut f32, r0: i32, nr: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_moe_mx_col_passes(r: i32, topk: i32, e: i32) -> i32;
    fn ds41_moe_mx_ws_ptrs(ws: *mut std::ffi::c_void, r: i32, i: i32, topk: i32, e: i32, out: *mut *mut f32, pairmap: *mut *mut i32) -> i32;
    fn ds41_moe_mx_combine(ws: *mut std::ffi::c_void, r: i32, i: i32, topk: i32, e: i32, add: *const f32, y: *mut f32, st: *mut std::ffi::c_void) -> i32;
    fn rs_stream_fork(n: i32) -> i32;
    fn rs_stream_set(i: i32) -> i32;
    fn rs_stream_join(n: i32) -> i32;
    fn ds41_moe_mx(xq: *const std::ffi::c_void, xs: *const std::ffi::c_void, r: i32, idx: *const i64, rw: *const f32, w1: *const std::ffi::c_void,
                   s1: *const std::ffi::c_void, w3: *const std::ffi::c_void, s3: *const std::ffi::c_void, w2: *const std::ffi::c_void,
                   s2: *const std::ffi::c_void, i: i32, add: *const f32, y: *mut f32, ws: *mut std::ffi::c_void, topk: i32, e: i32,
                   st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_mm_grouped_bf16(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut std::ffi::c_void, m: i32, n: i32, k: i32, groups: i32, xs: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_fp8_mm(x: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_hc_slices(t: i32, d: i32) -> i32;
    fn ds41_hc_mix_as(h: *const std::ffi::c_void, f: *const f32, scale: *const f32, base: *const f32, pre: *mut f32, post: *mut f32, comb: *mut f32,
                      part: *mut f32, t: i32, rows: i32, d: i32, norm_eps: f32, eps: f32, iters: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_hc_mix(h: *const std::ffi::c_void, f: *const f32, scale: *const f32, base: *const f32, pre: *mut f32, post: *mut f32, comb: *mut f32,
                   part: *mut f32, t: i32, d: i32, norm_eps: f32, eps: f32, iters: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_hc_pre(h: *const std::ffi::c_void, pre: *const f32, out: *mut std::ffi::c_void, t: i32, d: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_hc_post(x: *const std::ffi::c_void, res: *const std::ffi::c_void, post: *const f32, comb: *const f32, y: *mut std::ffi::c_void, t: i32, d: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_index_score(q: *const std::ffi::c_void, k: *const std::ffi::c_void, w: *const std::ffi::c_void, pos: *const i64, ratio: i32, out: *mut std::ffi::c_void,
                        t: i32, tc: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_sparse_attn(q: *const std::ffi::c_void, a: *const std::ffi::c_void, na: i32, b: *const std::ffi::c_void, nb: i32, c: *const std::ffi::c_void,
                        w: i32, start: i64, lo: i64, pos: *const i64, idx: *const i64, sink: *const f32, out: *mut std::ffi::c_void,
                        t: i32, h: i32, k: i32, scale: f32, ws: *mut std::ffi::c_void, st: *mut std::ffi::c_void) -> i32;
    fn ds41_sparse_attn_ws_bytes(t: i32, h: i32) -> usize;
    fn ds41_topk_idx(score: *const std::ffi::c_void, pos: *const i64, ratio: i32, t: i32, tc: i32, k: i32, offset: i64, out: *mut i64, st: *mut std::ffi::c_void) -> i32;
    fn ds41_sparse_attn_multi(q: *const std::ffi::c_void, seqs: *const AttnSeq, nseq: i32, w: i32, pos: *const i64, sink: *const f32, out: *mut std::ffi::c_void,
                              h: i32, scale: f32, ws: *mut std::ffi::c_void, st: *mut std::ffi::c_void) -> i32;
    fn ds41_bf16_gemv(x: *const std::ffi::c_void, w: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_rms_norm(x: *const std::ffi::c_void, w: *const std::ffi::c_void, y: *mut std::ffi::c_void, rows: i32, n: i32, eps: f32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_rms_norm_cl(x: *const std::ffi::c_void, w: *const std::ffi::c_void, y: *mut std::ffi::c_void, rows: i32, n: i32, eps: f32, cl: *const std::ffi::c_void, st: *mut std::ffi::c_void) -> i32;
    fn ds41_rope(x: *const std::ffi::c_void, pos: *const i64, freqs: *const f32, y: *mut std::ffi::c_void, rows: i32, h: i32, d: i32, rd: i32, inverse: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_rope_act_quant(x: *const std::ffi::c_void, pos: *const i64, freqs: *const f32, y: *mut std::ffi::c_void, rows: i32, h: i32, d: i32, rd: i32, inverse: i32, st: *mut std::ffi::c_void) -> i32;
    fn ds41_moe_ws_bytes(r: i32, i: i32, topk: i32, e: i32) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn ds41_moe(xq: *const std::ffi::c_void, r: i32, idx: *const i64, rw: *const f32, w1: *const std::ffi::c_void, s1: *const std::ffi::c_void,
                w3: *const std::ffi::c_void, s3: *const std::ffi::c_void, w2: *const std::ffi::c_void, s2: *const std::ffi::c_void, i: i32,
                add: *const f32, y: *mut f32, ws: *mut std::ffi::c_void, topk: i32, e: i32, st: *mut std::ffi::c_void) -> i32;
    fn rs_current_stream() -> *mut std::ffi::c_void;
}

/// act_quant (UE8M0 per 32) of a contiguous BF16 tensor, returned dequantized (BF16).
pub fn act_quant(x: &Tensor) -> Tensor {
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let out = Tensor::empty_like(&x);
    assert_eq!(unsafe { ds41_act_quant32(x.data_ptr(), out.data_ptr(), x.numel() as i64, rs_current_stream()) }, 0, "ds41_act_quant32");
    out
}

/// y [m, n] FP32 = x [m, k] (BF16, already act-quantized) . W^T for W FP8 E4M3 [n, k] (Uint8 bytes) with E8M0 [n/32, k/32].
pub fn fp8_gemv(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    assert_eq!(unsafe { ds41_fp8_gemv(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_fp8_gemv");
    y
}

/// y [m, n] FP32 = x [m, k] . W^T for BF16 W [n, k] and m <= 8 (FP32 products and sums, as the official FP32 head).
/// A weight registered with `c12_register` is read from its C12 coding (shim/c12.cu c12_gemv_f32: 12 of 16 bits, the
/// same FMA sequence and reduction as ds41_bf16_gemv on the exact BF16 values: bitwise equal, L0).
pub fn bf16_gemv(x: &Tensor, w: &Tensor) -> Tensor {
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    if let Some(c) = c12_lookup(w) {
        extern "C" { fn glm53_c12_gemv_f32_cuda(x: *const std::ffi::c_void, m8: *const std::ffi::c_void, e4: *const std::ffi::c_void, eb: *const std::ffi::c_void,
            ptr: *const i32, col: *const i32, val: *const std::ffi::c_void, y: *mut f32, m: i32, n: i32, k: i32, pdl: i32, st: *mut std::ffi::c_void) -> i32;
            fn ds41_pdl_enabled() -> i32; }
        assert_eq!(unsafe { glm53_c12_gemv_f32_cuda(x.data_ptr(), c[0].data_ptr(), c[1].data_ptr(), c[2].data_ptr(), c[3].data_ptr().cast(), c[4].data_ptr().cast(),
            c[5].data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, ds41_pdl_enabled(), rs_current_stream()) }, 0, "c12_gemv_f32");
        return y;
    }
    assert_eq!(unsafe { ds41_bf16_gemv(x.data_ptr(), w.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_bf16_gemv");
    y
}

/// DS41_C12 (default on; 0 off): the decode-sized BF16 GEMVs (head, routers, compressor / indexer projections, the
/// draft's Markov head) read a C12 coding of the weight; the BF16 source stays for the prefill-sized FP32 matmuls.
pub fn c12_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_C12").map_or(true, |v| v != "0")) }
type C12Parts = [Tensor; 6];
/// Codings keyed by (data pointer, rows, cols) of the BF16 source; process lifetime (the sources live as long).
fn c12_map() -> &'static std::sync::Mutex<std::collections::HashMap<(usize, i64, i64), C12Parts>> {
    static M: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(usize, i64, i64), C12Parts>>> = std::sync::OnceLock::new();
    M.get_or_init(Default::default)
}
fn c12_lookup(w: &Tensor) -> Option<C12Parts> {
    if !c12_on() || w.dim() != 2 { return None; }
    let key = (w.data_ptr() as usize, w.size()[0], w.size()[1]);
    c12_map().lock().unwrap().get(&key).map(|p| [p[0].shallow_clone(), p[1].shallow_clone(), p[2].shallow_clone(), p[3].shallow_clone(), p[4].shallow_clone(), p[5].shallow_clone()])
}
/// Encode a BF16 [n, k] weight for `bf16_gemv` (n % 16 == 0, k % 256 == 0, contiguous; others are skipped) and prove
/// at load that every decoded weight equals its BF16 bits. Returns the coded bytes (0 when skipped).
pub fn c12_register(w: &Tensor) -> i64 {
    if !c12_on() || w.kind() != Kind::BFloat16 || w.dim() != 2 || !w.is_contiguous() || !w.device().is_cuda() { return 0; }
    let (n, k) = (w.size()[0], w.size()[1]);
    if n % 16 != 0 || k % 256 != 0 { return 0; }
    let key = (w.data_ptr() as usize, n, k);
    if c12_map().lock().unwrap().contains_key(&key) { return 0; }
    let p = crate::c12::encode_parts(w);
    extern "C" { fn glm53_c12_verify_exact_cuda(src: *const std::ffi::c_void, m8: *const std::ffi::c_void, e4: *const std::ffi::c_void, eb: *const std::ffi::c_void,
        ptr: *const i32, col: *const i32, val: *const std::ffi::c_void, n: i32, k: i32, bad: *mut u64, st: *mut std::ffi::c_void) -> i32; }
    let bad = Tensor::zeros([1], (Kind::Int64, w.device()));
    assert_eq!(unsafe { glm53_c12_verify_exact_cuda(w.data_ptr(), p[0].data_ptr(), p[1].data_ptr(), p[2].data_ptr(), p[3].data_ptr().cast(), p[4].data_ptr().cast(),
        p[5].data_ptr(), n as i32, k as i32, bad.data_ptr().cast(), rs_current_stream()) }, 0, "c12_verify_exact");
    let b = bad.int64_value(&[0]);
    assert_eq!(b, 0, "DS41 C12: {b} decoded weights of [{n}, {k}] differ from the BF16 source");
    let bytes: i64 = p.iter().map(|t| t.numel() as i64 * t.kind().elt_size_in_bytes() as i64).sum();
    c12_map().lock().unwrap().insert(key, p);
    bytes
}

/// DS41_GEMV_BX=0: wo_a on the CUDA-core FP8 GEMV (cost linear in the rows) instead of the BF16-MMA one.
pub fn gemv_bx_on() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_GEMV_BX").map_or(true, |v| v != "0")) }
/// Largest row count of fp8_gemv_grouped.
pub fn grouped_max_rows() -> i64 { if gemv_bx_on() { 16 } else { 8 } }

/// Grouped (block-diagonal) FP8 GEMV: W [G * rpg, k] whose row group g reads the input's g-th k elements; x [m, G * k]
/// (m <= grouped_max_rows()) -> FP32 [m, G * rpg] (wo_a's groups in one launch).
pub fn fp8_gemv_grouped(x: &Tensor, w: &Tensor, ws: &Tensor, rpg: i64) -> Tensor {
    let (m, gk) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let k = w.size()[1];
    assert_eq!(gk, (n / rpg) * k, "fp8_gemv_grouped shapes");
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    if gemv_bx_on() && k % 64 == 0 {
        assert_eq!(unsafe { ds41_fp8_gemv_bx(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, gk as i32, rpg as i32, rs_current_stream()) }, 0, "ds41_fp8_gemv_bx");
        return y;
    }
    assert_eq!(unsafe { ds41_fp8_gemv_ex(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, gk as i32, rpg as i32, rs_current_stream()) }, 0, "ds41_fp8_gemv_ex");
    y
}

/// MoE routing (sqrtsoftplus, biased top-k, normalized weights * scale) from FP32 logits [T, E]: (experts [T, k] Int64,
/// weights [T, k] FP32).
pub fn route(logits: &Tensor, bias: &Tensor, k: i64, scale: f64) -> (Tensor, Tensor) {
    let (t, e) = (logits.size()[0], logits.size()[1]);
    let l = logits.to_kind(Kind::Float).contiguous();
    let b = bias.to_kind(Kind::Float).contiguous();
    let idx = Tensor::empty([t, k], (Kind::Int64, l.device()));
    let w = Tensor::empty([t, k], (Kind::Float, l.device()));
    assert_eq!(unsafe { ds41_route(l.data_ptr() as *const f32, b.data_ptr() as *const f32, idx.data_ptr() as *mut i64, w.data_ptr() as *mut f32,
        t as i32, e as i32, k as i32, scale as f32, rs_current_stream()) }, 0, "ds41_route");
    (idx, w)
}
/// Router logits FP32 [m, n] of x [m, k] (BF16 or FP32) and gate_f32 [n, k] with the bits these rows get in the FP32
/// matmul of a `full`-row call (lt.cpp rs_lt_rows_*: the cublasLt configuration that reproduces at::matmul's for that
/// shape, found and checked once per shape, run on the m rows only). None: no such configuration (the caller does the
/// full-shape call). DS41_ROUTER_ROWS=0: always None.
pub fn router_rows(x: &Tensor, gate_f32: &Tensor, full: i64) -> Option<Tensor> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS41_ROUTER_ROWS").map_or(true, |v| v != "0")) { return None; }
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = gate_f32.size()[0];
    assert!(gate_f32.kind() == Kind::Float && gate_f32.is_contiguous() && gate_f32.size()[1] == k, "router_rows: gate");
    let verbose = std::env::var("DS41_ROUTER_ROWS_LOG").is_ok_and(|v| v == "1") as i32;
    if unsafe { rs_lt_rows_find(gate_f32.data_ptr() as *const f32, full as i32, n as i32, k as i32, verbose) } != 0 { return None; }
    let xf = x.to_kind(Kind::Float).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    let rc = unsafe { rs_lt_rows_mm(xf.data_ptr() as *const f32, gate_f32.data_ptr() as *const f32, y.data_ptr() as *mut f32, full as i32, m as i32, n as i32, k as i32) };
    (rc == 0).then_some(y)
}
/// Shared-expert SwiGLU + act_quant from the stacked gate/up FP32 sums gu [T, 2 I] -> BF16 [T, I] (w2's input).
/// Engram gate: h [T, hc, D] BF16, kv [T, (hc + 1) D] BF16 (keys of every stream, then the value), qk FP32 [hc, D]
/// or [D] -> bf16(h + gate * value) [T, hc, D] (model.rs Model::engram), one CTA per (row, stream).
pub fn engram_gate(h: &Tensor, kv: &Tensor, qk: &Tensor, eps: f64) -> Tensor {
    let (t, hc, d) = (h.size()[0], h.size()[1], h.size()[2]);
    let h = h.to_kind(Kind::BFloat16).contiguous();
    let kv = kv.to_kind(Kind::BFloat16).contiguous();
    let qk = qk.to_kind(Kind::Float).contiguous();
    let stride = if qk.dim() == 1 { 0 } else { d as i32 };
    let out = Tensor::empty_like(&h);
    assert_eq!(unsafe { ds41_engram_gate(h.data_ptr(), kv.data_ptr(), qk.data_ptr() as *const f32, stride, out.data_ptr(), t as i32, hc as i32, d as i32, eps as f32, rs_current_stream()) }, 0, "ds41_engram_gate");
    out
}
/// Window commit: ring [w, d] BF16 row pos[i] % w = kv[i] for the last min(t, w) of the t rows (one kernel instead of
/// remainder + index_copy).
pub fn win_commit(ring: &Tensor, kv: &Tensor, pos: &Tensor) {
    let (w, d) = (ring.size()[0], ring.size()[1]);
    let t = kv.size()[0];
    let kv = kv.to_kind(Kind::BFloat16).contiguous();
    let pos = pos.to_kind(Kind::Int64).contiguous();
    assert!(ring.is_contiguous() && ring.kind() == Kind::BFloat16 && kv.size()[1] == d);
    assert_eq!(unsafe { ds41_win_commit(ring.data_ptr(), kv.data_ptr(), pos.data_ptr() as *const i64, t as i32, w as i32, d as i32, rs_current_stream()) }, 0, "ds41_win_commit");
}
/// FP4 E2M1 fake quantization of BF16 x per `block` (ops::fp4_quant) in one kernel.
pub fn fp4_quant(x: &Tensor, block: i64, e4m3_scale: bool) -> Tensor {
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty_like(&x);
    assert_eq!(unsafe { ds41_fp4_quant(x.data_ptr(), y.data_ptr(), x.numel() as i64, block as i32, e4m3_scale as i32, rs_current_stream()) }, 0, "ds41_fp4_quant");
    y
}
pub fn swiglu_quant(gu: &Tensor, lim: f64) -> Tensor {
    let (t, i2) = (gu.size()[0], gu.size()[1]);
    let g = gu.to_kind(Kind::Float).contiguous();
    let out = Tensor::empty([t, i2 / 2], (Kind::BFloat16, g.device()));
    assert_eq!(unsafe { ds41_swiglu_quant(g.data_ptr() as *const f32, out.data_ptr(), t as i32, (i2 / 2) as i32, lim as f32, rs_current_stream()) }, 0, "ds41_swiglu_quant");
    out
}

/// `bf16_gemv` for 1..16 rows on the BF16 MMA. Not used: unlike the FP8 one, the CUDA-core BF16 GEMV stays memory-bound
/// up to 8 rows (head [43136 x 5120]: 1.86 ms at m = 1, 1.89 ms at m = 8; this one 2.07 ms).
pub fn bf16_gemv_rows(x: &Tensor, w: &Tensor) -> Tensor {
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    if m == 1 || n % 8 != 0 || k % 32 != 0 || m > 16 { return bf16_gemv(x, w); }
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    assert_eq!(unsafe { ds41_bf16_gemv_mx(x.data_ptr(), w.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_bf16_gemv_mx");
    y
}

/// Grouped `fp8_mm` rounded to BF16 (wo_a): x BF16 [m, groups k], W [groups n, k] -> BF16 [m, groups n], group g's
/// columns x[:, g k .. +k] . W[g n .. +n]^T: the bits of `bf16(fp8_mm(x[:, g].contiguous(), W[g]))` per group, cat
/// (no input copies, FP32 partials or cat; one launch).
pub fn fp8_mm_grouped_bf16(x: &Tensor, w: &Tensor, ws: &Tensor, n: i64, groups: i64) -> Tensor {
    let m = x.size()[0];
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let k = x.size()[1] / groups;
    assert!(x.size()[1] == groups * k && w.size() == [groups * n, k], "fp8_mm_grouped_bf16");
    let y = Tensor::empty([m, groups * n], (Kind::BFloat16, x.device()));
    assert_eq!(unsafe { ds41_fp8_mm_grouped_bf16(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr(), m as i32, n as i32, k as i32, groups as i32, (groups * k) as i32, rs_current_stream()) }, 0, "ds41_fp8_mm_grouped_bf16");
    y
}
/// Same contract as `fp8_gemv` for any m (BF16 MMA tiles; used for m > 8).
pub fn fp8_mm(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    assert_eq!(unsafe { ds41_fp8_mm(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_fp8_mm");
    y
}

/// y [m, n] FP32 = x [m, k] (BF16, act-quantized) . W^T for MXFP4 W [n, k/2] (Uint8) with E8M0 [n, k/32].
pub fn mxfp4_gemv(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([m, n], (Kind::Float, x.device()));
    assert_eq!(unsafe { ds41_mxfp4_gemv(x.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_mxfp4_gemv");
    y
}

/// mHC mixes (shim/ds41_hc.cu): h [T, 4, D] BF16 -> (pre [T, 4], post [T, 4], comb [T, 4, 4]) FP32.
pub fn hc_mix(h: &Tensor, f: &Tensor, scale: &Tensor, base: &Tensor, norm_eps: f64, eps: f64, iters: i64) -> (Tensor, Tensor, Tensor) {
    let (t, d) = (h.size()[0], h.size()[2]);
    let h = h.to_kind(Kind::BFloat16).contiguous();
    let o = (Kind::Float, h.device());
    let (pre, post, comb) = (Tensor::empty([t, 4], o), Tensor::empty([t, 4], o), Tensor::empty([t, 4, 4], o));
    let part = Tensor::empty([unsafe { ds41_hc_slices(t as i32, d as i32) } as i64 * t * 25], o);
    let rc = unsafe { ds41_hc_mix(h.data_ptr(), f.data_ptr() as *const f32, scale.data_ptr() as *const f32, base.data_ptr() as *const f32,
        pre.data_ptr() as *mut f32, post.data_ptr() as *mut f32, comb.data_ptr() as *mut f32, part.data_ptr() as *mut f32,
        t as i32, d as i32, norm_eps as f32, eps as f32, iters as i32, rs_current_stream()) };
    assert_eq!(rc, 0, "ds41_hc_mix");
    (pre, post, comb)
}
/// `hc_mix` of these rows as a call over `rows` rows computes them (sequence-parallel prefill: a rank's share of a
/// chunk); None when `rows` does not take the register-blocked path.
pub fn hc_mix_as(h: &Tensor, rows: i64, f: &Tensor, scale: &Tensor, base: &Tensor, norm_eps: f64, eps: f64, iters: i64) -> Option<(Tensor, Tensor, Tensor)> {
    let (t, d) = (h.size()[0], h.size()[2]);
    let h = h.to_kind(Kind::BFloat16).contiguous();
    let o = (Kind::Float, h.device());
    let (pre, post, comb) = (Tensor::empty([t, 4], o), Tensor::empty([t, 4], o), Tensor::empty([t, 4, 4], o));
    let part = Tensor::empty([unsafe { ds41_hc_slices(rows as i32, d as i32) } as i64 * t * 25], o);
    let rc = unsafe { ds41_hc_mix_as(h.data_ptr(), f.data_ptr() as *const f32, scale.data_ptr() as *const f32, base.data_ptr() as *const f32,
        pre.data_ptr() as *mut f32, post.data_ptr() as *mut f32, comb.data_ptr() as *mut f32, part.data_ptr() as *mut f32,
        t as i32, rows as i32, d as i32, norm_eps as f32, eps as f32, iters as i32, rs_current_stream()) };
    (rc == 0).then_some((pre, post, comb))
}
/// sum_j pre[t][j] h[t][j] -> BF16 [T, D]
pub fn hc_pre(h: &Tensor, pre: &Tensor) -> Tensor {
    let (t, d) = (h.size()[0], h.size()[2]);
    let h = h.to_kind(Kind::BFloat16).contiguous();
    let out = Tensor::empty([t, d], (Kind::BFloat16, h.device()));
    assert_eq!(unsafe { ds41_hc_pre(h.data_ptr(), pre.contiguous().data_ptr() as *const f32, out.data_ptr(), t as i32, d as i32, rs_current_stream()) }, 0, "ds41_hc_pre");
    out
}
/// y[t][k] = post[t][k] x[t] + sum_j comb[t][j][k] res[t][j] -> BF16 [T, 4, D]
pub fn hc_post(x: &Tensor, res: &Tensor, post: &Tensor, comb: &Tensor) -> Tensor {
    let (t, d) = (x.size()[0], x.size()[1]);
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let res = res.to_kind(Kind::BFloat16).contiguous();
    let y = Tensor::empty([t, 4, d], (Kind::BFloat16, x.device()));
    assert_eq!(unsafe { ds41_hc_post(x.data_ptr(), res.data_ptr(), post.contiguous().data_ptr() as *const f32, comb.contiguous().data_ptr() as *const f32,
        y.data_ptr(), t as i32, d as i32, rs_current_stream()) }, 0, "ds41_hc_post");
    y
}

/// Sparse attention (shim/ds41_attn.cu), same contract as ops::sparse_attn for H <= 32 heads (BF16 output); `kv2`
/// (optional) is a second pool addressed by indices >= kv rows (read in place, no concatenation). None when the kernel
/// cannot run (more heads, or the device refused the shared memory).
pub fn sparse_attn(q: &Tensor, kv: &Tensor, kv2: Option<&Tensor>, idx: &Tensor, sink: &Tensor, scale: f64) -> Option<Tensor> {
    sparse_attn_pools(q, [Some(kv), None, kv2], None, idx, sink, scale)
}

/// The general form: pools [a, b, c] read in place (index ranges in that order), and `window` = (W, start, pos): the
/// first W indices of every query are its sliding window (ring a of W slots by position % W, chunk keys b from
/// position `start`), computed by the kernel; `idx` [T, K - W] holds the rest.
pub fn sparse_attn_pools(q: &Tensor, pools: [Option<&Tensor>; 3], window: Option<(i64, i64, &Tensor)>, idx: &Tensor, sink: &Tensor, scale: f64) -> Option<Tensor> {
    sparse_attn_into(q, pools, window, 0, idx, sink, scale, None)
}
/// `sparse_attn_pools` writing into `into` (contiguous BF16 [T, H, 512]) when given; window keys before position `lo`
/// are not attended (SWA bounded replay; 0 = none).
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_into(q: &Tensor, pools: [Option<&Tensor>; 3], window: Option<(i64, i64, &Tensor)>, lo: i64, idx: &Tensor, sink: &Tensor, scale: f64, into: Option<&Tensor>) -> Option<Tensor> {
    let (t, h, d) = (q.size()[0], q.size()[1], q.size()[2]);
    let w = window.map_or(0, |x| x.0);
    let k = w + idx.size()[1];
    if h > 32 || d != 512 || k == 0 { return None; }
    let q = q.to_kind(Kind::BFloat16).contiguous();
    let p: Vec<Option<Tensor>> = pools.iter().map(|p| p.map(|x| x.to_kind(Kind::BFloat16).contiguous())).collect();
    let rows = |i: usize| p[i].as_ref().map_or(0, |x| x.size()[0] as i32);
    let ptr = |i: usize| p[i].as_ref().map_or(std::ptr::null(), |x| x.data_ptr() as *const std::ffi::c_void);
    let idx = idx.to_kind(Kind::Int64).contiguous();
    let pos = window.map(|x| x.2.to_kind(Kind::Int64).contiguous());
    let sink = sink.to_kind(Kind::Float).contiguous();
    let out = match into {
        Some(o) if o.is_contiguous() && o.kind() == Kind::BFloat16 && o.size() == [t, h, d] => o.shallow_clone(),
        _ => Tensor::empty([t, h, d], (Kind::BFloat16, q.device())),
    };
    let wsb = unsafe { ds41_sparse_attn_ws_bytes(t as i32, h as i32) } as i64;
    let ws = (wsb > 0).then(|| Tensor::empty([wsb], (Kind::Uint8, q.device())));
    let rc = unsafe { ds41_sparse_attn(q.data_ptr(), ptr(0), rows(0), ptr(1), rows(1), ptr(2), w as i32, window.map_or(0, |x| x.1), lo,
        pos.as_ref().map_or(std::ptr::null(), |x| x.data_ptr() as *const i64), idx.data_ptr() as *const i64, sink.data_ptr() as *const f32, out.data_ptr(),
        t as i32, h as i32, k as i32, scale as f32, ws.as_ref().map_or(std::ptr::null_mut(), |w| w.data_ptr()), rs_current_stream()) };
    if rc == 2 { return None; }
    assert_eq!(rc, 0, "ds41_sparse_attn");
    Some(out)
}

/// One sequence of sparse_attn_multi (shim Ds41AttnSeq).
#[repr(C)]
pub struct AttnSeq { a: *const std::ffi::c_void, b: *const std::ffi::c_void, c: *const std::ffi::c_void, idx: *const i64, start: i64, lo: i64, na: i32, n: i32, k: i32, pad: i32 }

/// The sparse attention of several sequences' rows in one launch (decode / verify batches; same bits as one call per
/// sequence): q [T, H, 512] and out [T, H, 512] BF16 of all rows, pos [T]; per sequence, in row order, (ring, chunk keys
/// [n, 512] (n = its rows), compressed pool, indices [n, K - W], start, lo). false: not supported for this batch
/// (nothing ran; the caller goes sequence by sequence).
#[allow(clippy::type_complexity)]
pub fn sparse_attn_multi(q: &Tensor, seqs: &[(&Tensor, &Tensor, Option<&Tensor>, &Tensor, i64, i64)], w: i64, pos: &Tensor, sink: &Tensor, scale: f64, out: &Tensor) -> bool {
    let (t, h, d) = (q.size()[0], q.size()[1], q.size()[2]);
    let bf = |x: &Tensor| x.kind() == Kind::BFloat16 && x.is_contiguous();
    if d != 512 || h > 32 || t > 16 || !bf(q) || !bf(out) || out.size() != q.size() || pos.kind() != Kind::Int64 || !pos.is_contiguous() { return false; }
    if seqs.iter().any(|(a, b, c, i, _, _)| !bf(a) || !bf(b) || c.is_some_and(|c| !bf(c)) || i.kind() != Kind::Int64 || !i.is_contiguous()) { return false; }
    if seqs.iter().map(|s| s.1.size()[0]).sum::<i64>() != t { return false; }
    let desc: Vec<AttnSeq> = seqs.iter().map(|(a, b, c, i, start, lo)| AttnSeq {
        a: a.data_ptr(), b: b.data_ptr(), c: c.map_or(std::ptr::null(), |c| c.data_ptr() as *const std::ffi::c_void), idx: i.data_ptr() as *const i64,
        start: *start, lo: *lo, na: a.size()[0] as i32, n: b.size()[0] as i32, k: (w + i.size()[1]) as i32, pad: 0 }).collect();
    let sink = sink.to_kind(Kind::Float).contiguous();
    let wsb = unsafe { ds41_sparse_attn_ws_bytes(t as i32, h as i32) } as i64;
    let ws = Tensor::empty([wsb], (Kind::Uint8, q.device()));
    let rc = unsafe { ds41_sparse_attn_multi(q.data_ptr(), desc.as_ptr(), desc.len() as i32, w as i32, pos.data_ptr() as *const i64, sink.data_ptr() as *const f32,
        out.data_ptr(), h as i32, scale as f32, ws.data_ptr(), rs_current_stream()) };
    if rc == 2 { return false; }
    assert_eq!(rc, 0, "ds41_sparse_attn_multi");
    true
}

/// The indexer's selection: the k largest of each score row [T, Tc] BF16 (ties to the lower index) in ascending index
/// order, + offset, -1 where the index >= (pos + 1) / ratio -> [T, k] Int64.
pub fn topk_idx(score: &Tensor, pos: &Tensor, ratio: i64, k: i64, offset: i64) -> Tensor {
    let (t, tc) = (score.size()[0], score.size()[1]);
    let score = score.to_kind(Kind::BFloat16).contiguous();
    let pos = pos.to_kind(Kind::Int64).contiguous();
    let out = Tensor::empty([t, k], (Kind::Int64, score.device()));
    assert_eq!(unsafe { ds41_topk_idx(score.data_ptr(), pos.data_ptr() as *const i64, ratio as i32, t as i32, tc as i32, k as i32, offset, out.data_ptr() as *mut i64, rs_current_stream()) }, 0, "ds41_topk_idx");
    out
}

/// Indexer scores [T, Tc] BF16 (q [T, 32, 128], k [Tc, 128], w [T, 32] BF16; -inf past (pos + 1) / ratio).
pub fn index_score(q: &Tensor, k: &Tensor, w: &Tensor, pos: &Tensor, ratio: i64) -> Tensor {
    let (t, tc) = (q.size()[0], k.size()[0]);
    assert_eq!((q.size()[1], q.size()[2], k.size()[1]), (32, 128, 128), "index_score: shapes");
    let q = q.to_kind(Kind::BFloat16).contiguous();
    let k = k.to_kind(Kind::BFloat16).contiguous();
    let w = w.to_kind(Kind::BFloat16).contiguous();
    let pos = pos.to_kind(Kind::Int64).contiguous();
    let out = Tensor::empty([t, tc], (Kind::BFloat16, q.device()));
    assert_eq!(unsafe { ds41_index_score(q.data_ptr(), k.data_ptr(), w.data_ptr(), pos.data_ptr() as *const i64, ratio as i32, out.data_ptr(), t as i32, tc as i32, rs_current_stream()) }, 0, "ds41_index_score");
    out
}

/// RMSNorm of BF16 rows (last dim) with a BF16 weight -> BF16 (ops::rms_norm semantics).
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Tensor {
    let n = *x.size().last().unwrap();
    let x = x.contiguous();
    let y = Tensor::empty_like(&x);
    assert_eq!(unsafe { ds41_rms_norm(x.data_ptr(), w.data_ptr(), y.data_ptr(), (x.numel() as i64 / n) as i32, n as i32, eps as f32, rs_current_stream()) }, 0, "ds41_rms_norm");
    y
}
/// rms_norm with the conditional-clamp epilogue fused into the vec kernel (docs/05-消融.md §3a, GLM FUSE=2 style):
/// the site's z is clamped on its fp32 register values before the BF16 store -- no fp32 round-trip, no extra kernel.
/// `a`'s count pointer may already be offset to the caller's row base (encoder two-micro-batch). None when the shape
/// has no fused path (non-BF16, n not vec-eligible, or the kernel declined) -- the caller falls back to the row kernel.
pub fn rms_norm_cl(x: &Tensor, w: &Tensor, eps: f64, a: &crate::ablate::ClampArgs) -> Option<Tensor> {
    let n = *x.size().last().unwrap();
    if x.kind() != Kind::BFloat16 || w.kind() != Kind::BFloat16 || n % 8 != 0 || n > 8 * 2 * 512 { return None; }
    let x = x.contiguous();
    let y = Tensor::empty_like(&x);
    let rc = unsafe { ds41_rms_norm_cl(x.data_ptr(), w.data_ptr(), y.data_ptr(), (x.numel() as i64 / n) as i32, n as i32, eps as f32, a as *const _ as *const std::ffi::c_void, rs_current_stream()) };
    if rc != 0 { return None; }
    Some(y)
}
/// Rotary embedding of BF16 x [n, (H,) d] at positions pos [n] (Int64) with the given frequencies (ops::rope semantics).
pub fn rope(x: &Tensor, pos: &Tensor, freqs: &Tensor, rd: i64, inverse: bool) -> Tensor {
    let sz = x.size();
    let (n, d) = (sz[0], *sz.last().unwrap());
    let h = x.numel() as i64 / n / d;
    let x = x.contiguous();
    let y = Tensor::empty_like(&x);
    let pos = pos.to_kind(Kind::Int64).contiguous();
    assert_eq!(unsafe { ds41_rope(x.data_ptr(), pos.data_ptr() as *const i64, freqs.data_ptr() as *const f32, y.data_ptr(), n as i32, h as i32, d as i32, rd as i32, inverse as i32, rs_current_stream()) }, 0, "ds41_rope");
    y
}

/// `rope` in place on a contiguous BF16 tensor this caller owns (only the rotary dims are read and written).
/// act_quant(rope(x)) in one pass (BF16 [n, .., d]): bitwise `act_quant(&rope(x, ..))`.
pub fn rope_act_quant(x: &Tensor, pos: &Tensor, freqs: &Tensor, rd: i64, inverse: bool) -> Tensor {
    let sz = x.size();
    let (n, d) = (sz[0], *sz.last().unwrap());
    let h = x.numel() as i64 / n / d;
    let x = x.contiguous();
    let y = Tensor::empty_like(&x);
    let pos = pos.to_kind(Kind::Int64).contiguous();
    assert_eq!(unsafe { ds41_rope_act_quant(x.data_ptr(), pos.data_ptr() as *const i64, freqs.data_ptr() as *const f32, y.data_ptr(), n as i32, h as i32, d as i32, rd as i32, inverse as i32, rs_current_stream()) }, 0, "ds41_rope_act_quant");
    y
}
pub fn rope_inplace(x: &Tensor, pos: &Tensor, freqs: &Tensor, rd: i64, inverse: bool) {
    let sz = x.size();
    let (n, d) = (sz[0], *sz.last().unwrap());
    let h = x.numel() as i64 / n / d;
    assert!(x.is_contiguous() && x.kind() == Kind::BFloat16);
    let pos = pos.to_kind(Kind::Int64).contiguous();
    assert_eq!(unsafe { ds41_rope(x.data_ptr(), pos.data_ptr() as *const i64, freqs.data_ptr() as *const f32, x.data_ptr(), n as i32, h as i32, d as i32, rd as i32, inverse as i32, rs_current_stream()) }, 0, "ds41_rope");
}

/// Packs stacked expert weights [E, rows, K/2] / scales [E, rows, K/32] into the tile layout of shim/ds41_moe.cu:
/// [E * rows / 8][K / 128][8][4][16] bytes and [E * rows / 8][K / 128][8][4] scales.
pub fn moe_pack(w: &Tensor, s: &Tensor) -> (Tensor, Tensor) {
    let (e, rows, kb) = (w.size()[0], w.size()[1], w.size()[2]);
    let steps = kb / 64;
    let wp = w.view([e * rows / 8, 8, steps, 4, 16]).permute([0, 2, 1, 3, 4]).contiguous();
    let sp = s.view([e * rows / 8, 8, steps, 4]).permute([0, 2, 1, 3]).contiguous();
    (wp.view([e, rows, kb]), sp.view([e, rows, kb / 16]))
}

/// The tile-packed expert weights (moe_pack) in the "mx" layout of shim/ds41_mx.cu: inside every tile's 128-wide K step
/// the lane-major 16-byte chunks become word-interleaved (lane q gets word q of each 32-block); scales are unchanged.
pub fn moe_pack_mx(wp: &Tensor) -> Tensor {
    let (e, rows, kb) = (wp.size()[0], wp.size()[1], wp.size()[2]);
    let steps = kb / 64;
    wp.view([e * rows / 8, steps, 8, 4, 4, 4]).permute([0, 1, 2, 4, 3, 5]).contiguous().view([e, rows, kb])
}
/// act_quant into E4M3 bytes + UE8M0 bytes per 32: x BF16 [R, K] -> (q Uint8 [R, K], s Uint8 [R, K / 32]).
pub fn act_quant_fp8(x: &Tensor) -> (Tensor, Tensor) {
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let (r, k) = (x.size()[0], x.size()[1]);
    let q = Tensor::empty([r, k], (Kind::Uint8, x.device()));
    let s = Tensor::empty([r, k / 32], (Kind::Uint8, x.device()));
    assert_eq!(unsafe { ds41_act_quant_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), x.numel() as i64, rs_current_stream()) }, 0, "ds41_act_quant_fp8");
    (q, s)
}
/// act_quant of rows stored as E4M3 + E8M0 per 32 (wq Uint8 [R, K], ws Uint8 [R, K / 32]): the bytes act_quant_fp8 gives
/// their BF16 dequantization (E4M3 x 2^(s - 127), ops::e8m0), without materializing it.
pub fn requant_fp8(wq: &Tensor, ws: &Tensor) -> (Tensor, Tensor) {
    let (wq, ws) = (wq.contiguous(), ws.contiguous());
    assert_eq!(wq.numel(), ws.numel() * 32);
    let q = Tensor::empty_like(&wq);
    let s = Tensor::empty_like(&ws);
    assert_eq!(unsafe { ds41_requant_fp8(wq.data_ptr(), ws.data_ptr(), q.data_ptr(), s.data_ptr(), wq.numel() as i64, rs_current_stream()) }, 0, "ds41_requant_fp8");
    (q, s)
}
/// FP8 GEMV for 1..16 rows on the block-scaled MMA: x BF16 [m, k] (act-quantized inside the kernel) . W^T -> FP32 [m, n].
pub fn fp8_gemv_mx(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor { gemv_mx(x, w, ws, false) }
/// Kernel attributes the MX GEMV needs, set before any graph capture (Model::load).
pub fn gemv_mx_init() { assert_eq!(unsafe { ds41_fp8_gemv_mx_init() }, 0, "ds41_fp8_gemv_mx_init"); }
/// `fp8_gemv_mx` with the sums rounded to BF16 in the kernel (the same bits as converting the FP32 result).
pub fn fp8_gemv_mx_bf16(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor { gemv_mx(x, w, ws, true) }
/// An mHC mix job for the next MX GEMVs (shim/ds41_hc_job.cuh; the C struct HcJob).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcJobC { kind: i32, ctas: i32, h: *const std::ffi::c_void, fn_: *const f32, part: *mut f32, t: i32, d: i32, scale: *const f32, base: *const f32,
                pre: *mut f32, post: *mut f32, comb: *mut f32, norm_eps: f32, eps: f32, iters: i32 }
extern "C" { fn ds41_fp8_gemv_mx_job(x: *const std::ffi::c_void, aq: *const std::ffi::c_void, as_: *const std::ffi::c_void, w: *const std::ffi::c_void, ws: *const std::ffi::c_void,
                                    y: *mut std::ffi::c_void, m: i32, n: i32, k: i32, ob16: i32, job: *const HcJobC, st: *mut std::ffi::c_void) -> i32; }
thread_local! {
    /// Jobs the next MX GEMVs take, in order (with the tensors they use, kept until the job is launched).
    static GEMV_JOBS: std::cell::RefCell<std::collections::VecDeque<(HcJobC, Vec<Tensor>)>> = std::cell::RefCell::new(Default::default());
}
/// `hc_mix` of a few rows (1..16, the 160-slice path) as side jobs of the next two MX GEMVs on this stream: the partial
/// sums in the first one's grid, the finalize in the second's (shim/ds41_hc_job.cuh; the same operations as hc_mix's
/// kernels, so the same bits). The caller makes sure those two GEMVs come next and that nothing reads the results
/// before the second. None (nothing queued) when the shape does not take that path.
pub fn hc_mix_jobs(h: &Tensor, f: &Tensor, scale: &Tensor, base: &Tensor, norm_eps: f64, eps: f64, iters: i64) -> Option<(Tensor, Tensor, Tensor)> {
    let (t, d) = (h.size()[0], h.size()[2]);
    if !(1..=16).contains(&t) || unsafe { ds41_hc_slices(t as i32, d as i32) } != 160 { return None; }
    let h = h.to_kind(Kind::BFloat16).contiguous();
    let o = (Kind::Float, h.device());
    let (pre, post, comb) = (Tensor::empty([t, 4], o), Tensor::empty([t, 4], o), Tensor::empty([t, 4, 4], o));
    let part = Tensor::empty([160 * t * 25], o);
    let j = HcJobC { kind: 1, ctas: 0, h: h.data_ptr(), fn_: f.data_ptr() as *const f32, part: part.data_ptr() as *mut f32, t: t as i32, d: d as i32,
                     scale: scale.data_ptr() as *const f32, base: base.data_ptr() as *const f32, pre: pre.data_ptr() as *mut f32,
                     post: post.data_ptr() as *mut f32, comb: comb.data_ptr() as *mut f32, norm_eps: norm_eps as f32, eps: eps as f32, iters: iters as i32 };
    GEMV_JOBS.with(|q| {
        let mut q = q.borrow_mut();
        q.push_back((j, vec![h.shallow_clone(), part.shallow_clone(), f.shallow_clone()]));
        q.push_back((HcJobC { kind: 2, ..j }, vec![h, part, scale.shallow_clone(), base.shallow_clone(), pre.shallow_clone(), post.shallow_clone(), comb.shallow_clone()]));
    });
    Some((pre, post, comb))
}
/// No mix job left queued (the GEMVs meant to take them ran).
pub fn hc_jobs_done() -> bool { GEMV_JOBS.with(|q| q.borrow().is_empty()) }
static GEMV_PREQ: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);
pub fn gemv_preq_rows() -> i64 {
    let v = GEMV_PREQ.load(std::sync::atomic::Ordering::Relaxed);
    if v >= 0 { return v; }
    let e = std::env::var("DS41_GEMV_PREQ_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    GEMV_PREQ.store(e, std::sync::atomic::Ordering::Relaxed);
    e
}
/// DS41_GEMV_PREQ_K1 (default on; 0 off): single rows with k >= 4096 also take the one-kernel quantization (qakv share,
/// shared w13: -2 us each; the k = 1280 GEMVs are faster quantizing in the CTA).
fn gemv_preq_k1() -> bool { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *E.get_or_init(|| std::env::var("DS41_GEMV_PREQ_K1").map_or(true, |v| v != "0")) }
pub fn set_gemv_preq_rows(v: i64) { GEMV_PREQ.store(v, std::sync::atomic::Ordering::Relaxed); }
fn gemv_mx(x: &Tensor, w: &Tensor, ws: &Tensor, ob16: bool) -> Tensor {
    let x = x.to_kind(Kind::BFloat16).contiguous();
    let (m, k) = (x.size()[0], x.size()[1]);
    let n = w.size()[0];
    let y = Tensor::empty([m, n], (if ob16 { Kind::BFloat16 } else { Kind::Float }, x.device()));
    // the input is act-quantized by every CTA in shared memory when it fits, else by a kernel first; from
    // DS41_GEMV_PREQ_ROWS rows (default 2; 99 off) always by one act_quant_fp8 kernel: the same bytes, so the same result,
    // and each CTA no longer quantizes all m x k inputs itself (wo_b at 8 rows 102 -> 69 us)
    let fits = (m * k + m * k / 32) as usize <= unsafe { ds41_fp8_gemv_mx_smem_max() } && m < gemv_preq_rows() && !(m == 1 && k >= 4096 && gemv_preq_k1());
    let q = (!fits).then(|| act_quant_fp8(&x));
    let (aq, asc) = q.as_ref().map_or((std::ptr::null_mut(), std::ptr::null_mut()), |(a, b)| (a.data_ptr(), b.data_ptr()));
    let job = GEMV_JOBS.with(|q| q.borrow_mut().pop_front());
    match &job {
        Some((j, _)) => assert_eq!(unsafe { ds41_fp8_gemv_mx_job(x.data_ptr(), aq, asc, w.data_ptr(), ws.data_ptr(), y.data_ptr(), m as i32, n as i32, k as i32, ob16 as i32, j, rs_current_stream()) }, 0, "ds41_fp8_gemv_mx_job"),
        None => assert_eq!(unsafe { ds41_fp8_gemv_mx_q(x.data_ptr(), aq, asc, w.data_ptr(), ws.data_ptr(), y.data_ptr(), m as i32, n as i32, k as i32, ob16 as i32, rs_current_stream()) }, 0, "ds41_fp8_gemv_mx"),
    }
    y
}

/// FP8 linear on the block-scaled tensor-core MMA (prefill): x BF16 [M, K] (act-quantized inside) . W^T, W E4M3 [N, K]
/// with E8M0 per 32x32 -> FP32 [M, N].
pub fn fp8_gemm_mx(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (xq, xs) = act_quant_fp8(x);
    fp8_gemm_mx_q(&xq, &xs, w, ws)
}
/// `fp8_gemm_mx` of an already quantized input (xq E4M3 bytes [M, K], xs UE8M0 bytes [M, K / 32], as act_quant_fp8).
pub fn fp8_gemm_mx_q(xq: &Tensor, xs: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (xq, xs) = (xq.contiguous(), xs.contiguous());
    let (m, k) = (xq.size()[0], xq.size()[1]);
    let n = w.size()[0];
    let y = Tensor::empty([m, n], (Kind::Float, xq.device()));
    assert_eq!(unsafe { ds41_fp8_gemm_mx(xq.data_ptr(), xs.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_fp8_gemm_mx");
    y
}

/// `fp8_gemm_mx` into `out` (FP32 [M, N], contiguous: e.g. a row range of a larger output).
pub fn fp8_gemm_mx_into(x: &Tensor, w: &Tensor, ws: &Tensor, out: &Tensor) {
    let (xq, xs) = act_quant_fp8(x);
    let (m, k) = (xq.size()[0], xq.size()[1]);
    let n = w.size()[0];
    assert!(out.is_contiguous() && out.kind() == Kind::Float && out.size() == [m, n], "fp8_gemm_mx_into");
    assert_eq!(unsafe { ds41_fp8_gemm_mx(xq.data_ptr(), xs.data_ptr(), w.data_ptr(), ws.data_ptr(), out.data_ptr() as *mut f32, m as i32, n as i32, k as i32, rs_current_stream()) }, 0, "ds41_fp8_gemm_mx");
}
/// `fp8_gemm_mx` with the output rounded to BF16 inside (the same bits as converting the FP32 result).
pub fn fp8_gemm_mx_bf16(x: &Tensor, w: &Tensor, ws: &Tensor) -> Tensor {
    let (xq, xs) = act_quant_fp8(x);
    let (m, k) = (xq.size()[0], xq.size()[1]);
    let n = w.size()[0];
    let y = Tensor::empty([m, n], (Kind::BFloat16, xq.device()));
    assert_eq!(unsafe { ds41_fp8_gemm_mx_ex(xq.data_ptr(), xs.data_ptr(), w.data_ptr(), ws.data_ptr(), y.data_ptr(), m as i32, n as i32, k as i32, 1, rs_current_stream()) }, 0, "ds41_fp8_gemm_mx");
    y
}

/// Routed experts on the block-scaled tensor-core MMA (shim/ds41_mx.cu): x BF16 [R, 5120] (quantized inside, like the
/// official act_quant), weights in the mx layout; otherwise the contract of `moe`.
pub fn moe_mx(x: &Tensor, idx: &Tensor, rw: &Tensor, w: [&Tensor; 6], add: Option<&Tensor>) -> Tensor {
    let (xq, xs) = act_quant_fp8(x);
    moe_mx_q(&xq, &xs, idx, rw, w, add)
}
/// `moe_mx` of the act_quant operands of x (E4M3 bytes [R, 5120], UE8M0 bytes [R, 160]: what moe_mx quantizes x into).
pub fn moe_mx_q(xq: &Tensor, xs: &Tensor, idx: &Tensor, rw: &Tensor, w: [&Tensor; 6], add: Option<&Tensor>) -> Tensor {
    let r = xq.size()[0];
    let (topk, e) = (idx.size()[1], w[0].size()[0]);
    let i = w[0].size()[1];
    let (xq, xs) = (xq.contiguous(), xs.contiguous());
    let idx = idx.to_kind(Kind::Int64).contiguous();
    let rw = rw.to_kind(Kind::Float).contiguous();
    let add = add.map(|a| a.to_kind(Kind::Float).contiguous());
    let y = Tensor::empty([r, 5120], (Kind::Float, xq.device()));
    let ws = Tensor::empty([unsafe { ds41_moe_mx_ws_bytes(r as i32, i as i32, topk as i32, e as i32) } as i64], (Kind::Uint8, xq.device()));
    let rc = unsafe {
        ds41_moe_mx(xq.data_ptr(), xs.data_ptr(), r as i32, idx.data_ptr() as *const i64, rw.data_ptr() as *const f32, w[0].data_ptr(), w[1].data_ptr(),
                    w[2].data_ptr(), w[3].data_ptr(), w[4].data_ptr(), w[5].data_ptr(), i as i32,
                    add.as_ref().map_or(std::ptr::null(), |a| a.data_ptr() as *const f32), y.data_ptr() as *mut f32, ws.data_ptr(),
                    topk as i32, e as i32, rs_current_stream())
    };
    assert_eq!(rc, 0, "ds41_moe_mx");
    y
}

/// `moe_mx_q` with the combine left for later, by row ranges (MoeRows::combine): None when the call's down / combine
/// run in column passes (then the caller takes moe_mx_q). The rows combined in parts are bitwise the whole call's.
pub struct MoeRows { ws: Tensor, pub y: Tensor, r: i64, i: i64, topk: i64, e: i64 }   // (y: [0] unless want_y)
pub fn moe_mx_q_rows(xq: &Tensor, xs: &Tensor, idx: &Tensor, rw: &Tensor, w: [&Tensor; 6], want_y: bool) -> Option<MoeRows> {
    let r = xq.size()[0];
    let (topk, e) = (idx.size()[1], w[0].size()[0]);
    let i = w[0].size()[1];
    if unsafe { ds41_moe_mx_col_passes(r as i32, topk as i32, e as i32) } != 1 { return None; }
    let (xq, xs) = (xq.contiguous(), xs.contiguous());
    let idx = idx.to_kind(Kind::Int64).contiguous();
    let rw = rw.to_kind(Kind::Float).contiguous();
    let y = Tensor::empty([if want_y { r } else { 0 }, 5120], (Kind::Float, xq.device()));
    let ws = Tensor::empty([unsafe { ds41_moe_mx_ws_bytes(r as i32, i as i32, topk as i32, e as i32) } as i64], (Kind::Uint8, xq.device()));
    // (skip_combine: y is not written)
    let rc = unsafe {
        ds41_moe_mx_ex(xq.data_ptr(), xs.data_ptr(), r as i32, idx.data_ptr() as *const i64, rw.data_ptr() as *const f32, w[0].data_ptr(), w[1].data_ptr(),
                       w[2].data_ptr(), w[3].data_ptr(), w[4].data_ptr(), w[5].data_ptr(), i as i32, std::ptr::null(), y.data_ptr() as *mut f32,
                       ws.data_ptr(), topk as i32, e as i32, 1, rs_current_stream())
    };
    assert_eq!(rc, 0, "ds41_moe_mx_ex (rows)");
    Some(MoeRows { ws, y, r, i, topk, e })
}
impl MoeRows {
    /// (out [P, 5120] FP32 of the down passes, pairmap [P], topk) inside the workspace
    pub fn ptrs(&self) -> (*const f32, *const i32, i64) {
        let (mut o, mut p) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(unsafe { ds41_moe_mx_ws_ptrs(self.ws.data_ptr(), self.r as i32, self.i as i32, self.topk as i32, self.e as i32, &mut o, &mut p) }, 0);
        (o as *const f32, p as *const i32, self.topk)
    }
    /// the workspace (kept by whoever reads it later)
    pub fn ws(&self) -> Tensor { self.ws.shallow_clone() }
    /// rows [r0, r0 + nr) of y = the routed sum + add (FP32 [R, 5120], the whole call's add)
    pub fn combine(&self, add: &Tensor, r0: i64, nr: i64) {
        assert!(self.y.size()[0] == self.r, "MoeRows::combine without want_y");
        assert!(add.kind() == Kind::Float && add.is_contiguous() && add.size() == [self.r, 5120], "MoeRows::combine add");
        assert_eq!(unsafe { ds41_moe_mx_combine_rows(self.ws.data_ptr(), self.r as i32, self.i as i32, self.topk as i32, self.e as i32, add.data_ptr() as *const f32,
                                                    self.y.data_ptr() as *mut f32, r0 as i32, nr as i32, rs_current_stream()) }, 0, "ds41_moe_mx_combine_rows");
    }
}
/// `moe_mx` with the shared expert `shared(x)` computed on a parallel stream while the router, grouping and routed
/// experts run (decode / verify: both are small and memory-bound, so each fills the other's kernel ramps and tails),
/// joined before the combine, which adds it exactly as `moe_mx(.., Some(shared))` does. `route(x)` -> (idx, rw).
pub fn moe_mx_par(x: &Tensor, route: impl FnOnce() -> (Tensor, Tensor), w: [&Tensor; 6], shared: impl FnOnce() -> Tensor) -> Tensor {
    assert_eq!(unsafe { rs_stream_fork(1) }, 0, "rs_stream_fork");
    assert_eq!(unsafe { rs_stream_set(0) }, 0, "rs_stream_set");
    let ys = shared().to_kind(Kind::Float).contiguous();
    assert_eq!(unsafe { rs_stream_set(-1) }, 0, "rs_stream_set");
    let (idx, rw) = route();
    let r = x.size()[0];
    let (topk, e) = (idx.size()[1], w[0].size()[0]);
    let i = w[0].size()[1];
    let (xq, xs) = act_quant_fp8(x);
    let idx = idx.to_kind(Kind::Int64).contiguous();
    let rw = rw.to_kind(Kind::Float).contiguous();
    let y = Tensor::empty([r, 5120], (Kind::Float, x.device()));
    let ws = Tensor::empty([unsafe { ds41_moe_mx_ws_bytes(r as i32, i as i32, topk as i32, e as i32) } as i64], (Kind::Uint8, x.device()));
    let rc = unsafe {
        ds41_moe_mx_ex(xq.data_ptr(), xs.data_ptr(), r as i32, idx.data_ptr() as *const i64, rw.data_ptr() as *const f32, w[0].data_ptr(), w[1].data_ptr(),
                       w[2].data_ptr(), w[3].data_ptr(), w[4].data_ptr(), w[5].data_ptr(), i as i32, std::ptr::null(), y.data_ptr() as *mut f32,
                       ws.data_ptr(), topk as i32, e as i32, 1, rs_current_stream())
    };
    assert_eq!(rc, 0, "ds41_moe_mx_ex");
    assert_eq!(unsafe { rs_stream_join(1) }, 0, "rs_stream_join");
    assert_eq!(unsafe { ds41_moe_mx_combine(ws.data_ptr(), r as i32, i as i32, topk as i32, e as i32, ys.data_ptr() as *const f32, y.data_ptr() as *mut f32, rs_current_stream()) }, 0, "ds41_moe_mx_combine");
    y
}

/// Routed experts of one layer (shim/ds41_moe.cu): xq [R, 5120] act-quantized BF16, idx [R, topk] Int64, rw [R, topk] FP32
/// route weights (incl. route_scale), stacked MXFP4 weights `w` = [w1, w1s, w3, w3s, w2, w2s] of this rank's intermediate
/// slice; `add` [R, 5120] FP32 is added (shared expert). Returns the FP32 partial [R, 5120].
pub fn moe(xq: &Tensor, idx: &Tensor, rw: &Tensor, w: [&Tensor; 6], add: Option<&Tensor>) -> Tensor {
    let r = xq.size()[0];
    let (topk, e) = (idx.size()[1], w[0].size()[0]);
    let i = w[0].size()[1];
    let xq = xq.to_kind(Kind::BFloat16).contiguous();
    let idx = idx.to_kind(Kind::Int64).contiguous();
    let rw = rw.to_kind(Kind::Float).contiguous();
    let add = add.map(|a| a.to_kind(Kind::Float).contiguous());
    let y = Tensor::empty([r, 5120], (Kind::Float, xq.device()));
    let ws = Tensor::empty([unsafe { ds41_moe_ws_bytes(r as i32, i as i32, topk as i32, e as i32) } as i64], (Kind::Uint8, xq.device()));
    let rc = unsafe {
        ds41_moe(xq.data_ptr(), r as i32, idx.data_ptr() as *const i64, rw.data_ptr() as *const f32, w[0].data_ptr(), w[1].data_ptr(),
                 w[2].data_ptr(), w[3].data_ptr(), w[4].data_ptr(), w[5].data_ptr(), i as i32,
                 add.as_ref().map_or(std::ptr::null(), |a| a.data_ptr() as *const f32), y.data_ptr() as *mut f32, ws.data_ptr(), topk as i32, e as i32, rs_current_stream())
    };
    assert_eq!(rc, 0, "ds41_moe");
    y
}

/// Reference for `moe`: per expert, dequantize (FP32) and multiply, same roundings (gate/up BF16, SwiGLU product BF16,
/// act_quant), FP32 outputs summed per row in slot order.
pub fn moe_ref(xq: &Tensor, idx: &Tensor, rw: &Tensor, w: [&Tensor; 6], table: &Tensor) -> Tensor {
    use super::load::dequant_mxfp4;
    let bf = |t: Tensor| t.to_kind(Kind::BFloat16).to_kind(Kind::Float);
    let (r, k) = (idx.size()[0], idx.size()[1]);
    let iv: Vec<i64> = Vec::<i64>::try_from(idx.to_device(tch::Device::Cpu).view(-1)).unwrap();
    let mut outs = Vec::new();
    for (p, &e) in iv.iter().enumerate() {
        let x = xq.get(p as i64 / k).unsqueeze(0).to_kind(Kind::Float);
        let g = bf(x.matmul(&dequant_mxfp4(&w[0].get(e), &w[1].get(e), table, Kind::Float).tr())).clamp_max(10.0);
        let u = bf(x.matmul(&dequant_mxfp4(&w[2].get(e), &w[3].get(e), table, Kind::Float).tr())).clamp(-10.0, 10.0);
        let hm = act_quant(&(g.silu() * u * rw.view(-1).get(p as i64)).to_kind(Kind::BFloat16)).to_kind(Kind::Float);
        outs.push(hm.matmul(&dequant_mxfp4(&w[4].get(e), &w[5].get(e), table, Kind::Float).tr()));
    }
    let o = Tensor::cat(&outs, 0).view([r, k, -1]);
    let mut y = o.select(1, 0);
    for j in 1..k { y = y + o.select(1, j); }
    y
}

/// `ds41-kernel-test <ckpt>`: the decode kernels on real weights against ops.rs (dequantize + FP32 matmul), m = 1..8:
/// act_quant bitwise; GEMVs within FP32 summation-order differences (max relative error reported); and the time per call.
/// `ds41-gemv-bench`: the decode GEMVs at this engine's per-rank TP3 shapes (random bytes: the speed does not depend on
/// the values), time and bandwidth per call, so the kernels can be tuned shape by shape.
/// `ds41-attn-bench`: the prefill sparse attention on the TP3 rank shape of a 4k CED encoder part (24 local heads, 2048
/// query rows at positions 2048.., window 128 over ring + chunk, 512 compressed indices of the 1024 the positions see,
/// sorted, -1 past them) and the decoder replay's 128 rows; random BF16 values. Time per call and, for each value of
/// DS41_ATTN_V in DS41_ATTN_VERSIONS (default "0"), whether the outputs are bitwise equal to the first.
pub fn attn_bench() {
    let _g = tch::no_grad_guard();
    let dev = tch::Device::Cuda(0);
    extern "C" { fn ds41_attn_version(v: i32) -> i32; }
    let versions: Vec<i32> = std::env::var("DS41_ATTN_VERSIONS").map_or(vec![0], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    let (h, w) = (24i64, 128i64);
    for (t, start) in [(2048i64, 2048i64), (128, 3968)] {
        let q = (Tensor::randn([t, h, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let ring = Tensor::randn([w, 512], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let kv = Tensor::randn([t, 512], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let clen = (start + t) / 4;
        let ckv = Tensor::randn([clen, 512], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let pos = Tensor::arange_start(start, start + t, (Kind::Int64, dev));
        // per row: the compressed entries its position sees (p / 4), 512 of them at random in ascending order, -1 past
        let mut iv = vec![-1i64; (t * 512) as usize];
        let mut rng = 0x9e3779b97f4a7c15u64;
        for r in 0..t {
            let n = ((start + r) / 4).min(clen);
            let mut sel: Vec<i64> = if n <= 512 { (0..n).collect() } else {
                let mut all: Vec<i64> = (0..n).collect();
                for i in 0..512usize { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; let j = i + (rng % (n as u64 - i as u64)) as usize; all.swap(i, j); }
                all.truncate(512); all };
            sel.sort();
            for (j, &c) in sel.iter().enumerate() { iv[(r * 512) as usize + j] = w + t + c; }
        }
        let idx = Tensor::from_slice(&iv).to_device(dev).view([t, 512]);
        let sink = Tensor::zeros([h], (Kind::Float, dev));
        let mut outs = Vec::new();
        let mut line = String::new();
        for &v in &versions {
            assert_eq!(unsafe { ds41_attn_version(v) }, 0);
            let f = || sparse_attn_into(&q, [Some(&ring), Some(&kv), Some(&ckv)], Some((w, start, &pos)), 0, &idx, &sink, 1.0 / (512f64).sqrt(), None).expect("sparse_attn");
            outs.push(f());
            for _ in 0..3 { let _ = f(); }
            tch::Cuda::synchronize(0);
            let t0 = std::time::Instant::now();
            for _ in 0..20 { let _ = f(); }
            tch::Cuda::synchronize(0);
            line += &format!("  v{v} {:8.1} us", t0.elapsed().as_secs_f64() * 1e6 / 20.0);
        }
        let same = outs.windows(2).all(|p| p[0].view_dtype(Kind::Int16).equal(&p[1].view_dtype(Kind::Int16)));
        eprintln!("[ds41-attn-bench] rows {t} at {start}:{line}  {}", if versions.len() > 1 { if same { "bitwise equal" } else { "DIFFERENT" } } else { "" });
    }
    unsafe { ds41_attn_version(0) };
}

/// `ds41-gemm-bench`: prefill FP8 GEMM (fp8_gemm_mx, BF16 out) on the TP3 rank shapes of a 4k CED prefill: time
/// per call (weights rotated past L2) of each kernel version in DS41_GEMM_VERSIONS (default 2,3: the previous and the
/// default) and whether their outputs are bitwise equal.
pub fn gemm_bench() {
    let _g = tch::no_grad_guard();
    let dev = tch::Device::Cuda(0);
    extern "C" { fn ds41_gemm_version(v: i32) -> i32; }
    let time = |f: &dyn Fn() -> Tensor| { for _ in 0..5 { let _ = f(); } tch::Cuda::synchronize(0);
        let t = std::time::Instant::now(); for _ in 0..40 { let _ = f(); } tch::Cuda::synchronize(0); t.elapsed().as_secs_f64() * 1e6 / 40.0 };
    // (name, m, n, k, calls per 4096-token CED prefill on a rank: DS41_GEMM_SHAPES=1 of an offline 4k round; m = 683 a
    // part's local rows, 2048 a part's gathered rows, 4096 both parts' rows, 128 the decoder replay)
    let shapes: [(&str, i64, i64, i64, f64); 13] = [
        ("wo_b", 2048, 5120, 3072, 40.0), ("wq_b", 2048, 12288, 1280, 40.0), ("shared w13", 4096, 1536, 5120, 20.0),
        ("shared w2", 4096, 5120, 768, 20.0), ("qakv local", 683, 1792, 5120, 34.0), ("engram full", 683, 25600, 6144, 4.0),
        ("x qakv", 2048, 1792, 5120, 6.0), ("idx 682", 682, 4096, 1280, 6.0), ("r shared w13", 128, 1536, 5120, 20.0),
        ("r qakv", 128, 1792, 5120, 20.0), ("r shared w2", 128, 5120, 768, 20.0), ("r wo_b", 128, 5120, 3072, 20.0),
        ("r wq_b", 128, 12288, 1280, 20.0)];
    let versions: Vec<i32> = std::env::var("DS41_GEMM_VERSIONS").map_or(vec![2, 3], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    let mut tot = vec![0.0f64; versions.len()];
    for (name, m, n, k, calls) in shapes {
        let copies = ((96 << 20) / (n * k) + 1) as usize;
        let ws: Vec<Tensor> = (0..copies).map(|_| Tensor::randint(120, [n, k], (Kind::Uint8, dev))).collect();
        let s = Tensor::randint_low(110, 130, [(n + 31) / 32, k / 32], (Kind::Uint8, dev));
        {
            let x = Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let it = std::cell::Cell::new(0usize);
            let w = || { it.set(it.get() + 1); &ws[it.get() % copies] };
            let mut line = String::new();
            let mut outs = Vec::new();
            for (vi, &v) in versions.iter().enumerate() {
                assert_eq!(unsafe { ds41_gemm_version(v) }, 0, "ds41_gemm_version {v}");
                outs.push(fp8_gemm_mx_bf16(&x, &ws[0], &s));
                let aq = act_quant_fp8(&x);
                let us = time(&|| { let (q, sc) = (&aq.0, &aq.1); let y = Tensor::empty([m, n], (Kind::BFloat16, dev));
                    let wt = w();
                    assert_eq!(unsafe { ds41_fp8_gemm_mx_ex(q.data_ptr(), sc.data_ptr(), wt.data_ptr(), s.data_ptr(), y.data_ptr(), m as i32, n as i32, k as i32, 1, rs_current_stream()) }, 0); y });
                tot[vi] += us * calls;
                line += &format!("  v{v} {us:7.1} us {:5.1} TF", (2 * m * n * k) as f64 / us / 1e6);
            }
            let same = outs.windows(2).all(|p| p[0].view_dtype(Kind::Int16).equal(&p[1].view_dtype(Kind::Int16)));
            eprintln!("[ds41-gemm-bench] {name:12} m={m:4} [{n:5}x{k:5}] x{calls:2}:{line}  {}", if versions.len() > 1 { if same { "bitwise equal" } else { "DIFFERENT" } } else { "" });
        }
    }
    let t: Vec<String> = versions.iter().zip(&tot).map(|(v, t)| format!("v{v} {:.1} ms", t / 1e3)).collect();
    eprintln!("[ds41-gemm-bench] per 4k prefill (calls weighted): {}", t.join(", "));
    unsafe { ds41_gemm_version(0) };
    // the router's FP32 matmul (cuBLAS): are a row's logits the same whatever the row count (sequence-parallel routing)?
    {
        let gate = Tensor::randn([384, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let x = Tensor::randn([4100, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let full = x.to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr());
        for (r0, m) in [(0i64, 683i64), (683, 683), (1366, 682), (0, 1366), (1366, 1366), (2732, 1366), (0, 2048), (2048, 2048), (100, 171), (0, 65), (5, 300), (0, 4096)] {
            let part = x.narrow(0, r0, m).to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr());
            let d = (&part - full.narrow(0, r0, m)).abs().max().double_value(&[]);
            eprintln!("[ds41-gemm-bench] router rows [{r0}, +{m}) vs 4100-row call: max |diff| {d:e}");
        }
        for t in [4096i64, 2048, 8192] {
            let xf = Tensor::randn([t, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let f = xf.to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr());
            let w = 3i64; let l = (t + w - 1) / w;
            let mut worst = 0f64;
            for r in 0..w { let n = l.min(t - r * l); let p = xf.narrow(0, r * l, n).to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr()); worst = worst.max((&p - f.narrow(0, r * l, n)).abs().max().double_value(&[])); }
            for (a, n) in [(0, t / 2), (t / 2, t - t / 2)] { let l2 = (n + w - 1) / w; for r in 0..w { let k = l2.min(n - r * l2); let p = xf.narrow(0, a + r * l2, k).to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr()); worst = worst.max((&p - f.narrow(0, a + r * l2, k)).abs().max().double_value(&[])); } }
            eprintln!("[ds41-gemm-bench] router T={t}: SP shares vs whole chunk max |diff| {worst:e}");
            // the same call shape (T rows) with every row but a rank's share zeroed: are the share's logits the whole
            // chunk's bits? (then a rank can route its own rows before the FFN gather)
            let mut worst_z = 0f64;
            for (a, n) in [(0, t / 2), (t / 2, t - t / 2)] { let l2 = (n + w - 1) / w; for r in 0..w { let k = l2.min(n - r * l2);
                let z = Tensor::zeros([t, 5120], (Kind::BFloat16, dev)); let _ = z.narrow(0, a + r * l2, k).copy_(&xf.narrow(0, a + r * l2, k));
                let p = z.to_kind(Kind::Float).matmul(&gate.to_kind(Kind::Float).tr());
                worst_z = worst_z.max((p.narrow(0, a + r * l2, k) - f.narrow(0, a + r * l2, k)).abs().max().double_value(&[])); } }
            eprintln!("[ds41-gemm-bench] router T={t}: zero-padded same-shape shares vs whole chunk max |diff| {worst_z:e}");
        }
        // wo_a (prefill): one grouped BF16-output fp8_mm against per-group contiguous copies + fp8_mm + bf16 + cat
        for (m, gl) in [(2048i64, 3i64), (683, 3), (4096, 3), (100, 3), (1366, 2)] {
            let (n, k) = (1024i64, 4096i64);
            let w = Tensor::randint(120, [gl * n, k], (Kind::Uint8, dev));
            let s = Tensor::randint_low(110, 130, [gl * n / 32, k / 32], (Kind::Uint8, dev));
            let x = act_quant(&Tensor::randn([m, gl * k], (Kind::Float, dev)).to_kind(Kind::BFloat16));
            let per = || { let o3 = x.view([m, gl, k]);
                let p: Vec<Tensor> = (0..gl).map(|g| fp8_mm(&o3.select(1, g).contiguous(), &w.narrow(0, g * n, n), &s.narrow(0, g * n / 32, n / 32)).to_kind(Kind::BFloat16)).collect();
                Tensor::cat(&p, 1) };
            let a = per();
            let b = fp8_mm_grouped_bf16(&x, &w, &s, n, gl);
            let same = a.view_dtype(Kind::Int16).equal(&b.view_dtype(Kind::Int16));
            let (u0, u1) = (time(&per), time(&|| fp8_mm_grouped_bf16(&x, &w, &s, n, gl)));
            eprintln!("[ds41-gemm-bench] wo_a m={m} groups {gl}: {}; per group + cat {u0:.0} us, grouped {u1:.0} us", if same { "bitwise equal" } else { "DIFFERENT" });
        }
        // the row-subset GEMM with the whole call's bits (lt.cpp rs_lt_rows_*): found configuration, bits and time of a
        // rank's share against the zero-padded full call
        let gf = gate.to_kind(Kind::Float).contiguous();
        for t in [4096i64, 2048, 8192, 4083, 1366] {
            let t0 = std::time::Instant::now();
            let rc = unsafe { rs_lt_rows_find(gf.data_ptr() as *const f32, t as i32, 384, 5120, 1) };
            let find_ms = t0.elapsed().as_secs_f64() * 1e3;
            if rc != 0 { eprintln!("[ds41-gemm-bench] router rows T={t}: no configuration ({find_ms:.0} ms)"); continue; }
            let xf = Tensor::randn([t, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16).to_kind(Kind::Float);
            let f = xf.matmul(&gf.tr());
            let l = (t + 2) / 3;
            let mut same = true;
            for (o, m) in [(0, l), (l, l), (2 * l, t - 2 * l), (5, 1), (t / 2, t / 2)] {
                let y = Tensor::empty([m, 384], (Kind::Float, dev));
                assert_eq!(unsafe { rs_lt_rows_mm((xf.data_ptr() as *const f32).add((o * 5120) as usize), gf.data_ptr() as *const f32, y.data_ptr() as *mut f32, t as i32, m as i32, 384, 5120) }, 0);
                same &= y.equal(&f.narrow(0, o, m));
            }
            let us_full = time(&|| xf.matmul(&gf.tr()));
            let xs = xf.narrow(0, 0, l).contiguous();
            let us_rows = time(&|| { let y = Tensor::empty([l, 384], (Kind::Float, dev));
                assert_eq!(unsafe { rs_lt_rows_mm(xs.data_ptr() as *const f32, gf.data_ptr() as *const f32, y.data_ptr() as *mut f32, t as i32, l as i32, 384, 5120) }, 0); y });
            eprintln!("[ds41-gemm-bench] router rows T={t}: found in {find_ms:.0} ms; shares {}; full call {us_full:.0} us, a rank's {l} rows {us_rows:.0} us",
                      if same { "bitwise equal" } else { "DIFFERENT" });
        }
    }
    // routed experts (moe_mx) on the TP3 rank shapes (384 experts, I = 768, top-6 uniform random routing): time per call
    // against the weight-read floor, and the MMA rate
    let (e, i, topk) = (384i64, 768i64, 6i64);
    let w1 = Tensor::randint(256, [e, i, 2560], (Kind::Uint8, dev));
    let w3 = Tensor::randint(256, [e, i, 2560], (Kind::Uint8, dev));
    let w2 = Tensor::randint(256, [e, 5120, 384], (Kind::Uint8, dev));
    let (s1, s3, s2) = (Tensor::randint_low(118, 124, [e, i, 160], (Kind::Uint8, dev)), Tensor::randint_low(118, 124, [e, i, 160], (Kind::Uint8, dev)),
                        Tensor::randint_low(118, 124, [e, 5120, 24], (Kind::Uint8, dev)));
    let (w1, w3, w2) = (moe_pack_mx(&w1), moe_pack_mx(&w3), moe_pack_mx(&w2));
    let bytes = (w1.numel() + w3.numel() + w2.numel() + s1.numel() + s3.numel() + s2.numel()) as f64;
    for r in [128i64, 2048, 4096, 8192] {
        let x = (Tensor::randn([r, 5120], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let idx = Tensor::rand([r, e], (Kind::Float, dev)).argsort(-1, false).narrow(1, 0, topk).contiguous();
        let rw = Tensor::rand([r, topk], (Kind::Float, dev));
        let us = time(&|| moe_mx(&x, &idx, &rw, [&w1, &s1, &w3, &s3, &w2, &s2], None));
        let fl = 2.0 * (r * topk) as f64 * (3 * i * 5120) as f64;
        eprintln!("[ds41-gemm-bench] moe_mx R={r:5}: {us:8.1} us, {:.1} TFLOPS, weight floor {:.0} us at 273 GB/s", fl / us / 1e6, bytes / 273e3);
    }
}

pub fn gemv_bench() {
    let _g = tch::no_grad_guard();
    let dev = tch::Device::Cuda(0);
    gemv_mx_init();
    let time = |f: &dyn Fn() -> Tensor| { for _ in 0..5 { let _ = f(); } tch::Cuda::synchronize(0);
        let t = std::time::Instant::now(); for _ in 0..50 { let _ = f(); } tch::Cuda::synchronize(0); t.elapsed().as_secs_f64() * 1e6 / 50.0 };
    let shapes: [(&str, i64, i64, i64); 10] = [("wq_a", 1280, 5120, 40), ("wq_b", 12288, 1280, 40), ("wkv", 512, 5120, 40), ("wo_a(group)", 1024, 4096, 120),
        ("wo_b", 5120, 3072, 40), ("shared w1/w3", 768, 5120, 80), ("shared w2", 5120, 768, 40), ("indexer wq_b", 4096, 1280, 8),
        ("engram wkv", 25600, 6144, 2), ("mtp main_proj", 5120, 15360, 0)];
    let mut total = 0.0;
    for (name, n, k, per_token) in shapes {
        // enough copies of the weight (>= 256 MB) that every call reads it from memory, not L2 (as in decode)
        let copies = ((256 << 20) / (n * k) + 1) as usize;
        let ws: Vec<Tensor> = (0..copies).map(|_| Tensor::randint(120, [n, k], (Kind::Uint8, dev))).collect();
        let s = Tensor::full([(n + 31) / 32, k / 32], 120, (Kind::Uint8, dev));
        let x = Tensor::randn([1, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let it = std::cell::Cell::new(0usize);
        let w = || { it.set(it.get() + 1); &ws[it.get() % copies] };
        let us = time(&|| fp8_gemv(&x, w(), &s));
        total += us * per_token as f64;
        let mut by_m = String::new();
        for m in [1i64, 2, 3, 4, 6] {
            let xm = Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            if m > 1 { by_m += &format!(" m{m} {:.1}", time(&|| fp8_gemv(&xm, w(), &s))); }
            by_m += &format!(" mx{m} {:.1}", time(&|| fp8_gemv_mx(&xm, w(), &s)));
        }
        eprintln!("[ds41-gemv-bench] fp8 {name:14} [{n:5}x{k:5}] {:6.1} MB: {us:7.1} us, {:5.0} GB/s, x{per_token} per token;{by_m} us", (n * k) as f64 / 1e6, (n * k) as f64 / us / 1e3);
    }
    eprintln!("[ds41-gemv-bench] fp8 total per token {:.2} ms", total / 1e3);
    {
        // index_score: the tiled kernel (T >= 8) against the per-row one (T = 1 per call), bit for bit; prefill timing
        let (t, tc, ratio) = (64i64, 5000i64, 2i64);
        let q = (Tensor::randn([t, 32, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let k = (Tensor::randn([tc, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let w = Tensor::randn([t, 32], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let pos = Tensor::arange_start(tc * ratio - t, tc * ratio, (Kind::Int64, dev));
        let tiled = index_score(&q, &k, &w, &pos, ratio);
        let rows: Vec<Tensor> = (0..t).map(|i| index_score(&q.narrow(0, i, 1), &k, &w.narrow(0, i, 1), &pos.narrow(0, i, 1), ratio)).collect();
        let same = tiled.view_dtype(Kind::Int16).equal(&Tensor::cat(&rows, 0).view_dtype(Kind::Int16));
        let (tb, tcb) = (4096i64, 320_000i64);
        let qb = (Tensor::randn([tb, 32, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let kbig = (Tensor::randn([tcb, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let wb = Tensor::randn([tb, 32], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let pb = Tensor::arange_start(tcb * ratio - tb, tcb * ratio, (Kind::Int64, dev));
        let tm = |f: &dyn Fn() -> Tensor| { let _ = f(); tch::Cuda::synchronize(0); let s = std::time::Instant::now(); for _ in 0..3 { let _ = f(); } tch::Cuda::synchronize(0); s.elapsed().as_secs_f64() * 1e3 / 3.0 };
        let ms = tm(&|| index_score(&qb, &kbig, &wb, &pb, ratio));
        let ms1 = tm(&|| { let p: Vec<Tensor> = (0..tb).step_by(512).map(|r| index_score(&qb.narrow(0, r, 1), &kbig, &wb.narrow(0, r, 1), &pb.narrow(0, r, 1), ratio)).collect(); p[0].shallow_clone() }) * 512.0;
        eprintln!("[ds41-gemv-bench] index_score tiled vs per-row [{t} x {tc}]: {}; [{tb} x {tcb}] tiled {ms:.1} ms ({:.1} TFLOPS), per-row kernel ~{ms1:.0} ms (extrapolated)",
            if same { "bitwise equal" } else { "DIFFERENT" }, 2.0 * (tb * tcb * 32 * 128) as f64 / ms / 1e9);
    }
    for (tc, coarse) in [(4096i64, false), (16000, false), (16000, true)] {
        // topk_idx against libtorch topk + sort: the selected scores as a multiset (ties may pick other indices),
        // ascending order, and -1 past (pos + 1) / ratio; coarse: few distinct values (many ties)
        let (t, k, ratio, offset) = (4i64, 512i64, 4i64, 200i64);
        let mut sc = Tensor::randn([t, tc], (Kind::Float, dev));
        if coarse { sc = (sc * 4.0).round(); }
        let sc = sc.to_kind(Kind::BFloat16);
        let pos = Tensor::from_slice(&[tc * ratio - 1, tc * ratio - 9, 3000, 5]).to_device(dev);
        let lens = (&pos + 1).floor_divide_scalar(ratio);
        let cpos = Tensor::arange(tc, (Kind::Int64, dev));
        let sc = sc.masked_fill(&cpos.unsqueeze(0).ge_tensor(&lens.unsqueeze(-1)), f64::NEG_INFINITY);
        let ours = topk_idx(&sc, &pos, ratio, k, offset);
        let (vals, _) = sc.topk(k, -1, true, true);
        let raw = &ours - offset;
        let valid = ours.ge(0);
        let picked = sc.gather(1, &raw.clamp_min(0), false).masked_fill(&valid.logical_not(), f64::NEG_INFINITY);
        let (picked, _) = picked.sort(-1, true);
        let same_vals = picked.to_kind(Kind::Float).eq_tensor(&vals.to_kind(Kind::Float)).all().int64_value(&[]) == 1;
        let d = raw.narrow(1, 1, k - 1) - raw.narrow(1, 0, k - 1);
        let ascending = d.masked_select(&valid.narrow(1, 1, k - 1).logical_and(&valid.narrow(1, 0, k - 1))).gt(0).all().int64_value(&[]) == 1;
        let masked_ok = ours.ge(0).logical_and(&raw.ge_tensor(&lens.unsqueeze(-1))).any().int64_value(&[]) == 0;
        let us_ours = time(&|| topk_idx(&sc, &pos, ratio, k, offset));
        let us_torch = time(&|| { let (_, i) = sc.topk(k, -1, true, false); let (i, _) = i.sort(-1, false); (&i + offset).masked_fill(&i.ge_tensor(&lens.unsqueeze(-1)), -1) });
        eprintln!("[ds41-gemv-bench] topk_idx [{t}x{tc}] k {k}{}: values {} ascending {} masked {}; {us_ours:.1} us (libtorch {us_torch:.1} us)",
            if coarse { " (many ties)" } else { "" }, if same_vals { "ok" } else { "MISMATCH" }, if ascending { "ok" } else { "NO" }, if masked_ok { "ok" } else { "NO" });
    }
    {
        // wo_a of one rank in one launch: 3 groups of 1024 rows, each reading its 4096 input elements; the CUDA-core
        // GEMV (ds41_fp8_gemv_ex) against the BF16-MMA one (ds41_fp8_gemv_bx)
        let (n, k, rpg) = (3072i64, 4096i64, 1024i64);
        let copies = ((256 << 20) / (n * k) + 1) as usize;
        let ws: Vec<Tensor> = (0..copies).map(|_| Tensor::randint(120, [n, k], (Kind::Uint8, dev))).collect();
        let s = Tensor::randint_low(110, 130, [n / 32, k / 32], (Kind::Uint8, dev));
        let it = std::cell::Cell::new(0usize);
        let w = || { it.set(it.get() + 1); &ws[it.get() % copies] };
        let run = |bx: bool, x: &Tensor, w: &Tensor| {
            let m = x.size()[0];
            let y = Tensor::empty([m, n], (Kind::Float, dev));
            let f = if bx { ds41_fp8_gemv_bx } else { ds41_fp8_gemv_ex };
            assert_eq!(unsafe { f(x.data_ptr(), w.data_ptr(), s.data_ptr(), y.data_ptr() as *mut f32, m as i32, n as i32, k as i32, (3 * k) as i32, rpg as i32, rs_current_stream()) }, 0);
            y
        };
        let mut line = String::new();
        for m in [1i64, 2, 4, 8] {
            let x = Tensor::randn([m, 3 * k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let (a, b) = (run(false, &x, &ws[0]), run(true, &x, &ws[0]));
            let rel = f64::try_from((&a - &b).abs().max()).unwrap() / f64::try_from(a.abs().max()).unwrap();
            line += &format!(" m{m}: {:.1} -> {:.1} us (rel diff {rel:.1e});", time(&|| run(false, &x, w())), time(&|| run(true, &x, w())));
        }
        eprintln!("[ds41-gemv-bench] wo_a grouped [{n}x{k}]{line}");
    }
    for (name, n, k) in [("head", 43136i64, 5120i64), ("router", 384, 5120)] {
        let w = Tensor::randn([n, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let x = Tensor::randn([1, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let us = time(&|| bf16_gemv(&x, &w));
        eprintln!("[ds41-gemv-bench] bf16 {name:14} [{n:5}x{k:5}] {us:7.1} us, {:5.0} GB/s", (n * k * 2) as f64 / us / 1e3);
    }
    // routed experts at R = 1 (6 distinct experts of 384, packed layout, intermediate slice 768)
    let ws = [Tensor::randint(255, [384, 768, 2560], (Kind::Uint8, dev)), Tensor::full([384, 768, 160], 120, (Kind::Uint8, dev)),
              Tensor::randint(255, [384, 768, 2560], (Kind::Uint8, dev)), Tensor::full([384, 768, 160], 120, (Kind::Uint8, dev)),
              Tensor::randint(255, [384, 5120, 384], (Kind::Uint8, dev)), Tensor::full([384, 5120, 24], 120, (Kind::Uint8, dev))];
    let w = [&ws[0], &ws[1], &ws[2], &ws[3], &ws[4], &ws[5]];
    let xq = act_quant(&Tensor::randn([1, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16));
    let idx = Tensor::from_slice(&[3i64, 77, 150, 201, 299, 380]).view([1, 6]).to_device(dev);
    let rw = Tensor::full([1, 6], 0.2, (Kind::Float, dev));
    let wx: Vec<Tensor> = ws.iter().enumerate().map(|(i, t)| if i % 2 == 0 { moe_pack_mx(t) } else { t.shallow_clone() }).collect();
    let wxr = [&wx[0], &wx[1], &wx[2], &wx[3], &wx[4], &wx[5]];
    let x1 = Tensor::randn([1, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
    // fresh experts every call (as in decode), and DS41_BENCH_LAYERS copies of the layer's experts in turn (the
    // model's ~40 layers spread the expert reads over ~90 GB: TLB reach)
    let layers = std::env::var("DS41_BENCH_LAYERS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let lw: Vec<Vec<Tensor>> = (0..layers).map(|l| if l == 0 { wx.iter().map(|t| t.shallow_clone()).collect() } else { wx.iter().map(|t| t.copy()).collect() }).collect();
    for r in [1i64, 2, 3, 4, 6] {
        let xr = Tensor::randn([r, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let ids: Vec<Tensor> = (0..64).map(|_| Tensor::rand([r, 384], (Kind::Float, dev)).topk(6, -1, true, true).1).collect();
        let rwr = Tensor::full([r, 6], 0.2, (Kind::Float, dev));
        let it = std::cell::Cell::new(0usize);
        let us = time(&|| { let i = it.get(); it.set(i + 1); let l = &lw[i % layers]; moe_mx(&xr, &ids[i % 64], &rwr, [&l[0], &l[1], &l[2], &l[3], &l[4], &l[5]], None) });
        let bytes = (6 * r).min(384) as f64 * 5.9e6;
        eprintln!("[ds41-gemv-bench] moe_mx R={r} ({layers} layer copies): {us:7.1} us ({:.0} GB/s of expert bytes, if distinct)", bytes / us / 1e3);
    }
    let _ = (&x1, &xq, &idx, &rw, w);
}

/// ds41-kernel-test: every decode MX GEMV shape of a TP3 rank, cold, per row count with the in-CTA quantization and with
/// one act_quant_fp8 kernel first (DS41_GEMV_PREQ_ROWS), and whether the two agree bitwise.
fn mx_table(dev: tch::Device) {
    let time = |f: &dyn Fn() -> Tensor| { for _ in 0..3 { let _ = f(); } tch::Cuda::synchronize(0);
        let t = std::time::Instant::now(); for _ in 0..20 { let _ = f(); } tch::Cuda::synchronize(0); t.elapsed().as_secs_f64() * 1e6 / 20.0 };
    // every decode MX GEMV shape of a TP3 rank, cold (copies rotating over >= 96 MB), GB/s of weight + scale bytes
    for (name, n, k) in [("qakv", 1792i64, 5120i64), ("qakv/3", 608, 5120), ("wq_b 24h", 12288, 1280), ("wq_b 16h", 8192, 1280), ("wo_b", 5120, 3072),
                         ("wo_b 2g", 5120, 2048), ("sh w13", 1536, 5120), ("sh w2", 5120, 768), ("idx wq_b", 4096, 1280)] {
        let q = (Tensor::randn([n, k], (Kind::Float, dev)) * 0.05).to_kind(Kind::Float8e4m3fn).view_dtype(Kind::Uint8);
        let s = Tensor::randint_low(118, 126, [n / 32, k / 32], (Kind::Uint8, dev));
        let bytes = n * k + n * k / 1024;
        let copies = ((96 << 20) / bytes + 1) as usize;
        let set: Vec<(Tensor, Tensor)> = (0..copies).map(|_| (q.copy(), s.copy())).collect();
        let mut line = format!("[ds41-kernel-test] MX GEMV {name} [{n}x{k}] ({copies} copies):");
        for m in [1i64, 2, 4, 8, 16] {
            let x = Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let i = std::cell::Cell::new(0usize);
            set_gemv_preq_rows(99);
            let us = time(&|| { let j = i.get(); i.set(j + 1); fp8_gemv_mx_bf16(&x, &set[j % copies].0, &set[j % copies].1) });
            let a = fp8_gemv_mx(&x, &set[0].0, &set[0].1);
            set_gemv_preq_rows(1);
            let up = time(&|| { let j = i.get(); i.set(j + 1); fp8_gemv_mx_bf16(&x, &set[j % copies].0, &set[j % copies].1) });
            let same = fp8_gemv_mx(&x, &set[0].0, &set[0].1).equal(&a);
            line += &format!(" m={m} {us:.1} / pre-q {up:.1} us{};", if same { "" } else { " DIFFERENT" });
        }
        set_gemv_preq_rows(-1);
        eprintln!("{line}");
    }
}

pub fn test(ckpt: &std::path::Path) {
    use super::load::{dequant_fp8, dequant_mxfp4, fp4_table, Checkpoint};
    let _g = tch::no_grad_guard();
    let dev = tch::Device::Cuda(0);
    let ck = Checkpoint::open(ckpt, dev);
    gemv_mx_init();
    tch::manual_seed(7);
    // the clamped sublayer norm variants against the original row kernel (DS41_KT_ONLY=clp: only this)
    {
        extern "C" { fn ds41_rms_clp_selftest(rows: i32, n: i32, k: i32, minc: i32, out: *mut i64) -> i32; }
        for (rows, k, minc) in [(1, 3, 1), (2, 3, 1), (3, 3, 2), (4, 3, 2), (8, 3, 1), (16, 4, 2), (5, 2, 1)] {
            let mut o = [0i64; 5];
            assert_eq!(unsafe { ds41_rms_clp_selftest(rows, 5120, k, minc, o.as_mut_ptr()) }, 0);
            eprintln!("[ds41-kernel-test] clamp norm rows={rows} k={k} min={minc}: mismatches clp1 {} clp2 {} cl2 {} counts {} (triggers {})", o[0], o[1], o[2], o[3], o[4]);
        }
        if std::env::var("DS41_KT_ONLY").as_deref() == Ok("clp") { return; }
        if std::env::var("DS41_KT_ONLY").as_deref() == Ok("mx") { mx_table(dev); return; }
    }
    // act_quant
    let x = Tensor::randn([8, 5120], (Kind::Float, dev)) * 3.0;
    let a = act_quant(&x.to_kind(Kind::BFloat16));
    let b = super::ops::act_quant(&x.to_kind(Kind::BFloat16), 32);
    assert!(a.equal(&b), "act_quant differs from ops::act_quant");
    eprintln!("[ds41-kernel-test] act_quant bitwise == ops::act_quant");
    // requant_fp8 == act_quant_fp8 of engram::dequant (the BF16 rows the prefill GEMM used to quantize): random E4M3 bytes
    // (no NaN) under every scale byte 0..=254, then real engram table rows when the checkpoint has them
    {
        let check = |name: &str, wq: &Tensor, ws: &Tensor| {
            let (q1, s1) = requant_fp8(wq, ws);
            let (q2, s2) = act_quant_fp8(&super::engram::dequant(wq, ws));
            let (dq, ds) = (q1.ne_tensor(&q2).sum(Kind::Int64).int64_value(&[]), s1.ne_tensor(&s2).sum(Kind::Int64).int64_value(&[]));
            eprintln!("[ds41-kernel-test] requant_fp8 {name}: {dq} byte / {ds} scale mismatches of {} / {}", q1.numel(), s1.numel());
            assert!(dq == 0 && ds == 0, "requant_fp8 differs ({name})");
        };
        let r = 255 * 8;
        let wq = Tensor::randint(0x7f, [r, 256], (Kind::Int64, dev));
        let wq = (&wq + Tensor::randint(2, [r, 256], (Kind::Int64, dev)) * 128).to_kind(Kind::Uint8);   // sign bit
        let ws = (Tensor::arange(r * 8, (Kind::Int64, dev)).remainder(255)).to_kind(Kind::Uint8).view([r, 8]);
        check("random, scales 0..=254", &wq, &ws);
        if ck.has("layers.1.engram.embed.weight") {
            let (wb, sb) = (ck.bytes("layers.1.engram.embed.weight"), ck.bytes("layers.1.engram.embed.scale"));
            let rows: Vec<i64> = (0..65536i64).map(|i| (i * 5_860_103) % (wb.len() as i64 / 256)).collect();
            let (w, s) = super::engram::gather(wb, sb, &rows, 256);
            let up = |b: &Vec<u8>| Tensor::from_slice(b).to_device(dev);
            check("engram layer 1, 65536 rows", &up(&w).view([65536, 256]), &up(&s).view([65536, 8]));
        }
    }
    // hc_mix as GEMV side jobs (hc_mix_jobs) == hc_mix, and the GEMVs' own results unchanged, for 1..=16 rows
    {
        let (k, n) = (5120i64, 1792i64);
        let w = (Tensor::randint(0x7e, [n, k], (Kind::Int64, dev)) + Tensor::randint(2, [n, k], (Kind::Int64, dev)) * 128).to_kind(Kind::Uint8);
        let ws = (Tensor::randint(6, [n / 32, k / 32], (Kind::Int64, dev)) + 124).to_kind(Kind::Uint8);
        for t in 1..=16i64 {
            let h = (Tensor::randn([t, 4, 5120], (Kind::Float, dev)) * 2.0).to_kind(Kind::BFloat16);
            let f = Tensor::randn([24, 4 * 5120], (Kind::Float, dev)) * 0.01;
            let (sc, bs) = (Tensor::randn([3], (Kind::Float, dev)), Tensor::randn([24], (Kind::Float, dev)));
            let x = Tensor::randn([t, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let r = hc_mix(&h, &f, &sc, &bs, 1e-6, 1e-6, 20);
            let (y1, y2) = (fp8_gemv_mx(&x, &w, &ws), fp8_gemv_mx_bf16(&x, &w, &ws));
            let j = hc_mix_jobs(&h, &f, &sc, &bs, 1e-6, 1e-6, 20).expect("hc_mix_jobs path");
            let (z1, z2) = (fp8_gemv_mx(&x, &w, &ws), fp8_gemv_mx_bf16(&x, &w, &ws));
            assert!(hc_jobs_done());
            let same = r.0.equal(&j.0) && r.1.equal(&j.1) && r.2.equal(&j.2) && y1.equal(&z1) && y2.equal(&z2);
            assert!(same, "hc_mix_jobs differs from hc_mix at {t} rows");
        }
        eprintln!("[ds41-kernel-test] hc_mix_jobs (GEMV side jobs) bitwise == hc_mix, GEMV results unchanged, rows 1..=16");
    }
    let report = |name: &str, fast: &Tensor, refv: &Tensor| {
        let rel = ((fast - refv).abs().max().double_value(&[])) / refv.abs().max().double_value(&[]).max(1e-30);
        eprintln!("[ds41-kernel-test] {name}: max |diff| / max |ref| = {rel:.2e}");
        assert!(rel < 1e-5, "{name} differs");
    };
    let time = |f: &dyn Fn() -> Tensor| { for _ in 0..3 { let _ = f(); } tch::Cuda::synchronize(0);
        let t = std::time::Instant::now(); for _ in 0..20 { let _ = f(); } tch::Cuda::synchronize(0); t.elapsed().as_secs_f64() * 1e6 / 20.0 };
    // FP8: attention wq_b [32768, 1280] and wo_b [5120, 8192]
    for name in ["layers.3.attn.wq_b", "layers.3.attn.wo_b", "layers.3.ffn.shared_experts.w1"] {
        let w = ck.get(&format!("{name}.weight"), Kind::Uint8);
        let s = ck.get(&format!("{name}.scale"), Kind::Uint8);
        let wd = dequant_fp8(&w.view_dtype(Kind::Float8e4m3fn), &s).to_kind(Kind::Float);
        let (n, k) = (w.size()[0], w.size()[1]);
        for m in [1i64, 2, 3, 4, 8] {
            let xq = super::ops::act_quant(&(Tensor::randn([m, k], (Kind::Float, dev))).to_kind(Kind::BFloat16), 32);
            let fast = fp8_gemv(&xq, &w, &s);
            let refv = xq.to_kind(Kind::Float).matmul(&wd.tr());
            report(&format!("fp8 {name} m={m}"), &fast, &refv);
        }
        let xq = super::ops::act_quant(&Tensor::randn([1, k], (Kind::Float, dev)).to_kind(Kind::BFloat16), 32);
        let us = time(&|| fp8_gemv(&xq, &w, &s));
        eprintln!("[ds41-kernel-test] fp8 {name} [{n}x{k}] m=1: {us:.1} us, {:.0} GB/s", (n * k) as f64 / us / 1e3);
        if name.ends_with("wq_b") {
            // every (n, k) the ranks use: a row's result must not depend on the other rows
            for n2 in [512i64, 1024, 1536, 1792, 2048, 3072, 4096, 5120, 8192, 8536, 12288, 25600] {
                for k2 in [512i64, 768, 1024, 1280, 2048, 3072, 4096, 5120, 6144, 15360] {
                    let w2 = Tensor::randint(120, [n2, k2], (Kind::Uint8, dev));
                    let s2 = Tensor::randint_low(100, 130, [(n2 + 31) / 32, k2 / 32], (Kind::Uint8, dev));
                    let x1 = Tensor::randn([1, k2], (Kind::Float, dev)).to_kind(Kind::BFloat16);
                    let r1 = fp8_gemv_mx(&x1, &w2, &s2);
                    for m in [2i64, 5] {
                        let xm = x1.expand([m, k2], false).contiguous();
                        let d = (fp8_gemv_mx(&xm, &w2, &s2).narrow(0, 0, 1) - &r1).abs().max().double_value(&[]);
                        if d != 0.0 { eprintln!("[ds41-kernel-test] fp8_gemv_mx [{n2} x {k2}] m={m}: row 0 differs from m=1 by {d:.2e}"); }
                    }
                }
            }
        }
        let x16 = Tensor::randn([16, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let row0 = fp8_gemv_mx(&x16.narrow(0, 0, 1), &w, &s);
        for m in [2i64, 3, 16] {
            let d = (fp8_gemv_mx(&x16.narrow(0, 0, m), &w, &s).narrow(0, 0, 1) - &row0).abs().max().double_value(&[]);
            eprintln!("[ds41-kernel-test] fp8_gemv_mx {name}: row 0 at m={m} vs m=1 max |diff| {d:.2e}");
            assert!(d == 0.0, "fp8_gemv_mx: a row's result depends on m");
        }
        for m in [1i64, 3, 16] {
            let x = Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let xq = super::ops::act_quant(&x, 32);
            report(&format!("fp8_gemv_mx {name} m={m}"), &fp8_gemv_mx(&x, &w, &s), &xq.to_kind(Kind::Float).matmul(&wd.tr()));
        }
        for m in [9i64, 100, 2048, 8192] {
            let x = Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16);
            let xq = super::ops::act_quant(&x, 32);
            let fx = fp8_gemm_mx(&x, &w, &s);
            report(&format!("fp8_gemm_mx {name} m={m}"), &fx, &xq.to_kind(Kind::Float).matmul(&wd.tr()));
            let ux = time(&|| fp8_gemm_mx(&x, &w, &s));
            eprintln!("[ds41-kernel-test] fp8_gemm_mx {name} m={m}: {ux:.1} us ({:.1} TFLOPS)", (2 * m * n * k) as f64 / ux / 1e6);
            let fast = fp8_mm(&xq, &w, &s);
            let refv = xq.to_kind(Kind::Float).matmul(&wd.tr());
            report(&format!("fp8_mm {name} m={m}"), &fast, &refv);
            let us = time(&|| fp8_mm(&xq, &w, &s));
            let us0 = time(&|| xq.matmul(&dequant_fp8(&w.view_dtype(Kind::Float8e4m3fn), &s).tr()));
            eprintln!("[ds41-kernel-test] fp8_mm {name} m={m}: {us:.1} us ({:.1} TFLOPS); v0 dequant + cuBLAS {us0:.1} us", (2 * m * n * k) as f64 / us / 1e6);
        }
    }
    // MXFP4 expert
    let table = fp4_table(dev);
    for (name, _) in [("layers.3.ffn.experts.17.w1", 0), ("layers.3.ffn.experts.17.w2", 0)] {
        let w = ck.get(&format!("{name}.weight"), Kind::Uint8);
        let s = ck.get(&format!("{name}.scale"), Kind::Uint8);
        let wd = dequant_mxfp4(&w, &s, &table, Kind::Float);
        let (n, k) = (w.size()[0], w.size()[1] * 2);
        for m in [1i64, 2, 4, 8] {
            let xq = super::ops::act_quant(&Tensor::randn([m, k], (Kind::Float, dev)).to_kind(Kind::BFloat16), 32);
            let fast = mxfp4_gemv(&xq, &w, &s);
            let refv = xq.to_kind(Kind::Float).matmul(&wd.tr());
            report(&format!("mxfp4 {name} m={m}"), &fast, &refv);
        }
        let xq = super::ops::act_quant(&Tensor::randn([1, k], (Kind::Float, dev)).to_kind(Kind::BFloat16), 32);
        let us = time(&|| mxfp4_gemv(&xq, &w, &s));
        eprintln!("[ds41-kernel-test] mxfp4 {name} [{n}x{k}] m=1: {us:.1} us, {:.0} GB/s", (n * k / 2) as f64 / us / 1e3);
    }
    // rms_norm / rope against ops.rs
    for (rows, n) in [(1i64, 5120i64), (7, 1280), (300, 512)] {
        let x = (Tensor::randn([rows, n], (Kind::Float, dev)) * 2.0).to_kind(Kind::BFloat16);
        let w = (Tensor::randn([n], (Kind::Float, dev))).to_kind(Kind::BFloat16);
        let (a, b) = (rms_norm(&x, &w, 1e-20), super::ops::rms_norm(&x, &w, 1e-20));
        let d = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().max().double_value(&[]) / b.to_kind(Kind::Float).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] rms_norm [{rows}x{n}]: rel max diff {d:.2e}");
        assert!(d < 1e-2);
    }
    for (rows, h, inverse) in [(1i64, 24i64, false), (5, 24, true), (300, 1, false)] {
        let r = super::ops::Rope::new(64, 65536, 160000.0, 16.0, 32.0, 1.0).to(dev);
        let pos = Tensor::randint(60000, [rows], (Kind::Int64, dev));
        let x = (Tensor::randn([rows, h, 512], (Kind::Float, dev))).to_kind(Kind::BFloat16);
        let a = rope(&x, &pos, r.freqs(), 64, inverse);
        let b = super::ops::rope(&x, &r.at(&pos), 64, inverse);
        let d = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] rope rows={rows} H={h} inverse={inverse}: max |diff| {d:.2e}");
        assert!(d < 2e-2);
    }
    // routing and shared SwiGLU against the libtorch expressions
    for (t, e, k) in [(1i64, 384i64, 6i64), (7, 384, 6), (5, 128, 3), (300, 384, 6)] {
        let logits = Tensor::randn([t, e], (Kind::Float, dev)) * 2.0;
        let bias = Tensor::randn([e], (Kind::Float, dev)) * 0.1;
        let (i1, w1) = route(&logits, &bias, k, 1.5);
        let scores = logits.softplus().sqrt();
        let (_, i0) = (&scores + bias.unsqueeze(0)).topk(k, -1, true, true);
        let w0 = scores.gather(1, &i0, false);
        let w0 = if k > 1 { &w0 / (w0.sum_dim_intlist([-1].as_slice(), true, Kind::Float) + 1e-20) } else { w0 } * 1.5;
        let same = i1.eq_tensor(&i0).all().int64_value(&[]) == 1;
        let d = (&w1 - &w0).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] route T={t} E={e} k={k}: experts equal {same}, max |weight diff| {d:.2e}");
        assert!(same && d < 1e-5);
    }
    {
        let (hc, d) = (4i64, 5120i64);
        let qk = Tensor::randn([hc, d], (Kind::Float, dev)) * 0.05;
        let h = Tensor::randn([3, hc, d], (Kind::Float, dev)).to_kind(Kind::BFloat16);
        let kv = (Tensor::randn([3, (hc + 1) * d], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let a = engram_gate(&h, &kv, &qk, 1e-20);
        let key = kv.narrow(1, 0, hc * d).to_kind(Kind::Float).view([3, hc, d]);
        let value = kv.narrow(1, hc * d, d);
        let hf = h.to_kind(Kind::Float);
        let rstd = (hf.square().mean_dim([-1].as_slice(), false, Kind::Float) + 1e-20).rsqrt() * (key.square().mean_dim([-1].as_slice(), false, Kind::Float) + 1e-20).rsqrt();
        let dot = (&hf * qk.unsqueeze(0) * &key).sum_dim_intlist([-1].as_slice(), false, Kind::Float) * rstd * (d as f64).powf(-0.5);
        let gate = dot.abs().clamp_min(1e-6).sqrt().copysign(&dot).sigmoid();
        let b = (hf + gate.unsqueeze(-1) * value.to_kind(Kind::Float).unsqueeze(1)).to_kind(Kind::BFloat16);
        let diff = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().max().double_value(&[]);
        let r0 = engram_gate(&h.narrow(0, 0, 1), &kv.narrow(0, 0, 1), &qk, 1e-20);
        let dr = (a.narrow(0, 0, 1).to_kind(Kind::Float) - r0.to_kind(Kind::Float)).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] engram_gate: max |diff| vs libtorch {diff:.2e}, row 0 at T=3 vs T=1 {dr:.2e}");
        assert!(diff < 0.1 && dr == 0.0);
    }
    for (block, e4) in [(32i64, false), (16, true)] {
        for scale in [1e-3, 1.0, 40.0] {
            let x = (Tensor::randn([37, 4, 512], (Kind::Float, dev)) * scale).to_kind(Kind::BFloat16);
            let a = fp4_quant(&x, block, e4);
            let b = super::ops::fp4_quant(&x, block, e4);
            let d = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().max().double_value(&[]);
            eprintln!("[ds41-kernel-test] fp4_quant block {block} e4m3 {e4} scale {scale}: max |diff| {d:.2e}");
            assert!(d == 0.0);
        }
    }
    for t in [1i64, 9, 300] {
        let gu = Tensor::randn([t, 1536], (Kind::Float, dev)) * 3.0;
        let a = swiglu_quant(&gu, 10.0);
        let bfr = |x: Tensor| x.to_kind(Kind::BFloat16).to_kind(Kind::Float);
        let (g, u) = (bfr(gu.narrow(1, 0, 768)).clamp_max(10.0), bfr(gu.narrow(1, 768, 768)).clamp(-10.0, 10.0));
        let b = act_quant(&(g.silu() * u).to_kind(Kind::BFloat16));
        let d = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] swiglu_quant T={t}: max |diff| {d:.2e}");
        assert!(d <= b.to_kind(Kind::Float).abs().max().double_value(&[]) * 0.1);
    }
    // head GEMV (BF16 weight, FP32 math)
    {
        let w = ck.get("head.weight", Kind::BFloat16).narrow(0, 0, 43136).contiguous();
        let wf = w.to_kind(Kind::Float);
        for m in [1i64, 3, 8] {
            let x = (Tensor::randn([m, 5120], (Kind::Float, dev))).to_kind(Kind::BFloat16);
            report(&format!("bf16_gemv head m={m}"), &bf16_gemv(&x, &w), &x.to_kind(Kind::Float).matmul(&wf.tr()));
            report(&format!("bf16_gemv_rows head m={m}"), &bf16_gemv_rows(&x, &w), &x.to_kind(Kind::Float).matmul(&wf.tr()));
            let us = time(&|| bf16_gemv(&x, &w));
            let usr = time(&|| bf16_gemv_rows(&x, &w));
            eprintln!("[ds41-kernel-test] head m={m}: CUDA-core {us:.1} us, MMA {usr:.1} us");
        }
        let x = (Tensor::randn([1, 5120], (Kind::Float, dev))).to_kind(Kind::BFloat16);
        let us = time(&|| bf16_gemv(&x, &w));
        let us0 = time(&|| x.to_kind(Kind::Float).matmul(&w.to_kind(Kind::Float).tr()));
        eprintln!("[ds41-kernel-test] bf16_gemv head [43136x5120] m=1: {us:.1} us ({:.0} GB/s; v0 {us0:.1} us)", (43136 * 5120 * 2) as f64 / us / 1e3);
        // the joint q / kv projection (qakv [1792, 5120] FP8) whole vs a third of its rows (608: 19 of 56 row blocks),
        // 6 copies rotating (cold); the rows of the third equal the whole's rows
        {
            let q = (Tensor::randn([1792, 5120], (Kind::Float, dev)) * 0.05).to_kind(Kind::Float8e4m3fn).view_dtype(Kind::Uint8);
            let s = Tensor::randint_low(118, 126, [56, 160], (Kind::Uint8, dev));
            let full: Vec<(Tensor, Tensor)> = (0..6).map(|_| (q.copy(), s.copy())).collect();
            let third: Vec<(Tensor, Tensor)> = (0..6).map(|_| (q.narrow(0, 608, 608).copy(), s.narrow(0, 19, 19).copy())).collect();
            for m in [1i64, 2, 4, 8] {
                let x = Tensor::randn([m, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
                let same = fp8_gemv_mx_bf16(&x, &third[0].0, &third[0].1).equal(&fp8_gemv_mx_bf16(&x, &full[0].0, &full[0].1).narrow(1, 608, 608));
                let i = std::cell::Cell::new(0usize);
                let uf = time(&|| { let k = i.get(); i.set(k + 1); fp8_gemv_mx_bf16(&x, &full[k % 6].0, &full[k % 6].1) });
                let ut = time(&|| { let k = i.get(); i.set(k + 1); fp8_gemv_mx_bf16(&x, &third[k % 6].0, &third[k % 6].1) });
                eprintln!("[ds41-kernel-test] qakv m={m}: 1792 rows {uf:.1} us, 608 rows {ut:.1} us (rows bitwise: {same})");
            }
        }
        mx_table(dev);
        // DS41_C12: the coded GEMV must equal the BF16 one bit for bit (head, a router, the Markov head shape), every m;
        // timing on a rotating set of 4 head copies (1.8 GB > L2) for both
        if c12_on() {
            let mtp_head = ck.get("mtp.2.markov_head.head.weight", Kind::BFloat16).narrow(0, 0, 43136).contiguous();
            let gate = ck.get("layers.20.ffn.gate.weight", Kind::BFloat16);
            for (name, w) in [("head", &w), ("gate", &gate), ("markov", &mtp_head)] {
                let wc = w.copy();   // a fresh pointer: registered; the original stays on the BF16 kernel
                let bytes = c12_register(&wc);
                let mut same = bytes > 0;
                for m in 1..=8i64 {
                    let x = Tensor::randn([m, w.size()[1]], (Kind::Float, dev)).to_kind(Kind::BFloat16);
                    let (want, got) = (bf16_gemv(&x, w), bf16_gemv(&x, &wc));
                    if !got.equal(&want) { same = false; eprintln!("[ds41-kernel-test] C12 {name} m={m}: DIFFERENT"); }
                }
                eprintln!("[ds41-kernel-test] C12 {name} [{}x{}] ({:.1} MB coded / {:.1} MB): 1..8 rows {}", w.size()[0], w.size()[1],
                    bytes as f64 / 1e6, (w.numel() * 2) as f64 / 1e6, if same { "IDENTICAL" } else { "DIFFERENT" });
            }
            {
                let gs: Vec<Tensor> = (0..32).map(|_| gate.copy()).collect();
                let gc: Vec<Tensor> = (0..32).map(|_| gate.copy()).collect();
                for t in &gc { c12_register(t); }
                for m in [1i64, 4] {
                    let x = Tensor::randn([m, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
                    let i = std::cell::Cell::new(0usize);
                    let ub = time(&|| { let k = i.get(); i.set(k + 1); bf16_gemv(&x, &gs[k % 32]) });
                    let uc = time(&|| { let k = i.get(); i.set(k + 1); bf16_gemv(&x, &gc[k % 32]) });
                    eprintln!("[ds41-kernel-test] gate [384x5120] m={m} rotating: BF16 {ub:.1} us, C12 {uc:.1} us");
                }
            }
            let ws: Vec<Tensor> = (0..4).map(|_| w.copy()).collect();
            let wc: Vec<Tensor> = (0..4).map(|_| w.copy()).collect();
            for t in &wc { c12_register(t); }
            for m in [1i64, 2, 4, 8] {
                let x = Tensor::randn([m, 5120], (Kind::Float, dev)).to_kind(Kind::BFloat16);
                let i = std::cell::Cell::new(0usize);
                let ub = time(&|| { let k = i.get(); i.set(k + 1); bf16_gemv(&x, &ws[k % 4]) });
                let uc = time(&|| { let k = i.get(); i.set(k + 1); bf16_gemv(&x, &wc[k % 4]) });
                eprintln!("[ds41-kernel-test] head m={m} rotating: BF16 {ub:.1} us ({:.0} GB/s), C12 {uc:.1} us ({:.0} GB/s of BF16-equivalent)",
                    (43136 * 5120 * 2) as f64 / ub / 1e3, (43136 * 5120 * 2) as f64 / uc / 1e3);
            }
        }
    }
    // sparse attention against ops.rs: 24 and 16 heads, pool of 3000 rows, 640 indices with invalid ones
    for (t, h, k) in [(1i64, 24i64, 640i64), (2, 24, 640), (5, 16, 128), (8, 16, 640), (300, 24, 640), (2048, 24, 640)] {
        let q = (Tensor::randn([t, h, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let kv = (Tensor::randn([3000, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let idx = Tensor::randint(3000, [t, k], (Kind::Int64, dev));
        let drop = Tensor::rand([t, k], (Kind::Float, dev)).lt(0.2);
        let idx = idx.masked_fill(&drop, -1);
        if t > 1 { let _ = idx.get(0).fill_(-1); }   // a row without any position
        let sink = Tensor::randn([h], (Kind::Float, dev));
        let scale = 512f64.powf(-0.5);
        let fast = sparse_attn(&q, &kv.narrow(0, 0, 1000), Some(&kv.narrow(0, 1000, 2000)), &idx, &sink, scale).expect("sparse_attn kernel unavailable");
        let refv = super::ops::sparse_attn(&q, &kv, &idx, &sink, scale, 256);
        let d = (fast.to_kind(Kind::Float) - refv.to_kind(Kind::Float)).abs().max().double_value(&[]);
        let m = refv.to_kind(Kind::Float).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] sparse_attn T={t} H={h} K={k}: max |diff| {d:.2e} (max |ref| {m:.2e})");
        assert!(d <= m * 1e-2, "sparse_attn differs");
        let us = time(&|| sparse_attn(&q, &kv, None, &idx, &sink, scale).unwrap());
        let us0 = time(&|| super::ops::sparse_attn(&q, &kv, &idx, &sink, scale, 256));
        eprintln!("[ds41-kernel-test] sparse_attn T={t} H={h} K={k}: {us:.1} us (v0 {us0:.1} us)");
    }
    // sparse attention with kernel-side window indices (ring + chunk + compressed pools) against explicit indices
    for (t, start) in [(1i64, 700i64), (3, 50), (200, 0), (200, 1000)] {
        let (w, h) = (128i64, 24i64);
        let q = (Tensor::randn([t, h, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let ring = (Tensor::randn([w, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let chunk = (Tensor::randn([t, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let comp = (Tensor::randn([900, 512], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let pos = Tensor::arange_start(start, start + t, (Kind::Int64, dev));
        let ci = (Tensor::randint(900, [t, 256], (Kind::Int64, dev)) + w + t).masked_fill(&Tensor::rand([t, 256], (Kind::Float, dev)).lt(0.1), -1);
        let off = Tensor::arange(w, (Kind::Int64, dev));
        let kpos = pos.unsqueeze(-1) - (w - 1) + off.unsqueeze(0);
        let widx = (&kpos - start + w).where_self(&kpos.ge(start), &kpos.remainder(w)).masked_fill(&kpos.lt(0), -1);
        let idx = Tensor::cat(&[widx, ci.shallow_clone()], 1);
        let full = Tensor::cat(&[ring.shallow_clone(), chunk.shallow_clone(), comp.shallow_clone()], 0);
        let sink = Tensor::randn([h], (Kind::Float, dev));
        let scale = 512f64.powf(-0.5);
        let fast = sparse_attn_pools(&q, [Some(&ring), Some(&chunk), Some(&comp)], Some((w, start, &pos)), &ci, &sink, scale).unwrap();
        let refv = sparse_attn(&q, &full, None, &idx, &sink, scale).unwrap();
        let d = (fast.to_kind(Kind::Float) - refv.to_kind(Kind::Float)).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] sparse_attn window pools T={t} start={start}: max |diff| vs explicit {d:.2e}");
        assert!(d == 0.0, "window indices differ");
    }
    // indexer scores against the libtorch expression
    for (t, tc, ratio) in [(1i64, 3000i64, 1i64), (4, 700, 2), (300, 1500, 2)] {
        let q = (Tensor::randn([t, 32, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let k = (Tensor::randn([tc, 128], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let w = (Tensor::randn([t, 32], (Kind::Float, dev)) * 0.1).to_kind(Kind::BFloat16);
        let pos = Tensor::randint(tc * ratio, [t], (Kind::Int64, dev));
        let fast = index_score(&q, &k, &w, &pos, ratio);
        let s = q.matmul(&k.tr().unsqueeze(0)).relu();
        let refv = (s * w.unsqueeze(-1)).sum_dim_intlist([1].as_slice(), false, Kind::BFloat16);
        let lens = (&pos + 1).floor_divide_scalar(ratio);
        let refv = refv.masked_fill(&Tensor::arange(tc, (Kind::Int64, dev)).unsqueeze(0).ge_tensor(&lens.unsqueeze(-1)), f64::NEG_INFINITY);
        let fin = refv.isfinite();
        let same_inf = fast.isfinite().eq_tensor(&fin).all().int64_value(&[]) == 1;
        let d = (fast.to_kind(Kind::Float) - refv.to_kind(Kind::Float)).masked_fill(&fin.logical_not(), 0.0).abs().max().double_value(&[]);
        let m = refv.to_kind(Kind::Float).masked_fill(&fin.logical_not(), 0.0).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] index_score T={t} Tc={tc}: max |diff| {d:.2e} (max |ref| {m:.2e}), masks equal {same_inf}");
        assert!(same_inf && d <= m * 2e-2);
        let us = time(&|| index_score(&q, &k, &w, &pos, ratio));
        let us0 = time(&|| (q.matmul(&k.tr().unsqueeze(0)).relu() * w.unsqueeze(-1)).sum_dim_intlist([1].as_slice(), false, Kind::BFloat16));
        eprintln!("[ds41-kernel-test] index_score T={t} Tc={tc}: {us:.1} us (v0 {us0:.1} us)");
    }
    // mHC against ops.rs (layer 3 weights)
    for t in [1i64, 7, 300, 2048] {
        let f = ck.get("layers.3.hc_attn_fn", Kind::Float);
        let (sc, b) = (ck.get("layers.3.hc_attn_scale", Kind::Float), ck.get("layers.3.hc_attn_base", Kind::Float));
        let h = (Tensor::randn([t, 4, 5120], (Kind::Float, dev)) * 0.3).to_kind(Kind::BFloat16);
        let (p, q, c) = hc_mix(&h, &f, &sc, &b, 1e-20, 1e-6, 20);
        let x = h.reshape([t, -1]).to_kind(Kind::Float);
        let rs = (x.square().mean_dim([-1].as_slice(), true, Kind::Float) + 1e-20).rsqrt();
        let (p0, q0, c0) = super::ops::hc_split_sinkhorn(&(x.matmul(&f.tr()) * &rs), &sc, &b, 4, 20, 1e-6);
        let d = |a: &Tensor, b: &Tensor| (a - b).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] hc_mix T={t}: max |diff| pre {:.2e} post {:.2e} comb {:.2e}", d(&p, &p0), d(&q, &q0), d(&c, &c0));
        assert!(d(&p, &p0) < 1e-4 && d(&q, &q0) < 1e-4 && d(&c, &c0) < 1e-4, "hc_mix differs");
        let pre0 = (p0.unsqueeze(-1) * h.to_kind(Kind::Float)).sum_dim_intlist([1].as_slice(), false, Kind::Float).to_kind(Kind::BFloat16);
        let xin = (Tensor::randn([t, 5120], (Kind::Float, dev))).to_kind(Kind::BFloat16);
        let post0 = (q0.unsqueeze(-1) * xin.to_kind(Kind::Float).unsqueeze(1)
            + (c0.unsqueeze(-1) * h.to_kind(Kind::Float).unsqueeze(2)).sum_dim_intlist([1].as_slice(), false, Kind::Float)).to_kind(Kind::BFloat16);
        let (a, b2) = (hc_pre(&h, &p0), hc_post(&xin, &h, &q0, &c0));
        let rel = |a: &Tensor, b: &Tensor| d(&a.to_kind(Kind::Float), &b.to_kind(Kind::Float)) / b.to_kind(Kind::Float).abs().max().double_value(&[]);
        eprintln!("[ds41-kernel-test] hc_pre / hc_post T={t}: rel max diff {:.2e} / {:.2e}", rel(&a, &pre0), rel(&b2, &post0));
        assert!(rel(&a, &pre0) < 1e-2 && rel(&b2, &post0) < 1e-2);
        let us = time(&|| hc_mix(&h, &f, &sc, &b, 1e-20, 1e-6, 20).0);
        let us0 = time(&|| super::ops::hc_split_sinkhorn(&(x.matmul(&f.tr()) * &rs), &sc, &b, 4, 20, 1e-6).0);
        eprintln!("[ds41-kernel-test] hc_mix T={t}: {us:.1} us (v0 {us0:.1} us)");
    }
    // fused routed experts, layer 3, intermediate slice of rank 0 under TP3 (I = 768)
    let (i0, il) = (0i64, 768i64);
    let f = "layers.3.ffn.experts";
    let mut st: Vec<Vec<Tensor>> = (0..6).map(|_| Vec::new()).collect();
    for e in 0..384 {
        st[0].push(ck.rows(&format!("{f}.{e}.w1.weight"), i0, il, Kind::Uint8));
        st[1].push(ck.rows(&format!("{f}.{e}.w1.scale"), i0, il, Kind::Uint8));
        st[2].push(ck.rows(&format!("{f}.{e}.w3.weight"), i0, il, Kind::Uint8));
        st[3].push(ck.rows(&format!("{f}.{e}.w3.scale"), i0, il, Kind::Uint8));
        st[4].push(ck.cols(&format!("{f}.{e}.w2.weight"), i0 / 2, il / 2, Kind::Uint8));
        st[5].push(ck.cols(&format!("{f}.{e}.w2.scale"), i0 / 32, il / 32, Kind::Uint8));
    }
    let st: Vec<Tensor> = st.iter().map(|v| Tensor::stack(v, 0)).collect();
    let w = [&st[0], &st[1], &st[2], &st[3], &st[4], &st[5]];
    let (p1, q1) = moe_pack(&st[0], &st[1]);
    let (p3, q3) = moe_pack(&st[2], &st[3]);
    let (p2, q2) = moe_pack(&st[4], &st[5]);
    let wp = [&p1, &q1, &p3, &q3, &p2, &q2];
    let wx = [&p1, &q1, &moe_pack_mx(&p3), &q3, &moe_pack_mx(&p2), &q2];
    let p1x = moe_pack_mx(&p1);
    let wx = [&p1x, wx[1], wx[2], wx[3], wx[4], wx[5]];
    for r in [1i64, 3, 8, 40, 300, 2000] {
        let x = (Tensor::randn([r, 5120], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let (_, idx) = Tensor::rand([r, 384], (Kind::Float, dev)).topk(6, -1, true, true);
        let rw = Tensor::rand([r, 6], (Kind::Float, dev)) * 0.5;
        let fast = moe_mx(&x, &idx, &rw, wx, None);
        let refv = moe_ref(&super::ops::act_quant(&x, 32), &idx, &rw, w, &table);
        let l2 = ((&fast - &refv).square().sum(Kind::Float) / refv.square().sum(Kind::Float)).sqrt().double_value(&[]);
        eprintln!("[ds41-kernel-test] moe_mx R={r}: rel L2 vs reference {l2:.2e}");
        assert!(l2 < 1e-3, "moe_mx differs");
    }
    for r in [1i64, 2, 4, 8, 512, 2048, 4096, 8192] {
        let x = (Tensor::randn([r, 5120], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16);
        let (_, idx) = Tensor::rand([r, 384], (Kind::Float, dev)).topk(6, -1, true, true);
        let rw = Tensor::rand([r, 6], (Kind::Float, dev)) * 0.5;
        let us = time(&|| moe_mx(&x, &idx, &rw, wx, None));
        let us0 = time(&|| moe(&super::ops::act_quant(&x, 32), &idx, &rw, wp, None));
        eprintln!("[ds41-kernel-test] moe_mx R={r}: {us:.1} us (LUT kernels {us0:.1} us)");
    }
    for r in [1i64, 3, 8, 40] {
        let xq = super::ops::act_quant(&(Tensor::randn([r, 5120], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16), 32);
        let (_, idx) = Tensor::rand([r, 384], (Kind::Float, dev)).topk(6, -1, true, true);
        let rw = Tensor::rand([r, 6], (Kind::Float, dev)) * 0.5;
        let fast = moe(&xq, &idx, &rw, wp, None);
        let refv = moe_ref(&xq, &idx, &rw, w, &table);
        let rel = ((&fast - &refv).abs().max().double_value(&[])) / refv.abs().max().double_value(&[]).max(1e-30);
        let l2 = ((&fast - &refv).square().sum(Kind::Float) / refv.square().sum(Kind::Float)).sqrt().double_value(&[]);
        eprintln!("[ds41-kernel-test] moe R={r}: max |diff| / max |ref| = {rel:.2e}, rel L2 {l2:.2e}");
        assert!(l2 < 1e-3, "moe differs");
    }
    for r in [1i64, 2, 4, 8, 64, 512, 2048] {
        let xq = super::ops::act_quant(&(Tensor::randn([r, 5120], (Kind::Float, dev)) * 0.5).to_kind(Kind::BFloat16), 32);
        let (_, idx) = Tensor::rand([r, 384], (Kind::Float, dev)).topk(6, -1, true, true);
        let rw = Tensor::rand([r, 6], (Kind::Float, dev)) * 0.5;
        let us = time(&|| moe(&xq, &idx, &rw, wp, None));
        let experts = r.min(384) as f64 * 6.0;
        let bytes = experts.min(384.0) * 3.0 * (il * 5120 / 2) as f64;
        eprintln!("[ds41-kernel-test] moe R={r}: {us:.1} us (distinct-expert weight bytes {:.0} MB, {:.0} GB/s)", bytes / 1e6, bytes / us / 1e3);
    }
}
