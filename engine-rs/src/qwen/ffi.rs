//! FFI to the Qwen3.8 CUDA kernels (shim/qwen_*.cu). All launches go to torch's current stream.
use std::ffi::c_void;
use tch::Tensor;

type St = *mut c_void;
extern "C" {
    fn rs_current_stream() -> St;
    fn qwen_exl3_had_in(x: *const c_void, x_f32: i32, ldx: i64, suh: *const c_void, xh: *mut c_void, m: i32, k: i32, st: St) -> i32;
    fn qwen_exl3_gemv(xh: *const c_void, tr: *const c_void, part: *mut c_void, m: i32, k: i32, n: i32, bits: i32, s: i32, st: St) -> i32;
    fn qwen_exl3_finish(part: *const c_void, s: i32, svh: *const c_void, y: *mut c_void, y_f32: i32, ldy: i64, m: i32, n: i32, st: St) -> i32;
    fn qwen_gdn_conv(x: *const c_void, ldx: i64, state: *const c_void, w: *const c_void, y: *mut c_void, t: i32, st: St) -> i32;
    fn qwen_gdn_conv_commit(x: *const c_void, ldx: i64, state: *mut c_void, n: i32, st: St) -> i32;
    fn qwen_gdn_recur(y: *const c_void, ab: *const c_void, ldab: i64, z: *const c_void, ldz: i64, a_log: *const c_void,
                      dt_bias: *const c_void, norm_w: *const c_void, s: *mut c_void, out: *mut c_void, t: i32, write: i32, st: St) -> i32;
    fn qwen_qsa_select(qi: *const c_void, pooled: *const c_void, r: i32, pos0: *const c_void, scores: *mut c_void, ld: i32, sel: *mut c_void, cnt: *mut c_void, sel_ld: i32, st: St) -> i32;
    fn qwen_exl3_reconstruct(tr: *const c_void, w: *mut c_void, k: i32, n: i32, bits: i32, st: St) -> i32;
    fn qwen_exl3_fold(tr: *const c_void, suh: *const c_void, svh: *const c_void, k: i32, n: i32, bits: i32, out: *mut c_void, ldo: i64, st: St) -> i32;
    fn rs_lt_mm16(x: *const c_void, w: *const c_void, y: *mut c_void, m: i32, n: i32, k: i32, fp32: i32, algorithm: i32) -> i32;
    fn qwen_qsa_attn(q: *const c_void, r: i32, pos0: *const c_void, sel: *const c_void, cnt: *const c_void, sel_ld: i32,
                     kc: *const c_void, ksc: *const c_void, vc: *const c_void, vsc: *const c_void, gate: *const c_void,
                     part_ml: *mut c_void, part_acc: *mut c_void, splits: i32, out: *mut c_void, st: St) -> i32;
    fn qwen_moe_route(logits: *const c_void, r: i32, idx: *mut c_void, w: *mut c_void, st: St) -> i32;
    fn qwen_moe_ws_bytes(r: i32, s: i32) -> usize;
    fn qwen_moe_experts(x: *const c_void, ldx: i64, r: i32, idx: *const c_void, w: *const c_void, tab: *const c_void,
                        add: *const c_void, out: *mut c_void, ws: *mut c_void, s: i32, st: St) -> i32;
    fn qwen_hc_ws_bytes(r: i32) -> usize;
    fn qwen_f16_ws_bytes(m: i32, n: i32, k: i32) -> usize;
    fn qwen_hc_norm(x: *const c_void, r: i32, w1: *const c_void, n: *mut c_void, st: St) -> i32;
    fn qwen_hc_mid2(d: *const c_void, ldd: i32, t: *mut c_void, post: *mut c_void, r: i32, inject: i32, st: St) -> i32;
    fn qwen_hc_mix2(g: *const c_void, n: *const c_void, mixed: *mut c_void, r: i32, st: St) -> i32;
    fn qwen_f16_gemv(x: *const c_void, ldx: i64, m: i32, w: *const c_void, k: i32, n: i32, y: *mut c_void, ldy: i64, ws: *mut c_void, st: St) -> i32;
    fn qwen_hc_mix(x: *const c_void, r: i32, w1: *const c_void, w: *const c_void, inject: i32, up: *const c_void,
                   mixed: *mut c_void, post: *mut c_void, ws: *mut c_void, st: St) -> i32;
    fn qwen_hc_apply(x: *mut c_void, post: *const c_void, y: *const c_void, r: i32, st: St) -> i32;
    fn qwen_ple_decode(packed: *const c_void, words: i32, bits: i32, head_bias: *const c_void, emb: *mut c_void, r: i32, st: St) -> i32;
    fn qwen_ple_apply(x: *mut c_void, key: *const c_void, value: *const c_void, nk1: *const c_void, nq1: *const c_void,
                      nc1: *const c_void, conv_w: *const c_void, win: *const c_void, gated: *mut c_void, nrm: *mut c_void, r: i32, st: St) -> i32;
    fn qwen_exl3_finish_h(part: *const c_void, svh: *const c_void, y: *mut c_void, ldy: i64, m: i32, n: i32, st: St) -> i32;
    fn qwen_q8_encode(w: *const c_void, n: i32, k: i32, q: *mut c_void, s: *mut c_void, st: St) -> i32;
    fn qwen_q8_ws_bytes(m: i32, n: i32, k: i32) -> usize;
    fn qwen_q8_gemv(x: *const c_void, ldx: i64, m: i32, q: *const c_void, s: *const c_void, k: i32, n: i32, y: *mut c_void, ldy: i64,
                    ws: *mut c_void, st: St) -> i32;
    fn qwen_hc_apply_norm_h(x: *mut c_void, post: *const c_void, y: *const c_void, w1: *const c_void, n: *mut c_void, scale: *mut c_void,
                            r: i32, st: St) -> i32;
    fn qwen_hc_norm_h(x: *const c_void, r: i32, w1: *const c_void, n: *mut c_void, scale: *mut c_void, st: St) -> i32;
    fn qwen_hc_mid2h(d: *const c_void, ldd: i32, t: *mut c_void, post: *mut c_void, r: i32, inject: i32, st: St) -> i32;
    fn qwen_hc_mix2h(g: *const c_void, x: *const c_void, scale: *const c_void, w1: *const c_void, mixed: *mut c_void, r: i32, st: St) -> i32;
    fn qwen_ple_commit(win: *mut c_void, nrm: *const c_void, n: i32, nd: *const c_void, st: St) -> i32;
    fn qwen_gdn_conv_segs(x: *const c_void, ldx: i64, nseg: i32, r0: *const i32, t: *const i32, state: *const *mut c_void,
                          w: *const c_void, y: *mut c_void, st: St) -> i32;
    fn qwen_gdn_conv_commit_segs(x: *const c_void, ldx: i64, nseg: i32, r0: *const i32, t: *const i32, state: *const *mut c_void,
                                 nd: *const c_void, st: St) -> i32;
    fn qwen_gdn_recur_segs(y: *const c_void, ab: *const c_void, ldab: i64, z: *const c_void, ldz: i64, a_log: *const c_void,
                           dt_bias: *const c_void, norm_w: *const c_void, nseg: i32, r0: *const i32, t: *const i32,
                           s: *const *mut c_void, nd: *const c_void, out: *mut c_void, write: i32, ws: *mut c_void, st: St) -> i32;
    fn qwen_gdn_ws_bytes(r: i32) -> usize;
}

/// Row segments of a multi-sequence batch: (first row, rows) per sequence, with that sequence's state tensor.
pub struct Segs { r0: Vec<i32>, t: Vec<i32>, s: Vec<*mut c_void> }
impl Segs {
    pub fn new(rows: &[(i64, i64)], states: &[&Tensor]) -> Self {
        assert!(!rows.is_empty() && rows.len() <= 8 && rows.len() == states.len());
        Segs { r0: rows.iter().map(|r| r.0 as i32).collect(), t: rows.iter().map(|r| r.1 as i32).collect(),
               s: states.iter().map(|t| t.data_ptr()).collect() }
    }
    fn n(&self) -> i32 { self.r0.len() as i32 }
}

#[inline] fn st() -> St { unsafe { rs_current_stream() } }
#[inline] fn p(t: &Tensor) -> *mut c_void { t.data_ptr() }
#[inline] fn opt(t: Option<&Tensor>) -> *mut c_void { t.map_or(std::ptr::null_mut(), |t| t.data_ptr()) }
#[inline] fn ok(r: i32, what: &str) { assert_eq!(r, 0, "qwen kernel {what} failed ({r})"); }

pub fn exl3_had_in(x: &Tensor, suh: &Tensor, xh: &Tensor, m: i64, k: i64) {
    let f32 = (x.kind() == tch::Kind::Float) as i32;
    ok(unsafe { qwen_exl3_had_in(p(x), f32, x.stride()[0], p(suh), p(xh), m as i32, k as i32, st()) }, "exl3_had_in");
}
pub fn exl3_gemv(xh: &Tensor, tr: &Tensor, part: &Tensor, m: i64, k: i64, n: i64, bits: i32, s: i32) {
    ok(unsafe { qwen_exl3_gemv(p(xh), p(tr), p(part), m as i32, k as i32, n as i32, bits, s, st()) }, "exl3_gemv");
}
pub fn exl3_finish(part: &Tensor, s: i32, svh: &Tensor, y: &Tensor, m: i64, n: i64) {
    let f32 = (y.kind() == tch::Kind::Float) as i32;
    ok(unsafe { qwen_exl3_finish(p(part), s, p(svh), p(y), f32, y.stride()[0], m as i32, n as i32, st()) }, "exl3_finish");
}
pub fn gdn_conv(x: &Tensor, state: &Tensor, w: &Tensor, y: &Tensor, t: i64) {
    ok(unsafe { qwen_gdn_conv(p(x), x.stride()[0], p(state), p(w), p(y), t as i32, st()) }, "gdn_conv");
}
pub fn gdn_conv_commit(x: &Tensor, state: &Tensor, n: i64) {
    ok(unsafe { qwen_gdn_conv_commit(p(x), x.stride()[0], p(state), n as i32, st()) }, "gdn_conv_commit");
}
#[allow(clippy::too_many_arguments)]
pub fn gdn_recur(y: &Tensor, ab: &Tensor, z: &Tensor, a_log: &Tensor, dt_bias: &Tensor, norm: &Tensor, s: &Tensor,
                 out: Option<&Tensor>, t: i64, write: bool) {
    ok(unsafe { qwen_gdn_recur(p(y), p(ab), ab.stride()[0], p(z), z.stride()[0], p(a_log), p(dt_bias), p(norm), p(s), opt(out),
                               t as i32, write as i32, st()) }, "gdn_recur");
}
#[allow(clippy::too_many_arguments)]
pub fn qsa_prep(qp: &Tensor, kp: &Tensor, vp: &Tensor, ip: &Tensor, r: i64, pos0: &Tensor, qn: &Tensor, kn: &Tensor, iqn: &Tensor,
                q: &Tensor, gate: &Tensor, kc: &Tensor, ks: &Tensor, vc: &Tensor, vs: &Tensor, qi: &Tensor, ring: &Tensor, yarn: i32,
                mtab: &Tensor, mctl: &Tensor) {
    extern "C" { fn qwen_qsa_prep_m(qp: *const c_void, kp: *const c_void, vp: *const c_void, ip: *const c_void, r: i32, pos0: *const c_void,
                                    qn: *const c_void, kn: *const c_void, iqn: *const c_void, q: *mut c_void, gate: *mut c_void,
                                    kc: *mut c_void, ksc: *mut c_void, vc: *mut c_void, vsc: *mut c_void, qi: *mut c_void, ring: *mut c_void,
                                    yarn: i32, mtab: *const c_void, mctl: *const c_void, st: St) -> i32; }
    ok(unsafe { qwen_qsa_prep_m(p(qp), p(kp), p(vp), p(ip), r as i32, p(pos0), p(qn), p(kn), p(iqn), p(q), p(gate),
                                p(kc), p(ks), p(vc), p(vs), p(qi), p(ring), yarn, p(mtab), p(mctl), st()) }, "qsa_prep");
}
pub fn qsa_pool(ring: &Tensor, ip: &Tensor, r: i64, pos0: &Tensor, ikn: &Tensor, pooled: &Tensor, yarn: i32, mtab: &Tensor, mctl: &Tensor) {
    extern "C" { fn qwen_qsa_pool_m(ring: *const c_void, ip: *const c_void, r: i32, pos0: *const c_void, ikn: *const c_void, pooled: *mut c_void,
                                    yarn: i32, mtab: *const c_void, mctl: *const c_void, st: St) -> i32; }
    ok(unsafe { qwen_qsa_pool_m(p(ring), p(ip), r as i32, p(pos0), p(ikn), p(pooled), yarn, p(mtab), p(mctl), st()) }, "qsa_pool");
}
#[allow(clippy::too_many_arguments)]
pub fn qsa_select(qi: &Tensor, pooled: &Tensor, r: i64, pos0: &Tensor, scores: &Tensor, ld: i64, sel: &Tensor, cnt: &Tensor, sel_ld: i64) {
    ok(unsafe { qwen_qsa_select(p(qi), p(pooled), r as i32, p(pos0), p(scores), ld as i32, p(sel), p(cnt), sel_ld as i32, st()) }, "qsa_select");
}
/// Effective weight W^T [N, K] fp16 of an EXL3 linear (both Hadamards and scales folded in).
pub fn exl3_fold(tr: &Tensor, suh: &Tensor, svh: &Tensor, k: i64, n: i64, bits: i32, out: &Tensor) {
    ok(unsafe { qwen_exl3_fold(p(tr), p(suh), p(svh), k as i32, n as i32, bits, p(out), out.stride()[0], st()) }, "exl3_fold");
}
/// y [M, N] fp32 = x [M, K] fp16 @ w^T (w [N, K] fp16), cuBLASLt (fp32 accumulation), its first heuristic algorithm.
pub fn lt_mm16(x: &Tensor, w: &Tensor, y: &Tensor) {
    let (m, k, n) = (x.size()[0], x.size()[1], w.size()[0]);
    ok(unsafe { rs_lt_mm16(p(x), p(w), p(y), m as i32, n as i32, k as i32, 1, 0) }, "lt_mm16");
}
/// Decode linear in one launch (had_in + gemv + finish, bitwise equal): x [M <= 64, K] fp32 -> y [M, N] fp32.
#[allow(clippy::too_many_arguments)]
pub fn exl3_linear(x: &Tensor, suh: &Tensor, tr: &Tensor, svh: &Tensor, part: &Tensor, y: &Tensor, cnt: &Tensor, m: i64, k: i64, n: i64, bits: i32, s: i32) {
    extern "C" { fn qwen_exl3_linear(x: *const c_void, ldx: i64, suh: *const c_void, tr: *const c_void, svh: *const c_void, part: *mut c_void, y: *mut c_void,
                                     ldy: i64, cnt: *mut c_void, m: i32, k: i32, n: i32, bits: i32, s: i32, st: St) -> i32; }
    ok(unsafe { qwen_exl3_linear(p(x), x.stride()[0], p(suh), p(tr), p(svh), p(part), p(y), y.stride()[0], p(cnt), m as i32, k as i32, n as i32, bits, s, st()) }, "exl3_linear");
}
pub fn exl3_reconstruct(tr: &Tensor, w: &Tensor, k: i64, n: i64, bits: i32) {
    ok(unsafe { qwen_exl3_reconstruct(p(tr), p(w), k as i32, n as i32, bits, st()) }, "exl3_reconstruct");
}
#[allow(clippy::too_many_arguments)]
pub fn qsa_attn(q: &Tensor, r: i64, pos0: &Tensor, sel: Option<(&Tensor, &Tensor, i64)>, kc: &Tensor, ks: &Tensor, vc: &Tensor,
                vs: &Tensor, gate: &Tensor, ml: &Tensor, acc: &Tensor, splits: i64, out: &Tensor) {
    let (sp, cp, ld) = sel.map_or((std::ptr::null_mut(), std::ptr::null_mut(), 0), |(s, c, l)| (p(s), p(c), l as i32));
    ok(unsafe { qwen_qsa_attn(p(q), r as i32, p(pos0), sp, cp, ld, p(kc), p(ks), p(vc), p(vs), p(gate), p(ml), p(acc),
                              splits as i32, p(out), st()) }, "qsa_attn");
}
pub fn moe_route(logits: &Tensor, r: i64, idx: &Tensor, w: &Tensor) {
    ok(unsafe { qwen_moe_route(p(logits), r as i32, p(idx), p(w), st()) }, "moe_route");
}
pub fn moe_ws_bytes(r: i64, s: i32) -> i64 { unsafe { qwen_moe_ws_bytes(r as i32, s) as i64 } }
#[allow(clippy::too_many_arguments)]
pub fn moe_experts(x: &Tensor, r: i64, idx: &Tensor, w: &Tensor, tab: &Tensor, add: Option<&Tensor>, out: &Tensor, ws: &Tensor, s: i32) {
    ok(unsafe { qwen_moe_experts(p(x), x.stride()[0], r as i32, p(idx), p(w), p(tab), opt(add), p(out), p(ws), s, st()) }, "moe_experts");
}
pub fn hc_ws_bytes(r: i64) -> i64 { unsafe { qwen_hc_ws_bytes(r as i32) as i64 } }
#[allow(clippy::too_many_arguments)]
pub fn hc_mix(x: &Tensor, r: i64, w1: &Tensor, w: &Tensor, inject: bool, up: &Tensor, mixed: &Tensor, post: Option<&Tensor>, ws: &Tensor) {
    ok(unsafe { qwen_hc_mix(p(x), r as i32, p(w1), p(w), inject as i32, p(up), p(mixed), opt(post), p(ws), st()) }, "hc_mix");
}
pub fn hc_apply(x: &Tensor, post: &Tensor, y: &Tensor, r: i64) {
    ok(unsafe { qwen_hc_apply(p(x), p(post), p(y), r as i32, st()) }, "hc_apply");
}
pub fn ple_decode(packed: &Tensor, words: i64, bits: i32, head_bias: &Tensor, emb: &Tensor, r: i64) {
    ok(unsafe { qwen_ple_decode(p(packed), words as i32, bits, p(head_bias), p(emb), r as i32, st()) }, "ple_decode");
}
#[allow(clippy::too_many_arguments)]
pub fn ple_apply(x: &Tensor, key: &Tensor, value: &Tensor, nk1: &Tensor, nq1: &Tensor, nc1: &Tensor, conv: &Tensor, win: &Tensor,
                 gated: &Tensor, nrm: &Tensor, r: i64) {
    ok(unsafe { qwen_ple_apply(p(x), p(key), p(value), p(nk1), p(nq1), p(nc1), p(conv), p(win), p(gated), p(nrm), r as i32, st()) }, "ple_apply");
}
pub fn ple_commit(win: &Tensor, nrm: &Tensor, n: i64, nd: Option<&Tensor>) {
    ok(unsafe { qwen_ple_commit(p(win), p(nrm), n as i32, opt(nd), st()) }, "ple_commit");
}
pub fn gdn_conv_segs(x: &Tensor, sg: &Segs, w: &Tensor, y: &Tensor) {
    ok(unsafe { qwen_gdn_conv_segs(p(x), x.stride()[0], sg.n(), sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), p(w), p(y), st()) }, "gdn_conv_segs");
}
pub fn gdn_conv_commit_segs(x: &Tensor, sg: &Segs, nd: Option<&Tensor>) {
    ok(unsafe { qwen_gdn_conv_commit_segs(p(x), x.stride()[0], sg.n(), sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), opt(nd), st()) },
       "gdn_conv_commit_segs");
}
#[allow(clippy::too_many_arguments)]
pub fn gdn_recur_segs(y: &Tensor, ab: &Tensor, z: &Tensor, a_log: &Tensor, dt_bias: &Tensor, norm: &Tensor, sg: &Segs,
                      nd: Option<&Tensor>, out: Option<&Tensor>, write: bool) {
    let rows = sg.r0.iter().zip(&sg.t).map(|(a, b)| a + b).max().unwrap_or(0);
    let ws = Tensor::empty([unsafe { qwen_gdn_ws_bytes(rows) } as i64], (tch::Kind::Uint8, y.device()));
    ok(unsafe { qwen_gdn_recur_segs(p(y), p(ab), ab.stride()[0], p(z), z.stride()[0], p(a_log), p(dt_bias), p(norm), sg.n(),
                                    sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), opt(nd), opt(out), write as i32, p(&ws), st()) }, "gdn_recur_segs");
}

pub fn f16_ws_bytes(m: i64, n: i64, k: i64) -> i64 { unsafe { qwen_f16_ws_bytes(m as i32, n as i32, k as i32) as i64 } }
/// f16_gemv with the K-slice sum in the gemv's last blocks (cnt: [ceil(n / 4)] ints, zero; bitwise equal).
#[allow(clippy::too_many_arguments)]
pub fn f16_gemv_c(x: &Tensor, m: i64, w: &Tensor, k: i64, n: i64, y: &Tensor, ws: &Tensor, cnt: &Tensor) {
    extern "C" { fn qwen_f16_gemv_c(x: *const c_void, ldx: i64, m: i32, w: *const c_void, k: i32, n: i32, y: *mut c_void, ldy: i64, ws: *mut c_void, cnt: *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_f16_gemv_c(p(x), x.stride()[0], m as i32, p(w), k as i32, n as i32, p(y), y.stride()[0], p(ws), p(cnt), st()) }, "f16_gemv_c");
}
pub fn moe_route_ld(logits: &Tensor, r: i64, idx: &Tensor, w: &Tensor) {
    extern "C" { fn qwen_moe_route_ld(l: *const c_void, ld: i64, r: i32, idx: *mut c_void, w: *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_moe_route_ld(p(logits), logits.stride()[0], r as i32, p(idx), p(w), st()) }, "moe_route_ld");
}
pub fn moe_silu_mul(g: &Tensor, u: &Tensor, h: &Tensor) {
    extern "C" { fn qwen_moe_silu_mul(g: *const c_void, u: *const c_void, h: *mut c_void, n: i32, st: St) -> i32; }
    ok(unsafe { qwen_moe_silu_mul(p(g), p(u), p(h), g.numel() as i32, st()) }, "moe_silu_mul");
}
pub fn moe_gate_scale(gl: &Tensor, col: i64, d: &Tensor, out: &Tensor) {
    extern "C" { fn qwen_moe_gate_scale(gl: *const c_void, ldg: i64, col: i32, d: *const c_void, out: *mut c_void, r: i32, n: i32, st: St) -> i32; }
    ok(unsafe { qwen_moe_gate_scale(p(gl), gl.stride()[0], col as i32, p(d), p(out), d.size()[0] as i32, d.size()[1] as i32, st()) }, "moe_gate_scale");
}
/// X += post * y, then n = RMSNorm per stream (fp32), decode rows (bitwise equal to hc_apply + hc_norm).
pub fn hc_apply_norm(x: &Tensor, post: &Tensor, y: &Tensor, w1: &Tensor, n: &Tensor, r: i64) {
    extern "C" { fn qwen_hc_apply_norm(x: *mut c_void, post: *const c_void, y: *const c_void, w1: *const c_void, n: *mut c_void, r: i32, st: St) -> i32; }
    ok(unsafe { qwen_hc_apply_norm(p(x), p(post), p(y), p(w1), p(n), r as i32, st()) }, "hc_apply_norm");
}
/// One linear of an `exl3_multi` call: (suh, trellis, svh, bits, K slices, xh scratch [M, K] fp16, part scratch [S, M, N], y [M, N]).
pub struct MultiLin<'a> { pub suh: &'a Tensor, pub tr: &'a Tensor, pub svh: &'a Tensor, pub bits: i32, pub s: i32, pub xh: Tensor, pub part: Tensor, pub y: Tensor }
/// Up to 4 decode linears on one input in three launches (per linear bitwise equal to had_in + gemv + finish). The input is
/// x [M, K] fp32, or with `mix` = (g, n) the HC mix of g / n [M, 4K] computed here and written to `xout` [M, K].
/// flags: 1 = silu (two linears: y of the first gets silu(first) * second), 2 = one gemv launch per linear.
pub fn exl3_multi(x: Option<&Tensor>, mix: Option<(&Tensor, &Tensor)>, xout: Option<&Tensor>, m: i64, k: i64, lins: &[MultiLin], flags: i32) {
    extern "C" { fn qwen_exl3_multi(x: *const c_void, ldx: i64, mg: *const c_void, mn: *const c_void, xout: *mut c_void, m: i32, k: i32, n: i32,
                                    suh: *const *const c_void, xh: *const *mut c_void, tr: *const *const c_void, part: *const *mut c_void,
                                    nn: *const i32, s: *const i32, svh: *const *const c_void, y: *const *mut c_void, ldy: *const i64,
                                    bits: *const i32, flags: i32, st: St) -> i32; }
    let suh: Vec<*const c_void> = lins.iter().map(|l| p(l.suh) as *const c_void).collect();
    let xh: Vec<*mut c_void> = lins.iter().map(|l| p(&l.xh)).collect();
    let tr: Vec<*const c_void> = lins.iter().map(|l| p(l.tr) as *const c_void).collect();
    let part: Vec<*mut c_void> = lins.iter().map(|l| p(&l.part)).collect();
    let nn: Vec<i32> = lins.iter().map(|l| l.y.size()[1] as i32).collect();
    let ss: Vec<i32> = lins.iter().map(|l| l.s).collect();
    let svh: Vec<*const c_void> = lins.iter().map(|l| p(l.svh) as *const c_void).collect();
    let y: Vec<*mut c_void> = lins.iter().map(|l| p(&l.y)).collect();
    let ldy: Vec<i64> = lins.iter().map(|l| l.y.stride()[0]).collect();
    let bits: Vec<i32> = lins.iter().map(|l| l.bits).collect();
    let (xp, ldx) = x.map_or((std::ptr::null_mut(), 0), |t| (p(t), t.stride()[0]));
    let (mg, mn) = mix.map_or((std::ptr::null_mut(), std::ptr::null_mut()), |(g, n)| (p(g), p(n)));
    ok(unsafe { qwen_exl3_multi(xp, ldx, mg, mn, opt(xout), m as i32, k as i32, lins.len() as i32, suh.as_ptr(), xh.as_ptr(), tr.as_ptr(),
                                part.as_ptr(), nn.as_ptr(), ss.as_ptr(), svh.as_ptr(), y.as_ptr(), ldy.as_ptr(), bits.as_ptr(), flags, st()) }, "exl3_multi");
}
/// had_in of several linears from one read of x [M, K] fp32: xh[i] [M, K] fp16 with suh[i] (bitwise equal to exl3_had_in each).
pub fn exl3_had_multi(x: &Tensor, suh: &[&Tensor], xh: &[&Tensor]) {
    extern "C" { fn qwen_exl3_had_multi(x: *const c_void, ldx: i64, m: i32, k: i32, n: i32, suh: *const *const c_void, xh: *const *mut c_void, st: St) -> i32; }
    let s: Vec<*const c_void> = suh.iter().map(|t| p(t) as *const c_void).collect();
    let h: Vec<*mut c_void> = xh.iter().map(|t| p(t)).collect();
    ok(unsafe { qwen_exl3_had_multi(p(x), x.stride()[0], x.size()[0] as i32, x.size()[1] as i32, suh.len() as i32, s.as_ptr(), h.as_ptr(), st()) }, "exl3_had_multi");
}
/// Prefill GDN conv / conv commit on the unfinished qkv projection (fp16 GEMM output `part` and its svh).
pub fn gdn_conv_h(part: &Tensor, svh: &Tensor, sg: &Segs, w: &Tensor, y: &Tensor) {
    extern "C" { fn qwen_gdn_conv_h(part: *const c_void, ldp: i64, svh: *const c_void, nseg: i32, r0: *const i32, t: *const i32, state: *const *mut c_void,
                                    w: *const c_void, y: *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_gdn_conv_h(p(part), part.stride()[0], p(svh), sg.n(), sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), p(w), p(y), st()) }, "gdn_conv_h");
}
pub fn gdn_conv_commit_h(part: &Tensor, svh: &Tensor, sg: &Segs) {
    extern "C" { fn qwen_gdn_conv_commit_h(part: *const c_void, ldp: i64, svh: *const c_void, nseg: i32, r0: *const i32, t: *const i32,
                                           state: *const *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_gdn_conv_commit_h(p(part), part.stride()[0], p(svh), sg.n(), sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), st()) }, "gdn_conv_commit_h");
}
/// gdn_conv_h with the recurrence's prep fused (q / k normalized into ws, decay / beta too; y gets v only), then
/// gdn_recur_segs_zq on the same ws: prefill chunks (> 16 rows) only, bitwise equal to conv + prep.
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv_hq(part: &Tensor, svh: &Tensor, sg: &Segs, w: &Tensor, y: &Tensor, ab: &Tensor, a_log: &Tensor, dt_bias: &Tensor, ws: &Tensor) {
    extern "C" { fn qwen_gdn_conv_hq(part: *const c_void, ldp: i64, svh: *const c_void, nseg: i32, r0: *const i32, t: *const i32, state: *const *mut c_void,
                                     w: *const c_void, y: *mut c_void, ab: *const c_void, ldab: i64, a_log: *const c_void, dt_bias: *const c_void,
                                     ws: *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_gdn_conv_hq(p(part), part.stride()[0], p(svh), sg.n(), sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), p(w), p(y), p(ab),
                                 ab.stride()[0], p(a_log), p(dt_bias), p(ws), st()) }, "gdn_conv_hq");
}
#[allow(clippy::too_many_arguments)]
pub fn gdn_recur_segs_zq(y: &Tensor, ab: &Tensor, zp: &Tensor, z_svh: &Tensor, a_log: &Tensor, dt_bias: &Tensor, norm: &Tensor, sg: &Segs,
                         out: &Tensor, write: bool, ws: &Tensor) {
    extern "C" { fn qwen_gdn_recur_segs_zq(y: *const c_void, ab: *const c_void, ldab: i64, z: *const c_void, ldz: i64, z_svh: *const c_void,
                                           a_log: *const c_void, dt_bias: *const c_void, norm_w: *const c_void, nseg: i32, r0: *const i32, t: *const i32,
                                           s: *const *mut c_void, out: *mut c_void, write: i32, ws: *mut c_void, st: St) -> i32; }
    ok(unsafe { qwen_gdn_recur_segs_zq(p(y), p(ab), ab.stride()[0], p(zp), zp.stride()[0], p(z_svh), p(a_log), p(dt_bias), p(norm), sg.n(),
                                       sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), p(out), write as i32, p(ws), st()) }, "gdn_recur_segs_zq");
}
pub fn gdn_ws_bytes(r: i64) -> i64 { unsafe { qwen_gdn_ws_bytes(r as i32) as i64 } }
/// gdn_recur_segs with z the unfinished fp16 projection part (its svh): prefill chunks (> 16 rows) only.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recur_segs_z(y: &Tensor, ab: &Tensor, zp: &Tensor, z_svh: &Tensor, a_log: &Tensor, dt_bias: &Tensor, norm: &Tensor, sg: &Segs,
                        out: &Tensor, write: bool) {
    extern "C" { fn qwen_gdn_recur_segs_z(y: *const c_void, ab: *const c_void, ldab: i64, z: *const c_void, ldz: i64, z_svh: *const c_void,
                                          a_log: *const c_void, dt_bias: *const c_void, norm_w: *const c_void, nseg: i32, r0: *const i32, t: *const i32,
                                          s: *const *mut c_void, nd: *const c_void, out: *mut c_void, write: i32, ws: *mut c_void, st: St) -> i32; }
    let rows = sg.r0.iter().zip(&sg.t).map(|(a, b)| a + b).max().unwrap_or(0);
    let ws = Tensor::empty([unsafe { qwen_gdn_ws_bytes(rows) } as i64], (tch::Kind::Uint8, y.device()));
    ok(unsafe { qwen_gdn_recur_segs_z(p(y), p(ab), ab.stride()[0], p(zp), zp.stride()[0], p(z_svh), p(a_log), p(dt_bias), p(norm), sg.n(),
                                      sg.r0.as_ptr(), sg.t.as_ptr(), sg.s.as_ptr(), std::ptr::null_mut(), p(out), write as i32, p(&ws), st()) }, "gdn_recur_segs_z");
}
/// qsa_prep with q from q_proj's unfinished fp16 output (row stride of `qpart`) and its svh (prefill).
#[allow(clippy::too_many_arguments)]
pub fn qsa_prep_h(qpart: &Tensor, qsvh: &Tensor, kp: &Tensor, vp: &Tensor, ip: &Tensor, r: i64, pos0: &Tensor, qn: &Tensor, kn: &Tensor, iqn: &Tensor,
                  q: &Tensor, gate: &Tensor, kc: &Tensor, ks: &Tensor, vc: &Tensor, vs: &Tensor, qi: &Tensor, ring: &Tensor, yarn: i32,
                  mtab: &Tensor, mctl: &Tensor) {
    extern "C" { fn qwen_qsa_prep_hm(qpart: *const c_void, ldqp: i64, qsvh: *const c_void, kp: *const c_void, vp: *const c_void, ip: *const c_void, r: i32,
                                     pos0: *const c_void, qn: *const c_void, kn: *const c_void, iqn: *const c_void, q: *mut c_void, gate: *mut c_void,
                                     kc: *mut c_void, ksc: *mut c_void, vc: *mut c_void, vsc: *mut c_void, qi: *mut c_void, ring: *mut c_void,
                                     yarn: i32, mtab: *const c_void, mctl: *const c_void, st: St) -> i32; }
    ok(unsafe { qwen_qsa_prep_hm(p(qpart), qpart.stride()[0], p(qsvh), p(kp), p(vp), p(ip), r as i32, p(pos0), p(qn), p(kn), p(iqn), p(q), p(gate),
                                 p(kc), p(ks), p(vc), p(vs), p(qi), p(ring), yarn, p(mtab), p(mctl), st()) }, "qsa_prep_h");
}
pub fn f16_gemv(x: &Tensor, m: i64, w: &Tensor, k: i64, n: i64, y: &Tensor, ws: &Tensor) {
    ok(unsafe { qwen_f16_gemv(p(x), x.stride()[0], m as i32, p(w), k as i32, n as i32, p(y), y.stride()[0], p(ws), st()) }, "f16_gemv");
}

pub fn hc_norm(x: &Tensor, r: i64, w1: &Tensor, n: &Tensor) { ok(unsafe { qwen_hc_norm(p(x), r as i32, p(w1), p(n), st()) }, "hc_norm"); }
pub fn hc_mid2(d: &Tensor, t: &Tensor, post: Option<&Tensor>, r: i64, inject: bool) {
    ok(unsafe { qwen_hc_mid2(p(d), d.stride()[0] as i32, p(t), opt(post), r as i32, inject as i32, st()) }, "hc_mid2");
}
pub fn hc_mix2(g: &Tensor, n: &Tensor, mixed: &Tensor, r: i64) { ok(unsafe { qwen_hc_mix2(p(g), p(n), p(mixed), r as i32, st()) }, "hc_mix2"); }
pub fn hc_norm_h(x: &Tensor, r: i64, w1: &Tensor, n: &Tensor, scale: &Tensor) {
    ok(unsafe { qwen_hc_norm_h(p(x), r as i32, p(w1), p(n), p(scale), st()) }, "hc_norm_h");
}
pub fn hc_mid2h(d: &Tensor, t: &Tensor, post: Option<&Tensor>, r: i64, inject: bool) {
    ok(unsafe { qwen_hc_mid2h(p(d), d.stride()[0] as i32, p(t), opt(post), r as i32, inject as i32, st()) }, "hc_mid2h");
}
pub fn hc_upmix(t: &Tensor, up: &Tensor, x: &Tensor, scale: &Tensor, w1: &Tensor, mixed: &Tensor, r: i64) {
    extern "C" { fn qwen_hc_upmix(t: *const c_void, up: *const c_void, x: *const c_void, scale: *const c_void, w1: *const c_void, mixed: *mut c_void, r: i32, st: St) -> i32; }
    ok(unsafe { qwen_hc_upmix(p(t), p(up), p(x), p(scale), p(w1), p(mixed), r as i32, st()) }, "hc_upmix");
}
pub fn hc_mix2h(g: &Tensor, x: &Tensor, scale: &Tensor, w1: &Tensor, mixed: &Tensor, r: i64) {
    ok(unsafe { qwen_hc_mix2h(p(g), p(x), p(scale), p(w1), p(mixed), r as i32, st()) }, "hc_mix2h");
}
pub fn exl3_finish_h(part: &Tensor, svh: &Tensor, y: &Tensor, m: i64, n: i64) {
    ok(unsafe { qwen_exl3_finish_h(p(part), p(svh), p(y), y.stride()[0], m as i32, n as i32, st()) }, "exl3_finish_h");
}
pub fn q8_encode(w: &Tensor, n: i64, k: i64, q: &Tensor, s: &Tensor) {
    ok(unsafe { qwen_q8_encode(p(w), n as i32, k as i32, p(q), p(s), st()) }, "q8_encode");
}
pub fn q8_ws_bytes(m: i64, n: i64, k: i64) -> i64 { unsafe { qwen_q8_ws_bytes(m as i32, n as i32, k as i32) as i64 } }
pub fn q8_gemv(x: &Tensor, m: i64, q: &Tensor, s: &Tensor, k: i64, n: i64, y: &Tensor, ws: &Tensor) {
    ok(unsafe { qwen_q8_gemv(p(x), x.stride()[0], m as i32, p(q), p(s), k as i32, n as i32, p(y), y.stride()[0], p(ws), st()) }, "q8_gemv");
}
pub fn hc_apply_norm_h(x: &Tensor, post: &Tensor, y: &Tensor, w1: &Tensor, n: &Tensor, scale: &Tensor, r: i64) {
    ok(unsafe { qwen_hc_apply_norm_h(p(x), p(post), p(y), p(w1), p(n), p(scale), r as i32, st()) }, "hc_apply_norm_h");
}
