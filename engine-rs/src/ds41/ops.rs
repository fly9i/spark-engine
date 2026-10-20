//! Numeric primitives of DeepSeek-V4.1-Flash, each matching the official `inference/kernel.py` / `model.py` operation
//! (and reference/ds41/kernel.py, the golden reference): FP8/FP4 fake quantization with the same scale rules, RMSNorm,
//! RoPE with YaRN, hyper-connection Sinkhorn, sparse attention with sink. v0 builds them from libtorch ops; D4 replaces
//! the hot ones with fused kernels checked against these.
use tch::{Kind, Tensor};

/// 2^e for an Int32 tensor of exponents, built from the bits (exact).
pub fn pow2(e: &Tensor) -> Tensor {
    (e.to_kind(Kind::Int) + 127).bitwise_left_shift_tensor_scalar(23).view_dtype(Kind::Float)
}
/// ceil(log2(x)) of positive normal FP32 x from the exponent bits (kernel.py fast_log2_ceil).
pub fn log2_ceil(x: &Tensor) -> Tensor {
    let bits = x.to_kind(Kind::Float).contiguous().view_dtype(Kind::Int);
    let exp = bits.bitwise_right_shift_tensor_scalar(23).bitwise_and(0xFF);
    let man = bits.bitwise_and((1i64 << 23) - 1).ne(0).to_kind(Kind::Int);
    exp - 127 + man
}
/// UE8M0 scale of a block: 2^ceil(log2(amax * max_inv)).
fn round_scale(amax: &Tensor, max_inv: f32) -> Tensor { pow2(&log2_ceil(&(amax * max_inv as f64))) }

/// E8M0 scale bytes (Uint8) -> FP32 2^(b-127).
pub fn e8m0(s: &Tensor) -> Tensor { pow2(&(s.to_kind(Kind::Int) - 127)) }

/// FP8 E4M3 activation quantization with UE8M0 scales per `block` along the last dim, returned dequantized in the
/// input dtype (act_quant inplace / the operand an FP8 GEMM sees).
pub fn act_quant(x: &Tensor, block: i64) -> Tensor {
    let shape = x.size();
    let n = *shape.last().unwrap();
    assert_eq!(n % block, 0);
    let xb = x.to_kind(Kind::Float).reshape([-1, n / block, block]);
    let amax = xb.abs().amax([-1].as_slice(), true).clamp_min(1e-4);
    let s = round_scale(&amax, 1.0 / 448.0);
    let q = (&xb / &s).clamp(-448.0, 448.0).to_kind(Kind::Float8e4m3fn).to_kind(Kind::Float);
    (q * s).reshape(shape.as_slice()).to_kind(x.kind())
}

/// Round FP32 values already in [-6, 6] to the E2M1 grid, nearest with ties to the even code.
fn to_e2m1(y: &Tensor) -> Tensor {
    let a = y.abs();
    let mut q = a.zeros_like();
    for (gt, thr, val) in [(true, 0.25, 0.5), (false, 0.75, 1.0), (true, 1.25, 1.5), (false, 1.75, 2.0), (true, 2.5, 3.0), (false, 3.5, 4.0), (true, 5.0, 6.0)] {
        let m = if gt { a.gt(thr) } else { a.ge(thr) };
        q = q.where_self(&m.logical_not(), &Tensor::from(val as f32).to_device(a.device()));
    }
    q.copysign(y)
}
/// FP4 E2M1 fake quantization per `block`: `e4m3_scale` false = UE8M0 scales (indexer), true = E4M3 scales (compressed
/// KV, groups of 16). Returned dequantized in the input dtype.
pub fn fp4_quant(x: &Tensor, block: i64, e4m3_scale: bool) -> Tensor {
    let shape = x.size();
    let n = *shape.last().unwrap();
    let xb = x.to_kind(Kind::Float).reshape([-1, n / block, block]);
    let amax = xb.abs().amax([-1].as_slice(), true);
    let s = if e4m3_scale {
        (amax.clamp_min(6.0 * 2f64.powi(-9)) / 6.0).to_kind(Kind::Float8e4m3fn).to_kind(Kind::Float)
    } else {
        round_scale(&amax.clamp_min(6.0 * 2f64.powi(-126)), 1.0 / 6.0)
    };
    let q = to_e2m1(&(&xb / &s).clamp(-6.0, 6.0));
    (q * s).reshape(shape.as_slice()).to_kind(x.kind())
}

/// RMSNorm in FP32, result in the input dtype (model.py RMSNorm).
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Tensor {
    let xf = x.to_kind(Kind::Float);
    let var = xf.square().mean_dim([-1].as_slice(), true, Kind::Float);
    (w.to_kind(Kind::Float) * (xf * (var + eps).rsqrt())).to_kind(x.kind())
}

/// FP8 linear: y = act_quant(x) @ W^T with W already dequantized to BF16 (exact), BF16 output with FP32 accumulation.
pub fn linear_fp8(x: &Tensor, w: &Tensor) -> Tensor {
    act_quant(&x.to_kind(Kind::BFloat16), 32).matmul(&w.tr())
}

/// Rotary frequencies (model.py precompute_freqs_cis) for `dim` rotary dims at the given positions: (cos, sin) [n, dim/2].
pub struct Rope { freqs: Tensor }
impl Rope {
    pub fn new(dim: i64, original_seq_len: i64, base: f64, factor: f64, beta_fast: f64, beta_slow: f64) -> Rope {
        let half = (dim / 2) as usize;
        let mut freqs: Vec<f32> = (0..half).map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0f32 / (base as f32).powf(e)
        }).collect();
        if original_seq_len > 0 {
            let corrected = |rot: f64| dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
            let low = (corrected(beta_fast).floor()).max(0.0);
            let high = (corrected(beta_slow).ceil()).min(dim as f64 - 1.0);
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) / ((high - low) as f32).max(1e-3)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / factor as f32 * (1.0 - smooth) + *f * smooth;
            }
        }
        Rope { freqs: Tensor::from_slice(&freqs) }
    }
    /// The frequencies on `dev` (graph capture needs them there: no host copies inside a capture).
    pub fn to(self, dev: tch::Device) -> Rope { Rope { freqs: self.freqs.to_device(dev) } }
    pub fn freqs(&self) -> &Tensor { &self.freqs }
    /// (cos, sin) [n, dim/2] for positions `pos` (Int64, any device).
    pub fn at(&self, pos: &Tensor) -> (Tensor, Tensor) {
        let f = self.freqs.to_device(pos.device());
        let ang = pos.to_kind(Kind::Float).unsqueeze(-1) * f.unsqueeze(0);
        (ang.cos(), ang.sin())
    }
}
/// Rotate the last `rd` dims of x [n, ..., d] as adjacent complex pairs by the angles of row i (cos/sin [n, rd/2]);
/// `inverse` uses the conjugate. Computed in FP32, result in x's dtype (model.py apply_rotary_emb).
pub fn rope(x: &Tensor, cs: &(Tensor, Tensor), rd: i64, inverse: bool) -> Tensor {
    let d = *x.size().last().unwrap();
    let head = x.narrow(-1, 0, d - rd);
    let r = x.narrow(-1, d - rd, rd).to_kind(Kind::Float);
    let mut shape = r.size();
    let last = shape.pop().unwrap();
    shape.push(last / 2);
    shape.push(2);
    let r = r.reshape(shape.as_slice());
    let (re, im) = (r.select(-1, 0), r.select(-1, 1));
    let n = x.size()[0];
    let extra = x.dim() as i64 - 2;   // dims between the row and the rotary pairs (heads)
    let mut bshape = vec![n];
    bshape.extend(std::iter::repeat(1).take(extra as usize));
    bshape.push(rd / 2);
    let c = cs.0.view(bshape.as_slice());
    let s = if inverse { -cs.1.view(bshape.as_slice()) } else { cs.1.view(bshape.as_slice()) };
    let out_re = &re * &c - &im * &s;
    let out_im = &re * &s + &im * &c;
    let rot = Tensor::stack(&[out_re, out_im], -1).flatten(-2, -1).to_kind(x.kind());
    Tensor::cat(&[head, rot], -1)
}

/// Hyper-connection coefficients (kernel.py hc_split_sinkhorn): mixes [n, (2+hc)*hc] -> pre [n,hc], post [n,hc],
/// comb [n,hc,hc], all FP32.
pub fn hc_split_sinkhorn(mixes: &Tensor, scale: &Tensor, base: &Tensor, hc: i64, iters: i64, eps: f64) -> (Tensor, Tensor, Tensor) {
    let m = mixes.to_kind(Kind::Float);
    let (sc, b) = (scale.to_kind(Kind::Float), base.to_kind(Kind::Float));
    let pre = (m.narrow(-1, 0, hc) * sc.get(0) + b.narrow(0, 0, hc)).sigmoid() + eps;
    let post = (m.narrow(-1, hc, hc) * sc.get(1) + b.narrow(0, hc, hc)).sigmoid() * 2.0;
    let n = m.size()[0];
    let comb = (m.narrow(-1, 2 * hc, hc * hc) * sc.get(2) + b.narrow(0, 2 * hc, hc * hc)).view([n, hc, hc]);
    let comb = (&comb - comb.amax([-1].as_slice(), true)).exp();
    let mut comb = &comb / comb.sum_dim_intlist([-1].as_slice(), true, Kind::Float) + eps;
    comb = &comb / (comb.sum_dim_intlist([-2].as_slice(), true, Kind::Float) + eps);
    for _ in 0..iters - 1 {
        comb = &comb / (comb.sum_dim_intlist([-1].as_slice(), true, Kind::Float) + eps);
        comb = &comb / (comb.sum_dim_intlist([-2].as_slice(), true, Kind::Float) + eps);
    }
    (pre, post, comb)
}

/// Sparse attention (kernel.py sparse_attn): q [T, h, d] BF16, kv [N, d] BF16 (key and value are the same vector),
/// idx [T, K] Int64 (-1 = no position), sink [h] FP32. FP32 scores, probabilities rounded to BF16 before PV, sink in
/// the denominator only, rows without any position give zeros. Queries in chunks of `chunk`.
pub fn sparse_attn(q: &Tensor, kv: &Tensor, idx: &Tensor, sink: &Tensor, scale: f64, chunk: i64) -> Tensor {
    let (t, h, d) = (q.size()[0], q.size()[1], q.size()[2]);
    let mut outs = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let n = chunk.min(t - t0);
        let ix = idx.narrow(0, t0, n);
        let valid = ix.ge(0);
        let k = kv.index_select(0, &ix.clamp_min(0).view(-1)).view([n, -1, d]).to_kind(Kind::Float);   // [n, K, d]
        let s = q.narrow(0, t0, n).to_kind(Kind::Float).matmul(&k.transpose(1, 2)) * scale;           // [n, h, K]
        let s = s.masked_fill(&valid.unsqueeze(1).logical_not(), f64::NEG_INFINITY);
        let mx = s.amax([-1].as_slice(), true).clamp_min(-1e30);
        let p = (&s - &mx).exp();
        let denom = p.sum_dim_intlist([-1].as_slice(), false, Kind::Float) + (sink.to_kind(Kind::Float).view([1, h]) - mx.squeeze_dim(-1)).exp();
        let o = p.to_kind(Kind::BFloat16).to_kind(Kind::Float).matmul(&k);                               // [n, h, d]
        outs.push((o / denom.unsqueeze(-1)).to_kind(q.kind()));
        t0 += n;
    }
    Tensor::cat(&outs, 0)
}
