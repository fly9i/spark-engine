# SPDX-License-Identifier: MIT
"""Pure-torch stand-in for DeepSeek-V4.1-Flash `inference/kernel.py` (tilelang), same numerics.

Imported as `kernel` ahead of the official module (see export_engram.py), so the official `model.py` runs
unchanged on GB10 without tilelang. Each function follows the tilelang kernel it replaces:
- quantization: FP32 compute, amax floors, power-of-two (UE8M0) scale rounding by exponent bits, saturating
  round-to-nearest-even casts;
- GEMMs: FP8/FP4 operands times power-of-two block scales are exact in FP32, so dequantize-then-FP32-matmul
  equals the blockwise kernel up to the summation order; the output is cast to the default dtype (BF16);
- sparse attention: FP32 scores, probabilities rounded to BF16 before the PV product (the kernel's acc_s_cast),
  sink added to the denominator only, rows without any valid index give zeros;
- hc_split_sinkhorn: the same operation order in FP32.
"""
from typing import Optional, Tuple

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# E2M1 magnitudes by code (bit 3 is the sign), the table convert.py uses.
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for positive normal FP32 x from the exponent bits (the kernel's fast_log2_ceil)."""
    bits = x.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    return exp - 127 + (man != 0).to(torch.int32)


def _pow2(e: torch.Tensor) -> torch.Tensor:
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def _round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    return _pow2(_log2_ceil(amax * torch.tensor(max_inv, dtype=torch.float32, device=amax.device)))


def _to_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Round FP32 values already clamped to [-6, 6] to the E2M1 grid, nearest with ties to even code."""
    a = y.abs()
    # Midpoints between consecutive magnitudes and which side a tie goes to (the even code).
    q = torch.zeros_like(a)
    q = torch.where(a > 0.25, 0.5, q)
    q = torch.where(a >= 0.75, 1.0, q)
    q = torch.where(a > 1.25, 1.5, q)
    q = torch.where(a >= 1.75, 2.0, q)
    q = torch.where(a > 2.5, 3.0, q)
    q = torch.where(a >= 3.5, 4.0, q)
    q = torch.where(a > 5.0, 6.0, q)
    return torch.copysign(q, y)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block))


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    """Block-wise FP8 E4M3 quantization along the last dim; inplace=True writes quant+dequant back to x."""
    assert x.size(-1) % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _round_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)
    y = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    """FP4 E2M1 with E8M0 (indexer) or E4M3 (compressed KV) scales. Only the inplace (fake-quant) form is used."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert x.size(-1) % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _round_scale(amax, 1.0 / FP4_MAX)
    y = _to_e2m1((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((y * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    raise NotImplementedError("packed FP4 output is not needed by the reference")


def _dequant_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block_size: int) -> torch.Tensor:
    n, k = b.shape
    s = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    return b.float() * s


import os
# REF_GEMM=bf16: operands (exact in BF16) multiplied by a BF16 cuBLAS GEMM (FP32 accumulation, another summation order)
# instead of an FP32 matmul -- a second, equally valid implementation, to measure the model's noise floor.
_BF16_GEMM = os.environ.get("REF_GEMM") == "bf16"


def _mm(af, bf):
    if _BF16_GEMM:
        return torch.matmul(af.bfloat16(), bf.bfloat16().t())
    return torch.matmul(af, bf.t())


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size: int = 128):
    """C = A @ B^T with A FP8 [.., K] (scale per row and K block) and B FP8 [N, K] (scale per N×K block)."""
    k = a.size(-1)
    af = (a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    bf = _dequant_fp8_weight(b, b_s, block_size)
    return _mm(af.reshape(-1, k), bf).reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed E2M1 (low nibble first) with one E8M0 scale per 32 along K -> FP32 [N, K]."""
    u = b.view(torch.uint8)
    table = FP4_TABLE.to(u.device)
    lo, hi = u & 0x0F, u >> 4
    def val(c):
        v = table[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    w = torch.stack([val(lo), val(hi)], dim=-1).flatten(-2)
    return (w.unflatten(-1, (-1, 32)) * b_s.float().unsqueeze(-1)).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size: int = 128):
    """C = A_fp8 @ B_fp4^T; A has one scale per act_block_size along K, B one E8M0 scale per 32."""
    k = a.size(-1)
    af = (a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    bf = dequant_fp4_weight(b, b_s)
    return _mm(af.reshape(-1, k), bf).reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float) -> torch.Tensor:
    """q [b,m,h,d] BF16, kv [b,n,d] BF16 (keys and values are the same vector), topk_idxs [b,m,k] (-1 = none)."""
    b, m, h, d = q.shape
    topk_idxs = topk_idxs.to(q.device)
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    out = torch.empty_like(q)
    # Chunk the queries: the gathered keys are [m, k, d].
    step = max(1, (1 << 27) // max(1, topk_idxs.size(-1) * d))
    for bi in range(b):
        for m0 in range(0, m, step):
            m1 = min(m, m0 + step)
            kk = kv[bi][idx[bi, m0:m1]].float()                      # [mc, k, d]
            ok = valid[bi, m0:m1]                                    # [mc, k]
            s = torch.einsum("mhd,mkd->mhk", q[bi, m0:m1].float(), kk) * softmax_scale
            s = s.masked_fill(~ok.unsqueeze(1), float("-inf"))
            mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(s - mx)
            denom = p.sum(-1) + torch.exp(attn_sink.float().view(1, h) - mx.squeeze(-1))
            o = torch.einsum("mhk,mkd->mhd", p.to(torch.bfloat16).float(), kk)
            out[bi, m0:m1] = (o / denom.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc = hc_mult
    m = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
