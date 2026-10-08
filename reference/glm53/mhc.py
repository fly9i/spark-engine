# SPDX-License-Identifier: MIT
"""MHC(多残差流超连接)参考实现。数学对齐 vllm mhc kernels/torch.py
(检查点约定:sinkhorn 列/行归一序、post_mult=2.0、eps=1e-6、repeat=20)。

残差流 [T, n, H](n=4)。每层两次 hc_pre(attn/ffn)+ 一次延后的 hc_post。
"""
from __future__ import annotations

import torch

HC_EPS = 1e-6
SINKHORN_REPEAT = 20
POST_MULT = 2.0


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[T,H] → [T,n,H](复制)。"""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[T,n,H] → [T,H](平均)。"""
    return x.mean(dim=1)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    v = x.float()
    return weight.float() * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps))


def mhc_pre(residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
            base: torch.Tensor, norm_weight: torch.Tensor, rms_eps: float = 1e-5):
    """返回 (post_mix [T,n,1], comb_mix [T,n,n], layer_input [T,H] 已过 RMSNorm)。

    residual [T,n,H];fn [n²+2n, n·H];scale [3];base [n²+2n]。
    """
    T, n, H = residual.shape
    x = residual.reshape(T, n * H).float()
    mixes = x @ fn.t()
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (n * H) + rms_eps)
    pre_mix = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + HC_EPS
    post_mix = torch.sigmoid(mixes[:, n:2 * n] * scale[1] + base[n:2 * n]) * POST_MULT
    comb = torch.softmax(
        mixes[:, 2 * n:].view(T, n, n) * scale[2] + base[2 * n:].view(1, n, n), dim=-1
    ) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_REPEAT - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    layer_input = (pre_mix.unsqueeze(-1) * residual.float()).sum(1)
    layer_input = rmsnorm(layer_input, norm_weight, rms_eps)
    return post_mix.unsqueeze(-1), comb, layer_input


def mhc_post(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor,
             comb_mix: torch.Tensor) -> torch.Tensor:
    """新残差流:res'[j] = Σ_i comb[i,j]·res[i] + post[j]·x。x [T,H]。"""
    mixed = torch.einsum("tij,tih->tjh", comb_mix.float(), residual.float())
    return mixed + post_mix.float() * x.float().unsqueeze(-2)
