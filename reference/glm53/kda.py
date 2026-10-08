# SPDX-License-Identifier: MIT
"""KDA(gated delta net,有界安全门)参考实现 — 2026-09-21 对齐 GPU 内核验证。

已验证语义(T=24 随机全维,vs chunk_kda_with_fused_gate 相对误差 6e-6):
  - 状态 h [H, Dv, Dk];衰减 exp(λ) 乘在 K 列;
  - λ = lb / (1 + exp(-exp(A_log)·(g1 + dt_bias))),lb=-5,g1 = f_b(f_a(x)),dt_bias[H,D];
  - β = sigmoid(b_proj(x))(sigmoid 在层代码、内核调用之前——2026-09-21 修正);
  - 递归:h·k 校正:h += β(v − h·k) ⊗ kn;o = h · (l2norm(q)·scale),scale = D^-0.5;
  - q/k/v 先过因果短卷积(k=4,depthwise,y[t]=Σ_j w[j]·x[t-3+j] **+ silu**)再进 KDA
    (q/k 追加 l2norm eps 1e-6,q 再乘 scale);
  - 输出 o_norm:RMSNorm(D) × sigmoid(g2),g2 = g_b(g_a(x))。
"""
from __future__ import annotations

import numpy as np
import torch


def causal_conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [T, C],w [C, K] 或 [C, 1, K] → y [T, C]。左 pad K-1,无偏置。"""
    K = w.shape[-1]
    if w.dim() == 3:
        w = w[:, 0, :]
    xp = torch.cat([torch.zeros(K - 1, x.shape[1], dtype=x.dtype, device=x.device), x], dim=0)
    y = torch.zeros_like(x)
    for j in range(K):
        y += w[:, j] * xp[j:j + x.shape[0]]
    return y


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


class KdaLayer:
    def __init__(self, idx, layer: int, H: int = 64, D: int = 128, lb: float = -5.0):
        p = f"model.language_model.layers.{layer}.self_attn"
        from .nn import bf16_f32, f32_param
        bf = lambda n: bf16_f32(idx, f"{p}.{n}.weight")
        self.wq, self.wk, self.wv = bf("q_proj"), bf("k_proj"), bf("v_proj")
        self.wb = bf("b_proj")                          # [H, in]
        self.fa, self.fb = bf("f_a_proj"), bf("f_b_proj")
        self.ga, self.gb = bf("g_a_proj"), bf("g_b_proj")
        self.wo = bf("o_proj")                          # [in, 8192]
        cv = lambda n: bf16_f32(idx, f"{p}.{n}.weight").squeeze(1)
        self.conv_q, self.conv_k, self.conv_v = cv("q_conv1d"), cv("k_conv1d"), cv("v_conv1d")
        self.dt_bias = f32_param(idx, f"{p}.dt_bias").reshape(H, D)
        self.A_log = f32_param(idx, f"{p}.A_log").reshape(H)
        self.o_norm_w = bf16_f32(idx, f"{p}.o_norm.weight")
        self.H, self.D, self.lb = H, D, lb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, 4096] fp32 → [T, 4096]。逐 token 递归(正确性参考)。"""
        T = x.shape[0]
        H, D = self.H, self.D
        sc = D ** -0.5
        # 卷积带 silu(生产 activation="silu",q/k/v 全部)
        q = _l2(torch.nn.functional.silu(causal_conv1d(x @ self.wq.T, self.conv_q)).view(T, H, D)) * sc
        k = _l2(torch.nn.functional.silu(causal_conv1d(x @ self.wk.T, self.conv_k)).view(T, H, D))
        v = torch.nn.functional.silu(causal_conv1d(x @ self.wv.T, self.conv_v)).view(T, H, D)
        beta = torch.sigmoid(x @ self.wb.T)                    # [T,H] 层内预计算 sigmoid
        g1 = ((x @ self.fa.T) @ self.fb.T).view(T, H, D)
        lam = self.lb / (1.0 + torch.exp(
            -(torch.exp(self.A_log)[None, :, None] * (g1 + self.dt_bias[None]))))
        g2 = ((x @ self.ga.T) @ self.gb.T).view(T, H, D)

        h = torch.zeros(H, D, D, device=q.device)                              # [H, Dv, Dk]
        outs = []
        for t in range(T):
            h = h * torch.exp(lam[t])[:, None, :]              # 衰减乘 K 列
            hk = (h @ k[t].unsqueeze(-1)).squeeze(-1)          # [H,Dv]
            h = h + (beta[t][:, None] * (v[t] - hk))[:, :, None] * k[t][:, None, :]
            outs.append((h @ q[t].unsqueeze(-1)).squeeze(-1))  # [H,Dv]
        o = torch.stack(outs)                                  # [T,H,D]
        on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)
        o = self.o_norm_w * on * torch.sigmoid(g2)
        return o.reshape(T, H * D) @ self.wo.T
