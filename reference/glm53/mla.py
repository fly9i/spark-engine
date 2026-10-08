# SPDX-License-Identifier: MIT
"""DSA 层 MLA 参考(无 rope:qk_rope_head_dim=0,mla_use_nope)。

权重(q_a/q_b、kv_a 512、kv_b [16384|16384] = w_kc|w_vo、o_proj)。
M0 正确性:全因果注意力(indexer topk=2048 ≥ 短 prompt 上下文时与稀疏选择等价)。
"""
from __future__ import annotations

import numpy as np
import torch

from .mhc import rmsnorm


class MlaLayer:
    def __init__(self, idx, layer: int, H: int = 64, dk: int = 256, dv: int = 256,
                 kv_lora: int = 512, q_lora: int = 1536):
        p = f"model.language_model.layers.{layer}.self_attn"
        from .nn import bf16_f32
        g = lambda n: bf16_f32(idx, f"{p}.{n}.weight")
        self.q_a, self.q_b = g("q_a_proj"), g("q_b_proj")       # [1536,4096],[16384,1536]
        self.kv_a = g("kv_a_proj_with_mqa")                     # [512,4096]
        self.kv_b = g("kv_b_proj")                              # [32768,512]
        self.wo = g("o_proj")                                   # [4096,16384]
        self.q_a_ln = bf16_f32(idx, f"{p}.q_a_layernorm.weight")
        self.kv_a_ln = bf16_f32(idx, f"{p}.kv_a_layernorm.weight")
        self.H, self.dk, self.dv = H, dk, dv
        # kv_b 输出 = H×(dk+dv),逐头交错:view(T,H,dk+dv) 后前 dk 为 k、后 dv 为 v
        self.kv_b_out = self.kv_b                               # [32768,512] 原样

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [T,4096] → [T,4096]。全注意力,fp32。"""
        T = x.shape[0]
        H, dk, dv = self.H, self.dk, self.dv
        cq = rmsnorm(x @ self.q_a.T, self.q_a_ln)               # [T,1536]
        q = (cq @ self.q_b.T).view(T, H, dk)                    # [T,H,dk]
        ckv = rmsnorm(x @ self.kv_a.T, self.kv_a_ln)            # [T,512]
        kv = (ckv @ self.kv_b_out.T).view(T, H, dk + dv)        # [T,H,dk+dv]
        k, v = kv[:, :, :dk], kv[:, :, dk:]                     # 逐头拆分
        att = torch.einsum("thd,shd->hts", q, k) / (dk ** 0.5)  # [H,T,T]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        att = att.masked_fill(~causal, float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hts,shd->thd", att, v)                # [T,H,dv]
        return o.reshape(T, H * dv) @ self.wo.T

    def forward_sparse(self, x: torch.Tensor, indexer) -> torch.Tensor:
        """DSA 稀疏注意力参考(审阅 §3.3):indexer 选 topk → 逐 query 稀疏 mask。
        x [T,4096] → [T,4096]。T ≤ 2048+3 时与 forward(全因果)语义等价。
        indexer:DsaIndexer 实例(与层同号)。"""
        from .dsa import topk_to_mask
        T = x.shape[0]
        H, dk, dv = self.H, self.dk, self.dv
        cq = rmsnorm(x @ self.q_a.T, self.q_a_ln)
        q = (cq @ self.q_b.T).view(T, H, dk)
        ckv = rmsnorm(x @ self.kv_a.T, self.kv_a_ln)
        kv = (ckv @ self.kv_b_out.T).view(T, H, dk + dv)
        k, v = kv[:, :, :dk], kv[:, :, dk:]
        topk = indexer.forward(x, cq)                           # [T,2051] int32
        vis = topk_to_mask(topk.long(), T)                      # [T,T]
        att = torch.einsum("thd,shd->hts", q, k) / (dk ** 0.5)
        att = att.masked_fill(~vis[None], float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = torch.einsum("hts,shd->thd", att, v)
        return o.reshape(T, H * dv) @ self.wo.T
