# SPDX-License-Identifier: MIT
"""DSA(DeepSeek Sparse Attention)慢速正确参考 — 审阅 §3.3:覆盖 index_topk=2048 边界之外。

语义逐行对齐 engine/ref_modeling_glm5_next.py 的 Glm5NextTextIndexer
(transformers 5.17,生产镜像同源):
  indexer:q = wq_b(q_resid) [S,32,128];k = k_norm(wk(x)) [S,128](LayerNorm eps 1e-6);
  pool(kpool=4) 压缩:gate_scores + ape → softmax 加权平均 → pool_key [P,128];
  scores = relu(q·pool_k × 128^-0.5);weights_proj 逐头加权求和(×32^-0.5);
  pool 可选 ⇔ pool 末 token 对 query 因果可见且 pool 完整(尾部残缺 pool 恒不当候选);
  topk 512 pool → 展开 2048 token;追加可见尾部(当前残缺 pool 的已有部分);
  输出 [S, 2051] int32(-1 = 无效)。

限制(参考实现,B=1、无 padding、整段重算,use_cache=False 等价)。
"""
from __future__ import annotations

import torch

from .nn import bf16_f32


class DsaIndexer:
    TOPK = 2048
    KPOOL = 4
    N_HEADS = 32
    HEAD_DIM = 128

    def __init__(self, idx, layer: int):
        p = f"model.language_model.layers.{layer}.self_attn.indexer"
        self.wq_b = bf16_f32(idx, f"{p}.wq_b.weight")        # [4096, 1536]
        self.wk = bf16_f32(idx, f"{p}.wk.weight")            # [128, 4096]
        self.k_norm_w = bf16_f32(idx, f"{p}.k_norm.weight")  # [128]
        self.k_norm_b = bf16_f32(idx, f"{p}.k_norm.bias")    # [128]
        self.weights_proj = bf16_f32(idx, f"{p}.weights_proj.weight")  # [32, 4096]
        self.ape = bf16_f32(idx, f"{p}.index_kpool_compress_ape")        # [4, 128]
        self.cgate = bf16_f32(idx, f"{p}.index_kpool_compress_gate")     # [128, 4096]

    def _k_norm(self, k: torch.Tensor) -> torch.Tensor:
        v = k.float()
        mu = v.mean(-1, keepdim=True)
        var = (v - mu).pow(2).mean(-1, keepdim=True)
        return (v - mu) * torch.rsqrt(var + 1e-6) * self.k_norm_w + self.k_norm_b

    def forward(self, x: torch.Tensor, q_resid: torch.Tensor) -> torch.Tensor:
        """x [T,4096],q_resid [T,1536] → topk_indices [T, 2051] int32(-1 填充)。"""
        T = x.shape[0]
        H, D, KP = self.N_HEADS, self.HEAD_DIM, self.KPOOL
        dev = x.device

        q = (q_resid @ self.wq_b.T).view(T, H, D)                       # [T,H,D]
        k = self._k_norm(x @ self.wk.T)                                 # [T,D]
        gate = x @ self.cgate.T                                         # [T,D]

        # ── pool 压缩(first_key=0,无 padding)──
        P_full = T // KP                                                # 完整 pool 数
        pk = k[: P_full * KP].view(P_full, KP, D)
        pg = gate[: P_full * KP].view(P_full, KP, D)
        logits = pg + self.ape[None]                                    # [P,KP,D]
        prob = torch.softmax(logits, dim=1)
        pool_keys = (prob * pk).sum(1)                                  # [P,D]

        # ── 打分 ──
        scores = torch.relu(torch.einsum("thd,pd->thp", q, pool_keys) * (D ** -0.5))
        w = (x @ self.weights_proj.T) * (H ** -0.5)                     # [T,H]
        index_scores = torch.einsum("th,thp->tp", w, scores)            # [T,P]

        # pool 可选 ⇔ 末 token ≤ query(因果可见)
        pool_end = torch.arange(P_full, device=dev) * KP + (KP - 1)     # [P]
        visible = pool_end[None, :] <= torch.arange(T, device=dev)[:, None]  # [T,P]
        index_scores = index_scores.masked_fill(~visible, torch.finfo(torch.float32).min)

        select_k = min(self.TOPK // KP, P_full)
        sel = index_scores.topk(select_k, dim=-1).indices               # [T,512]
        tok_idx = (sel[..., None] * KP + torch.arange(KP, device=dev)).flatten(-2)  # [T,2048]
        sel_valid = visible.gather(-1, sel)                             # [T,512]
        tok_idx = tok_idx.masked_fill(~sel_valid[..., None].expand(-1, -1, KP).flatten(-2), -1)

        # ── 追加可见尾部(当前残缺 pool 已写入的部分)──
        qpos = torch.arange(T, device=dev)
        visible_count = qpos + 1
        tail_count = visible_count % KP
        tail_start = visible_count - tail_count                         # [T]
        offs = torch.arange(KP - 1, device=dev)
        tail = tail_start[:, None] + offs[None, :]                      # [T,3]
        tail = tail.masked_fill(~(offs[None, :] < tail_count[:, None]), -1)

        out = torch.cat([tok_idx, tail], dim=-1)                        # [T,2051]
        out = torch.nn.functional.pad(out, (0, self.TOPK + KP - 1 - out.shape[-1]), value=-1)
        return out[:, : self.TOPK + KP - 1].to(torch.int32)


def topk_to_mask(topk_indices: torch.Tensor, kv_len: int) -> torch.Tensor:
    """topk [T,W](-1 无效)→ bool mask [T, kv_len](True=可见)。
    与 ref build_attention_mask_from_topk 的 scatter_add 语义一致(集合语义,重复去重)。"""
    T = topk_indices.shape[0]
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    safe = topk_indices.clamp(0, kv_len - 1)
    counts = torch.zeros(T, kv_len, dtype=torch.int32, device=topk_indices.device)
    counts.scatter_add_(1, safe, valid.to(torch.int32))
    return counts.ne(0)
