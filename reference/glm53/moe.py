# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash MoE:路由 + 专家(EXL3 按需解码)+ 共享专家 + dense MLP。

路由语义(vllm glm5next / FusedMoE noaux_tc,scoring=sigmoid):
  logits = h_fp32 @ W_gate^T            (router_dtype=fp32)
  scores = sigmoid(logits)
  topk 选自 scores + e_score_correction_bias
  权重 = scores[选中](renormalize: norm_topk_prob=true → 除以和)
  最终 = 路由输出 × routed_scaling_factor(2.5) + 共享专家(无缩放)
激活:SiluAndMulWithClamp(limit=10):gate 只 clamp max,up 双侧。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from .exl3_torch import decode_weight_t
from .exl3 import decode_weight as _decode_weight_np

if TYPE_CHECKING:
    from .config import Glm53TextConfig
    from .loader import ShardIndex

_EXL3_SUFFIX = ("trellis", "suh", "svh")


class ExpertStore:
    """EXL3 专家按需物化(解码 → fp32 torch),LRU 缓存。"""

    def __init__(self, idx: "ShardIndex", cfg: "Glm53TextConfig", cap: int = 256,
                 fp16: bool = True):
        self.idx, self.cfg, self.cap = idx, cfg, cap
        self.fp16 = fp16
        self._cache: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.stats = {"decoded": 0, "hits": 0}

    def _load(self, layer: int, e: int, which: str):
        p = f"model.language_model.layers.{layer}.mlp.experts.{e}.{which}"
        return (self.idx.get(f"{p}.trellis"), self.idx.get(f"{p}.suh"), self.idx.get(f"{p}.svh"))

    def expert(self, layer: int, e: int):
        key = (layer, e)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats["hits"] += 1
            return self._cache[key]
        w = []
        for which in ("gate_proj", "up_proj", "down_proj"):
            tr, suh, svh = self._load(layer, e, which)
            from .nn import DEVICE
            wt = decode_weight_t(
                torch.from_numpy(np.ascontiguousarray(tr)).to(DEVICE),
                torch.from_numpy(np.ascontiguousarray(suh)).to(DEVICE),
                torch.from_numpy(np.ascontiguousarray(svh)).to(DEVICE))
            from .nn import DEVICE
            w.append(wt.to(DEVICE).to(torch.float16) if self.fp16 else wt.to(DEVICE))
        w = tuple(w)  # (Wg[in,out], Wu[in,out], Wd[in,out])
        self.stats["decoded"] += 1
        self._cache[key] = w
        if len(self._cache) > self.cap:
            self._cache.popitem(last=False)
        return w

    def warm(self, layer: int, experts: set[int]):
        for e in experts:
            self.expert(layer, e)


def route(h: torch.Tensor, w_gate: torch.Tensor, bias: torch.Tensor,
          topk: int, renormalize: bool = True, scaling: float = 1.0):
    """noaux_tc + sigmoid。返回 (专家 id [T,k], 权重 [T,k])。"""
    logits = h.float() @ w_gate.T                       # fp32 [T, E]
    scores = torch.sigmoid(logits)
    sel = scores + bias                                 # 选topk用
    topi = sel.topk(topk, dim=-1).indices
    w = torch.gather(scores, -1, topi)
    if renormalize:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    return topi, w * scaling


def expert_forward(x: torch.Tensor, w: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   limit: float) -> torch.Tensor:
    """单个专家:x [T,in] → [T,out]。W 为 K-major [in,out]。"""
    wg, wu, wd = w
    x = x.to(wg.dtype)
    g = x @ wg
    u = x @ wu
    d = g.shape[-1]
    gate = torch.clamp(g, max=limit)
    up = torch.clamp(u, min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up) @ wd


class Glm53MoE:
    """一个 sparse 层的 MoE(batch 任意,CPU 参考速度)。"""

    def __init__(self, idx: "ShardIndex", cfg: "Glm53TextConfig", layer: int,
                 store: ExpertStore | None = None):
        p = f"model.language_model.layers.{layer}.mlp"
        self.cfg, self.layer = cfg, layer
        self.store = store or ExpertStore(idx, cfg)
        from .nn import bf16_f32, f32_param
        # router:bf16 位 → fp32 语义(位重释统一走 nn.bf16_f32)
        self.w_gate = bf16_f32(idx, f"{p}.gate.weight")
        self.bias = f32_param(idx, f"{p}.gate.e_score_correction_bias")

    def shared(self, h: torch.Tensor) -> torch.Tensor:
        # shared 专家权重物化一次并缓存(上 DEVICE)
        if not hasattr(self, "_shared_w"):
            from .nn import DEVICE, bf16_f32
            p = f"model.language_model.layers.{self.layer}.mlp.shared_experts"
            idx = self.store.idx
            g = bf16_f32(idx, f"{p}.gate_proj.weight").T.to(DEVICE)
            u = bf16_f32(idx, f"{p}.up_proj.weight").T.to(DEVICE)
            d = bf16_f32(idx, f"{p}.down_proj.weight").T.to(DEVICE)
            self._shared_w = (g, u, d)
        g, u, d = self._shared_w
        limit = 10.0
        ga = torch.clamp(h @ g, max=limit)
        up = torch.clamp(h @ u, min=-limit, max=limit)
        return (torch.nn.functional.silu(ga) * up) @ d

    def forward(self, h: torch.Tensor, *, want_debug: bool = False):
        """h [T, H] fp32 → [T, H]。返回 (y, debug{touched, weights}) 可选。"""
        topi, w = route(h, self.w_gate, self.bias,
                        self.cfg.num_experts_per_tok, True, self.cfg.routed_scaling_factor)
        y = torch.zeros_like(h)
        touched = set()
        for t in range(h.shape[0]):
            for k in range(topi.shape[1]):
                e = int(topi[t, k])
                touched.add(e)
                y[t] += float(w[t, k]) * expert_forward(h[t:t + 1], self.store.expert(self.layer, e), 10.0)[0]
        y = y + self.shared(h)
        if want_debug:
            return y, {"touched": touched, "weights": w, "topi": topi}
        return y


class Glm53DenseMLP:
    """前 3 个 dense 层(12288)。权重 bf16。"""

    def __init__(self, idx: "ShardIndex", layer: int):
        from .nn import bf16_f32
        p = f"model.language_model.layers.{layer}.mlp"
        self.wg = bf16_f32(idx, f"{p}.gate_proj.weight")
        self.wu = bf16_f32(idx, f"{p}.up_proj.weight")
        self.wd = bf16_f32(idx, f"{p}.down_proj.weight")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 权重 [out, in],K-major 语义:y = act @ W^T
        gate = torch.clamp(h @ self.wg.T, max=10.0)
        up = torch.clamp(h @ self.wu.T, min=-10.0, max=10.0)
        return (torch.nn.functional.silu(gate) * up) @ self.wd.T
