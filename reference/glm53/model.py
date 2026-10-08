# SPDX-License-Identifier: MIT
"""Glm5Next 全栈 forward(M0 正确性参考,CPU/fp32)。

数据流:x = embed → hc_expand(4 流)→ 45 层
  [hc_pre(attn) → attn(KDA|MLA) → fused(hc_post+hc_pre(ffn)) → mlp(MoE|dense)]
  → 末层 hc_post → hc_contract → final norm → lm_head。
MTP(层45)与 vision 不参与文本生成。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .kda import KdaLayer
from .mhc import hc_contract, hc_expand, mhc_post, mhc_pre, rmsnorm
from .mla import MlaLayer
from .moe import ExpertStore, Glm53DenseMLP, Glm53MoE

if TYPE_CHECKING:
    from .config import Glm53TextConfig
    from .loader import ShardIndex


class Glm53Engine:
    def __init__(self, idx: "ShardIndex", cfg: "Glm53TextConfig", *, store: ExpertStore | None = None,
                 n_layers: int | None = None):
        self.idx, self.cfg = idx, cfg
        import os
        cap = int(os.environ.get("GLM53_EXPERT_CAP", "256"))
        self.store = store or ExpertStore(idx, cfg, cap=cap)
        P = "model.language_model"
        from .nn import bf16_f32, f32_param
        self.embed = bf16_f32(idx, f"{P}.embed_tokens.weight")
        self.final_norm = bf16_f32(idx, f"{P}.norm.weight")
        self.lm_head = bf16_f32(idx, "lm_head.weight")
        self.layers = []
        for plan in (cfg.layer_plans[:n_layers] if n_layers else cfg.layer_plans):
            li = plan.idx
            attn = KdaLayer(idx, li) if plan.attn == "kda" else MlaLayer(idx, li)
            mlp = Glm53MoE(idx, cfg, li, self.store) if plan.mlp == "sparse" else Glm53DenseMLP(idx, li)
            L = f"{P}.layers.{li}"
            # fn 是 BF16:位重释统一走 nn.bf16_f32(此前漏做 → 全模型乱码的根因)
            hc = {
                "attn_fn": bf16_f32(idx, f"{L}.hc_attn_fn"),
                "attn_scale": f32_param(idx, f"{L}.hc_attn_scale"),
                "attn_base": f32_param(idx, f"{L}.hc_attn_base"),
                "ffn_fn": bf16_f32(idx, f"{L}.hc_ffn_fn"),
                "ffn_scale": f32_param(idx, f"{L}.hc_ffn_scale"),
                "ffn_base": f32_param(idx, f"{L}.hc_ffn_base"),
                "in_ln": bf16_f32(idx, f"{L}.input_layernorm.weight"),
                "post_ln": bf16_f32(idx, f"{L}.post_attention_layernorm.weight"),
            }
            self.layers.append((plan, attn, mlp, hc))

    def to(self, dev):
        from .nn import move, set_device
        set_device(dev)
        self.embed, self.final_norm, self.lm_head = move([self.embed, self.final_norm, self.lm_head])
        for i, (plan, attn, mlp, hc) in enumerate(self.layers):
            attn.__dict__.update(move(attn.__dict__))
            if hasattr(mlp, "store"):
                for k, v in mlp.__dict__.items():
                    if torch.is_tensor(v):
                        setattr(mlp, k, v.to(dev))
            else:
                mlp.__dict__.update(move(mlp.__dict__))
            self.layers[i] = (plan, attn, mlp, move(hc))
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[T] → logits [T, V](fp32)。"""
        x = self.embed[input_ids]
        residual = hc_expand(x, 4)
        post = comb = None
        n_layers = len(self.layers)
        for i, (plan, attn, mlp, hc) in enumerate(self.layers):
            if post is None:  # 层 0
                post, comb, z = mhc_pre(residual, hc["attn_fn"], hc["attn_scale"],
                                        hc["attn_base"], hc["in_ln"])
            else:
                residual = mhc_post(post.pop("x"), residual, post["mix"], comb)
                post, comb, z = mhc_pre(residual, hc["attn_fn"], hc["attn_scale"],
                                        hc["attn_base"], hc["in_ln"])
            a = attn.forward(z)
            residual = mhc_post(a, residual, post, comb)
            post, comb, z = mhc_pre(residual, hc["ffn_fn"], hc["ffn_scale"],
                                    hc["ffn_base"], hc["post_ln"])
            m = mlp.forward(z) if not isinstance(mlp, Glm53MoE) else mlp.forward(z)
            if i == n_layers - 1:
                residual = mhc_post(m, residual, post, comb)
            else:
                post = {"mix": post, "x": m}   # 延后:下一层 fused post+pre
        y = hc_contract(residual, 4)
        return rmsnorm(y, self.final_norm) @ self.lm_head.T

    @torch.no_grad()
    def greedy(self, input_ids: torch.Tensor, max_new_tokens: int,
               eos_ids: tuple[int, ...] = (154820, 154827, 154829)) -> list[int]:
        ids = list(map(int, input_ids))
        out = []
        for _ in range(max_new_tokens):
            from .nn import DEVICE
            logits = self.forward(torch.tensor(ids, device=DEVICE))
            nxt = int(torch.argmax(logits[-1]))
            if nxt in eos_ids:
                break
            ids.append(nxt)
            out.append(nxt)
        return out
