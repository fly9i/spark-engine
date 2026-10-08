# SPDX-License-Identifier: MIT
"""config.json → 强类型配置 + 层平面图。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LayerPlan:
    idx: int
    attn: str            # "kda" | "dsa"
    mlp: str             # "dense" | "sparse"


@dataclass(frozen=True)
class Glm53TextConfig:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    layer_plans: tuple[LayerPlan, ...]
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    first_k_dense_replace: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    num_attention_heads: int
    kda_num_heads: int
    kda_head_dim: int
    index_topk: int
    index_kpool: int
    index_n_heads: int
    index_head_dim: int
    hc_mult: int
    routed_scaling_factor: float
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]

    @property
    def dsa_layers(self) -> tuple[int, ...]:
        return tuple(p.idx for p in self.layer_plans if p.attn == "dsa")

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(p.idx for p in self.layer_plans if p.attn == "kda")

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(p.idx for p in self.layer_plans if p.mlp == "sparse")


def load_config(model_dir: str | Path) -> Glm53TextConfig:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg["text_config"]
    layer_types = tc["layer_types"]          # "linear_attention" | "deepseek_sparse_attention"
    mlp_types = tc["mlp_layer_types"]        # "dense" | "sparse"
    plans = tuple(
        LayerPlan(
            idx=i,
            attn="kda" if a == "linear_attention" else "dsa",
            mlp=m,
        )
        for i, (a, m) in enumerate(zip(layer_types, mlp_types))
    )
    lin = tc["linear_attn_config"]
    return Glm53TextConfig(
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        vocab_size=tc["vocab_size"],
        layer_plans=plans,
        n_routed_experts=tc["n_routed_experts"],
        n_shared_experts=tc["n_shared_experts"],
        num_experts_per_tok=tc["num_experts_per_tok"],
        moe_intermediate_size=tc["moe_intermediate_size"],
        intermediate_size=tc["intermediate_size"],
        first_k_dense_replace=tc["first_k_dense_replace"],
        kv_lora_rank=tc["kv_lora_rank"],
        q_lora_rank=tc["q_lora_rank"],
        qk_nope_head_dim=tc["qk_nope_head_dim"],
        v_head_dim=tc["v_head_dim"],
        num_attention_heads=tc["num_attention_heads"],
        kda_num_heads=lin["num_heads"],
        kda_head_dim=lin["head_dim"],
        index_topk=tc["index_topk"],
        index_kpool=tc["index_kpool"],
        index_n_heads=tc["index_n_heads"],
        index_head_dim=tc["index_head_dim"],
        hc_mult=tc["hc_mult"],
        routed_scaling_factor=tc["routed_scaling_factor"],
        max_position_embeddings=tc["max_position_embeddings"],
        eos_token_ids=tuple(tc["eos_token_id"]),
    )
