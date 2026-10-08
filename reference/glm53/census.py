#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import os
"""架构普查:权重清点 × config 交叉核对。在 :exl3 容器内运行(需 numpy)。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.config import load_config
from glm53.loader import ShardIndex, find_model_dir

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))
REPO = "brandonmusic/GLM-5.3-Flash-tr3-4bpw"


def main():
    model_dir = find_model_dir(HUB, REPO)
    cfg = load_config(model_dir)
    idx = ShardIndex(model_dir)

    print(f"model dir : {model_dir}")
    print(f"layers    : {cfg.num_hidden_layers}"
          f" (KDA {len(cfg.kda_layers)} + DSA {len(cfg.dsa_layers)}),"
          f" MoE {len(cfg.moe_layers)} + dense {cfg.num_hidden_layers - len(cfg.moe_layers)}")
    print(f"experts   : {cfg.n_routed_experts} routed × top-{cfg.num_experts_per_tok}"
          f" + {cfg.n_shared_experts} shared; moe_inter={cfg.moe_intermediate_size}")
    print(f"mla       : kv_lora={cfg.kv_lora_rank} q_lora={cfg.q_lora_rank}"
          f" qk_nope={cfg.qk_nope_head_dim} v={cfg.v_head_dim} heads={cfg.num_attention_heads}")
    print(f"kda       : heads={cfg.kda_num_heads} head_dim={cfg.kda_head_dim}")
    print(f"indexer   : n_heads={cfg.index_n_heads} head_dim={cfg.index_head_dim}"
          f" topk={cfg.index_topk} kpool={cfg.index_kpool}")
    print(f"hc        : mult={cfg.hc_mult} | ctx_max={cfg.max_position_embeddings >> 10}K")

    # 权重分类
    cat = Counter()
    exl3_tensors = 0
    for name, (_f, _o, nb, dt, shape) in idx.entries.items():
        if any(name.endswith(s) for s in (".trellis", ".suh", ".svh", ".mcg")):
            exl3_tensors += 1
            cat["exl3(experts)"] += nb
        elif ".layers." in name:
            li = int(name.split(".layers.")[1].split(".")[0])
            cat[f"layer#{li}({'kda' if li in cfg.kda_layers else 'dsa'}"
                f"/{'sparse' if li in cfg.moe_layers else 'dense'})"] += nb
        else:
            cat["全局(embed/lm_head/final/mtp/vision)"] += nb
    total = sum(cat.values())
    print(f"\n权重字节账(全 {len(idx.entries)} 张量,EXL3 张量 {exl3_tensors} 个):")
    for k in sorted(cat, key=lambda x: -cat[x]):
        print(f"  {k:36s} {cat[k] / 2**30:8.2f} GiB  ({cat[k] / total:5.1%})")
    print(f"  {'TOTAL':36s} {total / 2**30:8.2f} GiB")

    # 交叉核对:每 MoE 层应有 288×3 个 trellis
    for li in (cfg.moe_layers[0], cfg.moe_layers[-1]):
        n = sum(1 for x in idx.names(f"model.language_model.layers.{li}.mlp.experts.")
                if x.endswith(".trellis"))
        assert n == 3 * cfg.n_routed_experts, (li, n)
    print(f"\n交叉核对 ✓(MoE 层 trellis 数 = {3 * cfg.n_routed_experts} = 288 experts × gate/up/down)")


if __name__ == "__main__":
    main()
