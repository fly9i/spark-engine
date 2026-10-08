# SPDX-License-Identifier: MIT
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.config import load_config
from glm53.loader import ShardIndex, find_model_dir

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))


def test_layer_census():
    cfg = load_config(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    assert cfg.num_hidden_layers == 45
    assert len(cfg.kda_layers) == 34
    assert cfg.dsa_layers == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    assert len(cfg.moe_layers) == 42
    assert cfg.first_k_dense_replace == 3
    assert cfg.n_routed_experts == 288
    assert cfg.num_experts_per_tok == 8
    # 前 3 层 dense MLP 且全为 KDA 注意力
    assert all(cfg.layer_plans[i].mlp == "dense" for i in range(3))


def test_weight_index_present():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    for name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.10.mlp.experts.287.down_proj.trellis",
        "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
        "lm_head.weight",
    ):
        assert name in idx, name
