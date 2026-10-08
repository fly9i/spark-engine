# SPDX-License-Identifier: MIT
import os
"""MoE 路由 + 专家执行测试。"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.config import load_config
from glm53.loader import ShardIndex, find_model_dir
from glm53.moe import Glm53DenseMLP, Glm53MoE, ExpertStore, expert_forward, route
from glm53.nn import silu_and_mul_with_clamp

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))
torch.manual_seed(0)


def test_router_noaux_tc_math():
    """手工复算 noaux_tc + sigmoid + renormalize + scaling。"""
    h = torch.randn(1, 8)
    wg = torch.randn(5, 8)
    bias = torch.randn(5)
    topi, w = route(h, wg, bias, topk=3, renormalize=True, scaling=2.5)
    scores = torch.sigmoid(h @ wg.T)[0]
    sel = (scores + bias).topk(3).indices.tolist()
    assert topi[0].tolist() == sel
    raw = torch.tensor([scores[i] for i in sel])
    expect = raw / raw.sum() * 2.5
    assert torch.allclose(w[0], expect, atol=1e-6)


def test_clamp_swiglu_semantics():
    x = torch.tensor([[3.0, -20.0, 0.5, 40.0]])  # gate | up
    out = silu_and_mul_with_clamp(x, limit=10.0)
    g = torch.clamp(x[..., :2], max=10.0)
    u = torch.clamp(x[..., 2:], min=-10.0, max=10.0)
    assert torch.allclose(out, torch.nn.functional.silu(g) * u)
    assert u[0, 0] == 0.5 and u[0, 1] == 10.0  # up 双侧 clamp 生效(40→10)


def test_dense_mlp_layer0():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    mlp = Glm53DenseMLP(idx, layer=0)
    assert mlp.wg.shape == (12288, 4096) and mlp.wd.shape == (4096, 12288)
    h = torch.randn(2, 4096)
    y = mlp.forward(h)
    assert y.shape == (2, 4096) and torch.isfinite(y).all()


def test_moe_layer4_forward():
    """首个 sparse 层(4):路由 ≤8 expert/token、共享专家、数值有限。"""
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    cfg = load_config(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    moe = Glm53MoE(idx, cfg, layer=4)
    assert moe.w_gate.shape == (288, 4096) and moe.bias.shape == (288,)

    h = torch.randn(2, 4096)
    y, dbg = moe.forward(h, want_debug=True)
    assert y.shape == (2, 4096) and torch.isfinite(y).all()
    assert dbg["topi"].shape == (2, 8)
    # 路由权重:renormalize × 2.5 → 和 ≈ 2.5
    assert torch.allclose(dbg["weights"].sum(-1), torch.full((2,), 2.5), atol=1e-4)

    # 锚点:token 0 手工复算(shared + 每个选中专家的加权和)
    y0 = moe.shared(h[:1])[0]
    for k in range(8):
        e = int(dbg["topi"][0, k])
        y0 += float(dbg["weights"][0, k]) * expert_forward(h[:1], moe.store.expert(4, e), 10.0)[0]
    assert torch.allclose(y[0], y0, atol=2e-2), (y[0] - y0).abs().max()  # fp16 专家路径容差

    # 缓存行为:同专家再取应命中
    d0 = moe.store.stats["decoded"]
    _ = moe.store.expert(4, int(dbg["topi"][0, 0]))
    assert moe.store.stats["decoded"] == d0
