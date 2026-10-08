# SPDX-License-Identifier: MIT
import os
"""KDA / MLA / MHC 模块 sanity(真权重,单层)。"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.kda import KdaLayer, causal_conv1d
from glm53.loader import ShardIndex, find_model_dir
from glm53.mhc import hc_contract, hc_expand, mhc_post, mhc_pre
from glm53.mla import MlaLayer

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))
torch.manual_seed(1)


def test_causal_conv():
    x = torch.randn(6, 3)
    w = torch.zeros(3, 1, 4)
    w[:, 0, 3] = 1.0                      # 只有最近 tap
    y = causal_conv1d(x, w)
    assert torch.allclose(y, x)           # w[3]·x[t] → 恒等
    w2 = torch.ones(3, 1, 4)
    y2 = causal_conv1d(x, w2)
    assert torch.allclose(y2[3:], x[3:] + x[2:-1] + x[1:-2] + x[:-3], atol=1e-5)
    assert (y2[:3].abs() > 0).all()       # pad 区也有输出


def test_mhc_shapes_and_idempotent_post():
    T, n, H = 5, 4, 64
    res = hc_expand(torch.randn(T, H), n)
    assert res.shape == (T, n, H) and torch.allclose(res[:, 0], res[:, 3])
    assert hc_contract(res, n).shape == (T, H)
    fn = torch.randn(24, n * H)
    post, comb, z = mhc_pre(res, fn, torch.randn(3), torch.randn(24), torch.ones(H))
    assert post.shape == (T, n, 1) and comb.shape == (T, n, n) and z.shape == (T, H)
    # sinkhorn 末次为列归一(dim=-2 和 ≈ 1);行和近似
    assert (comb.sum(-2) - 1).abs().max() < 1e-3
    assert (comb.sum(-1) - 1).abs().max() < 0.1
    res2 = mhc_post(torch.randn(T, H), res, post, comb)
    assert res2.shape == (T, n, H) and torch.isfinite(res2).all()


def test_kda_layer0_forward():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    kda = KdaLayer(idx, 0)
    x = torch.randn(6, 4096)
    y = kda.forward(x)
    assert y.shape == (6, 4096) and torch.isfinite(y).all()
    # 首位置不受未来影响(causal):截断输入后首输出不变
    y2 = KdaLayer(idx, 0).forward(x[:3])
    assert torch.allclose(y[:3], y2, atol=1e-4), (y[:3] - y2).abs().max()


def test_mla_layer3_forward():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    mla = MlaLayer(idx, 3)
    x = torch.randn(6, 4096)
    y = mla.forward(x)
    assert y.shape == (6, 4096) and torch.isfinite(y).all()
    y2 = MlaLayer(idx, 3).forward(x[:3])
    assert torch.allclose(y[:3], y2, atol=1e-4), (y[:3] - y2).abs().max()
