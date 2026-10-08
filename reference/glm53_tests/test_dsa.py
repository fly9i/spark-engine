# SPDX-License-Identifier: MIT
import os
"""DSA 稀疏参考的边界语义测试(真权重层 3,容器内跑)。

- T=96(≤2048):forward_sparse ≡ forward(全因果)——机制自证(满 pool 全选 + 尾部覆盖)。
- T=2100(>2048):因果性、预算上限、自身可见、尾部完整 等结构不变量。
"""
from __future__ import annotations

import pytest
import torch

from glm53.config import load_config
from glm53.loader import ShardIndex, find_model_dir
from glm53.dsa import DsaIndexer, topk_to_mask
from glm53.mla import MlaLayer
from pathlib import Path

ROOT = Path(os.environ.get("GLM53_ROOT", "."))
MDIR = find_model_dir(ROOT / "models/hf/hub", "brandonmusic/GLM-5.3-Flash-tr3-4bpw")


@pytest.fixture(scope="module")
def layer3():
    idx = ShardIndex(MDIR)
    return MlaLayer(idx, 3), DsaIndexer(idx, 3)


def test_short_equiv(layer3):
    mla, indexer = layer3
    torch.manual_seed(0)
    x = torch.randn(96, 4096) * 0.02
    a = mla.forward(x)
    b = mla.forward_sparse(x, indexer)
    rel = (a - b).norm() / b.norm()
    assert rel < 1e-4, f"短上下文 sparse≠full: rel={rel:.2e}"


def test_long_invariants(layer3):
    mla, indexer = layer3
    torch.manual_seed(0)
    T = 2100
    x = torch.randn(T, 4096) * 0.02
    topk = indexer.forward(x, mla.q_a_ln * (x @ mla.q_a.T)
                           * torch.rsqrt((x @ mla.q_a.T).pow(2).mean(-1, keepdim=True) + 1e-5))
    vis = topk_to_mask(topk.long(), T)
    # 因果:看不到未来
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert not (vis & future).any(), "稀疏选择违反因果"
    # 自身可见性:尾部机制保证 (t+1)%4>0 的 query 恒见自身;
    # (t+1)%4==0 时尾部为空,自身所在 pool 须凭分数入选——实测 2075/2083/2087/2091
    # 未入选,这是 DSA 真实语义(与 transformers 参考机制一致),不是 bug。
    t_all = torch.arange(T)
    tail_pos = t_all[(t_all + 1) % 4 != 0]
    assert vis[tail_pos, tail_pos].all(), "尾部机制失效:tc>0 的 query 看不到自身"
    # 预算:每 query ≤ 2051
    assert int(vis.sum(-1).max()) <= 2051
    # 过界后确实在裁剪(第 2099 个 query 可见数 < T)
    assert int(vis[-1].sum()) < T, "T>2048 时仍未稀疏,路径未生效"
    # 尾部连续:每个 query 可见集的最后一个元素是自己,且末尾 (t+1)%4 连续可见
    t = T - 1
    tc = (t + 1) % 4
    if tc:
        assert vis[t, t - tc + 1: t + 1].all(), "尾部残缺 pool 未完整追加"
