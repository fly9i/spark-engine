# SPDX-License-Identifier: MIT
import os
"""EXL3 解码回归测试。

注意:golden hash 锁定的是「本实现 vs PoC 验证语义」的一致性(防回归)。
与 exllamav3_ext 的容器内逐位复验需要 GPU 维护窗口(见 tests/README.md)。
"""
from pathlib import Path
import hashlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.exl3 import decode_inner, decode_weight, mcg_table, nib_src
from glm53.loader import ShardIndex, find_model_dir

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))
P = "model.language_model.layers.10.mlp.experts.0"


def test_mcg_table_invariants():
    F = mcg_table()
    assert F.shape == (65536,)
    # 锚点:词 0 → y = 0^0x3B603B60 = 0x3B603B60 → fp16(0x3B60)+fp16(0x3B60)
    lo = np.uint16(0x3B60).view(np.float16).astype(np.float32)
    assert F[0] == pytest.approx(2.0 * lo, abs=1e-6)
    assert not np.isnan(F[:1024]).any()


def test_nibmap_shape():
    src = nib_src()
    assert src.shape == (16, 16, 4)
    assert sorted(np.bincount(src.ravel(), minlength=256))[-1] == 4  # 每 nibble 恰好 4 用


def test_decode_real_expert():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    trellis = idx.get(f"{P}.gate_proj.trellis")
    suh, svh = idx.get(f"{P}.gate_proj.suh"), idx.get(f"{P}.gate_proj.svh")
    inner = decode_inner(trellis)
    assert inner.shape == (4096, 2048)
    assert inner.dtype == np.float16
    # golden:PoC 验证轮(与 exllamav3_ext 逐位一致)的哈希
    h = hashlib.sha256(inner.tobytes()).hexdigest()
    assert h == _GOLDEN_GATE_INNER, f"inner 回归!得到 {h}"


def test_decode_weight_finite():
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    w = decode_weight(idx.get(f"{P}.down_proj.trellis"),
                      idx.get(f"{P}.down_proj.suh"), idx.get(f"{P}.down_proj.svh"))
    assert w.shape == (2048, 4096)
    assert np.isfinite(w).all()
    assert w.std() > 0.001            # 非平凡


# 2026-09-20:PoC 验证算法(ext 逐位一致)+ 同源实现对拍通过后记录;
# 下次维护窗口做 ext 容器内复验后如有出入,以 ext 为准重录
_GOLDEN_GATE_INNER = "ef17bb644375ba57354548ab6c867b963db3248ee7f4ddf3adef1c8af0258567"
