# SPDX-License-Identifier: MIT
import os
"""torch 快路径 vs numpy 参考 / golden hash。"""
from pathlib import Path
import hashlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm53.exl3 import decode_inner, decode_weight
from glm53.exl3_torch import decode_inner_t, decode_weight_t
from glm53.loader import ShardIndex, find_model_dir

HUB = Path(os.environ.get("HF_HUB_DIR", "models/hf/hub"))
P = "model.language_model.layers.10.mlp.experts.0"


def _get(name):
    idx = ShardIndex(find_model_dir(HUB, "brandonmusic/GLM-5.3-Flash-tr3-4bpw"))
    return idx.get(f"{P}.{name}")


def test_torch_inner_bitexact_and_golden():
    tr = _get("gate_proj.trellis")
    inner_np = decode_inner(tr)
    inner_t = decode_inner_t(torch.from_numpy(np.ascontiguousarray(tr)))
    assert torch.from_numpy(inner_np.copy()).equal(inner_t), "torch 与 numpy inner 不一致"
    h = hashlib.sha256(inner_t.to(torch.float16).numpy().tobytes()).hexdigest()
    assert h == "ef17bb644375ba57354548ab6c867b963db3248ee7f4ddf3adef1c8af0258567", h


def test_torch_weight_close():
    tr, suh, svh = _get("down_proj.trellis"), _get("down_proj.suh"), _get("down_proj.svh")
    w_np = decode_weight(tr, suh, svh)
    w_t = decode_weight_t(torch.from_numpy(np.ascontiguousarray(tr)),
                          torch.from_numpy(np.ascontiguousarray(suh)),
                          torch.from_numpy(np.ascontiguousarray(svh))).numpy()
    d = np.abs(w_np - w_t).max()
    assert d < 1e-3, d
