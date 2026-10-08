# SPDX-License-Identifier: MIT
"""EXL3 解码 torch 快路径。inner 与 numpy 参考逐位一致(纯 gather);
Hadamard/尺度为 fp32 矩阵乘(与 numpy 数值等价,求和序可能差 ulp 级)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MCG_MULTIPLIER = 0xCBAC1FED
LOP3_B = 0x8FFF_8FFF
LOP3_C = 0x3B60_3B60
NIBMAP_PATH = Path(__file__).resolve().parents[2] / "bench" / "poc_nibmap_true.npy"

_F: torch.Tensor | None = None
_SRC: torch.Tensor | None = None
_H: torch.Tensor | None = None


_TABLE_CACHE: dict = {}

def _tables(dev=None):
    """按设备缓存查找表;dev=None 走旧全局(兼容 CPU 路径)。"""
    if dev is not None:
        key = str(dev)
        if key not in _TABLE_CACHE:
            f, s, h = _tables(None)
            _TABLE_CACHE[key] = (f.to(dev), s.to(dev), h.to(dev))
        return _TABLE_CACHE[key]
    global _F, _SRC, _H
    if _F is None:
        w = torch.arange(65536, dtype=torch.int64)
        p = (w * MCG_MULTIPLIER) & 0xFFFFFFFF
        y = (p & LOP3_B) ^ LOP3_C
        lo = (y & 0xFFFF).to(torch.int64).to(torch.uint16).view(torch.float16).float()
        hi = (y >> 16).to(torch.int64).to(torch.uint16).view(torch.float16).float()
        _F = (lo + hi).to(torch.float16).float()
        _SRC = torch.from_numpy(np.load(NIBMAP_PATH)).long()
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0).float()
        _H = h / float(128.0 ** 0.5)
    return _F, _SRC, _H


def decode_inner_t(trellis_int16: torch.Tensor) -> torch.Tensor:
    """[Kt,Nt,64] int16 → inner fp16 值(fp32 表示)[Kt*16, Nt*16]。逐位 = numpy。"""
    F, SRC, _ = _tables(trellis_int16.device)
    Kt, Nt, _ = trellis_int16.shape
    w16 = trellis_int16.to(torch.int64) & 0xFFFF
    u32 = w16[..., 0::2] | (w16[..., 1::2] << 16)                  # [Kt,Nt,32] int64
    pos = torch.arange(256, device=trellis_int16.device)
    sh = 28 - 4 * (pos % 8)
    nib = (u32[..., pos // 8] >> sh) & 0xF                         # [Kt,Nt,256]
    word = (nib[..., SRC[..., 0]] | (nib[..., SRC[..., 1]] << 4)
            | (nib[..., SRC[..., 2]] << 8) | (nib[..., SRC[..., 3]] << 12))
    return F[word].permute(0, 2, 1, 3).reshape(Kt * 16, Nt * 16)


def decode_weight_t(trellis_int16: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """→ 完整权重 fp32 [Kt*16, Nt*16](含 Hadamard × suh × Hadamard × svh)。"""
    _, _, H = _tables(trellis_int16.device)
    inner = decode_inner_t(trellis_int16)
    K, N = inner.shape
    w = torch.einsum("ij,kjn->kin", H, inner.reshape(K // 128, 128, N)).reshape(K, N)
    w = w * suh.float()[:, None]
    w = (w.reshape(K, N // 128, 128) @ H).reshape(K, N)
    return w * svh.float()[None, :]
