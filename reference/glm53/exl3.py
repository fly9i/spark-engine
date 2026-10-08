# SPDX-License-Identifier: MIT
"""EXL3/MCG 解码:生产语义的参考实现(2026-09-20 PoC 验证)。

语义(方案文档 §1.2,勿手推公式,以 bench/poc_nibmap_true.npy 映射表为准):
  1. trellis int16 [Kt, Nt, 64] → 每 tile 64 个 uint16 → 小端成对拼 32 个 uint32
     u32[j] = w16[2j] | w16[2j+1]<<16
  2. nibble 流(256/tile):nib[p] = (u32[p//8] >> (28 - 4*(p%8))) & 0xF   (MSB 优先)
  3. 词组装:cell(r,c) 的 16 位词 = nib[src[r,c,0]] | nib[src[r,c,1]]<<4
     | nib[src[r,c,2]]<<8 | nib[src[r,c,3]]<<12   (src = poc_nibmap_true)
  4. 码本 MCG:y = (word * 0xCBAC1FED) mod 2^32;y = (y & 0x8FFF8FFF) ^ 0x3B603B60;
     value = fp16(low16(y)) + fp16(high16(y))   (fp16 加法舍入)
  5. 完整权重:inner → 128 宽 Hadamard 行块 → *suh[:,None] → 128 宽 Hadamard 列块
     → *svh[None,:]   (inner 为 [in_features, out_features] K-major)

验证口径:真实权重 gate/up/down ×2 expert 与 exllamav3_ext.reconstruct 逐位一致;
全管线与 had_r_128+hgemm+had_r_128 cos=1.000000(bench/poc_exl3_format.py)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MCG_MULTIPLIER = 0xCBAC1FED
LOP3_B = 0x8FFF8FFF
LOP3_C = 0x3B603B60
_H128_CACHE: dict[int, np.ndarray] = {}
_F_CACHE: np.ndarray | None = None
_SRC_CACHE: np.ndarray | None = None

# 映射表路径:仓根 bench/(PoC 产物,与 engine 同仓演进,勿复制副本)
NIBMAP_PATH = Path(__file__).resolve().parents[2] / "bench" / "poc_nibmap_true.npy"


def mcg_table() -> np.ndarray:
    """全码本:65536 个 16 位词 → fp16 值(带 fp16 加法舍入)。"""
    global _F_CACHE
    if _F_CACHE is None:
        x = np.arange(65536, dtype=np.uint64)
        p = ((x * np.uint64(MCG_MULTIPLIER)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        y = (p & np.uint32(LOP3_B)) ^ np.uint32(LOP3_C)
        lo = (y & 0xFFFF).astype(np.uint16)
        hi = (y >> 16).astype(np.uint16)
        s = lo.view(np.float16).astype(np.float32) + hi.view(np.float16).astype(np.float32)
        _F_CACHE = s.astype(np.float16).astype(np.float32)
    return _F_CACHE


def nib_src() -> np.ndarray:
    """[16, 16, 4] 映射表:cell(r,c) 词的 4 个 nibble 位置(小端槽位序)。"""
    global _SRC_CACHE
    if _SRC_CACHE is None:
        _SRC_CACHE = np.load(NIBMAP_PATH)
    return _SRC_CACHE


def hadamard128() -> np.ndarray:
    if 128 not in _H128_CACHE:
        m = np.ones((1, 1), dtype=np.float32)
        while m.shape[0] < 128:
            m = np.block([[m, m], [m, -m]])
        _H128_CACHE[128] = (m / np.sqrt(np.float32(128.0))).astype(np.float32)
    return _H128_CACHE[128]


def decode_inner(trellis: np.ndarray) -> np.ndarray:
    """trellis int16 [Kt, Nt, 64] → inner fp16 [Kt*16, Nt*16](K-major,未乘尺度)。"""
    if trellis.dtype != np.int16 or trellis.ndim != 3 or trellis.shape[-1] != 64:
        raise ValueError(f"trellis 应为 int16 [Kt, Nt, 64],得到 {trellis.dtype} {trellis.shape}")
    Kt, Nt = trellis.shape[0], trellis.shape[1]
    w16 = np.ascontiguousarray(trellis).astype(np.uint16)
    u32 = w16[..., 0::2].astype(np.uint32) | (w16[..., 1::2].astype(np.uint32) << 16)

    pos = np.arange(256)
    sh = (28 - 4 * (pos % 8)).astype(np.uint32)
    nib = ((u32[..., pos // 8] >> sh[None, None, :]) & 0xF).astype(np.uint16)  # [Kt,Nt,256]

    src = nib_src()
    word = (
        nib[..., src[..., 0]]
        | (nib[..., src[..., 1]] << 4)
        | (nib[..., src[..., 2]] << 8)
        | (nib[..., src[..., 3]] << 12)
    )                                                                          # [Kt,Nt,16,16]
    inner = mcg_table()[word].astype(np.float16)
    # tile(k,n) 的 (r,c) → 矩阵 [k*16+r, n*16+c]
    return inner.transpose(0, 2, 1, 3).reshape(Kt * 16, Nt * 16)


def decode_weight(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """完整 bf16 语义权重 [in_features, out_features](fp32 表示,含 Hadamard×尺度)。"""
    inner = decode_inner(trellis).astype(np.float32)
    K, N = inner.shape
    if K % 128 or N % 128:
        raise ValueError(f"Hadamard 需要 K,N 为 128 倍数,得到 {K}×{N}")
    H = hadamard128()
    w = np.einsum("ij,kjn->kin", H, inner.reshape(K // 128, 128, N)).reshape(K, N)
    w *= suh.astype(np.float32)[:, None]
    w = (w.reshape(K, N // 128, 128) @ H).reshape(K, N)
    w *= svh.astype(np.float32)[None, :]
    return w
