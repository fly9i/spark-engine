# SPDX-License-Identifier: MIT
"""torch 基础件:RMSNorm / clamp-SwiGLU / bf16 桥接。语义对齐 vllm glm5next。"""
from __future__ import annotations

import numpy as np
import torch


def np_u2_to_bf16(arr: np.ndarray) -> torch.Tensor:
    """numpy '<u2'(bf16 位)→ torch bfloat16。"""
    return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)


# ── 检查点 dtype 约定唯一收口(审阅 §3.5)──
# 历史根因:hc_fn 漏做 uint16→bfloat16 位重释 → 全模型乱码。
# 纪律:模块内禁止再手写 view(np.uint16)/view(torch.bfloat16),一律走下面两个函数。

def bf16_f32(idx, name: str) -> torch.Tensor:
    """检查点 BF16 张量 → torch fp32(uint16→bfloat16 位重释的唯一入口)。"""
    return np_u2_to_bf16(idx.get(name)).float()


def f32_param(idx, name: str) -> torch.Tensor:
    """检查点 F32 小参数(dt_bias/A_log/hc_scale/hc_base 等)直取。"""
    return torch.from_numpy(np.ascontiguousarray(idx.get(name))).float()


def silu_and_mul_with_clamp(x: torch.Tensor, limit: float,
                            alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """vllm SiluAndMulWithClamp 语义:
    gate = clamp(x[..., :d], max=limit); up = clamp(x[..., d:], ±limit);
    out = gate * sigmoid(alpha·gate) * (up + beta)。GLM-5.3: alpha=1, beta=0。
    """
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=limit)
    up = torch.clamp(x[..., d:], min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.weight, self.eps = weight.float(), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * v


# ── 设备支持 ──
DEVICE = torch.device("cpu")

def set_device(d):
    global DEVICE
    DEVICE = torch.device(d)

def move(obj):
    """递归把 tensor/dict/list 搬到 DEVICE。"""
    if torch.is_tensor(obj):
        return obj.to(DEVICE)
    if isinstance(obj, dict):
        return {k: move(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move(v) for v in obj)
    return obj
