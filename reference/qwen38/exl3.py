"""EXL3 tensor decoding for the Qwen3.8-Flash-Next reference (fp32, readable, not fast).

Format (QTIP-style trellis, mul1 codebook), as stored by the quantizer:
  trellis [K_in/16, N_out/16, 16*bits] int16 : one 256-weight tile per (k-tile, n-tile).
  Tile bitstream: the int16 words viewed as little-endian uint32, read MSB-first; 256*bits bits,
  tail-biting.  The 16-bit state of sequence position t is the 16 bits ending at bit
  ((t+1)*bits + 256*bits) mod (256*bits), i.e. it starts at ((t+1)*bits - 16) mod (256*bits).
  Position t holds tile element PERM[t] (row = k within tile, col = n within tile).
  mul1 codebook: x = state * 0x83DCD12D (mod 2^32); s = sum of its 4 bytes;
                 value = fp16( fp16(1024 + s) * fp16(0x1eee) + fp16(0xc931) ).
  Weight (y = x @ W, W is [K_in, N_out]):
                 W = diag(svh) applied on N after H128 on N, of diag(suh) on K after H128 on K, i.e.
                 W = ((H_K @ inner) * suh[:, None]) @ H_N * svh[None, :]
  with H = Sylvester Walsh-Hadamard / sqrt(128), applied blockwise per 128.
"""
from __future__ import annotations
import math
import torch

MUL1 = 0x83DCD12D


def tensor_core_perm() -> list[int]:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8:t * 8 + 8] = [r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                                 r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1]
    return perm


_cb = {}


def mul1_codebook(device) -> torch.Tensor:
    """All 65536 decoded values (fp16 values held in fp32), single rounding of the fused fma."""
    if device in _cb:
        return _cb[device]
    s = torch.arange(65536, dtype=torch.int64)
    x = (s * MUL1) & 0xFFFFFFFF
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    h = (1024 + bsum).double()  # exactly representable in fp16 (1024..2044)
    k_inv = torch.tensor([0x1eee], dtype=torch.int16).view(torch.float16).double()
    k_bias = torch.tensor([0xc931 - 0x10000], dtype=torch.int16).view(torch.float16).double()
    v = (h * k_inv + k_bias).to(torch.float16).float()  # product and sum exact in fp64: one rounding
    _cb[device] = v.to(device)
    return _cb[device]


_had = {}


def hadamard(n: int, device) -> torch.Tensor:
    key = (n, device)
    if key not in _had:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _had[key] = (h / math.sqrt(n)).float().to(device)
    return _had[key]


def tile_states(trellis: torch.Tensor, bits: int) -> torch.Tensor:
    """trellis [..., 16*bits] int16 -> [..., 256] int64 trellis states (sequence order)."""
    lead = trellis.shape[:-1]
    w16 = trellis.reshape(-1, 16 * bits).to(torch.int64) & 0xFFFF
    w32 = w16[:, 0::2] | (w16[:, 1::2] << 16)                       # little-endian uint32 view
    nw, L = 8 * bits, 256 * bits
    t = torch.arange(256, device=trellis.device)
    b0 = ((t + 1) * bits - 16) % L                                  # first bit of the state (MSB-first)
    i0 = b0 >> 5
    i1 = (i0 + 1) % nw
    sh = 48 - (b0 & 31)                                             # 64-bit window [w[i0] : w[i1]]
    win = (w32[:, i0] << 32) | w32[:, i1]
    return ((win >> sh) & 0xFFFF).reshape(*lead, 256)


def decode_inner(trellis: torch.Tensor) -> torch.Tensor:
    """trellis [Kt, Nt, 16*bits] -> inner weight [Kt*16, Nt*16] fp32 (codebook values)."""
    Kt, Nt, W = trellis.shape
    bits = W // 16
    assert W == 16 * bits, f"fractional bitrate not handled ({W} words)"
    dev = trellis.device
    out = torch.empty(Kt, Nt, 256, dtype=torch.float32, device=dev)
    cb = mul1_codebook(dev)
    perm = torch.tensor(tensor_core_perm(), device=dev)
    step = max(1, (1 << 24) // (Nt * 256))                          # bound temporaries
    for k0 in range(0, Kt, step):
        st = tile_states(trellis[k0:k0 + step], bits)                # [s, Nt, 256]
        vals = cb[st]
        tile = torch.empty_like(vals)
        tile[..., perm] = vals                                       # position t -> element perm[t]
        out[k0:k0 + step] = tile
    return out.view(Kt, Nt, 16, 16).permute(0, 2, 1, 3).reshape(Kt * 16, Nt * 16)


def decode_weight(trellis, suh, svh, mul1=None) -> torch.Tensor:
    """Full weight W [K_in, N_out] fp32 with y = x @ W."""
    if mul1 is not None:
        assert int(mul1) & 0xFFFFFFFF == MUL1, f"unexpected mul1 multiplier {int(mul1):#x}"
    w = decode_inner(trellis)
    K, N = w.shape
    dev = w.device
    h = hadamard(128, dev)
    w = (h @ w.view(K // 128, 128, N)).view(K, N) * suh.float().view(K, 1)
    w = (w.view(K, N // 128, 128) @ h).view(K, N) * svh.float().view(1, N)
    return w


def ngram_rows(packed: torch.Tensor, bits: int, head_bias_rows: torch.Tensor) -> torch.Tensor:
    """N-gram table rows: tail-biting rings over the mul1 codebook.
    packed [N, 1 + dim*bits/16] int16: word 0 = fp16 scale; words 1.. = ring bitstream, little-endian
    uint16, LSB-first; stream bits [i*bits, (i+1)*bits) are the low bits of position i's state, whose
    higher bits are the symbols of positions i-1, i-2, ... (mod dim).
    row[i] = codebook(state_i) * scale + head_bias[head]."""
    dev = packed.device
    words = packed.shape[1] - 1
    dim = words * 16 // bits
    scale = packed[:, 0].contiguous().view(torch.float16).float()
    stream = packed[:, 1:].to(torch.int64) & 0xFFFF
    b = torch.arange(dim, device=dev) * bits
    window = stream[:, b >> 4] | (stream[:, ((b >> 4) + 1) % words] << 16)
    sym = (window >> (b & 15)) & ((1 << bits) - 1)
    state = torch.zeros_like(sym)
    for j in range((15 + bits) // bits):
        state |= torch.roll(sym, j, dims=1) << (j * bits)
    state &= 0xFFFF
    return mul1_codebook(dev)[state] * scale.unsqueeze(1) + head_bias_rows.float()
