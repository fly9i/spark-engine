"""Qwen3.8-Flash-Next reference forward (fp32, full-sequence recompute, readable).

Written from the official model definition (see docs/03a-Qwen3.8架构参考.md); weights come from the
EXL3 checkpoint and are decoded exactly (qwen38.exl3). Slow by design: it is the ground truth the
engine's kernels are compared against, layer by layer.
"""
from __future__ import annotations
import json, math, os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from safetensors import safe_open
from .exl3 import decode_weight, ngram_rows
from . import memguard  # noqa: F401  (starts the watchdog)

EPS = 1e-6
P = "model.language_model."


class Checkpoint:
    def __init__(self, root: str, device="cuda"):
        self.root, self.dev = root, device
        self.idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self.cfg = json.load(open(f"{root}/config.json"))["text_config"]
        self._files = {}

    def _f(self, name):
        if name not in self._files:
            self._files[name] = safe_open(f"{self.root}/{name}", "pt", device=self.dev)
        return self._files[name]

    def has(self, k):
        return k in self.idx

    def raw(self, k):
        return self._f(self.idx[k]).get_tensor(k)

    def lin(self, k):
        """Linear weight as fp32 [in, out] (y = x @ W): EXL3 decoded or plain (HF [out, in])."""
        if self.has(k + ".trellis"):
            return decode_weight(self.raw(k + ".trellis"), self.raw(k + ".suh"), self.raw(k + ".svh"),
                                 self._f(self.idx[k + ".mul1"]).get_tensor(k + ".mul1").cpu())
        return self.raw(k + ".weight").float().t()


def rms(x, w=None, one_plus=False):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    if w is None:
        return y
    w = w.float()
    return y * (1.0 + w if one_plus else w)


def rope_neox(x, pos, rot=64, theta=1e7):
    """NeoX partial rotary on the first `rot` dims of the last axis; x [T, ..., D], pos [T]."""
    half = rot // 2
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float64) * 2 / rot)
    ang = (pos.double().unsqueeze(1) * inv.unsqueeze(0)).float()          # [T, half]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [half]
    cos, sin = ang.cos().view(shape), ang.sin().view(shape)
    x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), -1)


class HC:
    """Gated residual hyper-connection site (4 streams, elementwise gated collapse)."""

    def __init__(self, ck: Checkpoint, key: str, inject=True):
        self.norm = ck.raw(key + ".hc_norm.weight").float().view(4, -1)
        self.down = ck.raw(key + ".input_mix_weight_down.weight").float()      # [320, 10240]
        self.up = ck.raw(key + ".input_mix_weight_up.weight").float()          # [10240, 320]
        self.inject = ck.raw(key + ".block_inject_weight.weight").float() if inject else None

    def mix(self, X):
        """X [T,4,D] -> (post [T,4] or None, mixed [T,D])."""
        n = rms(X) * (1.0 + self.norm)
        flat = n.flatten(1)
        t = F.silu(flat @ self.down.t() / 4)
        g = torch.sigmoid(t @ self.up.t()).view_as(n)
        mixed = (g * n).mean(1)
        post = 2 * torch.sigmoid(flat @ self.inject.t() / 4) if self.inject is not None else None
        return post, mixed


def causal_conv(x, w, dilation=1):
    """Depthwise causal conv over time; x [T, C], w [C, 1, k] (w[..., -1] multiplies the current step)."""
    k = w.shape[-1]
    xt = F.pad(x.t().unsqueeze(0), ((k - 1) * dilation, 0))
    return F.conv1d(xt, w.float(), groups=x.shape[1], dilation=dilation)[0].t()


class GDN:
    def __init__(self, ck, key):
        self.qkv, self.z = ck.lin(key + ".in_proj_qkv"), ck.lin(key + ".in_proj_z")
        self.a, self.b = ck.lin(key + ".in_proj_a"), ck.lin(key + ".in_proj_b")
        self.o = ck.lin(key + ".out_proj")
        self.conv = ck.raw(key + ".conv1d.weight")
        self.A_log = ck.raw(key + ".A_log").float()
        self.dt_bias = ck.raw(key + ".dt_bias").float()
        self.norm = ck.raw(key + ".norm.weight").float()

    def __call__(self, x, cap=None):
        T = x.shape[0]
        y = F.silu(causal_conv(x @ self.qkv, self.conv))
        q, k, v = y[:, :2048].view(T, 16, 128), y[:, 2048:4096].view(T, 16, 128), y[:, 4096:].view(T, 48, 128)
        g = -torch.exp(self.A_log) * F.softplus(x @ self.a + self.dt_bias)                 # [T, 48]
        beta = torch.sigmoid(x @ self.b)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * 128 ** -0.5
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
        q, k = q.repeat_interleave(3, 1), k.repeat_interleave(3, 1)                         # v head j -> k head j//3
        S = torch.zeros(48, 128, 128, device=x.device)                                    # [head, v, k]
        o = torch.empty(T, 48, 128, device=x.device)
        for t in range(T):
            S = S * torch.exp(g[t]).view(48, 1, 1)
            vp = beta[t].view(48, 1) * (v[t] - torch.einsum("hvk,hk->hv", S, k[t]))
            S = S + vp.unsqueeze(2) * k[t].unsqueeze(1)
            o[t] = torch.einsum("hvk,hk->hv", S, q[t])
        if cap is not None:
            cap["gdn_state"] = S
        z = (x @ self.z).view(T, 48, 128)
        o = rms(o, self.norm) * torch.sigmoid(z)
        return o.reshape(T, 6144) @ self.o


class QSA:
    def __init__(self, ck, key):
        self.q, self.k, self.v, self.o = (ck.lin(f"{key}.{n}_proj") for n in "qkvo")
        self.q_norm, self.k_norm = ck.raw(key + ".q_norm.weight"), ck.raw(key + ".k_norm.weight")
        self.idx = ck.lin(key + ".indexer.index_qk_proj")
        self.iq_norm = ck.raw(key + ".indexer.q_layernorm.weight")
        self.ik_norm = ck.raw(key + ".indexer.k_layernorm.weight")

    def select(self, x, pos):
        """Per query row: bool mask [T, T] of attended tokens (budget 2048 = 512 blocks of 4 + tail)."""
        T = x.shape[0]
        qk = x @ self.idx
        qi = rope_neox(rms(qk[:, :512].view(T, 4, 128), self.iq_norm, True), pos)
        kr = qk[:, 512:]
        nb = T // 4
        mask = torch.zeros(T, T, dtype=torch.bool, device=x.device)
        if nb:
            pooled = kr[: nb * 4].view(nb, 4, 128).mean(1)
            starts = torch.arange(nb, device=x.device) * 4
            kc = rope_neox(rms(pooled, self.ik_norm, True), pos[starts])
            score = F.relu(torch.einsum("thd,nd->thn", qi, kc)).sum(1) / math.sqrt(128)       # [T, nb]
        for p in range(T):
            vis = (p + 1) // 4
            if vis:
                blocks = torch.arange(vis, device=x.device) if vis <= 512 else torch.topk(score[p, :vis], 512).indices
                tok = (blocks.unsqueeze(1) * 4 + torch.arange(4, device=x.device)).flatten()
                mask[p, tok] = True
            mask[p, vis * 4: p + 1] = True
        return mask

    def __call__(self, x, pos, cap=None):
        T = x.shape[0]
        qg = (x @ self.q).view(T, 24, 512)
        q, gate = qg[..., :256], qg[..., 256:]
        k, v = (x @ self.k).view(T, 2, 256), (x @ self.v).view(T, 2, 256)
        q = rope_neox(rms(q, self.q_norm, True), pos)
        k = rope_neox(rms(k, self.k_norm, True), pos)
        mask = self.select(x, pos)
        if cap is not None:
            cap["qsa_mask"] = mask
        kk, vv = k.repeat_interleave(12, 1), v.repeat_interleave(12, 1)                      # q head h -> kv h//12
        s = torch.einsum("thd,shd->hts", q, kk) / 16.0
        s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
        o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), vv)
        o = o * torch.sigmoid(gate)
        return o.reshape(T, 6144) @ self.o


class MoE:
    def __init__(self, ck, key, cache):
        self.ck, self.key, self.cache = ck, key, cache
        self.gate = ck.raw(key + ".gate.weight").float()                                   # [512, 2560]
        self.sg = ck.raw(key + ".shared_expert_gate.weight").float()                       # [1, 2560]
        self.shared = [ck.lin(f"{key}.shared_expert.{n}_proj") for n in ("gate", "up", "down")]

    def expert(self, e):
        k = (self.key, e)
        if k not in self.cache:
            self.cache[k] = [self.ck.lin(f"{self.key}.experts.{e}.{n}_proj") for n in ("gate", "up", "down")]
            while len(self.cache) > self.cache.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(k)
        return self.cache[k]

    def __call__(self, x, cap=None):
        p = torch.softmax(x @ self.gate.t(), -1)
        w, idx = torch.topk(p, 10, -1)
        w = w / w.sum(-1, keepdim=True)
        if cap is not None:
            cap["route"] = idx
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            g, u, d = self.expert(e)
            xe = x[rows]
            out.index_add_(0, rows, (F.silu(xe @ g) * (xe @ u)) @ d * w[rows, slot].unsqueeze(1))
        g, u, d = self.shared
        return out + torch.sigmoid(x @ self.sg.t()) * ((F.silu(x @ g) * (x @ u)) @ d)


class PLE:
    def __init__(self, ck, key, ngram_file, eos):
        self.kp, self.vp = ck.lin(key + ".key_proj"), ck.lin(key + ".value_proj")
        self.nk, self.nq, self.nc = (ck.raw(f"{key}.{n}.weight").float().view(4, -1)
                                     for n in ("norm_key", "norm_query", "norm_conv"))
        self.conv = ck.raw(key + ".conv1d.weight")
        self.eos = eos
        self.f = safe_open(ngram_file, "pt", device="cpu")
        pre = key + ".ple_embedding.ngram_embedding."
        self.mult = self.f.get_tensor(pre + "layer_multipliers")
        self.off = self.f.get_tensor(pre + "head_offsets")
        self.size = self.f.get_tensor(pre + "head_vocab_sizes")
        self.bias = self.f.get_tensor(pre + "head_bias").to(ck.dev)
        self.table = self.f.get_slice(pre + "trellis")
        self.K = int(self.f.metadata()["K"])
        self.dev = ck.dev

    def ngram_ids(self, ids):
        """ids [T] (a whole sequence from position 0) -> [T, 16] global table rows (int64 hash wraps)."""
        T = ids.shape[0]
        pos = torch.arange(T)
        eos_pos = torch.where(ids == self.eos, pos, -1)
        prev = torch.cat((torch.tensor([-1]), torch.cummax(eos_pos, 0).values[:-1]))
        in_seg = pos - (prev + 1)

        def shifted(s):
            src = pos - s
            val = ids[src.clamp_min(0)]
            return torch.where((in_seg >= s) & (src >= 0), val, torch.full_like(val, self.eos))

        t0, t1, t2 = shifted(0), shifted(1), shifted(2)
        m = self.mult
        bi = t0 * m[0] ^ t1 * m[1]
        tri = bi ^ t2 * m[2]
        mixed = torch.stack([bi] * 8 + [tri] * 8, 1)
        return torch.remainder(mixed, self.size.view(1, 16)) + self.off.view(1, 16)

    def __call__(self, X, ids):
        T = X.shape[0]
        rows = self.ngram_ids(ids.cpu().long())
        uniq, inv = torch.unique(rows.flatten(), return_inverse=True)
        heads = torch.bucketize(uniq, self.off, right=True) - 1
        packed = torch.stack([self.table[int(r):int(r) + 1][0] for r in uniq]).to(self.dev)
        emb = ngram_rows(packed, self.K, self.bias[heads.to(self.dev)])[inv.to(self.dev)].view(T, 2560)
        key = rms((emb @ self.kp).view(T, 4, 2560)) * (1.0 + self.nk)
        value = emb @ self.vp
        query = rms(X) * (1.0 + self.nq)
        d = (key * query).sum(-1) / math.sqrt(2560)
        gate = torch.sigmoid(torch.sign(d) * torch.sqrt(d.abs().clamp_min(EPS)))
        gated = gate.unsqueeze(-1) * value.unsqueeze(1)                                    # [T, 4, 2560]
        normed = (rms(gated) * (1.0 + self.nc)).flatten(1)
        conv = F.silu(causal_conv(normed, self.conv, dilation=3)).view(T, 4, 2560)
        return gated + conv


class Model:
    def __init__(self, root, device="cuda", expert_cache=1200):
        ck = self.ck = Checkpoint(root, device)
        c = ck.cfg
        self.types = c["layer_types"]
        self.embed = ck.raw(P + "embed_tokens.weight")
        cache = OrderedDict()
        cache.limit = expert_cache
        self.layers = []
        for i in range(c["num_hidden_layers"]):
            key = f"{P}layers.{i}"
            attn = GDN(ck, key + ".linear_attn") if self.types[i] == "linear_attention" else QSA(ck, key + ".self_attn")
            ple = PLE(ck, key + ".ple", f"{root}/ngram_embedding.safetensors", c["eos_token_id"]) \
                if (i + 1) in c["ple_layer_ids"] else None
            self.layers.append((ple, HC(ck, key + ".attn_hyper_connection"), attn,
                                HC(ck, key + ".mlp_hyper_connection"), MoE(ck, key + ".mlp", cache)))
        self.mixer = HC(ck, P + "hyper_connection_mixer", inject=False)
        self.lm_head = ck.lin("lm_head")

    @torch.no_grad()
    def forward(self, ids, caps=None):
        """ids [T] (sequence from position 0) -> logits [T, vocab]; caps: list to collect per-layer
        captures (stream stack after each layer, attention/MoE details)."""
        ids = ids.to(self.ck.dev).long()
        T = ids.shape[0]
        pos = torch.arange(T, device=self.ck.dev)
        X = self.embed[ids].float().unsqueeze(1).repeat(1, 4, 1)
        for i, (ple, hca, attn, hcm, moe) in enumerate(self.layers):
            cap = {} if caps is not None else None
            if ple is not None:
                X = X + ple(X, ids)
            post, a_in = hca.mix(X)
            a = attn(a_in, cap) if isinstance(attn, GDN) else attn(a_in, pos, cap)
            X = X + post.unsqueeze(-1) * a.unsqueeze(1)
            post, m_in = hcm.mix(X)
            m = moe(m_in, cap)
            X = X + post.unsqueeze(-1) * m.unsqueeze(1)
            if caps is not None:
                cap["X"] = X.clone()
                caps.append(cap)
        _, s = self.mixer.mix(X)
        self.last_streams = X
        return s @ self.lm_head
