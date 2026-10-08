"""Kernel-path prototype of the Qwen3.8 forward (the engine's kernels via ctypes, torch glue for HC/PLE).

Processes a sequence in chains of <= 16 rows with incremental state (GDN state + conv window, FP8 KV,
pooled indexer keys, PLE conv window), exactly the decode/verify data flow the Rust engine will run.
"""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from .model import Checkpoint, rms, P, EPS
from .exl3 import ngram_rows
from . import kern

ROOT = "models/qwen38fn-exl3-4.05bpw"


class Lin:
    """EXL3 linear (trellis/suh/svh resident) or plain fp32 matrix [in, out]."""
    def __init__(self, ck, key):
        if ck.has(key + ".trellis"):
            self.q = (ck.raw(key + ".trellis"), ck.raw(key + ".suh"), ck.raw(key + ".svh"))
            self.w = None
        else:
            self.q, self.w = None, ck.raw(key + ".weight").float().t().contiguous()

    def __call__(self, x):
        if self.w is not None:
            return x @ self.w
        out = []
        for i in range(0, x.shape[0], 16):
            out.append(kern.exl3_linear(x[i:i + 16].contiguous(), *self.q))
        return torch.cat(out)


class HCf:
    def __init__(self, ck, key, inject=True):
        self.norm = 1.0 + ck.raw(key + ".hc_norm.weight").float().view(4, -1)
        down = ck.raw(key + ".input_mix_weight_down.weight").float()
        if inject:
            down = torch.cat((down, ck.raw(key + ".block_inject_weight.weight").float()))
        self.down = down.t().contiguous()                                  # [10240, 324]
        self.up = ck.raw(key + ".input_mix_weight_up.weight").float().t().contiguous()   # [320, 10240]
        self.inject = inject

    def mix(self, X):
        n = rms(X) * self.norm
        d = n.flatten(1) @ self.down
        g = torch.sigmoid(F.silu(d[:, :320] / 4) @ self.up).view_as(n)
        mixed = (g * n).mean(1)
        post = 2 * torch.sigmoid(d[:, 320:] / 4) if self.inject else None
        return post, mixed


class GDNf:
    def __init__(self, ck, key):
        self.qkv, self.z, self.o = Lin(ck, key + ".in_proj_qkv"), Lin(ck, key + ".in_proj_z"), Lin(ck, key + ".out_proj")
        self.ab = torch.cat((ck.raw(key + ".in_proj_a.weight"), ck.raw(key + ".in_proj_b.weight"))).float().t().contiguous()
        self.conv = ck.raw(key + ".conv1d.weight").contiguous()
        self.A_log, self.dt_bias, self.norm = (ck.raw(f"{key}.{n}").contiguous() for n in ("A_log", "dt_bias", "norm.weight"))

    def new_state(self):
        return [torch.zeros(48, 128, 128, device="cuda"), torch.zeros(10240, 3, device="cuda")]

    def __call__(self, x, state):
        T = x.shape[0]
        qkv, z, ab = self.qkv(x).contiguous(), self.z(x).contiguous(), (x @ self.ab).contiguous()
        y = torch.empty(T, 10240, device="cuda"); out = torch.empty(T, 6144, device="cuda")
        S, cs = state
        kern.ck(kern.lib.qwen_gdn_conv(kern.p(qkv), kern.ctypes.c_int64(10240), kern.p(cs), kern.p(self.conv), kern.p(y), T, kern.st()))
        kern.ck(kern.lib.qwen_gdn_recur(kern.p(y), kern.p(ab), kern.ctypes.c_int64(96), kern.p(z), kern.ctypes.c_int64(6144),
                                        kern.p(self.A_log), kern.p(self.dt_bias), kern.p(self.norm), kern.p(S), kern.p(out), T, 1, kern.st()))
        kern.ck(kern.lib.qwen_gdn_conv_commit(kern.p(qkv), kern.ctypes.c_int64(10240), kern.p(cs), T, kern.st()))
        return self.o(out)


class QSAf:
    def __init__(self, ck, key):
        self.q, self.k, self.v, self.o = (Lin(ck, f"{key}.{n}_proj") for n in "qkvo")
        self.idx = Lin(ck, key + ".indexer.index_qk_proj")
        self.qn, self.kn = ck.raw(key + ".q_norm.weight"), ck.raw(key + ".k_norm.weight")
        self.iqn, self.ikn = ck.raw(key + ".indexer.q_layernorm.weight"), ck.raw(key + ".indexer.k_layernorm.weight")

    def new_state(self, cap):
        return kern.QsaCache(cap)

    def __call__(self, x, pos0, c):
        R = x.shape[0]
        qp, kp, vp, ip = (self.q(x).contiguous(), self.k(x).contiguous(), self.v(x).contiguous(), self.idx(x).contiguous())
        q = torch.empty(R, 24, 256, device="cuda"); gate = torch.empty_like(q); qi = torch.empty(R, 4, 128, device="cuda")
        L = kern
        L.ck(L.lib.qwen_qsa_prep(L.p(qp), L.p(kp), L.p(vp), L.p(ip), R, pos0, L.p(self.qn), L.p(self.kn), L.p(self.iqn),
                                 L.p(q), L.p(gate), L.p(c.kc), L.p(c.ks), L.p(c.vc), L.p(c.vs), L.p(qi), L.p(c.ring), L.st()))
        L.ck(L.lib.qwen_qsa_pool(L.p(c.ring), L.p(ip), R, pos0, L.p(self.ikn), L.p(c.pooled), L.st()))
        n = pos0 + R
        assert n <= 2048 + 3, "selection (top-512) not wired into the prototype yet"
        splits = (n + 127) // 128
        ml = torch.empty(R * 24 * splits * 2, device="cuda"); acc = torch.empty(R * 24 * splits * 256, device="cuda")
        out = torch.empty(R, 6144, device="cuda")
        L.ck(L.lib.qwen_qsa_attn(L.p(q), R, pos0, None, None, 0, L.p(c.kc), L.p(c.ks), L.p(c.vc), L.p(c.vs), L.p(gate),
                                 L.p(ml), L.p(acc), splits, L.p(out), L.st()))
        return self.o(out)


class MoEf:
    def __init__(self, ck, key):
        self.gate = ck.raw(key + ".gate.weight").float().t().contiguous()            # [2560, 512]
        self.sg = ck.raw(key + ".shared_expert_gate.weight").float().t().contiguous()  # [2560, 1]
        self.sh = [Lin(ck, f"{key}.shared_expert.{n}_proj") for n in ("gate", "up", "down")]
        self.table = kern.ExpertTable(ck, key)

    def __call__(self, x):
        g, u, d = self.sh
        shared = torch.sigmoid(x @ self.sg) * d(F.silu(g(x)) * u(x))
        out, _, _ = kern.moe_routed(x.contiguous(), (x @ self.gate).contiguous(), self.table, add=shared.contiguous())
        return out


class PLEf:
    def __init__(self, ck, key, root, eos):
        from .model import PLE
        self.ref = PLE(ck, key, f"{root}/ngram_embedding.safetensors", eos)

    def new_state(self):
        return torch.zeros(9, 10240, device="cuda")      # last 9 normed rows (conv taps t-3, t-6, t-9)

    def __call__(self, X, ids_hist, R, conv_win):
        """X [R,4,2560]; ids_hist: all token ids of the sequence up to the last row (host tensor)."""
        r = self.ref
        rows = r.ngram_ids(ids_hist)[-R:]
        uniq, inv = torch.unique(rows.flatten(), return_inverse=True)
        heads = torch.bucketize(uniq, r.off, right=True) - 1
        packed = torch.stack([r.table[int(u):int(u) + 1][0] for u in uniq]).to("cuda")
        emb = ngram_rows(packed, r.K, r.bias[heads.to("cuda")])[inv.to("cuda")].view(R, 2560)
        key = rms((emb @ r.kp).view(R, 4, 2560)) * (1.0 + r.nk)
        value = emb @ r.vp
        query = rms(X) * (1.0 + r.nq)
        dd = (key * query).sum(-1) / math.sqrt(2560)
        gate = torch.sigmoid(torch.sign(dd) * torch.sqrt(dd.abs().clamp_min(EPS)))
        gated = gate.unsqueeze(-1) * value.unsqueeze(1)
        normed = (rms(gated) * (1.0 + r.nc)).flatten(1)                      # [R, 10240]
        hist = torch.cat((conv_win, normed))                                  # [9 + R, 10240]
        w = r.conv.float().view(10240, 4)
        conv = torch.stack([w[:, 0] * hist[t] + w[:, 1] * hist[t + 3] + w[:, 2] * hist[t + 6] + w[:, 3] * hist[t + 9]
                            for t in range(R)])
        conv_win.copy_(hist[-9:])
        return gated + F.silu(conv).view(R, 4, 2560)


class FastModel:
    def __init__(self, root=ROOT):
        ck = self.ck = Checkpoint(root)
        c = ck.cfg
        self.embed = ck.raw(P + "embed_tokens.weight")
        self.layers = []
        for i in range(c["num_hidden_layers"]):
            key = f"{P}layers.{i}"
            lin = c["layer_types"][i] == "linear_attention"
            attn = GDNf(ck, key + ".linear_attn") if lin else QSAf(ck, key + ".self_attn")
            ple = PLEf(ck, key + ".ple", root, c["eos_token_id"]) if (i + 1) in c["ple_layer_ids"] else None
            self.layers.append((ple, HCf(ck, key + ".attn_hyper_connection"), attn, HCf(ck, key + ".mlp_hyper_connection"),
                                MoEf(ck, key + ".mlp")))
        self.mixer = HCf(ck, P + "hyper_connection_mixer", inject=False)
        self.lm_head = Lin(ck, "lm_head")

    def new_seq(self, cap=4096):
        st = []
        for ple, _, attn, _, _ in self.layers:
            st.append((ple.new_state() if ple else None, attn.new_state() if isinstance(attn, GDNf) else attn.new_state(cap)))
        return {"layers": st, "ids": [], "pos": 0}

    @torch.no_grad()
    def step(self, seq, ids):
        """Run rows `ids` (<= 16) at the sequence's current position, commit them; returns logits [R, vocab]."""
        R = len(ids)
        pos0 = seq["pos"]
        seq["ids"] += list(ids)
        hist = torch.tensor(seq["ids"], dtype=torch.long)
        X = self.embed[torch.tensor(ids, device="cuda")].float().unsqueeze(1).repeat(1, 4, 1)
        for (ple, hca, attn, hcm, moe), (pst, ast) in zip(self.layers, seq["layers"]):
            if ple is not None:
                X = X + ple(X, hist, R, pst)
            post, a_in = hca.mix(X)
            a = attn(a_in, ast) if isinstance(attn, GDNf) else attn(a_in, pos0, ast)
            X = X + post.unsqueeze(-1) * a.unsqueeze(1)
            post, m_in = hcm.mix(X)
            X = X + post.unsqueeze(-1) * moe(m_in).unsqueeze(1)
        seq["pos"] += R
        _, s = self.mixer.mix(X)
        return self.lm_head(s)
