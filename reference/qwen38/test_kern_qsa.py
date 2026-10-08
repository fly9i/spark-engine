import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint, QSA, rms, rope_neox
from qwen38 import kern
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
a = QSA(ck, "model.language_model.layers.3.self_attn")
torch.manual_seed(0)
T = 300
x = torch.randn(T, 2560, device="cuda") * 0.5
pos = torch.arange(T, device="cuda")
rel = lambda u, v: float((u - v).pow(2).mean().sqrt() / v.pow(2).mean().sqrt())

def fp8q(t):  # per (token, head) scale amax/448, e4m3 round trip
    sc = t.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0
    return (t / sc).to(torch.float8_e4m3fn).float() * sc

def ref_attn(x, kv_fp8):
    qg = (x @ a.q).view(T, 24, 512); q, gate = qg[..., :256], qg[..., 256:]
    k, v = (x @ a.k).view(T, 2, 256), (x @ a.v).view(T, 2, 256)
    q = rope_neox(rms(q, a.q_norm, True), pos); k = rope_neox(rms(k, a.k_norm, True), pos)
    if kv_fp8: k, v = fp8q(k), fp8q(v)
    kk, vv = k.repeat_interleave(12, 1), v.repeat_interleave(12, 1)
    s = torch.einsum("thd,shd->hts", q, kk) / 16.0
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril().unsqueeze(0), float("-inf"))
    o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), vv) * torch.sigmoid(gate)
    return o.reshape(T, 6144)

cache = kern.QsaCache(4096)
outs = []
for p0 in range(0, T, 16):                      # chains of up to 16 rows
    o, _ = kern.qsa_layer(a, x[p0:p0 + 16], p0, cache); outs.append(o)
out = torch.cat(outs)
print(f"vs fp8-emulated ref: {rel(out, ref_attn(x, True)):.2e}   vs fp32 ref: {rel(out, ref_attn(x, False)):.2e}")
# full layer incl. o_proj vs model.QSA (fp32 KV; T <= 2048 so selection = causal)
full = out @ a.o; refl = a(x, pos)
print(f"layer output vs reference model QSA: {rel(full, refl):.2e}")
# pooled indexer keys vs reference
qk = x @ a.idx; kr = qk[:, 512:]; nb = T // 4
kcref = rope_neox(rms(kr[:nb * 4].view(nb, 4, 128).mean(1).bfloat16().float(), a.ik_norm, True), pos[torch.arange(nb, device='cuda') * 4])
print(f"pooled keys rel {rel(cache.pooled[:nb].float(), kcref):.2e}")
