import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint, GDN, rms, causal_conv, EPS
from qwen38 import kern
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
g = GDN(ck, "model.language_model.layers.0.linear_attn")
torch.manual_seed(0)
T = 37
x = torch.randn(T, 2560, device="cuda") * 0.5

def ref_o(x):  # reference GDN up to the gated norm (before out_proj)
    y = F.silu(causal_conv(x @ g.qkv, g.conv))
    q, k, v = y[:, :2048].view(T_, 16, 128), y[:, 2048:4096].view(T_, 16, 128), y[:, 4096:].view(T_, 48, 128)
    gg = -torch.exp(g.A_log) * F.softplus(x @ g.a + g.dt_bias); beta = torch.sigmoid(x @ g.b)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * 128 ** -0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
    q, k = q.repeat_interleave(3, 1), k.repeat_interleave(3, 1)
    S = torch.zeros(48, 128, 128, device=x.device); o = torch.empty(T_, 48, 128, device=x.device)
    for t in range(T_):
        S = S * torch.exp(gg[t]).view(48, 1, 1)
        vp = beta[t].view(48, 1) * (v[t] - torch.einsum("hvk,hk->hv", S, k[t]))
        S = S + vp.unsqueeze(2) * k[t].unsqueeze(1)
        o[t] = torch.einsum("hvk,hk->hv", S, q[t])
    z = (x @ g.z).view(T_, 48, 128)
    return (rms(o, g.norm) * torch.sigmoid(z)).reshape(T_, 6144), S

T_ = T
ref, Sref = ref_o(x)
S = torch.zeros(48, 128, 128, device="cuda"); cs = torch.zeros(10240, 3, device="cuda")
out, _ = kern.gdn_layer(g, x, S, cs, proj=kern.gdn_proj(g, x))
rel = lambda a, b: float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())
print(f"prefill T={T}: out rel {rel(out, ref):.2e}  state rel {rel(S, Sref):.2e}")
# chained: rows in pieces (5, then verify 6 without write + commit 4, then rest) must equal one pass
S2 = torch.zeros_like(S); cs2 = torch.zeros_like(cs); outs = []
PR = kern.gdn_proj(g, x)
sl = lambda a, b: tuple(t[a:b].contiguous() for t in PR)
o1, _ = kern.gdn_layer(g, x[:5], S2, cs2, proj=sl(0,5)); outs.append(o1)
ov, _ = kern.gdn_layer(g, x[5:11], S2, cs2, write=False, proj=sl(5,11))            # verify 6 rows, nothing written
oc, _ = kern.gdn_layer(g, x[5:9], S2, cs2, write=True, proj=sl(5,9))              # commit 4 accepted rows (replay)
print("verify rows == replayed rows:", torch.equal(ov[:4], oc))
o3, _ = kern.gdn_layer(g, x[9:], S2, cs2, proj=sl(9,T)); outs += [oc, o3]
chained = torch.cat(outs)
print("chained == one pass:", torch.equal(chained, out), " state equal:", torch.equal(S2, S), torch.equal(cs2, cs))
