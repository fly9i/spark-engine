import sys, os, torch, torch.nn.functional as F, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
from qwen38.model import Checkpoint, MoE
from qwen38 import kern
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
key = "model.language_model.layers.7.mlp"
cache = OrderedDict(); cache.limit = 400
moe = MoE(ck, key, cache)
tab = kern.ExpertTable(ck, key)
rel = lambda u, v: float((u - v).pow(2).mean().sqrt() / v.pow(2).mean().sqrt())
torch.manual_seed(0)
for R in (1, 4, 16):
    x = torch.randn(R, 2560, device="cuda") * 0.3
    logits = x @ moe.gate.t()
    out, idx, w = kern.moe_routed(x, logits, tab)
    p = torch.softmax(logits, -1); wr, ir = torch.topk(p, 10, -1); wr = wr / wr.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for r in range(R):
        for k in range(10):
            g, u, d = moe.expert(int(ir[r, k]))
            ref[r] += wr[r, k] * ((F.silu(x[r] @ g) * (x[r] @ u)) @ d)
    out2, _, _ = kern.moe_routed(x, logits, tab)
    print(f"R={R:2d} routing equal {torch.equal(idx.long(), ir)} w rel {rel(w, wr):.1e} out rel {rel(out, ref):.2e} deterministic {torch.equal(out, out2)}")
x = torch.randn(1, 2560, device="cuda") * 0.3; logits = x @ moe.gate.t()
for _ in range(3): kern.moe_routed(x, logits, tab)
torch.cuda.synchronize(); t = time.time()
for _ in range(50): kern.moe_routed(x, logits, tab)
torch.cuda.synchronize(); print(f"R=1 routed MoE {(time.time()-t)/50*1e6:.0f} us (incl. python launch overhead)")
