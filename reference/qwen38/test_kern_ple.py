import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint, PLE
from qwen38.fast import PLEf
from qwen38 import kern
ROOT = "models/qwen38fn-exl3-4.05bpw"
ck = Checkpoint(ROOT)
pf = PLEf(ck, "model.language_model.layers.1.ple", ROOT, 248044)
pk = kern.PleK(pf.ref)
rel = lambda u, v: float((u - v).pow(2).mean().sqrt() / v.pow(2).mean().sqrt())
torch.manual_seed(0)
ids = torch.randint(0, 248000, (40,)); ids[17] = 248044
wf, wk = torch.zeros(9, 10240, device="cuda"), torch.zeros(9, 10240, device="cuda")
for a, b in ((0, 16), (16, 32), (32, 40)):
    X = torch.randn(b - a, 4, 2560, device="cuda")
    yf = X + pf(X, ids[:b], b - a, wf)
    yk = pk(X.clone(), ids[:b], b - a, wk)
    print(f"rows {a}..{b}: X rel {rel(yk, yf):.2e}  window rel {rel(wk, wf):.2e}")
