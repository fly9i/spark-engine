import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint
from qwen38.exl3 import decode_weight
from qwen38 import kern
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
names = ["model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.out_proj",
         "model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
         "model.language_model.layers.7.mlp.experts.17.gate_proj", "model.language_model.layers.7.mlp.experts.17.down_proj",
         "lm_head"]
torch.manual_seed(0)
for n in names:
    tr, suh, svh = ck.raw(n + ".trellis"), ck.raw(n + ".suh"), ck.raw(n + ".svh")
    W = ck.lin(n)
    for M in (1, 4, 16):
        x = torch.randn(M, W.shape[0], device="cuda")
        ref = x @ W
        y = kern.exl3_linear(x, tr, suh, svh)
        err = (y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
        y2 = kern.exl3_linear(x, tr, suh, svh)
        print(f"{n.split('language_model.')[-1]:48s} bits {tr.shape[2]//16} M {M:2d} rel_err {err:.2e} deterministic {torch.equal(y, y2)}")
    # timing at M=1
    x = torch.randn(1, W.shape[0], device="cuda")
    for _ in range(3): kern.exl3_linear(x, tr, suh, svh)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): kern.exl3_linear(x, tr, suh, svh)
    torch.cuda.synchronize(); dt = (time.time() - t) / 50
    print(f"   M=1 {dt*1e6:.0f} us  {tr.numel()*2/dt/1e9:.0f} GB/s (weights only)")
