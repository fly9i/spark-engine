import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint, HC
from qwen38 import kern
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
rel = lambda u, v: float((u - v).pow(2).mean().sqrt() / v.pow(2).mean().sqrt())
for key, inj in (("model.language_model.layers.5.attn_hyper_connection", True), ("model.language_model.hyper_connection_mixer", False)):
    ref, k = HC(ck, key, inj), kern.HcK(ck, key, inj)
    for R in (1, 4, 16):
        X = torch.randn(R, 4, 2560, device="cuda") * torch.rand(1, 4, 1, device="cuda") * 3
        pr, mr = ref.mix(X); pk, mk = k.mix(X); pk2, mk2 = k.mix(X)
        print(f"{key.split('.')[-1][:24]:24s} R={R:2d} mixed rel {rel(mk, mr):.2e}" + (f" post rel {rel(pk, pr):.2e}" if inj else "") + f" det {torch.equal(mk, mk2)}")
    X = torch.randn(1, 4, 2560, device="cuda")
    for _ in range(3): k.mix(X)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(100): k.mix(X)
    torch.cuda.synchronize(); dt = (time.time() - t) / 100
    print(f"   R=1 {dt*1e6:.0f} us  -> {(k.W.numel() + k.up.numel())*2/dt/1e9:.0f} GB/s")
