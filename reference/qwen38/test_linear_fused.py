"""qwen_exl3_linear (one launch) vs had_in + gemv + finish: bitwise, and time, on real weights."""
import sys, os, torch, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.model import Checkpoint
from qwen38 import kern as L
ck = Checkpoint("models/qwen38fn-exl3-4.05bpw")
I64 = ctypes.c_int64
for key in ("model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.out_proj",
            "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.0.mlp.shared_expert.down_proj", "lm_head"):
    tr = ck.raw(key + ".trellis"); suh = ck.raw(key + ".suh"); svh = ck.raw(key + ".svh")
    K = tr.shape[0] * 16; N = tr.shape[1] * 16; bits = tr.shape[2] // 16
    S = max(1, min(512 // max(N // 128, 1), 16, K // 16 // 8))
    cnt = torch.zeros(N // 128, dtype=torch.int32, device="cuda")
    ok = True; tt = []
    for M in (1, 4, 11, 16, 17, 33, 64):
        x = torch.randn(M, K, device="cuda")
        xh = torch.empty(M, K, device="cuda", dtype=torch.half); part = torch.empty(S, M, N, device="cuda"); y0 = torch.empty(M, N, device="cuda")
        def old():
            L.ck(L.lib.qwen_exl3_had_in(L.p(x), 1, I64(K), L.p(suh), L.p(xh), M, K, L.st()))
            L.ck(L.lib.qwen_exl3_gemv(L.p(xh), L.p(tr), L.p(part), M, K, N, bits, S, L.st()))
            L.ck(L.lib.qwen_exl3_finish(L.p(part), S, L.p(svh), L.p(y0), 1, I64(N), M, N, L.st()))
        y1 = torch.empty(M, N, device="cuda"); part1 = torch.empty(S, M, N, device="cuda")
        def new():
            L.ck(L.lib.qwen_exl3_linear(L.p(x), I64(K), L.p(suh), L.p(tr), L.p(svh), L.p(part1), L.p(y1), I64(N), L.p(cnt), M, K, N, bits, S, L.st()))
        old(); new(); torch.cuda.synchronize()
        eq = torch.equal(y0, y1) and cnt.abs().sum().item() == 0
        ok &= eq
        ts = {}
        for name, f in (("old", old), ("new", new)):
            for _ in range(3): f()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
            for _ in range(50): f()
            e1.record(); torch.cuda.synchronize(); ts[name] = e0.elapsed_time(e1) / 50 * 1e3
        tt.append("M%d:%.0f/%.0fus%s" % (M, ts["old"], ts["new"], "" if eq else " DIFF"))
    print(key.split(".")[-1], f"K={K} N={N} S={S} bitwise={ok}", " ".join(tt))
