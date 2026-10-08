import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38 import kern
L = kern
torch.manual_seed(0)
for pos0, R in ((100, 4), (2040, 16), (5000, 3), (40001, 8)):
    nb = (pos0 + R) // 4 + 1
    qi = torch.randn(R, 4, 128, device="cuda")
    pooled = torch.randn(nb, 128, device="cuda").bfloat16()
    pooled[7] = pooled[9]                                # create exact ties
    ld = nb
    scores = torch.empty(R, ld, device="cuda"); sel = torch.full((R, 2051), -1, dtype=torch.int32, device="cuda")
    cnt = torch.empty(R, dtype=torch.int32, device="cuda")
    L.ck(L.lib.qwen_qsa_select(L.p(qi), L.p(pooled), R, pos0, L.p(scores), ld, L.p(sel), L.p(cnt), 2051, L.st()))
    ok = True
    for r in range(R):
        p = pos0 + r; vis = (p + 1) // 4
        s = torch.relu(torch.einsum("hd,nd->hn", qi[r], pooled[:vis].float())).sum(0) / 128 ** 0.5
        if vis <= 512:
            ref = set(range(p + 1))
        else:
            # deterministic reference: score desc, index asc
            order = sorted(range(vis), key=lambda g: (-float(s[g]), g))[:512]
            ref = {4 * g + i for g in order for i in range(4)} | set(range(4 * vis, p + 1))
        got = sel[r, :int(cnt[r])].tolist()
        if set(got) != ref or len(got) != len(ref) or got != sorted(got):
            ok = False
            print("  row", r, "mismatch", len(got), len(ref), len(set(got) ^ ref))
    print(f"pos0 {pos0} R {R}: {'OK' if ok else 'FAIL'}  cnt {cnt.tolist()[:4]}")
