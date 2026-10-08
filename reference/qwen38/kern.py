"""ctypes bindings for the engine's Qwen CUDA kernels (built as one test .so) + test helpers."""
import ctypes, os, subprocess, torch
SHIM = "spark-engine/engine-rs/shim"
SO = "/tmp/qwen_kern_test.so"
SRC = ["qwen_exl3.cu", "qwen_gdn.cu", "qwen_qsa.cu", "qwen_moe.cu", "qwen_hc.cu", "qwen_ple.cu", "qwen_exl3_bench.cu", "qwen_f16.cu"]

def build(force=False):
    srcs = [f"{SHIM}/{s}" for s in SRC]
    if force or not os.path.exists(SO) or max(os.path.getmtime(s) for s in srcs) > os.path.getmtime(SO):
        subprocess.check_call(["/usr/local/cuda-13.0/bin/nvcc", "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                               "--fmad=false", "-gencode=arch=compute_121,code=sm_121", *srcs, "-o", SO])
    return ctypes.CDLL(SO)

lib = build()
def st(): return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
def p(t): return ctypes.c_void_p(t.data_ptr())
def ck(r):
    assert r == 0, f"kernel error {r}"

def exl3_linear(x, tr, suh, svh, S=None, out_f32=True):
    """x [M, K] fp32/fp16 -> y [M, N] via had_in + gemv + finish."""
    M, K = x.shape
    N = tr.shape[1] * 16
    bits = tr.shape[2] // 16
    if S is None:
        S = max(1, min(16, (K // 16) // 8, 96 // max(1, N // 128)))
    xh = torch.empty(M, K, dtype=torch.half, device=x.device)
    part = torch.empty(S, M, N, dtype=torch.float32, device=x.device)
    y = torch.empty(M, N, dtype=torch.float32 if out_f32 else torch.half, device=x.device)
    ck(lib.qwen_exl3_had_in(p(x), int(x.dtype == torch.float32), ctypes.c_int64(x.stride(0)), p(suh), p(xh), M, K, st()))
    ck(lib.qwen_exl3_gemv(p(xh), p(tr), p(part), M, K, N, bits, S, st()))
    ck(lib.qwen_exl3_finish(p(part), S, p(svh), p(y), int(out_f32), ctypes.c_int64(N), M, N, st()))
    return y


def gdn_layer(g, x, S, conv_state, write=True, proj=None):
    """Kernel path of one GDN layer for rows x [T, 2560] (projections in torch fp32 from the reference
    weights). S [48,128,128], conv_state [10240,3] fp32 (modified when write). Returns o_norm [T,6144]."""
    T = x.shape[0]
    if proj is None:
        proj = gdn_proj(g, x)
    qkv, z, ab = proj
    y = torch.empty(T, 10240, device=x.device)
    out = torch.empty(T, 6144, device=x.device)
    ck(lib.qwen_gdn_conv(p(qkv), ctypes.c_int64(10240), p(conv_state), p(g.conv), p(y), T, st()))
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    alog, dtb, nw = bf(g.A_log), bf(g.dt_bias), bf(g.norm)
    ck(lib.qwen_gdn_recur(p(y), p(ab), ctypes.c_int64(96), p(z), ctypes.c_int64(6144), p(alog), p(dtb),
                          p(nw), p(S), p(out), T, int(write), st()))
    if write:
        ck(lib.qwen_gdn_conv_commit(p(qkv), ctypes.c_int64(10240), p(conv_state), T, st()))
    return out, (qkv, y, ab, z)


def gdn_proj(g, x):
    return (x @ g.qkv).contiguous(), (x @ g.z).contiguous(), torch.cat((x @ g.a, x @ g.b), 1).contiguous()


class QsaCache:
    def __init__(self, cap, dev="cuda"):
        self.kc = torch.zeros(cap, 2, 256, dtype=torch.uint8, device=dev); self.ks = torch.ones(cap, 2, device=dev)
        self.vc = torch.zeros_like(self.kc); self.vs = torch.ones_like(self.ks)
        self.pooled = torch.zeros(cap // 4 + 1, 128, dtype=torch.bfloat16, device=dev)
        self.ring = torch.zeros(32, 128, device=dev)

def qsa_layer(a, x, pos0, cache):
    """Kernel path of a QSA layer for chain rows x [R, 2560] at positions pos0.. (projections in torch)."""
    R = x.shape[0]
    qp, kp, vp, ip = (x @ a.q).contiguous(), (x @ a.k).contiguous(), (x @ a.v).contiguous(), (x @ a.idx).contiguous()
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    q = torch.empty(R, 24, 256, device=x.device); gate = torch.empty_like(q); qi = torch.empty(R, 4, 128, device=x.device)
    ck(lib.qwen_qsa_prep(p(qp), p(kp), p(vp), p(ip), R, pos0, p(bf(a.q_norm)), p(bf(a.k_norm)), p(bf(a.iq_norm)),
                         p(q), p(gate), p(cache.kc), p(cache.ks), p(cache.vc), p(cache.vs), p(qi), p(cache.ring), st()))
    ck(lib.qwen_qsa_pool(p(cache.ring), p(ip), R, pos0, p(bf(a.ik_norm)), p(cache.pooled), st()))
    n = pos0 + R
    splits = (n + 127) // 128
    ml = torch.empty(R * 24 * splits * 2, device=x.device); acc = torch.empty(R * 24 * splits * 256, device=x.device)
    out = torch.empty(R, 6144, device=x.device)
    ck(lib.qwen_qsa_attn(p(q), R, pos0, None, None, 0, p(cache.kc), p(cache.ks), p(cache.vc), p(cache.vs), p(gate),
                         p(ml), p(acc), splits, p(out), st()))
    return out, dict(q=q, gate=gate, qi=qi)


lib.qwen_moe_ws_bytes.restype = ctypes.c_size_t

class ExpertTable:
    """All 512 experts of one layer resident on the GPU + the [9][512] pointer table."""
    def __init__(self, ck, key):
        self.t = {}
        rows = [[] for _ in range(9)]
        for e in range(512):
            for i, n in enumerate(("gate", "up", "down")):
                b = f"{key}.experts.{e}.{n}_proj"
                tr, suh, svh = ck.raw(b + ".trellis"), ck.raw(b + ".suh"), ck.raw(b + ".svh")
                self.t[(e, n)] = (tr, suh, svh)
                rows[i].append(tr.data_ptr()); rows[3 + i].append(suh.data_ptr()); rows[6 + i].append(svh.data_ptr())
        self.tab = torch.tensor(rows, dtype=torch.int64, device="cuda").contiguous()

def moe_routed(x, logits, table, S=4, add=None):
    R = x.shape[0]
    idx = torch.empty(R, 10, dtype=torch.int32, device=x.device); w = torch.empty(R, 10, device=x.device)
    ck(lib.qwen_moe_route(p(logits.contiguous()), R, p(idx), p(w), st()))
    ws = torch.empty(lib.qwen_moe_ws_bytes(R, S), dtype=torch.uint8, device=x.device)
    out = torch.empty(R, 2560, device=x.device)
    ck(lib.qwen_moe_experts(p(x), ctypes.c_int64(x.stride(0)), R, p(idx), p(w), p(table.tab),
                            p(add) if add is not None else None, p(out), p(ws), S, st()))
    return out, idx, w


lib.qwen_hc_ws_bytes.restype = ctypes.c_size_t

class HcK:
    """Kernel HC site: (1 + w) fp32, [down | inject] F16 [324, 10240], up F16 [10240, 320]."""
    def __init__(self, ck, key, inject=True):
        self.w1 = (1.0 + ck.raw(key + ".hc_norm.weight").float()).contiguous()
        d = ck.raw(key + ".input_mix_weight_down.weight")
        if inject:
            d = torch.cat((d, ck.raw(key + ".block_inject_weight.weight")))
        self.W = d.half().contiguous(); self.up = ck.raw(key + ".input_mix_weight_up.weight").half().contiguous()
        self.inject = inject

    def mix(self, X):
        R = X.shape[0]
        X = X.contiguous()
        mixed = torch.empty(R, 2560, device=X.device); post = torch.empty(R, 4, device=X.device)
        ws = torch.empty(lib.qwen_hc_ws_bytes(R), dtype=torch.uint8, device=X.device)
        ck(lib.qwen_hc_mix(p(X), R, p(self.w1), p(self.W), int(self.inject), p(self.up), p(mixed), p(post), p(ws), st()))
        return (post if self.inject else None), mixed


class PleK:
    """Kernel PLE: host hashing + ring gather (from the mmap'd table), device decode / gate / conv."""
    def __init__(self, ref):
        self.r = ref
        self.kp = ref.kp.contiguous(); self.vp = ref.vp.contiguous()                  # fp32 [2560, 10240] / [2560, 2560]
        self.nk1, self.nq1, self.nc1 = ((1.0 + t).flatten().contiguous() for t in (ref.nk, ref.nq, ref.nc))
        self.conv = ref.conv.contiguous(); self.bias = ref.bias.half().contiguous()

    def __call__(self, X, hist, R, win, commit=True):
        r = self.r
        rows = r.ngram_ids(hist)[-R:].flatten()
        packed = torch.stack([r.table[int(u):int(u) + 1][0] for u in rows]).cuda().contiguous()
        emb = torch.empty(R, 2560, device="cuda")
        ck(lib.qwen_ple_decode(p(packed), packed.shape[1], r.K, p(self.bias), p(emb), R, st()))
        key = (emb @ self.kp).contiguous(); value = (emb @ self.vp).contiguous()
        gated = torch.empty(R, 10240, device="cuda"); nrm = torch.empty(R, 10240, device="cuda")
        X = X.contiguous()
        ck(lib.qwen_ple_apply(p(X), p(key), p(value), p(self.nk1), p(self.nq1), p(self.nc1), p(self.conv), p(win),
                              p(gated), p(nrm), R, st()))
        if commit:
            ck(lib.qwen_ple_commit(p(win), p(nrm), R, None, st()))
        return X
