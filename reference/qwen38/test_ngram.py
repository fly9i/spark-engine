"""Compare decoded EXL3 n-gram rows with the NVIDIA FP8 table (same row ids)."""
import sys, os, json, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.exl3 import ngram_rows
EXL = os.path.join(os.environ.get("QWEN_MODEL", "models/qwen38fn-exl3-4.05bpw"), "ngram_embedding.safetensors")
NV = os.path.join(os.environ.get("QWEN_REF_NVFP4") or sys.exit("set QWEN_REF_NVFP4"), "model-fp8-mtp-ple.safetensors")
p = "model.language_model.layers.1.ple.ple_embedding."
with safe_open(EXL, "pt") as f:
    off = f.get_tensor(p + "ngram_embedding.head_offsets"); sizes = f.get_tensor(p + "ngram_embedding.head_vocab_sizes")
    mult = f.get_tensor(p + "ngram_embedding.layer_multipliers"); bias = f.get_tensor(p + "ngram_embedding.head_bias")
    tr = f.get_slice(p + "ngram_embedding.trellis")
    print("multipliers", mult.tolist()); print("offsets", off.tolist()); print("sizes", sizes.tolist())
    g = torch.Generator().manual_seed(0)
    heads = torch.randint(0, 16, (64,), generator=g)
    rows = torch.stack([off[h] + torch.randint(0, int(sizes[h]), (1,), generator=g)[0] for h in heads])
    packed = torch.stack([tr[int(r):int(r) + 1][0] for r in rows])
    ours = ngram_rows(packed, 6, bias[heads])
NVD = os.path.dirname(NV)
nvi = json.load(open(f"{NVD}/model.safetensors.index.json"))["weight_map"]
def nvget(k, sl=None):
    with safe_open(f"{NVD}/{nvi[k]}", "pt") as f:
        return f.get_tensor(k) if sl is None else f.get_slice(k)[sl[0]:sl[1]]
scale = nvget(p + "ngram_embedding.weight_scale").float()
with safe_open(f"{NVD}/{nvi[p + 'ngram_embedding.shard_0.weight']}", "pt") as f:
    shard_rows = f.get_slice(p + "ngram_embedding.shard_0.weight").get_shape()[0]
ref = []
for r in rows.tolist():
    s_, i = divmod(r, shard_rows)
    ref.append(nvget(p + f"ngram_embedding.shard_{s_}.weight", (i, i + 1))[0].float())
ref = torch.stack(ref) * scale
err = (ours - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
cos = torch.nn.functional.cosine_similarity(ours, ref, dim=1)
print(f"rows {len(rows)} rel_rms {err:.4f} cos min {cos.min():.4f} mean {cos.mean():.4f}")
