"""Validate our EXL3 decoder against the original BF16 weights of the NVIDIA checkpoint."""
import sys, json, os, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen38.exl3 import decode_weight

EXL = os.environ.get("QWEN_MODEL", "models/qwen38fn-exl3-4.05bpw")
NV = os.environ.get("QWEN_REF_NVFP4")   # nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot (original BF16 tensors)
if not NV: sys.exit("set QWEN_REF_NVFP4 to an nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot")
ei = json.load(open(f"{EXL}/model.safetensors.index.json"))["weight_map"]
ni = json.load(open(f"{NV}/model.safetensors.index.json"))["weight_map"]

def get(root, idx, k, dev="cuda"):
    with safe_open(f"{root}/{idx[k]}", "pt", device="cpu") as f:
        return f.get_tensor(k).to(dev)

for name in sys.argv[1:] or [
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
    "model.language_model.layers.5.mlp.shared_expert.down_proj",
    "lm_head",
]:
    tr = get(EXL, ei, name + ".trellis"); suh = get(EXL, ei, name + ".suh"); svh = get(EXL, ei, name + ".svh")
    mul1 = get(EXL, ei, name + ".mul1", "cpu")
    w = decode_weight(tr, suh, svh, mul1)                      # [in, out]
    ref = get(NV, ni, name + ".weight").float().t()            # HF Linear [out, in] -> [in, out]
    assert w.shape == ref.shape, (w.shape, ref.shape)
    err = (w - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(w.flatten(), ref.flatten(), dim=0)
    print(f"{name}: bits {tr.shape[-1]//16} shape {tuple(w.shape)} rel_rms {err:.4f} cos {cos:.6f}")
