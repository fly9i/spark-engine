#!/usr/bin/env python3
"""Download the o_proj weights of an abliterated GLM-5.3-Flash donor for the engine's optional o_proj transplant
(GLM53_ABLIT=1, docs/abliteration.md). Only the needed tensors are fetched with HTTP range requests (about 2.7 GiB for
layers 15-44), into OUT/L{layer}.bin plus OUT/MANIFEST.json (key, dtype, shape, size, sha256) that the engine verifies.

Usage: fetch_ablit_transplant.py OUT [--donor dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4] [--revision main] [--layers 15-44]
Env:   HF_TOKEN (optional), HF_ENDPOINT (default https://huggingface.co)
Read the donor's model card and license before use; this script does not redistribute anything.
"""
import argparse, hashlib, json, os, struct, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--donor", default="dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4")
ap.add_argument("--revision", default="main")
ap.add_argument("--layers", default="15-44")
a = ap.parse_args()
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
token = os.environ.get("HF_TOKEN")
lo, hi = (int(v) for v in a.layers.split("-"))
os.makedirs(a.out, exist_ok=True)


def get(path, rng=None):
    req = urllib.request.Request(f"{endpoint}/{a.donor}/resolve/{a.revision}/{path}")
    if token: req.add_header("Authorization", f"Bearer {token}")
    if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


index = json.loads(get("model.safetensors.index.json"))["weight_map"]
headers = {}
manifest = {"donor": a.donor, "revision": a.revision, "method": "o_proj transplant (byte copy)", "tensors": {}}
mpath = os.path.join(a.out, "MANIFEST.json")
if os.path.exists(mpath):
    manifest["tensors"] = json.load(open(mpath)).get("tensors", {})
for layer in range(lo, hi + 1):
    key = next(k for k in index if k.endswith(f"layers.{layer}.self_attn.o_proj.weight"))
    shard = index[key]
    if shard not in headers:   # safetensors: 8-byte little-endian header length, then the JSON header
        n = struct.unpack("<Q", get(shard, (0, 7)))[0]
        headers[shard] = (8 + n, json.loads(get(shard, (8, 8 + n - 1))))
    base, h = headers[shard]
    meta = h[key]
    b0, b1 = meta["data_offsets"]
    dst = os.path.join(a.out, f"L{layer}.bin")
    ent = manifest["tensors"].get(str(layer))
    if ent and os.path.exists(dst) and os.path.getsize(dst) == b1 - b0 and \
            hashlib.sha256(open(dst, "rb").read()).hexdigest() == ent["sha256"]:
        print(f"layer {layer}: present"); continue
    data = get(shard, (base + b0, base + b1 - 1))
    assert len(data) == b1 - b0, f"short read for {key}"
    open(dst, "wb").write(data)
    manifest["tensors"][str(layer)] = {"key": key, "shard": shard, "dtype": meta["dtype"], "shape": meta["shape"],
                                       "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"layer {layer}: {key} {meta['dtype']} {meta['shape']} ({len(data) >> 20} MiB)")
print(f"done: {a.out} (set GLM53_ABLIT=1 GLM53_ABLIT_DIR={os.path.abspath(a.out)})")
