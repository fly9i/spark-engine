# SPDX-License-Identifier: MIT
"""120-shard safetensors 索引 + 张量读取(仅依赖 numpy,memmap 零拷贝)。

索引一次扫描全部 header(120 × ~500KB,数秒),缓存到 <model>/.engine-index.json。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "BF16": np.dtype("<u2"), "F16": np.dtype("<f2"), "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "I16": np.dtype("<i2"),
    "U8": np.dtype("u1"),
}
# BF16 视为 u2 读入后由调用方 view 为 bf16 语义(numpy 2 无 bfloat16)。


class ShardIndex:
    def __init__(self, model_dir: str | Path, *, use_cache: bool = True):
        self.dir = Path(model_dir)
        cache = self.dir / ".engine-index.json"
        if use_cache and cache.exists():
            self.entries = json.load(open(cache))
        else:
            self.entries = self._scan()
            json.dump(self.entries, open(cache, "w"))
        self._mmaps: dict[str, np.memmap] = {}

    def _scan(self) -> dict:
        entries = {}
        for f in sorted(self.dir.glob("model-*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                entries[name] = [f.name, base + meta["data_offsets"][0],
                                 meta["data_offsets"][1] - meta["data_offsets"][0],
                                 meta["dtype"], meta["shape"]]
        return entries

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.entries if n.startswith(prefix)]

    def get(self, name: str) -> np.ndarray:
        fname, off, nbytes, dtype, shape = self.entries[name]
        mm = self._mmaps.get(fname)
        if mm is None:
            mm = np.memmap(self.dir / fname, dtype=np.uint8, mode="r")
            self._mmaps[fname] = mm
        dt = _DTYPES[dtype]
        if dtype in ("BF16",):
            return mm[off:off + nbytes].view(dt).reshape(shape)
        return mm[off:off + nbytes].view(dt).reshape(shape)


def find_model_dir(hub_root: str | Path, repo_id: str) -> Path:
    """hub_root/models--org--name/snapshots/<rev>(取最新 snapshot)。"""
    snap = Path(hub_root) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    revs = sorted(snap.iterdir())
    if not revs:
        raise FileNotFoundError(snap)
    return revs[-1]
