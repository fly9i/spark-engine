# SPDX-License-Identifier: MIT
"""Export the engram n-gram hashing constants of DeepSeek-V4.1-Flash for the engine.

The official NgramHashState derives them from the tokenizer (compressed-token map) and the config (per layer and
n-gram/head primes, offsets, multipliers). The engine reads this file instead of re-implementing the tokenizer
normalization, so both sides use the same numbers by construction.

usage: export_engram.py --ckpt /models/DeepSeek-V4.1-Flash-1017 --out engram_hash.json [--check ids.json]
  needs torch, numpy, sympy, tokenizers; imports the checkpoint's own inference/model.py and engram.py, with kernel.py
  (this directory) standing in for its tilelang kernels.
  --check: also hash the ids of a sample with the official module and store the result for an engine self-test.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Tok:
    """What the official engram code needs from a HF tokenizer (`backend_tokenizer`, `len()`), plus encode, on the
    plain `tokenizers` library (no transformers dependency)."""

    def __init__(self, ckpt):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.ckpt, "inference"))
    sys.path.insert(0, HERE)
    import torch
    import model as M
    from engram import EngramLayout, NgramHashState

    cfg = json.load(open(os.path.join(a.ckpt, "inference", "config.json")))
    args = M.ModelArgs(**cfg, max_batch_size=1, max_seq_len=4096)
    tok = Tok(a.ckpt)
    layout = EngramLayout.from_args(args)
    st = NgramHashState(args, layout, tok)
    out = {
        "layer_ids": list(layout.layer_ids),
        "max_ngram_size": layout.max_ngram_size,
        "n_heads": layout.n_heads,
        "head_dim": layout.head_dim,
        "num_embeddings": list(layout.num_embeddings),
        "pad_id": int(st.pad_id),
        "primes": st.primes.tolist(),            # [n_layers][max_ngram-1][n_heads]
        "offsets": st.offsets.tolist(),          # [n_layers][(max_ngram-1)*n_heads]
        "multipliers": st.multipliers.tolist(),  # [n_layers][max_ngram]
        "token_map": st.token_map.tolist(),      # [vocab] compressed id per token
        "compressed_vocab_size": args.engram_compressed_vocab_size,
    }
    if a.check:
        spec = json.load(open(a.check))
        ids = spec["ids"] if "ids" in spec else tok.encode(spec["text"])
        h = st(torch.tensor([ids]), 0, None)[0]        # [L, n_layers, cols]
        out["check"] = {"ids": ids, "hashes": h.tolist()}
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out, "vocab", len(out["token_map"]), "layers", out["layer_ids"])


if __name__ == "__main__":
    main()
