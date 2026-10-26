#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decode throughput: greedy, top_p 1, thinking off, 32-token warmup, 256 completion tokens per stream, streamed;
per-stream decode tok/s = (completion_tokens - 1) / (last - first content token); aggregate = all streams' tokens over
the window from the first stream's first token to the last stream's last token. Prompts: prose (hash map
explanation), structured (count 1..200), code.

usage: decode_bench.py [--url http://127.0.0.1:8888] [--types prose,structured,code] [--conc 1,2,4] [--max-tokens 256]
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "code": "lru_cache\nImplement an LRU cache class in Python with get, put, delete and resize.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
}


def stream(url, prompt, n):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0, "top_p": 1,
            "reasoning_effort": "none", "stream": True}
    t0 = time.time()
    first = last = None
    toks = 0
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data:") or b"[DONE]" in raw:
                continue
            d = json.loads(raw[5:])
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.time()
                first = first or now
                last = now
            if d.get("usage"):
                toks = d["usage"].get("completion_tokens", toks)
    return {"ttft_ms": (first - t0) * 1e3, "first": first, "last": last, "tokens": toks or n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--types", default="prose,structured,code")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    stream(a.url, PROMPTS["prose"], 32)   # warmup
    for kind in a.types.split(","):
        for c in [int(x) for x in a.conc.split(",")]:
            out = [None] * c
            # one prompt per stream with a distinct first line (no shared prefix-cache block)
            th = [threading.Thread(target=lambda i=i: out.__setitem__(i, stream(a.url, f"{i}\n{PROMPTS[kind]}", a.max_tokens))) for i in range(c)]
            [t.start() for t in th]
            [t.join() for t in th]
            per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out]
            win = max(o["last"] for o in out) - min(o["first"] for o in out)
            agg = sum(o["tokens"] - 1 for o in out) / win
            print(f"{kind:10s} c{c}: aggregate {agg:6.1f} tok/s, per stream {[round(p, 1) for p in per]}, TTFT {[round(o['ttft_ms']) for o in out]} ms")


if __name__ == "__main__":
    main()
