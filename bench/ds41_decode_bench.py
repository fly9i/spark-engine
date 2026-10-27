#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decode throughput: greedy, top_p 1, thinking off, 32-token warmup, 256 completion tokens per stream, streamed;
per-stream decode tok/s = (completion_tokens - 1) / (last - first content token); aggregate = all streams' tokens over
the window from the first stream's first token to the last stream's last token. Prompts: prose (hash map
explanation), structured (count 1..200), code.

usage: decode_bench.py [--url http://127.0.0.1:8888] [--types prose,structured,code] [--conc 1,2,4] [--max-tokens 256]
                       [--temperature 0] [--effort none]   (temperature > 0: sampled, a fresh seed per request;
                       --types zh_prose,story,qa: open-ended texts, less predictable for the draft than the default three)
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "zh_prose": "写一篇八百字左右的散文，主题是秋天傍晚的城市，要有具体的场景、人物和细节。",
    "story": "Write an original short story about a lighthouse keeper who finds a mysterious letter washed ashore. Use vivid detail and dialogue.",
    "qa": "What were the main causes of the fall of the Western Roman Empire? Discuss the economic, military and political factors and how historians weigh them.",
    "code": "lru_cache\nImplement an LRU cache class in Python with get, put, delete and resize.\nOutput only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.",
}


TEMP = 0.0
EFFORT = "none"
DUMP = None   # --dump FILE: append every completion's text (one JSON line per request), to compare arms


def stream(url, prompt, n, seed=None):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": TEMP, "top_p": 1,
            "reasoning_effort": EFFORT, "stream": True}
    if seed is not None:
        body["seed"] = seed
    t0 = time.time()
    first = last = None
    toks = 0
    text = []
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data:") or b"[DONE]" in raw:
                continue
            d = json.loads(raw[5:])
            ch = d.get("choices") or []
            dl = (ch[0].get("delta") or {}) if ch else {}
            if dl.get("content") or dl.get("reasoning_content"):   # (thinking: the reasoning tokens count too)
                text.append((dl.get("reasoning_content") or "") + (dl.get("content") or ""))
                now = time.time()
                first = first or now
                last = now
            if d.get("usage"):
                toks = d["usage"].get("completion_tokens", toks)
    if DUMP:
        with open(DUMP, "a") as f:
            f.write(json.dumps({"prompt": prompt, "seed": seed, "text": "".join(text)}, ensure_ascii=False) + "\n")
    return {"ttft_ms": (first - t0) * 1e3, "first": first, "last": last, "tokens": toks or n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--types", default="prose,structured,code")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="none", help="reasoning_effort (none: thinking off)")
    ap.add_argument("--seed", type=int, default=None, help="request i of a run gets seed + i (sampled runs reproducible across arms)")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    global TEMP, EFFORT
    TEMP, EFFORT = a.temperature, a.effort
    global DUMP
    DUMP = a.dump
    stream(a.url, PROMPTS["prose"], 32)   # warmup
    for kind in a.types.split(","):
        for c in [int(x) for x in a.conc.split(",")]:
            out = [None] * c
            # one prompt per stream with a distinct first line (no shared prefix-cache block)
            th = [threading.Thread(target=lambda i=i: out.__setitem__(i, stream(a.url, f"{i}\n{PROMPTS[kind]}", a.max_tokens, None if a.seed is None else a.seed + i))) for i in range(c)]
            [t.start() for t in th]
            [t.join() for t in th]
            per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out]
            win = max(o["last"] for o in out) - min(o["first"] for o in out)
            agg = sum(o["tokens"] - 1 for o in out) / win
            print(f"{kind:10s} c{c}: aggregate {agg:6.1f} tok/s, per stream {[round(p, 1) for p in per]}, TTFT {[round(o['ttft_ms']) for o in out]} ms")


if __name__ == "__main__":
    main()
