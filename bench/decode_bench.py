#!/usr/bin/env python3
"""Decode speed of an OpenAI-compatible server at 1-4 concurrent streams.

Greedy (temperature 0), thinking off, 32-token warm-up, max_tokens 400 by default. Two prompt types:
  prose       long-form explanation (lower draft acceptance)
  structured  counting 1..200 (highly predictable, high draft acceptance)
Per stream: decode tok/s = (completion_tokens - 1) / (t_last - t_first); aggregate = sum of decode tokens / wall window.
Each concurrency level runs `repeats` times and the median is reported.

Usage: decode_bench.py [prose|structured] [repeats] [max_tokens]
Env:   SPARK_URL (default http://127.0.0.1:8888), SPARK_MODEL_NAME (default: the server's first model)
"""
import json, os, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("SPARK_URL", "http://127.0.0.1:8888").rstrip("/")
PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, "
             "including collision handling, resizing, and time complexity. Be thorough.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
}


def model_name():
    if os.environ.get("SPARK_MODEL_NAME"):
        return os.environ["SPARK_MODEL_NAME"]
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def stream(model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"].get("completion_tokens")
            d = (j.get("choices") or [{}])[0].get("delta", {}) if j.get("choices") else {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                now = time.perf_counter(); t_first = t_first or now; t_last = now; chunks += 1
    ct = usage if usage is not None else chunks
    dec = max(0, ct - 1)
    span = (t_last - t_first) if t_first and t_last and t_last > t_first else 0
    return {"tFirst": t_first, "tLast": t_last, "completion": ct, "decode": dec,
            "tps": dec / span if span else 0, "ttft_ms": (t_first - t0) * 1000 if t_first else 0}


def wave(model, base, n, max_tokens):
    prompts = [base] if n == 1 else [f"{base} (stream {i + 1}/{n})" for i in range(n)]
    res = [None] * n
    def go(i): res[i] = stream(model, prompts[i], max_tokens)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    win = max(r["tLast"] for r in res) - min(r["tFirst"] for r in res)
    return {"agg": sum(r["decode"] for r in res) / win, "per": statistics.mean(r["tps"] for r in res),
            "ttft": statistics.mean(r["ttft_ms"] for r in res), "tokens": [r["completion"] for r in res]}


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "prose"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    model = model_name()
    stream(model, PROMPTS[kind], 32)   # warm-up
    rows = {n: [] for n in (1, 2, 3, 4)}
    for rep in range(reps):
        for n in (1, 2, 3, 4):
            w = wave(model, PROMPTS[kind], n, mx); rows[n].append(w)
            print(f"rep{rep} {kind} C{n}: agg {w['agg']:.1f}  per {w['per']:.1f}  ttft {w['ttft']:.0f}ms  tokens {w['tokens']}", flush=True)
    print(f"\n== {model} {kind}, max_tokens {mx}, median of {reps}")
    for n, ws in rows.items():
        print(f"C{n}: agg {statistics.median(w['agg'] for w in ws):.1f} tok/s  per-stream {statistics.median(w['per'] for w in ws):.1f}  "
              f"TTFT {statistics.median(w['ttft'] for w in ws):.0f} ms")


if __name__ == "__main__":
    main()
