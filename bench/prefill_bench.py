#!/usr/bin/env python3
"""Prefill speed: a synthetic prompt of about N tokens (random words, so no prefix reuse across seeds), 24 output tokens.
Prints prompt tokens, the server's prefill time (when it reports one) and the end-to-end time to first token.

Usage: prefill_bench.py TOKENS [SEED]      Env: SPARK_URL (default http://127.0.0.1:8888), SPARK_MODEL_NAME
"""
import hashlib, json, os, random, sys, time, urllib.request

BASE = os.environ.get("SPARK_URL", "http://127.0.0.1:8888").rstrip("/")
n = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
random.seed(seed)
words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
         "psi omega river mountain cloud stone forest ocean signal vector matrix kernel memory thread cache packet").split()
text = " ".join(random.choice(words) + str(random.randint(0, 999)) for _ in range(int(n * 0.42)))
model = os.environ.get("SPARK_MODEL_NAME") or json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": text + "\nSummarize in one line."}], "max_tokens": 24,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=3600))
wall = time.time() - t
p = r["usage"]["prompt_tokens"]
pm = (r.get("glm53_timing") or r.get("timing") or {}).get("prefill_ms")
h = hashlib.sha256((r["choices"][0]["message"]["content"] or "").encode()).hexdigest()[:12]
if pm:
    print(f"prompt {p} tokens: prefill {pm / 1000:.2f} s -> {p / (pm / 1000):.0f} tok/s (request {wall:.2f} s, reply {h})")
else:
    print(f"prompt {p} tokens: request {wall:.2f} s (reply {h})")
