#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import os;os.environ['NO_PROXY']=os.environ['no_proxy']='*'
"""Prefill sweep: 4k / 8k / 16k / 32k / 64k, each sample its own unique prefix so the prefix cache cannot help, streamed;
per size `repeats` samples and the median-rate one reported.
prefill = the server's own prompt-processing time when it reports one, else tokens computed / (TTFT - calibrated request
overhead, the correction capped at half the TTFT) -- this front end reports none in a stream, so it is the TTFT rate; the engine's own prefill time (its log line for the request, DS41_LOG) is shown beside it. TTFT = request
start to the first streamed token. Prompt text: this repository's source (real text), cut to the size with the model's tokenizer.
usage: ds41_prefill_bench.py [repeats] [sizes, e.g. 4k,8k,16k,32k,64k]   (URL from AGREE_URL; DS41_TOKENIZER: the
checkpoint's tokenizer.json; SD_FRESH=1: never reuse text across samples; SD_CORPUS=all: more source text)"""
import glob, json, statistics, sys, time, urllib.request
from tokenizers import Tokenizer

URL = os.environ.get("AGREE_URL", "http://127.0.0.1:8888") + "/v1/chat/completions"
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
sizes = [int(s.rstrip('k')) * 1024 for s in (sys.argv[2] if len(sys.argv) > 2 else "4k,8k,16k,32k,64k").split(',')]
tk = Tokenizer.from_file(os.environ.get("DS41_TOKENIZER", "/models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))
root = os.path.dirname(os.path.abspath(__file__)) + "/.."
# SD_CORPUS=all: also the CUDA / Python sources and the docs (enough text for SD_FRESH=1 over 4k-64k x 3 without reuse)
pats = ["/engine-rs/**/*.rs"] + (["/engine-rs/**/*.cu", "/engine-rs/**/*.cuh", "/engine-rs/**/*.cpp", "/**/*.py", "/docs/**/*.md", "/*.md"] if os.environ.get("SD_CORPUS") == "all" else [])
src = "".join(open(f, errors="ignore").read() for pat in pats for f in sorted(glob.glob(root + pat, recursive=True)))
ids = tk.encode(src).ids


def run(text, max_tokens=1):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t1 = None; usage = timing = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            if j.get("glm53_timing"): timing = j["glm53_timing"]
            d = (j.get("choices") or [{}])[0].get("delta", {}) if j.get("choices") else {}
            if t1 is None and (d.get("content") or d.get("reasoning_content")): t1 = time.perf_counter()
    return (t1 - t0) * 1e3 if t1 else None, usage or {}, timing or {}


LOG = os.environ.get("DS41_LOG", "/tmp/spark-ds41-rank0.log")
def engine_prefill_ms():
    import re
    try: lines = [l for l in open(LOG, errors="ignore") if "[ds41-serve] request:" in l]
    except OSError: return None
    m = re.search(r"prefill ([0-9.]+) ms", lines[-1]) if lines else None
    return float(m.group(1)) if m else None


# request overhead: TTFT of a tiny prompt (median of 5)
over = statistics.median(run(f"[cal {time.time_ns()}] hi")[0] for _ in range(5))
print(f"calibrated request overhead {over:.0f} ms")
off = 0
for n in sizes:
    samples = []
    for k in range(reps):
        # unique prefix (a salt line) per sample; the body: this size's own part of the source, the same for its samples
        # (SD_FRESH=1: a part not used before for every sample -- the engram table pages cold each time)
        if k == 0 or os.environ.get("SD_FRESH") == "1":
            body_ids = ids[off:off + n - 40]; off += n
            if len(body_ids) < n - 40: off = 0; body_ids = ids[:n - 40]
        text = f"[sd-{time.time_ns()}-{n}-{k}]\n" + tk.decode(body_ids)
        ttft, u, tm = run(text)
        hit = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or tm.get("prefix_hit_tokens") or 0
        comp = u.get("prompt_tokens", 0) - hit
        srv = tm.get("prefill_ms") or engine_prefill_ms()
        rate = comp / max(ttft - min(over, ttft / 2), 1e-9) * 1e3   # (no server timing in the stream)
        rate_ttft = comp / srv * 1e3 if srv else 0.0
        samples.append((rate, ttft, comp, srv, rate_ttft))
    samples.sort()
    rate, ttft, comp, srv, rate_ttft = samples[len(samples) // 2]
    print(f"{n // 1024:3d}k  {comp:6d} tokens  TTFT {ttft:7.0f} ms  prefill {rate:7.1f} tok/s (server {srv:.0f} ms)  "
          f"engine {rate_ttft:7.1f} tok/s   [{', '.join(f'{s[0]:.0f}' for s in samples)}]", flush=True)
