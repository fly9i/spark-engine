#!/usr/bin/env python3
"""Greedy text check: 20 fixed prompts, 256 tokens each, one at a time; saves the replies and, given a baseline label,
reports how many are identical and where the others first diverge. Use it to compare builds or settings: rounding-level
changes legitimately diverge after some tokens (FP8 KV cache), schedule-only changes must stay identical.

Usage: text_check.py LABEL [BASELINE_LABEL]   (results in /tmp/text-check-LABEL.json)   Env: SPARK_URL, SPARK_MODEL_NAME
"""
import json, os, sys, urllib.request

BASE = os.environ.get("SPARK_URL", "http://127.0.0.1:8888").rstrip("/")
PROMPTS = [
    "Explain how a hash map handles collisions.", "Write a haiku about autumn rain.", "What is the capital of Australia? Answer briefly.",
    "Write a Python function that reverses a linked list.", "Summarize the causes of the French Revolution in five bullet points.",
    "用三句话介绍一下混合专家模型。", "Translate to French: The weather is nice today and we are going to the park.",
    "What is 17 * 23? Show the steps.", "Write a SQL query that finds the second highest salary in a table employees(salary).",
    "Give three tips for writing clear technical documentation.", "Describe the water cycle to a ten-year-old.",
    "List the planets of the solar system in order.", "Write a short story opening about a lighthouse keeper.",
    "解释一下什么是向量数据库，以及它适合哪些场景。", "What are the differences between TCP and UDP?",
    "Write a regular expression that matches an email address and explain it.", "Name five sorting algorithms and their complexities.",
    "Write a limerick about a cat who codes.", "How does public-key cryptography work?", "Convert this JSON to YAML: {\"a\": 1, \"b\": [2, 3]}",
]
label = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
model = os.environ.get("SPARK_MODEL_NAME") or json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]
out = []
for q in PROMPTS:
    b = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 256, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(b).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    out.append(r["choices"][0]["message"]["content"] or "")
json.dump(out, open(f"/tmp/text-check-{label}.json", "w"), ensure_ascii=False)
print(f"[{label}] saved {len(out)} replies")
if base and os.path.exists(f"/tmp/text-check-{base}.json"):
    ref = json.load(open(f"/tmp/text-check-{base}.json"))
    same = sum(a == b for a, b in zip(out, ref))
    divs = sorted(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) for a, b in zip(out, ref) if a != b)
    print(f"[{label}] vs {base}: identical {same}/{len(out)}; first divergence (characters) {divs}")
