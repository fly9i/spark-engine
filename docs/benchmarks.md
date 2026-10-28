# Benchmarks

## Tools

| Script | What it measures |
| --- | --- |
| `bench/decode_bench.py [prose\|structured] [repeats] [max_tokens] [--temperature T] [--thinking] [--streams 1,2,3,4]` | Decode speed at 1–4 concurrent streams (the DGX Spark community method, see [Decode method](#decode-method-deepseek-and-qwen-2026-10-29-v037)): greedy and thinking off by default, 32-token warm-up, 400 output tokens, a `Request ID` line before each concurrent prompt; per stream `(completion_tokens - 1) / (t_last - t_first)`, aggregate = all decode tokens / wall window; median of `repeats` runs |
| `bench/prefill_bench.py TOKENS [SEED]` | Prefill of a synthetic prompt of about `TOKENS` tokens (random words, so different seeds share no prefix); prints the engine's prefill time and tok/s |
| `bench/ds41_decode_bench.py [--url URL] [--types prose,structured,code] [--conc 1,2,4]` | The DeepSeek-V4.1-Flash numbers ([ds41.md](ds41.md#7-performance)): greedy, thinking off, 32-token warm-up, 256 tokens per stream, prose / structured / code prompts |
| `bench/ds41_prefill_bench.py [repeats] [sizes]` | The DeepSeek-V4.1-Flash prefill numbers: 4K–64K prompts of real text, each with a unique prefix, streamed; tok/s = prompt tokens / TTFT, median of `repeats` |
| `bench/text_check.py LABEL [BASELINE]` | 20 fixed prompts, 256 greedy tokens each, one at a time: compares two builds or settings (identical replies, first divergence) |

`decode_bench.py`, `prefill_bench.py` and `text_check.py` talk to `SPARK_URL` (default `http://127.0.0.1:8888`). Two prompt types are used for decoding:

- **prose**: a long technical explanation, typical chat output; drafts are accepted less often;
- **structured**: counting from 1 to 200; nearly every draft token is accepted, which shows the upper end of speculative
  decoding.

Speculative decoding makes decode speed depend on the generated text. Two builds that round differently (for example a
kernel change of precision class L1) generate slightly different text and can differ by a few percent on one prompt
without either being faster. Compare kernels with per-round timings at fixed row counts, not with end-to-end tok/s alone.

## Results

GLM-5.3-Flash and the prefill rows: measured on 2026-10-08 with the release binary and default settings (`spark.env`
defaults, `GLM53_PCACHE=0` for cold prefill), greedy, thinking off, the older `(stream i/n)` suffix on concurrent prompts,
median of 3. DeepSeek and Qwen decode: 2026-10-29, see [Decode method](#decode-method-deepseek-and-qwen-2026-10-29-v037).
Aggregate decode throughput over all streams; per-stream speed in parentheses.

### GLM-5.3-Flash, 2× DGX Spark (TP2)

| Streams | Prose | Structured | TTFT (prose) |
| --- | --- | --- | --- |
| 1 | 52.9 tok/s | 92.5 tok/s | 372 ms |
| 2 | 66.5 (33.5 each) | 119.1 (76.2 each) | 784 ms |
| 3 | 73.3 (25.7 each) | 158.0 (54.5 each) | 926 ms |
| 4 | 82.9 (21.9 each) | 168.0 (44.3 each) | 1137 ms |

| Prompt | Prefill time | Prefill speed |
| --- | --- | --- |
| 9,561 tokens | 6.13 s | 1,561 tok/s |
| 9,551 tokens | 6.27 s | 1,522 tok/s |
| 54,686 tokens | 35.17 s | 1,555 tok/s |
| 54,880 tokens | 35.22 s | 1,558 tok/s |

### Decode method (DeepSeek and Qwen, 2026-10-29, v0.3.7)

The method commonly used for DGX Spark: `bench/decode_bench.py` against the running server; prompts *prose* (a
step-by-step explanation of how a hash map works) and *structured* (count from 1 to 200); 400 output tokens per request
(every reply ran the full 400 here); 1 to 4 concurrent requests, each concurrent request with a `Request ID: decode-<i>`
line before the prompt (one request: the prompt alone); 32-token warm-up; median of 5 runs. Two samplings, labelled on
every column: **greedy, thinking off** (temperature 0) and **T 1, thinking on** (temperature 1, top-p 1; reasoning tokens
are counted). Per request: (completion tokens − 1) / (last token − first token); aggregate = all decode tokens / the
window from the first token of the earliest request to the last token of the latest. Speculative decoding makes the
speed depend on the text, so other prompts give other numbers.

### DeepSeek-V4.1-Flash, 3× DGX Spark (TP3)

Decode (aggregate tok/s; per request in parentheses). The service decodes 4 requests at once (`DS41_SERVE_STORES=4`). More numbers (code prompt, prefill): [ds41.md](ds41.md#7-performance).

| Streams | Prose, greedy, thinking off | Structured, greedy, thinking off | Prose, T 1, thinking on | Structured, T 1, thinking on |
| --- | --- | --- | --- | --- |
| 1 | 63.9 tok/s (TTFT 178 ms) | 120.5 tok/s (TTFT 168 ms) | 55.4 tok/s (TTFT 277 ms) | 100.7 tok/s (TTFT 227 ms) |
| 2 | 86.6 (45.9 each, TTFT 310 ms) | 168.5 (86.6 each, TTFT 297 ms) | 77.0 (39.8 each, TTFT 379 ms) | 142.8 (75.4 each, TTFT 384 ms) |
| 3 | 103.9 (36.4 each, TTFT 413 ms) | 182.6 (63.5 each, TTFT 403 ms) | 90.0 (31.5 each, TTFT 524 ms) | 150.7 (56.1 each, TTFT 513 ms) |
| 4 | 116.3 (30.6 each, TTFT 516 ms) | 213.3 (56.5 each, TTFT 515 ms) | 97.3 (26.0 each, TTFT 658 ms) | 166.9 (48.1 each, TTFT 654 ms) |

### Qwen3.8-Flash-Next, 1× DGX Spark

Decode (aggregate tok/s; per request in parentheses):

| Streams | Prose, greedy, thinking off | Structured, greedy, thinking off | Prose, T 1, thinking on | Structured, T 1, thinking on |
| --- | --- | --- | --- | --- |
| 1 | 65.0 tok/s (TTFT 141 ms) | 151.4 tok/s (TTFT 139 ms) | 48.5 tok/s (TTFT 209 ms) | 116.5 tok/s (TTFT 210 ms) |
| 2 | 93.0 (48.0 each, TTFT 252 ms) | 224.1 (116.0 each, TTFT 250 ms) | 70.6 (39.6 each, TTFT 342 ms) | 97.2 (62.2 each, TTFT 340 ms) |
| 3 | 115.7 (40.8 each, TTFT 353 ms) | 268.7 (95.0 each, TTFT 362 ms) | 85.2 (31.2 each, TTFT 488 ms) | 100.6 (43.0 each, TTFT 482 ms) |
| 4 | 133.3 (35.3 each, TTFT 461 ms) | 287.9 (78.0 each, TTFT 482 ms) | 96.8 (27.3 each, TTFT 647 ms) | 119.0 (37.9 each, TTFT 632 ms) |

With sampling (T 1) the counting prompt at 2–3 concurrent requests is slower in aggregate than one request; the five runs
agree (2 requests 94–98 tok/s), so this is the batched sampled path on highly predictable text, not noise. The prefill
rows below are from 2026-10-08.

| Prompt | Prefill time | Prefill speed |
| --- | --- | --- |
| 13,449 tokens | 7.16 s | 1,879 tok/s |
| 13,456 tokens | 7.00 s | 1,922 tok/s |
| 77,204 tokens | 41.20 s | 1,874 tok/s |
| 77,274 tokens | 41.17 s | 1,877 tok/s |

TTFT grows with the number of streams because new requests are admitted between decode rounds of the running ones.
Multi-turn follow-ups reuse the previous turn's state: on a 16K-token conversation the next turn's TTFT with Qwen is
about 0.3 s instead of a full prefill (prompt checkpoints).

### Reproducing

```bash
engine-rs/serve/start.sh qwen                # or ds41 / glm
for k in prose structured; do
  SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py $k 5 400                              # greedy, thinking off
  SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py $k 5 400 --temperature 1 --thinking   # T 1, thinking on
done
for s in 11 12; do bench/prefill_bench.py 8000 $s; done
for s in 21 22; do bench/prefill_bench.py 46000 $s; done
```

## Method notes

- The server was restarted before each model's measurements, with the persistent prefix cache off (`GLM53_PCACHE=0`) so
  that prefill numbers are cold.
- Each concurrency level of `decode_bench.py` runs 5 times (GLM: 3); the tables report the median.
- Prefill: two prompts per size with different seeds; the table reports both.
- GLM-5.3-Flash ran on two DGX Sparks connected by both ConnectX-7 ports (2 × 200 Gb/s); Qwen3.8-Flash-Next on one.
