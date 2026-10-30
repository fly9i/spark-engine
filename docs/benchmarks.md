# Benchmarks

## Tools

| Script | What it measures |
| --- | --- |
| `bench/decode_bench.py [prose\|structured] [repeats] [max_tokens] [--temperature T] [--thinking] [--streams 1,2,3,4]` | Decode speed at 1–4 concurrent streams (the DGX Spark community method, see [Decode method](#decode-method-deepseek-and-qwen-2026-10-29-v038)): greedy and thinking off by default, 32-token warm-up, 400 output tokens, a `Request ID` line before each concurrent prompt; per stream `(completion_tokens - 1) / (t_last - t_first)`, aggregate = all decode tokens / wall window; median of `repeats` runs |
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

GLM-5.3-Flash (decode and prefill): measured on 2026-10-08 on DGX OS 7.5.0 with the GPU clock locked at 2,200 MHz, with the release binary and default settings (`spark.env`
defaults, `GLM53_PCACHE=0` for cold prefill), greedy, thinking off, the older `(stream i/n)` suffix on concurrent prompts,
median of 3. DeepSeek and Qwen decode and Qwen prefill: 2026-10-29, see [Decode method](#decode-method-deepseek-and-qwen-2026-10-29-v038).
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

### Decode method (DeepSeek and Qwen, 2026-10-29, v0.3.8)

The method commonly used for DGX Spark: `bench/decode_bench.py` against the running server; prompts *prose* (a
step-by-step explanation of how a hash map works) and *structured* (count from 1 to 200); 400 output tokens per request
(every reply ran the full 400 here); measured on the updated system of the README's "Test system and GPU clock"
section at default GPU clocks, clock sampled every second, no throttling recorded; 1 to 4 concurrent requests, each concurrent request with a `Request ID: decode-<i>`
line before the prompt (one request: the prompt alone); 32-token warm-up; median of 5 runs. Two samplings, labelled on
every column: **greedy, thinking off** (temperature 0) and **T 1, thinking on** (temperature 1, top-p 1; reasoning tokens
are counted). Per request: (completion tokens − 1) / (last token − first token); aggregate = all decode tokens / the
window from the first token of the earliest request to the last token of the latest. Speculative decoding makes the
speed depend on the text, so other prompts give other numbers.

### DeepSeek-V4.1-Flash, 3× DGX Spark (TP3)

Decode (aggregate tok/s; per request in parentheses; 2026-10-30, v0.3.9). The service decodes 4 requests at once (`DS41_SERVE_STORES=4`). More numbers (code prompt, prefill): [ds41.md](ds41.md#7-performance).

| Streams | Prose, greedy, thinking off | Structured, greedy, thinking off | Prose, T 1, thinking on | Structured, T 1, thinking on |
| --- | --- | --- | --- | --- |
| 1 | 67.4 tok/s (TTFT 171 ms) | 126.6 tok/s (TTFT 162 ms) | 57.2 tok/s (TTFT 245 ms) | 109.2 tok/s (TTFT 218 ms) |
| 2 | 92.8 (49.2 each, TTFT 287 ms) | 181.9 (93.5 each, TTFT 281 ms) | 82.2 (41.8 each, TTFT 370 ms) | 153.3 (81.6 each, TTFT 356 ms) |
| 3 | 109.3 (37.4 each, TTFT 395 ms) | 194.8 (67.7 each, TTFT 381 ms) | 95.0 (33.5 each, TTFT 504 ms) | 153.5 (59.7 each, TTFT 485 ms) |
| 4 | 124.7 (32.4 each, TTFT 495 ms) | 226.0 (59.9 each, TTFT 493 ms) | 105.5 (27.8 each, TTFT 636 ms) | 180.9 (51.5 each, TTFT 610 ms) |

### Qwen3.8-Flash-Next, 1× DGX Spark

Decode (aggregate tok/s; per request in parentheses):

| Streams | Prose, greedy, thinking off | Structured, greedy, thinking off | Prose, T 1, thinking on | Structured, T 1, thinking on |
| --- | --- | --- | --- | --- |
| 1 | 68.4 tok/s (TTFT 136 ms) | 157.8 tok/s (TTFT 133 ms) | 51.2 tok/s (TTFT 203 ms) | 122.7 tok/s (TTFT 204 ms) |
| 2 | 98.0 (50.4 each, TTFT 242 ms) | 230.7 (119.7 each, TTFT 241 ms) | 74.3 (40.8 each, TTFT 330 ms) | 78.1 (40.1 each, TTFT 328 ms) |
| 3 | 122.4 (43.0 each, TTFT 342 ms) | 280.1 (99.2 each, TTFT 343 ms) | 90.9 (32.8 each, TTFT 467 ms) | 100.9 (39.0 each, TTFT 476 ms) |
| 4 | 139.2 (36.8 each, TTFT 438 ms) | 310.3 (83.3 each, TTFT 462 ms) | 103.8 (28.5 each, TTFT 615 ms) | 108.0 (36.5 each, TTFT 592 ms) |

With sampling (T 1) the counting prompt at 2–3 concurrent requests is slower in aggregate than one request; the five runs
agree (2 requests 77.8–80.6 tok/s), so this is the batched sampled path on highly predictable text, not noise. Prefill
(below) was measured on 2026-10-29 with the same build.

| Prompt | Prefill time | Prefill speed |
| --- | --- | --- |
| 13,449 tokens | 6.54 s | 2,056 tok/s |
| 13,456 tokens | 6.46 s | 2,084 tok/s |
| 77,204 tokens | 38.72 s | 1,994 tok/s |
| 77,274 tokens | 39.10 s | 1,976 tok/s |

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
