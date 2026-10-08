# Benchmarks

## Tools

| Script | What it measures |
| --- | --- |
| `bench/decode_bench.py [prose\|structured] [repeats] [max_tokens]` | Decode speed at 1, 2, 3 and 4 concurrent streams: greedy, thinking off, 32-token warm-up, 400 output tokens; per stream `(completion_tokens - 1) / (t_last - t_first)`, aggregate = all decode tokens / wall window; median of `repeats` runs |
| `bench/prefill_bench.py TOKENS [SEED]` | Prefill of a synthetic prompt of about `TOKENS` tokens (random words, so different seeds share no prefix); prints the engine's prefill time and tok/s |
| `bench/text_check.py LABEL [BASELINE]` | 20 fixed prompts, 256 greedy tokens each, one at a time: compares two builds or settings (identical replies, first divergence) |

All three talk to `SPARK_URL` (default `http://127.0.0.1:8888`). Two prompt types are used for decoding:

- **prose**: a long technical explanation, typical chat output; drafts are accepted less often;
- **structured**: counting from 1 to 200; nearly every draft token is accepted, which shows the upper end of speculative
  decoding.

Speculative decoding makes decode speed depend on the generated text. Two builds that round differently (for example a
kernel change of precision class L1) generate slightly different text and can differ by a few percent on one prompt
without either being faster. Compare kernels with per-round timings at fixed row counts, not with end-to-end tok/s alone.

## Results

Measured on 2026-10-08 with the release binary and default settings (`spark.env` defaults, `GLM53_PCACHE=0` for cold
prefill). Aggregate decode throughput over all streams; per-stream speed in parentheses.

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

### Qwen3.8-Flash-Next, 1× DGX Spark

| Streams | Prose | Structured | TTFT (prose) |
| --- | --- | --- | --- |
| 1 | 64.1 tok/s | 149.8 tok/s | 153 ms |
| 2 | 85.5 (44.1 each) | 192.9 (110.2 each) | 266 ms |
| 3 | 105.0 (36.0 each) | 232.6 (85.5 each) | 378 ms |
| 4 | 119.5 (31.7 each) | 217.4 (68.9 each) | 492 ms |

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
engine-rs/serve/start.sh glm                 # or qwen
SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py prose 3
SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py structured 3
for s in 11 12; do bench/prefill_bench.py 8000 $s; done
for s in 21 22; do bench/prefill_bench.py 46000 $s; done
```

## Method notes

- The server was restarted before each model's measurements, with the persistent prefix cache off (`GLM53_PCACHE=0`) so
  that prefill numbers are cold.
- Each concurrency level of `decode_bench.py` runs 3 times; the table reports the median.
- Prefill: two prompts per size with different seeds; the table reports both.
- GLM-5.3-Flash ran on two DGX Sparks connected by both ConnectX-7 ports (2 × 200 Gb/s); Qwen3.8-Flash-Next on one.
