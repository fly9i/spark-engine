spark-engine v0.3.13 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.12, both for DeepSeek-V4.1-Flash: with several streams, a request's output no longer depends on what
else the server is doing (`DS41_BATCH_INVARIANT`, on by default in the start scripts), and a fix for the optional
conditional clamp. GLM-5.3-Flash and Qwen3.8-Flash-Next are unchanged.

**Multi-stream outputs equal single-stream outputs (`DS41_BATCH_INVARIANT=1`, default in `serve/ds41-rank.sh`).**
Single-stream greedy outputs were already reproducible, but with two or more streams a request's text could differ from
the same request run alone: when several sequences verify their drafts in one call, five kernel paths switch with the
total row count (routed-expert item size, the FP8 GEMV vs. a GEMM, the mHC mix, the compressor / indexer projections and
`wo_a`). They sum in a different order, and one-ulp differences grow through the layers. With the switch every row of a
decode / verify call of up to 32 rows (4 streams × up to 8 rows) takes the kernels of the single-stream path, so its
result is bitwise the same whatever runs beside it. It implies `DS41_GEMV32` and `DS41_HC32` (v0.3.12) plus three new
switches, `DS41_INV_MOE16`, `DS41_INV_PROJ8` and `DS41_INV_WOA`, which exist separately only for diagnostics.
`DS41_BATCH_INVARIANT=0` in `spark.env` restores the v0.3.12 behaviour. The binary's own default is off; only the start
scripts turn it on.

Checks with the switch on: teacher-forced verify rounds of 9–32 rows (2, 4 and 12 sequences, mixed content and
context lengths; 4,162 + 2,210 rows) bitwise equal to the single-stream reference (2,210-row mixed set with the switch
off: 511 equal). Served texts identical to the same requests run alone: the same prompt at 2 / 4 / 5 streams,
interleaved long prefills, staggered arrivals of 250 / 2K / 10K / 30K-token prompts, prompts restored from the prefix
cache while other streams decode, concurrent image requests, and fixed-seed sampling at 4 streams.

Limits: calls of more than 32 rows (more than 4 streams, or longer drafts) fall back to the row-count-dependent paths.
Prefill is not affected. A prompt that extends a cached conversation (its earlier answer decoded token by token) can
still differ from the same prompt prefilled from scratch; that comes from how prefill closes a prompt, not from
concurrency, and is unchanged. `DS41_BATCH_INVARIANT` is part of the prefix-cache identity, so entries saved with the
other setting are not reused.

**Optional conditional clamp: per-row gate counters no longer carry over (`DS41_INV_CNT`, default on).** With the
optional conditional clamp (`DS41_ABLATE_CLAMP`) enabled, the per-row counters of its count gate could carry over between
requests: the last step of a prompt's prefill read counter rows that the prompt itself had not written, left by another
request's prefill, a decode step or the previous request. Only integer counters carried over, no text, activations or
KV, but it made outputs with the clamp enabled depend on earlier or concurrent requests and applied the clamp to rows
it was not meant for. Each sequence now carries its own counts through prefill chunks, checkpoints and prefix-cache
entries. With the clamp enabled, outputs can differ from v0.3.12, in particular for prompts of 1,024 tokens or more;
without the clamp nothing changes. `DS41_INV_CNT=0` restores the previous behaviour (for comparisons only).

**Speed** (v0.3.13, 2026-11-11, three DGX Sparks, DeepSeek-V4.1-Flash-1017, default settings plus the optional
conditional clamp as in the README's measurements; greedy, thinking off; same build with `DS41_BATCH_INVARIANT` off and
on). Decode with `bench/ds41_decode_bench.py --types prose,structured,code --conc 1,2,4` (256 tokens per stream,
aggregate tok/s), four service starts in the order off, on, on, off, 3 runs each: median of n = 6 per cell, off / on:

| Streams | Prose | Structured | Code |
| --- | --- | --- | --- |
| 1 | 69.3 / 69.3 | 126.3 / 127.1 | 111.6 / 112.5 |
| 2 | 99.0 / 95.9 | 179.8 / 188.1 | 148.4 / 147.2 |
| 4 | 126.2 / 128.3 | 220.6 / 242.9 | 169.1 / 184.1 |

Beyond the run-to-run spread: 4 streams structured +10.1% and code +8.9%, 2 streams structured +4.6%, and 2 streams prose
−3.1% (not located; timing the verify calls on the same rows shows no cost there). The other cells are within the spread
(up to about 4% between the two "off" starts). Prefill, warm, `bench/ds41_prefill_bench.py 5 8k,32k` in alternated
starts: 8K 5,404–5,477 and 32K 5,519–5,603 tok/s with the switch on or off, the same as v0.3.12 (5,464–5,474 /
5,550–5,556 in the same series).

Checks against v0.3.12 on our three DGX Sparks: DeepSeek release goldens without the clamp identical, paired-prefill
checks identical, serving, image (including `boundary`) and prefix-cache fork tests pass; GLM-5.3 TP2 and
Qwen3.8-Flash-Next greedy replies identical (20 / 20 each).
