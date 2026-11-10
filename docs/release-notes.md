spark-engine v0.3.12 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.11: an incremental detokenizer in the front ends, small hardening of the DeepSeek-V4.1-Flash serving
loop, dead code removed, and new optional DeepSeek switches that are **off by default**. With the default settings the
release checks below are bitwise identical to v0.3.11.

**Front ends: incremental detokenization.** Streaming responses used to re-decode the whole completion for every new
token, so the front end's CPU time grew with the square of the output length. The GLM, DeepSeek and Qwen front ends now
decode only a window of the newest tokens (it starts at the last token boundary where the bytes are complete UTF-8) and
search stop strings only in the text that can still hold a first match. Deltas, stop decisions and the final text are
the same as before (32,000 comparison cases over the three tokenizers, no difference). On 8,000-token outputs the front
end's CPU time is about 7× lower (1 stream 1.2 → 0.16 s, 4 streams 5.2 → 0.72 s); client tok/s is unchanged.
`GLM53_TEXTSTREAM_FULL=1` in `spark.env` restores the previous full re-decode.

**DeepSeek-V4.1-Flash serving loop.** The `max_tokens` capacity check uses a saturating add (a huge `max_tokens` could
wrap past it), the draft-choice comparison uses a total order (a NaN confidence can no longer panic every rank), stop
token ids are scanned only in the tokens that arrived in this step, and a few settings are read once instead of every
round.

**Optional switches (DeepSeek-V4.1-Flash, all off by default).** Set them in `spark.env` (every rank reads it) or through
`DS41_EXTRA_ENV`.

| Switch | What it does |
| --- | --- |
| `DS41_GEMV32=1` with `DS41_HC32=1` | Speculative verify rounds of 17–32 rows (several streams verifying together) stay on the ≤ 16-row decode kernels instead of the prefill-sized paths: the MXFP4/FP8 GEMV takes a second 16-row tile in the same weight pass, the router's C12 GEMV, the shared expert's parallel branch and the mHC mix take up to 32 rows. Each row gets bitwise the result of a ≤ 16-row call (`ds41-kernel-test` with `DS41_KT_ONLY=w32`). Served decode, same build with the switches on and off, 4 runs per arm: 4 streams code 164.1 → 189.2 tok/s, 4 streams structured 220.3 → 235.9 tok/s; 4 streams prose and 1 / 2 streams unchanged. **Outputs with several streams differ from the default path** (those rounds take different kernels than today's > 16-row path); single-stream outputs are identical. Kept off by default for that reason. `DS41_HC32` alone gives no measurable gain. |
| `DS41_ENGRAM_EARLY=1` | At the start of a speculative round, reads the Engram rows of each sequence's pending token into the page cache on a background thread, under the draft. Outputs unchanged; the Engram wait per round drops, but no decode throughput gain was measured. |
| `DS41_ENGRAM_ASYNC_MIN` (default `256`) | Prompt chunks of at least this many rows gather their Engram rows in the background (the behaviour so far); lower values add shorter chunks. Outputs unchanged; no TTFT difference measured for 40–500-token prompts. |
| `DS41_MOE_STATS=1` | Diagnostics: routed-expert calls count their rows and distinct experts (a device sync each) and rank 0 logs a `[ds41-serve] moe:` line per prompt. |
| `DS41_VPROBE=<plan.json>` | Measurement with `ds41-bench`: teacher-forced logits of multi-sequence verify rounds against the same rows run one sequence at a time. |

**Dead code removed.** Unused Rust functions, unused `shim.cpp` wrappers and `shim/c11.cuh` are gone; nothing that runs
changed.

**Speed** (v0.3.12, 2026-11-10, three DGX Sparks, DeepSeek-V4.1-Flash-1017, default settings plus the optional
conditional clamp as in the README's measurements; greedy, thinking off). Decode with
`bench/ds41_decode_bench.py --types prose,structured,code --conc 1,2,4 --max-tokens 400`, aggregate tok/s, one run each:

| Streams | Prose | Structured | Code |
| --- | --- | --- | --- |
| 1 | 69.3 | 127.1 | 112.3 |
| 2 | 97.7 | 180.0 | 150.4 |
| 4 | 129.4 | 219.4 | 176.5 |

Prefill with `bench/ds41_prefill_bench.py 5 8k,32k` (streamed, rate = prompt tokens / TTFT, each sample a unique prefix,
median of 5): 8K cold 5,400 / warm 5,432 tok/s, 32K cold 5,677 / warm 5,505 tok/s (cold = page caches dropped on all
three nodes and the service restarted, every sample new text). These are single release measurements, not a
side-by-side comparison with v0.3.11; decode runs move by a few percent between service starts.

Checks against v0.3.11 on our three DGX Sparks: DeepSeek release goldens and paired-prefill checks identical, serving,
image (including `boundary`) tests pass; GLM-5.3 TP2 and Qwen3.8-Flash-Next greedy replies identical (20 / 20 each).
Qwen3.8-Flash-Next decode after the check (prose, greedy, thinking off, 400 tokens, concurrent prompts marked with a
`(stream i/n)` suffix, median of 3): 3 / 4 streams 110.6 / 126.5 tok/s aggregate.
