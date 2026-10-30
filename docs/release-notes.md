spark-engine v0.3.10 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.9: faster DeepSeek-V4.1-Flash decode at long context and a faster GLM-5.3-Flash prefill on three
nodes. Every change keeps the output bitwise identical:
- DeepSeek: release goldens and paired-prefill checks are identical, and 64K / 256K-context dumps are byte-identical.
- GLM TP3: the paired prefill was checked against two plain chunks (logits, features and every layer's state, difference 0),
  and greedy replies match the previous schedule.
- GLM TP2 and Qwen3.8-Flash-Next share the binary; their greedy replies are bitwise identical to v0.3.9 (20 / 20 each).

**DeepSeek decode at long context** (all bitwise identical; `0` turns one off):
- At 512K context the indexer had added 4.8 ms to a decode step compared with 8K: 2.4 ms for scoring every key in four
  layers and 2.3 ms for a one-CTA top-k over each row.
- `DS41_TOPK_WIDE_MIN` (65536): top-k over rows of 64K keys or more runs on many CTAs (two 8-bit histogram passes over
  16K-key segments), with the same selection: the largest k, ties to the lower index, ascending output. A row 1M keys
  wide takes 870 → 59 µs.
- `DS41_INDEX_TILED_LONG`: the tiled scoring kernel is also used for a few rows against many keys (512K keys, one row:
  0.98 → 0.77 ms).
- `DS41_INDEX_KEY_SPLIT` (65536): in decode and verify, the full-scoring index layers split their keys over the three
  ranks. Each rank picks its local top 512, and one more selection over the 1,536 gathered entries gives exactly the
  full result. The candidate source layer also gathers each block's maximum. `DS41_INDEX_KEY_SPLIT_CHECK=1` also runs
  the replicated path and asserts that the indices are the same.
- `DS41_INDEX_SPLIT_CAND`: in prefill, the candidate source layer's rows are split over the ranks.
- Served, one stream, greedy, thinking off, 256 tokens after a needle-in-a-haystack prompt. Two fresh starts per
  version, alternated with v0.3.9:

  | Context | 256K | 512K | 1M |
  | --- | --- | --- | --- |
  | v0.3.9 | 48.9 / 51.2 tok/s | 49.8 / 46.2 tok/s | 42.0 / 40.5 tok/s |
  | v0.3.10 | 52.9 / 53.4 tok/s | 54.5 / 59.8 tok/s | 56.2 / 55.5 tok/s |
  | Change | +6% | +19% | +35% |

- Prefill is unchanged: a 1M-token prompt takes about 229 s to the first token (~4,540 tok/s). So is short-context
  decode.

**GLM-5.3-Flash prefill on three nodes** (`profiles/tp3-rdma.env` only; two-node runs do not read it, all L0):
- `GLM53_PREFILL_SP_PAIR` with `GLM53_SERVE_CHUNK_SOLO=8192`: a prompt served alone is sent in 8192-row units. Each unit
  runs as two 4096-row chunks taken through the layers in lockstep. Every collective goes on the communication stream
  and is waited on only where its result is used, so one chunk's gathers and reduce-scatters overlap the other chunk's
  attention and MoE.
  - This is the same computation as two consecutive 4096-row chunks. The clamp's count gate keeps separate per-row
    counters for the two chunks.
  - A unit that cannot be paired (the second chunk is under 128 rows) runs as the two chunks one after the other.
- `GLM53_RDMA_ARN_WAIT_SPLIT`: the row reduce-scatter / all-gather finish kernels wait for their peers in a separate
  one-CTA kernel. Without it, the finish kernels that the paired schedule issues early keep up to 96 CTAs spinning, and
  the overlapped compute ran 6% slower. Results are the same.
- `GLM53_COMM_STREAM_PRIO=1`: the communication stream runs at the highest priority.
- Served, alternated A/B (SPX1 V1.2 checkpoint): 8K 1,779 → 1,841 tok/s (+3.5%), 32K 1,694 → 1,754 tok/s (+3.3%).
- DeepSeek prefill already overlaps its communication in its own way. `GLM53_RDMA_ARN_WAIT_SPLIT` was measured there
  too (32K +0.4%, 8K +2% time) and stays off for it.

**Numbers**: short-context decode and prefill are as in v0.3.9 (`bench/decode_bench.py`, 400 tokens, median of 5,
1 / 2 / 3 / 4 streams, DeepSeek-V4.1-Flash on 3× DGX Spark, default GPU clocks):

| | Prose | Structured |
| --- | --- | --- |
| Greedy, thinking off | 67.4 / 92.8 / 109.3 / 124.7 tok/s | 126.6 / 181.9 / 194.8 / 226.0 tok/s |
| T 1, thinking on | 57.2 / 82.2 / 95.0 / 105.5 tok/s | 109.2 / 153.3 / 153.5 / 180.9 tok/s |

Switch descriptions: `docs/ds41.md` §5 (DeepSeek) and `docs/glm-switches.md` (GLM TP3).
