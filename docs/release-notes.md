spark-engine v0.3.9 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.8: DeepSeek-V4.1-Flash data-flow changes that cut memory and network traffic in prefill and decode.
Every change that is on by default keeps the output bitwise identical: DeepSeek release goldens and paired-prefill checks
identical, a 128K-context decode dump byte-identical with and without the indexer change. GLM-5.3-Flash and
Qwen3.8-Flash-Next share the binary and were checked unchanged (greedy replies bitwise identical, 20 / 20 each).

**DeepSeek prefill** (all bitwise identical):
- `DS41_SP_EMBED_RS`: each rank embeds only its own sequence-parallel rows; the rows are reduce-scattered instead of
  all-reduced.
- `DS41_SP_END_SPARSE`: at the encoder's exit only the rows the decoder reads are gathered.
- `GLM53_RDMA_ARN_COMB_VEC8`: the MoE combine fused into the RDMA reduce-scatter reads 8 elements per thread.
- Engram projection writes BF16 straight from the GEMM epilogue; prefill Engram rows and decode / draft inputs are
  uploaded through pinned staging buffers without blocking the host (`DS41_STAGED_BIG`, `DS41_STAGED_SMALL`).
- Offline 32K prefill −2.1%. Served, side by side with v0.3.8 (two fresh starts each, alternated): 4K 4,748 → 4,738,
  8K 5,439 → 5,467, 32K 5,545 → 5,646 tok/s (+1.8%).

**DeepSeek decode** (all bitwise identical):
- `DS41_INDEX_CAND`: past the candidate source layer, the indexer scores only the candidate blocks (16,384 compressed
  keys); blocks without a candidate are not loaded. A decode step at 128K context 28.4 → 27.0 ms (−5.0%); no change up
  to 64K, more saved at longer contexts.
- `DS41_AR_B16_WIRE`: BF16 all-reduces send BF16 and sum in FP32 in rank order (half the bytes, same result).
- `DS41_MTP_SEED_SPLIT`: the draft's seed projection split by rows over the ranks.
- C12 GEMV up to 16 rows, logits / router computed in 16-row chunks.
- `bench/ds41_decode_bench.py`, greedy, side by side with v0.3.8 (median of 6 runs over two fresh starts each):
  1 stream prose / structured / code 68.5 / 125.1 / 111.2 → 69.5 / 129.2 / 114.2 tok/s, 4 streams 121.7 / 215.6 /
  166.5 → 125.3 / 221.3 / 167.8.

**Experiments, off by default** (they change rounding or summation order): `DS41_MOE_PAIR_BF16` (routed-expert pair
outputs in BF16), `DS41_WO_A_MMA` (grouped `wo_a` on the block-scaled MMA), `DS41_COMP_TF32` (compressor projection on
tensor cores). They make prefill 1.7–2.1% faster, but the encoder output feeds the draft, which then accepts fewer tokens:
served decode was 0.75–1.8% slower.

**Numbers** (`bench/decode_bench.py`, 400 tokens, median of 5, 1 / 2 / 3 / 4 streams, DeepSeek-V4.1-Flash on 3× DGX
Spark, default GPU clocks):

| | Prose | Structured |
| --- | --- | --- |
| Greedy, thinking off | 67.4 / 92.8 / 109.3 / 124.7 tok/s | 126.6 / 181.9 / 194.8 / 226.0 tok/s |
| T 1, thinking on | 57.2 / 82.2 / 95.0 / 105.5 tok/s | 109.2 / 153.3 / 153.5 / 180.9 tok/s |

Prefill ~5,570 tok/s for 8K–64K prompts (4K ~4,760). Switch descriptions: `docs/ds41.md` §5.
