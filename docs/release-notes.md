spark-engine v0.3.0 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.2.1 (DeepSeek-V4.1-Flash; GLM-5.3-Flash and Qwen3.8-Flash-Next code paths are unchanged apart from the
shared RDMA transport). Every change keeps the outputs bitwise identical. Measured on three DGX Sparks, served, with the
1017 checkpoint (docs/ds41.md §7):

- **Prefill about 1.9× faster:** ~5,400 tok/s for 8K–64K prompts (32K in ~6.1 s), ~4,600 tok/s at 4K, the same for new
  and repeated text (v0.2.1: ~2,900 tok/s cold at 28–33K).
  - Sequence-parallel encoder: each rank keeps a third of the rows through the norms, mHC and Engram; attention and
    MoE inputs are all-gathered (as FP8 operands), outputs reduce-scattered over RDMA, with the scatters interleaved
    with the next part's attention and deferred into the next layer.
  - FP8 GEMM v3 (swizzled shared memory, 3-stage pipeline, 64-row tiles for small M).
  - Router logits only for the rank's own rows: a cuBLASLt configuration that reproduces the full matmul's bits.
  - wo_a as one grouped launch.
  - MoE combine fused into the reduce-scatter send.
  - Two 4096-token sub-chunks share one MoE call (`DS41_PREFILL_CHUNK` 8192).
  - PDL suspended during prefill: early-launched grids parked on their wait held SMs.
  - The communication stream at the highest priority; RDMA row transfers 12 deep (`GLM53_RDMA_ARN_ROWS_PIPE`).
  - Engram rows of every chunk, including the next one, read in the background with O_DIRECT. The next chunk's rows
    are now asked for with its real length (the prefix cache shortens the last chunk by one token). Before, new text
    paid a synchronous NVMe read in layer 1 of that chunk.
- **Decode about 4% faster per step:** the mHC mixes of each sublayer run as extra CTAs of the two GEMVs that follow
  them, off the critical path (`DS41_HC_JOBS`). Single stream 55.3 / 103.6 / 93.2 tok/s (prose / structured / code),
  4 streams 93.1 / 186.6 / 133.1.
- **`DS41_MIX_DECODE_SHARE`** (default 0): a share of the time for decoding streams while another prompt prefills,
  traded against that prompt's prefill.
- **RDMA:** the RoCE v2 GID index is looked up per device (it is not the same on every node).
- **API:** `usage.prompt_tokens_details.cached_tokens` reports prefix-cache hits.
- **Docs:** README links the project's own GLM-5.3-Flash quantization
  [fly9i/GLM-5.3-Flash-SPX1-V1.1](https://huggingface.co/fly9i/GLM-5.3-Flash-SPX1-V1.1). New
  `bench/ds41_prefill_bench.py`.

Node setting: the host settings of docs/deploy.md (no proactive compaction, no watermark boosting) are needed on all
three nodes. Memory reclaim on one node stalls every rank: a node without them gave random +0.4–1.5 s prefill spikes.

What it runs:

- GLM-5.3-Flash (EXL3 4 bpw) on two (or three) DGX Sparks with DFlash2 speculative decoding.
- DeepSeek-V4.1-Flash-1017 on three DGX Sparks with DSpark speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs,
`reference/ds41` (Engram hash export). Install PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not
included.
