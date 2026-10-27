spark-engine v0.3.1 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.0: DeepSeek-V4.1-Flash decode only (GLM-5.3-Flash and Qwen3.8-Flash-Next code paths are unchanged;
their outputs were checked identical). Every change keeps the outputs bitwise identical. Measured on three DGX Sparks,
served, with the 1017 checkpoint (docs/ds41.md §7):

- **Decode with speculation faster:** +7 / +11 / +11% for one stream, +13 / +5 / +9% for four (prose / structured /
  code). Single stream 59.1 / 114.8 / 103.3 tok/s (v0.3.0: 55.3 / 103.6 / 93.2), 4 streams 105.5 / 195.3 / 144.9
  (v0.3.0: 93.1 / 186.6 / 133.1).
- **Decode without speculation 6–9% less time per step:** 26.3 ms per token for one sequence (v0.3.0: 29.0), 33.4 /
  43.9 / 57.2 ms per step for 2 / 4 / 8 sequences (v0.3.0: 36.1 / 47.4 / 60.9).
  - The BF16 weights the decode GEMVs read (vocabulary head, routers, compressor and indexer projections, the draft's
    Markov head) are kept a second time in a lossless 12-bit coding (sign, 7-bit mantissa, 4-bit exponent window per
    row, exceptions stored exactly) and read from it: 25% fewer bytes, the same FMA order as the BF16 kernel, every
    weight checked against its BF16 value at load. Prefill keeps reading the BF16 originals. About +0.5 GB per rank
    (`DS41_C12`).
  - The joint q / KV low-rank projection, which every rank used to compute in full, is split by rows over the ranks and
    all-gathered over RDMA (a third of the bytes per rank; `DS41_QAKV_SPLIT`).
  - Engram rows are fetched and uploaded right before the layer that reads them, overlapping the earlier layers instead
    of holding the GPU at the start of every step (`DS41_ENGRAM_JIT`).
  - The clamped sublayer norm loads its clamp directions with vector loads before it waits for the previous kernel,
    instead of scalar loads inside its loops (it took 14–16 µs, twice per layer on the critical path; about 9 µs less
    per call; `DS41_RMS_CLP`).
  - Multi-row GEMVs (speculative verification, batches, the draft) quantize their input once in a separate kernel
    instead of in every thread block: 8 rows of wo_b 102 → 69 µs (`DS41_GEMV_PREQ_ROWS`, `DS41_GEMV_PREQ_K1`).
- **Tests:** `ds41-kernel-test` checks the 12-bit GEMV and the norm and GEMV variants bitwise against the originals;
  `ds41-ar-bench` times the decode-size row all-gather.

Prefill is unchanged (~5,400 tok/s for 8K–64K prompts).

What it runs:

- GLM-5.3-Flash (EXL3 4 bpw) on two (or three) DGX Sparks with DFlash2 speculative decoding.
- DeepSeek-V4.1-Flash-1017 on three DGX Sparks with DSpark speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs,
`reference/ds41` (Engram hash export). Install PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not
included.
