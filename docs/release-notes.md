spark-engine v0.3.4 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.3: DeepSeek-V4.1-Flash decode and the RDMA proxy (GLM-5.3-Flash and Qwen3.8-Flash-Next outputs were
checked identical). DeepSeek-V4.1 outputs stay bitwise identical (docs/ds41.md §7).

- **Routed-expert scales in a lossless 2-bit coding** (`DS41_MX_S2`, default on): the checkpoint's E8M0 expert scales
  are nearly constant (99.9% take one of two values), so per tile of 8 weight rows the engine keeps a base byte per row
  and four 2-bit offsets per row and 128-wide K step, instead of 32 bytes; tiles with a row spanning more than 3 keep
  their raw bytes. The decode, verify and prefill expert kernels rebuild the exact scale bytes for the block-scaled
  MMA. Expert bytes read −4.4%, about 3.8 GiB less memory per rank (more room for the KV pool); single-stream service
  decode +1.6% (prose) / +2.8% (structured) / +2.6% (code), batched decode steps of 2 and 4 sequences −1.5%.
- **RDMA proxy idle back-off:** after 5 ms without traffic the proxy threads poll about every 100 µs instead of
  spinning, so an idle engine no longer keeps one core busy per proxy (`GLM53_RDMA_ARN_IDLE_US`, default 5000).
- **Speculative sampling, optional** (`DS41_SPEC_RS=1`, default off): sampled requests can be verified by standard
  speculative sampling (accept with min(1, p/q), else resample max(p − q, 0)) instead of the default per-position key
  coupling. Both keep the target distribution exactly (checked with a distribution test of the verdict and end to end
  on the service); the default coupling also keeps a request's output a function of its seed alone. With the DSpark
  draft stages the two are within about one point of expected acceptance (0.615 vs 0.603 on open-ended texts at
  temperature 1) and served at the same speed, so the coupling stays the default. `DS41_RS_STATS=<file>` logs both
  schemes' expected acceptance per verified draft.
- `bench/ds41_decode_bench.py`: `--temperature`, `--effort` and open-ended prompts (`zh_prose`, `story`, `qa`).
