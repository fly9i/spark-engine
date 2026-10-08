spark-engine v0.1.1 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.1.0:

- **Pinned host memory no longer stalls decode on long-running nodes.** The host staging buffers (about 1.3 GB on GLM)
  are now anonymous pages registered with `cudaHostRegister` instead of `cudaHostAlloc` memory. Kernel compaction never
  isolates the former, while on a fragmented node it isolated the latter over and over. On by default;
  `GLM53_HOST_REGISTER=0` reverts. docs/deploy.md now also recommends `vm.watermark_boost_factor = 0` next to
  `vm.compaction_proactiveness = 0`.
- **Direction ablation (`SPARK_ABLATE`) is effectively free.** All k directions of a residual write are removed in one
  kernel instead of k launches. GLM with `subspace:16-40:8`: decode 42.9 → 46.0 tok/s with both ablation mechanisms
  on, and `SPARK_ABLATE` alone now measures within noise of no ablation. docs/abliteration.md has a new performance
  section: the o_proj transplant (`GLM53_ABLIT`) adds no compute but lowers draft acceptance by about 4%, so
  `SPARK_ABLATE` alone is recommended.
- Project banner in the README.

What it runs:

- GLM-5.3-Flash (EXL3 4 bpw) on two DGX Sparks (TP2) with DFlash2 speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs. Install
PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not included.
