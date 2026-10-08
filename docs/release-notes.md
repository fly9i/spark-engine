First public release of spark-engine for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

- GLM-5.3-Flash (EXL3 4 bpw) on two DGX Sparks (TP2) with DFlash2 speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs. Install
PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not included.
