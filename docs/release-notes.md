spark-engine v0.2.0 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.1.1:

- **DeepSeek-V4.1-Flash on three DGX Sparks (TP3).** The official `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint
  (MXFP4 experts, FP8 weights) runs unchanged: block-scaled tensor-core MMA kernels, CED prefill, Engram tables streamed
  from NVMe, DSpark speculative decoding, NVMe prefix cache, image input, the official prompt encoding with tool calls
  and thinking budgets. Single stream 50–55 tok/s on prose, ~95–100 tok/s on structured text, 88 / 173 tok/s with 4
  streams; cold prefill ~2,900 tok/s. Complete recipe (network, weights, sharding, Engram hash, settings):
  docs/ds41.md. `start.sh ds41`.
- **GLM-5.3-Flash on three nodes (`start.sh glm3`, `GLM_WORLD=3`).** Same binary, checkpoint and profile as TP2; KDA
  heads 22/21/21, experts 6/5/5 Hadamard blocks, sequence-parallel prefill, the new N-way RDMA all-reduce. Developed and
  accepted on the project's own SPX1 checkpoint (+21% decode over TP2); with the public EXL3 checkpoint TP3 has not
  been measured yet. Two-node splits are unchanged.
- **One code base for three models.** Shared modules: PDL launch primitives, top-k threshold selection, mHC gates /
  Sinkhorn, the KV pool (now on CUDA virtual memory, growing in place), multimodal canvases, and the NVMe prefix-cache
  core (with a byte-exact regression test, `spark-engine pcache-golden`).
- **GLM serving:** fork checkpoints (`GLM53_SERVE_FORK_CKPT=2048`), growable KV stores (`GLM53_KV_VMM=1`), lazy store
  warm-up after ready, Sinkhorn on its own CTAs (L0). A prefill fix for chunk tails with mixed-K experts.
- **SPX1 code path** (per-tensor non-expert weight format with rotated decode GEMV and INT8 prefill) is included; the
  SPX1 checkpoints themselves are not published, and the default profile uses the public EXL3 checkpoint.
- **Ablation:** a conditional clamp (`SPARK_ABLATE_CLAMP`, `DS41_ABLATE_CLAMP`) next to the two existing mechanisms;
  all off by default (docs/abliteration.md).
- **Scripts:** `spark.env` gains `SPARK_WORKERS`, `SPARK_MASTER_ADDRS` and `SPARK_TP3_IFNAME` (the old `GLM_WORKER` /
  `GLM_MASTER_ADDR` still work); the GLM profile is now `profiles/glm.env` (`glm-tp2.env` sources it); graceful
  shutdown that lets the prefix cache finish its writes.

GLM-5.3-Flash and Qwen3.8-Flash-Next speeds in the README were measured with v0.1.1 and have not been re-measured with
this release.

What it runs:

- GLM-5.3-Flash (EXL3 4 bpw) on two (or three) DGX Sparks with DFlash2 speculative decoding.
- DeepSeek-V4.1-Flash-1017 on three DGX Sparks with DSpark speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs,
`reference/ds41` (Engram hash export). Install PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not
included.
