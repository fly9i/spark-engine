spark-engine v0.3.3 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.2: DeepSeek-V4.1-Flash startup (GLM-5.3-Flash and Qwen3.8-Flash-Next outputs were checked identical).
Outputs stay bitwise identical; decode speed is unchanged (docs/ds41.md §7).

- **Startup ~31 s → ~16 s** (cold start, launch to serving, three DGX Sparks):
  - Shard files are read ahead whole on a thread with O_DIRECT into huge-page host memory, and the loader copies its
    tensors out with a GPU kernel that reads that memory in place (GB10's CPU and GPU share it). The reads run at about
    the disk's speed (~11.7 GB/s) while the layers are processed: loading 14 s → 8 s (`DS41_LOAD_PREFETCH`).
  - The original checkpoint (Engram tables) is opened and the vision encoder loaded on a thread during the shard load.
  - Only the single-sequence decode / verify graphs are captured before serving; the batched ones are captured in the
    first ~5 s of idle time after, with their memory set aside when the KV pool is sized (`DS41_WARM_DEFER`):
    warm-up 8 s → 4 s.
  - `start-ds41.sh` starts the workers together and the front end alongside the engine (it listens once the engine is
    ready; `openai_server.py --ready-file`).
- **Fix:** the decode Engram upload no longer logs two `Tensor.pin_memory(device)` deprecation warnings per step.
