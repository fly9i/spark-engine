spark-engine v0.3.2 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.1: DeepSeek-V4.1-Flash only (GLM-5.3-Flash and Qwen3.8-Flash-Next outputs were checked identical).
Outputs stay bitwise identical. Measured on three DGX Sparks, served, with the 1017 checkpoint (docs/ds41.md §7),
median of 3 runs:

- **Speculative decoding verifies more drafts per round:** the cost model that picks each sequence's draft count
  (`DS41_SPEC_COST`) was refitted to the faster decode steps of v0.3.1 (`24,0.6,5` → `22,0.5,2.5`). It changes only
  how many drafts are verified, never the output. Prose 59.1 → 62.4 tok/s for one stream, 82.1 → 89.6 for two,
  105.5 → 112.6 for four; structured 115.5 / 158.4 / 197.9; code 100.2 / 133.3 / 148.3 (code for one stream is within
  run-to-run spread of v0.3.1; in a same-session A/B the refit gave 100.3 → 101.6).
- **Fix:** with several sequences, the shared expert's parallel stream could still be using mHC-mix scratch buffers that
  the main stream had already released and reallocated (rare corruption of batched decode). The buffers are now held
  until the streams join.
- **Engram rows for decode** are uploaded through pinned memory without blocking, so the upload no longer waits for all
  queued work; their gather starts on background threads at the beginning of each step.
- **Experimental, off by default:** `DS41_MID_GRAPH=1` captures the attention middle (sparse attention and window
  commit) of the layers that are neither KV nor index sources into the decode / verify graphs, with per-step device
  descriptors. It is bitwise identical but not faster yet; the remaining idle time is in the source layers.

Prefill is unchanged (~5,400 tok/s for 8K–64K prompts).
