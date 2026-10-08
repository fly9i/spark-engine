# How the engine works

spark-engine is one Rust program (`spark-engine`) with two model paths: `serve` runs GLM-5.3-Flash on two DGX Sparks,
`qwen-serve` runs Qwen3.8-Flash-Next on one. A Python process in front of each speaks the OpenAI API, owns tokenization,
chat templates, reasoning parsing, tool-call parsing and media decoding, and talks to the engine over a Unix socket.

## Precision classes

Every optimization is classified, and the classification decides how it is tested:

| Class | Meaning | Acceptance |
| --- | --- | --- |
| L0 | bitwise identical outputs (schedule, layout or fusion changes) | logits, features and states compared bit for bit |
| L1 | rounding-level differences, not worse than the reference | error against an FP32 reference not larger than before |
| L2 | drafter side only: drafts may change, verified output cannot | verified tokens unchanged |
| L3 | lossy by design (chosen trade-off) | quality measured against the reference model |

The lossy choices in the default configuration: FP8 KV cache (both models); Q8 dense decode weights for GLM (int8 with
one FP32 scale per 128 inputs, about 0.7 % relative error); the checkpoints' own EXL3 quantization.

## Shared kernels

- **EXL3 decoding.** EXL3 stores each 16×16 weight tile as a trellis bitstream; values come from a codebook (`mcg` for
  GLM's experts, `mul1` for Qwen) and are decoded straight into tensor-core (mma.m16n8k16) fragments. A linear layer is
  `y = had128(had128(x·suh) · W) · svh` (128-point Hadamard transforms with sign/scale vectors), split along K for small
  row counts, and summed in a fixed order (deterministic).
- **MoE (`shim/moe_exl3.cuh`).** One template, instantiated for both models (expert count, top-k, sizes, codebook,
  activation): routed (row, expert) pairs are grouped by expert on the device with fixed grids (CUDA-graph friendly),
  gate/up products run as grouped GEMVs (decode) or GEMMs (prefill), SwiGLU and the down projection's input transform are
  fused into one pass, and each row's experts are summed in slot order.
  - GLM decode uses a persistent, fused kernel: gate/up, SwiGLU and down for all routed experts in one launch, with a
    per-expert ready flag instead of a kernel boundary, plus programmatic dependent launches between the remaining small
    kernels.
  - GLM prefill uses the fact that all experts of a layer share one input scale vector: the input transform runs once
    per token instead of once per (token, expert), and a paired gate/up GEMM (64 rows × 128 columns per CTA, two CTAs per
    SM) writes the activation of the down projection directly.
- **Graphs.** Every decode/verify shape is captured as a CUDA graph; allocations inside graphs come from per-graph pools.

## GLM-5.3-Flash (TP2)

Architecture handled: 45 layers mixing KDA (kimi-delta linear attention) and MLA (multi-head latent attention with a DSA
sparse indexer), 4-stream hyper-connections (mHC), 288 routed experts + a shared expert per MoE layer, dense MLP in the
first layers.

- **Tensor parallelism over two nodes.** Attention heads, the shared expert and the expert intermediate dimension (1024 of
  2048 per rank) are split; each layer needs a few all-reduces of small tensors. NCCL handles large messages; the engine's
  own RDMA all-reduce over both ConnectX-7 ports handles small ones (12–50 µs, fixed summation order, bitwise equal to
  NCCL), fused with the hyper-connection update that consumes it.
- **Weights.** Routed experts stay in the checkpoint's EXL3 4 bpw. Dense layers decode from Q8 copies (tiled so that
  each warp load is 512 contiguous bytes); a lossless 12-bit coding (C12, every decoded value equal to the original
  half) is available as an alternative. Prefill uses the dense weights at source precision with cuBLAS.
- **Speculative decoding.** The DFlash2 drafter (a small model conditioned on the target's hidden features) proposes
  chains/trees of tokens; the target verifies them in one forward. Verification keeps exact greedy semantics:
  - KDA states advance through a deferred correction replay, so rejected drafts cost no extra state copies;
  - MLA verification reads a shared base state for all branches;
  - DSA selection runs per verified row;
  - draft depth follows the drafter's confidence (product stop rule), and copy drafts reuse earlier context when the
    recent tokens repeat (code, JSON, quoting).
  Several sequences are verified in one batched forward.
- **Serving.** Up to 4 concurrent sequences by default, prefix reuse between requests (in-memory stores plus an optional
  persistent prefix cache on NVMe), a 1M-token KV budget, image and video input through the model's vision tower.

## Qwen3.8-Flash-Next (one node)

Architecture handled: gated DeltaNet linear-attention layers and sparse-attention layers (QSA, with a learned indexer
selecting key blocks), 512-expert top-10 MoE with a gated shared expert, 4-stream hyper-connections, per-layer n-gram
embeddings (PLE) from a 39 GB table, an MTP head, and a ViT vision tower.

- **Kernels.** All EXL3 products run on the engine's kernels (no exllamav3 runtime): fused multi-linear decode launches
  that share the input transform, GDN recurrence and convolution kernels with multi-sequence segments, QSA pooling,
  selection and attention kernels, fused hyper-connection mixes. The n-gram embedding rows are gathered from the memory
  mapped table with prefetch hints on parallel threads.
- **Speculative decoding.** The native MTP head drafts autoregressively; its output layer is a 4-bit copy of the
  lm_head restricted to the 65,536 most frequent tokens (`assets/qwen38/draft_vocab_65536.json`, tiled layout), so a
  draft step reads 84 MB instead of the full head. Prompt-lookup drafts are used where the context repeats. Up to 8
  sequences are verified in one batched forward.
- **Persistent prefix cache.** Optional NVMe prefix cache (`QWEN_PCACHE`), mirroring the GLM one: each boundary checkpoint is written to disk, and a later prompt that extends a cached prefix restores it instead of re-prefilling.
- **Long context and multi-turn.** A 1M-token KV pool in 16K-token granules (LRU); YaRN scaling only for sequences that
  pass 262,144 tokens. Each sequence store keeps a checkpoint at the last message boundary of its prompt, so a follow-up
  turn re-prefills only the new message even when the client re-serializes earlier turns. `max_tokens` reserves at most
  32K tokens up front; a sequence's KV range grows in place or moves while it decodes.
- **Vision.** Images (up to 16,384 tokens each) and video (2 fps, up to 768 frames) go through the ViT inside the engine,
  with interleaved multimodal RoPE positions.
- **Reasoning levels.** `reasoning_effort` maps none / low / medium / high to the chat template's thinking modes.

## Memory on GB10

GPU and CPU share 128 GB. The engine sizes the KV pool from `MemAvailable` after loading, releases the page cache of
the checkpoint shards, keeps host staging buffers small and pinned (anonymous pages registered with `cudaHostRegister`,
which kernel compaction never isolates), and the launch scripts compact memory once after start-up. See
[deploy.md](deploy.md) for the recommended host settings.
