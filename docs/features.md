# How the engine works

spark-engine is one Rust program (`spark-engine`) with three model paths: `serve` runs GLM-5.3-Flash on two or three
DGX Sparks, `ds41-serve` runs DeepSeek-V4.1-Flash on three, `qwen-serve` runs Qwen3.8-Flash-Next on one. Which model,
how many nodes and which switches are configuration only (start scripts, profiles, `spark.env`), never separate builds. A Python process in front of each speaks the OpenAI API, owns tokenization,
chat templates, reasoning parsing, tool-call parsing and media decoding, and talks to the engine over a Unix socket.

## Precision classes

Every optimization is classified, and the classification decides how it is tested:

| Class | Meaning | Acceptance |
| --- | --- | --- |
| L0 | bitwise identical outputs (schedule, layout or fusion changes) | logits, features and states compared bit for bit |
| L1 | rounding-level differences, not worse than the reference | error against an FP32 reference not larger than before |
| L2 | drafter side only: drafts may change, verified output cannot | verified tokens unchanged |
| L3 | lossy by design (chosen trade-off) | quality measured against the reference model |

The lossy choices in the default configuration: FP8 KV cache (GLM and Qwen); Q8 dense decode weights for GLM (int8 with
one FP32 scale per 128 inputs, about 0.7 % relative error); the checkpoints' own EXL3 quantization. DeepSeek runs the official checkpoint's formats
(MXFP4 experts, FP8 weights and KV, FP4 compressed KV) as specified.

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
- **Modules shared by all models.** `shim/pdl.cuh` (programmatic dependent launch primitives), `shim/topk_select.cuh`
  (one CTA per row top-k threshold search: GLM's DSA and DeepSeek's indexer), `shim/mhc_common.cuh` (mHC gates, row
  softmax, Sinkhorn), `src/kv_pool.rs` (a KV pool of per-token arrays on CUDA virtual memory that grows in place),
  `src/mm_canvas.rs` (multimodal canvases and salted placeholders that keep image identity in the prefix key),
  `src/pcache_core.rs` (the NVMe prefix cache: O_DIRECT writer / reader threads, checksums, segments shared between
  checkpoints of the same conversation, LRU, start-up reconciliation across ranks; each model describes its checkpoint
  as row arrays + fixed regions). `spark-engine pcache-golden <dir>` is its regression test (byte-exact manifest in
  `engine-rs/testdata`).
- **N-way RDMA (`shim/rdma_arn.cu`).** For three nodes: each rank publishes its RoCE devices with their subnet through
  the process group's store; a link is a pair of devices on a shared subnet. All-reduce sums in FP32 in fixed rank
  order (deterministic, equal on every rank); large messages are pipelined; prefill uses row-wise reduce-scatter /
  all-gather with an optional BF16 wire for the peers' partial sums.

## GLM-5.3-Flash (TP2 / TP3)

Architecture handled: 45 layers mixing KDA (kimi-delta linear attention) and MLA (multi-head latent attention with a DSA
sparse indexer), 4-stream hyper-connections (mHC), 288 routed experts + a shared expert per MoE layer, dense MLP in the
first layers.

- **Tensor parallelism over two nodes.** Attention heads, the shared expert and the expert intermediate dimension (1024 of
  2048 per rank) are split; each layer needs a few all-reduces of small tensors. NCCL handles large messages; the engine's
  own RDMA all-reduce over both ConnectX-7 ports handles small ones (12–50 µs, fixed summation order, bitwise equal to
  NCCL), fused with the hyper-connection update that consumes it.
- **Three nodes.** The same profile and checkpoint; splits follow the world size (`crate::tp::Part`), and with two nodes
  every split is bitwise as before. KDA heads 22/21/21, MLA 32/16/16 (16-head units), each routed expert's 16 Hadamard
  blocks 6/5/5 (the 6-block share rotates with the expert, so every node does the same work per layer; a width-variable
  MoE kernel skips the unused blocks), prefill sequence-parallel with rows padded to a multiple of 3.
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
- **Serving.** Up to 4 concurrent sequences by default, prefix reuse between requests (in-memory stores, fork checkpoints
  where a prompt branches off a cached sequence, and a persistent prefix cache on NVMe), a 1M-token KV budget on growable
  stores, image and video input through the model's vision tower.

## DeepSeek-V4.1-Flash (TP3)

Architecture handled: 40 layers (0–19 a causal encoder, 20–39 a decoder) with single-pass mHC; compressed sparse
attention (64 query heads of 512 with one shared KV head, a 128-token sliding window per layer, compressed global KV in
FP4, an FP4 indexer selecting top-512 with a hierarchical candidate pool in the decoder); a 384-expert top-6 MoE with a
shared expert (MXFP4); Engram n-gram tables in layers 1 and 14; three DSpark draft stages; a ViT with an MLP aligner.
Deployment recipe: [ds41.md](ds41.md).

- **Split.** Attention by head groups (3/3/2 of 8; the grouped low-rank output projection stays local), routed and
  shared experts by the intermediate dimension (768 of 2304 per node, so every node reads exactly a third), vocabulary
  in 128-row blocks. Each rank loads a pre-split shard (`ds41-shard`), one contiguous read per tensor.
- **Kernels.** Block-scaled tensor-core MMA (`mma ... kind::mxf8f6f4.block_scale`, sm_121a) for the MXFP4 MoE and FP8
  GEMMs; multi-row block-scaled GEMVs with the activation quantization done inside each CTA, so 1–8 rows cost about the
  same as one; fused sparse attention (window + selected blocks, FP8/FP4 dequantization, sink) with a deterministic
  split merge, one launch for several sequences; fused mHC, RMSNorm/RoPE and indexer scoring.
- **Prefill.** CED: the encoder runs every token, the decoder replays only the last window (decoder SWA bounded
  replay, as in training). Encoder attention runs as two micro-batches so the all-reduce of one overlaps the compute of
  the other; Engram rows (≈12.7 KB per token from NVMe) are fetched with sorted, deduplicated O_DIRECT reads one chunk
  ahead.
- **Speculative decoding.** DSpark: the three draft stages give 5 positions per step, a Markov head corrects them and a
  confidence head estimates each position's acceptance; a cost model (`DS41_SPEC_COST`) picks each sequence's draft
  count; verification graphs are keyed by total rows, so sequences with different draft lengths share one graph.
- **Serving.** 4 concurrent sequences, a shared KV pool (1.1 M tokens), in-memory prefix stores and an NVMe prefix cache
  with checkpoints every 512 tokens inside prefill chunks; image input (image tokens skip Engram, use `bias_vl`);
  the official `encoding.py` builds prompts, DSML tool calls and thinking budgets.

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
