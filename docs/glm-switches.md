# GLM-5.3-Flash engine switches

These are the `GLM53_*` switches set by `engine-rs/profiles/glm-tp2.env`, the serving profile for GLM-5.3-Flash on two DGX Sparks (tensor parallelism 2). The values below are the tuned defaults. Most switches select a kernel, a data layout or a launch schedule; they stay switches so that each change can be A/B-tested against the path it replaced. A value of 0 keeps the alternative path available for comparison; it is not used in serving.

Precision classes (effect of the listed setting on outputs, relative to the path it replaces):

| Class | Meaning |
| --- | --- |
| L0 | Bitwise identical output. |
| L1 | Rounding-level differences (summation order, Half/TF32 operand rounding), not worse than the reference. |
| L2 | Drafter side only: drafts and acceptance may change, the verified output does not. |
| L3 | Lossy, accepted by design (Q8 dense decode weights, FP8 KV cache). |
| — | Not numeric (memory, capacity, serving, host-side work). |

For switches set to 0, the class describes the switched-off path ("L3 if on").

Notes:

- The server launcher (`serve/glm-rank.sh`) clears the caller's `GLM53_*` variables, sources this profile, then applies `spark.env`, whose `GLM53_*` entries override the profile. It also sets `GLM53_MAX_CONTEXT` to the KV token budget.
- The `serve` command always turns on `GLM53_MHC_FUSED`, `GLM53_KDA_FUSED`, `GLM53_MLA_LATENT` and `GLM53_PREFILL_BATCH`.
- "Rows" means the token rows of one call: 1 for plain decode, 2..8 for a verified draft chain, up to 32 for batched multi-sequence verification, hundreds to thousands for prefill chunks. "Decode-sized" means 1..32 rows.
- C12 is a lossless 12-bit coding of the Half weights; Q8 is int8 with one FP32 scale per 128 inputs. Routed experts always use the checkpoint's EXL3 4-bit weights.
- Cross-references inside the Meaning column omit the `GLM53_` prefix.

## Serving and memory (11)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_SERVE_MAX_SEQS` | `4` | Maximum number of sequences decoded concurrently (1..8). Sequences advance round-robin: one prefill chunk or one verification round per turn. | — |
| `GLM53_SERVE_BATCH` | `1` | Batched serving rounds: all active sequences are verified together in one forward (and their drafts proposed together) instead of one sequence per round. Without it concurrent requests are served one round at a time and total throughput does not grow with concurrency. | L0 |
| `GLM53_SERVE_MAX_STORES` | `4` | Maximum live sequence stores (each holds KV rows, KDA state, a drafter slot and its own verifier graphs; at least SERVE_MAX_SEQS). Finished stores are kept for exact-prefix reuse until their space is needed. | — |
| `GLM53_KV_POOL` | `1` | Allocate one KV pool at startup, sized from free memory, instead of allocating per sequence; both ranks take the smaller size, and the pool size sets the KV token budget. | — |
| `GLM53_MEM_UTIL` | `0.92` | Target whole-machine memory use (fraction of MemTotal, 0.5..0.98) used to size the KV pool. | — |
| `GLM53_MEM_RESERVE_GIB` | `5` | GiB left outside the KV pool for what serving allocates later: verifier graphs, KDA state and drafter windows of every store, prefill workspaces. | — |
| `GLM53_MAX_CONTEXT` | `20480` | Capacity of the MLA latent / DSA caches in tokens. The server launcher (serve/glm-rank.sh) overrides it with the KV token budget (GLM_KV_TOKENS, default 1048576); the profile value applies to offline tools. | — |
| `GLM53_PCACHE` | `1` | Persistent prefix cache on local NVMe (directory GLM53_PCACHE_DIR): prompt and turn-boundary checkpoints of at least 1024 tokens are written in the background (O_DIRECT, checksums, segments shared across turns, LRU up to 100 GiB) and restored when a later prompt extends them; restored state is bitwise the saved state. | L0 |
| `GLM53_PREFIX_ADMISSION` | `1` | Prefix pool: admit only the chunk snapshots that will survive eviction, skipping those that would be evicted immediately. | L0 |
| `GLM53_VISION` | `1` | Load the BF16 vision tower on rank 0; image placeholders are filled into the prefill embedding before its allreduce. Loaded before the KV pool is sized, so the pool accounts for it. | — |
| `GLM53_HOST_TRIM` | `1` | After loading, call malloc_trim(0) to return the host allocator's free pages (about 1.2 GiB per node). | — |
| `GLM53_RELEASE_FILE_CACHE` | `1` | After all weights are resident on the GPU, drop the page cache of the safetensors files (madvise DONTNEED + fadvise); the mappings remain only as a fallback. | — |

## Tensor parallelism and communication (allreduce, RDMA) (11)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_DENSE_TP` | `1` | Shard the non-expert layers across the two ranks: KDA and MLA by heads, dense and shared-expert MLPs by intermediate dimension; outputs are summed by an allreduce of FP32 partials. | L1 |
| `GLM53_VOCAB_TP` | `1` | Shard the embedding and LM head by vocabulary (lookups across shards). | L0 |
| `GLM53_RDMA_AR` | `1` | FP32 allreduces up to 1 MiB use the engine's own RDMA path: one RoCE RC queue pair, a ring of pinned host-mapped slots, a GPU kernel that copies and spins on the arrival flag, and a CPU proxy that posts the RDMA writes. The sum x0+x1 is bitwise equal to a 2-rank NCCL sum; other messages use NCCL. | L0 |
| `GLM53_RDMA_AR_INIT` | `1` | Set up the RDMA allreduce channel when the process group initializes (needed by RDMA_AR). | — |
| `GLM53_RDMA_AR_DUAL` | `1` | Open a second queue pair on the other RoCE port; messages of at least 32 KiB are split across both links. The same bytes land in the same receive slot. | L0 |
| `GLM53_AR_FUSED` | `1` | The RDMA allreduce only sends and waits; its consumer (the four-stream mHC post) adds the local partial and the peer's receive slot itself, with the same rounding in the same order. Requires RDMA_AR. | L0 |
| `GLM53_HOST_REGISTER` | `1` | RDMA host buffers are anonymous pages pinned with cudaHostRegister instead of cudaHostAlloc memory, so kernel memory compaction does not try (and fail) to migrate them. | — |
| `GLM53_TP_SMALL_COMM` | `1` | Create a second NCCL communicator limited to 4 CTAs for payloads up to 256 KiB (used when the RDMA path does not apply). | L0 |
| `GLM53_TP_SMALL_COMM_ACTIVE` | `1` | Route NCCL allreduces of up to 256 KiB to that small communicator (runtime toggle of TP_SMALL_COMM). | L0 |
| `GLM53_TP_MOE_PACK` | `1` | For calls of up to TP_MOE_PACK_MAX_ROWS rows, pack the routed-expert and shared-expert FP32 partials into one buffer and reduce them with one allreduce per MoE layer instead of two. | L0 |
| `GLM53_TP_MOE_PACK_MAX_ROWS` | `8` | Row limit for TP_MOE_PACK (larger batches measured slower). | L0 |

## Dense weights (C12 coding, Q8, FP8, cuBLAS) (30)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_W_FP16` | `1` | Resident precision of the large non-expert projection weights: FP16 converted from the BF16 checkpoint (0 = FP32). About 0.25% of values fall into FP16 subnormals. | L1 |
| `GLM53_BF16_RESIDENT` | `1` | Keep the router gate weights and the mHC hc_*_fn weights in BF16 instead of an FP32 copy; kernels widen them exactly in-register. | L0 |
| `GLM53_TF32` | `1` | FP32 GEMMs run with TF32 multiply inputs (FP32 accumulation unchanged). This lowers multiply-input precision. | L1 |
| `GLM53_C12` | `1` | C12 coding: resident Half weights are encoded losslessly in 12 bits at load time (sign, 7-bit mantissa, 4-bit exponent window; out-of-window values stored as BF16 escapes in CSR form). Decode-sized calls (1..32 rows) read the coded copy (0.75 of the Half bytes); KDA q/k/v and shared gate/up are each one launch. Bitwise equal to the skinny Half path; the framework that Q8 builds on. | L0 |
| `GLM53_C12_INDEX` | `1` | The DSA indexer projections (stored FP32 with BF16 values) use the coded decode path (C12, or Q8 when Q8 is on), keys and gate in one launch, with Half inputs instead of TF32. | L1 |
| `GLM53_C12_KS` | `8` | Split-K of coded decode GEMM launches: 8 wherever K allows, instead of the skinny Half path's choice (a different summation split). | L1 |
| `GLM53_C12_QU` | `2` | The C12 GEMV kernel processes two quads per iteration (only when every split holds an even quad count); MMA order unchanged. | L0 |
| `GLM53_Q8` | `1` | Decode-sized GEMMs of the dense, KDA, shared-expert, MLA, DSA-indexer and LM-head weights read int8 weights with one FP32 scale per 128 inputs (8.25 bits) instead of the lossless C12 copy; relative RMS error 0.65-0.69% per class. Prefill keeps the source-precision weights. Requires C12; classes selectable with GLM53_Q8_SET. | L3 |
| `GLM53_Q8_TILED` | `1` | Store Q8 weights in (16-row tile, 128 k) blocks so each warp load is one contiguous 512 B; same decoded operands, MMA order and reduction. | L0 |
| `GLM53_Q8_XHALF` | `1` | Q8 GEMMs of more than 8 rows read a Half copy of the input X, removing the per-warp FP32 load and conversion (same operands). | L0 |
| `GLM53_HALF_SKINNY` | `1` | Half-weight GEMMs of at most 16 rows use the engine's skinny MMA kernel (FP32 input rounded to Half in-kernel, FP32 accumulation) instead of GEMV or cuBLAS. | L1 |
| `GLM53_DENSE_GEMV` | `1` | Single-row Half projections (N >= 1536, K in {128, 1536, 4096}) not served by a coded path use a bandwidth-bound GEMV kernel (FP32 accumulation). | L1 |
| `GLM53_DENSE_LT` | `0` | cuBLASLt shape-table path for Half projections; off (no stable end-to-end gain). When on it also disables the plain-Half prefill helpers. | — (off) |
| `GLM53_DENSE_SMALL` | `0` | Experimental small-N two-row projection kernel; off. | — (off) |
| `GLM53_DENSE_FP8` | `0` | Weight-only FP8 (e4m3, per-output-channel scale) for the dense MLP layers 0-2. Off: these weights stay at source precision (Q8 for decode). | L3 if on |
| `GLM53_KDA_FP8` | `0` | Register the KDA q/k/v/o weights as FP8 together with DENSE_FP8. Off: they stay at source precision. | L3 if on |
| `GLM53_FP8_HEAD` | `0` | Weight-only FP8 LM head. Off: the head stays at source precision (Q8 for decode). | L3 if on |
| `GLM53_FP8_MLA` | `0` | Weight-only FP8 for the MLA projections (q_a, q_b, kv_a, o). Off. | L3 if on |
| `GLM53_FP8_SHARED` | `0` | Weight-only FP8 for the shared-expert gate/up/down. Off. | L3 if on |
| `GLM53_FP8_SKINNY` | `1` | FP8 weight GEMMs of at most 16 rows use the skinny mma.m16n8k16 kernel (split-K reduced inside the CTA in fixed order; a variant reads BF16 activations directly). With the target's FP8 classes off, the FP8 kernels in this group serve the drafter's FP8 weights. | L1 |
| `GLM53_FP8_WMMA` | `3` | Mode of the small-row FP8 kernel used where the skinny kernel does not apply: 3 = WMMA with K split into parts merged in FP32, paired e4m3 decoding. | L1 |
| `GLM53_FP8_SPLITS` | `8` | Number of split-K parts of that small-row FP8 kernel. | L1 |
| `GLM53_FP8_SMALL_TRANSPOSE` | `1` | Transposed weight-tile layout for the small-row FP8 kernel. | L0 |
| `GLM53_FP8_SMALL_PAD` | `1` | Padded tile layout for the small-row FP8 kernel. | L0 |
| `GLM53_FP8_LARGE` | `5` | FP8 calls of 17..128 rows (mode 5) use a tile-fused kernel that decodes FP8 in shared memory and feeds the MMA directly; larger calls go to FP8_BIG. | L1 |
| `GLM53_FP8_BIG` | `1` | Prefill-sized FP8 calls (> 128 rows) run a tensor-core GEMM that reads the resident FP8 weight directly, with the per-row scale and Half rounding fused; no Half weight is expanded. Bitwise equal to the expand path. | L0 |
| `GLM53_FP8_EPILOGUE` | `1` | For library GEMMs over 128 rows on FP8 weights, scale the FP32 output and round it to Half in place (no temporary tensors). | L0 |
| `GLM53_FP8_FREE_SOURCE` | `1` | Free the source-precision copies of FP8-registered weights (possible because FP8_BIG never needs them). | — |
| `GLM53_FP8_PREFILL_HALF` | `1` | Prefill-sized calls (> 128 rows) use the plain Half cuBLAS GEMM on retained source-precision weights instead of an FP8 path (FP8_BIG takes precedence where only the FP8 copy exists). Also a precondition of PREFILL_QKV_INTO and PREFILL_HALF_GLUE. | L1 |
| `GLM53_FP8_QKV_FUSED` | `1` | When KDA q/k/v are FP8, concatenate them by rows and run one skinny launch. No effect while KDA_FP8=0 (the C12/Q8 group launch covers this). | L0 |

## KDA linear attention (23)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_KDA_FUSED` | `1` | Fused KDA recurrent-state kernel for decode steps and verification (the server always turns it on). | L1 |
| `GLM53_KDA_FORK_FUSED` | `1` | A verification branch reads its parent's state H and writes the new H directly, instead of copying and then updating. | L0 |
| `GLM53_KDA_CHAIN_NORM` | `1` | Chain verification: the KDA normalizations of all chain nodes in one batched call. | L0 |
| `GLM53_KDA_CHAIN_RECURRENT` | `1` | Chain verification: the KDA recurrence of all chain nodes in one batched call. | L0 |
| `GLM53_KDA_CONV_CHAIN` | `1` | Chain verification: the KDA short convolution of the whole chain in one kernel. | L0 |
| `GLM53_KDA_CONV_DEFERRED` | `1` | The convolution window is not materialized during verification; it is written at commit for the accepted tokens. | L0 |
| `GLM53_KDA_CONV_COMMIT_ONE` | `1` | Commit the selected convolution window of every KDA layer in one launch (pure copies) instead of about three copies per layer. | L0 |
| `GLM53_KDA_CONV_CACHED` | `1` | Use the convolution weights concatenated at load time instead of a concatenation per call. | L0 |
| `GLM53_KDA_CONV_SILU` | `1` | Apply SiLU inside the convolution kernel (same formula as the separate op). | L0 |
| `GLM53_KDA_CONV_L2` | `1` | Deferred convolution + SiLU + q/k/v normalization in one launch. Needs KDA_CONV_DEFERRED, KDA_CONV_SILU and NORM_FUSED. | L0 |
| `GLM53_KDA_CORRECTION_REPLAY` | `1` | Chain verification records the KDA correction terms instead of materializing each node's state H; at commit the accepted prefix is replayed in place. | L0 |
| `GLM53_KDA_CORR_WARP` | `4` | Correction-chain kernel choice per launch: 4 = thread-per-row kernel for 3 or more tokens, warp-per-row kernel otherwise (bitwise identical kernels). | L0 |
| `GLM53_KDA_GATE_FUSED` | `1` | Gate fusion for up to 8 rows: [fa;ga] and wb in one kernel producing beta, fb/gb in one kernel producing decay and sigmoid(g2), o-norm times gate in one kernel. | L1 |
| `GLM53_KDA_GATE_ONE` | `0` | Run the gate stages in one persistent kernel; off (measured slower). | L0 if on |
| `GLM53_KDA_GATE_WIDE` | `1` | Callers that would split 9..32 rows into 8-row pieces pass up to 32 rows to the fused gate in one call. | L0 |
| `GLM53_KDA_GATE_TILE` | `1` | 9..32-row gate calls use a tiled kernel: each block computes 8 outputs over one quarter of K with X staged in shared memory. | L0 |
| `GLM53_KDA_GATE_PT` | `1` | Gate GEMVs (1..32 rows) with one thread per (row, output) and weights converted once per block into shared memory; same lane summation order. | L0 |
| `GLM53_KDA_GATE_SIDE` | `1` | Inside the CUDA graph, the gate kernel chain runs on a side stream while the q/k/v projection runs on the main stream (scheduling only; only engine kernels run on the side stream). | L0 |
| `GLM53_KDA_MULTI_LAUNCH` | `1` | Batched multi-sequence verification: the convolution chains and correction chains of up to 8 sequences run as one launch each. | L0 |
| `GLM53_KDA_ONORM_SIGMOID` | `1` | Keep the raw g2; the o-norm gate kernel applies the sigmoid in place. | L0 |
| `GLM53_NORM_FUSED` | `1` | One kernel each for KDA q/k l2 normalization (with the v copy), the MLA RMSNorm, and Half rounding of row outputs. | L1 |
| `GLM53_KDA_PREFILL_FUSED` | `1` | Prefill: convolution + SiLU + l2 normalization in one kernel and the decay in one kernel that replays the original op sequence, without the [T+3, C] concatenation. | L1 |
| `GLM53_KDA_SEQUENCE` | `6` | Prefill recurrence kernel mode: 6 = four lanes per row, float4-interleaved blocks, four blocks per head. | L1 |

## MLA / DSA attention (37)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_MLA_LATENT` | `1` | Compressed-latent MLA: 512-dim latent cache with the K/V projections absorbed, incremental DSA selection, graph-compatible bounded caches (the server always turns it on). | L1 |
| `GLM53_KV_FP8` | `1` | Latent KV rows stored as 512 FP8 e4m3 values plus 4 FP32 tile scales (528 bytes), roughly halving KV memory. | L3 |
| `GLM53_MLA_SPARSE_FUSED` | `1` | Fused sparse latent attention that reads the compressed latent directly instead of gathering FP32 K/V. | L1 |
| `GLM53_MLA_ACTIVE_COPY` | `1` | Latent/DSA pool snapshots copy only the valid rows (length read on the GPU, so graph replays stay dynamic) instead of the full capacity. | L0 |
| `GLM53_MLA_WEIGHT_CACHE` | `1` | Cache the derived MLA projection weights (computed once, shared). | L0 |
| `GLM53_MLA_HALF_BMM` | `1` | Absorb/expand products with the derived per-head weights as batched skinny Half MMAs. | L1 |
| `GLM53_MLA_BMM_C12` | `1` | Absorb/expand read a C12 coding of the per-head Half weights (0.75 of the bytes); bitwise the Half BMM result. | L0 |
| `GLM53_MLA_BMM_WIDE` | `1` | The C12 absorb/expand kernel takes up to 32 rows per call, so batched verification reads each head's weights once instead of once per 8-row piece. | L0 |
| `GLM53_MLA_CHAIN_SHARED` | `1` | A chain verifier node aliases its base's latent cache and pools instead of copying them per node (it keeps its own copy of the one pool row later nodes may overwrite). | L0 |
| `GLM53_MLA_NODE_BATCH` | `1` | Chain verification with FP8 latent: all node appends in one launch and the chain's latent rows in one FP8 store. | L0 |
| `GLM53_MLA_MULTI_LAUNCH` | `1` | Batched multi-sequence verification: latent append, FP8 store, DSA score, top-k and attention of all sequences in one launch each. | L0 |
| `GLM53_MLA_COMMIT_FUSED` | `1` | Commit the chain-shared latent rows, tails and lengths of every MLA layer in one launch. | L0 |
| `GLM53_MLA_INDEX_SIDE` | `1` | Inside the CUDA graph, the DSA indexer projections run on a side stream while the main stream computes q_b and the absorb (scheduling only). | L0 |
| `GLM53_MLA_PREFILL_BATCHED` | `1` | Prefill chunks of at least PREFILL_MIN_ROWS rows compute MLA/DSA causally in batch instead of per token. | L1 |
| `GLM53_MLA_PREFILL_TC` | `1` | Prefill shared-latent attention on tensor cores (one query x 16 heads per CTA; mode 8, mode 9 with the FP8 KV cache). Required with KV_FP8. | L1 |
| `GLM53_MLA_PREFILL_F16` | `1` | The tensor-core prefill attention in FP16 form (Q and probabilities in FP16, FP32 accumulation; mode 10, or 11 with the FP8 KV cache). | L1 |
| `GLM53_MLA_TC16_HG` | `2` | The FP16 prefill attention kernel processes 2 groups of 16 heads per CTA, sharing each gathered key tile. | L0 |
| `GLM53_MLA_PREFILL_DENSE` | `6` | Prefill shared-latent attention mode used when MLA_PREFILL_TC is off (6 = several heads share an FP16 latent tile in shared memory, FP32 scores). Not used in this profile because MLA_PREFILL_TC=1. | L1 |
| `GLM53_MLA_PREFILL_STRIDED` | `1` | Prefill absorb/expand as strided TF32 GEMMs that read q in place and write the o_proj layout directly (no transpose copies). | L1 |
| `GLM53_MLA_SCORE_2D` | `1` | Compute MLA scores as a 2D shared-key FP32 GEMM over the head-strided slices; limited by MLA_SCORE_2D_SCOPE and MLA_SCORE_2D_MIN_ROWS. | L1 |
| `GLM53_MLA_SCORE_2D_MIN_ROWS` | `2048` | Minimum rows for MLA_SCORE_2D. | L1 |
| `GLM53_MLA_SCORE_2D_SCOPE` | `prefill` | Scope of MLA_SCORE_2D: prefill only (in decode it lowered acceptance). | L1 |
| `GLM53_DSA_INDEX_FUSED` | `1` | Fuse the integer/mask bookkeeping of ranked DSA selection (scores and top-k unchanged). | L0 |
| `GLM53_DSA_KEY_FUSED` | `1` | The indexer key LayerNorm and the gate's contiguous copy in one launch. | L1 |
| `GLM53_DSA_NODE_FUSED` | `1` | One launch per verifier node for the tail copies, tail slot writes, provisional pool (softmax over four slots) and pool write, latent row write and length update. | L0 |
| `GLM53_DSA_POSITION_CAPTURE` | `1` | Capture the pre-append positions inside the existing mask producer on the same stream instead of a copy per node. | L0 |
| `GLM53_DSA_SCORE_FUSED` | `3` | Per-node DSA scoring mode: 3 = one launch instead of five, reading only complete pools, with lengths taken on the device. | L1 |
| `GLM53_DSA_SCORE_MULTI` | `1` | Score all nodes of a chain in one pass over the shared pools, after all appends (needs DSA_NODE_FUSED and MLA_CHAIN_SHARED). | L0 |
| `GLM53_DSA_TOPK_BATCH` | `1` | Batch the top-k of the ranked verifier (per-row score arithmetic unchanged). | L0 |
| `GLM53_DSA_TOPK_FAST` | `1` | DSA top-512 over the valid prefix only, in the reference library's exact order, with scores written straight into the masked buffer. | L0 |
| `GLM53_DSA_ALL_VISIBLE` | `0` | For short contexts, select all visible keys instead of ranked selection (changes FP32 summation order); off. | L1 if on |
| `GLM53_DSA_VISIBLE_DIRECT` | `0` | Direct attention path for the all-visible case; off. | L1 if on |
| `GLM53_DSA_PREFILL_LIMIT` | `1` | Prefill DSA scores only the pools complete at each query block's boundary, keeping the full-width padding and top-k semantics. | L1 |
| `GLM53_DSA_PREFILL_QBLOCK` | `auto` | Queries per prefill DSA block: auto sizes the block so the [block, pools] score matrix stays near 64M floats (1024 at up to 64K pools). | L0 |
| `GLM53_DSA_PREFILL_SCORE_FUSED` | `1` | Prefill DSA index scores computed without materializing the [n, heads, pools] tensor. | L0 |
| `GLM53_DSA_PREFILL_SCORE_TILED` | `1` | Tiled prefill DSA scoring kernel that also writes the visibility mask itself. | L0 |
| `GLM53_DSA_PREFILL_TOPK_FAST` | `1` | Prefill DSA top-k (widths >= 1024) by the exact fast top-k kernel over each row's visible prefix plus a token-expansion kernel, instead of the generic masked top-k and index arithmetic. | L0 |

## mHC hyper-connections (10)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_MHC_FUSED` | `1` | Fused mHC (manifold-constrained hyper-connection) pre/post implementation for decode (the server always turns it on). | L1 |
| `GLM53_MHC_PRE_FUSED` | `1` | mhc_pre for up to 16 rows in two kernels: K-sliced partial sums of the 24 mixes and the square sum, then per row rstd, gate, softmax, Sinkhorn, contraction and RMSNorm (replacing about 30 library ops). | L1 |
| `GLM53_MHC_PRE_TC` | `1` | mhc_pre partial sums on TF32 tensor cores (same layout). | L1 |
| `GLM53_MHC_PRE_LARGE` | `1` | mhc_pre for large (prefill) row counts: split-TF32 partial sums (three MMAs, FP32-level accuracy) with the original finish. | L1 |
| `GLM53_MHC_FINISH_REG` | `1` | mhc_pre_finish keeps the collapsed row in registers instead of writing and re-reading it. | L0 |
| `GLM53_MHC_POST_FUSED` | `1` | Fused mhc_post kernel (handles the broadcast embedding's real strides). | L0 |
| `GLM53_MHC_POST_FOUR_STREAMS` | `1` | The four-stream variant of the fused mhc_post kernel (all four hyper-connection streams of an element in one pass); required by MHC_POST_PACKED and AR_FUSED. | L0 |
| `GLM53_MHC_POST_PACKED` | `1` | mhc_post consumes the packed [routed, shared] allreduce buffer directly. | L0 |
| `GLM53_MHC_POST_PRE_FUSED` | `1` | Prefill (> 16 rows): mhc_post and the next layer's mhc_pre partial in two kernels instead of three passes over the residual. | L0 |
| `GLM53_MHC_GROUP_ROWS` | `128` | Prefill fused post+pre in groups of this many rows, so the finish re-reads the new residual from L2 instead of DRAM. | L0 |

## MoE experts and router (21)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_MOE_COOP` | `1` | Decode and verification run the routed experts on the engine's own EXL3 MoE kernels (shim/moe_exl3.cuh), in launches of up to 32 rows. The name is historical. | L1 |
| `GLM53_MOE_BATCH` | `0` | Flat batched EXL3 expert path in 16-row pieces, used only when MOE_COOP is off; off. | — (off) |
| `GLM53_MOE_SCRATCH` | `zeros` | Initialization of the scratch buffers of the per-row EXL3 fallback path: zeros, empty or reuse. | — |
| `GLM53_MOE_NO_COPY` | `1` | When the routed experts produce a single output tensor, use it directly (no concatenation or extra copy). | L0 |
| `GLM53_MOE_INPUT_HALF_REUSE` | `1` | Routed and shared experts share one Half conversion of the input. | L0 |
| `GLM53_MOE_PACK_DIRECT` | `1` | The routed expert kernel and the shared down projection write the two halves of the packed allreduce buffer directly (no concatenation). Needs AR_FUSED. | L0 |
| `GLM53_MOE_ROUTE_SIDE` | `1` | Inside the CUDA graph, the fused router runs on a side stream while the main stream computes the shared expert (scheduling only). | L0 |
| `GLM53_ROUTER_FUSED` | `1` | Fused router for up to 16 rows (up to 320 experts, top-8 sigmoid + bias): logits and selection in two kernels. | L1 |
| `GLM53_ROUTER_V2` | `1` | Router logits kernel with the row count as a template parameter and weights preloaded; same per-lane FMA order and shuffle tree. | L0 |
| `GLM53_ROUTER_WIDE` | `1` | Use the fused router also for 17..32 rows in batched multi-sequence verification, replacing the TF32 library router there. | L1 |
| `GLM53_ROUTER_HALF_W` | `1` | Router selection also writes the Half routing weights the expert kernel consumes (no separate conversion). | L0 |
| `GLM53_SHARED_GU_FUSED` | `1` | Fused native-Half shared-expert gate/up. | L1 |
| `GLM53_SHARED_GU_ROWS` | `1` | FP8 shared gate and up in one row-concatenated skinny launch (the C12/Q8 path uses its own group launch). | L0 |
| `GLM53_SHARED_GU_F32` | `1` | The shared gate/up activation is also written as FP32 of the Half value, so the coded down projection needs no conversion launch. | L0 |
| `GLM53_PREFILL_COOP` | `1` | Prefill runs the routed experts on the engine's EXL3 MoE kernels (shim/moe_exl3.cuh): grouped per expert from 64 rows (see PREFILL_GROUPED), otherwise in launches of up to 32 rows (tails under 8 rows take the per-row EXL3 path). The name is historical. | L1 |
| `GLM53_PREFILL_GROUPED` | `recon` | Grouped prefill MoE for chunks of at least 64 rows (0 disables it, and with it sequence-parallel prefill). With the default kernels, recon and direct both run the grouped GEMM of moe_exl3.cuh; recon selects the reconstruct tier (decoded Half weights + cuBLAS) only when GLM53_MOE_RECON=1. | L1 |
| `GLM53_PREFILL_RECON_MIN_ROWS` | `64` | Reconstruct tier only (GLM53_MOE_RECON=1): experts with at least this many rows use decoded weights + cuBLAS, others the direct EXL3 GEMM. Inactive in this profile. | L1 |
| `GLM53_GROUPED_INDEX_PACK` | `1` | Reconstruct tier only: upload the gather/scatter indices of all experts once as one [2, T*8] tensor. Inactive in this profile. | L0 |
| `GLM53_GROUPED_SWIGLU` | `1` | Reconstruct tier only: gate/up conversion, clamp, SiLU, multiply and Half rounding in one kernel. Inactive in this profile. | L0 |
| `GLM53_GROUPED_REDUCE` | `1` | Reconstruct tier only: deterministic fixed-order reduction of the routed outputs (FP16 to FP32, Half routing weights, no atomics). Inactive in this profile. | L0 |
| `GLM53_PREFILL_EXPERT_STREAMS` | `0` | Run independent experts of the reconstruct tier on a small stream pool; 0 = off. | L0 if on |

## Prefill (14)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_PREFILL_CHUNK` | `4096` | Prefill chunk size in tokens; the server interleaves one chunk or one verification round per sequence turn. Chunk splits change GEMM shapes. | L1 |
| `GLM53_PREFILL_BATCH` | `1` | Chunked prefill reads each projection weight once per chunk and runs the forward batched (KDA recurrence and DSA causality kept in order); the server always turns it on. | L1 |
| `GLM53_PREFILL_MIN_ROWS` | `128` | Prefill chunks of at least this many rows use the KDA sequence kernel and batched MLA/DSA. | L1 |
| `GLM53_PREFILL_LAST_LOGITS` | `1` | Compute logits only for the last prompt row (state and drafter features still cover all rows). | L0 |
| `GLM53_PREFILL_SP` | `1` | Sequence-parallel prefill on two ranks (chunks of at least 128 rows, even count): the residual is split by rows, and reduce-scatter/all-gather replace allreduce. | L1 |
| `GLM53_PREFILL_SP_HALF_GATHER` | `1` | Sequence-parallel prefill: route the owned rows, then all-gather the Half expert input and the packed routes instead of the FP32 input. | L1 |
| `GLM53_PREFILL_MOE_SUM1` | `1` | Prefill (> 64 rows): add the shared expert's FP32 partial to the routed partial and reduce once (one collective instead of two per MoE layer). | L0 |
| `GLM53_PREFILL_SHARED_FUSE` | `1` | The shared-expert partial is computed first and added inside the grouped routed reduction, which writes the reduce-scatter input directly. | L0 |
| `GLM53_PREFILL_SHARED_HALF` | `1` | Prefill shared expert as Half gate/up GEMMs plus one SwiGLU kernel writing the Half down input, without FP32 gate/up tensors. | L0 |
| `GLM53_PREFILL_HALF_REUSE` | `1` | The shared expert reuses the routed lane's Half conversion of its input. | L0 |
| `GLM53_PREFILL_FEATURE_REUSE` | `1` | Drafter features are taken from the residual written by the next layer's fused post+pre instead of an extra mhc_post. | L0 |
| `GLM53_PREFILL_QKV_INTO` | `1` | Prefill projections on the plain Half GEMM path write straight into column slices of one concatenated output (no per-projection FP32 intermediates or concatenation). | L0 |
| `GLM53_PREFILL_HALF_GLUE` | `1` | KDA prefill: q/k/v and gate GEMM results stay Half (no FP32 round trips) and the o-norm gate writes the Half input of the output projection. | L0 |
| `GLM53_HALF_INPUT_CACHE` | `1` | Prefill-sized inputs feeding several Half GEMMs are converted to Half once and reused (checked by storage version). | L0 |

## Speculative decoding and drafter (DFlash2) (47)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_SPEC_MAX_DRAFT` | `7` | Maximum draft chain depth (1..7). | L2 |
| `GLM53_DEPTH_RULE` | `rate` | Single-sequence verification depth: rate = choose the depth that maximizes expected emitted tokens minus lambda times the round time, from calibrated prefix-acceptance probabilities (otherwise: stop at the first draft with confidence below tau). | L2 |
| `GLM53_DRAFT_CONF_TRUNC` | `1` | Compute a confidence per draft (softmax probability of the chosen token within the top 16) so the verified depth can be cut per round (tau default 0.7); fixed-chain graphs for all depths are cached and pre-captured. | L2 |
| `GLM53_SERVE_BATCH_CUMCONF` | `0.5` | Batched multi-sequence verification: each chain stops where the running product of its draft confidences drops below this value. | L2 |
| `GLM53_COPY_DRAFTS` | `1` | Copy (prompt-lookup) drafts: when the last 8 context tokens occurred earlier in prompt + reply, the round verifies what followed them instead of the drafter chain. | L2 |
| `GLM53_COPY_MAX` | `15` | Length of a long copied chain (up to 15): a copied chain is at most 7 drafts, or exactly this many when that many tokens follow (one extra 16-row chain graph per store). | L2 |
| `GLM53_CHAIN_SHARED_BASE` | `1` | Verification graphs of every chain depth share one committed-state storage, so switching depth needs no state restore. | L0 |
| `GLM53_STATE_INPLACE` | `1` | At commit, accepted-node state is written directly into the verifier graph's base (KDA correction replayed in place, convolution window, only new MLA rows copied); the next round detects the alias and skips the restore. | L0 |
| `GLM53_MULTI_HOST_ROOM` | `1` | Batched multi-sequence verification checks latent capacity once per round from host-known lengths instead of a device-to-host read per layer and sequence. | L0 |
| `GLM53_TARGET_TOP1_TP` | `1` | Exact verifier top-1 over the vocabulary-sharded head without a full-vocabulary reduction. | L0 |
| `GLM53_GUMBEL_FUSED` | `1` | For sampled requests, the Gumbel noise is evaluated inside the verifier top-1 kernel (FP32 screening, exact FP64 for candidates). | L0 |
| `GLM53_ACCEPTED_FEATURE_VIEW` | `1` | Pass the accepted prefix of the fixed-chain feature matrix to the drafter as a view (no concatenation). | L0 |
| `GLM53_SPEC_MODES` | `batch-graph-chain` | Speculative modes run by the offline speculative check/benchmark tool; batch-graph-chain = fixed chains verified with CUDA graphs (the mode the server uses). | — |
| `GLM53_SPEC_GRAPH_SLOTS` | `4` | Tree mode only: number of LRU slots for captured tree graphs. Not used with chain verification. | L0 |
| `GLM53_SPEC_TREE_CAPTURE_AFTER` | `3` | Tree mode only: capture a graph after a tree topology has been seen this many times. | L0 |
| `GLM53_SPEC_TREE_CAPTURE_MIN_REMAINING` | `128` | Tree mode only: admit a new tree topology for capture only when at least this many tokens remain. | L0 |
| `GLM53_SERVE_DRAFT_BATCH` | `1` | Several sequences' drafter proposals and context appends in one drafter forward, so drafter weights and head are read once per round. | L2 |
| `GLM53_DRAFT_MANY_GRAPH` | `1` | The 2..4-sequence drafter proposal is captured as a CUDA graph per slot combination and replayed. | L0 |
| `GLM53_DRAFT_GRAPH` | `1` | Capture the whole drafter proposal as one CUDA graph (fixed-address KV slab slots, device-side metadata, flash-decoding attention kernel). | L2 |
| `GLM53_DRAFT_APPEND_GRAPH` | `1` | Drafter context appends as CUDA graphs, one per row count 1..8. | L0 |
| `GLM53_DRAFT_APPEND_TRIM` | `1` | Before fc, drop rows beyond the drafter window; a full-window replacement does not copy the old KV (absolute positions and logical length kept). | L0 |
| `GLM53_DRAFT_KV_BUFFER` | `1` | Drafter KV appends go into a fixed-address slab buffer. | L0 |
| `GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT` | `2048` | Use the slab only for contexts of at least this many tokens. | L0 |
| `GLM53_DRAFT_ATTN_TP` | `1` | Shard drafter attention by heads across the ranks (q/k/v by head, o by column, one FP32 allreduce); halves the drafter KV per slot. | L2 |
| `GLM53_DRAFT_MLP_TP` | `1` | Shard the drafter MLP (gate/up by intermediate dimension, FP32 partial of down reduced across ranks, then BF16). | L2 |
| `GLM53_DRAFT_MLP_SHARD_LOAD` | `1` | Load only this rank's shard of the drafter MLP (no full load first). | L0 |
| `GLM53_DRAFT_HEAD_TP` | `1` | Shard the drafter LM head by vocabulary, keeping the full candidate-selection interface. | L2 |
| `GLM53_DRAFT_HEAD_SHARD_LOAD` | `1` | Load only this rank's shard of the drafter head. | L0 |
| `GLM53_DRAFT_TOPK_TP` | `1` | Distributed top-16 of the drafter head: only per-rank candidates are exchanged. | L2 |
| `GLM53_DRAFT_HEAD_INT4` | `1` | Coarse INT4 scores of this rank's drafter-head slice preselect 64 candidates per row; those rows of the FP8 head are rescored with the same kernel and the top 16 returned. | L2 |
| `GLM53_DRAFT_HEAD4_TILED` | `1` | The drafter's INT4 head copy stored in the tiled order the INT4 MMA kernel reads (drafts unchanged). | L0 |
| `GLM53_DRAFT_FP8_ATTN` | `1` | Drafter q/k/v/o projections as weight-only FP8 (skinny kernel reading BF16 activations). | L2 |
| `GLM53_DRAFT_FP8_CONV` | `1` | Drafter attention/MLP convolution projections as weight-only FP8. | L2 |
| `GLM53_DRAFT_FP8_FC` | `1` | Drafter fc projection as weight-only FP8. | L2 |
| `GLM53_DRAFT_FP8_HEAD` | `1` | Drafter LM head as weight-only FP8. | L2 |
| `GLM53_DRAFT_FP8_MLP` | `1` | Drafter MLP gate/up/down as weight-only FP8. | L2 |
| `GLM53_DRAFT_Q4` | `1` | Drafter body weights (every DRAFT_FP8_* class except the head) are also held as affine 4-bit and read by decode-sized calls (1..32 rows). | L2 |
| `GLM53_DRAFT_SKINNY32` | `1` | Raise the drafter BF16-activation FP8 skinny kernel's row limit from 16 to 32 (for batched multi-sequence drafting). | L2 |
| `GLM53_DRAFT_GQA` | `5` | Drafter attention mode 5: four query heads share one KV head; short histories (up to DRAFT_GQA_SHORT_MAX) use a fused two-segment BF16 kernel, long ones grouped strided BMMs. | L2 |
| `GLM53_DRAFT_GQA_SHORT_MAX` | `128` | History length threshold for the short GQA path. | L2 |
| `GLM53_DRAFT_GQA_PRECISE` | `1` | Long-history grouped probability x value product in FP32 (TF32 disabled for that product). | L2 |
| `GLM53_DRAFT_FUSED_NORM` | `1` | Fused drafter small kernels: add + RMSNorm (and related element-wise steps) in one kernel per row. | L2 |
| `GLM53_DRAFT_NORM_CACHE` | `1` | Cache the drafter's pre-expanded norm weights. | L0 |
| `GLM53_DRAFT_ROPE_CACHE` | `1` | Reuse the drafter RoPE position and trigonometric tables. | L0 |
| `GLM53_DRAFT_CONV_FUSED` | `1` | Fused drafter Conv::convolve (projection, norm and residual unchanged). | L0 |
| `GLM53_DRAFT_FINAL_NORM_SELECT` | `1` | The drafter's final residual + norm skips a BF16 residual cast that has no consumer. | L0 |
| `GLM53_DRAFT_SELECTOR_FUSED` | `1` | Fused drafter path selection (same edge/ID semantics). | L0 |

## Graphs and misc (3)

| Variable | Value | Meaning | Class |
| --- | --- | --- | --- |
| `GLM53_GRAPH` | `1` | Use CUDA graphs for decode/verification (enabled when set; tensor parallelism 2 only, since all experts must be resident). | L0 |
| `GLM53_GRAPH_SHARED_WS` | `1` | One cuBLAS/cuBLASLt workspace shared by every graph instead of one per capture stream (about 32 MiB per graph). | — |
| `GLM53_STATIC_TENSORS` | `0` | Make constant tensors (decay base, convolution weights, DSA constants) static; off (no measurable gain). | L0 if on |
