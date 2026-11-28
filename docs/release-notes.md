spark-engine v0.3.16 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.15: native `sm_121a` MXFP4 DSpark draft head (L2, 100% L0 bit-exact main-model output), 6× lighter
block-split BF16 All-Gather for Engram (`D8`/`P8`), batched multi-stream `mtp_seed`, recalibrated speculative
cost/confidence defaults, and a deterministic 64K cross-request lookup ring (with documented multi-tenant and
cold-benchmark opt-out).

### DeepSeek-V4.1-Flash optimizations (100% L0 bit-exact output)

- **Native `sm_121a` MXFP4 draft head** (`DS41_DRAFT_HEAD_FP4`, default `1`): adds GPU dense MXFP4 quantizer and
  shuffle-free `mxfp4_gemv_mx` MMA kernel (`shim/ds41_mx.cu`), shrinking DSpark's draft head from 220.9 MB (FP8) to
  117.3 MB (MXFP4) per rank (**−103.6 MB/rank VRAM**, **1.88× faster**, `1231.1 µs → 655.9 µs` at `m=5`). Main-model
  verification is untouched (22/22 greedy exact match, 0 draft/confidence graph mismatches).
- **6× lighter Engram RDMA & async gather (`D8`/`P8`)** (`DS41_ENGRAM_AG`, default `1`): replaces zero-padded full
  `[t, 25600]` Engram All-Reduce with direct BF16 epilogue + `[t, 8544]` block-split `rs_sp_all_gather` (6× fewer wire
  bytes), enables background async Engram host gather for tree verification and 17–32-row batches, and drops redundant
  `FP32 → BF16 → +0.0` conversion in prefill `sp_engram_kv`.
- **Batched multi-stream `mtp_seed`** (`DS41_MTP_SEED_BATCH`, default `1`): batches accepted hidden states across
  concurrent sequences into a single `main_proj`, single `3× wkv`, and single RDMA all-gather per round.
- **Recalibrated speculation cost & confidence defaults**: `DS41_SPEC_COST` updated to `21,0.5,2.2` and greedy
  `DS41_SPEC_CALIB` to `1.165,0.085;1,0` (fitted with `bench/ds41/spec_calib.py` under `DS41_DRAFT_HEAD_FP4=1`).
- **Cross-request rolling lookup ring** (`DS41_GLOBAL_LOOKUP`, default `1`): deterministic 64K-token ring updated
  identically on every TP rank, providing `>= 4`-gram draft fallback and tree branches for repeated Agent/template
  workloads. **Limitations**: outputs remain 100% L0 bit-exact (no content leakage), but set `DS41_GLOBAL_LOOKUP=0` in
  multi-tenant environments (timing side-channel) and when running repeated-prompt cold benchmarks.
- **Bug fix & optional L2 switches**: fixes `r % b_mod` indexing in `c12_markov_top2_part` (`shim/c12.cu`) and adds
  optional `DS41_MTP_DENSE_FP4` (`ffn`/`safe`/`all`, default `0`) for code/structured-heavy batch workloads.

### Performance vs. v0.3.15 (pure cold prompts, `DS41_GLOBAL_LOOKUP=0`)

Measured on three DGX Sparks (TP3, default GPU clocks, `DS41_BATCH_INVARIANT=1`, greedy, thinking off,
`DS41_GLOBAL_LOOKUP=0`; `bench/ds41_decode_bench.py --types prose,structured,code --conc 1,2,4`, 256 tokens/stream,
median of 3):

| Streams | Prose (v0.3.15 → **v0.3.16**) | Structured (v0.3.15 → **v0.3.16**) | Code (v0.3.15 → **v0.3.16**) |
| --- | --- | --- | --- |
| 1 stream | 69.7 → **70.9** (+1.7%) | 127.6 → **128.5** (+0.7%) | 112.1 → **111.7** (−0.4%) |
| 2 streams | 95.8 → **95.4** (−0.4%) | 188.2 → **189.2** (+0.5%) | 149.7 → **161.2** (+7.7%) |
| 4 streams | 128.3 → **132.3** (+3.1%) | 247.8 → **250.5** (+1.1%) | 184.1 → **194.3** (+5.5%) |

---

spark-engine v0.3.15 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.14: five strictly L0 bit-exact speed optimizations for DeepSeek-V4.1-Flash (all on by default, each
with an environment toggle), batch-invariant verify row capping (`DS41_INV_ROWS`), and a comprehensive codebase audit
and bug-fix pass across the Rust core, CUDA shims, serving front ends, and launch scripts.

### DeepSeek-V4.1-Flash optimizations (L0 bit-exact)

- **Fused Markov draft sampling** (`DS41_MARKOV_FUSED`, default `1`): fuses the C12 `markov_head` GEMV, logits addition,
  and Gumbel top-1 / top-2 argmax into a single weight pass (`shim/gumbel.cuh`, `shim/c12.cu`).
- **Single-graph MTP body** (`DS41_MTP_SINGLE_GRAPH`, default `1`): captures all 3 MTP layers, `MODE=3` split-K window
  attention (`S=3` for 1–3 streams, `S=2` for 4 streams), and `draft_head` into one CUDA graph with double-buffered
  pinned host buffers and event fences.
- **32-row single-read `wo_a`** (`DS41_WOA_BX32`, default `1`): extends `fp8_gemv_bx` to 2×16-row tiles (`17..=32` rows)
  so batched invariant verification reads `wo_a` weights once instead of twice.
- **Prompt-lookup / N-gram draft extension** (`DS41_PROMPT_LOOKUP`, default `1`): extends drafts and overrides
  low-confidence MTP proposals from matching n-grams in the prompt and generated history (`src/ds41/lookup.rs`, capped
  per concurrency level to stay inside pre-warmed CUDA graphs, deterministic across TP ranks).
- **DSpark confidence tree verification** (`DS41_SPEC_TREE`, default `1`, `DS41_TREE_MAX_BATCH=1`): captures top-2 draft
  branches and continuation chains in the same C12 pass and verifies the tree with ancestor-masked sparse attention,
  async pinned indexer uploads, and zero-copy main-chain commit (`src/ds41/tree.rs`).
- **Batch-invariant row cap** (`DS41_INV_ROWS`, default `32`, optional `64`): caps verify rows consistently across ranks
  and optionally splits 33–64-row invariant calls into ≤32-row chunks bitwise identical to single-stream.

### Codebase audit & fixes

- **Rust core** (`ds41`, `qwen`, `glm`, `tp`, `pcache_core`, `vision`, `sampling`, `ablate`, `c12`, `dflash`, `kda`,
  `mhc`, `mla_latent`, `mm_canvas`, `speculative`, `weights`): boundary and overflow checks, prefix-cache robustness,
  tensor-parallel consistency, and sampling / speculative state-machine fixes.
- **CUDA shims** (`c12`, `ds41_attn`, `ds41_mx`, `ds41_ops`, `fp8`, `kda`, `latent`, `mhc`, `qwen_hc`, `rdma_arn`,
  `shim.cpp`, `spx_mm8`): thread/bounds guards, synchronization and ring-pointer safety, and RDMA/shim cleanup.
- **Serving front ends & scripts** (`openai_server.py`, `qwen_server.py`, `mm.py`, `mm_qwen.py`,
  `chat_template_mm.jinja`, `build.rs`, `start-ds41.sh`, `start-qwen.sh`, `stop-qwen.sh`): concurrency/input validation
  fixes, header-hash + PID-scoped object caching in `build.rs`, full-cluster cleanup on `start-ds41.sh` failure,
  stale-log / memguard hardening in `start-qwen.sh`, and graceful `qwen::pcache` flush before exit in `stop-qwen.sh`.

### Performance vs. v0.3.14 (v0.3.13 DeepSeek-V4.1-Flash baseline)

Measured on three DGX Sparks (TP3, default GPU clocks, `DS41_BATCH_INVARIANT=1`, greedy, thinking off; DeepSeek-V4.1-Flash
was unchanged between v0.3.13 and v0.3.14).

Standard decode (`bench/ds41_decode_bench.py --types prose,structured,code --conc 1,2,4`, 256 tokens/stream, aggregate
tok/s; v0.3.14 baseline median of 6 vs. v0.3.15 median of 3):

| Streams | Prose (v0.3.14 → **v0.3.15**) | Structured (v0.3.14 → **v0.3.15**) | Code (v0.3.14 → **v0.3.15**) |
| --- | --- | --- | --- |
| 1 stream | 69.3 → **69.7** (+0.6%) | 127.1 → **127.6** (+0.4%) | 112.5 → **112.1** (−0.4%) |
| 2 streams | 95.9 → **95.8** (−0.1%) | 188.1 → **188.2** (+0.1%) | 147.2 → **149.7** (+1.7%) |
| 4 streams | 128.3 → **128.3** (0.0%) | 242.9 → **247.8** (+2.0%) | 184.1 → **184.1** (0.0%, peak 187.9) |

Context-reuse / editing / RAG workloads (single stream, greedy, thinking off, same cluster, v0.3.14 baseline vs. v0.3.15):

| Workload | Output tokens | v0.3.14 baseline | **v0.3.15** | Speedup |
| --- | --- | --- | --- | --- |
| Python code refactoring | 268 | 93.9 tok/s | **110.1 tok/s** | **+17.3%** |
| Repeated structured JSON | 408 | 105.6 tok/s | **122.5 tok/s** | **+16.0%** |
| Verbatim code / doc RAG copy | 450 | 100.6 tok/s | **127.3 tok/s** | **+26.5%** |

### Verification

- `cargo test`: 57 / 57 passed.
- Kernel equivalence: `DS41_BATCH_INVARIANT=1 DS41_INV_ROWS=64 DS41_KT_ONLY=w64 ds41-kernel-test` — 0 mismatches (`wo_a`
  32-row bitwise equal to 8-row and 1-row).
- Draft graph & TP sync checks: `DS41_DRAFT_GRAPH_CHECK=1` (600+ live rounds across 1 / 2 / 4 streams, greedy and
  `temperature=0.8`) — 0 draft or confidence mismatches; `DS41_CTRL_CHECK=1` — 0 rank disagreements across all 3 nodes.
- End-to-end L0 text equivalence (all optimizations on vs. off): 38 / 38 exact character-for-character matches (13 greedy
  and sampled `T=0.8` stress prompts, 22 `ds41_decode_bench.py` streams, and 3 code-refactor / JSON / RAG-copy prompts).
