# Qwen3.8-Flash-Next: environment switches

The Qwen path has no profile file: every default is compiled in and is the tuned setting. To change one, set the
variable in `spark.env` or in the environment of `engine-rs/serve/start-qwen.sh`. Unless noted, the engine reads a
variable once at load or once per call; "per call" switches that affect decode take effect when the CUDA graphs are
captured.

Precision classes:

- **L0**: bitwise identical output.
- **L1**: rounding-level difference, not worse than the FP32 reference.
- **L2**: draft side only; the verified output is unchanged.
- **L3**: lossy.
- **—**: not a numerics switch (resources, paths, logging). **n/r**: not recorded.

Boolean conventions: "on (`=0` off)" means any value other than `0` keeps it on; "off (`=1`)" means only `1` turns it
on; "set" means any value (presence) counts.

## Serving settings

| Variable | Default | Meaning | Class |
| --- | --- | --- | --- |
| `QWEN_MODEL` | none (set in `spark.env`) | Model directory (EXL3 checkpoint), passed by `start-qwen.sh` to the engine (`qwen-serve`) and to the front end (`--model-dir`). | — |
| `QWEN_ASSETS` | the model directory (`start-qwen.sh` sets `$SPARK_HOME/assets/qwen38`) | Directory with `draft_vocab_<N>.json`. The release ships `assets/qwen38/draft_vocab_65536.json`. | — |
| `QWEN_VISION` | engine: off (`=1`); `start-qwen.sh`: `1` | `1` loads the vision tower (about 0.5 GB) and accepts image / video requests; also starts the front end with `--vision on`. `0` = text only; media requests then get an error. | — |
| `QWEN_MEMGUARD_GIB` | `8` | `start-qwen.sh` watchdog: polls `MemAvailable` every 0.5 s and kills the engine if it falls below this many GiB (unified-memory exhaustion can hang the GB10). | — |
| `QWEN_SERVE_MAX_SEQS` | `8` (clamped to 1..8) | Sequences decoded concurrently (one batched speculative round per turn). | L0 |
| `QWEN_SERVE_STORES` | `8` (at least `QWEN_SERVE_MAX_SEQS`) | Sequence stores: each keeps its committed history for exact-prefix reuse (the recurrent GDN state cannot rewind, so only exact prefixes are reused). | L0 |
| `QWEN_KV_TOKENS` | `1048576` | KV pool size in tokens, rounded to 16384-token granules and shared by all stores (equal shares at start-up; a larger request takes a larger range, idle stores give theirs up, LRU). The longest request (prompt + new tokens) is the pool minus 32. | — |
| `QWEN_SERVE_CAP` | not read | Mentioned only in a source comment; the per-request capacity derives from `QWEN_KV_TOKENS`. | — |
| `QWEN_SERVE_RESERVE` | `32768` (min 256) | Positions reserved for output at admission: `min(max_tokens, this)`. A longer output grows the store's range while decoding (in place, or moved to a free gap), so a `max_tokens` near the context limit no longer takes the whole pool. | L0 |
| `QWEN_SERVE_CKPT` | on (`=0` off) | Prompt checkpoint at the last `<\|im_start\|>` of the prompt (GDN state, conv window, indexer key ring, PLE window, pending MTP rows; about 116 MB per store). The next turn of a conversation restores it and prefills only the suffix, even when the client's echo of the previous reply does not match the generated tokens. | L1 |
| `QWEN_SERVE_CKPT_MIN` | `1024` | Minimum prompt length (tokens before the checkpoint position) for taking a checkpoint. | L1 |
| `QWEN_SERVE_GRAPH` | on (`=0` off) | CUDA graphs for the batched verify / commit / MTP draft chain (needs the MTP head). `0`: eager, one sequence at a time. | L0 |
| `QWEN_SERVE_GRAPH_MB` | `3000` | Budget for the graphs of multi-store sets (about 5 MB per captured verify row); least recently used sets are dropped above it, single-store graphs are kept. | — |
| `QWEN_PREFILL_MACRO` | `16384` | Tokens per layer-major prefill pass (all of the pass's activations stay resident); also the prompt tokens a sequence prefills per serving turn between decode rounds. | L1 (moves chunk boundaries) |
| `QWEN_YARN` | auto | RoPE mode per sequence: YaRN (factor 4) when the sequence's capacity exceeds the native 262144, plain RoPE otherwise. `1` / `0` force always / never. Static YaRN costs some quality on short texts. | changes output when forced |
| `QWEN_MTP` | on (`=0` off) | Load the MTP head (speculative drafts). `0`: no drafts, no serving graphs. | L2 |
| `QWEN_SPEC_CUMCONF` | `0.7` | Draft length policy θ: the chain keeps drafting while some sequence's product of draft confidences stays ≥ θ; each sequence verifies its drafts up to the last one inside the bound. `0`: fixed `QWEN_SPEC_K` drafts. | L2 |
| `QWEN_SPEC_KMAX` | `10` (`QWEN_SPEC_K` when θ = 0) | Most drafts per chain (in a batch also limited to 64 verify rows in total). | L2 |
| `QWEN_SPEC_K` | `3` | Fixed drafts per round when θ = 0; minimum depth of the batch `all` rule; most drafts of a sequence's first chain. | L2 |
| `QWEN_SPEC_CUMCONF_BATCH` | unset (single-sequence θ) | θ for batches of 2+ sequences: a number in (0, 1) = its own θ, `0` = fixed k. | L2 |
| `QWEN_SPEC_BATCH_RULE` | `all` | Batch depth rule. `all`: at least k, deeper only while every sequence is inside the bound. `max`: as deep as the most confident sequence (measured 6–25% slower on prose). | L2 |
| `QWEN_SPEC_CALIB` | unset (identity) | Piecewise-linear map from draft-head confidence to acceptance rate, `"c:p,c:p,..."` (ascending c); `fit` = the built-in fitted table. No consistent gain measured. | L2 |
| `QWEN_DRAFT_VOCAB` | `65536` | Draft head sub-vocabulary: a number N loads `$QWEN_ASSETS/draft_vocab_N.json`, anything else is a file path; `0` or a missing file disables the draft head (drafts then use the full `lm_head`). | L2 |
| `QWEN_DRAFT_Q4` | on (`=0` Q8) | Draft head in affine 4-bit (`0`: int8 + FP32 scale per 128). Q4 measured +3–7% tok/s at one sequence, same acceptance. | L2 |
| `QWEN_LOOKUP` | on (`=0` off) | Prompt-lookup drafts: the continuation of the latest earlier occurrence of the last n tokens (prompt and output), used when it starts with the MTP head's first draft; it then replaces the rest of the MTP chain for that sequence. | L2 |
| `QWEN_LOOKUP_N` | `3` (clamped 1..8) | n-gram length for prompt lookup. | L2 |
| `QWEN_LOOKUP_MIN` | `0.6` | Minimum running per-token acceptance of a sequence's lookup proposals for them to be used (lookup rounds must also yield at least 1/1.1 of the tokens of MTP rounds; every 16th round tries the other source). | L2 |
| `QWEN_MTP_PREFILL_TAIL` | `0` (off) | MTP catch-up of a prompt covers only its last n rows (earlier MTP cache rows zeroed). Faster long prefill, but lower draft acceptance afterwards. | L2 |
| `QWEN_REQLOG` | `/tmp/qwen38-requests.jsonl` (`""` = off) | Front end: one JSON line per request (sizes, prefix hit, queue / prefill / TTFT, where the prompt first differs from the closest earlier sequence, with the text around that point). | — |
| `QWEN_NGRAM_HOT` | off | File of hot n-gram table rows (from `ngram_hot.py`) held in RAM in front of the mmapped n-gram table. Helps only when the page cache is squeezed. | L0 |
| `QWEN_NGRAM_CACHE_MB` | `0` (off) | Run-time 2-way set-associative RAM cache (MB) of n-gram table rows. Same use case as `QWEN_NGRAM_HOT`. | L0 |
| `QWEN_KEEP_SHARD_CACHE` | off (`=1`) | `1` keeps the loaded model shards in the page cache. By default they are released after load (the n-gram table stays): more free memory and a faster first prefill, at the cost of a slower reload. | — |

## Kernel and schedule switches

Kept for A/B testing; the defaults are the measured best. Classes are relative to the default.

| Variable | Default | Meaning | Class |
| --- | --- | --- | --- |
| `QWEN_PREFILL_CHUNK` | `2048` | Attention sub-chunk rows within a layer-major prefill pass (1% faster than 1024). | L1 |
| `QWEN_PREFILL_MOE_ROWS` | `16384` | Rows per MoE call within a layer-major prefill pass. | n/r |
| `QWEN_PREFILL_CM` | off (`=1`) | Former chunk-major prefill (a whole forward per chunk) instead of layer-major. | L0 |
| `QWEN_EXL3_FOLD` | off (`=1`) | Many-row EXL3 linears use the folded fp16 effective weight with a cuBLASLt GEMM instead of transforming the activations. Retested not worse, but slower in the fused prefill path. | L1 |
| `QWEN_EXL3_SLICES_TUNED` | on (`=0` off) | Tuned EXL3 GEMV K slices (10 for K = 6144, 6 for N ≥ 6144); `0`: the general rule. 1–2% faster per round. | L1 |
| `QWEN_EXL3_FUSED` | off (`=1`) | Up to 8 decode rows in one launch (input transform + GEMV + finish). About 5% slower per round. | L0 |
| `QWEN_EXL3_FIN_F32` | off (`=1`) | Many-row EXL3 finish converts the GEMM output to fp32 first (former path) instead of reading the fp16 output. | L0 |
| `QWEN_EXL3_MULTI` | on (`=0` off) | Decode linears sharing one input: one input transform (with the HC mix fused) and one GEMV launch; `0`: one linear at a time. | L0 |
| `QWEN_MIX_LAZY` | on (`=0` off) | Decode: the HC mix is computed by the first consumer's input transform; `0`: mixed first. | L0 |
| `QWEN_PREFILL_HAD_MULTI` | on (`=0` off) | Prefill: one input-transform launch for all linears on the same input; `0`: one each. | L0 |
| `QWEN_PREFILL_FIN_SPLIT` | off (`=1`) | Prefill: projections' finishes as separate launches instead of fused into their consumers (conv, output norm, QSA prep). | L0 |
| `QWEN_PREFILL_PREP_SPLIT` | off (`=1`) | Prefill: GDN recurrence prep (q/k normalization, decay/beta) as a separate kernel instead of inside the conv. | L0 |
| `QWEN_PREFILL_SHARED_TORCH` | off (`=1`) | Prefill: shared expert elementwise ops in torch instead of the decode kernels. | L0 |
| `QWEN_PLE_PACK_SYNC` | off (`=1`) | Prefill: pack the PLE n-gram rows before queuing the layers instead of on another thread overlapped with the first layers. | L0 |
| `QWEN_PLE_SERIAL` | off (set) | PLE n-gram table rows read one page fault at a time (former path) instead of `MADV_WILLNEED` + parallel copy. | L0 |
| `QWEN_PREFILL_APPLY_SPLIT` | off (`=1`) | Prefill: apply each layer's MoE update right away instead of fusing it into the next layer's attention HC norm. | L0 |
| `QWEN_DEC_APPLY_SPLIT` | off (`=1`) | Decode: same as above for decode / verify rows. | L0 |
| `QWEN_HC_APPLY_SPLIT` | off (`=1`) | HC residual update (apply) and the following norm as two kernels instead of one. | L0 |
| `QWEN_HC_UPMIX_SPLIT` | off (`=1`) | Prefill HC: cuBLAS up GEMM (fp16 g) + separate mix instead of the fused tensor-core up+mix (g kept fp32). | L1 |
| `QWEN_PREFILL_HC_TORCH` | off (`=1`) | Prefill HC via the former torch-op path instead of the v2 kernels (the v2 path is 11% faster and closer to the reference). | L1 |
| `QWEN_HC_V1` | off (set) | Decode HC via the v1 single-kernel `hc_mix` instead of the v2 kernels (norm, down, mid, up, mix). | n/r |
| `QWEN_Q8_DENSE` | off (`=1`) | Q8 copies of the F16 dense weights (HC, router, PLE, GDN a/b) for decode / verify rows. One-sequence verify −5.6%, but the model is very sensitive to HC weight error. | L3 |
| `QWEN_Q8_SCOPE` | all | With `QWEN_Q8_DENSE=1`: comma list of weight groups to quantize (`hc_down`, `hc_up`, `router`, `gdn_ab`, `ple`). | L3 |
| `QWEN_Q8_OFF` | off (`=1`) | With `QWEN_Q8_DENSE=1`: use F16 again (same-process A/B). | — |
| `QWEN_F16_TC` | on (`=0` off) | F16 GEMV (HC, router, GDN a/b, PLE) on tensor cores with fp16 hi/lo split of x, batch-size independent; `0`: scalar v2 kernel. | L1 |
| `QWEN_F16_V1` | off (`=1`) | Where the scalar F16 GEMV is used: v1 kernel instead of v2 (bitwise equal to each other). | L0 vs v2 |
| `QWEN_F16_NJ` | auto (4, 2 or 1) | v2 F16 GEMV: weight rows per warp; default the largest that still gives ≥ 192 blocks. | L0 |
| `QWEN_F16_FUSED` | off (`=1`) | F16 GEMV with the slice sum in its last blocks (scalar v1 kernel, no tensor cores). Not faster. | L1 vs default |
| `QWEN_GDN_V1` | off (`=1`) | GDN prefill recurrence: former per-head kernel instead of the row-parallel kernels. | L0 |
| `QWEN_GDN_RECUR_V1` | off (`=1`) | GDN decode recurrence: former kernel instead of the shared-memory staged one (which falls back automatically for many sequences × long chains). | L0 |
| `QWEN_GDN_ROWS1` | off (`=1`) | GDN prefill recurrence: one lane per state row instead of four. | L1 |
| `QWEN_QSA_ATTN_VER` | tensor core, fp32-level (`attn_tcp`) | Decode QSA attention kernel: `t` = fp16-operand tensor-core kernel (less accurate), `3` / `2` = pipelined scalar kernels bitwise equal to the original scalar kernel. | L1 |
| `QWEN_QSA_ATTN_V1` | off (`=1`) | Decode QSA attention: the original scalar kernel. | L1 vs default |
| `QWEN_QSA_SCORE_V1` | off (`=1`) | Indexer scores: scalar kernel instead of tensor cores. | L1 |
| `QWEN_QSA_SELECT_V1` | off (`=1`) | Indexer top-k selection: block-scan kernel instead of the warp-ballot kernel. | L0 |
| `QWEN_PREFILL_ATTN_SPLIT` | off (`=1`) | Prefill QSA attention: split tensor-core kernel + combine instead of the flash-style kernel. | L1 |
| `QWEN_PREFILL_ATTN_SCALAR` | off (`=1`) | Prefill QSA attention: the decode-row kernels (as chosen by `QWEN_QSA_ATTN_VER` / `QWEN_QSA_ATTN_V1`) + combine instead of the flash-style kernel. | L1 |
| `QWEN_MOE_UNFUSED` | off (`=1`) | MoE decode: separate grouping / finish / activation / transform kernels instead of the fused ones. | L0 |
| `QWEN_SHARED_UNFUSED` | off (`=1`) | Decode: shared expert elementwise ops as torch ops and a copied router output, and no multi-linear launch for the shared expert. | L0 |
| `QWEN_MOE_YD16` | off (`=1`) | Routed experts' outputs stored in fp16 (halves the down GEMM writes). Prefill −3.2%. | L3 |
| `QWEN_MOE_GEMV` | unset | Override string for the persistent MoE decode GEMV schedule (`tpw,pf,warps,s1,s2,ctas,apf,fused`). Parsed, but the Qwen MoE does not use the persistent path, so it has no effect here. | — |
| `QWEN_MOE_ROUTE_SIDE` | off (`=1`) | Decode: router GEMV + top-k on a side stream next to the shared-expert EXL3 GEMVs. No measurable gain. | L0 |
| `QWEN_GDN_AB_SIDE` | off (`=1`) | Decode: GDN a/b projection on a side stream next to the qkv / z EXL3 GEMVs. No measurable gain. | L0 |
| `QWEN_COMMIT_SIDE` | off | Experimental: `1` runs the state commit on a side stream next to the MTP chain (joined by the next verify); `2` joins right away (diagnostic). `1` changed verified output in some runs (unresolved race); do not use. | not deterministic |
| `QWEN_DRAFT_Q4_TILED` | on (`=0` off) | Q4 draft head in the tiled layout (contiguous 512 B warp loads); `0`: row layout. | L0 |
| `QWEN_DRAFT_ALTS` | unset | Extra draft heads built at load for same-process A/B, e.g. `q4:65536;q8:49152` (`q4` tiled, `q4r` row layout, `q8`; the number as in `QWEN_DRAFT_VOCAB`). | L2 |
| `QWEN_DRAFT_PICK` | unset | `i` selects `QWEN_DRAFT_ALTS[i-1]` as the draft head (read at graph capture). | L2 |
| `QWEN_VIS_EXACT` | on (`=0` off) | Vision tower EXL3 linears take the GEMM output in fp32 (cuBLASLt) and finish in fp32; `0`: fp16 GEMM output (much larger error). +3–10% encode time. | L1 |
| `QWEN_VIS_ATT32` | off (`=1`) | Vision attention in fp32 instead of fp16 flash (slightly more accurate, +55% time). | L1 |

## Diagnostics and developer tools

Used by the `qwen-gen`, `qwen-spec` and `qwen-batch` probes and for timing; not needed for serving.

| Variable | Meaning |
| --- | --- |
| `QWEN_TIMING` | Set: `qwen-spec` / single-sequence generation synchronizes around draft, verify, commit and catch-up and accumulates ms per phase. |
| `QWEN_SERVE_TIMING` | Set: the server prints synchronized phase times of batched rounds per batch size every 100 rounds (slows serving). |
| `QWEN_MOE_DUMP` | `n`: write the input rows of the n-th prefill MoE call to `/tmp/moe_x_<n>.f32` (for kernel benchmarks). |
| `QWEN_STATE_DUMP` | `dir`: with `QWEN_TAIL_ARMS`, write the sequence state after the prefill, one fp32 file per layer and tensor. |
| `QWEN_DUMP_LAST` | `qwen-gen`: keep only the last n rows when dumping prompt logits (default all). |
| `QWEN_CAP` | `qwen-gen`: sequence capacity (default 8192, raised to fit prompt + output). |
| `QWEN_PREFILL_STEP` | Set: `qwen-gen` prefills in chains of 16 rows (decode / verify kernels) instead of the prefill path. |
| `QWEN_TAIL` | `qwen-gen`: prefill all but the last n tokens, then run those n with all their logits (for comparison against a reference). With `QWEN_TAIL_ARMS` the default is 512. |
| `QWEN_TAIL_STEP` | `qwen-gen`: run the tail in chains of n rows (the decode / verify kernels, like speculative rounds of n−1 drafts); default the whole tail. |
| `QWEN_TAIL_ARMS` | `qwen-gen`: `"A;B;..."`, each a comma list of `VAR=value`; in one process, for every arm and every prompt of a comma list of id files, run the tail and dump logits to `<dump>_<arm>_<prompt>.f32`. |
| `QWEN_PREFILL_REPS` | `qwen-gen`: n−1 timed warm-up prefills before the measured one. |
| `QWEN_PREFILL_AB` | `qwen-gen`: the warm-up reps alternate `VAR=1` / unset (comma list allowed; `VAR=value` items are set to value) for a same-process A/B. |
| `QWEN_SKIP_PLAIN` | Set: `qwen-spec` skips the plain greedy reference run. |
| `QWEN_GRAPH` | `qwen-spec`: `0` runs speculative decoding eagerly instead of with CUDA graphs. |
| `QWEN_BATCH_ARMS` | `qwen-batch`: `"A;B;..."`, each a comma list of `VAR=value`; one run per arm in the same process (fresh sequences, graphs recaptured). |
| `QWEN_BATCH_LOG` | `qwen-batch`: file with one JSON line per sequence and round (`conf`, `acc`, verified drafts `w`). |
| `QWEN_BATCH_NOLAP` | Set: `qwen-batch` inserts no phase synchronizations (for profiling with nsys). |
| `QWEN_HOST_ACCEPT` | Set: `qwen-batch` uses the host-path accept (verify, argmax copy, host-written commit / catch-up inputs) instead of the device accept. |
