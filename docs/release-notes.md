spark-engine v0.3.7 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.6: Qwen3.8-Flash-Next only (GLM-5.3-Flash and DeepSeek-V4.1-Flash outputs were checked identical:
release goldens, prefill-path dumps and GLM / Qwen fixed-prompt texts match v0.3.6). Every change keeps Qwen's verified
output unchanged: greedy replies are bitwise identical, sampled replies identical per seed.

Startup (load → serving, cold): ~54 s → **~19 s**.

Served decode at temperature 1 (fixed seeds, the same texts in every arm, tok/s summed over streams), v0.3.6 → v0.3.7:

| Streams | Chinese prose | Story | Essay question |
| --- | --- | --- | --- |
| 1 | 37.8 → 38.6 (+2.1%) | 45.8 → 47.0 (+2.8%) | 50.4 → 53.8 (+6.7%) |
| 4 | 70.5 → 82.7 (+17.4%) | 78.5 → 89.9 (+14.7%) | 98.8 → 105.0 (+6.3%) |

(Interleaved A B B A service runs, each arm a fresh start; mean of the two runs per arm; every reply identical between
the versions per request and seed.)

Greedy decode (`bench/decode_bench.py`, median of 3): prose 65.4 / 88.2 / 107.9 / 123.3 tok/s and structured
152.3 / 196.6 / 234.5 / 227.9 tok/s at 1 / 2 / 3 / 4 streams. Greedy prose at 3 / 4 streams gained about 2% from the
GDN commit folding (105.5 → 107.9 and 120.3 → 122.2 in the same-day A/B); the rest of the changes target sampled
requests and the batched rounds that contain them.

- **Read-ahead loading** (`QWEN_LOAD_AHEAD`, default on): one thread reads each layer's byte ranges of the shards with
  O_DIRECT into huge-page host buffers ahead of use; one kernel copies a layer's ~6,200 tensors to their places on the GPU
  (the GPU reads the host buffer in place). Page-cache reads ran at ~1.4 GB/s, O_DIRECT at ~10–12 GB/s. The read-ahead
  is a shared module (`src/readahead.rs`) now also used by the DeepSeek loader.
- **Coupled drafts for sampled requests** (`QWEN_DRAFT_COUPLED`, default on): a draft takes the argmax of its logits plus
  the same Gumbel noise the verify will add at that position, so it guesses the token that will actually be sampled.
  Accepted drafts per round at T = 1 0.84 → 0.90.
- **GDN commit folded into the next verify** (`QWEN_GDN_FUSED_COMMIT`, default on): the accepted rows' gated-DeltaNet
  state updates are no longer a separate pass over the 36 layers' states; their inputs are kept per sequence and the next
  round's recurrence kernel applies them first (same operations, same order). 4 streams +1.3–4.9%.
- **Verify width by rate in rounds with a sampled sequence** (`QWEN_SPEC_RATE=t`, default): the batch verifies the number
  of drafts that maximizes expected tokens per estimated round time instead of a fixed two-width rule, and keeps drafting
  only while one more step would pay off. All-greedy rounds keep the previous rule (on a greedy prose benchmark the rate
  rule was 1–4% slower even with a separately fitted greedy cost, `QWEN_SPEC_COST_GREEDY`).
- **Online acceptance calibration** (`QWEN_SPEC_ONLINE`, default on): the rate rule's acceptance estimates per
  confidence bucket are learned while serving (per sequence and process-wide, greedy and sampled apart), with the
  built-in table as prior. 2 / 4 streams at T = 1: −0.6 to +4.4%.
- **Narrow launches for small EXL3 GEMVs** (`QWEN_EXL3_NARROW`, default on): decode GEMVs with fewer than 128 blocks
  (N = 512 / 640 or K = 640) use 2-warp blocks of one 16-column tile; same per-output MMA order. 5–20% faster per kernel;
  end to end within noise. The large dense EXL3 shapes already read at 207–227 GB/s (GB10 cold-read limit ~232 GB/s).
- Measured and not adopted: the rate rule in greedy rounds (above); the MoE expert GEMVs on GLM's persistent schedule
  (no gain); PDL on every Qwen kernel (−0.6 to −1.2%); verifying a second-choice sibling at the first rejection; a 12-bit
  coding of the F16 weights (4% on the down projections only). New switches: [docs/qwen-switches.md](qwen-switches.md).
