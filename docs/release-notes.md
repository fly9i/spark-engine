spark-engine v0.3.6 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.5: DeepSeek-V4.1-Flash speculative decoding rounds (GLM-5.3-Flash and Qwen3.8-Flash-Next outputs
were checked identical). All changes keep outputs bitwise identical: the release goldens, prefill-path dumps and
fixed-seed service outputs match v0.3.5 (docs/ds41.md §5, §7).

Served decode, `bench/ds41_decode_bench.py`, tok/s summed over streams, median of 3 runs (v0.3.4 in parentheses):

| Streams | Prose | Structured | Code |
| --- | --- | --- | --- |
| 1 | 66.4 (62.9) | 120.1 (117.0) | 107.5 (105.1) |
| 2 | 95.0 (88.9) | 167.8 (161.7) | 134.3 (129.3) |
| 4 | 117.7 (112.8) | 208.8 (199.8) | 165.1 (159.0) |

On open-ended texts sampled at temperature 1 (Chinese prose, a story, an essay question) single-stream decode is
+1.6 to +2.2% and 4 streams +1.7 to +3.1% over the previous release (fixed seeds, the same texts in every arm).

- **Verified rows sampled from vocabulary slices** (`DS41_VERIFY_SPLIT`, default on): each rank adds the same
  per-column Gumbel noise to its slice of the logits and takes the maximum; only (value, index) packets are exchanged.
  This is the same token as the full-vocabulary argmax (ties go to the lower index) and replaces a ~2 MB full-logit
  all-reduce per round plus rank 0 sampling every row on the host.
- **Every rank decides a round itself** (`DS41_CTRL_LOCAL`, default on): drafts, confidences and sampled rows are
  bitwise identical on all ranks (the all-reduces sum in rank order), so every rank chooses how many drafts to verify,
  accepts and rolls back on its own. Before, the other ranks waited for two control messages from rank 0 in every
  round; profiling all three nodes aligned on their collectives showed ~1.4 ms per round of such waiting at the round
  boundaries. `DS41_CTRL_CHECK=1` asserts that the ranks agree.
- **Window-ring checkpoint and rollback in one launch each** (`DS41_RING_FUSED`, default on): the 40 layers'
  index_select / index_copy pairs (80-120 small kernels per round, launched one by one from the host) become two
  kernels.
- Measured and not adopted: a cap of 16 verified rows per round (`DS41_SPEC_MAX_ROWS`; slower at 4 streams), several
  draft candidates per position (with a measured ~4.1 ms per verified row, a chain plus sibling candidates is 1-7%
  slower; `DS41_SPEC_RANKS=<file>` logs the data), and gathering decode's Engram rows on the main thread without
  readahead (`DS41_ENGRAM_ASYNC=0` / `DS41_ENGRAM_WILLNEED=0`: faster with warm pages, stalls on cold ones).
