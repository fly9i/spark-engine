spark-engine v0.3.5 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.4: DeepSeek-V4.1-Flash speculative decoding of sampled requests (GLM-5.3-Flash and
Qwen3.8-Flash-Next outputs were checked identical). Outputs stay bitwise identical for a given seed; greedy decoding
is unchanged (docs/ds41.md §5, §7).

- **Sharper drafts for sampled requests** (`DS41_DRAFT_TSCALE`, default 0.7): at temperature T > 0 the DSpark drafts
  now sample at 0.7 × T. Drafts and target share the per-position sampling keys, so every emitted token is still the
  target's own keyed sample and only the proposed drafts change. A sharper draft matches the target's sample more
  often: on open-ended texts at temperature 1 (Chinese prose, a story, an essay question, two prompts with thinking)
  the expected acceptance per draft goes 0.609 → 0.619, close to the 0.622 bound of standard speculative sampling
  (`DS41_SPEC_RS=1`) but without its overhead and while keeping outputs reproducible. Measured on the service with
  fixed seeds (the same texts byte for byte in every arm): tokens per verify round +2.0%, decode speed 44.3 → 45.5
  tok/s (+2.8%). `1` restores drafts at the target's temperature.
- `DS41_RS_STATS_X=1` adds a draft-temperature sweep and the list-coupling bound (target sample within the draft's top 2
  / top 3) to the per-draft acceptance statistics.
- `bench/ds41_decode_bench.py`: `--seed` (request i gets seed + i) and `--dump` (saves the texts), for comparing
  sampled runs on identical outputs.
