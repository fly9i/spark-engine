spark-engine v0.2.1 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.2.0 (DeepSeek-V4.1-Flash; GLM-5.3-Flash and Qwen3.8-Flash-Next code paths are unchanged):

- **The DeepSeek front end follows the official API** (deepseek-ai/deepseek-recipe). Control tokens now come only
  from the prompt template: special-token text inside a request (`</think>`, `<｜DSML｜…>`, `<｜User｜>` in content,
  reasoning, tool arguments, results or schemas) is encoded as ordinary text, and the output parser recognizes
  `</think>` / `｜DSML｜` by token id. Before, an agent reading a file with such text could open turns or tool blocks,
  and a model quoting DSML markup produced a real tool call. Prompts without such text encode to the same tokens as
  before. Request handling as the official API: thinking mode samples at temperature 1.0
  (`DS41_THINKING_TEMPERATURE`), `max_tokens` defaults to 8K / 64K thinking / 128K at `max` effort, effort levels map
  `minimal`/`low` 50, `medium`/`high`/`xhigh` 75, `max`/`ultra` 100 (was `medium` 62, `xhigh` 100), top-level
  `thinking: {type}`, tool definitions normalized to `{name, description, parameters}`, `tool_choice` `none` /
  `required` / a named function, `response_format` `json_object` / `json_schema`, `"prefix": true` continuation.
  docs/ds41.md §6.
- **DSpark draft sampling runs as one CUDA graph** (`DS41_DRAFT_GRAPH`, default on): Markov bias, Gumbel top-1,
  cross-rank choice and confidence for all five drafts, one read-back instead of two host round trips per draft.
  Bitwise the same drafts (`DS41_DRAFT_GRAPH_CHECK=1` compares every round). Draft time 4.31 → 3.96 ms per round;
  single-stream decode +0.2 to +4.6% across prose / structured / code, within run-to-run noise; 2–4 streams unchanged.
- **Engram prefill without FP32 / BF16 intermediates** (`DS41_ENGRAM_Q`, default on): a requantize kernel builds the
  projection's FP8 operands straight from the table bytes, bitwise the same (~450 MB → 52 MB of memory traffic per
  4K-token chunk and Engram layer; about −5 ms per 8K prefill).
- **Smaller prefix-cache checkpoints:** checkpoints inside a long prefill leave out the 20 decoder window rings
  (2.5 MiB each, 12–21% of the file), which a restore never reads (checked by filling them with NaN). Checkpoints
  written by an older binary are not reused (the cache is keyed by the binary).

What it runs:

- GLM-5.3-Flash (EXL3 4 bpw) on two (or three) DGX Sparks with DFlash2 speculative decoding.
- DeepSeek-V4.1-Flash-1017 on three DGX Sparks with DSpark speculative decoding.
- Qwen3.8-Flash-Next (EXL3 4.05 bpw) on one DGX Spark with MTP speculative decoding, image/video input and 1M context.
- OpenAI-compatible API, Prometheus metrics.

Archive: `bin/spark-engine`, `lib/libexllamav3_ext.so` (exllamav3 v1.4.9, MIT), serve scripts, profiles, docs,
`reference/ds41` (Engram hash export). Install PyTorch 2.13 (cu130) separately; see docs/deploy.md. Models are not
included.
