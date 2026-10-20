# Configuration

Three layers of settings:

1. **`spark.env`** – deployment: paths, nodes, network, port (this page).
2. **Engine switches** – environment variables read by the engine. GLM's tuned values live in
   `engine-rs/profiles/glm.env` ([full reference](glm-switches.md)); Qwen's and DeepSeek's defaults are compiled in
   ([Qwen reference](qwen-switches.md), [DeepSeek settings](ds41.md#5-settings)). Any of them can be overridden in `spark.env`.
3. **Front-end options** – command-line flags of `openai_server.py` / `qwen_server.py` / `ds41_server.py`, set by the start scripts.

## spark.env

| Setting | Example / default | Meaning |
| --- | --- | --- |
| `SPARK_HOME` | `/opt/spark-engine` | Release (or repository) directory; must be the same path on every node. |
| `SPARK_PYTHON` | `$SPARK_HOME/.venv/bin/python` | Python with PyTorch 2.13 (cu130) and the front-end packages. `scripts/env.sh` derives the library paths from it. |
| `SPARK_BIN` | `$SPARK_HOME/bin/spark-engine` | Engine binary (release) or `engine-rs/target/release/spark-engine` (source build). |
| `SPARK_PORT` | `8888` | HTTP port of the OpenAI-compatible API (all models use it; one model runs at a time). |
| `GLM_MODEL` | `/models/GLM-5.3-Flash-tr3-4bpw` | GLM-5.3-Flash EXL3 checkpoint directory (both nodes). |
| `GLM_DRAFTER` | `/models/GLM-5.3-Flash-DFlash2` | DFlash2 drafter directory (both nodes). |
| `SPARK_WORKERS` | `"root@10.0.1.1 root@10.0.3.2"` | ssh targets of ranks 1 and 2 (key-based login from rank 0's node); two-node runs use the first. Replaces `GLM_WORKER`, which still works. |
| `SPARK_MASTER_ADDRS` | `"10.0.1.2 10.0.1.2 10.0.3.1"` | Address of rank 0 as seen from rank 0, 1, 2 (each worker over its own RoCE link). Replaces `GLM_MASTER_ADDR` (one address for every rank), which still works. |
| `SPARK_TP3_IFNAME` | — | Three nodes: interface on which all of them reach each other (NCCL bootstrap only; the data goes over RDMA). |
| `GLM_WORLD` | `2` | Nodes for GLM: 2 or 3 (`start.sh glm3` runs 3 regardless). |
| `GLM_MASTER_PORT` | `29931` | TCP port of the rendezvous store. |
| `GLM_PROFILE` | `$SPARK_HOME/engine-rs/profiles/glm.env` | Engine switch profile for GLM (TP2 and TP3; the transport of each world size is in `profiles/tp<world>-rdma.env`). |
| `GLM_MODEL_NAME` | `GLM-5.3-Flash` | Model name reported by the API. |
| `GLM_KV_TOKENS` | `1048576` | KV-cache budget in tokens, also the longest context; lower it to leave more free memory. |
| `GLM_KV_GRANULE` | `16384` | KV allocation granule per sequence (tokens). |
| `GLM_EXTRA_ENV` | — | Space-separated `NAME=value` overrides applied to every rank last (diagnostics, experiments). |
| `NCCL_IB_HCA` | `rocep1s0f0,roceP2p1s0f0` | Two nodes: RDMA devices NCCL uses (the two ConnectX-7 functions on DGX Spark). |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | Interface for NCCL's bootstrap traffic. |
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 GID index. |
| `GLM53_RDMA_AR_DEV`, `GLM53_RDMA_AR_DEV1`, `GLM53_RDMA_AR_GID` | `rocep1s0f0`, `roceP2p1s0f0`, `3` | Devices and GID of the engine's own RDMA all-reduce (defaults match DGX Spark). |
| `DS41_CKPT`, `DS41_SHARDS`, `DS41_ENGRAM_HASH` | — | DeepSeek-V4.1-Flash checkpoint, per-rank shard root and Engram hash file (every node, same paths; [ds41.md](ds41.md)). |
| `DS41_MODEL_NAME` | `DeepSeek-V4.1-Flash-1017` | Model name reported by the API. |
| `DS41_*` serving settings, `DS41_EXTRA_ENV` | see [ds41.md](ds41.md#5-settings) | Concurrency, speculation, prefix cache, vision, memory; extra settings for every rank. |
| `QWEN_MODEL` | `/models/Qwen3.8-Flash-Next-exl3` | Qwen3.8-Flash-Next EXL3 checkpoint directory. |
| `QWEN_ASSETS` | `$SPARK_HOME/assets/qwen38` | Directory with `draft_vocab_65536.json` (MTP draft head vocabulary). |
| `GLM53_ABLIT`, `GLM53_ABLIT_DIR`, `GLM53_ABLIT_LAYERS` | off, —, `15-44` | GLM o_proj transplant ([abliteration.md](abliteration.md)). |
| `SPARK_ABLATE_CLAMP`, `SPARK_ABLATE_CLAMP_MIN` (DeepSeek: `DS41_ABLATE_CLAMP`, `DS41_ABLATE_MIN`) | off | Conditional clamp ablation ([abliteration.md](abliteration.md)). |
| `SPARK_ABLATE`, `SPARK_ABLATE_MODE` | off, `single` | Direction ablation for either model ([abliteration.md](abliteration.md)); `single` / `per-layer` / `subspace:a-b:k`. |
| `QWEN_PCACHE`, `QWEN_PCACHE_DIR` | off, `/tmp/qwen38-prefix-cache` | Qwen persistent prefix cache on NVMe (mirrors `GLM53_PCACHE`): a prompt extending a cached boundary checkpoint restores it from disk instead of re-prefilling. |
| `QWEN_VISION` | `1` | Load the vision tower (about 0.5 GB) and accept images/videos; `0` = text only. |
| `QWEN_MEMGUARD_GIB` | `8` | The Qwen launcher stops the engine if `MemAvailable` drops below this many GiB. |
| `SPARK_COMPACT` | `1` | Compact host memory once after GLM or DeepSeek starts (every node) (avoids page-migration stalls on unified memory). |
| `GLM53_HOST_REGISTER` | `1` | Pinned host staging buffers as anonymous pages + `cudaHostRegister` (kernel compaction never isolates them); `0` = `cudaHostAlloc`. Both models. |
| `SPARK_MIN_FREE_GIB` | `100` | When switching models, `start.sh` waits until this much memory is free. |

## Most useful engine switches

| Variable | Model | Default | Meaning |
| --- | --- | --- | --- |
| `GLM53_SERVE_MAX_SEQS` | GLM | `4` | Concurrent sequences (more requests queue). More sequences raise total throughput and memory use. |
| `GLM53_SERVE_MAX_STORES` | GLM | `4` | Sequence stores (each holds one sequence's KV and states; idle stores keep their prefix for reuse). |
| `GLM53_PREFILL_CHUNK` | GLM | `4096` | Prompt tokens per prefill step (larger = faster prefill, more scratch memory). |
| `GLM53_VISION` | GLM | `1` | Image/video input through the vision tower. |
| `GLM53_PCACHE`, `GLM53_PCACHE_DIR` | GLM | `1`, `$SPARK_HOME/var/prefix-cache` | Persistent prefix cache on NVMe: long prompts seen before are restored instead of prefilled. |
| `GLM53_SERVE_LOG` | GLM | `0` | Per-round log lines (rows, timings) in the rank logs. |
| `QWEN_SERVE_MAX_SEQS` | Qwen | `8` | Concurrent sequences (1–8). |
| `QWEN_SERVE_STORES` | Qwen | `8` | Sequence stores (at least `QWEN_SERVE_MAX_SEQS`); idle stores keep their prefix for reuse. |
| `QWEN_KV_TOKENS` | Qwen | `1048576` | KV pool size in tokens (shared by all stores in 16K-token granules). |
| `QWEN_SERVE_RESERVE` | Qwen | `32768` | Tokens reserved per request up front; the range grows while decoding. |
| `QWEN_SERVE_CKPT`, `QWEN_SERVE_CKPT_MIN` | Qwen | on, `1024` | Prompt checkpoint at the last message boundary (fast multi-turn follow-ups) for prompts of at least this many tokens. |
| `QWEN_MTP` | Qwen | on | Native MTP drafting (`0` = plain decoding). |

See [glm-switches.md](glm-switches.md) and [qwen-switches.md](qwen-switches.md) for every switch with its value and
precision class. Most of them select between a reference path and an optimized kernel and are kept for A/B testing; the
shipped values are the measured best.

## Front-end options

Set by `start-glm.sh` / `start-qwen.sh`; edit the scripts to change them.

| Option | Default | Meaning |
| --- | --- | --- |
| `--port`, `--host` | `8888`, `0.0.0.0` | Listen address. |
| `--model-name` | `GLM-5.3-Flash-EXL3` / `Qwen3.8-Flash-Next` | Model id reported by `/v1/models` (requests may use any name). |
| `--max-model-len` | KV budget | Longest prompt + completion accepted. |
| `--max-tokens` | `8192` | `max_tokens` when a request omits it. |
| `--vision on/off` | on | Accept image/video content (must match the engine). |
| `--max-image-tokens` | GLM 8000 / Qwen 16384 | Token cap per image (larger images are resized). |
| `--max-video-tokens`, `--video-fps`, `--video-max-frames` | GLM 65536 / Qwen 12288, `2`, GLM 2048 / Qwen 768 | Video sampling and token budget. |
| `--allowed-media-domains`, `--media-deny-private`, `--allowed-local-media-path` | any, off, none | Restrictions for media URLs. |
| `--media-timeout`, `--media-max-bytes`, `--video-max-bytes` | 30 s, 64 MB, 2 GB | Media download limits. |

Request parameters: `temperature` (0 = greedy; sampling is exact speculative sampling), `seed`, `max_tokens` /
`max_completion_tokens`, `stop`, `stop_token_ids`, `stream`, `stream_options.include_usage`, `tools` / `tool_choice`,
`chat_template_kwargs` (e.g. `{"enable_thinking": false}`), `reasoning_effort` (Qwen). `top_p`, `top_k` and penalties are
accepted but not applied.
