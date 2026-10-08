# spark-engine

[中文](README.zh-CN.md)

A from-scratch inference engine for **NVIDIA DGX Spark (GB10, aarch64, CUDA 13 / sm_121)** that serves two large hybrid-MoE
models through one OpenAI-compatible API, at EXL3 4-bit, with speculative decoding and 1M-token context:

- **GLM-5.3-Flash** across **two DGX Sparks** (tensor parallel over the ConnectX-7 RoCE link), with the DFlash2 drafter;
- **Qwen3.8-Flash-Next** on **one DGX Spark**, with native MTP drafting, image/video input and the full hybrid stack.

Everything that matters for performance is the project's own code — the EXL3 GEMM/MoE kernels, the GDN/KDA linear-attention
and sparse-attention kernels, the speculative-decoding state machine, the RDMA all-reduce, FP8 KV, the NVMe prefix cache
and the serving front end. It links only PyTorch's libtorch (tensor plumbing, cuBLAS, NCCL) and the EXL3 dequant kernels.
One binary runs either model; `start.sh glm|qwen` picks one.

## At a glance

Measured on 2026-10-08 with this release's binary, default settings, greedy, thinking off (method and more numbers:
[docs/benchmarks.md](docs/benchmarks.md)).

| | **GLM-5.3-Flash** | **Qwen3.8-Flash-Next** |
| --- | --- | --- |
| Hardware | 2× DGX Spark (TP2, RoCE) | 1× DGX Spark |
| Architecture | 45 layers: KDA linear-attn + MLA/DSA sparse-attn, 288-expert MoE | 48 layers: gated-DeltaNet + QSA sparse-attn, 512-expert top-10 MoE, per-layer n-gram |
| **Decode — prose** (1/2/3/4 streams) | 52.9 / 66.5 / 73.3 / 82.9 tok/s | 64.1 / 85.5 / 105.0 / 119.5 tok/s |
| **Decode — structured** (1/2/3/4 streams) | 92.5 / 119.1 / 158.0 / 168.0 tok/s | 149.8 / 192.9 / 232.6 / 226.5 tok/s |
| **Prefill** | ~1,560 tok/s (10K–55K prompts) | ~1,900 tok/s (13K–77K prompts) |
| **TTFT** (short prompt) | ~370 ms | ~150 ms |
| **Startup** (load → serving) | ~33 s | ~54 s |
| Speculative decoding | DFlash2 drafter, tree/chain verify | native MTP + prompt-lookup |

"Prose" = a long explanation (lower draft acceptance); "structured" = counting 1–200 (high acceptance). Aggregate is the
sum over streams; decode speed depends on the text because of speculation.

### Quantization & runtime precision

| Component | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| Routed experts (weights) | EXL3 **4 bpw**, mcg codebook | EXL3 **4.05 bpw** (head 6, MTP 4), mul1 codebook |
| Dense weights — prefill | source precision (FP16), cuBLAS | source precision |
| Dense weights — decode | **Q8** (int8 + FP32 scale / 128, ~0.67% RMS; lossless 12-bit C12 optional) | FP16 (int8 copies optional) |
| Attention compute | MLA/DSA in FP32/TF32; DSA indexer sparse | QSA FP32 accumulate; learned sparse selection |
| **KV cache** | **FP8** (e4m3) | **FP8** (e4m3) |
| Output head | FP16 | 6 bpw |

Precision classes used throughout: **L0** bitwise-identical, **L1** rounding-level (not worse than FP32 ref), **L2**
drafter-only, **L3** lossy by design. The lossy choices (L3) are the FP8 KV cache and GLM's Q8 dense-decode weights;
both are measured equivalent to the reference. Routed experts keep the checkpoint's EXL3 quantization.

### Context length & concurrency

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| Max context (KV budget) | **1,048,576 tokens** | **1,048,576 tokens** |
| KV-cache memory at that budget | ~7.05 GiB / node (11 MLA layers, 7,216 B/token) | ~14.0 GiB (12 sparse-attn + MTP, 14,364 B/token) |
| Concurrent sequences (default) | 4 | 8 |
| Batched speculative verify | yes (sequences share one forward) | yes (up to 8) |
| Prefix reuse | in-memory stores + NVMe persistent cache (`GLM53_PCACHE`) | multi-turn prompt checkpoints + NVMe cache (`QWEN_PCACHE`) |
| Long context > 262K | — | YaRN scaling |

Both models are hybrid: most layers are linear/gated attention whose state is fixed-size (does not grow with context),
so a 1M-token window costs only 7–14 GiB of KV — the rest of GB10's 128 GB unified memory holds the weights.

## What this project is

- **Written for GB10 specifically.** Kernels target sm_121's shared-memory and occupancy limits; decode weights are
  tiled so each warp load is one contiguous 512 B (near the ~230–250 GB/s streaming ceiling); CUDA graphs capture every
  decode shape; memory is sized against `MemAvailable` on the shared CPU/GPU address space.
- **Two-node tensor parallelism over RoCE** with the engine's own small-message RDMA all-reduce (fixed summation order,
  bitwise identical to NCCL, 12–50 µs), fused into the hyper-connection update that consumes it.
- **Deterministic, graph-captured speculative decoding** with exact greedy semantics: KDA state-correction replay, MLA
  shared base states, DSA per-row selection, confidence-truncated draft trees, and prompt-lookup ("copy") drafts.
- **Self-contained MoE**: one templated EXL3 MoE kernel shared by both models (fused persistent decode; grouped
  tensor-core GEMM prefill); no external MoE library.
- **Full multimodal and long-context serving**: image/video through the model's own vision tower, 1M-token KV pool in
  16K granules with LRU, grow-in-place KV ranges, and a persistent NVMe prefix cache.
- **One OpenAI-compatible binary** for both models, with Prometheus metrics and a memory guard for GB10's unified memory.

## Features

**Both models**
- OpenAI-compatible HTTP API: `/v1/chat/completions` (streaming, usage, stop strings, tool calls, reasoning content
  split from the answer), `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, and Prometheus
  `/metrics` with vLLM-compatible metric names.
- Concurrent sequences with batched speculative verification (several sequences share one forward), CUDA graphs for
  every decode shape, KV cache in FP8 with a pool sized from free memory.
- Image and video input (OpenAI `image_url`, `video_url`, data URLs, base64, HTTP(S)), encoded by the model's own
  vision tower inside the engine.
- Long context: 1M-token KV budget; prefix reuse across requests.
- Built for GB10's unified memory: weights, KV and scratch are sized against `MemAvailable`, and the launch scripts
  include a memory guard.

**GLM-5.3-Flash (two nodes)**
- Tensor parallelism over two DGX Sparks: NCCL over RoCE plus the engine's own RDMA all-reduce for small messages
  (fixed summation order, bitwise identical to NCCL, 12–50 µs per call).
- EXL3 4 bpw routed experts (288 per rank) through the engine's MoE kernels: one fused persistent kernel for decode
  (gate/up, SwiGLU, down) and grouped tensor-core GEMMs for prefill, with programmatic dependent launches.
- Dense decode weights as Q8 (int8 + FP32 scale per 128, tiled for 512-byte loads); a lossless 12-bit coding (C12) is
  available; prefill runs dense weights at source precision.
- DFlash2 drafter with tree/chain verification, KDA (linear attention) state correction replay, MLA shared base states,
  DSA sparse-attention selection and confidence-based truncation; copy (prompt-lookup) drafts.
- Persistent prefix cache on NVMe (`GLM53_PCACHE`).
- Optional abliteration, off by default ([docs/abliteration.md](docs/abliteration.md)): o_proj transplant (`GLM53_ABLIT`) and direction ablation (`SPARK_ABLATE`).

**Qwen3.8-Flash-Next (one node)**
- The full hybrid architecture: gated DeltaNet linear attention, sparse attention with an indexer, a 512-expert top-10
  MoE, 4-stream hyper-connections and per-layer n-gram embeddings (memory-mapped from NVMe).
- All EXL3 products (mul1 codebook) on the engine's own kernels; no exllamav3 runtime.
- Native MTP drafts with a 4-bit sub-vocabulary draft head and prompt-lookup drafts, verified in batches of up to 8
  sequences.
- 1M context (YaRN above 262,144 tokens) with a granular KV pool; multi-turn prompt checkpoints at the last message
  boundary (follow-up TTFT about 0.3 s on a 16K conversation); `max_tokens` reserved on demand instead of up front.
- Reasoning effort levels (`reasoning_effort`: none / low / medium / high).

## Quick start (release binary)

Requirements: DGX Spark (GB10) with DGX OS / Ubuntu 24.04, NVIDIA driver with CUDA 13 support, Python 3.12.
GLM-5.3-Flash needs two DGX Sparks connected by their ConnectX-7 ports. Full instructions: [docs/deploy.md](docs/deploy.md).

```bash
# 1. the release (same path on both nodes for GLM)
sudo mkdir -p /opt/spark-engine && cd /opt/spark-engine
curl -L https://github.com/fly9i/spark-engine/releases/latest/download/spark-engine-linux-aarch64-cu130-sm121.tar.gz | sudo tar xz --strip-components=1

# 2. Python environment: PyTorch 2.13 (CUDA 13.0) for the engine, plus the front end's packages
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install jinja2 tokenizers numpy pillow av "huggingface_hub[cli]"

# 3. models (read their licenses first)
PATH=$PWD/.venv/bin:$PATH scripts/download-models.sh qwen /models     # and/or: glm

# 4. settings, then start
cp spark.env.example spark.env && $EDITOR spark.env
engine-rs/serve/start.sh qwen            # or: glm (run on the first node)

curl http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":200}'
```

The release asset names carry the version, for example `spark-engine-v0.1.0-linux-aarch64-cu130-sm121.tar.gz`; pick it from
the [releases page](https://github.com/fly9i/spark-engine/releases).

## Build from source

See [docs/build.md](docs/build.md). In short: CUDA 13.0 toolkit, Rust, a Python 3.12 environment with PyTorch 2.13
(cu130), then

```bash
scripts/fetch-deps.sh                                   # pinned third-party sources (torch-sys, NCCL header)
PYTHON=.venv/bin/python scripts/build-exllamav3.sh      # EXL3 GEMM library from exllamav3 v1.4.9 (MIT)
export LIBTORCH=$(.venv/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
cd engine-rs && cargo build --release                   # -> engine-rs/target/release/spark-engine
```

GitHub Actions builds every push on an arm64 runner and publishes release archives for tags.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/deploy.md](docs/deploy.md) | Hardware, installation, model download, two-node setup, start / stop, troubleshooting |
| [docs/build.md](docs/build.md) | Building from source, dependencies, CI |
| [docs/features.md](docs/features.md) | How the engine works: GLM and Qwen paths, kernels, speculative decoding, memory |
| [docs/configuration.md](docs/configuration.md) | `spark.env`, profiles, the most useful engine and front-end switches |
| [docs/benchmarks.md](docs/benchmarks.md) | How to measure speed, reference numbers |
| [docs/abliteration.md](docs/abliteration.md) | Optional refusal-removal (both mechanisms): what it is, how to enable, risks |

## Models and licenses

This repository contains code only. Models are downloaded separately and keep their own licenses:

| Model | Source | License |
| --- | --- | --- |
| GLM-5.3-Flash EXL3 4 bpw | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | see the model card |
| DFlash2 drafter for GLM | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | **CC BY-NC-ND 4.0 (non-commercial)** |
| Qwen3.8-Flash-Next EXL3 | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (`4.05bpw_h6_ng6`) | Qwen license (see the model card) |

Third-party software used at build or run time is listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## License

[MIT](LICENSE).
