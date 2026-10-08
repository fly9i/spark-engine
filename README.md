# spark-engine

[中文](README.zh-CN.md)

An inference engine for **NVIDIA DGX Spark (GB10)** that serves two large MoE models through an OpenAI-compatible API:

- **GLM-5.3-Flash** (EXL3 4 bpw) across **two DGX Sparks** (tensor parallel over the ConnectX-7 RoCE link), with DFlash2
  speculative decoding;
- **Qwen3.8-Flash-Next** (EXL3 4.05 bpw) on **one DGX Spark**, with its native MTP speculative decoding, image and video
  input and a 1M-token context.

The engine is written in Rust and CUDA (PyTorch's C++ library for tensor plumbing, cuBLAS for dense prefill GEMMs). The
EXL3 kernels for both models, the MoE pipeline, the speculative-decoding state machinery, the RDMA all-reduce and the
serving front end are part of this repository. One binary runs either model; `start.sh glm|qwen` picks one.

## Performance

Measured on 2026-10-08 with this release's binary and default settings, using [bench/decode_bench.py](bench/decode_bench.py)
and [bench/prefill_bench.py](bench/prefill_bench.py): greedy decoding, thinking off, 400 output tokens, median of 3 runs,
aggregate tok/s over 1–4 concurrent streams; TTFT of a short prompt. See [docs/benchmarks.md](docs/benchmarks.md) for the method and more numbers.

| Model | Hardware | Prose, 1 / 2 / 3 / 4 streams | Structured, 1 / 2 / 3 / 4 streams | Prefill | TTFT |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 2× DGX Spark | 52.9 / 66.5 / 73.3 / 82.9 tok/s | 92.5 / 119.1 / 158.0 / 168.0 tok/s | 1,520–1,560 tok/s (10K and 55K prompts) | ~370 ms |
| Qwen3.8-Flash-Next | 1× DGX Spark | 64.1 / 85.5 / 105.0 / 119.5 tok/s | 149.8 / 192.9 / 232.6 / 217.4 tok/s | 1,870–1,920 tok/s (13K and 77K prompts) | ~150 ms |

"Prose" asks for a long explanation (lower draft acceptance); "structured" asks to count to 200 (highly predictable).
Speculative decoding makes single-stream speed depend on the text: the same build can differ by a few percent between
prompts or between two builds that round differently.

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
