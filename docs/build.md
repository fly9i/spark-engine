# Building from source

The engine targets NVIDIA DGX Spark: aarch64, CUDA 13.0, compute capability 12.1 (`sm_121`). Building does not need a
GPU (GitHub Actions builds on a GPU-less arm64 runner), but running does.

## Requirements

| Item | Version | Notes |
| --- | --- | --- |
| OS | Ubuntu 24.04 / DGX OS (aarch64) | |
| CUDA toolkit | 13.0 | `nvcc`, cuBLAS / cuBLASLt development files, NVTX headers |
| Python | 3.12 with `venv` and development headers | `python3.12-dev python3.12-venv` |
| PyTorch | 2.13.0, CUDA 13.0 build | from `https://download.pytorch.org/whl/cu130`; provides libtorch and NCCL |
| Rust | stable (1.85+) | via rustup |
| System libraries | `libibverbs-dev` | RDMA all-reduce |
| Tools | `git`, `curl`, `patch`, `ninja` (pip) | |

## Steps

```bash
git clone https://github.com/fly9i/spark-engine && cd spark-engine

# Python environment (also used at run time)
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel ninja
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

# 1. pinned third-party sources into third_party/: torch-sys 0.20.0 (+ patch for libtorch 2.13), NCCL 2.29.7 header
scripts/fetch-deps.sh

# 2. the EXL3 GEMM library: exllamav3 v1.4.9, CUDA parts only -> third_party/exllamav3/lib/libexllamav3_ext.so
#    (20-60 minutes depending on cores; MAX_JOBS limits parallel nvcc processes, each needs a few GB of memory)
PYTHON=$PWD/.venv/bin/python MAX_JOBS=8 scripts/build-exllamav3.sh

# 3. the engine
export LIBTORCH=$(.venv/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export CUDA_HOME=/usr/local/cuda-13.0
cd engine-rs && cargo build --release
```

The binary is `engine-rs/target/release/spark-engine`. Point `SPARK_BIN` in `spark.env` at it (or package it with
`scripts/package.sh`, which produces the same archive as a release).

### Build variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `LIBTORCH` | (required) | PyTorch installation to compile and link against (the `torch` package directory) |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 13 toolkit |
| `EXL3_LIB_DIR` | `third_party/exllamav3/lib` | directory with `libexllamav3_ext.so`; also embedded as an rpath next to `$ORIGIN/../lib` |
| `PYTHON_INCLUDE` | `/usr/include/python3.12` | Python headers (included through PyTorch's extension headers) |
| `SPARK_CUDA_ARCH` | `121` | SM version for the engine's kernels (GB10) |
| `BUILD_JOBS` | `8` | parallel C++/CUDA compiles in `build.rs` (each nvcc process can take several GB) |
| `MAX_JOBS`, `TORCH_CUDA_ARCH_LIST` | `nproc`, `12.1` | for `scripts/build-exllamav3.sh` |

`build.rs` caches compiled objects by content hash in `engine-rs/target/obj-cache`, so changing one `.cu` file rebuilds
only that object.

## Repository layout

| Path | Contents |
| --- | --- |
| `engine-rs/src` | engine (Rust): loading, GLM forward and speculative decoding, tensor parallelism, serving loop; `src/qwen/` the Qwen3.8 model, serving and drafting |
| `engine-rs/shim` | CUDA / C++ kernels and the libtorch bridge; `moe_exl3.cuh` (EXL3 MoE shared by both models), `qwen_*.cu`, `glm_moe.cu`, `c12*`, `q8.cuh`, `rdma_*.cu`, ... |
| `engine-rs/serve` | OpenAI-compatible front ends (`openai_server.py`, `qwen_server.py`, media handling) and start/stop scripts |
| `engine-rs/profiles` | `glm-tp2.env`: the GLM serving profile ([switch reference](glm-switches.md)) |
| `scripts` | dependency fetch / build, runtime environment, packaging, model download |
| `patches` | patch applied to the fetched `torch-sys` |
| `bench` | speed and text-consistency tools |
| `reference` | Python reference implementations used to validate the kernels (developer tools; need extra packages) |
| `assets` | the Qwen MTP draft sub-vocabulary |

## Continuous integration

`.github/workflows/build.yml` runs on GitHub's `ubuntu-24.04-arm` runner: CUDA 13.0 toolkit from NVIDIA's apt repository,
PyTorch 2.13 cu130, the exllamav3 library (cached between runs), then `cargo build --release` and `scripts/package.sh`.
The archive is uploaded as a workflow artifact; for `v*` tags it is attached to a GitHub release.
