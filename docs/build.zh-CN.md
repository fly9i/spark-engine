# 从源码构建

引擎的目标平台是 NVIDIA DGX Spark：aarch64、CUDA 13.0、计算能力 12.1（`sm_121`）。构建不需要 GPU
（GitHub Actions 在没有 GPU 的 arm64 runner 上构建），但运行需要。

## 要求

| 项目 | 版本 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 24.04 / DGX OS（aarch64） | |
| CUDA 工具链 | 13.0 | `nvcc`、cuBLAS / cuBLASLt 开发文件、NVTX 头文件 |
| Python | 3.12，带 `venv` 和开发头文件 | `python3.12-dev python3.12-venv` |
| PyTorch | 2.13.0，CUDA 13.0 版本 | 来自 `https://download.pytorch.org/whl/cu130`；提供 libtorch 和 NCCL |
| Rust | stable（1.85+） | 通过 rustup 安装 |
| 系统库 | `libibverbs-dev` | RDMA all-reduce |
| 工具 | `git`、`curl`、`patch`、`ninja`（pip） | |

## 步骤

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

生成的二进制为 `engine-rs/target/release/spark-engine`。将 `spark.env` 中的 `SPARK_BIN` 指向它（或用
`scripts/package.sh` 打包，生成与发布版相同的归档包）。

### 构建变量

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `LIBTORCH` | （必填） | 用于编译和链接的 PyTorch 安装（`torch` 包所在目录） |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 13 工具链 |
| `EXL3_LIB_DIR` | `third_party/exllamav3/lib` | 包含 `libexllamav3_ext.so` 的目录；同时作为 rpath 嵌入，与 `$ORIGIN/../lib` 并列 |
| `PYTHON_INCLUDE` | `/usr/include/python3.12` | Python 头文件（通过 PyTorch 的扩展头文件引入） |
| `SPARK_CUDA_ARCH` | `121` | 引擎内核的 SM 版本（GB10） |
| `BUILD_JOBS` | `8` | `build.rs` 中并行的 C++/CUDA 编译数（每个 nvcc 进程可能占用数 GB） |
| `MAX_JOBS`、`TORCH_CUDA_ARCH_LIST` | `nproc`、`12.1` | 用于 `scripts/build-exllamav3.sh` |

`build.rs` 在 `engine-rs/target/obj-cache` 中按内容哈希缓存编译产物，因此修改一个 `.cu` 文件只会重新构建对应的目标文件。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `engine-rs/src` | 引擎（Rust）：加载、GLM 前向与投机解码、张量并行、服务循环；`src/qwen/` 为 Qwen3.8 模型、服务和草稿生成 |
| `engine-rs/shim` | CUDA / C++ 内核和 libtorch 桥接；`moe_exl3.cuh`（两个模型共用的 EXL3 MoE）、`qwen_*.cu`、`glm_moe.cu`、`c12*`、`q8.cuh`、`rdma_*.cu` 等 |
| `engine-rs/serve` | OpenAI 兼容前端（`openai_server.py`、`qwen_server.py`、媒体处理）和启动/停止脚本 |
| `engine-rs/profiles` | `glm-tp2.env`：GLM 服务 profile（[开关说明](glm-switches.zh-CN.md)） |
| `scripts` | 依赖拉取 / 构建、运行时环境、打包、模型下载、abliteration 供体拉取 |
| `patches` | 应用到拉取的 `torch-sys` 上的补丁 |
| `bench` | 速度和文本一致性工具 |
| `reference` | 用于验证内核的 Python 参考实现（开发者工具；需要额外的包） |
| `assets` | Qwen MTP 草稿子词表 |

## 持续集成

`.github/workflows/build.yml` 运行在 GitHub 的 `ubuntu-24.04-arm` runner 上：从 NVIDIA 的 apt 仓库安装 CUDA 13.0 工具链，
安装 PyTorch 2.13 cu130，构建 exllamav3 库（在多次运行之间缓存），然后执行 `cargo build --release` 和 `scripts/package.sh`。
归档包作为 workflow artifact 上传；对于 `v*` tag，会附加到 GitHub release。
