# spark-engine

[English](README.md)

一个面向 **NVIDIA DGX Spark（GB10）** 的推理引擎，通过 OpenAI 兼容 API 服务两个大型 MoE 模型：

- **GLM-5.3-Flash**（EXL3 4 bpw），运行在**两台 DGX Spark** 上（通过 ConnectX-7 RoCE 链路做张量并行），使用 DFlash2
  投机解码；
- **Qwen3.8-Flash-Next**（EXL3 4.05 bpw），运行在**一台 DGX Spark** 上，使用其原生 MTP 投机解码，支持图像和视频输入以及
  1M token 上下文。

引擎由 Rust 和 CUDA 编写（张量相关的底层管理使用 PyTorch 的 C++ 库，稠密 prefill GEMM 使用 cuBLAS）。两个模型的 EXL3
内核、MoE 流水线、投机解码状态管理、RDMA all-reduce 以及服务前端都包含在本仓库中。同一个二进制可运行任一模型，由
`start.sh glm|qwen` 选择。

## 性能

于 2026-10-08 使用本版本的二进制和默认设置测得，测量工具为 [bench/decode_bench.py](bench/decode_bench.py)
和 [bench/prefill_bench.py](bench/prefill_bench.py)：贪心解码，关闭思考，输出 400 个 token，取 3 次运行的中位数，
1–4 个并发流的总 tok/s；TTFT 为短提示词的首 token 时延。测量方法和更多数据见 [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md)。

| 模型 | 硬件 | 散文，1 / 2 / 3 / 4 流 | 结构化，1 / 2 / 3 / 4 流 | Prefill | TTFT |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 2× DGX Spark | 52.9 / 66.5 / 73.3 / 82.9 tok/s | 92.5 / 119.1 / 158.0 / 168.0 tok/s | 1,520–1,560 tok/s（10K 和 55K 提示词） | ~370 ms |
| Qwen3.8-Flash-Next | 1× DGX Spark | 64.1 / 85.5 / 105.0 / 119.5 tok/s | 149.8 / 192.9 / 232.6 / 217.4 tok/s | 1,870–1,920 tok/s（13K 和 77K 提示词） | ~150 ms |

“散文”要求输出一段长篇解释（草稿接受率较低）；“结构化”要求从 1 数到 200（高度可预测）。
投机解码使单流速度依赖于生成的文本：同一构建在不同提示词之间，或舍入方式不同的两个构建之间，可能相差几个百分点。

## 功能

**两个模型通用**
- OpenAI 兼容 HTTP API：`/v1/chat/completions`（流式输出、usage、停止字符串、工具调用、与回答分离的推理内容）、
  `/v1/completions`、`/v1/models`、`/tokenize`、`/detokenize`、`/health`，以及使用 vLLM 兼容指标名的 Prometheus
  `/metrics`。
- 多序列并发，批量投机验证（多个序列共享一次前向），所有 decode 形状均使用 CUDA graph，FP8 KV 缓存，缓存池大小根据空闲内存确定。
- 图像和视频输入（OpenAI `image_url`、`video_url`、data URL、base64、HTTP(S)），由引擎内模型自带的视觉塔编码。
- 长上下文：1M token KV 预算；跨请求复用前缀。
- 针对 GB10 统一内存设计：权重、KV 和临时缓冲区的大小均按 `MemAvailable` 确定，启动脚本内置内存保护。

**GLM-5.3-Flash（双节点）**
- 在两台 DGX Spark 上做张量并行：基于 RoCE 的 NCCL，加上引擎自带的用于小消息的 RDMA all-reduce
  （固定求和顺序，与 NCCL 逐位一致，每次调用 12–50 µs）。
- EXL3 4 bpw 路由专家（每个 rank 288 个）运行在引擎的 MoE 内核上：decode 使用一个融合的常驻（persistent）内核
  （gate/up、SwiGLU、down），prefill 使用分组 tensor core GEMM，并启用 programmatic dependent launch。
- decode 稠密权重使用 Q8（int8 + 每 128 个一个 FP32 scale，按 512 字节加载分块排布）；另提供无损 12 bit 编码（C12）；
  prefill 以原始精度运行稠密权重。
- DFlash2 草稿模型，支持树/链验证、KDA（线性注意力）状态修正重放、MLA 共享基础状态、DSA 稀疏注意力选择和基于置信度的截断；
  支持复制（prompt-lookup）草稿。
- NVMe 上的持久化前缀缓存（`GLM53_PCACHE`）。
- 可选的 o_proj 移植（来自 abliterated 供体 checkpoint），默认关闭（[docs/abliteration.zh-CN.md](docs/abliteration.zh-CN.md)）。

**Qwen3.8-Flash-Next（单节点）**
- 完整的混合架构：门控 DeltaNet 线性注意力、带索引器的稀疏注意力、512 专家 top-10 MoE、4 流 hyper-connection
  以及逐层 n-gram 嵌入（从 NVMe 内存映射）。
- 所有 EXL3 乘法（mul1 码本）均运行在引擎自有内核上；不依赖 exllamav3 运行时。
- 原生 MTP 草稿（使用 4 bit 子词表草稿头）和 prompt-lookup 草稿，最多 8 个序列批量验证。
- 1M 上下文（超过 262,144 token 时使用 YaRN），KV 池按粒度分配；多轮对话在最后一条消息边界处保存提示词检查点
  （16K 对话的后续轮次 TTFT 约 0.3 s）；`max_tokens` 按需预留，而非预先全部预留。
- 推理强度等级（`reasoning_effort`：none / low / medium / high）。

## 快速开始（发布版二进制）

要求：DGX Spark（GB10），DGX OS / Ubuntu 24.04，支持 CUDA 13 的 NVIDIA 驱动，Python 3.12。
GLM-5.3-Flash 需要两台通过 ConnectX-7 端口互连的 DGX Spark。完整说明见 [docs/deploy.zh-CN.md](docs/deploy.zh-CN.md)。

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

发布包的文件名带有版本号，例如 `spark-engine-v0.1.0-linux-aarch64-cu130-sm121.tar.gz`；请从
[发布页面](https://github.com/fly9i/spark-engine/releases)选择。

## 从源码构建

见 [docs/build.zh-CN.md](docs/build.zh-CN.md)。简而言之：CUDA 13.0 工具链、Rust、带 PyTorch 2.13（cu130）的 Python 3.12 环境，
然后执行

```bash
scripts/fetch-deps.sh                                   # pinned third-party sources (torch-sys, NCCL header)
PYTHON=.venv/bin/python scripts/build-exllamav3.sh      # EXL3 GEMM library from exllamav3 v1.4.9 (MIT)
export LIBTORCH=$(.venv/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
cd engine-rs && cargo build --release                   # -> engine-rs/target/release/spark-engine
```

GitHub Actions 在 arm64 runner 上构建每次推送，并为 tag 发布归档包。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/deploy.zh-CN.md](docs/deploy.zh-CN.md) | 硬件、安装、模型下载、双节点配置、启动 / 停止、故障排查 |
| [docs/build.zh-CN.md](docs/build.zh-CN.md) | 从源码构建、依赖、CI |
| [docs/features.zh-CN.md](docs/features.zh-CN.md) | 引擎工作原理：GLM 与 Qwen 路径、内核、投机解码、内存 |
| [docs/configuration.zh-CN.md](docs/configuration.zh-CN.md) | `spark.env`、profile、最常用的引擎与前端开关 |
| [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md) | 如何测量速度、参考数据 |
| [docs/abliteration.zh-CN.md](docs/abliteration.zh-CN.md) | 可选的拒答移除移植：是什么、如何启用、风险 |

## 模型与许可证

本仓库只包含代码。模型需单独下载，并保留各自的许可证：

| 模型 | 来源 | 许可证 |
| --- | --- | --- |
| GLM-5.3-Flash EXL3 4 bpw | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | 见模型卡 |
| GLM 的 DFlash2 草稿模型 | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | **CC BY-NC-ND 4.0（非商业）** |
| Qwen3.8-Flash-Next EXL3 | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)（`4.05bpw_h6_ng6`） | Qwen 许可证（见模型卡） |

构建或运行时使用的第三方软件列于 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 许可证

[MIT](LICENSE)。
