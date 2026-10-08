![spark-engine](docs/assets/banner.png)

# spark-engine

[English](README.md)

一个为 **NVIDIA DGX Spark（GB10，aarch64，CUDA 13 / sm_121）** 从零写的推理引擎，用一套 OpenAI 兼容接口服务两个大型
混合 MoE 模型，EXL3 4-bit 量化、带投机解码、支持 1M token 上下文：

- **GLM-5.3-Flash**：**两台 DGX Spark**（经 ConnectX-7 RoCE 做张量并行），配 DFlash2 草稿器；
- **Qwen3.8-Flash-Next**：**单台 DGX Spark**，原生 MTP 草稿、图片/视频输入、完整混合架构。

所有影响性能的部分都是本项目自研代码——EXL3 的 GEMM/MoE 内核、GDN/KDA 线性注意力与稀疏注意力内核、投机解码状态机、
RDMA all-reduce、FP8 KV、NVMe 前缀缓存、服务前端。只链接 PyTorch 的 libtorch（张量胶水、cuBLAS、NCCL）和 EXL3 反量化
内核。一个二进制跑两个模型，`start.sh glm|qwen` 选一个。

## 概览

2026-10-08 用本发布版二进制、默认配置、贪心、关思考实测（方法与更多数据见 [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md)）。

| | **GLM-5.3-Flash** | **Qwen3.8-Flash-Next** |
| --- | --- | --- |
| 硬件 | 2× DGX Spark（TP2，RoCE） | 1× DGX Spark |
| 架构 | 45 层：KDA 线性注意力 + MLA/DSA 稀疏注意力，288 专家 MoE | 48 层：门控 DeltaNet + QSA 稀疏注意力，512 专家 top-10 MoE，逐层 n-gram |
| **解码 — prose**（1/2/3/4 路） | 52.9 / 66.5 / 73.3 / 82.9 tok/s | 64.1 / 85.5 / 105.0 / 119.5 tok/s |
| **解码 — structured**（1/2/3/4 路） | 92.5 / 119.1 / 158.0 / 168.0 tok/s | 149.8 / 192.9 / 232.6 / 226.5 tok/s |
| **prefill** | ~1,560 tok/s（10K–55K 提示） | ~1,900 tok/s（13K–77K 提示） |
| **TTFT**（短提示） | ~370 ms | ~150 ms |
| **启动**（加载→可服务） | ~33 s | ~54 s |
| 投机解码 | DFlash2 草稿器，树/链验证 | 原生 MTP + prompt-lookup |

"prose" = 长篇解释（草稿接受率低）；"structured" = 从 1 数到 200（接受率高）。聚合是各路相加；因为有投机，解码速度和文本相关。

### 量化与运行时精度

| 部件 | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| 路由专家（权重） | EXL3 **4 bpw**，mcg 码本 | EXL3 **4.05 bpw**（头 6、MTP 4），mul1 码本 |
| 稠密权重 — prefill | 原精度（FP16），cuBLAS | 原精度 |
| 稠密权重 — decode | **Q8**（int8 + 每 128 一个 FP32 scale，约 0.67% RMS；可选无损 12-bit C12） | FP16（可选 int8 副本） |
| 注意力计算 | MLA/DSA 用 FP32/TF32；DSA indexer 稀疏 | QSA FP32 累加；学习式稀疏选择 |
| **KV cache** | **FP8**（e4m3） | **FP8**（e4m3） |
| 输出头 | FP16 | 6 bpw |

全程用的精度分级：**L0** 逐位相同、**L1** 舍入级（不劣于 FP32 参考）、**L2** 仅草稿侧、**L3** 有意的有损。有损项（L3）
是 FP8 KV cache 和 GLM 的 Q8 decode 稠密权重，二者都实测与参考等价；路由专家保留检查点自带的 EXL3 量化。

### 上下文长度与并发

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| 最大上下文（KV 预算） | **1,048,576 token** | **1,048,576 token** |
| 该预算下 KV 显存 | ~7.05 GiB / 节点（11 个 MLA 层，7,216 B/token） | ~14.0 GiB（12 个稀疏注意力 + MTP，14,364 B/token） |
| 默认并发序列 | 4 | 8 |
| 批量投机验证 | 是（多序列共用一次前向） | 是（最多 8） |
| 前缀复用 | 内存存储 + NVMe 持久缓存（`GLM53_PCACHE`） | 多轮 prompt 检查点 + NVMe 缓存（`QWEN_PCACHE`） |
| 超长上下文 > 262K | — | YaRN 缩放 |

两个模型都是混合架构：大多数层是线性/门控注意力，状态大小固定（不随上下文增长），所以 1M token 窗口只花 7–14 GiB KV——
GB10 的 128 GB 统一内存其余部分放权重。

## 本项目的特点

- **专为 GB10 编写**：内核按 sm_121 的共享内存/占用率限制设计；decode 权重重排成每条 warp 加载正好连续 512 B（贴近
  ~230–250 GB/s 流式读上限）；每个 decode 形状都捕获 CUDA 图；内存按统一地址空间上的 `MemAvailable` 分配。
- **两节点经 RoCE 做张量并行**，配自研的小消息 RDMA all-reduce（固定求和顺序、与 NCCL 逐位相同、12–50 µs），并融进消费它的
  hyper-connection 更新里。
- **确定性、图捕获的投机解码**，保持精确贪心语义：KDA 状态修正回放、MLA 共享基态、DSA 逐行选择、置信度截断的草稿树、
  prompt-lookup（copy）草稿。
- **自研 MoE**：一套模板化 EXL3 MoE 内核两个模型共用（融合式持久 decode；分组张量核 prefill），不依赖任何外部 MoE 库。
- **完整的多模态与长上下文服务**：图片/视频走模型自带视觉塔，1M token KV 池按 16K 粒度 LRU、KV 区间原地增长、NVMe 持久
  前缀缓存。
- **一个 OpenAI 兼容二进制**服务两个模型，带 Prometheus 指标和针对 GB10 统一内存的内存护栏。

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
| [docs/abliteration.zh-CN.md](docs/abliteration.zh-CN.md) | 可选的拒答移除（两种机制）：是什么、如何启用、风险 |
| [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md) | 如何测量速度、参考数据 |

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
