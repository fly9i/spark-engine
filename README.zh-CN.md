![spark-engine](docs/assets/banner-tp3.png)

# spark-engine

[English](README.md)

一个为 **NVIDIA DGX Spark（GB10，aarch64，CUDA 13 / sm_121）** 从零写的推理引擎，用一套 OpenAI 兼容接口服务三个大型
MoE 模型，带投机解码、支持 1M token 上下文：

- **GLM-5.3-Flash**（EXL3 4-bit）：**两台或三台 DGX Spark**（经 ConnectX-7 RoCE 链路做张量并行），配 DFlash2 草稿器；
- **DeepSeek-V4.1-Flash**（官方 MXFP4/FP8 检查点，`-1017`）：**三台 DGX Spark**，DSpark 投机解码、Engram 表从 NVMe
  流式读取、图片输入；
- **Qwen3.8-Flash-Next**（EXL3 4-bit）：**单台 DGX Spark**，原生 MTP 草稿、图片/视频输入、完整混合架构。

所有影响性能的部分都是本项目自研代码——EXL3 的 GEMM/MoE 内核、GDN/KDA 线性注意力与稀疏注意力内核、投机解码状态机、
RDMA all-reduce、FP8 KV、NVMe 前缀缓存、服务前端。只链接 PyTorch 的 libtorch（张量胶水、cuBLAS、NCCL）和 EXL3 反量化
内核。一个二进制跑所有模型，`start.sh glm|glm3|ds41|qwen` 选一个。

## 概览

**解码测速方法**（DeepSeek 2026-10-30，v0.3.9；Qwen 2026-10-29，v0.3.8；与 DGX Spark 社区常用的方法一致）：用 `bench/decode_bench.py` 测运行中的服务，两种提示——
*prose*（逐步讲解哈希表原理）与 *structured*（从 1 数到 200）——每个请求输出 400 token，1~4 路并发（多路时每路提示前加一行 `Request ID: decode-<i>`），
每档 5 次取中位数。每行注明采样方式：**贪心、关思考**（温度 0）或 **T 1、开思考**（温度 1，思考 token 计入）。聚合 = 全部解码 token ÷ 从最早一路首 token
到最晚一路末 token 的时间窗。GLM 的数据较旧（2026-10-08，v0.1.1，贪心、关思考，多路用句尾 `(stream i/n)` 而不是 `Request ID` 行）。预填、TTFT 与启动：
DeepSeek 预填 2026-10-30，Qwen 预填 2026-10-29，启动 2026-10-28。更多见 [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md)、[docs/ds41.zh-CN.md](docs/ds41.zh-CN.md#7-性能)。DeepSeek 与 Qwen 在
[测试系统与 GPU 频率](#测试系统与-gpu-频率)所述的更新后系统上、默认 GPU 时钟下测得（连续满载，每秒采样时钟，未记录到降频）；GLM 一列是旧数据（DGX OS 7.5.0，GPU 时钟锁定 2,200 MHz）。

| | **GLM-5.3-Flash** | **DeepSeek-V4.1-Flash** | **Qwen3.8-Flash-Next** |
| --- | --- | --- | --- |
| 硬件 | 2× DGX Spark（TP2，RoCE）；也支持 3× | 3× DGX Spark（TP3，RoCE 全互连） | 1× DGX Spark |
| 架构 | 45 层：KDA 线性注意力 + MLA/DSA 稀疏注意力，288 专家 MoE | 40 层（20 编码器 + 20 解码器）：带分层 indexer 的压缩稀疏注意力，384 专家 top-6 MoE，Engram n-gram 表，mHC | 48 层：门控 DeltaNet + QSA 稀疏注意力，512 专家 top-10 MoE，逐层 n-gram |
| **解码 — prose，贪心、关思考**（1/2/3/4 路） | 52.9 / 66.5 / 73.3 / 82.9 tok/s | 67.4 / 92.8 / 109.3 / 124.7 tok/s | 68.4 / 98.0 / 122.4 / 139.2 tok/s |
| **解码 — structured，贪心、关思考**（1/2/3/4 路） | 92.5 / 119.1 / 158.0 / 168.0 tok/s | 126.6 / 181.9 / 194.8 / 226.0 tok/s | 157.8 / 230.7 / 280.1 / 310.3 tok/s |
| **解码 — prose，T 1、开思考**（1/2/3/4 路） | 未测 | 57.2 / 82.2 / 95.0 / 105.5 tok/s | 51.2 / 74.3 / 90.9 / 103.8 tok/s |
| **解码 — structured，T 1、开思考**（1/2/3/4 路） | 未测 | 109.2 / 153.3 / 153.5 / 180.9 tok/s | 122.7 / 78.1 / 100.9 / 108.0 tok/s |
| **prefill** | ~1,560 tok/s（10K–55K 提示） | ~5,570 tok/s（8K–64K 提示，新文本；4K 约 4,760）；128K 上下文 decode 每步 27.0 ms；1M token 上下文：prefill 约 4,540 tok/s，decode 约 56 tok/s（单路，贪心） | ~2,000 tok/s（13K–77K 提示） |
| **TTFT**（短提示） | ~370 ms | ~200 ms；7K 提示从 NVMe 前缀缓存恢复时 185 ms | ~140 ms |
| **启动**（加载→可服务） | ~33 s | ~16 s（冷启动，分片直接从 NVMe 读） | ~19 s（冷启动） |
| 投机解码 | DFlash2 草稿器，树/链验证 | DSpark（检查点自带的草稿级），每序列草稿数由代价模型决定 | 原生 MTP + prompt-lookup |

"prose" = 长篇解释（草稿接受率低）；"structured" = 从 1 数到 200（接受率高）。因为有投机，解码速度和文本相关。

### 量化与运行时精度

| 部件 | GLM-5.3-Flash | DeepSeek-V4.1-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- | --- |
| 路由专家（权重） | EXL3 **4 bpw**，mcg 码本 | 官方 **MXFP4**（E2M1 + 每 32 一个 E8M0） | EXL3 **4.05 bpw**（头 6、MTP 4），mul1 码本 |
| 稠密权重 — prefill | 原精度（FP16），cuBLAS | 官方 FP8（32×32 块 scale）/ BF16，块缩放张量核 MMA | 原精度 |
| 稠密权重 — decode | **Q8**（int8 + 每 128 一个 FP32 scale，约 0.67% RMS；可选无损 12-bit C12） | 官方 FP8 / BF16，块缩放 GEMV | FP16（可选 int8 副本） |
| 注意力计算 | MLA/DSA 用 FP32/TF32；DSA indexer 稀疏 | 稀疏注意力，FP32 累加；FP4 indexer，官方 KV 格式 | QSA FP32 累加；学习式稀疏选择 |
| **KV cache** | **FP8**（e4m3） | 官方：FP8 窗口，FP4 压缩全局 KV | **FP8**（e4m3） |
| 输出头 | FP16 | BF16（FP8 副本仅用于草稿） | 6 bpw |

全程用的精度分级：**L0** 逐位相同、**L1** 舍入级（不劣于 FP32 参考）、**L2** 仅草稿侧、**L3** 有意的有损。有损项（L3）
是 FP8 KV cache 和 GLM 的 Q8 decode 稠密权重，二者都实测与参考等价；路由专家保留检查点自带的 EXL3 量化。DeepSeek 运行官方检查点
自带的格式：对官方参考的 prefill KL 为 0.0236，噪声地板为 0.0224。

### 上下文长度与并发

| | GLM-5.3-Flash | DeepSeek-V4.1-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- | --- |
| 最大上下文（KV 预算） | **1,048,576 token** | **1,048,576 token** | **1,048,576 token** |
| 该预算下 KV 显存 | ~7.05 GiB / 节点（11 个 MLA 层，7,216 B/token） | ~3.3 GiB / 节点（1.1 M token 池） | ~14.0 GiB（12 个稀疏注意力 + MTP，14,364 B/token） |
| 默认并发序列 | 4 | 4 | 8 |
| 批量投机验证 | 是（多序列共用一次前向） | 是（最多 4 个序列，超过则普通批量解码） | 是（最多 8） |
| 前缀复用 | 内存存储 + NVMe 持久缓存（`GLM53_PCACHE`） | 内存存储 + NVMe 缓存（`DS41_PCACHE`，每 512 token 一个检查点） | 多轮 prompt 检查点 + NVMe 缓存（`QWEN_PCACHE`） |
| 超长上下文 > 262K | — | — | YaRN 缩放 |

GLM 与 Qwen 是混合架构：大多数层是线性/门控注意力，状态大小固定（不随上下文增长），所以 1M token 窗口只花 7–14 GiB KV。
DeepSeek 压缩全局 KV（一个共享 KV 头，FP4），每层只保留 128 token 的 FP8 窗口。GB10 的 128 GB 统一内存其余部分放权重。

## 本项目的特点

- **专为 GB10 编写**：内核按 sm_121 的共享内存/占用率限制设计；decode 权重重排成每条 warp 加载正好连续 512 B（贴近
  ~230–250 GB/s 流式读上限）；每个 decode 形状都捕获 CUDA 图；内存按统一地址空间上的 `MemAvailable` 分配。
- **两节点、三节点经 RoCE 做张量并行**，配自研的 RDMA all-reduce（固定求和顺序、两节点时与 NCCL 逐位相同、12–50 µs），并融进
  消费它的 hyper-connection 更新里；三节点时在直连全互连上做 N 路 RDMA all-reduce / reduce-scatter / all-gather，链路按子网配对。
- **一份代码、一个二进制、按模型配置**：PDL 启动、top-k 选择、mHC、KV 池（CUDA VMM，原地增长）、多模态画布和 NVMe 前缀缓存
  核心由三个模型共用。
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

**GLM-5.3-Flash（两节点或三节点）**
- 在两台 DGX Spark 上做张量并行：基于 RoCE 的 NCCL，加上引擎自带的用于小消息的 RDMA all-reduce
  （固定求和顺序，与 NCCL 逐位一致，每次调用 12–50 µs）。三节点（`start.sh glm3`）：KDA 22/21/21 头，路由专家每个
  6/5/5 个 Hadamard 块，序列并行 prefill，N 路 RDMA（其在公开检查点上的验证状态见 docs/deploy.zh-CN.md）。
- EXL3 4 bpw 路由专家（每个 rank 288 个）运行在引擎的 MoE 内核上：decode 使用一个融合的常驻（persistent）内核
  （gate/up、SwiGLU、down），prefill 使用分组 tensor core GEMM，并启用 programmatic dependent launch。
- decode 稠密权重使用 Q8（int8 + 每 128 个一个 FP32 scale，按 512 字节加载分块排布）；另提供无损 12 bit 编码（C12）；
  prefill 以原始精度运行稠密权重。
- DFlash2 草稿模型，支持树/链验证、KDA（线性注意力）状态修正重放、MLA 共享基础状态、DSA 稀疏注意力选择和基于置信度的截断；
  支持复制（prompt-lookup）草稿。
- NVMe 上的持久化前缀缓存（`GLM53_PCACHE`）；提示词从已缓存序列分叉处保存分叉检查点。
- 可选 abliteration，默认关闭（[docs/abliteration.zh-CN.md](docs/abliteration.zh-CN.md)）：o_proj 移植（`GLM53_ABLIT`）、方向消融（`SPARK_ABLATE`）和条件钳制（`SPARK_ABLATE_CLAMP`，DeepSeek 也可用）。

**DeepSeek-V4.1-Flash（三节点）**——部署方法：[docs/ds41.zh-CN.md](docs/ds41.zh-CN.md)
- 官方检查点原样使用（MXFP4 专家、FP8 块缩放权重），切分到三台 DGX Spark：注意力按头组切，专家和共享专家按中间维切（完全均衡），
  词表按行切。
- MXFP4 MoE 与 FP8 GEMM 用原生块缩放张量核 MMA（sm_121a）；多行 GEMV 的开销几乎不随行数增长，因此批量解码和投机验证几乎免费。
- CED prefill（解码器只跑最后一个窗口），prefill 通信与计算重叠；Engram 行用 O_DIRECT 从 NVMe 读取，提前一个块预取。
- DSpark 投机解码，每序列草稿数由代价模型决定；NVMe 前缀缓存，每 512 token 一个检查点；图片输入；官方提示词编码、工具调用和
  思考预算。

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
GLM-5.3-Flash 需要两台（或三台）通过 ConnectX-7 端口互连的 DGX Spark，DeepSeek-V4.1-Flash 需要三台。完整说明见
[docs/deploy.zh-CN.md](docs/deploy.zh-CN.md)、[docs/ds41.zh-CN.md](docs/ds41.zh-CN.md)。

```bash
# 1. the release (same path on every node for GLM / DeepSeek)
sudo mkdir -p /opt/spark-engine && cd /opt/spark-engine
curl -L https://github.com/fly9i/spark-engine/releases/latest/download/spark-engine-linux-aarch64-cu130-sm121.tar.gz | sudo tar xz --strip-components=1

# 2. Python environment: PyTorch 2.13 (CUDA 13.0) for the engine, plus the front end's packages
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install jinja2 tokenizers numpy pillow av "huggingface_hub[cli]"

# 3. models (read their licenses first)
PATH=$PWD/.venv/bin:$PATH scripts/download-models.sh qwen /models     # and/or: glm, ds41

# 4. settings, then start
cp spark.env.example spark.env && $EDITOR spark.env
engine-rs/serve/start.sh qwen            # or: glm / glm3 / ds41 (run on the first node; ds41 needs the shard step of docs/ds41.md)

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

GitHub Actions 在 arm64 runner 上为版本 tag（以及手动触发）构建，并发布归档包。

## 测试系统与 GPU 频率

本仓库数据所用的三台 DGX Spark（GB10），2026-10-29 系统与固件更新后的版本（三台完全一致）：

| 组件 | 版本 |
| --- | --- |
| DGX OS（`dgx-release`） | 7.6.0 |
| 内核 | 7.0.0-1019-nvidia |
| NVIDIA 驱动 | 580.178.04 |
| CUDA toolkit | 13.0.3 |
| EC 嵌入式控制器固件 | 0x03000d07 |
| UEFI 固件 | 0x02009e0f / 0x00000519 |
| BIOS | 5.36_0ACUM028 |
| GPU VBIOS | 9A.0B.34.00.00 |
| ConnectX-7 固件 | 28.50.1002 |
| NVMe（Samsung MZALC4T0HBL1，自带 4 TB）固件 | NXHB202Q |

主机设置：`vm.compaction_proactiveness = 0`、`vm.watermark_boost_factor = 0`、关闭 swap、关闭 CPU 深度空闲（保留 WFI）。
2026-10-29 之前节点运行 DGX OS 7.5.0（内核 6.17、驱动 580.173.02、较旧的 EC / UEFI / ConnectX-7 固件），且一个遗留服务把
GPU 时钟锁在 2,200 MHz；本仓库此前的数据都是在这种状态下测的。更新后的系统上用 `bench/decode_bench.py` 连续满载测试（每种设置约 11 分钟，
每秒采样时钟，两种设置都未记录到降频），默认时钟与锁定 2,200 MHz 相差在 ±5% 以内，贪心时默认时钟快 1–2%。

更新后的系统上 DeepSeek-V4.1-Flash-1017（TP3）在默认时钟与锁定 2,200 MHz（`nvidia-smi -lgc 2200`）下的对比。两组使用同一
版本与设置（默认设置加可选的条件钳制 `DS41_ABLATE_CLAMP`），脚本与 [docs/ds41.zh-CN.md](docs/ds41.zh-CN.md#7-性能) 相同：
`bench/ds41_decode_bench.py`（贪心、关思考、每流 256 token、各流合计 tok/s、3 次中位）和 `bench/ds41_prefill_bench.py`
（每个长度同一文本 5 次，取中位）。

| Decode，tok/s | Prose | Structured | Code |
| --- | --- | --- | --- |
| 1 路，默认时钟 | **72.3** | **132.1** | **117.6** |
| 1 路，2,200 MHz | 70.7 | 129.1 | 112.4 |
| 2 路，默认时钟 | **102.2** | **179.5** | **152.3** |
| 2 路，2,200 MHz | 102.0 | 175.5 | 150.3 |
| 4 路，默认时钟 | **130.0** | **220.0** | **175.4** |
| 4 路，2,200 MHz | 122.4 | 214.5 | 171.7 |

| Prefill | 4K | 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- |
| 默认时钟，tok/s | **4,749** | **5,467** | **5,565** | **5,622** | **5,623** |
| 2,200 MHz，tok/s | 4,602 | 5,258 | 5,385 | 5,425 | 5,425 |
| TTFT，默认时钟 | 0.94 s | 1.58 s | 3.02 s | 5.91 s | 11.73 s |

负载下默认时钟的 GPU 频率为 2,430–2,530 MHz（各节点中位），最高 84–89 °C；锁频时为 2,184–2,190 MHz，最高 75–84 °C。频率
高约 12%，decode 只快 2–4%（decode 受内存带宽限制），prefill 快约 3.5%。系统更新本身的影响更大：同样锁 2,200 MHz，单路
decode 从旧系统的 66.4 / 119.3 / 105.2 tok/s（旧系统那组未开钳制，钳制对 decode 速度的影响小于 1%）提高到 70.7 / 129.1 / 112.4。若长时间满载温度过高，可用
`nvidia-smi -lgc 300,2400` 之类的上限以少量速度换温度；`nvidia-smi -rgc` 恢复默认。

## 其他用户实测

一套独立部署的 DeepSeek-V4.1-Flash-1017（3 台 DGX Spark，TP3），由部署者本人于 2026-10-29 用公开发布版、默认引擎设置测得，
脚本与 [docs/ds41.zh-CN.md](docs/ds41.zh-CN.md#7-性能) 相同（`bench/ds41_decode_bench.py`：贪心、关思考、每流 256 token、
3 次中位；`bench/ds41_prefill_bench.py`）。括号内为我们 2026-10-29 系统更新后的数据（[测试系统与 GPU 频率](#测试系统与-gpu-频率)，默认时钟）。

| 流数 | Prose | Structured | Code |
| --- | --- | --- | --- |
| 1 | 70.6（72.3） | 127.9（132.1） | 113.7（117.6） |
| 2 | 97.5（102.2） | 176.0（179.5） | 142.0（152.3） |
| 4 | 129.8（130.0） | 222.3（220.0） | 173.8（175.4） |

| 提示长度 | 4K | 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- |
| Prefill，tok/s | 4,659（4,749） | 5,445（5,467） | 5,583（5,565） | 5,666（5,622） | 5,681（5,623） |

64K TTFT 11.6 s（我们 11.7 s）。同一套机器第一次测得的数要低得多（prefill 2,648–3,810 tok/s，上下文 687,872），按下面的
[常见问题](#常见问题)逐项排查后达到上表。在我们自己更新系统之前，他们的数比我们高 4–8%，原因是我们节点的系统较旧且 GPU 时钟被锁定
（见上一节）。欢迎以 issue 提交你自己部署的实测结果。

## 常见问题

**最大上下文不到 1,048,576 token，而且每次重启都不一样。**
KV 池取的是加载完那一刻**最紧的那台**还剩下的内存，所以它反映的是某台机器当时有多少空闲，不是引擎的上限。从 rank 日志找出
是哪台：`grep -hE 'ds41-mem\].*margin|KV pool' /tmp/spark-ds41-rank*.log`，`ranks [a, b, c] granules` 中最小的就是瓶颈
rank，它的 `MemAvailable` 一行是当时的可用内存（我们 rank 0 约 16 GiB，仍能放下 220 万 token）。停掉引擎后每台的
`MemAvailable` 应在 100 GiB 以上；检查其他推理栈或容器、dashboard 与监控 exporter、`/dev/shm` 里的文件、大页和卡住的
（D 状态）进程。清掉之后三台都 `echo 3 > /proc/sys/vm/drop_caches` 再启动。不需要调低 `DS41_MEM_MARGIN_GB`、关闭草稿模型或
改变启动顺序。

**要不要等引擎就绪后再启动前端，免得它占用 KV 池的内存？**
不需要。视觉编码器由 rank 0 引擎在确定 KV 池大小之前加载，已经计入；前端（`ds41_server.py`）不加载 torch、不使用 GPU，
主机内存约 0.7 GB。按 `start.sh` 的方式两者同时启动，能省约 4 s。

**Prefill（或多路 decode）远低于参考值。**
DeepSeek 的张量并行通信走引擎自己的 RDMA 链路；prefill 传输的数据最多，缺一条链路时最先在这里显现。按顺序检查：
1. 链路：`grep '\[rdma_arn\]' /tmp/spark-ds41-rank*.log` 中每对节点都应列出**两条**链路（每根线缆的两个 PCIe function）；
   `no RoCE subnet shared` 表示缺地址。再看 `ibv_devinfo`（端口 `PORT_ACTIVE`）、每对节点的 `ib_write_bw`，以及 prefill
   期间每个设备的计数（`/sys/class/infiniband/*/ports/1/counters/port_xmit_data`）是否都在增长。
2. 版本与启动方式：三台二进制相同（`sha256sum`）、使用最新发布版、经 `start.sh ds41` 启动（它会设置 `DS41_MOE_OUT_MB`、
   `GLM53_RDMA_ARN_ROWS_PIPE` 和 `GLM53_COMM_STREAM_PRIO`）。
3. 每台的主机设置（[docs/deploy.zh-CN.md](docs/deploy.zh-CN.md) §1）：`vm.compaction_proactiveness = 0`、
   `vm.watermark_boost_factor = 0`、关闭 swap。我们有一台漏设时 prefill 会随机多出 0.4–1.5 s；各 rank 同步运行，
   一台卡住会拖慢全部。
4. 检查点与分片放在本地 NVMe 上（prefill 期间从中读取 Engram 行）。

**启动比约 16 s 慢。**
约 16 s 指引擎加载（约 10 s）加预热（约 4 s），用的是 DGX Spark 自带的 4 TB NVMe（我们用的就是这块盘，O_DIRECT 读分片约
11.7 GB/s）；停掉旧模型、`drop_caches` 和外层脚本不在其中。`grep -hE 'loaded in|waited .* for reads|warm-up in'
/tmp/spark-ds41-rank*.log` 能看出时间花在哪。如果读盘等待占大头，检查盘的 `current_link_speed`
（`/sys/class/nvme/nvme0/device/`，应为 32 GT/s）、温度（`nvme smart-log`）和直接读速
（`dd if=<分片> of=/dev/null bs=64M iflag=direct`）；`DS41_LOAD_PREFETCH`、`DS41_DIRECT_LOAD`、`DS41_WARM_DEFER` 保持默认。
如果加载很快但启动仍慢，多半是脚本在等每台 `MemAvailable` 回到 100 GiB（上一个模型还在释放内存），或在等某台慢节点。

**我测的 decode 与 README 差几个百分点。**
先看测法：README 主表用 `bench/decode_bench.py`（400 token、5 次中位），DeepSeek 页用 `bench/ds41_decode_bench.py`
（256 token、3 次中位）；两者都报时间窗内的总吞吐，而把每路速度相加在 2–4 路时会多 3–10%。投机解码与文本相关，2–4 路的数
每次运行会差约 5%。机器方面：GPU 时钟没有被锁定（`nvidia-smi -q -d CLOCK`；`nvidia-smi -rgc` 可恢复）、没有服务把 CUDA
线程钉到 A725 小核、三台的 OS / 内核 / 固件一致，散热（我们的 GB10 负载下约 2.4–2.5 GHz），以及 DGX OS / 固件版本（见[测试系统与 GPU 频率](#测试系统与-gpu-频率)）。

**节点卡死或自动重启。**
GB10 的 CPU 与 GPU 共用一块内存，默认没有 swap，内存耗尽的节点可能失去响应，节点上不要运行其他负载。内核 oops / panic
（例如没有 OOM、没有 Xid 的 paging request 错误）来自内核或固件，而不是用户态进程：让各节点的 DGX OS 与固件一致，并开启
kdump 以便下次留下调用栈。卡在 ConnectX-7 上的 D 状态进程（例如插拔线缆或模块后的 `mstflint`）只能靠重启清除，重启后
请重新检查链路。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/deploy.zh-CN.md](docs/deploy.zh-CN.md) | 硬件、安装、模型下载、两节点与三节点配置、启动 / 停止、故障排查 |
| [docs/ds41.zh-CN.md](docs/ds41.zh-CN.md) | 三节点运行 DeepSeek-V4.1-Flash-1017：网络、权重、分片、设置、API 说明、性能 |
| [docs/build.zh-CN.md](docs/build.zh-CN.md) | 从源码构建、依赖、CI |
| [docs/features.zh-CN.md](docs/features.zh-CN.md) | 引擎工作原理：GLM、DeepSeek 与 Qwen 路径、内核、投机解码、内存 |
| [docs/configuration.zh-CN.md](docs/configuration.zh-CN.md) | `spark.env`、profile、最常用的引擎与前端开关 |
| [docs/abliteration.zh-CN.md](docs/abliteration.zh-CN.md) | 可选的拒答移除（全部机制）：是什么、如何启用、风险 |
| [docs/benchmarks.zh-CN.md](docs/benchmarks.zh-CN.md) | 如何测量速度、参考数据 |

## 模型与许可证

本仓库只包含代码。模型需单独下载，并保留各自的许可证：

| 模型 | 来源 | 许可证 |
| --- | --- | --- |
| GLM-5.3-Flash EXL3 4 bpw | [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | 见模型卡 |
| GLM-5.3-Flash SPX1 V1.1（本项目自己的量化） | [fly9i/GLM-5.3-Flash-SPX1-V1.1](https://huggingface.co/fly9i/GLM-5.3-Flash-SPX1-V1.1) | 见模型卡 |
| GLM 的 DFlash2 草稿模型 | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | **CC BY-NC-ND 4.0（非商业）** |
| DeepSeek-V4.1-Flash-1017 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)（`1bae0e10`） | MIT（见模型卡） |
| Qwen3.8-Flash-Next EXL3 | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)（`4.05bpw_h6_ng6`） | Qwen 许可证（见模型卡） |

**GLM-5.3-Flash SPX1 V1.1**（[fly9i/GLM-5.3-Flash-SPX1-V1.1](https://huggingface.co/fly9i/GLM-5.3-Flash-SPX1-V1.1)，156 GB）是本项目为这个引擎做的 GLM-5.3-Flash 量化：路由专家用
EXL3 trellis 编码、按专家混合位宽（3–6 bit，平均 4.0 bpw），其余线性层用 6 bit 的 SPX1 格式（INT6 + Hadamard 旋转，GPTQ）。
对官方 FP8 模型的评测 KL 为 0.0171（统一 4 bpw 的 EXL3 检查点为 0.0231）。把 `GLM_MODEL` 指向它即可，引擎会识别 SPX1 张量
（`*.spx_q`）并替换原张量加载。（[docs/deploy.zh-CN.md](docs/deploy.zh-CN.md) 里 GLM TP3 的数据是用其后续未公开的 V1.2 版本测的。）DFlash2 草稿模型与
EXL3 检查点一样另行下载。

构建或运行时使用的第三方软件列于 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 许可证

[MIT](LICENSE)。
