# 配置

设置分为三层：

1. **`spark.env`** – 部署相关：路径、节点、网络、端口（本页）。
2. **引擎开关** – 引擎读取的环境变量。GLM 的调优取值位于
   `engine-rs/profiles/glm.env`（[完整说明](glm-switches.zh-CN.md)）；Qwen 和 DeepSeek 的默认值编译在程序中
   （[Qwen 说明](qwen-switches.zh-CN.md)、[DeepSeek 设置](ds41.zh-CN.md#5-设置)）。任何一项都可以在 `spark.env` 中覆盖。
3. **前端选项** – `openai_server.py` / `qwen_server.py` / `ds41_server.py` 的命令行参数，由启动脚本设置。

## spark.env

| 设置 | 示例 / 默认值 | 含义 |
| --- | --- | --- |
| `SPARK_HOME` | `/opt/spark-engine` | 发布包（或仓库）目录；在每个节点上路径必须相同。 |
| `SPARK_PYTHON` | `$SPARK_HOME/.venv/bin/python` | 装有 PyTorch 2.13（cu130）和前端依赖包的 Python。`scripts/env.sh` 据此推导库路径。 |
| `SPARK_BIN` | `$SPARK_HOME/bin/spark-engine` | 引擎二进制（发布版），或 `engine-rs/target/release/spark-engine`（源码构建）。 |
| `SPARK_PORT` | `8888` | OpenAI 兼容 API 的 HTTP 端口（所有模型共用；同一时间只运行一个模型）。 |
| `GLM_MODEL` | `/models/GLM-5.3-Flash-tr3-4bpw` | GLM-5.3-Flash EXL3 checkpoint 目录（两个节点）。 |
| `GLM_DRAFTER` | `/models/GLM-5.3-Flash-DFlash2` | DFlash2 草稿模型目录（两个节点）。 |
| `SPARK_WORKERS` | `"root@10.0.1.1 root@10.0.3.2"` | rank 1 和 rank 2 的 ssh 目标（需从 rank 0 所在节点以密钥登录）；两节点运行时用第一个。取代 `GLM_WORKER`（仍然可用）。 |
| `SPARK_MASTER_ADDRS` | `"10.0.1.2 10.0.1.2 10.0.3.1"` | 从 rank 0、1、2 看到的 rank 0 地址（每个工作节点经各自的 RoCE 链路）。取代 `GLM_MASTER_ADDR`（所有 rank 用同一地址，仍然可用）。 |
| `SPARK_TP3_IFNAME` | — | 三节点：三台互相可达的接口（仅用于 NCCL 引导；数据走 RDMA）。 |
| `GLM_WORLD` | `2` | GLM 的节点数：2 或 3（`start.sh glm3` 总是用 3）。 |
| `GLM_MASTER_PORT` | `29931` | 汇合存储（rendezvous store）的 TCP 端口。 |
| `GLM_PROFILE` | `$SPARK_HOME/engine-rs/profiles/glm.env` | GLM 的引擎开关 profile（TP2 与 TP3 通用；各 world size 的传输设置在 `profiles/tp<world>-rdma.env`）。 |
| `GLM_MODEL_NAME` | `GLM-5.3-Flash` | API 报告的模型名。 |
| `GLM_KV_TOKENS` | `1048576` | 以 token 计的 KV 缓存预算，也是最长上下文；调低可留出更多空闲内存。 |
| `GLM_KV_GRANULE` | `16384` | 每个序列的 KV 分配粒度（token）。 |
| `GLM_EXTRA_ENV` | — | 以空格分隔的 `NAME=value` 覆盖项，最后应用到每个 rank（用于诊断、实验）。 |
| `NCCL_IB_HCA` | `rocep1s0f0,roceP2p1s0f0` | 两节点：NCCL 使用的 RDMA 设备（DGX Spark 上 ConnectX-7 的两个功能）。 |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | NCCL 引导流量使用的接口。 |
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 GID 索引。 |
| `GLM53_RDMA_AR_DEV`、`GLM53_RDMA_AR_DEV1`、`GLM53_RDMA_AR_GID` | `rocep1s0f0`、`roceP2p1s0f0`、`3` | 引擎自带 RDMA all-reduce 使用的设备和 GID（默认值与 DGX Spark 匹配）。 |
| `DS41_CKPT`、`DS41_SHARDS`、`DS41_ENGRAM_HASH` | — | DeepSeek-V4.1-Flash 检查点、各 rank 分片根目录和 Engram 哈希文件（每个节点，路径相同；[ds41.zh-CN.md](ds41.zh-CN.md)）。 |
| `DS41_MODEL_NAME` | `DeepSeek-V4.1-Flash-1017` | API 报告的模型名。 |
| `DS41_*` 服务设置、`DS41_EXTRA_ENV` | 见 [ds41.zh-CN.md](ds41.zh-CN.md#5-设置) | 并发、投机、前缀缓存、视觉、内存；对每个 rank 生效的额外设置。 |
| `QWEN_MODEL` | `/models/Qwen3.8-Flash-Next-exl3` | Qwen3.8-Flash-Next EXL3 checkpoint 目录。 |
| `QWEN_ASSETS` | `$SPARK_HOME/assets/qwen38` | 包含 `draft_vocab_65536.json`（MTP 草稿头词表）的目录。 |
| `GLM53_ABLIT`、`GLM53_ABLIT_DIR`、`GLM53_ABLIT_LAYERS` | 关、—、`15-44` | GLM o_proj 移植（[abliteration.zh-CN.md](abliteration.zh-CN.md)）。 |
| `SPARK_ABLATE_CLAMP`、`SPARK_ABLATE_CLAMP_MIN`（DeepSeek：`DS41_ABLATE_CLAMP`、`DS41_ABLATE_MIN`） | 关 | 条件钳制消融（[abliteration.zh-CN.md](abliteration.zh-CN.md)）。 |
| `SPARK_ABLATE`、`SPARK_ABLATE_MODE` | 关、`single` | 方向消融，两个模型通用（[abliteration.zh-CN.md](abliteration.zh-CN.md)）；`single` / `per-layer` / `subspace:a-b:k`。 |
| `QWEN_PCACHE`、`QWEN_PCACHE_DIR` | 关、`/tmp/qwen38-prefix-cache` | Qwen NVMe 持久前缀缓存（镜像 `GLM53_PCACHE`）：prompt 命中盘上已缓存的边界检查点时从盘恢复，省去重新 prefill。 |
| `QWEN_VISION` | `1` | 加载视觉塔（约 0.5 GB）并接受图像/视频；`0` = 仅文本。 |
| `QWEN_MEMGUARD_GIB` | `8` | 当 `MemAvailable` 低于该 GiB 数时，Qwen 启动脚本会停止引擎。 |
| `SPARK_COMPACT` | `1` | GLM 或 DeepSeek 启动后执行一次主机内存规整（每个节点）（避免统一内存上的页迁移卡顿）。 |
| `GLM53_HOST_REGISTER` | `1` | pinned 主机暂存缓冲使用匿名页 + `cudaHostRegister`（内核内存规整从不隔离）；`0` = `cudaHostAlloc`。两个模型通用。 |
| `SPARK_MIN_FREE_GIB` | `100` | 切换模型时，`start.sh` 会等到空闲内存达到该值。 |

## 最常用的引擎开关

| 变量 | 模型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `GLM53_SERVE_MAX_SEQS` | GLM | `4` | 并发序列数（更多请求会排队）。序列越多，总吞吐和内存占用越高。 |
| `GLM53_SERVE_MAX_STORES` | GLM | `4` | 序列存储数（每个存储保存一个序列的 KV 和状态；空闲存储保留其前缀以供复用）。 |
| `GLM53_PREFILL_CHUNK` | GLM | `4096` | 每个 prefill 步处理的提示词 token 数（越大 prefill 越快，临时内存占用越多）。 |
| `GLM53_VISION` | GLM | `1` | 通过视觉塔处理图像/视频输入。 |
| `GLM53_PCACHE`、`GLM53_PCACHE_DIR` | GLM | `1`、`$SPARK_HOME/var/prefix-cache` | NVMe 上的持久化前缀缓存：之前见过的长提示词直接恢复，而不是重新 prefill。 |
| `GLM53_SERVE_LOG` | GLM | `0` | 在 rank 日志中输出每轮的日志行（行数、耗时）。 |
| `QWEN_SERVE_MAX_SEQS` | Qwen | `8` | 并发序列数（1–8）。 |
| `QWEN_SERVE_STORES` | Qwen | `8` | 序列存储数（至少为 `QWEN_SERVE_MAX_SEQS`）；空闲存储保留其前缀以供复用。 |
| `QWEN_KV_TOKENS` | Qwen | `1048576` | 以 token 计的 KV 池大小（所有存储以 16K token 为粒度共享）。 |
| `QWEN_SERVE_RESERVE` | Qwen | `32768` | 每个请求预先预留的 token 数；区间在 decode 过程中增长。 |
| `QWEN_SERVE_CKPT`、`QWEN_SERVE_CKPT_MIN` | Qwen | 开启、`1024` | 对至少达到该 token 数的提示词，在最后一条消息边界处保存提示词检查点（多轮后续请求更快）。 |
| `QWEN_MTP` | Qwen | 开启 | 原生 MTP 草稿（`0` = 普通解码）。 |

全部开关及其取值和精度等级见 [glm-switches.zh-CN.md](glm-switches.zh-CN.md) 和 [qwen-switches.zh-CN.md](qwen-switches.zh-CN.md)。
其中大多数用于在参考路径和优化内核之间切换，保留下来是为了做 A/B 测试；发布的取值是实测最优值。

## 前端选项

由 `start-glm.sh` / `start-qwen.sh` 设置；如需修改请编辑这些脚本。

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--port`、`--host` | `8888`、`0.0.0.0` | 监听地址。 |
| `--model-name` | `GLM-5.3-Flash-EXL3` / `Qwen3.8-Flash-Next` | `/v1/models` 返回的模型 id（请求中可使用任意名称）。 |
| `--max-model-len` | KV 预算 | 可接受的最长提示词 + 补全长度。 |
| `--max-tokens` | `8192` | 请求未指定时使用的 `max_tokens`。 |
| `--vision on/off` | 开启 | 接受图像/视频内容（必须与引擎设置一致）。 |
| `--max-image-tokens` | GLM 8000 / Qwen 16384 | 每张图像的 token 上限（更大的图像会被缩放）。 |
| `--max-video-tokens`、`--video-fps`、`--video-max-frames` | GLM 65536 / Qwen 12288、`2`、GLM 2048 / Qwen 768 | 视频采样和 token 预算。 |
| `--allowed-media-domains`、`--media-deny-private`、`--allowed-local-media-path` | 任意、关闭、无 | 媒体 URL 的限制。 |
| `--media-timeout`、`--media-max-bytes`、`--video-max-bytes` | 30 s、64 MB、2 GB | 媒体下载限制。 |

请求参数：`temperature`（0 = 贪心；采样为精确投机采样）、`seed`、`max_tokens` /
`max_completion_tokens`、`stop`、`stop_token_ids`、`stream`、`stream_options.include_usage`、`tools` / `tool_choice`、
`chat_template_kwargs`（例如 `{"enable_thinking": false}`）、`reasoning_effort`（Qwen）。`top_p`、`top_k` 和各类惩罚参数
会被接受，但不生效。
