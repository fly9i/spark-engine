# 部署

## 1. 硬件与系统

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| 机器 | 2× DGX Spark（GB10），每台 128 GB 统一内存 | 1× DGX Spark |
| 互连 | 两台 Spark 的两个 ConnectX-7 QSFP 端口直连（2 × 200 Gb/s RoCE） | — |
| 磁盘 | 每个节点约 170 GB（checkpoint 164 GB + 草稿模型 2.2 GB），两个节点路径相同 | 约 101 GB（含一个运行时从 NVMe 读取的 39 GB n-gram 嵌入表） |
| 软件 | DGX OS 7 / Ubuntu 24.04 aarch64，支持 CUDA 13 的 NVIDIA 驱动，Python 3.12 | 同左 |

两个模型都会占用 128 GB 中的大部分：每台机器同一时间只运行一个模型，GPU 上不要运行其他任务。引擎在启动时根据空闲内存确定
KV 缓存大小；服务期间请保持至少 8 GB `MemAvailable`（GB10 的内存与 CPU 共享，耗尽可能导致机器卡死）。Qwen 启动脚本内置内存保护（`QWEN_MEMGUARD_GIB`，默认 8）。

推荐的主机设置（两个节点）：关闭主动内存规整（proactive compaction）。否则它会成批迁移内存页，在统一内存上导致 decode
一次变慢数秒：

```bash
echo 'vm.compaction_proactiveness = 0' | sudo tee /etc/sysctl.d/90-spark-engine.conf && sudo sysctl --system
```

## 2. GLM 的网络配置（双节点）

将每台 ConnectX-7 的 0 号端口连到另一台 Spark 的 0 号端口（如果有两根线缆，两个 QSFP 端口都连上）。为每个节点的两个接口配置点对点地址，例如：

| 接口 | 节点 A（rank 0，主节点） | 节点 B（rank 1，工作节点） |
| --- | --- | --- |
| `enp1s0f0np0`（RDMA 设备 `rocep1s0f0`） | 10.0.1.2/30 | 10.0.1.1/30 |
| `enP2p1s0f0np0`（RDMA 设备 `roceP2p1s0f0`） | 10.0.2.2/30 | 10.0.2.1/30 |

用 `ibv_devinfo -d rocep1s0f0`（`PORT_ACTIVE`，链路层为 Ethernet）和 `ping 10.0.1.1` 检查。在 DGX OS 上 RoCE v2 使用 GID 索引 3；
NCCL 和引擎自带的 RDMA all-reduce 会同时使用两个 RDMA 设备。

节点 A 通过 ssh 在节点 B 上启动 rank 1：为运行服务的用户配置从 A 到 B 的密钥登录（示例使用 `root@10.0.1.1`）。在**两个节点的相同路径下**安装发布包、Python 环境和模型；如果两个节点的引擎二进制不一致，启动脚本会拒绝启动。

## 3. 安装

在每个节点上执行：

```bash
sudo mkdir -p /opt/spark-engine && sudo chown $USER /opt/spark-engine && cd /opt/spark-engine
# download the archive from https://github.com/fly9i/spark-engine/releases, then
tar xzf spark-engine-<version>-linux-aarch64-cu130-sm121.tar.gz --strip-components=1

python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install jinja2 tokenizers numpy pillow av "huggingface_hub[cli]"
```

检查二进制能否找到所需的库：`source scripts/env.sh && ldd bin/spark-engine | grep "not found"` 应无任何输出。

## 4. 模型

```bash
PATH=/opt/spark-engine/.venv/bin:$PATH /opt/spark-engine/scripts/download-models.sh glm /models    # both nodes
PATH=/opt/spark-engine/.venv/bin:$PATH /opt/spark-engine/scripts/download-models.sh qwen /models
```

脚本固定了引擎测试时使用的版本（revision）。DFlash2 草稿模型的许可证为 CC BY-NC-ND 4.0（仅限非商业用途）。

## 5. 设置

```bash
cp spark.env.example spark.env
```

编辑 `spark.env`（两个节点内容相同）：路径、工作节点的 ssh 目标、互连网络地址和 NCCL 接口名。每项设置在文件内和
[configuration.zh-CN.md](configuration.zh-CN.md) 中都有说明。

## 6. 启动、使用、停止

```bash
engine-rs/serve/start.sh glm      # on node A; about a minute (loading 82 GB per node)
engine-rs/serve/start.sh qwen     # one node; about 50 s
engine-rs/serve/start.sh status   # glm | qwen | none
engine-rs/serve/start.sh stop
```

`start.sh` 会先停止另一个模型，并等待其内存释放。服务在所有网络接口上监听 `SPARK_PORT`（默认 8888）：

```bash
curl http://<node-A>:8888/v1/models
curl http://<node-A>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about the sea."}],
  "max_tokens": 300, "stream": false}'
```

思考：两个模型默认都会思考。可在单个请求中用 `"chat_template_kwargs": {"enable_thinking": false}` 关闭；
Qwen 还接受 `"reasoning_effort": "none" | "low" | "medium" | "high"`。图像：在消息内容中放入 `{"type": "image_url",
"image_url": {"url": "https://... or data:image/png;base64,..."}}`；视频：`"type": "video_url"`。

日志：`/tmp/spark-glm-rank0.log`、`/tmp/spark-glm-rank1.log`（在节点 B 上）、`/tmp/spark-glm-front.log`、
`/tmp/spark-qwen-engine.log`、`/tmp/spark-qwen-front.log`。Prometheus 指标：`GET /metrics`。

## 7. 故障排查

| 现象 | 原因 / 解决方法 |
| --- | --- |
| `engine binary differs between the nodes` | 在两个节点上安装同一个发布版本（相同的 `SPARK_BIN` 路径）。 |
| `rank 0 exited` 并伴有 NCCL 错误 | 检查互连网络地址、`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`ibv_devinfo` 端口状态，以及 rank 1 是否已启动（节点 B 上的 `/tmp/spark-glm-rank1.log`）。 |
| 启动时内存不足 | 有其他进程占用 GPU 内存，或另一个模型仍在释放内存：执行 `start.sh stop`，等待后再启动。调低 `GLM_KV_TOKENS`（或 `QWEN_KV_TOKENS`）以缩小 KV 池。 |
| 找不到 `libexllamav3_ext.so` / `libtorch_cuda.so` | 在 `SPARK_PYTHON` 指向装有 torch 2.13 cu130 的环境时 source `scripts/env.sh`。 |
| 前端返回 503 | 引擎仍在加载；等待引擎日志中出现 `ready`。 |
| 图像/视频请求失败 | Python 环境中缺少 `pillow` / `av`，或视觉功能已关闭（`QWEN_VISION=0`、`GLM53_VISION=0`）。 |
