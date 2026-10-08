# Deployment

## 1. Hardware and system

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- |
| Machines | 2× DGX Spark (GB10), each 128 GB unified memory | 1× DGX Spark |
| Interconnect | the two ConnectX-7 QSFP ports cabled directly between the Sparks (2 × 200 Gb/s RoCE) | — |
| Disk | ~170 GB per node (checkpoint 164 GB + drafter 2.2 GB), same path on both nodes | ~101 GB (includes a 39 GB n-gram embedding table read from NVMe at run time) |
| Software | DGX OS 7 / Ubuntu 24.04 aarch64, NVIDIA driver with CUDA 13 support, Python 3.12 | same |

Both models use most of the 128 GB: run one model at a time per machine, and nothing else on the GPU. The engine sizes its
KV cache from free memory at start-up; keep at least 8 GB `MemAvailable` while serving (GB10 memory is shared with the
CPU, and exhausting it can hang the machine). The Qwen launcher includes a memory guard (`QWEN_MEMGUARD_GIB`, default 8).

Recommended host setting (both nodes): disable proactive memory compaction, which otherwise migrates pages in bursts and
slows decoding for seconds at a time on unified memory:

```bash
echo 'vm.compaction_proactiveness = 0' | sudo tee /etc/sysctl.d/90-spark-engine.conf && sudo sysctl --system
```

## 2. Networking for GLM (two nodes)

Connect port 0 of each ConnectX-7 to the other Spark's port 0 (both QSFP ports if you have two cables). Give the two
interfaces of each node a point-to-point address, for example:

| Interface | Node A (rank 0, head) | Node B (rank 1, worker) |
| --- | --- | --- |
| `enp1s0f0np0` (RDMA device `rocep1s0f0`) | 10.0.1.2/30 | 10.0.1.1/30 |
| `enP2p1s0f0np0` (RDMA device `roceP2p1s0f0`) | 10.0.2.2/30 | 10.0.2.1/30 |

Check with `ibv_devinfo -d rocep1s0f0` (`PORT_ACTIVE`, link layer Ethernet) and `ping 10.0.1.1`. RoCE v2 uses GID index 3
on DGX OS; NCCL and the engine's RDMA all-reduce use both RDMA devices.

Node A starts rank 1 on node B over ssh: set up key-based login from A to B for the user that runs the server (the
example uses `root@10.0.1.1`). Install the release, the Python environment and the models **at the same paths on both
nodes**; the launcher refuses to start if the two engine binaries differ.

## 3. Install

On every node:

```bash
sudo mkdir -p /opt/spark-engine && sudo chown $USER /opt/spark-engine && cd /opt/spark-engine
# download the archive from https://github.com/fly9i/spark-engine/releases, then
tar xzf spark-engine-<version>-linux-aarch64-cu130-sm121.tar.gz --strip-components=1

python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install jinja2 tokenizers numpy pillow av "huggingface_hub[cli]"
```

Check that the binary resolves its libraries: `source scripts/env.sh && ldd bin/spark-engine | grep "not found"` prints
nothing.

## 4. Models

```bash
PATH=/opt/spark-engine/.venv/bin:$PATH /opt/spark-engine/scripts/download-models.sh glm /models    # both nodes
PATH=/opt/spark-engine/.venv/bin:$PATH /opt/spark-engine/scripts/download-models.sh qwen /models
```

The script pins the revisions the engine was tested with. The DFlash2 drafter is licensed CC BY-NC-ND 4.0
(non-commercial use only).

## 5. Settings

```bash
cp spark.env.example spark.env
```

Edit `spark.env` (same content on both nodes): paths, the worker's ssh target, fabric address and NCCL interface names.
Every setting is explained in the file and in [configuration.md](configuration.md).

## 6. Start, use, stop

```bash
engine-rs/serve/start.sh glm      # on node A; about a minute (loading 82 GB per node)
engine-rs/serve/start.sh qwen     # one node; about 50 s
engine-rs/serve/start.sh status   # glm | qwen | none
engine-rs/serve/start.sh stop
```

`start.sh` stops the other model first and waits for its memory to come back. The server listens on `SPARK_PORT`
(default 8888) on all interfaces:

```bash
curl http://<node-A>:8888/v1/models
curl http://<node-A>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about the sea."}],
  "max_tokens": 300, "stream": false}'
```

Thinking: both models think by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`;
Qwen also accepts `"reasoning_effort": "none" | "low" | "medium" | "high"`. Images: put `{"type": "image_url",
"image_url": {"url": "https://... or data:image/png;base64,..."}}` in the message content; videos: `"type": "video_url"`.

Logs: `/tmp/spark-glm-rank0.log`, `/tmp/spark-glm-rank1.log` (on node B), `/tmp/spark-glm-front.log`,
`/tmp/spark-qwen-engine.log`, `/tmp/spark-qwen-front.log`. Prometheus metrics: `GET /metrics`.

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `engine binary differs between the nodes` | Install the same release on both nodes (same `SPARK_BIN` path). |
| `rank 0 exited` with NCCL errors | Check the fabric addresses, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `ibv_devinfo` port state, and that rank 1 started (`/tmp/spark-glm-rank1.log` on node B). |
| Out of memory at start | Another process holds GPU memory, or the other model is still releasing memory: `start.sh stop`, wait, start again. Lower `GLM_KV_TOKENS` (or `QWEN_KV_TOKENS`) to shrink the KV pool. |
| `libexllamav3_ext.so` / `libtorch_cuda.so` not found | Source `scripts/env.sh` with `SPARK_PYTHON` pointing at the environment that has torch 2.13 cu130. |
| Front end returns 503 | The engine is still loading; wait for `ready` in the engine log. |
| Image/video requests fail | `pillow` / `av` missing in the Python environment, or vision disabled (`QWEN_VISION=0`, `GLM53_VISION=0`). |
