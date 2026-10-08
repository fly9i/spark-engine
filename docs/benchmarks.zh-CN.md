# 基准测试

## 工具

| 脚本 | 测量内容 |
| --- | --- |
| `bench/decode_bench.py [prose\|structured] [repeats] [max_tokens]` | 1、2、3、4 个并发流下的 decode 速度：贪心，关闭思考，32 token 预热，输出 400 个 token；单流速度为 `(completion_tokens - 1) / (t_last - t_first)`，总速度 = 全部 decode token / 墙钟时间窗口；取 `repeats` 次运行的中位数 |
| `bench/prefill_bench.py TOKENS [SEED]` | 对约 `TOKENS` 个 token 的合成提示词做 prefill（随机单词，因此不同 seed 之间没有共享前缀）；输出引擎的 prefill 时间和 tok/s |
| `bench/text_check.py LABEL [BASELINE]` | 20 个固定提示词，每个贪心生成 256 个 token，逐个运行：比较两个构建或两组设置（回复是否一致、首个分歧点） |

三个脚本都连接 `SPARK_URL`（默认 `http://127.0.0.1:8888`）。decode 测试使用两类提示词：

- **散文（prose）**：一段长篇技术解释，典型的聊天输出；草稿接受率较低；
- **结构化（structured）**：从 1 数到 200；几乎每个草稿 token 都被接受，体现投机解码的上限。

投机解码使 decode 速度依赖于生成的文本。舍入方式不同的两个构建（例如精度等级为 L1 的内核改动）会生成略有不同的文本，
在单个提示词上可能相差几个百分点，但并不说明哪一个更快。比较内核时应在固定行数下比较每轮耗时，而不能只看端到端 tok/s。

## 结果

于 2026-10-08 使用发布版二进制和默认设置测得（`spark.env` 默认值，`GLM53_PCACHE=0` 以测量冷 prefill）。
表中为所有流的 decode 总吞吐；括号内为单流速度。

### GLM-5.3-Flash，2× DGX Spark（TP2）

| 流数 | 散文 | 结构化 | TTFT（散文） |
| --- | --- | --- | --- |
| 1 | 52.9 tok/s | 92.5 tok/s | 372 ms |
| 2 | 66.5（每流 33.5） | 119.1（每流 76.2） | 784 ms |
| 3 | 73.3（每流 25.7） | 158.0（每流 54.5） | 926 ms |
| 4 | 82.9（每流 21.9） | 168.0（每流 44.3） | 1137 ms |

| 提示词 | Prefill 时间 | Prefill 速度 |
| --- | --- | --- |
| 9,561 token | 6.13 s | 1,561 tok/s |
| 9,551 token | 6.27 s | 1,522 tok/s |
| 54,686 token | 35.17 s | 1,555 tok/s |
| 54,880 token | 35.22 s | 1,558 tok/s |

### Qwen3.8-Flash-Next，1× DGX Spark

| 流数 | 散文 | 结构化 | TTFT（散文） |
| --- | --- | --- | --- |
| 1 | 64.1 tok/s | 149.8 tok/s | 153 ms |
| 2 | 85.5（每流 44.1） | 192.9（每流 110.2） | 266 ms |
| 3 | 105.0（每流 36.0） | 232.6（每流 85.5） | 378 ms |
| 4 | 119.5（每流 31.7） | 217.4（每流 68.9） | 492 ms |

| 提示词 | Prefill 时间 | Prefill 速度 |
| --- | --- | --- |
| 13,449 token | 7.16 s | 1,879 tok/s |
| 13,456 token | 7.00 s | 1,922 tok/s |
| 77,204 token | 41.20 s | 1,874 tok/s |
| 77,274 token | 41.17 s | 1,877 tok/s |

TTFT 随流数增加而增长，因为新请求是在正在运行的请求的 decode 轮次之间被接纳的。
多轮对话的后续轮次会复用上一轮的状态：在 16K token 的对话中，Qwen 下一轮的 TTFT 约为 0.3 s，而无需完整 prefill
（提示词检查点）。

### 复现

```bash
engine-rs/serve/start.sh glm                 # or qwen
SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py prose 3
SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py structured 3
for s in 11 12; do bench/prefill_bench.py 8000 $s; done
for s in 21 22; do bench/prefill_bench.py 46000 $s; done
```

## 方法说明

- 每个模型测量前都重启了服务，并关闭持久化前缀缓存（`GLM53_PCACHE=0`），以保证 prefill 数据为冷启动结果。
- `decode_bench.py` 的每个并发级别运行 3 次；表中为中位数。
- Prefill：每种长度使用两个不同 seed 的提示词；表中两者都列出。
- GLM-5.3-Flash 运行在通过两个 ConnectX-7 端口互连（2 × 200 Gb/s）的两台 DGX Spark 上；Qwen3.8-Flash-Next 运行在一台上。
