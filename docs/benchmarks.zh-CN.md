# 基准测试

## 工具

| 脚本 | 测量内容 |
| --- | --- |
| `bench/decode_bench.py [prose\|structured] [repeats] [max_tokens] [--temperature T] [--thinking] [--streams 1,2,3,4]` | 1~4 路并发下的 decode 速度（DGX Spark 社区常用方法，见[解码测速方法](#解码测速方法deepseek-与-qwen2026-10-29v038)）：默认贪心、关闭思考，32 token 预热，输出 400 个 token，多路时每路提示前加 `Request ID` 行；单流速度为 `(completion_tokens - 1) / (t_last - t_first)`，总速度 = 全部 decode token / 墙钟时间窗口；取 `repeats` 次运行的中位数 |
| `bench/prefill_bench.py TOKENS [SEED]` | 对约 `TOKENS` 个 token 的合成提示词做 prefill（随机单词，因此不同 seed 之间没有共享前缀）；输出引擎的 prefill 时间和 tok/s |
| `bench/ds41_decode_bench.py [--url URL] [--types prose,structured,code] [--conc 1,2,4]` | DeepSeek-V4.1-Flash 的数据（[ds41.zh-CN.md](ds41.zh-CN.md#7-性能)）：贪心，关闭思考，32 token 预热，每流 256 个 token，prose / structured / code 提示词 |
| `bench/ds41_prefill_bench.py [repeats] [sizes]` | DeepSeek-V4.1-Flash 的 prefill 数据：4K–64K 的真实文本提示，每个唯一前缀，流式；tok/s = 提示 token 数 / TTFT，取 `repeats` 次的中位 |
| `bench/text_check.py LABEL [BASELINE]` | 20 个固定提示词，每个贪心生成 256 个 token，逐个运行：比较两个构建或两组设置（回复是否一致、首个分歧点） |

`decode_bench.py`、`prefill_bench.py` 和 `text_check.py` 连接 `SPARK_URL`（默认 `http://127.0.0.1:8888`）。decode 测试使用两类提示词：

- **散文（prose）**：一段长篇技术解释，典型的聊天输出；草稿接受率较低；
- **结构化（structured）**：从 1 数到 200；几乎每个草稿 token 都被接受，体现投机解码的上限。

投机解码使 decode 速度依赖于生成的文本。舍入方式不同的两个构建（例如精度等级为 L1 的内核改动）会生成略有不同的文本，
在单个提示词上可能相差几个百分点，但并不说明哪一个更快。比较内核时应在固定行数下比较每轮耗时，而不能只看端到端 tok/s。

## 结果

GLM-5.3-Flash（decode 与 prefill）：2026-10-08 在 DGX OS 7.5.0、GPU 时钟锁定 2,200 MHz 下，使用发布版二进制和默认设置测得（`spark.env` 默认值，`GLM53_PCACHE=0` 以测量冷 prefill），贪心、关思考，
多路用旧的句尾 `(stream i/n)`，3 次中位数。DeepSeek 与 Qwen 的 decode 以及 Qwen 的 prefill：2026-10-29，见[解码测速方法](#解码测速方法deepseek-与-qwen2026-10-29v038)。
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

### 解码测速方法（DeepSeek 与 Qwen，2026-10-29，v0.3.8）

与 DGX Spark 社区常用的方法一致：用 `bench/decode_bench.py` 测运行中的服务；提示为 *prose*（逐步讲解哈希表原理）与 *structured*（从 1 数到 200）；
每个请求输出 400 token（本次每条回复都跑满 400）；在 README「测试系统与 GPU 频率」一节所述的更新后系统上、默认 GPU 时钟下测得，每秒采样时钟，未记录到降频；1~4 路并发，多路时每路提示前加一行 `Request ID: decode-<i>`（单路为原提示）；32 token 预热；
每档 5 次取中位数。两种采样方式在每列标注：**贪心、关思考**（温度 0）与 **T 1、开思考**（温度 1、top-p 1，思考 token 计入）。每路速度 =
（completion token − 1）÷（末 token − 首 token）；聚合 = 全部解码 token ÷ 从最早一路首 token 到最晚一路末 token 的时间窗。因为有投机解码，
速度随文本变化，换提示会得到不同数字。

### DeepSeek-V4.1-Flash，3× DGX Spark（TP3）

解码（聚合 tok/s；括号内为每路；2026-10-30，v0.3.9）。服务同时解码 4 个请求（`DS41_SERVE_STORES=4`）。更多数据（代码提示、预填）见 [ds41.zh-CN.md](ds41.zh-CN.md#7-性能)。

| 流数 | Prose，贪心、关思考 | Structured，贪心、关思考 | Prose，T 1、开思考 | Structured，T 1、开思考 |
| --- | --- | --- | --- | --- |
| 1 | 67.4 tok/s（TTFT 171 ms） | 126.6 tok/s（TTFT 162 ms） | 57.2 tok/s（TTFT 245 ms） | 109.2 tok/s（TTFT 218 ms） |
| 2 | 92.8（每路 49.2，TTFT 287 ms） | 181.9（每路 93.5，TTFT 281 ms） | 82.2（每路 41.8，TTFT 370 ms） | 153.3（每路 81.6，TTFT 356 ms） |
| 3 | 109.3（每路 37.4，TTFT 395 ms） | 194.8（每路 67.7，TTFT 381 ms） | 95.0（每路 33.5，TTFT 504 ms） | 153.5（每路 59.7，TTFT 485 ms） |
| 4 | 124.7（每路 32.4，TTFT 495 ms） | 226.0（每路 59.9，TTFT 493 ms） | 105.5（每路 27.8，TTFT 636 ms） | 180.9（每路 51.5，TTFT 610 ms） |

### Qwen3.8-Flash-Next，1× DGX Spark

解码（聚合 tok/s；括号内为每路）：

| 流数 | Prose，贪心、关思考 | Structured，贪心、关思考 | Prose，T 1、开思考 | Structured，T 1、开思考 |
| --- | --- | --- | --- | --- |
| 1 | 68.4 tok/s（TTFT 136 ms） | 157.8 tok/s（TTFT 133 ms） | 51.2 tok/s（TTFT 203 ms） | 122.7 tok/s（TTFT 204 ms） |
| 2 | 98.0（每路 50.4，TTFT 242 ms） | 230.7（每路 119.7，TTFT 241 ms） | 74.3（每路 40.8，TTFT 330 ms） | 78.1（每路 40.1，TTFT 328 ms） |
| 3 | 122.4（每路 43.0，TTFT 342 ms） | 280.1（每路 99.2，TTFT 343 ms） | 90.9（每路 32.8，TTFT 467 ms） | 100.9（每路 39.0，TTFT 476 ms） |
| 4 | 139.2（每路 36.8，TTFT 438 ms） | 310.3（每路 83.3，TTFT 462 ms） | 103.8（每路 28.5，TTFT 615 ms） | 108.0（每路 36.5，TTFT 592 ms） |

采样（T 1）下数数提示 2~3 路的聚合反而低于单路；5 次结果一致（2 路 77.8~80.6 tok/s），是批量采样路径在高可预测文本上的表现，不是噪声。下面的 prefill
是 2026-10-29 用同一版本测的。

| 提示词 | Prefill 时间 | Prefill 速度 |
| --- | --- | --- |
| 13,449 token | 6.54 s | 2,056 tok/s |
| 13,456 token | 6.46 s | 2,084 tok/s |
| 77,204 token | 38.72 s | 1,994 tok/s |
| 77,274 token | 39.10 s | 1,976 tok/s |

TTFT 随流数增加而增长，因为新请求是在正在运行的请求的 decode 轮次之间被接纳的。
多轮对话的后续轮次会复用上一轮的状态：在 16K token 的对话中，Qwen 下一轮的 TTFT 约为 0.3 s，而无需完整 prefill
（提示词检查点）。

### 复现

```bash
engine-rs/serve/start.sh qwen                # 或 ds41 / glm
for k in prose structured; do
  SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py $k 5 400                              # 贪心、关思考
  SPARK_URL=http://127.0.0.1:8888 bench/decode_bench.py $k 5 400 --temperature 1 --thinking   # T 1、开思考
done
for s in 11 12; do bench/prefill_bench.py 8000 $s; done
for s in 21 22; do bench/prefill_bench.py 46000 $s; done
```

## 方法说明

- 每个模型测量前都重启了服务，并关闭持久化前缀缓存（`GLM53_PCACHE=0`），以保证 prefill 数据为冷启动结果。
- `decode_bench.py` 的每个并发级别运行 5 次（GLM 为 3 次）；表中为中位数。
- Prefill：每种长度使用两个不同 seed 的提示词；表中两者都列出。
- GLM-5.3-Flash 运行在通过两个 ConnectX-7 端口互连（2 × 200 Gb/s）的两台 DGX Spark 上；Qwen3.8-Flash-Next 运行在一台上。
