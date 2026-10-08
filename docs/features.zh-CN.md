# 引擎工作原理

spark-engine 是一个 Rust 程序（`spark-engine`），包含两条模型路径：`serve` 在两台 DGX Spark 上运行 GLM-5.3-Flash，
`qwen-serve` 在一台上运行 Qwen3.8-Flash-Next。每条路径前面各有一个 Python 进程，负责提供 OpenAI API，处理分词、
聊天模板、推理内容解析、工具调用解析和媒体解码，并通过 Unix socket 与引擎通信。

## 精度等级

每项优化都有分级，分级决定了它的测试方式：

| 等级 | 含义 | 验收标准 |
| --- | --- | --- |
| L0 | 输出逐位一致（调度、布局或融合方面的改动） | 逐位比较 logits、特征和状态 |
| L1 | 舍入级差异，不劣于参考 | 相对 FP32 参考的误差不大于改动前 |
| L2 | 仅影响草稿侧：草稿可以变化，验证后的输出不能变 | 验证后的 token 不变 |
| L3 | 按设计有损（有意选择的取舍） | 对照参考模型衡量质量 |

默认配置中的有损选择：FP8 KV 缓存（两个模型）；GLM 的 decode 稠密权重使用 Q8（int8，每 128 个输入一个 FP32 scale，
相对误差约 0.7 %）；checkpoint 自身的 EXL3 量化。

## 共享内核

- **EXL3 解码。** EXL3 将每个 16×16 权重块存储为 trellis 比特流；数值来自码本（GLM 的专家用 `mcg`，Qwen 用 `mul1`），
  并直接解码到 tensor core（mma.m16n8k16）的 fragment 中。线性层计算为
  `y = had128(had128(x·suh) · W) · svh`（带符号/缩放向量的 128 点 Hadamard 变换），行数较少时沿 K 维切分，
  并按固定顺序求和（结果确定）。
- **MoE（`shim/moe_exl3.cuh`）。** 一个模板，针对两个模型分别实例化（专家数、top-k、尺寸、码本、激活函数）：
  路由得到的（行，专家）对在设备端以固定网格按专家分组（便于 CUDA graph 捕获），gate/up 乘法以分组 GEMV（decode）
  或 GEMM（prefill）运行，SwiGLU 和 down 投影的输入变换融合为一趟，每行的各专家结果按槽位顺序求和。
  - GLM decode 使用一个常驻的融合内核：所有路由专家的 gate/up、SwiGLU 和 down 在一次启动中完成，用逐专家的就绪标志
    代替内核边界，其余小内核之间使用 programmatic dependent launch。
  - GLM prefill 利用了同一层所有专家共享一个输入缩放向量这一点：输入变换每个 token 只做一次，而不是每个（token，专家）
    做一次；成对的 gate/up GEMM（每个 CTA 64 行 × 128 列，每个 SM 两个 CTA）直接写出 down 投影的激活值。
- **Graph。** 每种 decode/验证形状都被捕获为 CUDA graph；graph 内的内存分配来自各 graph 独立的内存池。

## GLM-5.3-Flash（TP2）

支持的架构：45 层，混合 KDA（kimi-delta 线性注意力）和 MLA（多头潜在注意力，带 DSA 稀疏索引器），4 流
hyper-connection（mHC），每个 MoE 层 288 个路由专家 + 1 个共享专家，前几层为稠密 MLP。

- **双节点张量并行。** 切分注意力头、共享专家和专家中间维度（每个 rank 负责 2048 中的 1024）；每层需要若干次
  小张量的 all-reduce。NCCL 处理大消息；引擎自带的、跑在两个 ConnectX-7 端口上的 RDMA all-reduce 处理小消息
  （12–50 µs，固定求和顺序，与 NCCL 逐位相同），并与使用其结果的 hyper-connection 更新融合。
- **权重。** 路由专家保持 checkpoint 中的 EXL3 4 bpw。稠密层从 Q8 副本解码（分块排布，使每次 warp 加载为 512 个连续字节）；
  另提供无损 12 bit 编码（C12，每个解码值都等于原始 half 值）作为备选。prefill 以原始精度的稠密权重配合 cuBLAS 计算。
- **投机解码。** DFlash2 草稿模型（一个以目标模型隐藏特征为条件的小模型）提出 token 链/树；目标模型在一次前向中完成验证。
  验证严格保持贪心语义：
  - KDA 状态通过延迟修正重放推进，因此被拒绝的草稿不产生额外的状态拷贝；
  - MLA 验证为所有分支读取同一个共享基础状态；
  - DSA 选择按每个验证行分别进行；
  - 草稿深度跟随草稿模型的置信度（乘积停止规则）；当最近的 token 出现重复（代码、JSON、引用）时，复制草稿会复用之前的上下文。
  多个序列在一次批量前向中完成验证。
- **服务。** 默认最多 4 个并发序列，请求之间复用前缀（内存中的存储，外加可选的 NVMe 持久化前缀缓存），1M token KV 预算，
  图像和视频输入经由模型的视觉塔处理。

## Qwen3.8-Flash-Next（单节点）

支持的架构：门控 DeltaNet 线性注意力层和稀疏注意力层（QSA，由学习得到的索引器选择 key 块），带门控共享专家的 512 专家
top-10 MoE，4 流 hyper-connection，来自 39 GB 表的逐层 n-gram 嵌入（PLE），一个 MTP 头，以及 ViT 视觉塔。

- **内核。** 所有 EXL3 乘法都运行在引擎自有内核上（不使用 exllamav3 运行时）：共享输入变换的融合多线性层 decode 启动、
  支持多序列分段的 GDN 递推和卷积内核、QSA 池化/选择/注意力内核、融合的 hyper-connection 混合。n-gram 嵌入行由并行线程
  从内存映射表中收集，并带有预取提示。
- **投机解码。** 原生 MTP 头以自回归方式生成草稿；其输出层是 lm_head 的 4 bit 副本，仅保留 65,536 个最高频 token
  （`assets/qwen38/draft_vocab_65536.json`，分块布局），因此每个草稿步只读取 84 MB，而不是完整的输出头。上下文出现重复时
  使用 prompt-lookup 草稿。最多 8 个序列在一次批量前向中完成验证。
- **长上下文与多轮对话。** 1M token KV 池，以 16K token 为粒度（LRU）；仅对超过 262,144 token 的序列使用 YaRN 缩放。
  每个序列存储在其提示词的最后一条消息边界处保存一个检查点，因此即使客户端重新序列化了之前的轮次，后续轮次也只需对新消息
  重新 prefill。`max_tokens` 预先最多预留 32K token；序列的 KV 区间在 decode 过程中原地增长或迁移。
- **视觉。** 图像（每张最多 16,384 token）和视频（2 fps，最多 768 帧）由引擎内的 ViT 处理，使用交错的多模态 RoPE 位置。
- **推理等级。** `reasoning_effort` 将 none / low / medium / high 映射到聊天模板的思考模式。

## GB10 上的内存

GPU 和 CPU 共享 128 GB。引擎在加载后根据 `MemAvailable` 确定 KV 池大小，释放 checkpoint 分片占用的页缓存，保持主机端
暂存缓冲区小且锁页，启动脚本在启动完成后执行一次内存规整。推荐的主机设置见 [deploy.zh-CN.md](deploy.zh-CN.md)。
