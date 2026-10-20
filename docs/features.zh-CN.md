# 引擎工作原理

spark-engine 是一个 Rust 程序（`spark-engine`），包含三条模型路径：`serve` 在两台或三台 DGX Spark 上运行 GLM-5.3-Flash，
`ds41-serve` 在三台上运行 DeepSeek-V4.1-Flash，`qwen-serve` 在一台上运行 Qwen3.8-Flash-Next。运行哪个模型、用几个节点、
开哪些开关都只由配置决定（启动脚本、profile、`spark.env`），从不分别构建。每条路径前面各有一个 Python 进程，负责提供 OpenAI API，处理分词、
聊天模板、推理内容解析、工具调用解析和媒体解码，并通过 Unix socket 与引擎通信。

## 精度等级

每项优化都有分级，分级决定了它的测试方式：

| 等级 | 含义 | 验收标准 |
| --- | --- | --- |
| L0 | 输出逐位一致（调度、布局或融合方面的改动） | 逐位比较 logits、特征和状态 |
| L1 | 舍入级差异，不劣于参考 | 相对 FP32 参考的误差不大于改动前 |
| L2 | 仅影响草稿侧：草稿可以变化，验证后的输出不能变 | 验证后的 token 不变 |
| L3 | 按设计有损（有意选择的取舍） | 对照参考模型衡量质量 |

默认配置中的有损选择：FP8 KV 缓存（GLM 和 Qwen）；GLM 的 decode 稠密权重使用 Q8（int8，每 128 个输入一个 FP32 scale，
相对误差约 0.7 %）；checkpoint 自身的 EXL3 量化。DeepSeek 按规定运行官方 checkpoint 的格式（MXFP4 专家、FP8 权重和 KV、
FP4 压缩 KV）。

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
- **所有模型共用的模块。** `shim/pdl.cuh`（programmatic dependent launch 原语）、`shim/topk_select.cuh`（每行一个 CTA
  的 top-k 阈值搜索：GLM 的 DSA 和 DeepSeek 的 indexer）、`shim/mhc_common.cuh`（mHC 门控、行 softmax、Sinkhorn）、
  `src/kv_pool.rs`（建在 CUDA 虚拟内存上、可原地增长的 KV 池，按每 token 数组组织）、`src/mm_canvas.rs`（多模态画布和
  加盐占位符，使图像身份保留在前缀键中）、`src/pcache_core.rs`（NVMe 前缀缓存：O_DIRECT 写/读线程、校验和、同一对话
  各检查点之间共享的分段、LRU、启动时跨 rank 对账；每个模型把自己的检查点描述为行数组 + 固定区域）。
  `spark-engine pcache-golden <dir>` 是它的回归测试（逐字节清单在 `engine-rs/testdata`）。
- **N 路 RDMA（`shim/rdma_arn.cu`）。** 用于三节点：每个 rank 通过进程组的 store 发布自己的 RoCE 设备及其子网；共享
  子网的一对设备构成一条链路。all-reduce 以 FP32 按固定 rank 顺序求和（确定性，各 rank 结果相同）；大消息做流水线；
  prefill 使用按行的 reduce-scatter / all-gather，对端部分和可选用 BF16 传输。

## GLM-5.3-Flash（TP2 / TP3）

支持的架构：45 层，混合 KDA（kimi-delta 线性注意力）和 MLA（多头潜在注意力，带 DSA 稀疏索引器），4 流
hyper-connection（mHC），每个 MoE 层 288 个路由专家 + 1 个共享专家，前几层为稠密 MLP。

- **双节点张量并行。** 切分注意力头、共享专家和专家中间维度（每个 rank 负责 2048 中的 1024）；每层需要若干次
  小张量的 all-reduce。NCCL 处理大消息；引擎自带的、跑在两个 ConnectX-7 端口上的 RDMA all-reduce 处理小消息
  （12–50 µs，固定求和顺序，与 NCCL 逐位相同），并与使用其结果的 hyper-connection 更新融合。
- **三节点。** 相同的 profile 和 checkpoint；切分随 world size 变化（`crate::tp::Part`），两节点时每个切分都与之前逐位
  相同。KDA 头 22/21/21，MLA 32/16/16（以 16 头为单位），每个路由专家的 16 个 Hadamard 块 6/5/5（6 块的份额随专家轮换，
  因此每层每个节点的工作量相同；宽度可变的 MoE 内核跳过未用的块），prefill 序列并行，行数填充到 3 的倍数。
- **权重。** 路由专家保持 checkpoint 中的 EXL3 4 bpw。稠密层从 Q8 副本解码（分块排布，使每次 warp 加载为 512 个连续字节）；
  另提供无损 12 bit 编码（C12，每个解码值都等于原始 half 值）作为备选。prefill 以原始精度的稠密权重配合 cuBLAS 计算。
- **投机解码。** DFlash2 草稿模型（一个以目标模型隐藏特征为条件的小模型）提出 token 链/树；目标模型在一次前向中完成验证。
  验证严格保持贪心语义：
  - KDA 状态通过延迟修正重放推进，因此被拒绝的草稿不产生额外的状态拷贝；
  - MLA 验证为所有分支读取同一个共享基础状态；
  - DSA 选择按每个验证行分别进行；
  - 草稿深度跟随草稿模型的置信度（乘积停止规则）；当最近的 token 出现重复（代码、JSON、引用）时，复制草稿会复用之前的上下文。
  多个序列在一次批量前向中完成验证。
- **服务。** 默认最多 4 个并发序列，请求之间复用前缀（内存中的存储、提示词从已缓存序列分叉处的分叉检查点，以及 NVMe
  持久化前缀缓存），建在可增长存储上的 1M token KV 预算，图像和视频输入经由模型的视觉塔处理。

## DeepSeek-V4.1-Flash（TP3）

支持的架构：40 层（0–19 为因果编码器，20–39 为解码器），单遍 mHC；压缩稀疏注意力（64 个 512 维查询头共享一个 KV 头，
每层 128 token 滑动窗口，FP4 压缩全局 KV，FP4 indexer 选择 top-512，解码器中带分层候选池）；带共享专家的 384 专家 top-6
MoE（MXFP4）；第 1 层和第 14 层的 Engram n-gram 表；三个 DSpark 草稿级；带 MLP 对齐器的 ViT。部署方法：
[ds41.zh-CN.md](ds41.zh-CN.md)。

- **切分。** 注意力按头组切分（8 组按 3/3/2；分组低秩输出投影留在本地），路由专家和共享专家按中间维度切分（每个节点负责
  2304 中的 768，因此每个节点正好读取三分之一），词表按 128 行块切分。每个 rank 加载预先切好的分片（`ds41-shard`），
  每个张量一次连续读取。
- **内核。** MXFP4 MoE 和 FP8 GEMM 使用块缩放张量核 MMA（`mma ... kind::mxf8f6f4.block_scale`，sm_121a）；多行块缩放
  GEMV 在每个 CTA 内完成激活量化，因此 1–8 行的开销与 1 行相当；融合的稀疏注意力（窗口 + 选中块、FP8/FP4 反量化、sink），
  带确定性的切分合并，一次启动处理多个序列；融合的 mHC、RMSNorm/RoPE 和 indexer 打分。
- **Prefill。** CED：编码器处理每个 token，解码器只重放最后一个窗口（decoder SWA bounded replay，与训练一致）。编码器
  注意力分成两个微批，使其中一个的 all-reduce 与另一个的计算重叠；Engram 行（每 token 约 12.7 KB，来自 NVMe）用排序、
  去重的 O_DIRECT 读取提前一个块取回。
- **投机解码。** DSpark：三个草稿级每步给出 5 个位置，Markov 头对其修正，置信度头估计每个位置的接受率；代价模型
  （`DS41_SPEC_COST`）为每个序列选择草稿数；验证 graph 以总行数为键，因此草稿长度不同的序列共用一个 graph。
- **服务。** 4 个并发序列，共享 KV 池（1.1 M token），内存中的前缀存储和 NVMe 前缀缓存（在 prefill 块内每 512 token
  一个检查点）；图像输入（图像 token 跳过 Engram，使用 `bias_vl`）；官方 `encoding.py` 构建提示词、DSML 工具调用和思考预算。

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
暂存缓冲区小且锁页（用 `cudaHostRegister` 注册的匿名页，内核内存规整从不隔离），启动脚本在启动完成后执行一次内存规整。
推荐的主机设置见 [deploy.zh-CN.md](deploy.zh-CN.md)。
