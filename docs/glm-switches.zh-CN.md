# GLM-5.3-Flash 引擎开关说明

本文列出 `engine-rs/profiles/glm-tp2.env` 设置的 `GLM53_*` 开关。该 profile 用于在两台 DGX Spark 上（张量并行 2）服务 GLM-5.3-Flash，下表中的取值即调优后的默认值。大多数开关用于选择内核、数据布局或启动调度；保留为开关是为了能把每项改动与它替换的路径做 A/B 对比。取值为 0 表示保留替代路径以便对照，服务中不使用。

精度等级（所列取值对输出的影响，相对于被替换的路径）：

| 等级 | 含义 |
| --- | --- |
| L0 | 输出逐位一致。 |
| L1 | 仅舍入级差异（求和顺序、Half/TF32 操作数舍入），不劣于参考。 |
| L2 | 只影响草稿侧：草稿与接受数可能变化，验证后的输出不变。 |
| L3 | 有损，按设计接受（decode 稠密权重 Q8、FP8 KV 缓存）。 |
| — | 与数值无关（内存、容量、服务、主机侧工作）。 |

对取值为 0 的开关，等级描述的是被关闭的路径（如“L3（开启时）”）。

说明：

- 服务启动脚本（`serve/glm-rank.sh`）先清除调用方的 `GLM53_*` 变量，再加载本 profile，然后应用 `spark.env`，其中的 `GLM53_*` 会覆盖 profile。脚本还会把 `GLM53_MAX_CONTEXT` 设为 KV token 预算。
- `serve` 命令总是开启 `GLM53_MHC_FUSED`、`GLM53_KDA_FUSED`、`GLM53_MLA_LATENT` 与 `GLM53_PREFILL_BATCH`。
- “行”指一次调用中的 token 行数：普通 decode 为 1，验证一条草稿链为 2..8，多序列批量验证最多 32，prefill 块为数百到数千。“decode 规模”指 1..32 行。
- C12 是 Half 权重的无损 12 bit 编码；Q8 是 int8 加每 128 个输入一个 FP32 scale。路由专家始终使用 checkpoint 自带的 EXL3 4 bit 权重。
- “含义”列中引用其他开关时省略 `GLM53_` 前缀。

## 服务与内存 (11)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_SERVE_MAX_SEQS` | `4` | 同时解码的最大序列数（1..8）。各序列轮转推进：每轮一个 prefill 块或一次验证。 | — |
| `GLM53_SERVE_BATCH` | `1` | 批量服务轮：所有活跃序列在一次前向里一起验证（草稿也一起生成），而不是每轮只处理一个序列。关掉时并发请求逐轮轮流处理，总吞吐不随并发增加。 | L0 |
| `GLM53_SERVE_MAX_STORES` | `4` | 最多同时存在的序列存储（每个含 KV 行、KDA 状态、一个草稿器槽位和自己的验证图；不少于 SERVE_MAX_SEQS）。已结束的存储保留用于精确前缀复用，直到空间被需要。 | — |
| `GLM53_KV_POOL` | `1` | 启动时按可用内存一次性分配 KV 池（而不是按序列分配）；两个 rank 取较小值，池大小即 KV token 预算。 | — |
| `GLM53_MEM_UTIL` | `0.92` | 整机内存目标占用率（MemTotal 的比例，0.5..0.98），用于计算 KV 池大小。 | — |
| `GLM53_MEM_RESERVE_GIB` | `5` | 为服务后续在 KV 池外的分配预留的 GiB：各存储的验证图、KDA 状态和草稿器窗口，以及 prefill 工作区。 | — |
| `GLM53_MAX_CONTEXT` | `20480` | MLA latent / DSA 缓存容量（token）。服务启动脚本（serve/glm-rank.sh）会用 KV token 预算覆盖它（GLM_KV_TOKENS，默认 1048576）；profile 中的值只用于离线工具。 | — |
| `GLM53_PCACHE` | `1` | 本地 NVMe 上的持久前缀缓存（目录 GLM53_PCACHE_DIR）：≥1024 token 的 prompt 与轮次边界检查点在后台写入（O_DIRECT、校验和、跨轮共享分段、LRU 上限 100 GiB），后续 prompt 延伸该前缀时恢复；恢复的状态与保存时逐位一致。 | L0 |
| `GLM53_PREFIX_ADMISSION` | `1` | 前缀池只接纳最终能保留下来的分块快照，跳过注定会被立即淘汰的快照。 | L0 |
| `GLM53_VISION` | `1` | 在 rank 0 加载 BF16 视觉塔；媒体占位在 prefill embedding 的 allreduce 之前填入。它在 KV 池定容前加载，池大小已扣除其占用。 | — |
| `GLM53_HOST_TRIM` | `1` | 加载结束后调用 malloc_trim(0)，把主机分配器的空闲页还给系统（每节点约 1.2 GiB）。 | — |
| `GLM53_RELEASE_FILE_CACHE` | `1` | 权重全部驻留 GPU 后，丢弃 safetensors 文件的页缓存（madvise DONTNEED + fadvise）；映射仅作为回退保留。 | — |

## 张量并行与通信（allreduce、RDMA） (11)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_DENSE_TP` | `1` | 非专家层在两个 rank 间切分：KDA 与 MLA 按头切，稠密 MLP 与共享专家按中间维切；输出用 FP32 部分和 allreduce 求和。 | L1 |
| `GLM53_VOCAB_TP` | `1` | embedding 与 LM head 按词表切分（跨分片查表）。 | L0 |
| `GLM53_RDMA_AR` | `1` | ≤1 MiB 的 FP32 allreduce 走引擎自带的 RDMA 路径：一个 RoCE RC 队列对、锁页并映射到 GPU 的主机槽位环、GPU 核负责拷贝并自旋等待到达标志、CPU 代理线程发起 RDMA 写。求和 x0+x1 与 2-rank NCCL 求和逐位一致；其他消息走 NCCL。 | L0 |
| `GLM53_RDMA_AR_INIT` | `1` | 进程组初始化时建立 RDMA allreduce 通道（RDMA_AR 依赖它）。 | — |
| `GLM53_RDMA_AR_DUAL` | `1` | 在另一个 RoCE 端口上再建一个队列对；≥32 KiB 的消息拆到两条链路上并行发送，写入同一接收槽位的字节不变。 | L0 |
| `GLM53_AR_FUSED` | `1` | RDMA allreduce 只负责发送和等待；其消费者（四流 mHC post）自己把本地部分和与对端接收槽相加，舍入与顺序不变。依赖 RDMA_AR。 | L0 |
| `GLM53_HOST_REGISTER` | `1` | RDMA 主机缓冲改用 cudaHostRegister 锁定的匿名页，而不是 cudaHostAlloc 内存，避免内核内存规整尝试迁移它们并失败。 | — |
| `GLM53_TP_SMALL_COMM` | `1` | 额外创建一个限定 4 个 CTA 的 NCCL 通信组，用于 ≤256 KiB 的消息（RDMA 路径不适用时生效）。 | L0 |
| `GLM53_TP_SMALL_COMM_ACTIVE` | `1` | 把 ≤256 KiB 的 NCCL allreduce 实际交给上述小通信组（TP_SMALL_COMM 的运行时开关）。 | L0 |
| `GLM53_TP_MOE_PACK` | `1` | 行数 ≤ TP_MOE_PACK_MAX_ROWS 时，把路由专家与共享专家的 FP32 部分和打包进一个缓冲，每个 MoE 层只做一次 allreduce（原为两次）。 | L0 |
| `GLM53_TP_MOE_PACK_MAX_ROWS` | `8` | TP_MOE_PACK 的行数上限（更大的行数实测更慢）。 | L0 |

## 稠密权重（C12 编码、Q8、FP8、cuBLAS） (30)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_W_FP16` | `1` | 非专家大投影权重的驻留精度：由 BF16 checkpoint 转为 FP16（0 = FP32）。约 0.25% 的值落入 FP16 次正规数。 | L1 |
| `GLM53_BF16_RESIDENT` | `1` | 路由 gate 权重与 mHC hc_*_fn 权重以 BF16 驻留（不再保存 FP32 副本），核内无损扩展为 FP32。 | L0 |
| `GLM53_TF32` | `1` | FP32 GEMM 使用 TF32 乘法输入（FP32 累加不变）。乘法输入精度降低。 | L1 |
| `GLM53_C12` | `1` | C12 编码：加载时把驻留的 Half 权重无损编码为 12 bit（符号 + 7 位尾数 + 4 位指数窗口；窗口外的值以 CSR 形式按 BF16 逃逸存储）。decode 规模的调用（1..32 行）读取编码副本（Half 字节数的 0.75）；KDA q/k/v 与共享专家 gate/up 各一次启动。与 skinny Half 路径逐位一致；Q8 也建立在这一框架上。 | L0 |
| `GLM53_C12_INDEX` | `1` | DSA indexer 投影（FP32 存储、数值为 BF16）改走编码 decode 路径（C12；Q8 开启时为 Q8），keys 与 gate 一次启动，输入为 Half 而非 TF32。 | L1 |
| `GLM53_C12_KS` | `8` | 编码 decode GEMM 启动的 K 切分：在 K 允许时固定为 8，而非沿用 skinny Half 路径的选择（求和切分不同）。 | L1 |
| `GLM53_C12_QU` | `2` | C12 GEMV 核每次迭代处理两个 quad（仅当每个切分的 quad 数为偶数时）；MMA 顺序不变。 | L0 |
| `GLM53_Q8` | `1` | decode 规模的稠密、KDA、共享专家、MLA、DSA indexer 与 LM head 权重 GEMM 读取 int8 权重 + 每 128 个输入一个 FP32 scale（8.25 bit），替代无损 C12；各类相对 RMS 误差 0.65–0.69%。prefill 仍用原精度权重。依赖 C12；可用 GLM53_Q8_SET 选择权重类别。 | L3 |
| `GLM53_Q8_TILED` | `1` | Q8 权重按（16 行 tile, 128 k）分块存储，每次 warp 加载为连续 512 B；解码后的操作数、MMA 顺序与归约均不变。 | L0 |
| `GLM53_Q8_XHALF` | `1` | 超过 8 行的 Q8 GEMM 读取输入 X 的 Half 副本，省去每个 warp 的 FP32 读取与转换（操作数相同）。 | L0 |
| `GLM53_HALF_SKINNY` | `1` | ≤16 行的 Half 权重 GEMM 改用引擎的 skinny MMA 核（FP32 输入在核内舍入为 Half，FP32 累加），替代 GEMV 或 cuBLAS。 | L1 |
| `GLM53_DENSE_GEMV` | `1` | 未被编码路径接管的单行 Half 投影（N ≥ 1536，K ∈ {128, 1536, 4096}）使用带宽型 GEMV 核（FP32 累加）。 | L1 |
| `GLM53_DENSE_LT` | `0` | Half 投影的 cuBLASLt 形状表路径；关闭（没有稳定的整模型收益）。开启时还会禁用 prefill 的纯 Half 辅助路径。 | —（关闭） |
| `GLM53_DENSE_SMALL` | `0` | 实验性的 small-N 两行投影核；关闭。 | —（关闭） |
| `GLM53_DENSE_FP8` | `0` | 稠密 MLP 层（第 0–2 层）权重 weight-only FP8（e4m3，逐输出通道 scale）。关闭：这些权重保持原精度（decode 时走 Q8）。 | L3（开启时） |
| `GLM53_KDA_FP8` | `0` | KDA q/k/v/o 权重随 DENSE_FP8 注册为 FP8。关闭：保持原精度。 | L3（开启时） |
| `GLM53_FP8_HEAD` | `0` | LM head weight-only FP8。关闭：head 保持原精度（decode 时走 Q8）。 | L3（开启时） |
| `GLM53_FP8_MLA` | `0` | MLA 投影（q_a、q_b、kv_a、o）weight-only FP8。关闭。 | L3（开启时） |
| `GLM53_FP8_SHARED` | `0` | 共享专家 gate/up/down weight-only FP8。关闭。 | L3（开启时） |
| `GLM53_FP8_SKINNY` | `1` | ≤16 行的 FP8 权重 GEMM 使用 skinny mma.m16n8k16 核（split-K 在 CTA 内按固定顺序归约；有直接读取 BF16 激活的变体）。目标模型的 FP8 类别均关闭时，本组 FP8 核服务于草稿器的 FP8 权重。 | L1 |
| `GLM53_FP8_WMMA` | `3` | skinny 核不适用时小行数 FP8 核的模式：3 = WMMA，K 切成多段后以 FP32 合并，e4m3 成对解码。 | L1 |
| `GLM53_FP8_SPLITS` | `8` | 上述小行数 FP8 核的 split-K 段数。 | L1 |
| `GLM53_FP8_SMALL_TRANSPOSE` | `1` | 小行数 FP8 核使用转置的权重 tile 布局。 | L0 |
| `GLM53_FP8_SMALL_PAD` | `1` | 小行数 FP8 核使用带填充的 tile 布局。 | L0 |
| `GLM53_FP8_LARGE` | `5` | 17..128 行的 FP8 调用（模式 5）使用 tile 融合核：在共享内存解码 FP8 后直接做 MMA；更大的调用交给 FP8_BIG。 | L1 |
| `GLM53_FP8_BIG` | `1` | prefill 规模的 FP8 调用（>128 行）使用张量核 GEMM 直接读取驻留的 FP8 权重，按行 scale 与 Half 舍入融合在尾处理中；不展开 Half 权重。与展开路径逐位一致。 | L0 |
| `GLM53_FP8_EPILOGUE` | `1` | FP8 权重上 >128 行的库 GEMM，在原地对 FP32 输出乘 scale 并按原语义舍入为 Half（不产生临时张量）。 | L0 |
| `GLM53_FP8_FREE_SOURCE` | `1` | 释放已注册为 FP8 的权重的原精度副本（FP8_BIG 不需要它们）。 | — |
| `GLM53_FP8_PREFILL_HALF` | `1` | prefill 规模的调用（>128 行）对保留的原精度权重使用普通 Half cuBLAS GEMM，而非 FP8 路径（只剩 FP8 副本时以 FP8_BIG 为准）。同时是 PREFILL_QKV_INTO 与 PREFILL_HALF_GLUE 的前提。 | L1 |
| `GLM53_FP8_QKV_FUSED` | `1` | KDA q/k/v 为 FP8 时按行拼接、一次 skinny 启动完成。KDA_FP8=0 时不生效（由 C12/Q8 的组启动承担）。 | L0 |

## KDA 线性注意力 (23)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_KDA_FUSED` | `1` | decode 步与验证使用融合的 KDA 递推状态核（服务端强制开启）。 | L1 |
| `GLM53_KDA_FORK_FUSED` | `1` | 验证分支直接读父节点状态 H 并写出新 H，不再先复制再更新。 | L0 |
| `GLM53_KDA_CHAIN_NORM` | `1` | 链式验证：所有链节点的 KDA 归一化批量一次完成。 | L0 |
| `GLM53_KDA_CHAIN_RECURRENT` | `1` | 链式验证：所有链节点的 KDA 递推批量一次完成。 | L0 |
| `GLM53_KDA_CONV_CHAIN` | `1` | 链式验证：整条链的 KDA 短卷积一个核完成。 | L0 |
| `GLM53_KDA_CONV_DEFERRED` | `1` | 验证期间不物化卷积窗口，提交时按已接受 token 写入。 | L0 |
| `GLM53_KDA_CONV_COMMIT_ONE` | `1` | 所有 KDA 层所选卷积窗口一次启动提交（纯拷贝），替代每层约 3 次拷贝。 | L0 |
| `GLM53_KDA_CONV_CACHED` | `1` | 使用加载时拼接好的卷积权重，不再每次调用拼接。 | L0 |
| `GLM53_KDA_CONV_SILU` | `1` | 在卷积核内完成 SiLU（与单独算子公式相同）。 | L0 |
| `GLM53_KDA_CONV_L2` | `1` | 延迟卷积 + SiLU + q/k/v 归一化一次启动完成。依赖 KDA_CONV_DEFERRED、KDA_CONV_SILU 与 NORM_FUSED。 | L0 |
| `GLM53_KDA_CORRECTION_REPLAY` | `1` | 链式验证记录 KDA 修正项而不物化每个节点的状态 H；提交时对已接受前缀原地重放。 | L0 |
| `GLM53_KDA_CORR_WARP` | `4` | 修正链核按每次启动选择：4 = token 数 ≥3 时用每行一线程的核，否则用每行一 warp 的核（两核逐位一致）。 | L0 |
| `GLM53_KDA_GATE_FUSED` | `1` | ≤8 行的门控融合：[fa;ga] 与 wb 一个核产出 beta，fb/gb 一个核产出 decay 与 sigmoid(g2)，o-norm×gate 一个核。 | L1 |
| `GLM53_KDA_GATE_ONE` | `0` | 门控各阶段合并为一个常驻核；关闭（实测更慢）。 | L0（开启时） |
| `GLM53_KDA_GATE_WIDE` | `1` | 原本把 9..32 行拆成 8 行分块的调用方，改为一次把最多 32 行交给融合门控。 | L0 |
| `GLM53_KDA_GATE_TILE` | `1` | 9..32 行的门控调用使用 tile 核：每个 block 计算 8 个输出、负责 K 的四分之一，X 放在共享内存。 | L0 |
| `GLM53_KDA_GATE_PT` | `1` | 门控 GEMV（1..32 行）每个（行, 输出）一个线程，权重每个 block 转换一次放入共享内存；lane 求和顺序不变。 | L0 |
| `GLM53_KDA_GATE_SIDE` | `1` | 在 CUDA Graph 内，门控小核链在旁路流上运行，同时主流做 q/k/v 投影（仅调度变化；旁路流只跑引擎自有核）。 | L0 |
| `GLM53_KDA_MULTI_LAUNCH` | `1` | 多序列批量验证：最多 8 条序列的卷积链与修正链各合为一次启动。 | L0 |
| `GLM53_KDA_ONORM_SIGMOID` | `1` | 保留原始 g2，由 o-norm 门控核原地完成 sigmoid。 | L0 |
| `GLM53_NORM_FUSED` | `1` | KDA q/k 的 l2 归一化（含 v 拷贝）、MLA RMSNorm、行输出 Half 舍入各合为一个核。 | L1 |
| `GLM53_KDA_PREFILL_FUSED` | `1` | prefill：卷积 + SiLU + l2 归一化一个核，decay 按原运算序列在一个核内计算，去掉 [T+3, C] 拼接。 | L1 |
| `GLM53_KDA_SEQUENCE` | `6` | prefill 递推核模式：6 = 每行 4 个 lane、float4 交错分块、每个 head 4 个 block。 | L1 |

## MLA / DSA 注意力 (37)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_MLA_LATENT` | `1` | 压缩 latent MLA：512 维 latent 缓存、吸收 K/V 投影、增量 DSA 选择、可用于 CUDA Graph 的有界缓存（服务端强制开启）。 | L1 |
| `GLM53_KV_FP8` | `1` | latent KV 行存为 512 个 FP8 e4m3 值 + 4 个 FP32 tile scale（528 字节），KV 显存约减半。 | L3 |
| `GLM53_MLA_SPARSE_FUSED` | `1` | 融合稀疏 latent 注意力，直接读取压缩 latent，不再 gather 出 FP32 K/V。 | L1 |
| `GLM53_MLA_ACTIVE_COPY` | `1` | latent/DSA pool 快照只复制有效行（长度在 GPU 上读取，图重放保持动态），而不是整个容量。 | L0 |
| `GLM53_MLA_WEIGHT_CACHE` | `1` | 缓存派生的 MLA 投影权重（只计算一次并共享）。 | L0 |
| `GLM53_MLA_HALF_BMM` | `1` | 用派生的逐头权重做 absorb/expand，采用批量 skinny Half MMA。 | L1 |
| `GLM53_MLA_BMM_C12` | `1` | absorb/expand 读取逐头 Half 权重的 C12 编码（字节数 0.75）；与 Half BMM 结果逐位一致。 | L0 |
| `GLM53_MLA_BMM_WIDE` | `1` | C12 absorb/expand 核每次最多处理 32 行，批量验证时每个 head 的权重只读一次（而不是每 8 行一次）。 | L0 |
| `GLM53_MLA_CHAIN_SHARED` | `1` | 链式验证节点直接别名其 base 的 latent 与 pools，不再逐节点复制（只自留后续节点可能覆盖的那一行 pool）。 | L0 |
| `GLM53_MLA_NODE_BATCH` | `1` | FP8 latent 下的链式验证：所有节点追加一次启动，链上 latent 行一次 FP8 写入。 | L0 |
| `GLM53_MLA_MULTI_LAUNCH` | `1` | 多序列批量验证：所有序列的 latent 追加、FP8 写入、DSA 打分、top-k 与注意力各一次启动。 | L0 |
| `GLM53_MLA_COMMIT_FUSED` | `1` | 所有 MLA 层链共享 latent 的行、尾部与长度提交合为一次启动。 | L0 |
| `GLM53_MLA_INDEX_SIDE` | `1` | 在 CUDA Graph 内，DSA indexer 投影在旁路流运行，主流同时计算 q_b 与 absorb（仅调度变化）。 | L0 |
| `GLM53_MLA_PREFILL_BATCHED` | `1` | ≥PREFILL_MIN_ROWS 行的 prefill 块批量按因果顺序计算 MLA/DSA，而不是逐 token。 | L1 |
| `GLM53_MLA_PREFILL_TC` | `1` | prefill 共享 latent 注意力使用张量核（每个 CTA 一个 query × 16 个 head；模式 8，FP8 KV 下为模式 9）。KV_FP8 时必需。 | L1 |
| `GLM53_MLA_PREFILL_F16` | `1` | 张量核 prefill 注意力的 FP16 形式（Q 与概率为 FP16，FP32 累加；模式 10，FP8 KV 下为 11）。 | L1 |
| `GLM53_MLA_TC16_HG` | `2` | FP16 prefill 注意力核每个 CTA 处理 2 组 16 头，共享每次 gather 的 key tile。 | L0 |
| `GLM53_MLA_PREFILL_DENSE` | `6` | MLA_PREFILL_TC 关闭时使用的 prefill 共享 latent 注意力模式（6 = 多个 head 共享共享内存中的 FP16 latent tile，FP32 打分）。本 profile 中 MLA_PREFILL_TC=1，故不使用。 | L1 |
| `GLM53_MLA_PREFILL_STRIDED` | `1` | prefill 的 absorb/expand 改为 strided TF32 GEMM，原地读取 q 并直接写成 o_proj 布局（无转置拷贝）。 | L1 |
| `GLM53_MLA_SCORE_2D` | `1` | 把跨 head 步长切片的 MLA 打分压成二维共享 key 的 FP32 GEMM；受 MLA_SCORE_2D_SCOPE 与 MLA_SCORE_2D_MIN_ROWS 限制。 | L1 |
| `GLM53_MLA_SCORE_2D_MIN_ROWS` | `2048` | MLA_SCORE_2D 的最小行数。 | L1 |
| `GLM53_MLA_SCORE_2D_SCOPE` | `prefill` | MLA_SCORE_2D 的作用范围：仅 prefill（用于 decode 会降低接受率）。 | L1 |
| `GLM53_DSA_INDEX_FUSED` | `1` | 融合 Ranked DSA 选择中的整数/mask 记账（score 与 top-k 不变）。 | L0 |
| `GLM53_DSA_KEY_FUSED` | `1` | indexer key 的 LayerNorm 与 gate 的连续化拷贝合为一次启动。 | L1 |
| `GLM53_DSA_NODE_FUSED` | `1` | 每个验证节点一次启动完成尾部拷贝、尾槽写入、临时 pool（四槽 softmax）与 pool 写入、latent 行写入和长度更新。 | L0 |
| `GLM53_DSA_POSITION_CAPTURE` | `1` | 追加前的位置在现有 mask 生产者中同流捕获，替代逐节点拷贝。 | L0 |
| `GLM53_DSA_SCORE_FUSED` | `3` | 逐节点 DSA 打分模式：3 = 五次启动合为一次，只读完整 pool，长度取自设备端。 | L1 |
| `GLM53_DSA_SCORE_MULTI` | `1` | 在所有追加完成后，一次扫描共享 pools 为整条链的节点打分（依赖 DSA_NODE_FUSED 与 MLA_CHAIN_SHARED）。 | L0 |
| `GLM53_DSA_TOPK_BATCH` | `1` | Ranked 验证器的 top-k 批量处理（逐行打分算术不变）。 | L0 |
| `GLM53_DSA_TOPK_FAST` | `1` | DSA top-512 只扫描有效前缀，排序与参考库实现完全一致，分数直接写入 mask 缓冲。 | L0 |
| `GLM53_DSA_ALL_VISIBLE` | `0` | 短上下文时直接选全部可见 key，替代 Ranked 选择（改变 FP32 求和顺序）；关闭。 | L1（开启时） |
| `GLM53_DSA_VISIBLE_DIRECT` | `0` | 全部可见情形下的直接注意力路径；关闭。 | L1（开启时） |
| `GLM53_DSA_PREFILL_LIMIT` | `1` | prefill DSA 只对每个查询块边界处已完整的 pool 打分，保留全宽 padding 与 top-k 语义。 | L1 |
| `GLM53_DSA_PREFILL_QBLOCK` | `auto` | prefill DSA 每块查询数：auto 使 [block, pools] 分数矩阵保持在约 64M 个 float（≤64K pools 时为 1024）。 | L0 |
| `GLM53_DSA_PREFILL_SCORE_FUSED` | `1` | prefill DSA index 打分不物化 [n, heads, pools] 张量。 | L0 |
| `GLM53_DSA_PREFILL_SCORE_TILED` | `1` | 分块（tiled）的 prefill DSA 打分核，同时自行写出可见性 mask。 | L0 |
| `GLM53_DSA_PREFILL_TOPK_FAST` | `1` | prefill DSA top-k（宽度 ≥1024）使用精确的快速 top-k 核（只读每行可见前缀）加 token 展开核，替代通用的 mask + top-k 与索引运算。 | L0 |

## mHC 超连接 (10)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_MHC_FUSED` | `1` | decode 使用融合的 mHC（流形约束超连接）pre/post 实现（服务端强制开启）。 | L1 |
| `GLM53_MHC_PRE_FUSED` | `1` | ≤16 行的 mhc_pre 用两个核完成：先按 K 切片求 24 个 mix 与平方和的部分和，再逐行完成 rstd、gate、softmax、Sinkhorn、收缩与 RMSNorm（替代约 30 个库算子）。 | L1 |
| `GLM53_MHC_PRE_TC` | `1` | mhc_pre 部分和使用 TF32 张量核（布局不变）。 | L1 |
| `GLM53_MHC_PRE_LARGE` | `1` | 大行数（prefill）的 mhc_pre：split-TF32 部分和（3 次 MMA，FP32 级精度）+ 原 finish。 | L1 |
| `GLM53_MHC_FINISH_REG` | `1` | mhc_pre_finish 把收缩后的行留在寄存器，不再写回再读。 | L0 |
| `GLM53_MHC_POST_FUSED` | `1` | 融合的 mhc_post 核（支持 broadcast embedding 的真实步长）。 | L0 |
| `GLM53_MHC_POST_FOUR_STREAMS` | `1` | 融合 mhc_post 核的四流变体（一个元素的四路超连接流一次处理）；MHC_POST_PACKED 与 AR_FUSED 依赖它。 | L0 |
| `GLM53_MHC_POST_PACKED` | `1` | mhc_post 直接消费打包的 [routed, shared] allreduce 缓冲。 | L0 |
| `GLM53_MHC_POST_PRE_FUSED` | `1` | prefill（>16 行）：mhc_post 与下一层 mhc_pre 的部分和用两个核完成，替代对残差的三遍读写。 | L0 |
| `GLM53_MHC_GROUP_ROWS` | `128` | prefill 的融合 post+pre 按此行数分组处理，使 finish 从 L2 而非 DRAM 重读新残差。 | L0 |

## MoE 专家与路由 (21)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_MOE_COOP` | `1` | decode 与验证的路由专家走引擎自有的 EXL3 MoE 核（shim/moe_exl3.cuh），每次启动最多 32 行。名称是历史遗留。 | L1 |
| `GLM53_MOE_BATCH` | `0` | 扁平批量 EXL3 专家路径（每 16 行一批），仅在 MOE_COOP 关闭时使用；关闭。 | —（关闭） |
| `GLM53_MOE_SCRATCH` | `zeros` | 逐行 EXL3 回退路径中 scratch 缓冲的初始化方式：zeros、empty 或 reuse。 | — |
| `GLM53_MOE_NO_COPY` | `1` | 路由专家只产出一个输出张量时直接使用（不拼接、不额外拷贝）。 | L0 |
| `GLM53_MOE_INPUT_HALF_REUSE` | `1` | 路由专家与共享专家共用同一次 Half 输入转换。 | L0 |
| `GLM53_MOE_PACK_DIRECT` | `1` | 路由专家核与共享专家 down 投影直接写入打包 allreduce 缓冲的两半（不拼接）。依赖 AR_FUSED。 | L0 |
| `GLM53_MOE_ROUTE_SIDE` | `1` | 在 CUDA Graph 内，融合路由在旁路流运行，主流同时计算共享专家（仅调度变化）。 | L0 |
| `GLM53_ROUTER_FUSED` | `1` | ≤16 行的融合路由（≤320 个专家、top-8 sigmoid + bias）：logits 与选择两个核。 | L1 |
| `GLM53_ROUTER_V2` | `1` | 路由 logits 核以行数为模板参数并预载权重；逐 lane FMA 顺序与 shuffle 归约树不变。 | L0 |
| `GLM53_ROUTER_WIDE` | `1` | 多序列批量验证中 17..32 行也使用融合路由，替代那里的 TF32 库路由。 | L1 |
| `GLM53_ROUTER_HALF_W` | `1` | 路由选择同时写出专家核所需的 Half 路由权重（无需单独转换）。 | L0 |
| `GLM53_SHARED_GU_FUSED` | `1` | 共享专家 gate/up 原生 Half 融合。 | L1 |
| `GLM53_SHARED_GU_ROWS` | `1` | FP8 共享专家 gate 与 up 按行拼接一次 skinny 启动（C12/Q8 路径用自己的组启动）。 | L0 |
| `GLM53_SHARED_GU_F32` | `1` | 共享专家 gate/up 激活同时写出其 Half 值的 FP32 形式，编码 down 投影无需额外转换启动。 | L0 |
| `GLM53_PREFILL_COOP` | `1` | prefill 的路由专家走引擎的 EXL3 MoE 核（shim/moe_exl3.cuh）：≥64 行时按专家分组（见 PREFILL_GROUPED），否则每次最多 32 行（不足 8 行的尾部走逐行 EXL3 路径）。名称是历史遗留。 | L1 |
| `GLM53_PREFILL_GROUPED` | `recon` | ≥64 行 prefill 块的分组 MoE（0 为关闭，同时会关闭序列并行 prefill）。默认核下 recon 与 direct 都运行 moe_exl3.cuh 的分组 GEMM；仅当 GLM53_MOE_RECON=1 时 recon 才选择重建档（解码为 Half 权重 + cuBLAS）。 | L1 |
| `GLM53_PREFILL_RECON_MIN_ROWS` | `64` | 仅用于重建档（GLM53_MOE_RECON=1）：行数不少于此值的专家用解码权重 + cuBLAS，其余走直接 EXL3 GEMM。本 profile 中不生效。 | L1 |
| `GLM53_GROUPED_INDEX_PACK` | `1` | 仅用于重建档：所有专家的 gather/scatter 索引一次上传为一个 [2, T*8] 张量。本 profile 中不生效。 | L0 |
| `GLM53_GROUPED_SWIGLU` | `1` | 仅用于重建档：gate/up 转换、clamp、SiLU、相乘与 Half 舍入一个核完成。本 profile 中不生效。 | L0 |
| `GLM53_GROUPED_REDUCE` | `1` | 仅用于重建档：路由输出按固定顺序确定性归约（FP16 转 FP32、Half 路由权重、无原子操作）。本 profile 中不生效。 | L0 |
| `GLM53_PREFILL_EXPERT_STREAMS` | `0` | 重建档中独立专家在小型流池上并发运行；0 = 关闭。 | L0（开启时） |

## Prefill (14)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_PREFILL_CHUNK` | `4096` | prefill 分块大小（token）；服务端每个序列轮次处理一个块或一次验证。分块位置改变 GEMM 形状。 | L1 |
| `GLM53_PREFILL_BATCH` | `1` | 分块 prefill 每块只读一次投影权重并批量前向（保留 KDA 递推与 DSA 因果顺序）；服务端强制开启。 | L1 |
| `GLM53_PREFILL_MIN_ROWS` | `128` | 行数不少于此值的 prefill 块使用 KDA 序列核与批量 MLA/DSA。 | L1 |
| `GLM53_PREFILL_LAST_LOGITS` | `1` | prompt 只计算最后一行的 logits（状态与草稿特征仍覆盖全部行）。 | L0 |
| `GLM53_PREFILL_SP` | `1` | 两 rank 序列并行 prefill（≥128 行且行数为偶数）：残差按行切分，以 reduce-scatter/all-gather 替代 allreduce。 | L1 |
| `GLM53_PREFILL_SP_HALF_GATHER` | `1` | 序列并行 prefill：先为本 rank 的行做路由，再 all-gather Half 专家输入与打包路由，替代 FP32 输入。 | L1 |
| `GLM53_PREFILL_MOE_SUM1` | `1` | prefill（>64 行）：把共享专家的 FP32 部分和加到路由部分和上，只归约一次（每个 MoE 层一次集合通信而非两次）。 | L0 |
| `GLM53_PREFILL_SHARED_FUSE` | `1` | 先算共享专家部分和，在分组路由归约内相加，并直接写入 reduce-scatter 输入。 | L0 |
| `GLM53_PREFILL_SHARED_HALF` | `1` | prefill 共享专家改为 Half gate/up GEMM + 一个 SwiGLU 核直接写出 Half 的 down 输入，不产生 FP32 gate/up 张量。 | L0 |
| `GLM53_PREFILL_HALF_REUSE` | `1` | 共享专家复用路由分支对输入做的 Half 转换。 | L0 |
| `GLM53_PREFILL_FEATURE_REUSE` | `1` | 草稿特征直接取下一层融合 post+pre 写出的残差，不再额外做一次 mhc_post。 | L0 |
| `GLM53_PREFILL_QKV_INTO` | `1` | 走普通 Half GEMM 的 prefill 投影直接写入拼接输出的列切片（无逐投影 FP32 中间结果与拼接）。 | L0 |
| `GLM53_PREFILL_HALF_GLUE` | `1` | KDA prefill：q/k/v 与门控 GEMM 结果保持 Half（无 FP32 往返），o-norm 门控直接写出输出投影的 Half 输入。 | L0 |
| `GLM53_HALF_INPUT_CACHE` | `1` | 供多个 Half GEMM 使用的 prefill 规模输入只做一次 Half 转换并复用（以存储版本号校验）。 | L0 |

## 投机解码与草稿器（DFlash2） (47)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_SPEC_MAX_DRAFT` | `7` | 草稿链最大深度（1..7）。 | L2 |
| `GLM53_DEPTH_RULE` | `rate` | 单序列验证深度规则：rate = 根据校准的前缀接受概率，取“期望产出 token 数 − λ × 本轮耗时”最大的深度（否则：在第一个置信度低于 τ 的草稿处截断）。 | L2 |
| `GLM53_DRAFT_CONF_TRUNC` | `1` | 为每个草稿计算置信度（所选 token 在 top-16 内的 softmax 概率），使每轮可截断验证深度（τ 默认 0.7）；各深度的固定链图缓存并预捕获。 | L2 |
| `GLM53_SERVE_BATCH_CUMCONF` | `0.5` | 多序列批量验证：每条链在草稿置信度累积乘积低于该值处停止。 | L2 |
| `GLM53_COPY_DRAFTS` | `1` | 复制（prompt-lookup）草稿：上下文最后 8 个 token 曾在 prompt + 回复中出现时，本轮验证其后续 token，而不是草稿器的链。 | L2 |
| `GLM53_COPY_MAX` | `15` | 长复制链长度（≤15）：复制链要么不超过 7 个草稿，要么在后续 token 足够时恰为此长度（每个存储多一张 16 行链图）。 | L2 |
| `GLM53_CHAIN_SHARED_BASE` | `1` | 各深度的链验证图共用同一份已提交状态存储，切换深度无需恢复状态。 | L0 |
| `GLM53_STATE_INPLACE` | `1` | 提交时把已接受节点的状态直接写入验证图的 base（KDA 修正原地重放、卷积窗口、MLA 只复制新增行）；下一轮检测到别名后跳过恢复。 | L0 |
| `GLM53_MULTI_HOST_ROOM` | `1` | 多序列批量验证每轮用主机已知长度检查一次 latent 容量，替代每层每序列一次设备到主机读取。 | L0 |
| `GLM53_TARGET_TOP1_TP` | `1` | 在按词表切分的 head 上做精确的验证 top-1，无需全词表归约。 | L0 |
| `GLM53_GUMBEL_FUSED` | `1` | 采样请求的 Gumbel 噪声在验证 top-1 核内计算（FP32 初筛，候选用精确 FP64）。 | L0 |
| `GLM53_ACCEPTED_FEATURE_VIEW` | `1` | 固定链特征矩阵的已接受前缀以 view 形式交给草稿器（无拼接）。 | L0 |
| `GLM53_SPEC_MODES` | `batch-graph-chain` | 离线投机检查/基准工具运行的模式；batch-graph-chain = 用 CUDA Graph 验证固定链（服务端使用的模式）。 | — |
| `GLM53_SPEC_GRAPH_SLOTS` | `4` | 仅树模式：已捕获树图的 LRU 槽数。链式验证不使用。 | L0 |
| `GLM53_SPEC_TREE_CAPTURE_AFTER` | `3` | 仅树模式：同一树拓扑出现此次数后才捕获图。 | L0 |
| `GLM53_SPEC_TREE_CAPTURE_MIN_REMAINING` | `128` | 仅树模式：剩余 token 数不少于此值时才接纳新树拓扑捕获。 | L0 |
| `GLM53_SERVE_DRAFT_BATCH` | `1` | 多条序列的草稿提议与上下文追加合并为一次草稿器前向，每轮只读一遍草稿器权重与 head。 | L2 |
| `GLM53_DRAFT_MANY_GRAPH` | `1` | 2..4 条序列的草稿提议按槽位组合捕获为 CUDA Graph 并重放。 | L0 |
| `GLM53_DRAFT_GRAPH` | `1` | 把整个草稿提议捕获为一张 CUDA Graph（固定地址 KV slab 槽位、设备侧元数据、flash-decoding 注意力核）。 | L2 |
| `GLM53_DRAFT_APPEND_GRAPH` | `1` | 草稿器上下文追加使用 CUDA Graph，按行数 1..8 各一张。 | L0 |
| `GLM53_DRAFT_APPEND_TRIM` | `1` | fc 之前截掉超出草稿器窗口的行；整窗口替换时不复制旧 KV（保留绝对位置与逻辑长度）。 | L0 |
| `GLM53_DRAFT_KV_BUFFER` | `1` | 草稿器 KV 追加写入固定地址的 slab 缓冲。 | L0 |
| `GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT` | `2048` | 仅在上下文不少于此 token 数时使用 slab。 | L0 |
| `GLM53_DRAFT_ATTN_TP` | `1` | 草稿器注意力按头在两 rank 间切分（q/k/v 按头、o 按列，一次 FP32 allreduce）；每槽草稿 KV 减半。 | L2 |
| `GLM53_DRAFT_MLP_TP` | `1` | 草稿器 MLP 切分（gate/up 按中间维，down 的 FP32 部分和跨 rank 归约后转 BF16）。 | L2 |
| `GLM53_DRAFT_MLP_SHARD_LOAD` | `1` | 只加载本 rank 的草稿器 MLP 分片（不先加载全量）。 | L0 |
| `GLM53_DRAFT_HEAD_TP` | `1` | 草稿器 LM head 按词表切分，保留完整的候选选择接口。 | L2 |
| `GLM53_DRAFT_HEAD_SHARD_LOAD` | `1` | 只加载本 rank 的草稿器 head 分片。 | L0 |
| `GLM53_DRAFT_TOPK_TP` | `1` | 草稿器 head 的分布式 top-16：只交换各 rank 的候选。 | L2 |
| `GLM53_DRAFT_HEAD_INT4` | `1` | 用本 rank 草稿 head 切片的粗粒度 INT4 分数为每行预选 64 个候选，再用同一核以 FP8 head 对这些行重打分并返回 top-16。 | L2 |
| `GLM53_DRAFT_HEAD4_TILED` | `1` | 草稿器 INT4 head 副本按 INT4 MMA 核读取的 tile 顺序存储（草稿不变）。 | L0 |
| `GLM53_DRAFT_FP8_ATTN` | `1` | 草稿器 q/k/v/o 投影为 weight-only FP8（skinny 核直接读 BF16 激活）。 | L2 |
| `GLM53_DRAFT_FP8_CONV` | `1` | 草稿器 attention/MLP 卷积投影为 weight-only FP8。 | L2 |
| `GLM53_DRAFT_FP8_FC` | `1` | 草稿器 fc 投影为 weight-only FP8。 | L2 |
| `GLM53_DRAFT_FP8_HEAD` | `1` | 草稿器 LM head 为 weight-only FP8。 | L2 |
| `GLM53_DRAFT_FP8_MLP` | `1` | 草稿器 MLP gate/up/down 为 weight-only FP8。 | L2 |
| `GLM53_DRAFT_Q4` | `1` | 草稿器主体权重（除 head 外的各 DRAFT_FP8_* 类）另存为仿射 4-bit，decode 规模调用（1..32 行）读取它。 | L2 |
| `GLM53_DRAFT_SKINNY32` | `1` | 草稿器 BF16 激活 FP8 skinny 核的行数上限从 16 提高到 32（用于多序列合批草稿）。 | L2 |
| `GLM53_DRAFT_GQA` | `5` | 草稿器注意力模式 5：4 个 query head 共享 1 个 KV head；短历史（≤DRAFT_GQA_SHORT_MAX）用双段 BF16 融合核，长历史用分组 strided BMM。 | L2 |
| `GLM53_DRAFT_GQA_SHORT_MAX` | `128` | GQA 短历史路径的长度阈值。 | L2 |
| `GLM53_DRAFT_GQA_PRECISE` | `1` | 长历史分组的概率×value 乘积固定用 FP32（该乘积禁用 TF32）。 | L2 |
| `GLM53_DRAFT_FUSED_NORM` | `1` | 草稿器小核融合：add + RMSNorm（及相关逐元素步骤）每行一个核。 | L2 |
| `GLM53_DRAFT_NORM_CACHE` | `1` | 缓存草稿器预先展开的 norm 权重。 | L0 |
| `GLM53_DRAFT_ROPE_CACHE` | `1` | 复用草稿器 RoPE 位置与三角函数表。 | L0 |
| `GLM53_DRAFT_CONV_FUSED` | `1` | 融合草稿器 Conv::convolve（投影、norm 与残差不变）。 | L0 |
| `GLM53_DRAFT_FINAL_NORM_SELECT` | `1` | 草稿器最终 residual + norm 省去没有消费者的 BF16 残差转换。 | L0 |
| `GLM53_DRAFT_SELECTOR_FUSED` | `1` | 融合草稿路径选择（edges/ID 语义不变）。 | L0 |

## CUDA Graph 与其他 (3)

| 变量 | 取值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `GLM53_GRAPH` | `1` | decode/验证使用 CUDA Graph（设置即启用；仅支持 TP2，因为需要全部专家常驻）。 | L0 |
| `GLM53_GRAPH_SHARED_WS` | `1` | 所有图共用一个 cuBLAS/cuBLASLt workspace，而不是每个捕获流一个（每图约 32 MiB）。 | — |
| `GLM53_STATIC_TENSORS` | `0` | 把常量张量（decay 基、卷积权重、DSA 常量）静态化；关闭（无可测收益）。 | L0（开启时） |
