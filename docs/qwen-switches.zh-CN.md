# Qwen3.8-Flash-Next：环境变量开关

Qwen 路径没有 profile 文件：所有默认值都编译在代码里，并且就是调好的设置。要改某一项，在 `spark.env` 或
`engine-rs/serve/start-qwen.sh` 的环境里设置对应变量。除非另有说明，引擎在加载时或每次调用时读取变量；影响 decode 的
"每次调用"开关在捕获 CUDA 图时生效。

精度等级：

- **L0**：输出逐位相同。
- **L1**：舍入级差别，与 FP32 参考相比不劣化。
- **L2**：只影响草稿侧；验证后的输出不变。
- **L3**：有损。
- **—**：不是数值开关（资源、路径、日志）。**n/r**：没有记录。

布尔约定："开（`=0` 关）"表示除 `0` 以外的任何值都保持开启；"关（`=1`）"表示只有 `1` 才打开；"设置即生效"表示只看变量是否存在。

## 服务设置

| 变量 | 默认值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `QWEN_MODEL` | 无（在 `spark.env` 中设置） | 模型目录（EXL3 检查点），由 `start-qwen.sh` 传给引擎（`qwen-serve`）和前端（`--model-dir`）。 | — |
| `QWEN_ASSETS` | 模型目录（`start-qwen.sh` 设为 `$SPARK_HOME/assets/qwen38`） | 存放 `draft_vocab_<N>.json` 的目录。发布包自带 `assets/qwen38/draft_vocab_65536.json`。 | — |
| `QWEN_VISION` | 引擎：关（`=1`）；`start-qwen.sh`：`1` | `1` 加载视觉塔（约 0.5 GB），接受图片 / 视频请求，同时以 `--vision on` 启动前端。`0` 只处理文本，带媒体的请求返回错误。 | — |
| `QWEN_MEMGUARD_GIB` | `8` | `start-qwen.sh` 的看门狗：每 0.5 s 检查 `MemAvailable`，低于这个 GiB 数就杀掉引擎（统一内存耗尽可能让 GB10 卡死）。 | — |
| `QWEN_SERVE_MAX_SEQS` | `8`（限制在 1..8） | 同时 decode 的序列数（每轮一次批量推测解码）。 | L0 |
| `QWEN_SERVE_STORES` | `8`（不少于 `QWEN_SERVE_MAX_SEQS`） | 序列存储数：每个存储保留已提交的历史，用于完全相同前缀的复用（GDN 递推状态无法回退，所以只复用完全相同的前缀）。 | L0 |
| `QWEN_KV_TOKENS` | `1048576` | KV 池大小（token），按 16384 token 的粒度取整，所有存储共用（启动时平均分配；更大的请求占更大的区间，空闲存储按 LRU 让出）。单个请求（prompt + 新 token）最长为池大小减 32。 | — |
| `QWEN_SERVE_CAP` | 不读取 | 只出现在源码注释里；单个请求的容量由 `QWEN_KV_TOKENS` 决定。 | — |
| `QWEN_SERVE_RESERVE` | `32768`（最小 256） | 准入时为输出预留的位置数：`min(max_tokens, 此值)`。输出更长时，decode 过程中扩展存储区间（原地扩展，或搬到空隙），所以接近上下文上限的 `max_tokens` 不会再占满整个池。 | L0 |
| `QWEN_SERVE_CKPT` | 开（`=0` 关） | 在 prompt 最后一个 `<\|im_start\|>` 处保存检查点（GDN 状态、conv 窗口、indexer key 环、PLE 窗口、MTP 待处理行；每个存储约 116 MB）。对话的下一轮恢复检查点，只 prefill 后面的部分，即使客户端回传的上一轮回复与生成的 token 对不上。 | L1 |
| `QWEN_SERVE_CKPT_MIN` | `1024` | 保存检查点的最短 prompt 长度（检查点位置之前的 token 数）。 | L1 |
| `QWEN_SERVE_GRAPH` | 开（`=0` 关） | 批量 verify / commit / MTP 草稿链使用 CUDA 图（需要 MTP 头）。`0`：eager 模式，一次一个序列。 | L0 |
| `QWEN_SERVE_GRAPH_MB` | `3000` | 多存储组合的图的显存预算（每个捕获的 verify 行约 5 MB）；超出时按 LRU 释放组合，单存储的图保留。 | — |
| `QWEN_PREFILL_MACRO` | `16384` | 每次层优先 prefill 处理的 token 数（这一段的激活全部常驻）；也是服务时每轮在 decode 之间为一个序列 prefill 的 prompt token 数。 | L1（改变分块边界） |
| `QWEN_YARN` | 自动 | 每个序列的 RoPE 模式：序列容量超过原生 262144 时用 YaRN（factor 4），否则用普通 RoPE。`1` / `0` 强制总是 / 从不。静态 YaRN 对短文本有一定质量损失。 | 强制时改变输出 |
| `QWEN_MTP` | 开（`=0` 关） | 加载 MTP 头（推测解码草稿）。`0`：没有草稿，也没有服务端 CUDA 图。 | L2 |
| `QWEN_SPEC_CUMCONF` | `0.7` | 草稿长度策略 θ：只要有序列的草稿置信度乘积仍 ≥ θ 就继续出草稿；每个序列验证到界内最后一个草稿为止。`0`：每轮固定 `QWEN_SPEC_K` 个草稿。 | L2 |
| `QWEN_SPEC_KMAX` | `10`（θ = 0 时为 `QWEN_SPEC_K`） | 每条草稿链最多的草稿数（批量时还受总共 64 个 verify 行的限制）。 | L2 |
| `QWEN_SPEC_K` | `3` | θ = 0 时每轮的固定草稿数；批量 `all` 规则的最小深度；序列第一条草稿链的上限。 | L2 |
| `QWEN_SPEC_CUMCONF_BATCH` | 未设置（沿用单序列 θ） | 2 个及以上序列成批时的 θ：(0, 1) 内的数为单独的 θ，`0` 为固定 k。 | L2 |
| `QWEN_SPEC_BATCH_RULE` | `all` | 批量深度规则。`all`：至少 k，所有序列都在界内才加深。`max`：按最有把握的序列加深（实测 prose 慢 6–25%）。 | L2 |
| `QWEN_SPEC_CALIB` | 未设置（恒等） | 把草稿头置信度映射为接受率的分段线性表，`"c:p,c:p,..."`（c 递增）；`fit` 为内置拟合表。实测没有稳定收益。 | L2 |
| `QWEN_DRAFT_VOCAB` | `65536` | 草稿头子词表：数字 N 加载 `$QWEN_ASSETS/draft_vocab_N.json`，其他值视为文件路径；`0` 或文件不存在时不建草稿头（草稿改用完整 `lm_head`）。 | L2 |
| `QWEN_DRAFT_Q4` | 开（`=0` 为 Q8） | 草稿头用仿射 4-bit（`0`：int8 + 每 128 一个 FP32 缩放）。Q4 实测单序列 tok/s +3–7%，接受率不变。 | L2 |
| `QWEN_LOOKUP` | 开（`=0` 关） | prompt lookup 草稿：取最后 n 个 token 在上下文（prompt 和输出）中最近一次出现后的续写；它的第一个 token 与 MTP 头的第一个草稿相同时使用，并替代该序列 MTP 链的其余部分。 | L2 |
| `QWEN_LOOKUP_N` | `3`（限制在 1..8） | prompt lookup 的 n-gram 长度。 | L2 |
| `QWEN_LOOKUP_MIN` | `0.6` | 序列的 lookup 提议按 token 的滚动接受率不低于此值才使用（并且 lookup 轮的每轮产出至少是 MTP 轮的 1/1.1；每 16 轮试一次另一边）。 | L2 |
| `QWEN_MTP_PREFILL_TAIL` | `0`（关） | prompt 的 MTP catch-up 只覆盖最后 n 行（之前的 MTP 缓存行清零）。长 prompt prefill 更快，但之后草稿接受率下降。 | L2 |
| `QWEN_REQLOG` | `/tmp/qwen38-requests.jsonl`（`""` 关闭） | 前端：每个请求记一行 JSON（大小、前缀命中、排队 / prefill / TTFT、prompt 与最接近的之前序列第一次分叉的位置及前后文本）。 | — |
| `QWEN_NGRAM_HOT` | 关 | 热 n-gram 表行文件（由 `ngram_hot.py` 生成），放在内存里，位于 mmap 的 n-gram 表之前。只在 page cache 被挤掉时有用。 | L0 |
| `QWEN_NGRAM_CACHE_MB` | `0`（关） | 运行时 n-gram 表行的 2 路组相联内存缓存（MB）。适用场景同 `QWEN_NGRAM_HOT`。 | L0 |
| `QWEN_KEEP_SHARD_CACHE` | 关（`=1`） | `1` 把加载过的模型分片留在 page cache。默认加载后释放（n-gram 表保留）：空闲内存更多、首次 prefill 更快，代价是重新加载更慢。 | — |

## 内核与调度开关

保留用于 A/B 测试；默认值即实测最优。等级相对于默认路径。

| 变量 | 默认值 | 含义 | 等级 |
| --- | --- | --- | --- |
| `QWEN_PREFILL_CHUNK` | `2048` | 层优先 prefill 中注意力子块的行数（比 1024 快 1%）。 | L1 |
| `QWEN_PREFILL_MOE_ROWS` | `16384` | 层优先 prefill 中每次 MoE 调用的行数。 | n/r |
| `QWEN_PREFILL_CM` | 关（`=1`） | 旧的块优先 prefill（每块跑一次完整前向），代替层优先。 | L0 |
| `QWEN_EXL3_FOLD` | 关（`=1`） | 多行 EXL3 线性层改用折叠后的 fp16 有效权重 + cuBLASLt GEMM，不再变换激活。复测精度不劣化，但在融合后的 prefill 路径下更慢。 | L1 |
| `QWEN_EXL3_SLICES_TUNED` | 开（`=0` 关） | 按形状调过的 EXL3 GEMV K 切分（K = 6144 用 10 片，N ≥ 6144 用 6 片）；`0`：通用规则。每轮快 1–2%。 | L1 |
| `QWEN_EXL3_FUSED` | 关（`=1`） | 最多 8 个 decode 行一次启动（输入变换 + GEMV + finish）。每轮慢约 5%。 | L0 |
| `QWEN_EXL3_FIN_F32` | 关（`=1`） | 多行 EXL3 的 finish 先把 GEMM 输出转成 fp32（旧路径），不直接读 fp16 输出。 | L0 |
| `QWEN_EXL3_MULTI` | 开（`=0` 关） | 同一输入的 decode 线性层：一次输入变换（融合 HC mix）、一次 GEMV 启动；`0`：逐个线性层。 | L0 |
| `QWEN_MIX_LAZY` | 开（`=0` 关） | decode：HC mix 由第一个使用者的输入变换计算；`0`：先算好 mix。 | L0 |
| `QWEN_PREFILL_HAD_MULTI` | 开（`=0` 关） | prefill：同一输入的所有线性层一次输入变换启动；`0`：各自启动。 | L0 |
| `QWEN_PREFILL_FIN_SPLIT` | 关（`=1`） | prefill：投影的 finish 单独启动，不融进使用者（conv、输出 norm、QSA prep）。 | L0 |
| `QWEN_PREFILL_PREP_SPLIT` | 关（`=1`） | prefill：GDN 递推的准备（q/k 归一化、decay/beta）单独一个 kernel，不放在 conv 里。 | L0 |
| `QWEN_PREFILL_SHARED_TORCH` | 关（`=1`） | prefill：shared expert 的逐元素算子用 torch，不用 decode 的 kernel。 | L0 |
| `QWEN_PLE_PACK_SYNC` | 关（`=1`） | prefill：先打包 PLE n-gram 行再排队各层，不在另一个线程上与前几层重叠。 | L0 |
| `QWEN_PLE_SERIAL` | 关（设置即生效） | PLE n-gram 表行逐个缺页读取（旧路径），不用 `MADV_WILLNEED` + 并行拷贝。 | L0 |
| `QWEN_PREFILL_APPLY_SPLIT` | 关（`=1`） | prefill：每层 MoE 的更新立即加上，不融进下一层 attn HC 的 norm。 | L0 |
| `QWEN_DEC_APPLY_SPLIT` | 关（`=1`） | decode：同上，作用于 decode / verify 行。 | L0 |
| `QWEN_HC_APPLY_SPLIT` | 关（`=1`） | HC 残差更新（apply）与随后的 norm 分成两个 kernel。 | L0 |
| `QWEN_HC_UPMIX_SPLIT` | 关（`=1`） | prefill HC：cuBLAS 做 up GEMM（g 为 fp16）+ 单独的 mix，不用融合的 tensor core up+mix（g 保持 fp32）。 | L1 |
| `QWEN_PREFILL_HC_TORCH` | 关（`=1`） | prefill HC 走旧的 torch 算子路径，不用 v2 kernel（v2 快 11%，且更接近参考）。 | L1 |
| `QWEN_HC_V1` | 关（设置即生效） | decode HC 用 v1 的单 kernel `hc_mix`，不用 v2 的几个 kernel（norm、down、mid、up、mix）。 | n/r |
| `QWEN_Q8_DENSE` | 关（`=1`） | 为 F16 稠密权重（HC、router、PLE、GDN a/b）做 Q8 副本，用于 decode / verify 行。单序列 verify −5.6%，但模型对 HC 权重误差极其敏感。 | L3 |
| `QWEN_Q8_SCOPE` | 全部 | 配合 `QWEN_Q8_DENSE=1`：要量化的权重类别，逗号分隔（`hc_down`、`hc_up`、`router`、`gdn_ab`、`ple`）。 | L3 |
| `QWEN_Q8_OFF` | 关（`=1`） | 配合 `QWEN_Q8_DENSE=1`：重新用 F16（同进程 A/B）。 | — |
| `QWEN_F16_TC` | 开（`=0` 关） | F16 GEMV（HC、router、GDN a/b、PLE）用 tensor core，x 拆成 fp16 hi/lo，结果与批大小无关；`0`：标量 v2 kernel。 | L1 |
| `QWEN_F16_V1` | 关（`=1`） | 在使用标量 F16 GEMV 的地方用 v1 kernel 代替 v2（两者逐位相同）。 | 相对 v2 为 L0 |
| `QWEN_F16_NJ` | 自动（4、2 或 1） | v2 F16 GEMV 每个 warp 处理的权重行数；默认取仍能得到 ≥ 192 个 block 的最大值。 | L0 |
| `QWEN_F16_FUSED` | 关（`=1`） | F16 GEMV 的切片求和放在最后到达的 block 里（标量 v1 kernel，不用 tensor core）。没有更快。 | 相对默认为 L1 |
| `QWEN_GDN_V1` | 关（`=1`） | GDN prefill 递推：旧的按头 kernel，不用行并行 kernel。 | L0 |
| `QWEN_GDN_RECUR_V1` | 关（`=1`） | GDN decode 递推：旧 kernel，不用经共享内存暂存的版本（后者在多序列 × 长链时会自动退回旧 kernel）。 | L0 |
| `QWEN_GDN_ROWS1` | 关（`=1`） | GDN prefill 递推：每个状态行一个 lane，而不是四个。 | L1 |
| `QWEN_QSA_ATTN_VER` | tensor core，fp32 级精度（`attn_tcp`） | decode QSA 注意力 kernel：`t` = fp16 操作数的 tensor core kernel（精度较低），`3` / `2` = 流水线标量 kernel，与原标量 kernel 逐位相同。 | L1 |
| `QWEN_QSA_ATTN_V1` | 关（`=1`） | decode QSA 注意力：原标量 kernel。 | 相对默认为 L1 |
| `QWEN_QSA_SCORE_V1` | 关（`=1`） | indexer 打分：标量 kernel，不用 tensor core。 | L1 |
| `QWEN_QSA_SELECT_V1` | 关（`=1`） | indexer top-k 选择：块扫描 kernel，不用 warp ballot kernel。 | L0 |
| `QWEN_PREFILL_ATTN_SPLIT` | 关（`=1`） | prefill QSA 注意力：分段 tensor core kernel + combine，不用 flash 式 kernel。 | L1 |
| `QWEN_PREFILL_ATTN_SCALAR` | 关（`=1`） | prefill QSA 注意力：使用 decode 行的 kernel（由 `QWEN_QSA_ATTN_VER` / `QWEN_QSA_ATTN_V1` 选择）+ combine，不用 flash 式 kernel。 | L1 |
| `QWEN_MOE_UNFUSED` | 关（`=1`） | MoE decode：分组 / finish / 激活 / 变换用分开的 kernel，不用融合版。 | L0 |
| `QWEN_SHARED_UNFUSED` | 关（`=1`） | decode：shared expert 的逐元素算子用 torch 算子、router 输出先复制一份，shared expert 也不走多线性层合并启动。 | L0 |
| `QWEN_MOE_YD16` | 关（`=1`） | 路由专家的输出以 fp16 存储（down GEMM 的写入减半）。prefill −3.2%。 | L3 |
| `QWEN_MOE_GEMV` | 未设置 | 持久化 MoE decode GEMV 调度的覆盖字符串（`tpw,pf,warps,s1,s2,ctas,apf,fused`）。会被解析，但 Qwen 的 MoE 不走持久化路径，所以在这里不起作用。 | — |
| `QWEN_MOE_ROUTE_SIDE` | 关（`=1`） | decode：router GEMV + top-k 放到旁路流，与 shared expert 的 EXL3 GEMV 并行。没有可测收益。 | L0 |
| `QWEN_GDN_AB_SIDE` | 关（`=1`） | decode：GDN a/b 投影放到旁路流，与 qkv / z 的 EXL3 GEMV 并行。没有可测收益。 | L0 |
| `QWEN_COMMIT_SIDE` | 关 | 实验性：`1` 把状态 commit 放到旁路流与 MTP 链并行（下一次 verify 时汇合）；`2` 立即汇合（诊断用）。`1` 在部分运行中改变了验证后的输出（竞争原因未查明），不要使用。 | 不确定 |
| `QWEN_DRAFT_Q4_TILED` | 开（`=0` 关） | Q4 草稿头用平铺布局（warp 一次读连续 512 B）；`0`：按行布局。 | L0 |
| `QWEN_DRAFT_ALTS` | 未设置 | 加载时额外构建的草稿头，用于同进程 A/B，例如 `q4:65536;q8:49152`（`q4` 平铺，`q4r` 按行布局，`q8`；数字含义同 `QWEN_DRAFT_VOCAB`）。 | L2 |
| `QWEN_DRAFT_PICK` | 未设置 | `i` 选 `QWEN_DRAFT_ALTS[i-1]` 作为草稿头（在捕获图时读取）。 | L2 |
| `QWEN_VIS_EXACT` | 开（`=0` 关） | 视觉塔的 EXL3 线性层以 fp32 取 GEMM 输出（cuBLASLt）并用 fp32 finish；`0`：GEMM 输出 fp16（误差大得多）。编码时间 +3–10%。 | L1 |
| `QWEN_VIS_ATT32` | 关（`=1`） | 视觉注意力用 fp32，不用 fp16 flash（精度略高，时间 +55%）。 | L1 |

## 诊断与开发工具

供 `qwen-gen`、`qwen-spec`、`qwen-batch` 探测工具和计时使用；服务时不需要。

| 变量 | 含义 |
| --- | --- |
| `QWEN_TIMING` | 设置后：`qwen-spec` / 单序列生成在草稿、verify、commit、catch-up 前后同步，按阶段累计毫秒。 |
| `QWEN_SERVE_TIMING` | 设置后：服务端每 100 轮按批大小打印批量轮各阶段的同步计时（会拖慢服务）。 |
| `QWEN_MOE_DUMP` | `n`：把第 n 次 prefill MoE 调用的输入行写到 `/tmp/moe_x_<n>.f32`（用于 kernel 基准）。 |
| `QWEN_STATE_DUMP` | `dir`：配合 `QWEN_TAIL_ARMS`，把 prefill 后的序列状态按层和张量各写一个 fp32 文件。 |
| `QWEN_DUMP_LAST` | `qwen-gen`：导出 prompt logits 时只保留最后 n 行（默认全部）。 |
| `QWEN_CAP` | `qwen-gen`：序列容量（默认 8192，不够时自动加大到 prompt + 输出）。 |
| `QWEN_PREFILL_STEP` | 设置后：`qwen-gen` 按 16 行一段 prefill（走 decode / verify kernel），不走 prefill 路径。 |
| `QWEN_TAIL` | `qwen-gen`：先 prefill 除最后 n 个 token 外的部分，再一次跑这 n 个并输出全部 logits（用来与参考对比）。配合 `QWEN_TAIL_ARMS` 时默认 512。 |
| `QWEN_TAIL_STEP` | `qwen-gen`：尾部按 n 行一段运行（走 decode / verify kernel，相当于 n−1 个草稿的推测轮）；默认整段一次。 |
| `QWEN_TAIL_ARMS` | `qwen-gen`：`"A;B;..."`，每项是逗号分隔的 `VAR=value`；在一个进程里对每个臂、逗号分隔的每个 id 文件跑尾部，并把 logits 写到 `<dump>_<臂>_<提示>.f32`。 |
| `QWEN_PREFILL_REPS` | `qwen-gen`：正式测量前先做 n−1 次计时的预热 prefill。 |
| `QWEN_PREFILL_AB` | `qwen-gen`：预热各次交替设置 `VAR=1` / 取消（可用逗号列出多个；`VAR=value` 项设为 value），做同进程 A/B。 |
| `QWEN_SKIP_PLAIN` | 设置后：`qwen-spec` 跳过普通贪心的对照运行。 |
| `QWEN_GRAPH` | `qwen-spec`：`0` 时推测解码以 eager 模式运行，不用 CUDA 图。 |
| `QWEN_BATCH_ARMS` | `qwen-batch`：`"A;B;..."`，每项是逗号分隔的 `VAR=value`；在同一进程里每个臂跑一次（重新建序列、重新捕获图）。 |
| `QWEN_BATCH_LOG` | `qwen-batch`：每个序列每轮写一行 JSON 的文件（`conf`、`acc`、验证通过的草稿数 `w`）。 |
| `QWEN_BATCH_NOLAP` | 设置后：`qwen-batch` 不插入阶段同步（用于 nsys 分析）。 |
| `QWEN_HOST_ACCEPT` | 设置后：`qwen-batch` 用主机端 accept（verify、argmax 拷贝、由主机写 commit / catch-up 输入），不用设备端 accept。 |
