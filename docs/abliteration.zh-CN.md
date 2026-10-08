# Abliteration（可选，默认关闭）

"Abliteration" 通过编辑或投影那些沿"拒绝方向"写入残差流的权重/激活，移除模型拒绝请求的倾向。它会改变模型的行为：经过 abliteration 的模型会回答原始模型会拒绝的请求，**包括有害的请求**。这里的一切都**默认关闭**，引擎中没有任何东西会隐式启用它。

引擎提供两种独立的机制，可以单独使用也可以一起使用，二者都通过环境变量选择性启用：

| 机制 | 开关 | 模型 | 作用 |
| --- | --- | --- | --- |
| o_proj 移植 | `GLM53_ABLIT` | GLM-5.3-Flash | 从同一模型的一个经过 abliteration 的捐赠者检查点加载第 15–44 层注意力的 `o_proj`（逐字节复制，带校验），替代原始权重。 |
| 方向消融 | `SPARK_ABLATE` | GLM, Qwen | 从每一层写入残差流的所有内容中移除一个或多个拒绝方向（由引擎从你自己的提示集计算得出）。 |

## 1. o_proj 移植（GLM-5.3-Flash）

从一个经过 abliteration 的捐赠者检查点加载第 15–44 层的注意力输出投影；所有其他权重保持为原始 EXL3 检查点。在加载时、张量并行分片之前应用，因此每条路径看到的都是编辑后的权重。

```bash
# fetch the donor's o_proj tensors (~2.7 GB; range requests) + MANIFEST.json (key/dtype/shape/sha256)
.venv/bin/python scripts/fetch_ablit_transplant.py /models/glm53-ablit-transplant \
    --donor <abliterated GLM-5.3-Flash donor on HF> --layers 15-44
# spark.env (both nodes)
GLM53_ABLIT=1
GLM53_ABLIT_DIR=/models/glm53-ablit-transplant
# GLM53_ABLIT_LAYERS=15-44   (default)
```

每一层都会记录 `[ablit] layer N o_proj ... transplanted (sha256 ok)`；任何缺失的文件、尺寸、dtype、形状或校验和不匹配都会中止加载（绝不会静默回退到不同的模型）。性能不变。

## 2. 方向消融（`SPARK_ABLATE`，GLM 和 Qwen）

引擎从两组提示（一个经过安全调优的模型会拒绝的提示，以及普通提示）计算出一个拒绝方向，然后将该方向从所选层的每一次残差流写入中投影出去：非量化权重在加载时被正交化（`W -= DᵀD·W`），量化投影在运行时被投影。当 `SPARK_ABLATE` 未设置时，这条路径完全是空操作（no-op）。

### 计算一个方向

提供两个 JSON 文件，每个都是一个已被标记化为 id 列表的提示列表（应用模型的聊天模板并带上生成提示；关闭思考）。你提供自己的提示——引擎不捆绑任何提示。然后：

```bash
# Qwen (one node)
./bin/spark-engine qwen-refusal-dir <model_dir> harmful_ids.json harmless_ids.json dirs.safetensors
# GLM (both TP2 ranks; rank 0 writes the file)
bash engine-rs/serve/glm-rank.sh 0 dirs ...   # or run `spark-engine glm-refusal-dir <model> harmful_ids.json harmless_ids.json dirs.safetensors` on each rank
```

`dirs.safetensors` 保存了 `directions` [layers, hidden]、`directions_pos` [layers, positions, hidden]、`scores` [layers]（每层的分离度）以及自动选定的 `direction`。日志会打印每一层的分离度分数。

### 启用

```bash
SPARK_ABLATE=/path/dirs.safetensors          # set in spark.env or the start script's environment
# SPARK_ABLATE_MODE=single                   # default: the auto-selected single direction
# SPARK_ABLATE_MODE=per-layer                # each layer removes its own direction
# SPARK_ABLATE_MODE=subspace:16-40:8         # SVD of layers 16-40's directions, remove the top 8 everywhere
# SPARK_ABLATE_LAYERS=a-b                     # limit the layers it applies to (default all)
# SPARK_ABLATE_FROM_LAYER=n                   # single mode: use directions[n]
# SPARK_ABLATE_POSITIONS=8                    # positions per prompt used when computing directions
```

单个方向往往较弱；通常需要一个小的子空间（例如 `subspace:16-40:8`）才能大幅降低拒绝率而不损害普通回答。请自行验证效果，并检查回答质量，而不仅仅是拒绝率。

## 风险与责任

- 编辑后的模型不再拒绝有害、危险或非法的请求。不要将这样的服务器暴露给不应拥有不受限制访问权限的用户，并遵守基础模型和任何捐赠者的许可证及可接受使用条款。
- Abliteration 可能会降低某些任务上的质量；请针对你的使用场景进行评估。
- 捐赠者检查点是第三方成果；请阅读其 model card。本仓库不分发任何捐赠者权重，也不分发任何提示集。
