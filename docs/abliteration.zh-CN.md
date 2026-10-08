# Abliteration（可选，默认关闭）

“Abliteration”通过编辑沿“拒答方向”写入残差流的权重，去除模型拒绝请求的倾向。它会改变模型的行为：经过 abliteration
的模型会回答原模型拒绝的请求，包括有害请求。该功能**默认关闭**，本仓库中没有任何内容会隐式启用它。

## GLM-5.3-Flash：o_proj 移植

引擎可以从同一模型的 abliterated 供体 checkpoint 中加载第 15–44 层的注意力输出投影（`o_proj`），替换原始的投影
（即已公开的“o_proj transplant”编辑；其余所有权重仍来自原始 EXL3 checkpoint）。供体张量为 BF16，在加载时、张量并行切分之前应用，
因此所有计算路径都使用编辑后的权重。

1. 拉取供体张量（约 2.7 GB；仅通过 HTTP range 请求下载这些张量）：

   ```bash
   .venv/bin/python scripts/fetch_ablit_transplant.py /models/glm53-ablit-transplant \
       --donor dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 --layers 15-44
   ```

   脚本会写出 `L15.bin … L44.bin` 和 `MANIFEST.json`（张量名、dtype、形状、大小、SHA-256）。
2. 在 `spark.env` 中启用（两个节点）：

   ```bash
   GLM53_ABLIT=1
   GLM53_ABLIT_DIR=/models/glm53-ablit-transplant
   # GLM53_ABLIT_LAYERS=15-44   (default)
   ```

3. 重启（`engine-rs/serve/start.sh glm`）。rank 日志会为每一层打印一行 `[ablit] layer N o_proj ... transplanted (sha256 ok)`。
   任何文件缺失，或大小、dtype、形状、校验和不匹配，都会中止加载：引擎绝不会静默回退到其他模型。

性能不变（移植的张量一对一替换原始张量）。

## 风险与责任

- 编辑后的模型不再拒绝有害、危险或违法的请求。不要将这样的服务开放给不应拥有无限制访问权限的用户，并遵守基础模型和供体模型的
  许可证及可接受使用条款。
- Abliteration 可能降低某些任务上的质量；请针对你的使用场景进行评估。
- 供体 checkpoint 是第三方作品；请阅读其模型卡。本仓库不分发其任何权重。
