# Abliteration (optional, off by default)

"Abliteration" removes a model's tendency to refuse requests by editing the weights that write into the residual stream
along a "refusal direction". It changes the model's behavior: an abliterated model will answer requests the original
model declines, including harmful ones. It is **off by default** and nothing in this repository enables it implicitly.

## GLM-5.3-Flash: o_proj transplant

The engine can load the attention output projections (`o_proj`) of layers 15–44 from an abliterated donor checkpoint
of the same model instead of the original ones (the published "o_proj transplant" edit; all other weights stay the
original EXL3 checkpoint). The donor tensors are BF16 and are applied at load time, before tensor-parallel sharding, so
every compute path uses the edited weights.

1. Fetch the donor tensors (about 2.7 GB; only these tensors are downloaded, with HTTP range requests):

   ```bash
   .venv/bin/python scripts/fetch_ablit_transplant.py /models/glm53-ablit-transplant \
       --donor dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 --layers 15-44
   ```

   The script writes `L15.bin … L44.bin` and `MANIFEST.json` (tensor name, dtype, shape, size, SHA-256).
2. Enable it in `spark.env` (both nodes):

   ```bash
   GLM53_ABLIT=1
   GLM53_ABLIT_DIR=/models/glm53-ablit-transplant
   # GLM53_ABLIT_LAYERS=15-44   (default)
   ```

3. Restart (`engine-rs/serve/start.sh glm`). The rank logs print one `[ablit] layer N o_proj ... transplanted (sha256 ok)`
   line per layer. Any missing file, size, dtype, shape or checksum mismatch stops the load: the engine never falls back
   silently to a different model.

Performance is unchanged (the transplanted tensors replace the originals one for one).

## Risks and responsibility

- The edited model no longer refuses harmful, dangerous or illegal requests. Do not expose such a server to users who
  should not have unrestricted access, and comply with the licenses and acceptable-use terms of the base model and the
  donor.
- Abliteration can lower quality on some tasks; evaluate it for your use case.
- The donor checkpoint is third-party work; read its model card. This repository does not distribute any of its weights.
