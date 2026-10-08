# Abliteration (optional, off by default)

"Abliteration" removes a model's tendency to refuse requests by editing or projecting the weights/activations that write
into the residual stream along a "refusal direction". It changes the model's behavior: an abliterated model will answer
requests the original model declines, **including harmful ones**. Everything here is **off by default** and nothing in
the engine enables it implicitly.

The engine has two independent mechanisms, usable alone or together, both opt-in via environment variables:

| Mechanism | Switch | Models | What it does |
| --- | --- | --- | --- |
| o_proj transplant | `GLM53_ABLIT` | GLM-5.3-Flash | Loads layers 15–44's attention `o_proj` from an abliterated donor checkpoint of the same model (byte copy, verified) instead of the originals. |
| Direction ablation | `SPARK_ABLATE` | GLM, Qwen | Removes one or more refusal directions (computed by the engine from your own prompt sets) from everything each layer writes into the residual stream. |

## 1. o_proj transplant (GLM-5.3-Flash)

Loads the attention output projection of layers 15–44 from an abliterated donor checkpoint; all other weights stay the
original EXL3 checkpoint. Applied at load time before tensor-parallel sharding, so every path sees the edited weights.

```bash
# fetch the donor's o_proj tensors (~2.7 GB; range requests) + MANIFEST.json (key/dtype/shape/sha256)
.venv/bin/python scripts/fetch_ablit_transplant.py /models/glm53-ablit-transplant \
    --donor <abliterated GLM-5.3-Flash donor on HF> --layers 15-44
# spark.env (both nodes)
GLM53_ABLIT=1
GLM53_ABLIT_DIR=/models/glm53-ablit-transplant
# GLM53_ABLIT_LAYERS=15-44   (default)
```

Each layer logs `[ablit] layer N o_proj ... transplanted (sha256 ok)`; any missing file, size, dtype, shape or checksum
mismatch aborts the load (never a silent fallback to a different model). The transplanted weights have the original
shape and dtype, so a decode step costs the same; see [Performance](#performance) for its effect on speculative decoding.

## 2. Direction ablation (`SPARK_ABLATE`, GLM and Qwen)

The engine computes a refusal direction from two sets of prompts (ones a safety-tuned model refuses, and ordinary ones),
then projects that direction out of every residual-stream write of the selected layers: non-quantized weights are
orthogonalized at load (`W -= DᵀD·W`), quantized projections are projected at run time. With `SPARK_ABLATE` unset the
path is a complete no-op.

### Compute a direction

Provide two JSON files, each a list of prompts already tokenized to id lists (apply the model's chat template with the
generation prompt; thinking off). You supply your own prompts — the engine does not bundle any. Then:

```bash
# Qwen (one node)
./bin/spark-engine qwen-refusal-dir <model_dir> harmful_ids.json harmless_ids.json dirs.safetensors
# GLM (both TP2 ranks; rank 0 writes the file)
bash engine-rs/serve/glm-rank.sh 0 dirs ...   # or run `spark-engine glm-refusal-dir <model> harmful_ids.json harmless_ids.json dirs.safetensors` on each rank
```

`dirs.safetensors` holds `directions` [layers, hidden], `directions_pos` [layers, positions, hidden], `scores` [layers]
(separation per layer) and the auto-selected `direction`. The logs print each layer's separation score.

### Enable

```bash
SPARK_ABLATE=/path/dirs.safetensors          # set in spark.env or the start script's environment
# SPARK_ABLATE_MODE=single                   # default: the auto-selected single direction
# SPARK_ABLATE_MODE=per-layer                # each layer removes its own direction
# SPARK_ABLATE_MODE=subspace:16-40:8         # SVD of layers 16-40's directions, remove the top 8 everywhere
# SPARK_ABLATE_LAYERS=a-b                     # limit the layers it applies to (default all)
# SPARK_ABLATE_FROM_LAYER=n                   # single mode: use directions[n]
# SPARK_ABLATE_POSITIONS=8                    # positions per prompt used when computing directions
```

A single direction is often weak; a small subspace (e.g. `subspace:16-40:8`) is usually needed to drop the refusal rate
substantially without hurting ordinary answers. Validate the effect yourself and check answer quality, not just refusal
rate.

## Performance

GLM-5.3-Flash, 1 stream, greedy, 400-token English technical prose, same binary, `SPARK_ABLATE_MODE=subspace:16-40:8`:

| Setting | Decode tok/s | Tokens per decode step | Step time |
| --- | --- | --- | --- |
| No ablation | 46.9 | 2.88 | 61.4 ms |
| `SPARK_ABLATE` | 47.4 | 2.88 | 60.7 ms |
| `SPARK_ABLATE` + `GLM53_ABLIT=1` | 45.7 | 2.76 | 60.4 ms |

- `SPARK_ABLATE` removes all k directions of a residual write in one kernel (each row read once), so its run-time cost is
  within measurement noise; on ordinary prompts it leaves the output, and therefore the draft acceptance, unchanged.
- `GLM53_ABLIT` adds no compute, but the transplanted `o_proj` also changes ordinary outputs, so the DFlash2 drafter
  (trained on the original model) is accepted less often: about 4% fewer tokens per step, about 2.6% lower decode speed.

Recommended: `SPARK_ABLATE` alone, adding `GLM53_ABLIT` only if your own evaluation shows you need its extra effect.

## Risks and responsibility

- The edited model no longer refuses harmful, dangerous or illegal requests. Do not expose such a server to users who
  should not have unrestricted access, and comply with the base model's and any donor's licenses and acceptable-use terms.
- Abliteration can lower quality on some tasks; evaluate it for your use case.
- A donor checkpoint is third-party work; read its model card. This repository distributes no donor weights and no
  prompt sets.
