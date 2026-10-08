#!/usr/bin/env bash
# Download the model checkpoints with the Hugging Face CLI (pip install -U "huggingface_hub[cli]").
# Usage: download-models.sh glm|qwen [DEST]   (DEST default /models). Read each model's license first.
set -euo pipefail
dest=${2:-/models}
case ${1:-} in
  glm)
    hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw --revision 5ab363a8dcf6405955fd5f99671e01a1c9fb124b --local-dir "$dest/GLM-5.3-Flash-tr3-4bpw"
    # speculative drafter, CC BY-NC-ND 4.0 (non-commercial)
    hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a --local-dir "$dest/GLM-5.3-Flash-DFlash2" ;;
  qwen)
    hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6 --local-dir "$dest/Qwen3.8-Flash-Next-exl3" ;;
  *) echo "usage: download-models.sh glm|qwen [DEST]" >&2; exit 2 ;;
esac
