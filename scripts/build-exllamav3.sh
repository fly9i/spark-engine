#!/usr/bin/env bash
# Build the exllamav3 v1.4.9 CUDA extension (turboderp-org/exllamav3, MIT) that the GLM path links against
# (exl3_gemm_gr, exl3_mgemm, reconstruct, had_r_128). Output: third_party/exllamav3/lib/libexllamav3_ext.so (+ LICENSE).
# Usage: PYTHON=/path/to/venv/bin/python scripts/build-exllamav3.sh   (the Python environment with torch 2.13 cu130)
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
py=${PYTHON:-python3}
src=$root/third_party/exllamav3-src
out=$root/third_party/exllamav3/lib
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$(dirname "$(command -v "$py")"):$PATH   # nvcc; ninja from the Python environment (parallel build)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
if [[ ! -d $src/.git ]]; then
  git clone -q https://github.com/turboderp-org/exllamav3 "$src"
fi
git -C "$src" checkout -q -f v1.4.9
git -C "$src" clean -qfdx -e build
# only the CUDA parts the engine links (no x86 CPU kernels, no Python bindings): scripts/exllamav3_min_setup.py
cp "$root/scripts/exllamav3_min_setup.py" "$src/"
(cd "$src" && "$py" exllamav3_min_setup.py build_ext --inplace)
mkdir -p "$out"
cp "$src"/exllamav3_ext*.so "$out/libexllamav3_ext.so"
cp "$src/LICENSE" "$out/../LICENSE"
echo "built $out/libexllamav3_ext.so"
