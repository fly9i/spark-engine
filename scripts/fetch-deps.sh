#!/usr/bin/env bash
# Fetch the pinned third-party sources the build needs into third_party/ (not part of this repository):
#   - torch-sys 0.20.0 (tch-rs, MIT/Apache-2.0) from crates.io, patched for libtorch 2.13 (patches/torch-sys-*.patch)
#   - NCCL 2.29.7 public header template (NVIDIA, BSD-3-Clause) matching the libtorch 2.13 build ABI
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tp=$root/third_party
mkdir -p "$tp"
if [[ ! -f $tp/torch-sys-0.20.0/.patched ]]; then
  rm -rf "$tp/torch-sys-0.20.0"
  curl -fsSL -A "spark-engine-build (https://github.com/fly9i/spark-engine)" \
    https://crates.io/api/v1/crates/torch-sys/0.20.0/download | tar xz -C "$tp"
  patch -d "$tp/torch-sys-0.20.0" -p1 < "$root/patches/torch-sys-0.20.0-libtorch-2.13.patch"
  touch "$tp/torch-sys-0.20.0/.patched"
fi
mkdir -p "$tp/nccl"
if [[ ! -f $tp/nccl/nccl.h.in ]]; then
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/nccl/v2.29.7-1/src/nccl.h.in -o "$tp/nccl/nccl.h.in"
  curl -fsSL https://raw.githubusercontent.com/NVIDIA/nccl/v2.29.7-1/LICENSE.txt -o "$tp/nccl/LICENSE.txt"
fi
echo "third_party ready: $(ls "$tp" | tr '\n' ' ')"
