#!/usr/bin/env bash
# One engine rank of the DeepSeek-V4.1-Flash server (three DGX Sparks). Usage: ds41-rank.sh RANK SOCKET MASTER_ADDR
# Every rank reads the same spark.env, so all three run with the same DS41_* settings (the prefix cache requires it).
set -euo pipefail
for v in ${!GLM53_@}; do unset "$v"; done
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rank=${1:?rank 0, 1 or 2}; socket=${2:?socket path}; master=${3:?address of rank 0}
export SPARK_PYTHON=$py
source "$SPARK_HOME/scripts/env.sh"
export GLM53_TP_RANK=$rank GLM53_TP_WORLD=3 GLM53_MASTER_ADDR=$master GLM53_MASTER_PORT=${DS41_MASTER_PORT:-29961}
# transport: NCCL only bootstraps over sockets; every collective goes over the engine's N-way RDMA (subnet-paired links)
export NCCL_SOCKET_IFNAME=${SPARK_TP3_IFNAME:?set SPARK_TP3_IFNAME in spark.env} NCCL_IB_DISABLE=1 GLM53_RDMA_ARN_INIT=1 GLM53_RDMA_AR=1
# prefill: RDMA row reduce-scatter / all-gather depth 12 in flight, the communication stream at the highest priority,
# MoE down-projection output buffer up to 1024 MiB (8192-row chunks in one pass, the combine fused into the scatter)
export GLM53_RDMA_ARN_ROWS_PIPE=${GLM53_RDMA_ARN_ROWS_PIPE:-12} GLM53_COMM_STREAM_PRIO=${GLM53_COMM_STREAM_PRIO:-1} DS41_MOE_OUT_MB=${DS41_MOE_OUT_MB:-1024}
# serving: 4 sequences, DSpark speculative decoding (the draft stages take ~2.7 GiB per rank), NVMe prefix cache, image input
# (rank 0 loads the vision encoder, ~1 GiB). Small per-capture cuBLAS workspaces shared by the graphs.
export DS41_SERVE_STORES=${DS41_SERVE_STORES:-4} DS41_MTP=${DS41_MTP:-1} DS41_SPEC=${DS41_SPEC:-1} DS41_PCACHE=${DS41_PCACHE:-1} \
  DS41_VISION=${DS41_VISION:-1} DS41_PCACHE_DIR=${DS41_PCACHE_DIR:-$SPARK_HOME/var/ds41-prefix-cache}
export CUBLAS_WORKSPACE_CONFIG=:16:8 GLM53_GRAPH_SHARED_WS=1
# host memory: after the warm-up the checkpoint's file pages are dropped from the process and glibc's free pages returned;
# glibc malloc gets fixed mmap / trim thresholds and at most 8 arenas (by default each prefill chunk's ~100 MiB of engram
# buffers stayed in the arenas when freed and host RSS grew by GBs). DS41_MALLOC_TUNE=0: glibc defaults.
export GLM53_RELEASE_FILE_CACHE=${DS41_RELEASE_FILE_CACHE:-1} GLM53_HOST_TRIM=${DS41_HOST_TRIM:-1}
[[ ${DS41_MALLOC_TUNE:-1} == 1 ]] && export GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=33554432:glibc.malloc.arena_max=8
# optional conditional-clamp ablation (off unless DS41_ABLATE_CLAMP names a calibration file; docs/abliteration.md)
export SPARK_ABLATE_CLAMP=${DS41_ABLATE_CLAMP:-} SPARK_ABLATE_CLAMP_MIN=${DS41_ABLATE_MIN:-12}
export CUDA_VISIBLE_DEVICES=0
for kv in ${DS41_EXTRA_ENV:-}; do export "$kv"; done   # diagnostics / overrides, e.g. DS41_EXTRA_ENV="DS41_STEP_STATS=1"
[[ $DS41_PCACHE == 0 ]] || mkdir -p "$DS41_PCACHE_DIR"
cd "$SPARK_HOME"
exec "$SPARK_BIN" ds41-serve "${DS41_SHARDS:?}/rank$rank" "$socket" "${DS41_ENGRAM_HASH:?}"
