#!/usr/bin/env bash
# One engine rank of the GLM-5.3-Flash TP2 server. Usage: glm-rank.sh RANK SOCKET
# Order: the caller's GLM53_* variables are dropped, the profile is loaded, then spark.env again (its GLM53_* overrides win).
set -euo pipefail
for v in ${!GLM53_@}; do unset "$v"; done
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rank=${1:?rank 0 or 1}; socket=${2:?socket path}
export SPARK_PYTHON=$py
source "$SPARK_HOME/scripts/env.sh"
source "${GLM_PROFILE:-$SPARK_HOME/engine-rs/profiles/glm-tp2.env}"
export GLM53_PCACHE_DIR=$SPARK_HOME/var/prefix-cache
set -a; source "$SPARK_ENV"; set +a
kv=${GLM_KV_TOKENS:-1048576}
export GLM53_TP_RANK=$rank GLM53_TP_WORLD=2 GLM53_MASTER_ADDR=${GLM_MASTER_ADDR:?} GLM53_MASTER_PORT=${GLM_MASTER_PORT:-29931}
export GLM53_MAX_CONTEXT=$kv GLM53_SERVE_KV_TOKENS=$kv GLM53_SERVE_KV_GRANULE=${GLM_KV_GRANULE:-16384} GLM53_NCCL_TIMEOUT_S=1800
export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0 NCCL_CROSS_NIC=0
export NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX CUDA_VISIBLE_DEVICES=0
for kv2 in ${GLM_EXTRA_ENV:-}; do export "$kv2"; done   # diagnostics / overrides, e.g. GLM_EXTRA_ENV="GLM53_SERVE_LOG=1"
mkdir -p "$GLM53_PCACHE_DIR"
exec "$SPARK_BIN" serve "$GLM_MODEL" "$GLM_DRAFTER" "$socket"
