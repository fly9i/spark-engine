#!/usr/bin/env bash
# One engine rank of the GLM-5.3-Flash server. Usage: glm-rank.sh RANK SOCKET [WORLD [MASTER_ADDR]]
# Order: the caller's GLM53_* variables are dropped, the profile is loaded, then spark.env (its GLM53_* overrides win), then
# the topology env of the world size (profiles/tp<world>-rdma.env: transport settings), then GLM_EXTRA_ENV.
set -euo pipefail
for v in ${!GLM53_@}; do unset "$v"; done
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rank=${1:?rank 0, 1 (or 2 under TP3)}; socket=${2:?socket path}; world=${3:-${GLM_WORLD:-2}}; master=${4:-${MASTERS[$rank]}}
export SPARK_PYTHON=$py
source "$SPARK_HOME/scripts/env.sh"
source "${GLM_PROFILE:-$SPARK_HOME/engine-rs/profiles/glm.env}"
export GLM53_PCACHE_DIR=$SPARK_HOME/var/prefix-cache
set -a; source "$SPARK_ENV"; set +a
source "$SPARK_HOME/engine-rs/profiles/tp$world-rdma.env"
kv=${GLM_KV_TOKENS:-1048576}
export GLM53_TP_RANK=$rank GLM53_TP_WORLD=$world GLM53_MASTER_ADDR=${master:?address of rank 0} GLM53_MASTER_PORT=${GLM_MASTER_PORT:-29931}
export GLM53_MAX_CONTEXT=$kv GLM53_SERVE_KV_TOKENS=$kv GLM53_SERVE_KV_GRANULE=${GLM_KV_GRANULE:-16384} GLM53_NCCL_TIMEOUT_S=1800
export CUDA_VISIBLE_DEVICES=0
for kv2 in ${GLM_EXTRA_ENV:-}; do export "$kv2"; done   # diagnostics / overrides, e.g. GLM_EXTRA_ENV="GLM53_SERVE_LOG=1"
mkdir -p "$GLM53_PCACHE_DIR"
exec "$SPARK_BIN" serve "$GLM_MODEL" "$GLM_DRAFTER" "$socket"
