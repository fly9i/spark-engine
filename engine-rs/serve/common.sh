# Shared by the serve scripts: settings from spark.env (SPARK_ENV, default <repo>/spark.env).
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
SPARK_ENV=${SPARK_ENV:-$repo/spark.env}
[[ -f $SPARK_ENV ]] || { echo "settings file $SPARK_ENV missing: copy spark.env.example to spark.env and edit it" >&2; exit 1; }
set -a; source "$SPARK_ENV"; set +a
SPARK_HOME=${SPARK_HOME:-$repo}
SPARK_PORT=${SPARK_PORT:-8888}
py=${SPARK_PYTHON:-python3}
# Multi-node layout (GLM TP2 / TP3, DeepSeek-V4.1-Flash TP3). Rank 0 runs here; WORKERS[r] is rank r's ssh target and
# MASTERS[r] rank r's address of rank 0 (each worker reaches rank 0 over its own RoCE link). SPARK_WORKERS /
# SPARK_MASTER_ADDRS list ranks 1.. and 0.. respectively; the older two-node GLM_WORKER / GLM_MASTER_ADDR still work.
read -ra _w <<< "${SPARK_WORKERS:-${GLM_WORKER:-}}"; WORKERS=(local "${_w[@]}")
read -ra MASTERS <<< "${SPARK_MASTER_ADDRS:-}"
(( ${#MASTERS[@]} )) || MASTERS=("${GLM_MASTER_ADDR:-}" "${GLM_MASTER_ADDR:-}" "${GLM_MASTER_ADDR:-}")
unset _w
ssho=(-o ControlMaster=auto -o ControlPath=/tmp/spark-ssh-%r@%h:%p -o ControlPersist=60)
# wait_memory WORLD: after stopping an engine, wait (up to 2 min) until every node has its memory back (MemAvailable >=
# SPARK_START_MIN_GIB, 100): a rank started while its node still frees the old process stalls in its CUDA / NCCL set-up
wait_memory() {
  local world=$1 min=${SPARK_START_MIN_GIB:-100} r a ok
  for _ in $(seq 240); do
    ok=1
    for ((r = 0; r < world; r++)); do
      if (( r == 0 )); then a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
      else a=$(ssh "${ssho[@]}" "${WORKERS[$r]}" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null || echo "$min"); fi
      (( a >= min )) || ok=0
    done
    (( ok )) && return 0
    sleep 0.5
  done
  echo "MemAvailable stays below $min GiB on some node; starting anyway" >&2
}
# need_nodes WORLD: every rank has a worker and an address of rank 0, and runs the same engine binary as this node
need_nodes() {
  local world=$1 r s0 s
  for ((r = 0; r < world; r++)); do
    [[ -n ${MASTERS[$r]:-} ]] || { echo "no address of rank 0 for rank $r: set SPARK_MASTER_ADDRS in $SPARK_ENV" >&2; return 1; }
    (( r == 0 )) || [[ -n ${WORKERS[$r]:-} ]] || { echo "no node for rank $r: set SPARK_WORKERS in $SPARK_ENV" >&2; return 1; }
  done
  s0=$(sha256sum "$SPARK_BIN" | cut -d' ' -f1)
  for ((r = 1; r < world; r++)); do
    s=$(ssh "${ssho[@]}" "${WORKERS[$r]}" "sha256sum $SPARK_BIN" | cut -d' ' -f1)
    [[ $s0 == "$s" ]] || { echo "engine binary differs on rank $r (${WORKERS[$r]}: $SPARK_BIN)" >&2; return 1; }
  done
}
# Compact physical memory on the given nodes once the engines hold their working set: on GB10's unified memory the
# kernel's proactive compaction otherwise migrates pages in bursts during the first requests (SPARK_COMPACT=0 skips).
compact_nodes() {
  [[ ${SPARK_COMPACT:-1} != 0 ]] || return 0
  local world=$1 r c
  (echo 1 > /proc/sys/vm/compact_memory) 2> /dev/null & c=$!
  for ((r = 1; r < world; r++)); do ssh "${ssho[@]}" "${WORKERS[$r]}" "echo 1 > /proc/sys/vm/compact_memory" 2> /dev/null || true; done
  wait $c || true
}
