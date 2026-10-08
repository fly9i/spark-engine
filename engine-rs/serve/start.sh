#!/usr/bin/env bash
# Run one of the two models on port $SPARK_PORT (they share it): start.sh glm | qwen | status | stop
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
running() {
  if pgrep -f " [s]erve .* /tmp/spark-glm.sock" > /dev/null; then echo glm
  elif pgrep -f "[q]wen-serve .* /tmp/spark-qwen.sock" > /dev/null; then echo qwen
  else echo none; fi
}
wait_memory() {   # unified memory: give the previous model's pages time to return
  for _ in $(seq 120); do (( $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) >= ${SPARK_MIN_FREE_GIB:-100} )) && return; sleep 1; done
}
case ${1:-${SPARK_MODEL:-status}} in
  glm)  [[ $(running) == qwen ]] && { bash "$here/stop-qwen.sh"; wait_memory; }; exec bash "$here/start-glm.sh" ;;
  qwen) [[ $(running) == glm ]] && { bash "$here/stop-glm.sh"; wait_memory; }; exec bash "$here/start-qwen.sh" ;;
  stop) bash "$here/stop-glm.sh" > /dev/null 2>&1; bash "$here/stop-qwen.sh" ;;
  status) running ;;
  *) echo "usage: start.sh glm|qwen|status|stop" >&2; exit 2 ;;
esac
