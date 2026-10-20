#!/usr/bin/env bash
# Run one model on port $SPARK_PORT (they share it and rank 0's GPU, so whatever runs is stopped first):
#   start.sh glm     GLM-5.3-Flash over GLM_WORLD Sparks (spark.env; default 2)
#   start.sh glm3    GLM-5.3-Flash over three Sparks (GLM_WORLD=3)
#   start.sh ds41    DeepSeek-V4.1-Flash over three Sparks
#   start.sh qwen    Qwen3.8-Flash-Next on this Spark
#   start.sh status | stop
# The model can also come from SPARK_MODEL.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
running() {
  if pgrep -f " [s]erve .* /tmp/spark-glm.sock" > /dev/null; then echo glm
  elif pgrep -f "[d]s41-serve .* /tmp/spark-ds41.sock" > /dev/null; then echo ds41
  elif pgrep -f "[q]wen-serve .* /tmp/spark-qwen.sock" > /dev/null; then echo qwen
  else echo none; fi
}
wait_memory() {   # unified memory: give the previous model's pages time to return
  for _ in $(seq 120); do (( $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) >= ${SPARK_MIN_FREE_GIB:-100} )) && return; sleep 1; done
}
stop_running() {
  case $(running) in
    glm) bash "$here/stop-glm.sh" ;;
    ds41) bash "$here/stop-ds41.sh" ;;
    qwen) bash "$here/stop-qwen.sh" ;;
    none) return 0 ;;
  esac
  wait_memory
}
case ${1:-${SPARK_MODEL:-status}} in
  glm) stop_running; exec bash "$here/start-glm.sh" ;;
  glm3) stop_running; exec bash "$here/start-glm.sh" 3 ;;
  ds41) stop_running; exec bash "$here/start-ds41.sh" ;;
  qwen) stop_running; exec bash "$here/start-qwen.sh" ;;
  stop) bash "$here/stop-glm.sh" > /dev/null 2>&1; bash "$here/stop-ds41.sh" > /dev/null 2>&1; bash "$here/stop-qwen.sh" ;;
  status) running ;;
  *) echo "usage: start.sh glm|glm3|ds41|qwen|status|stop" >&2; exit 2 ;;
esac
