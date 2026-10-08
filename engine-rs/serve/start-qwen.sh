#!/usr/bin/env bash
# Qwen3.8-Flash-Next on one DGX Spark: engine `spark-engine qwen-serve` + OpenAI-compatible front end on $SPARK_PORT,
# and a memory guard that stops the engine if MemAvailable falls below QWEN_MEMGUARD_GIB (default 8; unified memory
# exhaustion can hang a GB10).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
socket=/tmp/spark-qwen.sock
export QWEN_VISION=${QWEN_VISION:-1}            # vision tower (~0.5 GB) + image/video requests; 0 = text only
export QWEN_ASSETS=${QWEN_ASSETS:-$SPARK_HOME/assets/qwen38}
vision=$([[ $QWEN_VISION == 1 ]] && echo on || echo off)
bash "$here/stop-qwen.sh" > /dev/null 2>&1 || true
(
  export SPARK_PYTHON=$py; source "$SPARK_HOME/scripts/env.sh"
  exec "$SPARK_BIN" qwen-serve "$QWEN_MODEL" "$socket"
) > /tmp/spark-qwen-engine.log 2>&1 &
echo $! > /tmp/spark-qwen-engine.pid
(
  limit=${QWEN_MEMGUARD_GIB:-8}; pid=$(cat /tmp/spark-qwen-engine.pid)
  while kill -0 "$pid" 2> /dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [[ -z $a ]] || (( a < limit )); then echo "$(date '+%F %T') MemAvailable ${a} GiB < ${limit}: stopping $pid" >> /tmp/spark-qwen-memguard.log; kill -9 "$pid"; break; fi
    sleep 0.5
  done
) > /dev/null 2>&1 &
echo $! > /tmp/spark-qwen-guard.pid
for _ in $(seq 300); do
  grep -q "ready on" /tmp/spark-qwen-engine.log 2> /dev/null && break
  kill -0 "$(cat /tmp/spark-qwen-engine.pid)" 2> /dev/null || { echo "engine exited:"; tail -5 /tmp/spark-qwen-engine.log; exit 1; }
  sleep 1
done
nohup "$py" "$here/qwen_server.py" --socket "$socket" --model-dir "$QWEN_MODEL" --port "$SPARK_PORT" --vision "$vision" \
  > /tmp/spark-qwen-front.log 2>&1 &
echo $! > /tmp/spark-qwen-front.pid
for _ in $(seq 30); do
  curl -sf "http://127.0.0.1:$SPARK_PORT/health" > /dev/null 2>&1 && { echo "Qwen3.8-Flash-Next ready: http://$(hostname -I | awk '{print $1}'):$SPARK_PORT/v1"; exit 0; }
  sleep 1
done
echo "front end did not come up"; tail -5 /tmp/spark-qwen-front.log; exit 1
