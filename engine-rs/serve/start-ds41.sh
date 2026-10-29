#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on three DGX Sparks (TP3): ranks 1 and 2 on SPARK_WORKERS (ssh), rank 0 here, OpenAI-compatible front
# end on $SPARK_PORT. Each rank loads its pre-split shard ($DS41_SHARDS/rank<r>, docs/ds41.md). Logs: /tmp/spark-ds41-rank<r>.log
# (on each node), /tmp/spark-ds41-front.log. Stop with stop-ds41.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
socket=/tmp/spark-ds41.sock
bash "$here/stop-ds41.sh" > /dev/null 2>&1 || true
need_nodes 3
wait_memory 3
# ranks 1 and 2 start together, then rank 0, then the front end, which loads while the engine does and listens once the
# ready file holds the context limit
# (wait only for the two ssh launches: rank 0 is a job of this shell too and runs until stopped, so a bare `wait`
# never returned and the front end was never started)
launches=()
for r in 2 1; do
  ssh "${ssho[@]}" "${WORKERS[$r]}" "SPARK_ENV=$SPARK_ENV nohup bash $here/ds41-rank.sh $r /tmp/spark-ds41-unused.sock ${MASTERS[$r]} > /tmp/spark-ds41-rank$r.log 2>&1 < /dev/null &" &
  launches+=($!)
done
rm -f /tmp/spark-ds41-rank0.log
nohup bash "$here/ds41-rank.sh" 0 "$socket" "${MASTERS[0]}" > /tmp/spark-ds41-rank0.log 2>&1 < /dev/null &
wait "${launches[@]}"
ready=/tmp/spark-ds41-ready.$SPARK_PORT; rm -f "$ready"
vision=$([[ ${DS41_VISION:-1} == 1 ]] && echo on || echo off)
GLM53_SERVE_MAX_SEQS=${DS41_SERVE_STORES:-4} nohup "$py" "$here/ds41_server.py" --socket "$socket" --model-dir "${DS41_CKPT:?}" \
  --port "$SPARK_PORT" --model-name "${DS41_MODEL_NAME:-DeepSeek-V4.1-Flash-1017}" --vision "$vision" \
  --canvas-dir /dev/shm/spark-ds41-mm --ready-file "$ready" > /tmp/spark-ds41-front.log 2>&1 &
echo "engine loading (about 20 s with the shards on a fast NVMe)..."
started=$(date +%s)
until grep -q "\[ds41-serve\] ready" /tmp/spark-ds41-rank0.log 2>/dev/null; do
  if (( $(date +%s) - started > 30 )) && ! pgrep -f "[d]s41-serve .* $socket" > /dev/null; then
    pkill -f "[d]s41_server.py --socket $socket" || true; echo "rank 0 exited, see /tmp/spark-ds41-rank0.log" >&2; exit 1; fi
  if (( $(date +%s) - started > 1200 )); then pkill -f "[d]s41_server.py --socket $socket" || true; echo "timeout, see /tmp/spark-ds41-rank*.log" >&2; exit 1; fi
  sleep 0.1
done
# the engine sizes one shared KV pool from the memory left after its warm-up and reports the longest request it can take
cap=$(grep -o "\[ds41-serve\] ready on .* (max context [0-9]*)" /tmp/spark-ds41-rank0.log | grep -o "[0-9]*)" | tr -d ')')
echo "$cap" > "$ready.tmp" && mv "$ready.tmp" "$ready"
for _ in $(seq 600); do [[ $(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$SPARK_PORT/health") == 200 ]] && break; sleep 0.05; done
curl -sf "http://127.0.0.1:$SPARK_PORT/health" > /dev/null || { echo "front end not healthy, see /tmp/spark-ds41-front.log" >&2; exit 1; }
compact_nodes 3
echo "DeepSeek-V4.1-Flash ready (TP3, max context $cap): http://$(hostname -I | awk '{print $1}'):$SPARK_PORT/v1"
