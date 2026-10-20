#!/usr/bin/env bash
# GLM-5.3-Flash over GLM_WORLD DGX Sparks (2 or 3; default 2; `start-glm.sh 3` overrides): ranks 1.. on SPARK_WORKERS (ssh), rank 0 here, OpenAI-compatible
# front end on $SPARK_PORT. TP2 and TP3 use the same binary, checkpoint and profile; the topology env
# (profiles/tp<world>-rdma.env) carries the transport. Logs: /tmp/spark-glm-rank<r>.log (on each node), /tmp/spark-glm-front.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
world=${1:-${GLM_WORLD:-2}}   # an argument wins over spark.env
[[ $world == 2 || $world == 3 ]] || { echo "GLM_WORLD must be 2 or 3" >&2; exit 1; }
socket=/tmp/spark-glm.sock
bash "$here/stop-glm.sh" > /dev/null 2>&1 || true
need_nodes "$world"
for ((r = world - 1; r >= 1; r--)); do
  ssh "${ssho[@]}" "${WORKERS[$r]}" "SPARK_ENV=$SPARK_ENV nohup bash $here/glm-rank.sh $r $socket $world ${MASTERS[$r]} > /tmp/spark-glm-rank$r.log 2>&1 < /dev/null &"
done
rm -f /tmp/spark-glm-rank0.log
nohup bash "$here/glm-rank.sh" 0 "$socket" "$world" "${MASTERS[0]}" > /tmp/spark-glm-rank0.log 2>&1 < /dev/null &
vision=$(bash -c "source '${GLM_PROFILE:-$SPARK_HOME/engine-rs/profiles/glm.env}'; echo \${GLM53_VISION:-0}")
if [[ $vision == 1 ]]; then mm=(--vision on --chat-template "$here/chat_template_mm.jinja"); else mm=(--vision off); fi
nohup "$py" "$here/openai_server.py" --socket "$socket" --port "$SPARK_PORT" --max-model-len "${GLM_KV_TOKENS:-1048576}" "${mm[@]}" \
  --model-dir "$GLM_MODEL" --model-name "${GLM_MODEL_NAME:-GLM-5.3-Flash}" > /tmp/spark-glm-front.log 2>&1 &
echo "engine starting (TP$world, about a minute)..."
started=$(date +%s)
until grep -q "\[serve\] rank0 ready" /tmp/spark-glm-rank0.log 2>/dev/null; do
  if (( $(date +%s) - started > 60 )) && ! pgrep -f " serve .*$socket" > /dev/null; then echo "rank 0 exited, see /tmp/spark-glm-rank0.log" >&2; exit 1; fi
  if (( $(date +%s) - started > 1200 )); then echo "timeout, see /tmp/spark-glm-rank*.log" >&2; exit 1; fi
  sleep 0.2
done
for _ in $(seq 300); do [[ $(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$SPARK_PORT/health") == 200 ]] && break; sleep 0.1; done
curl -sf "http://127.0.0.1:$SPARK_PORT/health" > /dev/null || { echo "front end not healthy, see /tmp/spark-glm-front.log" >&2; exit 1; }
compact_nodes "$world"
echo "GLM-5.3-Flash ready (TP$world): http://$(hostname -I | awk '{print $1}'):$SPARK_PORT/v1"
