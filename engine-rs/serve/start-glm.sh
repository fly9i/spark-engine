#!/usr/bin/env bash
# GLM-5.3-Flash on two DGX Sparks: rank 1 on $GLM_WORKER (ssh), rank 0 here, OpenAI-compatible front end on $SPARK_PORT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
socket=/tmp/spark-glm.sock
ssho=(-o ControlMaster=auto -o ControlPath=/tmp/spark-ssh-%r@%h:%p -o ControlPersist=60)
bash "$here/stop-glm.sh" > /dev/null 2>&1 || true
# both ranks must run the same binary
s0=$(sha256sum "$SPARK_BIN" | cut -d' ' -f1); s1=$(ssh "${ssho[@]}" "$GLM_WORKER" "sha256sum $SPARK_BIN" | cut -d' ' -f1)
[[ $s0 == "$s1" ]] || { echo "engine binary differs between the nodes ($SPARK_BIN)" >&2; exit 1; }
ssh "${ssho[@]}" "$GLM_WORKER" "SPARK_ENV=$SPARK_ENV nohup bash $here/glm-rank.sh 1 $socket > /tmp/spark-glm-rank1.log 2>&1 &"
rm -f /tmp/spark-glm-rank0.log
nohup bash "$here/glm-rank.sh" 0 "$socket" > /tmp/spark-glm-rank0.log 2>&1 &
vision=$(bash -c "source '${GLM_PROFILE:-$SPARK_HOME/engine-rs/profiles/glm-tp2.env}'; echo \${GLM53_VISION:-0}")
if [[ $vision == 1 ]]; then mm=(--vision on --chat-template "$here/chat_template_mm.jinja"); else mm=(--vision off); fi
nohup "$py" "$here/openai_server.py" --socket "$socket" --port "$SPARK_PORT" --max-model-len "${GLM_KV_TOKENS:-1048576}" "${mm[@]}" \
  --model-dir "$GLM_MODEL" > /tmp/spark-glm-front.log 2>&1 &
echo "engine starting (about a minute)..."
started=$(date +%s)
until grep -q "\[serve\] rank0 ready" /tmp/spark-glm-rank0.log 2>/dev/null; do
  if (( $(date +%s) - started > 60 )) && ! pgrep -f " serve .*$socket" > /dev/null; then echo "rank 0 exited, see /tmp/spark-glm-rank0.log" >&2; exit 1; fi
  sleep 0.2
done
for _ in $(seq 300); do [[ $(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$SPARK_PORT/health") == 200 ]] && break; sleep 0.1; done
curl -sf "http://127.0.0.1:$SPARK_PORT/health" > /dev/null || { echo "front end not healthy, see /tmp/spark-glm-front.log" >&2; exit 1; }
# Compact physical memory once the engines hold their working set: on GB10's unified memory the kernel's proactive
# compaction otherwise migrates pages in bursts during the first requests (SPARK_COMPACT=0 skips).
if [[ ${SPARK_COMPACT:-1} != 0 ]]; then
  (echo 1 > /proc/sys/vm/compact_memory) 2> /dev/null & c=$!
  ssh "${ssho[@]}" "$GLM_WORKER" "echo 1 > /proc/sys/vm/compact_memory" 2> /dev/null || true; wait $c || true
fi
echo "GLM-5.3-Flash ready: http://$(hostname -I | awk '{print $1}'):$SPARK_PORT/v1"
