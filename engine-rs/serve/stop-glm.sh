#!/usr/bin/env bash
# Stop the GLM-5.3-Flash TP2 server (both ranks and the front end).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pkill -f "[o]penai_server.py --socket /tmp/spark-glm.sock" 2> /dev/null
pkill -f " [s]erve .* /tmp/spark-glm.sock" 2> /dev/null
ssh -o ConnectTimeout=5 "$GLM_WORKER" "pkill -f ' [s]erve .* /tmp/spark-glm.sock'" 2> /dev/null
for _ in $(seq 100); do pgrep -f " [s]erve .* /tmp/spark-glm.sock" > /dev/null || break; sleep 0.1; done
pkill -9 -f " [s]erve .* /tmp/spark-glm.sock" 2> /dev/null; rm -f /tmp/spark-glm.sock
echo "GLM stopped"
