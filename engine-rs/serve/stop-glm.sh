#!/usr/bin/env bash
# Stop the GLM-5.3-Flash server (TP2 or TP3): front end, then every rank. Checks every configured worker whatever the
# world size, so a TP3 rank left on the third node is found by a TP2 stop too.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pat=" [s]erve .* /tmp/spark-glm.sock"
pkill -f "[o]penai_server.py --socket /tmp/spark-glm.sock" 2> /dev/null
# graceful: every rank exits together and the prefix cache finishes its writes
python3 - /tmp/spark-glm.sock <<'PY' 2> /dev/null || true
import socket, sys
s = socket.socket(socket.AF_UNIX); s.settimeout(30); s.connect(sys.argv[1]); s.sendall(b'{"shutdown":true}\n'); s.recv(100)
PY
alive() { pgrep -f "$pat" > /dev/null && return 0; for h in "${WORKERS[@]:1}"; do ssh -o ConnectTimeout=5 "$h" "pgrep -f '$pat'" > /dev/null 2>&1 && return 0; done; return 1; }
for _ in $(seq 120); do alive || break; sleep 0.5; done
for sig in TERM KILL; do
  alive || break
  pkill -$sig -f "$pat" 2> /dev/null
  for h in "${WORKERS[@]:1}"; do ssh -o ConnectTimeout=5 "$h" "pkill -$sig -f '$pat'" 2> /dev/null; done
  sleep 3
done
rm -f /tmp/spark-glm.sock
echo "GLM stopped"
