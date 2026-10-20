#!/usr/bin/env bash
# Stop DeepSeek-V4.1-Flash: front end first, then a graceful engine shutdown (all ranks exit together, the prefix cache
# finishes its writes), then terminate leftovers on the three nodes.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
socket=/tmp/spark-ds41.sock
pat="[d]s41-serve .*/rank[0-9] "
pkill -f "[d]s41_server.py --socket $socket" 2> /dev/null
sleep 1
python3 - "$socket" <<'PY' 2> /dev/null || true
import socket, sys
s = socket.socket(socket.AF_UNIX); s.settimeout(30); s.connect(sys.argv[1]); s.sendall(b'{"shutdown":true}\n')
PY
alive() { pgrep -f "$pat" > /dev/null && return 0; for h in "${WORKERS[@]:1}"; do ssh -o ConnectTimeout=5 "$h" "pgrep -f '$pat'" > /dev/null 2>&1 && return 0; done; return 1; }
for _ in $(seq 120); do alive || break; sleep 0.5; done
for sig in TERM KILL; do
  alive || break
  pkill -$sig -f "$pat" 2> /dev/null
  for h in "${WORKERS[@]:1}"; do ssh -o ConnectTimeout=5 "$h" "pkill -$sig -f '$pat'" 2> /dev/null; done
  sleep 3
done
rm -f "$socket"
alive && echo "engine still running" || echo "DeepSeek-V4.1-Flash stopped"
