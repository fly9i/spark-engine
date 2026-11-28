#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next server.
sock=/tmp/spark-qwen.sock
for f in front guard; do
  p=/tmp/spark-qwen-$f.pid
  [[ -f $p ]] && { kill "$(cat "$p")" 2> /dev/null; rm -f "$p"; }
done
# Graceful shutdown first so qwen::pcache flushes pending writes before exit.
python3 - "$sock" <<'PY' 2> /dev/null || true
import socket, sys
s = socket.socket(socket.AF_UNIX); s.settimeout(10); s.connect(sys.argv[1]); s.sendall(b'{"shutdown":true}\n')
PY
ep=/tmp/spark-qwen-engine.pid
epid=$([[ -f $ep ]] && cat "$ep" 2> /dev/null || true)
alive() { ([[ -n "$epid" ]] && kill -0 "$epid" 2> /dev/null) || pgrep -f "[q]wen-serve .* $sock" > /dev/null 2>&1; }
for _ in $(seq 60); do alive || break; sleep 0.5; done
for sig in TERM KILL; do
  alive || break
  [[ -n "$epid" ]] && kill -$sig "$epid" 2> /dev/null || true
  pkill -$sig -f "[q]wen-serve .* $sock" 2> /dev/null || true
  sleep 2
done
rm -f "$ep" "$sock"
echo "Qwen stopped"
