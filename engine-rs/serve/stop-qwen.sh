#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next server.
for f in front guard engine; do
  p=/tmp/spark-qwen-$f.pid
  [[ -f $p ]] && { kill "$(cat "$p")" 2> /dev/null; rm -f "$p"; }
done
pkill -f "[q]wen-serve .* /tmp/spark-qwen.sock" 2> /dev/null
for _ in $(seq 100); do pgrep -f "[q]wen-serve .* /tmp/spark-qwen.sock" > /dev/null || break; sleep 0.1; done
pkill -9 -f "[q]wen-serve .* /tmp/spark-qwen.sock" 2> /dev/null; rm -f /tmp/spark-qwen.sock
echo "Qwen stopped"
