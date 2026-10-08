#!/usr/bin/env bash
# Package a release: dist/spark-engine-<version>-linux-aarch64-cu130-sm121.tar.gz (+ .sha256).
# Contents: bin/spark-engine, lib/libexllamav3_ext.so (exllamav3, MIT; its LICENSE alongside), the serve scripts,
# profiles, runtime env script, Qwen assets, benchmarks, docs and licenses. PyTorch, CUDA and models are not included.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ver=${1:-$(git -C "$root" describe --tags --always 2>/dev/null || echo dev)}
name=spark-engine-$ver-linux-aarch64-cu130-sm121
out=$root/dist/$name
rm -rf "$out"; mkdir -p "$out/bin" "$out/lib" "$out/engine-rs" "$out/scripts"
cp "$root/engine-rs/target/release/spark-engine" "$out/bin/"
cp "$root/third_party/exllamav3/lib/libexllamav3_ext.so" "$out/lib/"
cp "$root/third_party/exllamav3/LICENSE" "$out/lib/LICENSE.exllamav3"
cp -r "$root/engine-rs/serve" "$root/engine-rs/profiles" "$out/engine-rs/"
find "$out/engine-rs/serve" -name __pycache__ -prune -exec rm -rf {} +
cp "$root/scripts/env.sh" "$root/scripts/fetch_ablit_transplant.py" "$root/scripts/download-models.sh" "$out/scripts/"
cp -r "$root/assets" "$root/bench" "$root/docs" "$out/"
cp "$root/README.md" "$root/README.zh-CN.md" "$root/LICENSE" "$root/THIRD_PARTY.md" "$root/spark.env.example" "$out/"
sed -i "s|^SPARK_BIN=.*|SPARK_BIN=\$SPARK_HOME/bin/spark-engine|" "$out/spark.env.example"
(cd "$root/dist" && tar czf "$name.tar.gz" "$name" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "$root/dist/$name.tar.gz"
