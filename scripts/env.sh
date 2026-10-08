# Runtime environment of the engine. Source it: `source scripts/env.sh` (after setting SPARK_PYTHON if needed).
#   SPARK_PYTHON: the Python with torch 2.13 cu130 (and its NVIDIA wheels); default: <repo>/.venv/bin/python if present,
#                 else python3 on PATH
#   CUDA_HOME:    CUDA 13 runtime (default /usr/local/cuda)
_spark_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_spark_py=${SPARK_PYTHON:-}
[[ -n $_spark_py ]] || { [[ -x $_spark_root/.venv/bin/python ]] && _spark_py=$_spark_root/.venv/bin/python || _spark_py=python3; }
_torch=$("$_spark_py" -c 'import torch, os; print(os.path.dirname(torch.__file__))')
_site=$(dirname "$_torch")
export LIBTORCH=$_torch
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
_libs="$_spark_root/lib:$_spark_root/third_party/exllamav3/lib:$_torch/lib:$CUDA_HOME/lib64"
for d in "$_site"/nvidia/*/lib; do [[ -d $d ]] && _libs="$_libs:$d"; done
export LD_LIBRARY_PATH=$_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# libtorch_cuda registers the CUDA backend in static initializers: load it before the engine's first CUDA call.
export LD_PRELOAD=$_torch/lib/libtorch_cuda.so
unset _spark_py _torch _site _libs
