# Shared by the serve scripts: settings from spark.env (SPARK_ENV, default <repo>/spark.env).
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
SPARK_ENV=${SPARK_ENV:-$repo/spark.env}
[[ -f $SPARK_ENV ]] || { echo "settings file $SPARK_ENV missing: copy spark.env.example to spark.env and edit it" >&2; exit 1; }
set -a; source "$SPARK_ENV"; set +a
SPARK_HOME=${SPARK_HOME:-$repo}
SPARK_PORT=${SPARK_PORT:-8888}
py=${SPARK_PYTHON:-python3}
