#!/usr/bin/env bash
# Resolution checked on Linux aarch64; fresh installation remains to be validated.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {base|extra|imagereward} NEW_ENV_DIR SCRATCH_UV_CACHE PYTHON_INTERPRETER" >&2
  exit 2
fi
sca_metric_kind=$1
sca_env_target=$2
sca_cache_target=$3
sca_python_source=$4
case "$sca_metric_kind" in base|extra|imagereward) ;; *) exit 2 ;; esac
if [[ -e "$sca_env_target" || -L "$sca_env_target" ]]; then
  echo "Refusing to change existing environment: $sca_env_target" >&2
  exit 2
fi
sca_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export UV_CACHE_DIR="$sca_cache_target"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
uv venv --python "$sca_python_source" "$sca_env_target"
uv pip sync --python "$sca_env_target/bin/python" --torch-backend cu128 \
  --require-hashes "$sca_recipe_dir/$sca_metric_kind.lock"
uv pip check --python "$sca_env_target/bin/python"
