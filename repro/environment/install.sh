#!/usr/bin/env bash
# Run on a compute node. Refuse to change any existing environment.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 NEW_ENV_DIR SCRATCH_UV_CACHE PYTHON_INTERPRETER" >&2
  exit 2
fi
sca_env_target=$1
sca_cache_target=$2
sca_python_source=$3
if [[ -e "$sca_env_target" || -L "$sca_env_target" ]]; then
  echo "Refusing to change existing environment: $sca_env_target" >&2
  exit 2
fi
sca_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export UV_CACHE_DIR="$sca_cache_target"
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_INSTALLS=1
export UV_LINK_MODE=copy
uv venv --python "$sca_python_source" "$sca_env_target"
uv pip sync --python "$sca_env_target/bin/python" --torch-backend cu128 \
  --no-build --require-hashes "$sca_recipe_dir/generation.lock"
uv pip check --python "$sca_env_target/bin/python"
