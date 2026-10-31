#!/usr/bin/env bash
# Create (private) and upload one staged repo, then verify it. Resumable: rerun the same command after a failure.
# Usage: bash upload_repo.sh <staging> <repo_name> <repo_type> [extra hf upload args...]
set -euo pipefail
STAGING=$1; NAME=$2; TYPE=$3; shift 3
PY=${SCA_PY:-/work/09032/saini_2/vista/library/uv-envs/sca/bin/python}
HF=${SCA_HF:-$(dirname "$PY")/hf}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RID=shreshthsaini/$NAME
"$PY" -I -c "
from huggingface_hub import HfApi
api = HfApi(); api.create_repo('$RID', repo_type='$TYPE', private=True, exist_ok=True)
assert api.repo_info('$RID', repo_type='$TYPE').private is True, 'repo is not private'
print('$RID private ok')"
for attempt in 1 2 3 4 5; do
  if nice -n 19 "$HF" upload "$RID" "$STAGING/$NAME" . --repo-type "$TYPE" --private \
      --commit-message "SCA migration from TACC Vista (2026-10-31): $NAME" --format quiet "$@"; then
    break
  fi
  echo "upload attempt $attempt failed; retrying in 60 s" >&2; sleep 60
done
nice -n 19 "$PY" -I "$HERE/verify_hf.py" "$RID" "$TYPE" "$STAGING/$NAME"
