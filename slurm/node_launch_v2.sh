#!/bin/bash
# Freeze this directory and telemetry.sh before submission. One task per GH200 node.
set -euo pipefail
dispatch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SCA_DEADLINE_EPOCH:?batch root must pass the actual allocation end epoch}"
exec /work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  "$dispatch_dir/worker_loop_v2.py" node \
  --queue "${SCA_QUEUE:-/scratch/09032/saini_2/sca/taskq}" \
  --logs "${SCA_LOGS:-/scratch/09032/saini_2/sca/logs}" \
  --deadline "$SCA_DEADLINE_EPOCH" \
  --telemetry-script "$dispatch_dir/telemetry.sh" \
  --telemetry-dir "${SCA_TELEMETRY_DIR:-/scratch/09032/saini_2/sca/telemetry}"
