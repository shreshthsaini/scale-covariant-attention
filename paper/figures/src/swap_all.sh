#!/usr/bin/env bash
# Switch every generated table and figure to the final operator (sca_v2) for the evaluation runs that have landed;
# runs that have not landed keep their published values in grey (\pend{...}). Idempotent: rerun after each landing.
#
#   paper/figures/src/swap_all.sh            (from any directory)
#   SCA_V2_PENDING=1 paper/figures/src/swap_all.sh   renders everything as pending (published values)
#
# Order: gen_main_tables.py (main-text tables and headline macros), collect_latency_v2.py (if present),
# gen_app_tables.py (appendix tables), fig_paired_area.py, fig_latency.py, fig_baseline_deltas.py (with
# --no-results-csv, so nothing is written into results/), fig_dup.py (if present).
# A missing script is skipped; a failing one is reported and the remaining steps still run; exit status 1 if any
# step failed. SCA_V2_MOCK is always unset here, so mock test numbers can never reach paper/.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SRC/../../.." && pwd)"
unset SCA_V2_MOCK
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache
PY=(uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python)
cd "$ROOT" || exit 1

locate() {  # first existing copy of a script that another agent may put in src/ or scripts/
  local d
  for d in "$SRC" "$ROOT/scripts"; do
    if [ -f "$d/$1" ]; then echo "$d/$1"; return; fi
  done
  echo "$SRC/$1"
}

failed=()
step() {  # step <script> [args...]
  local script="$1" name
  shift
  name="$(basename "$script")"
  if [ ! -f "$script" ]; then
    echo "== $name: not present, skipped"
    return
  fi
  echo "== $name $*"
  if ! "${PY[@]}" "$script" "$@"; then
    failed+=("$name")
    echo "!! $name exited with an error; continuing"
  fi
}

echo "== run status (landed or pending)"
"${PY[@]}" "$SRC/v2swap.py" || failed+=("v2swap.py")

step "$SRC/gen_main_tables.py"
step "$(locate collect_latency_v2.py)"
step "$SRC/gen_app_tables.py"
step "$SRC/fig_paired_area.py"
step "$SRC/fig_latency.py"
step "$SRC/fig_baseline_deltas.py" --no-results-csv
step "$(locate fig_dup.py)"

if [ ${#failed[@]} -gt 0 ]; then
  echo "== swap_all: done with errors in: ${failed[*]}"
  exit 1
fi
echo "== swap_all: done"
