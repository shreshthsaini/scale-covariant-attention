#!/usr/bin/env bash
# Checkpoint work in progress on the paper to GitHub so another agent can continue from origin/main.
# One pass: stage paper sources, figures, snippets, agent notes and the handover; skip files over 90 MB; commit only
# if something changed; push (never forced). With --loop MINUTES it repeats until STOP_FILE exists or 20 hours pass.
# Workers that share the working tree do not run git; only the lead (or this script) commits.
set -uo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STOP_FILE=${STOP_FILE:-/scratch/09032/saini_2/sca/paper_builds/wip_push.stop}
PATHS=(paper/main.tex paper/appendix.tex paper/references.bib paper/figures web .agent/notes .agent/STATE.md HANDOVER.md scripts/wip_push.sh)
one_pass() {
  cd "$REPO" || return 1
  [[ $(git symbolic-ref -q --short HEAD) == main ]] || { echo "wip_push: HEAD is not main; skipped"; return 0; }
  [[ -e .git/index.lock ]] && { echo "wip_push: index locked; skipped"; return 0; }
  git add -A -- "${PATHS[@]}" 2>/dev/null
  while IFS= read -r -d '' f; do
    if [[ -f $f && $(stat -c %s "$f") -gt 90000000 ]]; then git reset -q -- "$f"; echo "wip_push: left out (over 90 MB) $f"; fi
  done < <(git diff --cached --name-only -z --diff-filter=d)
  if git diff --cached --quiet; then echo "wip_push: $(date '+%m-%d %H:%M') nothing new"; return 0; fi
  n=$(git diff --cached --name-only | wc -l)
  git commit -q -m "WIP: checkpoint $(date '+%Y-%m-%d %H:%M') ($n files)

Work in progress (see .agent/STATE.md for what is in flight). Delivered states are
tagged; this commit may be mid-edit." || return 1
  git push -q origin HEAD:main 2>&1 | tail -1
  echo "wip_push: $(date '+%m-%d %H:%M') pushed $(git rev-parse --short HEAD) ($n files)"
}
if [[ ${1:-} == --loop ]]; then
  mins=${2:-20}; end=$(( $(date +%s) + 72000 ))
  while [[ ! -e $STOP_FILE && $(date +%s) -lt $end ]]; do one_pass; sleep $(( mins * 60 )); done
  echo "wip_push: loop ended"
else
  one_pass
fi
