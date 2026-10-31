#!/usr/bin/env bash
# Keep GitHub and the four private Hugging Face repos current while work continues on Vista (MIGRATION.md Section 7).
# One run, safe to repeat; a lock stops overlapping runs, and a run that finds nothing new commits and uploads nothing.
#  1. GitHub: `git add -A` on results/ .agent/ paper/main.pdf paper/sca_overleaf.zip web/ HANDOVER.md README.md
#     MIGRATION.md; files over 95 MB outside Git LFS, symlinks into /scratch and secret-like files are unstaged and
#     logged. Only those paths are committed ("Sync: <date> <changed paths>"); changes other agents staged elsewhere stay
#     staged. If origin/main moved: `git pull --rebase origin main`; uncommitted edits of other agents are stashed and
#     restored with their staged state, and only when the incoming commits touch other files (otherwise the push waits
#     for a later run). Then `git push origin HEAD:main`, never forced.
#  2. Hugging Face: sync_hf.py uploads what is new since the last sync and records it in results/migration_manifest.csv
#     (phase-1 classes only; the account's private storage stays at or under SYNC_CAP_GB, default 97).
#  3. One status line goes to .agent/notes/sync_log.md. When the manifest changed, the manifest and the log are
#     committed and pushed; otherwise the line rides along with the next sync commit.
# Usage: bash repro/migration/sync_incremental.sh [--dry-run]
# Environment: SCA_PY (python with huggingface_hub), SYNC_CAP_GB, SYNC_DIR (lock and run logs).
set -uo pipefail
DRY=0
[[ ${1:-} == --dry-run ]] && DRY=1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel) || { echo "sync: $HERE is not inside a git checkout" >&2; exit 2; }
PY=${SCA_PY:-/work/09032/saini_2/vista/library/uv-envs/sca/bin/python}
SYNC_DIR=${SYNC_DIR:-/scratch/09032/saini_2/sca/hf_staging/sync}
CAP_GB=${SYNC_CAP_GB:-97}
SYNCLOG=.agent/notes/sync_log.md
MANIFEST=results/migration_manifest.csv
PATHS=(results .agent paper/main.pdf paper/sca_overleaf.zip web HANDOVER.md README.md MIGRATION.md)
MAX_BYTES=95000000
SECRET_NAME_RE='(^|/)(\.env(\..*)?|\.netrc|_netrc|\.git-credentials|\.hf_token|token|stored_tokens|credentials\.json|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|[^/]*\.(pem|key|p12|pfx))$'
SECRET_TEXT_RE='hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-(ant|proj)-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|WANDB_API_KEY[=: ]+[0-9a-f]{40}|machine api[.]wandb[.]ai'

export HF_HOME=${HF_HOME:-/scratch/09032/saini_2/tmp/hf}
if [[ -z ${HF_TOKEN:-} && ! -s $HF_HOME/token && -s $HOME/.cache/huggingface/token ]]; then
  export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
fi
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_MAX_CONCURRENT_FILE_INGESTION=2

mkdir -p "$SYNC_DIR"
exec 9>>"$SYNC_DIR/sync.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] sync: another run holds $SYNC_DIR/sync.lock; exiting" >&2
  exit 0
fi

NOW=$(date '+%Y-%m-%d %H:%M')
NOTES=()
COMMITS=()
PUSH="not attempted"
log() { printf '[%s] sync: %s\n' "$(date '+%F %T')" "$*" >&2; }
G() { git -C "$REPO" "$@"; }
GITDIR=$(G rev-parse --absolute-git-dir)

retry_git() {  # retry while another git process holds the index lock
  local i
  for i in 1 2 3 4 5 6; do
    "$@" && return 0
    [[ -e $GITDIR/index.lock ]] || return 1
    sleep 20
  done
  return 1
}

git_ready() {
  if [[ $(G symbolic-ref -q --short HEAD) != main ]]; then NOTES+=("git skipped: HEAD is not main"); return 1; fi
  local f
  for f in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
    if [[ -e $GITDIR/$f ]]; then NOTES+=("git skipped: $f in progress"); return 1; fi
  done
}

existing_paths() {  # the PATHS that exist on disk or are tracked (deleted files still count)
  local p
  for p in "${PATHS[@]}"; do
    if [[ -e $REPO/$p ]] || [[ -n $(G ls-files -- "$p" | head -1) ]]; then printf '%s\0' "$p"; fi
  done
}

guard_staged() {  # unstage what must never reach GitHub; arguments: pathspecs
  local f t size reason
  local -a staged secret
  mapfile -d '' -t staged < <(G diff --cached --name-only -z --diff-filter=d -- "$@")
  (( ${#staged[@]} )) || return 0
  mapfile -d '' -t secret < <(printf '%s\0' "${staged[@]}" |
    xargs -0 -r git -C "$REPO" --literal-pathspecs grep --cached -I -l -z -E "$SECRET_TEXT_RE" -- 2>/dev/null)
  for f in "${staged[@]}"; do
    reason=""
    if [[ -L $REPO/$f ]]; then
      t=$(readlink -- "$REPO/$f")
      [[ $t == /scratch/* ]] && reason="symlink into /scratch"
    elif [[ -f $REPO/$f ]]; then
      size=$(stat -c %s -- "$REPO/$f")
      if (( size > MAX_BYTES )) && [[ $(G check-attr filter -- "$f") != *": filter: lfs" ]]; then
        reason="over 95 MB outside Git LFS"
      fi
    fi
    [[ -z $reason && $f =~ $SECRET_NAME_RE ]] && reason="secret-like file name"
    if [[ -z $reason ]] && (( ${#secret[@]} )) && printf '%s\n' "${secret[@]}" | grep -qxF -- "$f"; then
      reason="secret-like content"
    fi
    if [[ -n $reason ]]; then
      retry_git env GIT_LITERAL_PATHSPECS=1 git -C "$REPO" reset -q -- "$f"
      NOTES+=("not committed: $f ($reason)")
      log "not committed: $f ($reason)"
    fi
  done
}

summarize() {  # short description of changed paths for the commit message
  printf '%s\n' "$@" | grep -vxF "$SYNCLOG" | awk -F/ '
    { k = ($1 == "results" && NF > 2) ? $1 "/" $2 : (($1 == "paper" || NF == 1) ? $0 : $1)
      if (!(k in c)) order[++n] = k
      c[k]++ }
    END { for (i = 1; i <= n && i <= 4; i++)
            printf "%s%s%s", (i > 1 ? ", " : ""), order[i], (c[order[i]] > 1 ? " (" c[order[i]] " files)" : "")
          if (n > 4) printf " and %d more", n - 4 }'
}

FILES=()
MSG=""
do_commit() {  # commit exactly FILES with MSG; whatever else is staged stays staged
  printf '%s\0' "${FILES[@]}" |
    GIT_LITERAL_PATHSPECS=1 git -C "$REPO" commit -q --only --pathspec-from-file=- --pathspec-file-nul -m "$MSG"
}

integrate_dirty() {  # origin/main moved and tracked files have uncommitted edits that the incoming commits do not touch
  local before after
  if (( $(G rev-list --count origin/main..HEAD) == 0 )); then
    retry_git G merge -q --ff-only origin/main  # fast-forward keeps unrelated staged and unstaged edits as they are
    return
  fi
  # Diverged: stash, rebase, and restore the stash with its index (the rebase autostash would unstage files that another
  # agent staged). Only a stash created here is popped.
  before=$(G rev-parse -q --verify refs/stash)
  retry_git G stash push -q -m "sync_incremental $NOW" || return 1
  after=$(G rev-parse -q --verify refs/stash)
  [[ -n $after && $after != "$before" ]] || { NOTES+=("stash push saved nothing"); return 1; }
  if retry_git G pull -q --rebase origin main; then
    G stash pop -q --index && return 0
    NOTES+=("rebased, but restoring the uncommitted edits failed; they are kept in stash@{0} ($after)")
    return 1
  fi
  [[ -d $GITDIR/rebase-merge || -d $GITDIR/rebase-apply ]] && G rebase --abort
  G stash pop -q --index || NOTES+=("uncommitted edits kept in stash@{0} ($after)")
  return 1
}

push_main() {  # integrate origin/main if it moved (pull --rebase), then push; never force
  local i overlap
  for i in 1 2 3; do
    if ! retry_git G fetch -q origin main; then PUSH="fetch failed"; return 1; fi
    if ! G merge-base --is-ancestor origin/main HEAD; then
      if G diff --quiet && G diff --cached --quiet; then
        if ! retry_git G pull -q --rebase origin main; then
          [[ -d $GITDIR/rebase-merge || -d $GITDIR/rebase-apply ]] && G rebase --abort
          PUSH="pull --rebase failed, not pushed"; return 1
        fi
      else
        overlap=$(comm -12 <(G diff --name-only HEAD...origin/main | sort -u) \
                           <({ G diff --name-only; G diff --cached --name-only; } | sort -u) | head -3 | tr '\n' ' ')
        if [[ -n $overlap ]]; then
          PUSH="not pushed: origin/main changed files with uncommitted local edits (${overlap% })"; return 1
        fi
        if ! integrate_dirty; then PUSH="could not integrate origin/main, not pushed"; return 1; fi
      fi
    fi
    if (( $(G rev-list --count origin/main..HEAD) == 0 )); then
      [[ $PUSH == pushed* ]] || PUSH="origin up to date"
      return 0
    fi
    if G push -q origin HEAD:main; then PUSH="pushed $(G rev-parse --short HEAD)"; return 0; fi
    sleep 30
  done
  PUSH="push failed"
  return 1
}

# ---------------------------------------------------------------------------- 1. GitHub
GITSUM="skipped"
if git_ready; then
  mapfile -d '' -t EX < <(existing_paths)
  if (( ${#EX[@]} == 0 )); then
    GITSUM="skipped: none of the synced paths exist"
  elif (( DRY )); then
    n=$(G status --porcelain -- "${EX[@]}" | grep -vc " ${SYNCLOG}\$")
    GITSUM="dry run, $n changed paths would be committed"
  else
    retry_git G add -A -- "${EX[@]}" || NOTES+=("git add failed")
    guard_staged "${EX[@]}"
    mapfile -d '' -t CH < <(G diff --cached --name-only -z -- "${EX[@]}")
    OTHERS=()
    for f in "${CH[@]}"; do [[ $f == "$SYNCLOG" ]] || OTHERS+=("$f"); done
    if (( ${#OTHERS[@]} == 0 )); then
      (( ${#CH[@]} )) && retry_git G reset -q -- "$SYNCLOG"
      GITSUM="no changes"
    else
      FILES=("${CH[@]}")
      MSG="Sync: $NOW $(summarize "${OTHERS[@]}")"
      if retry_git do_commit; then
        COMMITS+=("$(G rev-parse --short HEAD)")
        GITSUM="committed $(G rev-parse --short HEAD) (${MSG#Sync: $NOW })"
      else
        GITSUM="commit failed"
      fi
    fi
    push_main
    GITSUM+=", $PUSH"
  fi
fi
log "git: $GITSUM"

# ---------------------------------------------------------------------------- 2. Hugging Face
HFARGS=("$REPO" --cap-gb "$CAP_GB")
(( DRY )) && HFARGS+=(--dry-run)
HFOUT=$(nice -n 19 ionice -c3 "$PY" -I "$HERE/sync_hf.py" "${HFARGS[@]}")
HFRC=$?
HFSUM=$(printf '%s\n' "$HFOUT" | tail -1)
[[ $HFSUM == HF:* ]] || HFSUM="HF: no summary (sync_hf.py exit $HFRC)"
log "$HFSUM"

# ---------------------------------------------------------------------------- 3. status line, manifest commit
STATUS="- $NOW $(hostname -s): git $GITSUM; $HFSUM"
(( ${#NOTES[@]} )) && STATUS+="; notes: $(printf '%s; ' "${NOTES[@]}" | sed 's/; $//')"
if (( DRY )); then
  echo "$STATUS (dry run, nothing written)"
  exit 0
fi
if [[ ! -e $REPO/$SYNCLOG ]]; then
  printf '%s\n\n%s\n%s\n\n' "# Mirror sync log" \
    "One line per run of repro/migration/sync_incremental.sh (MIGRATION.md Section 7), oldest first. Lines of runs" \
    "that changed nothing reach GitHub with the next sync commit." > "$REPO/$SYNCLOG"
fi
printf '%s\n' "$STATUS" >> "$REPO/$SYNCLOG"
if ! G diff --quiet HEAD -- "$MANIFEST" && git_ready; then
  FILES=("$MANIFEST" "$SYNCLOG")
  MSG="Sync: $NOW Hugging Face uploads recorded in $MANIFEST"
  if retry_git G add -- "${FILES[@]}" && retry_git do_commit; then
    COMMITS+=("$(G rev-parse --short HEAD)")
    push_main
    log "manifest committed $(G rev-parse --short HEAD), $PUSH"
  else
    log "manifest commit failed; the next run commits it with results/"
  fi
fi
echo "$STATUS"
exit $(( HFRC == 0 ? 0 : 1 ))
