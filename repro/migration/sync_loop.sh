#!/usr/bin/env bash
# Run sync_incremental.sh now and then every 6 hours (SYNC_INTERVAL seconds) until the stop file exists
# (default /scratch/09032/saini_2/sca/taskq/SYNC_STOP, checked every minute). MIGRATION.md Section 7.
# Start on a compute node, detached, logging to scratch:
#   setsid nohup bash repro/migration/sync_loop.sh >> /scratch/09032/saini_2/sca/logs/sync_loop.log 2>&1 < /dev/null &
# The loop appends its start (with PID) and its exit to .agent/notes/sync_log.md. A second loop exits at once while one
# holds the loop lock. Stop: touch /scratch/09032/saini_2/sca/taskq/SYNC_STOP (remove that file before a restart).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
STOP=${SYNC_STOP:-/scratch/09032/saini_2/sca/taskq/SYNC_STOP}
INTERVAL=${SYNC_INTERVAL:-21600}
SYNC_DIR=${SYNC_DIR:-/scratch/09032/saini_2/sca/hf_staging/sync}
SYNCLOG=$REPO/.agent/notes/sync_log.md
mkdir -p "$SYNC_DIR"
exec 8>>"$SYNC_DIR/loop.lock"
if ! flock -n 8; then
  echo "[$(date '+%F %T')] sync loop: another loop holds $SYNC_DIR/loop.lock; exiting"
  exit 0
fi
note() {
  echo "[$(date '+%F %T')] sync loop: $*"
  [[ -e $SYNCLOG ]] && printf -- '- %s %s: sync loop %s\n' "$(date '+%Y-%m-%d %H:%M')" "$(hostname -s)" "$*" >> "$SYNCLOG"
}
trap 'note "PID $$ stopped by a signal"; exit 143' TERM INT HUP
note "started, PID $$, every $((INTERVAL / 3600)) h, log /scratch/09032/saini_2/sca/logs/sync_loop.log, stop file $STOP"
while [[ ! -e $STOP ]]; do
  echo "[$(date '+%F %T')] sync loop: run start"
  nice -n 19 ionice -c3 bash "$HERE/sync_incremental.sh" 8>&-
  echo "[$(date '+%F %T')] sync loop: run end, exit $?"
  waited=0
  while (( waited < INTERVAL )) && [[ ! -e $STOP ]]; do
    sleep 60 &
    wait $!
    waited=$((waited + 60))
  done
done
note "PID $$ exits: $STOP exists"
