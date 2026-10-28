#!/bin/bash
# Release parked duplication-fix evaluations (taskq/hold_dup_eval) into taskq/pending once their generation group has
# drained (no pending or running task matches the RELEASE_AFTER line of the evaluation). CPU only, one ls per
# 2 minutes; exits when the hold directory is empty or taskq/STOP exists. Same idea as slurm/release_evals.sh.
S=/scratch/09032/saini_2/sca; Q=$S/taskq; H=$Q/hold_dup_eval
while [ ! -f $Q/STOP ]; do
  n=0
  for f in $H/*.sh; do
    [ -f "$f" ] || continue; n=$((n+1))
    pat=$(sed -n 's/^# RELEASE_AFTER: //p' "$f" | head -1)
    [ -n "$pat" ] || continue
    if ! ls $Q/pending $Q/running 2>/dev/null | grep -q -E "$pat"; then
      mv "$f" $Q/pending/ && echo "$(date +%FT%T) released $(basename $f)"
    fi
  done
  [ $n -eq 0 ] && { echo "$(date +%FT%T) hold_dup_eval empty, exit"; exit 0; }
  sleep 120
done
