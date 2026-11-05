#!/bin/bash
# Review round 1 release loop: backup to the generation tasks' own release (same idea as slurm/release_dup_evals.sh,
# which is tied to taskq/hold_dup_eval and would send the 8192 CPU evaluation to the GPU queue).
# Every 120 s, for each task script left in taskq/hold_review_round:
#  * an evaluation with a "# RELEASE_AFTER: ^(<names>)[.]sh" line is released once no task of its generation group is
#    held here, pending or running (names matched anywhere in the file name, so prefixed copies count) and none is in
#    taskq/failed (that task must be rerun first; the evaluation would only park again). c* evaluations (the 8192 CPU
#    evaluation) go to taskq_cpu/pending, the others to taskq/pending. An evaluation that finds images missing parks
#    itself here again (exit 3) and is released again on a later pass.
#  * a CPU bootstrap c*_boot_* goes to taskq_cpu/pending once its evaluation wrote $E/DONE (normally the GPU evaluation
#    moves it itself when it writes DONE).
#  * anything else (generation tasks, the hand-moved item 4 and item 5 tasks) is left alone.
# Exits when no task script is left in the hold directory, or when taskq/STOP exists.
S=/scratch/09032/saini_2/sca; Q=$S/taskq; QC=$S/taskq_cpu; H=$Q/hold_review_round
declare -A said
log(){ echo "$(date +%FT%T) $*"; }
log "start PID $$ on $(hostname -s)"
while [ ! -f $Q/STOP ]; do
  n=0
  for f in $H/*.sh; do
    [ -f "$f" ] || continue; n=$((n+1)); b=$(basename "$f")
    pat=$(sed -n 's/^# RELEASE_AFTER: \^//p' "$f" | head -1)
    if [ -n "$pat" ]; then
      ls $H $Q/pending $Q/running 2>/dev/null | grep -q -E "$pat" && continue
      bad=$(ls $Q/failed 2>/dev/null | grep -E "$pat" | tr '\n' ' ')
      if [ -n "$bad" ]; then
        [ "${said[$b]:-}" == "$bad" ] || { log "waiting $b: generation in failed/: $bad"; said[$b]=$bad; }
        continue
      fi
      case $b in c*) d=$QC/pending ;; *) d=$Q/pending ;; esac
      mkdir -p $d && mv "$f" $d/ && log "released $b to $d"
    elif [[ $b == c*_boot_* ]]; then
      e=$(grep -o '; E=[^;]*' "$f" | head -1 | cut -d= -f2)
      if [ -n "$e" ] && [ -e "$e/DONE" ]; then
        mkdir -p $QC/pending && mv "$f" $QC/pending/ && log "released $b to $QC/pending ($e/DONE exists)"
      fi
    fi
  done
  [ $n -eq 0 ] && { log "no task script left in $H, exit"; exit 0; }
  sleep 120
done
log "taskq/STOP exists, exit"
