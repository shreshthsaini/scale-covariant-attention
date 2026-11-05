#!/bin/bash
# Compact status of the review-round 1 campaign (read-only).
S=/scratch/09032/saini_2/sca; Q=$S/taskq; QC=$S/taskq_cpu; R=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/results
RE='^(r[0-4]_|c[0-9]_)'
echo "== $(date +%FT%T)"
squeue -u saini_2 -h -o '%i %j %T %D %M %L %N' | grep -E ' SCA2[0-9] | cntrl-cmd '
printf 'gpu q: '; for d in pending running done failed; do printf '%s %s  ' $d "$(ls $Q/$d 2>/dev/null | grep -cE "$RE")"; done; echo
printf 'cpu q: '; for d in pending running done failed quarantine; do printf '%s %s  ' $d "$(ls $QC/$d 2>/dev/null | wc -l)"; done; echo
printf 'held : %s\n' "$(ls $Q/hold_review_round | grep -c '\.sh$')"
ls $Q/running 2>/dev/null | sed 's/^/  run  /'
ls $Q/failed 2>/dev/null | sed 's/^/  FAIL /'
ls $QC/running $QC/failed $QC/quarantine 2>/dev/null | grep -v '^$\|:$' | sed 's/^/  cpu  /'
for e in krea195_dype_sega a4k195_sched_bridge abl46_pair 6144 8192 flux2; do
  m=""; [ -e $S/eval/review1_$e/DONE ] && m="DONE $(cat $S/eval/review1_$e/DONE)"; [ -e $S/eval/review1_$e/CPU_BOOTSTRAP_COMPLETE ] && m="$m +boot"
  [ -n "$m" ] && echo "  eval review1_$e: $m"
done
[ -e $R/review1_latency/latency_summary.csv ] && echo "  latency summary present"
for g in a4k195_sched_bridge abl46_pair 6144 8192 flux2; do printf '  gen_done %s %s\n' $g "$(ls $S/eval/review1_gen_done/$g 2>/dev/null | wc -l)"; done | grep -v ' 0$'
exit 0
