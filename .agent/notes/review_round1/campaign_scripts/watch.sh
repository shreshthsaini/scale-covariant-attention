#!/bin/bash
# Background until-loop: checks every 600 s and exits (printing why) when something needs attention:
# fleet state change, a new failed/quarantined task, a new DONE / bootstrap / latency result, a running task whose
# host is not in a running fleet job (stranded), fleet held with idle GPUs while work is pending, or a heartbeat.
# Usage: watch.sh <max_hours> <fleet job ids comma-separated>
S=/scratch/09032/saini_2/sca; Q=$S/taskq; QC=$S/taskq_cpu; R=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/results
EXTRA=${4:-}; NOIDLE=${3:-}; MAXH=${1:-3}; JOBS=${2:-1123398}; END=$(( $(date +%s) + MAXH*3600 ))
fleets(){ if [ "$JOBS" = auto ]; then squeue -u saini_2 -h -o "%i %j" 2>/dev/null | awk '$2 ~ /^SCA[0-9]+$/ {print $1}' | paste -sd, -; else echo $JOBS; fi; }
sig(){ J=$(fleets); [ -n "$EXTRA" ] && J=${J:+$J,}$EXTRA; [ -n "$J" ] && squeue -h -t PD,CF,R -j $J -o "%i:%T" 2>/dev/null | sort | tr "\n" " "; }
marks(){ ls $S/eval/review1_*/DONE $S/eval/review1_*/CPU_BOOTSTRAP_COMPLETE $R/review1_latency/latency_summary.csv 2>/dev/null | wc -l; }
MINE='^(r[0-4]_|c[0-9]_|b0_r4)'
nfail(){ ls $Q/failed $QC/failed $Q/quarantine $QC/quarantine 2>/dev/null | grep -E "$MINE" | grep -c '\.sh'; }
s0=$(sig); m0=$(marks); f0=$(nfail); idle=0
while [ $(date +%s) -lt $END ]; do
  sleep 600
  s1=$(sig); m1=$(marks); f1=$(nfail)
  [ "$s1" != "$s0" ] && { echo "EVENT fleet state: [$s0] -> [$s1]"; exit 0; }
  [ $f1 -gt $f0 ] && { echo "EVENT new failed/quarantined task(s): $(ls $Q/failed $QC/failed $Q/quarantine $QC/quarantine 2>/dev/null | grep -E "$MINE" | tr "\n" " ")"; exit 0; }
  [ $m1 -gt $m0 ] && { echo "EVENT new result marker(s)"; exit 0; }
  nw=$(pgrep -u saini_2 -f "worker_loop_v2.py worker --queue /scratch/09032/saini_2/sca/taskq_cpu" | wc -l); np2=$(ls $QC/pending 2>/dev/null | grep -c "\.sh"); [ $nw -eq 0 ] && [ $np2 -gt 0 ] && { echo "EVENT CPU workers gone with $np2 CPU task(s) pending"; exit 0; }
  fx=$(ls $QC/pending $QC/running 2>/dev/null | grep "\.sh" | grep -v -E "$MINE" | head -3 | tr "\n" " "); [ -n "$fx" ] && { echo "EVENT non-review task in taskq_cpu (would run on the controller): $fx"; exit 0; }
  J=$(fleets); nodes=$([ -n "$J" ] && for j in $(squeue -h -j $J -t R -o %i 2>/dev/null); do scontrol show hostnames $(squeue -h -j $j -o %N); done | sort -u)
  if [ -n "$nodes" ]; then
    for f in $(ls $Q/running 2>/dev/null | grep -E "$MINE"); do h=${f##*.}; h=${h%-g*}
      echo "$nodes" | grep -qx "$h" || { echo "EVENT stranded? $f (host $h not in running fleet nodes)"; exit 0; }; done
    nn=$(echo "$nodes" | wc -l); nr=$(ls $Q/running | wc -l); np=$(ls $Q/pending | wc -l)
    if [ $nr -lt $nn ]; then idle=$((idle+1)); else idle=0; fi
    [ -z "$NOIDLE" ] && [ $idle -ge 2 ] && { echo "EVENT idle GPUs: $nr running on $nn nodes, $np pending (2 checks)"; exit 0; }
  else
    for f in $(ls $Q/running 2>/dev/null | grep -E "$MINE"); do echo "EVENT stranded? $f (no running fleet)"; exit 0; done
  fi
done
echo "HEARTBEAT after ${MAXH} h"
