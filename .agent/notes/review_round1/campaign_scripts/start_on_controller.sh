#!/bin/bash
# start_on_controller.sh <controller job id>: run ON the controller node (one ssh). Starts the review-round release loop
# and two niced CPU workers (frozen slurm/worker_loop_v2.py on taskq_cpu), CPU deadline = job end minus 20 min, idle exit
# 48 h. Refuses to start a second release loop on the same node. Prints the new PIDs.
J=$1; S=/scratch/09032/saini_2/sca; D=$S/dispatch/review_round1; C=$S/code/f479459; PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
[ -n "$J" ] || { echo "usage: $0 <controller job id>"; exit 2; }
END=$(squeue -h -j $J -o %e); [ -n "$END" ] || { echo "job $J not found"; exit 2; }
DL=$(( $(date -d "$END" +%s) - 1200 )); H=$(hostname -s)
echo "host $H, job $J ends $END, CPU worker deadline $(date -d @$DL +%FT%T)"
cd $D
if pgrep -u saini_2 -f "bash $D/release_review_evals.sh" >/dev/null; then echo "release loop already running here"; else
  setsid nohup nice -n 19 bash $D/release_review_evals.sh >> $S/logs/release_review_evals.log 2>&1 < /dev/null &
fi
for k in a b; do
  CUDA_VISIBLE_DEVICES= setsid nohup nice -n 19 $PY $C/slurm/worker_loop_v2.py worker --queue $S/taskq_cpu --logs $S/logs/cpu \
    --deadline $DL --idle-seconds 172800 --poll 60 >> $S/logs/cpu/worker_cpu_$k.$H.out 2>&1 < /dev/null &
done
sleep 4
ps -u saini_2 -o pid,ppid,ni,lstart,args | grep -E "bash $D/release_review_evals.sh|worker_loop_v2.py worker --queue $S/taskq_cpu" | grep -v grep | cut -c1-150
