#!/bin/bash
# Keep an SCA co-tenant worker on every GPU node this account holds (user directive 2026-10-20).
# Every 10 min: for each running gh allocation with more than 1 h left, ssh once per node (sequential, one
# connection at a time) and start guest_worker_cotenant.sh if no SCA worker is running there. Stops when
# taskq/STOP exists. Run once, in the background, from a compute node:
#   nohup bash slurm/occupy_all.sh > /scratch/09032/saini_2/sca/logs/occupy_all.log 2>&1 &
S=/scratch/09032/saini_2/sca; W=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/guest_worker_cotenant.sh
while [ ! -f $S/taskq/STOP ]; do
  squeue -u $USER -t R -h -p gh -o "%L %N" | while read left nodes; do
    case "$left" in *-*) ;; [0-9]:[0-9]*:[0-9]*|[0-9][0-9]:[0-9]*:[0-9]*) h=${left%%:*}; [ "${h#0}" -lt 1 ] && continue ;; *) continue ;; esac
    for n in $(scontrol show hostnames "$nodes"); do
      timeout 25 ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $n \
        "pgrep -u $USER -f 'guest_worker_[a-z]*[.]sh$' >/dev/null || { nohup bash $W > $S/logs/cotenant_$n.log 2>&1 < /dev/null & echo $(date +%FT%T) started on $n; }" 2>/dev/null
    done
  done
  sleep 600
done
echo "$(date +%FT%T) occupy_all exit (STOP present)"
