#!/bin/bash
# Keep an idle-only SCA guest worker on every GPU node (gh, gh-dev, gb) this account holds, so no held GPU sits idle
# while the SCA spool has work. Fleet jobs start their own workers; this covers any other allocation that frees up.
# Every 10 min, one ssh per node, sequential. Exits when taskq/STOP exists.
#   nohup bash slurm/gpu_watch.sh > /scratch/09032/saini_2/sca/logs/gpu_watch.log 2>&1 &
S=/scratch/09032/saini_2/sca; W=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/guest_worker_shared.sh
while [ ! -f $S/taskq/STOP ]; do
  if [ -n "$(ls $S/taskq/pending 2>/dev/null)" ]; then
    for n in $(squeue -u $USER -t R -h -p gh,gh-dev,gb -o "%L %N" | awk '$1 !~ /^[0-9]?[0-9]:[0-9][0-9]$/ {print $2}' | xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u); do
      timeout 25 ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $n \
        "pgrep -u $USER -f 'guest_worker_[a-z]*[.]sh\$' >/dev/null || pgrep -u $USER -f worker_loop.sh >/dev/null || { nohup bash $W > $S/logs/guest_\$(hostname -s)_watch.log 2>&1 < /dev/null & echo \$(date +%FT%T) started on $n; }" 2>/dev/null
    done
  fi
  sleep 600
done
echo "$(date +%FT%T) gpu_watch exit (STOP present)"
