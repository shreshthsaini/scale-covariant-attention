#!/bin/bash
# Keep one gh-dev showcase fleet (fleet_SCSdev.sbatch, 4x2h) queued or running while the SCA spool has work
# and the 10 h fleets SCS1/SCS2 have not started. Checks every 15 min; one ssh per check.
# Stops when taskq/STOP exists, the spool is empty, or the SU balance falls below 500.
#   nohup bash slurm/dev_resubmit.sh > /scratch/09032/saini_2/sca/logs/dev_resubmit.log 2>&1 &
S=/scratch/09032/saini_2/sca; F=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/fleet_SCSdev.sbatch
while [ ! -f $S/taskq/STOP ] && [ -n "$(ls $S/taskq/pending 2>/dev/null)" ]; do
  if ! squeue -u $USER -h -o %j | grep -q '^SCSd$' && ! squeue -u $USER -t R -h -o %j | grep -q -E '^SCS[12]$'; then
    out=$(timeout 90 ssh -o ConnectTimeout=15 login1 "/usr/local/etc/taccinfo 2>/dev/null | grep ASC26048" 2>/dev/null)
    bal=$(echo "$out" | awk '{print $3}')
    if [ -n "$bal" ] && [ "${bal%.*}" -ge 500 ] 2>/dev/null; then
      echo "$(date +%FT%T) balance $bal, submitting: $(timeout 90 ssh -o ConnectTimeout=15 login1 "sbatch $F" 2>&1 | tail -1)"
    else
      echo "$(date +%FT%T) balance '$bal' below floor or unreadable; exit"; exit 0
    fi
  fi
  sleep 900
done
echo "$(date +%FT%T) exit (STOP or empty spool)"
