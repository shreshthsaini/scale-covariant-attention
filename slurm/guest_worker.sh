#!/bin/bash
# Guest worker for idle nodes owned by other jobs: claims a task only while no foreign process
# uses the GPU; exits (after finishing its current task) as soon as one appears.
S=/scratch/09032/saini_2/sca; Q=$S/taskq; H=$(hostname -s)-guest; IDLE=0
/work/09032/saini_2/vista/library/bin/gpu_telemetry.sh $S/telemetry 30 & TEL=$!
trap 'kill $TEL 2>/dev/null' EXIT
while [ ! -f $Q/STOP ]; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  if [ "$n" -gt 0 ]; then echo "$(date +%FT%T) $H foreign GPU process seen; exiting"; break; fi
  t=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -z "$t" ]; then sleep 60; IDLE=$((IDLE+60)); [ $IDLE -ge 3600 ] && break; continue; fi
  if mv $Q/pending/$t $Q/running/$t.$H 2>/dev/null; then
    IDLE=0; echo "$(date +%FT%T) $H start $t"
    bash $Q/running/$t.$H > $S/logs/task_${t%.sh}.log 2>&1; rc=$?
    if [ $rc -eq 0 ]; then mv $Q/running/$t.$H $Q/done/$t; else mv $Q/running/$t.$H $Q/failed/$t; fi
    echo "$(date +%FT%T) $H end $t rc=$rc"
  fi
done
echo "$(date +%FT%T) $H exit"
