#!/bin/bash
# Sharing guest worker (user-approved 2026-10-18: use any idle GPU). Claims a task only when the GPU is
# nearly idle (util <= 10%) and at least MIN_FREE_MB is free; otherwise waits. Exits after 2 h idle.
S=/scratch/09032/saini_2/sca; Q=$S/taskq; H=$(hostname -s)-guest; IDLE=0; MIN_FREE_MB=${MIN_FREE_MB:-56000}
/work/09032/saini_2/vista/library/bin/gpu_telemetry.sh $S/telemetry 30 & TEL=$!
trap 'kill $TEL 2>/dev/null' EXIT
while [ ! -f $Q/STOP ]; do
  read util free <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -1 | tr -d ',')"
  if [ "${util:-100}" -gt 10 ] || [ "${free:-0}" -lt "$MIN_FREE_MB" ]; then sleep 120; IDLE=$((IDLE+120)); [ $IDLE -ge 7200 ] && break; continue; fi
  t=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -z "$t" ]; then sleep 60; IDLE=$((IDLE+60)); [ $IDLE -ge 7200 ] && break; continue; fi
  if mv $Q/pending/$t $Q/running/$t.$H 2>/dev/null; then
    IDLE=0; echo "$(date +%FT%T) $H start $t (util=$util free=${free}MB)"
    bash $Q/running/$t.$H > $S/logs/task_${t%.sh}.log 2>&1; rc=$?
    if [ $rc -eq 0 ]; then mv $Q/running/$t.$H $Q/done/$t; else mv $Q/running/$t.$H $Q/failed/$t; fi
    echo "$(date +%FT%T) $H end $t rc=$rc"
  fi
done
echo "$(date +%FT%T) $H exit"
