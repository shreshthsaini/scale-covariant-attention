#!/bin/bash
# SCA_MIN_SECONDS=1500
# E9: AReA on Krea (its own default checkpoint)
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
unset CROPE_TRANSFORMER CROPE_ROPE
cd /scratch/09032/saini_2/sca/code/PHASE2
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/run_area.py --seed 0 --guidance 4.5 --out /scratch/09032/saini_2/sca/out/krea/area/4096x4096 --model krea --res 4096x4096 --prompts /scratch/09032/saini_2/sca/eval100.jsonl --start 95 --end 100

# Promote evaluation only after exact complete paired coverage.
sca_generation_rc=$?
if [ "$sca_generation_rc" -eq 0 ]; then
  if ! /work/09032/saini_2/vista/library/uv-envs/sca/bin/python /scratch/09032/saini_2/sca/dispatch/20261017_eval_promotion/promote_ready_eval.py e9_krea --scratch /scratch/09032/saini_2/sca --baselines /scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017 --apply; then
    printf '%s\n' 'WARNING: evaluation promotion failed; generation status preserved' >&2
  fi
fi
exit "$sca_generation_rc"
