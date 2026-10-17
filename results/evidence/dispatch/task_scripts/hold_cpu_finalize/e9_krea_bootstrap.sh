#!/bin/bash
set -euo pipefail
# CPU compute only; all learned scores/features must already be cached.
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=scale-covariant-attention WANDB_DIR=/scratch/09032/saini_2/sca/wandb
unset CROPE_TRANSFORMER CROPE_ROPE
test -f /scratch/09032/saini_2/sca/eval/e9_krea_base/READY_FOR_CPU_BOOTSTRAP
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/evaluate_corrected.py --reference-clusters images --root /scratch/09032/saini_2/sca/out/evalroot/e9_krea --prompts /scratch/09032/saini_2/sca/eval100.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method area --res 4096x4096 --out /scratch/09032/saini_2/sca/eval/e9_krea_base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000
touch /scratch/09032/saini_2/sca/eval/e9_krea_base/CPU_BOOTSTRAP_COMPLETE
