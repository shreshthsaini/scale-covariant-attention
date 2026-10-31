#!/bin/bash
set -euo pipefail
# CPU only; clone of taskq/hold_cpu_finalize/a4k195_main_bootstrap.sh for the final operator (sca_v2) run, run directly
# on the operator's node (nice), not through the spool. Features were cached by z980_eval_a4k195_main_sca_v2.sh.
# Set-level 95% intervals for FID/KID/patch FID/patch KID with 1000 draws that resample generated prompt ids jointly
# across methods and the 168 distinct reference-image identities; deltas are method minus sca_v2.
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=
unset CROPE_TRANSFORMER CROPE_ROPE
S=/scratch/09032/saini_2/sca; P=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention; E=$S/eval/a4k195_main_sca_v2; R=$P/results/a4k195_main_sca_v2
test -f $E/DONE
mkdir -p $E/corrected_fidp_ci/features
cp -p $E/corrected_fidp/features/*.npz $E/corrected_fidp_ci/features/ && cp -p $E/corrected_fidp/evaluation_cache.json $E/corrected_fidp_ci/
cd $P
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python repro/evaluate_corrected.py --reference-clusters images --root $E/root --prompts /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method sca_v2 --res 4096x4096 --out $E/corrected_fidp_ci --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000 --threads 8 --no-wandb
mkdir -p $R/corrected_fidp_ci && cp $E/corrected_fidp_ci/summary.csv $E/corrected_fidp_ci/paired_deltas.csv $R/corrected_fidp_ci/
(cd $R && sha256sum $(find . -name '*.csv' | sort) > SHA256SUMS)
touch $E/CPU_BOOTSTRAP_COMPLETE
