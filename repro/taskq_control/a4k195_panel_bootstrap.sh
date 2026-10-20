#!/bin/bash
set -euo pipefail
# CPU compute only (run on a CPU node, never through the GPU spool); all features must already be cached by z961_eval_a4k195_panel.sh.
# Set-level 95% intervals for FID/KID/patch FID/patch KID with 1000 draws that resample generated prompt ids jointly
# across methods and the 168 distinct reference-image identities (clone of hold_cpu_finalize/e9_krea_bootstrap.sh).
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=disabled
unset CROPE_TRANSFORMER CROPE_ROPE
S=/scratch/09032/saini_2/sca; P=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention; E=$S/eval/a4k195_panel; R=$P/results/a4k195_panel
test -f $E/DONE
mkdir -p $E/corrected_fidp_ci/features
cp -p $E/corrected_fidp/features/*.npz $E/corrected_fidp_ci/features/ && cp -p $E/corrected_fidp/evaluation_cache.json $E/corrected_fidp_ci/
cd $P
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python repro/evaluate_corrected.py --reference-clusters images --root $E/root --prompts /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method sca_nb --res 4096x4096 --out $E/corrected_fidp_ci --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000 --threads 8 --no-wandb
mkdir -p $R/corrected_fidp_ci && cp $E/corrected_fidp_ci/summary.csv $E/corrected_fidp_ci/paired_deltas.csv $R/corrected_fidp_ci/
(cd $R && sha256sum *.csv corrected_fidp/*.csv corrected_fidp_ci/*.csv > SHA256SUMS)
touch $E/CPU_BOOTSTRAP_COMPLETE
