#!/bin/bash
# CPU only: corrected FID/KID/patch FID/patch KID (168 reference identities, 1000-draw intervals) for sca_v2_prog against
# AReA, HRDiT, UltraImage, DyPE, SEGA, full attention and sca_v2 on the 195 prompts at 4096^2, from the cached features
# of eval/a4k195_main, eval/a4k195_main_sca_v2 and eval/prog195_sca_v2 (no image re-encoded). Method owner, 2026-11-07.
set -euo pipefail
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache OMP_NUM_THREADS=8 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=
unset CROPE_TRANSFORMER CROPE_ROPE
E=/scratch/09032/saini_2/sca/eval/prog195_vs_baselines; P=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
O=$E/cfid; mkdir -p $O/features
for d in /scratch/09032/saini_2/sca/eval/a4k195_main/corrected_fidp /scratch/09032/saini_2/sca/eval/a4k195_main_sca_v2/corrected_fidp_ci /scratch/09032/saini_2/sca/eval/prog195_sca_v2/corrected_fidp; do cp -p -n $d/features/*.npz $O/features/; done
$PY -I - /scratch/09032/saini_2/sca/eval/a4k195_main/corrected_fidp/evaluation_cache.json /scratch/09032/saini_2/sca/eval/a4k195_main_sca_v2/corrected_fidp_ci/evaluation_cache.json /scratch/09032/saini_2/sca/eval/prog195_sca_v2/corrected_fidp/evaluation_cache.json $O/evaluation_cache.json <<'PYEOF'
import json, sys
*srcs, out = sys.argv[1:]
x = {}
for s in srcs:
    for k, v in json.load(open(s)).items():
        x.setdefault(k, v)
json.dump(x, open(out, "w"), indent=2); print(len(x), "cache entries")
PYEOF
cd $P
nice -n 10 $PY repro/evaluate_corrected.py --reference-clusters images --root $E/root --prompts /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method sca_v2_prog --res 4096x4096 --out $O --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000 --threads 8 --no-wandb
date -u +%FT%TZ > $E/SETLEVEL_DONE
