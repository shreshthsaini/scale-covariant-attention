#!/bin/bash
# CPU only (operator node, nice): set-level FID/KID/patch FID/patch KID with 1000-draw bootstrap intervals for progressive
# SCA (sca_v2_prog) against sca_v2 and AReA on the 195 prompts, from the cached features of eval/a4k195_main_sca_v2 and
# eval/prog195_sca_v2 (same evaluator, same reference pool; no image is re-encoded). Method owner, 2026-11-07.
set -euo pipefail
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache OMP_NUM_THREADS=8
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=
unset CROPE_TRANSFORMER CROPE_ROPE
S=/scratch/09032/saini_2/sca; P=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention; E=$S/eval/prog195_pairs
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
: # root built by the inline linker (see log)
for ref in sca_v2 area; do
  O=$E/cfid_vs_$ref; mkdir -p $O/features
  cp -p -n $S/eval/a4k195_main_sca_v2/corrected_fidp_ci/features/*.npz $O/features/
  cp -p -n $S/eval/prog195_sca_v2/corrected_fidp/features/*.npz $O/features/
  $PY -I - $S/eval/a4k195_main_sca_v2/corrected_fidp_ci/evaluation_cache.json $S/eval/prog195_sca_v2/corrected_fidp/evaluation_cache.json $O/evaluation_cache.json <<'PYEOF'
import json, sys
a, b, out = sys.argv[1:4]
x = json.load(open(a)); y = json.load(open(b))
for k, v in y.items():
    if k in x and x[k] != v:
        raise SystemExit(f"cache conflict {k}")
    x[k] = v
json.dump(x, open(out, "w"), indent=2)
print(len(x), "cache entries")
PYEOF
  cd $P
  nice -n 10 $PY repro/evaluate_corrected.py --reference-clusters images --root $E/root --prompts /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method $ref --res 4096x4096 --out $O --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000 --threads 8 --no-wandb
done
date -u +%FT%TZ > $E/SETLEVEL_DONE
