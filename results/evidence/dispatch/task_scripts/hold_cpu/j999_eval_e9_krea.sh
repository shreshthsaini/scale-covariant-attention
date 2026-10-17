#!/bin/bash
# SCA_MIN_SECONDS=2700
set -e
# E9 eval: Krea, 100 prompts; DyPE/SEGA Krea outputs reused from constancy_rope out_krea/main
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=scale-covariant-attention WANDB_DIR=/scratch/09032/saini_2/sca/wandb
unset CROPE_TRANSFORMER CROPE_ROPE
ls /scratch/09032/saini_2/sca/taskq/failed 2>/dev/null | grep -E '^(a_)?[j][0-9]{3}_(area_krea_4096x4096|sca_krea_4096x4096)[.]sh' | sed 's/^/WARNING failed generation task: /'
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/scripts/link_common.py --root /scratch/09032/saini_2/sca/out/evalroot/e9_krea --res 4096x4096 --seed 0 --prompts /scratch/09032/saini_2/sca/eval100.jsonl --min-count 100 sca_krea=/scratch/09032/saini_2/sca/out/krea/sca_krea/4096x4096 area=/scratch/09032/saini_2/sca/out/krea/area/4096x4096 dype=/scratch/09032/saini_2/constancy_rope/out_krea/main/dype/4096x4096 sega_official=/scratch/09032/saini_2/constancy_rope/out_krea/main/sega_official/4096x4096
/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/scripts/eval_extra.py --crope-root /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/evaluator_snapshot --root /scratch/09032/saini_2/sca/out/evalroot/e9_krea --prompts /scratch/09032/saini_2/sca/eval100.jsonl --out /scratch/09032/saini_2/sca/eval/e9_krea_extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method area --device cuda --batch-size 16 --threads 8
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/evaluate_corrected.py --reference-clusters images --root /scratch/09032/saini_2/sca/out/evalroot/e9_krea --prompts /scratch/09032/saini_2/sca/eval100.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method area --res 4096x4096 --out /scratch/09032/saini_2/sca/eval/e9_krea_base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0
cp -- /scratch/09032/saini_2/sca/eval/e9_krea_base/metrics.json /scratch/09032/saini_2/sca/eval/e9_krea_base/metrics_gpu_features.json
touch /scratch/09032/saini_2/sca/eval/e9_krea_base/READY_FOR_CPU_BOOTSTRAP
