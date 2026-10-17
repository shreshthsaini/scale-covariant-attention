#!/bin/bash
# SCA_MIN_SECONDS=2700
set -e
# E2 eval 6144x6144, reference area (Aesthetic-4K references are 4096 px)
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=scale-covariant-attention WANDB_DIR=/scratch/09032/saini_2/sca/wandb
unset CROPE_TRANSFORMER CROPE_ROPE
ls /scratch/09032/saini_2/sca/taskq/failed 2>/dev/null | grep -E '^[i][0-9]{3}_(area_6144x6144|sca_nb_6144x6144)[.]sh' | sed 's/^/WARNING failed generation task: /'
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/scripts/link_common.py --root /scratch/09032/saini_2/sca/out/evalroot/e2_6144 --res 6144x6144 --seed 0 --prompts /scratch/09032/saini_2/sca/eval100.jsonl --min-count 100 sca_nb=/scratch/09032/saini_2/sca/out/main/sca_nb/6144x6144 area=/scratch/09032/saini_2/sca/out/main/area/6144x6144
/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/scripts/eval_extra.py --crope-root /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/evaluator_snapshot --root /scratch/09032/saini_2/sca/out/evalroot/e2_6144 --prompts /scratch/09032/saini_2/sca/eval100.jsonl --out /scratch/09032/saini_2/sca/eval/e2_6144_extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method area --device cuda --batch-size 16 --threads 8
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/evaluate_corrected.py --reference-clusters images --musiq-query-chunk 256 --root /scratch/09032/saini_2/sca/out/evalroot/e2_6144 --prompts /scratch/09032/saini_2/sca/eval100.jsonl --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method area --res 6144x6144 --out /scratch/09032/saini_2/sca/eval/e2_6144_base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0
cp -- /scratch/09032/saini_2/sca/eval/e2_6144_base/metrics.json /scratch/09032/saini_2/sca/eval/e2_6144_base/metrics_gpu_features.json
touch /scratch/09032/saini_2/sca/eval/e2_6144_base/READY_FOR_CPU_BOOTSTRAP
