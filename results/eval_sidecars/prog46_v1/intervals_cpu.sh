#!/bin/bash
set -euo pipefail
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python /scratch/09032/saini_2/sca/code/PROG/repro/progressive_sca/prepare_eval.py verify --workspace /scratch/09032/saini_2/sca/eval/prog46_v1
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python /scratch/09032/saini_2/sca/eval/prog46_v1/code/repro/evaluate_corrected.py --root /scratch/09032/saini_2/sca/eval/prog46_v1/samples --prompts /scratch/09032/saini_2/sca/eval/prog46_v1/prompts.jsonl --reference-method sca_nb --res 4096x4096 --threads 4 --references /scratch/09032/saini_2/sca/eval/prog46_v1/references.jsonl --reference-clusters images --out /scratch/09032/saini_2/sca/eval/prog46_v1/base --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python /scratch/09032/saini_2/sca/code/PROG/repro/progressive_sca/prepare_eval.py verify --workspace /scratch/09032/saini_2/sca/eval/prog46_v1
