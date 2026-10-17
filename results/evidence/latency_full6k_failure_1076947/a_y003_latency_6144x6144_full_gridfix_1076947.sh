#!/bin/bash
# SCA_MIN_SECONDS=4800
set -e
# One excluded warm-up, then three timed images.
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
unset CROPE_TRANSFORMER CROPE_ROPE
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
cd /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947
# Preserve prior latency attempt; never resume timed samples.
if [ -d /scratch/09032/saini_2/sca/out/latency_curve/6144x6144/full ]; then
  mkdir -p /scratch/09032/saini_2/sca/out/latency_history/6144x6144/full
  sca_latency_previous=$(mktemp -d /scratch/09032/saini_2/sca/out/latency_history/6144x6144/full/gridfix_1076947_XXXXXXXX)
  mv -- /scratch/09032/saini_2/sca/out/latency_curve/6144x6144/full "$sca_latency_previous/attempt"
  echo "preserved prior latency attempt: $sca_latency_previous"
fi
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/sample.py --method full --res 6144x6144 --prompts /scratch/09032/saini_2/sca/latency4_warmup.jsonl --seed 0 --guidance 4.5 --out /scratch/09032/saini_2/sca/out/latency_curve/6144x6144/full
