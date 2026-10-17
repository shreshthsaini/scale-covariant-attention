#!/bin/bash
# SCA_MIN_SECONDS=4800
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c <<'SCA_FROZEN_HASHES'
09c201611917c5ad0d3dd10037d70ed69a8462191d972376c10a4beae7fa855f  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/sca/attn.py
076a4fc145769fbed1a47db8792470635694077e92ea0b2799c9243a59713484  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/sca/calib.py
511061ce6b8633aae398ce2c3476b8b08ca9fba34c633672c6cec21e4b454f99  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/sca/envfix.py
90a89569feae4fa07b1386d70e0e10bd424b95020111df11168c749f68531a12  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/sca/runtime.py
400c057d4027bb26749d1d75c327f1e360e196d59888a31ddedf8c65b980e54e  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/bench_attn.py
abae373c60b44ee5ef960366dffc6c3aa826616dfed3dbbb9fa9a091becd2f06  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/bench_step.py
db727c269c6a1234c44a78f46e69f6c1fd46a966d3d826fe21ad64d5c650c547  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/calibrate.py
f776b93c5f0ec7c33729efddef9efa11ef1fae1fe68d2d4a7c55dd2c4b83ae27  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/eval_extra.py
002cb68035cee380493c4e588a3f768d33dd8d47153ce6ab51f56b173924323a  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/run_area.py
75e0769b1dad082a0c5e937d837e83cdc9cc0b0c4058fc11c6098f1e615c59f1  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/scripts/sample.py
e8a4ca7550d36931de1a440accc3e8fb64c3891cc58e60999507f145b12dc2bd  /scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947/tests/test_sca.py
SCA_FROZEN_HASHES
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
