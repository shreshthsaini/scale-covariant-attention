#!/bin/bash
# SCA_MIN_SECONDS=420
set -euo pipefail
printf '%s\n' '20771b9e766d3c177259f4ac269458548aa2b7e6d1b109da7db54c942b326158  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/environment/gpu_smoke.py' | sha256sum -c -
printf '%s\n' '29238d938c36c1ef6bc6c50f7676bafe6ce6b322c707bae0e078c8a8c5e8d4fa  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/environment/generation.lock' | sha256sum -c -
printf '%s\n' '09c201611917c5ad0d3dd10037d70ed69a8462191d972376c10a4beae7fa855f  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/sca/attn.py' | sha256sum -c -
cd /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017/bin/python python repro/environment/gpu_smoke.py --out /scratch/09032/saini_2/sca/repro_env_gpu_20261017/smoke.json --archive results/evidence/environment/fresh_gpu_smoke.json
