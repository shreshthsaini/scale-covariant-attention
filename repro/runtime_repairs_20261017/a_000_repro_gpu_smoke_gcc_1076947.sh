#!/bin/bash
# SCA_MIN_SECONDS=420
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c <<'SCA_FROZEN_HASHES'
29238d938c36c1ef6bc6c50f7676bafe6ce6b322c707bae0e078c8a8c5e8d4fa  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/repro/environment/generation.lock
e02a2a7ba97b2571f43344518e2366ab84109acf9fc7eca4568953a7ede0e14d  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/repro/environment/gpu_smoke.py
09c201611917c5ad0d3dd10037d70ed69a8462191d972376c10a4beae7fa855f  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/sca/attn.py
076a4fc145769fbed1a47db8792470635694077e92ea0b2799c9243a59713484  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/sca/calib.py
511061ce6b8633aae398ce2c3476b8b08ca9fba34c633672c6cec21e4b454f99  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/sca/envfix.py
d50fbe6470ca29973be5177cb191a6103cb26dab3da5d73e272e0d55b4727755  /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/sca/runtime.py
SCA_FROZEN_HASHES
cd /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017/bin/python python /scratch/09032/saini_2/sca/code/gpu_smoke_gcc_1076947/repro/environment/gpu_smoke.py --cc /usr/bin/gcc --cxx /usr/bin/g++ --out /scratch/09032/saini_2/sca/repro_env_gpu_compiler_repair_1076947/gpu_retry.json --archive results/evidence/environment/fresh_gpu_smoke_gcc_1076947.json
