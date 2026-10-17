#!/bin/bash
# SCA_MIN_SECONDS=3600
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/09032/saini_2/sca/cache/baseline_expansion_20261017/inductor
export TRITON_CACHE_DIR=/scratch/09032/saini_2/sca/cache/baseline_expansion_20261017/triton
export CC=/usr/bin/gcc CXX=/usr/bin/g++
"$CC" --version | head -n 1
printf "int main(void){return 0;}\n" | "$CC" -x c -Wno-psabi -fsyntax-only -
sha256sum -c <<'SCA_BASELINE_HASHES'
5fa0dc4b9bd30a50e7ab742f2e9f6756936c46bdc245ee3996fd0d05b079c8b2  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/run.py
ec8fc81553e630971923b843405a07c5d4059632387704debf0b43a99553d169  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/sources.json
b7a600e951bcce23e6ce452b2113527cc5a8acb349d213246d992ed1638e32ec  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/model_manifest.json
3f9c6210eedd4e9a28cad8d5750a1d85e3d8082bb348e55103152c3afe5966fd  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/dependency_overlay.json
2e6c258f0e9af10831febd60b2d05879645c382736a4d51c71c9a49e0cd536e8  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/pilot46.jsonl
705cbd73004b2086ec7fce0fa79de13fee48e83db95b9a5423fc11c522c87785  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/LICENSE
03d6c1ac0091c6d835ab045bf272d051ac3d935a88f50800c03da531256d55d8  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/README.md
3008e8cfc5c0b2eeadf901159da1e1d584938762ab4632c24df2f0d7dca93413  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/configs/pipeline.json
f8953b8cb981bc85b90f9083978037caf5a656f82e62c3ced09c77944b51f02f  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/configs/scope_plan_flux.json
ecdf100125e5da6d64124d96bc2c68531ebe8e2a34af7cdcea8ce2e9097cec7e  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/__init__.py
22044ed73aa519ba043f20d87a995a7aaa6f277d532d9d4123e7363b249931aa  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/attention.py
be329b645c2c518576945a4bae6ab21182823bf2f053b898f20798f907187633  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/hap.py
0a64a6a4da8fcc68c3db0c9e5397b8d78902ec6f95d0ff28dd02c494dca4d82e  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/pipeline.py
84a6554678b401673399529be82e690641efac69f9e4dbb24b5a30e7f14f167e  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/spa.py
22329549070dadabb72fe9c7690254bcc99c9f8f8cc3f24700305e2e11b3eda4  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/transformer.py
f17cb4eec41562d0e424aee1ef5a82712c6de820dd3eb1b65a16b2b8e1a9a64f  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/inference.py
cae4a80a6e1ef46ede77315de3b73422cb7baee17dab9f54b8305b4e30393c88  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/requirements.txt
84f7993926962aed0eeaf63f530dfb13af137899f8e52961e95d3fc6ff00967d  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/LICENSE
bdb1ed63127c6337856314017b1781bcd4c410348fd5b9e595c7a1b650d2cb97  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/README.md
3ac61360c9d5cd7fe3e5cae837160f18a9bd703cd7455acc9ff498e809a5a278  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/flux/__init__.py
f1e061d94df07339d513ebfcfe5b6cfe41732246812e55754fce6a0a35949b00  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/flux/pipeline_flux.py
1c3d0e67e2b16df3832b317decc3dae60d4c4858fb15c2ede71079903c1d6758  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/flux/transformer_flux.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/qwen/__init__.py
4de95c67924cbeaca45de4a6a42297d0a5cc0c18d630198bab5b76953c6f7231  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/qwen/transformer_qwenimage.py
5076b184e4b5870aa4ecf0ed4de1e2413c2e7f74f7eba531c8ba5713e31b4a09  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/requirements.txt
ac0fa2f4a363e4611d0f2f54c3286e8174c6d26d0353f3f924cfc0dfdd99956a  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/run_sigma.py
d2a7d7c28813c3b9ff876a137c995b0c81aae69cf99b3c3d353e9896cb390a1d  /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/sigma/run_sigma_qwen.py
SCA_BASELINE_HASHES
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/run.py --profile hrdit_dev_author --prompts /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/pilot46.jsonl --start 0 --end 1 --seed 0 --out /scratch/09032/saini_2/sca/out/baseline_expansion_20261017_v1 --dev-snapshot /scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --mode generate --extra-site /work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017/lib/python3.11/site-packages
