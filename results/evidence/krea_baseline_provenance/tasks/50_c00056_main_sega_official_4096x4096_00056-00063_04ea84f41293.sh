#!/usr/bin/env bash
set -euo pipefail
# Fleet slurm/fleet.sbatch provides per-host GPU telemetry.
# Input fingerprint: 04e611ab83edb24f37be043a2370013c3223fb6a54b7967d9bdbc94293b9b7ea
cd /work/09032/saini_2/vista/Projects/Exploration/constancy_rope
export PATH=/work/09032/saini_2/vista/library/uv-envs/crope/bin:"$PATH"
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export CROPE_TRANSFORMER=/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28
export TORCH_HOME=/scratch/09032/saini_2/tmp/cache
export XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache
export WANDB_DIR=/scratch/09032/saini_2/constancy_rope/wandb
export WANDB_PROJECT=constancy-rope
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[[ -s /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl ]] || { printf '%s\n' 'Missing prerequisite: /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl. Generate this stage after its dependencies finish.' >&2; exit 2; }
/work/09032/saini_2/vista/library/uv-envs/crope/bin/python scripts/sample.py --method sega_official --res 4096x4096 --prompts /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --start 56 --end 63 --seed 0 --steps 28 --guidance 4.5 --out /scratch/09032/saini_2/constancy_rope/out_krea/main/sega_official/4096x4096
