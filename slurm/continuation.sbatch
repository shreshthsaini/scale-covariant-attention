#!/bin/bash
# One CPU controller serves the entire campaign, including both CPU finalizers.
# It does useful CPU work, then exits with its successor waiting on GPU events.
#SBATCH -J SCAC
#SBATCH -p gg
#SBATCH -A ASC26048
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /scratch/09032/saini_2/sca/continuation/slurm_%j.log
set -euo pipefail
export PATH=/home1/09032/saini_2/.local/bin:$PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
exec /work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  /scratch/09032/saini_2/sca/dispatch/20261017_continuation/continuation.py run
