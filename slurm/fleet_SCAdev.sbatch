#!/bin/bash
#SBATCH -J SCAdev
#SBATCH -p gh-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
# One held 8-node fleet for all remaining SCA GPU work; workers park up to 4 h when idle.
export IDLE_MAX=1800
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/node_launch.sh
