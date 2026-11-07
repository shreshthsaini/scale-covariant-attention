#!/bin/bash
#SBATCH -J SCA27
#SBATCH -p gh-dev
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
# Round 7 duplication fix (a7_ tasks first, then the review-round spool); workers exit after 15 min idle.
export IDLE_MAX=900
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/node_launch.sh
