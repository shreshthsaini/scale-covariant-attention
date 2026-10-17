#!/bin/bash
#SBATCH -J SCA7
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min=02:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/node_launch.sh
