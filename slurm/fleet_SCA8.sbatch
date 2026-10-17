#!/bin/bash
# Packed completion fleet. Frozen dispatcher paths are prepared before submission.
#SBATCH -J SCA8
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min=02:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
srun --ntasks-per-node=1 bash /scratch/09032/saini_2/sca/dispatch/20261017_completion/node_launch.sh
