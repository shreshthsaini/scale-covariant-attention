#!/bin/bash
#SBATCH -J SCS2
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
# Showcase fleet 2 (HANDOVER Section 32): 4 GH200 nodes draining taskq (00s* showcase tasks first); workers exit after 30 min idle.
export IDLE_MAX=1800
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/node_launch.sh
