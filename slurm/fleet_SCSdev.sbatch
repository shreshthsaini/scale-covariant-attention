#!/bin/bash
#SBATCH -J SCSd
#SBATCH -p gh-dev
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
# Showcase backfill (HANDOVER Section 32): 4 GH200 dev nodes for 2 h draining taskq (00s* first); the gh fleets SCS1/SCS2
# were estimated to start 2026-11-13. Workers exit after 10 min idle; on the walltime signal a running task goes back
# to pending (worker_loop.sh) and resumes on the next worker.
export IDLE_MAX=600
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/slurm/node_launch.sh
