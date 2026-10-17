#!/bin/bash
# Freeze worker_loop_v2.py, node_launch_v2.sh, and telemetry.sh before submission.
# Check the current SU balance before submitting; this file does not submit jobs.
#SBATCH -J SCA9
#SBATCH -p gh
#SBATCH --nodes=2-8
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min=01:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/sca/logs/fleet_%j.out
set -euo pipefail
dispatch=/scratch/09032/saini_2/sca/dispatch/20261017_v2
for controller in worker_loop_v2.py node_launch_v2.sh telemetry.sh; do
    test -r "$dispatch/$controller" || { echo "Missing frozen controller: $controller" >&2; exit 1; }
done

# This runs once in the batch root. Workers never query the scheduler.
now=$(date +%s)
deadline=${SLURM_JOB_END_TIME:-}
if [[ ! "$deadline" =~ ^[0-9]+$ ]] || (( deadline <= now )); then
    job_record=$(scontrol show job "${SLURM_JOB_ID:?}" -o)
    if [[ "$job_record" =~ (^|[[:space:]])EndTime=([^[:space:]]+) ]]; then
        deadline=$(date -d "${BASH_REMATCH[2]}" +%s)
    else
        echo "Cannot obtain the actual allocation EndTime" >&2
        exit 1
    fi
fi
if [[ ! "$deadline" =~ ^[0-9]+$ ]] || (( deadline <= now )); then
    echo "Invalid or expired allocation deadline: $deadline" >&2
    exit 1
fi
: "${SLURM_JOB_NUM_NODES:?actual allocated node count is required}"
[[ "$SLURM_JOB_NUM_NODES" =~ ^[1-9][0-9]*$ ]] || exit 1
export SCA_DEADLINE_EPOCH="$deadline"
export SLURM_EXPORT_ENV=ALL
echo "SCA9 nodes=$SLURM_JOB_NUM_NODES deadline=$SCA_DEADLINE_EPOCH dispatch=$dispatch"
exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    --ntasks-per-node=1 bash "$dispatch/node_launch_v2.sh"
