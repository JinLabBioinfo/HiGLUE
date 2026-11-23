#!/bin/bash
# SLURM wrapper of benchmark_main.sh: one preprocessing job, then an array of
# training shards that waits for it. From the repository root on the cluster:
#
#   mkdir -p logs   # SLURM opens the log files before the job starts
#   prep=$(sbatch --parsable --export=ALL,PREP_ONLY=1 benchmark_main.slurm)
#   sbatch --dependency=afterok:$prep --array=0-5 benchmark_main.slurm
#
# 6 datasets x 3 seeds = 18 runs, so --array=0-5 gives 3 runs per task (runs are
# assigned round-robin). Other benchmark_main.sh settings pass through the
# environment, e.g. --export=ALL,DATASETS=islet_atlas,islet_inhouse,SEEDS="1 2 3 4 5".
# Check the partition, resources and environment lines for your cluster.
#
#SBATCH --job-name=higlue_main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/higlue_main_%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# environment with HiGLUE's scglue, SCORE and harmonypy (see README)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate higlue

if [[ ${PREP_ONLY:-0} == 1 ]]; then
    exec ./benchmark_main.sh --prep_only
fi
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    export SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi
exec ./benchmark_main.sh
