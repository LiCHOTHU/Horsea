#!/bin/bash
# Source preparation: PickCube implementation check (ENV=PickCube-v1, no array) or source-skill training on the
# split's 8 YCB objects (ENV=ycb, --array=0-2 -> seed).  Resumable; exit 3 = time cap (chain another block).
#SBATCH --job-name=ms-source
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%A_%a.out
set -uo pipefail
source /storage/home/hcoda1/8/lwang831/workspace/Horsea/scripts/ms_common.sh
ENV=${ENV:-ycb}; RUN=${RUN:?}; TOTAL=${TOTAL_STEPS:-3000000}
if [ "$ENV" = "PickCube-v1" ]; then
  python adapt/train_source.py --env PickCube-v1 --seed 0 --out $MS_ROOT/runs/$RUN/pickcube_check \
    --num_envs 128 --total_steps ${TOTAL} --check_only --time_cap_h 7.4
else
  SEED=${SLURM_ARRAY_TASK_ID:-0}
  python adapt/train_source.py --env PickSingleYCBSplit-v1 --split $SPLIT --seed $SEED --out $MS_ROOT/runs/$RUN/source_s$SEED \
    --num_envs 128 --total_steps ${TOTAL} --time_cap_h 7.4
fi
rc=$?; echo "=== exit $rc at $(date) ==="; exit $rc
