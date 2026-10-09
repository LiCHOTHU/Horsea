#!/bin/bash
# Development adaptation array: 2 dev objects x 3 source seeds x 3 methods = 18 cells (--array=0-17).
# One cumulative stream per cell with snapshots at B in {0,500,2500,10000,25000}; resumable (exit 3 = time cap).
#SBATCH --job-name=ms-adapt
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%A_%a.out
set -uo pipefail
source /storage/home/hcoda1/8/lwang831/workspace/Horsea/scripts/ms_common.sh
RUN=${RUN:?}; I=${SLURM_ARRAY_TASK_ID:?}
METHODS=(A0 A1 A2); M=${METHODS[$((I % 3))]}; SEED=$(((I / 3) % 3)); OI=$((I / 9))
OBJ=$(python -c "import json,sys; print(json.load(open('$SPLIT'))['dev'][$OI])")
echo "cell: object=$OBJ source_seed=$SEED method=$M"
python adapt/adapt_run.py --split $SPLIT --object $OBJ --method $M --source_seed $SEED --adapt_seed 0 \
  --init_dir $MS_ROOT/runs/$RUN/source_s$SEED --out $MS_ROOT/runs/$RUN/dev/${OBJ}__s${SEED}__a0__${M} --time_cap_h 7.6
rc=$?; echo "=== exit $rc at $(date) ==="; exit $rc
