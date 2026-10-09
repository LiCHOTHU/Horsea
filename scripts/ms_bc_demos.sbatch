#!/bin/bash
# Long-term-memory data: motion-planning demos for the 8 source objects + PickCube, converted to the plan's controller
# and observation (adapt/bc_data.py).  One GPU node, one process per object in parallel (CPU physics, GPU renderer).
#SBATCH --job-name=ms-demos
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%j.out
set -uo pipefail
source /storage/home/hcoda1/8/lwang831/workspace/Horsea/scripts/ms_common.sh
RUN=${RUN:?}; N=${N_DEMOS:-400}; VEL=${VEL:-1.5}; K=${SUBSAMPLE:-2}; NZ=${EXEC_NOISE:-0.0}; MT=${MAX_TRIES:-6000}; OUT=$MS_ROOT/runs/$RUN/demos; mkdir -p $OUT
OBJECTS=${OBJECTS:-"$(python -c "import json; print(' '.join(json.load(open('$SPLIT'))['source']))") cube"}
for OBJ in $OBJECTS; do
  if [ "$OBJ" = cube ]; then
    python adapt/bc_data.py --env PickCube-v1 --object cube --n $N --max_tries $MT --exec_noise $NZ --vel $VEL --subsample $K --out $OUT > $OUT/cube.log 2>&1 &
  else
    python adapt/bc_data.py --object $OBJ --n $N --max_tries $MT --exec_noise $NZ --vel $VEL --subsample $K --out $OUT > $OUT/$OBJ.log 2>&1 &
  fi
done
wait
echo "--- summaries:"; for f in $OUT/*.json; do python -c "import json,sys; d=json.load(open('$f')); print(d['object'], 'kept', d['kept'], 'plan', round(d['plan_success'],2), 'conv', round(d['converted_success'],2), 'len', round(d['length_mean'] or 0,1), 'verify_pos_diff', [round(x,4) for x in d['verify_max_obs_diff']])"; done
for OBJ in $OBJECTS; do test -f $OUT/$OBJ.npz || { echo "missing $OBJ.npz"; rc=1; }; done; rc=${rc:-0}; echo "=== exit $rc at $(date) ==="; exit $rc
