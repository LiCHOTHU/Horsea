#!/bin/bash
# Observation attribution on PickCube with the stock SAC script: K = plan obs + grasp flag, L = plan obs - object velocity
#SBATCH --job-name=ms-diagN
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%j.out
set -uo pipefail
source /storage/home/hcoda1/8/lwang831/workspace/Horsea/scripts/ms_common.sh
REF=$MS_ROOT/ref; OUT=$MS_ROOT/diag_sbatch; mkdir -p $OUT; cd $REF
COMMON="--env_id PickCube-v1 --num_envs 32 --total_timesteps 150000 --eval_freq 25000 --num_eval_envs 128 --no-capture-video --no-track --control_mode pd_ee_delta_pose"
python sac_myobs_N.py $COMMON --exp_name sbN_myobs_no_vel > $OUT/K.log 2>&1 &
python sac_myobs_N.py $COMMON --seed 2 --exp_name sbN2_myobs_no_vel > $OUT/L.log 2>&1 &
wait
python - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for name in ['A_stock', 'G_stock_myobs', 'sbL_myobs_no_objvel', 'sbN_myobs_no_vel', 'sbN2_myobs_no_vel']:
    for f in glob.glob(f'/storage/project/r-agarg35-0/lwang831/maniskill/ref/runs/{name}/events*'):
        ea = EventAccumulator(f); ea.Reload()
        print(name, 'eval', [(e.step//1000, round(e.value, 3)) for e in ea.Scalars('eval/success_at_end')])
        print(name, 'return', [(e.step//1000, round(e.value, 1)) for e in ea.Scalars('train/return') if e.step % 25000 < 1700])
PY
echo "=== exit $? at $(date) ==="
