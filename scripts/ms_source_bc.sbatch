#!/bin/bash
# Long-term memory: BC on the demos (fixed input transform: standardise + mask velocities in the actor), then the
# shared source phases: BC gate -> (if < gate) warm start + dense-reward SAC fine-tune -> gate -> terminal-reward
# collection -> critic refit.  ENV=PickCube-v1 (implementation check, no array) or ENV=ycb with --array=0-2 (seed).
# Resumable; exit 3 = time cap (chain another block with afterany).
#SBATCH --job-name=ms-srcbc
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
ENV=${ENV:-ycb}; RUN=${RUN:?}; SEED=${SLURM_ARRAY_TASK_ID:-0}; R=$MS_ROOT/runs/$RUN
BC_ARGS="--standardize ${BC_STD:-1} --mask ${BC_MASK:-qvel,obj_lin_vel,obj_ang_vel} --input_noise ${BC_NOISE:-0.1} --iters ${BC_ITERS:-30000}"
if [ "$ENV" = "PickCube-v1" ]; then
  O=$R/pickcube_check; mkdir -p $O
  [ -f $O/bc_actor.pt ] || python adapt/bc_train.py --env PickCube-v1 --data_dir $R/demos --out $O --seed 0 $BC_ARGS || exit 1
  python adapt/train_source.py --env PickCube-v1 --seed 0 --out $O --init_actor $O/bc_actor.pt --check_only \
         --total_steps ${TOTAL_STEPS:-300000} --time_cap_h ${TIME_CAP_H:-7.4}
else
  O=$R/source_s$SEED; mkdir -p $O
  [ -f $O/bc_actor.pt ] || python adapt/bc_train.py --env PickSingleYCBSplit-v1 --split $SPLIT --data_dir $R/demos --out $O --seed $SEED $BC_ARGS || exit 1
  python adapt/train_source.py --env PickSingleYCBSplit-v1 --split $SPLIT --seed $SEED --out $O --init_actor $O/bc_actor.pt \
         --total_steps ${TOTAL_STEPS:-3000000} --time_cap_h ${TIME_CAP_H:-7.4}
fi
rc=$?; echo "=== exit $rc at $(date) ==="; exit $rc
