#!/bin/bash
# SHAKEDOWN of the DSRL / ReinFlow adapters (horsea/baselines) on the epoch-9/50 base (NOT the final model).
# Tiny budgets and lowered update thresholds so both methods reach updates and every snapshot path quickly.
# Separate root under experiments_e9_shakedown; numbers here are not results.
#SBATCH --job-name=horsea-bl-smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-a100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments_e9_shakedown/logs/%x_%j.out
set -u
echo "node: $(hostname)  start: $(date)  job: $SLURM_JOB_ID"
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
export HORSEA_EXP=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments_e9_shakedown MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/mpl
python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }
OUT=$HORSEA_EXP/baselines/smoke_$SLURM_JOB_ID
COMMON="--task 22 --shift rot30 --budgets 0,4,8 --par 4 --eval_inits 30,31,32,33 --seed 0"
echo "=== dsrl $(date)"
python -m horsea.baselines.dsrl $COMMON --start_online_updates 20 --utd 2 --out $OUT/dsrl || echo "DSRL FAILED"
echo "=== reinflow $(date)"
python -m horsea.baselines.reinflow $COMMON --episodes_per_itr 4 --n_critic_warmup_itr 0 --out $OUT/reinflow || echo "REINFLOW FAILED"
echo "=== resume check $(date)"  # a finished stream must exit immediately
python -m horsea.baselines.dsrl $COMMON --out $OUT/dsrl | tail -1
echo "=== END $(date)"; cat $OUT/*/snapshots.json | python -c "import sys;print(sys.stdin.read()[:4000])"
