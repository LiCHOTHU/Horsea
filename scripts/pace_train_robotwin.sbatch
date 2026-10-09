#!/bin/bash
# RoboTwin base policy: train until the validation loss stops improving.
#
#   sbatch scripts/pace_train_robotwin.sbatch [algo] [out_dir]
#
# Early stopping decides the length, so --epochs is only an upper bound. embers caps wall time at
# 8 h, so chain blocks with --dependency=afterany if a run needs longer; the best checkpoint is
# rewritten whenever validation improves, so a killed block never loses the best model.
#
# Account must be gts-agarg35: gts-agarg35-ideas_l40s is restricted to the l40s partition, where
# three nodes developed uncorrectable ECC faults.
#SBATCH --job-name=rt-train
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-a100,gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/home/hcoda1/8/lwang831/workspace/Horsea/experiments/robotwin/train_%j.out
set -uo pipefail
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
ALGO=${1:-fm_policy_M}
OUT=${2:-/storage/home/hcoda1/8/lwang831/workspace/Horsea/experiments/robotwin/${ALGO}}

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
# XPolicyLab supplies decode_image_bit; the stored image bits come in two byte formats and only
# that function distinguishes them.
export PYTHONPATH=/storage/home/hcoda1/8/lwang831/workspace/RoboTwin
export ROBOTWIN_DATA_ROOT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/RoboTwin-data/demo_clean
export PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$(dirname "$OUT")"

echo "=== node: $(hostname)  algo=$ALGO  start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total,ecc.errors.uncorrected.volatile.total --format=csv,noheader || true
# torch.cuda.is_available() is True on a node with uncorrectable ECC errors; only a real allocation
# exposes the fault. Fail in seconds rather than hours.
python -c 'import torch; torch.randn(4096,4096,device="cuda:0"); print("GPU alloc OK")' \
  || { echo "GPU UNUSABLE on $(hostname) -- failing fast"; exit 1; }

python scripts/train_robotwin.py --algo "$ALGO" --out "$OUT" \
    --epochs 400 --patience 15 --batch-size 64 --lr 1e-4 --workers 8 --val-fraction 0.15
rc=$?
echo "EXIT=$rc  end: $(date)"
exit $rc
