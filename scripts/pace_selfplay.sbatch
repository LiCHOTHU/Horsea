#!/bin/bash
# Stage 3: hidden-shift self-rollout ("explore") data -- the robot's own attempts that the writer
# learns from. Needs the base policy, so submit with --dependency=afterok:<base job id>.
#
# Shift sets follow the energy-Horsea protocol (README example + DEVLOG stage-2):
#   writer_train (70 tasks): none +-20 +-40 +-60   -> the training conditions
#   writer_dev   (10 tasks): none +-30 +-50        -> the dev conditions used to tune/select
# Resumable: one .pt per (task, shift); existing files are skipped.
#SBATCH --job-name=horsea-selfplay
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-01-004-33-0
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments/logs/%x_%j.out

set -u
echo "node: $(hostname)  start: $(date)"
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea

export HORSEA_EXP=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments

python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }

CK=$HORSEA_EXP/libero/libero_90_train80/base80/fm/multitask_model.pth
TLOG=$HORSEA_EXP/libero/libero_90_train80/base80/fm/training.log
[ -f "$CK" ] || { echo "base checkpoint missing: $CK"; exit 1; }
# the checkpoint is rewritten every 10 epochs, so its presence does NOT mean training finished
grep -q "Training completed successfully" "$TLOG" 2>/dev/null \
    || { echo "base training has not completed -- refusing to collect on a partial model"; exit 1; }

echo "=== writer_train: none +-20 +-40 +-60 ==="
python -m horsea.selfplay collect --split writer_train \
    --shifts none rot20 rot-20 rot40 rot-40 rot60 rot-60 --attempts 2 --par 8
rc=$?

echo "=== writer_dev: none +-30 +-50 ==="
python -m horsea.selfplay collect --split writer_dev \
    --shifts none rot30 rot-30 rot50 rot-50 --attempts 2 --par 8
rc2=$?

echo "EXIT train=$rc dev=$rc2  end: $(date)"
if [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; then
    echo "SELFPLAY COMPLETE"
    bash scripts/pace_launch_fair.sh   # fires the comparison array if Phi is ready too
    exit 0
fi
echo "incomplete; chaining another 8 h block (collection is resumable per task x shift)"
sbatch scripts/pace_selfplay.sbatch
