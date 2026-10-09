#!/bin/bash
# Stages 1 + 4: cached encoder features, then the Phi action autoencoder.
# Both are needed by `horsea.energy train` (the memory experiment) in addition to the self-play data.
# Runs after the base policy exists; launched automatically by pace_train_base.sbatch on completion.
#SBATCH --job-name=horsea-featphi
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
TLOG=$HORSEA_EXP/libero/libero_90_train80/base80/fm/training.log

python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }

grep -q "Training completed successfully" "$TLOG" 2>/dev/null \
    || { echo "base training has not completed -- refusing to build features from a partial model"; exit 1; }

echo "=== features: libero_90 ==="
python -m horsea.features --suite libero_90 || exit 1

# libero_10 is optional here: only the far-novel transfer stage needs it, and it depends on
# scripts/process_libero_10.sbatch having produced the processed demos.
if [ -n "$(ls /storage/home/hcoda1/8/lwang831/workspace/imitation/data/libero/libero_10 2>/dev/null)" ]; then
    echo "=== features: libero_10 ==="
    python -m horsea.features --suite libero_10 || echo "libero_10 features failed (non-fatal)"
else
    echo "=== skipping libero_10 features: processed demos not present ==="
fi

echo "=== phi: action autoencoder ==="
python -m horsea.phi train || exit 1

echo "=== protocol manifest ==="
python -m horsea.manifest || echo "manifest failed (non-fatal)"

echo "FEATURES+PHI COMPLETE  end: $(date)"
bash scripts/pace_launch_fair.sh   # fires the comparison array if self-play is ready too
