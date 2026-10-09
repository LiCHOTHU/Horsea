#!/bin/bash
# Frozen-code development capability/feedback sweep. No confirmation scenes.
# sbatch scripts/exploration_diagnostic.sbatch ABS_RUN_ROOT
# RUN_ROOT contains snapshot/{exploration,tests,scripts}, reference.json.
#SBATCH --job-name=exploration-dev
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%2
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=experiments/exploration/dev_%A_%a.out
set -euo pipefail
RUN_ROOT=$1
export EXPLORATION_SOURCE_ROOT="$RUN_ROOT/snapshot"
SCENE=$((220000 + SLURM_ARRAY_TASK_ID))
OUT="$RUN_ROOT/development/scene$SCENE"
bash "$EXPLORATION_SOURCE_ROOT/scripts/exploration_gpu.sh" -c \
    'import torch; x=torch.randn(512,512,device="cuda"); assert torch.isfinite(x@x).all(); print(torch.cuda.get_device_name())'
bash "$EXPLORATION_SOURCE_ROOT/scripts/exploration_gpu.sh" -m unittest discover -s tests/exploration
bash "$EXPLORATION_SOURCE_ROOT/scripts/exploration_gpu.sh" -m exploration.run sweep \
    --backend robotwin --split dev --scenes "$SCENE" --replicates 0 1 2 \
    --reference "$RUN_ROOT/reference.json" --out "$OUT"
bash "$EXPLORATION_SOURCE_ROOT/scripts/exploration_gpu.sh" -m exploration.development_audit --root "$OUT"
