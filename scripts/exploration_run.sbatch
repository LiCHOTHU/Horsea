#!/bin/bash
# One explicit experiment phase; capability audits precede full collection.
# sbatch scripts/exploration_run.sbatch BACKEND SPLIT STAGE REFERENCE PRIOR GATE
# STAGE is sweep|active|common. Empty optional arguments may be passed as "".
#SBATCH --job-name=exploration-run
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/exploration/run_%j.out
set -euo pipefail
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
BACKEND=${1:-synthetic}
SPLIT=${2:-dev}
STAGE=${3:-active}
REFERENCE=${4:-}
PRIOR=${5:-}
GATE=${6:-}
OUT="$PWD/experiments/exploration/run_${SLURM_JOB_ID:-local}"
ARGS=(-m exploration.run "$STAGE" --backend "$BACKEND" --split "$SPLIT" --out "$OUT")
if [ -n "$REFERENCE" ]; then ARGS+=(--reference "$REFERENCE"); fi
if [ -n "$PRIOR" ]; then ARGS+=(--prior "$PRIOR"); fi
if [ -n "$GATE" ]; then ARGS+=(--gate "$GATE"); fi
bash scripts/exploration_gpu.sh -m unittest discover -s tests/exploration
bash scripts/exploration_gpu.sh "${ARGS[@]}"
if [ "$STAGE" != sweep ]; then
    bash scripts/exploration_gpu.sh -m exploration.metrics --root "$OUT"
fi
