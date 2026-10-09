#!/bin/bash
#SBATCH --job-name=exploration-smoke
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-v100,gpu-a100,gpu-l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/exploration/slurm_%j.out
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
set -euo pipefail
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPLORE_OUT="$PWD/experiments/exploration/smoke_${SLURM_JOB_ID:-local}"
mkdir -p "$EXPLORE_OUT/runtime"
export TORCH_EXTENSIONS_DIR="$EXPLORE_OUT/runtime/torch_extensions"
export MPLCONFIGDIR="$EXPLORE_OUT/runtime/mpl"
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi
export MUJOCO_GL=egl
EXPLORE_PY=/storage/project/r-agarg35-0/lwang831/conda/envs/robotwin/bin/python
echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,ecc.errors.uncorrected.volatile.total --format=csv,noheader || true

"$EXPLORE_PY" -c 'import torch; torch.randn(4096,4096,device="cuda:0"); print("GPU alloc OK")' \
  || { echo "GPU UNUSABLE on $(hostname) -- failing fast"; exit 1; }

"$EXPLORE_PY" -m unittest discover -s tests/exploration -v
"$EXPLORE_PY" -m exploration.robotwin audit --out "$EXPLORE_OUT" --expert
