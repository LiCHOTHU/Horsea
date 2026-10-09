#!/bin/bash
set -euo pipefail
cd "${EXPLORATION_SOURCE_ROOT:-/storage/home/hcoda1/8/lwang831/workspace/Horsea}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export MPLCONFIGDIR="$PWD/experiments/exploration/runtime/mpl"
export TORCH_EXTENSIONS_DIR="$PWD/experiments/exploration/runtime/extensions"
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi
exec /storage/project/r-agarg35-0/lwang831/conda/envs/robotwin/bin/python "$@"
