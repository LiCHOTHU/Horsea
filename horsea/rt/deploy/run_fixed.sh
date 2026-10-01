#!/bin/bash
# usage: bash run_fixed.sh prevalidate --task T --st_seed S --n N --out F
#        bash run_fixed.sh evaluate --scenes F [--subset N] --overrides --ckpt_path P --seed E --graph G --K K --noise_rep R --ckpt_setting X
# Same runtime environment as eval.sh (NVIDIA Vulkan needs the X display; SAPIEN must not see CUDA_VISIBLE_DEVICES).
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES
cd "$(dirname "$0")/../.."
PYTHONWARNINGS=ignore::UserWarning exec python policy/HorseaFM/fixed_scenes.py "$@"
