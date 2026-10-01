#!/bin/bash
# usage: bash eval.sh <task_name> <task_config> <ckpt_path> <seed> <gpu_id> [ckpt_setting]
policy_name=HorseaFM
task_name=${1}; task_config=${2}; ckpt_path=${3}; seed=${4}; gpu_id=${5}; ckpt_setting=${6:-base}
# NVIDIA Vulkan on this host only initializes with an X connection; SAPIEN also must not see CUDA_VISIBLE_DEVICES
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # CLIP is cached; no network dependency at eval start
cd ../..
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides --task_name ${task_name} --task_config ${task_config} --ckpt_setting ${ckpt_setting} \
    --ckpt_path ${ckpt_path} --seed ${seed} --policy_name ${policy_name} \
    --graph ${HORSEA_GRAPH:-G0} --K ${HORSEA_K:-10} --noise_rep ${HORSEA_NOISE_REP:-0}
