#!/bin/bash
# Closed-loop RoboTwin eval of the FM policy:  sbatch scripts/pace_fm_eval.sbatch TASK [TEST_NUM]
#SBATCH --job-name=fm-eval
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-a100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/home/hcoda1/8/lwang831/workspace/Horsea/experiments/robotwin/eval_%x_%j.out
set -uo pipefail
TASK=$1; N=${2:-100}
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
export PYTHONPATH=/storage/home/hcoda1/8/lwang831/workspace/RoboTwin:/storage/home/hcoda1/8/lwang831/workspace/Horsea
export ROBOTWIN_DATA_ROOT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/RoboTwin-data/demo_clean
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache HF_HUB_CACHE=/storage/project/r-agarg35-0/lwang831/hf_cache/hub
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl TEST_NUM=$N
[ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ] && export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
echo "=== node $(hostname)  task=$TASK  N=$N  $(date) ==="; nvidia-smi --query-gpu=name,ecc.errors.uncorrected.volatile.total --format=csv,noheader
conda activate Horsea && python -c 'import torch; torch.randn(4096,4096,device="cuda:0"); print("GPU alloc OK")' || { echo "GPU UNUSABLE"; exit 1; }
cd /storage/home/hcoda1/8/lwang831/workspace/RoboTwin/XPolicyLab/policy/FM
bash eval.sh demo_clean "$TASK" fm_policy_M aloha_agilex ee 0 0 0 Horsea robotwin
R=$(ls -td /storage/home/hcoda1/8/lwang831/workspace/RoboTwin/eval_result/$TASK/FM/demo_clean/fm_policy_M/* | head -1)
echo "RESULT $TASK: $(cat $R/_result.txt 2>/dev/null)  dir=$R  $(date)"
