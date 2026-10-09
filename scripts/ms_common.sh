# shared environment for the ManiSkill adaptation jobs (sourced by every ms_*.sbatch)
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
export PYTHONPATH=/storage/home/hcoda1/8/lwang831/workspace/Horsea
export MS_ASSET_DIR=/storage/project/r-agarg35-0/lwang831/maniskill/assets
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/mpl
export PIP_CACHE_DIR=/storage/project/r-agarg35-0/lwang831/cache/pip
[ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ] && export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_ROOT=/storage/project/r-agarg35-0/lwang831/maniskill
export SPLIT=$MS_ROOT/split.json
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
echo "=== node $(hostname) job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-} start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
