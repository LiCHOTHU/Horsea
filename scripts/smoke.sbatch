#!/bin/bash
#SBATCH --job-name=horsea-smoke
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0
#SBATCH --output=%x_%j.out

echo "node: $(hostname)"
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
bash scripts/smoke.sh cuda:0
echo "EXIT=$?"
