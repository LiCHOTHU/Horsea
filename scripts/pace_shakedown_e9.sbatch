#!/bin/bash
# SHAKEDOWN of the LIBERO pipeline on the epoch-9/50 base checkpoint (NOT the final model): features -> Phi ->
# reduced self-play (writer_dev full, 14 writer_train tasks) -> one energy arm + one TTT arm, seed 0.
# Purpose: validate every downstream stage on real weights and get timing + preliminary numbers while the
# base finishes training. Separate experiment root; nothing here feeds the real run.
#SBATCH --job-name=horsea-shake
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-a100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments_e9_shakedown/logs/%x_%j.out
set -u
echo "node: $(hostname)  start: $(date)  job: $SLURM_JOB_ID"
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
export HORSEA_EXP=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments_e9_shakedown MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/mpl
python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }
t() { echo "=== $1  $(date)"; }
t features;  [ -f $HORSEA_EXP/features/libero_90.pt ] || python -m horsea.features --suite libero_90 --max_demos 25 || exit 1
t phi;       [ -f $HORSEA_EXP/phi/phi.pt ] || python -m horsea.phi train --steps 10000 || exit 1
t selfplay_dev;   python -m horsea.selfplay collect --split writer_dev --shifts none rot30 rot-30 rot50 rot-50 --attempts 2 --par 8 || exit 1
t selfplay_train; python -m horsea.selfplay collect --split writer_train --tasks 0 1 3 5 6 7 8 10 11 12 15 16 17 18 --shifts none rot20 rot-20 rot40 rot-40 rot60 rot-60 --attempts 2 --par 8 || exit 1
SH_TR="none rot20 rot-20 rot40 rot-40 rot60 rot-60"; SH_DEV="none rot30 rot-30 rot50 rot-50"
t energy_horsea; python -m horsea.energy train --solve final --n_iter 3 --max_step 0.05 --objective dphi --seed 0 --steps 1500 --train_shifts $SH_TR --dev_shifts $SH_DEV --out $HORSEA_EXP/fair/horsea_s0 || echo "energy train FAILED"
t writer_ttt2;   python -m horsea.writer train --arm ttt2 --seed 0 --lam_old 0 --train_shifts $SH_TR --dev_shifts $SH_DEV --out $HORSEA_EXP/fair/ttt2_s0 || echo "writer train FAILED"
t done; ls -la $HORSEA_EXP/fair/*/ 2>/dev/null | head -20; echo "=== SHAKEDOWN END $(date)"
