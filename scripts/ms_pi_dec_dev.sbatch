#!/bin/bash
# Policy Decorator dev grid on the one informative dev object (073-g_lego_duplo; the frozen prior already solves
# 065-b_cups): res_scale x prog_explore x source seed = 2 x 2 x 2 = 8 cells (--array=0-7). Resumable (exit 3).
#SBATCH --job-name=ms-pidec
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-004-31-0,atl1-1-01-004-33-0,atl1-1-03-004-29-0,atl1-1-03-007-31-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%A_%a.out
set -uo pipefail
source /storage/home/hcoda1/8/lwang831/workspace/Horsea/scripts/ms_common.sh
RUN=${RUN:?}; I=${SLURM_ARRAY_TASK_ID:?}; OBJ=${OBJ:-073-g_lego_duplo}
SCALES=(0.1 0.3); HS=(2500 10000); SEEDS=(1 2)
RS=${SCALES[$((I % 2))]}; H=${HS[$(((I / 2) % 2))]}; SEED=${SEEDS[$((I / 4))]}
echo "cell: object=$OBJ source_seed=$SEED res_scale=$RS prog_explore=$H"
python adapt/pi_dec_run.py --split $SPLIT --object $OBJ --source_seed $SEED --adapt_seed 0 \
  --base_actor $MS_ROOT/runs/$RUN/pi_dec/pd_base_s$SEED.pt --res_scale $RS --prog_explore $H \
  --out $MS_ROOT/runs/$RUN/pi_dec/dev/${OBJ}__s${SEED}__a0__rs${RS}_H${H} --time_cap_h 3.7
rc=$?; echo "=== exit $rc at $(date) ==="; exit $rc
