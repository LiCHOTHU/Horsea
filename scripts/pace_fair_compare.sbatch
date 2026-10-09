#!/bin/bash
# Fair Horsea-vs-TTT comparison: train every arm of the factorial, 3 seeds each.
#
#   idx  arm              information        objective  mechanism   isolates
#   ---  ---------------  -----------------  ---------  ----------  ----------------------------
#    A1  ttt2             c, a               FM         TTT         published baseline
#    A2  ttt_info         c, a, dc           FM         TTT         A2-A1 = value of information
#    A3  ttt_info_dphi    c, a, dc           d_Phi      TTT         A3-A2 = value of objective
#    A4  horsea           c, a, dc           d_Phi      energy      A4-A3 = MECHANISM  <-- the claim
#    A5  horsea-mse       c, a, dc           MSE        energy      objective, from Horsea's side
#
# Held equal (plan section 3), and why each is safe to assert here:
#   * same frozen base checkpoint; no arm fine-tunes the decoder (horsea.writer never touches it,
#     horsea.energy calls requires_grad_(False)) -- fixes plan item 6.1;
#   * --lam_old 0 everywhere, so no arm gets old-task replay -- fixes item 6.2;
#   * --structured is OFF for the energy arms, so no rotation-shaped write candidates -- item 6.3;
#   * same data, same seeds, same steps/optimizer defaults, same targets (oracle corrections, used
#     only in the outer loss).
# Capacity is matched where it matters: A3 and A4 have ~254k slow parameters each. A1 is smaller by
# construction, so A2-A1 buys information AND capacity -- a stated limit, not a hidden one.
#
# Submit as an array:  sbatch --array=0-14 scripts/pace_fair_compare.sbatch
#SBATCH --job-name=horsea-fair
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-01-004-33-0
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments/logs/%x_%A_%a.out

set -u
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea

export HORSEA_EXP=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments
OUT=$HORSEA_EXP/fair
TLOG=$HORSEA_EXP/libero/libero_90_train80/base80/fm/training.log
SP=$HORSEA_EXP/protocol_v2/selfplay/data

ARMS=(ttt2 ttt_info ttt_info_dphi horsea horsea_mse)
i=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$((i / 3))]}
SEED=$((i % 3))
echo "node: $(hostname)  arm=$ARM seed=$SEED  start: $(date)"

python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }

# --- prerequisites: refuse to produce a number from an incomplete pipeline -------------------
grep -q "Training completed successfully" "$TLOG" 2>/dev/null || { echo "base training incomplete"; exit 1; }
[ -f "$HORSEA_EXP/phi/phi.pt" ] || { echo "Phi missing -- run pace_features_phi.sbatch"; exit 1; }
[ -n "$(ls $SP/*.pt 2>/dev/null)" ] || { echo "self-play data missing -- run pace_selfplay.sbatch"; exit 1; }

TRAIN_SHIFTS="none rot20 rot-20 rot40 rot-40 rot60 rot-60"
DEV_SHIFTS="none rot30 rot-30 rot50 rot-50"

case "$ARM" in
  ttt2|ttt_info|ttt_info_dphi)
    # TTT mechanism. --lam_old 0 matches the energy arms, which have no replay term.
    python -m horsea.writer train --arm "$ARM" --seed "$SEED" --lam_old 0 \
        --train_shifts $TRAIN_SHIFTS --dev_shifts $DEV_SHIFTS \
        --out "$OUT/${ARM}_s${SEED}"
    ;;
  horsea|horsea_mse)
    # Energy mechanism. NOTE: --structured is deliberately omitted -- the structured candidate set
    # is built from +-20/40/60 degree rotations, i.e. knowledge of the testbed's answer that TTT is
    # not given (plan item 6.3). Generic candidates keep the write inductive bias equal.
    OBJ=dphi; [ "$ARM" = horsea_mse ] && OBJ=mse
    python -m horsea.energy train --solve final --n_iter 3 --max_step 0.05 --objective "$OBJ" \
        --seed "$SEED" --train_shifts $TRAIN_SHIFTS --dev_shifts $DEV_SHIFTS \
        --out "$OUT/${ARM}_s${SEED}"
    ;;
esac
rc=$?
echo "EXIT=$rc  arm=$ARM seed=$SEED  end: $(date)"
exit $rc
