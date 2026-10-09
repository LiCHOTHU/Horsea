#!/bin/bash
# Stage 0: train the base fm_policy_S on the 80 LIBERO-90 training tasks.
#
# Runs on the free `embers` QOS, which caps wall time at 8 h and is preemptible by inferno.
# The run needs longer than that, so submit a CHAIN OF BLOCKS UP FRONT with scripts/pace_chain_base.sh
# rather than relying on a job resubmitting itself:
#   * self-resubmission only runs at the end of the script, so a preemption (which kills the script
#     outright) breaks the chain -- this is exactly how job 13688314 died at epoch 10/50 and never
#     came back;
#   * --dependency=afterany means the next block starts whether this one completed, timed out or was
#     preempted;
#   * every block resumes from multitask_model_latest.pth and exits immediately if training is
#     already finished, so surplus blocks cost seconds.
#SBATCH --job-name=horsea-base
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-01-004-33-0
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments/logs/%x_%j.out

set -u
echo "node: $(hostname)  start: $(date)  job: $SLURM_JOB_ID"
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate Horsea
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
SELF=scripts/pace_train_base.sbatch

export HORSEA_EXP=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments
TLOG=$HORSEA_EXP/libero/libero_90_train80/base80/fm/training.log

# a node with uncorrectable DRAM ECC errors fails every alloc -- fail in seconds, not hours
if grep -q "Training completed successfully" "$TLOG" 2>/dev/null; then
    echo "base training already complete -- nothing to do"
    exit 0
fi
python -c "import torch; torch.randn(4096,4096,device='cuda:0')" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }

# stop short of the 8 h wall so the chaining logic below still gets to run
timeout 7.5h python scripts/train_base.py
rc=$?
echo "train_base exit=$rc  end: $(date)"

if grep -q "Training completed successfully" "$TLOG" 2>/dev/null; then
    echo "BASE TRAINING COMPLETE"
    # Launch the two downstream stages from here rather than via --dependency=afterok: this job
    # exits 0 on every chained block too, so afterok would fire while the model is only partly
    # trained. Both are needed before horsea.energy train; they are independent of each other.
    sbatch scripts/pace_selfplay.sbatch      # stage 3: explore / self-rollout data
    sbatch scripts/pace_features_phi.sbatch  # stages 1+4: cached features, then Phi
    exit 0
fi
# 124 = hit our timeout (expected, mid-run). Anything else non-zero is a real error: stop and report.
if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then
    echo "FAILED with exit $rc -- not chaining, investigate"
    exit $rc
fi
echo "incomplete -- the next pre-submitted block (afterany) continues from the latest checkpoint"
