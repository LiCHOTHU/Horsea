#!/usr/bin/env bash
# Submit N blocks of base training chained with --dependency=afterany.
#
# afterany, not afterok: the point is that the next block starts however the previous one ended --
# completed, timed out, or PREEMPTED. Relying on a job to resubmit itself does not survive
# preemption, because preemption kills the script before its tail can run (job 13688314, epoch
# 10/50, never came back).
#
# Each block resumes from multitask_model_latest.pth and exits in seconds if training is already
# finished, so over-provisioning blocks is cheap. Base training needs ~26 h at ~31 min/epoch and
# each block does 7.5 h, so 6 blocks gives comfortable headroom for preemption losses.
#
#     bash scripts/pace_chain_base.sh [n_blocks]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N=${1:-6}
prev=""
for i in $(seq 1 "$N"); do
    if [ -z "$prev" ]; then
        id=$(sbatch --parsable scripts/pace_train_base.sbatch)
    else
        id=$(sbatch --parsable --dependency=afterany:"$prev" scripts/pace_train_base.sbatch)
    fi
    echo "block $i: $id${prev:+  (afterany:$prev)}"
    prev=$id
done
echo "submitted $N chained blocks; the last completing block launches self-play and features+Phi"
