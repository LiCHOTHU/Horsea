#!/usr/bin/env bash
# Submit the fair-comparison array once ALL its prerequisites exist.
#
# pace_selfplay and pace_features_phi run in parallel and either may finish last, so both call this
# on success. `mkdir` is the lock: it is atomic on a shared filesystem, so exactly one caller wins
# and the array is never submitted twice.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXP=${HORSEA_EXP:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Horsea/experiments}
TLOG=$EXP/libero/libero_90_train80/base80/fm/training.log
SP=$EXP/protocol_v2/selfplay/data

grep -q "Training completed successfully" "$TLOG" 2>/dev/null || { echo "[fair] base incomplete; not submitting"; exit 0; }
[ -f "$EXP/phi/phi.pt" ]            || { echo "[fair] Phi not ready; not submitting"; exit 0; }
[ -n "$(ls $SP/*.pt 2>/dev/null)" ] || { echo "[fair] self-play data not ready; not submitting"; exit 0; }

mkdir -p "$EXP/fair"
if ! mkdir "$EXP/fair/.submitted" 2>/dev/null; then
    echo "[fair] already submitted by the other stage; nothing to do"
    exit 0
fi
echo "[fair] prerequisites complete -- submitting the 5-arm x 3-seed array"
sbatch --array=0-14 scripts/pace_fair_compare.sbatch
