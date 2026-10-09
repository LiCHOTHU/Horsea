#!/bin/bash
#SBATCH --job-name=exploration-smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/exploration/marker_smoke_%j.out
set -euo pipefail
RUN_ROOT=$1
export EXPLORATION_SOURCE_ROOT="$RUN_ROOT/snapshot"
RUNNER="$EXPLORATION_SOURCE_ROOT/scripts/exploration_gpu.sh"
bash "$RUNNER" -m unittest discover -s tests/exploration
if [ ! -f "$RUN_ROOT/reference.json" ]; then
    bash "$RUNNER" -m exploration.reference --out "$RUN_ROOT/reference"
    bash "$RUNNER" -m exploration.perception_audit --root "$RUN_ROOT/reference"
    cp "$RUN_ROOT/reference/reference_retracked.json" "$RUN_ROOT/reference.json"
fi
bash "$RUNNER" -m exploration.smoke --scene 220000 --grasp 1 --mode 0 \
    --reference "$RUN_ROOT/reference.json" --out "$RUN_ROOT/smoke"
bash "$RUNNER" -m exploration.development_audit --root "$RUN_ROOT/smoke"
bash "$RUNNER" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); a=json.loads((p/"development_audit.json").read_text()); o=json.loads((p/"observations.jsonl").read_text()); t=json.loads((p/"evaluation_only.jsonl").read_text()); assert a["local_feedback_check_passed"] and t["simulator_success"] and o["event"]["tail"]=="goal", "Robot smoke/feedback check failed; development sweep stays blocked"; print("ROBOT_SMOKE_PASS")' "$RUN_ROOT/smoke"
