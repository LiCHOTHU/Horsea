#!/usr/bin/env bash
# Keeps the orchestrator alive; it is idempotent (every job has a completion check).
cd "$(dirname "$0")/.."
# run with the intended env already active: conda activate Horsea
mkdir -p experiments
while true; do
  python scripts/pipeline.py >> experiments/pipeline.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "ALL DONE" experiments/pipeline.log; then echo "pipeline finished"; break; fi
  echo "$(date) orchestrator exited rc=$rc; restarting in 30s" >> experiments/pipeline.log
  sleep 30
done
