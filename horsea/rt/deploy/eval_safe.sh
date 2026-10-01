#!/bin/bash
# Hang-safe RoboTwin eval: each attempt capped at ${EVAL_TIMEOUT:-3600}s; up to 3 attempts with the SAME seeds.
# A rare cuRobo planner warm-up freeze then costs at most one timeout instead of hours. Same args as eval.sh.
cd "$(dirname "$0")"
for attempt in 1 2 3; do
  timeout ${EVAL_TIMEOUT:-3600} bash eval.sh "$@" && exit 0
  rc=$?
  echo "[eval_safe] attempt $attempt failed (rc=$rc; 124 = timeout/hang), retrying with the same seeds" >&2
done
exit 1
