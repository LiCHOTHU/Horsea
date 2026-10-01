#!/bin/bash
# Hang-safe fixed-scene runner (same args as run_fixed.sh). Each attempt runs in its own process group with live output and a
# PROGRESS WATCHDOG: if the run prints nothing for ${EVAL_STALL:-600}s (normal runs print a step counter on every
# action) or exceeds ${EVAL_TIMEOUT:-5400}s, the whole group is killed and the attempt is retried with the SAME seeds
# (up to 3 attempts). Only infrastructure stalls/crashes are retried; a completed run (rc 0) is never re-run.
cd "$(dirname "$0")"
STALL=${EVAL_STALL:-600}
CAP=${EVAL_TIMEOUT:-5400}
for attempt in 1 2 3; do
  log=$(mktemp /tmp/eval_safe.XXXXXX)
  setsid bash run_fixed.sh "$@" > "$log" 2>&1 &
  pid=$!
  tail -n +1 -f --pid=$pid "$log" &
  tpid=$!
  start=$(date +%s); last_size=-1; last_change=$start; why=""
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    size=$(stat -c %s "$log" 2>/dev/null || echo 0); now=$(date +%s)
    if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$now; fi
    if [ $((now - last_change)) -ge $STALL ]; then why="stall: no output for ${STALL}s"; fi
    if [ $((now - start)) -ge $CAP ]; then why="timeout: ${CAP}s"; fi
    if [ -n "$why" ]; then
      kill -TERM -- -$pid 2>/dev/null; sleep 10; kill -KILL -- -$pid 2>/dev/null
      break
    fi
  done
  wait $pid; rc=$?
  wait $tpid 2>/dev/null
  rm -f "$log"
  if [ -z "$why" ] && [ $rc -eq 0 ]; then exit 0; fi
  echo "[eval_safe] attempt $attempt failed (rc=$rc ${why:+; $why}), retrying with the same seeds" >&2
done
exit 1
