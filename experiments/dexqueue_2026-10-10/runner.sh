#!/bin/bash
# Sequential job queue (one simulator process at a time). jobs.txt: one "<name>|<command>" per line, run in file order;
# finished names are appended to done.txt; touch STOP to end after the current job.
Q=/home/licho/workspace/Horsea/experiments/dexqueue_2026-10-10
touch $Q/done.txt
while [ ! -f $Q/STOP ]; do
  next=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    name="${line%%|*}"
    grep -qx "$name" $Q/done.txt || { next="$line"; break; }
  done < $Q/jobs.txt
  if [ -z "$next" ]; then sleep 60; continue; fi
  name="${next%%|*}"; cmd="${next#*|}"
  echo "$(date '+%F %T') START $name" >> $Q/queue.log
  bash -c "$cmd" >> $Q/$name.log 2>&1
  echo "$(date '+%F %T') DONE  $name rc=$?" >> $Q/queue.log
  echo "$name" >> $Q/done.txt
done
echo "$(date '+%F %T') queue stopped" >> $Q/queue.log
