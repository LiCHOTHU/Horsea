#!/usr/bin/env bash
# Exit (and so notify) when any queue job fails/blocks, or when one of the named milestone jobs is done.
cd "$(dirname "$0")/.."
seen="$(python3 -c "import json;s=json.load(open('experiments/queue/state.json'));print(' '.join(sorted(k for k,v in s.items() if v['status'] in ('failed','blocked','lost','done'))))")"
while true; do
  sleep 60
  now="$(python3 -c "import json;s=json.load(open('experiments/queue/state.json'));print(' '.join(sorted(k for k,v in s.items() if v['status'] in ('failed','blocked','lost','done'))))")"
  new="$(comm -13 <(tr ' ' '\n' <<<"$seen" | sort) <(tr ' ' '\n' <<<"$now" | sort) | tr '\n' ' ')"
  bad="$(python3 -c "import json;s=json.load(open('experiments/queue/state.json'));print(' '.join(k for k in '$new'.split() if s[k]['status']!='done'))")"
  milestone="$(tr ' ' '\n' <<<"$new" | grep -E "^(sp_verify|sp_collect|W_horsea_s0|W_ttt2_s0|R0_)" | tr '\n' ' ')"
  if [ -n "$bad" ] || [ -n "$milestone" ]; then
    date +%H:%M; echo "NEW: $new"; echo "PROBLEM: $bad"; echo "MILESTONE: $milestone"; exit 0
  fi
  seen="$now"
done
