#!/bin/bash
# Lightweight reporting job (afterany): completeness + development gate; also summarises source gates.
#SBATCH --job-name=ms-gate
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/maniskill/runs/logs/%x_%j.out
set -uo pipefail
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh; conda activate Horsea
RUN=${RUN:?}; R=/storage/project/r-agarg35-0/lwang831/maniskill/runs/$RUN
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
echo "--- pickcube check:"; cat $R/pickcube_check/gate_source.json 2>/dev/null || echo missing
for s in 0 1 2; do echo "--- source seed $s:"; cat $R/source_s$s/gate_source.json 2>/dev/null || echo missing; done
echo "--- infrastructure: job exit lines"; grep -H "=== exit" $R/../logs/ms-*_*.out 2>/dev/null | sort | uniq -c | sort -rn | head -40
echo "--- development gate:"; python adapt/analyze.py $R/dev $R/gate_dev.json
