#!/bin/bash
# Submit the first batch with dependencies and return.  Usage: scripts/ms_launch.sh RUN_NAME [TOTAL_STEPS]
#   pickcube check -> (afterok) source training x3 seeds, 2 chained blocks -> (afterok) dev adaptation x18,
#   2 chained blocks -> (afterany) gate/report job
set -euo pipefail
RUN=${1:?run name}; TOTAL=${2:-3000000}
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
R=/storage/project/r-agarg35-0/lwang831/maniskill/runs/$RUN; mkdir -p $R
cp /storage/project/r-agarg35-0/lwang831/maniskill/split.json $R/split.json
sub() { sbatch --parsable "$@" | cut -d';' -f1; }
J_CUBE=$(RUN=$RUN ENV=PickCube-v1 TOTAL_STEPS=500000 sub --export=ALL,RUN=$RUN,ENV=PickCube-v1,TOTAL_STEPS=500000 scripts/ms_source.sbatch)
J_SRC1=$(sub --dependency=afterok:$J_CUBE --array=0-2 --export=ALL,RUN=$RUN,ENV=ycb,TOTAL_STEPS=$TOTAL scripts/ms_source.sbatch)
J_SRC2=$(sub --dependency=afterany:$J_SRC1 --array=0-2 --export=ALL,RUN=$RUN,ENV=ycb,TOTAL_STEPS=$TOTAL scripts/ms_source.sbatch)
J_DEV1=$(sub --dependency=afterok:$J_SRC2 --array=0-17 --export=ALL,RUN=$RUN scripts/ms_adapt_dev.sbatch)
J_DEV2=$(sub --dependency=afterany:$J_DEV1 --array=0-17 --export=ALL,RUN=$RUN scripts/ms_adapt_dev.sbatch)
J_GATE=$(sub --dependency=afterany:$J_DEV2 --export=ALL,RUN=$RUN scripts/ms_gate.sbatch)
cat > $R/jobs.json <<JSON
{"run": "$RUN", "submitted": "$(date -Is)", "git": "$(git rev-parse --short HEAD)", "total_steps": $TOTAL,
 "pickcube_check": "$J_CUBE",
 "source_block1": "$J_SRC1 (array 0-2, afterok:$J_CUBE)", "source_block2": "$J_SRC2 (afterany:$J_SRC1)",
 "dev_block1": "$J_DEV1 (array 0-17, afterok:$J_SRC2)", "dev_block2": "$J_DEV2 (afterany:$J_DEV1)",
 "gate": "$J_GATE (afterany:$J_DEV2)"}
JSON
cat $R/jobs.json
