#!/bin/bash
# BC long-term-memory chain.  Usage: scripts/ms_launch_bc.sh RUN_NAME [DEMO_JOB_FULL] [DEMO_JOB_CUBE]
#   demos (submitted here unless job ids are given) -> PickCube check (afterok: cube demos)
#   -> source x3 seeds, two resumable blocks (afterok: check + full demos) -> dev adaptation x18 -> gate report
set -euo pipefail
RUN=${1:?run name}; J_DEMO=${2:-}; J_DEMO_CUBE=${3:-}
cd /storage/home/hcoda1/8/lwang831/workspace/Horsea
R=/storage/project/r-agarg35-0/lwang831/maniskill/runs/$RUN; mkdir -p $R; cp -n /storage/project/r-agarg35-0/lwang831/maniskill/split.json $R/split.json || true
sub() { sbatch --parsable "$@" | cut -d';' -f1; }
if [ -z "$J_DEMO" ]; then
  J_DEMO=$(sub --export=ALL,RUN=$RUN,N_DEMOS=${N_DEMOS:-1000},EXEC_NOISE=${EXEC_NOISE:-0.05},MAX_TRIES=${MAX_TRIES:-6000} scripts/ms_bc_demos.sbatch)
fi
J_DEMO_CUBE=${J_DEMO_CUBE:-$J_DEMO}
J_CUBE=$(sub --dependency=afterok:$J_DEMO_CUBE --export=ALL,RUN=$RUN,ENV=PickCube-v1,TOTAL_STEPS=${CUBE_STEPS:-300000} scripts/ms_source_bc.sbatch)
J_SRC1=$(sub --dependency=afterok:$J_CUBE:$J_DEMO --array=0-2 --export=ALL,RUN=$RUN,ENV=ycb,TOTAL_STEPS=${TOTAL_STEPS:-3000000} scripts/ms_source_bc.sbatch)
J_SRC2=$(sub --dependency=afterany:$J_SRC1 --array=0-2 --export=ALL,RUN=$RUN,ENV=ycb,TOTAL_STEPS=${TOTAL_STEPS:-3000000} scripts/ms_source_bc.sbatch)
J_DEV1=$(sub --dependency=afterok:$J_SRC2 --array=0-17 --export=ALL,RUN=$RUN scripts/ms_adapt_dev.sbatch)
J_GATE=$(sub --dependency=afterany:$J_DEV1 --export=ALL,RUN=$RUN scripts/ms_gate.sbatch)
cat > $R/jobs.json <<JSON
{"run": "$RUN", "submitted": "$(date -Is)", "git": "$(git rev-parse --short HEAD)",
 "source_stage": "BC on DART motion-planning demos (exec noise ${EXEC_NOISE:-0.05}, 1000/object) with standardised, velocity-masked actor input; dense-reward SAC fine-tune from it if the BC policy is below the 0.8 gate",
 "demos_full": "$J_DEMO", "demos_cube": "$J_DEMO_CUBE",
 "pickcube_check": "$J_CUBE (afterok:$J_DEMO_CUBE)",
 "source_block1": "$J_SRC1 (array 0-2, afterok:$J_CUBE:$J_DEMO)", "source_block2": "$J_SRC2 (afterany:$J_SRC1)",
 "dev_block1": "$J_DEV1 (array 0-17, afterok:$J_SRC2)", "gate": "$J_GATE (afterany:$J_DEV1)"}
JSON
cat $R/jobs.json
