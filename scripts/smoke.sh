#!/usr/bin/env bash
# Environment smoke test: verifies the Horsea stack runs on this machine.
#
# Checks, in order: GPU health (uncorrectable ECC), CPU invariants of the base policy and all four
# memory arms, the selfplay recorder regression (audit P1), closed-loop LIBERO rollouts with real
# envs, and the sequential-writer regression (audit P2) against a throwaway Phi.
#
# None of this needs the trained base checkpoint: every test builds a randomly initialised
# fm_policy_S from tests/fixtures/. Writes only to $HORSEA_EXP (default experiments/smoke).
#
#     bash scripts/smoke.sh            # full suite (needs a GPU)
#     bash scripts/smoke.sh cpu        # CPU-only subset
set -uo pipefail

DEV="${1:-cuda:0}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# fixed, not overridable: tests/make_fake_ckpt.py writes here, and nothing should land in the
# real experiments tree
export HORSEA_EXP="$ROOT/experiments/smoke"
export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
mkdir -p "$HORSEA_EXP"

PY=$(command -v python)
fail=0
step() { printf '\n======== %s\n' "$1"; }
report() { if [ "$1" -eq 0 ]; then echo "PASS  $2"; else echo "FAIL  $2 (exit $1)"; fail=1; fi; }

echo "python   : $PY"
echo "device   : $DEV"
echo "HORSEA_EXP: $HORSEA_EXP"

step "0. GPU health"
if [ "$DEV" != cpu ]; then
    # A node with uncorrectable DRAM ECC errors fails every cuda allocation; catch it here rather
    # than 20 minutes into a rollout.
    nvidia-smi --query-gpu=name,memory.total,ecc.errors.uncorrected.volatile.total --format=csv || true
    "$PY" -c "import torch; x=torch.randn(4096,4096,device='$DEV'); print('cuda alloc+matmul OK', float((x@x).sum()))"
    report $? "cuda allocation"
    [ "$fail" -eq 1 ] && { echo; echo "GPU is unusable on $(hostname) -- stopping."; exit 1; }
else
    echo "skipped (cpu mode)"
fi

step "1. CPU invariants (tests/test_cpu.py)"
"$PY" tests/test_cpu.py 2>&1 | grep -E "^(PASS|FAIL|INFO|---|ALL)" ; report ${PIPESTATUS[0]} "test_cpu.py"

step "2. Recorder regression, audit P1 (tests/test_recorder.py)"
"$PY" tests/test_recorder.py 2>&1 | tail -1 ; report ${PIPESTATUS[0]} "test_recorder.py"

if [ "$DEV" = cpu ]; then
    step "done (cpu subset)"
    [ "$fail" -eq 0 ] && echo "SMOKE OK (cpu subset)" || echo "SMOKE FAILED"
    exit $fail
fi

step "3. Closed-loop LIBERO rollout (tests/smoke_rollout.py)"
"$PY" tests/smoke_rollout.py "$DEV" 2>&1 | grep -E "^(fmw|ttt|ROLLOUT)" ; report ${PIPESTATUS[0]} "smoke_rollout.py"

step "4. Sequential-writer regression, audit P2 (tests/test_adapt_seq.py)"
# Needs a Phi checkpoint. Train a throwaway one on the fake feature bank so the code path is
# exercised without the real base policy; weights are meaningless, the plumbing is what is tested.
if [ ! -f "$HORSEA_EXP/phi/phi.pt" ]; then
    "$PY" tests/make_fake_ckpt.py >/dev/null 2>&1 \
      && "$PY" -m horsea.phi train --steps 200 --bs 256 --device "$DEV" >/dev/null 2>&1
    report $? "throwaway Phi for the adapt_seq test"
fi
"$PY" tests/test_adapt_seq.py 2>&1 | tail -2 ; report ${PIPESTATUS[0]} "test_adapt_seq.py"

step "5. Energy meta-training step (tests/test_energy_step.py)"
"$PY" tests/test_energy_step.py "$DEV" 2>&1 | grep -E "^(PASS|FAIL|ENERGY)" ; report ${PIPESTATUS[0]} "test_energy_step.py"

step "summary"
[ "$fail" -eq 0 ] && echo "SMOKE OK -- the Horsea stack runs on $(hostname)" || echo "SMOKE FAILED"
exit $fail
