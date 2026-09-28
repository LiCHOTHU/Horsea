"""Initial-base checkpoint rows for protocol v2 (shared by every stream): theta_0, memory off, real
instruction, receding horizon, on the exact folds the lifecycle streams use.
  held-out tasks: validation fold (20) and generated test starts (50)
  old panel (20 scene-balanced training tasks): old_validation (5 each) and old_test (10 each)

    python -m horsea.base_v2   -> experiments/protocol_v2/base/rows.jsonl
"""
import argparse
import json
import os

import horsea  # noqa: F401
from horsea import starts as gen_starts
from horsea.base import Flow, load_policy
from horsea.manifest import FOLDS, panel_20
from horsea.paths import BASE_CKPT, EXP, HELDOUT_90
from horsea.rollout import install_sampler, make_runner, run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = os.path.join(EXP, "protocol_v2", "base")
    os.makedirs(out, exist_ok=True)
    rows_p = os.path.join(out, "rows.jsonl")
    done = set()
    if os.path.exists(rows_p):
        done = {(r["task"], r["start_fold"]) for r in map(json.loads, open(rows_p))}
    policy, sd = load_policy(BASE_CKPT, args.device)
    shape_meta = sd["config"]["task"]["shape_meta"]
    install_sampler(policy, Flow(policy))
    jobs = [(t, "test") for t in sorted(HELDOUT_90)] + [(t, "old_test") for t in panel_20()] + \
           [(t, "validation") for t in sorted(HELDOUT_90)] + [(t, "old_validation") for t in panel_20()]
    for task, fold in jobs:
        if (task, fold) in done:
            continue
        if fold == "test":
            st = gen_starts.load(task)[: args.n_test]
            runner = make_runner(shape_meta, "libero_90", len(st), args.par, 0, args.device, init_states=st,
                                 init_indices=list(range(len(st))))
        else:
            ids = FOLDS[fold]
            runner = make_runner(shape_meta, "libero_90", len(ids), args.par, 0, args.device, init_indices=ids)
        policy.temporal_agg, policy.action_horizon, policy.batch_size, policy.action_queue = False, 8, None, None
        r = run_task(runner, policy, task)
        row = {"checkpoint": "theta0", "task": task, "start_fold": fold, "memory_mode": "off", "instruction": "real",
               "reset_ids": r["init"], "success": r["success"], "length": r["length"], "rate": r["rate"]}
        with open(rows_p, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(task, fold, r["rate"], flush=True)


if __name__ == "__main__":
    main()
