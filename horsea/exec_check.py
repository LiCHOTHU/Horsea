"""Base policy under the two execution modes (protocol v2 needs clean executed-prefix masks):
  agg -- temporal aggregation (a 16-step chunk every env step, executed action = mean of up to 16)
  rh  -- receding horizon (generate 16, execute the first 8, regenerate)
Real instruction; held-out tasks on the validation fold, old-task panel on old_validation.

    python -m horsea.exec_check --mode rh
"""
import argparse
import json
import os

import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.manifest import FOLDS, panel_20
from horsea.paths import BASE_CKPT, EXP, HELDOUT_90
from horsea.rollout import install_sampler, make_runner, run_task


def set_mode(policy, mode):
    policy.temporal_agg = mode == "agg"
    policy.action_horizon = 8
    policy.batch_size = None
    policy.action_queue = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["agg", "rh"], required=True)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = os.path.join(EXP, "protocol_v2", "exec_check", args.mode)
    os.makedirs(out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, args.device)
    shape_meta = sd["config"]["task"]["shape_meta"]
    install_sampler(policy, Flow(policy))
    set_mode(policy, args.mode)
    jobs = [(t, FOLDS["validation"]) for t in sorted(HELDOUT_90)] + [(t, FOLDS["old_validation"]) for t in panel_20()]
    for task, fold in jobs:
        p = os.path.join(out, f"t{task}.json")
        if os.path.exists(p):
            continue
        runner = make_runner(shape_meta, "libero_90", len(fold), args.par, fold[0], args.device)
        set_mode(policy, args.mode)
        r = run_task(runner, policy, task)
        r.update(mode=args.mode, fold=fold)
        json.dump(r, open(p, "w"))
        print(args.mode, task, r["rate"], r["rollout_sec"], "s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
