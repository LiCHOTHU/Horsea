"""Stage-0 audit of the common base (spec sec. 5, step 1): per-task success of Plain on the 80 base-training
LIBERO-90 tasks, receding horizon (generate 16, execute 8, no temporal aggregation), horizon 300, starts from the
'adapt' fold (0-9). Used only to choose development tasks where Plain has some successes but room to improve.

    python -m horsea.rdm.audit --tasks 0 1 3 ... --out experiments/rdm/audit
"""
import argparse
import json
import multiprocessing
import os

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.paths import BASE_CKPT
from horsea.rollout import install_sampler, make_runner, run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="experiments/rdm/audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, args.device)
    install_sampler(policy, Flow(policy))
    policy.temporal_agg, policy.action_horizon = False, 8
    multiprocessing.set_start_method("spawn", force=True)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", args.n, args.par, 0, args.device, horizon=300)
    for t in args.tasks:
        p = os.path.join(args.out, f"t{t}.json")
        if os.path.exists(p):
            continue
        policy.reset()
        res = run_task(runner, policy, t)
        json.dump(res, open(p, "w"))
        print(t, res["rate"], flush=True)


if __name__ == "__main__":
    main()
