"""Retry-adaptation evaluation (spec sec. 7): per task, sequences on held-out validation starts; five attempts from
the same start with H kept across attempts; then one probe attempt from a fresh held-out start that reads (never
writes) H. Primary result: the trained stochastic policy (same exploration distribution as training).

    python -m horsea.rdm.evaluate --variant reread --ckpt experiments/rdm/runs/reread_s0/last.pt --tasks 23 32 \
        --n_seq 4 --out experiments/rdm/eval/reread_s0
"""
import argparse
import json
import multiprocessing
import os

import numpy as np
import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.paths import BASE_CKPT
from horsea.rdm.model import VARIANTS, make_model
from horsea.rdm.rollout import Controller, install, run_metaepisodes
from horsea.rollout import make_runner

SEQ_STARTS = list(range(30, 40))    # validation fold: attempt starts
PROBE_STARTS = list(range(40, 50))  # validation fold: fresh probe starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--n_seq", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=None, help="default: the training sigma")
    ap.add_argument("--deterministic", action="store_true", help="separately labelled no-noise ablation")
    ap.add_argument("--nowrite", action="store_true", help="intervention: history never updated")
    ap.add_argument("--pool", default="val", choices=["val", "train"],
                    help="val: held-out validation starts (reporting); train: training starts (settings selection only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()
    flow = Flow(policy)
    model = make_model(flow, args.variant).to(dev)
    sigma = args.sigma
    if args.ckpt:
        s = torch.load(args.ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        sigma = sigma if sigma is not None else s["args"]["sigma"]
    sigma = 0.1 if sigma is None else sigma
    model.eval().requires_grad_(False)
    multiprocessing.set_start_method("spawn", force=True)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", 2, 2, 0, dev, horizon=300)
    ctrl = Controller(model, flow, sigma, dev, deterministic=args.deterministic)
    install(policy, flow, ctrl)
    if args.nowrite:
        orig = ctrl.start
        ctrl.start = lambda B, bank=None: (orig(B, bank), setattr(ctrl, "write", False))
    for task in args.tasks:
        p = os.path.join(args.out, f"t{task}.json")
        if os.path.exists(p):
            continue
        if args.pool == "val":
            ids, pids = SEQ_STARTS[:args.n_seq], PROBE_STARTS[:args.n_seq]
        else:  # selection on training starts 0-29: attempts from 0.., probes from 15..
            ids, pids = list(range(args.n_seq)), list(range(15, 15 + args.n_seq))
        _, _, succ, probe = run_metaepisodes(runner, policy, flow, ctrl, task, ids, probe_ids=pids,
                                             log=lambda m: print(m, flush=True))
        res = {"task": task, "starts": ids, "probe_starts": pids, "success": succ.astype(int).tolist(),
               "probe": probe.astype(int).tolist(), "sigma": sigma, "deterministic": args.deterministic,
               "nowrite": args.nowrite, "S": succ.mean(0).round(3).tolist(), "S2_5": float(succ[:, 1:].mean()),
               "probe_rate": float(probe.mean())}
        json.dump(res, open(p, "w"))
        print(json.dumps({k: res[k] for k in ("task", "S", "S2_5", "probe_rate")}), flush=True)


if __name__ == "__main__":
    main()
