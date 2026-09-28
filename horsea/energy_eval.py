"""Closed-loop next-attempt success for the energy memory (horsea.energy), same harness and folds as the
R0 pilot (horsea.r0_eval): writer-dev tasks x hidden shifts, n_attempts attempts on validation starts,
memory re-derived from W0 after each attempt from the robot's OWN completed attempts (no labels, no
success input), actions from the path solver under the written energy.

    python -m horsea.energy_eval --ckpt experiments/energy/joint_s0/last.pt --cond write
"""
import argparse
import json
import multiprocessing
import os
import time

import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.energy import Model, adapt
from horsea.manifest import FOLDS, WRITER_DEV
from horsea.paths import BASE_CKPT, EXP
from horsea.phi import Phi
from horsea.r0_eval import merge
from horsea.rollout import make_runner
from horsea.selfplay import Recorder, run_batch
from horsea.writer import prep_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond", choices=["write", "nowrite", "mismatched", "shuffled", "last_only", "reorder"], required=True)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--shifts", nargs="+", default=["rot135", "rot-135", "rot90+grip_inv", "rot90", "rot180", "grip_inv"])
    ap.add_argument("--n_attempts", type=int, default=5)
    ap.add_argument("--start_offset", type=int, default=0,
                    help="first validation start used; 10 gives fresh starts 40.. for the frozen test")
    ap.add_argument("--max_step", type=float, default=None, help="override the solver's per-step bound at test time")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    tasks = args.tasks or WRITER_DEV
    os.makedirs(args.out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.requires_grad_(False)
    flow = Flow(policy)
    phi = Phi(device=dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a_ = ck["args"]
    from horsea.energy import BALANCED, WRITER, EnergyMemory
    WRITER["kind"] = a_.get("writer", "batch")
    EnergyMemory.nohist = a_.get("nohist", False)
    model = Model(a_["K"], a_["n_iter"], n_cand=a_.get("n_cand", 4), structured=a_.get("structured", False), max_step=a_.get("max_step"), solver=a_.get("solver", "grad")).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # architecture-aware: an old checkpoint (no gate) must be a batch-writer model; anything else missing is an error
    assert not unexpected and all(k.startswith("gate.") for k in missing), (missing, unexpected)
    assert not missing or ck["args"].get("writer", "batch") == "batch", "gated/seq checkpoint without gate weights"
    model.eval().requires_grad_(False)
    if args.max_step is not None:
        model.solver.max_step = args.max_step
    mode = a_["solve"]
    multiprocessing.set_start_method("spawn", force=True)
    B = len(args.shifts)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", max(2, B), max(2, B), 0, dev)
    rec = Recorder(policy, flow)
    starts = FOLDS["validation"][args.start_offset: args.start_offset + args.n_attempts]
    ctx = {"state": None}

    def custom(encm, z):
        if ctx["state"] is None and EnergyMemory.nohist:
            ctx["state"] = model.mem.init_state(encm.shape[0])
        if ctx["state"] is None:  # empty memory: E == 0, the solver returns the ordinary rollout
            return model.solver.solve(flow, None, encm.float(), z, "none", False)
        st = ctx["state"]
        with torch.enable_grad():
            efn = lambda a: model.mem.energy(phi, st, a, encm.float())
            return model.solver.solve(flow, efn, encm.float(), z, mode, False).detach()

    rec.custom = custom
    lat = []
    for task in tasks:
        p = os.path.join(args.out, f"t{task}.json")
        if os.path.exists(p):
            continue
        ctx["state"] = None
        hist = [[] for _ in range(B)]
        res = {"task": task, "shifts": args.shifts, "cond": args.cond, "solve": mode, "starts": starts, "attempts": []}
        for n, s in enumerate(starts):
            t0 = time.time()
            eps = run_batch(runner, rec, task, [s] * B, args.shifts)
            lat.append((time.time() - t0) / max(1, sum(len(e["calls"]) for e in eps) / B))
            res["attempts"].append({"success": [e["success"] for e in eps]})
            for b, e in enumerate(eps):
                if e["calls"]:
                    hist[b].append(prep_episode(e, dev))
            print(task, args.cond, mode, "attempt", n, [int(e["success"]) for e in eps], flush=True)
            if (args.cond == "nowrite" and not EnergyMemory.nohist) or n == len(starts) - 1 or any(not h for h in hist):
                continue
            H = [merge(h) for h in hist]
            if args.cond == "reorder":  # intact transitions, chronology destroyed
                perm = lambda n: torch.randperm(n)
                H = [{k: (v[p_] if torch.is_tensor(v) and v.shape[:1] == (h["n"],) else v) for k, v in h.items()}
                     for h in H for p_ in [perm(h["n"])]]
            if args.cond == "last_only":  # only the most recent completed interaction
                H = [{k: (v[-1:] if torch.is_tensor(v) else v) for k, v in h.items()} for h in H]
                for h in H:
                    h["n"] = 1
            if args.cond == "mismatched":
                H = [H[(b + 1) % B] for b in range(B)]
            with torch.enable_grad():
                ctx["state"] = adapt(model, phi, H, False, shuffle=args.cond == "shuffled")
        res["sec_per_decision_batch"] = round(sum(lat) / len(lat), 3)
        json.dump(res, open(p, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
