"""Closed-loop R0 pilot (protocol v2, sec. 7, 11, 13): next-attempt success from the robot's own attempts.

For one (task, hidden shift) the robot makes n_attempts attempts on distinct starts. Memory persists
across attempts (reset only at the lifecycle boundary). After attempt n completes, the memory is
re-derived from W0 by the method's write rule on the history of completed attempts (no success or
reward input, no correction labels). All shifts of a task run as one parallel batch, each env with its
own memory (batched fast weights).

Conditions:
  horsea      learned experience writer (horsea.writer checkpoint)
  ttt2        native TTT history writer trained on the same data/objective
  nowrite     the same memory at W0 throughout (writes disabled)
  mismatched  horsea, but written with the history of a DIFFERENT shift of the same task
Stored-acquisition probe (writes disabled while acting) = the attempts themselves: writes happen only
between attempts in this pilot.

    python -m horsea.r0_eval --cond horsea --ckpt experiments/protocol_v2/writer/horsea_s0/last.pt
"""
import argparse
import json
import multiprocessing
import os
import random
import time

import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.manifest import FOLDS, WRITER_DEV
from horsea.paths import BASE_CKPT, EXP
from horsea.rollout import make_runner
from horsea.selfplay import Recorder, run_batch
from horsea.writer import Meta, prep_episode


def merge(eps):
    """Concatenate prepared attempt dicts into one history."""
    keys = ["encm", "label", "chunk", "probes", "cmd", "dprop", "mask"]
    out = {k: torch.cat([e[k] for e in eps]) for k in keys}
    out["n"] = sum(e["n"] for e in eps)
    return out


def main():
    try:  # resumable job: let the kernel pick it first under memory pressure, not the lifecycle runs
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except OSError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["horsea", "ttt2", "nowrite", "mismatched", "write"])
    ap.add_argument("--arm", default=None, choices=[None, "horsea", "ttt2", "fwrite_selfimit", "res_selfimit", "ttt2_info", "ttt2_info_dphi"],
                    help="method (default: from --cond); with --cond write/nowrite/mismatched")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--shifts", nargs="+", default=["rot135", "rot-135", "rot90+grip_inv", "rot90", "rot180", "grip_inv"])
    ap.add_argument("--n_attempts", type=int, default=5)
    ap.add_argument("--start_offset", type=int, default=0,
                    help="first validation start used; 10 gives fresh starts 40.. for the frozen test")
    ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--within", action="store_true",
                    help="RoboTTT-style: also write after every executed chunk inside the episode (own rollout only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = args.tasks or WRITER_DEV[:5]
    out = args.out or os.path.join(EXP, "protocol_v2", "R0_pilot", args.cond)
    os.makedirs(out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.requires_grad_(False)
    flow = Flow(policy)
    arm = args.arm or ("ttt2" if args.cond == "ttt2" else "horsea")
    ablate = (torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"].get("ablate")
              if args.ckpt else None)
    cargs = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"] if args.ckpt else {}
    meta = Meta(arm, dev, flow, flow, n_inner=cargs.get("n_inner", args.n_inner), ablate=ablate)
    from horsea.writer import set_lr_cap
    set_lr_cap(meta.memory, cargs.get("lr_cap", 3.0))
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        meta.memory.load_state_dict(ck["memory"])
        if meta.writer is not None and ck.get("writer"):
            meta.writer.load_state_dict(ck["writer"])
    meta.memory.eval()
    multiprocessing.set_start_method("spawn", force=True)
    B = len(args.shifts)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", max(2, B), max(2, B), 0, dev)
    starts = FOLDS["validation"][args.start_offset: args.start_offset + args.n_attempts]  # distinct from the adapt-fold starts of the offline data
    rec = Recorder(policy, flow, meta.memory, None)  # built once: it patches the policy's action pipeline
    for task in tasks:
        p = os.path.join(out, f"t{task}.json")
        if os.path.exists(p):
            continue
        t0 = time.time()
        state = meta.memory.init_state(B)
        hist = [[] for _ in range(B)]
        res = {"task": task, "shifts": args.shifts, "cond": args.cond, "starts": starts, "attempts": []}
        for n, s in enumerate(starts):
            rec.state = state
            on_step = None
            if args.within and args.cond != "nowrite":
                from horsea.selfplay import EXEC
                from horsea.writer import prep_episode as _prep

                def on_step(step, steps, succ_step, hist=hist):
                    if step % EXEC or step < EXEC:
                        return
                    Hs = []
                    for b in range(B):
                        end = min(step, len(steps[b]))
                        calls = [dict(c, n_exec=int(min(EXEC, end - c["step"]))) for c in rec.calls[b] if c["step"] + EXEC <= end]
                        if not calls:
                            return
                        props = torch.from_numpy(__import__("numpy").stack([x[0] for x in steps[b][:end]] + [steps[b][end - 1][0]]))
                        cur = _prep({"calls": calls, "proprio": props, "success": False}, dev)
                        Hs.append(merge(hist[b] + [cur]))
                    if args.cond == "mismatched":
                        Hs = [Hs[(b + 1) % B] for b in range(B)]
                    with torch.enable_grad():
                        rec.state = meta.adapt(Hs, create_graph=False)
            eps = run_batch(runner, rec, task, [s] * B, args.shifts, on_step=on_step)
            res["attempts"].append({"success": [e["success"] for e in eps], "length": [e["length"] for e in eps]})
            for b, e in enumerate(eps):
                if e["calls"]:
                    hist[b].append(prep_episode(e, dev))
            print(task, args.cond, "attempt", n, [int(e["success"]) for e in eps], flush=True)
            if args.cond == "nowrite" or n == len(starts) - 1:
                continue
            H = [merge(h) if h else None for h in hist]
            if args.cond == "mismatched":  # each env gets the history of the next shift in the list
                H = [H[(b + 1) % B] for b in range(B)]
            if any(h is None for h in H):
                continue
            with torch.enable_grad():
                state = meta.adapt(H, create_graph=False)
        res["sec"] = round(time.time() - t0)
        json.dump(res, open(p, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
