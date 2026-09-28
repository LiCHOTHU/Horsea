"""Learning a NEW task from the robot's own exploration, for every algorithm, then consolidation.

For one held-out task and one method, starting from theta_0 (real instruction, no demos, no reward):
  1. explore: N_max attempts on the adapt-fold starts; after each attempt the short memory is updated
     from the attempts so far (on-policy: attempt k acts with the memory written by attempts < k);
  2. after N in --report explorations: new-task success with the short memory ON (validation fold);
     after N_max also the old-task panel with the memory ON;
  3. consolidation (memory methods): frozen teacher theta_0 + W collects n_teacher episodes (teacher
     fold); FM distillation into theta with old-demo replay; memory reset; new task and old panel with
     memory OFF.
Update rules from exploration (no correction labels, no success input):
  horsea -- learned experience writer (horsea.writer checkpoint), re-derived from W0 on the history
  ttt2   -- native TTT history write (horsea.writer ttt2 checkpoint), same
  fwrite -- F-write's FM write on its own executed chunks (self-imitation, unfiltered)
  res    -- residual action regression onto its own executed chunks (self-imitation, unfiltered)
  ft     -- fine-tune theta on its own executed chunks (self-imitation); theta IS the long memory, so
            "after consolidation" = the same fine-tuned theta

    python -m horsea.explore_cycle --method horsea --task 57
"""
import argparse
import copy
import json
import multiprocessing
import os
import random

import torch

import horsea  # noqa: F401
from horsea.adapt_eval import load_memory
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.finetune import train_decoder
from horsea.manifest import FOLDS, panel_20
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, TRAIN_90
from horsea.r0_eval import merge
from horsea.rollout import install_sampler, make_runner, run_task
from horsea.selfplay import Recorder, run_batch
from horsea.writer import Meta, prep_episode, set_lr_cap

WDIR = os.path.join(EXP, "protocol_v2", "writer")


def main():
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("300")
    except OSError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["horsea", "ttt2", "fwrite", "res", "ft"])
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n_explore", type=int, default=5)
    ap.add_argument("--report", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_teacher", type=int, default=20)
    ap.add_argument("--panel_n", type=int, default=10)
    ap.add_argument("--n_old", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--writer", default=None, help="writer checkpoint (horsea / ttt2)")
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory_generic"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev, T = args.device, args.task
    random.seed(0)
    torch.manual_seed(0)
    out = args.out or os.path.join(EXP, "protocol_v2", "explore", args.method)
    os.makedirs(out, exist_ok=True)
    p_out = os.path.join(out, f"t{T}.json")
    if os.path.exists(p_out):
        return
    multiprocessing.set_start_method("spawn", force=True)
    base, sd = load_policy(BASE_CKPT, dev)
    base.requires_grad_(False)
    flow0 = Flow(base)
    shape_meta = sd["config"]["task"]["shape_meta"]
    rbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    old_idx = torch.cat([rbank.task_frames(t, range(rbank.n_demos(t))) for t in TRAIN_90])
    panel = panel_20()[: args.panel_n]

    meta, memory = None, None
    if args.method in ("horsea", "ttt2"):
        ck = torch.load(args.writer or os.path.join(WDIR, f"{args.method}_s0", "last.pt"), map_location=dev, weights_only=False)
        meta = Meta(args.method, dev, flow0, flow0, n_inner=ck["args"].get("n_inner", 2))
        set_lr_cap(meta.memory, ck["args"].get("lr_cap", 3.0))
        meta.memory.load_state_dict(ck["memory"])
        if meta.writer is not None:
            meta.writer.load_state_dict(ck["writer"])
        memory = meta.memory
    elif args.method in ("fwrite", "res"):
        memory = load_memory({"fwrite": "fmw", "res": "res"}[args.method],
                             os.path.join(args.mem_dir, {"fwrite": "fmw", "res": "res"}[args.method], "final.pt"), dev)
        if args.method == "fwrite":
            memory.__dict__["feature_flow"] = flow0

    def evaluate(policy, task, fold, n, mem=None, W=None):
        ids = FOLDS[fold][:n]
        runner = make_runner(shape_meta, "libero_90", len(ids), args.par, 0, dev, init_indices=ids)
        pol = install_sampler(policy, Flow(policy), mem, W)
        pol.temporal_agg, pol.action_horizon, pol.batch_size, pol.action_queue = False, 8, None, None
        return run_task(runner, pol, task)["rate"]

    def panel_rate(policy, mem=None, W=None):
        return sum(evaluate(policy, t, "old_validation", args.n_old, mem, W) for t in panel) / len(panel)

    def update(hist, W, theta):
        """Short-memory update from the exploration history (list of prepared attempts)."""
        H = merge(hist)
        if meta is not None:
            with torch.enable_grad():
                return meta.adapt([H], create_graph=False), theta
        if args.method in ("fwrite", "res"):
            Wn = W
            e, a = hist[-1]["encm"].float(), hist[-1]["chunk"]  # write the newest attempt's own chunks
            with torch.enable_grad():
                for i in range(0, len(e), 8):
                    Wn = memory.write(flow0, Wn, e[i:i + 8], a[i:i + 8], create_graph=False)
            return {k: v.detach().requires_grad_(True) for k, v in Wn.items()}, theta
        # ft: fine-tune theta_0 on all own executed chunks so far, with old-demo replay
        e, a = H["encm"].float(), H["chunk"]

        def nb():
            i = torch.randint(len(e), (128,), device=dev)
            return e[i], a[i]

        def rb():
            return rbank.gather(old_idx[torch.randint(len(old_idx), (128,), device=dev)])

        th, _ = train_decoder(base, nb, args.steps, replay_batch=rb, tag=f"[ft explore {len(hist)}]")
        return None, th

    res = {"method": args.method, "task": T, "explore_success": [], "new_mem_on": {}, "base_pair": None}
    W = memory.init_state(1, requires_grad=True) if memory is not None else None
    theta, hist, rec_cache = base, [], {}
    runner = make_runner(shape_meta, "libero_90", 2, 2, 0, dev)
    for n in range(1, args.n_explore + 1):
        key = id(theta)
        if key not in rec_cache:  # one recorder per policy object (it patches the action pipeline)
            rec_cache[key] = Recorder(theta, Flow(theta), memory, W)
        rec = rec_cache[key]
        rec.state = W
        ep = run_batch(runner, rec, T, [FOLDS["adapt"][n - 1]], ["none"])[0]
        res["explore_success"].append(ep["success"])
        print(f"[{args.method} t{T}] exploration {n}: success {ep['success']}", flush=True)
        if ep["calls"]:
            hist.append(prep_episode(ep, dev))
            W, theta = update(hist, W, theta)
        if n in args.report:
            res["new_mem_on"][n] = evaluate(theta, T, "validation", args.n_val, memory, W)
            print(f"[{args.method} t{T}] after {n} explorations: new task {res['new_mem_on'][n]:.2f}", flush=True)
    res["old_mem_on"] = panel_rate(theta, memory, W)
    print(f"[{args.method} t{T}] old tasks with short memory on: {res['old_mem_on']:.2f}", flush=True)
    if memory is None:  # fine-tune: theta already is the long memory
        res["new_after"], res["old_after"], res["consolidation"] = res["new_mem_on"][args.n_explore], res["old_mem_on"], "n/a (theta updated directly)"
    else:
        rr = []
        ids = FOLDS["teacher"][: args.n_teacher]
        tr = make_runner(shape_meta, "libero_90", len(ids), args.par, 0, dev, init_indices=ids)
        pol = install_sampler(base, flow0, memory, W, None, None, rr)
        pol.temporal_agg, pol.action_horizon, pol.batch_size, pol.action_queue = False, 8, None, None
        res["teacher_collect"] = run_task(tr, pol, T)["rate"]
        ctx = torch.cat([torch.cat(rr).float().to(dev)] + [h["encm"].float() for h in hist])
        Wd = {k: v.detach() for k, v in W.items()}

        def nb():
            i = torch.randint(len(ctx), (128,), device=dev)
            with torch.no_grad():
                return ctx[i], memory.sample(flow0, Wd, ctx[i])

        def rb():
            return rbank.gather(old_idx[torch.randint(len(old_idx), (128,), device=dev)])

        student, _ = train_decoder(base, nb, args.steps, replay_batch=rb, tag=f"[{args.method} consolidate]")
        res["new_after"] = evaluate(student, T, "validation", args.n_val)
        res["old_after"] = panel_rate(student)
        res["consolidation"] = f"frozen teacher, {args.n_teacher} teacher episodes, {args.steps} FM steps + old replay"
    print(f"[{args.method} t{T}] after consolidation (memory off): new {res['new_after']:.2f} old {res['old_after']:.2f}", flush=True)
    json.dump(res, open(p_out, "w"), indent=1)


if __name__ == "__main__":
    main()
