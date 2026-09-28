"""Two cycles where the SHORT memory learns RoboTTT-style: from the robot's own history, written into
fast weights during each episode (reset every episode, no expert data at test time).

Backbone + memory + write rule: the two-stage LIBERO-10 models of horsea.history2 (trained on demos).
Per cycle (task T on long memory theta_j):
  1. short memory ON (live own-history writes): success on T (n_new episodes) and on the old
     LIBERO-90 panel (memory ON);
  2. consolidation: the teacher (theta_j + live writes) runs n_teacher episodes on disjoint starts;
     all executed chunks are recorded with their contexts; theta is trained (memory OFF) to reproduce
     them, with old LIBERO-90 demo replay and replay of earlier cycles' recorded teacher rollouts
     (no LIBERO-10 expert data enters the cycles);
  3. memory OFF after reset: success on T, on the previous cycle's task, and on the old panel.

    python -m horsea.cycles_ttt --mode horsea --tasks 3 5
"""
import argparse
import json
import os
import types

import torch

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.finetune import train_decoder
from horsea.history2 import EXEC, OUT, build, install
from horsea.manifest import panel_20
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, TRAIN_90
from horsea.rollout import install_sampler, make_runner, run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["plain", "ttt", "fwrite", "res", "horsea", "energy"])
    ap.add_argument("--tasks", type=int, nargs=2, default=[3, 5], help="LIBERO-10 task of cycle 1 and cycle 2")
    ap.add_argument("--tag", default="_2stage")
    ap.add_argument("--n_new", type=int, default=20)
    ap.add_argument("--n_teacher", type=int, default=20)
    ap.add_argument("--panel_n", type=int, default=10)
    ap.add_argument("--n_old", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--replay10", action="store_true",
                    help="LIBERO-10 is the backbone's training set too: replay its demos and add its other tasks to the old panel")
    ap.add_argument("--teacher_nowrite", action="store_true",
                    help="attribution control: the consolidation teacher runs with memory writes disabled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(EXP, "cycles_ttt", args.mode)
    os.makedirs(out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, dev)
    f = torch.load(os.path.join(OUT + args.tag, args.mode, "final.pt"), map_location=dev, weights_only=False)
    policy.velocity_net.load_state_dict(f["vnet"])
    policy.eval().requires_grad_(False)
    if args.mode == "energy":
        from horsea import history2 as _h2
        from horsea.phi import Phi
        _h2.PHI["phi"] = Phi(device=dev)
    memory, writer = build(args.mode, dev)
    if memory is not None:
        memory.load_state_dict(f["memory"])
        memory.eval().requires_grad_(False)
    if writer is not None:
        writer.load_state_dict(f["writer"])
        writer.eval().requires_grad_(False)
    shape_meta = sd["config"]["task"]["shape_meta"]
    b90 = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    b10 = FeatureBank(os.path.join(FEAT_DIR, "libero_10.pt"), dev)
    old_idx = torch.cat([b90.task_frames(t, range(b90.n_demos(t))) for t in TRAIN_90])
    panel = panel_20()[: args.panel_n]
    log = {"mode": args.mode, "tasks": args.tasks, "cycles": []}

    def run(pol, suite, task, n, offset, mem_on, rec=None, nowrite=False):
        runner = make_runner(shape_meta, suite, n, args.par, offset, dev)
        for k in ("reset", "sample_actions"):  # copies keep methods bound to the ORIGINAL policy: drop them
            pol.__dict__.pop(k, None)
        if mem_on:
            install(pol, Flow(pol), args.mode, memory, writer, nowrite=nowrite)
            if rec is not None:  # also record every decision's context + executed chunk
                inner = pol.sample_actions

                def sample_actions(self, data, inner=inner):
                    import copy
                    with torch.no_grad():
                        e = Flow(pol).encode(copy.deepcopy(data)).float()
                    a = inner(data)
                    rec.append((e.half().cpu(), torch.as_tensor(a).half()))
                    return a
                pol.sample_actions = types.MethodType(sample_actions, pol)
        else:
            install_sampler(pol, Flow(pol))
            pol.temporal_agg, pol.action_horizon, pol.batch_size, pol.action_queue = False, EXEC, None, None
        return run_task(runner, pol, task)["rate"]

    old10 = [t for t in range(10) if t not in args.tasks][:4] if args.replay10 else []
    idx10 = torch.cat([b10.task_frames(t, range(b10.n_demos(t))) for t in range(10)]) if args.replay10 else None

    def panel_rate(pol, mem_on):
        r90 = [run(pol, "libero_90", t, args.n_old, 30, mem_on) for t in panel]
        r10 = [run(pol, "libero_10", t, args.n_old, 30, mem_on) for t in old10]
        return {"libero90": sum(r90) / len(r90), "libero10_other": (sum(r10) / len(r10)) if r10 else None}

    theta, replay10 = policy, []
    for c, T in enumerate(args.tasks, start=1):
        cyc = {"cycle": c, "task": T}
        cyc["new_mem_on"] = run(theta, "libero_10", T, args.n_new, 30, True)
        cyc["old_mem_on"] = panel_rate(theta, True)
        print(f"[{args.mode} c{c} task {T}] short memory ON: new {cyc['new_mem_on']:.2f} old {cyc['old_mem_on']}", flush=True)
        rec = []
        cyc["teacher_collect"] = run(theta, "libero_10", T, args.n_teacher, 10, True, rec, nowrite=args.teacher_nowrite)
        e = torch.cat([r[0] for r in rec]).to(dev)
        a = torch.cat([r[1] for r in rec]).float().to(dev)

        def nb(e=e, a=a):
            i = torch.randint(len(e), (128,), device=dev)
            return e[i], a[i]

        def rb():
            n10 = 64 if replay10 else 0
            if idx10 is not None:  # half LIBERO-90, half LIBERO-10 demos (both are the backbone's old data)
                ea, aa = b90.gather(old_idx[torch.randint(len(old_idx), ((128 - n10) // 2,), device=dev)])
                eb, ab = b10.gather(idx10[torch.randint(len(idx10), (128 - n10 - (128 - n10) // 2,), device=dev)])
                er, ar = torch.cat([ea, eb]), torch.cat([aa, ab])
            else:
                er, ar = b90.gather(old_idx[torch.randint(len(old_idx), (128 - n10,), device=dev)])
            if replay10:
                pe, pa = torch.cat([r[0] for r in replay10]), torch.cat([r[1] for r in replay10])
                j = torch.randint(len(pe), (n10,), device=dev)
                er, ar = torch.cat([er, pe[j]]), torch.cat([ar, pa[j]])
            return er, ar

        theta, _ = train_decoder(theta, nb, args.steps, replay_batch=rb, tag=f"[{args.mode} c{c} consolidate]")
        replay10.append((e, a))  # this cycle's teacher rollouts, replayed in later cycles
        cyc["new_after"] = run(theta, "libero_10", T, args.n_new, 30, False)
        if c == 2:
            cyc["prev_after"] = run(theta, "libero_10", args.tasks[0], args.n_new, 30, False)
        cyc["old_after"] = panel_rate(theta, False)
        print(f"[{args.mode} c{c} task {T}] after consolidation (memory OFF): new {cyc['new_after']:.2f} "
              f"prev {cyc.get('prev_after')} old {cyc['old_after']}", flush=True)
        torch.save({k: v.half().cpu() for k, v in theta.velocity_net.state_dict().items()}, os.path.join(out, f"theta{c}.pt"))
        log["cycles"].append(cyc)
        json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
