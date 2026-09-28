"""Two-cycle proof of concept of the short -> long memory lifecycle, for any memory model.

    (theta_j, W0) --explore: fill W--> (theta_j, W*) --consolidate until converted--> (theta_{j+1}) --reset W--> ...

Each cycle takes one new task (the base scores 0% on the stream tasks, so memory-off success
after consolidation can only come from consolidation):
  1. EXPLORE / FILL: write demos into the short memory W (under the generic instruction, so the
     demos are the only task signal) in chunks, until the gain on held-out demos saturates
     (W is "full": more experience no longer helps) or K_max demos are used;
  2. SHORT-MEMORY SUCCESS: closed-loop success of theta_j + W on the task;
  3. CONSOLIDATE until converted: teacher (theta_j + W) rollouts -> endpoint FM distillation into
     theta under the task's REAL instruction, with replay of old demos + earlier consolidated
     tasks, and an INTERFACE ANCHOR that keeps theta's behaviour under the generic instruction
     (velocity + all decoder-layer tokens) equal to theta_0's. The anchor is what keeps the
     short-memory channel "blank" so the next cycle can learn again (renewed plasticity).
     Gate: memory-off success must reach `gate` x teacher success; otherwise distill another
     round (up to max_rounds). Only then is W reset to W0;
  4. EVALUATE: memory-off success on all stream tasks so far + 5 old training tasks.
A reference run of the same memory on theta_0 for every cycle task measures renewed plasticity:
cycle-2 short-memory success on theta_1 vs on theta_0.

    python -m horsea.cycles --arm fmw
"""
import argparse
import copy
import json
import os
import time

import torch

import horsea  # noqa: F401
from horsea.adapt_eval import load_memory
from horsea.bank import FeatureBank
from horsea.base import DECODER_PREFIXES, Flow, load_policy
from horsea.finetune import demo_batcher, train_decoder
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, RETENTION_90, TRAIN_90
from horsea.rollout import install_sampler, make_runner, run_task

VAL_DEMOS = (45, 46, 47, 48)  # never written: the held-out check for "is W full?"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ttt2", "kv", "fmw", "res", "ft"])
    ap.add_argument("--cycles", nargs="+", default=["57", "66"],
                    help="one entry per cycle: comma-separated LIBERO-90 tasks the short memory takes in that cycle")
    ap.add_argument("--K_max", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5, help="demos written between fullness checks")
    ap.add_argument("--sat_tol", type=float, default=0.01, help="W is full when a chunk adds < this gain")
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--n_collect", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000, help="distillation steps per task per round")
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--gate", type=float, default=0.9)
    ap.add_argument("--forget_tol", type=float, default=0.10,
                    help="reset only if previously learned tasks stay within this of their pre-cycle mean")
    ap.add_argument("--n_prev", type=int, default=20, help="rollouts per other stream task at every step")
    ap.add_argument("--eval_start", action="store_true", help="also measure theta_0 itself (step 0 of the table)")
    ap.add_argument("--anchor", type=float, default=1.0, help="weight of the generic-interface anchor (0 = off)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory_generic"))
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_reference", action="store_true",
                    help="skip re-measuring each later task's short memory on theta_0 (use existing adaptation runs)")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse()
    dev = args.device
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(EXP, "cycles_full", args.arm + ("" if args.anchor > 0 else "_noanchor"))
    os.makedirs(out, exist_ok=True)
    groups = [[int(t) for t in g.split(",")] for g in args.cycles]
    tasks = [t for g in groups for t in g]

    base, sd = load_policy(args.ckpt, dev)
    base_flow = Flow(base)
    shape_meta = sd["config"]["task"]["shape_meta"]
    rbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    gbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90_generic.pt"), dev)
    temb = gbank.task_emb[0]
    old_idx = torch.cat([rbank.task_frames(t, range(rbank.n_demos(t))) for t in TRAIN_90])
    anchor_idx = old_idx[::4]  # generic-channel anchor frames (same row layout in both banks)
    memory = None
    if args.arm != "ft":
        memory = load_memory(args.arm, os.path.join(args.mem_dir, args.arm, "final.pt"), dev)
        if args.arm == "fmw":
            memory.__dict__["feature_flow"] = base_flow  # frozen key space

    def runner(n, offset=0):
        return make_runner(shape_meta, "libero_90", n, args.par, offset, dev)

    def success(policy, task, n=None, mem=None, W=None, generic=False, offset=0, recorders=(None, None)):
        pol = install_sampler(policy, Flow(policy), mem, W, recorders[0], temb if generic else None, recorders[1])
        return run_task(runner(n or args.n_eval, offset), pol, task)["rate"]

    def val_gain(flow, W, task):
        e = torch.cat([gbank.demo(task, d)[0] for d in VAL_DEMOS])
        a = torch.cat([gbank.demo(task, d)[1] for d in VAL_DEMOS])
        with torch.no_grad():
            if args.arm == "res":  # action error, adapted vs bare, same noise
                torch.manual_seed(7)
                n = torch.randn(len(e), flow.chunk, flow.adim, device=dev)
                l0 = ((flow.sample(e, n) - a.clamp(-1, 1)) ** 2).mean().item()
                l1 = ((memory.sample(flow, W, e, n) - a.clamp(-1, 1)) ** 2).mean().item()
            else:  # FM loss of the adapted field vs the bare policy, same noise
                torch.manual_seed(7)
                l0 = flow.fm_loss(e, a).item()
                torch.manual_seed(7)
                l1 = memory.outer_loss(flow, W, e, a).item()
        return (l0 - l1) / max(l0, 1e-8)

    def explore(theta, task):
        """Fill W with demos, one chunk at a time, while held-out gain still improves. W is full
        when a chunk adds < sat_tol gain; that chunk is ROLLED BACK (it did not help, and past
        capacity it can hurt). Returns the kept W, demos kept, and the gain curve."""
        flow = Flow(theta)
        W = memory.init_state(1, requires_grad=True)
        curve, used, kept, best = [], 0, 0, None
        while used < args.K_max:
            W_prev = {k: v.detach().clone() for k, v in W.items()}
            for d in range(used, min(used + args.chunk, args.K_max)):
                e, a = gbank.demo(task, d)
                W = memory.write(flow, W, e, a, create_graph=False)
            used = min(used + args.chunk, args.K_max)
            g = round(val_gain(flow, W, task), 4)
            curve.append((used, g))
            if best is not None and g - best < args.sat_tol:
                W = W_prev  # full: roll back the chunk that no longer helped
                break
            best, kept = g, used
        return {k: v.detach().requires_grad_(True) for k, v in W.items()}, kept, curve

    def anchor_loss(flow):
        """Keep theta's generic-instruction channel (velocity and every decoder layer's tokens)
        equal to theta_0's: the short memory learns through this channel."""
        i = anchor_idx[torch.randint(len(anchor_idx), (args.bs,), device=dev)]
        e, a = gbank.gather(i)
        x1 = a.clamp(-1, 1)
        t = flow.sample_t(len(x1), dev)
        psi, _ = flow.interp(torch.randn_like(x1), x1, t)
        hs, hs0 = [], []
        v = flow.decode(psi, t, e, layer_hook=lambda l, x: (hs.append(x), x)[1])
        with torch.no_grad():
            v0 = base_flow.decode(psi, t, e, layer_hook=lambda l, x: (hs0.append(x), x)[1])
        lv = ((v - v0) ** 2).mean()
        lh = sum(((h - h0) ** 2).mean() / (h0 ** 2).mean().clamp_min(1e-6) for h, h0 in zip(hs, hs0)) / len(hs)
        return args.anchor * (lv + lh)

    # ---- resumable state -----------------------------------------------------------------------
    theta, buffers, log, c0 = base, [], {"args": vars(args), "reference_theta0": {}, "cycles": []}, 0
    prog = os.path.join(out, "progress.pt")
    if os.path.exists(prog):
        p = torch.load(prog, map_location="cpu", weights_only=False)
        theta = copy.deepcopy(base)
        theta.velocity_net.load_state_dict({**theta.velocity_net.state_dict(), **p["decoder"]})
        theta.eval().requires_grad_(False)
        buffers = [(e.to(dev), a.to(dev)) for e, a in p["buffers"]]
        log, c0 = p["log"], p["c"]
        print(f"resumed at cycle {c0}", flush=True)

    def save(c):
        torch.save({"decoder": {k: v.detach().cpu() for k, v in theta.velocity_net.state_dict().items()
                                if k.startswith(DECODER_PREFIXES)},
                    "buffers": [(e.cpu(), a.cpu()) for e, a in buffers], "log": log, "c": c}, prog + ".tmp")
        os.replace(prog + ".tmp", prog)
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    # reference: the same short memory on the ORIGINAL policy, for tasks learned after cycle 1
    # (cycle-1 tasks are learned on theta_0 itself). This is the renewed-plasticity yardstick.
    if memory is not None and not log["reference_theta0"] and not args.no_reference:
        for task in [t for g in groups[1:] for t in g]:
            W, used, curve = explore(base, task)
            log["reference_theta0"][task] = {"short_memory_success": success(base, task, mem=memory, W=W, generic=True),
                                             "demos_used": used, "gain_curve": curve}
        print("reference on theta_0:", log["reference_theta0"], flush=True)
        save(c0)

    def old_tasks(policy, mem=None, W=None):
        return {t: success(policy, t, n=10, mem=mem, W=W) for t in RETENTION_90[:5]}

    def dump(obj, name):  # every intermediate model is kept, so any table cell can be re-measured later
        torch.save(obj, os.path.join(out, name))

    # step 0: the original policy on every table cell (real instruction; generic instruction for reference)
    if args.eval_start and "start" not in log:
        log["start"] = {"stream_tasks": {t: success(base, t) for t in tasks},
                        "stream_tasks_generic": {t: success(base, t, generic=True) for t in tasks},
                        "old_tasks": old_tasks(base)}
        print("start (theta_0):", log["start"], flush=True)
        save(c0)

    mean = lambda xs: sum(xs) / max(1, len(xs))
    for c in range(c0, len(groups)):
        group, t0 = groups[c], time.time()
        rec = {"cycle": c + 1, "tasks": group, "per_task": {}}
        per_task_batches, stores, targets = [], [], {}
        for task in group:
            pt = rec["per_task"][task] = {}
            if memory is not None:
                # 1-2. fill this task's short-memory slot until full; short-memory success on the CURRENT long memory
                W, used, curve = explore(theta, task)
                pt.update(demos_used=used, gain_curve=curve,
                          short_memory_success=success(theta, task, mem=memory, W=W, generic=True))
                targets[task] = pt["short_memory_success"]
                dump({k: v.detach().cpu() for k, v in W.items()}, f"W_cycle{c + 1}_task{task}.pt")
                # the whole system while this short memory is loaded (memory ON, real instructions):
                # every other stream task and the old training tasks
                pt["memory_on_other_tasks"] = {t: success(theta, t, n=args.n_prev, mem=memory, W=W)
                                               for t in tasks if t not in group}
                pt["memory_on_old_tasks"] = old_tasks(theta, memory, W)
                print(f"[{args.arm}] cycle {c + 1} task {task}: slot full after {used} demos {curve}; "
                      f"short-memory success {pt['short_memory_success']:.2f}", flush=True)
                # teacher rollouts on start states disjoint from evaluation -> consolidation contexts
                rg, rr = [], []
                pt["teacher_collect_success"] = success(theta, task, n=args.n_collect, mem=memory, W=W, generic=True,
                                                        offset=25, recorders=(rg, rr))
                sup_g = torch.cat([gbank.demo(task, d)[0] for d in range(used)])
                sup_r = torch.cat([rbank.demo(task, d)[0] for d in range(used)])
                ctx_t = torch.cat([torch.cat(rg).float().to(dev), sup_g])
                ctx_s = torch.cat([torch.cat(rr).float().to(dev), sup_r])
                flow_T, Wd = Flow(theta), {k: v.detach() for k, v in W.items()}

                def task_batch(n, ctx_t=ctx_t, ctx_s=ctx_s, flow_T=flow_T, Wd=Wd):
                    i = torch.randint(len(ctx_t), (n,), device=dev)
                    with torch.no_grad():
                        return ctx_s[i], memory.sample(flow_T, Wd, ctx_t[i])

                with torch.no_grad():
                    i = torch.randperm(len(ctx_t), device=dev)[:6000]
                    stores.append((ctx_s[i].half(),
                                   torch.cat([memory.sample(flow_T, Wd, x) for x in ctx_t[i].split(512)]).half()))
            else:  # fine-tuning baseline: the same demos straight into the long memory
                used = args.K_max
                e = torch.cat([rbank.demo(task, d)[0] for d in range(used)])
                a = torch.cat([rbank.demo(task, d)[1] for d in range(used)])
                task_batch = (lambda n, e=e, a=a: demo_batcher(e, a, n)())
                stores.append((e.half(), a.half()))
                pt["demos_used"] = used
            per_task_batches.append(task_batch)

        def new_batch():  # balanced over the cycle's tasks
            k = torch.randint(len(per_task_batches), (args.bs,))
            parts = [per_task_batches[j](int((k == j).sum())) for j in k.unique().tolist()]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

        boost = [1]  # grows when earlier tasks are being forgotten: more replay of their stored examples

        def replay_batch():
            n_old = args.bs if not buffers else args.bs // 2
            e, a = rbank.gather(old_idx[torch.randint(len(old_idx), (n_old,), device=dev)])
            if buffers:
                be, ba = torch.cat([b[0] for b in buffers]), torch.cat([b[1] for b in buffers])
                j = torch.randint(len(be), ((args.bs - n_old) * boost[0],), device=dev)
                e, a = torch.cat([e, be[j].float()]), torch.cat([a, ba[j].float()])
            return e, a

        # 3. consolidate until (a) the new task is converted: memory-off >= gate x short-memory success, and
        #    (b) nothing is forgotten: previously learned tasks within forget_tol of their pre-cycle level.
        #    Every round evaluates the new task AND every previously learned task (memory off, real instruction).
        prev_tasks = [t for g in groups[:c] for t in g]
        prev_before = log["cycles"][-1]["after"]["stream_tasks"] if log["cycles"] else {}
        before_mean = mean([prev_before[t] for t in prev_tasks]) if prev_tasks else None
        rounds = []
        extra = anchor_loss if (memory is not None and args.anchor > 0) else None
        target = mean(list(targets.values())) if targets else None
        for r in range(args.max_rounds):
            theta, hist = train_decoder(theta, new_batch, args.steps * len(group), lr=args.lr,
                                        replay_batch=replay_batch, tag=f"[{args.arm} cycle {c + 1} round {r + 1}]",
                                        extra_loss=extra)
            s_new = {t: success(theta, t) for t in group}
            s_prev = {t: success(theta, t, n=args.n_prev) for t in prev_tasks}
            s_future = {t: success(theta, t, n=args.n_prev) for t in tasks if t not in group and t not in prev_tasks}
            s_old = old_tasks(theta)
            dump({k: v.detach().cpu() for k, v in theta.velocity_net.state_dict().items() if k.startswith(DECODER_PREFIXES)},
                 f"theta_cycle{c + 1}_round{r + 1}.pt")
            ok_new = target is None or mean(list(s_new.values())) >= args.gate * target
            ok_prev = not prev_tasks or mean(list(s_prev.values())) >= before_mean - args.forget_tol
            rounds.append({"round": r + 1, "new": s_new, "previous": s_prev, "future": s_future, "old": s_old,
                           "converted": ok_new,
                           "no_forgetting": ok_prev, "replay_boost": boost[0], "last_loss": hist[-1]})
            print(f"[{args.arm}] cycle {c + 1} round {r + 1}: new {mean(list(s_new.values())):.2f} {s_new}"
                  + (f" (short memory {target:.2f})" if target is not None else "")
                  + (f" | previous {mean(list(s_prev.values())):.2f} (before {before_mean:.2f}) {s_prev}" if prev_tasks else "")
                  + (f" | future {s_future}" if s_future else "") + f" | old {mean(list(s_old.values())):.2f}",
                  flush=True)
            if ok_new and ok_prev:
                break
            if not ok_prev:
                boost[0] *= 2
        rec["consolidation_rounds"] = rounds
        rec["converted"], rec["no_forgetting"] = rounds[-1]["converted"], rounds[-1]["no_forgetting"]
        buffers.extend(stores)
        # 4. reset every short-memory slot (W := W0); the long memory's state is the last round's evaluation
        rec["after"] = {"stream_tasks": {**rounds[-1]["previous"], **rounds[-1]["new"]},
                        "future_tasks": rounds[-1]["future"], "old_tasks": rounds[-1]["old"]}
        rec["sec"] = round(time.time() - t0, 1)
        log["cycles"].append(rec)
        st = rec["after"]["stream_tasks"]
        prev = {t: v for t, v in st.items() if t not in group}
        print(f"\n===== [{args.arm}] CYCLE {c + 1}: tasks {group} =====\n"
              + (f"  short memory (theta_{c} + W): {target:.2f} {targets}\n" if targets else "")
              + (f"  [same memory on theta_0: {mean([log['reference_theta0'][t]['short_memory_success'] for t in group]):.2f}]\n"
                 if memory is not None and c > 0 and log["reference_theta0"] else "")
              + f"  new tasks, memory OFF after consolidation: {mean([st[t] for t in group]):.2f} "
                f"{ {t: st[t] for t in group} } (rounds: {len(rounds)}, converted: {rec['converted']}, "
                f"no forgetting: {rec['no_forgetting']})\n"
              + f"  previous tasks: {mean(list(prev.values())) if prev else float('nan'):.2f} {prev}\n"
              + f"  old training tasks: {mean(list(rec['after']['old_tasks'].values())):.2f}\n", flush=True)
        save(c + 1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
