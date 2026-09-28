"""Horsea's lifecycle on a task stream: write to short memory -> when full, consolidate -> reset.

    (theta_j, W0) --experience--> (theta_j, W*) --consolidate + verify--> (theta_{j+1}, W0)

Tasks arrive one at a time (K demos each; demos are the experience for now). For each arrival:
  1. snapshot W, then write the task's K demos into the shared short memory W (generic
     instruction, so the demos are the only task signal; several tasks share one W);
  2. fullness check on held-out demo frames of every task in W: gain_i = relative FM-loss
     reduction W gives on task i. W is full if the new write pushed an earlier task's gain below
     tau x its gain right after it was written (interference), or (fixed mode) |W| reached C;
     in adaptive mode the offending write is rolled back, W is consolidated without it, and
     the new task is re-written into a fresh W;
  3. future learning: success of theta_j + W on the newly written task (teacher);
  4. on consolidation: teacher (theta_j + W) rollouts on every task in W -> endpoint FM
     distillation into theta's decoder (student sees the real instruction), replay of old
     demos + earlier consolidated tasks; verify: memory-off theta_{j+1} on the W tasks; then
     reset W; evaluate retention on all stream tasks so far + 5 old training tasks.
Horsea reads a FROZEN copy of theta_0's features (--frozen_features) so its key space does not
drift as theta is consolidated. Baseline --arm ft: sequential fine-tuning with replay.

    python -m horsea.lifecycle --arm fmw --capacity adaptive
"""
import argparse
import json
import os
import time

import torch

import horsea  # noqa: F401
from horsea.adapt_eval import load_memory, write_demos
from horsea.bank import FeatureBank
from horsea.base import DECODER_PREFIXES, Flow, load_policy
from horsea.finetune import demo_batcher, train_decoder
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, HELDOUT_90, RETENTION_90, TRAIN_90
from horsea.rollout import install_sampler, make_runner, run_task


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["fmw", "res", "ft"])
    ap.add_argument("--stream", nargs="+", default=[f"90:{t}" for t in HELDOUT_90] + [f"10:{t}" for t in range(10)],
                    help="tasks as suite:index, 90 = LIBERO-90, 10 = LIBERO-10")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--capacity", default="adaptive", help="'adaptive' or an integer C (tasks per short memory)")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n_eval", type=int, default=10)
    ap.add_argument("--n_collect", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--store", type=int, default=4000)
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--no_frozen_features", action="store_true")
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory_lifecycle"))
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse()
    dev = args.device
    torch.manual_seed(args.seed)
    tag = f"{args.arm}_{args.capacity}" + ("_nofrozen" if args.no_frozen_features else "")
    out = args.out or os.path.join(EXP, "lifecycle", tag)
    os.makedirs(out, exist_ok=True)

    base, sd = load_policy(args.ckpt, dev)
    shape_meta = sd["config"]["task"]["shape_meta"]
    bank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    banks = {("90", False): bank, ("90", True): FeatureBank(os.path.join(FEAT_DIR, "libero_90_generic.pt"), dev)}
    if any(t.startswith("10:") for t in args.stream):
        banks[("10", False)] = FeatureBank(os.path.join(FEAT_DIR, "libero_10.pt"), dev)
        banks[("10", True)] = FeatureBank(os.path.join(FEAT_DIR, "libero_10_generic.pt"), dev)
    temb = banks[("90", True)].task_emb[0]
    SUITE = {"90": "libero_90", "10": "libero_10"}

    def demo(task, d, generic):
        s, i = task.split(":")
        return banks[(s, generic)].demo(int(i), d)
    old_idx = torch.cat([bank.task_frames(t, range(bank.n_demos(t))) for t in TRAIN_90])
    memory = None
    if args.arm != "ft":
        memory = load_memory(args.arm, os.path.join(args.mem_dir, args.arm, "final.pt"), dev)
        if args.arm == "fmw" and not args.no_frozen_features:
            memory.__dict__["feature_flow"] = Flow(base)

    def runner(n, offset=0, suite="90"):
        return make_runner(shape_meta, SUITE[suite], n, args.par, offset, dev)

    def rate(policy, flow, task, n=args.n_eval, mem=None, state=None, generic=False, offset=0):
        s, i = task.split(":")
        pol = install_sampler(policy, flow, mem, state, task_emb=temb if generic else None)
        return run_task(runner(n, offset, s), pol, int(i))["rate"]

    # ---- state (resumable per task) -------------------------------------------------------------
    theta, buffers, events, i0 = base, [], [], 0
    W, in_W, gains0 = (memory.init_state(1, requires_grad=True) if memory else None), [], {}
    prog = os.path.join(out, "progress.pt")
    if os.path.exists(prog):
        import copy
        p = torch.load(prog, map_location="cpu", weights_only=False)
        theta = copy.deepcopy(base)
        theta.velocity_net.load_state_dict({**theta.velocity_net.state_dict(), **p["decoder"]})
        theta.eval().requires_grad_(False)
        buffers = [(e.to(dev), a.to(dev)) for e, a in p["buffers"]]
        events, i0, in_W, gains0 = p["events"], p["i"], p["in_W"], p["gains0"]
        if memory is not None:
            W = {k: v.to(dev).requires_grad_(True) for k, v in p["W"].items()}
        print(f"resumed at stream index {i0}", flush=True)

    def save(i):
        torch.save({"decoder": {k: v.detach().cpu() for k, v in theta.velocity_net.state_dict().items()
                                if k.startswith(DECODER_PREFIXES)},
                    "buffers": [(e.cpu(), a.cpu()) for e, a in buffers], "events": events, "i": i,
                    "in_W": in_W, "gains0": gains0,
                    "W": {k: v.detach().cpu() for k, v in W.items()} if W is not None else None}, prog + ".tmp")
        os.replace(prog + ".tmp", prog)
        with open(os.path.join(out, "events.json"), "w") as f:
            json.dump(events, f, indent=1)

    def val_gain(flow, state, task):
        """Relative FM-loss reduction from W on 2 held-out demos (never written), fixed noise."""
        e = torch.cat([demo(task, d, True)[0] for d in (args.K, args.K + 1)])
        a = torch.cat([demo(task, d, True)[1] for d in (args.K, args.K + 1)])
        with torch.no_grad():
            torch.manual_seed(123)
            noise = torch.randn(len(e), flow.chunk, flow.adim, device=e.device)
            if args.arm == "fmw":  # FM loss of the adapted field vs the bare policy
                torch.manual_seed(123)
                l0 = flow.fm_loss(e, a).item()
                torch.manual_seed(123)
                l1 = memory.outer_loss(flow, state, e, a).item()
            else:  # residual: action error of adapted vs bare samples, same noise
                l0 = ((flow.sample(e, noise) - a.clamp(-1, 1)) ** 2).mean().item()
                l1 = ((memory.sample(flow, state, e, noise) - a.clamp(-1, 1)) ** 2).mean().item()
        return (l0 - l1) / max(l0, 1e-8)

    def consolidate(tasks, cycle):
        nonlocal theta, buffers
        t0 = time.time()
        flow_T = Flow(theta)
        ev = {"event": "consolidate", "cycle": cycle, "tasks": list(tasks), "teacher": {}, "student": {}}
        if memory is not None:
            ctx_t, ctx_s = [], []
            for task in tasks:
                rec_g, rec_r = [], []
                pol = install_sampler(theta, flow_T, memory, W, rec_g, temb, rec_r)
                s, ti = task.split(":")
                ev["teacher"][task] = run_task(runner(args.n_collect, 25, s), pol, int(ti))["rate"]
                sup_g = torch.cat([demo(task, d, True)[0] for d in range(args.K)])
                sup_r = torch.cat([demo(task, d, False)[0] for d in range(args.K)])
                ctx_t.append(torch.cat([torch.cat(rec_g).float().to(dev), sup_g]))
                ctx_s.append(torch.cat([torch.cat(rec_r).float().to(dev), sup_r]))
            ctx_t, ctx_s = torch.cat(ctx_t), torch.cat(ctx_s)
            Wd = {k: v.detach() for k, v in W.items()}

            def new_batch():
                i = torch.randint(len(ctx_t), (args.bs,), device=dev)
                with torch.no_grad():
                    return ctx_s[i], memory.sample(flow_T, Wd, ctx_t[i])

            with torch.no_grad():
                i = torch.randperm(len(ctx_t), device=dev)[: args.store * len(tasks)]
                store = (ctx_s[i].half(), torch.cat([memory.sample(flow_T, Wd, c) for c in ctx_t[i].split(512)]).half())
        else:
            e = torch.cat([demo(t, d, False)[0] for t in tasks for d in range(args.K)])
            a = torch.cat([demo(t, d, False)[1] for t in tasks for d in range(args.K)])
            new_batch, store = demo_batcher(e, a, args.bs), (e.half(), a.half())

        def replay_batch():
            n_old = args.bs if not buffers else args.bs // 2
            e, a = bank.gather(old_idx[torch.randint(len(old_idx), (n_old,), device=dev)])
            if buffers:
                be, ba = torch.cat([b[0] for b in buffers]), torch.cat([b[1] for b in buffers])
                j = torch.randint(len(be), (args.bs - n_old,), device=dev)
                e, a = torch.cat([e, be[j].float()]), torch.cat([a, ba[j].float()])
            return e, a

        # same distillation budget per consolidated task as the fine-tune baseline
        theta, hist = train_decoder(theta, new_batch, args.steps * len(tasks), lr=args.lr,
                                    replay_batch=replay_batch, tag=f"[{tag} cycle {cycle}]")
        buffers.append(store)
        fl = Flow(theta)
        ev["student"] = {t: rate(theta, fl, t) for t in tasks}  # verify: memory-off, real instruction
        ev["distill_sec"] = round(time.time() - t0, 1)
        return ev

    def evaluate_all(upto, cycle):
        fl = Flow(theta)
        seen = args.stream[: upto + 1]
        ev = {"event": "eval", "cycle": cycle, "after_stream_index": upto,
              "stream_tasks": {t: rate(theta, fl, t) for t in seen},
              "old_tasks": {f"90:{t}": rate(theta, fl, f"90:{t}", n=10) for t in RETENTION_90[:5]}}
        last = [e for e in events if e["event"] in ("consolidate", "finetune")][-1]
        new = last["tasks"]
        prev = [t for t in seen if t not in new]
        m = lambda d, ks: sum(d[k] for k in ks) / max(1, len(ks))
        print(f"\n===== [{tag}] cycle {cycle}: consolidated {new} =====\n"
              f"  NEW tasks      : teacher(theta+W) {m(last['teacher'], new) if last['teacher'] else float('nan'):.2f} | "
              f"memory-off after consolidation {m(ev['stream_tasks'], new):.2f}\n"
              f"  PREVIOUS stream: {m(ev['stream_tasks'], prev) if prev else float('nan'):.2f} over {len(prev)} tasks\n"
              f"  OLD train tasks: {m(ev['old_tasks'], list(ev['old_tasks'])):.2f}\n"
              f"  per task: {ev['stream_tasks']}\n", flush=True)
        return ev

    cycle = sum(1 for e in events if e["event"] == "consolidate")
    fixed_C = None if args.capacity == "adaptive" else int(args.capacity)
    for i in range(i0, len(args.stream)):
        task = args.stream[i]
        flow_j = Flow(theta)
        if args.arm == "ft":  # sequential fine-tuning with replay, one task at a time
            ev = consolidate([task], cycle)
            ev["event"] = "finetune"
            events.append(ev)
            cycle += 1
            if (i + 1) % (fixed_C or 2) == 0 or i == len(args.stream) - 1:
                events.append(evaluate_all(i, cycle))
            save(i + 1)
            continue

        W_prev = {k: v.detach().clone().requires_grad_(True) for k, v in W.items()}
        W = write_demos_into(memory, flow_j, demo, task, args.K, dev, W)
        gains = {t: val_gain(flow_j, W, t) for t in in_W + [task]}
        full, why = False, ""
        if fixed_C is not None:
            full = len(in_W) + 1 >= fixed_C
            why = f"|W| reached C={fixed_C}"
        else:
            hurt = [t for t in in_W if gains0.get(t, 0) > 0 and gains[t] < args.tau * gains0[t]]
            if hurt:
                full, why = True, f"write of task {task} cut earlier gains {[(t, round(gains0[t], 3), round(gains[t], 3)) for t in hurt]}"
        teacher_new = rate(theta, flow_j, task, mem=memory, state=W, generic=True)
        events.append({"event": "write", "cycle": cycle, "task": task, "in_W_before": list(in_W),
                       "gains": {str(k): round(v, 4) for k, v in gains.items()}, "teacher_success": teacher_new,
                       "full": full, "why": why})
        print(f"[{tag}] write task {task} (W holds {in_W + [task]}): teacher {teacher_new:.2f}, "
              f"gains {({k: round(v, 3) for k, v in gains.items()})}{'  -> FULL: ' + why if full else ''}", flush=True)

        if full and fixed_C is None and in_W:  # roll back the interfering write, consolidate the rest
            W = W_prev
            events.append(consolidate(in_W, cycle))
            cycle += 1
            events.append(evaluate_all(i - 1, cycle))
            W, in_W, gains0 = memory.init_state(1, requires_grad=True), [], {}
            flow_j = Flow(theta)
            W = write_demos_into(memory, flow_j, demo, task, args.K, dev, W)
            g = val_gain(flow_j, W, task)
            t_new = rate(theta, flow_j, task, mem=memory, state=W, generic=True)
            events.append({"event": "write", "cycle": cycle, "task": task, "in_W_before": [], "gains": {str(task): round(g, 4)},
                           "teacher_success": t_new, "full": False, "why": "rewritten into fresh W"})
            in_W, gains0 = [task], {task: g}
        else:
            in_W.append(task)
            gains0[task] = gains[task]
            if full:
                events.append(consolidate(in_W, cycle))
                cycle += 1
                events.append(evaluate_all(i, cycle))
                W, in_W, gains0 = memory.init_state(1, requires_grad=True), [], {}
        save(i + 1)

    if memory is not None and in_W:  # end of stream: consolidate what is left
        events.append(consolidate(in_W, cycle))
        cycle += 1
        events.append(evaluate_all(len(args.stream) - 1, cycle))
        W, in_W, gains0 = memory.init_state(1, requires_grad=True), [], {}
        save(len(args.stream))
    with open(os.path.join(out, "final.json"), "w") as f:
        json.dump({"args": vars(args), "events": events}, f, indent=1)
    print("done", flush=True)


def write_demos_into(memory, flow, demo, task, K, dev, W):
    state = {k: v.detach().requires_grad_(True) for k, v in W.items()}
    for d in range(K):
        e, a = demo(task, d, True)
        state = memory.write(flow, state, e.to(dev), a.to(dev), create_graph=False)
    return {k: v.detach().requires_grad_(True) for k, v in state.items()}


if __name__ == "__main__":
    main()
