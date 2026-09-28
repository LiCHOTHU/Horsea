"""Protocol v2, expert arm (E): one ordered task pair, two memory cycles, fixed support budget.

    theta_0 --[task A: adapt 5 demos, frozen teacher, consolidate, gate, reset]--> theta_1
            --[task B: same]--> theta_2

Per cycle (task T, long memory theta_j):
  1. base: theta_j on T (validation fold, memory off, real instruction).
  2. adaptation with the fixed support budget (demos 0..4 in order); stored-acquisition curve after
     1, 2, 5 demos (clone, writes disabled). Fine-tune: theta_j fine-tuned on the first k demos.
  3. frozen teacher = theta_j + W_5 (immutable). Active-memory check: earlier task + old panel with W on.
  4. useful acquisition: teacher - base >= gain_min, else the stream stops (lifecycle incomplete).
  5. teacher collection: n_teacher episodes on the teacher fold (all kept, failures included).
  6. consolidation rounds (<= max_rounds, 3000 updates each): fresh-noise FM distillation of teacher
     samples + fixed-proportion replay (prior-cycle teacher buffers, old training demos).
     Gate (validation only): student gain >= retain x teacher gain; earlier task drop <= prev_tol;
     old-panel macro drop <= old_tol (vs the preceding accepted checkpoint). Failure after the last
     round: retain the teacher and last accepted theta, mark incomplete, stop (optionally continue
     as a labelled forced reset diagnostic).
  7. accepted student after reset: test starts for T and the earlier task, old panel test fold.
Fine-tune = sequential fine-tuning with replay (no teacher / reset); same evaluations.

Every evaluation is one raw row in rows.jsonl (schema of the plan, sec. 12).

    python -m horsea.lifecycle_v2 --method fwrite --pair 57 66 --instr real --mem_dir experiments/memory
"""
import argparse
import copy
import json
import os
import time

import torch

import horsea  # noqa: F401
from horsea import starts as gen_starts
from horsea.adapt_eval import load_memory
from horsea.bank import FeatureBank
from horsea.base import DECODER_PREFIXES, Flow, load_policy
from horsea.finetune import demo_batcher, train_decoder
from horsea.manifest import FOLDS, panel_20
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, TRAIN_90
from horsea.rollout import install_sampler, make_runner, run_task

ARM = {"fwrite": "fmw", "res": "res", "ttt2": "ttt2", "kv": "kv", "ft": None, "horsea": "horsea"}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(ARM))
    ap.add_argument("--pair", type=int, nargs=2, required=True)
    ap.add_argument("--instr", default="real", choices=["real", "generic"],
                    help="instruction for memory writes, memory-on evaluation and teacher runs (students/memory-off: real)")
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory"))
    ap.add_argument("--variant", default="")
    ap.add_argument("--writer_ckpt", default=None, help="horsea: meta-trained writer + reader (horsea.writer)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--n_old_val", type=int, default=5)
    ap.add_argument("--n_old_test", type=int, default=10)
    ap.add_argument("--n_teacher", type=int, default=20)
    ap.add_argument("--panel_n", type=int, default=20, help="old-task panel size (scene-balanced, preregistered order)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--gain_min", type=float, default=0.10)
    ap.add_argument("--retain", type=float, default=0.9)
    ap.add_argument("--prev_tol", type=float, default=0.10)
    ap.add_argument("--old_tol", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=None, help="generic-interface anchor weight (default: 1 if generic else 0)")
    ap.add_argument("--fast", action="store_true", help="skip test-start evaluation and the theta_0 reference curve")
    ap.add_argument("--forced_reset_diag", action="store_true", help="continue after a failed gate, labelled forced reset")
    ap.add_argument("--exec_mode", default="rh", choices=["rh", "agg"])
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    args = parse()
    dev = args.device
    torch.manual_seed(args.seed)
    if args.anchor is None:
        args.anchor = 1.0 if args.instr == "generic" else 0.0
    A, B = args.pair
    variant = args.variant or f"{args.instr}"
    out = args.out or os.path.join(EXP, "protocol_v2", "E", f"{args.method}_{variant}", f"p{A}_{B}")
    os.makedirs(out, exist_ok=True)
    rows_path, summ_path = os.path.join(out, "rows.jsonl"), os.path.join(out, "summary.json")
    if os.path.exists(summ_path) and json.load(open(summ_path)).get("finished"):
        print("already finished", out)
        return
    open(rows_path, "w").close()  # a stream is re-run from scratch if it did not finish

    base, sd = load_policy(BASE_CKPT, dev)
    base_flow = Flow(base)
    shape_meta = sd["config"]["task"]["shape_meta"]
    rbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    gbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90_generic.pt"), dev)
    wbank = gbank if args.instr == "generic" else rbank  # memory writes / teacher inputs
    temb = gbank.task_emb[0]
    old_idx = torch.cat([rbank.task_frames(t, range(rbank.n_demos(t))) for t in TRAIN_90])
    anchor_idx = old_idx[::4]
    arm = ARM[args.method]
    memory, meta = None, None
    if arm == "horsea":
        from horsea.writer import Meta
        meta = Meta("horsea", dev, base_flow, base_flow)
        ck = torch.load(args.writer_ckpt, map_location=dev, weights_only=False)
        meta.memory.load_state_dict(ck["memory"])
        meta.writer.load_state_dict(ck["writer"])
        meta.memory.eval().requires_grad_(False)
        meta.writer.eval().requires_grad_(False)
        memory = meta.memory
        memory.__dict__["feature_flow"] = base_flow  # frozen theta_0 features also after consolidation
    elif arm:
        memory = load_memory(arm, os.path.join(args.mem_dir, arm, "final.pt"), dev)
        if arm == "fmw":
            memory.__dict__["feature_flow"] = base_flow
    panel = panel_20()[: args.panel_n]
    cum = {"episodes": 0, "episodes_by_purpose": {}, "grad_steps": 0, "t0": time.time()}
    summary = {"method": args.method, "variant": variant, "regime": "E", "pair": [A, B], "order": "AB",
               "seed": args.seed, "args": vars(args), "cycles": [], "finished": False}

    def set_exec(policy):
        policy.temporal_agg = args.exec_mode == "agg"
        policy.action_horizon = 8
        policy.batch_size = None
        policy.action_queue = None

    def evaluate(policy, task, fold, purpose, ckpt, cycle, mem=None, W=None, instr="real", budget=None,
                 rnd=None, n=None, recorders=(None, None), extra=None):
        if fold == "test":
            st = gen_starts.load(task)
            if st is None:
                return None
            n = n or args.n_test
            runner = make_runner(shape_meta, "libero_90", n, args.par, 0, dev, init_states=st[:n],
                                 init_indices=list(range(n)))
        else:
            n = n or (args.n_val if fold == "validation" else None)
            ids = FOLDS[fold][: n] if n else FOLDS[fold]
            runner = make_runner(shape_meta, "libero_90", len(ids), args.par, 0, dev, init_indices=ids)
        pol = install_sampler(policy, Flow(policy), mem, W, recorders[0],
                              temb if (instr == "generic" and mem is not None) else None, recorders[1])
        set_exec(pol)
        r = run_task(runner, pol, task)
        cum["episodes"] += len(r["success"])
        cum["episodes_by_purpose"][purpose] = cum["episodes_by_purpose"].get(purpose, 0) + len(r["success"])
        row = {"method": args.method, "variant": variant, "regime": "E", "seed": args.seed, "pair": [A, B],
               "order": "AB", "cycle": cycle, "task": task, "budget": budget,
               "memory_mode": "off" if mem is None else "on_readonly", "instruction": instr if mem is not None else "real",
               "checkpoint": ckpt, "start_fold": fold, "reset_ids": r["init"], "success": r["success"],
               "length": r["length"], "rate": r["rate"], "purpose": purpose, "round": rnd,
               "cum_episodes": cum["episodes"], "cum_grad_steps": cum["grad_steps"],
               "wall_min": round((time.time() - cum["t0"]) / 60, 1), **(extra or {})}
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{args.method}/{variant} p{A}-{B} c{cycle}] {purpose:<16} task {task:>2} {ckpt:<14} "
              f"b={budget} r={rnd} {fold:<14} {r['rate']:.2f}", flush=True)
        return r["rate"]

    def panel_rate(policy, fold, purpose, ckpt, cycle, mem=None, W=None, rnd=None):
        n = args.n_old_val if fold == "old_validation" else args.n_old_test
        rates = [evaluate(policy, t, fold, purpose, ckpt, cycle, mem, W, "real", rnd=rnd, n=n) for t in panel]
        return mean(rates)

    def write_demos(theta, task, k, W=None, start=0):
        if arm == "horsea":  # learned writer on the demos as interaction histories, re-derived from W0
            with torch.enable_grad():
                return meta.adapt([demo_history(task, k)], create_graph=False)
        flow = Flow(theta)
        W = W if W is not None else memory.init_state(1, requires_grad=True)
        for d in range(start, k):
            e, a = wbank.demo(task, d)
            W = memory.write(flow, W, e, a, create_graph=False)
        return {kk: v.detach().requires_grad_(True) for kk, v in W.items()}

    hist_cache = {}

    def demo_history(task, k):
        """Demos 0..k-1 as experiences every 8 frames: context, demonstrated executed prefix, observed
        proprio change; probes from the frozen theta_0 reference sampler (fixed seeds)."""
        import h5py
        import numpy as np
        from imitation.envs.libero.utils import get_benchmark_instance
        from horsea.paths import DATA_PREFIX
        from horsea.writer import DPROP_SCALE, EXEC, K_PROBE
        if (task, k) in hist_cache:
            return hist_cache[(task, k)]
        bench = get_benchmark_instance("libero_90")
        path = os.path.join(DATA_PREFIX, "libero", bench.get_task_demonstration(task))
        rows = {"encm": [], "cmd": [], "dprop": [], "mask": [], "probes": [], "chunk": []}
        with h5py.File(path, "r") as f:
            names = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
            for d in range(k):
                o = f["data"][names[d]]["obs"]
                P = torch.from_numpy(np.concatenate([o["robot0_eef_pos"][()], o["robot0_gripper_qpos"][()]], 1)).float()
                e, a = rbank.demo(task, d)
                T = min(len(e), len(P))
                for i in range(0, T - 1, EXEC):
                    n = min(EXEC, T - 1 - i)
                    cmd, dp, m = torch.zeros(EXEC, 7), torch.zeros(EXEC, 5), torch.zeros(EXEC)
                    cmd[:n], dp[:n], m[:n] = a[i, :n].float().cpu(), (P[i + 1:i + 1 + n] - P[i:i + n]) * DPROP_SCALE, 1
                    g = torch.Generator(device=dev).manual_seed(1000 * d + i)
                    z = torch.randn(1, base_flow.chunk, base_flow.adim, device=dev, generator=g)
                    zs, t = [], torch.zeros(1, device=dev)
                    with torch.no_grad():
                        for kk in range(base_flow.n_steps):
                            if kk in K_PROBE:
                                zs.append(z[0].clone())
                            z = z + (1.0 / base_flow.n_steps) * base_flow.decode(z, t, e[i:i + 1].float())
                            t = t + 1.0 / base_flow.n_steps
                    rows["encm"].append(e[i]); rows["cmd"].append(cmd); rows["dprop"].append(dp); rows["mask"].append(m)
                    rows["probes"].append(torch.stack(zs)); rows["chunk"].append(a[i].float())
        H = {kk: torch.stack([v.to(dev) for v in vs]).float() if kk != "encm" else torch.stack(vs).to(dev)
             for kk, vs in rows.items()}
        H["n"] = len(rows["encm"])
        hist_cache[(task, k)] = H
        return H

    def anchor_loss(flow):
        i = anchor_idx[torch.randint(len(anchor_idx), (args.bs,), device=dev)]
        e, a = gbank.gather(i)
        x1 = a.clamp(-1, 1)
        t = flow.sample_t(len(x1), dev)
        psi, _ = flow.interp(torch.randn_like(x1), x1, t)
        hs, hs0 = [], []
        v = flow.decode(psi, t, e, layer_hook=lambda l, x: (hs.append(x), x)[1])
        with torch.no_grad():
            v0 = base_flow.decode(psi, t, e, layer_hook=lambda l, x: (hs0.append(x), x)[1])
        lh = sum(((h - h0) ** 2).mean() / (h0 ** 2).mean().clamp_min(1e-6) for h, h0 in zip(hs, hs0)) / len(hs)
        return args.anchor * (((v - v0) ** 2).mean() + lh)

    def replay_fn(buffers, boost):
        def replay_batch():
            n_old = args.bs if not buffers else args.bs // 2
            e, a = rbank.gather(old_idx[torch.randint(len(old_idx), (n_old,), device=dev)])
            if buffers:
                be, ba = torch.cat([b[0] for b in buffers]), torch.cat([b[1] for b in buffers])
                j = torch.randint(len(be), ((args.bs - n_old) * boost,), device=dev)
                e, a = torch.cat([e, be[j].float()]), torch.cat([a, ba[j].float()])
            return e, a
        return replay_batch

    def save_theta(theta, name):
        torch.save({k: v.detach().half().cpu() for k, v in theta.velocity_net.state_dict().items()
                    if k.startswith(DECODER_PREFIXES)}, os.path.join(out, name))

    theta, buffers, accepted = base, [], {"ckpt": "theta0"}
    ref_old = panel_rate(base, "old_validation", "base_panel", "theta0", 0)
    accepted["old_val"] = ref_old
    prev_val = {}  # earlier task -> its validation rate at the preceding accepted checkpoint
    stopped = False
    for c, T in enumerate([A, B], start=1):
        ckpt_in = accepted["ckpt"]
        cyc = {"cycle": c, "task": T, "theta_in": ckpt_in}
        s_base = evaluate(theta, T, "validation", "base", ckpt_in, c)
        cyc["base_val"] = s_base
        # ---- 2. adaptation curve ------------------------------------------------------------
        curve, W, teacher = {}, None, None
        for k in args.budgets:
            if memory is not None:
                W = write_demos(theta, T, k, W, start=max([b for b in args.budgets if b < k], default=0) if W is not None else 0)
                curve[k] = evaluate(theta, T, "validation", "acquisition", ckpt_in, c, memory, W, args.instr, budget=k)
            else:  # fine-tune theta_j on the first k demos (replay of old demos + earlier cycles)
                e = torch.cat([rbank.demo(T, d)[0] for d in range(k)])
                a = torch.cat([rbank.demo(T, d)[1] for d in range(k)])
                ft, hist = train_decoder(theta, demo_batcher(e, a, args.bs), args.steps, lr=args.lr,
                                         replay_batch=replay_fn(buffers, 1), tag=f"[ft c{c} k{k}]")
                cum["grad_steps"] += args.steps
                curve[k] = evaluate(ft, T, "validation", "acquisition", f"{ckpt_in}+ft{k}", c, budget=k)
                teacher = ft
        cyc["acquisition_curve"] = curve
        k_max = max(args.budgets)
        s_teacher = curve[k_max]
        cyc["teacher_val"] = s_teacher
        if c == 2 and memory is not None and not args.fast:  # same fresh memory on theta_0 for task B (renewed plasticity)
            W0B = None
            ref_curve = {}
            for k in args.budgets:
                W0B = write_demos(base, T, k, W0B, start=max([b for b in args.budgets if b < k], default=0) if W0B is not None else 0)
                ref_curve[k] = evaluate(base, T, "validation", "acquisition_theta0", "theta0", c, memory, W0B, args.instr, budget=k)
            cyc["acquisition_curve_on_theta0"] = ref_curve
        if memory is not None:  # active-memory takeover check (earlier + old tasks with W on)
            if c == 2:
                cyc["active_memory_prev"] = evaluate(theta, A, "validation", "active_memory", ckpt_in, c, memory, W, "real", budget=k_max)
            cyc["active_memory_old_panel"] = panel_rate(theta, "old_validation", "active_memory", ckpt_in, c, memory, W)
            torch.save({k: v.detach().cpu() for k, v in W.items()}, os.path.join(out, f"teacher_W_c{c}.pt"))
        gain = s_teacher - s_base
        cyc["teacher_gain"] = gain
        cyc["useful_acquisition"] = gain >= args.gain_min
        if gain < args.gain_min and memory is not None:  # (fine-tune carries its policy forward regardless)
            cyc["status"] = "no_useful_acquisition"
            summary["cycles"].append(cyc)
            if True:
                stopped = True
                print(f"cycle {c}: teacher gain {gain:.2f} < {args.gain_min}: no useful acquisition", flush=True)
                if not args.forced_reset_diag:
                    break
                cyc["forced_reset"] = True
                cyc["status"] += " (forced reset, no consolidation)"
                prev_val[T] = s_base
                continue
        if memory is None:
            # ---- fine-tune: the 5-demo model is the new long memory ----------------------------
            theta = teacher
            ck = f"theta{c}"
            save_theta(theta, f"{ck}.pt")
            s_new = curve[k_max]
            s_prev = evaluate(theta, A, "validation", "validation", ck, c) if c == 2 else None
            old = panel_rate(theta, "old_validation", "validation", ck, c)
            cyc.update(status="sequential_finetune", new_val=s_new, prev_val=s_prev, old_val=old,
                       prev_drop=None if s_prev is None else prev_val[A] - s_prev, old_drop=accepted["old_val"] - old)
            # replay buffer for the next cycle: the task's demos (fine-tune's "stored examples")
            e = torch.cat([rbank.demo(T, d)[0] for d in range(k_max)])
            a = torch.cat([rbank.demo(T, d)[1] for d in range(k_max)])
            buffers.append((e.half(), a.half()))
        else:
            # ---- 5. teacher collection (frozen theta_j + W, writes disabled) -------------------
            rg, rr = [], []
            ids = FOLDS["teacher"][: args.n_teacher]
            runner = make_runner(shape_meta, "libero_90", len(ids), args.par, 0, dev, init_indices=ids)
            pol = install_sampler(theta, Flow(theta), memory, W, rg, temb if args.instr == "generic" else None, rr)
            set_exec(pol)
            r = run_task(runner, pol, T)
            cum["episodes"] += len(r["success"])
            cum["episodes_by_purpose"]["teacher_collect"] = cum["episodes_by_purpose"].get("teacher_collect", 0) + len(r["success"])
            with open(rows_path, "a") as f:
                f.write(json.dumps({"method": args.method, "variant": variant, "regime": "E", "seed": args.seed,
                                    "pair": [A, B], "cycle": c, "task": T, "budget": k_max, "memory_mode": "on_readonly",
                                    "instruction": args.instr, "checkpoint": f"{ckpt_in}+W{k_max}", "start_fold": "teacher",
                                    "reset_ids": r["init"], "success": r["success"], "length": r["length"],
                                    "rate": r["rate"], "purpose": "teacher_collect", "cum_episodes": cum["episodes"]}) + "\n")
            cyc["teacher_collect_success"] = r["rate"]
            sup_t = torch.cat([wbank.demo(T, d)[0] for d in range(k_max)])
            sup_s = torch.cat([rbank.demo(T, d)[0] for d in range(k_max)])
            ctx_t = torch.cat([torch.cat(rg).float().to(dev), sup_t])
            ctx_s = torch.cat([torch.cat(rr).float().to(dev), sup_s])
            flow_T, Wd = Flow(copy.deepcopy(theta)), {k: v.detach() for k, v in W.items()}

            def new_batch():
                i = torch.randint(len(ctx_t), (args.bs,), device=dev)
                with torch.no_grad():
                    return ctx_s[i], memory.sample(flow_T, Wd, ctx_t[i])

            with torch.no_grad():
                i = torch.randperm(len(ctx_t), device=dev)[:6000]
                store = (ctx_s[i].half(), torch.cat([memory.sample(flow_T, Wd, x) for x in ctx_t[i].split(512)]).half())
            # ---- 6. consolidation rounds with the gate ----------------------------------------
            student, boost, rounds = theta, 1, []
            for rnd in range(1, args.max_rounds + 1):
                student, hist = train_decoder(student, new_batch, args.steps, lr=args.lr,
                                              replay_batch=replay_fn(buffers, boost),
                                              tag=f"[{args.method} c{c} r{rnd}]",
                                              extra_loss=anchor_loss if args.anchor > 0 else None)
                cum["grad_steps"] += args.steps
                ck = f"theta{c}_r{rnd}"
                s_new = evaluate(student, T, "validation", "validation", ck, c, rnd=rnd)
                s_prev = evaluate(student, A, "validation", "validation", ck, c, rnd=rnd) if c == 2 else None
                old = panel_rate(student, "old_validation", "validation", ck, c, rnd=rnd)
                ok_transfer = (s_new - s_base) >= args.retain * gain
                ok_prev = s_prev is None or (prev_val[A] - s_prev) <= args.prev_tol
                ok_old = (accepted["old_val"] - old) <= args.old_tol
                rounds.append({"round": rnd, "new_val": s_new, "prev_val": s_prev, "old_val": old,
                               "ok_transfer": ok_transfer, "ok_prev": ok_prev, "ok_old": ok_old,
                               "replay_boost": boost, "last_loss": hist[-1]})
                print(f"[{args.method} c{c} r{rnd}] new {s_new:.2f} (base {s_base:.2f}, teacher {s_teacher:.2f}) "
                      f"prev {s_prev} old {old:.2f} (ref {accepted['old_val']:.2f}) -> "
                      f"{ok_transfer and ok_prev and ok_old}", flush=True)
                if ok_transfer and ok_prev and ok_old:
                    break
                if not (ok_prev and ok_old):
                    boost *= 2  # preregistered replay schedule: more replay after a retention failure
            cyc["rounds"] = rounds
            last = rounds[-1]
            passed = last["ok_transfer"] and last["ok_prev"] and last["ok_old"]
            cyc.update(new_val=last["new_val"], prev_val=last["prev_val"], old_val=last["old_val"],
                       student_gain=last["new_val"] - s_base,
                       prev_drop=None if last["prev_val"] is None else prev_val[A] - last["prev_val"],
                       old_drop=accepted["old_val"] - last["old_val"])
            buffers.append(store)
            if passed:
                cyc["status"] = "accepted"
                theta = student
                save_theta(theta, f"theta{c}.pt")
            else:
                cyc["status"] = "gate_failed"
                save_theta(student, f"theta{c}_unaccepted.pt")
                stopped = True
                if not args.forced_reset_diag:
                    summary["cycles"].append(cyc)
                    break
                cyc["forced_reset"] = True
                theta = student
        # ---- 7. accepted (or forced) checkpoint after reset: test starts ------------------------
        ck = f"theta{c}"
        if args.fast:  # final table uses the (already measured) validation numbers
            prev_val[T] = cyc.get("new_val", curve[k_max])
            accepted = {"ckpt": ck, "old_val": cyc.get("old_val", accepted["old_val"])}
            summary["cycles"].append(cyc)
            json.dump(summary, open(summ_path, "w"), indent=1)
            continue
        cyc["test_new"] = evaluate(theta, T, "test", "test", ck, c)
        if c == 2:
            cyc["test_prev"] = evaluate(theta, A, "test", "test", ck, c)
        cyc["test_old_panel"] = panel_rate(theta, "old_test", "test", ck, c)
        prev_val[T] = cyc.get("new_val", curve[k_max])
        accepted = {"ckpt": ck, "old_val": cyc.get("old_val", accepted["old_val"])}
        summary["cycles"].append(cyc)
        summary["resources"] = {k: v for k, v in cum.items() if k != "t0"}
        json.dump(summary, open(summ_path, "w"), indent=1)
    summary["completed_two_cycles"] = (len(summary["cycles"]) == 2 and not stopped
                                       and all(cy["status"] in ("accepted", "sequential_finetune") for cy in summary["cycles"]))
    summary["resources"] = {k: v for k, v in cum.items() if k != "t0"}
    summary["resources"]["wall_min"] = round((time.time() - cum["t0"]) / 60, 1)
    summary["finished"] = True
    json.dump(summary, open(summ_path, "w"), indent=1)
    print("done", json.dumps({c["cycle"]: c["status"] for c in summary["cycles"]}), flush=True)


if __name__ == "__main__":
    main()
