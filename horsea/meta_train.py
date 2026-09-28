"""Meta-train one memory arm on the frozen base (80 LIBERO-90 training tasks).

Episode = one training task: K support demos (one write each, F evenly spaced frames per
write) followed by Q query frames from other demos of the same task. The query loss is
differentiated through all K writes (second order) into the memory's slow parameters, W0 and
the inner learning rates. The base policy never changes.

    python -m horsea.meta_train --arm fmw
Resumable: reloads <out>/ckpt.pt if present.
"""
import argparse
import json
import math
import os
import random
import time

import torch

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.memory import build_memory
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, HELDOUT_90, TRAIN_90


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ttt", "ttt2", "kv", "fmw", "res"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--E", type=int, default=16, help="episodes (tasks) per meta-batch")
    ap.add_argument("--F", type=int, default=16, help="frames per support demo (per write)")
    ap.add_argument("--Q", type=int, default=32, help="query frames per episode")
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--features", default=os.path.join(FEAT_DIR, "libero_90.pt"))
    ap.add_argument("--train_tasks", type=int, nargs="+", default=TRAIN_90)
    ap.add_argument("--val_tasks", type=int, nargs="+", default=HELDOUT_90)
    ap.add_argument("--tasks_per_episode", type=int, nargs="+", default=[1],
                    help="M values to sample from; M>1 = several tasks (distinct scenes) share one memory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def scene_of_tasks():
    from libero.libero.benchmark import get_benchmark
    b = get_benchmark("libero_90")()
    import re
    return {i: re.match(r"^(.*?SCENE\d+)", b.get_task(i).name).group(1) for i in range(b.n_tasks)}


def run_episodes(memory, flow, bank, sup_idx, qry_idx, E, create_graph):
    state = memory.init_state(E, requires_grad=not create_graph)
    for k in range(sup_idx.shape[0]):
        encm, act = bank.gather(sup_idx[k])
        state = memory.write(flow, state, encm, act, create_graph=create_graph)
    qe, qa = bank.gather(qry_idx)
    return memory.outer_loss(flow, state, qe, qa)


@torch.no_grad()
def monitor(memory, flow, bank, tasks, Ks, F, Q, seed):
    """Query loss after K writes on `tasks` (fixed episodes per call) -- a cheap adaptation curve."""
    out = {}
    for K in Ks:
        g = torch.Generator().manual_seed(seed)
        sup, qry = bank.sample_episodes(tasks, max(K, 1), F, Q, g)
        torch.manual_seed(seed)
        with torch.enable_grad():
            loss = run_episodes(memory, flow, bank, sup[:K], qry, len(tasks), create_graph=False)
        out[K] = round(float(loss), 5)
    return out


def main():
    args = parse()
    out = args.out or os.path.join(EXP, "memory", args.arm)
    os.makedirs(out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    policy, _ = load_policy(args.ckpt, args.device)
    flow = Flow(policy)
    bank = FeatureBank(args.features, args.device)
    memory = build_memory(args.arm).to(args.device)
    opt = torch.optim.AdamW(memory.parameters(), lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, rng = 0, random.Random(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        memory.load_state_dict(s["memory"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step, rng, g = s["step"], s["rng"], s["g"]
        print(f"resumed at step {step}", flush=True)
    else:
        with open(os.path.join(out, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    print(f"[{args.arm}] fast params/episode: {memory.fast_numel():,}  "
          f"slow params: {sum(p.numel() for p in memory.parameters()) - memory.fast_numel():,}", flush=True)

    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    scene = scene_of_tasks() if max(args.tasks_per_episode) > 1 else None
    while step < args.steps:
        K = rng.choice(args.K)
        M = rng.choice(args.tasks_per_episode)
        if M == 1:
            tasks = rng.sample(args.train_tasks, args.E)
            sup, qry = bank.sample_episodes(tasks, K, args.F, args.Q, g)
        else:
            groups = []
            for _ in range(args.E):
                scenes = rng.sample(sorted({scene[t] for t in args.train_tasks}), M)
                groups.append([rng.choice([t for t in args.train_tasks if scene[t] == sc]) for sc in scenes])
            sup, qry = bank.sample_multitask_episodes(groups, K, args.F, args.Q, g)
        loss = run_episodes(memory, flow, bank, sup, qry, args.E, create_graph=True)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(json.dumps({"step": step, "K": K, "skipped_nonfinite_loss": True}), flush=True)
            step += 1
            sched.step()
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1

        if step % 50 == 0:
            rec = {"step": step, "K": K, "loss": round(loss.item(), 5), "grad_norm": round(gn.item(), 4),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
                   "inner_lr": {k: round(v.item(), 4) for k, v in list(memory.lrs().items())[:4]}}
            if args.arm.startswith("ttt"):
                rec["gate"] = round(torch.tanh(memory.alpha).abs().mean().item(), 5)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            Ks = [0, 1, 5, 10]
            rec = {"step": step, "monitor": True,
                   "heldout": monitor(memory, flow, bank, args.val_tasks, Ks, args.F, args.Q, 1234),
                   "train_tasks": monitor(memory, flow, bank, args.train_tasks[:10], Ks, args.F, args.Q, 1234)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            tmp = ck + ".tmp"
            torch.save({"memory": memory.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "rng": rng, "g": g, "args": vars(args)}, tmp)
            os.replace(tmp, ck)
    torch.save({"memory": memory.state_dict(), "args": vars(args), "step": step}, os.path.join(out, "final.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
