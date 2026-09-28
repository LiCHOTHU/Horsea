"""RoboTTT-style within-episode memory: TTT fast weights written with the robot's own history.

Mirrors RoboTTT's main evaluation (Jiang et al. 2026, Sec. 3.2 / 4):
  * post-train the base on the downstream task data (here LIBERO-10), either
      plain  -- decoder only, no memory ("single-step context" baseline), or
      ttt    -- decoder + TTT layers trained jointly on whole demo sequences: at each timestep the
                FM loss uses the fast weights written by all earlier timesteps, then that timestep's
                tokens (noised actions, sequence action forcing) are written; truncated BPTT;
  * at test time every episode starts from W0, the robot acts with the current fast weights,
    then writes its OWN predicted chunk + observation (no expert data at test time).
Timesteps are every `stride` env steps (= one executed action chunk).

    python -m horsea.history train --mode ttt
    python -m horsea.history eval  --mode ttt
"""
import argparse
import json
import math
import os
import random
import time
import types

import torch

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, decoder_parameters, load_policy
from horsea.finetune import clone_trainable
from horsea.memory import build_memory
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, LIBERO_10
from horsea.rollout import make_runner, run_task


ARM = {"ttt": "ttt2", "horsea": "fmw"}  # memory used in each history mode


def sequences(bank, tasks, E, stride, rng):
    """E whole demos from the start, subsampled every `stride` frames, padded to the longest."""
    rows, lens = [], []
    for _ in range(E):
        t = rng.choice(tasks)
        d = rng.randrange(bank.n_demos(t))
        s, n = bank.ptr[t, d].tolist()
        idx = torch.arange(s, s + n, stride)
        rows.append(idx)
        lens.append(len(idx))
    L = max(lens)
    idx = torch.stack([torch.cat([r, r[-1:].expand(L - len(r))]) for r in rows])  # (E, L)
    mask = torch.stack([torch.arange(L) < l for l in lens]).float()  # (E, L)
    return idx.to(bank.device), mask.to(bank.device)


def train(args):
    dev = args.device
    base, sd = load_policy(args.ckpt, dev)
    student = clone_trainable(base)
    student.train()
    flow = Flow(student)
    bank = FeatureBank(os.path.join(FEAT_DIR, f"{args.suite}.pt"), dev)
    memory = build_memory(ARM[args.mode]).to(dev) if args.mode in ARM else None
    groups = [{"params": decoder_parameters(student), "lr": args.lr_dec}]
    if memory is not None:
        groups.append({"params": list(memory.parameters()), "lr": args.lr_mem})
    opt = torch.optim.AdamW(groups, weight_decay=1e-6)
    out = os.path.join(EXP, "history", args.mode)
    os.makedirs(out, exist_ok=True)
    ck = os.path.join(out, "ckpt.pt")
    step, rng = 0, random.Random(args.seed)
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        student.velocity_net.load_state_dict(s["vnet"])
        if memory is not None:
            memory.load_state_dict(s["memory"])
        opt.load_state_dict(s["opt"])
        step, rng = s["step"], s["rng"]
        print("resumed at", step, flush=True)
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    while step < args.steps:
        for g, lr in zip(opt.param_groups, [args.lr_dec, args.lr_mem]):
            g["lr"] = lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        idx, mask = sequences(bank, args.tasks, args.E, args.stride, rng)
        E, L = idx.shape
        opt.zero_grad(set_to_none=True)
        state = memory.init_state(E) if memory is not None else None
        total, seg, n_tok = 0.0, 0.0, mask.sum()
        for t in range(L):
            e, a = bank.gather(idx[:, t])
            m = mask[:, t]
            if memory is None:
                x1 = a.clamp(-1, 1)
                x0 = torch.randn_like(x1)
                tt = flow.sample_t(E, dev)
                psi, u = flow.interp(x0, x1, tt)
                per = ((flow.decode(psi, tt, e) - u) ** 2).mean((1, 2))
            else:
                per = memory_rowwise_loss(memory, flow, state, e, a)
                state = memory.write(flow, state, e, a, create_graph=True)
            seg = seg + (per * m).sum() / n_tok
            if memory is not None and ((t + 1) % args.tbptt == 0 or t == L - 1):
                seg.backward()
                total += seg.item()
                seg = 0.0
                # TBPTT: carry the weights, cut the gradient; W0 still learns through the first segment
                state = {k: v.detach().requires_grad_(True) for k, v in state.items()}
        if memory is None:
            seg.backward()
            total = seg.item()
        params = [p for g in opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            rec = {"step": step, "loss": round(total, 5), "L": L, "sec": round(time.time() - t0, 1)}
            if hasattr(memory, "alpha"):
                rec["gate"] = round(torch.tanh(memory.alpha).abs().mean().item(), 4)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % 250 == 0 or step == args.steps:
            s = {"vnet": student.velocity_net.state_dict(), "opt": opt.state_dict(), "step": step, "rng": rng}
            if memory is not None:
                s["memory"] = memory.state_dict()
            torch.save(s, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    torch.save({"vnet": student.velocity_net.state_dict(), "memory": memory.state_dict() if memory else None,
                "args": vars(args)}, os.path.join(out, "final.pt"))
    print("done", flush=True)


def memory_rowwise_loss(memory, flow, state, encm, act):
    """Per-row FM loss of the memory-adapted field (one row per episode)."""
    x1 = act.clamp(-1, 1)
    x0 = torch.randn_like(x1)
    t = flow.sample_t(x1.shape[0], x1.device)
    psi, u = flow.interp(x0, x1, t)
    v = memory.field(flow, state, encm)(psi, t)
    return ((v - u) ** 2).mean((1, 2))


def install_history_sampler(policy, flow, memory, stride):
    """Each env keeps its own fast weights (episode index = batch row); reset at episode start;
    act with the current weights, then every `stride` steps write the robot's own chunk."""
    ctx = {"state": None, "count": 0}
    orig_reset = policy.reset

    def reset(self):
        orig_reset()
        ctx["state"], ctx["count"] = None, 0

    def sample_actions(self, data):
        with torch.no_grad():
            encm = flow.encode(data)
            if memory is None:
                a = flow.sample(encm)
            else:
                if ctx["state"] is None:
                    ctx["state"] = memory.init_state(encm.shape[0], requires_grad=True)
                a = memory.sample(flow, ctx["state"], encm)
                if ctx["count"] % stride == 0:
                    ctx["state"] = memory.write(flow, ctx["state"], encm, a, create_graph=False)
            ctx["count"] += 1
            return a.cpu().numpy()

    policy.reset = types.MethodType(reset, policy)
    policy.sample_actions = types.MethodType(sample_actions, policy)
    policy.batch_size = None
    return policy


def evaluate(args):
    dev = args.device
    policy, sd = load_policy(args.ckpt, dev)
    out = os.path.join(EXP, "history", args.mode)
    f = torch.load(os.path.join(out, "final.pt"), map_location=dev, weights_only=False)
    policy.velocity_net.load_state_dict(f["vnet"])
    policy.eval()
    memory = None
    if args.mode in ARM:
        memory = build_memory(ARM[args.mode]).to(dev)
        memory.load_state_dict(f["memory"])
        memory.eval().requires_grad_(False)
    flow = Flow(policy)
    runner = make_runner(sd["config"]["task"]["shape_meta"], args.suite, args.n, args.par, 0, dev)
    install_history_sampler(policy, flow, memory, args.stride)
    for task in args.tasks:
        path = os.path.join(out, f"eval_{args.suite}_t{task}.json")
        if os.path.exists(path):
            continue
        res = run_task(runner, policy, task)
        res.update({"mode": args.mode, "suite": args.suite})
        with open(path + ".tmp", "w") as fh:
            json.dump(res, fh)
        os.replace(path + ".tmp", path)
        print(f"[history {args.mode}] {args.suite} task {task}: {res['rate']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--mode", required=True, choices=["plain", "ttt", "horsea"])
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", type=int, nargs="+", default=LIBERO_10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tbptt", type=int, default=16)
    ap.add_argument("--lr_dec", type=float, default=5e-5)
    ap.add_argument("--lr_mem", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
