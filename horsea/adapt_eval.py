"""Adaptation to novel tasks: closed-loop success vs number of demonstrations K.

    python -m horsea.adapt_eval --arm fmw --suite libero_90       # held-out LIBERO-90 tasks
    python -m horsea.adapt_eval --arm fmw --suite libero_10       # far-novel LIBERO-10 tasks

arms: base (zero-shot), ft (direct decoder fine-tune on the K demos, no memory),
      ttt | kv | fmw | res (write the K demos into fast memory: one write per demo, all frames).
--wrong writes the demos of a *different* novel task (is the gain memory content?).
Each (task, K) result is its own JSON, so the script resumes where it stopped.
"""
import argparse
import json
import os
import time

import torch

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.finetune import demo_batcher, train_decoder
from horsea.memory import build_memory
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, HELDOUT_90, LIBERO_10
from horsea.rollout import install_sampler, make_runner, run_task


def load_memory(arm, path, device):
    memory = build_memory(arm).to(device)
    memory.load_state_dict(torch.load(path, map_location=device, weights_only=False)["memory"])
    memory.eval().requires_grad_(False)
    return memory


def write_demos(memory, flow, bank, task, demos, device):
    state = memory.init_state(1, requires_grad=True)
    for d in demos:
        e, a = bank.demo(task, d)
        state = memory.write(flow, state, e.to(device), a.to(device), create_graph=False)
    return {k: v.detach() for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "ft", "ttt", "ttt2", "kv", "fmw", "res"])
    ap.add_argument("--suite", default="libero_90", choices=["libero_90", "libero_10"])
    ap.add_argument("--tasks", type=int, nargs="+", default=None)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--n", type=int, default=20, help="rollouts per (task, K)")
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--wrong", action="store_true")
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory"))
    ap.add_argument("--ft_steps", type=int, default=500)
    ap.add_argument("--ft_lr", type=float, default=5e-5)
    ap.add_argument("--ft_bs", type=int, default=128)
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--features", default=FEAT_DIR)
    ap.add_argument("--generic", action="store_true",
                    help="every task gets the same generic instruction; demos are the only task signal")
    ap.add_argument("--out", default=os.path.join(EXP, "adapt"))
    args = ap.parse_args()

    tasks = args.tasks or (HELDOUT_90 if args.suite == "libero_90" else LIBERO_10)
    Ks = [0] if args.arm == "base" else args.K
    out = os.path.join(args.out, args.arm, args.suite)
    os.makedirs(out, exist_ok=True)

    policy, sd = load_policy(args.ckpt, args.device)
    flow = Flow(policy)
    bank = FeatureBank(os.path.join(args.features, f"{args.suite}{'_generic' if args.generic else ''}.pt"), "cpu")
    temb = bank.task_emb[0] if args.generic else None
    memory = None
    if args.arm in ("ttt", "ttt2", "kv", "fmw", "res"):
        memory = load_memory(args.arm, os.path.join(args.mem_dir, args.arm, "final.pt"), args.device)
    runner = make_runner(sd["config"]["task"]["shape_meta"], args.suite, args.n, args.par, 0, args.device, args.horizon)

    for task in tasks:
        for K in Ks:
            tag = f"t{task}_K{K}" + ("_wrong" if args.wrong else "")
            path = os.path.join(out, tag + ".json")
            if os.path.exists(path):
                continue
            src = tasks[(tasks.index(task) + 1) % len(tasks)] if args.wrong else task
            torch.manual_seed(args.seed + 1000 * task + K)
            t0 = time.time()
            info = {}
            if args.arm == "base":
                pol = install_sampler(policy, flow, task_emb=temb)
            elif args.arm == "ft":
                e = torch.cat([bank.demo(src, d)[0] for d in range(K)]).to(args.device)
                a = torch.cat([bank.demo(src, d)[1] for d in range(K)]).to(args.device)
                student, hist = train_decoder(policy, demo_batcher(e, a, args.ft_bs), args.ft_steps,
                                              lr=args.ft_lr, tag=f"[ft {tag}]")
                pol = install_sampler(student, Flow(student), task_emb=temb)
                info["ft_hist"] = hist
            else:
                state = write_demos(memory, flow, bank, src, range(K), args.device)
                pol = install_sampler(policy, flow, memory, state, task_emb=temb)
            adapt_sec = time.time() - t0
            res = run_task(runner, pol, task)
            res.update({"arm": args.arm, "suite": args.suite, "K": K, "wrong": args.wrong, "support_task": src,
                        "generic": args.generic,
                        "adapt_sec": round(adapt_sec, 2), "n_support_frames": int(sum(
                            bank.ptr[src, d, 1] for d in range(K))) if K else 0, **info})
            if memory is not None:
                res["fast_params"] = memory.fast_numel()
            with open(path + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(path + ".tmp", path)
            print(f"[{args.arm} {args.suite}] task {task} K={K}{' WRONG' if args.wrong else ''}: "
                  f"{res['rate']:.2f}  (adapt {adapt_sec:.1f}s, rollout {res['rollout_sec']}s)", flush=True)
            if args.arm == "ft":
                del student
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
