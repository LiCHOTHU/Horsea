"""Paired closed-loop LIBERO evaluation for internal looping (spec 2026-09-30, sec. 8/10/13).

Control protocol: the base checkpoint's own (temporal aggregation ON: replan every env step, average overlapping
chunk predictions; chunk 16; horizon 300; native preprocessing, normalization, clipping, gripper conversion).
Sampler: native Euler over K evaluations of velocity_net.forward_dec (patched by horsea.loop.core.install), with
the initial FM noise of every generation drawn from a generator keyed by (eval seed, task, start id, replanning
index) -- independent of the loop configuration, so methods are paired on task, start and noise.
Episodes end at the first success or after 300 steps. Infrastructure failures are logged and retried with the same
episode keys; behavioural failures are never retried.

    python -m horsea.loop.evaluate --name base_K10 --tasks 23 32 81 --starts 0-19 --out experiments/loop/dev
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import time
import types

import numpy as np
import torch

import horsea  # noqa: F401
import imitation.envs.libero.wrappers as lw
from horsea.base import load_policy
from horsea.loop.core import CallCounter, LoopConfig, install
from horsea.paths import BASE_CKPT
from horsea.rollout import make_runner

HORIZON = 300


def key(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:15], 16)


def parse_starts(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def patch_sampler(policy, ctx):
    def sample_actions(self, data):
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            cond = self.get_cond(data)
            B, dev = cond.shape[0], cond.device
            noise = torch.stack([torch.randn(self.chunk_size, self.network_action_dim, device=dev,
                                             generator=torch.Generator(device=dev).manual_seed(
                                                 key(ctx["seed"], ctx["task"], ctx["starts"][b], ctx["call"])))
                                 for b in range(B)])
            enc = self.velocity_net.forward_enc(cond)
            K = self.num_inference_steps
            t = torch.zeros(B, device=dev, dtype=cond.dtype)
            z = noise
            for _ in range(K):
                z = z + (1.0 / K) * self.velocity_net.forward_dec(z, t, enc)
                t = t + 1.0 / K
            ctx["call"] += 1
            ctx["clip_frac"].append(float((z.abs() > 1).float().mean()))
            return torch.clamp(z, -1, 1).cpu().numpy()

    policy.sample_actions = types.MethodType(sample_actions, policy)


def run_batch(runner, policy, task, starts, ctx):
    B = len(starts)
    env_fn = lambda: lw.LiberoFrameStack(runner.env_factory(task_id=task, benchmark=runner.benchmark), 1)
    env = lw.LiberoVectorWrapper(env_fn, B)
    inits = runner.benchmark.get_task_init_states(task)[np.asarray(starts)]
    task_emb = {k: v.repeat(B, 1) for k, v in runner.benchmark.get_task_emb(task).items()}
    succ, when = [False] * B, [None] * B
    ctx.update(task=task, starts=starts, call=0)
    try:
        obs, _ = env.reset(init_states=inits)
        policy.reset()
        policy.batch_size = None  # temporal-aggregation buffers sized on the first call
        for step in range(HORIZON):
            act = policy.get_action(obs, task, **task_emb)
            obs, _, _, _, info = env.step(act)
            for b in range(B):
                if info[b]["success"] and not succ[b]:
                    succ[b], when[b] = True, step + 1
            if all(succ):
                break
    finally:
        try:
            env._env.close()
        except Exception:  # noqa: BLE001
            pass
    return succ, when


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="arm name (row label)")
    ap.add_argument("--tasks", type=int, nargs="+", required=True, help="LIBERO-90 task ids")
    ap.add_argument("--starts", required=True, help="e.g. 0-19")
    ap.add_argument("--K", type=int, default=None, help="outer Euler evaluations (default: checkpoint's 10)")
    ap.add_argument("--loop", default=None, help='LoopConfig JSON, e.g. {"enabled":true,"layer_ids":[1],"repeats":2}')
    ap.add_argument("--ckpt", default=None, help="continuation checkpoint (trainable params + loop meta)")
    ap.add_argument("--use_ckpt_loop", action="store_true", help="use the loop schedule stored in --ckpt")
    ap.add_argument("--seed", type=int, default=0, help="evaluation seed (noise keys)")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()
    cfg = LoopConfig(enabled=False)
    ckpt_meta = None
    if args.ckpt:
        from horsea.loop.train import load_trainable
        s = load_trainable(policy, args.ckpt)
        ckpt_meta = s["meta"]
        if args.use_ckpt_loop:
            cfg = LoopConfig(**ckpt_meta["loop"])
    if args.loop:
        cfg = LoopConfig(**json.loads(args.loop))
    if args.K:
        policy.num_inference_steps = args.K
    install(policy.velocity_net, cfg, CallCounter())
    ctx = {"seed": args.seed, "clip_frac": []}
    patch_sampler(policy, ctx)
    assert policy.temporal_agg, "control protocol: the checkpoint's temporal aggregation must stay on"
    json.dump({"name": args.name, "K": policy.num_inference_steps, "loop": json.loads(cfg.to_json()),
               "ckpt": args.ckpt, "ckpt_meta": ckpt_meta, "seed": args.seed, "horizon": HORIZON,
               "temporal_agg": policy.temporal_agg, "chunk": policy.chunk_size}, open(os.path.join(out, "config.json"), "w"),
              indent=1)
    multiprocessing.set_start_method("spawn", force=True)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", 2, 2, 0, dev, horizon=HORIZON)
    ep_path = os.path.join(out, "episodes.jsonl")
    done = set()
    if os.path.exists(ep_path):
        done = {(r["task"], r["start"]) for r in map(json.loads, open(ep_path)) if not r.get("infra_failure")}
    starts_all = parse_starts(args.starts)
    for task in args.tasks:
        todo = [s for s in starts_all if (task, s) not in done]
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            if len(chunk) == 1:
                chunk = chunk + chunk  # a 1-env batch renders in-process (breaks EGL for later batches)
            for attempt in range(3):  # infrastructure retries only, same episode keys
                t0 = time.time()
                ctx["clip_frac"] = []
                try:
                    succ, when = run_batch(runner, policy, task, chunk, ctx)
                    break
                except Exception as e:  # noqa: BLE001
                    with open(ep_path, "a") as f:
                        f.write(json.dumps({"task": task, "starts": chunk, "infra_failure": repr(e)[:300],
                                            "attempt": attempt}) + "\n")
                    if attempt == 2:
                        raise
            with open(ep_path, "a") as f:
                seen = set()
                for b, s in enumerate(chunk):
                    if s in seen:
                        continue
                    seen.add(s)
                    f.write(json.dumps({"task": task, "start": s, "seed": args.seed, "arm": args.name,
                                        "success": bool(succ[b]), "steps": when[b] or HORIZON,
                                        "termination": "success" if succ[b] else "horizon",
                                        "batch_sec": round(time.time() - t0, 1),
                                        "clip_frac": round(float(np.mean(ctx["clip_frac"])), 5)}) + "\n")
            print(f"[{args.name}] task {task} starts {chunk[0]}..{chunk[-1]}: {np.mean(succ):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
