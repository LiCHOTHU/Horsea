"""Train the RoboTwin multi-task FM policy (all 50 tasks, aloha-agilex clean, demos 0-44 per task).

Optimisation mirrors the LIBERO base: AdamW lr 1e-4, betas (0.95, 0.999), wd 1e-6, cosine schedule with 1,000
warm-up steps to 1% of peak, batch 64, grad-clip 100, fp32 (TF32 matmuls). Augmentation as in the base: batch-wise
colour jitter (brightness/contrast/saturation 0.3) and random translation (4 px at 120x160), applied on the GPU.

    python -m horsea.rt.train --steps 200000 --out experiments/rt/base
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from horsea.rt.data import TRAIN_EPS, RTDataset, clip_cache, compute_stats, instructions, tasks
from horsea.rt.policy import RTFlowPolicy


def augment(x, gen=None):
    """x (B, C3, 3, H, W) in [0, 1]. Batch-wise colour jitter + per-sample random translation (replicate pad)."""
    B, C, _, H, W = x.shape
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    b = 1 + 0.6 * (r(B, 1, 1, 1, 1) - 0.5)
    c = 1 + 0.6 * (r(B, 1, 1, 1, 1) - 0.5)
    s = 1 + 0.6 * (r(B, 1, 1, 1, 1) - 0.5)
    x = x * b
    m = x.mean((2, 3, 4), keepdim=True)
    x = (x - m) * c + m
    g = x.mean(2, keepdim=True)
    x = ((x - g) * s + g).clamp(0, 1)
    pad = 4
    xp = torch.nn.functional.pad(x.flatten(0, 1), (pad, pad, pad, pad), mode="replicate").view(B, C, 3, H + 2 * pad, W + 2 * pad)
    dx = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
    rows = (dy[:, None] + torch.arange(H, device=dev)[None])                     # (B, H)
    cols = (dx[:, None] + torch.arange(W, device=dev)[None])                     # (B, W)
    bi = torch.arange(B, device=dev)[:, None, None]
    xp = xp.permute(0, 3, 4, 1, 2)                                               # (B, Hp, Wp, C, 3)
    out = xp[bi, rows[:, :, None], cols[:, None, :]]                             # (B, H, W, C, 3)
    return out.permute(0, 3, 4, 1, 2)


def lr_at(step, total, peak=1e-4, warm=1000, end=0.01):
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (end + (1 - end) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset (smoke tests); default all 50")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    task_list = args.tasks or tasks()
    stats = compute_stats(tasks())
    clip = clip_cache([s for t in task_list for e in TRAIN_EPS for s in instructions(t, e)])
    ds = RTDataset(task_list, TRAIN_EPS, clip)
    model = RTFlowPolicy(stats).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    step = 0
    ck = os.path.join(args.out, "last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
    json.dump({"tasks": task_list, "train_eps": "0-44", "frames": len(ds), "batch": args.batch, "steps": args.steps,
               "params": sum(p.numel() for p in model.parameters()),
               "dit_params": sum(p.numel() for p in model.velocity_net.parameters())},
              open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(f"{len(task_list)} tasks, {len(ds)} frames, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params",
          flush=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    t0, acc, n = time.time(), 0.0, 0
    model.train()
    while step < args.steps:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                            pin_memory=True, persistent_workers=False,
                            worker_init_fn=lambda w: np.random.seed((args.seed * 1000 + step + w) % 2**32))
        for b in loader:
            imgs = b["imgs"].to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0
            imgs = augment(imgs, gen)
            loss = model.loss(imgs, b["state"].to(dev), b["lang"].to(dev), b["actions"].to(dev), gen)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.step()
            step += 1
            acc += loss.item()
            n += 1
            if step % 200 == 0:
                rec = {"step": step, "loss": round(acc / n, 5), "grad_norm": round(float(gn), 3),
                       "lr": opt.param_groups[0]["lr"], "it_s": round(n / (time.time() - t0), 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                t0, acc, n = time.time(), 0.0, 0
            if step % args.save_every == 0 or step == args.steps:
                s = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "stats": stats}
                torch.save(s, ck)
                torch.save({"model": model.state_dict(), "step": step, "stats": stats},
                           os.path.join(args.out, f"step{step}.pt"))
            if step >= args.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
