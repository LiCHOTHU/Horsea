"""Recurrence-aware vs ordinary FM continuation (spec 2026-09-30, sec. 9).

Both arms: same checkpoint, original FM loss (pi0-style beta time sampler t = (1 - sig_min)(1 - b), b ~ Beta(1.5, 1),
sampled by inverse CDF b = u^(1/1.5) -- exact because the second Beta parameter is 1), same demos, batch 64,
AdamW(betas 0.95/0.999, wd 1e-6), grad clip 100 (base config), lr = 0.1 x base peak (1e-5), 100 linear warm-up
updates then constant, fresh optimizer. Train scope: velocity_net.{time_net (not the fixed frequencies w), ac_proj,
dec_pos, decoder, eps_out}; perception + observation encoders frozen and in eval mode; decoder dropout off (the
whole policy stays in eval mode; augmentation is applied explicitly by preprocess_input(train_mode=True)).

Paired randomness across arms (independent of the loop): data order from a per-epoch generator, FM noise/time from a
per-update generator, augmentation from a per-update reseed of the global RNG. Resuming reproduces the same stream.

    python -m horsea.loop.train --arm L --updates 2000 --out experiments/loop/train/L_s0
    python -m horsea.loop.train --arm C --updates 2000 --extra_save 2600 --out experiments/loop/train/C_s0
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

import horsea  # noqa: F401
from horsea.base import load_policy
from horsea.loop.core import CallCounter, LoopConfig, install
from horsea.paths import BASE_CKPT

TRAIN_PREFIXES = ("velocity_net.time_net.", "velocity_net.ac_proj.", "velocity_net.dec_pos", "velocity_net.decoder.",
                  "velocity_net.eps_out.")
FIXED = ("velocity_net.time_net.w",)


def trainable_params(policy):
    names, params = [], []
    for n, p in policy.named_parameters():
        if n.startswith(TRAIN_PREFIXES) and n not in FIXED:
            names.append(n)
            params.append(p)
    return names, params


def arm_config(arm, layer):
    if arm == "C":
        return LoopConfig(enabled=False)
    return LoopConfig(enabled=True, mode="normalized", layer_ids=(layer,), repeats=2, strength=1.0, interval=(0.0, 1.0))


def fm_loss(policy, batch, gen, alpha, train_mode=True, t=None, x0=None):
    """Original FM loss with explicit generators. Returns per-example loss (B,), t."""
    with torch.no_grad():
        data = policy.preprocess_input(batch, train_mode=train_mode)
        cond = policy.get_cond(data)
        enc = policy.velocity_net.forward_enc(cond)
    actions = data["abs_actions"] if policy.abs_action else data["actions"]
    B, dev = actions.shape[0], actions.device
    if t is None:
        u = torch.rand(B, generator=gen, device=dev)
        t = policy.flow_t_max * (1 - u ** (1.0 / alpha))
    if x0 is None:
        x0 = torch.randn(actions.shape, generator=gen, device=dev)
    x1 = torch.clamp(actions, -1, 1)
    psi = policy._psi_t(x0, x1, t)
    v = policy.velocity_net.forward_dec(psi, t, enc)
    target = x1 - (1 - policy.flow_sig_min) * x0
    return ((v - target) ** 2).mean((1, 2)), t


def epoch_batches(n, bs, seed, epoch):
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    perm = torch.randperm(n, generator=g)
    return [perm[i:i + bs].tolist() for i in range(0, n - bs + 1, bs)]  # drop_last


def lr_at(step, base_lr, warmup):
    return base_lr * min(1.0, (step + 1) / warmup)


def load_trainable(policy, path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    names, params = trainable_params(policy)
    for n, p in zip(names, params):
        p.data.copy_(s["params"][n])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["C", "L"])
    ap.add_argument("--layer", type=int, default=1, help="looped block for arm L (default floor((4-1)/2) = 1)")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--extra_save", type=int, nargs="*", default=[], help="additional checkpoint updates (C-compute)")
    ap.add_argument("--lr_scale", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--profile_only", type=int, default=0, help="time N updates and exit (compute profiling)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()                                     # encoders frozen in eval mode; decoder dropout off
    cfg = arm_config(args.arm, args.layer)
    counter = CallCounter()
    install(policy.velocity_net, cfg, counter)
    pol_cfg = sd["config"]["algo"]["policy"]
    assert pol_cfg.get("flow_sampling") == "beta" and float(pol_cfg.get("flow_beta", 1.0)) == 1.0
    alpha = float(pol_cfg["flow_alpha"])
    names, params = trainable_params(policy)
    for p in params:
        p.requires_grad_(True)
    opt_cfg = sd["optimizers"][0]["param_groups"][0]
    base_lr = float(sd["config"]["algo"]["lr"]) * args.lr_scale
    opt = torch.optim.AdamW(params, lr=base_lr, betas=tuple(float(b) for b in opt_cfg["betas"]), eps=float(opt_cfg["eps"]),
                            weight_decay=float(sd["config"]["algo"]["weight_decay"]))
    grad_clip = float(sd["config"]["training"]["grad_clip"])
    from horsea.loop.data import TRAIN_DEMOS, build, to_device
    ds = build(sd["config"]["task"]["dataset"], TRAIN_DEMOS)
    bs = int(sd["config"]["algo"]["batch_size"])
    step = 0
    ck = os.path.join(args.out, "last.pt")
    if os.path.exists(ck):  # resume (exact: all random streams are keyed by epoch/update)
        s = load_trainable(policy, ck)
        opt.load_state_dict(s["opt"])
        step = s["step"]
    meta = {"arm": args.arm, "loop": json.loads(cfg.to_json()), "base_ckpt": BASE_CKPT, "train_params": names,
            "lr": base_lr, "warmup": args.warmup, "batch": bs, "grad_clip": grad_clip, "seed": args.seed,
            "betas": [float(b) for b in opt_cfg["betas"]], "weight_decay": float(sd["config"]["algo"]["weight_decay"]),
            "demos": "demo_0..44"}
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    saves = sorted(set([0, 500, 1000, 2000, args.updates] + list(args.extra_save)))

    def save(tag):
        torch.save({"params": {n: p.detach().cpu() for n, p in zip(names, params)}, "opt": opt.state_dict(),
                    "step": step, "meta": meta}, os.path.join(args.out, f"u{tag}.pt"))
        torch.save({"params": {n: p.detach().cpu() for n, p in zip(names, params)}, "opt": opt.state_dict(),
                    "step": step, "meta": meta}, ck)

    if step == 0 and not args.profile_only:
        save(0)
    log = open(os.path.join(args.out, "train_metrics.jsonl"), "a")
    per_epoch = len(ds) // bs
    t_acc, n_acc = 0.0, 0
    while step < (args.profile_only or args.updates):
        epoch, off = divmod(step, per_epoch)
        batches = epoch_batches(len(ds), bs, args.seed, epoch)[off:]
        loader = DataLoader(ds, batch_sampler=batches, num_workers=args.workers, pin_memory=True,
                            multiprocessing_context="fork" if args.workers else None)
        for batch in loader:
            batch = to_device(batch, dev)
            torch.manual_seed(args.seed * 1_000_003 + step)            # augmentation stream, keyed by update
            gen = torch.Generator(device=dev).manual_seed(args.seed * 7_000_003 + step)  # FM noise/time stream
            for g in opt.param_groups:
                g["lr"] = lr_at(step, base_lr, args.warmup)
            torch.cuda.synchronize()
            t0 = time.time()
            counter.reset()
            per, t = fm_loss(policy, batch, gen, alpha)
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gn):
                raise RuntimeError(f"non-finite loss/grad at update {step}: stop and fix before spending budget")
            opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            step += 1
            t_acc += dt
            n_acc += 1
            bins = [(t < 1 / 3), (t >= 1 / 3) & (t < 2 / 3), (t >= 2 / 3)]
            rec = {"update": step, "loss": round(loss.item(), 6), "grad_norm": round(float(gn), 4),
                   "lr": opt.param_groups[0]["lr"], "sec": round(dt, 4), "block_calls": counter.logical,
                   "loss_by_t": [round(float(per[b].mean()), 6) if b.any() else None for b in bins]}
            if step % 25 == 0 or step <= 5:
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if args.profile_only and step >= args.profile_only:
                break
            if step in saves:
                save(step)
                print(json.dumps({"saved": step, "loss": rec["loss"], "sec_per_update": round(t_acc / n_acc, 4)}),
                      flush=True)
            if step >= args.updates:
                break
    prof = {"arm": args.arm, "updates_timed": n_acc, "sec_per_update": t_acc / max(1, n_acc),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "block_calls_per_example_per_update": None}
    json.dump(prof, open(os.path.join(args.out, "profile.json" if args.profile_only else "train_cost.json"), "w"),
              indent=1)
    print(json.dumps(prof), flush=True)


if __name__ == "__main__":
    main()
