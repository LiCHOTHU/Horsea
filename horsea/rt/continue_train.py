"""Stage B continuation of the RoboTwin FM policy: ordinary C vs shared multi-graph M (spec 2026-10-01 rev. 2, sec. 10).

Both start from the identical step-300k checkpoint with a fresh optimizer:
  AdamW (betas 0.95/0.999, eps 1e-8, wd 1e-6) as in base training; lr = 0.1 x base peak (1e-5); 250 linear warm-up
  updates then constant; batch 64; grad-clip 100; the original 50-task mixture (demos 0-44 of every task), original
  colour-jitter + translation augmentation.
Trained: the FM denoiser -- velocity_net.{decoder blocks, ac_proj, dec_pos, time_net (not its fixed frequencies w),
  eps_out}. Frozen and in eval mode: vision (ResNet-18 + FiLM), language and state projections, camera embedding, and the
  DiT observation encoder that prepares the conditioning. Decoder dropout stays as in the original training (train mode).
Graphs: C always G0. M samples ONE graph per minibatch: P(G0) = 1/2, P(each of the 12 loop graphs) = 1/24, from its own
  random stream; each example's own FM time decides whether the graph's window is active (horsea.rt.graph executor,
  grouped rows, gradients through every visit). No new parameters.
Paired random streams (identical for C and M): data order per (seed, epoch); augmentation, FM noise/time, and graph
  sampling each from a dedicated generator keyed by (seed, stream, update).

    python -m horsea.rt.continue_train --arm M --updates 5000 --out experiments/rt/graphB/train/M_s0
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from horsea.rt.data import TRAIN_EPS, RTDataset, clip_cache, instructions, tasks
from horsea.rt.graph import WINDOWS, install, library
from horsea.rt.policy import RTFlowPolicy
from horsea.rt.train import augment

BASE = "experiments/rt/base/step300000.pt"
TRAIN_PREFIXES = ("velocity_net.time_net.", "velocity_net.ac_proj.", "velocity_net.dec_pos", "velocity_net.decoder.",
                  "velocity_net.eps_out.")
FIXED = ("velocity_net.time_net.w",)
LOOPS = [f"b{b}_{w}" for b in range(4) for w in ("early", "middle", "late")]


def key(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:15], 16)


def trainable(model):
    names, params = [], []
    for n, p in model.named_parameters():
        if n.startswith(TRAIN_PREFIXES) and n not in FIXED:
            names.append(n)
            params.append(p)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    return names, params


def set_modes(model):
    model.train()
    for m in (model.vision, model.lang, model.state, model.velocity_net.encoder, model.velocity_net.enc_pos):
        m.eval()


def sample_graph(arm, seed, step):
    if arm == "C":
        return "G0"
    g = torch.Generator().manual_seed(key(seed, "graph", step))
    u = float(torch.rand(1, generator=g))
    if u < 0.5:
        return "G0"
    return LOOPS[int(torch.randint(len(LOOPS), (1,), generator=g))]


def fm_per_example(model, imgs, state, lang, actions, gen):
    """Exactly RTFlowPolicy.loss (same RNG draw order), but returns per-example losses and FM times."""
    cond = model.obs_tokens(imgs, state, lang)
    x1 = model.norm_a(actions.float()).clamp(-1, 1)
    x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
    t = model.sample_t(x1.shape[0], x1.device, gen)
    tt = t[:, None, None]
    psi = (1 - (1 - model.sig_min) * tt) * x0 + tt * x1
    _, v = model.velocity_net(psi, t, cond)
    return ((v - (x1 - (1 - model.sig_min) * x0)) ** 2).mean((1, 2)), t


def epoch_batches(n, bs, seed, epoch):
    g = torch.Generator().manual_seed(key(seed, "order", epoch))
    perm = torch.randperm(n, generator=g)
    return [perm[i:i + bs].tolist() for i in range(0, n - bs + 1, bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["C", "M"])
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--save", type=int, nargs="*", default=[1000, 2500, 5000])
    ap.add_argument("--lr_scale", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    s = torch.load(BASE, map_location=dev, weights_only=False)
    model = RTFlowPolicy(s["stats"]).to(dev)
    model.load_state_dict(s["model"])
    install(model.velocity_net, None)
    lib = library("primary")
    names, params = trainable(model)
    base_lr = 1e-4 * a.lr_scale
    opt = torch.optim.AdamW(params, lr=base_lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    task_list = tasks()
    clip = clip_cache([x for t in task_list for e in TRAIN_EPS for x in instructions(t, e)])
    ds = RTDataset(task_list, TRAIN_EPS, clip)
    meta = {"arm": a.arm, "base": BASE, "base_sha256_16": hashlib.sha256(open(BASE, "rb").read()).hexdigest()[:16],
            "updates": a.updates, "lr": base_lr, "warmup": a.warmup, "batch": a.batch, "seed": a.seed,
            "graph_distribution": "G0 only" if a.arm == "C" else "P(G0)=1/2, P(each of 12 loops)=1/24, one graph per batch",
            "trainable": names, "n_trainable": sum(p.numel() for p in params), "frames": len(ds),
            "data": "all 50 tasks, demos 0-44 (same as base)", "dropout": "decoder dropout 0.1 active (original recipe)"}
    json.dump(meta, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    exposure = np.zeros((10, 4, 3), dtype=np.int64)    # (t decile, block, window): rows with an active repeat
    graph_counts = {}
    per_epoch = len(ds) // a.batch
    step, t_tot, n_tot = 0, 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    set_modes(model)
    while step < a.updates:
        epoch, off = divmod(step, per_epoch)
        batches = epoch_batches(len(ds), a.batch, a.seed, epoch)[off:]
        loader = DataLoader(ds, batch_sampler=batches, num_workers=a.workers, pin_memory=True,
                            worker_init_fn=lambda w, e=epoch: np.random.seed(key(a.seed, "worker", e, w) % 2**32))
        for b in loader:
            gname = sample_graph(a.arm, a.seed, step)
            graph_counts[gname] = graph_counts.get(gname, 0) + 1
            model.velocity_net._graph = lib[gname]
            model.velocity_net._graph_step = None
            for g in opt.param_groups:
                g["lr"] = base_lr * min(1.0, (step + 1) / a.warmup)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen_aug = torch.Generator(device=dev).manual_seed(key(a.seed, "aug", step))
            gen_fm = torch.Generator(device=dev).manual_seed(key(a.seed, "fm", step))
            imgs = b["imgs"].to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0
            imgs = augment(imgs, gen_aug)
            per, t = fm_per_example(model, imgs, b["state"].to(dev), b["lang"].to(dev), b["actions"].to(dev), gen_fm)
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 100.0)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError(f"non-finite loss/grad at update {step} (graph {gname})")
            opt.step()
            torch.cuda.synchronize()
            t_tot += time.perf_counter() - t0
            n_tot += 1
            if gname != "G0":
                blk, win = int(gname[1]), ("early", "middle", "late").index(gname.split("_")[1])
                lo, hi = WINDOWS[gname.split("_")[1]][1]
                tc = t.detach().cpu().numpy()
                act = (tc >= lo) & ((tc <= hi) if hi >= 1.0 else (tc < hi))
                for d in np.minimum((tc[act] * 10).astype(int), 9):
                    exposure[d, blk, win] += 1
            step += 1
            if step % 50 == 0 or step <= 3:
                tb = t.detach()
                bins = [(tb < 1 / 3), (tb >= 1 / 3) & (tb < 2 / 3), (tb >= 2 / 3)]
                log.write(json.dumps({"update": step, "graph": gname, "loss": round(loss.item(), 6), "grad_norm": round(float(gn), 4),
                                      "lr": opt.param_groups[0]["lr"], "sec": round(t_tot / n_tot, 4),
                                      "loss_by_t": [round(float(per[m].mean()), 6) if m.any() else None for m in bins]}) + "\n")
                log.flush()
            if step in a.save or step == a.updates:
                torch.save({"model": model.state_dict(), "stats": s["stats"], "step": step, "meta": meta},
                           os.path.join(a.out, f"u{step}.pt"))
                print(json.dumps({"saved": step, "loss": round(loss.item(), 5), "sec_per_update": round(t_tot / n_tot, 4)}),
                      flush=True)
            if step >= a.updates:
                break
    json.dump({"arm": a.arm, "updates": step, "sec_per_update": t_tot / max(1, n_tot), "gpu_sec_total": t_tot,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "graph_counts": graph_counts,
               "exposure_tdecile_block_window": exposure.tolist()}, open(os.path.join(a.out, "train_cost.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
