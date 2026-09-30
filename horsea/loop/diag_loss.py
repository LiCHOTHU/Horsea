"""Offline FM-loss diagnostics of continuation checkpoints on FIXED held-out records (spec sec. 9).

1024 records from the diagnostic demos (45-49) of all 80 training tasks, with fixed FM noise and fixed times drawn
from the training time distribution; deterministic (inference) preprocessing. Each checkpoint is evaluated with the
loop schedule stored in it. Used for learning curves and the predeclared 5,000-update extension trigger
(both arms reduce mean loss by >= 2% between updates 1,000 and 2,000, all finite).

    python -m horsea.loop.diag_loss --runs experiments/loop/train/C_s0 experiments/loop/train/L_s0
"""
import argparse
import glob
import json
import os

import torch
from torch.utils.data import DataLoader

import horsea  # noqa: F401
from horsea.base import load_policy
from horsea.loop.core import CallCounter, LoopConfig, install
from horsea.loop.data import DIAG_DEMOS, build, to_device
from horsea.loop.train import fm_loss, load_trainable
from horsea.paths import BASE_CKPT

N_REC = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda:0"
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()
    alpha = float(sd["config"]["algo"]["policy"]["flow_alpha"])
    cache = "experiments/loop/diag_records.pt"
    if os.path.exists(cache):
        rec = torch.load(cache, weights_only=False)
    else:
        ds = build(sd["config"]["task"]["dataset"], DIAG_DEMOS)
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:N_REC].tolist()
        batches = [b for b in DataLoader(ds, batch_size=128, sampler=idx, num_workers=4)]
        g = torch.Generator().manual_seed(args.seed + 11)
        rec = {"batches": batches, "t": [], "x0": []}
        for b in batches:
            n = b["actions"].shape[0]
            u = torch.rand(n, generator=g)
            rec["t"].append(policy.flow_t_max * (1 - u ** (1.0 / alpha)))
            rec["x0"].append(torch.randn(b["actions"].shape, generator=g))
        torch.save(rec, cache)
    for run in args.runs:
        out = {}
        for ck in sorted(glob.glob(os.path.join(run, "u*.pt")), key=lambda p: int(os.path.basename(p)[1:-3])):
            s = load_trainable(policy, ck)
            cfg = LoopConfig(**s["meta"]["loop"])
            install(policy.velocity_net, cfg, CallCounter())
            losses, ts = [], []
            with torch.no_grad():
                for b, t, x0 in zip(rec["batches"], rec["t"], rec["x0"]):
                    per, _ = fm_loss(policy, to_device(b, dev), None, alpha, train_mode=False, t=t.to(dev), x0=x0.to(dev))
                    losses.append(per.cpu())
                    ts.append(t)
            L_, T_ = torch.cat(losses), torch.cat(ts)
            bins = [(T_ < 1 / 3), (T_ >= 1 / 3) & (T_ < 2 / 3), (T_ >= 2 / 3)]
            u = int(os.path.basename(ck)[1:-3])
            out[u] = {"loss": float(L_.mean()), "finite": bool(torch.isfinite(L_).all()),
                      "by_t": [float(L_[m].mean()) for m in bins]}
            print(run, u, out[u], flush=True)
        json.dump(out, open(os.path.join(run, "diag_loss.json"), "w"), indent=1)



if __name__ == "__main__":
    main()
