"""Held-out FM validation loss by checkpoint x graph x FM-time bin (Stage B diagnostics, spec sec. 10).

Records: 1,024 frames from the held-out demos (episodes 45-49) of all 50 tasks, no augmentation, with fixed FM noise
and fixed FM times drawn once from the training time distribution. Every checkpoint is evaluated under G0 and all 12
loop graphs (training-mode gating by each record's FM time, dropout off).

    python -m horsea.rt.diag_graphB --ckpts experiments/rt/base/step300000.pt experiments/rt/graphB/train/C_s0/u5000.pt ...
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from horsea.rt.continue_train import LOOPS
from horsea.rt.data import HELD_EPS, RTDataset, clip_cache, instructions, tasks
from horsea.rt.graph import install, library
from horsea.rt.policy import RTFlowPolicy

CACHE = "experiments/rt/graphB/diag_records.pt"


def records(dev, n=1024, seed=0):
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location=dev, weights_only=False)
    tl = tasks()
    clip = clip_cache([x for t in tl for e in HELD_EPS for x in instructions(t, e)])
    np.random.seed(seed)
    ds = RTDataset(tl, HELD_EPS, clip)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    batches = list(DataLoader(ds, batch_size=128, sampler=idx, num_workers=4))
    g = torch.Generator().manual_seed(seed + 1)
    rec = []
    for b in batches:
        m = b["actions"].shape[0]
        u = torch.rand(m, generator=g)
        t = (1 - 0.001) * (1 - u ** (1.0 / 1.5))
        x0 = torch.randn(b["actions"].shape, generator=g)
        rec.append({"imgs": b["imgs"], "state": b["state"], "lang": b["lang"], "actions": b["actions"], "t": t, "x0": x0,
                    "task": b["task"]})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(rec, CACHE)
    return rec


@torch.no_grad()
def loss_of(model, rec, dev):
    out, ts = [], []
    for r in rec:
        imgs = r["imgs"].to(dev).permute(0, 1, 4, 2, 3).float() / 255.0
        cond = model.obs_tokens(imgs, r["state"].to(dev), r["lang"].to(dev))
        x1 = model.norm_a(r["actions"].to(dev).float()).clamp(-1, 1)
        x0, t = r["x0"].to(dev), r["t"].to(dev)
        tt = t[:, None, None]
        psi = (1 - (1 - model.sig_min) * tt) * x0 + tt * x1
        _, v = model.velocity_net(psi, t, cond)
        out.append(((v - (x1 - (1 - model.sig_min) * x0)) ** 2).mean((1, 2)).cpu())
        ts.append(r["t"])
    return torch.cat(out), torch.cat(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/rt/graphB/diag_losses.json")
    a = ap.parse_args()
    dev = "cuda:0"
    rec = records(dev)
    lib = library("primary")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for ck in a.ckpts:
        s = torch.load(ck, map_location=dev, weights_only=False)
        m = RTFlowPolicy(s["stats"]).to(dev).eval()
        m.load_state_dict(s["model"])
        install(m.velocity_net, None)
        m.velocity_net._graph_step = None
        row = {}
        for gname in ["G0"] + LOOPS:
            m.velocity_net._graph = lib[gname]
            L, T = loss_of(m, rec, dev)
            bins = [(T < 1 / 3), (T >= 1 / 3) & (T < 2 / 3), (T >= 2 / 3)]
            row[gname] = {"loss": float(L.mean()), "by_t": [float(L[b].mean()) for b in bins], "finite": bool(torch.isfinite(L).all())}
        res[ck] = row
        print(ck, {g: round(v["loss"], 5) for g, v in row.items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
