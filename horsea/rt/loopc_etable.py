"""Held-out FM-error table E[k, l] (FM-time interval k x repeated block l) and the Phase-2 schedule proposals of the
loop-consistency study (2026-10-01).

Records: n held-out frames (demos 45-49 of all 50 tasks, never trained on), no augmentation. For every record and every
interval k = 0..9 one FM time t ~ U[k/10, (k+1)/10) and one noise x0 are drawn once (fixed seed) and shared by all
graphs, so every cell difference is paired. Each (record, k) is scored under the original graph (code -1, reference only)
and under the repeat of block l = 0..3 (r2(l): whole-block half-residual, the study's single operator), dropout off.
    E[k, l] = mean per-example FM MSE of the cells of interval k under block l.

Proposals (all candidates have a repeat at every evaluation, 50 block calls):
    Tfree = argmin_l E[k, l] separately for every k
    T1    = argmin over the 112 schedules whose repeated block changes at most once (4 constants + 9 x 12)
            of sum_k E[k, s_k]   (every solver evaluation weighted equally)
Stability: the proposals are recomputed on 2,000 record bootstraps and on the two record halves.

    python -m horsea.rt.loopc_etable --ckpt experiments/rt/graphB/train/M_s0/u5000.pt --out experiments/rt/loopc/etable_M.json
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from horsea.rt.data import HELD_EPS, RTDataset, clip_cache, instructions, tasks
from horsea.rt.graph import K, L, install, schedule
from horsea.rt.policy import RTFlowPolicy

CACHE = "experiments/rt/loopc/etable_records.pt"


def candidates_t1():
    out = [(l,) * K for l in range(L)]
    for c in range(1, K):
        for a, b in itertools.permutations(range(L), 2):
            out.append((a,) * c + (b,) * (K - c))
    assert len(out) == 112 and len(set(out)) == 112
    return out


def records(n, seed=0):
    if os.path.exists(CACHE):
        return torch.load(CACHE, weights_only=False)
    tl = tasks()
    clip = clip_cache([x for t in tl for e in HELD_EPS for x in instructions(t, e)])
    np.random.seed(seed)
    ds = RTDataset(tl, HELD_EPS, clip)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    g = torch.Generator().manual_seed(seed + 1)
    rec = []
    for b in DataLoader(ds, batch_size=128, sampler=idx, num_workers=8):
        m = b["actions"].shape[0]
        t = (torch.arange(K)[None].float() + torch.rand(m, K, generator=g)) / K        # one t per interval
        x0 = torch.randn(m, K, *b["actions"].shape[1:], generator=g)                   # one noise per (record, interval)
        rec.append({"imgs": b["imgs"], "state": b["state"], "lang": b["lang"], "actions": b["actions"], "t": t, "x0": x0,
                    "task": b["task"]})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(rec, CACHE)
    return rec


@torch.no_grad()
def cell_losses(model, rec, dev, grid=False):
    """(n_records, K, 5) per-example FM MSE; last axis = code -1 (original), 0, 1, 2, 3."""
    vn = model.velocity_net
    install(vn, schedule("-" * K))
    vn._graph_step = None
    out = []
    for r in rec:
        imgs = r["imgs"].to(dev).permute(0, 1, 4, 2, 3).float() / 255.0
        cond = model.obs_tokens(imgs, r["state"].to(dev), r["lang"].to(dev))
        enc = vn.forward_enc(cond)
        x1 = model.norm_a(r["actions"].to(dev).float()).clamp(-1, 1)
        m = x1.shape[0]
        L_ = torch.zeros(m, K, 5)
        for k in range(K):
            t = torch.full((m,), k / K, device=dev) if grid else r["t"][:, k].to(dev)
            x0 = r["x0"][:, k].to(dev)
            tt = t[:, None, None]
            psi = (1 - (1 - model.sig_min) * tt) * x0 + tt * x1
            target = x1 - (1 - model.sig_min) * x0
            for ci, code in enumerate((-1, 0, 1, 2, 3)):
                vn._rowcode = torch.full((m,), code, device=dev)
                v = vn.forward_dec(psi, t, enc)
                L_[:, k, ci] = ((v - target) ** 2).mean((1, 2)).cpu()
        vn._rowcode = None
        out.append(L_)
    return torch.cat(out).double().numpy()


def propose(E):
    """E (K, 4) -> Tfree, T1 (as tuples of blocks) and the T1 scores."""
    tfree = tuple(int(np.argmin(E[k])) for k in range(K))
    cands = candidates_t1()
    scores = np.array([sum(E[k, s[k]] for k in range(K)) for s in cands])
    return tfree, cands[int(np.argmin(scores))], scores, cands


def sched_str(s):
    return "s:" + "".join(str(b) for b in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmode", default="interval", choices=["interval", "grid", "density"],
                    help="interval: t ~ U[k/10,(k+1)/10); grid: t = k/10 exactly (the solver time of evaluation k); "
                         "density: interval cells weighted by the training FM-time density")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rec = records(a.n)
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = RTFlowPolicy(s["stats"]).to(dev).eval()
    model.load_state_dict(s["model"])
    C = cell_losses(model, rec, dev, grid=a.tmode == "grid")   # (n, K, 5)
    if a.tmode == "density":   # importance weights: training t = (1-s)(1-b), b ~ Beta(1.5, 1) -> p(t) ~ (1 - t/(1-s))^0.5
        tt = torch.cat([r["t"] for r in rec]).double().numpy()                                       # (n, K)
        w = np.clip(1 - tt / (1 - 0.001), 0, None) ** 0.5
        C = C * (w / w.mean(0, keepdims=True))[:, :, None]
    tasks_ = torch.cat([r["task"] for r in rec]).numpy()
    E = C[:, :, 1:].mean(0)                                # (K, 4): blocks 0..3
    E_orig = C[:, :, 0].mean(0)
    tfree, t1, scores, cands = propose(E)
    # paired standard errors of each block vs the interval's best block
    se = np.zeros((K, L))
    for k in range(K):
        best = int(np.argmin(E[k]))
        for l in range(L):
            d = C[:, k, 1 + l] - C[:, k, 1 + best]
            se[k, l] = d.std(ddof=1) / np.sqrt(len(d))
    # stability: record bootstrap and halves
    rng = np.random.default_rng(0)
    n = C.shape[0]
    bt_free, bt_t1 = {}, {}
    for _ in range(a.boot):
        Eb = C[rng.integers(0, n, n), :, 1:].mean(0)
        f, o, _, _ = propose(Eb)
        bt_free[sched_str(f)] = bt_free.get(sched_str(f), 0) + 1
        bt_t1[sched_str(o)] = bt_t1.get(sched_str(o), 0) + 1
    perm = rng.permutation(n)
    halves = [propose(C[perm[i::2], :, 1:].mean(0))[:2] for i in (0, 1)]
    per_k_free = [{l: 0 for l in range(L)} for _ in range(K)]
    for sch, c in bt_free.items():
        for k, ch in enumerate(sch[2:]):
            per_k_free[k][int(ch)] += c
    order = np.argsort(scores)
    res = {"ckpt": a.ckpt, "tmode": a.tmode, "n_records": int(n), "n_tasks": int(len(set(tasks_.tolist()))),
           "E_block": E.tolist(), "E_original": E_orig.tolist(), "se_vs_interval_best": se.tolist(),
           "Tfree": sched_str(tfree), "T1": sched_str(t1),
           "T1_top10": [{"sched": sched_str(cands[i]), "score": float(scores[i])} for i in order[:10]],
           "constant_scores": {f"S{l}": float(scores[l]) for l in range(L)},
           "Tfree_score": float(sum(E[k, tfree[k]] for k in range(K))),
           "bootstrap": {"n": a.boot, "Tfree_top": sorted(bt_free.items(), key=lambda x: -x[1])[:5],
                         "T1_top": sorted(bt_t1.items(), key=lambda x: -x[1])[:5],
                         "Tfree_per_interval_block_freq": [[per_k_free[k][l] / a.boot for l in range(L)] for k in range(K)]},
           "halves": [{"Tfree": sched_str(f), "T1": sched_str(o)} for f, o in halves]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"E[k, l] (x1e3) from {a.ckpt}, {n} held-out records; rows = FM-time interval k, columns = original, block 0-3")
    for k in range(K):
        print(f"  k={k} [{k / K:.1f},{(k + 1) / K:.1f}): orig {1e3 * E_orig[k]:8.4f} | " +
              " ".join(f"{1e3 * E[k, l]:8.4f}{'*' if l == tfree[k] else ' '}" for l in range(L)))
    print(f"Tfree = {sched_str(tfree)} (score {1e3 * res['Tfree_score']:.4f}); T1 = {sched_str(t1)} "
          f"(score {1e3 * scores.min():.4f}); constants " + ", ".join(f"S{l} {1e3 * scores[l]:.4f}" for l in range(L)))
    print("bootstrap Tfree:", res["bootstrap"]["Tfree_top"][:3], " T1:", res["bootstrap"]["T1_top"][:3])
    print("halves:", res["halves"])


if __name__ == "__main__":
    main()
