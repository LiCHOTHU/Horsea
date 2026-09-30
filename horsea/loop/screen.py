"""Stage-1 offline screen of frozen single-block loops (spec sec. 8). A ranking diagnostic, not robot success.

128 diagnostic action chunks (held-out demos 45-49 of the development tasks, fixed seed), shared FM noise, two fixed
FM times per interval (noise-side 1/9, 2/9; middle 4/9, 5/9; action-side 7/9, 8/9). For every (layer, interval)
at R = 2, lambda = 1, normalized: paired FM MSE vs the unchanged network at the same (chunk, noise, t), velocity
change, looped-layer hidden norm, and clipping risk of the implied endpoint z + (1 - t) v.

    python -m horsea.loop.screen --out experiments/loop/screen
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import horsea  # noqa: F401
from horsea.base import load_policy
from horsea.loop.core import CallCounter, LoopConfig, decoder_forward, install
from horsea.loop.data import DIAG_DEMOS, build, to_device
from horsea.paths import BASE_CKPT

INTERVALS = {"noise": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "act": (2 / 3, 1.0)}
T_VALUES = {"noise": (1 / 9, 2 / 9), "mid": (4 / 9, 5 / 9), "act": (7 / 9, 8 / 9)}
DEV_TASK_IDX = None  # resolved from the manifest (train80 indices of the development tasks)


def velocity(vnet, psi, t, enc, cfg, cap_layer=None):
    te = vnet.time_net(t)
    x = vnet.ac_proj(psi).transpose(0, 1) + vnet.dec_pos
    box = {}

    def hook(l, h):
        if l == cap_layer:
            box["h"] = h
        return h

    x = decoder_forward(vnet.decoder.layers, x, te, enc, t, cfg, None, hook)
    return vnet.eps_out(x, te, enc[-1]), box.get("h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()
    vnet = policy.velocity_net
    install(vnet, LoopConfig(enabled=False), CallCounter())
    man = json.load(open("experiments/loop/manifest.json"))
    tr80 = man["train80_ids"]
    task_idx = [tr80.index(int(t)) for t in man["development_tasks"]]
    ds = build(sd["config"]["task"]["dataset"], DIAG_DEMOS, task_idx)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:args.n].tolist()
    batch = to_device(next(iter(DataLoader(ds, batch_size=args.n, sampler=idx, num_workers=0))), dev)
    with torch.no_grad():
        data = policy.preprocess_input(batch, train_mode=False)
        enc = vnet.forward_enc(policy.get_cond(data))
        x1 = torch.clamp(data["actions"], -1, 1)
        x0 = torch.randn(x1.shape, generator=torch.Generator(device=dev).manual_seed(args.seed + 1), device=dev)
        rows = []
        for iv, ts in T_VALUES.items():
            for tv in ts:
                t = torch.full((args.n,), tv, device=dev)
                psi = policy._psi_t(x0, x1, t)
                u = x1 - (1 - policy.flow_sig_min) * x0
                base_v = {l: velocity(vnet, psi, t, enc, LoopConfig(enabled=False), cap_layer=l) for l in range(4)}
                mse_b = ((base_v[0][0] - u) ** 2).mean((1, 2))
                for l in range(4):
                    cfg = LoopConfig(enabled=True, layer_ids=(l,), repeats=2, interval=INTERVALS[iv])
                    v, h = velocity(vnet, psi, t, enc, cfg, cap_layer=l)
                    mse = ((v - u) ** 2).mean((1, 2))
                    end_b = psi + (1 - tv) * base_v[l][0]
                    end_l = psi + (1 - tv) * v
                    rows.append({"layer": l, "interval": iv, "t": round(tv, 4),
                                 "dmse": (mse - mse_b).cpu().numpy(), "mse_base": float(mse_b.mean()),
                                 "dv": float((v - base_v[l][0]).norm(dim=(1, 2)).mean()),
                                 "h_norm_loop": float(h.norm(dim=-1).mean()),
                                 "h_norm_base": float(base_v[l][1].norm(dim=-1).mean()),
                                 "clip_base": float((end_b.abs() > 1).float().mean()),
                                 "clip_loop": float((end_l.abs() > 1).float().mean())})
    diag = open(os.path.join(args.out, "diagnostics.jsonl"), "w")
    summary = []
    rng = np.random.default_rng(args.seed)
    for l in range(4):
        for iv in INTERVALS:
            rs = [r for r in rows if r["layer"] == l and r["interval"] == iv]
            d = np.concatenate([r["dmse"] for r in rs])
            bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            s = {"layer": l, "interval": iv, "dmse_mean": float(d.mean()),
                 "dmse_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 "rel_dmse": float(d.mean() / np.mean([r["mse_base"] for r in rs])),
                 "dv": float(np.mean([r["dv"] for r in rs])),
                 "h_norm_loop": float(np.mean([r["h_norm_loop"] for r in rs])),
                 "h_norm_base": float(np.mean([r["h_norm_base"] for r in rs])),
                 "clip_base": float(np.mean([r["clip_base"] for r in rs])),
                 "clip_loop": float(np.mean([r["clip_loop"] for r in rs]))}
            summary.append(s)
            for r in rs:
                diag.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}) + "\n")
    best = {iv: min((s for s in summary if s["interval"] == iv), key=lambda s: s["dmse_mean"]) for iv in INTERVALS}
    res = {"summary": summary, "best_per_interval": best,
           "reference": {"layer": 1, "interval": "all", "note": "predeclared central block at all t"},
           "chunks": args.n, "tasks": list(man["development_tasks"]), "demos": "45-49"}
    json.dump(res, open(os.path.join(args.out, "screen.json"), "w"), indent=1)
    for s in summary:
        print(f"layer {s['layer']} {s['interval']:5s} dMSE {s['dmse_mean']:+.5f} [{s['dmse_ci95'][0]:+.5f}, "
              f"{s['dmse_ci95'][1]:+.5f}] rel {100 * s['rel_dmse']:+.1f}%  |dv| {s['dv']:.3f}  "
              f"|h| {s['h_norm_base']:.1f}->{s['h_norm_loop']:.1f}  clip {s['clip_base']:.3f}->{s['clip_loop']:.3f}")
    print("best per interval:", {k: (v["layer"], round(v["dmse_mean"], 5)) for k, v in best.items()})


if __name__ == "__main__":
    main()
