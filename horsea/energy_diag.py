"""Energy diagnostics (hidden-shift dev data): does the written energy rank actions correctly, and what
does the refinement do?

For each dev (task, shift) group: history = attempt 0; queries = decisions of attempt 1 (same shift).
Per query, a FIXED candidate set shared by all history conditions:
  base action a0 (the policy's own sample), corrective target (same noise), a0 rotated by
  +-15/30/45/60 deg, 2 noisy copies.
History conditions: correct, none (W0), mismatched (another shift's attempt), shuffled pairings.
Ranking: rank of the corrective target among candidates (0 = best), fraction of queries where the target
  beats a0, Spearman(energy, distance-to-target) over candidates.
Refinement (the implemented solver): energy before/after, ||a*-a0||, d(a*, target) vs d(a0, target),
  fraction of elements at the +-1 action limit.

    python -m horsea.energy_diag --ckpt experiments/energy/o1c_rotonly_final/last.pt
"""
import argparse
import json
import math
import random

import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.energy import EnergyMemory, Model, adapt
from horsea.manifest import WRITER_DEV
from horsea.paths import BASE_CKPT
from horsea.phi import Phi
from horsea.writer import load_groups


def rotate(a, deg):
    r = a.clone()
    t = math.radians(deg)
    r[..., 0] = math.cos(t) * a[..., 0] - math.sin(t) * a[..., 1]
    r[..., 1] = math.sin(t) * a[..., 0] + math.cos(t) * a[..., 1]
    return r.clamp(-1, 1)


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx * ry).sum() / (rx.norm() * ry.norm() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shifts", nargs="+", default=["rot30", "rot-30", "rot50", "rot-50"])
    ap.add_argument("--nq", type=int, default=24)
    ap.add_argument("--max_step", type=float, default=None, help="override the solver's per-step bound")
    ap.add_argument("--n_iter", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    random.seed(0)
    torch.manual_seed(0)
    pol, _ = load_policy(BASE_CKPT, dev)
    pol.requires_grad_(False)
    flow = Flow(pol)
    phi = Phi(device=dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a_ = ck["args"]
    EnergyMemory.nohist = a_.get("nohist", False)
    from horsea.energy import WRITER
    WRITER["kind"] = a_.get("writer", "batch")
    m = Model(a_["K"], a_["n_iter"], structured=a_.get("structured", False), max_step=a_.get("max_step")).to(dev)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    # architecture-aware: an old checkpoint (no gate) must be a batch-writer model; anything else missing is an error
    assert not unexpected and all(k.startswith("gate.") for k in missing), (missing, unexpected)
    assert not missing or ck["args"].get("writer", "batch") == "batch", "gated/seq checkpoint without gate weights"
    m.eval().requires_grad_(False)
    if args.max_step is not None:
        m.solver.max_step = args.max_step
    if args.n_iter is not None:
        m.solver.n_iter = args.n_iter
    groups = load_groups(WRITER_DEV, args.shifts, dev)
    by_task = {}
    for g in groups:
        by_task.setdefault(g["task"], []).append(g)
    agg = {c: {"target_rank": [], "target_beats_base": [], "spearman": [], "E_before": [], "E_after": [],
               "move": [], "d_base": [], "d_after": [], "at_limit": [], "grip_sign_flipped": [],
               "grip_sign_matches_target_base": [], "grip_sign_matches_target_after": [], "grip_mean_change": []} for c in ["correct", "none", "mismatched", "shuffled"]}
    for g in groups:
        h, f = g["eps"][0], g["eps"][1]
        others = [o for o in by_task[g["task"]] if o["shift"] != g["shift"]]
        states = {"none": m.mem.init_state(1)}
        with torch.enable_grad():
            states["correct"] = adapt(m, phi, [h], False)
            states["shuffled"] = adapt(m, phi, [h], False, shuffle=True)
            if others:
                states["mismatched"] = adapt(m, phi, [random.choice(others)["eps"][0]], False)
        q = torch.randperm(f["n"])[: args.nq].tolist()
        c, a0, tgt, eps = f["encm"][q].float(), f["chunk"][q], f["label"][q], f["probes"][q, 0]
        cands = torch.stack([a0, tgt] + [rotate(a0, d) for d in (-60, -45, -30, -15, 15, 30, 45, 60)] +
                            [(a0 + 0.2 * torch.randn_like(a0)).clamp(-1, 1) for _ in range(2)])  # (C, Q, 16, 7)
        C, Q = cands.shape[:2]
        dist_t = torch.stack([phi.dist(cands[i], tgt) for i in range(C)])  # (C, Q)
        for cond, st in states.items():
            with torch.no_grad():
                en = torch.stack([m.mem.energy(phi, st, cands[i], c) for i in range(C)])  # (C, Q)
            rank = (en < en[1:2]).sum(0).float()  # candidates cheaper than the target
            A = agg[cond]
            A["target_rank"].append(rank.mean().item())
            A["target_beats_base"].append((en[1] < en[0]).float().mean().item())
            A["spearman"].append(sum(spearman(en[:, j], dist_t[:, j]).item() for j in range(Q)) / Q)
            with torch.enable_grad():
                efn = lambda a, st=st: m.mem.energy(phi, st, a, c)
                raw = m.solver.solve(flow, efn, c, eps, a_["solve"], False).detach()
                base = m.solver.solve(flow, None, c, eps, "none", False).detach()
            with torch.no_grad():
                A["E_before"].append(m.mem.energy(phi, st, base, c).mean().item())
                A["E_after"].append(m.mem.energy(phi, st, raw, c).mean().item())
                A["move"].append((raw - base).abs().mean().item())
                A["d_base"].append(phi.dist(base, tgt).mean().item())
                A["d_after"].append(phi.dist(raw, tgt).mean().item())
                A["at_limit"].append((raw.abs() > 0.999).float().mean().item())
                # gripper (normalized dim 6; the executed command is its sign after unnormalization, threshold 0)
                gb, ga, gt = base[..., 6], raw[..., 6], tgt[..., 6]
                A["grip_sign_flipped"].append((torch.sign(ga) != torch.sign(gb)).float().mean().item())
                A["grip_sign_matches_target_base"].append((torch.sign(gb) == torch.sign(gt)).float().mean().item())
                A["grip_sign_matches_target_after"].append((torch.sign(ga) == torch.sign(gt)).float().mean().item())
                A["grip_mean_change"].append((ga - gb).abs().mean().item())
    res = {cond: {k: round(sum(v) / len(v), 4) for k, v in A.items() if v} for cond, A in agg.items()}
    for cond, r in res.items():
        print(cond, json.dumps(r), flush=True)
    if args.out:
        json.dump({"ckpt": args.ckpt, "shifts": args.shifts, "candidates": 12, "res": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
