"""Audit what a memory's writes do, on identical inputs (LIBERO-10 demo sequences, two-stage models).

For each method, along E demo sequences (one timestep = one 8-frame chunk; the executed chunk is the
demo's, as in training):
  * weight change   ||W_t - W0|| / ||W0||
  * velocity change ||v(W_t) - v(W0)|| / ||v(W0)|| at the next decision's context, same noise, at
    denoising times tau in {0, .3, .6, .9} (does the correction depend on tau / the noisy action?)
  * experience dependence: same query, memory written with shuffled action-outcome pairings
    (actions rolled across sequences): ||v(W_t) - v(W_t^shuffled)|| / ||v(W0)||
  * usefulness: FM loss on the NEXT demo action with W_t vs W0 vs W_t^shuffled (fresh noise, averaged)

    python -m horsea.audit_writes --modes horsea ttt
"""
import argparse
import json
import os
import random

import torch

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.history import sequences
from horsea.history2 import OUT, build, horsea_write
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, LIBERO_10


def norm(W):
    return torch.sqrt(sum((v.float() ** 2).sum() for v in W.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["horsea", "ttt"])
    ap.add_argument("--tag", default="_2stage")
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    bank = FeatureBank(os.path.join(FEAT_DIR, "libero_10.pt"), dev)
    res = {}
    for mode in args.modes:
        random.seed(0)
        torch.manual_seed(0)
        policy, _ = load_policy(BASE_CKPT, dev)
        f = torch.load(os.path.join(OUT + args.tag, mode, "final.pt"), map_location=dev, weights_only=False)
        policy.velocity_net.load_state_dict(f["vnet"])
        policy.eval().requires_grad_(False)
        flow = Flow(policy)
        memory, writer = build(mode, dev)
        memory.load_state_dict(f["memory"])
        memory.eval().requires_grad_(False)
        if writer is not None:
            writer.load_state_dict(f["writer"])
            writer.eval().requires_grad_(False)
        idx, mask = sequences(bank, LIBERO_10, args.E, 8, random.Random(0))
        T = min(args.T, idx.shape[1] - 1)
        W0 = memory.init_state(args.E, requires_grad=True)
        W, Ws = dict(W0), dict(W0)
        n0 = norm(memory.init_state(1))
        taus = [0.0, 0.3, 0.6, 0.9]
        rows = []

        def write(state, e_prev, a_prev, e_now):
            if mode == "horsea":
                st = horsea_write(memory, writer, flow, state, e_prev, a_prev, e_now, create_graph=False)
            else:
                st = memory.write(flow, state, e_prev, a_prev, create_graph=False)
            return {k: v.detach().requires_grad_(True) for k, v in st.items()}

        for t in range(T):
            e, a = bank.gather(idx[:, t])
            e_next, a_next = bank.gather(idx[:, t + 1])
            e, e_next = e.float(), e_next.float()
            W = write(W, e, a.clamp(-1, 1), e_next)
            Ws = write(Ws, e, torch.roll(a.clamp(-1, 1), 1, 0), e_next)
            with torch.no_grad():
                g = torch.Generator(device=dev).manual_seed(t)
                z = torch.randn(args.E, flow.chunk, flow.adim, device=dev, generator=g)
                dv, dvs = {}, {}
                for tau in taus:
                    tt = torch.full((args.E,), tau, device=dev)
                    v0 = memory.field(flow, W0, e_next)(z, tt)
                    v1 = memory.field(flow, W, e_next)(z, tt)
                    v2 = memory.field(flow, Ws, e_next)(z, tt)
                    dv[tau] = ((v1 - v0).norm() / v0.norm()).item()
                    dvs[tau] = ((v1 - v2).norm() / v0.norm()).item()
                losses = {}
                for name, st in [("W0", W0), ("written", W), ("shuffled", Ws)]:
                    torch.manual_seed(1000 + t)
                    losses[name] = sum(memory.outer_loss(flow, st, e_next, a_next).item() for _ in range(4)) / 4
            dW = torch.sqrt(sum(((W[k] - W0[k]).float() ** 2).sum() for k in W)).item() / (n0.item() * args.E ** 0.5)
            rows.append({"t": t + 1, "dW_rel": round(dW, 4), "dv_rel": {k: round(v, 4) for k, v in dv.items()},
                         "dv_vs_shuffled": {k: round(v, 4) for k, v in dvs.items()},
                         "loss_next": {k: round(v, 5) for k, v in losses.items()}})
        m = lambda key, sub=None: sum((r[key][sub] if sub is not None else r[key]) for r in rows) / len(rows)
        res[mode] = {"per_step": rows, "mean": {
            "dW_rel": round(m("dW_rel"), 4),
            "dv_rel": {tau: round(m("dv_rel", tau), 4) for tau in taus},
            "dv_vs_shuffled": {tau: round(m("dv_vs_shuffled", tau), 4) for tau in taus},
            "loss_next": {k: round(m("loss_next", k), 5) for k in ("W0", "written", "shuffled")}}}
        print(mode, json.dumps(res[mode]["mean"]), flush=True)
    out = os.path.join(EXP, "history2" + args.tag, "audit.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
