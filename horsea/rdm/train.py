"""PPO meta-training of the RDM readers on five-attempt own-rollout metaepisodes (spec sec. 5).

Policy on the augmented action space: eps ~ N(0, I) (fixed), u ~ N(mu_psi(c, H, eps), sigma^2 I) over the 8 x 7
executed prefix. Ratio = N(u; mu_new) / N(u; mu_old) with the logged eps and the exact causal history H_n.
Returns: Monte-Carlo, discount 1, over the WHOLE metaepisode (attempt resets do not cut them); baseline: critic
V(c, attempt, position, successes so far) -- training-only, identical for every variant.
theta (base encoder + DiT) is frozen and stays in eval mode (no dropout); psi = RDM reader (+ tokenizer).

    python -m horsea.rdm.train --variant reread --tasks 73 75 --iters 30 --seed 0 --out experiments/rdm/runs/reread_s0
"""
import argparse
import json
import multiprocessing
import os
import random
import time

import numpy as np
import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.paths import BASE_CKPT
from horsea.rdm.model import EXEC, N_ATT, N_DEC, VARIANTS, Value, make_model, policy_logp
from horsea.rdm.rollout import Controller, install, run_metaepisodes
from horsea.rollout import make_runner

TRAIN_STARTS = list(range(0, 30))   # 'adapt' + 'teacher' folds: training init states
EVAL_STARTS = list(range(30, 50))   # 'validation' fold: evaluation only


def returns(recs):
    """per-decision reward-to-go over the entire metaepisode (discount 1, across attempts)."""
    r = np.array([x["reward"] for x in recs], dtype=np.float32)
    return np.cumsum(r[::-1])[::-1].copy()


def build_dataset(batches):
    """batches: list of (recs [B][190], events dict (B, 190, ...)) -> flat decision tensors + event store."""
    D = {k: [] for k in ("meta", "encm", "eps", "u", "mu", "logp", "n_hist", "attempt", "dstep", "G", "nsucc")}
    events = []
    for recs, ev in batches:
        for b, rb in enumerate(recs):
            m = len(events)
            events.append({k: v[b] for k, v in ev.items()})
            per_att = np.zeros(N_ATT)
            for x in rb:
                per_att[x["attempt"]] += x["reward"]
            assert per_att.max() <= 1.0, f"more than one success reward in an attempt: {per_att}"  # R in [0, 5]
            G = returns(rb)
            ns = np.cumsum([0.0] + [x["reward"] for x in rb])[:-1]
            for i, x in enumerate(rb):
                D["meta"].append(m)
                D["encm"].append(x["encm"])
                D["eps"].append(x["eps"])
                D["u"].append(x["u"])
                D["mu"].append(x["mu"])
                D["logp"].append(x["logp"])
                D["n_hist"].append(x["n_hist"])
                D["attempt"].append(x["attempt"])
                D["dstep"].append(x["dstep"])
                D["G"].append(G[i])
                D["nsucc"].append(ns[i])
    out = {k: (torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(np.asarray(v))) for k, v in D.items()}
    return out, events


TTT_WINDOW = {"T": 8}


def recompute(model, data, events, idx, dev):
    """Current-psi mean for decisions idx with their exact causal histories (tokens recomputed, full K-step graph)."""
    metas = sorted(set(data["meta"][idx].tolist()))
    loc = {m: i for i, m in enumerate(metas)}
    tok = None
    if model.variant == "ttt_info":  # replay W_n from the causal events under the current slow weights
        ev = {k: torch.stack([events[m][k] for m in metas]).to(dev) for k in events[0]}
        nh = data["n_hist"][idx].to(dev)
        li = torch.tensor([loc[int(m)] for m in data["meta"][idx]], device=dev)
        state = model.replay_burnin(ev, li, nh, T=TTT_WINDOW["T"])
        z = model.mean(data["encm"][idx].to(dev).float(), data["eps"][idx].to(dev), state)
        return z[:, :EXEC], None
    if model.variant in ("readonce", "reread"):
        ev = {k: torch.stack([events[m][k] for m in metas]).to(dev) for k in events[0]}
        tok_all = model.event_tokens(ev)                                   # (m, 190, 9, d) with grad
        tok = tok_all[torch.tensor([loc[int(m)] for m in data["meta"][idx]], device=dev)]
    encm = data["encm"][idx].to(dev).float()
    eps = data["eps"][idx].to(dev)
    n_valid = data["n_hist"][idx].to(dev)
    z = model.mean(encm, eps, tok, n_valid)
    return z[:, :EXEC], encm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--B", type=int, default=8, help="sequences (init states) per task per iteration")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vlr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=128)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--ttt_window", type=int, default=8, help="ttt_info: writes replayed WITH gradient before each decision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    TTT_WINDOW["T"] = args.ttt_window
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, sd = load_policy(BASE_CKPT, dev)
    policy.eval()  # dropout off for rollouts and likelihood recomputation; theta frozen (requires_grad False)
    flow = Flow(policy)
    model = make_model(flow, args.variant).to(dev)
    value = Value().to(dev)
    psi = list(model.parameters())
    opt = torch.optim.Adam(psi, lr=args.lr) if psi else None
    vopt = torch.optim.Adam(value.parameters(), lr=args.vlr)
    ck = os.path.join(args.out, "last.pt")
    it0, log = 0, []
    if os.path.exists(ck):  # resume
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        if opt is not None:
            opt.load_state_dict(s["opt"])
        value.load_state_dict(s["value"])
        vopt.load_state_dict(s["vopt"])
        it0, log = s["iter"], s["log"]
    multiprocessing.set_start_method("spawn", force=True)
    runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", 2, 2, 0, dev, horizon=300)
    ctrl = Controller(model, flow, args.sigma, dev)
    install(policy, flow, ctrl)
    theta0 = {n: p.detach().clone() for n, p in policy.velocity_net.named_parameters()}
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    for it in range(it0, args.iters):
        t0 = time.time()
        batches, succ_all = [], []
        model.eval()
        for task in args.tasks:
            ids = random.sample(TRAIN_STARTS, args.B)
            recs, ev, succ, _ = run_metaepisodes(runner, policy, flow, ctrl, task, ids, log=say)
            batches.append((recs, ev))
            succ_all.append(succ)
        succ = np.concatenate(succ_all)                                    # (n_seq, 5)
        data, events = build_dataset(batches)
        rec = {"iter": it + 1, "S": [round(float(x), 3) for x in succ.mean(0)], "S2_5": round(float(succ[:, 1:].mean()), 3),
               "collect_min": round((time.time() - t0) / 60, 1)}
        # ---- critic + advantages ------------------------------------------------------------------------
        with torch.no_grad():
            V = value(data["encm"].to(dev).float(), data["attempt"].to(dev), data["dstep"].to(dev) / N_DEC,
                      data["nsucc"].to(dev)).cpu()
        adv = data["G"] - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        n = len(adv)
        kls, clipfrac, ratio0 = [], [], None
        t1 = time.time()
        for ep in range(args.epochs):
            perm = torch.randperm(n)
            stop = False
            for s0 in range(0, n, args.mb):
                idx = perm[s0:s0 + args.mb]
                # critic
                vpred = value(data["encm"][idx].to(dev).float(), data["attempt"][idx].to(dev),
                              data["dstep"][idx].to(dev) / N_DEC, data["nsucc"][idx].to(dev))
                vloss = ((vpred - data["G"][idx].to(dev)) ** 2).mean()
                vopt.zero_grad(set_to_none=True)
                vloss.backward()
                vopt.step()
                if opt is None:
                    continue
                mu, _ = recompute(model, data, events, idx, dev)
                logp = policy_logp(data["u"][idx].to(dev), mu, args.sigma)
                ratio = torch.exp(logp - data["logp"][idx].to(dev))
                if ratio0 is None:  # check: before any update the recomputed likelihood equals the logged one
                    ratio0 = float((ratio.detach() - 1).abs().max())
                a = adv[idx].to(dev)
                loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(psi, 1.0)
                if not torch.isfinite(gn):
                    say("non-finite gradient, minibatch skipped")
                    continue
                opt.step()
                with torch.no_grad():
                    lr_ = logp - data["logp"][idx].to(dev)
                    kls.append(float(((torch.exp(lr_) - 1) - lr_).mean()))
                    clipfrac.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                if kls[-1] > 1.5 * args.target_kl:
                    stop = True
                    break
            if stop:
                break
        # theta must be untouched by meta-training
        drift = max(float((p - theta0[nm]).abs().max()) for nm, p in policy.velocity_net.named_parameters())
        rec.update({"update_min": round((time.time() - t1) / 60, 1), "kl": round(float(np.mean(kls)), 5) if kls else None,
                    "clipfrac": round(float(np.mean(clipfrac)), 3) if clipfrac else None,
                    "ratio0_maxdev": ratio0, "theta_drift": drift, "vloss": round(float(vloss), 4),
                    "grip_flip": grip_flip_rate(data, args.sigma)})
        say(json.dumps(rec))
        log.append(rec)
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict() if opt else None, "value": value.state_dict(), "vopt": vopt.state_dict(),
                    "iter": it + 1, "log": log, "args": vars(args)}, ck)
        json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
    say("done")


def grip_flip_rate(data, sigma):
    """Exploration audit: fraction of executed gripper commands whose sign differs from the mean's sign."""
    return round(float((data["u"][..., 6].sign() != data["mu"][..., 6].sign()).float().mean()), 4)


if __name__ == "__main__":
    main()
