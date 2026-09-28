"""Horsea learned experience writer (protocol v2, sec. 6) and the matched TTT history baseline.

Horsea keeps F-write's external velocity reader (fast MLP on frozen theta_0 decoder features,
centred: empty memory = base) but replaces the expert FM target with a learned target:
    u_phi(e_i, z_k, t_k, c_i) = v_theta(z_k, t_k, c_i) + delta_phi(LN h(z_k, t_k, c_i), t_k, enc(e_i))
    L_write(W; H) = mean_(i,k) || r_W(z_k, t_k, c_i) - delta_phi(...) ||^2      (v_theta cancels)
    W+ = W - eta * grad_W L_write     (n_inner steps; differentiable during meta-training)
    min_{phi, W0, eta}  FM loss of v_theta + r_{W+} on verified corrections of the NEXT attempt
                        + lambda_old * FM loss on other training tasks' demos with W+ active
Experience e_i (R0: no success/reward input): real-instruction features c_i, the commanded executed
prefix (normalized), observed per-step proprio change over that prefix, execution mask.
Probes (z_k, t_k): denoising states of the recorded sampling path at solver steps K_PROBE; for the
theta_0-collected offline data this path is also the frozen-reference probe rule.
Future labels are used only in the outer loss (never in history, writer targets or probes).

TTT baseline (--arm ttt2): the same data and outer objective, but its native history write (KV
binding on internal DiT tokens of each experience's context + commanded chunk), n_inner writes.

Information-matched TTT (--arm ttt2_info): the same TTT2 write, but the written context also carries the
experience Horsea's writer sees (commanded prefix, observed proprio change, mask) through a zero-initialised
adapter: encm + A(cmd, dprop, mask). --arm ttt2_info_dphi: outer-objective ablation, the same arm trained
with Horsea's outer loss d_Phi(sampled action, corrective label) instead of the FM loss.

    python -m horsea.writer train --arm horsea --out experiments/protocol_v2/writer/horsea_s0
    python -m horsea.writer train --arm ttt2   --out experiments/protocol_v2/writer/ttt2_s0
"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, load_policy
from horsea.manifest import WRITER_DEV, WRITER_TRAIN
from horsea.memory import build_memory
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR

# self-play data directory; SELFPLAY_DATA selects a regenerated version (e.g. after the P1 recorder fix)
DATA = os.environ.get("SELFPLAY_DATA", os.path.join(EXP, "protocol_v2", "selfplay", "data"))
K_PROBE = (0, 3, 6, 9)
EXEC = 8
DPROP_SCALE = 50.0  # eef metres / gripper qpos per step -> O(0.1..1)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def prep_episode(e, dev):
    calls, P = e["calls"], e["proprio"].float()
    N = len(calls)
    cmd, dprop, mask = torch.zeros(N, EXEC, 7), torch.zeros(N, EXEC, 5), torch.zeros(N, EXEC)
    for i, c in enumerate(calls):
        s, n = c["step"], c["n_exec"]
        cmd[i, :n] = c["chunk"][:n].float()
        dprop[i, :n] = (P[s + 1:s + 1 + n] - P[s:s + n]) * DPROP_SCALE
        mask[i, :n] = 1
    return {"encm": torch.stack([c["encm"] for c in calls]).to(dev),
            "label": torch.stack([c["label"] for c in calls]).float().to(dev),  # outer loss only
            "chunk": torch.stack([c["chunk"] for c in calls]).float().to(dev),  # own commanded chunk
            "probes": torch.stack([c["probes"] for c in calls]).float().to(dev),
            "cmd": cmd.to(dev), "dprop": dprop.to(dev), "mask": mask.to(dev),
            "success": e["success"], "n": N}


def load_groups(tasks, shifts, dev):
    groups = []
    for t in tasks:
        for f in sorted(glob.glob(os.path.join(DATA, f"t{t}_*.pt"))):
            d = torch.load(f, map_location="cpu", weights_only=False)
            if shifts and d["shift"] not in shifts:
                continue
            eps = [prep_episode(e, dev) for e in d["episodes"] if len(e["calls"]) > 0]
            if len(eps) >= 2:
                groups.append({"task": d["task"], "shift": d["shift"], "eps": eps})
    return groups


# ---------------------------------------------------------------------------------------------
# writer u_phi
# ---------------------------------------------------------------------------------------------
def time_feat(t, n=8):
    f = torch.exp(torch.linspace(0, math.log(100.0), n, device=t.device))
    return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], -1)


class Writer(nn.Module):
    def __init__(self, D=256, adim=7, chunk=16, emb=64, hid=256, ablate=None):
        super().__init__()
        self.D, self.chunk, self.ablate = D, chunk, ablate  # ablate: None | "no_action" | "no_outcome"
        din = D + EXEC * 7 + EXEC * 5 + EXEC
        self.enc = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, emb))
        self.head = nn.Sequential(nn.Linear(D + emb + 16 + chunk, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                  nn.Linear(hid, adim))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()
        self.register_buffer("pos", torch.eye(chunk))

    def embed(self, encm, cmd, dprop, mask):
        c = F.layer_norm(encm.float().mean(1), (self.D,))
        if self.ablate == "no_action":   # executed commands removed from the writer's input
            cmd = torch.zeros_like(cmd)
        elif self.ablate == "no_outcome":  # observed consequences removed from the writer's input
            dprop = torch.zeros_like(dprop)
        return self.enc(torch.cat([c, cmd.flatten(1), dprop.flatten(1), mask], -1))

    def forward(self, H, t, e_emb):
        """H (R, chunk, D) layer-normed frozen features; t (R,); e_emb (R, emb) -> delta (R, chunk, adim)."""
        R = H.shape[0]
        x = torch.cat([H, e_emb[:, None].expand(R, self.chunk, -1), time_feat(t)[:, None].expand(R, self.chunk, -1),
                       self.pos[None].expand(R, -1, -1)], -1)
        return self.head(x)


# ---------------------------------------------------------------------------------------------
# meta-batch assembly and the inner/outer step
# ---------------------------------------------------------------------------------------------
def sample_rows(ep, n, gen=None):
    return torch.randint(ep["n"], (n,), generator=gen).tolist()


def write_rows(ep, n):
    """History rows: (call index, probe index) pairs -> z, t, encm, experience features, own chunk.
    Never returns correction labels (they belong to the outer objective only)."""
    idx = sample_rows(ep, n)
    k = torch.randint(len(K_PROBE), (n,)).tolist()
    dev = ep["encm"].device
    z = torch.stack([ep["probes"][i, kk] for i, kk in zip(idx, k)])
    t = torch.tensor([K_PROBE[kk] / 10.0 for kk in k], device=dev)
    return z, t, ep["encm"][idx].float(), ep["cmd"][idx], ep["dprop"][idx], ep["mask"][idx], ep["chunk"][idx]


def future_rows(ep, n):
    idx = sample_rows(ep, n)
    return ep["encm"][idx].float(), ep["label"][idx]


class InfoAdapter(nn.Module):
    """Experience features -> additive offset on every encoder token of the written context (zero at init)."""

    def __init__(self, D=256, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EXEC * 7 + EXEC * 5 + EXEC, hid), nn.GELU(), nn.Linear(hid, D))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, encm, cmd, dprop, mask):
        return encm + self.net(torch.cat([cmd.flatten(1), dprop.flatten(1), mask], -1))[:, None, :]


INFO_ARMS = ("ttt2_info", "ttt2_info_dphi")


class Meta:
    def __init__(self, arm, dev, flow, base_flow, n_inner=2, lam_old=0.5, pw=24, po=32, pold=16, ablate=None):
        self.arm, self.dev, self.flow = arm, dev, flow
        self.n_inner, self.lam_old, self.pw, self.po, self.pold = n_inner, lam_old, pw, po, pold
        self.memory = build_memory({"horsea": "fmw", "fwrite_selfimit": "fmw", "res_selfimit": "res",
                                    "ttt2_info": "ttt2", "ttt2_info_dphi": "ttt2"}.get(arm, arm)).to(dev)
        if arm in ("horsea", "fwrite_selfimit") and base_flow is not flow:
            # frozen theta_0 features when the long memory differs from theta_0 (after consolidation);
            # when theta == theta_0 one decoder pass gives both v_theta and the features
            self.memory.__dict__["feature_flow"] = base_flow
        self.writer = Writer(ablate=ablate).to(dev) if arm == "horsea" else (InfoAdapter().to(dev) if arm in INFO_ARMS else None)
        self.phi = None
        if arm == "ttt2_info_dphi":
            from horsea.phi import Phi
            self.phi = Phi(device=dev)

    def params(self):
        ps = list(self.memory.parameters())
        return ps + (list(self.writer.parameters()) if self.writer is not None else [])

    def write(self, state, rows, E, create_graph):
        """One inner write on E episodes' history rows (contiguous per episode)."""
        z, t, encm, cmd, dprop, mask, chunk = rows
        m = self.memory
        if self.arm == "horsea":
            W0 = m.init_state(E)
            with torch.no_grad():
                _, h = m._hidden(self.flow, z, t, encm)
            r = m._delta(state, W0, h, E)  # (E, R*chunk, adim)
            H = F.layer_norm(h.transpose(0, 1), (m.D,)).detach()
            d = self.writer(H, t, self.writer.embed(encm, cmd, dprop, mask))
            loss = ((r - d.reshape(E, -1, d.shape[-1])) ** 2).sum(-1).mean(-1).sum()
            return m.inner_step(state, loss, create_graph)
        if self.arm in ("fwrite_selfimit", "res_selfimit"):  # native write on its own executed chunks
            return m.write(self.flow, state, encm, chunk, create_graph)
        if self.arm in INFO_ARMS:  # same write, context carries the experience (action + consequence)
            return m.write(self.flow, state, self.writer(encm, cmd, dprop, mask), chunk, create_graph)
        # ttt2: native history write of each experience (its context + its own commanded chunk)
        return m.write(self.flow, state, encm, chunk, create_graph)

    def adapt(self, hist_eps, create_graph, n_inner=None):
        E = len(hist_eps)
        state = self.memory.init_state(E, requires_grad=not create_graph)
        for _ in range(self.n_inner if n_inner is None else n_inner):
            rows = [write_rows(ep, self.pw) for ep in hist_eps]
            cat = [torch.cat([r[j] for r in rows]) for j in range(7)]
            state = self.write(state, cat, E, create_graph)
        return state

    def outer(self, state, fut_eps, old_rows=None):
        rows = [future_rows(ep, self.po) for ep in fut_eps]
        encm = torch.cat([r[0] for r in rows])
        lab = torch.cat([r[1] for r in rows])
        if self.arm == "ttt2_info_dphi":  # Horsea's outer objective on the adapted sampler's action
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                loss = self.phi.dist(self.memory.sample(self.flow, state, encm), lab.clamp(-1, 1)).mean()
        else:
            loss = self.memory.outer_loss(self.flow, state, encm, lab)
        out = {"future": loss}
        if old_rows is not None and self.lam_old > 0:
            out["old"] = self.memory.outer_loss(self.flow, state, *old_rows)
        return out


def set_lr_cap(memory, mult):
    """Raise the inner-lr ceiling (buffers lr_max_k = 3 x init by default) to mult x init."""
    for k, v in memory.log_lr.items():
        getattr(memory, f"lr_max_{k}").mul_(mult / 3.0)


def old_batch(rbank, tasks_excl, E, n, dev, train_tasks):
    encs, acts = [], []
    for ex in tasks_excl:
        pool = [t for t in train_tasks if t != ex]
        for _ in range(n):
            t = random.choice(pool)
            d = random.randrange(rbank.n_demos(t))
            e, a = rbank.demo(t, d)
            j = random.randrange(len(e))
            encs.append(e[j:j + 1])
            acts.append(a[j:j + 1])
    return torch.cat(encs).float(), torch.cat(acts)


# ---------------------------------------------------------------------------------------------
# offline dev evaluation: future-label FM loss with no write / matched / mismatched history
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def fm_eval(meta, state, eps, draws=4):
    tot = 0.0
    for _ in range(draws):
        torch.manual_seed(random.randrange(1 << 30))
        tot += meta.outer(state, eps)["future"].item()
    return tot / draws


def offline_eval(meta, groups, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    res = {"none": [], "matched": [], "mismatched": []}
    by_task = {}
    for g in groups:
        by_task.setdefault(g["task"], []).append(g)
    for g in groups:
        h, f = g["eps"][0], g["eps"][1]
        others = [o for o in by_task[g["task"]] if o["shift"] != g["shift"]]
        s0 = meta.memory.init_state(1)
        res["none"].append(fm_eval(meta, s0, [f]))
        with torch.enable_grad():
            s1 = meta.adapt([h], create_graph=False)
        res["matched"].append(fm_eval(meta, s1, [f]))
        if others:
            with torch.enable_grad():
                s2 = meta.adapt([random.choice(others)["eps"][0]], create_graph=False)
            res["mismatched"].append(fm_eval(meta, s2, [f]))
    out = {k: sum(v) / max(1, len(v)) for k, v in res.items()}
    out["per_shift"] = {}
    for g, a, b in zip(groups, res["none"], res["matched"]):
        out["per_shift"].setdefault(g["shift"], []).append(round((a - b) / a, 4))
    out["per_shift"] = {k: round(sum(v) / len(v), 4) for k, v in out["per_shift"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "offline_eval"])
    ap.add_argument("--arm", default="horsea", choices=["horsea", "ttt2", "fwrite_selfimit", "res_selfimit", "ttt2_info", "ttt2_info_dphi"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--lam_old", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_shifts", nargs="*", default=None)
    ap.add_argument("--lr_cap", type=float, default=3.0, help="max inner lr as a multiple of its init (Memory default 3)")
    ap.add_argument("--ablate", default=None, choices=[None, "no_action", "no_outcome"])
    ap.add_argument("--dev_shifts", nargs="*", default=None)
    ap.add_argument("--train_tasks", type=int, nargs="*", default=None, help="override (smoke tests)")
    ap.add_argument("--dev_tasks", type=int, nargs="*", default=None, help="override (smoke tests)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(EXP, "protocol_v2", "writer", f"{args.arm}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    base, _ = load_policy(BASE_CKPT, dev)
    base.requires_grad_(False)
    flow = Flow(base)
    meta = Meta(args.arm, dev, flow, flow, n_inner=args.n_inner, lam_old=args.lam_old, ablate=args.ablate)
    set_lr_cap(meta.memory, args.lr_cap)
    train = load_groups(args.train_tasks or WRITER_TRAIN, args.train_shifts, dev)
    dev_groups = load_groups(args.dev_tasks or WRITER_DEV, args.dev_shifts, dev)
    print(f"{args.arm}: {len(train)} train groups, {len(dev_groups)} dev groups; fast weights "
          f"{meta.memory.fast_numel()}, writer params {sum(p.numel() for p in meta.writer.parameters()) if meta.writer else 0}",
          flush=True)
    if args.cmd == "offline_eval":
        sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
        meta.memory.load_state_dict(sd["memory"])
        if meta.writer is not None:
            meta.writer.load_state_dict(sd["writer"])
        print(json.dumps(offline_eval(meta, dev_groups)), flush=True)
        return
    rbank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    opt = torch.optim.Adam(meta.params(), lr=args.lr)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        gs = random.sample(train, min(args.E, len(train)))
        hist, fut = [], []
        for g in gs:
            a, b = random.sample(range(len(g["eps"])), 2)
            hist.append(g["eps"][a])
            fut.append(g["eps"][b])
        state = meta.adapt(hist, create_graph=True)
        old = old_batch(rbank, [g["task"] for g in gs], args.E, meta.pold, dev, WRITER_TRAIN) if args.lam_old > 0 else None
        L = meta.outer(state, fut, old)
        loss = L["future"] + args.lam_old * L.get("old", torch.zeros((), device=dev))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(meta.params(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                s0 = meta.memory.init_state(args.E)
                base_future = meta.outer(s0, fut)["future"].item()
            rec = {"step": step, "future": round(L["future"].item(), 5), "future_no_write": round(base_future, 5),
                   "old": round(L.get("old", torch.zeros(())).item(), 5), "min": round((time.time() - t0) / 60, 1),
                   "lrs": {k: round(v.item(), 5) for k, v in meta.memory.lrs().items()}}
            print(json.dumps(rec), flush=True)
            log.append(rec)
        if step % args.eval_every == 0 or step == args.steps:
            ev = offline_eval(meta, dev_groups)
            ev["step"] = step
            print("DEV", json.dumps(ev), flush=True)
            log.append({"dev": ev})
            torch.save({"memory": meta.memory.state_dict(),
                        "writer": meta.writer.state_dict() if meta.writer else None,
                        "args": vars(args), "step": step}, os.path.join(out, "last.pt"))
            json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
