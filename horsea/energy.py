"""Horsea-E prototype: short memory = a learned action-preference cost; actions come from optimizing
the denoising path under that cost, anchored to the policy's own transitions.

  E_{psi,W}(a, c) = g_W(f(a,c)) - g_{W0}(f(a,c)),  f = [Phi(a), P_psi(c)]       (E_{W0} == 0: reset = base)
  write (from a completed interaction e = context, executed chunk, observed consequence; no labels):
      W <- W - eta * grad_W  sum_j ( E_W(a~_j, c_e) - y_phi(e, a~_j) )^2,  a~ = {executed chunk, perturbations}
  path solve (joint): z_0 = eps fixed; z_{1:K} initialised with the ordinary rollout z_{k+1} = T(z_k);
      J = E_W(z_K, c) + 1/(2 lambda) sum_k ||z_{k+1} - T(z_k)||^2 / dt,   n_iter unrolled steps z <- z - beta grad J
  path solve (final): only z_K is optimised (earlier states fixed at the ordinary rollout)
  meta-training: outer loss d_Phi(a_adapted, corrective target for the SAME noise) on the next attempt's
      decisions, differentiated through the solver and the write (phi, psi, W0, eta, lambda, beta).
Data: hidden-shift self-rollouts on the training tasks (horsea.selfplay); corrective target = g^-1 of
the robot's own sample from the same noise (verified oracle), used only in the outer loss.

    python -m horsea.energy train --solve joint --out experiments/energy/joint_s0
    python -m horsea.energy train --solve final --out experiments/energy/final_s0
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.manifest import WRITER_DEV, WRITER_TRAIN
from horsea.memory import Memory, bmlp, per_episode
from horsea.paths import BASE_CKPT, EXP
from horsea.phi import LATENT, Phi
from horsea.writer import DPROP_SCALE, EXEC, load_groups


BALANCED = {"on": False}


class EnergyMemory(Memory):
    name = "energy"

    def __init__(self, D=256, dc=64, hid=96, lr=0.05):
        super().__init__()
        self.D = D
        self.cproj = nn.Linear(D, dc)  # slow psi
        din = LATENT + dc
        self._add_fast("W1", torch.randn(din, hid) / math.sqrt(din), lr)
        self._add_fast("b1", torch.zeros(hid), lr)
        self._add_fast("W2", torch.randn(hid, 1) * 0.1, lr)
        self._add_fast("b2", torch.zeros(1), lr)

    def feats(self, phi, a, c):
        cc = self.cproj(F.layer_norm(c.float().mean(1), (self.D,)))
        return torch.cat([phi(a), cc], -1)

    nohist = False  # control: a static (slow, non-centred) energy trained with the same supervision, no writes

    def energy(self, phi, state, a, c):
        """Rows grouped per episode (E episodes); returns (R,)."""
        E = next(iter(state.values())).shape[0]
        f = per_episode(self.feats(phi, a, c), E)
        W0 = self.init_state(E)
        if self.nohist:
            return bmlp(f, W0, "", F.gelu).reshape(-1)
        return (bmlp(f, state, "", F.gelu) - bmlp(f, W0, "", F.gelu)).reshape(-1)


class EnergyWriter(nn.Module):
    """y_phi(e, candidate): learned target cost of a candidate action given the experience."""

    def __init__(self, D=256, emb=64, hid=256):
        super().__init__()
        self.D = D
        din = D + EXEC * 7 + EXEC * 5 + EXEC
        self.enc = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, emb))
        self.head = nn.Sequential(nn.Linear(emb + 2 * LATENT, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                  nn.Linear(hid, 1))

    def forward(self, phi, encm, cmd, dprop, mask, cand, a_exec, shuffle=False):
        c = F.layer_norm(encm.float().mean(1), (self.D,))
        e = self.enc(torch.cat([c, cmd.flatten(1), dprop.flatten(1), mask], -1))
        pc, pa = phi(cand), phi(a_exec)
        return self.head(torch.cat([e, pc, pc - pa], -1)).squeeze(-1)


class Solver(nn.Module):
    def __init__(self, K=10, n_iter=3, max_step=None, kind="grad"):
        super().__init__()
        self.K, self.n_iter, self.max_step, self.kind = K, n_iter, max_step, kind
        self.trace = None  # set to [] to log total J / saturation per step (diagnostics)
        self.log_lam = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(0.05)))

    def step(self, d):
        """Bounded per-element update (keeps many-iteration solves stable)."""
        return d if self.max_step is None else self.max_step * torch.tanh(d / self.max_step)

    def rollout(self, flow, c, eps):
        dt = 1.0 / self.K
        zs, z, t = [], eps, torch.zeros(eps.shape[0], device=eps.device)
        for _ in range(self.K):
            z = z + dt * flow.decode(z, t, c)
            t = t + dt
            zs.append(z)
        return zs

    def T(self, flow, z, k, c):
        dt = 1.0 / self.K
        return z + dt * flow.decode(z, torch.full((z.shape[0],), k * dt, device=z.device), c)

    def solve(self, flow, energy_fn, c, eps, mode, create_graph):
        """mode: none | final | joint. Returns the final action chunk."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):  # the fused attention kernels have no double-backward
            return self._solve(flow, energy_fn, c, eps, mode, create_graph)

    def _solve(self, flow, energy_fn, c, eps, mode, create_graph):
        with torch.no_grad():
            base = self.rollout(flow, c, eps)
        if mode == "none":
            return base[-1].clamp(-1, 1)
        lam, beta, dt = self.log_lam.exp(), self.log_beta.exp(), 1.0 / self.K
        if mode == "final":
            prevT = self.T(flow, base[-2], self.K - 1, c).detach()
            zK = base[-1].detach().requires_grad_(True)
            Jf = lambda z: energy_fn(z) + ((z - prevT) ** 2).flatten(1).sum(1) / (2 * lam * dt)
            for _ in range(self.n_iter):
                if self.kind == "prox":
                    # linearise E at zK; solve the proximal step with the quadratic anchor EXACTLY:
                    # argmin_a <gE, a> + ||a - zK||^2/(2 beta) + ||a - prevT||^2/(2 lambda dt)
                    gE, = torch.autograd.grad(energy_fn(zK).sum(), zK, create_graph=create_graph)
                    c1, c2 = 1.0 / beta, 1.0 / (lam * dt)
                    d = (c1 * zK + c2 * prevT - gE) / (c1 + c2) - zK
                else:
                    g, = torch.autograd.grad(Jf(zK).sum(), zK, create_graph=create_graph)
                    d = -beta * g
                step = self.step(d)
                if self.trace is not None:
                    with torch.no_grad():
                        self.trace.append({"J_before": Jf(zK).mean().item(), "J_after": Jf(zK + step).mean().item(),
                                           "saturated": (d.abs() > (self.max_step or 1e9)).float().mean().item(),
                                           "move": step.abs().mean().item()})
                zK = zK + step
            return zK.clamp(-1, 1)
        Z = [z.detach().requires_grad_(True) for z in base]
        for _ in range(self.n_iter):
            pen = 0.0
            prev = eps
            for k in range(self.K):
                pen = pen + ((Z[k] - self.T(flow, prev, k, c)) ** 2).flatten(1).sum(1)
                prev = Z[k]
            J = energy_fn(Z[-1]) + pen / (2 * lam * dt)
            gs = torch.autograd.grad(J.sum(), Z, create_graph=create_graph)
            Z = [z - self.step(beta * g) for z, g in zip(Z, gs)]
        return Z[-1].clamp(-1, 1)


class Model(nn.Module):
    def __init__(self, K=10, n_iter=3, n_cand=4, sigma=0.3, structured=False, max_step=None, solver="grad"):
        super().__init__()
        self.structured = structured
        self.mem = EnergyMemory()
        self.writer = EnergyWriter()
        self.solver = Solver(K, n_iter, max_step, solver)
        self.gate = nn.Linear(64, 2)  # (alpha, beta) from the experience embedding (used by the gated writer)
        with torch.no_grad():
            self.gate.bias.fill_(2.0)  # start near alpha=beta~0.88
        self.n_cand, self.sigma = n_cand, sigma

    def write(self, phi, state, rows, create_graph, shuffle=False, scale=None):
        encm, cmd, dprop, mask, chunk = rows
        if shuffle:  # control: action-outcome pairing broken (executed chunks from other experiences)
            perm = torch.randperm(len(chunk), device=chunk.device)
            cmd, chunk = cmd[perm], chunk[perm]
        R = chunk.shape[0]
        if self.structured:  # executed chunk + xy-rotated versions + gripper-flipped + 2 random perturbations
            alts = [chunk]
            for deg in (-60, -40, -20, 20, 40, 60):
                a_ = math.radians(deg)
                r = chunk.clone()
                r[..., 0] = math.cos(a_) * chunk[..., 0] - math.sin(a_) * chunk[..., 1]
                r[..., 1] = math.sin(a_) * chunk[..., 0] + math.cos(a_) * chunk[..., 1]
                alts.append(r.clamp(-1, 1))
            f = chunk.clone()
            f[..., 6] = -chunk[..., 6]
            alts.append(f)
            # random perturbations: same noise pattern within every episode's block of rows, so an episode's
            # candidates never depend on which other episodes share its batch
            E_ = next(iter(state.values())).shape[0]
            per = R // E_
            g = torch.Generator(device=chunk.device).manual_seed(0)
            for _ in range(2):
                nz = torch.randn((per,) + tuple(chunk.shape[1:]), device=chunk.device, generator=g).repeat(E_, 1, 1)
                alts.append((chunk + self.sigma * nz).clamp(-1, 1))
            cands = torch.stack(alts)
        else:
            g = torch.Generator(device=chunk.device).manual_seed(0)
            noise = torch.randn(self.n_cand - 1, *chunk.shape, device=chunk.device, generator=g) * self.sigma
            cands = torch.cat([chunk[None], (chunk[None] + noise).clamp(-1, 1)])  # (n_cand, R, 16, 7)
        n_cand = cands.shape[0]
        # episode-major ordering expected by per_episode: interleave candidates within each row group
        E = next(iter(state.values())).shape[0]
        cand = cands.transpose(0, 1).reshape(R * n_cand, *chunk.shape[1:])
        rep = lambda x: x.repeat_interleave(n_cand, 0)
        y = self.writer(phi, rep(encm), rep(cmd), rep(dprop), rep(mask), cand, rep(chunk))
        with torch.enable_grad():
            en = self.mem.energy(phi, state, cand, rep(encm))
            loss = ((en - y) ** 2).reshape(E, -1).mean(-1).sum()
            return self.mem.inner_step(state, loss, create_graph, scale)


def hist_rows(ep, n):
    i = torch.randint(ep["n"], (n,)).tolist()
    return ep["encm"][i].float(), ep["cmd"][i], ep["dprop"][i], ep["mask"][i], ep["chunk"][i]


TARGET = {"kind": "corrective"}


def fut_rows(ep, n):
    i = torch.randint(ep["n"], (n,)).tolist()
    tgt = ep["label"][i] if TARGET["kind"] == "corrective" else ep["chunk"][i]  # own executed chunk (self-imitation)
    return ep["encm"][i].float(), ep["probes"][i, 0], tgt  # context, starting noise, target


WRITER = {"kind": "batch", "chunk": 4}  # batch: 2 steps on random history rows | seq | gated (Gated-DeltaNet-style)


def adapt(model, phi, hist, create_graph, shuffle=False, pw=16, n_inner=2):
    E = len(hist)
    state = model.mem.init_state(E, requires_grad=not create_graph)
    if model.mem.nohist:
        return state
    if WRITER["kind"] in ("seq", "gated"):
        return adapt_seq(model, phi, hist, create_graph, shuffle, gated=WRITER["kind"] == "gated")
    WRITE_COUNT["last"] = [n_inner] * E
    for _ in range(n_inner):
        rows = [hist_rows(h, pw) for h in hist]
        cat = [torch.cat([r[j] for r in rows]) for j in range(5)]
        state = model.write(phi, state, cat, create_graph, shuffle)
        if not create_graph:
            state = {k: v.detach().requires_grad_(True) for k, v in state.items()}
    return state


WRITE_COUNT = {"last": None}  # per-episode number of (non-skipped) writes of the last adapt() call


def adapt_seq(model, phi, hist, create_graph, shuffle, gated):
    """Sequential writes in time order, one chunk of experiences per write:
        W_t = W0 + alpha_t (W_{t-1} - W0) - beta_t * eta * grad L_write      (alpha=beta=1 if not gated).
    Ragged histories: every history is processed to its own end. Chunks are padded (repeat the last row)
    and an episode whose history has ended skips the whole update (write scale 0, no decay)."""
    E = len(hist)
    ch = WRITER["chunk"]
    lens = [h["n"] for h in hist]
    dev = hist[0]["encm"].device
    state = model.mem.init_state(E, requires_grad=not create_graph)
    W0 = model.mem.init_state(E)
    writes = [0] * E
    for s0 in range(0, max(lens), ch):
        active = torch.tensor([1.0 if s0 < L else 0.0 for L in lens], device=dev)
        rows = []
        for h, L in zip(hist, lens):
            idx = [min(i, L - 1) for i in range(s0, s0 + ch)]  # padded chunk (masked out if inactive)
            rows.append((h["encm"][idx].float(), h["cmd"][idx], h["dprop"][idx], h["mask"][idx], h["chunk"][idx]))
        cat = [torch.cat([r[j] for r in rows]) for j in range(5)]
        if gated:
            enc_in = torch.cat([F.layer_norm(cat[0].mean(1), (model.writer.D,)), cat[1].flatten(1),
                                cat[2].flatten(1), cat[3]], -1)
            g = torch.sigmoid(model.gate(model.writer.enc(enc_in)).reshape(E, ch, 2).mean(1))  # (E, 2)
            alpha = active * g[:, 0] + (1 - active)          # inactive: no decay
            beta = active * g[:, 1]                           # inactive: no write
            state = {k: W0[k] + alpha.view(-1, *[1] * (v.dim() - 1)) * (v - W0[k]) for k, v in state.items()}
            state = model.write(phi, state, cat, create_graph, shuffle, scale=beta)
        else:
            state = model.write(phi, state, cat, create_graph, shuffle, scale=active)
        writes = [w + int(a) for w, a in zip(writes, active.tolist())]
        if not create_graph:
            state = {k: v.detach().requires_grad_(True) for k, v in state.items()}
    WRITE_COUNT["last"] = writes
    return state


def outer(model, phi, flow, state, fut, mode, create_graph, po=16):
    rows = [fut_rows(f, po) for f in fut]
    c = torch.cat([r[0] for r in rows])
    eps = torch.cat([r[1] for r in rows])
    lab = torch.cat([r[2] for r in rows])
    efn = lambda a: model.mem.energy(phi, state, a, c)
    a = model.solver.solve(flow, efn, c, eps, mode, create_graph)
    if BALANCED["on"]:  # each action group contributes equally (a flipped gripper must not dominate)
        grp = [slice(0, 3), slice(3, 6), slice(6, 7)]
        terms = [((a[..., gsl] - lab[..., gsl]) ** 2).mean() / (((lab[..., gsl] - lab[..., gsl].mean()) ** 2).mean() + 1e-3)
                 for gsl in grp]
        return sum(terms) / len(terms)
    return phi.dist(a, lab).mean()


def evaluate_dev(model, phi, flow, groups, mode, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    by_task = {}
    for g in groups:
        by_task.setdefault(g["task"], []).append(g)
    out = {"none": [], "matched": [], "shuffled": [], "mismatched": []}
    per_shift = {}
    for g in groups:
        h, f = g["eps"][0], g["eps"][1]
        torch.manual_seed(seed)
        s0 = model.mem.init_state(1)
        with torch.no_grad():
            torch.manual_seed(1)
            out["none"].append(outer(model, phi, flow, s0, [f], "none", False).item())
        with torch.enable_grad():
            s1 = adapt(model, phi, [h], False)
            torch.manual_seed(1)
            out["matched"].append(outer(model, phi, flow, s1, [f], mode, False).item())
            s2 = adapt(model, phi, [h], False, shuffle=True)
            torch.manual_seed(1)
            out["shuffled"].append(outer(model, phi, flow, s2, [f], mode, False).item())
            others = [o for o in by_task[g["task"]] if o["shift"] != g["shift"]]
            if others:
                s3 = adapt(model, phi, [random.choice(others)["eps"][0]], False)
                torch.manual_seed(1)
                out["mismatched"].append(outer(model, phi, flow, s3, [f], mode, False).item())
        per_shift.setdefault(g["shift"], []).append((out["none"][-1], out["matched"][-1]))
    res = {k: round(sum(v) / max(1, len(v)), 5) for k, v in out.items()}
    res["per_shift_gain"] = {s: round(sum(a - b for a, b in v) / len(v), 4) for s, v in per_shift.items()}  # none - matched
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--solve", choices=["joint", "final"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_iter", type=int, default=3)
    ap.add_argument("--solver", default="grad", choices=["grad", "prox"])
    ap.add_argument("--writer", default="batch", choices=["batch", "seq", "gated"])
    ap.add_argument("--target", default="corrective", choices=["corrective", "own"],
                    help="outer target: corrective action at the visited state, or the own executed chunk (self-imitation)")
    ap.add_argument("--nohist", action="store_true", help="control: static energy, same supervision, no history")
    ap.add_argument("--balanced", action="store_true", help="per-action-group normalized outer loss")
    ap.add_argument("--structured", action="store_true", help="structured write candidates (rotations, gripper flip)")
    ap.add_argument("--max_step", type=float, default=None, help="bounded solver step per element")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--train_shifts", nargs="*", default=None)
    ap.add_argument("--dev_shifts", nargs="*", default=None)
    ap.add_argument("--train_tasks", type=int, nargs="*", default=None)
    ap.add_argument("--dev_tasks", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    pol, _ = load_policy(BASE_CKPT, dev)
    pol.requires_grad_(False)
    flow = Flow(pol)
    phi = Phi(device=dev)
    BALANCED["on"] = args.balanced
    TARGET["kind"] = args.target
    WRITER["kind"] = args.writer
    EnergyMemory.nohist = args.nohist
    model = Model(args.K, args.n_iter, structured=args.structured, max_step=args.max_step, solver=args.solver).to(dev)
    train = load_groups(args.train_tasks or WRITER_TRAIN, args.train_shifts, dev)
    devg = load_groups(args.dev_tasks or WRITER_DEV, args.dev_shifts, dev)
    print(f"{args.solve}: {len(train)} train / {len(devg)} dev groups; fast weights {model.mem.fast_numel()}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        gs = random.sample(train, min(args.E, len(train)))
        hist, fut = [], []
        for g in gs:
            a, b = random.sample(range(len(g["eps"])), 2)
            hist.append(g["eps"][a])
            fut.append(g["eps"][b])
        state = adapt(model, phi, hist, True)
        loss = outer(model, phi, flow, state, fut, args.solve, True)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss": round(loss.item(), 5), "lam": round(model.solver.log_lam.exp().item(), 4),
                   "beta": round(model.solver.log_beta.exp().item(), 4), "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.append(rec)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate_dev(model, phi, flow, devg, args.solve)
            ev["step"] = step
            ev["sec_per_solve_batch"] = None
            print("DEV", json.dumps(ev), flush=True)
            log.append({"dev": ev})
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, os.path.join(args.out, "last.pt"))
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
