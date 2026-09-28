"""RoboTTT-comparable experiment on LIBERO-10, fair across memory methods.

RoboTTT protocol (Jiang et al. 2026, Sec. 3.2 / 4), reproduced for every method:
  * post-train on the downstream tasks' expert demos (here all 10 LIBERO-10 tasks, demo data only),
    decoder + the method's memory jointly, on whole demo sequences: one timestep = one executed
    chunk (every 8 frames); at timestep t the flow-matching loss on the demo action uses the fast
    weights written by timesteps < t; truncated BPTT; W0 and the write rule are learned by that loss;
  * test on the same tasks from new start states: every episode starts from W0, and the robot writes
    ONLY its own history (its observations, its own executed chunks); no expert data at test time.
Methods (same backbone, data, steps, optimizer; ~17k fast weights each):
  plain  -- decoder only, single-step context (RoboTTT's GR00T baseline)
  ttt    -- TTT2 layers inside the DiT: native KV-binding write of (context, action tokens) [RoboTTT]
  fwrite -- external velocity memory, FM write of (context, executed chunk)
  res    -- final-action residual memory, regression write of (context, executed chunk)
  horsea -- external velocity memory, learned experience writer: the write for step t-1 uses its
            context, its executed chunk and the observed consequence (context change to step t);
            target = learned velocity correction at reference probes (noise, t=0); no action target
  energy2       -- the validated energy Horsea (protocol-v2 stage 2): energy memory + experience writer
                   (context, executed chunk, context change) + learned bounded final-action solver,
                   outer loss d_Phi(refined action, demo action). Decoder not trained (no gradient path).
  ttt_info      -- TTT2 given Horsea's information: the completed interaction t-1 -> t is written as
                   (c_{t-1} + A(c_t - c_{t-1}), a_{t-1}), A a zero-initialised linear adapter. FM outer loss.
  ttt_info_dphi -- outer-objective ablation: ttt_info trained with Horsea's outer objective
                   d_Phi(sampled action, demo action), differentiating through the adapted sampler.
Controls: every memory method is also evaluated with writes disabled (memory stays at W0).

    python -m horsea.history2 train --mode horsea
    python -m horsea.history2 eval  --mode horsea [--nowrite]
"""
import argparse
import json
import math
import os
import random
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.base import Flow, decoder_parameters, load_policy
from horsea.finetune import clone_trainable
from horsea.history import sequences
from horsea.memory import build_memory
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, LIBERO_10
from horsea.rollout import make_runner, run_task
from horsea.writer import time_feat

ARM = {"ttt": "ttt2", "fwrite": "fmw", "res": "res", "horsea": "fmw", "ttt_info": "ttt2", "ttt_info_dphi": "ttt2"}
INFO = ("ttt_info", "ttt_info_dphi")
ENERGY_ITERS, ENERGY_STEP = 3, 0.05


class HistEnergyWriter(nn.Module):
    """Target cost y_phi(e, candidate) for LIBERO histories: e = (context, executed chunk, context change)."""

    def __init__(self, D=256, emb=64, hid=256):
        super().__init__()
        from horsea.phi import LATENT
        self.D = D
        self.enc = nn.Sequential(nn.Linear(2 * D + EXEC * 7, hid), nn.GELU(), nn.Linear(hid, emb))
        self.head = nn.Sequential(nn.Linear(emb + 2 * LATENT, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                  nn.Linear(hid, 1))

    def forward(self, phi, c_prev, c_now, a_exec, cand):
        c0 = F.layer_norm(c_prev.float().mean(1), (self.D,))
        dc = F.layer_norm((c_now.float() - c_prev.float()).mean(1), (self.D,))
        e = self.enc(torch.cat([c0, dc, a_exec[:, :EXEC].flatten(1)], -1))
        pc, pa = phi(cand), phi(a_exec)
        return self.head(torch.cat([e, pc, pc - pa], -1)).squeeze(-1)


def energy_candidates(chunk):
    import math as _m
    alts = [chunk]
    for deg in (-60, -40, -20, 20, 40, 60):
        a_ = _m.radians(deg)
        r = chunk.clone()
        r[..., 0] = _m.cos(a_) * chunk[..., 0] - _m.sin(a_) * chunk[..., 1]
        r[..., 1] = _m.sin(a_) * chunk[..., 0] + _m.cos(a_) * chunk[..., 1]
        alts.append(r.clamp(-1, 1))
    f = chunk.clone()
    f[..., 6] = -chunk[..., 6]
    alts.append(f)
    g = torch.Generator(device=chunk.device).manual_seed(0)
    alts += [(chunk + 0.3 * torch.randn(chunk.shape, device=chunk.device, generator=g)).clamp(-1, 1) for _ in range(2)]
    return torch.stack(alts)  # (n, R, 16, 7)


GENERIC = {"on": False}  # energy2 with generic candidates: executed chunk + 9 random perturbations


def generic_candidates(chunk, E, n=10, sigma=0.3):
    g = torch.Generator(device=chunk.device).manual_seed(0)
    nz = torch.randn((n - 1, chunk.shape[0] // E) + tuple(chunk.shape[1:]), device=chunk.device, generator=g).repeat(1, E, 1, 1)
    return torch.cat([chunk[None], (chunk[None] + sigma * nz).clamp(-1, 1)])  # same pattern per episode block


def energy_write(mem, writer, phi, state, c_prev, a_prev, c_now, create_graph):
    E_ = next(iter(state.values())).shape[0]
    cands = generic_candidates(a_prev, E_) if GENERIC["on"] else energy_candidates(a_prev)
    n, R = cands.shape[:2]
    cand = cands.transpose(0, 1).reshape(R * n, *a_prev.shape[1:])
    rep = lambda x: x.repeat_interleave(n, 0)
    with torch.enable_grad():
        y = writer(phi, rep(c_prev), rep(c_now), rep(a_prev), cand)
        en = mem.energy(phi, state, cand, rep(c_prev))
        E = next(iter(state.values())).shape[0]
        loss = ((en - y) ** 2).reshape(E, -1).mean(-1).sum()
        return mem.inner_step(state, loss, create_graph)


def energy_sample(mem, phi, flow, state, c, create_graph=False):
    """Ordinary rollout, then bounded final-action descent on the written energy."""
    with torch.no_grad():
        a = flow.sample(c)
    a = a.detach().requires_grad_(True)
    with torch.enable_grad():
        for _ in range(ENERGY_ITERS):
            g, = torch.autograd.grad(mem.energy(phi, state, a, c).sum(), a, create_graph=create_graph)
            a = (a - ENERGY_STEP * torch.tanh(g / ENERGY_STEP)).clamp(-1, 1)
    return a
EXEC = 8
OUT = os.path.join(EXP, "history2")
PHI = {"phi": None}


class HistoryWriter(nn.Module):
    """u_phi for demo/rollout histories: experience = (context c_{t-1}, executed chunk prefix a_{t-1},
    consequence c_t - c_{t-1} in the policy's own feature space). Output: velocity correction per token."""

    def __init__(self, D=256, adim=7, chunk=16, emb=64, hid=256):
        super().__init__()
        self.D, self.chunk = D, chunk
        self.enc = nn.Sequential(nn.Linear(2 * D + EXEC * adim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                 nn.Linear(hid, emb))
        self.head = nn.Sequential(nn.Linear(D + emb + 16 + chunk, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                  nn.Linear(hid, adim))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()
        self.register_buffer("pos", torch.eye(chunk))

    def forward(self, H, t, c_prev, c_now, a_prev):
        c0 = F.layer_norm(c_prev.float().mean(1), (self.D,))
        dc = F.layer_norm((c_now.float() - c_prev.float()).mean(1), (self.D,))
        e = self.enc(torch.cat([c0, dc, a_prev[:, :EXEC].flatten(1)], -1))
        R = H.shape[0]
        x = torch.cat([H, e[:, None].expand(R, self.chunk, -1), time_feat(t)[:, None].expand(R, self.chunk, -1),
                       self.pos[None].expand(R, -1, -1)], -1)
        return self.head(x)


class InfoAdapter(nn.Module):
    """Gives TTT the observed consequence: write context = c_prev + A(c_now - c_prev), A zero-initialised."""

    def __init__(self, D=256):
        super().__init__()
        self.A = nn.Linear(D, D)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.A.bias)

    def forward(self, c_prev, c_now):
        return c_prev + self.A(c_now.float() - c_prev.float())


def energy2_adapt(mem, writer, phi, hist, E, create_graph, pw=16, n_inner=2):
    """Validated batch writer (horsea.energy.adapt): re-adapt from W0 with n_inner writes, each on up to
    pw interactions sampled from the episode's history [(c_prev, a_prev, c_now), ...] (tensors (E, ...)).
    The number of write steps is bounded however long the history is (sequential writes diverged)."""
    state = mem.init_state(E, requires_grad=not create_graph)
    for _ in range(n_inner):
        sel = sorted(random.sample(range(len(hist)), min(len(hist), pw)))
        cat = [torch.stack([hist[i][j] for i in sel], 1).flatten(0, 1) for j in range(3)]  # episode-major rows
        state = energy_write(mem, writer, phi, state, cat[0], cat[1], cat[2], create_graph)
        if not create_graph:
            state = {k: v.detach().requires_grad_(True) for k, v in state.items()}
    return state


class Energy2(nn.Module):
    """Writer + learned solver of the validated energy Horsea (horsea.energy.Solver, final-action mode)."""

    def __init__(self, max_step=0.05, n_iter=3):
        super().__init__()
        from horsea.energy import Solver
        self.writer = HistEnergyWriter()
        self.solver = Solver(K=10, n_iter=n_iter, max_step=max_step)

    def act(self, mem, phi, flow, state, c, create_graph):
        eps = torch.randn(c.shape[0], flow.chunk, flow.adim, device=c.device)
        with torch.enable_grad():
            return self.solver.solve(flow, lambda a: mem.energy(phi, state, a, c), c, eps, "final", create_graph)


MULTI_PROBE = {"on": False}
K_PROBE = (0, 3, 6, 9)


def horsea_write(memory, writer, flow, state, c_prev, a_prev, c_now, create_graph, gen=None):
    """Learned write of the completed interaction (c_prev, a_prev -> c_now). Probes: reference noise at
    t=0, or (MULTI_PROBE) the frozen reference sampler's denoising states at solver steps K_PROBE."""
    E = c_prev.shape[0]
    z = torch.randn(E, flow.chunk, flow.adim, device=c_prev.device, generator=gen)
    t = torch.zeros(E, device=c_prev.device)
    if MULTI_PROBE["on"]:
        zs, ts, zz, tt = [], [], z, t.clone()
        with torch.no_grad():
            for k in range(flow.n_steps):
                if k in K_PROBE:
                    zs.append(zz.clone())
                    ts.append(tt.clone())
                zz = zz + (1.0 / flow.n_steps) * flow.decode(zz, tt, c_now)
                tt = tt + 1.0 / flow.n_steps
        P = len(K_PROBE)  # episode-major rows: (e0 p0..p3, e1 p0..p3, ...)
        z = torch.stack(zs, 1).reshape(E * P, flow.chunk, flow.adim)
        t = torch.stack(ts, 1).reshape(E * P)
        c_prev, a_prev, c_now = (x.repeat_interleave(P, 0) for x in (c_prev, a_prev, c_now))
    with torch.enable_grad():
        _, h = memory._hidden(flow, z, t, c_now)  # probe the field where the NEXT decision is made
        W0 = memory.init_state(E)
        r = memory._delta(state, W0, h, E)
        H = F.layer_norm(h.transpose(0, 1), (memory.D,))
        if not create_graph:
            H = H.detach()
        d = writer(H, t, c_prev, c_now, a_prev)
        loss = ((r - d.reshape(E, -1, d.shape[-1])) ** 2).sum(-1).mean(-1).sum()
        return memory.inner_step(state, loss, create_graph)


def rowwise_loss(mode, memory, flow, state, e, a):
    x1 = a.clamp(-1, 1)
    if mode == "res":
        with torch.no_grad():
            a_base = flow.sample(e)
        pred = a_base + memory._delta(state, e, a_base, e.shape[0])
        return ((pred - x1) ** 2).mean((1, 2))
    x0 = torch.randn_like(x1)
    t = flow.sample_t(x1.shape[0], x1.device)
    psi, u = flow.interp(x0, x1, t)
    v = flow.decode(psi, t, e) if memory is None else memory.field(flow, state, e)(psi, t)
    return ((v - u) ** 2).mean((1, 2))


def keep_active(new, old, m):
    """Per-episode select: written state for active rows (m > 0), previous state for padded rows."""
    act = m > 0
    return {k: torch.where(act.view(-1, *([1] * (v.dim() - 1))), v, old[k]) for k, v in new.items()}


def build(mode, dev):
    if mode in ("energy", "energy2"):
        from horsea.energy import EnergyMemory
        return EnergyMemory().to(dev), (HistEnergyWriter() if mode == "energy" else Energy2()).to(dev)
    memory = build_memory(ARM[mode]).to(dev) if mode in ARM else None
    writer = HistoryWriter().to(dev) if mode == "horsea" else (InfoAdapter().to(dev) if mode in INFO else None)
    return memory, writer


def train(args):
    dev = args.device
    base, _ = load_policy(args.ckpt, dev)
    if args.init_vnet:
        base.velocity_net.load_state_dict(torch.load(args.init_vnet, map_location=dev, weights_only=False)["vnet"])
    student = clone_trainable(base)
    student.train()
    flow = Flow(student)
    bank = FeatureBank(os.path.join(FEAT_DIR, f"{args.suite}.pt"), dev)
    memory, writer = build(args.mode, dev)
    if args.lr_cap is not None and memory is not None:
        from horsea.writer import set_lr_cap
        set_lr_cap(memory, args.lr_cap)
    groups = [{"params": decoder_parameters(student), "lr": args.lr_dec}]
    slow = (list(memory.parameters()) if memory is not None else []) + (list(writer.parameters()) if writer else [])
    if slow:
        groups.append({"params": slow, "lr": args.lr_mem})
    opt = torch.optim.AdamW(groups, weight_decay=1e-6)
    out = os.path.join(OUT + args.tag, args.mode)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    log = open(os.path.join(out, "log.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps):
        sched = min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for i, (g, lr) in enumerate(zip(opt.param_groups, [args.lr_dec, args.lr_mem])):
            frozen = i == 0 and step < args.freeze_dec_steps and memory is not None
            g["lr"] = 0.0 if frozen else lr * sched
        idx, mask = sequences(bank, args.tasks, args.E, args.stride, rng)
        E, L = idx.shape
        opt.zero_grad(set_to_none=True)
        state = memory.init_state(E) if memory is not None else None
        total, seg, n_tok = 0.0, 0.0, mask.sum()
        prev, hist = None, []
        for t in range(L):
            e, a = bank.gather(idx[:, t])
            e = e.float()
            m = mask[:, t]
            old = state  # rows whose demo has ended (padding) keep their memory: no write, exact-zero loss
            if args.mode == "horsea" and prev is not None:  # write the completed interaction t-1 -> t
                state = horsea_write(memory, writer, flow, state, prev[0], prev[1], e, create_graph=True)
            if args.mode in INFO and prev is not None:  # write the completed interaction t-1 -> t
                state = memory.write(flow, state, writer(prev[0], e), prev[1], create_graph=True)
            if args.mode == "energy2":
                if prev is not None:
                    hist.append((prev[0], prev[1], e))
                    state = energy2_adapt(memory, writer.writer, PHI["phi"], hist, E, create_graph=True)
                per = PHI["phi"].dist(writer.act(memory, PHI["phi"], flow, state, e, True), a.clamp(-1, 1))
            elif args.mode == "ttt_info_dphi":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.MATH):
                    per = PHI["phi"].dist(memory.sample(flow, state, e), a.clamp(-1, 1))
            elif args.mode == "energy":
                if prev is not None:
                    state = energy_write(memory, writer, PHI["phi"], state, prev[0], prev[1], e, create_graph=True)
                with torch.no_grad():
                    a_own = flow.sample(e)  # the policy's own proposal for this step
                # contrastive: the demo's next action should cost less than the policy's own sample
                per = F.softplus(memory.energy(PHI["phi"], state, a.clamp(-1, 1), e)
                                 - memory.energy(PHI["phi"], state, a_own, e) + 0.1)
            else:
                per = rowwise_loss(args.mode, memory, flow, state, e, a)
            if args.mode in ("ttt", "fwrite", "res"):  # native write of (context, executed chunk)
                state = memory.write(flow, state, e, a, create_graph=True)
            if state is not None and state is not old:
                state = keep_active(state, old, m)
            prev = (e, a.clamp(-1, 1))
            seg = seg + torch.where(m > 0, per, torch.zeros_like(per)).sum() / n_tok
            if memory is not None and ((t + 1) % args.tbptt == 0 or t == L - 1) and torch.is_tensor(seg) and seg.requires_grad:
                seg.backward()
                total += seg.item()
                seg = 0.0
                state = {k: v.detach().requires_grad_(True) for k, v in state.items()}
                prev = (prev[0].detach(), prev[1].detach())
        if memory is None:
            seg.backward()
            total = seg.item()
        params = [p for g in opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):  # skip non-finite updates instead of poisoning the weights
            opt.zero_grad(set_to_none=True)
            print(f"step {step + 1}: non-finite gradient, update skipped", flush=True)
            continue
        opt.step()
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "loss": round(total, 5), "L": L, "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save({"vnet": student.velocity_net.state_dict(), "memory": memory.state_dict() if memory else None,
                "writer": writer.state_dict() if writer else None, "args": vars(args)}, os.path.join(out, "final.pt"))
    print("done", flush=True)


def install(policy, flow, mode, memory, writer, nowrite, variant="persist"):
    """Receding horizon (generate 16, execute 8). Each env row keeps its own fast weights, reset at
    episode start. Writes use only the robot's OWN completed interactions (after acting).
    variant: persist    -- accumulate writes over the episode (normal RoboTTT-style test)
             last_only  -- before each decision reset to W0 and write only the previous step
             nowrite    -- read frozen W0, never write
             bypass     -- no memory at all (backbone only)
             mismatched -- accumulate writes, but each written action comes from another episode (row)"""
    if nowrite:
        variant = "nowrite"
    ctx = {"state": None, "prev": None, "hist": []}
    orig_reset = policy.reset

    def reset(self):
        orig_reset()
        ctx["state"], ctx["prev"], ctx["hist"] = None, None, []

    def write(state, e_prev, a_prev, e_now):
        if variant == "mismatched":
            a_prev = torch.roll(a_prev, 1, 0)  # another episode's action, same context/outcome
        if mode == "energy":
            st = energy_write(memory, writer, PHI["phi"], state, e_prev, a_prev, e_now, create_graph=False)
            return {k: v.detach().requires_grad_(True) for k, v in st.items()}
        if mode == "energy2":  # batch re-adaptation from W0 on the accumulated own history
            ctx["hist"] = ([] if variant == "last_only" else ctx["hist"]) + [(e_prev, a_prev, e_now)]
            return energy2_adapt(memory, writer.writer, PHI["phi"], ctx["hist"], e_now.shape[0], create_graph=False)
        if mode in INFO:
            with torch.enable_grad():
                st = memory.write(flow, state, writer(e_prev, e_now), a_prev, create_graph=False)
        elif mode == "horsea":
            st = horsea_write(memory, writer, flow, state, e_prev, a_prev, e_now, create_graph=False)
        else:
            st = memory.write(flow, state, e_prev, a_prev, create_graph=False)
        return {k: v.detach().requires_grad_(True) for k, v in st.items()}

    def sample_actions(self, data):
        with torch.no_grad():
            encm = flow.encode(data).float()
            if memory is None or variant == "bypass":
                return flow.sample(encm).cpu().numpy()
            if ctx["state"] is None:
                ctx["state"] = memory.init_state(encm.shape[0], requires_grad=True)
            if variant != "nowrite" and ctx["prev"] is not None:
                base_state = memory.init_state(encm.shape[0], requires_grad=True) if variant == "last_only" else ctx["state"]
                ctx["state"] = write(base_state, ctx["prev"][0], ctx["prev"][1], encm)
            if mode == "energy2":
                a = writer.act(memory, PHI["phi"], flow, ctx["state"], encm, False).detach()
            elif mode == "energy":
                a = energy_sample(memory, PHI["phi"], flow, ctx["state"], encm).detach()
            else:
                a = memory.sample(flow, ctx["state"], encm)
            ctx["prev"] = (encm, a)
            return a.cpu().numpy()

    policy.reset = types.MethodType(reset, policy)
    policy.sample_actions = types.MethodType(sample_actions, policy)
    policy.temporal_agg, policy.action_horizon, policy.batch_size, policy.action_queue = False, EXEC, None, None
    return policy


def evaluate(args):
    dev = args.device
    policy, sd = load_policy(args.ckpt, dev)
    f = torch.load(os.path.join(OUT + args.tag, args.mode, "final.pt"), map_location=dev, weights_only=False)
    policy.velocity_net.load_state_dict(f["vnet"])
    GENERIC["on"] = bool(f["args"].get("generic_cands", False))
    policy.eval()
    memory, writer = build(args.mode, dev)
    if memory is not None:
        memory.load_state_dict(f["memory"])
        cap = f["args"].get("lr_cap")
        if cap is not None:
            from horsea.writer import set_lr_cap
            set_lr_cap(memory, cap)
        memory.eval().requires_grad_(False)
    if writer is not None:
        writer.load_state_dict(f["writer"])
        writer.eval().requires_grad_(False)
    if args.mode == "energy2" and args.energy_step is not None:
        writer.solver.max_step = args.energy_step
    flow = Flow(policy)
    tag = args.mode + ("_nowrite" if args.nowrite else ("" if args.variant == "persist" else "_" + args.variant)) + getattr(args, "tag_suffix", "") + args.eval_tag
    out = os.path.join(OUT + args.tag, "eval", tag)
    os.makedirs(out, exist_ok=True)
    runner = make_runner(sd["config"]["task"]["shape_meta"], args.suite, args.n, args.par, args.offset, dev)
    install(policy, flow, args.mode, memory, writer, args.nowrite, args.variant)
    for task in args.tasks:
        path = os.path.join(out, f"t{task}.json")
        if os.path.exists(path):
            continue
        res = run_task(runner, policy, task)
        res.update({"mode": args.mode, "nowrite": args.nowrite, "suite": args.suite})
        json.dump(res, open(path, "w"))
        print(f"[{tag}] {args.suite} task {task}: {res['rate']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--mode", required=True, choices=["plain", "ttt", "fwrite", "res", "horsea", "energy", "energy2", "ttt_info", "ttt_info_dphi"])
    ap.add_argument("--nowrite", action="store_true")
    ap.add_argument("--variant", default="persist", choices=["persist", "last_only", "nowrite", "bypass", "mismatched"])
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", type=int, nargs="+", default=LIBERO_10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--stride", type=int, default=EXEC)
    ap.add_argument("--tbptt", type=int, default=16)
    ap.add_argument("--lr_dec", type=float, default=5e-5)
    ap.add_argument("--lr_mem", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=30, help="start states 30.. (disjoint from demo inits)")
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--init_vnet", default=None, help="start from this post-trained velocity net (shared by all methods)")
    ap.add_argument("--freeze_dec_steps", type=int, default=0,
                    help="RoboTTT two-stage recipe: train only memory/writer (decoder lr 0) for the first N steps")
    ap.add_argument("--energy_step", type=float, default=None, help="override the energy solver step at test time")
    ap.add_argument("--multi_probe", action="store_true", help="Horsea writes fitted at several denoising times")
    ap.add_argument("--tag", default="", help="output sub-directory suffix")
    ap.add_argument("--lr_cap", type=float, default=None, help="max inner lr as a multiple of its init (default 3)")
    ap.add_argument("--generic_cands", action="store_true", help="energy2: generic write candidates (no rotation prior)")
    ap.add_argument("--eval_tag", default="", help="eval sub-directory suffix (e.g. _dev / _test)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.multi_probe:
        MULTI_PROBE["on"] = True
    GENERIC["on"] = args.generic_cands  # eval: overridden from the checkpoint's training args
    if args.mode in ("energy", "energy2", "ttt_info_dphi"):
        from horsea.phi import Phi
        PHI["phi"] = Phi(device=args.device)
    if args.mode == "energy2" and args.cmd == "eval" and args.energy_step is not None:
        args.tag_suffix = f"_step{args.energy_step}"
    if args.mode == "energy":
        if args.energy_step is not None:
            global ENERGY_STEP
            ENERGY_STEP = args.energy_step
            args.tag_suffix = f"_step{args.energy_step}"
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
