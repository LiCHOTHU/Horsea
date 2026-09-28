"""Four fast-memory arms on a frozen flow-matching policy.

Every arm shares the same skeleton so the comparison isolates *where memory acts* and *what
the write objective is*:

  * fast weights W, batched over episodes (leading dim E), initialised from a meta-learned W0;
  * one inner gradient step per experience (= one demonstration), with meta-learned
    per-tensor inner learning rates;
  * meta-training differentiates the query loss through the writes (second order);
  * ~16-19k fast parameters per episode.

Arms
  ttt  RoboTTT-style internal TTT: a TTT-MLP layer after each DiT decoder block, KV-binding
       write on the block's tokens, tanh-gated output (not centred, as published).
  kv   The proposal (Sec. 4-5): external associative MLP f_W(q) = tanh(qA+b)B + d0 written by
       cue->content association, read through a W0-centred velocity reader at every solver step.
  fmw  FM-write velocity memory: a correction field g_W on the frozen decoder's final hidden
       tokens, written with the flow-matching loss itself, W0-centred.
  res  Fast final-action residual: a correction to the finished action chunk from
       (observation, base action), written by action regression, W0-centred.

Shapes: encm (rows, L, D) encoder summaries; actions (rows, chunk, adim) normalized to [-1, 1].
A "rows" batch is always E episodes x N rows per episode, flattened episode-major.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def bmlp(x, W, pre, act):
    """Two-layer MLP with per-episode weights. x: (E, N, din)."""
    h = act(torch.baddbmm(W[pre + "b1"].unsqueeze(1), x, W[pre + "W1"]))
    return torch.baddbmm(W[pre + "b2"].unsqueeze(1), h, W[pre + "W2"])


def mlp(sizes, act=nn.GELU, last_scale=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    with torch.no_grad():
        layers[-1].weight.mul_(last_scale)
        layers[-1].bias.mul_(last_scale)
    return nn.Sequential(*layers)


def per_episode(x, E):
    """(E*N, ...) -> (E, N*..., last)  for token-wise fast nets."""
    return x.reshape(E, -1, x.shape[-1])


class Memory(nn.Module):
    """Common skeleton: W0 (slow), inner learning rates (slow), write / read (arm-specific)."""

    name = "base"
    centred = True

    def __init__(self):
        super().__init__()
        self.W0 = nn.ParameterDict()
        self.log_lr = nn.ParameterDict()

    def _add_fast(self, key, tensor, lr):
        self.W0[key] = nn.Parameter(tensor)
        self.log_lr[key] = nn.Parameter(torch.tensor(math.log(lr)))
        # meta-learning may raise a write step 3x above its calibrated stable value, not more
        self.register_buffer(f"lr_max_{key}", torch.tensor(3.0 * lr), persistent=False)

    def fast_numel(self):
        return sum(p.numel() for p in self.W0.values())

    def lrs(self):
        return {k: torch.minimum(v.exp().clamp_min(1e-5), getattr(self, f"lr_max_{k}"))
                for k, v in self.log_lr.items()}

    def init_state(self, E, requires_grad=False):
        """Fresh memory for E episodes. requires_grad=True for deployment-time writes."""
        state = {k: v.unsqueeze(0).expand(E, *v.shape) for k, v in self.W0.items()}
        if requires_grad:
            state = {k: v.detach().clone().requires_grad_(True) for k, v in state.items()}
        return state

    def inner_step(self, state, loss, create_graph, scale=None):
        """scale: optional per-episode write strength (E,), e.g. a learned forget/write gate."""
        keys = list(state)
        grads = torch.autograd.grad(loss, [state[k] for k in keys], create_graph=create_graph)
        lrs = self.lrs()
        if scale is not None:
            grads = [g * scale.view(-1, *([1] * (g.dim() - 1))) for g in grads]
        new = {k: state[k] - lrs[k] * g for k, g in zip(keys, grads)}
        if not create_graph:
            new = {k: v.detach().requires_grad_(True) for k, v in new.items()}
        return new

    # --- to implement ---------------------------------------------------------------------
    def write(self, flow, state, encm, act, create_graph):
        """One write of one experience per episode. encm (E*F, L, D), act (E*F, C, A)."""
        raise NotImplementedError

    def field(self, flow, state, encm):
        """Adapted velocity field v(z, t) on rows encm (E*N, L, D)."""
        raise NotImplementedError

    # --- shared ---------------------------------------------------------------------------
    def sample(self, flow, state, encm, noise=None):
        return torch.clamp(flow.euler(self.field(flow, state, encm), encm, noise), -1, 1)

    def outer_loss(self, flow, state, encm, act):
        """Meta-training objective on query rows: FM loss of the adapted field."""
        return flow.fm_loss(encm, act, field=self.field(flow, state, encm))


# ---------------------------------------------------------------------------------------
# (1) internal TTT, RoboTTT-style
# ---------------------------------------------------------------------------------------
class TTTMemory(Memory):
    """TTT-MLP layer after every DiT decoder block (Sun et al. 2024; RoboTTT 2026).

    Fast model per layer: f(x) = x + LN(MLP_W(x)) on d-dim projections; inner loss
    ||f(theta_K x) - theta_V x||^2 on the block's tokens; output tanh(alpha) * theta_O f(theta_Q x)
    added to the residual stream. Writes use noised support actions at random flow times
    (RoboTTT's sequence action forcing). Reads apply the fast weights without updating them.
    """

    name = "ttt"
    centred = False

    def __init__(self, D=256, d=32, hidden=64, n_layers=4, lr=0.03, gate_init=1e-3, n_noise=1,
                 prenorm=False, write_gate=False):
        super().__init__()
        self.n_layers, self.d, self.n_noise, self.D = n_layers, d, n_noise, D
        # RoboTTT reference (lucidrains/robo_ttt): RMSNorm before the qkv projections, and a learned
        # data-dependent gate on each write (their "learned forget"), sigmoid(MLP(mean token)).
        self.prenorm = prenorm
        self.write_gate = nn.ModuleList([nn.Linear(D, 1) for _ in range(n_layers)]) if write_gate else None
        if write_gate:
            for g in self.write_gate:
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, 2.0)  # writes start ~open (sigmoid(2) = 0.88)
        self.q = nn.ModuleList([nn.Linear(D, d) for _ in range(n_layers)])
        self.k = nn.ModuleList([nn.Linear(D, d) for _ in range(n_layers)])
        self.v = nn.ModuleList([nn.Linear(D, d) for _ in range(n_layers)])
        self.o = nn.ModuleList([nn.Linear(d, D) for _ in range(n_layers)])
        self.alpha = nn.Parameter(torch.full((n_layers, D), gate_init))
        self.ln_w = nn.Parameter(torch.ones(n_layers, d))
        self.ln_b = nn.Parameter(torch.zeros(n_layers, d))
        for l in range(n_layers):
            self._add_fast(f"l{l}_W1", torch.randn(d, hidden) / math.sqrt(d), lr)
            self._add_fast(f"l{l}_b1", torch.zeros(hidden), lr)
            self._add_fast(f"l{l}_W2", torch.randn(hidden, d) / math.sqrt(hidden), lr)
            self._add_fast(f"l{l}_b2", torch.zeros(d), lr)

    def _f(self, l, W, x):
        y = bmlp(x, W, f"l{l}_", F.gelu)
        return x + F.layer_norm(y, (self.d,), self.ln_w[l], self.ln_b[l])

    def _tokens(self, x, E):
        # (chunk, E*N, D) -> (E, N*chunk, D), rows of the same episode contiguous
        C, R, D = x.shape
        return x.transpose(0, 1).reshape(E, (R // E) * C, D)

    def _untokens(self, y, C):
        E, NC, D = y.shape
        return y.reshape(E * (NC // C), C, D).transpose(0, 1)

    def _proj(self, lin, X):
        return F.layer_norm(lin(X), (self.d,))  # unit-scale keys/values/queries: bounded write steps

    def _pre(self, X):
        return F.rms_norm(X, (self.D,)) if self.prenorm else X

    def _ttt_apply(self, l, W, x, E):
        X = self._pre(self._tokens(x, E))
        out = self.o[l](self._f(l, W, self._proj(self.q[l], X)))
        return x + torch.tanh(self.alpha[l]) * self._untokens(out, x.shape[0])

    def write(self, flow, state, encm, act, create_graph):
        E = next(iter(state.values())).shape[0]
        if self.n_noise > 1:
            encm = encm.repeat_interleave(self.n_noise, 0)
            act = act.repeat_interleave(self.n_noise, 0)
        x1 = torch.clamp(act, -1, 1)
        t = flow.sample_t(x1.shape[0], x1.device)
        psi, _ = flow.interp(torch.randn_like(x1), x1, t)
        new = dict(state)

        def hook(l, x):
            X = self._pre(self._tokens(x if create_graph else x.detach(), E))
            Wl = {k: new[k] for k in new if k.startswith(f"l{l}_")}
            loss = ((self._f(l, Wl, self._proj(self.k[l], X)) - self._proj(self.v[l], X)) ** 2).sum(-1).mean(-1).sum()
            scale = torch.sigmoid(self.write_gate[l](X.mean(1))).squeeze(-1) if self.write_gate is not None else None
            new.update(self.inner_step(Wl, loss, create_graph, scale))  # update ...
            return self._ttt_apply(l, new, x, E)  # ... then apply (RoboTTT's update-then-apply)

        with torch.enable_grad():
            flow.decode(psi, t, encm, layer_hook=hook)
        return new

    def field(self, flow, state, encm):
        E = next(iter(state.values())).shape[0]
        return lambda z, t: flow.decode(z, t, encm, layer_hook=lambda l, x: self._ttt_apply(l, state, x, E))


# ---------------------------------------------------------------------------------------
# (2) the proposal: external associative memory + centred velocity reader
# ---------------------------------------------------------------------------------------
def dct_basis(n, k):
    """First k DCT-II basis vectors over n steps, orthonormal, shape (n, k)."""
    t = torch.arange(n, dtype=torch.float32)
    B = torch.stack([torch.cos(math.pi * (t + 0.5) * j / n) for j in range(k)], dim=1)
    return B / B.norm(dim=0, keepdim=True)


class KVMemory(Memory):
    """Proposal eqs. (3)-(8). Cue: learned encoder of the observation summary. Content: a fixed
    event encoder -- the lowest n_dct DCT coefficients of the normalized action chunk (action
    chunks have a power-law spectrum, so a few low bands carry most of the chunk). Read: query
    from (z_t, t, observation), recall m = f_W(q), velocity r = R(.., m) - R(.., f_W0(q))."""

    name = "kv"

    def __init__(self, D=256, chunk=16, adim=7, dk=32, hidden=256, n_dct=4, lr=0.03, reader=512):
        super().__init__()
        self.chunk, self.adim, self.dk = chunk, adim, dk
        dv = n_dct * adim
        self.register_buffer("dct", dct_basis(chunk, n_dct))
        self.cue = mlp([D, 256, dk])
        self.query = mlp([chunk * adim + 2 * D, 256, dk])
        # small (not zero) output init: centring already gives r = 0 at W0, and a zero layer
        # would block every gradient into the memory at the first step
        self.reader = mlp([chunk * adim + 2 * D + dv, reader, reader, chunk * adim], last_scale=0.01)
        self._add_fast("W1", torch.randn(dk, hidden) / math.sqrt(dk), lr)  # A
        self._add_fast("b1", torch.zeros(hidden), lr)  # b
        self._add_fast("W2", torch.randn(hidden, dv) * 0.01, lr)  # B
        self._add_fast("b2", torch.zeros(dv), lr)  # d0

    def content(self, act):
        return torch.einsum("tc,btj->bcj", self.dct, torch.clamp(act, -1, 1)).flatten(1)

    def write(self, flow, state, encm, act, create_graph):
        E = next(iter(state.values())).shape[0]
        with torch.enable_grad():
            q = per_episode(F.layer_norm(self.cue(encm[:, -1]), (self.dk,)), E)
            y = per_episode(self.content(act), E).detach()  # sg(y_e)
            loss = 0.5 * ((bmlp(q, state, "", torch.tanh) - y) ** 2).sum(-1).mean(-1).sum()
            return self.inner_step(state, loss, create_graph)

    def field(self, flow, state, encm):
        E = next(iter(state.values())).shape[0]
        W0 = self.init_state(E)
        p = encm[:, -1]

        def v(z, t):
            te = flow.vnet.time_net(t)
            ctx = torch.cat([z.flatten(1), te, p], dim=-1)
            q = per_episode(F.layer_norm(self.query(ctx), (self.dk,)), E)
            m = bmlp(q, state, "", torch.tanh).flatten(0, 1)
            m0 = bmlp(q, W0, "", torch.tanh).flatten(0, 1)
            r = self.reader(torch.cat([ctx, m], -1)) - self.reader(torch.cat([ctx, m0], -1))
            return flow.decode(z, t, encm) + r.view(-1, self.chunk, self.adim)

        return v


# ---------------------------------------------------------------------------------------
# (3) FM-write velocity memory
# ---------------------------------------------------------------------------------------
class FMWriteMemory(Memory):
    """Fast weights parameterise a correction field g_W on the frozen decoder's final hidden
    tokens (so similarity between states is the policy's own), written with the FM loss on
    the experience's actions: W <- W - eta grad ||v_base + g_W - g_W0 - (x1 - (1-s)x0)||^2."""

    name = "fmw"

    def __init__(self, D=256, adim=7, hidden=64, lr=0.015, n_noise=2):
        super().__init__()
        self.D, self.n_noise = D, n_noise
        self._add_fast("W1", torch.randn(D, hidden) / math.sqrt(D), lr)
        self._add_fast("b1", torch.zeros(hidden), lr)
        self._add_fast("W2", torch.randn(hidden, adim) * 0.01, lr)
        self._add_fast("b2", torch.zeros(adim), lr)
        # Optional frozen feature source (a Flow of the ORIGINAL base): the memory's key space then
        # never drifts when the long-term policy is consolidated; the velocity still comes from the
        # current policy. Not a module attribute, so it is never saved or moved with the memory.
        self.__dict__["feature_flow"] = None

    def _hidden(self, flow, z, t, encm):
        """(v from the current policy, h from the feature source)."""
        if self.feature_flow is None:
            return flow.decode(z, t, encm, return_hidden=True)
        v = flow.decode(z, t, encm)
        with torch.no_grad():
            _, h = self.feature_flow.decode(z, t, encm, return_hidden=True)
        return v, h

    def _delta(self, state, W0, h, E):
        H = per_episode(F.layer_norm(h.transpose(0, 1), (self.D,)), E)  # (E, N*chunk, D)
        return bmlp(H, state, "", F.gelu) - bmlp(H, W0, "", F.gelu)

    def write(self, flow, state, encm, act, create_graph):
        E = next(iter(state.values())).shape[0]
        encm = encm.repeat_interleave(self.n_noise, 0)
        x1 = torch.clamp(act, -1, 1).repeat_interleave(self.n_noise, 0)
        t = flow.sample_t(x1.shape[0], x1.device)
        psi, u = flow.interp(torch.randn_like(x1), x1, t)
        with torch.no_grad():
            v_b, h = self._hidden(flow, psi, t, encm)
        W0 = self.init_state(E)
        with torch.enable_grad():
            v = v_b.reshape(E, -1, v_b.shape[-1]) + self._delta(state, W0, h, E)
            loss = ((v - u.reshape(E, -1, u.shape[-1])) ** 2).sum(-1).mean(-1).sum()
            return self.inner_step(state, loss, create_graph)

    def field(self, flow, state, encm):
        E = next(iter(state.values())).shape[0]
        W0 = self.init_state(E)

        def v(z, t):
            v_b, h = self._hidden(flow, z, t, encm)
            return v_b + self._delta(state, W0, h, E).reshape(v_b.shape)

        return v


# ---------------------------------------------------------------------------------------
# (4) fast final-action residual
# ---------------------------------------------------------------------------------------
class ResidualMemory(Memory):
    """a = a_base + rho_W(obs, a_base) - rho_W0(obs, a_base), per action token. Written by
    action regression onto the demonstrated chunk; acts once, after denoising."""

    name = "res"

    def __init__(self, D=256, chunk=16, adim=7, hidden=64, lr=0.05):
        super().__init__()
        self.D, self.chunk = D, chunk
        self.register_buffer("pos", torch.eye(chunk))
        din = D + adim + chunk
        self._add_fast("W1", torch.randn(din, hidden) / math.sqrt(din), lr)
        self._add_fast("b1", torch.zeros(hidden), lr)
        self._add_fast("W2", torch.randn(hidden, adim) * 0.01, lr)
        self._add_fast("b2", torch.zeros(adim), lr)

    def _delta(self, state, encm, a_base, E):
        R = a_base.shape[0]
        p = F.layer_norm(encm[:, -1], (self.D,)).unsqueeze(1).expand(R, self.chunk, self.D)
        x = torch.cat([p, a_base, self.pos.unsqueeze(0).expand(R, -1, -1)], -1)
        X = x.reshape(E, -1, x.shape[-1])
        W0 = self.init_state(E)
        return (bmlp(X, state, "", F.gelu) - bmlp(X, W0, "", F.gelu)).reshape(a_base.shape)

    def write(self, flow, state, encm, act, create_graph):
        E = next(iter(state.values())).shape[0]
        with torch.no_grad():
            a_base = flow.sample(encm)
        with torch.enable_grad():
            a = a_base + self._delta(state, encm, a_base, E)
            loss = ((a - torch.clamp(act, -1, 1)) ** 2).sum(-1).mean(-1).reshape(E, -1).mean(-1).sum()
            return self.inner_step(state, loss, create_graph)

    def sample(self, flow, state, encm, noise=None):
        E = next(iter(state.values())).shape[0]
        a_base = flow.sample(encm, noise)
        return torch.clamp(a_base + self._delta(state, encm, a_base, E), -1, 1)

    def field(self, flow, state, encm):
        raise TypeError("the residual arm acts on the finished action, not on the velocity")

    def outer_loss(self, flow, state, encm, act):
        E = next(iter(state.values())).shape[0]
        with torch.no_grad():
            a_base = flow.sample(encm)
        a = a_base + self._delta(state, encm, a_base, E)
        return ((a - torch.clamp(act, -1, 1)) ** 2).mean()


class TTT2Memory(TTTMemory):
    """TTT aligned with the RoboTTT reference code: pre-RMSNorm + learned write gate."""

    name = "ttt2"

    def __init__(self, **kw):
        super().__init__(prenorm=True, write_gate=True, **kw)


ARMS = {"ttt": TTTMemory, "ttt2": TTT2Memory, "kv": KVMemory, "fmw": FMWriteMemory, "res": ResidualMemory}


def build_memory(arm, **kw):
    return ARMS[arm](**kw)
