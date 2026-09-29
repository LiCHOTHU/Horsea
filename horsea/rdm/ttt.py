"""TTT-info for the RDM proof of concept (spec sec. 6): the existing TTT2 fast-weight layers inside the DiT
(RoboTTT-style KV-binding write, learned write gate), given the SAME allowed information as the RDM bank and trained
with the SAME outer objective (five-attempt PPO). This is an extension, not RoboTTT's published training recipe.

Write of one real event e_n (after its prefix executed), one inner step:
    context  = c_n + A(c_n+1 - c_n) + P(p_n, p_n+1, dp, mask)      (A, P zero-initialised)
    action   = executed prefix, padded to the chunk by repeating its last command
    probe    = deterministic flow state psi_t(x0 = 0, x1 = action) at t = 0.5 (no write randomness, so logged and
               recomputed likelihoods agree)
Read: the fast weights W_n act inside every DiT pass of the K-step sampler (fixed during one generation call).
Output gate initialised at 0 -> exactly the base sampler at initialisation.
Likelihood recomputation (burn-in approximation, documented): W_{n-T} is replayed WITHOUT gradient from the causal
event sequence under the CURRENT slow weights (never stale), then the last T = 8 writes before decision n are
replayed with the full write graph. Gradients to W0 therefore come only from decisions with n <= T.
"""
import torch
import torch.nn as nn

from horsea.memory import TTT2Memory
from horsea.rdm.model import D_MODEL, DPROP_SCALE, EXEC, N_DEC


class TTTInfo(nn.Module):
    variant = "ttt_info"

    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.mem = TTT2Memory(gate_init=0.0)
        self.A = nn.Linear(D_MODEL, D_MODEL)
        self.P = nn.Linear(15 + EXEC, D_MODEL)
        for lin in (self.A, self.P):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def init_state(self, E, requires_grad=False):
        return self.mem.init_state(E, requires_grad=requires_grad)

    def _ctx_act(self, ev):
        pre, post = ev["pre"].float(), ev["post"].float()
        pp, pn = ev["p_pre"].float(), ev["p_post"].float()
        info = torch.cat([pp, pn, (pn - pp) * DPROP_SCALE, ev["amask"].float()], -1)
        ctx = pre + self.A(post - pre) + self.P(info)[:, None]
        a = ev["act"].float()
        act = torch.cat([a, a[:, -1:].expand(-1, self.flow.chunk - EXEC, -1)], 1)
        return ctx, act

    def write(self, state, ev, create_graph):
        """ev: dict of (E, ...) -- one event per episode row."""
        m, fl = self.mem, self.flow
        ctx, act = self._ctx_act(ev)
        E = ctx.shape[0]
        x1 = act.clamp(-1, 1)
        t = torch.full((E,), 0.5, device=x1.device)
        psi, _ = fl.interp(torch.zeros_like(x1), x1, t)
        new = dict(state)

        def hook(l, x):
            X = m._pre(m._tokens(x if create_graph else x.detach(), E))
            Wl = {k: new[k] for k in new if k.startswith(f"l{l}_")}
            loss = ((m._f(l, Wl, m._proj(m.k[l], X)) - m._proj(m.v[l], X)) ** 2).sum(-1).mean(-1).sum()
            scale = torch.sigmoid(m.write_gate[l](X.mean(1))).squeeze(-1)
            new.update(m.inner_step(Wl, loss, create_graph, scale))
            return m._ttt_apply(l, new, x, E)

        with torch.enable_grad():
            fl.decode(psi, t, ctx, layer_hook=hook)
        return new

    def mean(self, encm, eps, state):
        fl = self.flow
        field = self.mem.field(fl, state, encm)
        z, dt = eps, 1.0 / fl.n_steps
        t = torch.zeros(encm.shape[0], device=encm.device)
        for _ in range(fl.n_steps):
            z = z + dt * field(z, t)
            t = t + dt
        return z

    def replay(self, ev, n_max):
        """States W_0..W_n_max for m metaepisodes (events dict of (m, N, ...)), with the write graph (truncated at
        attempt boundaries). Returns dict k -> (n_max + 1, m, ...)."""
        m_ = ev["pre"].shape[0]
        with torch.enable_grad():  # the inner writes need a graph even if the caller is under no_grad
            st = self.init_state(m_)
            out = [st]
            for i in range(n_max):
                if i > 0 and i % N_DEC == 0:
                    st = {k: v.detach().requires_grad_(True) for k, v in st.items()}
                st = self.write(st, {k: v[:, i] for k, v in ev.items()}, create_graph=True)
                out.append(st)
            return {k: torch.stack([s[k] for s in out]) for k in out[0]}

    def replay_burnin(self, ev, meta_idx, n_hist, T=8):
        """States W_n for decision rows: ev dict of (m, N, ...); meta_idx, n_hist (b,) long. Exact values; the
        gradient flows through the last T writes (and into W0 when n <= T)."""
        m_ = ev["pre"].shape[0]
        n_max = int(n_hist.max())
        start = (n_hist - T).clamp(min=0)
        # no-grad chain under the current slow weights: W_0 .. W_{n_max - 1}
        st = self.init_state(m_, requires_grad=True)
        chain = [{k: v.detach() for k, v in st.items()}]
        for i in range(max(0, n_max - 1)):
            st = self.write(st, {k: v[:, i] for k, v in ev.items()}, create_graph=False)
            chain.append({k: v.detach() for k, v in st.items()})
        W = {k: torch.stack([c[k] for c in chain]) for k in chain[0]}   # (n_max, m, ...)
        with torch.enable_grad():
            w0 = self.init_state(len(meta_idx))
            cur = {k: torch.where((start == 0).view(-1, *[1] * (w0[k].dim() - 1)), w0[k],
                                  W[k][start.clamp(max=W[k].shape[0] - 1), meta_idx]) for k in w0}
            for j in range(T):
                pos = start + j
                act = pos < n_hist
                if not bool(act.any()):
                    break
                rows = {k: v[meta_idx, pos.clamp(max=v.shape[1] - 1)] for k, v in ev.items()}
                new = self.write(cur, rows, create_graph=True)
                cur = {k: torch.where(act.view(-1, *[1] * (v.dim() - 1)), new[k], v) for k, v in cur.items()}
        return cur
