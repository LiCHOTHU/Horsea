"""Recurrent-denoising memory (RDM) on the frozen fm_policy_S DiT (spec 2026-09-29, sec. 3-4).

Three memories: slow weights (frozen base theta; trained psi = everything in this file), an experience bank H
(real executed transitions, appended only after execution), and a per-decision denoising workspace B (decoder
features of earlier denoising passes of the CURRENT action-generation call; cleared every physical decision).

One generation call (K = base's 10 Euler steps, same weights every pass), split at decoder layer SPLIT:
    g_k   = activation after decoder layer SPLIT                     (chunk tokens)
    b_k   = DepthRead(LN g_k, {(F_j, t_j): j < k} + null)            per action token, across passes
    r_k   = CrossAttn(Q(LN g_k, b_k, t_k), K(M_n), V(M_n))           M_n = event tokens of H_n + null
    g~_k  = (1 + gamma_k) * g_k + beta_k,  (gamma, beta) = P_cond([b_k, r_k]),  P_cond's last layer zero-init
    v_k, F_k = rest of the DiT;  z_{k+1} = z_k + dt v_k;  mean action chunk = z_K

Variants (same modules unless noted):
    plain      no psi at all (the frozen base sampler)
    adapter    same conditioning capacity, current observation only: M = tokens of the current encm, no workspace
    looped     workspace on, experience bank empty (null token only)
    readonce   workspace on, r_0 computed at k=0 and reused on every pass
    reread     workspace on, r_k recomputed on every pass (proposed)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256   # DiT width
SPLIT = 1       # modulate after decoder layer index 1 (of 0..3)
EXEC = 8        # executed prefix
N_ATT = 5       # attempts per metaepisode
N_DEC = 38      # decisions per attempt (horizon 304 = 38 x 8)
DPROP_SCALE = 10.0
VARIANTS = ("plain", "adapter", "looped", "readonce", "reread", "ttt_info")


def time_feat(t, n=8):
    f = torch.exp(torch.linspace(0, math.log(100.0), n, device=t.device))
    return torch.cat([torch.sin(t[..., None] * f), torch.cos(t[..., None] * f)], -1)


class EventTokenizer(nn.Module):
    """e = (f(o_n), p_n, A_exec, f(o_n+1), p_n+1, reset/attempt/step tags) -> 9 tokens (4 pre, 1 action, 4 post)."""

    def __init__(self, d=128, D=D_MODEL):
        super().__init__()
        self.d = d
        self.obs = nn.Linear(D, d)
        self.layer = nn.Parameter(torch.randn(4, d) * 0.02)
        self.typ = nn.Parameter(torch.randn(3, d) * 0.02)          # pre / action / post
        self.act = nn.Sequential(nn.Linear(EXEC * 7 + EXEC + 15, 256), nn.GELU(), nn.Linear(256, d))
        self.attempt = nn.Embedding(N_ATT + 1, d)
        self.step = nn.Embedding(N_DEC + 1, d)
        self.reset = nn.Embedding(2, d)
        self.mix = nn.TransformerEncoderLayer(d, 4, 256, dropout=0.0, batch_first=True, norm_first=True)

    def obs_tokens(self, encm):
        """encm (N, 4, D) -> (N, 4, d)"""
        return self.obs(F.layer_norm(encm.float(), (encm.shape[-1],))) + self.layer

    def forward(self, ev):
        """ev: dict of (N, ...) tensors -> (N, 9, d)"""
        pre = self.obs_tokens(ev["pre"]) + self.typ[0]
        post = self.obs_tokens(ev["post"]) + self.typ[2]
        pp, pn = ev["p_pre"].float(), ev["p_post"].float()
        a = torch.cat([ev["act"].float().flatten(1), ev["amask"].float(), pp, pn, (pn - pp) * DPROP_SCALE], -1)
        act = self.act(a)[:, None] + self.typ[1]
        tag = self.attempt(ev["attempt"]) + self.step(ev["dstep"].clamp(max=N_DEC)) + self.reset(ev["reset"])
        x = torch.cat([pre, act, post], 1) + tag[:, None]
        return self.mix(x)


class Reader(nn.Module):
    def __init__(self, variant, d=128, D=D_MODEL, heads=4):
        super().__init__()
        assert variant in VARIANTS and variant not in ("plain", "ttt_info")
        self.variant, self.d, self.h = variant, d, heads
        self.tok = EventTokenizer(d, D)
        self.qg = nn.Linear(D, d)
        self.tq = nn.Linear(16, d)
        # depth workspace (per action token, across denoising passes)
        self.dq, self.dk, self.dv = nn.Linear(d, d), nn.Linear(D + 16, d), nn.Linear(D + 16, d)
        self.d_null_k, self.d_null_v = nn.Parameter(torch.randn(d) * 0.02), nn.Parameter(torch.randn(d) * 0.02)
        # experience read
        self.hq = nn.Linear(3 * d, d)
        self.hk, self.hv, self.ho = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.m_null = nn.Parameter(torch.randn(1, d) * 0.02)
        # conditioning projection: last layer zero-init -> gamma = beta = 0 -> exactly the base sampler
        self.cond = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 2 * D))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    # ---- memory tokens ------------------------------------------------------------------------------
    def memory(self, ev_tokens, n_valid, encm=None):
        """ev_tokens (B, Nmax, 9, d) or None; n_valid (B,) events visible to each row -> (M, keymask)."""
        B = n_valid.shape[0] if encm is None else encm.shape[0]
        dev = self.m_null.device
        if self.variant == "adapter":  # current observation only
            M = self.tok.obs_tokens(encm) + self.tok.typ[0]
            mask = torch.ones(B, M.shape[1], dtype=torch.bool, device=dev)
        elif self.variant == "looped" or ev_tokens is None or ev_tokens.shape[1] == 0:
            M = torch.zeros(B, 0, self.d, device=dev)
            mask = torch.zeros(B, 0, dtype=torch.bool, device=dev)
        else:
            Bn, N, T, d = ev_tokens.shape
            M = ev_tokens.reshape(Bn, N * T, d)
            ev_ok = torch.arange(N, device=dev)[None] < n_valid[:, None]            # causal prefix
            mask = ev_ok[:, :, None].expand(Bn, N, T).reshape(Bn, N * T)
        M = torch.cat([self.m_null.expand(B, 1, self.d), M], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), mask], 1)
        return M, mask

    # ---- reads ----------------------------------------------------------------------------------------
    def depth_read(self, gq, ws):
        """gq (B, C, d) query; ws list of (F_j (B, C, D), tfeat (B, 16)) -> b (B, C, d)"""
        B, C, d = gq.shape
        q = self.dq(gq)
        ks = [self.d_null_k.expand(B, C, d)]
        vs = [self.d_null_v.expand(B, C, d)]
        if self.variant != "adapter":
            for Fj, tf in ws:
                x = torch.cat([F.layer_norm(Fj, (Fj.shape[-1],)), tf[:, None].expand(B, C, 16)], -1)
                ks.append(self.dk(x))
                vs.append(self.dv(x))
        K, V = torch.stack(ks, 2), torch.stack(vs, 2)                           # (B, C, J, d)
        att = torch.softmax((q[:, :, None] * K).sum(-1) / math.sqrt(d), -1)     # (B, C, J)
        return (att[..., None] * V).sum(2)

    def hist_read(self, gq, b, tq, M, mask):
        B, C, d = gq.shape
        q = self.hq(torch.cat([gq, b, tq[:, None].expand(B, C, d)], -1))
        h = self.h
        split = lambda x: x.reshape(B, x.shape[1], h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(split(q), split(self.hk(M)), split(self.hv(M)),
                                           attn_mask=mask[:, None, None, :])
        return self.ho(o.transpose(1, 2).reshape(B, C, d))


class RDM(nn.Module):
    """Mean action chunk mu_psi(c, H, eps) of the recurrent-denoising sampler."""

    def __init__(self, flow, variant, d=128):
        super().__init__()
        self.flow, self.variant = flow, variant
        self.reader = None if variant == "plain" else Reader(variant, d)

    def event_tokens(self, ev):
        """ev: dict of (B, Nmax, ...) -> (B, Nmax, 9, d) (recomputed under the current psi every call)."""
        if self.reader is None or self.variant in ("adapter", "looped"):
            return None
        B, N = ev["pre"].shape[:2]
        if N == 0:
            return None
        flat = {k: v.reshape(B * N, *v.shape[2:]) for k, v in ev.items()}
        return self.reader.tok(flat).reshape(B, N, 9, -1)

    def mean(self, encm, eps, ev_tokens=None, n_valid=None, trace=None):
        """encm (B, 4, D); eps (B, 16, 7) initial FM noise; returns z_K (B, 16, 7), unclamped."""
        fl = self.flow
        B, K = encm.shape[0], fl.n_steps
        z, dt = eps, 1.0 / K
        if self.reader is None:
            t = torch.zeros(B, device=encm.device)
            for _ in range(K):
                z = z + dt * fl.decode(z, t, encm)
                t = t + dt
            return z
        R = self.reader
        M, mask = R.memory(ev_tokens, n_valid if n_valid is not None else torch.zeros(B, dtype=torch.long, device=encm.device),
                           encm)
        ws, r0 = [], None
        t = torch.zeros(B, device=encm.device)
        for k in range(K):
            tf = time_feat(t)
            box = {}

            def hook(l, x, k=k, tf=tf):
                if l != SPLIT:
                    return x
                g = x.transpose(0, 1)                                           # (B, C, D)
                gq = R.qg(F.layer_norm(g, (g.shape[-1],)))
                b = R.depth_read(gq, ws)
                if self.variant == "readonce" and k > 0:
                    r = box["r0"]
                else:
                    r = R.hist_read(gq, b, R.tq(tf), M, mask)
                if k == 0:
                    box["r0"] = r
                gamma, beta = R.cond(torch.cat([b, r], -1)).chunk(2, -1)
                if trace is not None:
                    trace.append({"k": k, "gamma": gamma.abs().mean().item(), "beta": beta.abs().mean().item()})
                return ((1 + gamma) * g + beta).transpose(0, 1)

            if self.variant == "readonce" and k > 0:
                box["r0"] = r0
            v, h = fl.decode(z, t, encm, layer_hook=hook, return_hidden=True)
            if k == 0:
                r0 = box["r0"]
            ws.append((h.transpose(0, 1), tf))                                  # F_k, t_k (this call only)
            z = z + dt * v
            t = t + dt
        return z


def make_model(flow, variant):
    if variant == "ttt_info":
        from horsea.rdm.ttt import TTTInfo
        return TTTInfo(flow)
    return RDM(flow, variant)


class Value(nn.Module):
    """Critic V(c_n, attempt, decision position, successes so far). Training-only (not part of the actor);
    identical for every variant."""

    def __init__(self, D=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + N_ATT + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, encm, attempt, frac, nsucc):
        x = torch.cat([F.layer_norm(encm.float().mean(1), (encm.shape[-1],)), F.one_hot(attempt, N_ATT).float(),
                       frac[:, None].float(), nsucc[:, None].float()], -1)
        return self.net(x).squeeze(-1)


KAPPA = 4.0  # Bernoulli gripper calibration, common to every method: p(close) = sigmoid(KAPPA * mu_gripper)


def sample_action(mu, sigma, kappa=KAPPA):
    """Exploration distribution on the executed prefix (EXEC x 7): Gaussian on dims 0-5; explicit Bernoulli
    gripper (a Gaussian around +-1 essentially never crosses the sign threshold)."""
    u = mu.clone()
    u[..., :6] = mu[..., :6] + sigma * torch.randn_like(mu[..., :6])
    u[..., 6] = 2.0 * torch.bernoulli(torch.sigmoid(kappa * mu[..., 6])) - 1.0
    return u


def policy_logp(u, mu, sigma, kappa=KAPPA):
    """log p(u | mu) of sample_action, summed over the executed prefix."""
    g = (-0.5 * ((u[..., :6] - mu[..., :6]) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)
    close = (u[..., 6] > 0).float()
    lg = F.logsigmoid(kappa * mu[..., 6])
    lng = F.logsigmoid(-kappa * mu[..., 6])
    return g + (close * lg + (1 - close) * lng).sum(1)


def gauss_logp(u, mu, sigma):
    """log N(u; mu, sigma^2 I) summed over the executed prefix (EXEC x 7)."""
    return (-0.5 * ((u - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)
