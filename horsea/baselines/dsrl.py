"""DSRL (Wagenmaker et al. 2025, "Steering Your Diffusion Policy with Latent Space RL") on the frozen fm_policy_S.

PyTorch port of the official release (github.com/nakamotoo/dsrl_pi0 @ 7f48937: examples/train_utils_sim.py,
jaxrl2/agents/pixel_sac/*). The base policy is frozen; a SAC agent acts in the base's NOISE space: its action w
(tanh-squashed, |w| <= action_magnitude) replaces the Gaussian initial noise of the flow, and the base maps it to
an action chunk. As in the release, the latent covers `latent_steps` chunk positions and its last row is repeated
over the rest of the chunk (release: one 32-d row repeated over pi0's 50-step chunk -> here one 7-d row over 16).

Kept from the release: SAC with a 10-Q ensemble (mean reduction for target and actor), 128x3 MLPs, LayerNorm in
the critic, Dense->LayerNorm->tanh bottleneck (latent 50) per network, learned-std tanh-normal actor (log-std
clip [-20, 2], 1e-2 head init), lr actor 1e-4 / critic 3e-4 / temperature 3e-4, init temperature 1, target entropy
-dim/2, tau 0.005, no entropy backup, batch 256, UTD 20 per decision, updates start after 500 decisions,
sparse -1 per decision / 0 at success with mask 0, per-decision discount 0.999^EXEC. Before the first update the
collection noise is standard Gaussian (the base policy); evaluation before the first update uses i.i.d. noise for
every chunk position (the base policy exactly), afterwards the stochastic latent policy (release behaviour).

Adaptation (documented deviation): the release encodes 64x64 agentview pixels with a small CNN trained from
scratch; here the observation is the frozen base encoder summary encm (4x256, flattened) plus the 5-d proprio.

    python -m horsea.baselines.dsrl --task 22 --shift rot30 --out <dir>
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from horsea.baselines.common import Ctx, protocol_args, run_stream, transitions
from horsea.selfplay import EXEC


def _orth(layer, scale):
    nn.init.orthogonal_(layer.weight, scale)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(dims, layer_norm=False, activate_final=False):
    """jaxrl2.networks.MLP: orthogonal(sqrt 2) init, [LayerNorm], ReLU."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(_orth(nn.Linear(dims[i], dims[i + 1]), math.sqrt(2)))
        if i + 2 < len(dims) or activate_final:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    """PixelMultiplexer bottleneck: Dense(latent, xavier_normal) -> LayerNorm -> tanh; proprio concatenated."""

    def __init__(self, in_dim, latent):
        super().__init__()
        self.fc, self.ln = nn.Linear(in_dim, latent), nn.LayerNorm(latent)
        nn.init.xavier_normal_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, encm, low):
        return torch.cat([torch.tanh(self.ln(self.fc(encm.flatten(1)))), low], -1)


class Actor(nn.Module):
    def __init__(self, obs_dim, low_dim, act_dim, hidden, latent, magnitude):
        super().__init__()
        self.enc = Bottleneck(obs_dim, latent)
        self.trunk = mlp([latent + low_dim, *hidden], activate_final=True)
        self.mean = _orth(nn.Linear(hidden[-1], act_dim), 1e-2)
        self.log_std = _orth(nn.Linear(hidden[-1], act_dim), 1e-2)
        self.mag = magnitude

    def dist(self, encm, low):
        h = self.trunk(self.enc(encm, low))
        return self.mean(h), self.log_std(h).clamp(-20, 2)

    def sample(self, encm, low):
        """Tanh-normal rescaled to [-mag, mag] (distrax TanhMultivariateNormalDiag): action, log-prob."""
        mu, log_std = self.dist(encm, low)
        x = mu + log_std.exp() * torch.randn_like(mu)
        y = torch.tanh(x)
        logp = (-0.5 * ((x - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - x - F.softplus(-2 * x))).sum(-1) - mu.shape[-1] * math.log(self.mag)
        return y * self.mag, logp


class Critic(nn.Module):
    """StateActionEnsemble: num_qs independent MLPs with LayerNorm on [bottleneck(obs), action]."""

    def __init__(self, obs_dim, low_dim, act_dim, hidden, latent, num_qs):
        super().__init__()
        self.enc = Bottleneck(obs_dim, latent)
        self.qs = nn.ModuleList([mlp([latent + low_dim + act_dim, *hidden, 1], layer_norm=True) for _ in range(num_qs)])

    def forward(self, encm, low, a):
        x = torch.cat([self.enc(encm, low), a], -1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs])  # (num_qs, B)


class DSRL:
    def __init__(self, ctx, a):
        self.ctx, self.a, self.dev = ctx, a, ctx.device
        f = ctx.flow
        self.chunk, self.adim = f.chunk, f.adim
        self.lat_shape = (a.latent_steps, f.adim)
        act_dim, obs_dim, low_dim = a.latent_steps * f.adim, 4 * 256, 5
        hid = (a.hidden,) * 3
        self.actor = Actor(obs_dim, low_dim, act_dim, hid, a.latent_dim, a.action_magnitude).to(self.dev)
        self.critic = Critic(obs_dim, low_dim, act_dim, hid, a.latent_dim, a.num_qs).to(self.dev)
        self.target = Critic(obs_dim, low_dim, act_dim, hid, a.latent_dim, a.num_qs).to(self.dev)
        self.target.load_state_dict(self.critic.state_dict())
        self.log_temp = torch.zeros((), device=self.dev, requires_grad=True)  # log(init temperature 1.0)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=a.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=a.critic_lr)
        self.opt_t = torch.optim.Adam([self.log_temp], lr=a.temp_lr)
        self.target_entropy = -act_dim / 2
        self.discount = a.discount ** EXEC
        self.buf = {k: [] for k in ("encm", "low", "act", "rew", "mask", "nencm", "nlow")}
        self.updates = 0
        self.last = {}

    # ----- acting ---------------------------------------------------------------------------
    def _latent_to_noise(self, w):
        w = w.reshape(-1, *self.lat_shape)
        rep = w[:, -1:].expand(-1, self.chunk - w.shape[1], -1)
        return torch.cat([w, rep], 1)

    @torch.no_grad()
    def decide_train(self, encm, low):
        B = encm.shape[0]
        if self.updates == 0:  # release: initial collection with standard Gaussian latent
            w = torch.randn(B, int(np.prod(self.lat_shape)), device=self.dev)
        else:
            w, _ = self.actor.sample(encm.float(), low)
        a = self.ctx.flow.sample(encm, self._latent_to_noise(w))
        return a, [{"w": w[b].cpu()} for b in range(B)]

    @torch.no_grad()
    def decide_eval(self, encm, low):
        B = encm.shape[0]
        if self.updates == 0:  # the base policy exactly (i.i.d. noise over the whole chunk)
            return self.ctx.flow.sample(encm), [{} for _ in range(B)]
        w, _ = self.actor.sample(encm.float(), low)
        return self.ctx.flow.sample(encm, self._latent_to_noise(w)), [{} for _ in range(B)]

    # ----- learning -------------------------------------------------------------------------
    def observe(self, eps):
        n = 0
        for ep in eps:
            for d, (ne, nl), r, term in transitions(ep):
                for k, v in (("encm", d["encm"]), ("low", d["low"]), ("act", d["w"]), ("rew", r - 1.0),
                             ("mask", 0.0 if term else 1.0), ("nencm", ne), ("nlow", nl)):
                    self.buf[k].append(v)
                n += 1
        if len(self.buf["rew"]) > self.a.start_online_updates:
            data = {k: (torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(v)).float().to(self.dev)
                    for k, v in self.buf.items()}
            for _ in range(n * self.a.utd):
                self.update(data)

    def update(self, D):
        idx = torch.randint(0, D["rew"].shape[0], (self.a.batch_size,), device=self.dev)
        b = {k: v[idx] for k, v in D.items()}
        temp = self.log_temp.exp().detach()
        with torch.no_grad():
            na, _ = self.actor.sample(b["nencm"], b["nlow"])
            nq = self.target(b["nencm"], b["nlow"], na).mean(0)
            tq = b["rew"] + self.discount * b["mask"] * nq
        qs = self.critic(b["encm"], b["low"], b["act"])
        c_loss = ((qs - tq) ** 2).mean()
        self.opt_c.zero_grad(); c_loss.backward(); self.opt_c.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target.parameters()):
                tp.mul_(1 - self.a.tau).add_(self.a.tau * p)
        a, logp = self.actor.sample(b["encm"], b["low"])
        q = self.critic(b["encm"], b["low"], a).mean(0)
        a_loss = (logp * temp - q).mean()
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()
        entropy = -logp.detach().mean()
        t_loss = self.log_temp.exp() * (entropy - self.target_entropy)
        self.opt_t.zero_grad(); t_loss.backward(); self.opt_t.step()
        self.updates += 1
        self.last = dict(critic_loss=c_loss.item(), actor_loss=a_loss.item(), q=qs.mean().item(),
                         entropy=entropy.item(), temperature=temp.item())

    # ----- bookkeeping ----------------------------------------------------------------------
    def stats(self):
        return {"buffer": len(self.buf["rew"]), **self.last}

    def record(self):
        return dict(vars(self.a) | {"per_decision_discount": self.discount, "target_entropy": self.target_entropy,
                                    "observation": "frozen encm (4x256) + proprio (5); release uses 64x64 pixels"})

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(),
                    log_temp=self.log_temp.detach().cpu(), opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                    opt_t=self.opt_t.state_dict(), buf=self.buf, updates=self.updates)

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.target.load_state_dict(d["target"])
        with torch.no_grad():
            self.log_temp.copy_(d["log_temp"].to(self.dev))
        self.opt_a.load_state_dict(d["opt_a"]); self.opt_c.load_state_dict(d["opt_c"]); self.opt_t.load_state_dict(d["opt_t"])
        self.buf, self.updates = d["buf"], d["updates"]


def main():
    ap = protocol_args(__doc__)
    ap.add_argument("--latent_steps", type=int, default=1, help="chunk rows the latent covers (release: 1, repeated)")
    ap.add_argument("--action_magnitude", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=50)
    ap.add_argument("--num_qs", type=int, default=10)
    ap.add_argument("--actor_lr", type=float, default=1e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--temp_lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.999, help="per env step; per decision ** EXEC")
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--utd", type=int, default=20, help="gradient steps per decision (release multi_grad_step)")
    ap.add_argument("--start_online_updates", type=int, default=500, help="decisions before the first update")
    args = ap.parse_args()
    from adapt import common as C
    C.seed_all(args.seed)
    ctx = Ctx(args)
    run_stream(args, DSRL(ctx, args), "dsrl")


if __name__ == "__main__":
    main()
