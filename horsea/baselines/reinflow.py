"""ReinFlow (Zhang et al. 2025, "ReinFlow: Fine-tuning Flow Matching Policy with Online RL") on fm_policy_S.

PyTorch adapter around the official algorithm (github.com/ReinFlow/ReinFlow @ e722e15: model/flow/ft_ppo/ppoflow.py,
model/flow/mlp_flow.py NoisyFlowMLP/ExploreNoiseNet, agent/finetune/reinflow/train_ppo_flow_img_agent.py, buffer.py).
A trainable copy of the base's velocity decoder plus a learned noise network turn the Euler sampler into a
Markov chain x_{k+1} ~ N(x_k + v(x_k, t_k) dt, sigma(s)^2); PPO is run on the chain's exact log-likelihood.
The frozen base is kept only as the reference; the ResNet/DiT observation encoder is frozen (encm precomputed),
following the repo's decoder-only fine-tuning convention (horsea.base.decoder_parameters).

Kept from the release (cfg/robomimic/finetune/square/ft_ppo_reflow_mlp_img.yaml + agent code): time-independent
noise net (cond -> 384x3 Tanh -> chunk*adim log-variance, tanh-squashed into [min_std, max_std] = [0.08, 0.14]),
noise at every denoising step, samples clipped to +-3 sigma, final clamp to [-1, 1], log-prob including the initial
Gaussian and normalized by (denoising steps + 1) and by chunk*adim, log-prob clamp [-1, 1], PPO clip 0.01, target KL
0.01 early stop, 10 epochs, minibatch 500, 5x sample repetition, ent_coef 0.01, vf_coef 0.5, max grad norm 25,
gamma 0.999, GAE lambda 0.95, running reward scaling, 2 critic-warm-up iterations, AdamW lr actor 3.5e-6 (cosine,
10 warm-up iters) / critic 4.5e-4 -> 3e-4 (cosine, 25 warm-up), critic 256x3 Mish.

Adaptations (documented deviations):
  * denoising steps = the base's 10 Euler steps (release square config: 1). Every step injects noise >= min_std, so
    the end-to-end perturbation is larger than in the 1-step config; min/max std are flags to tune on dev.
  * intermediate-action clipping is OFF by default: our base was trained without it and its intermediate states
    leave [-1, 1]; clipping them would change the base policy itself.
  * iteration = `episodes_per_itr` episodes (release: 50 envs x 400 decisions), gradient accumulation capped at the
    number of minibatches per epoch so that at least one optimizer step happens per epoch.
  * GAE per episode, bootstrapping truncated episodes from V(final observation) (release buffer bootstraps from
    the next episode's first state on truncation).
  * no early "fine-tuning failed" exit (release exits if success < 5%; held-out conditions start near 0).
  * critic input: frozen encm (flattened) + proprio, no ViT; DiT/ResNet encoder not fine-tuned.

    python -m horsea.baselines.reinflow --task 22 --shift rot30 --out <dir>
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from horsea.base import DECODER_PREFIXES
from horsea.baselines.common import Ctx, protocol_args, run_stream, transitions


class NoiseNet(nn.Module):
    """ExploreNoiseNet: MLP -> log sigma^2, tanh-squashed into [log min^2, log max^2]."""

    def __init__(self, in_dim, out_dim, hidden, min_std, max_std):
        super().__init__()
        dims, layers = [in_dim, *hidden, out_dim], []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 2 < len(dims):
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.lo, self.hi = math.log(min_std ** 2), math.log(max_std ** 2)

    def forward(self, x):
        lv = self.lo + (self.hi - self.lo) * (torch.tanh(self.net(x)) + 1) / 2
        return torch.exp(0.5 * lv)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        dims, layers = [in_dim, *hidden, 1], []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 2 < len(dims):
                layers.append(nn.Mish())
        self.net = nn.Sequential(*layers)

    def forward(self, encm, low):
        return self.net(torch.cat([encm.flatten(1), low], -1)).squeeze(-1)


class RunningRewardScaler:
    """util/reward_scaling.py: divide rewards by the running std of backward discounted returns, clip 10."""

    def __init__(self, gamma, clip=10.0, eps=1e-8):
        self.gamma, self.clip, self.eps = gamma, clip, eps
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, rewards):  # one episode's reward sequence
        ret, rets = 0.0, []
        for r in rewards:
            ret = ret * self.gamma + r
            rets.append(ret)
        x = np.asarray(rets, dtype=np.float64)
        bm, bv, bc = x.mean(), x.var(), len(x)
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot
        return np.clip(np.asarray(rewards) / np.sqrt(self.var + self.eps), -self.clip, self.clip)


def cosine_warmup(itr, max_lr, min_lr, cycle, warmup):
    """CosineAnnealingWarmupRestarts (cycle_mult 1, gamma 1) evaluated at iteration itr."""
    s = itr % cycle
    if s < warmup:
        return min_lr + (max_lr - min_lr) * s / warmup
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * (s - warmup) / (cycle - warmup))) / 2


class ReinFlow:
    def __init__(self, ctx, a):
        self.ctx, self.a, self.dev = ctx, a, ctx.device
        # trainable copy of the velocity net only (the policy object carries the env hooks); the copy shares the
        # base's schedule/constants through a shallow copy of its Flow
        self.flow = copy.copy(ctx.flow)
        self.vnet = self.flow.vnet = copy.deepcopy(ctx.policy.velocity_net).eval()  # eval: no dropout, same net
        self.dec_params = [p for n, p in self.vnet.named_parameters() if n.startswith(DECODER_PREFIXES)]
        for p in self.dec_params:
            p.requires_grad_(True)
        self.K, self.chunk, self.adim = self.flow.n_steps, self.flow.chunk, self.flow.adim
        self.D = self.chunk * self.adim
        self.noise = NoiseNet(4 * 256, self.D, [a.noise_hidden] * 3, a.min_std, a.max_std).to(self.dev)
        self.critic = Critic(4 * 256 + 5, [256] * 3).to(self.dev)
        self.actor_params = self.dec_params + list(self.noise.parameters())
        self.opt_a = torch.optim.AdamW(self.actor_params, lr=a.actor_lr, weight_decay=0)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=a.critic_lr, weight_decay=0)
        self.scaler = RunningRewardScaler(a.gamma)
        self.pending, self.itr, self.updates, self.last = [], 0, 0, {}
        self._set_lr()

    def _set_lr(self):
        a = self.a
        for g in self.opt_a.param_groups:
            g["lr"] = cosine_warmup(self.itr, a.actor_lr, a.actor_min_lr, 100, 10)
        for g in self.opt_c.param_groups:
            g["lr"] = cosine_warmup(self.itr, a.critic_lr, a.critic_min_lr, 100, 25)

    # ----- the noisy flow (PPOFlow.get_actions / get_logprobs) -------------------------------
    def _step(self, x, t, encm, grad=False):
        v = self.flow.decode(x, t, encm)
        std = self.noise(encm.flatten(1))
        return v, (std if grad else std.detach())

    @torch.no_grad()
    def sample(self, encm, eval_mode):
        B, dt = encm.shape[0], 1.0 / self.K
        x = torch.randn(B, self.chunk, self.adim, device=self.dev)
        chain = [x]
        for i in range(self.K):
            t = torch.full((B,), i * dt, device=self.dev)
            v, std = self._step(x, t, encm)
            x = x + v * dt
            if self.a.clip_intermediate:
                x = x.clamp(-1, 1)
            std = std.clamp(min=self.a.min_std).reshape(x.shape)
            if not eval_mode:
                x = torch.max(torch.min(x + std * torch.randn_like(x), x + 3 * std), x - 3 * std)
            if i == self.K - 1:
                x = x.clamp(-1, 1)
            chain.append(x)
        return x, torch.stack(chain, 1)  # (B, K+1, chunk, adim)

    def logprobs(self, encm, chain, entropy=False):
        B, dt = chain.shape[0], 1.0 / self.K
        prev, nxt = chain[:, :-1].flatten(2), chain[:, 1:].flatten(2)
        init = Normal(torch.zeros(B, self.D, device=self.dev), 1.0)
        lp = init.log_prob(chain[:, 0].reshape(B, -1)).sum(-1)  # account_for_initial_stochasticity
        ent = init.entropy().sum(-1)
        vel, stds = [], []
        for i in range(self.K):
            v, std = self._step(chain[:, i], torch.full((B,), i * dt, device=self.dev), encm, grad=True)
            vel.append(v.flatten(1)); stds.append(std)
        mean = prev + torch.stack(vel, 1) * dt
        if self.a.clip_intermediate:
            mean = mean.clamp(-1, 1)
        dist = Normal(mean, torch.stack(stds, 1))
        lp = (lp + dist.log_prob(nxt).sum(-1).sum(-1)) / (self.K + 1) / self.D
        if not entropy:
            return lp
        ent = (ent + dist.entropy().sum(-1).sum(-1)) / (self.K + 1) / self.D
        return lp, ent, torch.stack(stds, 1).mean()

    # ----- protocol hooks ---------------------------------------------------------------------
    def decide_train(self, encm, low):
        a, chain = self.sample(encm, eval_mode=False)
        return a, [{"chain": chain[b].cpu()} for b in range(a.shape[0])]

    def decide_eval(self, encm, low):
        a, _ = self.sample(encm, eval_mode=True)
        return a, [{} for _ in range(a.shape[0])]

    def observe(self, eps):
        self.pending += eps
        if len(self.pending) >= self.a.episodes_per_itr:
            self.update(self.pending)
            self.pending = []

    # ----- PPO --------------------------------------------------------------------------------
    @torch.no_grad()
    def _dataset(self, eps):
        a, obs, low, chains, rets, advs, vals, lps = self.a, [], [], [], [], [], [], []
        for ep in eps:
            tr = transitions(ep)
            if not tr:
                continue
            e = torch.stack([d["encm"] for d, *_ in tr]).to(self.dev)
            l = torch.stack([d["low"] for d, *_ in tr]).to(self.dev)
            c = torch.stack([d["chain"] for d, *_ in tr]).to(self.dev)
            v = self.critic(e, l)
            lp = torch.cat([self.logprobs(e[i:i + 256], c[i:i + 256]) for i in range(0, len(tr), 256)])
            r = self.scaler([x[2] for x in tr]) * a.reward_scale_const
            term = tr[-1][3]
            nv = 0.0 if term else self.critic(tr[-1][1][0][None].to(self.dev), tr[-1][1][1][None].to(self.dev)).item()
            adv, last = torch.zeros(len(tr), device=self.dev), 0.0
            for t in reversed(range(len(tr))):
                nextv = nv if t == len(tr) - 1 else v[t + 1].item()
                nonterm = 0.0 if (t == len(tr) - 1 and term) else 1.0
                delta = r[t] + a.gamma * nextv * nonterm - v[t].item()
                last = delta + a.gamma * a.gae_lambda * nonterm * last
                adv[t] = last
            obs.append(e); low.append(l); chains.append(c); vals.append(v); lps.append(lp)
            advs.append(adv); rets.append(adv + v)
        cat = lambda xs: torch.cat(xs)
        return cat(obs), cat(low), cat(chains), cat(rets), cat(vals), cat(advs), cat(lps)

    def update(self, eps):
        a = self.a
        obs, low, chains, rets, vals, advs, oldlp = self._dataset(eps)
        N = obs.shape[0]
        total = N * a.repeat
        n_mb = max(1, math.ceil(total / a.batch_size))
        accum = min(a.grad_accumulate, n_mb)
        warm = self.itr < a.n_critic_warmup_itr
        kl, stopped, clipfracs, stds = 0.0, False, [], []
        ev = 1 - (rets - vals).var() / rets.var() if rets.var() > 0 else float("nan")
        for epoch in range(a.update_epochs):
            perm = torch.randperm(total, device=self.dev) % N
            self.opt_a.zero_grad(); self.opt_c.zero_grad()
            for mb in range(n_mb):
                if a.target_kl and kl > a.target_kl and not warm:
                    stopped = True
                    break
                ix = perm[mb * a.batch_size:(mb + 1) * a.batch_size]
                newlp, ent, std = self.logprobs(obs[ix], chains[ix], entropy=True)
                newlp = newlp.clamp(a.logprob_min, a.logprob_max)
                o = oldlp[ix].clamp(a.logprob_min, a.logprob_max)
                adv = advs[ix]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(ix) > 1 else adv
                logratio = newlp - o
                ratio = logratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1).abs() > a.clip_ploss_coef).float().mean().item())
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.clip_ploss_coef, 1 + a.clip_ploss_coef)).mean()
                vloss = 0.5 * ((self.critic(obs[ix], low[ix]) - rets[ix]) ** 2).mean()
                loss = pg - ent.mean() * a.ent_coef + vloss * a.vf_coef
                loss.backward()
                stds.append(std.item())
                if (mb + 1) % accum == 0:
                    if not warm:
                        torch.nn.utils.clip_grad_norm_(self.actor_params, a.max_grad_norm)
                        self.opt_a.step()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), a.max_grad_norm)
                    self.opt_c.step()
                    self.opt_a.zero_grad(); self.opt_c.zero_grad()
                    self.updates += 1
            if stopped:
                break
        self.itr += 1
        self._set_lr()
        self.last = dict(itr=self.itr, decisions=N, approx_kl=kl, kl_stop=stopped, explained_var=float(ev),
                         clipfrac=float(np.mean(clipfracs)) if clipfracs else None,
                         noise_std=float(np.mean(stds)) if stds else None, pg_loss=pg.item(), v_loss=vloss.item())

    # ----- bookkeeping --------------------------------------------------------------------------
    def stats(self):
        return dict(self.last, pending_episodes=len(self.pending))

    def record(self):
        return dict(vars(self.a) | {"denoising_steps": self.K, "chunk": self.chunk, "adim": self.adim,
                                    "trainable": "velocity decoder (DECODER_PREFIXES) + noise net; encoder frozen"})

    def state_dict(self):
        return dict(decoder={n: p.detach().cpu() for n, p in self.vnet.named_parameters() if p.requires_grad},
                    noise=self.noise.state_dict(), critic=self.critic.state_dict(), opt_a=self.opt_a.state_dict(),
                    opt_c=self.opt_c.state_dict(), scaler=vars(self.scaler), pending=self.pending, itr=self.itr,
                    updates=self.updates)

    def load_state_dict(self, d):
        named = dict(self.vnet.named_parameters())
        with torch.no_grad():
            for n, v in d["decoder"].items():
                named[n].copy_(v.to(self.dev))
        self.noise.load_state_dict(d["noise"]); self.critic.load_state_dict(d["critic"])
        self.opt_a.load_state_dict(d["opt_a"]); self.opt_c.load_state_dict(d["opt_c"])
        self.scaler.__dict__.update(d["scaler"])
        self.pending, self.itr, self.updates = d["pending"], d["itr"], d["updates"]
        self._set_lr()


def main():
    ap = protocol_args(__doc__)
    ap.add_argument("--episodes_per_itr", type=int, default=10)
    ap.add_argument("--min_std", type=float, default=0.08)
    ap.add_argument("--max_std", type=float, default=0.14)
    ap.add_argument("--noise_hidden", type=int, default=384)
    ap.add_argument("--clip_intermediate", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--reward_scale_const", type=float, default=1.0)
    ap.add_argument("--update_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=5, help="minibatch_duplicate_multiplier (repeat_samples)")
    ap.add_argument("--grad_accumulate", type=int, default=20)
    ap.add_argument("--actor_lr", type=float, default=3.5e-6)
    ap.add_argument("--actor_min_lr", type=float, default=3.5e-6)
    ap.add_argument("--critic_lr", type=float, default=4.5e-4)
    ap.add_argument("--critic_min_lr", type=float, default=3e-4)
    ap.add_argument("--n_critic_warmup_itr", type=int, default=2)
    ap.add_argument("--clip_ploss_coef", type=float, default=0.01)
    ap.add_argument("--target_kl", type=float, default=0.01)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=25.0)
    ap.add_argument("--logprob_min", type=float, default=-1.0)
    ap.add_argument("--logprob_max", type=float, default=1.0)
    args = ap.parse_args()
    from adapt import common as C
    C.seed_all(args.seed)
    ctx = Ctx(args)
    run_stream(args, ReinFlow(ctx, args), "reinflow")


if __name__ == "__main__":
    main()
