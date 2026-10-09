"""ManiSkill's SAC baseline (examples/baselines/sac/sac.py, ManiSkill 3.0.1) -- Actor, SoftQNetwork,
ReplayBuffer and the update math are copied verbatim.  Added, as the plan requires:
  * `sample_mixed`: RLPD symmetric sampling, half target / half source replay (A2)
  * `sac_update`: the baseline's inner-loop update as a function so source training and every
    adaptation arm run the SAME code; `update_actor=False` gives the critic refit on terminal-reward data
  * replay save/load for resumable runs
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs, next_obs, action, reward, done = obs.cpu(), next_obs.cpu(), action.cpu(), reward.cpu(), done.cpu()
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    @property
    def size(self):
        return (self.per_env_buffer_size if self.full else self.pos) * self.num_envs

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )

    # --- added: persistence ---
    def state_dict(self):
        n = self.per_env_buffer_size if self.full else self.pos
        return {k: getattr(self, k)[:n].cpu() for k in ("obs", "next_obs", "actions", "rewards", "dones")} | \
               {"pos": self.pos, "full": self.full, "num_envs": self.num_envs}

    def load_state_dict(self, d):
        assert d["num_envs"] == self.num_envs, (d["num_envs"], self.num_envs)
        n = d["obs"].shape[0]
        assert n <= self.per_env_buffer_size
        for k in ("obs", "next_obs", "actions", "rewards", "dones"):
            getattr(self, k)[:n] = d[k].to(self.storage_device)
        self.pos, self.full = d["pos"], d["full"]


def sample_mixed(target_rb: ReplayBuffer, source_rb, batch_size: int):
    """RLPD symmetric sampling: 50 % target, 50 % source. Without a source buffer: plain target sampling."""
    if source_rb is None:
        return target_rb.sample(batch_size)
    h = batch_size // 2
    t, s = target_rb.sample(h), source_rb.sample(batch_size - h)
    return ReplayBufferSample(*[torch.cat([getattr(t, k), getattr(s, k)]) for k in
                                ("obs", "next_obs", "actions", "rewards", "dones")])


class ObsNorm(nn.Module):
    """Fixed affine input transform + feature mask, stored as buffers so it travels with every checkpoint.
    Identity unless Agent.set_obs_norm / set_obs_mask is called (BC sets it from demo statistics)."""

    def __init__(self, n_obs):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(n_obs))
        self.register_buffer("obs_std", torch.ones(n_obs))
        self.register_buffer("obs_mask", torch.ones(n_obs))

    def forward(self, x):
        return (x - self.obs_mean) / self.obs_std * self.obs_mask


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.norm = ObsNorm(int(np.array(env.single_observation_space.shape).prod()))
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([self.norm(x), a], 1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.norm = ObsNorm(int(np.array(env.single_observation_space.shape).prod()))
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(self.norm(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(self.norm(x))
        mean = self.fc_mean(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Agent:
    """Bundle of the baseline's networks, targets and optimisers so they can be built/saved/loaded in one place."""

    def __init__(self, env, device, q_lr=3e-4, policy_lr=3e-4, gamma=0.8, tau=0.01):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.actor = Actor(env).to(device)
        self.qf1, self.qf2 = SoftQNetwork(env).to(device), SoftQNetwork(env).to(device)
        self.qf1_target, self.qf2_target = SoftQNetwork(env).to(device), SoftQNetwork(env).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=policy_lr)
        self.target_entropy = -float(np.prod(env.single_action_space.shape))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)
        self.global_update = 0

    def nets(self):
        return [self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target]

    def set_obs_norm(self, mean, std, actor_mask=None, critic_mask=None):
        """Fixed input transform (buffers; saved/loaded with the state dicts). Same affine part for actor and
        critics; the feature masks may differ (actor: what the cloned policy needs; critics: what SAC tolerates)."""
        for n in self.nets():
            mask = actor_mask if n is self.actor else critic_mask
            with torch.no_grad():
                n.norm.obs_mean.copy_(mean.to(n.norm.obs_mean)); n.norm.obs_std.copy_(std.to(n.norm.obs_std))
                if mask is not None:
                    n.norm.obs_mask.copy_(mask.to(n.norm.obs_mask))

    def state_dict(self, optimizers=True):
        d = {"actor": self.actor.state_dict(), "qf1": self.qf1.state_dict(), "qf2": self.qf2.state_dict(),
             "qf1_target": self.qf1_target.state_dict(), "qf2_target": self.qf2_target.state_dict(),
             "log_alpha": self.log_alpha.detach().cpu(), "global_update": self.global_update,
             "gamma": self.gamma, "tau": self.tau}
        if optimizers:
            d.update(q_optimizer=self.q_optimizer.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
                     a_optimizer=self.a_optimizer.state_dict())
        return d

    def load_state_dict(self, d, optimizers=True):
        self.actor.load_state_dict(d["actor"])
        self.qf1.load_state_dict(d["qf1"]); self.qf2.load_state_dict(d["qf2"])
        self.qf1_target.load_state_dict(d.get("qf1_target", d["qf1"]))
        self.qf2_target.load_state_dict(d.get("qf2_target", d["qf2"]))
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device).reshape(1))
        self.global_update = d.get("global_update", 0)
        if optimizers and "q_optimizer" in d:
            self.q_optimizer.load_state_dict(d["q_optimizer"])
            self.actor_optimizer.load_state_dict(d["actor_optimizer"])
            self.a_optimizer.load_state_dict(d["a_optimizer"])


def sac_update(ag: Agent, data: ReplayBufferSample, update_actor=True):
    """One gradient update: the baseline's loop body (autotune on, policy_frequency 1, target freq 1)."""
    alpha = ag.log_alpha.exp().item()
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = ag.actor.get_action(data.next_obs)
        qf1_next_target = ag.qf1_target(data.next_obs, next_state_actions)
        qf2_next_target = ag.qf2_target(data.next_obs, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * ag.gamma * (min_qf_next_target).view(-1)
    qf1_a_values = ag.qf1(data.obs, data.actions).view(-1)
    qf2_a_values = ag.qf2(data.obs, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss
    ag.q_optimizer.zero_grad()
    qf_loss.backward()
    ag.q_optimizer.step()
    info = {"qf_loss": qf_loss.item() / 2.0, "q": qf1_a_values.mean().item(), "alpha": alpha}
    if update_actor:
        pi, log_pi, _ = ag.actor.get_action(data.obs)
        qf1_pi = ag.qf1(data.obs, pi)
        qf2_pi = ag.qf2(data.obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
        ag.actor_optimizer.zero_grad()
        actor_loss.backward()
        ag.actor_optimizer.step()
        with torch.no_grad():
            _, log_pi, _ = ag.actor.get_action(data.obs)
        alpha_loss = (-ag.log_alpha.exp() * (log_pi + ag.target_entropy)).mean()
        ag.a_optimizer.zero_grad()
        alpha_loss.backward()
        ag.a_optimizer.step()
        info["actor_loss"] = actor_loss.item()
    for param, target_param in zip(ag.qf1.parameters(), ag.qf1_target.parameters()):
        target_param.data.copy_(ag.tau * param.data + (1 - ag.tau) * target_param.data)
    for param, target_param in zip(ag.qf2.parameters(), ag.qf2_target.parameters()):
        target_param.data.copy_(ag.tau * param.data + (1 - ag.tau) * target_param.data)
    ag.global_update += 1
    return info


@torch.no_grad()
def evaluate(ag: Agent, eval_env, seeds, horizon):
    """Read-only assessment on fixed independent starts (one episode per env, deterministic actions).
    Returns per-episode success at the final state. Nothing here updates policy, replay or statistics."""
    ag.actor.eval()
    obs, _ = eval_env.reset(seed=list(seeds))
    succ = None
    for _ in range(horizon):
        obs, rew, term, trunc, infos = eval_env.step(ag.actor.get_eval_action(obs))
    succ = infos["success_at_end"].clone()
    assert bool(torch.logical_or(term, trunc).all()), "evaluation episodes must end exactly at the horizon"
    ag.actor.train()
    return succ.cpu().numpy().astype(bool)
