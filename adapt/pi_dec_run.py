"""Policy Decorator (Yuan, Mu et al., ICLR 2025; github.com/tongzhoumu/policy_decorator @ 92ba9ba,
online/pi_dec_bet_maniskill2.py) as one adaptation stream on a held-out object, under the same protocol as
adapt_run.py: one env, terminal success reward at H, fixed independent eval starts, snapshots at the budgets.

  base policy  frozen source actor, deterministic mean action (official: base_policy.get_eval_action)
  residual     fresh SAC actor (official init: last layers std 0.01), final = base + res_scale * res
  critics      fresh, critic_input='sum' (Q(s, base + res_scale * res)), official default
  exploration  progressive: per step the residual is enabled with prob min(t / H, 1); random residuals before
               learning starts (official)

Deviations forced by the protocol (recorded in record.json): terminal 0/1 reward at H instead of sparse-1 with
early termination; one env and a 25k-transition budget instead of 16 envs and millions of steps, so
learning_starts / training_freq / utd are budget-scaled arguments; the residual actor and critics read the
source critics' fixed input transform (ObsNorm buffers) because our state vector is not normalised.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapt import common  # noqa: E402
from adapt.env import ENV_RECORD, HORIZON, make_env  # noqa: E402
from adapt.sac_ms3 import Actor, ReplayBuffer, SoftQNetwork  # noqa: E402

PI_DEC_REF = "tongzhoumu/policy_decorator@92ba9ba online/pi_dec_bet_maniskill2.py"

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--object", required=True)
p.add_argument("--base_actor", required=True, help="frozen source actor (pd_base_s*.pt from pi_dec_freeze)")
p.add_argument("--source_seed", type=int, required=True)
p.add_argument("--adapt_seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--budgets", default="0,500,2500,10000,25000")
p.add_argument("--eval_episodes", type=int, default=30)
p.add_argument("--eval_seed_base", type=int, default=10_000, help="fixed independent starts, shared by all cells")
# Policy Decorator's two tuned knobs (official README: res_scale 0.03-0.3, prog_explore 3e4-8e6)
p.add_argument("--res_scale", type=float, default=0.1)
p.add_argument("--prog_explore", type=int, default=10_000)
# official SAC settings
p.add_argument("--gamma", type=float, default=0.97)
p.add_argument("--tau", type=float, default=0.01)
p.add_argument("--policy_lr", type=float, default=3e-4)
p.add_argument("--q_lr", type=float, default=3e-4)
p.add_argument("--max_grad_norm", type=float, default=50.0)
p.add_argument("--log_std_min", type=float, default=-20.0)
# budget-scaled (official: learning_starts 8000, training_freq 64, utd 0.25, batch 1024)
p.add_argument("--learning_starts", type=int, default=500, help="transitions before the first update (10 attempts)")
p.add_argument("--training_freq", type=int, default=50, help="transitions between update rounds (one attempt)")
p.add_argument("--utd", type=float, default=1.0)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--retention_episodes_per_object", type=int, default=4)
p.add_argument("--time_cap_h", type=float, default=7.6)
args = p.parse_args()
assert (args.training_freq * args.utd).is_integer()

os.makedirs(args.out, exist_ok=True)
budgets = [int(b) for b in args.budgets.split(",")]
SNAP = os.path.join(args.out, "snapshots.json")
CKPT = os.path.join(args.out, "ckpt.pt")
snaps = json.load(open(SNAP)) if os.path.exists(SNAP) else {"snapshots": [], "complete": False}
if snaps["complete"]:
    print("already complete"); sys.exit(0)

timer = common.Timer()
common.gpu_guard()
seed = 1_000_000 * (args.source_seed + 1) + 1000 * args.adapt_seed + zlib.crc32(args.object.encode()) % 997   # stable across processes (str hash is salted)
common.seed_all(seed)
device = torch.device("cuda")
split = json.load(open(args.split))
assert args.object in split["dev"] + split["test"], f"{args.object} is not a held-out object"
assert args.object not in split["source"]

base_ck = torch.load(args.base_actor, map_location=device, weights_only=False)
record = dict(args=vars(args), method="PolicyDecorator", reference=PI_DEC_REF, git=common.git_rev(),
              versions=common.versions(), env=ENV_RECORD, seed=seed, base_actor_hash=common.file_hash(args.base_actor),
              base_record=base_ck["record"], split_hash=common.file_hash(args.split),
              eval_seeds=[args.eval_seed_base + i for i in range(args.eval_episodes)],
              deviations=["terminal 0/1 reward at H, no early termination (protocol) instead of sparse-1",
                          "1 env, budget-scaled learning_starts/training_freq/utd/batch",
                          "residual actor + critics use the source critics' fixed ObsNorm buffers"])
record["config_hash"] = common.cfg_hash({k: v for k, v in record.items() if k not in ("versions",)})
common.atomic_json(record, os.path.join(args.out, "record.json"))

env = make_env("PickSingleYCBSplit-v1", 1, [args.object], reward_mode="terminal")
eval_env = make_env("PickSingleYCBSplit-v1", args.eval_episodes, [args.object], reward_mode="terminal")
act_dim = int(np.prod(env.single_action_space.shape))
a_low = torch.tensor(env.single_action_space.low, device=device)
a_high = torch.tensor(env.single_action_space.high, device=device)


def layer_init(layer, std):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


base = Actor(env).to(device)
base.load_state_dict(base_ck["actor"])
base.eval().requires_grad_(False)

res_actor = Actor(env).to(device)        # official residual init: small last layers, wide log-std range
layer_init(res_actor.fc_mean, 0.01); layer_init(res_actor.fc_logstd, 0.01)
qf1, qf2 = SoftQNetwork(env).to(device), SoftQNetwork(env).to(device)
for q in (qf1, qf2):
    layer_init(q.net[-1], 0.01)
for n in (res_actor, qf1, qf2):
    n.norm.load_state_dict(base_ck["critic_norm"])
qf1_t, qf2_t = SoftQNetwork(env).to(device), SoftQNetwork(env).to(device)
qf1_t.load_state_dict(qf1.state_dict()); qf2_t.load_state_dict(qf2.state_dict())
q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
a_opt = torch.optim.Adam(res_actor.parameters(), lr=args.policy_lr)
target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
al_opt = torch.optim.Adam([log_alpha], lr=args.q_lr)


def res_sample(x):
    """Actor.get_action with the official log-std floor (-20) instead of sac_ms3's -5."""
    mean, log_std = res_actor(x)
    log_std = args.log_std_min + (log_std - (-5.0)) * (2.0 - args.log_std_min) / 7.0   # remap [-5,2] -> [min,2]
    normal = torch.distributions.Normal(mean, log_std.exp())
    x_t = normal.rsample(); y_t = torch.tanh(x_t)
    a = y_t * res_actor.action_scale + res_actor.action_bias
    logp = (normal.log_prob(x_t) - torch.log(res_actor.action_scale * (1 - y_t.pow(2)) + 1e-6)).sum(1, keepdim=True)
    return a, logp


def compose(base_a, res_a):
    # official: base + res_scale * res, clipped by the env; we clip explicitly so stored actions = executed actions
    return torch.clamp(base_a + args.res_scale * res_a, a_low, a_high)


# replay stores [res, base(s), base(s')] as the "action" (official critic_input='sum' layout)
class _Spec:
    pass


spec = _Spec(); spec.single_observation_space = env.single_observation_space
spec.single_action_space = type("S", (), {"shape": (3 * act_dim,)})()
rb = ReplayBuffer(spec, 1, budgets[-1] + HORIZON, device, device)

st = {"t": 0, "attempt": 0, "stream_success": [], "first_success_transition": None, "first_success_attempt": None,
      "updates": 0, "wall_h": 0.0, "learning_started": False}


def nets_state():
    return {"res_actor": res_actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
            "qf1_t": qf1_t.state_dict(), "qf2_t": qf2_t.state_dict(), "log_alpha": log_alpha.detach().cpu(),
            "q_opt": q_opt.state_dict(), "a_opt": a_opt.state_dict(), "al_opt": al_opt.state_dict()}


if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    n = ck["nets"]
    res_actor.load_state_dict(n["res_actor"]); qf1.load_state_dict(n["qf1"]); qf2.load_state_dict(n["qf2"])
    qf1_t.load_state_dict(n["qf1_t"]); qf2_t.load_state_dict(n["qf2_t"])
    with torch.no_grad():
        log_alpha.copy_(n["log_alpha"].to(device))
    q_opt.load_state_dict(n["q_opt"]); a_opt.load_state_dict(n["a_opt"]); al_opt.load_state_dict(n["al_opt"])
    rb.load_state_dict(ck["replay"]); common.set_rng_state(ck["rng"]); st = ck["st"]
    print("resumed at transition", st["t"])
done_budgets = {s["budget"] for s in snaps["snapshots"]}


@torch.no_grad()
def evaluate(e, seeds):
    """Read-only: deterministic base + scaled deterministic residual (official evaluate) on fixed starts."""
    res_actor.eval()
    obs, _ = e.reset(seed=list(seeds))
    for _ in range(HORIZON):
        obs, _, term, trunc, infos = e.step(compose(base.get_eval_action(obs), res_actor.get_eval_action(obs)))
    assert bool(torch.logical_or(term, trunc).all())
    res_actor.train()
    return infos["success_at_end"].cpu().numpy().astype(bool)


def snapshot(B):
    succ = evaluate(eval_env, record["eval_seeds"])
    entry = dict(budget=B, attempts=st["attempt"], updates=st["updates"], eval_success=float(succ.mean()),
                 eval_n=int(len(succ)), eval_successes=succ.tolist(),
                 stream_success_so_far=float(np.mean(st["stream_success"])) if st["stream_success"] else None,
                 first_success_transition=st["first_success_transition"],
                 first_success_attempt=st["first_success_attempt"], wall_h=st["wall_h"] + timer.wall() / 3600,
                 alpha=float(log_alpha.exp()), res_ratio=min(st["t"] / args.prog_explore, 1.0),
                 eval_sim_steps=int(len(succ) * HORIZON))
    n = args.retention_episodes_per_object * len(split["source"])
    if B == budgets[-1] and n > 0:
        renv = make_env("PickSingleYCBSplit-v1", n, split["source"], reward_mode="terminal")
        rs = evaluate(renv, [args.eval_seed_base + 5000 + i for i in range(n)])
        objs = np.array(renv.base_env.env_model_ids)
        entry["source_retention"] = {o: float(rs[objs == o].mean()) for o in split["source"]}
        entry["source_retention_mean"] = float(rs.mean())
        renv.close()
    snaps["snapshots"].append(entry)
    common.atomic_json(snaps, SNAP)
    common.atomic_torch_save({"nets": nets_state(), "replay": rb.state_dict(), "rng": common.rng_state(),
                              "st": st | {"wall_h": st["wall_h"] + timer.wall() / 3600}}, CKPT)
    print(json.dumps({k: v for k, v in entry.items() if k != "eval_successes"}), flush=True)


def update(data):
    alpha = log_alpha.exp().item()
    res_a, base_a, base_next = data.actions[:, :act_dim], data.actions[:, act_dim:2 * act_dim], data.actions[:, 2 * act_dim:]
    with torch.no_grad():
        nres, nlogp = res_sample(data.next_obs)
        nact = compose(base_next, nres)
        tq = torch.min(qf1_t(data.next_obs, nact), qf2_t(data.next_obs, nact)) - alpha * nlogp
        y = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * tq.view(-1)
    cur = compose(base_a, res_a)
    q_loss = F.mse_loss(qf1(data.obs, cur).view(-1), y) + F.mse_loss(qf2(data.obs, cur).view(-1), y)
    q_opt.zero_grad(); q_loss.backward()
    nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)
    nn.utils.clip_grad_norm_(qf2.parameters(), args.max_grad_norm)
    q_opt.step()
    pi_res, logp = res_sample(data.obs)
    pi = compose(base_a, pi_res)
    a_loss = (alpha * logp - torch.min(qf1(data.obs, pi), qf2(data.obs, pi))).mean()
    a_opt.zero_grad(); a_loss.backward()
    nn.utils.clip_grad_norm_(res_actor.parameters(), args.max_grad_norm)
    a_opt.step()
    with torch.no_grad():
        _, logp = res_sample(data.obs)
    al_loss = (-log_alpha * (logp + target_entropy)).mean()     # official uses log_alpha here
    al_opt.zero_grad(); al_loss.backward(); al_opt.step()
    for q, qt in ((qf1, qf1_t), (qf2, qf2_t)):
        for p_, tp in zip(q.parameters(), qt.parameters()):
            tp.data.copy_(args.tau * p_.data + (1 - args.tau) * tp.data)


n_updates = int(args.training_freq * args.utd)
obs, _ = env.reset(seed=seed + 1 + st["attempt"])
for B in budgets:
    if B in done_budgets:
        continue
    while st["t"] < B:
        if timer.wall() > args.time_cap_h * 3600:
            print("time cap; resubmit to continue"); sys.exit(3)
        with torch.no_grad():
            base_a = base.get_eval_action(obs)
            if st["learning_started"]:
                res_a, _ = res_sample(obs)
            else:   # official: uniform random residuals before learning starts
                res_a = torch.rand_like(base_a) * (a_high - a_low) + a_low
            if np.random.rand() >= min((st["t"] + 1) / args.prog_explore, 1.0):
                res_a = torch.zeros_like(res_a)     # progressive exploration: residual off for this step
            action = compose(base_a, res_a)
            next_obs, rew, term, trunc, infos = env.step(action)
            done = torch.logical_or(term, trunc)
            real_next = next_obs.clone()
            if "final_observation" in infos:
                real_next[done] = infos["final_observation"][done]
            base_next = base.get_eval_action(real_next)
        rb.add(obs, real_next, torch.cat([res_a, base_a, base_next], 1), rew, done.float())
        obs = next_obs
        st["t"] += 1
        if bool(done[0]):
            s = bool(infos["success_at_end"][0])
            st["attempt"] += 1
            st["stream_success"].append(s)
            if s and st["first_success_transition"] is None:
                st["first_success_transition"], st["first_success_attempt"] = st["t"], st["attempt"]
        if st["t"] >= args.learning_starts and st["t"] % args.training_freq == 0:
            st["learning_started"] = True
            for _ in range(n_updates):
                update(rb.sample(args.batch_size))
                st["updates"] += 1
    assert st["t"] == B, (st["t"], B)
    snapshot(B)

snaps["complete"] = True
snaps["record"] = dict(transitions=st["t"], attempts=st["attempt"], updates=st["updates"],
                       stream_success=st["stream_success"], wall_h=st["wall_h"] + timer.wall() / 3600)
common.atomic_json(snaps, SNAP)
print("COMPLETE", snaps["record"] | {"stream_success": float(np.mean(st["stream_success"]))})
