"""Behaviour cloning of the long-term memory (source policy) on motion-planning demos converted to the plan's
controller and observation (adapt/bc_data.py).  The network is the SAC baseline's Actor; only its deterministic
head is fitted (MSE to the demo action), the log-std head keeps its initialisation and is tuned later by SAC.
Evaluation is read-only on fixed starts, per object, with the same `evaluate` used everywhere else."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapt import common  # noqa: E402
from adapt.env import ENV_RECORD, HORIZON, StateObs, make_env  # noqa: E402
from adapt.sac_ms3 import Actor, evaluate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--env", default="PickSingleYCBSplit-v1")
p.add_argument("--split", default=None)
p.add_argument("--data_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--iters", type=int, default=30_000)
p.add_argument("--batch_size", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--eval_every", type=int, default=5_000)
p.add_argument("--eval_rounds", type=int, default=2)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--standardize", type=int, default=1, help="z-score inputs with demo statistics (stored in the actor)")
p.add_argument("--mask", default="", help="comma list of observation groups the policy ignores, e.g. qvel,obj_lin_vel,obj_ang_vel")
p.add_argument("--val_frac", type=float, default=0.1, help="held-out episodes for the validation loss")
p.add_argument("--input_noise", type=float, default=0.1, help="train-time Gaussian input noise, in units of each feature's std")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
common.seed_all(args.seed)
common.gpu_guard()
device = torch.device("cuda")
if args.env == "PickSingleYCBSplit-v1":
    split = json.load(open(args.split)); objects = split["source"]
else:
    objects = ["cube"]
files = [os.path.join(args.data_dir, f"{o}.npz") for o in objects]
missing = [f for f in files if not os.path.exists(f)]
assert not missing, f"missing demo files: {missing}"
obs, act, val, per_obj = [], [], [], {}
rng = np.random.default_rng(args.seed)
for o, f in zip(objects, files):
    d = np.load(f); obs.append(d["obs"]); act.append(d["act"])
    n_ep = int(d["ep_len"].shape[0]); v_eps = rng.choice(n_ep, max(1, int(round(args.val_frac * n_ep))), replace=False)
    val.append(np.isin(d["ep_id"], v_eps))
    per_obj[o] = dict(episodes=n_ep, steps=int(len(d["obs"])), len_mean=float(d["ep_len"].mean()), val_episodes=int(len(v_eps)))
obs = torch.as_tensor(np.concatenate(obs), device=device); act = torch.as_tensor(np.concatenate(act), device=device)
val = torch.as_tensor(np.concatenate(val), device=device)
clipped = float((act.abs() > 1).float().mean()); act = act.clamp(-1, 1)
obs_tr, act_tr, obs_va, act_va = obs[~val], act[~val], obs[val], act[val]
print("demo data", tuple(obs.shape), "train", tuple(obs_tr.shape), "val", tuple(obs_va.shape), "per object", per_obj, f"fraction of action dims clipped {clipped:.4f}", flush=True)

eval_env = make_env(args.env, args.num_envs, objects if args.env != "PickCube-v1" else None, reward_mode="terminal")
per_env_obj = getattr(eval_env.base_env, "env_model_ids", ["cube"] * args.num_envs)
EVAL_SEEDS = [[20_000 + r * args.num_envs + i for i in range(args.num_envs)] for r in range(args.eval_rounds)]
actor = Actor(eval_env).to(device)
layout = StateObs.layout()
assert sum(len(v) for v in layout.values()) == obs.shape[1], obs.shape
mask = torch.ones(obs.shape[1], device=device)
for g in [g for g in args.mask.split(",") if g]:
    mask[layout[g]] = 0.0
if args.standardize:
    obs_mean, obs_std = obs_tr.mean(0), obs_tr.std(0) + 1e-6
else:
    obs_mean, obs_std = torch.zeros(obs.shape[1], device=device), torch.ones(obs.shape[1], device=device)
with torch.no_grad():
    actor.norm.obs_mean.copy_(obs_mean); actor.norm.obs_std.copy_(obs_std); actor.norm.obs_mask.copy_(mask)


class _Ag:  # minimal shell so the shared `evaluate` can be reused
    def __init__(self, a): self.actor = a


def eval_by_object():
    succ = np.concatenate([evaluate(_Ag(actor), eval_env, s, HORIZON) for s in EVAL_SEEDS])
    objs = np.array(per_env_obj * args.eval_rounds)
    per = {o: float(succ[objs == o].mean()) for o in sorted(set(objs))}
    return float(np.mean(list(per.values()))), per


opt = torch.optim.Adam(list(actor.backbone.parameters()) + list(actor.fc_mean.parameters()), lr=args.lr)
record = dict(args=vars(args), git=common.git_rev(), versions=common.versions(), env=ENV_RECORD, objects=objects,
              data=per_obj, action_clip_fraction=clipped, data_hashes={o: common.file_hash(f) for o, f in zip(objects, files)},
              obs_norm=dict(standardize=bool(args.standardize), masked_groups=[g for g in args.mask.split(",") if g], input_noise=args.input_noise))
curve, best, t0 = [], (-1.0, None), time.time()
for it in range(1, args.iters + 1):
    idx = torch.randint(0, obs_tr.shape[0], (args.batch_size,), device=device)
    x = obs_tr[idx]
    if args.input_noise > 0:
        x = x + args.input_noise * torch.randn_like(x) * obs_std
    loss = torch.nn.functional.mse_loss(actor.get_eval_action(x), act_tr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if it % args.eval_every == 0 or it == args.iters:
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(actor.get_eval_action(obs_va), act_va).item()
        mean, per = eval_by_object()
        curve.append(dict(iter=it, loss=loss.item(), val_loss=val_loss, success=mean, per_object=per, wall_s=time.time() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if mean > best[0]:
            best = (mean, {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()})
            common.atomic_torch_save({"actor": best[1], "record": record, "success": mean, "per_object": per, "iter": it},
                                     os.path.join(args.out, "bc_actor.pt"))
common.atomic_json(dict(curve=curve, best_success=best[0], record=record), os.path.join(args.out, "bc_log.json"))
print("BC done: best success", best[0])
