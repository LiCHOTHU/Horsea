"""Section 5: one cumulative adaptation stream = (object, source seed, adaptation seed, method).

  A0  frozen pretrained policy: interacts (stochastic retries) but never updates
  A1  online adaptation: actor + critics updated from target data only
  A2  RLPD-style: same updates, critic/actor batches are 50 % target / 50 % source replay

The SAME object is kept for the whole stream; starts and goals follow the task's distribution.  Snapshots at
the budgets (true environment transitions) run a READ-ONLY evaluation on fixed independent starts; nothing in
evaluation touches policy, replay or statistics.  The stream continues to the full budget after the first
success; the first success is recorded separately.  Resumable: the checkpoint after each snapshot holds
networks, optimisers, target replay and RNG state, and finished snapshots are skipped on restart.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapt import common  # noqa: E402
from adapt.env import ENV_RECORD, HORIZON, make_env  # noqa: E402
from adapt.sac_ms3 import Agent, ReplayBuffer, evaluate, sac_update, sample_mixed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--object", required=True)
p.add_argument("--method", choices=["A0", "A1", "A2"], required=True)
p.add_argument("--init_dir", required=True, help="source-training output dir (adapt_init.pt, source_replay.pt)")
p.add_argument("--source_seed", type=int, required=True)
p.add_argument("--adapt_seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--budgets", default="0,500,2500,10000,25000")
p.add_argument("--eval_episodes", type=int, default=30)
p.add_argument("--eval_seed_base", type=int, default=10_000, help="fixed independent starts, shared by all cells")
p.add_argument("--utd", type=int, default=16, help="gradient updates per environment transition (RLPD ref: 16)")
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--learning_starts", type=int, default=50, help="transitions before the first update (1 attempt)")
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--tau", type=float, default=0.005)
p.add_argument("--retention_episodes_per_object", type=int, default=4)
p.add_argument("--time_cap_h", type=float, default=7.6)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
budgets = [int(b) for b in args.budgets.split(",")]
SNAP = os.path.join(args.out, "snapshots.json")
CKPT = os.path.join(args.out, "ckpt.pt")
snaps = json.load(open(SNAP)) if os.path.exists(SNAP) else {"snapshots": [], "complete": False}
if snaps["complete"]:
    print("already complete"); sys.exit(0)

timer = common.Timer()
common.gpu_guard()
seed = 1_000_000 * (args.source_seed + 1) + 1000 * args.adapt_seed + hash(args.object) % 997
common.seed_all(seed)
device = torch.device("cuda")
split = json.load(open(args.split))
assert args.object in split["dev"] + split["test"], f"{args.object} is not a held-out object"
assert args.object not in split["source"]

init_path, src_path = os.path.join(args.init_dir, "adapt_init.pt"), os.path.join(args.init_dir, "source_replay.pt")
init = torch.load(init_path, map_location=device, weights_only=False)
record = dict(args=vars(args), git=common.git_rev(), versions=common.versions(), env=ENV_RECORD, seed=seed,
              init_hash=common.file_hash(init_path), source_replay_hash=common.file_hash(src_path),
              split_hash=common.file_hash(args.split), source_record=init["record"],
              eval_seeds=[args.eval_seed_base + i for i in range(args.eval_episodes)],
              source_sampling="50/50 symmetric (RLPD reference)" if args.method == "A2" else "none",
              obs_normalization="none (identity; shared)")
record["config_hash"] = common.cfg_hash({k: v for k, v in record.items() if k not in ("versions",)})
common.atomic_json(record, os.path.join(args.out, "record.json"))

env = make_env("PickSingleYCBSplit-v1", 1, [args.object], reward_mode="terminal")
eval_env = make_env("PickSingleYCBSplit-v1", args.eval_episodes, [args.object], reward_mode="terminal")
ag = Agent(env, device, gamma=args.gamma, tau=args.tau)
ag.load_state_dict(init["agent"], optimizers=False)       # shared initialisation for every method
rb = ReplayBuffer(env, 1, budgets[-1] + HORIZON, device, device)
src = None
if args.method == "A2":
    d = torch.load(src_path, map_location=device, weights_only=False)["replay"]
    src = ReplayBuffer(env, d["num_envs"], d["obs"].shape[0] * d["num_envs"], device, device)
    src.load_state_dict(d)

st = {"t": 0, "attempt": 0, "stream_success": [], "first_success_transition": None, "first_success_attempt": None,
      "updates": 0, "wall_h": 0.0, "env_steps_total": 0}
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    ag.load_state_dict(ck["agent"]); rb.load_state_dict(ck["replay"]); common.set_rng_state(ck["rng"]); st = ck["st"]
    print("resumed at transition", st["t"])
done_budgets = {s["budget"] for s in snaps["snapshots"]}


def snapshot(B):
    succ = evaluate(ag, eval_env, record["eval_seeds"], HORIZON)
    entry = dict(budget=B, attempts=st["attempt"], updates=st["updates"], eval_success=float(succ.mean()),
                 eval_n=int(len(succ)), eval_successes=succ.tolist(),
                 stream_success_so_far=float(np.mean(st["stream_success"])) if st["stream_success"] else None,
                 first_success_transition=st["first_success_transition"],
                 first_success_attempt=st["first_success_attempt"], wall_h=st["wall_h"] + timer.wall() / 3600,
                 eval_sim_steps=int(len(succ) * HORIZON))
    if B == budgets[-1]:
        n = args.retention_episodes_per_object * len(split["source"])
        renv = make_env("PickSingleYCBSplit-v1", n, split["source"], reward_mode="terminal")
        rs = evaluate(ag, renv, [args.eval_seed_base + 5000 + i for i in range(n)], HORIZON)
        objs = np.array(renv.base_env.env_model_ids)
        entry["source_retention"] = {o: float(rs[objs == o].mean()) for o in split["source"]}
        entry["source_retention_mean"] = float(rs.mean())
        renv.close()
    snaps["snapshots"].append(entry)
    common.atomic_json(snaps, SNAP)
    common.atomic_torch_save({"agent": ag.state_dict(), "replay": rb.state_dict(), "rng": common.rng_state(),
                              "st": st | {"wall_h": st["wall_h"] + timer.wall() / 3600}}, CKPT)
    print(json.dumps({k: v for k, v in entry.items() if k != "eval_successes"}), flush=True)


obs, _ = env.reset(seed=seed + 1 + st["attempt"])
for B in budgets:
    if B in done_budgets:
        continue
    while st["t"] < B:
        if timer.wall() > args.time_cap_h * 3600:
            print("time cap; resubmit to continue"); sys.exit(3)
        with torch.no_grad():
            action, _, _ = ag.actor.get_action(obs)          # stochastic policy sampling (E0) for every arm
        next_obs, rew, term, trunc, infos = env.step(action)
        done = torch.logical_or(term, trunc)
        real_next = next_obs.clone()
        if "final_observation" in infos:
            real_next[done] = infos["final_observation"][done]
        rb.add(obs, real_next, action, rew, done.float())
        obs = next_obs
        st["t"] += 1
        if bool(done[0]):
            s = bool(infos["success_at_end"][0])
            st["attempt"] += 1
            st["stream_success"].append(s)
            if s and st["first_success_transition"] is None:
                st["first_success_transition"], st["first_success_attempt"] = st["t"], st["attempt"]
        if args.method != "A0" and st["t"] >= args.learning_starts:
            for _ in range(args.utd):
                sac_update(ag, sample_mixed(rb, src, args.batch_size))
                st["updates"] += 1
    assert st["t"] == B, (st["t"], B)
    snapshot(B)

snaps["complete"] = True
snaps["record"] = dict(transitions=st["t"], attempts=st["attempt"], updates=st["updates"],
                       stream_success=st["stream_success"], wall_h=st["wall_h"] + timer.wall() / 3600)
common.atomic_json(snaps, SNAP)
print("COMPLETE", snaps["record"] | {"stream_success": float(np.mean(st["stream_success"]))})
