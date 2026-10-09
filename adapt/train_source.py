"""Section 4: PickCube implementation check and shared source-skill training.

Phases (resumable through <out>/state.json; re-running a finished job is a no-op):
  train    baseline SAC on the dense task reward (allowed for SOURCE skills only; recorded), until
           --total_steps or the wall-clock cap.  Checkpoints agent + replay for resumption.
  gate     per-object success of the deterministic policy on fixed starts; pass if mean >= --gate
  collect  frozen actor, stochastic sampling, TERMINAL-reward labels, successes AND failures kept
           -> source_replay.pt   (skipped with --check_only)
  refit    critics re-fitted on the terminal-reward source replay with the actor frozen
           -> adapt_init.pt      (skipped with --check_only)
Exit codes: 0 done & gate passed | 3 wall-clock cap hit, resubmit to continue | 1 gate failed
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
from adapt.env import StateObs, ENV_RECORD, HORIZON, make_env  # noqa: E402
from adapt.sac_ms3 import Agent, ReplayBuffer, evaluate, sac_update  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--env", default="PickSingleYCBSplit-v1")
p.add_argument("--split", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--total_steps", type=int, default=3_000_000)
p.add_argument("--learning_starts", type=int, default=4_000)
p.add_argument("--training_freq", type=int, default=128, help="env transitions per update block (baseline: 64)")
p.add_argument("--utd", type=float, default=0.5)
p.add_argument("--batch_size", type=int, default=1024)
p.add_argument("--buffer_size", type=int, default=1_000_000)
p.add_argument("--gamma", type=float, default=0.8, help="ManiSkill SAC baseline default (dense source training)")
p.add_argument("--tau", type=float, default=0.01, help="ManiSkill SAC baseline default")
p.add_argument("--eval_freq", type=int, default=50_000)
p.add_argument("--eval_rounds", type=int, default=2, help="rounds of num_envs fixed-start episodes")
p.add_argument("--gate", type=float, default=0.8)
p.add_argument("--time_cap_h", type=float, default=7.4)
p.add_argument("--collect_episodes", type=int, default=2048)
p.add_argument("--refit_updates", type=int, default=50_000)
p.add_argument("--check_only", action="store_true")
p.add_argument("--init_actor", default=None, help="BC actor checkpoint (adapt/bc_train.py): skip dense RL, start at the gate")
p.add_argument("--refit_gamma", type=float, default=0.99, help="discount for the terminal-reward critic refit (and adaptation)")
p.add_argument("--bc_only", action="store_true", help="with --init_actor: no dense fine-tune if the BC policy fails the gate")
p.add_argument("--warm_steps", type=int, default=25_600, help="BC warm start: env steps collected with the frozen BC actor before any update")
p.add_argument("--warm_updates", type=int, default=5_000, help="BC warm start: critic-only updates on those steps before SAC proper")
p.add_argument("--obs_clip", type=float, default=0.0, help="clamp every obs feature to [-c, c]; 0 = off")
p.add_argument("--bootstrap_at_done", default="horizon", choices=["horizon", "always"], help="horizon: done=1 at H (fixed-horizon); always: baseline default, never stop bootstrapping")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
STATE = os.path.join(args.out, "state.json")
state = json.load(open(STATE)) if os.path.exists(STATE) else {"phase": "bc_gate" if args.init_actor else "train", "global_step": 0}
if state["phase"] == "done":
    print("already done:", state); sys.exit(0 if state.get("gate_pass", True) else 1)

timer = common.Timer()
common.gpu_guard()
common.seed_all(args.seed)
device = torch.device("cuda")
model_ids = None
if args.env == "PickSingleYCBSplit-v1":
    split = common.load_split(args.split) if hasattr(common, "load_split") else json.load(open(args.split))
    model_ids = split["source"]
record = dict(args=vars(args), git=common.git_rev(), versions=common.versions(), env=ENV_RECORD | dict(obs_clip=args.obs_clip, bootstrap_at_done=args.bootstrap_at_done),
              source_objects=model_ids, reward_for_source_training="normalized_dense (after BC)" if args.init_actor else "normalized_dense",
              init_actor=args.init_actor, init_actor_hash=common.file_hash(args.init_actor) if args.init_actor else None, refit_gamma=args.refit_gamma,
              source_stage=("BC on motion-planning demos" + ("" if args.bc_only else " + dense-reward SAC fine-tune if the BC policy fails the gate")) if args.init_actor else "dense-reward SAC",
              obs_normalization="none (identity; shared by all methods)",
              split_hash=common.file_hash(args.split) if args.split else None, config_hash=None)
record["config_hash"] = common.cfg_hash({k: v for k, v in record.items() if k != "versions"})
common.atomic_json(record, os.path.join(args.out, "record.json"))

envs = make_env(args.env, args.num_envs, model_ids, reward_mode="dense", seed=args.seed, obs_clip=args.obs_clip)
eval_envs = make_env(args.env, args.num_envs, model_ids, reward_mode="terminal", seed=args.seed, obs_clip=args.obs_clip)
EVAL_SEEDS = [[20_000 + r * args.num_envs + i for i in range(args.num_envs)] for r in range(args.eval_rounds)]
per_env_obj = getattr(envs.base_env, "env_model_ids", ["cube"] * args.num_envs)

ag = Agent(envs, device, gamma=args.gamma, tau=args.tau)
rb = ReplayBuffer(envs, args.num_envs, args.buffer_size, device, device)
CKPT = os.path.join(args.out, "train_ckpt.pt")
if args.init_actor and not os.path.exists(CKPT):
    bc = torch.load(args.init_actor, map_location=device, weights_only=False)
    ag.actor.load_state_dict(bc["actor"])
    critic_mask = ag.actor.norm.obs_mask.clone(); critic_mask[StateObs.layout()["qvel"]] = 1.0   # critics keep joint velocities
    ag.set_obs_norm(ag.actor.norm.obs_mean, ag.actor.norm.obs_std, ag.actor.norm.obs_mask, critic_mask)
    record["obs_norm"] = dict(actor_masked=[k for k, v in StateObs.layout().items() if ag.actor.norm.obs_mask[v].sum() == 0],
                              critic_masked=[k for k, v in StateObs.layout().items() if critic_mask[v].sum() == 0])
    print("loaded BC actor", args.init_actor, "BC success", bc.get("success"))
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    ag.load_state_dict(ck["agent"]); rb.load_state_dict(ck["replay"]); common.set_rng_state(ck["rng"])
    state = ck["state"]
    print("resumed", {k: v for k, v in state.items() if k != "eval_log"})


def save_train_ckpt():
    common.atomic_torch_save({"agent": ag.state_dict(), "replay": rb.state_dict(), "rng": common.rng_state(),
                              "state": state}, CKPT)
    common.atomic_json(state, STATE)


def eval_by_object():
    succ = np.concatenate([evaluate(ag, eval_envs, seeds, HORIZON) for seeds in EVAL_SEEDS])
    objs = np.array(per_env_obj * args.eval_rounds)
    per = {o: float(succ[objs == o].mean()) for o in sorted(set(objs))}
    return float(np.mean(list(per.values()))), per, int(len(succ))


def time_left():
    return timer.wall() < args.time_cap_h * 3600


# ------------------------------------------------------------------ train (dense) -------------------
# ------------------------------------------------------------------ BC warm start -------------------
if state["phase"] == "bc_gate":
    mean, per, n = eval_by_object()
    state.update(bc_gate_success=mean, bc_gate_per_object=per)
    print(f"BC GATE success {mean:.3f} (target {args.gate}) pass={mean >= args.gate} per-object {per}", flush=True)
    state["phase"] = "gate" if (mean >= args.gate or args.bc_only) else "warm"; save_train_ckpt()

if state["phase"] == "warm":
    # The BC actor is frozen; its own (stochastic) rollouts fill the replay buffer with dense-reward transitions and
    # the critics are fitted on them before SAC updates the actor, so the first actor gradients come from a
    # critic that has seen the policy, not from a random one.
    obs, _ = envs.reset(seed=args.seed + 7)
    while state["global_step"] < args.warm_steps:
        with torch.no_grad():
            actions, _, _ = ag.actor.get_action(obs)
        next_obs, rewards, term, trunc, infos = envs.step(actions)
        done = torch.logical_or(term, trunc); real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            real_next_obs[done] = infos["final_observation"][done]
        stop = done.float() if args.bootstrap_at_done == "horizon" else torch.zeros_like(done, dtype=torch.float32)
        rb.add(obs, real_next_obs, actions, rewards, stop); obs = next_obs
        state["global_step"] += args.num_envs
    for i in range(args.warm_updates):
        winfo = sac_update(ag, rb.sample(args.batch_size), update_actor=False)
        if i % 1000 == 0:
            print("warm critic", i, {k: round(float(v), 4) for k, v in winfo.items() if k in ("qf_loss", "q")}, flush=True)
    state["phase"] = "train"; state["warm"] = dict(steps=state["global_step"], updates=args.warm_updates); save_train_ckpt()

if state["phase"] == "train":
    grad_steps = int(args.training_freq * args.utd)
    steps_per_env = max(1, args.training_freq // args.num_envs)
    obs, _ = envs.reset(seed=args.seed + 1000 * (state["global_step"] > 0))
    state.setdefault("eval_log", [])
    last_eval = state.get("last_eval", -1)
    while state["global_step"] < args.total_steps:
        if not time_left():
            save_train_ckpt(); print(f"time cap at step {state['global_step']}; resubmit"); sys.exit(3)
        for _ in range(steps_per_env):
            if state["global_step"] < args.learning_starts:
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                with torch.no_grad():
                    actions, _, _ = ag.actor.get_action(obs)
            next_obs, rewards, term, trunc, infos = envs.step(actions)
            done = torch.logical_or(term, trunc)
            real_next_obs = next_obs.clone()
            if "final_observation" in infos:
                real_next_obs[done] = infos["final_observation"][done]
            stop = done.float() if args.bootstrap_at_done == "horizon" else torch.zeros_like(done, dtype=torch.float32)
            rb.add(obs, real_next_obs, actions, rewards, stop)
            obs = next_obs
            state["global_step"] += args.num_envs
        if state["global_step"] >= args.learning_starts:
            for _ in range(grad_steps):
                info = sac_update(ag, rb.sample(args.batch_size))
        if state["global_step"] // args.eval_freq > last_eval:
            last_eval = state["global_step"] // args.eval_freq
            mean, per, n = eval_by_object()
            entry = dict(step=state["global_step"], success=mean, per_object=per, n=n, wall_h=timer.wall() / 3600,
                         updates=ag.global_update, **({} if state["global_step"] < args.learning_starts else info))
            state["eval_log"].append(entry); state["last_eval"] = last_eval
            print(json.dumps(entry), flush=True)
            save_train_ckpt()
    state["phase"] = "gate"; save_train_ckpt()

# ------------------------------------------------------------------ gate ---------------------------
if state["phase"] == "gate":
    mean, per, n = eval_by_object()
    state.update(gate_success=mean, gate_per_object=per, gate_n=n, gate_pass=bool(mean >= args.gate))
    common.atomic_json(dict(success=mean, per_object=per, n=n, passed=state["gate_pass"], target=args.gate,
                            steps=state["global_step"], updates=ag.global_update),
                       os.path.join(args.out, "gate_source.json"))
    print(f"GATE source success {mean:.3f} (target {args.gate}) pass={state['gate_pass']} per-object {per}")
    common.atomic_torch_save({"agent": ag.state_dict(optimizers=False), "record": record,
                              "steps": state["global_step"]}, os.path.join(args.out, "dense_ckpt.pt"))
    state["phase"] = "done" if args.check_only else "collect"; save_train_ckpt()

# ------------------------------------------------------------------ collect (terminal labels) ------
if state["phase"] == "collect":
    cenv = eval_envs                      # same terminal-reward env as evaluation; avoids a third simulator instance
    src = ReplayBuffer(cenv, args.num_envs, args.collect_episodes * HORIZON + args.num_envs * HORIZON, device, device)
    rounds = int(np.ceil(args.collect_episodes / args.num_envs))
    obs, _ = cenv.reset(seed=args.seed + 7_000)
    succ_by_obj = {o: [] for o in set(per_env_obj)}
    with torch.no_grad():
        for r in range(rounds):
            for t in range(HORIZON):
                actions, _, _ = ag.actor.get_action(obs)
                next_obs, rew, term, trunc, infos = cenv.step(actions)
                done = torch.logical_or(term, trunc)
                real_next = next_obs.clone()
                if "final_observation" in infos:
                    real_next[done] = infos["final_observation"][done]
                src.add(obs, real_next, actions, rew, done.float())
                obs = next_obs
            s = infos["success_at_end"].cpu().numpy()
            for o, v in zip(per_env_obj, s):
                succ_by_obj[o].append(bool(v))
    coll = {o: float(np.mean(v)) for o, v in succ_by_obj.items()}
    common.atomic_torch_save({"replay": src.state_dict(), "episodes": rounds * args.num_envs,
                              "transitions": src.size, "reward": "terminal", "success_by_object": coll,
                              "record": record}, os.path.join(args.out, "source_replay.pt"))
    print(f"collected {src.size} terminal-reward transitions; stochastic-policy success {coll}")
    state["phase"] = "refit"; state["collect_success"] = coll; save_train_ckpt()

# ------------------------------------------------------------------ refit critics -------------------
if state["phase"] == "refit":
    ag.gamma = args.refit_gamma          # terminal reward at t<=H must reach the early steps
    d = torch.load(os.path.join(args.out, "source_replay.pt"), map_location=device, weights_only=False)
    src = ReplayBuffer(envs, args.num_envs, d["replay"]["obs"].shape[0] * args.num_envs, device, device)
    src.load_state_dict(d["replay"])
    for i in range(args.refit_updates):
        info = sac_update(ag, src.sample(256), update_actor=False)
        if i % 10_000 == 0:
            print("refit", i, info, flush=True)
    init = {"agent": ag.state_dict(optimizers=False), "record": record, "refit_updates": args.refit_updates, "refit_gamma": args.refit_gamma,
            "refit_batch": 256, "source_replay_hash": common.file_hash(os.path.join(args.out, "source_replay.pt")),
            "dense_steps": state["global_step"], "gate": state.get("gate_success")}
    common.atomic_torch_save(init, os.path.join(args.out, "adapt_init.pt"))
    print("saved adapt_init.pt", {k: v for k, v in init.items() if k not in ("agent", "record")})
    state["phase"] = "done"; save_train_ckpt()

state["wall_h_total"] = state.get("wall_h_total", 0) + timer.wall() / 3600
common.atomic_json(state, STATE)
print("DONE", {k: v for k, v in state.items() if k != "eval_log"})
sys.exit(0 if state.get("gate_pass", True) else 1)
