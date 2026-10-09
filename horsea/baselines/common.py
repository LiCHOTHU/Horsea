"""Shared protocol for the published-baseline adapters (DSRL, ReinFlow) on fm_policy_S in LIBERO-90.

One stream = (task, hidden shift, method, seed). The robot makes attempts from the adapt start fold (cycled);
the only learning signal is terminal success (an episode ends at its first success, as in selfplay.run_batch).
Snapshots at attempt budgets run a READ-ONLY evaluation on fixed starts; nothing in evaluation updates the
method. The stream continues after the first success, which is recorded separately. Resumable: the checkpoint
written after each snapshot holds the method state, RNG and counters; finished snapshots are skipped.

RL decision = one policy query: the base generates a 16-step chunk and the first EXEC=8 steps are executed open
loop (no temporal aggregation), exactly like the self-play collector. Methods plug in through `decide(encm, low)`
-> (normalized chunk (B, 16, 7), per-env info dicts), where encm (B, 4, 256) are the frozen encoder summaries
(horsea.base.Flow.encode) and low (B, 5) the raw proprio (eef pos, gripper qpos).
"""
import argparse
import json
import multiprocessing
import os
import sys
import types

import numpy as np
import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.manifest import FOLDS
from horsea.paths import BASE_CKPT
from horsea.rollout import make_runner
from horsea.selfplay import EXEC, apply_shift, shift_params
import imitation.envs.libero.wrappers as lw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adapt import common as C  # noqa: E402  (generic helpers only: atomic io, hashes, rng, timer)

HORIZON = 300


def protocol_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--task", type=int, required=True, help="libero_90 task id")
    ap.add_argument("--shift", default="none", help="hidden action-interface shift (selfplay.shift_params)")
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="0,5,20,50,100,200", help="cumulative training attempts (episodes)")
    ap.add_argument("--train_inits", default=",".join(map(str, FOLDS["adapt"])))
    ap.add_argument("--eval_inits", default=",".join(map(str, FOLDS["validation"])))
    ap.add_argument("--par", type=int, default=5, help="parallel envs per rollout batch (>= 2)")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--time_cap_h", type=float, default=7.6)
    ap.add_argument("--device", default="cuda:0")
    return ap


class Ctx:
    """Frozen base, its Flow, the LIBERO runner and the policy hook."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.policy, sd = load_policy(args.ckpt, args.device)
        self.flow = Flow(self.policy)
        multiprocessing.set_start_method("spawn", force=True)  # see selfplay.main: n_par >= 2 and spawn
        self.runner = make_runner(sd["config"]["task"]["shape_meta"], "libero_90", max(2, args.par),
                                  max(2, args.par), 0, args.device)
        self.shift = shift_params(args.shift)
        self.decide, self.calls, self.step = None, None, 0
        ctx, norm = self, self.policy.normalizer

        def sample_actions(self_, data):
            with torch.no_grad():
                low = torch.cat([data["obs"]["robot0_eef_pos"][:, -1], data["obs"]["robot0_gripper_qpos"][:, -1]],
                                -1).float().to(ctx.device).clone()
                encm = ctx.flow.encode(data)
                a, infos = ctx.decide(encm, low)
                for b in range(a.shape[0]):
                    ctx.calls[b].append(dict(step=ctx.step, encm=encm[b].float().cpu(), low=low[b].cpu(), **infos[b]))
                return a.cpu().numpy()

        orig_post = self.policy.postprocess_actions

        def postprocess_actions(self_, data):
            M, s = ctx.shift
            data["actions"] = torch.stack([apply_shift(a, M, s) for a in data["actions"]])
            return orig_post(data)

        self.policy.sample_actions = types.MethodType(sample_actions, self.policy)
        self.policy.postprocess_actions = types.MethodType(postprocess_actions, self.policy)
        self.policy.temporal_agg = False
        self.policy.action_horizon = EXEC
        self.policy.batch_size = None

    @torch.no_grad()
    def encode_obs(self, obs, task, task_emb):
        low = torch.cat([torch.as_tensor(obs["robot0_eef_pos"][:, -1]), torch.as_tensor(obs["robot0_gripper_qpos"][:, -1])],
                        -1).float()
        batch = self.policy._make_batch({k: v for k, v in obs.items()}, task, **task_emb)
        return self.flow.encode(batch).float().cpu(), low

    def rollout(self, task, init_ids, decide, horizon):
        """One vectorized batch of episodes. Returns per-env dicts: success, end (env steps), decisions
        (each with step, encm, low and the method's info), final_encm/final_low (observation after the last
        executed step of this env's episode, for bootstrapping truncated episodes)."""
        B = len(init_ids)
        assert B >= 2, "a 1-env batch renders in-process and breaks later forked workers (selfplay.run_batch)"
        env_fn = lambda: lw.LiberoFrameStack(self.runner.env_factory(task_id=task, benchmark=self.runner.benchmark), 1)
        env = lw.LiberoVectorWrapper(env_fn, B)
        inits = self.runner.benchmark.get_task_init_states(task)[np.asarray(init_ids)]
        task_emb = {k: v.repeat(B, 1) for k, v in self.runner.benchmark.get_task_emb(task).items()}
        self.decide, self.calls = decide, [[] for _ in range(B)]
        succ_step = [None] * B
        final = [None] * B
        try:
            obs, info = env.reset(init_states=inits)
            self.policy.reset()
            for step in range(horizon):
                self.step = step
                act = self.policy.get_action(obs, task, **task_emb)
                obs, reward, term, trunc, info = env.step(act)
                for b in range(B):
                    if succ_step[b] is None and info[b]["success"]:
                        succ_step[b] = step + 1
                if all(s is not None for s in succ_step):
                    break
            enc, low = self.encode_obs(obs, task, task_emb)
            for b in range(B):
                final[b] = (enc[b], low[b])
        finally:
            try:
                env._env.close() if hasattr(env, "_env") else env.close()
            except Exception:  # noqa: BLE001
                pass
        out = []
        for b in range(B):
            end = succ_step[b] if succ_step[b] is not None else horizon
            out.append(dict(init=int(init_ids[b]), success=succ_step[b] is not None, end=int(end),
                            decisions=[d for d in self.calls[b] if d["step"] < end],
                            final_encm=final[b][0], final_low=final[b][1]))
        return out


def transitions(ep):
    """Decision-level transitions of one episode: (decision, next (encm, low), reward, terminal).
    Reward 1 only on the decision whose executed prefix contains the first success; the episode then ends.
    Without success the last decision is truncated at the horizon (not terminal)."""
    ds, out = ep["decisions"], []
    for j, d in enumerate(ds):
        last = j == len(ds) - 1
        nxt = (ep["final_encm"], ep["final_low"]) if last else (ds[j + 1]["encm"], ds[j + 1]["low"])
        out.append((d, nxt, float(last and ep["success"]), bool(last and ep["success"])))
    return out


def run_stream(args, method, name):
    """Drive one stream: attempts in parallel batches, `method.observe(episodes)` after each batch,
    snapshots at the attempt budgets. `method` provides decide_train/decide_eval/observe/state_dict/
    load_state_dict/record and an `updates` counter."""
    os.makedirs(args.out, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]
    train_inits = [int(i) for i in args.train_inits.split(",")]
    eval_inits = [int(i) for i in args.eval_inits.split(",")]
    SNAP, CKPT = os.path.join(args.out, "snapshots.json"), os.path.join(args.out, "ckpt.pt")
    snaps = json.load(open(SNAP)) if os.path.exists(SNAP) else {"snapshots": [], "complete": False}
    if snaps["complete"]:
        print("already complete"); sys.exit(0)
    timer = C.Timer()
    record = dict(method=name, args=vars(args), git=C.git_rev(), versions=C.versions(), seed=args.seed,
                  ckpt_hash=C.file_hash(args.ckpt), exec_steps=EXEC, horizon=args.horizon,
                  reward="terminal success only (1 on the decision containing the first success; episode ends)",
                  method_config=method.record())
    C.atomic_json(record, os.path.join(args.out, "record.json"))
    st = {"attempt": 0, "transitions": 0, "stream_success": [], "first_success_attempt": None,
          "first_success_transition": None, "wall_h": 0.0}
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        method.load_state_dict(ck["method"]); C.set_rng_state(ck["rng"]); st = ck["st"]
        print("resumed at attempt", st["attempt"], flush=True)
    done = {s["budget"] for s in snaps["snapshots"]}

    def evaluate():
        succ = []
        for i in range(0, len(eval_inits), args.par):
            ids = eval_inits[i:i + args.par]
            pad = len(ids) == 1
            eps = method.ctx.rollout(args.task, ids * 2 if pad else ids, method.decide_eval, args.horizon)
            succ += [e["success"] for e in (eps[:1] if pad else eps)]
        return succ

    def snapshot(B):
        succ = evaluate()
        entry = dict(budget=B, attempts=st["attempt"], transitions=st["transitions"], updates=method.updates,
                     eval_success=float(np.mean(succ)), eval_n=len(succ), eval_successes=succ,
                     stream_success_so_far=float(np.mean(st["stream_success"])) if st["stream_success"] else None,
                     first_success_attempt=st["first_success_attempt"],
                     first_success_transition=st["first_success_transition"],
                     wall_h=st["wall_h"] + timer.wall() / 3600, **method.stats())
        snaps["snapshots"].append(entry)
        C.atomic_json(snaps, SNAP)
        C.atomic_torch_save({"method": method.state_dict(), "rng": C.rng_state(),
                             "st": st | {"wall_h": st["wall_h"] + timer.wall() / 3600}}, CKPT)
        print(json.dumps({k: v for k, v in entry.items() if k != "eval_successes"}), flush=True)

    for B in budgets:
        if B in done:
            continue
        while st["attempt"] < B:
            if timer.wall() > args.time_cap_h * 3600:
                print("time cap; resubmit to continue"); sys.exit(3)
            n = min(args.par, B - st["attempt"])
            ids = [train_inits[(st["attempt"] + k) % len(train_inits)] for k in range(n)]
            pad = n == 1
            eps = method.ctx.rollout(args.task, ids * 2 if pad else ids, method.decide_train, args.horizon)
            eps = eps[:1] if pad else eps
            for e in eps:
                st["attempt"] += 1
                st["transitions"] += e["end"]
                st["stream_success"].append(e["success"])
                if e["success"] and st["first_success_attempt"] is None:
                    st["first_success_attempt"], st["first_success_transition"] = st["attempt"], st["transitions"]
            method.observe(eps)
            print(f"attempt {st['attempt']} succ {[e['success'] for e in eps]} updates {method.updates}", flush=True)
        snapshot(B)
    snaps["complete"] = True
    snaps["record"] = dict(attempts=st["attempt"], transitions=st["transitions"], updates=method.updates,
                           stream_success=st["stream_success"], wall_h=st["wall_h"] + timer.wall() / 3600)
    C.atomic_json(snaps, SNAP)
    print("COMPLETE", json.dumps({k: v for k, v in snaps["record"].items() if k != "stream_success"}))
