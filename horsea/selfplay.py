"""Training-side self-rollout data with hidden control conditions (protocol v2, sec. 3 and ablation 13.4).

Each episode has a hidden action-interface shift g, applied to the policy's unnormalized network
action (world-frame deltas; eecf is off) before execution:  executed = g(commanded).
  rotK   -- xy-translation rotated by K degrees      (outcome: observed eef motion vs command)
  grip_inv -- gripper command inverted                 (outcome: finger opening vs command)
  none   -- identity (null condition: memory should change nothing)
The policy (theta_0, real instruction) does not see g. Correction source = the verified oracle
theta_0 o g^-1: at every reached state its action is g^-1(a), a ~ pi_0(.|o), clipped to the valid
range, so executing it realises a. Labels are stored in the network's normalized action space.

Per generation call (receding horizon: 16 generated, first 8 executed) we record the real-
instruction features, the commanded chunk, the executed-prefix mask, the correction chunk and the
denoising states of the actual sampling path (for theta_0-collected data this path is also the
reference-policy probe rule). Per env step: proprio (eef pos, gripper qpos) and the executed
action. Episode boundary = first success (sticky) or the horizon.

    python -m horsea.selfplay verify  --tasks 0 6 11 --shifts rot90 rot180 rot45 grip_inv
    python -m horsea.selfplay collect --split writer_train --shifts none rot90 ... --attempts 2
"""
import argparse
import json
import math
import multiprocessing
import os
import time
import types

import numpy as np
import torch

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.manifest import FOLDS, WRITER_DEV, WRITER_TRAIN
from horsea.paths import BASE_CKPT, EXP
from horsea.rollout import make_runner
import imitation.envs.libero.wrappers as lw

K_PROBE = (0, 3, 6, 9)  # solver steps whose (z_k, t_k) are recorded
EXEC = 8                # executed prefix of each 16-step chunk
TRAIN_SHIFTS = ["none", "rot45", "rot-45", "rot90", "rot-90", "rot180", "grip_inv"]
DEV_SHIFTS = ["rot135", "rot-135", "rot90+grip_inv"]


def shift_params(name):
    """(3x3 matrix on xyz translation, gripper sign)."""
    M, s = np.eye(3), 1.0
    for part in name.split("+"):
        if part.startswith("rot"):
            a = math.radians(float(part[3:]))
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            M = R @ M
        elif part == "grip_inv":
            s = -s
        elif part != "none":
            raise ValueError(part)
    return M, s


def apply_shift(a, M, s, inverse=False):
    """a: (..., 7) unnormalized network actions (torch). Returns g(a) or g^-1(a)."""
    Mt = torch.as_tensor(np.linalg.inv(M) if inverse else M, dtype=a.dtype, device=a.device)
    out = a.clone()
    out[..., :3] = a[..., :3] @ Mt.T
    out[..., 6] = s * a[..., 6]
    return out


class Recorder:
    """Hooks a policy: records per-call data and applies per-env hidden shifts before execution."""

    def __init__(self, policy, flow, memory=None, state=None):
        self.policy, self.flow = policy, flow
        self.memory, self.state = memory, state
        self.mode = "novice"   # novice: execute g(a); oracle: execute g(clip(g^-1(a)))
        self.shifts = None     # list of (M, s) per env
        self.step = 0
        self.calls = None
        norm = policy.normalizer
        rec = self

        def sample_actions(self_, data):
            with torch.no_grad():
                encm = rec.flow.encode(data)
                B = encm.shape[0]
                zs = []
                resid = rec.memory is not None and type(rec.memory).__name__ == "ResidualMemory"
                field = (lambda z, t: rec.flow.decode(z, t, encm)) if (rec.memory is None or resid) else \
                    rec.memory.field(rec.flow, rec.state, encm)
                z = torch.randn(B, rec.flow.chunk, rec.flow.adim, device=encm.device)
                custom = getattr(rec, "custom", None)  # e.g. energy-memory path solve: (encm, noise) -> action
                if custom is not None:
                    a = custom(encm, z)
                    zs = [z] * len(K_PROBE)
                    return self_finish(a, zs, encm, B)
                dt = 1.0 / rec.flow.n_steps
                t = torch.zeros(B, device=encm.device)
                for k in range(rec.flow.n_steps):
                    if k in K_PROBE:
                        zs.append(z.clone())
                    z = z + dt * field(z, t)
                    t = t + dt
                a = torch.clamp(z, -1, 1)
                return self_finish(a, zs, encm, B)

        def resid_of(r):
            return r.memory is not None and type(r.memory).__name__ == "ResidualMemory" and getattr(r, "custom", None) is None

        def self_finish(a, zs, encm, B):
            with torch.no_grad():
                if resid_of(rec):  # final-action residual on the finished base sample
                    E = next(iter(rec.state.values())).shape[0]
                    a = torch.clamp(a + rec.memory._delta(rec.state, encm, a, E), -1, 1)
                un = norm.unnormalize({"actions": a})["actions"]
                lab = torch.stack([apply_shift(un[b], *rec.shifts[b], inverse=True) for b in range(B)]).clamp(-1, 1)
                lab = norm.normalize({"actions": lab})["actions"].clamp(-1, 1)
                for b in range(B):
                    rec.calls[b].append({"step": rec.step, "encm": encm[b].half().cpu(), "chunk": a[b].half().cpu(),
                                         "label": lab[b].half().cpu(),
                                         "probes": torch.stack([zz[b] for zz in zs]).half().cpu()})
                return a.cpu().numpy()

        orig_post = policy.postprocess_actions

        def postprocess_actions(self_, data):
            a = data["actions"]
            outs = []
            for b in range(a.shape[0]):
                # envs beyond the recorded batch (the policy reused for plain evaluation): no shift
                M, s = rec.shifts[b] if rec.shifts is not None and b < len(rec.shifts) else (np.eye(3), 1.0)
                ab = a[b]
                if rec.mode == "oracle":
                    ab = apply_shift(ab, M, s, inverse=True).clamp(-1, 1)
                outs.append(apply_shift(ab, M, s))
            data["actions"] = torch.stack(outs)
            return orig_post(data)

        policy.sample_actions = types.MethodType(sample_actions, policy)
        policy.postprocess_actions = types.MethodType(postprocess_actions, policy)
        policy.temporal_agg = False
        policy.action_horizon = EXEC
        policy.batch_size = None


def run_batch(runner, rec, task, init_ids, shifts, horizon=300, takeover_at=None, on_step=None):
    """One vectorized batch of episodes (one per init id / shift). Returns per-env records.
    A single episode is padded to two: a 1-env batch would render in-process, and EGL state
    created in-process breaks the forked env workers of every later batch."""
    pad = len(init_ids) == 1
    if pad:
        init_ids, shifts = list(init_ids) * 2, list(shifts) * 2
    B = len(init_ids)
    env_fn = lambda: lw.LiberoFrameStack(runner.env_factory(task_id=task, benchmark=runner.benchmark), 1)
    env = lw.LiberoVectorWrapper(env_fn, B)
    inits = runner.benchmark.get_task_init_states(task)[np.asarray(init_ids)]
    task_emb = {k: v.repeat(B, 1) for k, v in runner.benchmark.get_task_emb(task).items()}
    rec.shifts = [shift_params(s) for s in shifts]
    rec.calls = [[] for _ in range(B)]
    rec.mode = "novice"
    steps = [[] for _ in range(B)]
    succ, succ_step = [False] * B, [None] * B
    try:
        obs, info = env.reset(init_states=inits)
        rec.policy.reset()
        for step in range(horizon):
            if takeover_at is not None and step == takeover_at:
                rec.mode = "oracle"
            rec.step = step
            prop = np.concatenate([obs["robot0_eef_pos"][:, -1], obs["robot0_gripper_qpos"][:, -1]], 1)
            act = rec.policy.get_action(obs, task, **task_emb)
            obs, reward, term, trunc, info = env.step(act)
            for b in range(B):
                if succ_step[b] is None:
                    steps[b].append((prop[b].astype(np.float32), np.asarray(act[b], dtype=np.float32)))
                if info[b]["success"] and not succ[b]:
                    succ[b], succ_step[b] = True, step + 1
            if on_step is not None:  # e.g. within-episode memory writes after each executed chunk
                on_step(step + 1, steps, succ_step)
            if all(succ):
                break
        prop = np.concatenate([obs["robot0_eef_pos"][:, -1], obs["robot0_gripper_qpos"][:, -1]], 1)
    finally:
        try:
            env._env.close()
        except Exception:  # noqa: BLE001
            pass
    out = []
    for b in range(B):
        end = succ_step[b] if succ_step[b] is not None else len(steps[b])
        calls = [c for c in rec.calls[b] if c["step"] < end]
        for c in calls:
            c["n_exec"] = int(min(EXEC, end - c["step"]))
        props = np.stack([s[0] for s in steps[b][:end]] + [prop[b].astype(np.float32)])  # (end+1, 5)
        out.append({"task": task, "shift": shifts[b], "init": int(init_ids[b]), "success": bool(succ[b]),
                    "length": int(end), "calls": calls, "proprio": torch.from_numpy(props),
                    "executed": torch.from_numpy(np.stack([s[1] for s in steps[b][:end]])) if end else torch.zeros(0, 7),
                    "takeover_at": takeover_at, "mode_at_start": "novice"})
    return out[:1] if pad else out


def main():
    try:  # resumable job: let the kernel pick it first under memory pressure, not the lifecycle runs
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except OSError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "collect"])
    ap.add_argument("--tasks", type=int, nargs="*")
    ap.add_argument("--split", choices=["writer_train", "writer_dev"])
    ap.add_argument("--shifts", nargs="+", default=TRAIN_SHIFTS)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--n", type=int, default=4, help="verify: episodes per (task, shift, mode)")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tasks = args.tasks or (WRITER_TRAIN if args.split == "writer_train" else WRITER_DEV)
    policy, sd = load_policy(BASE_CKPT, args.device)
    shape_meta = sd["config"]["task"]["shape_meta"]
    # env factory + benchmark only. n_par >= 2 matters: LiberoRunner switches multiprocessing to "spawn"
    # only then; forked env workers inherit the parent's CUDA/EGL state and cannot create a render context.
    multiprocessing.set_start_method("spawn", force=True)
    runner = make_runner(shape_meta, "libero_90", max(2, args.par), max(2, args.par), 0, args.device)
    rec = Recorder(policy, Flow(policy))
    if args.cmd == "verify":
        out = os.path.join(EXP, "protocol_v2", "selfplay", "verify")
        os.makedirs(out, exist_ok=True)
        ids = FOLDS["adapt"][: args.n]
        for task in tasks:
            for sh in args.shifts:
                p = os.path.join(out, f"t{task}_{sh}.json")
                if os.path.exists(p):
                    continue
                res = {}
                for mode, tk in [("novice", None), ("oracle", 0), ("takeover64", 64), ("takeover128", 128)]:
                    eps = run_batch(runner, rec, task, ids, [sh] * len(ids), takeover_at=tk)
                    res[mode] = [e["success"] for e in eps]
                json.dump({"task": task, "shift": sh, "init_ids": ids, **res}, open(p, "w"))
                print(task, sh, {k: float(np.mean(v)) for k, v in res.items()}, flush=True)
        return
    # collect: attempt a uses adapt-fold start a for every shift (all shifts of one attempt run as one
    # parallel batch); one file per (task, shift) holding its attempts in order
    out = os.path.join(EXP, "protocol_v2", "selfplay", "data")
    os.makedirs(out, exist_ok=True)
    for task in tasks:
        todo = [sh for sh in args.shifts if not os.path.exists(os.path.join(out, f"t{task}_{sh}.pt"))]
        if not todo:
            continue
        t0 = time.time()
        per = {sh: [] for sh in todo}
        for a in range(args.attempts):
            init = FOLDS["adapt"][a]
            for e in run_batch(runner, rec, task, [init] * len(todo), todo):
                e["attempt"] = a
                per[e["shift"]].append(e)
        for sh in todo:
            torch.save({"task": task, "shift": sh, "episodes": per[sh], "k_probe": K_PROBE, "exec": EXEC,
                        "collector": "theta_0 (base, no memory), real instruction, receding horizon"},
                       os.path.join(out, f"t{task}_{sh}.pt"))
        print(task, {sh: [e["success"] for e in per[sh]] for sh in todo}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
