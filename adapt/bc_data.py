"""Demonstrations for the long-term memory (BC source policy).

Per episode:  motion-planning solve on a pd_joint_pos CPU-sim env (ManiSkill's PickCube solver generalised to the
YCB object through its oriented-bounding-box grasp) -> the joint-space plan, time-subsampled by `--subsample`, is
converted to the plan's controller (pd_ee_delta_pose) with ManiSkill's `from_pd_joint_pos`, which is closed-loop on
a second env that carries the plan's observation wrapper (StateObs).  What is stored is exactly what BC trains on:
(StateObs vector, pd_ee_delta_pose action) pairs plus `--hold` terminal zero-motion steps, and the episode is kept
only if the task's `success` holds at its last step under the plan's controller.
`--verify` replays the stored actions from the stored start state and checks the observations reproduce.
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import sapien
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapt.env import HORIZON, StateObs  # noqa: E402  (registers PickSingleYCBSplit-v1)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (  # noqa: E402
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver  # noqa: E402
from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos  # noqa: E402


class ActLog(gym.Wrapper):
    def reset(self, **kw):
        self.actions = []
        return self.env.reset(**kw)

    def step(self, a):
        self.actions.append(np.array(a, dtype=np.float32).reshape(-1))
        return self.env.step(a)


class PairLog(gym.Wrapper):
    """Records (obs before step, action, info after step) on the target env.
    exec_noise > 0: DART-style - the *clean* action is recorded, a noisy one is executed; the converter re-plans
    the delta from the actual pose each step, so the labels are corrective and the data covers off-path states."""
    exec_noise = 0.0
    rng = np.random.default_rng(0)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.last_obs, self.obs, self.act, self.infos = obs, [], [], []
        return obs, info

    def refresh(self):
        self.last_obs = self.env.observation(self.env.unwrapped.get_obs())

    def step(self, a):
        self.obs.append(self.last_obs.clone())
        a = torch.as_tensor(a, dtype=torch.float32).reshape(1, -1)
        self.act.append(a.reshape(-1).clone())
        if self.exec_noise > 0:
            noisy = a.clone(); noisy[:, :-1] += torch.as_tensor(self.rng.normal(0, self.exec_noise, a.shape[-1] - 1), dtype=torch.float32)
            a = noisy.clamp(-1, 1)
        obs, r, te, tr, info = self.env.step(a)
        self.last_obs = obs
        self.infos.append(info)
        return obs, r, te, tr, info


def make_pair(env_id, obj):
    kw = dict(num_envs=1, obs_mode="state_dict", sim_backend="cpu", render_mode=None)
    if env_id == "PickSingleYCBSplit-v1":
        kw["model_ids"] = [obj]
    ori = ActLog(gym.make(env_id, control_mode="pd_joint_pos", **kw))
    tgt = PairLog(StateObs(gym.make(env_id, control_mode="pd_ee_delta_pose", **kw)))
    return ori, tgt


def target_object(e):
    return e.obj if hasattr(e, "obj") else e.cube


def plan(ori, seed, vel, gripper_t, offset):
    ori.reset(seed=seed)
    first_state = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in _deep_items(ori.unwrapped.get_state_dict())}
    e = ori.unwrapped
    planner = PandaArmMotionPlanningSolver(ori, debug=False, vis=False, base_pose=e.agent.robot.pose,
                                           visualize_target_grasp_pose=False, print_env_info=False,
                                           joint_vel_limits=vel, joint_acc_limits=vel)
    try:
        obb = get_actor_obb(target_object(e))
        approaching = np.array([0, 0, -1])
        target_closing = e.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        gi = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=0.025)
        grasp_pose = e.agent.build_grasp_pose(approaching, gi["closing"], gi["center"])
        r = planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -offset]))
        if r != -1:
            r = planner.move_to_pose_with_screw(grasp_pose)
        if r != -1:
            planner.close_gripper(t=gripper_t)
            r = planner.move_to_pose_with_screw(sapien.Pose(e.goal_site.pose.sp.p, grasp_pose.q))
        ok = isinstance(r, tuple) and bool(r[4]["success"].item())
    except Exception as ex:  # planner failures are data, not crashes
        print("plan error:", str(ex)[:100], flush=True)
        ok = False
    finally:
        planner.close()
    return ok, list(ori.actions), first_state


def _deep_items(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _deep_items(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _rebuild(flat):
    out = {}
    for path, v in flat.items():
        d = out
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = v
    return out


def convert(ori, tgt, seed, first_state, ori_actions, subsample, hold):
    st = _rebuild(first_state)
    ori.reset(seed=seed); ori.unwrapped.set_state_dict(st)
    tgt.reset(seed=seed); tgt.unwrapped.set_state_dict(st); tgt.refresh()
    sub = ori_actions[::subsample]
    if (len(ori_actions) - 1) % subsample:          # always keep the final waypoint
        sub = sub + [ori_actions[-1]]
    from_pd_joint_pos("pd_ee_delta_pose", sub, ori, tgt, render=False, pbar=None, verbose=False)
    hold_action = np.zeros(tgt.action_space.shape[-1], dtype=np.float32); hold_action[-1] = -1.0
    saved, tgt.exec_noise = tgt.exec_noise, 0.0        # hold steps are clean: the task needs a static robot at the end
    for _ in range(hold):
        tgt.step(torch.as_tensor(hold_action)[None])
    tgt.exec_noise = saved
    info = tgt.infos[-1]
    return dict(obs=torch.cat(tgt.obs).numpy(), act=torch.stack([a.reshape(-1) for a in tgt.act]).numpy(),
                success=bool(info["success"].item()), placed=bool(info["is_obj_placed"].item()), length=len(tgt.act))


def verify(tgt, seed, first_state, act):
    saved, tgt.exec_noise = tgt.exec_noise, 0.0
    tgt.reset(seed=seed); tgt.unwrapped.set_state_dict(_rebuild(first_state)); tgt.refresh()
    for a in act:
        tgt.step(torch.as_tensor(a)[None])
    tgt.exec_noise = saved
    obs = torch.cat(tgt.obs).numpy(); ref = act_obs_cache["obs"]
    pos = list(range(0, 9)) + list(range(18, 35)) + list(range(41, 51))   # everything except qvel and object velocities
    return float(np.abs(obs[:, pos] - ref[:, pos]).max()), bool(tgt.infos[-1]["success"].item())


act_obs_cache = {}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PickSingleYCBSplit-v1")
    p.add_argument("--object", default="cube")
    p.add_argument("--n", type=int, default=400, help="target number of successful demos")
    p.add_argument("--max_tries", type=int, default=None)
    p.add_argument("--seed0", type=int, default=100_000)
    p.add_argument("--vel", type=float, default=1.5, help="planner joint velocity/acceleration limit")
    p.add_argument("--gripper_t", type=int, default=6)
    p.add_argument("--offset", type=float, default=0.05)
    p.add_argument("--subsample", type=int, default=2)
    p.add_argument("--hold", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--verify", type=int, default=3, help="replay-check this many stored episodes")
    p.add_argument("--exec_noise", type=float, default=0.0, help="DART execution noise std (normalised action units) on arm dims")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.object if args.env != "PickCube-v1" else "cube"
    ori, tgt = make_pair(args.env, args.object)
    tgt.exec_noise = args.exec_noise; tgt.rng = np.random.default_rng(args.seed0)
    eps, stats, t0 = [], dict(tries=0, plan_ok=0, conv_ok=0, placed=0, lengths=[], over_horizon=0), time.time()
    seed = args.seed0
    while len(eps) < args.n and stats["tries"] < (args.max_tries or 3 * args.n):
        stats["tries"] += 1
        ok, acts, first = plan(ori, seed, args.vel, args.gripper_t, args.offset)
        if ok and len(acts) > 0:
            stats["plan_ok"] += 1
            ep = convert(ori, tgt, seed, first, acts, args.subsample, args.hold)
            stats["placed"] += ep["placed"]; stats["lengths"].append(ep["length"])
            stats["over_horizon"] += ep["length"] > HORIZON
            if ep["success"] and ep["length"] <= HORIZON:
                stats["conv_ok"] += 1
                ep.update(seed=seed, first_state={"/".join(k): v.numpy() for k, v in first.items()})
                eps.append(ep)
        seed += 1
        if stats["tries"] % 25 == 0:
            print(f"[{tag}] tries {stats['tries']} plan_ok {stats['plan_ok']} kept {len(eps)} "
                  f"len mean {np.mean(stats['lengths']) if stats['lengths'] else 0:.1f} {time.time()-t0:.0f}s", flush=True)
    # verification: stored actions reproduce stored observations and success from the stored start state
    ver = []
    for ep in eps[:args.verify]:
        act_obs_cache["obs"] = ep["obs"]
        fs = {tuple(k.split("/")): torch.as_tensor(v) for k, v in ep["first_state"].items()}
        ver.append(verify(tgt, ep["seed"], fs, ep["act"]))
    ori.close(); tgt.close()
    if eps:
        np.savez_compressed(os.path.join(args.out, f"{tag}.npz"),
                            obs=np.concatenate([e["obs"] for e in eps]).astype(np.float32),
                            act=np.concatenate([e["act"] for e in eps]).astype(np.float32),
                            ep_id=np.concatenate([[i] * e["length"] for i, e in enumerate(eps)]),
                            ep_len=np.array([e["length"] for e in eps]), seeds=np.array([e["seed"] for e in eps]))
    summary = dict(object=tag, env=args.env, kept=len(eps), tries=stats["tries"], plan_success=stats["plan_ok"] / max(1, stats["tries"]),
                   converted_success=stats["conv_ok"] / max(1, stats["plan_ok"]), placed_rate=stats["placed"] / max(1, stats["plan_ok"]),
                   length_mean=float(np.mean(stats["lengths"])) if stats["lengths"] else None,
                   length_max=int(max(stats["lengths"])) if stats["lengths"] else None, over_horizon=stats["over_horizon"],
                   verify_max_obs_diff=[v[0] for v in ver], verify_success=[v[1] for v in ver],
                   settings=dict(vel=args.vel, gripper_t=args.gripper_t, offset=args.offset, subsample=args.subsample, hold=args.hold, exec_noise=args.exec_noise),
                   seconds=time.time() - t0)
    json.dump(summary, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=1)
    print(json.dumps(summary))
