"""Environment wrapper (plan section 2-3): fixed object split, filtered state observation, terminal reward.

* PickSingleYCBSplit-v1: PickSingleYCB-v1 restricted to an explicit model list; sub-scene i gets model i % len.
* StateObs: the plan's observation for EVERY method -- joint pos/vel, gripper + tcp pose, object pose and
  motion, target position, object bounding-box extents, remaining time.  No object id, no is_grasped, no
  reward or progress fields.
* TerminalReward (over ManiSkillVectorEnv): r_t = 0 for t < H, r_H = 1[success at the final state];
  no early termination (ignore_terminations), auto-reset only at the horizon.  reward_mode="dense" passes the
  task's normalized dense reward through instead (source-skill acquisition only; recorded in every run).
"""
import os

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  (registers tasks)
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.utils.registration import register_env
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

HORIZON = 50
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json")


@register_env("PickSingleYCBSplit-v1", max_episode_steps=HORIZON, asset_download_ids=["ycb"])
class PickSingleYCBSplitEnv(PickSingleYCBEnv):
    """PickSingleYCB with the object list fixed by the experiment split instead of sampled from all of YCB."""

    def __init__(self, *args, model_ids=(), num_envs=1, reconfiguration_freq=0, **kwargs):
        assert len(model_ids) > 0, "model_ids must list the split's objects"
        self.split_model_ids = list(model_ids)
        super().__init__(*args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs)
        self.all_model_ids = np.array(self.split_model_ids)

    def _load_scene(self, options):
        # deterministic, balanced assignment: sub-scene i -> model i % K (instead of random choice over all YCB)
        self.all_model_ids = np.array(self.split_model_ids)
        ids = [self.split_model_ids[i % len(self.split_model_ids)] for i in range(self.num_envs)]
        self.env_model_ids = ids
        rng = self._batched_episode_rng

        class _Fixed:  # noqa: D401 - stand-in so the parent's `rng.choice(...)` yields our assignment
            def choice(self, *a, **k):
                return np.array(ids)

        self._batched_episode_rng = _Fixed()
        try:
            super()._load_scene(options)
        finally:
            self._batched_episode_rng = rng
        ext = {k: v for k, v in zip(*_extents_table())}
        self.env_bbox = torch.tensor([ext[m] for m in ids], dtype=torch.float32, device=self.device)


def _extents_table():
    from adapt.split import bbox_extents, info_path
    import json
    db = json.load(open(info_path()))
    keys = sorted(db)
    return keys, [bbox_extents(db[k]) for k in keys]


class StateObs(gym.ObservationWrapper):
    """Build the plan's state vector from obs_mode='state_dict'. Identical for every method."""

    KEYS = ["qpos", "qvel", "tcp_pose", "goal_pos", "obj_pose", "obj_lin_vel", "obj_ang_vel",
            "tcp_to_obj_pos", "obj_to_goal_pos", "bbox", "time_remaining"]

    DIMS = dict(qpos=9, qvel=9, tcp_pose=7, goal_pos=3, obj_pose=7, obj_lin_vel=3, obj_ang_vel=3, tcp_to_obj_pos=3,
                obj_to_goal_pos=3, bbox=3, time_remaining=1)

    @classmethod
    def layout(cls):
        """name -> indices in the plan's vector (used for the networks' fixed input mask)."""
        out, i = {}, 0
        for k in cls.KEYS:
            out[k] = list(range(i, i + cls.DIMS[k])); i += cls.DIMS[k]
        return out

    def __init__(self, env, clip=0.0, with_grasped=False, with_obj_vel=True, obj_vel_clip=None, with_qvel=True):
        self.base_env = env.unwrapped
        self.clip = float(clip)   # 0 = identity; otherwise every feature is clamped to [-clip, clip] (shared by all methods)
        self.with_grasped, self.with_obj_vel = with_grasped, with_obj_vel   # diagnostics only; defaults = the plan's vector
        self.obj_vel_clip = obj_vel_clip   # (lin, ang): object velocities clamped to +-c and divided by c (method-independent)
        self.with_qvel = with_qvel          # diagnostics only
        super().__init__(env)
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def _object(self):
        b = self.base_env
        return b.obj if hasattr(b, "obj") else b.cube

    def _bbox(self):
        b = self.base_env
        if hasattr(b, "env_bbox"):
            return b.env_bbox
        half = float(getattr(b, "cube_half_size", 0.02))
        return torch.full((b.num_envs, 3), 2 * half, device=b.device)

    def observation(self, obs):
        b = self.base_env
        o = self._object()
        ex = obs["extra"]
        vel = [o.linear_velocity, o.angular_velocity] if self.with_obj_vel else []
        if self.with_obj_vel and self.obj_vel_clip:
            vel = [torch.clamp(v, -c, c) / c for v, c in zip(vel, self.obj_vel_clip)]
        grasped = [ex["is_grasped"].float().unsqueeze(-1)] if self.with_grasped else []
        parts = [obs["agent"]["qpos"], *([obs["agent"]["qvel"]] if self.with_qvel else []), ex["tcp_pose"], ex["goal_pos"], ex["obj_pose"],
                 *vel, ex["tcp_to_obj_pos"], ex["obj_to_goal_pos"],
                 self._bbox(), ((HORIZON - b.elapsed_steps.float()) / HORIZON).unsqueeze(-1), *grasped]
        out = torch.cat([torch.as_tensor(p).float().to(b.device) for p in parts], dim=-1)
        return out.clamp(-self.clip, self.clip) if self.clip > 0 else out


class TerminalReward:
    """Wrap a ManiSkillVectorEnv: terminal success reward at the fixed horizon (or dense pass-through)."""

    def __init__(self, venv: ManiSkillVectorEnv, reward_mode="terminal"):
        assert reward_mode in ("terminal", "dense")
        self.reward_mode = reward_mode
        self.venv = venv
        self.num_envs = venv.num_envs
        self.single_observation_space = venv.single_observation_space
        self.single_action_space = venv.single_action_space
        self.action_space = venv.action_space
        self.observation_space = venv.observation_space
        self.base_env = venv.base_env

    def close(self):
        return self.venv.close()

    def reset(self, **kw):
        return self.venv.reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, infos = self.venv.step(action)
        done = torch.logical_or(term, trunc)
        if done.any():
            assert "final_info" in infos, "episodes must end at the horizon with auto-reset"
            success = infos["final_info"]["success"]
            el = infos["final_info"]["elapsed_steps"]
            assert bool((el[done] == HORIZON).all()), f"episode ended before the horizon: {el[done]}"
            infos["success_at_end"] = success & done
        else:
            infos["success_at_end"] = torch.zeros_like(done)
        if self.reward_mode == "terminal":
            rew = infos["success_at_end"].float()
        return obs, rew, term, trunc, infos


def make_env(env_id, num_envs, model_ids=None, reward_mode="terminal", seed=0, device="cuda", obs_clip=0.0):
    """GPU-sim vector env with the shared controller, observation and reward protocol."""
    kw = dict(obs_mode="state_dict", control_mode="pd_ee_delta_pose", robot_uids="panda",
              sim_backend="gpu", render_mode=None, num_envs=num_envs,
              reward_mode="normalized_dense" if reward_mode == "dense" else "sparse")
    if env_id == "PickSingleYCBSplit-v1":
        kw["model_ids"] = list(model_ids)
        kw["reconfiguration_freq"] = 0
    env = gym.make(env_id, **kw)
    env = StateObs(env, clip=obs_clip)
    venv = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, auto_reset=True, record_metrics=True)
    return TerminalReward(venv, reward_mode)


ENV_RECORD = dict(control_mode="pd_ee_delta_pose", robot="panda", obs=StateObs.KEYS, horizon=HORIZON,
                  success="PickSingleYCB native: object within 0.025 m of goal and robot static, at t=H",
                  early_termination=False)
