"""LIBERO adapter for the Astra exploration pilot: fixed initial scene, whitelisted observations, native control.

Observation boundary: the model receives only the two camera images (upright), the robot's own measurements
(end-effector pose, finger opening, joint angles), calibration-derived image directions, the step counter and the
task's terminal completion flag. Object poses, the BDDL goal, rewards and evaluation internals are written ONLY to the
evaluation-only record (`privileged()`), which the prompt builder never reads.

Verified conventions (probe of 2026-10-09, robosuite 1.4.1, LIBERO OSC_POSE):
  action = [dx, dy, dz, drx, dry, drz, gripper] in [-1, 1]; translation deltas are world-frame (output_max 0.05 m per
  step, measured ~0.8-1.1 cm actual motion per unit step), rotation deltas about world axes (0.5 rad commanded,
  ~3 deg measured per step); gripper +1 closes, -1 opens, 0 holds (incremental command); control 20 Hz.
  Images arrive in OpenGL (upside-down) convention and are flipped here. robosuite's projection returns display
  (row, col) coordinates (verified against the rendered gripper).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera  # noqa: E402

SUITE_HORIZON = {"libero_90": 300, "libero_10": 520}        # the project's native per-attempt horizons (horsea/rollout.py)
CAMERAS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
CONTROL_DT = 0.05


def _dir_text(d_row, d_col):
    parts = []
    if abs(d_row) >= 3:
        parts.append(f"{'down' if d_row > 0 else 'up'} {abs(d_row):.0f}px")
    if abs(d_col) >= 3:
        parts.append(f"{'right' if d_col > 0 else 'left'} {abs(d_col):.0f}px")
    return " and ".join(parts) if parts else "almost no change"


class LiberoTask:
    def __init__(self, suite="libero_90", task_id=48, init_index=0, image_size=512, seed=0):
        bench = benchmark.get_benchmark_dict()[suite]()
        task = bench.get_task(task_id)
        self.suite, self.task_id, self.init_index, self.image_size = suite, task_id, init_index, image_size
        self.name, self.instruction = task.name, task.language
        self.bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        states = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
                            weights_only=False)
        self.init_state = np.asarray(states[init_index]).copy()
        self.env = OffScreenRenderEnv(bddl_file_name=self.bddl, camera_heights=image_size, camera_widths=image_size)
        self.env.seed(seed)
        self.horizon = SUITE_HORIZON[suite]
        self.step_count = 0
        self._raw = None

    # ------------------------------------------------------------------ permitted observation
    def _observation(self, raw):
        self._raw = raw
        images = {name: np.asarray(raw[key])[::-1].copy() for name, key in CAMERAS.items()}
        g = np.asarray(raw["robot0_gripper_qpos"], dtype=float)
        proprio = {"eef_position_m_world": np.round(raw["robot0_eef_pos"], 4).tolist(),
                   "eef_quaternion_xyzw": np.round(raw["robot0_eef_quat"], 4).tolist(),
                   "gripper_width_cm": round(float((g[0] - g[1]) * 100.0), 2),
                   "joint_positions_rad": np.round(raw["robot0_joint_pos"], 3).tolist()}
        return {"images": images, "proprio": proprio, "camera_axes": self.camera_axes(raw["robot0_eef_pos"]),
                "env_step": self.step_count, "task_complete": bool(self.env.check_success())}

    def camera_axes(self, eef_pos, length_m=0.05):
        """Calibration only: where a +5 cm move of the hand along each world axis would appear in each image."""
        sim, S = self.env.env.sim, self.image_size
        out = {}
        for cam, key in (("agentview", "agentview"), ("wrist", "robot0_eye_in_hand")):
            M = get_camera_transform_matrix(sim, key, S, S)
            base = project_points_from_world_to_camera(np.asarray(eef_pos)[None], M, S, S)[0]
            desc = {}
            for axis, name in enumerate(("+x", "+y", "+z")):
                p = np.asarray(eef_pos, dtype=float).copy()
                p[axis] += length_m
                q = project_points_from_world_to_camera(p[None], M, S, S)[0]
                desc[name] = _dir_text(q[0] - base[0], q[1] - base[1])
            out[cam] = desc
        return out

    def reset(self):
        self.env.reset()
        raw = self.env.set_init_state(self.init_state)
        self.step_count = 0
        return self._observation(raw)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        assert a.shape == (7,) and np.all(np.abs(a) <= 1.0 + 1e-9), "actions must be validated before execution"
        raw, reward, done, info = self.env.step(a)
        self.step_count += 1
        return self._observation(raw), bool(self.env.check_success()), bool(done)

    # ------------------------------------------------------------------ evaluation-only record (never prompted)
    def privileged(self):
        raw, e = self._raw, self.env.env
        rec = {"env_step": self.step_count, "success": bool(self.env.check_success()), "reward": float(e.reward()),
               "goal": e.parsed_problem["goal_state"]}
        for obj in e.objects_dict:
            if f"{obj}_pos" in raw:
                rec[f"{obj}_pos"] = np.round(raw[f"{obj}_pos"], 4).tolist()
        return rec

    def describe(self):
        e = self.env.env
        c = e.robots[0].controller
        return {"suite": self.suite, "task_id": self.task_id, "task_name": self.name, "instruction": self.instruction,
                "bddl": os.path.basename(self.bddl), "init_state_index": self.init_index,
                "robot": e.robots[0].name, "controller": type(c).__name__,
                "controller_output_max": c.output_max.tolist(), "controller_output_min": c.output_min.tolist(),
                "control_freq_hz": e.control_freq, "control_dt_s": CONTROL_DT, "env_internal_horizon": e.horizon,
                "attempt_horizon_steps": self.horizon, "cameras": {k: v for k, v in CAMERAS.items()},
                "image_size": self.image_size, "gripper": type(e.robots[0].gripper).__name__}

    def close(self):
        self.env.close()
