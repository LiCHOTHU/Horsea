"""Isaac Sim adapter for the Astra dex-hand pilot: EgoVLA Ego-Humanoid benchmark (Unitree H1 + two Inspire hands,
IsaacLab 2.3 / Isaac Sim 5.1, conda env `hongyi`), task Humanoid-Stack-Can-Into-Drawer-v0 as installed on this machine
(the drawer starts OPEN; the environment's success = the can rests on the saucer inside the drawer AND the drawer is closed).

Interface offered to the model (disclosed as IK-ASSISTED, no planner, no collision avoidance): per hand an absolute wrist
(hand-base link) target in the world frame - position plus the hand's finger direction and palm direction - tracked by the
benchmark's own differential-IK controllers (the same `DifferentialIKController` setup as the project's evaluation agent)
for `steps` control steps at 30 Hz, plus finger/thumb closure fractions mapped onto the 12 hand joints. This is the
project's native action (50 joint position targets) produced the way the project itself produces it from wrist targets.

Observation boundary (what the model may see): fixed main camera, the robot's left eye camera, both hand cameras (all
downscaled), wrist poses, fingertip positions, measured closure fractions, index-finger contact force, arm joint angles,
calibration-derived image directions, steps used/remaining and the terminal task_complete flag. Object poses, drawer joint
state, the subtask flags and the success predicate internals live only in `privileged()` / the evaluation-only log.

Self-calibration at start-up (no task information involved): the hand-base frame's finger and palm directions and the
fingertip offsets are measured by opening/closing the hands once in the air.
"""
import contextlib
import io
import os
import random
import sys
import time

import numpy as np
from PIL import Image

BENCH = "/home/licho/workspace/egovla_release/sim/Ego_Humanoid_Manipulation_Benchmark/source/extensions/humanoid.tasks"
SEED_MAP = {"Humanoid-Pour-Balls-v0": 0, "Humanoid-Sort-Cans-v0": 1, "Humanoid-Insert-Cans-v0": 2, "Humanoid-Unload-Cans-v0": 3,
            "Humanoid-Insert-And-Unload-Cans-v0": 4, "Humanoid-Push-Box-v0": 5, "Humanoid-Open-Drawer-v0": 6, "Humanoid-Close-Drawer-v0": 7,
            "Humanoid-Open-Laptop-v0": 8, "Humanoid-Flip-Mug-v0": 9, "Humanoid-Stack-Can-v0": 10, "Humanoid-Stack-Can-Into-Drawer-v0": 11}
# the project's instruction for this goal (LANGUAGE_MAPPING["Stack-Can-From-Drawer"]; the "Stack-Can-Into-Drawer" string
# says "Open the drawer" and predates this installation's env, whose drawer starts open and must end closed)
INSTRUCTION = {"Humanoid-Stack-Can-Into-Drawer-v0": "Put can on the saucer, and Close the drawer"}
# the project's evaluation horizon per task (egovla human_plan/ego_bench_eval/utils.py TASK_MAX_HORIZON, 30 Hz control steps)
TASK_MAX_HORIZON = {"Humanoid-Push-Box-v0": 400, "Humanoid-Open-Drawer-v0": 400, "Humanoid-Close-Drawer-v0": 400, "Humanoid-Flip-Mug-v0": 400,
                    "Humanoid-Pour-Balls-v0": 400, "Humanoid-Sort-Cans-v0": 900, "Humanoid-Insert-Cans-v0": 500, "Humanoid-Unload-Cans-v0": 500,
                    "Humanoid-Insert-And-Unload-Cans-v0": 900, "Humanoid-Stack-Can-v0": 300, "Humanoid-Open-Laptop-v0": 300,
                    "Humanoid-Stack-Can-Into-Drawer-v0": 400}
HANDS = ("left", "right")
# hand command = four closure fractions in [0, 1] per hand, mapped onto the Inspire hand's 12 joints (0 = open default,
# 1 = the joint limit farther from it): the real Inspire hand's actuation groups, with the finger base and tip separated
GROUPS = ("finger_base", "finger_tip", "thumb_rotate", "thumb_bend")


def joint_group(name):
    if "thumb" in name:
        return "thumb_rotate" if "yaw" in name else "thumb_bend"
    return "finger_base" if "proximal" in name else "finger_tip"


OPEN = (0.0, 0.0, 0.0, 0.0)
CAMS = {"main_camera": "main_camera", "left_eye_camera": "left_eye_camera", "left_hand_camera": "left_hand_camera", "right_hand_camera": "right_hand_camera"}
IMAGE_SIZES = {"main_camera": (960, 540), "left_eye_camera": (640, 360), "left_hand_camera": (640, 360), "right_hand_camera": (640, 360)}
CAM_HAND = {"main_camera": "right", "left_eye_camera": "left", "left_hand_camera": "left", "right_hand_camera": "right"}


def protocol_scene_index(task, room, table, num_trials=1, num_episodes=3, episode=0):
    """The project's ik_agent_30hz test protocol: seed_map[task] shuffles range(10000); test scenes start at index 100;
    `curr_random_idx += 1` precedes each reset. Returns the randomize_idx of evaluation episode `episode` of (room, table).
    (batch_script_30hz_h5_full.py evaluates rooms 1-5 x tables 1-5 except tables 1 of rooms 1-3, which were seen in
    training, so the first evaluation scene is room 1, table 2, episode 0.)"""
    random.seed(SEED_MAP[task])
    idx = list(range(10000))
    random.shuffle(idx)
    return idx[100 + (room * 5 + table) * num_trials * num_episodes + 1 + episode]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-9)


def _basis(a, b):
    """Orthonormal basis [a, b', a x b'] with b' = b made perpendicular to a."""
    a = _unit(a)
    b = np.asarray(b, dtype=float)
    b = _unit(b - np.dot(a, b) * a)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def rot_from_dirs(fingers_w, palm_w, f_b, p_b):
    """Rotation R (world <- hand base) with R f_b = fingers_w and R p_b = palm_w (after orthogonalisation)."""
    return _basis(fingers_w, palm_w) @ _basis(f_b, p_b).T


def mat_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_from_mat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def quat_angle_deg(q0, q1):
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0, 1))))


def _dir_text(du, dv):
    parts = []
    if abs(dv) >= 3:
        parts.append(f"{'down' if dv > 0 else 'up'} {abs(dv):.0f}px")
    if abs(du) >= 3:
        parts.append(f"{'right' if du > 0 else 'left'} {abs(du):.0f}px")
    return " and ".join(parts) if parts else "almost no change"


class IsaacDexTask:
    MAX_LIN_M = 0.01          # commanded wrist path: at most 1 cm ...
    MAX_ANG_DEG = 3.0         # ... and 3 degrees per 30 Hz control step
    MAX_CLOSURE_STEP = 1.0 / 30   # hand closure fractions change by at most 1/30 per step (full open <-> closed in 1 s)
    def __init__(self, task="Humanoid-Stack-Can-Into-Drawer-v0", room=1, table=2, scene_index=None, settle_steps=60,
                 image_sizes=None, video_every=2, video_size=(640, 360)):
        from isaaclab.app import AppLauncher  # must precede every other isaaclab import
        self.app = AppLauncher(headless=True, enable_cameras=True, device="cuda").app
        import gymnasium as gym
        import torch
        sys.path.insert(0, BENCH)
        import humanoid.tasks  # noqa: F401  registers the benchmark tasks
        import isaaclab_tasks  # noqa: F401
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.utils import math as M
        from isaaclab_tasks.utils import parse_env_cfg
        self.torch, self.M = torch, M
        self.task, self.room, self.table = task, room, table
        env_cfg = parse_env_cfg(task, num_envs=1)
        env_cfg.episode_length_s = 60                       # as in every project script (60 s = 1800 control steps)
        env_cfg.randomize = True
        env_cfg.spawn_background = True
        env_cfg.room_idx, env_cfg.table_idx = room, table
        self.env = gym.make(task, cfg=env_cfg)
        self.U = self.env.unwrapped
        self.scene_index = int(scene_index if scene_index is not None else protocol_scene_index(task, room, table))
        self.U.cfg.randomize_idx = self.scene_index
        self._sink = io.StringIO()                           # the task env prints its subtask flags every step
        self.robot, self.dev = self.U.robot, self.U.device
        self.arm_ids = {"left": list(self.U.cfg.left_arm_cfg.joint_ids), "right": list(self.U.cfg.right_arm_cfg.joint_ids)}
        self.hand_ids = {"left": list(self.U.cfg.left_hand_cfg.joint_ids), "right": list(self.U.cfg.right_hand_cfg.joint_ids)}
        self.hand_names = {s: [self.robot.joint_names[i] for i in self.hand_ids[s]] for s in HANDS}
        self.ee_idx = {"left": int(self.U.left_ee_idx), "right": int(self.U.right_ee_idx)}
        self.tip_idx = {"left": list(self.U.left_finger_tips_idx), "right": list(self.U.right_finger_tips_idx)}
        self.tip_names = {s: [self.robot.body_names[i] for i in self.tip_idx[s]] for s in HANDS}
        lim = self.robot.data.soft_joint_pos_limits[0].cpu().numpy()
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.default_q = self.robot.data.default_joint_pos[0].cpu().numpy()
        # closure direction per hand joint: the limit farther from the default (open) pose
        self.closed_q = np.where(self.hi - self.default_q >= self.default_q - self.lo, self.hi, self.lo)
        self.ik = {"left": DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=self.dev),
                   "right": DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="pinv"), num_envs=1, device=self.dev)}
        self.action = torch.zeros((1, self.U.num_actions), device=self.dev)
        self.cams = {n: getattr(self.U, n) for n in CAMS}
        self.image_sizes = dict(IMAGE_SIZES, **(image_sizes or {}))
        self.video_every, self.video_size = video_every, video_size
        self.max_episode_length = int(self.U.max_episode_length)
        self.settle_steps = settle_steps
        # decision steps per attempt: the project's evaluation horizon (never beyond the env's own auto-reset at max_episode_length)
        self.horizon = min(TASK_MAX_HORIZON.get(task, 400), self.max_episode_length - settle_steps - 10)
        self.instruction = INSTRUCTION.get(task, task)
        self.steps_used, self._frames, self.open = 0, [], False
        self.hand_cmd = {s: OPEN for s in HANDS}
        self.hand_cmd_prev = {s: OPEN for s in HANDS}
        self.calibration = self._calibrate()

    # ------------------------------------------------------------------ low level
    def _step(self, action):
        with contextlib.redirect_stdout(self._sink):
            obs = self.env.step(action)[0]
        self._sink.seek(0)
        self._sink.truncate()
        return obs

    def _reset_env(self):
        self.hand_cmd_prev = {s: OPEN for s in HANDS}      # env reset restores the open default hand pose
        with contextlib.redirect_stdout(self._sink):
            self.env.reset()
        self._sink.seek(0)
        self._sink.truncate()
        for s in HANDS:
            self.ik[s].reset()

    def _ee(self, side):
        return self.robot.data.body_state_w[0, self.ee_idx[side], 0:7].cpu().numpy().copy()

    def _tips(self, side):
        return self.robot.data.body_state_w[0, self.tip_idx[side], 0:3].cpu().numpy().copy()

    def _hand_q(self, side):
        return self.robot.data.joint_pos[0, self.hand_ids[side]].cpu().numpy().copy()

    def _hand_targets(self, side, cmd):
        out = []
        for i, n in zip(self.hand_ids[side], self.hand_names[side]):
            f = cmd[GROUPS.index(joint_group(n))]
            out.append(float(self.default_q[i] + f * (self.closed_q[i] - self.default_q[i])))
        return self.torch.tensor(out, device=self.dev)

    def _closure_measured(self, side):
        q = self._hand_q(side)
        fr = {g: [] for g in GROUPS}
        for k, (i, n) in enumerate(zip(self.hand_ids[side], self.hand_names[side])):
            span = self.closed_q[i] - self.default_q[i]
            if abs(span) > 1e-6:
                fr[joint_group(n)].append((q[k] - self.default_q[i]) / span)
        return {k: round(float(np.clip(np.mean(v), -0.2, 1.2)), 2) for k, v in fr.items() if v}

    def _ik_targets(self, side, goal_w):
        M, torch = self.M, self.torch
        root = self.robot.data.root_state_w[:, 0:7]
        jac = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx[side] - 1, :, self.arm_ids[side]]
        ee_w = self.robot.data.body_state_w[:, self.ee_idx[side], 0:7]
        q = self.robot.data.joint_pos[:, self.arm_ids[side]]
        ee_pos_b, ee_quat_b = M.subtract_frame_transforms(root[:, :3], root[:, 3:7], ee_w[:, :3], ee_w[:, 3:7])
        goal = torch.tensor(np.asarray(goal_w, dtype=np.float32), device=self.dev).unsqueeze(0)
        g_pos_b, g_quat_b = M.subtract_frame_transforms(root[:, :3], root[:, 3:7], goal[:, :3], goal[:, 3:7])
        self.ik[side].set_command(torch.cat([g_pos_b, g_quat_b], dim=-1), ee_pos_b, ee_quat_b)
        return self.ik[side].compute(ee_pos_b, ee_quat_b, jac, q)[0]

    def _contact(self, side):
        s = self.U.left_hand_contact_sensor if side == "left" else self.U.right_hand_contact_sensor
        return float(self.torch.norm(s.data.net_forces_w, dim=-1).max().item())

    def _run(self, steps, targets=None, hands=None, hold_q=None, record=False, stop_on_success=False):
        """Execute `steps` control steps: arms with a target are IK-tracked along a straight-line wrist path (position
        interpolated at <= MAX_LIN_M per step, orientation slerped at <= MAX_ANG_DEG per step, the way the project feeds
        its IK small per-step targets), others hold `hold_q`; hand closure fractions move toward their targets at
        <= MAX_CLOSURE_STEP per step. Returns (last obs, steps executed, success step or None)."""
        targets, hands = targets or {}, hands or {}
        hold_q = hold_q if hold_q is not None else self.robot.data.joint_pos.clone()
        start = {s: self.torch.tensor(self._hand_q(s), device=self.dev) for s in HANDS}
        goal = {s: (self._hand_targets(s, hands[s]) if s in hands else start[s]) for s in HANDS}
        path = {}
        for s, tgt in targets.items():
            p0 = self._ee(s)
            q1 = np.asarray(tgt[3:7], dtype=float)
            n = max(1, int(np.ceil(np.linalg.norm(np.asarray(tgt[:3]) - p0[:3]) / self.MAX_LIN_M)),
                    int(np.ceil(quat_angle_deg(p0[3:7], q1) / self.MAX_ANG_DEG)))
            path[s] = (p0, np.asarray(tgt, dtype=float), n)
        self.last_path_steps = {s: v[2] for s, v in path.items()}
        # closure ramp: as many steps as the largest commanded closure change needs at MAX_CLOSURE_STEP
        dmax = max([abs(hands[s][k] - self.hand_cmd_prev[s][k]) for s in hands for k in range(len(GROUPS))] + [0.0])
        ramp = max(1, int(np.ceil(dmax / self.MAX_CLOSURE_STEP)))
        self.last_closure_ramp_steps = ramp
        obs, done_at, executed = None, None, 0
        for i in range(steps):
            self.action[:] = 0
            a_hand = min(1.0, (i + 1) / ramp)
            for s in HANDS:
                if s in path:
                    p0, tgt, n = path[s]
                    a = min(1.0, (i + 1) / n)
                    cmd = np.concatenate([p0[:3] + a * (tgt[:3] - p0[:3]), quat_slerp(p0[3:7], tgt[3:7], a)])
                    self.action[0, self.arm_ids[s]] = self._ik_targets(s, cmd)
                else:
                    self.action[0, self.arm_ids[s]] = hold_q[0, self.arm_ids[s]]
                self.action[0, self.hand_ids[s]] = start[s] + a_hand * (goal[s] - start[s])
            obs = self._step(self.action)
            executed = i + 1
            if record:
                self.steps_used += 1
                if self.steps_used % self.video_every == 0:
                    self._frames.append(self._image("main_camera", self.video_size))
                if stop_on_success and bool(obs["success"][0].item()):
                    done_at = self.steps_used
                    break
        frac = min(1.0, executed / ramp) if executed else 0.0
        for s in hands:
            self.hand_cmd_prev[s] = tuple(self.hand_cmd_prev[s][k] + frac * (hands[s][k] - self.hand_cmd_prev[s][k]) for k in range(len(GROUPS)))
        return obs, executed, done_at

    # ------------------------------------------------------------------ self-calibration (robot geometry only)
    def _calibrate(self):
        self._reset_env()
        hold = self.robot.data.joint_pos.clone()
        self._run(self.settle_steps, hold_q=hold)
        cal = {}
        open_tips = {s: self._tips(s) for s in HANDS}
        open_ee = {s: self._ee(s) for s in HANDS}
        knuckles = {s: np.stack([self.robot.data.body_state_w[0, i, 0:3].cpu().numpy() for i, n in enumerate(self.robot.body_names)
                                 if n.startswith(s[0].upper() + "_") and "proximal" in n and "thumb" not in n]) for s in HANDS}
        self._run(30, hands={s: (0.5, 0.5, 0.0, 0.0) for s in HANDS}, hold_q=hold)       # half curl: the tips move toward the palm
        half_tips = {s: self._tips(s) for s in HANDS}
        self._run(30, hands={s: (0.5, 0.5, 0.5, 0.5) for s in HANDS}, hold_q=hold)       # pinch: thumb meets the index finger
        pinch_tips = {s: self._tips(s) for s in HANDS}
        self._run(30, hands={s: (1.0, 1.0, 1.0, 1.0) for s in HANDS}, hold_q=hold)       # fully closed on nothing
        closed_tips = {s: self._tips(s) for s in HANDS}
        closed_q = {s: self._closure_measured(s) for s in HANDS}
        self._run(30, hands={s: OPEN for s in HANDS}, hold_q=hold)
        for s in HANDS:
            R = mat_from_quat(open_ee[s][3:7])
            to_base = lambda P: R.T @ (np.asarray(P) - open_ee[s][:3])
            tips_b = {n: to_base(open_tips[s][i]) for i, n in enumerate(self.tip_names[s])}
            mean_open = to_base(open_tips[s].mean(0))
            mean_half = to_base(half_tips[s].mean(0))
            f_b = _unit(mean_open)
            curl = mean_half - mean_open
            p_b = _unit(curl - np.dot(curl, f_b) * f_b)
            thumb = next(v for k, v in tips_b.items() if "thumb" in k)
            index = next(v for k, v in tips_b.items() if "index" in k)
            pinky = next(v for k, v in tips_b.items() if "pinky" in k)
            cl_thumb = next(to_base(closed_tips[s][i]) for i, n in enumerate(self.tip_names[s]) if "thumb" in n)
            cl_index = next(to_base(closed_tips[s][i]) for i, n in enumerate(self.tip_names[s]) if "index" in n)
            pi_thumb = next(to_base(pinch_tips[s][i]) for i, n in enumerate(self.tip_names[s]) if "thumb" in n)
            pi_index = next(to_base(pinch_tips[s][i]) for i, n in enumerate(self.tip_names[s]) if "index" in n)
            lat = np.cross(f_b, p_b)
            t_b = lat if np.dot(thumb - mean_open, lat) >= 0 else -lat          # the side of the hand the thumb is on
            knuckle_f = float(np.mean([np.dot(to_base(k), f_b) for k in knuckles[s]])) if len(knuckles[s]) else float("nan")
            cal[s] = {"f_b": f_b.round(4).tolist(), "p_b": p_b.round(4).tolist(), "thumb_side_b": t_b.round(4).tolist(),
                      "knuckle_offset_along_fingers_cm": round(100 * knuckle_f, 1),
                      "fingertip_offset_cm": round(100 * float(np.linalg.norm(mean_open)), 1),
                      "fingertip_offset_along_fingers_cm": round(100 * float(np.dot(mean_open, f_b)), 1),
                      "fingertip_offset_along_palm_cm": round(100 * float(np.dot(mean_open, p_b)), 1),
                      "index_pinky_spread_cm": round(100 * float(np.linalg.norm(index - pinky)), 1),
                      "thumb_index_open_cm": round(100 * float(np.linalg.norm(thumb - index)), 1),
                      "thumb_index_closed_cm": round(100 * float(np.linalg.norm(cl_thumb - cl_index)), 1),
                      "thumb_index_pinch_cm": round(100 * float(np.linalg.norm(pi_thumb - pi_index)), 1),
                      "half_curl_tip_offset_along_fingers_cm": round(100 * float(np.dot(mean_half, f_b)), 1),
                      "closed_measured_fractions": closed_q[s],
                      "tips_in_base_frame_open": {n: v.round(3).tolist() for n, v in tips_b.items()},
                      "hand_joint_names": self.hand_names[s]}
        return cal

    # ------------------------------------------------------------------ permitted observation
    def _image(self, cam, size):
        px = self.cams[cam].data.output["rgb"][0, :, :, :3].cpu().numpy()
        im = Image.fromarray(np.ascontiguousarray(px, dtype=np.uint8))
        if size and tuple(size) != im.size:
            im = im.resize(tuple(size), Image.LANCZOS)
        return np.asarray(im).copy()

    def _project(self, cam, P, size):
        d = self.cams[cam].data
        K = d.intrinsic_matrices[0].cpu().numpy()
        R = mat_from_quat(d.quat_w_world[0].cpu().numpy())
        p = R.T @ (np.asarray(P, dtype=float) - d.pos_w[0].cpu().numpy())
        if p[0] <= 0.02:                          # IsaacLab 'world' camera convention: forward +x, left +y, up +z
            return None
        u = K[0, 0] * (-p[1] / p[0]) + K[0, 2]
        v = K[1, 1] * (-p[2] / p[0]) + K[1, 2]
        h, w = d.image_shape
        return u * size[0] / w, v * size[1] / h

    def _dirs_world(self, side):
        R = mat_from_quat(self._ee(side)[3:7])
        c = self.calibration[side]
        return R @ np.array(c["f_b"]), R @ np.array(c["p_b"]), R @ np.array(c["thumb_side_b"])

    def camera_axes(self):
        """Per camera: where a +5 cm move along each world axis appears, at a reference point the camera actually sees: the
        named hand's mean fingertip point when it projects inside the image (not too close to the lens), otherwise the point
        0.6 m along the camera's optical axis (camera geometry only; no object information)."""
        out = {}
        for cam in CAMS:
            side = CAM_HAND[cam]
            size = self.image_sizes[cam]
            d = self.cams[cam].data
            cam_pos = d.pos_w[0].cpu().numpy()
            base, label = self._tips(side).mean(0), f"{side} hand fingertips"
            uv0 = self._project(cam, base, size)
            if uv0 is None or not (0 <= uv0[0] < size[0] and 0 <= uv0[1] < size[1]) or np.linalg.norm(base - cam_pos) < 0.25:
                fwd = mat_from_quat(d.quat_w_world[0].cpu().numpy())[:, 0]          # IsaacLab 'world' convention: forward = +x
                base, label = cam_pos + 0.6 * fwd, "point 0.6 m straight ahead of this camera (image centre)"
                uv0 = self._project(cam, base, size)
            desc = {}
            for axis, name in enumerate(("+x", "+y", "+z")):
                q = base.copy()
                q[axis] += 0.05
                uv = self._project(cam, q, size)
                desc[name] = "not visible from this camera" if uv0 is None or uv is None else _dir_text(uv[0] - uv0[0], uv[1] - uv0[1])
            out[cam] = {"reference_point": label, **desc}
        return out

    def _observation(self):
        images = {cam: self._image(cam, self.image_sizes[cam]) for cam in CAMS}
        proprio = {}
        for s in HANDS:
            ee = self._ee(s)
            tips = self._tips(s)
            names = self.tip_names[s]
            thumb = tips[next(i for i, n in enumerate(names) if "thumb" in n)]
            index = tips[next(i for i, n in enumerate(names) if "index" in n)]
            fd, pd, td = self._dirs_world(s)
            proprio[s] = {"wrist_position_m_world": ee[:3].round(4).tolist(), "fingers_direction_world": fd.round(3).tolist(),
                          "palm_direction_world": pd.round(3).tolist(), "thumb_side_direction_world": td.round(3).tolist(),
                          "mean_fingertip_position_m_world": tips.mean(0).round(4).tolist(),
                          "thumb_tip_m_world": thumb.round(4).tolist(), "index_tip_m_world": index.round(4).tolist(),
                          "thumb_index_distance_cm": round(100 * float(np.linalg.norm(thumb - index)), 1),
                          "closure_commanded": dict(zip(GROUPS, self.hand_cmd[s])),
                          "closure_measured": self._closure_measured(s), "index_contact_force_N": round(self._contact(s), 2),
                          "arm_joint_angles_rad": self.robot.data.joint_pos[0, self.arm_ids[s]].cpu().numpy().round(3).tolist()}
        success = bool(self.U.success[0].item())
        return {"images": images, "proprio": proprio, "camera_axes": self.camera_axes(), "steps_used": self.steps_used,
                "steps_remaining": self.horizon - self.steps_used, "task_complete": success}

    def reset(self):
        self._reset_env()
        self.steps_used, self._frames, self.open = 0, [], True
        self.hand_cmd = {s: OPEN for s in HANDS}
        hold = self.robot.data.joint_pos.clone()
        self._run(self.settle_steps, hands={s: OPEN for s in HANDS}, hold_q=hold)
        self._frames.append(self._image("main_camera", self.video_size))
        return self._observation()

    # ------------------------------------------------------------------ one decision
    def act(self, decision):
        """decision: validated {left, right, steps}. Executes exactly once. Returns (obs, result)."""
        req = int(decision["steps"])
        remaining = self.horizon - self.steps_used
        steps = max(0, min(req, remaining))
        clip = None if steps == req else f"requested {req} steps, executed {steps}: attempt horizon ({self.horizon} steps) reached"
        before = {s: self._ee(s) for s in HANDS}
        hold = self.robot.data.joint_pos.clone()
        targets, quats = {}, {}
        for s in HANDS:
            d = decision[s]
            self.hand_cmd[s] = tuple(float(d[g]) for g in GROUPS)
            if d["move"]:
                R = rot_from_dirs(d["fingers_dir"], d["palm_dir"], self.calibration[s]["f_b"], self.calibration[s]["p_b"])
                quats[s] = quat_from_mat(R)
                targets[s] = np.array(list(d["position_m"]) + list(quats[s]), dtype=float)
        t0 = time.time()
        obs_raw, executed, success_step = self._run(steps, targets=targets, hands={s: self.hand_cmd[s] for s in HANDS}, hold_q=hold, record=True, stop_on_success=True)
        wall = time.time() - t0
        obs = self._observation()
        result = {"steps_requested": req, "steps_executed": executed, "clipped": clip, "wall_s": round(wall, 2), "hands": {},
                  "commanded_path_steps": dict(getattr(self, "last_path_steps", {})),
                  "task_complete": obs["task_complete"], "success_at_step": success_step, "steps_used": self.steps_used}
        for s in HANDS:
            after = self._ee(s)
            rec = {"moved_cm": round(100 * float(np.linalg.norm(after[:3] - before[s][:3])), 2),
                   "wrist_position_m_world": after[:3].round(4).tolist(),
                   "closure_commanded": dict(zip(GROUPS, self.hand_cmd[s])),
                   "closure_measured": obs["proprio"][s]["closure_measured"],
                   "thumb_index_distance_cm": obs["proprio"][s]["thumb_index_distance_cm"],
                   "index_contact_force_N": obs["proprio"][s]["index_contact_force_N"]}
            if s in targets:
                tgt = targets[s]
                Rt, Ra = mat_from_quat(tgt[3:7]), mat_from_quat(after[3:7])
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Rt.T @ Ra) - 1) / 2, -1, 1))))
                rec.update(tracked=True, target_position_m=tgt[:3].round(4).tolist(), target_quaternion_wxyz=tgt[3:7].round(4).tolist(),
                           position_error_cm=round(100 * float(np.linalg.norm(after[:3] - tgt[:3])), 2), orientation_error_deg=round(ang, 1))
            else:
                rec.update(tracked=False)
            result["hands"][s] = rec
        return obs, result

    # ------------------------------------------------------------------ evaluation-only record (never prompted)
    def privileged(self):
        U = self.U
        rec = {"steps_used": self.steps_used, "success": bool(U.success[0].item())}
        for name, ob in U.scene.rigid_objects.items():
            st = ob.data.root_state_w[0].cpu().numpy()
            rec[f"{name}_pos"] = st[:3].round(4).tolist()
            rec[f"{name}_quat_wxyz"] = st[3:7].round(4).tolist()
        for name, ar in U.scene.articulations.items():
            if name == "robot":
                continue
            rec[f"{name}_pos"] = ar.data.root_state_w[0, :3].cpu().numpy().round(4).tolist()
            rec[f"{name}_joint_pos"] = ar.data.joint_pos[0].cpu().numpy().round(4).tolist()
            rec[f"{name}_joint_limits"] = ar.data.joint_limits[0].cpu().numpy().round(4).tolist()
        if hasattr(U, "drawer_handle_offset"):
            rec["drawer_handle_pos"] = (U.drawer.data.root_state_w[0, :3] + U.drawer_handle_offset).cpu().numpy().round(4).tolist()
        for flag in ("reach_drawer_success", "reach_can_success", "lift_success", "move_door_success", "is_drawer_close"):
            if hasattr(U, flag):
                rec[flag] = float(getattr(U, flag)[0].item())
        return rec

    def describe(self):
        return {"simulator": "Isaac Sim 5.1 / IsaacLab 2.3 (EgoVLA Ego-Humanoid Manipulation Benchmark)", "robot": "Unitree H1 with two Inspire dexterous hands (12 joints each)",
                "task": self.task, "task_name": self.task, "room_idx": self.room, "table_idx": self.table, "scene_index": self.scene_index,
                "instruction": self.instruction, "control_hz": 30, "attempt_horizon_steps": self.horizon, "env_episode_steps": self.max_episode_length,
                "settle_steps": self.settle_steps,
                "action_type": "IK-assisted: absolute wrist targets (position + finger/palm directions) tracked by the benchmark's differential-IK "
                               "controllers for a chosen number of 30 Hz steps (wrist path <= 1 cm and 3 deg per step); four closure fractions per hand "
                               "(finger_base, finger_tip, thumb_rotate, thumb_bend) on the 12 hand joints, <= 1/30 per step",
                "cameras": {c: f"{self.image_sizes[c][0]}x{self.image_sizes[c][1]} (downscaled from {self.cams[c].data.image_shape[1]}x{self.cams[c].data.image_shape[0]})" for c in CAMS},
                "calibration": self.calibration}

    def frames(self):
        return self._frames

    def close(self):
        try:
            self.env.close()
        finally:
            self.app.close()
