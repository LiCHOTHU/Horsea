"""RoboTwin adapter for the Astra exploration pilot (dual-arm Aloha, native `ee` action type).

Interface actually offered by RoboTwin and used here (disclosed as PLANNER-ASSISTED): one decision = absolute end-effector
target poses for both arms (world frame, [x, y, z, qw, qx, qy, qz]) plus a gripper target per arm in [0, 1]
(1 = open, 0 = closed); RoboTwin plans a joint path to each target with its own motion planner (robot-only collision
model: it does not know the table or the objects) and executes it to completion before the next observation. An
unreachable target fails to plan and that arm stays where it is. This is RoboTwin's `take_action(action, 'ee')`.

Observation boundary (what the model may see): head camera (640x480), front camera, both wrist cameras (320x240),
end-effector poses, measured finger separations and arm joint angles, calibration-derived image directions, motions
used, the terminal task_complete flag. Object poses, the success predicate internals and the expert check live only in
`privileged()` / the evaluation-only log.

Verified on this installation (probes of 2026-10-09): robot API poses are [x,y,z,qw,qx,qy,qz]; quaternion
(0.5,-0.5,0.5,0.5) points the gripper straight down with the fingers opening along world x; the end-effector reference
point is ~8.4 cm above the fingertips in that orientation; finger-link separation 13.9 cm open / 11.2 cm closed on
nothing; one planned motion ~0.2-3 s wall; head camera: +x -> image right, +y -> up (away), +z -> up.
"""
import os
import sys
import time

import numpy as np

ROBOTWIN = "/home/licho/workspace/RoboTwin"
for p in ("script", "", "policy", "policy/HorseaFM", "description/utils"):
    sys.path.insert(0, os.path.join(ROBOTWIN, p))

CAMS = ("head_camera", "front_camera", "left_camera", "right_camera")
FINGERS = {"left": ("fl_link7", "fl_link8"), "right": ("fr_link7", "fr_link8")}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema_rt import TOPDOWN_WXYZ  # noqa: E402


def _dir_text(du, dv):
    parts = []
    if abs(dv) >= 3:
        parts.append(f"{'down' if dv > 0 else 'up'} {abs(dv):.0f}px")
    if abs(du) >= 3:
        parts.append(f"{'right' if du > 0 else 'left'} {abs(du):.0f}px")
    return " and ".join(parts) if parts else "almost no change"


class CountedScene:
    """Proxy around the SAPIEN scene: counts physics steps and grabs a head-camera frame every `every` steps."""
    def __init__(self, scene, env, every=8):
        self._scene, self._env, self.every, self.steps, self.frames = scene, env, every, 0, []

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        r = self._scene.step()
        self.steps += 1
        if self.frames is not None and self.steps % self.every == 0:
            self._env._update_render()
            self._env.cameras.update_picture()
            self.frames.append(np.asarray(self._env.cameras.get_rgb()["head_camera"]["rgb"]).copy())
        return r


class RoboTwinTask:
    def __init__(self, task="place_container_plate", seed=1700000, task_config="astra_probe", instruction_seed=0):
        os.chdir(ROBOTWIN)
        import eval_policy as EP  # noqa: E402  RoboTwin's own evaluator helpers
        from fixed_scenes import build_args  # noqa: E402  the project's fixed-scene evaluator
        self.EP, self.task, self.seed = EP, task, seed
        self.args = build_args(task, task_config, "astra", policy_name="HorseaFM")
        self.args["render_freq"] = 0
        self.env = EP.class_decorator(task)
        # Admission + instruction, exactly as the project's fixed-scene protocol: the scripted expert must solve the seed
        # (privileged, done once before any model call, never exposed), then one instruction is drawn deterministically.
        self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
        info = self.env.play_once()
        self.expert_solvable = bool(self.env.plan_success and self.env.check_success())
        self.env.close_env()
        descriptions = EP.generate_episode_descriptions(task, [info["info"]], 1)
        rng = np.random.default_rng(instruction_seed)
        self.instruction = str(rng.choice(descriptions[0]["unseen"]))
        self.scene_info = info["info"]
        self.step_lim = int(self.env.step_lim)
        self.open = False
        self.motions = 0
        self.scene = None
        self._raw = None

    # ------------------------------------------------------------------ permitted observation
    def _finger_sep(self, arm):
        ent = self.env.robot.left_entity if arm == "left" else self.env.robot.right_entity
        pos = {l.get_name(): np.array(l.get_pose().p) for l in ent.get_links() if l.get_name() in FINGERS[arm]}
        a, b = FINGERS[arm]
        return float(np.linalg.norm(pos[a] - pos[b])) if a in pos and b in pos else None

    def _pose(self, arm):
        return [float(x) for x in (self.env.robot.get_left_ee_pose() if arm == "left" else self.env.robot.get_right_ee_pose())]

    def _project(self, raw, cam, pt):
        c = raw["observation"][cam]
        K, E = np.asarray(c["intrinsic_cv"]), np.asarray(c["extrinsic_cv"])
        p = E[:3, :3] @ np.asarray(pt, dtype=float) + E[:3, 3]
        if p[2] <= 0.02:
            return None
        uv = K @ p
        return uv[:2] / uv[2]

    def camera_axes(self, raw):
        out = {}
        for cam in ("head_camera", "left_camera", "right_camera"):
            arm = "left" if cam == "left_camera" else "right" if cam == "right_camera" else "left"
            pose = np.array(self._pose(arm))
            import transforms3d as t3d
            tip = pose[:3] + t3d.quaternions.quat2mat(pose[3:])[:, 0] * 0.084     # fingertip region along the approach axis
            base = self._project(raw, cam, tip)
            desc = {}
            for axis, name in enumerate(("+x", "+y", "+z")):
                q = tip.copy()
                q[axis] += 0.05
                uv = self._project(raw, cam, q)
                desc[name] = "not visible from this camera" if base is None or uv is None else _dir_text(uv[0] - base[0], uv[1] - base[1])
            out[cam] = desc
        return out

    def _observation(self):
        raw = self.env.get_obs()
        self._raw = raw
        images = {cam: np.asarray(raw["observation"][cam]["rgb"]).copy() for cam in CAMS}
        js = np.asarray(raw["joint_action"]["vector"], dtype=float)
        proprio = {}
        for arm, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            pose = self._pose(arm)
            sep = self._finger_sep(arm)
            proprio[arm] = {"ee_position_m_world": [round(x, 4) for x in pose[:3]],
                            "ee_quaternion_wxyz": [round(x, 4) for x in pose[3:]],
                            "gripper_joint_0closed_1open": round(float(js[sl][-1]), 3),
                            "finger_link_separation_cm": None if sep is None else round(100 * sep, 2),
                            "arm_joint_angles_rad": [round(float(x), 3) for x in js[sl][:-1]]}
        return {"images": images, "proprio": proprio, "camera_axes": self.camera_axes(raw), "motions_used": self.motions,
                "task_complete": bool(self.env.eval_success or self.env.check_success())}

    def reset(self):
        if self.open:
            self.env.close_env(clear_cache=True)
        self.env.setup_demo(now_ep_num=0, seed=self.seed, is_test=True, **self.args)
        self.env.set_instruction(instruction=self.instruction)
        self.env.eval_success = False
        self.scene = CountedScene(self.env.scene, self.env)
        self.env.scene = self.scene
        self.open, self.motions = True, 0
        return self._observation()

    # ------------------------------------------------------------------ one planned motion
    def act(self, decision):
        """decision: validated {left: {...}, right: {...}}. Executes exactly one RoboTwin `ee` action. Returns (obs, result)."""
        before = {arm: np.array(self._pose(arm)) for arm in ("left", "right")}
        targets, plan = {}, {}
        for arm in ("left", "right"):
            d = decision[arm]
            tgt = (d["position_m"] + d["quaternion_wxyz"]) if d["move"] else before[arm].tolist()
            targets[arm] = [float(x) for x in tgt]
            if d["move"]:
                r = (self.env.robot.left_plan_path if arm == "left" else self.env.robot.right_plan_path)(targets[arm])
                plan[arm] = {"status": r.get("status"), "waypoints": int(len(r["position"])) if "position" in r else 0}
            else:
                plan[arm] = {"status": "hold", "waypoints": 0}
        action = np.array(targets["left"] + [decision["left"]["gripper"]] + targets["right"] + [decision["right"]["gripper"]], dtype=float)
        s0, f0 = self.scene.steps, len(self.scene.frames)
        t0 = time.time()
        self.env.take_action(action, action_type="ee")
        wall = time.time() - t0
        self.motions += 1
        obs = self._observation()
        import transforms3d as t3d
        result = {"planner": plan, "sim_steps": self.scene.steps - s0, "wall_s": round(wall, 2), "arms": {},
                  "task_complete": obs["task_complete"], "motions_used": self.motions}
        for arm in ("left", "right"):
            after = np.array(self._pose(arm))
            tgt = np.array(targets[arm])
            q_err = t3d.quaternions.qmult(tgt[3:], t3d.quaternions.qinverse(after[3:]))
            ang = float(np.degrees(2 * np.arccos(np.clip(abs(q_err[0]), 0, 1))))
            result["arms"][arm] = {"target_position_m": [round(x, 4) for x in tgt[:3]], "achieved_position_m": [round(x, 4) for x in after[:3]],
                                   "position_error_cm": round(100 * float(np.linalg.norm(after[:3] - tgt[:3])), 2),
                                   "orientation_error_deg": round(ang, 1), "moved_cm": round(100 * float(np.linalg.norm(after[:3] - before[arm][:3])), 2),
                                   "gripper_target": decision[arm]["gripper"],
                                   "finger_link_separation_cm": obs["proprio"][arm]["finger_link_separation_cm"]}
        return obs, result

    # ------------------------------------------------------------------ evaluation-only record (never prompted)
    def privileged(self):
        rec = {"motions_used": self.motions, "sim_steps": self.scene.steps if self.scene else 0,
               "success": bool(self.env.eval_success or self.env.check_success())}
        for name in ("container", "plate", "can", "basket", "object", "target", "cup", "coaster", "pot", "microphone", "bottle1",
                     "bottle2", "hamburg", "frenchfries", "tray", "object1", "object2", "plasticbox"):
            obj = getattr(self.env, name, None)
            if obj is not None and hasattr(obj, "get_pose"):
                pose = obj.get_pose()
                rec[f"{name}_pos"] = [round(float(x), 4) for x in pose.p]
                rec[f"{name}_quat_wxyz"] = [round(float(x), 4) for x in pose.q]
        return rec

    def describe(self):
        return {"simulator": "RoboTwin 2.0 (SAPIEN)", "robot": "aloha-agilex dual arm", "task": self.task, "scene_seed": self.seed,
                "scene_info": self.scene_info, "instruction": self.instruction, "expert_solvable_seed": self.expert_solvable,
                "task_config": self.args["task_config"], "step_limit_motions": self.step_lim,
                "action_type": "ee: absolute end-effector targets executed by RoboTwin's motion planner (planner-assisted)",
                "pose_format": "[x, y, z, qw, qx, qy, qz] world frame", "gripper": "1 open, 0 closed",
                "topdown_quaternion_wxyz": list(TOPDOWN_WXYZ), "cameras": {"head_camera": "640x480", "front_camera": "320x240",
                                                                           "left_camera": "320x240 (left wrist)", "right_camera": "320x240 (right wrist)"}}

    def frames(self):
        return self.scene.frames if self.scene else []

    def close(self):
        if self.open:
            self.env.close_env(clear_cache=True)
            self.open = False
