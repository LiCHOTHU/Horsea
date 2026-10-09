"""Shared staged controller for the instrumented microwave diagnostic.

The fixed skill is recorded once on a training scene. At execution, its robot
poses are translated using the RGB-D body markers. No object geometry, joint,
contact, or success query is available to the prefix or tail controller.
Scene construction/reset and evaluation are explicitly separate methods.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from .events import Prefix
from .feedback import Feedback
from .instrumentation import add_cameras, fixed_asset_environment
from .protocol import Protocol, atomic_json, digest
from .reference import sensors, save_sensors
from .robotwin import CountedScene, DEFAULT_ROOT, configuration, load_api
from .trials import PrefixResult, TailResult
from .vision import MarkerTracker


class RobotwinBackend:
    def __init__(self, protocol, reference, out, robotwin=DEFAULT_ROOT):
        self.protocol = protocol
        self.reference_path = Path(reference).resolve()
        self.reference = json.loads(self.reference_path.read_text())
        if not self.reference.get("ready_for_replay"):
            raise ValueError("A successful training-side instrumented skill recording is required")
        if self.reference["seed"] not in Protocol().train:
            raise ValueError("Reference must come from the training split")
        self.reference_hash = digest(self.reference)
        self.out = Path(out).resolve()
        self.api = load_api(robotwin)
        self.cfg = add_cameras(configuration(self.api, self.out))
        self.feedback = Feedback()
        self.env = None
        self.serial = 0

    def reset(self, scene, seed):
        """Fresh simulator world every attempt; no residual door or contact state."""
        import torch
        self.close()
        self.serial += 1
        self.scene, self.seed = int(scene), int(seed)
        self.attempt_dir = self.out / "sensors" / f"s{scene}_seed{seed}_trial{self.serial:05d}"
        self.attempt_dir.mkdir(parents=True, exist_ok=False)
        self.env = fixed_asset_environment(self.api)
        random.seed(scene)
        np.random.seed(scene)
        torch.manual_seed(scene)
        self.env.setup_demo(now_ep_num=0, seed=scene, is_test=True, **self.cfg)
        self.env.scene = CountedScene(self.env.scene)
        self.env._take_picture = lambda: None
        # Planner sampling is fresh per physical attempt, paired across methods.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.tracker = MarkerTracker()
        self.tracker.handle_local = self.reference["handle_local"]
        from .evaluation import AuditRecorder
        self.audit_recorder = AuditRecorder()
        self.frames, self.commands = [], []
        self.initial = self.capture("reset")
        body = self.tracker.last["body"]
        self.translation = None if body is None else (
            np.asarray(body)[:3, 3] - np.asarray(self.reference["initial_body"])[:3, 3])

    def capture(self, phase):
        data = sensors(self.env)
        tracked = self.tracker.observe(data)
        index = len(self.frames)
        path = self.attempt_dir / f"frame{index:03d}.npz"
        save_sensors(path, data)
        self.frames.append({"phase": phase, "tracked": asdict(tracked),
                            "vision": self.tracker.last, "sensor_file": str(path),
                            "physics_steps": self.env.scene.steps})
        self.audit_recorder.capture(self.env, phase)
        return tracked

    def translated_pose(self, pose, offset, pre_distance=0.):
        from scipy.spatial.transform import Rotation
        # This diagnostic varies object translation only, with fixed orientation.
        out = np.asarray(pose, dtype=float).copy()
        out[:3] += self.translation + offset
        if self.tilt:
            rotation = Rotation.from_quat(out[[4, 5, 6, 3]]).as_matrix()
            # RoboTwin's documented robot tool convention places its TCP 12 cm
            # along local x from the reported EEF. Rotate around that tool point,
            # using the stored robot command, not an object contact query.
            axis = np.cross(rotation[:, 0], [0., 0., 1.])
            axis /= max(np.linalg.norm(axis), 1e-9)
            adjusted = Rotation.from_rotvec(axis * self.tilt).as_matrix() @ rotation
            length = .12 + pre_distance
            out[:3] += length * (rotation[:, 0] - adjusted[:, 0])
            q = Rotation.from_matrix(adjusted).as_quat()
            out[3:] = q[[3, 0, 1, 2]]
        return out.tolist()

    def propose_grasps(self):
        if self.translation is None:
            return []
        return [{"grasp": r, "offset_m": [0., 0., 0.], "tilt_radians": np.pi / 6 * r,
                 "source": "fixed training skill aligned by RGB-D housing markers"}
                for r in range(self.protocol.grasps)]

    def command(self, action, phase):
        from envs.utils import Action, ArmTag
        item = dict(action, phase=phase)
        self.commands.append(item)
        if action["action"] == "move":
            act = Action("left", "move", target_pose=action["pose"])
        else:
            # Use close/open constructors: RoboTwin's generic gripper constructor
            # does not assign target_gripper_pos in this installed version.
            act = Action("left", "close", target_gripper_pos=float(action["gripper"]))
        self.env.move((ArmTag("left"), [act]))
        return self.capture(phase)

    def execute_prefix(self, grasp, rng):
        if not 0 <= grasp < self.protocol.grasps:
            raise ValueError(grasp)
        # This method has no mode argument; its RNG use and verification are shared.
        noise = rng.normal(0., .001, 3)
        specification = {"reference": self.reference_hash, "grasp": grasp,
                         "scene": self.scene, "execution_seed": self.seed,
                         "noise_m": noise.tolist(), "verification_m": .025,
                         "tilt_radians": np.pi / 6 * grasp}
        start = len(self.commands)
        frames_start = len(self.frames) - 1
        if not self.propose_grasps():
            return PrefixResult(Prefix.UNKNOWN, 0, digest(specification),
                                {"frames": self.frames[frames_start:], "commands": [],
                                 "reason": "housing markers unavailable"})
        self.offset = np.asarray(self.propose_grasps()[grasp]["offset_m"]) + noise
        self.tilt = self.propose_grasps()[grasp]["tilt_radians"]
        approach = [self.initial]
        for index, action in enumerate(self.reference["commands"][0]):
            action = dict(action)
            if action["action"] == "move":
                action["pose"] = self.translated_pose(action["pose"], self.offset, .08 if index == 0 else 0.)
            approach.append(self.command(action, "approach"))
        held = approach[-1]
        pose = np.asarray(self.env.get_arm_pose("left"), dtype=float)
        # A fixed small pull along the demonstrated first tail displacement.
        first_tail = next(a["pose"] for group in self.reference["commands"][1:]
                          for a in group if a["action"] == "move")
        first_grasp = [a["pose"] for a in self.reference["commands"][0] if a["action"] == "move"][-1]
        direction = np.asarray(first_tail[:3]) - first_grasp[:3]
        direction /= max(np.linalg.norm(direction), 1e-9)
        pose[:3] += .025 * direction
        verified = self.command({"action": "move", "pose": pose.tolist(), "gripper": None}, "verification")
        self.tail_start = verified
        event = self.feedback.prefix(approach, [held, verified])
        return PrefixResult(event, len(self.commands) - start, digest(specification),
                            {"frames": self.frames[frames_start:], "commands": self.commands[start:],
                             "specification": specification, "thresholds": self.feedback.configuration(),
                             "physics_steps": self.env.scene.steps})

    def execute_mode(self, option, rng):
        if not 0 <= option.mode < self.protocol.modes:
            raise ValueError(option)
        start, frames_start = len(self.commands), len(self.frames) - 1
        tracked = [self.tail_start]
        poses = []
        for group in self.reference["commands"][1:]:
            moving = [a["pose"] for a in group if a["action"] == "move"]
            if moving:
                poses.append(self.translated_pose(moving[-1], self.offset))
        poses = poses[:self.protocol.tail_control_budget]
        if not poses:
            raise ValueError("Training recording contains no manipulation poses")
        initial = np.asarray(self.env.get_arm_pose("left"), dtype=float)
        if option.mode in (1, 2):
            # Tangent pull versus vertical lift: distinct executable motions with
            # the same grasp and command allowance as the demonstrated arc.
            tangent = np.asarray(poses[min(2, len(poses)-1)][:3]) - initial[:3]
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            direction = tangent if option.mode == 1 else np.array([0., 0., 1.])
            poses = []
            for length in np.linspace(.015, .18, 18):
                pose = initial.copy()
                pose[:3] += length * direction
                poses.append(pose.tolist())
        # All modes consume one tail perturbation, from a separate explicit stream.
        perturb = rng.normal(0., .0005, 3)
        for pose in poses:
            pose = np.asarray(pose)
            pose[:3] += perturb
            tracked.append(self.command({"action": "move", "pose": pose.tolist(), "gripper": None}, "tail"))
            if self.feedback._goal([tracked[-1]]):
                break
        return TailResult(self.feedback.tail(tracked), len(self.commands) - start,
                          {"frames": self.frames[frames_start:], "commands": self.commands[start:],
                           "physics_steps": self.env.scene.steps})

    def evaluation(self):
        """Privileged audit route. Its output never reaches feedback or memory."""
        return dict(self.audit_recorder.result(self.env), reference_hash=self.reference_hash)

    def close(self):
        if self.env is not None:
            self.env.close_env()
            self.env = None
