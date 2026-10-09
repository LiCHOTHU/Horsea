import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploration.events import Event, Option, Prefix, Tail
from exploration.feedback import Feedback, TrackedFrame
from exploration.gates import validate_gate
from exploration.memory import StructuredMemory
from exploration.metrics import summarize
from exploration.protocol import Protocol
from exploration.reference import actual_gripper
from exploration.run import load_prior
from exploration.trials import TrialManager
from exploration.vision import MarkerTracker, tool_point


def camera_frame(visible=True, distractor=False):
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    points = {"r": (40, 55, [255, 0, 0]), "g": (60, 55, [0, 255, 0]),
              "b": (40, 71, [0, 0, 255]), "c": (90, 30, [0, 255, 255]),
              "m": (106, 30, [255, 0, 255]), "y": (90, 46, [255, 255, 0])}
    if visible:
        for x, y, color in points.values():
            rgb[y-1:y+2, x-1:x+2] = color
    if distractor:
        rgb[85:99, 130:144] = [255, 0, 0]
    return {"left_eef": np.array([0., 0., 0., 1., 0., 0., 0.]), "left_gripper": .4,
            "cameras": {"camera": {"rgb": rgb, "depth_m": np.ones((120, 160)),
                                   "intrinsic": np.diag([200., 200., 1.]),
                                   "extrinsic": np.c_[np.eye(3), np.zeros(3)]}}}


class SensorContractTests(unittest.TestCase):
    def test_robot_logo_does_not_replace_marker(self):
        tracker = MarkerTracker()
        tracker.observe(camera_frame())
        first = np.asarray(tracker.last["door"])
        frame = tracker.observe(camera_frame(distractor=True))
        np.testing.assert_allclose(tracker.last["door"], first)
        self.assertAlmostEqual(frame.opening, 0.)

    def test_missing_reset_does_not_initialize_closed_pose_later(self):
        tracker = MarkerTracker()
        tracker.observe(camera_frame(visible=False))
        self.assertIsNone(tracker.observe(camera_frame()).opening)

    def test_measured_finger_joint_is_used_instead_of_drive_target(self):
        class Joint:
            def get_drive_target(self):
                raise AssertionError("Command target is not proprioception")
        joint = Joint()
        class Entity:
            def get_active_joints(self): return [joint]
            def get_qpos(self): return [.006]
        class Robot:
            left_entity = Entity()
            left_gripper = [(joint, 1., 0.)]
            left_gripper_scale = [0., .02]
            left_gripper_val = 0.
        self.assertAlmostEqual(actual_gripper(Robot(), "left"), .3)

    def test_tool_center_follows_measured_wrist_rotation(self):
        np.testing.assert_allclose(tool_point([0., 0., 0., 1., 0., 0., 0.]), [.12, 0., 0.])
        np.testing.assert_allclose(tool_point([0., 0., 0., 2**-.5, 0., 0., 2**-.5]), [0., .12, 0.], atol=1e-10)

    def test_stationary_part_is_not_ready_after_small_pull(self):
        before = TrackedFrame((0., 0., 0.), 1., (0., 0., 0.), 0., 1.)
        held = TrackedFrame((0., 0., 0.), .3, (0., 0., 0.), 0., 1.)
        moved = TrackedFrame((.01, 0., 0.), .3, (0., 0., 0.), 0., 1.)
        self.assertEqual(Feedback().prefix([before], [held, moved]), Prefix.NOT_READY)

    def test_verification_can_complete_gripper_closure(self):
        before = TrackedFrame((0., 0., 0.), 1., (0., 0., 0.), 0., 1.)
        held = TrackedFrame((0., 0., 0.), .85, (0., 0., 0.), 0., 1.)
        moved = TrackedFrame((.01, 0., 0.), .4, (.01, 0., 0.), .08, 1.)
        self.assertEqual(Feedback().prefix([before], [held, moved]), Prefix.READY)


class ExperimentContractTests(unittest.TestCase):
    def test_development_events_cannot_fit_prior(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "events.jsonl"
            path.write_text(json.dumps({"scene": Protocol().dev[0], "event": Event(Option(0, 0), Prefix.GOAL).record()}))
            with self.assertRaisesRegex(ValueError, "training"):
                load_prior(path, Protocol())

    def test_primary_success_is_not_the_observed_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            online = {"scene": 220000, "replicate": 0, "attempt": 3, "method": "prior", "phase": "exploit",
                      "event": Event(Option(0, 0), Prefix.READY, Tail.UNKNOWN).record(), "prefix_fingerprint": "p",
                      "execution_seed": 1, "prefix_steps": 4, "tail_steps": 3}
            truth = {"scene": 220000, "replicate": 0, "attempt": 3, "method": "prior", "backend": "robotwin",
                     "simulator_success": True}
            TrialManager.save(Path(d), [online], [truth])
            result = summarize(d)
            self.assertEqual(result["primary_actual_exploitation_success"]["prior"]["mean"], 1.)
            self.assertEqual(result["observed_goal_proxy"]["prior"]["mean"], 0.)
            self.assertFalse(result["exploration_pass"])

    def test_confirmation_requires_development_gate(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_gate(None, "unused")


if __name__ == "__main__":
    unittest.main()
