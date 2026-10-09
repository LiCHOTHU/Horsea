"""Offline boundary tests for the RoboTwin Astra harness (no model calls).

    cd astra_explore && python3 tests/test_offline_rt.py
    ROBOTWIN=1 /home/licho/anaconda3/envs/robotwin/bin/python tests/test_offline_rt.py     # + environment test (needs DISPLAY)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from memory import History  # noqa: E402
from run import Budget  # noqa: E402
from run_rt import SmokePolicyRT, build_prompt, run_attempt  # noqa: E402
from schema_rt import Invalid, validate  # noqa: E402

HOLD = {"move": False, "position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0], "gripper": 1.0}
MOVE = {"move": True, "position_m": [-0.25, -0.05, 0.95], "quaternion_wxyz": [0.5, -0.5, 0.5, 0.5], "gripper": 1.0}


def good(**kw):
    r = {"left": dict(HOLD), "right": dict(HOLD), "stop": False, "assessment_of_previous_action": None, "intent": "x", "hypothesis": None,
         "expected_change": "y", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
    r.update(kw)
    return r


def test_validation():
    clean, w = validate(good(left=dict(MOVE)), set())
    assert clean["left"]["move"] and w == []
    bad = [good(left=dict(MOVE, position_m=[-0.25, -0.05, 2.0])), good(left=dict(MOVE, position_m=[0.9, 0, 0.9])),
           good(left=dict(MOVE, quaternion_wxyz=[1, 1, 1, 1])), good(left=dict(MOVE, gripper=1.5)), good(left=dict(MOVE, gripper=-0.1)),
           good(left=dict(MOVE, position_m=[float("nan"), 0, 0.9])), good(left=dict(MOVE, move="yes")), good(stop=1),
           {k: v for k, v in good().items() if k != "right"}, good(extra=1), good(left={"move": True})]
    for b in bad:
        try:
            validate(b, set())
            raise AssertionError(f"accepted invalid response {b}")
        except Invalid:
            pass
    clean, w = validate(good(left=dict(MOVE, quaternion_wxyz=[0.52, -0.5, 0.5, 0.5])), set())
    assert abs(np.linalg.norm(clean["left"]["quaternion_wxyz"]) - 1) < 1e-9 and any("normalised" in x for x in w)
    clean, w = validate(good(left=dict(HOLD, position_m=[9, 9, 9])), set())        # a held arm's position is ignored
    return "box bounds, quaternion norm, gripper range, types and keys enforced; near-unit quaternions normalised with a warning"


class FakeRT:
    def __init__(self, succeed_at=None):
        self.succeed_at, self.motions, self.n_acts, self._frames = succeed_at, 0, 0, []
        self.scene = type("S", (), {"steps": 0, "frames": self._frames})()
        self.step_lim = 400

    def _obs(self):
        img = np.zeros((24, 32, 3), np.uint8)
        pro = {a: {"ee_position_m_world": [0, 0, 0.9], "ee_quaternion_wxyz": [1, 0, 0, 0], "gripper_joint_0closed_1open": 1.0,
                   "finger_link_separation_cm": 13.9, "arm_joint_angles_rad": [0] * 6} for a in ("left", "right")}
        return {"images": {c: img for c in ("head_camera", "front_camera", "left_camera", "right_camera")}, "proprio": pro,
                "camera_axes": {c: {"+x": "right 1px", "+y": "up 1px", "+z": "up 1px"} for c in ("head_camera", "left_camera", "right_camera")},
                "motions_used": self.motions, "task_complete": self.succeed_at is not None and self.motions >= self.succeed_at}

    def reset(self):
        self.motions = 0
        return self._obs()

    def act(self, decision):
        self.motions += 1
        self.n_acts += 1
        o = self._obs()
        return o, {"planner": {a: {"status": "hold", "waypoints": 0} for a in ("left", "right")}, "sim_steps": 10, "wall_s": 0.0,
                   "arms": {}, "task_complete": o["task_complete"], "motions_used": self.motions}

    def privileged(self):
        return {"motions_used": self.motions}

    def frames(self):
        return self._frames


CFG = {"task": {"instruction": "fake"}, "video": False}


def test_retry_never_duplicates_execution():
    out = tempfile.mkdtemp()
    task = FakeRT()
    pol = SmokePolicyRT(script=[good(left=dict(MOVE, position_m=[5, 0, 0.9])), good(), "timeout", good()])
    s = run_attempt(task, pol, History(), 1, Budget(1, 6, 120), out, dict(CFG))
    d0 = json.load(open(Path(out) / "attempt_1" / "decision_00" / "decision.json"))
    assert len(d0["calls"]) == 2 and not d0["calls"][0]["accepted"] and d0["calls"][1]["accepted"] and d0["executed"]
    d1 = json.load(open(Path(out) / "attempt_1" / "decision_01" / "decision.json"))
    assert d1["calls"][0]["reason"] == "timeout" and d1["calls"][1]["accepted"]
    assert task.n_acts == 4 and s["outcome"] == "budget_exhausted" and s["model_calls"] == 6
    shutil.rmtree(out)
    return "invalid first response / timeout -> one retry, exactly one execution per accepted decision; budget end classified"


def test_outcomes():
    out = tempfile.mkdtemp()
    s = run_attempt(FakeRT(succeed_at=3), SmokePolicyRT(), History(), 1, Budget(1, 10, 120), out, dict(CFG))
    assert s["outcome"] == "success" and s["motions"] == 3 and s["model_calls"] == 3
    h = History()
    s = run_attempt(FakeRT(), SmokePolicyRT([good(stop=True)]), h, 1, Budget(1, 10, 120), out, dict(CFG))
    assert s["outcome"] == "model_stopped" and s["motions"] == 0 and h.interactions[-1]["stop"]
    shutil.rmtree(out)
    return "success, stop and budget exhaustion are distinct outcomes"


def test_prompt_has_no_privileged_fields():
    p = build_prompt({"instruction": "fake"}, FakeRT()._obs(), History().view(), {"x": 1}, "")
    for banned in ("container_pos", "plate_pos", "expert", "success predicate", "scene_info"):
        assert banned not in p, banned
    assert "planner" in p and "stop=true is NOT success" in p
    return "prompt carries only whitelisted fields and discloses the planner-assisted interface"


def test_environment_boundary():
    os.environ.setdefault("DISPLAY", ":0")
    from robotwin_env import RoboTwinTask
    t = RoboTwinTask("place_container_plate", 1700000)
    assert t.expert_solvable and t.instruction
    o1 = t.reset()
    assert set(o1) == {"images", "proprio", "camera_axes", "motions_used", "task_complete"} and not o1["task_complete"]
    assert set(o1["images"]) == {"head_camera", "front_camera", "left_camera", "right_camera"} and o1["images"]["head_camera"].shape == (480, 640, 3)
    priv = t.privileged()
    assert "container_pos" in priv and "plate_pos" in priv
    dec = {"left": dict(MOVE), "right": dict(HOLD)}
    o2, r = t.act(dec)
    assert r["planner"]["left"]["status"] == "Success" and r["arms"]["left"]["position_error_cm"] < 1.5 and r["arms"]["left"]["moved_cm"] > 10
    assert r["planner"]["right"]["status"] == "hold" and r["arms"]["right"]["moved_cm"] < 0.5 and t.motions == 1 and len(t.frames()) > 0
    o3, r2 = t.act({"left": dict(MOVE, gripper=0.0), "right": dict(HOLD)})
    assert r2["arms"]["left"]["finger_link_separation_cm"] < r["arms"]["left"]["finger_link_separation_cm"] - 1.5
    o4, r3 = t.act({"left": dict(MOVE, position_m=[-0.25, 0.55, 0.95]), "right": dict(HOLD)})    # unreachable
    assert r3["planner"]["left"]["status"] == "Fail" and r3["arms"]["left"]["moved_cm"] < 1.0
    o5 = t.reset()
    assert t.motions == 0 and np.array_equal(o1["images"]["head_camera"], o5["images"]["head_camera"])
    t.close()
    return (f"obs whitelist ok; expert-admitted seed; instruction {t.instruction!r}; planned motion error "
            f"{r['arms']['left']['position_error_cm']} cm; gripper closes ({r2['arms']['left']['finger_link_separation_cm']} cm); "
            f"planner failure leaves the arm still; reset bit-identical")


if __name__ == "__main__":
    tests = [test_validation, test_retry_never_duplicates_execution, test_outcomes, test_prompt_has_no_privileged_fields]
    if os.environ.get("ROBOTWIN"):
        tests.append(test_environment_boundary)
    for f in tests:
        print("PASS", f.__name__, "--", f(), flush=True)
