"""Offline boundary tests for the Isaac Sim dex-hand Astra harness (no model calls, no simulator).

    cd astra_explore && python3 tests/test_offline_dex.py
    ISAAC=1 python tests/test_offline_dex.py        # + environment test (env `hongyi`, launches Isaac Sim headless, ~3 min)
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
from run_dex import SmokePolicyDex, build_prompt, run_attempt  # noqa: E402
from schema_dex import Invalid, validate  # noqa: E402

CL = {"finger_base": 0.0, "finger_tip": 0.0, "thumb_rotate": 0.0, "thumb_bend": 0.0}
HOLD = dict({"move": False, "position_m": [0, 0, 0], "fingers_dir": [1, 0, 0], "palm_dir": [0, 0, -1]}, **CL)
MOVE = dict({"move": True, "position_m": [0.35, 0.2, 1.15], "fingers_dir": [1, 0, 0], "palm_dir": [0, 0, -1]}, **CL)
CLOSED = {"finger_base": 1.0, "finger_tip": 1.0, "thumb_rotate": 1.0, "thumb_bend": 1.0}


def good(**kw):
    r = {"left": dict(HOLD), "right": dict(HOLD), "steps": 20, "stop": False, "assessment_of_previous_action": None, "intent": "x",
         "hypothesis": None, "expected_change": "y", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
    r.update(kw)
    return r


def test_validation():
    clean, w = validate(good(left=dict(MOVE)), set())
    assert clean["left"]["move"] and w == [] and clean["steps"] == 20
    bad = [good(left=dict(MOVE, position_m=[0.35, 0.2, 2.5])), good(left=dict(MOVE, position_m=[1.5, 0, 1.0])),
           good(left=dict(MOVE, fingers_dir=[0, 0, 0])), good(left=dict(MOVE, palm_dir=[1, 0, 0])),          # palm parallel to fingers
           good(left=dict(MOVE, finger_base=1.5)), good(left=dict(MOVE, thumb_bend=-0.1)), good(left=dict(MOVE, position_m=[float("nan"), 0, 1.0])),
           {**good(), "left": {k: v for k, v in MOVE.items() if k != "thumb_rotate"}}, good(left=dict(MOVE, fingers=1.0)),
           good(left=dict(MOVE, move="yes")), good(stop=1), good(steps=0), good(steps=151), good(steps=20.0), good(steps=True),
           {k: v for k, v in good().items() if k != "right"}, good(extra=1), good(left={"move": True})]
    for b in bad:
        try:
            validate(b, set())
            raise AssertionError(f"accepted invalid response {b}")
        except Invalid:
            pass
    clean, w = validate(good(left=dict(MOVE, palm_dir=[0.3, 0, -1])), set())                  # 17 deg off perpendicular -> fixed + warning
    assert abs(np.dot(clean["left"]["fingers_dir"], clean["left"]["palm_dir"])) < 1e-9 and any("orthogonalised" in x for x in w)
    clean, w = validate(good(left=dict(MOVE, fingers_dir=[2, 0, 0])), set())                   # non-unit direction normalised silently
    assert abs(np.linalg.norm(clean["left"]["fingers_dir"]) - 1) < 1e-9
    clean, w = validate(good(left=dict(HOLD, position_m=[9, 9, 9], fingers_dir=[0, 0, 0])), set())   # a held hand's pose fields are ignored
    return "box bounds, direction validity/orthogonality, closure ranges, integer steps range, types and keys enforced"


class FakeDex:
    def __init__(self, succeed_at=None, horizon=100):
        self.succeed_at, self.horizon, self.steps_used, self.n_acts, self._frames = succeed_at, horizon, 0, 0, []

    def _obs(self):
        img = np.zeros((18, 32, 3), np.uint8)
        pro = {s: {"wrist_position_m_world": [0.2, 0.3, 1.1], "fingers_direction_world": [1, 0, 0], "palm_direction_world": [0, 0, -1],
                   "thumb_side_direction_world": [0, -1, 0], "mean_fingertip_position_m_world": [0.35, 0.3, 1.08], "thumb_tip_m_world": [0.3, 0.25, 1.08],
                   "index_tip_m_world": [0.36, 0.3, 1.08], "thumb_index_distance_cm": 8.0, "closure_commanded": dict(CL),
                   "closure_measured": dict(CL), "index_contact_force_N": 0.0, "arm_joint_angles_rad": [0] * 7} for s in ("left", "right")}
        cams = ("main_camera", "left_eye_camera", "left_hand_camera", "right_hand_camera")
        return {"images": {c: img for c in cams}, "proprio": pro,
                "camera_axes": {c: {"reference_point": "x", "+x": "right 1px", "+y": "up 1px", "+z": "up 1px"} for c in cams},
                "steps_used": self.steps_used, "steps_remaining": self.horizon - self.steps_used,
                "task_complete": self.succeed_at is not None and self.n_acts >= self.succeed_at}

    def reset(self):
        self.steps_used = 0
        return self._obs()

    def act(self, decision):
        req = decision["steps"]
        steps = min(req, self.horizon - self.steps_used)
        self.steps_used += steps
        self.n_acts += 1
        o = self._obs()
        return o, {"steps_requested": req, "steps_executed": steps, "clipped": None if steps == req else "horizon", "wall_s": 0.0,
                   "hands": {s: {"tracked": False, "moved_cm": 0.0, "thumb_index_distance_cm": 8.0, "index_contact_force_N": 0.0} for s in ("left", "right")},
                   "task_complete": o["task_complete"], "success_at_step": None, "steps_used": self.steps_used}

    def privileged(self):
        return {"steps_used": self.steps_used, "can_pos": [0.38, 0, 1.02]}

    def frames(self):
        return self._frames


CFG = {"task": {"instruction": "fake", "attempt_horizon_steps": 100, "calibration": {}}, "video": False}


def test_retry_never_duplicates_execution():
    out = tempfile.mkdtemp()
    task = FakeDex()
    pol = SmokePolicyDex(script=[good(left=dict(MOVE, position_m=[5, 0, 1.0])), good(), "timeout", good()])
    s = run_attempt(task, pol, History(), 1, Budget(1, 6, 120), out, dict(CFG))
    d0 = json.load(open(Path(out) / "attempt_1" / "decision_00" / "decision.json"))
    assert len(d0["calls"]) == 2 and not d0["calls"][0]["accepted"] and d0["calls"][1]["accepted"] and d0["executed"]
    d1 = json.load(open(Path(out) / "attempt_1" / "decision_01" / "decision.json"))
    assert d1["calls"][0]["reason"] == "timeout" and d1["calls"][1]["accepted"]
    assert task.n_acts == 4 and s["outcome"] == "budget_exhausted" and s["model_calls"] == 6 and s["steps_used"] == 50     # 20 + 20 + 5 + 5
    shutil.rmtree(out)
    return "invalid first response / timeout -> one retry, exactly one execution per accepted decision; budget end classified"


def test_outcomes():
    out = tempfile.mkdtemp()
    s = run_attempt(FakeDex(succeed_at=3), SmokePolicyDex(), History(), 1, Budget(1, 10, 120), out, dict(CFG))
    assert s["outcome"] == "success" and s["model_calls"] == 3 and s["steps_used"] == 15
    h = History()
    s = run_attempt(FakeDex(), SmokePolicyDex([good(stop=True)]), h, 1, Budget(1, 10, 120), out, dict(CFG))
    assert s["outcome"] == "model_stopped" and s["steps_used"] == 0 and h.interactions[-1]["stop"]
    s = run_attempt(FakeDex(horizon=50), SmokePolicyDex([good(steps=30), good(steps=30), good(steps=30)]), History(), 1, Budget(1, 10, 120), out, dict(CFG))
    d1 = json.load(open(Path(out) / "attempt_1" / "decision_01" / "decision.json"))
    assert s["outcome"] == "horizon_reached" and s["steps_used"] == 50 and s["model_calls"] == 2 and d1["steps_executed"] == 20 and d1["steps_clip_reason"]
    shutil.rmtree(out)
    return "success, stop, budget exhaustion and horizon (with logged step clipping) are distinct outcomes"


def test_prompt_has_no_privileged_fields():
    p = build_prompt(CFG["task"], FakeDex()._obs(), History().view(), {"x": 1}, "")
    for banned in ("can_pos", "plate_pos", "drawer_joint", "handle_pos", "expert", "success predicate", "reach_drawer", "lift_success"):
        assert banned not in p, banned
    assert "inverse-kinematics" in p and "stop=true is NOT success" in p and "collision avoidance" in p
    return "prompt carries only whitelisted fields and discloses the IK-assisted interface"


def test_environment_boundary():
    from isaac_env import IsaacDexTask
    t = IsaacDexTask(settle_steps=30)
    assert t.instruction and t.horizon == 400 and set(t.calibration) == {"left", "right"}
    cal = t.calibration["left"]
    assert 8 < cal["fingertip_offset_cm"] < 25 and cal["thumb_index_closed_cm"] < cal["thumb_index_open_cm"]
    o1 = t.reset()
    assert set(o1) == {"images", "proprio", "camera_axes", "steps_used", "steps_remaining", "task_complete"} and not o1["task_complete"]
    assert set(o1["images"]) == {"main_camera", "left_eye_camera", "left_hand_camera", "right_hand_camera"} and o1["images"]["main_camera"].shape == (540, 960, 3)
    priv = t.privileged()
    assert "can_pos" in priv and "drawer_joint_pos" in priv
    start = np.array(o1["proprio"]["left"]["wrist_position_m_world"])
    tgt = (start + np.array([0.0, 0.0, 0.10])).tolist()
    o2, r = t.act({"left": dict(MOVE, position_m=tgt, fingers_dir=o1["proprio"]["left"]["fingers_direction_world"], palm_dir=o1["proprio"]["left"]["palm_direction_world"]),
                   "right": dict(HOLD), "steps": 45})
    assert r["hands"]["left"]["tracked"] and r["hands"]["left"]["position_error_cm"] < 2.0 and r["hands"]["left"]["moved_cm"] > 8
    assert not r["hands"]["right"]["tracked"] and r["hands"]["right"]["moved_cm"] < 1.0 and t.steps_used == 45 and len(t.frames()) > 10
    o3, r2 = t.act({"left": dict(HOLD, **CLOSED), "right": dict(HOLD), "steps": 35})
    assert r2["hands"]["left"]["thumb_index_distance_cm"] < r["hands"]["left"]["thumb_index_distance_cm"] - 2
    o4 = t.reset()
    assert t.steps_used == 0 and np.abs(o1["images"]["main_camera"].astype(int) - o4["images"]["main_camera"].astype(int)).mean() < 1.0
    msg = (f"obs whitelist ok; horizon {t.horizon}; instruction {t.instruction!r}; IK error {r['hands']['left']['position_error_cm']} cm after 45 steps; "
           f"hand closes ({r2['hands']['left']['thumb_index_distance_cm']} cm); reset reproducible")
    print("PASS test_environment_boundary --", msg, flush=True)       # Isaac's app.close() ends the process
    t.close()
    return msg


if __name__ == "__main__":
    tests = [test_validation, test_retry_never_duplicates_execution, test_outcomes, test_prompt_has_no_privileged_fields]
    if os.environ.get("ISAAC"):
        tests.append(test_environment_boundary)
    for f in tests:
        print("PASS", f.__name__, "--", f(), flush=True)
