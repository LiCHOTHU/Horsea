"""Offline boundary tests for the Astra exploration harness (no model calls).

    cd astra_explore && python tests/test_offline.py              # harness tests (any python3)
    LIBERO=1 /home/licho/anaconda3/envs/specter/bin/python tests/test_offline.py   # + environment tests
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from codex_astra import CodexAstra, landlock_spec  # noqa: E402
from memory import History  # noqa: E402
from run import Budget, SmokePolicy, run_attempt  # noqa: E402
from schema import ACTION_KEYS, Invalid, validate  # noqa: E402

PY = sys.executable


def good(**kw):
    r = {"action": {k: 0.0 for k in ACTION_KEYS}, "repeat": 3, "stop": False, "assessment_of_previous_action": None, "intent": "move", "hypothesis": None,
         "expected_change": "x", "references": [], "memory_update": {"observations": ["o"], "hypotheses": [], "summary": "s"}}
    r.update(kw)
    return r


def test_validation():
    clean, w = validate(good(), set())
    assert clean["repeat"] == 3 and w == []
    bad = [good(action={**good()["action"], "dx": 1.2}), good(action={**good()["action"], "dx": float("nan")}),
           good(repeat=0), good(repeat=21), good(repeat=2.0), good(repeat=True), good(stop="yes"),
           {k: v for k, v in good().items() if k != "intent"}, dict(good(), extra=1),
           good(action={k: 0.0 for k in ACTION_KEYS[:-1]}), good(memory_update={"observations": "x", "hypotheses": [], "summary": ""}),
           good(references=["3"]), good(action=[0] * 7)]
    for b in bad:
        try:
            validate(b, set())
            raise AssertionError(f"accepted invalid response {b}")
        except Invalid:
            pass
    clean, w = validate(good(references=[1, 7], intent="x" * 1000), {1})
    assert clean["references"] == [1] and any("unknown" in x for x in w) and len(clean["intent"]) == 400
    clean, w = validate(good(memory_update={"observations": [str(i) for i in range(20)], "hypotheses": [], "summary": "s"}), set())
    assert len(clean["memory_update"]["observations"]) == 8
    return "malformed actions, bounds, types, extra/missing keys rejected; references/lengths bounded with warnings"


class FakeTask:
    """Minimal stand-in with the adapter interface: counts steps, reports success after `succeed_at` steps."""
    def __init__(self, horizon=10, succeed_at=None):
        self.horizon, self.succeed_at, self.step_count, self.n_steps_total = horizon, succeed_at, 0, 0
        self.instruction = "fake"

    def _obs(self):
        img = np.zeros((32, 32, 3), np.uint8)
        return {"images": {"agentview": img, "wrist": img}, "proprio": {"eef_position_m_world": [0.0, 0.0, self.step_count * 0.01],
                "eef_quaternion_xyzw": [1, 0, 0, 0], "gripper_width_cm": 4.2, "joint_positions_rad": [0] * 7},
                "camera_axes": {"agentview": {"+x": "down 1px", "+y": "right 1px", "+z": "up 1px"}, "wrist": {"+x": "", "+y": "", "+z": ""}},
                "env_step": self.step_count, "task_complete": self.succeed_at is not None and self.step_count >= self.succeed_at}

    def reset(self):
        self.step_count = 0
        return self._obs()

    def step(self, action):
        self.step_count += 1
        self.n_steps_total += 1
        o = self._obs()
        return o, o["task_complete"], False

    def privileged(self):
        return {"env_step": self.step_count}


CFG = {"task": {"instruction": "fake", "control_dt_s": 0.05, "attempt_horizon_steps": 10, "gripper_note": "n"}, "video": False}


def test_retry_never_duplicates_execution():
    out = tempfile.mkdtemp()
    task = FakeTask(horizon=10)
    pol = SmokePolicy(script=[good(repeat=4, action={**good()["action"], "dx": 5.0}), good(repeat=4), "timeout", good(repeat=2)])
    b = Budget(1, 40, 120)
    s = run_attempt(task, pol, History(), 1, b, out, CFG)
    d0 = json.load(open(Path(out) / "attempt_1" / "decision_00" / "decision.json"))
    assert len(d0["calls"]) == 2 and not d0["calls"][0]["accepted"] and d0["calls"][1]["accepted"] and d0["repeat_executed"] == 4
    d1 = json.load(open(Path(out) / "attempt_1" / "decision_01" / "decision.json"))
    assert d1["calls"][0]["reason"] == "timeout" and d1["calls"][1]["accepted"] and d1["repeat_executed"] == 2
    assert task.n_steps_total == 10 and s["outcome"] == "horizon_reached" and s["model_calls"] == 8, (task.n_steps_total, s)
    shutil.rmtree(out)
    return "invalid first response then retry -> exactly one execution; timeout retried; horizon end classified"


def test_budget_vs_completion_vs_stop():
    out = tempfile.mkdtemp()
    s = run_attempt(FakeTask(horizon=100), SmokePolicy(), History(), 1, Budget(1, 2, 120), out, CFG)
    assert s["outcome"] == "budget_exhausted" and s["decisions_total"] == 2 and s["env_steps"] == 2
    s = run_attempt(FakeTask(horizon=100, succeed_at=3), SmokePolicy([good(repeat=20)]), History(), 1, Budget(1, 5, 120), out, CFG)
    assert s["outcome"] == "success" and s["env_steps"] == 3                        # execution stops at success
    h = History()
    s = run_attempt(FakeTask(horizon=100), SmokePolicy([good(stop=True)]), h, 1, Budget(1, 5, 120), out, CFG)
    assert s["outcome"] == "model_stopped" and s["env_steps"] == 0 and h.interactions[-1]["stop"]
    s = run_attempt(FakeTask(horizon=3), SmokePolicy([good(repeat=20)]), History(), 1, Budget(1, 5, 120), out, CFG)
    d = json.load(open(Path(out) / "attempt_1" / "decision_00" / "decision.json"))
    assert d["repeat_cap"] == 3 and d["repeat_cap_reason"] and s["outcome"] == "horizon_reached"
    shutil.rmtree(out)
    return "budget exhaustion, success, stop and horizon are distinct outcomes; repeat capped by remaining horizon and logged"


def test_memory_bounds_and_views():
    h = History()
    for i in range(9):
        h.add_interaction({"attempt": 1, "decision": i, "result": {}})
    h.apply_memory_update({"observations": ["a"], "hypotheses": ["b"], "summary": "c"})
    for k in range(3):
        h.end_attempt({"attempt": k + 1, "outcome": "horizon_reached"})
    v = h.view()
    assert len(v["recent_interactions"]) == 6 and v["older_interaction_ids_not_shown"] == [1, 2, 3]
    assert len(v["previous_attempts"]) == 2 and v["previous_attempts"][-1]["astra_final_memory"]["summary"] == "c"
    return "history view: last 6 interactions, <= 2 attempt summaries, running memory carried"


def test_landlock_isolation():
    d = Path(tempfile.mkdtemp())
    fake_home = d / "codex_home"
    fake_home.mkdir()
    (fake_home / "auth.json").write_text("{}")
    (d / "inputs").mkdir()
    spec = landlock_spec(d / "inputs", fake_home)
    spec["env"]["PATH"] = "/usr/bin:/bin"
    sp = d / "spec.json"
    sp.write_text(json.dumps(spec))
    run = lambda c: subprocess.run([PY, str(HERE / "landlock_exec.py"), str(sp), "--", "/bin/sh", "-c", c], capture_output=True, text=True)
    target = str(HERE.parent / "DEVLOG.md")
    r1 = run(f"cat {target}")
    r2 = run("ls /home/licho/workspace")
    r3 = run("ls /usr/bin > /dev/null && echo ok")
    r4 = run(f"cat {fake_home}/auth.json && ls {fake_home}")
    r5 = run(f"echo hi > {d}/inputs/out.txt && cat {d}/inputs/out.txt")
    r6 = run("ls /tmp")
    assert r1.returncode != 0 and "ermission denied" in r1.stderr, r1
    assert r2.returncode != 0 and "ermission denied" in r2.stderr, r2
    assert r3.returncode == 0 and r3.stdout.strip() == "ok", r3
    assert "{}" in r4.stdout and r4.returncode != 0 and "ermission denied" in r4.stderr, r4   # file by name ok, listing denied
    assert r5.stdout.strip() == "hi", r5
    assert r6.returncode != 0, r6
    shutil.rmtree(d)
    return "sandboxed child cannot read the repo, list the workspace, /tmp or the Codex home; can read its inputs and system dirs"


def test_timeout_kills_without_action():
    d = Path(tempfile.mkdtemp())
    p = CodexAstra(timeout=1.5, codex_home=d / "home")
    (d / "home").mkdir()
    p.command = lambda *a, **k: ["/bin/sleep", "30"]
    t0 = time.time()
    raw, meta = p.act(d / "dec", {"agentview": np.zeros((8, 8, 3), np.uint8)}, "prompt")
    assert raw is None and meta["outcome"] == "timeout" and time.time() - t0 < 6, meta
    assert subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True).returncode != 0, "sandboxed child not killed"
    shutil.rmtree(d)
    return "timeout kills the whole process group and returns no action"


def test_prompt_contains_no_privileged_fields():
    from codex_astra import build_prompt
    obs = FakeTask()._obs()
    p = build_prompt({"instruction": "fake", "control_dt_s": 0.05, "attempt_horizon_steps": 10, "gripper_note": "n"}, obs, History().view(),
                     {"x": 1}, "")
    for banned in ("ketchup_1_pos", "object-state", "goal", "contain_region", "reward"):
        assert banned not in p, banned
    return "prompt text carries only whitelisted fields"


def test_environment_boundary():
    from libero_env import LiberoTask
    t = LiberoTask("libero_90", 48, 0, 256)
    o1 = t.reset()
    assert set(o1) == {"images", "proprio", "camera_axes", "env_step", "task_complete"}
    assert set(o1["proprio"]) == {"eef_position_m_world", "eef_quaternion_xyzw", "gripper_width_cm", "joint_positions_rad"}
    assert not o1["task_complete"] and o1["images"]["agentview"].shape == (256, 256, 3)
    priv = t.privileged()
    assert "ketchup_1_pos" in priv and "goal" in priv
    for _ in range(5):
        o, s, _ = t.step([1, 0, 0, 0, 0, 0, -1])
    assert o["proprio"]["eef_position_m_world"][0] > o1["proprio"]["eef_position_m_world"][0] + 0.02 and t.step_count == 5
    o2 = t.reset()
    assert t.step_count == 0 and np.array_equal(o1["images"]["agentview"], o2["images"]["agentview"]) and \
        np.array_equal(o1["images"]["wrist"], o2["images"]["wrist"])
    # gripper semantics: +1 closes, 0 holds, -1 opens
    for _ in range(12):
        o, *_ = t.step([0, 0, 0, 0, 0, 0, 1])
    w_closed = o["proprio"]["gripper_width_cm"]
    for _ in range(10):
        o, *_ = t.step([0, 0, 0, 0, 0, 0, 0])
    w_hold = o["proprio"]["gripper_width_cm"]
    for _ in range(12):
        o, *_ = t.step([0, 0, 0, 0, 0, 0, -1])
    w_open = o["proprio"]["gripper_width_cm"]
    assert w_closed < 0.5 and abs(w_hold - w_closed) < 0.1 and w_open > 6.5, (w_closed, w_hold, w_open)
    t.close()
    return f"whitelisted obs only; privileged separate; reset bit-identical; gripper closed {w_closed} / held {w_hold} / open {w_open} cm"


if __name__ == "__main__":
    tests = [test_validation, test_retry_never_duplicates_execution, test_budget_vs_completion_vs_stop, test_memory_bounds_and_views,
             test_landlock_isolation, test_timeout_kills_without_action, test_prompt_contains_no_privileged_fields]
    if os.environ.get("LIBERO"):
        tests.append(test_environment_boundary)
    for f in tests:
        print("PASS", f.__name__, "--", f(), flush=True)
