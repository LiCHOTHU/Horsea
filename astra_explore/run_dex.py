"""Astra explores one Isaac Sim humanoid dex-hand scene: fixed initial state, 3 attempts, explicit memory across attempts.

    source ~/workspace/hongyi_deps/activate_hongyi.sh && cd ~/workspace/Horsea/astra_explore && \
        python -u run_dex.py --out ../experiments/astra_explore/<run> --policy codex
    ... --policy smoke --attempts 1 --calls-per-attempt 2          # offline pipeline check only (no model)

One decision = absolute wrist targets (position + finger/palm directions) for both hands plus four hand-closure
fractions, tracked by the benchmark's own differential-IK controllers for `steps` control steps at 30 Hz (IK-assisted,
no planner, no collision avoidance; disclosed in the prompt and the report). The attempt ends on environment success, on
Astra's stop request, when the model-call budget is exhausted, or when the attempt horizon (the environment's 60 s
episode minus the settle period) is reached. Inference retries never duplicate execution.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_astra import CodexAstra  # noqa: E402
from memory import History  # noqa: E402
from run import Budget, save_png, wjson  # noqa: E402
from run_rt import write_video  # noqa: E402
from schema_dex import BOX, HANDS, Invalid, STEPS, response_schema, validate  # noqa: E402


def HAND_FMT(h):
    c = f" hand b{h['finger_base']:.2f}/t{h['finger_tip']:.2f}/r{h['thumb_rotate']:.2f}/k{h['thumb_bend']:.2f}"
    if not h["move"]:
        return "hold" + c
    return f"pos {[round(x, 3) for x in h['position_m']]} fingers->{[round(x, 2) for x in h['fingers_dir']]} palm->{[round(x, 2) for x in h['palm_dir']]}" + c


class SmokePolicyDex:
    """Offline pipeline check: both hands hold, open, 5 steps. Never reported as an Astra result."""
    def __init__(self, script=None):
        self.script, self.calls = script or [], 0

    def act(self, decision_dir, images, prompt):
        Path(decision_dir).mkdir(parents=True, exist_ok=True)
        (Path(decision_dir) / "prompt.txt").write_text(prompt)
        hold = {"move": False, "position_m": [0, 0, 0], "fingers_dir": [1, 0, 0], "palm_dir": [0, 0, -1], "finger_base": 0.0, "finger_tip": 0.0,
                "thumb_rotate": 0.0, "thumb_bend": 0.0}
        resp = self.script[self.calls] if self.calls < len(self.script) else {
            "left": dict(hold), "right": dict(hold), "steps": 5, "stop": False, "assessment_of_previous_action": None, "intent": "smoke",
            "hypothesis": None, "expected_change": "none", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
        self.calls += 1
        if resp == "timeout":
            return None, {"outcome": "timeout", "requested_model": "smoke", "latency_s": 0.0}
        return resp, {"outcome": "response", "requested_model": "smoke", "latency_s": 0.0}


def build_prompt(task, obs, history_view, budgets, previous_image_note):
    ax = lambda cam: "; ".join(f"{k}: {v}" for k, v in obs["camera_axes"][cam].items())
    cal = task.get("calibration", {})
    cl = cal.get("left", {})
    cr = cal.get("right", {})
    lines = [
        "You directly decide the motions of a simulated humanoid robot (Unitree H1 upper body with two five-finger dexterous "
        "hands, left and right). There is no task policy, grasp planner, collision avoidance or object-pose sensor acting for you: "
        "your only information is the attached camera images, the robot's own measurements below, and your recorded interaction "
        "history. Reply with ONE JSON object matching the provided schema. Use no tools.",
        "",
        f"TASK: \"{task['instruction']}\"",
        "The attempt ends when the environment itself detects task completion, when you set stop=true, when the model-call budget "
        f"is exhausted, or when the attempt horizon of {task['attempt_horizon_steps']} control steps is used up. Only the environment "
        "judges completion; stop=true is NOT success.",
        "",
        "ACTION INTERFACE (verified on this exact environment):",
        "- One decision = both hands act for `steps` control steps (30 Hz; 1 to 150 steps = 1/30 s to 5 s). For EACH hand you give an "
        "absolute target for the WRIST point (the hand's base) in the world frame: position_m [x, y, z] in metres; fingers_dir = the "
        "world direction the straightened fingers should point; palm_dir = the world direction the palm should face (perpendicular to "
        "fingers_dir; the harness re-orthogonalises small deviations). Set move=false to keep an arm's joints where they are.",
        "- The commanded wrist pose moves from the current pose to your target along a straight line at most 1 cm and 3 degrees per "
        "step (so a 30 cm move needs at least 30 steps; extra steps let it settle), and the robot's differential inverse-kinematics "
        "controller tracks it. Nothing stops a hand or arm from pushing into the table, the furniture or an object. If a target pose "
        "(position AND orientation) is out of reach, the arm goes as far as it can, possibly with a different orientation, and the "
        "residual errors are reported. The achieved wrist position, position error, orientation error and distance moved are measured "
        "after every decision.",
        f"- Workspace box accepted by the harness (metres): x {list(BOX['x'])}, y {list(BOX['y'])}, z {list(BOX['z'])}. Values outside "
        "are rejected without execution. steps outside [1, 150] are rejected.",
        "- Hands (five-finger dexterous hands, 12 joints each): four closure fractions per hand in [0, 1] (0 = the open default, 1 = the "
        "joint limit; intermediate values allowed): finger_base = the knuckle joints of the four fingers (0 = straight in line with the "
        "palm, 1 = bent ~97 deg toward the palm side); finger_tip = the middle joints of the four fingers (0 = straight, 1 = bent ~97 "
        "deg); thumb_rotate = the thumb's base rotation (0 = thumb spread out to its side, 1 = rotated ~75 deg toward the palm to face "
        "the fingers); thumb_bend = the thumb's three bending joints together (0 = straight, 1 = curled). Measured with the hand in the "
        f"air: the four knuckles are ~{cr.get('knuckle_offset_along_fingers_cm')} cm along fingers_dir from the wrist point and the open "
        f"fingertips ~{cr.get('fingertip_offset_along_fingers_cm')} cm (fingers lie in the plane through the wrist point; the palm surface is "
        f"about 1 cm in front of it along palm_dir); index-to-pinky spread {cr.get('index_pinky_spread_cm')} cm; thumb-to-index-tip distance "
        f"{cr.get('thumb_index_open_cm')} cm open, {cr.get('thumb_index_pinch_cm')} cm with all four fractions at 0.5, {cr.get('thumb_index_closed_cm')} cm "
        "with all at 1 (closed on nothing). An object held between thumb and fingers keeps that distance larger. Closure fractions change "
        "at most 1/30 per step (fully open to fully closed takes 30 steps = 1 s); a decision that ends early leaves the hand part-way "
        "(the measured fractions are reported).",
        "- Each hand's measurements include the world directions its fingers and palm currently point to and the side its thumb is on "
        "(thumb_side_direction_world). The thumb side follows from your two directions: RIGHT hand thumb side = fingers_dir x palm_dir, "
        "LEFT hand thumb side = palm_dir x fingers_dir (cross products).",
        "- A contact sensor on each index finger's middle link reports a contact force (N); other fingers have no sensor.",
        "- Both hands act SIMULTANEOUSLY within one decision, so coordinated two-hand actions are possible.",
        "- World frame: x = forward from the robot toward the table, y = to the robot's LEFT, z = up; the robot's pelvis is at the "
        "origin. Example orientations: fingers_dir [1,0,0] with palm_dir [0,0,-1] = hand flat, palm down, fingers pointing away from "
        "the robot; fingers_dir [0,0,-1] with palm_dir [0,-1,0] = fingers pointing down with the palm facing the robot's right.",
        "",
        "IMAGES: image 1 = 'main_camera' (fixed camera above the robot's head looking forward/down over the table); image 2 = "
        "'left_eye_camera' (the robot's head, left eye); image 3 = 'left_hand_camera' (mounted on the left wrist, looks along the "
        "left hand); image 4 = 'right_hand_camera' (right wrist)." + previous_image_note,
        "Where a +5 cm move along each world axis appears in each image RIGHT NOW, at the stated reference point (from camera calibration):",
        f"  main_camera:        {ax('main_camera')}",
        f"  left_eye_camera:    {ax('left_eye_camera')}",
        f"  left_hand_camera:   {ax('left_hand_camera')}",
        f"  right_hand_camera:  {ax('right_hand_camera')}",
        "",
        "CURRENT ROBOT MEASUREMENTS: " + json.dumps(obs["proprio"]),
        f"control steps used this attempt: {obs['steps_used']}; steps remaining: {obs['steps_remaining']}; task_complete: {obs['task_complete']}.",
        "BUDGET: " + json.dumps(budgets),
        "",
        "INTERACTION HISTORY (recorded by the harness; `result` fields are measured, not your guesses):",
        json.dumps(history_view),
        "",
        "RESPONSE FIELDS: assessment_of_previous_action (first: compare the measured/visible effect of your previous decision with "
        "what you expected and say what it implies; null only if there was no previous decision in this attempt); left and right "
        "(move, position_m, fingers_dir, palm_dir, finger_base, finger_tip, thumb_rotate, thumb_bend as above); steps (integer 1-150); stop (true only to give up this "
        "attempt); intent (1-2 sentences: what this decision is for now); hypothesis (what you are testing with it, or null); "
        "expected_change (what should visibly / measurably change if it works); references (ids of past interactions you relied on); "
        "memory_update (REPLACES your running memory: 'observations' = established facts you saw or measured, citing interaction ids, "
        "max 8 items; 'hypotheses' = unconfirmed explanations, max 4; 'summary' <= 1200 chars, carry forward older facts with their "
        "ids). Distinguish observations from hypotheses. Be concise.",
    ]
    return "\n".join(lines)


def run_attempt(task, policy, history, k, budget, out, cfg):
    adir = Path(out) / f"attempt_{k}"
    adir.mkdir(parents=True, exist_ok=True)
    budget.new_attempt()
    obs = task.reset()
    t_attempt = time.monotonic()
    eval_f = open(adir / "evaluation_only.jsonl", "a")
    eval_f.write(json.dumps({"attempt": k, "phase": "reset", **task.privileged()}) + "\n")
    frames, captions = [], []
    f_seen = len(task.frames())
    frames.extend(task.frames()[:f_seen])
    captions.extend([[f"attempt {k} | reset"]] * f_seen)
    prev_main, decision_index, model_time, status = None, 0, 0.0, None
    interaction_ids, decisions = [], []
    while True:
        if obs["task_complete"]:
            status = "success"
            break
        if obs["steps_remaining"] <= 0:
            status = "horizon_reached"
            break
        if not budget.can_call():
            status = "budget_exhausted"
            break
        ddir = adir / f"decision_{decision_index:02d}"
        hashes = {n: save_png(ddir / f"pre_{n}.png", px) for n, px in obs["images"].items()}
        wjson(ddir / "observation.json", {"steps_used": obs["steps_used"], "steps_remaining": obs["steps_remaining"], "proprio": obs["proprio"],
                                          "camera_axes": obs["camera_axes"], "task_complete": obs["task_complete"], "image_sha256_16": hashes})
        bview = dict(budget.view(k), steps_used_this_attempt=obs["steps_used"], steps_remaining_this_attempt=obs["steps_remaining"])
        images = dict(obs["images"])
        note = ""
        if prev_main is not None:
            images["main_camera_before_previous_decision"] = prev_main
            note = " image 5 = 'main_camera' as it was BEFORE your previous decision (compare with image 1 to see what changed)."
        prompt = build_prompt(cfg["task"], obs, history.view(), bview, note)
        (ddir / "prompt.txt").write_text(prompt)
        wjson(ddir / "history_view.json", history.view())
        decision, calls = None, []
        for inference_try in range(2):
            if not budget.can_call():
                calls.append({"try": inference_try, "skipped": "budget exhausted"})
                break
            budget.charge()
            t0 = time.monotonic()
            raw, meta = policy.act(ddir / f"call_{inference_try}", images, prompt)
            model_time += time.monotonic() - t0
            rec = {"try": inference_try, "meta": meta}
            if raw is None:
                rec.update(accepted=False, reason=meta.get("outcome"))
                calls.append(rec)
                infra = meta.get("outcome") in ("cli_error", "isolation_violation", "no_structured_output", "model_error", "unparseable_output", "timeout")
                cfg["_infra_fail_streak"] = cfg.get("_infra_fail_streak", 0) + 1 if infra else 0
                if cfg["_infra_fail_streak"] >= 3 or (budget.total_used <= 2 and cfg["_infra_fail_streak"] >= 2):
                    raise RuntimeError(f"infrastructure blocker: {cfg['_infra_fail_streak']} consecutive failed model calls "
                                       f"(last outcome {meta.get('outcome')}: {meta.get('stderr_tail') or meta.get('errors') or meta.get('violations') or ''})")
                continue
            cfg["_infra_fail_streak"] = 0
            try:
                clean, warnings = validate(raw, history.known_ids())
                rec.update(accepted=True, warnings=warnings)
                decision = clean
                calls.append(rec)
                break
            except Invalid as ex:
                rec.update(accepted=False, reason=f"invalid response: {ex}", raw=raw)
                calls.append(rec)
        drec = {"attempt": k, "decision": decision_index, "steps_before": obs["steps_used"], "calls": calls,
                "accepted": decision is not None, "executed": False}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            decision_index += 1
            continue
        history.apply_memory_update(decision["memory_update"])
        drec["decision_content"] = decision
        base_rec = {"attempt": k, "decision": decision_index, "steps_before": obs["steps_used"],
                    "left": decision["left"], "right": decision["right"], "steps": decision["steps"],
                    "assessment_of_previous_action": decision["assessment_of_previous_action"], "intent": decision["intent"],
                    "hypothesis": decision["hypothesis"], "expected_change": decision["expected_change"], "references": decision["references"]}
        if decision["stop"]:
            iid = history.add_interaction(dict(base_rec, stop=True, result={"note": "stop requested; nothing executed; attempt ended", "task_complete": False}))
            drec.update(interaction_id=iid, stop=True)
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            interaction_ids.append(iid)
            status = "model_stopped"
            break
        before = obs
        f0 = len(task.frames())
        obs, result = task.act(decision)                       # executed exactly once per accepted decision
        new_frames = task.frames()[f0:]
        cap = [f"attempt {k} decision {decision_index} id#{history.next_id} steps {result['steps_executed']} (t={obs['steps_used']})",
               f"L {HAND_FMT(decision['left'])[:95]}", f"R {HAND_FMT(decision['right'])[:95]}", f"intent: {decision['intent'][:90]}"]
        frames.extend(new_frames)
        captions.extend([cap] * len(new_frames))
        eval_f.write(json.dumps({"attempt": k, "decision": decision_index, **task.privileged()}) + "\n")
        eval_f.flush()
        iid = history.add_interaction(dict(base_rec, stop=False, result=result))
        post = {n: save_png(ddir / f"post_{n}.png", px) for n, px in obs["images"].items()}
        drec.update(executed=True, interaction_id=iid, result=result, post_image_sha256_16=post, memory_after=history.running,
                    steps_requested=result["steps_requested"], steps_executed=result["steps_executed"], steps_clip_reason=result["clipped"])
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        interaction_ids.append(iid)
        prev_main = before["images"]["main_camera"]
        print(json.dumps({"attempt": k, "decision": decision_index, "id": iid, "steps": result["steps_executed"], "t": obs["steps_used"],
                          "left": HAND_FMT(decision["left"]), "right": HAND_FMT(decision["right"]),
                          "err_cm": {s: result["hands"][s].get("position_error_cm") for s in HANDS}, "success": result["task_complete"],
                          "calls_total": budget.total_used, "latency_s": calls[-1]["meta"].get("latency_s")}), flush=True)
        decision_index += 1
        if result["task_complete"]:
            status = "success"
            break
    eval_f.close()
    if cfg.get("video", True):
        write_video(adir / "main_camera.mp4", frames, captions, fps=15)
    outcome_text = {"success": "task COMPLETED (environment success condition met)",
                    "model_stopped": "not completed: you requested stop",
                    "budget_exhausted": "not completed: the model-call budget for this attempt was exhausted",
                    "horizon_reached": "not completed: the attempt horizon (control steps) was used up"}[status]
    summary = {"attempt": k, "outcome": status, "outcome_text": outcome_text, "decisions_total": len(decisions),
               "decisions_executed": sum(d["executed"] for d in decisions), "model_calls": budget.attempt_used,
               "steps_used": obs["steps_used"], "simulated_time_s": round(obs["steps_used"] / 30.0, 1),
               "final_left_wrist": obs["proprio"]["left"]["wrist_position_m_world"], "final_right_wrist": obs["proprio"]["right"]["wrist_position_m_world"],
               "interaction_ids": interaction_ids}
    history.end_attempt(summary)
    wjson(adir / "summary.json", dict(summary, model_wall_time_s=round(model_time, 1), attempt_wall_time_s=round(time.monotonic() - t_attempt, 1),
                                      history_after=history.snapshot()))
    return summary


def rebuild_history(out):
    """Reconstruct the explicit history of a finished run directory exactly as the harness held it after its last attempt:
    every interaction record (same fields and order as run_attempt writes them), the attempt summaries with Astra's final
    memories, Astra's running memory, and the number of model calls charged. Used to continue a run with more attempts."""
    import copy
    h, summaries, calls, last = History(), [], 0, None
    for adir in sorted(Path(out).glob("attempt_*"), key=lambda q: int(q.name.split("_")[1])):
        s = json.loads((adir / "summary.json").read_text())          # only finished attempts can be continued from
        for ddir in sorted(adir.glob("decision_*")):
            d = json.loads((ddir / "decision.json").read_text())
            calls += sum(1 for c in d["calls"] if "skipped" not in c)
            c = d.get("decision_content")
            if d.get("interaction_id") is None:
                continue
            base = {"attempt": d["attempt"], "decision": d["decision"], "steps_before": d["steps_before"], "left": c["left"], "right": c["right"],
                    "steps": c["steps"], "assessment_of_previous_action": c["assessment_of_previous_action"], "intent": c["intent"],
                    "hypothesis": c["hypothesis"], "expected_change": c["expected_change"], "references": c["references"]}
            if d.get("stop"):
                rec = dict(base, stop=True, result={"note": "stop requested; nothing executed; attempt ended", "task_complete": False})
            else:
                rec = dict(base, stop=False, result=d["result"])
            rec["id"] = d["interaction_id"]
            h.interactions.append(rec)
        summaries.append({k: v for k, v in s.items() if k not in ("model_wall_time_s", "attempt_wall_time_s", "history_after")})
        last = s
    if last is None:
        raise SystemExit("nothing to resume: no finished attempt in " + str(out))
    h.attempt_summaries = copy.deepcopy(last["history_after"]["attempt_summaries"])
    h.running = copy.deepcopy(last["history_after"]["running_memory"])
    h.next_id = max(r["id"] for r in h.interactions) + 1
    return h, summaries, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="Humanoid-Stack-Can-Into-Drawer-v0")
    ap.add_argument("--room", type=int, default=1)
    ap.add_argument("--table", type=int, default=2)
    ap.add_argument("--scene-index", type=int, default=None, help="randomize_idx; default = the project's first test scene for room/table")
    ap.add_argument("--settle-steps", type=int, default=60)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--calls-per-attempt", type=int, default=40)
    ap.add_argument("--total-calls", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--policy", choices=["codex", "smoke"], default="codex")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue the finished run in --out: rebuild its explicit history and run "
                    "attempts last+1 .. --attempts (budgets given here are the new cumulative totals)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    policy = CodexAstra(a.model, a.reasoning, a.timeout, schema_fn=response_schema) if a.policy == "codex" else SmokePolicyDex()
    cli = {}
    if a.policy == "codex":
        v = subprocess.run([policy.binary, "--version"], capture_output=True, text=True)
        s = subprocess.run([policy.binary, "login", "status"], capture_output=True, text=True)
        cli = {"codex_version": v.stdout.strip(), "login_status": (s.stdout + s.stderr).strip().splitlines()[-1] if (s.stdout + s.stderr).strip() else "",
               "binary": policy.binary}
    from isaac_env import IsaacDexTask
    task = IsaacDexTask(a.task, a.room, a.table, a.scene_index, settle_steps=a.settle_steps)
    desc = task.describe()
    cfg = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": a.policy, "is_astra_run": a.policy == "codex", "task": desc,
           "budgets": {"attempts": a.attempts, "model_calls_per_attempt": a.calls_per_attempt, "model_calls_total": a.total_calls,
                       "timeout_per_call_s": a.timeout, "inference_retries_per_decision": 1, "attempt_horizon_steps": desc["attempt_horizon_steps"],
                       "steps_per_decision": list(STEPS)},
           "model": {"requested": a.model, "reasoning_effort": a.reasoning, **cli},
           "observation_access": ["main camera 960x540 (fixed, above the head)", "left eye camera 640x360", "left hand camera 640x360", "right hand camera 640x360",
                                  "main image before the previous decision", "wrist poses as finger/palm/thumb-side directions", "fingertip positions",
                                  "thumb-index distances", "measured closure fractions", "index-finger contact forces", "arm joint angles",
                                  "calibration image directions", "steps used/remaining", "terminal task_complete flag", "remaining budgets", "explicit history"],
           "memory_format": "last 6 interaction records in full; <= 2 previous-attempt summaries; Astra's bounded running memory "
                            "(<= 8 observations, <= 4 hypotheses, summary <= 1200 chars) replaced by each decision",
           "video": not a.no_video, "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                                             cwd=Path(__file__).parent).stdout.strip()}
    budget = Budget(a.attempts, a.calls_per_attempt, a.total_calls)
    if a.resume:
        prev = json.loads((out / "config.json").read_text())
        if prev["task"]["scene_index"] != desc["scene_index"] or prev["task"]["task"] != desc["task"]:
            raise SystemExit("resume refused: different task or scene")
        history, summaries, prev_calls = rebuild_history(out)
        budget.total_used = prev_calls
        first = len(summaries) + 1
        prev.setdefault("continuations", []).append({"started": cfg["started"], "first_attempt": first, "attempts_total": a.attempts,
                                                     "budgets": cfg["budgets"], "git": cfg["git"], "model": cfg["model"],
                                                     "model_calls_before": prev_calls,
                                                     "note": "same scene and protocol; explicit history rebuilt from the run's records"})
        prev["budgets"] = cfg["budgets"]
        cfg = prev
    else:
        history, summaries, first = History(), [], 1
    wjson(out / "config.json", cfg)
    t0 = time.monotonic()
    try:
        for k in range(first, a.attempts + 1):
            if budget.total_used >= a.total_calls:
                summaries.append({"attempt": k, "outcome": "not_started", "outcome_text": "total model-call budget exhausted"})
                continue
            summaries.append(run_attempt(task, policy, history, k, budget, out, cfg))
            wjson(out / "result.json", {"status": "running", "attempts": summaries, "model_calls_total": budget.total_used})
        status, error = "complete", None
    except Exception as ex:  # noqa: BLE001
        import traceback
        status, error = "infrastructure_error", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
    # written BEFORE closing the simulator: Isaac Sim's app.close() terminates the Python process
    wjson(out / "result.json", {"status": status, "error": error, "attempts": summaries, "model_calls_total": budget.total_used,
                                "any_success": any(s.get("outcome") == "success" for s in summaries),
                                "wall_time_s": round(time.monotonic() - t0, 1), "history": history.snapshot()})
    print("RUN_DONE", status, flush=True)
    task.close()


if __name__ == "__main__":
    main()
