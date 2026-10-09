"""Astra explores one RoboTwin scene: fixed initial state, 3 attempts, explicit memory carried across attempts.

    cd astra_explore && /home/licho/anaconda3/envs/robotwin/bin/python run_rt.py --out ../experiments/astra_explore/<run> --policy codex
    ... --policy smoke --attempts 1 --calls-per-attempt 2          # offline pipeline check only (no model)

One decision = one RoboTwin `ee` action: absolute target poses for both arms + gripper targets, executed by RoboTwin's own
motion planner (planner-assisted interface, disclosed in the prompt and the report). The attempt ends on environment
success, on Astra's stop request, or when the model-call budget is exhausted; RoboTwin's own step limit (400 motions)
never binds under these budgets. Inference retries never duplicate execution.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_astra import CodexAstra  # noqa: E402
from memory import History  # noqa: E402
from run import Budget, FONT, save_png, wjson  # noqa: E402
from schema_rt import ARMS, BOX, Invalid, TOPDOWN_WXYZ, response_schema, validate  # noqa: E402

ARM_FMT = lambda a: ("hold" if not a["move"] else f"pos {[round(x, 3) for x in a['position_m']]} quat {[round(x, 2) for x in a['quaternion_wxyz']]}") + f" grip {a['gripper']:.2f}"


class SmokePolicyRT:
    """Offline pipeline check: both arms hold, grippers open. Never reported as an Astra result."""
    def __init__(self, script=None):
        self.script, self.calls = script or [], 0

    def act(self, decision_dir, images, prompt):
        Path(decision_dir).mkdir(parents=True, exist_ok=True)
        (Path(decision_dir) / "prompt.txt").write_text(prompt)
        hold = {"move": False, "position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0], "gripper": 1.0}
        resp = self.script[self.calls] if self.calls < len(self.script) else {
            "left": dict(hold), "right": dict(hold), "stop": False, "assessment_of_previous_action": None, "intent": "smoke",
            "hypothesis": None, "expected_change": "none", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
        self.calls += 1
        if resp == "timeout":
            return None, {"outcome": "timeout", "requested_model": "smoke", "latency_s": 0.0}
        return resp, {"outcome": "response", "requested_model": "smoke", "latency_s": 0.0}


def build_prompt(task, obs, history_view, budgets, previous_image_note):
    ax = lambda cam: "; ".join(f"{k}: {v}" for k, v in obs["camera_axes"][cam].items())
    lines = [
        "You directly decide the motions of a simulated dual-arm robot (RoboTwin 2.0, two 6-DoF arms with parallel-jaw grippers, "
        "left arm and right arm). There is no task policy, grasp detector or object-pose sensor acting for you: your only "
        "information is the attached camera images, the robot's own measurements below, and your recorded interaction history. "
        "Reply with ONE JSON object matching the provided schema. Use no tools.",
        "",
        f"TASK: \"{task['instruction']}\"",
        "The attempt ends when the environment itself detects task completion, when you set stop=true, or when the model-call "
        "budget is exhausted. Only the environment judges completion; stop=true is NOT success.",
        "",
        "ACTION INTERFACE (verified on this exact environment):",
        "- One decision = one motion of the robot. For EACH arm you give an absolute target pose in the world frame: position_m "
        "[x, y, z] in metres and orientation quaternion_wxyz [w, x, y, z], plus a gripper target in [0, 1] (1 = fully open, "
        "0 = fully closed; intermediate values are partial). Set move=false to keep an arm where it is (its gripper target still "
        "applies).",
        "- The robot's built-in motion planner computes a joint path to your target and executes it fully before the next "
        "observation (typically 100-500 physics steps, a few simulated seconds). The planner only avoids the robot's own body: "
        "it does NOT know about the table or the objects, so a target inside the table or an object makes the arm push into it "
        "and stop short. If a target is unreachable, planning fails and that arm does not move; you are told the planner status, "
        "the achieved pose and the position error after every motion.",
        f"- Workspace box accepted by the harness (metres): x {list(BOX['x'])}, y {list(BOX['y'])}, z {list(BOX['z'])}. Values "
        "outside are rejected without execution.",
        "- If both arms have move=true, their motions are executed SIMULTANEOUSLY within the one decision (verified: both arms "
        "reach their targets in the same motion), so coordinated two-arm actions are possible in a single decision.",
        "- Orientation references (quaternion_wxyz; 'approach' = the direction the gripper points, 'fingers' = the axis along which "
        "the two fingers open): "
        f"{list(TOPDOWN_WXYZ)} = pointing DOWN, fingers along x (top-down grasp); [0, -0.707, 0, 0.707] = pointing DOWN, fingers along y; "
        "[0.70, 0, 0, 0.71] = pointing FORWARD (+y), fingers along x (the home orientation); [1, 0, 0, 0] = pointing +x (to the right), "
        "fingers along y; [0, 0, 0, 1] = pointing -x (to the left), fingers along y; [0.707, -0.707, 0, 0] = pointing +x with the "
        "fingers VERTICAL (along z); [0, 0, 0.707, 0.707] = pointing -x with the fingers vertical. Other orientations are allowed. "
        "The reported end-effector position is the wrist reference point: the fingertips are about 8.4 cm further along the approach "
        "direction (e.g. 8.4 cm BELOW it when pointing down).",
        "- Finger state is reported as the measured separation of the two finger links: ~13.9 cm fully open, ~9 cm when fully "
        "closed on nothing; an object between the fingers keeps it larger. The gripper target is reached gradually during the "
        "arm's motion, so after a very short motion the fingers may not have finished closing or opening (check the reported "
        "separation; repeat the gripper target in the next motion if needed). The gripper joint value (0 closed .. 1 open) is also "
        "reported.",
        "- World frame: x = head-camera right, y = forward away from the robot (toward the far side of the table), z = up. "
        "Both arms start at y ~ -0.31 (robot side); the table extends toward positive y.",
        "",
        "IMAGES: image 1 = 'head_camera' (640x480, mounted on the robot's head looking forward over the table); image 2 = "
        "'front_camera' (fixed camera facing the robot from the far side of the table, so left/right are mirrored relative to "
        "the head camera); image 3 = 'left_camera' (left wrist, looks along the left gripper); image 4 = 'right_camera' (right "
        "wrist)." + previous_image_note,
        "Where a +5 cm move of the fingertips along each world axis appears RIGHT NOW (from camera calibration):",
        f"  head_camera:  {ax('head_camera')}",
        f"  left_camera:  {ax('left_camera')}",
        f"  right_camera: {ax('right_camera')}",
        "",
        "CURRENT ROBOT MEASUREMENTS: " + json.dumps(obs["proprio"]),
        f"motions used this attempt: {obs['motions_used']}; task_complete: {obs['task_complete']}.",
        "BUDGET: " + json.dumps(budgets),
        "",
        "INTERACTION HISTORY (recorded by the harness; `result` fields are measured, not your guesses):",
        json.dumps(history_view),
        "",
        "RESPONSE FIELDS: assessment_of_previous_action (first: compare the measured/visible effect of your previous motion with "
        "what you expected and say what it implies; null only if there was no previous motion in this attempt); left and right "
        "(move, position_m, quaternion_wxyz, gripper as above); stop (true only to give up this attempt); intent (1-2 sentences: "
        "what this motion is for now); hypothesis (what you are testing with it, or null); expected_change (what should visibly / "
        "measurably change if it works); references (ids of past interactions you relied on); memory_update (REPLACES your running "
        "memory: 'observations' = established facts you saw or measured, citing interaction ids, max 8 items; 'hypotheses' = "
        "unconfirmed explanations, max 4; 'summary' <= 1200 chars, carry forward older facts with their ids). Distinguish "
        "observations from hypotheses. Be concise.",
    ]
    return "\n".join(lines)


def write_video(path, frames, captions, fps=25):
    if not frames:
        return
    import imageio.v2 as imageio
    try:
        font = ImageFont.truetype(FONT, 15)
    except OSError:
        font = ImageFont.load_default()
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=None)
    for px, cap in zip(frames, captions):
        im = Image.fromarray(np.asarray(px, dtype=np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 18 * len(cap) + 6], fill=(0, 0, 0))
        for i, t in enumerate(cap):
            d.text((4, 3 + 18 * i), t[:90], fill=(255, 255, 255), font=font)
        w.append_data(np.asarray(im))
    w.close()


def run_attempt(task, policy, history, k, budget, out, cfg):
    adir = Path(out) / f"attempt_{k}"
    adir.mkdir(parents=True, exist_ok=True)
    budget.new_attempt()
    obs = task.reset()
    t_attempt = time.monotonic()
    eval_f = open(adir / "evaluation_only.jsonl", "a")
    eval_f.write(json.dumps({"attempt": k, "phase": "reset", **task.privileged()}) + "\n")
    frames, captions = [], []
    if obs["images"].get("head_camera") is not None:
        frames.append(obs["images"]["head_camera"])
        captions.append([f"attempt {k} | reset"])
    prev_head, decision_index, model_time, status = None, 0, 0.0, None
    interaction_ids, decisions = [], []
    while True:
        if obs["task_complete"]:
            status = "success"
            break
        if not budget.can_call():
            status = "budget_exhausted"
            break
        ddir = adir / f"decision_{decision_index:02d}"
        hashes = {n: save_png(ddir / f"pre_{n}.png", px) for n, px in obs["images"].items()}
        wjson(ddir / "observation.json", {"motions_used": obs["motions_used"], "proprio": obs["proprio"], "camera_axes": obs["camera_axes"],
                                          "task_complete": obs["task_complete"], "image_sha256_16": hashes})
        bview = dict(budget.view(k), motions_used_this_attempt=obs["motions_used"])
        images = dict(obs["images"])
        note = ""
        if prev_head is not None:
            images["head_camera_before_previous_motion"] = prev_head
            note = " image 5 = 'head_camera' as it was BEFORE your previous motion (compare with image 1 to see what changed)."
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
        drec = {"attempt": k, "decision": decision_index, "motions_before": obs["motions_used"], "calls": calls,
                "accepted": decision is not None, "executed": False}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            decision_index += 1
            continue
        history.apply_memory_update(decision["memory_update"])
        drec["decision_content"] = decision
        base_rec = {"attempt": k, "decision": decision_index, "motions_before": obs["motions_used"],
                    "left": decision["left"], "right": decision["right"],
                    "assessment_of_previous_action": decision["assessment_of_previous_action"], "intent": decision["intent"],
                    "hypothesis": decision["hypothesis"], "expected_change": decision["expected_change"], "references": decision["references"]}
        if decision["stop"]:
            iid = history.add_interaction(dict(base_rec, stop=True, result={"note": "stop requested; no motion executed; attempt ended", "task_complete": False}))
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
        cap = [f"attempt {k} decision {decision_index} id#{history.next_id} motion {result['motions_used']}",
               f"L {ARM_FMT(decision['left'])[:80]}", f"R {ARM_FMT(decision['right'])[:80]}", f"intent: {decision['intent'][:80]}"]
        frames.extend(new_frames)
        captions.extend([cap] * len(new_frames))
        eval_f.write(json.dumps({"attempt": k, "decision": decision_index, **task.privileged()}) + "\n")
        eval_f.flush()
        iid = history.add_interaction(dict(base_rec, stop=False, result=result))
        post = {n: save_png(ddir / f"post_{n}.png", px) for n, px in obs["images"].items()}
        drec.update(executed=True, interaction_id=iid, result=result, post_image_sha256_16=post, memory_after=history.running)
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        interaction_ids.append(iid)
        prev_head = before["images"]["head_camera"]
        print(json.dumps({"attempt": k, "decision": decision_index, "id": iid, "motions": result["motions_used"],
                          "left": ARM_FMT(decision["left"]), "right": ARM_FMT(decision["right"]),
                          "planner": {a: result["planner"][a]["status"] for a in ARMS}, "success": result["task_complete"],
                          "calls_total": budget.total_used, "latency_s": calls[-1]["meta"].get("latency_s")}), flush=True)
        decision_index += 1
        if result["task_complete"]:
            status = "success"
            break
    eval_f.close()
    if cfg.get("video", True):
        write_video(adir / "head_camera.mp4", frames, captions)
    outcome_text = {"success": "task COMPLETED (environment success condition met)",
                    "model_stopped": "not completed: you requested stop",
                    "budget_exhausted": "not completed: the model-call budget for this attempt was exhausted"}[status]
    summary = {"attempt": k, "outcome": status, "outcome_text": outcome_text, "decisions_total": len(decisions),
               "decisions_executed": sum(d["executed"] for d in decisions), "model_calls": budget.attempt_used,
               "motions": obs["motions_used"], "sim_steps": task.scene.steps if task.scene else None,
               "final_left_ee": obs["proprio"]["left"]["ee_position_m_world"], "final_right_ee": obs["proprio"]["right"]["ee_position_m_world"],
               "interaction_ids": interaction_ids}
    history.end_attempt(summary)
    wjson(adir / "summary.json", dict(summary, model_wall_time_s=round(model_time, 1), attempt_wall_time_s=round(time.monotonic() - t_attempt, 1),
                                      history_after=history.snapshot()))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="place_container_plate")
    ap.add_argument("--seed", type=int, default=1700000)
    ap.add_argument("--task-config", default="astra_probe")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--calls-per-attempt", type=int, default=40)
    ap.add_argument("--total-calls", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--policy", choices=["codex", "smoke"], default="codex")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from robotwin_env import RoboTwinTask
    task = RoboTwinTask(a.task, a.seed, a.task_config)
    desc = task.describe()
    policy = CodexAstra(a.model, a.reasoning, a.timeout, schema_fn=response_schema) if a.policy == "codex" else SmokePolicyRT()
    cli = {}
    if a.policy == "codex":
        v = subprocess.run([policy.binary, "--version"], capture_output=True, text=True)
        s = subprocess.run([policy.binary, "login", "status"], capture_output=True, text=True)
        cli = {"codex_version": v.stdout.strip(), "login_status": (s.stdout + s.stderr).strip().splitlines()[-1] if (s.stdout + s.stderr).strip() else "",
               "binary": policy.binary}
    cfg = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": a.policy, "is_astra_run": a.policy == "codex", "task": desc,
           "budgets": {"attempts": a.attempts, "model_calls_per_attempt": a.calls_per_attempt, "model_calls_total": a.total_calls,
                       "timeout_per_call_s": a.timeout, "inference_retries_per_decision": 1, "motion_limit_env": desc["step_limit_motions"]},
           "model": {"requested": a.model, "reasoning_effort": a.reasoning, **cli},
           "observation_access": ["head camera 640x480", "front camera", "left wrist camera", "right wrist camera",
                                  "head image before the previous motion", "ee poses", "finger separations", "gripper joint values",
                                  "arm joint angles", "calibration image directions", "motions used", "terminal task_complete flag",
                                  "remaining budgets", "explicit history"],
           "memory_format": "last 6 interaction records in full; <= 2 previous-attempt summaries; Astra's bounded running memory "
                            "(<= 8 observations, <= 4 hypotheses, summary <= 1200 chars) replaced by each decision",
           "video": not a.no_video, "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                                             cwd=Path(__file__).parent).stdout.strip()}
    wjson(out / "config.json", cfg)
    history, budget, summaries = History(), Budget(a.attempts, a.calls_per_attempt, a.total_calls), []
    t0 = time.monotonic()
    try:
        for k in range(1, a.attempts + 1):
            if budget.total_used >= a.total_calls:
                summaries.append({"attempt": k, "outcome": "not_started", "outcome_text": "total model-call budget exhausted"})
                continue
            summaries.append(run_attempt(task, policy, history, k, budget, out, cfg))
            wjson(out / "result.json", {"status": "running", "attempts": summaries, "model_calls_total": budget.total_used})
        status, error = "complete", None
    except Exception as ex:  # noqa: BLE001
        import traceback
        status, error = "infrastructure_error", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
    finally:
        task.close()
    wjson(out / "result.json", {"status": status, "error": error, "attempts": summaries, "model_calls_total": budget.total_used,
                                "any_success": any(s.get("outcome") == "success" for s in summaries),
                                "wall_time_s": round(time.monotonic() - t0, 1), "history": history.snapshot()})
    print("RUN_DONE", status, flush=True)


if __name__ == "__main__":
    main()
