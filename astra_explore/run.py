"""Astra explores one LIBERO scene: fixed initial state, 3 attempts, explicit memory carried across attempts.

    cd astra_explore && python run.py --out ../experiments/astra_explore/<run> --policy codex
    python run.py --out /tmp/x --policy smoke --attempts 1 --calls-per-attempt 3        # offline pipeline check only

Per decision: save observation -> build and save the exact prompt/images -> invoke Codex (simulator paused) ->
validate -> execute the accepted action exactly once (repeat capped by the remaining horizon) -> record the
measured outcome -> update the explicit history. Inference retries never duplicate execution. A model stop
request ends the attempt as "model_stopped" (not success). Budget exhaustion is reported separately from a
completed unsuccessful attempt.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_astra import CodexAstra, build_prompt  # noqa: E402
from memory import History  # noqa: E402
from schema import ACTION_KEYS, MAX_REPEAT, Invalid, validate  # noqa: E402

GRIPPER_NOTE = "measured: after closing, 10 steps of gripper=0 left the width unchanged"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def wjson(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, default=_default) + "\n")
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_png(path, px):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(px, dtype=np.uint8)).save(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class Budget:
    def __init__(self, attempts, calls_per_attempt, total_calls):
        self.attempts, self.calls_per_attempt, self.total_calls = attempts, calls_per_attempt, total_calls
        self.total_used, self.attempt_used = 0, 0

    def new_attempt(self):
        self.attempt_used = 0

    def can_call(self):
        return self.attempt_used < self.calls_per_attempt and self.total_used < self.total_calls

    def charge(self):
        self.attempt_used += 1
        self.total_used += 1

    def view(self, attempt):
        return {"attempt": f"{attempt} of {self.attempts}",
                "model_calls_remaining_this_attempt": min(self.calls_per_attempt - self.attempt_used,
                                                          self.total_calls - self.total_used),
                "model_calls_remaining_total": self.total_calls - self.total_used}


class SmokePolicy:
    """Offline pipeline check: scripted responses, no model. Never reported as an Astra result."""
    def __init__(self, script=None):
        self.script, self.calls = script or [], 0

    def act(self, decision_dir, images, prompt):
        Path(decision_dir).mkdir(parents=True, exist_ok=True)
        (Path(decision_dir) / "prompt.txt").write_text(prompt)
        resp = self.script[self.calls] if self.calls < len(self.script) else {
            "action": {k: 0.0 for k in ACTION_KEYS}, "repeat": 1, "stop": False, "assessment_of_previous_action": None, "intent": "smoke", "hypothesis": None,
            "expected_change": "none", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""}}
        self.calls += 1
        if resp == "timeout":
            return None, {"outcome": "timeout", "requested_model": "smoke", "latency_s": 0.0}
        return resp, {"outcome": "response", "requested_model": "smoke", "latency_s": 0.0}


class Overlay:
    def __init__(self, path, fps=20):
        import imageio.v2 as imageio
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=None)
        try:
            self.font = ImageFont.truetype(FONT, 15)
        except OSError:
            self.font = ImageFont.load_default()

    def add(self, px, lines):
        im = Image.fromarray(np.asarray(px, dtype=np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 18 * len(lines) + 6], fill=(0, 0, 0))
        for i, t in enumerate(lines):
            d.text((4, 3 + 18 * i), t[:70], fill=(255, 255, 255), font=self.font)
        self.w.append_data(np.asarray(im))

    def close(self):
        self.w.close()


def run_attempt(task, policy, history, k, budget, out, cfg):
    """One attempt from the fixed initial state. Returns the attempt summary (also pushed into the history)."""
    adir = Path(out) / f"attempt_{k}"
    adir.mkdir(parents=True, exist_ok=True)
    budget.new_attempt()
    obs = task.reset()
    t_attempt = time.monotonic()
    video = Overlay(adir / "agentview.mp4") if cfg.get("video", True) else None
    if video:
        video.add(obs["images"]["agentview"], [f"attempt {k} | reset | step 0"])
    eval_f = open(adir / "evaluation_only.jsonl", "a")
    eval_f.write(json.dumps({"attempt": k, "phase": "reset", **task.privileged()}) + "\n")
    prev_agentview, decision_index, model_time, status = None, 0, 0.0, None
    interaction_ids, decisions = [], []
    while True:
        if obs["task_complete"]:
            status = "success"
            break
        if task.step_count >= task.horizon:
            status = "horizon_reached"
            break
        if not budget.can_call():
            status = "budget_exhausted"
            break
        ddir = adir / f"decision_{decision_index:02d}"
        hashes = {n: save_png(ddir / f"pre_{n}.png", px) for n, px in obs["images"].items()}
        wjson(ddir / "observation.json", {"env_step": obs["env_step"], "proprio": obs["proprio"], "camera_axes": obs["camera_axes"],
                                          "task_complete": obs["task_complete"], "image_sha256_16": hashes})
        bview = dict(budget.view(k), env_steps_remaining_this_attempt=task.horizon - task.step_count,
                     max_repeat=min(MAX_REPEAT, task.horizon - task.step_count))
        images = {"agentview": obs["images"]["agentview"], "wrist": obs["images"]["wrist"]}
        note = ""
        if prev_agentview is not None:
            images["agentview_before_previous_action"] = prev_agentview
            note = " image 3 = 'agentview' as it was BEFORE your previous action (compare with image 1 to see what changed)."
        prompt = build_prompt(cfg["task"], obs, history.view(), bview, note)
        (ddir / "prompt.txt").write_text(prompt)
        wjson(ddir / "history_view.json", history.view())
        decision, calls = None, []
        for inference_try in range(2):                      # at most one retry per decision
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
                infra = meta.get("outcome") in ("cli_error", "isolation_violation", "no_structured_output", "model_error",
                                                "unparseable_output", "timeout")
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
        drec = {"attempt": k, "decision": decision_index, "env_step_before": obs["env_step"], "calls": calls,
                "accepted": decision is not None, "executed": False}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            decision_index += 1
            continue
        history.apply_memory_update(decision["memory_update"])
        drec["decision_content"] = decision
        if decision["stop"]:
            iid = history.add_interaction({"attempt": k, "decision": decision_index, "env_step_before": obs["env_step"],
                                           "action": decision["action"], "repeat_requested": decision["repeat"], "repeat_executed": 0,
                                           "stop": True, "assessment_of_previous_action": decision["assessment_of_previous_action"],
                                           "intent": decision["intent"], "hypothesis": decision["hypothesis"],
                                           "expected_change": decision["expected_change"], "references": decision["references"],
                                           "result": {"note": "stop requested; no action executed; attempt ended", "task_complete": False}})
            drec.update(interaction_id=iid, stop=True)
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            interaction_ids.append(iid)
            status = "model_stopped"
            break
        action = [decision["action"][key] for key in ACTION_KEYS]
        requested = decision["repeat"]
        cap = min(requested, task.horizon - task.step_count)
        before, steps, success = obs, [], False
        for i in range(cap):                                 # executed exactly once per accepted decision
            obs, success, env_done = task.step(action)
            steps.append({"env_step": task.step_count, "eef": obs["proprio"]["eef_position_m_world"],
                          "gripper_width_cm": obs["proprio"]["gripper_width_cm"], "task_complete": success, "env_done": env_done})
            eval_f.write(json.dumps({"attempt": k, "decision": decision_index, "rep": i + 1, **task.privileged()}) + "\n")
            if video:
                video.add(obs["images"]["agentview"], [
                    f"attempt {k} decision {decision_index} id#{history.next_id} step {task.step_count}/{task.horizon} rep {i + 1}/{cap}",
                    "a=[" + " ".join(f"{v:+.2f}" for v in action) + f"] width {obs['proprio']['gripper_width_cm']:.1f}cm",
                    f"intent: {decision['intent'][:60]}"])
            if success or env_done:
                break
        eval_f.flush()
        d_eef = np.round(np.subtract(obs["proprio"]["eef_position_m_world"], before["proprio"]["eef_position_m_world"]), 4).tolist()
        result = {"steps_executed": len(steps), "eef_before_m": before["proprio"]["eef_position_m_world"],
                  "eef_after_m": obs["proprio"]["eef_position_m_world"], "eef_delta_m": d_eef,
                  "gripper_width_before_cm": before["proprio"]["gripper_width_cm"],
                  "gripper_width_after_cm": obs["proprio"]["gripper_width_cm"], "task_complete": success}
        iid = history.add_interaction({"attempt": k, "decision": decision_index, "env_step_before": before["env_step"],
                                       "action": decision["action"], "repeat_requested": requested, "repeat_executed": len(steps),
                                       "stop": False, "assessment_of_previous_action": decision["assessment_of_previous_action"],
                                       "intent": decision["intent"], "hypothesis": decision["hypothesis"],
                                       "expected_change": decision["expected_change"], "references": decision["references"],
                                       "result": result})
        post = {n: save_png(ddir / f"post_{n}.png", px) for n, px in obs["images"].items()}
        drec.update(executed=True, interaction_id=iid, repeat_requested=requested, repeat_cap=cap,
                    repeat_cap_reason=None if cap == requested else "remaining attempt horizon",
                    repeat_executed=len(steps), early_end="success" if success else None, steps=steps, result=result,
                    post_image_sha256_16=post, memory_after=history.running)
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        interaction_ids.append(iid)
        prev_agentview = before["images"]["agentview"]
        print(json.dumps({"attempt": k, "decision": decision_index, "id": iid, "step": task.step_count, "action": action,
                          "rep": f"{len(steps)}/{requested}", "success": success, "calls_total": budget.total_used,
                          "latency_s": calls[-1]["meta"].get("latency_s")}), flush=True)
        decision_index += 1
        if success:
            status = "success"
            break
    eval_f.close()
    if video:
        video.close()
    outcome_text = {"success": "task COMPLETED (environment success condition met)",
                    "horizon_reached": "not completed: the attempt's step horizon was reached",
                    "model_stopped": "not completed: you requested stop",
                    "budget_exhausted": "not completed: the model-call budget for this attempt was exhausted"}[status]
    summary = {"attempt": k, "outcome": status, "outcome_text": outcome_text, "decisions_total": len(decisions),
               "decisions_executed": sum(d["executed"] for d in decisions), "model_calls": budget.attempt_used,
               "env_steps": task.step_count, "simulated_time_s": round(task.step_count * cfg["task"]["control_dt_s"], 2),
               "final_eef_position_m": obs["proprio"]["eef_position_m_world"], "final_gripper_width_cm": obs["proprio"]["gripper_width_cm"],
               "interaction_ids": interaction_ids}
    history.end_attempt(summary)
    wjson(adir / "summary.json", dict(summary, model_wall_time_s=round(model_time, 1),
                                      attempt_wall_time_s=round(time.monotonic() - t_attempt, 1),
                                      history_after=history.snapshot()))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--task", type=int, default=48)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--calls-per-attempt", type=int, default=40)
    ap.add_argument("--total-calls", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--policy", choices=["codex", "smoke"], default="codex")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from libero_env import LiberoTask
    task = LiberoTask(a.suite, a.task, a.init, a.image_size)
    desc = dict(task.describe(), gripper_note=GRIPPER_NOTE)
    policy = CodexAstra(a.model, a.reasoning, a.timeout) if a.policy == "codex" else SmokePolicy()
    cli = {}
    if a.policy == "codex":
        cli = {"codex_version": subprocess.run([policy.binary, "--version"], capture_output=True, text=True).stdout.strip(),
               "login_status": " ".join(subprocess.run([policy.binary, "login", "status"], capture_output=True, text=True).stdout.split() or
                                        subprocess.run([policy.binary, "login", "status"], capture_output=True, text=True).stderr.split()[-4:]),
               "binary": policy.binary}
    cfg = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": a.policy, "is_astra_run": a.policy == "codex",
           "task": desc, "budgets": {"attempts": a.attempts, "model_calls_per_attempt": a.calls_per_attempt,
                                     "model_calls_total": a.total_calls, "timeout_per_call_s": a.timeout,
                                     "inference_retries_per_decision": 1, "max_repeat": MAX_REPEAT,
                                     "attempt_horizon_steps": task.horizon},
           "model": {"requested": a.model, "reasoning_effort": a.reasoning, **cli},
           "observation_access": ["agentview image", "wrist image", "agentview image before the previous action",
                                  "eef position/quaternion", "gripper width", "joint positions", "calibration image directions",
                                  "env step", "terminal task_complete flag", "remaining budgets", "explicit history"],
           "memory_format": "last 6 interaction records in full; <= 2 previous-attempt summaries; Astra's bounded running "
                            "memory (<= 8 observations, <= 4 hypotheses, summary <= 1200 chars) replaced by each decision",
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
        status = "complete"
        error = None
    except Exception as ex:  # noqa: BLE001  infrastructure failures must stay distinguishable from manipulation failures
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
