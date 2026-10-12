"""Cup-transfer experiment runner (Isaac, conda env hongyi; run from egovla_release/egovla with PYTHONPATH=astra_explore).

    python -m cup_transfer.run_cup stage0 --obj cup_a [--render]              asset / reset / limits / feasibility checks
    python -m cup_transfer.run_cup source --obj cup_a                          Task 1: three Astra attempts on cup A (nominal executor)
    python -m cup_transfer.run_cup target --obj cup_b [--methods ...] [--starts t1,t2]   Tasks 2-3: starts x methods x <= 2 attempts
    (--policy smoke replaces Astra by a scripted reference grasp for pipeline tests; never reported as an Astra result)

Experiment root: experiments/cup_transfer_2026-10-12 (see the brief's deliverable layout). Every target condition
(object, start, method) starts from a fresh copy of what it may use: Astra's history at the end of the source run (every
method) and the numerical source memory D_A (nearest-neighbour, frozen, adaptive; empty starts with none). A condition's two
attempts share its own target history; nothing leaks between conditions. Conditions already finished are skipped (resume).
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex_astra import CodexAstra  # noqa: E402
from memory import History  # noqa: E402
from run import Budget, save_png, wjson  # noqa: E402
from run_dex import HAND_FMT  # noqa: E402
from run_rt import write_video  # noqa: E402
from schema_dex import Invalid, STEPS  # noqa: E402
from cup_transfer import system1 as S1  # noqa: E402
from cup_transfer.prompt_cup import build_prompt_cup, response_schema_cup, validate_cup  # noqa: E402

ROOT = Path("/home/licho/workspace/Horsea/experiments/cup_transfer_2026-10-12")
ASSETS = ROOT / "asset_and_task_definitions" / "assets"
SOURCE = ROOT / "source_interaction_dataset"
LOGS = ROOT / "per_attempt_transition_logs"
STARTS = {"s1": (0.0, 0.0), "s2": (0.015, 0.010), "s3": (-0.010, -0.015),           # source attempts on cup A
          "t1": (0.010, -0.010), "t2": (-0.015, 0.005)}                              # target starts (B and C)
INSTR = {"cup_a": "Pick up the cup, lift it at least 5 cm above its initial position, and hold it steadily for one second.",
         "cup_b": "Pick up this new cup, lift it at least 5 cm, and hold it steadily for one second.",
         "cyl_c": "Pick up the cylinder, lift it at least 5 cm, and hold it steadily for one second.",
         "cup_bp": "Pick up this new cup, lift it at least 5 cm, and hold it steadily for one second."}
METHODS = ["astra_history", "nearest_neighbor", "empty_model", "frozen_model", "adaptive_model"]
HORIZON, CALLS_PER_ATTEMPT = 400, 40


def specs():
    return json.loads((ASSETS / "asset_specs.json").read_text())


# ---------------------------------------------------------------------------- scripted smoke policy (pipeline tests only)
REF = {"offset": (-0.134, -0.066, 0.128), "fingers": [0.9821, 0.1880, 0.0060], "palm": [-0.0677, 0.3831, -0.9212],
       "close": [0.60, 0.60, 0.65, 0.50], "R0": 0.03185, "H0": 0.1218}


def reference_grasp(obj_pos, R, H):
    """Astra's successful Unload-Cans grasp (right hand, attempt 2) re-anchored to this object's base and size."""
    off = np.array([REF["offset"][0] - (R - REF["R0"]), REF["offset"][1], H + 0.006])
    tgt = np.array(obj_pos) + off
    hold = {"move": False, "position_m": [0, 0, 1], "fingers_dir": [1, 0, 0], "palm_dir": [0, 0, -1], "finger_base": 0.0, "finger_tip": 0.0,
            "thumb_rotate": 0.0, "thumb_bend": 0.0}

    def right(p, close, move=True):
        c = REF["close"] if close else [0.0, 0.0, 0.15, 0.0]
        return {"move": move, "position_m": [float(x) for x in p], "fingers_dir": REF["fingers"], "palm_dir": REF["palm"],
                "finger_base": c[0], "finger_tip": c[1], "thumb_rotate": c[2], "thumb_bend": c[3]}
    seq = [("approach", right(tgt + [0, 0, 0.115], False), 40), ("approach", right(tgt, False), 16), ("close", right(tgt, True, False), 22),
           ("lift", right(tgt + [0, 0, 0.125], True), 18), ("hold", right(tgt + [0, 0, 0.125], True), 40)]
    out = []
    for mode, r, steps in seq:
        out.append({"left": dict(hold), "right": r, "steps": steps, "stop": False, "assessment_of_previous_action": None, "intent": f"scripted {mode}",
                    "hypothesis": None, "expected_change": "scripted", "references": [], "memory_update": {"observations": [], "hypotheses": [], "summary": ""},
                    "interaction_mode": mode, "active_hand": "right"})
    return out


class SmokePolicy:
    def __init__(self, task):
        self.task, self.queue, self.calls = task, [], 0

    def act(self, decision_dir, images, prompt):
        Path(decision_dir).mkdir(parents=True, exist_ok=True)
        (Path(decision_dir) / "prompt.txt").write_text(prompt)
        if not self.queue:
            st = self.task.obj_state()
            self.queue = reference_grasp(st["pos"], self.task.R, self.task.H)
        self.calls += 1
        return self.queue.pop(0), {"outcome": "response", "requested_model": "smoke", "latency_s": 0.0}


# ---------------------------------------------------------------------------- methods
def make_method(name, source_store, hyper, seed):
    if name == "astra_history":
        return S1.Nominal(), None, False
    if name == "nearest_neighbor":
        st = source_store.clone(frozen=False)
        return S1.NearestNeighborReuse(st), st, True
    if name == "empty_model":
        st = S1.ExperienceStore()
        return S1.ShortHorizonController(st, h=hyper["h"], lam=hyper["lam"], seed=seed), st, True
    if name == "frozen_model":
        st = source_store.clone(frozen=True)
        return S1.ShortHorizonController(st, h=hyper["h"], lam=hyper["lam"], seed=seed), st, False
    if name == "adaptive_model":
        st = source_store.clone(frozen=False)
        return S1.ShortHorizonController(st, h=hyper["h"], lam=hyper["lam"], seed=seed), st, True
    raise ValueError(name)


# ---------------------------------------------------------------------------- one attempt
def run_attempt(T, policy, history, budget, adir, k, method, store, update, meta, cfg):
    from cup_transfer.cup_task import execute_decision
    adir.mkdir(parents=True, exist_ok=True)
    budget.new_attempt()
    T.set_start(STARTS[meta["start"]])
    obs = T.reset()
    t_attempt = time.monotonic()
    tl = open(adir / "transitions.jsonl", "a")
    frames, captions = list(T.frames()), [[f"{meta['arm']} attempt {k} | reset"]] * len(T.frames())
    prev_main, di, model_time, status = None, 0, 0.0, None
    decisions, ids = [], []
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
        ddir = adir / f"decision_{di:02d}"
        hashes = {n: save_png(ddir / f"pre_{n}.png", px) for n, px in obs["images"].items()}
        wjson(ddir / "observation.json", {"steps_used": obs["steps_used"], "proprio": obs["proprio"], "object": obs["object"], "image_sha256_16": hashes})
        bview = dict(budget.view(k), steps_used_this_attempt=obs["steps_used"], steps_remaining_this_attempt=obs["steps_remaining"])
        images = dict(obs["images"])
        note = ""
        if prev_main is not None:
            images["main_camera_before_previous_decision"] = prev_main
            note = " image 5 = 'main_camera' as it was BEFORE your previous decision (compare with image 1 to see what changed)."
        prompt = build_prompt_cup(cfg["task"], obs, history.view(), bview, note)
        (ddir / "prompt.txt").write_text(prompt)
        decision, calls = None, []
        for tr in range(2):
            if not budget.can_call():
                break
            budget.charge()
            t0 = time.monotonic()
            raw, m = policy.act(ddir / f"call_{tr}", images, prompt)
            model_time += time.monotonic() - t0
            rec = {"try": tr, "meta": m}
            if raw is None:
                rec.update(accepted=False, reason=m.get("outcome"))
                calls.append(rec)
                infra = m.get("outcome") in ("cli_error", "isolation_violation", "no_structured_output", "model_error", "unparseable_output", "timeout")
                cfg["_streak"] = cfg.get("_streak", 0) + 1 if infra else 0
                if cfg["_streak"] >= 3 or (budget.total_used <= 2 and cfg["_streak"] >= 2):
                    raise RuntimeError(f"infrastructure blocker: {cfg['_streak']} failed model calls ({m.get('outcome')}: {m.get('stderr_tail') or ''})")
                continue
            cfg["_streak"] = 0
            try:
                decision, warnings = validate_cup(raw, history.known_ids())
                rec.update(accepted=True, warnings=warnings)
                calls.append(rec)
                break
            except Invalid as ex:
                rec.update(accepted=False, reason=f"invalid response: {ex}", raw=raw)
                calls.append(rec)
        drec = {"attempt": k, "decision": di, "steps_before": obs["steps_used"], "calls": calls, "accepted": decision is not None, "executed": False}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            di += 1
            continue
        history.apply_memory_update(decision["memory_update"])
        drec["decision_content"] = decision
        base = {"attempt": k, "decision": di, "steps_before": obs["steps_used"], "left": decision["left"], "right": decision["right"],
                "steps": decision["steps"], "interaction_mode": decision["interaction_mode"], "active_hand": decision["active_hand"],
                "assessment_of_previous_action": decision["assessment_of_previous_action"], "intent": decision["intent"],
                "hypothesis": decision["hypothesis"], "expected_change": decision["expected_change"], "references": decision["references"]}
        if decision["stop"]:
            iid = history.add_interaction(dict(base, stop=True, result={"note": "stop requested; nothing executed; attempt ended", "task_complete": False}))
            drec.update(interaction_id=iid, stop=True)
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            ids.append(iid)
            status = "model_stopped"
            break
        before = obs
        f0 = len(T.frames())
        ctx_meta = {"episode_id": f"{meta['arm']}/a{k}", "decision": di, "arm": meta["arm"], "attempt": k, "provenance": meta.get("provenance", "target")}
        obs, result, rows = execute_decision(T, decision, method, store, ctx_meta, tl, update=update)
        nf = T.frames()[f0:]
        cap = [f"{meta['arm']} | attempt {k} decision {di} | {decision['interaction_mode']} ({decision['active_hand']}) | t={obs['steps_used']}",
               f"L {HAND_FMT(decision['left'])[:92]}", f"R {HAND_FMT(decision['right'])[:92]}",
               f"layer: {result['execution_layer']['adjusted_segments']}/{result['execution_layer']['segments']} segments adjusted | object rise {result['object']['rise_above_start_cm']} cm"]
        frames.extend(nf)
        captions.extend([cap] * len(nf))
        iid = history.add_interaction(dict(base, stop=False, result=result))
        post = {n: save_png(ddir / f"post_{n}.png", px) for n, px in obs["images"].items()}
        drec.update(executed=True, interaction_id=iid, result=result, post_image_sha256_16=post, memory_after=history.running, segments=len(rows))
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        ids.append(iid)
        prev_main = before["images"]["main_camera"]
        print(json.dumps({"arm": meta["arm"], "attempt": k, "decision": di, "mode": decision["interaction_mode"], "steps": result["steps_executed"],
                          "t": obs["steps_used"], "layer": result["execution_layer"], "object": result["object"], "success": result["task_complete"],
                          "calls": budget.total_used}), flush=True)
        di += 1
        if result["task_complete"]:
            status = "success"
            break
    tl.close()
    with open(adir / "eval_trace.jsonl", "w") as f:                    # privileged, evaluation only
        for r in T.trace:
            f.write(json.dumps(r) + "\n")
    write_video(adir / "main_camera.mp4", frames, captions, fps=15)
    text = {"success": "task COMPLETED (environment success condition met)", "model_stopped": "not completed: you requested stop",
            "budget_exhausted": "not completed: the model-call budget for this attempt was exhausted",
            "horizon_reached": "not completed: the attempt horizon (control steps) was used up"}[status]
    summary = {"attempt": k, "outcome": status, "outcome_text": text, "decisions_total": len(decisions),
               "decisions_executed": sum(d["executed"] for d in decisions), "model_calls": budget.attempt_used, "steps_used": obs["steps_used"],
               "simulated_time_s": round(obs["steps_used"] / 30.0, 1), "interaction_ids": ids}
    history.end_attempt(summary)
    wjson(adir / "summary.json", dict(summary, model_wall_time_s=round(model_time, 1), attempt_wall_time_s=round(time.monotonic() - t_attempt, 1),
                                      arm=meta["arm"], start=meta["start"], object=T.obj, memory_records_after=None if store is None else len(store)))
    return summary


def history_dump(h):
    return {"interactions": h.interactions, "attempt_summaries": h.attempt_summaries, "running": h.running, "next_id": h.next_id}


def history_load(d):
    h = History()
    h.interactions, h.attempt_summaries, h.running, h.next_id = copy.deepcopy(d["interactions"]), copy.deepcopy(d["attempt_summaries"]), copy.deepcopy(d["running"]), d["next_id"]
    return h


LIGHT = False


def make_task(obj, render=True, mode="lift", tray_xy=None):
    from cup_transfer.cup_task import CupTask
    return CupTask(obj, specs(), mode=mode, instruction=INSTR[obj], render=render, horizon=HORIZON, tray_xy=tray_xy, light=LIGHT)


def task_cfg(T, policy_name, model, extra):
    d = T.describe()
    d["attempt_horizon_steps"] = HORIZON
    d["instruction"] = T.instruction
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": policy_name, "task": d, "model": model,
            "budgets": {"model_calls_per_attempt": CALLS_PER_ATTEMPT, "attempt_horizon_steps": HORIZON, "steps_per_decision": list(STEPS)},
            "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip(), **extra}


def make_policy(a, T):
    if a.policy == "smoke":
        return SmokePolicy(T), {"requested": "scripted smoke policy (not Astra)"}
    p = CodexAstra(a.model, a.reasoning, a.timeout, schema_fn=response_schema_cup)
    v = subprocess.run([p.binary, "--version"], capture_output=True, text=True).stdout.strip()
    return p, {"requested": a.model, "reasoning_effort": a.reasoning, "codex_version": v}


# ---------------------------------------------------------------------------- stages
def stage_source(a):
    out = SOURCE / ("smoke" if a.policy == "smoke" else "run")
    out.mkdir(parents=True, exist_ok=True)
    T = make_task("cup_a", render=True)
    policy, model = make_policy(a, T)
    cfg = task_cfg(T, a.policy, model, {"stage": "source", "starts": ["s1", "s2", "s3"]})
    wjson(out / "config.json", cfg)
    store = S1.ExperienceStore()
    history = History()
    budget = Budget(3, CALLS_PER_ATTEMPT, 3 * CALLS_PER_ATTEMPT)
    summaries = []
    status, err = "complete", None
    try:
        for k, sid in enumerate(["s1", "s2", "s3"], start=1):
            summaries.append(run_attempt(T, policy, history, budget, out / f"attempt_{k}", k, S1.Nominal(), store, True,
                                         {"arm": "source", "start": sid, "provenance": "source"}, cfg))
            store.dump(out / "D_A.jsonl")
            wjson(out / "history_final.json", history_dump(history))
    except Exception as ex:  # noqa: BLE001
        import traceback
        status, err = "infrastructure_error", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
    store.dump(out / "D_A.jsonl")
    wjson(out / "history_final.json", history_dump(history))
    wjson(out / "result.json", {"status": status, "error": err, "attempts": summaries, "records": len(store), "model_calls_total": budget.total_used})
    print("RUN_DONE", status, flush=True)
    T.close()


def stage_target(a):
    src = SOURCE / ("smoke" if a.policy == "smoke" else "run")
    D_A = S1.ExperienceStore.load(src / "D_A.jsonl")
    hist0 = json.loads((src / "history_final.json").read_text())
    hyper_p = ROOT / "method_configurations" / ("hyper_smoke.json" if a.policy == "smoke" else "hyper.json")
    if not hyper_p.exists():
        hyper_p.write_text(json.dumps(S1.select_hyper(D_A), indent=1))
    hyper = json.loads(hyper_p.read_text())["best"]
    T = make_task(a.obj, render=True)
    policy, model = make_policy(a, T)
    methods = a.methods.split(",") if a.methods else METHODS
    starts = a.starts.split(",")
    root = LOGS / ("smoke" if a.policy == "smoke" else "run") / a.obj
    status, err = "complete", None
    try:
        for sid in starts:                                   # start-major: complete matched conditions across all methods
            for mi, mname in enumerate(methods):
                cdir = root / sid / mname
                if (cdir / "result.json").exists() and json.loads((cdir / "result.json").read_text()).get("status") == "complete":
                    continue
                if cdir.exists():                      # interrupted earlier (e.g. Astra unavailable): kept aside, restarted from scratch
                    cdir.rename(cdir.parent / f"{mname}_interrupted_{time.strftime('%Y%m%d_%H%M%S')}")
                cdir.mkdir(parents=True, exist_ok=True)
                method, store, update = make_method(mname, D_A, hyper, seed=1000 * (1 + starts.index(sid)) + mi)
                history = history_load(hist0)
                cfg = task_cfg(T, a.policy, model, {"stage": "target", "object": a.obj, "start": sid, "method": mname, "hyper": hyper,
                                                    "source_records": len(D_A), "memory_at_start": None if store is None else len(store)})
                wjson(cdir / "config.json", cfg)
                if isinstance(policy, SmokePolicy):
                    policy.queue = []
                budget = Budget(2, CALLS_PER_ATTEMPT, 2 * CALLS_PER_ATTEMPT)
                summ = []
                for k in (1, 2):
                    summ.append(run_attempt(T, policy, history, budget, cdir / f"attempt_{k}", k, method, store, update,
                                            {"arm": mname, "start": sid}, cfg))
                    if isinstance(policy, SmokePolicy):
                        policy.queue = []
                    if summ[-1]["outcome"] == "success":
                        break
                if store is not None:
                    store.dump(cdir / "memory_after.jsonl")
                wjson(cdir / "result.json", {"status": "complete", "attempts": summ, "model_calls_total": budget.total_used,
                                             "first_success": next((s["attempt"] for s in summ if s["outcome"] == "success"), None)})
                print("CONDITION_DONE", a.obj, sid, mname, [s["outcome"] for s in summ], flush=True)
    except Exception as ex:  # noqa: BLE001
        import traceback
        status, err = "infrastructure_error", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
        print("ERROR", err, flush=True)
    print("RUN_DONE", status, flush=True)
    T.close()


def stage0(a):
    """Assets, reset behaviour, readback of the simulator's mass properties / material, command limits, timing, and a
    scripted feasibility grasp (the reference grasp re-anchored to each object) through the common executor."""
    from cup_transfer.cup_task import execute_decision
    T = make_task(a.obj, render=a.render)
    out = ROOT / "asset_and_task_definitions" / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    ob = T.U.scene.rigid_objects["target"]
    v = ob.root_physx_view
    rep = {"object": a.obj, "spec": T.spec,
           "readback": {"mass": v.get_masses()[0].tolist(), "com_body": v.get_coms()[0].tolist(), "inertia": v.get_inertias()[0].tolist(),
                        "material_static_dynamic_restitution_first_shape": v.get_material_properties()[0, 0].tolist(), "n_shapes": int(v.max_shapes)},
           "starts": {}, "feasibility": []}
    for sid in ("s1", "s2", "s3", "t1", "t2"):
        T.set_start(STARTS[sid])
        T.reset()
        st = T.obj_state()
        rep["starts"][sid] = {"commanded_xy": [0.36 + STARTS[sid][0], -0.13 + STARTS[sid][1]], "rest_pos": st["pos"].round(4).tolist(),
                              "up_z": round(float(st["up"][2]), 4), "speed_after_settle": round(float(np.linalg.norm(st["vel"])), 4)}
    for rep_i in range(2):
        T.set_start(STARTS["t1"])
        T.reset()
        st = T.obj_state()
        tl = open(out / f"{a.obj}_feasibility_transitions.jsonl", "a")
        t0 = time.time()
        steps0 = T.steps_used
        outcome = []
        for i, d in enumerate(reference_grasp(st["pos"], T.R, T.H)):
            obs, res, rows = execute_decision(T, d, S1.Nominal(), None, {"episode_id": f"stage0/{rep_i}", "decision": i, "arm": "stage0", "attempt": rep_i}, tl, update=False)
            outcome.append({"mode": d["interaction_mode"], "object": res["object"], "success": res["task_complete"], "wrist_err_cm": res["hands"]["right"].get("position_error_cm")})
            if res["task_complete"]:
                break
        tl.close()
        rep["feasibility"].append({"steps": T.steps_used - steps0, "wall_s": round(time.time() - t0, 1), "steps_per_s": round((T.steps_used - steps0) / max(time.time() - t0, 1e-6), 1),
                                   "success": bool(T.U.success[0].item()), "max_rise_cm": round(100 * (max(r["obj"][2] for r in T.trace) - T.rest_z), 2), "decisions": outcome})
        if a.render and rep_i == 0:
            from PIL import Image
            Image.fromarray(T._image("main_camera", (960, 540))).save(out / f"{a.obj}{'_light' if LIGHT else ''}_main_after_grasp.png")
    if a.render:
        T.set_start(STARTS["t1"])
        obs = T.reset()
        from PIL import Image
        Image.fromarray(obs["images"]["main_camera"]).save(out / f"{a.obj}{'_light' if LIGHT else ''}_main_start.png")
        Image.fromarray(obs["images"]["right_hand_camera"]).save(out / f"{a.obj}{'_light' if LIGHT else ''}_right_hand_start.png")
    if a.dev_conservative:                                # the brief's development check, scripted (no Astra calls)
        import cup_transfer.cup_task as CT
        rep["conservative_check"] = []
        for setting, ex in (("normal", {"lin_cap_cm": 1.0, "closure_boost": 0.0}), ("conservative", {"lin_cap_cm": 0.5, "closure_boost": 0.15})):
            CT.EXEC.update(ex)
            for pert in ((0.0, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, -0.04), (0.04, 0.0, 0.0)):
                T.set_start(STARTS["t1"])
                T.reset()
                st = T.obj_state()
                tl = open(out / f"{a.obj}_conservative_transitions.jsonl", "a")
                seq = reference_grasp(np.array(st["pos"]) + np.array(pert), T.R, T.H)
                for i, d in enumerate(seq):
                    if setting == "conservative":
                        d["steps"] = int(d["steps"] * 2)
                    obs, res, rows = execute_decision(T, d, S1.Nominal(), None, {"episode_id": f"cons/{setting}", "decision": i, "arm": "dev", "attempt": 0}, tl, update=False)
                    if res["task_complete"]:
                        break
                tl.close()
                rep["conservative_check"].append({"setting": setting, "aim_error_m": list(pert), "success": bool(T.U.success[0].item()),
                                                  "max_rise_cm": round(100 * (max(r["obj"][2] for r in T.trace) - T.rest_z), 2), "steps": T.steps_used})
        CT.EXEC.update({"lin_cap_cm": 1.0, "closure_boost": 0.0})
        print("CONSERVATIVE", json.dumps(rep["conservative_check"]), flush=True)
    wjson(out / f"{a.obj}{'_light' if LIGHT else ''}.json", rep)
    print("STAGE0", json.dumps({k: rep[k] for k in ("readback", "starts")})[:1500], flush=True)
    print("FEASIBILITY", json.dumps([{k: f[k] for k in ("success", "max_rise_cm", "steps_per_s")} for f in rep["feasibility"]]), flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage0", "source", "target"])
    ap.add_argument("--obj", default="cup_a")
    ap.add_argument("--methods", default="")
    ap.add_argument("--starts", default="t1,t2")
    ap.add_argument("--policy", choices=["codex", "smoke"], default="codex")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--dev-conservative", action="store_true")
    ap.add_argument("--light-render", action="store_true", help="render only what Astra receives (for runs after 2026-10-12's primary run)")
    a = ap.parse_args()
    global LIGHT
    LIGHT = a.light_render
    {"stage0": stage0, "source": stage_source, "target": stage_target}[a.stage](a)


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
