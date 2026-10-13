"""Batch 3 (delegation study, 2026-10-13), simulator part. Isaac, env hongyi, cwd egovla_release/egovla, PYTHONPATH=astra_explore.

    python -m cup_transfer.batch3 online --obj cup_b --starts v01,v02 --specs verify [--smoke]
        FRESH EXECUTIONS of local methods under the common interface (local_exec): cup A's program re-anchored to the object up to
        the pre-closing decision, then every method spec runs its own scene - up to 3 grasp attempts, each restored to the same
        pre-closing snapshot (reset-based adaptation), the method's correction executed through the persistent reference offset,
        the measured task-aligned outcome fed back. Source memories come from the manifest's selections (a development-family
        target never sees its own records).
    python -m cup_transfer.batch3 propose --obj cup_b --starts v05 --set pilot     (rendered; Astra)
        ASTRA'S GRASP PROPOSAL through the hand-off: Astra runs its normal decision loop (same prompt and schema as the cup-transfer
        runs, its own history from the cup A source run, executed through the original executor) until it issues its first decision
        that closes the active hand. That decision is NOT executed: it is Astra's grasp proposal (wrist reference + closing command),
        and the state before it is the pre-closing state handed to the local executor. Cap: PROPOSAL_CAP calls.
    python -m cup_transfer.batch3 pbank --obj cup_b --starts v05 --set pilot       (no render)
        every bank correction at a proposal's pre-closing state (replayed from Astra's executed decisions; replay drift recorded)
    python -m cup_transfer.batch3 systems --obj cup_b --starts v05 --set pilot --systems E0,E1,E2,E3 [--policy smoke]
        the compared systems on one proposal: E0 Astra chooses the correction with current geometry feedback, E1 Astra with the same
        geometry plus structured experience records, E2 / E3 the frozen local methods (decisions.json); same bank, budget, feedback
Outputs: experiments/cup_delegation_2026-10-13/.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_transfer import system1 as S1  # noqa: E402
from cup_transfer import run_cup as RC  # noqa: E402
from cup_transfer import batch1 as B1  # noqa: E402
from cup_transfer import batch2 as B2  # noqa: E402
from cup_transfer import batch3_offline as BO3  # noqa: E402
from cup_transfer import local_exec as LX  # noqa: E402
from cup_transfer.prompt_cup import build_prompt_cup, response_schema_cup, validate_cup  # noqa: E402

OUT = BO3.OUT


def _starts(seed, prefix, n):
    rng = np.random.default_rng(seed)
    return {f"{prefix}{i + 1:02d}": tuple(np.round(rng.uniform(-0.02, 0.02, 2), 4).tolist()) for i in range(n)}


# fresh development starts (never used before this study): object offsets (m) drawn once, seeded; the locked test draws its own
FRESH = _starts(20261015, "v", 12)


def spec_name(s):
    if s["code"] == "A0":
        return "A0"
    out = f"{s['code']}_{s['rep']}_h{s['h']}_b{s['budget']}_s{s['selection']}_{s['retry']}"
    return out + (f"_w{s['update_w']}" if s["code"] in ("A3", "A5") else "")


def presets(name):
    E = json.loads((OUT / "config" / "decisions.json").read_text())["amendment_1"]["chosen"]
    hcf, rep, hctx = E["E2_family"]["h"], E["E3"]["rep"], E["E3"]["h"]
    cf = lambda code, k, retry="no_repeat", w=1.0: {"code": code, "rep": "none", "h": hcf, "budget": 64, "selection": k, "retry": retry, "update_w": w}
    cx = lambda code, k, retry="no_repeat", w=1.0: {"code": code, "rep": rep, "h": hctx, "budget": 64, "selection": k, "retry": retry, "update_w": w}
    if name == "verify":                     # experiment A, fresh executions of every method at the chosen settings
        return [{"code": "A0"}] + [f(c, k) for k in range(3) for c, f in (("A1", cf), ("A2", cf), ("A3", cf), ("A4", cx), ("A5", cx))]
    if name == "expC":                       # experiment C: update strength x retry rule (+ the context-free methods, same rules)
        out = []
        for k in range(3):
            for retry in ("no_repeat", "confirm_once"):
                out += [cx("A4", k, retry), cx("A5", k, retry, 0.25), cx("A5", k, retry, 1.0), cf("A2", k, retry), cf("A3", k, retry, 1.0)]
        return out
    raise ValueError(name)


def make_method(s, obj):
    if s["code"] == "A0":
        return LX.LocalMethod("A0", BO3.bank())
    return BO3.method(s["code"], s["selection"], s["budget"], s["rep"], obj, update_w=s.get("update_w", 1.0), retry=s.get("retry", "no_repeat"),
                      mk={"h": s["h"]})


def to_pre_close(T, start_xy, tl):
    """Reset at a start offset and run cup A's re-anchored program up to the pre-closing decision (nominal executor)."""
    from cup_transfer.cup_task import PersistentOffset, execute_decision
    B1.lean_reset(T, start_xy)
    T.persist = PersistentOffset("right")
    seq = B2.program(T)
    i_dec, k_done = B2.POINTS["pre_close"]
    for j in range(i_dec):
        execute_decision(T, seq[j], S1.Nominal(), None, {"episode_id": "prefix", "decision": j, "arm": "prefix", "attempt": 0}, tl, update=False)
    return seq, i_dec, k_done


def setup(job, extra, obj, smoke):
    global OUT
    if smoke:
        OUT = OUT / "smoke"
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    T = RC.make_task(obj, render=False)
    B1.lean(T)
    B1.OUT = OUT / "manifests"
    B1.manifest(T, job, dict(extra, stage=job, task_aligned_success=BO3.manifest()["objective"], tag="smoke" if smoke else ""))
    return T


def cmd_online(a):
    specs = presets(a.specs)
    T = setup(f"online_{a.specs}_{a.obj}_{a.starts.replace(',', '-')}", {"object": a.obj, "starts": {s: FRESH[s] for s in a.starts.split(",")},
                                                                        "specs": specs}, a.obj, a.smoke)
    stage = "mechanism" if a.specs in ("verify", "expC") else "initialization"
    out = OUT / stage
    out.mkdir(parents=True, exist_ok=True)
    tl = open(out / f"transitions_{a.specs}_{a.obj}.jsonl", "a")
    fout = out / f"online_{a.specs}_{a.obj}.jsonl"
    done = set()
    if fout.exists():                                        # resume: scenes already finished are skipped
        done = {(r["start"], r["spec_name"]) for r in B2.jl(fout)}
    for start in a.starts.split(","):
        t_pre = time.time()
        seq, i_dec, k_done = to_pre_close(T, FRESH[start], tl)
        sp = B2.snap(T)
        st = B2.state_record(T, start)
        prefix_s = time.time() - t_pre
        for s in specs:
            name = spec_name(s)
            if (start, name) in done:
                continue
            m = make_method(s, a.obj)
            full = []

            def execute(cand):
                tr = time.time()
                B2.restore(T, sp)
                reset_s = time.time() - tr
                t0 = time.time()
                meth = S1.Nominal() if cand["id"] == "nominal" else B2.Forced(cand["dp_cm"], None)
                rows, geo, oc = B2.rollout(T, seq, i_dec, k_done, meth, tl)
                full.append({"candidate": cand["id"], "outcome": oc, "end_of_close": B2.end_of_close(geo), "events": T.persist.events})
                return oc, {"execution_s": round(time.time() - t0, 2), "reset_s": round(reset_s, 3)}
            rep = LX.run_scene(m, st, execute)
            row = {"object": a.obj, "start": start, "start_xy": FRESH[start], "spec": s, "spec_name": name, "prefix_s": round(prefix_s, 1), **rep,
                   "executions": full, "label": "FRESH EXECUTION", "written": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(fout, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            print("ONLINE", a.obj, start, name, [(x["picked"], x.get("success")) for x in rep["attempts"]], rep["status"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- Astra hand-off
PROPOSAL_CAP = 8                                   # Astra calls for one grasp proposal (invalid-response retries included)
LIFT_DELTA = np.array([0.0013, 0.0002, 0.0932])    # cup A's program: lift target minus closing target (m), kept with Astra's closing
LIFT_STEPS, HOLD_STEPS = 30, 45                    # cup A's program


def wj(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def start_table(which):
    if which == "pilot":
        return FRESH
    if which == "locked":
        return {k: tuple(v) for k, v in json.loads((BO3.OUT / "config" / "prospective.json").read_text())["starts"].items()}
    raise ValueError(which)


def closes_hand(dec):
    side = dec.get("active_hand")
    if side not in ("left", "right"):
        return False
    d = dec[side]
    return dec.get("interaction_mode") == "close" or max(float(d["finger_base"]), float(d["finger_tip"])) >= 0.3


def closure_program(close_dec, wrist_now=None, dirs_now=None):
    """The local executor's program after the hand-off: Astra's closing decision (a hold-in-place closing becomes a target at the
    measured wrist, measured orientation), then the existing program's lift (cup A: +LIFT_DELTA, 30 steps) and hold (45 steps)
    with the closing decision's orientation and closures; the other hand keeps its command."""
    c = copy.deepcopy(close_dec)
    side = c["active_hand"]
    if not c[side]["move"]:
        c[side].update(move=True, position_m=[float(x) for x in wrist_now], fingers_dir=list(dirs_now[0]), palm_dir=list(dirs_now[1]))
        c["handoff_note"] = "closing decision had move=false: converted to a target at the measured wrist pose"
    lift = copy.deepcopy(c)
    lift.update(interaction_mode="lift", steps=LIFT_STEPS, intent="local executor: existing program lift")
    lift[side]["position_m"] = [float(a + b) for a, b in zip(c[side]["position_m"], LIFT_DELTA)]
    hold = copy.deepcopy(lift)
    hold.update(interaction_mode="hold", steps=HOLD_STEPS, intent="local executor: existing program hold")
    return [c, lift, hold]


def cmd_propose(a):
    from run import Budget
    from cup_transfer.cup_task import execute_decision
    global OUT
    starts = start_table(a.set)
    if a.smoke:
        OUT = OUT / "smoke"
    T = RC.make_task(a.obj, render=True)
    pa = argparse.Namespace(policy=a.policy, model="gpt-6-astra", reasoning="medium", timeout=180.0)
    policy, model = RC.make_policy(pa, T)
    hist0 = json.loads((RC.SOURCE / ("smoke" if a.policy == "smoke" else "run") / "history_final.json").read_text())
    stage = "prospective" if a.set == "locked" else "mechanism"
    for start in a.starts.split(","):
        pdir = OUT / stage / "proposals" / a.obj / start
        if (pdir / "proposal.json").exists():
            continue
        if pdir.exists():
            pdir.rename(pdir.parent / f"{start}_interrupted_{time.strftime('%Y%m%d_%H%M%S')}")
        pdir.mkdir(parents=True)
        cfg = RC.task_cfg(T, a.policy, model, {"stage": "proposal", "object": a.obj, "start": start, "start_xy": starts[start], "cap": PROPOSAL_CAP})
        wj(pdir / "config.json", cfg)
        history = RC.history_load(hist0)
        budget = Budget(1, PROPOSAL_CAP, PROPOSAL_CAP)
        budget.new_attempt()
        T.persist = None                                    # Astra's own decisions run through the original executor
        T.set_start(starts[start])
        t_scene = time.time()
        obs = T.reset()
        timeline = [{"what": "reset", "t0": t_scene, "t1": time.time()}]
        tl = open(pdir / "transitions.jsonl", "a")
        executed, decisions, prev_main, note, status, close = [], [], None, "", None, None
        di = 0
        while True:
            if obs["task_complete"]:
                status = "task_complete_before_handoff"
                break
            if not budget.can_call():
                status = "cap_reached"
                break
            ddir = pdir / f"decision_{di:02d}"
            images = dict(obs["images"])
            if prev_main is not None:
                images["main_camera_before_previous_decision"] = prev_main
                note = " image 5 = 'main_camera' as it was BEFORE your previous decision (compare with image 1 to see what changed)."
            bview = dict(budget.view(1), steps_used_this_attempt=obs["steps_used"], steps_remaining_this_attempt=obs["steps_remaining"])
            prompt = build_prompt_cup(cfg["task"], obs, history.view(), bview, note)
            dec, calls = None, []
            for tr in range(2):
                if not budget.can_call():
                    break
                budget.charge()
                tw0 = time.time()
                raw, m = policy.act(ddir / f"call_{tr}", images, prompt)
                timeline.append({"what": "astra call", "t0": tw0, "t1": time.time(), "decision": di, "try": tr})
                rec = {"try": tr, "meta": m}
                if raw is None:
                    calls.append(dict(rec, accepted=False, reason=m.get("outcome")))
                    continue
                try:
                    dec, warnings = validate_cup(raw, history.known_ids())
                    calls.append(dict(rec, accepted=True, warnings=warnings))
                    break
                except Exception as ex:  # noqa: BLE001
                    calls.append(dict(rec, accepted=False, reason=f"invalid response: {ex}", raw=raw))
            drec = {"decision": di, "steps_before": obs["steps_used"], "calls": calls, "accepted": dec is not None}
            if dec is None:
                decisions.append(drec)
                di += 1
                continue
            history.apply_memory_update(dec["memory_update"])
            drec["decision_content"] = dec
            if dec["stop"]:
                decisions.append(dict(drec, executed=False, stop=True))
                status = "model_stopped"
                break
            if closes_hand(dec):
                close = dec
                decisions.append(dict(drec, executed=False, handoff=True))
                status = "handoff"
                break
            te0 = time.time()
            obs, result, rows = execute_decision(T, dec, S1.Nominal(), None, {"episode_id": f"proposal/{start}", "decision": di, "arm": "proposal", "attempt": 1}, tl, update=False)
            timeline.append({"what": "execution", "t0": te0, "t1": time.time(), "decision": di, "steps": result["steps_executed"]})
            base = {"attempt": 1, "decision": di, "steps_before": drec["steps_before"], "left": dec["left"], "right": dec["right"], "steps": dec["steps"],
                    "interaction_mode": dec["interaction_mode"], "active_hand": dec["active_hand"], "assessment_of_previous_action": dec["assessment_of_previous_action"],
                    "intent": dec["intent"], "hypothesis": dec["hypothesis"], "expected_change": dec["expected_change"], "references": dec["references"]}
            history.add_interaction(dict(base, stop=False, result=result))
            executed.append(dec)
            decisions.append(dict(drec, executed=True, result=result))
            prev_main = images["main_camera"]
            di += 1
        tl.close()
        side = "right"
        pre = {"wrist_pose": T._ee(side).round(5).tolist(), "object_pos": T.obj_state()["pos"].round(5).tolist(), "steps_used": T.steps_used,
               "state": B2.state_record(T, start)}
        fd, pd, _ = T._dirs_world(side)
        prog = closure_program(close, T._ee(side)[:3], (fd.round(4).tolist(), pd.round(4).tolist())) if close is not None else None
        from PIL import Image
        Image.fromarray(obs["images"]["main_camera"]).save(pdir / "pre_close_main_camera.png")
        calls = [c for d in decisions for c in d["calls"]]
        prop = {"object": a.obj, "start": start, "start_xy": starts[start], "status": status, "executed_decisions": executed, "closing_decision": close,
                "closure_program": prog, "pre_close": pre, "decisions": decisions, "astra_calls": len(calls),
                "astra_latency_s": round(sum((c["meta"].get("latency_s") or 0.0) for c in calls), 2),
                "usage": [c["meta"].get("usage") for c in calls], "wall_s": round(time.time() - t_scene, 2), "timeline": timeline,
                "astra_running_memory": history.running, "policy": a.policy, "label": "SMOKE (scripted policy)" if a.policy == "smoke" else "ASTRA"}
        wj(pdir / "proposal.json", prop)
        print("PROPOSAL", a.obj, start, status, "calls", len(calls), "wall", prop["wall_s"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def load_proposal(stage, obj, start):
    return json.loads((OUT / stage / "proposals" / obj / start / "proposal.json").read_text())


def to_proposal(T, prop, tl):
    """Replays Astra's executed proposal decisions from the same start in a no-render simulator; returns the program for the
    local executor and the replay drift against the rendered proposal run."""
    from cup_transfer.cup_task import PersistentOffset, execute_decision
    B1.lean_reset(T, prop["start_xy"])
    T.persist = PersistentOffset("right")
    for j, d in enumerate(prop["executed_decisions"]):
        execute_decision(T, d, S1.Nominal(), None, {"episode_id": "proposal_replay", "decision": j, "arm": "prefix", "attempt": 0}, tl, update=False)
    w, o = T._ee("right"), T.obj_state()["pos"]
    drift = {"wrist_cm": round(100 * float(np.linalg.norm(w[:3] - np.array(prop["pre_close"]["wrist_pose"][:3]))), 3),
             "object_cm": round(100 * float(np.linalg.norm(o - np.array(prop["pre_close"]["object_pos"]))), 3),
             "steps": [T.steps_used, prop["pre_close"]["steps_used"]]}
    seq = list(prop["executed_decisions"]) + prop["closure_program"]
    return seq, len(prop["executed_decisions"]), 0, drift


def cmd_pbank(a):
    stage = "prospective" if a.set == "locked" else "mechanism"
    T = setup(f"pbank_{a.obj}_{a.starts.replace(',', '-')}", {"object": a.obj, "starts": a.starts, "set": a.set}, a.obj, a.smoke)
    tl = open("/dev/null", "w")
    for start in a.starts.split(","):
        prop = load_proposal(stage, a.obj, start)
        if prop["status"] != "handoff":
            print("PBANK skip", a.obj, start, prop["status"], flush=True)
            continue
        bdir = OUT / stage / "proposal_banks" / a.obj / start
        bdir.mkdir(parents=True, exist_ok=True)
        seq, i_dec, k_done, drift = to_proposal(T, prop, tl)
        sp = B2.snap(T)
        st = B2.state_record(T, start)
        wj(bdir / "state.json", dict(st, replay_drift=drift))
        done = {r["candidate"]["id"] for r in B2.jl(bdir / "rollouts.jsonl")} if (bdir / "rollouts.jsonl").exists() else set()
        for c in [b for b in BO3.bank() if LX.SELECTABLE(b)][:a.max_specs or None]:
            if c["id"] in done:
                continue
            B2.restore(T, sp)
            t0 = time.time()
            rows, geo, oc = B2.rollout(T, seq, i_dec, k_done, S1.Nominal() if c["id"] == "nominal" else B2.Forced(c["dp_cm"], None), tl)
            with open(bdir / "rollouts.jsonl", "a") as f:
                f.write(json.dumps({"start": start, "candidate": c, "rep": 0, "outcome": oc, "end_of_close": B2.end_of_close(geo), "wall_s": round(time.time() - t0, 1)},
                                   default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            print("PBANK", a.obj, start, c["id"], oc["task_success"], oc["max_rise_cm"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- the compared systems (experiment E)
def choice_schema(ids):
    return lambda: {"type": "object", "additionalProperties": False, "required": ["assessment", "correction_id", "reason"],
                    "properties": {"assessment": {"type": ["string", "null"]}, "correction_id": {"type": "string", "enum": list(ids) + ["stop"]},
                                   "reason": {"type": "string"}}}


def geometry_view(state):
    """What every system may observe just before closing, in readable units (the same quantities S1's features are built from)."""
    raw, g = state["raw_state"], state["geometry_before"]
    R, H = float(raw["R"]), float(raw["H"])
    o = np.array(raw["obj_pos"])
    w = np.array(raw["wrist"])
    tips = {k.replace("R_", "").replace("_tip", ""): {"gap_to_outer_wall_cm": v["surface_gap_cm"], "height_above_base_cm": v["height_cm"]}
            for k, v in g["tips"].items()}
    return {"object_diameter_cm": round(200 * R, 2), "object_height_cm": round(100 * H, 2),
            "wrist_relative_to_object_base_cm": (100 * (w - o)).round(2).tolist(), "fingertips": tips,
            "closure_measured": raw["closure"], "index_contact_force_N": round(float(raw["contact"]), 2)}


def experience_records(k, budget, exclude_obj=None):
    """E1's structured memory: the same source records the local methods are initialised with (selection k, budget; a development-family
    target never sees its own records), in readable form."""
    man = BO3.manifest()
    out = []
    for o in BO3.SOURCE_OBJECTS:
        if o == exclude_obj:
            continue
        sc = BO3.scenes(o)
        for st, cid in man["selections"][str(k)][o][:budget // len(BO3.SOURCE_OBJECTS)]:
            S = sc[st]
            r = next(x for x in S["rows"] if x["candidate"]["id"] == cid)
            gv = geometry_view(S["state"])
            out.append({"cup_diameter_cm": gv["object_diameter_cm"], "cup_height_cm": gv["object_height_cm"],
                        "wrist_relative_to_base_cm": gv["wrist_relative_to_object_base_cm"],
                        "fingertips_gap_height_cm": {t: [v["gap_to_outer_wall_cm"], v["height_above_base_cm"]] for t, v in gv["fingertips"].items()},
                        "correction": cid, "outcome": "held" if r["outcome"]["task_success"] else LX.failure_stage(r["outcome"]),
                        "max_rise_cm": r["outcome"]["max_rise_cm"]})
    return out


def choice_prompt(arm, instruction, obj_word, gview, attempts, n_left, bank, running_memory, records=None):
    lines = [
        "You are the planner of a simulated humanoid robot (Unitree H1 with a five-finger dexterous right hand). Your grasp proposal "
        f"for the {obj_word} has been executed up to the moment before closing: the right hand is at your grasp pose, still open. A local "
        "execution layer now runs grasp attempts from this pre-closing state: each attempt closes the hand with your closing command, "
        "lifts the wrist about 9.3 cm and holds still for about 2.8 s; then the scene is reset to this same pre-closing state for the "
        "next attempt. Reply with ONE JSON object matching the provided schema. Use no tools.",
        "",
        f"TASK: \"{instruction}\"",
        "SUCCESS of an attempt (measured by the harness): the environment's lift success (object at least 5 cm above its start and "
        "slow for 1 s) AND, over the last 30 control steps of the hold, the object at least 4 cm up, upright and slow, AND at least two "
        "fingertips within 2 cm of its surface.",
        "",
        "YOUR DECISION: choose the grasp correction for the next attempt from the list below, or \"stop\" to end this scene without "
        "further attempts. A correction translates the wrist reference of your grasp by (dx, dy, dz) cm in the object's frame (the "
        "object is upright, so these are world axes: x forward from the robot, y to the robot's left, z up), from just before closing "
        "through the lift and the hold; hand orientation and closures stay as you proposed. \"nominal\" = your proposal unchanged. "
        f"Grasp attempts left in this scene: {n_left} of 3.",
        "",
        "CORRECTIONS (id: dx, dy, dz in cm): " + "; ".join(f"{b['id']}: {b['dp_cm'][0]:+.2f}, {b['dp_cm'][1]:+.2f}, {b['dp_cm'][2]:+.2f}" for b in bank),
        "",
        "MEASURED JUST BEFORE CLOSING (harness; fingertip gap = distance from the object's axis minus its outer radius, negative = inside "
        "the outer wall's radius; height = above the object's base): " + json.dumps(gview),
        "",
        "ATTEMPTS IN THIS SCENE (measured): " + (json.dumps(attempts) if attempts else "none yet"),
    ]
    if arm == "E1":
        lines += ["", f"EXPERIENCE RECORDS ({len(records)} grasp attempts recorded by the harness on other cups, each from its own pre-closing "
                  "state measured as above, with the correction executed and the measured outcome): " + json.dumps(records)]
    lines += ["", "YOUR RUNNING MEMORY (from your earlier interactions): " + json.dumps(running_memory), "",
              "RESPONSE FIELDS: assessment (what the attempts so far imply, or null before the first attempt); correction_id (one id from "
              "the list, or \"stop\"); reason (1-2 sentences)."]
    return "\n".join(lines)


class AstraChooser:
    """E0 / E1: Astra chooses the correction for every attempt (same bank, same feedback as the local methods)."""

    def __init__(self, arm, policy_factory, sdir, instruction, obj_word, running_memory, records=None, cap=8):
        self.arm, self.pf, self.sdir, self.instr, self.word, self.mem, self.records, self.cap = arm, policy_factory, sdir, instruction, obj_word, running_memory, records, cap
        self.bank = [b for b in BO3.bank() if LX.SELECTABLE(b)]
        self.code, self.name = arm, {"E0": "astra_geometry", "E1": "astra_structured_memory"}[arm]

    def describe(self):
        return {"code": self.arm, "name": self.name, "records": None if self.records is None else len(self.records), "cap": self.cap}

    def begin(self, state):
        self.state, self.att, self.calls, self.log = state, [], 0, []
        self.policy = self.pf(choice_schema([b["id"] for b in self.bank]))

    def choose(self):
        n_left = 3 - len(self.att)
        prompt = choice_prompt(self.arm, self.instr, self.word, geometry_view(self.state), self.att, n_left, self.bank, self.mem, self.records)
        for tr in range(2):
            if self.calls >= self.cap:
                break
            self.calls += 1
            d = self.sdir / f"{self.arm}_attempt{len(self.att) + 1}_call{tr}"
            raw, m = self.policy.act(d, {}, prompt)
            entry = {"attempt": len(self.att) + 1, "try": tr, "meta": {k: m.get(k) for k in ("outcome", "latency_s", "usage")}, "prompt_chars": len(prompt), "raw": raw}
            self.log.append(entry)
            if raw is None or raw.get("correction_id") not in [b["id"] for b in self.bank] + ["stop"]:
                continue
            if raw["correction_id"] == "stop":
                return None, {"astra": "stop", "reason": raw.get("reason"), "calls": self.calls}
            b = next(b for b in self.bank if b["id"] == raw["correction_id"])
            return {"id": b["id"], "dp_cm": b["dp_cm"]}, {"astra_reason": raw.get("reason"), "astra_assessment": raw.get("assessment"), "calls": self.calls,
                                                          "astra_latency_s": m.get("latency_s"), "prompt_chars": len(prompt)}
        return None, {"astra": "no valid choice", "calls": self.calls}

    def observe(self, cand, oc):
        self.att.append({"attempt": len(self.att) + 1, "correction": cand["id"], "held": bool(oc["task_success"]), "failure_stage": LX.failure_stage(oc),
                         "max_rise_cm": oc.get("max_rise_cm"), "final_rise_cm": oc.get("final_rise_cm")})


def system_method(code, obj, cfg, policy_factory, sdir, prop, T):
    if code in ("E0", "E1"):
        recs = experience_records(cfg["selection"], cfg["E1_records_budget"], exclude_obj=obj) if code == "E1" else None
        from cup_transfer.cup_task import OBJ_WORD
        return AstraChooser(code, policy_factory, sdir, T.instruction, OBJ_WORD.get(obj, "cup"), prop.get("astra_running_memory"), recs, cap=cfg["astra_cap"])
    s = cfg[code]
    return make_method(s, obj)


def cmd_systems(a):
    stage = "prospective" if a.set == "locked" else "mechanism"
    cfg = json.loads((BO3.OUT / "config" / ("prospective.json" if a.set == "locked" else "systems_dev.json")).read_text())["systems"]
    T = setup(f"systems_{a.set}_{a.obj}_{a.starts.replace(',', '-')}", {"object": a.obj, "starts": a.starts, "set": a.set, "systems": cfg,
                                                                       "policy": a.policy}, a.obj, a.smoke)
    if a.policy == "smoke":
        def pf(schema):
            class P:
                def act(self, d, images, prompt):
                    Path(d).mkdir(parents=True, exist_ok=True)
                    (Path(d) / "prompt.txt").write_text(prompt)
                    return {"assessment": None, "correction_id": "nominal", "reason": "smoke"}, {"outcome": "response", "latency_s": 0.0}
            return P()
    else:
        from codex_astra import CodexAstra
        pf = lambda schema: CodexAstra("gpt-6-astra", "medium", 180.0, schema_fn=schema)
    tl = open("/dev/null", "w")
    fout = OUT / stage / f"systems_{a.set}_{a.obj}_{a.starts.replace(',', '-')}.jsonl"     # one file per job (no concurrent appends)
    done = {(r["start"], r["system"]) for r in B2.jl(fout)} if fout.exists() else set()
    for start in a.starts.split(","):
        prop = load_proposal(stage, a.obj, start)
        if prop["status"] != "handoff":
            print("SYSTEMS skip", a.obj, start, prop["status"], flush=True)
            continue
        t0 = time.time()
        seq, i_dec, k_done, drift = to_proposal(T, prop, tl)
        sp = B2.snap(T)
        st = B2.state_record(T, start)
        replay_s = time.time() - t0
        order = a.systems.split(",")
        if a.set == "locked":                                # seeded random order per scene (recorded)
            import hashlib
            rng = np.random.default_rng(int(hashlib.md5(f"{a.obj}/{start}".encode()).hexdigest()[:8], 16))
            order = [order[i] for i in rng.permutation(len(order))]
        for pos, code in enumerate(order):
            if (start, code) in done:
                continue
            sdir = OUT / stage / "systems_calls" / a.obj / start
            m = system_method(code, a.obj, cfg, pf, sdir, prop, T)
            full = []

            def execute(cand):
                tr = time.time()
                B2.restore(T, sp)
                reset_s = time.time() - tr
                te = time.time()
                rows, geo, oc = B2.rollout(T, seq, i_dec, k_done, S1.Nominal() if cand["id"] == "nominal" else B2.Forced(cand["dp_cm"], None), tl)
                full.append({"candidate": cand["id"], "outcome": oc, "end_of_close": B2.end_of_close(geo)})
                return oc, {"execution_s": round(time.time() - te, 2), "reset_s": round(reset_s, 3)}
            rep = LX.run_scene(m, st, execute)
            row = {"object": a.obj, "start": start, "system": code, "order": order, "position": pos, "proposal": {"astra_calls": prop["astra_calls"], "astra_latency_s": prop["astra_latency_s"],
                                                                                   "wall_s": prop["wall_s"]},
                   "replay_drift": drift, "replay_s": round(replay_s, 2), **rep, "executions": full, "policy": a.policy,
                   "astra_log": getattr(m, "log", None), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(fout, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            print("SYSTEM", a.obj, start, code, [(x["picked"], x.get("success")) for x in rep["attempts"]], rep["status"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("online")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--starts", default="v01,v02")
    p.add_argument("--specs", default="verify")
    p.add_argument("--max-specs", type=int, default=0, help="smoke: only the first n specs")
    p.add_argument("--smoke", action="store_true")
    for name in ("propose", "pbank", "systems"):
        p = sub.add_parser(name)
        p.add_argument("--obj", default="cup_b")
        p.add_argument("--starts", default="v05")
        p.add_argument("--set", default="pilot", choices=["pilot", "locked"])
        p.add_argument("--policy", default="codex", choices=["codex", "smoke"])
        p.add_argument("--systems", default="E0,E1,E2,E3")
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--max-specs", type=int, default=0)
    a = ap.parse_args()
    if a.max_specs and a.cmd == "online":
        _p = presets
        globals()["presets"] = lambda n: _p(n)[:a.max_specs]
    {"online": cmd_online, "propose": cmd_propose, "pbank": cmd_pbank, "systems": cmd_systems}[a.cmd](a)


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
