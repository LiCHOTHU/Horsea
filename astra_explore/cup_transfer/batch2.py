"""Batch 2 (System 1 diagnosis, 2026-10-13): simulator experiments. Isaac, env hongyi, cwd egovla_release/egovla,
PYTHONPATH=astra_explore. No Astra calls: the nominal grasp is cup A's known-good program (Astra's source attempt 1)
re-anchored to the object, with an identical scripted continuation in every branch.

    python -m cup_transfer.batch2 check   [--start d01]          Experiment 0 (simulator half): a correction forced through
                                                                 System 1's execution path (selection bypassed) - does the
                                                                 reference change, stay changed, not accumulate, end cleanly?
    python -m cup_transfer.batch2 duration [--starts d01,d03,d09] Experiment 1: nominal / transient (0.3 s) / persistent
                                                                 correction at approach, just before closing, mid-closing
    python -m cup_transfer.batch2 bank     [--starts d02,d06,d08] Experiment 2: a fixed bank of 32 persistent translation
                                                                 corrections just before closing, full grasp-and-lift each,
                                                                 with System 1-original's objective on predicted and on
                                                                 measured outcomes
Every rollout is a diagnostic interaction (counted in budget.json); nothing is written to any System 1 memory.
Outputs: experiments/cup_batch2_2026-10-13/.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_env import HANDS  # noqa: E402
from cup_transfer import system1 as S1  # noqa: E402
from cup_transfer import run_cup as RC  # noqa: E402
from cup_transfer import batch1 as B1  # noqa: E402

OUT = Path("/home/licho/workspace/Horsea/experiments/cup_batch2_2026-10-13")
DP_B = np.array([-0.5, 0.5, -0.3])        # cm, object frame: the development positive control (cup B grasps that held, sec. 20-21)
POINTS = {"approach": (1, 0), "pre_close": (2, 0), "mid_close": (2, 9)}   # (decision of the program, steps already done in it)
# fresh development starts for experiment 5's online run (never in the correction bank, so no recorded outcome exists there)
EXP5_STARTS = {"e1": (0.005, 0.005), "e2": (-0.005, -0.010), "e3": (0.015, -0.015), "e4": (-0.020, 0.010), "e5": (0.010, 0.020)}
TRANSIENT_STEPS = 9                       # System 1-original's horizon: 3 segments x 3 control steps = 0.3 s
HOLD_WINDOW = 40                          # control steps after the program's hold decision, wrist still
# task-aligned success, frozen before any comparison (2026-10-13): the environment's lift success fired (>= 5 cm, <= 0.10 m/s,
# 30 steps) AND over the last 30 steps of the hold window the object stayed >= 4 cm up, upright (up_z >= 0.9) and slow
# (<= 0.10 m/s) AND at the end at least two fingertip bodies are within 2 cm of its outer surface (held by the hand)
HOLD_MIN_CM, UPRIGHT, SPEED, TIP_NEAR_CM = 4.0, 0.9, 0.10, 2.0
# hard safety violations: a fingertip body more than 0.5 cm below the table top, the object tipped (up_z < 0.7) before the
# lift, or the wrist more than 3 cm from its reference at the end of a decision
TABLE_MARGIN_CM, TIP_OVER, TRACK_CM = 0.5, 0.7, 3.0


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def halton(n, dims=3, skip=1):
    def vdc(i, b):
        f, r = 1.0, 0.0
        while i > 0:
            f /= b
            r += f * (i % b)
            i //= b
        return r
    bases = (2, 3, 5)[:dims]
    return np.array([[vdc(i, b) for b in bases] for i in range(skip, skip + n)])


def candidate_bank():
    """Fixed bank (identical for every state): nominal, 30 persistent translation corrections (Halton points in the
    verified per-axis bound +-1.5 cm, object frame), and the development positive control (labelled)."""
    pts = (halton(30) * 2 - 1) * S1.BOUND[0]
    bank = [{"id": "nominal", "dp_cm": [0.0, 0.0, 0.0], "label": "nominal"}]
    bank += [{"id": f"h{k:02d}", "dp_cm": np.round(p, 3).tolist(), "label": "sampled"} for k, p in enumerate(pts)]
    bank.append({"id": "posctrl", "dp_cm": DP_B.tolist(), "label": "development positive control (not available to the benchmark)"})
    return bank


class Forced:
    """An S1-like method that bypasses selection only: at its first segment it asks the executor to activate a given
    correction (the same path an S1 decision would use), afterwards it returns the (corrected) nominal command."""
    name = "forced"

    def __init__(self, dp_cm, duration_steps=None):
        self.dp, self.dur, self.done = list(map(float, dp_cm)), duration_steps, False

    def plan(self, s, ctx, U_nom, limit, goal, R, H):
        out = {"method": self.name, "mode": ctx["mode"], "near": ctx["near"], "u_req": U_nom[0], "fallback": "forced correction (selection bypassed)"}
        if not self.done:
            out["activate_offset"] = {"dp_obj_cm": self.dp, "duration_steps": self.dur, "reason": "forced"}
            self.done = True
        return out


def program(T):
    dec, obj0 = B1.source_program()
    return [B1.shifted(d, T.obj_state()["pos"] - obj0) for d in dec]


def to_point(T, start, point, log):
    """Reset at a development start and run cup A's re-anchored program up to the intervention point (nominal executor)."""
    from cup_transfer.cup_task import PersistentOffset, execute_decision
    B1.lean_reset(T, B1.DEV_STARTS[start] if start in B1.DEV_STARTS else EXP5_STARTS[start])
    T.persist = PersistentOffset("right")
    seq = program(T)
    i_dec, k_done = POINTS[point]
    for j in range(i_dec):
        execute_decision(T, seq[j], S1.Nominal(), None, {"episode_id": f"prefix/{start}/{point}", "decision": j, "arm": "prefix", "attempt": 0}, log, update=False)
    if k_done:
        d = copy.deepcopy(seq[i_dec])
        d["steps"] = k_done
        execute_decision(T, d, S1.Nominal(), None, {"episode_id": f"prefix/{start}/{point}", "decision": i_dec, "arm": "prefix", "attempt": 0}, log, update=False)
    return seq, i_dec, k_done


def snap(T):
    s = B1.snap(T)
    s["persist"] = copy.deepcopy({k: getattr(T.persist, k) for k in ("dw", "active", "until", "t0", "dp_obj", "closed")})
    return s


def restore(T, s):
    B1.restore(T, s)
    T.persist.reset()
    for k, v in s["persist"].items():
        setattr(T.persist, k, copy.deepcopy(v))


def rollout(T, seq, i_dec, k_done, method_first, tl):
    """From the intervention point: the rest of the current decision with method_first (Forced or Nominal), every later
    decision of the program with the nominal executor, then HOLD_WINDOW control steps holding still. Returns per-segment
    rows (from the executor), geometry samples and the outcome."""
    from cup_transfer.cup_task import execute_decision
    t_start, rows, geo = T.steps_used, [], []
    for j in range(i_dec, len(seq)):
        d = copy.deepcopy(seq[j])
        if j == i_dec:
            d["steps"] = int(d["steps"]) - k_done
        meth = method_first if j == i_dec else S1.Nominal()
        total, done = int(d["steps"]), 0
        while done < total:                         # segment by segment so contact geometry is sampled every 3 steps
            dd = copy.deepcopy(d)
            dd["steps"] = min(3, total - done)
            _, res, r = execute_decision(T, dd, meth, None, {"episode_id": "branch", "decision": j, "arm": "branch", "attempt": 0}, tl, update=False)
            rows += [dict(x, program_decision=j) for x in r]
            ref = np.array(d["right"]["position_m"]) + T.persist.dw
            geo.append(dict(B1.tip_geometry(T), t=T.steps_used, mode=d["interaction_mode"], decision=j, last_of_decision=done + 3 >= total,
                            wrist_to_reference_cm=round(100 * float(np.linalg.norm(T._ee("right")[:3] - ref)), 3),
                            ref_offset_cm=(100 * T.persist.dw).round(3).tolist()))
            done += max(res["steps_executed"], 1) if res["steps_executed"] else total
            if res["steps_executed"] == 0:
                break
    hold = T.robot.data.joint_pos.clone()
    for k in range(HOLD_WINDOW):
        T._run(1, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold, record=True)
    return rows, geo, outcome(T, t_start, geo)


def outcome(T, t_start, geo):
    tr = T.trace
    seg = [r for r in tr if r["t"] > t_start]
    z0 = T.rest_z
    last = seg[-30:]
    rise = [100 * (r["obj"][2] - z0) for r in seg]
    env_success = any(r["success"] for r in seg)
    st = T.obj_state()
    tips = B1.tip_geometry(T)["tips"]
    near = sum(1 for v in tips.values() if v["surface_gap_cm"] <= TIP_NEAR_CM and -1.0 <= v["height_cm"] <= 100 * T.H + 1.0)
    stable = all(100 * (r["obj"][2] - z0) >= HOLD_MIN_CM and r["up_z"] >= UPRIGHT and r["obj_speed"] <= SPEED for r in last)
    lift_idx = next((i for i, g in enumerate(geo) if g["mode"] == "lift"), len(geo))
    tipped_before_lift = any(g["object_tilt_deg"] > np.degrees(np.arccos(TIP_OVER)) for g in geo[:lift_idx])
    below_table = any(v["height_cm"] < -TABLE_MARGIN_CM - 0.0 for g in geo for v in g["tips"].values()
                      if g["object_rise_cm"] < 0.5)          # only while the object is still on the table (height = above its base)
    track = any(g["wrist_to_reference_cm"] > TRACK_CM for g in geo if g["mode"] == "close" and g.get("last_of_decision"))
    return {"task_success": bool(env_success and stable and near >= 2), "env_success": bool(env_success), "stable_last30": bool(stable),
            "tips_near_at_end": near, "max_rise_cm": round(max(rise), 2) if rise else None, "final_rise_cm": round(rise[-1], 2) if rise else None,
            "final_up_z": round(float(st["up"][2]), 3), "safety": {"tipped_before_lift": bool(tipped_before_lift), "tip_below_table": bool(below_table),
                                                                   "wrist_far_from_reference": bool(track)},
            "safety_violation": bool(tipped_before_lift or below_table or track), "steps": T.steps_used - t_start}


def end_of_close(geo):
    g = [x for x in geo if x["mode"] == "close"]
    return None if not g else {"thumb": g[-1]["tips"]["R_thumb_tip"], "index": g[-1]["tips"]["R_index_tip"], "contact_N": g[-1]["contact_N"],
                               "rise_cm": g[-1]["object_rise_cm"], "tilt_deg": g[-1]["object_tilt_deg"], "wrist_to_reference_cm": g[-1]["wrist_to_reference_cm"]}


def budget_add(n, what):
    p = OUT / "budget.json"
    b = json.loads(p.read_text()) if p.exists() else {"diagnostic_rollouts": 0, "by_kind": {}}
    b["diagnostic_rollouts"] += n
    b["by_kind"][what] = b["by_kind"].get(what, 0) + n
    p.write_text(json.dumps(b, indent=1))


def setup(job, extra, smoke=False, obj="cup_b"):
    import cup_transfer.cup_task  # noqa: F401
    global OUT
    if smoke:
        OUT = OUT / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    T = RC.make_task(obj, render=False)
    B1.lean(T)
    B1.OUT = OUT                                      # manifests of this batch go to its own folder
    B1.manifest(T, job, dict(extra, stage=job, s1_original="git tag s1-original", dp_positive_control_cm=DP_B.tolist(),
                             task_aligned_success={"hold_min_cm": HOLD_MIN_CM, "upright_up_z": UPRIGHT, "speed": SPEED, "tips_near_cm": TIP_NEAR_CM,
                                                   "window": "last 30 of the 40-step hold window after the program"},
                             safety={"table_margin_cm": TABLE_MARGIN_CM, "tip_over_up_z": TIP_OVER, "tracking_cm": TRACK_CM}, tag=""))
    return T


# ---------------------------------------------------------------------------- Experiment 0 (simulator half)
def cmd_check(a):
    T = setup("check", {"start": a.start}, a.smoke)
    out = OUT / "exp0"
    out.mkdir(parents=True, exist_ok=True)
    tl = open(out / "transitions.jsonl", "a")
    res = {}
    for name, dur in (("persistent", None), ("transient", TRANSIENT_STEPS)):
        seq, i_dec, k_done = to_point(T, a.start, "pre_close", tl)
        R = __import__("cup_transfer.cup_task", fromlist=["obj_rotation"]).obj_rotation(T)
        dw = R @ (DP_B / 100)
        ref = np.array(seq[i_dec]["right"]["position_m"])
        t0 = time.time()
        rows, geo, oc = rollout(T, seq, i_dec, k_done, Forced(DP_B, dur), tl)
        budget_add(1, "exp0_forced")
        offs = [(r["lineage"]["t_steps_used_after"] - r["lineage"]["steps"], r["lineage"]["reference_offset_cm"], r["program_decision"]) for r in rows]
        active = [o for o in offs if np.linalg.norm(np.array(o[1]) - 100 * dw) < 1e-6]
        zero = [o for o in offs if np.linalg.norm(o[1]) < 1e-9]
        ends = {}
        for g in geo:
            ends[g["decision"]] = g
        reached = {j: g["wrist_to_reference_cm"] for j, g in ends.items()}      # wrist vs the reference in force at that time
        res[name] = {"dw_cm": (100 * dw).round(4).tolist(), "segments": len(offs), "segments_with_the_full_offset": len(active),
                     "segments_with_no_offset": len(zero), "first_segment_t": offs[0][0] if offs else None,
                     "offset_by_segment_cm": [[o[0], o[2]] + list(np.round(o[1], 4)) for o in offs],
                     "wrist_to_corrected_reference_at_decision_ends_cm": reached, "outcome": oc, "events": T.persist.events, "wall_s": round(time.time() - t0, 1)}
        # no accumulation: the commanded reference never exceeds the requested offset
        res[name]["max_offset_norm_cm"] = round(max(float(np.linalg.norm(o[1])) for o in offs), 4)
        print("CHECK", name, json.dumps({k: res[name][k] for k in ("segments", "segments_with_the_full_offset", "segments_with_no_offset", "max_offset_norm_cm", "outcome")}), flush=True)
    # release: after the persistent run, a scripted release decision (hand opens, same reference) must ramp the offset out
    seq, i_dec, k_done = to_point(T, a.start, "pre_close", tl)
    rows, geo, oc = rollout(T, seq, i_dec, k_done, Forced(DP_B, None), tl)
    budget_add(1, "exp0_forced")
    from cup_transfer.cup_task import execute_decision
    T.U.success[:] = False                       # the release must execute although the task already succeeded
    T.U.hold_count[:] = 0
    rel = copy.deepcopy(seq[-1])
    rel.update(interaction_mode="release", steps=12)
    for g in ("finger_base", "finger_tip", "thumb_rotate", "thumb_bend"):
        rel["right"][g] = 0.0
    _, r2, rows2 = execute_decision(T, rel, S1.Nominal(), None, {"episode_id": "release", "decision": 99, "arm": "check", "attempt": 0}, tl, update=False)
    ramp = [r["lineage"]["reference_offset_cm"] for r in rows2]
    steps = np.diff([np.linalg.norm(x) for x in ([res["persistent"]["dw_cm"]] + ramp)])
    res["release"] = {"offset_by_segment_cm": ramp, "largest_decrease_per_segment_cm": round(float(-min(steps)) if len(steps) else 0.0, 4),
                      "limit_per_segment_cm": 0.3 * 3, "events": T.persist.events[-2:]}
    print("RELEASE", json.dumps(res["release"]), flush=True)
    (out / "check.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- Experiment 1
def cmd_duration(a):
    T = setup("duration", {"starts": a.starts, "points": list(POINTS), "transient_steps": TRANSIENT_STEPS}, a.smoke)
    out = OUT / "exp1"
    out.mkdir(parents=True, exist_ok=True)
    tl = open(out / "transitions.jsonl", "a")
    results = []
    for start in a.starts.split(","):
        for point in a.points.split(","):
            seq, i_dec, k_done = to_point(T, start, point, tl)
            sp = snap(T)
            for cond in ("nominal", "transient", "persistent"):
                for rep in range(a.reps):
                    restore(T, sp)
                    meth = S1.Nominal() if cond == "nominal" else Forced(DP_B, TRANSIENT_STEPS if cond == "transient" else None)
                    t0 = time.time()
                    rows, geo, oc = rollout(T, seq, i_dec, k_done, meth, tl)
                    budget_add(1, "exp1")
                    r = {"start": start, "point": point, "condition": cond, "rep": rep, "outcome": oc, "end_of_close": end_of_close(geo),
                         "events": T.persist.events, "wall_s": round(time.time() - t0, 1),
                         "wrist_trajectory": [[g["t"], g["decision"], g["mode"]] + g["wrist_m"] + g["ref_offset_cm"] for g in geo],
                         "geometry": geo}
                    results.append(r)
                    with open(out / "rollouts.jsonl", "a") as f:
                        f.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
                    print("EXP1", start, point, cond, rep, oc["task_success"], oc["max_rise_cm"], oc["safety_violation"], r["wall_s"], flush=True)
    # repeat the decisive contrasts: where transient and persistent differ, run both twice more from the same point
    for start in a.starts.split(","):
        for point in a.points.split(","):
            rs = {c: [x for x in results if x["start"] == start and x["point"] == point and x["condition"] == c] for c in ("transient", "persistent")}
            if rs["transient"] and rs["persistent"] and rs["transient"][0]["outcome"]["task_success"] != rs["persistent"][0]["outcome"]["task_success"]:
                seq, i_dec, k_done = to_point(T, start, point, tl)
                sp = snap(T)
                for cond in ("transient", "persistent"):
                    for rep in range(a.reps, a.reps + a.extra_reps):
                        restore(T, sp)
                        rows, geo, oc = rollout(T, seq, i_dec, k_done, Forced(DP_B, TRANSIENT_STEPS if cond == "transient" else None), tl)
                        budget_add(1, "exp1_repeat")
                        r = {"start": start, "point": point, "condition": cond, "rep": rep, "outcome": oc, "end_of_close": end_of_close(geo),
                             "events": T.persist.events, "repeat_of_decisive_contrast": True, "geometry": geo}
                        with open(out / "rollouts.jsonl", "a") as f:
                            f.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
                        print("EXP1R", start, point, cond, rep, oc["task_success"], oc["max_rise_cm"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- Experiment 2
def cmd_bank(a):
    from cup_transfer.cup_task import make_limit, nominal_sequence
    T = setup(f"bank_{a.obj}_{a.starts.replace(',', '-')}", {"object": a.obj, "starts": a.starts, "point": "pre_close", "bank": candidate_bank()}, a.smoke, obj=a.obj)
    out = OUT / "exp2" / a.obj
    out.mkdir(parents=True, exist_ok=True)
    tl = open(out / "transitions.jsonl", "a")
    hyper = json.loads((RC.ROOT / "method_configurations" / "hyper.json").read_text())["best"]
    D_A = S1.ExperienceStore.load(RC.ROOT / "source_interaction_dataset" / "run" / "D_A.jsonl", frozen=True)
    bank = candidate_bank()[:a.max_cands] if a.max_cands < 32 else candidate_bank()
    from cup_transfer.cup_task import obj_rotation
    for start in a.starts.split(","):
        seq, i_dec, k_done = to_point(T, start, "pre_close", tl)
        sp = snap(T)
        dec = copy.deepcopy(seq[i_dec])
        side, mode = "right", dec["interaction_mode"]
        steps_rem = int(dec["steps"]) - k_done
        # System 1-original's view of every candidate at this state (no simulation): its state, its nominal, its local fit
        start_pose = {s: T._ee(s) for s in HANDS}
        o0 = T.obj_state()
        g = np.array(dec[side]["position_m"])
        goal = {"lift_m": 0.05, "w_target": (g - o0["pos"]) / np.array([T.R, T.R, T.H])}
        r0 = T.raw_state(side)
        s0 = S1.state_vector(r0)
        ctx = {"mode": mode, "near": S1.near_object(s0), "hand": side}
        n = min(3, steps_rem)
        steps_list = [n] + [min(3, max(1, steps_rem - n - 3 * j)) for j in range(2)]
        U_nom = nominal_sequence(T, side, dec[side], start_pose[side], 3, steps_rem, mode)
        limit = make_limit(T, dec, side, steps_list)
        ctrl = S1.ShortHorizonController(D_A, h=hyper["h"], lam=hyper["lam"])
        fit = ctrl.model.fit(s0, limit(U_nom)[0], ctx)
        R = obj_rotation(T)
        state = {"start": start, "point": "pre_close", "state_vector": s0.round(6).tolist(), "raw_state": r0, "ctx": ctx, "U_nom": U_nom.round(5).tolist(),
                 "fit_available": fit is not None, "fit_ess": None if fit is None else round(fit["ess"], 3), "geometry_before": B1.tip_geometry(T)}
        (out / f"state_{start}{a.tag}.json").write_text(json.dumps(state, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
        for c in bank:
            dw = R @ (np.array(c["dp_cm"]) / 100)
            U_req = nominal_sequence(T, side, dec[side], start_pose[side], 3, steps_rem, mode, offset=dw)
            dlt = (U_req - U_nom) / S1.BOUND
            pred = {"supported": None, "J_hat": None}
            if fit is not None:
                cost, _, _, row = ctrl.evaluate(s0, ctx, U_nom, limit, goal, T.R, T.H, fit, 0 if c["id"] == "nominal" else 1, dlt)
                pred = {"supported": row.get("rejected") is None, "J_hat": row.get("cost"), "J_hat_terms": row.get("goal_terms"),
                        "deviation": row.get("deviation"), "smoothness": row.get("smoothness"), "ess": row.get("ess_first_segment"),
                        "predicted_ds_first": row.get("predicted_ds_first")}
            for rep in range(a.reps):
                restore(T, sp)
                t0 = time.time()
                meth = S1.Nominal() if c["id"] == "nominal" else Forced(c["dp_cm"], None)
                rows, geo, oc = rollout(T, seq, i_dec, k_done, meth, tl)
                budget_add(1, "exp2")
                first3 = [r for r in rows if r["program_decision"] == i_dec][:3]
                r = {"start": start, "candidate": c, "rep": rep, "dlt": dlt.round(5).tolist(), "U_req": U_req.round(5).tolist(), "prediction": pred,
                     "outcome": oc, "end_of_close": end_of_close(geo), "events": T.persist.events, "wall_s": round(time.time() - t0, 1),
                     "first3": [{"before": x["observed_state_before"], "after": x["observed_state_after"], "u_app": x["applied_command_after_limits"],
                                 "u_req": x["requested_command"], "ds": x["ds"], "ref_offset_cm": x["lineage"]["reference_offset_cm"]} for x in first3],
                     "geometry": geo}
                with open(out / f"rollouts_{start}{a.tag}.jsonl", "a") as f:
                    f.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
                print("EXP2", start, c["id"], rep, oc["task_success"], oc["max_rise_cm"], pred.get("J_hat"), r["wall_s"], flush=True)
    # limited repeats: nominal, positive control and up to three successful sampled candidates per state, once more
    if a.no_repeats:
        print("RUN_DONE complete", flush=True)
        T.close()
        return
    for start in a.starts.split(","):
        rs = [x for x in jl(out / f"rollouts_{start}{a.tag}.jsonl") if x["rep"] == 0]
        ok = [x["candidate"]["id"] for x in rs if x["outcome"]["task_success"] and x["candidate"]["label"] == "sampled"][:3]
        ids = ["nominal", "posctrl"] + ok
        seq, i_dec, k_done = to_point(T, start, "pre_close", tl)
        sp = snap(T)
        for c in [b for b in bank if b["id"] in ids]:
            restore(T, sp)
            rws, geo, oc = rollout(T, seq, i_dec, k_done, S1.Nominal() if c["id"] == "nominal" else Forced(c["dp_cm"], None), tl)
            budget_add(1, "exp2_repeat")
            with open(out / f"repeats_{start}{a.tag}.jsonl", "a") as f:
                f.write(json.dumps({"start": start, "candidate": c, "rep": a.reps, "outcome": oc, "end_of_close": end_of_close(geo)}) + "\n")
            print("EXP2R", start, c["id"], oc["task_success"], oc["max_rise_cm"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def state_record(T, start):
    side = "right"
    return {"start": start, "point": "pre_close", "state_vector": S1.state_vector(T.raw_state(side)).round(6).tolist(), "raw_state": T.raw_state(side),
            "geometry_before": B1.tip_geometry(T)}


def cmd_exp5(a):
    """Experiment 5 online: System 1-v2 decides the persistent correction at the pre-closing decision of each target start, the
    executor carries it out, the measured grasp-and-lift outcome is stored (updating conditions) before the next attempt
    from the same start. Every attempt is a real rollout; each condition generates its own history."""
    from cup_transfer import s1v2 as V
    T = setup(f"exp5_{a.obj}", {"object": a.obj, "starts": a.starts, "conditions": a.conditions, "rep": a.rep, "source": a.source,
                                "attempts": a.attempts}, a.smoke, obj=a.obj)
    out = OUT / "exp5_online"
    out.mkdir(parents=True, exist_ok=True)
    tl = open(out / "transitions.jsonl", "a")
    bank = candidate_bank()
    src = V.OutcomeMemory()
    for sname in a.source.split(","):               # e.g. cup_a:d01  (that state's recorded bank outcomes)
        obj_s, st_s = sname.split(":")
        S = json.loads((OUT / "exp2" / obj_s / f"state_{st_s}.json").read_text())
        vf = OUT / "exp2" / obj_s / f"state_{st_s}_vec.json"
        if vf.exists():
            S["geometry_before"] = json.loads(vf.read_text())["geometry_before"]
        for r in jl(OUT / "exp2" / obj_s / f"rollouts_{st_s}.jsonl"):
            if r["rep"] == 0 and (r["candidate"]["label"] == "sampled" or r["candidate"]["id"] == "nominal"):
                x, _ = V.features(S, a.rep, r["candidate"]["dp_cm"])
                src.add(V.record(x, r["candidate"], r["outcome"], "source:" + sname))
    conds = {"frozen_source": (src.clone(frozen=True), False), "updating_source": (src.clone(frozen=False), True), "updating_empty": (V.OutcomeMemory(), True)}
    for start in a.starts.split(","):
        seq, i_dec, k_done = to_point(T, start, "pre_close", tl)
        sp = snap(T)
        st = state_record(T, start)
        for cname in a.conditions.split(","):
            mem0, upd = conds[cname]
            sel = V.Selector(mem0.clone(), a.rep, bank)
            for k in range(1, a.attempts + 1):
                pick, x, sc = sel.choose(st)
                x, _ = V.features(st, a.rep, pick["dp_cm"])          # the picked correction's own features
                restore(T, sp)
                t0 = time.time()
                meth = S1.Nominal() if pick["id"] == "nominal" else Forced(pick["dp_cm"], None)
                rows, geo, oc = rollout(T, seq, i_dec, k_done, meth, tl)
                budget_add(1, "exp5_online")
                entry = {"start": start, "condition": cname, "attempt": k, "picked": pick["id"], "dp_cm": pick["dp_cm"], "predicted_q": pick["q"],
                         "ess": pick["ess"], "retrieved": pick["neighbors"], "outcome": oc, "end_of_close": end_of_close(geo), "wall_s": round(time.time() - t0, 1),
                         "memory_before": len(sel.memory.records)}
                if upd:
                    before = {s_["id"]: s_["score"] for s_ in sc}
                    entry["stored"] = sel.memory.add(V.record(x, pick, oc, "target:" + start))
                    _, sc2 = sel.scores(st)
                    after = {s_["id"]: s_["score"] for s_ in sc2}
                    nxt, _, _ = sel.choose(st)
                    entry.update(prediction_change_on_picked=round(after[pick["id"]] - before[pick["id"]], 4), preferred_changed=nxt["id"] != pick["id"],
                                 next_pick=nxt["id"])
                with open(out / f"attempts_{a.obj}_{a.rep}.jsonl", "a") as f:
                    f.write(json.dumps(entry, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
                print("EXP5", start, cname, k, pick["id"], oc["task_success"], oc["max_rise_cm"], flush=True)
                if oc["task_success"]:
                    break
    print("RUN_DONE complete", flush=True)
    T.close()


def cmd_states(a):
    """Re-record the pre-closing state of each start (with per-tip vectors); no rollouts, nothing else changes."""
    T = setup(f"states_{a.obj}", {"object": a.obj, "starts": a.starts}, False, obj=a.obj)
    out = OUT / "exp2" / a.obj
    out.mkdir(parents=True, exist_ok=True)
    tl = open("/dev/null", "w")
    for start in a.starts.split(","):
        to_point(T, start, "pre_close", tl)
        (out / f"state_{start}_vec.json").write_text(json.dumps(state_record(T, start), default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
        print("STATE", a.obj, start, flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--start", default="d01")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("duration")
    p.add_argument("--starts", default="d01,d03,d09")
    p.add_argument("--points", default=",".join(POINTS))
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--extra-reps", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("bank")
    p.add_argument("--starts", default="d02,d06,d08")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--tag", default="", help="suffix for output files (e.g. _r2 for a full repeat of the bank)")
    p.add_argument("--no-repeats", action="store_true")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--max-cands", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("exp5")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--starts", default="e1,e2,e3,e4,e5")
    p.add_argument("--conditions", default="frozen_source,updating_source,updating_empty")
    p.add_argument("--rep", default="geometry")
    p.add_argument("--source", default="cup_a:d01,cup_a:d04")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("states")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--starts", default=",".join(B1.DEV_STARTS))
    a = ap.parse_args()
    {"check": cmd_check, "duration": cmd_duration, "bank": cmd_bank, "exp5": cmd_exp5, "states": cmd_states}[a.cmd](a)


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
