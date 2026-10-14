"""Control diagnostics (2026-10-14): does the current executor need a better System 0?

The user's staged plan, first batch. 12 development conditions x 2 repeats x 3 executors = 72 episodes, every executor given
the same coarse grasp proposal (cup A's program, Astra's own proposal in every run so far, re-anchored to the object, with the
condition's error), the same observations, actuator limits and simulated-time budget (the task horizon, 400 control steps).

    C0 current executor      the common segment executor, unchanged
    C1 motion feedback        C0 + per-segment correction of wrist and finger TRACKING errors through the same interface: the
                              next segment's wrist target adds the integrated shortfall of the achieved wrist behind the
                              commanded one (gain 0.5, at most 1.5 cm); each closure command adds 0.5 x (commanded - measured
                              closure), at most +0.15
    C2 local grasp feedback   C0 + a bounded contact / hold loop (rules in C2_RULES), at most 2 regrasps
Every command is instrumented (intended / commanded / achieved wrist and fingers, object motion, contact geometry) and every
failure is classified from simulator state (diagnosis only; no executor reads anything beyond what C2_RULES documents).

    python -m cup_transfer.control_diag run --cond B_nom [--reps r1,r2] [--execs C0,C1,C2] [--smoke]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_env import GROUPS, HANDS  # noqa: E402
from cup_transfer import system1 as S1  # noqa: E402
from cup_transfer import run_cup as RC  # noqa: E402
from cup_transfer import batch1 as B1  # noqa: E402
from cup_transfer import batch2 as B2  # noqa: E402

OUT = Path("/home/licho/workspace/Horsea/experiments/cup_control_2026-10-14")
BUDGET_STEPS = 400                       # the task horizon: every executor's simulated-time budget from reset
WINDOW = B2.HOLD_WINDOW                  # 40 evaluation steps after the program (holding still), as in batches 2-3
REPS = {"r1": (0.0062, -0.0048), "r2": (-0.0091, 0.0057)}     # two new object start offsets (m), fixed for this batch, shared by all
# condition = (object, proposal error): dp_cm = position error of the whole proposal in the object frame (x forward from the robot,
# y to its left = toward the cup for this side grasp, z up); yaw_deg = hand orientation error about the vertical
CONDITIONS = {
    "A_nom": ("cup_a", {}), "C_nom": ("cyl_c", {}), "B_nom": ("cup_b", {}), "g1_nom": ("dev_g1", {}), "g4_nom": ("dev_g4", {}),
    "g2_nom": ("dev_g2", {}),
    "A_x+1": ("cup_a", {"dp_cm": [1.0, 0.0, 0.0]}), "A_y-1": ("cup_a", {"dp_cm": [0.0, -1.0, 0.0]}),
    "C_z+1": ("cyl_c", {"dp_cm": [0.0, 0.0, 1.0]}), "g3_yaw12": ("dev_g3", {"yaw_deg": 12.0}),
    "A_touch": ("cup_a", {"dp_cm": [0.0, 0.8, 0.0]}),              # imperfect initial contact: palm / fingers press the cup
    "C_touchlow": ("cyl_c", {"dp_cm": [0.0, 0.6, -0.8]}),          # imperfect initial contact: fingertips meet it low and early
}
KIND = {"nom": "nominal grasp", "x+1": "pose error", "y-1": "pose error", "z+1": "pose error", "yaw12": "pose error (orientation)",
        "touch": "imperfect initial contact", "touchlow": "imperfect initial contact"}
C2_RULES = {
    "observes": "robot proprioception, fingertip positions from the robot's kinematics, and the tracked object pose and size (the "
                "observations every executor already receives; a state-input prototype, no estimation)",
    "follow_check": "during the lift, after every 3-step segment once the wrist has risen >= 1.5 cm: slip if the object's rise is "
                    "< 0.5 x the wrist's rise since the lift began, or its tilt grew > 10 deg since the end of closing",
    "on_slip": ["put back: wrist to the grasp height, closures held (9 steps)",
                "open: closures to their approach values (12 steps)",
                "retreat: 3 cm against the palm direction, clear of the object (12 steps)",
                "re-approach: to 1 cm short of the adjusted grasp reference along the palm direction, then onto it (15 steps)",
                "adjust the grasp reference (object frame, at most 1.5 cm per axis): if the thumb tip is above the object's top "
                "(height > H - 0.5 cm) or over its opening (radial gap < -0.3 cm): lower by min(1.5, thumb height - (H - 2.0)) cm; "
                "if the index / middle tips' mean gap to the wall > 1.0 cm: move along the palm direction by min(1.5, gap - 0.5) "
                "cm; if neither: 0.5 cm along the palm direction and 0.5 cm down",
                "re-close with every closure +0.10 (24 steps)", "lift at 2/3 speed (45 steps), follow check again", "hold"],
    "smoke_fix": ["2026-10-14 smoke test 1: the first version reopened and moved the hand while touching the cup and dragged it 5 cm; "
                  "retreat / re-approach steps were added and the lift slowed to 2/3 (not 1/2) so two regrasps fit the budget",
                  "smoke test 2: lowering 1.5 cm drove the fingertips into the table (the side grasp's little finger is about 1 cm above "
                  "it); lowering is now capped at the lowest fingertip's clearance minus 0.3 cm, and a blocked adjustment becomes 0.5 cm "
                  "along the palm direction"],
    "budget": "at most 2 regrasps and the 400-step budget; a slip after the last regrasp, or a drop, is a failure",
    "C2b": "C2 with a recalibrated trigger (development, after the first full run showed C2's check firing on normal lifts): once the "
           "wrist has risen >= 3 cm, slip if the object's rise < 0.4 x the wrist's; no tilt rule (successful lifts tilt up to 20 deg); "
           "regrasp rules unchanged",
}


def jdump(o):
    return json.dumps(o, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))


def rot_z(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return [float(c * v[0] - s * v[1]), float(s * v[0] + c * v[1]), float(v[2])]


def proposal(T, cond):
    """The coarse grasp proposal of a condition: cup A's program re-anchored to the object, plus the condition's error."""
    obj, err = CONDITIONS[cond]
    seq = B2.program(T)
    dp = np.array(err.get("dp_cm", [0.0, 0.0, 0.0])) / 100.0
    yaw = float(err.get("yaw_deg", 0.0))
    for d in seq:
        r = d["right"]
        if r["move"]:
            r["position_m"] = [float(a + b) for a, b in zip(r["position_m"], dp)]
            if yaw:
                r["fingers_dir"], r["palm_dir"] = rot_z(r["fingers_dir"], yaw), rot_z(r["palm_dir"], yaw)
    return seq


# ---------------------------------------------------------------------------- executors
class TrackingFeedback:
    """C1: corrects achieved tracking errors through the existing segment interface. Wrist: once the commanded segment target has
    reached the decision's reference (nominal step < 0.3 cm), the remaining shortfall of the achieved wrist is integrated (gain 0.5,
    at most 1.5 cm) and added to the target. Fingers: after closing (lift / hold / transport), each closure command adds
    0.5 x (commanded - measured), at most +0.15 (a finger held back by the object is driven further)."""
    name = "C1_tracking"

    def __init__(self, T, side="right", ki=0.5, max_cm=1.5, kc=0.5, max_closure=0.15):
        self.T, self.side, self.ki, self.max, self.kc, self.maxc = T, side, ki, max_cm / 100.0, kc, max_closure
        self.integ, self.ref, self.log = np.zeros(3), None, []

    def new_decision(self, ref):
        self.ref, self.integ = (None if ref is None else np.asarray(ref, float)), np.zeros(3)

    def plan(self, s, ctx, U_nom, limit, goal, R, H):
        T, side = self.T, self.side
        cur = T._ee(side)[:3]
        u = np.array(U_nom[0], float)
        if self.ref is not None and np.linalg.norm(U_nom[0][:3]) < 0.3:
            self.integ = self.integ + self.ki * (self.ref - cur)
            n = float(np.linalg.norm(self.integ))
            if n > self.max:
                self.integ *= self.max / n
            u[:3] = u[:3] + 100.0 * self.integ
        extra = np.zeros(4)
        if ctx["mode"] in ("lift", "hold", "transport"):
            meas = np.array([T._closure_measured(side).get(g, 0.0) for g in GROUPS])
            cmd = np.array(U_nom[0][3:], float)
            extra = np.clip(self.kc * (cmd - meas), 0.0, self.maxc) * (cmd > 0.05)
            u[3:] = np.clip(cmd + extra, 0.0, 1.0)
        self.log.append({"t": T.steps_used, "mode": ctx["mode"], "wrist_corr_cm": (100 * self.integ).round(3).tolist(), "closure_extra": extra.round(3).tolist()})
        return {"method": self.name, "mode": ctx["mode"], "near": ctx["near"], "u_req": u, "fallback": "tracking feedback"}


def snapshot(T, mode, decision, ref, sent_closure, wrist_lift0=None, obj_lift0=None):
    g = B1.tip_geometry(T)
    st = T.obj_state()
    rest = np.array(T.rest_xy) if hasattr(T, "rest_xy") else None
    out = {"t": T.steps_used, "mode": mode, "decision": decision, "tips": g["tips"], "contact_N": g["contact_N"],
           "object_rise_cm": g["object_rise_cm"], "object_tilt_deg": g["object_tilt_deg"], "wrist_m": g["wrist_m"], "object_m": g["object_m"],
           "object_speed": round(float(np.linalg.norm(st["vel"])), 4),
           "object_xy_shift_cm": None if rest is None else round(100 * float(np.linalg.norm(st["pos"][:2] - rest)), 3),
           "reference_m": None if ref is None else [round(float(x), 5) for x in ref],
           "wrist_to_reference_cm": None if ref is None else round(100 * float(np.linalg.norm(np.array(g["wrist_m"]) - ref)), 3),
           "closure_commanded": list(sent_closure), "closure_measured": [T._closure_measured("right").get(k, 0.0) for k in GROUPS],
           "hand_q_minus_target": g["hand_q_minus_target"]}
    if wrist_lift0 is not None:
        out["wrist_rise_cm"] = round(100 * (g["wrist_m"][2] - wrist_lift0), 3)
        out["object_rise_since_lift_cm"] = round(100 * (st["pos"][2] - obj_lift0), 3)
    return out


class Episode:
    def __init__(self, T, executor, tl):
        self.T, self.ex, self.tl, self.geo, self.rows, self.events = T, executor, tl, [], [], []
        self.method = TrackingFeedback(T) if executor == "C1" else S1.Nominal()

    def run_decision(self, d, j, monitor=None):
        """One program decision, segment by segment (3 control steps), instrumented. Returns 'ok', 'slip' or 'budget'."""
        from cup_transfer.cup_task import execute_decision
        T = self.T
        total, done = int(d["steps"]), 0
        ref = np.array(d["right"]["position_m"]) if d["right"]["move"] else None
        if hasattr(self.method, "new_decision"):
            self.method.new_decision(ref)
        lift0 = (T._ee("right")[2], T.obj_state()["pos"][2]) if d["interaction_mode"] == "lift" else (None, None)
        tilt0 = self.geo[-1]["object_tilt_deg"] if self.geo else 0.0
        while done < total:
            if T.steps_used >= BUDGET_STEPS:
                return "budget"
            dd = copy.deepcopy(d)
            dd["steps"] = min(3, total - done, BUDGET_STEPS - T.steps_used)
            _, res, r = execute_decision(T, dd, self.method, None, {"episode_id": self.ex, "decision": j, "arm": self.ex, "attempt": 0}, self.tl, update=False)
            self.rows += [dict(x, program_decision=j) for x in r]
            sent = r[-1]["lineage"]["sent"]["closure_targets"] if r else [0, 0, 0, 0]
            sn = snapshot(T, d["interaction_mode"], j, ref, sent, *lift0)
            if r:
                sn["wrist_target_sent_m"] = r[-1]["lineage"]["sent"]["wrist_target_m"]
            self.geo.append(sn)
            if res["steps_executed"] == 0:
                break
            done += res["steps_executed"]
            if monitor is not None and d["interaction_mode"] == "lift":
                v = monitor(sn, tilt0)
                if v:
                    self.events.append({"t": T.steps_used, "event": "slip detected", "detail": v})
                    return "slip"
        return "ok"


def c2_follow_check(sn, tilt0):
    """C2's follow check (C2_RULES)."""
    wr = sn.get("wrist_rise_cm")
    if wr is None or wr < 1.5:
        return None
    orise = sn["object_rise_since_lift_cm"]
    if orise < 0.5 * wr:
        return {"reason": "object rise < 0.5 x wrist rise", "wrist_rise_cm": wr, "object_rise_cm": orise}
    if sn["object_tilt_deg"] - tilt0 > 10.0:
        return {"reason": "tilt grew > 10 deg", "tilt_deg": sn["object_tilt_deg"]}
    return None


def c2b_follow_check(sn, tilt0):
    """C2b's follow check (development recalibration of C2's trigger, 2026-10-14): once the wrist has risen >= 3 cm, slip if the
    object's rise is < 0.4 x the wrist's rise; no tilt rule. Calibrated on the C0 / C1 lifts of this batch: at that point every
    successful lift had a ratio >= 0.65 and every failed one <= 0.18; successful lifts tilted the object up to 20 deg."""
    wr = sn.get("wrist_rise_cm")
    if wr is None or wr < 3.0:
        return None
    orise = sn["object_rise_since_lift_cm"]
    if orise < 0.4 * wr:
        return {"reason": "object rise < 0.4 x wrist rise (wrist >= 3 cm)", "wrist_rise_cm": wr, "object_rise_cm": orise}
    return None


def c2_adjust(T, geo_close):
    """C2's grasp-reference adjustment from the fingertip geometry at the end of closing (C2_RULES), object frame = world for upright
    objects; returns (delta_m, rule)."""
    H = T.H
    tips = geo_close["tips"]
    th = tips["R_thumb_tip"]
    pd = np.array(B1.tip_geometry(T)["palm_dir"])
    pd_h = pd.copy()
    pd_h[2] = 0.0
    pd_h /= max(np.linalg.norm(pd_h), 1e-9)
    gaps = [tips[k]["surface_gap_cm"] for k in ("R_index_tip", "R_middle_tip")]
    delta, rules = np.zeros(3), []
    if th["height_cm"] > 100 * H - 0.5 or th["surface_gap_cm"] < -0.3:
        dz = min(1.5, th["height_cm"] - (100 * H - 2.0))
        delta[2] -= max(dz, 0.0) / 100.0
        rules.append(f"thumb above the top / over the opening: lower {max(dz, 0.0):.2f} cm")
    if float(np.mean(gaps)) > 1.0:
        m = min(1.5, float(np.mean(gaps)) - 0.5)
        delta += pd_h * m / 100.0
        rules.append(f"fingers {np.mean(gaps):.2f} cm from the wall: {m:.2f} cm along the palm direction")
    if not rules:
        delta += pd_h * 0.005
        delta[2] -= 0.005
        rules.append("default: 0.5 cm along the palm direction, 0.5 cm down")
    clearance = min(v["height_cm"] for v in tips.values()) - 0.3            # never drive a fingertip into the table
    if -100 * delta[2] > clearance:
        rules.append(f"lowering capped at the fingertip clearance above the table ({max(clearance, 0.0):.2f} cm)")
        delta[2] = -max(clearance, 0.0) / 100.0
    if np.linalg.norm(delta) < 0.002:
        delta = pd_h * 0.005
        rules.append("blocked: 0.5 cm along the palm direction instead")
    delta = np.clip(delta, -0.015, 0.015)
    return delta, rules


def run_episode(T, cond, rep, executor, tl):
    from cup_transfer.cup_task import PersistentOffset
    B1.lean_reset(T, REPS[rep])
    T.persist = PersistentOffset("right")            # inactive: the executor is the common one
    st0 = T.obj_state()
    T.rest_xy = st0["pos"][:2].copy()
    seq = proposal(T, cond)
    ep = Episode(T, executor, tl)
    t_wall = time.time()
    status, regrasps = "program", 0
    # approach and closing: the proposal as given (C1 corrects tracking within it)
    for j in range(3):
        if ep.run_decision(seq[j], j) == "budget":
            status = "budget"
            break
    close_geo = ep.geo[-1] if ep.geo else None
    lift, hold = copy.deepcopy(seq[3]), copy.deepcopy(seq[4])
    close = copy.deepcopy(seq[2])
    if status != "budget":
        monitor = {"C2": c2_follow_check, "C2b": c2b_follow_check}.get(executor)
        while True:
            v = ep.run_decision(lift, 3, monitor)
            if v == "slip" and regrasps < 2:
                regrasps += 1
                delta, rules = c2_adjust(T, close_geo)
                ep.events.append({"t": T.steps_used, "event": f"regrasp {regrasps}", "delta_cm": (100 * delta).round(3).tolist(), "rules": rules})
                grasp_ref = np.array(close["right"]["position_m"])
                pd_h = np.array(B1.tip_geometry(T)["palm_dir"], float)
                pd_h[2] = 0.0
                pd_h /= max(np.linalg.norm(pd_h), 1e-9)
                down = copy.deepcopy(close)
                down.update(interaction_mode="reposition", steps=9)
                down["right"]["position_m"] = grasp_ref.tolist()
                opn = copy.deepcopy(down)
                opn.update(interaction_mode="release", steps=12)
                for k in GROUPS:                                  # the approach's (open) closures
                    opn["right"][k] = seq[1]["right"][k]
                back = copy.deepcopy(opn)
                back.update(interaction_mode="reposition", steps=12)
                back["right"]["position_m"] = (grasp_ref - 0.03 * pd_h).tolist()
                pre = copy.deepcopy(opn)
                pre.update(interaction_mode="approach", steps=8)
                pre["right"]["position_m"] = (grasp_ref + delta - 0.01 * pd_h).tolist()
                move = copy.deepcopy(opn)
                move.update(interaction_mode="approach", steps=7)
                move["right"]["position_m"] = (grasp_ref + delta).tolist()
                close = copy.deepcopy(close)
                close["right"]["position_m"] = (grasp_ref + delta).tolist()
                for k in GROUPS:
                    close["right"][k] = float(min(1.0, close["right"][k] + 0.10))
                lift = copy.deepcopy(lift)
                lift["right"]["position_m"] = (np.array(lift["right"]["position_m"]) + delta).tolist()
                lift["steps"] = int(round(int(seq[3]["steps"]) * 1.5))
                for k in GROUPS:
                    lift["right"][k] = close["right"][k]
                hold = copy.deepcopy(hold)
                hold["right"]["position_m"] = lift["right"]["position_m"]
                for k in GROUPS:
                    hold["right"][k] = close["right"][k]
                out = "ok"
                for jj, d in enumerate((down, opn, back, pre, move, close), start=10 * regrasps):
                    out = ep.run_decision(d, jj)
                    if out == "budget":
                        break
                if out == "budget":
                    status = "budget"
                    break
                close_geo = ep.geo[-1]
                continue
            if v == "slip":
                status = "slip after the last regrasp"
                ep.events.append({"t": T.steps_used, "event": "budget of regrasps spent"})
            elif v == "budget":
                status = "budget"
            break
        if status == "program":
            if ep.run_decision(hold, 4) == "budget":
                status = "budget"
    hold_q = T.robot.data.joint_pos.clone()
    for _ in range(WINDOW):
        T._run(1, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold_q, record=True)
    ep.geo.append(snapshot(T, "window", 99, None, T.hand_cmd["right"]))
    oc = B2.outcome(T, 0, [g for g in ep.geo if g["mode"] != "window"])
    return {"condition": cond, "object": CONDITIONS[cond][0], "error": CONDITIONS[cond][1], "rep": rep, "start_xy": REPS[rep], "executor": executor,
            "outcome": oc, "status": status, "regrasps": regrasps, "events": ep.events, "steps": T.steps_used,
            "diagnosis": classify(ep.geo, oc, T), "geometry": ep.geo, "c1_log": getattr(ep.method, "log", None), "wall_s": round(time.time() - t_wall, 1),
            "segments": [{"t": x["lineage"]["t_steps_used_after"], "decision": x["program_decision"], "mode": x["interaction_context"]["mode"],
                          "sent_wrist_m": x["lineage"]["sent"]["wrist_target_m"], "achieved_wrist_cm": x["lineage"]["achieved_wrist_cm"],
                          "sent_closure": x["lineage"]["sent"]["closure_targets"], "closure_after": x["lineage"]["closure_measured_after"],
                          "adjustment": x["adjustment"]} for x in ep.rows]}


def classify(geo, oc, T):
    """Failure category from simulator state (diagnosis only)."""
    d = {}
    first_lift = next((i for i, g in enumerate(geo) if g["mode"] == "lift"), len(geo))
    pre = [g for g in geo[:first_lift] if g["mode"] == "approach"]          # the proposal's own approach
    clo = [g for g in geo[:first_lift] if g["mode"] == "close"]             # its first closing
    second = next((i for i, g in enumerate(geo) if g["mode"] == "lift" and i > first_lift and geo[i - 1]["mode"] != "lift"), None)
    lif = [g for g in geo[first_lift:second] if g["mode"] == "lift"]       # its first lift
    ends = {}
    for g in geo:
        if g["mode"] in ("approach", "close", "lift", "hold", "reposition", "release") and g["wrist_to_reference_cm"] is not None:
            ends[(g["decision"], g["mode"])] = g["wrist_to_reference_cm"]
    fc = [g for g in geo if g["mode"] == "close"]
    if fc and fc[-1] is not (clo[-1] if clo else None):                     # a regrasp happened: its final closing
        th = fc[-1]["tips"]["R_thumb_tip"]
        d["final_close"] = {"thumb_gap_cm": th["surface_gap_cm"], "thumb_height_cm": th["height_cm"], "object_shift_cm": fc[-1]["object_xy_shift_cm"],
                            "tilt_deg": fc[-1]["object_tilt_deg"]}
    d["tracking_error_end_cm"] = {f"{k[1]}{k[0]}": v for k, v in ends.items()}
    d["approach_object_shift_cm"] = max([g["object_xy_shift_cm"] or 0.0 for g in pre] + [0.0])
    d["approach_object_tilt_deg"] = max([g["object_tilt_deg"] for g in pre] + [0.0])
    last_close = clo[-1] if clo else None
    if last_close:
        tips = last_close["tips"]
        th = tips["R_thumb_tip"]
        H = 100 * T.H
        d["end_of_close"] = {"thumb_gap_cm": th["surface_gap_cm"], "thumb_height_cm": th["height_cm"],
                             "fingers_near": sum(1 for k in ("R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip")
                                                 if tips[k]["surface_gap_cm"] <= 1.5 and -1.0 <= tips[k]["height_cm"] <= H + 0.5),
                             "contact_N": last_close["contact_N"], "object_shift_cm": last_close["object_xy_shift_cm"],
                             "tilt_deg": last_close["object_tilt_deg"], "closure_measured": last_close["closure_measured"]}
        opposing = (-0.5 <= th["surface_gap_cm"] <= 1.5) and (0.0 <= th["height_cm"] <= H)
        d["end_of_close"]["thumb_opposing"] = bool(opposing)
    if lif:
        wr = max([g.get("wrist_rise_cm") or 0.0 for g in lif] + [0.0])
        orr = max([g.get("object_rise_since_lift_cm") or 0.0 for g in lif] + [0.0])
        d["lift"] = {"wrist_rise_cm": round(wr, 2), "object_rise_cm": round(orr, 2), "follow_ratio": round(orr / wr, 3) if wr > 0.5 else None}
    reach_fail = any(v > 1.0 for (j, m), v in ends.items() if m == "approach" and j == 1)
    if oc["task_success"]:
        d["category"] = "success"
    elif reach_fail:
        d["category"] = "motion execution (the wrist did not reach the commanded grasp pose)"
    elif d["approach_object_shift_cm"] > 1.5 or d["approach_object_tilt_deg"] > 10:
        d["category"] = "planning / approach (the object was pushed or tipped before closing)"
    elif last_close and not (d["end_of_close"]["thumb_opposing"] and d["end_of_close"]["fingers_near"] >= 2):
        d["category"] = "grasp unsuitable (the hand reached the proposal, but thumb and fingers do not enclose the object)"
    else:
        d["category"] = "contact control (a reasonable grasp lost the object while closing / lifting / holding)"
    fr = (d.get("lift") or {}).get("follow_ratio")
    d["object_behaviour"] = ("followed the hand" if oc["task_success"] else
                             "remained supported (never left the table)" if (oc.get("max_rise_cm") or 0) < 0.5 else
                             "pushed away" if d["approach_object_shift_cm"] > 1.5 else
                             "slipped" if fr is not None and fr < 0.8 else "lifted, then lost")
    return d


def cmd_run(a):
    global OUT
    if a.smoke:
        OUT = OUT / "smoke"
    (OUT / "episodes").mkdir(parents=True, exist_ok=True)
    obj = CONDITIONS[a.cond][0]
    T = RC.make_task(obj, render=False)
    B1.lean(T)
    B1.OUT = OUT / "manifests"
    B1.manifest(T, f"control_{a.cond}", {"condition": a.cond, "spec": CONDITIONS[a.cond], "reps": {r: REPS[r] for r in a.reps.split(",")},
                                         "executors": a.execs, "budget_steps": BUDGET_STEPS, "c2_rules": C2_RULES, "tag": "smoke" if a.smoke else ""})
    fout = OUT / "episodes" / f"{a.cond}.jsonl"
    done = {(r["rep"], r["executor"]) for r in map(json.loads, open(fout))} if fout.exists() else set()
    tl = open("/dev/null", "w")
    for rep in a.reps.split(","):
        for ex in a.execs.split(","):
            if (rep, ex) in done:
                continue
            r = run_episode(T, a.cond, rep, ex, tl)
            with open(fout, "a") as f:
                f.write(jdump(r) + "\n")
            print("EPISODE", a.cond, rep, ex, r["outcome"]["task_success"], r["diagnosis"]["category"].split(" (")[0], "regrasps", r["regrasps"],
                  r["status"], r["wall_s"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cond", required=True, choices=list(CONDITIONS))
    p.add_argument("--reps", default="r1,r2")
    p.add_argument("--execs", default="C0,C1,C2")    # C2b: C2 with the recalibrated trigger
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
