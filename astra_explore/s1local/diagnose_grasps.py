"""Why do Astra's grasps fail? Log-only diagnosis of the Astra-only runs (system python).

    python -m s1local.diagnose_grasps --out <dir>

Writes <dir>/diagnosis.json and <dir>/DIAGNOSIS.md.
  1. Parallel grippers: does the robot reach the pose Astra commands (RoboTwin planner logs)? Where is the gripper
     relative to the object when it closes (LIBERO, privileged object positions)?
  2. Dex hand: every hand closure Astra commanded (a hand going from open < 0.3 to closed >= 0.5), classified by what
     happened to the nearest can: lifted and held / lifted then slipped while still closed / pushed away (> 2 cm, no
     lift) / closed on air (fingers reached >= 80 % of the commanded closure and index contact < 1 N) / touched but did
     not hold. For air-grasps: did the wrist reach the commanded pre-grasp pose (< 1.5 cm, < 10 deg)?
  3. Effort: decisions spent before the first lift of each attempt.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

EXP = "/home/licho/workspace/Horsea/experiments/astra_explore"
DEX = ["run_2026-10-10_astra_isaac_stack_can_drawer", "run_2026-10-10_astra_isaac_unload_cans",
       "run_2026-10-10_astra_isaac_insert_and_unload_cans_r4"]
GRIP = {"robotwin_single": "run_2026-10-09_astra_rt_place_container_plate", "robotwin_bimanual": "run_2026-10-09_astra_rt_lift_pot"}
LIBERO = "run_2026-10-09_astra_t48"


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def gripper_tracking():
    out = {}
    for name, r in GRIP.items():
        errs, unreachable, moves = [], 0, 0
        for f in glob.glob(f"{EXP}/{r}/attempt_*/decision_*/decision.json"):
            d = json.load(open(f))
            c, res = d.get("decision_content") or {}, d.get("result") or {}
            for h in ("left", "right"):
                a = (res.get("arms") or {}).get(h)
                if not (c.get(h) or {}).get("move") or not a:
                    continue
                moves += 1
                if a.get("moved_cm", 0) < 0.05 and a.get("position_error_cm", 0) > 1.0:
                    unreachable += 1
                else:
                    errs.append(a["position_error_cm"])
        out[name] = {"commanded_moves": moves, "reached": len(errs), "unreachable": unreachable,
                     "error_median_cm": round(float(np.median(errs)), 2), "error_p90_cm": round(float(np.percentile(errs, 90)), 2)}
    return out


def libero_closures():
    rows = []
    for a in sorted(glob.glob(f"{EXP}/{LIBERO}/attempt_*")):
        ev = {e["decision"]: e for e in jl(f"{a}/evaluation_only.jsonl") if "decision" in e}
        for f in sorted(glob.glob(f"{a}/decision_*/decision.json")):
            d = json.load(open(f))
            c, res = d.get("decision_content") or {}, d.get("result") or {}
            act = c.get("action") or {}
            if act.get("gripper", -1) > 0 and res.get("gripper_width_before_cm", 0) > 3 and "eef_after_m" in res and d.get("decision") in ev:
                kp, ee = ev[d["decision"]]["ketchup_1_pos"], res["eef_after_m"]
                rows.append({"attempt": os.path.basename(a), "decision": d["decision"], "offset_cm": [round(100 * (ee[i] - kp[i]), 1) for i in range(3)],
                             "width_after_cm": res.get("gripper_width_after_cm"), "held": res.get("gripper_width_after_cm", 0) > 1.0})
    missed = [r for r in rows if not r["held"]]
    return {"closures": len(rows), "missed": len(missed), "missed_abs_offset_cm_median": [round(float(np.median([abs(r["offset_cm"][i]) for r in missed])), 1) for i in range(3)],
            "missed_x_offset_range_cm": [min(r["offset_cm"][0] for r in missed), max(r["offset_cm"][0] for r in missed)], "rows": rows}


def dex_closures():
    cats, air_track, effort = {}, [], {"decisions": 0, "before_first_lift": 0, "attempts": 0}
    for r in DEX:
        for a in sorted(glob.glob(f"{EXP}/{r}/attempt_*"), key=lambda p: int(p.split("_")[-1])):
            evl = jl(f"{a}/evaluation_only.jsonl")
            reset = evl[0]
            ev = {e["decision"]: e for e in evl if "decision" in e}
            objs = [k for k in reset if k.endswith("_pos") and "can" in k]
            z0 = {k: reset[k][2] for k in objs}
            decs = {}
            for d in sorted(glob.glob(f"{a}/decision_*")):
                if os.path.exists(f"{d}/decision.json") and os.path.exists(f"{d}/observation.json"):
                    decs[int(d.split("_")[-1])] = (json.load(open(f"{d}/decision.json")), json.load(open(f"{d}/observation.json"))["proprio"])
            live = [i for i in sorted(decs) if decs[i][0].get("decision_content") and not decs[i][0]["decision_content"].get("stop")]
            first = next((i for i in live if any(ev.get(i, {}).get(k, [0, 0, 0])[2] - z0[k] >= 0.05 for k in objs)), None)
            effort["decisions"] += len(live)
            effort["before_first_lift"] += (live.index(first) + 1) if first is not None else len(live)
            effort["attempts"] += 1
            prev = {"left": 0.0, "right": 0.0}
            for i in live:
                dj, o = decs[i]
                c = dj["decision_content"]
                for h in ("left", "right"):
                    cmd = max(c[h]["finger_base"], c[h]["finger_tip"])
                    if cmd >= 0.5 and prev[h] < 0.3:
                        tip = np.array(o[h]["mean_fingertip_position_m_world"])
                        before = ev.get(i - 1, reset)
                        k = min(objs, key=lambda k: np.linalg.norm(np.array(before[k]) - tip))
                        after = ev.get(i, before)
                        moved = float(np.linalg.norm(np.array(after[k][:2]) - np.array(before[k][:2])))
                        lift_j = next((j for j in range(i, i + 4) if ev.get(j, {}).get(k, [0, 0, -9])[2] - z0[k] >= 0.05), None)
                        nxt = decs.get(i + 1)
                        if lift_j is not None:
                            dropped = False
                            for j in range(lift_j + 1, lift_j + 12):
                                if j not in decs or not decs[j][0].get("decision_content") or decs[j][0]["decision_content"].get("stop"):
                                    break
                                if max(decs[j][0]["decision_content"][h]["finger_base"], decs[j][0]["decision_content"][h]["finger_tip"]) < 0.3:
                                    break
                                if ev.get(j, {}).get(k, [0, 0, 9])[2] - z0[k] < 0.02:
                                    dropped = True
                                    break
                            cat = "lifted, then slipped out while still closed" if dropped else "lifted and held until released"
                        elif moved > 0.02:
                            cat = "pushed the can away (no lift)"
                        elif nxt is None:
                            cat = "unknown (attempt ended)"
                        else:
                            cm = nxt[1][h]["closure_measured"]
                            meas = max(cm.get("finger_base", 0), cm.get("finger_tip", 0))
                            cat = "closed on air (no contact)" if meas >= 0.8 * cmd and nxt[1][h]["index_contact_force_N"] < 1.0 else "touched the can, did not hold"
                        cats[cat] = cats.get(cat, 0) + 1
                        if cat.startswith("closed on air"):
                            js = [m for m in live if m < i and decs[m][0]["decision_content"][h]["move"]]
                            hr = decs[max(js)][0].get("result", {}).get("hands", {}).get(h) if js else None
                            if hr:
                                air_track.append((hr.get("position_error_cm", np.nan), hr.get("orientation_error_deg", np.nan)))
                    prev[h] = cmd
    air = np.array(air_track)
    reached = int(((air[:, 0] < 1.5) & (air[:, 1] < 10)).sum())
    return {"categories": cats, "closures": int(sum(cats.values())),
            "air_grasps_with_approach": len(air), "air_reached_commanded_pose": reached,
            "air_position_error_median_cm": round(float(np.nanmedian(air[:, 0])), 2), "air_orientation_error_median_deg": round(float(np.nanmedian(air[:, 1])), 1),
            "effort": effort}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g, lib, dex = gripper_tracking(), libero_closures(), dex_closures()
    (out / "diagnosis.json").write_text(json.dumps({"gripper_tracking": g, "libero_closures": lib, "dex": dex}, indent=1))
    cats = dex["categories"]
    n = dex["closures"]
    failed = n - cats.get("lifted and held until released", 0)
    order = ["closed on air (no contact)", "touched the can, did not hold", "pushed the can away (no lift)", "lifted, then slipped out while still closed"]
    eff = dex["effort"]
    L = ["# Why Astra's grasps fail (log-only diagnosis, 2026-10-11)", "",
         "Source: the Astra-only runs of Series A. Object positions are the simulators' privileged state, used only for this "
         "evaluation. Script: `astra_explore/s1local/diagnose_grasps.py`.", "",
         "## 1. Parallel grippers: the robot does what Astra says; Astra aims wrong", "",
         "| run | commanded moves | reached the commanded pose | median / 90th-percentile error | out of reach (arm did not move) |",
         "|---|---|---|---|---|"]
    for name, v in g.items():
        L.append(f"| {name.replace('_', ' ')} | {v['commanded_moves']} | {v['reached']} | {v['error_median_cm']} / {v['error_p90_cm']} cm | {v['unreachable']} |")
    L += ["", f"LIBERO (Astra commands motions directly): {lib['missed']} of {lib['closures']} gripper closures closed on nothing. At those "
          f"closures the gripper was {lib['missed_x_offset_range_cm'][0]} to {lib['missed_x_offset_range_cm'][1]} cm from the bottle along x "
          f"(median |offset| x / y / z = {' / '.join(str(v) for v in lib['missed_abs_offset_cm_median'])} cm): a depth error, the axis no camera "
          "shows well.", "",
          "## 2. Dex hand: most grasps close on air", "",
          f"{n} hand closures commanded by Astra; {cats.get('lifted and held until released', 0)} lifted the can and held it; {failed} failed.", "",
          "| what happened | grasps | share of failed grasps |", "|---|---|---|"]
    for k in order:
        L.append(f"| {k} | {cats.get(k, 0)} | {100 * cats.get(k, 0) / failed:.0f} % |")
    L += ["", f"Of the {dex['air_grasps_with_approach']} air-grasps, the wrist reached Astra's commanded pre-grasp pose in "
          f"{dex['air_reached_commanded_pose']} (median error {dex['air_position_error_median_cm']} cm, "
          f"{dex['air_orientation_error_median_deg']} deg): Astra aimed at the wrong place. In the rest the arm could not reach the "
          "commanded pose.", "",
          f"Effort: {eff['before_first_lift']} of {eff['decisions']} Astra decisions ({100 * eff['before_first_lift'] / eff['decisions']:.0f} %) "
          f"came before the first lift of an attempt ({eff['attempts']} attempts).", "",
          "## 3. What it means for System 1", "",
          "- About two thirds of failed dex grasps are aiming errors (the hand is not around the can). A precise object position "
          "plus a simple final-approach controller may fix most of them without a learned System 1.",
          "- About one third are contact failures (touched but not held, pushed away, slipped). That is the part a learned "
          "System 1 is for.",
          "- Next measurement: a controlled grasp test in simulation (hand placed at known-good poses relative to the can, then "
          "shifted 0.5-4 cm and rotated 5-30 deg; contact sensors on every fingertip) to measure the tolerance and the "
          "contact-only failure rate.", "",
          "Caveats: only the index finger has a contact sensor; positions are logged per decision, not per control step; "
          "'closed on air' uses the measured closure (>= 80 % of the commanded value) as the no-contact signal."]
    (out / "DIAGNOSIS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
