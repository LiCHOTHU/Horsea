"""Per-grasp check of the cup-transfer run (system python with numpy).

    python -m cup_transfer.grasp_check [--run run]

One row per grasp (a decision whose active hand goes from open, < 0.2, to closing, >= 0.3), from the privileged per-step
trace: wrist position relative to the object base when the closing starts, the closure command, index-finger contact
force after the closing decision, the object's rise and tilt while the hand closed, and the object's highest rise before
the next grasp. A grasp "held" when that rise reached 5 cm (the success height). Writes
evaluation_results/<run>/grasp_check.{json,md}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cup_transfer.analyze_cup import METHODS, ROOT, jl

NOMINAL = np.array([-14.5, -5.0, 6.4])  # cup A's grasp: wrist relative to the object base, cm (all three source attempts)


def grasps(adir, obj, label):
    tr = jl(adir / "eval_trace.jsonl")
    if not tr:
        return []
    z0 = tr[0]["obj"][2]
    decs = [json.loads((d / "decision.json").read_text()) for d in sorted(adir.glob("decision_*")) if (d / "decision.json").exists()]
    decs = [d for d in decs if d.get("decision_content") and not d["decision_content"].get("stop")]
    prev, starts = {"left": 0.0, "right": 0.0}, []
    for i, d in enumerate(decs):
        c = d["decision_content"]
        h = c.get("active_hand")
        for s in ("left", "right"):
            m = max(c[s]["finger_base"], c[s]["finger_tip"])
            if s == h and m >= 0.3 and prev[s] < 0.2:
                starts.append((i, s))
            prev[s] = m
    out = []
    for j, (i, s) in enumerate(starts):
        d = decs[i]
        t0 = d["steps_before"]
        t1 = min(t0 + d["result"]["steps_executed"], len(tr) - 1)
        tend = decs[starts[j + 1][0]]["steps_before"] if j + 1 < len(starts) else len(tr)
        hh = d["decision_content"][s]
        wrist = 100 * (np.array(tr[t0]["wrist"][s]) - np.array(tr[t0]["obj"]))
        rise_close = 100 * (tr[t1]["obj"][2] - tr[t0]["obj"][2])
        tilt_close = float(np.degrees(np.arccos(np.clip(min(r["up_z"] for r in tr[t0:t1 + 1]), -1, 1))))
        best = max(100 * (tr[t]["obj"][2] - z0) for t in range(t0, tend))
        out.append({"object": obj, "condition": label, "grasp": j + 1, "step": t0,
                    "wrist_rel_cm": wrist.round(1).tolist(), "offset_from_cup_a_grasp_cm": (wrist - NOMINAL).round(1).tolist(),
                    "at_cup_a_grasp": bool(np.all(np.abs(wrist - NOMINAL) <= 0.3)),
                    "closure_cmd": [hh["finger_base"], hh["finger_tip"], hh["thumb_rotate"], hh["thumb_bend"]],
                    "index_force_N": d["result"]["hands"][s].get("index_contact_force_N"),
                    "object_rise_while_closing_cm": round(rise_close, 2), "object_tilt_while_closing_deg": round(tilt_close, 1),
                    "max_rise_before_next_grasp_cm": round(best, 1), "held": bool(best >= 5.0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    a = ap.parse_args()
    rows = []
    for k in (1, 2, 3):
        rows += grasps(ROOT / f"source_interaction_dataset/{a.run}/attempt_{k}", "cup_a", f"source/s{k}/attempt_{k}")
    for obj in ("cup_b", "cyl_c"):
        for st in ("t1", "t2"):
            for m in METHODS:
                for k in (1, 2):
                    ad = ROOT / f"per_attempt_transition_logs/{a.run}/{obj}/{st}/{m}/attempt_{k}"
                    if (ad / "summary.json").exists():
                        rows += grasps(ad, obj, f"{st}/{m}/attempt_{k}")
    out = ROOT / f"evaluation_results/{a.run}"
    (out / "grasp_check.json").write_text(json.dumps(rows, indent=1))

    def rng(v):
        v = [x for x in v if x is not None]
        return f"{min(v):.1f}-{max(v):.1f}" if v else "-"

    L = ["# Per-grasp check (privileged trace)", "",
         "A grasp = a decision whose active hand goes from open to closing; held = the object reached 5 cm before the next grasp.",
         "Cup A's grasp = wrist at (-14.5, -5.0, +6.4) cm from the object base with closure 0.40/0.45 (all three source attempts).", "",
         "| object | grasps | held | at cup A's grasp (within 0.3 cm per axis) | held at cup A's grasp | index force at cup A's grasp, N | object rise / tilt while closing at cup A's grasp |",
         "|---|---|---|---|---|---|---|"]
    for obj in ("cup_a", "cup_b", "cyl_c"):
        R = [r for r in rows if r["object"] == obj]
        N = [r for r in R if r["at_cup_a_grasp"] and r["closure_cmd"][1] <= 0.5]
        L.append(f"| {obj} | {len(R)} | {sum(r['held'] for r in R)} | {len(N)} | {sum(r['held'] for r in N)} | {rng([r['index_force_N'] for r in N])} | "
                 f"{rng([r['object_rise_while_closing_cm'] for r in N])} cm / {rng([r['object_tilt_while_closing_deg'] for r in N])} deg |")
    L += ["", "Cup B grasps that held:", "", "| condition | grasp | wrist offset from cup A's grasp, cm | closure command | index force N | object rise / tilt while closing | max rise cm |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["object"] == "cup_b" and r["held"]:
            L.append(f"| {r['condition']} | {r['grasp']} | {r['offset_from_cup_a_grasp_cm']} | {r['closure_cmd'][:2]} | {r['index_force_N']} | "
                     f"{r['object_rise_while_closing_cm']} cm / {r['object_tilt_while_closing_deg']} deg | {r['max_rise_before_next_grasp_cm']} |")
    (out / "grasp_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
