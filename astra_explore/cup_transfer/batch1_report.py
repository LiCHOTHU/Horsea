"""Report of the first diagnostic batch (system python with numpy): python -m cup_transfer.batch1_report [--root <dir>]

Reads experiments/cup_batch1_2026-10-13/{phase0_offline, phase0_live, phase1, phase2_3, repair*} and writes REPORT_DATA.md
(tables) and gates.json (machine-readable gate statuses: PASS / FAIL / INCONCLUSIVE / INFRASTRUCTURE_BLOCKED, each with its
evidence and the next permitted action). Interpretation lives in REPORT.md (written by hand from these tables).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cup_transfer.costs import label_calls

ROOT = Path("/home/licho/workspace/Horsea/experiments/cup_batch1_2026-10-13")
OFF_ORDER = ("none", "shift", "x_only", "y_only", "z_only", "rim")


def jl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def rj(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


# ---------------------------------------------------------------------------- Phase 0 live
def live_attempts(root):
    out = []
    for res in sorted((root / "phase0_live").glob("*/*/result.json")):
        cdir = res.parent
        if "interrupted" in cdir.name:
            continue
        a = cdir / "attempt_1"
        r = rj(res)
        out.append({"cond": f"{cdir.parent.name}/{cdir.name}", "dir": a, "result": r})
    return out


def decisions(a):
    return [rj(d / "decision.json") for d in sorted(a.glob("decision_*")) if (d / "decision.json").exists()]


def time_breakdown(a):
    tl = rj(a / "timeline.json", [])
    if not tl:
        return None
    cats = defaultdict(float)
    for e in tl:
        cats[e["what"]] += e["t1"] - e["t0"]
    t_start, t_end = tl[0]["t0"], max(e["t1"] for e in tl)
    covered = sum(e["t1"] - e["t0"] for e in tl)
    sim = sum(e.get("sim_s", 0.0) for e in tl if e["what"].startswith("execution"))
    plan = sum(e.get("plan_s", 0.0) for e in tl if e["what"].startswith("execution"))
    ex = cats.get("execution (sim + System 1 + observation)", 0.0)
    obs = sum((d.get("result") or {}).get("observation_s", 0.0) for d in decisions(a))
    return {"wall_s": round(t_end - t_start, 2), "reset_s": round(cats.get("reset", 0.0), 2), "astra_s": round(cats.get("astra call", 0.0), 2),
            "sim_s": round(sim, 2), "system1_plan_s": round(plan, 3), "observation_render_s": round(obs, 2),
            "executor_other_s": round(ex - sim - plan - obs, 2), "obs_files_prompt_s": round(cats.get("observation files + prompt", 0.0), 2),
            "unaccounted_s": round((t_end - t_start) - covered, 2)}


INTENDED = {"approach": "wrist reaches Astra's target (< 1 cm) without moving the object (< 0.5 cm)",
            "reposition": "wrist reaches Astra's target (< 1 cm) without moving the object (< 0.5 cm)",
            "close": "fingers stop on the object (measured closure below the command) or index contact > 1 N; object not pushed away (< 2 cm) or tipped (< 20 deg)",
            "lift": "object rises with the hand (object rise >= 50 % of the wrist rise)",
            "hold": "object stays >= 5 cm up", "release": "object released", "transport": "object carried", "other": "-"}


def measured_effect(d, prev_rise):
    c, r = d.get("decision_content") or {}, d.get("result") or {}
    mode, h = c.get("interaction_mode", "other"), c.get("active_hand")
    if not r or h not in ("left", "right"):
        return mode, None, "no executed hand command"
    hr, ob = r["hands"][h], r.get("object", {})
    rise = ob.get("rise_above_start_cm", 0.0)
    if mode in ("approach", "reposition"):
        ok = (hr.get("position_error_cm") or 0) < 1.0 and ob.get("moved_cm", 0) < 0.5
        return mode, ok, f"wrist error {hr.get('position_error_cm')} cm, object moved {ob.get('moved_cm')} cm"
    if mode == "close":
        cm, cc = hr.get("closure_measured", {}), hr.get("closure_commanded", {})
        blocked = cm.get("finger_base", 0) < 0.9 * cc.get("finger_base", 0) or cm.get("finger_tip", 0) < 0.9 * cc.get("finger_tip", 0)
        tilt = float(np.degrees(np.arccos(np.clip(ob.get("up_axis_z", 1.0), -1, 1))))
        ok = (blocked or hr.get("index_contact_force_N", 0) > 1.0) and ob.get("moved_cm", 0) < 2.0 and tilt < 20
        return mode, ok, f"closure measured {cm} vs commanded {cc}, index {hr.get('index_contact_force_N')} N, object moved {ob.get('moved_cm')} cm, tilt {tilt:.0f} deg"
    if mode == "lift":
        wr = hr.get("moved_cm", 0)
        ok = (rise - prev_rise) >= 0.5 * wr if wr > 1 else None
        return mode, ok, f"wrist moved {wr} cm, object rise {prev_rise:.1f} -> {rise:.1f} cm"
    if mode == "hold":
        return mode, rise >= 5, f"object rise {rise} cm"
    return mode, None, f"object rise {rise} cm"


def failure_timeline(a):
    rows, prev_rise, first = [], 0.0, None
    for d in decisions(a):
        if not d.get("executed"):
            continue
        mode, ok, txt = measured_effect(d, prev_rise)
        rows.append({"decision": d["decision"], "step": d["steps_before"], "mode": mode, "intended": INTENDED.get(mode, "-"), "measured": txt, "as_intended": ok})
        if ok is False and first is None:
            first = rows[-1]
        prev_rise = (d.get("result") or {}).get("object", {}).get("rise_above_start_cm", prev_rise)
    return rows, first


def s1_events(a):
    tr = jl(a / "transitions.jsonl")
    cand = jl(a / "candidates.jsonl")
    ev = Counter()
    fb = Counter()
    for r in tr:
        p = r["plan"]
        ev["queried"] += 1
        if p.get("fallback") is None:
            ev["supported"] += 1
        else:
            fb[p["fallback"]] += 1
        if any(abs(x) > 1e-9 for x in r["adjustment"]):
            ev["changed"] += 1
    uniq, uniq_cl, n_sup, rej = [], [], [], Counter()
    for c in cand:
        rows = c.get("candidates_trace") or []
        if not rows:
            continue
        uniq.append(len({tuple(np.round(np.asarray(x["u_app"])[0], 3)) for x in rows}))
        uniq_cl.append(len({tuple(np.round(np.asarray(x["u_app"])[0][3:], 3)) for x in rows}))
        n_sup.append(sum(1 for x in rows if x.get("rejected") is None))
        for x in rows:
            if x.get("rejected"):
                rej[x["rejected"].split(" (")[0]] += 1
    # lineage: sent vs requested vs achieved (translation of the active hand)
    gap = [np.linalg.norm(np.asarray(r["lineage"]["achieved_wrist_cm"]) - np.asarray(r["applied_command_after_limits"][:3])) for r in tr if "lineage" in r]
    return {"segments": dict(ev), "fallbacks": dict(fb), "candidate_tables": len(n_sup),
            "supported_candidates_per_table": (min(n_sup), float(np.median(n_sup)), max(n_sup)) if n_sup else None,
            "unique_applied_first_commands": (min(uniq), float(np.median(uniq)), max(uniq)) if uniq else None,
            "unique_applied_first_closures": (min(uniq_cl), float(np.median(uniq_cl)), max(uniq_cl)) if uniq_cl else None,
            "rejections": dict(rej),
            "achieved_minus_applied_translation_cm": (round(float(np.median(gap)), 3), round(float(np.percentile(gap, 90)), 3)) if gap else None}


# ---------------------------------------------------------------------------- Phase 3
def branch_table(st):
    """Per candidate: predicted cost (if supported), realized cost, task effect; nominal = candidate 0."""
    rows = []
    for c in st["candidates"]:
        br = c["branches"]
        cont = [b["continuation"] for b in br if "continuation" in b]
        rows.append({"i": c["i"], "supported": c.get("rejected") is None, "rejected": c.get("rejected"), "J_hat": c.get("cost"),
                     "J_real": float(np.mean([b["J_real"] for b in br])), "J_real_sd": float(np.std([b["J_real"] for b in br])),
                     "held": [x["held"] for x in cont], "max_rise": [x["max_rise_cm"] for x in cont],
                     "eff": br[0]["effect"], "dlt_first": c["dlt"][0], "u_app_first": c["u_app"][0]})
    return rows


def classify(st, rows):
    nom = next(r for r in rows if r["i"] == 0)
    nom_held = float(np.mean(nom["held"])) if nom["held"] else None
    out = {"state": st["state"], "mode": st["mode"], "near": st["near"], "planner": st["plan"].get("fallback") or f"selected candidate {st['plan'].get('candidate')}",
           "nominal_held_rate": nom_held, "nominal_runs": len(nom["held"]), "nominal_J_real_sd": round(nom["J_real_sd"], 5),
           "unique_first_applied": st.get("unique_first_segment_applied_commands"), "unique_first_closures": st.get("unique_first_segment_applied_closures")}
    better = [r for r in rows if r["i"] != 0 and r["held"] and all(r["held"]) and (nom_held is not None and nom_held < 0.5)]
    out["better_task_effect"] = [r["i"] for r in better]
    if nom_held is None:
        out["category"] = "no continuation for the nominal (approach state without task outcome)"
        return out
    if nom_held >= 0.5:
        out["category"] = "nominal already succeeds here (no correction needed at this state)"
        return out
    if not better:
        out["category"] = "no candidate produces a better task effect: candidate coverage, action capability or horizon"
        return out
    pref_real = [r for r in better if r["J_real"] < nom["J_real"]]
    pref_hat = [r for r in better if r["supported"] and r["J_hat"] is not None and nom["J_hat"] is not None and r["J_hat"] < nom["J_hat"]]
    out.update(objective_prefers=[r["i"] for r in pref_real], model_prefers=[r["i"] for r in pref_hat],
               better_unsupported=[r["i"] for r in better if not r["supported"]])
    if not pref_real:
        out["category"] = "a better effect exists, but the realized-state objective rejects it: objective misaligned"
    elif not pref_hat:
        out["category"] = "the realized-state objective prefers it, but the model does not (or has no support): model / data coverage"
    else:
        out["category"] = "the model prefers a better candidate: check selection / execution"
    return out


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    L, gates = ["# Batch 1 data tables (generated)", ""], {}

    # ---- Phase 0 offline
    off = rj(root / "phase0_offline" / "costs.json", [])
    if off:
        W = sum(r["wall_s"] for r in off)
        L += ["## Phase 0, offline (29 attempts of the cup-transfer run)", "",
              f"Wall {W / 60:.1f} min: Astra {100 * sum(r['astra_s'] for r in off) / W:.0f} %, executor {100 * sum(r['execution_s'] for r in off) / W:.0f} %, "
              f"other {100 * sum(r['other_s'] for r in off) / W:.0f} %. Details: phase0_offline/costs.md.", ""]

    # ---- Phase 0 live
    live = live_attempts(root)
    if live:
        L += ["## Phase 0, live attempts (instrumented)", "",
              "| condition | outcome | steps | calls | wall s | Astra s | sim s | System 1 plan s | observation render s | other executor s | files + prompt s | reset s | unaccounted s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        aligned = True
        for x in live:
            s = (x["result"] or {}).get("summary", {})
            tb = time_breakdown(x["dir"])
            if tb is None:
                aligned = False
                continue
            if abs(tb["unaccounted_s"]) > 0.1 * tb["wall_s"]:
                aligned = False
            L.append(f"| {x['cond']} | {s.get('outcome', x['result'].get('status'))} | {s.get('steps_used')} | {s.get('model_calls')} | {tb['wall_s']} | {tb['astra_s']} | "
                     f"{tb['sim_s']} | {tb['system1_plan_s']} | {tb['observation_render_s']} | {tb['executor_other_s']} | {tb['obs_files_prompt_s']} | {tb['reset_s']} | {tb['unaccounted_s']} |")
        L += ["", "### Calls by purpose (live)", "", "| condition | " + " | ".join(("initial approach", "approach correction", "closure", "grasp verification", "recovery", "other")) + " |",
              "|---|---|---|---|---|---|---|"]
        for x in live:
            labs = Counter(label_calls(decisions(x["dir"])))
            L.append(f"| {x['cond']} | " + " | ".join(str(labs.get(k, 0)) for k in ("initial approach", "approach correction", "closure", "grasp verification", "recovery", "other")) + " |")
        L += ["", "### First divergence from the intended physical effect (measured)", ""]
        for x in live:
            rows, first = failure_timeline(x["dir"])
            L.append(f"- **{x['cond']}**: " + ("no divergence" if first is None else
                     f"decision {first['decision']} (step {first['step']}, {first['mode']}): intended {first['intended']}; measured {first['measured']}"))
        L += ["", "### System 1 in the live attempts", "", "| condition | segments queried / supported / changed | fallbacks | candidate tables | supported candidates per table (min, median, max) | distinct first applied commands | distinct first applied closures | rejection reasons | achieved - applied translation cm (median, p90) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in live:
            e = s1_events(x["dir"])
            sg = e["segments"]
            L.append(f"| {x['cond']} | {sg.get('queried', 0)} / {sg.get('supported', 0)} / {sg.get('changed', 0)} | {e['fallbacks']} | {e['candidate_tables']} | "
                     f"{e['supported_candidates_per_table']} | {e['unique_applied_first_commands']} | {e['unique_applied_first_closures']} | {e['rejections']} | "
                     f"{e['achieved_minus_applied_translation_cm']} |")
        blocked = [x for x in live if (x["result"] or {}).get("status") != "complete"]
        gates["G0_instrumentation"] = {"status": "INFRASTRUCTURE_BLOCKED" if blocked else ("PASS" if aligned else "FAIL"),
                                       "evidence": f"{len(live)} live attempts; timelines cover the attempt wall time within 10 %: {aligned}; "
                                                   "predictions are made before execution and scored before the record is added (execute_decision order)",
                                       "next": "interpret timing and System 1 events" if aligned and not blocked else "fix logging / rerun blocked attempts"}

    # ---- Phase 1
    p1 = root / "phase1"
    if p1.exists():
        L += ["", "## Phase 1: frames, action channels, known-good grasp", ""]
        ok_frames, ok_channels, hold_a = True, True, None
        for obj_dir in sorted(p1.glob("*")):
            fc = rj(obj_dir / "frame_and_channels.json")
            if fc:
                f = fc["object_frame"]
                L.append(f"- **{obj_dir.name}** frame: bbox {f['bbox_size_cm']} cm vs spec D x H {f['spec_diameter_height_cm']} cm; frame minus bbox base centre "
                         f"{f['frame_minus_bbox_base_centre_cm']} cm; up {f['up_axis']}")
                if max(abs(v) for v in f["frame_minus_bbox_base_centre_cm"]) > 0.2:
                    ok_frames = False
                for t in fc["channels"]["translation"]:
                    if t["error_cm"] > 0.3:
                        ok_channels = False
                L.append("  - translation: " + "; ".join(f"{t['command']} -> {t['achieved_cm']} (err {t['error_cm']})" for t in fc["channels"]["translation"]))
                L.append("  - rotation: " + "; ".join(f"{t['command']} -> {t['achieved_deg']} deg (err {t['error_deg']})" for t in fc["channels"]["rotation"]))
                L.append("  - closure: " + "; ".join(f"{t['group']} {t['command']} -> {t['measured_fractions']} in {t['steps_to_95pct']} steps, groups moved {t['joint_groups_moved']}"
                                                    for t in fc["channels"]["closure"]))
                for t in fc["channels"]["closure"]:
                    if t["joint_groups_moved"] != [t["group"]]:
                        ok_channels = False
            rep = jl(obj_dir / "replay.jsonl")
            if rep:
                by = defaultdict(list)
                for r in rep:
                    by[r["variant"]].append(r)
                for v, rs in by.items():
                    n_ok = sum(r["success"] for r in rs)
                    L.append(f"- **{obj_dir.name}** replay of cup A's grasp, variant {v}: held {n_ok}/{len(rs)} (max rise median {np.median([r['max_rise_cm'] for r in rs]):.1f} cm; "
                             f"per start: " + ", ".join(f"{r['start']}:{'Y' if r['success'] else 'n'}" for r in rs) + ")")
                    if obj_dir.name == "cup_a" and v == "original":
                        hold_a = n_ok / len(rs)
                    # end of closing: thumb / index tip position relative to the object, squeeze signature
                    H = rs[0].get("object_height_cm")
                    ends = [[g for g in r["geometry"] if g["mode"] == "close"][-1] for r in rs if any(g["mode"] == "close" for g in r["geometry"])]
                    if ends:
                        q = lambda f: f"{np.median([f(e) for e in ends]):.2f} [{min(f(e) for e in ends):.2f}, {max(f(e) for e in ends):.2f}]"
                        L.append(f"  - end of closing (median [min, max] over {len(ends)} runs): thumb tip gap to the outer wall {q(lambda e: e['tips']['R_thumb_tip']['surface_gap_cm'])} cm "
                                 f"at height {q(lambda e: e['tips']['R_thumb_tip']['height_cm'])} cm; index tip gap {q(lambda e: e['tips']['R_index_tip']['surface_gap_cm'])} cm at "
                                 f"{q(lambda e: e['tips']['R_index_tip']['height_cm'])} cm; index contact {q(lambda e: e['contact_N'])} N; object rise {q(lambda e: e['object_rise_cm'])} cm, "
                                 f"tilt {q(lambda e: e['object_tilt_deg'])} deg while closing")
        cam = sorted((root / "phase0_live").glob("*/camera_check.json"))
        cam_ok = None
        for c in cam:
            d = rj(c)
            pt = d["main_camera"]["points"].get("object_side_facing_main_camera", {})
            errs = [pt.get("depth_minus_forward_cm")]
            L.append(f"- camera check {c.parent.name}: at the visible object surface point, main-camera depth minus projected forward distance = {errs[0]} cm "
                     f"(other points are occluded or between surfaces: {[(k, p.get('depth_minus_forward_cm')) for k, p in d['main_camera']['points'].items()]})")
            if errs:
                cam_ok = (cam_ok is not False) and all(abs(e) < 1.5 for e in errs if e is not None)
        gates["G1_geometry_interface"] = {"status": "PASS" if (ok_frames and ok_channels and (hold_a or 0) >= 0.9) else "FAIL",
                                          "evidence": f"frames ok {ok_frames}; channels ok {ok_channels}; cup A known-good grasp held rate {hold_a}; camera depth check ok {cam_ok}",
                                          "next": "System 1 diagnosis" if (ok_frames and ok_channels) else "repair the interface first"}

    # ---- Phases 2-3
    st_files = sorted((root / "phase2_3").glob("[ACL][12].json"))
    if st_files:
        L += ["", "## Phases 2-3: the current candidate space from six cup B decision states", "",
              "| state | mode | near | planner | nominal held (runs) | nominal J_real sd | distinct first applied commands / closures | candidates held (all runs) | objective prefers | model prefers | better but unsupported | category |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        cats = {}
        for f in st_files:
            st = rj(f)
            rows = branch_table(st)
            c = classify(st, rows)
            cats[st["state"]] = c
            L.append(f"| {st['state']} | {st['mode']} | {st['near']} | {c['planner']} | {c['nominal_held_rate']} ({c['nominal_runs']}) | {c['nominal_J_real_sd']} | "
                     f"{c['unique_first_applied']} / {c['unique_first_closures']} | {c['better_task_effect']} | {c.get('objective_prefers', '-')} | {c.get('model_prefers', '-')} | "
                     f"{c.get('better_unsupported', '-')} | {c['category']} |")
            L += ["", f"<details><summary>{st['state']}: every candidate</summary>", "",
                  "| i | supported | J_hat | J_real (sd) | held | max rise cm | first-segment deviation (normalised) | contact N | object xy cm | tilt deg | tips gap cm |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                e = r["eff"]
                L.append(f"| {r['i']} | {r['supported']} | {r['J_hat']} | {r['J_real']:.4f} ({r['J_real_sd']:.4f}) | {r['held']} | {r['max_rise']} | "
                         f"{np.round(r['dlt_first'], 2).tolist()} | {e['contact_N']} | {e['object_xy_moved_cm']} | {e['object_tilt_deg']} | {e['tips_radial_gap_cm']} |")
            L += ["", "</details>", ""]
        gates["G2_system1_explained"] = {"status": "PASS" if len(cats) == 6 else "INCONCLUSIVE", "evidence": {k: v["category"] for k, v in cats.items()},
                                         "next": "repair the earliest failing category on these states"}
    # ---- persistent offsets (repair diagnostic from the same restored states)
    per = jl(root / "phase2_3" / "persist.jsonl")
    if per:
        L += ["", "## Persistent wrist-target offsets from the restored cup B states (rest of the decision + fixed continuation)", "",
              "| state | " + " | ".join(OFF_ORDER) + " |", "|---|" + "---|" * len(OFF_ORDER)]
        for stn in ("A2", "C1", "C2"):
            cells = []
            for off in OFF_ORDER:
                rs = [r for r in per if r["state"] == stn and r["offset"] == off]
                cells.append(f"{sum(r['held'] for r in rs)}/{len(rs)}" if rs else "-")
            L.append(f"| {stn} | " + " | ".join(cells) + " |")
        shift_ok = sum(r["held"] for r in per if r["offset"] == "shift" and r["state"] in ("A2", "C1"))
        shift_n = sum(1 for r in per if r["offset"] == "shift" and r["state"] in ("A2", "C1"))
        rep = jl(root / "phase1" / "cup_b" / "replay.jsonl")
        sh = [r for r in rep if r["variant"] == "shift"]
        gates["G3_minimal_repair"] = {"status": "PASS" if sh and all(r["success"] for r in sh) and shift_ok == shift_n else "FAIL",
                                      "repair": "geometry of the transferred grasp: wrist target offset (-0.5, +0.5, -0.3) cm (from the cup B grasps that held in the transfer run; development data, not a rule)",
                                      "evidence": f"replays from 10 development starts: {sum(r['success'] for r in sh)}/{len(sh)} held (original 0/20); from Astra's own approach / closing-start states {shift_ok}/{shift_n}; "
                                                  "from mid-closing 0/2; single axes and the rim-anchored height 0/2 each",
                                      "next": "derive the correction inside the system (contact geometry visible to Astra or a contact-geometry goal + varied data for System 1), then Phase 5"}
    (root / "REPORT_DATA.md").write_text("\n".join(L) + "\n")
    (root / "gates.json").write_text(json.dumps(gates, indent=1))
    print("\n".join(L[:80]))
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
