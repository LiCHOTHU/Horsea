"""Analysis of the cup-transfer experiment (system python).

    python -m cup_transfer.analyze_cup [--run run|smoke]

Reads experiments/cup_transfer_2026-10-12/{source_interaction_dataset,per_attempt_transition_logs}/<run> and writes
evaluation_results/<run>/{results.json, results.md, S_b.png, cases.md}. Failed attempts stay in every efficiency number as
budget-exhausted observations (censored at the steps they used); nothing is averaged over successful episodes only.
Events come from the privileged per-step trace (evaluation only):
  first contact   first step with index contact > 1 N on either hand, or the object moved > 3 mm or rose > 3 mm
  stable grasp    first step of 10 consecutive steps with the object >= 1 cm above its rest height
  closures        decisions whose active hand goes from open (< 0.3) to closed (>= 0.5); regrasps = closures after the first
  empty lift      a 'lift' decision during which the wrist rose >= 3 cm but the object rose < 1 cm
  slip / drop     the object was >= 2 cm up, then fell below 0.5 cm while the active hand stayed commanded closed
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/licho/workspace/Horsea/experiments/cup_transfer_2026-10-12")
METHODS = ["astra_history", "nearest_neighbor", "empty_model", "frozen_model", "adaptive_model"]
NAMES = {"astra_history": "Astra-history", "nearest_neighbor": "Nearest-neighbor", "empty_model": "Empty-model",
         "frozen_model": "Frozen-model", "adaptive_model": "Adaptive-model (ours)"}


def jl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def decisions(adir):
    out = []
    for d in sorted(adir.glob("decision_*")):
        if (d / "decision.json").exists():
            out.append(json.loads((d / "decision.json").read_text()))
    return out


def events(adir):
    tr = jl(adir / "eval_trace.jsonl")
    dec = decisions(adir)
    summ = json.loads((adir / "summary.json").read_text()) if (adir / "summary.json").exists() else {}
    if not tr:
        return {"outcome": summ.get("outcome"), "steps": summ.get("steps_used", 0)}
    z0, xy0 = tr[0]["obj"][2], np.array(tr[0]["obj"][:2])
    rise = np.array([r["obj"][2] - z0 for r in tr])
    moved = np.array([np.linalg.norm(np.array(r["obj"][:2]) - xy0) for r in tr])
    contact = np.array([max(r["contact"].values()) for r in tr])
    first_contact = next((i + 1 for i in range(len(tr)) if contact[i] > 1.0 or moved[i] > 0.003 or rise[i] > 0.003), None)
    stable = next((i + 1 for i in range(len(tr) - 9) if np.all(rise[i:i + 10] >= 0.01)), None)
    closures, prev, empty_lifts, slips = 0, {"left": 0.0, "right": 0.0}, 0, 0
    for d in dec:
        c = d.get("decision_content")
        if not c or c.get("stop"):
            continue
        h = c.get("active_hand")
        for s in ("left", "right"):
            m = max(c[s]["finger_base"], c[s]["finger_tip"])
            if s == h and m >= 0.5 and prev[s] < 0.3:
                closures += 1
            prev[s] = m
        if c.get("interaction_mode") == "lift" and d.get("result"):
            r = d["result"]
            wr = r["hands"].get(h, {}).get("moved_cm", 0) if h in ("left", "right") else 0
            if wr >= 3 and r.get("object", {}).get("rise_above_start_cm", 0) < 1:
                empty_lifts += 1
    up = False
    for v in rise:
        if v >= 0.02:
            up = True
        elif up and v < 0.005:
            slips += 1
            up = False
    return {"outcome": summ.get("outcome"), "success": summ.get("outcome") == "success", "steps": summ.get("steps_used", len(tr)),
            "first_contact_step": first_contact, "stable_grasp_step": stable, "closures": closures, "regrasps": max(0, closures - 1),
            "empty_lifts": empty_lifts, "slips_drops": slips, "model_calls": summ.get("model_calls"), "wall_s": summ.get("attempt_wall_time_s"),
            "max_rise_cm": round(100 * float(rise.max()), 2)}


def mechanism(adir):
    rows = jl(adir / "transitions.jsonl")
    if not rows:
        return {}
    plans = [r["plan"] for r in rows]
    fb = [p for p in plans if p.get("fallback")]
    adj = np.array([r["adjustment"] for r in rows])
    perr = [r["prediction_rmse_before_update"] for r in rows if r.get("prediction_rmse_before_update") is not None]
    src = tgt = 0
    for p in plans:
        for o in p.get("neighbor_objects", []) or []:
            if o == "cup_a":
                src += 1
            else:
                tgt += 1
        if p.get("record_object"):
            src += p["record_object"] == "cup_a"
            tgt += p["record_object"] != "cup_a"
    early = rows[:20]
    early_adj = [r for r in early if np.any(np.abs(r["adjustment"]) > 1e-9)]
    early_src = sum(1 for r in early_adj if any(o == "cup_a" for o in (r["plan"].get("neighbor_objects") or [r["plan"].get("record_object")])))
    obj_err = []
    for r in rows:
        if r["plan"].get("predicted_ds") is not None:
            obj_err.append(float(np.sqrt(np.mean((np.array(r["plan"]["predicted_ds"])[[10, 11, 12, 13]] - np.array(r["ds"])[[10, 11, 12, 13]]) ** 2))))
    return {"segments": len(rows), "fallback_fraction": round(len(fb) / len(rows), 3),
            "fallback_reasons": {k: sum(1 for p in fb if p["fallback"] == k) for k in sorted({p["fallback"] for p in fb})},
            "adjusted_fraction": round(float(np.mean(np.any(np.abs(adj) > 1e-9, axis=1))), 3),
            "mean_translation_change_cm": round(float(np.mean(np.linalg.norm(adj[:, :3], axis=1))), 3),
            "mean_closure_change": round(float(np.mean(np.abs(adj[:, 3:]))), 4),
            "prediction_rmse_before_update": round(float(np.mean(perr)), 4) if perr else None,
            "object_motion_prediction_rmse": round(float(np.mean(obj_err)), 4) if obj_err else None,
            "retrieved_source_records": src, "retrieved_target_records": tgt,
            "early_segments_adjusted": len(early_adj), "early_adjusted_using_source": early_src}


def wrist_error(adir):
    e = []
    for d in decisions(adir):
        for h in ("left", "right"):
            v = (d.get("result") or {}).get("hands", {}).get(h, {}).get("position_error_cm")
            if v is not None:
                e.append(v)
    return round(float(np.median(e)), 2) if e else None


def condition(cdir):
    res = json.loads((cdir / "result.json").read_text()) if (cdir / "result.json").exists() else None
    atts = []
    for k in (1, 2):
        ad = cdir / f"attempt_{k}"
        if (ad / "summary.json").exists():
            atts.append(dict(events(ad), mechanism=mechanism(ad), wrist_error_median_cm=wrist_error(ad)))
    if not atts:
        return None
    cum, first = 0, None
    for i, a in enumerate(atts):
        if a.get("success"):
            first = i + 1
            cum += a["steps"]
            break
        cum += a["steps"]
    return {"complete": bool(res and res.get("status") == "complete"), "attempts": atts, "first_attempt_success": bool(atts[0].get("success")),
            "success_within_two": first is not None, "first_success_attempt": first, "cumulative_steps_to_success_or_budget": cum,
            "censored": first is None, "astra_calls": sum(a.get("model_calls") or 0 for a in atts), "wall_s": round(sum(a.get("wall_s") or 0 for a in atts), 1),
            "regrasps": sum(a.get("regrasps", 0) for a in atts), "slips_drops": sum(a.get("slips_drops", 0) for a in atts),
            "empty_lifts": sum(a.get("empty_lifts", 0) for a in atts), "resets": len(atts) - 1}


def S_of_b(rows, budgets):
    """P(success within b cumulative control steps); failures count as not-yet-successful at every b."""
    out = []
    for b in budgets:
        out.append(float(np.mean([1.0 if (r["success_within_two"] and r["cumulative_steps_to_success_or_budget"] <= b) else 0.0 for r in rows])) if rows else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    a = ap.parse_args()
    base = ROOT / "per_attempt_transition_logs" / a.run
    out = ROOT / "evaluation_results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    res = {"source": {}, "targets": {}}
    src = ROOT / "source_interaction_dataset" / a.run
    if src.exists():
        D = jl(src / "D_A.jsonl")
        modes = {}
        for r in D:
            m = r["interaction_context"]["mode"]
            modes[m] = modes.get(m, 0) + 1
        sa = [dict(events(src / f"attempt_{k}"), wrist_error_median_cm=wrist_error(src / f"attempt_{k}")) for k in (1, 2, 3) if (src / f"attempt_{k}" / "summary.json").exists()]
        U = np.array([r["u_applied"] for r in D]) if D else np.zeros((0, 7))
        res["source"] = {"records": len(D), "modes": modes, "attempts": sa,
                         "stable_grasp_ever": any(x.get("stable_grasp_step") for x in sa), "success_any": any(x.get("success") for x in sa),
                         "command_std": U.std(0).round(3).tolist() if len(U) else None,
                         "near_object_fraction": round(float(np.mean([r["interaction_context"]["near"] for r in D])), 3) if D else None}
        hp = ROOT / "method_configurations" / ("hyper.json" if a.run == "run" else "hyper_smoke.json")
        if hp.exists():
            res["source"]["model_selection"] = json.loads(hp.read_text())
    budgets = list(range(0, 801, 25))
    for obj_dir in sorted(base.glob("*")) if base.exists() else []:
        obj = obj_dir.name
        res["targets"][obj] = {}
        for m in METHODS:
            rows = []
            for sdir in sorted(obj_dir.glob("*")):
                c = condition(sdir / m)
                if c:
                    rows.append(dict(c, start=sdir.name))
            res["targets"][obj][m] = {"conditions": rows, "S_b": S_of_b(rows, budgets)}
    res["budgets"] = budgets
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    L = [f"# Cup-transfer results ({a.run})", ""]
    if res["source"]:
        s = res["source"]
        L += ["## Source data (cup A, Astra with the common executor)", "",
              f"records {s['records']}; interaction modes {s['modes']}; fingertips near the object in {s['near_object_fraction']} of records; "
              f"stable grasp established: {s['stable_grasp_ever']}; task success in any source attempt: {s['success_any']}; command std {s['command_std']}", "",
              "| attempt | outcome | steps | first contact | stable grasp | closures | empty lifts | slips | max rise cm | median wrist error cm |", "|---|---|---|---|---|---|---|---|---|---|"]
        for i, x in enumerate(s["attempts"], 1):
            L.append(f"| {i} | {x.get('outcome')} | {x.get('steps')} | {x.get('first_contact_step')} | {x.get('stable_grasp_step')} | {x.get('closures')} | "
                     f"{x.get('empty_lifts')} | {x.get('slips_drops')} | {x.get('max_rise_cm')} | {x.get('wrist_error_median_cm')} |")
        if "model_selection" in s:
            L += ["", f"Local model settings chosen on source data only (leave-one-attempt-out): {s['model_selection']['best']}; "
                      f"zero-change baseline NMSE {s['model_selection']['zero_change_nmse']}."]
        L.append("")
    for obj, per in res["targets"].items():
        L += [f"## Target: {obj}", "", "| method | start | first-attempt success | success within two | cumulative steps (censored = no success) | Astra calls | "
              "regrasps | slips | empty lifts | first contact (att. 1) | stable grasp (att. 1) | fallback fraction (att. 1) | adjusted segments (att. 1) | "
              "pred. RMSE before update (att. 1) | source / target records retrieved (att. 1) |", "|" + "---|" * 15]
        for m in METHODS:
            for r in per[m]["conditions"]:
                a1 = r["attempts"][0]
                mech = a1.get("mechanism", {})
                L.append(f"| {NAMES[m]} | {r['start']} | {int(r['first_attempt_success'])} | {int(r['success_within_two'])} | "
                         f"{r['cumulative_steps_to_success_or_budget']}{' (censored)' if r['censored'] else ''} | {r['astra_calls']} | {r['regrasps']} | "
                         f"{r['slips_drops']} | {r['empty_lifts']} | {a1.get('first_contact_step')} | {a1.get('stable_grasp_step')} | "
                         f"{mech.get('fallback_fraction')} | {mech.get('adjusted_fraction')} | {mech.get('prediction_rmse_before_update')} | "
                         f"{mech.get('retrieved_source_records')} / {mech.get('retrieved_target_records')} |")
        L.append("")
    (out / "results.md").write_text("\n".join(L) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        objs = list(res["targets"])
        if objs:
            fig, axs = plt.subplots(1, len(objs), figsize=(5.5 * len(objs), 3.8), squeeze=False)
            for ax, obj in zip(axs[0], objs):
                for m in METHODS:
                    sb = res["targets"][obj][m]["S_b"]
                    if sb and sb[0] is not None:
                        ax.step(budgets, sb, where="post", label=NAMES[m])
                ax.set_title(f"{obj}: S(b) = P(success within b control steps)")
                ax.set_xlabel("cumulative control steps (two attempts)")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "S_b.png", dpi=110)
    except Exception as ex:  # noqa: BLE001
        print("plot skipped:", ex)
    print("\n".join(L))


if __name__ == "__main__":
    main()
