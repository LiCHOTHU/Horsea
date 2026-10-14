"""Control diagnostics (2026-10-14), analysis: success and failure categories per executor and condition, tracking and contact
measurements, C2's regrasps, and the user's decision gate.

    python -m cup_transfer.control_diag_offline summary
Outputs: experiments/cup_control_2026-10-14/{summary.json, REPORT_DATA.md}.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_transfer import control_diag as CD  # noqa: E402

OUT = CD.OUT
EXECS = ("C0", "C1", "C2", "C2b")


def load():
    rows = []
    for f in sorted((OUT / "episodes").glob("*.jsonl")):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    return rows


def kind(cond):
    return CD.KIND[cond.split("_", 1)[1]]


def short(cat):
    return cat.split(" (")[0]


def summary():
    rows = load()
    res = {"written": time.strftime("%Y-%m-%d %H:%M"), "episodes": len(rows), "by_executor": {}, "by_condition": {}, "categories": {},
           "tracking": {}, "c2": {}, "paired": {}}
    for ex in EXECS:
        rs = [r for r in rows if r["executor"] == ex]
        res["by_executor"][ex] = {"episodes": len(rs), "success": sum(r["outcome"]["task_success"] for r in rs),
                                  "nominal": f"{sum(r['outcome']['task_success'] for r in rs if kind(r['condition']) == 'nominal grasp')}/"
                                             f"{sum(1 for r in rs if kind(r['condition']) == 'nominal grasp')}",
                                  "perturbed": f"{sum(r['outcome']['task_success'] for r in rs if kind(r['condition']) != 'nominal grasp')}/"
                                               f"{sum(1 for r in rs if kind(r['condition']) != 'nominal grasp')}",
                                  "safety_violations": sum(1 for r in rs if r["outcome"]["safety_violation"]),
                                  "categories": dict(Counter(short(r["diagnosis"]["category"]) for r in rs)),
                                  "object_behaviour": dict(Counter(r["diagnosis"]["object_behaviour"] for r in rs))}
    for c in CD.CONDITIONS:
        res["by_condition"][c] = {"object": CD.CONDITIONS[c][0], "kind": kind(c), "error": CD.CONDITIONS[c][1]}
        for ex in EXECS:
            rs = [r for r in rows if r["condition"] == c and r["executor"] == ex]
            res["by_condition"][c][ex] = {"success": f"{sum(r['outcome']['task_success'] for r in rs)}/{len(rs)}",
                                          "categories": [short(r["diagnosis"]["category"]) for r in rs],
                                          "behaviour": [r["diagnosis"]["object_behaviour"] for r in rs],
                                          "max_rise_cm": [r["outcome"]["max_rise_cm"] for r in rs],
                                          "regrasps": [r["regrasps"] for r in rs]}
    # tracking: how far the wrist was from the commanded reference at the end of the second approach (free motion) and of
    # closing (in contact), and closure shortfall at the end of closing
    for ex in EXECS:
        rs = [r for r in rows if r["executor"] == ex]
        a1 = [r["diagnosis"]["tracking_error_end_cm"].get("approach1") for r in rs]
        c2_ = [r["diagnosis"]["tracking_error_end_cm"].get("close2") for r in rs]
        sh = [np.array(r["diagnosis"]["end_of_close"]["closure_measured"]) for r in rs if r["diagnosis"].get("end_of_close")]
        res["tracking"][ex] = {"approach_end_cm_median": round(float(np.median([x for x in a1 if x is not None])), 3) if a1 else None,
                               "approach_end_cm_max": round(float(np.max([x for x in a1 if x is not None])), 3) if a1 else None,
                               "close_end_cm_median": round(float(np.median([x for x in c2_ if x is not None])), 3) if c2_ else None,
                               "close_end_cm_max": round(float(np.max([x for x in c2_ if x is not None])), 3) if c2_ else None,
                               "closure_measured_end_of_close_mean": np.mean(sh, axis=0).round(3).tolist() if sh else None}
    c2 = [r for r in rows if r["executor"] == "C2b"]                 # the recalibrated loop (C2's own numbers are in by_executor)
    res["c2"] = {"episodes_with_regrasp": sum(1 for r in c2 if r["regrasps"]), "regrasps": sum(r["regrasps"] for r in c2),
                 "rescued": sum(1 for r in c2 if r["regrasps"] and r["outcome"]["task_success"]),
                 "rules_fired": dict(Counter(rule.split(":")[0] for r in c2 for e in r["events"] if e["event"].startswith("regrasp") for rule in e["rules"])),
                 "statuses": dict(Counter(r["status"] for r in c2))}
    idx = {(r["condition"], r["rep"], r["executor"]): r["outcome"]["task_success"] for r in rows}
    for a, b in (("C1", "C0"), ("C2", "C0"), ("C2b", "C0"), ("C2b", "C1")):
        keys = {(c, rp) for (c, rp, e) in idx if e == a} & {(c, rp) for (c, rp, e) in idx if e == b}
        res["paired"][f"{a}-{b}"] = {"a_better": sum(1 for k in keys if idx[k + (a,)] and not idx[k + (b,)]),
                                     "b_better": sum(1 for k in keys if idx[k + (b,)] and not idx[k + (a,)]), "pairs": len(keys)}
    # the gate (the user's rules), read mechanically from the C0 failures and what C1 / C2 fixed
    c0_fail = [r for r in rows if r["executor"] == "C0" and not r["outcome"]["task_success"]]
    fixed = {ex: sum(1 for r in c0_fail if idx.get((r["condition"], r["rep"], ex))) for ex in ("C1", "C2", "C2b")}
    cats = Counter(short(r["diagnosis"]["category"]) for r in c0_fail)
    res["gate"] = {"c0_failures": len(c0_fail), "c0_failure_categories": dict(cats), "fixed_by_C1": fixed["C1"], "fixed_by_C2": fixed["C2"],
                   "fixed_by_C2b": fixed["C2b"],
                   "nominal_success": {ex: res["by_executor"][ex]["nominal"] for ex in EXECS}}
    n = max(len(c0_fail), 1)
    if fixed["C1"] / n > 0.5:
        g = "C1 resolves most failures: keep C1; learned contact control is not justified yet"
    elif max(fixed["C2"], fixed["C2b"]) / n > 0.5:
        g = "C2 resolves most failures: freeze C2 as System 0 and go to the memory experiment"
    elif cats.get("grasp unsuitable", 0) / n > 0.5:
        g = "the chosen grasp itself is unsuitable in most failures: improve proposal selection before motor-policy training"
    elif cats.get("contact control", 0) / n > 0.5:
        g = "commands reach the intended region but contact handling fails, and C2 does not fix it: a learned System 0 pilot is justified"
    elif cats.get("motion execution", 0) / n > 0.5:
        g = "motion execution fails: fix tracking / the control interface"
    else:
        g = "mixed: no single gate condition holds for a majority of failures"
    res["gate"]["decision"] = g
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    L = [f"# Control diagnostics: data tables (generated {res['written']}; {len(rows)} episodes)", "",
         "## Success per executor", "", "| executor | all | nominal grasps | perturbed (pose errors, imperfect contacts) | safety violations |", "|---|---|---|---|---|"]
    for ex in EXECS:
        v = res["by_executor"][ex]
        L.append(f"| {ex} | {v['success']}/{v['episodes']} | {v['nominal']} | {v['perturbed']} | {v['safety_violations']} |")
    L += ["", "## Per condition (2 repeats each): successes, diagnosed category of every failure", "",
          "| condition | object | kind | C0 | C1 | C2 (regrasps) | C2b (regrasps) |", "|---|---|---|---|---|---|---|"]
    for c, v in res["by_condition"].items():
        cell = lambda ex: v[ex]["success"] + ("" if not [x for x in v[ex]["categories"] if x != "success"] else
                                              " - " + ", ".join(x for x in v[ex]["categories"] if x != "success"))
        L.append(f"| {c} | {v['object']} | {v['kind']} | {cell('C0')} | {cell('C1')} | {cell('C2')} ({v['C2']['regrasps']}) | {cell('C2b')} ({v['C2b']['regrasps']}) |")
    L += ["", "## Failure categories and object behaviour", ""]
    for ex in EXECS:
        L.append(f"- {ex}: {res['by_executor'][ex]['categories']}; object: {res['by_executor'][ex]['object_behaviour']}")
    L += ["", "## Tracking (wrist distance to the commanded reference at decision ends) and closures", "",
          "| executor | end of approach: median / max cm | end of closing (in contact): median / max cm | measured closures at the end of closing (mean) |",
          "|---|---|---|---|"]
    for ex in EXECS:
        t = res["tracking"][ex]
        L.append(f"| {ex} | {t['approach_end_cm_median']} / {t['approach_end_cm_max']} | {t['close_end_cm_median']} / {t['close_end_cm_max']} | {t['closure_measured_end_of_close_mean']} |")
    L += ["", f"## C2b's loop (the recalibrated trigger): {json.dumps(res['c2'])}", "", f"## Paired (same condition and start): {json.dumps(res['paired'])}", "",
          f"## Gate: {json.dumps(res['gate'])}"]
    (OUT / "REPORT_DATA.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: res[k] for k in ("by_executor", "tracking", "c2", "paired", "gate")}, indent=1))


if __name__ == "__main__":
    summary()
