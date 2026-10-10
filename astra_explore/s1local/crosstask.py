"""Cross-task scalability check of the local-execution-adaptation conclusion (system python).

    python -m s1local.crosstask --runs <run dir> [<run dir> ...] --out <dir>

For every task x operation x condition (matched block) it classifies what happened and pools the paired contrasts:
    ceiling      Astra's original program succeeded on attempt 1 (nothing for local search to fix)
    rescue       attempt 1 failed but B or C found a success within the nine attempts (local flexibility helped)
    noise        attempt 1 failed, B and C never succeeded, but A's unchanged repeats did (execution noise)
    none         nobody succeeded in nine attempts
Pooled contrasts (fresh-start success, attempts to first success) resample conditions within each task x operation stratum.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from s1local import analyze as A


def classify(row):
    a, b, c = row.get("A"), row.get("B"), row.get("C")
    if not a:
        return "init_failure"
    if a["first_success"] == 1:
        return "ceiling"
    if (b and b["first_success"]) or (c and c["first_success"]):
        return "rescue"
    if a["first_success"]:
        return "noise"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    strata, rows_md, per_task = {}, [], {}
    for run in a.runs:
        run = Path(run)
        man = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
        task = man.get("task") or man.get("environment", "").split(",")[0] or run.name
        attempts, rollouts, _ = A.load(run)
        names = []
        for r in attempts:
            if "op" in r and r["op"] not in names:
                names.append(r["op"])
        A.OPS[:] = names
        res = A.summarize(attempts, rollouts)
        per_task[task] = {"run": str(run), "summary": {k: v for k, v in res.items() if k != "conditions"}}
        for op, rows in res["conditions"].items():
            cls = {}
            for r in rows:
                k = classify(r)
                cls[k] = cls.get(k, 0) + 1
                strata.setdefault((task, op), []).append(r)
            s = res["per_op"].get(op, {})
            ev = " / ".join(f"{s.get(arm, {}).get('eval_success', '-')}/{s.get(arm, {}).get('n_eval', '-')}" for arm in "ABC")
            rows_md.append(f"| {task.replace('Humanoid-', '').replace('-v0', '')} | {op} | {len(rows)} | {cls.get('ceiling', 0)} | {cls.get('rescue', 0)} | "
                           f"{cls.get('noise', 0)} | {cls.get('none', 0)} | {ev} |")
    # pooled paired contrasts over task x operation strata
    pooled = {}
    for name, (x, y) in {"H1: B - A": ("B", "A"), "H2: C - B": ("C", "B")}.items():
        diffs = {f"{t}/{o}": [r[x]["eval_success"] / r[x]["n_eval"] - r[y]["eval_success"] / r[y]["n_eval"] for r in rows
                              if x in r and y in r and r[x]["n_eval"] and r[y]["n_eval"]] for (t, o), rows in strata.items()}
        est, ci = A.boot_ci(diffs)
        att = {f"{t}/{o}": [(r[y]["first_success"] or 10) - (r[x]["first_success"] or 10) for r in rows if x in r and y in r] for (t, o), rows in strata.items()}
        est_a, ci_a = A.boot_ci(att)
        pooled[name] = {"fresh_start_pp": None if est is None else 100 * est, "ci95_pp": [None if v is None else 100 * v for v in ci],
                        "attempts_saved": est_a, "ci95_attempts": ci_a, "strata": len([d for d in diffs.values() if d])}
    L = ["| task | operation | conditions | ceiling (original succeeds first try) | rescued by local search | succeeded only by repeating the original | no success | fresh-start success A / B / C |",
         "|---|---|---|---|---|---|---|---|"] + rows_md
    L += ["", "| pooled contrast | fresh-start success difference | 95 % CI | attempts saved to first success (censored at 10) | 95 % CI | strata |", "|---|---|---|---|---|---|"]
    f = lambda v: "n/a" if v is None else f"{v:+.1f}"  # noqa: E731
    for k, v in pooled.items():
        L.append(f"| {k} | {f(v['fresh_start_pp'])} pp | [{f(v['ci95_pp'][0])}, {f(v['ci95_pp'][1])}] | {f(v['attempts_saved'])} | "
                 f"[{f(v['ci95_attempts'][0])}, {f(v['ci95_attempts'][1])}] | {v['strata']} |")
    (out / "crosstask.md").write_text("\n".join(L) + "\n")
    (out / "crosstask.json").write_text(json.dumps({"per_task": per_task, "pooled": pooled}, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
