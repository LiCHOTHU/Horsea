"""Episode-level records and paired analysis for the RoboTwin graph study.

RoboTwin prints a cumulative 'Success rate: s/n => .., current seed: X' after each test episode. Per-episode success is
the increment of s; X is that episode's env seed; the episode index n-1 is the pairing key (policy-independent: the
expert check that accepts seeds does not depend on the policy). eval_safe.sh may retry a job; only the LAST attempt's
lines are used and the number of attempts is recorded.

    python -m horsea.rt.graph_results --dir experiments/rt/graph/discovery
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LINE = re.compile(r"Success rate:\s*(?:\x1b\[[0-9;]*m)?(\d+)/(\d+)(?:\x1b\[[0-9;]*m)?.*?current seed:\s*(?:\x1b\[[0-9;]*m)?(\d+)")


def parse_log(path):
    txt = open(path, errors="ignore").read()
    parts = txt.split("[eval_safe] attempt")
    last = parts[-1]
    eps, prev = [], 0
    for m in LINE.finditer(last):
        s, n, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        eps.append({"episode": n - 1, "env_seed": seed, "success": s - prev})
        prev = s
    return eps, len(parts) - 1


def collect(root):
    rows = []
    for log in sorted(glob.glob(os.path.join(root, "*", "*.log"))):
        task = os.path.basename(os.path.dirname(log))
        arm, rep = os.path.basename(log)[:-4].rsplit("_r", 1)
        eps, retries = parse_log(log)
        lat = []
        epl = log[:-4] + ".eplog"
        if os.path.exists(epl):
            lat = [json.loads(l)["sec"] for l in open(epl)]
        for e in eps:
            rows.append({"task": task, "arm": arm, "rep": int(rep), **e, "retries": retries})
    return rows


def macro(rows, arm):
    """task-macro success: replicates averaged within a start, starts within a task, tasks equally."""
    by = {}
    for r in rows:
        if r["arm"] == arm:
            by.setdefault(r["task"], {}).setdefault(r["episode"], []).append(r["success"])
    per_task = {t: float(np.mean([np.mean(v) for v in st.values()])) for t, st in by.items()}
    return (float(np.mean(list(per_task.values()))) if per_task else float("nan")), per_task


def paired_boot(rows, a, b, n=10000, seed=0, level=95):
    """task-stratified cluster bootstrap over starts; replicates of a start stay together; tasks weighted equally."""
    by = {}
    for r in rows:
        if r["arm"] in (a, b):
            by.setdefault(r["task"], {}).setdefault(r["episode"], {}).setdefault(r["arm"], []).append(r["success"])
    tasks = {t: [np.mean(v[a]) - np.mean(v[b]) for v in st.values() if a in v and b in v] for t, st in by.items()}
    tasks = {t: np.array(d) for t, d in tasks.items() if len(d)}
    rng = np.random.default_rng(seed)
    point = np.mean([d.mean() for d in tasks.values()])
    bs = np.mean([np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]) for d in tasks.values()], 0)
    lo = (100 - level) / 2
    return 100 * point, 100 * np.percentile(bs, lo), 100 * np.percentile(bs, 100 - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="G0")
    args = ap.parse_args()
    rows = collect(args.dir)
    with open(os.path.join(args.dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (x != args.ref, x))
    tasks = sorted({r["task"] for r in rows})
    print(f"| arm | n | task-macro | {' | '.join(tasks)} | vs {args.ref} [95% CI] |")
    print("|---|---|---|" + "---|" * len(tasks) + "---|")
    for a in arms:
        m, pt = macro(rows, a)
        n = sum(1 for r in rows if r["arm"] == a)
        d = "" if a == args.ref else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*paired_boot(rows, a, args.ref))
        print(f"| {a} | {n} | {100 * m:.1f}% | " + " | ".join(f"{100 * pt.get(t, float('nan')):.0f}" for t in tasks) + f" | {d} |")


if __name__ == "__main__":
    main()
