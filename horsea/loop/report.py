"""Success table for the internal-looping study (spec sec. 10, 13).

Paired differences vs a reference arm on identical (task, start, eval seed) keys; uncertainty by cluster bootstrap
that resamples start ids WITHIN each task (10,000 replicates, fixed analysis seed). Episodes are the unit.

    python -m horsea.loop.report --dir experiments/loop/dev --ref base_K10 [--csv summary.csv]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

PROFILE = "experiments/loop/profile_infer_B1_clean.json"
PROFILE_KEY = {"base_K10": "base_K10", "K11": "K11", "K12": "K12", "K13": "K13", "l0_noise": "l0_noise",
               "l0_mid": "l0_mid", "l2_act": "l2_act", "l1_all_ref": "l1_all", "C_u2000": "base_K10",
               "Ccomp_u2369": "base_K10", "Ccomp_u2369_K13": "K13", "L_u2000": "l0_all", "L_u2000_R1diag": "base_K10"}


def load(d):
    rs = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl")) if '"success"' in l]
    return {(r["task"], r["start"], r["seed"]): int(r["success"]) for r in rs}


def boot(a, b, keys, n=10000, seed=0, level=95):
    rng = np.random.default_rng(seed)
    tasks = sorted({k[0] for k in keys})
    # clusters = start states within each task; a cluster keeps all its evaluation/training-seed records
    per_task = {}
    for t in tasks:
        starts = sorted({k[1] for k in keys if k[0] == t})
        per_task[t] = [np.array([a[k] - b[k] for k in keys if k[0] == t and k[1] == st], dtype=float) for st in starts]
    stats = []
    for _ in range(n):
        tot, cnt = 0.0, 0
        for t, cl in per_task.items():
            for j in rng.integers(0, len(cl), len(cl)):
                tot += cl[j].sum()
                cnt += len(cl[j])
        stats.append(tot / cnt)
    lo = (100 - level) / 2
    diff = np.concatenate([c for cl in per_task.values() for c in cl]).mean()
    return 100 * diff, 100 * np.percentile(stats, lo), 100 * np.percentile(stats, 100 - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="base_K10")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    prof = json.load(open(PROFILE)) if os.path.exists(PROFILE) else {}
    arms = {os.path.basename(d): load(d) for d in sorted(glob.glob(os.path.join(args.dir, "*")))
            if os.path.exists(os.path.join(d, "episodes.jsonl"))}
    ref = arms[args.ref]
    rows = []
    print(f"| arm | n | success | per task | vs {args.ref} [95% CI] | block calls | median ms |")
    print("|---|---|---|---|---|---|---|")
    for name, a in arms.items():
        keys = sorted(set(a) & set(ref))
        tasks = sorted({k[0] for k in a})
        per = {t: np.mean([v for k, v in a.items() if k[0] == t]) for t in tasks}
        macro = np.mean(list(per.values()))
        d = boot(a, ref, keys) if name != args.ref and keys else None
        p = prof.get(PROFILE_KEY.get(name, ""), {})
        rows.append({"arm": name, "n": len(a), "successes": int(sum(a.values())), "success": np.mean(list(a.values())),
                     "task_macro": macro, **{f"task_{t}": per[t] for t in tasks},
                     "diff_vs_ref": d[0] if d else "", "ci_lo": d[1] if d else "", "ci_hi": d[2] if d else "",
                     "block_calls": p.get("block_calls", ""), "median_ms": p.get("median_ms", ""), "p95_ms": p.get("p95_ms", "")})
        dtxt = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*d) if d else "--"
        print(f"| {name} | {len(a)} | {100 * np.mean(list(a.values())):.1f}% | "
              f"{' / '.join(f'{100 * per[t]:.0f}' for t in tasks)} | {dtxt} | {p.get('block_calls', '?')} | "
              f"{p.get('median_ms', float('nan')):.2f} |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
