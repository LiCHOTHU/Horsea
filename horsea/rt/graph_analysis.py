"""Discovery-A analysis for the RoboTwin graph study (spec 2026-10-01, sec. 9).

All quantities use ONE state set: per task, the initial states (env seeds) evaluated by every arm (both noise
replicates pooled). Unit = initial state: replicates are averaged within a state, states within a task, tasks equally.

Reports: (1) success table with paired, task-stratified cluster-bootstrap CIs vs G0 on the common states;
(2) noise floor: G0 replicate 0 vs replicate 1 disagreement on identical states; (3) stability of graph effects across
noise replicates and across grouped split-halves of the states; (4) honest selection estimates: choose the global /
task-wise best graph on one half of the states, score it on the other half (and vice versa) -- no hindsight maximum.

    python -m horsea.rt.graph_analysis --dir experiments/rt/graph/discovery
"""
import argparse
import json
import os

import numpy as np

from horsea.rt.graph_results import collect, common_seeds

LOOPS = [f"b{b}_{w}" for b in range(4) for w in ("early", "middle", "late")]


def table_of(rows, seeds):
    """arm -> task -> seed -> mean success over replicates (restricted to `seeds`)."""
    T = {}
    for r in rows:
        if r["env_seed"] in seeds.get(r["task"], ()):
            T.setdefault(r["arm"], {}).setdefault(r["task"], {}).setdefault(r["env_seed"], []).append(r["success"])
    return {a: {t: {s: float(np.mean(v)) for s, v in st.items()} for t, st in ts.items()} for a, ts in T.items()}


def macro(T, arm, tasks, seeds_by_task=None):
    vals = []
    for t in tasks:
        ss = seeds_by_task[t] if seeds_by_task else T[arm][t].keys()
        vals.append(np.mean([T[arm][t][s] for s in ss]))
    return float(np.mean(vals)), {t: float(v) for t, v in zip(tasks, vals)}


def boot_diff(T, a, b, tasks, n=10000, seed=0, level=95):
    rng = np.random.default_rng(seed)
    d = {t: np.array([T[a][t][s] - T[b][t][s] for s in sorted(T[a][t])]) for t in tasks}
    point = np.mean([x.mean() for x in d.values()])
    bs = np.mean([np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]) for x in d.values()], 0)
    lo = (100 - level) / 2
    return 100 * point, 100 * np.percentile(bs, lo), 100 * np.percentile(bs, 100 - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    rows, inc = collect(args.dir)
    assert not inc, inc
    arms = sorted({r["arm"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    common = common_seeds(rows, arms)
    T = table_of(rows, common)
    out = {"common_states": {t: len(v) for t, v in common.items()}}
    print("common initial states per task:", out["common_states"], "total", sum(out["common_states"].values()))

    # (1) table
    print("\n(1) success on the common states (replicates averaged within a state); paired CI vs G0")
    print(f"| arm | macro | {' | '.join(tasks)} | vs G0 [95% CI] |\n|---|---|" + "---|" * len(tasks) + "---|")
    res = {}
    for a in ["G0", "K11", "K12"] + LOOPS:
        m, pt = macro(T, a, tasks)
        d = None if a == "G0" else boot_diff(T, a, "G0", tasks)
        res[a] = {"macro": m, "per_task": pt, "diff_vs_G0": d}
        print(f"| {a} | {100 * m:.1f}% | " + " | ".join(f"{100 * pt[t]:.0f}" for t in tasks) +
              (" | {:+.1f} [{:+.1f}, {:+.1f}] |".format(*d) if d else " | -- |"))
    out["table"] = res

    # (2) noise floor: G0 rep0 vs rep1 on identical states
    by = {}
    for r in rows:
        if r["arm"] == "G0" and r["env_seed"] in common[r["task"]]:
            by.setdefault((r["task"], r["env_seed"]), {})[r["rep"]] = r["success"]
    pairs = [v for v in by.values() if 0 in v and 1 in v]
    disc = np.mean([v[0] != v[1] for v in pairs])
    print(f"\n(2) noise floor: G0 replicate 0 vs 1 disagree on {100 * disc:.0f}% of {len(pairs)} identical initial states")
    out["G0_rep_disagreement"] = float(disc)

    # (3) stability across replicates: effect vs G0 estimated within each replicate
    def effects(filter_fn):
        R = [r for r in rows if filter_fn(r)]
        Tt = table_of(R, common)
        e = {}
        for a in ["K11", "K12"] + LOOPS:
            if a in Tt and all(t in Tt[a] and t in Tt["G0"] for t in tasks):
                shared = {t: sorted(set(Tt[a][t]) & set(Tt["G0"][t])) for t in tasks}
                e[a] = float(np.mean([np.mean([Tt[a][t][s] - Tt["G0"][t][s] for s in shared[t]]) for t in tasks]))
        return e
    e0, e1 = effects(lambda r: r["rep"] == 0), effects(lambda r: r["rep"] == 1)
    ks = sorted(set(e0) & set(e1))
    rho_rep = float(np.corrcoef([e0[k] for k in ks], [e1[k] for k in ks])[0, 1])
    print(f"(3a) graph effects vs G0, replicate 0 vs replicate 1: Pearson r = {rho_rep:+.2f} over {len(ks)} arms")
    # grouped split halves of the STATES (alternating sorted seeds within each task)
    halves = [{t: set(sorted(common[t])[i::2]) for t in tasks} for i in (0, 1)]
    TA, TB = table_of(rows, halves[0]), table_of(rows, halves[1])
    eff = lambda Tx: {a: float(np.mean([np.mean([Tx[a][t][s] - Tx["G0"][t][s] for s in Tx[a][t]]) for t in tasks]))
                      for a in ["K11", "K12"] + LOOPS}
    eA, eB = eff(TA), eff(TB)
    rho_half = float(np.corrcoef([eA[k] for k in eA], [eB[k] for k in eA])[0, 1])
    print(f"(3b) graph effects vs G0, state half A vs half B: Pearson r = {rho_half:+.2f}")
    out["stability"] = {"replicate_r": rho_rep, "split_half_r": rho_half}

    # (4) honest selection: pick on one half, evaluate on the other (both directions)
    def task_eff(Tx, a, t):
        return float(np.mean([Tx[a][t][s] - Tx["G0"][t][s] for s in Tx[a][t]]))
    sel = {}
    for name, (Tsel, Tev) in {"A->B": (TA, TB), "B->A": (TB, TA)}.items():
        g_best = max(LOOPS, key=lambda a: np.mean([task_eff(Tsel, a, t) for t in tasks]))
        glob_in = np.mean([task_eff(Tsel, g_best, t) for t in tasks])
        glob_out = np.mean([task_eff(Tev, g_best, t) for t in tasks])
        tw = {t: max(LOOPS, key=lambda a: task_eff(Tsel, a, t)) for t in tasks}
        tw_in = np.mean([task_eff(Tsel, tw[t], t) for t in tasks])
        tw_out = np.mean([task_eff(Tev, tw[t], t) for t in tasks])
        sel[name] = {"global_graph": g_best, "global_in_sample": glob_in, "global_held_out": glob_out,
                     "task_graphs": tw, "task_in_sample": tw_in, "task_held_out": tw_out}
        print(f"(4) select on {name[0]}, score on {name[-1]}: global best {g_best}: in-sample {100 * glob_in:+.1f}, "
              f"held-out {100 * glob_out:+.1f} | task-wise {tw}: in-sample {100 * tw_in:+.1f}, held-out {100 * tw_out:+.1f}")
    out["selection_split_half"] = sel
    # hindsight maximum (labelled as such; NOT achievable performance)
    hind = np.mean([np.mean([max(T[a][t][s] for a in LOOPS) for s in T["G0"][t]]) for t in tasks])
    print(f"(5) 'at least one loop graph succeeded in hindsight' (replicate-averaged per state; not achievable): "
          f"{100 * hind:.1f}% vs G0 {100 * res['G0']['macro']:.1f}%")
    out["hindsight_not_achievable"] = float(hind)
    json.dump(out, open(os.path.join(args.dir, "analysis_A.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
