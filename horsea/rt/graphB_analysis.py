"""Stage B analysis: B1 focused pilot and B2 same-M graph matrix (spec 2026-10-01 rev. 2, sec. 10-12).

Unit = initial scene (task, env seed) from the fixed prevalidated D list; noise replicates are averaged within a scene,
scenes within a task, tasks weighted equally. Intervals: task-stratified scene-cluster bootstrap (10,000, seed 0).

    python -m horsea.rt.graphB_analysis
"""
import glob
import json
import os

import numpy as np

from horsea.rt.graph_results import parse_log

R = "experiments/rt/graphB"
LOOPS = [f"b{b}_{w}" for b in range(4) for w in ("early", "middle", "late")]


def load(stage, n_expected):
    rows, bad = [], []
    for log in sorted(glob.glob(f"{R}/{stage}/*/*.log")):
        task = log.split("/")[-2]
        cfg, rep = os.path.basename(log)[:-4].rsplit("_r", 1)
        eps, retries = parse_log(log)
        if len(eps) != n_expected:
            bad.append(f"{task}/{cfg}_r{rep}: {len(eps)}/{n_expected}")
            continue
        for e in eps:
            rows.append({"task": task, "cfg": cfg, "rep": int(rep), "seed": e["env_seed"], "y": e["success"], "retries": retries})
    return rows, bad


def table(rows, scenes=None):
    """cfg -> task -> seed -> replicate-mean success."""
    T = {}
    for r in rows:
        if scenes is None or r["seed"] in scenes[r["task"]]:
            T.setdefault(r["cfg"], {}).setdefault(r["task"], {}).setdefault(r["seed"], []).append(r["y"])
    return {c: {t: {s: float(np.mean(v)) for s, v in st.items()} for t, st in ts.items()} for c, ts in T.items()}


def macro(T, c, tasks):
    per = {t: float(np.mean(list(T[c][t].values()))) for t in tasks}
    return float(np.mean(list(per.values()))), per


def contrast(T, terms, tasks, n=10000, seed=0, level=95):
    """terms: list of (coef, cfg). Paired on scenes present in every cfg of the contrast."""
    rng = np.random.default_rng(seed)
    per = {}
    for t in tasks:
        common = sorted(set.intersection(*[set(T[c][t]) for _, c in terms]))
        per[t] = np.array([sum(w * T[c][t][s] for w, c in terms) for s in common])
    point = np.mean([d.mean() for d in per.values()])
    bs = np.mean([np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]) for d in per.values()], 0)
    lo = (100 - level) / 2
    return 100 * point, 100 * np.percentile(bs, lo), 100 * np.percentile(bs, 100 - lo)


def fmt(c):
    return "{:+.1f} [{:+.1f}, {:+.1f}]".format(*c)


def b1():
    rows, bad = load("B1", 20)
    if bad:
        print("B1 EXCLUDED (incomplete):", bad)
    tasks = sorted({r["task"] for r in rows})
    T = table(rows)
    print(f"\nB1 focused pilot ({len(rows)} episodes; 20 D scenes x 2 replicates per task)")
    print(f"| config | success | {' | '.join(tasks)} |\n|---|---|" + "---|" * len(tasks))
    for c in ["orig_G0", "C_G0", "C_b0_middle", "C_K11", "M_G0", "M_b0_middle"]:
        if c in T:
            m, per = macro(T, c, tasks)
            print(f"| {c} | {100 * m:.1f}% | " + " | ".join(f"{100 * per[t]:.0f}" for t in tasks) + " |")
    out = {}
    for name, terms in [("M_b0_middle - C_G0 (main)", [(1, "M_b0_middle"), (-1, "C_G0")]),
                        ("M_b0_middle - C_K11 (main)", [(1, "M_b0_middle"), (-1, "C_K11")]),
                        ("M_b0_middle - M_G0 (runtime graph value, same weights)", [(1, "M_b0_middle"), (-1, "M_G0")]),
                        ("interaction I = (M_b0m - M_G0) - (C_b0m - C_G0)", [(1, "M_b0_middle"), (-1, "M_G0"), (-1, "C_b0_middle"), (1, "C_G0")]),
                        ("C_G0 - orig_G0 (ordinary continuation)", [(1, "C_G0"), (-1, "orig_G0")]),
                        ("M_G0 - C_G0 (multi-graph training, G0 execution)", [(1, "M_G0"), (-1, "C_G0")]),
                        ("C_b0_middle - C_G0 (frozen-style loop after C)", [(1, "C_b0_middle"), (-1, "C_G0")])]:
        if all(c in T for _, c in terms):
            out[name] = contrast(T, terms, tasks)
            print(f"  {name}: {fmt(out[name])}")
    return T, tasks


def b2(T1):
    rows, bad = load("B2", 12)
    if bad:
        print("B2 EXCLUDED (incomplete):", bad)
    rows = [r for r in rows if r["cfg"] != "M_G0_dup"]
    tasks = sorted({r["task"] for r in rows})
    sub = {t: sorted(T1["M_G0"][t])[:0] for t in tasks}  # placeholder, replaced below
    # the B2 subset = the first 12 listed D scenes (same order as the scene files)
    sub = {t: set(s["seed"] for s in json.load(open(f"{R}/scenes/D_{t}.json"))["scenes"][:12]) for t in tasks}
    T2 = table(rows)
    for c in ("M_G0", "M_b0_middle"):                       # valid reuse from B1 on the same 12 scenes
        T2[c] = {t: {s: v for s, v in T1[c][t].items() if s in sub[t]} for t in tasks}
    cfgs = ["M_G0"] + [f"M_{g}" for g in LOOPS] + ["M_K11", "M_K12"]
    print(f"\nB2 same-M matrix (12 D scenes x 2 replicates per task)")
    print(f"| config | success | {' | '.join(tasks)} | vs M_G0 [95% CI] |\n|---|---|" + "---|" * len(tasks) + "---|")
    eff = {}
    for c in cfgs:
        if c not in T2:
            continue
        m, per = macro(T2, c, tasks)
        d = None if c == "M_G0" else contrast(T2, [(1, c), (-1, "M_G0")], tasks)
        eff[c] = d
        print(f"| {c} | {100 * m:.1f}% | " + " | ".join(f"{100 * per[t]:.0f}" for t in tasks) + (f" | {fmt(d)} |" if d else " | -- |"))
    loops = [f"M_{g}" for g in LOOPS if f"M_{g}" in T2]
    if len(loops) == 12:
        rnd = np.mean([macro(T2, c, tasks)[0] for c in loops])
        print(f"  uniform random loop graph (expectation over the 12): {100 * rnd:.1f}%")
        # split-half stability over scenes and replicate stability
        halves = [{t: set(sorted(sub[t])[i::2]) for t in tasks} for i in (0, 1)]
        def effs(Tx):
            return {c: np.mean([np.mean([Tx[c][t][s] - Tx["M_G0"][t][s] for s in Tx[c][t]]) for t in tasks]) for c in loops}
        all_rows = rows + [{"task": t, "cfg": c, "rep": r["rep"], "seed": r["seed"], "y": r["y"]} for c in ("M_G0", "M_b0_middle")
                           for r in load("B1", 20)[0] if r["cfg"] == c for t in [r["task"]] if r["seed"] in sub[t]]
        eA, eB = effs(table(all_rows, halves[0])), effs(table(all_rows, halves[1]))
        r_half = float(np.corrcoef([eA[c] for c in loops], [eB[c] for c in loops])[0, 1])
        e0 = effs(table([r for r in all_rows if r["rep"] == 0], sub))
        e1 = effs(table([r for r in all_rows if r["rep"] == 1], sub))
        r_rep = float(np.corrcoef([e0[c] for c in loops], [e1[c] for c in loops])[0, 1])
        best = max(loops, key=lambda c: macro(T2, c, tasks)[0])
        print(f"  stability: split-half r = {r_half:+.2f}, replicate r = {r_rep:+.2f}; development-selected global graph: {best}")
        bA = max(loops, key=lambda c: eA[c]); bB = max(loops, key=lambda c: eB[c])
        print(f"  honest selection: best on half A = {bA} ({100 * eA[bA]:+.1f}) -> half B {100 * eB[bA]:+.1f}; "
              f"best on B = {bB} ({100 * eB[bB]:+.1f}) -> half A {100 * eA[bB]:+.1f}")
        tmap = {t: max(loops, key=lambda c: np.mean(list(T2[c][t].values()))) for t in tasks}
        print(f"  task-wise development mapping (in-sample): {tmap}")
    # simulator determinism: identical config + noise key
    dup, _ = load("B2", 12)
    D = {(r["task"], r["seed"]): r["y"] for r in dup if r["cfg"] == "M_G0_dup"}
    O = {(r["task"], r["seed"]): r["y"] for r in load("B1", 20)[0] if r["cfg"] == "M_G0" and r["rep"] == 0}
    k = [x for x in D if x in O]
    if k:
        print(f"  identical-config repeat (M_G0, replicate 0, same noise key): outcome differs on {100 * np.mean([D[x] != O[x] for x in k]):.0f}% "
              f"of {len(k)} scenes (simulator non-determinism)")
    return T2


if __name__ == "__main__":
    T1, _ = b1()
    if glob.glob(f"{R}/B2/*/*.log"):
        b2(T1)
