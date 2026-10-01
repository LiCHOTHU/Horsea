"""Development analysis of the loop-consistency study (experiments/rt/loopc/manifest.json).

Unit = initial scene (task, env seed) of the fixed prevalidated D list; the 2 noise replicates are averaged within a
scene, scenes within a task, tasks weighted equally. Intervals: task-stratified scene-cluster bootstrap (10,000, seed 0),
paired on scenes. S* follows the manifest rule (highest task-macro D success; tie -> lower own-graph held-out FM error;
tie -> lower block index).

    python -m horsea.rt.loopc_analysis
"""
import glob
import json
import os

import numpy as np

from horsea.rt.graphB_analysis import contrast, fmt, macro, table
from horsea.rt.graph_results import parse_log

R = "experiments/rt/loopc"
TASKS = ["handover_mic", "lift_pot", "open_microwave", "place_container_plate"]


def load(n_expected=20):
    rows, bad, retries = [], [], {}
    for log in sorted(glob.glob(f"{R}/D/*/*.log")):
        task = log.split("/")[-2]
        cfg, rep = os.path.basename(log)[:-4].rsplit("_r", 1)
        eps, nret = parse_log(log)
        retries[(task, cfg, rep)] = nret
        if len(eps) != n_expected:
            bad.append(f"{task}/{cfg}_r{rep}: {len(eps)}/{n_expected}")
            continue
        for e in eps:
            rows.append({"task": task, "cfg": cfg, "rep": int(rep), "seed": e["env_seed"], "y": e["success"]})
    return rows, bad, retries


def eplog_check(man):
    """Block calls per action chunk and latency from the per-decision logs (last watchdog attempt counts)."""
    exp = {**{m: man["phase1"]["block_calls_per_chunk"][m] for m in man["phase1"]["models"]},
           **{m: 50 for m in man["phase2"]["proposals"]}, **{m: 45 for m in ("U5", "R5", "A5")}}
    out = {}
    for p in sorted(glob.glob(f"{R}/D/*/*.eplog")):
        cfg = os.path.basename(p)[:-6].rsplit("_r", 1)[0]
        for line in open(p):
            d = json.loads(line)
            o = out.setdefault(cfg, {"calls": set(), "sec": [], "graphs": set()})
            o["calls"].add(d["block_calls"])
            o["sec"].append(d["sec"])
            o["graphs"].add(d["graph"])
    rep = {}
    for cfg, o in out.items():
        ok = o["calls"] == {exp.get(cfg)}
        rep[cfg] = {"block_calls": sorted(o["calls"]), "expected": exp.get(cfg), "ok": ok, "graphs": sorted(o["graphs"]),
                    "ms_per_decision_median": round(1e3 * float(np.median(o["sec"])), 2), "decisions": len(o["sec"])}
    return rep


def own_error(cfg, sched):
    p = f"{R}/diag/etable_{cfg}_s0.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    E, Eo = np.array(d["E_block"]), np.array(d["E_original"])
    return float(np.mean([Eo[k] if c == "-" else E[k, int(c)] for k, c in enumerate(sched[2:])]))


def main():
    man = json.load(open(f"{R}/manifest.json"))
    P1, P2 = man["phase1"]["models"], man["phase2"]["proposals"]
    rows, bad, retries = load()
    if bad:
        print("INCOMPLETE (excluded):", bad)
    T = table(rows)
    n_eps = {}
    for r in rows:
        n_eps[r["cfg"]] = n_eps.get(r["cfg"], 0) + 1
    complete = {c for c in {**P1, **P2} if n_eps.get(c, 0) == 160 and all(t in T.get(c, {}) for t in TASKS)}
    tasks = TASKS
    n_ret = sum(retries.values())
    print(f"loop-consistency development results: {len(rows)} episodes, watchdog retries {n_ret}; "
          f"complete models (160 episodes): {sorted(complete)}")
    print(f"| model | graph | block calls | success | {' | '.join(tasks)} | own-graph held-out FM err x1e3 |")
    print("|---|---|---|---|" + "---|" * len(tasks) + "---|")
    succ = {}
    for c, sch in {**P1, **P2}.items():
        oe = own_error(c, sch)
        calls = man["phase1"]["block_calls_per_chunk"].get(c, 50)
        if c not in complete:
            print(f"| {c} | `{sch}` | {calls} | incomplete ({n_eps.get(c, 0)}/160 episodes) |")
            continue
        m, per = macro(T, c, tasks)
        succ[c] = m
        print(f"| {c} | `{sch}` | {calls} | {100 * m:.1f}% | " + " | ".join(f"{100 * per[t]:.0f}" for t in tasks) +
              f" | {'--' if oe is None else f'{1e3 * oe:.4f}'} |")
    S = [f"S{l}" for l in range(4) if f"S{l}" in succ]
    if "N" in succ:
        for c in S:
            print(f"  {c} - N: {fmt(contrast(T, [(1, c), (-1, 'N')], tasks))}")
    if len(S) == 4 and "N" in succ:
        key = lambda c: (-succ[c], own_error(c, P1[c]) or 0.0, int(c[1]))
        s_star = sorted(S, key=key)[0]
        print(f"  S* (manifest rule, all 4 S models complete) = {s_star} ({100 * succ[s_star]:.1f}%)")
        comps = [("S* - N (selection-biased on D)", [(1, s_star), (-1, "N")]), ("T1 - S*", [(1, "T1"), (-1, s_star)]),
                 ("Tfree - S*", [(1, "Tfree"), (-1, s_star)]), ("Tfree - T1", [(1, "Tfree"), (-1, "T1")])]
        for name, terms in comps:
            if all(c in succ for _, c in terms):
                print(f"  {name}: {fmt(contrast(T, terms, tasks))}")
            else:
                print(f"  {name}: pending (incomplete models)")
    q = man["phase3_optional"].get("queued")
    if q:
        P3 = {"U5": q["U5"], "R5": q["R5"], "A5": q["A5"]}
        print(f"\nPhase 3 (block {q['block']} repeated at exactly 5 of 10 evaluations, 45 block calls):")
        for c, g in P3.items():
            if n_eps.get(c, 0) == 160 and all(t in T.get(c, {}) for t in TASKS):
                m, per = macro(T, c, tasks)
                succ[c] = m
                print(f"| {c} | `{g}` | 45 | {100 * m:.1f}% | " + " | ".join(f"{100 * per[t]:.0f}" for t in tasks) + " |")
            else:
                print(f"| {c} | `{g}` | 45 | incomplete ({n_eps.get(c, 0)}/160 episodes) |")
        s_star = f"S{q['block']}"
        for name, terms in [("A5 - U5 (A5 must beat)", [(1, "A5"), (-1, "U5")]), ("A5 - R5 (A5 must beat)", [(1, "A5"), (-1, "R5")]),
                            ("U5 - R5", [(1, "U5"), (-1, "R5")]),
                            (f"A5 - {s_star} (5 vs 10 looped evaluations)", [(1, "A5"), (-1, s_star)]),
                            (f"U5 - {s_star}", [(1, "U5"), (-1, s_star)]), (f"R5 - {s_star}", [(1, "R5"), (-1, s_star)]),
                            ("A5 - N", [(1, "A5"), (-1, "N")]), ("U5 - N", [(1, "U5"), (-1, "N")]), ("R5 - N", [(1, "R5"), (-1, "N")])]:
            if all(c in succ for _, c in terms):
                print(f"  {name}: {fmt(contrast(T, terms, tasks))}")
            else:
                print(f"  {name}: pending (incomplete models)")
    chk = eplog_check(man)
    for c, r in chk.items():
        print(f"  eplog {c}: block calls {r['block_calls']} (expected {r['expected']}) {'OK' if r['ok'] else 'MISMATCH'}; "
              f"graphs {r['graphs']}; {r['decisions']} decisions, median {r['ms_per_decision_median']} ms")


if __name__ == "__main__":
    main()
