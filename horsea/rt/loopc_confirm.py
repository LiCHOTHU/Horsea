"""Confirmation analysis of the loop-consistency study on the untouched T scenes (manifest 'confirmation').

Unit = scene (task, env seed). Success is averaged over the 3 training seeds within each scene, scenes within a task,
tasks weighted equally. Contrasts S3 - N, T1 - S3, Tfree - S3, Tfree - T1 with 95% CIs from a task-stratified
scene-cluster bootstrap (10,000, seed 0), training seeds held fixed; per-seed contrasts for consistency.

    python -m horsea.rt.loopc_confirm
"""
import glob
import os

import numpy as np

from horsea.rt.graphB_analysis import contrast, fmt, macro
from horsea.rt.graph_results import parse_log

R = "experiments/rt/loopc/T"
TASKS = ["handover_mic", "lift_pot", "open_microwave", "place_container_plate"]
MODELS = ["N", "S3", "T1", "Tfree"]
SEEDS = (0, 1, 2)
DEV = {"S3 - N": "+6.9 [+0.0, +13.1]", "T1 - S3": "-1.9 [-8.1, +5.0]", "Tfree - S3": "-13.8 [-21.2, -6.2]",
       "Tfree - T1": "-11.9 [-19.4, -5.0]"}


def load(n_expected=50):
    Y, bad, ret = {}, [], 0
    for log in sorted(glob.glob(f"{R}/*/*.log")):
        task = log.split("/")[-2]
        model, seed = os.path.basename(log)[:-4].rsplit("_s", 1)
        eps, r = parse_log(log)
        ret += r
        if len(eps) != n_expected:
            bad.append(f"{task}/{model}_s{seed}: {len(eps)}/{n_expected}")
            continue
        for e in eps:
            Y.setdefault(model, {}).setdefault(int(seed), {}).setdefault(task, {})[e["env_seed"]] = float(e["success"])
    return Y, bad, ret


def main():
    Y, bad, ret = load()
    if bad:
        print("INCOMPLETE (excluded):", bad)
    full = [m for m in MODELS if m in Y and all(s in Y[m] and all(t in Y[m][s] for t in TASKS) for s in SEEDS)]
    print(f"confirmation on T (50 scenes per task, 1 noise replicate): watchdog retries {ret}; complete models (3 seeds): {full}")
    # pooled table: scene success averaged over the training seeds
    P = {m: {t: {sc: float(np.mean([Y[m][s][t][sc] for s in SEEDS])) for sc in Y[m][0][t]} for t in TASKS} for m in full}
    print(f"| model | success (3-seed mean) | per seed | {' | '.join(TASKS)} |\n|---|---|---|" + "---|" * len(TASKS))
    for m in full:
        mm, per = macro(P, m, TASKS)
        ps = " / ".join(f"{100 * macro({m: Y[m][s]}, m, TASKS)[0]:.1f}" for s in SEEDS)
        print(f"| {m} | {100 * mm:.1f}% | {ps} | " + " | ".join(f"{100 * per[t]:.0f}" for t in TASKS) + " |")
    for name, a, b in [("S3 - N", "S3", "N"), ("T1 - S3", "T1", "S3"), ("Tfree - S3", "Tfree", "S3"), ("Tfree - T1", "Tfree", "T1")]:
        if a in full and b in full:
            c = contrast(P, [(1, a), (-1, b)], TASKS)
            per_seed = ", ".join(f"s{s} {100 * (macro({a: Y[a][s]}, a, TASKS)[0] - macro({b: Y[b][s]}, b, TASKS)[0]):+.1f}" for s in SEEDS)
            verdict = "CI excludes 0" if c[1] > 0 or c[2] < 0 else "CI includes 0"
            print(f"  {name}: {fmt(c)} ({verdict}; per seed {per_seed}; development {DEV[name]})")
        else:
            print(f"  {name}: pending")


if __name__ == "__main__":
    main()
