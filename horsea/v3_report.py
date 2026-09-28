"""Protocol-v2 stage 3, question 1: adaptation on seen LIBERO-10 tasks (RoboTTT protocol, horsea.history2).

    python -m horsea.v3_report select   # seed-0 dev results -> experiments/history2_s3/frozen.json
    python -m horsea.v3_report table    # test results (starts 40-49): success per arm, paired bootstrap

Tuning budget: 3 candidates per tunable method, chosen on dev (seed 0, starts 30-34) before any test run:
TTT and TTT-info inner-lr cap {1, 3, 10} (training-time), Horsea (energy2) solver bound {0.05, 0.1, 0.2}
(test-time). TTT-info-dPhi (the outer-objective ablation) uses TTT-info's cap. Plain has no knob.
Episodes end at success or 520 steps, identically for every method.
"""
import glob
import json
import os
import sys

import numpy as np

EXP = "experiments"
FROZEN = f"{EXP}/history2_s3/frozen.json"
CAPS = ["1", "3", "10"]
STEPS = ["0.05", "0.1", "0.2"]
SEEDS = [0, 1, 2]
PASS_RULE = ("Seen-task adaptation is claimed only if Horsea(write) minus Plain and minus Horsea(memory off) both "
             "have 95% paired bootstrap CIs above 0 on the test starts (3 seeds pooled). A mechanism advantage is "
             "claimed only if Horsea minus TTT-info (same information) also has a CI above 0; the objective effect "
             "is TTT-info-dPhi minus TTT-info.")


def mat(d):
    """success matrix [task, episode] from one eval dir"""
    fs = sorted(glob.glob(os.path.join(d, "t*.json")), key=lambda p: int(os.path.basename(p)[1:-5]))
    return np.array([json.load(open(f))["success"] for f in fs], dtype=float)


def done(d, n=10):
    return len(glob.glob(os.path.join(d, "t*.json"))) >= n


def ev(tag, name, suffix):
    return f"{EXP}/history2{tag}/eval/{name}{suffix}"


def select():
    out = {"rule": PASS_RULE, "dev": {}}
    for mode in ("ttt", "ttt_info"):
        res = {c: (float(mat(d).mean()) if done(d := ev(f"_s3_s0_lr{c}", mode, "_dev")) else float("nan")) for c in CAPS}
        out["dev"][mode] = res
        out[mode] = max((c for c in CAPS if not np.isnan(res[c])), key=lambda c: res[c], default="3")
    res = {s: (float(mat(d).mean()) if done(d := ev("_s3_s0", "energy2", f"_step{s}_dev")) else float("nan")) for s in STEPS}
    out["dev"]["energy2"] = res
    out["energy2"] = max((s for s in STEPS if not np.isnan(res[s])), key=lambda s: res[s], default="0.05")
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(out, open(FROZEN, "w"), indent=1)
    print(json.dumps(out, indent=1))


def boot(a, b, n=10000, seed=0):
    d = (a - b).reshape(-1)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def table():
    fz = json.load(open(FROZEN))
    step = fz["energy2"]
    arms = {"Plain": lambda s: ev(f"_s3_s{s}", "plain", "_test")}
    for mode, lab in [("ttt", "TTT (native)"), ("ttt_info", "TTT-info"), ("ttt_info_dphi", "TTT-info-dPhi")]:
        cap = fz["ttt_info" if mode == "ttt_info_dphi" else mode]
        for v, vl in [("", ""), ("_nowrite", ", memory off"), ("_mismatched", ", mismatched")]:
            arms[lab + vl] = (lambda m, c, v: lambda s: ev(f"_s3_s{s}_lr{c}", m, v + "_test"))(mode, cap, v)
    for v, vl in [("", ""), ("_nowrite", ", memory off"), ("_last_only", ", last only"), ("_mismatched", ", mismatched")]:
        arms["Horsea" + vl] = (lambda v: lambda s: ev(f"_s3_s{s}", "energy2", f"{v}_step{step}_test"))(v)
    M = {}
    for k, f in arms.items():
        ds = [f(s) for s in SEEDS]
        ok = [d for d in ds if done(d)]
        if ok:
            M[k] = (len(ok), np.concatenate([mat(d) for d in ok]))
    ref = M.get("Horsea")
    print(f"frozen: {json.dumps({k: fz[k] for k in ('ttt', 'ttt_info', 'energy2')})}")
    print("| method | seeds | success | Horsea minus this [95% CI] |\n|---|---|---|---|")
    for k, (n, m) in M.items():
        diff = ""
        if ref is not None and k != "Horsea" and m.shape == ref[1].shape:
            diff = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(ref[1], m))
        print(f"| {k} | {n} | {100 * m.mean():.1f}% | {diff} |")


if __name__ == "__main__":
    {"select": select, "table": table}[sys.argv[1]]()
