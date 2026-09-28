"""Protocol-v2 stage-1/2 reporting for the corrected hidden-rotation testbed.

    python -m horsea.v2_report select   # dev results -> experiments/protocol_v2/v2/frozen.json
    python -m horsea.v2_report table    # fresh (frozen) results: success, S1..S5, paired bootstrap

Score = mean success of attempts 2-5 over all (task, shift) sequences. Each method picks its own
configuration on dev (3 candidates each, mean over seeds); the choice is written before any fresh run.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "experiments/protocol_v2/v2"
BOUNDS = ["0.05", "0.1", "0.2"]
LR_CAPS = ["1", "3", "10"]
SEEDS = [0, 1, 2]
PASS_RULE = ("Horsea(write) minus each of {no-history, shuffled, last-only, memory-off} has a 95% paired "
             "bootstrap CI above 0 on the fresh test (3 seeds pooled); an advantage over TTT is claimed only "
             "if Horsea minus TTT also has a CI above 0 there.")


def seqs(d):
    """per-(task, shift) success matrix [n_seq, n_attempts] from one eval dir, ordered by task then shift"""
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "t*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
        r = json.load(open(f))
        rows += [[float(a["success"][i]) for a in r["attempts"]] for i in range(len(r["shifts"]))]
    return np.array(rows)


def score(d):
    m = seqs(d)
    return float(m[:, 1:].mean()) if m.size else float("nan")


def done(d, n=10):
    return len(glob.glob(os.path.join(d, "t*.json"))) >= n


def select():
    out = {"rule": PASS_RULE, "dev": {}}
    for name, key, cands in [("horsea", "proposed_b{}", BOUNDS), ("nohist", "nohist_b{}", BOUNDS),
                             ("ttt2", "ttt2_lr{}", LR_CAPS)]:
        res = {}
        for c in cands:
            ds = [f"{ROOT}/dev/s{s}/{key.format(c)}" for s in SEEDS]
            ok = [d for d in ds if done(d)]
            res[c] = float(np.mean([score(d) for d in ok])) if ok else float("nan")
        out["dev"][name] = res
        best = max((c for c in cands if not np.isnan(res[c])), key=lambda c: res[c], default=cands[-1])
        out[name] = best
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(f"{ROOT}/frozen.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def boot(a, b, n=10000, seed=0):
    d = a[:, 1:].mean(1) - b[:, 1:].mean(1)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def table(split="fresh"):
    conds = ["proposed_write", "proposed_nowrite", "proposed_shuffled", "proposed_last_only",
             "proposed_mismatched", "nohist", "ttt2", "ttt2_mismatched", "ttt2_info", "ttt2_info_dphi"]
    mats = {}
    for c in conds:
        ds = [f"{ROOT}/{split}/s{s}/{c}" for s in SEEDS]
        if all(done(d) for d in ds):
            mats[c] = np.concatenate([seqs(d) for d in ds])
    ref = mats.get("proposed_write")
    print(f"| condition | success (att. 2-5) | S1..S5 | Horsea minus this [95% CI] |\n|---|---|---|---|")
    for c, m in mats.items():
        s = " ".join(f"{100 * x:.0f}" for x in m.mean(0))
        diff = "" if c == "proposed_write" or ref is None else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(ref, m))
        print(f"| {c} | {100 * m[:, 1:].mean():.1f}% | {s} | {diff} |")


def select_info():
    """Tuning of the information-matched TTT arms (seed 0 dev, lr cap {1, 3, 10}) -> frozen_info.json."""
    out = {"dev": {}}
    for arm in ("ttt2_info", "ttt2_info_dphi"):
        res = {c: (score(d) if done(d := f"{ROOT}/dev/s0/{arm}_lr{c}") else float("nan")) for c in LR_CAPS}
        out["dev"][arm] = res
        out[arm] = max((c for c in LR_CAPS if not np.isnan(res[c])), key=lambda c: res[c], default="3")
    json.dump(out, open(f"{ROOT}/frozen_info.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"select": select, "select_info": select_info, "table": lambda: table(sys.argv[2] if len(sys.argv) > 2 else "fresh")}[sys.argv[1]]()

