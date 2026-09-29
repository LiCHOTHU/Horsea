"""RDM proof-of-concept bookkeeping.

    python -m horsea.rdm.report select_sigma   # rung 0 -> experiments/rdm/frozen.json
    python -m horsea.rdm.report table          # ladder table on validation starts

Rung-0 rule (declared before results): among sigma in {0.1, 0.05}, take the LARGEST whose stochastic-Plain mean
success over the five attempts is at most 10 points below deterministic Plain (more exploration unless it breaks
the base). The same sigma (and Bernoulli gripper kappa=4) is used for every method.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "experiments/rdm"


def load(d):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "t*.json")))]
    if not rows:
        return None
    return {"succ": np.concatenate([np.array(r["success"]) for r in rows]),
            "probe": np.concatenate([np.array(r["probe"]) for r in rows]), "tasks": [r["task"] for r in rows]}


def select_sigma():
    det = load(f"{ROOT}/rung0/plain_det")
    out = {"rule": __doc__.split("Rung-0 rule")[1].strip(), "det_mean": float(det["succ"].mean())}
    chosen = 0.05
    for s in (0.1, 0.05):
        r = load(f"{ROOT}/rung0/plain_s{s}")
        out[f"s{s}_mean"] = float(r["succ"].mean())
        if r["succ"].mean() >= det["succ"].mean() - 0.10:
            chosen = s
            break
    out["sigma"] = chosen
    json.dump(out, open(f"{ROOT}/frozen.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def boot(a, b, n=10000, seed=0):
    """paired bootstrap over complete sequences (the 5 dependent attempts stay together); metric: mean S2-S5."""
    d = a[:, 1:].mean(1) - b[:, 1:].mean(1)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def table():
    arms = ["plain_det", "plain", "adapter", "looped", "readonce", "reread", "ttt_info"]
    ladder = {"adapter": "plain", "looped": "adapter", "readonce": "looped", "reread": "readonce", "ttt_info": "reread"}
    R = {}
    for a in arms:
        ds = sorted(glob.glob(f"{ROOT}/eval/{a}_s*"))
        rs = [load(d) for d in ds]
        rs = [r for r in rs if r is not None]
        if rs:
            R[a] = {"succ": np.concatenate([r["succ"] for r in rs]), "probe": np.concatenate([r["probe"] for r in rs]),
                    "seeds": len(rs)}
    print("| arm | seeds | S1..S5 | mean S2-S5 | >=1 success | fresh probe | vs previous rung (S2-S5) [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    for a, r in R.items():
        s = r["succ"]
        prev = ladder.get(a)
        diff = ""
        if prev in R and R[prev]["succ"].shape == s.shape:
            diff = f"{prev}: " + "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(s, R[prev]["succ"]))
        print(f"| {a} | {r['seeds']} | {' '.join(f'{100 * x:.0f}' for x in s.mean(0))} | {100 * s[:, 1:].mean():.1f}% | "
              f"{100 * s.max(1).mean():.0f}% | {100 * r['probe'].mean():.0f}% | {diff} |")


if __name__ == "__main__":
    {"select_sigma": select_sigma, "table": table}[sys.argv[1]]()
