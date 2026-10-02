"""Offline replay of the capability audit through the memory and selectors (UPPER BOUND: perfect feedback).

Feedback events here come from the evaluation-only truth records (door angle, gripper-door contact), i.e. a perfect
extractor; the online method will have to read them from cameras/proprioception. Per scene, probing option c draws
one of its recorded attempts (with replacement); after 3 probes the posterior-mean option is executed, scored by the
mean success of that option's attempts in the scene that were not drawn during probing (all, if every one was drawn).
Priors are fitted leave-one-scene-out (kappa = 2).

    python -m horsea.explore.replay_audit --raw experiments/explore/audit_yaw/raw
"""
import argparse
import collections
import glob
import json

import numpy as np

from horsea.explore.memory import Scalar, Structured, choose, utility_index

OPTIONS = [(0, "track"), (0, "pull"), (1, "track"), (1, "pull")]
DOOR = "link_0"
FINGERS = ("fl_link7", "fl_link8")


def door_contact(rec):
    return any((a == DOOR and b in FINGERS) or (b == DOOR and a in FINGERS) for a, b in rec["gripper_contacts"])


def events(t, goal=0.6, eps=0.05):
    """(prefix event index, tail category index or None, success) from one truth record (plan's vocabulary/order)."""
    R = {r["tag"]: r for r in t["records"]}
    pre = R.get("prefix")
    end = t["records"][-1]
    if pre is None:
        return 2, None, False
    if pre["door_frac"] >= goal:
        return 0, None, end["success"]
    if not door_contact(pre):
        return 2, None, end["success"]                         # not_ready
    tails = [r for r in t["records"] if r["tag"].startswith("tail")]
    if not tails:
        return 1, 5, end["success"]                             # ready, tail unknown (did not execute)
    if end["success"]:
        return 1, 4, True                                       # goal
    if any(not door_contact(r) for r in tails):
        return 1, 0, False                                      # slip / loss of contact
    d = tails[-1]["door_frac"] - pre["door_frac"]
    if d >= eps:
        return 1, 3, False                                      # toward the goal
    if d <= -eps:
        return 1, 2, False                                      # away
    return 1, 1, False                                          # no response


def load(raw):
    A = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(glob.glob(f"{raw}/*.truth.json")):
        t = json.load(open(p))
        A[t["scene"]][(t["grasp"], t["mode"])].append(events(t))
    return A


def priors(A, exclude):
    qc = {r: np.zeros(4) for r in (0, 1)}
    pc = {c: np.zeros(6) for c in OPTIONS}
    for s, opts in A.items():
        if s == exclude:
            continue
        for c, evs in opts.items():
            for g, k, _ in evs:
                qc[c[0]][g] += 1
                if g == 1 and k is not None:
                    pc[c][k] += 1
    return ({r: v / v.sum() for r, v in qc.items()}, {c: v / max(v.sum(), 1) for c, v in pc.items()})


def meta_episode(A, s, method, rng, q0, p0, probes=3):
    mem = Structured([0, 1], ["track", "pull"], q0, p0, kappa=2.0)
    sc = Scalar(mem, conc=3.0) if method == "scalar_voi" else None
    used = collections.defaultdict(set)
    for _ in range(probes):
        if method == "prior":
            break
        if method == "scalar_voi":
            c = choose(sc, "voi", rng)
        else:
            c = choose(mem, {"greedy": "greedy", "thompson": "thompson", "voi": "voi", "random": "random"}[method], rng)
        i = int(rng.integers(len(A[s][c])))
        used[c].add(i)
        g, k, _ = A[s][c][i]
        if sc is not None:
            sc.update(c, utility_index(g, k))
        mem.update(c, g, k if g == 1 else None)
    final = max((sc or mem).options, key=(sc or mem).score)
    rest = [j for j in range(len(A[s][final])) if j not in used[final]] or list(range(len(A[s][final])))
    return float(np.mean([A[s][final][j][2] for j in rest])), final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/explore/audit_yaw/raw")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    A = load(a.raw)
    scenes = sorted(A)
    rate = {s: {c: np.mean([e[2] for e in A[s][c]]) for c in OPTIONS} for s in scenes}
    best_fixed = max(OPTIONS, key=lambda c: np.mean([rate[s][c] for s in scenes]))
    print(f"{a.raw}: {len(scenes)} scenes; best fixed option r{best_fixed[0]}+{best_fixed[1]} "
          f"{100 * np.mean([rate[s][best_fixed] for s in scenes]):.1f}%; per-scene oracle {100 * np.mean([max(rate[s].values()) for s in scenes]):.1f}%")
    ev = collections.Counter((c, g, k) for s in scenes for c in OPTIONS for g, k, _ in A[s][c])
    print("event frequencies (option, prefix, tail):", {f"r{c[0]}+{c[1]}": {f"{g}/{k}": n for (cc, g, k), n in ev.items() if cc == c} for c in OPTIONS})
    res = {}
    for method in ("prior", "scalar_voi", "random", "greedy", "thompson", "voi"):
        per_scene = []
        for s in scenes:
            q0, p0 = priors(A, s)
            rng = np.random.default_rng(hash((method, s)) % 2**32)
            per_scene.append(np.mean([meta_episode(A, s, method, rng, q0, p0)[0] for _ in range(a.reps)]))
        res[method] = np.array(per_scene)
    print("\nfresh exploitation success after 3 probes (perfect-feedback upper bound), mean over scenes:")
    for m, v in res.items():
        d = v - res["prior"]
        bs = [np.mean(d[np.random.default_rng(i).integers(0, len(d), len(d))]) for i in range(2000)]
        print(f"  {m:11} {100 * v.mean():5.1f}%   vs prior-only {100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
    d = res["voi"] - res["greedy"]
    bs = [np.mean(d[np.random.default_rng(i).integers(0, len(d), len(d))]) for i in range(2000)]
    print(f"  structured VOI - greedy {100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
    d = res["voi"] - res["scalar_voi"]
    bs = [np.mean(d[np.random.default_rng(i).integers(0, len(d), len(d))]) for i in range(2000)]
    print(f"  structured VOI - scalar VOI {100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")


if __name__ == "__main__":
    main()
