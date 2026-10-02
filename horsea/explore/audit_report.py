"""Capability-and-feedback audit report for open_microwave (exploration plan of 2026-10-02, stage 1).

Ground truth (door angle, contacts, success) is read ONLY from the *.truth.json evaluation records; the observation
records (*.obs.json) hold what an online writer may use.

    python -m horsea.explore.audit_report [--raw experiments/explore/audit_microwave/raw]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

OPTIONS = [(0, "track"), (0, "pull"), (1, "track"), (1, "pull")]


def load(raw):
    A = []
    for p in sorted(glob.glob(os.path.join(raw, "*.obs.json"))):
        o = json.load(open(p))
        t = json.load(open(p.replace(".obs.json", ".truth.json")))
        A.append({"obs": o, "truth": t})
    return A


def final(t):
    return t["truth"]["records"][-1] if t["truth"]["records"] else {"door_frac": 0.0, "success": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/explore/audit_microwave/raw")
    a = ap.parse_args()
    A = load(a.raw)
    S = defaultdict(lambda: defaultdict(list))          # scene -> option -> [success]
    door = defaultdict(lambda: defaultdict(list))
    model, yaw = {}, {}
    for x in A:
        o = x["obs"]
        k = (o["grasp"], o["mode"])
        f = final(x)
        S[o["scene"]][k].append(float(f["success"]))
        door[o["scene"]][k].append(f["door_frac"])
        model[o["scene"]] = x["truth"]["model_id"]
        yaw[o["scene"]] = x["truth"].get("yaw_deg", 0.0)
    scenes = sorted(S)
    n_att = len(A)
    errs = [x for x in A if x["obs"]["error"]]
    print(f"audit: {n_att} attempts, {len(scenes)} scenes, executor errors {len(errs)}")
    print("\n1) CAPABILITY: success rate per option and scene (3 repeats; truth record, evaluation only)")
    print("| scene | model | yaw | " + " | ".join(f"r{r}+{m}" for r, m in OPTIONS) + " | best |\n|---|---|---|" + "---|" * (len(OPTIONS) + 1))
    best_rate, best_opt, default = [], [], {}
    for s in scenes:
        rates = {k: np.mean(S[s][k]) if S[s][k] else float("nan") for k in OPTIONS}
        b = max(OPTIONS, key=lambda k: (rates[k], np.mean(door[s][k]) if door[s][k] else 0))
        best_rate.append(rates[b]); best_opt.append(b)
        print(f"| {s} | {model[s]} | {yaw[s]:+.0f} | " + " | ".join(f"{100 * rates[k]:.0f}% ({np.mean(door[s][k]):.2f})" if S[s][k] else "--" for k in OPTIONS) +
              f" | r{b[0]}+{b[1]} |")
    mean_rate = {k: np.mean([np.mean(S[s][k]) for s in scenes if S[s][k]]) for k in OPTIONS}
    d = max(OPTIONS, key=lambda k: mean_rate[k])
    print("\nmean success per option: " + ", ".join(f"r{r}+{m} {100 * mean_rate[(r, m)]:.1f}%" for r, m in OPTIONS))
    reliable = np.mean([b >= 2 / 3 for b in best_rate])
    oracle = np.mean(best_rate)
    print(f"Q1 at least one option reliable (>= 2/3) in {100 * reliable:.0f}% of scenes; per-scene oracle success {100 * oracle:.1f}%")
    print(f"Q2 best default option r{d[0]}+{d[1]} {100 * mean_rate[d]:.1f}%; oracle - default = {100 * (oracle - mean_rate[d]):+.1f} points "
          f"(room for scene-specific selection); scenes where another option beats the default: "
          f"{sum(1 for s in scenes if max(np.mean(S[s][k]) for k in OPTIONS if S[s][k]) > np.mean(S[s][d]) + 1e-9)}/{len(scenes)}")
    by_model = defaultdict(list)
    for s in scenes:
        by_model[model[s]].append(s)
    for mid, ss in sorted(by_model.items()):
        print(f"   model {mid} ({len(ss)} scenes): " + ", ".join(f"r{r}+{m} {100 * np.mean([np.mean(S[s][(r, m)]) for s in ss if S[s][(r, m)]]):.0f}%"
                                                         for r, m in OPTIONS))
    # 2) prefix identity across modes: same (scene, grasp, repeat) -> prefix end-effector pose and gripper
    print("\n2) PREFIX IDENTITY across modes (same scene, grasp, repeat): end-effector position / gripper after prefix")
    P = defaultdict(dict)
    for x in A:
        o = x["obs"]
        pre = [r for r in o["records"] if r["tag"] == "prefix"]
        if pre:
            P[(o["scene"], o["grasp"], o["repeat"])][o["mode"]] = (np.array(pre[0]["ee"][:3]), pre[0]["gripper"])
    dpos = [np.linalg.norm(v["track"][0] - v["pull"][0]) for v in P.values() if len(v) == 2]
    dgrip = [abs(v["track"][1] - v["pull"][1]) for v in P.values() if len(v) == 2]
    if dpos:
        print(f"   pairs {len(dpos)}: |delta ee| median {1e3 * np.median(dpos):.2f} mm, max {1e3 * np.max(dpos):.2f} mm; "
              f"|delta gripper| median {np.median(dgrip):.4f}, max {np.max(dgrip):.4f}")
    # 3) distinct executed actions: tail end-effector displacement per option
    print("\n3) EXECUTED ACTIONS DIFFER: tail end-effector displacement (start of tail -> end of tail), metres")
    for k in OPTIONS:
        disp = []
        for x in A:
            o = x["obs"]
            if (o["grasp"], o["mode"]) != k:
                continue
            pre = [r for r in o["records"] if r["tag"] == "prefix"]
            tl = [r for r in o["records"] if r["tag"].startswith("tail")]
            if pre and tl:
                disp.append(np.array(tl[-1]["ee"][:3]) - np.array(pre[0]["ee"][:3]))
        if disp:
            D = np.array(disp)
            print(f"   r{k[0]}+{k[1]}: mean displacement {np.round(D.mean(0), 3).tolist()}, |d| {np.linalg.norm(D, axis=1).mean():.3f}")
    # 4) observation-only signals vs truth: gripper opening after prefix (holding) vs truth contact; door
    print("\n4) FEEDBACK SIGNALS (observation) vs TRUTH")
    rows = []
    for x in A:
        o, t = x["obs"], x["truth"]
        pre_o = [r for r in o["records"] if r["tag"] == "prefix"]
        pre_t = [r for r in t["records"] if r["tag"] == "prefix"]
        if not pre_o or not pre_t:
            continue
        door_contact = any(("link" in c[0] or "link" in c[1]) and not all(n in str(c) for n in ("left", "right")) for c in pre_t[0]["gripper_contacts"])
        rows.append((pre_o[0]["gripper"], len(pre_t[0]["gripper_contacts"]) > 0, final(x)["door_frac"], final(x)["success"], o["grasp"], o["mode"]))
    if rows:
        g = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
        print(f"   gripper opening after prefix: with truth contact {np.round(np.percentile(g[c], [5, 50, 95]), 3).tolist() if c.any() else '-'}; "
              f"without {np.round(np.percentile(g[~c], [5, 50, 95]), 3).tolist() if (~c).any() else '-'} (5/50/95th pct)")
    print("\nfinal door fraction per option (truth): " + ", ".join(
        f"r{r}+{m} " + str(np.round(np.percentile([final(x)['door_frac'] for x in A if (x['obs']['grasp'], x['obs']['mode']) == (r, m)], [10, 50, 90]), 2).tolist())
        for r, m in OPTIONS))


if __name__ == "__main__":
    main()
