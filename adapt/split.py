"""Predefined, criteria-based YCB object split (section 2 of the plan).

Feasibility criteria (fixed BEFORE any method is run; test objects are never swapped for poor results):
  * graspable: smallest horizontal extent <= 0.065 m (Panda max opening 0.08 m, with margin)
  * compatible size / reachable: largest extent <= 0.20 m (object spawn region is 0.2 x 0.2 m)
  * not too flat: height >= 0.025 m (fingers must reach the object without hitting the table)
  * valid collision geometry: present in ManiSkill's curated info_pick_v0.json, minus the four objects the
    task itself excludes as non-graspable
Feasible objects are permuted with a fixed seed; the first 8 are source, the next 2 development, the next 3 test.
"""
import json
import os

import numpy as np

EXCLUDED_BY_TASK = ["022_windex_bottle", "028_skillet_lid", "029_plate", "059_chain"]
CRITERIA = dict(max_min_xy=0.065, max_extent=0.20, min_height=0.025, permutation_seed=0)
SIZES = dict(source=8, dev=2, test=3)


def info_path():
    from mani_skill import ASSET_DIR
    return os.path.join(ASSET_DIR, "assets/mani_skill2_ycb/info_pick_v0.json")


def bbox_extents(meta):
    s = meta.get("scales", [1.0])[0]
    mn, mx = meta["bbox"]["min"], meta["bbox"]["max"]
    return [float((mx[i] - mn[i]) * s) for i in range(3)]


def make_split():
    db = json.load(open(info_path()))
    feasible, rejected = [], {}
    for k in sorted(db):
        if k in EXCLUDED_BY_TASK:
            rejected[k] = "task-excluded"
            continue
        ext = bbox_extents(db[k])
        if min(ext[:2]) > CRITERIA["max_min_xy"]:
            rejected[k] = f"min_xy {min(ext[:2]):.3f} > {CRITERIA['max_min_xy']}"
        elif max(ext) > CRITERIA["max_extent"]:
            rejected[k] = f"max_extent {max(ext):.3f} > {CRITERIA['max_extent']}"
        elif ext[2] < CRITERIA["min_height"]:
            rejected[k] = f"height {ext[2]:.3f} < {CRITERIA['min_height']}"
        else:
            feasible.append(k)
    perm = np.random.RandomState(CRITERIA["permutation_seed"]).permutation(len(feasible))
    order = [feasible[i] for i in perm]
    n_s, n_d, n_t = SIZES["source"], SIZES["dev"], SIZES["test"]
    split = dict(
        criteria=CRITERIA, sizes=SIZES,
        source=sorted(order[:n_s]), dev=sorted(order[n_s:n_s + n_d]),
        test=sorted(order[n_s + n_d:n_s + n_d + n_t]),
        later_expansion=sorted(order[n_s + n_d + n_t:]),
        feasible=feasible, rejected=rejected,
        extents={k: bbox_extents(db[k]) for k in db},
    )
    return split


def load_split(path):
    return json.load(open(path))


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    s = make_split()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(s, open(out, "w"), indent=1)
    print("feasible:", len(s["feasible"]))
    for k in ("source", "dev", "test"):
        print(k, s[k])
    print("later_expansion:", s["later_expansion"])
