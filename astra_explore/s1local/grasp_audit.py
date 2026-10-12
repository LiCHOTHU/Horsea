"""Controlled grasp test on the dex hand: how precise must the aim be, and how often does a well-aimed grasp still fail?
(Isaac, rendering off; same harness, hand and action interface as Astra.)

    python -m s1local.grasp_audit --out <dir> [--repeats 2] [--smoke]

Reference: Astra's successful right-hand grasp of can_fanta_1 in Unload-Cans attempt 2 (decisions 1-3): wrist at
(-0.134, -0.066, +0.128) m from the can centre, fingers (0.982, 0.188, 0.006), palm (-0.068, 0.383, -0.921), closure
[0.60 0.60 0.65 0.50]. Each trial: reset; open hand to 11.5 cm above the target pose; descend (16 steps); close (22
steps); lift 12.5 cm (18 steps); hold 15 steps. Target = can position after reset + reference offset + perturbation.

Trial families: ref (no perturbation); pos (one world axis shifted by +-0.5 / 1 / 2 / 4 cm; x = depth from the robot);
yaw (hand rotated about the vertical by +-5 / 10 / 20 / 30 deg); phys (mass 0.8 / 1.0 / 1.2 x friction 0.7 / 1.0 of the
can and the hand links); closure (finger_base = finger_tip = 0.30 ... 1.00, thumb unchanged).
Outcome: held (can >= 8 cm above its rest height and within 12 cm of the fingertips over the whole hold) / slipped (rose
>= 5 cm, not held) / pushed (moved > 2 cm before the lift, never rose 5 cm) / air (fingers reached >= 80 % of the
commanded closure) / touched (closure blocked by the can, no lift).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from s1local import env as E  # noqa: E402
from s1local import generic as G  # noqa: E402

TASK, OBJ = "Humanoid-Unload-Cans-v0", "can_fanta_1"
OFFSET = np.array([-0.134, -0.066, 0.128])
FINGERS = np.array([0.9821453682725526, 0.18802783017845204, 0.0060008881971846395])
PALM = np.array([-0.06772238413858925, 0.3831415076940597, -0.9212037037314321])
CLOSE = [0.60, 0.60, 0.65, 0.50]
OPEN_R = [0.0, 0.0, 0.15, 0.0]
CONDS = [(1.0, 1.0), (0.8, 0.7), (1.2, 1.0), (1.0, 0.7), (0.8, 1.0), (1.2, 0.7)]


def rz(v, deg):
    a = math.radians(deg)
    return np.array([math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1], v[2]])


def hand(move, pos=None, fingers=None, palm=None, closure=OPEN_R):
    d = {"move": bool(move), "finger_base": closure[0], "finger_tip": closure[1], "thumb_rotate": closure[2], "thumb_bend": closure[3]}
    if move:
        d.update(position_m=[float(x) for x in pos], fingers_dir=[float(x) for x in fingers], palm_dir=[float(x) for x in palm])
    return d


def trials(repeats):
    out = [("ref", {}, r) for r in range(3 * repeats)]
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        for cm in (-4, -2, -1, -0.5, 0.5, 1, 2, 4):
            out += [("pos", {"axis": ax, "cm": cm}, r) for r in range(repeats)]
    for deg in (-30, -20, -10, -5, 5, 10, 20, 30):
        out += [("yaw", {"deg": deg}, r) for r in range(repeats)]
    for ci in range(1, 6):
        out += [("phys", {"cond": ci, "mass": CONDS[ci][0], "friction": CONDS[ci][1]}, r) for r in range(repeats)]
    for f in (0.30, 0.45, 0.75, 0.90, 1.00):
        out += [("closure", {"fingers": f}, r) for r in range(repeats)]
    return out


def run_trial(T, phys, fam, p):
    phys.nominal()
    E.reset(T)
    if fam == "phys":
        phys.apply({"physics": [OBJ], "hand": "right"}, p["mass"], p["friction"])
    can = T.U.scene.rigid_objects[OBJ]
    p0 = can.data.root_state_w[0, :3].cpu().numpy().copy()
    off = OFFSET.copy()
    if fam == "pos":
        off[{"x": 0, "y": 1, "z": 2}[p["axis"]]] += p["cm"] / 100.0
    fingers, palm = FINGERS.copy(), PALM.copy()
    if fam == "yaw":
        fingers, palm = rz(fingers, p["deg"]), rz(palm, p["deg"])
    close = list(CLOSE)
    if fam == "closure":
        close[0] = close[1] = p["fingers"]
    target = p0 + off
    left = hand(False)
    seq = [({"left": left, "right": hand(True, target + [0, 0, 0.115], fingers, palm), "steps": 40}, "approach"),
           ({"left": left, "right": hand(True, target, fingers, palm), "steps": 16}, "descend"),
           ({"left": left, "right": hand(False, closure=close), "steps": 22}, "close"),
           ({"left": left, "right": hand(True, target + [0, 0, 0.125], fingers, palm, closure=close), "steps": 18}, "lift")]
    rec, pre_lift = {}, None
    for d, name in seq:
        E.act(T, d)
        if name == "close":
            pre_lift = can.data.root_state_w[0, :3].cpu().numpy().copy()
            rec["closure_measured"] = T._closure_measured("right")
            rec["index_contact_N"] = round(T._contact("right"), 2)
            rec["wrist_error_cm"] = round(float(np.linalg.norm(T._ee("right")[:3] - target)) * 100, 2)
    z_hold, near = [], []

    def mon(k):
        c = can.data.root_state_w[0, :3].cpu().numpy()
        z_hold.append(float(c[2] - p0[2]))
        near.append(float(np.linalg.norm(c - T._tips("right").mean(0))))
    E.hold_window(T, 15, monitor=mon)
    pmax = max(z_hold)
    held = min(z_hold) >= 0.08 and max(near) < 0.12
    moved = float(np.linalg.norm(pre_lift[:2] - p0[:2]))
    meas = max(rec["closure_measured"].get("finger_base", 0), rec["closure_measured"].get("finger_tip", 0))
    if held:
        out = "held"
    elif pmax >= 0.05:
        out = "slipped"
    elif moved > 0.02:
        out = "pushed"
    elif meas >= 0.8 * close[0]:
        out = "air"
    else:
        out = "touched"
    return dict(rec, outcome=out, can_start=p0.round(4).tolist(), moved_before_lift_cm=round(100 * moved, 2),
                sustained_lift_m=round(min(z_hold), 4), max_rise_m=round(pmax, 4), max_can_tip_m=round(max(near), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from isaac_env import IsaacDexTask
    T = IsaacDexTask(TASK, cfg_hook=E.no_render)
    phys = G.Physics(T)
    todo = [("ref", {}, 0), ("pos", {"axis": "x", "cm": 2}, 0)] if a.smoke else trials(a.repeats)
    done = set()
    rp = out / "trials.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            done.add((r["family"], json.dumps(r["params"], sort_keys=True), r["repeat"]))
    print(f"READY {time.time() - t0:.0f}s, {len(todo)} trials", flush=True)
    for fam, p, r in todo:
        key = (fam, json.dumps(p, sort_keys=True), r)
        if key in done:
            continue
        t1 = time.time()
        res = run_trial(T, phys, fam, p)
        row = {"family": fam, "params": p, "repeat": r, **res, "wall_s": round(time.time() - t1, 1)}
        with open(rp, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("TRIAL", fam, json.dumps(p), r, res["outcome"], res["sustained_lift_m"], res["wrist_error_cm"], flush=True)
    phys.nominal()
    print(f"RUN DONE {time.time() - t0:.0f}s", flush=True)
    T.close()


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
