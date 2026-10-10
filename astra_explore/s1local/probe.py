"""Setup probe (not part of the formal run): throughput without rendering, replay of Astra's attempt-10 program,
snapshot / restore fidelity and determinism at the three operation starts, PhysX mass / material views."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from s1local import env as E  # noqa: E402

OUT = {}
t0 = time.time()
T = E.build(render=False)
OUT["startup_s"] = round(time.time() - t0, 1)
pilot_cal = json.loads((E.PILOT / "config.json").read_text())["task"]["calibration"]
OUT["calibration_max_abs_diff"] = max(abs(a - b) for s in ("left", "right") for k in ("f_b", "p_b")
                                      for a, b in zip(T.calibration[s][k], pilot_cal[s][k]))
print("startup", OUT["startup_s"], "s; calibration diff vs pilot", OUT["calibration_max_abs_diff"], flush=True)

# recorder on every control step
TRACE = []
_step0 = T._step


def _step(a):
    o = _step0(a)
    TRACE.append((T.U.can.data.root_state_w[0, :3].cpu().numpy().copy(), float(T.U.drawer.data.joint_pos[0, T.U.drawer_bottom_idx].item()),
                  T._ee("right")[:3].copy(), T._ee("left")[:3].copy()))
    return o


T._step = _step

# throughput
E.reset(T)
TRACE.clear()
t0 = time.time()
E.hold_window(T, 120)
OUT["steps_per_s_hold"] = round(120 / (time.time() - t0), 1)
print("throughput (hold, no render):", OUT["steps_per_s_hold"], "steps/s", flush=True)

prog = E.load_program(10)
print("program decisions:", len(prog), [d["steps"] for d in prog], flush=True)

# full replay of attempt 10
E.reset(T)
t0 = time.time()
rows = []
for i, d in enumerate(prog):
    n = E.act(T, d)
    st = E.object_state(T)
    rows.append({"d": i, "steps": n, "can": [round(x, 3) for x in st["can_pos"]], "door_q": round(st["door_q"], 3),
                 "can_plate_xy_cm": round(100 * float(np.linalg.norm(np.array(st["can_pos"][:2]) - np.array(st["plate_pos"][:2]))), 1),
                 "env_success": st["env_success"]})
    print(rows[-1], flush=True)
OUT["replay"] = rows
OUT["replay_s"] = round(time.time() - t0, 1)
OUT["door_limits"] = [E.object_state(T)["door_lo"], E.object_state(T)["door_hi"]]


def run_op(decs, settle=15):
    TRACE.clear()
    t = time.time()
    n = sum(E.act(T, d) for d in decs) + E.hold_window(T, settle)
    can = np.array([x[0] for x in TRACE]); door = np.array([x[1] for x in TRACE]); rw = np.array([x[2] for x in TRACE]); lw = np.array([x[3] for x in TRACE])
    return {"steps": n, "s": time.time() - t, "can": can, "door": door, "rw": rw, "lw": lw, "final": E.object_state(T)}


def compare(a, b):
    m = min(len(a["can"]), len(b["can"]))
    return {"can_max_dev_mm": round(1000 * float(np.abs(a["can"][:m] - b["can"][:m]).max()), 3),
            "door_max_dev_rad": round(float(np.abs(a["door"][:m] - b["door"][:m]).max()), 5),
            "rwrist_max_dev_mm": round(1000 * float(np.abs(a["rw"][:m] - b["rw"][:m]).max()), 3),
            "lwrist_max_dev_mm": round(1000 * float(np.abs(a["lw"][:m] - b["lw"][:m]).max()), 3)}


ops = {"grasp": (2, [2, 3]), "transport": (4, [4, 5, 6]), "door": (8, [8, 9, 10])}
OUT["restore"] = {}
for name, (k, idx) in ops.items():
    E.reset(T)
    for d in prog[:k]:
        E.act(T, d)
    snap = E.snapshot(T)
    r1 = run_op([prog[i] for i in idx])                     # continuous (never restored)
    E.restore(T, snap)
    r2 = run_op([prog[i] for i in idx])
    E.restore(T, snap)
    r3 = run_op([prog[i] for i in idx])
    res = {"steps": r1["steps"], "rollout_s": round(r2["s"], 2), "continuous_vs_restored": compare(r1, r2), "restored_vs_restored": compare(r2, r3),
           "final_continuous": {k2: r1["final"][k2] for k2 in ("can_pos", "door_q", "env_success")},
           "final_restored": {k2: r2["final"][k2] for k2 in ("can_pos", "door_q", "env_success")}}
    OUT["restore"][name] = res
    print(name, json.dumps(res), flush=True)

# PhysX views: masses / materials
U = T.U
for nm, a in (("can", U.can), ("plate", U.plate), ("drawer", U.drawer), ("robot", U.robot)):
    v = a.root_physx_view
    info = {"masses_shape": list(v.get_masses().shape), "materials_shape": list(v.get_material_properties().shape)}
    if nm in ("can", "plate"):
        info["mass"] = v.get_masses().tolist()
        info["material"] = v.get_material_properties()[0].tolist()
    if nm == "drawer":
        info["bodies"] = a.body_names
        info["masses"] = v.get_masses()[0].tolist()
        info["joints"] = a.joint_names
        info["material_first"] = v.get_material_properties()[0][:4].tolist()
        info["max_shapes"] = int(v.max_shapes) if hasattr(v, "max_shapes") else None
    if nm == "robot":
        info["n_bodies"] = len(a.body_names)
        info["hand_bodies_R"] = [b for b in a.body_names if b.startswith("R_")]
        info["hand_bodies_L"] = [b for b in a.body_names if b.startswith("L_")]
        info["max_shapes"] = int(v.max_shapes) if hasattr(v, "max_shapes") else None
    OUT[f"physx_{nm}"] = info
    print(nm, json.dumps(info)[:900], flush=True)
json.dump(OUT, open(sys.argv[1], "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("PROBE DONE", flush=True)
T.close()
