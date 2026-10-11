"""Task-generic version of the local-execution-adaptation pilot (same protocol as the Stack-Can-Into-Drawer pilot).

A task spec (specs.py) names the source Astra attempt and up to three operations; each operation declares
    kind      grasp | transport | push            (the parameter template)
    hand      left | right                          (the hand whose commands the parameters edit)
    object    rigid-object name, or [articulation, joint index] for push operations on an articulated part
    prefix    decision indices replayed (reference parameters) to rebuild the start, never teleported
    program   decision indices of the operation; transport: `release` = trailing indices kept unchanged
    window    evaluation hold / settle steps after the program
    success   grasp: sustained 8 cm lift with the object at the hand; transport: a region predicate of the task (from its
              own success code); push: a joint threshold of the task
    shift     objects displaced together by 8 mm (0 / 120 / 240 deg) at reset for the three fresh evaluation starts
    physics   rigid objects or [articulation, body] whose mass / inertia / friction the condition scales; the hand's links
              get the same friction factor (both bodies of the contact pair)
Parameter templates (at most four, bounds as in the first pilot):
    grasp     the hand's finger_base, finger_tip, thumb_rotate, thumb_bend in every program decision (reference +- 0.15)
    transport speed [0.5, 1.25] of the moving decisions; finger_base, finger_tip, thumb_bend held during them (+- 0.15);
              release decisions unchanged
    push      the hand's contact offsets along world x and z (+- 1 cm), push direction about world z (+- 10 deg), speed
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from s1local import env as E

HOLD, LIFT, NEAR = 15, 0.08, 0.12
NEAR_OPEN = 0.20      # grasp starts: the OPEN hand's fingertip mean is farther from the object than when closed (Unload-Cans: 0.13-0.14 m)
SHIFT_DIRS = {"e1": 0.0, "e2": 120.0, "e3": 240.0}
SHIFT_M = 0.008


def load_program(run_dir, attempt):
    out = []
    for d in sorted((Path(run_dir) / f"attempt_{attempt}").glob("decision_*")):
        r = json.loads((d / "decision.json").read_text())
        c = r.get("decision_content")
        if c and not c.get("stop"):
            out.append(c)
    return out


# ---------------------------------------------------------------------- parameters
def reference(op, prog):
    h = op["hand"]
    if op["kind"] == "grasp":
        x = prog[op["program"][0]][h]
        return np.array([x["finger_base"], x["finger_tip"], x["thumb_rotate"], x["thumb_bend"]], float)
    if op["kind"] == "transport":
        x = prog[op["program"][0]][h]
        return np.array([1.0, x["finger_base"], x["finger_tip"], x["thumb_bend"]], float)
    return np.array([0.0, 0.0, 0.0, 1.0], float)


def bounds(op, ref):
    if op["kind"] == "grasp":
        return np.clip(ref - 0.15, 0, 1), np.clip(ref + 0.15, 0, 1)
    if op["kind"] == "transport":
        return np.concatenate([[0.5], np.clip(ref[1:] - 0.15, 0, 1)]), np.concatenate([[1.25], np.clip(ref[1:] + 0.15, 0, 1)])
    return np.array([-0.01, -0.01, -10.0, 0.5]), np.array([0.01, 0.01, 10.0, 1.25])


def _rot_z(v, deg):
    a = math.radians(deg)
    return np.array([math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1], v[2]])


def edited_program(op, prog, theta, wrist0):
    """[(decision, speed)] with theta applied to the declared fields only (copies; the reference is never modified)."""
    th = [float(x) for x in theta]
    h = op["hand"]
    out = []
    if op["kind"] == "grasp":
        for i in op["program"]:
            d = copy.deepcopy(prog[i])
            d[h].update(finger_base=th[0], finger_tip=th[1], thumb_rotate=th[2], thumb_bend=th[3])
            out.append((d, 1.0))
        return out
    if op["kind"] == "transport":
        rel = set(op.get("release", []))
        for i in op["program"]:
            d = copy.deepcopy(prog[i])
            if i in rel:
                out.append((d, 1.0))
            else:
                d[h].update(finger_base=th[1], finger_tip=th[2], thumb_bend=th[3])
                out.append((d, th[0]))
        return out
    p0 = np.asarray(wrist0, float)
    for i in op["program"]:
        d = copy.deepcopy(prog[i])
        if d[h]["move"]:
            p = np.asarray(d[h]["position_m"], float)
            d[h]["position_m"] = (p0 + _rot_z(p - p0, th[2]) + np.array([th[0], 0.0, th[1]])).tolist()
        out.append((d, th[3]))
    return out


# ---------------------------------------------------------------------- state (evaluation only)
def object_state(T):
    U = T.U
    st = {"rigid": {}, "joints": {}, "tips": {s: T._tips(s).mean(0).tolist() for s in ("left", "right")},
          "env_success": bool(U.success[0].item())}
    for n, ob in U.scene.rigid_objects.items():
        r = ob.data.root_state_w[0].cpu().numpy()
        st["rigid"][n] = {"pos": r[:3].tolist(), "quat": r[3:7].tolist()}
    for n, ar in U.scene.articulations.items():
        if n == "robot":
            continue
        st["joints"][n] = {"q": ar.data.joint_pos[0].cpu().numpy().tolist(), "lim": ar.data.joint_pos_limits[0].cpu().numpy().tolist(),
                           "root": ar.data.root_state_w[0, :3].cpu().numpy().tolist()}
    return st


def track(T, op):
    """Per-step record used by the evaluator: tracked object position (or joint angle) and the hand's fingertip mean."""
    if isinstance(op["object"], list):
        art, j = op["object"]
        return [float(T.U.scene.articulations[art].data.joint_pos[0, j].item()), T._tips(op["hand"]).mean(0)]
    return [T.U.scene.rigid_objects[op["object"]].data.root_state_w[0, :3].cpu().numpy().copy(), T._tips(op["hand"]).mean(0)]


def upright(q):
    w, x, y, z = q
    return 1 - 2 * (x * x + y * y)                 # z component of the body's up axis in the world frame


def evaluate(op, start, trace, final, region_fns):
    out = {}
    if op["kind"] == "grasp":
        o = op["object"]
        z0 = start["rigid"][o]["pos"][2]
        win = trace[-HOLD:]
        gain = np.array([w[0][2] - z0 for w in win])
        near = np.array([np.linalg.norm(np.asarray(w[0]) - np.asarray(w[1])) for w in win])
        S = int(bool((gain >= LIFT).all() and (near < NEAR).all()))
        P = float(gain.min() / LIFT)
        p0, p1 = np.array(start["rigid"][o]["pos"]), np.array(final["rigid"][o]["pos"])
        D = int(bool(p1[2] < z0 - 0.05 or np.linalg.norm(p1[:2] - p0[:2]) > 0.25))
        out.update(sustained_lift_m=round(float(gain.min()), 4), max_hand_distance_m=round(float(near.max()), 4))
    elif op["kind"] == "transport":
        fn = region_fns[op["success"]]
        o = op["object"]
        ok, d0 = fn(start, o)
        ok1, d1 = fn(final, o)
        S = int(ok1)
        P = 1.0 - d1 / max(d0, 1e-6)
        D = int(bool(final["rigid"][o]["pos"][2] < 0.95))          # fell off the table (table surface ~1.0 m)
        out.update(target_distance_m=round(d1, 4))
    else:
        art, j = op["object"]
        q0, q1 = start["joints"][art]["q"][j], final["joints"][art]["q"][j]
        thr = op["success"]
        S = int(q1 >= thr)
        P = (q1 - q0) / max(thr - q0, 1e-6)
        r0, r1 = np.array(start["joints"][art]["root"]), np.array(final["joints"][art]["root"])
        D = int(bool(np.linalg.norm(r1 - r0) > 0.05))             # the articulated object itself displaced
        out.update(joint_end=round(q1, 4), joint_threshold=round(thr, 4))
    V = 0
    J = S + 0.25 * float(np.clip(P, -1, 1)) - D - V
    out.update(S=S, P=round(float(P), 4), D=D, V=V, J=round(J, 4))
    return out


def start_valid(op, st, rest_z):
    if op["kind"] in ("grasp", "transport"):
        o = op["object"]
        p = np.array(st["rigid"][o]["pos"])
        d = float(np.linalg.norm(np.array(st["tips"][op["hand"]]) - p))
        if op["kind"] == "grasp":
            ok = abs(p[2] - rest_z) < 0.02 and d < NEAR_OPEN
        else:
            ok = p[2] > rest_z + LIFT and d < NEAR
        return bool(ok), {"object_z": round(float(p[2]), 4), "hand_object_m": round(d, 4)}
    art, j = op["object"]
    return True, {"joint": round(st["joints"][art]["q"][j], 4)}


# ---------------------------------------------------------------------- physics conditions
class Physics:
    """Nominal values captured once; a condition scales mass + inertia and static / dynamic friction of the declared bodies
    and the friction of the acting hand's links."""

    def __init__(self, T):
        import torch
        self.T, self.ids = T, torch.tensor([0], dtype=torch.int32)
        U = T.U
        self.nom = {}
        for n, ob in U.scene.rigid_objects.items():
            v = ob.root_physx_view
            self.nom[("rigid", n)] = (v.get_masses().clone(), v.get_inertias().clone(), v.get_material_properties().clone())
        for n, ar in U.scene.articulations.items():
            v = ar.root_physx_view
            self.nom[("art", n)] = (v.get_masses().clone(), v.get_inertias().clone(), v.get_material_properties().clone())
        rob = U.robot
        self.hand_shapes = {s: E.shape_ids(rob, [b for b in rob.body_names if b.startswith(s[0].upper() + "_") and b.split("_", 1)[1].split("_")[0] in
                                                ("hand", "index", "middle", "ring", "pinky", "thumb")]) for s in ("left", "right")}

    def _view(self, key):
        U = self.T.U
        return (U.scene.rigid_objects[key[1]] if key[0] == "rigid" else U.scene.articulations[key[1]]).root_physx_view

    def nominal(self):
        for k, (m, I, mat) in self.nom.items():
            v = self._view(k)
            v.set_masses(m.clone(), self.ids)
            v.set_inertias(I.clone(), self.ids)
            v.set_material_properties(mat.clone(), self.ids)

    def apply(self, op, mass_f, fric_f):
        self.nominal()
        rec = {"mass_factor": mass_f, "friction_factor": fric_f, "changed": []}
        U = self.T.U
        for tgt in op["physics"]:
            if isinstance(tgt, str):
                key, body = ("rigid", tgt), None
            else:
                key, body = ("art", tgt[0]), tgt[1]
            m, I, mat = (x.clone() for x in self.nom[key])
            v = self._view(key)
            if body is None:
                m *= mass_f
                I *= mass_f
                mat[..., 0:2] *= fric_f
            else:
                art = U.scene.articulations[tgt[0]]
                if body == "@moving":                    # the link below the hinge of a two-link articulation (e.g. a laptop lid)
                    assert len(art.body_names) == 2, f"@moving needs a two-link articulation, {tgt[0]} has {art.body_names}"
                    body = art.body_names[1]
                bi = art.body_names.index(body)
                m[0, bi] *= mass_f
                I[0, bi] *= mass_f
                mat[0, E.shape_ids(art, [body]), 0:2] *= fric_f
            v.set_masses(m, self.ids)
            v.set_inertias(I, self.ids)
            v.set_material_properties(mat, self.ids)
            bidx = 0 if body is None else U.scene.articulations[tgt[0]].body_names.index(body)
            rec["changed"].append({"target": tgt, "body": body, "mass": [float(self.nom[key][0].flatten()[bidx]), float(m.flatten()[bidx])]})
        rm = self.nom[("art", "robot")][2].clone()
        rm[0, self.hand_shapes[op["hand"]], 0:2] *= fric_f
        self._view(("art", "robot")).set_material_properties(rm, self.ids)
        rec["hand_shapes_scaled"] = len(self.hand_shapes[op["hand"]])
        return rec


# ---------------------------------------------------------------------- fresh starts
def reset_with_shift(T, names, sid):
    """Env reset; for e1-e3 the named objects are displaced together by 8 mm at 0 / 120 / 240 deg before the settle steps."""
    T._reset_env()
    if sid in SHIFT_DIRS:
        a = math.radians(SHIFT_DIRS[sid])
        dx, dy = SHIFT_M * math.cos(a), SHIFT_M * math.sin(a)
        U = T.U
        for n in names:
            if n in U.scene.rigid_objects:
                ob = U.scene.rigid_objects[n]
                st = ob.data.root_state_w.clone()
                st[0, 0] += dx
                st[0, 1] += dy
                ob.write_root_state_to_sim(st)
            elif n in U.scene.articulations:
                ar = U.scene.articulations[n]
                st = ar.data.root_state_w.clone()
                st[0, 0] += dx
                st[0, 1] += dy
                ar.write_root_pose_to_sim(st[:, :7])
    T.steps_used, T._frames = 0, []
    T.hand_cmd = {s: E.OPEN for s in ("left", "right")}
    hold = T.robot.data.joint_pos.clone()
    T._run(T.settle_steps, hands={s: E.OPEN for s in ("left", "right")}, hold_q=hold)
