"""The three operation tests, their parameterizations of Astra's program and the state-based outcome evaluator.

Source program: pilot attempt 10 (11 decisions, environment success at step 298), used for all three operations.
    D0 approach (48 steps), D1 lower beside the can (16), D2 close [0.45 0.60 0.60 0.35] (20), D3 lift 19 cm (25),
    D4 carry over the saucer (38), D5 lower (14), D6 open = release (23), D7 withdraw right / left hand outside the
    open panel (40), D8-D10 left-palm push of the panel (36, 23, 25; D10 also withdraws the right hand).

Operations (start = state after a fixed prefix of the reference program; evaluation windows are holds of the final pose):
    grasp      start after D0-D1   program D2-D3 + 15-step hold (0.5 s)   params: the four right-hand closure fractions of D2-D3
    transport  start after D0-D3   program D4-D6 + 10-step settle        params: speed (D4-D5), finger_base / finger_tip /
                                                                                  thumb_bend held during D4-D5 (D6 release unchanged)
    door       start after D0-D7   program D8-D10 + 10-step settle       params: left-hand contact offsets along world x and z,
                                                                                  push-direction rotation about world z, speed (D8-D10)
No replanning, no extra grasp candidates, no extra degrees of freedom: each parameter edits the stated fields of the stated
decisions only; joint limits, the 1 cm / 3 deg per-step wrist interpolation and the 1/30 per-step closure ramp are untouched.
"""
import copy
import math

import numpy as np

OPS = {
    "grasp": {"prefix": [0, 1], "program": [2, 3], "settle": 15,
              "params": ["finger_base", "finger_tip", "thumb_rotate", "thumb_bend"]},
    "transport": {"prefix": [0, 1, 2, 3], "program": [4, 5, 6], "settle": 10,
                  "params": ["speed", "finger_base", "finger_tip", "thumb_bend"]},
    "door": {"prefix": [0, 1, 2, 3, 4, 5, 6, 7], "program": [8, 9, 10], "settle": 10,
             "params": ["contact_dx_m", "contact_dz_m", "push_dir_deg", "speed"]},
}
LIFT_M, HOLD_STEPS, WITH_HAND_M = 0.08, 15, 0.12      # 8 cm sustained lift, 0.5 s, the env's own reach_can radius (0.12 m)
PLATE_RADIUS, PLATE_DZ = 0.09, 0.02                    # the env's can_on_plate criterion
DOOR_CLOSED_FRAC = 0.8                                 # the env's success: bottom joint >= upper - 0.2 (upper - lower)


def reference(op, prog):
    if op == "grasp":
        r = prog[2]["right"]
        return np.array([r["finger_base"], r["finger_tip"], r["thumb_rotate"], r["thumb_bend"]], float)
    if op == "transport":
        r = prog[4]["right"]
        return np.array([1.0, r["finger_base"], r["finger_tip"], r["thumb_bend"]], float)
    return np.array([0.0, 0.0, 0.0, 1.0], float)


def bounds(op, ref):
    """Closure: reference +- 0.15 inside [0, 1]; speed [0.5, 1.25]; door contact offsets +- 1 cm; push direction +- 10 deg."""
    if op == "grasp":
        return np.clip(ref - 0.15, 0, 1), np.clip(ref + 0.15, 0, 1)
    if op == "transport":
        lo = np.concatenate([[0.5], np.clip(ref[1:] - 0.15, 0, 1)])
        hi = np.concatenate([[1.25], np.clip(ref[1:] + 0.15, 0, 1)])
        return lo, hi
    return np.array([-0.01, -0.01, -10.0, 0.5]), np.array([0.01, 0.01, 10.0, 1.25])


def rot_z(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def edited_program(op, prog, theta, left_wrist0=None):
    """[(decision, speed)] for the operation with parameters theta applied (copies; the reference is never modified)."""
    th = [float(x) for x in theta]
    if op == "grasp":
        out = []
        for i in OPS[op]["program"]:
            d = copy.deepcopy(prog[i])
            d["right"].update(finger_base=th[0], finger_tip=th[1], thumb_rotate=th[2], thumb_bend=th[3])
            out.append((d, 1.0))
        return out
    if op == "transport":
        out = []
        for i in (4, 5):
            d = copy.deepcopy(prog[i])
            d["right"].update(finger_base=th[1], finger_tip=th[2], thumb_bend=th[3])     # thumb_rotate stays at the reference
            out.append((d, th[0]))
        out.append((copy.deepcopy(prog[6]), 1.0))       # release: original open command and duration
        return out
    p0 = np.asarray(left_wrist0, float)
    out = []
    for i in OPS[op]["program"]:
        d = copy.deepcopy(prog[i])
        L = d["left"]
        if L["move"]:
            p = np.asarray(L["position_m"], float)
            L["position_m"] = (p0 + rot_z(p - p0, th[2]) + np.array([th[0], 0.0, th[1]])).tolist()
        out.append((d, th[3]))
    return out


# ---------------------------------------------------------------------- state-based outcome evaluation (never fed to actions)
def evaluate(op, start, trace, final):
    """start / final: object_state dicts at the operation start / end; trace: per-step [can_xyz, door_q, right_tips_mean_xyz]
    over the whole rollout (program + window). Returns the score components and J."""
    can0 = np.array(start["can_pos"])
    can1 = np.array(final["can_pos"])
    plate = np.array(final["plate_pos"])
    out = {}
    if op == "grasp":
        win = trace[-HOLD_STEPS:]
        gain = np.array([w[0][2] - can0[2] for w in win])
        with_hand = np.array([np.linalg.norm(np.array(w[0]) - np.array(w[2])) for w in win])
        S = int(bool((gain >= LIFT_M).all() and (with_hand < WITH_HAND_M).all()))
        P = float(gain.min() / LIFT_M)
        D = int(bool(can1[2] < can0[2] - 0.05 or np.linalg.norm(can1[:2] - can0[:2]) > 0.25))
        out.update(sustained_lift_m=round(float(gain.min()), 4), max_hand_distance_m=round(float(with_hand.max()), 4))
    elif op == "transport":
        d0 = float(np.linalg.norm(np.array(start["can_pos"][:2]) - np.array(start["plate_pos"][:2])))
        d1 = float(np.linalg.norm(can1[:2] - plate[:2]))
        on = d1 < PLATE_RADIUS and abs(can1[2] - plate[2]) < PLATE_DZ
        S = int(on)
        P = 1.0 - d1 / max(d0, 1e-6)
        D = int(bool(can1[2] < plate[2] - 0.10))
        out.update(can_plate_xy_m=round(d1, 4), can_plate_dz_m=round(float(can1[2] - plate[2]), 4))
    else:
        lo, hi = start["door_lo"], start["door_hi"]
        thr = hi - (1 - DOOR_CLOSED_FRAC) * (hi - lo)
        q0, q1 = start["door_q"], final["door_q"]
        S = int(q1 >= thr)
        P = (q1 - q0) / max(thr - q0, 1e-6)
        d1 = float(np.linalg.norm(can1[:2] - plate[:2]))
        D = int(not (d1 < PLATE_RADIUS and abs(can1[2] - plate[2]) < PLATE_DZ))   # the placed can knocked off the saucer
        out.update(door_q_end=round(q1, 4), door_threshold=round(thr, 4), can_plate_xy_m=round(d1, 4))
    V = 0          # no existing constraint-violation signal in this environment (documented in the manifest)
    J = S + 0.25 * float(np.clip(P, -1, 1)) - D - V
    out.update(S=S, P=round(float(P), 4), D=D, V=V, J=round(J, 4))
    return out


def start_valid(op, st, ref_state=None):
    """Is a reconstructed start a valid instance of the operation's starting condition?"""
    can, plate = np.array(st["can_pos"]), np.array(st["plate_pos"])
    tips = np.array(st["right_tips_mean"])
    if op == "grasp":                               # can resting on its support, right hand beside it
        ok = abs(can[2] - 1.02) < 0.02 and np.linalg.norm(tips - can) < WITH_HAND_M
        return bool(ok), {"can_z": round(float(can[2]), 4), "hand_can_m": round(float(np.linalg.norm(tips - can)), 4)}
    if op == "transport":                           # can lifted >= 8 cm and with the hand (a physical grasp, never teleported)
        ok = can[2] > 1.02 + LIFT_M and np.linalg.norm(tips - can) < WITH_HAND_M
        return bool(ok), {"can_z": round(float(can[2]), 4), "hand_can_m": round(float(np.linalg.norm(tips - can)), 4)}
    d = float(np.linalg.norm(can[:2] - plate[:2]))   # can placed on the saucer, door open
    ok = d < PLATE_RADIUS and abs(can[2] - plate[2]) < PLATE_DZ and st["door_q"] < 0.15
    return bool(ok), {"can_plate_xy_m": round(d, 4), "door_q": round(st["door_q"], 4)}
