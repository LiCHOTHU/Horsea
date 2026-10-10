"""Astra's strict response schema for the Isaac Sim humanoid dex-hand pilot (Unitree H1 + two Inspire hands).

One decision = for EACH hand an absolute wrist (hand-base) target in the world frame: position_m plus the hand's
orientation given as two world directions (fingers_dir = where the straight fingers point, palm_dir = where the palm
faces; the harness makes palm_dir perpendicular to fingers_dir and converts to the IK quaternion), tracked by the
benchmark's own differential-IK controller for `steps` control steps (30 Hz), plus two closure fractions per hand
(finger_base, finger_tip, thumb_rotate, thumb_bend in [0, 1]: 0 = open, 1 = closed, applied as fractions of the Inspire hand's
joint ranges: finger knuckles, finger middle joints, thumb rotation toward the palm, thumb bending). Nothing is
clipped silently: out-of-range values are rejected and the decision is not executed.
"""
import math

from schema import LIMITS, Invalid

HANDS = ("left", "right")
GROUPS = ("finger_base", "finger_tip", "thumb_rotate", "thumb_bend")
HAND_KEYS = {"move", "position_m", "fingers_dir", "palm_dir", *GROUPS}
STEPS = (1, 150)                      # control steps per decision (30 Hz): 1/30 s .. 5 s
# generous workspace box (m) in the world frame (robot pelvis at the origin, table in front at +x); outside -> rejected
BOX = {"x": (-0.2, 1.0), "y": (-0.8, 0.8), "z": (0.7, 1.8)}
MIN_DIR_ANGLE_DEG = 30.0              # fingers_dir and palm_dir must not be (anti)parallel


def response_schema():
    num = {"type": "number"}
    vec = {"type": "array", "items": num}
    hand = {"type": "object", "additionalProperties": False,
            "properties": dict({"move": {"type": "boolean"}, "position_m": vec, "fingers_dir": vec, "palm_dir": vec}, **{g: num for g in GROUPS}),
            "required": ["move", "position_m", "fingers_dir", "palm_dir", *GROUPS]}
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "left": hand, "right": hand,
                "steps": {"type": "integer"},
                "stop": {"type": "boolean"},
                "assessment_of_previous_action": {"type": ["string", "null"]},
                "intent": {"type": "string"},
                "hypothesis": {"type": ["string", "null"]},
                "expected_change": {"type": "string"},
                "references": {"type": "array", "items": {"type": "integer"}},
                "memory_update": {"type": "object", "additionalProperties": False,
                                  "properties": {"observations": {"type": "array", "items": {"type": "string"}},
                                                 "hypotheses": {"type": "array", "items": {"type": "string"}},
                                                 "summary": {"type": "string"}},
                                  "required": ["observations", "hypotheses", "summary"]}},
            "required": ["left", "right", "steps", "stop", "assessment_of_previous_action", "intent", "hypothesis",
                         "expected_change", "references", "memory_update"]}


def _nums(v, n, name):
    if not isinstance(v, list) or len(v) != n:
        raise Invalid(f"{name} must be a list of {n} numbers")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise Invalid(f"{name} must contain finite numbers")
        out.append(float(x))
    return out


def _frac(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise Invalid(f"{name} must be a number in [0, 1]")
    return float(v)


def _unit(v, name):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-6:
        raise Invalid(f"{name} must be a non-zero direction")
    return [x / n for x in v]


def validate_hand(h, name, warnings):
    if not isinstance(h, dict) or set(h) != HAND_KEYS:
        raise Invalid(f"{name} must have exactly move, position_m, fingers_dir, palm_dir, {', '.join(GROUPS)}")
    if not isinstance(h["move"], bool):
        raise Invalid(f"{name}.move must be boolean")
    pos = _nums(h["position_m"], 3, f"{name}.position_m")
    fd = _nums(h["fingers_dir"], 3, f"{name}.fingers_dir")
    pd = _nums(h["palm_dir"], 3, f"{name}.palm_dir")
    closure = {g: _frac(h[g], f"{name}.{g}") for g in GROUPS}
    if h["move"]:
        for v, ax in zip(pos, "xyz"):
            lo, hi = BOX[ax]
            if not lo <= v <= hi:
                raise Invalid(f"{name}.position_m {ax}={v} outside the workspace box {BOX[ax]}; refusing to clip")
        fd, pd = _unit(fd, f"{name}.fingers_dir"), _unit(pd, f"{name}.palm_dir")
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(fd, pd))))
        ang = math.degrees(math.acos(dot))
        if ang < MIN_DIR_ANGLE_DEG or ang > 180.0 - MIN_DIR_ANGLE_DEG:
            raise Invalid(f"{name}: fingers_dir and palm_dir are {ang:.0f} deg apart; they must be roughly perpendicular (>= {MIN_DIR_ANGLE_DEG:.0f} deg)")
        if abs(ang - 90.0) > 5.0:
            warnings.append(f"{name}.palm_dir re-orthogonalised against fingers_dir (was {ang:.0f} deg apart)")
        pd = [b - dot * a for a, b in zip(fd, pd)]
        pd = _unit(pd, f"{name}.palm_dir")
    return dict({"move": h["move"], "position_m": pos, "fingers_dir": fd, "palm_dir": pd}, **closure)


def validate(resp, known_ids):
    """Returns (clean decision, warnings). Raises Invalid for anything that must not be executed."""
    if not isinstance(resp, dict):
        raise Invalid("response is not an object")
    req = set(response_schema()["required"])
    if set(resp) != req:
        raise Invalid(f"keys {sorted(set(resp) ^ req)} missing or unexpected")
    warnings = []
    hands = {n: validate_hand(resp[n], n, warnings) for n in HANDS}
    if not isinstance(resp["stop"], bool):
        raise Invalid("stop must be boolean")
    steps = resp["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not STEPS[0] <= steps <= STEPS[1]:
        raise Invalid(f"steps must be an integer in [{STEPS[0]}, {STEPS[1]}]; refusing to clip")

    def text(v, name, limit, nullable=False):
        if v is None and nullable:
            return None
        if not isinstance(v, str):
            raise Invalid(f"{name} must be a string")
        if len(v) > limit:
            warnings.append(f"{name} truncated from {len(v)} to {limit} chars")
            v = v[:limit]
        return v
    mu = resp["memory_update"]
    if not isinstance(mu, dict) or set(mu) != {"observations", "hypotheses", "summary"}:
        raise Invalid("memory_update must have observations, hypotheses, summary")
    for key in ("observations", "hypotheses"):
        if not isinstance(mu[key], list) or not all(isinstance(s, str) for s in mu[key]):
            raise Invalid(f"memory_update.{key} must be a list of strings")
    refs = resp["references"]
    if not isinstance(refs, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in refs):
        raise Invalid("references must be a list of integers")
    unknown = [i for i in refs if i not in known_ids]
    if unknown:
        warnings.append(f"references to unknown interaction ids {unknown} dropped")
    refs = [i for i in refs if i in known_ids][:LIMITS["references"]]

    def items(lst, key):
        if len(lst) > LIMITS[key]:
            warnings.append(f"memory_update.{key} truncated from {len(lst)} to {LIMITS[key]} items")
        return [text(s, f"memory_update.{key}[{i}]", LIMITS["item"]) for i, s in enumerate(lst[:LIMITS[key]])]
    clean = {"left": hands["left"], "right": hands["right"], "steps": int(steps), "stop": resp["stop"],
             "assessment_of_previous_action": text(resp["assessment_of_previous_action"], "assessment_of_previous_action",
                                                   LIMITS["assessment"], nullable=True),
             "intent": text(resp["intent"], "intent", LIMITS["intent"]),
             "hypothesis": text(resp["hypothesis"], "hypothesis", LIMITS["hypothesis"], nullable=True),
             "expected_change": text(resp["expected_change"], "expected_change", LIMITS["expected_change"]),
             "references": refs,
             "memory_update": {"observations": items(mu["observations"], "observations"),
                               "hypotheses": items(mu["hypotheses"], "hypotheses"),
                               "summary": text(mu["summary"], "memory_update.summary", LIMITS["summary"])}}
    return clean, warnings
