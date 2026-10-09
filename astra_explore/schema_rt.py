"""Astra's strict response schema for the RoboTwin (dual-arm Aloha) pilot: absolute end-effector targets per arm."""
import math

from schema import LIMITS, Invalid

ARMS = ("left", "right")
TOPDOWN_WXYZ = (0.5, -0.5, 0.5, 0.5)      # verified: gripper pointing straight down, fingers opening along world x
# generous workspace box (m); outside -> rejected, never clipped. Table surface is around z 0.74; the robot can plan
# through the table, so the lower bound only excludes nonsense, not collisions.
BOX = {"x": (-0.7, 0.7), "y": (-0.55, 0.6), "z": (0.55, 1.45)}


def response_schema():
    num = {"type": "number"}
    arm = {"type": "object", "additionalProperties": False,
           "properties": {"move": {"type": "boolean"},
                          "position_m": {"type": "array", "items": num},
                          "quaternion_wxyz": {"type": "array", "items": num},
                          "gripper": num},
           "required": ["move", "position_m", "quaternion_wxyz", "gripper"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "left": arm, "right": arm,
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
            "required": ["left", "right", "stop", "assessment_of_previous_action", "intent", "hypothesis",
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


def validate_arm(a, name, warnings):
    if not isinstance(a, dict) or set(a) != {"move", "position_m", "quaternion_wxyz", "gripper"}:
        raise Invalid(f"{name} must have exactly move, position_m, quaternion_wxyz, gripper")
    if not isinstance(a["move"], bool):
        raise Invalid(f"{name}.move must be boolean")
    pos = _nums(a["position_m"], 3, f"{name}.position_m")
    quat = _nums(a["quaternion_wxyz"], 4, f"{name}.quaternion_wxyz")
    g = a["gripper"]
    if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g) or not 0.0 <= g <= 1.0:
        raise Invalid(f"{name}.gripper must be a number in [0, 1]")
    if a["move"]:
        for v, ax in zip(pos, "xyz"):
            lo, hi = BOX[ax]
            if not lo <= v <= hi:
                raise Invalid(f"{name}.position_m {ax}={v} outside the workspace box {BOX[ax]}; refusing to clip")
        n = math.sqrt(sum(q * q for q in quat))
        if not 0.9 <= n <= 1.1:
            raise Invalid(f"{name}.quaternion_wxyz norm {n:.3f} is not close to 1")
        if abs(n - 1.0) > 1e-6:
            warnings.append(f"{name}.quaternion normalised from norm {n:.4f}")
            quat = [q / n for q in quat]
    return {"move": a["move"], "position_m": pos, "quaternion_wxyz": quat, "gripper": float(g)}


def validate(resp, known_ids):
    """Returns (clean decision, warnings). Raises Invalid for anything that must not be executed."""
    if not isinstance(resp, dict):
        raise Invalid("response is not an object")
    req = set(response_schema()["required"])
    if set(resp) != req:
        raise Invalid(f"keys {sorted(set(resp) ^ req)} missing or unexpected")
    warnings = []
    arms = {n: validate_arm(resp[n], n, warnings) for n in ARMS}
    if not isinstance(resp["stop"], bool):
        raise Invalid("stop must be boolean")

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
    clean = {"left": arms["left"], "right": arms["right"], "stop": resp["stop"],
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
