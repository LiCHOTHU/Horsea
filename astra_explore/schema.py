"""Astra's strict response schema and validation (no silent clipping)."""
import math

ACTION_KEYS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
MAX_REPEAT = 20
LIMITS = {"intent": 400, "hypothesis": 400, "expected_change": 400, "assessment": 600, "summary": 1200, "item": 300,
          "observations": 8, "hypotheses": 4, "references": 12}


def response_schema():
    num = {"type": "number"}
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "action": {"type": "object", "additionalProperties": False,
                           "properties": {k: num for k in ACTION_KEYS}, "required": ACTION_KEYS},
                "repeat": {"type": "integer"},
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
            "required": ["action", "repeat", "stop", "assessment_of_previous_action", "intent", "hypothesis",
                         "expected_change", "references", "memory_update"]}


class Invalid(ValueError):
    pass


def _num(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise Invalid(f"{name} must be a finite number")
    if x < -1.0 or x > 1.0:
        raise Invalid(f"{name}={x} outside [-1, 1]; refusing to clip")
    return float(x)


def validate(resp, known_ids):
    """Returns (clean decision, warnings). Raises Invalid for anything that must not be executed."""
    if not isinstance(resp, dict):
        raise Invalid("response is not an object")
    req = set(response_schema()["required"])
    if set(resp) != req:
        raise Invalid(f"keys {sorted(set(resp) ^ req)} missing or unexpected")
    a = resp["action"]
    if not isinstance(a, dict) or set(a) != set(ACTION_KEYS):
        raise Invalid("action must have exactly dx,dy,dz,drx,dry,drz,gripper")
    action = {k: _num(a[k], k) for k in ACTION_KEYS}
    rep = resp["repeat"]
    if isinstance(rep, bool) or not isinstance(rep, int) or not 1 <= rep <= MAX_REPEAT:
        raise Invalid(f"repeat={rep!r} must be an integer in [1, {MAX_REPEAT}]")
    if not isinstance(resp["stop"], bool):
        raise Invalid("stop must be boolean")
    warnings = []

    def text(v, name, limit, nullable=False):
        nonlocal warnings
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
        nonlocal warnings
        if len(lst) > LIMITS[key]:
            warnings.append(f"memory_update.{key} truncated from {len(lst)} to {LIMITS[key]} items")
        return [text(s, f"memory_update.{key}[{i}]", LIMITS["item"]) for i, s in enumerate(lst[:LIMITS[key]])]
    clean = {"action": action, "repeat": rep, "stop": resp["stop"],
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
