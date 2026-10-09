"""Reject confirmation runs without a frozen, matching development audit."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .feedback import Thresholds
from .protocol import Protocol, digest


def implementation_hash():
    root = Path(__file__).parent
    files = sorted(root.glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def validate_gate(path, reference):
    if path is None:
        raise ValueError("Real confirmation requires --gate with a frozen development audit")
    gate = json.loads(Path(path).read_text())
    for flag in ("feedback_validated", "option_capability_passed", "prefix_identity_passed", "confirmation_allowed"):
        if gate.get(flag) is not True:
            raise ValueError(f"Development gate has not passed: {flag}")
    if set(gate.get("development_scenes", [])) != set(Protocol().dev):
        raise ValueError("Gate must cover the predeclared development scenes")
    if gate.get("reference_hash") != digest(json.loads(Path(reference).read_text())):
        raise ValueError("Reference changed after development gate")
    if gate.get("implementation_hash") != implementation_hash():
        raise ValueError("Implementation changed after development gate")
    if gate.get("thresholds") != asdict(Thresholds()):
        raise ValueError("Feedback thresholds changed after development gate")
    return gate
