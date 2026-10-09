"""Reprocess saved RGB-D frames; compare afterward with segregated audit truth."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .protocol import atomic_json, digest
from .reference import load_sensors
from .vision import MarkerTracker, tool_point


def audit(root):
    reference = json.loads((root / "reference.json").read_text())
    tracker = MarkerTracker()
    # Tracking completes before any evaluation label is opened.
    for i, frame in enumerate(reference["frames"]):
        data = load_sensors(root / f"sensors_{i:03d}.npz")
        tracked = tracker.observe(data)
        frame.update(tracked=asdict(tracked), vision=tracker.last)
    reference["initial_body"] = reference["frames"][0]["vision"]["body"]
    grasp = reference["frames"][1]
    if grasp["vision"]["door"] is None:
        raise ValueError("Training grasp markers unavailable: reference cannot be calibrated")
    reference["handle_local"] = (np.linalg.inv(grasp["vision"]["door"]) @ np.r_[tool_point(grasp["left_eef"]), 1.])[:3].tolist()
    reference["perception"] = "RGB-D color components with temporal and rigid-triangle consistency"
    atomic_json(root / "reference_retracked.json", reference)
    truth = json.loads((root / "evaluation_only.json").read_text())["frames"]
    errors, comparisons = [], []
    for f, t in zip(reference["frames"], truth):
        angle = f["vision"]["opening"]
        error = None if angle is None else abs(angle - t["joint"])
        if error is not None:
            errors.append(error)
        comparisons.append({"phase": f["phase"], "camera_angle": angle, "audit_joint": t["joint"], "error_rad": error})
    result = {"source": "training reference sequence", "frames": comparisons,
              "abstention_rate": 1 - len(errors) / len(truth),
              "mean_absolute_error_rad": float(np.mean(errors)) if errors else None,
              "max_absolute_error_rad": float(max(errors)) if errors else None,
              "reference_hash": digest(reference), "feedback_validated": False,
              "confirmation_allowed": False,
              "note": "Training engineering check only; independent development audit still required"}
    atomic_json(root / "perception_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root)
