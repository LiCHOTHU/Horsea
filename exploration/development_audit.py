"""Separate capability/perception audit; results never enter online memories."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .events import Event, Prefix
from .feedback import Thresholds
from .gates import implementation_hash
from .protocol import Protocol, atomic_json, digest


def audit(root):
    rows, labels = [], []
    complete = True
    for path in sorted(Path(root).rglob("observations.jsonl")):
        if not path.with_name("complete.json").exists():
            complete = False
        online = [json.loads(x) for x in path.read_text().splitlines()]
        truth = [json.loads(x) for x in path.with_name("evaluation_only.jsonl").read_text().splitlines()]
        if len(online) != len(truth):
            raise ValueError("Unpaired audit records")
        for a, b in zip(online, truth):
            if any(a[k] != b[k] for k in ("scene", "replicate", "attempt", "method")):
                raise ValueError("Misaligned audit records")
            if a["scene"] not in Protocol().dev:
                raise ValueError("Development audit accepts only development configurations")
            if b.get("backend") != "robotwin":
                raise ValueError("Synthetic outcomes cannot validate robot feedback")
        rows.extend(online)
        labels.extend(truth)
    if not rows:
        raise ValueError("No completed robot attempts to audit")
    rates, prefixes = defaultdict(list), defaultdict(list)
    errors, goal_confusion, contact_proxy = [], Counter(), Counter()
    missing_frames, frames_total, missing_truth = 0, 0, 0
    unknown_prefix, unknown_tail, tails, physical_steps = 0, 0, 0, 0
    modes, motion_signatures, reference_hashes = set(), set(), set()
    for online, truth in zip(rows, labels):
        event = Event.from_record(online["event"])
        rates[(online["scene"], event.option.grasp, event.option.mode)].append(float(truth["simulator_success"]))
        prefix = online["observations"]["prefix"]
        prefixes[(online["scene"], online["replicate"], online["attempt"], event.option.grasp,
                  online["execution_seed"])].append((online["prefix_fingerprint"], prefix["commands"]))
        reference_hashes.add(truth["reference_hash"])
        unknown_prefix += event.prefix == Prefix.UNKNOWN
        physical_steps += truth["physics_steps"]
        observed = prefix["frames"].copy()
        if event.tail is not None:
            tails += 1
            unknown_tail += event.tail.name == "UNKNOWN"
            tail = online["observations"]["tail"]
            observed += tail["frames"][1:]
            if truth["physics_steps"] > prefix["physics_steps"]:
                modes.add(event.option.mode)
                motion_signatures.add(digest([x["tracked"]["eef"] for x in tail["frames"]]))
        audited = truth.get("frames", [])
        if len(audited) != len(observed):
            missing_truth += 1
            continue
        for measured, actual in zip(observed, audited):
            angle, joint = measured["tracked"]["opening"], actual["joint"][0]
            frames_total += 1
            if angle is None:
                missing_frames += 1
            else:
                errors.append(abs(angle - joint))
                goal_confusion[f"observed_{int(angle >= Thresholds().goal_angle)}_actual_{int(actual['simulator_success'])}"] += 1
            if measured["phase"] == "verification":
                contact_proxy[f"{event.prefix.name.lower()}_contact_{int(actual['finger_object_contact'])}"] += 1
    per_scene = {}
    for scene in sorted({x[0] for x in rates}):
        options = {f"{r},{m}": {"success_rate": float(np.mean(v)), "attempts": len(v)}
                   for (s, r, m), v in rates.items() if s == scene}
        means = [v["success_rate"] for v in options.values()]
        per_scene[str(scene)] = {"options": options, "oracle": max(means), "worst": min(means),
                                 "headroom": max(means) > min(means)}
    identity_failures = 0
    max_command_delta = 0.
    for group in prefixes.values():
        if len({fp for fp, _ in group}) != 1:
            identity_failures += 1
        reference = group[0][1]
        for _, commands in group[1:]:
            if len(commands) != len(reference):
                identity_failures += 1
                continue
            for a, b in zip(reference, commands):
                if (a["action"], a.get("gripper")) != (b["action"], b.get("gripper")):
                    identity_failures += 1
                if a.get("pose") is not None and b.get("pose") is not None:
                    max_command_delta = max(max_command_delta, float(np.max(np.abs(np.subtract(a["pose"], b["pose"])))))
    abstention = missing_frames / max(frames_total, 1)
    feedback_pass = (bool(errors) and missing_truth == 0 and np.mean(errors) <= .04
                     and max(errors) <= .12 and abstention <= .1
                     and goal_confusion["observed_1_actual_0"] == 0)
    full_sweep = all(len(v["options"]) == 6 and all(o["attempts"] >= 3 for o in v["options"].values())
                     for v in per_scene.values())
    coverage_pass = full_sweep and all(v["oracle"] >= 2/3 for v in per_scene.values())
    identity_pass = identity_failures == 0 and max_command_delta <= 1e-5
    all_dev = set(map(int, per_scene)) == set(Protocol().dev)
    result = {"claim": "instrumented-scene diagnostic; fixed asset, variable placement",
              "attempts": len(rows), "complete": complete, "development_scenes": list(map(int, per_scene)),
              "per_scene": per_scene, "physics_steps": physical_steps,
              "opening_mean_absolute_error_rad": float(np.mean(errors)) if errors else None,
              "opening_max_absolute_error_rad": float(max(errors)) if errors else None,
              "frame_abstention_rate": abstention, "missing_audit_sequences": missing_truth,
              "prefix_abstention_rate": unknown_prefix / len(rows), "tail_abstention_rate": unknown_tail / max(tails, 1),
              "goal_confusion_vs_task_success": dict(goal_confusion),
              "readiness_vs_contact_proxy": dict(contact_proxy),
              "contact_note": "Contact presence is an imperfect audit proxy, not latent grasp readiness ground truth",
              "executed_tail_modes": sorted(modes), "distinct_tail_motion_traces": len(motion_signatures),
              "prefix_identity_violations": identity_failures, "prefix_max_command_delta": max_command_delta,
              "feedback_validated": bool(feedback_pass and all_dev),
              "local_feedback_check_passed": bool(feedback_pass),
              "option_capability_passed": bool(coverage_pass and all_dev),
              "prefix_identity_passed": bool(identity_pass and full_sweep),
              "thresholds": asdict(Thresholds()), "implementation_hash": implementation_hash(),
              "reference_hash": next(iter(reference_hashes)) if len(reference_hashes) == 1 else None,
              "confirmation_allowed": bool(complete and all_dev and feedback_pass and coverage_pass and identity_pass
                                           and len(modes) == 3 and len(reference_hashes) == 1
                                           and any(v["headroom"] for v in per_scene.values()))}
    atomic_json(Path(root) / "development_audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root)
