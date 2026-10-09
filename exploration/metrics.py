"""Primary exploitation success from evaluation-only records, clustered by scene."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .events import Event
from .protocol import atomic_json


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def paired_interval(a, b, draws=5000):
    # Average paired seeds inside configurations; resample configurations, not trials.
    per = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        per[key[0]].append(a[key] - b[key])
    if not per:
        return None
    delta = np.asarray([np.mean(v) for v in per.values()])
    rng = np.random.default_rng(0)
    boot = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(1)
    return {"difference": float(delta.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "configurations": len(delta), "paired_replicates": sum(map(len, per.values()))}


def summarize(root):
    primary, proxy = defaultdict(dict), defaultdict(dict)
    counts, costs, selectors = defaultdict(int), defaultdict(list), defaultdict(list)
    paired_prefix = defaultdict(set)
    physics, durations, episodes = defaultdict(int), defaultdict(float), defaultdict(list)
    calibration = defaultdict(lambda: defaultdict(list))
    missing_truth = 0
    backend = set()
    for path in sorted(Path(root).rglob("observations.jsonl")):
        eval_path = path.with_name("evaluation_only.jsonl")
        truth = {}
        if eval_path.exists():
            for x in rows(eval_path):
                tkey = (x["scene"], x["replicate"], x["attempt"], x.get("method"))
                if "option" in x:
                    tkey += (x["option"]["grasp"], x["option"]["mode"])
                truth[tkey] = x
        for record in rows(path):
            method, phase = record["method"], record["phase"]
            event = Event.from_record(record["event"])
            key = (record["scene"], record["replicate"])
            tkey = (*key, record["attempt"], method)
            label = truth.get((*tkey, event.option.grasp, event.option.mode), truth.get(tkey, {}))
            backend.add(label.get("backend", "unspecified"))
            counts[method] += 1
            costs[method].append(record["prefix_steps"] + record["tail_steps"])
            physics[method] += label.get("physics_steps", 0)
            durations[method] += record.get("seconds", 0.)
            episodes[(method, *key)].append((record, label))
            memory = record.get("memory_before", {})
            if memory.get("kind") == "structured":
                q = np.array(memory["alpha"])[event.option.grasp]
                q /= q.sum()
                calibration[method]["prefix_brier"].append(float(np.sum((q - np.eye(4)[event.prefix]) ** 2)))
                if event.tail is not None:
                    p = np.array(memory["gamma"])[event.option.grasp, event.option.mode]
                    p /= p.sum()
                    calibration[method]["tail_brier"].append(float(np.sum((p - np.eye(6)[event.tail]) ** 2)))
            elif memory.get("kind") == "scalar":
                p = np.asarray(memory["counts"][record["decision"]["index"]], dtype=float)
                p /= p.sum()
                calibration[method]["utility_brier"].append(float(np.sum((p - np.eye(3)[int(2 * event.utility())]) ** 2)))
            if "decision" in record:
                selectors[method].append(record["decision"]["seconds"])
            # Streams intentionally differ across attempts. Compare the same stream only.
            pkey = (*key, record["attempt"], event.option.grasp, record.get("execution_seed"))
            paired_prefix[pkey].add(record["prefix_fingerprint"])
            if phase != "exploit":
                continue
            proxy[method][key] = float(event.utility() == 1.)
            if "simulator_success" not in label:
                missing_truth += 1
                continue
            if key in primary[method]:
                raise ValueError("Duplicate exploitation key; report active and common histories separately")
            primary[method][key] = float(label["simulator_success"])
    rates = lambda d: {k: {"mean": float(np.mean(list(v.values()))), "n": len(v)} for k, v in d.items()}
    target = primary.get("structured_voi", {})
    comparisons = {m: paired_interval(target, v) for m, v in primary.items() if m != "structured_voi"}
    cumulative, first_success, retries = defaultdict(list), defaultdict(list), defaultdict(list)
    # Common-history collection is shared across readers and counted once in
    # physical totals, but belongs to each reader's cumulative success history.
    for (method, scene, replicate), sequence in list(episodes.items()):
        if method == "shared_collection":
            for reader in ("prior", "scalar_voi", "structured_voi"):
                reader_sequence = episodes.get((reader, scene, replicate))
                if reader_sequence is not None:
                    reader_sequence.extend(sequence)
    for (method, _, _), sequence in episodes.items():
        sequence.sort(key=lambda pair: pair[0]["attempt"])
        if not any(x["phase"] == "exploit" for x, _ in sequence):
            continue
        if not all("simulator_success" in label for _, label in sequence):
            continue
        successful = [i+1 for i, (_, label) in enumerate(sequence) if label.get("simulator_success")]
        cumulative[method].append(bool(successful))
        if successful:
            first_success[method].append(successful[0])
        for (before, _), (after, _) in zip(sequence, sequence[1:]):
            e = Event.from_record(before["event"])
            if e.prefix.name == "NOT_READY":
                retries[method].append(after["event"]["grasp"] == e.option.grasp)
    criterion = all(comparisons.get(m) is not None and comparisons[m]["ci95"][0] > 0
                    for m in ("prior", "structured_greedy")) and missing_truth == 0
    return {"backends": sorted(backend), "primary_actual_exploitation_success": rates(primary),
            "observed_goal_proxy": rates(proxy), "missing_evaluation_labels": missing_truth,
            "structured_voi_minus": comparisons,
            "attempt_counts": dict(counts),
            "control_commands": {k: int(sum(v)) for k, v in costs.items()},
            "physical_simulation_steps": dict(physics), "attempt_wall_seconds": dict(durations),
            "cumulative_success_including_probes": {k: float(np.mean(v)) for k, v in cumulative.items()},
            "first_success_attempt_conditional_on_success": {k: float(np.mean(v)) for k, v in first_success.items()},
            "reuse_grasp_after_not_ready_rate": {k: float(np.mean(v)) for k, v in retries.items()},
            "observed_event_prediction_calibration": {
                k: {name: {"mean": float(np.mean(v)), "n": len(v)} for name, v in d.items()}
                for k, d in calibration.items()},
            "selector_mean_seconds": {k: float(np.mean(v)) for k, v in selectors.items()},
            "paired_prefix_violations": sum(len(v) > 1 for v in paired_prefix.values()),
            "paired_statistical_criterion_met": criterion,
            "exploration_pass": criterion and backend == {"robotwin"}
                                and all(len(v) == 1 for v in paired_prefix.values()),
            "scope": "Synthetic outcomes are mechanism checks, never a robot success claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.root)
    atomic_json(args.root / "metrics.json", result)
    print(json.dumps(result, indent=2))
