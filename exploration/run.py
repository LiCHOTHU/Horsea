"""Experiment drivers: capability sweep, active-history ladder, common-history screen.

    python -m exploration.run sweep  --split train        --out DIR
    python -m exploration.run active --split confirmation --out DIR
    python -m exploration.run common --split confirmation --out DIR

`--backend synthetic` validates the pipeline without a robot. `--backend robotwin`
executes the fixed staged skill with measured RGB-D/proprioceptive feedback.

The sweep is development-only capability diagnosis: it reports, per scene, which options reach the
goal. It must never be consulted online, so it writes to its own directory and the active/common
drivers never read it.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .events import Event, Option, Prefix
from .memory import StructuredMemory
from .protocol import METHODS, Protocol, atomic_json
from .synthetic import SyntheticBackend
from .trials import TrialManager


def make_backend(args, protocol):
    name = args.backend
    if name == "synthetic":
        return SyntheticBackend(protocol, unknown_rate=args.unknown_rate)
    if name == "robotwin":
        from .robotwin_backend import RobotwinBackend
        return RobotwinBackend(protocol, args.reference, args.out)
    raise ValueError(name)


def split_of(protocol, name):
    return {"train": protocol.train, "dev": protocol.dev, "confirmation": protocol.confirmation}[name]


def scenes_of(args, protocol):
    allowed = split_of(protocol, args.split)
    scenes = args.scenes or allowed
    if not scenes or any(s not in allowed for s in scenes) or len(scenes) != len(set(scenes)):
        raise ValueError("Requested scenes must be unique members of the declared split")
    return scenes


# ---------------------------------------------------------------------------------------
def sweep(args, protocol):
    """Exhaustive option sweep: capability, headroom and the oracle ceiling. Dev-only."""
    if args.split == "confirmation":
        raise ValueError("Exhaustive sweeps are forbidden on confirmation configurations")
    out = Path(args.out)
    manager = TrialManager(make_backend(args, protocol), protocol)
    rows, truths = [], []
    for scene in scenes_of(args, protocol):
        for r in range(protocol.grasps):
            for m in range(protocol.modes):
                for rep in (args.replicates or range(protocol.train_repeats)):
                    event, rec, truth = manager.attempt(scene, rep, 0, Option(r, m), "sweep")
                    rec.update(method="sweep", phase="sweep")
                    truth.update(method="sweep", phase="sweep")
                    rows.append(rec)
                    truths.append(truth)
                    TrialManager.save(out, rows, truths, complete=False)
                    print("ATTEMPT", scene, rep, r, m, event.record(),
                          "evaluation_success", truth.get("simulator_success"), flush=True)
    TrialManager.save(out, rows, truths)
    atomic_json(out / "summary.json", {"observed_goal_proxy": sweep_summary(rows, protocol),
        "actual_success": sweep_summary(rows, protocol, truths),
        "claim": "instrumented-scene diagnostic" if args.backend == "robotwin" else "synthetic mechanism check"})
    return rows


def sweep_summary(rows, protocol, truths=None):
    """Per-scene per-option goal rate -> oracle ceiling, prior-mean floor, headroom."""
    per = {}
    if truths is not None and len(truths) != len(rows):
        raise ValueError("Missing evaluation rows")
    for i, rec in enumerate(rows):
        e = Event.from_record(rec["event"])
        key = (rec["scene"], e.option.grasp, e.option.mode)
        result = e.utility() >= 1.0 if truths is None else truths[i]["simulator_success"]
        per.setdefault(key, []).append(float(result))
    scenes = sorted({k[0] for k in per})
    out = {"scenes": {}, "n_options": protocol.grasps * protocol.modes}
    oracle, mean = [], []
    for s in scenes:
        rates = {f"{r},{m}": float(np.mean(per[(s, r, m)]))
                 for r in range(protocol.grasps) for m in range(protocol.modes) if (s, r, m) in per}
        best, avg = max(rates.values()), float(np.mean(list(rates.values())))
        # Headroom: at least one option works and not all of them do.
        out["scenes"][str(s)] = {"rates": rates, "oracle": best, "option_mean": avg,
                                 "headroom": bool(best > 0.0 and avg < best)}
        oracle.append(best)
        mean.append(avg)
    out["oracle_ceiling"] = float(np.mean(oracle)) if oracle else float("nan")
    out["option_mean_floor"] = float(np.mean(mean)) if mean else float("nan")
    out["gap"] = out["oracle_ceiling"] - out["option_mean_floor"]
    out["scenes_with_headroom"] = int(sum(v["headroom"] for v in out["scenes"].values()))
    return out


# ---------------------------------------------------------------------------------------
def load_prior(path, protocol):
    """Long memory: weak pseudocounts fitted from training-split event records."""
    if path is None:
        return StructuredMemory(protocol.grasps, protocol.modes)
    events = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("scene") not in protocol.train:
                raise ValueError("Priors may use training configurations only")
            events.append(Event.from_record(record["event"]))
    return StructuredMemory.fit_prior(events, protocol.grasps, protocol.modes, protocol.kappa)


def active(args, protocol):
    """Each method chooses its own probes, then one exploitation attempt. Per scene and method."""
    out = Path(args.out)
    prior = load_prior(args.prior, protocol)
    atomic_json(out / "prior.json", prior.snapshot())
    manager = TrialManager(make_backend(args, protocol), protocol)
    methods = args.methods or list(METHODS)
    for scene in scenes_of(args, protocol):
        for replicate in (args.replicates or protocol.rollout_seeds):
            for method in methods:
                manager.active(method, prior, scene, replicate,
                               out / f"scene{scene}" / f"rep{replicate}" / method)
    atomic_json(out / "complete.json", {"split": args.split, "methods": methods})


def common(args, protocol):
    """One shared probe history per scene, replayed into three readers."""
    out = Path(args.out)
    prior = load_prior(args.prior, protocol)
    atomic_json(out / "prior.json", prior.snapshot())
    manager = TrialManager(make_backend(args, protocol), protocol)
    for scene in scenes_of(args, protocol):
        manager.common_history(prior, scene, out / f"scene{scene}")
    atomic_json(out / "complete.json", {"split": args.split})


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sweep", "active", "common"])
    ap.add_argument("--split", default="train", choices=["train", "dev", "confirmation"])
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "robotwin"])
    ap.add_argument("--reference", type=Path, help="Successful training-side marker reference.json")
    ap.add_argument("--scenes", type=int, nargs="+", help="Optional pilot/shard scenes within declared split")
    ap.add_argument("--replicates", type=int, nargs="+", choices=[0, 1, 2])
    ap.add_argument("--gate", type=Path, help="Frozen development audit required for real confirmation")
    ap.add_argument("--scalar-concentration", type=float, choices=[1., 3., 6.], default=1.,
                    help="Declared development-only strength search; freeze before confirmation")
    ap.add_argument("--prior", default=None, help="observations.jsonl from a train-split sweep")
    ap.add_argument("--methods", nargs="*", default=None, choices=list(METHODS))
    ap.add_argument("--unknown_rate", type=float, default=0.0,
                    help="synthetic only: fraction of attempts with degraded tracking")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())
    args.reference = None if args.reference is None else args.reference.resolve()
    args.prior = None if args.prior is None else str(Path(args.prior).resolve())
    if args.backend == "robotwin" and args.reference is None:
        ap.error("--backend robotwin requires --reference")
    if args.backend == "robotwin" and args.split == "confirmation":
        from .gates import validate_gate
        validate_gate(args.gate, args.reference)
    protocol = replace(Protocol(), scalar_concentration=args.scalar_concentration)
    scenes_of(args, protocol)
    atomic_json(Path(args.out) / "protocol.json", dict(protocol.record(), backend=args.backend,
        split=args.split, scenes=list(scenes_of(args, protocol)),
        replicates=args.replicates or list(protocol.rollout_seeds),
        reference=None if args.reference is None else str(args.reference),
        metric_note="Actual simulator success is evaluation-only; observed goal is a separate proxy"))
    {"sweep": sweep, "active": active, "common": common}[args.command](args, protocol)
    print(f"{args.command} complete -> {args.out}")


if __name__ == "__main__":
    main()
