"""E2: audit the feedback extractor against labelled frame sequences and freeze its thresholds.

    python -m exploration.calibrate --out DIR

The extractor turns tracked frames into prefix/tail events. Everything downstream assumes those
events are right, so this measures them against sequences whose *intended* event is known by
construction, and reports a confusion matrix plus abstention rate.

`feedback_validated` is computed here, not asserted: it is True only if every intended event is
recovered at least `--min-recall` of the time and abstention stays under `--max-abstention`. The
single distinction the whole method rests on is NOT_READY (it slipped) versus READY+AWAY (it held
but moved the wrong way) -- a flat scalar cannot tell those apart, so if the extractor cannot
either, the structured arm has nothing to work with. That pair is checked separately and reported
as `critical_pair_separated`.

Labels here come from how each sequence is generated. On real RoboTwin clips they must come from
hand annotation; this module's job is to be the same measurement either way.
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from .events import Prefix, Tail
from .feedback import Feedback, Thresholds, TrackedFrame
from .protocol import atomic_json


def frame(eef, gripper, handle, opening, confidence=.95):
    return TrackedFrame(tuple(eef), gripper, None if handle is None else tuple(handle),
                        opening, confidence)


def prefix_case(kind, rng):
    """Build (approach, verification) whose intended Prefix event is `kind`."""
    base = np.array([.30, -.10, .90]) + rng.normal(0, .002, 3)
    conf = .4 if kind == "unknown_occluded" else .95
    approach = [frame(base - [.05, 0, 0], 1.0, base, None, conf)]
    verification = []
    for i in range(3):
        eef = base + [.004 * (i + 1), 0, 0]
        if kind in ("ready", "goal"):
            handle = eef                                  # relation held
        elif kind == "not_ready_slip":
            handle = base + [0, 0, .08]                   # handle far from gripper
        else:
            handle = eef
        opening = 1.3 if kind == "goal" else None
        verification.append(frame(eef, .25, handle, opening, conf))
    if kind == "unknown_still":                           # closed, in place, but never moved
        verification = [frame(base, .25, base, None, conf) for _ in range(3)]
    if kind == "not_ready_open":                          # never closed the gripper
        verification = [frame(base + [.004 * (i + 1), 0, 0], 1.0, base + [.004 * (i + 1), 0, 0],
                              None, conf) for i in range(3)]
    return approach, verification


PREFIX_INTENT = {"goal": Prefix.GOAL, "ready": Prefix.READY,
                 "not_ready_slip": Prefix.NOT_READY, "not_ready_open": Prefix.NOT_READY,
                 "unknown_occluded": Prefix.UNKNOWN, "unknown_still": Prefix.UNKNOWN}


def tail_case(kind, rng):
    base = np.array([.31, -.10, .90]) + rng.normal(0, .002, 3)
    frames = []
    for i in range(4):
        eef = base + [.01 * (i + 1), 0, 0]
        handle, opening, conf = eef, 0.0, .95
        if kind == "goal":
            opening = 1.3 * (i + 1) / 4
        elif kind == "toward":
            opening = .30 * (i + 1) / 4
        elif kind == "away":
            opening = .6 - .30 * (i + 1) / 4
        elif kind == "no_response":
            opening = .30                                 # constant: below progress threshold
        elif kind == "slip":
            handle = base if i >= 2 else eef              # relation lost mid-motion
        elif kind == "unknown_occluded":
            conf = .4
        elif kind == "unknown_no_opening":
            opening = None
        frames.append(frame(eef, .25, handle, opening, conf))
    if kind == "away":                                    # needs a positive start to fall from
        frames = [frame(base + [.01 * (i + 1), 0, 0], .25, base + [.01 * (i + 1), 0, 0],
                        .70 - .30 * (i + 1) / 4, .95) for i in range(4)]
    return frames


TAIL_INTENT = {"goal": Tail.GOAL, "toward": Tail.TOWARD, "away": Tail.AWAY,
               "no_response": Tail.NO_RESPONSE, "slip": Tail.SLIP,
               "unknown_occluded": Tail.UNKNOWN, "unknown_no_opening": Tail.UNKNOWN}


def audit(n, thresholds, seed=0):
    fb = Feedback(thresholds)
    rng = np.random.default_rng(seed)
    pre = collections.Counter()
    tail = collections.Counter()
    for kind, intent in PREFIX_INTENT.items():
        for _ in range(n):
            got = fb.prefix(*prefix_case(kind, rng))
            pre[(kind, intent.name, got.name)] += 1
    for kind, intent in TAIL_INTENT.items():
        for _ in range(n):
            got = fb.tail(tail_case(kind, rng))
            tail[(kind, intent.name, got.name)] += 1
    return pre, tail


def recall(counter, n):
    """kind -> fraction of sequences classified as their intended event."""
    per = collections.defaultdict(lambda: [0, 0])
    for (kind, intent, got), c in counter.items():
        per[kind][1] += c
        if got == intent:
            per[kind][0] += c
    return {k: v[0] / v[1] for k, v in per.items()}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=40, help="sequences per intended event")
    ap.add_argument("--min-recall", type=float, default=.9)
    ap.add_argument("--max-abstention", type=float, default=.3)
    args = ap.parse_args()

    t = Thresholds()
    pre, tail = audit(args.n, t)
    pre_recall, pre_per = recall(pre, args.n)
    tail_recall, tail_per = recall(tail, args.n)

    # abstention: sequences whose intended event was NOT unknown but were called unknown
    def abstention(counter):
        bad = tot = 0
        for (kind, intent, got), c in counter.items():
            if intent != "UNKNOWN":
                tot += c
                bad += c if got == "UNKNOWN" else 0
        return bad / max(tot, 1)

    # The distinction the structured arm exists to exploit.
    crit = all(got == intent for (kind, intent, got), c in pre.items()
               if kind == "not_ready_slip") and \
           all(got == intent for (kind, intent, got), c in tail.items() if kind == "away")

    worst = min(list(pre_recall.values()) + list(tail_recall.values()))
    abst = max(abstention(pre), abstention(tail))
    validated = bool(worst >= args.min_recall and abst <= args.max_abstention and crit)

    report = {
        "thresholds": t.__dict__,
        "n_per_event": args.n,
        "prefix_recall": pre_recall,
        "tail_recall": tail_recall,
        "prefix_confusion": {f"{k}|intended={i}|got={g}": c for (k, i, g), c in sorted(pre.items())},
        "tail_confusion": {f"{k}|intended={i}|got={g}": c for (k, i, g), c in sorted(tail.items())},
        "worst_recall": worst,
        "abstention": abst,
        "critical_pair_separated": bool(crit),
        "criteria": {"min_recall": args.min_recall, "max_abstention": args.max_abstention},
        "synthetic_rule_checks_passed": validated,
        "feedback_validated": False,
        "confirmation_allowed": False,
        "label_source": "synthetic construction; real clips require hand annotation",
    }
    atomic_json(Path(args.out) / "feedback_report.json", report)
    print(f"worst_recall={worst:.3f}  abstention={abst:.3f}  "
          f"critical_pair_separated={crit}  feedback_validated={validated}")
    for name, r in (("prefix", pre_recall), ("tail", tail_recall)):
        for k, v in sorted(r.items()):
            print(f"  {name:7} {k:22} recall={v:.3f}")
    return 0 if validated else 1


if __name__ == "__main__":
    raise SystemExit(main())
