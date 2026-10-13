"""Phase 0 (offline): where does an attempt's wall time go, and what are Astra's calls for? From the logs of the cup-transfer
run (experiments/cup_transfer_2026-10-12), no simulator needed.

    python -m cup_transfer.costs --out <dir>

Time per attempt (non-overlapping; the harness is single-threaded, so the components are sequential):
  astra       sum of Astra call latencies (Codex process start to exit, incl. retries of invalid responses)
  execution   sum of the common executor's wall time per decision (simulation steps + System 1 planning inside it)
  other       attempt wall time - astra - execution (observation rendering, image writes, prompt building, history)
Startup (simulator launch, calibration) happens once per process and is not part of any attempt (reported separately when
the process log has it).

Call purpose, from the decision's declared interaction mode and the measured events before it (never Astra's hidden
reasoning):
  initial approach     approach / reposition decisions before the attempt's first closure, the first of them
  approach correction  later approach / reposition decisions before the first closure
  closure              a decision that closes the active hand (open < 0.2 -> >= 0.3)
  grasp verification   lift / hold decisions while no failed lift has happened since the last closure
  recovery             any decision after a failed lift (wrist rose >= 3 cm, object < 1 cm) or after the hand reopened
                       following a closure, until the next closure
  other                anything else (mode "other", stop)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cup_transfer.analyze_cup import METHODS, ROOT, decisions, jl

PURPOSES = ("initial approach", "approach correction", "closure", "grasp verification", "recovery", "other")


def attempt_dirs():
    out = [("cup_a", "source", f"s{k}", ROOT / f"source_interaction_dataset/run/attempt_{k}") for k in (1, 2, 3)]
    for obj in ("cup_b", "cyl_c"):
        for st in ("t1", "t2"):
            for m in METHODS:
                for k in (1, 2):
                    a = ROOT / f"per_attempt_transition_logs/run/{obj}/{st}/{m}/attempt_{k}"
                    if (a / "summary.json").exists():
                        out.append((obj, m, f"{st}/a{k}", a))
    return out


def label_calls(decs):
    """One purpose label per decision record (executed or not)."""
    labels, closed_once, failed_since_close, prev_open = [], False, False, {"left": 0.0, "right": 0.0}
    first_approach_done = False
    for d in decs:
        c = d.get("decision_content")
        if not c:
            labels.append("other")
            continue
        mode, h = c.get("interaction_mode"), c.get("active_hand")
        closing = False
        if h in ("left", "right"):
            m = max(c[h]["finger_base"], c[h]["finger_tip"])
            closing = m >= 0.3 and prev_open[h] < 0.2
            reopened = closed_once and m < 0.2 and prev_open[h] >= 0.3
            prev_open[h] = m
        else:
            reopened = False
        if c.get("stop") or mode == "other":
            lab = "other"
        elif closing:
            lab = "closure"
            closed_once, failed_since_close = True, False
        elif failed_since_close or reopened:
            lab = "recovery"
            failed_since_close = True
        elif not closed_once and mode in ("approach", "reposition"):
            lab = "approach correction" if first_approach_done else "initial approach"
            first_approach_done = True
        elif mode in ("lift", "hold", "transport"):
            lab = "grasp verification"
        else:
            lab = "recovery" if closed_once else "approach correction"
        r = d.get("result") or {}
        if mode == "lift" and r and h in ("left", "right"):
            if r["hands"].get(h, {}).get("moved_cm", 0) >= 3 and r.get("object", {}).get("rise_above_start_cm", 0) < 1:
                failed_since_close = True
        labels.append(lab)
    return labels


def attempt_costs(adir):
    decs = decisions(adir)
    summ = json.loads((adir / "summary.json").read_text())
    labels = label_calls(decs)
    calls, astra, execu, tokens = [], 0.0, 0.0, defaultdict(int)
    for d, lab in zip(decs, labels):
        lat = sum((c.get("meta") or {}).get("latency_s") or 0.0 for c in d.get("calls", []))
        ex = (d.get("result") or {}).get("wall_s") or 0.0
        astra += lat
        execu += ex
        for c in d.get("calls", []):
            for k, v in ((c.get("meta") or {}).get("usage") or {}).items():
                tokens[k] += int(v or 0)
        calls.append({"decision": d["decision"], "purpose": lab, "mode": (d.get("decision_content") or {}).get("interaction_mode"),
                      "n_calls": len(d.get("calls", [])), "astra_s": round(lat, 2), "execution_s": round(ex, 2),
                      "steps": (d.get("result") or {}).get("steps_executed", 0)})
    wall = summ.get("attempt_wall_time_s") or 0.0
    return {"outcome": summ["outcome"], "steps": summ["steps_used"], "simulated_s": round(summ["steps_used"] / 30.0, 1),
            "wall_s": wall, "astra_s": round(astra, 1), "execution_s": round(execu, 1), "other_s": round(wall - astra - execu, 1),
            "model_calls": summ.get("model_calls"), "decisions": len(decs), "tokens": dict(tokens), "calls": calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for obj, arm, cond, adir in attempt_dirs():
        rows.append(dict(attempt_costs(adir), object=obj, arm=arm, condition=cond))
    (out / "costs.json").write_text(json.dumps(rows, indent=1))

    L = ["# Phase 0 (offline): where the time goes in the 29 cup-transfer attempts", "",
         "Source: experiments/cup_transfer_2026-10-12 logs (decision.json call latencies and executor wall times; summary.json "
         "attempt wall time). Components are sequential (single-threaded harness), so they add up to the attempt's wall time.", "",
         "## Per object (sums over attempts)", "",
         "| object | attempts | wall min | Astra waiting | executor (sim + System 1) | other (rendering, I/O, prompt) | simulated time | Astra calls | input tokens | output tokens |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for obj in ("cup_a", "cup_b", "cyl_c"):
        R = [r for r in rows if r["object"] == obj]
        w = sum(r["wall_s"] for r in R)
        f = lambda k: sum(r[k] for r in R)
        L.append(f"| {obj} | {len(R)} | {w / 60:.1f} | {f('astra_s') / 60:.1f} min ({100 * f('astra_s') / w:.0f} %) | {f('execution_s') / 60:.1f} min "
                 f"({100 * f('execution_s') / w:.0f} %) | {f('other_s') / 60:.1f} min ({100 * f('other_s') / w:.0f} %) | {f('simulated_s') / 60:.1f} min | "
                 f"{f('model_calls')} | {sum(r['tokens'].get('input_tokens', 0) for r in R):,} | {sum(r['tokens'].get('output_tokens', 0) for r in R):,} |")
    W = sum(r["wall_s"] for r in rows)
    L += ["", f"All 29 attempts: {W / 60:.1f} min wall, of which Astra {sum(r['astra_s'] for r in rows) / W * 100:.0f} %, executor "
          f"{sum(r['execution_s'] for r in rows) / W * 100:.0f} %, other {sum(r['other_s'] for r in rows) / W * 100:.0f} %. "
          f"Median Astra latency per call {np.median([c['astra_s'] / max(1, c['n_calls']) for r in rows for c in r['calls'] if c['n_calls']]):.1f} s; "
          f"executor speed {sum(r['steps'] for r in rows) / max(1e-9, sum(r['execution_s'] for r in rows)):.1f} control steps per wall second.", "",
          "## Astra calls by purpose (all attempts)", "",
          "| purpose | calls | Astra min | executor min | control steps | share of calls on cup B |", "|---|---|---|---|---|---|"]
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0])
    aggB = defaultdict(int)
    for r in rows:
        for c in r["calls"]:
            g = agg[c["purpose"]]
            g[0] += c["n_calls"]
            g[1] += c["astra_s"]
            g[2] += c["execution_s"]
            g[3] += c["steps"]
            if r["object"] == "cup_b":
                aggB[c["purpose"]] += c["n_calls"]
    nB = sum(aggB.values())
    for p in PURPOSES:
        g = agg[p]
        L.append(f"| {p} | {g[0]} | {g[1] / 60:.1f} | {g[2] / 60:.1f} | {g[3]} | {100 * aggB[p] / max(1, nB):.0f} % |")
    L += ["", "## Per attempt", "", "| object | arm | condition | outcome | wall s | Astra s | executor s | other s | simulated s | calls |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['object']} | {r['arm']} | {r['condition']} | {r['outcome']} | {r['wall_s']} | {r['astra_s']} | {r['execution_s']} | "
                 f"{r['other_s']} | {r['simulated_s']} | {r['model_calls']} |")
    (out / "costs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
