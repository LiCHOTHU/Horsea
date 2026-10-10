"""Human-readable overnight report from state.json (+ per-activity results). Regenerated after every activity.

    python -m s1s2.report --root <study dir>
"""
import argparse
import json
from pathlib import Path

from s1s2.registry import CELLS, PLATFORMS

QUESTIONS = [
    "Could Astra's behavior be converted into valid student training data?",
    "Did the standalone student improve?",
    "Did additional successful demonstrations improve stability?",
    "Did more frequent observation help?",
    "Did naive imitation of failed attempts help or hurt?",
    "Did Astra takeover help the same student?",
    "Did recent history improve takeover?",
    "After corrective training, did the student improve with Astra disabled?",
    "Was that improvement greater than extra training or additional reset demonstrations?",
    "What is the smallest next experiment justified by the evidence?",
]


def outcome(state, p, c):
    rec = state["cells"][p][c]
    r = rec.get("result") or {}
    if r.get("n"):
        return f"{r.get('successes')}/{r.get('n')}"
    return ""


def fmt_cell(state, p, c):
    rec = state["cells"][p][c]
    s = rec.get("status", "PENDING")
    o = outcome(state, p, c)
    alias = f" (= {rec['alias_of']})" if rec.get("alias_of") else ""
    return f"{s}{' ' + o if o else ''}{alias}"


def answers(state, p):
    """Data-driven answers where the needed cells are COMPLETE; otherwise say which cells are missing."""
    cells = state["cells"][p]
    pl = state["platforms"].get(p, {})

    def res(c):
        r = cells[c].get("result") or {}
        return (r.get("successes"), r.get("n")) if cells[c].get("status") in ("COMPLETE", "DUPLICATE") and r.get("n") else None

    def cmp(a, b, label):
        ra, rb = res(a), res(b)
        if not ra or not rb:
            missing = [x for x, r in ((a, ra), (b, rb)) if not r]
            return f"not answered: {', '.join(f'{m} {cells[m].get('status')}' for m in missing)}"
        return f"{label}: {a} {ra[0]}/{ra[1]} vs {b} {rb[0]}/{rb[1]} (exploratory, paired starts)"

    out = []
    audit = pl.get("audit", {})
    data = pl.get("data", {})
    out.append(f"{audit.get('alignment', 'not audited')}; valid teacher episodes {data.get('valid_episodes', 0)}, successes "
               f"{data.get('successes', 0)}; datasets {data.get('datasets', {})}")
    out.append(cmp("E05", "E02", "anchor vs original student") if res("E02") else cmp("E05", "E03", "anchor vs D1 student"))
    out.append(cmp("E05", "E03", "DN vs D1") + "; " + cmp("E04", "E03", "D3 vs D1"))
    out.append(cmp("E06", "E05", "half executed chunk vs anchor"))
    out.append(cmp("E07", "E05", "successes+failures vs successes only"))
    out.append(cmp("E08", "E05", "anchor + takeover vs anchor alone (assistance, not learning)"))
    out.append(cmp("E09", "E08", "takeover with history vs without"))
    out.append(cmp("E11", "E05", "after corrective training vs anchor"))
    out.append(cmp("E11", "E10", "corrections vs extra training") + "; " + cmp("E11", "E12", "corrections vs extra reset demos"))
    out.append(pl.get("next_experiment", "to be written by the author from the completed cells"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    state = json.loads((root / "state.json").read_text())
    ledger = {}
    try:
        from s1s2.ledger import Ledger
        ledger = Ledger(root).summary()
    except Exception:  # noqa: BLE001
        pass
    L = ["# Overnight System-1 / System-2 transfer study: status report", "",
         f"Study root: `{root}`. Created {state.get('created')}. Report regenerated after every activity.", "",
         "## Budget", "", "```", json.dumps(ledger, indent=1), "```", "",
         "## 12 x 4 matrix", "", "| cell | configuration | " + " | ".join(PLATFORMS) + " |", "|---|---|" + "---|" * len(PLATFORMS)]
    for c, desc in CELLS.items():
        L.append(f"| {c} | {desc} | " + " | ".join(fmt_cell(state, p, c) for p in PLATFORMS) + " |")
    L += ["", "Status vocabulary: COMPLETE, RUNNING, PENDING_BUDGET, BLOCKED_INTERFACE, BLOCKED_DATA, NO_STUDENT_SIGNAL, "
          "NOT_APPLICABLE, DUPLICATE, FAILED (plus BLOCKED_NO_CORRECTIONS for E11 and PENDING for not yet reached). "
          "Numbers are successes / evaluated episodes on the stated pool (dev unless marked test).", ""]
    for p in PLATFORMS:
        pl = state["platforms"].get(p, {})
        L += [f"## {p}", "", f"Cohort: {pl.get('cohort', 'unknown')}. Student: {pl.get('student', {}).get('summary', 'unknown')}.",
              f"Audit: {pl.get('audit', {}).get('summary', 'not run')}", ""]
        for i, (q, ans) in enumerate(zip(QUESTIONS, answers(state, p)), 1):
            L.append(f"{i}. **{q}** {ans}")
        L.append("")
        notes = pl.get("notes", [])
        if notes:
            L += ["Notes:", ""] + [f"- {n}" for n in notes] + [""]
    L += ["## Orchestrator log (latest 60 lines)", "", "```"] + state.get("log", [])[-60:] + ["```", ""]
    (root / "REPORT.md").write_text("\n".join(L))
    print("wrote", root / "REPORT.md")


if __name__ == "__main__":
    main()
