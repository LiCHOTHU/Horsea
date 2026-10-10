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
    out = []
    if r.get("n"):
        req = r.get("starts_requested")
        out.append(f"{r.get('successes')}/{r.get('n')}" + (f" of {req} requested" if req and req != r.get("n") else "") + f" {r.get('pool', 'dev')}")
    t = rec.get("test") or {}
    if t.get("n"):
        out.append(f"test {t.get('successes')}/{t.get('n')}")
    return "; ".join(out)


def fmt_cell(state, p, c):
    rec = state["cells"][p][c]
    s = rec.get("status", "PENDING")
    o = outcome(state, p, c)
    alias = f" (= {rec['alias_of']})" if rec.get("alias_of") else ""
    return f"{s}{' ' + o if o else ''}{alias}"


def answers(state, p):
    """Data-driven comparisons where the needed cells have evaluated episodes (complete, or partial with the exact
    denominator); otherwise say which cells are missing. Raw paired per-start outcomes are listed (exploratory)."""
    cells = state["cells"][p]
    pl = state["platforms"].get(p, {})

    def rec_of(c):
        r = cells[c]
        return cells[r["alias_of"]] if r.get("status") == "DUPLICATE" and r.get("alias_of") in cells else r

    def res(c):
        r = rec_of(c).get("result") or {}
        return r if r.get("n") else None

    def paired(ra, rb):
        a = {e.get("start_id"): bool(e.get("success")) for e in ra.get("per_episode") or [] if "start_id" in e and e.get("status") != "PENDING_BUDGET"}
        b = {e.get("start_id"): bool(e.get("success")) for e in rb.get("per_episode") or [] if "start_id" in e and e.get("status") != "PENDING_BUDGET"}
        common = [k for k in a if k in b]
        return ", ".join(f"{k}: {'S' if a[k] else 'F'}/{'S' if b[k] else 'F'}" for k in common) or "no common start"

    def cmp(a, b, label):
        ra, rb = res(a), res(b)
        if not ra or not rb:
            missing = [x for x, r in ((a, ra), (b, rb)) if not r]
            return f"{label}: not answered ({', '.join(f'{m} {cells[m].get('status')}' for m in missing)})"
        part = "" if all(cells[x]["status"] in ("COMPLETE", "DUPLICATE") for x in (a, b)) else " [partial]"
        return (f"{label}: {a} {ra['successes']}/{ra['n']} vs {b} {rb['successes']}/{rb['n']}{part}; per start ({a}/{b}) "
                f"{paired(ra, rb)} (exploratory)")

    out = []
    audit = pl.get("audit", {})
    data = pl.get("data", {})
    sets = data.get("sets") or {}
    out.append(f"{audit.get('alignment') or audit.get('summary') or 'not audited'}; source episodes {data.get('episodes', 0)} "
               f"({data.get('valid_episodes', 0)} valid, {data.get('successes', 0)} successful); N = {sets.get('N', 'n/a')}")
    out.append(cmp("E05", "E02", "anchor vs original pretrained student") if res("E02") else
               "no pretrained student (E02 " + cells["E02"].get("status", "") + "); " + cmp("E05", "E03", "anchor (DN) vs D1 student"))
    out.append(cmp("E04", "E03", "D3 vs D1") + "; " + cmp("E05", "E03", "DN vs D1"))
    out.append(cmp("E06", "E05", "half executed chunk vs anchor"))
    out.append(cmp("E07", "E05", "successes+failures vs successes only"))
    out.append(cmp("E08", "E05", "anchor + one takeover vs anchor alone (assistance, not learning)"))
    out.append(cmp("E09", "E08", "takeover with three student records vs current observation only"))
    out.append(cmp("E11", "E05", "after corrective training vs anchor"))
    out.append(cmp("E11", "E10", "corrections vs extra training") + "; " + cmp("E11", "E12", "corrections vs extra reset demos"))
    out.append(pl.get("next_experiment", "to be written from the completed cells"))
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
        why = [(c, r) for c, r in state["cells"][p].items() if r.get("reason")]
        if why:
            L += ["Cell status reasons:", ""] + [f"- {c} {r['status']}: {r['reason']}" for c, r in why] + [""]
        per_cell = {k.split("/", 1)[1]: v for k, v in (ledger.get("per_cell") or {}).items() if k.startswith(p + "/")}
        if per_cell:
            L += [f"Astra calls charged on this platform: {sum(per_cell.values())} ({', '.join(f'{k} {v}' for k, v in per_cell.items())}).", ""]
        notes = pl.get("notes", [])
        if notes:
            L += ["Notes:", ""] + [f"- {n}" for n in notes] + [""]
    L += ["## Orchestrator log (latest 60 lines)", "", "```"] + state.get("log", [])[-60:] + ["```", ""]
    (root / "REPORT.md").write_text("\n".join(L))
    print("wrote", root / "REPORT.md")


if __name__ == "__main__":
    main()
