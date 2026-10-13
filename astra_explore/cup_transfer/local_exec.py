"""What Astra delegates (batch 3, 2026-10-13): the local executor interface and the local methods compared under it.

Delegated responsibility: given a grasp intention and a nominal execution plan, select a reliable local execution, use
feedback to revise it when necessary, and report completion or the need for replanning. Astra keeps every change of task
strategy; the local executor only chooses a bounded, persistent, object-relative correction of the grasp reference (the
executor's PersistentOffset, applied just before closing) from a fixed bank, for at most `attempts` grasp attempts.

    Request (from Astra)                                     Report (to Astra)
    target object                                            success
    nominal grasp: wrist reference + closure program         the correction executed at every attempt
    desired outcome: task-aligned grasp, lift and hold       measured outcome and failure stage of every attempt
    permitted corrections: the bank (+-1.5 cm per axis)      attempts used, elapsed time (decision / execution / reset)
    local attempt budget (3 grasp attempts)                  status: completion | budget_exhausted | need_replan

Methods (same bank, observations and feedback; begin(state) / choose() / observe(candidate, outcome)):
    A0 nominal        the nominal grasp; standard retry rule = the same grasp again
    A1 one_correction the single correction a context-free model of the source records scores best; retried as is
    A2 ranked_search  the context-free source ranking, tried in order
    A3 cf_updating    the context-free model, refitted on source + this scene's attempts after every attempt
    A4 ctx_frozen     the contextual model (state + correction) of the source records, next candidate after a failure
    A5 ctx_updating   the contextual model, refitted on source + this scene's attempts
Retry rules for A2-A5: "no_repeat" excludes a candidate after one failure in the scene; "confirm_once" after two (one
confirmation retry is allowed, taken only if the method still ranks it first). Update strength: None = frozen, w = weight of
a new target record in the local fit (1 = the batch-2 update, < 1 = conservative).
"""
import time
from collections import Counter

import numpy as np

from cup_transfer import s1v2 as V

SELECTABLE = lambda b: b["label"] in ("nominal", "sampled")          # the labelled positive control is never selectable
KINDS = {"A0": "nominal", "A1": "one_correction", "A2": "ranked_search", "A3": "cf_updating", "A4": "ctx_frozen", "A5": "ctx_updating"}


def failure_stage(oc):
    """Where a grasp attempt failed, from the measured outcome (None = success)."""
    if oc["task_success"]:
        return None
    s = oc.get("safety") or {}
    if s.get("tipped_before_lift"):
        return "tipped before the lift"
    if s.get("tip_below_table"):
        return "fingertip below the table"
    rise = oc.get("max_rise_cm") or 0.0
    if rise < 1.0:
        return "not lifted"
    if not oc.get("env_success"):
        return "slipped during the lift"
    if not oc.get("stable_last30"):
        return "lifted, not held stably"
    return "lifted, not held by the fingertips"


class LocalMethod:
    def __init__(self, code, bank, source=None, rep="geometry", update_w=None, retry="no_repeat", **model_kw):
        self.code, self.retry = code, retry
        self.name = KINDS[code]
        self.bank = [b for b in bank if SELECTABLE(b)]
        self.rep = "none" if code in ("A1", "A2", "A3") else rep
        self.update_w = update_w if code in ("A3", "A5") else None
        self.source = source if source is not None else V.OutcomeMemory()
        self.model_kw = model_kw

    def describe(self):
        return {"code": self.code, "name": self.name, "rep": None if self.code == "A0" else self.rep, "update_w": self.update_w,
                "retry": "same grasp" if self.code in ("A0", "A1") else self.retry, "source_records": len(self.source.records)}

    def begin(self, state):
        self.state, self.fails, self.fixed = state, Counter(), None
        self.mem = self.source.clone(frozen=self.update_w is None)
        self.sel = V.Selector(self.mem, self.rep, self.bank, **self.model_kw)
        if self.code == "A0":
            self.fixed = next(b for b in self.bank if b["id"] == "nominal")
        elif self.code == "A1":
            pick, _, _ = self.sel.choose(state)
            self.fixed = pick

    def excluded(self):
        lim = 1 if self.retry == "no_repeat" else 2
        return {i for i, n in self.fails.items() if n >= lim}

    def choose(self):
        """-> (candidate or None = nothing admissible left, info)"""
        if self.fixed is not None:
            return {"id": self.fixed["id"], "dp_cm": self.fixed["dp_cm"]}, {"q": self.fixed.get("q"), "mass": self.fixed.get("ess")}
        ex = self.excluded()
        pick, _, sc = self.sel.choose(self.state, exclude=ex)
        if pick is None:
            return None, {"excluded": sorted(ex)}
        top = sorted((s for s in sc if s["id"] not in ex), key=lambda s: -s["score"])[:3]
        return {"id": pick["id"], "dp_cm": pick["dp_cm"]}, {"q": pick["q"], "mass": pick["ess"], "excluded": sorted(ex),
                                                            "top3": [[s["id"], round(s["score"], 3)] for s in top]}

    def observe(self, cand, oc):
        if not oc["task_success"]:
            self.fails[cand["id"]] += 1
        if self.update_w is not None:
            x, _ = V.features(self.state, self.rep, cand["dp_cm"])
            self.mem.add(V.record(x, cand, oc, "target", w=self.update_w))


def run_scene(method, state, execute, attempts=3):
    """One scene: up to `attempts` grasp attempts. execute(candidate) -> (outcome, timing dict: execution_s, reset_s).
    Returns the report the local executor gives Astra, with every decision and its latency."""
    t0 = time.perf_counter()
    method.begin(state)
    t_begin = time.perf_counter() - t0
    rows, status = [], "budget_exhausted"
    for k in range(1, attempts + 1):
        td = time.perf_counter()
        cand, info = method.choose()
        dec_s = time.perf_counter() - td + (t_begin if k == 1 else 0.0)
        if cand is None:
            status = "need_replan"
            rows.append({"attempt": k, "picked": None, "decision_s": round(dec_s, 5), **info})
            break
        oc, timing = execute(cand)
        tu = time.perf_counter()
        method.observe(cand, oc)
        upd_s = time.perf_counter() - tu
        rows.append({"attempt": k, "picked": cand["id"], "dp_cm": cand["dp_cm"], "success": bool(oc["task_success"]),
                     "failure_stage": failure_stage(oc), "max_rise_cm": oc.get("max_rise_cm"), "safety_violation": oc.get("safety_violation"),
                     "decision_s": round(dec_s, 5), "update_s": round(upd_s, 5), **timing, **info})
        if oc["task_success"]:
            status = "completion"
            break
    first = next((r["attempt"] for r in rows if r.get("success")), None)
    return {"method": method.describe(), "status": status, "success": first is not None, "first_success_attempt": first,
            "attempts_used": sum(1 for r in rows if r["picked"] is not None), "attempts": rows,
            "decision_s_total": round(sum(r["decision_s"] for r in rows), 5),
            "execution_s_total": round(sum(r.get("execution_s", 0.0) for r in rows), 2),
            "reset_s_total": round(sum(r.get("reset_s", 0.0) for r in rows), 2)}


def summarize(reports, attempts=3):
    n = len(reports)
    at = [r["first_success_attempt"] for r in reports]
    out = {"scenes": n}
    for k in range(1, attempts + 1):
        out[f"within_{k}"] = sum(1 for a in at if a is not None and a <= k)
    out["attempts_used_mean"] = round(float(np.mean([r["attempts_used"] for r in reports])), 3) if n else None
    out["attempts_censored_mean"] = round(float(np.mean([a or attempts + 1 for a in at])), 3) if n else None
    ds = [x["decision_s"] for r in reports for x in r["attempts"]]
    out["decision_ms_median"] = round(1000 * float(np.median(ds)), 2) if ds else None
    out["decision_ms_max"] = round(1000 * float(np.max(ds)), 2) if ds else None
    out["need_replan"] = sum(1 for r in reports if r["status"] == "need_replan")
    return out
