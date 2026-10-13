"""System 1 v2 (batch 2): persistent grasp corrections, chosen by a predicted COMPLETE-primitive outcome (numpy only).

Changes against S1-original (tag s1-original), one at a time as the batch-2 experiments justify them:
  action      a persistent, object-relative translation of the grasp reference (the executor's PersistentOffset), from a
              fixed bank of corrections inside S1-original's per-axis bound (+-1.5 cm); closures unchanged
  outcome     the task-aligned grasp-and-lift result (batch2.outcome: held, stable, near the hand), not a 0.3 s change
  model       S1-original's machinery reused: locally weighted ridge regression (Gaussian kernel on a fixed scaled metric,
              unshrunk intercept, effective-sample-size support check), now regressing the primitive outcome on
              [state features, correction]
  selection   1. never a correction predicted unsafe (not modelled yet: the bank stays inside verified bounds);
              2. highest predicted success, where a correction without local support (kernel mass < 0.5) counts as
                 unknown (prior q0 = 0.5: untried corrections are explored before known failures);
              3. among candidates within `tie` of the best, the smallest correction
Representations (experiment 4): "current" = S1-original's 27 state features; "geometry" = those plus each fingertip body's
gap to the object's outer surface (/R) and height above its base (/H).
"""
import copy
import json

import numpy as np

from cup_transfer import system1 as S1

TIPS = ("R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip")


def features(state, rep, dp_cm=None):
    """state: a batch-2 state record (state_vector, geometry_before, raw_state). Returns the representation vector and scales.
    "post_geometry" (exploratory, beyond the plan's two arms): every fingertip body's gap to the outer surface (/R) and height
    above the base (/H) AFTER the candidate correction is applied to the hand (object frame = world axes for upright cups),
    plus the object's size; it depends on the correction, so records and queries are built per correction."""
    s = np.asarray(state["state_vector"], float)
    if rep == "current":
        return s, S1.SIG_S.copy()
    if rep == "post_geometry":
        g, raw = state["geometry_before"], state["raw_state"]
        R, H = float(raw["R"]), float(raw["H"])
        o = np.asarray(raw["obj_pos"], float)
        d = np.zeros(3) if dp_cm is None else np.asarray(dp_cm, float) / 100
        out = []
        for t in TIPS:
            if "vec_cm" not in g["tips"][t]:
                raise KeyError("post_geometry needs per-tip vectors (states recorded after 2026-10-13 03:40)")
            v = np.asarray(g["tips"][t]["vec_cm"], float) / 100 + d         # tip relative to the object base, after the correction
            out += [(np.linalg.norm(v[:2]) - R) / R, v[2] / H]
        out += [20 * R, 10 * H]
        return np.array(out), np.array([0.3] * (2 * len(TIPS)) + [0.3, 0.3])
    g = state["geometry_before"]["tips"]
    R, H = float(state["raw_state"]["R"]), float(state["raw_state"]["H"])
    extra = []
    for t in TIPS:
        extra += [g[t]["surface_gap_cm"] / (100 * R), g[t]["height_cm"] / (100 * H)]
    return np.concatenate([s, extra]), np.concatenate([S1.SIG_S, np.full(len(extra), 0.3)])


class OutcomeMemory:
    """Experience records: state features, requested and applied correction, primitive outcome, provenance."""

    def __init__(self, records=None, frozen=False):
        self.records, self.frozen = list(records or []), frozen

    def clone(self, frozen=None):
        return OutcomeMemory(copy.deepcopy(self.records), self.frozen if frozen is None else frozen)

    def add(self, rec):
        if self.frozen:
            return False
        self.records.append(rec)
        return True

    def arrays(self):
        if not self.records:
            return np.zeros((0, 0)), np.zeros((0, 3)), np.zeros(0)
        X = np.array([r["x"] for r in self.records])
        D = np.array([r["dp_cm"] for r in self.records]) / S1.BOUND[:3]
        y = np.array([float(r["success"]) for r in self.records])
        return X, D, y


class OutcomeModel:
    """Q(x, dp) ~ P(complete grasp-and-lift success), locally weighted ridge regression (S1-original's form)."""

    def __init__(self, memory, scales, h=1.0, lam=0.1, mass_min=0.5, sig_dp=0.5, k=60):
        # support = total kernel weight near the query (an exact repeat of a tried correction weighs 1): unlike the effective
        # sample size it does not call one observed failure 'unknown', and far records barely count
        self.m, self.sc, self.h, self.lam, self.mass_min, self.sig_dp, self.k = memory, scales, h, lam, mass_min, sig_dp, k

    def predict(self, x, dp_cm, exclude=None):
        X, D, y = self.m.arrays()
        if not len(y):
            return None, 0.0, []
        d_n = dp_cm / S1.BOUND[:3]
        idx = np.arange(len(y)) if exclude is None else np.array([i for i in range(len(y)) if i not in exclude], int)
        if not len(idx):
            return None, 0.0, []
        dist = np.sqrt((((X[idx] - x) / self.sc) ** 2).sum(1) + (((D[idx] - d_n) / self.sig_dp) ** 2).sum(1))
        order = np.argsort(dist)[:self.k]
        idx, dist = idx[order], dist[order]
        w = np.exp(-dist ** 2 / (2 * self.h ** 2))
        ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
        if w.sum() < self.mass_min:
            return None, float(w.sum()), idx[:5].tolist()
        Z = np.hstack([(D[idx] - d_n) / self.sig_dp, np.ones((len(idx), 1))])    # local linear in the correction only
        reg = self.lam * np.eye(Z.shape[1])
        reg[-1, -1] = 1e-6
        th = np.linalg.solve(Z.T @ (w[:, None] * Z) + reg, Z.T @ (w * y[idx]))
        q = float(np.clip(th[-1], 0.0, 1.0))                                       # value at the query
        return q, float(w.sum()), idx[:5].tolist()


class Selector:
    """S1-v2's decision: one persistent correction from the bank (the development positive control is never in it)."""

    def __init__(self, memory, rep, bank, q0=0.5, tie=0.05, **model_kw):
        self.memory, self.rep, self.bank, self.q0, self.tie = memory, rep, [b for b in bank if b["label"] != "development positive control (not available to the benchmark)"], q0, tie
        self.model_kw = model_kw

    def scores(self, state):
        x, sc = features(state, self.rep)
        model = OutcomeModel(self.memory, sc, **self.model_kw)
        out = []
        for b in self.bank:
            if self.rep == "post_geometry":
                x, _ = features(state, self.rep, b["dp_cm"])
            q, ess, nb = model.predict(x, np.array(b["dp_cm"]))
            out.append({"id": b["id"], "dp_cm": b["dp_cm"], "q": q, "ess": round(ess, 3), "score": self.q0 if q is None else q, "neighbors": nb})  # ess = kernel mass
        return x, out

    def choose(self, state):
        x, sc = self.scores(state)
        best = max(s["score"] for s in sc)
        near = [s for s in sc if s["score"] >= best - self.tie]
        pick = min(near, key=lambda s: (float(np.linalg.norm(s["dp_cm"])), s["id"]))
        x, _ = features(state, self.rep, pick["dp_cm"])          # the picked correction's own features (post_geometry)
        return pick, x, sc


def record(state_x, cand, outcome, provenance, applied_cm=None):
    return {"x": np.asarray(state_x).tolist(), "dp_cm": list(map(float, cand["dp_cm"])), "applied_cm": applied_cm or list(map(float, cand["dp_cm"])),
            "success": bool(outcome["task_success"]), "max_rise_cm": outcome.get("max_rise_cm"), "provenance": provenance, "id": cand["id"]}


def episode(selector, state, lookup, attempts=3, update=True, log=None):
    """Sequential attempts on one target state: choose, observe the (recorded) outcome, update (if allowed), repeat until
    success. lookup(candidate id) -> outcome. Returns the attempts with the update effects."""
    hist = []
    for k in range(attempts):
        pick, x, sc = selector.choose(state)
        if selector.rep == "post_geometry":
            x, _ = features(state, selector.rep, pick["dp_cm"])
        oc = lookup(pick["id"])
        rec = record(x, pick, oc, "target")
        entry = {"attempt": k + 1, "picked": pick["id"], "dp_cm": pick["dp_cm"], "predicted_q": pick["q"], "ess": pick["ess"],
                 "success": oc["task_success"], "retrieved": pick["neighbors"]}
        if update:
            before = {s["id"]: s["score"] for s in sc}
            added = selector.memory.add(rec)
            _, sc2 = selector.scores(state)
            after = {s["id"]: s["score"] for s in sc2}
            nxt, _, _ = selector.choose(state)
            entry.update(stored=added, prediction_change_on_picked=None if pick["q"] is None else round(after[pick["id"]] - before[pick["id"]], 4),
                         preferred_changed=nxt["id"] != pick["id"], next_pick=nxt["id"])
        hist.append(entry)
        if log is not None:
            log.write(json.dumps(entry) + "\n")
        if oc["task_success"]:
            break
    return hist
