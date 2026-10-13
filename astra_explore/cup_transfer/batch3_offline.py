"""Batch 3 (delegation study, 2026-10-13), offline part: manifest, source-data selections and the replay evaluations.

What should Astra delegate, and does a learned System 1 do that work better than simpler alternatives? Every replay uses the
recorded outcome of every bank candidate at a development state (batch 2's correction banks), so a method's choices can be
scored without simulation. Replays are labelled REPLAY everywhere; fresh executions verify the differences that matter.

    python -m cup_transfer.batch3_offline manifest        source pool, 3 nested selections (16 / 64 / 256), settings (frozen first)
    python -m cup_transfer.batch3_offline expA            A0-A5 at 64 source interactions (replay)
    python -m cup_transfer.batch3_offline expB            A2 / A4 / A5 at 0 / 16 / 64 / 256 source interactions (replay)
    python -m cup_transfer.batch3_offline expC_weight     conservative update weight from development data (noisy replay)
    python -m cup_transfer.batch3_offline expD            representations: current vs geometry (+ post-correction, audited)
Outputs: experiments/cup_delegation_2026-10-13/.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cup_transfer import batch2_offline as BO  # noqa: E402
from cup_transfer import local_exec as LX  # noqa: E402
from cup_transfer import s1v2 as V  # noqa: E402

OUT = Path("/home/licho/workspace/Horsea/experiments/cup_delegation_2026-10-13")
SOURCE_OBJECTS = ("dev_f1", "dev_f2", "dev_f3", "dev_f4", "dev_g1", "dev_g2", "dev_g3", "dev_g4")
DEV_PRIMARY = ("cup_a", "cup_b", "cyl_c")
BUDGETS = (0, 16, 64, 256)
N_SEL, SEED = 3, 20261014
MODEL_KW = {"h": 1.0, "lam": 0.1, "mass_min": 0.5, "sig_dp": 0.5, "k": 60, "q0": 0.5, "tie": 0.05}       # batch-2 values
COMPARISONS = [("A1", "A0", "Was the original grasp simply poorly calibrated?"),
               ("A2", "A1", "Does trying a short list provide sufficient adaptation?"),
               ("A4", "A2", "Does conditioning on the current state improve selection?"),
               ("A3", "A2", "Does feedback-driven updating help beyond moving down a list?"),
               ("A5", "A4", "Does updating the contextual model help?"),
               ("A5", "A3", "Does context improve how feedback is generalized?")]


def wj(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------- data
_SCENES = {}


def scenes(obj):
    """{start: {"state", "rows", "repeats", "r2"}} of batch 2's bank at that object (cached)."""
    if obj not in _SCENES:
        _SCENES[obj] = BO.states(obj)
    return _SCENES[obj]


def bank():
    from cup_transfer.batch2 import candidate_bank
    return candidate_bank()


def pool_keys(obj):
    """Every selectable (start, candidate) of an object's bank, in a fixed order; never depends on outcomes."""
    ids = [b["id"] for b in bank() if LX.SELECTABLE(b)]
    return [(st, i) for st in sorted(scenes(obj)) for i in ids]


def make_selections():
    sel = {}
    for k in range(N_SEL):
        rng = np.random.default_rng(SEED + k)
        sel[k] = {}
        for o in SOURCE_OBJECTS:
            keys = pool_keys(o)
            sel[k][o] = [list(keys[i]) for i in rng.permutation(len(keys))]
    return sel


_MAN = {}


def manifest():
    if "m" not in _MAN:
        _MAN["m"] = json.loads((OUT / "config" / "manifest.json").read_text())
    return _MAN["m"]


def source_memory(k, budget, rep, exclude_obj=None):
    """Selection k's first budget/8 records of every source object (minus a held-out object's own records)."""
    man = manifest()
    mem = V.OutcomeMemory()
    n = budget // len(SOURCE_OBJECTS)
    for o in SOURCE_OBJECTS:
        if o == exclude_obj:
            continue
        sc = scenes(o)
        for st, cid in man["selections"][str(k)][o][:n]:
            S = sc[st]
            r = next(x for x in S["rows"] if x["candidate"]["id"] == cid)
            x, _ = V.features(S["state"], rep, r["candidate"]["dp_cm"])
            mem.add(V.record(x, r["candidate"], r["outcome"], f"source:{o}:{st}"))
    return mem


def dev_targets(sets=("primary", "additional")):
    out = []
    if "primary" in sets:
        out += [("primary", o, st) for o in DEV_PRIMARY for st in sorted(scenes(o))]
    if "additional" in sets:
        out += [("additional", o, st) for o in SOURCE_OBJECTS for st in sorted(scenes(o))]
    return out


def replay_exec(S):
    y = {r["candidate"]["id"]: r["outcome"] for r in S["rows"]}
    return lambda cand: (y[cand["id"]], {"execution_s": 0.0, "reset_s": 0.0})


def noisy_exec(S, rng):
    """Development noise model made only of recorded executions: an attempt's outcome is drawn from every recorded execution of
    that candidate at that state (bank pass, repeats, second pass); a candidate executed once keeps its outcome."""
    rec = {}
    for r in S["rows"]:
        rec.setdefault(r["candidate"]["id"], []).append(r["outcome"])
    for r in S["repeats"] + S["r2"]:
        rec.setdefault(r["candidate"]["id"], []).append(r["outcome"])
    return lambda cand: (rec[cand["id"]][int(rng.integers(len(rec[cand["id"]])))], {"execution_s": 0.0, "reset_s": 0.0})


def method(code, k, budget, rep, target_obj, update_w=1.0, retry="no_repeat", mk=None):
    excl = target_obj if target_obj in SOURCE_OBJECTS else None
    src = source_memory(k, budget, "none" if code in ("A1", "A2", "A3") else rep, exclude_obj=excl) if code != "A0" and budget else V.OutcomeMemory()
    return LX.LocalMethod(code, bank(), source=src, rep=rep, update_w=update_w, retry=retry, **dict(MODEL_KW, **(mk or {})))


def run(codes, budgets, reps, sels, targets, retry="no_repeat", update_w=1.0, tag="", noisy_draws=0, mk=None):
    rows = []
    for (tset, o, st) in targets:
        S = scenes(o)[st]
        nominal_ok = next(r for r in S["rows"] if r["candidate"]["id"] == "nominal")["outcome"]["task_success"]
        for b in budgets:
            for k in sels:
                for rep in reps:
                    for code in codes:
                        if code in ("A0",) and (k != sels[0] or rep != reps[0] or b != budgets[0]):
                            continue                                    # A0 uses no source data and no representation
                        if code in ("A1", "A2", "A3") and rep != reps[0]:
                            continue                                    # context-free: the representation does not enter
                        draws = range(noisy_draws) if noisy_draws else [None]
                        for d in draws:
                            m = method(code, k, b, rep, o, update_w=update_w, retry=retry, mk=mk)
                            ex = replay_exec(S) if d is None else noisy_exec(S, np.random.default_rng([SEED, int(hashlib.md5(f"{o}/{st}".encode()).hexdigest()[:8], 16), d]))
                            rep_ = LX.run_scene(m, S["state"], ex)
                            rows.append({"tag": tag, "target_set": tset, "object": o, "start": st, "nominal_held": nominal_ok, "budget": b,
                                         "effective_budget": b - (b // len(SOURCE_OBJECTS) if o in SOURCE_OBJECTS else 0),
                                         "selection": k, "rep": m.describe()["rep"], "code": code, "retry": retry, "update_w": m.update_w,
                                         "h": m.model_kw["h"],
                                         "draw": d, **{kk: rep_[kk] for kk in ("status", "success", "first_success_attempt", "attempts_used", "attempts",
                                                                               "decision_s_total")}})
    return rows


def table(rows, keys, filt=lambda r: True):
    groups = {}
    for r in rows:
        if filt(r):
            groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out[g] = LX.summarize([{"first_success_attempt": r["first_success_attempt"], "attempts_used": r["attempts_used"], "attempts": r["attempts"],
                                "status": r["status"]} for r in rs])
    return out


def paired(rows, a, b, filt=lambda r: True, key="first"):
    """Scene-paired comparison of methods a and b (same scene, selection, budget, representation where it applies)."""
    idx = {}
    for r in rows:
        if filt(r):
            idx[(r["code"], r["object"], r["start"], r["selection"], r["budget"], r["draw"])] = r
    w = l_ = t = 0
    for (c, o, st, k, bb, d), ra in idx.items():
        if c != a:
            continue
        rb = idx.get((b, o, st, k, bb, d)) or idx.get((b, o, st, 0, bb, d))          # A0 is run once (selection 0)
        if rb is None:
            continue
        fa, fb = ra["first_success_attempt"], rb["first_success_attempt"]
        if key == "first":
            sa, sb = fa == 1, fb == 1
        else:
            sa, sb = fa is not None, fb is not None
        w += sa and not sb
        l_ += sb and not sa
        t += sa == sb
    return {"a_better": w, "b_better": l_, "same": t}


# ---------------------------------------------------------------------------- commands
def cmd_manifest(a):
    p = OUT / "config" / "manifest.json"
    if p.exists() and not a.force:
        raise SystemExit(f"already written: {p}")
    sel = make_selections()
    bk = bank()
    locked = json.loads(Path("/home/licho/workspace/Horsea/experiments/cup_locked_targets_2026-10-13/locked_targets.json").read_text())
    m = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "What should Astra delegate, and does a learned System 1 do that work better than simpler alternatives? (user plan 2026-10-13)",
        "configuration_resolution": (
            "Batch 2 used two source configurations: 7 objects (cups A, B, C and dev f1-f4) for its round-2 online runs on dev f1 and f3 "
            "and for its section-10 proposal, 11 objects (+ dev g1-g4) for the final leave-one-object-out tables and the round-3 online runs "
            "on dev g1 and g4. Those numbers stand as recorded. This study fixes ONE configuration: source = the eight development-family "
            "cups (f1-f4, g1-g4); development targets = cups A, B and cylinder C at their recorded bank starts; the development family "
            "doubles as additional development targets, each with its OWN records removed from the source selection (effective budget "
            "7/8 of the nominal one); locked evaluation = cup_t1, cup_t2 (never loaded before the prospective batch)."),
        "task": "adapt the existing grasp-and-lift skill to a new object and the nominal grasp supplied for it (not arbitrary manipulation)",
        "source_objects": SOURCE_OBJECTS,
        "source_pool": {o: {"starts": sorted(scenes(o)), "records": len(pool_keys(o))} for o in SOURCE_OBJECTS},
        "source_record": "pre-closing state -> one bounded persistent correction from the bank -> one executed grasp-and-lift -> task-aligned outcome "
                         "(batch 2's correction banks, rep 0; the labelled positive control is excluded)",
        "selections": {"rule": f"for selection k = 0, 1, 2: numpy.random.default_rng({SEED} + k) permutes every source object's records "
                               "(object order as listed; records ordered by start, then candidate id); budget B takes the first B/8 of every "
                               "object's permutation, so 16 within 64 within 256; fixed before any target outcome was looked at",
                       "budgets": BUDGETS, "per_object": {str(b): b // len(SOURCE_OBJECTS) for b in BUDGETS}, **{str(k): v for k, v in sel.items()}},
        "development_targets": {"primary": {o: sorted(scenes(o)) for o in DEV_PRIMARY}, "additional": {o: sorted(scenes(o)) for o in SOURCE_OBJECTS}},
        "candidate_bank": [{"id": b["id"], "dp_cm": b["dp_cm"]} for b in bk if LX.SELECTABLE(b)],
        "candidate_bank_note": "nominal + 30 Halton points in the verified per-axis bound +-1.5 cm (object frame), persistent from just before closing "
                               "through lift and hold (cup_task.PersistentOffset); the development positive control is never selectable or used as data",
        "objective": "task-aligned success (batch 2, frozen 2026-10-13): environment lift success, last 30 hold steps >= 4 cm up, upright (up_z >= 0.9), "
                     "<= 0.10 m/s, and >= 2 fingertip bodies within 2 cm of the object; safety violations reported separately",
        "model": {"form": "locally weighted ridge regression of the outcome on [state features, correction] (s1v2.OutcomeModel, unchanged)", **MODEL_KW,
                  "support": "kernel mass >= mass_min, else unknown with prior q0", "selection": "highest score among admissible candidates; within tie of the best, "
                  "the smallest correction", "s1v2_sha16": sha16(Path(V.__file__)), "local_exec_sha16": sha16(Path(LX.__file__))},
        "representations": {"none": "context-free (A1-A3): the same machinery, no state input",
                            "current": "S1-original's 27 state features", "geometry": "current + each fingertip body's gap to the outer wall (/R) and height above the base (/H)",
                            "post_geometry": "exploratory (designed after cup B): fingertip gap/height after translating the pre-closing fingertip positions by the candidate "
                                             "correction, + object size; computed before execution from observed quantities (admissible), see the audit in REPORT.md"},
        "primary_contextual_representation": "geometry",
        "methods": {c: n for c, n in LX.KINDS.items()},
        "retry_rules": {"standard (A0, A1)": "the same grasp again", "no_repeat": "a failed candidate is excluded for the rest of the scene (main comparison, A2-A5)",
                        "confirm_once": "a candidate is excluded after its second failure (experiment C)"},
        "update": {"frozen": "no target records", "current": "target record weight 1 (batch 2)", "conservative": "target record weight w < 1, chosen on development data (experiment C)"},
        "attempts_per_scene": 3,
        "metrics": ["first-attempt success", "success within 2 and 3 attempts", "attempts used (including unsuccessful scenes)", "candidates chosen",
                    "local decision latency", "variation across the three selections"],
        "comparisons": [{"a": a, "b": b, "question": q} for a, b, q in COMPARISONS],
        "locked_evaluation": {k: {"usd_sha256": v["usd_sha256"], "loaded": False} for k, v in locked["objects"].items()},
    }
    wj(p, m)
    print("manifest", p, sha16(p))


def cmd_expA(a):
    out = OUT / "mechanism"
    t0 = time.time()
    rows = run(["A0", "A1", "A2", "A3", "A4", "A5"], [64], ["geometry"], list(range(N_SEL)), dev_targets(), retry="no_repeat", tag="expA")
    with open(out / "expA_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "REPLAY with recorded development outcomes (one recorded execution per candidate and state)", "budget": 64,
           "by_method_set": {f"{c}|{s}": v for (c, s), v in table(rows, ["code", "target_set"]).items()},
           "by_method_set_selection": {f"{c}|{s}|{k}": v for (c, s, k), v in table(rows, ["code", "target_set", "selection"]).items()},
           "by_method_object": {f"{c}|{o}": v for (c, o), v in table(rows, ["code", "object"]).items()},
           "nominal_failing_only": {f"{c}|{s}": v for (c, s), v in table(rows, ["code", "target_set"], lambda r: not r["nominal_held"]).items()},
           "paired": {f"{x}-{y}": {"question": q, **{f"{s}_{key}": paired(rows, x, y, lambda r, s=s: r["target_set"] == s, key)
                                                    for s in ("primary", "additional") for key in ("first", "within3")}}
                      for x, y, q in COMPARISONS},
           "picks": {}, "wall_s": round(time.time() - t0, 1)}
    for r in rows:
        res["picks"].setdefault(f"{r['code']}|{r['object']}", {}).setdefault(f"{r['start']}|s{r['selection']}", [x["picked"] for x in r["attempts"]])
    wj(out / "expA_summary.json", res)
    for g, v in res["by_method_set"].items():
        print("A", g, v)
    for g, v in res["paired"].items():
        print("P", g, v)


def cmd_expB(a):
    out = OUT / "initialization"
    rows = run(["A0", "A2", "A4", "A5"], list(BUDGETS), ["geometry"], list(range(N_SEL)), dev_targets(), retry="no_repeat", tag="expB")
    with open(out / "expB_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "REPLAY with recorded development outcomes",
           "by_method_budget_set": {f"{c}|{b}|{s}": v for (c, b, s), v in table(rows, ["code", "budget", "target_set"]).items()},
           "by_method_budget_set_selection": {f"{c}|{b}|{s}|{k}": v for (c, b, s, k), v in table(rows, ["code", "budget", "target_set", "selection"]).items()},
           "nominal_failing_only": {f"{c}|{b}|{s}": v for (c, b, s), v in table(rows, ["code", "budget", "target_set"], lambda r: not r["nominal_held"]).items()}}
    wj(out / "expB_summary.json", res)
    for g, v in res["by_method_budget_set"].items():
        print("B", g, v)


def cmd_expC_weight(a):
    """Conservative update weight from development data: noisy replay (outcomes drawn from recorded executions)."""
    out = OUT / "mechanism"
    rows = []
    for retry in ("no_repeat", "confirm_once"):
        rows += run(["A4"], [64], ["geometry"], list(range(N_SEL)), dev_targets(), retry=retry, tag=f"frozen|{retry}", noisy_draws=a.draws)
        for w in (0.1, 0.25, 0.5, 1.0):
            rows += run(["A5"], [64], ["geometry"], list(range(N_SEL)), dev_targets(), retry=retry, update_w=w, tag=f"w{w}|{retry}", noisy_draws=a.draws)
    with open(out / "expC_weight_noisy_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "NOISY REPLAY (outcomes drawn from recorded executions of the same candidate at the same state)", "draws": a.draws,
           "by_tag_set": {f"{t}|{s}": v for (t, s), v in table(rows, ["tag", "target_set"]).items()},
           "by_tag": {t: v for (t,), v in table(rows, ["tag"]).items()}}
    cand = {w: res["by_tag"][f"w{w}|confirm_once"]["within_3"] for w in (0.1, 0.25, 0.5)}
    best = max(cand.values())
    tied = [w for w, v in cand.items() if best - v <= 0.005 * res["by_tag"]["w0.25|confirm_once"]["scenes"]]
    res["rule"] = ("conservative weight = the value in {0.1, 0.25, 0.5} with the most successes within 3 attempts under confirm_once, pooled over "
                   "development scenes, selections and draws; ties (within 0.5 % of the scenes) go to 0.25")
    res["chosen_conservative_weight"] = 0.25 if 0.25 in tied else tied[0]
    wj(out / "expC_weight_summary.json", res)
    for t, v in res["by_tag"].items():
        print("C", t, v)
    print("chosen", res["chosen_conservative_weight"])


def cmd_expD(a):
    out = OUT / "mechanism"
    rows = run(["A4", "A5"], [64], ["current", "geometry", "post_geometry"], list(range(N_SEL)), dev_targets(), retry="no_repeat", tag="expD")
    with open(out / "expD_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "REPLAY with recorded development outcomes",
           "by_method_rep_set": {f"{c}|{p}|{s}": v for (c, p, s), v in table(rows, ["code", "rep", "target_set"]).items()},
           "nominal_failing_only": {f"{c}|{p}|{s}": v for (c, p, s), v in table(rows, ["code", "rep", "target_set"], lambda r: not r["nominal_held"]).items()}}
    wj(out / "expD_summary.json", res)
    for g, v in res["by_method_rep_set"].items():
        print("D", g, v)


def cmd_grid(a):
    """Settings grid (decisions.json): kernel bandwidth x representation x budget, every development scene and selection (REPLAY)."""
    out = OUT / "mechanism"
    rows = []
    for h in (0.5, 1.0, 2.0, 4.0):
        rows += run(["A1", "A2", "A3"], [16, 64, 256], ["geometry"], list(range(N_SEL)), dev_targets(), tag=f"h{h}", mk={"h": h})
        rows += run(["A4", "A5"], [16, 64, 256], ["current", "geometry", "post_geometry"], list(range(N_SEL)), dev_targets(), tag=f"h{h}", mk={"h": h})
    with open(out / "grid_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "REPLAY settings grid", "by": {"|".join(map(str, g)): v for g, v in table(rows, ["code", "rep", "h", "budget", "target_set"]).items()},
           "pooled": {"|".join(map(str, g)): v for g, v in table(rows, ["code", "rep", "h", "budget"]).items()},
           "nominal_failing": {"|".join(map(str, g)): v for g, v in table(rows, ["code", "rep", "h", "budget"], lambda r: not r["nominal_held"]).items()}}
    dec = {}
    for fam, upd, frz, reps in (("context_free", "A3", "A2", ["none"]), ("contextual", "A5", "A4", ["current", "geometry"])):
        cands = []
        for rep in reps:
            for h in (0.5, 1.0, 2.0, 4.0):
                u, f_ = res["pooled"][f"{upd}|{rep}|{h}|64"], res["pooled"][f"{frz}|{rep}|{h}|64"]
                default = h == 1.0 and rep in ("none", "geometry")
                cands.append(((u["within_3"], f_["within_1"], default), rep, h))
        cands.sort(reverse=True)
        dec[fam] = {"rep": cands[0][1], "h": cands[0][2], "within_3": cands[0][0][0], "first": cands[0][0][1],
                    "ranking": [[c[1], c[2], c[0][0], c[0][1]] for c in cands]}
    res["chosen"] = dec
    wj(out / "grid_summary.json", res)
    print(json.dumps(dec, indent=1))


def cmd_expB_chosen(a):
    """Experiment B at the per-role settings of amendment 1 (replay): budgets 0 / 16 / 64 / 256, three selections, all 32 scenes."""
    out = OUT / "initialization"
    E = json.loads((OUT / "config" / "decisions.json").read_text())["amendment_1"]["chosen"]
    rows = run(["A1", "A2", "A3"], list(BUDGETS), ["none"], list(range(N_SEL)), dev_targets(), tag="cf", mk={"h": E["E2_family"]["h"]})
    rows += run(["A4", "A5"], list(BUDGETS), [E["E3"]["rep"]], list(range(N_SEL)), dev_targets(), tag="ctx", mk={"h": E["E3"]["h"]})
    rows += run(["A0"], [0], ["none"], [0], dev_targets(), tag="nominal")
    with open(out / "expB_chosen_replay.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"label": "REPLAY at the chosen settings", "settings": E,
           "pooled": {f"{c}|{b}": v for (c, b), v in table(rows, ["code", "budget"]).items()},
           "by_set": {f"{c}|{b}|{s_}": v for (c, b, s_), v in table(rows, ["code", "budget", "target_set"]).items()},
           "by_selection": {f"{c}|{b}|{k}": v for (c, b, k), v in table(rows, ["code", "budget", "selection"]).items()},
           "nominal_failing": {f"{c}|{b}": v for (c, b), v in table(rows, ["code", "budget"], lambda r: not r["nominal_held"]).items()}}
    rule = {}
    for fam, upd in (("E2_family", "A2"), ("E3", "A5")):
        w3 = {b: res["pooled"][f"{upd}|{b}"]["within_3"] for b in (0, 16, 64)}
        best = max(w3.values())
        rule[fam] = {"within_3": w3, "chosen_budget": min(b for b, v in w3.items() if v >= best - 3)}
    res["budget_rule"] = rule
    wj(out / "expB_chosen_summary.json", res)
    for c in ("A0", "A1", "A2", "A3", "A4", "A5"):
        print(c, " | ".join(f"B{b}: {res['pooled'][f'{c}|{b}']['within_1']}/{res['pooled'][f'{c}|{b}']['within_2']}/{res['pooled'][f'{c}|{b}']['within_3']} of {res['pooled'][f'{c}|{b}']['scenes']}"
                            for b in BUDGETS if f"{c}|{b}" in res["pooled"]))
    print(json.dumps(rule))


def cmd_online_summary(a):
    """Fresh executions (batch3 online): experiment A verification and experiment C, with the replay's picks for comparison."""
    out = OUT / "mechanism"
    res = {"label": "FRESH EXECUTIONS (every attempt a new simulated grasp-and-lift from the restored pre-closing state)"}
    ver = [json.loads(x) for f in sorted(out.glob("online_verify_*.jsonl")) for x in open(f) if x.strip()]
    for r in ver:
        r["code"], r["selection"] = r["spec"]["code"], r["spec"].get("selection", 0)
        r["nominal_held"] = None
    # nominal held at this fresh start <=> A0's first attempt held
    a0 = {(r["object"], r["start"]): r["attempts"][0]["success"] for r in ver if r["code"] == "A0"}
    for r in ver:
        r["nominal_held"] = a0.get((r["object"], r["start"]))
        r["target_set"] = "primary" if r["object"] in DEV_PRIMARY else "additional"
        r["draw"], r["budget"] = None, r["spec"].get("budget", 0)
    if ver:
        res["verify"] = {"by_method_set": {f"{c}|{s_}": v for (c, s_), v in table(ver, ["code", "target_set"]).items()},
                         "by_method": {c: v for (c,), v in table(ver, ["code"]).items()},
                         "by_method_object": {f"{c}|{o}": v for (c, o), v in table(ver, ["code", "object"]).items()},
                         "nominal_failing": {c: v for (c,), v in table(ver, ["code"], lambda r: r["nominal_held"] is False).items()},
                         "paired": {f"{x}-{y}": {"question": q, "first": paired(ver, x, y, key="first"), "within3": paired(ver, x, y, key="within3")}
                                    for x, y, q in COMPARISONS},
                         "scenes": len({(r["object"], r["start"]) for r in ver}), "runs": len(ver),
                         "rollouts": sum(r["attempts_used"] for r in ver),
                         "execution_s_median": round(float(np.median([x["execution_s"] for r in ver for x in r["attempts"] if "execution_s" in x])), 1),
                         "safety_violations": sum(1 for r in ver for x in r["attempts"] if x.get("safety_violation"))}
    ec = [json.loads(x) for f in sorted(out.glob("online_expC_*.jsonl")) for x in open(f) if x.strip()]
    if ec:
        for r in ec:
            sp = r["spec"]
            r["cond"] = f"{sp['code']}|{'frozen' if sp['code'] in ('A2', 'A4') else 'w' + str(sp['update_w'])}|{sp['retry']}"
            r["code"], r["selection"], r["draw"], r["budget"] = sp["code"], sp["selection"], None, sp["budget"]
        # a candidate 'works' at a start if any fresh execution of it at that start held (any condition)
        works = {}
        for r in ec:
            for x in r["attempts"]:
                if x["picked"] is not None:
                    works.setdefault((r["object"], r["start"], x["picked"]), []).append(bool(x["success"]))
        cond = {}
        for r in ec:
            c = cond.setdefault(r["cond"], {"scenes": 0, "within_1": 0, "within_3": 0, "attempts": 0, "retries": 0, "retry_success": 0,
                                            "abandoned_working_after_one_failure": 0, "changed_after_failure": 0, "changed_and_improved": 0})
            c["scenes"] += 1
            c["within_1"] += r["first_success_attempt"] == 1
            c["within_3"] += r["first_success_attempt"] is not None
            c["attempts"] += r["attempts_used"]
            at = [x for x in r["attempts"] if x["picked"] is not None]
            for i in range(1, len(at)):
                prev, cur = at[i - 1], at[i]
                if prev["success"]:
                    continue
                if cur["picked"] == prev["picked"]:
                    c["retries"] += 1
                    c["retry_success"] += bool(cur["success"])
                else:
                    c["changed_after_failure"] += 1
                    c["changed_and_improved"] += bool(cur["success"])
                    if any(works.get((r["object"], r["start"], prev["picked"]), [])):
                        c["abandoned_working_after_one_failure"] += 1
        res["expC"] = {"by_condition": cond, "runs": len(ec), "rollouts": sum(r["attempts_used"] for r in ec),
                       "flips": {f"{o}|{st}|{cid}": v for (o, st, cid), v in works.items() if len(set(v)) > 1}}
    wj(out / "online_summary.json", res)
    for k in ("verify", "expC"):
        if k in res:
            print(k, json.dumps({kk: vv for kk, vv in res[k].items() if kk in ("by_method", "nominal_failing", "by_condition", "runs", "rollouts", "flips")}, indent=None)[:3000])


def proposal_scenes(stage="mechanism"):
    """Banks recorded at Astra's own proposals (batch3 pbank): {(object, start): {"state", "rows", "repeats": [], "r2": []}}."""
    out = {}
    for d in sorted((OUT / stage / "proposal_banks").glob("*/*")):
        if (d / "state.json").exists() and (d / "rollouts.jsonl").exists():
            rows = [json.loads(x) for x in open(d / "rollouts.jsonl") if x.strip()]
            if len(rows) >= 31:
                out[(d.parent.name, d.name)] = {"state": json.loads((d / "state.json").read_text()), "rows": rows, "repeats": [], "r2": []}
    return out


def cmd_pilot_replay(a):
    """Experiment D, interface pilot: every method replayed on the banks recorded at Astra's actual proposals (development cups)."""
    E = json.loads((OUT / "config" / "decisions.json").read_text())["amendment_1"]["chosen"]
    ps = proposal_scenes()
    for (o, st), S in ps.items():
        _SCENES.setdefault(f"proposal:{o}", {})[st] = S
    targets = [("astra_proposal", f"proposal:{o}", st) for (o, st) in ps]
    rows = []
    for (tset, okey, st) in targets:
        o = okey.split(":", 1)[1]
        S = _SCENES[okey][st]
        nominal_ok = next(r for r in S["rows"] if r["candidate"]["id"] == "nominal")["outcome"]["task_success"]
        for k in range(N_SEL):
            for code in ("A0", "A1", "A2", "A3", "A4", "A5"):
                if code == "A0" and k:
                    continue
                cf = code in ("A1", "A2", "A3")
                m = method(code, k, 64, "none" if cf else E["E3"]["rep"], o, mk={"h": E["E2_family"]["h"] if cf else E["E3"]["h"]})
                rep = LX.run_scene(m, S["state"], replay_exec(S))
                rows.append({"target_set": tset, "object": o, "start": st, "nominal_held": nominal_ok, "budget": 64, "selection": k, "code": code, "draw": None,
                             **{kk: rep[kk] for kk in ("status", "success", "first_success_attempt", "attempts_used", "attempts")}})
    held = {f"{o}|{st}": sorted(r["candidate"]["id"] for r in S["rows"] if r["outcome"]["task_success"]) for (o, st), S in ps.items()}
    drift = {f"{o}|{st}": S["state"].get("replay_drift") for (o, st), S in ps.items()}
    res = {"label": "REPLAY on banks executed at Astra's actual proposals (development cups, fresh starts)", "scenes": len(ps),
           "by_method": {c: v for (c,), v in table(rows, ["code"]).items()}, "by_method_object": {f"{c}|{o}": v for (c, o), v in table(rows, ["code", "object"]).items()},
           "holding_corrections": held, "replay_drift": drift,
           "picks": {f"{r['code']}|{r['object']}|{r['start']}|s{r['selection']}": [x["picked"] for x in r["attempts"]] for r in rows}}
    wj(OUT / "mechanism" / "pilot_replay_summary.json", res)
    for c, v in res["by_method"].items():
        print(c, v)
    print(json.dumps(drift))


def cmd_tables(a):
    """REPORT_DATA.md from every summary written so far (generated; numbers only)."""
    L = ["# Delegation study: data tables (generated " + time.strftime("%Y-%m-%d %H:%M") + ")", ""]
    rj = lambda p: json.loads(p.read_text()) if p.exists() else None
    names = {"A0": "nominal", "A1": "one correction", "A2": "ranked search", "A3": "context-free updating", "A4": "contextual frozen", "A5": "contextual updating"}
    cell = lambda v: f"{v['within_1']} / {v['within_2']} / {v['within_3']} of {v['scenes']}" if v else "-"
    A = rj(OUT / "mechanism" / "expA_summary.json")
    if A:
        L += ["## Experiment A, REPLAY, pre-registered settings (h = 1, geometry representation), 64 source interactions",
              "", "first / within 2 / within 3 attempts, scenes x 3 selections (A0: scenes)", "",
              "| method | primary (A, B, C) | additional (dev family, own records removed) | nominal-failing only, primary | nominal-failing only, additional |", "|---|---|---|---|---|"]
        for c in names:
            L.append(f"| {c} {names[c]} | {cell(A['by_method_set'].get(f'{c}|primary'))} | {cell(A['by_method_set'].get(f'{c}|additional'))} | "
                     f"{cell(A['nominal_failing_only'].get(f'{c}|primary'))} | {cell(A['nominal_failing_only'].get(f'{c}|additional'))} |")
        L += ["", "Paired (scene x selection; a better / b better / same):", "", "| comparison | question | primary, first | primary, within 3 | additional, first | additional, within 3 |", "|---|---|---|---|---|---|"]
        for k, v in A["paired"].items():
            f = lambda d: f"{d['a_better']} / {d['b_better']} / {d['same']}"
            L.append(f"| {k} | {v['question']} | {f(v['primary_first'])} | {f(v['primary_within3'])} | {f(v['additional_first'])} | {f(v['additional_within3'])} |")
    G = rj(OUT / "mechanism" / "grid_summary.json")
    if G:
        L += ["", "## Settings grid, REPLAY (all 32 development scenes x 3 selections = 96): first / within 3", "",
              "| method | representation | h | 16 | 64 | 256 |", "|---|---|---|---|---|---|"]
        for c, reps in (("A1", ["none"]), ("A2", ["none"]), ("A3", ["none"]), ("A4", ["current", "geometry", "post_geometry"]), ("A5", ["current", "geometry", "post_geometry"])):
            for rep in reps:
                for h in (0.5, 1.0, 2.0, 4.0):
                    L.append(f"| {c} | {rep} | {h} | " + " | ".join(f"{G['pooled'][f'{c}|{rep}|{h}|{b}']['within_1']} / {G['pooled'][f'{c}|{rep}|{h}|{b}']['within_3']}"
                                                                     for b in (16, 64, 256)) + " |")
    B = rj(OUT / "initialization" / "expB_chosen_summary.json")
    if B:
        L += ["", "## Experiment B, REPLAY at the chosen settings (context-free h = %s; contextual %s, h = %s): first / within 2 / within 3 of 96" % (
            B["settings"]["E2_family"]["h"], B["settings"]["E3"]["rep"], B["settings"]["E3"]["h"]), "",
              "| method | 0 | 16 | 64 | 256 |", "|---|---|---|---|---|"]
        for c in names:
            L.append(f"| {c} {names[c]} | " + " | ".join(cell(B["pooled"].get(f"{c}|{b}")) for b in BUDGETS) + " |")
        L += ["", "Per selection (within 3 of 32), to show the spread:", "", "| method | budget | selection 0 | selection 1 | selection 2 |", "|---|---|---|---|---|"]
        for c in ("A2", "A4", "A5"):
            for b in (16, 64, 256):
                L.append(f"| {c} | {b} | " + " | ".join(f"{B['by_selection'][f'{c}|{b}|{k}']['within_1']} / {B['by_selection'][f'{c}|{b}|{k}']['within_3']}" for k in range(3)) + " |")
        L += ["", f"Budget rule: {json.dumps(B['budget_rule'])}"]
    D = rj(OUT / "mechanism" / "expD_summary.json")
    if D:
        L += ["", "## Experiment D, REPLAY, representations at the pre-registered h = 1, 64 interactions", "", "| method | representation | primary | additional |", "|---|---|---|---|"]
        for c in ("A4", "A5"):
            for rep in ("current", "geometry", "post_geometry"):
                L.append(f"| {c} | {rep} | {cell(D['by_method_rep_set'].get(f'{c}|{rep}|primary'))} | {cell(D['by_method_rep_set'].get(f'{c}|{rep}|additional'))} |")
    C = rj(OUT / "mechanism" / "expC_weight_summary.json")
    if C:
        L += ["", f"## Experiment C, conservative weight from development data: NOISY REPLAY ({C['draws']} draws per scene and selection)", "",
              "| update | retry rule | first | within 2 | within 3 | of |", "|---|---|---|---|---|---|"]
        for t, v in C["by_tag"].items():
            u, r = t.split("|")
            L.append(f"| {u} | {r} | {v['within_1']} | {v['within_2']} | {v['within_3']} | {v['scenes']} |")
        L += ["", f"Chosen conservative weight: {C['chosen_conservative_weight']} ({C['rule']})"]
    O = rj(OUT / "mechanism" / "online_summary.json")
    if O and "verify" in O:
        V_ = O["verify"]
        L += ["", f"## Experiment A, FRESH EXECUTIONS at the chosen settings, 64 interactions ({V_['scenes']} scenes, {V_['runs']} runs, {V_['rollouts']} rollouts)", "",
              "| method | all scenes x selections | nominal-failing scenes only |", "|---|---|---|"]
        for c in names:
            L.append(f"| {c} {names[c]} | {cell(V_['by_method'].get(c))} | {cell(V_['nominal_failing'].get(c))} |")
        L += ["", "| comparison | first: a better / b better / same | within 3 |", "|---|---|---|"]
        for k, v in V_["paired"].items():
            L.append(f"| {k} | {v['first']['a_better']} / {v['first']['b_better']} / {v['first']['same']} | {v['within3']['a_better']} / {v['within3']['b_better']} / {v['within3']['same']} |")
        L += ["", "Per object (first / within 3 of runs):", "", "| method | " + " | ".join(sorted({k.split('|')[1] for k in V_["by_method_object"]})) + " |",
              "|---|" + "---|" * len({k.split('|')[1] for k in V_["by_method_object"]})]
        for c in names:
            L.append(f"| {c} | " + " | ".join((lambda v: f"{v['within_1']}/{v['within_3']} of {v['scenes']}" if v else "-")(V_["by_method_object"].get(f"{c}|{o}"))
                                              for o in sorted({k.split('|')[1] for k in V_["by_method_object"]})) + " |")
    if O and "expC" in O:
        C2 = O["expC"]
        L += ["", f"## Experiment C, FRESH EXECUTIONS ({C2['runs']} runs, {C2['rollouts']} rollouts)", "",
              "| method / update / retry | scenes | first | within 3 | retries (succeeded) | changed after a failure (then held) | abandoned a candidate that held elsewhere at this start |",
              "|---|---|---|---|---|---|---|"]
        for k, v in sorted(C2["by_condition"].items()):
            L.append(f"| {k} | {v['scenes']} | {v['within_1']} | {v['within_3']} | {v['retries']} ({v['retry_success']}) | {v['changed_after_failure']} ({v['changed_and_improved']}) | {v['abandoned_working_after_one_failure']} |")
        L += ["", "Candidates with both outcomes at one start (fresh executions): " + (json.dumps(C2["flips"]) if C2["flips"] else "none")]
    PR = rj(OUT / "mechanism" / "pilot_replay_summary.json")
    if PR:
        L += ["", f"## Experiment D, interface pilot: REPLAY on banks executed at Astra's actual proposals ({PR['scenes']} scenes)", "",
              "| method | first / within 2 / within 3 |", "|---|---|"]
        for c in names:
            L.append(f"| {c} {names[c]} | {cell(PR['by_method'].get(c))} |")
        L += ["", "Holding corrections per proposal: " + json.dumps(PR["holding_corrections"]), "", "Replay drift: " + json.dumps(PR["replay_drift"])]
    for setname, title in (("pilot", "Development pilot of the compared systems (Astra proposals on development cups)"), ("locked", "Experiment E, PROSPECTIVE (locked cups)")):
        S_ = rj(OUT / ("prospective" if setname == "locked" else "mechanism") / f"systems_{setname}_summary.json")
        if S_:
            L += ["", f"## {title}: {S_['scenes']} scenes", "", "| system | first | within 2 | within 3 | within 120 s | within 240 s | mean scene time (s) | Astra calls / scene | decision s / scene | proposal failures |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for sy, v in S_["systems"].items():
                L.append(f"| {sy} | {v['first']} | {v['within_2']} | {v['within_3']} | {v['within_120s']} | {v['within_240s']} | {v['scene_time_s_mean_all_scenes']} | {v['astra_calls_mean']} | {v['decision_s_mean']} | {v['proposal_failures']} |")
            L += ["", "| pair | key | a better | b better | sign test p |", "|---|---|---|---|---|"]
            for k, v in S_["paired"].items():
                pr_, key = k.split("|")
                L.append(f"| {pr_} | {key} | {v['a_better']} | {v['b_better']} | {v['sign_test_p']} |")
    (OUT / "REPORT_DATA.md").write_text("\n".join(L) + "\n")
    print("REPORT_DATA.md", len(L), "lines")


def sign_test(a_better, b_better):
    """Two-sided exact sign test on the discordant pairs (McNemar exact)."""
    from math import comb
    n, k = a_better + b_better, min(a_better, b_better)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def systems_summary(stage, label, budgets_s=(120, 240)):
    """Compared systems (experiment E or the development pilot): per system first-attempt / within-3 / within wall-clock budgets,
    costs, per object, paired comparisons. Scene time = the shared proposal's wall time + every attempt's decision, execution and reset."""
    rows = [json.loads(x) for f in sorted((OUT / stage).glob("systems_*.jsonl")) for x in open(f) if x.strip()]
    props = {}
    for f in sorted((OUT / stage / "proposals").glob("*/*/proposal.json")):
        pr = json.loads(f.read_text())
        props[(pr["object"], pr["start"])] = pr
    systems = sorted({r["system"] for r in rows})
    scenes = sorted(props)
    out = {"label": label, "scenes": len(scenes), "proposals": {f"{o}|{st}": {k: p[k] for k in ("status", "astra_calls", "astra_latency_s", "wall_s")}
                                                               for (o, st), p in props.items()}, "systems": {}}
    per = {}
    for r in rows:
        t = props[(r["object"], r["start"])]["wall_s"]
        tl, ok_at = [], None
        for x in r["attempts"]:
            t += x["decision_s"] + x.get("execution_s", 0.0) + x.get("reset_s", 0.0)
            tl.append(round(t, 1))
            if x.get("success") and ok_at is None:
                ok_at = t
        per[(r["object"], r["start"], r["system"])] = {"first": r["first_success_attempt"] == 1, "within3": r["first_success_attempt"] is not None,
                                                      "attempt": r["first_success_attempt"], "time_to_success_s": None if ok_at is None else round(ok_at, 1),
                                                      "scene_time_s": round(t, 1), "astra_calls": props[(r["object"], r["start"])]["astra_calls"] + sum(
                                                          (x.get("calls") or 0) for x in r["attempts"][-1:]) if r["system"] in ("E0", "E1") else
                                                      props[(r["object"], r["start"])]["astra_calls"],
                                                      "decision_s": round(sum(x["decision_s"] for x in r["attempts"]), 3),
                                                      "execution_s": round(sum(x.get("execution_s", 0.0) for x in r["attempts"]), 1),
                                                      "reset_s": round(sum(x.get("reset_s", 0.0) for x in r["attempts"]), 2),
                                                      "picks": [(x["picked"], x.get("success")) for x in r["attempts"]], "status": r["status"]}
    for sy in systems:
        v = [per.get((o, st, sy)) for (o, st) in scenes]
        failed_proposal = sum(1 for (o, st) in scenes if props[(o, st)]["status"] != "handoff")
        got = [x for x in v if x is not None]
        n = len(scenes)
        d = {"scenes": n, "proposal_failures": failed_proposal, "first": sum(x["first"] for x in got), "within_3": sum(x["within3"] for x in got),
             "within_2": sum(1 for x in got if x["attempt"] is not None and x["attempt"] <= 2)}
        for b in budgets_s:
            d[f"within_{b}s"] = sum(1 for x in got if x["time_to_success_s"] is not None and x["time_to_success_s"] <= b)
        d["scene_time_s_mean_all_scenes"] = round(float(np.mean([x["scene_time_s"] for x in got])), 1) if got else None
        d["astra_calls_mean"] = round(float(np.mean([x["astra_calls"] for x in got])), 2) if got else None
        d["decision_s_mean"] = round(float(np.mean([x["decision_s"] for x in got])), 3) if got else None
        d["execution_s_mean"] = round(float(np.mean([x["execution_s"] for x in got])), 1) if got else None
        d["attempts_mean"] = round(float(np.mean([len(x["picks"]) for x in got])), 2) if got else None
        d["per_object"] = {o: {"first": sum(per[(oo, st, sy)]["first"] for (oo, st) in scenes if oo == o and (oo, st, sy) in per),
                               "within_3": sum(per[(oo, st, sy)]["within3"] for (oo, st) in scenes if oo == o and (oo, st, sy) in per),
                               "scenes": sum(1 for (oo, st) in scenes if oo == o)} for o in sorted({o for o, _ in scenes})}
        out["systems"][sy] = d
    pairs = {}
    for i, a_ in enumerate(systems):
        for b_ in systems[i + 1:]:
            for key in ("first", "within3"):
                ab = sum(1 for (o, st) in scenes if (o, st, a_) in per and (o, st, b_) in per and per[(o, st, a_)][key] and not per[(o, st, b_)][key])
                ba = sum(1 for (o, st) in scenes if (o, st, a_) in per and (o, st, b_) in per and per[(o, st, b_)][key] and not per[(o, st, a_)][key])
                pairs[f"{a_}-{b_}|{key}"] = {"a_better": ab, "b_better": ba, "sign_test_p": round(sign_test(ab, ba), 4)}
    out["paired"] = pairs
    out["per_scene"] = {f"{o}|{st}|{sy}": v for (o, st, sy), v in per.items()}
    return out


def cmd_systems_summary(a):
    stage = "prospective" if a.set == "locked" else "mechanism"
    res = systems_summary(stage, "PROSPECTIVE (locked cups)" if a.set == "locked" else "DEVELOPMENT PILOT (Astra proposals on development cups)")
    wj(OUT / stage / f"systems_{a.set}_summary.json", res)
    for sy, v in res["systems"].items():
        print(sy, {k: v[k] for k in v if k != "per_object"})
    for k, v in res["paired"].items():
        print(k, v)


def cmd_prospective_config(a):
    """Freezes the prospective batch (experiment E) from the development decisions: systems, starts, caps, metrics. Written once."""
    p = OUT / "config" / "prospective.json"
    if p.exists() and not a.force:
        raise SystemExit(f"already frozen: {p}")
    dec = json.loads((OUT / "config" / "decisions.json").read_text())
    E = dec["amendment_1"]["chosen"]
    onl = json.loads((OUT / "mechanism" / "online_summary.json").read_text())["expC"]["by_condition"]
    pool = lambda w: {r: onl.get(f"A5|w{w}|{r}", {}).get("within_3", 0) for r in ("no_repeat", "confirm_once")}
    cons, cur = pool(0.25), pool(1.0)
    w = 0.25 if sum(cons.values()) > sum(cur.values()) else 1.0
    chosen_pool = cons if w == 0.25 else cur
    retry = "confirm_once" if chosen_pool["confirm_once"] > chosen_pool["no_repeat"] else "no_repeat"
    rng = np.random.default_rng(20261016)
    xy = np.round(rng.uniform(-0.02, 0.02, (20, 2)), 4)
    starts = {f"l{i + 1:02d}": xy[i].tolist() for i in range(20)}
    budget = dec["amendment_2"]["budget_rule_result"]
    cfg = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "FROZEN before any locked-cup load: every setting below comes from development data by the rules in decisions.json",
        "objects": {"cup_t1": [f"l{i:02d}" for i in range(1, 11)], "cup_t2": [f"l{i:02d}" for i in range(11, 21)]},
        "starts": starts, "starts_rule": "numpy.random.default_rng(20261016).uniform(-0.02, 0.02, (20, 2)) object offsets (m); l01-l10 cup_t1, l11-l20 cup_t2",
        "proposal": {"who": "Astra (gpt-6-astra, reasoning medium, timeout 180 s) through the Codex CLI harness", "prompt": "the cup-transfer decision prompt and schema "
                     "(prompt_cup.py), Astra's own history from the cup A source run", "handoff": "Astra's first decision that closes the active hand is not "
                     "executed; it is the grasp proposal, the state before it the pre-closing state", "cap_calls": 8, "render": "full (as the cup-transfer runs)",
                     "failure": "no hand-off within the cap (or stop) = the scene is unsuccessful for every system, its cost counted"},
        "local_program": "Astra's closing decision, then the existing program's lift (+9.32 cm, 30 steps) and hold (45 steps), then the 40-step evaluation window",
        "systems": {"selection": 0, "E1_records_budget": budget["E3"]["chosen_budget"], "astra_cap": 8,
                    "E0": "Astra chooses the correction (bank, measured pre-closing geometry, this scene's attempts, its running memory); no images",
                    "E1": "E0 + the structured experience records (the same source records E2/E3 start from, readable)",
                    "E2": {"code": dec["amendment_2"]["e2_rule_result"]["E2"], "rep": "none", "h": E["E2_family"]["h"], "budget": budget["E2_family"]["chosen_budget"],
                           "selection": 0, "retry": retry, "update_w": 1.0},
                    "E3": {"code": "A5", "rep": E["E3"]["rep"], "h": E["E3"]["h"], "budget": budget["E3"]["chosen_budget"], "selection": 0, "retry": retry, "update_w": w}},
        "experiment_C_basis": {"A5_w0.25_within3": cons, "A5_w1.0_within3": cur, "chosen_update_w": w, "chosen_retry": retry},
        "scene_protocol": "the four systems of a scene run in one no-render simulator that replays Astra's executed proposal decisions from the same start "
                          "(drift recorded), in a seeded random order per scene; each attempt restores the pre-closing snapshot (reset-based adaptation); "
                          "at most 3 grasp attempts; same bank (nominal + 30 corrections), same task-aligned success and failure report; 3 concurrent simulators",
        "metrics": {"primary": ["first-attempt success", "success within 3 attempts",
                                "success within wall-clock budgets of 120 s and 240 s (scene time = the shared proposal's wall time + every attempt's decision, execution and reset time)"],
                    "secondary": ["success after each attempt", "Astra calls and waiting time", "local decision time", "execution and reset time",
                                  "unsuccessful scenes", "per-object results", "replay drift", "source interactions"]},
        "statistics": "paired by scene; discordant counts with two-sided exact sign tests; two locked geometries = a targeted validation, not broad evidence; "
                      "small unresolved differences are reported as unresolved, never as equivalence",
        "recommendation_table": "the user's plan, section 11 (written into REPORT.md before the locked results are read)",
    }
    wj(p, cfg)
    print(json.dumps(cfg["systems"], indent=1), json.dumps(cfg["experiment_C_basis"]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--force", action="store_true")
    sub.add_parser("expA")
    sub.add_parser("expB")
    p = sub.add_parser("expC_weight")
    p.add_argument("--draws", type=int, default=10)
    sub.add_parser("expD")
    sub.add_parser("grid")
    sub.add_parser("online_summary")
    sub.add_parser("expB_chosen")
    sub.add_parser("pilot_replay")
    sub.add_parser("tables")
    p = sub.add_parser("prospective_config")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("systems_summary")
    p.add_argument("--set", default="pilot")
    a = ap.parse_args()
    {"manifest": cmd_manifest, "expA": cmd_expA, "expB": cmd_expB, "expC_weight": cmd_expC_weight, "expD": cmd_expD, "grid": cmd_grid,
     "online_summary": cmd_online_summary, "expB_chosen": cmd_expB_chosen,
     "systems_summary": cmd_systems_summary, "prospective_config": cmd_prospective_config,
     "pilot_replay": cmd_pilot_replay, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
