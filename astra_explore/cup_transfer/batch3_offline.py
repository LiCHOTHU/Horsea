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
    a = ap.parse_args()
    {"manifest": cmd_manifest, "expA": cmd_expA, "expB": cmd_expB, "expC_weight": cmd_expC_weight, "expD": cmd_expD, "grid": cmd_grid,
     "online_summary": cmd_online_summary, "expB_chosen": cmd_expB_chosen}[a.cmd](a)


if __name__ == "__main__":
    main()
