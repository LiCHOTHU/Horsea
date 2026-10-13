"""Batch 2, the parts that need no simulator (system python with numpy).

    python -m cup_transfer.batch2_offline exp0        Experiment 0 (offline half): what "frozen" and "adaptive" did to memory,
                                                      whether every update reached the next prediction (replayed from the
                                                      cup-transfer logs), and a controlled insertion into a disposable copy
Outputs: experiments/cup_batch2_2026-10-13/exp0/.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cup_transfer import system1 as S1

TR = Path("/home/licho/workspace/Horsea/experiments/cup_transfer_2026-10-12")
B1 = Path("/home/licho/workspace/Horsea/experiments/cup_batch1_2026-10-13")
OUT = Path("/home/licho/workspace/Horsea/experiments/cup_batch2_2026-10-13")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def hyper():
    return json.loads((TR / "method_configurations" / "hyper.json").read_text())["best"]


def replay_updates(cond_dir, D_A, hp):
    """Rebuild the store exactly as the run did (D_A + every record added, in order) and, for every supported plan,
    (a) recompute its logged one-segment prediction from the store as it was at that moment, (b) recompute it without
    the most recent record. (a) = the logged value means the decision used the updated memory; (b) != (a) means the
    latest update changed the next prediction."""
    store = S1.ExperienceStore(copy.deepcopy(D_A.records))
    rows_out = []
    for k in (1, 2):
        f = cond_dir / f"attempt_{k}" / "transitions.jsonl"
        if not f.exists():
            continue
        for r in jl(f):
            p = r["plan"]
            if p.get("fallback") is None and p.get("predicted_ds") is not None and len(store) > len(D_A.records):
                s, u, ctx = np.array(r["s"]), np.array(r["applied_command_after_limits"]), r["interaction_context"]
                m = S1.LocalExecutionModel(store, h=hp["h"], lam=hp["lam"])
                fit = m.fit(s, u, ctx)
                prev = S1.ExperienceStore(store.records[:-1])
                fit0 = S1.LocalExecutionModel(prev, h=hp["h"], lam=hp["lam"]).fit(s, u, ctx)
                if fit is not None:
                    pa = S1.LocalExecutionModel.predict(fit, s, u)
                    pb = None if fit0 is None else S1.LocalExecutionModel.predict(fit0, s, u)
                    rows_out.append({"attempt": k, "decision": r["decision"], "segment": r["segment"], "mode": ctx["mode"],
                                     "logged_vs_replayed_max_abs": float(np.max(np.abs(np.array(p["predicted_ds"]) - pa))),
                                     "last_update_effect_rms": None if pb is None else float(np.sqrt(np.mean((pa - pb) ** 2))),
                                     "last_record_used_in_fit": bool((len(store.records) - 1) in set(fit["idx"].tolist()))})
            if r.get("added_to_memory"):
                store.add({k2: r[k2] for k2 in S1.ExperienceStore.FIELDS + ("s", "u_applied", "ds", "provenance")})
    return rows_out


def cmd_exp0(a):
    out = OUT / "exp0"
    out.mkdir(parents=True, exist_ok=True)
    D_A = S1.ExperienceStore.load(TR / "source_interaction_dataset" / "run" / "D_A.jsonl")
    hp = hyper()
    res = {"meaning": {}, "update_replay": {}, "insertion_test": {}}
    # 1. what each arm did to memory (logs)
    for arm in ("frozen_model", "adaptive_model", "empty_model", "nearest_neighbor", "astra_history"):
        seg = added = 0
        sizes = []
        for f in (TR / "per_attempt_transition_logs" / "run").glob(f"*/*/{arm}/attempt_*/transitions.jsonl"):
            rows = jl(f)
            seg += len(rows)
            added += sum(r.get("added_to_memory", False) for r in rows)
        for f in (TR / "per_attempt_transition_logs" / "run").glob(f"*/*/{arm}/memory_after.jsonl"):
            sizes.append(sum(1 for _ in open(f)))
        res["meaning"][arm] = {"segments": seg, "records_added": added, "memory_sizes_after_conditions": sorted(set(sizes))}
    res["meaning"]["code"] = ("run_cup.make_method: frozen_model = ExperienceStore(D_A, frozen=True) with update=False (add() refuses, "
                              "nothing is written; every query refits the local model on the same 123 records); adaptive_model / "
                              "empty_model = frozen=False, update=True: after EVERY executed segment the transition is appended "
                              "(cup_task.execute_decision, after its prediction error was recorded) and the next segment's local "
                              "fit already includes it (the store's cache is reset on add). Updates are online, within the attempt, "
                              "never batched per episode.")
    # 2. replay: did every update reach the next prediction?
    rows_all = []
    for cond in sorted((TR / "per_attempt_transition_logs" / "run").glob("*/*/adaptive_model")):
        rows = replay_updates(cond, D_A, hp)
        rows_all += [dict(r, condition=str(cond.relative_to(TR / "per_attempt_transition_logs" / "run"))) for r in rows]
    eff = [r["last_update_effect_rms"] for r in rows_all if r["last_update_effect_rms"] is not None]
    res["update_replay"] = {"supported_plans_checked": len(rows_all),
                            "max_abs_logged_minus_replayed_prediction": max((r["logged_vs_replayed_max_abs"] for r in rows_all), default=None),
                            "plans_whose_fit_used_the_latest_record": sum(r["last_record_used_in_fit"] for r in rows_all),
                            "latest_update_changed_the_prediction (rms > 1e-6)": sum(e > 1e-6 for e in eff),
                            "median_effect_rms": float(np.median(eff)) if eff else None}
    (out / "update_replay.jsonl").write_text("\n".join(json.dumps(r) for r in rows_all) + "\n")
    # 3. controlled insertion into a disposable copy (batch-1 state C2 and its rescuing candidate 13)
    st = json.loads((B1 / "phase2_3" / "C2.json").read_text())
    s0 = np.array(st["state_vector"])
    ctx = {"mode": st["mode"], "near": st["near"], "hand": "right"}
    cands = {c["i"]: c for c in st["candidates"]}
    c13, c0 = cands[13], cands[0]
    u13, u0 = np.array(c13["u_app"][0]), np.array(c0["u_app"][0])
    real_ds = np.array(c13["branches"][0]["realized_ds_first"])
    ctrl_before = S1.ShortHorizonController(S1.ExperienceStore(copy.deepcopy(D_A.records), frozen=True), h=hp["h"], lam=hp["lam"])
    fit_b = ctrl_before.model.fit(s0, u0, ctx)
    pred_b = S1.LocalExecutionModel.predict(fit_b, s0, u13)
    probe = copy.deepcopy(D_A.records)
    for j in range(12):        # a cluster of test records at (s0, u13) with candidate 13's measured change
        r = copy.deepcopy(D_A.records[0])
        r.update(s=s0.tolist(), u_applied=u13.tolist(), ds=real_ds.tolist(), interaction_context=dict(ctx), object_instance_id="TEST",
                 episode_id="controlled_insertion", provenance="test")
        probe.append(r)
    ctrl_after = S1.ShortHorizonController(S1.ExperienceStore(probe, frozen=True), h=hp["h"], lam=hp["lam"])
    fit_a = ctrl_after.model.fit(s0, u0, ctx)
    pred_a = S1.LocalExecutionModel.predict(fit_a, s0, u13)
    names = S1.DELTA_NAMES
    res["insertion_test"] = {"state": "batch-1 C2 (cup B, mid-closing)", "candidate": 13,
                             "inserted": "12 copies of (state, candidate 13's applied command, candidate 13's measured change) into a disposable copy of D_A",
                             "object_xy_change_pred_before": [round(float(pred_b[11]), 4), round(float(pred_b[12]), 4)],
                             "object_xy_change_pred_after": [round(float(pred_a[11]), 4), round(float(pred_a[12]), 4)],
                             "object_xy_change_measured": [round(float(real_ds[11]), 4), round(float(real_ds[12]), 4)],
                             "rms_to_measured_before": round(float(np.sqrt(np.mean((pred_b - real_ds) ** 2))), 4),
                             "rms_to_measured_after": round(float(np.sqrt(np.mean((pred_a - real_ds) ** 2))), 4),
                             "fit_used_test_records": int(sum(1 for i in fit_a["idx"] if probe[i].get("provenance") == "test")),
                             "D_A_unchanged": len(D_A.records) == 123, "delta_names": names}
    (out / "exp0_offline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "meaning"}, indent=1))
    print(json.dumps(res["meaning"], indent=1)[:1500])


# ============================================================================ experiments 1-5 (offline analyses)
ALPHA, BETA = 0.5, 0.2
POINTS_ORDER = ("approach", "pre_close", "mid_close")


def exp1_table():
    rows = jl(OUT / "exp1" / "rollouts.jsonl") if (OUT / "exp1" / "rollouts.jsonl").exists() else []
    tab = {}
    for r in rows:
        k = (r["point"], r["condition"])
        tab.setdefault(k, []).append((r["start"], r["outcome"]["task_success"], r["outcome"]["max_rise_cm"], r["outcome"]["safety_violation"]))
    return rows, tab


def states(obj):
    d = OUT / "exp2" / obj
    out = {}
    for f in sorted(d.glob("state_d*.json")):
        if "_r" in f.stem.split("state_")[1] or f.stem.endswith("_vec"):
            continue
        st = json.loads(f.read_text())
        vf = d / f"state_{st['start']}_vec.json"
        if vf.exists():                     # the same state re-recorded with per-tip vectors (deterministic prefix)
            st["geometry_before"] = json.loads(vf.read_text())["geometry_before"]
        rows = [x for x in jl(d / f"rollouts_{st['start']}.jsonl") if x["rep"] == 0] if (d / f"rollouts_{st['start']}.jsonl").exists() else []
        if len(rows) >= 32:
            out[st["start"]] = {"state": st, "rows": rows,
                                "repeats": jl(d / f"repeats_{st['start']}.jsonl") if (d / f"repeats_{st['start']}.jsonl").exists() else [],
                                "r2": jl(d / f"rollouts_{st['start']}_r2.jsonl") if (d / f"rollouts_{st['start']}_r2.jsonl").exists() else []}
    return out


def j_existing_real(r):
    """S1-original's objective (mode 'close' at the intervention) on the MEASURED first three segments + its penalties."""
    f3 = r["first3"]
    if len(f3) < 3:
        return None
    R = float(f3[0]["before"]["R"])
    traj = [(S1.state_vector(f3[0]["before"]), np.zeros(2), 0.0)]
    d, z = np.zeros(2), 0.0
    for x in f3:
        d = d + (np.asarray(x["after"]["obj_pos"])[:2] - np.asarray(x["before"]["obj_pos"])[:2]) / R
        z = z + x["u_app"][2] / 100
        traj.append((S1.state_vector(x["after"]), d.copy(), z))
    dl = np.asarray(r["dlt"])
    return S1.goal_loss("close", traj, {"lift_m": 0.05}) + ALPHA * float(np.sum(dl ** 2)) / 3 + BETA * float(np.sum(np.diff(dl, axis=0) ** 2))


def task_aligned_key(r):
    """Experiment 3's objective on measured outcomes (lower = better): 1. hard safety violations last; 2. stable holds first;
    3. smaller corrections; 4. fewer steps."""
    o = r["outcome"]
    return (1 if o["safety_violation"] else 0, 0 if o["task_success"] else 1, float(np.linalg.norm(r["candidate"]["dp_cm"])), o["steps"])


def rank_of(sorted_ids, ids):
    return sorted(sorted_ids.index(i) + 1 for i in ids if i in sorted_ids)


def exp2_analysis(obj):
    res = {}
    for start, S in states(obj).items():
        rows = S["rows"]
        by_all = {r["candidate"]["id"]: r for r in rows}
        by = {i: r for i, r in by_all.items() if i != "posctrl"}          # the labelled positive control is never selectable
        succ = [i for i, r in by.items() if r["outcome"]["task_success"]]
        succ_sampled = [i for i in succ if by[i]["candidate"]["label"] == "sampled"]
        jr = {i: j_existing_real(r) for i, r in by.items()}
        order_real = sorted([i for i in by if jr[i] is not None], key=lambda i: jr[i])
        sup = {i: r["prediction"] for i, r in by.items()}
        order_hat_supported = sorted([i for i in by if sup[i].get("supported") and sup[i].get("J_hat") is not None], key=lambda i: sup[i]["J_hat"])
        order_hat_all = sorted([i for i in by if sup[i].get("J_hat") is not None], key=lambda i: sup[i]["J_hat"])
        order_task = sorted(by, key=lambda i: task_aligned_key(by[i]))
        tossed = [i for i, r in by.items() if not r["outcome"]["task_success"] and (r["outcome"]["max_rise_cm"] or 0) >= 5.0]
        pe = []
        for i, r in by.items():
            pd = sup[i].get("predicted_ds_first")
            if pd is not None and r["first3"]:
                pe.append((i, float(np.sqrt(np.mean((np.array(pd) - np.array(r["first3"][0]["ds"])) ** 2))),
                           float(np.linalg.norm(np.array(pd)[11:13])), float(np.linalg.norm(np.array(r["first3"][0]["ds"])[11:13]))))
        rep = {}
        for x in S["repeats"]:
            rep.setdefault(x["candidate"]["id"], []).append(x["outcome"]["task_success"])
        noise = None
        if S["r2"]:
            r2 = {x["candidate"]["id"]: x["outcome"]["task_success"] for x in S["r2"]}
            same = [by[i]["outcome"]["task_success"] == r2[i] for i in r2 if i in by]
            noise = {"candidates_repeated": len(same), "same_outcome": int(sum(same))}
        res[start] = {
            "n": len(by), "successes": sorted(succ), "successes_sampled": len(succ_sampled), "nominal_success": by["nominal"]["outcome"]["task_success"],
            "posctrl_success": by_all.get("posctrl", {}).get("outcome", {}).get("task_success"),
            "existing_objective_measured": {"selected": order_real[0] if order_real else None, "selected_success": bool(order_real and order_real[0] in succ),
                                            "success_ranks": rank_of(order_real, succ), "nominal_rank": rank_of(order_real, ["nominal"]),
                                            "top5_has_success": any(i in succ for i in order_real[:5])},
            "existing_objective_predicted": {"supported": len(order_hat_supported), "selected": order_hat_supported[0] if order_hat_supported else None,
                                             "selected_success": bool(order_hat_supported and order_hat_supported[0] in succ),
                                             "success_ranks_all": rank_of(order_hat_all, succ), "top5_has_success": any(i in succ for i in order_hat_supported[:5])},
            "task_aligned_measured": {"selected": order_task[0], "selected_success": order_task[0] in succ,
                                      "selected_dp_cm": by[order_task[0]]["candidate"]["dp_cm"], "lifted_but_not_held_or_tipped": tossed},
            "prediction_error_first_segment": {"median_rmse": round(float(np.median([e[1] for e in pe])), 4) if pe else None,
                                               "pred_vs_measured_object_xy_corr": round(float(np.corrcoef([e[2] for e in pe], [e[3] for e in pe])[0, 1]), 3) if len(pe) > 2 else None},
            "repeats": rep, "restore_noise_full_bank": noise,
            "success_dp_cm": {i: by[i]["candidate"]["dp_cm"] for i in succ}}
    return res


def representation_eval(train_states, test_states, rep, **kw):
    from cup_transfer import s1v2 as V
    mem = V.OutcomeMemory()
    for S in train_states:
        for r in S["rows"]:
            if r["candidate"]["label"] == "sampled" or r["candidate"]["id"] == "nominal":
                x, _ = V.features(S["state"], rep, r["candidate"]["dp_cm"])
                mem.add(V.record(x, r["candidate"], r["outcome"], "train"))
    out = []
    for S in test_states:
        sel = V.Selector(mem, rep, [r["candidate"] for r in S["rows"]], **kw)
        pick, x, sc = sel.choose(S["state"])
        y = {r["candidate"]["id"]: r["outcome"]["task_success"] for r in S["rows"]}
        ranked = sorted(sc, key=lambda s: -s["score"])
        qs = [(s["score"], y[s["id"]]) for s in sc]
        pos = [q for q, t in qs if t]
        neg = [q for q, t in qs if not t]
        auc = None if not pos or not neg else float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
        out.append({"start": S["state"]["start"], "picked": pick["id"], "picked_success": bool(y[pick["id"]]), "top5_success": int(sum(y[s["id"]] for s in ranked[:5])),
                    "auc": None if auc is None else round(auc, 3), "supported": sum(1 for s in sc if s["q"] is not None), "n_success": int(sum(y.values()))})
    return out


def cmd_analyze(a):
    res = {"exp1": {}, "exp2": {}, "exp4": {}, "exp5": {}}
    rows1, tab = exp1_table()
    res["exp1"] = {f"{k[0]}|{k[1]}": v for k, v in tab.items()}
    for obj in ("cup_b", "cup_a", "cyl_c"):
        res["exp2"][obj] = exp2_analysis(obj)
    B = list(states("cup_b").values())
    A = list(states("cup_a").values())
    C = list(states("cyl_c").values())
    hasvec = lambda S: "vec_cm" in S["state"]["geometry_before"]["tips"]["R_thumb_tip"]
    for rep in ("current", "geometry", "post_geometry"):
        Bs, As, Cs = (B, A, C) if rep != "post_geometry" else ([S for S in B if hasvec(S)], [S for S in A if hasvec(S)], [S for S in C if hasvec(S)])
        loso = []
        for i, S in enumerate(Bs):
            loso += representation_eval(Bs[:i] + Bs[i + 1:], [S], rep)
        res["exp4"][rep] = {"states_used": [S["state"]["start"] for S in Bs], "within_B_leave_one_start_out": loso,
                            "A_and_C_to_B": representation_eval(As + Cs, Bs, rep) if (As or Cs) else None}
    # experiment 5 (offline replay with the recorded outcomes of every bank candidate)
    from cup_transfer import s1v2 as V
    rep = a.rep
    src = V.OutcomeMemory()
    for S in A[:a.source_states]:
        for r in S["rows"]:
            if r["candidate"]["label"] == "sampled" or r["candidate"]["id"] == "nominal":
                x, _ = V.features(S["state"], rep, r["candidate"]["dp_cm"])
                src.add(V.record(x, r["candidate"], r["outcome"], "source:" + S["state"]["start"]))
    log = open(OUT / "exp5_offline_updates.jsonl", "w")
    for cond, mem0, upd in (("frozen_source", src.clone(frozen=True), False), ("updating_source", src.clone(frozen=False), True),
                            ("updating_empty", V.OutcomeMemory(), True)):
        eps = []
        for S in B:
            y = {r["candidate"]["id"]: r["outcome"] for r in S["rows"]}
            sel = V.Selector(mem0.clone(), rep, [r["candidate"] for r in S["rows"]])
            log.write(json.dumps({"condition": cond, "start": S["state"]["start"]}) + "\n")
            h = V.episode(sel, S["state"], lambda i: y[i], attempts=3, update=upd, log=log)
            eps.append({"start": S["state"]["start"], "attempts": h})
        succ_at = [next((e["attempt"] for e in ep["attempts"] if e["success"]), None) for ep in eps]
        res["exp5"][cond] = {"source_records": len(mem0.records), "states": len(eps), "first_attempt_success": sum(1 for s in succ_at if s == 1),
                             "success_within_3": sum(1 for s in succ_at if s), "attempts_to_success_censored4": float(np.mean([s or 4 for s in succ_at])) if succ_at else None,
                             "episodes": eps}
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(json.dumps({"exp1": res["exp1"]}, indent=0)[:2500])
    for obj, v in res["exp2"].items():
        for st, x in v.items():
            print(obj, st, "succ", len(x["successes"]), "nominal", x["nominal_success"], "posctrl", x["posctrl_success"],
                  "| existing(measured) sel", x["existing_objective_measured"]["selected"], x["existing_objective_measured"]["selected_success"],
                  "ranks", x["existing_objective_measured"]["success_ranks"][:5],
                  "| existing(pred) sel", x["existing_objective_predicted"]["selected"], x["existing_objective_predicted"]["selected_success"],
                  "| task sel", x["task_aligned_measured"]["selected"], x["task_aligned_measured"]["selected_success"], "| noise", x["restore_noise_full_bank"])
    for rep_, v in res["exp4"].items():
        l = v["within_B_leave_one_start_out"]
        print("exp4", rep_, "LOSO picked success", sum(r["picked_success"] for r in l), "/", len(l), "AUC", [r["auc"] for r in l],
              "| A+C->B", None if not v["A_and_C_to_B"] else (sum(r["picked_success"] for r in v["A_and_C_to_B"]), [r["auc"] for r in v["A_and_C_to_B"]]))
    for cond, v in res["exp5"].items():
        print("exp5", cond, {k: v[k] for k in ("source_records", "states", "first_attempt_success", "success_within_3", "attempts_to_success_censored4")})


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figd = OUT / "figures"
    figd.mkdir(parents=True, exist_ok=True)
    rows1, tab = exp1_table()
    if rows1:
        pts, conds = list(POINTS_ORDER), ["nominal", "transient", "persistent"]
        starts = sorted({r["start"] for r in rows1})
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(starts) * len(pts) + 1.2))
        ylab = []
        for yi, (pt, st) in enumerate([(pt, st) for pt in pts for st in starts]):
            ylab.append(f"{pt} | {st}")
            for xi, c in enumerate(conds):
                rs = [r for r in rows1 if r["point"] == pt and r["start"] == st and r["condition"] == c]
                ok = sum(r["outcome"]["task_success"] for r in rs)
                ax.add_patch(plt.Rectangle((xi, yi), 0.95, 0.9, color=("#16a34a" if ok == len(rs) and rs else "#dc2626" if ok == 0 else "#f59e0b")))
                ax.text(xi + 0.47, yi + 0.45, f"{ok}/{len(rs)}", ha="center", va="center", color="white", fontsize=9)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, len(ylab))
        ax.set_xticks([0.47, 1.47, 2.47])
        ax.set_xticklabels(conds)
        ax.set_yticks([i + 0.45 for i in range(len(ylab))])
        ax.set_yticklabels(ylab, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Experiment 1: the known 0.8 cm correction on cup B - held / runs")
        fig.tight_layout()
        fig.savefig(figd / "exp1_duration.png", dpi=110)
        plt.close(fig)
    B = states("cup_b")
    objs = [("cup_b", B), ("cup_a", states("cup_a")), ("cyl_c", states("cyl_c"))]
    cols = [(o, st) for o, S in objs for st in sorted(S)]
    if cols:
        ids = [r["candidate"]["id"] for r in next(iter(B.values()))["rows"]] if B else []
        fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 3.2, 9))
        for xi, (o, st) in enumerate(cols):
            S = dict(objs)[o][st]
            by = {r["candidate"]["id"]: r for r in S["rows"]}
            for yi, i in enumerate(ids):
                r = by.get(i)
                if r is None:
                    continue
                ok = r["outcome"]["task_success"]
                ax.add_patch(plt.Rectangle((xi, yi), 0.92, 0.88, color="#16a34a" if ok else ("#f59e0b" if (r["outcome"]["max_rise_cm"] or 0) >= 2 else "#dc2626")))
        ax.set_xlim(0, len(cols))
        ax.set_ylim(0, len(ids))
        ax.set_xticks([i + 0.46 for i in range(len(cols))])
        ax.set_xticklabels([f"{o.replace('_', ' ')}\n{st}" for o, st in cols], fontsize=7)
        ax.set_yticks([i + 0.44 for i in range(len(ids))])
        lab = {r["candidate"]["id"]: f"{r['candidate']['id']} {np.round(r['candidate']['dp_cm'], 1).tolist()}" for r in next(iter(B.values()))["rows"]} if B else {}
        ax.set_yticklabels([lab.get(i, i) for i in ids], fontsize=6)
        ax.invert_yaxis()
        ax.set_title("Experiment 2: persistent correction bank (green held, amber lifted >= 2 cm but not held, red failed)", fontsize=9)
        fig.tight_layout()
        fig.savefig(figd / "exp2_bank.png", dpi=110)
        plt.close(fig)
    an = OUT / "analysis.json"
    if an.exists():
        res = json.loads(an.read_text())["exp5"]
        if res:
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            for k, (c, v) in enumerate(res.items()):
                at = [next((e["attempt"] for e in ep["attempts"] if e["success"]), 4) for ep in v["episodes"]]
                ax.scatter(np.array(at) + (k - 1) * 0.12, np.arange(len(at)), label=f"{c} (mean {np.mean(at):.2f})", s=30)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(["1", "2", "3", "not within 3"])
            ax.set_xlabel("attempt of the first held grasp")
            ax.set_yticks(range(len(v["episodes"])))
            ax.set_yticklabels([ep["start"] for ep in v["episodes"]], fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            ax.set_title("Experiment 5 (offline replay): cup B starts", fontsize=9)
            fig.tight_layout()
            fig.savefig(figd / "exp5_attempts.png", dpi=110)
            plt.close(fig)
    print("figures in", figd)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exp0")
    p = sub.add_parser("analyze")
    p.add_argument("--rep", default="geometry")
    p.add_argument("--source-states", type=int, default=2)
    sub.add_parser("figures")
    a = ap.parse_args()
    {"exp0": cmd_exp0, "analyze": cmd_analyze, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
