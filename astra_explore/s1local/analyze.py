"""Analysis of the local-execution-adaptation pilot (system python: numpy + matplotlib).

    python -m s1local.analyze --out <run dir>

Writes results.json, results.md (tables), curves.png (success vs attempt), eval.png (fresh-start success), trajectories.png
(paired representative trajectories). Unsuccessful nine-attempt runs are censored (never dropped); confidence intervals
aggregate the three evaluation starts within a condition and resample conditions within each operation (paired).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OPS = ["grasp", "transport", "door"]
KINDS, THRESH = {}, {}                              # from the run's manifest (operation kind, push threshold)
ARMS = ["A", "B", "C"]
ARM_NAME = {"A": "A original", "B": "B fixed-distribution search", "C": "C history-adaptive search"}


def load(out):
    out = Path(out)
    rd = lambda n: [json.loads(x) for x in (out / n).read_text().splitlines() if x.strip()] if (out / n).exists() else []  # noqa: E731
    return rd("attempts.jsonl"), rd("rollouts.jsonl"), rd("blocks.jsonl")


def per_block(attempts):
    """{(op, cond): {arm: {"adapt": [rows by attempt], "eval": [rows]}}}"""
    B = {}
    for a in attempts:
        if "arm" not in a or "phase" not in a:
            continue
        d = B.setdefault((a["op"], a["cond"]), {}).setdefault(a["arm"], {"adapt": [], "eval": []})
        d[a["phase"]].append(a)
    for blk in B.values():
        for arm in blk.values():
            arm["adapt"].sort(key=lambda r: r["attempt"])
    return B


def boot_ci(diffs_by_op, n=10000, seed=0):
    """Paired, operation-stratified bootstrap: resample conditions within each operation, average the operation means."""
    rng = np.random.default_rng(seed)
    ops = [o for o, v in diffs_by_op.items() if len(v)]
    if not ops:
        return None, (None, None)
    est = float(np.mean([np.mean(diffs_by_op[o]) for o in ops]))
    bs = []
    for _ in range(n):
        bs.append(np.mean([np.mean(rng.choice(diffs_by_op[o], size=len(diffs_by_op[o]), replace=True)) for o in ops]))
    return est, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def summarize(attempts, rollouts):
    B = per_block(attempts)
    res = {"blocks": len(B), "per_op": {}, "overall": {}}
    cond_rows = {op: [] for op in OPS}
    for (op, ci), blk in sorted(B.items()):
        any_arm = next(iter(blk.values()))
        row = {"op": op, "cond": ci, "condition": (any_arm["adapt"] or any_arm["eval"])[0].get("condition")}
        for arm in ARMS:
            if arm not in blk:
                continue
            ad, ev = blk[arm]["adapt"], blk[arm]["eval"]
            first = next((r["attempt"] for r in ad if r["S"] == 1), None)
            steps, tot = None, 0
            for r in ad:
                tot += r["steps"]
                if r["S"] == 1:
                    steps = tot
                    break
            valid_ev = [r for r in ev if not r["init_failure"]]
            row[arm] = {"first_success": first, "censored": first is None, "steps_to_first": steps if steps is not None else tot,
                        "steps_censored": steps is None, "adapt_any": first is not None, "adapt_success": sum(r["S"] == 1 for r in ad), "n_adapt": len(ad),
                        "eval_success": sum(r["S"] == 1 for r in valid_ev), "n_eval": len(valid_ev), "eval_init_failures": len(ev) - len(valid_ev),
                        "eval_loss": sum(r["D"] == 1 for r in valid_ev), "adapt_loss": sum(r["D"] == 1 for r in ad if r["D"] is not None),
                        "eval_violation": sum((r["V"] or 0) == 1 for r in valid_ev), "adapt_steps_total": sum(r["steps"] for r in ad),
                        "eval_steps_total": sum(r["steps"] for r in valid_ev), "selected_theta": ev[0]["theta"] if ev else None,
                        "best_adapt_J": max((r["J"] for r in ad if r["J"] is not None), default=None)}
        cond_rows[op].append(row)
    for scope, ops in [(op, [op]) for op in OPS] + [("overall", OPS)]:
        rows = [r for op in ops for r in cond_rows[op]]
        if not rows:
            continue
        s = {"conditions": len(rows)}
        for arm in ARMS:
            xs = [r[arm] for r in rows if arm in r]
            if not xs:
                continue
            ne = sum(x["n_eval"] for x in xs)
            s[arm] = {"conditions_with_adapt_success": sum(x["adapt_any"] for x in xs), "first_success_attempts": [x["first_success"] for x in xs],
                      "restricted_mean_first_success_censored_at_10": float(np.mean([x["first_success"] or 10 for x in xs])),
                      "steps_to_first_success": [(x["steps_to_first"], x["steps_censored"]) for x in xs],
                      "eval_success": sum(x["eval_success"] for x in xs), "n_eval": ne, "eval_rate": (sum(x["eval_success"] for x in xs) / ne) if ne else None,
                      "eval_init_failures": sum(x["eval_init_failures"] for x in xs), "eval_loss_rate": (sum(x["eval_loss"] for x in xs) / ne) if ne else None,
                      "adapt_loss_rate": sum(x["adapt_loss"] for x in xs) / max(1, sum(x["n_adapt"] for x in xs)),
                      "eval_violation_rate": (sum(x["eval_violation"] for x in xs) / ne) if ne else None,
                      "adapt_steps_total": sum(x["adapt_steps_total"] for x in xs)}
        (res["per_op"].setdefault(scope, s) if scope != "overall" else res["overall"].update(s))
    # paired contrasts on condition-level fresh-start success fractions
    for name, (x, y) in {"H1_B_minus_A": ("B", "A"), "H2_C_minus_B": ("C", "B"), "C_minus_A": ("C", "A")}.items():
        diffs = {op: [r[x]["eval_success"] / r[x]["n_eval"] - r[y]["eval_success"] / r[y]["n_eval"] for r in cond_rows[op]
                      if x in r and y in r and r[x]["n_eval"] and r[y]["n_eval"]] for op in OPS}
        est, ci = boot_ci(diffs)
        att = {op: [(r[y]["first_success"] or 10) - (r[x]["first_success"] or 10) for r in cond_rows[op] if x in r and y in r] for op in OPS}
        est_a, ci_a = boot_ci(att)
        res.setdefault("contrasts", {})[name] = {"fresh_start_success_diff_pp": None if est is None else 100 * est,
                                                 "ci95_pp": [None if v is None else 100 * v for v in ci],
                                                 "per_op_pp": {op: (100 * float(np.mean(v)) if v else None) for op, v in diffs.items()},
                                                 "attempts_saved_to_first_success(censored_at_10)": est_a, "ci95_attempts": ci_a}
    res["conditions"] = cond_rows
    return res


def discovery_curve(cond_rows, arm, ops):
    rows = [r for op in ops for r in cond_rows[op] if arm in r]
    if not rows:
        return None
    return [sum(1 for r in rows if r[arm]["first_success"] is not None and r[arm]["first_success"] <= k) / len(rows) for k in range(1, 10)]


def plots(out, res, rollouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    cr = res["conditions"]
    col = {"A": "#64748b", "B": "#2563eb", "C": "#dc2626"}
    fig, ax = plt.subplots(1, len(OPS) + 1, figsize=(4 * (len(OPS) + 1), 3.6), sharey=True)
    for i, (title, ops) in enumerate([(o, [o]) for o in OPS] + [("all operations", OPS)]):
        for arm in ARMS:
            c = discovery_curve(cr, arm, ops)
            if c:
                ax[i].step(range(1, 10), c, where="post", color=col[arm], label=ARM_NAME[arm], lw=2, alpha=.85)
        n = sum(len(cr[o]) for o in ops)
        ax[i].set_title(f"{title} ({n} conditions)")
        ax[i].set_xlabel("adaptation attempt")
        ax[i].set_xticks(range(1, 10))
        ax[i].axvline(5.5, color="#999", ls=":", lw=1)
    ax[0].set_ylabel("fraction of conditions with a success so far")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Success vs attempt (cumulative discovery; attempt 1 = Astra's program for every arm; C updates after attempt 5)")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, len(OPS) + 1, figsize=(4 * (len(OPS) + 1), 3.4), sharey=True)
    for i, (title, scope) in enumerate([(o, o) for o in OPS] + [("all operations", "overall")]):
        s = res["per_op"].get(scope) if scope != "overall" else res["overall"]
        if not s:
            continue
        vals = [s.get(a, {}).get("eval_rate") or 0 for a in ARMS]
        ns = [f"{s.get(a, {}).get('eval_success', 0)}/{s.get(a, {}).get('n_eval', 0)}" for a in ARMS]
        ax[i].bar(ARMS, vals, color=[col[a] for a in ARMS])
        for j, t in enumerate(ns):
            ax[i].text(j, vals[j] + 0.02, t, ha="center", fontsize=9)
        ax[i].set_ylim(0, 1.15)
        ax[i].set_title(title)
    ax[0].set_ylabel("fresh-start success (frozen parameters)")
    fig.suptitle("Fresh-start evaluation: 3 new starts per condition, parameters frozen after adaptation")
    fig.tight_layout()
    fig.savefig(out / "eval.png", dpi=110)
    plt.close(fig)
    # representative paired trajectories: per operation the condition with the largest |C - A| fresh-start difference
    byk = {r["key"]: r for r in rollouts}
    fig, ax = plt.subplots(1, max(2, len(OPS)), figsize=(5.4 * max(2, len(OPS)), 3.8))
    for i, op in enumerate(OPS):
        rows = cr[op]
        if not rows:
            continue
        pick = max(rows, key=lambda r: (abs(r.get("C", {}).get("eval_success", 0) - r.get("A", {}).get("eval_success", 0)), -r["cond"]))
        for arm in ARMS:
            th = pick.get(arm, {}).get("selected_theta")
            if th is None:
                continue
            key = next((k for k, r in byk.items() if r["op"] == op and r["cond"] == pick["cond"] and r["start"] == "e1" and r["slot"] == "eval"
                        and np.allclose(r["theta"], th, atol=1e-6)), None)
            if key is None or not byk[key].get("trajectory_every2"):
                continue
            raw = byk[key]["trajectory_every2"]
            x = np.arange(len(raw)) * 2
            kind = KINDS.get(op, {"grasp": "grasp", "transport": "transport", "door": "push"}.get(op, "transport"))
            if not hasattr(raw[0], "__len__"):
                y, lab = np.array(raw, float), "joint angle (rad)"
            else:
                tr = np.array([r[:3] for r in raw], float)
                if kind == "grasp":
                    y, lab = tr[:, 2] - tr[0, 2], "object height gain (m)"
                else:
                    y, lab = np.linalg.norm(tr[:, :2] - tr[0, :2], axis=1), "object horizontal travel (m)"
            ax[i].plot(x, y, color=col[arm], lw=2, alpha=.85, ls="-" if arm != "B" else "--",
                       label=f"{arm}: {'success' if byk[key]['S'] == 1 else 'no success'} (J {byk[key]['J']})")
        if KINDS.get(op) == "grasp" or op == "grasp":
            ax[i].axhline(0.08, color="#999", ls=":", lw=1)
        if THRESH.get(op) is not None:
            ax[i].axhline(THRESH[op], color="#999", ls=":", lw=1)
        ax[i].set_title(f"{op}: condition {pick['cond']} {tuple(pick.get('condition', []))}, fresh start e1")
        ax[i].set_xlabel("control step")
        ax[i].set_ylabel(lab)
        ax[i].legend(fontsize=8)
    fig.suptitle("Representative paired trajectories on the same fresh start (selected parameters of each arm)")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=110)
    plt.close(fig)


def table_md(res):
    L = ["| scope | arm | conditions with a success in 9 attempts | first-success attempt per condition (- = none, censored) | restricted mean first success (censored = 10) | fresh-start success | init failures | object loss (eval / adapt) | constraint violations | adaptation control steps |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for scope in OPS + ["overall"]:
        s = res["per_op"].get(scope) if scope != "overall" else res["overall"]
        if not s:
            continue
        for arm in ARMS:
            x = s.get(arm)
            if not x:
                continue
            fs = " ".join(str(v) if v is not None else "-" for v in x["first_success_attempts"])
            er = f"{x['eval_success']}/{x['n_eval']}" + (f" ({100 * x['eval_rate']:.0f} %)" if x["eval_rate"] is not None else "")
            L.append(f"| {scope} | {arm} | {x['conditions_with_adapt_success']}/{s['conditions']} | {fs} | {x['restricted_mean_first_success_censored_at_10']:.2f} | {er} | "
                     f"{x['eval_init_failures']} | {100 * (x['eval_loss_rate'] or 0):.0f} % / {100 * x['adapt_loss_rate']:.0f} % | {100 * (x['eval_violation_rate'] or 0):.0f} % (signal unavailable) | {x['adapt_steps_total']} |")
    L += ["", "| contrast | fresh-start success difference (pp) | 95 % CI (paired, operation-stratified) | per operation (pp) | attempts saved to first success (censored at 10) | 95 % CI |",
          "|---|---|---|---|---|---|"]
    for k, c in res.get("contrasts", {}).items():
        po = ", ".join(f"{o} {v:+.0f}" if v is not None else f"{o} n/a" for o, v in c["per_op_pp"].items())
        f = lambda v: "n/a" if v is None else f"{v:+.1f}"  # noqa: E731
        L.append(f"| {k} | {f(c['fresh_start_success_diff_pp'])} | [{f(c['ci95_pp'][0])}, {f(c['ci95_pp'][1])}] | {po} | "
                 f"{f(c['attempts_saved_to_first_success(censored_at_10)'])} | [{f(c['ci95_attempts'][0])}, {f(c['ci95_attempts'][1])}] |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    attempts, rollouts, blocks = load(a.out)
    names = []
    for r in attempts:                              # operation names in first-seen order (any task)
        if "op" in r and r["op"] not in names:
            names.append(r["op"])
    OPS[:] = names
    man = Path(a.out) / "manifest.json"
    if man.exists():
        for k, v in (json.loads(man.read_text()).get("operations") or {}).items():
            KINDS[k] = v.get("kind", {"grasp": "grasp", "transport": "transport", "door": "push"}.get(k))
            if KINDS[k] == "push":
                THRESH[k] = v["success"] if isinstance(v.get("success"), (int, float)) else (1.3264 if k == "door" else None)
    res = summarize(attempts, rollouts)
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (Path(a.out) / "results.md").write_text(table_md(res) + "\n")
    plots(a.out, res, rollouts)
    print(table_md(res))


if __name__ == "__main__":
    main()
