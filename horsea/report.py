"""Aggregate every result JSON into experiments/report/{summary.json, summary.md, *.png}.

    python -m horsea.report
Uncertainty: 95% bootstrap intervals over tasks (tasks are the unit of generalisation here;
rollouts within a task are not independent replications of the method).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from horsea.paths import EXP, HELDOUT_90, LIBERO_10, RETENTION_90

METHODS = ["base", "ft", "ttt", "kv", "fmw", "res"]
LABEL = {"base": "zero-shot base", "ft": "fine-tune (no memory)", "ttt": "internal TTT (RoboTTT-style)",
         "kv": "KV memory + velocity reader (proposal)", "fmw": "FM-write velocity memory",
         "res": "fast final-action residual"}


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_adapt():
    rows = []
    for p in glob.glob(os.path.join(EXP, "adapt", "*", "*", "*.json")):
        rows.append(json.load(open(p)))
    return rows


def adaptation_tables(rows):
    out = {}
    for suite, tasks in [("libero_90", HELDOUT_90), ("libero_10", LIBERO_10)]:
        tab = {}
        base = {r["task"]: r["rate"] for r in rows if r["arm"] == "base" and r["suite"] == suite and r["task"] in tasks}
        for m in METHODS:
            per_k = defaultdict(dict)
            for r in rows:
                if r["arm"] == m and r["suite"] == suite and not r["wrong"] and r["task"] in tasks:
                    per_k[r["K"]][r["task"]] = r
            if m == "base":
                per_k = {0: {t: {"rate": v, "adapt_sec": 0} for t, v in base.items()}}
            tab[m] = {}
            for K, d in sorted(per_k.items()):
                rates = [d[t]["rate"] for t in sorted(d)]
                mean, lo, hi = boot(rates)
                tab[m][K] = {"mean": mean, "lo": lo, "hi": hi, "n_tasks": len(rates),
                             "adapt_sec": float(np.mean([d[t].get("adapt_sec", 0) for t in d])),
                             "per_task": {t: d[t]["rate"] for t in sorted(d)}}
        out[suite] = tab
    wrong = {}
    for m in ["ttt", "kv", "fmw", "res"]:
        right = {r["task"]: r["rate"] for r in rows if r["arm"] == m and r["suite"] == "libero_90" and r["K"] == 5 and not r["wrong"]}
        wr = {r["task"]: r["rate"] for r in rows if r["arm"] == m and r["suite"] == "libero_90" and r["K"] == 5 and r["wrong"]}
        common = sorted(set(right) & set(wr))
        if common:
            wrong[m] = {"right": boot([right[t] for t in common]), "wrong": boot([wr[t] for t in common]),
                        "n_tasks": len(common)}
    out["wrong_memory_K5"] = wrong
    ret = {r["task"]: r["rate"] for r in rows if r["arm"] == "base" and r["suite"] == "libero_90" and r["task"] in RETENTION_90}
    out["base_retention"] = boot(list(ret.values())) if ret else None
    out["base_retention_per_task"] = ret
    return out


def consolidation_table(base_heldout, base_ret, base_l10):
    out = {}
    for p in sorted(glob.glob(os.path.join(EXP, "consolidate", "*", "final.json"))):
        f = json.load(open(p))
        arm = f["arm"]
        seq = f["sequence"]
        teacher = [s["teacher"]["rate"] for s in seq if "teacher" in s]
        student = [s["student"]["rate"] for s in seq]
        ratios = []
        for s in seq:
            b0 = base_heldout.get(s["task"], np.nan)
            if "teacher" in s and s["teacher"]["rate"] - b0 > 0.05:
                ratios.append((s["student"]["rate"] - b0) / (s["teacher"]["rate"] - b0))
        final_novel = [r["rate"] for r in f["novel"]]
        after = {s["task"]: s["student"]["rate"] for s in seq}
        forget = [after[r["task"]] - r["rate"] for r in f["novel"] if r["task"] in after]
        out[arm] = {
            "teacher_success": boot(teacher) if teacher else None,
            "student_after_own_consolidation": boot(student),
            "transfer_ratio_median": float(np.median(ratios)) if ratios else None,
            "transfer_ratio_per_task": ratios,
            "final_novel_success": boot(final_novel),
            "forgetting_of_novel (after - final)": boot(forget),
            "retention_old_tasks": boot([r["rate"] for r in f["retention"]]),
            "retention_drop_vs_base": (boot([r["rate"] for r in f["retention"]])[0] - base_ret) if base_ret == base_ret else None,
            "libero10_zero_shot": boot([r["rate"] for r in f["libero_10"]]),
            "libero10_vs_base": boot([r["rate"] for r in f["libero_10"]])[0] - base_l10 if base_l10 == base_l10 else None,
            "distill_sec_mean": float(np.mean([s["distill_sec"] for s in seq])),
            "tasks_done": len(seq),
        }
    return out


def meta_curves():
    out = {}
    for arm in ["ttt", "kv", "fmw", "res"]:
        p = os.path.join(EXP, "memory", arm, "log.jsonl")
        if os.path.exists(p):
            recs = [json.loads(l) for l in open(p) if l.strip()]
            out[arm] = [r for r in recs if r.get("monitor")]
    return out


def plots(ad, cons, curves, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#888888", "ft": "#444444", "ttt": "#1f77b4", "kv": "#ff7f0e", "fmw": "#2ca02c", "res": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, suite in zip(axes, ["libero_90", "libero_10"]):
        tab = ad.get(suite, {})
        for m in METHODS:
            d = tab.get(m, {})
            if not d:
                continue
            Ks = sorted(d)
            y = [d[k]["mean"] for k in Ks]
            lo = [d[k]["lo"] for k in Ks]
            hi = [d[k]["hi"] for k in Ks]
            if m == "base":
                ax.axhline(y[0], color=colors[m], ls="--", label=LABEL[m])
                continue
            ax.errorbar(Ks, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], marker="o", capsize=3,
                        color=colors[m], label=LABEL[m])
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 5, 10])
        ax.set_xticklabels(["0", "1", "2", "5", "10"])
        ax.set_xlabel("demonstrations of the novel task (K)")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.set_title({"libero_90": "held-out LIBERO-90 (10 tasks)", "libero_10": "LIBERO-10 (10 tasks)"}[suite])
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "adaptation.png"), dpi=150)

    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.2))
        axes = np.atleast_1d(axes)
        for ax, (arm, recs) in zip(axes, curves.items()):
            steps = [r["step"] for r in recs]
            for K in ["0", "1", "5", "10"]:
                ax.plot(steps, [r["heldout"][K] for r in recs], label=f"K={K}")
            ax.set_title(f"{arm}: held-out query loss")
            ax.set_xlabel("meta step")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "meta_training.png"), dpi=150)


def fmt(x):
    return "n/a" if x is None or x != x else f"{x:.2f}"


def markdown(ad, cons):
    L = ["# Memory study: results\n"]
    for suite, title in [("libero_90", "Held-out LIBERO-90 (novel instruction, familiar scene)"),
                         ("libero_10", "LIBERO-10 (far-novel long-horizon tasks)")]:
        tab = ad.get(suite, {})
        if not tab:
            continue
        L.append(f"## Adaptation: {title}\n")
        L.append("| method | K=0 | K=1 | K=2 | K=5 | K=10 | adapt time (s, K=5) |")
        L.append("|---|---|---|---|---|---|---|")
        for m in METHODS:
            d = tab.get(m, {})
            if not d:
                continue
            cells = []
            for K in [0, 1, 2, 5, 10]:
                c = d.get(K)
                cells.append(f"{c['mean']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]" if c else "")
            at = d.get(5, {}).get("adapt_sec")
            L.append(f"| {LABEL[m]} | " + " | ".join(cells) + f" | {fmt(at)} |")
        L.append("")
    w = ad.get("wrong_memory_K5", {})
    if w:
        L.append("## Memory-content control (K=5, held-out LIBERO-90)\n")
        L.append("| arm | right demos | wrong task's demos |")
        L.append("|---|---|---|")
        for m, d in w.items():
            L.append(f"| {LABEL[m]} | {d['right'][0]:.2f} | {d['wrong'][0]:.2f} |")
        L.append("")
    if cons:
        L.append("## Fast -> slow consolidation (sequential over the 10 held-out tasks, K=5)\n")
        L.append("| arm | teacher | memory-off student | median transfer | final novel (after all 10) | "
                 "forgetting of novel | old-task retention (drop vs base) | LIBERO-10 zero-shot (vs base) | distill s/task |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for arm, d in cons.items():
            L.append(f"| {LABEL.get(arm, arm)} | {fmt(d['teacher_success'][0] if d['teacher_success'] else None)} | "
                     f"{fmt(d['student_after_own_consolidation'][0])} | {fmt(d['transfer_ratio_median'])} | "
                     f"{fmt(d['final_novel_success'][0])} | {fmt(d['forgetting_of_novel (after - final)'][0])} | "
                     f"{fmt(d['retention_old_tasks'][0])} ({fmt(d['retention_drop_vs_base'])}) | "
                     f"{fmt(d['libero10_zero_shot'][0])} ({fmt(d['libero10_vs_base'])}) | {d['distill_sec_mean']:.0f} |")
    return "\n".join(L) + "\n"


def main():
    outdir = os.path.join(EXP, "report")
    os.makedirs(outdir, exist_ok=True)
    rows = load_adapt()
    ad = adaptation_tables(rows)
    b90 = {t: r for t, r in ad["libero_90"].get("base", {}).get(0, {}).get("per_task", {}).items()}
    base_ret = ad["base_retention"][0] if ad.get("base_retention") else float("nan")
    b10 = ad["libero_10"].get("base", {}).get(0, {}).get("mean", float("nan"))
    cons = consolidation_table(b90, base_ret, b10)
    curves = meta_curves()
    summary = {"adaptation": ad, "consolidation": cons, "meta_curves": curves}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(markdown(ad, cons))
    try:
        plots(ad, cons, curves, outdir)
    except Exception as e:  # plotting must never hide the numbers
        print("plotting failed:", e)
    print(open(os.path.join(outdir, "summary.md")).read())


if __name__ == "__main__":
    main()
