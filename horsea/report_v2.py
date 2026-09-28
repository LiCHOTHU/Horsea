"""Protocol v2 report: expert-arm lifecycle streams, self-rollout verification, writer training and the
closed-loop R0 pilot. Works on partial results.

    python -m horsea.report_v2        -> prints markdown, writes experiments/protocol_v2/REPORT.md
"""
import glob
import json
import os

from horsea.paths import EXP

ROOT = os.path.join(EXP, "protocol_v2")


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def e_arm():
    L = ["## Expert arm (E): two-cycle lifecycle per task pair\n",
         "Validation fold (20 starts) for base/teacher/gates; test = 50 generated starts per task, old panel = "
         "20 scene-balanced training tasks × 10. Gains are percentage points over the pre-adaptation base.\n",
         "| method / variant | pair | c1 status | c1 base → teacher (1/2/5 demos) → student | c1 test A | c2 status | "
         "c2 base → teacher → student | c2 test B | c2 test A (prev) | old panel test after c2 | 2 cycles ✓ |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for sp in sorted(glob.glob(os.path.join(ROOT, "E", "*", "p*", "summary.json"))):
        s = json.load(open(sp))
        name = os.path.basename(os.path.dirname(os.path.dirname(sp)))
        pair = "–".join(map(str, s["pair"]))
        cells = []
        for c in (s["cycles"] + [None, None])[:2]:
            if c is None:
                cells += ["–", "–", "–"] if not cells else ["–", "–", "–", "–"]
                continue
            curve = c.get("acquisition_curve", {})
            cv = "/".join(pct(curve.get(str(k), curve.get(k))) for k in (1, 2, 5))
            stud = pct(c.get("new_val")) if c.get("status") not in ("no_useful_acquisition",) else "–"
            st = c["status"] + (" (forced)" if c.get("forced_reset") else "")
            if c["cycle"] == 1:
                cells += [st, f"{pct(c.get('base_val'))} → {cv} → {stud}", pct(c.get("test_new"))]
            else:
                cells += [st, f"{pct(c.get('base_val'))} → {cv} → {stud}", pct(c.get("test_new")), pct(c.get("test_prev"))]
        if len(s["cycles"]) < 2:
            cells = (cells + ["–"] * 7)[:7]
        last = s["cycles"][-1] if s["cycles"] else {}
        done = "✓" if s.get("completed_two_cycles") else ("✗" if s.get("finished") else "running")
        L.append(f"| {name} | {pair} | " + " | ".join(cells) + f" | {pct(last.get('test_old_panel'))} | {done} |")
        a = agg.setdefault(name, {"streams": 0, "finished": 0, "two": 0, "testA": [], "testB": [], "old": []})
        a["streams"] += 1
        a["finished"] += bool(s.get("finished"))
        a["two"] += bool(s.get("completed_two_cycles"))
        if len(s["cycles"]) == 2:
            a["testB"].append(s["cycles"][1].get("test_new"))
            a["testA"].append(s["cycles"][1].get("test_prev"))
            a["old"].append(s["cycles"][1].get("test_old_panel"))
    L.append("\n### Summary over pairs (streams with both cycles)\n")
    L.append("| method / variant | streams (finished) | completed two cycles | mean test B after c2 | mean test A after c2 | old panel |")
    L.append("|---|---|---|---|---|---|")
    for k, a in sorted(agg.items()):
        L.append(f"| {k} | {a['streams']} ({a['finished']}) | {a['two']} | {pct(mean(a['testB']))} | {pct(mean(a['testA']))} | {pct(mean(a['old']))} |")
    return "\n".join(L) + "\n"


def verify():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "selfplay", "verify", "*.json")))]
    if not rows:
        return ""
    L = ["## Correction-source verification (hidden control shifts, training tasks)\n",
         "| task | shift | θ₀ unaware (novice) | oracle θ₀∘g⁻¹ | oracle takes over at step 64 | at step 128 |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['shift']} | " + " | ".join(pct(mean(r[m])) for m in ("novice", "oracle", "takeover64", "takeover128")) + " |")
    return "\n".join(L) + "\n"


def collection():
    fs = glob.glob(os.path.join(ROOT, "selfplay", "data", "*.pt"))
    return f"Self-rollout data files (task × shift): {len(fs)}\n" if fs else ""


def writers():
    L = []
    for lp in sorted(glob.glob(os.path.join(ROOT, "writer", "*", "log.json"))):
        log = json.load(open(lp))
        dev = [r["dev"] for r in log if "dev" in r]
        if not dev:
            continue
        d = dev[-1]
        if not L:
            L = ["## Writer meta-training: offline dev metric (future-correction FM loss on writer-dev tasks)\n",
                 "| writer | step | no write | matched history | mismatched history | improvement (matched vs none) |",
                 "|---|---|---|---|---|---|"]
        imp = (d["none"] - d["matched"]) / d["none"] if d["none"] else None
        L.append(f"| {os.path.basename(os.path.dirname(lp))} | {d['step']} | {d['none']:.4f} | {d['matched']:.4f} | "
                 f"{d['mismatched']:.4f} | {pct(imp)} |")
    return "\n".join(L) + "\n" if L else ""


def r0():
    L = []
    for d in sorted(glob.glob(os.path.join(ROOT, "R0_pilot", "*"))):
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "t*.json")))]
        if not rs:
            continue
        n = max(len(r["attempts"]) for r in rs)
        per = [mean([mean(r["attempts"][i]["success"]) for r in rs if len(r["attempts"]) > i]) for i in range(n)]
        if not L:
            L = ["## Closed-loop R0 pilot: success per attempt (writer-dev tasks × hidden shifts)\n",
                 "| condition | tasks | " + " | ".join(f"attempt {i + 1}" for i in range(5)) + " |",
                 "|---|---|" + "---|" * 5]
        L.append(f"| {os.path.basename(d)} | {len(rs)} | " + " | ".join(pct(x) for x in (per + [None] * 5)[:5]) + " |")
    return "\n".join(L) + "\n" if L else ""


def explore():
    L = ["## New tasks from the robot's own exploration (no demos, no reward) → short memory → consolidation\n",
         "Per held-out task: success with the short memory ON after 1 and 5 explorations; old-task panel (10 training "
         "tasks × 5) with memory ON; after consolidation + reset (memory OFF). θ₀ scores 0–5% on the new tasks.\n",
         "| method | tasks done | new task, 1 exploration | new task, 5 explorations | old tasks, memory ON | new task after consolidation | old tasks after consolidation |",
         "|---|---|---|---|---|---|---|"]
    any_ = False
    for m in ["horsea", "ttt2", "fwrite", "res", "ft"]:
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "explore", m, "t*.json")))]
        if not rs:
            continue
        any_ = True
        g = lambda k, sub=None: mean([(r[k].get(str(sub), r[k].get(sub)) if sub else r[k]) for r in rs])
        L.append(f"| {m} | {len(rs)} | {pct(g('new_mem_on', 1))} | {pct(g('new_mem_on', 5))} | {pct(g('old_mem_on'))} | "
                 f"{pct(g('new_after'))} | {pct(g('old_after'))} |")
    return "\n".join(L) + "\n" if any_ else ""


def fair():
    L = ["## Fair RoboTTT-style comparison: test-time learning from the robot's own rollout (no action target, no reward)\n",
         "10 writer-dev tasks × 6 hidden control shifts × 5 attempts; memory written within each episode (every executed "
         "chunk) and between attempts; all writers meta-trained on the same self-rollout data and objective.\n",
         "| method | condition | tasks | " + " | ".join(f"attempt {i}" for i in range(1, 6)) + " | mean attempts 2–5 |",
         "|---|---|---|" + "---|" * 6]
    any_ = False
    for arm in ["horsea", "ttt2", "fwrite_selfimit", "res_selfimit"]:
        for cond in ["write", "nowrite", "mismatched"]:
            rs = [json.load(open(q)) for q in sorted(glob.glob(os.path.join(ROOT, "fair", f"{arm}_{cond}", "t*.json")))]
            if not rs:
                continue
            any_ = True
            per = [mean([mean(r["attempts"][i]["success"]) for r in rs if len(r["attempts"]) > i]) for i in range(5)]
            L.append(f"| {arm} | {cond} | {len(rs)} | " + " | ".join(pct(x) for x in per) + f" | {pct(mean(per[1:]))} |")
    return "\n".join(L) + "\n" if any_ else ""


def main():
    text = "\n".join(x for x in ["# Protocol v2 results (partial, auto-generated)\n", fair(), explore(), verify(), collection(), writers(), r0(),
                                 e_arm()] if x)
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "REPORT.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
