"""Final table: train on 80 LIBERO-90 tasks, learn the 10 held-out tasks as 5 two-task pairs (2 cycles each).

    python -m horsea.final_table   -> experiments/protocol_v2/final/TABLE.md
"""
import glob
import json
import os

from horsea.paths import EXP

ROOT = os.path.join(EXP, "protocol_v2", "final")
NAMES = {"horsea": "Horsea", "fwrite": "F-write", "res": "Residual", "ttt2": "TTT2", "ft": "Fine-tune"}


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    head = ("| method | pair | cycle | new task | base (θ before) | new task, short memory ON | old tasks, memory ON | "
            "after consolidation: new task | after consolidation: previous task | after consolidation: old tasks | status |")
    L = ["# Final table: base trained on 80 LIBERO-90 tasks; the 10 held-out tasks learned as 5 pairs, 2 cycles each\n",
         "Short memory = 5 demos, real instruction. New/previous task: 20 validation starts. Old tasks: 10 training tasks "
         "(one per scene) × 5 starts; θ₀ scores 92% on them. Fine-tune has no short memory: its 'memory ON' columns are the "
         "fine-tuned policy itself.\n", head, "|" + "---|" * 11]
    summ = {}
    for m in NAMES:
        for sp in sorted(glob.glob(os.path.join(ROOT, m, "p*", "summary.json"))):
            s = json.load(open(sp))
            pair = "→".join(map(str, s["pair"]))
            for c in s["cycles"]:
                on = c.get("teacher_val")
                old_on = c.get("active_memory_old_panel", c.get("old_val") if m == "ft" else None)
                new_a, prev_a, old_a = c.get("new_val"), c.get("prev_val"), c.get("old_val")
                st = c["status"] + (" (forced reset)" if c.get("forced_reset") else "")
                L.append(f"| {NAMES[m]} | {pair} | {c['cycle']} | {c['task']} | {pct(c.get('base_val'))} | {pct(on)} | {pct(old_on)} | "
                         f"{pct(new_a)} | {pct(prev_a) if c['cycle'] == 2 else '–'} | {pct(old_a)} | {st} |")
                d = summ.setdefault(m, {k: [] for k in ("on", "old_on", "new_a", "prev_a", "old_a")})
                d["on"].append(on); d["old_on"].append(old_on); d["new_a"].append(new_a); d["old_a"].append(old_a)
                if c["cycle"] == 2:
                    d["prev_a"].append(prev_a)
    L += ["\n## Mean over all pairs and cycles\n",
          "| method | new task, short memory ON | old tasks, memory ON | after consolidation: new task | previous task after cycle 2 | old tasks after consolidation |",
          "|---|---|---|---|---|---|"]
    for m, d in summ.items():
        L.append(f"| {NAMES[m]} | {pct(mean(d['on']))} | {pct(mean(d['old_on']))} | {pct(mean(d['new_a']))} | "
                 f"{pct(mean(d['prev_a']))} | {pct(mean(d['old_a']))} |")
    text = "\n".join(L) + "\n"
    os.makedirs(ROOT, exist_ok=True)
    open(os.path.join(ROOT, "TABLE.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
