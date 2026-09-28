"""Complete step-by-step tables for the 2-task, 2-cycle test (experiments/cycles_full/<arm>/log.json).

Cycle 1 learns task 57, cycle 2 learns task 66, one task per cycle. Every step measures task 57,
task 66 and the old training tasks.

    python -m horsea.cycles_report        -> prints markdown, writes experiments/cycles_full/REPORT.md
Works on partial runs (reports the steps finished so far).
"""
import json
import os

from horsea.paths import EXP

ARMS = [("fmw", "Horsea"), ("res", "Residual"), ("ttt2", "TTT2"), ("kv", "KV"), ("ft", "Fine-tune")]
ROOT = os.path.join(EXP, "cycles_full")
REF = os.path.join(EXP, "cycles")  # same memory on theta_0 (first 2-task run, same explore procedure)


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load(root, arm):
    p = os.path.join(root, arm, "log.json")
    return json.load(open(p)) if os.path.exists(p) else None


def s(d):
    return {str(k): v for k, v in (d or {}).items()}


def sub(i):
    return "θ" + str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def rows_for(arm, d, start, ref):
    """[(step, 57, 66, old, status)] for one algorithm."""
    tasks = [str(t) for c in d["cycles"] for t in c["tasks"]] or ["57", "66"]
    a, b = (tasks + ["66"])[:2]
    R = []
    if start:
        st = s(start["stream_tasks"])
        R.append(("**0. Start:** original policy θ₀", st.get(a), st.get(b), mean(start["old_tasks"].values()), ""))
    for c in d["cycles"]:
        n, new = c["cycle"], str(c["tasks"][0])
        pt = s(c.get("per_task")).get(new, {})
        if arm != "ft":
            other = s(pt.get("memory_on_other_tasks"))
            sm = {new: pt.get("short_memory_success"), **other}
            ref_txt = ""
            if n > 1 and ref and new in s(ref):
                ref_txt = f" (same memory on θ₀: {pct(s(ref)[new]['short_memory_success'])})"
            R.append((f"**Cycle {n}:** short memory filled with task {new} ({pt.get('demos_used')} demos), "
                      f"memory ON: {sub(n - 1)} + W", sm.get(a), sm.get(b),
                      mean(s(pt.get("memory_on_old_tasks")).values()), ref_txt.strip()))
        for r in c["consolidation_rounds"]:
            allt = {**s(r.get("previous")), **s(r.get("future")), **s(r["new"])}
            ok = ("converted ✓" if r.get("converted") else "not converted ✗") + (
                "" if not r.get("previous") else (", no forgetting ✓" if r.get("no_forgetting") else ", forgetting ✗"))
            what = (f"cycle {n}, fine-tune on task {new} ({pt.get('demos_used')} demos)" if arm == "ft"
                    else f"cycle {n}, consolidation round {r['round']}, memory OFF")
            R.append((what, allt.get(a), allt.get(b),
                      mean(s(r.get("old")).values()), ok))
        af = {**s(c["after"]["stream_tasks"]), **s(c["after"].get("future_tasks"))}
        end = f"**Cycle {n} end:** {sub(n)}" if arm == "ft" else f"**Cycle {n} end:** short memory reset, long memory {sub(n)} alone"
        R.append((end, af.get(a), af.get(b),
                  mean(c["after"]["old_tasks"].values()), ""))
    return a, b, R


def table(name, arm, d, start, ref):
    a, b, R = rows_for(arm, d, start, ref)
    L = [f"### {name}\n", f"| step | task {a} | task {b} | old tasks | note |", "|---|---|---|---|---|"]
    L += [f"| {st} | {pct(x)} | {pct(y)} | {pct(o)} | {note} |" for st, x, y, o, note in R]
    return "\n".join(L) + "\n"


def summary(logs, refs):
    L = ["## All algorithms, key steps (memory OFF unless marked)\n",
         "| algorithm | c1 short memory: task 57 | c1 end: task 57 | c1 end: old | c2 short memory: task 66 (on θ₀) "
         "| c2 end: task 57 (previous) | c2 end: task 66 | c2 end: old |", "|---|---|---|---|---|---|---|---|"]
    for arm, name in ARMS:
        d = logs.get(arm)
        cs = d["cycles"] if d else []
        g = lambda i, k, t=None: (s(cs[i]["after"]["stream_tasks"]).get(t) if k == "st" else mean(cs[i]["after"]["old_tasks"].values())) if len(cs) > i else None
        smv = lambda i, t: s(cs[i].get("per_task")).get(t, {}).get("short_memory_success") if len(cs) > i else None
        ref66 = s(refs.get(arm)).get("66", {}).get("short_memory_success")
        c2sm = "n/a" if arm == "ft" else f"{pct(smv(1, '66'))} ({pct(ref66)})"
        L.append(f"| {name} | {'n/a' if arm == 'ft' else pct(smv(0, '57'))} | {pct(g(0, 'st', '57'))} | {pct(g(0, 'old'))} | {c2sm} "
                 f"| {pct(g(1, 'st', '57'))} | {pct(g(1, 'st', '66'))} | {pct(g(1, 'old'))} |")
    return "\n".join(L) + "\n"


def main():
    logs = {arm: load(ROOT, arm) for arm, _ in ARMS}
    refs = {arm: (load(REF, arm) or {}).get("reference_theta0", {}) for arm, _ in ARMS}
    start = next((d["start"] for d in logs.values() if d and "start" in d), None)
    out = ["# 2-task, 2-cycle test: cycle 1 learns task 57, cycle 2 learns task 66 (one task per cycle)\n",
           "Base trained on the other 80 LIBERO-90 tasks. Short memory is written and run under the generic "
           "instruction \"complete the task\" (only the memory knows the task); every other cell uses the real "
           "instruction. Task 57 / task 66: 20 rollouts each; old tasks: 5 training tasks × 10 rollouts. "
           "\"memory ON\" rows run the whole system θ + W on every task. The end-of-cycle row is the last "
           "consolidation round (same model).\n",
           summary(logs, refs)]
    for arm, name in ARMS:
        d = logs.get(arm)
        if d and d["cycles"]:
            out.append(table(name, arm, d, start, refs.get(arm)))
    text = "\n".join(out)
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "REPORT.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
