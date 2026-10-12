"""Series C of the site: what System 1 is useful beneath Astra (dex hand, 2026-10-10/11).

Called by build_site.py (`--s1local <experiments dir>`). Writes <site>/s1local/index.html (question, answer, status of the
five tasks, the 11-operation outcome matrix, pooled contrasts, the scalability report) and one page per task (Astra's source
program, operations, per-condition outcome grid, curves / fresh-start / trajectory plots, re-executed videos, the pilot's
result tables). Media come from each pilot directory through the symlink s1local/<task>_run.
"""
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1local.crosstask import classify  # noqa: E402

ESC = html.escape
# key, title, pilot dir, Astra run dir (Series A case), source attempt, Series A case key
TASKS = [("drawer", "Stack-Can-Into-Drawer (first pilot)", "s1local_2026-10-10", "run_2026-10-10_astra_isaac_stack_can_drawer", 10, "dexhand"),
         ("unload", "Unload-Cans", "s1local_unload_cans_v2_2026-10-10", "run_2026-10-10_astra_isaac_unload_cans", 2, "dex_unload"),
         ("laptop", "Open-Laptop", "s1local_open_laptop_2026-10-10", "run_2026-10-10_astra_isaac_open_laptop_r3", 1, "dex_laptop"),
         ("insert", "Insert-And-Unload-Cans (insert stage)", "s1local_insert_and_unload_cans_2026-10-11", "run_2026-10-10_astra_isaac_insert_and_unload_cans_r4", 5, "dex_insert")]
UNREPRODUCED = {("insert", "grasp_right_can")}
CONDS = [(1.0, 1.0), (0.8, 0.7), (1.2, 1.0), (1.0, 0.7), (0.8, 1.0), (1.2, 0.7)]
CLS = {"ceiling": ("#2563eb", "original succeeds on the first try"), "noise": ("#d97706", "first try fails, repeating the original recovers"),
       "rescue": ("#16a34a", "rescued by local search"), "none": ("#dc2626", "no success in 9 attempts"),
       "unreproduced": ("#94a3b8", "start not reproduced (lead-in replay diverged)")}
EXTRA_CSS = """
.cm td{text-align:center;vertical-align:middle;font-size:13px}.cm td.l{text-align:left}.sq{display:inline-block;width:30px;height:22px;border-radius:4px;color:#fff;line-height:22px;font-weight:700;font-size:12px;text-align:center}
.legend span{display:inline-block;margin-right:14px;font-size:12px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.kpi{display:flex;gap:12px;flex-wrap:wrap}.kpi div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:160px}.kpi b{display:block;font-size:22px}
.og{border-collapse:separate;border-spacing:3px;width:auto}.og td,.og th{border:none;padding:0 2px;font-size:12px}.og th{background:none;color:var(--muted);font-weight:500}
.og .S{background:#16a34a}.og .F{background:#dc2626}.og .N{background:#e5e7eb;color:#9ca3af}.og td.rl{padding-right:10px;white-space:nowrap;font-size:13px;text-align:left}
.vids{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}.vid{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px}
.plots img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
"""


def rj(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return default


def rl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def md(path):
    p = Path(path)
    if not p.exists():
        return ""
    try:
        import markdown
        return markdown.markdown(p.read_text(), extensions=["tables", "fenced_code"])
    except Exception:  # noqa: BLE001
        return f"<pre>{ESC(p.read_text())}</pre>"


def link(out_dir, name, target):
    ln = out_dir / name
    if ln.is_symlink() or ln.exists():
        ln.unlink()
    ln.symlink_to(os.path.relpath(target, out_dir))


def block_rows(pilot):
    """{op: {cond: {"A": {...}, "B": ..., "C": ...}}} with the analysis fields crosstask.classify expects."""
    out = {}
    for b in rl(pilot / "blocks.jsonl"):
        if not b.get("selected"):
            continue
        row = {}
        for a in "ABC":
            ev = [x for x in b["eval_success"][a] if x is not None]
            row[a] = {"first_success": b["first_success"][a], "eval_success": sum(ev), "n_eval": len(ev), "eval_raw": b["eval_success"][a]}
        out.setdefault(b["op"], {})[b["cond"]] = row
    return out


def matrix_html(exp):
    rows, tot = [], {"cls": {}, "ev": {a: [0, 0] for a in "ABC"}}
    for key, title, pdir, _, _, _ in TASKS:
        pilot = exp / pdir
        for op, conds in block_rows(pilot).items():
            cells = []
            for ci in range(6):
                r = conds.get(ci)
                if r is None:
                    cells.append("<td><span class='sq' style='background:#e5e7eb;color:#9ca3af'>-</span></td>")
                    continue
                c = "unreproduced" if (key, op) in UNREPRODUCED else classify(r)
                tot["cls"][c] = tot["cls"].get(c, 0) + 1
                fs = r["A"]["first_success"]
                cells.append(f"<td title='{ESC(CLS[c][1])}; first success A/B/C: {r['A']['first_success']}/{r['B']['first_success']}/{r['C']['first_success']}'>"
                             f"<span class='sq' style='background:{CLS[c][0]}'>{fs if fs else '-'}</span></td>")
            ev = []
            for a in "ABC":
                k = sum(x[a]["eval_success"] for x in conds.values())
                n = sum(x[a]["n_eval"] for x in conds.values())
                tot["ev"][a][0] += k
                tot["ev"][a][1] += n
                ev.append(f"{k}/{n}")
            rows.append(f"<tr><td class='l'><a href='{key}.html'>{ESC(title.split(' (')[0])}</a></td><td class='l'>{ESC(op.replace('_', ' '))}</td>{''.join(cells)}"
                        f"<td>{' / '.join(ev)}</td></tr>")
    head = "".join(f"<th>m {m} · f {f}</th>" for m, f in CONDS)
    legend = "".join(f"<span><i style='background:{c}'></i>{ESC(t)}</span>" for c, t in CLS.values())
    totals = " · ".join(f"{CLS[k][1]}: <b>{v}</b>" for k, v in tot["cls"].items())
    return (f"<table class='cm'><tr><th>task</th><th>operation</th>{head}<th>fresh-start success A / B / C</th></tr>{''.join(rows)}</table>"
            f"<p class='legend' style='margin-top:8px'>{legend}</p><p class='small'>Number in a square: the attempt at which repeating Astra's original "
            f"program (arm A) first succeeded. Column = mass factor m and friction factor f of the condition. Totals: {totals}. Fresh starts: "
            f"A {tot['ev']['A'][0]}/{tot['ev']['A'][1]}, B {tot['ev']['B'][0]}/{tot['ev']['B'][1]}, C {tot['ev']['C'][0]}/{tot['ev']['C'][1]}.</p>"), tot


def pooled_html(ct_dir):
    rows = []
    for label, p in (("all 11 operations", ct_dir / "crosstask.json"), ("without the unreproduced start", ct_dir / "excluding_unreproduced_start" / "crosstask.json")):
        d = rj(p, {})
        for name, v in d.get("pooled", {}).items():
            f = lambda x: "n/a" if x is None else f"{x:+.1f}"  # noqa: E731
            rows.append(f"<tr><td>{ESC(name)}</td><td>{ESC(label)} ({v['strata']} strata)</td><td><b>{f(v['fresh_start_pp'])} pp</b></td>"
                        f"<td>[{f(v['ci95_pp'][0])}, {f(v['ci95_pp'][1])}]</td><td>{f(v['attempts_saved'])}</td><td>[{f(v['ci95_attempts'][0])}, {f(v['ci95_attempts'][1])}]</td></tr>")
    return ("<table><tr><th>contrast</th><th>pooled over</th><th>fresh-start success difference</th><th>95 % CI</th>"
            "<th>attempts saved to first success (censored at 10)</th><th>95 % CI</th></tr>" + "".join(rows) + "</table>"
            "<p class='small'>H1: B (fixed-distribution local search) minus A (repeat Astra's program); H2: C (history-adaptive search) minus B. "
            "Paired, operation-stratified bootstrap over conditions. Negative attempts saved = the search reached its first success later.</p>")


def outcome_grid(conds):
    """Per arm and condition: first-success attempt and the three fresh starts (S / F / invalid)."""
    head = "<tr><th></th>" + "".join(f"<th colspan='4'>m {m} · f {f}</th>" for m, f in CONDS) + "</tr>"
    sub = "<tr><th></th>" + "<th>1st</th><th>e1</th><th>e2</th><th>e3</th>" * 6 + "</tr>"
    rows = []
    for a, name in (("A", "A original"), ("B", "B local search"), ("C", "C history-adaptive")):
        cells = []
        for ci in range(6):
            r = conds.get(ci)
            if r is None:
                cells.append("<td colspan='4' class='small'>skipped</td>")
                continue
            fs = r[a]["first_success"]
            cells.append(f"<td><span class='sq' style='background:{'#2563eb' if fs == 1 else '#d97706' if fs else '#dc2626'}'>{fs or '-'}</span></td>")
            for x in r[a]["eval_raw"]:
                cls, t = ("S", "S") if x == 1 else ("F", "F") if x == 0 else ("N", "·")
                cells.append(f"<td><span class='sq {cls}'>{t}</span></td>")
        rows.append(f"<tr><td class='rl'>{ESC(name)}</td>{''.join(cells)}</tr>")
    return (f"<table class='og'>{head}{sub}{''.join(rows)}</table><p class='small'>1st: adaptation attempt of the first success (blue = the first "
            "try, amber = later, red '-' = none in 9). e1-e3: the fresh evaluation starts (objects shifted 8 mm at 0 / 120 / 240 deg; laptop: the laptop; "
            "drawer pilot door: door 0.03-0.09 rad less open); S success, F failure, · invalid start (the replayed lead-in did not reach the operation's start).</p>")


def task_page(exp, site_dir, key, title, pdir, run, att, case, page_fn):
    pilot = exp / pdir
    link(site_dir, f"{key}_run", pilot)
    man = rj(pilot / "manifest.json", {})
    ops = man.get("operations", {})
    op_rows = []
    for op, o in ops.items():
        kind = o.get("kind") or {"grasp": "grasp", "transport": "transport", "door": "push"}.get(op, "")
        prog = o.get("program") or o.get("program_decisions")
        pre = o.get("prefix") or o.get("prefix_decisions")
        par = o.get("params") or {"grasp": ["finger_base", "finger_tip", "thumb_rotate", "thumb_bend"], "transport": ["speed", "finger_base (held)", "finger_tip (held)", "thumb_bend (held)"],
                                  "push": ["contact dx (m)", "contact dz (m)", "push direction (deg)", "speed"]}.get(kind, [])
        rng = ", ".join(f"{ESC(str(n))} {r:g} [{lo:g}, {hi:g}]" for n, r, lo, hi in zip(par, o.get("reference", []), o.get("lower", []), o.get("upper", [])))
        op_rows.append(f"<tr><td><b>{ESC(op.replace('_', ' '))}</b></td><td>{ESC(kind)}</td><td>{ESC(str(o.get('hand', 'right' if op != 'door' else 'left')))}</td>"
                       f"<td>{ESC(str(o.get('object', '')))}</td><td>D{min(pre) if pre else 0:02d}-D{max(pre) if pre else 0:02d}</td>"
                       f"<td>{', '.join(f'D{i:02d}' for i in (prog or []))}</td><td class='small'>{rng}</td></tr>")
    rows = block_rows(pilot)
    grids = "".join(f"<div class='card'><h3 style='margin-top:0'>{ESC(op.replace('_', ' '))}"
                    + (" <span class='badge neutral' style='background:#94a3b8'>start not reproduced</span>" if (key, op) in UNREPRODUCED else "")
                    + f"</h3>{outcome_grid(conds)}</div>" for op, conds in rows.items())
    plots = "".join(f"<figure style='margin:0 0 12px'><a href='{key}_run/{f}'><img src='{key}_run/{f}'></a><figcaption class='small'>{ESC(c)}</figcaption></figure>"
                    for f, c in (("curves.png", "success vs adaptation attempt (per arm)"), ("eval.png", "fresh-start success (per arm and operation)"),
                                 ("trajectories.png", "representative paired trajectories")) if (pilot / f).exists())
    vids = []
    for v in rj(pilot / "videos" / "videos.json", []) or []:
        lab = v.get("label")
        ev = v.get("re_executed", {})
        ok = ev.get("S") == 1
        poster = f" poster='{key}_run/videos/{lab}.jpg'" if (pilot / "videos" / f"{lab}.jpg").exists() else ""
        vids.append(f"<div class='vid'><b>{ESC(lab.replace('_', ' '))}</b> <span class='badge {'ok' if ok else 'bad'}'>{'SUCCESS' if ok else 'NO SUCCESS'}</span>"
                    f"<div class='small'>operation {ESC(str(v.get('op')))} · condition {v.get('cond')} · start {ESC(str(v.get('start')))} · J {ev.get('J')}</div>"
                    f"<video controls preload='none'{poster} src='{key}_run/videos/{lab}.mp4'></video></div>")
    vid_html = (f"<div class='card'><h2>Videos (re-executed rollouts)</h2><p class='small'>Each video re-executes one logged rollout and states its "
                f"own outcome; restores are not bit-exact, so a re-execution can differ from the logged result.</p><div class='vids'>{''.join(vids)}</div></div>") if vids else ""
    report = md(pilot / "REPORT.md")
    body = (f"<p class='small'><a href='index.html'>← Series C overview</a></p><h1>{ESC(title)}</h1>"
            f"<p>Task <b>{ESC(man.get('task') or man.get('environment', '').split(',')[0])}</b>. Source program: Astra's attempt {att} of "
            f"<a href='../{case}/attempt_{att}.html'>{ESC(run)}</a> (its decisions, replayed open-loop). "
            "Arms per operation and condition: <b>A</b> repeats Astra's program 9 times; <b>B</b> samples bounded variations of up to four parameters from a fixed "
            "distribution; <b>C</b> samples the same way but updates the distribution from its own history after attempt 5. The best-scoring candidate of "
            "each arm is then run on three fresh starts.</p>"
            f"<div class='card'><h2>Operations</h2><table><tr><th>operation</th><th>template</th><th>hand</th><th>object</th><th>lead-in replayed</th>"
            f"<th>program decisions</th><th>parameters: reference [bounds]</th></tr>{''.join(op_rows)}</table></div>"
            f"<h2>Outcomes per condition</h2>{grids}"
            f"<div class='card plots'><h2>Plots</h2>{plots}</div>{vid_html}"
            f"<details class='card'><summary><b>Result tables (results.md)</b></summary><div class='report'>{md(pilot / 'results.md')}</div></details>"
            + (f"<details class='card'><summary><b>Full pilot report (REPORT.md)</b></summary><div class='report'>{report}</div></details>" if report else ""))
    (site_dir / f"{key}.html").write_text(page_fn(title, f"<style>{EXTRA_CSS}</style>{body}", 1))


def doc_page(site_dir, name, title, md_path, intro, page_fn):
    """A rendered markdown note (diagnosis, design memo) as its own Series C page."""
    body = (f"<p class='small'><a href='index.html'>← Series C overview</a></p><p class='small'>{intro}</p>"
            f"<div class='report'>{md(md_path)}</div>")
    (site_dir / name).write_text(page_fn(title, f"<style>{EXTRA_CSS}</style>{body}", 1))


def build(site_out, exp_root, page_fn):
    exp = Path(exp_root).resolve()
    site_dir = Path(site_out) / "s1local"
    site_dir.mkdir(parents=True, exist_ok=True)
    diag = exp / "s1_diagnosis_2026-10-11" / "DIAGNOSIS.md"
    design = exp.parent / "research" / "system1_dexhand_design.md"
    if diag.exists():
        doc_page(site_dir, "diagnosis.html", "Why grasps fail", diag,
                 "Log-only diagnosis of Astra's own runs: does the robot reach what Astra commands, and how do dex-hand grasps fail. "
                 "Source: experiments/s1_diagnosis_2026-10-11 (script astra_explore/s1local/diagnose_grasps.py).", page_fn)
    if design.exists():
        doc_page(site_dir, "design.html", "System 1 design", design,
                 "Research note (2026-10-11): the boundary between System 1 and System 2, the contract interface, short-term memory, "
                 "distribution-level updates, and the step-by-step plan, with references checked by four literature surveys. "
                 "Source: research/system1_dexhand_design.md.", page_fn)
    ct = exp / "s1local_crosstask_2026-10-10"
    mx, tot = matrix_html(exp)
    for key, title, pdir, run, att, case in TASKS:
        task_page(exp, site_dir, key, title, pdir, run, att, case, page_fn)
    n = sum(tot["cls"].values())
    status = ("<table><tr><th>task</th><th>Astra (System 2)</th><th>pilot</th></tr>"
              "<tr><td>Stack-Can-Into-Drawer</td><td>success at attempt 10 of 10</td><td><a href='drawer.html'>done</a>: 3 operations (first pilot)</td></tr>"
              "<tr><td>Unload-Cans</td><td>success at attempt 2</td><td><a href='unload.html'>done</a>: 3 operations</td></tr>"
              "<tr><td>Open-Laptop</td><td>success at attempt 1 (lid 1.270 rad vs 1.257 needed)</td><td><a href='laptop.html'>done</a>: 1 operation</td></tr>"
              "<tr><td>Insert-And-Unload-Cans</td><td>no full success in 5 attempts; attempt 5 put both cans into the tray, then ran out of time</td>"
              "<td><a href='insert.html'>done on the insert stage</a>: 4 operations</td></tr>"
              "<tr><td>Sort-Cans</td><td><span class='badge neutral' style='background:#94a3b8'>not run</span> Astra blocked by the workspace spend cap</td><td>-</td></tr></table>")
    body = (f"<style>{EXTRA_CSS}</style><h1>Series C · What System 1 is useful beneath Astra? (dex hand)</h1>"
            "<p>Astra (System 2) writes an open-loop program of hand motions. Would a fast System 1 underneath help, if all it does is adjust "
            "a few parameters of an operation (grip closures, speed, contact offsets) by local search, with or without memory of its own recent tries? "
            "First pilot on Stack-Can-Into-Drawer, then the same protocol on four more tasks of the same benchmark, hand and harness.</p>"
            "<div class='card series' style='border-left-color:#7c3aed'><h2 style='margin-top:0'>Answer</h2>"
            f"<div class='kpi'><div><span class='small'>matched blocks</span><b>{n}</b></div><div><span class='small'>rescued by local search</span><b>{tot['cls'].get('rescue', 0)}</b></div>"
            f"<div><span class='small'>fresh-start success A / B / C</span><b>{tot['ev']['A'][0]} / {tot['ev']['B'][0]} / {tot['ev']['C'][0]}</b><span class='small'>of {tot['ev']['A'][1]}</span></div>"
            "<div><span class='small'>B − A (H1)</span><b>0.0 pp</b><span class='small'>95 % CI [−1.0, +1.0]</span></div><div><span class='small'>C − B (H2)</span><b>0.0 pp</b><span class='small'>every operation</span></div></div>"
            "<p><b>The first pilot's conclusion held on all four tasks we could run.</b> Local search never rescued a block that repeating Astra's program "
            "did not also recover, and memory of recent tries changed nothing; searching only made the first success later. What fails is not what "
            "local search addresses: open-loop pushes that stop just short (door, laptop lid), grasps that work on the tuning start but fail on starts "
            "shifted by 8 mm, and long contact-heavy lead-ins that do not replay the same way. Useful instead: closed-loop completion of the final "
            "motion, closed-loop re-acquisition of the object, and scoring candidates on several starts.</p></div>"
            "<div class='card'><h2>Since the pilots</h2><p><a href='diagnosis.html'><b>Why grasps fail</b></a>: the robot reaches what "
            "Astra commands (0.3 cm median with grippers); on the dex hand 65 % of failed grasps close on air (Astra aimed wrong), "
            "35 % are contact failures (touched but not held, pushed away, slipped).</p>"
            "<p><a href='design.html'><b>System 1 design</b></a>: the boundary, the contract interface (object-relative target motion, "
            "tolerance, success check), short-term memory on a frozen pretrained base, distribution-level updates, the plan, and "
            "the verified literature.</p></div>"
            f"<div class='card'><h2>Tasks</h2>{status}</div>"
            f"<div class='card'><h2>Every operation × condition</h2>{mx}</div>"
            f"<div class='card'><h2>Pooled contrasts</h2>{pooled_html(ct)}</div>"
            f"<details class='card'><summary><b>Full scalability report (SCALABILITY.md)</b></summary><div class='report'>{md(ct / 'SCALABILITY.md')}</div></details>")
    (site_dir / "index.html").write_text(page_fn("System 1 beneath Astra", body, 1))
    return tot
