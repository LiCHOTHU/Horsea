"""Build the static visualization site for the Astra exploration runs.

    python build_site.py --out ../../experiments/astra_explore/site \
        --run libero=../../experiments/astra_explore/run_2026-10-09_astra_t48 \
        --run robotwin=../../experiments/astra_explore/run_2026-10-09_astra_rt_place_container_plate

Pages: index.html (both cases), <case>/index.html (summary, attempts, rendered REPORT.md), <case>/attempt_k.html
(video, contact sheet, one card per decision: observation images, Astra's assessment / intent / hypothesis / expected
change / references, the executed action, the measured effect, the memory rewrite, retries). Images and videos are
served from the run directories through a symlink <case>/run -> run directory (nothing is copied).
"""
import argparse
import html
import json
import os
import re
from pathlib import Path

ESC = html.escape
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6675;--line:#e3e6eb;--ok:#1f8a4c;--warn:#b7791f;--bad:#c0392b;--acc:#2b5fd9}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
header{background:#111827;color:#fff;padding:14px 24px}header a{color:#cbd5e1;margin-right:16px}header .title{font-weight:600;font-size:17px;margin-right:24px}
main{max-width:1380px;margin:0 auto;padding:20px 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}.ok{background:var(--ok)}.warn{background:var(--warn)}.bad{background:var(--bad)}.neutral{background:#6b7280}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:#f1f3f6}
.imgs{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.imgs figure{margin:0;text-align:center}.imgs img{max-height:260px;border:1px solid var(--line);border-radius:6px;background:#000}
.imgs figcaption{font-size:12px;color:var(--muted)}.post img{max-height:170px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-top:8px}.v{margin:2px 0 6px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
.quote{border-left:3px solid #c7d2fe;padding:4px 10px;background:#f8f9ff;border-radius:4px;margin:4px 0}
.evid{border-left:3px solid #bbf7d0;padding:4px 10px;background:#f6fff8;border-radius:4px;margin:4px 0}
.new{background:#fff7d6}.mem li{margin:2px 0}.small{font-size:13px;color:var(--muted)}
.dec{scroll-margin-top:60px}.dec h3{margin:0 0 6px}.nav{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.nav a{font-size:12px;padding:2px 7px;border:1px solid var(--line);border-radius:6px;background:#fff}
video{max-width:100%;border-radius:8px;background:#000}.report{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 22px}
.report table{font-size:13px}.report h2{border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--acc)}
"""


def rjson(p):
    return json.loads(Path(p).read_text())


def page(title, body, nav_links, depth=0):
    root = "../" * depth
    links = "".join(f'<a href="{root}{h}">{ESC(t)}</a>' for t, h in nav_links)
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{ESC(title)}</title><style>{CSS}</style></head><body>'
            f'<header><span class="title">Astra exploration pilots</span>{links}</header><main>{body}</main></body></html>')


def badge(outcome):
    cls = {"success": "ok", "model_stopped": "warn", "budget_exhausted": "warn", "horizon_reached": "warn", "not_started": "neutral"}.get(outcome, "bad")
    txt = {"success": "SUCCESS", "model_stopped": "STOPPED BY ASTRA (no success)", "horizon_reached": "HORIZON REACHED (no success)",
           "budget_exhausted": "BUDGET EXHAUSTED (no success)"}.get(outcome, str(outcome).upper())
    return f'<span class="badge {cls}">{ESC(txt)}</span>'


def load_run(run):
    run = Path(run)
    cfg, res = rjson(run / "config.json"), rjson(run / "result.json")
    attempts = []
    for adir in sorted(run.glob("attempt_*")):
        decs = []
        for ddir in sorted(adir.glob("decision_*")):
            if not (ddir / "decision.json").exists():
                continue
            d = rjson(ddir / "decision.json")
            d["dir"] = ddir
            d["obs"] = rjson(ddir / "observation.json") if (ddir / "observation.json").exists() else {}
            d["pre"] = sorted(p.name for p in ddir.glob("pre_*.png"))
            d["post"] = sorted(p.name for p in ddir.glob("post_*.png"))
            decs.append(d)
        summ = rjson(adir / "summary.json") if (adir / "summary.json").exists() else {}
        ev = [json.loads(l) for l in (adir / "evaluation_only.jsonl").read_text().splitlines()] if (adir / "evaluation_only.jsonl").exists() else []
        video = next((v.name for v in adir.glob("*.mp4")), None)
        attempts.append({"dir": adir, "k": int(adir.name.split("_")[1]), "decisions": decs, "summary": summ, "evaluation": ev, "video": video,
                         "sheet": "contact_sheet.png" if (adir / "contact_sheet.png").exists() else None})
    return {"dir": run, "cfg": cfg, "res": res, "attempts": attempts, "kind": "robotwin" if "left" in json.dumps(cfg.get("task", {})).lower() and "aloha" in json.dumps(cfg).lower() else "libero"}


def fmt_action(c, kind):
    if kind == "libero":
        a = c["action"]
        keys = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        return "[" + " ".join(f"{k}={a[k]:+.2f}" for k in keys) + f"] x repeat {c['repeat']}"
    parts = []
    for arm in ("left", "right"):
        x = c[arm]
        if x["move"]:
            parts.append(f"{arm}: to {[round(v, 3) for v in x['position_m']]} quat(wxyz) {[round(v, 2) for v in x['quaternion_wxyz']]}, gripper {x['gripper']:.2f}")
        else:
            parts.append(f"{arm}: hold pose, gripper {x['gripper']:.2f}")
    return "; ".join(parts)


def fmt_result(r, kind):
    if not r:
        return ""
    if kind == "libero":
        return (f"steps executed {r.get('steps_executed')}; end-effector {r.get('eef_before_m')} → {r.get('eef_after_m')} (Δ {r.get('eef_delta_m')} m); "
                f"finger width {r.get('gripper_width_before_cm')} → {r.get('gripper_width_after_cm')} cm; task_complete = {r.get('task_complete')}")
    out = []
    for arm in ("left", "right"):
        a = r.get("arms", {}).get(arm, {})
        pl = r.get("planner", {}).get(arm, {})
        out.append(f"{arm}: planner {pl.get('status')} ({pl.get('waypoints')} waypoints), moved {a.get('moved_cm')} cm, target error {a.get('position_error_cm')} cm / "
                   f"{a.get('orientation_error_deg')}°, finger-link separation {a.get('finger_link_separation_cm')} cm")
    return "; ".join(out) + f"; physics steps {r.get('sim_steps')}; task_complete = {r.get('task_complete')}"


def obs_text(d, kind):
    o = d.get("obs", {})
    p = o.get("proprio", {})
    if kind == "libero":
        return f"step {o.get('env_step')}: end-effector {p.get('eef_position_m_world')} m, finger width {p.get('gripper_width_cm')} cm"
    l, r = p.get("left", {}), p.get("right", {})
    return (f"motion {o.get('motions_used')}: left ee {l.get('ee_position_m_world')} m (fingers {l.get('finger_link_separation_cm')} cm); "
            f"right ee {r.get('ee_position_m_world')} m (fingers {r.get('finger_link_separation_cm')} cm)")


def mem_html(mem, prev):
    prev = prev or {"observations": [], "hypotheses": [], "summary": ""}
    def ul(items, old):
        return "<ul class='mem'>" + "".join(f"<li class='{'new' if x not in old else ''}'>{ESC(x)}</li>" for x in items) + "</ul>" if items else "<span class='small'>none</span>"
    cls = "new" if mem["summary"] != prev["summary"] else ""
    return (f"<div class='k'>Observations (established facts; new/changed items highlighted)</div>{ul(mem['observations'], prev['observations'])}"
            f"<div class='k'>Hypotheses (unconfirmed)</div>{ul(mem['hypotheses'], prev['hypotheses'])}"
            f"<div class='k'>Running summary</div><div class='v {cls}'>{ESC(mem['summary'])}</div>")


def evidence_html(att, kind):
    ev = att["evaluation"]
    if not ev:
        return ""
    keys = [k for k in ev[0] if k.endswith("_pos")]
    if not keys:
        return ""
    rows = []
    obj = next((k for k in keys if k.split("_pos")[0] in ("ketchup_1", "container", "can", "object")), keys[0])
    import numpy as np
    P = np.array([e[obj] for e in ev if obj in e])
    rows.append(f"<tr><td>{ESC(obj)}</td><td>{P[0].round(3).tolist()}</td><td>{P[-1].round(3).tolist()}</td><td>{P[:, 2].max() - P[0, 2]:+.3f} m</td></tr>")
    succ = [e for e in ev if e.get("success")]
    first = succ[0] if succ else None
    return (f"<details><summary>Recorded environment evidence (evaluation-only; never shown to Astra)</summary><table><tr><th>object</th><th>start</th><th>end</th><th>max height gain</th></tr>"
            + "".join(rows) + "</table>"
            + f"<p class='small'>environment success first recorded at: {ESC(json.dumps({k: first[k] for k in first if k in ('env_step', 'decision', 'motions_used')})) if first else 'never'}; "
            + f"other static objects at start: {ESC(json.dumps({k: ev[0][k] for k in keys if k != obj}))}</p></details>")


def build_attempt(case, run, att, out_dir, nav):
    kind = run["kind"]
    s = att["summary"]
    rel = f"run/{att['dir'].name}"
    b = [f"<h1>{ESC(case.upper())} · attempt {att['k']} {badge(s.get('outcome'))}</h1>",
         f"<p>{ESC(s.get('outcome_text', ''))}. Decisions {s.get('decisions_total')} (executed {s.get('decisions_executed')}), model calls {s.get('model_calls')}, "
         + (f"control steps {s.get('env_steps')} ({s.get('simulated_time_s')} s simulated)" if kind == "libero" else f"motions {s.get('motions')} ({s.get('sim_steps')} physics steps)")
         + f", model time {s.get('model_wall_time_s')} s, attempt wall time {s.get('attempt_wall_time_s')} s.</p>"]
    if att["video"]:
        b.append(f"<div class='card'><h2>Video (one frame per {'control step' if kind == 'libero' else '8 physics steps'}, commanded action overlaid)</h2>"
                 f"<video controls preload='metadata' src='{rel}/{att['video']}'></video></div>")
    if att["sheet"]:
        b.append(f"<div class='card'><h2>Contact sheet: the view before each decision</h2><a href='{rel}/contact_sheet.png'><img style='max-width:100%' src='{rel}/contact_sheet.png'></a></div>")
    b.append(f"<div class='card'><h2>Decision timeline</h2><p class='small'>Each card: what the environment recorded [E], what Astra stated [A] (quoted verbatim), and what changed in its memory. Images are the exact inputs given to Astra at that decision.</p>"
             "<div class='nav'>" + "".join(f"<a href='#d{d['decision']}'>D{d['decision']:02d}</a>" for d in att["decisions"]) + "</div>" + evidence_html(att, kind) + "</div>")
    prev_mem = None
    for d in att["decisions"]:
        c = d.get("decision_content")
        head = f"Decision {d['decision']:02d}" + (f" · interaction id {d.get('interaction_id')}" if d.get("interaction_id") else "")
        calls = "; ".join(f"call {x.get('try', 0) + 1}: {'accepted' if x.get('accepted') else ESC(str(x.get('reason') or x.get('skipped')))} ({x.get('meta', {}).get('latency_s', '?')} s)" for x in d["calls"])
        imgs = "".join(f"<figure><a href='{rel}/{d['dir'].name}/{p}'><img loading='lazy' src='{rel}/{d['dir'].name}/{p}'></a><figcaption>{ESC(p[4:-4])}</figcaption></figure>" for p in d["pre"])
        card = [f"<div class='card dec' id='d{d['decision']}'><h3>{ESC(head)}</h3><div class='small'>{ESC(obs_text(d, kind))} · model calls: {calls}</div>",
                f"<div class='k'>[E] What Astra saw (inputs to this decision)</div><div class='imgs'>{imgs}</div>"]
        if c is None:
            card.append("<div class='v'><span class='badge bad'>NO VALID RESPONSE</span> no action executed; memory unchanged.</div>")
            for x in d["calls"]:
                if x.get("raw") is not None:
                    card.append(f"<details><summary>rejected response</summary><pre class='mono'>{ESC(json.dumps(x['raw'], indent=1))}</pre></details>")
            card.append("</div>")
            b.append("".join(card))
            continue
        rs = [t for x in d["calls"] for t in x.get("meta", {}).get("reasoning_summaries", [])]
        card.append("<div class='grid'><div>")
        card.append(f"<div class='k'>[A] Astra's assessment of its previous action</div><div class='quote'>{ESC(c.get('assessment_of_previous_action') or '(first decision of the attempt: none)')}</div>")
        card.append(f"<div class='k'>[A] Intent</div><div class='quote'>{ESC(c['intent'])}</div>")
        card.append(f"<div class='k'>[A] Hypothesis being tested</div><div class='quote'>{ESC(c.get('hypothesis') or 'none stated')}</div>")
        card.append(f"<div class='k'>[A] Expected change</div><div class='quote'>{ESC(c['expected_change'])}</div>")
        card.append(f"<div class='k'>[A] References to earlier interactions</div><div class='v mono'>{ESC(str(c['references']))}</div>")
        if rs:
            card.append(f"<div class='k'>Codex reasoning summary (model-provided, not an internal transcript)</div><div class='quote small'>{ESC(' | '.join(rs))}</div>")
        card.append("</div><div>")
        if c.get("stop"):
            card.append("<div class='k'>Action</div><div class='v'><span class='badge warn'>STOP REQUESTED</span> no action executed; the attempt ended (not success).</div>")
        else:
            cap = ""
            if kind == "libero" and d.get("repeat_cap_reason"):
                cap = f" (requested repeat {d.get('repeat_requested')}, executed {d.get('repeat_executed')}: {d['repeat_cap_reason']})"
            card.append(f"<div class='k'>[E] Executed action</div><div class='v mono'>{ESC(fmt_action(c, kind))}{ESC(cap)}</div>")
            card.append(f"<div class='k'>[E] Measured effect</div><div class='evid'>{ESC(fmt_result(d.get('result'), kind))}</div>")
            if d.get("result", {}).get("task_complete"):
                card.append("<div class='v'><span class='badge ok'>ENVIRONMENT SUCCESS during this action</span></div>")
            if d["post"]:
                post = "".join(f"<figure><a href='{rel}/{d['dir'].name}/{p}'><img loading='lazy' src='{rel}/{d['dir'].name}/{p}'></a><figcaption>{ESC(p[5:-4])}</figcaption></figure>" for p in d["post"])
                card.append(f"<div class='k'>[E] View after the action</div><div class='imgs post'>{post}</div>")
        card.append("</div></div>")
        card.append(f"<details><summary>Memory after this decision (Astra's rewrite)</summary>{mem_html(c['memory_update'], prev_mem)}</details>")
        card.append(f"<details><summary>Raw records</summary><p class='small mono'>{ESC(str(d['dir'].relative_to(run['dir'])))}/ : prompt.txt, call_*/response.json, call_*/stdout.jsonl, decision.json</p>"
                    f"<a href='{rel}/{d['dir'].name}/prompt.txt'>prompt.txt</a> · <a href='{rel}/{d['dir'].name}/decision.json'>decision.json</a></details></div>")
        prev_mem = d.get("memory_after") or c["memory_update"]
        b.append("".join(card))
    (out_dir / f"attempt_{att['k']}.html").write_text(page(f"{case} attempt {att['k']}", "".join(b), nav, depth=1))


def build_case(case, run, out_dir, nav):
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / "run"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(run["dir"], out_dir))
    cfg, res, kind = run["cfg"], run["res"], run["kind"]
    t = cfg["task"]
    rows = "".join(f"<tr><td>attempt {a['k']}</td><td>{badge(a['summary'].get('outcome'))}</td><td>{a['summary'].get('decisions_total')}</td><td>{a['summary'].get('model_calls')}</td>"
                   f"<td>{a['summary'].get('env_steps') if kind == 'libero' else a['summary'].get('motions')}</td><td>{a['summary'].get('model_wall_time_s')} s</td>"
                   f"<td><a href='attempt_{a['k']}.html'>step-by-step</a> · <a href='run/{a['dir'].name}/{a['video']}'>video</a></td></tr>" for a in run["attempts"])
    summary_items = [("Task", t.get("task_name") or t.get("task")), ("Instruction", t.get("instruction")),
                     ("Interface", t.get("action_type") or f"native {t.get('controller')} deltas, {t.get('control_freq_hz')} Hz, horizon {t.get('attempt_horizon_steps')} steps"),
                     ("Observation access", ", ".join(cfg.get("observation_access", []))), ("Memory", cfg.get("memory_format")),
                     ("Budgets", json.dumps(cfg.get("budgets"))), ("Model", f"{cfg['model'].get('requested')} ({cfg['model'].get('reasoning_effort')}), {cfg['model'].get('codex_version')}, {cfg['model'].get('login_status')}"),
                     ("Run", f"{res.get('status')}; model calls {res.get('model_calls_total')}; wall time {res.get('wall_time_s')} s; started {cfg.get('started')}; code {cfg.get('git')}")]
    summary = "".join(f"<tr><th style='width:180px'>{ESC(k)}</th><td>{ESC(str(v))}</td></tr>" for k, v in summary_items)
    report_html = ""
    rp = run["dir"] / "REPORT.md"
    if rp.exists():
        try:
            import markdown
            report_html = markdown.markdown(rp.read_text(), extensions=["tables", "fenced_code"])
        except Exception:  # noqa: BLE001
            report_html = f"<pre>{ESC(rp.read_text())}</pre>"
    body = (f"<h1>{ESC(case.upper())}: {ESC(t.get('instruction', ''))}</h1><div class='card'><h2>Experiment summary</h2><table>{summary}</table></div>"
            f"<div class='card'><h2>Attempts</h2><table><tr><th>attempt</th><th>outcome</th><th>decisions</th><th>model calls</th><th>{'control steps' if kind == 'libero' else 'motions'}</th><th>model time</th><th>links</th></tr>{rows}</table></div>"
            f"<div class='card'><h2>Narrative report</h2><div class='report'>{report_html}</div></div>")
    (out_dir / "index.html").write_text(page(f"{case}", body, nav, depth=1))
    for att in run["attempts"]:
        build_attempt(case, run, att, out_dir, nav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="case=run_dir (e.g. libero=/path/run)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.run:
        case, path = spec.split("=", 1)
        runs[case] = load_run(Path(path).resolve())
    nav = [("Home", "index.html")] + [(c.upper(), f"{c}/index.html") for c in runs]
    cards = []
    for case, run in runs.items():
        build_case(case, run, out / case, nav)
        t = run["cfg"]["task"]
        att = "".join(f"<li>attempt {a['k']}: {badge(a['summary'].get('outcome'))} {a['summary'].get('decisions_total')} decisions · <a href='{case}/attempt_{a['k']}.html'>step-by-step</a></li>" for a in run["attempts"])
        cards.append(f"<div class='card'><h2><a href='{case}/index.html'>{ESC(case.upper())}</a>: {ESC(t.get('instruction', ''))}</h2>"
                     f"<p class='small'>{ESC(t.get('task_name') or t.get('task'))} · {ESC(t.get('action_type') or 'native delta control')} · model {ESC(run['cfg']['model'].get('requested'))}</p><ul>{att}</ul></div>")
    body = ("<h1>GPT-6 Astra explores a scene: how it observes, hypothesises, acts, measures the effect and remembers</h1>"
            "<p>Two pilots with the same model, memory format and budgets. Each attempt page shows every decision with the exact images Astra saw, "
            "its stated assessment of the previous action, its intent and hypothesis, the expected change, the executed action, the measured effect, "
            "and the memory it rewrote. Videos show the whole attempt with the commanded action overlaid.</p>" + "".join(cards))
    (out / "index.html").write_text(page("Astra exploration pilots", body, nav))
    print("site built at", out, "cases:", list(runs))


if __name__ == "__main__":
    main()
