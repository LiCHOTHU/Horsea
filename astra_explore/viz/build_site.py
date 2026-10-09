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
    obj = next((k for k in keys if k.split("_pos")[0] in ("ketchup_1", "container", "can", "object", "pot", "microphone", "bottle1")), keys[0])
    import numpy as np
    P = np.array([e[obj] for e in ev if obj in e])
    rows.append(f"<tr><td>{ESC(obj)}</td><td>{P[0].round(3).tolist()}</td><td>{P[-1].round(3).tolist()}</td><td>{P[:, 2].max() - P[0, 2]:+.3f} m</td></tr>")
    succ = [e for e in ev if e.get("success")]
    first = succ[0] if succ else None
    return (f"<details><summary>Recorded environment evidence (evaluation-only; never shown to Astra)</summary><table><tr><th>object</th><th>start</th><th>end</th><th>max height gain</th></tr>"
            + "".join(rows) + "</table>"
            + f"<p class='small'>environment success first recorded at: {ESC(json.dumps({k: first[k] for k in first if k in ('env_step', 'decision', 'motions_used')})) if first else 'never'}; "
            + f"other static objects at start: {ESC(json.dumps({k: ev[0][k] for k in keys if k != obj}))}</p></details>")


NARRATIVES = {
    ("libero", 1): ["Approaches the ketchup with the gripper open; closes on the upright bottle and lift-tests five times, but the hand is 3.6-7 cm in front of the bottle (a depth error neither camera shows) and the fingers close on nothing every time.",
                    "Decisions 12-13: the descent knocks the bottle onto its side.",
                    "Decisions 19-22: repositions from the wrist view, grasps the lying bottle's body (finger width 5.7 cm); the lift test passes.",
                    "Decisions 23-28: carries the bottle to the basket, three alignment corrections, lowers and releases; the environment registers success at control step 265."],
    ("libero", 2): ["Starts with attempt 1 in memory and aims for the bottle's body.",
                    "The descent collides with the bottle top (the hand drifts 5 cm, the bottle is pushed 3 cm); Astra reads the contact from the measured displacement, closes (1.1 cm) and the lift test shows the bottle rising 11 cm.",
                    "Carries the bottle straight to attempt 1's recorded release coordinates, lowers 3 cm, releases: success at step 87."],
    ("libero", 3): ["Replays attempt 2's grasp coordinates: five closures on the upright bottle miss by 6-11 cm in depth, each verified as failed by a lift test, with 2-3 cm corrections of alternating sign.",
                    "Decision 18: diagnoses the depth error from the wrist view and moves 6.5 cm back; the next descent tips the bottle over.",
                    "Three more grasps on the lying bottle contact it but do not hold; forward motion stalls at table height; with 22 steps left an unverified grasp is carried toward the basket and the fingers close on nothing.",
                    "Decision 38: Astra stops with 4 steps left, 'without claiming success'."],
    ("robotwin", 1): ["Moves the open left gripper above its estimate of the bowl (5 cm off), corrects from the head image while descending, closes.",
                      "Lift test: the bowl rises with the gripper (finger separation 5.5 cm = holding the rim).",
                      "Carries it to its estimate of the plate, corrects 3 cm, lowers, releases: success after 8 motions."],
    ("robotwin", 2): ["Replays attempt 1's grasp pose; the open descent nudges the bowl and the closure slides off (lift test fails).",
                      "Reopens, corrects, closes deeper; the lift test passes.",
                      "Reuses attempt 1's recorded release pose for transport and lowering: success after 11 motions."],
    ("robotwin", 3): ["Replays attempt 2's grasp pose: fails; descends open and closes separately: fails again.",
                      "Changes strategy: one finger inside the bowl, one outside (off-centre rim pinch), closes twice so the fingers finish closing; the lift test passes.",
                      "Reuses the release pose: success after 16 motions."],
}


def attempt_times(att):
    import datetime as dt
    starts = [p.stat().st_mtime for p in att["dir"].glob("decision_00/observation.json")]
    ends = [p.stat().st_mtime for p in att["dir"].glob("summary.json")]
    f = lambda t: dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    return (f(min(starts)) if starts else "?"), (f(max(ends)) if ends else "?")


def key_numbers(att, kind):
    s = att["summary"]
    decs = [d for d in att["decisions"] if d.get("decision_content")]
    if kind == "libero":
        closes = sum(1 for d in decs if d["decision_content"]["action"]["gripper"] > 0 and all(abs(d["decision_content"]["action"][a]) < 1e-9 for a in ("dx", "dy", "dz")))
        return f"{s.get('decisions_total')} decisions · {s.get('env_steps')} control steps ({s.get('simulated_time_s')} s simulated) · {closes} gripper closures · model time {s.get('model_wall_time_s')} s"
    closes = sum(1 for i, d in enumerate(decs) if d["decision_content"]["left"]["gripper"] < 0.5 and (i == 0 or decs[i - 1]["decision_content"]["left"]["gripper"] >= 0.5))
    return f"{s.get('decisions_total')} decisions (one planned motion each) · {s.get('sim_steps')} physics steps · {closes} grasp attempts · model time {s.get('model_wall_time_s')} s"


def attempt_block(case, att, kind, video_w=640, media_prefix="", page_prefix="", button=True):
    """One self-contained block: when, outcome, embedded video, what happened, link to the decision cards.
    media_prefix: path from the page to the case directory (holds the `run` symlink); page_prefix: path to attempt pages."""
    s = att["summary"]
    t0, t1 = attempt_times(att)
    rel = f"{media_prefix}run/{att['dir'].name}"
    bullets = "".join(f"<li>{ESC(b)}</li>" for b in NARRATIVES.get((case, att["k"]), []))
    video = f"<video controls preload='metadata' style='width:{video_w}px;max-width:100%' src='{rel}/{att['video']}'></video>" if att["video"] else "<p class='small'>no video</p>"
    return (f"<div class='card'><h2>Attempt {att['k']} {badge(s.get('outcome'))}</h2>"
            f"<div class='small'>happened {t0} → {t1} (workstation clock) · {ESC(key_numbers(att, kind))}</div>"
            f"<div class='grid' style='grid-template-columns:{video_w + 20}px 1fr;margin-top:10px'><div>{video}"
            f"<div class='small'>video: the whole attempt, commanded action overlaid · <a href='{rel}/{att['video']}'>open file</a></div></div>"
            f"<div><div class='k'>What happened (author's summary from the trace)</div><ul>{bullets}</ul>"
            + (f"<p><a href='{page_prefix}attempt_{att['k']}.html' class='badge neutral' style='background:var(--acc)'>Open the step-by-step decisions ({s.get('decisions_total')})</a></p>" if button else "")
            + "</div></div></div>")


def decision_table(att, kind, rel):
    rows = []
    for d in att["decisions"]:
        c = d.get("decision_content")
        when = f"step {d.get('env_step_before')}" if kind == "libero" else f"motion {d.get('motions_before')}"
        if not c:
            rows.append(f"<tr><td><a href='#d{d['decision']}'>D{d['decision']:02d}</a></td><td>{when}</td><td colspan='3'><span class='badge bad'>no valid response</span></td></tr>")
            continue
        r = d.get("result", {})
        if c.get("stop"):
            act, res = "<span class='badge warn'>STOP</span>", "attempt ended"
        elif kind == "libero":
            a = c["action"]
            act = ESC("[" + " ".join(f"{a[k]:+.1f}" for k in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")) + f"] x{d.get('repeat_executed')}")
            res = ESC(f"eef Δ {r.get('eef_delta_m')} m; width {r.get('gripper_width_before_cm')}→{r.get('gripper_width_after_cm')} cm")
        else:
            L = c["left"]
            act = ESC(("hold" if not L["move"] else f"left → {[round(v, 2) for v in L['position_m']]}") + f", grip {L['gripper']:.1f}")
            la = r.get("arms", {}).get("left", {})
            res = ESC(f"planner {r.get('planner', {}).get('left', {}).get('status')}, moved {la.get('moved_cm')} cm, fingers {la.get('finger_link_separation_cm')} cm")
        if r.get("task_complete"):
            res += " <span class='badge ok'>SUCCESS</span>"
        rows.append(f"<tr><td><a href='#d{d['decision']}'>D{d['decision']:02d}</a></td><td>{when}</td><td>{ESC(c['intent'])}</td><td class='mono'>{act}</td><td>{res}</td></tr>")
    return ("<table><tr><th>decision</th><th>when</th><th>Astra's stated intent</th><th>action</th><th>measured result</th></tr>" + "".join(rows) + "</table>")


def build_attempt(case, run, att, out_dir, nav):
    kind = run["kind"]
    s = att["summary"]
    rel = f"run/{att['dir'].name}"
    t0, t1 = attempt_times(att)
    b = [f"<h1>{ESC(case.upper())} · attempt {att['k']} {badge(s.get('outcome'))}</h1>",
         f"<p>{ESC(s.get('outcome_text', ''))}. Happened {t0} → {t1} (workstation clock). Decisions {s.get('decisions_total')} (executed {s.get('decisions_executed')}), model calls {s.get('model_calls')}, "
         + (f"control steps {s.get('env_steps')} ({s.get('simulated_time_s')} s simulated)" if kind == "libero" else f"motions {s.get('motions')} ({s.get('sim_steps')} physics steps)")
         + f", model time {s.get('model_wall_time_s')} s, attempt wall time {s.get('attempt_wall_time_s')} s.</p>",
         attempt_block(case, att, kind, video_w=560, media_prefix="", button=False),
         f"<div class='card'><h2>All decisions at a glance</h2><p class='small'>One row per decision; click the id to jump to its card.</p>{decision_table(att, kind, rel)}</div>"]
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
    blocks = "".join(attempt_block(case, a, kind, video_w=640, media_prefix="", page_prefix="") for a in run["attempts"])
    body = (f"<h1>{ESC(case.upper())}: {ESC(t.get('instruction', ''))}</h1>"
            f"<p class='small'>{ESC(t.get('task_name') or t.get('task'))} · {ESC(t.get('action_type') or 'native delta control')} · run started {ESC(cfg.get('started'))}</p>"
            f"<h2 style='margin-top:18px'>The attempts, in order</h2>{blocks}"
            f"<details class='card'><summary><b>Experiment summary (setup, budgets, model)</b></summary><table>{summary}</table>"
            f"<table style='margin-top:10px'><tr><th>attempt</th><th>outcome</th><th>decisions</th><th>model calls</th><th>{'control steps' if kind == 'libero' else 'motions'}</th><th>model time</th><th>links</th></tr>{rows}</table></details>"
            f"<details class='card'><summary><b>Full narrative report</b></summary><div class='report'>{report_html}</div></details>")
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
        blocks = "".join(attempt_block(case, a, run["kind"], video_w=480, media_prefix=f"{case}/", page_prefix=f"{case}/") for a in run["attempts"])
        cards.append(f"<h2 style='margin-top:26px'><a href='{case}/index.html'>{ESC(case.upper())}</a>: {ESC(t.get('instruction', ''))}</h2>"
                     f"<p class='small'>{ESC(t.get('task_name') or t.get('task'))} · {ESC(t.get('action_type') or 'native delta control')} · model {ESC(run['cfg']['model'].get('requested'))} · run started {ESC(run['cfg'].get('started'))}</p>{blocks}")
    body = ("<h1>GPT-6 Astra explores a scene: the attempts, with their videos</h1>"
            "<p>Two pilots with the same model, memory format and budgets. Each attempt is one rollout from the same initial state; "
            "Astra decides one action at a time and carries its memory into the next attempt. Press play to watch an attempt; open the decision cards "
            "to see, for every decision, the images Astra saw, what it stated (assessment, intent, hypothesis, expected change), what it did, "
            "what the environment measured, and how it rewrote its memory.</p>" + "".join(cards))
    (out / "index.html").write_text(page("Astra exploration pilots", body, nav))
    print("site built at", out, "cases:", list(runs))


if __name__ == "__main__":
    main()
