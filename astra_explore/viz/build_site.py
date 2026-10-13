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
header{background:#111827;color:#fff;padding:0 24px;height:54px;display:flex;align-items:stretch;gap:6px;position:sticky;top:0;z-index:50;box-shadow:0 1px 0 rgba(255,255,255,.06)}
header .title{display:flex;align-items:center;font-weight:650;font-size:17px;margin-right:22px;white-space:nowrap}header .title a{color:#fff}header .title a:hover{text-decoration:none}
.menu{display:flex;align-items:stretch;gap:2px}.menu .top{display:flex;align-items:center;gap:6px;color:#cbd5e1;padding:0 14px;font:inherit;font-size:14px;background:none;border:0;cursor:pointer;white-space:nowrap}
.menu .top:hover,.dd:hover>.top,.dd:focus-within>.top{color:#fff;background:#1f2937;text-decoration:none}.menu .top .tag{font-size:11px;font-weight:700;color:#111827;background:#cbd5e1;border-radius:4px;padding:0 5px}
.menu .top .car{font-size:10px;opacity:.7}.dd{position:relative;display:flex}
.dd-menu{display:none;position:absolute;top:100%;left:0;min-width:300px;background:#fff;border:1px solid var(--line);border-top:3px solid var(--c,#2b5fd9);border-radius:0 0 10px 10px;box-shadow:0 14px 34px rgba(17,24,39,.18);padding:6px;z-index:60}
.dd:hover .dd-menu,.dd:focus-within .dd-menu{display:block}.dd-menu a{display:block;padding:8px 12px;border-radius:7px;color:var(--ink);font-size:14px;font-weight:550;white-space:nowrap}
.dd-menu a:hover{background:#f1f5f9;text-decoration:none}.dd-menu a .sub{display:block;font-size:12px;font-weight:400;color:var(--muted);white-space:normal}
.dd-menu .hd{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;padding:8px 12px 2px}.dd-menu hr{border:0;border-top:1px solid var(--line);margin:4px 6px}
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
.dots span{display:inline-block;width:22px;height:20px;border-radius:4px;color:#fff;text-align:center;line-height:20px;font-size:11px;font-weight:700;margin-right:3px}
.dots .S{background:#16a34a}.dots .T{background:#d97706}.dots .F{background:#dc2626}
.cases{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px}.case{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
.series{border-left:5px solid var(--acc);padding-left:14px}
video{max-width:100%;border-radius:8px;background:#000}.report{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 22px}
.report table{font-size:13px}.report h2{border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--acc)}
"""


def rjson(p):
    return json.loads(Path(p).read_text())


def nav_html(nav_links, root):
    """nav_links: [(label, href)] or grouped [(group, [items])]; a group "" holds plain top-level links, any other group is a
    dropdown (hover or keyboard / tap focus). A group may be (name, tag, colour); an item is (label, href[, subtitle]) or
    ("#", heading) for a section heading inside the dropdown."""
    def item(it):
        if it[0] == "#":
            return f"<div class='hd'>{ESC(it[1])}</div>"
        sub = f"<span class='sub'>{ESC(it[2])}</span>" if len(it) > 2 and it[2] else ""
        return f'<a href="{root}{it[1]}">{ESC(it[0])}{sub}</a>'
    if not (nav_links and isinstance(nav_links[0][1], list)):
        nav_links = [("", nav_links)]
    out = []
    for g, links in nav_links:
        if not g:
            out += [f'<a class="top" href="{root}{h}">{ESC(t)}</a>' for t, h, *_ in links]
            continue
        name, tag, col = (g, "", "#2b5fd9") if isinstance(g, str) else g
        tag_html = f"<span class='tag' style='background:{col};color:#fff'>{ESC(tag)}</span>" if tag else ""
        out.append(f"<div class='dd' style='--c:{col}'><button class='top' type='button'>{tag_html}{ESC(name)} <span class='car'>&#9662;</span></button>"
                   f"<div class='dd-menu'>{''.join(item(it) for it in links)}</div></div>")
    return "<nav class='menu'>" + "".join(out) + "</nav>"


def page(title, body, nav_links, depth=0):
    root = "../" * depth
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{ESC(title)}</title><style>{CSS}</style></head><body>'
            f'<header><span class="title"><a href="{root}index.html">Astra studies</a></span>{nav_html(nav_links, root)}</header>'
            f'<main>{body}</main></body></html>')


def badge(outcome):
    cls = {"success": "ok", "model_stopped": "warn", "budget_exhausted": "warn", "horizon_reached": "warn", "not_started": "neutral", "interrupted": "neutral"}.get(outcome, "bad")
    txt = {"success": "SUCCESS", "model_stopped": "STOPPED BY ASTRA (no success)", "horizon_reached": "HORIZON REACHED (no success)",
           "budget_exhausted": "BUDGET EXHAUSTED (no success)", "interrupted": "INTERRUPTED (Astra unavailable)"}.get(outcome, str(outcome).upper())
    return f'<span class="badge {cls}">{ESC(txt)}</span>'


def load_run(run):
    run = Path(run)
    cfg, res = rjson(run / "config.json"), rjson(run / "result.json")
    attempts = []
    for adir in sorted(run.glob("attempt_*"), key=lambda p: int(p.name.split("_")[1])):
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
        summ = rjson(adir / "summary.json") if (adir / "summary.json").exists() else {
            "outcome": "interrupted", "outcome_text": "interrupted: Astra became unavailable (not counted as an attempt)", "decisions_total": len(decs)}
        ev = [json.loads(l) for l in (adir / "evaluation_only.jsonl").read_text().splitlines()] if (adir / "evaluation_only.jsonl").exists() else []
        video = next((v.name for v in adir.glob("*.mp4")), None)
        attempts.append({"dir": adir, "k": int(adir.name.split("_")[1]), "decisions": decs, "summary": summ, "evaluation": ev, "video": video,
                         "sheet": "contact_sheet.png" if (adir / "contact_sheet.png").exists() else None})
    sim = json.dumps(cfg.get("task", {})).lower()
    kind = "dexhand" if "isaac" in sim else "robotwin" if "aloha" in json.dumps(cfg).lower() else "libero"
    return {"dir": run, "cfg": cfg, "res": res, "attempts": attempts, "kind": kind}


def fmt_action(c, kind):
    if kind == "libero":
        a = c["action"]
        keys = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        return "[" + " ".join(f"{k}={a[k]:+.2f}" for k in keys) + f"] x repeat {c['repeat']}"
    parts = []
    if kind == "dexhand":
        for h in ("left", "right"):
            x = c[h]
            cl = f"hand base {x['finger_base']:.2f} tip {x['finger_tip']:.2f} thumb rotate {x['thumb_rotate']:.2f} bend {x['thumb_bend']:.2f}"
            parts.append(f"{h}: wrist to {[round(v, 3) for v in x['position_m']]} fingers->{[round(v, 2) for v in x['fingers_dir']]} palm->{[round(v, 2) for v in x['palm_dir']]}, {cl}" if x["move"] else f"{h}: hold arm, {cl}")
        return "; ".join(parts) + f"; {c['steps']} control steps"
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
    if kind == "dexhand":
        for h in ("left", "right"):
            a = r.get("hands", {}).get(h, {})
            if a.get("tracked"):
                out.append(f"{h}: moved {a.get('moved_cm')} cm, target error {a.get('position_error_cm')} cm / {a.get('orientation_error_deg')}°, thumb-index {a.get('thumb_index_distance_cm')} cm, "
                           f"closure measured {a.get('closure_measured')}, index contact {a.get('index_contact_force_N')} N")
            else:
                out.append(f"{h}: arm held (moved {a.get('moved_cm')} cm), thumb-index {a.get('thumb_index_distance_cm')} cm, closure measured {a.get('closure_measured')}, index contact {a.get('index_contact_force_N')} N")
        clip = f" ({r.get('clipped')})" if r.get("clipped") else ""
        return "; ".join(out) + f"; {r.get('steps_executed')} control steps executed{clip}; task_complete = {r.get('task_complete')}"
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
    if kind == "dexhand":
        return (f"step {o.get('steps_used')} ({o.get('steps_remaining')} left): left wrist {l.get('wrist_position_m_world')} m (thumb-index {l.get('thumb_index_distance_cm')} cm, contact {l.get('index_contact_force_N')} N); "
                f"right wrist {r.get('wrist_position_m_world')} m (thumb-index {r.get('thumb_index_distance_cm')} cm, contact {r.get('index_contact_force_N')} N)")
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
    extra = ""
    if "drawer_joint_pos" in ev[0]:
        extra = f"<p class='small'>drawer joints start {ev[0].get('drawer_joint_pos')} → end {ev[-1].get('drawer_joint_pos')} (limits {ev[0].get('drawer_joint_limits')}); handle at start {ev[0].get('drawer_handle_pos')}; final subtask flags {ESC(json.dumps({k: ev[-1].get(k) for k in ('reach_drawer_success', 'reach_can_success', 'lift_success', 'move_door_success', 'is_drawer_close') if k in ev[-1]}))}</p>"
    import numpy as np
    P = np.array([e[obj] for e in ev if obj in e])
    rows.append(f"<tr><td>{ESC(obj)}</td><td>{P[0].round(3).tolist()}</td><td>{P[-1].round(3).tolist()}</td><td>{P[:, 2].max() - P[0, 2]:+.3f} m</td></tr>")
    succ = [e for e in ev if e.get("success")]
    first = succ[0] if succ else None
    return (f"<details><summary>Recorded environment evidence (evaluation-only; never shown to Astra)</summary><table><tr><th>object</th><th>start</th><th>end</th><th>max height gain</th></tr>"
            + "".join(rows) + "</table>"
            + f"<p class='small'>environment success first recorded at: {ESC(json.dumps({k: first[k] for k in first if k in ('env_step', 'decision', 'motions_used')})) if first else 'never'}; "
            + f"other static objects at start: {ESC(json.dumps({k: ev[0][k] for k in keys if k != obj}))}</p>{extra}</details>")


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
    ("bimanual", 1): ["First targets (both grippers 1.08 m high above the handles) are unreachable: nothing moves. Astra recovers with closer intermediate poses, turns the grippers down and straddles both handles.",
                      "The descent is blocked by the lid: the fingers open along x and sit on it. Astra's hypothesis (decision 5): turn the fingers to open along y to clear the lid. Confirmed: the next descent reaches handle height.",
                      "Closes both grippers and lift-tests 4 cm: the pot rises but tilts; the left handle is held, the right grasp missed (right fingers close to the empty 4.9 cm).",
                      "Decisions 9-37: the right arm cannot plan to the higher and inward targets (reach limit), eight single-arm re-grasp tries of the right handle fail while the left arm holds the tilted pot, the pot slips, shifts 13 cm to the left and rotates; the right arm can no longer reach it. Decision 38: Astra stops, 'so a fresh attempt can establish better handle alignment'."],
    ("bimanual", 2): ["Starts with the fingers along y and a symmetric approach (lessons of attempt 1), but again asks for 1.10 m first: unreachable, then recovers.",
                      "Closes both grippers and waits one motion for the closure to settle (a new habit): left fingers reach the empty width, right fingers stop at 11 cm; Astra notes the asymmetry.",
                      "Lifts 3 cm then 5 cm: the pot rises level (held on one side); higher targets fail to plan; a move toward the robot while holding flings the pot off the table (evaluation log: it lands on the floor 4 m away).",
                      "Astra opens, withdraws, sees no pot in any camera, concludes 'the pot may have been displaced outside the visible workspace' and stops."],
    ("bimanual", 3): ["Approaches at a reachable height with the fingers along y, refines inward in two steps, then corrects both wrists toward negative x after judging the pot offset.",
                      "Closes both grippers and waits two motions for the closure to settle: separations stabilise at 8.7 cm (left) and 10.9 cm (right), both above the empty width.",
                      "Lift test of 2.6 cm: the pot rises level with clearance underneath. A further coordinated lift reaches 0.82 m and the environment registers success after 9 motions (both grippers within 3 cm of their handles, pot upright)."],
    ("dexhand", 1): ["Task: right hand puts the can on the saucer inside the cabinet, then the left hand swings the open door shut (the door's sweep hits the can, so the order is forced).",
                     "D00-D03: fingers-down and palm-down targets are unreachable (153° and 142° residual); Astra hypothesises that a palm facing left avoids the wrist limits, and it tracks to 2°.",
                     "D04-D10: three closures beside the can push it 9 cm left and tip it; each closure reads as empty (2 cm thumb-index gap, no contact) and a 12 cm lift test confirms the miss.",
                     "D12-D14: a fingers-down retry presses into the cabinet at 554 N; Astra stops with 26 steps left 'so a fresh attempt can use a different grasp approach'. Left arm never moved."],
    ("dexhand", 2): ["Starts from attempt 1's lessons (cites interactions 10-14): avoids fingers-down and tries palm-down; tracking is partial (34° then 17-25°).",
                     "D12: adopts the orientation the hand actually reached ('Keeping the achieved orientation may improve pose tracking'); errors drop below 1 cm and 1.2° for the rest of the attempt.",
                     "Five closures, five lift tests, all correctly read as empty; the can never rises more than 3 cm. The 400-step horizon ends the attempt."],
    ("dexhand", 3): ["Reuses attempt 2's orientation at once (0.16 cm error on the first move).",
                     "D11: 'Earlier grasps closed behind the can; a modest forward correction ... will place its upper body within the closing region'. D12 closes with less thumb bend: 13.7 N index contact, knuckles blocked at 0.37.",
                     "D13: lift test raises the can 14.5 cm (first lift of the can through this interface; the author's ~40 scripted grasps never managed one). D14: a quick further 7 cm lift loses it.",
                     "D15-D26: recovery grasps on the fallen can all miss; the horizon ends the attempt. Left arm and door never used."],
    ("dexhand", 4): ["Continuation at the user's request (10 attempts, no guidance). Returns to the palm-left side grasp and closes low with partial fractions: 9.1 N contact, 7 cm gap; an 8 cm lift test raises the can.",
                     "D04: a lift to an unreachable height (101° residual) swings the arm and flings the can off the table.",
                     "D06-D09: first use of the left hand: from outside the open panel it pushes the door to 0.51 rad ('The panel visibly rotated substantially toward closed'). A right-hand push presses at 125 N without closing it."],
    ("dexhand", 5): ["Cites attempt 4's grasp (interactions 68-70) and repeats it: 11.8 N contact, lift test passes.",
                     "D04-D05: carries the can 30 cm up and over the cabinet's front edge; it drops on the cabinet floor 11.8 cm from the saucer centre (success needs < 9 cm) and is then knocked back onto the table.",
                     "Lesson in its memory: 'preserve the initially successful low grasp and avoid the higher transport configuration that lost the can'."],
    ("dexhand", 6): ["The first approach bumps the can; four closures fail and the can rolls off the table's front edge.",
                     "Astra stops: 'the can is lost from view after repeated failed grasps'; memory: 'verify retention before transport'."],
    ("dexhand", 7): ["SUCCESS. Side grasp (17.9 N contact, 6.6 cm gap), 8 cm lift test, carried over the lip into the cabinet while held, lowered in 4 / 2.5 / 1 cm steps, released upright on the saucer (3.4 cm from centre).",
                     "Right hand withdraws out of the door's path; the left palm pushes the door along its arc: 0.57 -> 1.10 -> 1.35 rad. Environment success at step 378 of 400."],
    ("dexhand", 8): ["Starts with attempt 7's success in its history and writes the release pose into its own memory ('Prior successful release used right wrist [0.5165,-0.121,1.247] ...').",
                     "Five closures: contact twice (12-16 N) but both lift tests lose the can. Stops with 19 steps left: 'insufficient to recover the can, place it, and close the panel'."],
    ("dexhand", 9): ["Six closures with no contact (side and top-down grasps), four empty lift tests.",
                     "Stops with 14 steps left; keeps the attempt-7 recipe in memory."],
    ("dex_unload", 1): ["Right hand grasps the right can, lifts it 12 cm and sets it down outside the tray (that can counts as unloaded).",
                        "The left can is never lifted (at most 1.2 cm); the 500-step horizon ends the attempt after 21 decisions."],
    ("dex_unload", 2): ["SUCCESS at step 432 in 20 decisions, right hand only (the left hand never moves).",
                        "D00-D01 approach the right can, D02 close, D03 lift 10 cm, D04 carry sideways, D05 lower, D06 release on the table beside the tray.",
                        "D07-D11 move to and realign at the left can (the D09 closure fails and is reopened in D10), D12 close, D13 lift 12 cm, D14-D19 carry it out over the tray's edge."],
    ("dex_laptop", 1): ["SUCCESS at step 223 in 15 decisions; the right hand lifts the lid, the left hand only repositions.",
                        "D00-D09: the right hand probes the lid's front edge (a bump at D02 raises the lid to 0.35 rad and it falls back closed).",
                        "D10: palm-up fingers under the front edge, curled; D11-D14 lift and advance: lid 0.59 -> 0.86 -> 1.24 -> 1.27 rad (success needs 1.257, a margin of 0.013 rad)."],
    ("dex_insert", 1): ["45 decisions of approaches and grasps on both cans; neither rises more than 1.2 cm. Astra stops at step 878."],
    ("dex_insert", 2): ["The right can is lifted (up to 20 cm) and dropped into the tray at D17, meeting the env's insert condition for that can.",
                        "The left can never rises more than 3 cm; Astra stops after 23 decisions (step 468)."],
    ("dex_insert", 3): ["46 decisions; no can rises more than 3.2 cm. Astra stops at step 878."],
    ("dex_insert", 4): ["The right can is lifted 23 cm and held over the tray but not lowered into a slot; the left can is knocked off the table.",
                        "Astra stops after 20 decisions (step 375)."],
    ("dex_insert", 5): ["D00-D34: repeated failed grasps; D19 pushes the right can about 10 cm.",
                        "D35 both hands close together, D36 lift both cans 10 cm, D37-D39 carry them over the tray, D40 the left hand opens and both cans drop into the tray: the insert condition holds for both at step 871.",
                        "The 900-step horizon ends the attempt 29 steps later, before any unloading. (This attempt's insert stage is the program of the Series C pilot.)"],
    ("dex_insert", 6): ["Interrupted at decision 30: from 00:24 every Astra call failed with the workspace spend cap. Not counted as an attempt."],
    ("dexhand", 10): ["SUCCESS in 11 decisions. One side grasp (12.9 N, 6.9 cm gap), lift test, transport straight to the remembered attempt-7 release pose, lower onto the saucer, release (3.2 cm from centre).",
                      "Withdraws the right hand; the left palm pushes the door 0 -> 0.65 -> 1.34 rad. Environment success at step 298."],
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
    if kind == "dexhand":
        closes, prev = 0, False                       # closure events: a hand goes from open (< 0.3) to closed (knuckles or tips >= 0.5)
        for d in decs:
            c = d["decision_content"]
            if c["stop"]:
                continue
            m = max(max(c[h]["finger_base"], c[h]["finger_tip"]) for h in ("left", "right"))
            closes += int(m >= 0.5 and not prev)
            prev = m >= 0.3
        return f"{s.get('decisions_total')} decisions · {s.get('steps_used')} control steps ({s.get('simulated_time_s')} s simulated) · {closes} hand closures · model time {s.get('model_wall_time_s')} s"
    closes = sum(1 for i, d in enumerate(decs) if d["decision_content"]["left"]["gripper"] < 0.5 and (i == 0 or decs[i - 1]["decision_content"]["left"]["gripper"] >= 0.5))
    return f"{s.get('decisions_total')} decisions (one planned motion each) · {s.get('sim_steps')} physics steps · {closes} grasp attempts · model time {s.get('model_wall_time_s')} s"


def ensure_poster(video, poster):
    """Last-second frame of a video as its poster (written into the site directory, never into a run directory)."""
    import subprocess
    poster = Path(poster)
    if poster.exists() and poster.stat().st_mtime > Path(video).stat().st_mtime:
        return True
    poster.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.6", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(poster)])
    return r.returncode == 0 and poster.exists()


def attempt_block(case, att, kind, video_w=640, media_prefix="", page_prefix="", button=True):
    """One self-contained block: when, outcome, embedded video, what happened, link to the decision cards.
    media_prefix: path from the page to the case directory (holds the `run` symlink); page_prefix: path to attempt pages."""
    s = att["summary"]
    t0, t1 = attempt_times(att)
    rel = f"{media_prefix}run/{att['dir'].name}"
    bullets = "".join(f"<li>{ESC(b)}</li>" for b in NARRATIVES.get((case, att["k"]), []))
    poster = f" poster='{media_prefix}posters/attempt_{att['k']}.jpg'" if att.get("poster") else ""
    video = f"<video controls preload='none'{poster} style='width:{video_w}px;max-width:100%' src='{rel}/{att['video']}'></video>" if att["video"] else "<p class='small'>no video</p>"
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
        when = f"step {d.get('env_step_before')}" if kind == "libero" else f"step {d.get('steps_before')}" if kind == "dexhand" else f"motion {d.get('motions_before')}"
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
        elif kind == "dexhand":
            f = lambda h, x: ("hold" if not x["move"] else f"{h} → {[round(v, 2) for v in x['position_m']]}") + f" hand {x['finger_base']:.1f}/{x['finger_tip']:.1f}/{x['thumb_rotate']:.1f}/{x['thumb_bend']:.1f}"
            act = ESC(f("L", c["left"]) + "; " + f("R", c["right"]) + f"; {c['steps']} steps")
            hl, hr = r.get("hands", {}).get("left", {}), r.get("hands", {}).get("right", {})
            res = ESC(f"L err {hl.get('position_error_cm', '–')} cm, thumb-index {hl.get('thumb_index_distance_cm')} cm; R err {hr.get('position_error_cm', '–')} cm, thumb-index {hr.get('thumb_index_distance_cm')} cm")
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
         + (f"control steps {s.get('env_steps')} ({s.get('simulated_time_s')} s simulated)" if kind == "libero" else f"control steps {s.get('steps_used')} ({s.get('simulated_time_s')} s simulated)" if kind == "dexhand" else f"motions {s.get('motions')} ({s.get('sim_steps')} physics steps)")
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
            if kind == "dexhand" and d.get("steps_clip_reason"):
                cap = f" ({d['steps_clip_reason']})"
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
    for a in run["attempts"]:
        if a["video"]:
            a["poster"] = ensure_poster(a["dir"] / a["video"], out_dir / "posters" / f"attempt_{a['k']}.jpg")
    link = out_dir / "run"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(run["dir"], out_dir))
    cfg, res, kind = run["cfg"], run["res"], run["kind"]
    t = cfg["task"]
    rows = "".join(f"<tr><td>attempt {a['k']}</td><td>{badge(a['summary'].get('outcome'))}</td><td>{a['summary'].get('decisions_total')}</td><td>{a['summary'].get('model_calls')}</td>"
                   f"<td>{a['summary'].get('env_steps') if kind == 'libero' else a['summary'].get('steps_used') if kind == 'dexhand' else a['summary'].get('motions')}</td><td>{a['summary'].get('model_wall_time_s')} s</td>"
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
            f"<table style='margin-top:10px'><tr><th>attempt</th><th>outcome</th><th>decisions</th><th>model calls</th><th>{'motions' if kind == 'robotwin' else 'control steps'}</th><th>model time</th><th>links</th></tr>{rows}</table></details>"
            f"<details class='card'><summary><b>Full narrative report</b></summary><div class='report'>{report_html}</div></details>")
    (out_dir / "index.html").write_text(page(f"{case}", body, nav, depth=1))
    for att in run["attempts"]:
        build_attempt(case, run, att, out_dir, nav)


CASE_NAMES = {"libero": "LIBERO", "robotwin": "RoboTwin single-arm", "bimanual": "RoboTwin bimanual", "dexhand": "Dex hand (Isaac)",
              "dex_unload": "Dex hand: Unload-Cans", "dex_laptop": "Dex hand: Open-Laptop", "dex_insert": "Dex hand: Insert-And-Unload-Cans"}


def dots(run):
    out = []
    for a in run["attempts"]:
        o = a["summary"].get("outcome")
        cls, t = ("S", "S") if o == "success" else ("T", "stop") if o == "model_stopped" else ("F", "F")
        out.append(f"<span class='{cls}' title='attempt {a['k']}: {ESC(str(o))}'>{a['k']}</span>")
    return "<div class='dots'>" + "".join(out) + "</div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="case=run_dir (e.g. libero=/path/run)")
    ap.add_argument("--transfer", default=None, help="overnight transfer-study root (series B)")
    ap.add_argument("--s1local", default=None, help="experiments directory holding the System 1 pilots (series C)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.run:
        case, path = spec.split("=", 1)
        runs[case] = load_run(Path(path).resolve())
    SUB = {"libero": "LIBERO-90 task 48: ketchup into the basket", "robotwin": "place_container_plate: bowl onto the plate",
           "bimanual": "lift_pot with two arms", "dexhand": "can on the saucer, then shut the door",
           "dex_unload": "unload both cans from the tray", "dex_laptop": "open the laptop lid", "dex_insert": "insert both cans, then unload them"}
    sims = [c for c in runs if not c.startswith("dex")]
    dex = [c for c in runs if c.startswith("dex")]
    astra = ([("#", "RoboTwin / LIBERO")] if sims else []) + [(CASE_NAMES.get(c, c), f"{c}/index.html", SUB.get(c, "")) for c in sims]
    astra += ([("#", "Dex hand (Isaac, H1 + Inspire hands)")] if dex else []) + [(CASE_NAMES.get(c, c).replace("Dex hand: ", "").replace("Dex hand (Isaac)", "Stack-Can-Into-Drawer"),
                                                                                 f"{c}/index.html", SUB.get(c, "")) for c in dex]
    nav = [("", [("Home", "index.html")]), (("Astra on its own", "A", "#2b5fd9"), astra)]
    if a.transfer:
        nav.append((("Transfer study", "B", "#0d9488"), [("Overview", "transfer/index.html", "can a policy learn from Astra? the whole night at a glance"), ("#", "Platforms"),
                                                          ("RoboTwin single-arm", "transfer/robotwin_single.html"), ("RoboTwin bimanual", "transfer/robotwin_bimanual.html"),
                                                          ("Dex hand", "transfer/dexhand.html"), ("LIBERO", "transfer/libero.html")]))
    if a.s1local:
        nav.append((("System 1 beneath Astra", "C", "#7c3aed"), [("Overview", "s1local/index.html", "the answer, every operation x condition, pooled contrasts"), ("#", "Tasks"),
                                                                  ("Stack-Can-Into-Drawer", "s1local/drawer.html", "first pilot: grasp, transport, door"),
                                                                  ("Unload-Cans", "s1local/unload.html", "grasp and unload the right can, grasp the left can"),
                                                                  ("Open-Laptop", "s1local/laptop.html", "near-threshold lid push"),
                                                                  ("Insert-And-Unload-Cans", "s1local/insert.html", "insert stage: grasp and insert both cans"),
                                                                  ("#", "Since the pilots"),
                                                                  ("Why grasps fail", "s1local/diagnosis.html", "robot accuracy vs Astra's aim; dex-hand failure modes"),
                                                                  ("Cup transfer experiment", "s1local/cup_transfer.html", "cup A to cup B and cylinder C, five methods"),
                                                                  ("Bottleneck batch 1", "s1local/batch1.html", "where time goes, the first failure, why System 1 does not fix it"),
                                                                  ("System 1 batch 2", "s1local/batch2.html", "persistent corrections, objective, representation, memory updates"),
                                                                  ("System 1 design", "s1local/design.html", "boundary, interface, short memory, optimization, literature")]))
    cards = []
    for case, run in runs.items():
        build_case(case, run, out / case, nav)
        t = run["cfg"]["task"]
        att = run["attempts"]
        k = sum(1 for x in att if x["summary"].get("outcome") == "success")
        first = next((x for x in att if x["summary"].get("outcome") == "success" and x["video"]), None)
        vid = (f"<video controls preload='none' poster='{case}/posters/attempt_{first['k']}.jpg' style='width:100%' src='{case}/run/{first['dir'].name}/{first['video']}'></video>"
               f"<div class='small'>first successful attempt ({first['k']})</div>") if first else "<p class='small'>no successful attempt</p>"
        cards.append(f"<div class='case'><h3 style='margin:0'><a href='{case}/index.html'>{ESC(CASE_NAMES.get(case, case))}</a></h3>"
                     f"<div class='small'>{ESC(t.get('instruction', ''))}</div><div style='margin:8px 0'><b>{k}/{len(att)}</b> attempts succeeded · "
                     f"{run['res'].get('model_calls_total')} model calls</div>{dots(run)}<div style='margin-top:8px'>{vid}</div>"
                     f"<p><a href='{case}/index.html'>all attempts, videos and decision cards →</a></p></div>")
    series_b = ""
    if a.transfer:
        import build_transfer
        S = build_transfer.build(out, a.transfer, lambda title, body, depth: page(title, body, nav, depth))
        plat = "".join(f"<a href='transfer/{p}.html' class='badge neutral' style='background:var(--acc);margin-right:6px'>{ESC(n)}</a>" for p, n, _ in build_transfer.PLATFORMS)
        series_b = ("<div class='card series' style='border-left-color:#0d9488'><h2 style='margin-top:0'>Series B · Overnight transfer study: can a policy learn from Astra?</h2>"
                    "<p>Astra as the teacher (System 2), a learned robot policy as the student (System 1): twelve registered configurations on four simulators, "
                    "run overnight on 2026-10-10 under a fixed budget. Does Astra's own interaction data make the student better once Astra is switched off?</p>"
                    "<table><tr><th>platform</th><th>Astra on its own</th><th>original student</th><th>student trained on Astra data</th></tr>"
                    "<tr><td>RoboTwin single-arm</td><td>2/4 source · 1/1 reference</td><td>2/5 dev · 3/10 test</td><td>0/5 dev · 1/10 test</td></tr>"
                    "<tr><td>RoboTwin bimanual</td><td>0/4</td><td>3/3 dev · 2/10 test</td><td>no Astra success to learn from</td></tr>"
                    "<tr><td>Dex hand</td><td>0/6</td><td>none exists</td><td>no Astra success to learn from</td></tr>"
                    "<tr><td>LIBERO</td><td>1/3 reference</td><td>no compatible student</td><td>blocked by the interface</td></tr></table>"
                    "<p><b>Answer tonight:</b> no channel from Astra to the student produced a better student; takeover and correction cells were gated off.</p>"
                    f"<p><a href='transfer/index.html' class='badge neutral' style='background:#0d9488;font-size:14px;padding:5px 12px'>Open the study overview →</a></p>"
                    f"<p class='small'>Per platform: {plat}</p></div>")
    series_c = ""
    if a.s1local:
        import build_s1local
        T = build_s1local.build(out, a.s1local, lambda title, body, depth: page(title, body, nav, depth))
        series_c = ("<div class='card series' style='border-left-color:#7c3aed'><h2 style='margin-top:0'>Series C · What System 1 is useful beneath Astra? (dex hand, 2026-10-10/11)</h2>"
                    "<p>Astra writes an open-loop program; a fast System 1 underneath adjusts up to four parameters of one operation by local search, with or without "
                    "memory of its own tries. First pilot on Stack-Can-Into-Drawer, then the same protocol on Unload-Cans, Open-Laptop and Insert-And-Unload-Cans "
                    "(Sort-Cans blocked by the Astra spend cap).</p>"
                    f"<p><b>Answer:</b> across {sum(T['cls'].values())} matched blocks local search rescued {T['cls'].get('rescue', 0)}; fresh-start success "
                    f"A {T['ev']['A'][0]}/{T['ev']['A'][1]}, B {T['ev']['B'][0]}/{T['ev']['B'][1]}, C {T['ev']['C'][0]}/{T['ev']['C'][1]}. Repeating Astra's program "
                    "is as good as searching around it; the failures need closed-loop feedback, not parameter search.</p>"
                    "<p><a href='s1local/index.html' class='badge neutral' style='background:#7c3aed;font-size:14px;padding:5px 12px'>Open Series C →</a></p></div>")
    body = ("<h1>Astra studies</h1><p>Three series of experiments with GPT-6 Astra (via the Codex CLI) controlling simulated robots. "
            "Series A lets Astra act on its own and learn only from its recorded memory. Series B asks whether a learned policy can learn from what Astra did. "
            "Series C asks what kind of fast System 1 is useful beneath Astra.</p>"
            "<div class='card series'><h2 style='margin-top:0'>Series A · Astra on its own (pilots, 2026-10-09/11)</h2>"
            "<p>Each pilot: one fixed scene, attempts in order, Astra decides one action at a time and carries its bounded memory into the next attempt; "
            "no policy, no weight update. Squares: green = success, amber = Astra stopped, red = horizon or budget reached.</p>"
            f"<div class='cases'>{''.join(cards)}</div></div>{series_b}{series_c}")
    (out / "index.html").write_text(page("Astra studies", body, nav))
    print("site built at", out, "cases:", list(runs), "transfer:", bool(a.transfer), "s1local:", bool(a.s1local))


if __name__ == "__main__":
    main()
