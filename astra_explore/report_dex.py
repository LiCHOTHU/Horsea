"""Trace-based report tables and contact sheets for an Isaac Sim dex-hand Astra run (IK-assisted wrist + hand interface).

    python report_dex.py --run ../experiments/astra_explore/<run>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from report_rt import FONT, load_run, memory_change, rjson, short  # noqa: F401  (same run layout)


def hand_txt(h):
    c = f"hand base {h['finger_base']:.2f} tip {h['finger_tip']:.2f} thumb rot {h['thumb_rotate']:.2f} bend {h['thumb_bend']:.2f}"
    if not h["move"]:
        return "hold, " + c
    return f"to {[round(x, 3) for x in h['position_m']]} fingers->{[round(x, 2) for x in h['fingers_dir']]} palm->{[round(x, 2) for x in h['palm_dir']]}, " + c


def contact_sheet(att, thumb_w=256, per_row=4):
    decs = [d for d in att["decisions"] if (d["dir"] / "pre_main_camera.png").exists()]
    if not decs:
        return None
    try:
        font = ImageFont.truetype(FONT, 11)
    except OSError:
        font = ImageFont.load_default()
    th = int(thumb_w * 540 / 960)
    rows = (len(decs) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * thumb_w, rows * (th + 46)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, d in enumerate(decs):
        im = Image.open(d["dir"] / "pre_main_camera.png").resize((thumb_w, th))
        x, y = (i % per_row) * thumb_w, (i // per_row) * (th + 46)
        sheet.paste(im, (x, y))
        c = d.get("decision_content")
        draw.text((x + 3, y + th + 2), f"D{d['decision']:02d} t={d.get('steps_before')}", fill=(255, 255, 255), font=font)
        if c:
            f = lambda h: ("hold" if not h["move"] else ",".join(f"{v:.2f}" for v in h["position_m"])) + f" {h['finger_base']:.1f}/{h['finger_tip']:.1f}/{h['thumb_rotate']:.1f}/{h['thumb_bend']:.1f}"
            l = "STOP" if c["stop"] else "L " + f(c["left"])
            r = "" if c["stop"] else "R " + f(c["right"]) + f" {c['steps']}st"
            draw.text((x + 3, y + th + 16), l[:40], fill=(200, 200, 255), font=font)
            draw.text((x + 3, y + th + 30), r[:40], fill=(200, 255, 200), font=font)
        else:
            draw.text((x + 3, y + th + 16), "no valid response", fill=(255, 150, 150), font=font)
    out = att["dir"] / "contact_sheet.png"
    sheet.save(out)
    return out


def result_txt(r):
    parts = []
    for s in ("left", "right"):
        h = r.get("hands", {}).get(s, {})
        if h.get("tracked"):
            parts.append(f"{s[0].upper()}: moved {h.get('moved_cm')} cm, err {h.get('position_error_cm')} cm/{h.get('orientation_error_deg')}°, "
                         f"thumb-index {h.get('thumb_index_distance_cm')} cm, contact {h.get('index_contact_force_N')} N")
        else:
            parts.append(f"{s[0].upper()}: held (moved {h.get('moved_cm')} cm), thumb-index {h.get('thumb_index_distance_cm')} cm, contact {h.get('index_contact_force_N')} N")
    return "; ".join(parts) + f"; {r.get('steps_executed')} steps" + (f" ({r.get('clipped')})" if r.get("clipped") else "") + \
        f"; complete={r.get('task_complete')}" + (" **SUCCESS**" if r.get("task_complete") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run, cfg, res, attempts = load_run(a.run)
    t = cfg["task"]
    L = ["# Astra Isaac Sim dex-hand exploration run: trace tables", "", f"Run directory: `{run}`", "", "## Experiment summary", "",
         f"- Task: `{t['task']}` ({t['simulator']}), room {t.get('room_idx')} table {t.get('table_idx')}, scene index {t.get('scene_index')} (the project's first test scene); instruction \"{t['instruction']}\".",
         f"- Robot/interface: {t['robot']}; {t['action_type']}; control {t.get('control_hz')} Hz; attempt horizon {t.get('attempt_horizon_steps')} steps (env episode {t.get('env_episode_steps')} steps, {t.get('settle_steps')} settle steps).",
         f"- Self-calibration (robot geometry only): {json.dumps({s: {k: v for k, v in t.get('calibration', {}).get(s, {}).items() if k not in ('tips_in_base_frame_open', 'hand_joint_names')} for s in ('left', 'right')})}.",
         f"- Observation access: {', '.join(cfg['observation_access'])}.", f"- Memory format: {cfg['memory_format']}.", f"- Budgets: {json.dumps(cfg['budgets'])}.",
         f"- Model: requested `{cfg['model']['requested']}`, reasoning {cfg['model']['reasoning_effort']}; CLI `{cfg['model'].get('codex_version')}`; login: {cfg['model'].get('login_status')}.",
         f"- Run status: {res['status']}" + (f" (error: {short(res['error'], 300)})" if res.get("error") else "") + f"; model calls used {res.get('model_calls_total')}; any success: {res.get('any_success', 'run in progress')}; wall time {res.get('wall_time_s')} s.",
         f"- Policy: {cfg['policy']} (is_astra_run={cfg['is_astra_run']}); started {cfg['started']}; code {cfg.get('git')}.", ""]
    for att in attempts:
        s = att["summary"]
        L += [f"## Attempt {att['k']}: {s.get('outcome', '?')}", "",
              f"{s.get('outcome_text', '')}. Decisions {s.get('decisions_total')} (executed {s.get('decisions_executed')}), model calls {s.get('model_calls')}, control steps {s.get('steps_used')} ({s.get('simulated_time_s')} s simulated), model wall time {s.get('model_wall_time_s')} s, attempt wall time {s.get('attempt_wall_time_s')} s.", ""]
        sheet = contact_sheet(att)
        if sheet:
            L += [f"Contact sheet (pre-decision main-camera frames): `{sheet.relative_to(run)}`; video: `{att['dir'].name}/main_camera.mp4`.", ""]
        L += ["| Decision (id) | What Astra observed | Astra's assessment of its previous decision | Stated intent / hypothesis | Commanded action | Measured result | What changed in memory |",
              "|---|---|---|---|---|---|---|"]
        prev_mem = None
        if att["k"] > 1:
            for d in attempts[att["k"] - 2]["decisions"]:
                if d.get("memory_after"):
                    prev_mem = d["memory_after"]
        for d in att["decisions"]:
            o = d.get("observation", {}).get("proprio", {})
            obs_txt = (f"t={d.get('steps_before')}: L wrist {o.get('left', {}).get('wrist_position_m_world')} thumb-index {o.get('left', {}).get('thumb_index_distance_cm')} cm; "
                       f"R wrist {o.get('right', {}).get('wrist_position_m_world')} thumb-index {o.get('right', {}).get('thumb_index_distance_cm')} cm; imgs `{d['dir'].name}/pre_*.png`")
            c = d.get("decision_content")
            if c is None:
                L.append(f"| D{d['decision']:02d} (no id) | {short(obs_txt, 200)} | – | – | NONE: {'; '.join(short(x.get('reason') or x.get('skipped'), 80) for x in d['calls'])} | no execution | unchanged |")
                continue
            act = "STOP requested" if c["stop"] else f"L: {hand_txt(c['left'])}; R: {hand_txt(c['right'])}; {c['steps']} steps"
            r = d.get("result", {})
            res_txt = "attempt ended by stop" if c["stop"] else result_txt(r)
            retry = f" ({len(d['calls'])} calls)" if len(d["calls"]) > 1 else ""
            L.append(f"| D{d['decision']:02d} (id {d.get('interaction_id')}){retry} | {short(obs_txt, 200)} | {short(c['assessment_of_previous_action'])} | **intent:** {short(c['intent'])} **hyp:** {short(c['hypothesis'] or '–')} **expects:** {short(c['expected_change'])} refs {c['references']} | {short(act, 260)} | {short(res_txt, 300)} | {short(memory_change(prev_mem, d.get('memory_after') or c['memory_update']), 220)} |")
            prev_mem = d.get("memory_after") or c["memory_update"]
        L += ["", "### Per-decision detail (Astra's stated rationale, verbatim)", ""]
        for d in att["decisions"]:
            c = d.get("decision_content")
            L.append(f"**D{d['decision']:02d}** (`{d['dir'].relative_to(run)}`): calls " + "; ".join(f"try {x.get('try')}: {'accepted' if x.get('accepted') else (x.get('reason') or x.get('skipped'))}, {x.get('meta', {}).get('latency_s', '?')} s" for x in d["calls"]))
            if c:
                L += [f"- assessment_of_previous_action: {c['assessment_of_previous_action']}", f"- intent: {c['intent']}", f"- hypothesis: {c['hypothesis']}",
                      f"- expected_change: {c['expected_change']}", f"- references: {c['references']}", f"- left: {c['left']}", f"- right: {c['right']}", f"- steps: {c['steps']}",
                      f"- memory_update.observations: {c['memory_update']['observations']}", f"- memory_update.hypotheses: {c['memory_update']['hypotheses']}",
                      f"- memory_update.summary: {c['memory_update']['summary']}"]
                if d.get("result"):
                    L.append(f"- measured: {json.dumps(d['result'])}")
            L.append("")
        ev = [e for e in att["evaluation"] if "can_pos" in e]
        if ev:
            P = np.array([e["can_pos"] for e in ev])
            info = {"can_initial": P[0].round(3).tolist(), "can_final": P[-1].round(3).tolist(), "can_max_height_gain_m": round(float(P[:, 2].max() - P[0, 2]), 3),
                    "can_first_moved_at_decision": next((e.get("decision") for e, p in zip(ev, P) if np.linalg.norm(p - P[0]) > 0.005), None),
                    "plate_initial": ev[0].get("plate_pos"), "plate_final": ev[-1].get("plate_pos"),
                    "drawer_joint_initial": ev[0].get("drawer_joint_pos"), "drawer_joint_final": ev[-1].get("drawer_joint_pos"), "drawer_joint_limits": ev[0].get("drawer_joint_limits"),
                    "drawer_handle_initial": ev[0].get("drawer_handle_pos"),
                    "subtask_flags_final": {k: ev[-1].get(k) for k in ("reach_drawer_success", "reach_can_success", "lift_success", "move_door_success", "is_drawer_close")},
                    "success_decisions": [e.get("decision") for e in ev if e.get("success")][:3]}
            if ev[-1].get("plate_pos"):
                info["final_can_plate_xy_distance_m"] = round(float(np.linalg.norm(P[-1, :2] - np.array(ev[-1]["plate_pos"][:2]))), 3)
                info["final_can_plate_dz_m"] = round(float(P[-1, 2] - ev[-1]["plate_pos"][2]), 3)
            L += ["### Recorded environment evidence (evaluation-only; never shown to Astra)", "", "```", json.dumps(info, indent=1), "```", ""]
    L += ["## Cross-attempt metrics (automated)", "",
          "| attempt | outcome | decisions | accepted | failed calls | control steps | wrist targets with residual > 3 cm | closing decisions (finger base or tip >= 0.5) | both hands moved | references | identical decisions |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for att in attempts:
        decs = [d for d in att["decisions"] if d.get("decision_content") and not d["decision_content"]["stop"]]
        resid = sum(1 for d in decs for s in ("left", "right") if (d.get("result", {}).get("hands", {}).get(s, {}).get("position_error_cm") or 0) > 3)
        closes = sum(1 for d in decs if any(max(d["decision_content"][s]["finger_base"], d["decision_content"][s]["finger_tip"]) >= 0.5 for s in ("left", "right")))
        both = sum(1 for d in decs if d["decision_content"]["left"]["move"] and d["decision_content"]["right"]["move"])
        refs = sum(len(d["decision_content"]["references"]) for d in decs)
        sig = Counter(json.dumps([d["decision_content"]["left"], d["decision_content"]["right"], d["decision_content"]["steps"]], sort_keys=True) for d in decs)
        L.append(f"| {att['k']} | {att['summary'].get('outcome')} | {len(att['decisions'])} | {len(decs)} | {sum(1 for d in att['decisions'] for c in d['calls'] if not c.get('accepted'))} | {att['summary'].get('steps_used')} | {resid} | {closes} | {both} | {refs} | {sum(v - 1 for v in sig.values() if v > 1)} |")
    lat = [c.get("meta", {}).get("latency_s") for att in attempts for d in att["decisions"] for c in d["calls"] if c.get("meta", {}).get("latency_s") is not None]
    usage = Counter()
    for att in attempts:
        for d in att["decisions"]:
            for c in d["calls"]:
                for k, v in (c.get("meta", {}).get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] += v
    L += ["", "## Latency and usage", "", (f"- Model calls: {len(lat)}; latency median {np.median(lat):.1f} s, mean {np.mean(lat):.1f} s, max {np.max(lat):.1f} s." if lat else "- no calls"),
          f"- Token usage (CLI-reported, summed): {dict(usage) if usage else 'not exposed'}.", f"- Wall time {res.get('wall_time_s')} s.", ""]
    (run / "REPORT_TABLES.md").write_text("\n".join(L))
    print(f"wrote {run / 'REPORT_TABLES.md'}")


if __name__ == "__main__":
    main()
