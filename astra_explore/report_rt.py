"""Trace-based report tables and contact sheets for a RoboTwin Astra run (planner-assisted ee interface).

    python report_rt.py --run ../experiments/astra_explore/<run>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def rjson(p):
    return json.loads(Path(p).read_text())


def short(s, n=160):
    s = "" if s is None else str(s).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[:n - 1] + "…"


def arm_txt(a):
    if not a["move"]:
        return f"hold, grip {a['gripper']:.2f}"
    return f"to {[round(x, 3) for x in a['position_m']]} q{[round(x, 2) for x in a['quaternion_wxyz']]} grip {a['gripper']:.2f}"


def load_run(run):
    run = Path(run)
    cfg, res = rjson(run / "config.json"), rjson(run / "result.json")
    attempts = []
    for adir in sorted(run.glob("attempt_*")):
        decs = []
        for ddir in sorted(adir.glob("decision_*")):
            if (ddir / "decision.json").exists():
                d = rjson(ddir / "decision.json")
                d["dir"] = ddir
                d["observation"] = rjson(ddir / "observation.json") if (ddir / "observation.json").exists() else {}
                decs.append(d)
        summ = rjson(adir / "summary.json") if (adir / "summary.json").exists() else {}
        ev = [json.loads(l) for l in (adir / "evaluation_only.jsonl").read_text().splitlines()] if (adir / "evaluation_only.jsonl").exists() else []
        attempts.append({"dir": adir, "k": int(adir.name.split("_")[1]), "decisions": decs, "summary": summ, "evaluation": ev})
    return run, cfg, res, attempts


def memory_change(prev, new):
    if new is None:
        return "no accepted response: memory unchanged"
    prev = prev or {"observations": [], "hypotheses": [], "summary": ""}
    parts = []
    ao = [o for o in new["observations"] if o not in prev["observations"]]
    ah = [h for h in new["hypotheses"] if h not in prev["hypotheses"]]
    if ao:
        parts.append("+obs: " + " / ".join(short(x, 90) for x in ao))
    if len([o for o in prev["observations"] if o not in new["observations"]]):
        parts.append(f"-{len([o for o in prev['observations'] if o not in new['observations']])} obs")
    if ah:
        parts.append("+hyp: " + " / ".join(short(x, 90) for x in ah))
    if len([h for h in prev["hypotheses"] if h not in new["hypotheses"]]):
        parts.append(f"-{len([h for h in prev['hypotheses'] if h not in new['hypotheses']])} hyp")
    if new["summary"] != prev["summary"]:
        parts.append("summary rewritten")
    return "; ".join(parts) if parts else "unchanged"


def contact_sheet(att, thumb_w=256, per_row=4):
    decs = [d for d in att["decisions"] if (d["dir"] / "pre_head_camera.png").exists()]
    if not decs:
        return None
    try:
        font = ImageFont.truetype(FONT, 11)
    except OSError:
        font = ImageFont.load_default()
    th = int(thumb_w * 480 / 640)
    rows = (len(decs) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * thumb_w, rows * (th + 46)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, d in enumerate(decs):
        im = Image.open(d["dir"] / "pre_head_camera.png").resize((thumb_w, th))
        x, y = (i % per_row) * thumb_w, (i // per_row) * (th + 46)
        sheet.paste(im, (x, y))
        c = d.get("decision_content")
        draw.text((x + 3, y + th + 2), f"D{d['decision']:02d} motion {d['motions_before']}", fill=(255, 255, 255), font=font)
        if c:
            l = "STOP" if c["stop"] else ("L " + ("hold" if not c["left"]["move"] else ",".join(f"{v:.2f}" for v in c["left"]["position_m"])) + f" g{c['left']['gripper']:.1f}")
            r = "" if c["stop"] else ("R " + ("hold" if not c["right"]["move"] else ",".join(f"{v:.2f}" for v in c["right"]["position_m"])) + f" g{c['right']['gripper']:.1f}")
            draw.text((x + 3, y + th + 16), l[:40], fill=(200, 200, 255), font=font)
            draw.text((x + 3, y + th + 30), r[:40], fill=(200, 255, 200), font=font)
        else:
            draw.text((x + 3, y + th + 16), "no valid response", fill=(255, 150, 150), font=font)
    out = att["dir"] / "contact_sheet.png"
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run, cfg, res, attempts = load_run(a.run)
    t = cfg["task"]
    L = ["# Astra RoboTwin exploration run: trace tables", "", f"Run directory: `{run}`", "", "## Experiment summary", "",
         f"- Task: RoboTwin `{t['task']}`, scene seed {t['scene_seed']} (expert-solvable: {t['expert_solvable_seed']}); instruction \"{t['instruction']}\"; scene objects {t['scene_info']}.",
         f"- Robot/interface: {t['robot']}; {t['action_type']}; pose format {t['pose_format']}; gripper {t['gripper']}; RoboTwin motion limit {t['step_limit_motions']} (never binding).",
         f"- Observation access: {', '.join(cfg['observation_access'])}.", f"- Memory format: {cfg['memory_format']}.", f"- Budgets: {json.dumps(cfg['budgets'])}.",
         f"- Model: requested `{cfg['model']['requested']}`, reasoning {cfg['model']['reasoning_effort']}; CLI `{cfg['model'].get('codex_version')}`; login: {cfg['model'].get('login_status')}.",
         f"- Run status: {res['status']}" + (f" (error: {short(res['error'], 300)})" if res.get("error") else "") + f"; model calls used {res.get('model_calls_total')}; any success: {res.get('any_success', 'run in progress')}; wall time {res.get('wall_time_s')} s.",
         f"- Policy: {cfg['policy']} (is_astra_run={cfg['is_astra_run']}); started {cfg['started']}; code {cfg.get('git')}.", ""]
    for att in attempts:
        s = att["summary"]
        L += [f"## Attempt {att['k']}: {s.get('outcome', '?')}", "",
              f"{s.get('outcome_text', '')}. Decisions {s.get('decisions_total')} (executed {s.get('decisions_executed')}), model calls {s.get('model_calls')}, motions {s.get('motions')}, physics steps {s.get('sim_steps')}, model wall time {s.get('model_wall_time_s')} s, attempt wall time {s.get('attempt_wall_time_s')} s.", ""]
        sheet = contact_sheet(att)
        if sheet:
            L += [f"Contact sheet (pre-decision head-camera frames): `{sheet.relative_to(run)}`; video: `{att['dir'].name}/head_camera.mp4`.", ""]
        L += ["| Decision (id) | What Astra observed | Astra's assessment of its previous motion | Stated intent / hypothesis | Commanded motion | Measured result | What changed in memory |",
              "|---|---|---|---|---|---|---|"]
        prev_mem = None
        if att["k"] > 1:
            for d in attempts[att["k"] - 2]["decisions"]:
                if d.get("memory_after"):
                    prev_mem = d["memory_after"]
        for d in att["decisions"]:
            o = d.get("observation", {}).get("proprio", {})
            obs_txt = f"motion {d['motions_before']}: L ee {o.get('left', {}).get('ee_position_m_world')} sep {o.get('left', {}).get('finger_link_separation_cm')} cm; R ee {o.get('right', {}).get('ee_position_m_world')}; imgs `{d['dir'].name}/pre_*.png`"
            c = d.get("decision_content")
            if c is None:
                L.append(f"| D{d['decision']:02d} (no id) | {short(obs_txt, 200)} | – | – | NONE: {'; '.join(short(x.get('reason') or x.get('skipped'), 80) for x in d['calls'])} | no execution | unchanged |")
                continue
            act = "STOP requested" if c["stop"] else f"L: {arm_txt(c['left'])}; R: {arm_txt(c['right'])}"
            r = d.get("result", {})
            if c["stop"]:
                res_txt = "attempt ended by stop"
            else:
                parts = []
                for arm in ("left", "right"):
                    ar = r.get("arms", {}).get(arm, {})
                    parts.append(f"{arm[0].upper()}: planner {r.get('planner', {}).get(arm, {}).get('status')}, moved {ar.get('moved_cm')} cm, err {ar.get('position_error_cm')} cm/{ar.get('orientation_error_deg')}°, sep {ar.get('finger_link_separation_cm')} cm")
                res_txt = "; ".join(parts) + f"; {r.get('sim_steps')} steps; complete={r.get('task_complete')}" + (" **SUCCESS**" if r.get("task_complete") else "")
            retry = f" ({len(d['calls'])} calls)" if len(d["calls"]) > 1 else ""
            L.append(f"| D{d['decision']:02d} (id {d.get('interaction_id')}){retry} | {short(obs_txt, 200)} | {short(c['assessment_of_previous_action'])} | **intent:** {short(c['intent'])} **hyp:** {short(c['hypothesis'] or '–')} **expects:** {short(c['expected_change'])} refs {c['references']} | {short(act, 220)} | {short(res_txt, 260)} | {short(memory_change(prev_mem, d.get('memory_after') or c['memory_update']), 220)} |")
            prev_mem = d.get("memory_after") or c["memory_update"]
        L += ["", "### Per-decision detail (Astra's stated rationale, verbatim)", ""]
        for d in att["decisions"]:
            c = d.get("decision_content")
            L.append(f"**D{d['decision']:02d}** (`{d['dir'].relative_to(run)}`): calls " + "; ".join(f"try {x.get('try')}: {'accepted' if x.get('accepted') else (x.get('reason') or x.get('skipped'))}, {x.get('meta', {}).get('latency_s', '?')} s" for x in d["calls"]))
            if c:
                L += [f"- assessment_of_previous_action: {c['assessment_of_previous_action']}", f"- intent: {c['intent']}", f"- hypothesis: {c['hypothesis']}",
                      f"- expected_change: {c['expected_change']}", f"- references: {c['references']}", f"- left: {c['left']}", f"- right: {c['right']}",
                      f"- memory_update.observations: {c['memory_update']['observations']}", f"- memory_update.hypotheses: {c['memory_update']['hypotheses']}",
                      f"- memory_update.summary: {c['memory_update']['summary']}"]
                if d.get("result"):
                    L.append(f"- measured: {json.dumps(d['result'])}")
            L.append("")
        ev = [e for e in att["evaluation"] if "container_pos" in e or "object_pos" in e or "can_pos" in e]
        if ev:
            key = next(k for k in ("container_pos", "can_pos", "object_pos") if k in ev[0])
            tgt = next((k for k in ("plate_pos", "basket_pos", "target_pos") if k in ev[0]), None)
            P = np.array([e[key] for e in ev])
            info = {"object": key, "initial": P[0].round(3).tolist(), "final": P[-1].round(3).tolist(), "max_height_gain_m": round(float(P[:, 2].max() - P[0, 2]), 3),
                    "first_moved_at_decision": next((e.get("decision") for e, p in zip(ev, P) if np.linalg.norm(p - P[0]) > 0.005), None),
                    "success_decisions": [e.get("decision") for e in ev if e.get("success")][:3]}
            if tgt:
                T = np.array(ev[-1][tgt])
                info["target"] = tgt
                info["final_xy_distance_to_target_m"] = round(float(np.linalg.norm(P[-1, :2] - T[:2])), 3)
            L += ["### Recorded environment evidence (evaluation-only; never shown to Astra)", "", "```", json.dumps(info, indent=1), "```", ""]
    L += ["## Cross-attempt metrics (automated)", "", "| attempt | outcome | decisions | accepted | failed calls | motions | planner failures | left grasps (grip<0.5) | references | identical motions |", "|---|---|---|---|---|---|---|---|---|---|"]
    for att in attempts:
        decs = [d for d in att["decisions"] if d.get("decision_content")]
        pf = sum(1 for d in decs for a in ("left", "right") if d.get("result", {}).get("planner", {}).get(a, {}).get("status") == "Fail")
        grasps = sum(1 for d in decs if d["decision_content"]["left"]["gripper"] < 0.5)
        refs = sum(len(d["decision_content"]["references"]) for d in decs)
        sig = Counter(json.dumps([d["decision_content"]["left"], d["decision_content"]["right"]], sort_keys=True) for d in decs)
        L.append(f"| {att['k']} | {att['summary'].get('outcome')} | {len(att['decisions'])} | {len(decs)} | {sum(1 for d in att['decisions'] for c in d['calls'] if not c.get('accepted'))} | {att['summary'].get('motions')} | {pf} | {grasps} | {refs} | {sum(v - 1 for v in sig.values() if v > 1)} |")
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
