"""Trace-based report tables and media for an Astra exploration run.

    python report.py --run ../experiments/astra_explore/<run>

Writes REPORT_TABLES.md (experiment summary, per-decision chain tables, cross-attempt metrics, latency/usage,
evaluation-only object evidence), contact sheets (attempt_k/contact_sheet.png) and decisions.jsonl (flat trace).
Three kinds of statement are kept apart: recorded environment evidence (measured by the harness), Astra's stated
interpretation (its response fields), and the author's inference (written by hand in REPORT.md, not here).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from schema import ACTION_KEYS

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def rjson(p):
    return json.loads(Path(p).read_text())


def fmt_action(a):
    return "[" + " ".join(f"{a[k]:+.2f}" for k in ACTION_KEYS) + "]"


def short(s, n=160):
    s = "" if s is None else str(s).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[:n - 1] + "…"


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
            d["observation"] = rjson(ddir / "observation.json") if (ddir / "observation.json").exists() else {}
            for c in d["calls"]:
                cdir = ddir / f"call_{c.get('try', 0)}"
                c["reasoning_summaries"] = c.get("meta", {}).get("reasoning_summaries", [])
                c["dir"] = str(cdir.relative_to(run)) if cdir.exists() else None
            decs.append(d)
        summ = rjson(adir / "summary.json") if (adir / "summary.json").exists() else {}
        ev = [json.loads(l) for l in (adir / "evaluation_only.jsonl").read_text().splitlines()] if (adir / "evaluation_only.jsonl").exists() else []
        attempts.append({"dir": adir, "k": int(adir.name.split("_")[1]), "decisions": decs, "summary": summ, "evaluation": ev})
    return run, cfg, res, attempts


def memory_change(prev, new):
    if new is None:
        return "no accepted response: memory unchanged"
    if prev is None:
        prev = {"observations": [], "hypotheses": [], "summary": ""}
    added_o = [o for o in new["observations"] if o not in prev["observations"]]
    removed_o = [o for o in prev["observations"] if o not in new["observations"]]
    added_h = [h for h in new["hypotheses"] if h not in prev["hypotheses"]]
    removed_h = [h for h in prev["hypotheses"] if h not in new["hypotheses"]]
    parts = []
    if added_o:
        parts.append("+obs: " + " / ".join(short(x, 90) for x in added_o))
    if removed_o:
        parts.append(f"-{len(removed_o)} obs")
    if added_h:
        parts.append("+hyp: " + " / ".join(short(x, 90) for x in added_h))
    if removed_h:
        parts.append(f"-{len(removed_h)} hyp")
    if new["summary"] != prev["summary"]:
        parts.append("summary rewritten")
    return "; ".join(parts) if parts else "unchanged"


def contact_sheet(run, att, thumb=220, per_row=5):
    decs = [d for d in att["decisions"] if (d["dir"] / "pre_agentview.png").exists()]
    if not decs:
        return None
    try:
        font = ImageFont.truetype(FONT, 11)
    except OSError:
        font = ImageFont.load_default()
    rows = (len(decs) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * thumb, rows * (thumb + 34)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, d in enumerate(decs):
        im = Image.open(d["dir"] / "pre_agentview.png").resize((thumb, thumb))
        x, y = (i % per_row) * thumb, (i // per_row) * (thumb + 34)
        sheet.paste(im, (x, y))
        c = d.get("decision_content")
        cap = f"D{d['decision']:02d} step {d['env_step_before']}"
        cap2 = ("STOP" if c and c["stop"] else (",".join(f"{c['action'][k]:+.1f}".replace("+0.0", "0").replace("-0.0", "0") for k in ACTION_KEYS)
                                                 + f" x{d.get('repeat_executed', 0)}") if c else "no valid response")
        draw.text((x + 3, y + thumb + 2), cap, fill=(255, 255, 255), font=font)
        draw.text((x + 3, y + thumb + 17), cap2[:32], fill=(200, 200, 255), font=font)
    out = att["dir"] / "contact_sheet.png"
    sheet.save(out)
    return out


def attempt_metrics(att):
    decs = [d for d in att["decisions"] if d.get("decision_content")]
    acts = [tuple(round(d["decision_content"]["action"][k], 2) for k in ACTION_KEYS) + (d["decision_content"]["repeat"],) for d in decs]
    closes = sum(1 for d in decs if d["decision_content"]["action"]["gripper"] > 0)
    opens = sum(1 for d in decs if d["decision_content"]["action"]["gripper"] < 0)
    refs = [r for d in decs for r in d["decision_content"]["references"]]
    hyps = sum(1 for d in decs if d["decision_content"]["hypothesis"])
    z = [s["eef"][2] for d in decs for s in d.get("steps", [])]
    widths = [s["gripper_width_cm"] for d in decs for s in d.get("steps", [])]
    rep = Counter(acts)
    return {"decisions": len(att["decisions"]), "accepted": len(decs), "invalid_or_failed_calls": sum(1 for d in att["decisions"] for c in d["calls"] if not c.get("accepted")),
            "close_commands": closes, "open_commands": opens, "decisions_with_hypothesis": hyps,
            "references_total": len(refs), "distinct_referenced_ids": sorted(set(refs)),
            "identical_action_repeats": sum(v - 1 for v in rep.values() if v > 1),
            "eef_z_min_max_m": [round(min(z), 3), round(max(z), 3)] if z else None,
            "gripper_width_min_cm": round(min(widths), 2) if widths else None}


def evaluation_evidence(att, obj="ketchup_1", target="basket_1"):
    ev = [e for e in att["evaluation"] if f"{obj}_pos" in e]
    if not ev:
        return {}
    op = np.array([e[f"{obj}_pos"] for e in ev])
    bp = np.array([e[f"{target}_pos"] for e in ev]) if f"{target}_pos" in ev[0] else None
    out = {"object_initial_pos": op[0].round(3).tolist(), "object_final_pos": op[-1].round(3).tolist(),
           "object_max_height_gain_m": round(float(op[:, 2].max() - op[0, 2]), 3),
           "object_total_xy_displacement_m": round(float(np.linalg.norm(op[-1, :2] - op[0, :2])), 3),
           "object_first_moved_at_env_step": next((e["env_step"] for e, p in zip(ev, op) if np.linalg.norm(p - op[0]) > 0.005), None),
           "success_steps": [e["env_step"] for e in ev if e.get("success")][:3]}
    if bp is not None:
        out["basket_pos"] = bp[0].round(3).tolist()
        out["object_final_xy_distance_to_basket_m"] = round(float(np.linalg.norm(op[-1, :2] - bp[-1, :2])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run, cfg, res, attempts = load_run(a.run)
    L = []
    t = cfg["task"]
    L += ["# Astra exploration run: trace tables", "", f"Run directory: `{run}`", "",
          "## Experiment summary", "",
          f"- Task: {t['suite']} task {t['task_id']} `{t['task_name']}`; instruction \"{t['instruction']}\"; init state index {t['init_state_index']} (same physical start every attempt).",
          f"- Robot/controller: {t['robot']}, {t['controller']} (delta, output max {t['controller_output_max']}), {t['control_freq_hz']} Hz; attempt horizon {t['attempt_horizon_steps']} steps ({t['attempt_horizon_steps'] * t['control_dt_s']:.0f} s simulated).",
          f"- Action interface: 7 numbers in [-1,1] (dx dy dz drx dry drz gripper) + repeat 1-{cfg['budgets']['max_repeat']} (capped by remaining horizon).",
          f"- Observation access: {', '.join(cfg['observation_access'])}. Images {t['image_size']}x{t['image_size']}.",
          f"- Memory format: {cfg['memory_format']}.",
          f"- Budgets: {json.dumps(cfg['budgets'])}.",
          f"- Model: requested `{cfg['model']['requested']}`, reasoning {cfg['model']['reasoning_effort']}; CLI `{cfg['model'].get('codex_version')}`; login: {cfg['model'].get('login_status')}.",
          f"- Run status: {res['status']}" + (f" (error: {short(res['error'], 300)})" if res.get("error") else "") + f"; model calls used {res.get('model_calls_total')}; any success: {res.get('any_success', 'run in progress')}; wall time {res.get('wall_time_s')} s.",
          f"- Policy: {cfg['policy']} (is_astra_run={cfg['is_astra_run']}); started {cfg['started']}; code {cfg.get('git')}.", ""]
    ids_seen = set()
    for e in [c.get("meta", {}).get("model_identity_in_events", []) for att in attempts for d in att["decisions"] for c in d["calls"]]:
        ids_seen.update(e)
    L += [f"- Server-reported model identity in CLI events: {sorted(ids_seen) if ids_seen else 'not exposed by the CLI'}.", ""]
    flat = []
    for att in attempts:
        s = att["summary"]
        L += [f"## Attempt {att['k']}: {s.get('outcome', '?')}", "",
              f"{s.get('outcome_text', '')}. Decisions {s.get('decisions_total')} (executed {s.get('decisions_executed')}), model calls {s.get('model_calls')}, env steps {s.get('env_steps')} "
              f"({s.get('simulated_time_s')} s simulated), model wall time {s.get('model_wall_time_s')} s, attempt wall time {s.get('attempt_wall_time_s')} s.", ""]
        sheet = contact_sheet(run, att)
        if sheet:
            L += [f"Contact sheet (pre-decision agentview frames): `{sheet.relative_to(run)}`; video: `{att['dir'].name}/agentview.mp4`.", ""]
        L += ["| Decision (id) | What Astra observed | Astra's assessment of its previous action | Stated intent / hypothesis | Executed action | Measured result | What changed in memory |",
              "|---|---|---|---|---|---|---|"]
        prev_mem = None
        if att["k"] > 1:
            prev_att = attempts[att["k"] - 2]
            for d in prev_att["decisions"]:
                if d.get("memory_after"):
                    prev_mem = d["memory_after"]
        for d in att["decisions"]:
            o = d.get("observation", {}).get("proprio", {})
            obs_txt = f"step {d['env_step_before']}: eef {o.get('eef_position_m_world')}, width {o.get('gripper_width_cm')} cm; imgs `{d['dir'].name}/pre_*.png`"
            c = d.get("decision_content")
            if c is None:
                reasons = "; ".join(short(x.get("reason") or x.get("skipped"), 80) for x in d["calls"])
                L.append(f"| D{d['decision']:02d} (no id) | {short(obs_txt)} | – | – | NONE: {reasons} | no execution | unchanged |")
                flat.append({"attempt": att["k"], "decision": d["decision"], "accepted": False, "calls": d["calls"]})
                continue
            act = "STOP requested" if c["stop"] else f"{fmt_action(c['action'])} x{d.get('repeat_executed')}" + (f" (asked {d['repeat_requested']}, capped)" if d.get("repeat_cap_reason") else "")
            r = d.get("result", {})
            res_txt = ("attempt ended by stop" if c["stop"] else
                       f"eef Δ {r.get('eef_delta_m')} m; width {r.get('gripper_width_before_cm')}→{r.get('gripper_width_after_cm')} cm; complete={r.get('task_complete')}" + (" **SUCCESS**" if r.get("task_complete") else ""))
            retry = f" ({len(d['calls'])} calls)" if len(d["calls"]) > 1 else ""
            hyp = c["hypothesis"] or "–"
            L.append(f"| D{d['decision']:02d} (id {d.get('interaction_id')}){retry} | {short(obs_txt)} | {short(c['assessment_of_previous_action'])} | **intent:** {short(c['intent'])} **hyp:** {short(hyp)} **expects:** {short(c['expected_change'])} refs {c['references']} | {act} | {res_txt} | {short(memory_change(prev_mem, d.get('memory_after') or c['memory_update']), 220)} |")
            prev_mem = d.get("memory_after") or c["memory_update"]
            flat.append({"attempt": att["k"], "decision": d["decision"], "id": d.get("interaction_id"), "accepted": True, "content": c,
                         "result": r, "repeat_requested": d.get("repeat_requested"), "repeat_executed": d.get("repeat_executed"),
                         "reasoning_summaries": [x for cc in d["calls"] for x in cc.get("reasoning_summaries", [])],
                         "latency_s": [cc.get("meta", {}).get("latency_s") for cc in d["calls"]]})
        L += ["", "### Per-decision detail (Astra's stated rationale, verbatim; Codex reasoning summaries where emitted)", ""]
        for d in att["decisions"]:
            c = d.get("decision_content")
            L.append(f"**D{d['decision']:02d}** (`{d['dir'].relative_to(run)}`): calls " + "; ".join(
                f"try {x.get('try')}: {'accepted' if x.get('accepted') else (x.get('reason') or x.get('skipped'))}, {x.get('meta', {}).get('latency_s', '?')} s" for x in d["calls"]))
            if c:
                L += [f"- assessment_of_previous_action: {c['assessment_of_previous_action']}", f"- intent: {c['intent']}", f"- hypothesis: {c['hypothesis']}",
                      f"- expected_change: {c['expected_change']}", f"- references: {c['references']}",
                      f"- memory_update.observations: {c['memory_update']['observations']}", f"- memory_update.hypotheses: {c['memory_update']['hypotheses']}",
                      f"- memory_update.summary: {c['memory_update']['summary']}"]
                if d.get("result"):
                    L.append(f"- measured: {json.dumps(d['result'])}")
            rs = [x for cc in d["calls"] for x in cc.get("reasoning_summaries", [])]
            if rs:
                L.append("- Codex reasoning summary (model-provided, not an internal transcript): " + " | ".join(short(x, 400) for x in rs))
            L.append("")
        L += ["### Recorded environment evidence (evaluation-only; never shown to Astra)", "", "```", json.dumps(evaluation_evidence(att), indent=1), "```", ""]
    L += ["## Cross-attempt metrics (automated)", "", "| attempt | outcome | decisions | accepted | failed calls | close cmds | open cmds | with hypothesis | references | identical action repeats | eef z range (m) | min width (cm) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for att in attempts:
        m = attempt_metrics(att)
        L.append(f"| {att['k']} | {att['summary'].get('outcome')} | {m['decisions']} | {m['accepted']} | {m['invalid_or_failed_calls']} | {m['close_commands']} | {m['open_commands']} | {m['decisions_with_hypothesis']} | {m['references_total']} ({m['distinct_referenced_ids']}) | {m['identical_action_repeats']} | {m['eef_z_min_max_m']} | {m['gripper_width_min_cm']} |")
    lat = [x for att in attempts for d in att["decisions"] for x in [c.get("meta", {}).get("latency_s") for c in d["calls"]] if x is not None]
    usage = Counter()
    for att in attempts:
        for d in att["decisions"]:
            for c in d["calls"]:
                u = c.get("meta", {}).get("usage") or {}
                for k, v in u.items():
                    if isinstance(v, (int, float)):
                        usage[k] += v
    L += ["", "## Latency and usage", "", f"- Model calls: {len(lat)}; latency median {np.median(lat):.1f} s, mean {np.mean(lat):.1f} s, max {np.max(lat):.1f} s." if lat else "- no calls",
          f"- Token usage (CLI-reported, summed): {dict(usage) if usage else 'not exposed'}.",
          f"- Simulated time: {sum(att['summary'].get('simulated_time_s', 0) for att in attempts):.1f} s total; real wall time {res.get('wall_time_s')} s.", ""]
    (run / "REPORT_TABLES.md").write_text("\n".join(L))
    with open(run / "decisions.jsonl", "w") as f:
        for row in flat:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {run / 'REPORT_TABLES.md'} ({len(flat)} decisions)")


if __name__ == "__main__":
    main()
