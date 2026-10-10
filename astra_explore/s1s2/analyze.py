"""Evidence extraction for the human-readable overnight report (read-only; never changes the study state).

    python -m s1s2.analyze --root <study dir> [--platform robotwin_single]

Prints, per platform:
  - the chronological teacher-attempt bank (successes AND failures, calls, native steps, ceilings);
  - per evaluated cell, each episode's start, outcome, native steps, failure stage and the last trace lines
    ("objective -> controller -> executed action -> observed result -> next decision");
  - for E08 / E09 on the same start: the state at handover, Astra's first decisions after takeover (intent, targets) and
    whether the history package changed the first action choice (observation only; interpretation is left to the reader).
"""
import argparse
import json
from pathlib import Path


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def first_decisions(epdir, k=2):
    out = []
    for d in sorted((Path(epdir) / "astra").glob("decision_*"))[:k]:
        r = _load(d / "decision.json") or {}
        dc = r.get("decision_content") or {}
        tgt = {}
        for arm in ("left", "right"):
            a = dc.get(arm)
            if isinstance(a, dict):
                tgt[arm] = {kk: a.get(kk) for kk in ("move", "position_m", "gripper", "fingers_dir", "finger_base", "finger_tip") if kk in a}
        res = r.get("result") or {}
        out.append({"decision": d.name, "intent": dc.get("intent"), "assessment": dc.get("assessment_of_previous_action"),
                    "targets": tgt, "stop": dc.get("stop"), "steps": [r.get("native_steps_before"), r.get("native_steps_after")],
                    "result_complete": res.get("task_complete")})
    return out


def platform_report(root, p):
    root = Path(root)
    state = _load(root / "state.json") or {}
    L = [f"=== {p}"]
    bank = [json.loads(x) for x in (root / p / "bank.jsonl").read_text().splitlines()] if (root / p / "bank.jsonl").exists() else []
    if bank:
        L.append("teacher-attempt bank (chronological):")
        for e in bank:
            L.append(f"  {e.get('cell'):5s} start {e.get('start_id')} success {e.get('success')} valid {e.get('valid')} calls {e.get('astra_calls')} "
                     f"steps {e.get('native_steps')} ceilings {e.get('call_ceiling', 40)}/{e.get('step_ceiling', '-')}  {e.get('episode_id')}")
    cells = (state.get("cells") or {}).get(p, {})
    for c, rec in cells.items():
        r = rec.get("result") or {}
        eps = [e for e in r.get("per_episode") or [] if e.get("path")]
        if not eps:
            continue
        L.append(f"{c} {rec.get('status')} {r.get('successes')}/{r.get('n')} ({r.get('pool')})")
        for e in eps:
            meta = _load(Path(e["path"]) / "episode.json") or {}
            L.append(f"  start {e.get('start_id')}: success {e.get('success')} steps {e.get('native_steps')} policy calls {e.get('policy_inference_calls')} "
                     f"astra calls {e.get('astra_calls', 0)} handover {e.get('handover_step')} failure stage {e.get('failure_stage')}")
            for t in (meta.get("trace") or [])[-3:]:
                L.append(f"    {t.get('objective')} -> {t.get('controller')} -> {t.get('action')} -> {t.get('result')} -> {t.get('next')}")
    e08 = {e["start_id"]: e for e in ((cells.get("E08") or {}).get("result") or {}).get("per_episode") or [] if e.get("path")}
    e09 = {e["start_id"]: e for e in ((cells.get("E09") or {}).get("result") or {}).get("per_episode") or [] if e.get("path")}
    for s in sorted(set(e08) & set(e09)):
        a, b = _load(Path(e08[s]["path"]) / "episode.json") or {}, _load(Path(e09[s]["path"]) / "episode.json") or {}
        L.append(f"E08 vs E09 on start {s}: success {a.get('success')} / {b.get('success')}; handover step "
                 f"{(a.get('handover') or {}).get('native_step')} / {(b.get('handover') or {}).get('native_step')}")
        L.append(f"  state at handover (evaluation-only record): {json.dumps((a.get('handover') or {}).get('privileged_at_handover'))[:400]}")
        fa, fb = first_decisions(e08[s]["path"]), first_decisions(e09[s]["path"])
        for name, fd in (("E08", fa), ("E09", fb)):
            for d in fd:
                L.append(f"  {name} {d['decision']}: intent {d['intent']!r}; targets {json.dumps(d['targets'])[:300]}")
        if fa and fb:
            same = fa[0]["targets"] == fb[0]["targets"]
            L.append(f"  first action identical: {same}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--platform", default=None)
    a = ap.parse_args()
    from s1s2.registry import PLATFORMS
    for p in ([a.platform] if a.platform else PLATFORMS):
        print(platform_report(a.root, p))


if __name__ == "__main__":
    main()
