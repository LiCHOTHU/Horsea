"""Series B of the site: the overnight Astra -> policy transfer study (2026-10-10).

Called by build_site.py (`--transfer <study root>`). Writes <site>/transfer/index.html (overview: bottom line, 12 x 4 matrix,
results at a glance, timeline of the night, Astra-call budget, findings, full report) and one page per platform (Astra's own
episodes with video, filmstrip and decision trace; the per-start outcome grid of every student cell; each episode with video;
training curves; the data/interface diagnosis). Media come from the study directory through the symlink transfer/run.
"""
import datetime as dt
import html
import json
import os
from pathlib import Path

ESC = html.escape
PLATFORMS = [("robotwin_single", "RoboTwin single-arm", "place_container_plate"), ("robotwin_bimanual", "RoboTwin bimanual", "lift_pot"),
             ("dexhand", "Dex hand (Isaac)", "Humanoid-Stack-Can-Into-Drawer-v0"), ("libero", "LIBERO", "LIBERO-90 task 48 (ketchup to basket)")]
CELL_NAMES = {"E01": "Astra only", "E02": "original student", "E03": "trained on D1", "E04": "trained on D3", "E05": "trained on DN (anchor)",
              "E06": "anchor, half executed chunk", "E07": "trained on successes + failures", "E08": "anchor + Astra takeover",
              "E09": "takeover + student history", "E10": "anchor + extra training", "E11": "anchor + Astra corrections",
              "E12": "anchor + extra Astra demos"}
STATUS_COLOR = {"COMPLETE": "#2563eb", "PENDING_BUDGET": "#d97706", "NO_STUDENT_SIGNAL": "#b45309", "BLOCKED_DATA": "#6b7280",
                "BLOCKED_INTERFACE": "#9ca3af", "NOT_AVAILABLE": "#cbd5e1", "DUPLICATE": "#7c3aed", "FAILED": "#c0392b", "NOT_APPLICABLE": "#cbd5e1"}
KIND_COLOR = {"teacher": "#7c3aed", "eval": "#0d9488", "train": "#ea580c", "hybrid": "#db2777", "replay": "#64748b", "audit": "#94a3b8"}
EXTRA_CSS = """
.series{border-left:5px solid var(--acc);padding-left:14px}.chip{display:inline-block;padding:3px 8px;border-radius:6px;color:#fff;font-size:12px;font-weight:600}
.mx td{text-align:center;vertical-align:middle;font-size:13px}.mx td.l{text-align:left}.mx .chip{display:block}
.bars .row{display:grid;grid-template-columns:270px 1fr 70px;gap:8px;align-items:center;margin:3px 0;font-size:13px}
.bars .track{background:#eef0f4;border-radius:5px;height:16px;position:relative;overflow:hidden}.bars .fill{height:100%;border-radius:5px}
.tl{position:relative;border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px 8px 22px 150px}
.tl .lane{position:relative;height:26px;margin:4px 0}.tl .lab{position:absolute;left:-142px;top:4px;font-size:13px;width:136px}
.tl .seg{position:absolute;top:3px;height:20px;border-radius:4px;opacity:.9}.tl .tick{position:absolute;top:calc(100% + 4px);font-size:11px;color:var(--muted);transform:translateX(-50%)}
.tl .grid{position:absolute;top:0;bottom:0;border-left:1px dashed #e5e7eb}
.og{border-collapse:separate;border-spacing:3px}.og td,.og th{border:none;padding:0;font-size:12px}.og th{background:none;color:var(--muted);font-weight:500;padding:0 2px}
.og .sq{display:block;width:32px;height:24px;border-radius:4px;color:#fff;text-align:center;line-height:24px;font-weight:700;font-size:12px}
.og .S{background:#16a34a}.og .F{background:#dc2626}.og .N{background:#e5e7eb;color:#9ca3af}.og td.rl{padding-right:10px;white-space:nowrap;font-size:13px}
.eps{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}.ep{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px;scroll-margin-top:60px}
.ep img.strip{width:100%;border-radius:4px;border:1px solid var(--line)}.ep video{width:100%;max-height:420px;margin-top:6px;object-fit:contain}
.legend span{display:inline-block;margin-right:14px;font-size:12px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.kpi{display:flex;gap:12px;flex-wrap:wrap}.kpi div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}.kpi b{display:block;font-size:22px}
"""


def rj(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return default


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


class Study:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state = rj(self.root / "state.json", {})
        self.window = rj(self.root / "window.json", {})
        self.bank = {p: [json.loads(x) for x in (self.root / p / "bank.jsonl").read_text().splitlines() if x.strip()]
                     if (self.root / p / "bank.jsonl").exists() else [] for p, _, _ in PLATFORMS}
        self.events = [json.loads(x) for x in (self.root / "astra_ledger.jsonl").read_text().splitlines() if x.strip()] \
            if (self.root / "astra_ledger.jsonl").exists() else []

    def rel(self, path):
        try:
            return "run/" + str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return None

    def calls(self, p=None, c=None):
        return sum(1 for e in self.events if e.get("kind") == "call" and (p is None or e["platform"] == p) and (c is None or e.get("cell") == c))

    def tokens(self, p=None):
        t = {}
        for e in self.events:
            if e.get("kind") == "usage" and (p is None or e["platform"] == p):
                for k, v in (e.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        t[k] = t.get(k, 0) + v
        return t

    def cell(self, p, c):
        return self.state["cells"][p][c]


def frac(r):
    return f"{r.get('successes')}/{r.get('n')}" if r and r.get("n") else ""


def chip(status, text=""):
    return f"<span class='chip' style='background:{STATUS_COLOR.get(status, '#6b7280')}'>{ESC(status)}{('<br>' + ESC(text)) if text else ''}</span>"


def outcome_badge(ok):
    return "<span class='badge ok'>SUCCESS</span>" if ok else "<span class='badge bad'>NO SUCCESS</span>"


# ------------------------------------------------------------------ overview pieces
def matrix_html(S):
    head = "<tr><th>cell</th><th>configuration</th>" + "".join(f"<th>{ESC(n)}</th>" for _, n, _ in PLATFORMS) + "</tr>"
    rows = []
    for c, desc in CELL_NAMES.items():
        tds = []
        for p, _, _ in PLATFORMS:
            rec = S.cell(p, c)
            r = rec.get("result") or {}
            txt = []
            if r.get("n"):
                txt.append(f"{frac(r)} {r.get('pool', 'dev')}")
            if (rec.get("test") or {}).get("n"):
                txt.append(f"{frac(rec['test'])} test")
            inner = chip(rec["status"], " · ".join(txt))
            if rec["status"] == "COMPLETE" or txt:
                inner = f"<a href='{p}.html#{c}' style='text-decoration:none'>{inner}</a>"
            tds.append(f"<td title='{ESC(rec.get('reason') or '')}'>{inner}</td>")
        rows.append(f"<tr><td><b>{c}</b></td><td class='l'>{ESC(desc)}</td>{''.join(tds)}</tr>")
    legend = "".join(f"<span><i style='background:{v}'></i>{ESC(k)}</span>" for k, v in STATUS_COLOR.items() if k not in ("NOT_APPLICABLE",))
    return f"<table class='mx'>{head}{''.join(rows)}</table><p class='legend'>{legend}</p><p class='small'>Hover a cell for the reason of its status; click a result to open its episodes.</p>"


def bar(label, k, n, color, note=""):
    w = 0 if not n else 100.0 * k / n
    return (f"<div class='row'><div>{label}</div><div class='track'><div class='fill' style='width:{w:.1f}%;background:{color}'></div></div>"
            f"<div><b>{k}/{n}</b>{(' ' + note) if note else ''}</div></div>")


def glance_html(S):
    out = []
    for p, name, task in PLATFORMS:
        rows = []
        src = [e for e in S.bank[p] if e.get("cell") == "SRC"]
        if src:
            rows.append(bar("Astra from reset · source episodes", sum(1 for e in src if e["success"]), len(src), KIND_COLOR["teacher"]))
        e01 = S.cell(p, "E01").get("result") or {}
        if e01.get("n"):
            rows.append(bar("Astra from reset · E01 dev reference", e01["successes"], e01["n"], KIND_COLOR["teacher"]))
        for c in ("E02", "E03", "E05", "E06", "E07", "E10"):
            rec = S.cell(p, c)
            r = rec.get("result") or {}
            if rec["status"] == "COMPLETE" and r.get("n"):
                rows.append(bar(f"{c} {CELL_NAMES[c]} · dev", r["successes"], r["n"], "#5eead4" if c != "E02" else "#94a3b8"))
            t = rec.get("test") or {}
            if t.get("n"):
                rows.append(bar(f"{c} {CELL_NAMES[c]} · <b>test</b>", t["successes"], t["n"], "#0d9488" if c != "E02" else "#475569"))
        if not rows:
            continue
        out.append(f"<h3 style='margin:14px 0 4px'><a href='{p}.html'>{ESC(name)}</a> <span class='small'>{ESC(task)}</span></h3><div class='bars'>{''.join(rows)}</div>")
    legend = (f"<p class='legend'><span><i style='background:{KIND_COLOR['teacher']}'></i>Astra (System 2)</span><span><i style='background:#94a3b8'></i>original student, dev</span>"
              "<span><i style='background:#475569'></i>original student, test</span><span><i style='background:#5eead4'></i>student trained on Astra data, dev</span>"
              "<span><i style='background:#0d9488'></i>student trained on Astra data, test</span></p>")
    return legend + "".join(out) + "<p class='small'>Dev = 3-5 fixed development starts; test = 10 fresh test starts evaluated after all training was frozen. Bars are raw counts, not established rates.</p>"


def timeline_html(S):
    w0 = S.window.get("start")
    if not w0:
        return ""
    span = 3600.0 * S.window.get("hours", 8)
    lanes = []
    for p, name, _ in PLATFORMS:
        segs = []
        for aid, a in S.state.get("activities", {}).items():
            if a.get("platform") != p or not a.get("start") or not a.get("end"):
                continue
            x0, x1 = (ts(a["start"]) - w0) / span * 100, (ts(a["end"]) - w0) / span * 100
            res = rj(a.get("result"), {}) if a.get("result") else {}
            tip = f"{aid}: {a['start'][11:16]}-{a['end'][11:16]} ({round(a.get('duration_s', 0) / 60, 1)} min) {res.get('summary', '')}"
            segs.append(f"<div class='seg' title='{ESC(tip)}' style='left:{x0:.2f}%;width:{max(0.25, x1 - x0):.2f}%;background:{KIND_COLOR.get(a['kind'], '#999')}'></div>")
        lanes.append(f"<div class='lane'><div class='lab'>{ESC(name)}</div>{''.join(segs)}</div>")
    ticks = "".join(f"<div class='grid' style='left:{h / (span / 3600) * 100:.2f}%'></div><div class='tick' style='left:{h / (span / 3600) * 100:.2f}%'>"
                    f"{dt.datetime.fromtimestamp(w0 + 3600 * h).strftime('%H:%M')}</div>" for h in range(0, int(span / 3600) + 1))
    legend = "".join(f"<span><i style='background:{v}'></i>{k}</span>" for k, v in KIND_COLOR.items() if k != "hybrid")
    return (f"<div class='tl'><div style='position:relative'>{ticks}{''.join(lanes)}</div></div>"
            f"<p class='legend'>{legend.replace('teacher', 'Astra episodes').replace('>eval', '>student evaluation').replace('>train', '>student training')}</p>"
            "<p class='small'>One activity at a time (the spec's single active cell). Hover a bar for its id, time and result.</p>")


def budget_html(S):
    rows = []
    cap = 150
    for p, name, _ in PLATFORMS:
        used = S.calls(p)
        segs = []
        for c, col in (("SRC", "#7c3aed"), ("E01", "#a78bfa")):
            n = S.calls(p, c)
            if n:
                segs.append(f"<div class='fill' title='{c}: {n} calls' style='position:absolute;left:{100.0 * (sum(S.calls(p, x) for x in ('SRC',) if c == 'E01')) / cap:.1f}%;width:{100.0 * n / cap:.1f}%;background:{col}'></div>")
        rows.append(f"<div class='row'><div>{ESC(name)}</div><div class='track'>{''.join(segs)}</div><div><b>{used}</b>/{cap}</div></div>")
    t = S.tokens()
    return (f"<div class='bars'>{''.join(rows)}</div><p class='legend'><span><i style='background:#7c3aed'></i>source collection (teacher data)</span>"
            f"<span><i style='background:#a78bfa'></i>E01 Astra-only reference</span></p>"
            f"<p class='small'>{S.calls()} of 600 calls overall; every call returned a valid response. Tokens: input {t.get('input_tokens', 0):,} "
            f"(cached {t.get('cached_input_tokens', 0):,}), output {t.get('output_tokens', 0):,}. A 26-call dry run before launch is outside the ledger.</p>")


# ------------------------------------------------------------------ platform pieces
def ep_card(S, meta_or_entry, title, video=None, strip=None, extra="", poster=None):
    m = meta_or_entry
    eid = Path(m.get("path") or "").name
    v = f"<video controls preload='none'{f' poster={chr(39)}{poster}{chr(39)}' if poster else ''} src='{video}'></video>" if video else ""
    s = f"<a href='{video or strip}'><img class='strip' loading='lazy' src='{strip}'></a>" if strip else ""
    return f"<div class='ep' id='ep-{ESC(eid)}'><div><b>{title}</b> {outcome_badge(m.get('success'))}</div>{s}{v}<div class='small'>{extra}</div></div>"


SITE_OUT = None          # set by build(): posters of harness videos (LIBERO) are written into the site, not the study


def media(S, path):
    """(video, strip, poster) hrefs for an episode directory."""
    p = Path(path)
    vid = S.rel(p / "video.mp4") if (p / "video.mp4").exists() else None
    poster = S.rel(p / "poster.jpg") if (p / "poster.jpg").exists() else None
    if not vid:
        mp4 = next(iter(sorted(p.glob("attempt_*/*.mp4"))), None)
        vid = S.rel(mp4) if mp4 else None
        if mp4 and SITE_OUT is not None:
            from build_site import ensure_poster
            if ensure_poster(mp4, SITE_OUT / "posters" / f"{p.name}.jpg"):
                poster = f"posters/{p.name}.jpg"
    strip = S.rel(p / "strip.jpg") if (p / "strip.jpg").exists() else None
    return vid, strip, poster


def trace_table(meta):
    tr = meta.get("trace") or []
    if not tr:
        return ""
    rows = "".join(f"<tr><td>{i + 1}</td><td>{ESC(str(t.get('objective', '')))}</td><td class='mono'>{ESC(str(t.get('action', '')))[:160]}</td>"
                   f"<td>{ESC(str(t.get('result', '')))[:160]}</td></tr>" for i, t in enumerate(tr))
    return (f"<details><summary>Astra's decisions in this episode ({len(tr)}): objective → executed action → observed result</summary>"
            f"<table><tr><th>#</th><th>[A] stated objective</th><th>[E] executed action</th><th>[E] observed result</th></tr>{rows}</table></details>")


def astra_section(S, p):
    eps = [e for e in S.bank[p] if e.get("cell") in ("SRC", "E01", "CORR", "EXTRA")]
    if not eps:
        return ""
    cards = []
    for e in eps:
        meta = rj(Path(e["path"]) / "episode.json", {}) or {}
        vid, strip, poster = media(S, e["path"])
        role = {"SRC": "source collection (teacher data)", "E01": "E01 Astra-only reference (dev pool)"}.get(e["cell"], e["cell"])
        fs = ""
        pe = meta.get("privileged_end") or {}
        if p == "robotwin_single" and pe.get("container_pos"):
            fs = f"container end z {pe['container_pos'][2]:.3f} m"
        elif p == "robotwin_bimanual" and pe.get("pot_pos"):
            fs = f"pot end z {pe['pot_pos'][2]:.3f} m"
        elif p == "dexhand" and pe.get("can_pos"):
            fs = f"can end z {pe['can_pos'][2]:.3f} m, door {(pe.get('drawer_joint_pos') or [0])[0]:.2f} rad"
        extra = (f"{ESC(role)} · {e.get('astra_calls')} Astra calls · {e.get('native_steps')} native steps"
                 + (f" · {ESC(fs)}" if fs else "")
                 + f" · <a href='{S.rel(Path(e['path']) / ('astra' if (Path(e['path']) / 'astra').exists() else ''))}/'>decision records (images Astra saw, prompts, responses)</a>"
                 + trace_table(meta))
        cards.append(ep_card(S, e, f"{e['cell']} · start {e['start_id']}", vid, strip, extra, poster))
    n, k = len(eps), sum(1 for e in eps if e["success"])
    return (f"<div class='card'><h2 id='astra'>Astra's own episodes (System 2, from reset): {k}/{n} successful</h2>"
            "<p class='small'>Chronological teacher-attempt bank, failures included. Astra carries its own bounded memory across the episodes of a "
            "stream." + (" Purple frames = Astra in control. Click a filmstrip to open the video." if p != "libero" else
                         " Video: the harness's own recording with the commanded action overlaid.") + "</p>"
            f"<div class='eps'>{''.join(cards)}</div></div>")


def student_runs(S, p):
    """{(cell, kind): [episode entries]} for kind in dev / test / diag."""
    out = {}
    for c in CELL_NAMES:
        rec = S.cell(p, c)
        if c == "E01":
            continue
        r = rec.get("result") or {}
        if rec["status"] in ("COMPLETE", "PENDING_BUDGET") and r.get("per_episode"):
            out[(c, "dev")] = [e for e in r["per_episode"] if e.get("path")]
        if (rec.get("test") or {}).get("per_episode"):
            out[(c, "test")] = [e for e in rec["test"]["per_episode"] if e.get("path")]
    dg = (S.state["platforms"].get(p) or {}).get("diagnosis") or {}
    for c, r in (dg.get("results") or {}).items():
        out[(c, "diag")] = [e for e in r.get("per_episode") or [] if e.get("path")]
    return out


def outcome_grid(S, p, runs):
    pl = S.state["platforms"].get(p) or {}
    pools = pl.get("pools") or {}
    dev, test = pools.get("dev", [])[:5], pools.get("test", [])[:10]
    diag = (pl.get("diagnosis") or {}).get("starts") or []
    cols = [("dev", s) for s in dev] + [("test", s) for s in test] + [("diag", s) for s in diag]
    if not cols or not runs:
        return ""
    cells = sorted({c for c, _ in runs}, key=lambda c: list(CELL_NAMES).index(c))
    head = ("<tr><th></th>" + "".join(f"<th title='{k} start {s}'>{str(s)[-3:]}</th>" for k, s in cols) + "</tr>"
            f"<tr><th></th><th colspan='{len(dev)}' style='border-bottom:2px solid #94a3b8'>development starts</th>"
            f"<th colspan='{len(test)}' style='border-bottom:2px solid #0f766e'>test starts (after freezing)</th>"
            + (f"<th colspan='{len(diag)}' style='border-bottom:2px solid #a16207'>own training starts</th>" if diag else "") + "</tr>")
    rows = []
    for c in cells:
        tds = []
        summ = []
        for kind in ("dev", "test", "diag"):
            eps = runs.get((c, kind)) or []
            if eps:
                summ.append(f"{kind} {sum(1 for e in eps if e.get('success'))}/{len(eps)}")
        for kind, s in cols:
            e = next((x for x in runs.get((c, kind)) or [] if x.get("start_id") == s), None)
            if e is None:
                tds.append("<td><span class='sq N'>·</span></td>")
            else:
                cls = "S" if e.get("success") else "F"
                tds.append(f"<td><a href='#ep-{ESC(Path(e['path']).name)}' title='{c} {kind} start {s}: {cls}'><span class='sq {cls}'>{cls}</span></a></td>")
        rows.append(f"<tr><td class='rl'><a href='#{c}'><b>{c}</b></a> {ESC(CELL_NAMES[c])}<br><span class='small'>{' · '.join(summ)}</span></td>{''.join(tds)}</tr>")
    return (f"<table class='og'>{head}{''.join(rows)}</table><p class='small'>S = full-task success, F = no success, · = not evaluated. "
            "Every student shares the same start, so a column compares students on the same initial state. Click a square to jump to that episode.</p>")


def loss_svg(path):
    try:
        L = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except OSError:
        return ""
    if len(L) < 2:
        return ""
    xs, ys = [x["update"] for x in L], [x["loss"] for x in L]
    W, H = 260, 60
    mx, my = max(xs), max(ys)
    pts = " ".join(f"{4 + (W - 8) * x / mx:.1f},{H - 4 - (H - 8) * y / my:.1f}" for x, y in zip(xs, ys))
    return (f"<svg width='{W}' height='{H}' style='background:#fafafa;border:1px solid var(--line);border-radius:4px'><polyline fill='none' stroke='#ea580c' "
            f"stroke-width='2' points='{pts}'/></svg><div class='small'>training loss {ys[0]:.4f} → {ys[-1]:.4f} over {xs[-1]} updates</div>")


def cell_sections(S, p, runs):
    out = []
    acts = S.state.get("activities", {})
    for c in sorted({c for c, _ in runs}, key=lambda c: list(CELL_NAMES).index(c)):
        rec = S.cell(p, c)
        tr = acts.get(f"{p}__train__{c}__s0")
        train = ""
        if tr and tr.get("result"):
            meta = rj(Path(tr["out"]) / "train_meta.json", {}) or {}
            ds = "; ".join(f"{k}: {v.get('episodes')} episode(s), {v.get('anchors')} labelled frames" for k, v in (meta.get("datasets") or {}).items())
            train = (f"<div class='grid' style='grid-template-columns:300px 1fr'><div>{loss_svg(Path(tr['out']) / 'train_log.jsonl')}</div>"
                     f"<div class='small'>[E] trained from <span class='mono'>{ESC(Path(meta.get('init', '')).name)}</span> for {meta.get('updates')} updates "
                     f"(batch {meta.get('batch')}, mix {meta.get('mix')}); data {ESC(ds)}; {meta.get('train_s')} s.</div></div>")
        blocks = []
        for kind, label in (("dev", "development starts"), ("test", "test starts"), ("diag", "its own training starts (diagnosis)")):
            eps = runs.get((c, kind)) or []
            if not eps:
                continue
            cards = []
            for e in eps:
                vid, strip, poster = media(S, e["path"])
                extra = (f"{e.get('native_steps')} native steps · {e.get('policy_inference_calls')} policy calls"
                         + (f" · end state: {ESC(str(e.get('failure_stage')))}" if e.get("failure_stage") else "") + f" · {e.get('wall_s')} s")
                cards.append(ep_card(S, e, f"{c} · {kind} start {e.get('start_id')}", vid, strip, extra, poster))
            k = sum(1 for e in eps if e.get("success"))
            blocks.append(f"<h3>{ESC(label)}: {k}/{len(eps)}</h3><div class='eps'>{''.join(cards)}</div>")
        why = f"<p class='small'>Status reason: {ESC(rec['reason'])}</p>" if rec.get("reason") else ""
        out.append(f"<div class='card' id='{c}'><h2>{c} · {ESC(CELL_NAMES[c])} {chip(rec['status'])}</h2>{why}{train}{''.join(blocks)}</div>")
    return "".join(out)


def diagnosis_html(S, p):
    ol = rj(S.root / p / "diagnosis" / "openloop_train_frames.json")
    if not ol:
        return ""
    rows = "".join(f"<tr><td>{ESC(k.split('/')[0])}</td><td>{ESC(k.split('/')[1])}</td><td>{v['arm_mae_rad_first8']}</td><td>{v['gripper_mae_first8']}</td><td>{v['anchors']}</td></tr>"
                   for k, v in ol["results"].items())
    return (f"<div class='card'><h2>Data / interface diagnosis (outside the matrix)</h2><p class='small'>{ESC(ol.get('rule', ''))}. Lower error means the student "
            "predicts Astra's recorded commands more closely on the very frames it was trained on.</p><table><tr><th>student</th><th>Astra episode</th>"
            f"<th>arm joint error (rad, first 8 actions)</th><th>gripper error</th><th>frames checked</th></tr>{rows}</table></div>")


# ------------------------------------------------------------------ pages
def platform_page(S, p, name, task, page):
    pl = S.state["platforms"].get(p) or {}
    man = rj(next(iter((S.root / p / "activities").glob(f"{p}__audit*/manifest.json")), None) or "", {}) or {}
    setup = [("Task", task + (f"; instruction: {man['instruction']}" if man.get("instruction") else "")), ("Cohort", pl.get("cohort")),
             ("Student", (pl.get("student") or {}).get("summary")), ("Student observation", man.get("student_observation")),
             ("Student action", man.get("student_action")), ("Astra action", man.get("teacher_action")),
             ("Success", man.get("success_definition")), ("Audit / replay check", (pl.get("audit") or {}).get("alignment") or (pl.get("audit") or {}).get("summary")),
             ("Astra calls", f"{S.calls(p)} of 150")]
    setup_html = "".join(f"<tr><th style='width:190px'>{ESC(k)}</th><td>{ESC(str(v))}</td></tr>" for k, v in setup if v)
    runs = student_runs(S, p)
    grid = outcome_grid(S, p, runs)
    body = (f"<h1>{ESC(name)} · {ESC(task)}</h1><p><a href='index.html'>← transfer study overview</a></p>"
            f"<details class='card' open><summary><b>Setup and audit</b></summary><table>{setup_html}</table></details>"
            + (f"<div class='card'><h2>Students, start by start</h2>{grid}</div>" if grid else
               "<div class='card'><h2>Students</h2><p>No student cell ran on this platform: " + ESC("; ".join(sorted({S.cell(p, c)['status'] for c in CELL_NAMES if c != 'E01'}))) + ".</p></div>")
            + astra_section(S, p) + cell_sections(S, p, runs) + diagnosis_html(S, p))
    return page(f"Transfer study · {name}", body)


def overview_page(S, page, report_html, launch_html):
    tot = S.calls()
    kpis = (f"<div class='kpi'><div><b>4</b>platforms</div><div><b>12</b>registered configurations</div><div><b>{sum(1 for p, _, _ in PLATFORMS for c in CELL_NAMES if S.cell(p, c)['status'] == 'COMPLETE')}</b>cells complete</div>"
            f"<div><b>{tot}/600</b>Astra calls</div><div><b>{len(S.state.get('activities', {}))}</b>activities, one at a time</div></div>")
    bottom = ("<table><tr><th>platform</th><th>Astra on its own</th><th>original student</th><th>student trained on Astra data</th></tr>"
              "<tr><td><a href='robotwin_single.html'>RoboTwin single-arm</a></td><td>2/4 source · 1/1 reference</td><td>2/5 dev · 3/10 test</td><td>0/5 dev · 1/10 test (E05); 3/10 test after extra training (E10)</td></tr>"
              "<tr><td><a href='robotwin_bimanual.html'>RoboTwin bimanual</a></td><td>0/4</td><td>3/3 dev · 2/10 test</td><td>not possible: no Astra success</td></tr>"
              "<tr><td><a href='dexhand.html'>Dex hand</a></td><td>0/6</td><td>none exists</td><td>not possible: no Astra success</td></tr>"
              "<tr><td><a href='libero.html'>LIBERO</a></td><td>1/3 reference</td><td>no compatible student</td><td>blocked by the interface</td></tr></table>")
    findings = ("<ul><li><b>No channel from Astra to the student produced a better student.</b> Astra's recorded data are valid (replay reproduces its motion "
                "within 0.1 cm on RoboTwin, 1.3 cm on the dex hand), but training on Astra's two single-arm successes did not help, and the trained students "
                "failed even the starts they were trained on.</li><li><b>On the test starts each single-arm student succeeded on different starts</b>: the "
                "differences are noise.</li><li><b>Takeover and correction experiments never ran</b>: the registered gate stops them when the trained student has "
                "no success in five development starts.</li><li><b>Bimanual and dex had nothing to learn from</b>: Astra itself did not succeed; on lift_pot the "
                "unchanged student did better than Astra.</li></ul>")
    body = ("<div class='series'><h1>Series B · Overnight transfer study: can a policy learn from Astra?</h1>"
            "<p>Astra acts as the teacher (System 2), a learned robot policy as the student (System 1). Twelve registered configurations on four simulators "
            "test whether Astra's own interaction data, its takeovers and its corrections make the student better once Astra is switched off. "
            "Ran 2026-10-10, 06:16-14:06, under a fixed budget (8 hours, at most 150 Astra calls per platform).</p></div>"
            f"{kpis}<div class='card'><h2>Bottom line</h2>{bottom}{findings}</div>"
            f"<div class='card'><h2>Results at a glance</h2>{glance_html(S)}</div>"
            f"<div class='card'><h2>The 12 × 4 matrix</h2>{matrix_html(S)}</div>"
            f"<div class='card'><h2>The night, hour by hour</h2>{timeline_html(S)}</div>"
            f"<div class='card'><h2>Astra-call budget</h2>{budget_html(S)}</div>"
            f"<details class='card'><summary><b>Full overnight report (ten answers per platform, costs, decisions, cautions)</b></summary><div class='report'>{report_html}</div></details>"
            f"<details class='card'><summary><b>Launch record and every rule change during the night</b></summary><div class='report'>{launch_html}</div></details>")
    return page("Transfer study · overview", body)


def build(site_out, study_root, page_fn):
    """page_fn(title, body, depth) -> html. Returns the home-page summary card for series B."""
    global SITE_OUT
    S = Study(study_root)
    out = Path(site_out) / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    SITE_OUT = out
    link = out / "run"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(S.root, out))
    try:
        import markdown
        md = lambda t: markdown.markdown(t, extensions=["tables", "fenced_code"])  # noqa: E731
    except Exception:  # noqa: BLE001
        md = lambda t: f"<pre>{ESC(t)}</pre>"  # noqa: E731
    rep = (S.root / "OVERNIGHT_REPORT.md").read_text() if (S.root / "OVERNIGHT_REPORT.md").exists() else ""
    lau = (S.root / "LAUNCH.md").read_text() if (S.root / "LAUNCH.md").exists() else ""
    pg = lambda title, body: page_fn(title, f"<style>{EXTRA_CSS}</style>" + body, 1)  # noqa: E731
    (out / "index.html").write_text(overview_page(S, pg, md(rep), md(lau)))
    for p, name, task in PLATFORMS:
        (out / f"{p}.html").write_text(platform_page(S, p, name, task, pg))
    return S
