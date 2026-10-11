"""Representative videos for the task-generic pilot (rendering on; run after the formal blocks).

    python -m s1local.render_task --spec <name> --out <run dir> --jobs <jobs.json>

jobs.json: [{"op", "cond", "start", "theta", "label"}]. Each job rebuilds its start the way runner_task does (reset, the
operation's `shift` objects displaced for fresh starts, reference prefix), applies the condition, executes the edited
program and writes videos/<label>.mp4 (main camera, every 2nd control step, top bar: label, operation, condition, start,
step). Restores are not bit-exact (PhysX contact caches), so a re-executed rollout can differ from the logged one; each
video states its own re-executed outcome.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from s1local import env as E  # noqa: E402
from s1local import generic as G  # noqa: E402
from s1local.runner_task import CONDITIONS  # noqa: E402
from s1local.specs import SPECS, make_region_fns  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def bar(px, text, color):
    im = Image.fromarray(px)
    c = Image.new("RGB", (im.width, im.height + 26), color)
    c.paste(im, (0, 26))
    try:
        f = ImageFont.truetype(FONT, 15)
    except OSError:
        f = ImageFont.load_default()
    ImageDraw.Draw(c).text((8, 5), text, fill=(255, 255, 255), font=f)
    return np.asarray(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args()
    out = Path(a.out) / "videos"
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.jobs).read_text())
    spec = SPECS[a.spec]
    from isaac_env import IsaacDexTask
    T = IsaacDexTask(spec["task"])                       # rendering on (no cfg_hook)
    phys = G.Physics(T)
    prog = G.load_program(spec["source_run"], spec["source_attempt"])
    ops = spec["ops"]
    for o in ops.values():
        if isinstance(o["object"], list) and isinstance(o["object"][1], str):
            o["object"] = [o["object"][0], int(getattr(T.U, o["object"][1]))]
    region = make_region_fns(T)
    frames, trace, cur = [], [], {"op": None, "n": 0}
    step0 = T._step

    def _step(act):
        r = step0(act)
        if cur["op"] is not None:
            cur["n"] += 1
            trace.append(G.track(T, ops[cur["op"]]))
            if cur["n"] % 2 == 0:
                frames.append(T._image("main_camera", (640, 360)))
        return r
    T._step = _step
    snaps, results = {}, []
    for j in jobs:
        op, sid = j["op"], j["start"]
        o = ops[op]
        if (op, sid) not in snaps:
            phys.nominal()
            G.reset_with_shift(T, o["shift"], sid)
            for i in o["prefix"]:
                E.act(T, prog[i])
            snaps[(op, sid)] = E.snapshot(T)
        E.restore(T, snaps[(op, sid)])
        phys.apply(o, *CONDITIONS[j["cond"]])
        start = G.object_state(T)
        edited = G.edited_program(o, prog, np.asarray(j["theta"], float), T._ee(o["hand"])[:3])
        frames.clear()
        trace.clear()
        cur.update(op=op, n=0)
        steps = sum(E.act(T, d, speed=sp) for d, sp in edited) + E.hold_window(T, o["window"])
        cur["op"] = None
        ev = G.evaluate(o, start, list(trace), G.object_state(T), region)
        color = (31, 138, 76) if ev["S"] == 1 else (192, 57, 43)
        H, W = frames[0].shape[:2]
        path = out / f"{j['label']}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + 26}", "-r", "15",
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast", "-movflags", "+faststart", str(path)],
                              stdin=subprocess.PIPE)
        for i, fr in enumerate(frames):
            ff.stdin.write(bar(fr, f"{j['label']} · {op} · cond {j['cond']} {CONDITIONS[j['cond']]} · start {sid} · step {2 * (i + 1)}", (55, 65, 81)).tobytes())
        extra = f" · lid {ev['joint_end']:.3f} / {ev['joint_threshold']:.3f} rad" if "joint_end" in ev else ""
        for _ in range(15):
            ff.stdin.write(bar(frames[-1], f"{j['label']} · END: {'SUCCESS' if ev['S'] == 1 else 'NO SUCCESS'} (J {ev['J']}){extra} after {steps} steps", color).tobytes())
        ff.stdin.close()
        ff.wait()
        Image.fromarray(bar(frames[-1], f"{j['label']} · END: {'SUCCESS' if ev['S'] == 1 else 'NO SUCCESS'} (J {ev['J']}){extra}", color)).save(out / f"{j['label']}.jpg", quality=85)
        results.append(dict(j, re_executed=ev, steps=steps, video=str(path)))
        print("VIDEO", j["label"], ev["S"], ev["J"], flush=True)
    (out / "videos.json").write_text(json.dumps(results, indent=1, default=str))
    print("RENDER DONE", flush=True)
    T.close()


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:                                    # Isaac's app loop would keep a crashed process alive
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
