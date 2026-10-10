"""Representative paired videos (rendering on; run after the formal blocks, never concurrently with them).

    python -m s1local.render --out <run dir> --jobs <jobs.json>

jobs.json: [{"op", "cond", "start", "theta", "label"}]. Each job rebuilds its start the same way the runner did (reference
prefix from the reset scene, can offset or door offset for fresh starts), applies the condition, executes the edited
program and writes videos/<label>.mp4 (main camera, every 2nd control step, top bar: operation, arm, condition, step).
Restores are not bit-exact (PhysX contact caches), so a re-executed rollout can differ slightly from the logged one;
each video states its own re-executed outcome.
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
from s1local import ops as O  # noqa: E402
from s1local.runner import CAN_OFFSETS, CONDITIONS, DOOR_OFFSETS  # noqa: E402

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
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args()
    out = Path(a.out) / "videos"
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.jobs).read_text())
    T = E.build(render=True)
    phys = E.Physics(T)
    prog = E.load_program(10)
    frames = []
    step0 = T._step
    k = {"n": 0}

    def _step(act):
        o = step0(act)
        k["n"] += 1
        if k["n"] % 2 == 0:
            frames.append(T._image("main_camera", (640, 360)))
        return o
    T._step = _step
    snaps = {}
    results = []
    for j in jobs:
        op, sid = j["op"], j["start"]
        key = (op, "nominal" if op == "door" else sid)
        if key not in snaps:
            phys.nominal()
            E.reset_with_can_offset(T, CAN_OFFSETS["nominal" if op == "door" else sid])
            for i in O.OPS[op]["prefix"]:
                E.act(T, prog[i])
            snaps[key] = E.snapshot(T)
        E.restore(T, snaps[key])
        if op == "door" and DOOR_OFFSETS.get(sid):
            E.set_door(T, DOOR_OFFSETS[sid])
        phys.apply(op, *CONDITIONS[j["cond"]])
        start = E.object_state(T)
        edited = O.edited_program(op, prog, np.asarray(j["theta"], float), left_wrist0=T._ee("left")[:3])
        frames.clear()
        trace = []
        rec0 = T._step

        def _rec(act):
            o = rec0(act)
            U = T.U
            trace.append([U.can.data.root_state_w[0, :3].cpu().numpy().copy(), float(U.drawer.data.joint_pos[0, U.drawer_bottom_idx].item()),
                          T._tips("right").mean(0)])
            return o
        T._step = _rec
        steps = sum(E.act(T, d, speed=s) for d, s in edited) + E.hold_window(T, O.OPS[op]["settle"])
        T._step = rec0
        ev = O.evaluate(op, start, trace, E.object_state(T))
        color = (31, 138, 76) if ev["S"] == 1 else (192, 57, 43)
        H, W = frames[0].shape[:2]
        path = out / f"{j['label']}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + 26}", "-r", "15",
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast", "-movflags", "+faststart", str(path)],
                              stdin=subprocess.PIPE)
        for i, fr in enumerate(frames):
            ff.stdin.write(bar(fr, f"{j['label']} · {op} · cond {j['cond']} {CONDITIONS[j['cond']]} · start {sid} · step {2 * (i + 1)}", (55, 65, 81)).tobytes())
        for _ in range(15):
            ff.stdin.write(bar(frames[-1], f"{j['label']} · END: {'SUCCESS' if ev['S'] == 1 else 'NO SUCCESS'} (J {ev['J']}) after {steps} steps", color).tobytes())
        ff.stdin.close()
        ff.wait()
        Image.fromarray(bar(frames[-1], f"{j['label']} · END: {'SUCCESS' if ev['S'] == 1 else 'NO SUCCESS'} (J {ev['J']})", color)).save(out / f"{j['label']}.jpg", quality=85)
        results.append(dict(j, re_executed=ev, steps=steps, video=str(path)))
        print("VIDEO", j["label"], ev["S"], ev["J"], flush=True)
    (out / "videos.json").write_text(json.dumps(results, indent=1, default=str))
    print("RENDER DONE", flush=True)
    T.close()


if __name__ == "__main__":
    main()
