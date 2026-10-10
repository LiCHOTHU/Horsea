"""Episode media for the transfer-study pages: an MP4 and a filmstrip per recorded episode (idempotent).

    python make_episode_media.py --root experiments/s1s2_overnight_2026-10-10

Reads each episode's transitions.npz (the frames the student-rate recorder stored: RoboTwin head / left / right cameras,
dex main camera) and writes, next to it, video.mp4 (H.264, top bar: episode label, step, controller) and strip.jpg (eight
evenly spaced frames with their step numbers). LIBERO episodes already carry the harness's agentview.mp4. Nothing in the
study's records is modified.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ASTRA, STUDENT = (124, 58, 237), (13, 148, 136)


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def frames_of(d, kind):
    """(T, H, W, 3) uint8 frames shown in the video."""
    imgs = d["imgs"]
    if kind == "robotwin":                       # (T, 3, 120, 160, 3): head, left, right
        return np.concatenate([imgs[:, 0], imgs[:, 1], imgs[:, 2]], axis=2)
    return imgs                                  # dex: (T, 224, 224, 3)


def render(frame, scale, bar, text, color):
    im = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.BILINEAR)
    canvas = Image.new("RGB", (im.width, im.height + bar), color)
    canvas.paste(im, (0, bar))
    ImageDraw.Draw(canvas).text((8, 5), text, fill=(255, 255, 255), font=font(15))
    return np.asarray(canvas)


def make(epdir, kind, force=False):
    epdir = Path(epdir)
    npz, meta_p = epdir / "transitions.npz", epdir / "episode.json"
    out_v, out_s, out_p = epdir / "video.mp4", epdir / "strip.jpg", epdir / "poster.jpg"
    if not npz.exists() or not meta_p.exists():
        return "skip (no record)"
    if not force and out_v.exists() and out_s.exists() and out_p.exists() and out_v.stat().st_mtime > npz.stat().st_mtime:
        return "up to date"
    meta = json.loads(meta_p.read_text())
    d = np.load(npz)
    F = frames_of(d, kind)
    src = d["source"]
    T = len(F)
    if T == 0:
        return "skip (no frames)"
    label = f"{meta.get('cell')} · start {meta.get('start_id')}"
    scale, fps = (2, 15) if kind == "robotwin" else (2, 30)
    cams = " · head | left wrist | right wrist" if kind == "robotwin" else " · main camera"
    first = render(F[0], scale, 26, "", ASTRA)
    H, W = first.shape[:2]
    tmp = out_v.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast",
                           "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    for t in range(T):
        who = "ASTRA" if src[t] == 0 else "STUDENT"
        ff.stdin.write(render(F[t], scale, 26, f"{label} · step {t + 1}/{T} · {who} in control{cams}", ASTRA if src[t] == 0 else STUDENT).tobytes())
    ok = bool(meta.get("success"))
    end = render(F[-1], scale, 26, f"{label} · END: {'SUCCESS' if ok else 'NO SUCCESS'} at step {T}", (31, 138, 76) if ok else (192, 57, 43))
    Image.fromarray(end).save(out_p, quality=85)          # poster: the final frame with the outcome bar
    for _ in range(fps):
        ff.stdin.write(end.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        return "FAILED (ffmpeg)"
    tmp.replace(out_v)
    idx = np.linspace(0, T - 1, 8).round().astype(int)
    tiles = []
    for i in idx:
        fr = F[i][:, :160] if kind == "robotwin" else F[i]
        im = Image.fromarray(fr).resize((160, 120) if kind == "robotwin" else (140, 140), Image.BILINEAR)
        tile = Image.new("RGB", (im.width, im.height + 16), ASTRA if src[i] == 0 else STUDENT)
        tile.paste(im, (0, 16))
        ImageDraw.Draw(tile).text((3, 1), f"step {i + 1}", fill=(255, 255, 255), font=font(11))
        tiles.append(tile)
    strip = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + 4
    strip.save(out_s, quality=85)
    return f"made ({T} frames)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    n = 0
    for plat, kind in (("robotwin_single", "robotwin"), ("robotwin_bimanual", "robotwin"), ("dexhand", "dex")):
        for ep in sorted((root / plat / "episodes").glob("*")):
            if ep.is_dir():
                r = make(ep, kind, a.force)
                n += r.startswith("made")
                print(plat, ep.name, r, flush=True)
    print("videos made:", n)


if __name__ == "__main__":
    main()
