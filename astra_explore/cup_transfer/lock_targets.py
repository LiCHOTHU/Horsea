"""Locked target objects for the final transfer test of the bottleneck plan (2026-10-13). Built once, hashed, never loaded by
any development run. Pre-registered rule: two hollow cups built exactly like cups A and B (bottom disk + 24 wall slabs,
0.35 kg, simulator default friction), with diameter and height scales of cup A drawn uniformly from [0.85, 1.20] x
[0.85, 1.15] by numpy.random.default_rng(20261013), in draw order.

    PYTHONPATH=<omni.usd.libs> LD_LIBRARY_PATH=<omni.usd.libs>/bin python lock_targets.py --out <dir>
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from assets import D_REF, H_REF, MASS, BOTTOM, WALL, cup_props, write_usd

SEED, D_RANGE, H_RANGE = 20261013, (0.85, 1.20), (0.85, 1.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "locked_targets.json").exists():
        raise SystemExit("already locked: " + str(out / "locked_targets.json"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rec = {"rule": __doc__.split("never loaded by\nany development run. ")[1].split("\n\n")[0].replace("\n", " "), "seed": SEED,
           "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "LOCKED - do not load before the final transfer test", "objects": {}}
    for name, color in (("cup_t1", (0.80, 0.55, 0.85)), ("cup_t2", (0.55, 0.85, 0.60))):
        sd, sh = float(rng.uniform(*D_RANGE)), float(rng.uniform(*H_RANGE))
        R, H = D_REF / 2 * sd, H_REF * sh
        props = cup_props(R, H, MASS)
        path = out / f"{name}.usd"
        write_usd(path, "cup", R, H, props, color)
        rec["objects"][name] = {"diameter_scale": round(sd, 4), "height_scale": round(sh, 4), "diameter_m": round(2 * R, 4), "height_m": round(H, 4),
                                "mass_kg": round(props["mass"], 4), "com_m": [round(v, 5) for v in props["com"]],
                                "inertia_kgm2": [[round(v, 7) for v in row] for row in props["inertia"]], "wall_m": WALL, "bottom_m": BOTTOM,
                                "usd": str(path), "usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "locked_targets.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("diameter_scale", "height_scale", "diameter_m", "height_m")} for k, v in rec["objects"].items()}))


if __name__ == "__main__":
    main()
