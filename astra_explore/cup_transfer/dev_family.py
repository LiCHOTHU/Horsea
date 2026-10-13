"""Development cup family for batch 2's cross-object tests (2026-10-13). Rule fixed before any of its results: hollow cups built
exactly like cups A and B, at the four corners of the size range, diameter x {0.90, 1.20} and height x {0.88, 1.12} of cup A.
Separate from the locked test cups (cup_locked_targets_2026-10-13), which no development run loads.

    PYTHONPATH=<omni.usd.libs> LD_LIBRARY_PATH=<omni.usd.libs>/bin python dev_family.py --out <assets dir>
writes <name>.usd and asset_specs_dev.json next to the cup-transfer asset specs.
"""
import argparse
import json
from pathlib import Path

from assets import BOTTOM, D_REF, H_REF, MASS, WALL, cup_props, write_usd

FAMILY = {"dev_f1": (0.90, 0.88), "dev_f2": (0.90, 1.12), "dev_f3": (1.20, 0.88), "dev_f4": (1.20, 1.12),
          # second round (2026-10-13 06:10, rule fixed before any of its results): the midpoints of the rectangle's four edges
          "dev_g1": (1.05, 0.88), "dev_g2": (1.05, 1.12), "dev_g3": (0.90, 1.00), "dev_g4": (1.20, 1.00)}
COLORS = {"dev_f1": (0.95, 0.85, 0.40), "dev_f2": (0.60, 0.90, 0.90), "dev_f3": (0.90, 0.60, 0.60), "dev_f4": (0.70, 0.70, 0.95),
          "dev_g1": (0.85, 0.75, 0.55), "dev_g2": (0.55, 0.80, 0.70), "dev_g3": (0.80, 0.55, 0.80), "dev_g4": (0.65, 0.85, 0.50)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    specs = {}
    for name, (sd, sh) in FAMILY.items():
        R, H = D_REF / 2 * sd, H_REF * sh
        props = cup_props(R, H, MASS)
        path = out / f"{name}.usd"
        write_usd(path, "cup", R, H, props, COLORS[name])
        specs[name] = {"kind": "cup", "diameter_m": round(2 * R, 4), "height_m": round(H, 4), "mass_kg": round(props["mass"], 4),
                       "com_m": [round(v, 5) for v in props["com"]], "inertia_kgm2": [[round(v, 7) for v in row] for row in props["inertia"]],
                       "wall_m": WALL, "bottom_m": BOTTOM, "friction": "simulator default material (no binding)", "usd": str(path),
                       "family_rule": "corners (f) and edge midpoints (g) of diameter x [0.90, 1.20], height x [0.88, 1.12] of cup A", "scales": [sd, sh]}
    (out / "asset_specs_dev.json").write_text(json.dumps(specs, indent=1))
    print(json.dumps({k: [v["diameter_m"], v["height_m"]] for k, v in specs.items()}))


if __name__ == "__main__":
    main()
