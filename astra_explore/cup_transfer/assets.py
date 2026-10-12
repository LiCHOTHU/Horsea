"""Rigid test objects for the cup-transfer experiment (USD, metres, Z up, origin at the centre of the base).

    python assets.py --out <dir> [--scale-b 1.15 0.90] [--scale-c 0.90 1.10]

Reference = the benchmark's graspable can (data/can/can_fanta.usd): diameter 0.0637 m, height 0.1218 m, mass 0.35 kg,
no physics material bound (all objects therefore share the simulator's default material, i.e. equal friction).
Objects:
  cup_a   hollow handleless cup, D_A x H_A (the reference dimensions)
  cup_b   hollow cup, 1.15 D_A x 0.90 H_A (initial proposal; frozen after the stage-0 feasibility check)
  cyl_c   closed solid cylinder, 0.90 D_A x 1.10 H_A
  cup_bp  cup_b geometry, same total mass, part of it as a hidden ballast off the axis (centre of mass moved; follow-up)
Geometry is built from primitive colliders on one rigid body (the visual is the same primitives, so visual and collision
geometry agree): a cup = bottom disk (cylinder) + 24 wall slabs; the cylinder = one closed cylinder. Mass, centre of mass and
the inertia tensor are computed analytically for uniform density (plus the ballast for cup_bp) and written explicitly into
the body's MassAPI, so the simulator does not have to infer them. Needs pxr (Isaac's omni.usd.libs or usd-core).
"""
import argparse
import json
import math
from pathlib import Path

D_REF, H_REF, MASS = 0.0637, 0.1218, 0.35
WALL, BOTTOM, N_SLABS = 0.004, 0.006, 24


def cup_props(R, H, mass, wall=WALL, bottom=BOTTOM, ballast=None):
    """Uniform-density cup (annular wall + bottom disk); optional ballast = dict(mass, x, z, r, h) solid cylinder inside."""
    ri, hw = R - wall, H - bottom
    Vw, Vb = math.pi * (R * R - ri * ri) * hw, math.pi * R * R * bottom
    shell_mass = mass - (ballast["mass"] if ballast else 0.0)
    rho = shell_mass / (Vw + Vb)
    mw, mb = rho * Vw, rho * Vb
    parts = [  # (m, (x, y, z), Ixx, Iyy, Izz) about the part's own centre, axes aligned
        (mw, (0.0, 0.0, bottom + hw / 2), mw * (3 * (R * R + ri * ri) + hw * hw) / 12, mw * (3 * (R * R + ri * ri) + hw * hw) / 12, mw * (R * R + ri * ri) / 2),
        (mb, (0.0, 0.0, bottom / 2), mb * (3 * R * R + bottom * bottom) / 12, mb * (3 * R * R + bottom * bottom) / 12, mb * R * R / 2)]
    if ballast:
        m, r, h = ballast["mass"], ballast["r"], ballast["h"]
        parts.append((m, (ballast["x"], 0.0, ballast["z"]), m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2))
    return combine(parts)


def cylinder_props(R, H, mass):
    return combine([(mass, (0.0, 0.0, H / 2), mass * (3 * R * R + H * H) / 12, mass * (3 * R * R + H * H) / 12, mass * R * R / 2)])


def combine(parts):
    M = sum(p[0] for p in parts)
    c = [sum(p[0] * p[1][i] for p in parts) / M for i in range(3)]
    I = [[0.0] * 3 for _ in range(3)]
    for m, x, ixx, iyy, izz in parts:
        d = [x[i] - c[i] for i in range(3)]
        own = [[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]]
        dd = sum(v * v for v in d)
        for i in range(3):
            for j in range(3):
                I[i][j] += own[i][j] + m * ((dd if i == j else 0.0) - d[i] * d[j])
    return {"mass": M, "com": c, "inertia": I}


def principal(I):
    """Eigen-decomposition of a symmetric 3x3 inertia (numpy) -> diagonal + quaternion (w, x, y, z) of the principal axes."""
    import numpy as np
    w, V = np.linalg.eigh(np.array(I))
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    t = np.trace(V)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (V[2, 1] - V[1, 2]) / s, (V[0, 2] - V[2, 0]) / s, (V[1, 0] - V[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(V)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + V[i, i] - V[j, j] - V[k, k]) * 2
        q = [0.0] * 4
        q[0] = (V[k, j] - V[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (V[j, i] + V[i, j]) / s
        q[1 + k] = (V[k, i] + V[i, k]) / s
    return [float(x) for x in w], [float(x) for x in q]


def write_usd(path, kind, R, H, props, color):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root")
    st.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(st, "/Root/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    m = UsdPhysics.MassAPI.Apply(body.GetPrim())
    diag, q = principal(props["inertia"])
    m.CreateMassAttr(float(props["mass"]))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in props["com"]]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
    m.CreatePrincipalAxesAttr(Gf.Quatf(q[0], q[1], q[2], q[3]))

    def solid(prim, rgb):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    if kind == "cylinder":
        c = UsdGeom.Cylinder.Define(st, "/Root/body/shape")
        c.CreateRadiusAttr(R)
        c.CreateHeightAttr(H)
        c.CreateAxisAttr("Z")
        c.AddTranslateOp().Set(Gf.Vec3d(0, 0, H / 2))
        solid(c.GetPrim(), color)
    else:
        b = UsdGeom.Cylinder.Define(st, "/Root/body/bottom")
        b.CreateRadiusAttr(R)
        b.CreateHeightAttr(BOTTOM)
        b.CreateAxisAttr("Z")
        b.AddTranslateOp().Set(Gf.Vec3d(0, 0, BOTTOM / 2))
        solid(b.GetPrim(), color)
        rm, hw = R - WALL / 2, H - BOTTOM
        chord = 2 * rm * math.tan(math.pi / N_SLABS) * 1.04            # slight overlap closes the gaps between slabs
        for k in range(N_SLABS):
            th = 360.0 * k / N_SLABS
            s = UsdGeom.Cube.Define(st, f"/Root/body/wall_{k:02d}")
            s.CreateSizeAttr(1.0)
            s.AddTranslateOp().Set(Gf.Vec3d(rm * math.cos(math.radians(th)), rm * math.sin(math.radians(th)), BOTTOM + hw / 2))
            s.AddRotateZOp().Set(th)
            s.AddScaleOp().Set(Gf.Vec3f(WALL, chord, hw))
            solid(s.GetPrim(), color)
    st.GetRootLayer().Save()


def build(out, scale_b=(1.15, 0.90), scale_c=(0.90, 1.10)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R0 = D_REF / 2
    specs = {}
    dims = {"cup_a": ("cup", R0, H_REF), "cup_b": ("cup", R0 * scale_b[0], H_REF * scale_b[1]),
            "cyl_c": ("cylinder", R0 * scale_c[0], H_REF * scale_c[1]), "cup_bp": ("cup", R0 * scale_b[0], H_REF * scale_b[1])}
    colors = {"cup_a": (0.90, 0.90, 0.95), "cup_b": (0.55, 0.75, 0.95), "cyl_c": (0.95, 0.65, 0.30), "cup_bp": (0.55, 0.75, 0.95)}
    for name, (kind, R, H) in dims.items():
        if kind == "cylinder":
            props = cylinder_props(R, H, MASS)
        elif name == "cup_bp":       # same visible cup as B; 0.12 kg of the 0.35 kg sits in a hidden ballast near the wall
            props = cup_props(R, H, MASS, ballast={"mass": 0.12, "x": 0.55 * R, "z": BOTTOM + 0.012, "r": 0.010, "h": 0.020})
        else:
            props = cup_props(R, H, MASS)
        path = out / f"{name}.usd"
        write_usd(path, kind, R, H, props, colors[name])
        specs[name] = {"kind": kind, "diameter_m": round(2 * R, 4), "height_m": round(H, 4), "mass_kg": round(props["mass"], 4),
                       "com_m": [round(v, 5) for v in props["com"]], "inertia_kgm2": [[round(v, 7) for v in row] for row in props["inertia"]],
                       "wall_m": WALL if kind == "cup" else None, "bottom_m": BOTTOM if kind == "cup" else None,
                       "friction": "simulator default material (no binding), identical for all objects and the reference can",
                       "usd": str(path)}
    specs["_reference"] = {"source": "data/can/can_fanta.usd", "diameter_m": D_REF, "height_m": H_REF, "mass_kg": MASS}
    (out / "asset_specs.json").write_text(json.dumps(specs, indent=1))
    return specs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-b", type=float, nargs=2, default=(1.15, 0.90))
    ap.add_argument("--scale-c", type=float, nargs=2, default=(0.90, 1.10))
    a = ap.parse_args()
    print(json.dumps(build(a.out, a.scale_b, a.scale_c), indent=1))
