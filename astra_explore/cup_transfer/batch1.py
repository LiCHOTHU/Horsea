"""First diagnostic batch of the "bottlenecks and reuse" plan (2026-10-13). Isaac, conda env hongyi, cwd egovla_release/egovla,
PYTHONPATH=astra_explore. Every process writes a run manifest first (manifest_<job>.json).

    python -m cup_transfer.batch1 live --obj cup_b --start t2 --reps 2         Phase 0: instrumented live attempts of the
                                                                               current system (Astra + frozen System 1 with D_A)
    python -m cup_transfer.batch1 geometry --obj cup_b [--variants original,shift] [--starts d01,...] [--reps 2]
                                                                               Phase 1: object frame and size, action channels,
                                                                               replay of cup A's known-good grasp per start
    python -m cup_transfer.batch1 branches --obj cup_b                         Phases 2-3: six diagnostic states; System 1's full
                                                                               candidate pipeline; every candidate executed from the
                                                                               restored state (2 repeats) + a fixed continuation
Output root: experiments/cup_batch1_2026-10-13/. Nothing here writes to the locked memory D_A (branches never add records).
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_env import GROUPS, HANDS, IsaacDexTask, joint_group, quat_angle_deg, quat_slerp  # noqa: E402
from cup_transfer import system1 as S1  # noqa: E402
from cup_transfer import run_cup as RC  # noqa: E402

OUT = Path("/home/licho/workspace/Horsea/experiments/cup_batch1_2026-10-13")
SRC = RC.ROOT / "source_interaction_dataset" / "run"
CUPB = RC.ROOT / "per_attempt_transition_logs" / "run" / "cup_b"
AE = Path(__file__).resolve().parents[1]
DEV_STARTS = {"d01": (0.0, 0.0), "d02": (0.010, -0.010), "d03": (-0.015, 0.005), "d04": (0.015, 0.010), "d05": (-0.010, -0.015),
              "d06": (0.020, 0.0), "d07": (-0.020, 0.0), "d08": (0.0, 0.020), "d09": (0.0, -0.020), "d10": (0.010, 0.015)}
# development correction from the four upright cup B grasps that held in the cup-transfer run (achieved wrist minus cup A's
# grasp, per axis, rounded): evidence from development data, not a general rule
SHIFT_B = np.array([-0.005, 0.005, -0.003])
# geometric correction from the measured mechanism (cup B's thumb ends above its lower rim): keep the grasp height relative to
# the rim instead of the base, i.e. move every wrist target by (object height - cup A height) along z
H_SOURCE = 0.1218


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def wj(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------- manifest
def manifest(T, job, extra):
    U = T.U
    cfg = U.cfg
    try:
        import isaaclab
        il = getattr(isaaclab, "__version__", None)
    except Exception:  # noqa: BLE001
        il = None
    try:
        from importlib.metadata import version
        isim = version("isaacsim")
    except Exception:  # noqa: BLE001
        isim = None
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=AE).stdout.strip()
    dt, dec = float(cfg.sim.dt), int(cfg.decimation)
    hyper = json.loads((RC.ROOT / "method_configurations" / "hyper.json").read_text())["best"]
    m = {
        "job": job, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
        "environment": {"env_id": "Humanoid-Cup-v0", "isaacsim": isim, "isaaclab": il, "scene_index": T.scene_index, "room": T.room, "table": T.table,
                        "object": T.obj, "object_usd": T.spec["usd"], "object_usd_sha16": sha(T.spec["usd"]), "object_spec": T.spec,
                        "tray_xy": list(cfg.tray_xy), "object_xy": list(cfg.object_xy), "task_mode": cfg.task_mode},
        "robot": {"hand_joints": T.hand_names, "hand_joint_group": {s: [joint_group(n) for n in T.hand_names[s]] for s in HANDS},
                  "hand_open_q": {s: [round(float(T.default_q[i]), 4) for i in T.hand_ids[s]] for s in HANDS},
                  "hand_closed_q": {s: [round(float(T.closed_q[i]), 4) for i in T.hand_ids[s]] for s in HANDS},
                  "wrist_body": {s: T.robot.body_names[T.ee_idx[s]] for s in HANDS}, "fingertip_bodies": T.tip_names,
                  "ik": {"left": "DifferentialIK dls (pose, absolute)", "right": "DifferentialIK pinv (pose, absolute)"},
                  "calibration": T.calibration},
        "timing": {"physics_dt_s": dt, "decimation": dec, "control_hz": round(1.0 / (dt * dec), 3), "render_interval": int(cfg.sim.render_interval),
                   "observation": "after every Astra decision (images + proprio); System 1 state before / after every segment",
                   "memory_update": "after every executed segment, for updating methods only (never in batch-1 runs)"},
        "actions": {"wrist": "absolute wrist (hand-base link) target in the world frame [m] + fingers / palm directions, IK-tracked along a straight path",
                    "wrist_limits_per_control_step": {"lin_m": IsaacDexTask.MAX_LIN_M, "ang_deg": IsaacDexTask.MAX_ANG_DEG},
                    "closure": "four fractions per hand in [0, 1] (finger_base, finger_tip, thumb_rotate, thumb_bend) mapped linearly from open to the far joint limit",
                    "closure_ramp_per_control_step": IsaacDexTask.MAX_CLOSURE_STEP, "executor": dict(sys.modules["cup_transfer.cup_task"].EXEC)},
        "system1": {"segment_control_steps": S1.SEG_STEPS, "horizon_segments": 3,
                    "horizon_physical_s": round(3 * S1.SEG_STEPS * dt * dec, 4),
                    "H_meaning": "3 model transitions = 3 executed segments of 3 control steps = 9 control steps; only the first segment is executed, then replanned",
                    "n_candidates": 32, "alpha": 0.5, "beta": 0.2, "bound": S1.BOUND.tolist(), "sig_s": S1.SIG_S.tolist(), "sig_u": S1.SIG_U.tolist(),
                    "ess_min": S1.ESS_MIN, "k_neighbors": 80, "h": hyper["h"], "lam": hyper["lam"],
                    "action_input": "u_applied = the requested segment command after the executor's own limits (translation cap, closure ramp in command space), computed before execution from the previous command - not the achieved motion",
                    "state": S1.STATE_NAMES, "delta": S1.DELTA_NAMES, "objective_source_sha16": sha(AE / "cup_transfer" / "system1.py"),
                    "fallback": ["mode 'other' -> Astra's command", "nominal command without local support (ESS < 8) -> Astra's command",
                                 "no supported candidate -> Astra's command"]},
        "astra": {"model": "gpt-6-astra", "reasoning": "medium", "timeout_s": 180, "prompt_builder_sha16": sha(AE / "cup_transfer" / "prompt_cup.py"),
                  "schema_sha16": sha(AE / "schema_dex.py"), "history": "Astra's own interaction history; target runs start from the source run's history",
                  "model_identity": "not exposed in the Codex JSON events (recorded per call as model_identity_in_events)"},
        "memory": {"D_A": str(SRC / "D_A.jsonl"), "D_A_sha16": sha(SRC / "D_A.jsonl"), "records": len(jl(SRC / "D_A.jsonl")), "frozen": True},
        "evaluation": {"success": "object base >= rest + 5 cm and speed <= 0.10 m/s for 30 consecutive control steps (env_cups.CupEnv)",
                       "env_cups_sha16": sha(AE / "cup_transfer" / "env_cups.py"), "lift_height": cfg.lift_height, "hold_steps": cfg.hold_steps,
                       "horizon_steps": T.horizon, "calls_per_attempt": RC.CALLS_PER_ATTEMPT},
        "reproducibility": {"git_head": git("rev-parse", "HEAD"), "git_dirty_files": git("status", "--short", "--", ".").splitlines(),
                            "git_diff_sha16": hashlib.sha256(git("diff", "--", ".").encode()).hexdigest()[:16], "python": sys.version.split()[0],
                            "argv": sys.argv},
    }
    m.update(extra)
    wj(OUT / ("smoke" if "smoke" in str(extra.get("tag", "")) else "") / f"manifest_{job}.json", m)
    return m


# ---------------------------------------------------------------------------- snapshot / restore for the cup task
def snap(T):
    from s1local import env as E1
    s = E1.snapshot(T)
    U = T.U
    s["cup"] = {"hold_count": U.hold_count.clone(), "success": U.success.clone(), "lift_ref_z": U.lift_ref_z, "rest_z": T.rest_z,
                "trace_len": len(T.trace)}
    return s


def restore(T, s):
    from s1local import env as E1
    E1.restore(T, s)
    U = T.U
    U.hold_count[:] = s["cup"]["hold_count"]
    U.success[:] = s["cup"]["success"]
    U.lift_ref_z, T.rest_z = s["cup"]["lift_ref_z"], s["cup"]["rest_z"]
    del T.trace[s["cup"]["trace_len"]:]


def lean(T):
    """No video frames and no image observations during scripted / branch execution (state only)."""
    T.video_every = 10 ** 9
    T._observation_full = T._observation
    T._observation = lambda: {"task_complete": bool(T.U.success[0].item()), "proprio": {s: {"closure_measured": T._closure_measured(s),
                                                                                            "index_contact_force_N": round(T._contact(s), 2)} for s in HANDS},
                              "steps_used": T.steps_used, "steps_remaining": T.horizon - T.steps_used}


def lean_reset(T, start_xy):
    """Reset with the object at a start offset (as CupTask.reset, without images)."""
    T.set_start(start_xy)
    T.trace = []
    T._reset_env()
    T.steps_used, T._frames, T.open = 0, [], True
    T.hand_cmd = {s: (0.0, 0.0, 0.0, 0.0) for s in HANDS}
    hold = T.robot.data.joint_pos.clone()
    T._run(T.settle_steps, hands={s: (0.0, 0.0, 0.0, 0.0) for s in HANDS}, hold_q=hold)
    T.rest_z = float(T.obj_state()["pos"][2])
    T.U.lift_ref_z = T.rest_z
    T.trace = []


# ---------------------------------------------------------------------------- geometry helpers (evaluation only)
def tip_geometry(T, side="right"):
    """Per fingertip body: radial distance to the object's axis minus its radius (cm; < 0 = inside the outer wall), height
    above the object base (cm); palm / finger directions; hand joints measured vs commanded target."""
    st = T.obj_state()
    o, up = st["pos"], st["up"]
    tips = T._tips(side)
    out = {}
    for name, p in zip(T.tip_names[side], tips):
        v = p - o
        h = float(np.dot(v, up))
        radial = float(np.linalg.norm(v - h * up))
        out[name] = {"surface_gap_cm": round(100 * (radial - T.R), 2), "height_cm": round(100 * h, 2), "vec_cm": (100 * v).round(3).tolist()}
    fd, pd, _ = T._dirs_world(side)
    q = T._hand_q(side)
    tgt = T.robot.data.joint_pos_target[0, T.hand_ids[side]].cpu().numpy()
    return {"tips": out, "fingers_dir": fd.round(3).tolist(), "palm_dir": pd.round(3).tolist(),
            "hand_q_minus_target": (q - tgt).round(3).tolist(), "contact_N": round(T._contact(side), 2),
            "object_rise_cm": round(100 * (float(o[2]) - T.rest_z), 2), "object_tilt_deg": round(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))), 2),
            "wrist_m": T._ee(side)[:3].round(4).tolist(), "object_m": o.round(4).tolist()}


def source_program():
    """Cup A's known-good program: Astra's executed decisions of source attempt 1 and the cup position they refer to."""
    dec = []
    for d in sorted((SRC / "attempt_1").glob("decision_*")):
        r = json.loads((d / "decision.json").read_text())
        c = r.get("decision_content")
        if c and not c.get("stop") and r.get("executed"):
            dec.append(c)
    obj0 = np.array(jl(SRC / "attempt_1" / "eval_trace.jsonl")[0]["obj"])
    return dec, obj0


def shifted(dec, delta):
    import copy
    d = copy.deepcopy(dec)
    for s in HANDS:
        if d[s]["move"]:
            d[s]["position_m"] = [float(a + b) for a, b in zip(d[s]["position_m"], delta)]
    return d


# ---------------------------------------------------------------------------- Phase 0: live attempts
def camera_check(T, out):
    """Projection of known 3D points into the main and right-hand cameras vs the main camera's depth image."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    st = T.obj_state()
    o = st["pos"]
    pts = {"object_base_centre": o, "object_top_centre": o + np.array([0, 0, T.H]),
           "object_side_facing_main_camera": None, "right_fingertips_mean": T._tips("right").mean(0)}
    cam = T.cams["main_camera"].data
    cpos = cam.pos_w[0].cpu().numpy()
    v = (cpos - o)[:2]
    pts["object_side_facing_main_camera"] = o + np.array([*(T.R * v / np.linalg.norm(v)), T.H / 2])
    for cname in ("main_camera", "right_hand_camera"):
        d = T.cams[cname].data
        h, w = d.image_shape
        img = Image.fromarray(T._image(cname, (w, h)))
        dr = ImageDraw.Draw(img)
        depth = d.output.get("distance_to_image_plane")
        rows = {}
        for k, P in pts.items():
            uv = T._project(cname, P, (w, h))
            r = {"uv": None if uv is None else [round(uv[0], 1), round(uv[1], 1)]}
            if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
                dr.ellipse([uv[0] - 6, uv[1] - 6, uv[0] + 6, uv[1] + 6], outline=(255, 0, 0), width=3)
                from isaac_env import mat_from_quat
                pc = mat_from_quat(d.quat_w_world[0].cpu().numpy()).T @ (np.asarray(P) - d.pos_w[0].cpu().numpy())
                r["camera_forward_distance_m"] = round(float(pc[0]), 4)
                if depth is not None:
                    dv = float(depth[0, int(uv[1]), int(uv[0])].reshape(-1)[0].item())
                    r["depth_image_m"] = round(dv, 4)
                    r["depth_minus_forward_cm"] = round(100 * (dv - float(pc[0])), 2)
            rows[k] = r
        img.save(out / f"camera_check_{cname}.png")
        rep[cname] = {"image_wh": [w, h], "intrinsics": d.intrinsic_matrices[0].cpu().numpy().round(3).tolist(),
                      "pos_w": d.pos_w[0].cpu().numpy().round(4).tolist(), "quat_w_world": d.quat_w_world[0].cpu().numpy().round(4).tolist(),
                      "has_depth": depth is not None, "points": rows}
    wj(out / "camera_check.json", rep)
    return rep


def cmd_live(a):
    import cup_transfer.cup_task  # noqa: F401
    out = OUT / ("smoke" if a.policy == "smoke" else "") / "phase0_live"
    T = RC.make_task(a.obj, render=True)
    man = manifest(T, f"live_{a.obj}_{a.start}", {"stage": "phase0_live", "arm": a.arm, "start": a.start, "reps": a.reps, "tag": "smoke" if a.policy == "smoke" else ""})
    T.set_start(RC.STARTS[a.start])
    T.reset()
    camera_check(T, out / f"{a.obj}_{a.start}")
    policy, model = RC.make_policy(argparse.Namespace(policy=a.policy, model="gpt-6-astra", reasoning="medium", timeout=180.0), T)
    D_A = S1.ExperienceStore.load(SRC / "D_A.jsonl")
    hist0 = json.loads((SRC / "history_final.json").read_text())
    hyper = json.loads((RC.ROOT / "method_configurations" / "hyper.json").read_text())["best"]
    results = []
    for rep in range(1, a.reps + 1):
        cdir = out / f"{a.obj}_{a.start}" / f"{a.arm}_rep{rep}"
        if (cdir / "result.json").exists():
            results.append(json.loads((cdir / "result.json").read_text()))
            continue
        if cdir.exists():
            cdir.rename(cdir.parent / f"{cdir.name}_interrupted_{time.strftime('%Y%m%d_%H%M%S')}")
        method, store, update = RC.make_method(a.arm, D_A, hyper, seed=5000 + rep)
        if hasattr(method, "trace"):
            method.trace = True
        history = RC.history_load(hist0)
        cfg = RC.task_cfg(T, "codex", model, {"stage": "phase0_live", "object": a.obj, "start": a.start, "method": a.arm, "rep": rep, "hyper": hyper})
        RC.wjson(cdir / "config.json", cfg)
        budget = RC.Budget(1, RC.CALLS_PER_ATTEMPT, RC.CALLS_PER_ATTEMPT)
        if isinstance(policy, RC.SmokePolicy):
            policy.queue = []
        try:
            s = RC.run_attempt(T, policy, history, budget, cdir / "attempt_1", 1, method, store, update, {"arm": a.arm, "start": a.start}, cfg)
            res = {"status": "complete", "summary": s}
        except Exception as ex:  # noqa: BLE001
            import traceback
            res = {"status": "INFRASTRUCTURE_BLOCKED", "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()}
        RC.wjson(cdir / "result.json", res)
        results.append(res)
        print("LIVE_DONE", a.obj, a.start, rep, res.get("summary", {}).get("outcome", res["status"]), flush=True)
        if res["status"] != "complete":
            break
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- Phase 1: geometry, channels, known-good grasp
def object_frame_check(T):
    """Object frame vs its geometry: the object's LOCAL (untransformed) USD bounding box (visual = collision primitives)
    must span [-R, R] x [-R, R] x [0, H] if the frame is the base centre; the simulated pose is PhysX's (USD transforms are
    not synced back, so a world bounding box would show the spawn pose)."""
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    bb = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    lo, hi = np.array(bb.GetMin()), np.array(bb.GetMax())
    st = T.obj_state()
    return {"local_bbox_min_cm": (100 * lo).round(2).tolist(), "local_bbox_max_cm": (100 * hi).round(2).tolist(), "bbox_size_cm": (100 * (hi - lo)).round(2).tolist(),
            "spec_diameter_height_cm": [round(100 * 2 * T.R, 2), round(100 * T.H, 2)],
            "frame_minus_bbox_base_centre_cm": (-100 * np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])).round(2).tolist(),
            "simulated_frame_pos_m": st["pos"].round(4).tolist(), "up_axis": st["up"].round(4).tolist()}


def channel_checks(T):
    """Free-air checks of every action channel of the right hand (signs, scales, mapping)."""
    from cup_transfer.cup_task import target_quat
    rows = {"translation": [], "rotation": [], "closure": []}
    lean_reset(T, (0.0, 0.0))
    hold = T.robot.data.joint_pos.clone()
    home = T._ee("right").copy()                             # the reset pose: in free air, ~28 cm above the table, away from the object
    for axis, name in enumerate(("x", "y", "z")):
        for sign in (+1, -1):
            T._run(15, targets={"right": home}, hold_q=hold)
            p0 = T._ee("right").copy()
            tgt = p0.copy()
            tgt[axis] += sign * 0.02
            T._run(10, targets={"right": tgt}, hold_q=hold)
            p1 = T._ee("right")
            rows["translation"].append({"command": f"{'+' if sign > 0 else '-'}2 cm {name}", "achieved_cm": (100 * (p1[:3] - p0[:3])).round(3).tolist(),
                                        "error_cm": round(100 * float(np.linalg.norm(p1[:3] - tgt[:3])), 3),
                                        "orientation_drift_deg": round(quat_angle_deg(p0[3:7], p1[3:7]), 2)})
    for axis_name, ax in (("world z", np.array([0, 0, 1.0])), ("world x", np.array([1.0, 0, 0]))):
        for deg in (+10, -10):
            T._run(15, targets={"right": home}, hold_q=hold)
            p0 = T._ee("right").copy()
            th = np.radians(deg) / 2
            qd = np.array([np.cos(th), *(np.sin(th) * ax)])
            w0, x0, y0, z0 = qd
            w1, x1, y1, z1 = p0[3:7]
            q = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                          w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
            tgt = np.concatenate([p0[:3], q])
            T._run(10, targets={"right": tgt}, hold_q=hold)
            p1 = T._ee("right")
            rows["rotation"].append({"command": f"{deg:+d} deg about {axis_name}", "achieved_deg": round(quat_angle_deg(p0[3:7], p1[3:7]), 2),
                                     "error_deg": round(quat_angle_deg(tgt[3:7], p1[3:7]), 2), "position_drift_cm": round(100 * float(np.linalg.norm(p1[:3] - p0[:3])), 3)})
    for gi, g in enumerate(GROUPS):
        for level in (0.5, 1.0):
            T._run(40, targets={"right": home}, hands={"right": (0.0, 0.0, 0.0, 0.0), "left": (0.0, 0.0, 0.0, 0.0)}, hold_q=hold)
            q0 = T._hand_q("right")
            cmd = [0.0] * 4
            cmd[gi] = level
            reach, series = None, []
            for k in range(45):
                T._run(1, targets={"right": home}, hands={"right": tuple(cmd), "left": (0.0, 0.0, 0.0, 0.0)}, hold_q=hold)
                m = T._closure_measured("right")
                series.append(m.get(g, 0.0))
                if reach is None and m.get(g, 0.0) >= 0.95 * level:
                    reach = k + 1
            q1 = T._hand_q("right")
            moved = [T.hand_names["right"][i] for i in range(len(q1)) if abs(q1[i] - q0[i]) > 0.02]
            rows["closure"].append({"group": g, "command": level, "measured_fractions": T._closure_measured("right"), "steps_to_95pct": reach,
                                    "joints_moved": moved, "joint_groups_moved": sorted({joint_group(n) for n in moved})})
    T._run(30, targets={"right": home}, hands={"right": (0.0, 0.0, 0.0, 0.0), "left": (0.0, 0.0, 0.0, 0.0)}, hold_q=hold)
    return rows


def replay_grasp(T, start_xy, variant, tl):
    """Cup A's known-good program re-anchored to this object's base at this start (variant 'shift': plus SHIFT_B), executed by
    the common executor (no System 1, no Astra), then up to 40 hold steps; per-step contact geometry while closing and lifting."""
    from cup_transfer.cup_task import execute_decision
    dec, obj0 = source_program()
    lean_reset(T, start_xy)
    o = T.obj_state()["pos"]
    delta = o - obj0 + (SHIFT_B if variant == "shift" else 0.0) + (np.array([0.0, 0.0, T.H - H_SOURCE]) if variant == "rim" else 0.0)
    seq, out, t0 = [shifted(d, delta) for d in dec], {"decisions": []}, time.time()
    geo = []
    for i, d in enumerate(seq):
        mode = d["interaction_mode"]
        if mode in ("close", "lift", "hold"):
            # step through the decision segment by segment to record the closure sequence
            import copy
            total, done = int(d["steps"]), 0
            while done < total and not bool(T.U.success[0].item()):
                dd = copy.deepcopy(d)
                dd["steps"] = min(3, total - done)
                _, res, _ = execute_decision(T, dd, S1.Nominal(), None, {"episode_id": f"replay/{variant}", "decision": i, "arm": "replay", "attempt": 0}, tl, update=False)
                done += res["steps_executed"]
                geo.append(dict(tip_geometry(T), t=T.steps_used, mode=mode))
                if res["steps_executed"] == 0:
                    break
        else:
            _, res, _ = execute_decision(T, d, S1.Nominal(), None, {"episode_id": f"replay/{variant}", "decision": i, "arm": "replay", "attempt": 0}, tl, update=False)
        out["decisions"].append({"mode": mode, "wrist_error_cm": res["hands"]["right"].get("position_error_cm"), "object": res["object"],
                                 "success": bool(T.U.success[0].item())})
        if bool(T.U.success[0].item()):
            break
    hold = T.robot.data.joint_pos.clone()
    for k in range(40):
        if bool(T.U.success[0].item()):
            break
        T._run(1, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold, record=True)
    tr = T.trace
    rise = [100 * (r["obj"][2] - T.rest_z) for r in tr]
    out.update(success=bool(T.U.success[0].item()), max_rise_cm=round(max(rise), 2), final_rise_cm=round(rise[-1], 2),
               min_up_z=round(min(r["up_z"] for r in tr), 3), steps=T.steps_used, wall_s=round(time.time() - t0, 1),
               start_object_m=o.round(4).tolist(), shift_m=(delta - (o - obj0)).round(5).tolist(), geometry=geo)
    return out


def cmd_geometry(a):
    import cup_transfer.cup_task  # noqa: F401
    out = OUT / ("smoke" if a.smoke else "") / "phase1" / a.obj
    out.mkdir(parents=True, exist_ok=True)
    T = RC.make_task(a.obj, render=False)
    lean(T)
    manifest(T, f"geometry_{a.obj}", {"stage": "phase1_geometry", "variants": a.variants, "starts": a.starts, "reps": a.reps, "tag": "smoke" if a.smoke else ""})
    lean_reset(T, (0.0, 0.0))
    rep = {"object_frame": object_frame_check(T)}
    print("FRAME", json.dumps(rep["object_frame"]), flush=True)
    rep["channels"] = channel_checks(T)
    print("CHANNELS done", flush=True)
    wj(out / "frame_and_channels.json", rep)
    tl = open(out / "replay_transitions.jsonl", "a")
    rows = []
    for variant in a.variants.split(","):
        for sid in a.starts.split(","):
            for r in range(a.reps):
                res = replay_grasp(T, DEV_STARTS[sid], variant, tl)
                res.update(variant=variant, start=sid, rep=r)
                rows.append(res)
                with open(out / "replay.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print("REPLAY", a.obj, variant, sid, r, res["success"], res["max_rise_cm"], flush=True)
    tl.close()
    print("RUN_DONE complete", flush=True)
    T.close()


# ---------------------------------------------------------------------------- Phases 2-3: candidate branches
STATES = {  # diagnostic decision states on cup B: (trajectory, decision index, control steps already executed in it)
    "A1": ("t1", 0, 18), "A2": ("t2", 0, 45), "C1": ("t1", 1, 0), "C2": ("t2", 1, 9), "L1": ("t1", 2, 0), "L2": ("t2", 2, 6)}


def logged_decisions(start):
    out = []
    for d in sorted((CUPB / start / "frozen_model" / "attempt_1").glob("decision_*")):
        r = json.loads((d / "decision.json").read_text())
        c = r.get("decision_content")
        if c and not c.get("stop") and r.get("executed"):
            out.append(c)
    return out


def run_candidate(T, decision, side, U_req, steps_list):
    """Open-loop execution of a K-segment requested sequence of the active hand, constructed exactly as execute_decision
    does for a command changed by System 1; the other hand follows Astra's command. Returns raw states (K+1)."""
    from cup_transfer.cup_task import EXEC, boosted, target_quat
    mode = decision["interaction_mode"]
    hold = T.robot.data.joint_pos.clone()
    qt = {s: target_quat(T, s, decision[s]) for s in HANDS if decision[s]["move"]}
    raws, sent = [T.raw_state(side)], []
    for k, u in enumerate(U_req):
        n = steps_list[k]
        targets, hands = {}, {}
        for s in HANDS:
            d = decision[s]
            hands[s] = tuple(float(x) for x in boosted(d, mode))
            if d["move"]:
                cur = T._ee(s)
                step = np.array(d["position_m"]) - cur[:3]
                cap = 0.01 * EXEC["lin_cap_cm"] * n
                if np.linalg.norm(step) > cap:
                    step *= cap / np.linalg.norm(step)
                ang = quat_angle_deg(cur[3:7], qt[s])
                al = 1.0 if ang <= 3.0 * n else 3.0 * n / ang
                targets[s] = np.concatenate([cur[:3] + step, quat_slerp(cur[3:7], qt[s], al)])
        cur = T._ee(side)
        base_q = targets[side][3:7] if side in targets else cur[3:7]
        targets[side] = np.concatenate([cur[:3] + np.asarray(u[:3]) / 100, base_q])
        hands[side] = tuple(float(x) for x in np.clip(u[3:], 0, 1))
        T._run(n, targets=targets, hands=hands, hold_q=hold, record=True, stop_on_success=False)
        for s in HANDS:
            T.hand_cmd[s] = hands[s]
        raws.append(T.raw_state(side))
        sent.append({"wrist_target_m": targets[side][:3].round(5).tolist(), "closure_targets": list(hands[side]),
                     "path_steps": dict(T.last_path_steps), "closure_ramp_steps": T.last_closure_ramp_steps})
    return raws, sent


def realized(raws, U_app, R):
    """The planner's trajectory format from realized states: (s_k, cumulative object xy displacement / R, cumulative
    commanded wrist z [m])."""
    s0 = S1.state_vector(raws[0])
    traj, d, z = [(s0, np.zeros(2), 0.0)], np.zeros(2), 0.0
    for k in range(1, len(raws)):
        d = d + (np.asarray(raws[k]["obj_pos"])[:2] - np.asarray(raws[k - 1]["obj_pos"])[:2]) / R
        z = z + U_app[k - 1][2] / 100
        traj.append((S1.state_vector(raws[k]), d.copy(), z))
    return traj


def effect(T, raws, goal_w, rest_z):
    r0, r1 = raws[0], raws[-1]
    o0, o1 = np.asarray(r0["obj_pos"]), np.asarray(r1["obj_pos"])
    tips = np.asarray(r1["tips"])
    return {"contact_N": round(float(r1["contact"]), 2), "object_rise_cm": round(100 * (o1[2] - rest_z), 3),
            "object_rise_change_cm": round(100 * (o1[2] - o0[2]), 3), "wrist_rise_cm": round(100 * (r1["wrist"][2] - r0["wrist"][2]), 3),
            "object_xy_moved_cm": round(100 * float(np.linalg.norm(o1[:2] - o0[:2])), 3),
            "object_tilt_deg": round(float(np.degrees(np.arccos(np.clip(r1["obj_up"][2], -1, 1)))), 2),
            "tips_radial_gap_cm": round(100 * (float(np.linalg.norm(tips[:2] - o1[:2])) - T.R), 2),
            "wrist_to_astra_target_cm": None if goal_w is None else round(100 * float(np.linalg.norm(np.asarray(r1["wrist"]) - goal_w)), 3),
            "closure_measured": r1["closure"]}


def continuation(T, seq, i_dec, steps_left):
    """Fixed continuation after a branch: the rest of the current decision and every later logged decision up to and
    including the first lift, all with the nominal executor, then 40 hold steps. Returns the outcome."""
    import copy
    from cup_transfer.cup_task import execute_decision
    tl = open("/dev/null", "w")
    z0 = T.rest_z
    if steps_left > 0:
        d = copy.deepcopy(seq[i_dec])
        d["steps"] = steps_left
        execute_decision(T, d, S1.Nominal(), None, {"episode_id": "cont", "decision": i_dec, "arm": "cont", "attempt": 0}, tl, update=False)
    j = i_dec + 1
    if seq[i_dec]["interaction_mode"] != "lift":
        while j < len(seq) and not bool(T.U.success[0].item()):
            execute_decision(T, seq[j], S1.Nominal(), None, {"episode_id": "cont", "decision": j, "arm": "cont", "attempt": 0}, tl, update=False)
            if seq[j]["interaction_mode"] == "lift":
                break
            j += 1
    hold = T.robot.data.joint_pos.clone()
    for k in range(40):
        if bool(T.U.success[0].item()):
            break
        T._run(1, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold, record=True)
    rise = [100 * (r["obj"][2] - z0) for r in T.trace]
    st = T.obj_state()
    return {"held": bool(T.U.success[0].item()), "max_rise_cm": round(max(rise), 2) if rise else None,
            "final_rise_cm": round(100 * (float(st["pos"][2]) - z0), 2), "final_up_z": round(float(st["up"][2]), 3)}


def cmd_branches(a):
    import copy
    from cup_transfer.cup_task import execute_decision, make_limit, nominal_sequence
    out = OUT / ("smoke" if a.smoke else "") / "phase2_3"
    out.mkdir(parents=True, exist_ok=True)
    T = RC.make_task(a.obj, render=False)
    lean(T)
    manifest(T, f"branches_{a.obj}", {"stage": "phase2_3_branches", "states": STATES, "reps": a.reps, "tag": "smoke" if a.smoke else ""})
    D_A = S1.ExperienceStore.load(SRC / "D_A.jsonl", frozen=True)
    hyper = json.loads((RC.ROOT / "method_configurations" / "hyper.json").read_text())["best"]
    tl = open(out / "prefix_transitions.jsonl", "a")
    for name in a.states.split(","):
        start, i_dec, k_done = STATES[name]
        if (out / f"{name}.json").exists():
            continue
        seq = logged_decisions(start)
        lean_reset(T, RC.STARTS[start])
        for j in range(i_dec):
            execute_decision(T, seq[j], S1.Nominal(), None, {"episode_id": f"prefix/{name}", "decision": j, "arm": "prefix", "attempt": 0}, tl, update=False)
        if k_done:
            d = copy.deepcopy(seq[i_dec])
            d["steps"] = k_done
            execute_decision(T, d, S1.Nominal(), None, {"episode_id": f"prefix/{name}", "decision": i_dec, "arm": "prefix", "attempt": 0}, tl, update=False)
        dec = copy.deepcopy(seq[i_dec])
        steps_rem = int(dec["steps"]) - k_done
        side, mode = dec["active_hand"], dec["interaction_mode"]
        sp = snap(T)
        # System 1's decision at this state, exactly as execute_decision would build it (traced planner, frozen D_A)
        start_pose = {s: T._ee(s) for s in HANDS}
        o0 = T.obj_state()
        g = np.array(dec[side]["position_m"]) if dec[side]["move"] else start_pose[side][:3]
        goal = {"lift_m": 0.05, "w_target": (g - o0["pos"]) / np.array([T.R, T.R, T.H])}
        r0 = T.raw_state(side)
        s0 = S1.state_vector(r0)
        ctx = {"mode": mode, "near": S1.near_object(s0), "hand": side}
        n = min(3, steps_rem)
        steps_list = [n] + [min(3, max(1, steps_rem - n - 3 * j)) for j in range(2)]
        U_nom = nominal_sequence(T, side, dec[side], start_pose[side], 3, steps_rem, mode)
        limit = make_limit(T, dec, side, steps_list)
        ctrl = S1.ShortHorizonController(D_A, h=hyper["h"], lam=hyper["lam"], seed=7000 + list(STATES).index(name), trace=True)
        plan = ctrl.plan(s0, ctx, U_nom, limit, goal, T.R, T.H)
        cands = plan.get("candidates_trace") or []
        res = {"state": name, "start": start, "decision": i_dec, "steps_done_in_decision": k_done, "mode": mode, "near": ctx["near"],
               "steps_remaining_in_decision": steps_rem, "plan": {k: v for k, v in plan.items() if k not in ("candidates_trace",)},
               "state_vector": s0.round(5).tolist(), "raw_state": r0, "candidates": []}
        if not cands:                                   # the planner fell back before generating candidates
            res["note"] = "planner fell back before candidates: " + str(plan.get("fallback"))
            dl = ctrl.candidates()
            fit = ctrl.model.fit(s0, limit(U_nom)[0], ctx)
            cands = []
            for i, dlt in enumerate(dl):
                U_req = U_nom + dlt * S1.BOUND
                U_req[:, 3:] = np.clip(U_req[:, 3:], 0, 1)
                cands.append({"i": i, "dlt": dlt.round(4).tolist(), "u_req": U_req.round(4).tolist(), "u_app": limit(U_req).round(4).tolist(),
                              "rejected": "planner fallback: " + str(plan.get("fallback")), "fit_available": fit is not None})
        uniq = {tuple(np.round(np.asarray(c["u_app"])[0], 3)) for c in cands}
        res["unique_first_segment_applied_commands"] = len(uniq)
        res["unique_first_segment_applied_closures"] = len({tuple(np.round(np.asarray(c["u_app"])[0][3:], 3)) for c in cands})
        with_cont = set(range(len(cands))) if mode in ("close", "lift") else set(range(8))
        for c in cands[:a.max_cands]:
            U_req = np.asarray(c["u_req"])
            U_app = np.asarray(c["u_app"])
            reps = []
            for r in range(a.reps + (2 if c["i"] == 0 else 0)):     # the nominal twice more: restore variability
                restore(T, sp)
                raws, sent = run_candidate(T, dec, side, U_req, steps_list)
                traj = realized(raws, U_app, T.R)
                Jr = S1.goal_loss(mode, traj, goal) + ctrl.alpha * float(np.sum(np.asarray(c["dlt"]) ** 2)) / 3 \
                    + ctrl.beta * float(np.sum(np.diff(np.asarray(c["dlt"]), axis=0) ** 2))
                rr = {"rep": r, "J_real": round(Jr, 6), "J_real_goal_terms": {k: round(v, 6) for k, v in S1.goal_terms(mode, traj, goal).items()},
                      "realized_ds_first": S1.delta_vector(raws[0], raws[1]).round(5).tolist(),
                      "effect": effect(T, raws, g if dec[side]["move"] else None, T.rest_z), "sent_first": sent[0]}
                if c["i"] in with_cont and (r == 0 or c["i"] == 0):
                    used = sum(steps_list)
                    rr["continuation"] = continuation(T, seq, i_dec, max(0, steps_rem - used))
                reps.append(rr)
            res["candidates"].append(dict(c, branches=reps))
            print("BRANCH", name, c["i"], c.get("rejected") is None, c.get("cost"), [x["J_real"] for x in reps],
                  [x.get("continuation", {}).get("held") for x in reps], flush=True)
        wj(out / f"{name}.json", res)
    tl.close()
    print("RUN_DONE complete", flush=True)
    T.close()


OFFSETS = {"shift": SHIFT_B, "x_only": np.array([SHIFT_B[0], 0, 0]), "y_only": np.array([0, SHIFT_B[1], 0]),
           "z_only": np.array([0, 0, SHIFT_B[2]]), "rim": None, "none": np.zeros(3)}


def cmd_persist(a):
    """Phase 3 complement: from the same restored decision states as the branches, apply a wrist-target offset to Astra's
    targets PERSISTENTLY (rest of the current decision and the fixed continuation), i.e. a correction that survives the
    nominal re-aiming that undoes System 1's 3-segment deviations. Is a useful correction inside the action interface?"""
    import copy
    from cup_transfer.cup_task import execute_decision
    out = OUT / "phase2_3"
    T = RC.make_task(a.obj, render=False)
    lean(T)
    manifest(T, f"persist_{a.obj}", {"stage": "phase3_persistent_offsets", "states": a.states, "offsets": a.offsets, "reps": a.reps, "tag": ""})
    tl = open("/dev/null", "w")
    rows = []
    for name in a.states.split(","):
        start, i_dec, k_done = STATES[name]
        seq = logged_decisions(start)
        for off in a.offsets.split(","):
            delta = np.array([0.0, 0.0, T.H - H_SOURCE]) if off == "rim" else OFFSETS[off]
            for r in range(a.reps):
                lean_reset(T, RC.STARTS[start])
                for j in range(i_dec):
                    execute_decision(T, seq[j], S1.Nominal(), None, {"episode_id": "prefix", "decision": j, "arm": "prefix", "attempt": 0}, tl, update=False)
                if k_done:
                    d = copy.deepcopy(seq[i_dec])
                    d["steps"] = k_done
                    execute_decision(T, d, S1.Nominal(), None, {"episode_id": "prefix", "decision": i_dec, "arm": "prefix", "attempt": 0}, tl, update=False)
                z0 = T.rest_z
                rest = copy.deepcopy(seq[i_dec])
                rest["steps"] = int(rest["steps"]) - k_done
                todo = [rest] + ([] if rest["interaction_mode"] == "lift" else [])
                if rest["interaction_mode"] != "lift":
                    for j in range(i_dec + 1, len(seq)):
                        todo.append(copy.deepcopy(seq[j]))
                        if seq[j]["interaction_mode"] == "lift":
                            break
                geo = None
                for d in todo:
                    d = shifted(d, delta)
                    execute_decision(T, d, S1.Nominal(), None, {"episode_id": "persist", "decision": 0, "arm": "persist", "attempt": 0}, tl, update=False)
                    if d["interaction_mode"] == "close":
                        geo = tip_geometry(T)
                    if bool(T.U.success[0].item()):
                        break
                hold = T.robot.data.joint_pos.clone()
                for k in range(40):
                    if bool(T.U.success[0].item()):
                        break
                    T._run(1, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold, record=True)
                rise = [100 * (x["obj"][2] - z0) for x in T.trace]
                res = {"state": name, "offset": off, "delta_m": np.round(delta, 4).tolist(), "rep": r, "held": bool(T.U.success[0].item()),
                       "max_rise_cm": round(max(rise), 2) if rise else None, "end_of_close_geometry": geo}
                rows.append(res)
                with open(out / "persist.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print("PERSIST", name, off, r, res["held"], res["max_rise_cm"], flush=True)
    print("RUN_DONE complete", flush=True)
    T.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live")
    p.add_argument("--obj", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--arm", default="frozen_model")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--policy", choices=["codex", "smoke"], default="codex")
    p = sub.add_parser("geometry")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--obj", required=True)
    p.add_argument("--variants", default="original,shift")
    p.add_argument("--starts", default=",".join(DEV_STARTS))
    p.add_argument("--reps", type=int, default=2)
    p = sub.add_parser("branches")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--states", default=",".join(STATES))
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-cands", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("persist")
    p.add_argument("--obj", default="cup_b")
    p.add_argument("--states", default="A2,C1,C2")
    p.add_argument("--offsets", default="none,shift,x_only,y_only,z_only,rim")
    p.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    {"live": cmd_live, "geometry": cmd_geometry, "branches": cmd_branches, "persist": cmd_persist}[a.cmd](a)


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
