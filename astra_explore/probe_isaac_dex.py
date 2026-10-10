"""Interface probe for the EgoVLA humanoid dex-hand benchmark (Isaac Sim 5.1 / IsaacLab 2.3, env `hongyi`).

    source ~/workspace/hongyi_deps/activate_hongyi.sh && cd ~/workspace/egovla_release/egovla && \
        python -u ~/workspace/Horsea/astra_explore/probe_isaac_dex.py --out <dir>

Measures, on the exact environment the Astra pilot will use (Humanoid-Unload-Cans-v0, room 1, table 1, the project's
test-protocol scene index): robot joint/body layout, hand joint limits, default hand-base frame and fingertip offsets,
camera intrinsics/poses + a projection check, step timing and GPU memory, differential-IK convergence per step,
reachability over a grid, hand closure geometry (fingers/thumb fractions), reset determinism and the table height.
Privileged object poses are used here ONLY to place probe targets; nothing from this script is shown to the model.
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="Humanoid-Stack-Can-Into-Drawer-v0")
ap.add_argument("--room", type=int, default=1)
ap.add_argument("--table", type=int, default=1)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True, device="cuda").app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

BENCH = "/home/licho/workspace/egovla_release/sim/Ego_Humanoid_Manipulation_Benchmark/source/extensions/humanoid.tasks"
sys.path.insert(0, BENCH)
import humanoid.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

R = {}
log = lambda *x: print("PROBE", *x, flush=True)


def gpu_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return next((int(l.split(",")[1]) for l in out.splitlines() if l.split(",")[0].strip() == str(os.getpid())), None)
    except Exception:  # noqa: BLE001
        return None


def scene_index(task, room, table, num_trials=1, num_episodes=3):
    """The project's ik_agent_30hz test protocol: seed_map[task] shuffles range(10000); test scenes start at index 100."""
    import random
    seed_map = {"Humanoid-Pour-Balls-v0": 0, "Humanoid-Sort-Cans-v0": 1, "Humanoid-Insert-Cans-v0": 2, "Humanoid-Unload-Cans-v0": 3,
                "Humanoid-Insert-And-Unload-Cans-v0": 4, "Humanoid-Push-Box-v0": 5, "Humanoid-Open-Drawer-v0": 6, "Humanoid-Close-Drawer-v0": 7,
                "Humanoid-Open-Laptop-v0": 8, "Humanoid-Flip-Mug-v0": 9, "Humanoid-Stack-Can-v0": 10, "Humanoid-Stack-Can-Into-Drawer-v0": 11}
    random.seed(seed_map[task])
    idx = list(range(10000))
    random.shuffle(idx)
    cur = 100 + (room * 5 + table) * num_trials * num_episodes + 1      # first episode: `curr_random_idx += 1` before reset
    return idx[cur]


t0 = time.time()
env_cfg = parse_env_cfg(a.task, num_envs=1)
env_cfg.episode_length_s = 60
env_cfg.randomize = True
env_cfg.spawn_background = True
env_cfg.room_idx, env_cfg.table_idx = a.room, a.table
env = gym.make(a.task, cfg=env_cfg)
U = env.unwrapped
ridx = scene_index(a.task, a.room, a.table)
U.cfg.randomize_idx = ridx
sink = io.StringIO()
with contextlib.redirect_stdout(sink):
    obs, _ = env.reset()
log(f"env created in {time.time() - t0:.0f} s; randomize_idx {ridx}; gpu {gpu_mb()} MB; step_dt {U.step_dt}; max_episode_length {U.max_episode_length}")
R["randomize_idx"], R["step_dt"], R["max_episode_length"], R["gpu_mb_after_reset"] = ridx, float(U.step_dt), int(U.max_episode_length), gpu_mb()

robot = U.robot
dev = U.device
R["joint_names"] = list(robot.joint_names)
R["body_names"] = list(robot.body_names)
lo, hi = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy(), robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
dflt = robot.data.default_joint_pos[0].cpu().numpy()
R["joint_limits"] = {n: [round(float(lo[i]), 3), round(float(hi[i]), 3), round(float(dflt[i]), 3)] for i, n in enumerate(robot.joint_names)}
arm_ids = {"left": list(U.cfg.left_arm_cfg.joint_ids), "right": list(U.cfg.right_arm_cfg.joint_ids)}
hand_ids = {"left": list(U.cfg.left_hand_cfg.joint_ids), "right": list(U.cfg.right_hand_cfg.joint_ids)}
hand_names = {s: [robot.joint_names[i] for i in hand_ids[s]] for s in hand_ids}
ee_idx = {"left": U.left_ee_idx, "right": U.right_ee_idx}
tip_idx = {"left": list(U.left_finger_tips_idx), "right": list(U.right_finger_tips_idx)}
R["arm_joint_ids"], R["hand_joint_ids"], R["hand_joint_names"], R["ee_body_idx"], R["tip_body_idx"] = arm_ids, hand_ids, hand_names, ee_idx, tip_idx
R["tip_body_names"] = {s: [robot.body_names[i] for i in tip_idx[s]] for s in tip_idx}
log("hand joints (lo, hi, default):", json.dumps({n: R["joint_limits"][n] for n in hand_names["left"]}))
log("ee bodies:", {s: robot.body_names[ee_idx[s]] for s in ee_idx}, "tips:", R["tip_body_names"])

# ---------------------------------------------------------------- helpers
ik = {"left": DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=dev),
      "right": DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="pinv"), num_envs=1, device=dev)}
action = torch.zeros((1, U.num_actions), device=dev)
hold_q = robot.data.joint_pos.clone()


def ee_pose(side):
    return robot.data.body_state_w[0, ee_idx[side], 0:7].cpu().numpy().copy()


def tips(side):
    return robot.data.body_state_w[0, tip_idx[side], 0:3].cpu().numpy().copy()


def hand_targets(side, fingers, thumb):
    out = []
    for i, n in zip(hand_ids[side], hand_names[side]):
        f = thumb if "thumb" in n else fingers
        out.append(float(lo[i] + f * (hi[i] - lo[i])))
    return torch.tensor(out, device=dev)


def ik_joint_targets(side, goal_w):
    root = robot.data.root_state_w[:, 0:7]
    jac = robot.root_physx_view.get_jacobians()[:, ee_idx[side] - 1, :, arm_ids[side]]
    ee_w = robot.data.body_state_w[:, ee_idx[side], 0:7]
    q = robot.data.joint_pos[:, arm_ids[side]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], ee_w[:, :3], ee_w[:, 3:7])
    goal = torch.tensor(goal_w, dtype=torch.float32, device=dev).unsqueeze(0)
    g_pos_b, g_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], goal[:, :3], goal[:, 3:7])
    ik[side].set_command(torch.cat([g_pos_b, g_quat_b], dim=-1), ee_pos_b, ee_quat_b)
    return ik[side].compute(ee_pos_b, ee_quat_b, jac, q)[0]


def step(goals=None, hands=None, hold=None):
    """goals: {side: 7-vector world pose} for arms under IK; others hold `hold` joint positions. hands: {side: (fingers, thumb)}."""
    action[:] = 0
    base = hold if hold is not None else robot.data.joint_pos
    for side in ("left", "right"):
        if goals and side in goals:
            action[0, arm_ids[side]] = ik_joint_targets(side, goals[side])
        else:
            action[0, arm_ids[side]] = base[0, arm_ids[side]]
        if hands and side in hands:
            action[0, hand_ids[side]] = hand_targets(side, *hands[side])
        else:
            action[0, hand_ids[side]] = base[0, hand_ids[side]]
    with contextlib.redirect_stdout(sink):
        o = env.step(action)[0]
    return o


def contact(side):
    s = U.left_hand_contact_sensor if side == "left" else U.right_hand_contact_sensor
    return float(torch.norm(s.data.net_forces_w, dim=-1).max().item())


def save(img_t, name):
    px = img_t.cpu().numpy() if hasattr(img_t, "cpu") else img_t
    Image.fromarray(np.asarray(px, dtype=np.uint8)).save(OUT / name)


def cam_info(cam):
    d = cam.data
    return {"K": d.intrinsic_matrices[0].cpu().numpy().round(2).tolist(), "pos_w": d.pos_w[0].cpu().numpy().round(4).tolist(),
            "quat_w_world_wxyz": d.quat_w_world[0].cpu().numpy().round(4).tolist(), "shape": list(d.image_shape)}


def project(cam, P):
    """world point -> pixel using IsaacLab's 'world' camera convention (forward +x, left +y, up +z)."""
    d = cam.data
    K = d.intrinsic_matrices[0].cpu().numpy()
    Rm = matrix_from_quat(d.quat_w_world[0:1])[0].cpu().numpy()
    p = Rm.T @ (np.asarray(P, float) - d.pos_w[0].cpu().numpy())
    if p[0] <= 0.02:
        return None
    u = K[0, 0] * (-p[1] / p[0]) + K[0, 2]
    v = K[1, 1] * (-p[2] / p[0]) + K[1, 2]
    return float(u), float(v)


cams = {"main": U.main_camera, "left_eye": U.left_eye_camera, "right_eye": U.right_eye_camera, "left_hand": U.left_hand_camera, "right_hand": U.right_hand_camera}

# ---------------------------------------------------------------- settle at the default pose, timing
t1 = time.time()
for _ in range(60):
    o = step(hold=hold_q)
R["step_wall_s_mean_settle"] = round((time.time() - t1) / 60, 3)
R["gpu_mb_running"] = gpu_mb()
log(f"settled 60 steps: {R['step_wall_s_mean_settle']} s/step; gpu {R['gpu_mb_running']} MB")
R["obs_keys"] = sorted(o.keys())
R["obs_shapes"] = {k: list(v.shape) for k, v in o.items() if hasattr(v, "shape")}
log("obs keys:", R["obs_keys"]); log("shapes:", R["obs_shapes"])
R["cameras"] = {n: cam_info(c) for n, c in cams.items()}
log("cameras:", json.dumps(R["cameras"]))

# privileged (probe only): object poses and table bounds
objs = {n: ob.data.root_state_w[0, :7].cpu().numpy().round(4).tolist() for n, ob in U.scene.rigid_objects.items()}
arts = {}
for n, ar in U.scene.articulations.items():
    if n == "robot":
        continue
    arts[n] = {"root_pose": ar.data.root_state_w[0, :7].cpu().numpy().round(4).tolist(), "joint_names": list(ar.joint_names),
               "joint_pos": ar.data.joint_pos[0].cpu().numpy().round(4).tolist(), "joint_limits": ar.data.joint_limits[0].cpu().numpy().round(4).tolist(),
               "body_names": list(ar.body_names), "body_pos": ar.data.body_state_w[0, :, :3].cpu().numpy().round(4).tolist()}
if hasattr(U, "drawer_handle_offset"):
    arts["drawer"]["handle_pos_w_from_offset"] = (U.drawer.data.root_state_w[0, :3] + U.drawer_handle_offset).cpu().numpy().round(4).tolist()
R["objects_probe_only"], R["articulations_probe_only"] = objs, arts
log("objects:", objs); log("articulations:", json.dumps(arts))
try:
    from pxr import Usd, UsdGeom  # noqa: E402
    import omni.usd  # noqa: E402
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Table")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    R["table_bbox_world"] = {"min": [round(x, 4) for x in bbox.GetMin()], "max": [round(x, 4) for x in bbox.GetMax()]}
    log("table bbox:", R["table_bbox_world"])
except Exception as ex:  # noqa: BLE001
    log("table bbox failed:", repr(ex))

# default-pose frames: ee pose, fingertip offsets expressed in the hand-base frame
frames = {}
for side in ("left", "right"):
    pose = ee_pose(side)
    Rm = matrix_from_quat(torch.tensor(pose[3:7]).unsqueeze(0))[0].numpy()
    T = tips(side)
    frames[side] = {"ee_pos_w": pose[:3].round(4).tolist(), "ee_quat_wxyz": pose[3:7].round(4).tolist(),
                    "base_axes_world_cols_xyz": Rm.round(3).tolist(),
                    "tips_world": {n: T[i].round(4).tolist() for i, n in enumerate(R["tip_body_names"][side])},
                    "tips_in_base_frame": {n: (Rm.T @ (T[i] - pose[:3])).round(4).tolist() for i, n in enumerate(R["tip_body_names"][side])},
                    "mean_tip_offset_in_base": (Rm.T @ (T.mean(0) - pose[:3])).round(4).tolist(),
                    "arm_joint_pos": robot.data.joint_pos[0, arm_ids[side]].cpu().numpy().round(3).tolist()}
R["default_frames"] = frames
log("default frames:", json.dumps(frames))

# images + projection check (draw ee and fingertips of both hands into every camera)
for n, c in cams.items():
    img = c.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for side, col in (("left", (255, 60, 60)), ("right", (60, 120, 255))):
        for P in [ee_pose(side)[:3]] + list(tips(side)):
            uv = project(c, P)
            if uv and 0 <= uv[0] < im.width and 0 <= uv[1] < im.height:
                dr.ellipse([uv[0] - 5, uv[1] - 5, uv[0] + 5, uv[1] + 5], outline=col, width=2)
    for on, pose in objs.items():
        uv = project(c, pose[:3])
        if uv and 0 <= uv[0] < im.width and 0 <= uv[1] < im.height:
            dr.rectangle([uv[0] - 6, uv[1] - 6, uv[0] + 6, uv[1] + 6], outline=(255, 255, 0), width=2)
            dr.text((uv[0] + 8, uv[1] - 6), on, fill=(255, 255, 0))
    if "drawer" in arts and "handle_pos_w_from_offset" in arts["drawer"]:
        uv = project(c, arts["drawer"]["handle_pos_w_from_offset"])
        if uv and 0 <= uv[0] < im.width and 0 <= uv[1] < im.height:
            dr.rectangle([uv[0] - 6, uv[1] - 6, uv[0] + 6, uv[1] + 6], outline=(0, 255, 0), width=2)
            dr.text((uv[0] + 8, uv[1] - 6), "handle", fill=(0, 255, 0))
    im.save(OUT / f"default_{n}_marked.png")
    save(img, f"default_{n}.png")
save(o["rgb"][0], "default_rgb_eyes_concat.png")
R["projection_check_images"] = sorted(p.name for p in OUT.glob("default_*_marked.png"))

# ---------------------------------------------------------------- IK convergence: translate each hand 12 cm up then 12 cm forward
conv = {}
for side in ("left", "right"):
    start = ee_pose(side)
    goal = start.copy()
    goal[2] += 0.12
    goal[0] += 0.12
    errs = []
    for s in range(90):
        step(goals={side: goal}, hold=hold_q)
        if s + 1 in (5, 10, 15, 20, 30, 45, 60, 90):
            e = ee_pose(side)
            q_err = quat_mul(torch.tensor(goal[3:7]).unsqueeze(0), quat_inv(torch.tensor(e[3:7]).unsqueeze(0)))[0].numpy()
            errs.append({"step": s + 1, "pos_err_cm": round(100 * float(np.linalg.norm(e[:3] - goal[:3])), 2),
                         "ang_err_deg": round(float(np.degrees(2 * np.arccos(np.clip(abs(q_err[0]), 0, 1)))), 1)})
    conv[side] = errs
    log(f"IK convergence {side} (12 cm up+forward):", errs)
R["ik_convergence"] = conv
R["step_wall_s_mean_ik"] = None

# ---------------------------------------------------------------- hand closure geometry
hand_geo = {}
for fingers, thumb in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.7, 0.7), (1.0, 1.0)):
    goals = {s: ee_pose(s) for s in ("left", "right")}
    for _ in range(30):
        step(goals=goals, hands={"left": (fingers, thumb), "right": (fingers, thumb)}, hold=hold_q)
    rec = {}
    for side in ("left", "right"):
        pose = ee_pose(side)
        Rm = matrix_from_quat(torch.tensor(pose[3:7]).unsqueeze(0))[0].numpy()
        T = tips(side)
        names = R["tip_body_names"][side]
        ti = {n: T[i] for i, n in enumerate(names)}
        th = next(v for k, v in ti.items() if "thumb" in k)
        ix = next(v for k, v in ti.items() if "index" in k)
        pk = next(v for k, v in ti.items() if "pinky" in k)
        rec[side] = {"thumb_index_cm": round(100 * float(np.linalg.norm(th - ix)), 2), "index_pinky_cm": round(100 * float(np.linalg.norm(ix - pk)), 2),
                     "mean_tip_offset_in_base": (Rm.T @ (T.mean(0) - pose[:3])).round(4).tolist(),
                     "tips_in_base_frame": {n: (Rm.T @ (T[i] - pose[:3])).round(4).tolist() for i, n in enumerate(names)},
                     "hand_joint_pos": robot.data.joint_pos[0, hand_ids[side]].cpu().numpy().round(3).tolist(), "index_contact_force_N": round(contact(side), 3)}
    hand_geo[f"fingers{fingers}_thumb{thumb}"] = rec
    save(U.left_hand_camera.data.output["rgb"][0, :, :, :3], f"hand_f{fingers}_t{thumb}_left_hand_cam.png")
    save(U.main_camera.data.output["rgb"][0, :, :, :3], f"hand_f{fingers}_t{thumb}_main.png")
    log(f"hand fingers={fingers} thumb={thumb}:", json.dumps(rec["left"]))
R["hand_geometry"] = hand_geo
for _ in range(20):      # reopen
    step(goals={s: ee_pose(s) for s in ("left", "right")}, hands={"left": (0, 0), "right": (0, 0)}, hold=hold_q)

# ---------------------------------------------------------------- reset determinism
with contextlib.redirect_stdout(sink):
    o1, _ = env.reset()
for _ in range(60):
    o1 = step(hold=hold_q)
img1 = U.main_camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
obj1 = {n: U.scene.rigid_objects[n].data.root_state_w[0, :7].cpu().numpy().copy() for n in objs}
with contextlib.redirect_stdout(sink):
    o2, _ = env.reset()
for _ in range(60):
    o2 = step(hold=hold_q)
img2 = U.main_camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
obj2 = {n: U.scene.rigid_objects[n].data.root_state_w[0, :7].cpu().numpy().copy() for n in objs}
R["reset_determinism"] = {"drawer_joint_pos": U.drawer.data.joint_pos[0].cpu().numpy().round(4).tolist() if hasattr(U, "drawer") else None, "image_identical": bool(np.array_equal(img1, img2)), "image_mean_abs_diff": round(float(np.abs(img1.astype(int) - img2.astype(int)).mean()), 4),
                          "object_max_pos_diff_m": round(max(float(np.linalg.norm(obj1[n][:3] - obj2[n][:3])) for n in objs), 6),
                          "episode_length_buf_after_reset": int(U.episode_length_buf[0].item())}
log("reset determinism:", R["reset_determinism"])

# ---------------------------------------------------------------- reachability grid (45 IK steps per target, default orientation)
reach = []
for side in ("left", "right"):
    q0 = ee_pose(side)[3:7]
    ys = [0.35, 0.15, 0.0] if side == "left" else [-0.35, -0.15, 0.0]
    for x in (0.25, 0.40, 0.55, 0.70):
        for y in ys:
            for z in (1.02, 1.15, 1.30):
                with contextlib.redirect_stdout(sink):
                    env.reset()
                for _ in range(10):
                    step(hold=hold_q)
                goal = np.array([x, y, z, *q0])
                for _ in range(45):
                    step(goals={side: goal}, hold=hold_q)
                e = ee_pose(side)
                reach.append({"side": side, "target": [x, y, z], "pos_err_cm": round(100 * float(np.linalg.norm(e[:3] - goal[:3])), 1),
                              "achieved": e[:3].round(3).tolist()})
R["reach_grid"] = reach
log("reach grid:", json.dumps(reach))
with contextlib.redirect_stdout(sink):
    env.reset()
R["gpu_mb_end"] = gpu_mb()
(OUT / "probe_results.json").write_text(json.dumps(R, indent=1))
log("DONE wrote", OUT / "probe_results.json", "gpu", R["gpu_mb_end"])
env.close()
app.close()
