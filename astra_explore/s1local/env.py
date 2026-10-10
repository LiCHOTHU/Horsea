"""Environment layer of the local-execution-adaptation pilot (System 1 beneath Astra), dex-hand setup.

Uses the existing harness unchanged (astra_explore/isaac_env.py IsaacDexTask: EgoVLA Humanoid-Stack-Can-Into-Drawer-v0,
Unitree H1 + two Inspire hands, wrist targets tracked by the benchmark's differential IK along a <= 1 cm / 3 deg per step
path, four closure fractions per hand ramped at <= 1/30 per step). Only additions:
  * rendering off for bulk runs (render interval and camera update period beyond any reachable step; the cameras stay
    in the scene because the benchmark builds and reads them unconditionally);
  * lean decision execution (the harness's act() without image observations);
  * snapshot / restore of the complete start state of an operation;
  * runtime mass / friction scaling through the PhysX views (as Isaac Lab's own randomization events do);
  * a state-based outcome reader (evaluation only; never an input to action generation).
"""
import json
import sys
from pathlib import Path

import numpy as np

AE = "/home/licho/workspace/Horsea/astra_explore"
if AE not in sys.path:
    sys.path.insert(0, AE)
from isaac_env import GROUPS, HANDS, OPEN, IsaacDexTask, quat_angle_deg, quat_from_mat, rot_from_dirs  # noqa: E402

PILOT = Path("/home/licho/workspace/Horsea/experiments/astra_explore/run_2026-10-10_astra_isaac_stack_can_drawer")
CAMERAS = ("left_eye_camera", "right_eye_camera", "main_camera", "left_hand_camera", "right_hand_camera")


def load_program(attempt=10):
    """Astra's executed decisions of a pilot attempt (decision_content, in order)."""
    out = []
    for d in sorted((PILOT / f"attempt_{attempt}").glob("decision_*")):
        r = json.loads((d / "decision.json").read_text())
        c = r.get("decision_content")
        if c and not c.get("stop"):
            out.append(c)
    return out


def no_render(cfg):
    cfg.sim.render_interval = 10 ** 9
    for c in CAMERAS:
        getattr(cfg, c).update_period = 1e9


def build(render=False, scene_index=5788):
    return IsaacDexTask(scene_index=scene_index, cfg_hook=None if render else no_render)


# ---------------------------------------------------------------------- execution
def reset(T):
    """Env reset + the harness's 60 settle steps with open hands (as IsaacDexTask.reset, without image observations)."""
    T._reset_env()
    T.steps_used, T._frames = 0, []
    T.hand_cmd = {s: OPEN for s in HANDS}
    hold = T.robot.data.joint_pos.clone()
    T._run(T.settle_steps, hands={s: OPEN for s in HANDS}, hold_q=hold)


def targets_of(T, d):
    """Wrist targets of a decision exactly as IsaacDexTask.act computes them (calibrated finger / palm directions)."""
    targets = {}
    for s in HANDS:
        x = d[s]
        T.hand_cmd[s] = tuple(float(x[g]) for g in GROUPS)
        if x["move"]:
            R = rot_from_dirs(x["fingers_dir"], x["palm_dir"], T.calibration[s]["f_b"], T.calibration[s]["p_b"])
            targets[s] = np.array(list(x["position_m"]) + list(quat_from_mat(R)), dtype=float)
    return targets


def path_steps(T, targets, lin, ang):
    """Steps the harness's interpolated wrist path needs at per-step limits (lin m, ang deg)."""
    n = 0
    for s, tgt in targets.items():
        p0 = T._ee(s)
        n = max(n, int(np.ceil(np.linalg.norm(tgt[:3] - p0[:3]) / lin)), int(np.ceil(quat_angle_deg(p0[3:7], tgt[3:7]) / ang)))
    return n


def act(T, d, speed=1.0):
    """Execute one decision through the harness's own _run. speed < 1 slows the wrist path (per-step limits x speed) and
    stretches the decision to ceil(steps / speed); speed > 1 shortens the decision to ceil(steps / speed) but never below
    the steps the path needs at the EXISTING limits (1 cm, 3 deg per step), which are never exceeded. Returns steps executed."""
    hold = T.robot.data.joint_pos.clone()
    targets = targets_of(T, d)
    f = min(1.0, float(speed))
    lin, ang = IsaacDexTask.MAX_LIN_M * f, IsaacDexTask.MAX_ANG_DEG * f
    steps = int(np.ceil(int(d["steps"]) / float(speed)))
    if targets:
        steps = max(steps, path_steps(T, targets, lin, ang))
    T.MAX_LIN_M, T.MAX_ANG_DEG = lin, ang                 # instance attributes shadow the class limits for this call only
    try:
        T._run(steps, targets=targets, hands={s: T.hand_cmd[s] for s in HANDS}, hold_q=hold)
    finally:
        del T.MAX_LIN_M, T.MAX_ANG_DEG
    return steps


def hold_window(T, n, monitor=None):
    """n control steps holding both arms at their current joint positions and the hands at their last commands; calls
    monitor(k) after every step (state-based evaluation only)."""
    hold = T.robot.data.joint_pos.clone()
    hands = {s: T.hand_cmd[s] for s in HANDS}
    for k in range(n):
        T._run(1, hands=hands, hold_q=hold)
        if monitor is not None:
            monitor(k)
    return n


# ---------------------------------------------------------------------- snapshot / restore
def snapshot(T):
    """Complete start state: every articulation (root pose / velocity, joint position / velocity AND the commanded joint
    position / velocity / effort targets), every rigid object, the env's episode counter, and the harness's own command
    state (closure commands and the previous command used for the closure ramp)."""
    U = T.U
    st = U.scene.get_state(is_relative=False)
    st = {k: {n: {f: v.clone() for f, v in e.items()} for n, e in g.items()} for k, g in st.items()}
    tg = {n: (a.data.joint_pos_target.clone(), a.data.joint_vel_target.clone(), a.data.joint_effort_target.clone())
          for n, a in U.scene.articulations.items()}
    return {"scene": st, "targets": tg, "episode_length_buf": U.episode_length_buf.clone(),
            "hand_cmd": dict(T.hand_cmd), "hand_cmd_prev": dict(T.hand_cmd_prev), "steps_used": T.steps_used}


def restore(T, snap):
    U = T.U
    U.scene.reset_to(snap["scene"], is_relative=False)
    for n, a in U.scene.articulations.items():             # reset_to sets targets = measured state; put the commanded ones back
        p, v, e = snap["targets"][n]
        a.set_joint_position_target(p)
        a.set_joint_velocity_target(v)
        a.set_joint_effort_target(e)
    U.scene.write_data_to_sim()
    U.episode_length_buf[:] = snap["episode_length_buf"]
    T.hand_cmd, T.hand_cmd_prev, T.steps_used = dict(snap["hand_cmd"]), dict(snap["hand_cmd_prev"]), snap["steps_used"]
    for s in HANDS:
        T.ik[s].reset()
    U.sim.forward()                                         # refresh link kinematics before anything reads them
    U.scene.update(dt=0.0)


# ---------------------------------------------------------------------- physics properties
def object_state(T):
    """Evaluation-only physical state (never given to the search)."""
    U = T.U
    can = U.can.data.root_state_w[0].cpu().numpy()
    plate = U.plate.data.root_state_w[0].cpu().numpy()
    q = U.drawer.data.joint_pos[0].cpu().numpy()
    lim = U.drawer.data.joint_limits[0].cpu().numpy()
    tips_r = T._tips("right").mean(0)
    return {"can_pos": can[:3].tolist(), "can_quat": can[3:7].tolist(), "can_lin_vel": can[7:10].tolist(), "plate_pos": plate[:3].tolist(),
            "door_q": float(q[U.drawer_bottom_idx]), "door_lo": float(lim[U.drawer_bottom_idx, 0]), "door_hi": float(lim[U.drawer_bottom_idx, 1]),
            "right_tips_mean": tips_r.tolist(), "env_success": bool(U.success[0].item())}


# ---------------------------------------------------------------------- physical conditions
def shapes_per_body(asset):
    """Collision shapes per body, in body order (Isaac Lab's own approach in events.randomize_rigid_body_material)."""
    return [asset._physics_sim_view.create_rigid_body_view(lp).max_shapes for lp in asset.root_physx_view.link_paths[0]]


def shape_ids(asset, bodies):
    nums = shapes_per_body(asset)
    starts = np.cumsum([0] + nums[:-1])
    ids = []
    for b in bodies:
        i = asset.body_names.index(b)
        ids += list(range(int(starts[i]), int(starts[i] + nums[i])))
    return ids


class Physics:
    """Nominal masses / inertias / materials captured once; a condition = multiplicative factors on the interacting bodies.
    grasp / transport: the can (mass, inertia, static + dynamic friction) and every right-hand link (static + dynamic
    friction). door: the door panel link 'bottom' of the cabinet articulation (mass, inertia, friction) and every left-hand
    link (friction). Everything else keeps its nominal values. Friction is scaled on BOTH bodies of each contact pair so that
    the combined contact friction scales by the factor (average / min / max combine modes)."""

    def __init__(self, T):
        import torch
        self.torch, self.T = torch, T
        U = T.U
        self.env_ids = torch.tensor([0], dtype=torch.int32)
        self.can_mass0 = U.can.root_physx_view.get_masses().clone()
        self.can_inertia0 = U.can.root_physx_view.get_inertias().clone()
        self.can_mat0 = U.can.root_physx_view.get_material_properties().clone()
        self.robot_mat0 = U.robot.root_physx_view.get_material_properties().clone()
        self.drawer_mass0 = U.drawer.root_physx_view.get_masses().clone()
        self.drawer_inertia0 = U.drawer.root_physx_view.get_inertias().clone()
        self.drawer_mat0 = U.drawer.root_physx_view.get_material_properties().clone()
        hands = {s: [b for b in U.robot.body_names if b.startswith(s[0].upper() + "_") and b.split("_", 1)[1].split("_")[0] in
                     ("hand", "index", "middle", "ring", "pinky", "thumb")] for s in HANDS}
        self.hand_bodies = hands
        self.hand_shapes = {s: shape_ids(U.robot, hands[s]) for s in HANDS}
        self.door_body = U.drawer.body_names.index("bottom")
        self.door_shapes = shape_ids(U.drawer, ["bottom"])
        self.current = None

    def nominal(self):
        U = self.T.U
        U.can.root_physx_view.set_masses(self.can_mass0.clone(), self.env_ids)
        U.can.root_physx_view.set_inertias(self.can_inertia0.clone(), self.env_ids)
        U.can.root_physx_view.set_material_properties(self.can_mat0.clone(), self.env_ids)
        U.robot.root_physx_view.set_material_properties(self.robot_mat0.clone(), self.env_ids)
        U.drawer.root_physx_view.set_masses(self.drawer_mass0.clone(), self.env_ids)
        U.drawer.root_physx_view.set_inertias(self.drawer_inertia0.clone(), self.env_ids)
        U.drawer.root_physx_view.set_material_properties(self.drawer_mat0.clone(), self.env_ids)
        self.current = None

    def apply(self, op, mass_f, fric_f):
        """Returns a record of exactly what was changed (before / after values)."""
        U = self.T.U
        self.nominal()
        rec = {"op": op, "mass_factor": mass_f, "friction_factor": fric_f}
        if op in ("grasp", "transport"):
            m = self.can_mass0.clone() * mass_f
            I = self.can_inertia0.clone() * mass_f
            mat = self.can_mat0.clone()
            mat[..., 0:2] *= fric_f
            U.can.root_physx_view.set_masses(m, self.env_ids)
            U.can.root_physx_view.set_inertias(I, self.env_ids)
            U.can.root_physx_view.set_material_properties(mat, self.env_ids)
            rm = self.robot_mat0.clone()
            rm[0, self.hand_shapes["right"], 0:2] *= fric_f
            U.robot.root_physx_view.set_material_properties(rm, self.env_ids)
            rec.update(can_mass_kg=[float(self.can_mass0.flatten()[0]), float(m.flatten()[0])],
                       can_friction_static_dynamic=[self.can_mat0[0, 0, 0:2].tolist(), mat[0, 0, 0:2].tolist()],
                       right_hand_shapes=len(self.hand_shapes["right"]),
                       right_hand_friction_first=[self.robot_mat0[0, self.hand_shapes["right"][0], 0:2].tolist(), rm[0, self.hand_shapes["right"][0], 0:2].tolist()])
        else:
            m = self.drawer_mass0.clone()
            m[0, self.door_body] *= mass_f
            I = self.drawer_inertia0.clone()
            I[0, self.door_body] *= mass_f
            mat = self.drawer_mat0.clone()
            mat[0, self.door_shapes, 0:2] *= fric_f
            U.drawer.root_physx_view.set_masses(m, self.env_ids)
            U.drawer.root_physx_view.set_inertias(I, self.env_ids)
            U.drawer.root_physx_view.set_material_properties(mat, self.env_ids)
            rm = self.robot_mat0.clone()
            rm[0, self.hand_shapes["left"], 0:2] *= fric_f
            U.robot.root_physx_view.set_material_properties(rm, self.env_ids)
            rec.update(door_mass_kg=[float(self.drawer_mass0[0, self.door_body]), float(m[0, self.door_body])],
                       door_shapes=len(self.door_shapes), door_friction_first=[self.drawer_mat0[0, self.door_shapes[0], 0:2].tolist(), mat[0, self.door_shapes[0], 0:2].tolist()],
                       left_hand_shapes=len(self.hand_shapes["left"]),
                       left_hand_friction_first=[self.robot_mat0[0, self.hand_shapes["left"][0], 0:2].tolist(), rm[0, self.hand_shapes["left"][0], 0:2].tolist()])
        self.current = (op, mass_f, fric_f)
        return rec

    def readback(self):
        U = self.T.U
        return {"can_mass": float(U.can.root_physx_view.get_masses().flatten()[0]),
                "can_friction": U.can.root_physx_view.get_material_properties()[0, 0, 0:2].tolist(),
                "door_mass": float(U.drawer.root_physx_view.get_masses()[0, self.door_body]),
                "door_friction": U.drawer.root_physx_view.get_material_properties()[0, self.door_shapes[0], 0:2].tolist(),
                "right_hand_friction": U.robot.root_physx_view.get_material_properties()[0, self.hand_shapes["right"][0], 0:2].tolist(),
                "left_hand_friction": U.robot.root_physx_view.get_material_properties()[0, self.hand_shapes["left"][0], 0:2].tolist()}


# ---------------------------------------------------------------------- start states
def reset_with_can_offset(T, dxy=(0.0, 0.0)):
    """Env reset with the can displaced on its support by dxy (m) before the settle steps (a fresh, valid start)."""
    T._reset_env()
    if dxy[0] or dxy[1]:
        st = T.U.can.data.root_state_w.clone()
        st[0, 0] += float(dxy[0])
        st[0, 1] += float(dxy[1])
        T.U.can.write_root_state_to_sim(st)
    T.steps_used, T._frames = 0, []
    T.hand_cmd = {s: OPEN for s in HANDS}
    hold = T.robot.data.joint_pos.clone()
    T._run(T.settle_steps, hands={s: OPEN for s in HANDS}, hold_q=hold)


def set_door(T, q):
    """Place the (passive) door hinge at q rad, at rest."""
    dr = T.U.drawer
    jp = dr.data.joint_pos.clone()
    jp[0, T.U.drawer_bottom_idx] = float(q)
    dr.write_joint_state_to_sim(jp, self_zero(jp))
    T.U.sim.forward()
    T.U.scene.update(dt=0.0)


def self_zero(x):
    return x * 0
