"""Cup-transfer harness: IsaacDexTask on Humanoid-Cup-v0 + the COMMON segment executor every method uses.

CupTask adds, for every method alike: the tracked object pose (geometric base frame, never the centre-of-mass frame), its
velocity and its observable dimensions in the observation; a privileged per-control-step trace for evaluation only (never
prompted); predefined start offsets; the lift / place success rule of env_cups.

execute_decision runs one Astra decision in segments of SEG_STEPS control steps (the last one shorter). Before each segment
the method (Nominal / ShortHorizonController / NearestNeighborReuse) may replace the active hand's nominal segment command
(wrist translation, four closure targets) by a bounded alternative; orientation always follows Astra's command (at most
3 deg per step, as in the original harness). Translation per segment is capped at 1 cm per step (the harness limit), so
the requested segment target is always reachable by the path interpolation; closure changes follow the harness's
1/30-per-step ramp. Every executed segment is logged as a transition (requested and applied commands kept apart) and, for
updating methods, added to the experience store after the prediction error has been recorded.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_env import GROUPS, HANDS, IsaacDexTask, quat_angle_deg, quat_from_mat, quat_slerp, rot_from_dirs  # noqa: E402
from cup_transfer import system1 as S1  # noqa: E402

SEG = S1.SEG_STEPS
# common executor settings for EVERY method (the brief's "conservative execution" check may change them before the Astra runs)
EXEC = {"lin_cap_cm": 1.0, "closure_boost": 0.0}
OBJ_WORD = {"cup_a": "cup", "cup_b": "cup", "cup_bp": "cup", "cyl_c": "cylinder", "dev_f1": "cup", "dev_f2": "cup", "dev_f3": "cup", "dev_f4": "cup", "dev_g1": "cup", "dev_g2": "cup", "dev_g3": "cup", "dev_g4": "cup"}


def no_render(cfg):
    cfg.sim.render_interval = 10 ** 9
    for name in ("left_eye_camera", "right_eye_camera", "main_camera", "left_hand_camera", "right_hand_camera"):
        getattr(cfg, name).update_period = 10 ** 9


def light_render(cfg):
    """Render only what Astra receives: the four prompted cameras at exactly the sizes it is given (no downscaling step),
    rgb only except the main camera's depth (read by the benchmark's own observation code), and the unused right-eye camera
    at the left eye's size (the benchmark concatenates the two eye images), rgb only, never updated. Field of view unchanged."""
    sizes = {"main_camera": (960, 540), "left_eye_camera": (640, 360), "left_hand_camera": (640, 360), "right_hand_camera": (640, 360)}
    for name, (w, h) in sizes.items():
        c = getattr(cfg, name)
        c.width, c.height = w, h
        c.data_types = ["rgb", "distance_to_image_plane"] if name == "main_camera" else ["rgb"]
    r = cfg.right_eye_camera
    r.width, r.height, r.data_types, r.update_period = 640, 360, ["rgb"], 10 ** 9


class CupTask(IsaacDexTask):
    def __init__(self, obj, specs, mode="lift", instruction=None, render=True, horizon=400, tray_xy=None, light=False, **kw):
        self.obj, self.spec = obj, specs[obj]

        def hook(cfg):
            cfg.target.spawn.usd_path = self.spec["usd"]
            cfg.task_mode = mode
            if tray_xy is not None:
                cfg.tray_xy = tuple(tray_xy)
                cfg.container.init_state.pos = (tray_xy[0], tray_xy[1], cfg.container.init_state.pos[2])
            if not render:
                no_render(cfg)
            elif light:
                light_render(cfg)
        super().__init__("Humanoid-Cup-v0", scene_index=0, task_modules=["cup_transfer.env_cups"], cfg_hook=hook, **kw)
        self.horizon = horizon
        word = OBJ_WORD[obj]
        self.instruction = instruction or (f"Pick up the {word}, lift it at least 5 cm above its initial position, and hold it steadily for one second.")
        self.R, self.H = self.spec["diameter_m"] / 2, self.spec["height_m"]
        self.rest_z, self.trace, self._tracing = None, [], True
        base = self._step

        def traced(action):
            o = base(action)
            if self._tracing:
                self.trace.append(self.priv_step())
            return o
        self._step = traced

    # ------------------------------------------------------------------ object state (permitted for every method)
    def obj_state(self):
        ob = self.U.scene.rigid_objects["target"]
        p = ob.data.root_link_pose_w[0].cpu().numpy() - np.concatenate([self.U.scene.env_origins[0].cpu().numpy(), [0, 0, 0, 0]])
        w, x, y, z = p[3:7]
        up = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        v = ob.data.root_com_lin_vel_w[0].cpu().numpy()
        return {"pos": p[:3], "quat": p[3:7], "up": np.array(up), "vel": v}

    def set_start(self, offset_xy):
        self.U.cfg.start_offset_xy = (float(offset_xy[0]), float(offset_xy[1]))

    def reset(self):
        self.trace = []
        obs = super().reset()
        self.rest_z = float(self.obj_state()["pos"][2])
        self.U.lift_ref_z = self.rest_z
        self.trace = []
        return self._observation()

    def _observation(self):
        o = super()._observation()
        st = self.obj_state()
        o["object"] = {"name": OBJ_WORD[self.obj], "base_centre_position_m_world": st["pos"].round(4).tolist(),
                       "up_axis_world": st["up"].round(3).tolist(), "linear_velocity_m_s": st["vel"].round(3).tolist(),
                       "diameter_m": round(2 * self.R, 4), "height_m": round(self.H, 4),
                       "rise_above_start_cm": None if self.rest_z is None else round(100 * (st["pos"][2] - self.rest_z), 1)}
        return o

    # ------------------------------------------------------------------ evaluation-only trace (never prompted)
    def priv_step(self):
        st = self.obj_state()
        return {"t": self.steps_used, "obj": st["pos"].round(4).tolist(), "up_z": round(float(st["up"][2]), 4),
                "obj_speed": round(float(np.linalg.norm(st["vel"])), 4),
                "tips": {s: self._tips(s).mean(0).round(4).tolist() for s in HANDS}, "wrist": {s: self._ee(s)[:3].round(4).tolist() for s in HANDS},
                "contact": {s: round(self._contact(s), 2) for s in HANDS}, "success": bool(self.U.success[0].item())}

    # ------------------------------------------------------------------ System 1 raw state
    def raw_state(self, side):
        st = self.obj_state()
        wv = [0.0, 0.0, 0.0]
        if len(self.trace) > SEG:
            a, b = self.trace[-1 - SEG]["wrist"][side], self.trace[-1]["wrist"][side]
            wv = ((np.array(b) - np.array(a)) * 30.0 / SEG).tolist()
        fd, pd, _ = self._dirs_world(side)
        cm = self._closure_measured(side)
        return {"wrist": self._ee(side)[:3].tolist(), "tips": self._tips(side).mean(0).tolist(), "fingers_dir": fd.tolist(), "palm_dir": pd.tolist(),
                "closure": [cm.get(g, 0.0) for g in GROUPS], "contact": self._contact(side), "wrist_vel": wv,
                "obj_pos": st["pos"].tolist(), "obj_up": st["up"].tolist(), "obj_vel": st["vel"].tolist(), "rest_z": self.rest_z,
                "R": self.R, "H": self.H}


# ---------------------------------------------------------------------------- persistent grasp correction (batch 2)
def obj_rotation(T):
    w, x, y, z = T.obj_state()["quat"]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PersistentOffset:
    """A correction of the active hand's grasp REFERENCE, p_ref,t = pbar_ref,t + R_WO(t0) dp_O (dp_O object-relative, cm,
    R_WO captured at activation): added to every commanded wrist target (move = true) of that hand, never to the current
    wrist, so it cannot accumulate. It belongs to one grasp attempt: it survives later decisions of the attempt, ends on
    release (the hand reopens after a closure), a new attempt (reset) or when its duration elapses, and an ending
    persistent correction is ramped out at <= ramp_cm per control step (no target jump). duration_steps = transient
    correction: it ends after that many control steps and the reference returns at once, as System 1's original
    deviations did. Attach as T.persist; absent or inactive, the executor is unchanged."""

    def __init__(self, side="right", ramp_cm=0.3):
        self.side, self.ramp = side, ramp_cm / 100.0
        self.reset()

    def reset(self):
        self.dw, self.active, self.until, self.t0, self.dp_obj, self.closed, self.events = np.zeros(3), False, None, None, None, False, []

    def activate(self, T, dp_obj_cm, duration_steps=None, reason=""):
        R = obj_rotation(T)
        self.dp_obj = np.asarray(dp_obj_cm, float)
        self.dw = R @ (self.dp_obj / 100.0)
        self.active, self.t0 = True, T.steps_used
        self.until = None if duration_steps is None else T.steps_used + int(duration_steps)
        self.events.append({"event": "activate", "t": T.steps_used, "dp_obj_cm": self.dp_obj.round(4).tolist(),
                            "dw_m": self.dw.round(5).tolist(), "duration_steps": duration_steps, "reason": reason})

    def end(self, T, reason, ramp=True):
        if self.active or np.any(self.dw):
            self.events.append({"event": "end", "t": T.steps_used, "reason": reason, "ramp": ramp})
        self.active, self.until = False, None
        if not ramp:
            self.dw = np.zeros(3)

    def observe(self, T, decision):
        """At every decision start: a release (the hand reopens after a closure) ends the grasp attempt's correction."""
        d = decision.get(self.side) or {}
        m = max(float(d.get("finger_base", 0.0)), float(d.get("finger_tip", 0.0)))
        if m >= 0.3:
            self.closed = True
        elif self.closed and m < 0.2:
            self.closed = False
            self.end(T, "release", ramp=True)

    def world_offset(self, T, n):
        """Offset (m, world) for the segment about to run (n control steps)."""
        if self.active and self.until is not None and T.steps_used >= self.until:
            self.end(T, "duration elapsed", ramp=False)
        if not self.active and np.any(self.dw):
            nrm = float(np.linalg.norm(self.dw))
            self.dw = np.zeros(3) if nrm <= self.ramp * n else self.dw * (1 - self.ramp * n / nrm)
        return self.dw.copy()


# ---------------------------------------------------------------------------- the common executor
def target_quat(T, side, d):
    R = rot_from_dirs(d["fingers_dir"], d["palm_dir"], T.calibration[side]["f_b"], T.calibration[side]["p_b"])
    return quat_from_mat(R)


def nominal_sequence(T, side, d, start_pose, K, steps_left, mode="other", offset=None):
    """K nominal REQUESTED segment commands (dp [cm], closure targets) toward Astra's target (plus an active persistent
    reference offset, if any) from the current wrist."""
    cur = T._ee(side)[:3]
    goal = (np.array(d["position_m"]) + (0.0 if offset is None else offset)) if d["move"] else np.array(start_pose[:3])
    rem = goal - cur
    out = []
    for k in range(K):
        n = min(SEG, max(1, steps_left - k * SEG)) if steps_left > k * SEG else SEG
        cap = 0.01 * EXEC["lin_cap_cm"] * n
        step = rem if np.linalg.norm(rem) <= cap else rem * cap / np.linalg.norm(rem)
        if not d["move"]:
            step = np.zeros(3)
        rem = rem - step
        out.append(np.concatenate([100 * step, boosted(d, mode)]))
    return np.array(out)


def boosted(d, mode):
    """Closure command of a hand after the common executor's closure boost (only while closing / lifting / holding / carrying)."""
    c = np.array([d[g] for g in GROUPS], float)
    if EXEC["closure_boost"] and mode in ("close", "lift", "hold", "transport") and c[0] > 0.2:
        c = np.clip(c + EXEC["closure_boost"], 0, 1)
    return c


def make_limit(T, decision, side, steps_list):
    """The executor's own limits applied to a REQUESTED sequence of the active hand: translation <= 1 cm per step; closure
    commands advance along the harness ramp (joint ramp over both hands at 1/30 per step, from the previous command)."""
    other = "left" if side == "right" else "right"

    def limit(U_req):
        prev = {s: np.array(T.hand_cmd_prev[s], float) for s in HANDS}
        out = []
        for k, u in enumerate(U_req):
            n = steps_list[min(k, len(steps_list) - 1)]
            dp = u[:3].copy()
            nrm = np.linalg.norm(dp)
            cap = n * EXEC["lin_cap_cm"]
            if nrm > cap:
                dp *= cap / nrm
            tgt = {side: np.clip(u[3:], 0, 1), other: boosted(decision[other], decision.get("interaction_mode", "other"))}
            dmax = max(float(np.max(np.abs(tgt[s] - prev[s]))) for s in HANDS)
            ramp = max(1, int(math.ceil(dmax / IsaacDexTask.MAX_CLOSURE_STEP - 1e-9)))
            frac = min(1.0, n / ramp)
            for s in HANDS:
                prev[s] = prev[s] + frac * (tgt[s] - prev[s])
            out.append(np.concatenate([dp, prev[side].copy()]))
        return np.array(out)
    return limit


def hand_target(T, s, d, n, qt, off):
    """The segment's wrist target toward the hand's reference (Astra's target + persistent offset), capped at the executor's
    per-step limits (1 cm and 3 deg per control step). off = 0 gives exactly the original computation."""
    cur = T._ee(s)
    step = np.array(d["position_m"]) + off - cur[:3]
    cap = 0.01 * EXEC["lin_cap_cm"] * n
    if np.linalg.norm(step) > cap:
        step *= cap / np.linalg.norm(step)
    ang = quat_angle_deg(cur[3:7], qt[s])
    a = 1.0 if ang <= 3.0 * n else 3.0 * n / ang
    return np.concatenate([cur[:3] + step, quat_slerp(cur[3:7], qt[s], a)])


def active_side(decision):
    s = decision.get("active_hand", "none")
    return s if s in HANDS else None


def execute_decision(T, decision, method, store, ctx_meta, log_f, update=True, K=3, cand_f=None):
    """Executes one validated Astra decision through the common segment executor. Returns (obs, result, seg_rows).
    Lineage per segment (diagnostics, behaviour unchanged): wall time of planning / simulation, the wrist target and closure
    targets actually sent to the simulator, the achieved wrist and closure change, the harness's path / ramp steps; a traced
    planner's candidate table goes to cand_f."""
    side = active_side(decision)
    mode = decision.get("interaction_mode", "other")
    total = max(0, min(int(decision["steps"]), T.horizon - T.steps_used))
    hold = T.robot.data.joint_pos.clone()
    start_pose = {s: T._ee(s) for s in HANDS}
    qt = {s: target_quat(T, s, decision[s]) for s in HANDS if decision[s]["move"]}
    o0 = T.obj_state()
    po = getattr(T, "persist", None)                    # persistent grasp correction (batch 2); None = original executor
    if po is not None:
        po.observe(T, decision)
    goal = {"lift_m": 0.05}
    if side is not None:
        g = np.array(decision[side]["position_m"]) if decision[side]["move"] else start_pose[side][:3]
        if po is not None and side == po.side and decision[side]["move"]:
            g = g + po.dw
        goal["w_target"] = (g - o0["pos"]) / np.array([T.R, T.R, T.H])
    rows, done, success_step, adj = [], 0, None, []
    t0 = time.time()
    while done < total and not bool(T.U.success[0].item()):
        n = min(SEG, total - done)
        steps_list = [n] + [min(SEG, max(1, total - done - n - SEG * j)) for j in range(K - 1)]
        off = {s: np.zeros(3) for s in HANDS}
        if po is not None:
            off[po.side] = po.world_offset(T, n)
        targets, hands = {}, {}
        for s in HANDS:
            d = decision[s]
            hands[s] = tuple(float(x) for x in boosted(d, mode))
            if d["move"]:
                targets[s] = hand_target(T, s, d, n, qt, off[s])
        plan, rec, tp = None, None, 0.0
        if side is not None:
            r0 = T.raw_state(side)
            s0 = S1.state_vector(r0)
            ctx = {"mode": mode, "near": S1.near_object(s0), "hand": side}
            U_nom = nominal_sequence(T, side, decision[side], start_pose[side], K, total - done, mode, offset=off[side])
            limit = make_limit(T, decision, side, steps_list)
            tp0 = time.perf_counter()
            plan = method.plan(s0, ctx, U_nom, limit, goal, T.R, T.H)
            tp = time.perf_counter() - tp0
            ctrace = {k: plan.pop(k) for k in ("candidates_trace", "fit_neighbors") if k in plan}
            act = plan.pop("activate_offset", None)
            if act is not None and po is not None and side == po.side:
                # the method asked for a persistent (or timed) correction of the grasp reference: it takes effect in
                # this very segment, and this segment's command becomes the nominal step toward the corrected reference
                po.activate(T, act["dp_obj_cm"], act.get("duration_steps"), act.get("reason", ""))
                off[side] = po.world_offset(T, n)
                if decision[side]["move"]:
                    targets[side] = hand_target(T, side, decision[side], n, qt, off[side])
                U_nom = nominal_sequence(T, side, decision[side], start_pose[side], K, total - done, mode, offset=off[side])
                plan["u_req"] = U_nom[0]
            u_req = np.asarray(plan["u_req"], float)
            dlt = u_req - U_nom[0]
            adj.append(dlt)
            if np.any(np.abs(dlt) > 1e-9):                      # the method changed the active hand's segment command
                cur = T._ee(side)
                base_q = targets[side][3:7] if side in targets else cur[3:7]
                targets[side] = np.concatenate([cur[:3] + u_req[:3] / 100, base_q])
                hands[side] = tuple(float(x) for x in np.clip(u_req[3:], 0, 1))
            u_app = limit(np.array([u_req]))[0]
        sent = {"wrist_target_m": None if side not in targets else [round(float(x), 5) for x in targets[side][:3]],
                "closure_targets": None if side is None else list(hands[side])}
        ts0 = time.perf_counter()
        obs_raw, executed, s_step = T._run(n, targets=targets, hands=hands, hold_q=hold, record=True, stop_on_success=True)
        ts = time.perf_counter() - ts0
        for s in HANDS:
            T.hand_cmd[s] = hands[s]
        done += executed
        if s_step is not None:
            success_step = s_step
        if side is not None:
            r1 = T.raw_state(side)
            ds = S1.delta_vector(r0, r1)
            pred = plan.get("predicted_ds")
            perr = None if pred is None else float(np.sqrt(np.mean((np.array(pred) - ds) ** 2)))
            rec = {"timestamp": time.time(), "episode_id": ctx_meta["episode_id"], "object_instance_id": T.obj,
                   "interaction_context": {"mode": mode, "near": ctx["near"], "hand": side},
                   "observed_state_before": r0, "requested_command": [round(float(x), 5) for x in u_req],
                   "applied_command_after_limits": [round(float(x), 5) for x in u_app], "observed_state_after": r1,
                   "duration": executed / 30.0, "observation_validity": bool(np.all(np.isfinite(ds))) and executed == n,
                   "s": s0.round(6).tolist(), "u_applied": [round(float(x), 6) for x in u_app], "ds": ds.round(6).tolist(),
                   "provenance": ctx_meta.get("provenance", "target")}
            added = store.add(rec) if (update and store is not None) else False
            lineage = {"sent": sent, "achieved_wrist_cm": [round(100 * (b - a), 3) for a, b in zip(r0["wrist"], r1["wrist"])],
                       "reference_offset_cm": [round(100 * float(x), 4) for x in off[side]],
                       "reference_m": None if not decision[side]["move"] else [round(float(x), 5) for x in np.array(decision[side]["position_m"]) + off[side]],
                       "closure_measured_before": r0["closure"], "closure_measured_after": r1["closure"],
                       "path_steps": dict(getattr(T, "last_path_steps", {})), "closure_ramp_steps": getattr(T, "last_closure_ramp_steps", None),
                       "steps": executed, "t_steps_used_after": T.steps_used}
            if cand_f is not None and ctrace:
                cand_f.write(json.dumps({"decision": ctx_meta["decision"], "segment": len(rows), "mode": mode, "near": ctx["near"],
                                         "selected": plan.get("candidate"), "fallback": plan.get("fallback"), **ctrace}) + "\n")
            row = dict(rec, decision=ctx_meta["decision"], segment=len(rows), plan={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan.items()},
                       lineage=lineage, timing={"plan_s": round(tp, 5), "sim_s": round(ts, 4)},
                       nominal_command=[round(float(x), 5) for x in U_nom[0]], adjustment=[round(float(x), 5) for x in dlt],
                       prediction_rmse_before_update=perr, added_to_memory=added, method=method.name, arm=ctx_meta["arm"], attempt=ctx_meta["attempt"])
            log_f.write(json.dumps(row) + "\n")
            rows.append(row)
        if executed < n:
            break
    log_f.flush()
    tob = time.perf_counter()
    obs = T._observation()
    tob = time.perf_counter() - tob
    result = {"observation_s": round(tob, 3), "steps_requested": int(decision["steps"]), "steps_executed": done, "clipped": None if done == int(decision["steps"]) else
              (f"executed {done} of {decision['steps']} steps" + (": attempt horizon reached" if T.steps_used >= T.horizon else ": task completed" if success_step else "")),
              "wall_s": round(time.time() - t0, 2), "hands": {}, "task_complete": obs["task_complete"], "success_at_step": success_step, "steps_used": T.steps_used}
    for s in HANDS:
        after = T._ee(s)
        r = {"moved_cm": round(100 * float(np.linalg.norm(after[:3] - start_pose[s][:3])), 2), "wrist_position_m_world": after[:3].round(4).tolist(),
             "closure_commanded": dict(zip(GROUPS, T.hand_cmd[s])), "closure_measured": obs["proprio"][s]["closure_measured"],
             "index_contact_force_N": obs["proprio"][s]["index_contact_force_N"]}
        if decision[s]["move"]:
            tq = qt[s]
            ang = quat_angle_deg(tq, after[3:7])
            r.update(tracked=True, target_position_m=decision[s]["position_m"], position_error_cm=round(100 * float(np.linalg.norm(after[:3] - np.array(decision[s]["position_m"]))), 2),
                     orientation_error_deg=round(ang, 1))
        else:
            r.update(tracked=False)
        result["hands"][s] = r
    A = np.array(adj) if adj else np.zeros((0, 7))
    result["execution_layer"] = {"segments": len(rows) if side is not None else int(math.ceil(done / SEG)),
                                 "adjusted_segments": int(np.sum(np.any(np.abs(A) > 1e-9, axis=1))) if len(A) else 0,
                                 "mean_translation_adjustment_cm": round(float(np.mean(np.linalg.norm(A[:, :3], axis=1))), 2) if len(A) else 0.0,
                                 "mean_closure_adjustment": round(float(np.mean(np.abs(A[:, 3:]))), 3) if len(A) else 0.0}
    st = T.obj_state()
    result["object"] = {"moved_cm": round(100 * float(np.linalg.norm(st["pos"][:2] - o0["pos"][:2])), 2),
                        "rise_above_start_cm": round(100 * (float(st["pos"][2]) - T.rest_z), 2), "up_axis_z": round(float(st["up"][2]), 3)}
    return obs, result, rows
