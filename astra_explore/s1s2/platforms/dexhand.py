"""Dexterous-hand adapter of the System-1 / System-2 transfer study (EgoVLA Ego-Humanoid benchmark,
Humanoid-Stack-Can-Into-Drawer-v0, Unitree H1 + two Inspire hands, Isaac Sim 5.1 / IsaacLab 2.3, conda env `hongyi`).

Student (System 1): stateless ACT from the project's pipeline (egovla_release/policy/policy_rl/act: ResNet-18 + CVAE transformer,
4 enc / 7 dec layers, hidden 512, ff 3200, 8 heads, chunk 25, kl 10, AdamW lr 1e-5 / wd 1e-4, batch 8, masked L1 + KL), trained
FROM SCRATCH on Astra's recorded episodes (no robot-space demonstrations of this task exist on this machine; EgoVLA fine-tuning
is infeasible on the shared GPU) -> cohort = distillation from scratch. Native interface: the world-fixed main camera
(`fixed_rgb`, 1280x720) resized to 224x224 (INTER_AREA, as the project's deploy code), proprio zeroed (STATELESS); action
38-D = [left wrist xyz + quat wxyz | right wrist xyz + quat wxyz | left hand 12 joint targets | right hand 12], world frame,
converted to joint targets by the benchmark's differential IK (left DLS, right pinv) every 30 Hz step; 24 of 25 chunk steps
executed open loop before re-querying (the project's n_action_exec).

Teacher (System 2): Astra through the existing dex bridge (absolute wrist targets interpolated <= 1 cm / 3 deg per step + four
hand-closure fractions, rate-limited). For student-compatible labels both controllers execute through ONE native function:
every control step sends a 38-D command (wrist goals to the same IK, hand joint targets directly). Instrumentation change vs the
pilots, disclosed: an arm Astra holds (move=false) is held by IK-tracking its pose at the decision start (the student's control
law) instead of a joint hold, so every executed step has a 38-D label.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HORSEA = "/home/licho/workspace/Horsea"
AE = HORSEA + "/astra_explore"
ACT_DIR = "/home/licho/workspace/egovla_release/policy/policy_rl/act"
for _p in (AE, HORSEA, ACT_DIR, ACT_DIR + "/detr"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK_ID = "Humanoid-Stack-Can-Into-Drawer-v0"
ROOM, TABLE = 1, 2
STEP_BUDGET = 400                    # the project's TASK_MAX_HORIZON for this task (30 Hz control steps)
SETTLE = 60
CHUNK, EXEC_CHUNK, IMG = 25, 24, 224
ACT_CFG = {"lr": 1e-5, "kl_weight": 10, "chunk_size": 25, "hidden_dim": 512, "dim_feedforward": 3200, "batch": 8}
RECIPE = {"source": "egovla_release/policy/policy_rl/act (imitate_episodes.py / train_eval.sh)", "policy": "ACT, STATELESS=1",
          **ACT_CFG, "optimizer": "AdamW (policy.configure_optimizers: lr 1e-5, backbone lr 1e-5, weight decay 1e-4), no LR schedule, no clipping",
          "U": 3000, "dU": 1500,
          "note": "the project's recipe is 2000 epochs of one sampled timestep per episode; the study fixes the UPDATE count instead "
                  "(same for D1 / D3 / DN / failures), as the spec requires equal update budgets across data sizes"}
POOL_SLICES = {"source": (0, 15), "dev": (15, 20), "adapt": (20, 30), "test": (30, 40)}

from s1s2.registry import BUDGET as _BUDGET  # noqa: E402
RESET_CALLS = _BUDGET["astra_calls_per_episode_cap"]     # 40 calls per from-reset episode (the pilots' ceiling)
TAKEOVER_CALLS = _BUDGET["astra_calls_per_takeover"]     # 20 calls per takeover (E08 / E09 / corrections)

_TASK = {}


def _git_rev():
    import subprocess
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HORSEA).stdout.strip()


def pools():
    """randomize_idx values of the project's evaluation protocol for room 1 / table 2 (seed_map[task] = 11; first evaluation
    episode at shuffled position 100 + (room*5 + table)*3 + 1 = 122); consecutive positions form the disjoint pools."""
    random.seed(11)
    idx = list(range(10000))
    random.shuffle(idx)
    base = 100 + (ROOM * 5 + TABLE) * 3 + 1
    return {k: [idx[base + j] for j in range(a, b)] for k, (a, b) in POOL_SLICES.items()}


# ====================================================================== environment with native 38-D execution
def get_task():
    if "t" not in _TASK:
        from isaac_env import IsaacDexTask
        _TASK["t"] = StudyDex.build(IsaacDexTask)
    return _TASK["t"]


class StudyDex:
    """Wraps IsaacDexTask (the pilots' adapter, unchanged) with a native 38-D step and per-step recording."""

    @classmethod
    def build(cls, IsaacDexTask):
        self = cls()
        self.T = IsaacDexTask(TASK_ID, ROOM, TABLE, pools()["source"][0], settle_steps=SETTLE)
        self.torch = self.T.torch
        self.rec = None
        self.native_steps = 0
        return self

    def reset(self, scene_index):
        T = self.T
        T.U.cfg.randomize_idx = int(scene_index)
        T.scene_index = int(scene_index)
        obs = T.reset()                     # env reset + SETTLE steps holding the default pose (as in the pilots)
        self.native_steps = 0
        self.prev_quat = {}
        return obs

    def image224(self):
        import cv2
        px = self.T.cams["main_camera"].data.output["rgb"][0, :, :, :3].cpu().numpy()
        return cv2.resize(np.ascontiguousarray(px, dtype=np.uint8), (IMG, IMG), interpolation=cv2.INTER_AREA)

    def success(self):
        return bool(self.T.U.success[0].item())

    def apply38(self, a38, source, record=True):
        """One native control step: wrist goals (world) -> benchmark IK -> arm joint targets; hand joint targets directly."""
        T, torch = self.T, self.torch
        a38 = np.asarray(a38, dtype=np.float64)
        if record and self.rec is not None:
            lab = a38.copy()
            for i, side in ((3, "left"), (10, "right")):            # quaternion sign continuity for the L1 label
                q = lab[i:i + 4]
                if side in self.prev_quat and float(np.dot(q, self.prev_quat[side])) < 0:
                    lab[i:i + 4] = -q
                self.prev_quat[side] = lab[i:i + 4].copy()
            self.rec(self.image224(), lab.astype(np.float32), source, self.native_steps)
        T.action[:] = 0
        T.action[0, T.arm_ids["left"]] = T._ik_targets("left", a38[0:7])
        T.action[0, T.arm_ids["right"]] = T._ik_targets("right", a38[7:14])
        T.action[0, T.hand_ids["left"]] = torch.tensor(a38[14:26], dtype=torch.float32, device=T.dev)
        T.action[0, T.hand_ids["right"]] = torch.tensor(a38[26:38], dtype=torch.float32, device=T.dev)
        obs = T._step(T.action)
        T.steps_used += 1
        self.native_steps += 1
        if T.steps_used % T.video_every == 0:
            T._frames.append(T._image("main_camera", T.video_size))
        return obs

    def hand_q(self, side):
        return self.T._hand_q(side)

    def act_native(self, decision, step_budget):
        """Execute one validated Astra decision through apply38 (the pilots' act(), with IK holds). Returns (obs, result)."""
        from isaac_env import GROUPS, mat_from_quat, quat_from_mat, quat_slerp, quat_angle_deg, rot_from_dirs
        T = self.T
        req = int(decision["steps"])
        steps = max(0, min(req, step_budget - self.native_steps))
        clip = None if steps == req else f"requested {req} steps, executed {steps}: physical budget reached"
        before = {s: T._ee(s) for s in ("left", "right")}
        path, hand_goal, hand_start = {}, {}, {}
        for s in ("left", "right"):
            d = decision[s]
            T.hand_cmd[s] = tuple(float(d[g]) for g in GROUPS)
            p0 = before[s]
            if d["move"]:
                R = rot_from_dirs(d["fingers_dir"], d["palm_dir"], T.calibration[s]["f_b"], T.calibration[s]["p_b"])
                tgt = np.concatenate([np.asarray(d["position_m"], float), quat_from_mat(R)])
            else:
                tgt = p0.copy()                                     # IK hold at the decision-start pose
            n = max(1, int(np.ceil(np.linalg.norm(tgt[:3] - p0[:3]) / T.MAX_LIN_M)), int(np.ceil(quat_angle_deg(p0[3:7], tgt[3:7]) / T.MAX_ANG_DEG)))
            path[s] = (p0, tgt, n)
            hand_start[s] = np.asarray(self.hand_q(s), float)
            hand_goal[s] = T._hand_targets(s, T.hand_cmd[s]).cpu().numpy().astype(float)
        dmax = max(abs(T.hand_cmd[s][k] - T.hand_cmd_prev[s][k]) for s in ("left", "right") for k in range(len(GROUPS)))
        ramp = max(1, int(np.ceil(dmax / T.MAX_CLOSURE_STEP)))
        t0 = time.time()
        executed, success_step = 0, None
        for i in range(steps):
            a38 = np.zeros(38)
            for s, off in (("left", 0), ("right", 7)):
                p0, tgt, n = path[s]
                a = min(1.0, (i + 1) / n)
                a38[off:off + 3] = p0[:3] + a * (tgt[:3] - p0[:3])
                a38[off + 3:off + 7] = quat_slerp(p0[3:7], tgt[3:7], a)
            ah = min(1.0, (i + 1) / ramp)
            a38[14:26] = hand_start["left"] + ah * (hand_goal["left"] - hand_start["left"])
            a38[26:38] = hand_start["right"] + ah * (hand_goal["right"] - hand_start["right"])
            self.apply38(a38, "astra")
            executed = i + 1
            if self.success():
                success_step = self.native_steps
                break
        frac = min(1.0, executed / ramp) if executed else 0.0
        for s in ("left", "right"):
            T.hand_cmd_prev[s] = tuple(T.hand_cmd_prev[s][k] + frac * (T.hand_cmd[s][k] - T.hand_cmd_prev[s][k]) for k in range(len(GROUPS)))
        obs = T._observation()
        result = {"steps_requested": req, "steps_executed": executed, "clipped": clip, "wall_s": round(time.time() - t0, 2), "hands": {},
                  "task_complete": obs["task_complete"], "success_at_step": success_step, "steps_used": self.native_steps}
        for s in ("left", "right"):
            after = T._ee(s)
            p0, tgt, n = path[s]
            Rt, Ra = mat_from_quat(tgt[3:7]), mat_from_quat(after[3:7])
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Rt.T @ Ra) - 1) / 2, -1, 1))))
            result["hands"][s] = {"tracked": bool(decision[s]["move"]), "moved_cm": round(100 * float(np.linalg.norm(after[:3] - before[s][:3])), 2),
                                  "wrist_position_m_world": after[:3].round(4).tolist(), "target_position_m": tgt[:3].round(4).tolist(),
                                  "position_error_cm": round(100 * float(np.linalg.norm(after[:3] - tgt[:3])), 2), "orientation_error_deg": round(ang, 1),
                                  "closure_commanded": dict(zip(GROUPS, T.hand_cmd[s])), "closure_measured": obs["proprio"][s]["closure_measured"],
                                  "thumb_index_distance_cm": obs["proprio"][s]["thumb_index_distance_cm"],
                                  "index_contact_force_N": obs["proprio"][s]["index_contact_force_N"]}
        return obs, result

    def close(self):
        self.T.close()


def shutdown():
    if "t" in _TASK:
        _TASK["t"].close()       # Isaac's app.close() ends the process (result.json is already written)


# ====================================================================== recording
class Log:
    def __init__(self):
        self.img, self.act, self.src, self.step, self.seg = [], [], [], [], []
        self.cur = 0

    def new_segment(self):
        self.cur += 1

    def __call__(self, img, a38, source, step):
        self.img.append(img)
        self.act.append(a38)
        self.src.append(0 if source == "astra" else 1)
        self.step.append(step)
        self.seg.append(self.cur)

    def steps(self, source=None):
        return len(self.src) if source is None else sum(1 for s in self.src if s == (0 if source == "astra" else 1))

    def save(self, path):
        np.savez_compressed(path, imgs=np.stack(self.img) if self.img else np.zeros((0, IMG, IMG, 3), np.uint8),
                            action=np.stack(self.act) if self.act else np.zeros((0, 38), np.float32),
                            source=np.asarray(self.src, np.int8), step=np.asarray(self.step, np.int32), segment=np.asarray(self.seg, np.int32))


# ====================================================================== student (ACT)
def build_act(device="cuda"):
    """ACTPolicy exactly as imitate_episodes.make_policy builds it (its argparse needs the training CLI's required flags)."""
    saved = sys.argv
    sys.argv = ["s1s2", "--ckpt_dir", "/tmp/unused", "--policy_class", "ACT", "--task_name", "sim_s1s2", "--seed", "0",
                "--num_epochs", "1", "--kl_weight", str(ACT_CFG["kl_weight"]), "--chunk_size", str(ACT_CFG["chunk_size"]),
                "--hidden_dim", str(ACT_CFG["hidden_dim"]), "--batch_size", str(ACT_CFG["batch"]), "--dim_feedforward", str(ACT_CFG["dim_feedforward"]),
                "--lr", str(ACT_CFG["lr"])]
    try:
        from policy import ACTPolicy
        cfg = {"lr": ACT_CFG["lr"], "num_queries": CHUNK, "kl_weight": ACT_CFG["kl_weight"], "hidden_dim": ACT_CFG["hidden_dim"],
               "dim_feedforward": ACT_CFG["dim_feedforward"], "lr_backbone": 1e-5, "backbone": "resnet18", "enc_layers": 4, "dec_layers": 7,
               "nheads": 8, "camera_names": ["top"], "state_dim": 38}
        pol = ACTPolicy(cfg)
    finally:
        sys.argv = saved
    return pol.to(device)


class Student:
    def __init__(self, ckpt):
        import torch
        self.torch = torch
        s = torch.load(ckpt, map_location="cuda", weights_only=False)
        self.policy = build_act()
        self.policy.load_state_dict(s["model"])
        self.policy.eval()
        self.stats = {k: np.asarray(v, np.float32) for k, v in s["stats"].items()}
        self.calls = 0

    def act(self, img224):
        torch = self.torch
        im = torch.from_numpy(img224).permute(2, 0, 1).float().div(255.0)[None, None].cuda()
        q = torch.zeros(1, 38, device="cuda")                        # STATELESS: proprio zeroed after normalization
        with torch.no_grad():
            a = self.policy(q, im)[0].cpu().numpy()
        self.calls += 1
        return a * self.stats["action_std"] + self.stats["action_mean"]   # (25, 38)


def student_segment(D, student, exec_chunk, step_budget, stop_at=None, trace=None):
    decision, records = 0, []
    while True:
        if D.success():
            return "success", decision, records
        if D.native_steps >= step_budget:
            return "horizon", decision, records
        if stop_at is not None and D.native_steps >= stop_at:
            return "handover", decision, records
        img = D.image224()
        chunk = student.act(img)
        rec = {"decision": decision, "native_step_before": D.native_steps, "wrists_before": [D.T._ee(s)[:3].round(4).tolist() for s in ("left", "right")],
               "image_before": img, "executed_subsampled": []}
        for k, a in enumerate(chunk[:exec_chunk]):
            D.apply38(a, "student")                       # pre-action image + the dispatched 38-D command are recorded
            if k % 8 == 0:
                rec["executed_subsampled"].append(np.asarray(a).round(3).tolist())
            if D.success() or D.native_steps >= step_budget:
                break
        rec["wrists_after"] = [D.T._ee(s)[:3].round(4).tolist() for s in ("left", "right")]
        rec["thumb_index_cm_after"] = [D.T._observation()["proprio"][s]["thumb_index_distance_cm"] for s in ("left", "right")] if False else None
        records.append(rec)
        if trace is not None:
            trace.append({"objective": "task (student policy)", "controller": "student", "action": f"chunk {decision}: {exec_chunk} steps",
                          "result": f"native step {D.native_steps}; success {D.success()}", "next": "next chunk"})
        decision += 1


# ====================================================================== teacher segment (existing dex prompt / schema)
TAKEOVER_NOTE = ("NOTE: you are taking over control in the middle of an episode. Until now a learned policy controlled the robot; "
                 "the robot and the objects are in the state shown. The remaining physical budget (control steps) is given in BUDGET.")


def astra_segment(D, policy, history, budget, log, out_dir, task_cfg, step_budget, episode_id, takeover=False, student_records=None, trace=None):
    from run import save_png, wjson
    from run_dex import build_prompt
    from schema_dex import HANDS, Invalid, validate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = D.T
    obs = T._observation()
    prev_main, decision_index, status, decisions = None, 0, None, []
    policy.episode, policy.decision = episode_id, 0
    D.rec = log
    while True:
        if D.success():
            status = "success"
            break
        if D.native_steps >= step_budget:
            status = "horizon_reached"
            break
        if not budget.can_call():
            status = "budget_exhausted"
            break
        ddir = out_dir / f"decision_{decision_index:02d}"
        for n, px in obs["images"].items():
            save_png(ddir / f"pre_{n}.png", px)
        bview = dict(budget.view(1), steps_used_this_attempt=D.native_steps, steps_remaining_this_attempt=step_budget - D.native_steps)
        obs = dict(obs, steps_used=D.native_steps, steps_remaining=step_budget - D.native_steps)
        images = dict(obs["images"])
        note = ""
        if prev_main is not None:
            images["main_camera_before_previous_decision"] = prev_main
            note = " image 5 = 'main_camera' as it was BEFORE your previous decision (compare with image 1 to see what changed)."
        hv = history.view()
        if student_records:
            first = len(images) + 1
            for i, r in enumerate(student_records):
                images[f"student_record_{i + 1}_main_before"] = r["image_before"]
            note += (f" Images {first}-{first + len(student_records) - 1} = the fixed main camera (224x224, the learned policy's own input) at the "
                     f"START of the learned policy's last {len(student_records)} interaction records (see 'learned_policy_records' in the history).")
            hv = dict(hv, learned_policy_records=[{k: v for k, v in r.items() if k not in ("image_before",)} for r in student_records])
        prompt = build_prompt(task_cfg, obs, hv, bview, note)
        if takeover:
            prompt = prompt.replace("\nACTION INTERFACE", "\n" + TAKEOVER_NOTE + "\n\nACTION INTERFACE", 1)
        (ddir / "prompt.txt").write_text(prompt)
        wjson(ddir / "history_view.json", hv)
        decision, calls = None, []
        for tr in range(2):
            if not budget.can_call():
                calls.append({"try": tr, "skipped": "budget exhausted"})
                break
            budget.charge()
            raw, meta = policy.act(ddir / f"call_{tr}", images, prompt)
            rec = {"try": tr, "meta": {k: meta.get(k) for k in ("outcome", "latency_s", "usage", "returncode")}}
            if raw is None:
                rec.update(accepted=False, reason=meta.get("outcome"))
                calls.append(rec)
                if meta.get("outcome") == "budget_refused":
                    break
                continue
            try:
                clean, warnings = validate(raw, history.known_ids())
                rec.update(accepted=True, warnings=warnings)
                decision = clean
                calls.append(rec)
                break
            except Invalid as ex:
                rec.update(accepted=False, reason=f"invalid response: {ex}", raw=raw)
                calls.append(rec)
        drec = {"decision": decision_index, "steps_before": D.native_steps, "calls": calls, "accepted": decision is not None}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            decision_index += 1
            if any(c.get("reason") == "budget_refused" for c in calls):
                status = "budget_exhausted"
                break
            continue
        history.apply_memory_update(decision["memory_update"])
        base_rec = {"decision": decision_index, "steps_before": D.native_steps, "left": decision["left"], "right": decision["right"],
                    "steps": decision["steps"], "assessment_of_previous_action": decision["assessment_of_previous_action"],
                    "intent": decision["intent"], "hypothesis": decision["hypothesis"], "expected_change": decision["expected_change"],
                    "references": decision["references"]}
        if decision["stop"]:
            iid = history.add_interaction(dict(base_rec, stop=True, result={"note": "stop requested; nothing executed", "task_complete": False}))
            drec.update(decision_content=decision, interaction_id=iid, stop=True)
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            status = "model_stopped"
            if trace is not None:
                trace.append({"objective": decision["intent"], "controller": "astra", "action": "STOP", "result": "attempt ended", "next": "-"})
            break
        before_main = obs["images"]["main_camera"]
        obs, result = D.act_native(decision, step_budget)
        iid = history.add_interaction(dict(base_rec, stop=False, result=result))
        for n, px in obs["images"].items():
            save_png(ddir / f"post_{n}.png", px)
        drec.update(decision_content=decision, executed=True, interaction_id=iid, result=result, memory_after=history.running)
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        prev_main = before_main
        if trace is not None:
            trace.append({"objective": decision["intent"], "controller": "astra",
                          "action": "; ".join(f"{h}: {'hold' if not decision[h]['move'] else [round(v, 2) for v in decision[h]['position_m']]} "
                                              f"base {decision[h]['finger_base']:.2f}" for h in HANDS) + f"; {decision['steps']} steps",
                          "result": f"thumb-index {[result['hands'][h]['thumb_index_distance_cm'] for h in HANDS]} cm; contact "
                                    f"{[result['hands'][h]['index_contact_force_N'] for h in HANDS]} N; complete {result['task_complete']}",
                          "next": "next decision"})
        decision_index += 1
    D.rec = None
    return {"status": status, "decisions": len(decisions), "astra_calls": sum(1 for d in decisions for c in d["calls"] if "skipped" not in c),
            "success": D.success()}


# ====================================================================== activities
def _ledger(args):
    from s1s2.ledger import Ledger
    from s1s2.registry import BUDGET
    return Ledger(args["study_root"], BUDGET["astra_calls_global"], BUDGET["astra_calls_per_platform"])


def _epdir(root, cell, idx):
    d = Path(root) / "dexhand" / "episodes" / f"{cell}_{idx}_{int(time.time())}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _priv(D):
    p = D.T.privileged()
    return {k: p.get(k) for k in ("can_pos", "plate_pos", "drawer_joint_pos", "success")}


def activity_audit(platform, args, out):
    D = get_task()
    pl = pools()
    D.reset(pl["dev"][0])
    img = D.image224()
    desc = D.T.describe()
    manifest = {
        "platform": platform, "environment": "Isaac Sim 5.1 / IsaacLab 2.3, EgoVLA Ego-Humanoid benchmark", "task": TASK_ID,
        "instruction": desc["instruction"], "success_definition": "env success: can upright on the saucer (< 9 cm, < 2 cm) AND door >= 80 % closed",
        "reset": f"env.reset() with cfg.randomize_idx = start id (room {ROOM}, table {TABLE}), then {SETTLE} settle steps holding the default pose; "
                 "no demonstration-derived warm-up", "initial_state_ids": pl,
        "teacher_observation": "main 960x540, left eye, both wrist cameras, previous main image; wrist poses / directions, fingertips, "
                               "thumb-index distance, closure fractions, index contact force, arm joints, calibration hints, budgets, history",
        "student_observation": "fixed main camera (1280x720) resized to 224x224 INTER_AREA, RGB /255 + ImageNet norm; proprio zeroed (STATELESS)",
        "teacher_action": "wrist targets (position + finger/palm directions) interpolated <= 1 cm / 3 deg per step + 4 hand-closure fractions; "
                          "executed through the same native 38-D step as the student (IK holds for held arms)",
        "student_action": "38-D [L wrist xyz+quat wxyz, R wrist xyz+quat wxyz, L hand 12, R hand 12] absolute, world frame; IK (L DLS, R pinv) every step",
        "control_rate_hz": 30, "predicted_chunk": CHUNK, "executed_chunk": EXEC_CHUNK,
        "normalization": "per-dimension mean/std over the training set's timesteps (std >= 0.01), ACT convention; continuations keep the anchor's stats",
        "label_rule": "action_t = the 38-D command dispatched at step t after observing o_t; chunk = actions t..t+24 inside the same teacher segment, "
                      "zero-padded with is_pad beyond it (ACT convention); quaternion signs made continuous",
        "student_init": {"kind": "ACT from scratch (ImageNet ResNet-18 backbone as in the project's ACT)", "cohort": "distillation from scratch"},
        "astra": {"model": "gpt-6-astra", "reasoning": "medium"}, "budgets": {"native_steps_per_episode": STEP_BUDGET, "astra_calls_per_episode": 40},
        "recipe": RECIPE, "code_revision": _git_rev(), "checks": {"image224": list(img.shape), "calibration": desc["calibration"]["right"].get("knuckle_offset_along_fingers_cm")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return {"status": "OK", "teacher_status": "OK", "summary": f"pools from the protocol shuffle (room {ROOM}, table {TABLE}); student ACT from scratch",
            "cohort": "distillation from scratch (ACT, stateless)", "student": {"summary": "ACT trained from scratch on Astra episodes"},
            "pools": pl, "pretrained_ckpt": None, "init_ckpt": "SCRATCH", "manifest": str(out / "manifest.json"),
            "runtime": {"U": RECIPE["U"], "dU": RECIPE["dU"], "exec_chunk": EXEC_CHUNK, "source_calls": 70, "corr_calls": 25}}


def run_teacher_episode(args, idx, history, ledger, cell, call_cap=None, step_cap=None):
    from s1s2.teacher import LedgerBudget, LedgeredAstra
    from schema_dex import response_schema
    call_cap, step_cap = call_cap or RESET_CALLS, step_cap or STEP_BUDGET
    D = get_task()
    ep = _epdir(args["study_root"], cell, idx)
    D.reset(idx)
    log, trace = Log(), []
    policy = LedgeredAstra(ledger, "dexhand", cell, schema_fn=response_schema)
    budget = LedgerBudget(ledger, "dexhand", call_cap)
    budget.new_attempt()
    t0 = time.time()
    p0 = _priv(D)
    res = astra_segment(D, policy, history, budget, log, ep / "astra", D.T.describe(), step_cap, ep.name, trace=trace)
    log.save(ep / "transitions.npz")
    history.end_attempt({"attempt": len(history.attempt_summaries) + 1, "outcome": res["status"], "decisions_total": res["decisions"],
                         "steps_used": D.native_steps, "scene_index": idx})
    meta = {"episode_id": ep.name, "platform": "dexhand", "cell": cell, "pool": args.get("pool"), "start_id": idx, "success": D.success(),
            "status": res["status"], "native_steps": D.native_steps, "astra_calls": res["astra_calls"], "astra_decisions": res["decisions"],
            "policy_inference_calls": 0, "wall_s": round(time.time() - t0, 1), "privileged_start": p0, "privileged_end": _priv(D), "trace": trace,
            "valid_frames": log.steps("astra"), "call_ceiling": call_cap, "step_ceiling": step_cap}
    (ep / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta, ep


def activity_teacher(platform, args, out):
    from s1s2.datasets import append_bank, read_bank
    from s1s2.teacher import load_stream, save_stream
    ledger = _ledger(args)
    cell = args["cell"]
    stream = Path(args["study_root"]) / "dexhand" / f"stream_{cell}.json"
    history = load_stream(stream)
    done = [e["start_id"] for e in read_bank(args["study_root"], "dexhand") if e.get("cell") == cell]
    calls0 = ledger.summary()["per_cell"].get(f"dexhand/{cell}", 0)
    new, steps_used, first_valid, stop = [], 0, None, "pool exhausted"
    allot = args.get("call_allotment", 10 ** 9)
    for idx in args["starts"]:
        if idx in done:
            continue
        used = ledger.summary()["per_cell"].get(f"dexhand/{cell}", 0) - calls0
        if used >= allot:
            stop = "call allotment spent"
            break
        if args.get("step_allotment") and steps_used >= args["step_allotment"]:
            stop = "step allotment spent"
            break
        prior = sum(1 for e in read_bank(args["study_root"], "dexhand") if e.get("cell") == cell and e.get("success") and e.get("valid"))
        if prior >= args.get("target_successes", 10 ** 6):
            stop = "target successes reached"
            break
        call_cap, step_cap = RESET_CALLS, STEP_BUDGET
        if args.get("strict_allotment"):                   # E12 extra demos: the last episode is cut to the matched budget
            call_cap = min(call_cap, allot - used)
            if args.get("step_allotment"):
                step_cap = min(step_cap, args["step_allotment"] - steps_used)
        if ledger.remaining("dexhand") < call_cap:         # never let the study budget truncate an episode
            stop = "PENDING_BUDGET: remaining study calls below this episode's call ceiling"
            break
        meta, ep = run_teacher_episode(args, idx, history, ledger, cell, call_cap, step_cap)
        save_stream(stream, history)
        valid = meta["valid_frames"] > 0
        rec = {"episode_id": meta["episode_id"], "pool": args.get("pool"), "start_id": idx, "cell": cell, "success": meta["success"], "valid": valid,
               "invalid_reason": None if valid else "no teacher-controlled step", "native_steps": meta["native_steps"], "astra_calls": meta["astra_calls"],
               "path": str(ep), "call_ceiling": call_cap, "step_ceiling": step_cap}
        append_bank(args["study_root"], "dexhand", rec)
        new.append(rec)
        steps_used += meta["native_steps"]
        first_valid = first_valid or (str(ep) if valid else None)
        print(json.dumps({"episode": meta["episode_id"], "success": meta["success"], "status": meta["status"], "steps": meta["native_steps"],
                          "calls": meta["astra_calls"]}), flush=True)
    mine = [e for e in read_bank(args["study_root"], "dexhand") if e.get("cell") == cell]
    data = {"episodes": len(mine), "successes": sum(1 for e in mine if e["success"] and e["valid"]), "valid_episodes": sum(1 for e in mine if e["valid"]),
            "astra_calls": sum(e["astra_calls"] for e in mine), "native_steps": sum(e["native_steps"] for e in mine)}
    return {"status": "OK", "summary": f"{cell}: {len(new)} new episodes, stop: {stop}; stream totals {data}", "data": data,
            "first_valid_episode": first_valid, "episodes": new, "stop": stop}


_STUDENT = {}


def get_student(ckpt):
    if ckpt not in _STUDENT:
        _STUDENT.clear()
        _STUDENT[ckpt] = Student(ckpt)
    return _STUDENT[ckpt]


def _failure_stage(p):
    can, plate, door = p.get("can_pos"), p.get("plate_pos"), (p.get("drawer_joint_pos") or [0])[0]
    if not can:
        return "unknown"
    d = math.dist(can[:2], plate[:2]) if plate else None
    return f"can z {can[2]:.3f}, can-saucer xy {d:.3f} m, door {door:.2f} rad"


def activity_eval(platform, args, out):
    D = get_task()
    student = get_student(args["ckpt"])
    ex = int(args.get("exec_chunk") or EXEC_CHUNK)
    per = []
    for idx in args["starts"]:
        ep = _epdir(args["study_root"], args["cell"], idx)
        D.reset(idx)
        log, trace = Log(), []
        D.rec = log
        t0, c0 = time.time(), student.calls
        status, ndec, _ = student_segment(D, student, ex, STEP_BUDGET, trace=trace)
        D.rec = None
        log.save(ep / "transitions.npz")
        pv = _priv(D)
        meta = {"episode_id": ep.name, "cell": args["cell"], "pool": args.get("pool"), "start_id": idx, "ckpt": args["ckpt"], "exec_chunk": ex,
                "success": D.success(), "status": status, "native_steps": D.native_steps, "policy_inference_calls": student.calls - c0,
                "astra_calls": 0, "wall_s": round(time.time() - t0, 1), "privileged_end": pv, "trace": trace[-12:],
                "failure_stage": None if D.success() else _failure_stage(pv)}
        (ep / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
        per.append({k: meta[k] for k in ("start_id", "success", "status", "native_steps", "policy_inference_calls", "wall_s", "failure_stage")} | {"path": str(ep)})
        print(json.dumps(per[-1]), flush=True)
    n, k = len(per), sum(1 for e in per if e["success"])
    return {"status": "OK", "summary": f"{args['cell']} {k}/{n} (exec_chunk {ex})", "successes": k, "n": n, "per_episode": per, "out": str(out)}


def activity_hybrid(platform, args, out):
    from s1s2.datasets import append_bank, write_membership
    from s1s2.teacher import LedgerBudget, LedgeredAstra, fresh_history
    from schema_dex import response_schema
    D = get_task()
    student = get_student(args["ckpt"])
    ledger = _ledger(args)
    k_hist = int(args.get("history", 0))
    cell, collect = args["cell"], bool(args.get("collect"))
    from s1s2.datasets import read_bank
    calls0 = ledger.summary()["per_cell"].get(f"dexhand/{cell}", 0)
    done_corr = {e["start_id"] for e in read_bank(args["study_root"], "dexhand") if e.get("cell") == "CORR"} if collect else set()
    per, stop = [], "pool exhausted"
    for idx in args["starts"]:
        if idx in done_corr:
            continue                                       # resumed collection: never redo an episode that already ran
        if collect and ledger.summary()["per_cell"].get(f"dexhand/{cell}", 0) - calls0 >= args.get("call_allotment", 10 ** 9):
            stop = "call allotment spent"
            break
        if ledger.remaining("dexhand") < TAKEOVER_CALLS:   # never let the study budget truncate a takeover
            per.append({"start_id": idx, "status": "PENDING_BUDGET"})
            stop = "PENDING_BUDGET: remaining study calls below the takeover call ceiling"
            break
        ep = _epdir(args["study_root"], cell, idx)
        D.reset(idx)
        log, trace = Log(), []
        D.rec = log
        t0, c0 = time.time(), student.calls
        status, ndec, records = student_segment(D, student, EXEC_CHUNK, STEP_BUDGET, stop_at=STEP_BUDGET // 2, trace=trace)
        handover, astra = None, {"status": None, "astra_calls": 0, "decisions": 0}
        if status == "handover":
            handover = {"native_step": D.native_steps, "student_decisions": ndec, "privileged_at_handover": _priv(D),
                        "wrists_at_handover": [D.T._ee(s)[:3].round(4).tolist() for s in ("left", "right")]}
            policy = LedgeredAstra(ledger, "dexhand", cell, schema_fn=response_schema)
            budget = LedgerBudget(ledger, "dexhand", TAKEOVER_CALLS)
            budget.new_attempt()
            log.new_segment()
            astra = astra_segment(D, policy, fresh_history(), budget, log, ep / "astra", D.T.describe(), STEP_BUDGET, ep.name, takeover=True,
                                  student_records=records[-k_hist:] if k_hist else None, trace=trace)
        D.rec = None
        log.save(ep / "transitions.npz")
        pv = _priv(D)
        meta = {"episode_id": ep.name, "cell": cell, "pool": args.get("pool"), "start_id": idx, "ckpt": args["ckpt"], "history_records": k_hist,
                "takeover_call_ceiling": TAKEOVER_CALLS, "success": D.success(), "student_status": status, "astra_status": astra["status"], "handover": handover,
                "recovered_by_astra": bool(handover and D.success()), "native_steps": D.native_steps, "student_steps": log.steps("student"),
                "astra_steps": log.steps("astra"), "policy_inference_calls": student.calls - c0, "astra_calls": astra["astra_calls"],
                "astra_decisions": astra["decisions"], "wall_s": round(time.time() - t0, 1), "privileged_end": pv, "trace": trace[-16:],
                "failure_stage": None if D.success() else _failure_stage(pv)}
        (ep / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
        if collect:
            append_bank(args["study_root"], "dexhand", {"episode_id": ep.name, "pool": args.get("pool"), "start_id": idx, "cell": "CORR",
                                                         "success": meta["success"], "valid": meta["recovered_by_astra"],
                                                         "invalid_reason": None if meta["recovered_by_astra"] else "no successful Astra recovery",
                                                         "native_steps": meta["native_steps"], "student_steps": meta["student_steps"],
                                                         "astra_calls": meta["astra_calls"], "path": str(ep)})
        per.append({k: meta[k] for k in ("start_id", "success", "student_status", "astra_status", "recovered_by_astra", "native_steps",
                                         "student_steps", "astra_steps", "policy_inference_calls", "astra_calls", "wall_s", "failure_stage")}
                   | {"handover_step": handover and handover["native_step"], "path": str(ep)})
        print(json.dumps(per[-1]), flush=True)
    done = [e for e in per if e.get("status") != "PENDING_BUDGET"]
    n, k = len(done), sum(1 for e in done if e.get("success"))
    res = {"status": "OK" if done else "PENDING_BUDGET", "summary": f"{cell} {k}/{n} (history {k_hist}; takeover ceiling {TAKEOVER_CALLS}); stop: {stop}",
           "successes": k, "n": n, "per_episode": per, "astra_calls": sum(e.get("astra_calls", 0) for e in done),
           "native_steps": sum(e.get("native_steps", 0) for e in done), "out": str(out), "stop": stop, "takeover_call_ceiling": TAKEOVER_CALLS}
    if collect:                                            # totals over every correction episode of this platform (restart-safe)
        bank = [e for e in read_bank(args["study_root"], "dexhand") if e.get("cell") == "CORR"]
        corr = [e["episode_id"] for e in bank if e.get("valid")]
        res.update(n=len(bank), successes=sum(1 for e in bank if e.get("success")), astra_calls=sum(e["astra_calls"] for e in bank),
                   native_steps=sum(e["native_steps"] for e in bank), student_steps=sum(e.get("student_steps", 0) for e in bank),
                   usable_windows=len(corr))
        res["summary"] = (f"CORR: {len(bank)} adaptation episodes, {len(corr)} successful Astra recoveries, {res['astra_calls']} calls, "
                          f"{res['native_steps']} native steps ({res['student_steps']} student-prefix); stop: {stop}")
        res["dataset"] = write_membership(args["study_root"], "dexhand", "Dcorr", corr,
                                          "Astra-controlled portions of successful takeover episodes on the adaptation pool") if corr else None
    return res


def activity_replay(platform, args, out):
    ep = Path(args["episode"])
    meta = json.loads((ep / "episode.json").read_text())
    d = np.load(ep / "transitions.npz")
    D = get_task()
    D.reset(meta["start_id"])
    errs = []
    wr = []
    for t in range(len(d["source"])):
        if d["source"][t] != 0:
            continue
        D.apply38(d["action"][t], "astra", record=False)
        wr.append(np.concatenate([D.T._ee("left")[:3], D.T._ee("right")[:3]]))
        if D.success():
            break
    succ = D.success()
    pv = _priv(D)
    ref = meta.get("privileged_end") or {}
    can_err = math.dist(pv["can_pos"], ref["can_pos"]) if ref.get("can_pos") and pv.get("can_pos") else None
    door_err = abs((pv["drawer_joint_pos"] or [0])[0] - (ref.get("drawer_joint_pos") or [0])[0]) if ref else None
    # wrist commands are absolute: compare the commanded wrist positions with the achieved ones along the replay
    cmd = d["action"][d["source"] == 0][:len(wr)]
    trk = np.linalg.norm(np.asarray(wr)[:, :3] - cmd[:, 0:3], axis=1).tolist() + np.linalg.norm(np.asarray(wr)[:, 3:] - cmd[:, 7:10], axis=1).tolist()
    med = float(np.median(trk)) if trk else 0.0
    ok = med < 0.03
    rep = {"episode": meta["episode_id"], "recorded_success": meta["success"], "replay_success": succ, "steps": len(wr),
           "wrist_tracking_median_m": round(med, 4), "final_can_error_m": None if can_err is None else round(can_err, 4),
           "final_door_error_rad": None if door_err is None else round(door_err, 3), "criterion": "median wrist tracking error < 3 cm"}
    (out / "replay.json").write_text(json.dumps(rep, indent=1))
    return {"status": "OK" if ok else "BLOCKED_DATA",
            "summary": f"replay of {meta['episode_id']}: wrist tracking median {med * 100:.1f} cm; final can error {rep['final_can_error_m']} m; "
                       f"success recorded {meta['success']} / replayed {succ}", **rep}


# ---------------------------------------------------------------------- training
def _dataset_paths(root, name):
    from s1s2.datasets import read_bank
    bank = {e["episode_id"]: e for e in read_bank(root, "dexhand")}
    mem = json.loads((Path(root) / "dexhand" / "datasets" / f"{name}.json").read_text())
    return [bank[e]["path"] for e in mem["episodes"] if e in bank]


class ActData:
    def __init__(self, paths):
        self.eps = []
        for p in paths:
            d = np.load(Path(p) / "transitions.npz")
            src, seg = d["source"], d["segment"]
            anchors = [t for t in range(len(src)) if src[t] == 0]
            if anchors:
                self.eps.append({"imgs": d["imgs"], "act": d["action"], "src": src, "seg": seg, "anchors": np.asarray(anchors)})
        self.n_anchors = sum(len(e["anchors"]) for e in self.eps)

    def all_actions(self):
        return np.concatenate([e["act"][e["src"] == 0] for e in self.eps])

    def sample(self, n, rng):
        imgs, acts, pads = [], [], []
        for _ in range(n):
            e = self.eps[rng.integers(len(self.eps))]
            t = e["anchors"][rng.integers(len(e["anchors"]))]
            idx = [k for k in range(t, min(len(e["src"]), t + CHUNK)) if e["src"][k] == 0 and e["seg"][k] == e["seg"][t]]
            a = np.zeros((CHUNK, 38), np.float32)
            a[:len(idx)] = e["act"][idx]
            pad = np.ones(CHUNK, bool)
            pad[:len(idx)] = False
            imgs.append(e["imgs"][t])
            acts.append(a)
            pads.append(pad)
        return np.stack(imgs), np.stack(acts), np.stack(pads)


def activity_train(platform, args, out):
    import torch
    root = args["study_root"]
    names = args["datasets"]
    mix = args.get("mix") or [1.0 / len(names)] * len(names)
    datas = []
    for nm in names:
        paths = _dataset_paths(root, nm)
        ad = ActData(paths) if paths else None
        if not ad or ad.n_anchors == 0:
            return {"status": "BLOCKED_DATA", "summary": f"dataset {nm} has no teacher-controlled step"}
        datas.append(ad)
    torch.manual_seed(0)
    if args["init"] == "SCRATCH":
        policy = build_act()
        allacts = np.concatenate([d.all_actions() for d in datas])
        stats = {"action_mean": allacts.mean(0), "action_std": np.clip(allacts.std(0), 1e-2, None)}
    else:
        s = torch.load(args["init"], map_location="cuda", weights_only=False)
        policy = build_act()
        policy.load_state_dict(s["model"])
        stats = {k: np.asarray(v, np.float32) for k, v in s["stats"].items()}       # continuations keep the anchor's normalization
    policy.train()
    opt = policy.configure_optimizers()
    U = int(args["updates"])
    rng = np.random.default_rng(int(hashlib_key(args["cell"])))
    mean, std = torch.tensor(stats["action_mean"], device="cuda"), torch.tensor(stats["action_std"], device="cuda")
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    for step in range(U):
        parts = [d.sample(max(1, int(round(ACT_CFG["batch"] * w))), rng) for d, w in zip(datas, mix)]
        imgs = torch.from_numpy(np.concatenate([p[0] for p in parts])).permute(0, 3, 1, 2).float().div(255.0)[:, None].cuda()
        acts = (torch.from_numpy(np.concatenate([p[1] for p in parts])).cuda() - mean) / std
        pads = torch.from_numpy(np.concatenate([p[2] for p in parts])).cuda()
        q = torch.zeros(imgs.shape[0], 38, device="cuda")
        ld = policy(q, imgs, acts, pads)
        opt.zero_grad()
        ld["loss"].backward()
        if not torch.isfinite(ld["loss"]):
            return {"status": "FAILED", "summary": f"non-finite loss at update {step}"}
        opt.step()
        if step % 100 == 0 or step == U - 1:
            log.write(json.dumps({"update": step + 1, "loss": round(ld["loss"].item(), 5), "l1": round(ld["l1"].item(), 5),
                                  "kl": round(ld["kl"].item(), 5), "sec": round(time.time() - t0, 1)}) + "\n")
            log.flush()
    ckpt = out / "ckpt.pt"
    torch.save({"model": policy.state_dict(), "stats": {k: np.asarray(v) for k, v in stats.items()}, "updates": U, "init": args["init"],
                "datasets": names, "mix": mix}, ckpt)
    torch.save({"optimizer": opt.state_dict()}, out / "optimizer.pt")
    meta = {"datasets": {nm: {"episodes": len(d.eps), "anchors": d.n_anchors} for nm, d in zip(names, datas)}, "updates": U, "mix": mix,
            "init": args["init"], "recipe": RECIPE, "final_loss": round(ld["loss"].item(), 5), "train_s": round(time.time() - t0, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    return {"status": "OK", "summary": f"{args['cell']}: {U} updates on {names}; final loss {ld['loss'].item():.4f}", "ckpt": str(ckpt), **meta}


def hashlib_key(*xs):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)
