"""RoboTwin adapter of the System-1 / System-2 transfer study (place_container_plate, lift_pot).

Student (System 1): the Horsea FM policy (horsea/rt/policy.py RTFlowPolicy, base checkpoint step300000.pt trained on all 50
RoboTwin tasks, which include both study tasks -> cohort = pretrained-policy adaptation on SEEN tasks). Native interface:
3 cameras (head, left, right) resized to 120x160 (INTER_AREA), state = RoboTwin `joint_action/vector` (14-D: 6 commanded
joint targets + commanded gripper value per arm), CLIP ViT-B/32 instruction embedding; output a chunk of 16 future
`joint_action` vectors, the first 8 executed with `take_action(a, 'qpos')` (one native step each), receding horizon.

Teacher (System 2): Astra through the existing RoboTwin bridge (absolute end-effector targets executed by RoboTwin's planner,
`take_action(a, 'ee')`). Student-compatible labels are recorded exactly the way RoboTwin recorded its own expert demos
(task_config save_freq = 15 physics steps per frame): every 15th physics step during the planner's execution the environment's
own `get_obs()` gives the student observation and the commanded `joint_action` vector (joint drive targets + commanded gripper
values, robot.get_*_arm_jointState). Training sample: state = vector[t], actions = vector[t+1 .. t+16] padded with the last
frame of the same teacher-controlled segment (the student's own dataset convention), never crossing an episode or a
controller switch.

Physical budget: 400 native control steps per episode (RoboTwin's step limit for both tasks): one student `qpos` action = one
step; an Astra planner motion costs the number of 15-physics-step frames it took.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROBOTWIN = "/home/licho/workspace/RoboTwin"
HORSEA = "/home/licho/workspace/Horsea"
AE = HORSEA + "/astra_explore"
for _p in (AE, HORSEA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK = {"robotwin_single": "place_container_plate", "robotwin_bimanual": "lift_pot"}
POOL_BASE = {"source": 1810000, "dev": 1820000, "adapt": 1830000, "test": 1840000}
POOL_SIZE = {"source": 15, "dev": 5, "adapt": 10, "test": 10}
SAVE_FREQ = 15
STEP_BUDGET = 400
BASE_CKPT = HORSEA + "/experiments/rt/base/step300000.pt"
TASK_CONFIG = "astra_probe"          # the Astra bridge's config (Large_D435 head camera); used for EVERY cell of this platform
EXEC_CHUNK = 8                       # deploy_policy.yml execute_steps
RECIPE = {"source": "horsea/rt/continue_train.py", "optimizer": "AdamW(betas 0.95/0.999, eps 1e-8, wd 1e-6)", "lr": 1e-5,
          "warmup": 250, "batch": 64, "grad_clip": 100.0,
          "trained": "velocity_net decoder (decoder blocks, ac_proj, dec_pos, time_net except fixed freqs, eps_out)",
          "frozen": "vision (ResNet-18 + FiLM), language and state projections, camera embedding, DiT observation encoder",
          "augmentation": "horsea.rt.train.augment (colour jitter + translation)", "U": 1000, "dU": 500}

_ENV = {}


def _git_rev():
    import subprocess
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HORSEA).stdout.strip()


# ====================================================================== environment
def _rt_modules():
    os.chdir(ROBOTWIN)
    for p in ("script", "", "policy", "policy/HorseaFM", "description/utils"):
        q = os.path.join(ROBOTWIN, p)
        if q not in sys.path:
            sys.path.insert(0, q)
    import eval_policy as EP  # noqa: E402
    from fixed_scenes import build_args  # noqa: E402
    return EP, build_args


class StudyEnv:
    """One RoboTwin task env reused across seeds; Astra's RoboTwinTask observation/act code via composition."""

    def __init__(self, task):
        from robotwin_env import RoboTwinTask
        self.EP, build_args = _rt_modules()
        self.task = task
        self.args = build_args(task, TASK_CONFIG, "astra", policy_name="HorseaFM")
        self.args["render_freq"] = 0
        self.env = self.EP.class_decorator(task)
        # a RoboTwinTask shell (no expert run in __init__): reuse its observation / act / privileged code unchanged
        self.rt = RoboTwinTask.__new__(RoboTwinTask)
        self.rt.EP, self.rt.task, self.rt.args, self.rt.env = self.EP, task, self.args, self.env
        self.rt.open, self.rt.motions, self.rt.scene, self.rt._raw = False, 0, None, None
        self.rec_cb = None
        self.native_steps = 0

    # ---------------------------------------------------------------- admission (RoboTwin's expert check, privileged)
    def admit(self, seed):
        env = self.env
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
            info = env.play_once()
            ok = bool(env.plan_success and env.check_success())
        except Exception as ex:  # noqa: BLE001  unsolvable or broken scene -> not admitted
            env.close_env()
            return {"seed": seed, "admitted": False, "reason": f"{type(ex).__name__}: {ex}"[:200]}
        env.close_env()
        if not ok:
            return {"seed": seed, "admitted": False, "reason": "expert failed"}
        desc = self.EP.generate_episode_descriptions(self.task, [info["info"]], 1)[0]
        rng = np.random.default_rng(0)
        return {"seed": seed, "admitted": True, "instruction": str(rng.choice(desc["unseen"])), "seen": list(desc["seen"]),
                "scene_info": info["info"]}

    # ---------------------------------------------------------------- episodes
    def reset(self, seed, instruction):
        rt = self.rt
        if rt.open:
            self.env.close_env(clear_cache=True)
        self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
        self.env.set_instruction(instruction=instruction)
        self.env.eval_success = False
        from robotwin_env import CountedScene
        outer = self

        class RecordingScene(CountedScene):
            def step(self):
                r = CountedScene.step(self)
                if outer.rec_cb is not None:          # the callback itself applies the 15-step frame rule (local to the motion)
                    outer.rec_cb()
                return r
        rt.scene = RecordingScene(self.env.scene, self.env)
        self.env.scene = rt.scene
        rt.open, rt.motions, rt.seed, rt.instruction = True, 0, seed, instruction
        self.native_steps = 0
        return rt._observation()

    def student_obs(self, raw=None):
        import cv2
        from horsea.rt.policy import CAMS, IMG_HW
        raw = raw or self.env.get_obs()
        imgs = np.stack([cv2.resize(raw["observation"][c]["rgb"], (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA) for c in CAMS])
        return {"imgs": imgs.astype(np.uint8), "state": np.asarray(raw["joint_action"]["vector"], dtype=np.float32)}

    def ee6(self):
        return np.asarray(self.rt._pose("left")[:3] + self.rt._pose("right")[:3], np.float32)

    def command_vector(self):
        r = self.env.robot
        return np.asarray(r.get_left_arm_jointState() + r.get_right_arm_jointState(), dtype=np.float32)

    def success(self):
        return bool(self.env.eval_success or self.env.check_success())

    def close(self):
        if self.rt.open:
            self.env.close_env(clear_cache=True)
            self.rt.open = False


def get_env(platform):
    if platform not in _ENV:
        _ENV[platform] = StudyEnv(TASK[platform])
    return _ENV[platform]


SCENE_LIST = HORSEA + "/experiments/rt/graphB/scenes/T_{task}.json"   # 50 expert-admitted seeds per task (prevalidated)
POOL_SLICES = {"source": (0, 15), "dev": (15, 20), "adapt": (20, 30), "test": (30, 40)}


def pools(platform):
    d = json.loads(Path(SCENE_LIST.format(task=TASK[platform])).read_text())
    sc = d["scenes"]
    return {k: [x["seed"] for x in sc[a:b]] for k, (a, b) in POOL_SLICES.items()}, {x["seed"]: x for x in sc}, d


def seed_info(platform, seed):
    _, by_seed, _ = pools(platform)
    return by_seed[int(seed)]


# ====================================================================== recording
class FrameLog:
    """Student-native frames at the native step rate. Astra frames: the label of frame t is the commanded vector of frame
    t+1 inside the same teacher segment (RoboTwin / student dataset convention); a segment's last frame has no label.
    Student frames: the label is the executed qpos target (equal to the next frame's commanded vector)."""

    def __init__(self):
        self.imgs, self.state, self.action, self.valid, self.source, self.seg, self.ee = [], [], [], [], [], [], []
        self.cur_seg = 0

    def new_segment(self):
        self.cur_seg += 1

    def frame(self, obs, source, action=None, ee=None):
        if source == "astra" and self.source and self.source[-1] == 0 and self.seg[-1] == self.cur_seg:
            self.action[-1] = obs["state"].copy()          # previous astra frame's label = this frame's commanded vector
            self.valid[-1] = True
        self.imgs.append(obs["imgs"])
        self.state.append(obs["state"])
        self.action.append(np.full(14, np.nan, np.float32) if action is None else np.asarray(action, np.float32))
        self.valid.append(action is not None)
        self.source.append(0 if source == "astra" else 1)
        self.seg.append(self.cur_seg)
        self.ee.append(np.full(6, np.nan, np.float32) if ee is None else np.asarray(ee, np.float32))

    def native_steps(self, source=None):
        if source is None:
            return len(self.source)
        code = 0 if source == "astra" else 1
        return sum(1 for s in self.source if s == code)

    def save(self, path):
        np.savez_compressed(path, imgs=np.stack(self.imgs) if self.imgs else np.zeros((0, 3, 120, 160, 3), np.uint8),
                            state=np.stack(self.state) if self.state else np.zeros((0, 14), np.float32),
                            action=np.stack(self.action) if self.action else np.zeros((0, 14), np.float32),
                            valid=np.asarray(self.valid, bool), source=np.asarray(self.source, np.int8), segment=np.asarray(self.seg, np.int32),
                            ee=np.stack(self.ee) if self.ee else np.zeros((0, 6), np.float32))


# ====================================================================== student (System 1)
def _key(*xs):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:15], 16)


class Student:
    def __init__(self, ckpt, device="cuda:0"):
        import torch
        from horsea.rt.policy import RTFlowPolicy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        s = torch.load(ckpt, map_location=device, weights_only=False)
        self.policy = RTFlowPolicy(s["stats"]).to(device).eval()
        self.policy.load_state_dict(s["model"])
        self.policy.K = 10
        self.dev, self.torch = device, torch
        self.a_min, self.a_max = s["stats"]["a_min"], s["stats"]["a_max"]
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self._lang = {}
        self.calls = 0

    def lang(self, text):
        if text not in self._lang:
            with self.torch.no_grad():
                self._lang[text] = self.clip(**self.tok([text], return_tensors="pt", padding=True, truncation=True).to(self.dev)).text_embeds
        return self._lang[text]

    def act(self, obs, instruction, seed, task, decision, rep=0):
        torch = self.torch
        from horsea.rt.policy import ADIM, CHUNK
        imgs = torch.from_numpy(obs["imgs"]).to(self.dev).permute(0, 3, 1, 2).float()[None] / 255.0
        state = torch.from_numpy(obs["state"]).to(self.dev)[None]
        g = torch.Generator(device=self.dev).manual_seed(_key(0, task, seed, decision, rep))   # the deploy wrapper's keyed noise
        noise = torch.randn(1, CHUNK, ADIM, device=self.dev, generator=g)
        self.calls += 1
        with torch.no_grad():
            return self.policy.sample(imgs, state, self.lang(instruction), noise=noise)[0].cpu().numpy()


def student_segment(env, student, log, seed, instruction, task, exec_chunk, step_budget, stop_at=None, trace=None):
    """Closed-loop student from the current state until success, the native step budget, or the first chunk boundary at or
    after `stop_at` (handover). Returns (status, decisions, records) where records are E09's student interaction records."""
    decision, records = 0, []
    while True:
        if env.success():
            return "success", decision, records
        if env.native_steps >= step_budget:
            return "horizon", decision, records
        if stop_at is not None and env.native_steps >= stop_at:
            return "handover", decision, records
        obs0 = env.student_obs()
        chunk = student.act(obs0, instruction, seed, task, decision)
        rec = {"decision": decision, "native_step_before": env.native_steps, "state_before": obs0["state"].round(4).tolist(),
               "head_before": obs0["imgs"][0], "executed": []}
        obs = obs0
        for a in chunk[:exec_chunk]:
            log.frame(obs, "student", action=a, ee=env.ee6())   # pre-action observation + the command actually dispatched
            env.env.take_action(a, action_type="qpos")
            env.native_steps += 1
            rec["executed"].append(np.asarray(a).round(4).tolist())
            if env.success() or env.native_steps >= step_budget:
                break
            obs = env.student_obs()
        post = env.student_obs()
        rec.update(state_after=post["state"].round(4).tolist(), head_after=post["imgs"][0])
        records.append(rec)
        if trace is not None:
            trace.append({"objective": "task (student policy)", "controller": "student", "action": f"chunk {decision}: {len(rec['executed'])} qpos targets",
                          "result": f"native step {env.native_steps}; success {env.success()}", "next": "next chunk"})
        decision += 1


# ====================================================================== teacher (System 2) segment
TAKEOVER_NOTE = ("NOTE: you are taking over control in the middle of an episode. Until now a learned policy controlled the robot; "
                 "the robot and the objects are in the state shown. The remaining physical budget is given in BUDGET.")


def astra_segment(env, policy, history, budget, log, out_dir, task_cfg, step_budget, platform, episode_id, takeover=False,
                  student_records=None, trace=None):
    """Astra decides motions from the CURRENT state (no reset) until success, stop, call budget, or the native step budget.
    Every executed motion is recorded at the student rate by the scene hook. Returns a summary dict."""
    from run import save_png, wjson
    from run_rt import build_prompt
    from schema_rt import ARMS, Invalid, validate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = env.rt
    obs = rt._observation()
    prev_head, decision_index, status, decisions, model_time = None, 0, None, [], 0.0
    policy.episode, policy.decision = episode_id, 0
    extra_images, extra_note = {}, ""
    if student_records:
        # E09: the last three student interaction records (observations as robot state + head images, executed commands)
        for i, r in enumerate(student_records):
            extra_images[f"student_record_{i + 1}_head_before"] = r["head_before"]
        extra_note = (" Images {a}-{b} = head camera at the START of the learned policy's last {n} interaction records (see "
                      "'learned_policy_records' in the history).")
    while True:
        if env.success():
            status = "success"
            break
        if env.native_steps >= step_budget:
            status = "horizon_reached"
            break
        if not budget.can_call():
            status = "budget_exhausted"
            break
        ddir = out_dir / f"decision_{decision_index:02d}"
        for n, px in obs["images"].items():
            save_png(ddir / f"pre_{n}.png", px)
        bview = dict(budget.view(1), native_steps_used=env.native_steps, native_steps_remaining=step_budget - env.native_steps)
        images = dict(obs["images"])
        note = ""
        if prev_head is not None:
            images["head_camera_before_previous_motion"] = prev_head
            note = " image 5 = 'head_camera' as it was BEFORE your previous motion (compare with image 1 to see what changed)."
        hv = history.view()
        if student_records:
            first = len(images) + 1
            images.update(extra_images)
            note += extra_note.format(a=first, b=first + len(extra_images) - 1, n=len(extra_images))
            hv = dict(hv, learned_policy_records=[{k: v for k, v in r.items() if k not in ("head_before", "head_after")} for r in student_records])
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
            t0 = time.monotonic()
            raw, meta = policy.act(ddir / f"call_{tr}", images, prompt)
            model_time += time.monotonic() - t0
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
        drec = {"decision": decision_index, "native_steps_before": env.native_steps, "calls": calls, "accepted": decision is not None}
        if decision is None:
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            decision_index += 1
            if any(c.get("reason") == "budget_refused" for c in calls):
                status = "budget_exhausted"
                break
            continue
        history.apply_memory_update(decision["memory_update"])
        base_rec = {"decision": decision_index, "native_steps_before": env.native_steps, "left": decision["left"], "right": decision["right"],
                    "assessment_of_previous_action": decision["assessment_of_previous_action"], "intent": decision["intent"],
                    "hypothesis": decision["hypothesis"], "expected_change": decision["expected_change"], "references": decision["references"]}
        if decision["stop"]:
            iid = history.add_interaction(dict(base_rec, stop=True, result={"note": "stop requested; no motion executed", "task_complete": False}))
            drec.update(decision_content=decision, interaction_id=iid, stop=True)
            wjson(ddir / "decision.json", drec)
            decisions.append(drec)
            status = "model_stopped"
            if trace is not None:
                trace.append({"objective": decision["intent"], "controller": "astra", "action": "STOP", "result": "attempt ended", "next": "-"})
            break
        before_head = obs["images"]["head_camera"]
        # record at the student rate: frame before the motion, every 15 local physics steps, and after it (RoboTwin's recorder)
        log.new_segment() if not log.source or log.source[-1] != 0 else None
        log.frame(env.student_obs(), "astra", ee=env.ee6())
        steps0 = env.rt.scene.steps
        env.rt.scene.local0 = steps0

        def _cb():
            if (env.rt.scene.steps - steps0 - 1) % SAVE_FREQ == 0:
                log.frame(env.student_obs(), "astra", ee=env.ee6())
        env.rec_cb = _cb
        try:
            obs, result = rt.act(decision)              # executed exactly once per accepted decision
        finally:
            env.rec_cb = None
        log.frame(env.student_obs(), "astra", ee=env.ee6())
        env.native_steps = log.native_steps()
        iid = history.add_interaction(dict(base_rec, stop=False, result=result))
        for n, px in obs["images"].items():
            save_png(ddir / f"post_{n}.png", px)
        drec.update(decision_content=decision, executed=True, interaction_id=iid, result=result, native_steps_after=env.native_steps,
                    memory_after=history.running)
        wjson(ddir / "decision.json", drec)
        decisions.append(drec)
        prev_head = before_head
        if trace is not None:
            trace.append({"objective": decision["intent"], "controller": "astra",
                          "action": "; ".join(f"{a}: {'hold' if not decision[a]['move'] else [round(v, 3) for v in decision[a]['position_m']]} g{decision[a]['gripper']:.1f}" for a in ARMS),
                          "result": f"planner {[result['planner'][a]['status'] for a in ARMS]}; complete {result['task_complete']}; native step {env.native_steps}",
                          "next": "next decision"})
        decision_index += 1
    return {"status": status, "decisions": len(decisions), "astra_calls": sum(1 for d in decisions for c in d["calls"] if "skipped" not in c),
            "model_time_s": round(model_time, 1), "success": env.success()}


# ====================================================================== activities
def _task_cfg(platform, seed):
    si = seed_info(platform, seed)
    return {"instruction": si["instruction"], "task": TASK[platform]}


def _new_episode_dir(root, platform, cell, seed, tag=""):
    d = Path(root) / platform / "episodes" / f"{cell}_{seed}{('_' + tag) if tag else ''}_{int(time.time())}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ledger(args):
    from s1s2.ledger import Ledger
    from s1s2.registry import BUDGET
    return Ledger(args["study_root"], BUDGET["astra_calls_global"], BUDGET["astra_calls_per_platform"])


def activity_audit(platform, args, out):
    import torch
    pl, by_seed, meta = pools(platform)
    env = get_env(platform)
    seed = pl["dev"][0]
    o = env.reset(seed, by_seed[seed]["instruction"])
    so = env.student_obs()
    # camera parity: head image from this config (640x480 -> 120x160) vs the student's native 320x240 rendering is checked by
    # rendering the same frame at half resolution and resizing (same fovy 37, same aspect)
    import cv2
    raw = env.env.get_obs()
    head = raw["observation"]["head_camera"]["rgb"]
    half = cv2.resize(head, (320, 240), interpolation=cv2.INTER_AREA)
    via_half = cv2.resize(half, (160, 120), interpolation=cv2.INTER_AREA)
    direct = cv2.resize(head, (160, 120), interpolation=cv2.INTER_AREA)
    parity = float(np.abs(via_half.astype(np.int16) - direct.astype(np.int16)).mean())
    vec = env.command_vector()
    s = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    inside = bool(np.all(vec >= np.asarray(s["stats"]["a_min"]) - 1e-6) and np.all(vec <= np.asarray(s["stats"]["a_max"]) + 1e-6))
    env.close()
    import hashlib
    sha = hashlib.sha256(open(BASE_CKPT, "rb").read(1 << 24)).hexdigest()[:16]
    manifest = {
        "platform": platform, "environment": "RoboTwin 2.0 (SAPIEN 3.0.0b1)", "task": TASK[platform],
        "instruction_protocol": "per-seed stored unseen instruction (prevalidated T list), identical for teacher and student",
        "success_definition": "RoboTwin task check_success() (env.eval_success) within the 400-step limit",
        "reset": f"setup_demo(seed, is_test=True), task_config {TASK_CONFIG} (Large_D435 head camera, same fovy 37 as D435)",
        "initial_state_ids": {k: v for k, v in pl.items()}, "seed_source": SCENE_LIST.format(task=TASK[platform]),
        "teacher_observation": "head (640x480), front, left, right cameras; ee poses; finger separations; joint angles; calibration hints; budgets; history",
        "student_observation": "head, left, right camera RGB resized to 120x160 (INTER_AREA); state = joint_action vector (14); CLIP ViT-B/32 instruction",
        "teacher_action": "absolute ee targets per arm + gripper, executed by RoboTwin's planner (take_action 'ee', planner-assisted)",
        "student_action": "joint_action vector (14: 6 joint drive targets rad + gripper command [0 closed, 1 open] per arm), absolute; one take_action('qpos') per native step",
        "control_rate": "native step = one demo frame (15 physics steps at 250 Hz in recorded expert / Astra motions); qpos actions executed with TOPP",
        "normalization": "min-max to [-1, 1] with checkpoint stats (a_min/a_max over 50 tasks)", "predicted_chunk": 16, "executed_chunk": EXEC_CHUNK,
        "label_rule": "state = vector[t]; actions = vector[t+1..t+16] padded with the last frame of the same teacher segment",
        "planner": "cuRobo plan_path for Astra ee targets; TOPP (mplib) for student qpos targets",
        "student_init": {"path": BASE_CKPT, "sha256_16MiB": sha, "trained_on": "RoboTwin aloha-agilex_clean_50, 50 tasks incl. this one (demos 0-44)",
                         "cohort": "pretrained-policy adaptation on a SEEN task"},
        "astra": {"model": "gpt-6-astra", "reasoning": "medium", "cli": "codex exec (sandboxed, existing integration)"},
        "budgets": {"native_steps_per_episode": STEP_BUDGET, "astra_calls_per_episode": 40}, "recipe": RECIPE, "code_revision": _git_rev(),
        "checks": {"camera_parity_mean_abs_px(640->160 vs 640->320->160)": round(parity, 3), "command_vector_inside_stats": inside,
                   "student_obs_shapes": {k: list(v.shape) for k, v in so.items()}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = inside and so["imgs"].shape == (3, 120, 160, 3)
    return {"status": "OK" if ok else "BLOCKED_INTERFACE", "teacher_status": "OK", "summary": f"pools from {Path(SCENE_LIST.format(task=TASK[platform])).name}; camera parity {parity:.2f} px; vector inside stats {inside}",
            "cohort": "pretrained-policy adaptation (seen task)", "student": {"summary": f"Horsea FM base step300000 ({sha})", "path": BASE_CKPT},
            "pools": pl, "pretrained_ckpt": BASE_CKPT, "init_ckpt": BASE_CKPT, "manifest": str(out / "manifest.json"),
            "runtime": {"U": RECIPE["U"], "dU": RECIPE["dU"], "exec_chunk": EXEC_CHUNK, "source_calls": 60, "corr_calls": 25}}


def run_teacher_episode(platform, args, seed, history, ledger, cell):
    from s1s2.teacher import LedgerBudget, LedgeredAstra
    from schema_rt import response_schema
    env = get_env(platform)
    epdir = _new_episode_dir(args["study_root"], platform, cell, seed)
    env.reset(seed, seed_info(platform, seed)["instruction"])
    log = FrameLog()
    trace = []
    policy = LedgeredAstra(ledger, platform, cell, schema_fn=response_schema)
    budget = LedgerBudget(ledger, platform, 40)
    budget.new_attempt()
    t0 = time.time()
    priv0 = env.rt.privileged()
    res = astra_segment(env, policy, history, budget, log, epdir / "astra", _task_cfg(platform, seed), STEP_BUDGET, platform, epdir.name, trace=trace)
    success = env.success()
    priv1 = env.rt.privileged()
    log.save(epdir / "transitions.npz")
    summary = {"attempt": len(history.attempt_summaries) + 1, "outcome": res["status"], "decisions_total": res["decisions"],
               "native_steps": env.native_steps, "seed": seed}
    history.end_attempt(summary)
    meta = {"episode_id": epdir.name, "platform": platform, "cell": cell, "pool": args.get("pool"), "start_id": seed,
            "controller_timeline": [{"from": 0, "to": env.native_steps, "source": "astra"}], "success": success, "status": res["status"],
            "native_steps": env.native_steps, "astra_calls": res["astra_calls"], "astra_decisions": res["decisions"], "policy_inference_calls": 0,
            "wall_s": round(time.time() - t0, 1), "privileged_start": priv0, "privileged_end": priv1, "trace": trace,
            "valid_frames": int(np.sum(np.asarray(log.valid))), "frames": len(log.source)}
    (epdir / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta, epdir


def activity_teacher(platform, args, out):
    """Astra-from-reset episodes on a pool, in order, as one continuing teacher stream (Astra's own bounded memory carries
    across the stream's episodes, as in the pilots). Stops at target successes, call allotment, or step allotment."""
    from s1s2.datasets import append_bank, read_bank
    from s1s2.teacher import load_stream, save_stream
    ledger = _ledger(args)
    cell = args["cell"]
    stream_path = Path(args["study_root"]) / platform / f"stream_{cell}.json"
    history = load_stream(stream_path)
    bank = read_bank(args["study_root"], platform)
    done_starts = [e["start_id"] for e in bank if e.get("cell") == cell]
    calls0 = ledger.summary()["per_cell"].get(f"{platform}/{cell}", 0)
    episodes, succ, steps_used, first_valid = [], 0, 0, None
    for seed in args["starts"]:
        if seed in done_starts:
            continue                                       # resumed activity: never redo an episode that already ran
        used = ledger.summary()["per_cell"].get(f"{platform}/{cell}", 0) - calls0
        if used >= args.get("call_allotment", 10 ** 9) or not ledger.can_call(platform):
            break
        if args.get("step_allotment") and steps_used >= args["step_allotment"]:
            break
        prior = sum(1 for e in read_bank(args["study_root"], platform) if e.get("cell") == cell and e.get("success") and e.get("valid"))
        if prior >= args.get("target_successes", 10 ** 6):
            break
        meta, epdir = run_teacher_episode(platform, args, seed, history, ledger, cell)
        save_stream(stream_path, history)
        valid = meta["valid_frames"] > 0
        rec = {"episode_id": meta["episode_id"], "pool": args.get("pool"), "start_id": seed, "cell": cell, "success": meta["success"],
               "valid": valid, "invalid_reason": None if valid else "no labelled teacher frame", "native_steps": meta["native_steps"],
               "astra_calls": meta["astra_calls"], "path": str(epdir)}
        append_bank(args["study_root"], platform, rec)
        episodes.append(rec)
        succ += int(meta["success"])
        steps_used += meta["native_steps"]
        first_valid = first_valid or (str(epdir) if valid else None)
        print(json.dumps({"episode": meta["episode_id"], "success": meta["success"], "status": meta["status"], "steps": meta["native_steps"],
                          "calls": meta["astra_calls"]}), flush=True)
    bank = read_bank(args["study_root"], platform)
    mine = [e for e in bank if e.get("cell") == cell]
    data = {"episodes": len(mine), "successes": sum(1 for e in mine if e["success"] and e["valid"]),
            "valid_episodes": sum(1 for e in mine if e["valid"]), "astra_calls": sum(e["astra_calls"] for e in mine),
            "native_steps": sum(e["native_steps"] for e in mine)}
    return {"status": "OK", "summary": f"{cell}: {len(episodes)} new episodes ({succ} successes); stream totals {data}", "data": data,
            "first_valid_episode": first_valid, "episodes": episodes}


_STUDENT = {}


def get_student(ckpt):
    if ckpt not in _STUDENT:
        _STUDENT.clear()
        _STUDENT[ckpt] = Student(ckpt)
    return _STUDENT[ckpt]


def activity_eval(platform, args, out):
    """Standalone student (Astra disabled, no teacher history, no online updates) on the given starts."""
    env = get_env(platform)
    student = get_student(args["ckpt"])
    exec_chunk = int(args.get("exec_chunk") or EXEC_CHUNK)
    per = []
    for seed in args["starts"]:
        epdir = _new_episode_dir(args["study_root"], platform, args["cell"], seed)
        env.reset(seed, seed_info(platform, seed)["instruction"])
        log, trace = FrameLog(), []
        t0, calls0 = time.time(), student.calls
        status, decisions, _ = student_segment(env, student, log, seed, seed_info(platform, seed)["instruction"], TASK[platform],
                                               exec_chunk, STEP_BUDGET, trace=trace)
        priv = env.rt.privileged()
        log.save(epdir / "transitions.npz")
        meta = {"episode_id": epdir.name, "cell": args["cell"], "pool": args.get("pool"), "start_id": seed, "ckpt": args["ckpt"],
                "exec_chunk": exec_chunk, "success": env.success(), "status": status, "native_steps": env.native_steps,
                "policy_inference_calls": student.calls - calls0, "astra_calls": 0, "wall_s": round(time.time() - t0, 1),
                "privileged_end": priv, "trace": trace[-12:],
                "failure_stage": None if env.success() else _failure_stage(platform, priv)}
        (epdir / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
        per.append({k: meta[k] for k in ("start_id", "success", "status", "native_steps", "policy_inference_calls", "wall_s", "failure_stage")} | {"path": str(epdir)})
        print(json.dumps(per[-1]), flush=True)
    n, k = len(per), sum(1 for e in per if e["success"])
    return {"status": "OK", "summary": f"{args['cell']} {k}/{n} (exec_chunk {exec_chunk})", "successes": k, "n": n, "per_episode": per, "out": str(out)}


def _failure_stage(platform, priv):
    """Coarse, evidence-only stage from the evaluation-only record (never shown to any policy)."""
    if platform == "robotwin_single":
        c, p = priv.get("container_pos"), priv.get("plate_pos")
        if not c:
            return "unknown"
        moved = priv.get("container_pos")[2] > 0.80
        d = math.dist(c[:2], p[:2]) if p else None
        return f"container z {c[2]:.3f}, xy distance to plate {d:.3f} m" if d is not None else f"container z {c[2]:.3f}"
    pot = priv.get("pot_pos")
    return f"pot z {pot[2]:.3f}" if pot else "unknown"


def activity_hybrid(platform, args, out):
    """E08 / E09 evaluation, or adaptation-pool correction collection (cell CORR): the frozen student acts until the first chunk
    boundary at or after half the native step budget, then ONE handover to Astra (fresh session; E09 adds the last three
    student interaction records) for the remaining budget. No reset at handover; no unexecuted student tail is kept."""
    from s1s2.datasets import append_bank
    from s1s2.teacher import LedgerBudget, LedgeredAstra, fresh_history
    from schema_rt import response_schema
    env = get_env(platform)
    student = get_student(args["ckpt"])
    ledger = _ledger(args)
    hist_k = int(args.get("history", 0))
    cell = args["cell"]
    collect = bool(args.get("collect"))
    calls0 = ledger.summary()["per_cell"].get(f"{platform}/{cell}", 0)
    per, corr_eps = [], []
    for seed in args["starts"]:
        if collect and ledger.summary()["per_cell"].get(f"{platform}/{cell}", 0) - calls0 >= args.get("call_allotment", 10 ** 9):
            break
        if not ledger.can_call(platform):
            per.append({"start_id": seed, "status": "PENDING_BUDGET"})
            break
        instr = seed_info(platform, seed)["instruction"]
        epdir = _new_episode_dir(args["study_root"], platform, cell, seed)
        env.reset(seed, instr)
        log, trace = FrameLog(), []
        t0, calls_s0 = time.time(), student.calls
        handover_at = STEP_BUDGET // 2
        status, decisions, records = student_segment(env, student, log, seed, instr, TASK[platform], EXEC_CHUNK, STEP_BUDGET,
                                                     stop_at=handover_at, trace=trace)
        handover = None
        astra = {"status": None, "astra_calls": 0, "decisions": 0}
        if status == "handover":
            priv_h = env.rt.privileged()
            handover = {"native_step": env.native_steps, "student_decisions": decisions, "privileged_at_handover": priv_h,
                        "state_at_handover": env.command_vector().round(4).tolist()}
            policy = LedgeredAstra(ledger, platform, cell, schema_fn=response_schema)
            budget = LedgerBudget(ledger, platform, 40)
            budget.new_attempt()
            hist = fresh_history()                       # explicit, empty: no hidden student or teacher context
            recs = records[-hist_k:] if hist_k else None
            log.new_segment()
            astra = astra_segment(env, policy, hist, budget, log, epdir / "astra", {"instruction": instr, "task": TASK[platform]},
                                  STEP_BUDGET, platform, epdir.name, takeover=True, student_records=recs, trace=trace)
        priv = env.rt.privileged()
        log.save(epdir / "transitions.npz")
        meta = {"episode_id": epdir.name, "cell": cell, "pool": args.get("pool"), "start_id": seed, "ckpt": args["ckpt"], "history_records": hist_k,
                "success": env.success(), "student_status": status, "astra_status": astra["status"], "handover": handover,
                "recovered_by_astra": bool(handover and env.success()), "native_steps": env.native_steps,
                "student_steps": log.native_steps("student"), "astra_steps": log.native_steps("astra"),
                "policy_inference_calls": student.calls - calls_s0, "astra_calls": astra["astra_calls"], "astra_decisions": astra["decisions"],
                "wall_s": round(time.time() - t0, 1), "privileged_end": priv, "trace": trace[-16:],
                "failure_stage": None if env.success() else _failure_stage(platform, priv)}
        (epdir / "episode.json").write_text(json.dumps(meta, indent=1, default=str))
        if collect:
            append_bank(args["study_root"], platform, {"episode_id": epdir.name, "pool": args.get("pool"), "start_id": seed, "cell": "CORR",
                                                       "success": meta["success"], "valid": bool(meta["recovered_by_astra"]),
                                                       "invalid_reason": None if meta["recovered_by_astra"] else "no successful Astra recovery",
                                                       "native_steps": meta["native_steps"], "astra_calls": meta["astra_calls"], "path": str(epdir)})
            if meta["recovered_by_astra"]:
                corr_eps.append(epdir.name)
        per.append({k: meta[k] for k in ("start_id", "success", "student_status", "astra_status", "recovered_by_astra", "native_steps",
                                         "student_steps", "astra_steps", "policy_inference_calls", "astra_calls", "wall_s", "failure_stage")}
                   | {"handover_step": handover and handover["native_step"], "path": str(epdir)})
        print(json.dumps(per[-1]), flush=True)
    done = [e for e in per if e.get("status") != "PENDING_BUDGET"]
    n, k = len(done), sum(1 for e in done if e.get("success"))
    res = {"status": "OK" if done else "PENDING_BUDGET", "summary": f"{cell} {k}/{n} (history {hist_k})", "successes": k, "n": n, "per_episode": per,
           "astra_calls": sum(e.get("astra_calls", 0) for e in done), "native_steps": sum(e.get("native_steps", 0) for e in done), "out": str(out)}
    if collect:
        from s1s2.datasets import write_membership
        res["usable_windows"] = len(corr_eps)
        res["dataset"] = write_membership(args["study_root"], platform, "Dcorr", corr_eps,
                                          "Astra-controlled portions of successful takeover episodes on the adaptation pool") if corr_eps else None
    return res


def activity_replay(platform, args, out):
    """Replay the recorded native commands (one take_action('qpos') per recorded frame) from the recorded initial state and
    compare end-effector positions frame by frame and the final task outcome."""
    epdir = Path(args["episode"])
    meta = json.loads((epdir / "episode.json").read_text())
    d = np.load(epdir / "transitions.npz")
    env = get_env(platform)
    seed = meta["start_id"]
    env.reset(seed, seed_info(platform, seed)["instruction"])
    errs = []
    T = len(d["state"])
    for t in range(1, T):
        if d["source"][t] != 0:
            continue
        env.env.take_action(d["state"][t], action_type="qpos")
        if not np.any(np.isnan(d["ee"][t])):
            errs.append(float(np.linalg.norm(env.ee6()[:3] - d["ee"][t][:3])) / 1.0)
            errs.append(float(np.linalg.norm(env.ee6()[3:] - d["ee"][t][3:])))
        if env.success():
            break
    succ = env.success()
    errs = np.asarray(errs) if errs else np.zeros(1)
    med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    ok = med < 0.02
    rep = {"episode": meta["episode_id"], "recorded_success": meta["success"], "replay_success": succ, "frames": T,
           "ee_error_median_m": round(med, 4), "ee_error_p90_m": round(p90, 4), "criterion": "median end-effector deviation < 2 cm"}
    (out / "replay.json").write_text(json.dumps(rep, indent=1))
    return {"status": "OK" if ok else "BLOCKED_DATA",
            "summary": f"replay of {meta['episode_id']}: ee deviation median {med * 100:.1f} cm, p90 {p90 * 100:.1f} cm; success recorded {meta['success']} / replayed {succ}",
            **rep}


# ====================================================================== training (existing continuation recipe)
def _dataset_episodes(root, platform, name):
    from s1s2.datasets import read_bank
    bank = {e["episode_id"]: e for e in read_bank(root, platform)}
    mem = json.loads((Path(root) / platform / "datasets" / f"{name}.json").read_text())
    return [bank[e]["path"] for e in mem["episodes"] if e in bank]


class EpisodeData:
    """Anchors and chunks from recorded episodes: teacher-controlled frames with a label; chunk = commanded vectors of the
    next 16 frames inside the same teacher segment, padded with that segment's last frame (student dataset convention)."""

    def __init__(self, paths, platform, clip):
        from horsea.rt.policy import CHUNK
        EP, _ = _rt_modules()
        self.eps = []
        for p in paths:
            meta = json.loads((Path(p) / "episode.json").read_text())
            d = np.load(Path(p) / "transitions.npz")
            src, seg, valid = d["source"], d["segment"], d["valid"]
            anchors = [t for t in range(len(src)) if src[t] == 0 and valid[t]]
            if not anchors:
                continue
            si = seed_info(platform, meta["start_id"])
            seen = EP.generate_episode_descriptions(TASK[platform], [si["info"]], 1)[0]["seen"]
            clip_map = clip(list(seen))
            chunks = []
            for t in anchors:
                nxt = [k for k in range(t + 1, min(len(src), t + 1 + CHUNK)) if src[k] == 0 and seg[k] == seg[t]]
                a = d["state"][nxt] if nxt else d["state"][t:t + 1]
                if len(a) < CHUNK:
                    a = np.concatenate([a, np.repeat(a[-1:], CHUNK - len(a), 0)])
                chunks.append(a.astype(np.float32))
            self.eps.append({"imgs": d["imgs"], "state": d["state"], "anchors": np.asarray(anchors), "chunks": np.stack(chunks),
                             "lang": [clip_map[s] for s in seen], "id": meta["episode_id"]})
        self.n_anchors = sum(len(e["anchors"]) for e in self.eps)

    def sample(self, n, rng):
        out = {"imgs": [], "state": [], "actions": [], "lang": []}
        for _ in range(n):
            e = self.eps[rng.integers(len(self.eps))]                  # episodes uniformly, then valid anchors
            j = rng.integers(len(e["anchors"]))
            t = e["anchors"][j]
            out["imgs"].append(e["imgs"][t])
            out["state"].append(e["state"][t])
            out["actions"].append(e["chunks"][j])
            out["lang"].append(e["lang"][rng.integers(len(e["lang"]))])
        return out


def activity_train(platform, args, out):
    import torch
    from horsea.rt.continue_train import set_modes, trainable
    from horsea.rt.data import clip_cache
    from horsea.rt.policy import RTFlowPolicy
    from horsea.rt.train import augment
    root = args["study_root"]
    names = args["datasets"]
    mix = args.get("mix") or [1.0 / len(names)] * len(names)
    datas = []
    for nm in names:
        paths = _dataset_episodes(root, platform, nm)
        if not paths:
            return {"status": "BLOCKED_DATA", "summary": f"dataset {nm} is empty"}
        ed = EpisodeData(paths, platform, lambda texts: clip_cache(texts))
        if ed.n_anchors == 0:
            return {"status": "BLOCKED_DATA", "summary": f"dataset {nm} has no valid teacher-controlled anchor"}
        datas.append(ed)
    dev = "cuda:0"
    s = torch.load(args["init"], map_location=dev, weights_only=False)
    model = RTFlowPolicy(s["stats"]).to(dev)
    model.load_state_dict(s["model"])
    tnames, params = trainable(model)
    set_modes(model)
    opt = torch.optim.AdamW(params, lr=RECIPE["lr"], betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    U, B = int(args["updates"]), RECIPE["batch"]
    rng = np.random.default_rng(_key(0, platform, args["cell"], "data") % 2 ** 32)
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    for step in range(U):
        for g in opt.param_groups:
            g["lr"] = RECIPE["lr"] * min(1.0, (step + 1) / RECIPE["warmup"])
        parts = [d.sample(int(round(B * w)), rng) for d, w in zip(datas, mix)]
        batch = {k: sum((pt[k] for pt in parts), []) for k in parts[0]}
        imgs = torch.from_numpy(np.stack(batch["imgs"])).to(dev).permute(0, 1, 4, 2, 3).float() / 255.0
        gen_aug = torch.Generator(device=dev).manual_seed(_key(0, args["cell"], "aug", step))
        gen_fm = torch.Generator(device=dev).manual_seed(_key(0, args["cell"], "fm", step))
        torch.cuda.manual_seed(_key(0, args["cell"], "dropout", step) % 2 ** 63)
        imgs = augment(imgs, gen_aug)
        state = torch.from_numpy(np.stack(batch["state"])).to(dev)
        actions = torch.from_numpy(np.stack(batch["actions"])).to(dev)
        lang = torch.stack([torch.as_tensor(l).float() for l in batch["lang"]]).to(dev)
        loss = model.loss(imgs, state, lang, actions, gen=gen_fm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, RECIPE["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            return {"status": "FAILED", "summary": f"non-finite loss / gradient at update {step}"}
        opt.step()
        if step % 50 == 0 or step == U - 1:
            log.write(json.dumps({"update": step + 1, "loss": round(loss.item(), 6), "grad_norm": round(float(gn), 4),
                                  "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}) + "\n")
            log.flush()
    ckpt = out / "ckpt.pt"
    torch.save({"model": model.state_dict(), "stats": s["stats"], "step": U, "init": args["init"], "datasets": names, "mix": mix,
                "trainable": tnames}, ckpt)
    torch.save({"optimizer": opt.state_dict(), "rng": rng.bit_generator.state}, out / "optimizer.pt")
    meta = {"datasets": {nm: {"episodes": len(d.eps), "anchors": d.n_anchors, "ids": [e["id"] for e in d.eps]} for nm, d in zip(names, datas)},
            "updates": U, "batch": B, "mix": mix, "init": args["init"], "recipe": RECIPE, "final_loss": round(loss.item(), 6),
            "train_s": round(time.time() - t0, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    return {"status": "OK", "summary": f"{args['cell']}: {U} updates on {names} ({', '.join(str(d.n_anchors) for d in datas)} anchors); final loss {loss.item():.4f}",
            "ckpt": str(ckpt), **meta}


def shutdown():
    for e in _ENV.values():
        try:
            e.close()
        except Exception:  # noqa: BLE001
            pass
