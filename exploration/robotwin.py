"""RoboTwin sensor/capability engineering audit, separate from online learning.

The task expert is privileged and is used ONLY to audit capabilities and record
training-side reference skills. Its success never enters an online memory.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np

from .protocol import Protocol, atomic_json


DEFAULT_ROOT = Path.home() / "workspace/RoboTwin"


def load_api(root):
    root = Path(root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    path = root / "scripts/eval_policy_xpolicylab.py"
    spec = importlib.util.spec_from_file_location("exploration_robotwin_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(api, out):
    cfg, _ = api.load_task_args({"task_name": "open_microwave", "task_config": "demo_clean",
                                 "policy_name": "exploration"})
    cfg.update(eval_mode=True, collect_data=False, save_data=False, render_freq=0,
               eval_video_log=False, eval_video_save_dir=None, save_path=str(Path(out).resolve()), save_freq=15)
    cfg["data_type"].update(rgb=True, depth=True, endpose=True, qpos=True, pointcloud=False,
                             mesh_segmentation=False, actor_segmentation=False)
    return cfg


def sensor_record(obs):
    """Whitelist camera measurements and robot proprioception; discard every other field."""
    c = obs["observation"]["head_camera"]
    rgb, depth = np.asarray(c["rgb"]), np.asarray(c["depth"])
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or rgb.std() < 1:
        raise ValueError("Missing or blank RGB camera")
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or not (depth > 0).any():
        raise ValueError("Missing or invalid depth camera")
    end = obs["endpose"]
    return {"rgb": rgb.copy(), "depth_m": depth.astype(np.float32) / 1000.,
            "intrinsic": np.asarray(c["intrinsic_cv"]).copy(),
            "extrinsic": np.asarray(c["extrinsic_cv"]).copy(),
            "left_eef": np.asarray(end["left_endpose"]).copy(),
            "right_eef": np.asarray(end["right_endpose"]).copy(),
            "left_gripper": float(end["left_gripper"]),
            "right_gripper": float(end["right_gripper"]),
            "qpos": np.asarray(obs["joint_action"]["vector"]).copy()}


class CountedScene:
    def __init__(self, scene):
        self.scene, self.steps = scene, 0

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        self.steps += 1
        return self.scene.step()


def audit(args):
    import torch
    import imageio.v2 as imageio
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.seed in Protocol().confirmation:
        raise ValueError("Engineering audits must not open confirmation configurations")
    x = torch.randn(512, 512, device="cuda:0")
    gpu = {"name": torch.cuda.get_device_name(), "matmul_finite": bool(torch.isfinite(x @ x).all())}
    atomic_json(out / "gpu.json", gpu)
    api = load_api(args.robotwin)
    cfg = configuration(api, out)
    atomic_json(out / "configuration.json", cfg)
    env = api.class_decorator("open_microwave")
    records, boundaries = [], []
    try:
        random.seed(args.seed)
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **cfg)
        env.scene = CountedScene(env.scene)

        def capture():
            rec = sensor_record(env.get_obs())
            rec["physics_steps"] = env.scene.steps
            records.append(rec)

        capture()
        np.savez_compressed(out / "reset_sensors.npz", **records[0])
        imageio.imwrite(out / "reset_rgb.png", records[0]["rgb"])
        initial_success = bool(env.check_success())
        if initial_success:
            raise RuntimeError("Engineering configuration starts solved")
        if args.expert:
            original_move = env.move

            def move(*a, **kw):
                result = original_move(*a, **kw)
                capture()
                boundaries.append({"frame": len(records) - 1, "physics_steps": env.scene.steps})
                return result

            env.move = move
            env._take_picture = capture
            env.play_once()
            capture()
        success = bool(env.check_success())
        if records:
            np.savez_compressed(out / "sensor_sequence.npz", **{
                k: np.asarray([r[k] for r in records]) for k in records[0]})
            imageio.mimsave(out / "audit.mp4", [r["rgb"] for r in records], fps=10, macro_block_size=None)
        # Privileged values are segregated and explicitly named, including the expert's access.
        atomic_json(out / "evaluation_only.json", {
            "seed": args.seed, "simulator_success": success, "expert_requested": args.expert,
            "expert_plan_success": bool(env.plan_success),
            "privileged_access": ["check_success", "microwave joint state for audit",
                                   *(["task expert contact points and actor geometry"] if args.expert else [])],
            "joint": np.asarray(env.microwave.get_qpos()).tolist(),
            "limits": np.asarray(env.microwave.get_qlimits()).tolist()})
        result = {"gpu": gpu, "seed": args.seed, "sensor_frames": len(records),
                  "enabled_sensors": cfg["data_type"], "physics_steps": env.scene.steps,
                  "move_boundaries": boundaries, "sensor_audit_passed": True,
                  "expert_capability_passed": bool(args.expert and success and env.plan_success),
                  "option_capability_passed": False, "feedback_validated": False,
                  "confirmation_allowed": False,
                  "note": "Expert success alone does not validate staged sensor-only options"}
        atomic_json(out / "audit.json", result)
        print("EXPLORATION_SENSOR_AUDIT", result, flush=True)
    finally:
        env.close_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("--robotwin", type=Path, default=Path(os.environ.get("EXPLORATION_ROBOTWIN", DEFAULT_ROOT)))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=210000)
    parser.add_argument("--expert", action="store_true")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
