"""Training-side skill recording in the explicitly instrumented environment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from .instrumentation import add_cameras, fixed_asset_environment
from .protocol import Protocol, atomic_json
from .robotwin import CountedScene, DEFAULT_ROOT, configuration, load_api
from .vision import MarkerTracker, tool_point


def actual_gripper(robot, side):
    """Read measured finger joints; RoboTwin's gripper getter is a command cache."""
    entity = getattr(robot, side + "_entity")
    joint, multiplier, bias = getattr(robot, side + "_gripper")[0]
    index = list(entity.get_active_joints()).index(joint)
    position = (float(entity.get_qpos()[index]) - bias) / multiplier
    low, high = getattr(robot, side + "_gripper_scale")
    return float(np.clip((position - low) / (high - low), 0., 1.))


def sensors(env):
    obs = env.get_obs()
    cameras = {}
    for name in ("head_camera", "explore_left", "explore_right", "explore_front"):
        c = obs["observation"][name]
        cameras[name] = {"rgb": np.asarray(c["rgb"]).copy(),
                         "depth_m": np.asarray(c["depth"], dtype=np.float32) / 1000.,
                         "intrinsic": np.asarray(c["intrinsic_cv"]).copy(),
                         "extrinsic": np.asarray(c["extrinsic_cv"]).copy()}
    end = obs["endpose"]
    return {"cameras": cameras, "left_eef": np.asarray(end["left_endpose"]),
            "right_eef": np.asarray(end["right_endpose"]),
            "left_gripper": actual_gripper(env.robot, "left"),
            "right_gripper": actual_gripper(env.robot, "right")}


def save_sensors(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in data.items() if k != "cameras"}
    for camera, values in data["cameras"].items():
        flat.update({f"{camera}__{k}": v for k, v in values.items()})
    np.savez_compressed(path, **flat)


def load_sensors(path):
    with np.load(path) as d:
        out = {k: d[k].copy() for k in d.files if "__" not in k}
        out["cameras"] = {}
        for k in d.files:
            if "__" in k:
                camera, name = k.split("__")
                out["cameras"].setdefault(camera, {})[name] = d[k].copy()
    return out


def record(args):
    import imageio.v2 as imageio
    import torch
    if args.seed not in Protocol().train:
        raise ValueError("Reference skills are recorded on training configurations only")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    api = load_api(args.robotwin)
    cfg = add_cameras(configuration(api, out))
    env = fixed_asset_environment(api)
    tracker = MarkerTracker()
    traces, truth, dense, commands, frames = [], [], [], [], []
    print("REFERENCE_GPU", torch.cuda.get_device_name(), flush=True)
    try:
        random.seed(args.seed)
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **cfg)
        env.scene = CountedScene(env.scene)
        atomic_json(out / "instrumentation.json", {"markers": env.marker_layout,
            "sensors": cfg["data_type"], "asset": "044_microwave/7167",
            "claim": "instrumented-scene diagnostic", "setup_access": "asset links only for visual markers"})

        def capture(phase):
            data = sensors(env)
            tracked = tracker.observe(data)
            record = {"phase": phase, "tracked": asdict(tracked), "vision": tracker.last,
                      "left_eef": data["left_eef"].tolist(), "right_eef": data["right_eef"].tolist(),
                      "left_gripper": data["left_gripper"], "physics_steps": env.scene.steps}
            traces.append(record)
            save_sensors(out / f"sensors_{len(traces)-1:03d}.npz", data)
            for name, camera in data["cameras"].items():
                imageio.imwrite(out / f"{phase}_{name}.png", camera["rgb"])
            frames.append(np.concatenate([c["rgb"] for c in data["cameras"].values()], axis=1))
            truth.append({"phase": phase, "joint": float(env.microwave.get_qpos()[0]),
                          "simulator_success": bool(env.check_success()),
                          "privileged_access": ["object_joint", "simulator_success"]})
            print("REFERENCE_FRAME", phase, "opening", tracker.last["opening"],
                  "views", tracker.last["views_per_marker"], flush=True)

        capture("reset")
        if not args.reset_only:
            original_move = env.move

            def capture_dense():
                dense.append({"eef": env.get_arm_pose("left"),
                              "gripper": actual_gripper(env.robot, "left"), "physics_steps": env.scene.steps})

            def move(*a, **kw):
                acts = a[0][1]
                commands.append([{"action": x.action,
                                  "pose": None if x.target_pose is None else np.asarray(x.target_pose).tolist(),
                                  "gripper": x.target_gripper_pos} for x in acts])
                result = original_move(*a, **kw)
                capture(f"move{len(commands):02d}")
                return result

            env._take_picture = capture_dense
            env.move = move
            env.play_once()
        body = traces[0]["vision"]["body"]
        ready = len(traces) > 1 and body is not None and traces[1]["vision"]["door"] is not None
        handle = None
        if ready:
            door = np.asarray(traces[1]["vision"]["door"])
            handle = (np.linalg.inv(door) @ np.r_[tool_point(traces[1]["left_eef"]), 1.])[:3].tolist()
        atomic_json(out / "reference.json", {"seed": args.seed, "asset": "044_microwave/7167",
            "claim": "instrumented-scene diagnostic", "frames": traces, "commands": commands,
            "dense_proprio": dense, "initial_body": body, "handle_local": handle,
            "ready_for_replay": bool(ready and env.plan_success and env.check_success()),
            "source": "training-side privileged expert; online executor replays only stored robot poses"})
        atomic_json(out / "evaluation_only.json", {"frames": truth, "expert_success": bool(env.check_success())})
        imageio.mimsave(out / "reference.mp4", frames, fps=3, macro_block_size=None)
    finally:
        env.close_env()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--robotwin", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seed", type=int, default=210000)
    p.add_argument("--reset-only", action="store_true")
    record(p.parse_args())
