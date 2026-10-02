"""Which door contact points give a valid, plannable grasp from the canonical (closed-door) start? Audit helper."""
import sys
sys.path.insert(0, "script"); sys.path.append("./"); sys.path.append("./policy"); sys.path.append("./policy/HorseaFM"); sys.path.append("./description/utils")
import json, numpy as np  # noqa: E401,E402
import eval_policy as EP  # noqa: E402
from fixed_scenes import build_args  # noqa: E402
args = build_args("open_microwave", "explore_audit", "probe", policy_name="HorseaExplore"); args["render_freq"] = 0
env = EP.class_decorator("open_microwave")
for scene in [int(x) for x in sys.argv[1:]]:
    env.setup_demo(now_ep_num=0, seed=scene, is_test=True, **args)
    n = len(env.microwave.config.get("contact_points", [])) if hasattr(env.microwave, "config") else 8
    row = {}
    for cp in range(n):
        try:
            pre, pose = env.choose_grasp_pose(env.microwave, arm_tag="left", pre_dis=0.08, contact_point_id=cp)
        except Exception as ex:  # noqa: BLE001
            row[cp] = f"err {type(ex).__name__}"; continue
        if pre is None or pose is None:
            row[cp] = "no pose"; continue
        ok_pre = env.robot.left_plan_path(pre)["status"] == "Success"
        ok = ok_pre and env.robot.left_plan_path(pose)["status"] == "Success"
        row[cp] = ("plannable" if ok else "unreachable") + f" z={pose[2]:.3f} y={pose[1]:.3f}"
    print(scene, "model", env.model_id, json.dumps(row), flush=True)
    env.close_env(clear_cache=True)
