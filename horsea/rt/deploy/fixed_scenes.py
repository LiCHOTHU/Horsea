"""Fixed-scene RoboTwin evaluation for the Horsea graph study (spec 2026-10-01 rev. 2, sec. 8-9).

Why: RoboTwin admits a seed only if its scripted expert solves it, and that check is not deterministic (cuRobo planning),
so different evaluation runs admitted different scene sets (Stage A: 41 of 48 planned scenes common; open_microwave: no
scene admitted by all 30 runs). Here scenes are prevalidated ONCE and every arm evaluates exactly the same list.

  prevalidate  -- for seeds st_seed, st_seed+1, ...: setup_demo + expert play_once + (plan_success and check_success),
                  exactly as script/eval_policy.py; for each admitted seed, also reproduce the original instruction
                  derivation (setup_demo again, generate_episode_descriptions, np.random.choice) and store it.
                  Admission never depends on any policy.
  evaluate     -- for each listed seed: setup_demo(seed) (fresh scene, as the original does after its expert check),
                  set the stored instruction, run the policy loop of script/eval_policy.py unchanged, close_env with
                  the original cache-clearing schedule; prints the original 'Success rate: s/n ..., current seed: X'.

    cd RoboTwin && python policy/HorseaFM/fixed_scenes.py prevalidate --task lift_pot --st_seed 1400000 --n 20 \
        --out <scenes.json> --config policy/HorseaFM/deploy_policy.yml
    cd RoboTwin && python policy/HorseaFM/fixed_scenes.py evaluate --scenes <scenes.json> --config ... --overrides ...
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "script")
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import eval_policy as EP  # noqa: E402  (RoboTwin's own evaluator: config helpers, decorators, instruction generator)


def build_args(task_name, task_config, ckpt_setting="fixed", policy_name="HorseaFM"):
    """Replicates the environment-argument construction of script/eval_policy.py main()."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"], args["task_config"], args["ckpt_setting"] = task_name, task_config, ckpt_setting
    embodiment_type = args.get("embodiment")
    with open(os.path.join(EP.CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    robot_file = lambda e: _emb[e]["file_path"]
    with open(EP.CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _cam = yaml.load(f.read(), Loader=yaml.FullLoader)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = _cam[head]["h"], _cam[head]["w"]
    assert len(embodiment_type) == 1, "this study uses the single aloha-agilex embodiment"
    args["left_robot_file"] = args["right_robot_file"] = robot_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = EP.get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = EP.get_embodiment_config(args["right_robot_file"])
    args["policy_name"] = policy_name
    args["eval_mode"] = True
    args["eval_video_log"] = False      # no per-episode video in the graph study (does not affect the policy)
    args["eval_video_save_dir"] = None
    return args


def prevalidate(a):
    args = build_args(a.task, a.task_config)
    TASK_ENV = EP.class_decorator(args["task_name"])
    scenes, rejected, seed = [], [], a.st_seed
    while len(scenes) < a.n:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        ok, info = False, None
        try:
            TASK_ENV.setup_demo(now_ep_num=len(scenes), seed=seed, is_test=True, **args)
            info = TASK_ENV.play_once()
            TASK_ENV.close_env()
            ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        except Exception as e:  # noqa: BLE001  (UnStableError or planner errors -> not admitted, as in the original)
            TASK_ENV.close_env()
            rejected.append({"seed": seed, "reason": type(e).__name__})
            seed += 1
            args["render_freq"] = render_freq
            continue
        args["render_freq"] = render_freq
        if not ok:
            rejected.append({"seed": seed, "reason": "expert_failed"})
            seed += 1
            continue
        # instruction exactly as the original derives it (after a fresh setup_demo with this seed)
        TASK_ENV.setup_demo(now_ep_num=len(scenes), seed=seed, is_test=True, **args)
        results = EP.generate_episode_descriptions(args["task_name"], [info["info"]], a.n)
        instruction = str(np.random.choice(results[0][a.instruction_type]))
        TASK_ENV.close_env()
        scenes.append({"seed": seed, "instruction": instruction, "info": info["info"]})
        print(f"admitted seed {seed} ({len(scenes)}/{a.n}): {instruction}", flush=True)
        seed += 1
    json.dump({"task": a.task, "task_config": a.task_config, "instruction_type": a.instruction_type,
               "st_seed": a.st_seed, "scenes": scenes, "rejected": rejected}, open(a.out, "w"), indent=1, default=str)
    print(f"DONE {a.task}: {len(scenes)} admitted, {len(rejected)} rejected", flush=True)


def evaluate(a, usr_args):
    man = json.load(open(a.scenes))
    if a.subset:
        man["scenes"] = man["scenes"][:a.subset]
    task = man["task"]
    usr_args.update({"task_name": task, "task_config": man["task_config"], "policy_name": "HorseaFM",
                     "instruction_type": man["instruction_type"]})
    args = build_args(task, man["task_config"], usr_args.get("ckpt_setting", "fixed"))
    TASK_ENV = EP.class_decorator(args["task_name"])
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    model = EP.eval_function_decorator("HorseaFM", "get_model")(usr_args)
    eval_func = EP.eval_function_decorator("HorseaFM", "eval")
    reset_func = EP.eval_function_decorator("HorseaFM", "reset_model")
    clear_cache_freq = args["clear_cache_freq"]
    TASK_ENV.suc = 0
    # Resumable retries: with HORSEA_SCENE_RESULTS set, every finished scene is appended there; a retried attempt (after a
    # simulator hang killed by the watchdog) replays those result lines and continues with the remaining scenes. Each
    # scene is independent (fresh setup_demo(seed), stored instruction, noise keyed by env seed, policy reset).
    res_path = os.environ.get("HORSEA_SCENE_RESULTS")
    done = {}
    if res_path and os.path.exists(res_path):
        for line in open(res_path):
            d = json.loads(line)
            done[d["seed"]] = d
    for i, sc in enumerate(man["scenes"]):
        if sc["seed"] in done:
            d = done[sc["seed"]]
            TASK_ENV.suc += int(d["success"])
            print(f"{task} | HorseaFM | {man['task_config']} | {usr_args.get('ckpt_setting', 'fixed')}\n"
                  f"Success rate: {TASK_ENV.suc}/{i + 1} => {round(TASK_ENV.suc / (i + 1) * 100, 1)}%, "
                  f"current seed: {sc['seed']} steps: {d['steps']} [resumed from a previous attempt]", flush=True)
            continue
        TASK_ENV.test_num = i
        TASK_ENV.setup_demo(now_ep_num=i, seed=sc["seed"], is_test=True, **args)
        TASK_ENV.set_instruction(instruction=sc["instruction"])
        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        if succ:
            TASK_ENV.suc += 1
        # original: close_env(clear_cache=((succ_seed + 1) % freq == 0)) where succ_seed == i + 1 at this point
        TASK_ENV.close_env(clear_cache=((i + 2) % clear_cache_freq == 0))
        if res_path:
            with open(res_path, "a") as f:
                f.write(json.dumps({"seed": sc["seed"], "success": bool(succ), "steps": int(TASK_ENV.take_action_cnt)}) + "\n")
        print(f"{task} | HorseaFM | {man['task_config']} | {usr_args.get('ckpt_setting', 'fixed')}\n"
              f"Success rate: {TASK_ENV.suc}/{i + 1} => {round(TASK_ENV.suc / (i + 1) * 100, 1)}%, "
              f"current seed: {sc['seed']} steps: {TASK_ENV.take_action_cnt}", flush=True)
    print("FIXED_EVAL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prevalidate", "evaluate"])
    ap.add_argument("--task")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--instruction_type", default="unseen")
    ap.add_argument("--st_seed", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out")
    ap.add_argument("--scenes")
    ap.add_argument("--subset", type=int, default=0, help="evaluate only the first N listed scenes (predeclared subset)")
    ap.add_argument("--config", default="policy/HorseaFM/deploy_policy.yml")
    ap.add_argument("--overrides", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    if a.cmd == "prevalidate":
        prevalidate(a)
        return
    with open(a.config, "r", encoding="utf-8") as f:
        usr_args = yaml.safe_load(f)
    ov = a.overrides
    for i in range(0, len(ov), 2):
        k, v = ov[i].lstrip("-"), ov[i + 1]
        try:
            v = eval(v)  # same convention as script/eval_policy.py parse_override_pairs
        except Exception:  # noqa: BLE001
            pass
        usr_args[k] = v
    evaluate(a, usr_args)


if __name__ == "__main__":
    main()
