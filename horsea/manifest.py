"""Protocol v2 manifest (experiment plan addendum, 2026-09-25): exact task split, writer split,
task pairs, start-state folds, old-task panels, software versions and checkpoint hashes.

    python -m horsea.manifest        -> experiments/manifest/protocol_v2.json
Everything that later scripts need is also importable from here (FOLDS, PAIRS, PANEL_20, ...).
"""
import hashlib
import json
import os
import re
import subprocess

from horsea.paths import BASE_CKPT, EXP, HELDOUT_90, TRAIN_90

# Five ordered pairs of the ten held-out tasks. Preregistered rule, independent of any method's
# performance: sort the held-out task IDs and pair consecutive IDs; the lower ID is task A (cycle 1).
PAIRS = [tuple(sorted(HELDOUT_90)[i:i + 2]) for i in range(0, 10, 2)]

# Writer-development split of the 80 training tasks (rule: every 8th training task, starting at
# the 4th, in ID order). The backbone has seen all 80; this split tests writer generalization only.
WRITER_DEV = sorted(TRAIN_90)[3::8]
WRITER_TRAIN = [t for t in sorted(TRAIN_90) if t not in WRITER_DEV]

# Disjoint start-state folds over LIBERO's 50 fixed initial states per task.
FOLDS = {
    "adapt": list(range(0, 10)),        # self-rollout adaptation attempts
    "teacher": list(range(10, 30)),     # consolidation-teacher data collection
    "validation": list(range(30, 50)),  # held-out tasks: gates and stored-acquisition curves
    "old_validation": list(range(30, 35)),  # old-task panel: frequent gate checks (5 per task)
    "old_test": list(range(40, 50)),    # old-task panel: final checks (10 per task)
    # held-out final test: newly generated and validated starts (horsea/starts.py), not these 50
}


def scene_of(name):
    m = re.match(r"^([A-Z_]+_SCENE\d+)_", name)
    return m.group(1) if m else name


def benchmark_tasks():
    from imitation.envs.libero.utils import get_benchmark_instance
    b = get_benchmark_instance("libero_90")
    return [{"id": i, "name": b.get_task(i).name, "language": b.get_task(i).language,
             "scene": scene_of(b.get_task(i).name), "bddl": b.get_task(i).bddl_file} for i in range(b.n_tasks)]


def panel_20(tasks=None):
    """Scene-balanced old-task panel for frequent checks: the lowest-ID training task per scene."""
    tasks = tasks or benchmark_tasks()
    first = {}
    for t in tasks:
        if t["id"] in TRAIN_90 and t["scene"] not in first:
            first[t["scene"]] = t["id"]
    return sorted(first.values())


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def versions():
    out = {}
    for mod in ("torch", "robosuite", "mujoco", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            out[mod] = f"unavailable: {e}"
    try:
        import libero
        p = os.path.dirname(os.path.dirname(libero.__file__))
        out["libero_path"] = p
        out["libero_git"] = subprocess.run(["git", "-C", p, "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip() or "not a git checkout"
    except Exception as e:  # noqa: BLE001
        out["libero"] = f"unavailable: {e}"
    return out


def main():
    import horsea  # noqa: F401  (LIBERO config path)
    import torch
    tasks = benchmark_tasks()
    sd = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = sd["config"]
    norm = {k: v for k, v in sd.items() if "normal" in k.lower()}
    man = {
        "protocol": "v2 (plan addendum 2026-09-25)",
        "benchmark": "LIBERO-90 via imitation.envs.libero (task order = benchmark instance order)",
        "tasks": tasks,
        "train_80": sorted(TRAIN_90), "heldout_10": sorted(HELDOUT_90),
        "writer_train_70": WRITER_TRAIN, "writer_dev_10": WRITER_DEV,
        "pairs_A_then_B": PAIRS,
        "start_folds": FOLDS,
        "old_panel_20": panel_20(tasks), "old_panel_full": sorted(TRAIN_90),
        "horizon_env_steps": 300,
        "reset": "env.set_init_state(LIBERO fixed init state[i]) via imitation LiberoRunner; generated test starts: horsea/starts.py",
        "instruction": "real task instruction everywhere (generic instruction = separate ablation)",
        "action_space": cfg["task"]["shape_meta"]["actions"],
        "observation": cfg["task"]["shape_meta"]["observation"],
        "chunk_size": cfg["algo"]["chunk_size"], "action_horizon": cfg["algo"]["action_horizon"],
        "temporal_agg_in_training_config": cfg["algo"]["temporal_agg"],
        "num_inference_steps": cfg["algo"]["num_inference_steps"],
        "base_checkpoint": BASE_CKPT, "base_sha256": sha256(BASE_CKPT),
        "normalizer_keys": list(norm.keys()),
        "versions": versions(),
        "previously_consulted": {
            "all_10_heldout": "adaptation-vs-K sweeps (experiments/adapt, adapt_generic), generic and real instruction",
            "57_66": "2-task lifecycle tests (experiments/cycles, cycles_full)",
            "note": "the ten are held out from backbone training, not a pristine blind test",
        },
    }
    os.makedirs(os.path.join(EXP, "manifest"), exist_ok=True)
    p = os.path.join(EXP, "manifest", "protocol_v2.json")
    with open(p, "w") as f:
        json.dump(man, f, indent=1, default=str)
    print("wrote", p)
    print("pairs", PAIRS, "| writer_dev", WRITER_DEV, "| panel_20", man["old_panel_20"])


if __name__ == "__main__":
    main()
