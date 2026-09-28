"""Generate and validate extra start states (protocol v2 final tests need ~100 independent starts per
held-out task; LIBERO ships only 50 fixed ones, which are split into adapt/teacher/validation folds).

A start = LIBERO's own placement sampler (env.seed(s); env.reset()) -> flattened MuJoCo state.
Accepted only if (1) the task is not already solved, (2) objects settle no more than LIBERO's fixed starts do (+1 cm) over 20 no-op steps, (3) it differs from every fixed init state.

    python -m horsea.starts --tasks 2 9 13 25 43 48 57 66 79 84 --n 100
    -> experiments/starts/libero_90_t{task}.npy  (+ .json with seeds and rejection counts)
"""
import argparse
import json
import os

import numpy as np

import horsea  # noqa: F401
from horsea.paths import EXP

SEED0 = 100_000  # generated starts use seeds SEED0 + task*10_000 + k (disjoint from any training seed)


def object_positions(obs):
    return np.concatenate([np.asarray(v).ravel() for k, v in sorted(obs.items())
                           if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k])


def generate(task, n, max_tries):
    from imitation.envs.libero.utils import get_benchmark_instance
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    b = get_benchmark_instance("libero_90")
    t = b.get_task(task)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    fixed = b.get_task_init_states(task)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    noop = np.array([0, 0, 0, 0, 0, 0, -1.0])

    def drift(state):
        env.set_init_state(state)
        obs, _, _, _ = env.step(noop)  # positions in set_init_state's observation are stale
        solved, p0 = env.check_success(), object_positions(obs)
        for _ in range(20):
            obs, _, _, _ = env.step(noop)
        return float(np.max(np.abs(object_positions(obs) - p0))), solved

    # LIBERO's own fixed starts settle a little (e.g. ~4.5 cm for one object in some scenes): accept
    # generated starts that settle no more than the fixed ones do (+1 cm)
    ref = max(drift(fixed[i])[0] for i in range(5))
    out, seeds, rej = [], [], {"solved": 0, "unstable": 0, "duplicate": 0, "ref_drift": round(ref, 4)}
    k = 0
    while len(out) < n and k < max_tries:
        seed = SEED0 + task * 10_000 + k
        k += 1
        env.seed(seed)
        np.random.seed(seed)
        env.reset()
        state = env.sim.get_state().flatten()
        if np.min(np.abs(fixed - state[None]).sum(1)) < 1e-6:
            rej["duplicate"] += 1
            continue
        d, solved = drift(state)
        if solved:
            rej["solved"] += 1
            continue
        if d > ref + 0.01:
            rej["unstable"] += 1
            continue
        out.append(state)
        seeds.append(seed)
    env.close()
    return np.stack(out), seeds, rej, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_tries", type=int, default=400)
    args = ap.parse_args()
    d = os.path.join(EXP, "starts")
    os.makedirs(d, exist_ok=True)
    for task in args.tasks:
        p = os.path.join(d, f"libero_90_t{task}.npy")
        if os.path.exists(p):
            continue
        states, seeds, rej, tries = generate(task, args.n, args.max_tries)
        np.save(p, states)
        json.dump({"task": task, "n": len(states), "seeds": seeds, "rejected": rej, "tries": tries,
                   "rule": "LIBERO placement sampler; not solved; object drift over 20 no-op steps <= max drift of 5 fixed starts + 1cm; not a fixed init"},
                  open(p.replace(".npy", ".json"), "w"))
        print(task, len(states), "accepted of", tries, rej, flush=True)


def load(task):
    """Generated test starts for a task, or None."""
    p = os.path.join(EXP, "starts", f"libero_90_t{task}.npy")
    return np.load(p) if os.path.exists(p) else None


if __name__ == "__main__":
    main()
