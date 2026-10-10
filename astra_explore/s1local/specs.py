"""Task specs for the task-generic pilot. Operations are declared from each task's successful Astra trace (filled in after
the Astra run, before the pilot; frozen in the pilot's manifest). Region predicates reuse each task's own success geometry.
"""
import math

import numpy as np

EXP = "/home/licho/workspace/Horsea/experiments/astra_explore"


def _inside(p, lo, hi):
    return lo[0] < p[0] < hi[0] and lo[1] < p[1] < hi[1]


def _exit_dist(p, lo, hi):
    """Distance the object still has to travel to leave the xy box (0 once outside)."""
    if not _inside(p, lo, hi):
        return 0.0
    return float(min(p[0] - lo[0], hi[0] - p[0], p[1] - lo[1], hi[1] - p[1]))


def _enter_dist(p, lo, hi):
    """Distance to the xy box (0 once inside)."""
    dx = max(lo[0] - p[0], 0.0, p[0] - hi[0])
    dy = max(lo[1] - p[1], 0.0, p[1] - hi[1])
    return float(math.hypot(dx, dy))


def _up(q):
    w, x, y, z = q
    return 1 - 2 * (x * x + y * y)


def make_region_fns(T):
    """Region predicates fn(state, object) -> (satisfied, distance to satisfaction), read from the live task env."""
    U = T.U
    fns = {}
    if hasattr(U, "container_xy_lower"):
        lo = U.container_xy_lower.cpu().numpy().copy()
        hi = U.container_xy_upper.cpu().numpy().copy()
        if hasattr(U, "curr_episode_offset"):
            off = U.curr_episode_offset.cpu().numpy().copy()
            lo, hi = lo + off, hi + off

        def unload(st, o, lo=lo, hi=hi):          # Unload-Cans per-can unload_success: outside the container, upright, above the table
            p, q = st["rigid"][o]["pos"], st["rigid"][o]["quat"]
            ok = (not _inside(p, lo, hi)) and _up(q) >= 0.5 and p[2] >= 1.0
            return ok, _exit_dist(p, lo, hi) + (0.0 if _up(q) >= 0.5 else 0.05)

        def insert(st, o, lo=lo, hi=hi):          # Insert-And-Unload-Cans insert: inside the container and down in a slot
            p = st["rigid"][o]["pos"]
            return _inside(p, lo, hi) and p[2] <= 1.065, _enter_dist(p, lo, hi) + max(0.0, p[2] - 1.065)

        def leave(st, o, lo=lo, hi=hi):           # Insert-And-Unload-Cans unload (after insertion): outside the container
            p = st["rigid"][o]["pos"]
            return not _inside(p, lo, hi), _exit_dist(p, lo, hi)
        fns.update(unload=unload, insert=insert, leave=leave)
    if hasattr(U, "right_container_bot_right_pos"):
        org = U.scene.env_origins[0].cpu().numpy()
        boxes = {"right_box": (U.right_container_bot_right_pos.cpu().numpy(), U.right_container_top_left_pos.cpu().numpy()),
                 "left_box": (U.left_container_bot_right_pos.cpu().numpy(), U.left_container_top_left_pos.cpu().numpy())}
        for name, (b_lo, b_hi) in boxes.items():
            def sort_fn(st, o, b_lo=b_lo, b_hi=b_hi):  # Sort-Cans: inside the target box in x, y and z (env-origin frame)
                p = np.array(st["rigid"][o]["pos"]) - org
                ok = all(b_lo[i] <= p[i] <= b_hi[i] for i in range(3))
                return ok, _enter_dist(p, b_lo, b_hi) + max(0.0, b_lo[2] - p[2], p[2] - b_hi[2])
            fns[name] = sort_fn
    if hasattr(U, "plate") and hasattr(U, "plate_radius"):
        def on_plate(st, o, r=float(U.plate_radius)):  # Stack-Can tasks: the env's can_on_plate (xy < radius, |dz| < 0.02)
            p, q = np.array(st["rigid"][o]["pos"]), np.array(st["rigid"]["plate"]["pos"])
            d = float(np.linalg.norm(p[:2] - q[:2]))
            return d < r and abs(p[2] - q[2]) < 0.02, d
        fns["on_plate"] = on_plate
    return fns


# Filled from each task's successful Astra trace before its pilot (see the pilot manifest for the frozen copy).
SPECS = {
    # regression of the generic code against the first pilot (same task, program and operations); smoke-only
    "drawer_regression": {
        "task": "Humanoid-Stack-Can-Into-Drawer-v0", "source_run": EXP + "/run_2026-10-10_astra_isaac_stack_can_drawer", "source_attempt": 10,
        "ops": {
            "grasp": {"kind": "grasp", "hand": "right", "object": "can", "prefix": [0, 1], "program": [2, 3], "window": 15,
                      "shift": ["can"], "physics": ["can"]},
            "transport": {"kind": "transport", "hand": "right", "object": "can", "prefix": [0, 1, 2, 3], "program": [4, 5, 6], "release": [6],
                          "window": 10, "success": "on_plate", "shift": ["can"], "physics": ["can"]},
            "door": {"kind": "push", "hand": "left", "object": ["drawer", "drawer_bottom_idx"], "prefix": [0, 1, 2, 3, 4, 5, 6, 7], "program": [8, 9, 10],
                     "window": 10, "success": 1.3264, "shift": [], "physics": [["drawer", "bottom"]]},
        },
    },
}
