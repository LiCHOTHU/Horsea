"""Locked registry of the System-1 / System-2 transfer study (user spec of 2026-10-10).

Astra = System 2 (teacher, existing Codex integration). The platform's learned policy = System 1 (student, native interface).
Twelve configurations (E01-E12) x four platforms. Nothing here is tuned after results are seen; the orchestrator copies this
registry into the run's locked manifest before the first experiment cell starts.
"""

PLATFORMS = ("libero", "robotwin_single", "robotwin_bimanual", "dexhand")

TASKS = {
    "libero": {"suite": "libero_90", "task_id": 48,
               "name": "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket"},
    "robotwin_single": {"task": "place_container_plate"},
    "robotwin_bimanual": {"task": "lift_pot"},
    "dexhand": {"task": "Humanoid-Stack-Can-Into-Drawer-v0"},
}

CELLS = {
    "E01": "Astra-only (teacher capability and interaction / computation cost)",
    "E02": "Original pretrained S1, unchanged",
    "E03": "S1 trained on D1 (first successful episode)",
    "E04": "S1 trained on D3 (first three successful episodes)",
    "E05": "S1 trained on DN (first N = min(10, available) successful episodes) - main anchor",
    "E06": "E05 with a shorter executed action chunk (about half, >= 1)",
    "E07": "S1 trained on all valid teacher trajectories up to the N-th success (successes + failures)",
    "E08": "E05 then one scheduled Astra takeover at half the native step budget, current observation only",
    "E09": "E08 plus the last three student interaction records before handover",
    "E10": "E05 continued for dU updates on its original data",
    "E11": "E05 continued for dU updates on original data + Astra corrections (half / half batches)",
    "E12": "E05 continued for dU updates on original data + additional Astra-from-reset demos (half / half)",
}

STATUSES = ("COMPLETE", "RUNNING", "PENDING_BUDGET", "BLOCKED_INTERFACE", "BLOCKED_DATA", "NO_STUDENT_SIGNAL",
            "NOT_APPLICABLE", "DUPLICATE", "FAILED", "BLOCKED_NO_CORRECTIONS", "NOT_AVAILABLE", "PENDING")

BUDGET = {
    "window_hours": 8.0,
    "astra_calls_global": 600,
    "astra_calls_per_platform": 150,
    "active_cells": 1,
    "training_seeds": 1,
    "infra_retries": 2,              # per episode / per job, simulator crash or transient service failure only
    "dev_starts_initial": 3,
    "dev_starts_expanded": 5,
    "test_starts": 10,
    "astra_calls_per_episode_cap": 40,   # the existing per-attempt ceiling of the pilots
}

TRAINING_DEFAULTS = {               # used only where the platform has no fine-tuning recipe of its own
    "optimizer": "AdamW", "lr": 1e-5, "batch": 32, "U": 1000, "dU": 500, "grad_clip": 1.0,
}

# Scheduling order of the spec (section 10). Each stage lists the activities run per eligible platform.
STAGES = [
    ("audit", "audit interfaces and existing datasets; replay diagnostic"),
    ("source_min", "unlock minimal valid source data (aim: 3 successes per platform before going to 10 anywhere)"),
    ("ref", "E02 original student and E03 (D1)"),
    ("data_scale", "E04 (D3) and E05 (DN)"),
    ("anchor_uses", "E06, E08, E09 on the anchor student"),
    ("consolidation", "adaptation-pool collection (corrections, extra demos), then E10-E12"),
    ("failures", "E07 (successes + failures)"),
    ("locked_eval", "E01 Astra-only reference and test-pool evaluation of the principal comparisons"),
]
