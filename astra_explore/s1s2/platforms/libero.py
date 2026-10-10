"""LIBERO adapter of the System-1 / System-2 transfer study (LIBERO-90 task 48, ketchup -> basket).

Student: BLOCKED_INTERFACE. No pretrained LIBERO student exists (the Horsea base80 FM checkpoint was deleted on 2026-09-30, and it
held task 48 out anyway). The from-scratch FM pipeline that survives (imitation.algos.fm_policy.FlowMatchingPolicy, env `specter`)
is a Hydra-composed policy whose encoder consumes processed LIBERO observations (hand_mat, camera extrinsics / intrinsics produced
by process_libero_data.py from raw LIBERO-90 demos that are no longer on disk); a bridge from Astra-recorded episodes to that
pipeline was not built within the overnight setup window. All student cells are therefore BLOCKED_INTERFACE for this platform.

Teacher: the existing LIBERO Astra harness (astra_explore/run.py), unchanged, behind the global call ledger. Only E01 (Astra-only
reference) runs, on the development pool, as one continuing teacher stream (the pilots' memory format).
"""
import json
import sys
import time
from pathlib import Path

HORSEA = "/home/licho/workspace/Horsea"
AE = HORSEA + "/astra_explore"
for _p in (AE, HORSEA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POOL_SLICES = {"source": (0, 15), "dev": (15, 20), "adapt": (20, 30), "test": (30, 40)}
REASON = ("no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM "
          "pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted "
          "LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window")


def pools():
    return {k: list(range(a, b)) for k, (a, b) in POOL_SLICES.items()}


def activity_audit(platform, args, out):
    from libero.libero import benchmark
    b = benchmark.get_benchmark_dict()["libero_90"]()
    t = b.get_task(48)
    manifest = {"platform": platform, "environment": "LIBERO (robosuite 1.4.1, mujoco 3.1.6), env specter", "task": t.name,
                "instruction": t.language, "success_definition": "LIBERO BDDL goal check within 300 steps", "init_states": "50 (pruned_init)",
                "initial_state_ids": pools(), "teacher_action": "native OSC_POSE 7-D delta x repeat (existing harness)",
                "student": "BLOCKED_INTERFACE: " + REASON}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"status": "BLOCKED_INTERFACE", "teacher_status": "OK", "summary": "student cells blocked: " + REASON, "reasons": [REASON],
            "cohort": "none (student blocked)", "student": {"summary": "none available"}, "pools": pools(), "pretrained_ckpt": None,
            "manifest": str(out / "manifest.json"), "runtime": {}}


def activity_teacher(platform, args, out):
    """E01: Astra-from-reset episodes through the existing harness (run.run_attempt), one continuing stream."""
    from libero_env import LiberoTask
    from run import GRIPPER_NOTE, run_attempt
    from s1s2.datasets import append_bank, read_bank
    from s1s2.ledger import Ledger
    from s1s2.registry import BUDGET
    from s1s2.teacher import LedgerBudget, LedgeredAstra, load_stream, save_stream
    ledger = Ledger(args["study_root"], BUDGET["astra_calls_global"], BUDGET["astra_calls_per_platform"])
    cell = args["cell"]
    stream = Path(args["study_root"]) / platform / f"stream_{cell}.json"
    history = load_stream(stream)
    done = [e["start_id"] for e in read_bank(args["study_root"], platform) if e.get("cell") == cell]
    for init in args["starts"]:
        if init in done:
            continue
        if ledger.remaining(platform) < 40:                 # never let the study budget truncate an episode (40-call ceiling)
            break
        task = LiberoTask("libero_90", 48, init, 512, 0)
        policy = LedgeredAstra(ledger, platform, cell)
        policy.episode = f"{cell}_{init}"
        budget = LedgerBudget(ledger, platform, 40, attempts=len(args["starts"]))   # view reads "attempt k of <starts>"
        epdir = Path(args["study_root"]) / platform / "episodes" / f"{cell}_{init}_{int(time.time())}"
        cfg = {"task": dict(task.describe(), gripper_note=GRIPPER_NOTE), "video": True}   # exactly as the pilot's run.main()
        t0 = time.time()
        k = len(history.attempt_summaries) + 1
        summ = run_attempt(task, policy, history, k, budget, epdir, cfg)
        save_stream(stream, history)
        success = summ.get("outcome") == "success"
        calls = sum(1 for e in ledger._events() if e.get("kind") == "call" and e.get("episode") == policy.episode)
        rec = {"episode_id": epdir.name, "pool": args.get("pool"), "start_id": init, "cell": cell, "success": success, "valid": False,
               "invalid_reason": "teacher-only reference (student blocked; no student-native recording)", "native_steps": summ.get("env_steps"),
               "astra_calls": calls, "path": str(epdir), "wall_s": round(time.time() - t0, 1)}
        append_bank(args["study_root"], platform, rec)
        print(json.dumps(rec), flush=True)
        try:
            task.close()
        except Exception:  # noqa: BLE001
            pass
    mine = [e for e in read_bank(args["study_root"], platform) if e.get("cell") == cell]
    data = {"episodes": len(mine), "successes": sum(1 for e in mine if e["success"]), "astra_calls": sum(e["astra_calls"] for e in mine),
            "native_steps": sum(e.get("native_steps") or 0 for e in mine)}
    return {"status": "OK", "summary": f"{cell}: {data}", "data": data}
