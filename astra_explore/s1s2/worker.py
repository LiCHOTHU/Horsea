"""Worker entry point: `python -m s1s2.worker <platform> <activity> <args.json>`.

Runs one activity of one platform in that platform's own interpreter and writes `result.json` next to args.json. Exit code 0
means the activity ran to completion (its experimental outcome may still be a failure); non-zero means an infrastructure
failure, which the orchestrator may retry within its bounded policy.
"""
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

ADAPTERS = {"libero": "s1s2.platforms.libero", "robotwin_single": "s1s2.platforms.robotwin",
            "robotwin_bimanual": "s1s2.platforms.robotwin", "dexhand": "s1s2.platforms.dexhand"}


def main():
    platform, activity, args_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    args = json.loads(args_path.read_text())
    out = args_path.parent
    t0 = time.time()
    mod = importlib.import_module(ADAPTERS[platform])
    fn = getattr(mod, "activity_" + activity, None)
    if fn is None:
        res = {"status": "BLOCKED_INTERFACE", "summary": f"activity {activity} not implemented for {platform}"}
    else:
        try:
            res = fn(platform, args, out)
        except Exception as ex:  # noqa: BLE001  infrastructure failure: no result.json -> orchestrator may retry
            (out / "exception.txt").write_text(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")
            print("WORKER EXCEPTION", type(ex).__name__, ex, flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            raise SystemExit(2)
    res.setdefault("wall_s", round(time.time() - t0, 1))
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(out / "result.json")
    print("WORKER DONE", res.get("status"), res.get("summary", ""), flush=True)
    # Isaac Sim's app.close() ends the process; adapters that hold a simulator close it after result.json is written
    closer = getattr(mod, "shutdown", None)
    if closer:
        closer()


if __name__ == "__main__":
    main()
