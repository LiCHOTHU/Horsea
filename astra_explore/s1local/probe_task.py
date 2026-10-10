"""Feasibility probe of a benchmark task in the existing dex harness (not an experiment): builds the task scene of the
project's first evaluation scene (room 1, table 2), resets it as the Astra harness does, saves the camera images Astra would
see and prints the evaluation-only object state."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from isaac_env import IsaacDexTask  # noqa: E402

task, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
T = IsaacDexTask(task)
obs = T.reset()
from PIL import Image  # noqa: E402
for name, px in obs["images"].items():
    Image.fromarray(px).save(out / f"{name}.png")
priv = T.privileged()
rec = {"task": task, "describe": {k: v for k, v in T.describe().items() if k != "calibration"}, "privileged_at_start": priv,
       "task_complete_at_start": obs["task_complete"], "rigid_objects": list(T.U.scene.rigid_objects.keys()),
       "articulations": list(T.U.scene.articulations.keys())}
(out / "probe.json").write_text(json.dumps(rec, indent=1, default=str))
print("PROBE", json.dumps({k: rec[k] for k in ("task", "task_complete_at_start", "rigid_objects", "articulations")}), flush=True)
print("STATE", json.dumps(priv)[:900], flush=True)
T.close()
