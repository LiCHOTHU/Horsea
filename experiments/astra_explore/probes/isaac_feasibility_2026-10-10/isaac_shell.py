"""Long-lived Isaac process for interactive feasibility probing (privileged; never part of what Astra sees).
Drop python files into CMD_DIR as cmd_<n>.py; each is exec'd with `task`, `np`, `torch`, `ns` (persistent dict) in scope;
stdout goes to out_<n>.txt, then done_<n> is touched."""
import io, os, sys, time, traceback, contextlib, json
from pathlib import Path
CMD = Path(sys.argv[1]); CMD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from isaac_env import IsaacDexTask
import numpy as np
task = IsaacDexTask(room=int(sys.argv[2]) if len(sys.argv) > 2 else 1, table=int(sys.argv[3]) if len(sys.argv) > 3 else 2, settle_steps=60)
import torch
ns = {"task": task, "np": np, "torch": torch, "json": json, "Path": Path}
print("SHELL READY", json.dumps(task.calibration), flush=True)
(CMD / "ready").write_text("1")
seen = set()
while True:
    cmds = sorted(CMD.glob("cmd_*.py"), key=lambda p: int(p.stem.split("_")[1]))
    todo = [c for c in cmds if c.name not in seen]
    if not todo:
        time.sleep(0.5)
        continue
    c = todo[0]
    seen.add(c.name)
    n = c.stem.split("_")[1]
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(c.read_text(), str(c), "exec"), ns)
    except SystemExit:
        (CMD / f"out_{n}.txt").write_text(buf.getvalue() + "\nEXIT")
        (CMD / f"done_{n}").write_text("1")
        break
    except Exception:
        buf.write("\nERROR\n" + traceback.format_exc())
    buf.write(f"\n[{time.time() - t0:.1f} s]")
    (CMD / f"out_{n}.txt").write_text(buf.getvalue())
    (CMD / f"done_{n}").write_text("1")
task.close()
