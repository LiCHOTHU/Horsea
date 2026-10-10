"""Every 30 min (max 12 h): one minimal gpt-6-astra call through the harness's own Codex wrapper. When the model is reachable
again, re-queue the three blocked Astra runs (fresh _r2 output directories) and restart the sequential queue."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from codex_astra import CodexAstra  # noqa: E402
from schema_dex import response_schema  # noqa: E402

Q = Path("/home/licho/workspace/Horsea/experiments/dexqueue_2026-10-10")
E = "/home/licho/workspace/Horsea/experiments/astra_explore"
PRE = ("source /home/licho/workspace/hongyi_deps/activate_hongyi.sh >/dev/null 2>&1; cd /home/licho/workspace/egovla_release/egovla && "
       "export OMNI_KIT_ACCEPT_EULA=YES && python -u /home/licho/workspace/Horsea/astra_explore/run_dex.py")
JOBS = [("astra_open_laptop_r2", "Humanoid-Open-Laptop-v0", "open_laptop", 30),
        ("astra_insert_and_unload_cans_r2", "Humanoid-Insert-And-Unload-Cans-v0", "insert_and_unload_cans", 90),
        ("astra_sort_cans_r2", "Humanoid-Sort-Cans-v0", "sort_cans", 90)]
for i in range(24):
    d = Q / "access_checks" / f"check_{i:02d}"
    raw, meta = CodexAstra("gpt-6-astra", "medium", 120.0, schema_fn=response_schema).act(d, {}, "Access check only: reply with any JSON object that matches the schema, with stop set to true.")
    text = ""
    for f in d.rglob("stdout.jsonl"):
        text += f.read_text(errors="ignore")
    blocked = meta.get("outcome") == "cli_error" and "not supported" in text
    print(time.strftime("%F %T"), "check", i, meta.get("outcome"), "blocked" if blocked else "REACHABLE", flush=True)
    if not blocked:
        lines = [ln for ln in (Q / "jobs.txt").read_text().splitlines() if ln.strip()]
        for name, task, short, calls in JOBS:
            lines.append(f"{name}|{PRE} --task {task} --out {E}/run_2026-10-10_astra_isaac_{short}_r2 --attempts 10 "
                         f"--calls-per-attempt {calls} --total-calls {10 * calls} --stop-on-success")
        (Q / "jobs.txt").write_text("\n".join(lines) + "\n")
        (Q / "STOP").unlink(missing_ok=True)
        subprocess.run(["systemd-run", "--user", f"--unit=dex-queue-r2-{int(time.time())}", "--setenv=HOME=/home/licho", str(Q / "runner.sh")])
        print(time.strftime("%F %T"), "ACCESS RESTORED: re-queued", [j[0] for j in JOBS], flush=True)
        break
    time.sleep(1800)
else:
    print(time.strftime("%F %T"), "still blocked after 12 h; giving up", flush=True)
