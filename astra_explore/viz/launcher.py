"""Launcher for the Astra visualization server (daemonocle-style: start / stop / restart / status / build, PID file).

    python launcher.py start      # (re)build the site, start the server detached, verify it is alive and healthy
    python launcher.py status     # PID verified against its command line and the bound port, /health probed
    python launcher.py stop       # SIGTERM the verified PID, wait up to 10 s, SIGKILL if needed
    python launcher.py restart
    python launcher.py build      # rebuild the static site only

Beyond daemonocle: after start it polls /health; before stop it confirms the PID really is this server (command line
contains serve.py with this root, and the port is bound by that PID). Survives logout (new session, no controlling tty).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HORSEA = HERE.parent.parent
SITE = HORSEA / "experiments" / "astra_explore" / "site"
RUNS = {"libero": HORSEA / "experiments" / "astra_explore" / "run_2026-10-09_astra_t48",
        "robotwin": HORSEA / "experiments" / "astra_explore" / "run_2026-10-09_astra_rt_place_container_plate",
        "bimanual": HORSEA / "experiments" / "astra_explore" / "run_2026-10-09_astra_rt_lift_pot"}
PORT, BIND = 8765, "127.0.0.1"
PID_FILE, LOG_FILE = SITE / ".server.pid", SITE / ".server.log"
PY = sys.executable


def build():
    cmd = [PY, str(HERE / "build_site.py"), "--out", str(SITE)] + [x for c, r in RUNS.items() for x in ("--run", f"{c}={r}")]
    subprocess.run(cmd, check=True)


def read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def port_owner():
    """PID bound to BIND:PORT according to `ss -ltnp`, or None."""
    try:
        out = subprocess.run(["ss", "-ltnp", f"sport = :{PORT}"], capture_output=True, text=True).stdout
    except OSError:
        return None
    for line in out.splitlines():
        if f":{PORT} " in line and "pid=" in line:
            return int(line.split("pid=")[1].split(",")[0])
    return None


def verified_pid():
    pid = read_pid()
    if pid is None or not alive(pid):
        return None
    if "serve.py" not in cmdline(pid) or str(SITE) not in cmdline(pid):
        return None
    if port_owner() not in (pid, None):
        return None
    return pid


def health():
    try:
        with urllib.request.urlopen(f"http://{BIND}:{PORT}/health", timeout=2) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def start(rebuild=True):
    if verified_pid():
        print(f"already running: pid {read_pid()}, healthy={health()}")
        return 0
    if port_owner() is not None:
        print(f"port {PORT} is bound by another process (pid {port_owner()}); refusing to start")
        return 1
    if rebuild:
        build()
    SITE.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "a")
    proc = subprocess.Popen([PY, str(HERE / "serve.py"), "--root", str(SITE), "--port", str(PORT), "--bind", BIND],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(SITE))
    PID_FILE.write_text(str(proc.pid))
    for _ in range(40):
        time.sleep(0.25)
        if proc.poll() is not None:
            print(f"server exited immediately (rc {proc.returncode}); see {LOG_FILE}")
            PID_FILE.unlink(missing_ok=True)
            return 1
        if health():
            print(f"started: pid {proc.pid}, http://{BIND}:{PORT}/ healthy; log {LOG_FILE}")
            return 0
    print(f"server process {proc.pid} is alive but /health did not answer within 10 s; see {LOG_FILE}")
    return 1


def stop():
    pid = verified_pid()
    if pid is None:
        stale = read_pid()
        print("not running" + (f" (stale pid file {stale} removed)" if stale else ""))
        PID_FILE.unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        time.sleep(0.25)
        if not alive(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    print(f"stopped pid {pid}" + ("" if not alive(pid) else " (still alive after SIGKILL?)"))
    return 0


def status():
    pid = verified_pid()
    if pid is None:
        print(f"stopped (pid file: {read_pid()}, port owner: {port_owner()})")
        return 3
    print(f"running: pid {pid}, verified command line and port {PORT}, healthy={health()}")
    print(f"local URL: http://{BIND}:{PORT}/   pages: /libero/  /robotwin/")
    print(f"from your laptop: ssh -L {PORT}:{BIND}:{PORT} licho@130.207.121.116   then open http://localhost:{PORT}/")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "restart", "status", "build"])
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if a.action == "build":
        build()
        return 0
    if a.action == "start":
        return start(rebuild=not a.no_build)
    if a.action == "stop":
        return stop()
    if a.action == "restart":
        stop()
        return start(rebuild=not a.no_build)
    return status()


if __name__ == "__main__":
    sys.exit(main())
