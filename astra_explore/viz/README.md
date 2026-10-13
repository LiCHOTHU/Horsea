# Astra exploration visualization server

A static site of the Astra runs and the studies built on them, served on `127.0.0.1:8765` of this workstation
(130.207.121.116). The header has one dropdown menu per series:
- **Series A, Astra on its own:** every decision with the exact images Astra saw, its assessment of the previous action,
  intent, hypothesis, expected change, the executed action, the measured effect and the memory rewrite; per-attempt videos
  and contact sheets; the narrative reports.
- **Series B:** the overnight transfer study (can a policy learn from Astra?).
- **Series C:** System 1 beneath Astra (pilots, grasp-failure diagnosis, cup transfer experiment, design note).

## View it from your laptop

```bash
ssh -L 8765:127.0.0.1:8765 licho@130.207.121.116
# then open in your browser:
#   http://localhost:8765/              overview of all series
#   http://localhost:8765/libero/       A: LIBERO "pick up the ketchup and put it in the basket"
#   http://localhost:8765/robotwin/     A: RoboTwin bowl onto the plate (planner-assisted interface)
#   http://localhost:8765/bimanual/     A: RoboTwin lift_pot (bimanual)
#   http://localhost:8765/dexhand/      A: dex hand Stack-Can-Into-Drawer (also dex_unload/, dex_laptop/, dex_insert/)
#   http://localhost:8765/transfer/     B: overnight transfer study
#   http://localhost:8765/s1local/      C: System 1 beneath Astra (cup_transfer.html, diagnosis.html, design.html)
```

Keep the tunnel alive automatically (autossh, `-M 0` disables the monitor port; `-N` = no remote command):

```bash
autossh -M 0 -N -L 8765:127.0.0.1:8765 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 licho@130.207.121.116
```

The server binds to localhost only, so it is reachable solely through the tunnel.

## Manage the server on the workstation

```bash
cd ~/workspace/Horsea/astra_explore/viz
python3 launcher.py status     # verified PID (command line + bound port) and /health
python3 launcher.py start      # rebuilds the site, starts the server detached (survives logout), checks /health
python3 launcher.py restart    # e.g. after new runs are added to RUNS in launcher.py
python3 launcher.py stop       # SIGTERM the verified PID, wait up to 10 s, then SIGKILL
python3 launcher.py build      # rebuild the static pages only
```

State: PID file `experiments/astra_explore/site/.server.pid`, log `experiments/astra_explore/site/.server.log`.
Pages are plain HTML in `experiments/astra_explore/site/`; images and videos are served from the run directories through
the symlinks `site/<case>/run`.

Optional supervisor-style autorestart: a systemd user unit can wrap `serve.py` with `Restart=always`
(`loginctl enable-linger licho` is needed for it to outlive logins); not installed, to avoid two instances on one port.

## Files

- `build_site.py`: generator (`--out site --run libero=<run dir> --run robotwin=<run dir>`).
- `serve.py`: `ThreadingHTTPServer` with HTTP Range support (video seeking), `/health`, no-cache HTML.
- `launcher.py`: start / stop / restart / status / build with PID verification and health polling.

## Series C (System 1 pilots)

`build_s1local.py` (called by `build_site.py --s1local <experiments dir>`, wired in `launcher.py`) writes `site/s1local/`:
an overview (answer, task status, the operation x condition outcome matrix, pooled contrasts, `SCALABILITY.md`) and one
page per task (`drawer`, `unload`, `laptop`, `insert`: operations, per-condition outcome grid, plots, re-executed videos,
result tables). Media come from each pilot directory through the symlinks `site/s1local/<task>_run`.
It also writes `diagnosis.html` (why grasps fail, from `experiments/s1_diagnosis_2026-10-11/DIAGNOSIS.md`), `design.html`
(`research/system1_dexhand_design.md`), `cup_transfer.html` (`experiments/cup_transfer_2026-10-12/REPORT.md`, the S(b) plot,
attempt-1 videos, result tables and the per-grasp check, through `site/s1local/cup_run`), and an "In progress" box read from
the experiment files at build time.
