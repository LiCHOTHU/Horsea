# Astra exploration visualization server

A static site built from the two Astra runs (every decision with the exact images Astra saw, its assessment of the
previous action, intent, hypothesis, expected change, the executed action, the measured effect, the memory rewrite, per-attempt
videos and contact sheets, and the narrative reports), served on `127.0.0.1:8765` of this workstation (130.207.121.116).

## View it from your laptop

```bash
ssh -L 8765:127.0.0.1:8765 licho@130.207.121.116
# then open in your browser:
#   http://localhost:8765/            both cases
#   http://localhost:8765/libero/     LIBERO: "pick up the ketchup and put it in the basket"
#   http://localhost:8765/robotwin/   RoboTwin: bowl onto the plate (planner-assisted interface)
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
