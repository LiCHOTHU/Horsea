"""Overnight job queue: keeps the GPU busy without oversubscribing RAM/CPU.

Jobs are appended to experiments/queue/jobs.jsonl, one JSON object per line:
    {"id": "e_fwrite_p1", "cmd": "python -m ...", "envs": 3, "after": ["manifest"], "group": "E", "priority": 0}
`envs` = simulator processes the job starts (0 for pure GPU training). A job starts when all
jobs in `after` are done, the total running envs stays <= MAX_ENVS, running jobs <= MAX_JOBS and its
group is under its cap (experiments/queue/limits.json). Higher `priority` starts first.
State: experiments/queue/state.json; logs: experiments/logs/q_<id>.log. Failed jobs are not
retried. The file is re-read every poll, so jobs can be added while the queue runs.

    nohup setsid python scripts/queue.py > experiments/queue/queue.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "experiments", "queue")
LOGS = os.path.join(ROOT, "experiments", "logs")
MAX_ENVS = int(os.environ.get("Q_MAX_ENVS", 16))
MAX_JOBS = int(os.environ.get("Q_MAX_JOBS", 6))
PY = sys.executable  # whichever env launched the queue (conda "Horsea" on PACE)


def load_jobs():
    jobs = []
    p = os.path.join(QDIR, "jobs.jsonl")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    print("bad job line:", line[:120], flush=True)
    return jobs


def main():
    os.makedirs(QDIR, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    sp = os.path.join(QDIR, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    procs, adopted = {}, {}
    for k, v in state.items():  # jobs still running from a previous queue process: adopt by pid
        if v["status"] == "running":
            if v.get("pid") and os.path.exists(f"/proc/{v['pid']}"):
                adopted[k] = v["pid"]
            else:
                v["status"] = "lost"
    while True:
        # reap
        for jid, p in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                state[jid].update(status="done" if rc == 0 else "failed", rc=rc, end=time.strftime("%H:%M:%S"))
                print(time.strftime("%H:%M:%S"), "finished", jid, "rc", rc, flush=True)
                del procs[jid]
        for jid, pid in list(adopted.items()):
            if not os.path.exists(f"/proc/{pid}"):
                state[jid].update(status="done", rc="adopted", end=time.strftime("%H:%M:%S"))
                print(time.strftime("%H:%M:%S"), "finished (adopted)", jid, flush=True)
                del adopted[jid]
        jobs = load_jobs()
        running = set(procs) | set(adopted)
        used = sum(j.get("envs", 0) for j in jobs if j["id"] in running)
        lp = os.path.join(QDIR, "limits.json")  # per-group caps, e.g. {"E": 4}
        limits = json.load(open(lp)) if os.path.exists(lp) else {}
        groups = {}
        for j in jobs:
            if j["id"] in running:
                groups[j.get("group")] = groups.get(j.get("group"), 0) + 1
        order = sorted(range(len(jobs)), key=lambda i: (-jobs[i].get("priority", 0), i))
        for j in (jobs[i] for i in order):
            jid = j["id"]
            if jid in state and state[jid]["status"] != "pending":
                continue
            state.setdefault(jid, {"status": "pending"})
            deps = [state.get(d, {}).get("status") for d in j.get("after", [])]
            if any(s in ("failed", "lost") for s in deps):
                state[jid]["status"] = "blocked"
                continue
            if not all(s == "done" for s in deps):
                continue
            g = j.get("group")
            if len(running) >= MAX_JOBS or used + j.get("envs", 0) > MAX_ENVS:
                continue
            if g in limits and groups.get(g, 0) >= limits[g]:
                continue
            cmd = j["cmd"].replace("python ", PY + " ", 1) if j["cmd"].startswith("python ") else j["cmd"]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=ROOT)
            log = open(os.path.join(LOGS, f"q_{jid}.log"), "a")
            procs[jid] = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            used += j.get("envs", 0)
            running.add(jid)
            groups[g] = groups.get(g, 0) + 1
            state[jid] = {"status": "running", "pid": procs[jid].pid, "start": time.strftime("%H:%M:%S"),
                          "envs": j.get("envs", 0)}
            print(time.strftime("%H:%M:%S"), "started", jid, "envs", j.get("envs", 0), flush=True)
        with open(sp + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(sp + ".tmp", sp)
        time.sleep(20)


if __name__ == "__main__":
    main()
