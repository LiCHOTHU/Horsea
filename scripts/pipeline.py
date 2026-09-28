"""Run the whole memory study as a dependency graph of resumable jobs.

    nohup python scripts/pipeline.py > experiments/pipeline.log 2>&1 &

* waits until CUDA works (the NVIDIA module/userspace mismatch must be fixed first);
* every job is resumable and has a completion check; a job that dies without completing is
  restarted (up to MAX_TRIES) -- long background jobs on this machine get silently SIGKILLed;
* concurrency classes: 'excl' (base training, runs alone), 'gpu' (meta-training / features),
  'sim' (LIBERO rollouts, CPU-heavy);
* status in experiments/pipeline_status.json, per-job logs in experiments/logs/.
"""
import json
import os
import subprocess
import sys
import time

HORSEA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HORSEA)
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, HELDOUT_90, LIBERO_10, RETENTION_90  # noqa: E402

PY = "/home/licho/anaconda3/envs/specter/bin/python"
LOGS = os.path.join(EXP, "logs")
LIMITS = {"excl": 1, "gpu": 4, "sim": 3}
MAX_TRIES = 6
ARMS = ["ttt", "kv", "fmw", "res"]


def exists(*p):
    return lambda: os.path.exists(os.path.join(*p))


def all_json(arm, suite, tasks, Ks, wrong=False, root="adapt"):
    d = os.path.join(EXP, root, arm, suite)
    names = [f"t{t}_K{k}" + ("_wrong" if wrong else "") + ".json" for t in tasks for k in Ks]
    return lambda: all(os.path.exists(os.path.join(d, n)) for n in names)


def base_done():
    log = os.path.join(os.path.dirname(BASE_CKPT), "training.log")
    return os.path.exists(BASE_CKPT) and os.path.exists(log) and "Training completed successfully" in open(log).read()


def mod(m, *a):
    return [PY, "-m", m, *map(str, a)]


def jobs():
    J = []

    def add(name, cmd, done, deps=(), cls="gpu"):
        J.append({"name": name, "cmd": cmd, "done": done, "deps": list(deps), "cls": cls})

    add("base", [PY, os.path.join(HORSEA, "scripts", "train_base.py")], base_done, cls="excl")
    add("feat90", mod("horsea.features", "--suite", "libero_90"), exists(FEAT_DIR, "libero_90.pt"), ["base"])
    add("feat10", mod("horsea.features", "--suite", "libero_10"), exists(FEAT_DIR, "libero_10.pt"), ["base"])
    for arm in ARMS:
        add(f"meta_{arm}", mod("horsea.meta_train", "--arm", arm), exists(EXP, "memory", arm, "final.pt"), ["feat90"])

    # --- closed-loop evaluation, in priority order --------------------------------------------
    H, R, L = HELDOUT_90, RETENTION_90, LIBERO_10
    add("eval_base_90", mod("horsea.adapt_eval", "--arm", "base", "--suite", "libero_90"),
        all_json("base", "libero_90", H, [0]), ["feat90"], "sim")
    add("eval_base_ret", mod("horsea.adapt_eval", "--arm", "base", "--suite", "libero_90", "--n", 10, "--tasks", *R),
        all_json("base", "libero_90", R, [0]), ["feat90"], "sim")
    add("eval_ft_90", mod("horsea.adapt_eval", "--arm", "ft", "--suite", "libero_90"),
        all_json("ft", "libero_90", H, [1, 2, 5, 10]), ["feat90"], "sim")
    for arm in ARMS:
        Ks = [0, 1, 2, 5, 10] if arm == "ttt" else [1, 2, 5, 10]
        add(f"eval_{arm}_90", mod("horsea.adapt_eval", "--arm", arm, "--suite", "libero_90", "--K", *Ks),
            all_json(arm, "libero_90", H, Ks), [f"meta_{arm}"], "sim")
    # --- generic-instruction protocol (RoboTTT one-shot): demos are the only task signal ---------
    G = "complete the task"
    GA = ["ttt2", "kv", "fmw", "res"]
    for suite in ["libero_90", "libero_10"]:
        add(f"gfeat_{suite}", mod("horsea.features", "--suite", suite, "--instruction", G,
                                   "--out", os.path.join(FEAT_DIR, f"{suite}_generic.pt")),
            exists(FEAT_DIR, f"{suite}_generic.pt"), ["base"])
    for arm in GA:
        add(f"gmeta_{arm}", mod("horsea.meta_train", "--arm", arm, "--features",
                                 os.path.join(FEAT_DIR, "libero_90_generic.pt"), "--out",
                                 os.path.join(EXP, "memory_generic", arm)),
            exists(EXP, "memory_generic", arm, "final.pt"), ["gfeat_libero_90"])
    gcommon = ["--generic", "--mem_dir", os.path.join(EXP, "memory_generic"), "--out", os.path.join(EXP, "adapt_generic")]
    for suite, tasks in [("libero_90", H), ("libero_10", L)]:
        for arm in ["base", "ft"] + GA:
            Ks = [0] if arm == "base" else [1, 2, 5, 10]
            deps = [f"gfeat_{suite}"] + ([f"gmeta_{arm}"] if arm in GA else [])
            add(f"geval_{arm}_{suite[-2:]}", mod("horsea.adapt_eval", "--arm", arm, "--suite", suite, "--K", *Ks, *gcommon),
                all_json(arm, suite, tasks, Ks, root="adapt_generic"), deps, "sim")
        if suite == "libero_90":
            # original-protocol consolidation is queued after the held-out generic evals
            for arm in ["ft"] + ARMS:
                deps = ["feat90"] + ([f"meta_{arm}"] if arm != "ft" else [])
                add(f"cons_{arm}", mod("horsea.consolidate", "--arm", arm),
                    exists(EXP, "consolidate", arm, "final.json"), deps, "sim")
            # 2026-09-24 13:50 user request: larger K, and fast->slow transfer with the fixed memories
            for arm in GA:  # generic-teacher memory -> real-instruction memory-free student
                add(f"gcons_{arm}", mod("horsea.consolidate", "--arm", arm, "--generic", "--mem_dir",
                                         os.path.join(EXP, "memory_generic"), "--out",
                                         os.path.join(EXP, "consolidate_generic", arm)),
                    exists(EXP, "consolidate_generic", arm, "final.json"), ["gfeat_libero_90", f"gmeta_{arm}"], "sim")
            for arm in ["ft"] + GA:
                add(f"gK_{arm}", mod("horsea.adapt_eval", "--arm", arm, "--suite", suite, "--K", 20, 40, *gcommon),
                    all_json(arm, suite, tasks, [20, 40], root="adapt_generic"),
                    ["gfeat_libero_90"] + ([f"gmeta_{arm}"] if arm in GA else []), "sim")
    for arm in []:
        deps = ["feat90"] + ([f"meta_{arm}"] if arm != "ft" else [])
        add(f"cons_{arm}", mod("horsea.consolidate", "--arm", arm), exists(EXP, "consolidate", arm, "final.json"),
            deps, "sim")
    add("eval_base_10", mod("horsea.adapt_eval", "--arm", "base", "--suite", "libero_10"),
        all_json("base", "libero_10", L, [0]), ["feat10"], "sim")
    add("eval_ft_10", mod("horsea.adapt_eval", "--arm", "ft", "--suite", "libero_10"),
        all_json("ft", "libero_10", L, [1, 2, 5, 10]), ["feat10"], "sim")
    for arm in ARMS:
        Ks = [0, 1, 2, 5, 10] if arm == "ttt" else [1, 2, 5, 10]
        add(f"eval_{arm}_10", mod("horsea.adapt_eval", "--arm", arm, "--suite", "libero_10", "--K", *Ks),
            all_json(arm, "libero_10", L, Ks), [f"meta_{arm}", "feat10"], "sim")
    for arm in ARMS:
        add(f"wrong_{arm}", mod("horsea.adapt_eval", "--arm", arm, "--suite", "libero_90", "--K", 5, "--wrong"),
            all_json(arm, "libero_90", H, [5], wrong=True), [f"meta_{arm}"], "sim")
    # --- RoboTTT-style within-episode memory (self-rollout history) on LIBERO-10 -----------------
    # training may start now (GPU is idle during rollouts); evaluation waits for the larger-K runs
    gk = [f"gK_{a}" for a in ["ft"] + GA]
    for m in ["plain", "ttt", "horsea"]:
        add(f"hist_train_{m}", mod("horsea.history", "train", "--mode", m),
            exists(EXP, "history", m, "final.pt"), ["feat10"], "gpu")
    for arm in GA:  # clean transfer: frozen-base teacher (the drifting-teacher run above measures interface drift)
        add(f"gcons2_{arm}", mod("horsea.consolidate", "--arm", arm, "--generic", "--teacher_base", "--mem_dir",
                                  os.path.join(EXP, "memory_generic"), "--out",
                                  os.path.join(EXP, "consolidate_generic_fixedteacher", arm)),
            exists(EXP, "consolidate_generic_fixedteacher", arm, "final.json"), ["gfeat_libero_90", f"gmeta_{arm}"] + gk, "sim")
    for m in ["plain", "ttt", "horsea"]:
        add(f"hist_eval_{m}", mod("horsea.history", "eval", "--mode", m),
            lambda m=m: all(os.path.exists(os.path.join(EXP, "history", m, f"eval_libero_10_t{t}.json")) for t in L),
            [f"hist_train_{m}"] + gk, "sim")
    add("report", mod("horsea.report"), lambda: False, [j["name"] for j in J], "gpu")
    pri = ("hist_", "gcons2_")
    first = [j for j in J if j["name"].startswith(pri)]
    first = [j for j in first if j["name"].startswith("hist_")] + [j for j in first if j["name"].startswith("gcons2_")]
    rest = [j for j in J if not j["name"].startswith(pri)]
    cut = max(i for i, j in enumerate(rest) if j["name"].startswith("gK_")) + 1
    return rest[:cut] + first + rest[cut:]


JOB_PATTERNS = ("horsea.meta_train", "horsea.adapt_eval", "horsea.consolidate", "horsea.features",
                "scripts/train_base.py")


def kill_orphans():
    """A restarted orchestrator must not run a second copy of a job an earlier instance left
    running; every job resumes from its own checkpoints, so stopping it loses little.
    Only python processes of this env whose argv names a job entrypoint are touched."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="ignore") for a in argv if a]
        if not argv or argv[0] != PY:
            continue
        if any(a in JOB_PATTERNS or a.endswith("scripts/train_base.py") for a in argv[1:3]):
            print(time.strftime("%H:%M:%S"), f"stopping leftover job pid {pid}: {' '.join(argv)[:140]}", flush=True)
            try:
                os.kill(int(pid), 15)
            except OSError:
                pass


def cuda_ok():
    r = subprocess.run([PY, "-c", "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()"],
                       capture_output=True)
    return r.returncode == 0


def main():
    os.makedirs(LOGS, exist_ok=True)
    kill_orphans()
    while not cuda_ok():
        print(time.strftime("%H:%M:%S"), "waiting for CUDA (fix the NVIDIA driver mismatch)", flush=True)
        time.sleep(120)
    J = jobs()
    running, tries, finished = {}, {j["name"]: 0 for j in J}, set()
    while True:
        for j in J:
            if j["name"] not in finished and j["name"] not in running and j["done"]():
                finished.add(j["name"])
        for name, (proc, j, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            ok = j["done"]()
            print(time.strftime("%H:%M:%S"), f"{name} exited rc={proc.returncode} after "
                  f"{(time.time() - t0) / 60:.1f} min, complete={ok}", flush=True)
            if ok:
                finished.add(name)
        settled = lambda n: n in finished or (tries[n] >= MAX_TRIES and n not in running)
        if all(settled(j["name"]) for j in J if j["name"] != "report"):
            subprocess.run(jobs()[-1]["cmd"], cwd=HORSEA, stdout=open(os.path.join(LOGS, "report.log"), "a"),
                           stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
            return
        load = {c: sum(1 for (_, j, _) in running.values() if j["cls"] == c) for c in LIMITS}
        for j in J:
            n = j["name"]
            if n == "report" or n in finished or n in running or not all(d in finished for d in j["deps"]):
                continue
            if load["excl"] or (j["cls"] == "excl" and running) or load[j["cls"]] >= LIMITS[j["cls"]]:
                continue
            if tries[n] >= MAX_TRIES:
                continue
            tries[n] += 1
            log = open(os.path.join(LOGS, f"{n}.log"), "a")
            log.write(f"\n===== attempt {tries[n]} at {time.ctime()} =====\n")
            log.flush()
            # cap per-process threads: 3 rollout jobs x (1 + 5 env subprocesses) otherwise each spawn a
            # thread per core (load 36 on 16 cores observed), which slows every rollout down
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            running[n] = (subprocess.Popen(j["cmd"], cwd=HORSEA, stdout=log, stderr=subprocess.STDOUT, env=env),
                          j, time.time())
            load[j["cls"]] += 1
            print(time.strftime("%H:%M:%S"), f"start {n} (attempt {tries[n]})", flush=True)
        failed = [n for n, k in tries.items() if k >= MAX_TRIES and n not in finished and n not in running]
        with open(os.path.join(EXP, "pipeline_status.json"), "w") as f:
            json.dump({"time": time.ctime(), "finished": sorted(finished), "running": sorted(running),
                       "failed": failed, "tries": tries}, f, indent=1)
        time.sleep(30)


if __name__ == "__main__":
    main()
