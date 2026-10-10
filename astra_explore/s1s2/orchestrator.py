"""Sequential, resumable orchestrator of the System-1 / System-2 transfer study.

    python -m s1s2.orchestrator --root <study dir> [--hours 8] [--platforms libero,robotwin_single,...]

One activity at a time. Each activity is a worker subprocess (`python -m s1s2.worker <platform> <activity> <args.json>`) run
with the platform's own interpreter and environment, writing `<root>/<platform>/activities/<id>/result.json`. The
orchestrator only reads results, applies the registered gates, updates `<root>/state.json` and regenerates the report.
Restarting the orchestrator resumes from state.json: finished activities are never re-run, an interrupted activity is re-run
from its start (its partial output is moved aside and accounted for).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from s1s2.ledger import Ledger  # noqa: E402
from s1s2.registry import BUDGET, CELLS, PLATFORMS  # noqa: E402

PLATFORM_RUNTIME = {     # interpreter, cwd, extra env for each platform's worker
    "libero": {"python": "/home/licho/anaconda3/envs/specter/bin/python", "cwd": "/home/licho/workspace/LIBERO",
               "env": {"MUJOCO_GL": "egl"}},
    "robotwin_single": {"python": "/home/licho/anaconda3/envs/robotwin/bin/python", "cwd": "/home/licho/workspace/RoboTwin",
                        "env": {"DISPLAY": ":0", "XAUTHORITY": "/home/licho/.Xauthority",
                                "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json"}, "unset": ["CUDA_VISIBLE_DEVICES"]},
    "robotwin_bimanual": {"python": "/home/licho/anaconda3/envs/robotwin/bin/python", "cwd": "/home/licho/workspace/RoboTwin",
                          "env": {"DISPLAY": ":0", "XAUTHORITY": "/home/licho/.Xauthority",
                                  "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json"}, "unset": ["CUDA_VISIBLE_DEVICES"]},
    "dexhand": {"python": "/home/licho/anaconda3/envs/hongyi/bin/python", "cwd": "/home/licho/workspace/egovla_release/egovla",
                "env": {"OMNI_KIT_ACCEPT_EULA": "YES"}, "activate": "/home/licho/workspace/hongyi_deps/activate_hongyi.sh"},
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Study:
    def __init__(self, root, hours, platforms):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.root, BUDGET["astra_calls_global"], BUDGET["astra_calls_per_platform"], hours)
        self.platforms = platforms
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "created": now(), "activities": {}, "cells": {p: {c: {"status": "PENDING"} for c in CELLS} for p in PLATFORMS},
            "platforms": {p: {} for p in PLATFORMS}, "log": []}
        self.log_path = self.root / "orchestrator.log"

    # ------------------------------------------------------------------ persistence
    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def note(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        self.state["log"].append(line)
        self.state["log"] = self.state["log"][-400:]

    def cell(self, p, c, status=None, **kw):
        rec = self.state["cells"][p][c]
        if status:
            rec["status"] = status
        rec.update(kw)
        rec["updated"] = now()
        self.save()
        return rec

    # ------------------------------------------------------------------ workers
    def duration_estimate(self, kind, platform):
        ds = [a["duration_s"] for a in self.state["activities"].values() if a.get("kind") == kind and a.get("platform") == platform
              and a.get("duration_s")]
        if not ds:
            ds = [a["duration_s"] for a in self.state["activities"].values() if a.get("kind") == kind and a.get("duration_s")]
        return max(ds) if ds else None

    def run(self, platform, kind, args, act_id=None, est_s=None):
        """Run one worker activity with bounded infrastructure retries. Returns its result dict (or a failure dict)."""
        act_id = act_id or f"{platform}__{kind}__{args.get('cell', '')}__{args.get('tag', '')}".rstrip("_")
        acts = self.state["activities"]
        if acts.get(act_id, {}).get("status") == "done":
            return json.loads(Path(acts[act_id]["result"]).read_text())
        est = est_s or self.duration_estimate(kind, platform)
        if est and self.ledger.seconds_left() < est * 1.1:
            self.note(f"SKIP {act_id}: estimated {est:.0f} s > {self.ledger.seconds_left():.0f} s left in the window")
            return {"status": "PENDING_BUDGET", "reason": "wall-clock window"}
        out = self.root / platform / "activities" / act_id
        rt = PLATFORM_RUNTIME[platform]
        env = dict(os.environ)
        for k in rt.get("unset", []):
            env.pop(k, None)
        env.update(rt.get("env", {}))
        env["PATH"] = os.path.dirname(rt["python"]) + ":" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        env.setdefault("HOME", os.path.expanduser("~"))
        env["PYTHONPATH"] = str(HERE.parent) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if platform == "dexhand":
            # the hongyi env needs its activation script (EGL vendor fix, PYTHONPATH of the benchmark)
            pre = f"source {rt['activate']} >/dev/null 2>&1; export PYTHONPATH={HERE.parent}:$PYTHONPATH; "
        else:
            pre = ""
        retries = BUDGET["infra_retries"]
        for attempt in range(retries + 1):
            if out.exists() and not (out / "result.json").exists():
                shutil.move(str(out), str(out) + f".interrupted_{int(time.time())}")    # discarded partial work, kept for accounting
            out.mkdir(parents=True, exist_ok=True)
            (out / "args.json").write_text(json.dumps(dict(args, platform=platform, kind=kind, study_root=str(self.root))))
            acts[act_id] = {"platform": platform, "kind": kind, "status": "running", "start": now(), "attempt": attempt, "out": str(out)}
            self.save()
            self.note(f"START {act_id} (try {attempt + 1}/{retries + 1})")
            t0 = time.time()
            cmd = f"{pre}cd {rt['cwd']} && exec {rt['python']} -u -m s1s2.worker {platform} {kind} {out / 'args.json'}"
            with open(out / "worker.log", "a") as logf:
                rc = subprocess.call(["bash", "-c", cmd], env=env, stdout=logf, stderr=subprocess.STDOUT)
            dur = time.time() - t0
            res_path = out / "result.json"
            if rc == 0 and res_path.exists():
                acts[act_id].update(status="done", end=now(), duration_s=round(dur, 1), result=str(res_path))
                self.save()
                res = json.loads(res_path.read_text())
                self.note(f"DONE  {act_id} in {dur:.0f} s: {res.get('summary', '')}")
                return res
            acts[act_id].update(status="failed_try", end=now(), duration_s=round(dur, 1), rc=rc)
            self.save()
            self.note(f"FAIL  {act_id} rc={rc} after {dur:.0f} s (see {out / 'worker.log'})")
            if res_path.exists():
                res = json.loads(res_path.read_text())
                if res.get("status") in ("BLOCKED_INTERFACE", "BLOCKED_DATA", "PENDING_BUDGET"):
                    acts[act_id].update(status="done", result=str(res_path))
                    self.save()
                    return res
        acts[act_id]["status"] = "failed"
        self.save()
        return {"status": "FAILED", "reason": f"worker failed {retries + 1} times (last rc {rc})", "out": str(out)}


# ====================================================================== the registered plan
def dev_starts(pl, n):
    return pl["pools"]["dev"][:n]


def student_cells_blocked(study, p, status, reason, cells=("E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12")):
    for c in cells:
        if study.state["cells"][p][c]["status"] in ("PENDING", "RUNNING"):
            study.cell(p, c, status, reason=reason)


def stage_audit(study, p):
    res = study.run(p, "audit", {"tag": "v1"})
    pl = study.state["platforms"][p]
    pl["audit"] = {k: res.get(k) for k in ("status", "summary", "alignment", "manifest", "reasons")}
    pl["cohort"] = res.get("cohort", "unknown")
    pl["student"] = res.get("student", {})
    pl["pools"] = res.get("pools", {})
    pl["runtime"] = res.get("runtime", {})
    pl["pretrained_ckpt"] = res.get("pretrained_ckpt")
    pl["init_ckpt"] = res.get("init_ckpt")
    study.save()
    if res.get("status") != "OK":
        student_cells_blocked(study, p, res.get("status", "FAILED"), res.get("summary"))
        if res.get("teacher_status") != "OK":
            study.cell(p, "E01", res.get("status", "FAILED"), reason=res.get("summary"))
    if not res.get("pretrained_ckpt") and res.get("status") in ("OK", "BLOCKED_INTERFACE"):
        study.cell(p, "E02", "NOT_AVAILABLE", reason="no compatible pretrained student exists for this platform")
    return res.get("status") == "OK"


def stage_source(study, p, target_successes, call_allotment, tag):
    pl = study.state["platforms"][p]
    res = study.run(p, "teacher", {"cell": "SRC", "pool": "source", "starts": pl["pools"]["source"], "target_successes": target_successes,
                                   "call_allotment": call_allotment, "tag": tag})
    pl.setdefault("data", {})
    pl["data"].update(res.get("data", {}))
    study.save()
    if res.get("status") != "OK":
        return False
    if not pl.get("replay_done") and res.get("first_valid_episode"):
        rr = study.run(p, "replay", {"episode": res["first_valid_episode"], "tag": "first"})
        pl["audit"]["alignment"] = rr.get("summary")
        pl["replay"] = rr
        pl["replay_done"] = True
        study.save()
        if rr.get("status") != "OK":
            student_cells_blocked(study, p, "BLOCKED_DATA", f"replay diagnostic failed: {rr.get('summary')}")
            return False
    return True


def eval_cell(study, p, c, ckpt, n_starts, extra=None, kind="eval"):
    pl = study.state["platforms"][p]
    starts = dev_starts(pl, n_starts)
    args = dict({"cell": c, "ckpt": ckpt, "starts": starts, "pool": "dev", "tag": f"dev{n_starts}"}, **(extra or {}))
    study.cell(p, c, "RUNNING")
    res = study.run(p, kind, args)
    if res.get("status") in ("PENDING_BUDGET",):
        study.cell(p, c, "PENDING_BUDGET", reason=res.get("reason"))
        return None
    if res.get("status") != "OK":
        study.cell(p, c, "FAILED" if res.get("status") in (None, "FAILED") else res["status"], reason=res.get("summary") or res.get("reason"))
        return None
    status = "COMPLETE" if res["n"] >= len(starts) else "PENDING_BUDGET"     # some starts not reached: exact denominator kept
    study.cell(p, c, status, result={"successes": res["successes"], "n": res["n"], "pool": "dev", "per_episode": res.get("per_episode"),
                                     "path": res.get("out"), "starts_requested": len(starts)},
               reason=None if status == "COMPLETE" else f"{len(starts) - res['n']} of {len(starts)} starts not reached: {res.get('stop')}")
    return res


def train_cell(study, p, c, init, datasets, updates, mix=None):
    res = study.run(p, "train", {"cell": c, "init": init, "datasets": datasets, "updates": updates, "mix": mix, "tag": "s0"})
    if res.get("status") != "OK":
        status = res.get("status") if res.get("status") in ("BLOCKED_DATA", "BLOCKED_INTERFACE", "PENDING_BUDGET") else "FAILED"
        study.cell(p, c, status, reason=res.get("summary") or res.get("reason"))
        return None
    study.state["platforms"][p].setdefault("checkpoints", {})[c] = res["ckpt"]
    study.save()
    return res["ckpt"]


def stage_reference_and_scale(study, p, n_dev):
    """E02, E03 (D1), E04 (D3), E05 (DN) from the same initialization and update budget."""
    pl = study.state["platforms"][p]
    pl.setdefault("checkpoints", {})
    from s1s2.datasets import aliases, read_bank, source_sets, write_membership
    sets = source_sets(read_bank(study.root, p))
    pl.setdefault("data", {})["sets"] = sets
    for name in ("D1", "D3", "DN", "Dfail"):
        write_membership(study.root, p, name, sets[name] or [], "chronological prefix rule (registry)" +
                         (" - unavailable: fewer than three successful source episodes" if sets[name] is None else ""), {"N": sets["N"]})
    study.save()
    if pl.get("pretrained_ckpt") and study.state["cells"][p]["E02"]["status"] == "PENDING":
        eval_cell(study, p, "E02", pl["pretrained_ckpt"], n_dev)
    if sets["N"] == 0:
        student_cells_blocked(study, p, "BLOCKED_DATA", "no successful valid source episode", cells=("E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12"))
        return None
    U = pl.get("runtime", {}).get("U", 1000)
    init = pl.get("init_ckpt")
    al = aliases(sets)
    anchor = None
    for c, d in (("E03", "D1"), ("E04", "D3"), ("E05", "DN")):
        if study.state["cells"][p][c]["status"] in ("COMPLETE", "DUPLICATE"):
            anchor = pl["checkpoints"].get(c) or pl["checkpoints"].get(study.state["cells"][p][c].get("alias_of"))
            continue
        if c == "E04" and sets["D3"] is None:
            study.cell(p, c, "BLOCKED_DATA", reason=f"D3 needs three successful source episodes; only {sets['N']} were obtained within the source allotment")
            continue
        if c in al:
            src = al[c]
            study.cell(p, c, "DUPLICATE", alias_of=src, result=study.state["cells"][p][src].get("result"),
                       reason=f"{d} is identical to the {src} source set")
            pl["checkpoints"][c] = pl["checkpoints"].get(src)
            anchor = pl["checkpoints"][c]
            continue
        ck = train_cell(study, p, c, init, [d], U)
        if ck:
            eval_cell(study, p, c, ck, n_dev)
            anchor = ck
    pl["anchor"] = anchor
    study.save()
    return anchor


def anchor_signal(study, p, n_dev_expanded):
    """Gate: an anchor with no success in five easy dev evaluations does not trigger expensive hybrid / collection runs."""
    pl = study.state["platforms"][p]
    rec = study.state["cells"][p]["E05"]
    if rec.get("alias_of"):
        rec = study.state["cells"][p][rec["alias_of"]]
    r = rec.get("result") or {}
    if r.get("successes", 0) > 0:
        return True
    if r.get("n", 0) < n_dev_expanded and pl.get("anchor"):
        res = study.run(p, "eval", {"cell": "E05", "ckpt": pl["anchor"], "starts": dev_starts(pl, n_dev_expanded)[r.get("n", 0):],
                                    "pool": "dev", "tag": f"dev_expand{n_dev_expanded}"})
        if res.get("status") == "OK":
            succ, n = r.get("successes", 0) + res["successes"], r.get("n", 0) + res["n"]
            c = rec.get("alias_of") or "E05"
            study.state["cells"][p]["E05"].setdefault("result", {})
            study.state["cells"][p]["E05"]["result"].update(successes=succ, n=n,
                                                           per_episode=(r.get("per_episode") or []) + (res.get("per_episode") or []))
            study.save()
            return succ > 0
    return False


def stage_anchor_uses(study, p, n_dev, n_dev_expanded):
    pl = study.state["platforms"][p]
    anchor = pl.get("anchor")
    if not anchor:
        return
    if study.state["cells"][p]["E06"]["status"] == "PENDING":
        ex = pl.get("runtime", {}).get("exec_chunk")
        if not ex or ex <= 1:
            study.cell(p, "E06", "NOT_APPLICABLE", reason="the student already executes one action before replanning" if ex else "unknown executed chunk")
        else:
            eval_cell(study, p, "E06", anchor, n_dev, extra={"exec_chunk": max(1, ex // 2)})
    if not anchor_signal(study, p, n_dev_expanded):
        for c in ("E08", "E09", "E11", "E12"):
            if study.state["cells"][p][c]["status"] == "PENDING":
                study.cell(p, c, "NO_STUDENT_SIGNAL", reason=f"anchor (E05) had no success in {n_dev_expanded} dev evaluations")
        pl["no_student_signal"] = True
        study.save()
        return
    if any(study.state["cells"][p][c]["status"] in ("PENDING", "RUNNING") for c in ("E08", "E09")):
        stage_hybrid_pairs(study, p, n_dev)


def stage_hybrid_pairs(study, p, n_dev):
    """E08 and E09 run start by start (E08, then E09 on the same start). A start is begun only if the remaining budget covers
    both takeover ceilings, so a binding budget never leaves an unpaired start (the student prefix is identical in both)."""
    pl = study.state["platforms"][p]
    anchor = pl["anchor"]
    starts = dev_starts(pl, n_dev)
    need = 2 * BUDGET["astra_calls_per_takeover"]
    agg = {}
    for c in ("E08", "E09"):
        r = study.state["cells"][p][c].get("result") or {}
        agg[c] = {"successes": r.get("successes", 0), "n": r.get("n", 0), "per_episode": list(r.get("per_episode") or []), "pool": "dev"}
        study.cell(p, c, "RUNNING")
    stop, failed = None, {}
    for s in starts:
        have = {c: {e.get("start_id") for e in agg[c]["per_episode"]} for c in agg}
        if s in have["E08"] and s in have["E09"]:
            continue
        if study.ledger.remaining(p) < need:
            stop = f"PENDING_BUDGET: {study.ledger.remaining(p)} platform calls left, below two takeover ceilings ({need})"
            break
        for c, hist in (("E08", 0), ("E09", 3)):
            if s in have[c]:
                continue
            res = study.run(p, "hybrid", {"cell": c, "ckpt": anchor, "starts": [s], "pool": "dev", "history": hist, "tag": f"dev_{s}"})
            if res.get("status") == "OK":
                eps = [e for e in res.get("per_episode", []) if e.get("status") != "PENDING_BUDGET"]
                agg[c]["per_episode"] += eps
                agg[c]["n"] += len(eps)
                agg[c]["successes"] += sum(1 for e in eps if e.get("success"))
                study.cell(p, c, "RUNNING", result=dict(agg[c]))
            elif res.get("status") == "PENDING_BUDGET":
                stop = f"PENDING_BUDGET: {res.get('reason') or res.get('stop') or 'budget'}"
                break
            else:
                failed[c] = res.get("reason") or res.get("summary") or "worker failed"
                break
        if stop or failed:
            break
    for c in ("E08", "E09"):
        if c in failed:
            study.cell(p, c, "FAILED", result=dict(agg[c], starts_requested=len(starts)), reason=failed[c])
            continue
        done = agg[c]["n"] >= len(starts)
        study.cell(p, c, "COMPLETE" if done else "PENDING_BUDGET", result=dict(agg[c], starts_requested=len(starts)),
                   reason=None if done else (stop or "the paired cell failed; remaining starts not run"))


def stage_consolidation(study, p, n_dev, corr_calls):
    pl = study.state["platforms"][p]
    anchor = pl.get("anchor")
    if not anchor:
        return
    dU = pl.get("runtime", {}).get("dU", 500)
    # E10 needs no new data and runs even without student signal (cheap control)
    if study.state["cells"][p]["E10"]["status"] == "PENDING":
        ck = train_cell(study, p, "E10", anchor, ["DN"], dU)
        if ck:
            eval_cell(study, p, "E10", ck, n_dev)
    if pl.get("no_student_signal"):
        return
    # adaptation-pool collection: corrections (takeover from the student's state) and matched extra reset demos
    if not pl.get("corr_collected"):
        spent = study.ledger.summary()["per_cell"].get(f"{p}/CORR", 0)     # calls of an interrupted earlier try count too
        res = study.run(p, "hybrid", {"cell": "CORR", "ckpt": anchor, "starts": pl["pools"]["adapt"], "pool": "adapt", "history": 0,
                                      "call_allotment": max(0, corr_calls - spent), "collect": True, "tag": "corr"})
        pl["corr"] = {k: res.get(k) for k in ("status", "summary", "n", "successes", "astra_calls", "native_steps", "student_steps",
                                              "usable_windows", "dataset", "stop")}
        pl["corr_collected"] = res.get("status") == "OK"
        study.save()
    if not pl.get("extra_collected") and pl.get("corr_collected"):
        c = pl["corr"]
        from s1s2.datasets import read_bank as _rb
        ex_bank = [e for e in _rb(study.root, p) if e.get("cell") == "EXTRA"]
        call_left = (c.get("astra_calls") or 0) - study.ledger.summary()["per_cell"].get(f"{p}/EXTRA", 0)
        step_left = (c.get("native_steps") or 0) - sum(e.get("native_steps", 0) for e in ex_bank)
        if call_left < 1 or step_left < 1:
            res = {"status": "OK", "summary": f"no matched budget left for extra demos (correction collection used {c.get('astra_calls')} calls, "
                                              f"{c.get('native_steps')} native steps)", "data": {"episodes": len(ex_bank)}}
        else:   # same additional environment-interaction budget and the same call ceiling as correction collection
            res = study.run(p, "teacher", {"cell": "EXTRA", "pool": "adapt", "starts": pl["pools"]["adapt"], "target_successes": 10 ** 6,
                                           "call_allotment": call_left, "step_allotment": step_left, "strict_allotment": True, "tag": "extra"})
        pl["extra"] = {k: res.get(k) for k in ("status", "summary", "data", "stop")}
        pl["extra_collected"] = res.get("status") == "OK"
        from s1s2.datasets import read_bank, write_membership
        ex_ids = [e["episode_id"] for e in read_bank(study.root, p) if e.get("cell") == "EXTRA" and e.get("success") and e.get("valid")]
        pl["extra"]["dataset"] = write_membership(study.root, p, "Dextra", ex_ids,
                                                  "successful Astra-from-reset episodes on the adaptation pool (matched collection budget)")
        pl["extra"]["data"] = dict(pl["extra"].get("data") or {}, successes=len(ex_ids))
        study.save()
    def _budget_limited(rec):
        return bool(rec) and (rec.get("status") == "PENDING_BUDGET" or str(rec.get("stop") or "").startswith("PENDING_BUDGET"))

    if study.state["cells"][p]["E11"]["status"] == "PENDING":
        corr = pl.get("corr") or {}
        if not corr.get("dataset"):
            if _budget_limited(corr) or not corr:
                study.cell(p, "E11", "PENDING_BUDGET", reason=f"correction collection was budget-limited before any recovery: {corr.get('summary') or corr.get('stop')}")
            else:
                study.cell(p, "E11", "BLOCKED_NO_CORRECTIONS", reason=f"no successful recovery on the adaptation pool: {corr.get('summary')}")
        else:
            ck = train_cell(study, p, "E11", anchor, ["DN", "Dcorr"], dU, mix=[0.5, 0.5])
            if ck:
                eval_cell(study, p, "E11", ck, n_dev)
    if study.state["cells"][p]["E12"]["status"] == "PENDING":
        ex = ((pl.get("extra") or {}).get("data") or {})
        if not ex.get("successes"):
            if not pl.get("corr_collected") or _budget_limited(pl.get("extra")):
                study.cell(p, "E12", "PENDING_BUDGET", reason=f"matched extra-demo collection not completed: {(pl.get('extra') or pl.get('corr') or {}).get('summary')}")
            else:
                study.cell(p, "E12", "BLOCKED_DATA", reason=f"no successful additional reset demonstration under the matched budget: {(pl.get('extra') or {}).get('summary')}")
        else:
            ck = train_cell(study, p, "E12", anchor, ["DN", "Dextra"], dU, mix=[0.5, 0.5])
            if ck:
                eval_cell(study, p, "E12", ck, n_dev)


def stage_failures(study, p, n_dev):
    pl = study.state["platforms"][p]
    if study.state["cells"][p]["E07"]["status"] != "PENDING" or not pl.get("anchor"):
        return
    sets = pl["data"].get("sets", {})
    if sets.get("Dfail") == sets.get("DN"):
        study.cell(p, "E07", "DUPLICATE", alias_of="E05", result=study.state["cells"][p]["E05"].get("result"),
                   reason="no failed valid episode before the N-th success: identical to DN")
        return
    ck = train_cell(study, p, "E07", pl.get("init_ckpt"), ["Dfail"], pl.get("runtime", {}).get("U", 1000))
    if ck:
        eval_cell(study, p, "E07", ck, n_dev)


def stage_reference(study, p, n_dev):
    """E01: Astra from reset on the same development starts as the students, with whatever platform budget remains."""
    pl = study.state["platforms"][p]
    if study.state["cells"][p]["E01"]["status"] != "PENDING" or not pl.get("pools"):
        return
    left = study.ledger.remaining(p)
    if left < BUDGET["astra_calls_per_episode_cap"]:
        study.cell(p, "E01", "PENDING_BUDGET", reason=f"{left} platform calls left, below one episode's {BUDGET['astra_calls_per_episode_cap']}-call ceiling")
        return
    study.cell(p, "E01", "RUNNING")
    res = study.run(p, "teacher", {"cell": "E01", "pool": "dev", "starts": dev_starts(pl, n_dev), "target_successes": 10 ** 6, "tag": "e01"})
    d = res.get("data") or {}
    if res.get("status") == "OK" and d.get("episodes"):
        study.cell(p, "E01", "COMPLETE" if d["episodes"] >= n_dev else "PENDING_BUDGET",
                   result={"successes": d.get("successes"), "n": d.get("episodes"), "pool": "dev", "astra_calls": d.get("astra_calls"),
                           "native_steps": d.get("native_steps"), "starts_requested": n_dev},
                   reason=None if d["episodes"] >= n_dev else res.get("stop"))
    else:
        study.cell(p, "E01", "PENDING_BUDGET" if res.get("status") == "PENDING_BUDGET" else "FAILED", reason=res.get("summary") or res.get("reason"))


PRINCIPAL = ("E02", "E05", "E10", "E11", "E12")      # standalone students of the principal comparisons (spec sections 9-10)


def stage_locked_test(study, p):
    """Settings are frozen (all training finished): the principal standalone students on fresh test starts. A duplicate cell
    is evaluated once, under the cell it aliases."""
    pl = study.state["platforms"][p]
    test = (pl.get("pools") or {}).get("test", [])[:BUDGET["test_starts"]]
    if not test:
        return
    for c in PRINCIPAL:
        rec = study.state["cells"][p][c]
        name = rec.get("alias_of") if rec["status"] == "DUPLICATE" else c
        src = study.state["cells"][p].get(name, {})
        if src.get("status") != "COMPLETE" or src.get("test"):
            continue
        ck = pl.get("pretrained_ckpt") if name == "E02" else pl.get("checkpoints", {}).get(name)
        if not ck:
            continue
        res = study.run(p, "eval", {"cell": name, "ckpt": ck, "starts": test, "pool": "test", "tag": f"test{len(test)}"})
        if res.get("status") == "OK":
            src["test"] = {"successes": res["successes"], "n": res["n"], "pool": "test", "per_episode": res.get("per_episode")}
        else:
            src["test"] = {"status": res.get("status"), "reason": res.get("reason") or res.get("summary")}
        study.save()


def stage_diagnosis(study, p):
    """Spec section 10: an anchor without student signal -> 'prioritize data/interface diagnosis'. Cheapest decisive check
    (no Astra calls, no new configuration): the original student, the D1 student and the anchor on the SOURCE starts whose
    Astra episodes they were trained on. Failure there points at the data/interface/recipe; success there but not on
    development starts points at generalization from very few demonstrations. Stored outside the 12 x 4 matrix."""
    pl = study.state["platforms"][p]
    if not pl.get("no_student_signal") or not pl.get("anchor") or pl.get("diagnosis"):
        return
    from s1s2.datasets import read_bank
    bank = {e["episode_id"]: e for e in read_bank(study.root, p)}
    starts = [bank[e]["start_id"] for e in (pl.get("data", {}).get("sets", {}).get("DN") or []) if e in bank]
    if not starts:
        return
    out = {"rule": "E02, E03, E05 on the source starts of DN (their own training starts); no Astra calls", "starts": starts, "results": {}}
    for c, ck in (("E02", pl.get("pretrained_ckpt")), ("E03", pl.get("checkpoints", {}).get("E03")), ("E05", pl.get("anchor"))):
        if not ck:
            continue
        res = study.run(p, "eval", {"cell": f"DIAG_{c}", "ckpt": ck, "starts": starts, "pool": "source", "tag": "train_starts"})
        out["results"][c] = ({k: res.get(k) for k in ("successes", "n", "per_episode")} if res.get("status") == "OK"
                             else {"status": res.get("status"), "reason": res.get("reason") or res.get("summary")})
        pl["diagnosis"] = out
        study.save()


EXPAND_ORDER = ("E05", "E02", "E03", "E06", "E07", "E10", "E11", "E12", "E04")   # standalone students only (no Astra calls)


def stage_dev_expand(study, p, n_exp):
    """Spec section 10: expand important comparisons to five development starts if time permits. Last stage, after the
    locked test evaluation that the spec prefers; standalone students only (hybrids would spend Astra calls)."""
    pl = study.state["platforms"][p]
    starts = dev_starts(pl, n_exp)
    ex = pl.get("runtime", {}).get("exec_chunk")
    for c in EXPAND_ORDER:
        rec = study.state["cells"][p][c]
        r = rec.get("result") or {}
        if rec.get("status") != "COMPLETE" or r.get("pool", "dev") != "dev" or r.get("n", 0) >= n_exp or study.ledger.seconds_left() < 300:
            continue
        if c == "E02":
            ck, extra = pl.get("pretrained_ckpt"), {}
        elif c == "E06":
            ck, extra = pl.get("anchor"), {"exec_chunk": max(1, ex // 2)} if ex else {}
        else:
            ck, extra = pl.get("checkpoints", {}).get(c), {}
        if not ck:
            continue
        have = {e.get("start_id") for e in r.get("per_episode") or []}
        todo = [s for s in starts if s not in have]
        res = study.run(p, "eval", dict({"cell": c, "ckpt": ck, "starts": todo, "pool": "dev", "tag": f"dev_expand{n_exp}"}, **extra))
        if res.get("status") != "OK":
            continue                                   # time-skipped or failed: the 3-start result stands
        r.update(successes=r.get("successes", 0) + res["successes"], n=r.get("n", 0) + res["n"],
                 per_episode=(r.get("per_episode") or []) + (res.get("per_episode") or []), starts_requested=n_exp)
        rec["expanded_to"] = n_exp
        study.save()


def report(study):
    try:
        subprocess.call([sys.executable, "-m", "s1s2.report", "--root", str(study.root)], cwd=str(HERE.parent),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def enabled_platforms(study):
    f = study.root / "platforms_enabled.json"
    if f.exists():
        try:
            return [p for p in json.loads(f.read_text()) if p in PLATFORMS]
        except json.JSONDecodeError:
            pass
    return list(study.platforms)


def run_stage(study, stage, p, n_dev, n_exp):
    pl = study.state["platforms"][p]
    rt = pl.get("runtime", {})
    if stage == "audit":
        stage_audit(study, p)
        return
    if pl.get("audit", {}).get("status") != "OK" and stage != "reference":
        return                                         # students blocked: only the Astra-only reference (E01) can still run
    if stage == "source_min":
        stage_source(study, p, 3, rt.get("source_calls", 60), "min3")
    elif stage == "source_unlock":
        # declared 07:30 (before dex source collection; applies to every platform): a source allotment that ended without any
        # success continues one full-ceiling episode at a time, within the platform's 150-call reservation, until the first
        # success (spec step 2, "unlock minimal valid source data", ranks above the step-8 Astra-only reference)
        d = pl.get("data", {})
        if d.get("episodes", 0) > 0 and d.get("successes", 0) == 0 and study.ledger.remaining(p) >= BUDGET["astra_calls_per_episode_cap"]:
            stage_source(study, p, 1, study.ledger.remaining(p), "unlock")
    elif stage == "source_to10":
        src_total = rt.get("source_calls", 60)
        used = study.ledger.summary()["per_cell"].get(f"{p}/SRC", 0)
        if pl.get("data", {}).get("successes", 0) >= 3 and src_total - used >= 5:
            stage_source(study, p, 10, src_total - used, "to10")
    elif stage == "scale":
        stage_reference_and_scale(study, p, n_dev)
    elif stage == "anchor":
        stage_anchor_uses(study, p, n_dev, n_exp)
    elif stage == "diagnosis":
        stage_diagnosis(study, p)
    elif stage == "consolidation":
        stage_consolidation(study, p, n_dev, rt.get("corr_calls", 25))   # declared at launch in each audit (25)
    elif stage == "failures":
        stage_failures(study, p, n_dev)
    elif stage == "reference":
        stage_reference(study, p, n_dev)
    elif stage == "locked_test":
        stage_locked_test(study, p)
    elif stage == "dev_expand":
        stage_dev_expand(study, p, n_exp)


STAGE_ORDER = ["audit", "source_min", "source_unlock", "source_to10", "scale", "anchor", "diagnosis", "consolidation", "failures", "reference", "locked_test", "dev_expand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hours", type=float, default=BUDGET["window_hours"])
    ap.add_argument("--platforms", default=",".join(PLATFORMS), help="platforms whose adapters exist at launch; later ones join via platforms_enabled.json")
    a = ap.parse_args()
    study = Study(a.root, a.hours, [p for p in a.platforms.split(",") if p])
    if not (study.root / "platforms_enabled.json").exists():
        (study.root / "platforms_enabled.json").write_text(json.dumps(study.platforms))
    study.note(f"orchestrator up (pid {os.getpid()}); window ends {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(study.ledger.deadline()))}")
    n_dev, n_exp = BUDGET["dev_starts_initial"], BUDGET["dev_starts_expanded"]
    done = study.state.setdefault("stages_done", {})
    while True:
        enabled = enabled_platforms(study)
        progressed = False
        for stage in STAGE_ORDER:                      # stage-major across platforms (spec section 10)
            for p in enabled:
                if stage in done.get(p, []):
                    continue
                if study.ledger.seconds_left() < 120:
                    break
                run_stage(study, stage, p, n_dev, n_exp)
                done.setdefault(p, []).append(stage)
                study.save()
                report(study)
                progressed = True
                if (study.root / "RELOAD").exists():
                    (study.root / "RELOAD").unlink()
                    study.note("RELOAD requested: re-executing the orchestrator with the current code (state resumes)")
                    os.execv(sys.executable, [sys.executable, "-u", "-m", "s1s2.orchestrator"] + sys.argv[1:])
                break                                  # re-evaluate from the earliest stage (newly enabled platforms catch up)
            if progressed:
                break
        if progressed:
            continue
        if study.ledger.seconds_left() < 120:
            break
        waiting = [p for p in PLATFORMS if p not in enabled and study.state["cells"][p]["E01"]["status"] == "PENDING"]
        if not waiting:
            break
        time.sleep(60)                                 # adapters for these platforms are still being prepared
    for p in PLATFORMS:
        for c, rec in study.state["cells"][p].items():
            if rec["status"] in ("PENDING", "RUNNING"):
                why = "adapter not enabled before the window ended" if p not in enabled_platforms(study) else "not reached within the window / budget"
                study.cell(p, c, "PENDING_BUDGET" if p in enabled_platforms(study) else "BLOCKED_INTERFACE", reason=why)
    report(study)
    study.note(f"orchestrator finished; ledger {json.dumps(study.ledger.summary())}")


if __name__ == "__main__":
    main()
