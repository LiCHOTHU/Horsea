"""Task-generic local-execution-adaptation pilot (spec-driven copy of runner.py; same arms, schedule, logging, budget).

    python -m s1local.runner --out <dir> [--budget-min 120] [--reserve-min 14] [--ops grasp,transport,door] [--conds 0-5]
                             [--attempts 9] [--smoke-only]

Writes <out>/manifest.json, smoke.json, rollouts.jsonl (one physical execution each, with a compact trajectory),
attempts.jsonl (one line per arm x attempt, incl. proposal state and updates), blocks.jsonl (per matched block summary),
timing.json. Resumable: completed rollouts (by key) are reused after a restart.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/licho/workspace/Horsea/astra_explore")
from s1local import env as E  # noqa: E402
from s1local import generic as G  # noqa: E402
from s1local import search as S  # noqa: E402
from s1local.specs import SPECS, make_region_fns  # noqa: E402

# predetermined, balanced condition order (mass factor, friction factor): nominal first, extremes interleaved
CONDITIONS = [(1.0, 1.0), (0.8, 0.7), (1.2, 1.0), (1.0, 0.7), (0.8, 1.0), (1.2, 0.7)]
EVAL_STARTS = ["e1", "e2", "e3"]
OP_ORDER = []                     # set from the spec in main()


def jl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


class Runner:
    def __init__(self, spec, out, budget_s, reserve_s):
        self.spec = spec
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self.budget_s, self.reserve_s = budget_s, reserve_s
        from isaac_env import IsaacDexTask
        self.T = IsaacDexTask(spec["task"], cfg_hook=E.no_render)
        self.t_ready = time.time()
        self.phys = G.Physics(self.T)
        self.prog = G.load_program(spec["source_run"], spec["source_attempt"])
        self.ops = spec["ops"]
        for o in self.ops.values():                 # joint given by the env's own attribute name -> its index
            if isinstance(o["object"], list) and isinstance(o["object"][1], str):
                o["object"] = [o["object"][0], int(getattr(self.T.U, o["object"][1]))]
        self.region = make_region_fns(self.T)
        self.recording, self.trace = False, []
        step0 = self.T._step

        def _step(a):
            o = step0(a)
            if self.recording:
                self.trace.append(G.track(self.T, self.ops[self.cur_op]))
            return o
        self.T._step = _step
        self.done = {}
        rp = self.out / "rollouts.jsonl"
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done[r["key"]] = r
        self.starts = {}
        self.op_rate = {}
        self.cur_op = None
        self.rest_z = {}

    def left(self):
        return self.budget_s - (time.time() - self.t_start)

    # ------------------------------------------------------------------ starts
    def build_starts(self):
        rec, cache = {}, {}
        for op in OP_ORDER:
            o = self.ops[op]
            for sid in ["nominal"] + EVAL_STARTS:
                key = (tuple(o["prefix"]), tuple(o["shift"]), sid)
                if key in cache:                    # same prefix, shift and start as an earlier operation: share its start
                    snap, rest = cache[key]
                    E.restore(self.T, snap)
                else:
                    self.phys.nominal()
                    G.reset_with_shift(self.T, o["shift"], sid)
                    rest = {n: float(ob.data.root_state_w[0, 2].item()) for n, ob in self.T.U.scene.rigid_objects.items()}
                    for i in o["prefix"]:
                        E.act(self.T, self.prog[i])
                    snap = E.snapshot(self.T)
                    cache[key] = (snap, rest)
                if not isinstance(o["object"], list):
                    self.rest_z[(op, sid)] = rest[o["object"]]
                st = G.object_state(self.T)
                ok, info = G.start_valid(o, st, self.rest_z.get((op, sid)))
                self.starts[(op, sid)] = {"snap": snap, "valid": ok, "info": info}
                rec[f"{op}/{sid}"] = {"valid": ok, **info}
        return rec

    def _restore(self, op, sid):
        E.restore(self.T, self.starts[(op, sid)]["snap"])

    # ------------------------------------------------------------------ one physical execution
    def rollout(self, op, ci, sid, theta, slot):
        theta = np.asarray(theta, float)
        key = f"{op}|c{ci}|{sid}|{slot}|" + ",".join(f"{x:.6f}" for x in theta)
        if key in self.done:
            return self.done[key]
        if not self.starts[(op, sid)]["valid"]:
            r = {"key": key, "op": op, "cond": ci, "start": sid, "slot": slot, "theta": theta.tolist(), "init_failure": True,
                 "J": None, "S": None, "steps": 0, "wall_s": 0.0}
            jl(self.out / "rollouts.jsonl", r)
            self.done[key] = r
            return r
        o = self.ops[op]
        t0 = time.time()
        self._restore(op, sid)
        phys = self.phys.apply(o, *CONDITIONS[ci])
        start = G.object_state(self.T)
        edited = G.edited_program(o, self.prog, theta, self.T._ee(o["hand"])[:3])
        self.cur_op, self.trace, self.recording = op, [], True
        steps = 0
        for d, sp in edited:
            steps += E.act(self.T, d, speed=sp)
        steps += E.hold_window(self.T, o["window"])
        self.recording = False
        final = G.object_state(self.T)
        ev = G.evaluate(o, start, self.trace, final, self.region)
        tr = [([round(float(x), 4) for x in t[0]] if hasattr(t[0], "__len__") else round(float(t[0]), 4)) for t in self.trace[::2]]
        r = {"key": key, "op": op, "cond": ci, "condition": CONDITIONS[ci], "start": sid, "slot": slot, "theta": theta.round(6).tolist(),
             "init_failure": False, "steps": steps, "wall_s": round(time.time() - t0, 2), **ev, "physics_applied": phys, "trajectory_every2": tr}
        jl(self.out / "rollouts.jsonl", r)
        self.done[key] = r
        self.op_rate.setdefault(op, []).append(r["wall_s"])
        return r

    # ------------------------------------------------------------------ one matched block
    def block(self, op, ci, n_adapt=9):
        ref = G.reference(self.ops[op], self.prog)
        lo, hi = G.bounds(self.ops[op], ref)
        q0 = S.initial(ref, lo, hi)
        U = S.uniforms(op, ci, len(ref))
        if not self.starts[(op, "nominal")]["valid"]:          # no valid adaptation start: initialization failure, not a task result
            summ = {"op": op, "cond": ci, "condition": CONDITIONS[ci], "init_failure": "adaptation start invalid",
                    "start_info": self.starts[(op, "nominal")]["info"], "time": time.strftime("%H:%M:%S")}
            jl(self.out / "blocks.jsonl", summ)
            return dict(summ, first_success=None, eval_success=None, selected=None)
        att = {a: [] for a in "ABC"}

        def log(arm, k, theta, r, prop=None, upd=None):
            rec = {"op": op, "cond": ci, "condition": CONDITIONS[ci], "arm": arm, "attempt": k, "phase": "adapt" if isinstance(k, int) else "eval",
                   "start": r["start"], "theta": list(map(float, theta)), "rollout_key": r["key"], "J": r["J"], "S": r["S"], "P": r.get("P"),
                   "D": r.get("D"), "V": r.get("V"), "steps": r["steps"], "init_failure": r["init_failure"],
                   "proposal": None if prop is None else {"mu": prop["mu"].round(5).tolist(), "sigma": prop["sigma"].round(5).tolist()},
                   "update": upd}
            att[arm].append(rec)
            jl(self.out / "attempts.jsonl", rec)

        r1 = self.rollout(op, ci, "nominal", ref, "a1")                       # attempt 1: Astra's program, every arm
        for a in "ABC":
            log(a, 1, ref, r1, q0 if a != "A" else None)
        for k in range(2, n_adapt + 1):                                       # A: repeat Astra's program
            log("A", k, ref, self.rollout(op, ci, "nominal", ref, f"A{k}"))
        mem = {"B": [(ref, r1["J"], 1)], "C": [(ref, r1["J"], 1)]}
        for k in range(2, min(5, n_adapt) + 1):                               # B = C: four shared draws from q0
            th = S.sample(q0["mu"], q0["sigma"], lo, hi, U[k - 1])
            r = self.rollout(op, ci, "nominal", th, f"BC{k}")
            for a in "BC":
                log(a, k, th, r, q0)
                mem[a].append((th, r["J"], k))
        qC, upd = S.cem_update(q0, mem["C"])                                  # C's only update, after attempt 5
        jl(self.out / "attempts.jsonl", {"op": op, "cond": ci, "arm": "C", "event": "update_after_5", **upd})
        for k in range(6, n_adapt + 1):
            thB = S.sample(q0["mu"], q0["sigma"], lo, hi, U[k - 1])
            thC = S.sample(qC["mu"], qC["sigma"], lo, hi, U[k - 1])
            same = bool(np.allclose(thB, thC, atol=1e-9))
            rB = self.rollout(op, ci, "nominal", thB, f"BC{k}" if same else f"B{k}")
            rC = rB if same else self.rollout(op, ci, "nominal", thC, f"C{k}")
            log("B", k, thB, rB, q0)
            log("C", k, thC, rC, qC, upd if k == 6 else None)
            mem["B"].append((thB, rB["J"], k))
            mem["C"].append((thC, rC["J"], k))
        sel = {"A": (ref, r1["J"], 1), "B": S.best(mem["B"]), "C": S.best(mem["C"])}
        ev = {}
        for a in "ABC":
            th = sel[a][0]
            for sid in EVAL_STARTS:                                           # identical theta -> identical (shared) execution
                r = self.rollout(op, ci, sid, th, "eval")
                log(a, f"E-{sid}", th, r)
                ev.setdefault(a, []).append(r)
        summ = {"op": op, "cond": ci, "condition": CONDITIONS[ci],
                "selected": {a: {"theta": list(map(float, sel[a][0])), "J": sel[a][1], "attempt": sel[a][2]} for a in "ABC"},
                "first_success": {a: next((x["attempt"] for x in att[a] if x["phase"] == "adapt" and x["S"] == 1), None) for a in "ABC"},
                "steps_to_first_success": {a: self._steps_to_first(att[a]) for a in "ABC"},
                "adapt_any_success": {a: any(x["S"] == 1 for x in att[a] if x["phase"] == "adapt") for a in "ABC"},
                "eval_success": {a: [x["S"] for x in ev[a]] for a in "ABC"}, "eval_loss": {a: [x.get("D") for x in ev[a]] for a in "ABC"},
                "C_update": upd, "time": time.strftime("%H:%M:%S")}
        jl(self.out / "blocks.jsonl", summ)
        return summ

    @staticmethod
    def _steps_to_first(rows):
        tot = 0
        for x in rows:
            if x["phase"] != "adapt":
                continue
            tot += x["steps"]
            if x["S"] == 1:
                return tot
        return None                                                            # censored: no success within the budget


# ---------------------------------------------------------------------- smoke test (inside the budget, before formal blocks)
def smoke(R):
    out = {"t_start": time.strftime("%H:%M:%S")}
    t0 = time.time()
    out["starts"] = R.build_starts()
    out["starts_s"] = round(time.time() - t0, 1)
    checks = {}
    for op in OP_ORDER:
        o = R.ops[op]
        ref = G.reference(o, R.prog)
        lo, hi = G.bounds(o, ref)
        rows = {}
        for name, th, ci, sid in (("reference", ref, 0, "nominal"), ("reference_repeat", ref, 0, "nominal"), ("lower_bounds", lo, 0, "nominal"),
                                  ("upper_bounds", hi, 0, "nominal"), ("reference_heavy_slippery", ref, 5, "nominal"), ("reference_e1", ref, 0, "e1")):
            r = R.rollout(op, ci, sid, th, f"smoke-{name}")
            rows[name] = {k: r.get(k) for k in ("S", "P", "D", "J", "steps", "wall_s", "sustained_lift_m", "target_distance_m", "joint_end", "init_failure")}
        checks[op] = rows
        print("SMOKE", op, json.dumps(rows), flush=True)
    out["checks"] = checks
    out["rollout_s_mean"] = {op: round(float(np.mean(v)), 2) for op, v in R.op_rate.items()}
    R.phys.nominal()
    out["smoke_s"] = round(time.time() - t0, 1)
    (R.out / "smoke.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return out


def manifest(R, args):
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd="/home/licho/workspace/Horsea").stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "astra_explore"], capture_output=True, text=True, cwd="/home/licho/workspace/Horsea").stdout.strip()
    ops = {}
    for op in OP_ORDER:
        o = R.ops[op]
        ref = G.reference(o, R.prog)
        lo, hi = G.bounds(o, ref)
        ops[op] = dict(o, reference=ref.tolist(), lower=lo.tolist(), upper=hi.tolist(), sigma0=((hi - lo) / 4).tolist())
    m = {"experiment": "local execution adaptation beneath Astra (System 1 pilot), task-generic replication", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "task": R.spec["task"], "spec": {k: v for k, v in R.spec.items() if k != "ops"}, "code_revision": rev, "uncommitted_changes": dirty.splitlines(),
         "program": R.prog, "operations": ops, "conditions": CONDITIONS, "eval_starts": "objects in each operation's `shift` displaced together by 8 mm at 0 / 120 / 240 deg at reset, then the reference prefix",
         "score": "J = S + 0.25 clip(P, -1, 1) - D - V (generic.py: grasp = sustained 8 cm lift with the object at the hand over a 15-step hold; transport = the task's own region predicate; push = the task's joint threshold; V unavailable)",
         "search": {"attempts": args.attempts, "elites": S.ELITES, "sigma_floor": S.FLOOR, "tie_tolerance": S.TIE_TOL},
         "budget": {"runtime_min": args.budget_min, "reserve_for_analysis_min": args.reserve_min}}
    (R.out / "manifest.json").write_text(json.dumps(m, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=120)
    ap.add_argument("--reserve-min", type=float, default=14)
    ap.add_argument("--ops", default="")
    ap.add_argument("--conds", default="0-5")
    ap.add_argument("--attempts", type=int, default=9)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    a, b = (int(x) for x in args.conds.split("-"))
    spec = SPECS[args.spec]
    OP_ORDER[:] = list(spec["ops"]) if not args.ops else [o for o in spec["ops"] if o in args.ops.split(",")]
    ops = list(OP_ORDER)
    R = Runner(spec, args.out, args.budget_min * 60, args.reserve_min * 60)
    manifest(R, args)
    sm = smoke(R)
    print("SMOKE DONE", json.dumps({k: sm[k] for k in ("starts", "starts_s", "rollout_s_mean", "smoke_s")}), flush=True)
    timing = {"startup_s": round(R.t_ready - R.t_start, 1), "smoke_end_s": round(time.time() - R.t_start, 1), "blocks": []}
    if not args.smoke_only:
        for ci in range(a, b + 1):                                            # condition-major: complete matched blocks
            for op in ops:
                est = 1.15 * 21.0 * float(np.mean(R.op_rate.get(op, [12.0]))) + 9 * float(np.mean(R.op_rate.get(op, [12.0])))
                if R.left() - R.reserve_s < est:
                    timing["blocks"].append({"op": op, "cond": ci, "skipped": f"estimated {est:.0f} s > {R.left() - R.reserve_s:.0f} s left before the analysis reserve"})
                    print("SKIP", op, ci, flush=True)
                    continue
                t0 = time.time()
                s = R.block(op, ci, args.attempts)
                timing["blocks"].append({"op": op, "cond": ci, "s": round(time.time() - t0, 1), "est_s": round(est, 1)})
                print("BLOCK", op, ci, json.dumps({k: s[k] for k in ("first_success", "eval_success", "selected")})[:600], flush=True)
                (R.out / "timing.json").write_text(json.dumps(timing, indent=1))
    timing["total_s"] = round(time.time() - R.t_start, 1)
    (R.out / "timing.json").write_text(json.dumps(timing, indent=1))
    print("RUN DONE", timing["total_s"], flush=True)
    R.T.close()


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except Exception:                                             # Isaac's app loop would keep a crashed process alive
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
