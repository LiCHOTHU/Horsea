"""Local-execution-adaptation pilot: smoke test + matched A/B/C blocks, one Isaac process, rendering off.

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
from s1local import ops as O  # noqa: E402
from s1local import search as S  # noqa: E402

# predetermined, balanced condition order (mass factor, friction factor): nominal first, extremes interleaved
CONDITIONS = [(1.0, 1.0), (0.8, 0.7), (1.2, 1.0), (1.0, 0.7), (0.8, 1.0), (1.2, 0.7)]
OP_ORDER = ["grasp", "transport", "door"]
CAN_OFFSETS = {"nominal": (0.0, 0.0), "e1": (0.008, 0.0), "e2": (-0.004, 0.0069), "e3": (-0.004, -0.0069)}   # 8 mm at 0/120/240 deg
DOOR_OFFSETS = {"nominal": 0.0, "e1": 0.03, "e2": 0.06, "e3": 0.09}     # rad: door slightly less open than its reset state
EVAL_STARTS = ["e1", "e2", "e3"]


def jl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


class Runner:
    def __init__(self, out, budget_s, reserve_s):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self.budget_s, self.reserve_s = budget_s, reserve_s
        self.T = E.build(render=False)
        self.t_ready = time.time()
        self.phys = E.Physics(self.T)
        self.prog = E.load_program(10)
        self.recording, self.trace = False, []
        step0 = self.T._step

        def _step(a):
            o = step0(a)
            if self.recording:
                U = self.T.U
                self.trace.append([U.can.data.root_state_w[0, :3].cpu().numpy().copy(), float(U.drawer.data.joint_pos[0, U.drawer_bottom_idx].item()),
                                   self.T._tips("right").mean(0)])
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

    def left(self):
        return self.budget_s - (time.time() - self.t_start)

    # ------------------------------------------------------------------ starts
    def build_starts(self):
        rec = {}
        for op in OP_ORDER:
            for sid in ["nominal"] + EVAL_STARTS:
                if op == "door" and sid != "nominal":
                    continue
                self.phys.nominal()
                E.reset_with_can_offset(self.T, CAN_OFFSETS[sid])
                for i in O.OPS[op]["prefix"]:
                    E.act(self.T, self.prog[i])
                snap = E.snapshot(self.T)
                st = E.object_state(self.T)
                ok, info = O.start_valid(op, st)
                self.starts[(op, sid)] = {"snap": snap, "door_offset": 0.0, "valid": ok, "info": info, "can_offset_m": CAN_OFFSETS[sid]}
                rec[f"{op}/{sid}"] = {"valid": ok, **info, "can_offset_m": CAN_OFFSETS[sid]}
            if op == "door":
                for sid in EVAL_STARTS:                      # same arrangement, door slightly less open (validated at restore)
                    base = self.starts[("door", "nominal")]
                    self.starts[("door", sid)] = dict(base, door_offset=DOOR_OFFSETS[sid])
                    self._restore("door", sid)
                    E.hold_window(self.T, 2)
                    st = E.object_state(self.T)
                    ok, info = O.start_valid("door", st)
                    ok = ok and abs(st["door_q"] - DOOR_OFFSETS[sid]) < 0.02
                    self.starts[("door", sid)]["valid"] = ok
                    rec[f"door/{sid}"] = {"valid": ok, **info, "door_offset_rad": DOOR_OFFSETS[sid]}
        return rec

    def _restore(self, op, sid):
        st = self.starts[(op, sid)]
        E.restore(self.T, st["snap"])
        if st.get("door_offset"):
            E.set_door(self.T, st["door_offset"])

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
        t0 = time.time()
        self._restore(op, sid)
        phys = self.phys.apply(op, *CONDITIONS[ci])
        start = E.object_state(self.T)
        edited = O.edited_program(op, self.prog, theta, left_wrist0=self.T._ee("left")[:3])
        self.trace, self.recording = [], True
        steps = 0
        for d, sp in edited:
            steps += E.act(self.T, d, speed=sp)
        steps += E.hold_window(self.T, O.OPS[op]["settle"])
        self.recording = False
        final = E.object_state(self.T)
        ev = O.evaluate(op, start, self.trace, final)
        tr = [[round(float(x), 4) for x in list(t[0]) + [t[1]]] for t in self.trace[::2]]
        r = {"key": key, "op": op, "cond": ci, "condition": CONDITIONS[ci], "start": sid, "slot": slot, "theta": theta.round(6).tolist(),
             "init_failure": False, "steps": steps, "wall_s": round(time.time() - t0, 2), **ev,
             "final": {k: final[k] for k in ("can_pos", "door_q", "env_success")}, "physics_applied": phys, "trajectory_every2": tr}
        jl(self.out / "rollouts.jsonl", r)
        self.done[key] = r
        self.op_rate.setdefault(op, []).append(r["wall_s"])
        return r

    # ------------------------------------------------------------------ one matched block
    def block(self, op, ci, n_adapt=9):
        ref = O.reference(op, self.prog)
        lo, hi = O.bounds(op, ref)
        q0 = S.initial(ref, lo, hi)
        U = S.uniforms(op, ci, len(ref))
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
        ref = O.reference(op, R.prog)
        lo, hi = O.bounds(op, ref)
        rows = {}
        for name, th, ci, sid in (("reference", ref, 0, "nominal"), ("reference_repeat", ref, 0, "nominal"), ("lower_bounds", lo, 0, "nominal"),
                                  ("upper_bounds", hi, 0, "nominal"), ("reference_heavy_slippery", ref, 5, "nominal"), ("reference_e1", ref, 0, "e1")):
            r = R.rollout(op, ci, sid, th, f"smoke-{name}")
            rows[name] = {k: r.get(k) for k in ("S", "P", "D", "J", "steps", "wall_s", "sustained_lift_m", "can_plate_xy_m", "door_q_end", "init_failure")}
        checks[op] = rows
        print("SMOKE", op, json.dumps(rows), flush=True)
    out["checks"] = checks
    out["rollout_s_mean"] = {op: round(float(np.mean(v)), 2) for op, v in R.op_rate.items()}
    out["readback_condition5"] = (R.phys.apply("grasp", *CONDITIONS[5]), R.phys.readback())
    R.phys.nominal()
    out["smoke_s"] = round(time.time() - t0, 1)
    (R.out / "smoke.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return out


def manifest(R, args):
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd="/home/licho/workspace/Horsea").stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "astra_explore"], capture_output=True, text=True, cwd="/home/licho/workspace/Horsea").stdout.strip()
    ops = {}
    for op in OP_ORDER:
        ref = O.reference(op, R.prog)
        lo, hi = O.bounds(op, ref)
        ops[op] = {"prefix_decisions": O.OPS[op]["prefix"], "program_decisions": O.OPS[op]["program"], "evaluation_window_steps": O.OPS[op]["settle"],
                   "params": O.OPS[op]["params"], "reference": ref.tolist(), "lower": lo.tolist(), "upper": hi.tolist(),
                   "sigma0": ((hi - lo) / 4).tolist(), "uniform_seed": S.seed_of("s1local", op, "c*")}
    m = {"experiment": "local execution adaptation beneath Astra (System 1 pilot)", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "environment": "Humanoid-Stack-Can-Into-Drawer-v0, EgoVLA Ego-Humanoid benchmark, Isaac Sim 5.1 / IsaacLab 2.3, room 1 table 2, randomize_idx 5788",
         "robot": "Unitree H1 + two Inspire hands (12 joints each); harness astra_explore/isaac_env.py (differential IK wrist targets, 4 closure fractions per hand)",
         "physics": "dt 1/240, decimation 8 (30 Hz control), enhanced determinism on; rendering off (render interval and camera update period unreachable)",
         "code_revision": rev, "uncommitted_changes": dirty.splitlines(), "source_program": "pilot attempt 10 (11 decisions, success at step 298)",
         "program": R.prog, "operations": ops, "conditions": CONDITIONS, "condition_order": "predetermined, balanced; blocks run condition-major, operations inner",
         "starts": {"adaptation": "nominal start (reference prefix from the reset scene)",
                    "evaluation": {"grasp/transport": {k: v for k, v in CAN_OFFSETS.items() if k != "nominal"},
                                   "door": {k: v for k, v in DOOR_OFFSETS.items() if k != "nominal"}},
                    "note": "grasp / transport fresh starts: can displaced 8 mm on its support at reset, then the reference prefix (transport: a physical grasp, never teleported); door: the nominal arrangement with the door 0.03-0.09 rad less open"},
         "score": "J = S + 0.25 clip(P, -1, 1) - D - V; grasp: S = can >= 8 cm above its start AND within 0.12 m of the right fingertips at every step of a 15-step (0.5 s) hold; transport: S = env can_on_plate (xy < 0.09 m, |dz| < 0.02 m) after a 10-step settle; door: S = env closed criterion (>= upper - 0.2 range) after a 10-step settle; P: grasp sustained lift / 0.08, transport 1 - d_end / d_start, door angle progress / required; D: grasp can fell off its support or flung > 25 cm, transport can fell below the saucer by > 10 cm, door placed can knocked off the saucer; V: no existing constraint signal in this environment (always 0)",
         "search": {"arms": "A original, B fixed q0, C CEM-style update after attempt 5", "attempts": args.attempts, "eval_starts": EVAL_STARTS,
                    "elites": S.ELITES, "mean_blend": 0.5, "sigma_blend": 0.5, "sigma_floor": S.FLOOR, "tie_tolerance": S.TIE_TOL,
                    "bounded_sampling": "per-dimension inverse-CDF truncated Gaussian, common uniforms per (operation, condition)"},
         "budget": {"runtime_min": args.budget_min, "reserve_for_analysis_min": args.reserve_min}, "feedback": "state-based evaluator (richer than the original visual / sparse success); never given mass, friction, future outcomes or optimal parameters"}
    (R.out / "manifest.json").write_text(json.dumps(m, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=120)
    ap.add_argument("--reserve-min", type=float, default=14)
    ap.add_argument("--ops", default=",".join(OP_ORDER))
    ap.add_argument("--conds", default="0-5")
    ap.add_argument("--attempts", type=int, default=9)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    a, b = (int(x) for x in args.conds.split("-"))
    ops = [o for o in OP_ORDER if o in args.ops.split(",")]
    R = Runner(args.out, args.budget_min * 60, args.reserve_min * 60)
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
    main()
