"""System 1 for the cup-transfer experiment (numpy only; no learned encoder, no neural policy).

ExperienceStore        raw transition records (the brief's fields), queryable; requested and applied commands kept apart.
LocalExecutionModel    locally weighted ridge regression refitted at every query: Δŝ = Θ [ (s-s_q)/σ_s, (u-u_q)/σ_u, 1 ],
                       weights w_i = exp(-d_i²/2h²) on a FIXED scaled metric, compatible records only (same declared
                       interaction mode and same near/far relation to the object; never hidden simulator labels).
ShortHorizonController 32 bounded candidate 3-segment sequences around Astra's nominal sequence; predict with the local model
                       only; cost = goal loss + α‖δ‖² + β Σ‖δ_k - δ_k-1‖² (δ in units of the bounds); keep candidates whose
                       first segment has local support (effective sample size >= ESS_MIN); execute the first segment; fall
                       back to the nominal command when nothing is supported.
NearestNeighborReuse   retrieve the compatible record whose state is closest and whose observed effect best fits the local
                       goal; reuse its object-relative action inside the same bounds.

State s (27, object-relative; geometric distances normalised by the observable radius R and height H):
  w = (wrist - object base) / [R, R, H] (3), t = (fingertip mean - object base) / [R, R, H] (3), fingers_dir (3), palm_dir (3),
  measured closure (4), object rise above rest x10 [m] (1), tilt = 1 - up_z (1), object velocity x10 [m/s] (3),
  index contact / 10 [N] (1), wrist velocity x10 [m/s] (3), [2R, H] x10 [m] (2).
Change Δs (15): Δw (3), Δt (3), Δclosure (4), Δrise x10 (1), object xy displacement / R (2), Δtilt (1), Δcontact / 10 (1).
Action u (7): wrist translation over the segment [cm] (3, physical magnitude), closure targets (4) - the APPLIED values.
"""
import copy
import json
import math

import numpy as np

MODES = ("approach", "close", "lift", "hold", "transport", "release", "reposition", "other")
STATE_NAMES = (["w_x", "w_y", "w_z", "t_x", "t_y", "t_z", "fd_x", "fd_y", "fd_z", "pd_x", "pd_y", "pd_z",
                "cl_fb", "cl_ft", "cl_tr", "cl_tb", "rise10", "tilt", "ov_x10", "ov_y10", "ov_z10", "contact10",
                "wv_x10", "wv_y10", "wv_z10", "dia10", "h10"])
DELTA_NAMES = ["dw_x", "dw_y", "dw_z", "dt_x", "dt_y", "dt_z", "dcl_fb", "dcl_ft", "dcl_tr", "dcl_tb", "drise10", "dobj_x", "dobj_y", "dtilt", "dcontact10"]
ACT_NAMES = ["dp_x_cm", "dp_y_cm", "dp_z_cm", "c_fb", "c_ft", "c_tr", "c_tb"]
W, T, FD, PD, CL, RISE, TILT, OV, CON, WV, DIM = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 16), 16, 17,
                                                  slice(18, 21), 21, slice(22, 25), slice(25, 27))
SEG_STEPS, HZ = 3, 30.0
# fixed metric (scales chosen from the physical meaning of each feature, never tuned on target tests)
SIG_S = np.array([0.5] * 6 + [0.3] * 6 + [0.15] * 4 + [0.2, 0.1] + [0.5] * 3 + [0.3] + [0.5] * 3 + [0.3, 0.3])
SIG_U = np.array([1.0, 1.0, 1.0, 0.15, 0.15, 0.15, 0.15])
BOUND = np.array([1.5, 1.5, 1.5, 0.15, 0.15, 0.15, 0.15])      # candidate deviations from Astra's command (cm, closure)
SIG_D = np.array([0.3] * 6 + [0.1] * 4 + [0.1, 0.1, 0.1, 0.05, 0.2])  # fixed scales of the change features (effect matching)
ESS_MIN = 8.0


# ---------------------------------------------------------------------------- state
def state_vector(r):
    """r: raw state dict (metric, world frame) -> s (27)."""
    o, R, H = np.asarray(r["obj_pos"], float), float(r["R"]), float(r["H"])
    sc = np.array([R, R, H])
    return np.concatenate([(np.asarray(r["wrist"]) - o) / sc, (np.asarray(r["tips"]) - o) / sc, r["fingers_dir"], r["palm_dir"],
                           r["closure"], [10 * (o[2] - r["rest_z"]), 1 - float(r["obj_up"][2])], 10 * np.asarray(r["obj_vel"]),
                           [float(r["contact"]) / 10], 10 * np.asarray(r["wrist_vel"]), [20 * R, 10 * H]]).astype(float)


def delta_vector(r0, r1):
    s0, s1 = state_vector(r0), state_vector(r1)
    R = float(r0["R"])
    dobj = (np.asarray(r1["obj_pos"])[:2] - np.asarray(r0["obj_pos"])[:2]) / R
    return np.concatenate([s1[W] - s0[W], s1[T] - s0[T], s1[CL] - s0[CL], [s1[RISE] - s0[RISE]], dobj,
                           [s1[TILT] - s0[TILT]], [s1[CON] - s0[CON]]])


def near_object(s):
    """Interaction context from observed geometry: fingertips within ~1.5 radii (horizontal) and within the object's height band."""
    return bool(math.hypot(s[3], s[4]) < 2.5 and -0.3 < s[5] < 1.4)


def apply_delta(s, ds, R, H):
    """Propagate a predicted change for multi-segment planning (velocities from the predicted displacements)."""
    n = s.copy()
    n[W] += ds[0:3]
    n[T] += ds[3:6]
    n[CL] = np.clip(n[CL] + ds[6:10], -0.2, 1.2)
    n[RISE] += ds[10]
    n[TILT] = max(0.0, n[TILT] + ds[13])
    n[CON] = max(0.0, n[CON] + ds[14])
    dt = SEG_STEPS / HZ
    obj = np.array([ds[11] * R, ds[12] * R, ds[10] / 10])
    n[OV] = 10 * obj / dt
    n[WV] = 10 * (ds[0:3] * np.array([R, R, H]) + obj) / dt
    return n


# ---------------------------------------------------------------------------- memory
class ExperienceStore:
    FIELDS = ("timestamp", "episode_id", "object_instance_id", "interaction_context", "observed_state_before", "requested_command",
              "applied_command_after_limits", "observed_state_after", "duration", "observation_validity")

    def __init__(self, records=None, frozen=False):
        self.records = list(records or [])
        self.frozen = frozen
        self._cache = None

    def __len__(self):
        return len(self.records)

    def add(self, rec):
        if self.frozen:
            return False
        if not rec.get("observation_validity", True):
            return False
        self.records.append(rec)
        self._cache = None
        return True

    def requested(self):
        """Requested commands of the records (what was decided), aligned with arrays()."""
        return np.array([r["requested_command"] for r in self.records]) if self.records else np.zeros((0, 7))

    def clone(self, frozen=None):
        return ExperienceStore(copy.deepcopy(self.records), self.frozen if frozen is None else frozen)

    def arrays(self):
        if self._cache is None:
            if not self.records:
                self._cache = (np.zeros((0, 27)), np.zeros((0, 7)), np.zeros((0, 15)), np.zeros(0, int), np.zeros(0, bool), [])
            else:
                S = np.array([r["s"] for r in self.records])
                U = np.array([r["u_applied"] for r in self.records])
                D = np.array([r["ds"] for r in self.records])
                M = np.array([MODES.index(r["interaction_context"]["mode"]) for r in self.records])
                N = np.array([r["interaction_context"]["near"] for r in self.records])
                self._cache = (S, U, D, M, N, [r["object_instance_id"] for r in self.records])
        return self._cache

    def dump(self, path):
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    @staticmethod
    def load(path, frozen=False):
        with open(path) as f:
            return ExperienceStore([json.loads(x) for x in f if x.strip()], frozen)


# ---------------------------------------------------------------------------- local model
class LocalExecutionModel:
    def __init__(self, store, h=1.5, lam=0.1, k=80):
        self.store, self.h, self.lam, self.k = store, h, lam, k

    def compatible(self, ctx):
        S, U, D, M, N, ids = self.store.arrays()
        if not len(S):
            return np.zeros(0, int)
        return np.where((M == MODES.index(ctx["mode"])) & (N == ctx["near"]))[0]

    def weights(self, s, u, ctx):
        S, U, D, M, N, ids = self.store.arrays()
        idx = self.compatible(ctx)
        if not len(idx):
            return idx, np.zeros(0), np.zeros(0)
        d = np.sqrt((((S[idx] - s) / SIG_S) ** 2).sum(1) + (((U[idx] - u) / SIG_U) ** 2).sum(1))
        order = np.argsort(d)[:self.k]
        idx, d = idx[order], d[order]
        return idx, d, np.exp(-d ** 2 / (2 * self.h ** 2))

    @staticmethod
    def ess(w):
        return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)) if len(w) else 0.0

    def support(self, s, u, ctx):
        return self.ess(self.weights(s, u, ctx)[2])

    def fit(self, s, u, ctx):
        idx, d, w = self.weights(s, u, ctx)
        e = self.ess(w)
        if e < ESS_MIN:
            return None
        S, U, D, M, N, ids = self.store.arrays()
        X = np.hstack([(S[idx] - s) / SIG_S, (U[idx] - u) / SIG_U, np.ones((len(idx), 1))])
        Wd = w[:, None]
        reg = self.lam * np.eye(X.shape[1])
        reg[-1, -1] = 1e-6                                        # the intercept is not shrunk
        theta = np.linalg.solve(X.T @ (Wd * X) + reg, X.T @ (Wd * D[idx]))
        return {"theta": theta, "s_q": s.copy(), "u_q": u.copy(), "idx": idx, "d": d, "w": w, "ess": e,
                "sources": [ids[i] for i in idx[:20]]}

    @staticmethod
    def predict(fit, s, u):
        x = np.concatenate([(s - fit["s_q"]) / SIG_S, (u - fit["u_q"]) / SIG_U, [1.0]])
        return x @ fit["theta"]


# ---------------------------------------------------------------------------- goals
def goal_loss(mode, traj, goal):
    """traj: list of (s_k, cumulative object xy displacement / R, cumulative commanded wrist z [m]); s_0 = traj[0][0]."""
    s0 = traj[0][0]
    sK, dK, zK = traj[-1]
    if mode in ("approach", "reposition"):
        return float(np.mean([np.sum((s[W] - goal["w_target"]) ** 2) for s, _, _ in traj[1:]]) + 10 * np.sum(dK ** 2) + 10 * sK[TILT] ** 2)
    if mode == "close":
        r = math.hypot(sK[3], sK[4])
        return float(10 * np.sum(dK ** 2) + 10 * sK[TILT] ** 2 + 5 * max(0.0, r - 1.3) ** 2)
    if mode == "lift":
        rise = (sK[RISE] - s0[RISE]) / 10
        return float(100 * (rise - zK) ** 2 + 10 * sK[TILT] ** 2 + 2 * np.sum((sK[T] - s0[T]) ** 2))
    if mode == "hold":
        return float(100 * max(0.0, goal.get("lift_m", 0.05) - sK[RISE] / 10) ** 2 + np.sum(sK[OV] ** 2) + 10 * sK[TILT] ** 2)
    if mode == "transport":
        return float(2 * np.sum((sK[T] - s0[T]) ** 2) + 10 * sK[TILT] ** 2 + 100 * max(0.0, (s0[RISE] - sK[RISE]) / 10) ** 2)
    if mode == "release":
        return float(10 * np.sum(dK ** 2) + 10 * sK[TILT] ** 2)
    return 0.0


def rollout(fit, s, U_app, R, H):
    traj, d, z = [(s, np.zeros(2), 0.0)], np.zeros(2), 0.0
    cur = s
    for u in U_app:
        ds = LocalExecutionModel.predict(fit, cur, u)
        cur = apply_delta(cur, ds, R, H)
        d = d + ds[11:13]
        z = z + u[2] / 100
        traj.append((cur, d.copy(), z))
    return traj


# ---------------------------------------------------------------------------- controllers
class ShortHorizonController:
    name = "local_model"

    def __init__(self, store, n_cand=32, horizon=3, alpha=0.5, beta=0.2, h=1.5, lam=0.1, seed=0):
        self.model = LocalExecutionModel(store, h=h, lam=lam)
        self.n, self.K, self.alpha, self.beta = n_cand, horizon, alpha, beta
        self.rng = np.random.default_rng(seed)

    def candidates(self):
        c = np.zeros((self.n, self.K, 7))
        for i in range(1, self.n):
            first = self.rng.uniform(-1, 1, 7)
            c[i, 0] = first
            for k in range(1, self.K):
                c[i, k] = np.clip(c[i, k - 1] + self.rng.normal(0, 0.25, 7), -1, 1)
        return c

    def plan(self, s, ctx, U_nom, limit, goal, R, H):
        """U_nom: (K, 7) REQUESTED nominal sequence; limit(U_req) -> (K, 7) applied sequence (the executor's own limits)."""
        info = {"method": self.name, "mode": ctx["mode"], "near": ctx["near"]}
        if ctx["mode"] == "other":
            return dict(info, u_req=U_nom[0], fallback="mode other: no adjustment")
        U_nom_app = limit(U_nom)
        fit = self.model.fit(s, U_nom_app[0], ctx)
        if fit is None:
            return dict(info, u_req=U_nom[0], fallback="no local support", ess=self.model.support(s, U_nom_app[0], ctx))
        best, rows = None, []
        for i, dlt in enumerate(self.candidates()):
            U_req = U_nom + dlt * BOUND
            U_req[:, 3:] = np.clip(U_req[:, 3:], 0, 1)
            U_app = limit(U_req)
            e = self.model.support(s, U_app[0], ctx) if i else fit["ess"]
            if e < ESS_MIN:
                continue
            traj = rollout(fit, s, U_app, R, H)
            cost = goal_loss(ctx["mode"], traj, goal) + self.alpha * float(np.sum(dlt ** 2)) / self.K \
                + self.beta * float(np.sum(np.diff(dlt, axis=0) ** 2))
            rows.append((cost, i))
            if best is None or cost < best[0]:
                best = (cost, i, U_req, traj)
        if best is None:
            return dict(info, u_req=U_nom[0], fallback="no supported candidate", ess=fit["ess"])
        cost, i, U_req, traj = best
        nominal_cost = next((c for c, j in rows if j == 0), None)
        pred = LocalExecutionModel.predict(fit, s, limit(U_req)[0])
        return dict(info, u_req=U_req[0], fallback=None, candidate=i, cost=round(cost, 5),
                    nominal_cost=None if nominal_cost is None else round(nominal_cost, 5), n_supported=len(rows),
                    ess=round(fit["ess"], 2), n_neighbors=int(len(fit["idx"])), mean_dist=round(float(np.mean(fit["d"][:20])), 3),
                    neighbor_objects=fit["sources"], predicted_ds=[round(float(x), 5) for x in pred])


class NearestNeighborReuse:
    """Replay baseline: among compatible records (same declared mode, same near/far relation) close in state, pick the one
    whose OBSERVED effect best matches the desired local effect of Astra's nominal segment, and reuse that record's REQUESTED
    command (object-relative translation, closure targets) inside the same bounds as the local-model controller.
    Desired effect (kinematic, no model): approach / reposition / close / hold / release = the wrist moves by the nominal
    translation relative to a still object and the closures move toward the command; lift / transport = the object moves
    with the hand (relative pose unchanged, object displaced by the nominal translation)."""
    name = "nearest_neighbor"

    def __init__(self, store, k=40, d_max=4.0):
        self.store, self.k, self.d_max = store, k, d_max

    def desired_effect(self, s, mode, u_app, R, H):
        dp = u_app[:3] / 100.0
        de = np.zeros(15)
        if mode in ("lift", "transport"):
            de[10] = 10 * dp[2]
            de[11:13] = dp[:2] / R
        else:
            de[0:3] = dp / np.array([R, R, H])
            de[3:6] = dp / np.array([R, R, H])
        de[6:10] = u_app[3:] - s[CL]
        return de

    def plan(self, s, ctx, U_nom, limit, goal, R, H):
        info = {"method": self.name, "mode": ctx["mode"], "near": ctx["near"]}
        if ctx["mode"] == "other":
            return dict(info, u_req=U_nom[0], fallback="mode other: no adjustment")
        S, U, D, M, N, ids = self.store.arrays()
        idx = np.where((M == MODES.index(ctx["mode"])) & (N == ctx["near"]))[0] if len(S) else np.zeros(0, int)
        if not len(idx):
            return dict(info, u_req=U_nom[0], fallback="no compatible record")
        d_s = np.sqrt((((S[idx] - s) / SIG_S) ** 2).sum(1))
        order = np.argsort(d_s)[:self.k]
        idx, d_s = idx[order], d_s[order]
        keep = d_s <= self.d_max
        if not keep.any():
            return dict(info, u_req=U_nom[0], fallback="no record within the state radius", nearest=round(float(d_s[0]), 3))
        idx, d_s = idx[keep], d_s[keep]
        de = self.desired_effect(s, ctx["mode"], limit(U_nom)[0], R, H)
        d_e = np.sqrt((((D[idx] - de) / SIG_D) ** 2).sum(1))
        j = int(np.argmin(d_s + d_e))
        rec = idx[j]
        u = self.store.requested()[rec].copy()          # the record's decided command: object-relative translation [cm], closure targets
        dlt = np.clip((u - U_nom[0]) / BOUND, -1, 1)     # the same bounded set as the local-model controller
        u_req = U_nom[0] + dlt * BOUND
        u_req[3:] = np.clip(u_req[3:], 0, 1)
        return dict(info, u_req=u_req, fallback=None, record_index=int(rec), record_object=ids[rec], state_dist=round(float(d_s[j]), 3),
                    effect_dist=round(float(d_e[j]), 3), n_candidates=int(len(idx)))


class Nominal:
    name = "nominal"

    def plan(self, s, ctx, U_nom, limit, goal, R, H):
        return {"method": self.name, "mode": ctx["mode"], "near": ctx["near"], "u_req": U_nom[0], "fallback": "nominal executor"}


# ---------------------------------------------------------------------------- development-data model selection
def select_hyper(store, hs=(0.75, 1.5, 3.0), lams=(0.01, 0.1, 1.0)):
    """Leave-one-episode-out one-segment prediction error on the SOURCE data only; returns the table and the best (h, lam)."""
    S, U, D, M, N, ids = store.arrays()
    eps = np.array([r["episode_id"] for r in store.records])
    scale = D.std(0) + 1e-6
    table = []
    for h in hs:
        for lam in lams:
            errs, n_sup, n = [], 0, 0
            for e in np.unique(eps):
                train = ExperienceStore([r for r, ep in zip(store.records, eps) if ep != e])
                m = LocalExecutionModel(train, h=h, lam=lam)
                for i in np.where(eps == e)[0]:
                    n += 1
                    ctx = store.records[i]["interaction_context"]
                    fit = m.fit(S[i], U[i], ctx)
                    if fit is None:
                        continue
                    n_sup += 1
                    errs.append(np.mean(((LocalExecutionModel.predict(fit, S[i], U[i]) - D[i]) / scale) ** 2))
            table.append({"h": h, "lam": lam, "nmse": float(np.mean(errs)) if errs else None, "supported": n_sup, "queries": n})
    zero = float(np.mean((D / scale) ** 2)) if len(D) else None
    ok = [t for t in table if t["nmse"] is not None and t["supported"] >= 0.3 * max(1, t["queries"])]
    best = min(ok, key=lambda t: t["nmse"]) if ok else {"h": 1.5, "lam": 0.1}
    return {"table": table, "zero_change_nmse": zero, "best": {"h": best["h"], "lam": best["lam"]}}
