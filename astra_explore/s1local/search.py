"""Arms A / B / C and the CEM-style adaptive search distribution (not Bayesian inference).

Proposal q(theta) = bounded diagonal Gaussian N(mu, diag sigma^2) inside the box Theta = [lo, hi].
Bounded sampling: per dimension an inverse-CDF draw from the Gaussian truncated to [lo, hi], driven by a uniform u in (0, 1).
Common random numbers: one uniform matrix U (9 x d) per (operation, condition), seeded; attempt k of every arm uses U[k], so B
and C share attempts 2-5 exactly and their attempts 6-9 differ only through their own (mu, sigma).

Schedule (per physical condition, memory cleared between conditions):
    attempt 1     theta_Astra for every arm
    attempts 2-5  B and C: the same four draws from q0 = N(theta_Astra, (width / 4)^2)
    after 5       C only: elites = the two highest-J attempts among 1-5 (ties -> earlier attempt);
                  mu <- 0.5 mu + 0.5 mean(elites); sigma <- 0.5 sigma + 0.5 std(elites), floored at 0.25 sigma0;
                  if all five scores are equal within 1e-3, C keeps q0 (no invented direction)
    attempts 6-9  B from q0, C from its updated proposal
    selection     A: theta_Astra; B, C: the best actually evaluated candidate (highest J, ties -> earlier attempt)
    evaluation    the selected theta on three fresh starts, memory frozen
"""
import hashlib
from statistics import NormalDist

import numpy as np

_N = NormalDist()
ELITES, TIE_TOL, FLOOR = 2, 1e-3, 0.25


def seed_of(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def uniforms(op, cond, d, n=9):
    return np.random.default_rng(seed_of("s1local", op, cond)).random((n, d))


def sample(mu, sigma, lo, hi, u):
    out = []
    for m, s, a, b, x in zip(mu, sigma, lo, hi, u):
        ca, cb = _N.cdf((a - m) / s), _N.cdf((b - m) / s)
        p = min(max(ca + x * (cb - ca), 1e-12), 1 - 1e-12)
        out.append(float(np.clip(m + s * _N.inv_cdf(p), a, b)))
    return np.array(out)


def initial(ref, lo, hi):
    sigma0 = (hi - lo) / 4.0
    return {"mu": ref.astype(float).copy(), "sigma": sigma0.copy(), "sigma0": sigma0.copy()}


def cem_update(q, memory):
    """memory: list of (theta, J, attempt). Returns (new proposal, update record)."""
    Js = np.array([m[1] for m in memory])
    if Js.max() - Js.min() <= TIE_TOL:
        return dict(q), {"updated": False, "reason": f"all {len(Js)} scores within {TIE_TOL}: proposal retained"}
    order = sorted(range(len(memory)), key=lambda i: (-memory[i][1], memory[i][2]))
    el = np.array([memory[i][0] for i in order[:ELITES]])
    mu = 0.5 * q["mu"] + 0.5 * el.mean(0)
    sigma = np.maximum(0.5 * q["sigma"] + 0.5 * el.std(0), FLOOR * q["sigma0"])
    new = {"mu": mu, "sigma": sigma, "sigma0": q["sigma0"]}
    return new, {"updated": True, "elite_attempts": [memory[i][2] for i in order[:ELITES]], "elite_J": [memory[i][1] for i in order[:ELITES]],
                 "mu_old": q["mu"].round(4).tolist(), "mu_new": mu.round(4).tolist(), "sigma_old": q["sigma"].round(4).tolist(),
                 "sigma_new": sigma.round(4).tolist()}


def best(memory):
    """Best actually evaluated candidate: highest J, ties -> earliest attempt."""
    i = sorted(range(len(memory)), key=lambda k: (-memory[k][1], memory[k][2]))[0]
    return memory[i]
