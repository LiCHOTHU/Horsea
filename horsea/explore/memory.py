"""Explicit short memory and probe selectors for prior-guided exploration (plan of 2026-10-02).

Structured memory (per scene):  q_r ~ Dir(alpha_r) over prefix events G in PREFIX,
                                p_{r,m} ~ Dir(gamma_{r,m}) over tail categories K in TAIL (only after G = ready).
Scalar memory (comparison arm): one Dir over the utility values {0, 1/2, 1} per complete option.
Selectors: prior-only, greedy (posterior mean), Thompson sampling, one-step value of information (VOI).
"""
import itertools

import numpy as np

PREFIX = ("goal", "ready", "not_ready", "unknown")
TAIL = ("slip", "no_response", "away", "toward", "goal", "unknown")
U_TAIL = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.0])          # u_g(k)
U_VALUES = np.array([0.0, 0.5, 1.0])                        # scalar arm's observation alphabet


class Structured:
    def __init__(self, grasps, modes, q0=None, p0=None, kappa=2.0):
        self.grasps, self.modes = list(grasps), list(modes)
        self.options = [(r, m) for r in self.grasps for m in self.modes]
        self.alpha = {r: 0.25 + kappa * (np.asarray(q0[r]) if q0 else np.full(4, 0.25)) for r in self.grasps}
        self.gamma = {c: 1 / 6 + kappa * (np.asarray(p0[c]) if p0 else np.full(6, 1 / 6)) for c in self.options}
        if not q0:          # uniform Dirichlet with total concentration one
            self.alpha = {r: np.full(4, 0.25) for r in self.grasps}
        if not p0:
            self.gamma = {c: np.full(6, 1 / 6) for c in self.options}

    def copy(self):
        n = Structured.__new__(Structured)
        n.grasps, n.modes, n.options = self.grasps, self.modes, self.options
        n.alpha = {k: v.copy() for k, v in self.alpha.items()}
        n.gamma = {k: v.copy() for k, v in self.gamma.items()}
        return n

    def update(self, c, g, k=None):
        """g: prefix event index; k: tail category index (only when g == ready and the tail executed)."""
        r, _ = c
        self.alpha[r][g] += 1
        if g == 1 and k is not None:
            self.gamma[c][k] += 1

    def score(self, c, q=None, p=None):
        r, _ = c
        q = self.alpha[r] / self.alpha[r].sum() if q is None else q
        p = self.gamma[c] / self.gamma[c].sum() if p is None else p
        return q[0] + q[1] * float(p @ U_TAIL)

    def value(self):
        return max(self.score(c) for c in self.options)

    def branches(self, c):
        """The nine (probability, prefix event, tail category) outcomes of probing c."""
        r, _ = c
        q = self.alpha[r] / self.alpha[r].sum()
        p = self.gamma[c] / self.gamma[c].sum()
        out = [(q[g], g, None) for g in (0, 2, 3)]
        out += [(q[1] * p[k], 1, k) for k in range(6)]
        return out

    def voi(self, c):
        v0 = self.value()
        tot = 0.0
        for pr, g, k in self.branches(c):
            m = self.copy()
            m.update(c, g, k)
            tot += pr * m.value()
        return tot - v0

    def thompson(self, rng):
        qs = {r: rng.dirichlet(self.alpha[r]) for r in self.grasps}
        return max(self.options, key=lambda c: self.score(c, qs[c[0]], rng.dirichlet(self.gamma[c])))


class Scalar:
    """One Dirichlet over the utility values {0, 1/2, 1} per complete option; prior means matched to the smoothed
    structured priors' marginals."""

    def __init__(self, structured_prior, conc=3.0):
        self.options = structured_prior.options
        self.beta = {}
        for c in self.options:
            r, _ = c
            q = structured_prior.alpha[r] / structured_prior.alpha[r].sum()
            p = structured_prior.gamma[c] / structured_prior.gamma[c].sum()
            p1 = q[0] + q[1] * p[4]
            ph = q[1] * p[3]
            self.beta[c] = conc * np.array([1 - p1 - ph, ph, p1])

    def copy(self):
        n = Scalar.__new__(Scalar)
        n.options, n.beta = self.options, {k: v.copy() for k, v in self.beta.items()}
        return n

    def update(self, c, u_index):
        self.beta[c][u_index] += 1

    def score(self, c):
        b = self.beta[c]
        return float(b @ U_VALUES / b.sum())

    def value(self):
        return max(self.score(c) for c in self.options)

    def voi(self, c):
        v0, b = self.value(), self.beta[c]
        tot = 0.0
        for i in range(3):
            m = self.copy()
            m.update(c, i)
            tot += b[i] / b.sum() * m.value()
        return tot - v0


def utility_index(g, k):
    """Scalar reduction of a structured event: goal -> 1, toward -> 1/2, everything else -> 0."""
    if g == 0 or (g == 1 and k == 4):
        return 2
    if g == 1 and k == 3:
        return 1
    return 0


def choose(mem, rule, rng, tie_eps=1e-12):
    """Probe choice. prior/greedy: posterior-mean best; thompson: sampled; voi: largest one-step VOI (fallback to the
    posterior-mean best if all values are numerically zero); random: uniform."""
    opts = mem.options
    if rule in ("prior", "greedy"):
        return max(opts, key=mem.score)
    if rule == "thompson":
        return mem.thompson(rng)
    if rule == "random":
        return opts[rng.integers(len(opts))]
    if rule == "voi":
        v = {c: mem.voi(c) for c in opts}
        if max(v.values()) <= tie_eps:
            return max(opts, key=mem.score)
        return max(opts, key=lambda c: v[c])
    raise ValueError(rule)


def synthetic_voi_check():
    """Plan trace 4: VOI can prefer a less immediately rewarding option when its feedback improves the later choice,
    and is zero when only one final option exists."""
    m = Structured([0, 1], ["a"])
    m.alpha = {0: np.array([0.01, 20.0, 0.01, 0.01]), 1: np.array([0.01, 20.0, 0.01, 0.01])}
    m.gamma = {(0, "a"): np.array([0.01, 0.01, 0.01, 0.01, 60.0, 0.01]) + np.array([0, 40.0, 0, 0, 0, 0]),   # known: 0.6
               (1, "a"): np.array([0.01, 0.6, 0.01, 0.01, 0.5, 0.01])}                                     # uncertain, mean ~0.45
    greedy = max(m.options, key=m.score)
    voi = {c: m.voi(c) for c in m.options}
    one = Structured([0], ["a"])
    return greedy, voi, one.voi((0, "a"))


if __name__ == "__main__":
    g, v, v1 = synthetic_voi_check()
    print("greedy picks", g, "| VOI", {k: round(x, 4) for k, x in v.items()}, "| single-option VOI", v1)
