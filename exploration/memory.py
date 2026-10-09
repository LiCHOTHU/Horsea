"""Conjugate models of stationary *observed events*, not latent mechanics."""
import numpy as np

from .events import Event, Option, Prefix, Tail


TAIL_UTILITY = np.array([0., 0., 0., .5, 1., 0.])


def positive_array(value, shape, name):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError(f"{name}: expected positive finite array {shape}, got {a.shape}")
    return a


def means(counts):
    return counts / counts.sum(axis=-1, keepdims=True)


class StructuredMemory:
    def __init__(self, grasps=2, modes=3, alpha=None, gamma=None, success_only=False):
        if not 1 <= grasps <= 2 or not 1 <= modes <= 3:
            raise ValueError("Pilot supports one/two grasps and one to three modes")
        self.grasps, self.modes = grasps, modes
        self.alpha = positive_array(np.full((grasps, 4), .25) if alpha is None else alpha,
                                    (grasps, 4), "alpha")
        self.gamma = positive_array(np.full((grasps, modes, 6), 1/6) if gamma is None else gamma,
                                    (grasps, modes, 6), "gamma")
        self.success_only = success_only
        self.utility = TAIL_UTILITY.copy()
        if success_only:
            self.utility[Tail.TOWARD] = 0.

    @property
    def options(self):
        return [Option(r, m) for r in range(self.grasps) for m in range(self.modes)]

    def copy(self):
        return StructuredMemory(self.grasps, self.modes, self.alpha, self.gamma, self.success_only)

    def _validate_option(self, c):
        if c not in self.options:
            raise ValueError(f"Option outside frozen repertoire: {c}")

    def update(self, event):
        if not isinstance(event, Event):
            raise TypeError("Memory accepts only an observation Event")
        self._validate_option(event.option)
        r, m = event.option.grasp, event.option.mode
        self.alpha[r, event.prefix] += 1
        if event.prefix == Prefix.READY:
            self.gamma[r, m, event.tail] += 1

    def scores(self, rng=None):
        if rng is None:
            q, p = means(self.alpha), means(self.gamma)
        else:
            # One shared prefix draw per grasp, not a different draw per mode.
            q = np.stack([rng.dirichlet(a) for a in self.alpha])
            p = np.stack([rng.dirichlet(g) for g in self.gamma.reshape(-1, 6)])
            p = p.reshape(self.grasps, self.modes, 6)
        return (q[:, Prefix.GOAL, None] + q[:, Prefix.READY, None] * (p @ self.utility)).ravel()

    def branches(self, c):
        self._validate_option(c)
        r, m = c.grasp, c.mode
        q, p = means(self.alpha)[r], means(self.gamma)[r, m]
        result = [(float(q[j]), Event(c, j)) for j in (Prefix.GOAL, Prefix.NOT_READY, Prefix.UNKNOWN)]
        result += [(float(q[Prefix.READY] * p[k]), Event(c, Prefix.READY, k)) for k in Tail]
        return result

    def snapshot(self):
        return {"kind": "structured", "alpha": self.alpha.tolist(), "gamma": self.gamma.tolist(),
                "success_only": self.success_only}

    @classmethod
    def fit_prior(cls, events, grasps=2, modes=3, kappa=2., success_only=False):
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("Prior strength must be finite and nonnegative")
        q, p = np.zeros((grasps, 4)), np.zeros((grasps, modes, 6))
        base = cls(grasps, modes, success_only=success_only)
        for e in events:
            base._validate_option(e.option)
            q[e.option.grasp, e.prefix] += 1
            if e.tail is not None:
                p[e.option.grasp, e.option.mode, e.tail] += 1
        # Unobserved rows retain concentration one, not a fabricated empirical prior.
        aq, gp = q.sum(-1, keepdims=True), p.sum(-1, keepdims=True)
        base.alpha += kappa * q / np.maximum(aq, 1)
        base.gamma += kappa * p / np.maximum(gp, 1)
        return base


class ScalarMemory:
    """A flat Dirichlet over {0, .5, 1}, initialized from smoothed q/p means."""
    def __init__(self, structured, concentration=1., counts=None):
        if structured.success_only:
            raise ValueError("Graded scalar comparison requires the graded utility")
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError("Scalar concentration must be positive")
        self.grasps, self.modes = structured.grasps, structured.modes
        self.options = structured.options
        q, p = means(structured.alpha), means(structured.gamma)
        full = q[:, Prefix.GOAL, None] + q[:, Prefix.READY, None] * p[..., Tail.GOAL]
        half = q[:, Prefix.READY, None] * p[..., Tail.TOWARD]
        marginal = np.stack([1 - full - half, half, full], -1).reshape(-1, 3)
        self.counts = positive_array(marginal * concentration if counts is None else counts,
                                    (len(self.options), 3), "scalar counts")

    def copy(self):
        base = StructuredMemory(self.grasps, self.modes)
        return ScalarMemory(base, counts=self.counts)

    def update(self, event):
        if not isinstance(event, Event):
            raise TypeError("Scalar memory accepts only an observation Event")
        self.counts[self.options.index(event.option), int(2 * event.utility())] += 1

    def scores(self, rng=None):
        p = means(self.counts) if rng is None else np.stack([rng.dirichlet(a) for a in self.counts])
        return p @ np.array([0., .5, 1.])

    def branches(self, c):
        # Canonical event representatives; only their scalar utility is used by update.
        return list(zip(means(self.counts)[self.options.index(c)],
                        [Event(c, Prefix.NOT_READY), Event(c, Prefix.READY, Tail.TOWARD),
                         Event(c, Prefix.GOAL)]))

    def snapshot(self):
        return {"kind": "scalar", "counts": self.counts.tolist()}
