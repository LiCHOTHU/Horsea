"""Exact one-observation VOI; repeating it for three probes is myopic."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Decision:
    index: int
    strategy: str
    scores: list
    values: list
    seconds: float


def first_max(values, tolerance=1e-12):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite decision values")
    return int(np.flatnonzero(values >= values.max() - tolerance)[0])


def voi(memory):
    scores = memory.scores()
    baseline = scores.max()
    values = []
    for c in memory.options:
        branches = memory.branches(c)
        if not np.isclose(sum(p for p, _ in branches), 1., atol=1e-12):
            raise ValueError("Predictive event probabilities must sum to one")
        future = 0.
        for prob, event in branches:
            imagined = memory.copy()
            imagined.update(event)
            future += prob * imagined.scores().max()
        gain = future - baseline
        if gain < -1e-10:
            raise ArithmeticError("Negative VOI violates the predictive martingale")
        values.append(max(0., float(gain)))
    return np.array(values)


def select(memory, strategy="voi", rng=None):
    start = time.perf_counter()
    scores = memory.scores()
    if strategy == "voi":
        values = voi(memory)
        index = first_max(scores if values.max() <= 1e-12 else values)
    elif strategy == "greedy":
        values = scores.copy()
        index = first_max(values)
    elif strategy == "thompson":
        if rng is None:
            raise ValueError("Thompson sampling needs an explicit independent RNG")
        values = memory.scores(rng)
        index = first_max(values)
    else:
        raise ValueError(strategy)
    return Decision(index, strategy, scores.tolist(), values.tolist(), time.perf_counter() - start)
