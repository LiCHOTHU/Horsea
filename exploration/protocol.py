"""Frozen split/budget identities and independent deterministic random streams."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("prior", "scalar_voi", "structured_greedy", "structured_thompson", "structured_voi")


def seed_for(*parts):
    data = json.dumps(parts, separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


def rng_for(*parts):
    return np.random.default_rng(seed_for(*parts))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


@dataclass(frozen=True)
class Protocol:
    version: str = "exploration-v1"
    task: str = "open_microwave"
    train: tuple = tuple(range(210000, 210030))
    dev: tuple = tuple(range(220000, 220010))
    confirmation: tuple = tuple(range(230000, 230030))
    rollout_seeds: tuple = (0, 1, 2)
    probes: int = 3
    grasps: int = 2
    modes: int = 3
    train_repeats: int = 3
    kappa: float = 2.
    scalar_concentration: float = 1.
    scalar_strength_search: tuple = (1., 3., 6.)
    prefix_control_budget: int = 16
    tail_control_budget: int = 24
    goal_utility: tuple = (0., 0., 0., .5, 1., 0.)
    common_history_options: tuple = ((0, 0), (1, 0), (0, 1))

    def __post_init__(self):
        splits = [self.train, self.dev, self.confirmation]
        flat = [x for split in splits for x in split]
        if len(flat) != len(set(flat)):
            raise ValueError("Configuration splits overlap")
        if self.probes != 3 or not 1 <= self.grasps <= 2 or not 1 <= self.modes <= 3:
            raise ValueError("Invalid pilot budget")

    def record(self):
        d = asdict(self)
        d["methods"] = list(METHODS)
        d["budgets"] = {
            "train": len(self.train) * self.grasps * self.modes * self.train_repeats,
            "active": len(self.confirmation) * len(self.rollout_seeds) * len(METHODS) * (self.probes + 1),
            "common_history": len(self.confirmation) * (self.probes + 3),
        }
        d["claim"] = "reset-based selection within a fixed repertoire; no unseen-mechanism claim"
        return d
