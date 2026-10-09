"""A deterministic stand-in backend, so the experiment pipeline can be validated without a tracker.

This exists to answer "is the experiment runnable and correct?" separately from "can we see the
microwave?". It fabricates *sensor frames*, not events: every frame goes through the real
`Feedback` extractor, so feedback.py, memory.py, selectors.py and trials.py are all exercised on the
same code path a RoboTwin backend would use. Only the pixels-to-TrackedFrame step is replaced.

Ground truth per scene (which grasp holds, which mode opens) is derived from the scene seed, is
reported only through `evaluation()`, and never reaches the returned observation record.

Prefix behaviour depends on the grasp and NOT on the mode -- that is the property the shared-prefix
argument rests on, and `fingerprint` is computed so a violation would show up as a differing hash.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from .events import Option, Prefix
from .feedback import Feedback, TrackedFrame
from .trials import PrefixResult, TailResult


@dataclass(frozen=True)
class SyntheticTruth:
    """Latent scene facts. Privileged: reported via evaluation(), never via observations."""
    good_grasp: int
    good_mode: int
    hold_rate_good: float = .9
    hold_rate_bad: float = .1


# v0's long memory is a POOLED count table over (grasp, mode) -- it does not condition on the
# observation. So the only thing it can transfer is a population bias: some grasps/modes work more
# often across scenes. Drawing the answer uniformly per scene (the first version of this fixture)
# makes the fitted prior uninformative by construction and the arm ladder measures noise.
# `bias` is the probability that the population-favoured grasp/mode is the one that works here.
GRASP_BIAS = .8
MODE_BIAS = .6


def truth_for(scene, grasps, modes, grasp_bias=GRASP_BIAS, mode_bias=MODE_BIAS):
    rng = np.random.default_rng(scene)
    if grasps > 1 and rng.random() < grasp_bias:
        good_grasp = 0                      # the population-favoured grasp
    else:
        good_grasp = int(rng.integers(grasps))
    if modes > 1 and rng.random() < mode_bias:
        good_mode = 0
    else:
        good_mode = int(rng.integers(modes))
    return SyntheticTruth(good_grasp, good_mode)


class SyntheticBackend:
    """Implements the TrialManager backend contract with scripted sensor frames."""

    def __init__(self, protocol, feedback=None, unknown_rate=0.0):
        self.protocol = protocol
        self.feedback = feedback or Feedback()
        self.unknown_rate = unknown_rate      # fraction of attempts with degraded tracking
        self.truth = None
        self.scene = None
        self._held = None
        self._opening = 0.0

    # --- lifecycle ----------------------------------------------------------------
    def reset(self, scene, seed):
        self.scene = scene
        self.truth = truth_for(scene, self.protocol.grasps, self.protocol.modes)
        self._held = None
        self._opening = 0.0
        self._reset_seed = seed

    def close(self):
        pass

    def evaluation(self):
        return {"backend": "synthetic", "good_grasp": self.truth.good_grasp,
                "good_mode": self.truth.good_mode, "held": self._held,
                "opening": float(self._opening), "simulator_success": bool(self._opening >= 1.0),
                "privileged_access": ["synthetic latent opening; not a robot simulator"]}

    # --- prefix -------------------------------------------------------------------
    def execute_prefix(self, grasp, rng):
        t = self.protocol
        hold_p = self.truth.hold_rate_good if grasp == self.truth.good_grasp else self.truth.hold_rate_bad
        self._held = bool(rng.random() < hold_p)
        degraded = rng.random() < self.unknown_rate
        conf = 0.4 if degraded else 0.95

        # A closing gripper: open before contact, closed during verification.
        base = np.array([0.30 + 0.01 * grasp, -0.10, 0.90])
        approach = [TrackedFrame(tuple(base - np.array([0.05, 0, 0])), 1.0,
                                 tuple(base), None, conf)]
        # Verification motion: the hand moves a little; the handle follows iff the grasp held.
        verification = []
        for i in range(3):
            eef = base + np.array([0.004 * (i + 1), 0, 0])
            if self._held:
                handle = eef                      # relation maintained -> READY
            else:
                handle = base + np.array([0.0, 0.0, 0.08])   # far away -> NOT_READY
            verification.append(TrackedFrame(tuple(eef), 0.25, tuple(handle), None, conf))
        self._eef = np.array(verification[-1].eef)
        self._handle = np.array(verification[-1].handle)

        event = self.feedback.prefix(approach, verification)
        steps = min(t.prefix_control_budget, 4 + len(verification))
        return PrefixResult(event, steps, self._fingerprint(grasp), self._obs(approach + verification))

    def _fingerprint(self, grasp):
        """Depends on the grasp only. A mode-dependent prefix would change this hash."""
        # Compare only paired execution streams; outcomes can differ between attempts.
        return hashlib.sha256(f"prefix|grasp={grasp}|scene={self.scene}|seed={self._reset_seed}|held={self._held}".encode()).hexdigest()[:16]

    # --- tail ---------------------------------------------------------------------
    def execute_mode(self, option, rng):
        t = self.protocol
        opens = option.mode == self.truth.good_mode
        slip = rng.random() < 0.05                     # occasional slip during manipulation
        frames = []
        for i in range(4):
            eef = self._eef + np.array([0.01 * (i + 1), 0, 0])
            if slip and i >= 2:
                handle = self._handle                  # relation lost -> SLIP
                opening = 0.0
            else:
                handle = eef
                opening = (1.2 if opens else 0.0) * (i + 1) / 3.0
            frames.append(TrackedFrame(tuple(eef), 0.25, tuple(handle), float(opening), 0.95))
        self._opening = frames[-1].opening
        event = self.feedback.tail(frames)
        steps = min(t.tail_control_budget, 6 + len(frames))
        return TailResult(event, max(1, steps), self._obs(frames))

    # --- observation record -------------------------------------------------------
    @staticmethod
    def _obs(frames):
        """Only observable quantities. No ground-truth route."""
        return {"frames": [{"eef": list(f.eef), "gripper": f.gripper,
                            "handle": None if f.handle is None else list(f.handle),
                            "opening": f.opening, "confidence": f.confidence} for f in frames]}
