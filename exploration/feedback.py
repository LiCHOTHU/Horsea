"""Deterministic observation-only event extraction after visual tracking.

The tracker supplies metric handle/EEF positions and a panel-to-body opening
angle. Missing measurements remain None. Ground truth has no input route here.
Thresholds are provisional until the development audit freezes them.
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from .events import Prefix, Tail


@dataclass(frozen=True)
class TrackedFrame:
    eef: Tuple[float, float, float]
    gripper: float
    handle: Optional[Tuple[float, float, float]]
    opening: Optional[float]
    confidence: float

    def __post_init__(self):
        if not np.isfinite(self.eef).all() or len(self.eef) != 3:
            raise ValueError("Invalid proprioceptive EEF position")
        if not np.isfinite(self.gripper) or not 0 <= self.gripper <= 1:
            raise ValueError("Gripper opening must be normalized to [0,1]")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid tracking confidence")
        if self.handle is not None and (len(self.handle) != 3 or not np.isfinite(self.handle).all()):
            raise ValueError("Invalid tracked handle")
        if self.opening is not None and not np.isfinite(self.opening):
            raise ValueError("Missing opening must be None, never NaN")


@dataclass(frozen=True)
class Thresholds:
    confidence: float = .7
    goal_angle: float = 1.0
    progress_angle: float = .05
    grasp_distance: float = .045
    relation_tolerance: float = .025
    min_verification_motion: float = .003
    min_closure: float = .2
    max_closed_opening: float = .6


class Feedback:
    def __init__(self, thresholds=None):
        self.t = thresholds or Thresholds()

    def _visible(self, f):
        return f.handle is not None and f.confidence >= self.t.confidence

    def _goal(self, frames):
        return any(f.opening is not None and f.confidence >= self.t.confidence
                   and f.opening >= self.t.goal_angle for f in frames)

    def prefix(self, approach, verification):
        all_frames = list(approach) + list(verification)
        if self._goal(all_frames):
            return Prefix.GOAL
        if not approach or len(verification) < 2:
            return Prefix.UNKNOWN
        before, a, b = approach[0], verification[0], verification[-1]
        if not all(self._visible(f) for f in verification):
            return Prefix.UNKNOWN
        closed = before.gripper - b.gripper >= self.t.min_closure and b.gripper <= self.t.max_closed_opening
        dist = np.linalg.norm(np.subtract(b.handle, b.eef))
        if not closed or dist > self.t.grasp_distance:
            return Prefix.NOT_READY
        moved = np.linalg.norm(np.subtract(b.eef, a.eef))
        if moved < self.t.min_verification_motion:
            return Prefix.UNKNOWN
        part_moved = np.linalg.norm(np.subtract(b.handle, a.handle))
        if part_moved < self.t.min_verification_motion:
            return Prefix.NOT_READY
        relative = [np.subtract(f.handle, f.eef) for f in verification]
        error = max(np.linalg.norm(x - relative[0]) for x in relative)
        return Prefix.READY if error <= self.t.relation_tolerance else Prefix.NOT_READY

    def tail(self, frames):
        if self._goal(frames):
            return Tail.GOAL
        if len(frames) < 2 or not all(self._visible(f) for f in frames):
            return Tail.UNKNOWN
        relative = [np.subtract(f.handle, f.eef) for f in frames]
        if max(np.linalg.norm(x - relative[0]) for x in relative) > self.t.relation_tolerance:
            return Tail.SLIP
        if any(f.opening is None for f in frames):
            return Tail.UNKNOWN
        progress = frames[-1].opening - frames[0].opening
        if abs(progress) < self.t.progress_angle:
            return Tail.NO_RESPONSE
        return Tail.TOWARD if progress > 0 else Tail.AWAY

    def configuration(self):
        return asdict(self.t)
