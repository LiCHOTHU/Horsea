"""RGB-D marker tracking. No simulator imports, joints, contacts, or segmentation.

Camera intrinsics/extrinsics are ordinary calibrated sensor parameters. Marker
centres are measured from rendered colors and depth, never scene transforms.
"""
import numpy as np

from .feedback import TrackedFrame


COLORS = {"door0": (1, 0, 0), "door1": (0, 1, 0), "door2": (0, 0, 1),
          "body0": (0, 1, 1), "body1": (1, 0, 1), "body2": (1, 1, 0)}


def tool_point(eef):
    """Measured RoboTwin EEF pose -> tool center using the fixed robot calibration."""
    from scipy.spatial.transform import Rotation
    eef = np.asarray(eef)
    return eef[:3] + .12 * Rotation.from_quat(eef[[4, 5, 6, 3]]).as_matrix()[:, 0]


def backproject(uv, depth_m, intrinsic, extrinsic):
    p = np.linalg.solve(intrinsic, np.array([uv[0], uv[1], 1.])) * depth_m
    return extrinsic[:3, :3].T @ (p - extrinsic[:3, 3])


def frame_from_markers(points, part):
    if any(part + str(i) not in points for i in range(3)):
        return None
    a, b, c = [np.asarray(points[part + str(i)]) for i in range(3)]
    x, y = b - a, c - a
    if np.linalg.norm(x) < .025 or np.linalg.norm(y) < .025:
        return None
    x = x / np.linalg.norm(x)
    y = y - np.dot(x, y) * x
    if np.linalg.norm(y) < .02:
        return None
    y /= np.linalg.norm(y)
    out = np.eye(4)
    out[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
    out[:3, 3] = a
    return out


class MarkerTracker:
    def __init__(self):
        self.closed_relative = None
        self.handle_local = None
        self.last = None
        self.first_observation = True
        self.last_points = {}
        self.marker_lengths = {}

    def observe(self, sensors):
        import cv2
        estimates = {k: [] for k in COLORS}
        for camera_id, camera in sensors["cameras"].items():
            rgb = np.asarray(camera["rgb"], dtype=float) / 255.
            for name, code in COLORS.items():
                high, low = np.array(code, dtype=bool), ~np.array(code, dtype=bool)
                mask = (rgb[..., high].min(-1) > .45) & (rgb[..., low].max(-1) < .35)
                n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                candidates = [i for i in range(1, n) if 3 <= stats[i, cv2.CC_STAT_AREA] <= 500]
                if not candidates:
                    continue
                for i in candidates:
                    uv = centers[i]
                    valid = (labels == i) & np.isfinite(camera["depth_m"]) & (camera["depth_m"] > .1)
                    if valid.sum() < 3:
                        continue
                    depth = float(np.median(camera["depth_m"][valid]))
                    point = backproject(uv, depth, camera["intrinsic"], camera["extrinsic"])
                    # Reject similarly colored robot logos using camera-measured
                    # temporal proximity, never simulator marker coordinates.
                    if name in self.last_points and np.linalg.norm(point - self.last_points[name]) > .06:
                        continue
                    estimates[name].append((camera_id, point))
        points = {}
        for name, values in estimates.items():
            if values:
                clusters = [[(c, v) for c, v in values if np.linalg.norm(v - p) < .018] for _, p in values]
                def quality(cluster):
                    center = np.mean([v for _, v in cluster], axis=0)
                    distance = np.linalg.norm(center - self.last_points[name]) if name in self.last_points else 0.
                    return len({c for c, _ in cluster}), -distance
                good = max(clusters, key=quality)
                points[name] = np.mean([v for _, v in good], axis=0)
        # Enforce the rigid triangle measured in the closed reset image. A bad
        # correspondence becomes missing feedback instead of a spurious goal.
        for part in ("door", "body"):
            names = [part + str(i) for i in range(3)]
            if all(n in points for n in names):
                lengths = np.array([np.linalg.norm(points[names[a]] - points[names[b]])
                                    for a, b in ((0, 1), (0, 2), (1, 2))])
                if self.first_observation:
                    self.marker_lengths[part] = lengths
                elif part not in self.marker_lengths or np.max(abs(lengths - self.marker_lengths[part])) > .012:
                    for name in names:
                        points.pop(name)
        self.last_points.update(points)
        door, body = frame_from_markers(points, "door"), frame_from_markers(points, "body")
        opening, handle, confidence = None, None, 0.
        if door is not None:
            confidence = .95
            if self.handle_local is not None:
                handle = tuple((door @ np.r_[self.handle_local, 1.])[:3])
        if door is not None and body is not None:
            relative = body[:3, :3].T @ door[:3, :3]
            if self.closed_relative is None and self.first_observation:
                self.closed_relative = relative.copy()
            if self.closed_relative is not None:
                rotation = relative @ self.closed_relative.T
                opening = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1., 1.)))
        self.first_observation = False
        tracked = TrackedFrame(tuple(tool_point(sensors["left_eef"])), float(sensors["left_gripper"]),
                               handle, opening, confidence)
        self.last = {"markers": {k: v.tolist() for k, v in points.items()},
                     "door": None if door is None else door.tolist(),
                     "body": None if body is None else body.tolist(),
                     "opening": opening, "confidence": confidence,
                     "views_per_marker": {k: len({camera for camera, _ in v}) for k, v in estimates.items()}}
        return tracked
