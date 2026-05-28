"""
Activity classification from YOLO11-pose keypoints.

Designed for monitoring use cases (e.g. baby/person watch) where you want to
know if the subject is asleep, moving, sitting, etc., without training a new
model. All logic is heuristic and runs on the 17 COCO keypoints already
produced by the workbench pipeline.

Outputs:
    Posture  ∈ {unknown, lying, sitting, standing, upright, transitioning}
    Motion   ∈ {unknown, still, moving, active}
    activity (str)  — human-readable combination, e.g. "asleep", "playing"
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

# COCO 17 keypoints (Ultralytics YOLO-pose order)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

_KP_CONF_MIN = 0.15  # ignore keypoints below this confidence (lowered for small/distant subjects)


class Posture(str, Enum):
    UNKNOWN = "unknown"
    LYING = "lying"
    SITTING = "sitting"
    STANDING = "standing"
    UPRIGHT = "upright"            # standing/sitting indistinguishable
    TRANSITIONING = "transitioning"


class Motion(str, Enum):
    UNKNOWN = "unknown"
    STILL = "still"
    MOVING = "moving"
    ACTIVE = "active"


@dataclass
class PoseState:
    posture: Posture = Posture.UNKNOWN
    motion: Motion = Motion.UNKNOWN
    activity: str = "out_of_frame"
    motion_score: float = 0.0       # 0..1 normalized (1.0 ≈ active threshold)
    still_seconds: float = 0.0      # contiguous time stayed still
    posture_angle_deg: float = 0.0  # shoulder→hip vector angle from vertical


def _angle_at(b: tuple[float, float], a: tuple[float, float], c: tuple[float, float]) -> float:
    """Interior angle at vertex b between segments ba and bc, in degrees."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos))


class PoseStateTracker:
    """
    Rolling per-frame classifier:
        history_seconds:    sliding window used to smooth motion magnitude
        still_for_sleep_s:  how long "still + lying" before activity → 'asleep'
        motion_still_norm:  per-frame keypoint displacement / frame_h below which we call it 'still'
        motion_active_norm: per-frame displacement / frame_h above which we call it 'active'
    """

    def __init__(
        self,
        *,
        history_seconds: float = 3.0,
        still_for_sleep_s: float = 30.0,
        motion_still_norm: float = 0.003,
        motion_active_norm: float = 0.015,
        hold_seconds: float = 8.0,
    ) -> None:
        self.history_seconds = history_seconds
        self.still_for_sleep_s = still_for_sleep_s
        self.motion_still_norm = motion_still_norm
        self.motion_active_norm = motion_active_norm
        # When YOLO temporarily loses the subject (occluded by crib bar /
        # blanket / etc.), keep the previously classified state for up to
        # this many seconds before flipping to out_of_frame. Cuts on/off
        # flicker that's especially common with small / partially-visible
        # subjects like a baby in a crib.
        self.hold_seconds = hold_seconds

        self._buf: deque[tuple[float, Any, Any]] = deque()
        self._t: float = 0.0
        self._still_since: float | None = None
        self._prev_motion: Motion = Motion.UNKNOWN
        self._last_valid_t: float | None = None
        self.state = PoseState()

    def reset(self) -> None:
        self._buf.clear()
        self._still_since = None
        self._prev_motion = Motion.UNKNOWN
        self._last_valid_t = None
        self.state = PoseState()

    def update(
        self,
        keypoints_xy: Any | None,
        keypoints_conf: Any | None,
        dt: float,
        *,
        frame_h: int,
    ) -> PoseState:
        self._t += max(0.0, dt)

        if keypoints_xy is None or keypoints_conf is None or frame_h <= 0:
            # Subject momentarily lost. If we had a valid state within the
            # hold window, carry it forward so the UI / recorder don't see
            # transient flicker. Past that, fall back to out_of_frame.
            if (
                self._last_valid_t is not None
                and (self._t - self._last_valid_t) < self.hold_seconds
                and self.state.activity != "out_of_frame"
            ):
                return self.state
            self._buf.clear()
            self._still_since = None
            self._prev_motion = Motion.UNKNOWN
            self.state = PoseState(activity="out_of_frame")
            return self.state

        posture, angle = self._classify_posture(keypoints_xy, keypoints_conf)

        self._buf.append((self._t, keypoints_xy, keypoints_conf))
        while self._buf and self._t - self._buf[0][0] > self.history_seconds:
            self._buf.popleft()
        motion_norm = self._motion_score(frame_h)
        motion = self._classify_motion(motion_norm)

        if motion == Motion.STILL:
            if self._prev_motion != Motion.STILL:
                self._still_since = self._t
        else:
            self._still_since = None
        self._prev_motion = motion
        still_seconds = (self._t - self._still_since) if self._still_since is not None else 0.0

        activity = self._activity(posture, motion, still_seconds)
        self.state = PoseState(
            posture=posture,
            motion=motion,
            activity=activity,
            motion_score=min(1.0, motion_norm / max(self.motion_active_norm, 1e-6)),
            still_seconds=still_seconds,
            posture_angle_deg=angle,
        )
        self._last_valid_t = self._t
        return self.state

    # ---- internal -------------------------------------------------------

    def _classify_posture(self, xy: Any, conf: Any) -> tuple[Posture, float]:
        s_pts = [(xy[i][0], xy[i][1]) for i in (L_SHOULDER, R_SHOULDER) if conf[i] >= _KP_CONF_MIN]
        h_pts = [(xy[i][0], xy[i][1]) for i in (L_HIP, R_HIP) if conf[i] >= _KP_CONF_MIN]
        if not s_pts or not h_pts:
            return Posture.UNKNOWN, 0.0
        sx = sum(p[0] for p in s_pts) / len(s_pts)
        sy = sum(p[1] for p in s_pts) / len(s_pts)
        hx = sum(p[0] for p in h_pts) / len(h_pts)
        hy = sum(p[1] for p in h_pts) / len(h_pts)
        dx, dy = hx - sx, hy - sy
        # angle of (shoulder→hip) from the vertical axis, mapped to [0, 90]
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        if angle > 60:
            return Posture.LYING, angle
        if angle > 30:
            return Posture.TRANSITIONING, angle

        # Upright — try to distinguish sitting vs standing via knee bend
        knee_angles: list[float] = []
        for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
            if conf[hip] >= _KP_CONF_MIN and conf[knee] >= _KP_CONF_MIN and conf[ankle] >= _KP_CONF_MIN:
                knee_angles.append(_angle_at(
                    (float(xy[knee][0]), float(xy[knee][1])),
                    (float(xy[hip][0]),  float(xy[hip][1])),
                    (float(xy[ankle][0]),float(xy[ankle][1])),
                ))
        if knee_angles:
            mean_knee = sum(knee_angles) / len(knee_angles)
            return (Posture.SITTING, angle) if mean_knee < 130 else (Posture.STANDING, angle)
        return Posture.UPRIGHT, angle

    def _motion_score(self, frame_h: int) -> float:
        buf = list(self._buf)
        if len(buf) < 2:
            return 0.0
        pair_means: list[float] = []
        for (_, xy_a, conf_a), (_, xy_b, conf_b) in zip(buf, buf[1:]):
            ds: list[float] = []
            n = min(len(xy_a), len(xy_b), len(conf_a), len(conf_b))
            for i in range(n):
                if conf_a[i] < _KP_CONF_MIN or conf_b[i] < _KP_CONF_MIN:
                    continue
                dx = float(xy_a[i][0]) - float(xy_b[i][0])
                dy = float(xy_a[i][1]) - float(xy_b[i][1])
                ds.append(math.hypot(dx, dy))
            if ds:
                pair_means.append(sum(ds) / len(ds))
        if not pair_means:
            return 0.0
        avg_per_frame_px = sum(pair_means) / len(pair_means)
        return avg_per_frame_px / max(1, frame_h)

    def _classify_motion(self, norm: float) -> Motion:
        if norm <= 0.0 and len(self._buf) < 2:
            return Motion.UNKNOWN
        if norm < self.motion_still_norm:
            return Motion.STILL
        if norm < self.motion_active_norm:
            return Motion.MOVING
        return Motion.ACTIVE

    def _activity(self, posture: Posture, motion: Motion, still_s: float) -> str:
        if posture == Posture.UNKNOWN:
            return "uncertain"
        if posture == Posture.LYING:
            if motion == Motion.STILL:
                return "asleep" if still_s >= self.still_for_sleep_s else "resting"
            if motion == Motion.MOVING:
                return "fidgeting"
            if motion == Motion.ACTIVE:
                return "restless"
            return "lying"
        if posture == Posture.SITTING:
            if motion == Motion.STILL:
                return "sitting_calm"
            if motion == Motion.MOVING:
                return "playing"
            if motion == Motion.ACTIVE:
                return "very_active"
            return "sitting"
        if posture == Posture.STANDING:
            if motion == Motion.STILL:
                return "standing"
            if motion == Motion.MOVING:
                return "walking"
            if motion == Motion.ACTIVE:
                return "running"
            return "standing"
        if posture == Posture.UPRIGHT:
            return "upright_still" if motion == Motion.STILL else "upright_moving"
        if posture == Posture.TRANSITIONING:
            return "transitioning"
        return "uncertain"
