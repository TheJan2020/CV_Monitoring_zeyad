"""Baby-monitor mode: persistent lock + simplified activity states.

Design goals (purpose-built, not a generic add-on):

1. **Lock-on detection**. YOLO frequently loses a baby that's small,
   partially under a blanket, or behind crib bars. Once the model has
   confirmed a baby's bounding box for N consecutive frames, we LOCK
   onto that bbox. From then on, the locked bbox is the source of truth
   for downstream scoring; YOLO is just a refresh signal. The lock
   survives extended detection gaps (default 60 s) before being released.

2. **Simplified state**. The generic pipeline classifies 11+ activities.
   For a sleeping baby that matters less than the user wanting to know
   whether the baby is:

       out_of_frame  — no lock at all
       asleep        — lying still longer than sleep threshold
       lying         — lying down, not yet asleep
       sitting       — sitting up
       moving_a_lot  — high motion in any posture

3. **Continuity beats accuracy**. False negatives on the baby (it's
   actually there) are much worse than the occasional held-too-long.
   Defaults err toward stickiness.

Hooking this into the existing worker:
    tracker = BabyTracker(...)
    locked = tracker.observe(t, person_dets, posture, motion, frame_h)
        # person_dets: list of YoloDetection person boxes this frame
        # posture/motion: from PoseStateTracker on the locked subject's pose
        # returns the locked detection (or None) — synth it into yolo_detections

    state = tracker.activity  # one of the 5 strings above
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


class LockState(str, Enum):
    NONE = "none"
    ACQUIRING = "acquiring"
    LOCKED = "locked"


@dataclass
class LockedBox:
    x1: int
    y1: int
    x2: int
    y2: int
    last_seen_t: float
    age_s: float          # how long the lock has been alive
    age_since_seen_s: float  # how long since last fresh YOLO detection
    confidence: float     # confidence of most recent fresh detection

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class BabyTracker:
    """Persistent baby lock + simplified 5-state activity machine."""

    BABY_STATES = ("out_of_frame", "asleep", "lying", "sitting", "moving_a_lot")

    def __init__(
        self,
        *,
        acquire_frames: int = 4,
        iou_match: float = 0.20,
        hold_seconds: float = 60.0,
        smooth: float = 0.4,              # 0 = snap to new, 1 = ignore new
        sleep_seconds: float = 30.0,
        moving_a_lot_norm: float = 0.10,  # motion-score threshold (per second normalized)
    ) -> None:
        self.acquire_frames = max(1, int(acquire_frames))
        self.iou_match = iou_match
        self.hold_seconds = hold_seconds
        self.smooth = smooth
        self.sleep_seconds = sleep_seconds
        self.moving_a_lot_norm = moving_a_lot_norm

        self.state: LockState = LockState.NONE
        self.lock: LockedBox | None = None
        self._candidate: tuple[int, int, int, int] | None = None
        self._candidate_conf: float = 0.0
        self._acquire_count: int = 0
        self._lock_started_t: float | None = None

        # Activity state machine (only valid while LOCKED)
        self.activity: str = "out_of_frame"
        self._still_since: float | None = None
        self._still_seconds: float = 0.0
        self._last_t: float = 0.0

    # -- internals --------------------------------------------------------

    def _smooth_update(self, old: LockedBox, new: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        a = self.smooth
        return (
            int(round(old.x1 * a + new[0] * (1 - a))),
            int(round(old.y1 * a + new[1] * (1 - a))),
            int(round(old.x2 * a + new[2] * (1 - a))),
            int(round(old.y2 * a + new[3] * (1 - a))),
        )

    def _release(self) -> None:
        self.state = LockState.NONE
        self.lock = None
        self._candidate = None
        self._candidate_conf = 0.0
        self._acquire_count = 0
        self._lock_started_t = None
        self._still_since = None
        self._still_seconds = 0.0
        self.activity = "out_of_frame"

    # -- public -----------------------------------------------------------

    def observe(
        self,
        t: float,
        person_boxes: Iterable[tuple[tuple[int, int, int, int], float]],
        *,
        posture: str | None,
        motion: str | None,
    ) -> LockedBox | None:
        """Feed one frame's worth of evidence.

        Args:
            t: monotonically increasing time (seconds; any origin)
            person_boxes: iterable of (bbox, confidence) for person-class detections
            posture: "lying" | "sitting" | "standing" | "upright" | "transitioning" | "unknown" | None
            motion: "still" | "moving" | "active" | "unknown" | None
            motion_score: normalized motion magnitude (0..1)

        Returns:
            The currently locked box (with metadata) or None if no lock.
        """
        self._last_t = t
        boxes = list(person_boxes)

        # Pick the YOLO box most relevant to the lock (largest if no lock yet,
        # best IoU if already locked/acquiring).
        match: tuple[int, int, int, int] | None = None
        match_conf: float = 0.0
        if boxes:
            if self.lock is not None or self._candidate is not None:
                target = self.lock.as_tuple() if self.lock else self._candidate
                best = max(boxes, key=lambda b: _iou(target, b[0]))
                if _iou(target, best[0]) >= self.iou_match:
                    match, match_conf = best
            else:
                # No lock yet — pick the largest box (most likely the baby in this ROI)
                def area(b):
                    bx, _ = b
                    return (bx[2] - bx[0]) * (bx[3] - bx[1])
                bx, c = max(boxes, key=area)
                match, match_conf = bx, c

        # Lock state machine
        if self.state == LockState.NONE:
            if match is not None:
                self.state = LockState.ACQUIRING
                self._candidate = match
                self._candidate_conf = match_conf
                self._acquire_count = 1

        elif self.state == LockState.ACQUIRING:
            if match is not None:
                self._candidate = match
                self._candidate_conf = match_conf
                self._acquire_count += 1
                if self._acquire_count >= self.acquire_frames:
                    self.state = LockState.LOCKED
                    self.lock = LockedBox(
                        x1=match[0], y1=match[1], x2=match[2], y2=match[3],
                        last_seen_t=t, age_s=0.0, age_since_seen_s=0.0,
                        confidence=match_conf,
                    )
                    self._lock_started_t = t
            else:
                self._acquire_count = max(0, self._acquire_count - 1)
                if self._acquire_count == 0:
                    self._candidate = None
                    self.state = LockState.NONE

        elif self.state == LockState.LOCKED:
            assert self.lock is not None
            if match is not None:
                new = self._smooth_update(self.lock, match)
                self.lock = LockedBox(
                    x1=new[0], y1=new[1], x2=new[2], y2=new[3],
                    last_seen_t=t,
                    age_s=t - (self._lock_started_t or t),
                    age_since_seen_s=0.0,
                    confidence=match_conf,
                )
            else:
                # No fresh detection; extend lock by elapsed time
                gap = t - self.lock.last_seen_t
                if gap > self.hold_seconds:
                    self._release()
                    return None
                self.lock = LockedBox(
                    x1=self.lock.x1, y1=self.lock.y1, x2=self.lock.x2, y2=self.lock.y2,
                    last_seen_t=self.lock.last_seen_t,
                    age_s=t - (self._lock_started_t or t),
                    age_since_seen_s=gap,
                    confidence=self.lock.confidence,
                )

        # Update activity state (only when LOCKED)
        self._update_activity(posture, motion, t)
        return self.lock

    def _update_activity(self, posture: str | None, motion: str | None, t: float) -> None:
        if self.state != LockState.LOCKED:
            self.activity = "out_of_frame"
            self._still_since = None
            self._still_seconds = 0.0
            return

        # Track contiguous still time (used for asleep)
        is_still = (motion == "still")
        if is_still:
            if self._still_since is None:
                self._still_since = t
            self._still_seconds = t - self._still_since
        else:
            self._still_since = None
            self._still_seconds = 0.0

        # Map to 5 simplified states.
        # Note: motion_score is already normalized 0..1 against the 'active'
        # threshold in PoseStateTracker, so checking it again here was
        # double-counting and firing on any small shift. Rely on the
        # categorical motion classification + upright postures only.
        moving_a_lot = (
            motion == "active"
            or posture in ("standing", "walking", "running")
        )
        if moving_a_lot:
            self.activity = "moving_a_lot"
        elif posture == "lying":
            if is_still and self._still_seconds >= self.sleep_seconds:
                self.activity = "asleep"
            else:
                self.activity = "lying"
        elif posture == "sitting":
            self.activity = "sitting"
        elif posture in ("upright", "transitioning"):
            self.activity = "sitting"
        else:
            # Posture unknown but locked. Hold previous activity if it was set;
            # otherwise default to lying (most common for a crib).
            if self.activity not in self.BABY_STATES or self.activity == "out_of_frame":
                self.activity = "lying"

    @property
    def still_seconds(self) -> float:
        return self._still_seconds
