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
    """Persistent baby lock + binary in-crib state.

    Simplified (2026-06-05): the 5-state activity machine was producing
    a 75% error rate on 'moving_a_lot', and the system's real job is
    just answering 'is the baby in the crib?'. We now emit one of two
    states:

        in_crib       — tracker is LOCKED to a person bbox inside the
                        crib ROI (the YOLO detection layer ROI-filters
                        before the tracker ever sees a box)
        out_of_frame  — no lock / no baby

    Posture and motion are accepted as optional inputs but only used by
    the existing release safeguards (upright postures + sustained
    activity release locks the system is confident shouldn't be held).
    They do not feed the state output.
    """

    BABY_STATES = ("out_of_frame", "in_crib")

    def __init__(
        self,
        *,
        acquire_frames: int = 4,
        iou_match: float = 0.20,
        hold_seconds: float = 60.0,
        smooth: float = 0.4,              # 0 = snap to new, 1 = ignore new
        sleep_seconds: float = 30.0,
        moving_a_lot_norm: float = 0.10,  # motion-score threshold (per second normalized)
        # Stale-lock release: if YOLO has been feeding us low-confidence
        # detections for stale_lock_seconds straight without a single
        # detection above stale_lock_max_conf, release the lock — almost
        # certainly a textured pattern / stuffed toy / chair back rather
        # than a real subject.
        stale_lock_seconds: float = 90.0,
        stale_lock_max_conf: float = 0.40,
        # Position-stability stale-lock release: even babies that are
        # sound asleep show 8-15+ pixels of bbox-center drift over a
        # couple of minutes from breathing and small shifts. An
        # inanimate subject (toy, blanket fold, shadow) drifts 0-4
        # pixels because there's literally nothing moving — the
        # smoothing in _smooth_update keeps the box pinned. We
        # observed exactly this with the 11:07-11:12 false positive:
        # center moved 2 px in x and 4 px in y across 4.5 minutes.
        #
        # After ``stable_lock_grace`` seconds of lock age, if the bbox
        # center has drifted less than ``stable_lock_max_drift`` pixels
        # over the trailing ``stable_lock_window`` window, release.
        # Position-stability tuning (2026-06-01): bumped after audit of
        # 999 incorrect labels on a real day showed the bear-on-crib FPs
        # were drifting 6-12 px frame-to-frame within the old 120 s /
        # 8 px window — narrowly above the release threshold. Wider
        # window + slightly looser drift catches the sustained sub-15 px
        # noise without releasing sleeping-baby locks that genuinely
        # show 15+ px of breathing motion over 3 minutes.
        stable_lock_grace: float = 60.0,
        stable_lock_window: float = 180.0,
        stable_lock_max_drift: float = 12.0,
        # Absolute age cap: even a long-sleeping baby gets re-acquired
        # periodically. Without this, the lock can survive for hours on
        # whatever subject originally captured it (audit of 03/06–04/06
        # showed individual locks running 2-11 hours, almost always
        # tracking an adult or a pillow rather than the baby).
        max_lock_seconds: float = 1800.0,
        # Upright postures inside a crib ROI almost certainly belong to an
        # adult leaning over the crib. A baby that can stand/walk would
        # be in a playpen, not a monitored crib. Releasing immediately on
        # these postures stops the lock from drifting onto the adult.
        upright_postures_release: tuple[str, ...] = (
            "standing", "walking", "running",
        ),
        # Sustained 'moving_a_lot' is the single biggest error signature
        # in the audit (260 / 366 incorrect snapshots, 75.6% error rate).
        # Real babies fidget in bursts of a few seconds; an adult moving
        # around the crib sustains high motion for far longer. After this
        # many seconds of continuous active motion, release the lock and
        # require a fresh re-acquisition.
        sustained_active_seconds: float = 30.0,
    ) -> None:
        self.acquire_frames = max(1, int(acquire_frames))
        self.iou_match = iou_match
        self.hold_seconds = hold_seconds
        self.smooth = smooth
        self.sleep_seconds = sleep_seconds
        self.moving_a_lot_norm = moving_a_lot_norm
        self.stale_lock_seconds = stale_lock_seconds
        self.stale_lock_max_conf = stale_lock_max_conf
        self.stable_lock_grace = stable_lock_grace
        self.stable_lock_window = stable_lock_window
        self.stable_lock_max_drift = stable_lock_max_drift
        self.max_lock_seconds = max_lock_seconds
        self.upright_postures_release = tuple(upright_postures_release)
        self.sustained_active_seconds = sustained_active_seconds

        self.state: LockState = LockState.NONE
        self.lock: LockedBox | None = None
        self._candidate: tuple[int, int, int, int] | None = None
        self._candidate_conf: float = 0.0
        self._acquire_count: int = 0
        self._lock_started_t: float | None = None
        self._max_conf_since_lock: float = 0.0
        # (t, cx, cy) ring of recent matched centers for drift check.
        self._center_history: list[tuple[float, float, float]] = []

        # Activity state machine (only valid while LOCKED)
        self.activity: str = "out_of_frame"
        self._still_since: float | None = None
        self._still_seconds: float = 0.0
        self._last_t: float = 0.0
        # Tracks the moment we first entered moving_a_lot within the
        # current lock — reset when activity flips out of it. Used for
        # the sustained_active_seconds release rule.
        self._moving_a_lot_since: float | None = None

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
        self._max_conf_since_lock = 0.0
        self._center_history = []
        self._still_since = None
        self._still_seconds = 0.0
        self._moving_a_lot_since = None
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
                    self._max_conf_since_lock = match_conf
                    self._center_history = [(
                        t,
                        (match[0] + match[2]) / 2.0,
                        (match[1] + match[3]) / 2.0,
                    )]
            else:
                self._acquire_count = max(0, self._acquire_count - 1)
                if self._acquire_count == 0:
                    self._candidate = None
                    self.state = LockState.NONE

        elif self.state == LockState.LOCKED:
            assert self.lock is not None
            if match is not None:
                # Stale-lock check: track best confidence seen during this
                # lock, and bail out if YOLO has been feeding only weak
                # detections for long enough that this clearly isn't a real
                # subject (stuffed toy, blanket pattern, chair back).
                if match_conf > self._max_conf_since_lock:
                    self._max_conf_since_lock = match_conf
                age = t - (self._lock_started_t or t)
                if (
                    age > self.stale_lock_seconds
                    and self._max_conf_since_lock < self.stale_lock_max_conf
                ):
                    self._release()
                    return None

                # Position-stability guard: any real subject (even a sleeping
                # baby) breathes and shifts; a toy / shadow / blanket fold
                # doesn't. After the grace window, if the bbox center has
                # drifted less than ``stable_lock_max_drift`` pixels across
                # the trailing ``stable_lock_window`` seconds, the lock is
                # almost certainly on an inanimate subject — release.
                cx = (match[0] + match[2]) / 2.0
                cy = (match[1] + match[3]) / 2.0
                self._center_history.append((t, cx, cy))
                cutoff = t - self.stable_lock_window
                self._center_history = [
                    e for e in self._center_history if e[0] >= cutoff
                ]
                if age > self.stable_lock_grace and len(self._center_history) >= 2:
                    span = t - self._center_history[0][0]
                    if span >= self.stable_lock_window * 0.5:
                        xs = [e[1] for e in self._center_history]
                        ys = [e[2] for e in self._center_history]
                        drift = max(
                            max(xs) - min(xs),
                            max(ys) - min(ys),
                        )
                        if drift < self.stable_lock_max_drift:
                            self._release()
                            return None

                new = self._smooth_update(self.lock, match)
                self.lock = LockedBox(
                    x1=new[0], y1=new[1], x2=new[2], y2=new[3],
                    last_seen_t=t,
                    age_s=age,
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
        self._update_activity(motion, t)

        # --------- additional release rules (post-audit 2026-06-04) ---------
        # The lock can drift onto an adult or a pillow that keeps getting
        # confidently re-detected; the original release rules (hold gap,
        # stale max-conf, position drift) miss this case because the new
        # subject IS moving and IS confidently classified. These rules
        # catch the specific patterns the audit surfaced.
        if self.state == LockState.LOCKED and self.lock is not None:
            # Rule 1 — absolute age cap. Forces a periodic re-acquisition
            # so a stuck lock can't survive for hours.
            if (
                self._lock_started_t is not None
                and (t - self._lock_started_t) > self.max_lock_seconds
            ):
                self._release()
                return None

            # Rule 2 — upright postures inside a crib ROI are an adult.
            # Babies that can stand/walk aren't in a monitored crib.
            if posture in self.upright_postures_release:
                self._release()
                return None

            # Rule 3 — sustained moving_a_lot. Babies fidget in short
            # bursts; long-running active motion is an adult.
            if self.activity == "moving_a_lot":
                if self._moving_a_lot_since is None:
                    self._moving_a_lot_since = t
                elif (t - self._moving_a_lot_since) > self.sustained_active_seconds:
                    self._release()
                    return None
            else:
                self._moving_a_lot_since = None

        return self.lock

    def _update_activity(self, motion: str | None, t: float) -> None:
        """Compute the binary in_crib / out_of_frame state. ``motion``
        is accepted so still-time tracking keeps working for callers
        that still classify motion; it no longer drives the state."""
        if self.state != LockState.LOCKED:
            self.activity = "out_of_frame"
            self._still_since = None
            self._still_seconds = 0.0
            return
        self.activity = "in_crib"
        # Still-time tracking is kept because re-train logic may want to
        # know how long the baby has been quiet — it just no longer maps
        # to a derived 'asleep' label.
        is_still = (motion == "still")
        if is_still:
            if self._still_since is None:
                self._still_since = t
            self._still_seconds = t - self._still_since
        else:
            self._still_since = None
            self._still_seconds = 0.0

    @property
    def still_seconds(self) -> float:
        return self._still_seconds
