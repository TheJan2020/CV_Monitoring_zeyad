"""Score frames for workbench activity (pose + optional YOLO)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

# COCO class names useful on a workbench (YOLOv8 COCO)
DEFAULT_TOOL_CLASSES = frozenset(
    {
        "person",
        "knife",
        "scissors",
        "spoon",
        "fork",
        "cup",
        "bowl",
        "bottle",
        "wine glass",
        "book",
        "laptop",
        "cell phone",
        "mouse",
        "keyboard",
        "backpack",
        "handbag",
    }
)
TOOL_CLASSES = DEFAULT_TOOL_CLASSES - {"person"}

# Shown in UI — YOLO only knows COCO names, not "screwdriver" / "soldering iron"
COCO_TOOL_LABELS = sorted(TOOL_CLASSES)

# Furniture / scene — never drawn as tools
BENCH_OBJ_SKIP = frozenset(
    {"chair", "couch", "bed", "dining table", "bench", "potted plant"}
)

# COCO classes that are not tools (animals, vehicles, food, sports, …)
COCO_NOT_TOOLS = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "umbrella",
        "tie",
        "suitcase",
    }
)

# Handheld tools: ignore furniture + huge boxes in tool scoring
HANDHELD_CLASSES = TOOL_CLASSES | {"cell phone"}


def is_tool_like(name: str, *, show_all_coco: bool = True) -> bool:
    """True for COCO objects we should show/count as tools (not person/furniture/food/animals)."""
    if not name or name == "person":
        return False
    if name in BENCH_OBJ_SKIP or name in COCO_NOT_TOOLS:
        return False
    if show_all_coco:
        return name not in HAND_WORK_DEVICE_NAMES
    return name in TOOL_CLASSES
PHONE_CLASS_NAMES = frozenset({"cell phone"})
YOLO_CELL_PHONE_CLASS_ID = 67  # COCO class for cell phone
YOLO_LAPTOP_CLASS_ID = 63  # COCO class for laptop
LAPTOP_CLASS_NAMES = frozenset({"laptop"})
YOLO_KEYBOARD_CLASS_ID = 66  # COCO class for keyboard
KEYBOARD_CLASS_NAMES = frozenset({"keyboard"})
YOLO_MOUSE_CLASS_ID = 64  # COCO class for mouse
MOUSE_CLASS_NAMES = frozenset({"mouse"})
HAND_WORK_DEVICE_NAMES = (
    PHONE_CLASS_NAMES | LAPTOP_CLASS_NAMES | KEYBOARD_CLASS_NAMES | MOUSE_CLASS_NAMES
)
WORK_DEVICE_CLASS_IDS = (
    YOLO_CELL_PHONE_CLASS_ID,
    YOLO_LAPTOP_CLASS_ID,
    YOLO_KEYBOARD_CLASS_ID,
    YOLO_MOUSE_CLASS_ID,
)
_CLASS_ID_TO_NAMES: dict[int, frozenset[str]] = {
    YOLO_CELL_PHONE_CLASS_ID: PHONE_CLASS_NAMES,
    YOLO_LAPTOP_CLASS_ID: LAPTOP_CLASS_NAMES,
    YOLO_KEYBOARD_CLASS_ID: KEYBOARD_CLASS_NAMES,
    YOLO_MOUSE_CLASS_ID: MOUSE_CLASS_NAMES,
}
class ActivityState(str, Enum):
    IDLE = "idle"
    PRESENT = "present"
    WORKING = "working"


def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm. Points are normalized (0..1)."""
    n = len(points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class BenchROI:
    """Region of interest. Either a normalized rectangle (x1,y1,x2,y2) or
    an ordered list of normalized polygon vertices (`points`). When both
    are set, the polygon wins.
    """

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0
    points: list[tuple[float, float]] | None = None

    def is_polygon(self) -> bool:
        return self.points is not None and len(self.points) >= 3

    def is_full_frame(self) -> bool:
        if self.is_polygon():
            xs = [p[0] for p in self.points]  # type: ignore[union-attr]
            ys = [p[1] for p in self.points]  # type: ignore[union-attr]
            return (
                min(xs) <= 0.01 and min(ys) <= 0.01
                and max(xs) >= 0.99 and max(ys) >= 0.99
            )
        return self.x1 <= 0.01 and self.y1 <= 0.01 and self.x2 >= 0.99 and self.y2 >= 0.99

    def contains(self, x: float, y: float) -> bool:
        if self.is_polygon():
            return _point_in_polygon(x, y, self.points)  # type: ignore[arg-type]
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Bounding box of the region in pixel coordinates."""
        if self.is_polygon():
            xs = [p[0] for p in self.points]  # type: ignore[union-attr]
            ys = [p[1] for p in self.points]  # type: ignore[union-attr]
            return (int(min(xs) * w), int(min(ys) * h),
                    int(max(xs) * w), int(max(ys) * h))
        return (
            int(self.x1 * w),
            int(self.y1 * h),
            int(self.x2 * w),
            int(self.y2 * h),
        )

    def polygon_pixels(self, w: int, h: int) -> list[tuple[int, int]] | None:
        """Vertices in pixel coords, or None for a rectangle ROI."""
        if not self.is_polygon():
            return None
        return [(int(p[0] * w), int(p[1] * h)) for p in self.points]  # type: ignore[union-attr]


@dataclass
class YoloDetection:
    """One YOLO box with explainability flags."""

    name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    in_roi: bool
    near_hand: bool
    role: str  # person | tool_ok | tool_miss | bench_obj | phone | other
    inside_person: bool = False  # center of this box inside a person box (likely held)
    from_hand_crop: bool = False  # legacy flag (unused)
    is_primary: bool = False  # single tracked person used for scoring
    stable: bool = False  # seen ≥stable_seconds with conf ≥stable_min_conf
    stable_age_s: float = 0.0  # current qualifying streak length (seconds)


@dataclass
class PersonPose:
    """Skeleton for one person (MediaPipe landmarks and/or YOLO11 COCO keypoints)."""

    landmarks: Any | None = None
    is_primary: bool = False
    keypoints_xy: Any | None = None  # (17, 2) pixel coords when from YOLO-pose
    keypoints_conf: Any | None = None  # (17,) per-keypoint confidence


@dataclass
class WristBox:
    """Square hand region (pixels), centered past the wrist toward the fingers."""

    label: str  # e.g. left, right, P1_left
    x1: int
    y1: int
    x2: int
    y2: int
    person_index: int = 0
    is_primary_person: bool = False

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    def area(self) -> int:
        return max(1, (self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass
class HandObjectOverlap:
    """Overlap between one hand region box and a phone or laptop detection."""

    wrist_label: str
    object_name: str  # cell phone | laptop
    object_conf: float
    object_x1: int
    object_y1: int
    iou_percent: float  # intersection / union × 100
    hand_coverage_percent: float  # intersection / hand area × 100
    object_coverage_percent: float  # intersection / object area × 100


# Backward-compatible alias
WristPhoneOverlap = HandObjectOverlap
WristLaptopOverlap = HandObjectOverlap


# COCO 17-keypoint skeleton (Ultralytics YOLO-pose)
COCO_POSE_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# Map COCO keypoint index → MediaPipe Pose index for score_pose()
_COCO_TO_MP = {
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
}


@dataclass
class FrameScore:
    person_visible: bool = False
    person_in_roi: float = 0.0
    hands_in_roi: float = 0.0
    active_pose: float = 0.0
    tool_near_bench: float = 0.0
    named_tool_score: float = 0.0
    bench_clutter_score: float = 0.0
    tool_named_hits: int = 0
    tool_named_checks: int = 0
    bench_obj_count: int = 0
    on_phone: bool = False
    phone_near_hand: bool = False  # overlap now (before stability timer)
    on_laptop: bool = False
    on_keyboard: bool = False
    on_mouse: bool = False
    medium_score: float = 0.0
    strict_score: float = 0.0
    pose_used_fallback: bool = False  # True only when YOLO person box updated scores
    best_phone_wrist_iou_pct: float = 0.0  # max IoU% across hand×phone pairs this frame
    best_laptop_hand_iou_pct: float = 0.0  # max IoU% across hand×laptop pairs this frame
    best_keyboard_hand_iou_pct: float = 0.0  # max IoU% across hand×keyboard pairs this frame
    best_mouse_hand_iou_pct: float = 0.0  # max IoU% across hand×mouse pairs this frame

    @property
    def at_computer_work(self) -> bool:
        """Hands overlapping stable laptop or keyboard — drives WORKING state/banner."""
        return self.on_laptop or self.on_keyboard

    @property
    def hands_on_laptop_or_keyboard(self) -> bool:
        """Alias for activity rules (laptop/keyboard only, not mouse)."""
        return self.at_computer_work

    @property
    def phone_in_hand(self) -> bool:
        """Stable in-hand phone, or immediate hand overlap (banner / red box)."""
        return self.on_phone or self.phone_near_hand

    def compute_totals(self, *, use_yolo: bool) -> None:
        hands = self.hands_in_roi
        if self.phone_in_hand:
            # Phone in hand — not bench/computer work
            hands = min(hands, 0.15)
        elif self.at_computer_work:
            # Typing / using laptop — treat as active hands at workstation
            hands = max(hands, 0.45)

        # Phase 2 — medium
        self.medium_score = (
            0.35 * self.person_in_roi
            + 0.35 * hands
            + 0.30 * self.active_pose
        )
        # Phase 3 — strict: prefer named COCO tools; clutter only if no named tools seen
        if use_yolo:
            tool_part = 0.20 * (
                self.named_tool_score if self.tool_named_checks > 0 else self.bench_clutter_score
            )
        else:
            tool_part = 0.0
        self.strict_score = min(1.0, self.medium_score + tool_part)
        self.tool_near_bench = (
            self.named_tool_score if self.tool_named_checks > 0 else self.bench_clutter_score
        )


@dataclass
class ActivityTracker:
    """Time-based state machine with hysteresis."""

    work_on_s: float = 10.0
    work_off_s: float = 4.0
    present_on_s: float = 3.0
    medium_threshold: float = 0.50
    strict_threshold: float = 0.62
    present_threshold: float = 0.28

    state: ActivityState = ActivityState.IDLE
    medium_streak: float = 0.0
    strict_streak: float = 0.0
    present_streak: float = 0.0
    work_surface_streak: float = 0.0  # hands on laptop/keyboard (stable)
    working_cooldown: float = 0.0

    def update(self, score: FrameScore, dt: float) -> ActivityState:
        hands_work = score.hands_on_laptop_or_keyboard

        if score.person_visible:
            if score.medium_score >= self.present_threshold:
                self.present_streak += dt
            else:
                self.present_streak = max(0.0, self.present_streak - dt)
        else:
            self.present_streak = max(0.0, self.present_streak - dt * 2.0)

        if score.medium_score >= self.medium_threshold:
            self.medium_streak += dt
        else:
            self.medium_streak = max(0.0, self.medium_streak - dt)

        if score.strict_score >= self.strict_threshold:
            self.strict_streak += dt
        else:
            self.strict_streak = max(0.0, self.strict_streak - dt)

        if hands_work and not score.on_phone:
            self.work_surface_streak += dt
        else:
            self.work_surface_streak = max(0.0, self.work_surface_streak - dt * 2.0)

        if self.state == ActivityState.WORKING:
            still_active = score.person_visible and hands_work and not score.on_phone
            if still_active:
                self.working_cooldown = self.work_off_s
            else:
                self.working_cooldown = max(0.0, self.working_cooldown - dt)
                if self.working_cooldown <= 0:
                    self.state = (
                        ActivityState.PRESENT
                        if self.present_streak >= self.present_on_s
                        else ActivityState.IDLE
                    )
                    self.work_surface_streak = 0.0
        else:
            if (
                hands_work
                and not score.on_phone
                and self.work_surface_streak >= self.work_on_s
            ):
                self.state = ActivityState.WORKING
                self.working_cooldown = self.work_off_s
            elif self.present_streak >= self.present_on_s:
                self.state = ActivityState.PRESENT
            elif not score.person_visible and self.present_streak <= 0:
                self.state = ActivityState.IDLE

        return self.state


def _lm(landmarks: Any, idx: int) -> tuple[float, float, float]:
    p = landmarks.landmark[idx]
    return p.x, p.y, p.visibility


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Angle at b in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos))


def _best_yolo_person_norm(
    frame: Any,
    results: Any,
    roi: BenchROI,
) -> tuple[float, float, float, float] | None:
    """Best YOLO person in ROI: largest area when ROI is full frame, else largest overlap."""
    if results is None or len(results) == 0:
        return None
    h, w = frame.shape[:2]
    names = results[0].names
    best_metric = 0.0
    best: tuple[float, float, float, float] | None = None
    use_area = roi.is_full_frame()

    for box in results[0].boxes:
        if names.get(int(box.cls[0]), "") != "person":
            continue
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        nx1, ny1, nx2, ny2 = bx1 / w, by1 / h, bx2 / w, by2 / h
        ix1 = max(roi.x1, nx1)
        iy1 = max(roi.y1, ny1)
        ix2 = min(roi.x2, nx2)
        iy2 = min(roi.y2, ny2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        metric = (nx2 - nx1) * (ny2 - ny1) if use_area else (ix2 - ix1) * (iy2 - iy1)
        if metric > best_metric:
            best_metric = metric
            best = (nx1, ny1, nx2, ny2)
    return best


def count_yolo_persons(results: Any) -> int:
    if results is None or len(results) == 0:
        return 0
    names = results[0].names
    return sum(
        1 for box in results[0].boxes if names.get(int(box.cls[0]), "") == "person"
    )


def score_frame_yolo_person(
    fs: FrameScore,
    frame: Any,
    results: Any,
    roi: BenchROI,
) -> bool:
    """YOLO-only person/hands/activity scores (no MediaPipe pose)."""
    if not apply_yolo_person_fallback(fs, frame, results, roi):
        return False
    fs.active_pose = min(1.0, 0.45 + 0.35 * fs.hands_in_roi)
    return True


def merge_yolo_detection_lists(
    person_dets: list[YoloDetection],
    full_dets: list[YoloDetection],
) -> list[YoloDetection]:
    """Keep all person boxes from the person pass; other classes from the full pass."""
    return list(person_dets) + [d for d in full_dets if d.name != "person"]


def merge_phone_detections(
    detections: list[YoloDetection],
    phone_dets: list[YoloDetection],
    *,
    iou_threshold: float = 0.45,
) -> list[YoloDetection]:
    """Add phones from a class-67-only YOLO pass if not already matched on the full pass."""
    return merge_class_detections(
        detections, phone_dets, class_names=PHONE_CLASS_NAMES, iou_threshold=iou_threshold
    )


def merge_class_detections(
    detections: list[YoloDetection],
    extra: list[YoloDetection],
    *,
    class_names: frozenset[str],
    iou_threshold: float = 0.45,
) -> list[YoloDetection]:
    """Add boxes from a class-filtered YOLO pass if not already matched."""
    out = list(detections)
    existing = [d for d in out if d.name in class_names]
    for d in extra:
        if d.name not in class_names:
            continue
        if any(_det_iou(d, e) >= iou_threshold for e in existing):
            continue
        out.append(d)
        existing.append(d)
    return out


def detect_phones_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    """YOLO class 67 (cell phone) on the full frame."""
    return _detect_coco_class_full_frame(
        model,
        frame,
        roi,
        class_id=YOLO_CELL_PHONE_CLASS_ID,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        show_all_coco_tools=show_all_coco_tools,
    )


def detect_laptops_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    """YOLO class 63 (laptop) on the full frame — catches distant/small laptops."""
    return _detect_coco_class_full_frame(
        model,
        frame,
        roi,
        class_id=YOLO_LAPTOP_CLASS_ID,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        show_all_coco_tools=show_all_coco_tools,
    )


def detect_keyboards_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    """YOLO class 66 (keyboard) on the full frame."""
    return _detect_coco_class_full_frame(
        model,
        frame,
        roi,
        class_id=YOLO_KEYBOARD_CLASS_ID,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        show_all_coco_tools=show_all_coco_tools,
    )


def detect_mice_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    """YOLO class 64 (mouse) on the full frame."""
    return _detect_coco_class_full_frame(
        model,
        frame,
        roi,
        class_id=YOLO_MOUSE_CLASS_ID,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        show_all_coco_tools=show_all_coco_tools,
    )


def detect_work_devices_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    class_ids: list[int],
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    """
    Pass 2 — one YOLO inference for phone, laptop, keyboard, mouse (COCO 67/63/66/64).
    """
    if not class_ids:
        return []
    allowed_names: set[str] = set()
    for cid in class_ids:
        allowed_names.update(_CLASS_ID_TO_NAMES.get(cid, ()))
    res = model(
        frame,
        verbose=False,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        classes=class_ids,
    )
    dets = collect_yolo_detections(
        frame, res, roi, wrist_points=[], show_all_coco_tools=show_all_coco_tools
    )
    return [d for d in dets if d.name in allowed_names]


def merge_work_device_detections(
    detections: list[YoloDetection],
    device_dets: list[YoloDetection],
    *,
    iou_threshold: float = 0.45,
) -> list[YoloDetection]:
    """Merge unified devices-pass boxes into the frame detection list."""
    out = list(detections)
    for class_names in (
        PHONE_CLASS_NAMES,
        LAPTOP_CLASS_NAMES,
        KEYBOARD_CLASS_NAMES,
        MOUSE_CLASS_NAMES,
    ):
        out = merge_class_detections(
            out, device_dets, class_names=class_names, iou_threshold=iou_threshold
        )
    return out


def _detect_coco_class_full_frame(
    model: Any,
    frame: Any,
    roi: BenchROI,
    *,
    class_id: int,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
) -> list[YoloDetection]:
    res = model(
        frame,
        verbose=False,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        classes=[class_id],
    )
    return collect_yolo_detections(
        frame, res, roi, wrist_points=[], show_all_coco_tools=show_all_coco_tools
    )


def detect_phones_on_hand_regions(
    model: Any,
    frame: Any,
    roi: BenchROI,
    hand_boxes: list[WristBox],
    *,
    conf: float,
    imgsz: int,
    iou: float,
    show_all_coco_tools: bool,
    pad_frac: float = 0.40,
    min_crop_px: int = 64,
) -> list[YoloDetection]:
    """
    Zoomed phone-only YOLO on the union of primary hand boxes (finds small phones in hand).
    """
    if not hand_boxes:
        return []
    fh, fw = frame.shape[:2]
    x1 = min(wb.x1 for wb in hand_boxes)
    y1 = min(wb.y1 for wb in hand_boxes)
    x2 = max(wb.x2 for wb in hand_boxes)
    y2 = max(wb.y2 for wb in hand_boxes)
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_frac), 32)
    pad_y = max(int(bh * pad_frac), 32)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(fw, x2 + pad_x)
    y2 = min(fh, y2 + pad_y)
    if x2 - x1 < min_crop_px or y2 - y1 < min_crop_px:
        return []
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    crop_imgsz = min(imgsz, max(640, ch, cw))
    res = model(
        crop,
        verbose=False,
        conf=conf,
        imgsz=crop_imgsz,
        iou=iou,
        classes=[YOLO_CELL_PHONE_CLASS_ID],
    )
    dets = collect_yolo_detections(
        crop,
        res,
        roi,
        wrist_points=[],
        show_all_coco_tools=show_all_coco_tools,
        full_w=fw,
        full_h=fh,
        offset_x=x1,
        offset_y=y1,
    )
    for d in dets:
        if d.name in PHONE_CLASS_NAMES:
            d.from_hand_crop = True
    return dets


def log_phone_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every cell-phone box at this pipeline stage; return count."""
    return log_class_detections(stage, detections, frame_w, frame_h, class_names=PHONE_CLASS_NAMES, tag="phone")


def log_laptop_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every laptop box at this pipeline stage; return count."""
    return log_class_detections(stage, detections, frame_w, frame_h, class_names=LAPTOP_CLASS_NAMES, tag="laptop")


def log_keyboard_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every keyboard box at this pipeline stage; return count."""
    return log_class_detections(
        stage, detections, frame_w, frame_h, class_names=KEYBOARD_CLASS_NAMES, tag="keyboard"
    )


def log_mouse_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every mouse box at this pipeline stage; return count."""
    return log_class_detections(
        stage, detections, frame_w, frame_h, class_names=MOUSE_CLASS_NAMES, tag="mouse"
    )


def log_class_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
    *,
    class_names: frozenset[str],
    tag: str,
) -> int:
    hits = [d for d in detections if d.name in class_names]
    if not hits:
        print(f"[{tag}] {stage}: none", flush=True)
        return 0
    for i, d in enumerate(hits):
        cx = (d.x1 + d.x2) / 2 / max(1, frame_w)
        cy = (d.y1 + d.y2) / 2 / max(1, frame_h)
        print(
            f"[{tag}] {stage} #{i + 1}: conf={d.confidence:.2f} role={d.role} "
            f"stable={d.stable} streak={d.stable_age_s:.1f}s "
            f"in_person={d.inside_person} overlaps_hand={d.near_hand} "
            f"center=({cx:.2f},{cy:.2f}) box=({d.x1},{d.y1})-({d.x2},{d.y2})",
            flush=True,
        )
    return len(hits)


def _format_detection_line(
    d: YoloDetection, frame_w: int, frame_h: int, *, index: int | None = None
) -> str:
    cx = (d.x1 + d.x2) / 2 / max(1, frame_w)
    cy = (d.y1 + d.y2) / 2 / max(1, frame_h)
    prefix = f"#{index} " if index is not None else ""
    crop = " hand_crop" if d.from_hand_crop else ""
    primary = " PRIMARY" if d.is_primary else ""
    return (
        f"{prefix}{d.name} conf={d.confidence:.2f} role={d.role} stable={d.stable} "
        f"streak={d.stable_age_s:.1f}s in_roi={d.in_roi} near_hand={d.near_hand} "
        f"in_person={d.inside_person}{crop}{primary} "
        f"center=({cx:.2f},{cy:.2f}) box=({d.x1},{d.y1})-({d.x2},{d.y2})"
    )


def log_all_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every YOLO box at this pipeline stage."""
    print(f"[det] {stage}: {len(detections)} box(es)", flush=True)
    if not detections:
        return 0
    for i, d in enumerate(
        sorted(detections, key=lambda x: (-x.confidence, x.name, x.x1, x.y1))
    ):
        print(f"[det] {stage} {_format_detection_line(d, frame_w, frame_h, index=i + 1)}", flush=True)
    return len(detections)


def log_person_detections(
    stage: str,
    detections: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> int:
    """Print every person box at this pipeline stage."""
    persons = [d for d in detections if d.name == "person"]
    print(f"[person] {stage}: {len(persons)} person(s)", flush=True)
    if not persons:
        return 0
    for i, d in enumerate(
        sorted(persons, key=lambda x: (-x.confidence, -int(x.is_primary)))
    ):
        print(
            f"[person] {stage} {_format_detection_line(d, frame_w, frame_h, index=i + 1)}",
            flush=True,
        )
    return len(persons)


def log_wrist_and_hands(
    wrist_boxes: list[WristBox],
    wrist_points_norm: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> None:
    """Print stable wrist points and hand region boxes."""
    print(f"[hands] wrist points (norm): {len(wrist_points_norm)}", flush=True)
    for i, (nx, ny) in enumerate(wrist_points_norm):
        print(f"[hands] wrist #{i + 1}: ({nx:.3f},{ny:.3f})", flush=True)
    print(f"[hands] hand boxes: {len(wrist_boxes)}", flush=True)
    for wb in wrist_boxes:
        cx = (wb.x1 + wb.x2) / 2 / max(1, frame_w)
        cy = (wb.y1 + wb.y2) / 2 / max(1, frame_h)
        prim = " primary" if wb.is_primary_person else ""
        print(
            f"[hands] {wb.label}{prim}: box=({wb.x1},{wb.y1})-({wb.x2},{wb.y2}) "
            f"center=({cx:.2f},{cy:.2f})",
            flush=True,
        )


def log_hand_overlaps(tag: str, overlaps: list[HandObjectOverlap]) -> None:
    """Print hand×object overlap metrics for one device type."""
    print(f"[overlap] {tag}: {len(overlaps)} pair(s)", flush=True)
    if not overlaps:
        return
    for i, o in enumerate(overlaps):
        print(
            f"[overlap] {tag} #{i + 1}: hand={o.wrist_label} obj={o.object_name} "
            f"conf={o.object_conf:.2f} IoU={o.iou_percent:.1f}% "
            f"hand_cov={o.hand_coverage_percent:.1f}% obj_cov={o.object_coverage_percent:.1f}% "
            f"box=({o.object_x1},{o.object_y1})",
            flush=True,
        )


def log_frame_scores(
    fs: FrameScore,
    *,
    state: str,
    n_persons: int,
    n_poses: int,
    pose_fallback: bool,
    near_hands_label: str,
) -> None:
    """Print activity scores and flags for this frame."""
    print(
        f"[score] state={state} persons={n_persons} poses={n_poses} "
        f"pose_yolo_fallback={pose_fallback}",
        flush=True,
    )
    print(
        f"[score] person_in_roi={fs.person_in_roi:.2f} hands_in_roi={fs.hands_in_roi:.2f} "
        f"active_pose={fs.active_pose:.2f} person_visible={fs.person_visible}",
        flush=True,
    )
    print(
        f"[score] on_phone={fs.on_phone} phone_near_hand={fs.phone_near_hand} "
        f"on_laptop={fs.on_laptop} on_keyboard={fs.on_keyboard} "
        f"on_mouse={fs.on_mouse} at_computer_work={fs.at_computer_work}",
        flush=True,
    )
    print(
        f"[score] IoU% phone={fs.best_phone_wrist_iou_pct:.0f} laptop={fs.best_laptop_hand_iou_pct:.0f} "
        f"keyboard={fs.best_keyboard_hand_iou_pct:.0f} mouse={fs.best_mouse_hand_iou_pct:.0f}",
        flush=True,
    )
    print(
        f"[score] M={fs.medium_score:.2f} S={fs.strict_score:.2f} "
        f"tools={fs.tool_named_hits}/{fs.tool_named_checks} named_tool={fs.named_tool_score:.2f}",
        flush=True,
    )
    print(f"[score] near_hands: {near_hands_label}", flush=True)


def _det_iou(a: YoloDetection, b: YoloDetection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / (area_a + area_b - inter)


def dedupe_person_poses(
    poses: list["PersonPose"],
    *,
    iomin_threshold: float = 0.5,
    kp_conf_min: float = 0.15,
) -> list["PersonPose"]:
    """Collapse pose detections that fall on the same subject.

    YOLO-pose sometimes emits two skeletons for the same baby — keypoints
    jitter between them frame-to-frame, which PoseStateTracker reads as
    motion and prevents ``still_seconds`` from advancing. As a result the
    activity classifier stays in ``sitting`` while the baby is actually
    sleeping.

    Strategy: compute each pose's bbox from its high-confidence
    keypoints, sort by summed keypoint confidence (descending), and drop
    any pose whose bbox is heavily inside an already-kept one
    (intersection / min-area >= ``iomin_threshold``). The surviving
    skeleton drives the motion calc cleanly.

    If a pose has fewer than 3 high-confidence keypoints we can't form a
    reliable bbox — it's kept as-is (the rare case where this matters,
    erring on the side of preserving signal).
    """
    if len(poses) <= 1:
        return list(poses)

    def _bbox_and_score(pose: "PersonPose"):
        if pose.keypoints_xy is None or pose.keypoints_conf is None:
            return None, 0.0
        n = min(len(pose.keypoints_xy), len(pose.keypoints_conf))
        xs: list[float] = []
        ys: list[float] = []
        score = 0.0
        for i in range(n):
            c = float(pose.keypoints_conf[i])
            if c < kp_conf_min:
                continue
            xs.append(float(pose.keypoints_xy[i][0]))
            ys.append(float(pose.keypoints_xy[i][1]))
            score += c
        if len(xs) < 3:
            return None, score
        return (min(xs), min(ys), max(xs), max(ys)), score

    def _bbox_iomin(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / min(area_a, area_b)

    enriched = [(p, *_bbox_and_score(p)) for p in poses]
    # Higher score first; bbox-less poses sink to the end (kept regardless).
    enriched.sort(key=lambda t: (0 if t[1] is not None else 1, -t[2]))

    kept: list["PersonPose"] = []
    kept_bboxes: list = []
    for pose, bbox, _score in enriched:
        if bbox is None:
            kept.append(pose)
            continue
        if any(_bbox_iomin(bbox, kb) >= iomin_threshold for kb in kept_bboxes):
            continue
        kept.append(pose)
        kept_bboxes.append(bbox)
    return kept


def _det_iomin(a: YoloDetection, b: YoloDetection) -> float:
    """Intersection-over-min-area. 1.0 means one box is entirely inside
    the other (regardless of how much bigger the outer one is). Catches
    the 'head box inside whole-body box' case that IoU misses."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / min(area_a, area_b)


def dedupe_detections(
    detections: list[YoloDetection],
    *,
    iou_threshold: float = 0.45,
    containment_threshold: float | None = None,
) -> list[YoloDetection]:
    """Per-class greedy NMS — removes duplicate boxes on the same person.

    When ``containment_threshold`` is set (e.g. 0.70), two boxes are also
    treated as duplicates if the smaller one is mostly inside the larger
    (IoMin > threshold). This catches YOLO emitting both a whole-body
    box and a head-only box for the same baby at confidences too far
    apart for plain IoU NMS to merge them.
    """
    by_name: dict[str, list[YoloDetection]] = {}
    for d in detections:
        by_name.setdefault(d.name, []).append(d)

    out: list[YoloDetection] = []
    for group in by_name.values():
        group = sorted(group, key=lambda d: -d.confidence)
        kept: list[YoloDetection] = []
        for d in group:
            is_dup = False
            for k in kept:
                if _det_iou(d, k) >= iou_threshold:
                    is_dup = True
                    break
                if (
                    containment_threshold is not None
                    and _det_iomin(d, k) >= containment_threshold
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
        out.extend(kept)
    return out


def filter_plausible_person_boxes(
    persons: list[YoloDetection],
    frame_w: int,
    frame_h: int,
) -> list[YoloDetection]:
    """Drop chair/armrest-sized flat boxes; keep upright person-shaped detections."""
    kept: list[YoloDetection] = []
    frame_area = max(1, frame_w * frame_h)
    for d in persons:
        bw = d.x2 - d.x1
        bh = d.y2 - d.y1
        if bh < 40 or bw < 20:
            continue
        if bh / max(bw, 1) < 0.75:
            continue
        if (bw * bh) / frame_area < 0.002:
            continue
        kept.append(d)
    return kept


def select_primary_person(
    persons: list[YoloDetection],
    roi: BenchROI,
    frame_w: int,
    frame_h: int,
) -> YoloDetection | None:
    """One person for scoring: largest in ROI, prefer higher confidence on ties."""
    best: YoloDetection | None = None
    best_metric = 0.0
    for d in persons:
        nx1, ny1 = d.x1 / frame_w, d.y1 / frame_h
        nx2, ny2 = d.x2 / frame_w, d.y2 / frame_h
        ix1 = max(roi.x1, nx1)
        iy1 = max(roi.y1, ny1)
        ix2 = min(roi.x2, nx2)
        iy2 = min(roi.y2, ny2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        metric = (ix2 - ix1) * (iy2 - iy1)
        if roi.is_full_frame():
            metric = (nx2 - nx1) * (ny2 - ny1)
        metric *= 0.5 + d.confidence
        if metric > best_metric:
            best_metric = metric
            best = d
    return best


def wrists_from_pose_landmarks(landmarks: Any, *, vis_min: float = 0.35) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for idx in (15, 16):
        p = landmarks.landmark[idx]
        if p.visibility >= vis_min:
            pts.append((p.x, p.y))
    return pts


def _clamp_box(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return x1, y1, x2, y2


def _box_intersection_area(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> int:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def overlap_between_boxes(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> tuple[float, float, float]:
    """
    Return (iou_percent, coverage_of_a_percent, coverage_of_b_percent).
    coverage_of_a = intersection / area(a) × 100.
    """
    inter = _box_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    if inter <= 0:
        return 0.0, 0.0, 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    iou = 100.0 * inter / max(1, union)
    cov_a = 100.0 * inter / area_a
    cov_b = 100.0 * inter / area_b
    return iou, cov_a, cov_b


def _fallback_hand_half_px(
    frame_w: int,
    frame_h: int,
    *,
    person: YoloDetection | None = None,
    half_frac: float = 0.022,
    min_half: int = 14,
) -> int:
    if person is not None:
        bh = person.y2 - person.y1
        return max(min_half, int(bh * 0.045))
    return max(min_half, int(half_frac * min(frame_w, frame_h)))


def _hand_center_and_half(
    wrist_xy: tuple[float, float],
    elbow_xy: tuple[float, float] | None,
    frame_w: int,
    frame_h: int,
    *,
    person: YoloDetection | None = None,
    half_frac: float = 0.022,
    forearm_scale: float = 0.38,
    hand_extend_frac: float = 0.22,
    min_half: int = 14,
) -> tuple[float, float, int]:
    """
    Hand-focused box: smaller than forearm, center shifted from wrist toward fingers.
    Direction = elbow → wrist, extended past the wrist.
    """
    wx, wy = wrist_xy
    if elbow_xy is not None:
        ex, ey = elbow_xy
        dx, dy = wx - ex, wy - ey
        forearm = math.hypot(dx, dy)
        if forearm > 8:
            ux, uy = dx / forearm, dy / forearm
            cx = wx + ux * forearm * hand_extend_frac
            cy = wy + uy * forearm * hand_extend_frac
            half = max(min_half, int(forearm * forearm_scale))
            return cx, cy, half
    half = _fallback_hand_half_px(
        frame_w, frame_h, person=person, half_frac=half_frac, min_half=min_half
    )
    return wx, wy, half


def _person_box_for_pose(
    pose: PersonPose,
    persons: list[YoloDetection],
    primary_person: YoloDetection | None,
) -> YoloDetection | None:
    if pose.is_primary and primary_person is not None:
        return primary_person
    if pose.keypoints_xy is None or pose.keypoints_conf is None:
        return None
    xy, kc = pose.keypoints_xy, pose.keypoints_conf
    pts: list[tuple[float, float]] = []
    for i in (5, 6, 11, 12):
        if kc[i] >= 0.25:
            pts.append((float(xy[i][0]), float(xy[i][1])))
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    for p in persons:
        if p.x1 <= cx <= p.x2 and p.y1 <= cy <= p.y2:
            return p
    return None


def _hand_boxes_from_pose(
    pose: PersonPose,
    person_index: int,
    frame_w: int,
    frame_h: int,
    *,
    person: YoloDetection | None = None,
    half_frac: float = 0.022,
    forearm_scale: float = 0.38,
    hand_extend_frac: float = 0.22,
    kpt_conf_min: float = 0.25,
    multi_person: bool = False,
) -> list[WristBox]:
    boxes: list[WristBox] = []
    prefix = f"P{person_index + 1}_" if multi_person else ""

    if pose.keypoints_xy is not None and pose.keypoints_conf is not None:
        xy = pose.keypoints_xy
        kc = pose.keypoints_conf
        for side, wrist_i, elbow_i in (("left", 9, 7), ("right", 10, 8)):
            if kc[wrist_i] < kpt_conf_min:
                continue
            wx, wy = float(xy[wrist_i][0]), float(xy[wrist_i][1])
            elbow_xy = None
            if kc[elbow_i] >= kpt_conf_min:
                elbow_xy = (float(xy[elbow_i][0]), float(xy[elbow_i][1]))
            cx, cy, half = _hand_center_and_half(
                (wx, wy),
                elbow_xy,
                frame_w,
                frame_h,
                person=person,
                half_frac=half_frac,
                forearm_scale=forearm_scale,
                hand_extend_frac=hand_extend_frac,
            )
            x1, y1, x2, y2 = _clamp_box(
                int(cx - half),
                int(cy - half),
                int(cx + half),
                int(cy + half),
                frame_w,
                frame_h,
            )
            boxes.append(
                WristBox(
                    label=f"{prefix}{side}",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    person_index=person_index,
                    is_primary_person=pose.is_primary,
                )
            )
        return boxes

    if pose.landmarks is not None:
        lm = pose.landmarks
        for side, idx in (("left", 15), ("right", 16)):
            p = lm.landmark[idx]
            if p.visibility < kpt_conf_min:
                continue
            wx, wy = p.x * frame_w, p.y * frame_h
            cx, cy, half = _hand_center_and_half(
                (wx, wy),
                None,
                frame_w,
                frame_h,
                person=person,
                half_frac=half_frac,
                forearm_scale=forearm_scale,
                hand_extend_frac=hand_extend_frac,
            )
            x1, y1, x2, y2 = _clamp_box(
                int(cx - half),
                int(cy - half),
                int(cx + half),
                int(cy + half),
                frame_w,
                frame_h,
            )
            boxes.append(
                WristBox(
                    label=f"{prefix}{side}",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    person_index=person_index,
                    is_primary_person=pose.is_primary,
                )
            )
    return boxes


def build_wrist_boxes(
    frame_w: int,
    frame_h: int,
    *,
    person_poses: list[PersonPose] | None = None,
    primary_pose: PersonPose | None = None,
    person_detections: list[YoloDetection] | None = None,
    wrist_points_norm: list[tuple[float, float]] | None = None,
    primary_person: YoloDetection | None = None,
    half_frac: float = 0.022,
    forearm_scale: float = 0.38,
    hand_extend_frac: float = 0.22,
    kpt_conf_min: float = 0.25,
) -> list[WristBox]:
    """Hand boxes for every person with pose; fallback to norm wrist points if no pose."""
    persons = person_detections or []
    poses = list(person_poses or [])
    if not poses and primary_pose is not None:
        poses = [primary_pose]

    boxes: list[WristBox] = []
    multi = len(poses) > 1
    for idx, pp in enumerate(poses):
        person = _person_box_for_pose(pp, persons, primary_person)
        boxes.extend(
            _hand_boxes_from_pose(
                pp,
                idx,
                frame_w,
                frame_h,
                person=person,
                half_frac=half_frac,
                forearm_scale=forearm_scale,
                hand_extend_frac=hand_extend_frac,
                kpt_conf_min=kpt_conf_min,
                multi_person=multi,
            )
        )

    if boxes:
        return boxes

    if wrist_points_norm:
        labels = ("left", "right") if len(wrist_points_norm) >= 2 else ("hand",)
        for i, (nx, ny) in enumerate(wrist_points_norm):
            side = labels[i] if i < len(labels) else f"hand{i}"
            wx, wy = nx * frame_w, ny * frame_h
            cx, cy, half = _hand_center_and_half(
                (wx, wy),
                None,
                frame_w,
                frame_h,
                person=primary_person,
                half_frac=half_frac,
                forearm_scale=forearm_scale,
                hand_extend_frac=hand_extend_frac,
            )
            x1, y1, x2, y2 = _clamp_box(
                int(cx - half),
                int(cy - half),
                int(cx + half),
                int(cy + half),
                frame_w,
                frame_h,
            )
            boxes.append(
                WristBox(
                    label=side,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    person_index=0,
                    is_primary_person=True,
                )
            )
    return boxes


def hand_box_centers_norm(
    wrist_boxes: list[WristBox], frame_w: int, frame_h: int
) -> list[tuple[float, float]]:
    fw, fh = max(1, frame_w), max(1, frame_h)
    return [((wb.x1 + wb.x2) / 2 / fw, (wb.y1 + wb.y2) / 2 / fh) for wb in wrist_boxes]


def object_overlaps_hand_boxes(
    det: YoloDetection,
    hand_boxes: list[WristBox],
    *,
    min_iou_percent: float = 8.0,
    min_hand_coverage_percent: float = 12.0,
    min_object_coverage_percent: float = 15.0,
) -> bool:
    """
    True only when the object box intersects a hand region (no distance-to-point fallback).
    """
    if not hand_boxes:
        return False
    ocx = (det.x1 + det.x2) // 2
    ocy = (det.y1 + det.y2) // 2
    for wb in hand_boxes:
        if wb.x1 <= ocx <= wb.x2 and wb.y1 <= ocy <= wb.y2:
            return True
        iou, hand_cov, obj_cov = overlap_between_boxes(
            wb.x1, wb.y1, wb.x2, wb.y2, det.x1, det.y1, det.x2, det.y2
        )
        if (
            iou >= min_iou_percent
            or hand_cov >= min_hand_coverage_percent
            or obj_cov >= min_object_coverage_percent
        ):
            return True
    return False


def phone_overlaps_hand_boxes(
    det: YoloDetection,
    hand_boxes: list[WristBox],
    *,
    min_iou_percent: float = 8.0,
    min_wrist_coverage_percent: float = 12.0,
    min_phone_coverage_percent: float = 15.0,
) -> bool:
    return object_overlaps_hand_boxes(
        det,
        hand_boxes,
        min_iou_percent=min_iou_percent,
        min_hand_coverage_percent=min_wrist_coverage_percent,
        min_object_coverage_percent=min_phone_coverage_percent,
    )


def primary_hand_boxes(wrist_boxes: list[WristBox]) -> list[WristBox]:
    """Hand regions for the tracked (primary) person only."""
    primary = [wb for wb in wrist_boxes if wb.is_primary_person]
    return primary if primary else list(wrist_boxes)


def phone_associated_with_primary(
    det: YoloDetection,
    primary: YoloDetection | None,
    wrist_boxes: list[WristBox],
    frame_w: int,
    frame_h: int,
) -> bool:
    """Phone counts for primary worker: inside person box or on primary person's hands."""
    if primary is None:
        return True
    if _center_in_box(det, primary, frame_w, frame_h):
        return True
    pcx = (det.x1 + det.x2) // 2
    pcy = (det.y1 + det.y2) // 2
    for wb in wrist_boxes:
        if not wb.is_primary_person:
            continue
        if wb.x1 <= pcx <= wb.x2 and wb.y1 <= pcy <= wb.y2:
            return True
        iou, wrist_cov, phone_cov = overlap_between_boxes(
            wb.x1, wb.y1, wb.x2, wb.y2, det.x1, det.y1, det.x2, det.y2
        )
        if iou > 0 or wrist_cov > 0 or phone_cov > 0:
            return True
    return False


def compute_hand_overlaps(
    hand_boxes: list[WristBox],
    detections: list[YoloDetection],
    class_names: frozenset[str],
    *,
    stable_only: bool = False,
) -> list[HandObjectOverlap]:
    """All hand×object pairs with IoU and coverage percentages."""
    out: list[HandObjectOverlap] = []
    objects = [d for d in detections if d.name in class_names]
    if stable_only:
        objects = [d for d in objects if d.stable]
    for wb in hand_boxes:
        for obj in objects:
            iou, hand_cov, obj_cov = overlap_between_boxes(
                wb.x1, wb.y1, wb.x2, wb.y2, obj.x1, obj.y1, obj.x2, obj.y2
            )
            if iou <= 0 and hand_cov <= 0 and obj_cov <= 0:
                continue
            out.append(
                HandObjectOverlap(
                    wrist_label=wb.label,
                    object_name=obj.name,
                    object_conf=obj.confidence,
                    object_x1=obj.x1,
                    object_y1=obj.y1,
                    iou_percent=iou,
                    hand_coverage_percent=hand_cov,
                    object_coverage_percent=obj_cov,
                )
            )
    return out


def compute_wrist_phone_overlaps(
    wrist_boxes: list[WristBox],
    detections: list[YoloDetection],
    *,
    stable_only: bool = True,
) -> list[HandObjectOverlap]:
    return compute_hand_overlaps(
        wrist_boxes, detections, PHONE_CLASS_NAMES, stable_only=stable_only
    )


def compute_wrist_laptop_overlaps(
    hand_boxes: list[WristBox],
    detections: list[YoloDetection],
    *,
    stable_only: bool = True,
) -> list[HandObjectOverlap]:
    return compute_hand_overlaps(
        hand_boxes, detections, LAPTOP_CLASS_NAMES, stable_only=stable_only
    )


def compute_wrist_keyboard_overlaps(
    hand_boxes: list[WristBox],
    detections: list[YoloDetection],
    *,
    stable_only: bool = True,
) -> list[HandObjectOverlap]:
    return compute_hand_overlaps(
        hand_boxes, detections, KEYBOARD_CLASS_NAMES, stable_only=stable_only
    )


def compute_wrist_mouse_overlaps(
    hand_boxes: list[WristBox],
    detections: list[YoloDetection],
    *,
    stable_only: bool = True,
) -> list[HandObjectOverlap]:
    return compute_hand_overlaps(
        hand_boxes, detections, MOUSE_CLASS_NAMES, stable_only=stable_only
    )


def best_hand_overlap_iou(overlaps: list[HandObjectOverlap]) -> float:
    if not overlaps:
        return 0.0
    return max(o.iou_percent for o in overlaps)


def best_wrist_phone_iou(overlaps: list[HandObjectOverlap]) -> float:
    return best_hand_overlap_iou(overlaps)


def _landmarks_crop_to_frame(
    crop_landmarks: Any,
    x1: int,
    y1: int,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> Any:
    """Map pose landmarks from a person crop back to full-frame normalized coords."""
    from mediapipe.framework.formats import landmark_pb2

    out = landmark_pb2.NormalizedLandmarkList()
    for p in crop_landmarks.landmark:
        lm = out.landmark.add()
        lm.x = (p.x * crop_w + x1) / frame_w
        lm.y = (p.y * crop_h + y1) / frame_h
        lm.z = p.z
        lm.visibility = p.visibility
    return out


def estimate_poses_for_persons(
    frame_bgr: Any,
    persons: list[YoloDetection],
    pose: Any,
    *,
    pad_frac: float = 0.25,
    max_persons: int = 8,
    min_crop_px: int = 48,
) -> list[PersonPose]:
    """
    Run MediaPipe pose on each YOLO person crop (MediaPipe full-frame = 1 person only).
    Largest boxes first; primary person flagged for scoring / wrist dots.
    """
    fh, fw = frame_bgr.shape[:2]
    boxes = sorted(
        [p for p in persons if p.name == "person"],
        key=lambda p: -((p.x2 - p.x1) * (p.y2 - p.y1)),
    )[:max_persons]
    poses: list[PersonPose] = []
    for person in boxes:
        bw = person.x2 - person.x1
        bh = person.y2 - person.y1
        pad_x = int(bw * pad_frac)
        pad_y = int(bh * pad_frac)
        x1 = max(0, person.x1 - pad_x)
        y1 = max(0, person.y1 - pad_y)
        x2 = min(fw, person.x2 + pad_x)
        y2 = min(fh, person.y2 + pad_y)
        cw, ch = x2 - x1, y2 - y1
        if cw < min_crop_px or ch < min_crop_px:
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks is None:
            continue
        frame_lms = _landmarks_crop_to_frame(
            result.pose_landmarks, x1, y1, cw, ch, fw, fh
        )
        poses.append(PersonPose(landmarks=frame_lms, is_primary=person.is_primary))
    return poses


def _person_det_iou(a: YoloDetection, b: YoloDetection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / max(1.0, area_a + area_b - inter)


def coco_keypoints_to_mediapipe_landmarks(
    xy_px: Any,
    kpt_conf: Any,
    frame_w: int,
    frame_h: int,
    *,
    vis_min: float = 0.25,
) -> Any:
    """Convert YOLO11 COCO keypoints to MediaPipe-style landmarks for score_pose()."""
    from mediapipe.framework.formats import landmark_pb2

    out = landmark_pb2.NormalizedLandmarkList()
    for _ in range(33):
        out.landmark.add()
    fw, fh = max(1, frame_w), max(1, frame_h)
    for coco_idx, mp_idx in _COCO_TO_MP.items():
        x, y = float(xy_px[coco_idx][0]), float(xy_px[coco_idx][1])
        c = float(kpt_conf[coco_idx])
        lm = out.landmark[mp_idx]
        lm.x = x / fw
        lm.y = y / fh
        lm.z = 0.0
        lm.visibility = c if c >= vis_min else 0.0
    return out


def collect_yolo_pose_persons(
    pose_results: Any,
    frame_w: int,
    frame_h: int,
    primary: YoloDetection | None,
    *,
    box_conf: float = 0.25,
    kpt_conf_min: float = 0.25,
    max_persons: int = 8,
    roi: BenchROI | None = None,
) -> list[PersonPose]:
    """Parse YOLO11-pose results into PersonPose list (multi-person skeletons).

    When `roi` is set and not full-frame, skeletons whose bounding-box
    center falls outside the ROI are dropped (so the activity classifier
    doesn't lock onto a person standing next to the crib).
    """
    if not pose_results:
        return []
    r = pose_results[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes
    kpts = r.keypoints
    ranked: list[tuple[float, int]] = []
    roi_active = roi is not None and not roi.is_full_frame()
    for i in range(len(boxes)):
        if int(boxes.cls[i]) != 0:
            continue
        if float(boxes.conf[i]) < box_conf:
            continue
        xyxy = boxes.xyxy[i].cpu().numpy()
        if roi_active:
            cx = float((xyxy[0] + xyxy[2]) / 2.0 / max(1, frame_w))
            cy = float((xyxy[1] + xyxy[3]) / 2.0 / max(1, frame_h))
            if not roi.contains(cx, cy):  # type: ignore[union-attr]
                continue
        area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
        ranked.append((area, i))
    ranked.sort(key=lambda t: -t[0])
    ranked = ranked[:max_persons]

    poses: list[PersonPose] = []
    for _area, i in ranked:
        data = kpts.data[i].cpu().numpy()
        xy = data[:, :2]
        kconf = data[:, 2]
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].cpu().numpy())
        cand = YoloDetection(
            name="person",
            confidence=float(boxes.conf[i]),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            in_roi=True,
            near_hand=False,
            role="person",
        )
        is_primary = primary is not None and _person_det_iou(cand, primary) >= 0.45
        landmarks = coco_keypoints_to_mediapipe_landmarks(
            xy, kconf, frame_w, frame_h, vis_min=kpt_conf_min
        )
        poses.append(
            PersonPose(
                landmarks=landmarks,
                is_primary=is_primary,
                keypoints_xy=xy,
                keypoints_conf=kconf,
            )
        )

    if primary is not None and poses and not any(p.is_primary for p in poses):
        best_iou = 0.0
        best_j = 0
        for j, (_area, idx) in enumerate(ranked):
            xyxy = boxes.xyxy[idx].cpu().numpy()
            cand = YoloDetection(
                "person",
                1.0,
                int(xyxy[0]),
                int(xyxy[1]),
                int(xyxy[2]),
                int(xyxy[3]),
                True,
                False,
                "person",
            )
            iou = _person_det_iou(cand, primary)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= 0.3:
            for j, p in enumerate(poses):
                p.is_primary = j == best_j

    return poses


def estimate_pose_full_frame(frame_bgr: Any, pose: Any) -> PersonPose | None:
    """Fallback when no YOLO person boxes: single full-frame MediaPipe pass."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks is None:
        return None
    return PersonPose(landmarks=result.pose_landmarks, is_primary=True)


def merge_pose_into_score(fs: FrameScore, pose_fs: FrameScore) -> None:
    """Blend pose into scores already set from YOLO (take the stronger signal per field)."""
    fs.person_visible = fs.person_visible or pose_fs.person_visible
    fs.person_in_roi = max(fs.person_in_roi, pose_fs.person_in_roi)
    fs.hands_in_roi = max(fs.hands_in_roi, pose_fs.hands_in_roi)
    fs.active_pose = max(fs.active_pose, pose_fs.active_pose)


def apply_person_box_to_score(
    fs: FrameScore,
    person: YoloDetection,
    roi: BenchROI,
    frame_w: int,
    frame_h: int,
) -> None:
    """Score from the same box we draw (avoids mismatch with raw YOLO picks)."""
    nx1, ny1 = person.x1 / frame_w, person.y1 / frame_h
    nx2, ny2 = person.x2 / frame_w, person.y2 / frame_h
    cx = (nx1 + nx2) / 2
    cy = (ny1 + ny2) / 2
    fs.person_visible = True
    fs.person_in_roi = 1.0 if roi.contains(cx, cy) else 0.6
    hand_y = ny1 + 0.75 * (ny2 - ny1)
    if roi.contains(cx, hand_y) or roi.contains(cx, ny2):
        fs.hands_in_roi = max(fs.hands_in_roi, 0.7)
    elif roi.contains(cx, cy):
        fs.hands_in_roi = max(fs.hands_in_roi, 0.4)
    fs.active_pose = min(1.0, 0.45 + 0.35 * fs.hands_in_roi)


def consolidate_person_detections(
    detections: list[YoloDetection],
    roi: BenchROI,
    frame_w: int,
    frame_h: int,
    *,
    iou_threshold: float = 0.45,
) -> tuple[list[YoloDetection], YoloDetection | None]:
    """
    Dedupe person boxes; return **all** people for display.
    Largest in ROI is ``is_primary`` (phone / WORKING score).
    """
    persons = [d for d in detections if d.name == "person"]
    other = [d for d in detections if d.name != "person"]
    # Containment threshold catches "head box inside whole-body box" —
    # YOLO sometimes emits both for the same baby at a low confidence
    # spread, and pure-IoU NMS lets them both through (small box's area
    # is too low for its overlap with the big box to clear IoU 0.45).
    persons = dedupe_detections(
        persons,
        iou_threshold=iou_threshold,
        containment_threshold=0.70,
    )
    persons = filter_plausible_person_boxes(persons, frame_w, frame_h)
    persons = dedupe_detections(
        persons,
        iou_threshold=iou_threshold,
        containment_threshold=0.70,
    )
    primary = select_primary_person(persons, roi, frame_w, frame_h)

    marked: list[YoloDetection] = []
    for d in persons:
        is_primary = primary is not None and (
            d.x1 == primary.x1
            and d.y1 == primary.y1
            and d.x2 == primary.x2
            and d.y2 == primary.y2
        )
        marked.append(
            YoloDetection(
                name=d.name,
                confidence=d.confidence,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                in_roi=d.in_roi,
                near_hand=d.near_hand,
                role=d.role,
                inside_person=d.inside_person,
                from_hand_crop=d.from_hand_crop,
                is_primary=is_primary,
            )
        )
    primary_out = next((p for p in marked if p.is_primary), None)
    return other + marked, primary_out


def count_person_detections(detections: list[YoloDetection]) -> int:
    return sum(1 for d in detections if d.name == "person")


def synthesize_person_from_pose(
    pose: "PersonPose",
    frame_w: int,
    frame_h: int,
    *,
    kp_conf_min: float = 0.45,
    min_kps: int = 5,
    min_avg_conf: float = 0.40,
    require_upper_and_lower: bool = True,
    pad_frac: float = 0.10,
) -> "YoloDetection | None":
    """Build a synthetic person YoloDetection from pose keypoints.

    Used when the YOLO pose model returns a skeleton but the YOLO person
    detector missed the bounding box — without this, person_count and
    downstream scoring stay zero while the skeleton is visible on screen.

    Anti-hallucination guards (added after observing the pose model
    materialise "skeletons" on hairbrushes, feeding bottles, and folded
    blankets which then drove false BabyTracker locks at conf=0.50):

      * ``kp_conf_min``           per-keypoint confidence floor (raised)
      * ``min_kps``                minimum qualifying keypoints (raised)
      * ``min_avg_conf``           NEW — average over qualifying keypoints
      * ``require_upper_and_lower`` NEW — at least one COCO keypoint in
                                  [0..10] (head/torso/arms) AND one in
                                  [11..16] (hips/legs), so a localised
                                  noise cluster on one object can't pass

    Returned ``confidence`` is the average keypoint confidence (not a
    fixed 0.50 anymore), so downstream filters and the BabyTracker's
    stale-lock guard can see how strong the underlying evidence is.
    """
    if pose.keypoints_xy is None or pose.keypoints_conf is None:
        return None
    n = min(len(pose.keypoints_conf), len(pose.keypoints_xy))

    qualifying: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    conf_sum = 0.0
    for i in range(n):
        c = float(pose.keypoints_conf[i])
        if c < kp_conf_min:
            continue
        qualifying.append(i)
        xs.append(float(pose.keypoints_xy[i][0]))
        ys.append(float(pose.keypoints_xy[i][1]))
        conf_sum += c

    if len(qualifying) < min_kps:
        return None

    avg_conf = conf_sum / len(qualifying)
    if avg_conf < min_avg_conf:
        return None

    if require_upper_and_lower:
        # COCO keypoint indices: 0..10 are head/torso/arms (upper),
        # 11..16 are hips/knees/ankles (lower). A real body shows at
        # least one of each; a localised noise cluster on a hairbrush
        # or bottle doesn't.
        has_upper = any(i <= 10 for i in qualifying)
        has_lower = any(i >= 11 for i in qualifying)
        if not (has_upper and has_lower):
            return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad_x = box_w * pad_frac
    pad_y = box_h * pad_frac
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None

    return YoloDetection(
        name="person",
        confidence=min(0.95, avg_conf),
        x1=x1, y1=y1, x2=x2, y2=y2,
        in_roi=True,
        near_hand=False,
        role="person",
        inside_person=False,
        is_primary=bool(pose.is_primary),
        stable=False,
        stable_age_s=0.0,
    )


def _center_in_box(
    det: YoloDetection, box: YoloDetection, frame_w: int, frame_h: int
) -> bool:
    cx = (det.x1 + det.x2) / 2
    cy = (det.y1 + det.y2) / 2
    return box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2


def retag_phone_detections(
    detections: list[YoloDetection],
    wrist_boxes: list[WristBox] | None = None,
    *,
    min_conf: float = 0.25,
    min_iou_percent: float = 8.0,
    min_wrist_coverage_percent: float = 12.0,
    min_phone_coverage_percent: float = 15.0,
) -> None:
    """
    Tag phones for overlay/scoring. ``phone`` = in-hand only when the phone box
    overlaps a primary-person hand region (no distance-to-point fallback).
    """
    hand_boxes = primary_hand_boxes(wrist_boxes or [])
    for d in detections:
        if d.name not in PHONE_CLASS_NAMES:
            continue
        overlaps = object_overlaps_hand_boxes(
            d,
            hand_boxes,
            min_iou_percent=min_iou_percent,
            min_hand_coverage_percent=min_wrist_coverage_percent,
            min_object_coverage_percent=min_phone_coverage_percent,
        )
        d.near_hand = overlaps
        d.role = "phone" if overlaps and d.confidence >= min_conf else "other"


def retag_laptop_detections(
    detections: list[YoloDetection],
    wrist_boxes: list[WristBox] | None = None,
    *,
    min_conf: float = 0.20,
    min_iou_percent: float = 10.0,
    min_hand_coverage_percent: float = 15.0,
    min_laptop_coverage_percent: float = 12.0,
) -> None:
    """
    Tag laptops: ``laptop_hands`` when overlapping primary hand boxes; else tool_ok/desk.
    """
    hand_boxes = primary_hand_boxes(wrist_boxes or [])
    for d in detections:
        if d.name not in LAPTOP_CLASS_NAMES:
            continue
        overlaps = object_overlaps_hand_boxes(
            d,
            hand_boxes,
            min_iou_percent=min_iou_percent,
            min_hand_coverage_percent=min_hand_coverage_percent,
            min_object_coverage_percent=min_laptop_coverage_percent,
        )
        d.near_hand = overlaps
        if overlaps and d.confidence >= min_conf:
            d.role = "laptop_hands"
        elif d.role not in ("tool_ok", "tool_miss"):
            d.role = "tool_ok" if d.in_roi else "tool_miss"


def retag_keyboard_detections(
    detections: list[YoloDetection],
    wrist_boxes: list[WristBox] | None = None,
    *,
    min_conf: float = 0.20,
    min_iou_percent: float = 10.0,
    min_hand_coverage_percent: float = 15.0,
    min_keyboard_coverage_percent: float = 12.0,
) -> None:
    """
    Tag keyboards: ``keyboard_hands`` when overlapping primary hand boxes; else desk tool.
    """
    hand_boxes = primary_hand_boxes(wrist_boxes or [])
    for d in detections:
        if d.name not in KEYBOARD_CLASS_NAMES:
            continue
        overlaps = object_overlaps_hand_boxes(
            d,
            hand_boxes,
            min_iou_percent=min_iou_percent,
            min_hand_coverage_percent=min_hand_coverage_percent,
            min_object_coverage_percent=min_keyboard_coverage_percent,
        )
        d.near_hand = overlaps
        if overlaps and d.confidence >= min_conf:
            d.role = "keyboard_hands"
        elif d.role not in ("tool_ok", "tool_miss"):
            d.role = "tool_ok" if d.in_roi else "tool_miss"


def retag_mouse_detections(
    detections: list[YoloDetection],
    wrist_boxes: list[WristBox] | None = None,
    *,
    min_conf: float = 0.20,
    min_iou_percent: float = 10.0,
    min_hand_coverage_percent: float = 15.0,
    min_mouse_coverage_percent: float = 12.0,
) -> None:
    """Tag mice: ``mouse_hands`` when overlapping primary hand boxes; else desk tool."""
    hand_boxes = primary_hand_boxes(wrist_boxes or [])
    for d in detections:
        if d.name not in MOUSE_CLASS_NAMES:
            continue
        overlaps = object_overlaps_hand_boxes(
            d,
            hand_boxes,
            min_iou_percent=min_iou_percent,
            min_hand_coverage_percent=min_hand_coverage_percent,
            min_object_coverage_percent=min_mouse_coverage_percent,
        )
        d.near_hand = overlaps
        if overlaps and d.confidence >= min_conf:
            d.role = "mouse_hands"
        elif d.role not in ("tool_ok", "tool_miss"):
            d.role = "tool_ok" if d.in_roi else "tool_miss"


def infer_wrists_from_yolo(
    frame: Any,
    results: Any,
    roi: BenchROI,
) -> list[tuple[float, float]]:
    """Wrist proxy from bottom-center of best person box in ROI."""
    best = _best_yolo_person_norm(frame, results, roi)
    if best is None:
        return []
    nx1, ny1, nx2, ny2 = best
    return [((nx1 + nx2) / 2, max(0.0, ny2 - 0.05))]


def apply_yolo_person_fallback(
    fs: FrameScore,
    frame: Any,
    results: Any,
    roi: BenchROI,
) -> bool:
    """When pose fails (e.g. back to camera), use YOLO person box for roi/hands hints."""
    best = _best_yolo_person_norm(frame, results, roi)
    if best is None:
        return False

    nx1, ny1, nx2, ny2 = best
    cx = (nx1 + nx2) / 2
    cy = (ny1 + ny2) / 2
    fs.person_visible = True
    fs.person_in_roi = max(fs.person_in_roi, 1.0 if roi.contains(cx, cy) else 0.6)
    hand_y = ny1 + 0.75 * (ny2 - ny1)
    if roi.contains(cx, hand_y) or roi.contains(cx, ny2):
        fs.hands_in_roi = max(fs.hands_in_roi, 0.7)
    elif roi.contains(cx, cy):
        fs.hands_in_roi = max(fs.hands_in_roi, 0.4)
    return True


def apply_wrists_hands_score(
    fs: FrameScore,
    wrist_points: list[tuple[float, float]],
    roi: BenchROI,
) -> None:
    """Stabilize hands score from held/pose/YOLO wrist points (pose wrists often flicker)."""
    if not wrist_points:
        return
    in_hands = sum(1 for wx, wy in wrist_points if roi.contains(wx, wy))
    fs.hands_in_roi = max(fs.hands_in_roi, in_hands / len(wrist_points))


def score_pose(
    landmarks: Any,
    roi: BenchROI,
    *,
    vis_min: float = 0.35,
) -> FrameScore:
    out = FrameScore()
    ls = 11
    rs = 12
    le = 13
    re = 14
    lw = 15
    rw = 16
    lh = 23
    rh = 24

    parts = [_lm(landmarks, i) for i in (ls, rs, le, re, lw, rw, lh, rh)]
    if not parts or max(p[2] for p in parts) < vis_min:
        return out

    out.person_visible = True

    def vis(i: int) -> bool:
        return landmarks.landmark[i].visibility >= vis_min

    shoulders = []
    hips = []
    wrists = []
    for idx in (ls, rs):
        if vis(idx):
            shoulders.append(_lm(landmarks, idx)[:2])
    for idx in (lh, rh):
        if vis(idx):
            hips.append(_lm(landmarks, idx)[:2])
    for idx in (lw, rw):
        if vis(idx):
            wrists.append(_lm(landmarks, idx)[:2])

    if shoulders and hips:
        torso_x = sum(p[0] for p in shoulders + hips) / len(shoulders + hips)
        torso_y = sum(p[1] for p in shoulders + hips) / len(shoulders + hips)
        if roi.contains(torso_x, torso_y):
            out.person_in_roi = 1.0
        else:
            # partial credit if any landmark in roi
            in_count = sum(1 for p in shoulders + hips if roi.contains(p[0], p[1]))
            out.person_in_roi = min(1.0, in_count / max(1, len(shoulders + hips)))

    if wrists:
        in_hands = sum(1 for wx, wy in wrists if roi.contains(wx, wy))
        out.hands_in_roi = in_hands / len(wrists)

    # Active pose: bent elbows and/or hands below shoulders
    active_parts: list[float] = []
    if vis(le) and vis(ls) and vis(lw):
        ang = _angle(_lm(landmarks, ls)[:2], _lm(landmarks, le)[:2], _lm(landmarks, lw)[:2])
        if ang < 130:
            active_parts.append(1.0)
        elif ang < 150:
            active_parts.append(0.5)
    if vis(re) and vis(rs) and vis(rw):
        ang = _angle(_lm(landmarks, rs)[:2], _lm(landmarks, re)[:2], _lm(landmarks, rw)[:2])
        if ang < 130:
            active_parts.append(1.0)
        elif ang < 150:
            active_parts.append(0.5)

    if shoulders and wrists:
        sy = sum(p[1] for p in shoulders) / len(shoulders)
        for wx, wy in wrists:
            if wy > sy + 0.05:  # hands lower than shoulders (image y down)
                active_parts.append(0.7)

    if shoulders and hips and len(shoulders) >= 1 and len(hips) >= 1:
        sx = sum(p[0] for p in shoulders) / len(shoulders)
        sy = sum(p[1] for p in shoulders) / len(shoulders)
        hx = sum(p[0] for p in hips) / len(hips)
        hy = sum(p[1] for p in hips) / len(hips)
        lean = abs(sx - hx) + (hy - sy) * 0.5
        if lean > 0.08:
            active_parts.append(0.6)

    out.active_pose = min(1.0, max(active_parts) if active_parts else 0.0)
    return out


def _box_norm(box: Any, w: int, h: int) -> tuple[float, float, float, float]:
    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
    return bx1 / w, by1 / h, bx2 / w, by2 / h


def _overlaps_roi(nx1: float, ny1: float, nx2: float, ny2: float, roi: BenchROI) -> bool:
    """True if bbox is inside the ROI.

    Polygon ROI: tests the bbox center via point-in-polygon (matches what
    most operators expect — "is the thing in my region?"). This used to
    ignore the polygon entirely and only check the rectangle bounding box,
    so detections outside the polygon were still flagged as in_roi.

    Rectangle ROI: bbox overlap (preserved for backwards compat).
    """
    if roi.is_polygon():
        cx = (nx1 + nx2) / 2.0
        cy = (ny1 + ny2) / 2.0
        return roi.contains(cx, cy)
    ix1 = max(roi.x1, nx1)
    iy1 = max(roi.y1, ny1)
    ix2 = min(roi.x2, nx2)
    iy2 = min(roi.y2, ny2)
    return ix2 > ix1 and iy2 > iy1


def collect_yolo_detections(
    frame: Any,
    results: Any,
    roi: BenchROI,
    *,
    tool_names: frozenset[str] = TOOL_CLASSES,
    show_all_coco_tools: bool = True,
    wrist_points: list[tuple[float, float]] | None = None,
    full_w: int | None = None,
    full_h: int | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[YoloDetection]:
    if results is None or len(results) == 0:
        return []
    h, w = frame.shape[:2]
    fw = full_w or w
    fh = full_h or h
    names = results[0].names
    out: list[YoloDetection] = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = names.get(cls_id, "")
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        nx1, ny1, nx2, ny2 = _box_norm(box, w, h)
        px1 = int(nx1 * w) + offset_x
        py1 = int(ny1 * h) + offset_y
        px2 = int(nx2 * w) + offset_x
        py2 = int(ny2 * h) + offset_y
        cx = (px1 + px2) / 2 / fw
        cy = (py1 + py2) / 2 / fh
        nx1, ny1, nx2, ny2 = px1 / fw, py1 / fh, px2 / fw, py2 / fh
        in_roi = _overlaps_roi(nx1, ny1, nx2, ny2, roi)
        # ROI is a true filter: drop anything outside when an ROI is set
        # (full-frame ROI keeps the previous behaviour of accepting all).
        # This stops false-positive 'person' detections on tables, chair
        # backs, blankets, etc. that fall outside the user-drawn region
        # from polluting the pipeline and the on-screen overlay.
        if not in_roi and not roi.is_full_frame():
            continue
        near_hand = False
        if wrist_points:
            for wx, wy in wrist_points:
                if math.hypot(cx - wx, cy - wy) < 0.22:
                    near_hand = True
                    break

        if name == "person":
            role = "person"
        elif name in PHONE_CLASS_NAMES and near_hand:
            role = "phone"
        elif is_tool_like(name, show_all_coco=show_all_coco_tools) or (
            not show_all_coco_tools and name in tool_names
        ):
            role = "tool_ok" if (in_roi or near_hand) else "tool_miss"
        elif in_roi and name not in BENCH_OBJ_SKIP and not show_all_coco_tools:
            role = "bench_obj"
        else:
            role = "other"

        out.append(
            YoloDetection(
                name=name,
                confidence=conf,
                x1=px1,
                y1=py1,
                x2=px2,
                y2=py2,
                in_roi=in_roi,
                near_hand=near_hand,
                role=role,
            )
        )

    person_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in out if d.name == "person"]
    if person_boxes:
        for d in out:
            if d.name == "person":
                continue
            cx = (d.x1 + d.x2) / 2
            cy = (d.y1 + d.y2) / 2
            for px1, py1, px2, py2 in person_boxes:
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    d.inside_person = True
                    if d.role == "tool_miss" and (d.near_hand or d.in_roi):
                        d.role = "tool_ok"
                    break
    return out


def count_phones_in_frame(detections: list[YoloDetection]) -> int:
    return sum(1 for d in detections if d.name in PHONE_CLASS_NAMES)


@dataclass
class WristHoldover:
    """Keep last wrist position briefly when keypoints drop for a frame."""

    hold_s: float = 2.0
    _pts: list[tuple[float, float]] = field(default_factory=list)
    _t: float = 0.0

    def resolve(self, wrist_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        now = time.monotonic()
        if wrist_points:
            self._pts = list(wrist_points)
            self._t = now
            return wrist_points
        if self._pts and now - self._t <= self.hold_s:
            return self._pts
        return []

    def clear(self) -> None:
        """Drop held wrists (e.g. when no person is in frame)."""
        self._pts = []
        self._t = 0.0


@dataclass
class NearHandsStabilizer:
    """
    Debounce console + on_phone when YOLO returns 0 boxes for one poll
    or pose wrists drop for a frame (common at FRIGATE_HTTP_FPS=2).
    """

    phone_off_frames: int = 3
    empty_frames: int = 2
    _frames_no_phone: int = 0
    _frames_empty: int = 0
    on_phone: bool = False
    last_log: str = ""

    def observe_phone(self, saw_phone: bool) -> None:
        if saw_phone:
            self._frames_no_phone = 0
            self.on_phone = True
        else:
            self._frames_no_phone += 1
            if self.on_phone and self._frames_no_phone >= self.phone_off_frames:
                self.on_phone = False

    def observe_detections(self, n: int) -> None:
        if n <= 0:
            self._frames_empty += 1
        else:
            self._frames_empty = 0

    def status_label(self) -> str:
        """Current near-hands label (for verbose per-frame logs)."""
        if self.on_phone:
            return "phone in hand (hand-box overlap)"
        if self._frames_empty >= self.empty_frames:
            return "nothing detected near hands"
        return "monitoring (objects in frame, no stable phone-in-hand)"

    def log_line(self) -> str | None:
        """Emit a line only when the stable label changes."""
        new = self.status_label()
        if new == "monitoring (objects in frame, no stable phone-in-hand)":
            return None
        if new == self.last_log:
            return None
        self.last_log = new
        return new


@dataclass
class _ObjectTrack:
    """Tracks one phone/laptop box across frames for temporal stability."""

    track_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    streak_start: float | None = None
    last_seen: float = 0.0
    last_conf: float = 0.0

    def update_box(self, d: YoloDetection, now: float) -> None:
        self.x1, self.y1, self.x2, self.y2 = d.x1, d.y1, d.x2, d.y2
        self.last_seen = now
        self.last_conf = d.confidence

    def streak_age(self, now: float) -> float:
        if self.streak_start is None:
            return 0.0
        return max(0.0, now - self.streak_start)


@dataclass
class ObjectDetectionStabilizer:
    """
    Require conf ≥ min_conf on every frame for stable_seconds before a
    phone/laptop counts (overlap, on_phone, on_laptop, overlay in-hand).
    """

    class_names: frozenset[str]
    stable_seconds: float = 3.0
    min_conf: float = 0.5
    match_iou: float = 0.25
    _tracks: dict[int, _ObjectTrack] = field(default_factory=dict)
    _next_id: int = 1

    def apply(self, detections: list[YoloDetection], *, now: float | None = None) -> None:
        t = now if now is not None else time.monotonic()
        candidates = [d for d in detections if d.name in self.class_names]
        for d in candidates:
            d.stable = False
            d.stable_age_s = 0.0

        matched_ids: set[int] = set()
        for d in candidates:
            if d.confidence < self.min_conf:
                continue
            tid, _ = self._match_track(d)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _ObjectTrack(
                    track_id=tid,
                    class_name=d.name,
                    x1=d.x1,
                    y1=d.y1,
                    x2=d.x2,
                    y2=d.y2,
                    streak_start=t,
                    last_seen=t,
                    last_conf=d.confidence,
                )
            else:
                tr = self._tracks[tid]
                if tr.streak_start is None:
                    tr.streak_start = t
                tr.update_box(d, t)
            matched_ids.add(tid)
            tr = self._tracks[tid]
            age = tr.streak_age(t)
            d.stable_age_s = age
            d.stable = age >= self.stable_seconds

        for tid, tr in list(self._tracks.items()):
            if tid in matched_ids:
                continue
            tr.streak_start = None

        self._purge_stale(t, max(1.0, self.stable_seconds * 2))

        for d in candidates:
            if d.stable:
                continue
            if d.role == "phone":
                d.role = "other"
            elif d.role == "laptop_hands":
                d.role = "tool_ok" if d.in_roi else "tool_miss"
            elif d.role == "keyboard_hands":
                d.role = "tool_ok" if d.in_roi else "tool_miss"
            elif d.role == "mouse_hands":
                d.role = "tool_ok" if d.in_roi else "tool_miss"

    def _match_track(self, d: YoloDetection) -> tuple[int | None, float]:
        best_tid: int | None = None
        best_iou = 0.0
        for tid, tr in self._tracks.items():
            if tr.class_name != d.name:
                continue
            iou = _box_iou_pixels(tr.x1, tr.y1, tr.x2, tr.y2, d.x1, d.y1, d.x2, d.y2)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        if best_tid is None or best_iou < self.match_iou:
            return None, 0.0
        return best_tid, best_iou

    def _purge_stale(self, now: float, max_age: float) -> None:
        for tid in list(self._tracks.keys()):
            if now - self._tracks[tid].last_seen > max_age:
                del self._tracks[tid]


def _box_iou_pixels(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def score_yolo_from_detections(detections: list[YoloDetection]) -> FrameScore:
    """Fill tool-related fields on a fresh FrameScore (other fields set elsewhere)."""
    fs = FrameScore()
    fs.on_phone = any(d.role == "phone" and d.stable for d in detections)
    fs.on_laptop = any(d.role == "laptop_hands" and d.stable for d in detections)
    fs.on_keyboard = any(d.role == "keyboard_hands" and d.stable for d in detections)
    fs.on_mouse = any(d.role == "mouse_hands" and d.stable for d in detections)
    fs.tool_named_hits = sum(1 for d in detections if d.role == "tool_ok")
    fs.tool_named_checks = sum(
        1 for d in detections if d.role in ("tool_ok", "tool_miss")
    )
    fs.bench_obj_count = sum(1 for d in detections if d.role == "bench_obj")
    if fs.tool_named_checks > 0:
        fs.named_tool_score = min(1.0, fs.tool_named_hits / fs.tool_named_checks)
    if fs.bench_obj_count > 0:
        fs.bench_clutter_score = min(1.0, 0.35 + 0.15 * fs.bench_obj_count)
    return fs


def score_yolo(
    frame: Any,
    results: Any,
    roi: BenchROI,
    *,
    tool_names: frozenset[str] = TOOL_CLASSES,
    wrist_points: list[tuple[float, float]] | None = None,
) -> FrameScore:
    dets = collect_yolo_detections(
        frame, results, roi, tool_names=tool_names, wrist_points=wrist_points
    )
    return score_yolo_from_detections(dets)
