"""Explainability overlay — why the workbench model chose each state."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from workbench_logic import (
    COCO_POSE_CONNECTIONS,
    COCO_TOOL_LABELS,
    ActivityState,
    ActivityTracker,
    BenchROI,
    FrameScore,
    WristBox,
    HandObjectOverlap,
    WristPhoneOverlap,
    YoloDetection,
)

# BGR colors
C_ROI = (0, 255, 255)
C_PERSON = (255, 0, 0)  # BGR blue for YOLO person
C_TOOL_OK = (0, 255, 0)
C_WORKING = (0, 255, 0)
C_LAPTOP = (0, 255, 255)  # cyan — desk / not at hands
C_LAPTOP_HANDS = (0, 165, 255)  # orange — laptop overlapping hand box
C_LINE_PHONE = (0, 0, 255)  # red — hand↔phone overlap line
C_LINE_LAPTOP = (0, 165, 255)
C_KEYBOARD = (255, 180, 100)  # light purple — desk
C_KEYBOARD_HANDS = (255, 80, 200)  # pink — at hands
C_LINE_KEYBOARD = (255, 80, 200)
C_MOUSE = (0, 220, 255)  # yellow — desk mouse
C_LINE_MOUSE = (0, 200, 255)
C_TOOL_MISS = (100, 100, 100)
C_BENCH = (0, 165, 255)
C_OTHER = (180, 180, 180)
C_PHONE = (0, 0, 255)  # BGR red — phone overlapping hand/wrist box
C_PHONE_DESK = (0, 0, 0)  # desk / not in hand
C_WRIST_OK = (0, 255, 0)
C_WRIST_NO = (0, 0, 255)
C_WRIST_BOX_L = (255, 220, 0)  # BGR: left wrist region
C_WRIST_BOX_R = (0, 200, 255)  # right wrist region
C_WRIST_BOX = (200, 255, 200)
C_TORSO = (255, 0, 255)


def _bar(
    img: Any,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    label: str,
    *,
    thresh: float | None = None,
    contrib: str = "",
) -> int:
    """Draw one labeled score bar; return y below it."""
    value = max(0.0, min(1.0, value))
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill_w = int(w * value)
    color = (0, 200, 0) if thresh is None or value >= thresh else (0, 140, 255)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    if thresh is not None:
        tx = x + int(w * thresh)
        cv2.line(img, (tx, y), (tx, y + h), (255, 255, 255), 1)
    txt = f"{label} {value:.2f}"
    if contrib:
        txt += f"  ({contrib})"
    cv2.putText(img, txt, (x + 4, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
    return y + h + 6


def _state_reason(
    state: ActivityState,
    score: FrameScore,
    tracker: ActivityTracker,
    th: dict[str, float],
) -> str:
    if state == ActivityState.WORKING:
        if not score.at_computer_work and tracker.working_cooldown > 0:
            return (
                f"held — exit in {max(0.0, tracker.working_cooldown):.1f}s "
                f"(hands off laptop/keyboard)"
            )
        parts = []
        if score.on_laptop:
            parts.append("laptop")
        if score.on_keyboard:
            parts.append("keyboard")
        if parts:
            return f"hands on {' + '.join(parts)} ({tracker.work_surface_streak:.1f}s)"
        return f"need hands on laptop/keyboard ({tracker.work_surface_streak:.1f}s)"
    if state == ActivityState.PRESENT:
        return f"in zone {tracker.present_streak:.1f}s (person, low activity)"
    if score.person_visible:
        return "person seen but scores below present threshold"
    return "no YOLO person in frame"


def draw_explain_overlay(
    frame: Any,
    roi: BenchROI,
    score: FrameScore,
    state: ActivityState,
    tracker: ActivityTracker,
    th: dict[str, float],
    *,
    yolo_detections: list[YoloDetection],
    wrists: list[tuple[float, float]],
    phones_in_frame: int = 0,
    yolo_person_count: int = 0,
    person_poses: list[Any] | None = None,
    pose_enabled: bool = False,
    wrist_boxes: list[WristBox] | None = None,
    wrist_phone_overlaps: list[WristPhoneOverlap] | None = None,
    wrist_laptop_overlaps: list[HandObjectOverlap] | None = None,
    wrist_keyboard_overlaps: list[HandObjectOverlap] | None = None,
    wrist_mouse_overlaps: list[HandObjectOverlap] | None = None,
    minimal: bool = False,
) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]
    rx1, ry1, rx2, ry2 = roi.as_pixels(w, h)

    # --- ROI (full frame = label only) ---
    roi_poly_px = roi.polygon_pixels(w, h)
    if minimal:
        if roi_poly_px is not None:
            pts = np.array([roi_poly_px], dtype=np.int32)
            cv2.polylines(out, pts, isClosed=True, color=C_ROI, thickness=1)
        elif not roi.is_full_frame():
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), C_ROI, 1)
    elif roi_poly_px is not None:
        pts = np.array([roi_poly_px], dtype=np.int32)
        cv2.polylines(out, pts, isClosed=True, color=C_ROI, thickness=2)
        cv2.putText(
            out,
            "ROI polygon",
            (roi_poly_px[0][0] + 4, max(roi_poly_px[0][1] - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            C_ROI,
            1,
        )
    elif roi.is_full_frame():
        cv2.putText(
            out,
            "ROI: full frame",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            C_ROI,
            2,
        )
    else:
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), C_ROI, 2)
        cv2.putText(
            out,
            "bench ROI",
            (rx1 + 4, max(ry1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            C_ROI,
            1,
        )

    phone_overlaps = wrist_phone_overlaps or []
    laptop_overlaps = wrist_laptop_overlaps or []
    keyboard_overlaps = wrist_keyboard_overlaps or []
    mouse_overlaps = wrist_mouse_overlaps or []

    def _best_overlap_map(
        overlaps: list[HandObjectOverlap],
    ) -> dict[tuple[str, int, int], HandObjectOverlap]:
        out: dict[tuple[str, int, int], HandObjectOverlap] = {}
        for o in overlaps:
            key = (o.object_name, o.object_x1, o.object_y1)
            prev = out.get(key)
            if prev is None or o.iou_percent > prev.iou_percent:
                out[key] = o
        return out

    overlap_by_phone = _best_overlap_map(phone_overlaps)
    overlap_by_laptop = _best_overlap_map(laptop_overlaps)
    overlap_by_keyboard = _best_overlap_map(keyboard_overlaps)
    overlap_by_mouse = _best_overlap_map(mouse_overlaps)

    # --- Hand regions (small box past wrist toward fingers) ---
    if not minimal:
        for wb in wrist_boxes or []:
            side = wb.label.split("_")[-1] if "_" in wb.label else wb.label
            col = (
                C_WRIST_BOX_L
                if side == "left"
                else C_WRIST_BOX_R
                if side == "right"
                else C_WRIST_BOX
            )
            cv2.rectangle(out, (wb.x1, wb.y1), (wb.x2, wb.y2), col, 2)
            tag = f"{wb.label} hand" if wb.label else "hand"
            cv2.putText(
                out,
                tag,
                (wb.x1, max(wb.y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                col,
                1,
            )

    # --- YOLO boxes ---
    for d in yolo_detections:
        label_color = C_OTHER
        if d.role == "person":
            if d.is_primary:
                color, thick, tag = C_PERSON, 4, "person (primary)"
            else:
                color, thick, tag = C_PERSON, 2, "person"
        elif d.role == "mouse_hands":
            color, thick, tag = C_WORKING, 3, "mouse"
        elif d.name == "mouse" and not d.stable:
            color, thick, tag = (70, 70, 70), 1, f"mouse (pending {d.stable_age_s:.1f}s)"
            label_color = (180, 180, 180)
        elif d.name == "mouse":
            color, thick, tag = C_MOUSE, 2, "mouse (desk)"
        elif d.role == "keyboard_hands":
            color, thick, tag = C_WORKING, 3, "keyboard"
        elif d.name == "keyboard" and not d.stable:
            color, thick, tag = (70, 70, 70), 1, f"keyboard (pending {d.stable_age_s:.1f}s)"
            label_color = (180, 180, 180)
        elif d.name == "keyboard":
            color, thick, tag = C_KEYBOARD, 2, "keyboard (desk)"
        elif d.role == "laptop_hands":
            color, thick, tag = C_WORKING, 3, "laptop"
        elif d.name == "laptop" and not d.stable:
            color, thick, tag = (70, 70, 70), 1, f"laptop (pending {d.stable_age_s:.1f}s)"
            label_color = (180, 180, 180)
        elif d.name == "laptop":
            color, thick, tag = C_LAPTOP, 2, "laptop (desk)"
        elif d.role == "tool_ok":
            color, thick, tag = C_TOOL_OK, 2, d.name
        elif d.role == "tool_miss":
            if d.name == "laptop":
                color, thick, tag = C_LAPTOP, 2, "laptop (weak)"
            else:
                color, thick, tag = (0, 160, 0), 1, f"{d.name} (weak)"
        elif d.role == "phone" or (d.name == "cell phone" and d.near_hand):
            color, thick, tag = C_PHONE, 3, "PHONE"
            label_color = C_PHONE
        elif d.name == "cell phone" and not d.stable:
            color, thick, tag = (70, 70, 70), 1, f"phone (pending {d.stable_age_s:.1f}s)"
            label_color = (180, 180, 180)
        elif d.name == "cell phone":
            color, thick, tag = C_PHONE_DESK, 2, "phone (desk)"
            label_color = (240, 240, 240)
        elif d.role == "bench_obj":
            color, thick, tag = C_BENCH, 2, f"{d.name} clutter"
        else:
            color, thick, tag = C_OTHER, 1, d.name
        if d.name != "cell phone":
            label_color = color
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thick)
        label = f"{tag} {d.confidence:.2f}"
        ov_phone = overlap_by_phone.get(("cell phone", d.x1, d.y1))
        ov_laptop = overlap_by_laptop.get(("laptop", d.x1, d.y1))
        if ov_phone is not None and d.name == "cell phone":
            label += (
                f" | IoU {ov_phone.iou_percent:.0f}%"
                f" (hand {ov_phone.hand_coverage_percent:.0f}%"
                f" phone {ov_phone.object_coverage_percent:.0f}%)"
            )
        elif ov_laptop is not None and d.name == "laptop":
            label += (
                f" | IoU {ov_laptop.iou_percent:.0f}%"
                f" (hand {ov_laptop.hand_coverage_percent:.0f}%"
                f" laptop {ov_laptop.object_coverage_percent:.0f}%)"
            )
        ov_keyboard = overlap_by_keyboard.get(("keyboard", d.x1, d.y1))
        if ov_keyboard is not None and d.name == "keyboard":
            label += (
                f" | IoU {ov_keyboard.iou_percent:.0f}%"
                f" (hand {ov_keyboard.hand_coverage_percent:.0f}%"
                f" kb {ov_keyboard.object_coverage_percent:.0f}%)"
            )
        ov_mouse = overlap_by_mouse.get(("mouse", d.x1, d.y1))
        if ov_mouse is not None and d.name == "mouse":
            label += (
                f" | IoU {ov_mouse.iou_percent:.0f}%"
                f" (hand {ov_mouse.hand_coverage_percent:.0f}%"
                f" mouse {ov_mouse.object_coverage_percent:.0f}%)"
            )
        if d.from_hand_crop:
            label += " [hand-zoom]"
        if d.inside_person:
            label += " in person box"
        if d.near_hand:
            label += " overlaps hand"
        if d.in_roi:
            label += " in ROI"
        cv2.putText(
            out,
            label,
            (d.x1, max(d.y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
            1,
        )
        if d.role in ("laptop_hands", "keyboard_hands"):
            cx = (d.x1 + d.x2) // 2
            cy = (d.y1 + d.y2) // 2
            cv2.putText(
                out,
                "WORKING",
                (max(d.x1, cx - 40), cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                C_WORKING,
                2,
            )
        elif d.role == "phone" or (d.name == "cell phone" and d.near_hand):
            cx = (d.x1 + d.x2) // 2
            cy = (d.y1 + d.y2) // 2
            cv2.putText(
                out,
                "PHONE",
                (max(d.x1, cx - 28), cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                C_PHONE,
                2,
            )

    def _draw_hand_object_lines(
        overlaps: list[HandObjectOverlap], line_color: tuple[int, int, int]
    ) -> None:
        for o in overlaps:
            if o.iou_percent < 1.0 and o.hand_coverage_percent < 1.0:
                continue
            wb = next((b for b in (wrist_boxes or []) if b.label == o.wrist_label), None)
            obj = next(
                (
                    d
                    for d in yolo_detections
                    if d.name == o.object_name and d.x1 == o.object_x1 and d.y1 == o.object_y1
                ),
                None,
            )
            if wb is None or obj is None:
                continue
            ocx, ocy = (obj.x1 + obj.x2) // 2, (obj.y1 + obj.y2) // 2
            cv2.line(out, (wb.cx, wb.cy), (ocx, ocy), line_color, 1)
            mx, my = (wb.cx + ocx) // 2, (wb.cy + ocy) // 2
            cv2.putText(
                out,
                f"{o.object_name} IoU {o.iou_percent:.0f}%",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                line_color,
                1,
            )

    if not minimal:
        _draw_hand_object_lines(phone_overlaps, C_LINE_PHONE)
        _draw_hand_object_lines(laptop_overlaps, C_LINE_LAPTOP)
        _draw_hand_object_lines(keyboard_overlaps, C_LINE_KEYBOARD)
        _draw_hand_object_lines(mouse_overlaps, C_LINE_MOUSE)

    # --- MediaPipe pose skeleton (one per YOLO person crop) ---
    poses = person_poses or []
    primary_pose_lm = next(
        (p.landmarks for p in poses if getattr(p, "is_primary", False)),
        poses[0].landmarks if poses else None,
    )
    if poses:
        for pp in poses:
            if pp.keypoints_xy is not None and pp.keypoints_conf is not None:
                xy = pp.keypoints_xy
                kc = pp.keypoints_conf
                for i, j in COCO_POSE_CONNECTIONS:
                    if kc[i] < 0.25 or kc[j] < 0.25:
                        continue
                    p1 = (int(xy[i][0]), int(xy[i][1]))
                    p2 = (int(xy[j][0]), int(xy[j][1]))
                    cv2.line(out, p1, p2, C_TORSO, 2)
                for idx in range(len(xy)):
                    if kc[idx] < 0.25:
                        continue
                    cv2.circle(out, (int(xy[idx][0]), int(xy[idx][1])), 3, C_TORSO, -1)
            elif pp.landmarks is not None:
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                mp_drawing.draw_landmarks(
                    out,
                    pp.landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=style,
                )
        if primary_pose_lm is not None:
            for idx in (15, 16):
                p = primary_pose_lm.landmark[idx]
                if p.visibility < 0.35:
                    continue
                px, py = int(p.x * w), int(p.y * h)
                in_r = roi.contains(p.x, p.y)
                cv2.circle(out, (px, py), 8, C_WRIST_OK if in_r else C_WRIST_NO, -1)
                cv2.putText(
                    out,
                    "wrist",
                    (px + 6, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    C_WRIST_OK if in_r else C_WRIST_NO,
                    1,
                )
    elif pose_enabled:
        cv2.putText(
            out,
            "pose: no skeleton (use YOLO person box for hands)",
            (8, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )

    # --- Hand point fallback (pose wrists or YOLO) ---
    if primary_pose_lm is None:
        for wx, wy in wrists:
            px, py = int(wx * w), int(wy * h)
            in_r = roi.contains(wx, wy)
            cv2.circle(out, (px, py), 10, C_WRIST_OK if in_r else C_WRIST_NO, -1)
            cv2.circle(out, (px, py), 12, (0, 255, 255), 2)

    if minimal:
        # Skip all the explainability chrome: status text, top banner, bottom
        # score panel and the right-side legend. Just return the annotated frame.
        return out

    # --- YOLO person status ---
    status_y = h - 8 if roi.is_full_frame() else min(ry2 + 22, h - 8)
    if yolo_person_count > 0:
        cv2.putText(
            out,
            f"YOLO: {yolo_person_count} person(s) — primary = thick blue (WORKING score)",
            (8, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            C_PERSON,
            2,
        )
    else:
        cv2.putText(
            out,
            "YOLO: no person detected (try yolo11s.pt / lower YOLO_CONF)",
            (8, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # --- Top banner: state ---
    state_color = {
        ActivityState.IDLE: (140, 140, 140),
        ActivityState.PRESENT: (255, 200, 0),
        ActivityState.WORKING: (0, 255, 0),
    }[state]
    banner_extra = (
        score.phone_in_hand
        or score.at_computer_work
        or score.best_phone_wrist_iou_pct > 0
        or score.best_laptop_hand_iou_pct > 0
        or score.best_keyboard_hand_iou_pct > 0
        or score.best_mouse_hand_iou_pct > 0
        or bool(yolo_detections)
    )
    banner_h = 86 if banner_extra else 52
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
    state_label = state.value.upper()
    if state == ActivityState.WORKING and not score.at_computer_work:
        state_label = "WORKING (held)"
    title = ""
    state_color = (140, 140, 140)
    subtitle = ""
    if score.phone_in_hand:
        title = "PHONE"
        subtitle = "not working" if score.on_phone else "in hand (confirming)"
        state_color = C_PHONE
    elif score.at_computer_work:
        title = "WORKING"
        parts = []
        if score.on_laptop:
            parts.append("laptop")
        if score.on_keyboard:
            parts.append("keyboard")
        subtitle = f"hands on {' + '.join(parts)}" if parts else "hands on laptop/keyboard"
        state_color = C_WORKING
    else:
        state_color = {
            ActivityState.IDLE: (140, 140, 140),
            ActivityState.PRESENT: (255, 200, 0),
            ActivityState.WORKING: (0, 255, 0),
        }[state]
        if state == ActivityState.WORKING and not score.at_computer_work:
            state_color = (0, 180, 255)
        title = f"{state_label}  |  M={score.medium_score:.2f}  S={score.strict_score:.2f}"
        subtitle = _state_reason(state, score, tracker, th)
    cv2.putText(out, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)
    if subtitle:
        cv2.putText(out, subtitle, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    sub_y = 68
    if score.best_phone_wrist_iou_pct > 0:
        cv2.putText(
            out,
            f"max hand-phone IoU: {score.best_phone_wrist_iou_pct:.0f}%",
            (10, sub_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            C_LINE_PHONE,
            1,
        )
        sub_y += 14
    if score.best_laptop_hand_iou_pct > 0:
        cv2.putText(
            out,
            f"max hand-laptop IoU: {score.best_laptop_hand_iou_pct:.0f}%",
            (10, sub_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            C_LINE_LAPTOP,
            1,
        )
        sub_y += 14
    if score.best_keyboard_hand_iou_pct > 0:
        cv2.putText(
            out,
            f"max hand-keyboard IoU: {score.best_keyboard_hand_iou_pct:.0f}%",
            (10, sub_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            C_LINE_KEYBOARD,
            1,
        )
        sub_y += 14
    elif phones_in_frame == 0 and not score.on_phone:
        cv2.putText(
            out,
            "No cell phone box from YOLO (lower YOLO_PHONE_CONF)",
            (10, sub_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 140, 255),
            1,
        )
    elif phones_in_frame > 0 and not score.on_phone:
        cv2.putText(
            out,
            f"{phones_in_frame} phone(s) in frame — not overlapping any hand box",
            (10, sub_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 140, 255),
            1,
        )

    # --- Bottom panel: score breakdown ---
    panel_h = 230
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    y0 = 8
    bw = w - 20
    m_roi = 0.35 * score.person_in_roi
    eff_hands = (
        min(score.hands_in_roi, 0.15)
        if score.phone_in_hand
        else score.hands_in_roi
    )
    m_hands = 0.35 * eff_hands
    m_pose = 0.30 * score.active_pose
    y0 = _bar(panel, 10, y0, bw, 18, score.person_in_roi, "person in ROI", contrib=f"+{m_roi:.2f} to M")
    if score.phone_in_hand:
        hands_lbl = (
            "hands in ROI (phone — not working)"
            if score.on_phone
            else "hands in ROI (phone — confirming)"
        )
    elif score.at_computer_work:
        hands_lbl = "hands in ROI (computer work)"
    else:
        hands_lbl = "hands in ROI"
    y0 = _bar(panel, 10, y0, bw, 18, eff_hands, hands_lbl, contrib=f"+{m_hands:.2f} to M")
    y0 = _bar(
        panel,
        10,
        y0,
        bw,
        18,
        score.active_pose,
        "active pose" if pose_enabled else "activity (YOLO)",
        contrib=f"+{m_pose:.2f} to M",
    )
    y0 = _bar(
        panel,
        10,
        y0,
        bw,
        18,
        score.medium_score,
        "MEDIUM (phase 2)",
        thresh=th["medium_threshold"],
        contrib="sum above",
    )
    y0 = _bar(
        panel,
        10,
        y0,
        bw,
        18,
        score.named_tool_score,
        f"named COCO tools ({score.tool_named_hits}/{max(1, score.tool_named_checks)})",
        contrib=f"+{0.2 * score.named_tool_score:.2f} to S if any",
    )
    y0 = _bar(
        panel,
        10,
        y0,
        bw,
        18,
        score.bench_clutter_score,
        f"bench clutter ({score.bench_obj_count} objs, not chairs)",
        contrib="fallback if 0 named tools",
    )
    y0 = _bar(
        panel,
        10,
        y0,
        bw,
        18,
        score.strict_score,
        "STRICT (phase 3)",
        thresh=th["strict_threshold"],
    )
    n_ok = sum(1 for d in yolo_detections if d.role == "tool_ok")
    n_miss = sum(1 for d in yolo_detections if d.role == "tool_miss")
    n_bench = sum(1 for d in yolo_detections if d.role == "bench_obj")
    phone_n = sum(1 for d in yolo_detections if d.role == "phone")
    n_person = sum(1 for d in yolo_detections if d.name == "person")
    cv2.putText(
        panel,
        f"persons: {n_person} | tools (green): {n_ok} | weak: {n_miss} | "
        f"other: {n_bench} | phone: {phone_n}",
        (10, min(y0 + 12, panel_h - 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 180, 180),
        1,
    )
    tools_line = "COCO tool list: " + ", ".join(COCO_TOOL_LABELS[:8]) + ", ..."
    cv2.putText(
        panel,
        tools_line,
        (10, panel_h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (140, 140, 140),
        1,
    )

    # Timers
    cv2.putText(
        panel,
        f"timers: laptop/keyboard {tracker.work_surface_streak:.1f}s "
        f"(need {th['work_on_s']:.1f}s) | present {tracker.present_streak:.1f}s",
        (10, panel_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (160, 220, 255),
        1,
    )

    # Legend (top-right on video)
    leg_x = w - 210
    for i, (txt, col) in enumerate(
        [
            ("cyan = laptop", C_LAPTOP),
            ("green = tool (COCO)", C_TOOL_OK),
            ("dark green = tool (weak conf)", (0, 160, 0)),
            ("black = phone (desk)", C_PHONE_DESK),
            ("red = phone (in hand)", C_PHONE),
            ("cyan = laptop (desk)", C_LAPTOP),
            ("orange = laptop (hands)", C_LAPTOP_HANDS),
            ("gray = other / furniture", C_OTHER),
            ("red line = hand-phone IoU", C_LINE_PHONE),
            ("orange line = hand-laptop IoU", C_LINE_LAPTOP),
            ("purple = keyboard (desk)", C_KEYBOARD),
            ("green WORKING = hands on laptop/keyboard", C_WORKING),
            ("red PHONE = in hand (not working)", C_PHONE),
            ("pink line = hand-keyboard IoU", C_LINE_KEYBOARD),
            ("yellow = mouse (desk)", C_MOUSE),
            ("yellow line = hand-mouse IoU", C_LINE_MOUSE),
            ("gold/cyan = hand region", C_WRIST_BOX_L),
            ("blue = person (YOLO)", C_PERSON),
            (
                f"skeleton = {'YOLO11-pose' if any(getattr(p, 'keypoints_xy', None) is not None for p in poses) else 'MediaPipe'} ({len(poses)} shown)",
                C_TORSO,
            ),
            (f"persons: {yolo_person_count}", C_PERSON),
        ]
    ):
        cv2.putText(out, txt, (leg_x, 58 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    combined = np.vstack([out, panel])
    return combined
