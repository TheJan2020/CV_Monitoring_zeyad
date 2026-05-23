"""
Detect workbench/hallway activity: YOLO11 (people + tools) + YOLO11-pose or MediaPipe skeletons.

Run:
  python workbench_activity.py

Tune WORKBENCH_ROI in .env (default: full frame 0,0,1,1).
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
from ultralytics import YOLO

from env_settings import (
    get_frigate_base_url,
    get_frigate_camera,
    get_frigate_fps,
    get_phone_min_conf,
    get_workbench_roi,
    get_workbench_thresholds,
    get_yolo_conf,
    get_yolo_imgsz,
    get_yolo_model,
    get_yolo_iou,
    get_yolo_person_conf,
    get_max_pose_persons,
    get_pose_crop_pad_frac,
    get_pose_min_detection_conf,
    get_yolo_pose_conf,
    get_yolo_pose_model,
    get_hand_box_extend_frac,
    get_hand_box_forearm_scale,
    get_wrist_box_half_frac,
    get_wrist_phone_min_iou_pct,
    get_wrist_phone_min_wrist_cov_pct,
    get_wrist_phone_min_phone_cov_pct,
    get_wrist_laptop_min_iou_pct,
    get_wrist_laptop_min_hand_cov_pct,
    get_wrist_laptop_min_laptop_cov_pct,
    get_laptop_min_conf,
    get_phone_stable_seconds,
    get_phone_stable_min_conf,
    get_laptop_stable_seconds,
    get_laptop_stable_min_conf,
    get_wrist_keyboard_min_iou_pct,
    get_wrist_keyboard_min_hand_cov_pct,
    get_wrist_keyboard_min_keyboard_cov_pct,
    get_keyboard_min_conf,
    get_keyboard_stable_seconds,
    get_keyboard_stable_min_conf,
    get_yolo_keyboard_conf,
    get_yolo_keyboard_imgsz,
    use_yolo_keyboard_pass,
    get_wrist_mouse_min_iou_pct,
    get_wrist_mouse_min_hand_cov_pct,
    get_wrist_mouse_min_mouse_cov_pct,
    get_mouse_min_conf,
    get_mouse_stable_seconds,
    get_mouse_stable_min_conf,
    get_yolo_mouse_conf,
    get_yolo_mouse_imgsz,
    use_yolo_mouse_pass,
    get_yolo_phone_conf,
    get_yolo_phone_imgsz,
    get_yolo_phone_model,
    get_yolo_laptop_conf,
    get_yolo_laptop_imgsz,
    get_phone_hand_crop_pad,
    use_phone_hand_crop,
    use_yolo_laptop_pass,
    use_yolo_phone_pass,
    use_yolo_people_pass,
    use_yolo_devices_pass,
    needs_device_yolo_model,
    enabled_detection_passes_label,
    enabled_work_device_class_ids,
    get_yolo_devices_conf,
    get_yolo_devices_imgsz,
    pipeline_passes_label,
    show_all_coco_tools,
    use_pose,
    use_yolo_pose,
    log_verbose,
)
from frigate_http import latest_jpeg_url, poll_frames
from workbench_logic import (
    ActivityState,
    ActivityTracker,
    FrameScore,
    NearHandsStabilizer,
    WristHoldover,
    apply_person_box_to_score,
    apply_wrists_hands_score,
    best_wrist_phone_iou,
    best_hand_overlap_iou,
    build_wrist_boxes,
    collect_yolo_detections,
    compute_wrist_phone_overlaps,
    compute_wrist_laptop_overlaps,
    compute_wrist_keyboard_overlaps,
    collect_yolo_pose_persons,
    consolidate_person_detections,
    count_person_detections,
    count_phones_in_frame,
    infer_wrists_from_yolo,
    estimate_pose_full_frame,
    estimate_poses_for_persons,
    merge_pose_into_score,
    merge_phone_detections,
    log_phone_detections,
    log_all_detections,
    log_person_detections,
    log_wrist_and_hands,
    log_hand_overlaps,
    log_frame_scores,
    detect_work_devices_full_frame,
    merge_work_device_detections,
    detect_phones_on_hand_regions,
    HAND_WORK_DEVICE_NAMES,
    compute_wrist_mouse_overlaps,
    KEYBOARD_CLASS_NAMES,
    LAPTOP_CLASS_NAMES,
    MOUSE_CLASS_NAMES,
    PHONE_CLASS_NAMES,
    ObjectDetectionStabilizer,
    log_laptop_detections,
    log_keyboard_detections,
    log_mouse_detections,
    primary_hand_boxes,
    retag_phone_detections,
    retag_laptop_detections,
    retag_keyboard_detections,
    retag_mouse_detections,
    score_frame_yolo_person,
    score_pose,
    score_yolo_from_detections,
    wrists_from_pose_landmarks,
)
from workbench_viz import draw_explain_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Workbench activity (YOLO + optional pose).")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--camera", default="")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--no-show", action="store_true", help="Print state only, no window.")
    parser.add_argument("--no-pose", action="store_true", help="YOLO only, no skeleton overlay.")
    args = parser.parse_args()

    base = (args.base_url or get_frigate_base_url()).strip().rstrip("/")
    camera = (args.camera or get_frigate_camera()).strip()
    fps = args.fps if args.fps > 0 else get_frigate_fps()
    if not base or not camera:
        raise SystemExit("Set FRIGATE_BASE_URL and FRIGATE_CAMERA in .env")

    roi = get_workbench_roi()
    th = get_workbench_thresholds()
    model = YOLO(get_yolo_model())
    yolo_imgsz = get_yolo_imgsz()
    phone_model = model
    phone_model_name = get_yolo_model()
    if needs_device_yolo_model():
        phone_model_name = get_yolo_phone_model()
        phone_model = model if phone_model_name == get_yolo_model() else YOLO(phone_model_name)
    yolo_devices_imgsz = get_yolo_devices_imgsz()
    device_class_ids = enabled_work_device_class_ids()
    enable_pose = use_pose() and not args.no_pose
    use_yolo_pose_path = enable_pose and use_yolo_pose()
    pose_model = YOLO(get_yolo_pose_model()) if use_yolo_pose_path else None

    print(f"Frigate: {latest_jpeg_url(base, camera)}")
    roi_label = "full frame" if roi.is_full_frame() else f"{roi.x1},{roi.y1} — {roi.x2},{roi.y2}"
    print(f"ROI: {roi_label}")
    print(f"Pipeline: {pipeline_passes_label()}")
    if use_yolo_people_pass():
        print(
            f"  Pass 1 — People: {get_yolo_model()} person_conf={get_yolo_person_conf():.2f} "
            f"imgsz={yolo_imgsz}"
        )
        if use_yolo_pose_path:
            print(
                f"           Pose: {get_yolo_pose_model()} conf≥{get_yolo_pose_conf():.2f}"
            )
        elif enable_pose:
            print(
                f"           Pose: MediaPipe conf≥{get_pose_min_detection_conf():.2f}"
            )
        else:
            print("           Pose: OFF")
    if use_yolo_devices_pass() and device_class_ids:
        print(
            f"  Pass 2 — Devices: {phone_model_name} classes={device_class_ids} "
            f"conf={get_yolo_devices_conf():.2f} imgsz={yolo_devices_imgsz} "
            f"({enabled_detection_passes_label()})"
        )
        if use_yolo_phone_pass() and use_phone_hand_crop():
            print("           Phone hand-crop zoom: ON")
    if show_all_coco_tools():
        print(
            f"  Extra — COCO tools: {get_yolo_model()} conf={get_yolo_conf():.2f} (SHOW_ALL_TOOLS=1)"
        )
    print("Primary person = thick blue box; WORKING = stable hands on laptop or keyboard only")
    print(
        f"WORKING after {th['work_on_s']:.1f}s on laptop/keyboard | "
        f"present {th['present_on_s']:.1f}s | exit hold {th['work_off_s']:.1f}s"
    )
    if log_verbose():
        print("Logging: VERBOSE — every frame, all pipeline stages (LOG_VERBOSE=0 for compact)")
    else:
        print("Logging: compact — device logs ~0.5s; state/score on change (LOG_VERBOSE=1 for full)")
    print("Press Q to quit.\n")

    verbose_log = log_verbose()
    frame_no = 0
    last_state = ActivityState.IDLE
    phone_hint_shown = False
    wrist_hold = WristHoldover()
    near_hands = NearHandsStabilizer()
    phone_stabilizer = ObjectDetectionStabilizer(
        class_names=PHONE_CLASS_NAMES,
        stable_seconds=get_phone_stable_seconds(),
        min_conf=get_phone_stable_min_conf(),
    )
    laptop_stabilizer = ObjectDetectionStabilizer(
        class_names=LAPTOP_CLASS_NAMES,
        stable_seconds=get_laptop_stable_seconds(),
        min_conf=get_laptop_stable_min_conf(),
    )
    keyboard_stabilizer = ObjectDetectionStabilizer(
        class_names=KEYBOARD_CLASS_NAMES,
        stable_seconds=get_keyboard_stable_seconds(),
        min_conf=get_keyboard_stable_min_conf(),
    )
    mouse_stabilizer = ObjectDetectionStabilizer(
        class_names=MOUSE_CLASS_NAMES,
        stable_seconds=get_mouse_stable_seconds(),
        min_conf=get_mouse_stable_min_conf(),
    )
    interval = max(0.05, 1.0 / max(0.1, fps))
    phone_log_interval = max(0.5, interval)
    last_phone_log_t = 0.0
    tracker = ActivityTracker(
        work_on_s=th["work_on_s"],
        work_off_s=th["work_off_s"],
        present_on_s=th["present_on_s"],
        medium_threshold=th["medium_threshold"],
        strict_threshold=th["strict_threshold"],
        present_threshold=th["present_threshold"],
    )

    pose_det_conf = get_pose_min_detection_conf()
    mp_pose = mp.solutions.pose
    mp_pose_ctx = (
        mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=pose_det_conf,
            min_tracking_confidence=pose_det_conf,
        )
        if enable_pose and not use_yolo_pose_path
        else nullcontext(None)
    )

    with mp_pose_ctx as mp_pose_inst:
        for frame in poll_frames(base, camera, fps=fps):
            t0 = time.perf_counter()
            fs = FrameScore()
            fs.pose_used_fallback = False
            person_poses: list = []
            primary_pose = None
            yolo_detections: list = []
            wrist_boxes: list = []
            wrist_phone_overlaps: list = []
            wrist_laptop_overlaps: list = []
            wrist_keyboard_overlaps: list = []
            wrist_mouse_overlaps: list = []
            fh, fw = frame.shape[:2]
            pose_wrists: list[tuple[float, float]] = []
            frame_no += 1
            if verbose_log:
                print(
                    f"\n========== frame #{frame_no} {time.strftime('%H:%M:%S')} ==========",
                    flush=True,
                )

            # --- Pass 1: People (person boxes; pose runs below) ---
            yolo_conf = get_yolo_conf()
            person_conf = get_yolo_person_conf()
            yolo_iou = get_yolo_iou()
            show_tools = show_all_coco_tools()
            person_dets: list = []
            tool_dets: list = []
            person_res = None
            yolo_res = None

            if use_yolo_people_pass():
                person_res = model(
                    frame,
                    verbose=False,
                    conf=person_conf,
                    classes=[0],
                    imgsz=yolo_imgsz,
                    iou=yolo_iou,
                )
                person_dets = collect_yolo_detections(
                    frame, person_res, roi, wrist_points=[], show_all_coco_tools=show_tools
                )
                if verbose_log:
                    log_person_detections("people_pass", person_dets, fw, fh)

            if show_tools:
                yolo_res = model(
                    frame, verbose=False, conf=yolo_conf, imgsz=yolo_imgsz, iou=yolo_iou
                )
                full_dets = collect_yolo_detections(
                    frame, yolo_res, roi, wrist_points=[], show_all_coco_tools=show_tools
                )
                tool_dets = [
                    d
                    for d in full_dets
                    if d.name != "person" and d.name not in HAND_WORK_DEVICE_NAMES
                ]
                if verbose_log:
                    log_all_detections("coco_tools_pass", tool_dets, fw, fh)

            merged = list(person_dets) + tool_dets

            # --- Pass 2: Work devices (phone, laptop, keyboard, mouse — one YOLO run) ---
            if device_class_ids:
                device_dets = detect_work_devices_full_frame(
                    phone_model,
                    frame,
                    roi,
                    class_ids=device_class_ids,
                    conf=get_yolo_devices_conf(),
                    imgsz=yolo_devices_imgsz,
                    iou=yolo_iou,
                    show_all_coco_tools=show_tools,
                )
                merged = merge_work_device_detections(
                    merged, device_dets, iou_threshold=yolo_iou
                )
                if verbose_log:
                    log_all_detections("devices_pass", merged, fw, fh)

            yolo_detections, primary_person = consolidate_person_detections(
                merged, roi, fw, fh, iou_threshold=yolo_iou
            )
            n_persons = count_person_detections(yolo_detections)

            now_log = time.perf_counter()
            log_devices = verbose_log or (now_log - last_phone_log_t >= phone_log_interval)
            if verbose_log:
                log_all_detections("after_consolidate", yolo_detections, fw, fh)
                log_person_detections("after_consolidate", yolo_detections, fw, fh)
            elif log_devices:
                log_person_detections("people_pass", person_dets, fw, fh)
                if device_class_ids:
                    log_phone_detections("devices_pass", yolo_detections, fw, fh)
                    log_laptop_detections("devices_pass", yolo_detections, fw, fh)
                    log_keyboard_detections("devices_pass", yolo_detections, fw, fh)
                    log_mouse_detections("devices_pass", yolo_detections, fw, fh)
                last_phone_log_t = now_log

            # --- Pose (part of people pass) ---
            if use_yolo_people_pass() and pose_model is not None:
                pose_res = pose_model(
                    frame,
                    verbose=False,
                    conf=get_yolo_pose_conf(),
                    imgsz=yolo_imgsz,
                    classes=[0],
                )
                person_poses = collect_yolo_pose_persons(
                    pose_res,
                    fw,
                    fh,
                    primary_person,
                    box_conf=get_yolo_pose_conf(),
                    max_persons=get_max_pose_persons(),
                )
            elif use_yolo_people_pass() and mp_pose_inst is not None:
                persons = [d for d in yolo_detections if d.name == "person"]
                if persons:
                    person_poses = estimate_poses_for_persons(
                        frame,
                        persons,
                        mp_pose_inst,
                        max_persons=get_max_pose_persons(),
                        pad_frac=get_pose_crop_pad_frac(),
                    )
                else:
                    single = estimate_pose_full_frame(frame, mp_pose_inst)
                    if single is not None:
                        person_poses = [single]

            if person_poses:
                primary_pose = next(
                    (p for p in person_poses if p.is_primary),
                    person_poses[0],
                )
                if primary_pose.landmarks is not None:
                    pose_fs = score_pose(primary_pose.landmarks, roi)
                    merge_pose_into_score(fs, pose_fs)
                    pose_wrists = wrists_from_pose_landmarks(primary_pose.landmarks)

            wrists_for_yolo: list[tuple[float, float]] = []
            wrists_stable: list[tuple[float, float]] = []
            if use_yolo_people_pass():
                if primary_person is not None:
                    wrists_for_yolo = [
                        (
                            (primary_person.x1 + primary_person.x2) / 2 / fw,
                            primary_person.y2 / fh - 0.05,
                        )
                    ]
                    apply_person_box_to_score(fs, primary_person, roi, fw, fh)
                    fs.pose_used_fallback = True
                else:
                    if person_res is not None:
                        wrists_for_yolo = infer_wrists_from_yolo(frame, person_res, roi)
                    if not wrists_for_yolo and yolo_res is not None:
                        wrists_for_yolo = infer_wrists_from_yolo(frame, yolo_res, roi)
                    if person_res is not None and score_frame_yolo_person(
                        fs, frame, person_res, roi
                    ):
                        fs.pose_used_fallback = True
                    elif yolo_res is not None and score_frame_yolo_person(
                        fs, frame, yolo_res, roi
                    ):
                        fs.pose_used_fallback = True

                if n_persons == 0:
                    wrist_hold.clear()
                else:
                    wrists_stable = wrist_hold.resolve(
                        pose_wrists if pose_wrists else wrists_for_yolo
                    )

            wrist_boxes = build_wrist_boxes(
                fw,
                fh,
                person_poses=person_poses,
                person_detections=[
                    d for d in yolo_detections if d.name == "person"
                ],
                wrist_points_norm=wrists_stable if not person_poses else None,
                primary_person=primary_person,
                half_frac=get_wrist_box_half_frac(),
                forearm_scale=get_hand_box_forearm_scale(),
                hand_extend_frac=get_hand_box_extend_frac(),
            )
            primary_hands = primary_hand_boxes(wrist_boxes)
            if verbose_log:
                log_wrist_and_hands(wrist_boxes, wrists_stable, fw, fh)

            if use_yolo_phone_pass() and use_phone_hand_crop() and primary_hands:
                crop_phones = detect_phones_on_hand_regions(
                    phone_model,
                    frame,
                    roi,
                    primary_hands,
                    conf=get_yolo_phone_conf(),
                    imgsz=yolo_devices_imgsz,
                    iou=yolo_iou,
                    show_all_coco_tools=show_tools,
                    pad_frac=get_phone_hand_crop_pad(),
                )
                yolo_detections = merge_phone_detections(
                    yolo_detections, crop_phones, iou_threshold=yolo_iou
                )
                if verbose_log:
                    log_all_detections("hand_crop_phones", yolo_detections, fw, fh)
                elif time.perf_counter() - last_phone_log_t < phone_log_interval:
                    log_phone_detections("hand_crop_phones", yolo_detections, fw, fh)

            if use_yolo_phone_pass():
                retag_phone_detections(
                    yolo_detections,
                    wrist_boxes,
                    min_conf=get_phone_min_conf(),
                    min_iou_percent=get_wrist_phone_min_iou_pct(),
                    min_wrist_coverage_percent=get_wrist_phone_min_wrist_cov_pct(),
                    min_phone_coverage_percent=get_wrist_phone_min_phone_cov_pct(),
                )
            if use_yolo_laptop_pass():
                retag_laptop_detections(
                    yolo_detections,
                    wrist_boxes,
                    min_conf=get_laptop_min_conf(),
                    min_iou_percent=get_wrist_laptop_min_iou_pct(),
                    min_hand_coverage_percent=get_wrist_laptop_min_hand_cov_pct(),
                    min_laptop_coverage_percent=get_wrist_laptop_min_laptop_cov_pct(),
                )
            if use_yolo_keyboard_pass():
                retag_keyboard_detections(
                    yolo_detections,
                    wrist_boxes,
                    min_conf=get_keyboard_min_conf(),
                    min_iou_percent=get_wrist_keyboard_min_iou_pct(),
                    min_hand_coverage_percent=get_wrist_keyboard_min_hand_cov_pct(),
                    min_keyboard_coverage_percent=get_wrist_keyboard_min_keyboard_cov_pct(),
                )
            if use_yolo_mouse_pass():
                retag_mouse_detections(
                    yolo_detections,
                    wrist_boxes,
                    min_conf=get_mouse_min_conf(),
                    min_iou_percent=get_wrist_mouse_min_iou_pct(),
                    min_hand_coverage_percent=get_wrist_mouse_min_hand_cov_pct(),
                    min_mouse_coverage_percent=get_wrist_mouse_min_mouse_cov_pct(),
                )

            if use_yolo_phone_pass():
                phone_stabilizer.apply(yolo_detections)
            if use_yolo_laptop_pass():
                laptop_stabilizer.apply(yolo_detections)
            if use_yolo_keyboard_pass():
                keyboard_stabilizer.apply(yolo_detections)
            if use_yolo_mouse_pass():
                mouse_stabilizer.apply(yolo_detections)

            if use_yolo_phone_pass():
                fs.phone_near_hand = any(
                    d.name in PHONE_CLASS_NAMES
                    and d.near_hand
                    and d.confidence >= get_phone_min_conf()
                    for d in yolo_detections
                )
                wrist_phone_overlaps = compute_wrist_phone_overlaps(
                    primary_hands, yolo_detections, stable_only=False
                )
                fs.best_phone_wrist_iou_pct = best_wrist_phone_iou(wrist_phone_overlaps)
            if use_yolo_laptop_pass():
                wrist_laptop_overlaps = compute_wrist_laptop_overlaps(
                    primary_hands, yolo_detections, stable_only=True
                )
                fs.best_laptop_hand_iou_pct = best_hand_overlap_iou(wrist_laptop_overlaps)
            if use_yolo_keyboard_pass():
                wrist_keyboard_overlaps = compute_wrist_keyboard_overlaps(
                    primary_hands, yolo_detections, stable_only=True
                )
                fs.best_keyboard_hand_iou_pct = best_hand_overlap_iou(wrist_keyboard_overlaps)
            if use_yolo_mouse_pass():
                wrist_mouse_overlaps = compute_wrist_mouse_overlaps(
                    primary_hands, yolo_detections, stable_only=True
                )
                fs.best_mouse_hand_iou_pct = best_hand_overlap_iou(wrist_mouse_overlaps)

            if verbose_log:
                log_all_detections("after_retag", yolo_detections, fw, fh)
                if use_yolo_phone_pass():
                    log_hand_overlaps("phone", wrist_phone_overlaps)
                if use_yolo_laptop_pass():
                    log_hand_overlaps("laptop", wrist_laptop_overlaps)
                if use_yolo_keyboard_pass():
                    log_hand_overlaps("keyboard", wrist_keyboard_overlaps)
                if use_yolo_mouse_pass():
                    log_hand_overlaps("mouse", wrist_mouse_overlaps)
            elif time.perf_counter() - last_phone_log_t < phone_log_interval:
                if use_yolo_phone_pass():
                    log_phone_detections("after_retag", yolo_detections, fw, fh)
                if use_yolo_laptop_pass():
                    log_laptop_detections("after_laptop_retag", yolo_detections, fw, fh)
                if use_yolo_keyboard_pass():
                    log_keyboard_detections("after_keyboard_retag", yolo_detections, fw, fh)
                if use_yolo_mouse_pass():
                    log_mouse_detections("after_mouse_retag", yolo_detections, fw, fh)

            apply_wrists_hands_score(fs, wrists_stable, roi)
            tool_fs = score_yolo_from_detections(yolo_detections)
            fs.on_phone = tool_fs.on_phone
            fs.on_laptop = tool_fs.on_laptop
            fs.on_keyboard = tool_fs.on_keyboard
            fs.on_mouse = tool_fs.on_mouse
            fs.tool_named_hits = tool_fs.tool_named_hits
            fs.tool_named_checks = tool_fs.tool_named_checks
            fs.bench_obj_count = tool_fs.bench_obj_count
            fs.named_tool_score = tool_fs.named_tool_score
            fs.bench_clutter_score = tool_fs.bench_clutter_score

            saw_phone = (
                use_yolo_phone_pass()
                and any(d.role == "phone" and d.stable for d in yolo_detections)
            )
            near_hands.observe_phone(saw_phone)
            near_hands.observe_detections(len(yolo_detections))
            fs.on_phone = tool_fs.on_phone or near_hands.on_phone
            if n_persons == 0:
                fs.on_phone = False
                fs.phone_near_hand = False
                fs.on_laptop = False
                fs.on_keyboard = False
                fs.on_mouse = False
                fs.best_phone_wrist_iou_pct = 0.0
                fs.best_laptop_hand_iou_pct = 0.0
                fs.best_keyboard_hand_iou_pct = 0.0
                fs.best_mouse_hand_iou_pct = 0.0
                near_hands.observe_phone(False)

            if n_persons == 0 and not phone_hint_shown:
                print(
                    "tip: no YOLO person — lower YOLO_PERSON_CONF or use YOLO_MODEL=yolo11s.pt",
                    flush=True,
                )
                phone_hint_shown = True
            elif n_persons > 0:
                phone_hint_shown = False

            fs.compute_totals(use_yolo=True)
            state = tracker.update(fs, interval)

            if verbose_log:
                log_frame_scores(
                    fs,
                    state=state.value,
                    n_persons=n_persons,
                    n_poses=len(person_poses),
                    pose_fallback=fs.pose_used_fallback,
                    near_hands_label=near_hands.status_label(),
                )
            else:
                log_line = near_hands.log_line()
                if log_line:
                    print(log_line, flush=True)

            if state != last_state:
                tier = "strict" if fs.strict_score >= th["strict_threshold"] else "medium"
                print(
                    f"[{time.strftime('%H:%M:%S')}] {last_state.value} → {state.value} "
                    f"(M={fs.medium_score:.2f} S={fs.strict_score:.2f} persons={n_persons} via {tier})",
                    flush=True,
                )
                last_state = state

            if not args.no_show:
                vis = draw_explain_overlay(
                    frame,
                    roi,
                    fs,
                    state,
                    tracker,
                    th,
                    yolo_detections=yolo_detections,
                    wrists=wrists_stable,
                    phones_in_frame=sum(
                        1
                        for d in yolo_detections
                        if d.name in PHONE_CLASS_NAMES and d.stable
                    ),
                    yolo_person_count=n_persons,
                    person_poses=person_poses,
                    pose_enabled=enable_pose,
                    wrist_boxes=wrist_boxes,
                    wrist_phone_overlaps=wrist_phone_overlaps,
                    wrist_laptop_overlaps=wrist_laptop_overlaps,
                    wrist_keyboard_overlaps=wrist_keyboard_overlaps,
                    wrist_mouse_overlaps=wrist_mouse_overlaps,
                )
                cv2.imshow("Workbench activity (explain)", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
