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
    resolve_video_source,
    use_frigate_http,
    get_yolo_device,
    show_explain_overlay,
    get_pose_sleep_seconds,
    get_pose_hold_seconds,
    get_pose_motion_still,
    get_pose_motion_active,
)
from frigate_http import latest_jpeg_url, poll_frames
from video_source import open_capture, read_loop
import web_server
from pose_state import PoseStateTracker
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
    parser.add_argument("--source", default="", help="Override VIDEO_SOURCE (direct RTSP/webcam).")
    parser.add_argument("--direct", action="store_true", help="Force direct VIDEO_SOURCE even if FRIGATE_BASE_URL is set.")
    parser.add_argument("--no-show", action="store_true", help="Print state only, no window.")
    parser.add_argument("--no-pose", action="store_true", help="YOLO only, no skeleton overlay.")
    parser.add_argument("--web", action="store_true", help="Serve live dashboard on http://<host>:<port>/")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web dashboard bind address (default 0.0.0.0).")
    parser.add_argument("--web-port", type=int, default=8000, help="Web dashboard port (default 8000).")
    parser.add_argument("--device", default="", help="YOLO device override: '', 'cuda:0', 'cpu', 'mps'.")
    args = parser.parse_args()

    base = (args.base_url or get_frigate_base_url()).strip().rstrip("/")
    camera = (args.camera or get_frigate_camera()).strip()
    fps = args.fps if args.fps > 0 else get_frigate_fps()
    use_direct = args.direct or not (base and use_frigate_http())
    source = (args.source or resolve_video_source()).strip() if use_direct else ""
    if not use_direct and (not base or not camera):
        raise SystemExit("Set FRIGATE_BASE_URL and FRIGATE_CAMERA in .env, or set VIDEO_SOURCE for direct mode")

    roi = get_workbench_roi()
    th = get_workbench_thresholds()
    yolo_device = (args.device or get_yolo_device()).strip()

    def _to_device(m: YOLO) -> YOLO:
        if yolo_device:
            m.to(yolo_device)
        return m

    model = _to_device(YOLO(get_yolo_model()))
    yolo_imgsz = get_yolo_imgsz()
    phone_model = model
    phone_model_name = get_yolo_model()
    if needs_device_yolo_model():
        phone_model_name = get_yolo_phone_model()
        phone_model = model if phone_model_name == get_yolo_model() else _to_device(YOLO(phone_model_name))
    yolo_devices_imgsz = get_yolo_devices_imgsz()
    device_class_ids = enabled_work_device_class_ids()
    enable_pose = use_pose() and not args.no_pose
    use_yolo_pose_path = enable_pose and use_yolo_pose()
    pose_model = _to_device(YOLO(get_yolo_pose_model())) if use_yolo_pose_path else None

    if args.web:
        web_server.start(host=args.web_host, port=args.web_port)

    # Baby-monitor mode (CAMERA_TYPE=baby): use the BabyTracker for a
    # persistent lock + simplified 5-state activity machine.
    camera_type = (os.environ.get("CAMERA_TYPE") or "general").strip().lower()
    baby_tracker = None
    if camera_type == "baby":
        from baby_tracker import BabyTracker
        baby_tracker = BabyTracker(
            sleep_seconds=get_pose_sleep_seconds(),
            hold_seconds=get_pose_hold_seconds(),
        )
        print(f"Camera type: baby (BabyTracker lock+states active)")

    pose_state_tracker = PoseStateTracker(
        still_for_sleep_s=get_pose_sleep_seconds(),
        hold_seconds=get_pose_hold_seconds(),
        motion_still_norm=get_pose_motion_still(),
        motion_active_norm=get_pose_motion_active(),
    )

    if use_direct:
        print(f"Direct video source: {source!r}")
    else:
        print(f"Frigate: {latest_jpeg_url(base, camera)}")
    print(f"YOLO device: {yolo_device or 'auto (CPU unless ultralytics finds CUDA/MPS)'}")
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

    cap = None
    if use_direct:
        cap = open_capture(source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source: {source!r}")
        frame_iter = read_loop(cap)
    else:
        frame_iter = poll_frames(base, camera, fps=fps)

    prev_t = time.perf_counter()
    with mp_pose_ctx as mp_pose_inst:
        for frame in frame_iter:
            t0 = time.perf_counter()
            dt = max(1e-3, t0 - prev_t)
            prev_t = t0
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
                    roi=roi,
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

            # Synthesize person box(es) from skeleton when YOLO person pass missed.
            # The pose model uses a lower confidence threshold and often finds
            # partial bodies the box detector rejects — without this, persons=0
            # while skeleton renders on screen.
            if n_persons == 0 and person_poses:
                from workbench_logic import synthesize_person_from_pose
                synthesized = False
                for pp in person_poses:
                    synth = synthesize_person_from_pose(pp, fw, fh)
                    if synth is None:
                        continue
                    yolo_detections.append(synth)
                    synthesized = True
                    if primary_person is None and synth.is_primary:
                        primary_person = synth
                if synthesized:
                    n_persons = count_person_detections(yolo_detections)
                    if primary_person is None:
                        for d in yolo_detections:
                            if d.name == "person":
                                d.is_primary = True
                                primary_person = d
                                break

            if primary_pose is not None and primary_pose.keypoints_xy is not None:
                pose_state = pose_state_tracker.update(
                    primary_pose.keypoints_xy,
                    primary_pose.keypoints_conf,
                    dt,
                    frame_h=fh,
                )
            else:
                pose_state = pose_state_tracker.update(None, None, dt, frame_h=fh)

            # Baby-mode persistent lock: feed every frame, get back the locked
            # bbox (may carry across YOLO drops up to hold_seconds). When a
            # lock is held but YOLO didn't detect this frame, synth the lock
            # into yolo_detections so downstream rendering shows it.
            baby_lock = None
            if baby_tracker is not None:
                person_evidence = [
                    ((d.x1, d.y1, d.x2, d.y2), float(d.confidence))
                    for d in yolo_detections if d.name == "person"
                ]
                baby_lock = baby_tracker.observe(
                    t0,
                    person_evidence,
                    posture=pose_state.posture.value,
                    motion=pose_state.motion.value,
                )
                # If we have a lock but no fresh YOLO box, synthesize one.
                if baby_lock is not None and not person_evidence:
                    from workbench_logic import YoloDetection as _YD
                    synth = _YD(
                        name="person",
                        confidence=float(baby_lock.confidence),
                        x1=baby_lock.x1, y1=baby_lock.y1, x2=baby_lock.x2, y2=baby_lock.y2,
                        in_roi=True,
                        near_hand=False,
                        role="person",
                        inside_person=False,
                        is_primary=True,
                        stable=True,
                        stable_age_s=float(baby_lock.age_s),
                    )
                    yolo_detections.append(synth)
                    n_persons = max(1, n_persons)
                    if primary_person is None:
                        primary_person = synth

            # bbox-shape fallback: when posture is UNKNOWN but a person box exists,
            # use the box aspect ratio. Tall = upright, wide = lying.
            if primary_person is not None and pose_state.posture.value == "unknown":
                bw = max(1, primary_person.x2 - primary_person.x1)
                bh = max(1, primary_person.y2 - primary_person.y1)
                aspect = bh / bw
                from pose_state import Posture as _P
                if aspect >= 1.5:
                    pose_state.posture = _P.UPRIGHT
                elif aspect <= 0.7:
                    pose_state.posture = _P.LYING
                # rebuild activity with the new posture (motion remains as classified)
                pose_state.activity = pose_state_tracker._activity(
                    pose_state.posture, pose_state.motion, pose_state.still_seconds
                )

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
            state = tracker.update(fs, dt)

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

            need_vis = (not args.no_show) or args.web
            if need_vis:
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
                    minimal=not show_explain_overlay(),
                )
                if args.web:
                    fps_now = 1.0 / dt if dt > 0 else 0.0
                    # In baby mode, the simplified 5-state activity from the
                    # tracker is what the dashboard / timeline cares about.
                    activity_value = (
                        baby_tracker.activity if baby_tracker is not None
                        else pose_state.activity
                    )
                    web_state = {
                        "state": state.value,
                        "activity": activity_value,
                        "camera_type": camera_type,
                        "baby_lock": (
                            None if baby_lock is None else {
                                "box": [baby_lock.x1, baby_lock.y1, baby_lock.x2, baby_lock.y2],
                                "confidence": baby_lock.confidence,
                                "age_s": baby_lock.age_s,
                                "age_since_seen_s": baby_lock.age_since_seen_s,
                            }
                        ),
                        "posture": pose_state.posture.value,
                        "motion": pose_state.motion.value,
                        "motion_score": float(pose_state.motion_score),
                        "still_seconds": float(pose_state.still_seconds),
                        "posture_angle_deg": float(pose_state.posture_angle_deg),
                        "medium_score": float(fs.medium_score),
                        "strict_score": float(fs.strict_score),
                        "person_count": int(n_persons),
                        "on_phone": bool(fs.on_phone),
                        "phone_near_hand": bool(fs.phone_near_hand),
                        "on_laptop": bool(fs.on_laptop),
                        "on_keyboard": bool(fs.on_keyboard),
                        "on_mouse": bool(fs.on_mouse),
                        "best_phone_iou_pct": float(fs.best_phone_wrist_iou_pct),
                        "best_laptop_iou_pct": float(fs.best_laptop_hand_iou_pct),
                        "best_keyboard_iou_pct": float(fs.best_keyboard_hand_iou_pct),
                        "best_mouse_iou_pct": float(fs.best_mouse_hand_iou_pct),
                        "work_streak_s": float(tracker.work_surface_streak),
                        "work_threshold_s": float(th["work_on_s"]),
                        "present_streak_s": float(tracker.present_streak),
                        "fps": fps_now,
                        "source": source if use_direct else f"frigate {camera}",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "detections": [
                            {
                                "name": d.name,
                                "role": d.role,
                                "confidence": float(d.confidence),
                                "stable": bool(d.stable),
                                "near_hand": bool(d.near_hand),
                                "is_primary": bool(d.is_primary),
                                "box": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                            }
                            for d in sorted(
                                yolo_detections, key=lambda x: -x.confidence
                            )[:20]
                        ],
                    }
                    web_server.publish(vis, web_state)
                if not args.no_show:
                    cv2.imshow("Workbench activity (explain)", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break

            if not use_direct:
                elapsed = time.perf_counter() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    if cap is not None:
        cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
