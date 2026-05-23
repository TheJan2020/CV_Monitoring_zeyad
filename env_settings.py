"""Load `.env` and resolve settings for the CV scripts."""

from __future__ import annotations

import os
from urllib.parse import quote

from dotenv import load_dotenv

# Load once when this module is imported by the demos.
load_dotenv()


def resolve_video_source() -> str:
    """
    VIDEO_SOURCE: webcam index (e.g. 0) or full RTSP/HTTP URL.

    If VIDEO_SOURCE is unset or blank, builds from FRIGATE_RTSP_* vars when
    FRIGATE_RTSP_HOST and FRIGATE_STREAM_NAME are set.
    """
    direct = (os.getenv("VIDEO_SOURCE") or "").strip()
    if direct:
        return direct

    host = (os.getenv("FRIGATE_RTSP_HOST") or "").strip()
    stream = (os.getenv("FRIGATE_STREAM_NAME") or "").strip()
    if host and stream:
        port = (os.getenv("FRIGATE_RTSP_PORT") or "8554").strip() or "8554"
        user = (os.getenv("FRIGATE_RTSP_USER") or "").strip()
        password = (os.getenv("FRIGATE_RTSP_PASSWORD") or "").strip()
        if user and password:
            u = quote(user, safe="")
            p = quote(password, safe="")
            return f"rtsp://{u}:{p}@{host}:{port}/{stream}"
        return f"rtsp://{host}:{port}/{stream}"

    return "0"


def get_yolo_model() -> str:
    return (os.getenv("YOLO_MODEL") or "yolo11s.pt").strip() or "yolo11s.pt"


def get_yolo_pose_model() -> str:
    """YOLO11 pose model — person keypoints (multi-person, same pass as detection)."""
    return (os.getenv("YOLO_POSE_MODEL") or "yolo11s-pose.pt").strip() or "yolo11s-pose.pt"


def use_yolo_pose() -> bool:
    """Use YOLO11-pose for skeletons instead of MediaPipe on crops."""
    return (os.getenv("USE_YOLO_POSE") or "1").strip().lower() not in ("0", "false", "no")


def get_yolo_pose_conf() -> float:
    return _float_env("YOLO_POSE_CONF", 0.25)


def get_wrist_box_half_frac() -> float:
    """Fallback hand box half-size as fraction of min(frame width, height)."""
    return _float_env("WRIST_BOX_HALF_FRAC", 0.022)


def get_hand_box_forearm_scale() -> float:
    """Hand box half-size ≈ this × elbow–wrist distance (smaller = tighter on hand)."""
    return _float_env("HAND_BOX_FOREARM_SCALE", 0.38)


def get_hand_box_extend_frac() -> float:
    """Shift hand box center past wrist toward fingers (× forearm length)."""
    return _float_env("HAND_BOX_EXTEND_FRAC", 0.22)


def get_wrist_phone_min_iou_pct() -> float:
    """Min IoU%% (hand∩phone / union) to count phone as in-hand."""
    return _float_env("WRIST_PHONE_MIN_IOU_PCT", 5.0)


def get_wrist_phone_min_wrist_cov_pct() -> float:
    """Min %% of hand box covered by phone to count as in-hand."""
    return _float_env("WRIST_PHONE_MIN_WRIST_COV_PCT", 8.0)


def get_wrist_phone_min_phone_cov_pct() -> float:
    """Min %% of phone box overlapping a hand region."""
    return _float_env("WRIST_PHONE_MIN_PHONE_COV_PCT", 10.0)


def get_wrist_laptop_min_iou_pct() -> float:
    """Min IoU%% (hand∩laptop / union) to count laptop as at hands."""
    return _float_env("WRIST_LAPTOP_MIN_IOU_PCT", 10.0)


def get_wrist_laptop_min_hand_cov_pct() -> float:
    """Min %% of hand box covered by laptop."""
    return _float_env("WRIST_LAPTOP_MIN_HAND_COV_PCT", 15.0)


def get_wrist_laptop_min_laptop_cov_pct() -> float:
    """Min %% of laptop box overlapping a hand region."""
    return _float_env("WRIST_LAPTOP_MIN_LAPTOP_COV_PCT", 12.0)


def get_laptop_min_conf() -> float:
    """Min confidence to tag laptop as laptop_hands (overlaps primary hands)."""
    return _float_env("LAPTOP_MIN_CONF", 0.20)


def get_phone_stable_seconds() -> float:
    """Seconds a phone must be seen at ≥ PHONE_STABLE_MIN_CONF before it counts."""
    return _float_env("PHONE_STABLE_SECONDS", 1.5)


def get_phone_stable_min_conf() -> float:
    """Per-frame confidence required to extend the phone stability streak."""
    return _float_env("PHONE_STABLE_MIN_CONF", 0.35)


def get_laptop_stable_seconds() -> float:
    """Seconds a laptop must be seen at ≥ LAPTOP_STABLE_MIN_CONF before it counts."""
    return _float_env("LAPTOP_STABLE_SECONDS", 3.0)


def get_laptop_stable_min_conf() -> float:
    """Per-frame confidence required to extend the laptop stability streak."""
    return _float_env("LAPTOP_STABLE_MIN_CONF", 0.5)


def get_wrist_keyboard_min_iou_pct() -> float:
    """Min IoU%% (hand∩keyboard / union) to count keyboard as at hands."""
    return _float_env("WRIST_KEYBOARD_MIN_IOU_PCT", 10.0)


def get_wrist_keyboard_min_hand_cov_pct() -> float:
    """Min %% of hand box covered by keyboard."""
    return _float_env("WRIST_KEYBOARD_MIN_HAND_COV_PCT", 15.0)


def get_wrist_keyboard_min_keyboard_cov_pct() -> float:
    """Min %% of keyboard box overlapping a hand region."""
    return _float_env("WRIST_KEYBOARD_MIN_KEYBOARD_COV_PCT", 12.0)


def get_keyboard_min_conf() -> float:
    """Min confidence to tag keyboard as keyboard_hands (overlaps primary hands)."""
    return _float_env("KEYBOARD_MIN_CONF", 0.20)


def get_keyboard_stable_seconds() -> float:
    """Seconds a keyboard must be seen at ≥ KEYBOARD_STABLE_MIN_CONF before it counts."""
    return _float_env("KEYBOARD_STABLE_SECONDS", 3.0)


def get_keyboard_stable_min_conf() -> float:
    """Per-frame confidence required to extend the keyboard stability streak."""
    return _float_env("KEYBOARD_STABLE_MIN_CONF", 0.5)


def get_yolo_conf() -> float:
    return _float_env("YOLO_CONF", 0.12)


def get_yolo_person_conf() -> float:
    """Lower threshold for class person only (COCO id 0) — seated/distant people."""
    return _float_env("YOLO_PERSON_CONF", 0.02)


def use_pose() -> bool:
    """MediaPipe skeleton + hands (alongside YOLO boxes)."""
    return (os.getenv("USE_POSE") or "1").strip().lower() not in ("0", "false", "no")


def get_max_pose_persons() -> int:
    """Max skeletons per frame (one MediaPipe pass per YOLO person crop)."""
    return max(1, int(_float_env("MAX_POSE_PERSONS", 8)))


def get_pose_min_detection_conf() -> float:
    """MediaPipe pose min_detection_confidence (lower = more skeleton attempts)."""
    return _float_env("POSE_MIN_DETECTION_CONF", 0.3)


def get_pose_crop_pad_frac() -> float:
    """Extra padding around each YOLO person box before running pose (fraction of box size)."""
    return _float_env("POSE_CROP_PAD", 0.25)


def show_all_coco_tools() -> bool:
    """When true, label all COCO tool-like classes (not only the short TOOL_CLASSES list)."""
    return (os.getenv("SHOW_ALL_TOOLS") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def get_phone_min_conf() -> float:
    """Min confidence to count as on-phone (ignores 0.09 lap false positives)."""
    return _float_env("PHONE_MIN_CONF", 0.25)


def get_yolo_phone_model() -> str:
    """Model for phone passes (defaults to YOLO_MODEL; yolo11m.pt helps small phones)."""
    return (os.getenv("YOLO_PHONE_MODEL") or get_yolo_model()).strip() or get_yolo_model()


def get_yolo_phone_conf() -> float:
    """Confidence for YOLO class-67 (cell phone) dedicated pass."""
    return _float_env("YOLO_PHONE_CONF", 0.03)


def get_yolo_phone_imgsz() -> int:
    """Inference size for phone passes (full-frame + hand-crop; lower = faster)."""
    return int(_float_env("YOLO_PHONE_IMGSZ", 960))


def _use_env_flag(name: str, *, default_on: bool = True) -> bool:
    """True unless env is 0 / false / no."""
    default = "1" if default_on else "0"
    return (os.getenv(name) or default).strip().lower() not in ("0", "false", "no")


def use_yolo_people_pass() -> bool:
    """Pass 1: YOLO person boxes + pose (skeleton / wrists). Set 0 to skip."""
    return _use_env_flag("USE_YOLO_PEOPLE_PASS")


def use_yolo_devices_pass() -> bool:
    """Pass 2: one YOLO run for phone/laptop/keyboard/mouse. Set 0 to skip all devices."""
    return _use_env_flag("USE_YOLO_DEVICES_PASS")


def use_yolo_phone_pass() -> bool:
    """Include cell phone in devices pass + phone overlap/stability. Set 0 to skip phone only."""
    return _use_env_flag("USE_YOLO_PHONE_PASS") and use_yolo_devices_pass()


def use_yolo_laptop_pass() -> bool:
    """Include laptop in devices pass. Set 0 to skip laptop only."""
    return _use_env_flag("USE_YOLO_LAPTOP_PASS") and use_yolo_devices_pass()


def use_yolo_keyboard_pass() -> bool:
    """Include keyboard in devices pass. Set 0 to skip keyboard only."""
    return _use_env_flag("USE_YOLO_KEYBOARD_PASS") and use_yolo_devices_pass()


def use_yolo_mouse_pass() -> bool:
    """Include mouse in devices pass. Set 0 to skip mouse only."""
    return _use_env_flag("USE_YOLO_MOUSE_PASS") and use_yolo_devices_pass()


def _device_pass_enabled(name: str) -> bool:
    """Per-device flag without requiring USE_YOLO_DEVICES_PASS (for class-id list)."""
    return _use_env_flag(name)


def enabled_work_device_class_ids() -> list[int]:
    """COCO class ids for the unified devices pass (empty if pass off or all devices off)."""
    if not use_yolo_devices_pass():
        return []
    ids: list[int] = []
    if _device_pass_enabled("USE_YOLO_PHONE_PASS"):
        ids.append(67)
    if _device_pass_enabled("USE_YOLO_LAPTOP_PASS"):
        ids.append(63)
    if _device_pass_enabled("USE_YOLO_KEYBOARD_PASS"):
        ids.append(66)
    if _device_pass_enabled("USE_YOLO_MOUSE_PASS"):
        ids.append(64)
    return ids


def needs_device_yolo_model() -> bool:
    """True if devices pass runs (loads YOLO_PHONE_MODEL / yolo11m)."""
    return bool(enabled_work_device_class_ids())


def get_yolo_devices_conf() -> float:
    """Confidence for unified devices YOLO pass (min of enabled per-class confs)."""
    raw = (os.getenv("YOLO_DEVICES_CONF") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    confs: list[float] = []
    if _device_pass_enabled("USE_YOLO_PHONE_PASS"):
        confs.append(get_yolo_phone_conf())
    if _device_pass_enabled("USE_YOLO_LAPTOP_PASS"):
        confs.append(get_yolo_laptop_conf())
    if _device_pass_enabled("USE_YOLO_KEYBOARD_PASS"):
        confs.append(get_yolo_keyboard_conf())
    if _device_pass_enabled("USE_YOLO_MOUSE_PASS"):
        confs.append(get_yolo_mouse_conf())
    return min(confs) if confs else 0.12


def get_yolo_devices_imgsz() -> int:
    """Inference size for unified devices pass."""
    return int(_float_env("YOLO_DEVICES_IMGSZ", 960))


def enabled_detection_passes_label() -> str:
    """Which of the 4 work devices are enabled inside the devices pass."""
    parts: list[str] = []
    if _device_pass_enabled("USE_YOLO_PHONE_PASS"):
        parts.append("phone")
    if _device_pass_enabled("USE_YOLO_LAPTOP_PASS"):
        parts.append("laptop")
    if _device_pass_enabled("USE_YOLO_KEYBOARD_PASS"):
        parts.append("keyboard")
    if _device_pass_enabled("USE_YOLO_MOUSE_PASS"):
        parts.append("mouse")
    return ", ".join(parts) if parts else "none"


def pipeline_passes_label() -> str:
    """Short summary of active pipeline passes for startup log."""
    parts: list[str] = []
    if use_yolo_people_pass():
        pose = "YOLO-pose" if use_yolo_pose() and use_pose() else "pose off"
        parts.append(f"people ({pose})")
    if use_yolo_devices_pass() and enabled_work_device_class_ids():
        parts.append(f"devices [{enabled_detection_passes_label()}]")
    if show_all_coco_tools():
        parts.append("extra COCO tools")
    return " | ".join(parts) if parts else "none"


def get_yolo_laptop_conf() -> float:
    """Confidence for YOLO class-63 (laptop) dedicated pass."""
    return _float_env("YOLO_LAPTOP_CONF", 0.12)


def get_yolo_laptop_imgsz() -> int:
    """Inference size for laptop pass (lower = faster)."""
    return int(_float_env("YOLO_LAPTOP_IMGSZ", 960))


def get_yolo_keyboard_conf() -> float:
    """Confidence for YOLO class-66 (keyboard) dedicated pass."""
    return _float_env("YOLO_KEYBOARD_CONF", 0.12)


def get_yolo_keyboard_imgsz() -> int:
    """Inference size for keyboard pass (lower = faster)."""
    return int(_float_env("YOLO_KEYBOARD_IMGSZ", 960))


def get_wrist_mouse_min_iou_pct() -> float:
    return _float_env("WRIST_MOUSE_MIN_IOU_PCT", 10.0)


def get_wrist_mouse_min_hand_cov_pct() -> float:
    return _float_env("WRIST_MOUSE_MIN_HAND_COV_PCT", 15.0)


def get_wrist_mouse_min_mouse_cov_pct() -> float:
    return _float_env("WRIST_MOUSE_MIN_MOUSE_COV_PCT", 12.0)


def get_mouse_min_conf() -> float:
    return _float_env("MOUSE_MIN_CONF", 0.20)


def get_mouse_stable_seconds() -> float:
    return _float_env("MOUSE_STABLE_SECONDS", 3.0)


def get_mouse_stable_min_conf() -> float:
    return _float_env("MOUSE_STABLE_MIN_CONF", 0.5)


def get_yolo_mouse_conf() -> float:
    return _float_env("YOLO_MOUSE_CONF", 0.12)


def get_yolo_mouse_imgsz() -> int:
    return int(_float_env("YOLO_MOUSE_IMGSZ", 960))


def use_phone_hand_crop() -> bool:
    """Zoomed YOLO phone pass on hand regions (requires USE_YOLO_PHONE_PASS=1)."""
    return _use_env_flag("USE_PHONE_HAND_CROP") and use_yolo_phone_pass()


def get_phone_hand_crop_pad() -> float:
    """Padding around union of primary hand boxes for phone zoom pass."""
    return _float_env("PHONE_HAND_CROP_PAD", 0.40)


def phone_require_primary() -> bool:
    """If true, in-hand phones must overlap primary person / their hand boxes."""
    return (os.getenv("PHONE_REQUIRE_PRIMARY") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def get_yolo_iou() -> float:
    """NMS IoU — lower merges more duplicate boxes on the same person."""
    return _float_env("YOLO_IOU", 0.45)


def get_yolo_imgsz() -> int:
    return int(_float_env("YOLO_IMGSZ", 1280))


def get_frigate_base_url() -> str:
    """``FRIGATE_BASE_URL`` preferred (matches other tools); ``FRIGATE_HTTP_URL`` as alias."""
    u = (os.getenv("FRIGATE_BASE_URL") or os.getenv("FRIGATE_HTTP_URL") or "").strip()
    return u.rstrip("/")


def get_frigate_camera() -> str:
    return (os.getenv("FRIGATE_CAMERA") or os.getenv("FRIGATE_CAMERA_NAME") or "workshop").strip() or "workshop"


def get_frigate_fps() -> float:
    raw = (os.getenv("FRIGATE_HTTP_FPS") or os.getenv("FRIGATE_FPS") or "2").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 2.0


def use_frigate_http() -> bool:
    """True when Frigate HTTP should be used (FRIGATE_BASE_URL set and not disabled)."""
    if (os.getenv("USE_FRIGATE_HTTP") or "").strip().lower() in ("0", "false", "no"):
        return False
    return bool(get_frigate_base_url())


def log_verbose() -> bool:
    """Print full per-frame pipeline logs (all stages, persons, hands, scores)."""
    return _use_env_flag("LOG_VERBOSE", default_on=True)


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_workbench_roi() -> "BenchROI":
    from workbench_logic import BenchROI

    raw = (os.getenv("WORKBENCH_ROI") or "0,0,1,1").strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return BenchROI()
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
        return BenchROI(
            x1=min(x1, x2),
            y1=min(y1, y2),
            x2=max(x1, x2),
            y2=max(y1, y2),
        )
    except ValueError:
        return BenchROI()


def get_workbench_thresholds() -> dict[str, float]:
    return {
        "work_on_s": _float_env("WORK_SECONDS", 2.5),
        "work_off_s": _float_env("WORK_OFF_SECONDS", 2.0),
        "present_on_s": _float_env("PRESENT_SECONDS", 1.0),
        "medium_threshold": _float_env("SCORE_MEDIUM", 0.50),
        "strict_threshold": _float_env("SCORE_STRICT", 0.62),
        "present_threshold": _float_env("SCORE_PRESENT", 0.28),
    }
