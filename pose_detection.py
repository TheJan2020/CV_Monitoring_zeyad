"""
Pose detection (MediaPipe).

**Frigate (default when FRIGATE_BASE_URL is in .env):**
  HTTP poll GET /api/{camera}/latest.jpg — same as frigate_viewer.py --transport http.

**Fallback:** VIDEO_SOURCE = webcam index or RTSP URL.

Run:
  python pose_detection.py
"""

from __future__ import annotations

import argparse
import os
import warnings

# Quieter console (MediaPipe / TFLite still work; these are informational only).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp

from env_settings import (
    get_frigate_base_url,
    get_frigate_camera,
    get_frigate_fps,
    resolve_video_source,
    use_frigate_http,
)
from frigate_http import latest_jpeg_url, poll_frames
from video_source import open_capture, read_loop


def run_frigate_http(*, base: str, camera: str, fps: float) -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    url = latest_jpeg_url(base, camera)
    print(f"Frigate HTTP poll: {url}")
    print("Press Q to quit.")

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        for frame in poll_frames(base, camera, fps=fps):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            rgb.flags.writeable = True
            output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            cv2.imshow("Pose detection (MediaPipe, Frigate HTTP)", output)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    cv2.destroyAllWindows()


def run_direct_source(*, source: str) -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    print(f"Direct video source: {source!r}")
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source!r}")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:
        print("Press Q to quit.")
        for frame in read_loop(cap):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            rgb.flags.writeable = True
            output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            cv2.imshow("Pose detection (MediaPipe)", output)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose detection with MediaPipe.")
    parser.add_argument("--source", default="", help="Override VIDEO_SOURCE (direct RTSP/webcam).")
    parser.add_argument("--base-url", default="", help="Override FRIGATE_BASE_URL.")
    parser.add_argument("--camera", default="", help="Override FRIGATE_CAMERA.")
    parser.add_argument("--fps", type=float, default=0.0, help="Frigate HTTP poll rate.")
    parser.add_argument("--direct", action="store_true", help="Force direct VIDEO_SOURCE.")
    args = parser.parse_args()

    base = (args.base_url or get_frigate_base_url()).strip().rstrip("/")
    camera = (args.camera or get_frigate_camera()).strip()
    fps = args.fps if args.fps > 0 else get_frigate_fps()

    if not args.direct and (base or use_frigate_http()):
        if not base:
            raise SystemExit("Set FRIGATE_BASE_URL in .env.")
        if not camera:
            raise SystemExit("Set FRIGATE_CAMERA in .env.")
        run_frigate_http(base=base, camera=camera, fps=fps)
        return

    source = (args.source or resolve_video_source()).strip()
    run_direct_source(source=source)


if __name__ == "__main__":
    main()
