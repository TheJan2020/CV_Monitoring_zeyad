"""
Object detection (YOLO).

**Frigate (default when FRIGATE_BASE_URL is in .env):**
  HTTP poll GET /api/{camera}/latest.jpg — same as frigate_viewer.py --transport http.

**Fallback:** VIDEO_SOURCE = webcam index or RTSP URL.

Run:
  python object_detection.py
"""

from __future__ import annotations

import argparse

import cv2
from ultralytics import YOLO

from env_settings import (
    get_frigate_base_url,
    get_frigate_camera,
    get_frigate_fps,
    get_yolo_model,
    resolve_video_source,
    use_frigate_http,
)
from frigate_http import latest_jpeg_url, poll_frames
from video_source import open_capture, read_loop


def run_frigate_http(*, model_name: str, base: str, camera: str, fps: float) -> None:
    model = YOLO(model_name)
    url = latest_jpeg_url(base, camera)
    print(f"Frigate HTTP poll: {url}")
    print("Press Q to quit.")

    for frame in poll_frames(base, camera, fps=fps):
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Object detection (YOLO, Frigate HTTP)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cv2.destroyAllWindows()


def run_direct_source(*, model_name: str, source: str) -> None:
    if "CHANGE_ME" in source:
        print(
            "Warning: VIDEO_SOURCE contains CHANGE_ME. For Frigate, set FRIGATE_BASE_URL "
            "and FRIGATE_CAMERA in .env and leave VIDEO_SOURCE unset."
        )

    print(f"Direct video source: {source!r}")
    model = YOLO(model_name)
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source!r}")

    print("Press Q to quit.")
    for frame in read_loop(cap):
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Object detection (YOLO)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection with YOLOv8.")
    parser.add_argument("--model", default=get_yolo_model())
    parser.add_argument("--source", default="", help="Override VIDEO_SOURCE (direct RTSP/webcam).")
    parser.add_argument("--base-url", default="", help="Override FRIGATE_BASE_URL.")
    parser.add_argument("--camera", default="", help="Override FRIGATE_CAMERA.")
    parser.add_argument("--fps", type=float, default=0.0, help="Frigate HTTP poll rate (default: FRIGATE_HTTP_FPS).")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Force direct VIDEO_SOURCE even if FRIGATE_BASE_URL is set.",
    )
    args = parser.parse_args()

    base = (args.base_url or get_frigate_base_url()).strip().rstrip("/")
    camera = (args.camera or get_frigate_camera()).strip()
    fps = args.fps if args.fps > 0 else get_frigate_fps()

    if not args.direct and (base or use_frigate_http()):
        if not base:
            raise SystemExit("Set FRIGATE_BASE_URL in .env (e.g. http://192.168.100.42:5000).")
        if not camera:
            raise SystemExit("Set FRIGATE_CAMERA in .env (e.g. workshop).")
        run_frigate_http(model_name=args.model, base=base, camera=camera, fps=fps)
        return

    source = (args.source or resolve_video_source()).strip()
    run_direct_source(model_name=args.model, source=source)


if __name__ == "__main__":
    main()
