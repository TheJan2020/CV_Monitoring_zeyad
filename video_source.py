"""Open a webcam index or a network stream (RTSP/HTTP) for OpenCV."""

from __future__ import annotations

import os
import time

import cv2


def _is_url(source: str) -> bool:
    s = source.strip().lower()
    return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://")


def _configure_ffmpeg_for_stream(source: str) -> None:
    """RTSP over TCP + timeouts; must run before VideoCapture for this process."""
    if not _is_url(source):
        return
    # Semicolon-separated key;value pairs for FFmpeg in OpenCV 4.x
    opts = os.getenv(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts


def open_capture(source: str) -> cv2.VideoCapture:
    """
    source: webcam index as decimal digits, e.g. "0", or a URL such as
    rtsp://user:pass@192.168.1.50:8554/kitchen
    """
    s = source.strip()
    if s.isdigit():
        cap = cv2.VideoCapture(int(s))
    else:
        _configure_ffmpeg_for_stream(s)
        cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
        # Smaller buffer reduces stale frames on RTSP (ignored on some builds).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


def read_loop(
    cap: cv2.VideoCapture,
    *,
    max_consecutive_failures: int = 200,
    sleep_on_fail_s: float = 0.05,
):
    """
    Yield valid BGR frames until user stops or too many consecutive read failures.
    Prints one-line hints on total failure.
    """
    fails = 0
    while fails < max_consecutive_failures:
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            fails = 0
            yield frame
            continue
        fails += 1
        time.sleep(sleep_on_fail_s)

    print(
        f"No frames received after {max_consecutive_failures} attempts. "
        "Check: RTSP URL and password in .env, camera reachable from this PC, "
        "and try opening the same URL in VLC. For RTSP, TCP transport is enabled by default."
    )
