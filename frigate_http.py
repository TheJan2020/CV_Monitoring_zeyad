"""
Frigate frames over HTTP — same contract as ``frigate_viewer.py`` / FRIGATE_CONNECTION.md.

  GET {FRIGATE_BASE_URL}/api/{camera}/latest.jpg

No MQTT. No direct RTSP to the camera.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


def latest_jpeg_url(base_url: str, camera: str, *, height: int | None = None) -> str:
    base = base_url.rstrip("/")
    url = f"{base}/api/{camera}/latest.jpg"
    if height is not None:
        url += f"?h={int(height)}"
    return url


def fetch_latest_jpeg(
    base_url: str,
    camera: str,
    *,
    timeout_s: float = 10.0,
    height: int | None = None,
) -> bytes:
    """GET latest.jpg; returns raw JPEG bytes (raises on HTTP errors)."""
    url = latest_jpeg_url(base_url, camera, height=height)
    req = urllib.request.Request(url, headers={"User-Agent": "cv-poc-frigate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def decode_jpeg(data: bytes) -> Any | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def fetch_latest_frame(
    base_url: str,
    camera: str,
    *,
    timeout_s: float = 10.0,
    height: int | None = None,
) -> Any:
    data = fetch_latest_jpeg(base_url, camera, timeout_s=timeout_s, height=height)
    frame = decode_jpeg(data)
    if frame is None:
        raise ValueError("Frigate returned bytes that are not a valid JPEG.")
    return frame


def error_placeholder(text: str, *, w: int = 480, h: int = 240) -> Any:
    buf = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(buf, text[:40], (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return buf


def poll_frames(
    base_url: str,
    camera: str,
    *,
    fps: float = 2.0,
    timeout_s: float = 10.0,
):
    """
    Yield BGR frames from Frigate HTTP poll (like ``frigate_viewer.run_http_latest``).
    On fetch errors, yields a placeholder frame and continues.
    """
    interval = max(0.05, 1.0 / max(0.1, fps))
    while True:
        t0 = time.monotonic()
        try:
            data = fetch_latest_jpeg(base_url, camera, timeout_s=timeout_s)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.reason}", flush=True)
            yield error_placeholder(f"HTTP {e.code}")
        except Exception as e:
            print(f"Fetch error: {e}", flush=True)
            yield error_placeholder("No image")
        else:
            img = decode_jpeg(data)
            if img is None:
                print("Could not decode JPEG", flush=True)
                yield error_placeholder("Bad JPEG")
            else:
                yield img

        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
