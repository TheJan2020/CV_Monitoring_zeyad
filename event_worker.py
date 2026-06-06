"""Event-driven camera worker.

Polls an RTSP camera at low fps, runs YOLO11s person detection on
every sampled frame, and saves a snapshot (image + DB row) whenever a
person appears. Designed for cameras whose job is to capture training
data and, in later phases, fire MQTT triggers when specific
conditions are met (baby in a defined region while the door is open).

Crucially **separate from workbench_activity.py** — this file is the
only thing baby-camera workers couldn't break. The hub dispatches to
us when a camera's config has ``type: "event"``.

Lifecycle:
    cv_hub.py spawns ``python event_worker.py --web --no-show
    --web-host 0.0.0.0 --web-port <port> --direct`` with CAMERA_ID
    + VIDEO_SOURCE in the env. We read RTSP via OpenCV, run YOLO at
    EVENT_TARGET_FPS (default 2), and debounce snapshot saves by
    EVENT_DEBOUNCE_S (default 30 s) so a person standing still for a
    minute doesn't fill the disk with near-identical frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


_state_lock = threading.Lock()
_state: dict = {
    "alive": False,
    "camera_id": "",
    "rtsp": "",
    "last_capture_ts": 0.0,
    "last_person": False,
    "last_person_count": 0,
    "snapshots_today": 0,
    "category": "event",
}


def _start_web(host: str, port: int) -> None:
    """Tiny Flask app the hub healthz pings to know we're alive.
    Runs in a daemon thread so the main loop never blocks on the
    network."""
    try:
        from flask import Flask, jsonify
    except ImportError:
        print("[event_worker] flask not available — skipping web server",
              file=sys.stderr)
        return
    app = Flask(__name__)

    @app.route("/state")
    def _state_route():
        with _state_lock:
            return jsonify(dict(_state))

    @app.route("/healthz")
    def _healthz_route():
        return jsonify({"ok": True})

    def _runner():
        # threaded=True so simultaneous /state pings don't queue, and
        # use_reloader=False because we manage the lifecycle from the
        # parent script.
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    threading.Thread(target=_runner, name="event-worker-web", daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    # Mirror workbench_activity.py's CLI shape so the hub spawn args
    # work unchanged. We just ignore the ones that don't apply.
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, required=True)
    ap.add_argument("--direct", action="store_true")
    args, _unknown = ap.parse_known_args()

    cam_id = os.environ.get("CAMERA_ID") or ""
    rtsp = os.environ.get("VIDEO_SOURCE") or ""
    if not cam_id or not rtsp:
        print("[event_worker] missing CAMERA_ID or VIDEO_SOURCE env",
              file=sys.stderr)
        return 1

    target_fps = _env_float("EVENT_TARGET_FPS", 2.0)
    debounce_s = _env_float("EVENT_DEBOUNCE_S", 30.0)
    person_conf = _env_float("EVENT_PERSON_CONF", 0.40)
    imgsz = _env_int("EVENT_IMGSZ", 640)
    interval = 1.0 / max(0.1, target_fps)

    with _state_lock:
        _state["camera_id"] = cam_id
        _state["rtsp"] = rtsp
        _state["alive"] = True

    if args.web:
        _start_web(args.web_host, args.web_port)
    print(f"[event_worker] {cam_id} starting — RTSP={rtsp} fps={target_fps} "
          f"debounce={debounce_s}s conf>={person_conf}")

    # Lazy import so the web server can come up first (the hub's
    # healthz takes about a second to ping after spawn).
    from ultralytics import YOLO
    import state_recorder
    state_recorder.init()

    model = YOLO("yolo11s.pt")

    def _open_capture():
        c = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        # Reduce buffering so we read the freshest frame.
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return c

    cap = _open_capture()
    if not cap.isOpened():
        print(f"[event_worker] ERROR: could not open RTSP {rtsp}",
              file=sys.stderr)
        return 2

    last_snap_ts = 0.0
    consecutive_read_failures = 0

    while True:
        loop_t = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_read_failures += 1
            if consecutive_read_failures >= 5:
                print(f"[event_worker] {cam_id} RTSP read failed {consecutive_read_failures}× — reconnecting",
                      file=sys.stderr)
                try: cap.release()
                except Exception: pass
                time.sleep(2.0)
                cap = _open_capture()
                consecutive_read_failures = 0
            else:
                time.sleep(0.3)
            continue
        consecutive_read_failures = 0

        # Person-only inference at modest imgsz keeps GPU contention low
        # even when cam_1 is running at 10 fps next door.
        results = model.predict(
            source=frame, conf=person_conf, classes=[0],
            imgsz=imgsz, verbose=False, device=0,
        )
        boxes = results[0].boxes if results else None
        person_count = int(len(boxes)) if boxes is not None else 0
        person_detected = person_count > 0

        with _state_lock:
            _state["last_person"] = person_detected
            _state["last_person_count"] = person_count

        if person_detected and (loop_t - last_snap_ts) >= debounce_s:
            try:
                _save_snapshot(state_recorder, cam_id, frame, person_count, rtsp)
                last_snap_ts = loop_t
                with _state_lock:
                    _state["last_capture_ts"] = loop_t
                    _state["snapshots_today"] = _state.get("snapshots_today", 0) + 1
            except Exception as e:
                print(f"[event_worker] snapshot save failed: {e}",
                      file=sys.stderr)

        elapsed = time.time() - loop_t
        sleep = max(0.0, interval - elapsed)
        if sleep:
            time.sleep(sleep)


def _save_snapshot(state_recorder, cam_id: str, frame, person_count: int,
                   rtsp: str) -> None:
    """Write the BGR frame to disk under config/snapshots/<cam>/<day>/
    and insert a row in the snapshots table — matching the shape
    state_recorder.take_snapshot uses so the same /api/snapshots routes
    just work."""
    now = time.time()
    dt = datetime.fromtimestamp(now)
    day = dt.strftime("%Y-%m-%d")
    base = dt.strftime("%H%M%S")
    folder = state_recorder.SNAPSHOTS_DIR / cam_id / day
    folder.mkdir(parents=True, exist_ok=True)
    jpg_path = folder / f"{base}.jpg"
    if not cv2.imwrite(str(jpg_path), frame):
        raise RuntimeError(f"cv2.imwrite failed for {jpg_path}")

    state_obj = {
        # activity is the field the labeling UI reads to know whether
        # the snapshot represents a 'baby in crib' moment. For an event
        # camera nothing here is binary — what we really mean is 'a
        # person appeared'. Using a distinct activity value so the
        # baby-cam summary / accuracy stats never try to attribute
        # in-bed time to this camera.
        "activity": "person_detected",
        "person_count": person_count,
        "camera_type": "event",
        "source": rtsp,
        "captured_by": "event_worker",
        "timestamp": dt.strftime("%H:%M:%S"),
    }
    file_rel = f"{cam_id}/{day}/{base}.jpg"
    with state_recorder._LOCK:
        db = state_recorder._open()
        db.execute(
            "INSERT INTO snapshots (camera_id, captured_at, file_rel, state_json) "
            "VALUES (?, ?, ?, ?)",
            (cam_id, now, file_rel, json.dumps(state_obj)),
        )


if __name__ == "__main__":
    sys.exit(main())
