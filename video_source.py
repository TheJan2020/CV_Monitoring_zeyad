"""Open a webcam index, a network stream (RTSP/HTTP), or a raw-frame
pipe from a co-spawned FramePump for OpenCV-compatible reading."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from frame_pump import FramePump


def _is_url(source: str) -> bool:
    s = source.strip().lower()
    return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://")


def _is_pipe(source: str) -> bool:
    """A pipe source spec is ``pipe:<W>:<H>`` — geometry of raw BGR frames
    that arrive on the FramePump's stdout. Parsed by :class:`PipeCapture`."""
    return source.strip().lower().startswith("pipe:")


class PipeCapture:
    """cv2.VideoCapture-shaped reader over a co-spawned FramePump.

    A background **drainer thread** continuously reads frames from the
    FramePump's stdout and keeps only the most recent one in memory.
    ``read()`` blocks (with timeout) until a NEW frame arrives, then
    returns it. Older buffered frames are dropped silently.

    Why this matters: the worker drains slower (YOLO ~6 fps) than ffmpeg
    produces (10 fps cap). Without the drainer, frames pile up in the OS
    pipe buffer, ffmpeg blocks on the pipe write, and the HLS output
    (which shares the same ffmpeg decode loop) stops getting fed. With
    the drainer, ffmpeg never blocks and the HLS cadence stays smooth.

    A short read on the pipe means ffmpeg died — the drainer exits and
    subsequent ``read()`` calls return ``(False, None)``. The caller's
    read_loop trips its failure budget, the worker exits, and the hub
    watchdog respawns the unit (worker + FramePump together).
    """

    def __init__(self, pump: FramePump) -> None:
        self.pump = pump
        self._frame_bytes = pump.frame_bytes
        self._h = pump.video_h
        self._w = pump.video_w
        self._stdout = pump.stdout
        if self._stdout is None:
            raise RuntimeError("FramePump has no stdout — call start() first")

        # Drainer state: latest decoded frame + monotonic ID so read()
        # can block-until-fresh. Condition variable lets the drainer
        # notify a waiting read() without polling.
        self._latest_frame = None
        self._frame_id = 0
        self._last_read_id = 0
        self._stop = threading.Event()
        self._cond = threading.Condition()

        self._drainer = threading.Thread(
            target=self._drain_loop,
            name=f"pipe-drainer-{pump.camera_id}",
            daemon=True,
        )
        self._drainer.start()

    def _drain_loop(self) -> None:
        try:
            while not self._stop.is_set():
                buf = bytearray()
                need = self._frame_bytes
                while need > 0:
                    chunk = self._stdout.read(need)
                    if not chunk:
                        # ffmpeg's stdout closed → end the drainer; read()
                        # will see pump not alive and return (False, None).
                        return
                    buf.extend(chunk)
                    need -= len(chunk)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    (self._h, self._w, 3),
                )
                with self._cond:
                    self._latest_frame = frame
                    self._frame_id += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API parity)
        return self.pump.alive() and self._drainer.is_alive()

    def read(self, timeout: float = 5.0):
        """Block until a new frame is available (or ``timeout`` elapses /
        the pump dies). Returns ``(ok, ndarray | None)``."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: (
                    self._frame_id > self._last_read_id
                    or self._stop.is_set()
                    or not self._drainer.is_alive()
                ),
                timeout=timeout,
            )
            if (
                ok
                and self._frame_id > self._last_read_id
                and self._latest_frame is not None
            ):
                self._last_read_id = self._frame_id
                return True, self._latest_frame
        return False, None

    def release(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        try:
            self.pump.stop()
        except Exception:
            pass

    # cv2.VideoCapture exposes set/get for properties — most callers only
    # poke CAP_PROP_BUFFERSIZE; make it a no-op for the pipe.
    def set(self, *_args, **_kwargs) -> bool:
        return True

    def get(self, *_args, **_kwargs) -> float:
        return 0.0


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


def open_pipe_capture(
    camera_id: str,
    rtsp_url: str,
    hls_dir,
    video_w: int = 1280,
    video_h: int = 720,
    stderr_log_path=None,
) -> "PipeCapture":
    """Spawn a co-located FramePump and return a cv2-shaped capture over
    its raw-frame pipe. Use this instead of :func:`open_capture` when the
    camera is configured with ``use_frame_pump=true`` — gives 1 RTSP
    session per camera shared between worker + clips + audio."""
    # Local import so workers that don't use the pump don't pay the
    # frame_pump import cost (and don't crash if imageio_ffmpeg is missing).
    from frame_pump import FramePump
    pump = FramePump(
        camera_id=camera_id,
        rtsp_url=rtsp_url,
        hls_dir=hls_dir,
        video_w=video_w,
        video_h=video_h,
        stderr_log_path=stderr_log_path,
    )
    pump.start()
    return PipeCapture(pump)


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
