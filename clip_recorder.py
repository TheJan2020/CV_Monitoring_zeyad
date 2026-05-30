"""Per-camera ffmpeg HLS ring buffer + on-demand clip extraction.

A ``ClipBuffer`` continuously demuxes a camera's RTSP stream into a ring of
2-second .ts segments (video stream-copied, audio re-encoded to AAC mono to
keep the timestamp issues out of the segmenter). The state recorder calls
``extract_clip(anchor_ts, before_s, after_s)`` after enough wall-clock time
has elapsed; the buffer locates segments whose mtime range covers the
requested window, concats them via ``ffmpeg -f concat -c copy``, and writes
an MP4 next to the snapshot. No re-encoding on extraction.

``ClipBufferManager.update(cameras)`` reconciles the desired set of buffers
against the live config (only cameras with ``enabled=True`` *and*
``save_history=True`` get a buffer). Idempotent — safe to call after every
PATCH/POST/DELETE on a camera.

Limits / things to know:
  - Two RTSP sessions per camera (worker + clip buffer). Most modern IP
    cameras allow this. If yours doesn't, capping concurrent streams in
    the camera's config will let the worker win and the buffer error out.
  - Audio is included if the source has an audio track (Reolink AAC works).
  - ffmpeg comes from imageio-ffmpeg so no system install is required.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


SEGMENT_DURATION_S = 2  # length of each ring-buffer segment


def _ffmpeg_exe() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to system PATH


class ClipBuffer:
    """Continuous HLS-style ring buffer for one camera.

    Two modes:
      * ``borrowed=False`` (default): owns an ffmpeg subprocess that
        pulls RTSP and writes .ts segments. Legacy path.
      * ``borrowed=True``: the worker's :class:`frame_pump.FramePump`
        is already writing .ts segments to ``buf_dir`` from its single
        unified ffmpeg. We don't spawn anything — start/stop are no-ops
        and ``alive()`` reports True so ``extract_clip()`` is always
        callable. This is how the single-RTSP-session refactor avoids
        two ffmpegs writing the same directory.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        base_dir: Path,
        *,
        borrowed: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.buf_dir = (base_dir / "buffer" / camera_id).resolve()
        self.borrowed = borrowed
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------

    def start(self, total_buffer_s: int = 30) -> None:
        if self.borrowed:
            # FramePump owns the ffmpeg; just make sure the directory
            # exists in case extract_clip races with first-segment write.
            self.buf_dir.mkdir(parents=True, exist_ok=True)
            return
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return
            if self.buf_dir.exists():
                shutil.rmtree(self.buf_dir, ignore_errors=True)
            self.buf_dir.mkdir(parents=True, exist_ok=True)
            list_size = max(8, total_buffer_s // SEGMENT_DURATION_S + 2)
            m3u8 = self.buf_dir / "buf.m3u8"
            seg_pattern = str(self.buf_dir / "seg_%05d.ts")
            cmd = [
                _ffmpeg_exe(),
                "-hide_banner", "-loglevel", "warning",
                "-rtsp_transport", "tcp",
                "-fflags", "+genpts",
                "-use_wallclock_as_timestamps", "1",
                "-i", self.rtsp_url,
                "-c:v", "copy",
                # Re-encode audio so the muxer doesn't choke on RTP timestamp
                # jitter from Reolink (same fix we needed for /api/audio).
                "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "22050",
                "-af", "aresample=async=1000",
                "-f", "hls",
                "-hls_time", str(SEGMENT_DURATION_S),
                "-hls_list_size", str(list_size),
                "-hls_flags", "delete_segments+independent_segments+omit_endlist",
                "-hls_segment_filename", seg_pattern,
                str(m3u8),
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.proc = None

    def stop(self) -> None:
        if self.borrowed:
            # FramePump's lifecycle is owned by the worker; the worker
            # is responsible for tearing down ffmpeg. We leave the
            # segments on disk so an in-flight extract_clip can still
            # complete; the worker will wipe them on its next start().
            return
        with self._lock:
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None
            shutil.rmtree(self.buf_dir, ignore_errors=True)

    def alive(self) -> bool:
        if self.borrowed:
            # As far as the recorder is concerned, a borrowed buffer is
            # always alive — the worker's FramePump runs continuously
            # whenever the worker process is up, with no presence gating.
            return True
        return bool(self.proc and self.proc.poll() is None)

    # --- extraction ------------------------------------------------------

    def extract_clip(
        self,
        output_path: Path,
        anchor_ts: float,
        before_s: int,
        after_s: int,
    ) -> bool:
        """Stitch the segments covering ``[anchor_ts - before_s,
        anchor_ts + after_s]`` into ``output_path``. Returns True on success.

        Caller must wait until at least ``after_s`` seconds of wall clock
        have elapsed since ``anchor_ts`` before invoking — otherwise the
        forward portion of the window won't be in the buffer yet.
        """
        with self._lock:
            if not self.buf_dir.exists():
                return False
            # mtime of each segment = time it was finished writing, so it
            # contains content from (mtime - SEGMENT_DURATION_S, mtime].
            segments = sorted(
                self.buf_dir.glob("seg_*.ts"),
                key=lambda p: p.stat().st_mtime,
            )
            if not segments:
                return False
            window_start = anchor_ts - before_s
            window_end = anchor_ts + after_s
            relevant: list[Path] = []
            for seg in segments:
                try:
                    seg_end = seg.stat().st_mtime
                except OSError:
                    continue
                seg_start = seg_end - SEGMENT_DURATION_S
                if seg_end < window_start or seg_start > window_end:
                    continue
                relevant.append(seg)
            if not relevant:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            concat_file = self.buf_dir / f".concat_{int(anchor_ts)}.txt"
            try:
                with concat_file.open("w") as f:
                    for s in relevant:
                        f.write(f"file '{s.name}'\n")
                cmd = [
                    _ffmpeg_exe(),
                    "-hide_banner", "-loglevel", "warning",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    "-movflags", "+faststart",
                    "-y",
                    str(output_path),
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=20,
                )
                ok = (
                    proc.returncode == 0
                    and output_path.exists()
                    and output_path.stat().st_size > 0
                )
                return ok
            except (subprocess.TimeoutExpired, OSError, Exception):
                return False
            finally:
                try:
                    concat_file.unlink()
                except OSError:
                    pass


class ClipBufferManager:
    """Tracks per-camera ``ClipBuffer`` instances. Reconciled against the
    current camera list on every config change AND idled-out based on
    person presence reported by the state recorder.

    ``update(cameras)`` only creates / drops buffer objects for cameras
    that should have one. It does not start them — that's driven by
    ``notice_presence(camera_id, has_person, clip_seconds)``, which the
    recorder calls on every /state poll:

      * ``has_person=True``  → start (or refresh) the buffer
      * ``has_person=False`` for > ``idle_pause_seconds``
        seconds → stop the buffer (free up the RTSP session)

    This keeps each baby camera at one RTSP session most of the day
    (just the worker), and only spins up a second session when there's
    actually someone to clip.
    """

    def __init__(self, base_dir: Path, idle_pause_seconds: float = 60.0) -> None:
        self.base_dir = base_dir
        self.idle_pause_seconds = idle_pause_seconds
        self._buffers: dict[str, ClipBuffer] = {}
        self._last_seen_with_person: dict[str, float] = {}
        self._lock = threading.Lock()

    # --- config reconciliation ------------------------------------------

    def update(self, cameras: list[dict]) -> None:
        """Reconcile buffer objects against the desired camera set.
        Does NOT start buffers — they remain paused until notice_presence
        is called with has_person=True."""
        with self._lock:
            wanted = {
                c["id"]: c
                for c in cameras
                if c.get("enabled", True) and c.get("save_history")
                and (c.get("rtsp_url") or "").strip()
            }
            # Stop and drop removed / disabled
            for cam_id in list(self._buffers.keys()):
                if cam_id not in wanted:
                    self._buffers[cam_id].stop()
                    self._buffers.pop(cam_id, None)
                    self._last_seen_with_person.pop(cam_id, None)
            # Ensure objects exist (paused) for kept cameras. A camera
            # using the unified FramePump gets a *borrowed* buffer that
            # just points at the pump's HLS directory — no second ffmpeg.
            for cam_id, cam in wanted.items():
                borrowed = bool(cam.get("use_frame_pump"))
                existing = self._buffers.get(cam_id)
                if existing is None:
                    new = ClipBuffer(
                        cam_id, cam["rtsp_url"], self.base_dir,
                        borrowed=borrowed,
                    )
                    if borrowed:
                        new.start()  # idempotent, just creates buf_dir
                    self._buffers[cam_id] = new
                elif existing.borrowed != borrowed:
                    # Mode flipped between borrowed and owned. Tear the
                    # old one down (no-op if borrowed) and rebuild.
                    existing.stop()
                    new = ClipBuffer(
                        cam_id, cam["rtsp_url"], self.base_dir,
                        borrowed=borrowed,
                    )
                    if borrowed:
                        new.start()
                    self._buffers[cam_id] = new

    # --- presence-driven start / pause -----------------------------------

    def notice_presence(
        self, camera_id: str, has_person: bool, clip_seconds: int,
    ) -> None:
        """Called by the state recorder on every /state poll."""
        now = _wall_time()
        with self._lock:
            buf = self._buffers.get(camera_id)
            if buf is None:
                return
            # Borrowed buffers don't get presence-gated — the FramePump
            # runs continuously alongside the worker, so the HLS is
            # always on. Nothing to start or stop here.
            if buf.borrowed:
                if has_person:
                    self._last_seen_with_person[camera_id] = now
                return
            if has_person:
                self._last_seen_with_person[camera_id] = now
                if not buf.alive():
                    buf.start(total_buffer_s=max(1, clip_seconds) * 2 + 4)
            else:
                last = self._last_seen_with_person.get(camera_id, 0.0)
                if buf.alive() and now - last > self.idle_pause_seconds:
                    buf.stop()

    # --- access ----------------------------------------------------------

    def get(self, camera_id: str) -> ClipBuffer | None:
        with self._lock:
            buf = self._buffers.get(camera_id)
            # Only return alive buffers — extract_clip on a paused buffer
            # would produce nothing useful.
            return buf if (buf is not None and buf.alive()) else None

    def stop_all(self) -> None:
        with self._lock:
            for buf in self._buffers.values():
                buf.stop()
            self._buffers.clear()
            self._last_seen_with_person.clear()


def _wall_time() -> float:
    import time as _t
    return _t.time()
