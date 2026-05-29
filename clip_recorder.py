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
    """Continuous HLS-style ring buffer for one camera."""

    def __init__(self, camera_id: str, rtsp_url: str, base_dir: Path) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.buf_dir = (base_dir / "buffer" / camera_id).resolve()
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------

    def start(self, total_buffer_s: int = 30) -> None:
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
    """Tracks per-camera ``ClipBuffer`` instances; reconciled against the
    current camera list on every change."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._buffers: dict[str, ClipBuffer] = {}
        self._lock = threading.Lock()

    def update(self, cameras: list[dict]) -> None:
        with self._lock:
            wanted = {
                c["id"]: c
                for c in cameras
                if c.get("enabled", True) and c.get("save_history")
                and (c.get("rtsp_url") or "").strip()
            }
            # Stop and forget removed / disabled
            for cam_id in list(self._buffers.keys()):
                if cam_id not in wanted:
                    self._buffers[cam_id].stop()
                    self._buffers.pop(cam_id, None)
            # Start new
            for cam_id, cam in wanted.items():
                buf = self._buffers.get(cam_id)
                if buf and buf.alive():
                    continue
                clip_s = int(cam.get("clip_seconds", 5) or 5)
                if buf is None:
                    buf = ClipBuffer(cam_id, cam["rtsp_url"], self.base_dir)
                    self._buffers[cam_id] = buf
                buf.start(total_buffer_s=clip_s * 2 + 4)

    def get(self, camera_id: str) -> ClipBuffer | None:
        with self._lock:
            return self._buffers.get(camera_id)

    def stop_all(self) -> None:
        with self._lock:
            for buf in self._buffers.values():
                buf.stop()
            self._buffers.clear()
