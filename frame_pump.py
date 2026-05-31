"""Single-ffmpeg per-camera fan-out: raw frame pipe + HLS segments with audio.

A ``FramePump`` starts one ffmpeg process that pulls a camera's RTSP feed
exactly once and produces two outputs:

  1. Raw BGR24 frames at a fixed resolution on stdout — consumed by the
     worker (workbench_activity.py via video_source.PipeCapture) for
     YOLO inference. This replaces cv2.VideoCapture's own RTSP session.

  2. HLS .ts segments to disk (with audio) — consumed by clip_recorder
     for clip extraction AND by the /api/audio endpoint for live audio
     playback. This replaces clip_recorder's own ffmpeg AND the audio
     endpoint's on-demand ffmpeg.

Result: 1 RTSP session per camera instead of 3, which removes the Frigate
contention that was knocking the baby camera black.

The worker owns its FramePump — when the worker exits, ffmpeg is killed
with it. When the hub's watchdog respawns the worker, ffmpeg respawns
with it. No separate process to monitor.

Wire format on the pipe: width * height * 3 bytes per frame, BGR24,
no header. PipeCapture in video_source.py knows the geometry from the
``VIDEO_SOURCE=pipe:W:H`` URL and reads in exact-size chunks.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ffmpeg_exe() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


SEGMENT_DURATION_S = 2


class FramePump:
    """One ffmpeg subprocess: raw BGR frames on stdout + HLS .ts on disk.

    Args:
        camera_id: used to name the HLS buffer directory.
        rtsp_url:  source URL (Frigate restream is fine).
        hls_dir:   base directory; segments land in ``hls_dir / camera_id``.
        video_w, video_h: target resolution for the raw-frame pipe. ffmpeg
                  scales as needed. Worker must use the same geometry.
        total_buffer_s: rough seconds of HLS history to keep on disk.
        stderr_log_path: where to redirect ffmpeg's stderr (default /dev/null).
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        hls_dir: Path,
        video_w: int = 1280,
        video_h: int = 720,
        pipe_fps: int = 10,
        total_buffer_s: int = 30,
        stderr_log_path: Path | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.video_w = video_w
        self.video_h = video_h
        # Caps the raw-frame rate on the pipe to ``pipe_fps``. ffmpeg drops
        # excess frames at the filter graph rather than blocking on the
        # pipe — which is critical because the HLS output shares the same
        # decode loop. Without the cap, the worker (YOLO-bound at 5-7 fps)
        # can't drain the pipe fast enough, ffmpeg blocks on the pipe
        # write, and the HLS segment cadence collapses (we observed
        # segment gaps growing from 4 s to 65 s during burn-in). 10 fps
        # gives the worker some headroom; the HLS output stays at native
        # camera rate via -c:v copy.
        self.pipe_fps = pipe_fps
        self.total_buffer_s = total_buffer_s
        self.buf_dir = (hls_dir / camera_id).resolve()
        self.stderr_log_path = stderr_log_path
        self.proc: subprocess.Popen | None = None
        self._stderr_fh = None

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        # Clean any leftover segments from a prior run so the worker's
        # buffer-walking logic in clip_recorder doesn't pick up stale .ts.
        if self.buf_dir.exists():
            shutil.rmtree(self.buf_dir, ignore_errors=True)
        self.buf_dir.mkdir(parents=True, exist_ok=True)

        list_size = max(8, self.total_buffer_s // SEGMENT_DURATION_S + 2)
        seg_pattern = str(self.buf_dir / "seg_%05d.ts")
        m3u8_path = str(self.buf_dir / "buf.m3u8")

        cmd = [
            _ffmpeg_exe(),
            "-hide_banner", "-loglevel", "warning",
            # RTSP input flags — same as clip_recorder used, proven on
            # Reolink + Frigate restreams.
            "-rtsp_transport", "tcp",
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-i", self.rtsp_url,

            # Output 1: raw BGR24 frames at fixed resolution + capped rate
            # on stdout for the worker. fps= before scale so dropped frames
            # never get scaled (cheaper). vsync 2 = vfr — drop late frames.
            "-map", "0:v:0",
            "-vf", f"fps={self.pipe_fps},scale={self.video_w}:{self.video_h}",
            "-vsync", "2",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",

            # Output 2: HLS .ts segments with audio for clips + /api/audio.
            #
            # Codec: Reolink main streams are HEVC/H.265 at 4K and Chrome /
            # Firefox refuse to play HEVC in <video> tags (no native
            # decoder on Windows without hardware). Re-encode to H.264
            # baseline+yuv420p so the lightbox can actually show the
            # clip. Downscale to 1280x720 — the lightbox preview is
            # nowhere near 4K and the smaller frame keeps re-encode CPU
            # in check (we share one box with YOLO).
            #
            # Audio reencoded to AAC because Reolink's RTP timestamps make
            # the muxer choke otherwise.
            "-map", "0:v:0",
            "-map", "0:a:0?",  # optional — some Frigate restreams have no audio
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1280:720",
            "-g", str(SEGMENT_DURATION_S * 30),  # keyframe ~every segment
            "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "22050",
            "-af", "aresample=async=1000",
            "-f", "hls",
            "-hls_time", str(SEGMENT_DURATION_S),
            "-hls_list_size", str(list_size),
            "-hls_flags", "delete_segments+independent_segments+omit_endlist",
            "-hls_segment_filename", seg_pattern,
            m3u8_path,
        ]

        # ffmpeg's stderr can be chatty under network jitter; pin it to a
        # file so we can post-mortem without it filling the worker log.
        if self.stderr_log_path is not None:
            self.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = open(self.stderr_log_path, "ab", buffering=0)
            stderr = self._stderr_fh
        else:
            stderr = subprocess.DEVNULL

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
            bufsize=0,
        )

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    # --- access ----------------------------------------------------------

    @property
    def stdout(self):
        """The pipe end that delivers raw BGR frames. Read in chunks of
        exactly ``video_w * video_h * 3`` bytes."""
        return self.proc.stdout if self.proc else None

    @property
    def frame_bytes(self) -> int:
        return self.video_w * self.video_h * 3
