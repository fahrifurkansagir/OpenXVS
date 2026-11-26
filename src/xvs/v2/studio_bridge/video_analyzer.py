"""Real-world `.mp4`/`.webm`/`.mov` video analyzer & H.264 annotated proof exporter (`OpenXVS v4.5`).

Runs background inference jobs using our REAL 5/5 OpenXVS checkpoints (`StudioInferenceEngine`)
under strict safety and resource bounds:
  - Max upload size: `150 MiB`
  - Valid channels: `("d2", "land", "thermal", "nav")`
  - `sample_fps` clamped `[1, 30]` (default `10`); `max_frames` clamped `[5, 300]` (default `120`)
  - Strict 12-char lowercase hex job IDs (`^[a-f0-9]{12}$`), bounded retention (`max 16` jobs)
  - Isolated per-job `D2TrackerState` (never corrupts live `demo_tick` state)
  - Browser-playable H.264 `.mp4` via host `ffmpeg` (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`)
    with graceful OpenCV `mp4v` fallback if `ffmpeg` is absent.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np

from .inference_engine import (
    COL_AMBER,
    COL_BG,
    COL_CYAN,
    COL_EMERALD,
    COL_ROSE,
    COL_TEXT,
    StudioInferenceEngine,
    _draw_badge,
    _encode_jpeg_b64,
)
from .models_loader import REPO_ROOT

MAX_VIDEO_UPLOAD_BYTES: int = 150 * 1024 * 1024  # 150 MiB
VALID_VIDEO_CHANNELS: tuple[str, ...] = ("d2", "land", "thermal", "nav")
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
MAX_RETAINED_JOBS: int = 16

DEFAULT_AOT_PROOF_MP4 = REPO_ROOT / "outputs" / "xvs_activerange_aot_proof_1080p.mp4"


def is_valid_job_id(job_id: str) -> bool:
    """Strictly validate 12-char lowercase hex job ID (blocks path traversal)."""
    return bool(isinstance(job_id, str) and JOB_ID_PATTERN.match(job_id))


def _clamp_even_dims(w: int, h: int, max_w: int = 1280, max_h: int = 720) -> tuple[int, int]:
    scale = min(1.0, float(max_w) / float(max(1, w)), float(max_h) / float(max(1, h)))
    ow = max(64, int(round(w * scale)))
    oh = max(64, int(round(h * scale)))
    ow = ow - (ow % 2)
    oh = oh - (oh % 2)
    return ow, oh


def _render_hud_overlay(
    frame_bgr: np.ndarray,
    *,
    channel: str,
    result: dict[str, Any],
    frame_idx: int,
    total_frames: int,
    timestamp_s: float,
    ckpt_sha8: str,
) -> np.ndarray:
    """Draw crisp telemetry HUD overlay (detections/keypoints/vacuity/nav + provenance) onto BGR frame."""
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()
    metrics = result["metrics"]

    if channel == "d2":
        det = metrics["detection"]
        ar = metrics["active_range"]
        if isinstance(det, dict):
            p1 = (int(round(float(det["x1"]))), int(round(float(det["y1"]))))
            p2 = (int(round(float(det["x2"]))), int(round(float(det["y2"]))))
            cv2.rectangle(vis, p1, p2, COL_EMERALD, 2, cv2.LINE_AA)
            _draw_badge(
                vis,
                f"XVS-D2 {float(det['score']):.2f}",
                (max(6, p1[0]), max(22, p1[1] - 6)),
                COL_EMERALD,
            )
        banner = (
            f"OPENXVS D2+ACTIVERANGE | {ar['mode']} | "
            f"R_low {float(ar['r_safe_lower_m']):.0f}m | sig {float(ar['sigma_r_m']):.0f}m"
        )
        accent = COL_EMERALD if bool(metrics["detected"]) else COL_CYAN
    elif channel == "land":
        kpts = metrics["keypoints_xy_256"]
        sx = float(w) / 256.0
        sy = float(h) / 256.0
        pts = [(int(round(float(x) * sx)), int(round(float(y) * sy))) for x, y in kpts]
        for idx, pt in enumerate(pts):
            cv2.circle(vis, pt, 4, COL_CYAN if idx % 2 == 0 else COL_EMERALD, -1, cv2.LINE_AA)
        if len(pts) >= 12:
            cv2.line(vis, pts[0], pts[9], COL_CYAN, 2, cv2.LINE_AA)
            cv2.line(vis, pts[2], pts[11], COL_CYAN, 2, cv2.LINE_AA)
            cv2.line(vis, pts[0], pts[2], COL_EMERALD, 2, cv2.LINE_AA)
        banner = (
            f"OPENXVS LAND | GS {float(metrics['glideslope_deg']):.1f}deg | "
            f"HPL {float(metrics['hpl_m']):.0f}m | VPL {float(metrics['vpl_m']):.0f}m"
        )
        accent = COL_CYAN
    elif channel == "thermal":
        vac = float(metrics["mean_vacuity"])
        bar_w = int(round(160.0 * max(0.0, min(1.0, vac))))
        cv2.rectangle(vis, (12, 68), (176, 84), COL_BG, -1)
        cv2.rectangle(
            vis,
            (14, 70),
            (14 + bar_w, 82),
            COL_AMBER if vac < 0.7 else COL_ROSE,
            -1,
        )
        banner = f"OPENXVS THERMAL | {metrics['simplex_mode']} | Vac {vac:.2f}"
        accent = COL_AMBER
    else:
        spoof = bool(metrics["spoof_detected"])
        banner = (
            f"OPENXVS NAV | {metrics['nav_mode']} | "
            f"SIM {float(metrics['cosine_sim']):.2f} | HPL {float(metrics['hpl_vis_m']):.0f}m"
        )
        accent = COL_ROSE if spoof else COL_EMERALD

    _draw_badge(vis, banner, (12, 26), accent)
    footer = (
        f"FRAME {frame_idx + 1}/{max(1, total_frames)} | "
        f"T+{timestamp_s:05.2f}s | {float(result['latency_ms']):.1f}ms | SHA {ckpt_sha8}"
    )
    _draw_badge(vis, footer, (12, 52), COL_CYAN)
    return vis


class _Mp4StreamWriter:
    """Encodes BGR frames to browser-playable H.264 MP4 via ffmpeg (with OpenCV mp4v fallback)."""

    def __init__(self, out_path: Path, width: int, height: int, fps: float) -> None:
        self.out_path = Path(out_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self._proc: subprocess.Popen[bytes] | None = None
        self._cv_writer: cv2.VideoWriter | None = None
        self.codec_used = "unknown"

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is not None:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.3f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.out_path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self.codec_used = "libx264"
                return
            except Exception:
                self._proc = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(
            str(self.out_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        self.codec_used = "mp4v"

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                self._proc.kill()
                self._proc.wait(timeout=5.0)
            except Exception:
                pass
            finally:
                self._proc = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
            except (BrokenPipeError, OSError) as exc:
                self._kill_proc()
                raise RuntimeError(f"ffmpeg pipe write failed: {exc}") from exc
        elif self._cv_writer is not None:
            self._cv_writer.write(frame_bgr)

    def close(self) -> None:
        if self._proc is not None:
            proc = self._proc
            try:
                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    _, stderr_bytes = proc.communicate(timeout=30.0)
                except subprocess.TimeoutExpired as exc:
                    proc.kill()
                    try:
                        proc.wait(timeout=5.0)
                    except Exception:
                        pass
                    raise RuntimeError("ffmpeg timed out during finalize and was killed") from exc
                if proc.returncode != 0:
                    err = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {err}")
            finally:
                self._proc = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None


class VideoJobManager:
    """Bounded background video analyzer & annotated H.264 MP4 exporter."""

    def __init__(
        self,
        engine: StudioInferenceEngine,
        *,
        work_root: Path | None = None,
        max_retained_jobs: int = MAX_RETAINED_JOBS,
    ) -> None:
        self.engine = engine
        if work_root is None:
            self._work_root = Path(tempfile.mkdtemp(prefix="openxvs_video_jobs_")).resolve()
            self._owns_root = True
        else:
            self._work_root = Path(work_root).resolve()
            self._work_root.mkdir(parents=True, exist_ok=True)
            self._owns_root = False
        self._max_retained = max(2, int(max_retained_jobs))
        self._lock = threading.Lock()
        self._builtin_cache_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._workers: dict[str, threading.Thread] = {}

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            threads = list(self._workers.values())
        for th in threads:
            if th.is_alive():
                th.join(timeout=5.0)
        if self._owns_root and self._work_root.exists():
            shutil.rmtree(self._work_root, ignore_errors=True)

    def _evict_if_needed_locked(self) -> None:
        while len(self._jobs) > self._max_retained:
            oldest_id, oldest_rec = next(iter(self._jobs.items()))
            if oldest_rec["status"] in ("queued", "running") and len(self._jobs) <= self._max_retained + 4:
                break
            self._jobs.pop(oldest_id, None)
            jdir = self._work_root / oldest_id
            shutil.rmtree(jdir, ignore_errors=True)

    def _validate_params(self, channel: str, sample_fps: int, max_frames: int) -> tuple[str, int, int]:
        ch = channel.strip().lower()
        if ch not in VALID_VIDEO_CHANNELS:
            raise ValueError(
                f"Unsupported video channel {channel!r}; expected one of {VALID_VIDEO_CHANNELS}"
            )
        sfps = max(1, min(30, int(sample_fps)))
        mframes = max(5, min(300, int(max_frames)))
        return ch, sfps, mframes

    def _compile_frames_to_cached_mp4(
        self,
        clip_name: str,
        src_frames: list[Path],
        *,
        max_w: int = 640,
        max_h: int = 480,
        fps: float = 10.0,
    ) -> Path:
        cache_dir = (self._work_root / "_builtin_cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_mp4 = cache_dir / f"{clip_name}.mp4"
        with self._builtin_cache_lock:
            if cached_mp4.is_file() and cached_mp4.stat().st_size > 1000:
                return cached_mp4

            first = cv2.imread(str(src_frames[0]), cv2.IMREAD_COLOR)
            if first is None:
                raise RuntimeError(f"Failed to read built-in raw frame {src_frames[0]}")
            ow, oh = _clamp_even_dims(first.shape[1], first.shape[0], max_w=max_w, max_h=max_h)
            tmp_mp4 = cache_dir / f"{clip_name}_{secrets.token_hex(6)}.mp4"
            writer = _Mp4StreamWriter(tmp_mp4, ow, oh, fps=fps)
            try:
                for p in src_frames:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is not None:
                        writer.write(img)
            except Exception:
                try:
                    writer.close()
                except Exception:
                    pass
                tmp_mp4.unlink(missing_ok=True)
                raise
            writer.close()
            tmp_mp4.replace(cached_mp4)
            return cached_mp4

    def _materialize_builtin_clip(self, clip: str) -> Path:
        """Assemble and cache a clean RAW (zero-prebaked-overlay) real airborne clip."""
        clip_name = clip.strip().lower()
        if clip_name == "aot_realsky":
            # Strictly prefer RAW real Amazon AOT airborne frames (zero prebaked matplotlib overlays!)
            src_frames: list[Path] = []
            eo_dir = self.engine.studio_media_dir / "eo_sky"
            if eo_dir.is_dir():
                src_frames = sorted(eo_dir.glob("*.jpg"))[:60]
                if not src_frames:
                    src_frames = sorted(eo_dir.glob("*.png"))[:60]
            if not src_frames and self.engine.aot_pack_dir.is_dir():
                src_frames = sorted(self.engine.aot_pack_dir.glob("enc0_*.jpg"))[:40]
                if not src_frames:
                    src_frames = sorted(self.engine.aot_pack_dir.glob("*.png"))[:40]
                if not src_frames:
                    src_frames = sorted(self.engine.aot_pack_dir.glob("*.jpg"))[:40]
            if src_frames:
                return self._compile_frames_to_cached_mp4(
                    "aot_realsky_raw",
                    src_frames,
                    max_w=640,
                    max_h=480,
                    fps=10.0,
                )
            if DEFAULT_AOT_PROOF_MP4.is_file():
                return DEFAULT_AOT_PROOF_MP4
            raise FileNotFoundError("No built-in real-sky AOT frames or clip found on disk")

        if clip_name in ("runway_approach", "ksfo_approach"):
            rwy_dir = self.engine.studio_media_dir / "runway_land"
            src_frames = sorted(rwy_dir.glob("*.jpg"))[:36] if rwy_dir.is_dir() else []
            if not src_frames:
                raise FileNotFoundError(
                    "Built-in runway_approach frames not found in data/studio_media/runway_land"
                )
            return self._compile_frames_to_cached_mp4(
                "runway_approach_raw",
                src_frames,
                max_w=640,
                max_h=360,
                fps=10.0,
            )

        raise ValueError(
            f"Unknown built-in clip {clip!r}; supported: 'aot_realsky', 'runway_approach', 'ksfo_approach'"
        )

    def _spawn_job(
        self,
        *,
        input_path: Path,
        channel: str,
        sample_fps: int,
        max_frames: int,
        clip_label: str | None = None,
    ) -> dict[str, Any]:
        job_id = secrets.token_hex(6)  # exact 12-char lowercase hex
        job_dir = (self._work_root / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = job_dir / f"annotated_{job_id}.mp4"

        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "channel": channel,
            "clip": clip_label,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "progress_pct": 0.0,
            "frames_done": 0,
            "total_frames": 0,
            "latest_preview_b64": None,
            "error": None,
            "summary": None,
            "_out_mp4": str(out_mp4),
        }
        worker = threading.Thread(
            target=self._run_job,
            args=(job_id, input_path, out_mp4, channel, sample_fps, max_frames),
            daemon=True,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._workers[job_id] = worker
            self._evict_if_needed_locked()

        worker.start()

        resp: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "channel": channel,
        }
        if clip_label is not None:
            resp["clip"] = clip_label
        return resp

    def allocate_upload_path(self) -> Path:
        """Allocate a unique per-upload staging path for chunked streaming writes."""
        staging_dir = self._work_root / f"in_{secrets.token_hex(6)}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir / "input_video.mp4"

    def start_upload_file_job(
        self,
        input_path: Path,
        *,
        channel: str = "d2",
        sample_fps: int = 10,
        max_frames: int = 120,
    ) -> dict[str, Any]:
        """Validate a streamed uploaded video file on disk (<= 150 MiB) and enqueue background job."""
        p = Path(input_path)
        if not p.is_file():
            raise ValueError("Uploaded video file missing")
        sz = int(p.stat().st_size)
        if sz <= 0:
            raise ValueError("Empty video payload")
        if sz > MAX_VIDEO_UPLOAD_BYTES:
            raise ValueError(f"Video payload exceeds {MAX_VIDEO_UPLOAD_BYTES} bytes (150 MiB) limit")
        ch, sfps, mframes = self._validate_params(channel, sample_fps, max_frames)
        return self._spawn_job(
            input_path=p,
            channel=ch,
            sample_fps=sfps,
            max_frames=mframes,
            clip_label=None,
        )

    def start_upload_job(
        self,
        video_bytes: bytes,
        *,
        channel: str = "d2",
        sample_fps: int = 10,
        max_frames: int = 120,
    ) -> dict[str, Any]:
        """Validate uploaded raw video bytes (<= 150 MiB) and enqueue background analysis job."""
        if not video_bytes:
            raise ValueError("Empty video payload")
        if len(video_bytes) > MAX_VIDEO_UPLOAD_BYTES:
            raise ValueError(f"Video payload exceeds {MAX_VIDEO_UPLOAD_BYTES} bytes (150 MiB) limit")
        ch, sfps, mframes = self._validate_params(channel, sample_fps, max_frames)

        # Save raw bytes to job-scoped temp file before spawning worker
        in_path = self.allocate_upload_path()
        in_path.write_bytes(video_bytes)
        return self._spawn_job(
            input_path=in_path,
            channel=ch,
            sample_fps=sfps,
            max_frames=mframes,
            clip_label=None,
        )

    def start_builtin_job(
        self,
        *,
        clip: str = "aot_realsky",
        channel: str = "d2",
        sample_fps: int = 10,
        max_frames: int = 120,
    ) -> dict[str, Any]:
        """Enqueue background analysis job on a verified RAW built-in real flight clip."""
        ch, sfps, mframes = self._validate_params(channel, sample_fps, max_frames)
        in_path = self._materialize_builtin_clip(clip)
        return self._spawn_job(
            input_path=in_path,
            channel=ch,
            sample_fps=sfps,
            max_frames=mframes,
            clip_label=clip.strip().lower(),
        )

    def _run_job(
        self,
        job_id: str,
        input_path: Path,
        out_mp4: Path,
        channel: str,
        sample_fps: int,
        max_frames: int,
    ) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"

        cap = cv2.VideoCapture(str(input_path))
        writer: _Mp4StreamWriter | None = None
        try:
            if not cap.isOpened():
                raise ValueError("Unable to open video stream with OpenCV VideoCapture")

            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(src_fps) or src_fps <= 0.1 or src_fps > 240.0:
                src_fps = float(sample_fps)
            raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(1, int(round(src_fps / float(sample_fps))))
            est_total = (
                min(max_frames, max(1, (raw_total + stride - 1) // stride))
                if raw_total > 0
                else max_frames
            )
            out_fps = min(float(sample_fps), src_fps)

            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["total_frames"] = est_total

            # Job-isolated D2 tracker state (never contaminates live demo_tick)
            job_d2_state = self.engine.new_d2_tracker_state()
            ckpt_sha = self.engine.checkpoint_sha256_for_channel(channel)
            ckpt_sha8 = ckpt_sha[:8] if ckpt_sha else "local"

            frames_processed = 0
            detected_count = 0
            ar_locked_frames = 0
            latencies_ms: list[float] = []
            max_conf = 0.0
            raw_idx = 0

            while frames_processed < max_frames and not self._stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                if raw_idx % stride != 0:
                    raw_idx += 1
                    continue
                raw_idx += 1

                if writer is None:
                    ow, oh = _clamp_even_dims(frame_bgr.shape[1], frame_bgr.shape[0], max_w=1280, max_h=720)
                    writer = _Mp4StreamWriter(out_mp4, ow, oh, fps=out_fps)

                work_bgr = cv2.resize(frame_bgr, (writer.width, writer.height), interpolation=cv2.INTER_AREA)
                res = self.engine.analyze_frame_fast(
                    work_bgr,
                    channel=channel,
                    return_jpeg=False,
                    tracker_state=job_d2_state,
                )
                lat = float(res["latency_ms"])
                conf = float(res["confidence"])
                latencies_ms.append(lat)
                if conf > max_conf:
                    max_conf = conf

                metrics = res["metrics"]
                if channel == "d2":
                    if bool(metrics["detected"]):
                        detected_count += 1
                    ar = metrics["active_range"]
                    if str(ar["mode"]) == "LOCKED":
                        ar_locked_frames += 1
                elif channel == "land":
                    if conf >= 0.4:
                        detected_count += 1
                elif channel == "thermal":
                    if float(metrics["mean_vacuity"]) < 0.7:
                        detected_count += 1
                else:
                    if float(metrics["cosine_sim"]) >= 0.35:
                        detected_count += 1

                t_sec = float(frames_processed) / max(1.0, out_fps)
                annotated = _render_hud_overlay(
                    work_bgr,
                    channel=channel,
                    result=res,
                    frame_idx=frames_processed,
                    total_frames=est_total,
                    timestamp_s=t_sec,
                    ckpt_sha8=ckpt_sha8,
                )
                writer.write(annotated)
                frames_processed += 1

                preview_small = cv2.resize(annotated, (384, 216), interpolation=cv2.INTER_AREA)
                preview_uri = f"data:image/jpeg;base64,{_encode_jpeg_b64(preview_small, quality=80)}"
                pct = round(min(99.0, 100.0 * float(frames_processed) / float(max(1, est_total))), 1)

                with self._lock:
                    if job_id in self._jobs:
                        rec = self._jobs[job_id]
                        rec["frames_done"] = frames_processed
                        rec["total_frames"] = max(est_total, frames_processed)
                        rec["progress_pct"] = pct
                        rec["latest_preview_b64"] = preview_uri

            if frames_processed == 0 or writer is None:
                raise ValueError("Video contained zero decodable frames")

            writer.close()
            writer = None
            out_bytes = int(out_mp4.stat().st_size) if out_mp4.is_file() else 0
            mean_lat = round(float(sum(latencies_ms)) / float(len(latencies_ms)), 2) if latencies_ms else 0.0
            det_rate = round(float(detected_count) / float(frames_processed), 4)

            summary = {
                "frames_processed": frames_processed,
                "detection_rate": det_rate,
                "mean_latency_ms": mean_lat,
                "max_confidence": round(max_conf, 4),
                "active_range_locked_frames": ar_locked_frames,
                "checkpoint_sha256": ckpt_sha,
                "output_mp4_bytes": out_bytes,
                "download_url": f"/api/video_job/{job_id}/download",
            }
            with self._lock:
                if job_id in self._jobs:
                    rec = self._jobs[job_id]
                    rec["status"] = "done"
                    rec["progress_pct"] = 100.0
                    rec["frames_done"] = frames_processed
                    rec["total_frames"] = frames_processed
                    rec["summary"] = summary
        except Exception as exc:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            with self._lock:
                if job_id in self._jobs:
                    rec = self._jobs[job_id]
                    rec["status"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()
            with self._lock:
                self._workers.pop(job_id, None)
            # Clean up per-upload staging dir ONLY (preserve cached _builtin_cache & shared files)
            try:
                if (
                    input_path.resolve().is_relative_to(self._work_root)
                    and input_path.parent.name.startswith("in_")
                ):
                    shutil.rmtree(input_path.parent, ignore_errors=True)
            except Exception:
                pass

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Return public JSON status dict for a validated `job_id`, or `None` if unknown."""
        if not is_valid_job_id(job_id):
            raise ValueError(f"Invalid job_id format: {job_id!r}")
        with self._lock:
            if job_id not in self._jobs:
                return None
            rec = self._jobs[job_id]
            return {
                "job_id": rec["job_id"],
                "status": rec["status"],
                "channel": rec["channel"],
                "progress_pct": rec["progress_pct"],
                "frames_done": rec["frames_done"],
                "total_frames": rec["total_frames"],
                "latest_preview_b64": rec["latest_preview_b64"],
                "error": rec["error"],
                "summary": rec["summary"],
            }

    def get_job_mp4_path(self, job_id: str) -> Path | None:
        """Return verified path to finished annotated `.mp4` strictly inside `self._work_root`."""
        if not is_valid_job_id(job_id):
            raise ValueError(f"Invalid job_id format: {job_id!r}")
        with self._lock:
            if job_id not in self._jobs:
                return None
            rec = self._jobs[job_id]
            if rec["status"] != "done":
                return None
            out_path = Path(str(rec["_out_mp4"])).resolve()
        if not out_path.is_relative_to(self._work_root) or not out_path.is_file():
            return None
        return out_path
