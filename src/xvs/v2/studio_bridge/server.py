"""Stdlib threaded HTTP/JSON/MP4 inference server for `OpenXVS Studio v4.5` (`http://127.0.0.1:8765`).

Endpoints:
  - `GET  /api/health`
  - `GET  /api/demo_tick?scenario=<cbdr|forcedland|thermal_fog|gps_spoof>&frame=<idx>&images=<0|1>`
  - `POST /api/analyze_image?channel=<d2|land|thermal|nav>`
  - `POST /api/analyze_frame_fast?channel=<d2|land|thermal|nav>`
  - `POST /api/analyze_video?channel=<d2|land|thermal|nav>&sample_fps=<int>&max_frames=<int>`
  - `POST /api/analyze_builtin_video?channel=<d2|land|thermal|nav>&clip=aot_realsky&sample_fps=<int>&max_frames=<int>`
  - `GET  /api/video_job/<job_id>`
  - `GET  /api/video_job/<job_id>/download`

Strict CORS allowlist (`D16-W19-m2`): `Access-Control-Allow-Origin` is emitted ONLY when the
request `Origin` header is explicitly inside `ALLOWED_ORIGINS` (zero wildcard `*`).
"""

from __future__ import annotations

import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .buyer_pack import build_buyer_assurance_pack_zip
from .hil_bridge import (
    EU_PLD_ART4_7_LOCKOUT_MSG,
    NON_FLIGHT_WATERMARK,
    NON_FLIGHT_WATERMARK_ASCII,
    NonFlightSimOnlyError,
    StudioHilBridgeManager,
    is_non_flight_sim_only,
)
from .inference_engine import StudioInferenceEngine
from .video_analyzer import MAX_VIDEO_UPLOAD_BYTES, VideoJobManager, is_valid_job_id

MAX_IMAGE_UPLOAD_BYTES: int = 16 * 1024 * 1024  # 16 MiB cap for single-frame uploads
ALLOWED_ORIGINS: frozenset[str] = frozenset(
    {
        "http://127.0.0.1:4174",
        "http://localhost:4174",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    }
)


def _decode_uploaded_image_bytes(raw_body: bytes, content_type: str) -> np.ndarray:
    """Decode raw JPEG/PNG bytes or JSON `{image_b64: ...}` payload into a BGR numpy frame."""
    if not raw_body:
        raise ValueError("Empty request body")
    if len(raw_body) > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"Upload exceeds {MAX_IMAGE_UPLOAD_BYTES} byte limit")

    payload = raw_body
    if "application/json" in content_type.lower():
        doc = json.loads(raw_body.decode("utf-8"))
        if not isinstance(doc, dict) or "image_b64" not in doc:
            raise ValueError("JSON upload requires 'image_b64' string field")
        b64_str = str(doc["image_b64"])
        if "," in b64_str and b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        payload = base64.b64decode(b64_str)

    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG/PNG image bytes")
    return img


def _qs_first(qs: dict[str, list[str]], key: str, default: str) -> str:
    if key in qs and qs[key]:
        return str(qs[key][0])
    return default


def _parse_finite_float_qs(qs: dict[str, list[str]], key: str, default: str) -> float:
    raw = _qs_first(qs, key, default)
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f"Query parameter {key!r} must be finite, got {raw!r}")
    return val


class StudioBridgeHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying `StudioInferenceEngine`, `VideoJobManager`, and `StudioHilBridgeManager`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: StudioInferenceEngine,
        video_manager: VideoJobManager | None = None,
        hil_manager: StudioHilBridgeManager | None = None,
    ) -> None:
        self.engine = engine
        self.video_manager = VideoJobManager(engine) if video_manager is None else video_manager
        self.hil_manager = StudioHilBridgeManager() if hil_manager is None else hil_manager
        super().__init__(server_address, StudioBridgeHandler)

    def server_close(self) -> None:
        try:
            self.hil_manager.shutdown()
        finally:
            try:
                self.video_manager.shutdown()
            finally:
                super().server_close()


class StudioBridgeHandler(BaseHTTPRequestHandler):
    """Strict-CORS REST handler for OpenXVS Studio v4.5 live AI, camera & video inference."""

    server: StudioBridgeHTTPServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_cors_headers(self) -> None:
        """D16-W19-m2: Emit Access-Control-Allow-Origin strictly for allowlisted origins only."""
        origin = self.headers["Origin"] if "Origin" in self.headers else ""
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Range")

    def _send_non_flight_headers(self) -> None:
        locked = is_non_flight_sim_only()
        self.send_header("X-OpenXVS-Non-Flight-Sim-Only", "1" if locked else "0")
        self.send_header("X-OpenXVS-Watermark", NON_FLIGHT_WATERMARK_ASCII)

    def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        locked = is_non_flight_sim_only()
        payload = dict(body)
        payload["non_flight_sim_only"] = locked
        payload["watermark"] = NON_FLIGHT_WATERMARK
        if locked:
            payload["actuator_authority"] = False
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self._send_cors_headers()
        self._send_non_flight_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream_mp4_file(self, mp4_path: Path, job_id: str) -> None:
        file_size = int(mp4_path.stat().st_size)
        range_hdr = self.headers["Range"] if "Range" in self.headers else ""
        start = 0
        end = max(0, file_size - 1)
        partial = False
        range_unsatisfiable = False

        if range_hdr.startswith("bytes="):
            spec = range_hdr[len("bytes=") :].split(",", 1)[0].strip()
            if "-" in spec:
                s_part, e_part = spec.split("-", 1)
                try:
                    if not s_part and e_part:
                        # RFC 7233 suffix-byte-range-spec: `bytes=-500`
                        suffix_len = int(e_part)
                        if suffix_len <= 0 or file_size <= 0:
                            range_unsatisfiable = True
                        else:
                            start = max(0, file_size - suffix_len)
                            end = file_size - 1
                            partial = True
                    elif s_part:
                        start = int(s_part)
                        end = int(e_part) if e_part else (file_size - 1)
                        if start < 0 or file_size <= 0 or start >= file_size or end < start:
                            range_unsatisfiable = True
                        else:
                            end = min(file_size - 1, end)
                            partial = True
                    else:
                        range_unsatisfiable = True
                except ValueError:
                    range_unsatisfiable = True
            else:
                range_unsatisfiable = True

        if range_unsatisfiable:
            self.send_response(int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE))
            self._send_cors_headers()
            self._send_non_flight_headers()
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        length = max(0, end - start + 1) if file_size > 0 else 0
        self.send_response(int(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK))
        self._send_cors_headers()
        self._send_non_flight_headers()
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", f'inline; filename="openxvs_{job_id}.mp4"')
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with mp4_path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_OPTIONS(self) -> None:
        self.send_response(int(HTTPStatus.NO_CONTENT))
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/health":
            summary = self.server.engine.bundle.health_summary()
            summary["gpu_name"] = summary["host_gpu_name"]
            summary["device"] = summary["compute_device"]
            summary["media_pack_ready"] = self.server.engine.media_pack_ready()
            self._send_json(HTTPStatus.OK, summary)
            return

        if parsed.path == "/api/demo_tick":
            scenario = _qs_first(qs, "scenario", "cbdr")
            frame_str = _qs_first(qs, "frame", "0")
            images_str = _qs_first(qs, "images", "1")
            try:
                frame_idx = max(0, int(frame_str))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid frame index: {frame_str!r}"})
                return
            include_images = images_str not in ("0", "false", "False")
            try:
                payload = self.server.engine.run_demo_tick(
                    scenario=scenario,
                    frame_idx=frame_idx,
                    include_images=include_images,
                )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"demo_tick failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, payload)
            return

        # GET /api/video_job/<job_id> and GET /api/video_job/<job_id>/download
        if parsed.path.startswith("/api/video_job/"):
            rest = parsed.path[len("/api/video_job/") :]
            if rest.endswith("/download"):
                job_id = rest[: -len("/download")]
                if not is_valid_job_id(job_id):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid job_id: {job_id!r}"})
                    return
                mp4_path = self.server.video_manager.get_job_mp4_path(job_id)
                if mp4_path is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Annotated MP4 not ready for job {job_id!r}"})
                    return
                self._stream_mp4_file(mp4_path, job_id)
                return

            job_id = rest
            if not is_valid_job_id(job_id):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid job_id: {job_id!r}"})
                return
            status_doc = self.server.video_manager.get_job_status(job_id)
            if status_doc is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown job_id: {job_id!r}"})
                return
            self._send_json(HTTPStatus.OK, status_doc)
            return

        if parsed.path == "/api/hil/status":
            self._send_json(HTTPStatus.OK, self.server.hil_manager.status())
            return

        if parsed.path == "/api/export_buyer_pack":
            try:
                zip_bytes = build_buyer_assurance_pack_zip(
                    self.server.engine,
                    self.server.hil_manager,
                    self.server.video_manager,
                )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Buyer pack export failed: {type(exc).__name__}: {exc}"},
                )
                return
            self.send_response(int(HTTPStatus.OK))
            self._send_cors_headers()
            self._send_non_flight_headers()
            self.send_header("Content-Type", "application/zip")
            self.send_header(
                "Content-Disposition",
                'attachment; filename="openxvs_v4_5_buyer_assurance_pack.zip"',
            )
            self.send_header("Content-Length", str(len(zip_bytes)))
            self.end_headers()
            self.wfile.write(zip_bytes)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {parsed.path}"})

    def _read_bounded_body(self, max_bytes: int, *, allow_empty: bool = False) -> bytes | None:
        cl_raw = self.headers["Content-Length"] if "Content-Length" in self.headers else "0"
        try:
            content_len = int(cl_raw)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header"})
            return None
        if content_len > max_bytes:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Payload size {content_len} exceeds {max_bytes} byte limit"},
            )
            return None
        if content_len <= 0:
            if allow_empty:
                return b""
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Empty payload (must be 1..{max_bytes} bytes)"},
            )
            return None
        return self.rfile.read(content_len)

    def _stream_bounded_body_to_file(self, dest_path: Path, max_bytes: int) -> int | None:
        """Stream POST request body to disk in 64 KiB chunks (eliminates 150 MiB RAM spikes)."""
        cl_raw = self.headers["Content-Length"] if "Content-Length" in self.headers else "0"
        try:
            content_len = int(cl_raw)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header"})
            return None
        if content_len > max_bytes:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Payload size {content_len} exceeds {max_bytes} byte limit"},
            )
            return None
        if content_len <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Empty payload (must be 1..{max_bytes} bytes)"},
            )
            return None
        remaining = content_len
        written = 0
        try:
            with dest_path.open("wb") as fout:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    fout.write(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise
        if written <= 0:
            dest_path.unlink(missing_ok=True)
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty video payload"})
            return None
        return written

    def _drain_small_body(self, max_drain_bytes: int = 65536) -> None:
        """Drain small unread POST bodies before early 4xx responses to avoid Windows TCP RST."""
        cl_raw = self.headers["Content-Length"] if "Content-Length" in self.headers else "0"
        try:
            content_len = int(cl_raw)
        except ValueError:
            return
        if 0 < content_len <= max_drain_bytes:
            try:
                self.rfile.read(content_len)
            except Exception:
                pass

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        # 1. Single-frame image upload (`POST /api/analyze_image`)
        if parsed.path == "/api/analyze_image":
            channel = _qs_first(qs, "channel", "d2").strip().lower()
            if channel not in ("d2", "land", "thermal", "nav"):
                self._drain_small_body()
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Unsupported channel {channel!r}; expected one of d2, land, thermal, nav"},
                )
                return
            raw_body = self._read_bounded_body(MAX_IMAGE_UPLOAD_BYTES)
            if raw_body is None:
                return
            ctype = self.headers["Content-Type"] if "Content-Type" in self.headers else "application/octet-stream"
            try:
                img_bgr = _decode_uploaded_image_bytes(raw_body, ctype)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Image decode error: {exc}"})
                return
            q_eo_val = 0.85
            q_lwir_val = 0.95
            fog_val = 0.0
            gps_off_val = 0.0
            try:
                if channel == "thermal":
                    q_eo_val = _parse_finite_float_qs(qs, "q_eo", "0.85")
                    q_lwir_val = _parse_finite_float_qs(qs, "q_lwir", "0.95")
                    fog_val = _parse_finite_float_qs(qs, "fog", "0.0")
                elif channel == "nav":
                    gps_off_val = _parse_finite_float_qs(qs, "gps_offset_m", "0.0")
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid query float parameter: {exc}"})
                return
            try:
                if channel == "d2":
                    res = self.server.engine.infer_eo_sky(img_bgr, return_jpeg=True, multi_crop_fallback=True)
                elif channel == "land":
                    res = self.server.engine.infer_runway_land(img_bgr, return_jpeg=True)
                elif channel == "thermal":
                    res = self.server.engine.infer_thermal_dve(
                        img_bgr,
                        q_eo=q_eo_val,
                        q_lwir=q_lwir_val,
                        fog_density=fog_val,
                        return_jpeg=True,
                    )
                else:
                    res = self.server.engine.infer_crossview_nav(
                        img_bgr,
                        gps_offset_m=gps_off_val,
                        return_jpeg=True,
                    )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Inference failed on channel {channel!r}: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, res)
            return

        # 2. Fast live webcam/USB frame analysis (`POST /api/analyze_frame_fast`)
        if parsed.path == "/api/analyze_frame_fast":
            channel = _qs_first(qs, "channel", "d2").strip().lower()
            if channel not in ("d2", "land", "thermal", "nav"):
                self._drain_small_body()
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Unsupported channel {channel!r}; expected one of d2, land, thermal, nav"},
                )
                return
            raw_body = self._read_bounded_body(MAX_IMAGE_UPLOAD_BYTES)
            if raw_body is None:
                return
            ctype = self.headers["Content-Type"] if "Content-Type" in self.headers else "application/octet-stream"
            try:
                img_bgr = _decode_uploaded_image_bytes(raw_body, ctype)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Image decode error: {exc}"})
                return
            try:
                res = self.server.engine.analyze_frame_fast(img_bgr, channel=channel, return_jpeg=True)
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Fast frame analysis failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, res)
            return

        # 3. Full uploaded `.mp4`/`.webm`/`.mov` video analyzer (`POST /api/analyze_video`)
        if parsed.path == "/api/analyze_video":
            channel = _qs_first(qs, "channel", "d2").strip().lower()
            try:
                sample_fps = int(_qs_first(qs, "sample_fps", "10"))
                max_frames = int(_qs_first(qs, "max_frames", "120"))
            except ValueError:
                self._drain_small_body()
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "sample_fps and max_frames must be integers"})
                return
            staged_path = self.server.video_manager.allocate_upload_path()
            written = self._stream_bounded_body_to_file(staged_path, MAX_VIDEO_UPLOAD_BYTES)
            if written is None:
                shutil.rmtree(staged_path.parent, ignore_errors=True)
                return
            try:
                job = self.server.video_manager.start_upload_file_job(
                    staged_path,
                    channel=channel,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                )
            except ValueError as exc:
                shutil.rmtree(staged_path.parent, ignore_errors=True)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                shutil.rmtree(staged_path.parent, ignore_errors=True)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Video job start failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, job)
            return

        # 4. Built-in 1-click real flight clip analyzer (`POST /api/analyze_builtin_video`)
        if parsed.path == "/api/analyze_builtin_video":
            channel = _qs_first(qs, "channel", "d2").strip().lower()
            clip = _qs_first(qs, "clip", "aot_realsky").strip().lower()
            try:
                sample_fps = int(_qs_first(qs, "sample_fps", "10"))
                max_frames = int(_qs_first(qs, "max_frames", "120"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "sample_fps and max_frames must be integers"})
                return
            # Drain any optional small request body cleanly
            _ = self._read_bounded_body(65536, allow_empty=True)
            try:
                job = self.server.video_manager.start_builtin_job(
                    clip=clip,
                    channel=channel,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                )
            except (ValueError, FileNotFoundError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Built-in video job start failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, job)
            return

        # 5. 1-Click Zero-GPL HIL Loopback Self-Test (`POST /api/hil/loopback_test`)
        if parsed.path == "/api/hil/loopback_test":
            _ = self._read_bounded_body(65536, allow_empty=True)
            try:
                report = self.server.hil_manager.run_loopback_test()
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"HIL loopback test failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, report)
            return

        # 6. Localhost 127.0.0.1 UDP HIL Bridge Connect/Stop (`POST /api/hil/connect`)
        if parsed.path == "/api/hil/connect":
            raw_body = self._read_bounded_body(65536, allow_empty=False)
            if raw_body is None:
                return
            if is_non_flight_sim_only():
                self.server.hil_manager.force_non_flight_lockout()
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {
                        "error": EU_PLD_ART4_7_LOCKOUT_MSG,
                        "code": "NON_FLIGHT_SIM_ONLY_LOCKOUT",
                        "eu_pld_reference": "Directive (EU) 2024/2853 Art. 4(7)",
                        "tx_enabled": False,
                        "actuator_authority": False,
                    },
                )
                return
            try:
                doc = json.loads(raw_body.decode("utf-8"))
                if not isinstance(doc, dict):
                    raise ValueError("Expected JSON object body")
                mode = str(doc["mode"]) if "mode" in doc else "idle"
                rx_port = int(doc["rx_port"]) if "rx_port" in doc and doc["rx_port"] is not None else None
                tx_port = int(doc["tx_port"]) if "tx_port" in doc and doc["tx_port"] is not None else None
                status = self.server.hil_manager.connect(mode=mode, rx_port=rx_port, tx_port=tx_port)
            except NonFlightSimOnlyError as exc:
                self.server.hil_manager.force_non_flight_lockout()
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {
                        "error": str(exc),
                        "code": "NON_FLIGHT_SIM_ONLY_LOCKOUT",
                        "eu_pld_reference": "Directive (EU) 2024/2853 Art. 4(7)",
                        "tx_enabled": False,
                        "actuator_authority": False,
                    },
                )
                return
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid HIL connect request: {exc}"})
                return
            except OSError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": f"UDP socket bind error: {exc}"})
                return
            self._send_json(HTTPStatus.OK, status)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {parsed.path}"})


def create_studio_http_server(
    engine: StudioInferenceEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    video_manager: VideoJobManager | None = None,
    hil_manager: StudioHilBridgeManager | None = None,
) -> StudioBridgeHTTPServer:
    """Create a bound `StudioBridgeHTTPServer` ready for `.serve_forever()`."""
    return StudioBridgeHTTPServer(
        (host, int(port)),
        engine,
        video_manager=video_manager,
        hil_manager=hil_manager,
    )
