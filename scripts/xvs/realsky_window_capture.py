#!/usr/bin/env python3
"""OpenXVS V3.0 Tier-0A ($0) Day-1 Real-Sky Window Capture & Auto-Label CLI (`DR-15` + `DR-19`).

Zero-hardware-cost CLI that any user can run from a window ledge or balcony today:
  - Enforces `DR-15` polite public ADS-B polling (`>= 3.5 s` on `adsb.lol`, `>= 2.5 s` on
    `FlyItalyADSB`, automatic exponential backoff on HTTP 429; hard-blocks `airplanes.live`
    HTTP 403 and non-commercial `opensky` / `adsb.fi` on permissive dataset releases).
  - Caches `AviationWeather.gov` METAR QNH (`>= 15 min` TTL, `<= 100 req/min` rate ceiling).
  - Executes `xvs.v2.xvs_realsky.autolabel` (exact ISA/QNH/GEOID18 datum reconciliation,
    WGS-84 ECEF -> Local NED, offline non-causal Cubic Hermite spline, Brown-Conrady
    projection, mandatory `>= +5.0 deg` sky-only elevation mask, and `--strip-upstream-adsb`
    ODbL/CC-BY-SA/ICAO24/callsign scrubbing).
  - Emits validated `Croissant 1.1 + RAI 1.0` (`croissant.json`) metadata and a deterministic
    SHA-256 shard manifest (`shard_manifest.json`) ready for Ed25519 attestation (`DR-19` §2).

Strictly stdlib + `numpy` + `xvs.v2.xvs_realsky`. Zero GPL/LGPL deps, zero internal ENU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.xvs_realsky.autolabel import (
    ISA_P0_HPA,
    ISA_T0_K,
    MIN_SKY_ELEVATION_MASK_DEG,
    AirframeDimensionsM,
    CameraIntrinsics,
    ContrastSnapResult,
    RealSkyComplianceError,
    apply_sky_elevation_mask,
    assert_adsb_provider_allowed,
    compute_sky_elevation_mask,
    ed25519_public_key_from_seed,
    ed25519_sign,
    ed25519_verify,
    estimate_airframe_box_prior_px,
    geodetic_to_local_ned,
    interpolate_cubic_hermite_ned,
    load_camera_calibration,
    ned_velocity_from_adsb,
    project_ned_to_pixel,
    reconcile_ellipsoidal_altitude_m,
    sanitize_realsky_label_record,
    sign_manifest_sha256,
    snap_target_contrast_centroid,
    verify_manifest_signature,
)

CROISSANT_1_1_URI = "http://mlcommons.org/croissant/1.1"
CROISSANT_RAI_1_0_URI = "http://mlcommons.org/croissant/RAI/1.0"
DEFAULT_XVS_D2_ONNX: Path = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"
SUPPORTED_VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov"})
D2_CROP_SIZE: int = 256

MIN_POLL_INTERVAL_SEC: dict[str, float] = {
    "adsb.lol": 3.5,
    "flyitalyadsb": 2.5,
}
METAR_MIN_TTL_SEC: float = 900.0  # 15 minutes (`DR-15` Part 1)
METAR_MAX_REQ_PER_MIN: float = 100.0
METAR_MIN_SPACING_SEC: float = 60.0 / METAR_MAX_REQ_PER_MIN  # 0.6 s

ALLOWED_RELEASE_LICENSES: frozenset[str] = frozenset({"CDLA-Permissive-2.0", "CC-BY-4.0"})

QUARANTINED_SOURCE_TOKENS: tuple[str, ...] = (
    "lard 2.0",
    "lard_2.0",
    "lard_2_0",
    "lard2",
    "lardon",
    "google earth studio",
    "bingmaps",
    "bing maps",
    "arcgis",
    "msfs",
    "x-plane",
    "xplane",
    "opencellid",
    "space-track",
    "airplanes.live",
    "infiray",
    "yantai iray",
)

FORBIDDEN_LEAK_KEYS: frozenset[str] = frozenset(
    {
        "icao24",
        "icao",
        "hex",
        "callsign",
        "flight",
        "tail_number",
        "registration",
        "raw_adsb_json",
        "upstream_payload",
    }
)

AERODYNAMIC_DIMENSIONS_BY_CLASS: dict[str, AirframeDimensionsM] = {
    "LIGHT_SINGLE": AirframeDimensionsM(wingspan_m=11.0, length_m=8.28, tail_height_m=2.72),
    "TWIN_TURBOPROP": AirframeDimensionsM(wingspan_m=17.65, length_m=14.22, tail_height_m=4.37),
    "NARROWBODY_JET": AirframeDimensionsM(wingspan_m=35.8, length_m=37.57, tail_height_m=11.76),
    "WIDEBODY_JET": AirframeDimensionsM(wingspan_m=60.9, length_m=63.7, tail_height_m=18.5),
    "ROTORCRAFT": AirframeDimensionsM(wingspan_m=10.7, length_m=12.9, tail_height_m=3.3),
    "GLIDER_ULTRALIGHT": AirframeDimensionsM(wingspan_m=15.0, length_m=6.8, tail_height_m=1.3),
    "CIVIL_UNKNOWN": AirframeDimensionsM(wingspan_m=14.0, length_m=10.5, tail_height_m=3.2),
}


@dataclass(frozen=True)
class MetarObservation:
    """Cached AviationWeather.gov METAR pressure/temperature datum for QNH reconciliation."""

    station_icao: str
    qnh_hpa: float
    t_amb_k: float
    station_elev_m: float
    fetched_mono_s: float
    source: str


@dataclass(frozen=True)
class AdsbFix:
    """Transient in-memory ADS-B fix discarded immediately after 2D visual label sanitization."""

    track_key: str
    t_rx_s: float
    lat_deg: float
    lon_deg: float
    baro_alt_ft: float | None
    geom_alt_ft: float | None
    gva: int
    ground_speed_kt: float
    track_deg: float
    vertical_rate_fpm: float
    aerodynamic_class: str
    raw_pii_metadata: Mapping[str, Any]


def _normalize_provider(provider: str) -> str:
    norm = provider.strip().lower().replace("_", ".").replace("-", ".")
    if norm in ("adsb.lol", "adsblol"):
        return "adsb.lol"
    if norm in ("flyitalyadsb", "fly.italy.adsb"):
        return "flyitalyadsb"
    return norm


def compute_backoff_delay_s(base_interval_s: float, retry_index: int, max_backoff_s: float = 60.0) -> float:
    """Compute polite exponential backoff delay on HTTP 429 responses (`DR-15` Part 1)."""
    if retry_index < 0:
        raise ValueError(f"retry_index must be >= 0, got {retry_index}")
    delay = float(base_interval_s) * (2.0 ** int(retry_index))
    return min(float(max_backoff_s), delay)


def validate_polite_poll_config(
    provider: str,
    poll_interval_s: float,
    *,
    commercial_release: bool = True,
    strip_upstream_json: bool = True,
) -> tuple[str, str]:
    """Fail-closed validation of DR-15 ADS-B provider legality and polite poll spacing."""
    verdict = assert_adsb_provider_allowed(
        provider,
        commercial_release=commercial_release,
        strip_upstream_json=strip_upstream_json,
    )
    canonical = _normalize_provider(provider)
    if canonical in MIN_POLL_INTERVAL_SEC:
        min_s = MIN_POLL_INTERVAL_SEC[canonical]
        if float(poll_interval_s) < min_s:
            raise RealSkyComplianceError(
                f"Poll interval {poll_interval_s:.2f} s violates DR-15 polite floor "
                f">= {min_s:.1f} s for '{canonical}'."
            )
    return canonical, verdict


class MetarQnhCache:
    """Rate-limited (`<= 100 req/min`) and TTL-cached (`>= 15 min`) AviationWeather.gov METAR client."""

    def __init__(
        self,
        *,
        ttl_s: float = METAR_MIN_TTL_SEC,
        clock_fn: Callable[[], float] = time.monotonic,
        fetch_fn: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        if float(ttl_s) < METAR_MIN_TTL_SEC:
            raise ValueError(f"METAR cache TTL must be >= {METAR_MIN_TTL_SEC:.0f} s (15 min), got {ttl_s}")
        self._ttl_s = float(ttl_s)
        self._clock = clock_fn
        self._fetch_fn = fetch_fn if fetch_fn is not None else self._http_fetch_metar
        self._cache: dict[str, MetarObservation] = {}
        self._last_request_mono_s: float | None = None

    @staticmethod
    def _http_fetch_metar(station_icao: str) -> Mapping[str, Any]:
        icao = station_icao.strip().upper()
        url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "OpenXVS-RealSky-Tier0A/3.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        return {}

    def get_observation(
        self,
        station_icao: str,
        *,
        fallback_qnh_hpa: float = ISA_P0_HPA,
        fallback_t_amb_k: float = ISA_T0_K,
        fallback_station_elev_m: float = 0.0,
    ) -> MetarObservation:
        icao = station_icao.strip().upper()
        now = float(self._clock())
        if icao in self._cache:
            cached = self._cache[icao]
            if (now - cached.fetched_mono_s) < self._ttl_s:
                return cached

        if self._last_request_mono_s is not None:
            elapsed = now - self._last_request_mono_s
            if elapsed < METAR_MIN_SPACING_SEC:
                time.sleep(METAR_MIN_SPACING_SEC - elapsed)
                now = float(self._clock())

        self._last_request_mono_s = now
        qnh = float(fallback_qnh_hpa)
        t_amb = float(fallback_t_amb_k)
        elev = float(fallback_station_elev_m)
        source = "ISA_FALLBACK"

        try:
            rec = self._fetch_fn(icao)
            if "altim" in rec and rec["altim"] is not None:
                qnh = float(rec["altim"])
                source = "AVIATIONWEATHER_METAR"
            if "temp" in rec and rec["temp"] is not None:
                t_amb = float(rec["temp"]) + 273.15
            if "elev" in rec and rec["elev"] is not None:
                elev = float(rec["elev"])
        except Exception:
            source = "ISA_FALLBACK"

        obs = MetarObservation(
            station_icao=icao,
            qnh_hpa=qnh,
            t_amb_k=t_amb,
            station_elev_m=elev,
            fetched_mono_s=now,
            source=source,
        )
        self._cache[icao] = obs
        return obs


def build_adsb_poll_url(provider: str, lat_deg: float, lon_deg: float, radius_nm: float) -> str:
    """Build DR-15 verified 2026 public ADS-B endpoint URL."""
    canonical = _normalize_provider(provider)
    r_val = max(1, int(round(float(radius_nm))))
    if canonical == "adsb.lol":
        return f"https://api.adsb.lol/v2/point/{float(lat_deg):.5f}/{float(lon_deg):.5f}/{r_val}"
    if canonical == "flyitalyadsb":
        r_km = max(1, min(500, int(round(float(radius_nm) * 1.852))))
        return f"https://api.flyitalyadsb.com/api/aircraft/v2/point/{float(lat_deg):.5f}/{float(lon_deg):.5f}/{r_km}"
    raise RealSkyComplianceError(f"No verified DR-15 public endpoint for provider {provider!r}")


def _classify_adsb_category(category_code: str) -> str:
    code = category_code.strip().upper()
    if code == "A1":
        return "LIGHT_SINGLE"
    if code == "A2":
        return "TWIN_TURBOPROP"
    if code in ("A3", "A4"):
        return "NARROWBODY_JET"
    if code == "A5":
        return "WIDEBODY_JET"
    if code == "A7":
        return "ROTORCRAFT"
    if code in ("B1", "B4"):
        return "GLIDER_ULTRALIGHT"
    return "CIVIL_UNKNOWN"


def parse_adsbx_v2_snapshot(payload: Mapping[str, Any], *, rx_time_s: float) -> list[AdsbFix]:
    """Parse transient ADSBx v2 / readsb JSON into in-memory `AdsbFix` items (never persisted raw)."""
    ac_list = payload["ac"] if "ac" in payload and isinstance(payload["ac"], list) else []
    fixes: list[AdsbFix] = []
    for item in ac_list:
        if not isinstance(item, Mapping):
            continue
        if "lat" not in item or "lon" not in item:
            continue
        if item["lat"] is None or item["lon"] is None:
            continue

        raw_hex = str(item["hex"]) if "hex" in item and item["hex"] is not None else "anon"
        baro_raw = item["alt_baro"] if "alt_baro" in item else None
        baro_ft = float(baro_raw) if isinstance(baro_raw, (int, float)) else None
        geom_raw = item["alt_geom"] if "alt_geom" in item else None
        geom_ft = float(geom_raw) if isinstance(geom_raw, (int, float)) else None
        if baro_ft is None and geom_ft is None:
            continue

        gva_raw = item["gva"] if "gva" in item and isinstance(item["gva"], (int, float)) else 0
        gs_raw = item["gs"] if "gs" in item and isinstance(item["gs"], (int, float)) else 0.0
        trk_raw = item["track"] if "track" in item and isinstance(item["track"], (int, float)) else 0.0
        vr_raw = item["baro_rate"] if "baro_rate" in item and isinstance(item["baro_rate"], (int, float)) else 0.0
        cat_raw = str(item["category"]) if "category" in item and item["category"] is not None else ""

        fixes.append(
            AdsbFix(
                track_key=raw_hex,
                t_rx_s=float(rx_time_s),
                lat_deg=float(item["lat"]),
                lon_deg=float(item["lon"]),
                baro_alt_ft=baro_ft,
                geom_alt_ft=geom_ft,
                gva=int(gva_raw),
                ground_speed_kt=float(gs_raw),
                track_deg=float(trk_raw),
                vertical_rate_fpm=float(vr_raw),
                aerodynamic_class=_classify_adsb_category(cat_raw),
                raw_pii_metadata={
                    "icao24": raw_hex,
                    "callsign": str(item["flight"]).strip() if "flight" in item and item["flight"] is not None else "",
                    "tail_number": str(item["r"]).strip() if "r" in item and item["r"] is not None else "",
                    "raw_adsb_json": json.dumps(dict(item), sort_keys=True),
                },
            )
        )
    return fixes


def poll_live_adsb_snapshots(
    *,
    provider: str,
    lat_deg: float,
    lon_deg: float,
    radius_nm: float,
    poll_interval_s: float,
    num_polls: int,
    strip_upstream_json: bool = True,
    api_key: str | None = None,
    max_429_retries: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[list[AdsbFix]]:
    """Politely poll a DR-15 approved public ADS-B API with exponential HTTP 429 backoff."""
    canonical, _ = validate_polite_poll_config(
        provider,
        poll_interval_s,
        commercial_release=True,
        strip_upstream_json=strip_upstream_json,
    )
    url = build_adsb_poll_url(canonical, lat_deg, lon_deg, radius_nm)
    headers = {"User-Agent": "OpenXVS-RealSky-Tier0A/3.0"}
    if api_key is not None and api_key.strip():
        headers["X-API-Key"] = api_key.strip()

    snapshots: list[list[AdsbFix]] = []
    for idx in range(int(num_polls)):
        if idx > 0:
            sleep_fn(float(poll_interval_s))
        retry = 0
        while True:
            req = urllib.request.Request(url, headers=headers)
            try:
                t_now = time.time()
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                snapshots.append(parse_adsbx_v2_snapshot(payload, rx_time_s=t_now))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and retry < int(max_429_retries):
                    delay = compute_backoff_delay_s(poll_interval_s, retry)
                    retry += 1
                    sleep_fn(delay)
                    continue
                raise
    return snapshots


def build_sky_tilted_r_ned_to_cam(*, azimuth_deg: float = 0.0, pitch_up_deg: float = 22.0) -> np.ndarray:
    """Build static `R_ned_to_cam` looking along `azimuth_deg` (from North) pitched up `pitch_up_deg`.

    Camera convention: `+Z_cam` optical axis (forward), `+X_cam` right, `+Y_cam` down.
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(pitch_up_deg))
    z_axis = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)], dtype=np.float64)
    x_axis = np.array([-math.sin(az), math.cos(az), 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def _ephemeral_track_id(session_salt: str, track_key: str) -> str:
    digest = hashlib.sha256(f"{session_salt}:{track_key}".encode("utf-8")).hexdigest()[:12]
    return f"trk_{digest}"


def assert_no_pii_or_upstream_leak(
    records: Sequence[Mapping[str, Any]],
    *,
    forbidden_values: Sequence[str] = (),
) -> None:
    """Fail-closed audit proving zero ICAO24, callsign, tail number, or raw ADS-B JSON leaked."""
    serialized = json.dumps(list(records), sort_keys=True)
    for rec in records:
        for key in rec.keys():
            if str(key).lower() in FORBIDDEN_LEAK_KEYS:
                raise RealSkyComplianceError(f"Forbidden PII/upstream key leaked into output: {key!r}")
    for secret in forbidden_values:
        tok = str(secret).strip()
        if len(tok) >= 4 and tok in serialized:
            raise RealSkyComplianceError(f"Sensitive identifier '{tok}' leaked into sanitized output!")


def _assert_sources_not_quarantined(sources: Sequence[str]) -> None:
    for src in sources:
        low = src.strip().lower()
        if low == "pass":
            raise RealSkyComplianceError("Quarantined source 'PASS' barred by DR-13/DR-19 policy.")
        for tok in QUARANTINED_SOURCE_TOKENS:
            if tok in low:
                raise RealSkyComplianceError(
                    f"Source '{src}' matched quarantined/sanctioned token '{tok}' (DR-15/DR-19 hard block)."
                )


def parse_utc_iso8601_to_epoch_s(iso_utc: str) -> float:
    """Parse timezone-aware ISO-8601 UTC timestamp string into UNIX epoch seconds (fail-closed on naive)."""
    raw = str(iso_utc).strip()
    if not raw:
        raise ValueError("Empty UTC ISO-8601 timestamp")
    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp {iso_utc!r} is missing explicit timezone (append 'Z' or '+00:00' for exact UTC alignment)"
        )
    return float(dt.astimezone(timezone.utc).timestamp())


def epoch_s_to_utc_iso8601(epoch_s: float) -> str:
    """Format UNIX epoch seconds as canonical millisecond ISO-8601 UTC string."""
    dt = datetime.fromtimestamp(float(epoch_s), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _parse_optional_timestamp_s(record: Mapping[str, Any]) -> float | None:
    for key in ("t_rx_s", "rx_time_s", "timestamp_s", "now"):
        if key in record and record[key] is not None:
            val = float(record[key])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite ADS-B timestamp '{key}': {record[key]!r}")
            return val
    for key in ("t_utc", "utc_iso", "timestamp_utc"):
        if key in record and record[key] is not None:
            return parse_utc_iso8601_to_epoch_s(str(record[key]))
    return None


def load_recorded_adsb_jsonl(path: str | Path) -> list[list[AdsbFix]]:
    """Load offline recorded ADS-B JSONL into transient in-memory `AdsbFix` snapshots.

    Supports both snapshot-per-line ADSBx v2 records (`{"rx_time_s"|"t_utc": ..., "ac": [...]}`)
    and per-fix JSONL lines (`{"t_rx_s"|"t_utc": ..., "lat": ..., "lon": ..., ...}`).
    Raw upstream fields are kept strictly in memory and stripped before export.
    """
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Recorded ADS-B JSONL file not found: {jsonl_path}")

    buckets: dict[float, list[AdsbFix]] = {}
    for line_no, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, Mapping):
            raise ValueError(f"ADS-B JSONL line {line_no} must be a JSON object")

        if "ac" in obj and isinstance(obj["ac"], list):
            rx_s = _parse_optional_timestamp_s(obj)
            if rx_s is None:
                raise ValueError(f"ADS-B snapshot JSONL line {line_no} missing rx_time_s/now/t_utc")
            parsed = parse_adsbx_v2_snapshot(obj, rx_time_s=rx_s)
            if rx_s not in buckets:
                buckets[rx_s] = []
            buckets[rx_s].extend(parsed)
        else:
            rx_s = _parse_optional_timestamp_s(obj)
            if rx_s is None:
                raise ValueError(f"ADS-B fix JSONL line {line_no} missing t_rx_s/rx_time_s/t_utc")
            parsed = parse_adsbx_v2_snapshot({"ac": [dict(obj)]}, rx_time_s=rx_s)
            if rx_s not in buckets:
                buckets[rx_s] = []
            buckets[rx_s].extend(parsed)

    return [buckets[ts] for ts in sorted(buckets.keys())]


def ingest_smartphone_video_frames(
    video_path: str | Path,
    *,
    video_start_utc: str | float,
    clock_offset_ms: float = 0.0,
    sample_fps: float = 2.0,
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    """Decode Tier-0A smartphone `.mp4`/`.mov` frames via OpenCV and compute exact UTC timestamps.

    Per-frame UTC timestamp: `t_utc_s = start_utc_s + pts_s + clock_offset_ms / 1000.0`.
    Frames stay strictly in memory here and are sky-masked (`>= +5.0 deg`) BEFORE any disk write.
    """
    vpath = Path(video_path)
    if not vpath.is_file():
        raise FileNotFoundError(f"Smartphone video file not found: {vpath}")
    if vpath.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video suffix {vpath.suffix!r}; expected one of {sorted(SUPPORTED_VIDEO_SUFFIXES)}"
        )

    fps_req = float(sample_fps)
    if not math.isfinite(fps_req) or fps_req <= 0.0:
        raise ValueError(f"sample_fps must be finite and > 0, got {sample_fps!r}")
    offset_ms = float(clock_offset_ms)
    if not math.isfinite(offset_ms):
        raise ValueError(f"clock_offset_ms must be finite, got {clock_offset_ms!r}")

    if isinstance(video_start_utc, (int, float)) and not isinstance(video_start_utc, bool):
        start_utc_s = float(video_start_utc)
        if not math.isfinite(start_utc_s):
            raise ValueError(f"Non-finite video_start_utc: {video_start_utc!r}")
    else:
        start_utc_s = parse_utc_iso8601_to_epoch_s(str(video_start_utc))

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV failed to open video file: {vpath}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(src_fps) or src_fps <= 0.0:
            raise ValueError(f"Video '{vpath.name}' reported invalid FPS: {src_fps!r}")

        step_s = 1.0 / fps_req
        frames_bgr: list[np.ndarray] = []
        frame_times_s: list[float] = []
        raw_idx = 0
        prev_pos_msec: float | None = None
        next_sample_pts_s: float | None = None
        width_px = 0
        height_px = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            # Prefer container PTS when strictly monotonic; fall back to exact frame index / src_fps
            if raw_idx == 0:
                pts_s = max(0.0, pos_msec / 1000.0) if (math.isfinite(pos_msec) and pos_msec >= 0.0) else 0.0
                prev_pos_msec = pos_msec
            elif math.isfinite(pos_msec) and prev_pos_msec is not None and pos_msec > prev_pos_msec:
                pts_s = pos_msec / 1000.0
                prev_pos_msec = pos_msec
            else:
                pts_s = float(raw_idx) / src_fps
                prev_pos_msec = pts_s * 1000.0

            height_px, width_px = int(frame.shape[0]), int(frame.shape[1])
            if next_sample_pts_s is None or pts_s >= (next_sample_pts_s - 1e-6):
                t_utc_s = start_utc_s + pts_s + (offset_ms / 1000.0)
                frames_bgr.append(frame.copy())
                frame_times_s.append(t_utc_s)
                if next_sample_pts_s is None:
                    next_sample_pts_s = pts_s + step_s
                else:
                    while next_sample_pts_s <= pts_s + 1e-6:
                        next_sample_pts_s += step_s
            raw_idx += 1
    finally:
        cap.release()

    if not frames_bgr:
        raise ValueError(f"No decodable video frames found in {vpath}")

    meta: dict[str, Any] = {
        "video_name": vpath.name,
        "src_fps": round(src_fps, 4),
        "sample_fps": round(fps_req, 4),
        "total_raw_frames": raw_idx,
        "sampled_frames": len(frames_bgr),
        "width_px": width_px,
        "height_px": height_px,
        "video_start_utc": epoch_s_to_utc_iso8601(start_utc_s),
        "clock_offset_ms": round(offset_ms, 3),
    }
    return frames_bgr, frame_times_s, meta


def export_sky_masked_frame_shard(
    frames_bgr: Sequence[np.ndarray],
    *,
    intrinsics: CameraIntrinsics,
    r_ned_to_cam: np.ndarray,
    elevation_mask_deg: float,
    out_tar_path: Path,
    frame_format: str = "png",
) -> tuple[list[np.ndarray], np.ndarray, list[str], str, int]:
    """Apply `>= +5.0 deg` sky-only pixel mask BEFORE encoding and write a deterministic tar shard.

    Zero unmasked pixels ever touch disk. Every pixel `< elevation_mask_deg` is zeroed in memory
    and asserted `== 0` before PNG/JPEG serialization into `out_tar_path`.
    """
    if not frames_bgr:
        raise ValueError("frames_bgr must be non-empty")
    fmt = frame_format.strip().lower()
    if fmt not in ("png", "jpg"):
        raise ValueError(f"Unsupported frame_format {frame_format!r}; expected 'png' or 'jpg'")

    h_px, w_px = int(frames_bgr[0].shape[0]), int(frames_bgr[0].shape[1])
    sky_mask = compute_sky_elevation_mask(
        intrinsics,
        r_ned_to_cam,
        min_elev_deg=elevation_mask_deg,
        image_shape=(h_px, w_px),
    )
    has_ground_pixels = bool(np.any(~sky_mask))

    ext = ".png" if fmt == "png" else ".jpg"
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3] if fmt == "png" else [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    masked_frames: list[np.ndarray] = []
    frame_rel_names: list[str] = []
    out_tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_tar_path, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for idx, raw_frame in enumerate(frames_bgr):
            if int(raw_frame.shape[0]) != h_px or int(raw_frame.shape[1]) != w_px:
                raise ValueError(
                    f"Frame {idx} shape {raw_frame.shape[:2]} mismatches video shape {(h_px, w_px)}"
                )
            masked = apply_sky_elevation_mask(
                raw_frame,
                intrinsics,
                r_ned_to_cam,
                min_elev_deg=elevation_mask_deg,
                precomputed_mask=sky_mask,
            )
            if has_ground_pixels and int(masked[~sky_mask].max()) != 0:
                raise RealSkyComplianceError(
                    f"Pre-encode sky mask postcondition failed on frame {idx}: below-horizon pixel != 0"
                )
            ok, buf = cv2.imencode(ext, masked, encode_params)
            if not ok:
                raise RuntimeError(f"OpenCV image encoding ({ext}) failed on frame {idx}")
            img_bytes = buf.tobytes()
            rel_name = f"frames/frame_{idx:06d}{ext}"
            ti = tarfile.TarInfo(name=rel_name)
            ti.size = len(img_bytes)
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(img_bytes))
            masked_frames.append(masked)
            frame_rel_names.append(rel_name)

    tar_bytes = out_tar_path.read_bytes()
    tar_sha256 = hashlib.sha256(tar_bytes).hexdigest()
    return masked_frames, sky_mask, frame_rel_names, tar_sha256, len(tar_bytes)


def _d2_scene_normalize_gray(gray_f32: np.ndarray, min_std: float = 8.0, clamp_val: float = 5.0) -> np.ndarray:
    mean = float(gray_f32.mean())
    std = max(float(gray_f32.std()), float(min_std))
    return np.clip((gray_f32 - mean) / std, -clamp_val, clamp_val).astype(np.float32)


def _d2_sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -40.0, 40.0)))


def _d2_decode_peaks(
    hm_logits: np.ndarray,
    offset: np.ndarray,
    log_scale: np.ndarray,
    peak_mask: np.ndarray,
    *,
    score_thr: float = 0.25,
    stride: float = 2.0,
    max_dets: int = 16,
) -> tuple[list[tuple[float, float, float, float, float]], float]:
    """Decode local peaks from pinned `XVS-D2` ONNX outputs (`hm_logits, offset, log_scale, peak_mask`)."""
    hm = hm_logits[0, 0].astype(np.float32)
    off = offset[0].astype(np.float32)
    lsc = log_scale[0].astype(np.float32)
    pmk = peak_mask[0, 0].astype(np.float32)

    probs = _d2_sigmoid_np(hm)
    h_sz, w_sz = probs.shape
    max_prob = float(probs.max())

    keep = (probs >= float(score_thr)) & (pmk > 0.5)
    ys, xs = np.where(keep)
    if ys.size == 0:
        return [], max_prob

    scores = probs[ys, xs]
    order = np.argsort(-scores)[: int(max_dets)]
    dets: list[tuple[float, float, float, float, float]] = []
    for idx in order:
        cy = int(ys[idx])
        cx = int(xs[idx])
        score = float(probs[cy, cx])

        y0, y1 = max(0, cy - 1), min(h_sz, cy + 2)
        x0, x1 = max(0, cx - 1), min(w_sz, cx + 2)
        patch = hm[y0:y1, x0:x1]
        exp_p = np.exp(patch - float(patch.max()))
        w_p = exp_p / max(float(exp_p.sum()), 1.0e-12)
        gy = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
        gx = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
        soft_u = float((w_p * gx).sum())
        soft_v = float((w_p * gy).sum())

        cell_u = 0.5 * (float(cx) + float(off[0, cy, cx])) + 0.5 * soft_u
        cell_v = 0.5 * (float(cy) + float(off[1, cy, cx])) + 0.5 * soft_v
        u_px = (cell_u + 0.5) * stride
        v_px = (cell_v + 0.5) * stride

        s_mom = math.exp(max(-4.0, min(8.0, float(lsc[0, cy, cx]))))
        r_wh = math.exp(max(-3.0, min(3.0, float(lsc[1, cy, cx]))))
        bw = max(6.0, min(96.0, s_mom * math.sqrt(r_wh)))
        bh = max(6.0, min(96.0, s_mom / math.sqrt(r_wh)))
        dets.append((u_px - 0.5 * bw, v_px - 0.5 * bh, u_px + 0.5 * bw, v_px + 0.5 * bh, score))
    return dets, max_prob


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_realsky_xvs_d2(
    *,
    frames_tar_path: Path,
    labels: Sequence[Mapping[str, Any]],
    model_path: Path = DEFAULT_XVS_D2_ONNX,
    score_thr: float = 0.25,
) -> dict[str, Any]:
    """Run pinned `XVS-D2` FP16 ONNX zero-shot evaluation over exported sky-masked frames (`DR-15`)."""
    import onnxruntime as ort

    mpath = Path(model_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"Pinned XVS-D2 ONNX model not found: {mpath}")
    if not frames_tar_path.is_file():
        raise FileNotFoundError(f"Sky-masked frame shard tar not found: {frames_tar_path}")

    model_sha256 = hashlib.sha256(mpath.read_bytes()).hexdigest()
    sess = ort.InferenceSession(str(mpath), providers=["CPUExecutionProvider"])
    in_types = {inp.name: inp.type for inp in sess.get_inputs()}
    x_dtype = np.float16 if ("x_t" in in_types and "float16" in in_types["x_t"]) else np.float32
    h_dtype = np.float16 if ("h_prev_warped" in in_types and "float16" in in_types["h_prev_warped"]) else np.float32

    # Group sanitized GT boxes by frame_idx
    gt_by_frame: dict[int, list[dict[str, float]]] = {}
    for rec in labels:
        f_idx = int(rec["frame_idx"])
        if f_idx not in gt_by_frame:
            gt_by_frame[f_idx] = []
        gt_by_frame[f_idx].append(
            {
                "u_px": float(rec["u_px"]),
                "v_px": float(rec["v_px"]),
                "w_px": float(rec["w_px"]),
                "h_px": float(rec["h_px"]),
            }
        )

    # Read sky-masked frames back from exported tar shard in deterministic frame order
    decoded_frames: list[tuple[int, str, np.ndarray]] = []
    with tarfile.open(frames_tar_path, mode="r") as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)
        for idx, member in enumerate(members):
            file_obj = tf.extractfile(member)
            if file_obj is None:
                continue
            buf = np.frombuffer(file_obj.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to decode sky-masked frame '{member.name}' from {frames_tar_path}")
            decoded_frames.append((idx, member.name, img))

    if not decoded_frames:
        raise ValueError(f"No sky-masked frames decoded from {frames_tar_path}")

    h_prev = np.zeros((1, 24, D2_CROP_SIZE // 4, D2_CROP_SIZE // 4), dtype=np.float32)
    gray_hist: list[np.ndarray] = []
    ones_mask = np.ones((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)
    zeros_plane = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_preds = 0
    center_errors: list[float] = []
    latencies_ms: list[float] = []
    max_probs: list[float] = []
    per_frame: list[dict[str, Any]] = []

    for frame_idx, rel_name, frame_bgr in decoded_frames:
        t0 = time.perf_counter()
        h_in, w_in = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        crop_bgr = cv2.resize(frame_bgr, (D2_CROP_SIZE, D2_CROP_SIZE), interpolation=cv2.INTER_AREA)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        c0 = _d2_scene_normalize_gray(crop_gray)
        if len(gray_hist) >= 1:
            c1 = _d2_scene_normalize_gray(gray_hist[-1])
            r1 = ones_mask
        else:
            c1 = zeros_plane
            r1 = zeros_plane

        if len(gray_hist) >= 5:
            c2 = _d2_scene_normalize_gray(gray_hist[-5])
            r5 = ones_mask
        else:
            c2 = zeros_plane
            r5 = zeros_plane

        c3 = (c0 - c1) * r1
        c4 = (c0 - c2) * r5
        c5 = r1 * r5
        x_t = np.stack([c0, c1, c2, c3, c4, c5], axis=0)[np.newaxis, :, :, :].astype(np.float32)

        outs = sess.run(
            None,
            {
                "x_t": x_t.astype(x_dtype),
                "h_prev_warped": h_prev.astype(h_dtype),
            },
        )
        hm_logits = outs[0].astype(np.float32)
        offset = outs[1].astype(np.float32)
        log_scale = outs[2].astype(np.float32)
        peak_mask = outs[4].astype(np.float32)
        h_prev = outs[5].astype(np.float32)

        gray_hist.append(crop_gray)
        if len(gray_hist) > 8:
            gray_hist.pop(0)

        sx = float(w_in) / float(D2_CROP_SIZE)
        sy = float(h_in) / float(D2_CROP_SIZE)
        raw_dets, max_prob = _d2_decode_peaks(
            hm_logits,
            offset,
            log_scale,
            peak_mask,
            score_thr=score_thr,
        )
        frame_preds: list[tuple[float, float, float, float, float]] = [
            (d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy, d[4]) for d in raw_dets
        ]

        # Multi-crop sky tile probe when high-res full-frame pass yields zero hits
        if not frame_preds and max(w_in, h_in) > 384:
            tile_w = min(w_in, max(D2_CROP_SIZE, w_in // 2))
            tile_h = min(h_in, max(D2_CROP_SIZE, h_in // 2))
            sky_y = max(0, min(h_in - tile_h, int(h_in * 0.08)))
            h_zero = np.zeros((1, 24, D2_CROP_SIZE // 4, D2_CROP_SIZE // 4), dtype=np.float32)
            for tx0, ty0 in (
                (max(0, (w_in - tile_w) // 2), sky_y),
                (0, sky_y),
                (max(0, w_in - tile_w), sky_y),
            ):
                patch = frame_bgr[ty0 : ty0 + tile_h, tx0 : tx0 + tile_w]
                p_rs = cv2.resize(patch, (D2_CROP_SIZE, D2_CROP_SIZE), interpolation=cv2.INTER_AREA)
                p_gray = cv2.cvtColor(p_rs, cv2.COLOR_BGR2GRAY).astype(np.float32)
                pc0 = _d2_scene_normalize_gray(p_gray)
                px_t = np.stack([pc0, zeros_plane, zeros_plane, zeros_plane, zeros_plane, zeros_plane], axis=0)[
                    np.newaxis, :, :, :
                ].astype(np.float32)
                p_outs = sess.run(
                    None,
                    {
                        "x_t": px_t.astype(x_dtype),
                        "h_prev_warped": h_zero.astype(h_dtype),
                    },
                )
                tdets, tprob = _d2_decode_peaks(
                    p_outs[0].astype(np.float32),
                    p_outs[1].astype(np.float32),
                    p_outs[2].astype(np.float32),
                    p_outs[4].astype(np.float32),
                    score_thr=score_thr,
                )
                if tprob > max_prob:
                    max_prob = tprob
                tsx = float(tile_w) / float(D2_CROP_SIZE)
                tsy = float(tile_h) / float(D2_CROP_SIZE)
                for td in tdets:
                    frame_preds.append(
                        (tx0 + td[0] * tsx, ty0 + td[1] * tsy, tx0 + td[2] * tsx, ty0 + td[3] * tsy, td[4])
                    )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        max_probs.append(max_prob)

        gts = gt_by_frame[frame_idx] if frame_idx in gt_by_frame else []
        matched_gt: set[int] = set()
        tp_f = 0
        fp_f = 0

        for pred in sorted(frame_preds, key=lambda d: -d[4]):
            pu = 0.5 * (pred[0] + pred[2])
            pv = 0.5 * (pred[1] + pred[3])
            pbox = (pred[0], pred[1], pred[2], pred[3])
            best_j: int | None = None
            best_dist = float("inf")
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                gu, gv, gw, gh = gt["u_px"], gt["v_px"], gt["w_px"], gt["h_px"]
                gbox = (gu - 0.5 * gw, gv - 0.5 * gh, gu + 0.5 * gw, gv + 0.5 * gh)
                dist = math.hypot(pu - gu, pv - gv)
                match_rad = max(24.0, 1.5 * max(gw, gh))
                if (_box_iou(pbox, gbox) >= 0.1 or dist <= match_rad) and dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None:
                matched_gt.add(best_j)
                tp_f += 1
                center_errors.append(best_dist)
            else:
                fp_f += 1

        fn_f = len(gts) - len(matched_gt)
        total_tp += tp_f
        total_fp += fp_f
        total_fn += fn_f
        total_gt += len(gts)
        total_preds += len(frame_preds)
        per_frame.append(
            {
                "frame_idx": frame_idx,
                "frame_file": rel_name,
                "max_heatmap_prob": round(max_prob, 5),
                "num_preds": len(frame_preds),
                "num_gt": len(gts),
                "tp": tp_f,
                "fp": fp_f,
                "fn": fn_f,
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return {
        "schema": "openxvs.realsky.xvs_d2_eval/1",
        "model_name": mpath.name,
        "model_sha256": model_sha256,
        "score_thr": float(score_thr),
        "frames_evaluated": len(decoded_frames),
        "total_gt_targets": total_gt,
        "total_predictions": total_preds,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "mean_center_err_px": round(float(np.mean(center_errors)), 3) if center_errors else None,
        "mean_max_heatmap_prob": round(float(np.mean(max_probs)), 5),
        "max_heatmap_prob": round(float(np.max(max_probs)), 5),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "per_frame": per_frame,
    }


def build_realsky_croissant_manifest(
    *,
    dataset_name: str,
    version: str,
    license_spdx: str,
    labels_file_name: str,
    labels_sha256: str,
    labels_size_bytes: int,
    record_count: int,
    elevation_mask_deg: float,
    provider_used: str,
    date_published: str = "2026-11-21",
    frames_shard_file_name: str | None = None,
    frames_shard_sha256: str | None = None,
    frames_shard_size_bytes: int | None = None,
    frames_count: int = 0,
    frame_encoding_format: str = "image/png",
) -> dict[str, Any]:
    """Emit validated Croissant 1.1 + RAI 1.0 JSON-LD metadata for OpenXVS-RealSky-v1 (`DR-15`/`DR-19`)."""
    if license_spdx not in ALLOWED_RELEASE_LICENSES:
        raise RealSkyComplianceError(
            f"Unsupported release license {license_spdx!r}; must be in {sorted(ALLOWED_RELEASE_LICENSES)}"
        )
    if float(elevation_mask_deg) < MIN_SKY_ELEVATION_MASK_DEG:
        raise RealSkyComplianceError(
            f"elevation_mask_deg={elevation_mask_deg} violates >= {MIN_SKY_ELEVATION_MASK_DEG} deg sky-only lock."
        )
    _assert_sources_not_quarantined([dataset_name, provider_used])

    distribution: list[dict[str, Any]] = [
        {
            "@type": "cr:FileObject",
            "@id": "realsky_labels_jsonl",
            "name": str(labels_file_name),
            "contentUrl": str(labels_file_name),
            "encodingFormat": "application/x-ndjson",
            "sha256": str(labels_sha256),
            "contentSize": f"{int(labels_size_bytes)} B",
        }
    ]
    record_fields: list[dict[str, str]] = [
        {"@type": "cr:Field", "name": "ephemeral_track_id", "dataType": "sc:Text"},
        {"@type": "cr:Field", "name": "aerodynamic_class", "dataType": "sc:Text"},
        {"@type": "cr:Field", "name": "frame_idx", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "u_px", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "v_px", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "w_px", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "h_px", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "elevation_mask_deg", "dataType": "sc:Float"},
    ]

    if frames_shard_file_name is not None:
        if frames_shard_sha256 is None or frames_shard_size_bytes is None:
            raise ValueError(
                "frames_shard_sha256 and frames_shard_size_bytes are required when frames_shard_file_name is set"
            )
        ext_glob = "frames/*.png" if frame_encoding_format == "image/png" else "frames/*.jpg"
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": "realsky_frames_tar",
                "name": str(frames_shard_file_name),
                "contentUrl": str(frames_shard_file_name),
                "encodingFormat": "application/x-tar",
                "sha256": str(frames_shard_sha256),
                "contentSize": f"{int(frames_shard_size_bytes)} B",
            }
        )
        distribution.append(
            {
                "@type": "cr:FileSet",
                "@id": "realsky_masked_frames",
                "name": "realsky_masked_frames",
                "description": (
                    f"{int(frames_count)} sky-masked (>= +{float(elevation_mask_deg):.1f} deg) "
                    "optical frames packed pre-encode into WebDataset-style tar shard."
                ),
                "containedIn": {"@id": "realsky_frames_tar"},
                "encodingFormat": str(frame_encoding_format),
                "includes": ext_glob,
            }
        )
        record_fields.extend(
            [
                {"@type": "cr:Field", "name": "frame_file", "dataType": "sc:Text"},
                {"@type": "cr:Field", "name": "frame_utc_iso", "dataType": "sc:Text"},
                {"@type": "cr:Field", "name": "visual_snapped", "dataType": "sc:Boolean"},
                {"@type": "cr:Field", "name": "visual_snap_snr", "dataType": "sc:Float"},
            ]
        )

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "dct:conformsTo": [CROISSANT_1_1_URI, CROISSANT_RAI_1_0_URI],
        "name": str(dataset_name),
        "version": str(version),
        "description": (
            "OpenXVS-RealSky-v1 Tier-0A civil aviation real-sky optical auto-labeled shard "
            "generated via offline non-causal Cubic Hermite projection with upstream ADS-B JSON "
            "and aircraft identifiers permanently stripped."
        ),
        "license": str(license_spdx),
        "datePublished": str(date_published),
        "creator": {"@type": "Organization", "name": "OpenXVS Civil Aviation Safety Research"},
        "isLiveDataset": False,
        "inLanguage": "en",
        "distribution": distribution,
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "realsky_2d_labels",
                "name": "realsky_2d_labels",
                "description": f"{int(record_count)} sanitized 2D civil aircraft bounding-box labels.",
                "field": record_fields,
            }
        ],
        "prov:wasDerivedFrom": "Original contributor optical camera frames (transient ADS-B pointing cue stripped)",
        "prov:wasGeneratedBy": "OpenXVS scripts/xvs/realsky_window_capture.py (DR-15 Cubic Hermite + Sanitizer)",
        "rai:dataCollection": (
            "Static window/balcony 4K optical capture paired with transient public ADS-B pointing cue "
            "and METAR QNH/GEOID18 vertical datum reconciliation; zero upstream API JSON redistributed."
        ),
        "rai:dataCollectionType": ["Direct Measurement"],
        "rai:dataCollectionRawData": "Contributor-owned optical frames; upstream ADS-B telemetry discarded pre-export.",
        "rai:hasSyntheticData": False,
        "rai:personalSensitiveInformation": (
            f"None — >= +{float(elevation_mask_deg):.1f} deg sky-only elevation mask enforced pre-encode "
            "and all ICAO24 hex codes, callsigns, and tail numbers permanently stripped."
        ),
        "rai:dataBiases": (
            "Biased toward cooperative transponder-equipped civil traffic along local approach corridors; "
            "non-cooperative GA/glider aircraft are underrepresented."
        ),
        "rai:dataLimitations": (
            "Tier-0A rolling-shutter smartphone capture carries ~15-100 ms NTP/stopwatch jitter prior to "
            "visual contrast lock; not certified for airworthiness."
        ),
        "rai:dataUseCases": (
            "Strictly civil aviation Detect-and-Avoid (DAA) safety research and domain-gap evaluation; "
            "prohibits weapons, targeting, fire-control, and military IFF applications."
        ),
        "rai:dataSocialImpact": (
            "Improves civil unmanned and general aviation mid-air collision avoidance with zero ground-PII capture."
        ),
    }


def validate_realsky_croissant_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail-closed validation of mandatory DR-15/DR-19 Croissant 1.1 + RAI 1.0 fields."""
    if "dct:conformsTo" not in manifest:
        raise ValueError("Missing dct:conformsTo in Croissant manifest")
    conforms = list(manifest["dct:conformsTo"])
    if CROISSANT_1_1_URI not in conforms or CROISSANT_RAI_1_0_URI not in conforms:
        raise ValueError(f"Manifest must conform to Croissant 1.1 and RAI 1.0, got {conforms!r}")
    if "license" not in manifest or manifest["license"] not in ALLOWED_RELEASE_LICENSES:
        lic_val = manifest["license"] if "license" in manifest else None
        raise RealSkyComplianceError(f"Non-permissive manifest license: {lic_val!r}")
    if "rai:hasSyntheticData" not in manifest or manifest["rai:hasSyntheticData"] is not False:
        raise ValueError("OpenXVS-RealSky-v1 manifest must declare rai:hasSyntheticData == false")
    if "distribution" not in manifest or not isinstance(manifest["distribution"], list) or not manifest["distribution"]:
        raise ValueError("Croissant manifest must include non-empty distribution list")
    for dist_item in manifest["distribution"]:
        if isinstance(dist_item, Mapping) and dist_item["@type"] == "cr:FileObject":
            if "sha256" not in dist_item or len(str(dist_item["sha256"])) != 64:
                raise ValueError(f"Croissant cr:FileObject missing 64-hex sha256: {dist_item!r}")
            bytes.fromhex(str(dist_item["sha256"]))
            if "contentSize" not in dist_item or not str(dist_item["contentSize"]).strip():
                raise ValueError(f"Croissant cr:FileObject missing contentSize: {dist_item!r}")
    required_rai = (
        "rai:dataCollection",
        "rai:dataCollectionType",
        "rai:dataCollectionRawData",
        "rai:personalSensitiveInformation",
        "rai:dataBiases",
        "rai:dataLimitations",
        "rai:dataUseCases",
        "rai:dataSocialImpact",
        "prov:wasDerivedFrom",
        "prov:wasGeneratedBy",
    )
    for key in required_rai:
        if key not in manifest or not manifest[key]:
            raise ValueError(f"Missing mandatory Croissant RAI field: {key}")
    raw_dump = json.dumps(dict(manifest), sort_keys=True).lower()
    for tok in QUARANTINED_SOURCE_TOKENS:
        if tok in raw_dump:
            raise RealSkyComplianceError(f"Quarantined source token '{tok}' found in Croissant manifest!")


def run_realsky_window_pipeline(
    *,
    snapshots: Sequence[Sequence[AdsbFix]],
    frame_times_s: Sequence[float],
    site_lat_deg: float,
    site_lon_deg: float,
    site_alt_ellip_m: float,
    geoid_undulation_m: float,
    metar_obs: MetarObservation,
    r_ned_to_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    provider: str,
    poll_interval_s: float,
    elevation_mask_deg: float = MIN_SKY_ELEVATION_MASK_DEG,
    strip_upstream_adsb: bool = True,
    license_spdx: str = "CDLA-Permissive-2.0",
    session_salt: str = "openxvs-realsky-v1-session",
    out_dir: Path,
    video_frames_bgr: Sequence[np.ndarray] | None = None,
    frame_format: str = "png",
    enable_visual_snap: bool = True,
    snap_sigma_px: float = 15.0,
    snap_min_snr: float = 4.5,
    ed25519_private_seed: bytes | None = None,
    eval_xvs_d2: bool = False,
    xvs_d2_model_path: Path = DEFAULT_XVS_D2_ONNX,
    xvs_d2_score_thr: float = 0.25,
) -> dict[str, Any]:
    """Run offline non-causal Cubic Hermite auto-labeling, sky-mask shard export, Ed25519 sign & optional XVS-D2 eval."""
    canonical_provider, _ = validate_polite_poll_config(
        provider,
        poll_interval_s,
        commercial_release=True,
        strip_upstream_json=strip_upstream_adsb,
    )
    if float(elevation_mask_deg) < MIN_SKY_ELEVATION_MASK_DEG:
        raise RealSkyComplianceError(
            f"elevation_mask_deg={elevation_mask_deg:.2f} deg < mandatory {MIN_SKY_ELEVATION_MASK_DEG:.1f} deg."
        )
    if eval_xvs_d2 and video_frames_bgr is None:
        raise ValueError("eval_xvs_d2=True requires video_frames_bgr so sky-masked frames can be evaluated")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-encode >= +5 deg sky-only pixel masking & deterministic tar shard export (zero unmasked disk writes!)
    masked_frames: list[np.ndarray] | None = None
    sky_mask: np.ndarray | None = None
    frame_rel_names: list[str] | None = None
    frames_tar_path: Path | None = None
    frames_tar_sha256: str | None = None
    frames_tar_size: int | None = None
    eff_intrinsics = intrinsics

    if video_frames_bgr is not None:
        if len(video_frames_bgr) != len(frame_times_s):
            raise ValueError(
                f"video_frames_bgr length ({len(video_frames_bgr)}) != frame_times_s length ({len(frame_times_s)})"
            )
        if not video_frames_bgr:
            raise ValueError("video_frames_bgr must be non-empty when provided")
        h_vid, w_vid = int(video_frames_bgr[0].shape[0]), int(video_frames_bgr[0].shape[1])
        if (w_vid, h_vid) != (intrinsics.width_px, intrinsics.height_px):
            sx = float(w_vid) / float(intrinsics.width_px)
            sy = float(h_vid) / float(intrinsics.height_px)
            eff_intrinsics = CameraIntrinsics(
                fx=intrinsics.fx * sx,
                fy=intrinsics.fy * sy,
                cx=intrinsics.cx * sx,
                cy=intrinsics.cy * sy,
                k1=intrinsics.k1,
                k2=intrinsics.k2,
                p1=intrinsics.p1,
                p2=intrinsics.p2,
                k3=intrinsics.k3,
                width_px=w_vid,
                height_px=h_vid,
            )
        frames_tar_path = out_dir / "frames_shard.tar"
        masked_frames, sky_mask, frame_rel_names, frames_tar_sha256, frames_tar_size = (
            export_sky_masked_frame_shard(
                video_frames_bgr,
                intrinsics=eff_intrinsics,
                r_ned_to_cam=r_ned_to_cam,
                elevation_mask_deg=elevation_mask_deg,
                out_tar_path=frames_tar_path,
                frame_format=frame_format,
            )
        )

    # Group fixes per transient track key sorted by rx timestamp
    tracks: dict[str, list[AdsbFix]] = {}
    sensitive_tokens: list[str] = []
    for snap in snapshots:
        for fix in snap:
            if fix.track_key not in tracks:
                tracks[fix.track_key] = []
            tracks[fix.track_key].append(fix)
            for pii_key in ("icao24", "callsign", "tail_number"):
                if pii_key in fix.raw_pii_metadata:
                    val = str(fix.raw_pii_metadata[pii_key]).strip()
                    if val:
                        sensitive_tokens.append(val)

    clean_records: list[dict[str, Any]] = []
    rejected_below_mask = 0
    rejected_behind_cam = 0
    snapped_count = 0
    unsnapped_count = 0

    for track_key in sorted(tracks.keys()):
        fixes = sorted(tracks[track_key], key=lambda f: f.t_rx_s)
        if len(fixes) < 2:
            continue
        ephemeral_id = _ephemeral_track_id(session_salt, track_key)
        ned_positions: list[np.ndarray] = []
        ned_velocities: list[np.ndarray] = []
        for fix in fixes:
            alt_ellip_m = reconcile_ellipsoidal_altitude_m(
                baro_alt_ft=fix.baro_alt_ft,
                geom_alt_ft=fix.geom_alt_ft,
                gva=fix.gva,
                qnh_hpa=metar_obs.qnh_hpa,
                t_amb_k=metar_obs.t_amb_k,
                station_elev_m=metar_obs.station_elev_m,
                geoid_undulation_m=geoid_undulation_m,
            )
            p_ned = geodetic_to_local_ned(
                fix.lat_deg,
                fix.lon_deg,
                alt_ellip_m,
                origin_lat_deg=site_lat_deg,
                origin_lon_deg=site_lon_deg,
                origin_alt_ellip_m=site_alt_ellip_m,
            )
            v_ned = ned_velocity_from_adsb(
                ground_speed_kt=fix.ground_speed_kt,
                track_deg=fix.track_deg,
                vertical_rate_fpm=fix.vertical_rate_fpm,
            )
            ned_positions.append(p_ned)
            ned_velocities.append(v_ned)

        dims = (
            AERODYNAMIC_DIMENSIONS_BY_CLASS[fixes[0].aerodynamic_class]
            if fixes[0].aerodynamic_class in AERODYNAMIC_DIMENSIONS_BY_CLASS
            else AERODYNAMIC_DIMENSIONS_BY_CLASS["CIVIL_UNKNOWN"]
        )

        for frame_idx, t_mid in enumerate(frame_times_s):
            t_f = float(t_mid)
            bracket_idx: int | None = None
            for i in range(len(fixes) - 1):
                if fixes[i].t_rx_s <= t_f <= fixes[i + 1].t_rx_s:
                    bracket_idx = i
                    break
            if bracket_idx is None:
                continue

            p_interp = interpolate_cubic_hermite_ned(
                ned_positions[bracket_idx],
                ned_velocities[bracket_idx],
                ned_positions[bracket_idx + 1],
                ned_velocities[bracket_idx + 1],
                t_mid_s=t_f,
                t_k_s=fixes[bracket_idx].t_rx_s,
                t_k1_s=fixes[bracket_idx + 1].t_rx_s,
            )
            proj = project_ned_to_pixel(p_interp, r_ned_to_cam, eff_intrinsics)
            if proj is None:
                rejected_behind_cam += 1
                continue
            if proj.elevation_deg < float(elevation_mask_deg):
                rejected_below_mask += 1
                continue
            if not proj.in_image_bounds:
                continue

            w_px, h_px = estimate_airframe_box_prior_px(
                slant_range_m=proj.slant_range_m,
                intrinsics=eff_intrinsics,
                dimensions=dims,
                look_elevation_rad=math.radians(proj.elevation_deg),
            )
            u_final = proj.u_px
            v_final = proj.v_px
            snap_res: ContrastSnapResult | None = None
            if masked_frames is not None and enable_visual_snap:
                k_adapt = max(9, min(31, int(math.ceil(max(w_px, h_px) * 1.5)) | 1))
                snap_res = snap_target_contrast_centroid(
                    masked_frames[frame_idx],
                    proj.u_px,
                    proj.v_px,
                    sigma_u_px=snap_sigma_px,
                    sigma_v_px=snap_sigma_px,
                    max_sigma=3.0,
                    min_snr=snap_min_snr,
                    kernel_size_px=k_adapt,
                    sky_mask=sky_mask,
                )
                if snap_res.snapped:
                    u_final = snap_res.u_px
                    v_final = snap_res.v_px
                    snapped_count += 1
                else:
                    unsnapped_count += 1

            raw_candidate: dict[str, Any] = {
                "frame_idx": frame_idx,
                "u_px": u_final,
                "v_px": v_final,
                "w_px": w_px,
                "h_px": h_px,
                "slant_range_m": proj.slant_range_m,
            }
            if frame_rel_names is not None:
                raw_candidate["frame_file"] = frame_rel_names[frame_idx]
                raw_candidate["frame_utc_iso"] = epoch_s_to_utc_iso8601(t_f)
            if snap_res is not None:
                raw_candidate["visual_snapped"] = snap_res.snapped
                raw_candidate["visual_snap_snr"] = snap_res.snr
                raw_candidate["visual_snap_du_px"] = snap_res.du_px
                raw_candidate["visual_snap_dv_px"] = snap_res.dv_px
                raw_candidate["visual_snap_polarity"] = snap_res.polarity
            for k, v in fixes[bracket_idx].raw_pii_metadata.items():
                raw_candidate[k] = v

            clean_rec = sanitize_realsky_label_record(
                raw_candidate,
                provider=canonical_provider,
                ephemeral_track_id=ephemeral_id,
                aerodynamic_class=fixes[bracket_idx].aerodynamic_class,
                elevation_mask_deg=elevation_mask_deg,
                commercial_release=True,
                strip_upstream_json=strip_upstream_adsb,
            )
            clean_records.append(clean_rec)

    assert_no_pii_or_upstream_leak(clean_records, forbidden_values=sensitive_tokens)

    labels_path = out_dir / "labels.jsonl"
    labels_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in clean_records)
    labels_bytes = labels_text.encode("utf-8")
    labels_path.write_bytes(labels_bytes)
    labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()

    frame_mime = "image/png" if frame_format.strip().lower() == "png" else "image/jpeg"
    croissant = build_realsky_croissant_manifest(
        dataset_name="OpenXVS-RealSky-v1-WindowShard",
        version="1.0.0",
        license_spdx=license_spdx,
        labels_file_name=labels_path.name,
        labels_sha256=labels_sha256,
        labels_size_bytes=len(labels_bytes),
        record_count=len(clean_records),
        elevation_mask_deg=elevation_mask_deg,
        provider_used=canonical_provider,
        frames_shard_file_name=frames_tar_path.name if frames_tar_path is not None else None,
        frames_shard_sha256=frames_tar_sha256,
        frames_shard_size_bytes=frames_tar_size,
        frames_count=len(masked_frames) if masked_frames is not None else 0,
        frame_encoding_format=frame_mime,
    )
    validate_realsky_croissant_manifest(croissant)

    croissant_path = out_dir / "croissant.json"
    croissant_bytes = (json.dumps(croissant, indent=2, sort_keys=True) + "\n").encode("utf-8")
    croissant_path.write_bytes(croissant_bytes)
    croissant_sha256 = hashlib.sha256(croissant_bytes).hexdigest()

    manifest_files: list[dict[str, Any]] = [
        {"name": labels_path.name, "sha256": labels_sha256, "size_bytes": len(labels_bytes)},
        {"name": croissant_path.name, "sha256": croissant_sha256, "size_bytes": len(croissant_bytes)},
    ]
    if frames_tar_path is not None and frames_tar_sha256 is not None and frames_tar_size is not None:
        manifest_files.append(
            {"name": frames_tar_path.name, "sha256": frames_tar_sha256, "size_bytes": frames_tar_size}
        )

    xvs_d2_eval_summary: dict[str, Any] | None = None
    if eval_xvs_d2 and frames_tar_path is not None:
        eval_report = evaluate_realsky_xvs_d2(
            frames_tar_path=frames_tar_path,
            labels=clean_records,
            model_path=xvs_d2_model_path,
            score_thr=xvs_d2_score_thr,
        )
        eval_path = out_dir / "xvs_d2_eval.json"
        eval_bytes = (json.dumps(eval_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        eval_path.write_bytes(eval_bytes)
        eval_sha256 = hashlib.sha256(eval_bytes).hexdigest()
        manifest_files.append({"name": eval_path.name, "sha256": eval_sha256, "size_bytes": len(eval_bytes)})
        xvs_d2_eval_summary = {
            "model_sha256": eval_report["model_sha256"],
            "frames_evaluated": eval_report["frames_evaluated"],
            "precision": eval_report["precision"],
            "recall": eval_report["recall"],
            "f1": eval_report["f1"],
            "mean_latency_ms": eval_report["mean_latency_ms"],
        }

    shard_manifest: dict[str, Any] = {
        "schema": "openxvs.realsky.shard_manifest/1",
        "license": license_spdx,
        "provider_cue": canonical_provider,
        "upstream_adsb_stripped": bool(strip_upstream_adsb),
        "elevation_mask_deg": float(elevation_mask_deg),
        "metar_source": metar_obs.source,
        "metar_qnh_hpa": round(metar_obs.qnh_hpa, 2),
        "record_count": len(clean_records),
        "rejected_below_sky_mask": rejected_below_mask,
        "rejected_behind_camera": rejected_behind_cam,
        "signature_algorithm": "Ed25519",
        "files": manifest_files,
    }
    if masked_frames is not None:
        shard_manifest["exported_frame_count"] = len(masked_frames)
        shard_manifest["visual_snap_locked"] = snapped_count
        shard_manifest["visual_snap_fallback"] = unsnapped_count
    if xvs_d2_eval_summary is not None:
        shard_manifest["xvs_d2_eval"] = xvs_d2_eval_summary

    canonical_manifest = json.dumps(shard_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shard_manifest["manifest_sha256"] = hashlib.sha256(canonical_manifest).hexdigest()

    manifest_path = out_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Detached RFC 8032 Ed25519 signature over manifest_sha256 (`DR-15` / `DR-19`)
    manifest_sig = sign_manifest_sha256(
        shard_manifest["manifest_sha256"],
        private_seed=ed25519_private_seed,
    )
    if not verify_manifest_signature(shard_manifest, manifest_sig):
        raise RuntimeError("Ed25519 manifest signature self-verification failed")
    sig_path = out_dir / "manifest_sig.json"
    sig_path.write_text(json.dumps(manifest_sig, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return shard_manifest


def build_offline_smoke_snapshots(
    *,
    site_lat_deg: float = 37.6188,
    site_lon_deg: float = -122.3750,
    site_alt_ellip_m: float = -28.0,
) -> tuple[list[list[AdsbFix]], list[float]]:
    """Build deterministic 2-aircraft turning-traffic offline scenario (with synthetic PII to verify stripping)."""
    # 1 deg lat ~= 111,132 m; place aircraft ~1.2 km North, ~520 m above site (~23.4 deg elevation > +5 deg)
    # Plus a low-elevation ground-skimming decoy at ~1.5 deg elevation that MUST be rejected by >= +5 deg mask!
    snapshots: list[list[AdsbFix]] = [
        [
            AdsbFix(
                track_key="A12345",
                t_rx_s=0.0,
                lat_deg=site_lat_deg + 0.0108,
                lon_deg=site_lon_deg - 0.0018,
                baro_alt_ft=1850.0,
                geom_alt_ft=1720.0,
                gva=2,
                ground_speed_kt=135.0,
                track_deg=85.0,
                vertical_rate_fpm=120.0,
                aerodynamic_class="NARROWBODY_JET",
                raw_pii_metadata={
                    "icao24": "A12345",
                    "callsign": "UAL999",
                    "tail_number": "N12345",
                    "raw_adsb_json": '{"hex":"A12345","flight":"UAL999"}',
                },
            ),
            AdsbFix(
                track_key="B54321",
                t_rx_s=0.0,
                lat_deg=site_lat_deg + 0.0120,
                lon_deg=site_lon_deg + 0.0005,
                baro_alt_ft=90.0,
                geom_alt_ft=None,
                gva=0,
                ground_speed_kt=95.0,
                track_deg=260.0,
                vertical_rate_fpm=0.0,
                aerodynamic_class="LIGHT_SINGLE",
                raw_pii_metadata={
                    "icao24": "B54321",
                    "callsign": "N54321",
                    "tail_number": "N54321",
                    "raw_adsb_json": '{"hex":"B54321","flight":"N54321"}',
                },
            ),
        ],
        [
            AdsbFix(
                track_key="A12345",
                t_rx_s=3.5,
                lat_deg=site_lat_deg + 0.0110,
                lon_deg=site_lon_deg + 0.0009,
                baro_alt_ft=1860.0,
                geom_alt_ft=1730.0,
                gva=2,
                ground_speed_kt=135.0,
                track_deg=96.0,
                vertical_rate_fpm=100.0,
                aerodynamic_class="NARROWBODY_JET",
                raw_pii_metadata={
                    "icao24": "A12345",
                    "callsign": "UAL999",
                    "tail_number": "N12345",
                    "raw_adsb_json": '{"hex":"A12345","flight":"UAL999"}',
                },
            ),
            AdsbFix(
                track_key="B54321",
                t_rx_s=3.5,
                lat_deg=site_lat_deg + 0.0119,
                lon_deg=site_lon_deg - 0.0012,
                baro_alt_ft=90.0,
                geom_alt_ft=None,
                gva=0,
                ground_speed_kt=95.0,
                track_deg=265.0,
                vertical_rate_fpm=0.0,
                aerodynamic_class="LIGHT_SINGLE",
                raw_pii_metadata={
                    "icao24": "B54321",
                    "callsign": "N54321",
                    "tail_number": "N54321",
                    "raw_adsb_json": '{"hex":"B54321","flight":"N54321"}',
                },
            ),
        ],
    ]
    frame_times_s = [0.5, 1.5, 2.5, 3.0]
    return snapshots, frame_times_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenXVS V3.0 Tier-0A ($0) Day-1 Real-Sky Window Capture, Video Ingest & Croissant 1.1 CLI (DR-15/DR-19)"
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--smoke-offline", action="store_true", help="Run deterministic offline smoke shard export")
    mode.add_argument("--poll-adsb-lol", action="store_true", help="Live polite polling via api.adsb.lol (>= 3.5 s)")
    mode.add_argument("--poll-flyitaly", action="store_true", help="Live polite polling via FlyItalyADSB (>= 2.5 s)")
    mode.add_argument(
        "--adsb-jsonl",
        type=Path,
        default=None,
        help="Offline recorded ADS-B snapshots/fixes JSONL (stripped before permissive export)",
    )

    # Real Tier-0A smartphone video ingest, UTC alignment, calibration & XVS-D2 zero-shot eval
    parser.add_argument("--video-path", type=Path, default=None, help="Smartphone .mp4/.mov video path to ingest")
    parser.add_argument(
        "--video-start-utc",
        type=str,
        default=None,
        help="Exact UTC ISO-8601 start timestamp of frame 0 (e.g. 2026-11-22T18:00:00Z; required with --video-path)",
    )
    parser.add_argument(
        "--clock-offset-ms",
        type=float,
        default=0.0,
        help="Measured camera-HAL-to-UTC clock offset in milliseconds (added to start_utc + pts)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Target frame sampling rate (Hz) when decoding --video-path (default: 2.0)",
    )
    parser.add_argument(
        "--camera-calib",
        type=Path,
        default=None,
        help="OpenCV checkerboard camera_calib.json (fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height)",
    )
    parser.add_argument(
        "--eval-xvs-d2",
        action="store_true",
        help="Run pinned XVS-D2 FP16 ONNX zero-shot detection eval over exported sky-masked frames",
    )
    parser.add_argument(
        "--xvs-d2-model",
        type=Path,
        default=DEFAULT_XVS_D2_ONNX,
        help="Path to pinned XVS-D2 ONNX model (default: apps/livedeck/public/models/xvs_d2_web_fp16.onnx)",
    )
    parser.add_argument("--xvs-d2-score-thr", type=float, default=0.25, help="XVS-D2 heatmap score threshold")
    parser.add_argument(
        "--frame-format",
        choices=("png", "jpg"),
        default="png",
        help="Encoded image format inside sky-masked frames_shard.tar (default: lossless png)",
    )
    parser.add_argument(
        "--visual-snap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dual-polarity top-hat/bottom-hat contrast-centroid visual snap (SNR >= 4.5, 3-sigma gate)",
    )
    parser.add_argument("--snap-sigma-px", type=float, default=15.0, help="1-sigma visual snap search radius (px)")
    parser.add_argument("--snap-min-snr", type=float, default=4.5, help="Minimum local contrast SNR for visual snap")
    parser.add_argument(
        "--ed25519-seed-hex",
        type=str,
        default=None,
        help="Optional 64-hex-char 32-byte Ed25519 private seed (ephemeral CSPRNG seed used if omitted)",
    )

    parser.add_argument("--lat", type=float, default=37.6188, help="Camera site WGS-84 latitude (deg)")
    parser.add_argument("--lon", type=float, default=-122.3750, help="Camera site WGS-84 longitude (deg)")
    parser.add_argument("--alt-m", type=float, default=-28.0, help="Camera site WGS-84 ellipsoidal height (m)")
    parser.add_argument("--geoid-n-m", type=float, default=-32.5, help="Local GEOID18 undulation N (m)")
    parser.add_argument("--radius-nm", type=float, default=10.0, help="ADS-B search radius (NM)")
    parser.add_argument("--station-icao", type=str, default="KSFO", help="Nearest METAR ICAO station")
    parser.add_argument("--provider", type=str, default="adsb.lol", help="ADS-B provider cue label for --adsb-jsonl")
    parser.add_argument("--qnh-hpa", type=float, default=ISA_P0_HPA, help="Offline METAR QNH pressure (hPa)")
    parser.add_argument("--temp-c", type=float, default=15.0, help="Offline ambient temperature (C)")
    parser.add_argument("--station-elev-m", type=float, default=0.0, help="Offline METAR station elevation (m)")
    parser.add_argument("--live-metar", action="store_true", help="Fetch live AviationWeather.gov METAR in --adsb-jsonl mode")
    parser.add_argument("--poll-interval-s", type=float, default=3.5, help="ADS-B poll interval in seconds")
    parser.add_argument("--polls", type=int, default=3, help="Number of live ADS-B polls")
    parser.add_argument("--azimuth-deg", type=float, default=0.0, help="Camera look azimuth from North (deg)")
    parser.add_argument("--pitch-up-deg", type=float, default=22.0, help="Camera pitch-up elevation (deg)")
    parser.add_argument("--elevation-mask-deg", type=float, default=5.0, help="Sky-only privacy mask (>= 5.0 deg)")
    parser.add_argument("--license", type=str, default="CDLA-Permissive-2.0", help="Release SPDX license")
    parser.add_argument("--flyitaly-api-key", type=str, default=None, help="Optional FlyItalyADSB API key")
    parser.add_argument(
        "--strip-upstream-adsb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip all raw upstream ADS-B JSON & PII (mandatory for ODbL/CC-BY-SA providers)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "realsky_smoke",
        help="Output directory for labels.jsonl, croissant.json, shard_manifest.json, and manifest_sig.json",
    )
    args = parser.parse_args(argv)

    has_adsb_source = bool(args.smoke_offline or args.poll_adsb_lol or args.poll_flyitaly or args.adsb_jsonl is not None)
    if not has_adsb_source:
        parser.error("Must specify one ADS-B source: --smoke-offline, --poll-adsb-lol, --poll-flyitaly, or --adsb-jsonl")
    if args.video_path is not None and not args.video_start_utc:
        parser.error("--video-start-utc <ISO-8601> is required with --video-path for exact UTC alignment")
    if args.eval_xvs_d2 and args.video_path is None:
        parser.error("--eval-xvs-d2 requires --video-path with sky-masked frames")

    r_ned_to_cam = build_sky_tilted_r_ned_to_cam(azimuth_deg=args.azimuth_deg, pitch_up_deg=args.pitch_up_deg)

    video_frames_bgr: list[np.ndarray] | None = None
    video_frame_times_s: list[float] | None = None
    video_meta: dict[str, Any] | None = None
    if args.video_path is not None:
        video_frames_bgr, video_frame_times_s, video_meta = ingest_smartphone_video_frames(
            args.video_path,
            video_start_utc=str(args.video_start_utc),
            clock_offset_ms=args.clock_offset_ms,
            sample_fps=args.sample_fps,
        )

    if args.camera_calib is not None:
        intrinsics = load_camera_calibration(args.camera_calib)
    elif video_meta is not None:
        vw = int(video_meta["width_px"])
        vh = int(video_meta["height_px"])
        scale = float(vw) / 3840.0
        intrinsics = CameraIntrinsics(
            fx=2560.0 * scale,
            fy=2560.0 * scale,
            cx=0.5 * float(vw),
            cy=0.5 * float(vh),
            width_px=vw,
            height_px=vh,
        )
    else:
        intrinsics = CameraIntrinsics(fx=2560.0, fy=2560.0, cx=1920.0, cy=1080.0, width_px=3840, height_px=2160)

    if args.smoke_offline:
        provider = "adsb.lol"
        metar_obs = MetarObservation(
            station_icao=args.station_icao,
            qnh_hpa=1016.2,
            t_amb_k=287.15,
            station_elev_m=4.0,
            fetched_mono_s=0.0,
            source="OFFLINE_SMOKE_METAR",
        )
        snapshots, smoke_frame_times_s = build_offline_smoke_snapshots(
            site_lat_deg=args.lat,
            site_lon_deg=args.lon,
            site_alt_ellip_m=args.alt_m,
        )
        frame_times_s = video_frame_times_s if video_frame_times_s is not None else smoke_frame_times_s
    elif args.adsb_jsonl is not None:
        provider = str(args.provider)
        if args.live_metar:
            metar_obs = MetarQnhCache().get_observation(args.station_icao)
        else:
            metar_obs = MetarObservation(
                station_icao=args.station_icao.strip().upper(),
                qnh_hpa=float(args.qnh_hpa),
                t_amb_k=float(args.temp_c) + 273.15,
                station_elev_m=float(args.station_elev_m),
                fetched_mono_s=0.0,
                source="OFFLINE_METAR_PARAMS",
            )
        snapshots = load_recorded_adsb_jsonl(args.adsb_jsonl)
        if video_frame_times_s is not None:
            frame_times_s = video_frame_times_s
        else:
            all_ts = sorted({fix.t_rx_s for snap in snapshots for fix in snap})
            frame_times_s = [all_ts[0] + 0.5 * (all_ts[-1] - all_ts[0])] if len(all_ts) >= 2 else []
    else:
        provider = "adsb.lol" if args.poll_adsb_lol else "flyitalyadsb"
        metar_cache = MetarQnhCache()
        metar_obs = metar_cache.get_observation(args.station_icao)
        snapshots = poll_live_adsb_snapshots(
            provider=provider,
            lat_deg=args.lat,
            lon_deg=args.lon,
            radius_nm=args.radius_nm,
            poll_interval_s=args.poll_interval_s,
            num_polls=args.polls,
            strip_upstream_json=args.strip_upstream_adsb,
            api_key=args.flyitaly_api_key,
        )
        if video_frame_times_s is not None:
            frame_times_s = video_frame_times_s
        else:
            all_ts = sorted({fix.t_rx_s for snap in snapshots for fix in snap})
            frame_times_s = (
                [all_ts[0] + 0.5 * (all_ts[-1] - all_ts[0])] if len(all_ts) >= 2 else []
            )

    private_seed: bytes | None = None
    if args.ed25519_seed_hex is not None:
        private_seed = bytes.fromhex(args.ed25519_seed_hex.strip())

    manifest = run_realsky_window_pipeline(
        snapshots=snapshots,
        frame_times_s=frame_times_s,
        site_lat_deg=args.lat,
        site_lon_deg=args.lon,
        site_alt_ellip_m=args.alt_m,
        geoid_undulation_m=args.geoid_n_m,
        metar_obs=metar_obs,
        r_ned_to_cam=r_ned_to_cam,
        intrinsics=intrinsics,
        provider=provider,
        poll_interval_s=args.poll_interval_s,
        elevation_mask_deg=args.elevation_mask_deg,
        strip_upstream_adsb=args.strip_upstream_adsb,
        license_spdx=args.license,
        out_dir=args.out_dir,
        video_frames_bgr=video_frames_bgr,
        frame_format=args.frame_format,
        enable_visual_snap=args.visual_snap,
        snap_sigma_px=args.snap_sigma_px,
        snap_min_snr=args.snap_min_snr,
        ed25519_private_seed=private_seed,
        eval_xvs_d2=args.eval_xvs_d2,
        xvs_d2_model_path=args.xvs_d2_model,
        xvs_d2_score_thr=args.xvs_d2_score_thr,
    )
    extra = ""
    if "exported_frame_count" in manifest:
        extra += (
            f" frames={manifest['exported_frame_count']}"
            f" snapped={manifest['visual_snap_locked']}"
            f" fallback={manifest['visual_snap_fallback']}"
        )
    if "xvs_d2_eval" in manifest:
        ev = manifest["xvs_d2_eval"]
        extra += f" d2_f1={ev['f1']:.3f} d2_ms={ev['mean_latency_ms']:.1f}"
    print(
        f"realsky_window_capture: ok records={manifest['record_count']} "
        f"rejected_below_mask={manifest['rejected_below_sky_mask']}{extra} "
        f"manifest_sha256={manifest['manifest_sha256']} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
