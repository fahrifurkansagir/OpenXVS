"""DR-15 Tier-0A ($0 Window Capture) & Tier-0B Exact Geodetic ADS-B-to-Pixel Auto-Labeler.

Implements the zero-handwaving 5-step pipeline from `DR-15`:
1. Vertical datum reconciliation (`reconcile_ellipsoidal_altitude_m`): geometric altitude
   when `GVA >= 1`, or exact ISA QNH + ICAO cold-temperature + GEOID18 ellipsoidal height.
2. Exact WGS-84 geodetic <-> ECEF -> local North-East-Down (NED) transforms (`CONVENTIONS.md`
   v1.0 — strictly NED / BODY FRD, zero internal ENU).
3. Dual kinematic propagation:
   - Tier-0B live causal turn propagation (`propagate_causal_turn_ned`, `dt <= 0.5 s`).
   - Tier-0A offline non-causal Cubic Hermite spline (`interpolate_cubic_hermite_ned`),
     cutting turning-aircraft error at 1 km / 3 s poll from `~28.8 px` down to `< 1.0 px`.
4. Camera projection with full Brown-Conrady `(k1, k2, p1, p2, k3)` distortion, behind-camera
   `Z_c <= 0` rejection, and aspect-aware aerodynamic bounding-box prior.
5. Fail-closed legal, license, privacy, and OFAC sanctions guards:
   - Blocks `airplanes.live` (HTTP 403 closed Aug 2026), blocks non-commercial `opensky` /
     `adsb.fi` on commercial releases, enforces upstream JSON stripping on `adsb.lol`
     (ODbL 1.0) & `flyitalyadsb` (CC BY-SA 4.0), and allows local 1090ES RF natively.
   - Hard-blocks OFAC SDN-sanctioned Yantai IRay / InfiRay LWIR cores (`tiny1-c`, `p2 pro`,
     `p2s`, `xinfrared`).
   - Enforces `>= +5.0 deg` sky-only elevation mask and strips `icao24`, `callsign`,
     `tail_number`, and raw ADS-B JSON for clean `CC-BY-4.0` `OpenXVS-RealSky-v1` labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

# Exact WGS-84 ellipsoid constants
WGS84_A: float = 6378137.0
WGS84_F: float = 1.0 / 298.257223563
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F * WGS84_F

# Exact ISA & aviation SI conversion constants (`DR-15` Step 1)
FT_TO_M: float = 0.3048
KT_TO_MPS: float = 0.514444
FPM_TO_MPS: float = 0.00508
ISA_P0_HPA: float = 1013.25
ISA_T0_K: float = 288.15
ISA_LAPSE_K_PER_M: float = 0.0065
ISA_R_AIR: float = 287.05287
ISA_G0_MPS2: float = 9.80665

MIN_SKY_ELEVATION_MASK_DEG: float = 5.0

ALLOWED_AERODYNAMIC_CLASSES: frozenset[str] = frozenset(
    {
        "LIGHT_SINGLE",
        "TWIN_TURBOPROP",
        "NARROWBODY_JET",
        "WIDEBODY_JET",
        "ROTORCRAFT",
        "GLIDER_ULTRALIGHT",
        "CIVIL_UNKNOWN",
    }
)

_FORBIDDEN_PII_FIELDS: frozenset[str] = frozenset(
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

_OFAC_SANCTIONED_LWIR_TOKENS: tuple[str, ...] = (
    "infiray",
    "yantai iray",
    "yantai_iray",
    "tiny1-c",
    "tiny1_c",
    "tiny1c",
    "p2 pro",
    "p2_pro",
    "p2pro",
    "p2s",
    "xinfrared",
)


class RealSkyComplianceError(PermissionError):
    """Raised when a DR-15 legal, license, privacy, or OFAC sanctions guard trips."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Calibrated pinhole + Brown-Conrady `(k1, k2, p1, p2, k3)` camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    width_px: int = 3840
    height_px: int = 2160


@dataclass(frozen=True)
class AirframeDimensionsM:
    """Physical civil airframe dimensions (meters) for bounding-box span prior."""

    wingspan_m: float
    length_m: float
    tail_height_m: float


@dataclass(frozen=True)
class PixelProjection:
    """Projected 2D image coordinates and 3D camera/range geometry."""

    u_px: float
    v_px: float
    x_cam_m: float
    y_cam_m: float
    z_cam_m: float
    slant_range_m: float
    elevation_deg: float
    in_image_bounds: bool


def reconcile_ellipsoidal_altitude_m(
    *,
    baro_alt_ft: float | None = None,
    geom_alt_ft: float | None = None,
    gva: int = 0,
    geom_is_orthometric_msl: bool = False,
    qnh_hpa: float = ISA_P0_HPA,
    t_amb_k: float = ISA_T0_K,
    station_elev_m: float = 0.0,
    geoid_undulation_m: float = 0.0,
) -> float:
    """Reconcile ADS-B altitude to WGS-84 ellipsoidal height (`DR-15` Step 1).

    Prefers `geom_alt_ft` when present and `gva >= 1` (`0.3048 * geom_alt_ft`, adding
    `geoid_undulation_m` when flagged orthometric MSL). Otherwise applies exact ISA QNH
    pressure correction + ICAO cold-temperature scaling above `station_elev_m` +
    `geoid_undulation_m`.
    """
    if geom_alt_ft is not None and int(gva) >= 1:
        h_geom = FT_TO_M * float(geom_alt_ft)
        if not math.isfinite(h_geom):
            raise ValueError(f"Non-finite geom_alt_ft: {geom_alt_ft!r}")
        return h_geom + (float(geoid_undulation_m) if geom_is_orthometric_msl else 0.0)

    if baro_alt_ft is None:
        raise ValueError("Either geom_alt_ft (with gva >= 1) or baro_alt_ft must be provided")

    h_p = FT_TO_M * float(baro_alt_ft)
    p_qnh = float(qnh_hpa)
    t_amb = float(t_amb_k)
    h_sta = float(station_elev_m)
    n_geoid = float(geoid_undulation_m)
    for name, val in (
        ("baro_alt_ft", h_p),
        ("qnh_hpa", p_qnh),
        ("t_amb_k", t_amb),
        ("station_elev_m", h_sta),
        ("geoid_undulation_m", n_geoid),
    ):
        if not math.isfinite(val):
            raise ValueError(f"Non-finite altitude parameter '{name}': {val!r}")
    if p_qnh <= 0.0 or t_amb <= 0.0:
        raise ValueError(f"qnh_hpa ({p_qnh}) and t_amb_k ({t_amb}) must be strictly positive")

    exponent = (ISA_R_AIR * ISA_LAPSE_K_PER_M) / ISA_G0_MPS2
    h_qnh = h_p + (ISA_T0_K / ISA_LAPSE_K_PER_M) * ((p_qnh / ISA_P0_HPA) ** exponent - 1.0)
    t_isa_station = ISA_T0_K - ISA_LAPSE_K_PER_M * h_sta
    h_msl = h_sta + (h_qnh - h_sta) * (t_amb / t_isa_station)
    return h_msl + n_geoid


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_ellip_m: float) -> np.ndarray:
    """Convert WGS-84 `(lat_deg, lon_deg, alt_ellip_m)` to ECEF `[X, Y, Z]` (m)."""
    phi = math.radians(float(lat_deg))
    lam = math.radians(float(lon_deg))
    h = float(alt_ellip_m)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)
    r_n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    x = (r_n + h) * cos_phi * cos_lam
    y = (r_n + h) * cos_phi * sin_lam
    z = (r_n * (1.0 - WGS84_E2) + h) * sin_phi
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(ecef_m: Sequence[float]) -> tuple[float, float, float]:
    """Convert WGS-84 ECEF `[X, Y, Z]` (m) back to `(lat_deg, lon_deg, alt_ellip_m)`."""
    x, y, z = float(ecef_m[0]), float(ecef_m[1]), float(ecef_m[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:
        lat = math.copysign(0.5 * math.pi, z)
        b_semi = WGS84_A * (1.0 - WGS84_F)
        alt = abs(z) - b_semi
        return math.degrees(lat), math.degrees(lon), alt

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        r_n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - r_n
        lat_next = math.atan2(z, p * (1.0 - WGS84_E2 * r_n / (r_n + alt)))
        if abs(lat_next - lat) < 1e-13:
            lat = lat_next
            break
        lat = lat_next
    sin_lat = math.sin(lat)
    r_n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - r_n
    return math.degrees(lat), math.degrees(lon), alt


def _r_ecef_to_ned(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    phi = math.radians(float(lat0_deg))
    lam = math.radians(float(lon0_deg))
    s_p, c_p = math.sin(phi), math.cos(phi)
    s_l, c_l = math.sin(lam), math.cos(lam)
    return np.array(
        [
            [-s_p * c_l, -s_p * s_l, c_p],
            [-s_l, c_l, 0.0],
            [-c_p * c_l, -c_p * s_l, -s_p],
        ],
        dtype=np.float64,
    )


def geodetic_to_local_ned(
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_ellip_m: float,
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_ellip_m: float,
) -> np.ndarray:
    """Convert target WGS-84 geodetic position into pod local NED `[N, E, D]` (m)."""
    p_t = geodetic_to_ecef(target_lat_deg, target_lon_deg, target_alt_ellip_m)
    p_0 = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_ellip_m)
    r_mat = _r_ecef_to_ned(origin_lat_deg, origin_lon_deg)
    return r_mat @ (p_t - p_0)


def local_ned_to_ecef(
    p_ned_m: Sequence[float],
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_ellip_m: float,
) -> np.ndarray:
    """Inverse transform from local NED `[N, E, D]` (m) back to WGS-84 ECEF (m)."""
    p_0 = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_ellip_m)
    r_mat = _r_ecef_to_ned(origin_lat_deg, origin_lon_deg)
    vec = np.asarray(p_ned_m, dtype=np.float64)
    return p_0 + (r_mat.T @ vec)


def ned_velocity_from_adsb(
    *,
    ground_speed_kt: float,
    track_deg: float,
    vertical_rate_fpm: float = 0.0,
) -> np.ndarray:
    """Convert ADS-B `(ground_speed_kt, track_deg, vertical_rate_fpm)` to NED velocity (m/s)."""
    v_g = KT_TO_MPS * float(ground_speed_kt)
    chi = math.radians(float(track_deg))
    h_dot = FPM_TO_MPS * float(vertical_rate_fpm)
    return np.array([v_g * math.cos(chi), v_g * math.sin(chi), -h_dot], dtype=np.float64)


def propagate_causal_turn_ned(
    p_ned_k: Sequence[float],
    *,
    ground_speed_mps: float,
    track_rad: float,
    vertical_rate_mps: float = 0.0,
    turn_rate_rad_s: float = 0.0,
    dt_s: float = 0.0,
) -> np.ndarray:
    """Tier-0B live causal propagation (`DR-15` Step 3): `p_k + v*dt + 0.5 * a_turn * dt^2`."""
    p_k = np.asarray(p_ned_k, dtype=np.float64)
    v_g = float(ground_speed_mps)
    chi = float(track_rad)
    chi_dot = float(turn_rate_rad_s)
    dt = float(dt_s)
    v_ned = np.array([v_g * math.cos(chi), v_g * math.sin(chi), -float(vertical_rate_mps)], dtype=np.float64)
    a_turn = np.array(
        [-v_g * chi_dot * math.sin(chi), v_g * chi_dot * math.cos(chi), 0.0],
        dtype=np.float64,
    )
    return p_k + v_ned * dt + 0.5 * a_turn * (dt * dt)


def interpolate_cubic_hermite_ned(
    p_k: Sequence[float],
    v_k: Sequence[float],
    p_k1: Sequence[float],
    v_k1: Sequence[float],
    *,
    t_mid_s: float,
    t_k_s: float,
    t_k1_s: float,
) -> np.ndarray:
    """Tier-0A offline non-causal Cubic Hermite spline between bracketing ADS-B fixes (`DR-15` Step 3)."""
    dt_span = float(t_k1_s) - float(t_k_s)
    if dt_span <= 0.0:
        raise ValueError(f"Bracketing interval t_k1_s - t_k_s must be > 0, got {dt_span}")
    s = (float(t_mid_s) - float(t_k_s)) / dt_span
    if s < -1e-9 or s > 1.0 + 1e-9:
        raise ValueError(f"Normalized Hermite time s={s:.6f} outside [0, 1]")
    s = min(1.0, max(0.0, s))

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2

    pk_arr = np.asarray(p_k, dtype=np.float64)
    vk_arr = np.asarray(v_k, dtype=np.float64)
    pk1_arr = np.asarray(p_k1, dtype=np.float64)
    vk1_arr = np.asarray(v_k1, dtype=np.float64)
    return h00 * pk_arr + h10 * dt_span * vk_arr + h01 * pk1_arr + h11 * dt_span * vk1_arr


def project_ned_to_pixel(
    p_ned_m: Sequence[float],
    r_ned_to_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> PixelProjection | None:
    """Project local NED target vector through `R_ned_to_cam` + Brown-Conrady distortion (`DR-15` Step 4).

    Returns `None` if the target lies on or behind the camera image plane (`Z_c <= 0`).
    """
    p_ned = np.asarray(p_ned_m, dtype=np.float64)
    rot = np.asarray(r_ned_to_cam, dtype=np.float64)
    if p_ned.shape != (3,) or rot.shape != (3, 3):
        raise ValueError(f"Expected p_ned (3,) and r_ned_to_cam (3,3), got {p_ned.shape} and {rot.shape}")

    p_cam = rot @ p_ned
    xc, yc, zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if zc <= 0.0:
        return None

    x = xc / zc
    y = yc / zc
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6
    x_d = x * radial + 2.0 * intrinsics.p1 * x * y + intrinsics.p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + intrinsics.p1 * (r2 + 2.0 * y * y) + 2.0 * intrinsics.p2 * x * y

    u_px = intrinsics.fx * x_d + intrinsics.cx
    v_px = intrinsics.fy * y_d + intrinsics.cy

    horiz_m = math.hypot(float(p_ned[0]), float(p_ned[1]))
    slant_range_m = math.hypot(horiz_m, float(p_ned[2]))
    # In NED, Down is negative above horizon -> elevation = atan2(-D, horiz)
    elevation_deg = math.degrees(math.atan2(-float(p_ned[2]), horiz_m))
    in_bounds = (
        0.0 <= u_px < float(intrinsics.width_px) and 0.0 <= v_px < float(intrinsics.height_px)
    )

    return PixelProjection(
        u_px=u_px,
        v_px=v_px,
        x_cam_m=xc,
        y_cam_m=yc,
        z_cam_m=zc,
        slant_range_m=slant_range_m,
        elevation_deg=elevation_deg,
        in_image_bounds=in_bounds,
    )


def estimate_airframe_box_prior_px(
    *,
    slant_range_m: float,
    intrinsics: CameraIntrinsics,
    dimensions: AirframeDimensionsM,
    aspect_angle_rad: float = 0.0,
    look_elevation_rad: float = 0.0,
    bank_angle_rad: float = 0.0,
) -> tuple[float, float]:
    """Aspect-aware physical bounding-box prior `(w_px, h_px)` (`DR-15` Step 4)."""
    r_m = float(slant_range_m)
    if r_m <= 0.0:
        raise ValueError(f"slant_range_m must be > 0, got {r_m}")
    w_eff = math.hypot(
        dimensions.wingspan_m * math.cos(float(aspect_angle_rad)),
        dimensions.length_m * math.sin(float(aspect_angle_rad)),
    )
    h_eff = dimensions.tail_height_m * math.cos(float(look_elevation_rad)) + (
        dimensions.wingspan_m * abs(math.sin(float(bank_angle_rad)))
    )
    w_px = (float(intrinsics.fx) / r_m) * w_eff
    h_px = (float(intrinsics.fy) / r_m) * h_eff
    return max(1.0, w_px), max(1.0, h_px)


def assert_adsb_provider_allowed(
    provider: str,
    *,
    commercial_release: bool = True,
    strip_upstream_json: bool = False,
) -> str:
    """Enforce DR-15 2026 Public ADS-B API license & availability rules fail-closed."""
    key = provider.strip().lower().replace("-", "_").replace(".", "_")
    if key in ("airplanes_live", "airplaneslive"):
        raise RealSkyComplianceError(
            "Provider 'airplanes.live' is blocked (HTTP 403 closed to non-feeders in Aug 2026; DR-15)."
        )
    if key in ("opensky", "opensky_network"):
        if commercial_release:
            raise RealSkyComplianceError(
                "Provider 'OpenSky Network' is non-profit research only and barred from permissive/commercial dataset releases."
            )
        return "ACADEMIC_NONCOMMERCIAL_ONLY"
    if key in ("adsb_fi", "adsbfi"):
        if commercial_release:
            raise RealSkyComplianceError(
                "Provider 'adsb.fi' is personal non-commercial only and barred from permissive/commercial dataset releases."
            )
        return "PERSONAL_NONCOMMERCIAL_ONLY"
    if key in ("local_rf_1090es", "local_rf_978uat", "local_rf"):
        return "LOCAL_RF_NATIVE_CC_BY_4"
    if key in ("adsb_lol", "adsblol", "flyitalyadsb", "fly_italy_adsb"):
        if not strip_upstream_json:
            raise RealSkyComplianceError(
                f"Provider '{provider}' carries Share-Alike terms (ODbL 1.0 / CC BY-SA 4.0); "
                "strip_upstream_json=True is mandatory to keep OpenXVS-RealSky-v1 clean CC-BY-4.0."
            )
        return "TRANSIENT_CUE_STRIPPED_OK"
    raise RealSkyComplianceError(f"Unrecognized or unvetted ADS-B provider: {provider!r}")


def assert_thermal_sensor_sanctions_clean(vendor_or_model: str) -> str:
    """Hard-block OFAC SDN-sanctioned Yantai IRay / InfiRay LWIR cores (`DR-15` Thermal Audit)."""
    norm = vendor_or_model.strip().lower()
    if "shanghai iray" in norm and "infiray" not in norm and "yantai" not in norm:
        return "SHANGHAI_IRAY_MEDICAL_NON_AFFILIATED"
    for token in _OFAC_SANCTIONED_LWIR_TOKENS:
        if token in norm:
            raise RealSkyComplianceError(
                f"Thermal sensor '{vendor_or_model}' matches OFAC SDN-sanctioned Yantai IRay / "
                f"InfiRay token '{token}' (E.O. 14024, May 1 2024; hard-blocked by DR-15)."
            )
    return "SANCTIONS_CLEAN"


def sanitize_realsky_label_record(
    raw_record: Mapping[str, Any],
    *,
    provider: str,
    ephemeral_track_id: str,
    aerodynamic_class: str,
    elevation_mask_deg: float,
    commercial_release: bool = True,
    strip_upstream_json: bool = True,
) -> dict[str, Any]:
    """Emit a clean `CC-BY-4.0` `OpenXVS-RealSky-v1` label stripping all PII & upstream ADS-B JSON.

    Enforces `elevation_mask_deg >= 5.0` (sky-only privacy lock — zero homes/faces/plates)
    and strips `icao24`, `callsign`, `tail_number`, and raw ADS-B JSON (LADD/PIA/47 U.S.C. §605
    and ODbL/CC-BY-SA hygiene).
    """
    provider_verdict = assert_adsb_provider_allowed(
        provider,
        commercial_release=commercial_release,
        strip_upstream_json=strip_upstream_json,
    )
    elev_mask = float(elevation_mask_deg)
    if not math.isfinite(elev_mask) or elev_mask < MIN_SKY_ELEVATION_MASK_DEG:
        raise RealSkyComplianceError(
            f"Sky elevation mask {elev_mask:.2f} deg is below mandatory >= {MIN_SKY_ELEVATION_MASK_DEG:.1f} deg "
            "(DR-15 zero-ground-PII privacy lock)."
        )

    cls_norm = str(aerodynamic_class).strip().upper()
    if cls_norm not in ALLOWED_AERODYNAMIC_CLASSES:
        raise ValueError(f"Unsupported aerodynamic_class {aerodynamic_class!r}")

    track_id = str(ephemeral_track_id).strip()
    if not track_id or len(track_id) > 64:
        raise ValueError(f"Invalid ephemeral_track_id: {ephemeral_track_id!r}")

    for req in ("frame_idx", "u_px", "v_px", "w_px", "h_px"):
        if req not in raw_record:
            raise KeyError(f"Missing required visual label field '{req}' in raw_record")

    # Explicitly exclude every PII / LADD / PIA / upstream raw-JSON key
    stripped_keys = sorted(k for k in raw_record.keys() if str(k).lower() in _FORBIDDEN_PII_FIELDS)

    clean: dict[str, Any] = {
        "schema": "openxvs.realsky.label/1",
        "license": "CC-BY-4.0",
        "provider_verdict": provider_verdict,
        "ephemeral_track_id": track_id,
        "aerodynamic_class": cls_norm,
        "frame_idx": int(raw_record["frame_idx"]),
        "u_px": round(float(raw_record["u_px"]), 3),
        "v_px": round(float(raw_record["v_px"]), 3),
        "w_px": round(float(raw_record["w_px"]), 3),
        "h_px": round(float(raw_record["h_px"]), 3),
        "elevation_mask_deg": round(elev_mask, 3),
        "pii_stripped": True,
        "upstream_adsb_stripped": bool(strip_upstream_json),
        "stripped_field_count": len(stripped_keys),
    }
    if "slant_range_m" in raw_record and raw_record["slant_range_m"] is not None:
        clean["slant_range_m"] = round(float(raw_record["slant_range_m"]), 2)
    if "frame_utc_iso" in raw_record and raw_record["frame_utc_iso"] is not None:
        clean["frame_utc_iso"] = str(raw_record["frame_utc_iso"])
    if "frame_file" in raw_record and raw_record["frame_file"] is not None:
        clean["frame_file"] = str(raw_record["frame_file"])
    if "visual_snapped" in raw_record and raw_record["visual_snapped"] is not None:
        clean["visual_snapped"] = bool(raw_record["visual_snapped"])
    if "visual_snap_snr" in raw_record and raw_record["visual_snap_snr"] is not None:
        clean["visual_snap_snr"] = round(float(raw_record["visual_snap_snr"]), 3)
    if "visual_snap_du_px" in raw_record and raw_record["visual_snap_du_px"] is not None:
        clean["visual_snap_du_px"] = round(float(raw_record["visual_snap_du_px"]), 3)
    if "visual_snap_dv_px" in raw_record and raw_record["visual_snap_dv_px"] is not None:
        clean["visual_snap_dv_px"] = round(float(raw_record["visual_snap_dv_px"]), 3)
    if "visual_snap_polarity" in raw_record and raw_record["visual_snap_polarity"] is not None:
        pol = str(raw_record["visual_snap_polarity"]).strip().upper()
        if pol not in ("BRIGHT_TOPHAT", "DARK_BOTTOMHAT", "NONE"):
            raise ValueError(f"Unsupported visual_snap_polarity: {pol!r}")
        clean["visual_snap_polarity"] = pol
    return clean


@dataclass(frozen=True)
class ContrastSnapResult:
    """Result of bounded dual-polarity morphological contrast-centroid visual snap (`DR-15` Step 5)."""

    snapped: bool
    u_px: float
    v_px: float
    prior_u_px: float
    prior_v_px: float
    du_px: float
    dv_px: float
    snr: float
    polarity: str
    search_radius_px: float


def _parse_strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Camera calibration '{name}' must be a positive integer, got bool {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Camera calibration '{name}' must be > 0, got {value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
            raise ValueError(f"Camera calibration '{name}' must be a finite positive integer, got {value!r}")
        return int(value)
    raise ValueError(f"Camera calibration '{name}' must be a positive integer, got {value!r}")


def _parse_strict_finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Camera calibration '{name}' must be numeric, got {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"Camera calibration '{name}' must be finite, got {value!r}")
    if positive and val <= 0.0:
        raise ValueError(f"Camera calibration '{name}' must be > 0, got {val}")
    return val


def load_camera_calibration(path: str | Path) -> CameraIntrinsics:
    """Load and strictly validate OpenCV checkerboard camera calibration JSON (`DR-15`).

    Supports both flat OpenXVS/OpenCV keys (`fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height`)
    and nested OpenCV `camera_matrix` (3x3) + `dist_coeffs` (4- or 5-element vector).
    """
    calib_path = Path(path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Camera calibration JSON not found: {calib_path}")
    raw = json.loads(calib_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Camera calibration JSON must be an object, got {type(raw).__name__}")

    # Resolve image dimensions (accept width/height, width_px/height_px, or image_width/image_height)
    width_val: Any = None
    for key in ("width", "width_px", "image_width"):
        if key in raw and raw[key] is not None:
            width_val = raw[key]
            break
    height_val: Any = None
    for key in ("height", "height_px", "image_height"):
        if key in raw and raw[key] is not None:
            height_val = raw[key]
            break
    if width_val is None or height_val is None:
        raise ValueError("Camera calibration JSON must specify positive integer 'width' and 'height'")
    width_px = _parse_strict_positive_int(width_val, "width")
    height_px = _parse_strict_positive_int(height_val, "height")

    # Resolve intrinsics (flat keys take precedence; otherwise parse OpenCV 3x3 camera_matrix)
    if all(k in raw and raw[k] is not None for k in ("fx", "fy", "cx", "cy")):
        fx = _parse_strict_finite_float(raw["fx"], "fx", positive=True)
        fy = _parse_strict_finite_float(raw["fy"], "fy", positive=True)
        cx = _parse_strict_finite_float(raw["cx"], "cx", positive=False)
        cy = _parse_strict_finite_float(raw["cy"], "cy", positive=False)
    elif "camera_matrix" in raw and isinstance(raw["camera_matrix"], Sequence):
        mat = np.asarray(raw["camera_matrix"], dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"camera_matrix must have shape (3, 3), got {mat.shape}")
        fx = _parse_strict_finite_float(float(mat[0, 0]), "camera_matrix[0,0]", positive=True)
        fy = _parse_strict_finite_float(float(mat[1, 1]), "camera_matrix[1,1]", positive=True)
        cx = _parse_strict_finite_float(float(mat[0, 2]), "camera_matrix[0,2]", positive=False)
        cy = _parse_strict_finite_float(float(mat[1, 2]), "camera_matrix[1,2]", positive=False)
    else:
        raise ValueError("Camera calibration JSON requires ('fx', 'fy', 'cx', 'cy') or 3x3 'camera_matrix'")

    if not (0.0 <= cx <= float(width_px)) or not (0.0 <= cy <= float(height_px)):
        raise ValueError(
            f"Principal point (cx={cx}, cy={cy}) lies outside calibrated image bounds ({width_px}x{height_px})"
        )

    # Resolve Brown-Conrady distortion (k1, k2, p1, p2, k3)
    dist_vals: dict[str, float] = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
    has_flat_dist = any(k in raw and raw[k] is not None for k in dist_vals)
    if has_flat_dist:
        for k in dist_vals:
            if k in raw and raw[k] is not None:
                dist_vals[k] = _parse_strict_finite_float(raw[k], k, positive=False)
    elif "dist_coeffs" in raw and raw["dist_coeffs"] is not None:
        coeffs_raw = np.asarray(raw["dist_coeffs"], dtype=np.float64).ravel()
        if coeffs_raw.size not in (4, 5):
            raise ValueError(f"dist_coeffs must contain 4 or 5 elements (k1,k2,p1,p2[,k3]), got {coeffs_raw.size}")
        names = ("k1", "k2", "p1", "p2", "k3")
        for idx in range(coeffs_raw.size):
            dist_vals[names[idx]] = _parse_strict_finite_float(float(coeffs_raw[idx]), names[idx], positive=False)

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k1=dist_vals["k1"],
        k2=dist_vals["k2"],
        p1=dist_vals["p1"],
        p2=dist_vals["p2"],
        k3=dist_vals["k3"],
        width_px=width_px,
        height_px=height_px,
    )


def compute_sky_elevation_mask(
    calib: CameraIntrinsics,
    r_ned_cam: np.ndarray,
    *,
    min_elev_deg: float = MIN_SKY_ELEVATION_MASK_DEG,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute deterministic per-pixel NED sky-only boolean mask (`elevation_deg >= min_elev_deg`).

    Enforces the DR-15 zero-ground-PII privacy lock (`min_elev_deg >= +5.0 deg`) fail-closed.
    Inverts Brown-Conrady distortion `(k1, k2, p1, p2, k3)` per pixel and checks exact NED
    ray elevation `atan2(-ray_d, hypot(ray_n, ray_e)) >= min_elev_deg`. Any pixel whose
    undistortion fails to converge stays masked (`False`) fail-closed.
    """
    min_el = float(min_elev_deg)
    if not math.isfinite(min_el) or min_el < MIN_SKY_ELEVATION_MASK_DEG:
        raise RealSkyComplianceError(
            f"Sky elevation mask {min_el:.2f} deg violates mandatory >= +{MIN_SKY_ELEVATION_MASK_DEG:.1f} deg "
            "pre-encode privacy lock (DR-15)."
        )

    rot = np.asarray(r_ned_cam, dtype=np.float64)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        raise ValueError(f"r_ned_cam must be finite (3, 3) rotation matrix, got shape {rot.shape}")
    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3, dtype=np.float64), ord="fro"))
    if ortho_err > 1e-3:
        raise ValueError(f"r_ned_cam is not orthonormal (||R^T R - I||_F = {ortho_err:.4e})")

    if calib.fx <= 0.0 or calib.fy <= 0.0 or calib.width_px <= 0 or calib.height_px <= 0:
        raise ValueError("Invalid CameraIntrinsics: fx, fy, width_px, height_px must be strictly positive")

    h_px, w_px = (int(calib.height_px), int(calib.width_px)) if image_shape is None else (int(image_shape[0]), int(image_shape[1]))
    if h_px <= 0 or w_px <= 0:
        raise ValueError(f"Invalid image_shape ({h_px}, {w_px})")

    # Scale pinhole intrinsics linearly if frame was uniformly downsampled from calibrated resolution
    sx = float(w_px) / float(calib.width_px)
    sy = float(h_px) / float(calib.height_px)
    fx_s = float(calib.fx) * sx
    fy_s = float(calib.fy) * sy
    cx_s = float(calib.cx) * sx
    cy_s = float(calib.cy) * sy

    u_grid = np.arange(w_px, dtype=np.float32)[np.newaxis, :]
    v_grid = np.arange(h_px, dtype=np.float32)[:, np.newaxis]
    x_d = np.broadcast_to((u_grid - np.float32(cx_s)) / np.float32(fx_s), (h_px, w_px))
    y_d = np.broadcast_to((v_grid - np.float32(cy_s)) / np.float32(fy_s), (h_px, w_px))

    has_dist = any(abs(c) > 0.0 for c in (calib.k1, calib.k2, calib.p1, calib.p2, calib.k3))
    if not has_dist:
        x = x_d
        y = y_d
        converged = np.ones((h_px, w_px), dtype=bool)
    else:
        k1 = np.float32(calib.k1)
        k2 = np.float32(calib.k2)
        p1 = np.float32(calib.p1)
        p2 = np.float32(calib.p2)
        k3 = np.float32(calib.k3)
        x = x_d.copy()
        y = y_d.copy()
        for _ in range(8):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = np.float32(1.0) + k1 * r2 + k2 * r4 + k3 * r6
            dx_t = np.float32(2.0) * p1 * x * y + p2 * (r2 + np.float32(2.0) * x * x)
            dy_t = p1 * (r2 + np.float32(2.0) * y * y) + np.float32(2.0) * p2 * x * y
            radial_safe = np.where(np.abs(radial) > np.float32(1e-4), radial, np.float32(1.0))
            x = (x_d - dx_t) / radial_safe
            y = (y_d - dy_t) / radial_safe
        # Forward reprojection check: fail closed on non-monotonic foldover corners
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = np.float32(1.0) + k1 * r2 + k2 * r4 + k3 * r6
        x_re = x * radial + np.float32(2.0) * p1 * x * y + p2 * (r2 + np.float32(2.0) * x * x)
        y_re = y * radial + p1 * (r2 + np.float32(2.0) * y * y) + np.float32(2.0) * p2 * x * y
        err_px = np.hypot((x_re - x_d) * np.float32(fx_s), (y_re - y_d) * np.float32(fy_s))
        converged = np.isfinite(err_px) & (err_px < np.float32(2.0))

    # Map camera ray [x, y, 1] back into local NED: ray_ned = R_ned_cam.T @ [x, y, 1]
    r_cam_to_ned = rot.T.astype(np.float32)
    ray_n = r_cam_to_ned[0, 0] * x + r_cam_to_ned[0, 1] * y + r_cam_to_ned[0, 2]
    ray_e = r_cam_to_ned[1, 0] * x + r_cam_to_ned[1, 1] * y + r_cam_to_ned[1, 2]
    ray_d = r_cam_to_ned[2, 0] * x + r_cam_to_ned[2, 1] * y + r_cam_to_ned[2, 2]
    horiz = np.hypot(ray_n, ray_e)

    if min_el >= 90.0:
        sky_mask = converged & (-ray_d > np.float32(0.0)) & (horiz <= np.float32(1e-7))
    else:
        tan_min = np.float32(math.tan(math.radians(min_el)))
        # In NED, Down is negative above horizon: elevation = atan2(-D, horiz) >= min_el
        sky_mask = converged & ((-ray_d) >= (horiz * tan_min))
    return sky_mask


def apply_sky_elevation_mask(
    frame_bgr: np.ndarray,
    calib: CameraIntrinsics,
    r_ned_cam: np.ndarray,
    min_elev_deg: float = MIN_SKY_ELEVATION_MASK_DEG,
    *,
    precomputed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Zero every image pixel below `min_elev_deg` (`>= +5.0 deg`) BEFORE disk encoding (`DR-15`).

    Guarantees private homes, buildings, vehicles, license plates, and people below the sky
    horizon threshold are deterministically zeroed in memory before PNG/JPEG serialization.
    """
    min_el = float(min_elev_deg)
    if not math.isfinite(min_el) or min_el < MIN_SKY_ELEVATION_MASK_DEG:
        raise RealSkyComplianceError(
            f"Sky elevation mask {min_el:.2f} deg violates mandatory >= +{MIN_SKY_ELEVATION_MASK_DEG:.1f} deg "
            "pre-encode privacy lock (DR-15)."
        )
    arr = np.asarray(frame_bgr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D grayscale or 3D BGR image array, got shape {arr.shape}")

    h_px, w_px = int(arr.shape[0]), int(arr.shape[1])
    if precomputed_mask is not None:
        sky_mask = np.asarray(precomputed_mask, dtype=bool)
        if sky_mask.shape != (h_px, w_px):
            raise ValueError(
                f"precomputed_mask shape {sky_mask.shape} does not match frame shape {(h_px, w_px)}"
            )
    else:
        sky_mask = compute_sky_elevation_mask(
            calib,
            r_ned_cam,
            min_elev_deg=min_el,
            image_shape=(h_px, w_px),
        )

    masked = arr.copy()
    masked[~sky_mask] = 0
    return masked


def snap_target_contrast_centroid(
    frame_bgr: np.ndarray,
    prior_u_px: float,
    prior_v_px: float,
    *,
    sigma_u_px: float = 15.0,
    sigma_v_px: float = 15.0,
    max_sigma: float = 3.0,
    min_snr: float = 4.5,
    kernel_size_px: int = 11,
    sky_mask: np.ndarray | None = None,
) -> ContrastSnapResult:
    """Local dual-polarity top-hat / bottom-hat contrast-centroid visual snap (`DR-15` Step 5).

    Searches strictly within the bounded `max_sigma` (default `3σ`) Mahalanobis ellipse around
    `(prior_u_px, prior_v_px)` using morphological top-hat (bright sunlit fuselages) and
    bottom-hat (dark silhouettes) filters. Snaps ONLY when local peak contrast exceeds
    `min_snr` (default `4.5`); otherwise falls back honestly with `snapped=False` and the
    unmodified ADS-B prior coordinates (never hallucinating a visual lock on noise).
    """
    arr = np.asarray(frame_bgr)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif arr.ndim == 2:
        gray = arr.astype(np.float32)
    else:
        raise ValueError(f"Expected 2D grayscale or 3D BGR image array, got shape {arr.shape}")

    sig_u = float(sigma_u_px)
    sig_v = float(sigma_v_px)
    m_sig = float(max_sigma)
    thr_snr = float(min_snr)
    u0 = float(prior_u_px)
    v0 = float(prior_v_px)
    for name, val in (
        ("prior_u_px", u0),
        ("prior_v_px", v0),
        ("sigma_u_px", sig_u),
        ("sigma_v_px", sig_v),
        ("max_sigma", m_sig),
        ("min_snr", thr_snr),
    ):
        if not math.isfinite(val):
            raise ValueError(f"Non-finite contrast snap parameter '{name}': {val!r}")
    if sig_u <= 0.0 or sig_v <= 0.0 or m_sig <= 0.0 or thr_snr <= 0.0:
        raise ValueError("sigma_u_px, sigma_v_px, max_sigma, and min_snr must all be > 0")

    k = max(5, int(kernel_size_px)) | 1
    rad_u = max(3.0, m_sig * sig_u)
    rad_v = max(3.0, m_sig * sig_v)
    search_radius = max(rad_u, rad_v)

    h_img, w_img = gray.shape
    pad = k
    x0 = max(0, int(math.floor(u0 - rad_u)) - pad)
    x1 = min(w_img, int(math.ceil(u0 + rad_u)) + pad + 1)
    y0 = max(0, int(math.floor(v0 - rad_v)) - pad)
    y1 = min(h_img, int(math.ceil(v0 + rad_v)) + pad + 1)

    def _no_snap(snr_val: float = 0.0) -> ContrastSnapResult:
        return ContrastSnapResult(
            snapped=False,
            u_px=u0,
            v_px=v0,
            prior_u_px=u0,
            prior_v_px=v0,
            du_px=0.0,
            dv_px=0.0,
            snr=float(max(0.0, snr_val)),
            polarity="NONE",
            search_radius_px=search_radius,
        )

    if (x1 - x0) < 5 or (y1 - y0) < 5:
        return _no_snap(0.0)

    roi = gray[y0:y1, x0:x1].copy()
    roi_h, roi_w = roi.shape

    # Sky-mask boundary protection: fill masked-out ground pixels with sky median so the
    # artificial >= +5 deg horizon cut never produces a false morphological contrast peak.
    if sky_mask is not None:
        mask_arr = np.asarray(sky_mask, dtype=bool)
        if mask_arr.shape != (h_img, w_img):
            raise ValueError(f"sky_mask shape {mask_arr.shape} does not match image shape {(h_img, w_img)}")
        roi_valid = mask_arr[y0:y1, x0:x1]
        if not np.any(roi_valid):
            return _no_snap(0.0)
        sky_med = float(np.median(roi[roi_valid]))
        roi[~roi_valid] = sky_med
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2))
        safe_valid = cv2.erode(roi_valid.astype(np.uint8), erode_k) > 0
        if not np.any(safe_valid):
            safe_valid = roi_valid
    else:
        roi_valid = np.ones((roi_h, roi_w), dtype=bool)
        safe_valid = roi_valid

    yy, xx = np.mgrid[0:roi_h, 0:roi_w]
    gu = xx.astype(np.float64) + float(x0)
    gv = yy.astype(np.float64) + float(y0)
    mahal2 = ((gu - u0) / sig_u) ** 2 + ((gv - v0) / sig_v) ** 2
    in_ellipse = (mahal2 <= (m_sig * m_sig)) & safe_valid
    if int(np.count_nonzero(in_ellipse)) < 9:
        return _no_snap(0.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    roi_s = cv2.GaussianBlur(roi, (3, 3), 0.7)
    tophat = cv2.morphologyEx(roi_s, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(roi_s, cv2.MORPH_BLACKHAT, kernel)

    sig_bg_blur = max(2.0, float(k) / 2.5)
    hp_s = roi_s - cv2.GaussianBlur(roi_s, (0, 0), sig_bg_blur)
    hp_raw = roi - cv2.GaussianBlur(roi, (0, 0), sig_bg_blur)

    best_snr = 0.0
    best_pol = "NONE"
    best_resp: np.ndarray | None = None
    best_py = 0
    best_px = 0
    best_med_r = 0.0

    bg_pool = (mahal2 <= (m_sig * m_sig)) & roi_valid
    core_rad2 = float((k // 2 + 1) ** 2)

    for pol, resp, sign in (
        ("BRIGHT_TOPHAT", tophat, 1.0),
        ("DARK_BOTTOMHAT", blackhat, -1.0),
    ):
        cand_map = np.where(in_ellipse, resp, np.float32(-1.0))
        py_i, px_i = np.unravel_index(int(np.argmax(cand_map)), resp.shape)
        py, px = int(py_i), int(px_i)
        core = ((xx - px) ** 2 + (yy - py) ** 2) <= core_rad2
        bg_mask = bg_pool & (~core)
        if int(np.count_nonzero(bg_mask)) < 8:
            bg_mask = roi_valid & (~core)
        if int(np.count_nonzero(bg_mask)) < 4:
            continue

        hp_bg = hp_s[bg_mask]
        med_hp = float(np.median(hp_bg))
        sigma_hp = max(1.4826 * float(np.median(np.abs(hp_bg - med_hp))), float(np.std(hp_bg)), 1.0)

        hp_rbg = hp_raw[bg_mask]
        med_raw = float(np.median(hp_rbg))
        sigma_raw = max(1.4826 * float(np.median(np.abs(hp_rbg - med_raw))), float(np.std(hp_rbg)), 1.0)

        r_bg = resp[bg_mask]
        med_r = float(np.median(r_bg))
        sigma_r = max(1.4826 * float(np.median(np.abs(r_bg - med_r))), float(np.std(r_bg)), 1.0)

        denom = max(sigma_r, sigma_hp, 0.7 * sigma_raw, 1.0)
        snr_morph = (float(resp[py, px]) - med_r) / denom
        snr_hp = (sign * (float(hp_s[py, px]) - med_hp)) / denom
        snr = min(snr_morph, snr_hp)

        if snr > best_snr:
            best_snr = snr
            best_pol = pol
            best_resp = resp
            best_py = py
            best_px = px
            best_med_r = med_r

    if best_resp is None or best_snr < thr_snr:
        return _no_snap(best_snr)

    # Sub-pixel intensity-weighted centroid inside the winning peak neighborhood
    r_c = max(2, k // 2)
    cy0 = max(0, best_py - r_c)
    cy1 = min(roi_h, best_py + r_c + 1)
    cx0 = max(0, best_px - r_c)
    cx1 = min(roi_w, best_px + r_c + 1)

    patch = best_resp[cy0:cy1, cx0:cx1]
    patch_valid = safe_valid[cy0:cy1, cx0:cx1]
    w_raw = np.where(patch_valid, np.maximum(patch - np.float32(best_med_r), np.float32(0.0)), np.float32(0.0))
    w_max = float(w_raw.max())
    if w_max <= 1e-6:
        return _no_snap(best_snr)
    weights = np.where(w_raw >= np.float32(0.25 * w_max), w_raw, np.float32(0.0)).astype(np.float64)
    w_sum = float(weights.sum())
    if w_sum <= 1e-9:
        return _no_snap(best_snr)

    py_grid, px_grid = np.mgrid[cy0:cy1, cx0:cx1]
    u_snap = float(x0) + float((weights * px_grid.astype(np.float64)).sum()) / w_sum
    v_snap = float(y0) + float((weights * py_grid.astype(np.float64)).sum()) / w_sum

    # Re-verify snapped centroid remains strictly within the 3σ Mahalanobis ellipse & sky mask
    if ((u_snap - u0) / sig_u) ** 2 + ((v_snap - v0) / sig_v) ** 2 > (m_sig * m_sig):
        return _no_snap(best_snr)
    if sky_mask is not None:
        iu = max(0, min(w_img - 1, int(round(u_snap))))
        iv = max(0, min(h_img - 1, int(round(v_snap))))
        if not bool(sky_mask[iv, iu]):
            return _no_snap(best_snr)

    return ContrastSnapResult(
        snapped=True,
        u_px=u_snap,
        v_px=v_snap,
        prior_u_px=u0,
        prior_v_px=v0,
        du_px=u_snap - u0,
        dv_px=v_snap - v0,
        snr=float(best_snr),
        polarity=best_pol,
        search_radius_px=search_radius,
    )


# ---------------------------------------------------------------------------
# RFC 8032 Pure-Stdlib Ed25519 Attestation (`DR-15` / `DR-19` Shard Signing)
# ---------------------------------------------------------------------------
_ED25519_P: int = (1 << 255) - 19
_ED25519_L: int = (1 << 252) + 27742317777372353535851937790883648493
_ED25519_D: int = (-121665 * pow(121666, -1, _ED25519_P)) % _ED25519_P
_ED25519_I: int = pow(2, (_ED25519_P - 1) // 4, _ED25519_P)
_ED25519_BY: int = (4 * pow(5, -1, _ED25519_P)) % _ED25519_P
_ED25519_BX: int = 15112221349535400772501151409588531511454012693041857206046113283949847762202
_ED25519_B: tuple[int, int, int, int] = (
    _ED25519_BX,
    _ED25519_BY,
    1,
    (_ED25519_BX * _ED25519_BY) % _ED25519_P,
)
_ED25519_ID: tuple[int, int, int, int] = (0, 1, 1, 0)


def _ed25519_point_add(
    p1: tuple[int, int, int, int],
    p2: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    p = _ED25519_P
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % p
    b = ((y1 + x1) * (y2 + x2)) % p
    c = (t1 * 2 * _ED25519_D * t2) % p
    d = (z1 * 2 * z2) % p
    e = (b - a) % p
    f = (d - c) % p
    g = (d + c) % p
    h = (b + a) % p
    return (e * f) % p, (g * h) % p, (f * g) % p, (e * h) % p


def _ed25519_scalar_mult(scalar: int, point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    acc = _ED25519_ID
    addend = point
    k = int(scalar)
    while k > 0:
        if k & 1:
            acc = _ed25519_point_add(acc, addend)
        addend = _ed25519_point_add(addend, addend)
        k >>= 1
    return acc


def _ed25519_encode_point(point: tuple[int, int, int, int]) -> bytes:
    p = _ED25519_P
    x, y, z, _ = point
    zinv = pow(z, p - 2, p)
    xa = (x * zinv) % p
    ya = (y * zinv) % p
    enc = ya | ((xa & 1) << 255)
    return enc.to_bytes(32, "little")


def _ed25519_decode_point(raw: bytes) -> tuple[int, int, int, int] | None:
    if len(raw) != 32:
        return None
    p = _ED25519_P
    y = int.from_bytes(raw, "little") & ((1 << 255) - 1)
    sign = (raw[31] >> 7) & 1
    if y >= p:
        return None
    u = (y * y - 1) % p
    v = (_ED25519_D * y * y + 1) % p
    v3 = pow(v, 3, p)
    v7 = (v3 * v3 * v) % p
    x = (u * v3 * pow((u * v7) % p, (p - 5) // 8, p)) % p
    vx2 = (v * x * x) % p
    if vx2 == u:
        pass
    elif vx2 == (-u) % p:
        x = (x * _ED25519_I) % p
    else:
        return None
    if x == 0 and sign == 1:
        return None
    if (x & 1) != sign:
        x = (p - x) % p
    return x, y, 1, (x * y) % p


def _ed25519_expand_seed(seed: bytes) -> tuple[int, bytes]:
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != 32:
        raise ValueError(f"Ed25519 private seed must be exactly 32 bytes, got {len(seed) if isinstance(seed, (bytes, bytearray)) else type(seed).__name__}")
    digest = hashlib.sha512(bytes(seed)).digest()
    a_buf = bytearray(digest[:32])
    a_buf[0] &= 248
    a_buf[31] &= 127
    a_buf[31] |= 64
    return int.from_bytes(a_buf, "little"), digest[32:]


def ed25519_public_key_from_seed(seed: bytes) -> bytes:
    """Derive 32-byte RFC 8032 Ed25519 public key from a 32-byte private seed."""
    scalar_a, _ = _ed25519_expand_seed(seed)
    return _ed25519_encode_point(_ed25519_scalar_mult(scalar_a, _ED25519_B))


def ed25519_sign(seed: bytes, message: bytes) -> bytes:
    """Create a deterministic 64-byte RFC 8032 Ed25519 signature over `message`."""
    scalar_a, prefix = _ed25519_expand_seed(seed)
    pub_bytes = _ed25519_encode_point(_ed25519_scalar_mult(scalar_a, _ED25519_B))
    r = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % _ED25519_L
    r_bytes = _ed25519_encode_point(_ed25519_scalar_mult(r, _ED25519_B))
    k = int.from_bytes(hashlib.sha512(r_bytes + pub_bytes + message).digest(), "little") % _ED25519_L
    s = (r + k * scalar_a) % _ED25519_L
    return r_bytes + s.to_bytes(32, "little")


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify a 64-byte RFC 8032 Ed25519 signature (rejecting non-canonical `S >= L` or off-curve points)."""
    if not isinstance(public_key, (bytes, bytearray)) or not isinstance(signature, (bytes, bytearray)):
        return False
    pub_b = bytes(public_key)
    sig_b = bytes(signature)
    if len(pub_b) != 32 or len(sig_b) != 64:
        return False
    a_pt = _ed25519_decode_point(pub_b)
    if a_pt is None:
        return False
    r_bytes = sig_b[:32]
    r_pt = _ed25519_decode_point(r_bytes)
    if r_pt is None:
        return False
    s = int.from_bytes(sig_b[32:], "little")
    if s >= _ED25519_L:
        return False
    k = int.from_bytes(hashlib.sha512(r_bytes + pub_b + message).digest(), "little") % _ED25519_L
    lhs = _ed25519_encode_point(_ed25519_scalar_mult(s, _ED25519_B))
    rhs = _ed25519_encode_point(_ed25519_point_add(r_pt, _ed25519_scalar_mult(k, a_pt)))
    return secrets.compare_digest(lhs, rhs)


def sign_manifest_sha256(
    manifest_sha256: str,
    *,
    private_seed: bytes | None = None,
) -> dict[str, Any]:
    """Emit detached RFC 8032 Ed25519 signature metadata over a 64-hex `manifest_sha256`."""
    digest_hex = str(manifest_sha256).strip().lower()
    if len(digest_hex) != 64:
        raise ValueError(f"manifest_sha256 must be 64 hex chars, got length {len(digest_hex)}")
    bytes.fromhex(digest_hex)  # validate hex

    seed = secrets.token_bytes(32) if private_seed is None else bytes(private_seed)
    pub_bytes = ed25519_public_key_from_seed(seed)
    sig_bytes = ed25519_sign(seed, digest_hex.encode("ascii"))
    key_fp = hashlib.sha256(pub_bytes).hexdigest()[:16]
    return {
        "schema": "openxvs.realsky.manifest_sig/1",
        "algorithm": "Ed25519",
        "rfc": "RFC-8032",
        "signed_field": "manifest_sha256",
        "manifest_sha256": digest_hex,
        "public_key_hex": pub_bytes.hex(),
        "signature_hex": sig_bytes.hex(),
        "key_fingerprint_sha256": key_fp,
    }


def verify_manifest_signature(
    shard_manifest: Mapping[str, Any],
    manifest_sig: Mapping[str, Any],
) -> bool:
    """Verify both `shard_manifest.json` canonical SHA-256 integrity AND detached Ed25519 signature."""
    if not isinstance(shard_manifest, Mapping) or not isinstance(manifest_sig, Mapping):
        return False
    if "manifest_sha256" not in shard_manifest or "manifest_sha256" not in manifest_sig:
        return False
    if "algorithm" not in manifest_sig or manifest_sig["algorithm"] != "Ed25519":
        return False
    if "public_key_hex" not in manifest_sig or "signature_hex" not in manifest_sig:
        return False

    manifest_hash = str(shard_manifest["manifest_sha256"]).strip().lower()
    sig_hash = str(manifest_sig["manifest_sha256"]).strip().lower()
    if len(manifest_hash) != 64 or not secrets.compare_digest(manifest_hash, sig_hash):
        return False

    # Recompute canonical SHA-256 over shard_manifest excluding the manifest_sha256 field
    body = {k: v for k, v in shard_manifest.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recomputed = hashlib.sha256(canonical).hexdigest()
    if not secrets.compare_digest(recomputed, manifest_hash):
        return False

    try:
        pub_bytes = bytes.fromhex(str(manifest_sig["public_key_hex"]).strip())
        sig_bytes = bytes.fromhex(str(manifest_sig["signature_hex"]).strip())
    except ValueError:
        return False
    return ed25519_verify(pub_bytes, manifest_hash.encode("ascii"), sig_bytes)
