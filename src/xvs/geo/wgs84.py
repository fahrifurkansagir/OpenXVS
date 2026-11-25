"""WGS84 ellipsoid geometry, coordinate transforms, and local NED frames.

Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 §2 and §3.
"""

from __future__ import annotations
import math
import numpy as np

# WGS84 ellipsoid constants (CONVENTIONS.md §2)
WGS84_A: float = 6378137.0
WGS84_F: float = 1.0 / 298.257223563
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # 6356752.314245...
WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)  # 0.006694379990141...
WGS84_E_PRIME2: float = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)


def radius_n(lat_rad: float) -> float:
    """Prime vertical radius of curvature N(phi) in metres."""
    sin_phi = math.sin(lat_rad)
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)


def radius_m(lat_rad: float) -> float:
    """Meridional radius of curvature M(phi) in metres."""
    sin_phi = math.sin(lat_rad)
    denom = 1.0 - WGS84_E2 * sin_phi * sin_phi
    return WGS84_A * (1.0 - WGS84_E2) / (denom * math.sqrt(denom))


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Convert geodetic coordinates (lat deg, lon deg, h metres) to ECEF (metres).

    All internal calculations use float64.
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)

    n = radius_n(lat_rad)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF coordinates (metres) to geodetic (lat deg, lon deg, h metres).

    Uses Bowring's method iterated to full convergence (< 1e-10 m accuracy).
    Returns (lat_deg, lon_deg, h_m).
    """
    p = math.hypot(x, y)
    if p < 1e-10:
        # Near polar axis
        lat_rad = math.pi / 2.0 if z >= 0.0 else -math.pi / 2.0
        h_m = abs(z) - WGS84_B
        lon_rad = 0.0
        return math.degrees(lat_rad), math.degrees(lon_rad), h_m

    lon_rad = math.atan2(y, x)

    # Initial Bowring estimation
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    phi = math.atan2(
        z + WGS84_E_PRIME2 * WGS84_B * sin_t * sin_t * sin_t,
        p - WGS84_E2 * WGS84_A * cos_t * cos_t * cos_t,
    )

    # Iterate to machine convergence (typically 2-3 iterations)
    for _ in range(6):
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
        h = p / cos_phi - n
        phi_next = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(phi_next - phi) < 1e-14:
            phi = phi_next
            break
        phi = phi_next

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    h_m = p / cos_phi - n

    return math.degrees(phi), math.degrees(lon_rad), h_m


def ned_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local NED basis unit vectors expressed in ECEF at (lat_deg, lon_deg).

    Conforms to CONVENTIONS.md §3:
      n = (-sinφ cosλ, -sinφ sinλ, cosφ)
      e = (-sinλ, cosλ, 0)
      d = (-cosφ cosλ, -cosφ sinλ, -sinφ)
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    n = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    e = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    d = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat], dtype=np.float64)
    return n, e, d


def r_ecef_ned(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Compute rotation matrix R_ecef_ned mapping NED vectors to ECEF.

    Columns are [n, e, d] expressed in ECEF.
    v_ecef = R_ecef_ned @ v_ned
    """
    n, e, d = ned_basis(lat_deg, lon_deg)
    return np.column_stack([n, e, d])


def ned_offset_to_geodetic(
    lat_deg: float, lon_deg: float, dn_m: float, de_m: float
) -> tuple[float, float, float]:
    """Exact tangent-plane construction of geodetic point offset by (dn, de) metres.

    Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.2:
      1. P = geodetic_to_ecef(lat, lon, 0) + R_ecef_ned(lat, lon) @ [dn, de, 0]
      2. (lat', lon', _) = ecef_to_geodetic(P)
    Returns (lat_deg, lon_deg, h_m).
    """
    p_ref = geodetic_to_ecef(lat_deg, lon_deg, 0.0)
    r_mat = r_ecef_ned(lat_deg, lon_deg)
    offset_ecef = r_mat @ np.array([dn_m, de_m, 0.0], dtype=np.float64)
    p = p_ref + offset_ecef
    return ecef_to_geodetic(p[0], p[1], p[2])
