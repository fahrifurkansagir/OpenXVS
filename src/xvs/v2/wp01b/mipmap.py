"""Exact L0..L(log2 N) uint16 Min-Max Mipmap DEM pyramid builder and KSFO XVS3DEP1 loader.

Strictly obeys CONVENTIONS.md v1.0 (WGS-84 ellipsoidal heights, local NED tangent coordinates,
zero internal ENU) and SP2_M2_MODELS_DESIGN.md §3.4:
  - 256x256 real KSFO 3DEP+GEOID18 pack -> 9 levels (L0..L8)
  - 4096x4096 synthetic/production DEM grid -> 13 levels (L0..L12)
  - Invariant enforced strictly at every level: min_parent <= min_child <= max_child <= max_parent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from xvs.geo.wgs84 import WGS84_A

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_KSFO_PACK_PATH = (
    REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.bin"
)
MAGIC_XVS3DEP1 = b"XVS3DEP1"


@dataclass(frozen=True)
class MinMaxMipmapPyramid:
    """Hierarchical uint16 min/max DEM pyramid in local NED tangent coordinates."""

    levels_min: tuple[np.ndarray, ...]
    levels_max: tuple[np.ndarray, ...]
    h_min_m: float
    h_step_m: float
    origin_lat_deg: float
    origin_lon_deg: float
    lat_min_deg: float
    lat_max_deg: float
    lon_min_deg: float
    lon_max_deg: float
    north_min_m: float
    north_max_m: float
    east_min_m: float
    east_max_m: float
    source_name: str
    terrain_mode: str
    geoid_model: str
    ksfo_28l_thr_lat_deg: float | None
    ksfo_28l_thr_lon_deg: float | None
    ksfo_28l_thr_wgs84_h_m: float | None
    runways: tuple[dict[str, Any], ...]

    @property
    def num_levels(self) -> int:
        return len(self.levels_min)

    @property
    def grid_size(self) -> int:
        return int(self.levels_min[0].shape[0])

    @property
    def extent_north_m(self) -> float:
        return self.north_max_m - self.north_min_m

    @property
    def extent_east_m(self) -> float:
        return self.east_max_m - self.east_min_m


def u16_to_meters(q: np.ndarray | int | float, h_min_m: float, h_step_m: float) -> np.ndarray | float:
    """Convert quantized uint16 height code(s) to WGS-84 ellipsoidal metres."""
    if isinstance(q, np.ndarray):
        return h_min_m + q.astype(np.float64) * h_step_m
    return float(h_min_m + float(q) * h_step_m)


def build_minmax_mipmap(
    height_u16: np.ndarray,
    *,
    h_min_m: float = -100.0,
    h_step_m: float = 0.1,
    origin_lat_deg: float = 37.6188056,
    origin_lon_deg: float = -122.3754167,
    lat_min_deg: float = 37.5738144,
    lat_max_deg: float = 37.6637968,
    lon_min_deg: float = -122.4320746,
    lon_max_deg: float = -122.3187588,
    north_min_m: float | None = None,
    north_max_m: float | None = None,
    east_min_m: float | None = None,
    east_max_m: float | None = None,
    source_name: str = "synthetic",
    terrain_mode: str = "synthetic_u16",
    geoid_model: str = "WGS84",
    ksfo_28l_thr_lat_deg: float | None = None,
    ksfo_28l_thr_lon_deg: float | None = None,
    ksfo_28l_thr_wgs84_h_m: float | None = None,
    runways: tuple[dict[str, Any], ...] = (),
) -> MinMaxMipmapPyramid:
    """Build exact L0..L(log2 N) uint16 Min-Max Mipmap pyramid from a power-of-two square grid."""
    arr = np.asarray(height_u16)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D heightfield array, got shape {arr.shape}")
    if arr.dtype != np.uint16:
        raise ValueError(f"Expected uint16 heightfield dtype, got {arr.dtype}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h != w or h < 2 or (h & (h - 1)) != 0:
        raise ValueError(f"Heightfield must be square power-of-two >= 2, got {arr.shape}")
    if not (math.isfinite(h_min_m) and math.isfinite(h_step_m) and h_step_m > 0.0):
        raise ValueError("h_min_m and positive h_step_m must be finite")

    n_levels = int(round(math.log2(h))) + 1
    cur_min = np.ascontiguousarray(arr, dtype=np.uint16)
    cur_max = cur_min.copy()
    levels_min: list[np.ndarray] = [cur_min]
    levels_max: list[np.ndarray] = [cur_max]

    for _ in range(1, n_levels):
        sz = int(cur_min.shape[0])
        half = sz // 2
        cur_min = cur_min.reshape(half, 2, half, 2).min(axis=(1, 3)).astype(np.uint16, copy=False)
        cur_max = cur_max.reshape(half, 2, half, 2).max(axis=(1, 3)).astype(np.uint16, copy=False)
        levels_min.append(np.ascontiguousarray(cur_min))
        levels_max.append(np.ascontiguousarray(cur_max))

    # Derive local NED bounds at (origin_lat_deg, origin_lon_deg) if not explicitly provided
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(origin_lat_deg))
    n_min = (lat_min_deg - origin_lat_deg) * m_per_deg_lat if north_min_m is None else float(north_min_m)
    n_max = (lat_max_deg - origin_lat_deg) * m_per_deg_lat if north_max_m is None else float(north_max_m)
    e_min = (lon_min_deg - origin_lon_deg) * m_per_deg_lon if east_min_m is None else float(east_min_m)
    e_max = (lon_max_deg - origin_lon_deg) * m_per_deg_lon if east_max_m is None else float(east_max_m)

    if not (n_max > n_min and e_max > e_min):
        raise ValueError("Invalid NED bounds: require north_max_m > north_min_m and east_max_m > east_min_m")

    return MinMaxMipmapPyramid(
        levels_min=tuple(levels_min),
        levels_max=tuple(levels_max),
        h_min_m=float(h_min_m),
        h_step_m=float(h_step_m),
        origin_lat_deg=float(origin_lat_deg),
        origin_lon_deg=float(origin_lon_deg),
        lat_min_deg=float(lat_min_deg),
        lat_max_deg=float(lat_max_deg),
        lon_min_deg=float(lon_min_deg),
        lon_max_deg=float(lon_max_deg),
        north_min_m=n_min,
        north_max_m=n_max,
        east_min_m=e_min,
        east_max_m=e_max,
        source_name=source_name,
        terrain_mode=terrain_mode,
        geoid_model=geoid_model,
        ksfo_28l_thr_lat_deg=ksfo_28l_thr_lat_deg,
        ksfo_28l_thr_lon_deg=ksfo_28l_thr_lon_deg,
        ksfo_28l_thr_wgs84_h_m=ksfo_28l_thr_wgs84_h_m,
        runways=runways,
    )


def verify_mipmap_invariant(pyramid: MinMaxMipmapPyramid) -> bool:
    """Strictly verify `min_parent <= min_child <= max_child <= max_parent` at EVERY texel & level."""
    n_levels = pyramid.num_levels
    if n_levels < 2:
        raise ValueError("Mipmap pyramid must contain at least 2 levels")

    expected_sz = pyramid.grid_size
    for lvl in range(n_levels):
        mn = pyramid.levels_min[lvl]
        mx = pyramid.levels_max[lvl]
        if mn.shape != (expected_sz, expected_sz) or mx.shape != (expected_sz, expected_sz):
            raise AssertionError(f"Level {lvl} shape mismatch: {mn.shape} / {mx.shape} vs {(expected_sz, expected_sz)}")
        if mn.dtype != np.uint16 or mx.dtype != np.uint16:
            raise AssertionError(f"Level {lvl} dtype must be uint16")
        if not bool(np.all(mn <= mx)):
            raise AssertionError(f"Level {lvl} violates min <= max")
        if lvl > 0:
            child_min = pyramid.levels_min[lvl - 1]
            child_max = pyramid.levels_max[lvl - 1]
            p_min_rep = np.repeat(np.repeat(mn, 2, axis=0), 2, axis=1)
            p_max_rep = np.repeat(np.repeat(mx, 2, axis=0), 2, axis=1)
            if not bool(np.all(p_min_rep <= child_min)):
                raise AssertionError(f"Level {lvl} violates min_parent <= min_child")
            if not bool(np.all(child_min <= child_max)):
                raise AssertionError(f"Level {lvl - 1} violates min_child <= max_child")
            if not bool(np.all(child_max <= p_max_rep)):
                raise AssertionError(f"Level {lvl} violates max_child <= max_parent")
        expected_sz = max(1, expected_sz // 2)

    if pyramid.levels_min[-1].shape != (1, 1):
        raise AssertionError("Top level of Min-Max Mipmap must be 1x1")
    return True


def load_ksfo_3dep_mipmap(
    bin_path: Path | str = DEFAULT_KSFO_PACK_PATH,
    *,
    ring_name: str = "inner_ksfo",
) -> MinMaxMipmapPyramid:
    """Load real M1 `ksfo_3dep_geoid18_v1.bin` (`XVS3DEP1`) into a 9-level uint16 Min-Max Mipmap."""
    path = Path(bin_path)
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"Terrain pack {path} truncated (< 16 bytes)")
    magic, version, hdr_len = struct.unpack("<8sII", raw[:16])
    if magic != MAGIC_XVS3DEP1:
        raise ValueError(f"Invalid terrain pack magic {magic!r}; expected {MAGIC_XVS3DEP1!r}")
    if version != 1:
        raise ValueError(f"Unsupported XVS3DEP1 version {version}")
    if 16 + hdr_len > len(raw):
        raise ValueError("Terrain pack JSON header extends past EOF")

    hdr: dict[str, Any] = json.loads(raw[16 : 16 + hdr_len].decode("utf-8"))
    rings: list[dict[str, Any]] = hdr["rings"]
    selected: dict[str, Any] | None = None
    for r in rings:
        if r["name"] == ring_name:
            selected = r
            break
    if selected is None:
        avail = [r["name"] for r in rings]
        raise KeyError(f"Ring {ring_name!r} not found in {path}; available: {avail}")

    w = int(selected["width"])
    h = int(selected["height"])
    byte_off = int(selected["byte_offset"])
    byte_len = int(selected["byte_length"])
    expected_bytes = w * h * 2
    if byte_len != expected_bytes or byte_off + byte_len > len(raw):
        raise ValueError(f"Ring {ring_name!r} byte slice invalid ({byte_off}..{byte_off + byte_len})")

    grid_u16 = np.frombuffer(raw, dtype="<u2", count=w * h, offset=byte_off).reshape(h, w).copy()
    quant = hdr["quantization"]
    runways_list = tuple(hdr["runways"]) if "runways" in hdr else ()

    return build_minmax_mipmap(
        grid_u16,
        h_min_m=float(quant["h_min_m"]),
        h_step_m=float(quant["h_step_m"]),
        origin_lat_deg=float(hdr["origin_lat_deg"]),
        origin_lon_deg=float(hdr["origin_lon_deg"]),
        lat_min_deg=float(selected["lat_min"]),
        lat_max_deg=float(selected["lat_max"]),
        lon_min_deg=float(selected["lon_min"]),
        lon_max_deg=float(selected["lon_max"]),
        source_name=f"{path.name}:{ring_name}",
        terrain_mode=str(hdr["terrain_mode"]),
        geoid_model=str(hdr["geoid_model"]),
        ksfo_28l_thr_lat_deg=float(hdr["ksfo_28l_thr_lat_deg"]),
        ksfo_28l_thr_lon_deg=float(hdr["ksfo_28l_thr_lon_deg"]),
        ksfo_28l_thr_wgs84_h_m=float(hdr["ksfo_28l_thr_wgs84_h_m"]),
        runways=runways_list,
    )


def sample_height_bilinear_m(
    pyramid: MinMaxMipmapPyramid,
    north_m: float,
    east_m: float,
) -> float:
    """Sample exact bilinear WGS-84 ellipsoidal height (m) from L0 at local NED `(north_m, east_m)`."""
    n = pyramid.grid_size
    # Row 0 = north_max_m, Row N-1 = north_min_m; Col 0 = east_min_m, Col N-1 = east_max_m
    u_col = (east_m - pyramid.east_min_m) / pyramid.extent_east_m * float(n - 1)
    v_row = (pyramid.north_max_m - north_m) / pyramid.extent_north_m * float(n - 1)

    u_clamped = max(0.0, min(float(n - 1), u_col))
    v_clamped = max(0.0, min(float(n - 1), v_row))

    c0 = int(math.floor(u_clamped))
    r0 = int(math.floor(v_clamped))
    c1 = min(c0 + 1, n - 1)
    r1 = min(r0 + 1, n - 1)
    fc = u_clamped - c0
    fr = v_clamped - r0

    g = pyramid.levels_min[0]
    h00 = pyramid.h_min_m + float(g[r0, c0]) * pyramid.h_step_m
    h10 = pyramid.h_min_m + float(g[r0, c1]) * pyramid.h_step_m
    h01 = pyramid.h_min_m + float(g[r1, c0]) * pyramid.h_step_m
    h11 = pyramid.h_min_m + float(g[r1, c1]) * pyramid.h_step_m

    top = h00 + (h10 - h00) * fc
    bot = h01 + (h11 - h01) * fc
    return float(top + (bot - top) * fr)
