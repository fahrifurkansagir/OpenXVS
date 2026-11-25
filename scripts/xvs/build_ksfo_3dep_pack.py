"""Build offline dual-ring 16-bit USGS 3DEP + NOAA GEOID18 + OurAirports KSFO terrain clipmap pack.

Strictly enforces `docs/xvs/spec/data_allowlist.json` v1.4 (`URLClassifier`) on every upstream
HTTP request, caches raw downloads under `data/cache/` (gitignored), converts NAVD88 orthometric
heights and OurAirports ft MSL to WGS-84 ellipsoidal heights (`h = H + N_geoid18`, KSFO 28L
threshold `h ~= -28.3 m`), and packs outer SF Bay (40x40 km) + inner KSFO (10x10 km) heightfields
plus `28L`/`28R`/`Taxiway C` runway threshold polygons into:
  - `apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin` (`magic="XVS3DEP1"`, Brotli <= 300 KiB)
  - `apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.manifest.json`
  - `output/xvs/v2_m1/ksfo_3dep_data_sources.json`

Supports explicit `--terrain-mode procedural-ci --out <path>` for hermetic CI testing (emits
`terrain_mode="procedural_ci"` in the binary header; NEVER silently falls back in default mode).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.geo import wgs84
from xvs.sim.allowlist import URLClassifier

DEFAULT_OUT_BIN = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.bin"
DEFAULT_OUT_MANIFEST = (
    REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.manifest.json"
)
DEFAULT_OUT_SOURCES = REPO_ROOT / "output" / "xvs" / "v2_m1" / "ksfo_3dep_data_sources.json"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "cache"
DEFAULT_OURAIRPORTS_REF = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"

KSFO_CENTER_LAT_DEG = 37.6188056
KSFO_CENTER_LON_DEG = -122.3754167

GEOID18_PRIMARY_URL = "https://cdn.proj.org/us_noaa_g2018u0.tif"
OURAIRPORTS_RUNWAYS_URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
USGS_3DEP_IMAGESERVER = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)

USER_AGENT = "OpenXVS-research/0.1"
FT_TO_M = 0.3048
WEB_MERCATOR_R_M = 6378137.0
MAX_BROTLI_BYTES = 307_200


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def measure_brotli_bytes(path: Path) -> int:
    js = (
        "const fs=require('fs'),z=require('zlib');"
        "const b=fs.readFileSync(process.argv[1]);"
        "const c=z.brotliCompressSync(b,{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}});"
        "process.stdout.write(String(c.length));"
    )
    proc = subprocess.run(
        ["node", "-e", js, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout.strip())


def lonlat_to_epsg3857(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    x = WEB_MERCATOR_R_M * math.radians(lon_deg)
    y = WEB_MERCATOR_R_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))
    return x, y


def allowlisted_fetch(
    url: str,
    cache_path: Path,
    classifier: URLClassifier,
    sources_log: list[dict[str, Any]],
    *,
    timeout_s: float = 90.0,
    max_retries: int = 4,
) -> Path:
    """Validate URL against `data_allowlist.json` v1.4 and fetch with caching & backoff."""
    verdict, reason, entry = classifier.classify_upstream_url(url)
    if verdict != "ALLOWED" or entry is None:
        raise PermissionError(
            f"Upstream URL blocked by data_allowlist.json ({verdict}): {url} — {reason}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_hit = cache_path.is_file() and cache_path.stat().st_size > 0
    if not cached_hit:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    status = int(getattr(resp, "status", 200))
                    if status != 200:
                        raise RuntimeError(f"Unexpected HTTP {status} for {url}")
                    with tmp_path.open("wb") as out_f:
                        while True:
                            chunk = resp.read(256 * 1024)
                            if not chunk:
                                break
                            out_f.write(chunk)
                tmp_path.replace(cache_path)
                last_exc = None
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt + 1 < max_retries:
                    time.sleep(2.0 ** attempt)
        if last_exc is not None:
            raise RuntimeError(f"Failed allowlisted fetch {url}: {last_exc}") from last_exc

    sources_log.append(
        {
            "allowlist_id": str(entry["id"]),
            "url": url,
            "license": str(entry["license"]),
            "attribution": str(entry["attribution"]),
            "cache_path": str(cache_path.relative_to(REPO_ROOT).as_posix())
            if cache_path.is_relative_to(REPO_ROOT)
            else str(cache_path),
            "cached_hit": bool(cached_hit),
            "bytes": int(cache_path.stat().st_size),
            "sha256": file_sha256(cache_path),
            "verified_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return cache_path


def _parse_geotiff_tags(path: Path) -> dict[str, Any]:
    """Parse classic TIFF/GeoTIFF IFD0 tags (33550 ModelPixelScale, 33922 ModelTiepoint, 42113 NoData)."""
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"File too small to be TIFF: {path}")
    endian = "<" if data[:2] == b"II" else (">" if data[:2] == b"MM" else "")
    if not endian:
        raise ValueError(f"Invalid TIFF byte-order marker {data[:2]!r} in {path}")
    magic = struct.unpack_from(endian + "H", data, 2)[0]
    if magic != 42:
        raise ValueError(f"Unsupported TIFF magic {magic} in {path}")

    ifd_offset = struct.unpack_from(endian + "I", data, 4)[0]
    num_entries = struct.unpack_from(endian + "H", data, ifd_offset)[0]

    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    type_fmts = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d"}

    tags: dict[int, Any] = {}
    pos = ifd_offset + 2
    for _ in range(num_entries):
        tag, dtype, count = struct.unpack_from(endian + "HHI", data, pos)
        val_or_off = data[pos + 8 : pos + 12]
        pos += 12
        elem_size = type_sizes.get(dtype, 1)  # default-ok: fallback byte size for unknown TIFF tag types
        total_bytes = elem_size * count
        if total_bytes <= 4:
            raw = val_or_off[:total_bytes]
        else:
            off = struct.unpack_from(endian + "I", val_or_off, 0)[0]
            raw = data[off : off + total_bytes]
        if dtype == 2:
            tags[tag] = raw.rstrip(b"\x00").decode("ascii", errors="replace")
        elif dtype in type_fmts:
            vals = struct.unpack_from(endian + f"{count}{type_fmts[dtype]}", raw, 0)
            tags[tag] = vals[0] if count == 1 else list(vals)

    if 33550 not in tags or 33922 not in tags:
        raise ValueError(f"GeoTIFF missing ModelPixelScale (33550) or ModelTiepoint (33922): {path}")

    scale = tags[33550]
    tie = tags[33922]
    nodata = None
    if 42113 in tags:
        try:
            nodata = float(str(tags[42113]).strip())
        except ValueError:
            nodata = None

    return {
        "width": int(tags[256]),
        "height": int(tags[257]),
        "scale_x": float(scale[0]),
        "scale_y": float(scale[1]),
        "tie_i": float(tie[0]),
        "tie_j": float(tie[1]),
        "tie_x": float(tie[3]),
        "tie_y": float(tie[4]),
        "nodata": nodata,
    }


def load_geotiff_f32(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a Float32 GeoTIFF raster and its parsed GeoTIFF transform tags."""
    meta = _parse_geotiff_tags(path)
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        with Image.open(path) as im:
            arr = np.asarray(im, dtype=np.float32)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != (meta["height"], meta["width"]):
        raise ValueError(
            f"GeoTIFF raster shape {arr.shape} != header {(meta['height'], meta['width'])} for {path}"
        )
    return arr, meta


def sample_geotiff_bilinear(
    raster: np.ndarray,
    meta: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    wrap_lon_360: bool = False,
    nodata_fill: float | None = None,
) -> np.ndarray:
    """Sample GeoTIFF at CRS coordinates `(xs, ys)` via bilinear interpolation."""
    x_use = np.asarray(xs, dtype=np.float64)
    y_use = np.asarray(ys, dtype=np.float64)
    if wrap_lon_360 and meta["tie_x"] > 180.0:
        x_use = np.where(x_use < 0.0, x_use + 360.0, x_use)

    # Pixel-is-area convention: tiepoint (tie_i, tie_j) maps to (tie_x, tie_y);
    # pixel center (col + 0.5, row + 0.5) corresponds to CRS coordinate.
    cols = (x_use - meta["tie_x"]) / meta["scale_x"] + meta["tie_i"] - 0.5
    rows = (meta["tie_y"] - y_use) / meta["scale_y"] + meta["tie_j"] - 0.5

    work = raster.astype(np.float64, copy=True)
    nodata = meta["nodata"]
    invalid = ~np.isfinite(work) | (np.abs(work) > 1e6)
    if nodata is not None and math.isfinite(nodata):
        invalid |= np.isclose(work, nodata, rtol=0.0, atol=1e-3)
    if np.any(invalid):
        fill_val = 0.0 if nodata_fill is None else float(nodata_fill)
        work[invalid] = fill_val

    sampled = map_coordinates(
        work,
        [rows.ravel(), cols.ravel()],
        order=1,
        mode="nearest",
    ).reshape(x_use.shape)
    return sampled


def _ring_lat_lon_bounds(center_lat_deg: float, center_lon_deg: float, extent_km: float) -> dict[str, float]:
    half_m = 0.5 * extent_km * 1000.0
    m_per_deg_lat = 111_132.92
    m_per_deg_lon = 111_412.84 * math.cos(math.radians(center_lat_deg))
    dlat = half_m / m_per_deg_lat
    dlon = half_m / m_per_deg_lon
    return {
        "lat_min": center_lat_deg - dlat,
        "lat_max": center_lat_deg + dlat,
        "lon_min": center_lon_deg - dlon,
        "lon_max": center_lon_deg + dlon,
    }


def _load_ksfo_runways_from_ourairports(
    runways_csv_path: Path,
    geoid_sampler: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Extract KSFO 28L & 28R from OurAirports CSV and construct Taxiway C polygon in WGS-84 ellipsoidal."""
    ksfo_rows: dict[str, dict[str, str]] = {}
    with runways_csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["airport_ident"] == "KSFO":
                ksfo_rows[row["he_ident"]] = row

    if "28L" not in ksfo_rows or "28R" not in ksfo_rows:
        raise ValueError("OurAirports runways.csv missing KSFO 28L or 28R records")

    p_ref_ecef = wgs84.geodetic_to_ecef(KSFO_CENTER_LAT_DEG, KSFO_CENTER_LON_DEG, 0.0)
    r_ecef_ned = wgs84.r_ecef_ned(KSFO_CENTER_LAT_DEG, KSFO_CENTER_LON_DEG)

    def _geo_to_ned(lat_d: float, lon_d: float, h_m: float) -> list[float]:
        p = wgs84.geodetic_to_ecef(lat_d, lon_d, h_m)
        ned = r_ecef_ned.T @ (p - p_ref_ecef)
        return [float(ned[0]), float(ned[1]), float(ned[2])]

    def _build_rect_polygon(
        lat0: float, lon0: float, h0: float, lat1: float, lon1: float, h1: float, width_m: float
    ) -> list[dict[str, Any]]:
        n0 = np.array(_geo_to_ned(lat0, lon0, h0), dtype=np.float64)
        n1 = np.array(_geo_to_ned(lat1, lon1, h1), dtype=np.float64)
        axis = n1[:2] - n0[:2]
        axis_u = axis / max(1e-9, float(np.linalg.norm(axis)))
        # Right-hand perpendicular in NE plane
        perp = np.array([-axis_u[1], axis_u[0]], dtype=np.float64) * (0.5 * width_m)
        corners_ned = [
            np.array([n0[0] - perp[0], n0[1] - perp[1], n0[2]]),
            np.array([n0[0] + perp[0], n0[1] + perp[1], n0[2]]),
            np.array([n1[0] + perp[0], n1[1] + perp[1], n1[2]]),
            np.array([n1[0] - perp[0], n1[1] - perp[1], n1[2]]),
        ]
        poly: list[dict[str, Any]] = []
        for c_ned in corners_ned:
            p_e = p_ref_ecef + r_ecef_ned @ c_ned
            c_lat, c_lon, c_h = wgs84.ecef_to_geodetic(float(p_e[0]), float(p_e[1]), float(p_e[2]))
            poly.append(
                {
                    "lat_deg": float(c_lat),
                    "lon_deg": float(c_lon),
                    "wgs84_h_m": float(c_h),
                    "ned_m": [float(c_ned[0]), float(c_ned[1]), float(c_ned[2])],
                }
            )
        return poly

    runways_out: list[dict[str, Any]] = []
    rwy28l_summary: dict[str, float] = {}

    for ident in ("28L", "28R"):
        row = ksfo_rows[ident]
        thr_lat = float(row["he_latitude_deg"])
        thr_lon = float(row["he_longitude_deg"])
        thr_navd88_m = float(row["he_elevation_ft"]) * FT_TO_M
        opp_lat = float(row["le_latitude_deg"])
        opp_lon = float(row["le_longitude_deg"])
        opp_navd88_m = float(row["le_elevation_ft"]) * FT_TO_M
        width_m = float(row["width_ft"]) * FT_TO_M
        length_m = float(row["length_ft"]) * FT_TO_M
        hdg_deg = float(row["he_heading_degT"])

        n_thr = float(geoid_sampler(thr_lat, thr_lon))
        n_opp = float(geoid_sampler(opp_lat, opp_lon))
        thr_wgs84_h = thr_navd88_m + n_thr
        opp_wgs84_h = opp_navd88_m + n_opp

        poly = _build_rect_polygon(
            thr_lat, thr_lon, thr_wgs84_h, opp_lat, opp_lon, opp_wgs84_h, width_m
        )
        runways_out.append(
            {
                "id": ident,
                "kind": "runway",
                "width_m": width_m,
                "length_m": length_m,
                "true_heading_deg": hdg_deg,
                "threshold_lat_deg": thr_lat,
                "threshold_lon_deg": thr_lon,
                "threshold_navd88_m": thr_navd88_m,
                "threshold_geoid18_n_m": n_thr,
                "threshold_wgs84_h_m": thr_wgs84_h,
                "threshold_ned_m": _geo_to_ned(thr_lat, thr_lon, thr_wgs84_h),
                "opposite_end_lat_deg": opp_lat,
                "opposite_end_lon_deg": opp_lon,
                "opposite_end_wgs84_h_m": opp_wgs84_h,
                "polygon": poly,
            }
        )
        if ident == "28L":
            rwy28l_summary = {
                "lat_deg": thr_lat,
                "lon_deg": thr_lon,
                "navd88_m": thr_navd88_m,
                "geoid_n_m": n_thr,
                "wgs84_h_m": thr_wgs84_h,
                "opp_lat_deg": opp_lat,
                "opp_lon_deg": opp_lon,
                "opp_wgs84_h_m": opp_wgs84_h,
                "hdg_deg": hdg_deg,
                "length_m": length_m,
            }

    # Construct parallel Taxiway C (75 ft = 22.86 m wide, ~168 m southwest of 28L centerline)
    # Matching KSFO geometry and SP1_LIVEDECK_DESIGN.md §4.2 width discrimination (61 m vs 23 m)
    n_28l_thr = np.array(
        _geo_to_ned(rwy28l_summary["lat_deg"], rwy28l_summary["lon_deg"], rwy28l_summary["wgs84_h_m"]),
        dtype=np.float64,
    )
    n_28l_opp = np.array(
        _geo_to_ned(
            rwy28l_summary["opp_lat_deg"],
            rwy28l_summary["opp_lon_deg"],
            rwy28l_summary["opp_wgs84_h_m"],
        ),
        dtype=np.float64,
    )
    axis_28l = n_28l_opp[:2] - n_28l_thr[:2]
    axis_u = axis_28l / max(1e-9, float(np.linalg.norm(axis_28l)))
    # Left of 28L heading (~298 deg -> left is ~208 deg, southwest toward terminal/Taxiway C)
    sw_perp = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)
    # Verify sw_perp points South (negative North component)
    if sw_perp[0] > 0.0:
        sw_perp = -sw_perp
    offset_m = 168.0
    twy_thr_ned = np.array(
        [n_28l_thr[0] + sw_perp[0] * offset_m, n_28l_thr[1] + sw_perp[1] * offset_m, n_28l_thr[2]]
    )
    twy_opp_ned = np.array(
        [n_28l_opp[0] + sw_perp[0] * offset_m, n_28l_opp[1] + sw_perp[1] * offset_m, n_28l_opp[2]]
    )
    p_twy0 = p_ref_ecef + r_ecef_ned @ twy_thr_ned
    p_twy1 = p_ref_ecef + r_ecef_ned @ twy_opp_ned
    twy0_lat, twy0_lon, twy0_h = wgs84.ecef_to_geodetic(float(p_twy0[0]), float(p_twy0[1]), float(p_twy0[2]))
    twy1_lat, twy1_lon, twy1_h = wgs84.ecef_to_geodetic(float(p_twy1[0]), float(p_twy1[1]), float(p_twy1[2]))
    twy_width_m = 75.0 * FT_TO_M  # 22.86 m
    runways_out.append(
        {
            "id": "Taxiway C",
            "kind": "taxiway",
            "width_m": twy_width_m,
            "length_m": float(rwy28l_summary["length_m"]),
            "true_heading_deg": float(rwy28l_summary["hdg_deg"]),
            "threshold_lat_deg": float(twy0_lat),
            "threshold_lon_deg": float(twy0_lon),
            "threshold_wgs84_h_m": float(twy0_h),
            "threshold_ned_m": [float(twy_thr_ned[0]), float(twy_thr_ned[1]), float(twy_thr_ned[2])],
            "opposite_end_lat_deg": float(twy1_lat),
            "opposite_end_lon_deg": float(twy1_lon),
            "opposite_end_wgs84_h_m": float(twy1_h),
            "polygon": _build_rect_polygon(
                twy0_lat, twy0_lon, twy0_h, twy1_lat, twy1_lon, twy1_h, twy_width_m
            ),
        }
    )

    return runways_out, rwy28l_summary


def _pack_binary_terrain(
    out_bin: Path,
    *,
    terrain_mode: str,
    geoid_model: str,
    ring_specs: list[dict[str, Any]],
    ring_grids_m: list[np.ndarray],
    runways: list[dict[str, Any]],
    rwy28l_summary: dict[str, float],
    h_min_m: float = -100.0,
    h_step_m: float = 0.1,
) -> dict[str, Any]:
    """Quantize dual-ring float heightfields to uint16 LE and write XVS3DEP1 binary pack."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    quantized_bytes: list[bytes] = []
    ring_headers: list[dict[str, Any]] = []
    for spec, grid in zip(ring_specs, ring_grids_m):
        q = np.clip(np.rint((grid - h_min_m) / h_step_m), 0.0, 65535.0).astype("<u2")
        q_bytes = q.tobytes()
        quantized_bytes.append(q_bytes)
        ring_headers.append(
            {
                "name": spec["name"],
                "extent_km": float(spec["extent_km"]),
                "width": int(grid.shape[1]),
                "height": int(grid.shape[0]),
                "lat_min": float(spec["lat_min"]),
                "lat_max": float(spec["lat_max"]),
                "lon_min": float(spec["lon_min"]),
                "lon_max": float(spec["lon_max"]),
                "min_wgs84_h_m": float(np.min(grid)),
                "max_wgs84_h_m": float(np.max(grid)),
                "byte_offset": 0,
                "byte_length": len(q_bytes),
            }
        )

    header_doc: dict[str, Any] = {
        "schema": "xvs.terrain.ksfo_3dep_pack/1",
        "terrain_mode": terrain_mode,
        "geoid_model": geoid_model,
        "datum": "WGS84_ELLIPSOIDAL",
        "origin_lat_deg": KSFO_CENTER_LAT_DEG,
        "origin_lon_deg": KSFO_CENTER_LON_DEG,
        "ksfo_28l_thr_lat_deg": float(rwy28l_summary["lat_deg"]),
        "ksfo_28l_thr_lon_deg": float(rwy28l_summary["lon_deg"]),
        "ksfo_28l_thr_navd88_m": float(rwy28l_summary["navd88_m"]),
        "ksfo_28l_geoid_undulation_m": float(rwy28l_summary["geoid_n_m"]),
        "ksfo_28l_thr_wgs84_h_m": float(rwy28l_summary["wgs84_h_m"]),
        "quantization": {
            "h_min_m": float(h_min_m),
            "h_step_m": float(h_step_m),
            "dtype": "uint16_le",
        },
        "rings": ring_headers,
        "runways": runways,
    }

    # Iterate twice so exact byte_offset fields inside JSON match final padded header length
    for _ in range(3):
        raw_hdr = json.dumps(header_doc, separators=(",", ":")).encode("utf-8")
        pad_len = (-len(raw_hdr)) % 4
        hdr_len = len(raw_hdr) + pad_len
        cursor = 16 + hdr_len
        for rh in header_doc["rings"]:
            rh["byte_offset"] = cursor
            cursor += int(rh["byte_length"])
        raw_hdr2 = json.dumps(header_doc, separators=(",", ":")).encode("utf-8")
        pad_len2 = (-len(raw_hdr2)) % 4
        if len(raw_hdr2) + pad_len2 == hdr_len:
            raw_hdr = raw_hdr2 + (b" " * pad_len2)
            break
    else:
        raw_hdr = json.dumps(header_doc, separators=(",", ":")).encode("utf-8")
        pad_len = (-len(raw_hdr)) % 4
        raw_hdr = raw_hdr + (b" " * pad_len)
        cursor = 16 + len(raw_hdr)
        for rh in header_doc["rings"]:
            rh["byte_offset"] = cursor
            cursor += int(rh["byte_length"])
        # Re-serialize with fixed width check
        raw_hdr = json.dumps(header_doc, separators=(",", ":")).encode("utf-8")
        pad_len = (-len(raw_hdr)) % 4
        raw_hdr = raw_hdr + (b" " * pad_len)

    preamble = struct.pack("<8sII", b"XVS3DEP1", 1, len(raw_hdr))
    with out_bin.open("wb") as f:
        f.write(preamble)
        f.write(raw_hdr)
        for qb in quantized_bytes:
            f.write(qb)

    return header_doc


def build_procedural_ci_pack(out_bin: Path, grid_size: int = 256) -> dict[str, Any]:
    """Build deterministic offline procedural CI pack explicitly tagged `terrain_mode='procedural_ci'`."""
    geoid_const = -32.25
    ring_defs = [
        ("outer_sf_bay", 40.0),
        ("inner_ksfo", 10.0),
    ]
    ring_specs: list[dict[str, Any]] = []
    ring_grids: list[np.ndarray] = []

    for name, extent_km in ring_defs:
        b = _ring_lat_lon_bounds(KSFO_CENTER_LAT_DEG, KSFO_CENTER_LON_DEG, extent_km)
        lats = np.linspace(b["lat_max"], b["lat_min"], grid_size, dtype=np.float64)
        lons = np.linspace(b["lon_min"], b["lon_max"], grid_size, dtype=np.float64)
        lon_g, lat_g = np.meshgrid(lons, lats)
        # Simple deterministic analytical SF Peninsula ridge west of KSFO + bay east of KSFO
        west_ridge = np.maximum(0.0, (-122.41 - lon_g) * 3500.0) * np.exp(
            -((lat_g - 37.62) / 0.14) ** 2
        )
        grid_wgs84 = np.clip(west_ridge, 0.0, 550.0) + geoid_const
        ring_specs.append({"name": name, "extent_km": extent_km, **b})
        ring_grids.append(grid_wgs84)

    if DEFAULT_OURAIRPORTS_REF.is_file():
        runways, rwy28l = _load_ksfo_runways_from_ourairports(
            DEFAULT_OURAIRPORTS_REF, lambda _lat, _lon: geoid_const
        )
    else:
        rwy28l = {
            "lat_deg": 37.61172,
            "lon_deg": -122.358367,
            "navd88_m": 3.9624,
            "geoid_n_m": geoid_const,
            "wgs84_h_m": 3.9624 + geoid_const,
        }
        runways = []

    return _pack_binary_terrain(
        out_bin,
        terrain_mode="procedural_ci",
        geoid_model="procedural_ci_constant",
        ring_specs=ring_specs,
        ring_grids_m=ring_grids,
        runways=runways,
        rwy28l_summary=rwy28l,
    )


def build_usgs_3dep_geoid18_pack(
    *,
    out_bin: Path = DEFAULT_OUT_BIN,
    out_manifest: Path = DEFAULT_OUT_MANIFEST,
    out_sources: Path = DEFAULT_OUT_SOURCES,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    grid_size: int = 256,
) -> dict[str, Any]:
    """Fetch allowlisted USGS 3DEP + NOAA GEOID18 + OurAirports and pack KSFO ellipsoidal clipmap."""
    clf = URLClassifier()
    sources_log: list[dict[str, Any]] = []

    # 1. NOAA GEOID18 CONUS grid (cached once under data/cache/geoid18/)
    geoid_cache = cache_root / "geoid18" / "us_noaa_g2018u0.tif"
    geoid_path = allowlisted_fetch(GEOID18_PRIMARY_URL, geoid_cache, clf, sources_log, timeout_s=180.0)
    geoid_raster, geoid_meta = load_geotiff_f32(geoid_path)

    def _sample_geoid_scalar(lat_d: float, lon_d: float) -> float:
        val = sample_geotiff_bilinear(
            geoid_raster,
            geoid_meta,
            np.array([lon_d]),
            np.array([lat_d]),
            wrap_lon_360=True,
        )
        return float(val[0])

    # 2. OurAirports runways.csv (prefer verified local data/reference cache, still allowlist-checked)
    if DEFAULT_OURAIRPORTS_REF.is_file():
        runways_path = allowlisted_fetch(
            OURAIRPORTS_RUNWAYS_URL, DEFAULT_OURAIRPORTS_REF, clf, sources_log
        )
    else:
        runways_path = allowlisted_fetch(
            OURAIRPORTS_RUNWAYS_URL, cache_root / "ourairports" / "runways.csv", clf, sources_log
        )
    runways, rwy28l_summary = _load_ksfo_runways_from_ourairports(runways_path, _sample_geoid_scalar)

    # Sanity-lock KSFO 28L WGS-84 ellipsoidal threshold height
    h_28l = float(rwy28l_summary["wgs84_h_m"])
    if not (-35.0 <= h_28l <= -22.0):
        raise RuntimeError(
            f"KSFO 28L ellipsoidal height {h_28l:.3f} m outside expected [-35.0, -22.0] m "
            f"(NAVD88={rwy28l_summary['navd88_m']:.3f} m, N_geoid={rwy28l_summary['geoid_n_m']:.3f} m)"
        )

    # 3. USGS 3DEP ImageServer dual rings (native EPSG:3857 Float32 GeoTIFF -> WGS-84 ellipsoidal)
    ring_defs = [
        ("outer_sf_bay", 40.0),
        ("inner_ksfo", 10.0),
    ]
    ring_specs: list[dict[str, Any]] = []
    ring_grids_wgs84: list[np.ndarray] = []

    for name, extent_km in ring_defs:
        b = _ring_lat_lon_bounds(KSFO_CENTER_LAT_DEG, KSFO_CENTER_LON_DEG, extent_km)
        x0, y0 = lonlat_to_epsg3857(b["lon_min"], b["lat_min"])
        x1, y1 = lonlat_to_epsg3857(b["lon_max"], b["lat_max"])
        url_3dep = (
            f"{USGS_3DEP_IMAGESERVER}"
            f"?bbox={x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}"
            f"&bboxSR=3857&imageSR=3857&size={grid_size},{grid_size}"
            f"&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image"
        )
        tif_cache = cache_root / "usgs_3dep" / f"ksfo_{name}_{grid_size}_3857.tif"
        tif_path = allowlisted_fetch(url_3dep, tif_cache, clf, sources_log, timeout_s=90.0)
        dem_raster, dem_meta = load_geotiff_f32(tif_path)

        lats = np.linspace(b["lat_max"], b["lat_min"], grid_size, dtype=np.float64)
        lons = np.linspace(b["lon_min"], b["lon_max"], grid_size, dtype=np.float64)
        lon_g, lat_g = np.meshgrid(lons, lats)

        x_3857 = WEB_MERCATOR_R_M * np.radians(lon_g)
        y_3857 = WEB_MERCATOR_R_M * np.log(np.tan(np.pi / 4.0 + np.radians(lat_g) / 2.0))

        h_navd88 = sample_geotiff_bilinear(
            dem_raster, dem_meta, x_3857, y_3857, wrap_lon_360=False, nodata_fill=0.0
        )
        n_geoid = sample_geotiff_bilinear(
            geoid_raster, geoid_meta, lon_g, lat_g, wrap_lon_360=True
        )
        h_wgs84 = h_navd88 + n_geoid
        ring_specs.append({"name": name, "extent_km": extent_km, **b})
        ring_grids_wgs84.append(h_wgs84)

    header_doc = _pack_binary_terrain(
        out_bin,
        terrain_mode="usgs_3dep_geoid18_ellipsoidal",
        geoid_model="GEOID18",
        ring_specs=ring_specs,
        ring_grids_m=ring_grids_wgs84,
        runways=runways,
        rwy28l_summary=rwy28l_summary,
    )

    raw_bytes = int(out_bin.stat().st_size)
    br_bytes = measure_brotli_bytes(out_bin)
    pack_sha = file_sha256(out_bin)
    if br_bytes > MAX_BROTLI_BYTES:
        raise RuntimeError(f"KSFO terrain pack Brotli size {br_bytes} > {MAX_BROTLI_BYTES} bytes")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_doc = {
        "schema": "xvs.terrain.ksfo_3dep_manifest/1",
        "pack_filename": out_bin.name,
        "pack_sha256": pack_sha,
        "raw_bytes": raw_bytes,
        "brotli_bytes": br_bytes,
        "brotli_budget_bytes": MAX_BROTLI_BYTES,
        "terrain_mode": header_doc["terrain_mode"],
        "geoid_model": header_doc["geoid_model"],
        "datum": header_doc["datum"],
        "ksfo_28l_thr_navd88_m": header_doc["ksfo_28l_thr_navd88_m"],
        "ksfo_28l_geoid_undulation_m": header_doc["ksfo_28l_geoid_undulation_m"],
        "ksfo_28l_thr_wgs84_h_m": header_doc["ksfo_28l_thr_wgs84_h_m"],
        "rings": header_doc["rings"],
        "runway_ids": [r["id"] for r in header_doc["runways"]],
    }
    out_manifest.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")

    out_sources.parent.mkdir(parents=True, exist_ok=True)
    sources_doc = {
        "schema": "xvs.v2_m1.ksfo_3dep_data_sources/1",
        "allowlist_version": str(clf.policy["version"]),
        "terrain_mode": header_doc["terrain_mode"],
        "pack_sha256": pack_sha,
        "sources": sources_log,
    }
    out_sources.write_text(json.dumps(sources_doc, indent=2) + "\n", encoding="utf-8")
    return manifest_doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KSFO USGS 3DEP + GEOID18 terrain clipmap pack")
    parser.add_argument(
        "--terrain-mode",
        choices=("usgs-3dep", "procedural-ci"),
        default="usgs-3dep",
        help="Terrain data source mode (default: usgs-3dep; procedural-ci is explicit CI-only)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_BIN)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--sources-out", type=Path, default=DEFAULT_OUT_SOURCES)
    parser.add_argument("--grid-size", type=int, default=256)
    args = parser.parse_args()

    if args.terrain_mode == "procedural-ci":
        hdr = build_procedural_ci_pack(args.out, grid_size=args.grid_size)
        br_bytes = measure_brotli_bytes(args.out)
        print(
            f"Built procedural-ci terrain pack: {args.out} "
            f"(raw={args.out.stat().st_size} B, brotli={br_bytes} B, mode={hdr['terrain_mode']})"
        )
        return 0

    manifest = build_usgs_3dep_geoid18_pack(
        out_bin=args.out,
        out_manifest=args.manifest_out,
        out_sources=args.sources_out,
        grid_size=args.grid_size,
    )
    print(
        f"Built USGS 3DEP + GEOID18 KSFO pack: {args.out} "
        f"(raw={manifest['raw_bytes']} B, brotli={manifest['brotli_bytes']} B <= {MAX_BROTLI_BYTES} B, "
        f"KSFO 28L WGS-84 h={manifest['ksfo_28l_thr_wgs84_h_m']:.3f} m)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
