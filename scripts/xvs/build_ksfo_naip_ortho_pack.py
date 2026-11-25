"""Build real public-domain USGS National Map Orthoimagery textures for KSFO Live Deck.

Reads the exact `outer_sf_bay` (40x40 km) and `inner_ksfo` (10x10 km) WGS84
bounding boxes from `apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.manifest.json`,
fetches real USGS The National Map: Orthoimagery Web Mercator tiles via the
strict fail-closed `xvs.sim.allowlist.URLClassifier`, reprojects them onto
equirectangular (lat, lon) grids matching `pass2_terrain_cdlod.ts` UV space:
  - u = 0..1 -> lon_min..lon_max (West -> East, col = 0..W-1)
  - v = 0..1 -> lat_max..lat_min (North -> South, row = 0..H-1, top row y=0 is North)

Outputs:
  - apps/livedeck/public/assets/ksfo_ortho_inner_v1.jpg (1024x1024, <= 350 KB)
  - apps/livedeck/public/assets/ksfo_ortho_outer_v1.jpg (1024x1024, <= 250 KB)
  - apps/livedeck/public/assets/ksfo_ortho_v1.manifest.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xvs.sim.allowlist import URLClassifier

USGS_ORTHO_TILE_TEMPLATE = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
)
USER_AGENT = "OpenXVS-research/0.1"
TILE_SIZE_PX = 256
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
DEFAULT_TERRAIN_MANIFEST = (
    REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.manifest.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "assets"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "usgs_ortho"

# KSFO 28L threshold & origin references (WGS84, matches ksfo_3dep_geoid18_v1)
WGS84_A_M = 6378137.0
KSFO_ORIGIN_LAT = 37.6188056
KSFO_ORIGIN_LON = -122.3754167
KSFO_28L_LAT = 37.61356
KSFO_28L_LON = -122.35714

# 2048x2048 KSFO 2x2 km core ortho (~0.98 m/texel) centered across 28L/28R thresholds & touchdown zone
CORE2K_CENTER_LAT = 37.6155
CORE2K_CENTER_LON = -122.3625
CORE2K_EXTENT_KM = 2.0
CORE2K_SIZE_PX = 2048
CORE2K_ZOOM = 16
CORE2K_MAX_BYTES = 1_150_000
DEFAULT_CORE2K_OUT = DEFAULT_OUT_DIR / "ksfo_ortho_core2k_v2.jpg"


@dataclass(frozen=True)
class RingOrthoSpec:
    name: str
    filename: str
    zoom: int
    initial_jpeg_quality: int
    max_bytes: int
    extent_km: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def lon_to_tile_x_float(lon_deg: float | np.ndarray, zoom: int) -> float | np.ndarray:
    """Convert WGS84 longitude (degrees) to fractional Web Mercator tile X."""
    scale = float(1 << int(zoom))
    return (lon_deg + 180.0) / 360.0 * scale


def lat_to_tile_y_float(lat_deg: float | np.ndarray, zoom: int) -> float | np.ndarray:
    """Convert WGS84 latitude (degrees) to fractional Web Mercator tile Y (North=0)."""
    scale = float(1 << int(zoom))
    lat_rad = np.radians(lat_deg)
    merc = np.arcsinh(np.tan(lat_rad)) / math.pi
    return (1.0 - merc) * 0.5 * scale


def equirect_pixel_to_latlon(
    row: float,
    col: float,
    *,
    width: int,
    height: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[float, float]:
    """Map equirectangular output pixel (row, col) to WGS84 (lat, lon).

    Exact match to `pass2_terrain_cdlod.ts`:
      - row = 0 (v = 0) -> lat_max (North)
      - row = height - 1 (v = 1) -> lat_min (South)
      - col = 0 (u = 0) -> lon_min (West)
      - col = width - 1 (u = 1) -> lon_max (East)
    """
    u = float(col) / float(width - 1)
    v = float(row) / float(height - 1)
    lat = lat_max - v * (lat_max - lat_min)
    lon = lon_min + u * (lon_max - lon_min)
    return lat, lon


def latlon_to_equirect_uv(
    lat: float,
    lon: float,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[float, float]:
    """Map WGS84 (lat, lon) to normalized UV (u W->E, v N->S)."""
    u = (lon - lon_min) / (lon_max - lon_min)
    v = (lat_max - lat) / (lat_max - lat_min)
    return u, v


def build_usgs_tile_url(zoom: int, tile_y: int, tile_x: int) -> str:
    """Format USGS National Map Orthoimagery WMTS/REST tile URL."""
    return USGS_ORTHO_TILE_TEMPLATE.format(z=int(zoom), y=int(tile_y), x=int(tile_x))


def enforce_allowlist(url: str, classifier: URLClassifier) -> str:
    """Validate URL against fail-closed `URLClassifier`. Raises PermissionError if blocked."""
    verdict, reason, _entry = classifier.classify_upstream_url(url)
    if verdict != "ALLOWED":
        raise PermissionError(f"Blocked upstream URL ({verdict}: {reason}): {url}")
    return verdict


def tile_bounds_for_ring(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int,
) -> tuple[int, int, int, int]:
    """Return inclusive `(x_min, x_max, y_min, y_max)` slippy tile indices."""
    x_min = int(math.floor(float(lon_to_tile_x_float(lon_min, zoom))))
    x_max = int(math.floor(float(lon_to_tile_x_float(lon_max, zoom))))
    # North (lat_max) has smaller Web Mercator Y than South (lat_min).
    y_min = int(math.floor(float(lat_to_tile_y_float(lat_max, zoom))))
    y_max = int(math.floor(float(lat_to_tile_y_float(lat_min, zoom))))
    return x_min, x_max, y_min, y_max


def fetch_tile_rgb(
    *,
    zoom: int,
    tile_y: int,
    tile_x: int,
    cache_dir: Path,
    classifier: URLClassifier,
    max_retries: int = 4,
    timeout_s: float = 20.0,
) -> tuple[np.ndarray, bool]:
    """Fetch a single 256x256 RGB tile with allowlist enforcement, disk cache, and backoff.

    Returns `(rgb_uint8[256, 256, 3], was_cache_hit)`.
    """
    url = build_usgs_tile_url(zoom, tile_y, tile_x)
    enforce_allowlist(url, classifier)

    cache_path = cache_dir / f"z{zoom}" / f"{tile_y}_{tile_x}.jpg"
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        with Image.open(cache_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if rgb.shape == (TILE_SIZE_PX, TILE_SIZE_PX, 3):
            return rgb, True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "image/jpeg,image/png,image/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = int(resp.status)
                payload = resp.read()
            if status != 200 or len(payload) == 0:
                raise RuntimeError(f"HTTP {status} or empty payload for {url}")
            with Image.open(io.BytesIO(payload)) as im:
                rgb_im = im.convert("RGB")
                rgb = np.asarray(rgb_im, dtype=np.uint8)
            if rgb.shape != (TILE_SIZE_PX, TILE_SIZE_PX, 3):
                rgb = cv2.resize(
                    rgb,
                    (TILE_SIZE_PX, TILE_SIZE_PX),
                    interpolation=cv2.INTER_AREA,
                )
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(cache_path)
            return rgb, False
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                backoff_s = 0.75 * (2.0**attempt)
                time.sleep(backoff_s)

    raise RuntimeError(f"Failed to fetch USGS tile z={zoom} y={tile_y} x={tile_x} ({url}): {last_err}")


def reproject_mercator_mosaic_to_equirect(
    mosaic_rgb: np.ndarray,
    *,
    x_min: int,
    y_min: int,
    zoom: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    out_width: int = OUTPUT_WIDTH,
    out_height: int = OUTPUT_HEIGHT,
) -> np.ndarray:
    """Reproject a stitched Web Mercator tile mosaic onto exact equirectangular UV space.

    Row 0 corresponds to `lat_max` (North, v=0); row `out_height-1` corresponds to
    `lat_min` (South, v=1). Col 0 corresponds to `lon_min` (West, u=0); col
    `out_width-1` corresponds to `lon_max` (East, u=1).
    """
    rows = np.linspace(lat_max, lat_min, out_height, dtype=np.float64)
    cols = np.linspace(lon_min, lon_max, out_width, dtype=np.float64)

    x_f = lon_to_tile_x_float(cols, zoom)  # [W]
    y_f = lat_to_tile_y_float(rows, zoom)  # [H]

    # Pixel center of tile (x, y) local pixel p in 0..255 sits at (p + 0.5) / 256.
    px = (x_f - float(x_min)) * float(TILE_SIZE_PX) - 0.5
    py = (y_f - float(y_min)) * float(TILE_SIZE_PX) - 0.5

    map_x = np.broadcast_to(px[np.newaxis, :], (out_height, out_width)).astype(np.float32)
    map_y = np.broadcast_to(py[:, np.newaxis], (out_height, out_width)).astype(np.float32)

    remapped = cv2.remap(
        mosaic_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return remapped


def encode_jpeg_within_budget(
    rgb: np.ndarray,
    *,
    initial_quality: int,
    max_bytes: int,
    min_quality: int = 58,
) -> tuple[bytes, int]:
    """Encode RGB array to progressive optimized JPEG within `max_bytes`."""
    pil_img = Image.fromarray(rgb, mode="RGB")
    quality = int(initial_quality)
    while quality >= min_quality:
        buf = io.BytesIO()
        pil_img.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling="4:2:0",
        )
        payload = buf.getvalue()
        if len(payload) <= max_bytes:
            return payload, quality
        quality -= 2
    raise ValueError(
        f"Could not encode JPEG within {max_bytes} bytes even at quality={min_quality} "
        f"(last size={len(payload)} bytes)"
    )


def load_ring_specs_from_terrain_manifest(manifest_path: Path) -> list[RingOrthoSpec]:
    """Read exact `outer_sf_bay` and `inner_ksfo` ring bounds from 3DEP manifest."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    rings_by_name: dict[str, dict[str, object]] = {}
    for ring in raw["rings"]:
        rings_by_name[str(ring["name"])] = ring

    for required in ("inner_ksfo", "outer_sf_bay"):
        if required not in rings_by_name:
            raise KeyError(f"Missing ring '{required}' in terrain manifest {manifest_path}")

    inner = rings_by_name["inner_ksfo"]
    outer = rings_by_name["outer_sf_bay"]
    return [
        RingOrthoSpec(
            name="inner_ksfo",
            filename="ksfo_ortho_inner_v1.jpg",
            zoom=13,
            initial_jpeg_quality=86,
            max_bytes=350_000,
            extent_km=float(inner["extent_km"]),
            lat_min=float(inner["lat_min"]),
            lat_max=float(inner["lat_max"]),
            lon_min=float(inner["lon_min"]),
            lon_max=float(inner["lon_max"]),
        ),
        RingOrthoSpec(
            name="outer_sf_bay",
            filename="ksfo_ortho_outer_v1.jpg",
            zoom=11,
            initial_jpeg_quality=84,
            max_bytes=250_000,
            extent_km=float(outer["extent_km"]),
            lat_min=float(outer["lat_min"]),
            lat_max=float(outer["lat_max"]),
            lon_min=float(outer["lon_min"]),
            lon_max=float(outer["lon_max"]),
        ),
    ]


def fetch_reprojected_ring_rgb(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int,
    cache_dir: Path,
    classifier: URLClassifier,
    max_workers: int = 4,
    out_width: int = OUTPUT_WIDTH,
    out_height: int = OUTPUT_HEIGHT,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fetch, stitch, and reproject USGS tiles at `zoom` onto the equirectangular ring."""
    x_min, x_max, y_min, y_max = tile_bounds_for_ring(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        zoom=zoom,
    )
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    tile_coords = [(ty, tx) for ty in range(y_min, y_max + 1) for tx in range(x_min, x_max + 1)]

    mosaic = np.zeros((rows * TILE_SIZE_PX, cols * TILE_SIZE_PX, 3), dtype=np.uint8)
    cache_hits = 0

    workers = max(1, min(4, int(max_workers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                fetch_tile_rgb,
                zoom=zoom,
                tile_y=ty,
                tile_x=tx,
                cache_dir=cache_dir,
                classifier=classifier,
            ): (ty, tx)
            for (ty, tx) in tile_coords
        }
        for fut in as_completed(future_map):
            ty, tx = future_map[fut]
            tile_rgb, was_hit = fut.result()
            if was_hit:
                cache_hits += 1
            oy = (ty - y_min) * TILE_SIZE_PX
            ox = (tx - x_min) * TILE_SIZE_PX
            mosaic[oy : oy + TILE_SIZE_PX, ox : ox + TILE_SIZE_PX, :] = tile_rgb

    equirect_rgb = reproject_mercator_mosaic_to_equirect(
        mosaic,
        x_min=x_min,
        y_min=y_min,
        zoom=zoom,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        out_width=out_width,
        out_height=out_height,
    )
    grid_meta = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "tile_count": len(tile_coords),
        "cache_hits": cache_hits,
    }
    return equirect_rgb, grid_meta


def detect_nodata_black_mask(rgb: np.ndarray, *, min_area_px: int = 256) -> np.ndarray:
    """Detect solid-black USGS open-water no-data cutouts (`max(R,G,B) <= 6`, area >= min_area_px)."""
    dark = (rgb.max(axis=2) <= 6).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    mask = np.zeros(dark.shape, dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            mask[labels == label_id] = 255
    return mask


def composite_usgs_nodata_fallback(
    primary_rgb: np.ndarray,
    fallback_rgb: np.ndarray,
    nodata_mask_u8: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Composite lower-zoom USGS imagery into open-water no-data cutouts per component."""
    nodata_frac = float((nodata_mask_u8 > 0).mean())
    if nodata_frac <= 0.0:
        return primary_rgb, 0.0

    result = primary_rgb.astype(np.float32)
    fb_f32 = fallback_rgb.astype(np.float32)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

    num_labels, labels = cv2.connectedComponents(nodata_mask_u8, connectivity=8)
    for label_id in range(1, num_labels):
        comp_mask = ((labels == label_id).astype(np.uint8)) * 255
        dilated = cv2.dilate(comp_mask, dilate_kernel, iterations=1)
        border_ring = (cv2.dilate(dilated, ring_kernel, iterations=1) > 0) & (nodata_mask_u8 == 0) & (dilated == 0)

        adjusted = fb_f32.copy()
        if int(border_ring.sum()) >= 32:
            pri_mean = primary_rgb[border_ring].astype(np.float32).mean(axis=0)
            fb_mean = fb_f32[border_ring].mean(axis=0)
            adjusted = np.clip(
                adjusted + (pri_mean - fb_mean)[np.newaxis, np.newaxis, :],
                0.0,
                255.0,
            )

        alpha = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (25, 25), 7.0)
        alpha = np.maximum(alpha, (comp_mask > 0).astype(np.float32))
        alpha3 = alpha[:, :, np.newaxis]
        result = result * (1.0 - alpha3) + adjusted * alpha3

    return np.clip(np.rint(result), 0, 255).astype(np.uint8), nodata_frac


def build_ring_ortho(
    spec: RingOrthoSpec,
    *,
    out_dir: Path,
    cache_dir: Path,
    classifier: URLClassifier,
    max_workers: int = 4,
) -> dict[str, object]:
    """Fetch, stitch, reproject, encode, and validate one KSFO orthoimagery ring."""
    equirect_rgb, grid_meta = fetch_reprojected_ring_rgb(
        lat_min=spec.lat_min,
        lat_max=spec.lat_max,
        lon_min=spec.lon_min,
        lon_max=spec.lon_max,
        zoom=spec.zoom,
        cache_dir=cache_dir,
        classifier=classifier,
        max_workers=max_workers,
    )

    nodata_mask = detect_nodata_black_mask(equirect_rgb)
    nodata_filled_frac = 0.0
    fallback_zoom: int | None = None
    if int((nodata_mask > 0).sum()) > 0:
        fallback_zoom = min(8, spec.zoom - 1)
        fallback_rgb, _fb_meta = fetch_reprojected_ring_rgb(
            lat_min=spec.lat_min,
            lat_max=spec.lat_max,
            lon_min=spec.lon_min,
            lon_max=spec.lon_max,
            zoom=fallback_zoom,
            cache_dir=cache_dir,
            classifier=classifier,
            max_workers=max_workers,
        )
        equirect_rgb, nodata_filled_frac = composite_usgs_nodata_fallback(
            equirect_rgb, fallback_rgb, nodata_mask
        )

    rgb_mean = [round(float(v), 3) for v in equirect_rgb.mean(axis=(0, 1))]
    rgb_std = [round(float(v), 3) for v in equirect_rgb.std(axis=(0, 1))]
    if max(rgb_std) < 8.0:
        raise RuntimeError(
            f"Ring '{spec.name}' appears blank or nearly uniform (rgb_std={rgb_std})"
        )

    jpeg_bytes, used_quality = encode_jpeg_within_budget(
        equirect_rgb,
        initial_quality=spec.initial_jpeg_quality,
        max_bytes=spec.max_bytes,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec.filename
    out_path.write_bytes(jpeg_bytes)
    sha256_hex = hashlib.sha256(jpeg_bytes).hexdigest()

    u_28l, v_28l = latlon_to_equirect_uv(
        KSFO_28L_LAT,
        KSFO_28L_LON,
        lat_min=spec.lat_min,
        lat_max=spec.lat_max,
        lon_min=spec.lon_min,
        lon_max=spec.lon_max,
    )

    return {
        "name": spec.name,
        "filename": spec.filename,
        "sha256": sha256_hex,
        "bytes": len(jpeg_bytes),
        "byte_budget": spec.max_bytes,
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
        "zoom": spec.zoom,
        "nodata_fallback_zoom": fallback_zoom,
        "nodata_filled_fraction": round(nodata_filled_frac, 6),
        "jpeg_quality": used_quality,
        "extent_km": spec.extent_km,
        "lat_min": spec.lat_min,
        "lat_max": spec.lat_max,
        "lon_min": spec.lon_min,
        "lon_max": spec.lon_max,
        "tile_grid": grid_meta,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "ksfo_28l_thr_uv": {
            "u": round(u_28l, 6),
            "v": round(v_28l, 6),
            "px_x": round(u_28l * (OUTPUT_WIDTH - 1), 2),
            "px_y": round(v_28l * (OUTPUT_HEIGHT - 1), 2),
        },
    }


def build_ksfo_ortho_pack(
    *,
    terrain_manifest_path: Path = DEFAULT_TERRAIN_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_workers: int = 4,
) -> dict[str, object]:
    """Build both KSFO USGS orthoimagery rings and write `ksfo_ortho_v1.manifest.json`."""
    classifier = URLClassifier()
    specs = load_ring_specs_from_terrain_manifest(terrain_manifest_path)

    ring_entries: list[dict[str, object]] = []
    for spec in specs:
        entry = build_ring_ortho(
            spec,
            out_dir=out_dir,
            cache_dir=cache_dir,
            classifier=classifier,
            max_workers=max_workers,
        )
        ring_entries.append(entry)

    terrain_raw = json.loads(terrain_manifest_path.read_text(encoding="utf-8"))
    manifest: dict[str, object] = {
        "schema": "xvs.terrain.ksfo_ortho_manifest/1",
        "source_service": "USGS The National Map: Orthoimagery (USGSImageryOnly)",
        "tile_url_template": USGS_ORTHO_TILE_TEMPLATE,
        "license": "U.S. Public Domain",
        "attribution": "USDA, USGS The National Map: Orthoimagery",
        "aligned_terrain_manifest": terrain_manifest_path.name,
        "aligned_terrain_pack_sha256": str(terrain_raw["pack_sha256"]),
        "projection": "WGS84_EQUIRECTANGULAR_FROM_EPSG3857",
        "uv_orientation": {
            "u": "0..1 maps lon_min..lon_max (West -> East, col=0..W-1)",
            "v": "0..1 maps lat_max..lat_min (North -> South, row=0..H-1, top row y=0 is North)",
            "matches_shader": "apps/livedeck/src/passes/pass2_terrain_cdlod.ts",
        },
        "rings": ring_entries,
    }

    manifest_path = out_dir / "ksfo_ortho_v1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def compute_core2k_bounds(
    *,
    center_lat: float = CORE2K_CENTER_LAT,
    center_lon: float = CORE2K_CENTER_LON,
    extent_km: float = CORE2K_EXTENT_KM,
) -> tuple[float, float, float, float]:
    """Compute exact WGS84 (lat_min, lat_max, lon_min, lon_max) for the 2x2 km KSFO core."""
    half_m = extent_km * 500.0
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(KSFO_ORIGIN_LAT))
    d_lat = half_m / m_per_deg_lat
    d_lon = half_m / m_per_deg_lon
    return (
        center_lat - d_lat,
        center_lat + d_lat,
        center_lon - d_lon,
        center_lon + d_lon,
    )


def render_procedural_ksfo_core2k_rgb(
    *,
    width: int = CORE2K_SIZE_PX,
    height: int = CORE2K_SIZE_PX,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Deterministic high-res procedural KSFO 28L/28R + SF Bay fallback (zero network)."""
    half_m = CORE2K_EXTENT_KM * 500.0
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(KSFO_ORIGIN_LAT))

    # Build local North/East meter coordinates relative to KSFO 28L threshold
    u_lin = np.linspace(0.0, 1.0, width, dtype=np.float32)
    v_lin = np.linspace(0.0, 1.0, height, dtype=np.float32)
    lon_grid = lon_min + u_lin * (lon_max - lon_min)
    lat_grid = lat_max - v_lin * (lat_max - lat_min)
    east_m = ((lon_grid - KSFO_28L_LON) * m_per_deg_lon)[np.newaxis, :]
    north_m = ((lat_grid - KSFO_28L_LAT) * m_per_deg_lat)[:, np.newaxis]

    # Runway 28L frame: true heading 298.0 deg (forward along runway = (-sin62, -cos62) -> N=+0.4695, E=-0.8829)
    hdg_rad = math.radians(298.0)
    fwd_n = math.cos(hdg_rad)
    fwd_e = math.sin(hdg_rad)
    right_n = -fwd_e
    right_e = fwd_n

    along_28l = north_m * fwd_n + east_m * fwd_e
    cross_28l = north_m * right_n + east_m * right_e

    # Base San Francisco Bay water with subtle coastal depth gradient
    bay_t = np.clip((along_28l + 600.0) / 1400.0, 0.0, 1.0)
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = 24.0 + 10.0 * bay_t
    img[:, :, 1] = 48.0 + 14.0 * bay_t
    img[:, :, 2] = 66.0 + 16.0 * bay_t

    # Airfield reclaimed peninsula fill & grass infield (along >= -55 m, wide west of seawall)
    seawall_along = -55.0 + 18.0 * np.sin(cross_28l * 0.008)
    land_mask = (along_28l >= seawall_along) & (cross_28l > -420.0) & (cross_28l < 620.0)
    seawall_mask = (np.abs(along_28l - seawall_along) < 14.0) & (cross_28l > -430.0) & (cross_28l < 630.0)

    # Dry coastal California airfield turf / soil
    turf_var = 6.0 * np.sin(north_m * 0.045) * np.cos(east_m * 0.038)
    img[land_mask, 0] = 112.0 + turf_var[land_mask]
    img[land_mask, 1] = 116.0 + turf_var[land_mask]
    img[land_mask, 2] = 92.0 + 0.5 * turf_var[land_mask]

    # Riprap seawall embankment
    img[seawall_mask, 0] = 136.0
    img[seawall_mask, 1] = 132.0
    img[seawall_mask, 2] = 122.0

    # Parallel taxiways & apron
    taxi_mask = (
        (along_28l >= 40.0)
        & (along_28l <= 1900.0)
        & ((np.abs(cross_28l + 115.0) < 15.0) | (np.abs(cross_28l - 345.0) < 15.0))
    )
    img[taxi_mask, 0] = 96.0
    img[taxi_mask, 1] = 98.0
    img[taxi_mask, 2] = 100.0

    # Terminal apron concrete SW of 28L
    apron_mask = (along_28l >= 450.0) & (along_28l <= 1650.0) & (cross_28l < -150.0) & (cross_28l > -410.0)
    img[apron_mask, 0] = 154.0
    img[apron_mask, 1] = 152.0
    img[apron_mask, 2] = 146.0

    # Runways 28L (cross=0) and 28R (cross=+228.6 m = 750 ft parallel separation)
    for rwy_cross in (0.0, 228.6):
        c_rel = cross_28l - rwy_cross
        rwy_body = (along_28l >= 0.0) & (along_28l <= 2400.0) & (np.abs(c_rel) <= 30.5)
        # Asphalt runway surface with rubber touchdown darkening
        tdz_darken = 14.0 * np.exp(-(((along_28l - 360.0) / 220.0) ** 2)) * np.exp(-((c_rel / 12.0) ** 2))
        img[rwy_body, 0] = 66.0 - tdz_darken[rwy_body]
        img[rwy_body, 1] = 68.0 - tdz_darken[rwy_body]
        img[rwy_body, 2] = 72.0 - tdz_darken[rwy_body]

        # White threshold piano keys (along 8..52 m)
        piano = (
            (along_28l >= 8.0)
            & (along_28l <= 52.0)
            & (np.abs(c_rel) <= 26.0)
            & (np.abs(c_rel) >= 2.5)
            & ((np.floor((c_rel + 28.0) / 3.2) % 2) == 0)
        )
        # Aiming point blocks (along 300..345 m)
        aiming = (
            (along_28l >= 300.0)
            & (along_28l <= 345.0)
            & (np.abs(c_rel) >= 9.0)
            & (np.abs(c_rel) <= 18.0)
        )
        # Centerline dashes (36 m stripe, 24 m gap)
        centerline = (
            (along_28l >= 80.0)
            & (along_28l <= 2350.0)
            & (np.abs(c_rel) <= 1.2)
            & ((along_28l % 60.0) < 36.0)
        )
        # Edge stripes
        edges = (
            (along_28l >= 4.0)
            & (along_28l <= 2380.0)
            & (np.abs(np.abs(c_rel) - 28.5) <= 1.1)
        )
        paint = piano | aiming | centerline | edges
        img[paint, 0] = 232.0
        img[paint, 1] = 234.0
        img[paint, 2] = 230.0

    # Crossing Runways 1L/1R (~028 deg true, crossing near along=1050 m)
    hdg_1r = math.radians(28.0)
    a_1r = (north_m - 120.0) * math.cos(hdg_1r) + (east_m + 960.0) * math.sin(hdg_1r)
    c_1r = -(north_m - 120.0) * math.sin(hdg_1r) + (east_m + 960.0) * math.cos(hdg_1r)
    for cross_off in (0.0, 228.6):
        c_rel = c_1r - cross_off
        rwy_cross_mask = (np.abs(a_1r) <= 1200.0) & (np.abs(c_rel) <= 28.0) & land_mask
        img[rwy_cross_mask, 0] = 70.0
        img[rwy_cross_mask, 1] = 72.0
        img[rwy_cross_mask, 2] = 76.0

    # Deterministic micro-grain so 0.98 m/texel surface reads crisply
    rng = np.random.default_rng(28028)
    grain = rng.normal(0.0, 2.8, size=(height, width, 1)).astype(np.float32)
    img = np.clip(img + grain, 0.0, 255.0)
    _ = half_m  # kept explicit for readability
    return np.rint(img).astype(np.uint8)


def build_ksfo_core2k_ortho(
    out_path: Path = DEFAULT_CORE2K_OUT,
    *,
    offline_procedural: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_workers: int = 4,
) -> dict[str, object]:
    """Build 2048x2048 KSFO 2x2 km core ortho (~0.98 m/texel) with offline fallback.

    Attempts public-domain USGS National Map `USGSImageryOnly` zoom-16 tiles via
    `URLClassifier` when `offline_procedural=False`, degrading cleanly to a
    deterministic procedural KSFO airfield render if offline or unreachable.
    """
    lat_min, lat_max, lon_min, lon_max = compute_core2k_bounds()
    provenance = "procedural_fallback"
    grid_meta: dict[str, int] | None = None
    equirect_rgb: np.ndarray | None = None

    if not offline_procedural:
        classifier = URLClassifier()
        try:
            live_rgb, live_meta = fetch_reprojected_ring_rgb(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                zoom=CORE2K_ZOOM,
                cache_dir=cache_dir,
                classifier=classifier,
                max_workers=max_workers,
                out_width=CORE2K_SIZE_PX,
                out_height=CORE2K_SIZE_PX,
            )
            nodata_mask = detect_nodata_black_mask(live_rgb)
            if int((nodata_mask > 0).sum()) > 0:
                fallback_rgb, _fb_meta = fetch_reprojected_ring_rgb(
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    zoom=12,
                    cache_dir=cache_dir,
                    classifier=classifier,
                    max_workers=max_workers,
                    out_width=CORE2K_SIZE_PX,
                    out_height=CORE2K_SIZE_PX,
                )
                live_rgb, _filled = composite_usgs_nodata_fallback(
                    live_rgb, fallback_rgb, nodata_mask
                )
            rgb_std_live = [float(v) for v in live_rgb.std(axis=(0, 1))]
            if max(rgb_std_live) >= 8.0:
                equirect_rgb = live_rgb
                grid_meta = live_meta
                provenance = "usgs_naip_zoom16"
        except (OSError, RuntimeError, urllib.error.URLError, ValueError):
            equirect_rgb = None
            provenance = "procedural_fallback"

    if equirect_rgb is None:
        equirect_rgb = render_procedural_ksfo_core2k_rgb(
            width=CORE2K_SIZE_PX,
            height=CORE2K_SIZE_PX,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        provenance = "procedural_fallback"

    jpeg_bytes, used_quality = encode_jpeg_within_budget(
        equirect_rgb,
        initial_quality=84,
        max_bytes=CORE2K_MAX_BYTES,
        min_quality=50,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(jpeg_bytes)
    sha256_hex = hashlib.sha256(jpeg_bytes).hexdigest()

    rgb_mean = [round(float(v), 3) for v in equirect_rgb.mean(axis=(0, 1))]
    rgb_std = [round(float(v), 3) for v in equirect_rgb.std(axis=(0, 1))]
    u_28l, v_28l = latlon_to_equirect_uv(
        KSFO_28L_LAT,
        KSFO_28L_LON,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    return {
        "schema": "xvs.terrain.ksfo_core2k_ortho/2",
        "provenance": provenance,
        "filename": out_path.name,
        "sha256": sha256_hex,
        "bytes": len(jpeg_bytes),
        "byte_budget": CORE2K_MAX_BYTES,
        "width": CORE2K_SIZE_PX,
        "height": CORE2K_SIZE_PX,
        "jpeg_quality": used_quality,
        "extent_km": CORE2K_EXTENT_KM,
        "meters_per_texel": round((CORE2K_EXTENT_KM * 1000.0) / float(CORE2K_SIZE_PX), 4),
        "center_lat": CORE2K_CENTER_LAT,
        "center_lon": CORE2K_CENTER_LON,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "tile_grid": grid_meta,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "ksfo_28l_thr_uv": {
            "u": round(u_28l, 6),
            "v": round(v_28l, 6),
            "px_x": round(u_28l * (CORE2K_SIZE_PX - 1), 2),
            "px_y": round(v_28l * (CORE2K_SIZE_PX - 1), 2),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KSFO public-domain USGS National Map Orthoimagery textures."
    )
    parser.add_argument(
        "--terrain-manifest",
        type=Path,
        default=DEFAULT_TERRAIN_MANIFEST,
        help="Path to ksfo_3dep_geoid18_v1.manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for JPEGs and manifest",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Tile cache directory (gitignored)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent tile downloads (capped at 4)",
    )
    parser.add_argument(
        "--core2k",
        action="store_true",
        help="Build the 2048x2048 KSFO 2x2 km core orthoimagery texture (ksfo_ortho_core2k_v2.jpg)",
    )
    parser.add_argument(
        "--core2k-out",
        type=Path,
        default=DEFAULT_CORE2K_OUT,
        help="Output path for --core2k JPEG",
    )
    parser.add_argument(
        "--offline-procedural",
        action="store_true",
        help="Use deterministic procedural KSFO fallback for --core2k without network requests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.core2k:
        meta = build_ksfo_core2k_ortho(
            out_path=args.core2k_out,
            offline_procedural=args.offline_procedural,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
        )
        print(json.dumps(meta, indent=2))
        return 0
    manifest = build_ksfo_ortho_pack(
        terrain_manifest_path=args.terrain_manifest,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
