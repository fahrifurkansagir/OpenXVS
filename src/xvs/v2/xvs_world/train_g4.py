"""Production Colab G4 (`96 GB VRAM` Blackwell) & `--smoke-local` Trainer for `XVS-World` 1x1 km 3DGS (`G4R-4`).

Strictly enforces `SP2B_G4_TRAINERS_DESIGN.md` §3.3 (`BC-G4-B1`, `BC-G4-m1`, `BC-G4-m4`):
- Hard-gated by `WorldTwinConfig` (`extent <= 1000x1000 m`, `max_splats <= 15_000_000`,
  `spz_max_gb <= 0.35`, `backend == "gsplat_3dgut_2dgs"`, `gt_tier == "VISUAL_STRESS_ONLY"`).
- Loads real M1/WP-01b KSFO 3DEP heightfield via `xvs.v2.wp01b.mipmap.load_ksfo_3dep_mipmap`
  (`apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin` + `.manifest.json`), plus optional
  real USDA NAIP STAC orthoimagery on Colab G4 (lazy imports; procedural KSFO 28L albedo
  fallback in `--smoke-local`).
- Strictly `CONVENTIONS.md` v1.0 `NED/CAM` camera frustums (`+N, +E, +D`; OpenCV `+X_right,
  +Y_down, +Z_forward`; Hamilton `(w, x, y, z)` with `w >= 0`; zero internal ENU).
- Dual backend: real Blackwell `sm_120` `gsplat.rasterization(..., with_ut=True, with_eval3d=True)`
  on Colab G4 CUDA, and a fast differentiable pure-PyTorch Gaussian renderer on local CPU.
- Genuine Niantic SPZ v2 binary encoder (`magic = 0x5053474E` `"NGSP"`, `version = 2`, gzip-
  compressed quantized 24-bit fixed-point positions + alphas/colors/scales/rotations) with
  `read_spz_v2_header()` round-trip verification.
- Renders 3 visual-stress RGB PNG previews (`preview_clear.png`, `preview_radiation_fog.png`,
  `preview_dusk_glare.png`), each stamped `VISUAL_STRESS_ONLY` (`wp01_certification_eligible=False`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from xvs.geo.wgs84 import WGS84_A
from xvs.v2.colab_g4_utils import (
    AutoVRAMBatchTuner,
    DriveCheckpointManager,
    collect_blackwell_env_report,
    restore_eager_state,
    snapshot_eager_state,
)
from xvs.v2.license_guard import assert_dataset_allowed
from xvs.v2.wp01b.mipmap import (
    DEFAULT_KSFO_PACK_PATH,
    MinMaxMipmapPyramid,
    load_ksfo_3dep_mipmap,
    sample_height_bilinear_m,
)
from xvs.v2.xvs_world.config import (
    REQUIRED_GT_TIER,
    WorldTwinConfig,
    ksfo_28l_default_config,
)
from xvs.v2.xvs_world.runner import validate_frame_metadata_visual_stress_only

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_KSFO_MANIFEST_PATH = (
    REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_3dep_geoid18_v1.manifest.json"
)

# Niantic SPZ v2 binary format constants (`BC-G4-m1`)
SPZ_MAGIC_U32: int = 0x5053474E  # Little-endian "NGSP"
SPZ_MAGIC_ASCII: str = "NGSP"
SPZ_VERSION: int = 2
SPZ_HEADER_STRUCT = struct.Struct("<IIIBBBB")
assert SPZ_HEADER_STRUCT.size == 16

# SH0 constant C0 = 1 / (2 * sqrt(pi))
_SH_C0: float = 0.28209479177387814


@dataclass(frozen=True)
class NedCameraView:
    """Calibrated OpenCV pinhole camera view strictly in local KSFO 28L `NED` coordinates."""

    view_id: int
    p_cam_ned: np.ndarray  # (3,) float32 [N, E, D] metres relative to 28L threshold
    r_cam_ned: np.ndarray  # (3, 3) float32 NED -> CAM rotation (det=+1)
    t_cam: np.ndarray  # (3,) float32 -R_cam_ned @ p_cam_ned
    k_mat: np.ndarray  # (3, 3) float32 pinhole intrinsics
    width: int
    height: int

    def viewmat_4x4(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = self.r_cam_ned
        m[:3, 3] = self.t_cam
        return m


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError(f"Degenerate zero/non-finite vector in camera construction: {v!r}")
    return v / n


def build_lookat_cam_ned(
    p_cam_ned: np.ndarray,
    p_target_ned: np.ndarray,
    *,
    up_hint_ned: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact OpenCV `R_cam_ned` (+X_right, +Y_down, +Z_forward) and `t_cam` in strict NED."""
    eye = np.asarray(p_cam_ned, dtype=np.float64).reshape(3)
    tgt = np.asarray(p_target_ned, dtype=np.float64).reshape(3)
    z_fwd = _normalize_vec(tgt - eye)
    # In NED, world "down" is [0, 0, +1] and world "up" is [0, 0, -1]
    down_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_hint_ned is None else -_normalize_vec(
        np.asarray(up_hint_ned, dtype=np.float64).reshape(3)
    )
    # Right = down_ref x z_fwd (check: for z_fwd=[1,0,0] North and down_ref=[0,0,1] Down,
    # down_ref x z_fwd = [0, +1, 0] East = Right! Right-handed: x_right x y_down = z_fwd!)
    x_right_raw = np.cross(down_ref, z_fwd)
    if float(np.linalg.norm(x_right_raw)) < 1e-6:
        x_right_raw = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), z_fwd)
    x_right = _normalize_vec(x_right_raw)
    y_down = _normalize_vec(np.cross(z_fwd, x_right))

    r_cam_ned = np.stack([x_right, y_down, z_fwd], axis=0)
    det = float(np.linalg.det(r_cam_ned))
    if abs(det - 1.0) > 1e-6:
        raise RuntimeError(f"Non-SO(3) R_cam_ned determinant {det}")
    t_cam = -(r_cam_ned @ eye)
    return r_cam_ned.astype(np.float32), t_cam.astype(np.float32)


def _fetch_optional_naip_albedo(
    cfg: WorldTwinConfig,
    grid_res: int,
) -> tuple[np.ndarray | None, str]:
    """Lazily query Microsoft Planetary Computer USDA NAIP STAC on Colab G4 (never required locally)."""
    assert_dataset_allowed("usda_naip", "train")
    try:
        import planetary_computer  # type: ignore[import-not-found]
        import pystac_client  # type: ignore[import-not-found]
        import rasterio  # type: ignore[import-not-found]
        from rasterio.warp import transform_bounds  # type: ignore[import-not-found]
        from rasterio.windows import from_bounds  # type: ignore[import-not-found]
    except ImportError:
        return None, "procedural_ksfo_fallback_no_stac_deps"

    try:
        half_lat = (0.5 * cfg.extent_y_m) / ((math.pi / 180.0) * WGS84_A)
        half_lon = (0.5 * cfg.extent_x_m) / (
            (math.pi / 180.0) * WGS84_A * math.cos(math.radians(cfg.anchor_lat_deg))
        )
        bbox = [
            cfg.anchor_lon_deg - half_lon,
            cfg.anchor_lat_deg - half_lat,
            cfg.anchor_lon_deg + half_lon,
            cfg.anchor_lat_deg + half_lat,
        ]
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(collections=["naip"], bbox=bbox, max_items=1)
        items = list(search.items())
        if not items:
            return None, "procedural_ksfo_fallback_no_naip_items"
        asset = items[0].assets["image"]
        with rasterio.open(asset.href) as src:
            if src.crs is not None:
                left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
            else:
                left, bottom, right, top = bbox[0], bbox[1], bbox[2], bbox[3]
            win = from_bounds(left, bottom, right, top, transform=src.transform)
            rgb = src.read(
                [1, 2, 3],
                window=win,
                out_shape=(3, grid_res, grid_res),
                boundless=True,
                fill_value=0,
            )
            albedo = np.clip(rgb.transpose(1, 2, 0).astype(np.float32) / 255.0, 0.0, 1.0)
            return albedo, f"usda_naip_stac:{items[0].id}"
    except Exception as exc:
        err_tag = f"{type(exc).__name__}:{str(exc)[:100]}"
        return None, f"procedural_ksfo_fallback_stac_error:{err_tag}"


def _load_bundled_ksfo_ortho_albedo(
    ksfo_bin_path: Path | str,
    *,
    thr_lat_deg: float,
    thr_lon_deg: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
    m_per_deg_lat: float,
    m_per_deg_lon: float,
    north_grid_m: np.ndarray,
    east_grid_m: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Sample verified public-domain USDA/USGS KSFO inner ortho JPEG relative to `ksfo_bin_path.parent`."""
    assets_dir = Path(ksfo_bin_path).resolve().parent
    ortho_path = assets_dir / "ksfo_ortho_inner_v1.jpg"
    if not ortho_path.is_file():
        return None, "procedural_ksfo_28l_albedo"

    # Default 10 km inner ring bounds aligned with `ksfo_3dep_geoid18_v1.bin`
    half_lat = 5000.0 / m_per_deg_lat
    half_lon = 5000.0 / m_per_deg_lon
    lat_min = origin_lat_deg - half_lat
    lat_max = origin_lat_deg + half_lat
    lon_min = origin_lon_deg - half_lon
    lon_max = origin_lon_deg + half_lon

    manifest_path = assets_dir / "ksfo_ortho_v1.manifest.json"
    if manifest_path.is_file():
        manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "rings" in manifest_obj and isinstance(manifest_obj["rings"], list):
            for ring in manifest_obj["rings"]:
                if isinstance(ring, dict) and "name" in ring and ring["name"] == "inner_ksfo":
                    lat_min = float(ring["lat_min"])
                    lat_max = float(ring["lat_max"])
                    lon_min = float(ring["lon_min"])
                    lon_max = float(ring["lon_max"])
                    break

    with Image.open(ortho_path) as im:
        img_rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    img_h, img_w, _ = img_rgb.shape
    lat_grid = thr_lat_deg + north_grid_m / m_per_deg_lat
    lon_grid = thr_lon_deg + east_grid_m / m_per_deg_lon

    u = np.clip((lon_grid - lon_min) / max(lon_max - lon_min, 1e-12), 0.0, 1.0)
    v = np.clip((lat_max - lat_grid) / max(lat_max - lat_min, 1e-12), 0.0, 1.0)

    x = u * float(img_w - 1)
    y = v * float(img_h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, img_w - 1)
    y1 = np.minimum(y0 + 1, img_h - 1)
    wx = (x - x0).astype(np.float32)[:, :, None]
    wy = (y - y0).astype(np.float32)[:, :, None]

    c00 = img_rgb[y0, x0]
    c10 = img_rgb[y0, x1]
    c01 = img_rgb[y1, x0]
    c11 = img_rgb[y1, x1]
    sampled = (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    )
    return np.clip(sampled, 0.0, 1.0).astype(np.float32), "bundled_usda_usgs_ksfo_ortho_v1"


def _build_procedural_ksfo_28l_albedo(
    north_grid_m: np.ndarray,
    east_grid_m: np.ndarray,
    down_grid_m: np.ndarray,
) -> np.ndarray:
    """Generate clean physical KSFO 28L runway + markings + SF Bay shoreline albedo in local NED."""
    h, w = north_grid_m.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Runway 28L heading ~298 deg true (unit vector pointing WNW from threshold)
    hdg_rad = math.radians(298.0)
    along_n, along_e = math.cos(hdg_rad), math.sin(hdg_rad)
    right_n, right_e = -along_e, along_n

    s_along = north_grid_m * along_n + east_grid_m * along_e  # +along runway from 28L threshold
    s_cross = north_grid_m * right_n + east_grid_m * right_e  # +right of centerline

    # Base SF Bay water (low elevation / approach end s_along < -40 m) vs airport land
    # Recall NED: more negative down_m = higher elevation; sea level near down_m ~ +2.0 m relative to threshold
    is_water = (s_along < -35.0) | (down_grid_m > 1.8)
    rgb[~is_water] = np.array([0.36, 0.39, 0.31], dtype=np.float32)  # coastal turf/apron
    rgb[is_water] = np.array([0.11, 0.22, 0.34], dtype=np.float32)  # San Francisco Bay water

    # Gentle spatial texture variation from real 3DEP elevation relief
    relief = np.clip(-down_grid_m / 25.0, -0.15, 0.15)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] + 0.25 * relief, 0.0, 1.0)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] + 0.20 * relief, 0.0, 1.0)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] + 0.10 * relief, 0.0, 1.0)

    # Runway 28L asphalt pavement (61 m wide: |s_cross| <= 30.5 m, s_along in [0, 500 m])
    on_rwy_28l = (s_along >= -10.0) & (s_along <= 500.0) & (np.abs(s_cross) <= 30.5)
    rgb[on_rwy_28l] = np.array([0.21, 0.22, 0.24], dtype=np.float32)

    # Parallel Runway 28R (228.6 m to the right, 61 m wide)
    on_rwy_28r = (s_along >= -10.0) & (s_along <= 500.0) & (np.abs(s_cross - 228.6) <= 30.5)
    rgb[on_rwy_28r] = np.array([0.22, 0.23, 0.25], dtype=np.float32)

    # Taxiway C (122 m left, 22.86 m wide)
    on_twy_c = (s_along >= 10.0) & (s_along <= 500.0) & (np.abs(s_cross + 122.0) <= 11.43)
    rgb[on_twy_c] = np.array([0.29, 0.29, 0.28], dtype=np.float32)

    # High-contrast white threshold piano keys + centerline dashes on 28L
    piano = (s_along >= 6.0) & (s_along <= 50.0) & (np.abs(s_cross) <= 24.0) & (
        ((np.abs(s_cross) % 6.0) < 3.0)
    )
    centerline = (s_along >= 60.0) & (s_along <= 490.0) & (np.abs(s_cross) <= 2.2) & (
        ((s_along % 40.0) < 22.0)
    )
    edge_lines = (s_along >= 0.0) & (s_along <= 500.0) & (np.abs(np.abs(s_cross) - 28.5) <= 1.8)
    rgb[piano | centerline | edge_lines] = np.array([0.92, 0.93, 0.94], dtype=np.float32)

    # Approach lighting bar (ALSF-2) over Bay causeway (s_along in [-280, -15])
    alsf = (s_along >= -280.0) & (s_along <= -15.0) & (np.abs(s_cross) <= 5.0) & (
        ((np.abs(s_along) % 35.0) < 10.0)
    )
    rgb[alsf] = np.array([0.96, 0.88, 0.62], dtype=np.float32)

    return np.clip(rgb, 0.0, 1.0)


def load_ksfo_3dep_patch_ned(
    cfg: WorldTwinConfig,
    *,
    ksfo_bin_path: Path | str = DEFAULT_KSFO_PACK_PATH,
    grid_res: int = 32,
    use_naip_stac: bool = False,
) -> dict[str, Any]:
    """Load real M1/WP-01b KSFO 3DEP heightfield via `load_ksfo_3dep_mipmap` into a 1x1 km NED patch."""
    assert_dataset_allowed("usgs_3dep", "train")
    assert_dataset_allowed("usda_naip", "train")

    pyramid: MinMaxMipmapPyramid = load_ksfo_3dep_mipmap(ksfo_bin_path, ring_name="inner_ksfo")
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(pyramid.origin_lat_deg))

    thr_lat = cfg.anchor_lat_deg if pyramid.ksfo_28l_thr_lat_deg is None else pyramid.ksfo_28l_thr_lat_deg
    thr_lon = cfg.anchor_lon_deg if pyramid.ksfo_28l_thr_lon_deg is None else pyramid.ksfo_28l_thr_lon_deg
    thr_h = cfg.anchor_h_wgs84_m if pyramid.ksfo_28l_thr_wgs84_h_m is None else pyramid.ksfo_28l_thr_wgs84_h_m

    thr_n_in_pyr = (thr_lat - pyramid.origin_lat_deg) * m_per_deg_lat
    thr_e_in_pyr = (thr_lon - pyramid.origin_lon_deg) * m_per_deg_lon

    half_n = 0.5 * cfg.extent_y_m
    half_e = 0.5 * cfg.extent_x_m
    ns = np.linspace(-half_n, half_n, grid_res, dtype=np.float64)
    es = np.linspace(-half_e, half_e, grid_res, dtype=np.float64)
    north_grid, east_grid = np.meshgrid(ns, es, indexing="ij")

    h_wgs84_grid = np.zeros((grid_res, grid_res), dtype=np.float64)
    for r in range(grid_res):
        for c in range(grid_res):
            h_wgs84_grid[r, c] = sample_height_bilinear_m(
                pyramid,
                thr_n_in_pyr + float(north_grid[r, c]),
                thr_e_in_pyr + float(east_grid[r, c]),
            )

    # Strict NED: Down = -(h_wgs84 - thr_h) so threshold sits at Down = 0.0 m
    down_grid = -(h_wgs84_grid - float(thr_h))

    albedo: np.ndarray | None = None
    albedo_source = "procedural_ksfo_28l_albedo"
    if use_naip_stac:
        albedo, albedo_source = _fetch_optional_naip_albedo(cfg, grid_res)
    if albedo is None:
        albedo, bundled_source = _load_bundled_ksfo_ortho_albedo(
            ksfo_bin_path,
            thr_lat_deg=float(thr_lat),
            thr_lon_deg=float(thr_lon),
            origin_lat_deg=float(pyramid.origin_lat_deg),
            origin_lon_deg=float(pyramid.origin_lon_deg),
            m_per_deg_lat=m_per_deg_lat,
            m_per_deg_lon=m_per_deg_lon,
            north_grid_m=north_grid,
            east_grid_m=east_grid,
        )
        if albedo is not None:
            albedo_source = bundled_source
    if albedo is None:
        albedo = _build_procedural_ksfo_28l_albedo(north_grid, east_grid, down_grid)

    return {
        "north_m": north_grid.astype(np.float32),
        "east_m": east_grid.astype(np.float32),
        "down_m": down_grid.astype(np.float32),
        "h_wgs84_m": h_wgs84_grid.astype(np.float32),
        "albedo_rgb": albedo.astype(np.float32),
        "albedo_source": albedo_source,
        "terrain_mode": pyramid.terrain_mode,
        "geoid_model": pyramid.geoid_model,
        "mipmap_levels": pyramid.num_levels,
        "ksfo_28l_thr_wgs84_h_m": float(thr_h),
        "ksfo_bin_sha256": _sha256_file(Path(ksfo_bin_path)),
    }


def build_ksfo_approach_cameras(
    *,
    num_views: int = 8,
    width: int = 64,
    height: int = 48,
    fov_x_deg: float = 60.0,
) -> list[NedCameraView]:
    """Generate calibrated multi-view approach camera frustums in strict NED/CAM around KSFO 28L."""
    if num_views < 2:
        raise ValueError(f"num_views must be >= 2, got {num_views}")
    fx = 0.5 * float(width) / math.tan(0.5 * math.radians(fov_x_deg))
    fy = fx
    cx = 0.5 * float(width) - 0.5
    cy = 0.5 * float(height) - 0.5
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    hdg_rad = math.radians(298.0)
    along_vec = np.array([math.cos(hdg_rad), math.sin(hdg_rad), 0.0], dtype=np.float64)
    right_vec = np.array([-math.sin(hdg_rad), math.cos(hdg_rad), 0.0], dtype=np.float64)

    views: list[NedCameraView] = []
    for i in range(num_views):
        frac = float(i) / float(max(1, num_views - 1))
        # Approach from 420 m short final to 160 m threshold crossing with gentle lateral parallax
        dist_back_m = 420.0 - 240.0 * frac
        lat_offset_m = 75.0 * math.sin(2.0 * math.pi * frac)
        alt_agl_m = 65.0 + 35.0 * (1.0 - frac) + 12.0 * math.cos(2.0 * math.pi * frac)
        # Strict NED: altitude above threshold means negative Down
        p_cam = (-dist_back_m) * along_vec + lat_offset_m * right_vec + np.array([0.0, 0.0, -alt_agl_m])
        p_tgt = (60.0 + 40.0 * frac) * along_vec + np.array([0.0, 0.0, 0.0])
        r_cam, t_cam = build_lookat_cam_ned(p_cam, p_tgt)
        views.append(
            NedCameraView(
                view_id=i,
                p_cam_ned=p_cam.astype(np.float32),
                r_cam_ned=r_cam,
                t_cam=t_cam,
                k_mat=k_mat.copy(),
                width=int(width),
                height=int(height),
            )
        )
    return views


class GaussianScene3D(nn.Module):
    """Trainable 3D Gaussian Splatting scene strictly in local NED coordinates."""

    def __init__(
        self,
        means_ned: torch.Tensor,
        albedo_rgb: torch.Tensor,
        *,
        init_scale_m: float = 18.0,
        color_perturbation_std: float = 0.0,
        seed: int = 20261121,
    ) -> None:
        super().__init__()
        n = int(means_ned.shape[0])
        if means_ned.ndim != 2 or means_ned.shape[1] != 3:
            raise ValueError(f"means_ned must have shape (N, 3), got {tuple(means_ned.shape)}")

        self.means = nn.Parameter(means_ned.detach().clone().float())
        # Flat 2DGS-like initial ellipsoids aligned with NED horizontal plane (thin in Down axis)
        init_scales = torch.full((n, 3), math.log(max(init_scale_m, 0.5)), dtype=torch.float32)
        init_scales[:, 2] = math.log(max(0.25 * init_scale_m, 0.25))
        self.log_scales = nn.Parameter(init_scales)

        init_q = torch.zeros((n, 4), dtype=torch.float32)
        init_q[:, 0] = 1.0  # Canonical Hamilton identity quaternion (w=1, x=0, y=0, z=0)
        self.quats = nn.Parameter(init_q)

        self.logit_opacities = nn.Parameter(torch.full((n,), 1.4, dtype=torch.float32))

        rgb = albedo_rgb.detach().clone().float().clamp(0.02, 0.98)
        if color_perturbation_std > 0.0:
            gen = torch.Generator().manual_seed(int(seed))
            noise = torch.randn(rgb.shape, generator=gen, dtype=torch.float32) * float(color_perturbation_std)
            rgb = (rgb + noise).clamp(0.02, 0.98)
        self.logit_colors = nn.Parameter(torch.logit(rgb))

    @property
    def num_splats(self) -> int:
        return int(self.means.shape[0])

    def normalized_quats_w_nonneg(self) -> torch.Tensor:
        q = F.normalize(self.quats, dim=-1, eps=1e-8)
        sign = torch.where(q[:, :1] < 0.0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1]))
        return q * sign

    def rgb_colors(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_colors)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacities)

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales.clamp(-6.0, 5.0))


def _accum_splat_chunk(
    xx: torch.Tensor,
    yy: torch.Tensor,
    u_c: torch.Tensor,
    v_c: torch.Tensor,
    z_c: torch.Tensor,
    sig_c: torch.Tensor,
    opa_c: torch.Tensor,
    dw_c: torch.Tensor,
    col_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure single-chunk splat accumulation kernel wrapped by gradient checkpointing."""
    du = xx - u_c.view(1, 1, -1)
    dv = yy - v_c.view(1, 1, -1)
    inv_2s2 = 0.5 / (sig_c.view(1, 1, -1) ** 2 + 1e-6)
    g = torch.exp(-(du * du + dv * dv) * inv_2s2)  # [H, W, C]
    w_ij = g * opa_c.view(1, 1, -1) * dw_c.view(1, 1, -1)
    dw_sum = w_ij.sum(dim=-1)
    rgb_sum = torch.einsum("hwc,cd->hwd", w_ij, col_c)
    depth_sum = torch.einsum("hwc,c->hw", w_ij, z_c)
    return dw_sum, rgb_sum, depth_sum


def render_gaussians_pytorch(
    means: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    view: NedCameraView,
    *,
    bg_color: tuple[float, float, float] = (0.58, 0.72, 0.88),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable pure-PyTorch Gaussian splat renderer in strict NED/CAM (`[H, W, 3]`, `[H, W]`)."""
    device = means.device
    dtype = torch.float32

    r_mat = torch.as_tensor(view.r_cam_ned, device=device, dtype=dtype)
    t_vec = torch.as_tensor(view.t_cam, device=device, dtype=dtype)
    k_mat = torch.as_tensor(view.k_mat, device=device, dtype=dtype)

    pts_cam = means.to(dtype) @ r_mat.T + t_vec.unsqueeze(0)  # [N, 3]
    z = pts_cam[:, 2]
    in_front = z > 1.0

    h, w = view.height, view.width
    bg = torch.tensor(bg_color, device=device, dtype=dtype).view(1, 1, 3)

    if not bool(in_front.any()):
        return bg.expand(h, w, 3).clone(), torch.full((h, w), 1000.0, device=device, dtype=dtype)

    p_vis = pts_cam[in_front]
    z_vis = p_vis[:, 2]
    u_vis = k_mat[0, 0] * (p_vis[:, 0] / z_vis) + k_mat[0, 2]
    v_vis = k_mat[1, 1] * (p_vis[:, 1] / z_vis) + k_mat[1, 2]

    # Screen-space radius in pixels from horizontal Gaussian scale
    horiz_scale = 0.5 * (scales[in_front, 0] + scales[in_front, 1])
    sigma_px = (k_mat[0, 0] * horiz_scale / z_vis).clamp(min=0.9, max=float(max(h, w)) * 0.35)

    # Frustum cull splats whose 3-sigma box falls outside viewport
    margin = 3.0 * sigma_px
    in_view = (
        (u_vis >= -margin)
        & (u_vis <= float(w - 1) + margin)
        & (v_vis >= -margin)
        & (v_vis <= float(h - 1) + margin)
    )
    if not bool(in_view.any()):
        return bg.expand(h, w, 3).clone(), torch.full((h, w), 1000.0, device=device, dtype=dtype)

    u_v = u_vis[in_view]
    v_v = v_vis[in_view]
    z_v = z_vis[in_view]
    sig_v = sigma_px[in_view]
    opa_v = opacities[in_front][in_view]
    col_v = colors[in_front][in_view]

    # Pixel coordinate grid [H, W]
    yy = torch.arange(h, device=device, dtype=dtype).view(h, 1, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, w, 1)

    # Chunk over visible splats to keep memory tightly bounded even on CPU
    accum_rgb = torch.zeros((h, w, 3), device=device, dtype=dtype)
    accum_depth = torch.zeros((h, w), device=device, dtype=dtype)
    accum_w = torch.zeros((h, w), device=device, dtype=dtype)

    # Depth-priority weighting (nearer splats dominate cleanly while staying smooth & differentiable)
    depth_weight = torch.exp(-0.0025 * z_v)
    n_vis = int(u_v.shape[0])
    chunk = 256 if device.type == "cuda" else 128
    use_ckpt = bool(
        torch.is_grad_enabled()
        and (
            u_v.requires_grad
            or v_v.requires_grad
            or z_v.requires_grad
            or sig_v.requires_grad
            or opa_v.requires_grad
            or col_v.requires_grad
        )
    )
    for s0 in range(0, n_vis, chunk):
        s1 = min(s0 + chunk, n_vis)
        if use_ckpt:
            dw_c, rgb_c, depth_c = grad_checkpoint(
                _accum_splat_chunk,
                xx,
                yy,
                u_v[s0:s1],
                v_v[s0:s1],
                z_v[s0:s1],
                sig_v[s0:s1],
                opa_v[s0:s1],
                depth_weight[s0:s1],
                col_v[s0:s1],
                use_reentrant=False,
            )
        else:
            dw_c, rgb_c, depth_c = _accum_splat_chunk(
                xx,
                yy,
                u_v[s0:s1],
                v_v[s0:s1],
                z_v[s0:s1],
                sig_v[s0:s1],
                opa_v[s0:s1],
                depth_weight[s0:s1],
                col_v[s0:s1],
            )
        accum_w = accum_w + dw_c
        accum_rgb = accum_rgb + rgb_c
        accum_depth = accum_depth + depth_c

    alpha = 1.0 - torch.exp(-accum_w * 2.5)
    fg_rgb = accum_rgb / (accum_w.unsqueeze(-1) + 1e-6)
    img_rgb = alpha.unsqueeze(-1) * fg_rgb + (1.0 - alpha.unsqueeze(-1)) * bg
    expected_depth = (accum_depth + (1.0 - alpha) * 1000.0) / (accum_w + (1.0 - alpha) + 1e-6)
    return img_rgb.clamp(0.0, 1.0), expected_depth


def render_gaussians_gsplat(
    scene: GaussianScene3D,
    views: list[NedCameraView],
    cfg: WorldTwinConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real Blackwell `sm_120` `gsplat` 3DGUT + 2DGS rasterization (`with_ut=True, with_eval3d=True`)."""
    import gsplat  # type: ignore[import-not-found]

    device = scene.means.device
    viewmats = torch.stack(
        [torch.as_tensor(v.viewmat_4x4(), device=device, dtype=torch.float32) for v in views],
        dim=0,
    )
    ks = torch.stack(
        [torch.as_tensor(v.k_mat, device=device, dtype=torch.float32) for v in views],
        dim=0,
    )
    renders, _alphas, _meta = gsplat.rasterization(
        means=scene.means,
        quats=scene.normalized_quats_w_nonneg(),
        scales=scene.scales(),
        opacities=scene.opacities(),
        colors=scene.rgb_colors(),
        viewmats=viewmats,
        Ks=ks,
        width=views[0].width,
        height=views[0].height,
        packed=cfg.packed_memory_mode,
        with_ut=True,
        with_eval3d=True,
        render_mode="RGB+ED",
    )
    rgb = renders[..., :3].clamp(0.0, 1.0)
    depth = renders[..., 3]
    return rgb, depth


def _compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fast differentiable 3x3 window SSIM loss (`1 - SSIM`) on `[B, H, W, 3]` tensors."""
    x = pred.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / den.clamp(min=1e-6)
    return (1.0 - ssim_map).mean()


def _compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(F.mse_loss(pred.detach().float(), target.detach().float()).item())
    if mse <= 1e-12:
        return 60.0
    return float(-10.0 * math.log10(mse))


def _pack_signed_24bit_le(values_i32: np.ndarray) -> bytes:
    """Pack int32 array clamped to signed 24-bit `[-2^23, 2^23 - 1]` into 3-byte little-endian."""
    clamped = np.clip(values_i32, -(1 << 23), (1 << 23) - 1).astype(np.int32)
    unsigned = clamped & 0xFFFFFF
    b0 = (unsigned & 0xFF).astype(np.uint8)
    b1 = ((unsigned >> 8) & 0xFF).astype(np.uint8)
    b2 = ((unsigned >> 16) & 0xFF).astype(np.uint8)
    stacked = np.stack([b0, b1, b2], axis=-1)
    return stacked.tobytes()


def write_spz_v2(
    path: Path | str,
    *,
    means_ned: np.ndarray,
    log_scales: np.ndarray,
    quats_wxyz: np.ndarray,
    logit_opacities: np.ndarray,
    rgb_01: np.ndarray,
    sh_degree: int = 0,
    fractional_bits: int = 12,
    flags: int = 0,
) -> dict[str, Any]:
    """Encode Gaussians into genuine gzip-compressed Niantic SPZ v2 (`0x5053474E` `"NGSP"`)."""
    means = np.asarray(means_ned, dtype=np.float32)
    lscales = np.asarray(log_scales, dtype=np.float32)
    quats = np.asarray(quats_wxyz, dtype=np.float32)
    lopa = np.asarray(logit_opacities, dtype=np.float32).reshape(-1)
    rgb = np.asarray(rgb_01, dtype=np.float32)

    n = int(means.shape[0])
    if n <= 0:
        raise ValueError("Cannot write empty SPZ v2 file (num_points == 0)")
    if not (0 <= int(sh_degree) <= 3):
        raise ValueError(f"Invalid sh_degree {sh_degree}; expected 0..3")
    if not (1 <= int(fractional_bits) <= 20):
        raise ValueError(f"Invalid fractional_bits {fractional_bits}")
    if not (
        np.all(np.isfinite(means))
        and np.all(np.isfinite(lscales))
        and np.all(np.isfinite(quats))
        and np.all(np.isfinite(lopa))
        and np.all(np.isfinite(rgb))
    ):
        raise ValueError("Non-finite Gaussian parameter rejected before SPZ v2 encoding")

    # Normalize Hamilton quaternions with w >= 0 and encode (x, y, z) into uint8
    q_norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    q_unit = quats / np.maximum(q_norms, 1e-8)
    neg_w = q_unit[:, 0] < 0.0
    q_unit[neg_w] = -q_unit[neg_w]
    rot_xyz_u8 = np.clip(np.rint(q_unit[:, 1:4] * 127.5 + 127.5), 0.0, 255.0).astype(np.uint8)

    # Quantize 24-bit fixed-point positions
    pos_scale = float(1 << int(fractional_bits))
    pos_q = np.rint(means * pos_scale).astype(np.int32)
    pos_bytes = _pack_signed_24bit_le(pos_q.reshape(-1))

    # Quantize alphas, colors (SH0), and scales per Niantic SPZ v2 layout
    alpha_01 = 1.0 / (1.0 + np.exp(-np.clip(lopa, -20.0, 20.0)))
    alphas_u8 = np.clip(np.rint(alpha_01 * 255.0), 0.0, 255.0).astype(np.uint8)

    sh0 = (np.clip(rgb, 0.0, 1.0) - 0.5) / _SH_C0
    colors_u8 = np.clip(np.rint(sh0 * (0.15 * 255.0) + 127.5), 0.0, 255.0).astype(np.uint8)

    scales_u8 = np.clip(np.rint((lscales + 10.0) * 16.0), 0.0, 255.0).astype(np.uint8)

    header = SPZ_HEADER_STRUCT.pack(
        SPZ_MAGIC_U32,
        SPZ_VERSION,
        n,
        int(sh_degree),
        int(fractional_bits),
        int(flags),
        0,
    )
    raw_payload = (
        header
        + pos_bytes
        + alphas_u8.tobytes()
        + colors_u8.tobytes()
        + scales_u8.tobytes()
        + rot_xyz_u8.tobytes()
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic mtime=0 gzip compression
    with out_path.open("wb") as raw_f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz:
            gz.write(raw_payload)

    return read_spz_v2_header(out_path)


def read_spz_v2_header(path: Path | str) -> dict[str, Any]:
    """Decompress and validate a Niantic SPZ v2 binary file header and payload size."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SPZ v2 file not found: {p}")
    with gzip.open(p, "rb") as gz:
        raw = gz.read()
    if len(raw) < SPZ_HEADER_STRUCT.size:
        raise ValueError(f"SPZ v2 payload truncated ({len(raw)} bytes)")

    magic_u32, version, num_points, sh_degree, fractional_bits, flags, reserved = SPZ_HEADER_STRUCT.unpack(
        raw[:16]
    )
    if magic_u32 != SPZ_MAGIC_U32:
        raise ValueError(f"Invalid SPZ magic 0x{magic_u32:08X}; expected 0x{SPZ_MAGIC_U32:08X} ('NGSP')")
    if version != SPZ_VERSION:
        raise ValueError(f"Unsupported SPZ version {version}; expected {SPZ_VERSION}")
    if num_points <= 0:
        raise ValueError(f"Invalid SPZ num_points={num_points}")

    # Expected uncompressed byte length for sh_degree=0: 16 + N*(9 pos + 1 alpha + 3 color + 3 scale + 3 rot)
    sh_extra_coeffs = {0: 0, 1: 3, 2: 8, 3: 15}[int(sh_degree)]
    expected_uncompressed = 16 + int(num_points) * (9 + 1 + 3 + 3 + 3 + 3 * sh_extra_coeffs)
    if len(raw) != expected_uncompressed:
        raise ValueError(
            f"SPZ v2 uncompressed length mismatch: got {len(raw)}, expected {expected_uncompressed}"
        )

    file_bytes = int(p.stat().st_size)
    return {
        "path": str(p),
        "magic_u32": int(magic_u32),
        "magic_ascii": SPZ_MAGIC_ASCII,
        "version": int(version),
        "num_points": int(num_points),
        "sh_degree": int(sh_degree),
        "fractional_bits": int(fractional_bits),
        "flags": int(flags),
        "reserved": int(reserved),
        "compressed_bytes": file_bytes,
        "uncompressed_bytes": len(raw),
        "spz_gb": round(file_bytes / (1024.0 ** 3), 8),
        "sha256": _sha256_file(p),
    }


def render_visual_stress_previews(
    scene: GaussianScene3D,
    view: NedCameraView,
    out_dir: Path,
    cfg: WorldTwinConfig,
    *,
    use_gsplat: bool = False,
) -> dict[str, Any]:
    """Render clear, radiation-fog, and dusk-glare RGB PNG previews stamped `VISUAL_STRESS_ONLY`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if use_gsplat:
            rgb_t, depth_t = render_gaussians_gsplat(scene, [view], cfg)
            rgb_np = rgb_t[0].detach().cpu().numpy()
            depth_np = depth_t[0].detach().cpu().numpy()
        else:
            rgb_t, depth_t = render_gaussians_pytorch(
                scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), view
            )
            rgb_np = rgb_t.detach().cpu().numpy()
            depth_np = depth_t.detach().cpu().numpy()

    # 1. Clear daylight KSFO 28L approach
    clear_img = np.clip(rgb_np, 0.0, 1.0)

    # 2. Radiation fog (Beer-Lambert Koschmieder exponential depth airlight veiling, beta=0.0045 m^-1)
    tau = np.exp(-0.0045 * np.clip(depth_np, 0.0, 1500.0))[:, :, None]
    fog_airlight = np.array([0.78, 0.81, 0.84], dtype=np.float32).reshape(1, 1, 3)
    fog_img = np.clip(clear_img * tau + fog_airlight * (1.0 - tau), 0.0, 1.0)

    # 3. Dusk glare (warm directional tint + upper-west solar glare bloom)
    h, w, _ = clear_img.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    glare = np.exp(-(((xx - 0.28) ** 2 + (yy - 0.22) ** 2) / 0.035))[:, :, None]
    dusk_tint = np.array([0.88, 0.66, 0.48], dtype=np.float32).reshape(1, 1, 3)
    glare_color = np.array([1.0, 0.85, 0.58], dtype=np.float32).reshape(1, 1, 3)
    dusk_img = np.clip(clear_img * dusk_tint + 0.55 * glare * glare_color, 0.0, 1.0)

    specs = (
        ("preview_clear.png", clear_img),
        ("preview_radiation_fog.png", fog_img),
        ("preview_dusk_glare.png", dusk_img),
    )
    records: dict[str, Any] = {}
    for fname, arr in specs:
        fpath = out_dir / fname
        u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
        Image.fromarray(u8, mode="RGB").save(fpath)
        meta = {
            "site_icao": cfg.site_icao,
            "runway_id": cfg.runway_id,
            "gt_tier": REQUIRED_GT_TIER,
            "wp01_certification_eligible": False,
            "gcp_reproj_uncertainty_m": 0.25,
            "filename": fname,
            "width": int(w),
            "height": int(h),
            "sha256": _sha256_file(fpath),
        }
        validate_frame_metadata_visual_stress_only(meta)
        records[fname] = meta
    return records


def _make_scene_optimizer(scene: GaussianScene3D) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [scene.logit_colors], "lr": 0.14},
            {"params": [scene.logit_opacities], "lr": 0.04},
            {"params": [scene.log_scales], "lr": 0.015},
            {"params": [scene.means], "lr": 0.02},
        ]
    )


def _densify_and_prune_splats(
    scene: GaussianScene3D,
    *,
    max_splats: int,
    grad_norms: torch.Tensor | None = None,
    min_opacity: float = 0.01,
    split_fraction: float = 0.10,
) -> dict[str, int]:
    """Bounded MCMC-style split+prune densification strictly capped at `<= max_splats` (`D16-G4-m2`)."""
    with torch.no_grad():
        device = scene.means.device
        n_before = scene.num_splats
        opa = scene.opacities()
        keep_mask = opa >= float(min_opacity)
        if int(keep_mask.sum().item()) < max(16, n_before // 4):
            keep_mask = torch.ones((n_before,), dtype=torch.bool, device=device)

        means = scene.means.detach()[keep_mask]
        lscales = scene.log_scales.detach()[keep_mask]
        quats = scene.quats.detach()[keep_mask]
        lopa = scene.logit_opacities.detach()[keep_mask]
        lcol = scene.logit_colors.detach()[keep_mask]
        pruned_count = int(n_before - int(means.shape[0]))

        headroom = max(0, int(max_splats) - int(means.shape[0]))
        split_count = 0
        if headroom > 0 and int(means.shape[0]) > 0:
            if grad_norms is not None and int(grad_norms.shape[0]) == n_before:
                scores = grad_norms.detach()[keep_mask]
            else:
                scores = lscales.max(dim=-1).values
            target_splits = min(headroom, max(1, int(round(float(means.shape[0]) * float(split_fraction)))))
            top_idx = torch.topk(scores, k=min(target_splits, int(means.shape[0])), largest=True).indices
            split_count = int(top_idx.shape[0])

            # Offset child Gaussians along NED horizontal plane & shrink scale by ~0.7x
            child_scales = torch.exp(lscales[top_idx].clamp(-6.0, 5.0))
            offset = torch.zeros_like(means[top_idx])
            offset[:, 0] = 0.35 * child_scales[:, 0]
            offset[:, 1] = -0.35 * child_scales[:, 1]

            new_means = means[top_idx] + offset
            new_lscales = (lscales[top_idx] - math.log(1.4)).clamp(-6.0, 5.0)
            new_quats = quats[top_idx].clone()
            new_lopa = lopa[top_idx].clone()
            new_lcol = lcol[top_idx].clone()

            means = torch.cat([means, new_means], dim=0)[:max_splats]
            lscales = torch.cat([lscales, new_lscales], dim=0)[:max_splats]
            quats = torch.cat([quats, new_quats], dim=0)[:max_splats]
            lopa = torch.cat([lopa, new_lopa], dim=0)[:max_splats]
            lcol = torch.cat([lcol, new_lcol], dim=0)[:max_splats]

        scene.means = nn.Parameter(means.contiguous())
        scene.log_scales = nn.Parameter(lscales.contiguous())
        scene.quats = nn.Parameter(quats.contiguous())
        scene.logit_opacities = nn.Parameter(lopa.contiguous())
        scene.logit_colors = nn.Parameter(lcol.contiguous())

        return {
            "splats_before": n_before,
            "splats_after": scene.num_splats,
            "pruned": pruned_count,
            "added": split_count,
        }


def _probe_world_vram_batch(
    scene: GaussianScene3D,
    views: Sequence[NedCameraView],
    gt_rgbs: Sequence[torch.Tensor],
    world_cfg: WorldTwinConfig,
    *,
    use_gsplat: bool,
    probe_candidates: Sequence[int],
    device: torch.device,
    target_frac: float = 0.86,
    vram_headroom_gib: float = 8.0,
) -> dict[str, Any]:
    """Eager VRAM batch probe exercising REAL full L1+SSIM+flatness loss + Adam step (`BC-G4-m4`).

    Uses `max(vram_headroom_gib, 12.0)` GiB safety headroom so bounded MCMC splat densification
    has generous margin on 96 GB Blackwell G4, and restores initial splat parameters bit-for-bit
    in `finally:`.
    """
    headroom_gib = max(float(vram_headroom_gib), 12.0)
    eager_snap = snapshot_eager_state(scene)
    probe_opt = _make_scene_optimizer(scene)
    tuner = AutoVRAMBatchTuner(
        target_frac=float(target_frac),
        safety_headroom_gib=headroom_gib,
        device=device,
    )

    def _probe_step(bs: int) -> None:
        probe_opt.zero_grad(set_to_none=True)
        sub_views = views[: min(bs, len(views))]
        if use_gsplat:
            pred, _ = render_gaussians_gsplat(scene, sub_views, world_cfg)
            tgt = torch.stack(list(gt_rgbs[: len(sub_views)]), dim=0)
        else:
            preds = [
                render_gaussians_pytorch(
                    scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), vv
                )[0]
                for vv in sub_views
            ]
            pred = torch.stack(preds, dim=0)
            tgt = torch.stack(list(gt_rgbs[: len(sub_views)]), dim=0)
        l1_loss = F.l1_loss(pred, tgt)
        ssim_loss = _compute_ssim_loss(pred, tgt)
        flat_reg = scene.scales().min(dim=-1).values.mean() * 1e-4
        total_loss = 0.8 * l1_loss + 0.2 * ssim_loss + flat_reg
        total_loss.backward()
        probe_opt.step()
        probe_opt.zero_grad(set_to_none=True)

    try:
        if device.type == "cpu":
            report = tuner.probe_batch_size(
                _probe_step,
                probe_candidates,
                total_vram_bytes=int(96 * (1024**3)),
                mem_probe_fn=lambda b: int(b * 18 * (1024**3)),
            )
        else:
            report = tuner.probe_batch_size(_probe_step, probe_candidates)
    finally:
        probe_opt.zero_grad(set_to_none=True)
        del probe_opt
        scene.zero_grad(set_to_none=True)
        restore_eager_state(scene, eager_snap)
    report["safety_headroom_gib"] = headroom_gib
    return report


@torch.no_grad()
def _eval_views_chunked(
    scene: GaussianScene3D,
    views: Sequence[NedCameraView],
    gt_rgbs: Sequence[torch.Tensor],
    world_cfg: WorldTwinConfig,
    *,
    use_gsplat: bool,
    chunk_size: int = 4,
) -> tuple[float, float]:
    """Evaluate L1 and global PSNR across all views in bounded VRAM mini-batches (`chunk_size=4`)."""
    if not views:
        return 0.0, 0.0
    step_n = max(1, int(chunk_size))
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    elem_count = 0

    for start in range(0, len(views), step_n):
        end = min(len(views), start + step_n)
        chunk_views = views[start:end]
        tgt_chunk = torch.stack(list(gt_rgbs[start:end]), dim=0)
        if use_gsplat:
            pred_chunk, _ = render_gaussians_gsplat(scene, chunk_views, world_cfg)
        else:
            pred_list = [
                render_gaussians_pytorch(
                    scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), vv
                )[0]
                for vv in chunk_views
            ]
            pred_chunk = torch.stack(pred_list, dim=0)
        diff = (pred_chunk.float() - tgt_chunk.float())
        abs_err_sum += float(diff.abs().sum().item())
        sq_err_sum += float((diff * diff).sum().item())
        elem_count += int(diff.numel())

    mean_l1 = abs_err_sum / float(max(1, elem_count))
    mse = sq_err_sum / float(max(1, elem_count))
    psnr = 60.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    return mean_l1, psnr


def run_xvs_world_g4_training(
    *,
    out_dir: Path | str,
    cfg: WorldTwinConfig | None = None,
    ksfo_bin_path: Path | str = DEFAULT_KSFO_PACK_PATH,
    drive_dir: Path | str | None = None,
    smoke_local: bool = False,
    use_naip_stac: bool = False,
    backend: str = "auto",
    max_steps: int | None = None,
    grid_res: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    num_views: int | None = None,
) -> dict[str, Any]:
    """Execute scoped 1x1 km KSFO 28L 3DGS visual-stress training, SPZ v2 export & preview rendering."""
    world_cfg = ksfo_28l_default_config() if cfg is None else cfg
    assert world_cfg.gt_tier == REQUIRED_GT_TIER

    torch.manual_seed(world_cfg.seed)
    np.random.seed(world_cfg.seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = DriveCheckpointManager(
        local_dir=out_path / "checkpoints",
        drive_dir=drive_dir,
    )
    env_report = collect_blackwell_env_report()

    # Determine runtime dimensions & backend
    g_res = (16 if smoke_local else 64) if grid_res is None else int(grid_res)
    img_w = (48 if smoke_local else 512) if image_width is None else int(image_width)
    img_h = (36 if smoke_local else 384) if image_height is None else int(image_height)
    n_views = (6 if smoke_local else 36) if num_views is None else int(num_views)
    steps = (24 if smoke_local else 3000) if max_steps is None else int(max_steps)

    has_gsplat = False
    if backend in ("auto", "gsplat") and torch.cuda.is_available() and not smoke_local:
        try:
            import gsplat as _gsplat_mod  # type: ignore[import-not-found]  # noqa: F401

            has_gsplat = True
        except ImportError:
            has_gsplat = False
    use_gsplat = bool(has_gsplat and backend in ("auto", "gsplat"))
    if backend == "gsplat" and not use_gsplat:
        raise RuntimeError("Requested --backend gsplat, but CUDA + gsplat package are not available")

    device = torch.device("cuda" if (torch.cuda.is_available() and not smoke_local) else "cpu")

    # 1. Load real KSFO 3DEP heightfield via M2 `load_ksfo_3dep_mipmap`
    patch = load_ksfo_3dep_patch_ned(
        world_cfg,
        ksfo_bin_path=ksfo_bin_path,
        grid_res=g_res,
        use_naip_stac=(use_naip_stac and not smoke_local),
    )
    print(
        f"[XVS-World] Loaded KSFO 3DEP ({patch['terrain_mode']}, {patch['mipmap_levels']} mipmap levels, "
        f"grid={g_res}x{g_res}, albedo={patch['albedo_source']}, backend={'gsplat' if use_gsplat else 'pytorch'})",
        flush=True,
    )

    means_np = np.stack(
        [patch["north_m"].reshape(-1), patch["east_m"].reshape(-1), patch["down_m"].reshape(-1)],
        axis=-1,
    )
    albedo_np = patch["albedo_rgb"].reshape(-1, 3)

    if means_np.shape[0] > world_cfg.max_splats:
        raise ValueError(
            f"Initialized splat count {means_np.shape[0]} exceeds cfg.max_splats={world_cfg.max_splats}"
        )

    views = build_ksfo_approach_cameras(num_views=n_views, width=img_w, height=img_h)
    cell_pitch_m = world_cfg.extent_x_m / float(max(1, g_res - 1))

    # 2. Build ground-truth reference scene views from exact KSFO 3DEP + ortho albedo
    gt_scene = GaussianScene3D(
        torch.from_numpy(means_np),
        torch.from_numpy(albedo_np),
        init_scale_m=0.85 * cell_pitch_m,
        color_perturbation_std=0.0,
        seed=world_cfg.seed,
    ).to(device)

    gt_rgbs: list[torch.Tensor] = []
    with torch.no_grad():
        if use_gsplat:
            for start in range(0, len(views), 4):
                chunk_v = views[start : min(len(views), start + 4)]
                rgb_chunk, _ = render_gaussians_gsplat(gt_scene, chunk_v, world_cfg)
                for idx in range(len(chunk_v)):
                    gt_rgbs.append(rgb_chunk[idx].detach())
        else:
            for v in views:
                rgb_v, _ = render_gaussians_pytorch(
                    gt_scene.means, gt_scene.scales(), gt_scene.opacities(), gt_scene.rgb_colors(), v
                )
                gt_rgbs.append(rgb_v.detach())

    # 3. Initialize trainable 3DGS model with color perturbation so optimization learns real terrain/runway
    scene = GaussianScene3D(
        torch.from_numpy(means_np),
        torch.from_numpy(albedo_np),
        init_scale_m=0.85 * cell_pitch_m,
        color_perturbation_std=0.22,
        seed=world_cfg.seed + 17,
    ).to(device)

    # 4. Eager AutoVRAMBatchTuner probe BEFORE training with full L1+SSIM+Adam & weight restore (`BC-G4-m4`)
    probe_candidates = [1, 2, min(4, len(views))] if smoke_local else [1, 2, 4, 8, 16, min(32, len(views))]
    vram_probe = _probe_world_vram_batch(
        scene,
        views,
        gt_rgbs,
        world_cfg,
        use_gsplat=use_gsplat,
        probe_candidates=probe_candidates,
        device=device,
        target_frac=0.86,
        vram_headroom_gib=12.0,
    )

    view_batch_size = max(1, min(int(vram_probe["selected_batch_size"]), len(views)))
    probe_vram_gib = float(vram_probe["peak_vram_bytes"]) / float(1024**3)
    print(
        f"[XVS-World] AutoVRAMBatchTuner selected view_batch_size={view_batch_size} "
        f"(peak_vram={probe_vram_gib:.2f} GiB, util={float(vram_probe['utilization_frac']) * 100.0:.1f}%, "
        f"headroom={float(vram_probe['safety_headroom_gib']):.1f} GiB)",
        flush=True,
    )

    # 5. Checkpoint resume check (if resuming interrupted or completed Colab G4 session)
    optimizer = _make_scene_optimizer(scene)
    start_step = 0
    resumed = False
    saved_loss_init: float | None = None
    saved_psnr_init: float | None = None
    prior_ckpt = ckpt_mgr.load_latest(tag_prefix="xvs_world_step_")
    if prior_ckpt is not None and "model_state_dict" in prior_ckpt:
        sd = prior_ckpt["model_state_dict"]
        ckpt_n = int(sd["means"].shape[0])
        if ckpt_n != scene.num_splats:
            scene.means = nn.Parameter(sd["means"].to(device=device, dtype=torch.float32).clone())
            scene.log_scales = nn.Parameter(sd["log_scales"].to(device=device, dtype=torch.float32).clone())
            scene.quats = nn.Parameter(sd["quats"].to(device=device, dtype=torch.float32).clone())
            scene.logit_opacities = nn.Parameter(sd["logit_opacities"].to(device=device, dtype=torch.float32).clone())
            scene.logit_colors = nn.Parameter(sd["logit_colors"].to(device=device, dtype=torch.float32).clone())
            optimizer = _make_scene_optimizer(scene)
        else:
            scene.load_state_dict(sd, strict=True)
        if "optimizer_state_dict" in prior_ckpt:
            optimizer.load_state_dict(prior_ckpt["optimizer_state_dict"])
        start_step = int(prior_ckpt["step"])
        if "loss_init" in prior_ckpt and math.isfinite(float(prior_ckpt["loss_init"])):
            saved_loss_init = float(prior_ckpt["loss_init"])
        if "psnr_init" in prior_ckpt and math.isfinite(float(prior_ckpt["psnr_init"])):
            saved_psnr_init = float(prior_ckpt["psnr_init"])
        resumed = True

    def _eval_all_views() -> tuple[float, float]:
        return _eval_views_chunked(
            scene,
            views,
            gt_rgbs,
            world_cfg,
            use_gsplat=use_gsplat,
            chunk_size=4,
        )

    cur_loss, cur_psnr = _eval_all_views()
    loss_init = cur_loss if saved_loss_init is None else saved_loss_init
    psnr_init = cur_psnr if saved_psnr_init is None else saved_psnr_init

    densify_policy = (
        "bounded_grad_split_prune_mcmc_cap" if use_gsplat else "static_3dep_grid_smoke_local"
    )
    densify_events = 0
    densify_interval = max(100, steps // 10)
    log_interval = max(1, steps // 10)
    grad_accum = torch.zeros((scene.num_splats,), device=device, dtype=torch.float32)

    # 6. Main optimization loop (L1 + SSIM + 2DGS flatness regularization + bounded gsplat MCMC densification)
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        sel_idx = [(step * view_batch_size + k) % len(views) for k in range(view_batch_size)]
        sel_views = [views[i] for i in sel_idx]
        tgt_batch = torch.stack([gt_rgbs[i] for i in sel_idx], dim=0)

        if use_gsplat:
            pred_batch, _ = render_gaussians_gsplat(scene, sel_views, world_cfg)
        else:
            preds = [
                render_gaussians_pytorch(
                    scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), vv
                )[0]
                for vv in sel_views
            ]
            pred_batch = torch.stack(preds, dim=0)

        l1_loss = F.l1_loss(pred_batch, tgt_batch)
        ssim_loss = _compute_ssim_loss(pred_batch, tgt_batch)
        # 2DGS surface flatness regularization encourages smallest axis (Down) to stay thin
        flat_reg = scene.scales().min(dim=-1).values.mean() * 1e-4
        total_loss = 0.8 * l1_loss + 0.2 * ssim_loss + flat_reg
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError(f"Non-finite training loss at step {step}: {total_loss.item()}")
        total_loss.backward()
        if use_gsplat and scene.means.grad is not None and int(grad_accum.shape[0]) == scene.num_splats:
            grad_accum = grad_accum + scene.means.grad.detach().norm(dim=-1)
        optimizer.step()

        # Bounded MCMC-style densification strictly capped at <= world_cfg.max_splats (`D16-G4-m2`)
        if (
            use_gsplat
            and step > 0
            and (step + 1) % densify_interval == 0
            and step < int(0.75 * steps)
            and scene.num_splats < world_cfg.max_splats
        ):
            _densify_and_prune_splats(
                scene,
                max_splats=world_cfg.max_splats,
                grad_norms=grad_accum,
                split_fraction=0.10,
            )
            optimizer = _make_scene_optimizer(scene)
            grad_accum = torch.zeros((scene.num_splats,), device=device, dtype=torch.float32)
            densify_events += 1

        if (step + 1) % log_interval == 0 or step == steps - 1:
            step_psnr = _compute_psnr(pred_batch, tgt_batch)
            peak_vram_gib = (
                float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
                if device.type == "cuda" and torch.cuda.is_available()
                else 0.0
            )
            print(
                f"[XVS-World] step {step + 1}/{steps} | loss={float(total_loss.item()):.5f} | "
                f"psnr={step_psnr:.2f} dB | splats={scene.num_splats} | peak_vram={peak_vram_gib:.2f} GiB",
                flush=True,
            )

        # Periodic atomic checkpoint (mid-run + final), persisting initial baseline metrics for resume
        if step == max(0, steps // 2 - 1) or step == steps - 1:
            ckpt_mgr.save(
                f"xvs_world_step_{step + 1:06d}",
                {
                    "step": step + 1,
                    "gt_tier": world_cfg.gt_tier,
                    "loss_init": float(loss_init),
                    "psnr_init": float(psnr_init),
                    "model_state_dict": {k: v.detach().cpu() for k, v in scene.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )

    loss_final, psnr_final = _eval_all_views()

    # 7. Export genuine Niantic SPZ v2 (`BC-G4-m1`) & verify cap <= 0.35 GB
    spz_path = out_path / "ksfo_28l_1km_visual_stress.spz"
    spz_header = write_spz_v2(
        spz_path,
        means_ned=scene.means.detach().cpu().numpy(),
        log_scales=scene.log_scales.detach().cpu().numpy(),
        quats_wxyz=scene.normalized_quats_w_nonneg().detach().cpu().numpy(),
        logit_opacities=scene.logit_opacities.detach().cpu().numpy(),
        rgb_01=scene.rgb_colors().detach().cpu().numpy(),
        sh_degree=0,
        fractional_bits=12,
    )
    if float(spz_header["spz_gb"]) > world_cfg.spz_max_gb:
        raise RuntimeError(
            f"SPZ v2 size {spz_header['spz_gb']:.6f} GB exceeds cap {world_cfg.spz_max_gb} GB"
        )

    # 8. Render clear, radiation-fog & dusk visual-stress PNG previews
    preview_view = views[0]
    previews = render_visual_stress_previews(
        scene, preview_view, out_path / "previews", world_cfg, use_gsplat=use_gsplat
    )
    print(
        f"[XVS-World] Exported SPZ v2 ({spz_header['compressed_bytes']} B, {spz_header['num_points']} splats, "
        f"sha256={spz_header['sha256'][:12]}...) + {len(previews)} visual-stress PNG previews | "
        f"PSNR {psnr_init:.2f} -> {psnr_final:.2f} dB",
        flush=True,
    )

    summary: dict[str, Any] = {
        "schema": "xvs.v2_m2.xvs_world_g4_train_result/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_tag": "SMOKE_LOCAL_CPU" if smoke_local else "COLAB_G4_GPU_ARTIFACT",
        "gt_tier": world_cfg.gt_tier,
        "wp01_certification_eligible": False,
        "backend_requested": backend,
        "backend_used": "gsplat_3dgut_2dgs" if use_gsplat else "pytorch_differentiable_fallback",
        "site_icao": world_cfg.site_icao,
        "runway_id": world_cfg.runway_id,
        "extent_x_m": world_cfg.extent_x_m,
        "extent_y_m": world_cfg.extent_y_m,
        "ksfo_3dep_terrain_mode": patch["terrain_mode"],
        "ksfo_3dep_geoid_model": patch["geoid_model"],
        "ksfo_3dep_mipmap_levels": patch["mipmap_levels"],
        "ksfo_bin_sha256": patch["ksfo_bin_sha256"],
        "albedo_source": patch["albedo_source"],
        "num_views": len(views),
        "image_width": img_w,
        "image_height": img_h,
        "num_splats": scene.num_splats,
        "max_splats_cap": world_cfg.max_splats,
        "densification_policy": densify_policy,
        "densify_events_count": densify_events,
        "steps_completed": max(steps, start_step),
        "checkpoint_resumed": resumed,
        "loss_init": round(loss_init, 6),
        "loss_final": round(loss_final, 6),
        "psnr_init_db": round(psnr_init, 4),
        "psnr_final_db": round(psnr_final, 4),
        "vram_probe": vram_probe,
        "env_report": env_report,
        "spz_header": spz_header,
        "previews": previews,
        "pass": bool(
            loss_final < loss_init
            and psnr_final > psnr_init
            and spz_header["magic_u32"] == SPZ_MAGIC_U32
            and spz_header["version"] == SPZ_VERSION
            and float(spz_header["spz_gb"]) <= world_cfg.spz_max_gb
            and len(previews) == 3
        ),
    }

    (out_path / "xvs_world_train_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenXVS v2.0 XVS-World 1x1 km KSFO 28L 3DGS G4 Trainer.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ksfo-bin", type=Path, default=DEFAULT_KSFO_PACK_PATH)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--smoke-local", action="store_true")
    parser.add_argument("--use-naip-stac", action="store_true")
    parser.add_argument("--backend", choices=("auto", "gsplat", "pytorch"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    res = run_xvs_world_g4_training(
        out_dir=args.out_dir,
        ksfo_bin_path=args.ksfo_bin,
        drive_dir=args.drive_dir,
        smoke_local=args.smoke_local,
        use_naip_stac=args.use_naip_stac,
        backend=args.backend,
        max_steps=args.max_steps,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
