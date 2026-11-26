"""Production Colab G4 (96 GB VRAM Blackwell) 2-stage trainer for XVS-Land (SP-2B §3.2 / DR-03).

Features:
  - Strict fail-closed `BC-G4-M2` guard: ALWAYS requires `--lard-manifest <path>` and executes
    `validate_lard_real_airports_manifest(lard_manifest, train_icaos, heldout_icaos)` BEFORE
    any training (raises `PermissionError` if missing, `< 36` ICAOs, overlapping, or KSFO-leaking).
  - Parses real `data/reference/ourairports/runways.csv` (non-LARD, non-KSFO paved CONUS runways)
    to generate perspective 3D runway approach crops strictly in `CONVENTIONS.md` NED/CAM frames.
  - Full 2-stage GPU loop on `XvsLandModel`:
      * Stage 2A: `SimCC` 1D KL-divergence + `OKLS` continuous Hesse keyline loss (`bf16` no
        `GradScaler` + FP32 loss, eager `AutoVRAMBatchTuner` before `torch.compile`, atomic resume).
      * Stage 2B: Freeze `backbone`/`simcc_head`/`keyline_head`, train `CholeskyBivariateGnllHead`
        via `bivariate_gnll_loss`, and fit `StudentTCalibrator(nu=6.0)` post-hoc scale.
  - Evaluates `simcc_rmse_px`, `okls_angle_err_deg`, `pnpl_pos_rmse_m` (`solve_4dof_map_pnpl`),
    `v_raim_cat1_availability` (`compute_hybrid_conformal_raim_pl`), and `hmi_count`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xvs.v2.colab_g4_utils import (
    AutoVRAMBatchTuner,
    DriveCheckpointManager,
    collect_blackwell_env_report,
)
from xvs.v2.license_guard import assert_dataset_allowed, assert_model_weight_allowed
from xvs.v2.xvs_land.dataset_guard import (
    DatasetSplitVerification,
    ICAO_PATTERN,
    validate_lard_real_airports_manifest,
)
from xvs.v2.xvs_land.model import (
    KEYLINE_ENDPOINT_INDICES,
    KEYLINE_SIGMA_MAX_PX,
    KEYLINE_SIGMA_MIN_PX,
    NUM_KEYLINES,
    NUM_KEYPOINTS,
    StudentTCalibrator,
    XvsLandModel,
    XvsLandV52Model,
    XvsLandV52OnnxWrapper,
    bivariate_gnll_loss,
    focal_bce_with_logits_loss,
    keyline_endpoint_loss,
    keyline_keypoint_consistency_loss,
    keyline_sigma_nll_loss,
    monotonic_degradation_ranking_loss,
    okls_keyline_loss,
)
from xvs.v2.xvs_land.pnpl_ekf import (
    build_approach_trajectory_poses,
    build_ksfo_28l_reference_features,
    compute_hesse_line_from_points,
    evaluate_approach_ekf_trajectory,
    project_runway_points_ned,
    solve_4dof_map_pnpl,
)
from xvs.v2.xvs_land.raim_landing import (
    KSFO_28L_WIDTH_M,
    KSFO_28R_LATERAL_OFFSET_M,
    KSFO_TAXIWAY_C_WIDTH_M,
    check_runway_association,
    compute_hybrid_conformal_raim_pl,
)

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[4]
DEFAULT_RUNWAYS_CSV = REPO_ROOT_DEFAULT / "data" / "reference" / "ourairports" / "runways.csv"
FT_TO_M: float = 0.3048


@dataclass(frozen=True)
class RunwaySpec:
    """Paved CONUS runway geometry parsed from OurAirports `runways.csv`."""

    icao: str
    length_m: float
    width_m: float
    heading_deg: float
    le_lat_deg: float | None = None
    le_lon_deg: float | None = None


@dataclass(frozen=True)
class NaipRunwayPatch:
    """Georeferenced USDA NAIP ortho patch aligned to a runway's physical `(s, lat)` corridor."""

    icao: str
    image_rgb: np.ndarray  # [H, W, 3] float32 in [0, 1]
    s_near_m: float
    s_far_m: float
    lat_half_m: float
    meters_per_px: float
    heading_deg: float
    georeferenced: bool = True


@dataclass(frozen=True)
class XvsLandG4Config:
    """Configuration for production Colab L4/G4 & offline `--smoke-local` XVS-Land training."""

    out_dir: Path
    lard_manifest: Path | None
    runways_csv: Path = DEFAULT_RUNWAYS_CSV
    drive_ckpt_dir: Path | None = None
    arch: str = "v45"
    smoke_local: bool = False
    real_world_scale: bool = False
    use_naip_stac: bool = False
    naip_stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    naip_max_airports: int = 36
    export_onnx: bool = True
    export_fp16_onnx: bool = True
    stage2a_epochs: int = 12
    stage2b_epochs: int = 6
    steps_per_epoch: int = 160
    num_train_samples: int = 1024
    num_val_samples: int = 128
    ekf_eval_frames: int = 40
    roi_size: int = 256
    split_ratio: float = 2.0
    batch_candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    lr_stage2a: float = 4.0e-4
    lr_stage2b: float = 2.0e-3
    weight_decay: float = 1.0e-4
    student_t_nu: float = 6.0
    use_compile: bool = False
    vram_target_frac: float = 0.86
    vram_headroom_gib: float = 12.0
    seed: int = 20261121
    v53_mode: bool = False
    neg_fraction: float = 0.35
    coord_dropout_prob: float = 0.50
    roi_jitter_frac: float = 0.35


def load_conus_runways_from_csv(
    runways_csv: Path,
    *,
    exclude_icaos: set[str] | frozenset[str],
    max_airports: int = 400,
    min_length_ft: float = 2200.0,
    min_width_ft: float = 50.0,
) -> list[RunwaySpec]:
    """Parse real paved CONUS `K*` runways from `ourairports/runways.csv`, excluding LARD, KSFO & KORS."""
    if not runways_csv.is_file():
        raise FileNotFoundError(f"OurAirports runways.csv not found: {runways_csv}")

    paved_tokens = ("ASP", "CON", "BIT", "PEM")
    by_icao: dict[str, RunwaySpec] = {}
    with runways_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ident = str(row["airport_ident"]).strip().upper()
            if not ident.startswith("K") or not ICAO_PATTERN.match(ident):
                continue
            if ident in exclude_icaos or ident in ("KSFO", "KORS"):
                continue
            if str(row["closed"]).strip() == "1":
                continue
            surf = str(row["surface"]).strip().upper()
            if not any(tok in surf for tok in paved_tokens):
                continue
            try:
                len_ft = float(row["length_ft"])
                wid_ft = float(row["width_ft"])
            except (TypeError, ValueError):
                continue
            if not (
                math.isfinite(len_ft)
                and math.isfinite(wid_ft)
                and len_ft >= float(min_length_ft)
                and wid_ft >= float(min_width_ft)
            ):
                continue
            hdg_raw = str(row["le_heading_degT"]).strip()
            try:
                hdg_deg = float(hdg_raw) if hdg_raw else 270.0
            except ValueError:
                hdg_deg = 270.0
            if not math.isfinite(hdg_deg):
                hdg_deg = 270.0

            lat_val: float | None = None
            lon_val: float | None = None
            lat_raw = str(row["le_latitude_deg"]).strip() if "le_latitude_deg" in row else ""
            lon_raw = str(row["le_longitude_deg"]).strip() if "le_longitude_deg" in row else ""
            if lat_raw and lon_raw:
                try:
                    lat_f = float(lat_raw)
                    lon_f = float(lon_raw)
                    if math.isfinite(lat_f) and math.isfinite(lon_f) and (24.0 <= lat_f <= 50.0) and (-126.0 <= lon_f <= -66.0):
                        lat_val = lat_f
                        lon_val = lon_f
                except ValueError:
                    lat_val = None
                    lon_val = None

            spec = RunwaySpec(
                icao=ident,
                length_m=len_ft * FT_TO_M,
                width_m=wid_ft * FT_TO_M,
                heading_deg=hdg_deg % 360.0,
                le_lat_deg=lat_val,
                le_lon_deg=lon_val,
            )
            if ident not in by_icao or spec.length_m > by_icao[ident].length_m:
                by_icao[ident] = spec
            if len(by_icao) >= max_airports:
                break

    return [by_icao[k] for k in sorted(by_icao.keys())]


def enforce_xvs_land_licenses_and_split(
    cfg: XvsLandG4Config,
) -> tuple[DatasetSplitVerification, list[RunwaySpec]]:
    """Enforce license guard AND fail-closed `BC-G4-M2` LARD real-airports manifest guard."""
    assert_dataset_allowed("faa_nasr", "train")
    assert_model_weight_allowed("dfine_m_coco", allow_non_osi_attribution=False)
    assert_model_weight_allowed("convnext_v1_small", allow_non_osi_attribution=False)

    # Read LARD ICAOs only if manifest exists (otherwise let validate_lard_real_airports_manifest raise PermissionError)
    lard_icaos: set[str] = set()
    if cfg.lard_manifest is not None and Path(cfg.lard_manifest).is_file():
        raw_doc = json.loads(Path(cfg.lard_manifest).read_text(encoding="utf-8"))
        if isinstance(raw_doc, dict) and "airports" in raw_doc and isinstance(raw_doc["airports"], list):
            lard_icaos = {str(x).strip().upper() for x in raw_doc["airports"] if isinstance(x, str)}

    # Hold out Campaign 1 / KSFO / KORS FIRST (strict zero leakage before GA runway inclusion)
    heldout_icaos = ["KSFO", "KOAK", "KSJC", "KORS"]
    excluded = set(lard_icaos) | set(heldout_icaos)
    max_ap = 64 if cfg.smoke_local else (1200 if cfg.real_world_scale else 400)
    train_runways = load_conus_runways_from_csv(
        cfg.runways_csv,
        exclude_icaos=excluded,
        max_airports=max_ap,
        min_length_ft=2200.0,
        min_width_ft=50.0,
    )
    train_icaos = [r.icao for r in train_runways]

    split_ver = validate_lard_real_airports_manifest(
        cfg.lard_manifest,
        train_icaos=train_icaos,
        heldout_icaos=heldout_icaos,
    )
    return split_ver, train_runways


def _runway_corridor_corners_wgs84(
    rwy: RunwaySpec,
    *,
    s_near_m: float,
    s_far_m: float,
    lat_half_m: float,
) -> tuple[list[float], list[float]]:
    """Compute WGS-84 `(lons, lats)` for the 4 runway corridor corners matching `_warp_ground_plane_perspective`."""
    assert rwy.le_lat_deg is not None and rwy.le_lon_deg is not None
    lat0 = float(rwy.le_lat_deg)
    lon0 = float(rwy.le_lon_deg)
    hdg = math.radians(float(rwy.heading_deg))
    cos_h = math.cos(hdg)
    sin_h = math.sin(hdg)
    cos_lat = max(0.1, math.cos(math.radians(lat0)))
    # Order matches src_uv in _warp_ground_plane_perspective:
    # 0: (s_near, -lat_half) bottom-left, 1: (s_near, +lat_half) bottom-right,
    # 2: (s_far, +lat_half) top-right,    3: (s_far, -lat_half) top-left
    corners_sl = (
        (s_near_m, -lat_half_m),
        (s_near_m, +lat_half_m),
        (s_far_m, +lat_half_m),
        (s_far_m, -lat_half_m),
    )
    lons: list[float] = []
    lats: list[float] = []
    for s_m, lat_off_m in corners_sl:
        d_north = s_m * cos_h - lat_off_m * sin_h
        d_east = s_m * sin_h + lat_off_m * cos_h
        lats.append(lat0 + d_north / 111_320.0)
        lons.append(lon0 + d_east / (111_320.0 * cos_lat))
    return lons, lats


def fetch_conus_airport_naip_textures(
    runways: Sequence[RunwaySpec],
    stac_url: str,
    *,
    max_airports: int = 36,
    patch_size: int = 384,
    max_wall_s: float = 90.0,
) -> list[NaipRunwayPatch]:
    """Lazily fetch georeferenced U.S. Public Domain `USDA NAIP` STAC runway-aligned ortho patches."""
    assert_dataset_allowed("usda_naip", "train")
    try:
        import planetary_computer  # type: ignore[import-not-found]
        import pystac_client  # type: ignore[import-not-found]
        import rasterio  # type: ignore[import-not-found]
        from rasterio.warp import transform as rio_transform  # type: ignore[import-not-found]
        from rasterio.windows import Window  # type: ignore[import-not-found]
    except Exception as exc:
        print(
            f"[XVS-Land] NAIP STAC optional dependencies unavailable ({type(exc).__name__}: {exc}); "
            "falling back to procedural CONUS terrain.",
            flush=True,
        )
        return []

    geo_runways = [r for r in runways if r.le_lat_deg is not None and r.le_lon_deg is not None]
    if not geo_runways:
        return []
    step = max(1, len(geo_runways) // max(1, int(max_airports)))
    chosen = geo_runways[::step][: int(max_airports)]

    patches: list[NaipRunwayPatch] = []
    t_start = time.monotonic()
    try:
        catalog = pystac_client.Client.open(
            stac_url,
            modifier=planetary_computer.sign_inplace,
        )
        with rasterio.Env(
            GDAL_HTTP_TIMEOUT="15",
            GDAL_HTTP_MAX_RETRY="1",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        ):
            for rwy in chosen:
                if (time.monotonic() - t_start) >= float(max_wall_s):
                    break
                try:
                    s_near_m = -200.0
                    s_far_m = float(rwy.length_m) + 300.0
                    lat_half_m = max(260.0, float(rwy.width_m) * 6.0)
                    lons, lats = _runway_corridor_corners_wgs84(
                        rwy, s_near_m=s_near_m, s_far_m=s_far_m, lat_half_m=lat_half_m
                    )
                    bbox = [min(lons) - 0.002, min(lats) - 0.002, max(lons) + 0.002, max(lats) + 0.002]
                    search = catalog.search(collections=["naip"], bbox=bbox, max_items=2)
                    items = list(search.items())
                    if not items:
                        continue
                    for item in items:
                        if "image" not in item.assets:
                            continue
                        href = item.assets["image"].href
                        with rasterio.open(href) as src:
                            w, h = int(src.width), int(src.height)
                            if int(src.count) < 3 or src.crs is None or w < 64 or h < 64:
                                continue
                            xs, ys = rio_transform("EPSG:4326", src.crs, lons, lats)
                            cols: list[float] = []
                            rows: list[float] = []
                            for gx, gy in zip(xs, ys):
                                r_f, c_f = src.index(gx, gy, op=lambda v: v)
                                rows.append(float(r_f))
                                cols.append(float(c_f))
                            # Verify runway corridor actually lies inside this GeoTIFF tile
                            in_count = sum(
                                1 for r_f, c_f in zip(rows, cols) if (0.0 <= r_f < h and 0.0 <= c_f < w)
                            )
                            if in_count < 3:
                                continue
                            pad = 16
                            c0 = max(0, int(math.floor(min(cols))) - pad)
                            r0 = max(0, int(math.floor(min(rows))) - pad)
                            c1 = min(w, int(math.ceil(max(cols))) + pad)
                            r1 = min(h, int(math.ceil(max(rows))) + pad)
                            win_w = c1 - c0
                            win_h = r1 - r0
                            if win_w < 32 or win_h < 32 or win_w > 4096 or win_h > 4096:
                                continue
                            arr = src.read([1, 2, 3], window=Window(c0, r0, win_w, win_h))
                            if arr.shape[0] != 3 or float(arr.std()) < 3.0:
                                continue
                            tile_rgb = np.ascontiguousarray(
                                np.transpose(arr.astype(np.float32) / 255.0, (1, 2, 0))
                            )
                            # Exact affine warp from GeoTIFF corridor corners to runway-aligned ortho
                            src_tri = np.array(
                                [
                                    [cols[0] - c0, rows[0] - r0],  # bottom-left (s_near, -lat_half)
                                    [cols[1] - c0, rows[1] - r0],  # bottom-right (s_near, +lat_half)
                                    [cols[3] - c0, rows[3] - r0],  # top-left (s_far, -lat_half)
                                ],
                                dtype=np.float32,
                            )
                            sz = int(patch_size)
                            dst_tri = np.array(
                                [[0.0, float(sz - 1)], [float(sz - 1), float(sz - 1)], [0.0, 0.0]],
                                dtype=np.float32,
                            )
                            m_aff = cv2.getAffineTransform(src_tri, dst_tri)
                            aligned = cv2.warpAffine(
                                tile_rgb,
                                m_aff,
                                (sz, sz),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT101,
                            )
                            valid_frac = float(np.mean(np.max(aligned, axis=-1) > 0.01))
                            if valid_frac < 0.85 or float(aligned.std()) < 0.015:
                                continue
                            mpp = float((s_far_m - s_near_m) / float(max(1, sz)))
                            patches.append(
                                NaipRunwayPatch(
                                    icao=rwy.icao,
                                    image_rgb=np.clip(aligned, 0.0, 1.0).astype(np.float32),
                                    s_near_m=s_near_m,
                                    s_far_m=s_far_m,
                                    lat_half_m=lat_half_m,
                                    meters_per_px=mpp,
                                    heading_deg=float(rwy.heading_deg),
                                    georeferenced=True,
                                )
                            )
                            break
                except Exception:
                    continue
    except Exception as exc:
        print(
            f"[XVS-Land] NAIP STAC catalog query interrupted ({type(exc).__name__}: {exc}); "
            f"continuing with {len(patches)} georeferenced patches.",
            flush=True,
        )
        return patches

    print(
        f"[XVS-Land] NAIP STAC: loaded {len(patches)}/{len(chosen)} georeferenced USDA NAIP runway patches "
        f"in {time.monotonic() - t_start:.1f}s.",
        flush=True,
    )
    return patches


def _build_procedural_ortho_canvas(
    rng: np.random.Generator,
    *,
    canvas_size: int = 384,
    scene_kind: str = "airport_surround",
) -> np.ndarray:
    """Build a top-down RGB ground orthophoto canvas in `[0, 1]` for 3D `cv2.warpPerspective` projection."""
    sz = int(canvas_size)
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32) / float(max(1, sz - 1))
    base_col = rng.uniform(0.18, 0.42, size=(1, 1, 3)).astype(np.float32)
    tex = (
        0.06 * np.sin(2.0 * math.pi * (rng.uniform(3.0, 10.0) * xx + rng.uniform(2.0, 7.0) * yy))
        + 0.05 * np.cos(2.0 * math.pi * (rng.uniform(4.0, 12.0) * yy - rng.uniform(1.0, 5.0) * xx))
        + rng.normal(0.0, 0.025, size=(sz, sz)).astype(np.float32)
    )
    canvas = np.clip(base_col + tex[:, :, None], 0.04, 0.90).astype(np.float32)

    if scene_kind == "interstate_highway":
        # Multi-lane divided interstate with asphalt lanes, double yellow median & white dashes
        x_mid = int(sz * rng.uniform(0.30, 0.70))
        hw_w = max(14, int(sz * rng.uniform(0.09, 0.16)))
        asph = float(rng.uniform(0.16, 0.28))
        canvas[:, max(0, x_mid - hw_w) : min(sz, x_mid + hw_w), :] = asph
        # Double yellow median
        med_col = np.array([0.85, 0.72, 0.16], dtype=np.float32)
        cv2.line(canvas, (x_mid - 2, 0), (x_mid - 2, sz - 1), tuple(float(c) for c in med_col), 2)
        cv2.line(canvas, (x_mid + 2, 0), (x_mid + 2, sz - 1), tuple(float(c) for c in med_col), 2)
        # Dashed white lane lines + shoulders
        for side in (-1, 1):
            lx = int(np.clip(x_mid + side * (hw_w // 2), 2, sz - 3))
            for y0 in range(0, sz, 24):
                cv2.line(canvas, (lx, y0), (lx, min(sz - 1, y0 + 12)), (0.88, 0.88, 0.86), 1)
            ex = int(np.clip(x_mid + side * (hw_w - 2), 2, sz - 3))
            cv2.line(canvas, (ex, 0), (ex, sz - 1), (0.82, 0.82, 0.80), 1)
        # Crossing overpass deck
        if float(rng.random()) < 0.65:
            ov_y = int(sz * rng.uniform(0.25, 0.75))
            cv2.rectangle(canvas, (0, ov_y - 10), (sz - 1, ov_y + 10), (0.48, 0.48, 0.50), -1)
    elif scene_kind == "coastal_bridge":
        # Water surface + long concrete bridge causeway deck + shoreline
        water = np.array([0.10, 0.22, 0.34], dtype=np.float32)
        wave = 0.03 * np.sin(2.0 * math.pi * 18.0 * yy + 3.0 * xx)
        canvas = np.clip(water[None, None, :] + wave[:, :, None], 0.04, 0.75).astype(np.float32)
        bx = int(sz * rng.uniform(0.30, 0.70))
        bw = max(12, int(sz * rng.uniform(0.06, 0.11)))
        cv2.rectangle(canvas, (bx - bw, 0), (bx + bw, sz - 1), (0.58, 0.58, 0.56), -1)
        cv2.line(canvas, (bx - bw, 0), (bx - bw, sz - 1), (0.85, 0.85, 0.84), 2)
        cv2.line(canvas, (bx + bw, 0), (bx + bw, sz - 1), (0.85, 0.85, 0.84), 2)
        # Shoreline strip at one side
        shore_x = int(sz * rng.uniform(0.08, 0.22))
        cv2.rectangle(canvas, (0, 0), (shore_x, sz - 1), (0.62, 0.58, 0.44), -1)
    elif scene_kind == "beach_coastline":
        # Ocean waves + white surf foam + sandy beach + coastal avenue
        split_x = int(sz * rng.uniform(0.35, 0.65))
        canvas[:, :split_x, :] = np.array([0.11, 0.25, 0.38], dtype=np.float32)
        canvas[:, split_x : min(sz, split_x + 12), :] = np.array([0.84, 0.87, 0.89], dtype=np.float32)
        beach_end = min(sz, split_x + int(sz * rng.uniform(0.16, 0.28)))
        canvas[:, min(sz, split_x + 12) : beach_end, :] = np.array([0.74, 0.67, 0.50], dtype=np.float32)
        if beach_end + 16 < sz:
            canvas[:, beach_end : beach_end + 14, :] = 0.22
    elif scene_kind == "urban_boulevard":
        # City block grid + parallel avenues + rectangular roofs
        step = max(32, sz // int(rng.integers(5, 9)))
        for g in range(step // 2, sz, step):
            cv2.rectangle(canvas, (g - 5, 0), (g + 5, sz - 1), (0.24, 0.24, 0.25), -1)
            cv2.rectangle(canvas, (0, g - 4), (sz - 1, g + 4), (0.26, 0.26, 0.27), -1)
        for _ in range(28):
            rx = int(rng.integers(4, sz - 28))
            ry = int(rng.integers(4, sz - 28))
            rw = int(rng.integers(10, 26))
            rh = int(rng.integers(10, 26))
            rc = float(rng.uniform(0.45, 0.78))
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (rc, rc * 0.96, rc * 0.92), -1)
    elif scene_kind == "agricultural_fields":
        # Striped agricultural fields + gravel farm track
        stripe = 0.09 * np.sin(2.0 * math.pi * rng.uniform(14.0, 28.0) * xx)
        field_rgb = np.array(
            [float(rng.uniform(0.24, 0.42)), float(rng.uniform(0.32, 0.52)), float(rng.uniform(0.16, 0.28))],
            dtype=np.float32,
        )
        canvas = np.clip(field_rgb[None, None, :] + stripe[:, :, None], 0.05, 0.85).astype(np.float32)
        track_x = int(sz * rng.uniform(0.25, 0.75))
        cv2.rectangle(canvas, (track_x - 6, 0), (track_x + 6, sz - 1), (0.56, 0.49, 0.36), -1)
    else:
        # Airport surround: parallel taxiway + aprons + grass strips + service roads
        for tx_frac in (0.22, 0.78):
            tx = int(sz * tx_frac)
            cv2.rectangle(canvas, (tx - 7, 0), (tx + 7, sz - 1), (0.25, 0.25, 0.26), -1)
            cv2.line(canvas, (tx, 0), (tx, sz - 1), (0.82, 0.74, 0.22), 1)
    return canvas


def _warp_ground_plane_perspective(
    ortho_tex: np.ndarray,
    roi_size: int,
    *,
    cam_pos_ned: np.ndarray,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fwd: np.ndarray,
    right: np.ndarray,
    s_near_m: float,
    s_far_m: float,
    lat_half_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a top-down ground orthophoto into 3D camera perspective via `cv2.warpPerspective`.

    Returns `(warped_ground_rgb [H, W, 3], ground_mask [H, W] float32 in {0, 1})`.
    """
    th, tw = ortho_tex.shape[:2]
    ground_corners_ned = np.vstack(
        [
            fwd * s_near_m - right * lat_half_m,
            fwd * s_near_m + right * lat_half_m,
            fwd * s_far_m + right * lat_half_m,
            fwd * s_far_m - right * lat_half_m,
        ]
    ).astype(np.float64)
    dst_uv, dst_z = project_runway_points_ned(
        ground_corners_ned, cam_pos_ned, yaw_rad, pitch_rad, roll_rad, fx, fy, cx, cy
    )
    src_uv = np.array(
        [[0.0, float(th - 1)], [float(tw - 1), float(th - 1)], [float(tw - 1), 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    if np.all(dst_z > 5.0) and np.all(np.isfinite(dst_uv)):
        h_mat = cv2.getPerspectiveTransform(src_uv, dst_uv.astype(np.float32))
        warped = cv2.warpPerspective(
            ortho_tex.astype(np.float32),
            h_mat,
            (roi_size, roi_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    else:
        warped = cv2.resize(ortho_tex.astype(np.float32), (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    # Compute exact tilted horizon line from two distant horizontal rays at camera altitude
    h_dist = 50_000.0
    cam_alt_ned = float(cam_pos_ned[2])
    h_left = np.array(
        [
            float(cam_pos_ned[0]) + h_dist * math.cos(yaw_rad - 0.55),
            float(cam_pos_ned[1]) + h_dist * math.sin(yaw_rad - 0.55),
            cam_alt_ned,
        ],
        dtype=np.float64,
    )
    h_right = np.array(
        [
            float(cam_pos_ned[0]) + h_dist * math.cos(yaw_rad + 0.55),
            float(cam_pos_ned[1]) + h_dist * math.sin(yaw_rad + 0.55),
            cam_alt_ned,
        ],
        dtype=np.float64,
    )
    h_uv, h_z = project_runway_points_ned(
        np.vstack([h_left, h_right]), cam_pos_ned, yaw_rad, pitch_rad, roll_rad, fx, fy, cx, cy
    )
    yy, xx = np.mgrid[0:roi_size, 0:roi_size].astype(np.float32)
    if np.all(h_z > 10.0) and abs(float(h_uv[1, 0] - h_uv[0, 0])) > 1.0:
        slope = float((h_uv[1, 1] - h_uv[0, 1]) / (h_uv[1, 0] - h_uv[0, 0]))
        y_hor = float(h_uv[0, 1]) + slope * (xx - float(h_uv[0, 0]))
    else:
        y_hor = np.full_like(yy, 0.35 * float(roi_size))
    ground_mask = (yy >= y_hor).astype(np.float32)
    return warped, ground_mask


def _compose_tilted_sky(
    ground_img: np.ndarray,
    ground_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    domain_randomize: bool = True,
) -> np.ndarray:
    """Composite realistic daylight/golden-hour/overcast sky above the tilted perspective horizon."""
    roi_size = ground_img.shape[0]
    t_sky = np.linspace(0.0, 1.0, roi_size, dtype=np.float32)[:, None, None]
    if domain_randomize:
        sky_mode = int(rng.integers(0, 3))
        if sky_mode == 0:
            sky_top = np.array([0.36, 0.55, 0.85], dtype=np.float32)
            sky_bot = np.array([0.74, 0.83, 0.93], dtype=np.float32)
        elif sky_mode == 1:
            sky_top = np.array([0.34, 0.46, 0.68], dtype=np.float32)
            sky_bot = np.array([0.93, 0.72, 0.50], dtype=np.float32)
        else:
            sky_top = np.array([0.56, 0.60, 0.65], dtype=np.float32)
            sky_bot = np.array([0.79, 0.81, 0.83], dtype=np.float32)
    else:
        sky_top = np.array([0.46, 0.62, 0.84], dtype=np.float32)
        sky_bot = np.array([0.66, 0.76, 0.88], dtype=np.float32)
    sky_grad = sky_top[None, None, :] * (1.0 - t_sky) + sky_bot[None, None, :] * t_sky
    m = ground_mask[:, :, None]
    return np.clip(ground_img * m + sky_grad * (1.0 - m), 0.0, 1.0).astype(np.float32)


def _render_perspective_non_runway_negative(
    roi_size: int,
    rng: np.random.Generator,
    *,
    naip_bg_textures: Sequence[NaipRunwayPatch | np.ndarray] | None = None,
) -> dict[str, Any]:
    """Render a realistic 3D perspective-warped non-runway negative (`y_present = 0.0`).

    Strictly avoids blending airport/runway NAIP tiles into non-runway negatives so the
    presence head is never trained to reject real airport runway imagery.
    """
    scene_types = (
        "interstate_highway",
        "coastal_bridge",
        "beach_coastline",
        "urban_boulevard",
        "agricultural_fields",
    )
    scene_kind = scene_types[int(rng.integers(0, len(scene_types)))]
    ortho = _build_procedural_ortho_canvas(rng, canvas_size=max(256, roi_size), scene_kind=scene_kind)
    # Never contaminate non-runway negatives with georeferenced airport runway NAIP patches
    del naip_bg_textures

    hdg_deg = float(rng.uniform(0.0, 360.0))
    hdg = math.radians(hdg_deg)
    fwd = np.array([math.cos(hdg), math.sin(hdg), 0.0], dtype=np.float64)
    right = np.array([-math.sin(hdg), math.cos(hdg), 0.0], dtype=np.float64)
    slant_m = float(math.exp(rng.uniform(math.log(200.0), math.log(2200.0))))
    alt_m = float(rng.uniform(35.0, 350.0))
    lat_m = float(rng.uniform(-120.0, 120.0))
    cam_pos_ned = -fwd * slant_m + right * lat_m + np.array([0.0, 0.0, -alt_m], dtype=np.float64)

    yaw_rad = hdg + math.radians(float(rng.uniform(-28.0, 28.0)))
    pitch_rad = math.radians(float(rng.uniform(-14.0, -2.5)))
    roll_rad = math.radians(float(rng.uniform(-18.0, 18.0)))
    fov_deg = float(rng.uniform(45.0, 92.0))
    fx = (0.5 * float(roi_size)) / math.tan(0.5 * math.radians(fov_deg))
    fy = fx
    cx = 0.5 * float(roi_size - 1)
    cy = 0.5 * float(roi_size - 1)

    s_near = max(40.0, -slant_m + 120.0)
    s_far = s_near + float(rng.uniform(900.0, 2400.0))
    warped, gmask = _warp_ground_plane_perspective(
        ortho,
        roi_size,
        cam_pos_ned=cam_pos_ned,
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fwd=fwd,
        right=right,
        s_near_m=s_near,
        s_far_m=s_far,
        lat_half_m=float(rng.uniform(220.0, 600.0)),
    )
    img = _compose_tilted_sky(warped, gmask, rng, domain_randomize=True)
    noise = rng.normal(0.0, float(rng.uniform(0.008, 0.028)), size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    if float(rng.random()) < 0.55:
        u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(45, 90))])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None and dec.shape == u8.shape:
                img = dec.astype(np.float32) / 255.0

    dummy_kl = np.zeros((NUM_KEYLINES, 3), dtype=np.float32)
    dummy_kl[:, 1] = 1.0
    return {
        "image": np.transpose(img, (2, 0, 1)).astype(np.float32),
        "kp_xy": np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32),
        "keylines_hesse": dummy_kl,
        "runway_present": 0.0,
        "negative_scene_type": scene_kind,
    }


def apply_v53_severe_corruption(
    img_batch: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """GPU-native vectorized PyTorch corruptions (glare, 25-60% occlusion, motion blur, rain, JPEG-proxy).

    Executes strictly on `img_batch.device` with zero CPU/OpenCV transfers, eliminating the
    single-threaded CPU Stage-2B stall while training calibrated `cov_head` & `sigma_head`.
    """
    dev = img_batch.device
    dt = img_batch.dtype
    x = img_batch.detach().float().clamp(0.0, 1.0)
    b_sz, c_sz, h, w = x.shape
    if b_sz == 0:
        return img_batch

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(0, 2**31 - 1)))

    modes = torch.randint(0, 4, (b_sz,), generator=gen, device=dev)
    u_probs = torch.rand((b_sz, 4), generator=gen, device=dev)
    m_glare = (modes == 0) | (u_probs[:, 0] < 0.45)
    m_occ = (modes == 1) | (u_probs[:, 1] < 0.50)
    m_blur = (modes == 2) | (u_probs[:, 2] < 0.40)
    m_rain_jpeg = (modes == 3) | (u_probs[:, 3] < 0.40)

    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)

    # 1. Vectorized directional sun-glare bloom saturating 15-45% of frame
    if bool(m_glare.any().item()):
        gx = (0.15 + 0.70 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)) * float(w)
        gy = (0.08 + 0.52 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)) * float(h)
        sig_g = (0.22 + 0.20 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)) * float(max(h, w))
        amp_g = 0.65 + 0.70 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        rad2 = (xx - gx).pow(2) + (yy - gy).pow(2)
        glare = amp_g * torch.exp(-rad2 / (2.0 * sig_g * sig_g))
        x = torch.where(m_glare.view(b_sz, 1, 1, 1), (x + glare).clamp(0.0, 1.0), x)

    # 2. Vectorized 25-60% rectangular/threshold cutout occlusion
    if bool(m_occ.any().item()):
        occ_frac = 0.25 + 0.35 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        occ_side = torch.sqrt(occ_frac)
        oh = (occ_side * float(h)).clamp(min=8.0, max=float(h))
        ow = (occ_side * float(w)).clamp(min=8.0, max=float(w))
        oy0 = torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev) * (float(h) - oh).clamp(min=0.0)
        ox0 = torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev) * (float(w) - ow).clamp(min=0.0)
        in_box = (yy >= oy0) & (yy < (oy0 + oh)) & (xx >= ox0) & (xx < (ox0 + ow))
        is_dark = torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev) < 0.60
        dark_v = 0.02 + 0.20 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        bright_v = 0.75 + 0.21 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        fill_v = torch.where(is_dark, dark_v, bright_v)
        occ_mask = m_occ.view(b_sz, 1, 1, 1) & in_box
        x = torch.where(occ_mask, fill_v.expand_as(x), x)

    # 3. Vectorized per-sample anisotropic directional motion blur (grouped F.conv2d on GPU)
    if bool(m_blur.any().item()):
        k_sz = 15
        half = float(k_sz // 2)
        ky = torch.arange(k_sz, device=dev, dtype=torch.float32) - half
        kx = torch.arange(k_sz, device=dev, dtype=torch.float32) - half
        gy_k, gx_k = torch.meshgrid(ky, kx, indexing="ij")  # [K, K]
        angles = torch.rand((b_sz, 1, 1), generator=gen, device=dev) * math.pi
        half_len = 3.0 + 4.5 * torch.rand((b_sz, 1, 1), generator=gen, device=dev)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        along = gx_k.unsqueeze(0) * cos_a + gy_k.unsqueeze(0) * sin_a
        perp = -gx_k.unsqueeze(0) * sin_a + gy_k.unsqueeze(0) * cos_a
        k_raw = torch.exp(-(perp.pow(2)) / 0.65) * (along.abs() <= half_len).float()
        k_norm = k_raw / k_raw.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0e-6)  # [B, K, K]
        k_grp = k_norm.unsqueeze(1).expand(b_sz, c_sz, k_sz, k_sz).reshape(b_sz * c_sz, 1, k_sz, k_sz)
        x_pad = F.pad(x.reshape(1, b_sz * c_sz, h, w), (k_sz // 2, k_sz // 2, k_sz // 2, k_sz // 2), mode="reflect")
        blurred = F.conv2d(x_pad, k_grp, groups=b_sz * c_sz).reshape(b_sz, c_sz, h, w)
        x = torch.where(m_blur.view(b_sz, 1, 1, 1), blurred.clamp(0.0, 1.0), x)

    # 4. Vectorized diagonal rain streaks + 8x8 block-quantized JPEG Q25-50 proxy on GPU
    if bool(m_rain_jpeg.any().item()):
        seeds = (torch.rand((b_sz, 1, h, w), generator=gen, device=dev) < 0.0022).float()
        rain_k = torch.zeros((1, 1, 15, 5), device=dev, dtype=torch.float32)
        for ri in range(15):
            ci = min(4, max(0, int(round(ri * 4.0 / 14.0))))
            rain_k[0, 0, ri, ci] = 1.0
        streaks = F.conv2d(seeds, rain_k, padding=(7, 2)).clamp(0.0, 1.0)
        rain_col = 0.72 + 0.23 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        x_rain = (x * (1.0 - 0.65 * streaks) + rain_col * (0.65 * streaks)).clamp(0.0, 1.0)
        # 8x8 block subsampling + coarse luminance/chroma quantization proxy for JPEG Q25-50
        blk_h = max(1, h // 8)
        blk_w = max(1, w // 8)
        blk = F.interpolate(
            F.adaptive_avg_pool2d(x_rain, (blk_h, blk_w)),
            size=(h, w),
            mode="nearest",
        )
        q_levels = 14.0 + 18.0 * torch.rand((b_sz, 1, 1, 1), generator=gen, device=dev)
        x_jpeg = (0.72 * x_rain + 0.28 * blk)
        x_jpeg = (torch.round(x_jpeg * q_levels) / q_levels).clamp(0.0, 1.0)
        x = torch.where(m_rain_jpeg.view(b_sz, 1, 1, 1), x_jpeg, x)

    noise = torch.randn(x.shape, generator=gen, device=dev, dtype=torch.float32) * 0.03
    return (x + noise).clamp(0.0, 1.0).to(dtype=dt)


def _render_runway_crop_at_pose(
    runway: RunwaySpec,
    roi_size: int,
    rng: np.random.Generator,
    *,
    cam_pos_ned: np.ndarray,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    slant_m: float,
    domain_randomize: bool = False,
    rich_markings: bool = True,
    naip_bg_textures: Sequence[NaipRunwayPatch | np.ndarray] | None = None,
    use_perspective_ortho_warp: bool = False,
) -> dict[str, Any]:
    """Render a calibrated 3D perspective runway crop at an exact NED camera pose."""
    pts_ned, kl_pairs = build_ksfo_28l_reference_features(
        runway_width_m=runway.width_m,
        runway_length_m=runway.length_m,
        true_heading_deg=runway.heading_deg,
    )
    hdg = math.radians(runway.heading_deg)
    fwd = np.array([math.cos(hdg), math.sin(hdg), 0.0], dtype=np.float64)
    right = np.array([-math.sin(hdg), math.cos(hdg), 0.0], dtype=np.float64)
    hw = 0.5 * float(runway.width_m)

    uv_px, z_cam = project_runway_points_ned(
        pts_ned, cam_pos_ned, yaw_rad, pitch_rad, roll_rad, fx, fy, cx, cy
    )
    uv_clamped = np.clip(uv_px, 2.0, float(roi_size - 3))

    # Ground surround: real georeferenced 3D cv2.warpPerspective NAIP ortho (extended) or procedural surround
    matched_geo_naip: NaipRunwayPatch | None = None
    if naip_bg_textures:
        for item in naip_bg_textures:
            if isinstance(item, NaipRunwayPatch) and item.icao == runway.icao:
                matched_geo_naip = item
                break
    use_naip_tex = bool(naip_bg_textures) and (
        matched_geo_naip is not None or (not domain_randomize) or (float(rng.random()) < 0.80)
    )
    horizon_v = int(max(4, min(roi_size // 2, np.min(uv_clamped[:, 1]) - 4.0)))
    used_georef_naip = False
    if use_perspective_ortho_warp:
        s_near = max(-slant_m + 35.0, -200.0)
        s_far = float(runway.length_m) + 300.0
        lat_half = max(260.0, float(runway.width_m) * 6.0)
        if matched_geo_naip is not None:
            ortho_base = matched_geo_naip.image_rgb.astype(np.float32)
            s_near = float(matched_geo_naip.s_near_m)
            s_far = float(matched_geo_naip.s_far_m)
            lat_half = float(matched_geo_naip.lat_half_m)
            used_georef_naip = True
        elif use_naip_tex and naip_bg_textures:
            chosen_tex = naip_bg_textures[int(rng.integers(0, len(naip_bg_textures)))]
            if isinstance(chosen_tex, NaipRunwayPatch):
                ortho_base = chosen_tex.image_rgb.astype(np.float32)
                s_near = float(chosen_tex.s_near_m)
                s_far = float(chosen_tex.s_far_m)
                lat_half = float(chosen_tex.lat_half_m)
                used_georef_naip = True
            else:
                ortho_base = np.asarray(chosen_tex, dtype=np.float32)
        else:
            ortho_base = _build_procedural_ortho_canvas(rng, canvas_size=max(256, roi_size), scene_kind="airport_surround")
        warped_ground, gmask = _warp_ground_plane_perspective(
            ortho_base,
            roi_size,
            cam_pos_ned=cam_pos_ned,
            yaw_rad=yaw_rad,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fwd=fwd,
            right=right,
            s_near_m=s_near,
            s_far_m=s_far,
            lat_half_m=lat_half,
        )
        img = _compose_tilted_sky(warped_ground, gmask, rng, domain_randomize=domain_randomize)
    elif use_naip_tex and naip_bg_textures:
        chosen_tex = naip_bg_textures[int(rng.integers(0, len(naip_bg_textures)))]
        tex = chosen_tex.image_rgb if isinstance(chosen_tex, NaipRunwayPatch) else chosen_tex
        img = cv2.resize(tex, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    elif domain_randomize:
        yy, xx = np.mgrid[0:roi_size, 0:roi_size].astype(np.float32) / float(max(1, roi_size - 1))
        base_col = rng.uniform(0.16, 0.44, size=(1, 1, 3)).astype(np.float32)
        pat = (
            0.06 * np.sin(2.0 * math.pi * (rng.uniform(2.0, 7.0) * xx + rng.uniform(1.0, 4.0) * yy))
            + 0.05 * np.cos(2.0 * math.pi * (rng.uniform(3.0, 9.0) * yy))
        ).astype(np.float32)
        img = np.clip(base_col + pat[:, :, None], 0.05, 0.85).astype(np.float32)
    else:
        bg_color = rng.uniform(0.20, 0.42, size=(1, 1, 3)).astype(np.float32)
        img = np.broadcast_to(bg_color, (roi_size, roi_size, 3)).copy()

    # Horizon sky gradient (daylight / golden-hour / overcast domain randomization) for non-warped path
    if not use_perspective_ortho_warp:
        if domain_randomize:
            sky_mode = int(rng.integers(0, 3))
            if sky_mode == 0:
                sky_top = np.array([0.38, 0.56, 0.84], dtype=np.float32)
                sky_bot = np.array([0.72, 0.82, 0.92], dtype=np.float32)
            elif sky_mode == 1:  # Golden hour warm horizon
                sky_top = np.array([0.36, 0.48, 0.68], dtype=np.float32)
                sky_bot = np.array([0.92, 0.72, 0.52], dtype=np.float32)
            else:  # Overcast / marine layer
                sky_top = np.array([0.58, 0.62, 0.66], dtype=np.float32)
                sky_bot = np.array([0.78, 0.80, 0.82], dtype=np.float32)
            t_sky = np.linspace(0.0, 1.0, max(1, horizon_v), dtype=np.float32)[:, None, None]
            img[:horizon_v, :, :] = sky_top[None, None, :] * (1.0 - t_sky) + sky_bot[None, None, :] * t_sky
        else:
            sky_rgb = np.array([0.52, 0.66, 0.82], dtype=np.float32)
            img[:horizon_v, :, :] = sky_rgb[None, None, :]

    # Runway surface: preserve true georeferenced NAIP runway pixels directly (NO 70% fake gray overwrite!)
    poly = np.rint(np.stack([uv_clamped[0], uv_clamped[2], uv_clamped[11], uv_clamped[9]])).astype(np.int32)
    rwy_val = float(rng.uniform(0.14, 0.30))
    if used_georef_naip:
        # Keep genuine NAIP runway surface intact; only subtle exposure jitter
        exp_gain = float(rng.uniform(0.92, 1.08))
        img = np.clip(img * exp_gain, 0.0, 1.0).astype(np.float32)
    elif use_perspective_ortho_warp:
        # Procedural ortho fallback: blend textured asphalt surface
        rwy_mask = np.zeros((roi_size, roi_size), dtype=np.uint8)
        cv2.fillConvexPoly(rwy_mask, poly, 255)
        asphalt_grain = rng.normal(0.0, 0.025, size=(roi_size, roi_size, 1)).astype(np.float32)
        asphalt_tint = np.array(
            [rwy_val, rwy_val * float(rng.uniform(0.96, 1.03)), rwy_val * float(rng.uniform(0.95, 1.05))],
            dtype=np.float32,
        )
        asphalt_tex = np.clip(0.30 * img + 0.70 * asphalt_tint[None, None, :] + asphalt_grain, 0.05, 0.75)
        m_rwy = (rwy_mask > 0)[:, :, None]
        img = np.where(m_rwy, asphalt_tex, img).astype(np.float32)
    else:
        cv2.fillConvexPoly(img, poly, color=(rwy_val, rwy_val, rwy_val))

    if rich_markings and not used_georef_naip:
        # Subtle worn shoulder border along left/right edges (edge wear)
        wear_alpha = float(rng.uniform(0.03, 0.08))
        wear_col = max(0.06, rwy_val - wear_alpha) if float(rng.random()) < 0.5 else min(0.55, rwy_val + wear_alpha)
        for edge_pair in ((0, 9, 1, 10), (2, 11, 1, 10)):
            e0, e1, c0, c1 = edge_pair
            strip = np.rint(
                np.stack(
                    [
                        uv_clamped[e0],
                        0.90 * uv_clamped[e0] + 0.10 * uv_clamped[c0],
                        0.90 * uv_clamped[e1] + 0.10 * uv_clamped[c1],
                        uv_clamped[e1],
                    ]
                )
            ).astype(np.int32)
            cv2.fillConvexPoly(img, strip, color=(wear_col, wear_col, wear_col))

        # Touchdown rubber skid darkening between threshold and aiming zone
        tdz_poly = np.rint(
            np.stack(
                [
                    0.78 * uv_clamped[0] + 0.22 * uv_clamped[1],
                    0.78 * uv_clamped[2] + 0.22 * uv_clamped[1],
                    0.78 * uv_clamped[5] + 0.22 * uv_clamped[4],
                    0.78 * uv_clamped[3] + 0.22 * uv_clamped[4],
                ]
            )
        ).astype(np.int32)
        skid_val = max(0.05, rwy_val - float(rng.uniform(0.05, 0.11)))
        cv2.fillConvexPoly(img, tdz_poly, color=(skid_val, skid_val, skid_val))

        # Vectorized 3D FAA markings: threshold piano keys + twin aiming-point bars + dashed centerline
        mark_pts_3d: list[np.ndarray] = []
        # 1. Painted threshold piano keys (8 stripes: 4 left + 4 right of centerline, s in [6 m, 44 m])
        n_keys_side = 4
        key_s0, key_s1 = 6.0, 44.0
        for side in (-1.0, 1.0):
            for k_i in range(n_keys_side):
                lat_in = side * hw * (0.14 + 0.19 * k_i)
                lat_out = side * hw * (0.14 + 0.19 * k_i + 0.10)
                l0, l1 = (lat_in, lat_out) if lat_in < lat_out else (lat_out, lat_in)
                mark_pts_3d.extend(
                    [
                        fwd * key_s0 + right * l0,
                        fwd * key_s0 + right * l1,
                        fwd * key_s1 + right * l1,
                        fwd * key_s1 + right * l0,
                    ]
                )
        n_piano_quads = 2 * n_keys_side

        # 2. Twin aiming-point bars (s in [295 m, 345 m])
        aim_s0, aim_s1 = 295.0, min(345.0, float(runway.length_m) * 0.45)
        for side in (-1.0, 1.0):
            l0, l1 = (side * hw * 0.24, side * hw * 0.56)
            if l0 > l1:
                l0, l1 = l1, l0
            mark_pts_3d.extend(
                [
                    fwd * aim_s0 + right * l0,
                    fwd * aim_s0 + right * l1,
                    fwd * aim_s1 + right * l1,
                    fwd * aim_s1 + right * l0,
                ]
            )
        n_aim_quads = 2

        # 3. Dashed centerline (FAA 36 m dash, 24 m gap from 65 m up to min(2200 m, L-60 m))
        dash_start_idx = len(mark_pts_3d)
        max_dash_s = min(float(runway.length_m) - 60.0, 2200.0)
        s_cur = 65.0
        n_dashes = 0
        while s_cur + 36.0 <= max_dash_s and n_dashes < 32:
            mark_pts_3d.append(fwd * s_cur)
            mark_pts_3d.append(fwd * (s_cur + 36.0))
            s_cur += 60.0
            n_dashes += 1

        if mark_pts_3d:
            m_uv, m_z = project_runway_points_ned(
                np.vstack(mark_pts_3d), cam_pos_ned, yaw_rad, pitch_rad, roll_rad, fx, fy, cx, cy
            )
            paint_c = float(rng.uniform(0.76, 0.95)) if domain_randomize else 0.88
            paint_rgb = (paint_c, paint_c, paint_c * 0.98)
            for q_idx in range(n_piano_quads + n_aim_quads):
                base = q_idx * 4
                if np.all(m_z[base : base + 4] > 5.0):
                    q_poly = np.rint(m_uv[base : base + 4]).astype(np.int32)
                    cv2.fillConvexPoly(img, q_poly, color=paint_rgb)
            dash_thick = 2 if slant_m < 650.0 else 1
            for d_idx in range(n_dashes):
                b0 = dash_start_idx + d_idx * 2
                if m_z[b0] > 5.0 and m_z[b0 + 1] > 5.0:
                    dp0 = (int(round(m_uv[b0, 0])), int(round(m_uv[b0, 1])))
                    dp1 = (int(round(m_uv[b0 + 1, 0])), int(round(m_uv[b0 + 1, 1])))
                    cv2.line(img, dp0, dp1, color=paint_rgb, thickness=dash_thick, lineType=cv2.LINE_AA)

    # Runway white markings (edges, threshold bar, aiming bar; dashed centerline already drawn if rich)
    for idx_l, (i0, i1) in enumerate(kl_pairs):
        if rich_markings and idx_l == 2:
            continue
        p0 = (int(round(uv_clamped[i0, 0])), int(round(uv_clamped[i0, 1])))
        p1 = (int(round(uv_clamped[i1, 0])), int(round(uv_clamped[i1, 1])))
        thick = 2 if idx_l in (3, 4) else 1
        # Simulate slight edge wear opacity variation under domain randomization
        mark_c = float(rng.uniform(0.70, 0.95)) if domain_randomize else 0.90
        cv2.line(img, p0, p1, color=(mark_c, mark_c, mark_c * 0.98), thickness=thick, lineType=cv2.LINE_AA)

    if domain_randomize:
        # Cross-sun directional illumination gradient + occasional cross-runway shadow band
        yy_s, xx_s = np.mgrid[0:roi_size, 0:roi_size].astype(np.float32) / float(max(1, roi_size - 1))
        sun_ang = float(rng.uniform(0.0, 2.0 * math.pi))
        sun_proj = (xx_s - 0.5) * math.cos(sun_ang) + (yy_s - 0.5) * math.sin(sun_ang)
        illum = 1.0 + float(rng.uniform(-0.14, 0.14)) * sun_proj
        img = img * illum[:, :, None]
        if float(rng.random()) < 0.45:
            sh_center = float(rng.uniform(-0.35, 0.35))
            sh_width = float(rng.uniform(0.08, 0.24))
            sh_depth = float(rng.uniform(0.14, 0.32))
            sh_mask = np.exp(-0.5 * ((sun_proj - sh_center) / sh_width) ** 2)
            # Apply shadow primarily to ground/runway below horizon
            sh_mask[:horizon_v, :] = 0.0
            img = img * (1.0 - sh_depth * sh_mask[:, :, None])

        # Coastal fog / atmospheric haze veil
        haze_alpha = float(rng.uniform(0.0, 0.35))
        if haze_alpha > 0.02:
            veil = float(rng.uniform(0.72, 0.90))
            img = img * (1.0 - haze_alpha) + veil * haze_alpha
        # Occasional directional sun glare bloom near upper sky
        if float(rng.random()) < 0.35:
            gx = float(rng.uniform(0.15, 0.85)) * float(roi_size)
            gy = float(rng.uniform(0.05, 0.35)) * float(roi_size)
            yy_g, xx_g = np.mgrid[0:roi_size, 0:roi_size].astype(np.float32)
            rad2 = (xx_g - gx) ** 2 + (yy_g - gy) ** 2
            glare = float(rng.uniform(0.15, 0.40)) * np.exp(-rad2 / (2.0 * ((0.18 * roi_size) ** 2)))
            img = img + glare[:, :, None]

    noise_std = float(rng.uniform(0.008, 0.028)) if domain_randomize else 0.015
    noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    if domain_randomize and float(rng.random()) < 0.65:
        # Simulate real cockpit `.mp4`/H.264/JPEG video compression artifacts
        u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        q_jpg = int(rng.integers(48, 92))
        ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), q_jpg])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None and dec.shape == u8.shape:
                img = dec.astype(np.float32) / 255.0

    img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

    # Continuous Hesse keylines `(5, 3)`
    kl_hesse = np.stack(
        [compute_hesse_line_from_points(uv_clamped[i0], uv_clamped[i1]) for i0, i1 in kl_pairs],
        axis=0,
    ).astype(np.float32)

    return {
        "image": img_chw,
        "kp_xy": uv_clamped.astype(np.float32),
        "keylines_hesse": kl_hesse,
        "pts_ned": pts_ned.astype(np.float64),
        "keyline_pairs": kl_pairs,
        "cam_pos_ned": cam_pos_ned.astype(np.float64),
        "yaw_rad": float(yaw_rad),
        "pitch_rad": float(pitch_rad),
        "roll_rad": float(roll_rad),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "range_m": float(slant_m),
        "min_z_cam": float(np.min(z_cam)),
    }


def _render_perspective_runway_crop(
    runway: RunwaySpec,
    roi_size: int,
    rng: np.random.Generator,
    *,
    domain_randomize: bool = False,
    rich_markings: bool = True,
    naip_bg_textures: Sequence[np.ndarray] | None = None,
    v53_mode: bool = False,
    roi_jitter_frac: float = 0.35,
) -> dict[str, Any]:
    """Render a calibrated 3D perspective runway approach crop strictly in NED/CAM coordinates."""
    hdg = math.radians(runway.heading_deg)
    fwd = np.array([math.cos(hdg), math.sin(hdg), 0.0], dtype=np.float64)
    right = np.array([-math.sin(hdg), math.cos(hdg), 0.0], dtype=np.float64)

    if v53_mode:
        # extended wide off-center framing breaking (192, 185) center-template collapse:
        # u in [0.15, 0.85], v in [0.22, 0.80], FOV in [45, 92] deg, roll in [-18, +18] deg, slant 200..2800 m
        slant_m = float(math.exp(rng.uniform(math.log(200.0), math.log(2800.0))))
        gs_deg = float(rng.uniform(2.3, 5.4))
        lat_m = float(rng.uniform(-45.0, 45.0)) * min(1.2, slant_m / 700.0)
        alt_m = max(12.0, slant_m * math.tan(math.radians(gs_deg)) + float(rng.uniform(-6.0, 6.0)))
        roll_deg = float(rng.uniform(-18.0, 18.0))
        fov_deg = float(rng.uniform(45.0, 92.0))
        fx = (0.5 * float(roi_size)) / math.tan(0.5 * math.radians(fov_deg))
        fy = fx
        cx = 0.5 * float(roi_size - 1)
        cy = 0.5 * float(roi_size - 1)

        cam_pos_ned = -fwd * slant_m + right * lat_m + np.array([0.0, 0.0, -alt_m], dtype=np.float64)
        aim_ned = fwd * min(450.0, float(runway.length_m) * 0.25)
        to_aim = aim_ned - cam_pos_ned
        aim_bearing = math.atan2(float(to_aim[1]), float(to_aim[0]))
        aim_horiz = max(10.0, math.hypot(float(to_aim[0]), float(to_aim[1])))
        aim_elev = math.atan2(float(-to_aim[2]), aim_horiz)

        # Target off-center runway centroid across u in [0.15, 0.85], v in [0.22, 0.80]
        u_tgt = float(rng.uniform(0.15, 0.85)) * float(roi_size)
        v_tgt = float(rng.uniform(0.22, 0.80)) * float(roi_size)
        du_rad = math.atan2(u_tgt - cx, fx)
        dv_rad = math.atan2(v_tgt - cy, fy)
        yaw_rad = aim_bearing - du_rad
        pitch_rad = max(math.radians(-22.0), min(math.radians(8.0), aim_elev + dv_rad))
        roll_rad = math.radians(roll_deg)

        sample = _render_runway_crop_at_pose(
            runway,
            roi_size,
            rng,
            cam_pos_ned=cam_pos_ned,
            yaw_rad=yaw_rad,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            slant_m=slant_m,
            domain_randomize=True,
            rich_markings=True,
            naip_bg_textures=naip_bg_textures,
            use_perspective_ortho_warp=True,
        )
        # Stage-1-simulated random ROI crop jitter (+/-35% shift, 0.5x..1.8x scale)
        if float(roi_jitter_frac) > 0.0:
            j_scale = float(rng.uniform(0.60, 1.55))
            j_dx = float(rng.uniform(-roi_jitter_frac, roi_jitter_frac)) * float(roi_size)
            j_dy = float(rng.uniform(-roi_jitter_frac, roi_jitter_frac)) * float(roi_size)
            kp_orig = sample["kp_xy"].astype(np.float32)
            # Apply affine transform about ROI center
            m_aff = np.array(
                [
                    [j_scale, 0.0, (1.0 - j_scale) * cx + j_dx],
                    [0.0, j_scale, (1.0 - j_scale) * cy + j_dy],
                ],
                dtype=np.float32,
            )
            kp_cand = np.empty_like(kp_orig)
            kp_cand[:, 0] = m_aff[0, 0] * kp_orig[:, 0] + m_aff[0, 2]
            kp_cand[:, 1] = m_aff[1, 1] * kp_orig[:, 1] + m_aff[1, 2]
            in_bounds = (
                (kp_cand[:, 0] >= 2.0)
                & (kp_cand[:, 0] <= float(roi_size - 3))
                & (kp_cand[:, 1] >= 2.0)
                & (kp_cand[:, 1] <= float(roi_size - 3))
            )
            if int(np.sum(in_bounds)) >= 8:
                img_hwc = np.transpose(sample["image"], (1, 2, 0))
                warped_j = cv2.warpAffine(
                    img_hwc,
                    m_aff,
                    (roi_size, roi_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                kp_clamped = np.clip(kp_cand, 2.0, float(roi_size - 3)).astype(np.float32)
                kl_new = np.stack(
                    [
                        compute_hesse_line_from_points(kp_clamped[i0], kp_clamped[i1])
                        for i0, i1 in sample["keyline_pairs"]
                    ],
                    axis=0,
                ).astype(np.float32)
                sample["image"] = np.transpose(np.clip(warped_j, 0.0, 1.0), (2, 0, 1)).astype(np.float32)
                sample["kp_xy"] = kp_clamped
                sample["keylines_hesse"] = kl_new
        sample["runway_present"] = 1.0
        return sample

    if domain_randomize:
        # Realistic final approach glideslope: 250 m..2500 m slant range, 2.7..4.8 deg GS, +/-10 deg bank/crab
        slant_m = float(math.exp(rng.uniform(math.log(250.0), math.log(2500.0))))
        gs_deg = float(rng.uniform(2.7, 4.8))
        lat_m = float(rng.uniform(-26.0, 26.0)) * min(1.0, slant_m / 800.0)
        alt_m = max(14.0, slant_m * math.tan(math.radians(gs_deg)) + float(rng.uniform(-5.0, 5.0)))
        crab_deg = float(rng.uniform(-8.0, 8.0))
        roll_deg = float(rng.uniform(-10.0, 10.0))
        pitch_deg = -gs_deg - float(rng.uniform(0.2, 2.0))
    else:
        slant_m = float(rng.uniform(450.0, 1500.0))
        lat_m = float(rng.uniform(-16.0, 16.0)) * min(1.0, slant_m / 800.0)
        alt_m = max(25.0, slant_m * math.tan(math.radians(3.1)) + float(rng.uniform(-5.0, 5.0)))
        crab_deg = float(rng.uniform(-2.5, 2.5))
        roll_deg = float(rng.uniform(-3.0, 3.0))
        pitch_deg = float(rng.uniform(-5.0, -2.8))

    cam_pos_ned = -fwd * slant_m + right * lat_m + np.array([0.0, 0.0, -alt_m], dtype=np.float64)

    # Center yaw loosely toward aiming zone so all 12 runway keypoints stay cleanly framed
    aim_ned = fwd * 300.0
    to_aim = aim_ned - cam_pos_ned
    aim_bearing = math.atan2(float(to_aim[1]), float(to_aim[0]))
    yaw_rad = aim_bearing + math.radians(crab_deg * 0.35) if domain_randomize else (aim_bearing + math.radians(crab_deg * 0.25))
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    fx = float(roi_size) * (float(rng.uniform(1.05, 1.35)) if domain_randomize else 1.15)
    fy = fx
    cx = 0.5 * float(roi_size - 1)
    cy = 0.5 * float(roi_size - 1)

    return _render_runway_crop_at_pose(
        runway,
        roi_size,
        rng,
        cam_pos_ned=cam_pos_ned,
        yaw_rad=yaw_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        slant_m=slant_m,
        domain_randomize=domain_randomize,
        rich_markings=rich_markings,
        naip_bg_textures=naip_bg_textures,
    )


def _make_simcc_targets(
    kp_xy: torch.Tensor,
    bins_x: int,
    bins_y: int,
    split_ratio: float,
    sigma_bins: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate normalized 1D Gaussian SimCC probability distributions `[B, K, bins]`."""
    device = kp_xy.device
    gx = torch.arange(bins_x, device=device, dtype=torch.float32).view(1, 1, -1)
    gy = torch.arange(bins_y, device=device, dtype=torch.float32).view(1, 1, -1)
    ux = (kp_xy[..., 0:1].float() * float(split_ratio)).clamp(0.0, float(bins_x - 1))
    vy = (kp_xy[..., 1:2].float() * float(split_ratio)).clamp(0.0, float(bins_y - 1))
    denom = 2.0 * float(sigma_bins) * float(sigma_bins)
    px = torch.exp(-((gx - ux) ** 2) / denom)
    py = torch.exp(-((gy - vy) ** 2) / denom)
    px = px / px.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    py = py / py.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return px, py


def simcc_kl_loss(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    kp_xy_gt: torch.Tensor,
    split_ratio: float = 2.0,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 SimCC 1D KL-divergence keypoint classification loss (supports extended negative masking)."""
    tx, ty = _make_simcc_targets(
        kp_xy_gt,
        bins_x=int(logits_x.shape[-1]),
        bins_y=int(logits_y.shape[-1]),
        split_ratio=split_ratio,
    )
    if sample_mask is None:
        lx = F.kl_div(F.log_softmax(logits_x.float(), dim=-1), tx, reduction="batchmean")
        ly = F.kl_div(F.log_softmax(logits_y.float(), dim=-1), ty, reduction="batchmean")
        return 0.5 * (lx + ly)
    lx_ps = F.kl_div(F.log_softmax(logits_x.float(), dim=-1), tx, reduction="none").sum(dim=(1, 2))
    ly_ps = F.kl_div(F.log_softmax(logits_y.float(), dim=-1), ty, reduction="none").sum(dim=(1, 2))
    m = sample_mask.float().reshape(-1)
    return (0.5 * (lx_ps + ly_ps) * m).sum() / m.sum().clamp(min=1.0)


class PerspectiveRunwayDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Deterministic perspective 3D runway approach dataset for XVS-Land Stage-2A/2B.

    Uses eager in-memory caching for small datasets (`<= 1024`) and memory-safe deterministic
    on-the-fly rendering (`lazy=True`) for large G4 runs (e.g. `24,000-30,000` samples at `384x384`).
    When `v53_mode=True`, emits 4-tuples `(img, kp, kl, runway_present)` with `65%` off-center
    3D perspective-warped runways and `35%` non-runway negatives.
    """

    def __init__(
        self,
        runways: Sequence[RunwaySpec],
        num_samples: int,
        roi_size: int,
        seed: int = 20261121,
        *,
        domain_randomize: bool = False,
        rich_markings: bool = True,
        naip_bg_textures: Sequence[np.ndarray] | None = None,
        lazy: bool | None = None,
        v53_mode: bool = False,
        neg_fraction: float = 0.35,
        roi_jitter_frac: float = 0.35,
    ) -> None:
        self.runways = list(runways)
        self.num_samples = int(num_samples)
        self.roi_size = int(roi_size)
        self.seed = int(seed)
        self.domain_randomize = bool(domain_randomize)
        self.rich_markings = bool(rich_markings)
        self.naip_bg_textures = list(naip_bg_textures) if naip_bg_textures else None
        self.v53_mode = bool(v53_mode)
        self.neg_fraction = float(max(0.0, min(0.95, neg_fraction)))
        self.roi_jitter_frac = float(roi_jitter_frac)
        self._pos_mod_cut = max(1, min(20, int(round(20.0 * (1.0 - self.neg_fraction)))))
        n_rwy = len(self.runways)
        if n_rwy == 0:
            raise ValueError("PerspectiveRunwayDataset requires at least 1 runway spec")

        self.lazy = bool(self.num_samples > 1024) if lazy is None else bool(lazy)
        self.samples: list[dict[str, Any]] = []
        self._ram_cache_imgs: np.ndarray | None = None
        self._ram_cache_kp: np.ndarray | None = None
        self._ram_cache_kl: np.ndarray | None = None
        self._ram_cache_present: np.ndarray | None = None
        if not self.lazy:
            rng = np.random.default_rng(self.seed)
            for idx in range(self.num_samples):
                if self.v53_mode and ((idx % 20) >= self._pos_mod_cut):
                    self.samples.append(
                        _render_perspective_non_runway_negative(
                            self.roi_size,
                            rng,
                            naip_bg_textures=None,
                        )
                    )
                else:
                    rwy = self.runways[idx % n_rwy]
                    self.samples.append(
                        _render_perspective_runway_crop(
                            rwy,
                            self.roi_size,
                            rng,
                            domain_randomize=self.domain_randomize,
                            rich_markings=self.rich_markings,
                            naip_bg_textures=self.naip_bg_textures,
                            v53_mode=self.v53_mode,
                            roi_jitter_frac=self.roi_jitter_frac,
                        )
                    )
        elif self.num_samples <= 32768:
            # Compact uint8 RAM pre-cache on high-RAM hosts (e.g. G4 177 GiB RAM -> 30k x 384px = 12.5 GiB).
            req_bytes = int(self.num_samples) * 3 * int(self.roi_size) * int(self.roi_size)
            total_ram_bytes = 0
            if hasattr(os, "sysconf"):
                try:
                    total_ram_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
                except (ValueError, OSError):
                    total_ram_bytes = 0
            if total_ram_bytes > 0 and req_bytes < int(0.25 * total_ram_bytes):
                from concurrent.futures import ThreadPoolExecutor

                t_cache = time.perf_counter()
                imgs_u8 = np.empty((self.num_samples, 3, self.roi_size, self.roi_size), dtype=np.uint8)
                kps_f32 = np.empty((self.num_samples, NUM_KEYPOINTS, 2), dtype=np.float32)
                kls_f32 = np.empty((self.num_samples, NUM_KEYLINES, 3), dtype=np.float32)
                pres_f32 = np.ones((self.num_samples,), dtype=np.float32)
                n_thr = min(24, max(4, (os.cpu_count() or 4) // 2))

                def _fill_chunk(chunk_range: tuple[int, int]) -> None:
                    c0, c1 = chunk_range
                    for i in range(c0, c1):
                        s = self._render_index(i)
                        imgs_u8[i] = np.clip(s["image"] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
                        kps_f32[i] = s["kp_xy"]
                        kls_f32[i] = s["keylines_hesse"]
                        pres_f32[i] = float(s["runway_present"]) if "runway_present" in s else 1.0

                step = max(64, (self.num_samples + n_thr - 1) // n_thr)
                ranges = [(i, min(self.num_samples, i + step)) for i in range(0, self.num_samples, step)]
                with ThreadPoolExecutor(max_workers=n_thr) as pool:
                    list(pool.map(_fill_chunk, ranges))
                self._ram_cache_imgs = imgs_u8
                self._ram_cache_kp = kps_f32
                self._ram_cache_kl = kls_f32
                self._ram_cache_present = pres_f32
                print(
                    f"[XVS-Land] Pre-cached {self.num_samples}x{self.roi_size}px uint8 images "
                    f"({req_bytes / float(1024**3):.2f} GiB RAM) with {n_thr} threads in "
                    f"{time.perf_counter() - t_cache:.1f}s.",
                    flush=True,
                )

    def __len__(self) -> int:
        return self.num_samples

    def _render_index(self, idx: int) -> dict[str, Any]:
        if not self.lazy:
            return self.samples[idx]
        sample_rng = np.random.default_rng(self.seed + int(idx) * 1_000_003)
        if self.v53_mode and ((idx % 20) >= self._pos_mod_cut):
            return _render_perspective_non_runway_negative(
                self.roi_size,
                sample_rng,
                naip_bg_textures=None,
            )
        rwy = self.runways[idx % len(self.runways)]
        return _render_perspective_runway_crop(
            rwy,
            self.roi_size,
            sample_rng,
            domain_randomize=self.domain_randomize,
            rich_markings=self.rich_markings,
            naip_bg_textures=self.naip_bg_textures,
            v53_mode=self.v53_mode,
            roi_jitter_frac=self.roi_jitter_frac,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if (
            self._ram_cache_imgs is not None
            and self._ram_cache_kp is not None
            and self._ram_cache_kl is not None
        ):
            img_t = torch.from_numpy(self._ram_cache_imgs[idx]).to(dtype=torch.float32).mul_(1.0 / 255.0)
            if self.v53_mode:
                pres = float(self._ram_cache_present[idx]) if self._ram_cache_present is not None else 1.0
                return (
                    img_t,
                    torch.from_numpy(self._ram_cache_kp[idx]),
                    torch.from_numpy(self._ram_cache_kl[idx]),
                    torch.tensor(pres, dtype=torch.float32),
                )
            return (
                img_t,
                torch.from_numpy(self._ram_cache_kp[idx]),
                torch.from_numpy(self._ram_cache_kl[idx]),
            )
        s = self._render_index(idx)
        if self.v53_mode:
            pres = float(s["runway_present"]) if "runway_present" in s else 1.0
            return (
                torch.from_numpy(s["image"]),
                torch.from_numpy(s["kp_xy"]),
                torch.from_numpy(s["keylines_hesse"]),
                torch.tensor(pres, dtype=torch.float32),
            )
        return (
            torch.from_numpy(s["image"]),
            torch.from_numpy(s["kp_xy"]),
            torch.from_numpy(s["keylines_hesse"]),
        )


@torch.no_grad()
def evaluate_xvs_land_approach(
    model: nn.Module,
    val_ds: PerspectiveRunwayDataset,
    calibrator: StudentTCalibrator,
    device: torch.device,
    max_pnpl_eval: int = 12,
    *,
    keyline_sigma_scale: float = 1.0,
    ekf_eval_frames: int = 40,
    ekf_seed: int = 20261199,
) -> dict[str, Any]:
    """Evaluate SimCC RMSE, OKLS angle error, single-frame PnPL, AND 5 Hz ApproachEkf5Hz trajectory."""
    model.eval()
    kp_sq_errs: list[float] = []
    okls_ang_errs: list[float] = []
    pnpl_pos_errs: list[float] = []
    v_hpls: list[float] = []
    v_vpls: list[float] = []
    pred_centroids_u: list[float] = []
    pred_centroids_v: list[float] = []
    corrupt_kpsig_ratios: list[float] = []
    neg_total = 0
    neg_fp = 0
    pos_pres_total = 0
    pos_pres_tp = 0
    pnpl_eval_done = 0
    cat1_ok = 0
    hmi_count = 0
    corrupt_eval_rng = np.random.default_rng(int(ekf_seed) + 404)

    n_total = len(val_ds)
    for idx in range(n_total):
        sample = val_ds._render_index(idx)
        is_pos = (float(sample["runway_present"]) if "runway_present" in sample else 1.0) >= 0.5
        x = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        out = model(x)

        if "runway_present_logit" in out:
            p_pres = float(torch.sigmoid(out["runway_present_logit"].float()).reshape(-1)[0].detach().cpu().item())
            if is_pos:
                pos_pres_total += 1
                if p_pres >= 0.65:
                    pos_pres_tp += 1
            else:
                neg_total += 1
                if p_pres >= 0.65:
                    neg_fp += 1
        elif not is_pos:
            neg_total += 1

        if not is_pos:
            continue

        pred_kp = out["keypoints_xy"][0].detach().cpu().numpy().astype(np.float64)
        gt_kp = sample["kp_xy"].astype(np.float64)
        pred_kl = out["keylines_hesse"][0].detach().cpu().numpy().astype(np.float64)
        gt_kl = sample["keylines_hesse"].astype(np.float64)
        cov_cal = calibrator.calibrate_covariance(out["cov_2x2"])[0].detach().cpu().numpy().astype(np.float64)

        pred_centroids_u.append(float(np.mean(pred_kp[:, 0])))
        pred_centroids_v.append(float(np.mean(pred_kp[:, 1])))

        if len(corrupt_kpsig_ratios) < min(32, max(8, max_pnpl_eval * 2)):
            x_corr = apply_v53_severe_corruption(x, corrupt_eval_rng)
            out_corr = model(x_corr)
            sig_clean = float(
                torch.sqrt(
                    ((out["cov_2x2"].float()[..., 0, 0] + out["cov_2x2"].float()[..., 1, 1]) * 0.5).clamp(min=1e-6)
                ).mean().detach().cpu().item()
            )
            sig_corr = float(
                torch.sqrt(
                    (
                        (out_corr["cov_2x2"].float()[..., 0, 0] + out_corr["cov_2x2"].float()[..., 1, 1]) * 0.5
                    ).clamp(min=1e-6)
                ).mean().detach().cpu().item()
            )
            if math.isfinite(sig_clean) and math.isfinite(sig_corr) and sig_clean > 1e-6:
                corrupt_kpsig_ratios.append(sig_corr / sig_clean)

        pred_kl_sigs: np.ndarray | None = None
        if "keyline_sigma_px" in out:
            raw_sigs = out["keyline_sigma_px"][0].detach().cpu().numpy().astype(np.float64)
            pred_kl_sigs = np.clip(
                raw_sigs * float(keyline_sigma_scale),
                KEYLINE_SIGMA_MIN_PX,
                KEYLINE_SIGMA_MAX_PX,
            )

        kp_sq_errs.append(float(np.mean(np.sum((pred_kp - gt_kp) ** 2, axis=-1))))
        cos_sim = np.clip(pred_kl[:, 1] * gt_kl[:, 1] + pred_kl[:, 2] * gt_kl[:, 2], -1.0, 1.0)
        okls_ang_errs.extend(np.degrees(np.arccos(cos_sim)).tolist())

        if pnpl_eval_done < min(max_pnpl_eval, n_total):
            pnpl_eval_done += 1
            init_pos = sample["cam_pos_ned"] + np.array([15.0, -5.0, 3.0], dtype=np.float64)
            init_yaw = sample["yaw_rad"] + math.radians(1.0)
            sol = solve_4dof_map_pnpl(
                sample["pts_ned"],
                pred_kp,
                cov_cal,
                pitch_rad=sample["pitch_rad"],
                roll_rad=sample["roll_rad"],
                init_pos_ned=init_pos,
                init_yaw_rad=init_yaw,
                fx=sample["fx"],
                fy=sample["fy"],
                cx=sample["cx"],
                cy=sample["cy"],
                keyline_pairs=sample["keyline_pairs"],
                obs_keylines_hesse=pred_kl,
                keyline_sigmas_px=pred_kl_sigs,
                huber_delta_sigma=3.0,
            )
            err_vec = sol.pos_ned_m - sample["cam_pos_ned"]
            pos_err = float(np.linalg.norm(err_vec))
            horiz_err = float(math.hypot(float(err_vec[0]), float(err_vec[1])))
            vert_err = abs(float(err_vec[2]))
            pnpl_pos_errs.append(pos_err)

            raim = compute_hybrid_conformal_raim_pl(
                sol.jacobian_obs,
                sol.cov_4x4,
                range_m=float(sample["range_m"]),
            )
            v_hpls.append(float(raim.v_hpl_m))
            v_vpls.append(float(raim.v_vpl_m))
            if raim.hpl_within_limit and raim.vpl_within_limit:
                cat1_ok += 1
                if horiz_err > raim.v_hpl_m or vert_err > raim.v_vpl_m:
                    hmi_count += 1

    # 5 Hz ApproachEkf5Hz multi-frame trajectory evaluation (1500 m -> 250 m into held-out KSFO 28L)
    ekf_metrics: dict[str, Any] | None = None
    if int(ekf_eval_frames) >= 2:
        ksfo_spec = RunwaySpec(icao="KSFO", length_m=3471.0, width_m=61.0, heading_deg=297.7)
        traj_poses = build_approach_trajectory_poses(
            runway_heading_deg=ksfo_spec.heading_deg,
            start_range_m=1500.0,
            end_range_m=250.0,
            glideslope_deg=3.0,
            num_frames=int(ekf_eval_frames),
            dt=0.2,
        )
        traj_rng = np.random.default_rng(int(ekf_seed))
        fx_traj = float(val_ds.roi_size) * 1.15
        cx_traj = 0.5 * float(val_ds.roi_size - 1)
        ekf_frames: list[dict[str, Any]] = []
        for pose in traj_poses:
            rendered = _render_runway_crop_at_pose(
                ksfo_spec,
                val_ds.roi_size,
                traj_rng,
                cam_pos_ned=pose["cam_pos_ned"],
                yaw_rad=pose["yaw_rad"],
                pitch_rad=pose["pitch_rad"],
                roll_rad=pose["roll_rad"],
                fx=fx_traj,
                fy=fx_traj,
                cx=cx_traj,
                cy=cx_traj,
                slant_m=pose["range_m"],
                domain_randomize=False,
                rich_markings=val_ds.rich_markings,
            )
            x_t = torch.from_numpy(rendered["image"]).unsqueeze(0).to(device)
            out_t = model(x_t)
            obs_uv = out_t["keypoints_xy"][0].detach().cpu().numpy().astype(np.float64)
            obs_kl = out_t["keylines_hesse"][0].detach().cpu().numpy().astype(np.float64)
            obs_cov = calibrator.calibrate_covariance(out_t["cov_2x2"])[0].detach().cpu().numpy().astype(np.float64)
            obs_kl_sigs: np.ndarray | None = None
            if "keyline_sigma_px" in out_t:
                obs_kl_sigs = np.clip(
                    out_t["keyline_sigma_px"][0].detach().cpu().numpy().astype(np.float64) * float(keyline_sigma_scale),
                    KEYLINE_SIGMA_MIN_PX,
                    KEYLINE_SIGMA_MAX_PX,
                )
            ekf_frames.append(
                {
                    "pts_ned": rendered["pts_ned"],
                    "obs_uv": obs_uv,
                    "obs_cov_2x2": obs_cov,
                    "keyline_pairs": rendered["keyline_pairs"],
                    "obs_keylines_hesse": obs_kl,
                    "keyline_sigmas_px": obs_kl_sigs,
                    "cam_pos_ned": pose["cam_pos_ned"],
                    "vel_ned_mps": pose["vel_ned_mps"],
                    "accel_ned_mps2": pose["accel_ned_mps2"],
                    "yaw_rad": pose["yaw_rad"],
                    "pitch_rad": pose["pitch_rad"],
                    "roll_rad": pose["roll_rad"],
                    "yaw_rate_rad_s": pose["yaw_rate_rad_s"],
                    "fx": fx_traj,
                    "fy": fx_traj,
                    "cx": cx_traj,
                    "cy": cx_traj,
                    "range_m": pose["range_m"],
                }
            )
        ekf_res = evaluate_approach_ekf_trajectory(
            ekf_frames,
            dt=0.2,
            huber_delta_sigma=3.0,
            init_pos_offset_ned=np.array([8.0, -4.0, 2.0], dtype=np.float64),
            init_yaw_offset_rad=math.radians(0.5),
        )
        ekf_metrics = {
            "num_frames": ekf_res.num_frames,
            "accepted_updates": ekf_res.accepted_updates,
            "rejected_updates": ekf_res.rejected_updates,
            "ekf_pos_rmse_m": ekf_res.ekf_pos_rmse_m,
            "ekf_horiz_rmse_m": ekf_res.ekf_horiz_rmse_m,
            "ekf_vert_rmse_m": ekf_res.ekf_vert_rmse_m,
            "single_frame_pnpl_rmse_m": ekf_res.single_frame_pnpl_rmse_m,
            "v_hpl_median_m": ekf_res.v_hpl_median_m,
            "v_vpl_median_m": ekf_res.v_vpl_median_m,
            "v_hpl_p95_m": ekf_res.v_hpl_p95_m,
            "v_vpl_p95_m": ekf_res.v_vpl_p95_m,
            "cat1_availability": ekf_res.cat1_availability,
            "hmi_count": ekf_res.hmi_count,
            "start_range_m": ekf_res.start_range_m,
            "end_range_m": ekf_res.end_range_m,
        }

    # Verify parallel-runway / taxiway anti-misassociation gate on KSFO 28L
    assoc_28l = check_runway_association(KSFO_28L_WIDTH_M, 2.0)
    assoc_twy_c = check_runway_association(KSFO_TAXIWAY_C_WIDTH_M, 2.0)
    assoc_28r = check_runway_association(KSFO_28L_WIDTH_M, KSFO_28R_LATERAL_OFFSET_M)

    n_pnpl = max(1, len(pnpl_pos_errs))
    centroid_std_px = (
        float(math.hypot(float(np.std(pred_centroids_u)), float(np.std(pred_centroids_v))))
        if len(pred_centroids_u) >= 2
        else 0.0
    )
    non_runway_fpr = float(neg_fp) / float(neg_total) if neg_total > 0 else None
    runway_present_tpr = float(pos_pres_tp) / float(pos_pres_total) if pos_pres_total > 0 else None
    corrupt_over_clean_kpsig_median = (
        float(np.median(corrupt_kpsig_ratios)) if corrupt_kpsig_ratios else None
    )
    return {
        "simcc_rmse_px": float(math.sqrt(max(0.0, float(np.mean(kp_sq_errs))))) if kp_sq_errs else 0.0,
        "okls_angle_err_deg": float(np.mean(okls_ang_errs)) if okls_ang_errs else 0.0,
        "pnpl_pos_rmse_m": (
            float(math.sqrt(float(np.mean(np.square(pnpl_pos_errs))))) if pnpl_pos_errs else None
        ),
        "v_raim_cat1_availability": float(cat1_ok) / float(n_pnpl),
        "v_hpl_median_m": float(np.median(v_hpls)) if v_hpls else None,
        "v_vpl_median_m": float(np.median(v_vpls)) if v_vpls else None,
        "hmi_count": int(hmi_count),
        "non_runway_fpr": non_runway_fpr,
        "runway_present_tpr": runway_present_tpr,
        "centroid_std_px": centroid_std_px,
        "corrupt_over_clean_kpsig_median": corrupt_over_clean_kpsig_median,
        "ekf_trajectory": ekf_metrics,
        "ekf_pos_rmse_m": ekf_metrics["ekf_pos_rmse_m"] if ekf_metrics is not None else None,
        "ekf_cat1_availability": ekf_metrics["cat1_availability"] if ekf_metrics is not None else None,
        "ekf_hmi_count": int(ekf_metrics["hmi_count"]) if ekf_metrics is not None else 0,
        "student_t_nu": float(calibrator.nu),
        "student_t_scale": float(calibrator.scale),
        "keyline_sigma_scale": float(keyline_sigma_scale),
        "ksfo_anti_misassociation_pass": bool(
            assoc_28l.accepted and (not assoc_twy_c.accepted) and (not assoc_28r.accepted)
        ),
    }


class _XvsLandOnnxWrapper(nn.Module):
    """Flat-tuple ONNX export wrapper for `XvsLandModel`."""

    def __init__(self, core: XvsLandModel) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, roi_crop: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.core(roi_crop)
        return (
            out["keypoints_xy"],
            out["keylines_hesse"],
            out["cov_2x2"],
            out["logits_x"],
            out["logits_y"],
        )


def export_xvs_land_onnx(
    model: XvsLandModel,
    out_onnx_path: Path,
    roi_size: int,
    *,
    split_ratio: float = 2.0,
    opset: int = 17,
    max_diff_tol: float = 1.0e-3,
) -> dict[str, Any]:
    """Export `XvsLandModel` to FP32 ONNX (opset 17) and verify ORT parity `< 1e-3` vs PyTorch."""
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_copy = XvsLandModel(
        in_channels=3,
        roi_width=int(roi_size),
        roi_height=int(roi_size),
        split_ratio=float(split_ratio),
    )
    cpu_copy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    cpu_copy.eval()
    wrapper = _XvsLandOnnxWrapper(cpu_copy).eval()

    rng = np.random.default_rng(20261122)
    x_in = torch.from_numpy(rng.uniform(0.0, 1.0, size=(1, 3, int(roi_size), int(roi_size))).astype(np.float32))

    export_kwargs: dict[str, Any] = {
        "input_names": ["roi_crop"],
        "output_names": ["keypoints_xy", "keylines_hesse", "cov_2x2", "logits_x", "logits_y"],
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(wrapper, (x_in,), str(out_onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (x_in,), str(out_onnx_path), **export_kwargs)

    with torch.no_grad():
        pt_outs = [t.detach().cpu().numpy() for t in wrapper(x_in)]

    max_abs_diff = 0.0
    ort_verified = False
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        sess = ort.InferenceSession(str(out_onnx_path), providers=["CPUExecutionProvider"])
        ort_outs = sess.run(None, {"roi_crop": x_in.numpy()})
        diffs = [float(np.max(np.abs(a - b))) for a, b in zip(pt_outs, ort_outs)]
        max_abs_diff = max(diffs) if diffs else 0.0
        ort_verified = bool(math.isfinite(max_abs_diff) and max_abs_diff < float(max_diff_tol))
        if not ort_verified:
            raise ValueError(
                f"XVS-Land ONNX parity check failed: max_abs_diff={max_abs_diff:.6e} >= {max_diff_tol}"
            )
    except ImportError:
        ort_verified = False

    sha = hashlib.sha256(out_onnx_path.read_bytes()).hexdigest()
    return {
        "path": str(out_onnx_path),
        "bytes": int(out_onnx_path.stat().st_size),
        "sha256": sha,
        "opset": int(opset),
        "roi_size": int(roi_size),
        "max_abs_diff": float(max_abs_diff),
        "ort_verified": bool(ort_verified),
    }


def convert_xvs_land_onnx_selective_fp16(fp32_model: Any) -> Any:
    """Pure-`onnx` Selective FP16/FP32 converter for `XvsLandV52Model` (TRT 10/11 `kSTRONGLY_TYPED`).

    Converts backbone & FPN `Conv`/`ConvTranspose` layers (majority of model weights) to FP16
    with explicit `Cast` boundaries, while strictly keeping stem/CoordConv, ALL normalizations
    (`GroupNormalization`, `InstanceNormalization`, `LayerNormalization`, `BatchNormalization`),
    reductions, `Softmax`, `Exp`, `Sqrt`, `Div`, `Clip`, `Where`, and ALL output heads
    (`simcc_head`, `keyline_head`, `cov_head`) in FP32 for tight numerical parity (`< 1e-2`).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.ModelProto()
    model.CopyFrom(fp32_model)
    graph = model.graph

    graph_output_names = {go.name for go in graph.output}
    alias_map: dict[str, str] = {}
    pruned_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in alias_map:
                node.input[i] = alias_map[inp]
        if (
            node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.output[0] not in graph_output_names
        ):
            alias_map[node.output[0]] = node.input[0]
            continue
        pruned_nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(pruned_nodes)

    inferred = onnx.shape_inference.infer_shapes(model)
    tensor_dtype: dict[str, int] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if vi.type.HasField("tensor_type"):
            tensor_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        tensor_dtype[init.name] = init.data_type

    graph_input_names = {gi.name for gi in graph.input}
    fp32_node_indices: set[int] = set()
    always_fp32_ops = {
        "LayerNormalization",
        "GroupNormalization",
        "InstanceNormalization",
        "BatchNormalization",
        "MatMul",
        "Gemm",
        "Softmax",
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceL2",
        "Pow",
        "Sqrt",
        "Div",
        "Reciprocal",
        "Exp",
        "Log",
        "Erf",
        "Where",
        "Less",
        "Greater",
        "GreaterOrEqual",
        "LessOrEqual",
        "ArgMax",
        "Abs",
        "Sub",
        "MaxPool",
        "Clip",
        "Tanh",
        "Concat",
        "Reshape",
        "Transpose",
        "Shape",
        "Slice",
        "Expand",
    }
    always_fp32_name_tags = (
        "downsample_layers.0",
        "downsample_layers/downsample_layers.0",
        "stages.0",
        "stages/stages.0",
        "lat2",
        "lat3",
        "lat4",
        "lat5",
        "up5_to_4",
        "up4_to_3",
        "up3_to_2",
        "fpn_refine",
        "simcc_head",
        "keyline_head",
        "cov_head",
        "runway_present_head",
        "norm",
        "decode_simcc",
        "sigma_head",
        "endpoint",
        "coord",
    )

    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            continue
        lname = node.name.lower()
        is_forced_op = node.op_type in always_fp32_ops
        is_forced_tag = any(tag in lname for tag in always_fp32_name_tags)
        is_fp16_eligible = node.op_type in {"Conv", "Relu", "Add", "Mul"} and ("backbone" in lname)
        if is_forced_op or is_forced_tag or not is_fp16_eligible:
            fp32_node_indices.add(idx)

    tensor_consumers: dict[str, set[int]] = {}
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                tensor_consumers.setdefault(inp, set()).add(idx)

    def _is_f32_only(tname: str) -> bool:
        consumers = tensor_consumers.get(tname)
        return bool(consumers) and consumers.issubset(fp32_node_indices)

    def _is_f16_only(tname: str) -> bool:
        consumers = tensor_consumers.get(tname)
        return bool(consumers) and consumers.isdisjoint(fp32_node_indices)

    init_is_f16: set[str] = set()
    shared_init_f16_copy: dict[str, str] = {}
    extra_inits: list[onnx.TensorProto] = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if _is_f16_only(init.name):
            arr = numpy_helper.to_array(init)
            arr16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr16, name=init.name))
            init_is_f16.add(init.name)
        elif not _is_f32_only(init.name):
            arr = numpy_helper.to_array(init)
            arr16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
            copy_name = f"{init.name}__f16_copy"
            extra_inits.append(numpy_helper.from_array(arr16, name=copy_name))
            shared_init_f16_copy[init.name] = copy_name
            init_is_f16.add(copy_name)
    graph.initializer.extend(extra_inits)

    for node in graph.node:
        if node.op_type == "Constant" and node.output and _is_f16_only(node.output[0]):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    arr16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr16))
                    init_is_f16.add(node.output[0])

    live_dtype: dict[str, int] = dict(tensor_dtype)
    for name in init_is_f16:
        live_dtype[name] = TensorProto.FLOAT16
    for gi in graph.input:
        live_dtype[gi.name] = TensorProto.FLOAT

    out_orig_names = [o.name for o in graph.output]
    out_internal_map = {n: f"{n}__pre_out_cast" for n in out_orig_names}

    new_nodes: list[onnx.NodeProto] = []
    cast_cache: dict[tuple[str, int], str] = {}
    cast_counter = 0

    def _ensure_dtype(tname: str, target_dt: int) -> str:
        nonlocal cast_counter
        if not tname:
            return tname
        cur_dt = live_dtype.get(tname, TensorProto.FLOAT)
        if cur_dt not in (TensorProto.FLOAT, TensorProto.FLOAT16) or cur_dt == target_dt:
            return tname
        key = (tname, target_dt)
        if key in cast_cache:
            return cast_cache[key]
        suffix = "f16" if target_dt == TensorProto.FLOAT16 else "f32"
        cname = f"{tname}__cast_{suffix}_{cast_counter}"
        cast_counter += 1
        new_nodes.append(
            helper.make_node("Cast", [tname], [cname], name=f"cast_{suffix}_{cast_counter}", to=target_dt)
        )
        live_dtype[cname] = target_dt
        cast_cache[key] = cname
        return cname

    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            new_nodes.append(node)
            continue
        node_dt = TensorProto.FLOAT if idx in fp32_node_indices else TensorProto.FLOAT16

        if node.op_type == "Cast":
            orig_in = node.input[0]
            remapped_in = out_internal_map.get(orig_in, orig_in)
            out_name = node.output[0]
            new_out = out_internal_map.get(out_name, out_name)
            target_to = TensorProto.FLOAT
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i == TensorProto.FLOAT and not _is_f32_only(out_name):
                        attr.i = node_dt
                    target_to = attr.i
            node.ClearField("input")
            node.input.append(remapped_in)
            node.ClearField("output")
            node.output.append(new_out)
            new_nodes.append(node)
            live_dtype[new_out] = target_to
            continue

        updated_inputs: list[str] = []
        for inp in node.input:
            if not inp:
                updated_inputs.append("")
                continue
            src_name = out_internal_map.get(inp, inp)
            if node_dt == TensorProto.FLOAT16 and src_name in shared_init_f16_copy:
                updated_inputs.append(shared_init_f16_copy[src_name])
                continue
            inp_dt = live_dtype.get(src_name, TensorProto.FLOAT)
            if inp_dt in (TensorProto.FLOAT, TensorProto.FLOAT16):
                updated_inputs.append(_ensure_dtype(src_name, node_dt))
            else:
                updated_inputs.append(src_name)

        updated_outputs: list[str] = []
        for out_name in node.output:
            mapped = out_internal_map.get(out_name, out_name)
            updated_outputs.append(mapped)
            orig_dt = tensor_dtype.get(out_name, TensorProto.FLOAT)
            if orig_dt in (TensorProto.FLOAT, TensorProto.FLOAT16):
                live_dtype[mapped] = node_dt
            else:
                live_dtype[mapped] = orig_dt

        node.ClearField("input")
        node.input.extend(updated_inputs)
        node.ClearField("output")
        node.output.extend(updated_outputs)
        new_nodes.append(node)

    for out_name in out_orig_names:
        internal_name = out_internal_map[out_name]
        cur_dt = live_dtype.get(internal_name, TensorProto.FLOAT)
        if cur_dt != TensorProto.FLOAT:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [internal_name],
                    [out_name],
                    name=f"out_cast_f32_{out_name}",
                    to=TensorProto.FLOAT,
                )
            )
        else:
            new_nodes.append(
                helper.make_node("Identity", [internal_name], [out_name], name=f"out_id_{out_name}")
            )

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    graph.ClearField("value_info")
    inferred_out = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_out)
    return inferred_out


def export_xvs_land_v52_onnx(
    model: XvsLandV52Model,
    out_dir: Path,
    roi_size: int,
    *,
    split_ratio: float = 2.0,
    opset: int = 17,
    fp32_tol: float = 1.0e-3,
    fp16_tol: float = 1.0e-2,
    export_fp16: bool = True,
    ort_threads: int = 6,
    bench_warmup: int = 3,
    bench_iters: int = 15,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    """Export `XvsLandV52Model` (v5.2 6-output or extended 7-output) to static Opset-17 FP32 + Selective FP16 ONNX."""
    import onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        str(filename_prefix)
        if filename_prefix is not None
        else ("xvs_land_v53" if model.enable_runway_present_head else "xvs_land_v52")
    )
    fp32_path = out_dir / f"{prefix}_fp32.onnx"
    fp16_path = out_dir / f"{prefix}_fp16.onnx"

    cpu_copy = XvsLandV52Model(
        in_channels=model.in_channels,
        roi_width=int(roi_size),
        roi_height=int(roi_size),
        split_ratio=float(split_ratio),
        dims=model.dims,
        depths=model.depths,
        fpn_dim=model.fpn_dim,
        fpn_stride=model.fpn_stride,
        enable_runway_present_head=model.enable_runway_present_head,
        coord_dropout_prob=0.0,
        use_endpoint_prior=model.use_endpoint_prior,
        enable_simcc_uncertainty_cues=model.enable_simcc_uncertainty_cues,
    )
    cpu_copy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    cpu_copy.eval()
    wrapper = XvsLandV52OnnxWrapper(cpu_copy).eval()
    total_params = sum(int(p.numel()) for p in cpu_copy.parameters())

    rng = np.random.default_rng(20261124)
    # Deterministic structured runway-like probe (avoids degenerate p0==p1 collapse on uniform noise)
    yy_p, xx_p = np.meshgrid(
        np.linspace(0.0, 1.0, int(roi_size), dtype=np.float32),
        np.linspace(0.0, 1.0, int(roi_size), dtype=np.float32),
        indexing="ij",
    )
    probe = np.stack(
        [
            0.35 + 0.35 * yy_p + 0.15 * np.abs(xx_p - 0.5),
            0.40 + 0.25 * yy_p + 0.20 * ((np.abs(xx_p - 0.5) < (0.04 + 0.14 * yy_p)) & (yy_p > 0.25)).astype(np.float32),
            0.55 - 0.25 * yy_p,
        ],
        axis=0,
    )
    probe = np.clip(probe + rng.normal(0.0, 0.03, size=probe.shape).astype(np.float32), 0.0, 1.0)
    x_in = torch.from_numpy(probe[None, ...])
    out_names = list(wrapper.output_names)

    export_kwargs: dict[str, Any] = {
        "input_names": ["roi_crop"],
        "output_names": out_names,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(wrapper, (x_in,), str(fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (x_in,), str(fp32_path), **export_kwargs)

    fp32_proto = onnx.load(str(fp32_path))
    onnx.checker.check_model(fp32_proto)

    fp16_bytes: int | None = None
    fp16_sha: str | None = None
    fp16_init_count = 0
    if export_fp16:
        fp16_proto = convert_xvs_land_onnx_selective_fp16(fp32_proto)
        onnx.save(fp16_proto, str(fp16_path))
        fp16_bytes = int(fp16_path.stat().st_size)
        fp16_sha = hashlib.sha256(fp16_path.read_bytes()).hexdigest()
        fp16_init_count = sum(
            1 for init in fp16_proto.graph.initializer if init.data_type == onnx.TensorProto.FLOAT16
        )

    with torch.no_grad():
        pt_outs = [t.detach().cpu().numpy() for t in wrapper(x_in)]

    fp32_max_abs_diff = 0.0
    fp32_ort_verified = False
    fp16_max_abs_diff: float | None = None
    fp16_ort_verified = False
    fp32_per_output_diff: dict[str, float] = {}
    fp16_per_output_diff: dict[str, float] = {}
    ort_fp32_p50_ms: float | None = None
    ort_fp16_p50_ms: float | None = None
    ort_fp16_p95_ms: float | None = None

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        def _make_sess(path: Path) -> Any:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(ort_threads)
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.add_session_config_entry("session.dynamic_block_base", "4")
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

        x_np = x_in.numpy()
        if int(bench_iters) > 0:
            time.sleep(0.25)
        sess32 = _make_sess(fp32_path)
        ort32_outs = sess32.run(out_names, {"roi_crop": x_np})
        fp32_per_output_diff = {
            name: float(np.max(np.abs(a - b))) for name, a, b in zip(out_names, pt_outs, ort32_outs)
        }
        fp32_max_abs_diff = max(fp32_per_output_diff.values()) if fp32_per_output_diff else 0.0
        # Hesse rho = cos*mid_u + sin*mid_v scales unit-vector rounding by ~roi_size/2 (allow 5e-3 px on keylines_hesse)
        fp32_allclose = all(
            bool(
                np.all(np.isfinite(b))
                and np.allclose(
                    a,
                    b,
                    rtol=1e-3,
                    atol=(max(float(fp32_tol), 5.0e-3) if name == "keylines_hesse" else float(fp32_tol)),
                )
            )
            for name, a, b in zip(out_names, pt_outs, ort32_outs)
        )
        fp32_ort_verified = bool(math.isfinite(fp32_max_abs_diff) and fp32_allclose)
        if not fp32_ort_verified:
            raise ValueError(
                f"XVS-Land v5.2 FP32 ONNX parity failed: max_abs_diff={fp32_max_abs_diff:.6e} "
                f"(atol={fp32_tol}, rtol=1e-3, per_output={fp32_per_output_diff})"
            )
        if int(bench_iters) > 0:
            for _ in range(int(bench_warmup)):
                sess32.run(out_names, {"roi_crop": x_np})
            t32 = []
            for _ in range(int(bench_iters)):
                t0 = time.perf_counter()
                sess32.run(out_names, {"roi_crop": x_np})
                t32.append((time.perf_counter() - t0) * 1000.0)
            ort_fp32_p50_ms = float(np.percentile(t32, 50))
        del sess32
        gc.collect()

        if export_fp16 and fp16_path.is_file():
            sess16 = _make_sess(fp16_path)
            ort16_outs = sess16.run(out_names, {"roi_crop": x_np})
            fp16_per_output_diff = {
                name: float(np.max(np.abs(a - b))) for name, a, b in zip(out_names, pt_outs, ort16_outs)
            }
            fp16_max_abs_diff = max(fp16_per_output_diff.values()) if fp16_per_output_diff else 0.0
            fp16_allclose = all(
                bool(np.all(np.isfinite(b)) and np.allclose(a, b, rtol=2e-2, atol=float(fp16_tol)))
                for a, b in zip(pt_outs, ort16_outs)
            )
            fp16_ort_verified = bool(math.isfinite(fp16_max_abs_diff) and fp16_allclose)
            if not fp16_ort_verified:
                raise ValueError(
                    f"XVS-Land v5.2 FP16 ONNX parity failed: max_abs_diff={fp16_max_abs_diff:.6e} "
                    f"(atol={fp16_tol}, rtol=2e-2, per_output={fp16_per_output_diff})"
                )
            if int(bench_iters) > 0:
                for _ in range(int(bench_warmup)):
                    sess16.run(out_names, {"roi_crop": x_np})
                t16 = []
                for _ in range(int(bench_iters)):
                    t0 = time.perf_counter()
                    sess16.run(out_names, {"roi_crop": x_np})
                    t16.append((time.perf_counter() - t0) * 1000.0)
                ort_fp16_p50_ms = float(np.percentile(t16, 50))
                ort_fp16_p95_ms = float(np.percentile(t16, 95))
            del sess16
            gc.collect()
    except ImportError:
        fp32_ort_verified = False
        fp16_ort_verified = False

    fp32_bytes = int(fp32_path.stat().st_size)
    fp32_sha = hashlib.sha256(fp32_path.read_bytes()).hexdigest()
    return {
        "arch": str(model.arch_id),
        "path": str(fp16_path if (export_fp16 and fp16_path.is_file()) else fp32_path),
        "bytes": int(fp16_bytes if fp16_bytes is not None else fp32_bytes),
        "sha256": str(fp16_sha if fp16_sha is not None else fp32_sha),
        "opset": int(opset),
        "roi_size": int(roi_size),
        "output_names": out_names,
        "total_params": int(total_params),
        "max_abs_diff": float(fp32_max_abs_diff),
        "ort_verified": bool(fp32_ort_verified and (fp16_ort_verified if export_fp16 else True)),
        "fp32_onnx_path": str(fp32_path),
        "fp32_onnx_bytes": fp32_bytes,
        "fp32_onnx_mib": round(fp32_bytes / float(1024 * 1024), 4),
        "fp32_onnx_sha256": fp32_sha,
        "fp32_max_abs_diff": float(fp32_max_abs_diff),
        "fp32_per_output_diff": fp32_per_output_diff,
        "fp32_ort_verified": bool(fp32_ort_verified),
        "fp16_onnx_path": str(fp16_path) if export_fp16 else None,
        "fp16_onnx_bytes": fp16_bytes,
        "fp16_onnx_mib": round(fp16_bytes / float(1024 * 1024), 4) if fp16_bytes is not None else None,
        "fp16_onnx_sha256": fp16_sha,
        "fp16_initializer_count": int(fp16_init_count),
        "fp16_max_abs_diff": float(fp16_max_abs_diff) if fp16_max_abs_diff is not None else None,
        "fp16_per_output_diff": fp16_per_output_diff,
        "fp16_ort_verified": bool(fp16_ort_verified),
        "ort_fp32_p50_ms": ort_fp32_p50_ms,
        "ort_fp16_p50_ms": ort_fp16_p50_ms,
        "ort_fp16_p95_ms": ort_fp16_p95_ms,
    }


def _build_xvs_land_model(
    arch: str,
    *,
    roi_size: int,
    split_ratio: float,
    coord_dropout_prob: float = 0.50,
) -> XvsLandModel | XvsLandV52Model:
    arch_norm = str(arch).strip().lower()
    if arch_norm == "v53":
        return XvsLandV52Model(
            in_channels=3,
            roi_width=int(roi_size),
            roi_height=int(roi_size),
            split_ratio=float(split_ratio),
            enable_runway_present_head=True,
            coord_dropout_prob=float(coord_dropout_prob),
            use_endpoint_prior=False,
            enable_simcc_uncertainty_cues=True,
        )
    if arch_norm == "v52":
        return XvsLandV52Model(
            in_channels=3,
            roi_width=int(roi_size),
            roi_height=int(roi_size),
            split_ratio=float(split_ratio),
        )
    if arch_norm == "v45":
        return XvsLandModel(
            in_channels=3,
            roi_width=int(roi_size),
            roi_height=int(roi_size),
            split_ratio=float(split_ratio),
        )
    raise ValueError(f"Unsupported XVS-Land architecture {arch!r}; expected 'v45', 'v52', or 'v53'")


def _compute_stage2a_loss(
    out: dict[str, torch.Tensor],
    kp_t: torch.Tensor,
    kl_t: torch.Tensor,
    *,
    arch: str,
    roi_size: int,
    split_ratio: float,
    present_t: torch.Tensor | None = None,
) -> torch.Tensor:
    pos_mask = present_t.float() if (present_t is not None and arch == "v53") else None
    l_simcc = simcc_kl_loss(out["logits_x"], out["logits_y"], kp_t, split_ratio, sample_mask=pos_mask)
    l_okls = okls_keyline_loss(out["keylines_hesse"].float(), kl_t.float(), sample_mask=pos_mask)
    if arch in ("v52", "v53"):
        inv_roi = 1.0 / float(max(1, roi_size))
        if pos_mask is None:
            l_kp_l1 = F.smooth_l1_loss(out["keypoints_xy"].float(), kp_t.float()) * inv_roi * 10.0
        else:
            kp_elem = F.smooth_l1_loss(out["keypoints_xy"].float(), kp_t.float(), reduction="none").mean(dim=(1, 2))
            l_kp_l1 = ((kp_elem * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)) * inv_roi * 10.0
        l_ep = (
            keyline_endpoint_loss(out["keyline_endpoints"].float(), kp_t.float(), sample_mask=pos_mask)
            * inv_roi
            * 10.0
        )
        l_geo = (
            keyline_keypoint_consistency_loss(
                out["keylines_hesse"].float(),
                out["keypoints_xy"].float(),
                sample_mask=pos_mask,
            )
            * inv_roi
            * 5.0
        )
        geom_loss = l_simcc + 0.5 * l_kp_l1 + 0.5 * l_okls + 0.5 * l_ep + 0.25 * l_geo
        if arch == "v53" and "runway_present_logit" in out and present_t is not None:
            l_pres = focal_bce_with_logits_loss(out["runway_present_logit"], present_t, alpha=0.35, gamma=2.0)
            return geom_loss + 2.0 * l_pres
        return geom_loss
    return l_simcc + 0.5 * l_okls


def run_xvs_land_g4_training(cfg: XvsLandG4Config) -> dict[str, Any]:
    """Execute full 2-stage Colab L4/G4 or `--smoke-local` XVS-Land training with fail-closed LARD guard."""
    arch = str(cfg.arch).strip().lower()
    is_v53 = bool(cfg.v53_mode or arch == "v53")
    if is_v53:
        arch = "v53"
    if arch not in ("v45", "v52", "v53"):
        raise ValueError(f"Unsupported arch {cfg.arch!r}; expected 'v45', 'v52', or 'v53'")

    split_ver, train_runways = enforce_xvs_land_licenses_and_split(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.smoke_local else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    env_report = collect_blackwell_env_report()

    roi_size = 128 if cfg.smoke_local else int(cfg.roi_size)
    if cfg.smoke_local:
        stage2a_epochs = 2
        stage2b_epochs = 2
        steps_per_epoch = 4
        num_train = 24
        num_val = 8
        ekf_eval_frames = min(10, int(cfg.ekf_eval_frames))
    elif cfg.real_world_scale:
        stage2a_epochs = max(36 if is_v53 else 24, int(cfg.stage2a_epochs))
        stage2b_epochs = max(14 if is_v53 else 10, int(cfg.stage2b_epochs))
        steps_per_epoch = max(48, int(cfg.steps_per_epoch))
        num_train = max(2400, int(cfg.num_train_samples))
        num_val = max(128, int(cfg.num_val_samples))
        ekf_eval_frames = int(cfg.ekf_eval_frames)
    else:
        stage2a_epochs = int(cfg.stage2a_epochs)
        stage2b_epochs = int(cfg.stage2b_epochs)
        steps_per_epoch = int(cfg.steps_per_epoch)
        num_train = int(cfg.num_train_samples)
        num_val = int(cfg.num_val_samples)
        ekf_eval_frames = int(cfg.ekf_eval_frames)
    candidates = (4, 8) if cfg.smoke_local else cfg.batch_candidates

    naip_textures: list[np.ndarray] = []
    if cfg.use_naip_stac and not cfg.smoke_local:
        naip_textures = fetch_conus_airport_naip_textures(
            train_runways,
            cfg.naip_stac_url,
            max_airports=int(cfg.naip_max_airports),
            patch_size=roi_size,
        )

    ksfo_val_spec = [
        RunwaySpec(icao="KSFO", length_m=3471.0, width_m=61.0, heading_deg=297.7)
    ]
    train_ds = PerspectiveRunwayDataset(
        train_runways,
        num_samples=num_train,
        roi_size=roi_size,
        seed=cfg.seed,
        domain_randomize=bool(cfg.real_world_scale or arch in ("v52", "v53")),
        rich_markings=bool(arch in ("v52", "v53") or cfg.real_world_scale),
        naip_bg_textures=naip_textures if naip_textures else None,
        v53_mode=is_v53,
        neg_fraction=float(cfg.neg_fraction) if is_v53 else 0.0,
        roi_jitter_frac=float(cfg.roi_jitter_frac) if is_v53 else 0.0,
    )
    # In extended, val_ds MUST run with v53_mode=True so StudentTCalibrator & val proxies match extended distribution!
    val_ds = PerspectiveRunwayDataset(
        ksfo_val_spec,
        num_samples=num_val,
        roi_size=roi_size,
        seed=cfg.seed + 500,
        domain_randomize=bool(is_v53),
        rich_markings=bool(arch in ("v52", "v53") or cfg.real_world_scale),
        lazy=False,
        v53_mode=bool(is_v53),
        neg_fraction=float(cfg.neg_fraction) if is_v53 else 0.0,
        roi_jitter_frac=float(cfg.roi_jitter_frac) if is_v53 else 0.0,
    )

    eager_model = _build_xvs_land_model(
        arch,
        roi_size=roi_size,
        split_ratio=cfg.split_ratio,
        coord_dropout_prob=float(cfg.coord_dropout_prob),
    ).to(device)
    total_params = sum(int(p.numel()) for p in eager_model.parameters())

    # BC-G4-m4: Eager AutoVRAMBatchTuner BEFORE torch.compile + weight snapshot restore
    init_weights = {k: v.detach().cpu().clone() for k, v in eager_model.state_dict().items()}
    probe_opt = torch.optim.AdamW(eager_model.parameters(), lr=cfg.lr_stage2a, weight_decay=cfg.weight_decay)

    def _probe_step(bs: int) -> None:
        probe_opt.zero_grad(set_to_none=True)
        x_d = torch.randn(bs, 3, roi_size, roi_size, device=device)
        kp_d = torch.rand(bs, NUM_KEYPOINTS, 2, device=device) * float(roi_size - 4) + 2.0
        kl_d = torch.zeros(bs, NUM_KEYLINES, 3, device=device)
        kl_d[..., 1] = 1.0
        pres_d = torch.ones(bs, device=device) if is_v53 else None
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = eager_model(x_d)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            loss = _compute_stage2a_loss(
                out,
                kp_d,
                kl_d,
                arch=arch,
                roi_size=roi_size,
                split_ratio=cfg.split_ratio,
                present_t=pres_d,
            )
        loss.backward()
        probe_opt.step()

    tuner = AutoVRAMBatchTuner(
        target_frac=cfg.vram_target_frac,
        safety_headroom_gib=cfg.vram_headroom_gib,
        device=device,
    )
    try:
        tuner_report = tuner.probe_batch_size(_probe_step, candidates)
    finally:
        eager_model.load_state_dict(init_weights)
        eager_model.zero_grad(set_to_none=True)
        del probe_opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_bs = min(int(tuner_report["selected_batch_size"]), len(train_ds))
    tuner_peak_gib = float(tuner_report["peak_vram_bytes"]) / float(1024**3)
    print(
        f"[XVS-Land arch={arch} params={total_params:,}] LARD guard PASS "
        f"(lard={split_ver.lard_real_count}, train_runways={len(train_runways)}) | "
        f"AutoVRAMBatchTuner: selected_batch={selected_bs} peak_vram_gib={tuner_peak_gib:.2f} "
        f"util={float(tuner_report['utilization_frac'])*100.0:.1f}%",
        flush=True,
    )

    ckpt_mgr = DriveCheckpointManager(
        local_dir=cfg.out_dir / "checkpoints",
        drive_dir=cfg.drive_ckpt_dir,
    )
    import os as _os

    loader_workers = 0
    if device.type == "cuda" and _os.name != "nt":
        if train_ds._ram_cache_imgs is not None:
            loader_workers = min(8, max(2, (_os.cpu_count() or 4) // 6))
        elif train_ds.lazy:
            loader_workers = min(24, max(2, (_os.cpu_count() or 4) // 2))
    loader = DataLoader(
        train_ds,
        batch_size=max(2, selected_bs),
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(loader_workers > 0),
    )

    # ---- Stage 2A: SimCC + OKLS (+ endpoint, geometric consistency & extended runway_present_logit) ----
    stage2a_params = (
        eager_model.stage2a_parameters()
        if isinstance(eager_model, XvsLandV52Model)
        else (
            list(eager_model.backbone.parameters())
            + list(eager_model.simcc_head.parameters())
            + list(eager_model.keyline_head.parameters())
        )
    )
    opt_2a = torch.optim.AdamW(
        stage2a_params,
        lr=cfg.lr_stage2a,
        weight_decay=cfg.weight_decay,
    )
    start_2a = 0
    history_2a: list[dict[str, float]] = []
    resumed_2a = False
    latest_2a = ckpt_mgr.load_latest(tag_prefix="xvs_land_stage2a_epoch_")
    if latest_2a is not None and "model_state_dict" in latest_2a:
        eager_model.load_state_dict(latest_2a["model_state_dict"])
        if "optimizer_state_dict" in latest_2a:
            opt_2a.load_state_dict(latest_2a["optimizer_state_dict"])
        start_2a = int(latest_2a["epoch"]) + 1
        if "history" in latest_2a and isinstance(latest_2a["history"], list):
            history_2a = list(latest_2a["history"])
        resumed_2a = True

    train_model: nn.Module = (
        torch.compile(eager_model)
        if (cfg.use_compile and device.type == "cuda" and not cfg.smoke_local)
        else eager_model
    )

    for pg in opt_2a.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = float(cfg.lr_stage2a)
    sched_2a = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_2a, T_max=max(1, stage2a_epochs), eta_min=1.0e-5, last_epoch=start_2a - 1
    )

    for epoch in range(start_2a, stage2a_epochs):
        train_model.train()
        ep_losses: list[float] = []
        step_idx = 0
        while step_idx < steps_per_epoch:
            for batch in loader:
                if is_v53:
                    img_b, kp_b, kl_b, pres_b = batch
                    pres_t: torch.Tensor | None = pres_b.to(device, non_blocking=True)
                else:
                    img_b, kp_b, kl_b = batch
                    pres_t = None
                opt_2a.zero_grad(set_to_none=True)
                img_t = img_b.to(device, non_blocking=True)
                kp_t = kp_b.to(device, non_blocking=True)
                kl_t = kl_b.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    out = train_model(img_t)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    loss = _compute_stage2a_loss(
                        out,
                        kp_t,
                        kl_t,
                        arch=arch,
                        roi_size=roi_size,
                        split_ratio=cfg.split_ratio,
                        present_t=pres_t,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite XVS-Land Stage-2A loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(eager_model.parameters(), max_norm=2.0)
                opt_2a.step()
                ep_losses.append(float(loss.detach().cpu().item()))
                step_idx += 1
                if step_idx >= steps_per_epoch:
                    break
        sched_2a.step()

        mean_loss = float(np.mean(ep_losses))
        history_2a.append({"epoch": float(epoch), "stage2a_loss": mean_loss})
        peak_gib = (
            float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[XVS-Land Stage2A arch={arch}] epoch={epoch + 1}/{stage2a_epochs} "
            f"loss={mean_loss:.4f} peak_vram_gib={peak_gib:.2f}",
            flush=True,
        )
        raw_2a = getattr(train_model, "_orig_mod", train_model)
        ckpt_mgr.save(
            f"xvs_land_stage2a_epoch_{epoch:03d}",
            {
                "arch": arch,
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_2a.state_dict().items()},
                "optimizer_state_dict": opt_2a.state_dict(),
                "history": history_2a,
            },
        )

    # ---- Stage 2B: Frozen-Mean Cholesky 2x2 Bivariate GNLL (+ extended corrupted ranking & neg-uncertainty) ----
    raw_model: XvsLandModel | XvsLandV52Model = getattr(train_model, "_orig_mod", train_model)
    for p in raw_model.parameters():
        p.requires_grad = False
    stage2b_params = (
        raw_model.stage2b_parameters()
        if isinstance(raw_model, XvsLandV52Model)
        else list(raw_model.cov_head.parameters())
    )
    for p in stage2b_params:
        p.requires_grad = True

    opt_2b = torch.optim.AdamW(stage2b_params, lr=cfg.lr_stage2b, weight_decay=cfg.weight_decay)
    start_2b = 0
    history_2b: list[dict[str, float]] = []
    resumed_2b = False
    latest_2b = ckpt_mgr.load_latest(tag_prefix="xvs_land_stage2b_epoch_")
    if latest_2b is not None and "model_state_dict" in latest_2b:
        raw_model.load_state_dict(latest_2b["model_state_dict"])
        if "optimizer_state_dict" in latest_2b:
            opt_2b.load_state_dict(latest_2b["optimizer_state_dict"])
        start_2b = int(latest_2b["epoch"]) + 1
        if "history_2b" in latest_2b and isinstance(latest_2b["history_2b"], list):
            history_2b = list(latest_2b["history_2b"])
        resumed_2b = True

    for pg in opt_2b.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = float(cfg.lr_stage2b)
    sched_2b = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_2b, T_max=max(1, stage2b_epochs), eta_min=5.0e-5, last_epoch=start_2b - 1
    )
    corrupt_rng = np.random.default_rng(cfg.seed + 7777)

    for epoch in range(start_2b, stage2b_epochs):
        raw_model.backbone.eval()
        raw_model.simcc_head.eval()
        raw_model.keyline_head.eval()
        raw_model.cov_head.train()
        if isinstance(raw_model, XvsLandV52Model):
            raw_model.keyline_head.sigma_head.train()
            if raw_model.runway_present_head is not None:
                raw_model.runway_present_head.eval()
        ep_losses_2b: list[float] = []
        step_idx = 0
        while step_idx < steps_per_epoch:
            for batch in loader:
                if is_v53:
                    img_b, kp_b, _, pres_b = batch
                    pres_t2: torch.Tensor | None = pres_b.to(device, non_blocking=True)
                else:
                    img_b, kp_b, _ = batch
                    pres_t2 = None
                opt_2b.zero_grad(set_to_none=True)
                img_t = img_b.to(device, non_blocking=True)
                kp_t = kp_b.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    out = raw_model(img_t)
                    out_corrupt = (
                        raw_model(apply_v53_severe_corruption(img_t, corrupt_rng))
                        if is_v53
                        else None
                    )
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    residual_xy = out["keypoints_xy"].detach().float() - kp_t.float()
                    pos_m = (pres_t2 > 0.5).float() if (is_v53 and pres_t2 is not None) else None
                    loss_gnll = bivariate_gnll_loss(residual_xy, out["cholesky_l"].float(), sample_mask=pos_m)
                    if isinstance(raw_model, XvsLandV52Model):
                        loss_kl_nll = keyline_sigma_nll_loss(
                            out["keylines_hesse"].detach().float(),
                            out["keyline_sigma_px"].float(),
                            kp_t.float(),
                            sample_mask=pos_m,
                        )
                        loss_gnll = loss_gnll + 0.5 * loss_kl_nll

                    if is_v53 and out_corrupt is not None and isinstance(raw_model, XvsLandV52Model):
                        # Corrupted GNLL + Monotonic Degradation Ranking Loss (glare/occlusion/rain/blur)
                        res_corr = out_corrupt["keypoints_xy"].detach().float() - kp_t.float()
                        loss_gnll_c = bivariate_gnll_loss(
                            res_corr, out_corrupt["cholesky_l"].float(), sample_mask=pos_m
                        )
                        loss_kl_c = keyline_sigma_nll_loss(
                            out_corrupt["keylines_hesse"].detach().float(),
                            out_corrupt["keyline_sigma_px"].float(),
                            kp_t.float(),
                            sample_mask=pos_m,
                        )
                        cov_c_tr = (out["cov_2x2"].float()[..., 0, 0] + out["cov_2x2"].float()[..., 1, 1]) * 0.5
                        cov_d_tr = (
                            out_corrupt["cov_2x2"].float()[..., 0, 0] + out_corrupt["cov_2x2"].float()[..., 1, 1]
                        ) * 0.5
                        log_sig_c = 0.5 * torch.log(cov_c_tr.clamp(min=1.0e-6))
                        log_sig_d = 0.5 * torch.log(cov_d_tr.clamp(min=1.0e-6))
                        err_c = torch.linalg.norm(residual_xy, dim=-1)
                        err_d = torch.linalg.norm(res_corr, dim=-1)
                        l_mono_kp = monotonic_degradation_ranking_loss(
                            log_sig_c, log_sig_d, err_c, err_d, margin=0.35, err_ratio_thr=1.25, sample_mask=pos_m
                        )
                        log_kls_c = torch.log(out["keyline_sigma_px"].float().clamp(min=KEYLINE_SIGMA_MIN_PX))
                        log_kls_d = torch.log(out_corrupt["keyline_sigma_px"].float().clamp(min=KEYLINE_SIGMA_MIN_PX))
                        l_mono_kl = monotonic_degradation_ranking_loss(
                            log_kls_c,
                            log_kls_d,
                            err_c.mean(dim=-1, keepdim=True).expand_as(log_kls_c),
                            err_d.mean(dim=-1, keepdim=True).expand_as(log_kls_d),
                            margin=0.35,
                            err_ratio_thr=1.25,
                            sample_mask=pos_m,
                        )
                        loss_gnll = 0.5 * (loss_gnll + loss_gnll_c + 0.5 * loss_kl_c) + 1.5 * (l_mono_kp + 0.5 * l_mono_kl)

                        # Push uncertainty HIGH on negative non-runway frames (`y_present == 0`)
                        if pos_m is not None:
                            neg_m = 1.0 - pos_m
                            if float(neg_m.sum().detach().item()) > 0.0:
                                target_neg_log_sig = math.log(35.0)
                                neg_kp_hinge = F.relu(target_neg_log_sig - log_sig_c.mean(dim=-1))
                                neg_kl_hinge = F.relu(target_neg_log_sig - log_kls_c.mean(dim=-1))
                                loss_neg_unc = (
                                    ((neg_kp_hinge + neg_kl_hinge) * neg_m).sum() / neg_m.sum().clamp(min=1.0)
                                )
                                loss_gnll = loss_gnll + 1.0 * loss_neg_unc

                if not torch.isfinite(loss_gnll):
                    raise FloatingPointError(f"Non-finite XVS-Land Stage-2B GNLL at epoch {epoch}")
                loss_gnll.backward()
                torch.nn.utils.clip_grad_norm_(stage2b_params, max_norm=2.0)
                opt_2b.step()
                ep_losses_2b.append(float(loss_gnll.detach().cpu().item()))
                step_idx += 1
                if step_idx >= steps_per_epoch:
                    break
        sched_2b.step()
        mean_gnll = float(np.mean(ep_losses_2b))
        history_2b.append({"epoch": float(epoch), "stage2b_gnll": mean_gnll})
        peak_gib_2b = (
            float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[XVS-Land Stage2B arch={arch}] epoch={epoch + 1}/{stage2b_epochs} "
            f"gnll={mean_gnll:.4f} peak_vram_gib={peak_gib_2b:.2f}",
            flush=True,
        )
        ckpt_mgr.save(
            f"xvs_land_stage2b_epoch_{epoch:03d}",
            {
                "arch": arch,
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
                "optimizer_state_dict": opt_2b.state_dict(),
                "history_2b": history_2b,
            },
        )

    # Fit Student-t post-hoc variance calibrator (+ keyline sigma scale for v52) on validation residuals
    calibrator = StudentTCalibrator(nu=cfg.student_t_nu, scale=1.0)
    raw_model.eval()
    m2_chunks: list[torch.Tensor] = []
    kl_z2_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=min(32, max(2, len(val_ds))), shuffle=False):
            if val_ds.v53_mode:
                img_b, kp_b, _, pres_b = batch
                pos_idx = pres_b > 0.5
                if not bool(pos_idx.any().item()):
                    continue
                img_b = img_b[pos_idx]
                kp_b = kp_b[pos_idx]
            else:
                img_b, kp_b, _ = batch
            kp_dev = kp_b.to(device).float()
            out = raw_model(img_b.to(device))
            res = (out["keypoints_xy"].float() - kp_dev).unsqueeze(-1)  # [B, K, 2, 1]
            sigma_inv = torch.linalg.inv(
                out["cov_2x2"].float() + 1e-5 * torch.eye(2, device=device).view(1, 1, 2, 2)
            )
            m2 = torch.matmul(res.transpose(-1, -2), torch.matmul(sigma_inv, res)).squeeze(-1).squeeze(-1)
            m2_chunks.append(m2.reshape(-1).cpu())

            if "keyline_sigma_px" in out:
                kl_p = out["keylines_hesse"].float()
                kl_s = out["keyline_sigma_px"].float().clamp(min=KEYLINE_SIGMA_MIN_PX, max=KEYLINE_SIGMA_MAX_PX)
                z2_list: list[torch.Tensor] = []
                for k_idx, (i0, i1) in enumerate(KEYLINE_ENDPOINT_INDICES):
                    rho_k = kl_p[:, k_idx, 0]
                    c_k = kl_p[:, k_idx, 1]
                    s_k = kl_p[:, k_idx, 2]
                    sig_k = kl_s[:, k_idx]
                    d0 = (c_k * kp_dev[:, i0, 0] + s_k * kp_dev[:, i0, 1] - rho_k) / sig_k
                    d1 = (c_k * kp_dev[:, i1, 0] + s_k * kp_dev[:, i1, 1] - rho_k) / sig_k
                    z2_list.append(d0 * d0)
                    z2_list.append(d1 * d1)
                kl_z2_chunks.append(torch.cat(z2_list, dim=0).cpu())

    calibrator.fit_scale_from_mahalanobis(torch.cat(m2_chunks, dim=0), target_dof=2.0)
    keyline_sigma_scale = 1.0
    if kl_z2_chunks:
        kl_z2 = torch.cat(kl_z2_chunks, dim=0)
        finite_z2 = kl_z2[torch.isfinite(kl_z2)]
        if finite_z2.numel() > 0:
            keyline_sigma_scale = float(math.sqrt(max(0.25, float(finite_z2.mean().item()))))

    eval_metrics = evaluate_xvs_land_approach(
        raw_model,
        val_ds,
        calibrator,
        device,
        keyline_sigma_scale=keyline_sigma_scale,
        ekf_eval_frames=ekf_eval_frames,
    )

    final_ckpt_path = cfg.out_dir / "xvs_land_final.pt"
    torch.save(
        {
            "arch": arch,
            "roi_size": int(roi_size),
            "split_ratio": float(cfg.split_ratio),
            "total_params": int(total_params),
            "artifact_tag": "SMOKE_LOCAL" if cfg.smoke_local else "COLAB_G4_GPU_ARTIFACT",
            "real_world_scale": bool(cfg.real_world_scale),
            "naip_textures_loaded": len(naip_textures),
            "lard_verification": {
                "manifest_verified": split_ver.manifest_verified,
                "lard_real_count": split_ver.lard_real_count,
                "train_count": split_ver.train_count,
                "heldout_count": split_ver.heldout_count,
                "ksfo_heldout": split_ver.ksfo_heldout,
            },
            "student_t_nu": calibrator.nu,
            "student_t_scale": calibrator.scale,
            "keyline_sigma_scale": keyline_sigma_scale,
            "model_state_dict": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            "metrics": eval_metrics,
        },
        final_ckpt_path,
    )

    onnx_meta: dict[str, Any] | None = None
    if cfg.export_onnx:
        if isinstance(raw_model, XvsLandV52Model):
            onnx_meta = export_xvs_land_v52_onnx(
                raw_model,
                cfg.out_dir,
                roi_size=roi_size,
                split_ratio=cfg.split_ratio,
                export_fp16=bool(cfg.export_fp16_onnx),
                ort_threads=2 if cfg.smoke_local else 6,
                bench_warmup=2 if cfg.smoke_local else 5,
                bench_iters=5 if cfg.smoke_local else 20,
            )
        else:
            onnx_meta = export_xvs_land_onnx(
                raw_model,
                cfg.out_dir / "xvs_land_v45.onnx",
                roi_size=roi_size,
                split_ratio=cfg.split_ratio,
            )

    s2a_init: float | None = float(history_2a[0]["stage2a_loss"]) if history_2a else None
    s2a_final: float | None = float(history_2a[-1]["stage2a_loss"]) if history_2a else None
    s2b_init: float | None = float(history_2b[0]["stage2b_gnll"]) if history_2b else None
    s2b_final: float | None = float(history_2b[-1]["stage2b_gnll"]) if history_2b else None

    s2a_ok = s2a_final is not None and math.isfinite(s2a_final)
    s2b_ok = s2b_final is not None and math.isfinite(s2b_final)
    s2a_dec = bool(
        s2a_init is not None
        and s2a_final is not None
        and math.isfinite(s2a_init)
        and math.isfinite(s2a_final)
        and s2a_final < s2a_init
    )
    s2b_dec = bool(
        s2b_init is not None
        and s2b_final is not None
        and math.isfinite(s2b_init)
        and math.isfinite(s2b_final)
        and s2b_final < s2b_init
    )

    report: dict[str, Any] = {
        "schema": (
            "xvs.v2_m2.g4_xvs_land_report/3"
            if arch == "v53"
            else ("xvs.v2_m2.g4_xvs_land_report/2" if arch == "v52" else "xvs.v2_m2.g4_xvs_land_report/1")
        ),
        "arch": arch,
        "v53_mode": bool(is_v53),
        "roi_size": int(roi_size),
        "total_params": int(total_params),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_local": bool(cfg.smoke_local),
        "real_world_scale": bool(cfg.real_world_scale),
        "use_naip_stac": bool(cfg.use_naip_stac),
        "naip_textures_loaded": len(naip_textures),
        "num_train_samples_effective": len(train_ds),
        "num_val_samples_effective": len(val_ds),
        "device": str(device),
        "bf16_enabled": use_bf16,
        "resumed_stage2a": resumed_2a,
        "resumed_stage2b": resumed_2b,
        "lard_guard": {
            "manifest_verified": split_ver.manifest_verified,
            "lard_real_count": split_ver.lard_real_count,
            "train_count": split_ver.train_count,
            "heldout_count": split_ver.heldout_count,
            "ksfo_heldout": split_ver.ksfo_heldout,
        },
        "selected_batch_size": selected_bs,
        "vram_tuner": tuner_report,
        "env_report": env_report,
        "stage2a_initial_loss": s2a_init,
        "stage2a_final_loss": s2a_final,
        "stage2a_loss_decreased": s2a_dec,
        "stage2b_initial_gnll": s2b_init,
        "stage2b_final_gnll": s2b_final,
        "stage2b_gnll_decreased": s2b_dec,
        "history_stage2a": history_2a,
        "history_stage2b": history_2b,
        "eval_metrics": eval_metrics,
        "final_checkpoint": str(final_ckpt_path),
        "onnx_export": onnx_meta,
        "pass": bool(
            split_ver.manifest_verified
            and split_ver.ksfo_heldout
            and s2a_ok
            and s2b_ok
            and eval_metrics["ksfo_anti_misassociation_pass"]
            and int(eval_metrics["hmi_count"]) == 0
            and int(eval_metrics["ekf_hmi_count"]) == 0
        ),
    }
    (cfg.out_dir / "xvs_land_g4_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    onnx_diff_str = (
        f"{float(onnx_meta['max_abs_diff']):.3e}"
        if onnx_meta is not None and "max_abs_diff" in onnx_meta
        else "disabled"
    )
    print(
        f"[XVS-Land arch={arch}] DONE pass={report['pass']} | simcc_rmse_px={eval_metrics['simcc_rmse_px']:.3f} "
        f"okls_deg={eval_metrics['okls_angle_err_deg']:.3f} hmi={eval_metrics['hmi_count']} "
        f"ekf_rmse={eval_metrics['ekf_pos_rmse_m']} ekf_cat1={eval_metrics['ekf_cat1_availability']} "
        f"naip={len(naip_textures)} onnx_diff={onnx_diff_str}",
        flush=True,
    )
    return report


def _parse_args(argv: list[str] | None = None) -> XvsLandG4Config:
    p = argparse.ArgumentParser(description="OpenXVS v1.1.1 XVS-Land Colab L4/G4 2-stage trainer")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--lard-manifest", type=Path, required=True)
    p.add_argument("--runways-csv", type=Path, default=DEFAULT_RUNWAYS_CSV)
    p.add_argument("--drive-ckpt-dir", type=Path, default=None)
    p.add_argument("--arch", type=str, choices=("v45", "v52", "v53"), default="v52")
    p.add_argument("--v53-mode", action="store_true")
    p.add_argument("--neg-fraction", type=float, default=0.35)
    p.add_argument("--coord-dropout-prob", type=float, default=0.50)
    p.add_argument("--roi-jitter-frac", type=float, default=0.35)
    p.add_argument("--roi-size", type=int, default=None)
    p.add_argument("--smoke-local", action="store_true")
    p.add_argument("--real-world-scale", action="store_true")
    p.add_argument("--use-naip-stac", action="store_true")
    p.add_argument("--naip-stac-url", type=str, default="https://planetarycomputer.microsoft.com/api/stac/v1")
    p.add_argument("--naip-max-airports", type=int, default=36)
    p.add_argument("--no-export-onnx", action="store_true")
    p.add_argument("--no-export-fp16-onnx", action="store_true")
    p.add_argument("--stage2a-epochs", type=int, default=12)
    p.add_argument("--stage2b-epochs", type=int, default=6)
    p.add_argument("--steps-per-epoch", type=int, default=160)
    p.add_argument("--num-train-samples", type=int, default=1024)
    p.add_argument("--num-val-samples", type=int, default=128)
    p.add_argument("--ekf-eval-frames", type=int, default=40)
    p.add_argument("--lr-stage2a", type=float, default=4.0e-4)
    p.add_argument("--lr-stage2b", type=float, default=2.0e-3)
    p.add_argument("--vram-headroom-gib", type=float, default=12.0)
    p.add_argument("--use-compile", action="store_true")
    p.add_argument("--seed", type=int, default=20261121)
    ns = p.parse_args(argv)
    eff_arch = "v53" if (bool(ns.v53_mode) or str(ns.arch) == "v53") else str(ns.arch)
    roi_size = int(ns.roi_size) if ns.roi_size is not None else (384 if eff_arch in ("v52", "v53") else 256)
    return XvsLandG4Config(
        out_dir=ns.out_dir,
        lard_manifest=ns.lard_manifest,
        runways_csv=ns.runways_csv,
        drive_ckpt_dir=ns.drive_ckpt_dir,
        arch=eff_arch,
        smoke_local=bool(ns.smoke_local),
        real_world_scale=bool(ns.real_world_scale),
        use_naip_stac=bool(ns.use_naip_stac),
        naip_stac_url=str(ns.naip_stac_url),
        naip_max_airports=int(ns.naip_max_airports),
        export_onnx=not bool(ns.no_export_onnx),
        export_fp16_onnx=not bool(ns.no_export_fp16_onnx),
        stage2a_epochs=int(ns.stage2a_epochs),
        stage2b_epochs=int(ns.stage2b_epochs),
        steps_per_epoch=int(ns.steps_per_epoch),
        num_train_samples=int(ns.num_train_samples),
        num_val_samples=int(ns.num_val_samples),
        ekf_eval_frames=int(ns.ekf_eval_frames),
        roi_size=roi_size,
        lr_stage2a=float(ns.lr_stage2a),
        lr_stage2b=float(ns.lr_stage2b),
        vram_headroom_gib=float(ns.vram_headroom_gib),
        use_compile=bool(ns.use_compile),
        seed=int(ns.seed),
        v53_mode=bool(ns.v53_mode or eff_arch == "v53"),
        neg_fraction=float(ns.neg_fraction),
        coord_dropout_prob=float(ns.coord_dropout_prob),
        roi_jitter_frac=float(ns.roi_jitter_frac),
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    report = run_xvs_land_g4_training(cfg)
    print(json.dumps({"pass": report["pass"], "eval_metrics": report["eval_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
