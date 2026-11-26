"""Production Colab G4 (`96 GB VRAM` Blackwell) 4-Stage Training Engine for `XVS-D2` (`SP-2B` §3.1).

Implements:
- Real Amazon AOT S3/HTTPS encounter & `groundtruth.csv` streamer (`part1`, `part2`, `part3`),
  deterministically clipping non-empty `gt_left..gt_bottom` boxes to `[0, 2448] x [0, 2048]`,
  skipping & logging degenerate `w <= 0` / `h <= 0` rows (`dropped_degenerate_boxes`),
  converting `xyxy -> [left, top, width, height]` into `parse_aot_bbox_ltwh` (`m_box=0` on
  `6x6` / `16x16` circles), enforcing strict UTC `capture_day` zero-leakage splits, and
  mixing `70%` AOT + `30%` OpenXVS G3a synthetic clips.
- 6-channel dual-interval `[I_t, I_{t-1}^w, I_{t-5}^w, (I_t - I_{t-1}^w)*R_1, ...]` with
  phase-correlation GMC + `(I - mu) / max(sigma, 8.0)` clamped to `[-5, +5]`.
- 4-stage `bfloat16` (NO `GradScaler`!) + FP32 loss training pipeline with eager
  `AutoVRAMBatchTuner` BEFORE `torch.compile` + weight restore (`BC-G4-m4`), `DriveCheckpointManager`
  atomic epoch checkpointing/resume, `getattr(model, "_orig_mod", model)` unwrapping (`BC-G4-M3`),
  real `Pd@FPI`, `AP50`, `centroid_rmse_px`, `above/below_horizon_pd` evaluation, and
  `export_xvs_d2_web_onnx` FP32/FP16 export.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import urllib.request
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xvs.detection.aot_s3_ingest import (
    AOT_HTTPS_BASE,
    AOT_S3_BUCKET,
    download_s3_object_with_https_fallback,
    map_aot_horizon_label,
    parse_valid_encounters_csv,
)
from xvs.v2.colab_g4_utils import (
    AutoVRAMBatchTuner,
    DriveCheckpointManager,
    collect_blackwell_env_report,
    restore_eager_state,
    snapshot_eager_state,
)
from xvs.v2.license_guard import assert_dataset_allowed, assert_model_weight_allowed
from xvs.v2.xvs_d2.aot_wds import (
    AOT_IMG_HEIGHT,
    AOT_IMG_WIDTH,
    AotG3aBatchMixer,
    AotParsedBox,
    extract_capture_day_utc,
    parse_aot_bbox_ltwh,
    split_flights_by_capture_day,
)
from xvs.v2.xvs_d2.export_onnx import export_xvs_d2_web_onnx
from xvs.v2.xvs_d2.heads import (
    CROP_PRIOR_BIAS,
    FULL_FRAME_PRIOR_BIAS,
    decode_peaks_local_softargmax,
)
from xvs.v2.xvs_d2.losses import (
    compute_xvs_d2_composite_loss,
    cross_kd_loss,
    cscwd_distillation_loss,
)
from xvs.v2.xvs_d2.stem import build_dual_interval_input
from xvs.v2.xvs_d2.tiers import XvsD2Model, build_xvs_d2, count_parameters

__all__ = [
    "AotFrameAnnotation",
    "AotGroundtruthParseResult",
    "AotS3EncounterDataset",
    "LazyD2ClipDataset",
    "SyntheticG3aClipDataset",
    "XvsD2G4TrainConfig",
    "affine_gmc_warp",
    "download_external_clutter_datasets",
    "load_external_clutter_negative_clips",
    "materialize_clip_tensor",
    "parse_aot_groundtruth_csv",
    "phase_correlate_gmc_warp",
    "build_dual_interval_clip_tensor",
    "evaluate_xvs_d2",
    "run_xvs_d2_g4_training",
]

FUJII_BIRD_MIRROR_URLS: tuple[str, ...] = (
    "https://github.com/FumiyaFujii/DistantBirdDataset/archive/refs/heads/main.zip",
    "https://github.com/FumiyaFujii/DistantBirdDataset/archive/refs/heads/master.zip",
)
DET_FLY_MIRROR_URLS: tuple[str, ...] = (
    "https://github.com/bn-b/Det-Fly/archive/refs/heads/main.zip",
    "https://github.com/bn-b/Det-Fly/archive/refs/heads/master.zip",
    "https://github.com/dronedetectionbench/Det-Fly/archive/refs/heads/main.zip",
)


@dataclass(frozen=True)
class AotFrameAnnotation:
    """Sanitized AOT frame record with validated `[left, top, width, height]` boxes."""

    part: str
    flight_id: str
    img_name: str
    frame_idx: int
    capture_day: str
    horizon: str
    boxes: tuple[AotParsedBox, ...]
    range_distance_m: float | None = None

    @property
    def is_negative(self) -> bool:
        return len(self.boxes) == 0


@dataclass
class AotGroundtruthParseResult:
    """Result of parsing an AOT `groundtruth.csv` file with border-clip & drop telemetry."""

    positive_frames: list[AotFrameAnnotation] = field(default_factory=list)
    negative_frames: list[AotFrameAnnotation] = field(default_factory=list)
    flight_records: list[dict[str, Any]] = field(default_factory=list)
    dropped_degenerate_boxes: int = 0
    clipped_border_boxes: int = 0
    circle_approx_boxes: int = 0
    total_rows_read: int = 0


def _try_parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return val


def parse_aot_groundtruth_csv(
    csv_path: Path | str,
    *,
    part: str = "part1",
    img_w: int = AOT_IMG_WIDTH,
    img_h: int = AOT_IMG_HEIGHT,
    allowed_flights: set[str] | None = None,
    max_rows: int | None = None,
) -> AotGroundtruthParseResult:
    """Parse AOT `groundtruth.csv`, clipping `xyxy` to image bounds and dropping degenerate boxes (`BC-G4-M3`)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"AOT groundtruth CSV not found: {path}")

    res = AotGroundtruthParseResult()
    by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    flight_first_img: dict[str, str] = {}

    w_lim = float(img_w)
    h_lim = float(img_h)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_rows is not None and res.total_rows_read >= int(max_rows):
                break
            res.total_rows_read += 1

            flight_id = str(row["flight_id"]).strip() if "flight_id" in row and row["flight_id"] is not None else ""
            img_name = str(row["img_name"]).strip() if "img_name" in row and row["img_name"] is not None else ""
            if not flight_id or not img_name:
                continue
            if allowed_flights is not None and flight_id not in allowed_flights:
                continue

            frame_raw = row["frame"] if "frame" in row else "0"
            frame_val = _try_parse_float(frame_raw)
            frame_idx = int(frame_val) if frame_val is not None else 0

            hz_raw = row["is_above_horizon"] if "is_above_horizon" in row else ""
            horizon = map_aot_horizon_label(hz_raw)
            rng_m = _try_parse_float(row["range_distance_m"]) if "range_distance_m" in row else None

            if flight_id not in flight_first_img:
                flight_first_img[flight_id] = img_name

            fkey = (flight_id, img_name)
            if fkey not in by_frame:
                by_frame[fkey] = {
                    "flight_id": flight_id,
                    "img_name": img_name,
                    "frame_idx": frame_idx,
                    "horizon": horizon,
                    "range_distance_m": rng_m,
                    "boxes": [],
                }
            elif by_frame[fkey]["horizon"] == "unknown" and horizon != "unknown":
                by_frame[fkey]["horizon"] = horizon

            obj_id = str(row["id"]).strip() if "id" in row and row["id"] is not None else ""
            l_raw = _try_parse_float(row["gt_left"]) if "gt_left" in row else None
            t_raw = _try_parse_float(row["gt_top"]) if "gt_top" in row else None
            r_raw = _try_parse_float(row["gt_right"]) if "gt_right" in row else None
            b_raw = _try_parse_float(row["gt_bottom"]) if "gt_bottom" in row else None

            if not obj_id and l_raw is None and t_raw is None and r_raw is None and b_raw is None:
                continue

            if l_raw is None or t_raw is None or r_raw is None or b_raw is None:
                res.dropped_degenerate_boxes += 1
                continue

            x1 = min(max(l_raw, 0.0), w_lim)
            y1 = min(max(t_raw, 0.0), h_lim)
            x2 = min(max(r_raw, 0.0), w_lim)
            y2 = min(max(b_raw, 0.0), h_lim)

            if x1 != l_raw or y1 != t_raw or x2 != r_raw or y2 != b_raw:
                res.clipped_border_boxes += 1

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0.0 or bh <= 0.0:
                res.dropped_degenerate_boxes += 1
                continue

            parsed = parse_aot_bbox_ltwh([x1, y1, bw, bh], img_w=img_w, img_h=img_h)
            if parsed.is_circle_approx:
                res.circle_approx_boxes += 1
            by_frame[fkey]["boxes"].append(parsed)

    for flight_id in sorted(flight_first_img.keys()):
        sample_img = flight_first_img[flight_id]
        day = extract_capture_day_utc(sample_img)
        res.flight_records.append(
            {
                "flight_id": flight_id,
                "part": part,
                "sample_filename": sample_img,
                "capture_day": day,
            }
        )

    day_lookup = {rec["flight_id"]: rec["capture_day"] for rec in res.flight_records}
    for (_, _), entry in sorted(by_frame.items(), key=lambda kv: (kv[0][0], int(kv[1]["frame_idx"]), kv[0][1])):
        ann = AotFrameAnnotation(
            part=part,
            flight_id=str(entry["flight_id"]),
            img_name=str(entry["img_name"]),
            frame_idx=int(entry["frame_idx"]),
            capture_day=str(day_lookup[str(entry["flight_id"])]),
            horizon=str(entry["horizon"]),
            boxes=tuple(entry["boxes"]),
            range_distance_m=entry["range_distance_m"],
        )
        if ann.is_negative:
            res.negative_frames.append(ann)
        else:
            res.positive_frames.append(ann)

    return res


def phase_correlate_gmc_warp(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align `prev_frame` onto `curr_frame` via translational phase correlation GMC."""
    prev_f = np.asarray(prev_frame, dtype=np.float32)
    curr_f = np.asarray(curr_frame, dtype=np.float32)
    if prev_f.shape != curr_f.shape or prev_f.ndim != 2:
        raise ValueError(f"Expected matching 2D grayscale frames, got {prev_f.shape} vs {curr_f.shape}")
    h, w = curr_f.shape
    hanning = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, _response = cv2.phaseCorrelate(prev_f, curr_f, hanning)
    dx = float(shift[0]) if math.isfinite(float(shift[0])) else 0.0
    dy = float(shift[1]) if math.isfinite(float(shift[1])) else 0.0
    dx = max(-float(w) * 0.25, min(float(w) * 0.25, dx))
    dy = max(-float(h) * 0.25, min(float(h) * 0.25, dy))

    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        prev_f,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    ones = np.ones((h, w), dtype=np.float32)
    mask = cv2.warpAffine(
        ones,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid_mask = (mask > 0.5).astype(np.float32)
    return warped, valid_mask


def affine_gmc_warp(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    *,
    max_corners: int = 200,
    min_inliers: int = 6,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Align `prev_frame` onto `curr_frame` via 2D Euclidean/Affine GMC (`estimateAffinePartial2D`) with phase-correlation fallback."""
    prev_f = np.asarray(prev_frame, dtype=np.float32)
    curr_f = np.asarray(curr_frame, dtype=np.float32)
    if prev_f.shape != curr_f.shape or prev_f.ndim != 2:
        raise ValueError(f"Expected matching 2D grayscale frames, got {prev_f.shape} vs {curr_f.shape}")
    h, w = curr_f.shape
    prev_u8 = np.clip(prev_f, 0.0, 255.0).astype(np.uint8)
    curr_u8 = np.clip(curr_f, 0.0, 255.0).astype(np.uint8)

    try:
        pts_prev = cv2.goodFeaturesToTrack(
            prev_u8,
            maxCorners=int(max_corners),
            qualityLevel=0.01,
            minDistance=max(4, min(h, w) // 24),
            blockSize=5,
        )
        if pts_prev is not None and len(pts_prev) >= int(min_inliers):
            pts_curr, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_u8,
                curr_u8,
                pts_prev,
                None,
                winSize=(15, 15),
                maxLevel=2,
            )
            if pts_curr is not None and status is not None:
                good_mask = status.reshape(-1) == 1
                p0 = pts_prev.reshape(-1, 2)[good_mask]
                p1 = pts_curr.reshape(-1, 2)[good_mask]
                if len(p0) >= int(min_inliers):
                    mat, inliers = cv2.estimateAffinePartial2D(
                        p0,
                        p1,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=200,
                    )
                    if mat is not None and inliers is not None and int(np.sum(inliers)) >= int(min_inliers):
                        a00, a01, tx = float(mat[0, 0]), float(mat[0, 1]), float(mat[0, 2])
                        ty = float(mat[1, 2])
                        scale = math.hypot(a00, a01)
                        if (
                            math.isfinite(scale)
                            and 0.90 <= scale <= 1.10
                            and abs(tx) <= 0.25 * float(w)
                            and abs(ty) <= 0.25 * float(h)
                        ):
                            mat_f = mat.astype(np.float32)
                            warped = cv2.warpAffine(
                                prev_f,
                                mat_f,
                                (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0,
                            )
                            ones = np.ones((h, w), dtype=np.float32)
                            mask = cv2.warpAffine(
                                ones,
                                mat_f,
                                (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0,
                            )
                            return warped, (mask > 0.5).astype(np.float32), "affine_partial2d"
    except cv2.error:
        pass

    warped_fb, mask_fb = phase_correlate_gmc_warp(prev_f, curr_f)
    return warped_fb, mask_fb, "phase_correlate_fallback"


def build_dual_interval_clip_tensor(
    frames_gray: Sequence[np.ndarray],
    *,
    clip_len: int | None = None,
    use_affine_gmc: bool = False,
) -> torch.Tensor:
    """Build `[T, 6, H, W]` dual-interval normalized tensor from a sequence of 2D grayscale frames."""
    n_frames = len(frames_gray)
    if n_frames == 0:
        raise ValueError("frames_gray must be non-empty")
    t_out = n_frames if clip_len is None else min(int(clip_len), n_frames)
    start = n_frames - t_out

    steps: list[torch.Tensor] = []
    for idx in range(start, n_frames):
        curr = np.asarray(frames_gray[idx], dtype=np.float32)
        idx_m1 = max(0, idx - 1)
        idx_m5 = max(0, idx - 5)
        if use_affine_gmc:
            w1, r1, _ = affine_gmc_warp(frames_gray[idx_m1], curr)
            w5, r5, _ = affine_gmc_warp(frames_gray[idx_m5], curr)
        else:
            w1, r1 = phase_correlate_gmc_warp(frames_gray[idx_m1], curr)
            w5, r5 = phase_correlate_gmc_warp(frames_gray[idx_m5], curr)

        t_curr = torch.from_numpy(curr).unsqueeze(0).unsqueeze(0)
        t_w1 = torch.from_numpy(w1).unsqueeze(0).unsqueeze(0)
        t_w5 = torch.from_numpy(w5).unsqueeze(0).unsqueeze(0)
        t_r1 = torch.from_numpy(r1).unsqueeze(0).unsqueeze(0)
        t_r5 = torch.from_numpy(r5).unsqueeze(0).unsqueeze(0)
        x_t = build_dual_interval_input(t_curr, t_w1, t_w5, t_r1, t_r5, already_normalized=False)
        steps.append(x_t[0])

    return torch.stack(steps, dim=0)


def _safe_extract_zip_clutter(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract zip archive with strict Zip-Slip path traversal protection."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            member_name = info.filename.replace("\\", "/")
            if member_name.startswith("/") or (len(member_name) >= 2 and member_name[1] == ":"):
                raise ValueError(f"Zip-Slip blocked (absolute path): {info.filename!r}")
            parts = [p for p in member_name.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts):
                raise ValueError(f"Zip-Slip blocked (traversal): {info.filename!r}")
            target_path = (dest_resolved / Path(*parts)).resolve() if parts else dest_resolved
            try:
                target_path.relative_to(dest_resolved)
            except ValueError as exc:
                raise ValueError(f"Zip-Slip blocked (outside destination): {info.filename!r}") from exc
        zf.extractall(dest_resolved)


def _count_images_under(root: Path) -> int:
    if not root.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def download_external_clutter_datasets(
    cache_dir: Path | str,
    *,
    allow_fallback: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Download MIT-licensed `fujii_distant_bird` & `det_fly` clutter datasets into `cache_dir`."""
    assert_dataset_allowed("fujii_distant_bird", "train")
    assert_dataset_allowed("det_fly", "train")

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    specs = (
        ("fujii_distant_bird", root / "fujii_distant_bird", FUJII_BIRD_MIRROR_URLS),
        ("det_fly", root / "det_fly", DET_FLY_MIRROR_URLS),
    )
    report: dict[str, Any] = {"cache_dir": str(root), "datasets": {}}
    for ds_id, ds_dir, urls in specs:
        existing = _count_images_under(ds_dir)
        if existing > 0:
            report["datasets"][ds_id] = {"dir": str(ds_dir), "images": existing, "status": "cached"}
            continue
        ds_dir.mkdir(parents=True, exist_ok=True)
        last_err = "no mirror attempted"
        downloaded = False
        for url in urls:
            zpath = root / f"{ds_id}.zip"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "OpenXVS-Trainer/1.1"})
                with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp, zpath.open("wb") as fout:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
                if zipfile.is_zipfile(zpath):
                    _safe_extract_zip_clutter(zpath, ds_dir)
                    zpath.unlink(missing_ok=True)
                    downloaded = True
                    break
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                zpath.unlink(missing_ok=True)
                continue
        n_imgs = _count_images_under(ds_dir)
        status = "downloaded" if (downloaded and n_imgs > 0) else f"unavailable ({last_err})"
        report["datasets"][ds_id] = {"dir": str(ds_dir), "images": n_imgs, "status": status}
        print(f"[XVS-D2 Clutter] {ds_id}: images={n_imgs} status={status} dir={ds_dir}", flush=True)

    total_imgs = sum(int(v["images"]) for v in report["datasets"].values())
    report["total_images"] = total_imgs
    if total_imgs == 0 and not allow_fallback:
        raise RuntimeError(
            f"Fail-closed: downloaded/discovered 0 real external bird/drone clutter images under {root} "
            f"(report={report['datasets']}). Pass allow_fallback=True to permit AOT/G3a-only fallback."
        )
    return report


def load_external_clutter_negative_clips(
    clutter_dirs: Sequence[Path | str],
    *,
    dataset_id: str,
    crop_hw: tuple[int, int] = (640, 640),
    clip_len: int = 5,
    max_clips: int = 500,
    use_affine_gmc: bool = True,
    seed: int = 20261121,
    lazy: bool | None = None,
) -> list[dict[str, Any]]:
    """Load real MIT-licensed clutter images (`fujii_distant_bird` / `det_fly`) as lazy hard-negative clips."""
    assert_dataset_allowed(dataset_id, "train")
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths: list[Path] = []
    for d in clutter_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for fp in sorted(p.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in img_exts:
                paths.append(fp)
    if not paths:
        return []

    rng = np.random.default_rng(int(seed))
    ch, cw = int(crop_hw[0]), int(crop_hw[1])
    t_len = int(clip_len)
    use_lazy = bool(ch * cw > 128 * 128) if lazy is None else bool(lazy)
    clips: list[dict[str, Any]] = []

    step = max(1, len(paths) // max(1, int(max_clips)))
    for idx, img_p in enumerate(paths[::step][: int(max_clips)]):
        pan_vx = float(rng.uniform(-1.5, 1.5))
        pan_vy = float(rng.uniform(-0.8, 0.8))
        crop_seed = int(rng.integers(0, 2**31 - 1))
        desc: dict[str, Any] = {
            "key": f"{dataset_id}_neg_{idx:05d}_{img_p.stem}",
            "lazy_kind": "clutter_image",
            "image_path": str(img_p),
            "crop_hw": [ch, cw],
            "clip_len": t_len,
            "pan_vx": pan_vx,
            "pan_vy": pan_vy,
            "crop_seed": crop_seed,
            "use_affine_gmc": bool(use_affine_gmc),
            "boxes_per_step": [None for _ in range(t_len)],
            "has_target": False,
            "horizon": "above",
            "source_domain": str(dataset_id),
        }
        if not use_lazy:
            desc["x_clip"] = materialize_clip_tensor(desc)
        clips.append(desc)
    return clips


def _render_g3a_clip_spec(
    *,
    height: int,
    width: int,
    clip_len: int,
    seed: int,
    has_target: bool,
    above_horizon: bool,
    clip_idx: int,
    use_affine_gmc: bool,
    v53_mode: bool,
    boxes_only: bool = False,
) -> tuple[torch.Tensor | None, list[dict[str, Any] | None]]:
    """Deterministically generate G3a clip tensor (or just `boxes_per_step` metadata with zero pixel RAM)."""
    rng = np.random.default_rng(int(seed) + int(clip_idx) * 10_007 + (0 if has_target else 5_000_011))
    h, w = int(height), int(width)
    t_len = int(clip_len)
    n_raw = t_len + 5
    horizon_y = int(h * rng.uniform(0.42, 0.58))

    cloud_phase = float(rng.uniform(0.0, 6.28)) if (v53_mode and not has_target) else 0.0
    pan_vx = float(rng.uniform(-0.8, 0.8))
    pan_vy = float(rng.uniform(-0.4, 0.4))
    roll_step_deg = float(rng.uniform(-0.35, 0.35)) if v53_mode else 0.0

    if has_target:
        margin = max(6.0, min(float(h), float(w)) * 0.14)
        cx0 = float(rng.uniform(margin, float(w) - margin))
        if above_horizon:
            y_hi = max(margin + 2.0, float(horizon_y) - 4.0)
            cy0 = float(rng.uniform(margin, y_hi))
        else:
            y_lo = min(float(h) - margin - 2.0, float(horizon_y) + 4.0)
            cy0 = float(rng.uniform(y_lo, float(h) - margin))
        tvx = float(rng.uniform(-0.9, 0.9))
        tvy = float(rng.uniform(-0.5, 0.5))
        if v53_mode and (clip_idx % 2 == 0):
            bw = float(rng.uniform(5.0, 13.0))
            bh = float(rng.uniform(5.0, 12.0))
        else:
            bw = float(rng.uniform(4.0, 11.0))
            bh = float(rng.uniform(4.0, 10.0))
        is_circle = (clip_idx % 5 == 0)
        if is_circle:
            bw, bh = 6.0, 6.0
    else:
        cx0, cy0, tvx, tvy, bw, bh, is_circle = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False

    neg_distractor = bool(v53_mode and (not has_target) and (clip_idx % 2 == 0))
    d_cx0 = float(rng.uniform(8.0, max(9.0, float(w) - 8.0))) if neg_distractor else 0.0
    d_cy0 = float(rng.uniform(6.0, max(7.0, float(horizon_y) - 3.0))) if neg_distractor else 0.0
    d_vx = float(rng.uniform(-1.8, 1.8)) if neg_distractor else 0.0
    d_vy = float(rng.uniform(-0.8, 0.8)) if neg_distractor else 0.0

    boxes_per_step: list[dict[str, Any] | None] = []
    if boxes_only:
        for k in range(n_raw - t_len, n_raw):
            if has_target:
                cx = float(np.clip(cx0 + tvx * k, 3.0, float(w) - 4.0))
                cy = float(np.clip(cy0 + tvy * k, 3.0, float(h) - 4.0))
                boxes_per_step.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "w": bw,
                        "h": bh,
                        "m_box": 0.0 if is_circle else 1.0,
                        "is_circle_approx": is_circle,
                    }
                )
            else:
                boxes_per_step.append(None)
        return None, boxes_per_step

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky_grad = 185.0 - 35.0 * (yy / float(max(h, 1)))
    ground_tex = 92.0 + 18.0 * np.sin(xx * 0.14) * np.cos(yy * 0.11)
    if v53_mode and not has_target:
        cloud_bank = 26.0 * np.sin(xx * 0.08 + cloud_phase) * np.cos(yy * 0.12)
        sky_grad = np.clip(sky_grad + cloud_bank, 30.0, 248.0)
        ground_tex = np.clip(
            ground_tex + 24.0 * np.sign(np.sin(xx * 0.25)) * (yy > float(horizon_y)),
            15.0,
            235.0,
        )
    base_bg = np.where(yy < float(horizon_y), sky_grad, ground_tex).astype(np.float32)

    raw_frames: list[np.ndarray] = []
    for k in range(n_raw):
        if abs(roll_step_deg) > 1e-4:
            shift_mat = cv2.getRotationMatrix2D((0.5 * w, 0.5 * h), roll_step_deg * k, 1.0).astype(np.float32)
            shift_mat[0, 2] += pan_vx * k
            shift_mat[1, 2] += pan_vy * k
        else:
            shift_mat = np.array([[1.0, 0.0, pan_vx * k], [0.0, 1.0, pan_vy * k]], dtype=np.float32)
        bg = cv2.warpAffine(
            base_bg,
            shift_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        noise = rng.normal(0.0, 2.2, size=(h, w)).astype(np.float32)
        frame = bg + noise
        if neg_distractor:
            dcx = float(np.clip(d_cx0 + d_vx * k, 3.0, float(w) - 4.0))
            dcy = float(np.clip(d_cy0 + d_vy * k + 1.5 * math.sin(1.4 * k), 3.0, float(h) - 4.0))
            d_blob = np.exp(-((xx - dcx) ** 2 + (yy - dcy) ** 2) / (2.0 * 1.4 * 1.4))
            pol = -38.0 if (clip_idx % 4 == 0) else 48.0
            frame = frame + (pol * d_blob).astype(np.float32)
        box_meta: dict[str, Any] | None = None
        if has_target:
            cx = float(np.clip(cx0 + tvx * k, 3.0, float(w) - 4.0))
            cy = float(np.clip(cy0 + tvy * k, 3.0, float(h) - 4.0))
            sigma = max(1.1, 0.25 * math.sqrt(bw * bh))
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
            polarity = -46.0 if above_horizon else 44.0
            frame = frame + (polarity * blob).astype(np.float32)
            box_meta = {
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
                "m_box": 0.0 if is_circle else 1.0,
                "is_circle_approx": is_circle,
            }
        raw_frames.append(np.clip(frame, 0.0, 255.0).astype(np.float32))
        if k >= n_raw - t_len:
            boxes_per_step.append(box_meta)

    x_clip = build_dual_interval_clip_tensor(
        raw_frames, clip_len=t_len, use_affine_gmc=use_affine_gmc
    )
    return x_clip, boxes_per_step


def materialize_clip_tensor(clip: Mapping[str, Any]) -> torch.Tensor:
    """Materialize `[T, 6, H, W]` FP32 clip tensor lazily from disk uint8 PNGs or deterministic G3a spec.

    Eliminates eager RAM retention of `[T, 6, 640, 640]` float32 tensors (`156-194 GiB` OOM) by loading
    uint8 frames from disk and applying `affine_gmc_warp` / `phase_correlate_gmc_warp` on the fly.
    """
    if "x_clip" in clip and clip["x_clip"] is not None:
        return clip["x_clip"].to(dtype=torch.float32)

    kind = str(clip["lazy_kind"]) if "lazy_kind" in clip else ""
    use_aff = bool(clip["use_affine_gmc"]) if "use_affine_gmc" in clip else False
    clip_len = int(clip["clip_len"]) if "clip_len" in clip else 5

    if kind == "aot_crop":
        ch, cw = int(clip["crop_hw"][0]), int(clip["crop_hw"][1])
        y0, x0 = int(clip["crop_origin"][0]), int(clip["crop_origin"][1])
        cropped_imgs: list[np.ndarray] = []
        for p_str in clip["frame_paths"]:
            im = cv2.imread(str(p_str), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise RuntimeError(f"Failed to decode AOT grayscale PNG: {p_str}")
            h_img, w_img = im.shape[:2]
            if ch >= h_img and cw >= w_img:
                patch = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)
            else:
                cy0 = max(0, min(y0, h_img - ch))
                cx0 = max(0, min(x0, w_img - cw))
                patch = im[cy0 : cy0 + ch, cx0 : cx0 + cw].astype(np.float32)
            cropped_imgs.append(patch)
        return build_dual_interval_clip_tensor(cropped_imgs, clip_len=clip_len, use_affine_gmc=use_aff)

    if kind == "aot_full":
        fh, fw = int(clip["full_hw"][0]), int(clip["full_hw"][1])
        native_imgs: list[np.ndarray] = []
        for p_str in clip["frame_paths"]:
            im = cv2.imread(str(p_str), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise RuntimeError(f"Failed to decode cached native AOT PNG: {p_str}")
            h_img, w_img = im.shape[:2]
            if h_img == fh and w_img == fw:
                frame_f = im.astype(np.float32)
            else:
                interp = cv2.INTER_AREA if (fh <= h_img and fw <= w_img) else cv2.INTER_LINEAR
                frame_f = cv2.resize(im, (fw, fh), interpolation=interp).astype(np.float32)
            native_imgs.append(frame_f)
        return build_dual_interval_clip_tensor(native_imgs, clip_len=clip_len, use_affine_gmc=use_aff)

    if kind == "clutter_image":
        ch, cw = int(clip["crop_hw"][0]), int(clip["crop_hw"][1])
        im = cv2.imread(str(clip["image_path"]), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to decode clutter image: {clip['image_path']}")
        ih, iw = im.shape[:2]
        if ih < ch or iw < cw:
            im = cv2.resize(im, (max(cw + 16, iw), max(ch + 16, ih)), interpolation=cv2.INTER_LINEAR)
            ih, iw = im.shape[:2]
        loc_rng = np.random.default_rng(int(clip["crop_seed"]))
        base_y = int(loc_rng.integers(0, max(1, ih - ch)))
        base_x = int(loc_rng.integers(0, max(1, iw - cw)))
        pan_vx = float(clip["pan_vx"])
        pan_vy = float(clip["pan_vy"])
        n_raw = clip_len + 5
        raw_frames: list[np.ndarray] = []
        for k in range(n_raw):
            x0 = max(0, min(iw - cw, int(round(base_x + pan_vx * k))))
            y0 = max(0, min(ih - ch, int(round(base_y + pan_vy * k))))
            raw_frames.append(im[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32))
        return build_dual_interval_clip_tensor(raw_frames, clip_len=clip_len, use_affine_gmc=use_aff)

    if kind == "g3a_synth":
        x_clip, _ = _render_g3a_clip_spec(
            height=int(clip["height"]),
            width=int(clip["width"]),
            clip_len=clip_len,
            seed=int(clip["g3a_seed"]),
            has_target=bool(clip["has_target"]),
            above_horizon=(str(clip.get("horizon", "above")) == "above"),
            clip_idx=int(clip["clip_idx"]),
            use_affine_gmc=use_aff,
            v53_mode=bool(clip.get("v53_mode", False)),
            boxes_only=False,
        )
        assert x_clip is not None
        return x_clip

    raise KeyError(f"Clip record missing 'x_clip' and unknown 'lazy_kind={kind!r}': {clip.get('key', '')!r}")


class LazyD2ClipDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Lazy disk-backed PyTorch `Dataset` for `XVS-D2` clips (zero eager RAM retention).

    `DataLoader` workers load uint8 frames from disk and apply `affine_gmc_warp` on the fly,
    keeping Colab host RAM strictly `< 16 GiB` even at `640x640` with thousands of clips.
    """

    def __init__(
        self,
        clip_records: Sequence[Mapping[str, Any]],
        *,
        target_hw: tuple[int, int],
    ) -> None:
        self.clip_records: list[dict[str, Any]] = [dict(c) for c in clip_records]
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))

    def __len__(self) -> int:
        return len(self.clip_records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        clip = self.clip_records[idx]
        x_c = materialize_clip_tensor(clip)
        th, tw = self.target_hw
        _, _, ch, cw = x_c.shape
        if ch != th or cw != tw:
            x_c = F.interpolate(x_c, size=(th, tw), mode="bilinear", align_corners=False)
            sx = float(tw) / float(cw)
            sy = float(th) / float(ch)
            new_steps: list[dict[str, Any] | None] = []
            for b in clip["boxes_per_step"]:
                if b is None:
                    new_steps.append(None)
                else:
                    new_steps.append(
                        {
                            "cx": float(b["cx"]) * sx,
                            "cy": float(b["cy"]) * sy,
                            "w": max(1.0, float(b["w"]) * sx),
                            "h": max(1.0, float(b["h"]) * sy),
                            "m_box": float(b["m_box"]),
                            "is_circle_approx": bool(b["is_circle_approx"]),
                        }
                    )
            meta = {
                "key": clip.get("key", ""),
                "boxes_per_step": new_steps,
                "has_target": bool(clip.get("has_target", False)),
                "horizon": str(clip.get("horizon", "unknown")),
                "source_domain": str(clip.get("source_domain", "")),
            }
            return x_c, meta
        meta = {
            "key": clip.get("key", ""),
            "boxes_per_step": clip["boxes_per_step"],
            "has_target": bool(clip.get("has_target", False)),
            "horizon": str(clip.get("horizon", "unknown")),
            "source_domain": str(clip.get("source_domain", "")),
        }
        return x_c, meta


def _collate_lazy_clip_batch(
    items: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    xs = [it[0] for it in items]
    metas = [it[1] for it in items]
    return torch.stack(xs, dim=0), metas


class SyntheticG3aClipDataset:
    """Deterministic OpenXVS G3a synthetic clip generator (`30%` mix + `--smoke-local` CPU mode).

    Uses lazy on-the-fly rendering when `height * width > 128 * 128` (`lazy=True`) so large
    `640x640` / `2048x2448` synthetic pools use zero host RAM.
    """

    def __init__(
        self,
        num_pos: int,
        num_neg: int,
        *,
        height: int = 64,
        width: int = 64,
        clip_len: int = 3,
        seed: int = 20261121,
        use_affine_gmc: bool = False,
        v53_mode: bool = False,
        lazy: bool | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.clip_len = int(clip_len)
        self.seed = int(seed)
        self.use_affine_gmc = bool(use_affine_gmc)
        self.v53_mode = bool(v53_mode)
        self.lazy = bool(self.height * self.width > 128 * 128) if lazy is None else bool(lazy)
        self.positive_clips: list[dict[str, Any]] = []
        self.negative_clips: list[dict[str, Any]] = []
        self._build(int(num_pos), int(num_neg))

    def _build_descriptor(
        self,
        *,
        has_target: bool,
        above_horizon: bool,
        clip_idx: int,
    ) -> dict[str, Any]:
        x_clip, boxes_per_step = _render_g3a_clip_spec(
            height=self.height,
            width=self.width,
            clip_len=self.clip_len,
            seed=self.seed,
            has_target=has_target,
            above_horizon=above_horizon,
            clip_idx=clip_idx,
            use_affine_gmc=self.use_affine_gmc,
            v53_mode=self.v53_mode,
            boxes_only=self.lazy,
        )
        rec: dict[str, Any] = {
            "key": f"g3a_{'pos' if has_target else 'neg'}_{clip_idx:05d}",
            "lazy_kind": "g3a_synth",
            "g3a_seed": self.seed,
            "height": self.height,
            "width": self.width,
            "clip_len": self.clip_len,
            "clip_idx": int(clip_idx),
            "use_affine_gmc": self.use_affine_gmc,
            "v53_mode": self.v53_mode,
            "boxes_per_step": boxes_per_step,
            "has_target": has_target,
            "horizon": "above" if above_horizon else "below",
            "source_domain": "g3a",
        }
        if x_clip is not None:
            rec["x_clip"] = x_clip
        return rec

    def _build(self, num_pos: int, num_neg: int) -> None:
        for i in range(num_pos):
            self.positive_clips.append(
                self._build_descriptor(has_target=True, above_horizon=(i % 2 == 0), clip_idx=i)
            )
        for j in range(num_neg):
            self.negative_clips.append(
                self._build_descriptor(has_target=False, above_horizon=(j % 2 == 0), clip_idx=j)
            )


class AotS3EncounterDataset:
    """Real Amazon AOT S3/HTTPS encounter dataset with strict UTC `capture_day` zero-leakage split."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        parts: Sequence[str] = ("part1",),
        crop_hw: tuple[int, int] = (640, 640),
        clip_len: int = 5,
        max_flights: int = 24,
        max_clips_per_flight: int = 4,
        max_neg_clips: int = 32,
        seed: int = 20261121,
        s3_bucket: str = AOT_S3_BUCKET,
        https_base: str = AOT_HTTPS_BASE,
        num_workers: int = 8,
        use_affine_gmc: bool = False,
    ) -> None:
        assert_dataset_allowed("aot", "train")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parts = tuple(str(p) for p in parts)
        self.crop_hw = (int(crop_hw[0]), int(crop_hw[1]))
        self.clip_len = int(clip_len)
        self.max_flights = int(max_flights)
        self.max_clips_per_flight = int(max_clips_per_flight)
        self.max_neg_clips = int(max_neg_clips)
        self.seed = int(seed)
        self.s3_bucket = str(s3_bucket)
        self.https_base = str(https_base)
        self.num_workers = max(1, int(num_workers))
        self.use_affine_gmc = bool(use_affine_gmc)

        self.parse_stats: dict[str, int] = {
            "dropped_degenerate_boxes": 0,
            "clipped_border_boxes": 0,
            "circle_approx_boxes": 0,
            "total_rows_read": 0,
        }
        self.split_flights: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.train_pos_clips: list[dict[str, Any]] = []
        self.train_pos_win_records: list[dict[str, Any]] = []
        self.train_neg_win_records: list[dict[str, Any]] = []
        self.train_neg_clips: list[dict[str, Any]] = []
        self.val_clips: list[dict[str, Any]] = []

    def build_native_full_frame_clips(
        self,
        full_hw: tuple[int, int] = (AOT_IMG_HEIGHT, AOT_IMG_WIDTH),
        *,
        max_clips: int | None = None,
        lazy: bool = False,
    ) -> list[dict[str, Any]]:
        """Build genuine native-resolution Stage-3 positive clips from cached AOT PNGs (ZERO crop upsampling).

        Pass `lazy=True` in G4 training to avoid eager RAM allocation of full-frame float32 tensors;
        default `lazy=False` materializes `x_clip` for backward-compatible callers and unit tests.
        """
        fh, fw = int(full_hw[0]), int(full_hw[1])
        records = self.train_pos_win_records if max_clips is None else self.train_pos_win_records[: int(max_clips)]
        full_clips: list[dict[str, Any]] = []
        for rec in records:
            paths: list[Path] = rec["paths"]
            win_anns: list[AotFrameAnnotation] = rec["anns"]
            native_imgs: list[np.ndarray] = []
            boxes_step: list[dict[str, Any] | None] = []
            if lazy:
                # Inspect first frame dimensions lazily (exact scaling even for non-2448x2048 caches)
                first_im = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE) if paths else None
                if first_im is not None:
                    h_img, w_img = first_im.shape[:2]
                    sx = float(fw) / float(w_img)
                    sy = float(fh) / float(h_img)
                else:
                    sx = float(fw) / float(AOT_IMG_WIDTH)
                    sy = float(fh) / float(AOT_IMG_HEIGHT)
                for ann in win_anns[-self.clip_len :]:
                    primary = ann.boxes[0] if ann.boxes else None
                    if primary is None:
                        boxes_step.append(None)
                    else:
                        boxes_step.append(
                            {
                                "cx": primary.cx * sx,
                                "cy": primary.cy * sy,
                                "w": max(1.0, primary.width * sx),
                                "h": max(1.0, primary.height * sy),
                                "m_box": primary.m_box,
                                "is_circle_approx": primary.is_circle_approx,
                            }
                        )
                full_clips.append(
                    {
                        "key": f"aot_full_{win_anns[-1].flight_id}_{win_anns[-1].frame_idx}",
                        "lazy_kind": "aot_full",
                        "frame_paths": [str(p) for p in paths],
                        "full_hw": [fh, fw],
                        "clip_len": self.clip_len,
                        "use_affine_gmc": self.use_affine_gmc,
                        "boxes_per_step": boxes_step,
                        "has_target": True,
                        "horizon": win_anns[-1].horizon,
                        "source_domain": "aot",
                        "capture_day": win_anns[-1].capture_day,
                        "native_hw": [fh, fw],
                    }
                )
            else:
                for idx_w, (p, ann) in enumerate(zip(paths, win_anns)):
                    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        raise RuntimeError(f"Failed to decode cached native AOT PNG: {p}")
                    h_img, w_img = im.shape[:2]
                    if h_img == fh and w_img == fw:
                        frame_f = im.astype(np.float32)
                        sx, sy = 1.0, 1.0
                    else:
                        interp = cv2.INTER_AREA if (fh <= h_img and fw <= w_img) else cv2.INTER_LINEAR
                        frame_f = cv2.resize(im, (fw, fh), interpolation=interp).astype(np.float32)
                        sx = float(fw) / float(w_img)
                        sy = float(fh) / float(h_img)
                    native_imgs.append(frame_f)
                    if idx_w >= len(win_anns) - self.clip_len:
                        primary = ann.boxes[0] if ann.boxes else None
                        if primary is None:
                            boxes_step.append(None)
                        else:
                            boxes_step.append(
                                {
                                    "cx": primary.cx * sx,
                                    "cy": primary.cy * sy,
                                    "w": max(1.0, primary.width * sx),
                                    "h": max(1.0, primary.height * sy),
                                    "m_box": primary.m_box,
                                    "is_circle_approx": primary.is_circle_approx,
                                }
                            )
                x_clip = build_dual_interval_clip_tensor(
                    native_imgs, clip_len=self.clip_len, use_affine_gmc=self.use_affine_gmc
                )
                full_clips.append(
                    {
                        "key": f"aot_full_{win_anns[-1].flight_id}_{win_anns[-1].frame_idx}",
                        "x_clip": x_clip,
                        "boxes_per_step": boxes_step,
                        "has_target": True,
                        "horizon": win_anns[-1].horizon,
                        "source_domain": "aot",
                        "capture_day": win_anns[-1].capture_day,
                        "native_hw": [fh, fw],
                    }
                )
        return full_clips

    def build_native_full_frame_neg_clips(
        self,
        full_hw: tuple[int, int] = (AOT_IMG_HEIGHT, AOT_IMG_WIDTH),
        *,
        max_clips: int = 8,
    ) -> list[dict[str, Any]]:
        """Build lazy disk-backed native-resolution Stage-3 negative clips to prevent HNM unlearning."""
        fh, fw = int(full_hw[0]), int(full_hw[1])
        records = self.train_neg_win_records[: max(0, int(max_clips))]
        neg_clips: list[dict[str, Any]] = []
        for rec in records:
            paths: list[Path] = rec["paths"]
            win_anns: list[AotFrameAnnotation] = rec["anns"]
            neg_clips.append(
                {
                    "key": f"aot_full_neg_{win_anns[-1].flight_id}_{win_anns[-1].frame_idx}",
                    "lazy_kind": "aot_full",
                    "frame_paths": [str(p) for p in paths],
                    "full_hw": [fh, fw],
                    "clip_len": self.clip_len,
                    "use_affine_gmc": self.use_affine_gmc,
                    "boxes_per_step": [None for _ in range(self.clip_len)],
                    "has_target": False,
                    "horizon": win_anns[-1].horizon,
                    "source_domain": "aot",
                    "capture_day": win_anns[-1].capture_day,
                    "native_hw": [fh, fw],
                }
            )
        return neg_clips

    def _download_key(self, s3_key: str) -> Path:
        dest = self.cache_dir / s3_key
        return download_s3_object_with_https_fallback(
            s3_key=s3_key,
            dest_path=dest,
            s3_bucket=self.s3_bucket,
            https_base=self.https_base,
            prefer_aws_cli=False,
        )

    def _crop_frame_and_box(
        self,
        img: np.ndarray,
        box: AotParsedBox | None,
        crop_origin: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        ch, cw = self.crop_hw
        y0, x0 = crop_origin
        h_img, w_img = img.shape[:2]
        if ch >= h_img and cw >= w_img:
            resized = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            sx = float(cw) / float(w_img)
            sy = float(ch) / float(h_img)
            if box is None:
                return resized.astype(np.float32), None
            return resized.astype(np.float32), {
                "cx": box.cx * sx,
                "cy": box.cy * sy,
                "w": max(1.0, box.width * sx),
                "h": max(1.0, box.height * sy),
                "m_box": box.m_box,
                "is_circle_approx": box.is_circle_approx,
            }

        y0 = max(0, min(y0, h_img - ch))
        x0 = max(0, min(x0, w_img - cw))
        patch = img[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
        if box is None:
            return patch, None
        rel_cx = box.cx - float(x0)
        rel_cy = box.cy - float(y0)
        if not (0.0 <= rel_cx < float(cw) and 0.0 <= rel_cy < float(ch)):
            return patch, None
        return patch, {
            "cx": rel_cx,
            "cy": rel_cy,
            "w": box.width,
            "h": box.height,
            "m_box": box.m_box,
            "is_circle_approx": box.is_circle_approx,
        }

    def prepare_from_s3(self) -> dict[str, Any]:
        """Download selected AOT metadata + bounded PNGs and build lazy disk-backed split descriptors."""
        rng = random.Random(self.seed)
        all_flight_records: list[dict[str, Any]] = []
        pos_by_flight: dict[str, list[AotFrameAnnotation]] = {}
        neg_by_flight: dict[str, list[AotFrameAnnotation]] = {}

        # Enforce GLOBAL max_flights cap across ALL parts (never 3x per-part multiplication!)
        remaining_flights = max(1, int(self.max_flights))
        total_selected_flights = 0
        for part_idx, part in enumerate(self.parts):
            if remaining_flights <= 0:
                break
            parts_left = max(1, len(self.parts) - part_idx)
            part_quota = max(1, (remaining_flights + parts_left - 1) // parts_left)
            enc_key = f"{part}/ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.csv"
            gt_key = f"{part}/ImageSets/groundtruth.csv"
            print(
                f"[XVS-D2 AOT] Downloading {part} ImageSets (global quota={part_quota}, rem={remaining_flights})...",
                flush=True,
            )
            enc_csv = self._download_key(enc_key)
            gt_csv = self._download_key(gt_key)

            encounters = parse_valid_encounters_csv(enc_csv)
            flight_ids = sorted({str(e["flight_id"]) for e in encounters})
            rng.shuffle(flight_ids)
            selected_flights = set(flight_ids[: min(part_quota, remaining_flights)])
            remaining_flights -= len(selected_flights)
            total_selected_flights += len(selected_flights)

            parsed = parse_aot_groundtruth_csv(
                gt_csv,
                part=part,
                allowed_flights=selected_flights,
            )
            self.parse_stats["dropped_degenerate_boxes"] += parsed.dropped_degenerate_boxes
            self.parse_stats["clipped_border_boxes"] += parsed.clipped_border_boxes
            self.parse_stats["circle_approx_boxes"] += parsed.circle_approx_boxes
            self.parse_stats["total_rows_read"] += parsed.total_rows_read
            print(
                f"[XVS-D2 AOT] Parsed {part}: {len(selected_flights)} flights (global total={total_selected_flights}/{self.max_flights}), "
                f"{len(parsed.positive_frames)} pos frames, {len(parsed.negative_frames)} neg frames "
                f"(clipped={parsed.clipped_border_boxes}, dropped={parsed.dropped_degenerate_boxes})",
                flush=True,
            )

            all_flight_records.extend(parsed.flight_records)
            for ann in parsed.positive_frames:
                pos_by_flight.setdefault(ann.flight_id, []).append(ann)
            for ann in parsed.negative_frames:
                neg_by_flight.setdefault(ann.flight_id, []).append(ann)

        if not all_flight_records:
            raise RuntimeError("No AOT flights found after parsing ImageSets CSVs.")

        self.split_flights = split_flights_by_capture_day(all_flight_records, seed=self.seed)
        flight_to_split: dict[str, str] = {}
        for split_name, recs in self.split_flights.items():
            for rec in recs:
                flight_to_split[str(rec["flight_id"])] = split_name

        # Plan positive & disk-budgeted negative clip windows (reuse up to 4 distinct 640x640 spatial
        # tiles per full-res 2448x2048 negative sequence to cut PNG disk/download 4x!)
        pos_plan: list[tuple[str, str, list[AotFrameAnnotation]]] = []
        for fid, frames in sorted(pos_by_flight.items()):
            split_name = flight_to_split[fid] if fid in flight_to_split else "train"
            frames_sorted = sorted(frames, key=lambda a: a.frame_idx)
            if len(frames_sorted) < self.clip_len:
                continue
            stride = max(1, (len(frames_sorted) - self.clip_len) // max(1, self.max_clips_per_flight))
            built = 0
            for start_i in range(0, len(frames_sorted) - self.clip_len + 1, stride):
                if built >= self.max_clips_per_flight:
                    break
                win_start = max(0, start_i - 5)
                win_anns = frames_sorted[win_start : start_i + self.clip_len]
                pos_plan.append((fid, split_name, win_anns))
                built += 1

        crops_per_neg_win = 4 if (self.max_neg_clips >= 64 and self.crop_hw[0] <= 1024 and self.crop_hw[1] <= 1224) else 1
        max_neg_windows = max(1, (self.max_neg_clips + crops_per_neg_win - 1) // crops_per_neg_win)
        neg_win_plan: list[tuple[str, str, list[AotFrameAnnotation]]] = []
        n_neg_flights = max(1, len(neg_by_flight))
        max_neg_win_per_flight = max(1, (max_neg_windows + n_neg_flights - 1) // n_neg_flights)
        win_len_neg = self.clip_len + 5
        for fid, frames in sorted(neg_by_flight.items()):
            if len(neg_win_plan) >= max_neg_windows:
                break
            split_name = flight_to_split[fid] if fid in flight_to_split else "train"
            frames_sorted = sorted(frames, key=lambda a: a.frame_idx)
            if len(frames_sorted) < self.clip_len:
                continue
            if max_neg_win_per_flight <= 1 or len(frames_sorted) <= win_len_neg:
                win_anns = frames_sorted[:win_len_neg]
                neg_win_plan.append((fid, split_name, win_anns))
            else:
                stride_neg = max(self.clip_len, (len(frames_sorted) - win_len_neg) // max(1, max_neg_win_per_flight))
                taken = 0
                for s_i in range(0, len(frames_sorted) - self.clip_len + 1, stride_neg):
                    if taken >= max_neg_win_per_flight or len(neg_win_plan) >= max_neg_windows:
                        break
                    win_anns = frames_sorted[s_i : s_i + win_len_neg]
                    neg_win_plan.append((fid, split_name, win_anns))
                    taken += 1

        unique_keys = sorted(
            {
                f"{a.part}/Images/{a.flight_id}/{a.img_name}"
                for _fid, _sp, anns in (pos_plan + neg_win_plan)
                for a in anns
            }
        )
        total_pngs = len(unique_keys)
        prefetch_workers = max(24, self.num_workers)
        key_to_path: dict[str, Path] = {}
        if total_pngs > 0:
            print(
                f"[XVS-D2 AOT] Bulk-prefetching {total_pngs} unique PNGs via direct HTTPS "
                f"(workers={prefetch_workers}, crops_per_neg_win={crops_per_neg_win}, lazy_ram=True)...",
                flush=True,
            )
            report_step = max(1, total_pngs // 10)
            completed = 0
            with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
                fut_to_key = {pool.submit(self._download_key, k): k for k in unique_keys}
                for fut in as_completed(fut_to_key):
                    s3_key = fut_to_key[fut]
                    key_to_path[s3_key] = fut.result()
                    completed += 1
                    if completed % report_step == 0 or completed == total_pngs:
                        pct = 100.0 * float(completed) / float(total_pngs)
                        print(
                            f"[XVS-D2 AOT] Prefetched {completed}/{total_pngs} PNGs ({pct:.1f}%)...",
                            flush=True,
                        )

        ch, cw = self.crop_hw
        # Build LAZY positive encounter descriptors (ZERO eager float32 tensors in RAM!)
        for fid, split_name, win_anns in pos_plan:
            win_paths = [key_to_path[f"{a.part}/Images/{a.flight_id}/{a.img_name}"] for a in win_anns]
            anchor_box = win_anns[-1].boxes[0]
            y0 = max(0, min(AOT_IMG_HEIGHT - ch, int(anchor_box.cy - 0.5 * ch)))
            x0 = max(0, min(AOT_IMG_WIDTH - cw, int(anchor_box.cx - 0.5 * cw)))
            boxes_step: list[dict[str, Any] | None] = []
            for ann in win_anns[-self.clip_len :]:
                primary = ann.boxes[0] if ann.boxes else None
                if primary is None:
                    boxes_step.append(None)
                else:
                    rel_cx = primary.cx - float(x0)
                    rel_cy = primary.cy - float(y0)
                    if 0.0 <= rel_cx < float(cw) and 0.0 <= rel_cy < float(ch):
                        boxes_step.append(
                            {
                                "cx": rel_cx,
                                "cy": rel_cy,
                                "w": primary.width,
                                "h": primary.height,
                                "m_box": primary.m_box,
                                "is_circle_approx": primary.is_circle_approx,
                            }
                        )
                    else:
                        boxes_step.append(None)
            clip_obj: dict[str, Any] = {
                "key": f"aot_pos_{fid}_{win_anns[-1].frame_idx}",
                "lazy_kind": "aot_crop",
                "frame_paths": [str(p) for p in win_paths],
                "crop_origin": [y0, x0],
                "crop_hw": [ch, cw],
                "clip_len": self.clip_len,
                "use_affine_gmc": self.use_affine_gmc,
                "boxes_per_step": boxes_step,
                "has_target": True,
                "horizon": win_anns[-1].horizon,
                "source_domain": "aot",
                "capture_day": win_anns[-1].capture_day,
            }
            if split_name == "train":
                self.train_pos_clips.append(clip_obj)
                self.train_pos_win_records.append({"paths": win_paths, "anns": win_anns})
            else:
                self.val_clips.append(clip_obj)

        # Build LAZY target-free negative descriptors (multi-crop spatial tiles per prefetched window)
        neg_built = 0
        for fid, split_name, win_anns in neg_win_plan:
            neg_paths = [key_to_path[f"{a.part}/Images/{a.flight_id}/{a.img_name}"] for a in win_anns]
            if split_name == "train":
                self.train_neg_win_records.append({"paths": neg_paths, "anns": win_anns})
            for crop_i in range(crops_per_neg_win):
                if neg_built >= self.max_neg_clips:
                    break
                y0 = rng.randint(0, max(0, AOT_IMG_HEIGHT - ch))
                x0 = rng.randint(0, max(0, AOT_IMG_WIDTH - cw))
                clip_obj = {
                    "key": f"aot_neg_{fid}_{win_anns[-1].frame_idx}_c{crop_i}",
                    "lazy_kind": "aot_crop",
                    "frame_paths": [str(p) for p in neg_paths],
                    "crop_origin": [y0, x0],
                    "crop_hw": [ch, cw],
                    "clip_len": self.clip_len,
                    "use_affine_gmc": self.use_affine_gmc,
                    "boxes_per_step": [None for _ in range(self.clip_len)],
                    "has_target": False,
                    "horizon": win_anns[-1].horizon,
                    "source_domain": "aot",
                    "capture_day": win_anns[-1].capture_day,
                }
                if split_name == "train":
                    self.train_neg_clips.append(clip_obj)
                else:
                    self.val_clips.append(clip_obj)
                neg_built += 1

        print(
            f"[XVS-D2] Indexed {len(self.train_pos_clips)} train_pos, {len(self.train_neg_clips)} train_neg, "
            f"{len(self.val_clips)} val lazy AOT clips (0 float32 clips held in RAM; global flights={total_selected_flights})",
            flush=True,
        )
        return {
            "train_pos_clips": len(self.train_pos_clips),
            "train_neg_clips": len(self.train_neg_clips),
            "val_clips": len(self.val_clips),
            "global_flights_selected": total_selected_flights,
            "unique_pngs_prefetched": total_pngs,
            "lazy_disk_backed": True,
            "parse_stats": dict(self.parse_stats),
            "split_flight_counts": {k: len(v) for k, v in self.split_flights.items()},
        }


@dataclass
class XvsD2G4TrainConfig:
    """Configuration for 4-stage `XVS-D2` Colab G4 training & `--smoke-local` verification."""

    out_dir: Path
    drive_ckpt_dir: Path | None = None
    smoke_local: bool = False
    parts: tuple[str, ...] = ("part1", "part2", "part3")
    aot_cache_dir: Path | None = None
    max_flights: int = 36
    max_full_frame_clips: int = 16
    crop_hw: tuple[int, int] = (640, 640)
    full_frame_hw: tuple[int, int] = (2048, 2448)
    clip_len: int = 5
    epochs_stage1_warmup: int = 20
    epochs_stage2_hnm: int = 20
    epochs_stage3_fullframe: int = 5
    epochs_stage4_distill: int = 35
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_torch_compile: bool = True
    student_tiers: tuple[str, ...] = ("edge", "edge_slim", "web")
    vram_target_frac: float = 0.86
    vram_headroom_gib: float = 8.0
    seed: int = 20261121
    v53_mode: bool = False
    max_neg_clips: int = 32
    stage4_neg_ratio: float = 0.15
    small_target_oversample_prob: float = 0.0
    use_affine_gmc: bool = False
    hnm_refresh_every_epochs: int = 5
    fujii_bird_dirs: tuple[Path, ...] = ()
    det_fly_dirs: tuple[Path, ...] = ()
    download_external_clutter: bool = False
    require_external_clutter: bool = False
    allow_external_clutter_fallback: bool = False
    onnx_export_hw: tuple[int, int] | None = None


def _build_step_targets(
    batch_clips: Sequence[Mapping[str, Any]],
    step_idx: int,
    h_sz: int,
    w_sz: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, bool]]]:
    """Build stride-2 target tensors for one temporal step `step_idx`."""
    b_sz = len(batch_clips)
    hs = h_sz // 2
    ws = w_sz // 2
    gt_hm = torch.zeros((b_sz, 1, hs, ws), dtype=torch.float32, device=device)
    gt_boxes = torch.zeros((b_sz, 4), dtype=torch.float32, device=device)
    box_mask = torch.zeros((b_sz,), dtype=torch.float32, device=device)
    gt_mean = torch.zeros((b_sz, 3), dtype=torch.float32, device=device)
    mean_mask = torch.zeros((b_sz,), dtype=torch.float32, device=device)
    cell_coords: list[tuple[int, int, bool]] = []

    yy = torch.arange(hs, dtype=torch.float32, device=device).unsqueeze(1)
    xx = torch.arange(ws, dtype=torch.float32, device=device).unsqueeze(0)

    for bi, clip in enumerate(batch_clips):
        steps = clip["boxes_per_step"]
        box = steps[step_idx] if step_idx < len(steps) else None
        if box is None:
            cell_coords.append((0, 0, False))
            continue

        cx_px = float(box["cx"])
        cy_px = float(box["cy"])
        bw_px = max(1.0, float(box["w"]))
        bh_px = max(1.0, float(box["h"]))
        m_box = float(box["m_box"])

        cell_x_f = cx_px / 2.0 - 0.5
        cell_y_f = cy_px / 2.0 - 0.5
        ix = max(0, min(ws - 1, int(round(cell_x_f))))
        iy = max(0, min(hs - 1, int(round(cell_y_f))))

        sigma = max(1.0, min(4.0, 0.25 * math.sqrt(bw_px * bh_px) / 2.0))
        gauss = torch.exp(-((xx - float(ix)) ** 2 + (yy - float(iy)) ** 2) / (2.0 * sigma * sigma))
        gt_hm[bi, 0] = torch.maximum(gt_hm[bi, 0], gauss)
        gt_hm[bi, 0, iy, ix] = 1.0

        gt_boxes[bi, 0] = cx_px
        gt_boxes[bi, 1] = cy_px
        gt_boxes[bi, 2] = bw_px
        gt_boxes[bi, 3] = bh_px
        box_mask[bi] = m_box

        gt_mean[bi, 0] = max(-1.5, min(1.5, cell_x_f - float(ix)))
        gt_mean[bi, 1] = max(-1.5, min(1.5, cell_y_f - float(iy)))
        gt_mean[bi, 2] = math.log(max(1.0, math.sqrt(bw_px * bh_px)))
        mean_mask[bi] = 1.0
        cell_coords.append((iy, ix, True))

    return gt_hm, gt_boxes, box_mask, gt_mean, mean_mask, cell_coords


def _gather_predictions_at_cells(
    out: Mapping[str, Any],
    cell_coords: Sequence[tuple[int, int, bool]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather `[B, 4]` predicted boxes, `[B, 3]` mean targets, and `[B, 3]` log-vars at GT cells."""
    off = out["offset"].to(dtype=torch.float32)
    lsc = out["log_scale"].to(dtype=torch.float32)
    lvr = out["log_var"].to(dtype=torch.float32)
    b_sz = off.shape[0]
    dev = off.device

    pred_boxes = torch.zeros((b_sz, 4), dtype=torch.float32, device=dev)
    pred_mean = torch.zeros((b_sz, 3), dtype=torch.float32, device=dev)
    pred_lv = torch.zeros((b_sz, 3), dtype=torch.float32, device=dev)

    for bi, (iy, ix, _is_pos) in enumerate(cell_coords):
        du = off[bi, 0, iy, ix]
        dv = off[bi, 1, iy, ix]
        ln_s = lsc[bi, 0, iy, ix].clamp(min=-2.0, max=5.0)
        ln_r = lsc[bi, 1, iy, ix].clamp(min=-2.0, max=2.0)

        pred_cx = (float(ix) + du + 0.5) * 2.0
        pred_cy = (float(iy) + dv + 0.5) * 2.0
        scale = torch.exp(ln_s)
        aspect = torch.exp(0.5 * ln_r)
        pred_w = scale * aspect
        pred_h = scale / aspect

        pred_boxes[bi, 0] = pred_cx
        pred_boxes[bi, 1] = pred_cy
        pred_boxes[bi, 2] = pred_w
        pred_boxes[bi, 3] = pred_h

        pred_mean[bi, 0] = du
        pred_mean[bi, 1] = dv
        pred_mean[bi, 2] = ln_s
        pred_lv[bi] = lvr[bi, :, iy, ix]

    return pred_boxes, pred_mean, pred_lv


def _resize_clip_batch(
    batch_clips: Sequence[Mapping[str, Any]],
    target_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Materialize & stack clips to `[B, T, 6, H, W]`, resizing spatial dims and scaling GT boxes if needed."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    xs: list[torch.Tensor] = []
    scaled_clips: list[dict[str, Any]] = []
    for clip in batch_clips:
        x_c = materialize_clip_tensor(clip)
        _, _, ch, cw = x_c.shape
        if ch != th or cw != tw:
            x_resized = F.interpolate(x_c, size=(th, tw), mode="bilinear", align_corners=False)
            sx = float(tw) / float(cw)
            sy = float(th) / float(ch)
            new_steps: list[dict[str, Any] | None] = []
            for b in clip["boxes_per_step"]:
                if b is None:
                    new_steps.append(None)
                else:
                    new_steps.append(
                        {
                            "cx": float(b["cx"]) * sx,
                            "cy": float(b["cy"]) * sy,
                            "w": max(1.0, float(b["w"]) * sx),
                            "h": max(1.0, float(b["h"]) * sy),
                            "m_box": float(b["m_box"]),
                            "is_circle_approx": bool(b["is_circle_approx"]),
                        }
                    )
            xs.append(x_resized)
            scaled_clips.append({**clip, "boxes_per_step": new_steps})
        else:
            xs.append(x_c)
            scaled_clips.append(dict(clip))
    return torch.stack(xs, dim=0).to(device=device, non_blocking=(device.type == "cuda")), scaled_clips


def _train_one_clip_batch_tensors(
    model_fn: nn.Module,
    x_seq: torch.Tensor,
    scaled_clips: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    use_bf16: bool,
    teacher_model: XvsD2Model | None = None,
    cscwd_adapters: nn.ModuleDict | None = None,
    fp2_adapter: nn.Module | None = None,
    lambda_kd: float = 0.0,
) -> torch.Tensor:
    """Execute unrolled ConvGRU forward on pre-materialized `[B, T, 6, H, W]` in `bf16` with FP32 loss."""
    x_seq = x_seq.to(device=device, non_blocking=(device.type == "cuda"))
    _, t_len, _, h_sz, w_sz = x_seq.shape
    dev_type = "cuda" if device.type == "cuda" else "cpu"

    h_prev: torch.Tensor | None = None
    h_teacher: torch.Tensor | None = None
    total_loss = torch.zeros((), dtype=torch.float32, device=device)

    for t in range(t_len):
        x_t = x_seq[:, t]
        need_pyr = teacher_model is not None and float(lambda_kd) > 0.0

        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model_fn(x_t, h_prev, return_pyramid=need_pyr)

        h_prev = out["h_next"]
        gt_hm, gt_boxes, box_mask, gt_mean, mean_mask, coords = _build_step_targets(
            scaled_clips, t, h_sz, w_sz, device
        )
        pred_boxes, pred_mean, pred_lv = _gather_predictions_at_cells(out, coords)

        kd_term: torch.Tensor | None = None
        if need_pyr and teacher_model is not None:
            with torch.no_grad():
                with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                    t_out = teacher_model(x_t, h_teacher, return_pyramid=True)
            h_teacher = t_out["h_next"]
            l_cscwd = cscwd_distillation_loss(
                out["pyramid"],
                t_out["pyramid"],
                adapters=cscwd_adapters,
            )
            l_crosskd = cross_kd_loss(
                out["pyramid"]["p2"],
                teacher_model.heads,
                t_out,
                fp2_adapter=fp2_adapter,
            )
            kd_term = l_cscwd + l_crosskd

        loss_dict = compute_xvs_d2_composite_loss(
            hm_logits=out["hm_logits"],
            gt_heatmap=gt_hm,
            pred_boxes_cxcywh=pred_boxes,
            gt_boxes_cxcywh=gt_boxes,
            pred_mean=pred_mean,
            gt_mean=gt_mean,
            pred_log_var=pred_lv,
            box_mask=box_mask,
            mean_mask=mean_mask,
            kd_loss=kd_term,
            lambda_kd=lambda_kd,
        )
        total_loss = total_loss + loss_dict["total"]

    return total_loss / float(max(1, t_len))


def _train_one_clip_batch(
    model_fn: nn.Module,
    batch_clips: Sequence[Mapping[str, Any]],
    *,
    target_hw: tuple[int, int],
    device: torch.device,
    use_bf16: bool,
    teacher_model: XvsD2Model | None = None,
    cscwd_adapters: nn.ModuleDict | None = None,
    fp2_adapter: nn.Module | None = None,
    lambda_kd: float = 0.0,
) -> torch.Tensor:
    """Materialize clips and execute unrolled ConvGRU forward in `bf16` with strictly FP32 loss."""
    x_seq, scaled_clips = _resize_clip_batch(batch_clips, target_hw, device)
    return _train_one_clip_batch_tensors(
        model_fn,
        x_seq,
        scaled_clips,
        device=device,
        use_bf16=use_bf16,
        teacher_model=teacher_model,
        cscwd_adapters=cscwd_adapters,
        fp2_adapter=fp2_adapter,
        lambda_kd=lambda_kd,
    )


def _make_epoch_dataloader(
    flat_clips: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    target_hw: tuple[int, int],
    device: torch.device,
    smoke_local: bool,
) -> DataLoader[tuple[torch.Tensor, list[dict[str, Any]]]]:
    """Build a worker-backed lazy `DataLoader` that reads uint8 PNGs & runs GMC on the fly."""
    n_workers = 0
    if device.type == "cuda" and (not smoke_local) and os.name != "nt":
        n_workers = min(8, max(2, (os.cpu_count() or 4) // 4))
    ds = LazyD2ClipDataset(flat_clips, target_hw=target_hw)
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": n_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": _collate_lazy_clip_batch,
    }
    if n_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def _probe_batch_eager(
    model: XvsD2Model,
    sample_pool: Sequence[Mapping[str, Any]],
    *,
    target_hw: tuple[int, int],
    candidates: Sequence[int],
    device: torch.device,
    use_bf16: bool,
    target_frac: float,
    headroom_gib: float,
    teacher_model: XvsD2Model | None = None,
    cscwd_adapters: nn.ModuleDict | None = None,
    fp2_adapter: nn.Module | None = None,
    lambda_kd: float = 0.0,
) -> dict[str, Any]:
    """Probe optimal batch size on EAGER model BEFORE `torch.compile`, then restore weights (`BC-G4-m4`).

    When `teacher_model` and `lambda_kd > 0.0` are provided (Stage 4 CSCWD + CrossKD),
    the probe executes the true distillation forward+backward graph and snapshots/restores
    `cscwd_adapters` and `fp2_adapter` alongside `model`.
    """
    tuner = AutoVRAMBatchTuner(
        target_frac=target_frac,
        safety_headroom_gib=headroom_gib,
        device=device,
    )
    saved_model = snapshot_eager_state(model)
    saved_cscwd = snapshot_eager_state(cscwd_adapters) if cscwd_adapters is not None else None
    saved_fp2 = snapshot_eager_state(fp2_adapter) if fp2_adapter is not None else None
    teacher_was_training = bool(teacher_model.training) if teacher_model is not None else False

    model.train()
    if cscwd_adapters is not None:
        cscwd_adapters.train()
    if fp2_adapter is not None:
        fp2_adapter.train()
    if teacher_model is not None:
        teacher_model.eval()

    def _zero_probe_grads() -> None:
        model.zero_grad(set_to_none=True)
        if cscwd_adapters is not None:
            cscwd_adapters.zero_grad(set_to_none=True)
        if fp2_adapter is not None:
            fp2_adapter.zero_grad(set_to_none=True)
        if teacher_model is not None:
            teacher_model.zero_grad(set_to_none=True)

    def _probe_step(bs: int) -> None:
        batch = [sample_pool[i % len(sample_pool)] for i in range(bs)]
        _zero_probe_grads()
        try:
            loss = _train_one_clip_batch(
                model,
                batch,
                target_hw=target_hw,
                device=device,
                use_bf16=use_bf16,
                teacher_model=teacher_model,
                cscwd_adapters=cscwd_adapters,
                fp2_adapter=fp2_adapter,
                lambda_kd=lambda_kd,
            )
            loss.backward()
            del loss
        finally:
            _zero_probe_grads()

    try:
        report = tuner.probe_batch_size(_probe_step, candidates)
    finally:
        _zero_probe_grads()
        restore_eager_state(model, saved_model)
        if cscwd_adapters is not None and saved_cscwd is not None:
            restore_eager_state(cscwd_adapters, saved_cscwd)
        if fp2_adapter is not None and saved_fp2 is not None:
            restore_eager_state(fp2_adapter, saved_fp2)
        if teacher_model is not None and teacher_was_training:
            teacher_model.train()
    return report


@torch.no_grad()
def _mine_hard_negatives(
    model: nn.Module,
    neg_clips: Sequence[Mapping[str, Any]],
    *,
    target_hw: tuple[int, int],
    device: torch.device,
    use_bf16: bool,
    top_k: int,
    smoke_local: bool = False,
) -> list[dict[str, Any]]:
    """Score lazy target-free negative clips via DataLoader workers and select highest FP-score clips."""
    if not neg_clips:
        return []
    was_training = model.training
    model.eval()
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    scored: list[tuple[float, dict[str, Any]]] = []
    batch_sz = 1 if smoke_local else min(16, max(1, len(neg_clips)))
    loader = _make_epoch_dataloader(
        neg_clips, batch_size=batch_sz, target_hw=target_hw, device=device, smoke_local=smoke_local
    )
    offset = 0
    for x_seq, _metas in loader:
        x_dev = x_seq.to(device=device, non_blocking=(device.type == "cuda"))
        b_cur = int(x_dev.shape[0])
        h_prev: torch.Tensor | None = None
        max_probs = torch.zeros((b_cur,), dtype=torch.float32, device=device)
        for t in range(x_dev.shape[1]):
            with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(x_dev[:, t], h_prev, return_pyramid=False)
            h_prev = out["h_next"]
            step_max = torch.sigmoid(out["hm_logits"].to(dtype=torch.float32)).amax(dim=(1, 2, 3))
            max_probs = torch.maximum(max_probs, step_max)
        for bi, prob_val in enumerate(max_probs.detach().cpu().tolist()):
            scored.append((float(prob_val), dict(neg_clips[offset + bi])))
        offset += b_cur
    if was_training:
        model.train()
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[: max(1, min(int(top_k), len(scored)))]]


@torch.no_grad()
def evaluate_xvs_d2(
    model: XvsD2Model,
    eval_clips: Sequence[Mapping[str, Any]],
    *,
    target_hw: tuple[int, int],
    device: torch.device,
    match_radius_px: float = 12.0,
    op_score_thr: float = 0.50,
) -> dict[str, Any]:
    """Compute real `Pd@FPI={0.1, 0.5, 1.0}`, `AP50`, operational `>= op_score_thr` `above/below_horizon_pd`, `centroid_rmse_px` & `clutter_raw_frame_fpr`."""
    raw = getattr(model, "_orig_mod", model)
    was_training = raw.training
    raw.eval()

    pos_best_scores: list[tuple[float, str, float]] = []
    all_detections: list[tuple[float, bool]] = []
    neg_det_scores: list[float] = []
    num_pos = 0
    num_neg = 0
    neg_frames_with_fp = 0
    op_thr = float(op_score_thr)

    for clip in eval_clips:
        x_seq, scaled = _resize_clip_batch([clip], target_hw, device)
        sc = scaled[0]
        h_prev: torch.Tensor | None = None
        last_out: dict[str, Any] = {}
        for t in range(x_seq.shape[1]):
            last_out = raw(x_seq[:, t], h_prev, return_pyramid=False)
            h_prev = last_out["h_next"]

        dets = decode_peaks_local_softargmax(
            last_out["hm_logits"],
            last_out["offset"],
            last_out["log_scale"],
            last_out["log_var"],
            last_out["peak_mask"],
            score_thr=1e-4,
            max_k=32,
        )
        gt_box = sc["boxes_per_step"][-1]
        horizon = str(sc["horizon"]) if "horizon" in sc else "unknown"

        if gt_box is None:
            num_neg += 1
            if any(float(d["score"]) >= op_thr for d in dets):
                neg_frames_with_fp += 1
            for d in dets:
                s = float(d["score"])
                neg_det_scores.append(s)
                all_detections.append((s, False))
        else:
            num_pos += 1
            gt_cx = float(gt_box["cx"])
            gt_cy = float(gt_box["cy"])
            matched = False
            best_score = -1.0
            best_err = float("inf")
            for d in dets:
                dist = math.hypot(float(d["u_px"]) - gt_cx, float(d["v_px"]) - gt_cy)
                s = float(d["score"])
                if not matched and dist <= float(match_radius_px):
                    matched = True
                    best_score = s
                    best_err = dist
                    all_detections.append((s, True))
                else:
                    neg_det_scores.append(s)
                    all_detections.append((s, False))
            pos_best_scores.append((best_score, horizon, best_err))

    if was_training:
        raw.train()

    n_imgs = max(1, len(eval_clips))
    neg_det_scores.sort(reverse=True)

    def _pd_at_fpi(fpi_target: float) -> float:
        if num_pos == 0:
            return 0.0
        allowed_fp = int(math.floor(float(fpi_target) * float(n_imgs)))
        if allowed_fp >= len(neg_det_scores):
            thr = 0.0
        else:
            thr = neg_det_scores[allowed_fp] + 1e-12
        hits = sum(1 for sc_val, _, _ in pos_best_scores if sc_val >= thr and sc_val >= 0.0)
        return float(hits) / float(num_pos)

    # Average Precision (AP50-equivalent centroid match PR curve)
    all_detections.sort(key=lambda pair: pair[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    recalls: list[float] = [0.0]
    precisions: list[float] = [1.0]
    for _sc, is_tp in all_detections:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        if num_pos > 0:
            recalls.append(float(tp_cum) / float(num_pos))
            precisions.append(float(tp_cum) / float(max(1, tp_cum + fp_cum)))
    for i in range(len(precisions) - 2, -1, -1):
        if precisions[i] < precisions[i + 1]:
            precisions[i] = precisions[i + 1]
    ap50 = 0.0
    for i in range(1, len(recalls)):
        ap50 += (recalls[i] - recalls[i - 1]) * precisions[i]

    # Operational threshold (>= 0.50) prevents score_thr=1e-4 from inflating Pd to 1.00!
    matched_errs = [err for sc_val, _, err in pos_best_scores if sc_val >= op_thr and math.isfinite(err)]
    centroid_rmse_px = (
        math.sqrt(sum(e * e for e in matched_errs) / float(len(matched_errs))) if matched_errs else 0.0
    )

    above_pos = [sc_val for sc_val, hz, _ in pos_best_scores if hz == "above"]
    below_pos = [sc_val for sc_val, hz, _ in pos_best_scores if hz == "below"]
    above_pd = (
        float(sum(1 for s in above_pos if s >= op_thr)) / float(len(above_pos)) if above_pos else 0.0
    )
    below_pd = (
        float(sum(1 for s in below_pos if s >= op_thr)) / float(len(below_pos)) if below_pos else 0.0
    )
    clutter_raw_frame_fpr = (
        float(neg_frames_with_fp) / float(num_neg) if num_neg > 0 else 0.0
    )

    return {
        "pd_at_fpi_0p1": float(_pd_at_fpi(0.1)),
        "pd_at_fpi_0p5": float(_pd_at_fpi(0.5)),
        "pd_at_fpi_1p0": float(_pd_at_fpi(1.0)),
        "ap50": float(ap50),
        "centroid_rmse_px": float(centroid_rmse_px),
        "above_horizon_pd": float(above_pd),
        "below_horizon_pd": float(below_pd),
        "clutter_raw_frame_fpr": float(clutter_raw_frame_fpr),
        "op_score_thr": float(op_thr),
        "num_eval_clips": int(len(eval_clips)),
        "num_pos_clips": int(num_pos),
        "num_neg_clips": int(num_neg),
    }


def _unwrap_model(model: nn.Module) -> XvsD2Model:
    return getattr(model, "_orig_mod", model)


def _clip_has_small_target(clip: Mapping[str, Any], *, max_px: float = 14.0) -> bool:
    """Return True if any step in `clip` has a target box with `max(w, h) <= max_px`."""
    steps = clip["boxes_per_step"] if "boxes_per_step" in clip else ()
    for b in steps:
        if b is not None:
            bw = float(b["w"])
            bh = float(b["h"])
            if max(bw, bh) <= float(max_px):
                return True
    return False


def _sample_pos_batch_with_small_oversample(
    mixer: AotG3aBatchMixer,
    small_pos_clips: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    small_prob: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample positive clips with `small_prob` (`0.50` in extended) dedicated to `<= 14 px` distant targets."""
    bs = max(1, int(batch_size))
    if float(small_prob) <= 0.0 or not small_pos_clips:
        return list(mixer.sample_batch(bs))
    n_small = min(bs, max(0, int(round(float(bs) * float(small_prob)))))
    n_reg = bs - n_small
    reg_part = list(mixer.sample_batch(n_reg)) if n_reg > 0 else []
    n_avail = len(small_pos_clips)
    small_part = [dict(small_pos_clips[rng.randrange(n_avail)]) for _ in range(n_small)]
    return reg_part + small_part


def _sample_stage4_batch(
    mixer: AotG3aBatchMixer,
    neg_clips: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    neg_ratio: float = 0.15,
    rng_index: int = 0,
    small_pos_clips: Sequence[Mapping[str, Any]] | None = None,
    small_prob: float = 0.0,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Sample a Stage-4 distillation batch mixing hard-negative clutter clips (`0.15` v5.2 / `0.40` extended).

    Ensures student models suppress ground/sky/bird clutter just like the Stage-2 HNM Teacher:
    - When `batch_size >= 4` and `neg_clips` is non-empty, includes at least 1 negative
      clip (`max(1, round(batch_size * neg_ratio))`, capped at `batch_size - 1`).
    - Safely falls back to pure positive sampling when `neg_clips` is empty or `batch_size < 4`.
    """
    bs = max(1, int(batch_size))
    local_rng = rng if rng is not None else random.Random(int(rng_index) + 17)
    if bs < 4 or not neg_clips or float(neg_ratio) <= 0.0:
        return _sample_pos_batch_with_small_oversample(
            mixer, small_pos_clips or (), batch_size=bs, small_prob=small_prob, rng=local_rng
        )
    n_neg = max(1, int(round(float(bs) * float(neg_ratio))))
    n_neg = min(n_neg, bs - 1)
    n_pos = bs - n_neg
    pos_part = _sample_pos_batch_with_small_oversample(
        mixer, small_pos_clips or (), batch_size=n_pos, small_prob=small_prob, rng=local_rng
    )
    n_avail = len(neg_clips)
    neg_part = [dict(neg_clips[(int(rng_index) + j) % n_avail]) for j in range(n_neg)]
    return pos_part + neg_part


def run_xvs_d2_g4_training(cfg: XvsD2G4TrainConfig) -> dict[str, Any]:
    """Run the full 4-stage `XVS-D2` G4 Blackwell training, evaluation & ONNX export pipeline."""
    assert_dataset_allowed("aot", "train")
    assert_dataset_allowed("openxvs_g3a", "train")
    assert_model_weight_allowed("xvs_d2_scratch")

    is_v53 = bool(cfg.v53_mode)
    if is_v53:
        assert_dataset_allowed("fujii_distant_bird", "train")
        assert_dataset_allowed("det_fly", "train")

    eff_max_neg_clips = max(2000, int(cfg.max_neg_clips)) if is_v53 else int(cfg.max_neg_clips)
    eff_stage4_neg_ratio = max(0.40, float(cfg.stage4_neg_ratio)) if is_v53 else float(cfg.stage4_neg_ratio)
    eff_small_prob = max(0.50, float(cfg.small_target_oversample_prob)) if is_v53 else float(cfg.small_target_oversample_prob)
    eff_use_affine_gmc = bool(cfg.use_affine_gmc or is_v53)
    eff_hnm_refresh = max(1, int(cfg.hnm_refresh_every_epochs))

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    sample_rng = random.Random(int(cfg.seed) + 99)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dir = out_dir / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    env_report = collect_blackwell_env_report()
    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.smoke_local) else "cpu")
    use_bf16 = device.type == "cuda"
    can_compile = bool(cfg.use_torch_compile and device.type == "cuda" and hasattr(torch, "compile"))
    artifact_tag = "SMOKE_LOCAL_CPU" if cfg.smoke_local else "COLAB_G4_GPU_ARTIFACT"

    def _release_compiled_graph(dev: torch.device) -> None:
        if can_compile and hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        gc.collect()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()

    ckpt_mgr = DriveCheckpointManager(local_dir=ckpts_dir, drive_dir=cfg.drive_ckpt_dir)
    aot_ds: AotS3EncounterDataset | None = None

    if cfg.smoke_local:
        crop_hw = (64, 64)
        full_hw = (96, 96)
        clip_len = 3
        ep_s1 = max(1, min(2, int(cfg.epochs_stage1_warmup)))
        ep_s2 = max(1, min(2, int(cfg.epochs_stage2_hnm)))
        ep_s3 = max(1, min(2, int(cfg.epochs_stage3_fullframe)))
        ep_s4 = max(1, min(2, int(cfg.epochs_stage4_distill)))
        n_smoke_neg = 10 if is_v53 else 6
        g3a_train = SyntheticG3aClipDataset(
            num_pos=8,
            num_neg=n_smoke_neg,
            height=crop_hw[0],
            width=crop_hw[1],
            clip_len=clip_len,
            seed=cfg.seed,
            use_affine_gmc=eff_use_affine_gmc,
            v53_mode=is_v53,
        )
        aot_sim = SyntheticG3aClipDataset(
            num_pos=8,
            num_neg=n_smoke_neg,
            height=crop_hw[0],
            width=crop_hw[1],
            clip_len=clip_len,
            seed=cfg.seed + 17,
            use_affine_gmc=eff_use_affine_gmc,
            v53_mode=is_v53,
        )
        aot_pos_clips = [{**c, "source_domain": "aot"} for c in aot_sim.positive_clips]
        aot_neg_clips = [{**c, "source_domain": "aot"} for c in aot_sim.negative_clips]
        if is_v53:
            aot_neg_clips.extend(g3a_train.negative_clips)
        g3a_pos_clips = g3a_train.positive_clips
        eval_clips = aot_pos_clips[:4] + aot_neg_clips[:4]
        dataset_summary: dict[str, Any] = {
            "mode": "smoke_local_v53_contract" if is_v53 else "smoke_local_synthetic_contract",
            "train_pos_aot": len(aot_pos_clips),
            "train_pos_g3a": len(g3a_pos_clips),
            "train_neg_aot": len(aot_neg_clips),
            "eval_clips": len(eval_clips),
        }
        stage1_candidates = [2, 4]
        stage3_candidates = [2]
        stage4_candidates = [2, 4]
    else:
        crop_hw = cfg.crop_hw
        full_hw = cfg.full_frame_hw
        clip_len = int(cfg.clip_len)
        ep_s1 = int(cfg.epochs_stage1_warmup)
        ep_s2 = int(cfg.epochs_stage2_hnm)
        ep_s3 = int(cfg.epochs_stage3_fullframe)
        ep_s4 = int(cfg.epochs_stage4_distill)
        cache_dir = cfg.aot_cache_dir if cfg.aot_cache_dir is not None else (out_dir / "aot_cache")
        aot_ds = AotS3EncounterDataset(
            cache_dir=cache_dir,
            parts=cfg.parts,
            crop_hw=crop_hw,
            clip_len=clip_len,
            max_flights=cfg.max_flights,
            max_neg_clips=eff_max_neg_clips,
            seed=cfg.seed,
            use_affine_gmc=eff_use_affine_gmc,
        )
        dataset_summary = aot_ds.prepare_from_s3()
        aot_pos_clips = aot_ds.train_pos_clips
        aot_neg_clips = list(aot_ds.train_neg_clips)

        fujii_dirs = list(cfg.fujii_bird_dirs)
        detfly_dirs = list(cfg.det_fly_dirs)
        if cfg.download_external_clutter and not cfg.smoke_local:
            dl_rep = download_external_clutter_datasets(
                cache_dir / "external_clutter",
                allow_fallback=bool(cfg.allow_external_clutter_fallback or not cfg.require_external_clutter),
            )
            dataset_summary["external_clutter_download"] = dl_rep
            fujii_p = Path(str(dl_rep["datasets"]["fujii_distant_bird"]["dir"]))
            detfly_p = Path(str(dl_rep["datasets"]["det_fly"]["dir"]))
            if fujii_p not in fujii_dirs:
                fujii_dirs.append(fujii_p)
            if detfly_p not in detfly_dirs:
                detfly_dirs.append(detfly_p)

        fujii_negs = (
            load_external_clutter_negative_clips(
                fujii_dirs,
                dataset_id="fujii_distant_bird",
                crop_hw=crop_hw,
                clip_len=clip_len,
                max_clips=500,
                use_affine_gmc=eff_use_affine_gmc,
                seed=cfg.seed + 101,
                lazy=True,
            )
            if fujii_dirs
            else []
        )
        detfly_negs = (
            load_external_clutter_negative_clips(
                detfly_dirs,
                dataset_id="det_fly",
                crop_hw=crop_hw,
                clip_len=clip_len,
                max_clips=300,
                use_affine_gmc=eff_use_affine_gmc,
                seed=cfg.seed + 202,
                lazy=True,
            )
            if detfly_dirs
            else []
        )
        aot_neg_clips.extend(fujii_negs)
        aot_neg_clips.extend(detfly_negs)
        dataset_summary["fujii_distant_bird_neg_clips"] = len(fujii_negs)
        dataset_summary["det_fly_neg_clips"] = len(detfly_negs)
        print(
            f"[XVS-D2 Clutter] Real external hard-negative clips discovered: "
            f"fujii_distant_bird={len(fujii_negs)}, det_fly={len(detfly_negs)} "
            f"(dirs: fujii={[str(p) for p in fujii_dirs]}, det_fly={[str(p) for p in detfly_dirs]})",
            flush=True,
        )
        if (len(fujii_negs) + len(detfly_negs)) == 0:
            if cfg.require_external_clutter and not cfg.allow_external_clutter_fallback:
                raise RuntimeError(
                    "Fail-closed: require_external_clutter=True but 0 real fujii_distant_bird / det_fly "
                    "clips were discovered. Provide valid directories or set allow_external_clutter_fallback=True."
                )
            if is_v53:
                print(
                    "[XVS-D2 Clutter WARNING] 0 real fujii_distant_bird/det_fly clips loaded — "
                    "explicit fallback active (using AOT + G3a hard negatives only).",
                    flush=True,
                )

        g3a_train = SyntheticG3aClipDataset(
            num_pos=max(12, len(aot_pos_clips) // 2),
            num_neg=max(8, min(256, len(aot_neg_clips) // 4)),
            height=crop_hw[0],
            width=crop_hw[1],
            clip_len=clip_len,
            seed=cfg.seed + 31,
            use_affine_gmc=eff_use_affine_gmc,
            v53_mode=is_v53,
            lazy=True,
        )
        g3a_pos_clips = g3a_train.positive_clips
        if is_v53:
            aot_neg_clips.extend(g3a_train.negative_clips)
        eval_clips = aot_ds.val_clips if aot_ds.val_clips else (aot_pos_clips[:8] + aot_neg_clips[:8])
        stage1_candidates = [4, 8, 16, 24, 32, 48, 64, 96, 128]
        stage3_candidates = [1, 2, 4, 6, 8]
        stage4_candidates = [8, 16, 24, 32, 48, 64, 96, 128]

    small_pos_clips = [c for c in (aot_pos_clips + g3a_pos_clips) if _clip_has_small_target(c, max_px=14.0)]

    def _peak_vram_gib(dev: torch.device) -> float:
        if dev.type == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(dev)) / float(1024**3)
        return 0.0

    def _log_vram_probe(stage_label: str, probe: Mapping[str, Any]) -> None:
        bs = int(probe["selected_batch_size"])
        pk = float(probe["peak_vram_bytes"]) / float(1024**3)
        tot = float(probe["total_vram_bytes"]) / float(1024**3)
        pct = float(probe["utilization_frac"]) * 100.0
        print(
            f"[XVS-D2 {stage_label}] AutoVRAM batch={bs} -> {pk:.1f}/{tot:.1f} GiB ({pct:.1f}%)",
            flush=True,
        )

    mixer = AotG3aBatchMixer(aot_pos_clips, g3a_pos_clips, aot_ratio=0.70, seed=cfg.seed)
    print(
        f"[XVS-D2] Ready for 4-stage training: {len(aot_pos_clips)} AOT pos, {len(g3a_pos_clips)} G3a pos, "
        f"{len(aot_neg_clips)} neg, {len(eval_clips)} eval clips (device={device})",
        flush=True,
    )

    # ---------------- Teacher Resume Check (Skip Stages 1-3 if completed) ----------------
    prev_teacher = ckpt_mgr.load_latest("xvs_d2_teacher")
    prev_s1 = ckpt_mgr.load_latest("stage1_teacher_ep")
    prev_s2 = ckpt_mgr.load_latest("stage2_hnm_ep")
    prev_s3 = ckpt_mgr.load_latest("stage3_fullframe_ep")

    resumed_s1_epoch = int(prev_s1["epoch"]) if (prev_s1 is not None and "epoch" in prev_s1) else 0
    resumed_s2_epoch = int(prev_s2["epoch"]) if (prev_s2 is not None and "epoch" in prev_s2) else 0
    resumed_s3_epoch = int(prev_s3["epoch"]) if (prev_s3 is not None and "epoch" in prev_s3) else 0

    s1_needs_more = prev_s1 is not None and resumed_s1_epoch < ep_s1
    s2_needs_more = prev_s2 is not None and resumed_s2_epoch < ep_s2
    s3_needs_more = prev_s3 is not None and resumed_s3_epoch < ep_s3
    has_completed_teacher_ckpt = (
        prev_teacher is not None and "model_state_dict" in prev_teacher
    ) or (
        prev_s3 is not None and "model_state_dict" in prev_s3 and resumed_s3_epoch >= ep_s3
    )
    teacher_done = bool(
        has_completed_teacher_ckpt
        and not s1_needs_more
        and not s2_needs_more
        and not s3_needs_more
    )

    stage1_losses: list[float] = []
    stage2_losses: list[float] = []
    stage3_losses: list[float] = []
    hard_negs: list[dict[str, Any]] = []
    full_frame_clips_built = 0
    bs_s1 = int(stage1_candidates[0])
    vram_s1: dict[str, Any] = {"skipped_resumed": True}
    bs_s3 = int(stage3_candidates[0])
    vram_s3: dict[str, Any] = {"skipped_resumed": True}
    max_ff_aot = max(1, int(cfg.max_full_frame_clips))
    max_ff_g3a = max(1, max_ff_aot // 2)

    neg_by_key: dict[str, dict[str, Any]] = {str(c["key"]): c for c in aot_neg_clips if "key" in c}

    def _restore_hard_negs_from_ckpt(ckpt_obj: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if ckpt_obj is None or "hard_neg_keys" not in ckpt_obj:
            return []
        restored = [neg_by_key[str(k)] for k in ckpt_obj["hard_neg_keys"] if str(k) in neg_by_key]
        return restored

    if teacher_done:
        raw_teacher = build_xvs_d2("teacher", prior_bias=CROP_PRIOR_BIAS).to(device)
        if prev_teacher is not None and "model_state_dict" in prev_teacher:
            raw_teacher.load_state_dict(prev_teacher["model_state_dict"], strict=True)
            hard_negs = _restore_hard_negs_from_ckpt(prev_teacher) or _restore_hard_negs_from_ckpt(prev_s2)
            if not (ckpts_dir / "xvs_d2_teacher.pt").is_file():
                ckpt_mgr.save(
                    "xvs_d2_teacher",
                    {
                        "stage": "stage3_fullframe",
                        "tier": "teacher",
                        "artifact_tag": artifact_tag,
                        "hard_neg_keys": [str(c["key"]) for c in hard_negs if "key" in c],
                        "model_state_dict": {k: v.detach().cpu() for k, v in raw_teacher.state_dict().items()},
                    },
                )
        else:
            assert prev_s3 is not None
            raw_teacher.load_state_dict(prev_s3["model_state_dict"], strict=True)
            hard_negs = _restore_hard_negs_from_ckpt(prev_s3) or _restore_hard_negs_from_ckpt(prev_s2)
            ckpt_mgr.save(
                "xvs_d2_teacher",
                {
                    "stage": "stage3_fullframe",
                    "tier": "teacher",
                    "artifact_tag": artifact_tag,
                    "hard_neg_keys": [str(c["key"]) for c in hard_negs if "key" in c],
                    "model_state_dict": {k: v.detach().cpu() for k, v in raw_teacher.state_dict().items()},
                },
            )
        if is_v53 and not hard_negs and aot_neg_clips:
            hard_negs = _mine_hard_negatives(
                raw_teacher,
                aot_neg_clips,
                target_hw=crop_hw,
                device=device,
                use_bf16=use_bf16,
                top_k=max(2, len(aot_neg_clips) // 2),
                smoke_local=cfg.smoke_local,
            )
        resumed_s1_epoch = resumed_s1_epoch if prev_s1 is not None else ep_s1
        resumed_s2_epoch = resumed_s2_epoch if prev_s2 is not None else ep_s2
        resumed_s3_epoch = resumed_s3_epoch if prev_s3 is not None else ep_s3
        print(
            f"[XVS-D2] Found completed teacher checkpoint -> skipping Stages 1-3 cleanly! (hard_negs={len(hard_negs)})",
            flush=True,
        )
    else:
        # ---------------- Stage 1: Crop Warmup (`teacher`) ----------------
        teacher = build_xvs_d2("teacher", prior_bias=CROP_PRIOR_BIAS).to(device)
        vram_s1 = _probe_batch_eager(
            teacher,
            aot_pos_clips + g3a_pos_clips,
            target_hw=crop_hw,
            candidates=stage1_candidates,
            device=device,
            use_bf16=use_bf16,
            target_frac=cfg.vram_target_frac,
            headroom_gib=cfg.vram_headroom_gib,
        )
        bs_s1 = int(vram_s1["selected_batch_size"])
        _log_vram_probe("Stage1", vram_s1)
        teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        if prev_s1 is not None and "model_state_dict" in prev_s1:
            teacher.load_state_dict(prev_s1["model_state_dict"], strict=True)
            if "optimizer_state_dict" in prev_s1:
                teacher_opt.load_state_dict(prev_s1["optimizer_state_dict"])

        teacher_exec: nn.Module = torch.compile(teacher) if can_compile else teacher
        steps_per_ep = 2 if cfg.smoke_local else max(4, (len(aot_pos_clips) + len(g3a_pos_clips)) // max(1, bs_s1))

        for epoch in range(resumed_s1_epoch, ep_s1):
            teacher_exec.train()
            ep_loss = 0.0
            ep_clips_s1: list[dict[str, Any]] = []
            for _ in range(steps_per_ep):
                ep_clips_s1.extend(
                    _sample_pos_batch_with_small_oversample(
                        mixer,
                        small_pos_clips,
                        batch_size=bs_s1,
                        small_prob=eff_small_prob,
                        rng=sample_rng,
                    )
                )
            s1_loader = _make_epoch_dataloader(
                ep_clips_s1, batch_size=bs_s1, target_hw=crop_hw, device=device, smoke_local=cfg.smoke_local
            )
            for x_seq_b, metas_b in s1_loader:
                teacher_opt.zero_grad(set_to_none=True)
                loss = _train_one_clip_batch_tensors(
                    teacher_exec,
                    x_seq_b,
                    metas_b,
                    device=device,
                    use_bf16=use_bf16,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
                teacher_opt.step()
                ep_loss += float(loss.detach().cpu().item())
            mean_loss = ep_loss / float(steps_per_ep)
            stage1_losses.append(mean_loss)
            print(
                f"[XVS-D2 Stage1 ep {epoch + 1:02d}/{ep_s1:02d}] loss={mean_loss:.4f} peak_vram={_peak_vram_gib(device):.2f} GiB",
                flush=True,
            )
            raw_t = _unwrap_model(teacher_exec)
            ckpt_mgr.save(
                f"stage1_teacher_ep{epoch + 1:03d}",
                {
                    "stage": "stage1_warmup",
                    "tier": "teacher",
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "artifact_tag": artifact_tag,
                    "model_state_dict": {k: v.detach().cpu() for k, v in raw_t.state_dict().items()},
                    "optimizer_state_dict": teacher_opt.state_dict(),
                },
            )

        # ---------------- Stage 2: Hard Negative Mining (`teacher`, resumable) ----------------
        raw_teacher = _unwrap_model(teacher_exec)
        if prev_s2 is not None and "model_state_dict" in prev_s2:
            raw_teacher.load_state_dict(prev_s2["model_state_dict"], strict=True)
            if "optimizer_state_dict" in prev_s2:
                teacher_opt.load_state_dict(prev_s2["optimizer_state_dict"])
            hard_negs = _restore_hard_negs_from_ckpt(prev_s2)

        if resumed_s2_epoch < ep_s2 and not hard_negs:
            hard_negs = _mine_hard_negatives(
                raw_teacher,
                aot_neg_clips,
                target_hw=crop_hw,
                device=device,
                use_bf16=use_bf16,
                top_k=max(2, len(aot_neg_clips) // 2),
                smoke_local=cfg.smoke_local,
            )
        for epoch in range(resumed_s2_epoch, ep_s2):
            if is_v53 and epoch > resumed_s2_epoch and ((epoch - resumed_s2_epoch) % eff_hnm_refresh == 0):
                hard_negs = _mine_hard_negatives(
                    _unwrap_model(teacher_exec),
                    aot_neg_clips,
                    target_hw=crop_hw,
                    device=device,
                    use_bf16=use_bf16,
                    top_k=max(2, len(aot_neg_clips) // 2),
                    smoke_local=cfg.smoke_local,
                )
            teacher_exec.train()
            ep_loss = 0.0
            ep_clips_s2: list[dict[str, Any]] = []
            for step_s2 in range(steps_per_ep):
                if is_v53 and hard_negs:
                    n_neg_s2 = min(max(1, int(round(float(bs_s1) * eff_stage4_neg_ratio))), max(1, bs_s1 - 1))
                    n_pos_s2 = max(1, bs_s1 - n_neg_s2)
                    pos_part = _sample_pos_batch_with_small_oversample(
                        mixer,
                        small_pos_clips,
                        batch_size=n_pos_s2,
                        small_prob=eff_small_prob,
                        rng=sample_rng,
                    )
                    neg_pick = [
                        hard_negs[(epoch * max(1, steps_per_ep) + step_s2 + j) % len(hard_negs)]
                        for j in range(n_neg_s2)
                    ]
                    ep_clips_s2.extend(pos_part + neg_pick)
                else:
                    pos_part = mixer.sample_batch(max(1, bs_s1 - 1))
                    neg_pick = [hard_negs[epoch % len(hard_negs)]] if hard_negs else []
                    ep_clips_s2.extend(pos_part + neg_pick)
            s2_loader = _make_epoch_dataloader(
                ep_clips_s2, batch_size=bs_s1, target_hw=crop_hw, device=device, smoke_local=cfg.smoke_local
            )
            for x_seq_b, metas_b in s2_loader:
                teacher_opt.zero_grad(set_to_none=True)
                loss = _train_one_clip_batch_tensors(
                    teacher_exec,
                    x_seq_b,
                    metas_b,
                    device=device,
                    use_bf16=use_bf16,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
                teacher_opt.step()
                ep_loss += float(loss.detach().cpu().item())
            mean_loss = ep_loss / float(steps_per_ep)
            stage2_losses.append(mean_loss)
            print(
                f"[XVS-D2 Stage2 HNM ep {epoch + 1:02d}/{ep_s2:02d}] loss={mean_loss:.4f} peak_vram={_peak_vram_gib(device):.2f} GiB",
                flush=True,
            )
            ckpt_mgr.save(
                f"stage2_hnm_ep{epoch + 1:03d}",
                {
                    "stage": "stage2_hnm",
                    "tier": "teacher",
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "artifact_tag": artifact_tag,
                    "hard_neg_keys": [str(c["key"]) for c in hard_negs if "key" in c],
                    "model_state_dict": {k: v.detach().cpu() for k, v in _unwrap_model(teacher_exec).state_dict().items()},
                    "optimizer_state_dict": teacher_opt.state_dict(),
                },
            )

        # Explicitly release Stage 1/2 compiled graph & optimizer state before Stage 3 (`D16-G4-M5`)
        raw_teacher = _unwrap_model(teacher_exec)
        del teacher_exec
        del teacher_opt
        _release_compiled_graph(device)

        # ---------------- Stage 3: Native Full-Frame Fine-Tune (`teacher`, resumable, GMC + Negs!) ----------------
        if prev_s3 is not None and "model_state_dict" in prev_s3:
            raw_teacher.load_state_dict(prev_s3["model_state_dict"], strict=True)
            if not hard_negs:
                hard_negs = _restore_hard_negs_from_ckpt(prev_s3)
        else:
            with torch.no_grad():
                raw_teacher.heads.hm_out.bias.mul_(0.5).add_(0.5 * FULL_FRAME_PRIOR_BIAS)

        if resumed_s3_epoch < ep_s3:
            if cfg.smoke_local:
                n_smoke_aot = min(4, max_ff_aot)
                n_smoke_g3a = min(4, max_ff_g3a)
                g3a_full = SyntheticG3aClipDataset(
                    num_pos=n_smoke_g3a,
                    num_neg=2,
                    height=full_hw[0],
                    width=full_hw[1],
                    clip_len=clip_len,
                    seed=cfg.seed + 41,
                    use_affine_gmc=eff_use_affine_gmc,
                    v53_mode=is_v53,
                )
                aot_full_sim = SyntheticG3aClipDataset(
                    num_pos=n_smoke_aot,
                    num_neg=2,
                    height=full_hw[0],
                    width=full_hw[1],
                    clip_len=clip_len,
                    seed=cfg.seed + 53,
                    use_affine_gmc=eff_use_affine_gmc,
                    v53_mode=is_v53,
                )
                full_aot_clips = [{**c, "source_domain": "aot", "native_hw": list(full_hw)} for c in aot_full_sim.positive_clips]
                full_g3a_clips = [{**c, "native_hw": list(full_hw)} for c in g3a_full.positive_clips]
                full_neg_clips = (
                    (
                        [{**c, "source_domain": "aot", "native_hw": list(full_hw)} for c in aot_full_sim.negative_clips]
                        + [{**c, "native_hw": list(full_hw)} for c in g3a_full.negative_clips]
                    )
                    if is_v53
                    else []
                )
            else:
                assert aot_ds is not None
                full_aot_clips = aot_ds.build_native_full_frame_clips(full_hw, max_clips=max_ff_aot, lazy=True)
                n_g3a_ff = min(max_ff_g3a, max(1, len(full_aot_clips) // 2))
                g3a_full = SyntheticG3aClipDataset(
                    num_pos=n_g3a_ff,
                    num_neg=max(2, n_g3a_ff),
                    height=full_hw[0],
                    width=full_hw[1],
                    clip_len=clip_len,
                    seed=cfg.seed + 41,
                    use_affine_gmc=eff_use_affine_gmc,
                    v53_mode=is_v53,
                    lazy=True,
                )
                full_g3a_clips = [{**c, "native_hw": list(full_hw)} for c in g3a_full.positive_clips]
                full_neg_clips = (
                    (
                        aot_ds.build_native_full_frame_neg_clips(full_hw, max_clips=max(4, max_ff_aot // 2))
                        + [{**c, "native_hw": list(full_hw)} for c in g3a_full.negative_clips]
                    )
                    if is_v53
                    else []
                )

            full_frame_clips_built = len(full_aot_clips) + len(full_g3a_clips) + len(full_neg_clips)
            full_mixer = AotG3aBatchMixer(full_aot_clips, full_g3a_clips, aot_ratio=0.70, seed=cfg.seed + 7)

            vram_s3 = _probe_batch_eager(
                raw_teacher,
                full_aot_clips + full_g3a_clips,
                target_hw=full_hw,
                candidates=stage3_candidates,
                device=device,
                use_bf16=use_bf16,
                target_frac=cfg.vram_target_frac,
                headroom_gib=cfg.vram_headroom_gib,
            )
            bs_s3 = int(vram_s3["selected_batch_size"])
            _log_vram_probe("Stage3 FullFrame", vram_s3)
            teacher_ff_opt = torch.optim.AdamW(raw_teacher.parameters(), lr=cfg.lr * 0.25, weight_decay=cfg.weight_decay)
            if prev_s3 is not None and "optimizer_state_dict" in prev_s3:
                teacher_ff_opt.load_state_dict(prev_s3["optimizer_state_dict"])
            teacher_ff_exec: nn.Module = torch.compile(raw_teacher) if can_compile else raw_teacher

            for epoch in range(resumed_s3_epoch, ep_s3):
                teacher_ff_exec.train()
                ep_loss = 0.0
                ep_clips_s3: list[dict[str, Any]] = []
                for step_s3 in range(steps_per_ep):
                    # Mix ~25-30% full-frame negatives in Stage 3 (extended) to prevent HNM unlearning!
                    if is_v53 and full_neg_clips and bs_s3 >= 2:
                        n_neg_s3 = max(1, int(round(float(bs_s3) * 0.30)))
                        n_neg_s3 = min(n_neg_s3, bs_s3 - 1)
                        pos_s3 = full_mixer.sample_batch(bs_s3 - n_neg_s3)
                        neg_s3 = [
                            dict(full_neg_clips[(epoch * max(1, steps_per_ep) + step_s3 + j) % len(full_neg_clips)])
                            for j in range(n_neg_s3)
                        ]
                        ep_clips_s3.extend(pos_s3 + neg_s3)
                    else:
                        ep_clips_s3.extend(full_mixer.sample_batch(bs_s3))
                s3_loader = _make_epoch_dataloader(
                    ep_clips_s3, batch_size=bs_s3, target_hw=full_hw, device=device, smoke_local=cfg.smoke_local
                )
                for x_seq_b, metas_b in s3_loader:
                    teacher_ff_opt.zero_grad(set_to_none=True)
                    loss = _train_one_clip_batch_tensors(
                        teacher_ff_exec,
                        x_seq_b,
                        metas_b,
                        device=device,
                        use_bf16=use_bf16,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(raw_teacher.parameters(), max_norm=5.0)
                    teacher_ff_opt.step()
                    ep_loss += float(loss.detach().cpu().item())
                mean_loss = ep_loss / float(steps_per_ep)
                stage3_losses.append(mean_loss)
                print(
                    f"[XVS-D2 Stage3 FullFrame ep {epoch + 1:02d}/{ep_s3:02d}] loss={mean_loss:.4f} peak_vram={_peak_vram_gib(device):.2f} GiB",
                    flush=True,
                )
                ckpt_mgr.save(
                    f"stage3_fullframe_ep{epoch + 1:03d}",
                    {
                        "stage": "stage3_fullframe",
                        "tier": "teacher",
                        "epoch": epoch + 1,
                        "loss": mean_loss,
                        "native_resolution_hw": list(full_hw),
                        "artifact_tag": artifact_tag,
                        "hard_neg_keys": [str(c["key"]) for c in hard_negs if "key" in c],
                        "model_state_dict": {k: v.detach().cpu() for k, v in _unwrap_model(teacher_ff_exec).state_dict().items()},
                        "optimizer_state_dict": teacher_ff_opt.state_dict(),
                    },
                )

            raw_teacher = _unwrap_model(teacher_ff_exec)
            del teacher_ff_exec
            del teacher_ff_opt
            del full_aot_clips
            del full_g3a_clips
            del full_neg_clips
            del full_mixer
            _release_compiled_graph(device)

        ckpt_mgr.save(
            "xvs_d2_teacher",
            {
                "stage": "stage3_fullframe",
                "tier": "teacher",
                "artifact_tag": artifact_tag,
                "hard_neg_keys": [str(c["key"]) for c in hard_negs if "key" in c],
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_teacher.state_dict().items()},
            },
        )

    # ---------------- Stage 4: CSCWD + CrossKD Distillation (per-tier/per-epoch resumable) ----------------
    raw_teacher.eval()
    for p in raw_teacher.parameters():
        p.requires_grad_(False)

    tier_summary: dict[str, Any] = {"teacher_params": count_parameters(raw_teacher)}
    stage4_summary: dict[str, Any] = {}
    web_model_final: XvsD2Model | None = None
    stage4_headroom_gib = max(float(cfg.vram_headroom_gib), 12.0)

    for student_tier in cfg.student_tiers:
        student = build_xvs_d2(student_tier, prior_bias=CROP_PRIOR_BIAS).to(device)
        tier_summary[f"{student_tier}_params"] = count_parameters(student)

        t_spec = raw_teacher.spec
        s_spec = student.spec
        cscwd_adapters = nn.ModuleDict(
            {
                "p2": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
                "p3": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
                "p4": nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False),
            }
        ).to(device)
        fp2_adapter = nn.Conv2d(s_spec.c_fpn, t_spec.c_fpn, kernel_size=1, bias=False).to(device)

        resumed_s4_epoch = 0
        prev_s4 = ckpt_mgr.load_latest(f"stage4_distill_{student_tier}_ep")
        if prev_s4 is not None and "epoch" in prev_s4:
            resumed_s4_epoch = int(prev_s4["epoch"])

        if resumed_s4_epoch < ep_s4:
            vram_s4 = _probe_batch_eager(
                student,
                aot_pos_clips + g3a_pos_clips,
                target_hw=crop_hw,
                candidates=stage4_candidates,
                device=device,
                use_bf16=use_bf16,
                target_frac=cfg.vram_target_frac,
                headroom_gib=stage4_headroom_gib,
                teacher_model=raw_teacher,
                cscwd_adapters=cscwd_adapters,
                fp2_adapter=fp2_adapter,
                lambda_kd=0.5,
            )
            bs_s4 = int(vram_s4["selected_batch_size"])
            _log_vram_probe(f"Stage4 {student_tier}", vram_s4)
        else:
            vram_s4 = {"skipped_resumed": True}
            bs_s4 = int(stage4_candidates[0])

        params = list(student.parameters()) + list(cscwd_adapters.parameters()) + list(fp2_adapter.parameters())
        s_opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if prev_s4 is not None and "model_state_dict" in prev_s4:
            student.load_state_dict(prev_s4["model_state_dict"], strict=True)
            if "cscwd_adapters_state_dict" in prev_s4:
                cscwd_adapters.load_state_dict(prev_s4["cscwd_adapters_state_dict"], strict=True)
            if "fp2_adapter_state_dict" in prev_s4:
                fp2_adapter.load_state_dict(prev_s4["fp2_adapter_state_dict"], strict=True)
            if "optimizer_state_dict" in prev_s4:
                s_opt.load_state_dict(prev_s4["optimizer_state_dict"])

        steps_s4 = 2 if cfg.smoke_local else max(4, (len(aot_pos_clips) + len(g3a_pos_clips)) // max(1, bs_s4))
        student_exec: nn.Module = (
            torch.compile(student) if (can_compile and resumed_s4_epoch < ep_s4) else student
        )
        s_losses: list[float] = []

        for epoch in range(resumed_s4_epoch, ep_s4):
            student_exec.train()
            cscwd_adapters.train()
            fp2_adapter.train()
            ep_loss = 0.0
            ep_clips_s4: list[dict[str, Any]] = []
            for step_i in range(steps_s4):
                ep_clips_s4.extend(
                    _sample_stage4_batch(
                        mixer,
                        hard_negs if (is_v53 and hard_negs) else aot_neg_clips,
                        batch_size=bs_s4,
                        neg_ratio=eff_stage4_neg_ratio,
                        rng_index=epoch * max(1, steps_s4) + step_i,
                        small_pos_clips=small_pos_clips if is_v53 else None,
                        small_prob=eff_small_prob,
                        rng=sample_rng,
                    )
                )
            s4_loader = _make_epoch_dataloader(
                ep_clips_s4, batch_size=bs_s4, target_hw=crop_hw, device=device, smoke_local=cfg.smoke_local
            )
            for x_seq_b, metas_b in s4_loader:
                s_opt.zero_grad(set_to_none=True)
                loss = _train_one_clip_batch_tensors(
                    student_exec,
                    x_seq_b,
                    metas_b,
                    device=device,
                    use_bf16=use_bf16,
                    teacher_model=raw_teacher,
                    cscwd_adapters=cscwd_adapters,
                    fp2_adapter=fp2_adapter,
                    lambda_kd=1.0,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                s_opt.step()
                ep_loss += float(loss.detach().cpu().item())
            mean_s_loss = ep_loss / float(steps_s4)
            s_losses.append(mean_s_loss)
            print(
                f"[XVS-D2 Stage4 {student_tier} ep {epoch + 1:02d}/{ep_s4:02d}] loss={mean_s_loss:.4f} peak_vram={_peak_vram_gib(device):.2f} GiB",
                flush=True,
            )
            ckpt_mgr.save(
                f"stage4_distill_{student_tier}_ep{epoch + 1:03d}",
                {
                    "stage": "stage4_distill",
                    "tier": student_tier,
                    "epoch": epoch + 1,
                    "loss": mean_s_loss,
                    "artifact_tag": artifact_tag,
                    "model_state_dict": {k: v.detach().cpu() for k, v in _unwrap_model(student_exec).state_dict().items()},
                    "cscwd_adapters_state_dict": {k: v.detach().cpu() for k, v in cscwd_adapters.state_dict().items()},
                    "fp2_adapter_state_dict": {k: v.detach().cpu() for k, v in fp2_adapter.state_dict().items()},
                    "optimizer_state_dict": s_opt.state_dict(),
                },
            )

        raw_student = _unwrap_model(student_exec)
        ckpt_mgr.save(
            f"xvs_d2_{student_tier}",
            {
                "stage": "stage4_distill",
                "tier": student_tier,
                "artifact_tag": artifact_tag,
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_student.state_dict().items()},
            },
        )
        stage4_summary[student_tier] = {
            "resumed_from_epoch": resumed_s4_epoch,
            "selected_batch_size": bs_s4,
            "epochs_executed_this_run": len(s_losses),
            "final_loss": float(s_losses[-1]) if s_losses else (float(prev_s4["loss"]) if prev_s4 is not None and "loss" in prev_s4 else 0.0),
            "vram_probe": vram_s4,
        }
        if student_tier == "web":
            web_model_final = raw_student
        else:
            del raw_student

        del student_exec
        del student
        del s_opt
        del cscwd_adapters
        del fp2_adapter
        del params
        _release_compiled_graph(device)

    # ---------------- Evaluation & Unwrapped ONNX Export (`BC-G4-M3`) ----------------
    eval_teacher = evaluate_xvs_d2(raw_teacher, eval_clips, target_hw=crop_hw, device=device)
    if web_model_final is None:
        web_model_final = build_xvs_d2("web").to(device)
    eval_web = evaluate_xvs_d2(web_model_final, eval_clips, target_hw=crop_hw, device=device)

    raw_web_cpu = _unwrap_model(web_model_final).eval().cpu()
    eff_onnx_hw = (
        (int(cfg.onnx_export_hw[0]), int(cfg.onnx_export_hw[1]))
        if cfg.onnx_export_hw is not None
        else ((640, 640) if is_v53 else (256, 256))
    )
    onnx_report = export_xvs_d2_web_onnx(
        out_dir / "onnx", model=raw_web_cpu, height=eff_onnx_hw[0], width=eff_onnx_hw[1]
    )

    eval_payload = {"teacher": eval_teacher, "web": eval_web}
    (out_dir / "tier_summary.json").write_text(json.dumps(tier_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "onnx_export_report.json").write_text(json.dumps(onnx_report, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "schema": "xvs.v2_m2.xvs_d2_g4_train_summary/2" if is_v53 else "xvs.v2_m2.xvs_d2_g4_train_summary/1",
        "artifact_tag": artifact_tag,
        "smoke_local": bool(cfg.smoke_local),
        "v53_mode": bool(is_v53),
        "v53_config": {
            "max_neg_clips": int(eff_max_neg_clips),
            "stage4_neg_ratio": float(eff_stage4_neg_ratio),
            "small_target_oversample_prob": float(eff_small_prob),
            "use_affine_gmc": bool(eff_use_affine_gmc),
            "hnm_refresh_every_epochs": int(eff_hnm_refresh),
            "small_pos_clips_count": int(len(small_pos_clips)),
            "onnx_export_hw": [int(eff_onnx_hw[0]), int(eff_onnx_hw[1])],
        },
        "env_report": env_report,
        "dataset_summary": dataset_summary,
        "stage1_warmup": {
            "resumed_from_epoch": resumed_s1_epoch,
            "selected_batch_size": bs_s1,
            "losses": stage1_losses,
            "vram_probe": vram_s1,
        },
        "stage2_hnm": {
            "resumed_from_epoch": resumed_s2_epoch,
            "mined_hard_negatives": len(hard_negs),
            "losses": stage2_losses,
        },
        "stage3_fullframe": {
            "resumed_from_epoch": resumed_s3_epoch,
            "native_resolution_hw": list(full_hw),
            "upsampled_from_crop": False,
            "max_full_frame_clips_cap": max_ff_aot,
            "full_frame_clips_built": full_frame_clips_built,
            "selected_batch_size": bs_s3,
            "losses": stage3_losses,
            "vram_probe": vram_s3,
        },
        "stage4_distill": stage4_summary,
        "tier_summary": tier_summary,
        "eval_metrics": eval_payload,
        "onnx_export_report": onnx_report,
        "pass": bool(onnx_report["all_checks_passed"]) and all(
            math.isfinite(float(v))
            for v in (
                eval_teacher["pd_at_fpi_0p1"],
                eval_teacher["ap50"],
                eval_teacher["centroid_rmse_px"],
                eval_web["pd_at_fpi_0p1"],
                eval_web["ap50"],
            )
        ),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenXVS v1.1.1 XVS-D2 4-Stage Colab G4 Trainer")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--drive-ckpt-dir", type=Path, default=None)
    parser.add_argument("--smoke-local", action="store_true")
    parser.add_argument("--v53-mode", action="store_true")
    parser.add_argument("--parts", nargs="+", default=["part1", "part2", "part3"])
    parser.add_argument("--max-flights", type=int, default=36)
    parser.add_argument("--max-neg-clips", type=int, default=32)
    parser.add_argument("--stage4-neg-ratio", type=float, default=0.15)
    parser.add_argument("--small-target-oversample-prob", type=float, default=0.0)
    parser.add_argument("--use-affine-gmc", action="store_true")
    parser.add_argument("--fujii-bird-dir", type=Path, action="append", default=[])
    parser.add_argument("--det-fly-dir", type=Path, action="append", default=[])
    parser.add_argument("--epochs-stage1", type=int, default=20)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--epochs-stage3", type=int, default=5)
    parser.add_argument("--epochs-stage4", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args(argv)

    cfg = XvsD2G4TrainConfig(
        out_dir=args.out_dir,
        drive_ckpt_dir=args.drive_ckpt_dir,
        smoke_local=bool(args.smoke_local),
        parts=tuple(args.parts),
        max_flights=int(args.max_flights),
        epochs_stage1_warmup=int(args.epochs_stage1),
        epochs_stage2_hnm=int(args.epochs_stage2),
        epochs_stage3_fullframe=int(args.epochs_stage3),
        epochs_stage4_distill=int(args.epochs_stage4),
        seed=int(args.seed),
        v53_mode=bool(args.v53_mode),
        max_neg_clips=int(args.max_neg_clips),
        stage4_neg_ratio=float(args.stage4_neg_ratio),
        small_target_oversample_prob=float(args.small_target_oversample_prob),
        use_affine_gmc=bool(args.use_affine_gmc),
        fujii_bird_dirs=tuple(args.fujii_bird_dir),
        det_fly_dirs=tuple(args.det_fly_dir),
    )
    summary = run_xvs_d2_g4_training(cfg)
    print(json.dumps({"pass": summary["pass"], "eval_web": summary["eval_metrics"]["web"]}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
