"""Production Colab G4 (96 GB VRAM Blackwell) trainer for XVS-ThermalNet & real-feature XVS-VRAIM (SP-2B §3.2 / DR-04).

Features:
  - Strict license guard (`irstd1k` MIT, `hit_uav` CC-BY-4.0, `xvs_thermal_scratch`).
  - Optional Colab downloaders for `HIT-UAV` and `IRSTD-1k` (pure stdlib `urllib.request`/`zipfile`,
    strictly gated by `download_datasets=True` & `assert_dataset_allowed`, zero network in `--smoke-local`).
  - Paired physics EO + 2-channel LWIR scene generator via `src/xvs/sensor/lwir.py:simulate_lwir_frame`
    under exact `40/20/20/10/10` `FaultCurriculumSampler`.
  - Eager `AutoVRAMBatchTuner` before `torch.compile` (`BC-G4-m4`), `bf16` autocast without `GradScaler`
    + FP32 `compute_thermal_multitask_loss`, atomic `DriveCheckpointManager` save/resume.
  - 4-mode fault verification (nominal, EO-blind, LWIR-blind, dual-blind `vacuity == 1.0`).
  - Real-feature `XVS-VRAIM` offline audit (`BC-G4-m2`): extracts `[B, 64]` penultimate vectors via
    global average pooling of `out["f_fused"]` (zero edits to `model.py`) across ID vs fog/crossover
    OOD shifts -> `PenultimateMahalanobisOOD` + `TopKSparseAutoencoder` + `AdaptiveStressTester` ->
    emits `easa_rlm_07_11_evidence.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any
import urllib.request
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xvs.sensor.lwir import LwirSensorConfig, ThermalFaultConfig, simulate_lwir_frame
from xvs.v2.colab_g4_utils import (
    AutoVRAMBatchTuner,
    DriveCheckpointManager,
    collect_blackwell_env_report,
)
from xvs.v2.license_guard import assert_dataset_allowed, assert_model_weight_allowed
from xvs.v2.xvs_thermal.curriculum import (
    CurriculumSample,
    FaultCurriculumSampler,
    summarize_curriculum_distribution,
)
from xvs.v2.xvs_thermal.datasets import HitUavDataset, Irstd1kDataset, ThermalSample
from xvs.v2.xvs_thermal.model import (
    ThermalSimplexMode,
    XvsThermalNet,
    compute_thermal_multitask_loss,
    select_thermal_simplex_mode,
)
from xvs.v2.xvs_vraim import (
    AdaptiveStressTester,
    PenultimateMahalanobisOOD,
    TopKSparseAutoencoder,
    audit_content_vs_style_reliance,
    build_easa_rlm_07_11_report,
    compute_auroc,
)

IRSTD1K_ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download?id=1JoGDGF96v4CncKZprDnoIor0k1opaLZa&export=download&confirm=t"
)
IRSTD1K_MIRROR_URLS: tuple[str, ...] = (
    IRSTD1K_ARCHIVE_URL,
    "https://github.com/RuiZhang97/ISNet/archive/refs/heads/master.zip",
)
HIT_UAV_ARCHIVE_URL = (
    "https://huggingface.co/datasets/huseyincavus/nighteye-hit-uav/resolve/main/"
    "HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip"
)
HIT_UAV_MIRROR_URLS: tuple[str, ...] = (
    HIT_UAV_ARCHIVE_URL,
    "https://zenodo.org/records/7633134/files/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip?download=1",
    "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/archive/refs/heads/main.zip",
)


@dataclass(frozen=True)
class XvsThermalG4Config:
    """Configuration for production Colab G4 & offline `--smoke-local` XVS-ThermalNet + VRAIM."""

    out_dir: Path
    drive_ckpt_dir: Path | None = None
    data_cache_dir: Path | None = None
    smoke_local: bool = False
    download_datasets: bool = False
    require_real_thermal: bool = False
    min_real_thermal_samples: int = 100
    export_onnx: bool = True
    epochs: int = 12
    steps_per_epoch: int = 160
    num_train_samples: int = 512
    eo_hw: tuple[int, int] = (256, 320)
    lwir_hw: tuple[int, int] = (128, 160)
    batch_candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    lr: float = 4.0e-4
    weight_decay: float = 1.0e-4
    lambda_aux: float = 0.4
    lambda_ev: float = 0.2
    run_vraim: bool = True
    use_compile: bool = False
    vram_target_frac: float = 0.86
    vram_headroom_gib: float = 12.0
    seed: int = 20261121
    v53_mode: bool = False
    export_fp16_onnx: bool = False
    parallax_aug_prob: float = 0.35
    parallax_max_shift_lwir_px: float = 12.0
    parallax_max_roll_deg: float = 5.0


def enforce_xvs_thermal_licenses() -> None:
    """Fail-closed license verification for XVS-ThermalNet training datasets and weights."""
    assert_dataset_allowed("irstd1k", "train")
    assert_dataset_allowed("hit_uav", "train")
    assert_model_weight_allowed("xvs_thermal_scratch", allow_non_osi_attribution=False)


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract zip archive with strict Zip-Slip path traversal protection (`D16-G4-m4`)."""
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


def _extract_nested_zips(root_dir: Path, *, max_depth: int = 3) -> int:
    """Recursively unpack nested `.zip` archives inside `root_dir` via `_safe_extract_zip`."""
    if not root_dir.is_dir():
        return 0
    extracted: set[Path] = set()
    total_unpacked = 0
    for _depth in range(max(1, int(max_depth))):
        zips = sorted(p for p in root_dir.rglob("*.zip") if p.is_file() and p.resolve() not in extracted)
        if not zips:
            break
        for zp in zips:
            rp = zp.resolve()
            extracted.add(rp)
            if not zipfile.is_zipfile(zp):
                continue
            target_dir = zp.parent / f"{zp.stem}_unpacked"
            marker = target_dir / ".openxvs_unpacked"
            if not marker.is_file():
                _safe_extract_zip(zp, target_dir)
                marker.write_text("ok\n", encoding="utf-8")
                total_unpacked += 1
    return total_unpacked


def download_hit_uav_and_irstd1k(cache_dir: Path, *, timeout_s: float = 120.0) -> dict[str, str]:
    """Optional Colab downloader for `HIT-UAV` (CC0/CC-BY-4.0) & `IRSTD-1k` (MIT) with mirror fallback."""
    enforce_xvs_thermal_licenses()
    cache_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    targets = (
        ("irstd1k", IRSTD1K_MIRROR_URLS),
        ("hit_uav", HIT_UAV_MIRROR_URLS),
    )
    for name, mirror_urls in targets:
        dest_dir = cache_dir / name
        if dest_dir.is_dir() and any(dest_dir.iterdir()):
            _extract_nested_zips(dest_dir)
            if discover_real_thermal_samples(dest_dir, (64, 64), max_samples=1):
                status[name] = "cached"
                continue
        zip_path = cache_dir / f"{name}.zip"
        tmp_path = cache_dir / f"{name}.zip.tmp"
        last_err = "no_mirror"
        downloaded_ok = False
        for url in mirror_urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "OpenXVS-research/0.1"})
                with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                    with tmp_path.open("wb") as out_f:
                        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
                if not zipfile.is_zipfile(tmp_path):
                    tmp_path.unlink(missing_ok=True)
                    last_err = "non_zip_payload"
                    continue
                tmp_path.replace(zip_path)
                _safe_extract_zip(zip_path, dest_dir)
                _extract_nested_zips(dest_dir)
                if discover_real_thermal_samples(dest_dir, (64, 64), max_samples=1):
                    status[name] = "downloaded"
                    downloaded_ok = True
                    break
                last_err = "no_thermal_images_in_archive"
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                last_err = type(exc).__name__
                # Do not stall across multiple mirrors when the socket itself timed out
                if isinstance(exc, TimeoutError):
                    break
        if not downloaded_ok:
            status[name] = f"fallback_physics_only ({last_err})"
    return status


def discover_real_thermal_samples(
    cache_dir: Path | None,
    lwir_hw: tuple[int, int],
    *,
    max_samples: int = 256,
) -> list[ThermalSample]:
    """Discover & load real `IRSTD-1k` (MIT) and `HIT-UAV` (CC0/CC-BY-4.0) thermal frames from `cache_dir`."""
    if cache_dir is None or not Path(cache_dir).is_dir():
        return []
    enforce_xvs_thermal_licenses()
    root = Path(cache_dir)
    _extract_nested_zips(root)

    # 1. Discover IRSTD-1k directories (supports nested archive & alternative folder layouts)
    irstd_roots: list[Path] = []
    irstd_candidates: list[Path] = [root, root / "irstd1k"]
    for marker_name in ("IRSTD1k_Img", "IRSTD1k_Label", "masks"):
        for cand in sorted(root.rglob(marker_name)):
            if cand.is_dir():
                irstd_candidates.append(cand.parent)
    seen_irstd: set[Path] = set()
    for p in irstd_candidates:
        if not p.is_dir():
            continue
        rp = p.resolve()
        if rp in seen_irstd:
            continue
        has_irstd = (
            (p / "IRSTD1k_Img").is_dir()
            or ((p / "images").is_dir() and (p / "masks").is_dir())
            or ((p / "image").is_dir() and (p / "mask").is_dir())
        )
        if has_irstd:
            seen_irstd.add(rp)
            irstd_roots.append(p)

    # 2. Discover HIT-UAV directories (YOLO `images/`+`labels/`, `dataset/images/`, `normal_json/`, `yolo_labels/`)
    hituav_roots: list[Path] = []
    hituav_candidates: list[Path] = [root, root / "hit_uav"]
    for marker_name in ("labels", "yolo_labels", "normal_json", "normal_images", "dataset"):
        for cand in sorted(root.rglob(marker_name)):
            if cand.is_dir():
                hituav_candidates.append(cand.parent)
                if marker_name == "dataset":
                    hituav_candidates.append(cand)
    seen_hituav: set[Path] = set()
    for p in hituav_candidates:
        if not p.is_dir():
            continue
        rp = p.resolve()
        if rp in seen_hituav or rp in seen_irstd:
            continue
        has_img = any((p / d).is_dir() for d in ("images", "normal_images", "dataset", "train"))
        has_ann = any((p / d).is_dir() for d in ("labels", "yolo_labels", "normal_json", "annotations", "train"))
        if has_img and has_ann:
            seen_hituav.add(rp)
            hituav_roots.append(p)

    irstd_samples: list[ThermalSample] = []
    seen_img_files: set[Path] = set()
    half_cap = max(1, int(max_samples) // 2)

    for r_dir in irstd_roots:
        ds = Irstd1kDataset(r_dir, split_purpose="train", target_hw=lwir_hw)
        for idx, img_p in enumerate(ds.image_paths):
            rp = img_p.resolve()
            if rp in seen_img_files:
                continue
            seen_img_files.add(rp)
            irstd_samples.append(ds[idx])
            if len(irstd_samples) >= max_samples:
                break
        if len(irstd_samples) >= max_samples:
            break

    hituav_samples: list[ThermalSample] = []
    for r_dir in hituav_roots:
        ds = HitUavDataset(r_dir, split_purpose="train", target_hw=lwir_hw)
        for idx, img_p in enumerate(ds.image_paths):
            rp = img_p.resolve()
            if rp in seen_img_files:
                continue
            seen_img_files.add(rp)
            hituav_samples.append(ds[idx])
            if len(hituav_samples) >= max_samples:
                break
        if len(hituav_samples) >= max_samples:
            break

    # Balance across both datasets when both are present so neither dataset starves the other
    if irstd_samples and hituav_samples:
        take_i = min(len(irstd_samples), max(half_cap, int(max_samples) - len(hituav_samples)))
        take_h = min(len(hituav_samples), int(max_samples) - take_i)
        combined: list[ThermalSample] = []
        for idx in range(max(take_i, take_h)):
            if idx < take_i:
                combined.append(irstd_samples[idx])
            if idx < take_h:
                combined.append(hituav_samples[idx])
        return combined[:max_samples]
    return (irstd_samples + hituav_samples)[:max_samples]


def _synthesize_eo_lwir_pair(
    sample: CurriculumSample,
    eo_hw: tuple[int, int],
    lwir_hw: tuple[int, int],
    rng: np.random.Generator,
    *,
    target_uv_norm: tuple[float, float] | None = None,
    target_scale: float = 1.0,
    style_shift: float = 0.0,
    hot_ground_negative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthesize co-registered physics EO `[3, H, W]` + 2-channel LWIR `[2, H_l, W_l]` pair."""
    h_eo, w_eo = eo_hw
    h_lw, w_lw = lwir_hw

    # Target location in normalized [0.2..0.8] coordinates
    if target_uv_norm is None:
        tu = float(rng.uniform(0.20, 0.80))
        tv = float(rng.uniform(0.20, 0.80))
    else:
        tu, tv = float(target_uv_norm[0]), float(target_uv_norm[1])

    yy_eo, xx_eo = np.meshgrid(
        np.linspace(0.0, 1.0, h_eo, dtype=np.float32),
        np.linspace(0.0, 1.0, w_eo, dtype=np.float32),
        indexing="ij",
    )
    if target_scale <= 0.0 or hot_ground_negative:
        blob_eo = np.zeros((h_eo, w_eo), dtype=np.float32)
    else:
        sigma_blob = 0.045 * float(target_scale)
        dist2_eo = (xx_eo - tu) ** 2 + (yy_eo - tv) ** 2
        blob_eo = np.exp(-dist2_eo / (2.0 * (sigma_blob**2))).astype(np.float32)

    if hot_ground_negative:
        # Realistic below-horizon hot ground / powerplant / fire clutter distractors (`hm=0, cls=0`)
        ground_pat = (
            0.30
            + 0.14 * np.sin(9.0 * xx_eo + float(rng.uniform(0, 6.28))) * np.cos(7.0 * yy_eo)
            + 0.08 * np.cos(14.0 * xx_eo)
        ).astype(np.float32)
        eo_img = np.stack(
            [ground_pat * 0.96, ground_pat * 0.90, ground_pat * 0.78],
            axis=0,
        ).astype(np.float32)
    else:
        # Clean EO scene: sky gradient + subtle cloud/terrain texture + aircraft contrast
        sky_grad = (0.35 + 0.35 * (1.0 - yy_eo) + 0.04 * style_shift).astype(np.float32)
        eo_img = np.stack(
            [
                sky_grad * 0.88 + 0.05 * np.sin(6.0 * xx_eo),
                sky_grad * 0.94 + 0.04 * np.cos(5.0 * yy_eo),
                sky_grad * 1.05,
            ],
            axis=0,
        ).astype(np.float32)
        # Dark fuselage / sunlit wing glint on EO
        eo_img = eo_img * (1.0 - 0.55 * blob_eo[None, :, :]) + 0.15 * blob_eo[None, :, :]

    # Apply EO degradation consistent with curriculum bucket
    if sample.q_eo <= 0.1:
        if sample.submode == "pitch_black_night":
            eo_img = eo_img * 0.04 + rng.normal(0.0, 0.015, size=eo_img.shape).astype(np.float32)
        else:
            # Dense fog or solar glare washes EO contrast to uniform veil
            veil = 0.88 if sample.submode == "solar_glare" else 0.72
            eo_img = 0.08 * eo_img + 0.92 * veil
    elif sample.q_eo < 0.95:
        eo_img = sample.q_eo * eo_img + (1.0 - sample.q_eo) * 0.70

    eo_img = np.clip(eo_img + rng.normal(0.0, 0.01, size=eo_img.shape).astype(np.float32), 0.0, 1.0)

    # Co-registered physics LWIR forward pass via src/xvs/sensor/lwir.py
    t_air_k = 286.0 + 1.5 * float(style_shift)
    temp_k_hr = torch.full((h_eo, w_eo), t_air_k - 3.0, dtype=torch.float32)
    if hot_ground_negative:
        # Extended hot roof / powerplant / volcanic vent blobs (+18..+42 K) strictly labeled 0
        clutter_k = np.zeros((h_eo, w_eo), dtype=np.float32)
        for _ in range(int(rng.integers(2, 6))):
            cu = float(rng.uniform(0.10, 0.90))
            cv_c = float(rng.uniform(0.25, 0.92))
            su = float(rng.uniform(0.05, 0.18))
            sv = float(rng.uniform(0.04, 0.15))
            dk = float(rng.uniform(18.0, 42.0))
            clutter_k += dk * np.exp(-(((xx_eo - cu) / su) ** 2 + ((yy_eo - cv_c) / sv) ** 2) * 0.5)
        temp_k_hr = temp_k_hr + torch.from_numpy(clutter_k)
    else:
        # Hot engine/exhaust plume (+24 K) on target blob
        temp_k_hr = temp_k_hr + torch.from_numpy(blob_eo) * (24.0 * max(1.0, float(target_scale)))
    emiss_hr = torch.full((h_eo, w_eo), 0.94, dtype=torch.float32)
    range_km_hr = torch.full((h_eo, w_eo), 0.65, dtype=torch.float32)
    elev_deg_hr = torch.linspace(12.0, 1.0, steps=h_eo, dtype=torch.float32).view(h_eo, 1).expand(h_eo, w_eo)

    sensor_cfg = LwirSensorConfig(out_hw=(h_lw, w_lw), netd_mk=40.0)
    prev_rad = (
        torch.full((h_lw, w_lw), 28.0, dtype=torch.float32)
        if sample.lwir_fault.ffc_freeze
        else None
    )
    gen = torch.Generator().manual_seed(int(rng.integers(1, 2**31 - 1)))
    lwir_out = simulate_lwir_frame(
        temp_k_hr=temp_k_hr,
        emiss_hr=emiss_hr,
        range_km_hr=range_km_hr,
        elev_deg_hr=elev_deg_hr,
        t_air_k=t_air_k,
        rvr_km=sample.rvr_km,
        regime=sample.regime,
        sensor_cfg=sensor_cfg,
        fault_cfg=sample.lwir_fault,
        prev_meas_rad=prev_rad,
        generator=gen,
        noise_off=False,
    )
    two_ch = lwir_out["two_channel"]
    assert isinstance(two_ch, torch.Tensor)

    # Stride-4 target heatmap & 3-class evidential labels matching XvsThermalNet output grid
    fh = h_eo // 4
    fw = w_eo // 4
    if hot_ground_negative or (sample.q_eo <= 0.05 and sample.q_lwir <= 0.05):
        hm_target = np.zeros((fh, fw), dtype=np.float32)
    else:
        yy_f, xx_f = np.meshgrid(
            np.linspace(0.0, 1.0, fh, dtype=np.float32),
            np.linspace(0.0, 1.0, fw, dtype=np.float32),
            indexing="ij",
        )
        hm_target = np.exp(-((xx_f - tu) ** 2 + (yy_f - tv) ** 2) / (2.0 * (0.06**2))).astype(np.float32)
    cls_target = np.where(hm_target > 0.35, 1, 0).astype(np.int64)

    return (
        torch.from_numpy(eo_img),
        two_ch.float(),
        torch.tensor(sample.q_eo, dtype=torch.float32),
        torch.tensor(sample.q_lwir, dtype=torch.float32),
        torch.from_numpy(hm_target[None, :, :]),
        torch.from_numpy(cls_target),
    )


def _apply_v53_parallax_and_roll(
    lwir_2ch: torch.Tensor,
    hm_target: torch.Tensor,
    q_eo: float,
    q_lwir: float,
    rng: np.random.Generator,
    *,
    dx_lwir_px: float | None = None,
    dy_lwir_px: float | None = None,
    droll_deg: float | None = None,
    max_shift_lwir_px: float = 12.0,
    max_roll_deg: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Apply `(±12 LWIR px, ±5 deg roll)` EO<->LWIR parallax warp and compute conflict flag (`shift >= 6 px`).

    Returns `(warped_lwir_2ch, warped_lwir_hm, shift_lwir_px, parallax_conflict_flag)`.
    """
    h_l, w_l = int(lwir_2ch.shape[1]), int(lwir_2ch.shape[2])
    fh, fw = int(hm_target.shape[1]), int(hm_target.shape[2])

    if dx_lwir_px is None or dy_lwir_px is None:
        if float(rng.random()) < 0.55:
            # Gross boresight conflict regime: shift in [6.0, max_shift_lwir_px] LWIR px
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad = float(rng.uniform(6.0, max(6.5, float(max_shift_lwir_px))))
            dx = rad * math.cos(ang)
            dy = rad * math.sin(ang)
        else:
            # Mild aligned jitter regime: shift <= 2.5 LWIR px
            dx = float(rng.uniform(-2.0, 2.0))
            dy = float(rng.uniform(-2.0, 2.0))
    else:
        dx = float(dx_lwir_px)
        dy = float(dy_lwir_px)

    roll = float(rng.uniform(-max_roll_deg, max_roll_deg)) if droll_deg is None else float(droll_deg)
    shift_px = float(math.hypot(dx, dy))

    mat_lw = cv2.getRotationMatrix2D((0.5 * float(w_l), 0.5 * float(h_l)), roll, 1.0).astype(np.float32)
    mat_lw[0, 2] += float(dx)
    mat_lw[1, 2] += float(dy)
    lw_np = lwir_2ch.detach().cpu().float().numpy()
    lw_warp = np.stack(
        [
            cv2.warpAffine(
                lw_np[c],
                mat_lw,
                (w_l, h_l),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            for c in range(lw_np.shape[0])
        ],
        axis=0,
    ).astype(np.float32)

    sx_hm = float(fw) / float(max(1, w_l))
    sy_hm = float(fh) / float(max(1, h_l))
    mat_hm = cv2.getRotationMatrix2D((0.5 * float(fw), 0.5 * float(fh)), roll, 1.0).astype(np.float32)
    mat_hm[0, 2] += float(dx) * sx_hm
    mat_hm[1, 2] += float(dy) * sy_hm
    hm_np = hm_target[0].detach().cpu().float().numpy()
    hm_warp = cv2.warpAffine(
        hm_np,
        mat_hm,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)

    both_healthy = (float(q_eo) >= 0.5) and (float(q_lwir) >= 0.5)
    is_conflict = 1.0 if (both_healthy and shift_px >= 6.0) else 0.0
    return (
        torch.from_numpy(lw_warp),
        torch.from_numpy(np.clip(hm_warp, 0.0, 1.0)[None, :, :]),
        shift_px,
        is_conflict,
    )


def _synthesize_distinct_eo_for_real_lwir(
    real_sample: ThermalSample,
    sample: CurriculumSample,
    eo_hw: tuple[int, int],
    rng: np.random.Generator,
    *,
    is_hot_ground_neg: bool,
) -> torch.Tensor:
    """Synthesize genuinely distinct EO RGB appearance paired with real `IRSTD-1k` or `HIT-UAV` LWIR (ZERO upsampled-LWIR copy!)."""
    h_eo, w_eo = eo_hw
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h_eo, dtype=np.float32),
        np.linspace(0.0, 1.0, w_eo, dtype=np.float32),
        indexing="ij",
    )
    if is_hot_ground_neg:
        # Distinct below-horizon EO urban road / parking lot / building terrain texture
        base_g = (
            float(rng.uniform(0.22, 0.44))
            + 0.08 * np.sin(2.0 * math.pi * (rng.uniform(2.0, 6.0) * xx + rng.uniform(1.0, 4.0) * yy))
            + 0.06 * np.cos(2.0 * math.pi * rng.uniform(3.0, 8.0) * yy)
        ).astype(np.float32)
        eo_np = np.stack(
            [
                base_g * float(rng.uniform(0.92, 1.08)),
                base_g * float(rng.uniform(0.90, 1.04)),
                base_g * float(rng.uniform(0.80, 0.96)),
            ],
            axis=0,
        ).astype(np.float32)
    else:
        # Distinct realistic EO sky + clouds + dark-fuselage aircraft silhouette at IRSTD-1k mask location
        sky_top = np.array([0.36, 0.56, 0.84], dtype=np.float32)
        sky_bot = np.array([0.74, 0.82, 0.90], dtype=np.float32)
        clouds = (
            0.09
            * np.sin(2.0 * math.pi * (rng.uniform(1.5, 4.0) * xx + rng.uniform(0.5, 2.0) * yy))
            * np.cos(2.0 * math.pi * rng.uniform(1.0, 3.0) * yy)
        ).astype(np.float32)
        eo_np = (
            sky_top[:, None, None] * (1.0 - yy[None, :, :])
            + sky_bot[:, None, None] * yy[None, :, :]
            + clouds[None, :, :]
        ).astype(np.float32)
        mask_eo = (
            F.interpolate(
                real_sample.mask.unsqueeze(0).float(),
                size=(h_eo, w_eo),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        # Dark-fuselage EO silhouette contrast (physically opposite polarity to bright-plume LWIR!)
        dark_val = float(rng.uniform(0.10, 0.24))
        eo_np = eo_np * (1.0 - 0.85 * mask_eo[None, :, :]) + dark_val * (0.85 * mask_eo[None, :, :])

    eo_rgb = torch.from_numpy(np.clip(eo_np, 0.0, 1.0).astype(np.float32))
    if sample.q_eo <= 0.1:
        veil = 0.04 if sample.submode == "pitch_black_night" else 0.78
        eo_rgb = 0.08 * eo_rgb + 0.92 * veil
    elif sample.q_eo < 0.95:
        eo_rgb = float(sample.q_eo) * eo_rgb + (1.0 - float(sample.q_eo)) * 0.70
    noise = torch.from_numpy(rng.normal(0.0, 0.012, size=(3, h_eo, w_eo)).astype(np.float32))
    return (eo_rgb + noise).clamp(0.0, 1.0)


def _build_real_thermal_curriculum_item(
    real_sample: ThermalSample,
    sample: CurriculumSample,
    eo_hw: tuple[int, int],
    rng: np.random.Generator,
    *,
    v53_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair a real `IRSTD-1k`/`HIT-UAV` 2-ch LWIR frame + mask with curriculum-degraded EO & targets.

    In `v53_mode=True`:
      - Eliminates upsampled-LWIR fake EO copy (synthesizes physically distinct dark-fuselage EO).
      - Repurposes `HIT-UAV` (`CC-BY-4.0`) as `100%` hot-ground hard negatives (`hm=0, cls=0`).
    """
    h_eo, w_eo = eo_hw
    lwir_2ch = real_sample.lwir_2ch.float().clone()  # [2, H_l, W_l]
    if sample.q_lwir <= 0.1:
        # Apply LWIR degradation (crossover/washout collapse) consistent with fault curriculum
        lwir_2ch[0] = lwir_2ch[0].mean() + 0.02 * (lwir_2ch[0] - lwir_2ch[0].mean())
        lwir_2ch[1] = lwir_2ch[1] * 0.05
    elif sample.q_lwir < 0.95:
        lwir_2ch = lwir_2ch * float(sample.q_lwir)

    is_hit_uav_neg = bool(v53_mode and real_sample.dataset_id == "hit_uav")
    if v53_mode:
        eo_rgb = _synthesize_distinct_eo_for_real_lwir(
            real_sample, sample, eo_hw, rng, is_hot_ground_neg=is_hit_uav_neg
        )
    else:
        # Legacy v5.2 co-registered EO luminance proxy upsampled from real LWIR structure
        eo_base = F.interpolate(
            real_sample.lwir_2ch[:1].unsqueeze(0).float(),
            size=(h_eo, w_eo),
            mode="bilinear",
            align_corners=False,
        )[0]  # [1, H_eo, W_eo]
        eo_rgb = eo_base.repeat(3, 1, 1).clamp(0.0, 1.0)
        if sample.q_eo <= 0.1:
            veil = 0.04 if sample.submode == "pitch_black_night" else 0.78
            eo_rgb = 0.08 * eo_rgb + 0.92 * veil
        elif sample.q_eo < 0.95:
            eo_rgb = float(sample.q_eo) * eo_rgb + (1.0 - float(sample.q_eo)) * 0.70
        noise = torch.from_numpy(rng.normal(0.0, 0.01, size=(3, h_eo, w_eo)).astype(np.float32))
        eo_rgb = (eo_rgb + noise).clamp(0.0, 1.0)

    fh, fw = h_eo // 4, w_eo // 4
    if is_hit_uav_neg or (sample.q_eo <= 0.05 and sample.q_lwir <= 0.05):
        hm_target = torch.zeros((1, fh, fw), dtype=torch.float32)
    else:
        m_src = real_sample.mask.unsqueeze(0).float()
        mh, mw = int(m_src.shape[-2]), int(m_src.shape[-1])
        if mh >= fh and mw >= fw:
            # Peak-preserving max-pool downscale so tiny 2-4 px IRSTD-1k targets never vanish
            hm_target = F.adaptive_max_pool2d(m_src, output_size=(fh, fw))[0].clamp(0.0, 1.0)
        else:
            mid_h, mid_w = max(mh, fh), max(mw, fw)
            if (mid_h, mid_w) != (mh, mw):
                m_src = F.interpolate(m_src, size=(mid_h, mid_w), mode="nearest")
            hm_target = F.adaptive_max_pool2d(m_src, output_size=(fh, fw))[0].clamp(0.0, 1.0)
    cls_target = (hm_target[0] > 0.20).to(dtype=torch.long)

    return (
        eo_rgb,
        lwir_2ch,
        torch.tensor(sample.q_eo, dtype=torch.float32),
        torch.tensor(sample.q_lwir, dtype=torch.float32),
        hm_target,
        cls_target,
    )


class ThermalCurriculumDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Paired EO + 2-channel LWIR dataset sampled under exact `40/20/20/10/10` fault curriculum.

    Automatically blends real `IRSTD-1k` / `HIT-UAV` frames discovered in `real_samples` with
    physics-rendered EO/LWIR pairs (`D16-G4-M4`), falling back cleanly to 100% physics pairs in
    offline `--smoke-local` runs when no real cache directory is provided.

    When `v53_mode=True`, repurposes `HIT-UAV` as `100%` hot-ground hard negatives, eliminates
    upsampled-LWIR fake EO copies, injects `(±12 LWIR px, ±5° roll)` parallax augmentation,
    and returns 9-tuples `(eo, lwir, q_eo, q_lwir, fused_hm, cls, eo_hm, lwir_hm, conflict)`.
    """

    def __init__(
        self,
        num_samples: int,
        eo_hw: tuple[int, int],
        lwir_hw: tuple[int, int],
        *,
        real_samples: list[ThermalSample] | None = None,
        seed: int = 20261121,
        v53_mode: bool = False,
        real_frac: float = 0.60,
        parallax_aug_prob: float = 0.35,
        parallax_max_shift_lwir_px: float = 12.0,
        parallax_max_roll_deg: float = 5.0,
    ) -> None:
        sampler = FaultCurriculumSampler(seed=seed)
        self.curriculum_samples: list[CurriculumSample] = sampler.sample_batch(int(num_samples))
        self.bucket_distribution: dict[str, float] = summarize_curriculum_distribution(
            self.curriculum_samples
        )
        self.v53_mode = bool(v53_mode)
        rng = np.random.default_rng(seed + 13)
        real_pool: list[ThermalSample] = [] if real_samples is None else list(real_samples)
        self.real_samples_loaded: int = len(real_pool)
        self.real_items_count: int = 0
        self.physics_items_count: int = 0
        self.hit_uav_hot_neg_count: int = 0
        self.parallax_conflict_count: int = 0
        self.items: list[tuple[torch.Tensor, ...]] = []
        n_total = len(self.curriculum_samples)
        clamped_real_frac = max(0.0, min(1.0, float(real_frac)))
        n_real_target = int(round(n_total * clamped_real_frac)) if real_pool else 0
        for idx, csample in enumerate(self.curriculum_samples):
            # Exact ratio mixing (default 60% real / 40% physics) so large real pools never starve physics curriculum
            use_real = bool(
                real_pool
                and n_real_target > 0
                and (((idx + 1) * n_real_target) // max(1, n_total) > (idx * n_real_target) // max(1, n_total))
            )
            if use_real:
                rs = real_pool[self.real_items_count % len(real_pool)]
                base_item = _build_real_thermal_curriculum_item(
                    rs, csample, eo_hw, rng, v53_mode=self.v53_mode
                )
                self.real_items_count += 1
                if self.v53_mode and rs.dataset_id == "hit_uav":
                    self.hit_uav_hot_neg_count += 1
            else:
                hot_neg = bool(self.v53_mode and (idx % 3 == 2))
                base_item = _synthesize_eo_lwir_pair(
                    csample, eo_hw, lwir_hw, rng, hot_ground_negative=hot_neg
                )
                self.physics_items_count += 1
            if self.v53_mode:
                eo_t, lw_t, qe_t, ql_t, hm_t, cls_t = base_item
                eo_hm = hm_t.clone()
                lw_hm = hm_t.clone()
                conflict_f = 0.0
                has_fg_target = bool(float(hm_t.max().item()) > 0.05)
                if (
                    has_fg_target
                    and float(rng.random()) < float(parallax_aug_prob)
                    and float(qe_t.item()) >= 0.5
                    and float(ql_t.item()) >= 0.5
                ):
                    lw_t, lw_hm, _shift_px, conflict_f = _apply_v53_parallax_and_roll(
                        lw_t,
                        hm_t,
                        float(qe_t.item()),
                        float(ql_t.item()),
                        rng,
                        max_shift_lwir_px=float(parallax_max_shift_lwir_px),
                        max_roll_deg=float(parallax_max_roll_deg),
                    )
                    if conflict_f > 0.5:
                        hm_t = torch.zeros_like(hm_t)
                        self.parallax_conflict_count += 1
                self.items.append(
                    (
                        eo_t,
                        lw_t,
                        qe_t,
                        ql_t,
                        hm_t,
                        cls_t,
                        eo_hm,
                        lw_hm,
                        torch.tensor(conflict_f, dtype=torch.float32),
                    )
                )
            else:
                self.items.append(base_item)
        self.real_item_fraction: float = (
            float(self.real_items_count) / float(max(1, len(self.items)))
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self.items[idx]


@torch.no_grad()
def evaluate_thermal_four_modes(
    model: XvsThermalNet,
    eo_hw: tuple[int, int],
    lwir_hw: tuple[int, int],
    device: torch.device,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Verify nominal, EO-blind, LWIR-blind, and dual-blind (`vacuity == 1.0`) Simplex behavior."""
    model.eval()
    rng = np.random.default_rng(seed + 777)
    probe_specs = (
        ("nominal", CurriculumSample("nominal", "clear_nominal", "clear", 10.0, 1.0, 1.0, 0.0, ThermalFaultConfig())),
        ("eo_blind", CurriculumSample("eo_degraded", "radiation_fog", "radiation_fog", 0.45, 0.0, 1.0, 0.0, ThermalFaultConfig())),
        ("lwir_blind", CurriculumSample("lwir_degraded", "thermal_crossover", "clear", 10.0, 1.0, 0.0, 0.0, ThermalFaultConfig(thermal_crossover=True))),
        ("dual_blind", CurriculumSample("simultaneous_dve", "in_stratus_cloud", "stratus_cloud", 0.12, 0.0, 0.0, 1.0, ThermalFaultConfig(cloud_washout=True))),
    )
    results: dict[str, Any] = {}
    for name, csample in probe_specs:
        x_eo, x_lw, q_eo, q_lw, _, _ = _synthesize_eo_lwir_pair(csample, eo_hw, lwir_hw, rng)
        out = model(
            x_eo.unsqueeze(0).to(device),
            x_lw.unsqueeze(0).to(device),
            q_eo.view(1).to(device),
            q_lw.view(1).to(device),
        )
        vac_mean = float(out["vacuity"].mean().detach().cpu().item())
        vac_min = float(out["vacuity"].min().detach().cpu().item())
        mode = select_thermal_simplex_mode(float(csample.q_eo), float(csample.q_lwir), vac_mean)
        results[name] = {
            "vacuity_mean": vac_mean,
            "vacuity_min": vac_min,
            "w_eo_eff": float(out["w_eo_eff"].mean().detach().cpu().item()),
            "w_lwir_eff": float(out["w_lwir_eff"].mean().detach().cpu().item()),
            "simplex_mode": mode.name,
        }

    dual_vac_exact = abs(float(results["dual_blind"]["vacuity_mean"]) - 1.0) <= 1e-6
    modes_correct = (
        results["nominal"]["simplex_mode"] == ThermalSimplexMode.NOMINAL_FUSION.name
        and results["eo_blind"]["simplex_mode"] == ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY.name
        and results["lwir_blind"]["simplex_mode"] == ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY.name
        and results["dual_blind"]["simplex_mode"] == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK.name
    )
    results["dual_blind_vacuity_exact_one"] = bool(dual_vac_exact)
    results["four_mode_simplex_pass"] = bool(modes_correct and dual_vac_exact)

    if getattr(model, "enable_disagreement_vacuity", False):
        nom_spec = probe_specs[0][1]
        xe_p, xl_p, qe_p, ql_p, hm_p, _ = _synthesize_eo_lwir_pair(nom_spec, eo_hw, lwir_hw, rng)
        for tag, shift_xy in (("parallax_6x6", (6.0, 6.0)), ("parallax_10x10", (10.0, 10.0))):
            xl_shift, _, _, _ = _apply_v53_parallax_and_roll(
                xl_p,
                hm_p,
                1.0,
                1.0,
                rng,
                dx_lwir_px=shift_xy[0],
                dy_lwir_px=shift_xy[1],
                droll_deg=0.0,
            )
            out_p = model(
                xe_p.unsqueeze(0).to(device),
                xl_shift.unsqueeze(0).to(device),
                qe_p.view(1).to(device),
                ql_p.view(1).to(device),
            )
            results[f"{tag}_vacuity_mean"] = float(out_p["vacuity"].mean().detach().cpu().item())
            results[f"{tag}_disagreement_hm_max"] = (
                float(out_p["disagreement_hm"].max().detach().cpu().item())
                if "disagreement_hm" in out_p
                else 0.0
            )
    return results


@torch.no_grad()
def _extract_penultimate_f_fused_gap(
    model: XvsThermalNet,
    eo_batch: torch.Tensor,
    lwir_batch: torch.Tensor,
    q_eo_batch: torch.Tensor,
    q_lwir_batch: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Extract `[B, 64]` penultimate vectors via chunked global average pooling of `out['f_fused']` (`BC-G4-m2`)."""
    n_total = int(eo_batch.shape[0])
    step = max(1, int(chunk_size))
    chunks: list[torch.Tensor] = []
    for start in range(0, n_total, step):
        end = min(n_total, start + step)
        out = model(
            eo_batch[start:end].to(device),
            lwir_batch[start:end].to(device),
            q_eo_batch[start:end].to(device),
            q_lwir_batch[start:end].to(device),
        )
        # Zero edits to model.py: `out["f_fused"]` has shape [b, 64, H/4, W/4]
        chunks.append(out["f_fused"].float().mean(dim=(-2, -1)).detach().cpu())
    return torch.cat(chunks, dim=0)


def run_real_feature_xvs_vraim_audit(
    model: XvsThermalNet,
    eo_hw: tuple[int, int],
    lwir_hw: tuple[int, int],
    out_dir: Path,
    device: torch.device,
    *,
    smoke_local: bool = False,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Run real-feature `GAP(out['f_fused'])` `[B, 64]` XVS-VRAIM audit & emit `easa_rlm_07_11_evidence.json`."""
    model.eval()
    rng = np.random.default_rng(seed + 909)
    n_id = 48 if smoke_local else 180
    n_ood = 24 if smoke_local else 80
    n_sae = 24 if smoke_local else 64

    id_sample = CurriculumSample("nominal", "clear_nominal", "clear", 12.0, 1.0, 1.0, 0.0, ThermalFaultConfig())
    ood_sample = CurriculumSample(
        "simultaneous_dve",
        "coastal_advection_fog_night",
        "advection_fog",
        0.15,
        0.15,
        0.15,
        0.95,
        ThermalFaultConfig(cloud_washout=True, thermal_crossover=True),
    )

    def _collect_features(csample: CurriculumSample, count: int) -> np.ndarray:
        eo_l, lw_l, qe_l, ql_l = [], [], [], []
        for _ in range(count):
            xe, xl, qe, ql, _, _ = _synthesize_eo_lwir_pair(csample, eo_hw, lwir_hw, rng)
            eo_l.append(xe)
            lw_l.append(xl)
            qe_l.append(qe)
            ql_l.append(ql)
        feats = _extract_penultimate_f_fused_gap(
            model,
            torch.stack(eo_l),
            torch.stack(lw_l),
            torch.stack(qe_l),
            torch.stack(ql_l),
            device,
        )
        return feats.numpy().astype(np.float64)

    id_feats = _collect_features(id_sample, n_id)
    ood_feats = _collect_features(ood_sample, n_ood)
    n_fit = int(n_id * 0.50)
    n_cal = int(n_id * 0.25)

    ood_det = PenultimateMahalanobisOOD(shrinkage_lambda=1.0e-2)
    ood_det.fit(id_feats[:n_fit], np.zeros(n_fit, dtype=np.int64))
    cal_thr = ood_det.calibrate_conformal_threshold(id_feats[n_fit : n_fit + n_cal], alpha=0.05)
    id_scores = ood_det.score(id_feats[n_fit + n_cal :])
    ood_scores = ood_det.score(ood_feats)
    ood_auroc = float(compute_auroc(id_scores, ood_scores))
    ood_id_fpr = float(np.mean(id_scores > cal_thr))

    # Real penultimate feature pairs for Top-K SAE content-vs-style shortcut audit:
    # Content pair: close-range aircraft target vs empty sky (same clear weather style).
    # Style pair: identical aircraft scene under subtle illumination/ambient-temperature shift.
    ca_l, cb_l, sa_l, sb_l = [], [], [], []
    for _ in range(n_sae):
        uv_a = (float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.30, 0.70)))
        xe_a, xl_a, qe_a, ql_a, _, _ = _synthesize_eo_lwir_pair(
            id_sample, eo_hw, lwir_hw, rng, target_uv_norm=uv_a, target_scale=2.8, style_shift=0.0
        )
        xe_b, xl_b, qe_b, ql_b, _, _ = _synthesize_eo_lwir_pair(
            id_sample, eo_hw, lwir_hw, rng, target_uv_norm=uv_a, target_scale=0.0, style_shift=0.0
        )
        xe_sb, xl_sb, qe_sb, ql_sb, _, _ = _synthesize_eo_lwir_pair(
            id_sample, eo_hw, lwir_hw, rng, target_uv_norm=uv_a, target_scale=2.8, style_shift=0.08
        )
        ca_l.append((xe_a, xl_a, qe_a, ql_a))
        cb_l.append((xe_b, xl_b, qe_b, ql_b))
        sa_l.append((xe_a, xl_a, qe_a, ql_a))
        sb_l.append((xe_sb, xl_sb, qe_sb, ql_sb))

    def _pack_extract(items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return _extract_penultimate_f_fused_gap(
            model,
            torch.stack([t[0] for t in items]),
            torch.stack([t[1] for t in items]),
            torch.stack([t[2] for t in items]),
            torch.stack([t[3] for t in items]),
            device,
        )

    feat_dim = int(id_feats.shape[1])
    torch.manual_seed(seed)
    sae = TopKSparseAutoencoder(in_dim=feat_dim, dict_dim=feat_dim * 2, top_k=8)
    sae_res = audit_content_vs_style_reliance(
        sae,
        _pack_extract(ca_l),
        _pack_extract(cb_l),
        _pack_extract(sa_l),
        _pack_extract(sb_l),
        train_steps=30 if smoke_local else 80,
    )

    ast_runner = AdaptiveStressTester(seed=seed)
    ast_trajs = ast_runner.run_search(
        n_generations=3 if smoke_local else 6,
        pop_size=12 if smoke_local else 24,
        elite_k=4 if smoke_local else 6,
    )
    easa_doc = build_easa_rlm_07_11_report(
        seed=seed,
        ood_auroc=ood_auroc,
        ood_conformal_threshold=float(cal_thr),
        ood_id_fpr=ood_id_fpr,
        sae_result=sae_res,
        ast_results=ast_trajs,
    )
    easa_doc["all_requirements_pass"] = all(
        bool(easa_doc[req_key]["pass"])
        for req_key in ("R-LM-07", "R-LM-08", "R-LM-09", "R-LM-10", "R-LM-11")
    )
    easa_doc["penultimate_feature_source"] = "XvsThermalNet.f_fused_global_avg_pool"
    easa_doc["penultimate_feature_dim"] = feat_dim
    easa_path = out_dir / "easa_rlm_07_11_evidence.json"
    easa_path.write_text(json.dumps(easa_doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return easa_doc


class _XvsThermalOnnxWrapper(nn.Module):
    """Flat-tuple ONNX export wrapper for `XvsThermalNet`."""

    def __init__(self, core: XvsThermalNet) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        x_eo: torch.Tensor,
        x_lwir: torch.Tensor,
        q_eo: torch.Tensor,
        q_lwir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.core(x_eo, x_lwir, q_eo, q_lwir)
        return (
            out["fused_hm_logits"],
            out["fused_box_reg"],
            out["vacuity"],
            out["probs"],
            out["w_eo_eff"],
            out["w_lwir_eff"],
        )


def convert_xvs_thermal_onnx_selective_fp16(fp32_model: Any) -> Any:
    """Pure-`onnx` Selective FP16 converter keeping `health_gate` & `evidential_head` strictly FP32.

    Guarantees exact `dual_blind vacuity == 1.0000` with `FP16 vs FP32 diff == 0.0` while keeping
    `xvs_thermal_v53_fp16.onnx <= 2.2 MiB`.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.ModelProto()
    model.CopyFrom(fp32_model)
    graph = model.graph

    inferred = onnx.shape_inference.infer_shapes(model)
    tensor_dtype: dict[str, int] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if vi.type.HasField("tensor_type"):
            tensor_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        tensor_dtype[init.name] = init.data_type

    always_fp32_ops = {
        "Where",
        "Greater",
        "Less",
        "Equal",
        "And",
        "Or",
        "Not",
        "IsNaN",
        "IsInf",
        "Softplus",
        "Sigmoid",
        "Div",
        "Sqrt",
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "Softmax",
        "Max",
        "Exp",
        "Clip",
        "BatchNormalization",
        "Resize",
        "Concat",
        "Abs",
        "Sub",
        "Expand",
        "Reshape",
        "Shape",
        "Slice",
    }
    always_fp32_tags = (
        "health_gate",
        "evidential_head",
        "fused_head",
        "eo_aux_head",
        "lwir_aux_head",
        "softplus",
        "where",
        "vacuity",
        "dual_blind",
    )

    fp32_node_indices: set[int] = set()
    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            continue
        lname = node.name.lower()
        is_enc_conv = node.op_type == "Conv" and ("eo_encoder" in lname or "lwir_encoder" in lname)
        if node.op_type in always_fp32_ops or any(t in lname for t in always_fp32_tags) or not is_enc_conv:
            fp32_node_indices.add(idx)

    tensor_consumers: dict[str, set[int]] = {}
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                tensor_consumers.setdefault(inp, set()).add(idx)

    def _is_f16_only(tname: str) -> bool:
        consumers = tensor_consumers[tname] if tname in tensor_consumers else None
        return bool(consumers) and consumers.isdisjoint(fp32_node_indices)

    def _is_f32_only(tname: str) -> bool:
        consumers = tensor_consumers[tname] if tname in tensor_consumers else None
        return bool(consumers) and consumers.issubset(fp32_node_indices)

    init_is_f16: set[str] = set()
    shared_f16: dict[str, str] = {}
    extra_inits: list[onnx.TensorProto] = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if _is_f16_only(init.name):
            arr16 = np.clip(numpy_helper.to_array(init), -65504.0, 65504.0).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr16, name=init.name))
            init_is_f16.add(init.name)
        elif not _is_f32_only(init.name):
            arr16 = np.clip(numpy_helper.to_array(init), -65504.0, 65504.0).astype(np.float16)
            cname = f"{init.name}__f16_copy"
            extra_inits.append(numpy_helper.from_array(arr16, name=cname))
            shared_f16[init.name] = cname
            init_is_f16.add(cname)
    graph.initializer.extend(extra_inits)

    live_dtype: dict[str, int] = dict(tensor_dtype)
    for name in init_is_f16:
        live_dtype[name] = TensorProto.FLOAT16
    for gi in graph.input:
        live_dtype[gi.name] = TensorProto.FLOAT

    out_orig_names = [o.name for o in graph.output]
    out_internal = {n: f"{n}__pre_out" for n in out_orig_names}
    new_nodes: list[onnx.NodeProto] = []
    cast_cache: dict[tuple[str, int], str] = {}
    cast_idx = 0

    def _ensure_dt(tname: str, target_dt: int) -> str:
        nonlocal cast_idx
        if not tname:
            return tname
        cur_dt = live_dtype[tname] if tname in live_dtype else TensorProto.FLOAT
        if cur_dt not in (TensorProto.FLOAT, TensorProto.FLOAT16) or cur_dt == target_dt:
            return tname
        key = (tname, target_dt)
        if key in cast_cache:
            return cast_cache[key]
        suf = "f16" if target_dt == TensorProto.FLOAT16 else "f32"
        cname = f"{tname}__cast_{suf}_{cast_idx}"
        cast_idx += 1
        new_nodes.append(helper.make_node("Cast", [tname], [cname], name=f"th_cast_{suf}_{cast_idx}", to=target_dt))
        live_dtype[cname] = target_dt
        cast_cache[key] = cname
        return cname

    for idx, node in enumerate(graph.node):
        if node.op_type == "Constant":
            new_nodes.append(node)
            continue
        node_dt = TensorProto.FLOAT if idx in fp32_node_indices else TensorProto.FLOAT16
        up_in: list[str] = []
        for inp in node.input:
            if not inp:
                up_in.append("")
                continue
            src = out_internal[inp] if inp in out_internal else inp
            if node_dt == TensorProto.FLOAT16 and src in shared_f16:
                up_in.append(shared_f16[src])
                continue
            idt = live_dtype[src] if src in live_dtype else TensorProto.FLOAT
            if idt in (TensorProto.FLOAT, TensorProto.FLOAT16):
                up_in.append(_ensure_dt(src, node_dt))
            else:
                up_in.append(src)
        up_out: list[str] = []
        for oname in node.output:
            mapped = out_internal[oname] if oname in out_internal else oname
            up_out.append(mapped)
            odt = tensor_dtype[oname] if oname in tensor_dtype else TensorProto.FLOAT
            live_dtype[mapped] = node_dt if odt in (TensorProto.FLOAT, TensorProto.FLOAT16) else odt
        node.ClearField("input")
        node.input.extend(up_in)
        node.ClearField("output")
        node.output.extend(up_out)
        new_nodes.append(node)

    for oname in out_orig_names:
        iname = out_internal[oname]
        cdt = live_dtype[iname] if iname in live_dtype else TensorProto.FLOAT
        if cdt != TensorProto.FLOAT:
            new_nodes.append(helper.make_node("Cast", [iname], [oname], name=f"th_out_f32_{oname}", to=TensorProto.FLOAT))
        else:
            new_nodes.append(helper.make_node("Identity", [iname], [oname], name=f"th_out_id_{oname}"))

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    graph.ClearField("value_info")
    inferred_out = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_out)
    return inferred_out


def export_xvs_thermal_onnx(
    model: XvsThermalNet,
    out_onnx_path: Path,
    eo_hw: tuple[int, int],
    lwir_hw: tuple[int, int],
    *,
    opset: int = 17,
    max_diff_tol: float = 1.0e-3,
    export_fp16: bool = False,
    fp16_onnx_path: Path | None = None,
    ort_threads: int = 2,
) -> dict[str, Any]:
    """Export `XvsThermalNet` to FP32 (+ optional Selective FP16) ONNX (opset 17) and verify ORT parity & exact dual_blind vac=1.0."""
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_copy = XvsThermalNet(
        eo_in_ch=model.eo_in_ch,
        lwir_in_ch=model.lwir_in_ch,
        feat_ch=model.feat_ch,
        num_evidential_classes=model.num_evidential_classes,
        enable_disagreement_vacuity=model.enable_disagreement_vacuity,
    )
    cpu_copy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    cpu_copy.eval()
    wrapper = _XvsThermalOnnxWrapper(cpu_copy).eval()

    rng = np.random.default_rng(20261122)
    x_eo = torch.from_numpy(rng.uniform(0.0, 1.0, size=(1, 3, eo_hw[0], eo_hw[1])).astype(np.float32))
    x_lwir = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(1, 2, lwir_hw[0], lwir_hw[1])).astype(np.float32))
    q_eo = torch.tensor([1.0], dtype=torch.float32)
    q_lwir = torch.tensor([1.0], dtype=torch.float32)

    export_kwargs: dict[str, Any] = {
        "input_names": ["x_eo", "x_lwir", "q_eo", "q_lwir"],
        "output_names": ["fused_hm_logits", "fused_box_reg", "vacuity", "probs", "w_eo_eff", "w_lwir_eff"],
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(
            wrapper,
            (x_eo, x_lwir, q_eo, q_lwir),
            str(out_onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            (x_eo, x_lwir, q_eo, q_lwir),
            str(out_onnx_path),
            **export_kwargs,
        )

    fp16_path: Path | None = None
    fp16_bytes: int | None = None
    fp16_sha: str | None = None
    if export_fp16:
        import onnx

        fp32_proto = onnx.load(str(out_onnx_path))
        onnx.checker.check_model(fp32_proto)
        fp16_proto = convert_xvs_thermal_onnx_selective_fp16(fp32_proto)
        fp16_path = (
            Path(fp16_onnx_path)
            if fp16_onnx_path is not None
            else out_onnx_path.with_name(out_onnx_path.stem.replace("_fp32", "") + "_fp16.onnx")
        )
        onnx.save(fp16_proto, str(fp16_path))
        fp16_bytes = int(fp16_path.stat().st_size)
        fp16_sha = hashlib.sha256(fp16_path.read_bytes()).hexdigest()

    with torch.no_grad():
        pt_outs = [t.detach().cpu().numpy() for t in wrapper(x_eo, x_lwir, q_eo, q_lwir)]

    max_abs_diff = 0.0
    ort_verified = False
    fp16_max_abs_diff: float | None = None
    fp16_ort_verified = False
    dual_blind_vac_fp32: float | None = None
    dual_blind_vac_fp16: float | None = None
    dual_blind_fp16_vs_fp32_diff: float | None = None
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(max(1, ort_threads))
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(out_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        nom_feed = {
            "x_eo": x_eo.numpy(),
            "x_lwir": x_lwir.numpy(),
            "q_eo": q_eo.numpy(),
            "q_lwir": q_lwir.numpy(),
        }
        ort_outs = sess.run(None, nom_feed)
        diffs = [float(np.max(np.abs(a - b))) for a, b in zip(pt_outs, ort_outs)]
        max_abs_diff = max(diffs) if diffs else 0.0
        ort_verified = bool(math.isfinite(max_abs_diff) and max_abs_diff < float(max_diff_tol))
        if not ort_verified:
            raise ValueError(
                f"XVS-ThermalNet ONNX parity check failed: max_abs_diff={max_abs_diff:.6e} >= {max_diff_tol}"
            )

        db_feed = {
            "x_eo": x_eo.numpy(),
            "x_lwir": x_lwir.numpy(),
            "q_eo": np.array([0.0], dtype=np.float32),
            "q_lwir": np.array([0.0], dtype=np.float32),
        }
        db_outs_32 = sess.run(None, db_feed)
        dual_blind_vac_fp32 = float(np.mean(db_outs_32[2]))

        if export_fp16 and fp16_path is not None and fp16_path.is_file():
            sess16 = ort.InferenceSession(str(fp16_path), sess_options=opts, providers=["CPUExecutionProvider"])
            ort16_outs = sess16.run(None, nom_feed)
            diffs16 = [float(np.max(np.abs(a - b))) for a, b in zip(pt_outs, ort16_outs)]
            fp16_max_abs_diff = max(diffs16) if diffs16 else 0.0
            fp16_ort_verified = bool(
                math.isfinite(fp16_max_abs_diff) and fp16_max_abs_diff < 5.0e-2
            )
            db_outs_16 = sess16.run(None, db_feed)
            dual_blind_vac_fp16 = float(np.mean(db_outs_16[2]))
            dual_blind_fp16_vs_fp32_diff = float(np.max(np.abs(db_outs_16[2] - db_outs_32[2])))
            if dual_blind_fp16_vs_fp32_diff != 0.0 or abs(dual_blind_vac_fp16 - 1.0) > 1e-6:
                raise ValueError(
                    f"XVS-ThermalNet FP16 dual_blind vacuity lock violated: "
                    f"vac16={dual_blind_vac_fp16}, diff={dual_blind_fp16_vs_fp32_diff}"
                )
    except ImportError:
        ort_verified = False

    sha = hashlib.sha256(out_onnx_path.read_bytes()).hexdigest()
    return {
        "path": str(out_onnx_path),
        "bytes": int(out_onnx_path.stat().st_size),
        "sha256": sha,
        "opset": int(opset),
        "eo_hw": [int(eo_hw[0]), int(eo_hw[1])],
        "lwir_hw": [int(lwir_hw[0]), int(lwir_hw[1])],
        "enable_disagreement_vacuity": bool(model.enable_disagreement_vacuity),
        "max_abs_diff": float(max_abs_diff),
        "ort_verified": bool(ort_verified),
        "dual_blind_vacuity_fp32": dual_blind_vac_fp32,
        "fp16_onnx_path": str(fp16_path) if fp16_path is not None else None,
        "fp16_onnx_bytes": fp16_bytes,
        "fp16_onnx_mib": round(fp16_bytes / float(1024 * 1024), 4) if fp16_bytes is not None else None,
        "fp16_onnx_sha256": fp16_sha,
        "fp16_max_abs_diff": fp16_max_abs_diff,
        "fp16_ort_verified": bool(fp16_ort_verified),
        "dual_blind_vacuity_fp16": dual_blind_vac_fp16,
        "dual_blind_fp16_vs_fp32_diff": dual_blind_fp16_vs_fp32_diff,
    }


def run_xvs_thermal_g4_training(cfg: XvsThermalG4Config) -> dict[str, Any]:
    """Execute full Colab G4/L4 or `--smoke-local` XVS-ThermalNet training + real-feature XVS-VRAIM."""
    enforce_xvs_thermal_licenses()
    is_v53 = bool(cfg.v53_mode)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.smoke_local else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    env_report = collect_blackwell_env_report()

    cache_dir: Path | None = cfg.data_cache_dir
    dl_status: dict[str, str] = {}
    if cfg.download_datasets and not cfg.smoke_local:
        cache_dir = (cfg.out_dir / "datasets") if cache_dir is None else cache_dir
        dl_status = download_hit_uav_and_irstd1k(cache_dir)

    epochs = 2 if cfg.smoke_local else int(cfg.epochs)
    steps_per_epoch = 4 if cfg.smoke_local else int(cfg.steps_per_epoch)
    num_samples = 24 if cfg.smoke_local else int(cfg.num_train_samples)
    eo_hw = (64, 80) if cfg.smoke_local else cfg.eo_hw
    lwir_hw = (32, 40) if cfg.smoke_local else cfg.lwir_hw
    candidates = (4, 8) if cfg.smoke_local else cfg.batch_candidates

    real_samples = discover_real_thermal_samples(
        cache_dir,
        lwir_hw,
        max_samples=max(num_samples, int(cfg.min_real_thermal_samples)),
    )
    if cfg.require_real_thermal and len(real_samples) < int(cfg.min_real_thermal_samples):
        raise RuntimeError(
            f"Fail-closed: require_real_thermal=True but discovered only {len(real_samples)} "
            f"< {cfg.min_real_thermal_samples} real thermal frames under {cache_dir!r} "
            f"(dl_status={dl_status})."
        )
    real_breakdown: dict[str, int] = {}
    for rs in real_samples:
        real_breakdown[rs.dataset_id] = (real_breakdown[rs.dataset_id] + 1) if rs.dataset_id in real_breakdown else 1

    train_ds = ThermalCurriculumDataset(
        num_samples=num_samples,
        eo_hw=eo_hw,
        lwir_hw=lwir_hw,
        real_samples=real_samples,
        seed=cfg.seed,
        v53_mode=is_v53,
        parallax_aug_prob=float(cfg.parallax_aug_prob),
        parallax_max_shift_lwir_px=float(cfg.parallax_max_shift_lwir_px),
        parallax_max_roll_deg=float(cfg.parallax_max_roll_deg),
    )
    print(
        f"[XVS-ThermalNet v53={is_v53}] Dataset ready: samples={len(train_ds)} real_thermal={train_ds.real_samples_loaded} "
        f"real_frac={train_ds.real_item_fraction*100.0:.1f}% breakdown={real_breakdown} "
        f"hit_uav_hot_neg={train_ds.hit_uav_hot_neg_count} parallax_conflicts={train_ds.parallax_conflict_count} "
        f"dl_status={dl_status} device={device} bf16={use_bf16}",
        flush=True,
    )

    eager_model = XvsThermalNet(
        eo_in_ch=3,
        lwir_in_ch=2,
        feat_ch=64,
        num_evidential_classes=3,
        enable_disagreement_vacuity=is_v53,
    ).to(device)
    # BC-G4-m4: Eager AutoVRAMBatchTuner BEFORE torch.compile + clean weight restore
    init_weights = {k: v.detach().cpu().clone() for k, v in eager_model.state_dict().items()}
    probe_opt = torch.optim.AdamW(eager_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _probe_step(bs: int) -> None:
        probe_opt.zero_grad(set_to_none=True)
        xe = torch.randn(bs, 3, eo_hw[0], eo_hw[1], device=device)
        xl = torch.randn(bs, 2, lwir_hw[0], lwir_hw[1], device=device)
        qe = torch.ones(bs, device=device)
        ql = torch.ones(bs, device=device)
        thm = torch.zeros(bs, 1, eo_hw[0] // 4, eo_hw[1] // 4, device=device)
        tcls = torch.zeros(bs, eo_hw[0] // 4, eo_hw[1] // 4, dtype=torch.long, device=device)
        pconf = torch.zeros(bs, device=device) if is_v53 else None
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = eager_model(xe, xl, qe, ql)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            losses = compute_thermal_multitask_loss(
                out,
                thm,
                tcls,
                lambda_aux=cfg.lambda_aux,
                lambda_ev=cfg.lambda_ev,
                parallax_conflict_mask=pconf,
            )
        losses["total"].backward()
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
        f"[XVS-ThermalNet] AutoVRAMBatchTuner: selected_batch={selected_bs} "
        f"peak_vram_gib={tuner_peak_gib:.2f} util={float(tuner_report['utilization_frac'])*100.0:.1f}%",
        flush=True,
    )

    ckpt_mgr = DriveCheckpointManager(
        local_dir=cfg.out_dir / "checkpoints",
        drive_dir=cfg.drive_ckpt_dir,
    )
    optimizer = torch.optim.AdamW(eager_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch = 0
    history: list[dict[str, float]] = []
    resumed = False

    latest_ckpt = ckpt_mgr.load_latest(tag_prefix="xvs_thermal_epoch_")
    if latest_ckpt is not None and "model_state_dict" in latest_ckpt:
        eager_model.load_state_dict(latest_ckpt["model_state_dict"])
        if "optimizer_state_dict" in latest_ckpt:
            optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])
        start_epoch = int(latest_ckpt["epoch"]) + 1
        if "history" in latest_ckpt and isinstance(latest_ckpt["history"], list):
            history = list(latest_ckpt["history"])
        resumed = True

    train_model: nn.Module = (
        torch.compile(eager_model)
        if (cfg.use_compile and device.type == "cuda" and not cfg.smoke_local)
        else eager_model
    )
    loader = DataLoader(train_ds, batch_size=max(2, selected_bs), shuffle=True)

    for epoch in range(start_epoch, epochs):
        train_model.train()
        ep_losses: list[float] = []
        step_idx = 0
        while step_idx < steps_per_epoch:
            for batch in loader:
                if is_v53:
                    xe_b, xl_b, qe_b, ql_b, thm_b, tcls_b, eohm_b, lwhm_b, pconf_b = batch
                else:
                    xe_b, xl_b, qe_b, ql_b, thm_b, tcls_b = batch
                    eohm_b = None
                    lwhm_b = None
                    pconf_b = None
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    out = train_model(
                        xe_b.to(device, non_blocking=True),
                        xl_b.to(device, non_blocking=True),
                        qe_b.to(device, non_blocking=True),
                        ql_b.to(device, non_blocking=True),
                    )
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    losses = compute_thermal_multitask_loss(
                        out,
                        thm_b.to(device, non_blocking=True),
                        tcls_b.to(device, non_blocking=True),
                        lambda_aux=cfg.lambda_aux,
                        lambda_ev=cfg.lambda_ev,
                        eo_target_hm=eohm_b.to(device, non_blocking=True) if eohm_b is not None else None,
                        lwir_target_hm=lwhm_b.to(device, non_blocking=True) if lwhm_b is not None else None,
                        parallax_conflict_mask=pconf_b.to(device, non_blocking=True) if pconf_b is not None else None,
                    )
                    loss = losses["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite XVS-ThermalNet loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(eager_model.parameters(), max_norm=2.0)
                optimizer.step()
                ep_losses.append(float(loss.detach().cpu().item()))
                step_idx += 1
                if step_idx >= steps_per_epoch:
                    break

        mean_loss = float(np.mean(ep_losses))
        history.append({"epoch": float(epoch), "train_total_loss": mean_loss})
        peak_gib = (
            float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[XVS-ThermalNet v53={is_v53}] epoch={epoch + 1}/{epochs} loss={mean_loss:.4f} peak_vram_gib={peak_gib:.2f}",
            flush=True,
        )
        raw_m = getattr(train_model, "_orig_mod", train_model)
        ckpt_mgr.save(
            f"xvs_thermal_epoch_{epoch:03d}",
            {
                "epoch": epoch,
                "v53_mode": is_v53,
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_m.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
        )

    raw_final: XvsThermalNet = getattr(train_model, "_orig_mod", train_model)
    four_mode_eval = evaluate_thermal_four_modes(raw_final, eo_hw, lwir_hw, device, seed=cfg.seed)
    vraim_doc = (
        run_real_feature_xvs_vraim_audit(
            raw_final,
            eo_hw,
            lwir_hw,
            cfg.out_dir,
            device,
            smoke_local=cfg.smoke_local,
            seed=cfg.seed,
        )
        if cfg.run_vraim
        else None
    )

    final_ckpt_path = cfg.out_dir / "xvs_thermal_final.pt"
    torch.save(
        {
            "artifact_tag": "SMOKE_LOCAL" if cfg.smoke_local else "COLAB_G4_GPU_ARTIFACT",
            "v53_mode": is_v53,
            "enable_disagreement_vacuity": bool(raw_final.enable_disagreement_vacuity),
            "curriculum": "40_20_20_10_10",
            "curriculum_distribution": train_ds.bucket_distribution,
            "real_thermal_samples_loaded": int(train_ds.real_samples_loaded),
            "real_item_fraction": float(train_ds.real_item_fraction),
            "real_dataset_breakdown": real_breakdown,
            "hit_uav_hot_neg_count": int(train_ds.hit_uav_hot_neg_count),
            "parallax_conflict_count": int(train_ds.parallax_conflict_count),
            "model_state_dict": {k: v.detach().cpu() for k, v in raw_final.state_dict().items()},
            "four_mode_eval": four_mode_eval,
        },
        final_ckpt_path,
    )

    onnx_meta: dict[str, Any] | None = None
    if cfg.export_onnx:
        export_fp16_flag = bool(cfg.export_fp16_onnx or is_v53)
        fp32_name = "xvs_thermal_v53_fp32.onnx" if is_v53 else "xvs_thermal_v45.onnx"
        fp16_p = (cfg.out_dir / "xvs_thermal_v53_fp16.onnx") if export_fp16_flag else None
        onnx_meta = export_xvs_thermal_onnx(
            raw_final,
            cfg.out_dir / fp32_name,
            eo_hw=eo_hw,
            lwir_hw=lwir_hw,
            export_fp16=export_fp16_flag,
            fp16_onnx_path=fp16_p,
            ort_threads=2,
        )

    init_loss: float | None = float(history[0]["train_total_loss"]) if history else None
    final_loss: float | None = float(history[-1]["train_total_loss"]) if history else None
    loss_finite = final_loss is not None and math.isfinite(final_loss)
    loss_decreased = bool(
        init_loss is not None
        and final_loss is not None
        and math.isfinite(init_loss)
        and math.isfinite(final_loss)
        and final_loss < init_loss
    )
    report: dict[str, Any] = {
        "schema": "xvs.v2_m2.g4_xvs_thermal_report/2" if is_v53 else "xvs.v2_m2.g4_xvs_thermal_report/1",
        "v53_mode": is_v53,
        "enable_disagreement_vacuity": bool(raw_final.enable_disagreement_vacuity),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_local": bool(cfg.smoke_local),
        "device": str(device),
        "bf16_enabled": use_bf16,
        "resumed_from_checkpoint": resumed,
        "dataset_download_status": dl_status,
        "real_thermal_samples_loaded": int(train_ds.real_samples_loaded),
        "real_item_fraction": float(train_ds.real_item_fraction),
        "real_dataset_breakdown": real_breakdown,
        "hit_uav_hot_neg_count": int(train_ds.hit_uav_hot_neg_count),
        "parallax_conflict_count": int(train_ds.parallax_conflict_count),
        "require_real_thermal": bool(cfg.require_real_thermal),
        "curriculum_distribution": train_ds.bucket_distribution,
        "selected_batch_size": selected_bs,
        "vram_tuner": tuner_report,
        "env_report": env_report,
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "history": history,
        "four_mode_eval": four_mode_eval,
        "vraim_easa_rlm_07_11_pass": bool(vraim_doc["all_requirements_pass"]) if vraim_doc is not None else False,
        "vraim_penultimate_dim": int(vraim_doc["penultimate_feature_dim"]) if vraim_doc is not None else 0,
        "final_checkpoint": str(final_ckpt_path),
        "onnx_export": onnx_meta,
        "pass": bool(
            loss_finite
            and four_mode_eval["four_mode_simplex_pass"]
            and (not cfg.run_vraim or (vraim_doc is not None and bool(vraim_doc["all_requirements_pass"])))
            and (not cfg.require_real_thermal or int(train_ds.real_samples_loaded) >= int(cfg.min_real_thermal_samples))
        ),
    }
    (cfg.out_dir / "xvs_thermal_g4_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _parse_args(argv: list[str] | None = None) -> XvsThermalG4Config:
    p = argparse.ArgumentParser(description="OpenXVS v1.1.1 XVS-ThermalNet + XVS-VRAIM Colab L4/G4 trainer")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--drive-ckpt-dir", type=Path, default=None)
    p.add_argument("--data-cache-dir", type=Path, default=None)
    p.add_argument("--smoke-local", action="store_true")
    p.add_argument("--v53-mode", action="store_true")
    p.add_argument("--export-fp16-onnx", action="store_true")
    p.add_argument("--parallax-aug-prob", type=float, default=0.35)
    p.add_argument("--download-datasets", action="store_true")
    p.add_argument("--require-real-thermal", action="store_true")
    p.add_argument("--min-real-thermal-samples", type=int, default=100)
    p.add_argument("--no-export-onnx", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=160)
    p.add_argument("--num-train-samples", type=int, default=512)
    p.add_argument("--skip-vraim", action="store_true")
    p.add_argument("--use-compile", action="store_true")
    p.add_argument("--seed", type=int, default=20261121)
    ns = p.parse_args(argv)
    return XvsThermalG4Config(
        out_dir=ns.out_dir,
        drive_ckpt_dir=ns.drive_ckpt_dir,
        data_cache_dir=ns.data_cache_dir,
        smoke_local=bool(ns.smoke_local),
        download_datasets=bool(ns.download_datasets),
        require_real_thermal=bool(ns.require_real_thermal),
        min_real_thermal_samples=int(ns.min_real_thermal_samples),
        export_onnx=not bool(ns.no_export_onnx),
        epochs=int(ns.epochs),
        steps_per_epoch=int(ns.steps_per_epoch),
        num_train_samples=int(ns.num_train_samples),
        run_vraim=not bool(ns.skip_vraim),
        use_compile=bool(ns.use_compile),
        seed=int(ns.seed),
        v53_mode=bool(ns.v53_mode),
        export_fp16_onnx=bool(ns.export_fp16_onnx or ns.v53_mode),
        parallax_aug_prob=float(ns.parallax_aug_prob),
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    report = run_xvs_thermal_g4_training(cfg)
    print(json.dumps({"pass": report["pass"], "four_mode_eval": report["four_mode_eval"]}, indent=2))


if __name__ == "__main__":
    main()
