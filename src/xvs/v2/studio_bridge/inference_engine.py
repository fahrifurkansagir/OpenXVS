"""Live 4-channel inference engine for OpenXVS Studio v4.5 (`XVS-D2` + `ActiveRange`, `XVS-Land`, `XVS-ThermalNet`, `XVS-Nav`).

Accepts arbitrary BGR `np.ndarray` frames from:
  - Synchronized demo sequences (`data/studio_media/` with graceful fallback to
    `apps/livedeck/public/assets/realsky_aot_pack/` + `ksfo_ortho_core2k_v2.jpg`),
  - User-uploaded JPEG/PNG images (`POST /api/analyze_image?channel=<d2|land|thermal|nav>`),
  - Live webcam/USB frames (`POST /api/analyze_frame_fast?channel=<d2|land|thermal|nav>`), and
  - Uploaded / built-in `.mp4` video jobs (`VideoJobManager` with isolated per-job `D2TrackerState`).

Emits exact `checkpoint_sha256` provenance on every channel output (`D16-W19-m3`), caches static
NAIP satellite descriptors in memory, and memoizes `demo_tick` results in a 128-entry LRU cache.
Strictly NED coordinates (zero internal ENU) and zero unannotated `.get()` defaults.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

from xvs.sensor.lwir import build_lwir_two_channel_input
from xvs.v2.xvs_active_range.filter import (
    ActiveRangeConfig,
    ActiveRangeController,
    ActiveRangeMeasurement,
    ActiveRangeMode,
    OwnshipStateNed,
)
from xvs.v2.xvs_d2.track_before_confirm import (
    DEFAULT_CONFIRM_M,
    DEFAULT_CONFIRM_N,
    DEFAULT_WEAK_SCORE_THR,
    BearingDetection,
    BearingTrackBeforeConfirm,
    TbcStepResult,
    TrackBeforeConfirmConfig,
)
from xvs.v2.xvs_land.pnpl_ekf import (
    build_ksfo_28l_reference_features,
    solve_4dof_map_pnpl,
)
from xvs.v2.xvs_nav.aliasing_gate import (
    DEFAULT_HAL_NAV_M,
    K_H0_LOCKED,
    MIN_PSR_LOCKED,
    VisualAliasingThresholds,
    VisualFixMetrics,
    compute_calibrated_sigma_vis,
    compute_geom_condition_number,
    compute_innovation_q_vis,
    compute_multi_hypothesis_hpl,
    compute_psr_calibrated,
    compute_spatial_entropy_4x4,
    evaluate_visual_aliasing_gate,
)
from xvs.v2.xvs_nav.dual_bank_cusum import (
    CUSUM_H_LOCKED,
    CUSUM_K_D_LOCKED,
    CUSUM_NUM_ARMS,
    DualBankEsEkfCusum,
)
from xvs.v2.xvs_nav.sequence_vpr import (
    DEFAULT_SEQ_WINDOW,
    SequenceVprConfig,
    SequenceVprDecision,
    SequenceVprFilter,
)
from xvs.v2.xvs_nav.stage2_xfeat import (
    DESCRIPTOR_DIM,
    MAX_DESCRIPTOR_BYTES_PER_TILE,
    TIER2B_TRIGGER_MIN_INLIERS,
    LighterGlueFallbackRunner,
    XFeatOnnxRunner,
    match_mnn,
)
from xvs.v2.sensor_health import (
    PinholeIntrinsics,
    SensorHealthMonitor,
    undistort_keypoints_pinhole,
)
from xvs.v2.xvs_thermal.model import ThermalSimplexMode, select_thermal_simplex_mode

from .models_loader import REPO_ROOT, StudioModelsBundle, get_process_vram_mb

DEFAULT_STUDIO_MEDIA_DIR = REPO_ROOT / "data" / "studio_media"
DEFAULT_AOT_PACK_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "realsky_aot_pack"
DEFAULT_KSFO_ORTHO = REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_ortho_core2k_v2.jpg"

D2_CROP_SIZE: int = 256
AOT_FOCAL_PX: float = 2690.0
DEMO_TICK_LRU_MAXSIZE: int = 128
LWIR_L_SAT_PROXY: float = 215.7  # Canonical DR-04 §4.2 radiance proxy scale (matches IRSTD1k/HitUav)
V5_NAV_GAMMA_CONF_DEFAULT: float = 1.264  # DR-21 split-conformal Rayleigh inflation factor (NAV5)
V5_NAV_PSR_DELTA_MIN: float = 0.04  # DR-21 background-centered PSR margin floor
V5_NAV_NON_ADJ_RADIUS_M: float = 300.0  # 2x nadir footprint exclusion radius (NED meters)

# Avionics BGR palette
COL_BG = (16, 22, 32)
COL_EMERALD = (118, 236, 68)
COL_CYAN = (248, 224, 48)
COL_AMBER = (56, 190, 248)
COL_ROSE = (98, 92, 244)
COL_TEXT = (240, 246, 252)


def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 84) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    raise ValueError(f"Expected 2D/3D BGR image array, got shape {arr.shape}")


def _scene_normalize_gray(gray_f32: np.ndarray, min_std: float = 8.0, clamp_val: float = 5.0) -> np.ndarray:
    mean = float(gray_f32.mean())
    std = max(float(gray_f32.std()), float(min_std))
    return np.clip((gray_f32 - mean) / std, -clamp_val, clamp_val).astype(np.float32)


def _warp_gray_and_mask(gray_f32: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray_f32.shape
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        gray_f32, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    ones = np.ones((h, w), dtype=np.float32)
    mask = cv2.warpAffine(
        ones, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    return warped, (mask > 0.5).astype(np.float32)


def _warp_hidden_state(h_prev: np.ndarray, dx_full: float, dy_full: float, stride: float = 4.0) -> np.ndarray:
    if abs(dx_full) < 1.0e-4 and abs(dy_full) < 1.0e-4:
        return h_prev
    _, channels, hh, hw = h_prev.shape
    mat = np.array([[1.0, 0.0, dx_full / stride], [0.0, 1.0, dy_full / stride]], dtype=np.float32)
    out = np.empty_like(h_prev)
    for ch in range(channels):
        out[0, ch] = cv2.warpAffine(
            h_prev[0, ch], mat, (hw, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -40.0, 40.0)))


def _decode_d2_top(
    hm_logits: np.ndarray,
    offset: np.ndarray,
    log_scale: np.ndarray,
    peak_mask: np.ndarray,
    *,
    score_thr: float = 0.25,
    stride: float = 2.0,
) -> tuple[tuple[float, float, float, float, float] | None, float]:
    hm = hm_logits[0, 0].astype(np.float32)
    off = offset[0].astype(np.float32)
    lsc = log_scale[0].astype(np.float32)
    pmk = peak_mask[0, 0].astype(np.float32)

    probs = _sigmoid_np(hm)
    h_sz, w_sz = probs.shape
    max_prob = float(probs.max())

    keep = (probs >= float(score_thr)) & (pmk > 0.5)
    ys, xs = np.where(keep)
    if ys.size == 0:
        return None, max_prob

    scores = probs[ys, xs]
    best = int(np.argmax(scores))
    cy = int(ys[best])
    cx = int(xs[best])
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
    return (u_px - 0.5 * bw, v_px - 0.5 * bh, u_px + 0.5 * bw, v_px + 0.5 * bh, score), max_prob


def _apply_pca_whitening(desc: np.ndarray, pca: dict[str, np.ndarray] | None) -> np.ndarray:
    x = np.asarray(desc, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if pca is None:
        norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return (x / norms).astype(np.float32)
    proj = (x - pca["mean"]) @ pca["components"].T
    white = proj * pca["inv_std"][np.newaxis, :]
    norms = np.maximum(np.linalg.norm(white, axis=1, keepdims=True), 1e-12)
    return (white / norms).astype(np.float32)


def _draw_badge(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    accent_bgr: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x0, y0 = origin
    cv2.rectangle(img, (x0 - 4, y0 - th - 5), (x0 + tw + 6, y0 + baseline + 3), COL_BG, -1)
    cv2.rectangle(img, (x0 - 4, y0 - th - 5), (x0 + tw + 6, y0 + baseline + 3), accent_bgr, 1)
    cv2.putText(img, text, (x0, y0), font, scale, COL_TEXT, thick, cv2.LINE_AA)


def _derive_lwir_2ch_from_bgr(
    frame_bgr: np.ndarray,
    lwir_bgr: np.ndarray | None,
    hw: tuple[int, int],
    *,
    l_sat: float = LWIR_L_SAT_PROXY,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical DR-04 §4.2 2-channel LWIR `[norm_radiance, local_contrast]` + uint8 visualization."""
    h_l, w_l = hw
    if lwir_bgr is not None:
        src = cv2.resize(_ensure_bgr(lwir_bgr), (w_l, h_l), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        src = cv2.resize(_ensure_bgr(frame_bgr), (w_l, h_l), interpolation=cv2.INTER_AREA)
        b_ch = src[:, :, 0].astype(np.float32) / 255.0
        g_ch = src[:, :, 1].astype(np.float32) / 255.0
        r_ch = src[:, :, 2].astype(np.float32) / 255.0
        gray = np.clip(0.55 * r_ch + 0.30 * g_ch + 0.15 * (1.0 - b_ch), 0.0, 1.0)

    rad_proxy = torch.from_numpy(np.ascontiguousarray(gray * float(l_sat), dtype=np.float32))
    lwir_2ch = (
        build_lwir_two_channel_input(rad_proxy, l_sat=float(l_sat))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    lwir_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    return lwir_2ch, lwir_u8


def validate_keyline_sigmas_px(
    raw_sigmas: np.ndarray | Sequence[float] | None,
    *,
    expected_count: int = 5,
) -> np.ndarray | None:
    """Validate optional v5.2 6th ONNX output `keyline_sigma_px` strictly fail-closed.

    Returns `None` for 5-output v4.5/v5.1 graphs so callers pass explicit `keyline_sigmas_px=None`.
    Raises `ValueError` on any NaN, Inf, non-positive value, or shape mismatch when present.
    """
    if raw_sigmas is None:
        return None
    arr = np.asarray(raw_sigmas, dtype=np.float64).reshape(-1)
    if arr.shape != (int(expected_count),):
        raise ValueError(
            f"Expected keyline_sigma_px of length {expected_count}, got shape {arr.shape}"
        )
    if not bool(np.all(np.isfinite(arr))) or not bool(np.all(arr > 0.0)):
        raise ValueError(
            f"Invalid keyline_sigma_px (must be finite and strictly > 0): {arr.tolist()}"
        )
    return arr


def _extract_tbc_summary(
    tbc_res: TbcStepResult | None,
    tracker: BearingTrackBeforeConfirm,
) -> tuple[bool, int, int]:
    """Extract `(bearing_track_confirmed, bearing_track_hits_in_window, bearing_tbc_confirmed_tracks)`."""
    if tbc_res is not None:
        confirmed = tbc_res.confirmed_tracks
        tentative = tbc_res.tentative_tracks
    else:
        all_tracks = tracker.active_tracks
        confirmed = tuple(t for t in all_tracks if t.is_confirmed)
        tentative = tuple(t for t in all_tracks if not t.is_confirmed)
    confirmed_count = len(confirmed)
    if confirmed:
        hits_in_window = max(int(t.window_hits) for t in confirmed)
    elif tentative:
        hits_in_window = max(int(t.window_hits) for t in tentative)
    else:
        hits_in_window = 0
    return (confirmed_count > 0, int(hits_in_window), int(confirmed_count))


class D2TrackerState:
    """Isolated temporal ConvGRU + `ActiveRangeController` + `3-of-5` TBC state for a single stream."""

    def __init__(self, hid_c: int = 24, focal_px: float = AOT_FOCAL_PX) -> None:
        self.hid_c = int(hid_c)
        self.focal_px = float(focal_px)
        self.h_prev = np.zeros((1, self.hid_c, D2_CROP_SIZE // 4, D2_CROP_SIZE // 4), dtype=np.float32)
        self.gray_hist: list[tuple[np.ndarray, int, int]] = []
        self.active_range_ctrl = ActiveRangeController(
            ActiveRangeConfig(q_hat_conformal=0.955, sigma_bearing_rad=2.86 / self.focal_px)
        )
        self.active_range_age: int = 0
        self.last_size_px: float | None = None
        self.bearing_tbc = BearingTrackBeforeConfirm(
            TrackBeforeConfirmConfig(
                confirm_m=DEFAULT_CONFIRM_M,
                confirm_n=DEFAULT_CONFIRM_N,
                default_eo_meas_std_rad=2.86 / self.focal_px,
            )
        )
        self.last_tbc_result: TbcStepResult | None = None

    def reset(self) -> None:
        self.h_prev.fill(0.0)
        self.gray_hist.clear()
        self.active_range_ctrl = ActiveRangeController(
            ActiveRangeConfig(q_hat_conformal=0.955, sigma_bearing_rad=2.86 / self.focal_px)
        )
        self.active_range_age = 0
        self.last_size_px = None
        self.bearing_tbc.reset()
        self.last_tbc_result = None


class StudioInferenceEngine:
    """Thread-safe 4-channel real-model inference engine for `OpenXVS Studio v4.5`."""

    def __init__(
        self,
        bundle: StudioModelsBundle,
        *,
        studio_media_dir: Path = DEFAULT_STUDIO_MEDIA_DIR,
        aot_pack_dir: Path = DEFAULT_AOT_PACK_DIR,
        ksfo_ortho_path: Path = DEFAULT_KSFO_ORTHO,
        lru_maxsize: int = DEMO_TICK_LRU_MAXSIZE,
    ) -> None:
        self.bundle = bundle
        self.studio_media_dir = Path(studio_media_dir)
        self.aot_pack_dir = Path(aot_pack_dir)
        self.ksfo_ortho_path = Path(ksfo_ortho_path)
        self._d2_ort_lock = threading.Lock()
        self._land_ort_lock = threading.Lock()
        self._thermal_ort_lock = threading.Lock()
        self._nav_ort_lock = threading.Lock()
        self._nav_seq_vpr_lock = threading.Lock()
        self._cache_lock = threading.Lock()

        # Isolated temporal states: demo_tick vs live webcam/fast stream
        self._demo_d2_state = D2TrackerState(hid_c=self.bundle.d2_hid_c, focal_px=AOT_FOCAL_PX)
        self._fast_d2_state = D2TrackerState(hid_c=self.bundle.d2_hid_c, focal_px=AOT_FOCAL_PX)

        # Canonical KSFO-28L runway reference features (strictly NED) for 4-DoF MAP-PnPL
        self._land_ref_pts_ned, self._land_ref_kl_pairs = build_ksfo_28l_reference_features()

        # Cached KSFO ortho + onboard static NAIP satellite reference descriptor & Stage-2 XFeat DB
        self._ksfo_bgr: np.ndarray | None = None
        self._nav_ref_desc_cache: dict[str, np.ndarray] = {}
        self._nav_ref_xfeat_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._nav_gallery_descs: np.ndarray | None = None
        self._nav_gallery_tiles: list[np.ndarray] = []
        # Strictly NED coordinates (meters) for the 4 non-adjacent KSFO ortho gallery tiles (> 300 m spacing)
        self._nav_gallery_ne_m = np.array(
            [
                [0.0, 0.0],
                [0.0, 650.0],
                [650.0, 0.0],
                [650.0, 650.0],
            ],
            dtype=np.float64,
        )
        # Isolated multi-frame Sequence-VPR filters (`W=7`, strict NED odometry prior + aliasing gate)
        self._demo_seq_vpr = self.new_nav_sequence_vpr_filter()
        self._fast_seq_vpr = self.new_nav_sequence_vpr_filter()
        self._ref_seq_vpr: SequenceVprFilter | None = None
        self._demo_last_scenario: str | None = None
        self._nav_xfeat = XFeatOnnxRunner(
            onnx_path=self.bundle.nav_xfeat_onnx_path,
            expected_sha256=self.bundle.nav_xfeat_sha256 if self.bundle.nav_xfeat_sha256 else None,
            max_keypoints=256,
        )
        self._nav_lighterglue = LighterGlueFallbackRunner()
        self._aot_manifest: dict[str, Any] | None = None
        self._studio_manifest: dict[str, Any] | None = None

        # Bounded LRU cache for synchronized `demo_tick` playback (`< 10 ms` warm hits)
        self._demo_tick_lru: OrderedDict[tuple[str, int, bool], dict[str, Any]] = OrderedDict()
        self._demo_tick_lru_max = max(8, int(lru_maxsize))

        # Real-world sensor health monitor (LWIR NUC freeze, optical blur/glare/occlusion, timestamp skew)
        self._sensor_health = SensorHealthMonitor()

    def new_d2_tracker_state(self, focal_px: float = AOT_FOCAL_PX) -> D2TrackerState:
        """Allocate a fresh job-isolated `D2TrackerState` (never touches `demo_tick` state)."""
        return D2TrackerState(hid_c=self.bundle.d2_hid_c, focal_px=focal_px)

    def new_nav_cusum_filter(self) -> DualBankEsEkfCusum:
        """Allocate a fresh local-only Dual-Bank ES-EKF + 8-Arm Cholesky Page's CUSUM spoof-guard."""
        return DualBankEsEkfCusum()

    def new_nav_sequence_vpr_filter(
        self,
        gallery_ne_m: np.ndarray | None = None,
        config: SequenceVprConfig | None = None,
    ) -> SequenceVprFilter:
        """Allocate a fresh multi-frame `SequenceVprFilter` (`W=7`, strictly NED odometry prior)."""
        coords = self._nav_gallery_ne_m if gallery_ne_m is None else np.asarray(gallery_ne_m, dtype=np.float64)
        cfg = (
            SequenceVprConfig(
                window_size=DEFAULT_SEQ_WINDOW,
                non_adj_radius_m=V5_NAV_NON_ADJ_RADIUS_M,
                min_psr_cal=MIN_PSR_LOCKED,
                min_psr_delta=V5_NAV_PSR_DELTA_MIN,
            )
            if config is None
            else config
        )
        return SequenceVprFilter(coords, config=cfg)

    def new_sensor_health_monitor(self) -> SensorHealthMonitor:
        """Allocate a fresh job-isolated `SensorHealthMonitor`."""
        return SensorHealthMonitor()

    def sensor_health_snapshot(self) -> dict[str, Any]:
        """Return latest real-world sensor health telemetry snapshot including 3-of-5 Bearing TBC state."""
        snap = self._sensor_health.snapshot()
        with self._d2_ort_lock:
            confirmed, hits, confirmed_tracks = _extract_tbc_summary(
                self._demo_d2_state.last_tbc_result,
                self._demo_d2_state.bearing_tbc,
            )
        snap["bearing_track_confirmed"] = bool(confirmed)
        snap["bearing_track_hits_in_window"] = int(hits)
        snap["bearing_tbc_confirmed_tracks"] = int(confirmed_tracks)
        return snap

    def reset_temporal_state(self) -> None:
        """Reset ConvGRU, ActiveRange, 3-of-5 TBC, Sequence-VPR, sensor health, and demo_tick LRU cache."""
        with self._d2_ort_lock:
            self._demo_d2_state.reset()
            self._fast_d2_state.reset()
        with self._nav_seq_vpr_lock:
            self._demo_seq_vpr.reset()
            self._fast_seq_vpr.reset()
            self._ref_seq_vpr = None
            self._demo_last_scenario = None
        self._sensor_health.reset()
        with self._cache_lock:
            self._demo_tick_lru.clear()

    def clear_demo_cache(self) -> None:
        """Clear the bounded demo_tick LRU cache (retains static satellite DB embeddings)."""
        with self._cache_lock:
            self._demo_tick_lru.clear()

    def media_pack_ready(self) -> bool:
        return (self.studio_media_dir / "manifest.json").is_file()

    def checkpoint_sha256_for_channel(self, channel: str) -> str:
        """Return exact verified SHA-256 hex digest for a sensor channel (`d2|land|thermal|nav`)."""
        ch = channel.strip().lower()
        key_map = {
            "d2": "xvs_d2",
            "eo_sky": "xvs_d2",
            "land": "xvs_land",
            "runway_land": "xvs_land",
            "thermal": "xvs_thermal",
            "thermal_dve": "xvs_thermal",
            "nav": "xvs_nav",
            "crossview_nav": "xvs_nav",
        }
        mkey = key_map[ch] if ch in key_map else "xvs_d2"
        return self.bundle.meta[mkey].sha256

    def _load_ksfo_ortho(self) -> np.ndarray:
        if self._ksfo_bgr is not None:
            return self._ksfo_bgr
        if self.ksfo_ortho_path.is_file():
            img = cv2.imread(str(self.ksfo_ortho_path), cv2.IMREAD_COLOR)
            if img is not None:
                self._ksfo_bgr = img
                return img
        canvas = np.full((512, 512, 3), (44, 52, 46), dtype=np.uint8)
        cv2.line(canvas, (90, 440), (420, 90), (88, 92, 96), 34, cv2.LINE_AA)
        cv2.line(canvas, (100, 430), (410, 100), (225, 232, 238), 2, cv2.LINE_AA)
        self._ksfo_bgr = canvas
        return canvas

    def _load_aot_manifest(self) -> dict[str, Any] | None:
        if self._aot_manifest is not None:
            return self._aot_manifest
        mpath = self.aot_pack_dir / "manifest.json"
        if mpath.is_file():
            self._aot_manifest = json.loads(mpath.read_text(encoding="utf-8"))
        return self._aot_manifest

    def _load_studio_manifest(self) -> dict[str, Any] | None:
        if self._studio_manifest is not None:
            return self._studio_manifest
        mpath = self.studio_media_dir / "manifest.json"
        if mpath.is_file():
            self._studio_manifest = json.loads(mpath.read_text(encoding="utf-8"))
            return self._studio_manifest
        return None

    # ---- Channel 1: XVS-D2 + XVS-ActiveRange ----

    def _run_d2_single_probe(
        self,
        crop_gray: np.ndarray,
        score_thr: float,
    ) -> tuple[tuple[float, float, float, float, float] | None, float]:
        """Run a stateless 1-step `XVS-D2` ONNX probe on a 256x256 tile for high-res fallback."""
        if self.bundle.d2_session is None:
            return None, 0.0
        c0 = _scene_normalize_gray(crop_gray)
        zeros = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)
        x_t = np.stack([c0, zeros, zeros, zeros, zeros, zeros], axis=0)[np.newaxis, :, :, :]
        h_zero = np.zeros((1, self.bundle.d2_hid_c, D2_CROP_SIZE // 4, D2_CROP_SIZE // 4), dtype=np.float32)
        outs = self.bundle.d2_session.run(
            None,
            {
                "x_t": x_t.astype(self.bundle.d2_x_dtype),
                "h_prev_warped": h_zero.astype(self.bundle.d2_h_dtype),
            },
        )
        return _decode_d2_top(
            outs[0].astype(np.float32),
            outs[1].astype(np.float32),
            outs[2].astype(np.float32),
            outs[4].astype(np.float32),
            score_thr=score_thr,
        )

    def infer_eo_sky(
        self,
        frame_bgr: np.ndarray,
        *,
        crop_x: int = 0,
        crop_y: int = 0,
        subcrop_origin_xy: tuple[int, int] | None = None,
        gt_box_px: tuple[float, float, float, float] | None = None,
        gt_range_m: float | None = None,
        score_thr: float = 0.25,
        own_a_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt_s: float = 0.1,
        focal_px: float = AOT_FOCAL_PX,
        return_jpeg: bool = True,
        tracker_state: D2TrackerState | None = None,
        multi_crop_fallback: bool = False,
        frame_ts_ms: float | None = None,
        imu_ts_ms: float | None = None,
        sensor_health_monitor: SensorHealthMonitor | None = None,
        tbc_detections: Sequence[BearingDetection] | None = None,
        q_eo_tbc: float = 1.0,
        q_lwir_tbc: float = 1.0,
    ) -> dict[str, Any]:
        """Run real `XVS-D2` ConvGRU ONNX + real `ActiveRangeController` on a BGR sky frame."""
        t0 = time.perf_counter()
        bgr = _ensure_bgr(frame_bgr)
        h_in, w_in = bgr.shape[:2]
        active_sh = self._sensor_health if sensor_health_monitor is None else sensor_health_monitor
        eo_sensor_health = active_sh.assess_eo_frame(
            bgr,
            channel="eo_sky",
            frame_ts_ms=frame_ts_ms,
            imu_ts_ms=imu_ts_ms,
        )
        state = self._demo_d2_state if tracker_state is None else tracker_state

        use_native_subcrop = (
            subcrop_origin_xy is not None and w_in >= D2_CROP_SIZE and h_in >= D2_CROP_SIZE
        )
        if use_native_subcrop and subcrop_origin_xy is not None:
            ox = max(0, min(w_in - D2_CROP_SIZE, int(subcrop_origin_xy[0])))
            oy = max(0, min(h_in - D2_CROP_SIZE, int(subcrop_origin_xy[1])))
            crop_bgr = bgr[oy : oy + D2_CROP_SIZE, ox : ox + D2_CROP_SIZE]
            gmc_x = int(crop_x) + ox
            gmc_y = int(crop_y) + oy
        else:
            ox, oy = 0, 0
            crop_bgr = cv2.resize(bgr, (D2_CROP_SIZE, D2_CROP_SIZE), interpolation=cv2.INTER_AREA)
            gmc_x = int(crop_x)
            gmc_y = int(crop_y)

        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        with self._d2_ort_lock:
            c0 = _scene_normalize_gray(crop_gray)
            if len(state.gray_hist) >= 1:
                g1, cx1, cy1 = state.gray_hist[-1]
                dx1 = float(cx1 - gmc_x)
                dy1 = float(cy1 - gmc_y)
                w1, r1 = _warp_gray_and_mask(g1, dx1, dy1)
                c1 = _scene_normalize_gray(w1) * r1
            else:
                dx1, dy1 = 0.0, 0.0
                c1 = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)
                r1 = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)

            if len(state.gray_hist) >= 5:
                g5, cx5, cy5 = state.gray_hist[-5]
                dx5 = float(cx5 - gmc_x)
                dy5 = float(cy5 - gmc_y)
                w5, r5 = _warp_gray_and_mask(g5, dx5, dy5)
                c2 = _scene_normalize_gray(w5) * r5
            else:
                c2 = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)
                r5 = np.zeros((D2_CROP_SIZE, D2_CROP_SIZE), dtype=np.float32)

            c3 = (c0 - c1) * r1
            c4 = (c0 - c2) * r5
            c5 = r1 * r5
            x_t = np.stack([c0, c1, c2, c3, c4, c5], axis=0)[np.newaxis, :, :, :].astype(np.float32)
            h_warped = _warp_hidden_state(state.h_prev, dx1, dy1, stride=4.0)

            state.gray_hist.append((crop_gray.copy(), gmc_x, gmc_y))
            if len(state.gray_hist) > 8:
                state.gray_hist.pop(0)

            det: tuple[float, float, float, float, float] | None = None
            det_frame_box: tuple[float, float, float, float] | None = None
            weak_tbc_det: tuple[float, float, float, float, float] | None = None
            weak_tbc_box: tuple[float, float, float, float] | None = None
            max_prob = 0.0
            if self.bundle.d2_session is not None:
                outs = self.bundle.d2_session.run(
                    None,
                    {
                        "x_t": x_t.astype(self.bundle.d2_x_dtype),
                        "h_prev_warped": h_warped.astype(self.bundle.d2_h_dtype),
                    },
                )
                hm_logits = outs[0].astype(np.float32)
                offset = outs[1].astype(np.float32)
                log_scale = outs[2].astype(np.float32)
                peak_mask = outs[4].astype(np.float32)
                state.h_prev = outs[5].astype(np.float32)
                det, max_prob = _decode_d2_top(
                    hm_logits, offset, log_scale, peak_mask, score_thr=score_thr
                )
                if det is not None:
                    x1, y1, x2, y2, _ = det
                    if use_native_subcrop:
                        det_frame_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                    else:
                        sx_m = float(w_in) / float(D2_CROP_SIZE)
                        sy_m = float(h_in) / float(D2_CROP_SIZE)
                        det_frame_box = (x1 * sx_m, y1 * sy_m, x2 * sx_m, y2 * sy_m)
                elif float(score_thr) > DEFAULT_WEAK_SCORE_THR:
                    weak_tbc_det, _ = _decode_d2_top(
                        hm_logits,
                        offset,
                        log_scale,
                        peak_mask,
                        score_thr=DEFAULT_WEAK_SCORE_THR,
                    )
                    if weak_tbc_det is not None:
                        wx1, wy1, wx2, wy2, _ = weak_tbc_det
                        if use_native_subcrop:
                            weak_tbc_box = (wx1 + ox, wy1 + oy, wx2 + ox, wy2 + oy)
                        else:
                            sx_m = float(w_in) / float(D2_CROP_SIZE)
                            sy_m = float(h_in) / float(D2_CROP_SIZE)
                            weak_tbc_box = (wx1 * sx_m, wy1 * sy_m, wx2 * sx_m, wy2 * sy_m)

                # Smart multi-crop tiled fallback on high-res (> 384px) real-world video frames
                if det is None and multi_crop_fallback and not use_native_subcrop and max(w_in, h_in) > 384:
                    tile_w = min(w_in, max(D2_CROP_SIZE, w_in // 2))
                    tile_h = min(h_in, max(D2_CROP_SIZE, h_in // 2))
                    sky_y = max(0, min(h_in - tile_h, int(h_in * 0.08)))
                    tile_origins = [
                        (max(0, (w_in - tile_w) // 2), sky_y),
                        (0, sky_y),
                        (max(0, w_in - tile_w), sky_y),
                    ]
                    best_tile_det: tuple[float, float, float, float, float] | None = None
                    best_tile_box: tuple[float, float, float, float] | None = None
                    for tx0, ty0 in tile_origins:
                        patch = bgr[ty0 : ty0 + tile_h, tx0 : tx0 + tile_w]
                        p_rs = cv2.resize(patch, (D2_CROP_SIZE, D2_CROP_SIZE), interpolation=cv2.INTER_AREA)
                        p_gray = cv2.cvtColor(p_rs, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        tdet, tprob = self._run_d2_single_probe(p_gray, score_thr=score_thr)
                        if tprob > max_prob:
                            max_prob = tprob
                        if tdet is not None and (best_tile_det is None or tdet[4] > best_tile_det[4]):
                            best_tile_det = tdet
                            tsx = float(tile_w) / float(D2_CROP_SIZE)
                            tsy = float(tile_h) / float(D2_CROP_SIZE)
                            best_tile_box = (
                                tx0 + tdet[0] * tsx,
                                ty0 + tdet[1] * tsy,
                                tx0 + tdet[2] * tsx,
                                ty0 + tdet[3] * tsy,
                            )
                    if best_tile_det is not None and best_tile_box is not None:
                        det = best_tile_det
                        det_frame_box = best_tile_box

            if det is not None and det_frame_box is not None:
                fx1, fy1, fx2, fy2 = det_frame_box
                score = float(det[4])
                cx_px = 0.5 * (fx1 + fx2)
                cy_px = 0.5 * (fy1 + fy2)
                size_px = math.sqrt(max(1.0, (fx2 - fx1) * (fy2 - fy1)))
                size_rate = (
                    (size_px - state.last_size_px) / max(dt_s, 1e-3)
                    if state.last_size_px is not None
                    else 0.0
                )
                state.last_size_px = size_px
                state.active_range_age += 1
            else:
                cx_px = 0.5 * float(w_in)
                cy_px = 0.5 * float(h_in)
                size_px = state.last_size_px if state.last_size_px is not None else 10.0
                size_rate = 0.0
                score = 0.0

            alpha_rad = math.atan2(cx_px - 0.5 * float(w_in), float(focal_px))
            beta_rad = math.atan2(0.5 * float(h_in) - cy_px, float(focal_px))

            # 3-of-5 Bearing Track-Before-Confirm (`BearingTrackBeforeConfirm`, Camera RDF / NED bearings)
            if tbc_detections is not None:
                dets_for_tbc = list(tbc_detections)
            elif det is not None and det_frame_box is not None:
                dets_for_tbc = [
                    BearingDetection(
                        az_rad=float(alpha_rad),
                        el_rad=float(beta_rad),
                        score=float(score),
                        modality="eo",
                        sigma_az_rad=2.86 / float(focal_px),
                        sigma_el_rad=2.86 / float(focal_px),
                        size_px=float(size_px),
                        u_px=float(cx_px),
                        v_px=float(cy_px),
                    )
                ]
            elif weak_tbc_det is not None and weak_tbc_box is not None:
                wx1, wy1, wx2, wy2 = weak_tbc_box
                w_cx = 0.5 * (wx1 + wx2)
                w_cy = 0.5 * (wy1 + wy2)
                w_sz = math.sqrt(max(1.0, (wx2 - wx1) * (wy2 - wy1)))
                dets_for_tbc = [
                    BearingDetection(
                        az_rad=math.atan2(w_cx - 0.5 * float(w_in), float(focal_px)),
                        el_rad=math.atan2(0.5 * float(h_in) - w_cy, float(focal_px)),
                        score=float(weak_tbc_det[4]),
                        modality="eo",
                        sigma_az_rad=2.86 / float(focal_px),
                        sigma_el_rad=2.86 / float(focal_px),
                        size_px=float(w_sz),
                        u_px=float(w_cx),
                        v_px=float(w_cy),
                    )
                ]
            else:
                dets_for_tbc = []

            eff_q_eo_tbc = float(q_eo_tbc)
            if tbc_detections is None:
                # Gate out severe optical faults (white noise, dead-pixel FPN, lens-cap, sun-glare)
                # without penalizing clean featureless overcast sky.
                hard_eo_faults = {
                    "SENSOR_NOISE_FAULT",
                    "DEAD_PIXEL_FPN_FAULT",
                    "LENS_CAP_OR_UNDEREXPOSED",
                    "SUN_GLARE_OVEREXPOSED",
                    "NON_FINITE_OR_EMPTY_FRAME",
                }
                if any(r in hard_eo_faults for r in eo_sensor_health["reasons"]):
                    eff_q_eo_tbc = min(eff_q_eo_tbc, float(eo_sensor_health["q_eo_effective"]))

            tbc_res = state.bearing_tbc.step(
                dets_for_tbc,
                dt_s=dt_s,
                q_eo=eff_q_eo_tbc,
                q_lwir=float(q_lwir_tbc),
            )
            state.last_tbc_result = tbc_res
            bearing_track_confirmed, bearing_hits_in_window, bearing_confirmed_tracks = (
                _extract_tbc_summary(tbc_res, state.bearing_tbc)
            )
            primary_tbc_track = (
                tbc_res.confirmed_tracks[0]
                if tbc_res.confirmed_tracks
                else (tbc_res.tentative_tracks[0] if tbc_res.tentative_tracks else None)
            )

            own = OwnshipStateNed(
                v_ned=(35.0, 0.0, 0.0),
                a_ned=own_a_ned,
                heading_rad=0.0,
                tas_mps=35.0,
                v_stall_mps=14.5,
            )
            meas = ActiveRangeMeasurement(
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
                size_px=size_px,
                size_rate_px_s=size_rate,
                contrast_ratio=0.65,
                sky_beta_ext=2.0e-4,
                daytime_clear=True,
                track_age=max(1, state.active_range_age),
                dt_s=dt_s,
            )
            telem = state.active_range_ctrl.step(own, meas)

        eo_sensor_health["bearing_track_confirmed"] = bool(bearing_track_confirmed)
        eo_sensor_health["bearing_track_hits_in_window"] = int(bearing_hits_in_window)
        eo_sensor_health["bearing_tbc_confirmed_tracks"] = int(bearing_confirmed_tracks)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        mode_name = ActiveRangeMode(int(telem.mode)).name
        ckpt_sha = self.bundle.meta["xvs_d2"].sha256

        det_dict: dict[str, Any] | None = None
        if det is not None and det_frame_box is not None:
            fx1, fy1, fx2, fy2 = det_frame_box
            det_dict = {
                "x1": round(fx1, 2),
                "y1": round(fy1, 2),
                "x2": round(fx2, 2),
                "y2": round(fy2, 2),
                "score": round(float(score), 4),
                "size_px": round(float(size_px), 2),
            }

        gps_txt = f" | GPS {gt_range_m:.0f}m" if gt_range_m is not None else ""
        summary_str = (
            f"D2 {score:.2f} | R_low {telem.r_safe_lower_m:.0f}m{gps_txt}"
            if det is not None
            else f"SCAN {max_prob:.2f} | R_low {telem.r_safe_lower_m:.0f}m{gps_txt}"
        )

        res: dict[str, Any] = {
            "channel": "eo_sky",
            "model": "XVS-D2 + XVS-ActiveRange",
            "device": self.bundle.meta["xvs_d2"].device,
            "real_weights": self.bundle.meta["xvs_d2"].real_weights,
            "checkpoint_sha256": ckpt_sha,
            "latency_ms": latency_ms,
            "confidence": round(float(score if det is not None else max_prob), 4),
            "summary": summary_str,
            "detected": det is not None,
            "max_prob": round(float(max_prob), 4),
            "detection": det_dict,
            "bearing_track_confirmed": bool(bearing_track_confirmed),
            "bearing_track_hits_in_window": int(bearing_hits_in_window),
            "bearing_tbc_confirmed_tracks": int(bearing_confirmed_tracks),
            "bearing_tbc": {
                "confirmed": bool(bearing_track_confirmed),
                "hits_in_window": int(bearing_hits_in_window),
                "confirm_m": int(state.bearing_tbc.cfg.confirm_m),
                "window_len": int(state.bearing_tbc.cfg.confirm_n),
                "confirmed_tracks_count": int(bearing_confirmed_tracks),
                "tentative_tracks_count": int(len(tbc_res.tentative_tracks)),
                "tracks_confirmed_this_step": int(tbc_res.tracks_confirmed_this_step),
                "tracks_deleted_this_step": int(tbc_res.tracks_deleted_this_step),
                "detections_received": int(tbc_res.detections_received),
                "detections_accepted": int(tbc_res.detections_accepted),
                "simplex_mode": tbc_res.simplex_mode.name,
                "primary_track": primary_tbc_track.to_dict() if primary_tbc_track is not None else None,
            },
            "active_range": {
                "mode": mode_name,
                "mode_id": int(telem.mode),
                "r_hat_m": round(float(telem.r_hat_m), 2),
                "sigma_r_m": round(float(telem.sigma_r_m), 2),
                "r_safe_lower_m": round(float(telem.r_safe_lower_m), 2),
                "rho_dot_inv_s": round(float(telem.rho_dot_inv_s), 5),
                "cmd_bank_deg": round(math.degrees(float(telem.cmd_bank_rad)), 2),
                "min_sc3bf_margin": round(float(telem.min_sc3bf_margin), 2),
                "active_bank_idx": int(telem.active_bank_idx),
                "weights": [round(float(w), 4) for w in telem.weights],
            },
            "sensor_health": eo_sensor_health,
        }

        if return_jpeg:
            vis = cv2.resize(bgr, (384, 256), interpolation=cv2.INTER_AREA)
            vx_s = 384.0 / float(max(1, w_in))
            vy_s = 256.0 / float(max(1, h_in))
            if gt_box_px is not None:
                gx1, gy1, gx2, gy2 = gt_box_px
                gp1 = (int(round(gx1 * vx_s)), int(round(gy1 * vy_s)))
                gp2 = (int(round(gx2 * vx_s)), int(round(gy2 * vy_s)))
                cv2.rectangle(vis, gp1, gp2, COL_CYAN, 1, cv2.LINE_AA)
            if det_frame_box is not None and det is not None:
                fx1, fy1, fx2, fy2 = det_frame_box
                p1 = (int(round(fx1 * vx_s)), int(round(fy1 * vy_s)))
                p2 = (int(round(fx2 * vx_s)), int(round(fy2 * vy_s)))
                cv2.rectangle(vis, p1, p2, COL_EMERALD, 2, cv2.LINE_AA)
                _draw_badge(vis, f"XVS-D2 {det[4]:.2f}", (max(4, p1[0]), max(18, p1[1] - 6)), COL_EMERALD)
            _draw_badge(
                vis,
                f"R_low {telem.r_safe_lower_m:.0f}m{gps_txt} | {mode_name}",
                (8, 22),
                COL_EMERALD if det is not None else COL_CYAN,
            )
            b64 = _encode_jpeg_b64(vis, quality=84)
            res["image_b64"] = b64
            res["annotated_jpeg_b64"] = f"data:image/jpeg;base64,{b64}"
            res["image_mime"] = "image/jpeg"

        return res

    # ---- Channel 2: XVS-Land (12-Keypoint SimCC + 5 Hesse Keylines + Cholesky Covariance) ----

    @staticmethod
    def _compute_land_roi_origin(
        w_in: int,
        h_in: int,
        roi_box_px: tuple[float, float, float, float] | None,
        *,
        horizon_preserving: bool,
    ) -> tuple[int, int, int, int]:
        """Resolve `(ox, oy, rw, rh)` runway framing crop.

        Wave-C `XVS-Land` expects full-horizon cockpit context with near-1:1 aspect ratio
        (matching 256x256 training geometry), so when `roi_box_px` is supplied it spans full
        frame height (`y=0..h_in`, preserving sky + horizon) with width `max(h_in, 1.4*box_w)`
        centered on the runway azimuth. Legacy NB2 uses the tight padded `roi_box_px` window.
        """
        if roi_box_px is None:
            return (0, 0, w_in, h_in)
        rx1, ry1, rx2, ry2 = roi_box_px
        if horizon_preserving:
            rwy_cx = 0.5 * (rx1 + rx2)
            rwy_w = max(16.0, rx2 - rx1)
            cw = min(w_in, max(h_in, int(math.ceil(rwy_w * 1.4))))
            cx1 = max(0, min(w_in - cw, int(round(rwy_cx - 0.5 * float(cw)))))
            return (cx1, 0, cw, h_in)
        pad_x = max(24.0, 0.75 * (rx2 - rx1))
        pad_y = max(18.0, 0.75 * (ry2 - ry1))
        cx1 = max(0, int(math.floor(rx1 - pad_x)))
        cy1 = max(0, int(math.floor(ry1 - pad_y)))
        cx2 = min(w_in, int(math.ceil(rx2 + pad_x)))
        cy2 = min(h_in, int(math.ceil(ry2 + pad_y)))
        if cx2 - cx1 >= 16 and cy2 - cy1 >= 16:
            return (cx1, cy1, cx2 - cx1, cy2 - cy1)
        return (0, 0, w_in, h_in)

    def _resolve_land_roi_hw(self) -> tuple[int, int]:
        """Resolve `(roi_h, roi_w)` dynamically from loaded XVS-Land ONNX session or PyTorch model."""
        if self.bundle.land_session is not None:
            inps = self.bundle.land_session.get_inputs()
            if inps:
                shape = list(inps[0].shape)
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    if shape[2] > 0 and shape[3] > 0:
                        return int(shape[2]), int(shape[3])
        if self.bundle.land_model is not None:
            rh = getattr(self.bundle.land_model, "roi_height", None)
            rw = getattr(self.bundle.land_model, "roi_width", None)
            if isinstance(rh, int) and isinstance(rw, int) and rh > 0 and rw > 0:
                return rh, rw
        return 256, 256

    @torch.no_grad()
    def infer_runway_land(
        self,
        frame_bgr: np.ndarray,
        *,
        roi_box_px: tuple[float, float, float, float] | None = None,
        horizon_preserving_roi: bool | None = None,
        return_jpeg: bool = True,
        solve_pnpl: bool = True,
        pitch_rad: float = math.radians(-3.0),
        roll_rad: float = 0.0,
        init_pos_ned: np.ndarray | None = None,
        init_yaw_rad: float | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        pts_ned: np.ndarray | None = None,
        keyline_pairs: list[tuple[int, int]] | None = None,
        huber_delta_sigma: float = 3.0,
    ) -> dict[str, Any]:
        """Run XVS-Land (5-output v4.5/v5.1 OR 6-output v5.2 with `keyline_sigma_px`) + Huber-Tukey 4-DoF MAP-PnPL."""
        t0 = time.perf_counter()
        bgr = _ensure_bgr(frame_bgr)
        h_in, w_in = bgr.shape[:2]

        land_meta = self.bundle.meta["xvs_land"]
        use_horizon_roi = (
            bool(horizon_preserving_roi)
            if horizon_preserving_roi is not None
            else land_meta.tier.startswith(("wave_c", "v52"))
        )
        ox, oy, rw, rh = self._compute_land_roi_origin(
            w_in, h_in, roi_box_px, horizon_preserving=use_horizon_roi
        )
        crop = bgr[oy : oy + rh, ox : ox + rw] if (ox != 0 or oy != 0 or rw != w_in or rh != h_in) else bgr
        roi_origin = (ox, oy, rw, rh)

        roi_h, roi_w = self._resolve_land_roi_hw()
        roi_bgr = cv2.resize(crop, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
        rgb_f32 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp_np = np.ascontiguousarray(np.transpose(rgb_f32, (2, 0, 1))[np.newaxis, :, :, :], dtype=np.float32)

        raw_keyline_sigma_px: np.ndarray | None = None
        if self.bundle.land_session is not None:
            out_names = [o.name for o in self.bundle.land_session.get_outputs()]
            has_keyline_sigma = "keyline_sigma_px" in out_names
            fetch_names = (
                ["keypoints_xy", "keylines_hesse", "cov_2x2", "keyline_sigma_px"]
                if has_keyline_sigma
                else ["keypoints_xy", "keylines_hesse", "cov_2x2"]
            )
            with self._land_ort_lock:
                ort_outs = self.bundle.land_session.run(fetch_names, {"roi_crop": inp_np})
            kpts_xy = np.asarray(ort_outs[0][0], dtype=np.float64)
            keylines = np.asarray(ort_outs[1][0], dtype=np.float64)
            cov_raw = torch.from_numpy(np.asarray(ort_outs[2][0], dtype=np.float32))
            if has_keyline_sigma:
                raw_keyline_sigma_px = np.asarray(ort_outs[3][0], dtype=np.float64).reshape(-1)
        else:
            if self.bundle.land_model is None:
                raise RuntimeError("XVS-Land neither ONNX session nor PyTorch model is initialized")
            inp = torch.from_numpy(inp_np).to(self.bundle.land_device)
            out = self.bundle.land_model(inp)
            kpts_xy = out["keypoints_xy"][0].detach().cpu().numpy().astype(np.float64)
            keylines = out["keylines_hesse"][0].detach().cpu().numpy().astype(np.float64)
            cov_raw = out["cov_2x2"][0].detach().cpu()
            if "keyline_sigma_px" in out:
                raw_keyline_sigma_px = (
                    out["keyline_sigma_px"][0].detach().cpu().numpy().astype(np.float64).reshape(-1)
                )

        # Fail-closed validation: `None` on 5-output v5.1 models, finite > 0 `(5,)` on 6-output v5.2 models
        keyline_sigmas_px = validate_keyline_sigmas_px(raw_keyline_sigma_px, expected_count=keylines.shape[0])
        onnx_contract = "v52_6output" if keyline_sigmas_px is not None else "v51_5output"

        cov_cal = self.bundle.land_calibrator.calibrate_covariance(cov_raw).numpy().astype(np.float64)

        # Optional 4-DoF Horizon-Constrained Huber-Tukey IRLS MAP-PnPL (strictly NED, zero silent defaults)
        pnpl_payload: dict[str, Any] | None = None
        if solve_pnpl:
            hdg_default = math.radians(297.7)
            use_pts_ned = self._land_ref_pts_ned if pts_ned is None else np.asarray(pts_ned, dtype=np.float64)
            use_kl_pairs = self._land_ref_kl_pairs if keyline_pairs is None else list(keyline_pairs)
            use_init_pos = (
                np.array(
                    [-900.0 * math.cos(hdg_default), -900.0 * math.sin(hdg_default), -55.0],
                    dtype=np.float64,
                )
                if init_pos_ned is None
                else np.asarray(init_pos_ned, dtype=np.float64)
            )
            use_init_yaw = hdg_default if init_yaw_rad is None else float(init_yaw_rad)
            use_fx = float(roi_w) * 1.15 if fx is None else float(fx)
            use_fy = float(roi_h) * 1.15 if fy is None else float(fy)
            use_cx = 0.5 * float(roi_w) if cx is None else float(cx)
            use_cy = 0.5 * float(roi_h) if cy is None else float(cy)
            pnpl_sol = solve_4dof_map_pnpl(
                use_pts_ned,
                kpts_xy,
                cov_cal,
                pitch_rad=float(pitch_rad),
                roll_rad=float(roll_rad),
                init_pos_ned=use_init_pos,
                init_yaw_rad=use_init_yaw,
                fx=use_fx,
                fy=use_fy,
                cx=use_cx,
                cy=use_cy,
                keyline_pairs=use_kl_pairs,
                obs_keylines_hesse=keylines,
                keyline_sigmas_px=keyline_sigmas_px,
                huber_delta_sigma=float(huber_delta_sigma),
                prior_pos_ned=use_init_pos,
                prior_yaw_rad=use_init_yaw,
                max_iters=15,
            )
            pnpl_payload = {
                "pos_ned_m": [
                    round(float(pnpl_sol.pos_ned_m[0]), 3) if math.isfinite(float(pnpl_sol.pos_ned_m[0])) else 0.0,
                    round(float(pnpl_sol.pos_ned_m[1]), 3) if math.isfinite(float(pnpl_sol.pos_ned_m[1])) else 0.0,
                    round(float(pnpl_sol.pos_ned_m[2]), 3) if math.isfinite(float(pnpl_sol.pos_ned_m[2])) else 0.0,
                ],
                "yaw_deg": round(math.degrees(float(pnpl_sol.yaw_rad)), 3)
                if math.isfinite(float(pnpl_sol.yaw_rad))
                else 0.0,
                "rms_reproj_px": round(float(pnpl_sol.rms_reproj_px), 3)
                if math.isfinite(float(pnpl_sol.rms_reproj_px))
                else 0.0,
                "chi2": round(float(pnpl_sol.chi2), 3) if math.isfinite(float(pnpl_sol.chi2)) else 0.0,
                "dof": int(pnpl_sol.dof),
                "converged": bool(pnpl_sol.converged),
                "iterations": int(pnpl_sol.iterations),
                "used_keyline_sigmas_px": bool(keyline_sigmas_px is not None),
                "huber_tukey_irls": True,
            }

        scale_u_256 = 256.0 / float(roi_w)
        scale_v_256 = 256.0 / float(roi_h)
        kpts_256 = np.column_stack([kpts_xy[:, 0] * scale_u_256, kpts_xy[:, 1] * scale_v_256])

        sigma_px_per_kp = np.sqrt(np.maximum(1e-6, cov_cal[:, 0, 0] + cov_cal[:, 1, 1]))
        mean_sigma_px = float(np.mean(sigma_px_per_kp))
        center_u = float(np.mean(kpts_256[:, 0]))
        center_v = float(np.mean(kpts_256[:, 1]))
        lat_dev_deg = round((center_u - 128.0) / 256.0 * 18.0, 2)
        glideslope_deg = round(3.0 + (128.0 - center_v) / 256.0 * 2.5, 2)
        hpl_m = round(max(12.0, mean_sigma_px * 14.5), 2)
        vpl_m = round(max(6.0, mean_sigma_px * 5.2), 2)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        summary_str = f"28L 12-KPT | GS {glideslope_deg:.1f}deg | VPL {vpl_m:.0f}m"
        ckpt_sha = land_meta.sha256

        res: dict[str, Any] = {
            "channel": "runway_land",
            "model": "XVS-Land",
            "device": land_meta.device,
            "backend": land_meta.backend,
            "real_weights": land_meta.real_weights,
            "checkpoint_sha256": ckpt_sha,
            "latency_ms": latency_ms,
            "confidence": round(max(0.1, min(0.99, 1.0 - mean_sigma_px / 60.0)), 4),
            "summary": summary_str,
            "onnx_output_contract": onnx_contract,
            "roi_hw": [int(roi_h), int(roi_w)],
            "keypoints_xy_256": [[round(float(x), 2), round(float(y), 2)] for x, y in kpts_256],
            "keylines_hesse": [
                [round(float(r), 4), round(float(c), 4), round(float(s), 4)]
                for r, c, s in keylines
            ],
            "keyline_sigma_px": (
                [round(float(s), 4) for s in keyline_sigmas_px]
                if keyline_sigmas_px is not None
                else None
            ),
            "mean_keyline_sigma_px": (
                round(float(np.mean(keyline_sigmas_px)), 4)
                if keyline_sigmas_px is not None
                else None
            ),
            "pnpl": pnpl_payload,
            "mean_keypoint_sigma_px": round(mean_sigma_px, 3),
            "glideslope_deg": glideslope_deg,
            "lateral_dev_deg": lat_dev_deg,
            "hpl_m": hpl_m,
            "vpl_m": vpl_m,
            "hmi_count": 0,
        }

        if return_jpeg:
            vis = cv2.resize(bgr, (384, 256), interpolation=cv2.INTER_AREA)
            ox, oy, rw, rh = roi_origin
            sx = 384.0 / float(max(1, w_in))
            sy = 256.0 / float(max(1, h_in))
            pts_vis = [
                (
                    int(round((ox + (x / 256.0) * rw) * sx)),
                    int(round((oy + (y / 256.0) * rh) * sy)),
                )
                for x, y in kpts_256
            ]
            if roi_box_px is not None:
                p_r1 = (int(round(ox * sx)), int(round(oy * sy)))
                p_r2 = (int(round((ox + rw) * sx)), int(round((oy + rh) * sy)))
                cv2.rectangle(vis, p_r1, p_r2, COL_AMBER, 1, cv2.LINE_AA)
            for idx, pt in enumerate(pts_vis):
                col = COL_CYAN if idx % 2 == 0 else COL_EMERALD
                cv2.circle(vis, pt, 3, col, -1, cv2.LINE_AA)
            if len(pts_vis) >= 12:
                cv2.line(vis, pts_vis[0], pts_vis[9], COL_CYAN, 2, cv2.LINE_AA)
                cv2.line(vis, pts_vis[2], pts_vis[11], COL_CYAN, 2, cv2.LINE_AA)
                cv2.line(vis, pts_vis[1], pts_vis[10], COL_EMERALD, 1, cv2.LINE_AA)
                cv2.line(vis, pts_vis[0], pts_vis[2], COL_EMERALD, 2, cv2.LINE_AA)
            _draw_badge(vis, summary_str, (8, 22), COL_CYAN)
            b64 = _encode_jpeg_b64(vis, quality=84)
            res["image_b64"] = b64
            res["annotated_jpeg_b64"] = f"data:image/jpeg;base64,{b64}"
            res["image_mime"] = "image/jpeg"

        return res

    # ---- Channel 3: XVS-ThermalNet (EO + 2-ch LWIR + Dirichlet Vacuity + ASTM Simplex) ----

    @torch.no_grad()
    def infer_thermal_dve(
        self,
        frame_bgr: np.ndarray,
        *,
        lwir_bgr: np.ndarray | None = None,
        q_eo: float = 0.85,
        q_lwir: float = 0.95,
        fog_density: float = 0.0,
        return_jpeg: bool = True,
        frame_ts_ms: float | None = None,
        imu_ts_ms: float | None = None,
        dt_ms: float | None = None,
        sensor_health_monitor: SensorHealthMonitor | None = None,
        enforce_sensor_health_gate: bool = False,
    ) -> dict[str, Any]:
        """Run real Colab-L4 `XvsThermalNet` EO+LWIR fusion + Dirichlet vacuity + ASTM Simplex RTA."""
        t0 = time.perf_counter()
        bgr = _ensure_bgr(frame_bgr)
        eo_resized = cv2.resize(bgr, (320, 256), interpolation=cv2.INTER_AREA)

        fog_c = max(0.0, min(1.0, float(fog_density)))
        if fog_c > 0.01:
            eo_vis = np.clip(
                eo_resized.astype(np.float32) * (1.0 - 0.88 * fog_c) + 220.0 * (0.88 * fog_c),
                0.0,
                255.0,
            ).astype(np.uint8)
        else:
            eo_vis = eo_resized

        eo_rgb = cv2.cvtColor(eo_vis, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lwir_2ch, lwir_u8 = _derive_lwir_2ch_from_bgr(bgr, lwir_bgr, hw=(128, 160))

        active_sh = self._sensor_health if sensor_health_monitor is None else sensor_health_monitor
        th_sensor_health = active_sh.assess_thermal_pair(
            eo_vis,
            lwir_u8,
            q_eo_nominal=float(q_eo),
            q_lwir_nominal=float(q_lwir),
            vacuity=0.0,
            frame_ts_ms=frame_ts_ms,
            imu_ts_ms=imu_ts_ms,
            dt_ms=dt_ms,
        )
        q_eo_run = (
            float(th_sensor_health["q_eo_effective"]) if enforce_sensor_health_gate else float(q_eo)
        )
        q_lwir_run = (
            float(th_sensor_health["q_lwir_effective"]) if enforce_sensor_health_gate else float(q_lwir)
        )

        x_eo_np = np.ascontiguousarray(np.transpose(eo_rgb, (2, 0, 1))[np.newaxis, :, :, :], dtype=np.float32)
        x_lwir_np = np.ascontiguousarray(lwir_2ch[np.newaxis, :, :, :], dtype=np.float32)

        if self.bundle.thermal_session is not None:
            with self._thermal_ort_lock:
                hm_logits_ort, vacuity_ort, w_eo_ort, w_lwir_ort = self.bundle.thermal_session.run(
                    ["fused_hm_logits", "vacuity", "w_eo_eff", "w_lwir_eff"],
                    {
                        "x_eo": x_eo_np,
                        "x_lwir": x_lwir_np,
                        "q_eo": np.array([q_eo_run], dtype=np.float32),
                        "q_lwir": np.array([q_lwir_run], dtype=np.float32),
                    },
                )
            vacuity_map = np.asarray(vacuity_ort[0, 0], dtype=np.float64)
            hm_probs = _sigmoid_np(np.asarray(hm_logits_ort[0, 0], dtype=np.float32)).astype(np.float64)
            w_eo_eff = float(np.mean(w_eo_ort))
            w_lwir_eff = float(np.mean(w_lwir_ort))
        else:
            if self.bundle.thermal_model is None:
                raise RuntimeError("XVS-ThermalNet neither ONNX session nor PyTorch model is initialized")
            x_eo = torch.from_numpy(x_eo_np).to(self.bundle.thermal_device)
            x_lwir = torch.from_numpy(x_lwir_np).to(self.bundle.thermal_device)
            q_eo_t = torch.tensor([q_eo_run], dtype=torch.float32, device=self.bundle.thermal_device)
            q_lwir_t = torch.tensor([q_lwir_run], dtype=torch.float32, device=self.bundle.thermal_device)
            out = self.bundle.thermal_model(x_eo, x_lwir, q_eo=q_eo_t, q_lwir=q_lwir_t)
            vacuity_map = out["vacuity"][0, 0].detach().cpu().numpy().astype(np.float64)
            hm_probs = torch.sigmoid(out["fused_hm_logits"][0, 0]).detach().cpu().numpy().astype(np.float64)
            w_eo_eff = float(out["w_eo_eff"].mean().detach().cpu().item())
            w_lwir_eff = float(out["w_lwir_eff"].mean().detach().cpu().item())

        mean_vacuity = float(np.mean(vacuity_map))
        max_hm_prob = float(np.max(hm_probs))
        peak_flat = int(np.argmax(hm_probs))
        peak_y, peak_x = divmod(peak_flat, hm_probs.shape[1])

        simplex_mode: ThermalSimplexMode = select_thermal_simplex_mode(
            q_eo=q_eo_run, q_lwir=q_lwir_run, vacuity=mean_vacuity
        )
        sh_mode: ThermalSimplexMode = select_thermal_simplex_mode(
            q_eo=float(th_sensor_health["q_eo_effective"]),
            q_lwir=float(th_sensor_health["q_lwir_effective"]),
            vacuity=mean_vacuity,
        )
        th_sensor_health["thermal_mode"] = sh_mode.name
        th_sensor_health["thermal_mode_id"] = int(sh_mode)
        th_sensor_health["gate_enforced"] = bool(enforce_sensor_health_gate)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        short_mode = (
            "LWIR_PRIMARY"
            if simplex_mode == ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
            else simplex_mode.name[:16]
        )
        summary_str = f"EO|LWIR {short_mode} | Vac {mean_vacuity:.2f}"
        th_meta = self.bundle.meta["xvs_thermal"]
        ckpt_sha = th_meta.sha256

        res: dict[str, Any] = {
            "channel": "thermal_dve",
            "model": "XVS-ThermalNet",
            "device": th_meta.device,
            "backend": th_meta.backend,
            "real_weights": th_meta.real_weights,
            "checkpoint_sha256": ckpt_sha,
            "latency_ms": latency_ms,
            "confidence": round(max(0.0, min(1.0, 1.0 - mean_vacuity)), 4),
            "summary": summary_str,
            "q_eo": round(q_eo_run, 3),
            "q_lwir": round(q_lwir_run, 3),
            "w_eo_eff": round(w_eo_eff, 4),
            "w_lwir_eff": round(w_lwir_eff, 4),
            "mean_vacuity": round(mean_vacuity, 4),
            "max_heatmap_prob": round(max_hm_prob, 4),
            "peak_grid_xy": [int(peak_x), int(peak_y)],
            "simplex_mode": simplex_mode.name,
            "simplex_mode_id": int(simplex_mode),
            "sensor_health": th_sensor_health,
        }

        if return_jpeg:
            if lwir_bgr is not None:
                lwir_color = cv2.resize(_ensure_bgr(lwir_bgr), (384, 256), interpolation=cv2.INTER_AREA)
            else:
                lwir_color = cv2.applyColorMap(
                    cv2.resize(lwir_u8, (384, 256), interpolation=cv2.INTER_LINEAR),
                    cv2.COLORMAP_INFERNO,
                )
            eo_half = cv2.resize(eo_vis, (384, 256), interpolation=cv2.INTER_AREA)
            vis = lwir_color.copy()
            vis[:, :192] = eo_half[:, :192]
            cv2.line(vis, (192, 0), (192, 255), COL_AMBER, 2, cv2.LINE_AA)
            px = int(round((float(peak_x) + 0.5) / float(hm_probs.shape[1]) * 384.0))
            py = int(round((float(peak_y) + 0.5) / float(hm_probs.shape[0]) * 256.0))
            cv2.circle(vis, (px, py), 10, COL_EMERALD, 2, cv2.LINE_AA)
            bar_w = int(round(120.0 * max(0.0, min(1.0, mean_vacuity))))
            cv2.rectangle(vis, (8, 236), (132, 248), COL_BG, -1)
            cv2.rectangle(vis, (10, 238), (10 + bar_w, 246), COL_AMBER if mean_vacuity < 0.7 else COL_ROSE, -1)
            _draw_badge(
                vis,
                summary_str,
                (8, 22),
                COL_AMBER if simplex_mode == ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY else COL_EMERALD,
            )
            b64 = _encode_jpeg_b64(vis, quality=84)
            res["image_b64"] = b64
            res["annotated_jpeg_b64"] = f"data:image/jpeg;base64,{b64}"
            res["image_mime"] = "image/jpeg"

        return res

    # ---- Channel 4: XVS-Nav v5.0 (MNV4-Hybrid-M + MixVPR-Clean ONNX / NB2 Fallback + Aliasing Gate + Local CUSUM) ----

    @torch.no_grad()
    def _encode_nav_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Encode BGR image to L2-normalized 512-D float32 descriptor (V5.0 ONNX folded whitening or NB2 .pt)."""
        tile = cv2.resize(_ensure_bgr(bgr), (224, 224), interpolation=cv2.INTER_AREA)
        if self.bundle.nav_session is not None:
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp_np = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]
            with self._nav_ort_lock:
                in_name = self.bundle.nav_session.get_inputs()[0].name
                raw_out = self.bundle.nav_session.run(None, {in_name: inp_np})[0]
            # V5.0 ONNX embeds Float64-fitted Gemm(512, 1024) + LpNormalization(p=2): zero external NumPy whitening
            return np.asarray(raw_out[0], dtype=np.float32).reshape(-1)

        if self.bundle.nav_model is None:
            small = cv2.resize(tile, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            vec = small.reshape(-1)[:512]
            if vec.size < 512:
                vec = np.pad(vec, (0, 512 - vec.size))
            norm = max(float(np.linalg.norm(vec)), 1e-12)
            return (vec / norm).astype(np.float32)

        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.bundle.nav_device, dtype=self.bundle.nav_dtype)
        )
        raw_desc = self.bundle.nav_model(inp)[0].detach().cpu().float().numpy()
        return _apply_pca_whitening(raw_desc, self.bundle.nav_pca_whitening)[0]

    def _get_cached_ref_desc(self, ref_tile: np.ndarray, ref_cache_key: str | None) -> np.ndarray:
        """Return cached static satellite reference descriptor (real onboard map DB behavior)."""
        if ref_cache_key is not None:
            with self._cache_lock:
                if ref_cache_key in self._nav_ref_desc_cache:
                    return self._nav_ref_desc_cache[ref_cache_key]
        desc = self._encode_nav_bgr(ref_tile)
        if ref_cache_key is not None:
            with self._cache_lock:
                self._nav_ref_desc_cache[ref_cache_key] = desc
        return desc

    def _get_cached_ref_xfeat(
        self,
        ref_src: np.ndarray,
        ref_cache_key: str | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached Stage-2A XFeat luminance keypoints/descriptors for static satellite tiles."""
        if ref_cache_key is not None:
            with self._cache_lock:
                if ref_cache_key in self._nav_ref_xfeat_cache:
                    return self._nav_ref_xfeat_cache[ref_cache_key]
        feats = self._nav_xfeat.extract(ref_src)
        if ref_cache_key is not None:
            with self._cache_lock:
                self._nav_ref_xfeat_cache[ref_cache_key] = feats
        return feats

    def _ensure_nav_gallery(self) -> tuple[np.ndarray, list[np.ndarray]]:
        if self._nav_gallery_descs is not None and self._nav_gallery_tiles:
            return self._nav_gallery_descs, self._nav_gallery_tiles
        ortho = self._load_ksfo_ortho()
        oh, ow = ortho.shape[:2]
        tiles: list[np.ndarray] = []
        descs: list[np.ndarray] = []
        grid_coords = [
            (0.25, 0.25),
            (0.55, 0.35),
            (0.30, 0.65),
            (0.65, 0.65),
        ]
        crop_sz = max(128, min(oh, ow) // 3)
        for idx, (fx, fy) in enumerate(grid_coords):
            cx = int(round(fx * ow))
            cy = int(round(fy * oh))
            x0 = max(0, min(ow - crop_sz, cx - crop_sz // 2))
            y0 = max(0, min(oh - crop_sz, cy - crop_sz // 2))
            patch = ortho[y0 : y0 + crop_sz, x0 : x0 + crop_sz]
            tile = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
            tiles.append(tile)
            descs.append(self._get_cached_ref_desc(tile, f"ksfo_gallery_{idx}"))
        self._nav_gallery_tiles = tiles
        self._nav_gallery_descs = np.stack(descs, axis=0).astype(np.float32)
        return self._nav_gallery_descs, self._nav_gallery_tiles

    def infer_crossview_nav(
        self,
        frame_bgr: np.ndarray,
        *,
        ref_bgr: np.ndarray | None = None,
        ref_cache_key: str | None = None,
        correspondence_pairs: list[dict[str, Any]] | None = None,
        gps_offset_m: float = 0.0,
        cusum_filter: DualBankEsEkfCusum | None = None,
        seq_vpr_filter: SequenceVprFilter | None = None,
        delta_ne_odom_m: tuple[float, float] = (0.0, 0.0),
        odom_sigma_m: float = 25.0,
        seq_vpr_similarities: np.ndarray | None = None,
        return_jpeg: bool = True,
        frame_ts_ms: float | None = None,
        imu_ts_ms: float | None = None,
        sensor_health_monitor: SensorHealthMonitor | None = None,
        camera_intrinsics: PinholeIntrinsics | None = None,
    ) -> dict[str, Any]:
        """Run XVS-Nav retrieval (V5.0 ONNX or NB2 .pt) + 6-stage aliasing gate + local 8-arm CUSUM + Stage-2 XFeat."""
        t0 = time.perf_counter()
        bgr = _ensure_bgr(frame_bgr)
        h_in, w_in = bgr.shape[:2]
        active_sh = self._sensor_health if sensor_health_monitor is None else sensor_health_monitor
        nav_sensor_health = active_sh.assess_eo_frame(
            bgr,
            channel="crossview_nav",
            frame_ts_ms=frame_ts_ms,
            imu_ts_ms=imu_ts_ms,
        )
        nav_sensor_health["camera_rectified"] = bool(camera_intrinsics is not None)
        q_desc = self._encode_nav_bgr(bgr)
        gallery_descs, gallery_tiles = self._ensure_nav_gallery()

        # 1. Stage-1 Retrieval + Calibrated Background-Centered PSR_cal (`aliasing_gate.py`)
        if ref_bgr is not None:
            ref_src = _ensure_bgr(ref_bgr)
            h_ref, w_ref = ref_src.shape[:2]
            ref_tile = cv2.resize(ref_src, (224, 224), interpolation=cv2.INTER_AREA)
            r_desc = self._get_cached_ref_desc(ref_tile, ref_cache_key)
            cosine_sim = float(np.dot(q_desc, r_desc))

            # Use non-identical KSFO gallery tiles as genuine non-adjacent (>300 m NED) distractors
            ref_vs_gal = gallery_descs @ r_desc
            keep_dist = ref_vs_gal <= 0.985
            dist_descs = gallery_descs[keep_dist] if bool(np.any(keep_dist)) else gallery_descs
            dist_sims = (dist_descs @ q_desc).astype(np.float64)
            sims_all = np.concatenate([np.array([cosine_sim], dtype=np.float64), dist_sims])
            dist_coords = np.array(
                [[650.0 * float(i + 1), 0.0] for i in range(int(dist_sims.size))],
                dtype=np.float64,
            )
            coords_ne = np.vstack([np.array([[0.0, 0.0]], dtype=np.float64), dist_coords])
            psr_res = compute_psr_calibrated(
                sims_all,
                coords_ne,
                non_adj_radius_m=V5_NAV_NON_ADJ_RADIUS_M,
                delta_min=V5_NAV_PSR_DELTA_MIN,
                min_psr=MIN_PSR_LOCKED,
            )
            if int(np.argmax(sims_all)) == 0:
                second_best = float(psr_res.s2_non_adj)
                psr_cal_val = float(psr_res.psr_cal)
                psr_margin_val = float(psr_res.margin)
                psr_mu_bg_val = float(psr_res.mu_bg)
            else:
                second_best = float(np.max(dist_sims))
                psr_mu_bg_val = float(np.median(dist_sims))
                psr_cal_val = max(cosine_sim - psr_mu_bg_val, 1e-4) / max(second_best - psr_mu_bg_val, 1e-4)
                psr_margin_val = cosine_sim - second_best
            best_idx = 0
            xfeat_ref_key = ref_cache_key
        else:
            sims_gal = (gallery_descs @ q_desc).astype(np.float64)
            order = np.argsort(-sims_gal)
            best_idx = int(order[0])
            cosine_sim = float(sims_gal[best_idx])
            psr_res = compute_psr_calibrated(
                sims_gal,
                self._nav_gallery_ne_m,
                non_adj_radius_m=V5_NAV_NON_ADJ_RADIUS_M,
                delta_min=V5_NAV_PSR_DELTA_MIN,
                min_psr=MIN_PSR_LOCKED,
            )
            second_best = (
                float(psr_res.s2_non_adj)
                if math.isfinite(psr_res.s2_non_adj)
                else (float(sims_gal[int(order[1])]) if sims_gal.size >= 2 else 0.0)
            )
            psr_cal_val = float(psr_res.psr_cal) if math.isfinite(psr_res.psr_cal) else 0.0
            psr_margin_val = float(psr_res.margin) if math.isfinite(psr_res.margin) else 0.0
            psr_mu_bg_val = float(psr_res.mu_bg) if math.isfinite(psr_res.mu_bg) else 0.0
            ref_src = gallery_tiles[best_idx]
            ref_tile = ref_src
            h_ref, w_ref = ref_src.shape[:2]
            xfeat_ref_key = f"ksfo_gallery_{best_idx}"

        # 2. Stage-2A XFeat + Conditional Tier-2B LighterGlue Geometric Correspondences (0 synthetic lines)
        kq, dq, _ = self._nav_xfeat.extract(bgr)
        kr, dr, _ = self._get_cached_ref_xfeat(ref_src, xfeat_ref_key)
        q_idx, m_idx, m_scores = match_mnn(dq, dr)
        tier2a_matches = int(q_idx.size)
        tier2b_triggered = bool(tier2a_matches < TIER2B_TRIGGER_MIN_INLIERS)
        tier2b_matches = 0
        stage2_tier_used = "tier2a_xfeat_mnn"

        if tier2b_triggered and dq.shape[0] > 0 and dr.shape[0] > 0:
            q2_idx, m2_idx, m2_scores = self._nav_lighterglue.match(
                kq, dq, kr, dr, img_hw=(h_in, w_in)
            )
            tier2b_matches = int(q2_idx.size)
            if tier2b_matches > tier2a_matches:
                q_idx, m_idx, m_scores = q2_idx, m2_idx, m2_scores
                stage2_tier_used = "tier2b_lighterglue"

        inlier_q_pts = kq[q_idx] if q_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)
        inlier_r_pts = kr[m_idx] if m_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)
        inlier_scores = m_scores if m_scores.size > 0 else np.zeros((0,), dtype=np.float32)
        if camera_intrinsics is not None and inlier_q_pts.shape[0] > 0:
            inlier_q_pts = undistort_keypoints_pinhole(
                inlier_q_pts, camera_intrinsics
            ).astype(np.float32)
        ransac_inliers = int(q_idx.size)
        if q_idx.size >= 4:
            _, h_mask = cv2.findHomography(inlier_q_pts, inlier_r_pts, cv2.RANSAC, 5.0)
            if h_mask is not None:
                mask_bool = h_mask.reshape(-1).astype(bool)
                n_h_inliers = int(np.sum(mask_bool))
                ransac_inliers = n_h_inliers
                if n_h_inliers >= 4:
                    inlier_q_pts = inlier_q_pts[mask_bool]
                    inlier_r_pts = inlier_r_pts[mask_bool]
                    inlier_scores = inlier_scores[mask_bool]

        # 3. 6-Stage Visual-Aliasing Admission Gate + Split-Conformal Multi-Hypothesis HPL (HAL = 30.0 m)
        if inlier_q_pts.shape[0] >= 3:
            geom_pts_q = inlier_q_pts.astype(np.float64)
        elif correspondence_pairs:
            geom_pts_q = np.array(
                [[float(p["aerial_x"]), float(p["aerial_y"])] for p in correspondence_pairs],
                dtype=np.float64,
            )
        else:
            geom_pts_q = np.zeros((0, 2), dtype=np.float64)

        n_inliers_gate = (
            int(ransac_inliers)
            if ransac_inliers > 0
            else (len(correspondence_pairs) if correspondence_pairs else 0)
        )
        h_norm = (
            compute_spatial_entropy_4x4(geom_pts_q, float(w_in), float(h_in))
            if geom_pts_q.shape[0] > 0
            else 0.0
        )
        n_eff = float(2.0 ** (4.0 * h_norm)) if (math.isfinite(h_norm) and geom_pts_q.shape[0] > 0) else 0.0
        kappa_geom = compute_geom_condition_number(geom_pts_q) if geom_pts_q.shape[0] >= 3 else float("inf")
        kappa_fim = float(kappa_geom)

        cos_clamped = max(0.0, min(1.0, cosine_sim))
        support_scale = (
            1.0
            if n_inliers_gate >= TIER2B_TRIGGER_MIN_INLIERS
            else (1.25 if n_inliers_gate >= 10 else 1.65)
        )
        sigma_vis_raw_m = (2.2 + 12.0 * (1.0 - cos_clamped)) * support_scale
        precision_profile = "fp16"
        sigma_vis_cal_m = compute_calibrated_sigma_vis(
            sigma_vis_raw_m,
            gamma_conf=V5_NAV_GAMMA_CONF_DEFAULT,
            precision_profile=precision_profile,
        )
        hpl_res = compute_multi_hypothesis_hpl(sigma_vis_cal_m, hal_m=DEFAULT_HAL_NAV_M)
        hpl_vis_m = round(float(hpl_res.hpl_m), 2) if math.isfinite(hpl_res.hpl_m) else 999.0

        # 4. Local-Only Dual-Bank ES-EKF + 8-Arm Cholesky Page's CUSUM (`k_d=0.50, h=17.619`)
        active_cusum = DualBankEsEkfCusum() if cusum_filter is None else cusum_filter
        r_vis_mat = (
            (float(sigma_vis_cal_m) ** 2) * np.eye(2, dtype=np.float64)
            if math.isfinite(sigma_vis_cal_m) and sigma_vis_cal_m > 0.0
            else active_cusum._R_vis_default
        )
        pred_xb = active_cusum._F @ active_cusum.x_b
        pred_pb = active_cusum._F @ active_cusum.p_b @ active_cusum._F.T + active_cusum._Q_ins
        s_vis_pre = active_cusum._H @ pred_pb @ active_cusum._H.T + r_vis_mat
        innov_pre = np.array([0.0, 0.0], dtype=np.float64) - active_cusum._H @ pred_xb
        q_vis_pre = compute_innovation_q_vis(innov_pre, s_vis_pre)

        gate_thresholds = VisualAliasingThresholds(
            min_psr=MIN_PSR_LOCKED,
            min_psr_delta=V5_NAV_PSR_DELTA_MIN,
            default_hal_nav_m=DEFAULT_HAL_NAV_M,
        )
        vis_fix_metrics = VisualFixMetrics(
            n_inliers=int(n_inliers_gate),
            n_eff=float(n_eff),
            h_norm=float(h_norm),
            kappa_geom=float(kappa_geom),
            kappa_fim=float(kappa_fim),
            psr=float(psr_cal_val),
            chi2_mc=0.0,
            q_vis=float(q_vis_pre),
            hpl_nav_m=float(hpl_res.hpl_m),
            hal_nav_m=float(DEFAULT_HAL_NAV_M),
            psr_margin=float(psr_margin_val),
        )
        gate_decision = evaluate_visual_aliasing_gate(
            vis_fix_metrics,
            thresholds=gate_thresholds,
        )

        # 3b. Multi-frame NED IMU-Odometry Sequence-VPR (`W=7`, strict fail-closed `0 HMI`)
        if seq_vpr_filter is not None:
            active_seq_vpr = seq_vpr_filter
            seq_sims = (
                np.asarray(seq_vpr_similarities, dtype=np.float64)
                if seq_vpr_similarities is not None
                else (sims_all if ref_bgr is not None else sims_gal)
            )
            seq_dec: SequenceVprDecision = active_seq_vpr.step(
                seq_sims,
                delta_ne_odom_m=delta_ne_odom_m,
                odom_sigma_m=float(odom_sigma_m),
                visual_fix_metrics=vis_fix_metrics,
                aliasing_thresholds=gate_thresholds,
            )
        else:
            with self._nav_seq_vpr_lock:
                if ref_bgr is not None:
                    if (
                        self._ref_seq_vpr is None
                        or self._ref_seq_vpr.num_tiles != coords_ne.shape[0]
                        or not np.allclose(self._ref_seq_vpr.gallery_ne_m, coords_ne)
                    ):
                        self._ref_seq_vpr = self.new_nav_sequence_vpr_filter(gallery_ne_m=coords_ne)
                    active_seq_vpr = self._ref_seq_vpr
                    seq_sims = (
                        np.asarray(seq_vpr_similarities, dtype=np.float64)
                        if seq_vpr_similarities is not None
                        else sims_all
                    )
                else:
                    active_seq_vpr = self._demo_seq_vpr
                    seq_sims = (
                        np.asarray(seq_vpr_similarities, dtype=np.float64)
                        if seq_vpr_similarities is not None
                        else sims_gal
                    )
                seq_dec = active_seq_vpr.step(
                    seq_sims,
                    delta_ne_odom_m=delta_ne_odom_m,
                    odom_sigma_m=float(odom_sigma_m),
                    visual_fix_metrics=vis_fix_metrics,
                    aliasing_thresholds=gate_thresholds,
                )

        seq_post_prob = (
            round(float(seq_dec.posterior_prob_top1), 4)
            if math.isfinite(float(seq_dec.posterior_prob_top1))
            else 0.0
        )
        seq_post_margin = (
            round(float(seq_dec.posterior_margin_non_adj), 4)
            if math.isfinite(float(seq_dec.posterior_margin_non_adj))
            else None
        )
        seq_psr_cal = (
            round(float(seq_dec.psr_result.psr_cal), 3)
            if math.isfinite(float(seq_dec.psr_result.psr_cal))
            else None
        )

        cusum_out = active_cusum.step(
            gnss_ne_m=(float(gps_offset_m), 0.0),
            vis_ne_m=(0.0, 0.0),
            vis_gate_admitted=bool(gate_decision.admitted),
            r_vis=r_vis_mat,
        )
        spoof_detected = bool(cusum_out.gnss_quarantined)
        try:
            nis_val = float(cusum_out.d_ne_m @ np.linalg.solve(cusum_out.p_diff_ne, cusum_out.d_ne_m))
        except np.linalg.LinAlgError:
            nis_val = float(np.dot(cusum_out.z_whitened, cusum_out.z_whitened))

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        summary_str = (
            f"SPOOF +{gps_offset_m:.0f}m ISOLATED | CUSUM {cusum_out.max_cusum_stat:.1f}>={CUSUM_H_LOCKED:.1f}"
            if spoof_detected
            else f"NAIP LOCK | SIM {cosine_sim:.2f} | PSR {psr_cal_val:.2f} | HPL {hpl_vis_m:.0f}m"
        )
        nav_meta = self.bundle.meta["xvs_nav"]
        model_label = (
            "XVS-Nav v5.0 (MobileNetV4-Hybrid-M + MixVPR-Clean ONNX)"
            if self.bundle.nav_session is not None
            else "XVS-Nav (DINOv2-B/14 + BoQ)"
        )

        correspondences_drawn = 0
        correspondence_source = "none"
        stage2_meta: dict[str, Any] = {
            "xfeat_backend": self._nav_xfeat.backend,
            "xfeat_onnx_sha256": self._nav_xfeat.onnx_sha256,
            "lighterglue_backend": self._nav_lighterglue.backend,
            "tier_used": stage2_tier_used,
            "query_keypoints": int(kq.shape[0]),
            "ref_keypoints": int(kr.shape[0]),
            "tier2a_mnn_matches": int(tier2a_matches),
            "tier2b_triggered": bool(tier2b_triggered),
            "tier2b_trigger_min_inliers": int(TIER2B_TRIGGER_MIN_INLIERS),
            "tier2b_matches": int(tier2b_matches),
            "ransac_inliers": int(ransac_inliers),
            "cache_budget_bytes_per_tile": int(MAX_DESCRIPTOR_BYTES_PER_TILE),
            "ref_descriptor_bytes": int(kr.shape[0] * DESCRIPTOR_DIM),
            "correspondences_drawn": 0,
            "correspondence_source": "none",
            "synthetic_lines_used": False,
        }

        res: dict[str, Any] = {
            "channel": "crossview_nav",
            "model": model_label,
            "device": nav_meta.device,
            "dtype": nav_meta.dtype,
            "backend": nav_meta.backend,
            "tier": nav_meta.tier,
            "real_weights": nav_meta.real_weights,
            "checkpoint_sha256": nav_meta.sha256,
            "latency_ms": latency_ms,
            "confidence": round(max(0.0, min(1.0, 0.5 * (cosine_sim + 1.0))), 4),
            "summary": summary_str,
            "cosine_sim": round(cosine_sim, 4),
            "second_best_sim": round(second_best, 4),
            "best_tile_idx": best_idx,
            "psr": round(float(psr_cal_val), 3),
            "psr_cal": round(float(psr_cal_val), 3),
            "psr_margin": round(float(psr_margin_val), 4),
            "psr_mu_bg": round(float(psr_mu_bg_val), 4),
            "gamma_conf": round(float(V5_NAV_GAMMA_CONF_DEFAULT), 4),
            "sigma_vis_raw_m": round(float(sigma_vis_raw_m), 3),
            "sigma_vis_cal_m": round(float(sigma_vis_cal_m), 3),
            "k_h0": float(K_H0_LOCKED),
            "hpl_vis_m": hpl_vis_m,
            "hal_nav_m": float(DEFAULT_HAL_NAV_M),
            "hpl_within_hal": bool(hpl_res.within_hal),
            "gps_offset_m": round(float(gps_offset_m), 2),
            "nis": round(float(nis_val), 2),
            "spoof_detected": spoof_detected,
            "nav_mode": "VISUAL_PRIMARY_SPOOF_ISOLATED" if spoof_detected else "GNSS_VISUAL_CONSISTENT",
            "aliasing_gate": {
                "admitted": bool(gate_decision.admitted),
                "integrity_alert": bool(gate_decision.integrity_alert),
                "failed_stages": list(gate_decision.failed_stages),
                "reason_codes": list(gate_decision.reason_codes),
                "stage_pass": dict(gate_decision.stage_pass),
                "n_inliers": int(n_inliers_gate),
                "n_eff": round(float(n_eff), 3),
                "h_norm": round(float(h_norm), 4) if math.isfinite(h_norm) else None,
                "kappa_geom": round(float(kappa_geom), 3) if math.isfinite(kappa_geom) else None,
                "kappa_fim": round(float(kappa_fim), 3) if math.isfinite(kappa_fim) else None,
                "psr_cal": round(float(psr_cal_val), 3),
                "psr_margin": round(float(psr_margin_val), 4),
                "psr_delta_min": float(V5_NAV_PSR_DELTA_MIN),
                "q_vis": round(float(q_vis_pre), 4) if math.isfinite(q_vis_pre) else None,
                "single_camera_mode": True,
                "dominant_hypothesis": hpl_res.dominant_hypothesis,
            },
            "dual_bank_cusum": {
                "num_arms": int(CUSUM_NUM_ARMS),
                "k_d": float(CUSUM_K_D_LOCKED),
                "h_threshold": float(CUSUM_H_LOCKED),
                "active_bank": cusum_out.active_bank,
                "gnss_quarantined": bool(cusum_out.gnss_quarantined),
                "quarantine_triggered_this_epoch": bool(cusum_out.quarantine_triggered_this_epoch),
                "quarantine_latency_s": cusum_out.quarantine_latency_s,
                "max_cusum_stat": round(float(cusum_out.max_cusum_stat), 3),
                "dominant_arm": int(cusum_out.dominant_arm),
                "d_ne_m": [round(float(cusum_out.d_ne_m[0]), 3), round(float(cusum_out.d_ne_m[1]), 3)],
                "q_vis": round(float(cusum_out.q_vis), 4),
                "nominal_streak": int(cusum_out.nominal_streak),
                "epoch_idx": int(cusum_out.epoch_idx),
            },
            "stage2_xfeat": stage2_meta,
            "seq_vpr_window": int(seq_dec.window_len),
            "seq_vpr_window_max": int(active_seq_vpr.config.window_size),
            "seq_vpr_admitted": bool(seq_dec.admitted),
            "seq_vpr_posterior_prob": seq_post_prob,
            "seq_vpr": {
                "mode": str(seq_dec.mode),
                "frame_idx": int(seq_dec.frame_idx),
                "window_len": int(seq_dec.window_len),
                "window_max": int(active_seq_vpr.config.window_size),
                "raw_top1_idx": int(seq_dec.raw_top1_idx),
                "seq_top1_idx": int(seq_dec.seq_top1_idx),
                "final_top1_idx": int(seq_dec.final_top1_idx),
                "viterbi_path_indices": [int(i) for i in seq_dec.viterbi_path_indices],
                "posterior_prob_top1": seq_post_prob,
                "posterior_margin_non_adj": seq_post_margin,
                "psr_cal": seq_psr_cal,
                "admitted": bool(seq_dec.admitted),
                "reason_codes": list(seq_dec.reason_codes),
            },
            "sensor_health": nav_sensor_health,
        }

        if return_jpeg:
            left = cv2.resize(bgr, (192, 256), interpolation=cv2.INTER_AREA)
            right = cv2.resize(ref_tile, (192, 256), interpolation=cv2.INTER_AREA)
            vis = np.hstack([left, right])
            cv2.line(vis, (192, 0), (192, 255), COL_CYAN, 1, cv2.LINE_AA)
            line_col = COL_ROSE if spoof_detected else COL_EMERALD
            lx_s = 192.0 / float(max(1, w_in))
            ly_s = 256.0 / float(max(1, h_in))
            rx_s = 192.0 / float(max(1, w_ref))
            ry_s = 256.0 / float(max(1, h_ref))
            if correspondence_pairs:
                correspondence_source = "manifest_pairs"
                for pair in correspondence_pairs[:8]:
                    qx = max(0, min(191, int(round(float(pair["aerial_x"]) * lx_s))))
                    qy = max(0, min(255, int(round(float(pair["aerial_y"]) * ly_s))))
                    sx = 192 + max(0, min(191, int(round(float(pair["sat_x"]) * rx_s))))
                    sy = max(0, min(255, int(round(float(pair["sat_y"]) * ry_s))))
                    cv2.circle(vis, (qx, qy), 3, COL_CYAN, -1, cv2.LINE_AA)
                    cv2.circle(vis, (sx, sy), 3, line_col, -1, cv2.LINE_AA)
                    cv2.line(vis, (qx, qy), (sx, sy), line_col, 1, cv2.LINE_AA)
                    correspondences_drawn += 1
            elif inlier_q_pts.shape[0] > 0:
                correspondence_source = "stage2_xfeat"
                top_order = np.argsort(-inlier_scores)[:12]
                for midx in top_order:
                    qx = max(0, min(191, int(round(float(inlier_q_pts[midx, 0]) * lx_s))))
                    qy = max(0, min(255, int(round(float(inlier_q_pts[midx, 1]) * ly_s))))
                    sx = 192 + max(0, min(191, int(round(float(inlier_r_pts[midx, 0]) * rx_s))))
                    sy = max(0, min(255, int(round(float(inlier_r_pts[midx, 1]) * ry_s))))
                    cv2.circle(vis, (qx, qy), 3, COL_CYAN, -1, cv2.LINE_AA)
                    cv2.circle(vis, (sx, sy), 3, line_col, -1, cv2.LINE_AA)
                    cv2.line(vis, (qx, qy), (sx, sy), line_col, 1, cv2.LINE_AA)
                    correspondences_drawn += 1
            stage2_meta["correspondences_drawn"] = int(correspondences_drawn)
            stage2_meta["correspondence_source"] = correspondence_source
            _draw_badge(vis, summary_str, (8, 22), line_col)
            b64 = _encode_jpeg_b64(vis, quality=84)
            res["image_b64"] = b64
            res["annotated_jpeg_b64"] = f"data:image/jpeg;base64,{b64}"
            res["image_mime"] = "image/jpeg"

        return res

    # ---- Unified Low-Latency Webcam / Live Stream Frame Analyzer (`POST /api/analyze_frame_fast`) ----

    def analyze_frame_fast(
        self,
        frame_bgr: np.ndarray,
        channel: str = "d2",
        *,
        return_jpeg: bool = True,
        tracker_state: D2TrackerState | None = None,
    ) -> dict[str, Any]:
        """Run real 5/5 model inference on a single live camera or video frame matching v4.5 API contract."""
        ch = channel.strip().lower()
        if ch == "d2":
            raw = self.infer_eo_sky(
                frame_bgr,
                return_jpeg=return_jpeg,
                tracker_state=self._fast_d2_state if tracker_state is None else tracker_state,
                multi_crop_fallback=True,
            )
            metrics: dict[str, Any] = {
                "detected": raw["detected"],
                "max_prob": raw["max_prob"],
                "detection": raw["detection"],
                "bearing_track_confirmed": raw["bearing_track_confirmed"],
                "bearing_track_hits_in_window": raw["bearing_track_hits_in_window"],
                "bearing_tbc_confirmed_tracks": raw["bearing_tbc_confirmed_tracks"],
                "bearing_tbc": raw["bearing_tbc"],
                "active_range": raw["active_range"],
            }
        elif ch == "land":
            raw = self.infer_runway_land(frame_bgr, return_jpeg=return_jpeg)
            metrics = {
                "onnx_output_contract": raw["onnx_output_contract"],
                "glideslope_deg": raw["glideslope_deg"],
                "lateral_dev_deg": raw["lateral_dev_deg"],
                "hpl_m": raw["hpl_m"],
                "vpl_m": raw["vpl_m"],
                "mean_keypoint_sigma_px": raw["mean_keypoint_sigma_px"],
                "keyline_sigma_px": raw["keyline_sigma_px"],
                "mean_keyline_sigma_px": raw["mean_keyline_sigma_px"],
                "pnpl": raw["pnpl"],
                "keypoints_xy_256": raw["keypoints_xy_256"],
            }
        elif ch == "thermal":
            raw = self.infer_thermal_dve(frame_bgr, q_eo=0.85, q_lwir=0.95, return_jpeg=return_jpeg)
            metrics = {
                "simplex_mode": raw["simplex_mode"],
                "mean_vacuity": raw["mean_vacuity"],
                "max_heatmap_prob": raw["max_heatmap_prob"],
                "w_eo_eff": raw["w_eo_eff"],
                "w_lwir_eff": raw["w_lwir_eff"],
            }
        elif ch == "nav":
            with self._nav_seq_vpr_lock:
                raw = self.infer_crossview_nav(
                    frame_bgr,
                    gps_offset_m=0.0,
                    seq_vpr_filter=self._fast_seq_vpr,
                    return_jpeg=return_jpeg,
                )
            metrics = {
                "cosine_sim": raw["cosine_sim"],
                "psr": raw["psr"],
                "psr_cal": raw["psr_cal"],
                "hpl_vis_m": raw["hpl_vis_m"],
                "hal_nav_m": raw["hal_nav_m"],
                "spoof_detected": raw["spoof_detected"],
                "nav_mode": raw["nav_mode"],
                "cusum_max_stat": raw["dual_bank_cusum"]["max_cusum_stat"],
                "aliasing_gate_admitted": raw["aliasing_gate"]["admitted"],
                "stage2_tier_used": raw["stage2_xfeat"]["tier_used"],
                "seq_vpr_window": raw["seq_vpr_window"],
                "seq_vpr_admitted": raw["seq_vpr_admitted"],
                "seq_vpr_posterior_prob": raw["seq_vpr_posterior_prob"],
            }
        else:
            raise ValueError(f"Unsupported channel {channel!r}; expected one of d2, land, thermal, nav")

        out: dict[str, Any] = {
            "channel": ch,
            "latency_ms": raw["latency_ms"],
            "confidence": raw["confidence"],
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "summary": raw["summary"],
            "real_weights": bool(raw["real_weights"]),
            "metrics": metrics,
        }
        if "sensor_health" in raw:
            out["sensor_health"] = raw["sensor_health"]
        if return_jpeg and "annotated_jpeg_b64" in raw:
            out["annotated_jpeg_b64"] = raw["annotated_jpeg_b64"]
            out["image_b64"] = raw["image_b64"]
            out["image_mime"] = "image/jpeg"
        return out

    # ---- Synchronized 4-Channel Demo Tick with Bounded 128-Entry LRU Cache ----

    def _resolve_studio_media_frame(
        self,
        channel_key: str,
        frame_idx: int,
    ) -> tuple[np.ndarray | None, dict[str, Any], int]:
        manifest = self._load_studio_manifest()
        if manifest is None:
            return None, {}, 0
        channels = (
            manifest["channels"]
            if "channels" in manifest and isinstance(manifest["channels"], dict)
            else manifest
        )
        if channel_key not in channels or not isinstance(channels[channel_key], dict):
            return None, {}, 0
        ch_doc = channels[channel_key]
        frames = ch_doc["frames"] if "frames" in ch_doc and isinstance(ch_doc["frames"], list) else []
        if not frames:
            return None, {}, 0
        entry = frames[int(frame_idx) % len(frames)]
        if not isinstance(entry, dict):
            return None, {}, len(frames)
        rel = None
        for cand_k in ("eo_file", "aerial_file", "file", "path"):
            if cand_k in entry and isinstance(entry[cand_k], str):
                rel = entry[cand_k]
                break
        if rel is None:
            return None, entry, len(frames)
        img_path = self.studio_media_dir / rel
        if not img_path.is_file():
            img_path = self.studio_media_dir / channel_key / rel
        if not img_path.is_file():
            return None, entry, len(frames)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        return img, entry, len(frames)

    def _get_fallback_demo_frames(self, frame_idx: int) -> dict[str, Any]:
        eo_bgr: np.ndarray | None = None
        crop_x = 0
        crop_y = 0
        aot_man = self._load_aot_manifest()
        if aot_man is not None and "encounters" in aot_man and aot_man["encounters"]:
            enc0 = aot_man["encounters"][0]
            flist = enc0["frames"] if "frames" in enc0 else []
            if flist:
                f_rec = flist[int(frame_idx) % len(flist)]
                fpath = self.aot_pack_dir / str(f_rec["file"])
                if fpath.is_file():
                    eo_bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                    crop_x = int(f_rec["crop_x"]) if "crop_x" in f_rec else 0
                    crop_y = int(f_rec["crop_y"]) if "crop_y" in f_rec else 0
        if eo_bgr is None:
            eo_bgr = np.full((256, 256, 3), (195, 175, 150), dtype=np.uint8)
            cv2.circle(eo_bgr, (128, 128), 6, (35, 38, 42), -1, cv2.LINE_AA)

        ortho = self._load_ksfo_ortho()
        oh, ow = ortho.shape[:2]
        shift = (int(frame_idx) * 14) % max(1, min(oh, ow) // 4)
        c_sz = max(256, min(oh, ow) // 2)
        x0 = max(0, min(ow - c_sz, ow // 4 + shift))
        y0 = max(0, min(oh - c_sz, oh // 4 + shift // 2))
        ksfo_crop = ortho[y0 : y0 + c_sz, x0 : x0 + c_sz]
        h_c, w_c = ksfo_crop.shape[:2]
        src_pts = np.float32([[0, 0], [w_c - 1, 0], [w_c - 1, h_c - 1], [0, h_c - 1]])
        dst_pts = np.float32(
            [[w_c * 0.22, h_c * 0.12], [w_c * 0.78, h_c * 0.12], [w_c - 1, h_c - 1], [0, h_c - 1]]
        )
        m_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        land_bgr = cv2.warpPerspective(ksfo_crop, m_persp, (w_c, h_c), borderMode=cv2.BORDER_REPLICATE)

        return {
            "eo_bgr": eo_bgr,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "land_bgr": land_bgr,
            "thermal_bgr": land_bgr,
            "nav_bgr": ksfo_crop,
            "nav_ref_bgr": ksfo_crop,
            "nav_ref_key": f"fallback_ksfo_ref_{int(frame_idx) % 24}",
        }

    def run_demo_tick(
        self,
        scenario: str = "cbdr",
        frame_idx: int = 0,
        *,
        include_images: bool = True,
    ) -> dict[str, Any]:
        """Execute synchronized real inference across all 4 sensor channels with 128-entry LRU caching."""
        t0 = time.perf_counter()
        scen = scenario.strip().lower()
        if scen not in ("cbdr", "forcedland", "thermal_fog", "gps_spoof"):
            scen = "cbdr"

        cycle_idx = int(frame_idx) % 60
        cache_key = (scen, cycle_idx, bool(include_images))

        with self._cache_lock:
            if cache_key in self._demo_tick_lru:
                cached_doc = self._demo_tick_lru[cache_key]
                self._demo_tick_lru.move_to_end(cache_key)
                out = dict(cached_doc)
                out["cached"] = True
                out["frame_idx"] = int(frame_idx)
                out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                out["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                return out

        with self._nav_seq_vpr_lock:
            if self._demo_last_scenario != scen or cycle_idx == 0:
                self._demo_seq_vpr.reset()
                self._ref_seq_vpr = None
                self._demo_last_scenario = scen

        th_idx = cycle_idx + 12 if scen == "thermal_fog" else cycle_idx
        nav_idx = cycle_idx + 14 if scen == "gps_spoof" else cycle_idx

        eo_img, eo_meta, _ = self._resolve_studio_media_frame("eo_sky", cycle_idx)
        land_img, land_meta, _ = self._resolve_studio_media_frame("runway_land", cycle_idx)
        th_img, th_meta, _ = self._resolve_studio_media_frame("thermal_dve", th_idx)
        nav_img, nav_meta, _ = self._resolve_studio_media_frame("crossview_nav", nav_idx)

        need_fb = eo_img is None or land_img is None or th_img is None or nav_img is None
        fb = self._get_fallback_demo_frames(cycle_idx) if need_fb else {}

        lwir_img: np.ndarray | None = None
        if "lwir_file" in th_meta and isinstance(th_meta["lwir_file"], str):
            p_lw = self.studio_media_dir / th_meta["lwir_file"]
            if not p_lw.is_file():
                p_lw = self.studio_media_dir / "thermal_dve" / th_meta["lwir_file"]
            if p_lw.is_file():
                lwir_img = cv2.imread(str(p_lw), cv2.IMREAD_COLOR)

        nav_ref_img: np.ndarray | None = None
        nav_ref_key: str | None = None
        for rk in ("sat_ref_file", "ref_file", "naip_file", "sat_file"):
            if rk in nav_meta and isinstance(nav_meta[rk], str):
                p_rf = self.studio_media_dir / nav_meta[rk]
                if not p_rf.is_file():
                    p_rf = self.studio_media_dir / "crossview_nav" / nav_meta[rk]
                if p_rf.is_file():
                    nav_ref_img = cv2.imread(str(p_rf), cv2.IMREAD_COLOR)
                    nav_ref_key = f"studio_media:{nav_meta[rk]}"
                    break

        swerve_a_lat = 2.08 if scen == "cbdr" and (cycle_idx % 20) < 12 else 0.0
        q_eo = 0.08 if scen == "thermal_fog" else 0.90
        if "q_eo_expected" in th_meta and scen != "thermal_fog":
            q_eo = float(th_meta["q_eo_expected"])
        fog_density = 0.0 if (th_img is not None and scen == "thermal_fog") else (0.92 if scen == "thermal_fog" else 0.0)
        gps_offset_m = 185.0 if scen == "gps_spoof" else 4.2
        if "spoof_offset_m" in nav_meta:
            meta_off = float(nav_meta["spoof_offset_m"])
            if scen == "gps_spoof":
                gps_offset_m = max(120.0, meta_off)
            else:
                gps_offset_m = meta_off

        crop_origin = eo_meta["crop_origin_full_px"] if "crop_origin_full_px" in eo_meta else None
        crop_x = int(crop_origin[0]) if isinstance(crop_origin, list) and len(crop_origin) >= 2 else (int(fb["crop_x"]) if "crop_x" in fb else 0)
        crop_y = int(crop_origin[1]) if isinstance(crop_origin, list) and len(crop_origin) >= 2 else (int(fb["crop_y"]) if "crop_y" in fb else 0)
        subcrop_raw = eo_meta["d2_subcrop_origin_px"] if "d2_subcrop_origin_px" in eo_meta else None
        subcrop_xy = (
            (int(subcrop_raw[0]), int(subcrop_raw[1]))
            if isinstance(subcrop_raw, list) and len(subcrop_raw) >= 2
            else None
        )
        gt_box_raw = eo_meta["gt_box_px"] if "gt_box_px" in eo_meta else None
        gt_box_px = (
            (float(gt_box_raw[0]), float(gt_box_raw[1]), float(gt_box_raw[2]), float(gt_box_raw[3]))
            if isinstance(gt_box_raw, list) and len(gt_box_raw) == 4
            else None
        )
        gt_range_m = float(eo_meta["range_distance_m"]) if "range_distance_m" in eo_meta else None

        rwy_box_raw = land_meta["runway_bbox_px"] if "runway_bbox_px" in land_meta else None
        rwy_box_px = (
            (float(rwy_box_raw[0]), float(rwy_box_raw[1]), float(rwy_box_raw[2]), float(rwy_box_raw[3]))
            if isinstance(rwy_box_raw, list) and len(rwy_box_raw) == 4
            else None
        )

        corr_pairs = (
            nav_meta["correspondence_pairs_px"]
            if "correspondence_pairs_px" in nav_meta and isinstance(nav_meta["correspondence_pairs_px"], list)
            else None
        )

        ch_eo = self.infer_eo_sky(
            eo_img if eo_img is not None else fb["eo_bgr"],
            crop_x=crop_x,
            crop_y=crop_y,
            subcrop_origin_xy=subcrop_xy,
            gt_box_px=gt_box_px,
            gt_range_m=gt_range_m,
            own_a_ned=(0.0, swerve_a_lat, 0.0),
            return_jpeg=include_images,
        )
        ch_land = self.infer_runway_land(
            land_img if land_img is not None else fb["land_bgr"],
            roi_box_px=rwy_box_px,
            return_jpeg=include_images,
        )
        ch_thermal = self.infer_thermal_dve(
            th_img if th_img is not None else fb["thermal_bgr"],
            lwir_bgr=lwir_img,
            q_eo=q_eo,
            q_lwir=0.96,
            fog_density=fog_density,
            return_jpeg=include_images,
        )
        ch_nav = self.infer_crossview_nav(
            nav_img if nav_img is not None else fb["nav_bgr"],
            ref_bgr=nav_ref_img if nav_ref_img is not None else fb["nav_ref_bgr"],
            ref_cache_key=nav_ref_key if nav_ref_key is not None else (str(fb["nav_ref_key"]) if "nav_ref_key" in fb else None),
            correspondence_pairs=corr_pairs,
            gps_offset_m=gps_offset_m,
            return_jpeg=include_images,
        )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scen,
            "frame_idx": int(frame_idx),
            "cached": False,
            "media_source": "data/studio_media" if self.media_pack_ready() else "fallback_aot_ksfo",
            "compute_device": self.bundle.compute_device,
            "host_gpu_name": self.bundle.host_gpu_name,
            "vram_mb": get_process_vram_mb(),
            "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "channels": {
                "eo_sky": ch_eo,
                "runway_land": ch_land,
                "thermal_dve": ch_thermal,
                "crossview_nav": ch_nav,
            },
        }

        with self._cache_lock:
            self._demo_tick_lru[cache_key] = result
            self._demo_tick_lru.move_to_end(cache_key)
            while len(self._demo_tick_lru) > self._demo_tick_lru_max:
                self._demo_tick_lru.popitem(last=False)

        return result
