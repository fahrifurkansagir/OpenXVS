"""Real-world avionics sensor health, timing integrity, and pinhole distortion front-end.

Dependency-light (`numpy` + `cv2` + Python stdlib only — zero PyTorch / SciPy dependency),
deterministic, strictly fail-closed, and aligned 1:1 with the OpenXVS Rust engine semantics
(`apps/studio/src-tauri/src/rust_engine.rs` & `xvs.v2.xvs_thermal.model.ThermalSimplexMode`).

Modules provided:
  1. `LwirNucFreezeMonitor`: Detects frozen/stuck LWIR microbolometer NUC/FFC shutters
     (`MAD < 0.25` for `>= 3` consecutive ticks OR `> 250 ms` static duration) -> fail-closed
     `q_lwir = 0.0` (`LWIR_DEGRADED_EO_PRIMARY` or `DUAL_DVE_SVS_FALLBACK`).
  2. `evaluate_frame_optical_quality`: Measures Laplacian focus/vibration-blur variance,
     over-exposure/sun-glare saturation (`>= 250`), under-exposure/lens-cap occlusion (`<= 5`),
     and robust `P95 - P05` contrast dynamic range -> calibrated `q_effective` in `[0.0, 1.0]`.
  3. `TimestampSkewGuard`: Rejects non-monotonic timestamps (`dt <= 0`) and excessive
     camera-IMU latency skew (`> 200 ms`) fail-closed.
  4. `PinholeIntrinsics`, `distort_keypoints_pinhole`, `undistort_keypoints_pinhole`,
     `undistort_frame_pinhole`: Exact Brown-Conrady `(k1, k2, p1, p2, k3)` Camera RDF
     rectification prior to Stage-2 `XFeat` SQPnP (`< 1e-6 px` round-trip accuracy).
  5. `SensorHealthMonitor`: Thread-safe unified telemetry facade for `StudioInferenceEngine`.

Coordinate standard: Camera `RDF` `(u, v)` / Local `NED` / `WGS-84 ECEF` (`0` internal ENU).
Zero unannotated `.get(..., numeric_default)` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import threading
from typing import Any

import cv2
import numpy as np

# Canonical real-world sensor health thresholds (aligned 1:1 with Rust `rust_engine.rs`)
DEFAULT_LWIR_NUC_MAD_THRESHOLD: float = 0.25
DEFAULT_LWIR_NUC_MAX_FREEZE_TICKS: int = 3
DEFAULT_LWIR_NUC_MAX_FREEZE_MS: float = 250.0

DEFAULT_OVER_EXPOSE_PIXEL: float = 250.0
DEFAULT_UNDER_EXPOSE_PIXEL: float = 5.0
DEFAULT_EO_LAPLACIAN_HALF_VAR: float = 6.0
DEFAULT_LWIR_LAPLACIAN_HALF_VAR: float = 5.0
DEFAULT_CONTRAST_REF_RANGE: float = 30.0
DEFAULT_SIMPLEX_TAU_OK: float = 0.5
DEFAULT_SIMPLEX_TAU_VAC: float = 0.7

DEFAULT_FPN_DEAD_LINE_FLAG_FRAC: float = 0.03
DEFAULT_FPN_DEAD_LINE_FAIL_FRAC: float = 0.10
DEFAULT_NOISE_STD_MIN: float = 15.0
DEFAULT_NOISE_AUTOCORR_MIN: float = 0.25
DEFAULT_LWIR_LOCAL_PEAK_MIN_DN: float = 40.0
DEFAULT_LWIR_LOCAL_PEAK_DELTA_DN: float = 25.0
DEFAULT_LWIR_LOCAL_PEAK_SNR_MIN: float = 6.0

DEFAULT_MAX_CAMERA_IMU_SKEW_MS: float = 200.0


class ThermalSimplexMode(IntEnum):
    """ASTM F3269-21 Simplex RTA modes for EO/LWIR fusion (exact parity with `xvs_thermal.model`)."""

    NOMINAL_FUSION = 0
    EO_DEGRADED_LWIR_PRIMARY = 1
    LWIR_DEGRADED_EO_PRIMARY = 2
    DUAL_DVE_SVS_FALLBACK = 3


def select_thermal_simplex_mode_fail_closed(
    q_eo: float,
    q_lwir: float,
    vacuity: float,
    tau_ok: float = DEFAULT_SIMPLEX_TAU_OK,
    tau_vac: float = DEFAULT_SIMPLEX_TAU_VAC,
) -> ThermalSimplexMode:
    """Deterministic fail-closed ASTM F3269-21 Simplex RTA mode selector (no PyTorch required)."""
    if not (
        math.isfinite(tau_ok)
        and (0.0 < tau_ok <= 1.0)
        and math.isfinite(tau_vac)
        and (0.0 < tau_vac <= 1.0)
    ):
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK

    vac_ok = math.isfinite(vacuity) and (0.0 <= vacuity < tau_vac)
    if not vac_ok:
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK

    eo_ok = math.isfinite(q_eo) and (tau_ok <= q_eo <= 1.0)
    lwir_ok = math.isfinite(q_lwir) and (tau_ok <= q_lwir <= 1.0)

    if not eo_ok and not lwir_ok:
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
    if eo_ok and lwir_ok:
        return ThermalSimplexMode.NOMINAL_FUSION
    if lwir_ok and not eo_ok:
        return ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
    return ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY


def _to_grayscale_float64(frame: np.ndarray) -> np.ndarray | None:
    """Convert 2D/3D BGR/4D BGRA frame to finite 2D float64 intensity on `[0, 255]` scale."""
    arr = np.asarray(frame)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return None
    if arr.ndim == 2:
        gray = arr.astype(np.float64, copy=False)
    elif arr.shape[2] == 1:
        gray = arr[:, :, 0].astype(np.float64, copy=False)
    elif arr.shape[2] in (3, 4):
        if np.issubdtype(arr.dtype, np.floating):
            bgr = arr[:, :, :3].astype(np.float64, copy=False)
            if not bool(np.all(np.isfinite(bgr))):
                return None
            gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        else:
            u8 = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
            code = cv2.COLOR_BGRA2GRAY if arr.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(u8, code).astype(np.float64)
    else:
        return None

    if gray.size == 0 or not bool(np.all(np.isfinite(gray))):
        return None
    return gray


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


# ---------------------------------------------------------------------------
# 1. LWIR NUC Shutter Freeze Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LwirNucFreezeStatus:
    """Result of a single LWIR microbolometer NUC shutter freeze check."""

    frozen: bool
    freeze_ticks: int
    freeze_duration_ms: float
    mad: float | None
    q_lwir_gate: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lwir_nuc_frozen": bool(self.frozen),
            "lwir_freeze_ticks": int(self.freeze_ticks),
            "lwir_freeze_duration_ms": round(float(self.freeze_duration_ms), 3),
            "lwir_mad": round(float(self.mad), 5) if self.mad is not None and math.isfinite(self.mad) else None,
            "q_lwir_gate": round(float(self.q_lwir_gate), 4),
            "reason": self.reason,
        }


class LwirNucFreezeMonitor:
    """Detects stuck LWIR microbolometer NUC/FFC shutter or frozen thermal DMA frames.

    Triggers fail-closed `q_lwir_gate = 0.0` whenever consecutive thermal frames exhibit
    mean absolute pixel difference `MAD < mad_threshold` (default `0.25` intensity levels)
    for `>= max_freeze_ticks` (default `3` ticks) OR `> max_freeze_duration_ms` (default
    `250.0 ms`). Recovers cleanly once live thermal motion (`MAD >= mad_threshold`) resumes
    for `recovery_ticks` consecutive frames.
    """

    def __init__(
        self,
        *,
        mad_threshold: float = DEFAULT_LWIR_NUC_MAD_THRESHOLD,
        max_freeze_ticks: int = DEFAULT_LWIR_NUC_MAX_FREEZE_TICKS,
        max_freeze_duration_ms: float = DEFAULT_LWIR_NUC_MAX_FREEZE_MS,
        recovery_ticks: int = 1,
    ) -> None:
        if not math.isfinite(mad_threshold) or mad_threshold <= 0.0:
            raise ValueError(f"mad_threshold must be finite and > 0, got {mad_threshold}")
        if max_freeze_ticks < 1:
            raise ValueError(f"max_freeze_ticks must be >= 1, got {max_freeze_ticks}")
        if not math.isfinite(max_freeze_duration_ms) or max_freeze_duration_ms <= 0.0:
            raise ValueError(
                f"max_freeze_duration_ms must be finite and > 0, got {max_freeze_duration_ms}"
            )
        if recovery_ticks < 1:
            raise ValueError(f"recovery_ticks must be >= 1, got {recovery_ticks}")

        self.mad_threshold = float(mad_threshold)
        self.max_freeze_ticks = int(max_freeze_ticks)
        self.max_freeze_duration_ms = float(max_freeze_duration_ms)
        self.recovery_ticks = int(recovery_ticks)

        self._prev_gray: np.ndarray | None = None
        self._last_ts_ms: float | None = None
        self._freeze_ticks: int = 0
        self._freeze_duration_ms: float = 0.0
        self._live_recovery_streak: int = 0
        self._frozen: bool = False

    def reset(self) -> None:
        self._prev_gray = None
        self._last_ts_ms = None
        self._freeze_ticks = 0
        self._freeze_duration_ms = 0.0
        self._live_recovery_streak = 0
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return bool(self._frozen)

    @property
    def freeze_ticks(self) -> int:
        return int(self._freeze_ticks)

    @property
    def freeze_duration_ms(self) -> float:
        return float(self._freeze_duration_ms)

    def update(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: float | None = None,
        dt_ms: float | None = None,
    ) -> LwirNucFreezeStatus:
        """Ingest one thermal frame and return deterministic fail-closed NUC freeze status."""
        gray_f64 = _to_grayscale_float64(frame)
        if gray_f64 is None:
            self._frozen = True
            self._live_recovery_streak = 0
            return LwirNucFreezeStatus(
                frozen=True,
                freeze_ticks=self._freeze_ticks,
                freeze_duration_ms=self._freeze_duration_ms,
                mad=None,
                q_lwir_gate=0.0,
                reason="NON_FINITE_OR_EMPTY_LWIR_FRAME",
            )

        gray = gray_f64.astype(np.float32, copy=False)
        valid_ts = timestamp_ms is not None and math.isfinite(float(timestamp_ms))

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            if valid_ts:
                self._last_ts_ms = float(timestamp_ms)
            self._freeze_ticks = 0
            self._freeze_duration_ms = 0.0
            self._live_recovery_streak = 0
            self._frozen = False
            return LwirNucFreezeStatus(
                frozen=False,
                freeze_ticks=0,
                freeze_duration_ms=0.0,
                mad=None,
                q_lwir_gate=1.0,
                reason="INITIAL_FRAME",
            )

        if gray.shape != self._prev_gray.shape:
            self._prev_gray = gray.copy()
            if valid_ts:
                self._last_ts_ms = float(timestamp_ms)
            self._freeze_ticks = 0
            self._freeze_duration_ms = 0.0
            self._live_recovery_streak = 0
            self._frozen = False
            return LwirNucFreezeStatus(
                frozen=False,
                freeze_ticks=0,
                freeze_duration_ms=0.0,
                mad=None,
                q_lwir_gate=1.0,
                reason="SHAPE_RESET",
            )

        mad = float(np.mean(np.abs(gray - self._prev_gray)))

        step_dt_ms = 0.0
        if valid_ts:
            ts_val = float(timestamp_ms)
            if self._last_ts_ms is not None and ts_val > self._last_ts_ms:
                step_dt_ms = ts_val - self._last_ts_ms
            self._last_ts_ms = (
                ts_val if self._last_ts_ms is None else max(self._last_ts_ms, ts_val)
            )
        elif dt_ms is not None and math.isfinite(float(dt_ms)) and float(dt_ms) > 0.0:
            step_dt_ms = float(dt_ms)

        if mad < self.mad_threshold:
            self._freeze_ticks += 1
            self._freeze_duration_ms += step_dt_ms
            self._live_recovery_streak = 0
            if (
                self._freeze_ticks >= self.max_freeze_ticks
                or self._freeze_duration_ms > self.max_freeze_duration_ms
            ):
                self._frozen = True
        else:
            if self._frozen:
                self._live_recovery_streak += 1
                if self._live_recovery_streak >= self.recovery_ticks:
                    self._frozen = False
                    self._freeze_ticks = 0
                    self._freeze_duration_ms = 0.0
                    self._live_recovery_streak = 0
            else:
                self._freeze_ticks = 0
                self._freeze_duration_ms = 0.0
                self._live_recovery_streak = 0

        self._prev_gray = gray.copy()

        if self._frozen:
            reason = "LWIR_NUC_SHUTTER_FROZEN"
        elif self._freeze_ticks > 0:
            reason = "STATIC_WATCH"
        else:
            reason = "OK"

        return LwirNucFreezeStatus(
            frozen=self._frozen,
            freeze_ticks=self._freeze_ticks,
            freeze_duration_ms=self._freeze_duration_ms,
            mad=mad,
            q_lwir_gate=0.0 if self._frozen else 1.0,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# 2. EO / Thermal Optical Quality Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameOpticalQuality:
    """Calibrated single-frame EO or LWIR optical integrity telemetry."""

    modality: str
    laplacian_var: float
    blur_score: float
    over_exposed_ratio: float
    under_exposed_ratio: float
    glare_score: float
    occlusion_score: float
    dynamic_range_p95_p05: float
    contrast_score: float
    q_effective: float
    degraded: bool
    flags: tuple[str, ...]
    dead_line_fraction: float = 0.0
    fpn_score: float = 1.0
    spatial_autocorr: float = 1.0
    noise_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "laplacian_var": round(float(self.laplacian_var), 3),
            "blur_score": round(float(self.blur_score), 4),
            "over_exposed_ratio": round(float(self.over_exposed_ratio), 4),
            "under_exposed_ratio": round(float(self.under_exposed_ratio), 4),
            "glare_score": round(float(self.glare_score), 4),
            "occlusion_score": round(float(self.occlusion_score), 4),
            "dynamic_range_p95_p05": round(float(self.dynamic_range_p95_p05), 3),
            "contrast_score": round(float(self.contrast_score), 4),
            "dead_line_fraction": round(float(self.dead_line_fraction), 4),
            "fpn_score": round(float(self.fpn_score), 4),
            "spatial_autocorr": round(float(self.spatial_autocorr), 4),
            "noise_score": round(float(self.noise_score), 4),
            "q_effective": round(float(self.q_effective), 4),
            "degraded": bool(self.degraded),
            "flags": list(self.flags),
        }


def _detect_dead_line_fpn(gray: np.ndarray) -> tuple[float, float]:
    """Detect stuck/dead sensor readout rows & columns (1-2 px FPN stripes).

    Uses per-row/column median + stuck-pixel fraction (`>= 70%` within `+-1.5 DN` of line median)
    paired with a two-sided local neighbor spike jump (`>= 15 DN` against both sides within 2 px),
    which catches dead lines even when perpendicular dead columns intersect them, while yielding
    exact `0.0` on natural step edges, horizons, and letterbox borders.
    """
    h, w = gray.shape[:2]
    if h < 6 or w < 6 or float(np.std(gray)) < 1.0:
        return 0.0, 1.0

    row_med = np.median(gray, axis=1)
    row_stuck = np.mean(np.abs(gray - row_med[:, None]) <= 1.5, axis=1) >= 0.70
    r_up = np.maximum(np.abs(row_med[2:-2] - row_med[1:-3]), np.abs(row_med[2:-2] - row_med[:-4]))
    r_dn = np.maximum(np.abs(row_med[2:-2] - row_med[3:-1]), np.abs(row_med[2:-2] - row_med[4:]))
    dead_rows = row_stuck[2:-2] & (np.minimum(r_up, r_dn) >= 15.0)
    f_rows = float(np.sum(dead_rows)) / float(h)

    col_med = np.median(gray, axis=0)
    col_stuck = np.mean(np.abs(gray - col_med[None, :]) <= 1.5, axis=0) >= 0.70
    c_lt = np.maximum(np.abs(col_med[2:-2] - col_med[1:-3]), np.abs(col_med[2:-2] - col_med[:-4]))
    c_rt = np.maximum(np.abs(col_med[2:-2] - col_med[3:-1]), np.abs(col_med[2:-2] - col_med[4:]))
    dead_cols = col_stuck[2:-2] & (np.minimum(c_lt, c_rt) >= 15.0)
    f_cols = float(np.sum(dead_cols)) / float(w)

    dead_frac = min(1.0, f_rows + f_cols)
    if dead_frac < DEFAULT_FPN_DEAD_LINE_FLAG_FRAC:
        return dead_frac, 1.0
    span = max(DEFAULT_FPN_DEAD_LINE_FAIL_FRAC - DEFAULT_FPN_DEAD_LINE_FLAG_FRAC, 1e-6)
    fpn_score = _clamp01(0.45 * (1.0 - (dead_frac - DEFAULT_FPN_DEAD_LINE_FLAG_FRAC) / span))
    return dead_frac, fpn_score


def _detect_white_noise_fault(gray: np.ndarray) -> tuple[float, float]:
    """Detect spatially uncorrelated impulse/white/binary noise bypassing Laplacian focus checks.

    Natural EO/LWIR scenes have high spatial lag-1 autocorrelation (`> 0.65`), whereas uncorrelated
    random/binary sensor noise has `~0.0` while exhibiting large pixel std (`> 15 DN`).
    """
    h, w = gray.shape[:2]
    std_val = float(np.std(gray))
    if h < 4 or w < 4 or not math.isfinite(std_val) or std_val <= DEFAULT_NOISE_STD_MIN:
        return 1.0, 1.0

    g0 = gray - float(np.mean(gray))
    var = float(np.mean(g0 * g0))
    if not math.isfinite(var) or var <= 1e-6:
        return 1.0, 1.0

    ac_h = float(np.mean(g0[:, :-1] * g0[:, 1:]) / var)
    ac_v = float(np.mean(g0[:-1, :] * g0[1:, :]) / var)
    ac = max(ac_h, ac_v)
    if not math.isfinite(ac):
        return 0.0, 0.0
    if ac < DEFAULT_NOISE_AUTOCORR_MIN:
        noise_score = _clamp01(0.10 * max(0.0, ac) / DEFAULT_NOISE_AUTOCORR_MIN)
        return ac, noise_score
    return ac, 1.0


def evaluate_frame_optical_quality(
    frame: np.ndarray,
    *,
    modality: str = "eo",
    over_expose_threshold: float = DEFAULT_OVER_EXPOSE_PIXEL,
    under_expose_threshold: float = DEFAULT_UNDER_EXPOSE_PIXEL,
    laplacian_half_var: float | None = None,
    contrast_ref_range: float = DEFAULT_CONTRAST_REF_RANGE,
    degraded_threshold: float = DEFAULT_SIMPLEX_TAU_OK,
) -> FrameOpticalQuality:
    """Compute calibrated optical/radiometric quality `q_effective in [0.0, 1.0]`.

    Uses multiplicative weakest-link composition across six physical channels:
      - Focus / vibration-blur sharpness: `lap_var / (lap_var + half_var)` via 2nd-order Laplacian.
      - Sun-glare / over-exposure saturation (`pixel >= 250`): zero penalty up to 2%, collapses to
        `0.0` at `>= 50%` saturated pixels.
      - Lens-cap / under-exposure occlusion (`pixel <= 5`): zero penalty up to 10%, collapses to
        `0.0` at `>= 75%` crushed-black pixels — EXCEPT for clean cold/night-sky LWIR frames
        containing a crisp localized hot target (`peak >= 40 DN`, `peak - P50 >= 25 DN`, SNR `>= 6`).
      - Contrast dynamic range (`P95 - P05`, augmented by localized hot-target contrast in LWIR).
      - Dead-pixel row/column FPN stripe detector (`>= 3%` flagged, `>= 10%` collapses `q <= 0.15`).
      - Spatially uncorrelated white/binary noise detector (`std > 15 DN`, lag-1 autocorr `< 0.25`).

    Empty or non-finite frames fail closed with `q_effective = 0.0`.
    """
    mod = modality.strip().lower()
    if mod not in ("eo", "lwir"):
        raise ValueError(f"Unsupported modality {modality!r}; expected 'eo' or 'lwir'")

    half_var = (
        float(laplacian_half_var)
        if laplacian_half_var is not None
        else (DEFAULT_EO_LAPLACIAN_HALF_VAR if mod == "eo" else DEFAULT_LWIR_LAPLACIAN_HALF_VAR)
    )
    if not math.isfinite(half_var) or half_var <= 0.0:
        raise ValueError(f"laplacian_half_var must be finite and > 0, got {half_var}")
    if not math.isfinite(contrast_ref_range) or contrast_ref_range <= 0.0:
        raise ValueError(f"contrast_ref_range must be finite and > 0, got {contrast_ref_range}")

    gray = _to_grayscale_float64(frame)
    if gray is None:
        return FrameOpticalQuality(
            modality=mod,
            laplacian_var=0.0,
            blur_score=0.0,
            over_exposed_ratio=0.0,
            under_exposed_ratio=1.0,
            glare_score=0.0,
            occlusion_score=0.0,
            dynamic_range_p95_p05=0.0,
            contrast_score=0.0,
            q_effective=0.0,
            degraded=True,
            flags=("NON_FINITE_OR_EMPTY_FRAME",),
            dead_line_fraction=0.0,
            fpn_score=0.0,
            spatial_autocorr=0.0,
            noise_score=0.0,
        )

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not math.isfinite(lap_var) or lap_var < 0.0:
        lap_var = 0.0
    blur_score = _clamp01(lap_var / (lap_var + half_var)) if lap_var > 0.0 else 0.0

    over_ratio = float(np.mean(gray >= float(over_expose_threshold)))
    under_ratio = float(np.mean(gray <= float(under_expose_threshold)))

    glare_score = _clamp01(1.0 - max(0.0, over_ratio - 0.02) / 0.48)
    occlusion_score = _clamp01(1.0 - max(0.0, under_ratio - 0.10) / 0.65)

    p05, p50, p95, p999 = np.percentile(gray, [5.0, 50.0, 95.0, 99.9])
    dyn_range = max(0.0, float(p95 - p05)) if (math.isfinite(p05) and math.isfinite(p95)) else 0.0

    # Dead-pixel row/column FPN & white/binary noise detectors (both EO and LWIR)
    dead_line_frac, fpn_score = _detect_dead_line_fpn(gray)
    spatial_autocorr, noise_score = _detect_white_noise_fault(gray)
    sensor_clean_for_peak = (
        dead_line_frac < DEFAULT_FPN_DEAD_LINE_FLAG_FRAC
        and noise_score >= float(degraded_threshold)
    )

    eff_dyn_range = dyn_range
    if mod == "lwir" and sensor_clean_for_peak and math.isfinite(p50) and math.isfinite(p999):
        # 2x2 morphological min rejects isolated 1-pixel hot noise while preserving >=2x2 hot targets
        if gray.shape[0] >= 2 and gray.shape[1] >= 2:
            min2x2_max = float(
                np.max(
                    np.minimum(
                        np.minimum(gray[:-1, :-1], gray[:-1, 1:]),
                        np.minimum(gray[1:, :-1], gray[1:, 1:]),
                    )
                )
            )
        else:
            min2x2_max = float(p999)
        peak_val = max(float(p999), min2x2_max)
        peak_above_bg = max(0.0, peak_val - float(p50))
        bg_mad = float(np.median(np.abs(gray - float(p50))))
        bg_sigma = max(1.4826 * bg_mad, 1.0)
        peak_snr = peak_above_bg / bg_sigma
        has_localized_lwir_signal = (
            peak_val >= DEFAULT_LWIR_LOCAL_PEAK_MIN_DN
            and peak_above_bg >= DEFAULT_LWIR_LOCAL_PEAK_DELTA_DN
            and (peak_snr >= DEFAULT_LWIR_LOCAL_PEAK_SNR_MIN or peak_above_bg >= 35.0)
        )
        if has_localized_lwir_signal:
            eff_dyn_range = max(dyn_range, peak_above_bg)
            occlusion_score = 1.0

    contrast_score = _clamp01(eff_dyn_range / float(contrast_ref_range))

    q_effective = _clamp01(
        blur_score * contrast_score * glare_score * occlusion_score * fpn_score * noise_score
    )

    flags: list[str] = []
    if blur_score < degraded_threshold:
        flags.append("MOTION_OR_FOCUS_BLUR")
    if glare_score < degraded_threshold:
        flags.append("SUN_GLARE_OVEREXPOSED")
    if occlusion_score < degraded_threshold:
        flags.append("LENS_CAP_OR_UNDEREXPOSED")
    if contrast_score < degraded_threshold:
        flags.append("LOW_CONTRAST_DYNAMIC_RANGE")
    if dead_line_frac >= DEFAULT_FPN_DEAD_LINE_FLAG_FRAC:
        flags.append("DEAD_PIXEL_FPN_FAULT")
    if noise_score < degraded_threshold:
        flags.append("SENSOR_NOISE_FAULT")

    return FrameOpticalQuality(
        modality=mod,
        laplacian_var=lap_var,
        blur_score=blur_score,
        over_exposed_ratio=over_ratio,
        under_exposed_ratio=under_ratio,
        glare_score=glare_score,
        occlusion_score=occlusion_score,
        dynamic_range_p95_p05=dyn_range,
        contrast_score=contrast_score,
        q_effective=q_effective,
        degraded=q_effective < float(degraded_threshold),
        flags=tuple(flags),
        dead_line_fraction=dead_line_frac,
        fpn_score=fpn_score,
        spatial_autocorr=spatial_autocorr,
        noise_score=noise_score,
    )


# ---------------------------------------------------------------------------
# 3. Monotonic Timestamp & Camera-IMU Skew Guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimestampGuardStatus:
    """Result of monotonic frame timestamp and camera-IMU latency skew verification."""

    valid: bool
    monotonic_ok: bool
    skew_ok: bool
    dt_ms: float | None
    skew_ms: float
    signed_skew_ms: float
    q_time_gate: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_valid": bool(self.valid),
            "timestamp_monotonic_ok": bool(self.monotonic_ok),
            "timestamp_skew_ok": bool(self.skew_ok),
            "dt_ms": round(float(self.dt_ms), 3) if self.dt_ms is not None and math.isfinite(self.dt_ms) else None,
            "timestamp_skew_ms": round(float(self.skew_ms), 3) if math.isfinite(self.skew_ms) else None,
            "signed_skew_ms": round(float(self.signed_skew_ms), 3) if math.isfinite(self.signed_skew_ms) else None,
            "q_time_gate": round(float(self.q_time_gate), 4),
            "timestamp_reasons": list(self.reasons),
        }


class TimestampSkewGuard:
    """Fail-closed monotonic timestamp and camera-IMU synchronization skew guard.

    Rejects:
      - Non-finite or negative timestamps,
      - Non-monotonic camera timestamps (`dt_ms <= 0.0`, covering duplicates & backward jumps),
      - Excessive camera-IMU skew (`abs(frame_ts_ms - imu_ts_ms) > max_skew_ms`, default `200.0 ms`).
    """

    def __init__(self, *, max_skew_ms: float = DEFAULT_MAX_CAMERA_IMU_SKEW_MS) -> None:
        if not math.isfinite(max_skew_ms) or max_skew_ms <= 0.0:
            raise ValueError(f"max_skew_ms must be finite and > 0, got {max_skew_ms}")
        self.max_skew_ms = float(max_skew_ms)
        self._last_frame_ts_ms: float | None = None

    def reset(self) -> None:
        self._last_frame_ts_ms = None

    @property
    def last_frame_ts_ms(self) -> float | None:
        return self._last_frame_ts_ms

    def check(
        self,
        frame_ts_ms: float,
        imu_ts_ms: float | None = None,
    ) -> TimestampGuardStatus:
        """Validate frame timestamp monotonicity and optional camera-IMU skew."""
        reasons: list[str] = []
        if not math.isfinite(frame_ts_ms) or frame_ts_ms < 0.0:
            return TimestampGuardStatus(
                valid=False,
                monotonic_ok=False,
                skew_ok=False,
                dt_ms=None,
                skew_ms=0.0,
                signed_skew_ms=0.0,
                q_time_gate=0.0,
                reasons=("NON_FINITE_OR_NEGATIVE_FRAME_TIMESTAMP",),
            )

        ts_f = float(frame_ts_ms)
        monotonic_ok = True
        dt_ms: float | None = None
        if self._last_frame_ts_ms is not None:
            dt_ms = ts_f - self._last_frame_ts_ms
            if dt_ms <= 0.0:
                monotonic_ok = False
                reasons.append("NON_MONOTONIC_TIMESTAMP")

        skew_ok = True
        skew_ms = 0.0
        signed_skew_ms = 0.0
        if imu_ts_ms is not None:
            if not math.isfinite(imu_ts_ms) or imu_ts_ms < 0.0:
                skew_ok = False
                reasons.append("NON_FINITE_OR_NEGATIVE_IMU_TIMESTAMP")
            else:
                signed_skew_ms = ts_f - float(imu_ts_ms)
                skew_ms = abs(signed_skew_ms)
                if skew_ms > self.max_skew_ms:
                    skew_ok = False
                    reasons.append("CAMERA_IMU_SKEW_EXCEEDED")

        # Only advance monotonic reference on strictly increasing finite frame timestamps
        if monotonic_ok:
            self._last_frame_ts_ms = ts_f

        valid = monotonic_ok and skew_ok
        return TimestampGuardStatus(
            valid=valid,
            monotonic_ok=monotonic_ok,
            skew_ok=skew_ok,
            dt_ms=dt_ms,
            skew_ms=skew_ms,
            signed_skew_ms=signed_skew_ms,
            q_time_gate=1.0 if valid else 0.0,
            reasons=tuple(reasons),
        )


# ---------------------------------------------------------------------------
# 4. Camera Intrinsics & Brown-Conrady Distortion Rectification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole camera intrinsics with OpenCV/Brown-Conrady distortion `(k1, k2, p1, p2, k3)`."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self) -> None:
        vals = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("All PinholeIntrinsics parameters must be finite")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError(f"Focal lengths fx, fy must be > 0, got ({self.fx}, {self.fy})")

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(
            [float(self.k1), float(self.k2), float(self.p1), float(self.p2), float(self.k3)],
            dtype=np.float64,
        )


def distort_keypoints_pinhole(
    keypoints_uv: np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> np.ndarray:
    """Apply exact Brown-Conrady forward distortion `(u, v) -> (u_dist, v_dist)` in Camera RDF."""
    pts = np.asarray(keypoints_uv, dtype=np.float64)
    if pts.ndim == 1:
        if pts.shape[0] != 2:
            raise ValueError(f"Expected shape (2,) or (N, 2), got {pts.shape}")
        single = True
        pts_2d = pts[np.newaxis, :]
    elif pts.ndim == 2 and pts.shape[1] == 2:
        single = False
        pts_2d = pts
    else:
        raise ValueError(f"Expected shape (2,) or (N, 2), got {pts.shape}")

    if pts_2d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if not bool(np.all(np.isfinite(pts_2d))):
        raise ValueError("keypoints_uv contains non-finite values")

    x = (pts_2d[:, 0] - float(intrinsics.cx)) / float(intrinsics.fx)
    y = (pts_2d[:, 1] - float(intrinsics.cy)) / float(intrinsics.fy)

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    rad = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6
    xd = x * rad + 2.0 * intrinsics.p1 * x * y + intrinsics.p2 * (r2 + 2.0 * x * x)
    yd = y * rad + intrinsics.p1 * (r2 + 2.0 * y * y) + 2.0 * intrinsics.p2 * x * y

    ud = float(intrinsics.fx) * xd + float(intrinsics.cx)
    vd = float(intrinsics.fy) * yd + float(intrinsics.cy)
    out = np.column_stack([ud, vd])
    return out[0] if single else out


def undistort_keypoints_pinhole(
    keypoints_uv_dist: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    max_iter: int = 25,
    tol: float = 1e-12,
    return_normalized: bool = False,
) -> np.ndarray:
    """Exact Brown-Conrady `(k1, k2, p1, p2, k3)` keypoint undistortion via damped Newton-Raphson.

    Inverts OpenCV/Brown-Conrady radial + tangential distortion in `float64` for real camera
    frames prior to Stage-2 `XFeat` homography / SQPnP pose estimation. Achieves `< 1e-6 px`
    round-trip accuracy on realistic wide-angle and tangential lenses with zero SciPy dependency.

    Parameters
    ----------
    keypoints_uv_dist : np.ndarray
        Distorted pixel coordinates of shape `(N, 2)` or `(2,)`.
    intrinsics : PinholeIntrinsics
        Camera focal lengths, principal point, and `(k1, k2, p1, p2, k3)`.
    max_iter : int
        Maximum Newton-Raphson iterations (default `25`).
    tol : float
        Normalized-plane convergence tolerance (default `1e-12`).
    return_normalized : bool
        If True, return undistorted Camera RDF normalized rays `(x, y)` directly for SQPnP;
        if False (default), return rectified pixel coordinates `(u, v)`.
    """
    pts = np.asarray(keypoints_uv_dist, dtype=np.float64)
    if pts.ndim == 1:
        if pts.shape[0] != 2:
            raise ValueError(f"Expected shape (2,) or (N, 2), got {pts.shape}")
        single = True
        pts_2d = pts[np.newaxis, :]
    elif pts.ndim == 2 and pts.shape[1] == 2:
        single = False
        pts_2d = pts
    else:
        raise ValueError(f"Expected shape (2,) or (N, 2), got {pts.shape}")

    if pts_2d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if not bool(np.all(np.isfinite(pts_2d))):
        raise ValueError("keypoints_uv_dist contains non-finite values")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    xd = (pts_2d[:, 0] - float(intrinsics.cx)) / float(intrinsics.fx)
    yd = (pts_2d[:, 1] - float(intrinsics.cy)) / float(intrinsics.fy)

    k1 = float(intrinsics.k1)
    k2 = float(intrinsics.k2)
    p1 = float(intrinsics.p1)
    p2 = float(intrinsics.p2)
    k3 = float(intrinsics.k3)

    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0 and k3 == 0.0:
        if return_normalized:
            out = np.column_stack([xd, yd])
            return out[0] if single else out
        return pts_2d[0].copy() if single else pts_2d.copy()

    x = xd.copy()
    y = yd.copy()

    for _ in range(int(max_iter)):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        drad_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4

        fx_res = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd
        fy_res = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd

        j11 = rad + 2.0 * x * x * drad_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
        j12 = 2.0 * x * y * drad_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
        j22 = rad + 2.0 * y * y * drad_dr2 + 6.0 * p1 * y + 2.0 * p2 * x

        det = j11 * j22 - j12 * j12
        valid_det = np.abs(det) >= 1e-15
        safe_det = np.where(valid_det, det, 1.0)

        dx = np.where(valid_det, (j22 * fx_res - j12 * fy_res) / safe_det, 0.0)
        dy = np.where(valid_det, (-j12 * fx_res + j11 * fy_res) / safe_det, 0.0)

        # Damped step cap protects stability on extreme wide-angle periphery
        step_norm = np.hypot(dx, dy)
        scale = np.where(step_norm > 0.5, 0.5 / np.maximum(step_norm, 1e-15), 1.0)
        dx *= scale
        dy *= scale

        x -= dx
        y -= dy

        if float(np.max(step_norm * scale)) < float(tol):
            break

    if return_normalized:
        out = np.column_stack([x, y])
    else:
        u = float(intrinsics.fx) * x + float(intrinsics.cx)
        v = float(intrinsics.fy) * y + float(intrinsics.cy)
        out = np.column_stack([u, v])
    return out[0] if single else out


def undistort_frame_pinhole(frame: np.ndarray, intrinsics: PinholeIntrinsics) -> np.ndarray:
    """Rectify a full 2D/3D camera image using OpenCV Brown-Conrady `cv2.undistort`."""
    arr = np.asarray(frame)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError(f"Expected non-empty 2D/3D image array, got shape {arr.shape}")
    k_mat = intrinsics.camera_matrix
    return cv2.undistort(arr, k_mat, intrinsics.dist_coeffs, None, k_mat)


# ---------------------------------------------------------------------------
# 5. Unified Thread-Safe SensorHealthMonitor Facade
# ---------------------------------------------------------------------------


class SensorHealthMonitor:
    """Unified thread-safe EO, LWIR, and timing integrity monitor for OpenXVS Studio & Bridge."""

    def __init__(
        self,
        *,
        lwir_mad_threshold: float = DEFAULT_LWIR_NUC_MAD_THRESHOLD,
        lwir_max_freeze_ticks: int = DEFAULT_LWIR_NUC_MAX_FREEZE_TICKS,
        lwir_max_freeze_duration_ms: float = DEFAULT_LWIR_NUC_MAX_FREEZE_MS,
        lwir_recovery_ticks: int = 1,
        max_camera_imu_skew_ms: float = DEFAULT_MAX_CAMERA_IMU_SKEW_MS,
    ) -> None:
        self._lock = threading.Lock()
        self.lwir_nuc_monitor = LwirNucFreezeMonitor(
            mad_threshold=lwir_mad_threshold,
            max_freeze_ticks=lwir_max_freeze_ticks,
            max_freeze_duration_ms=lwir_max_freeze_duration_ms,
            recovery_ticks=lwir_recovery_ticks,
        )
        self.eo_time_guard = TimestampSkewGuard(max_skew_ms=max_camera_imu_skew_ms)
        self.thermal_time_guard = TimestampSkewGuard(max_skew_ms=max_camera_imu_skew_ms)
        self.nav_time_guard = TimestampSkewGuard(max_skew_ms=max_camera_imu_skew_ms)
        self._last_snapshot: dict[str, Any] = {
            "eo_blur_score": 1.0,
            "eo_glare_ratio": 0.0,
            "lwir_nuc_frozen": False,
            "lwir_freeze_ticks": 0,
            "timestamp_skew_ms": 0.0,
            "timestamp_valid": True,
            "q_eo_effective": 1.0,
            "q_lwir_effective": 1.0,
            "thermal_mode": ThermalSimplexMode.NOMINAL_FUSION.name,
            "thermal_mode_id": int(ThermalSimplexMode.NOMINAL_FUSION),
        }

    def reset(self) -> None:
        with self._lock:
            self.lwir_nuc_monitor.reset()
            self.eo_time_guard.reset()
            self.thermal_time_guard.reset()
            self.nav_time_guard.reset()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_snapshot)

    def _check_optional_time(
        self,
        guard: TimestampSkewGuard,
        frame_ts_ms: float | None,
        imu_ts_ms: float | None,
    ) -> TimestampGuardStatus:
        if frame_ts_ms is None:
            return TimestampGuardStatus(
                valid=True,
                monotonic_ok=True,
                skew_ok=True,
                dt_ms=None,
                skew_ms=0.0,
                signed_skew_ms=0.0,
                q_time_gate=1.0,
                reasons=(),
            )
        return guard.check(frame_ts_ms, imu_ts_ms=imu_ts_ms)

    def assess_eo_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        channel: str = "eo_sky",
        frame_ts_ms: float | None = None,
        imu_ts_ms: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate EO optical quality + optional timestamp/IMU skew guard for EO/Nav channels."""
        opt = evaluate_frame_optical_quality(frame_bgr, modality="eo")
        with self._lock:
            guard = self.nav_time_guard if channel == "crossview_nav" else self.eo_time_guard
            ts_status = self._check_optional_time(guard, frame_ts_ms, imu_ts_ms)
            q_eff = _clamp01(opt.q_effective * ts_status.q_time_gate)
            reasons = list(opt.flags) + list(ts_status.reasons)
            report: dict[str, Any] = {
                "channel": channel,
                "eo_blur_score": round(opt.blur_score, 4),
                "eo_laplacian_var": round(opt.laplacian_var, 3),
                "eo_glare_ratio": round(opt.over_exposed_ratio, 4),
                "eo_under_exposed_ratio": round(opt.under_exposed_ratio, 4),
                "eo_dynamic_range": round(opt.dynamic_range_p95_p05, 3),
                "timestamp_valid": bool(ts_status.valid),
                "timestamp_monotonic_ok": bool(ts_status.monotonic_ok),
                "timestamp_skew_ok": bool(ts_status.skew_ok),
                "timestamp_skew_ms": round(float(ts_status.skew_ms), 3),
                "q_eo_effective": round(q_eff, 4),
                "degraded": bool(q_eff < DEFAULT_SIMPLEX_TAU_OK or not ts_status.valid),
                "reasons": reasons,
            }
            self._last_snapshot["eo_blur_score"] = report["eo_blur_score"]
            self._last_snapshot["eo_glare_ratio"] = report["eo_glare_ratio"]
            self._last_snapshot["timestamp_skew_ms"] = report["timestamp_skew_ms"]
            self._last_snapshot["timestamp_valid"] = report["timestamp_valid"]
            self._last_snapshot["q_eo_effective"] = report["q_eo_effective"]
            return report

    def assess_thermal_pair(
        self,
        eo_bgr: np.ndarray,
        lwir_frame: np.ndarray,
        *,
        q_eo_nominal: float = 1.0,
        q_lwir_nominal: float = 1.0,
        vacuity: float = 0.0,
        frame_ts_ms: float | None = None,
        imu_ts_ms: float | None = None,
        dt_ms: float | None = None,
        update_nuc_state: bool = True,
    ) -> dict[str, Any]:
        """Evaluate paired EO + LWIR frames, NUC freeze state, timing guard, and Simplex coupling."""
        eo_opt = evaluate_frame_optical_quality(eo_bgr, modality="eo")
        lwir_opt = evaluate_frame_optical_quality(lwir_frame, modality="lwir")

        with self._lock:
            ts_status = self._check_optional_time(self.thermal_time_guard, frame_ts_ms, imu_ts_ms)
            if update_nuc_state:
                nuc_status = self.lwir_nuc_monitor.update(
                    lwir_frame, timestamp_ms=frame_ts_ms, dt_ms=dt_ms
                )
            else:
                nuc_status = LwirNucFreezeStatus(
                    frozen=self.lwir_nuc_monitor.is_frozen,
                    freeze_ticks=self.lwir_nuc_monitor.freeze_ticks,
                    freeze_duration_ms=self.lwir_nuc_monitor.freeze_duration_ms,
                    mad=None,
                    q_lwir_gate=0.0 if self.lwir_nuc_monitor.is_frozen else 1.0,
                    reason=(
                        "LWIR_NUC_SHUTTER_FROZEN" if self.lwir_nuc_monitor.is_frozen else "OK"
                    ),
                )

            q_eo_nom = _clamp01(float(q_eo_nominal))
            q_lwir_nom = _clamp01(float(q_lwir_nominal))

            q_eo_eff = _clamp01(min(q_eo_nom, eo_opt.q_effective) * ts_status.q_time_gate)
            q_lwir_eff = _clamp01(
                min(q_lwir_nom, lwir_opt.q_effective)
                * nuc_status.q_lwir_gate
                * ts_status.q_time_gate
            )

            mode = select_thermal_simplex_mode_fail_closed(
                q_eo=q_eo_eff,
                q_lwir=q_lwir_eff,
                vacuity=float(vacuity),
            )

            reasons: list[str] = []
            for f in eo_opt.flags:
                reasons.append(f"EO_{f}")
            for f in lwir_opt.flags:
                reasons.append(f"LWIR_{f}")
            if nuc_status.frozen:
                reasons.append(nuc_status.reason)
            reasons.extend(ts_status.reasons)

            report: dict[str, Any] = {
                "channel": "thermal_dve",
                "eo_blur_score": round(eo_opt.blur_score, 4),
                "eo_laplacian_var": round(eo_opt.laplacian_var, 3),
                "eo_glare_ratio": round(eo_opt.over_exposed_ratio, 4),
                "eo_under_exposed_ratio": round(eo_opt.under_exposed_ratio, 4),
                "eo_dynamic_range": round(eo_opt.dynamic_range_p95_p05, 3),
                "lwir_blur_score": round(lwir_opt.blur_score, 4),
                "lwir_laplacian_var": round(lwir_opt.laplacian_var, 3),
                "lwir_glare_ratio": round(lwir_opt.over_exposed_ratio, 4),
                "lwir_under_exposed_ratio": round(lwir_opt.under_exposed_ratio, 4),
                "lwir_dynamic_range": round(lwir_opt.dynamic_range_p95_p05, 3),
                "lwir_nuc_frozen": bool(nuc_status.frozen),
                "lwir_freeze_ticks": int(nuc_status.freeze_ticks),
                "lwir_freeze_duration_ms": round(float(nuc_status.freeze_duration_ms), 3),
                "lwir_mad": (
                    round(float(nuc_status.mad), 5)
                    if nuc_status.mad is not None and math.isfinite(nuc_status.mad)
                    else None
                ),
                "lwir_nuc_reason": nuc_status.reason,
                "timestamp_valid": bool(ts_status.valid),
                "timestamp_monotonic_ok": bool(ts_status.monotonic_ok),
                "timestamp_skew_ok": bool(ts_status.skew_ok),
                "timestamp_skew_ms": round(float(ts_status.skew_ms), 3),
                "q_eo_effective": round(q_eo_eff, 4),
                "q_lwir_effective": round(q_lwir_eff, 4),
                "thermal_mode": mode.name,
                "thermal_mode_id": int(mode),
                "reasons": reasons,
            }

            self._last_snapshot = {
                "eo_blur_score": report["eo_blur_score"],
                "eo_glare_ratio": report["eo_glare_ratio"],
                "lwir_nuc_frozen": report["lwir_nuc_frozen"],
                "lwir_freeze_ticks": report["lwir_freeze_ticks"],
                "timestamp_skew_ms": report["timestamp_skew_ms"],
                "timestamp_valid": report["timestamp_valid"],
                "q_eo_effective": report["q_eo_effective"],
                "q_lwir_effective": report["q_lwir_effective"],
                "thermal_mode": report["thermal_mode"],
                "thermal_mode_id": report["thermal_mode_id"],
            }
            return report
