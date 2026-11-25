"""Deterministic Multi-Frame M-of-N Track-Before-Confirm & EO/LWIR Cross-Modal Bearing Tracker.

Implements OpenXVS v5.2 Track-Before-Confirm (`XVS-D2` + `XVS-ThermalNet` bearing track fusion):
  - Deterministic 4-state constant-velocity Kalman filter in Camera RDF / NED bearing space:
      `x = [az_rad, el_rad, az_dot_rad_s, el_dot_rad_s]^T`
    with exact Joseph-form covariance updates and analytic 2x2 Mahalanobis innovation gating.
  - M-of-N (`3-of-5` confirm, `5`-frame confirmed coast delete) weak-hit association
    (`p_weak >= 0.15` vs single-frame `0.45`), recovering continuous tracks on low-SNR small
    fixed-wing Encounter `884.0`-class trajectories while rejecting transient sky clutter.
  - Cross-modal EO / LWIR precision-weighted track fusion incorporating calibrated sensor health
    weights `(q_eo, q_lwir)` — strictly fail-closed on `NaN`, `Inf`, out-of-range health weights,
    non-finite detections, non-positive `dt_s`, or dual-blind (`q_eo == 0 && q_lwir == 0`).

Dependency-light (Python stdlib + `numpy` only — zero PyTorch / SciPy / OpenCV required).
Coordinate convention: strictly Camera RDF / NED bearings (`az_rad` positive right/East,
`el_rad` positive up = `-Down`, `0` internal ENU).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any

import numpy as np

# Canonical OpenXVS v5.2 Track-Before-Confirm defaults
DEFAULT_CONFIRM_M: int = 3
DEFAULT_CONFIRM_N: int = 5
DEFAULT_MAX_COAST_FRAMES: int = 5
DEFAULT_WEAK_SCORE_THR: float = 0.15
DEFAULT_SINGLE_FRAME_SCORE_THR: float = 0.45
DEFAULT_MAHALANOBIS_GATE_SQ: float = 9.2103  # 2-DoF chi-square 99.0% gate
DEFAULT_TENTATIVE_ASSOC_RAD: float = 0.016  # ~0.92 deg (~43 px at 2690 px) for 1-hit initiation
DEFAULT_MAX_ASSOC_RAD: float = 0.030  # ~1.72 deg (~81 px at 2690 px) for tracked targets
DEFAULT_NMS_RADIUS_RAD: float = 0.010  # ~0.57 deg (~27 px at 2690 px) same-modality angular NMS
DEFAULT_SIMPLEX_TAU_OK: float = 0.5
DEFAULT_INIT_POS_STD_RAD: float = 0.0020
DEFAULT_INIT_VEL_STD_RAD_S: float = 0.065
DEFAULT_PROCESS_ACCEL_STD_RAD_S2: float = 0.025
DEFAULT_EO_MEAS_STD_RAD: float = 0.0012
DEFAULT_LWIR_MEAS_STD_RAD: float = 0.0035
DEFAULT_MAX_TRACKS: int = 64

_HALF_PI: float = 0.5 * math.pi
_TWO_PI: float = 2.0 * math.pi
_MIN_DET_S: float = 1.0e-18
_MIN_HEALTH_SCALE: float = 1.0e-3


class TrackStatus(IntEnum):
    """Lifecycle state of a bearing-space track."""

    TENTATIVE = 0
    CONFIRMED = 1


class CrossModalSimplexMode(IntEnum):
    """ASTM F3269-21 Simplex RTA sensor mode (exact parity with `ThermalSimplexMode`)."""

    NOMINAL_FUSION = 0
    EO_DEGRADED_LWIR_PRIMARY = 1
    LWIR_DEGRADED_EO_PRIMARY = 2
    DUAL_DVE_SVS_FALLBACK = 3


def wrap_pi(angle_rad: float) -> float:
    """Wrap finite angle in radians strictly into `(-pi, +pi]`."""
    if not math.isfinite(angle_rad):
        return 0.0
    wrapped = (angle_rad + math.pi) % _TWO_PI - math.pi
    if wrapped <= -math.pi:
        wrapped += _TWO_PI
    return float(wrapped)


def clamp_elevation(el_rad: float) -> float:
    """Clamp elevation angle strictly to `[-pi/2, +pi/2]`."""
    if not math.isfinite(el_rad):
        return 0.0
    if el_rad < -_HALF_PI:
        return -_HALF_PI
    if el_rad > _HALF_PI:
        return _HALF_PI
    return float(el_rad)


def sanitize_health_weight(q: float) -> float:
    """Fail-closed sensor health sanitizer: any `NaN`, `Inf`, `< 0`, or `> 1` maps strictly to `0.0`."""
    if isinstance(q, bool) or not isinstance(q, (int, float, np.floating, np.integer)):
        return 0.0
    qf = float(q)
    if not math.isfinite(qf) or qf < 0.0 or qf > 1.0:
        return 0.0
    return qf


def select_cross_modal_simplex_mode(
    q_eo: float,
    q_lwir: float,
    tau_ok: float = DEFAULT_SIMPLEX_TAU_OK,
) -> CrossModalSimplexMode:
    """Determine ASTM F3269-21 EO/LWIR Simplex mode strictly fail-closed on non-finite health."""
    qe = sanitize_health_weight(q_eo)
    ql = sanitize_health_weight(q_lwir)
    if not math.isfinite(tau_ok) or tau_ok <= 0.0 or tau_ok > 1.0:
        return CrossModalSimplexMode.DUAL_DVE_SVS_FALLBACK
    eo_ok = qe >= tau_ok
    lwir_ok = ql >= tau_ok
    if eo_ok and lwir_ok:
        return CrossModalSimplexMode.NOMINAL_FUSION
    if lwir_ok and not eo_ok:
        return CrossModalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
    if eo_ok and not lwir_ok:
        return CrossModalSimplexMode.LWIR_DEGRADED_EO_PRIMARY
    return CrossModalSimplexMode.DUAL_DVE_SVS_FALLBACK


def pixel_to_bearing_rad(
    u_px: float,
    v_px: float,
    width_px: float,
    height_px: float,
    focal_px: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[float, float]:
    """Convert Camera RDF pixel `(u_px, v_px)` to NED boresight bearing `(az_rad, el_rad)`.

    Matches `StudioInferenceEngine.infer_eo_sky` (`az = atan2(u - cx, f)`, `el = atan2(cy - v, f)`).
    """
    vals = (u_px, v_px, width_px, height_px, focal_px)
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("All pixel/camera arguments must be finite")
    if float(focal_px) <= 0.0 or float(width_px) <= 0.0 or float(height_px) <= 0.0:
        raise ValueError("focal_px, width_px, and height_px must be > 0")
    cx_f = 0.5 * float(width_px) if cx is None else float(cx)
    cy_f = 0.5 * float(height_px) if cy is None else float(cy)
    if not (math.isfinite(cx_f) and math.isfinite(cy_f)):
        raise ValueError("Principal point (cx, cy) must be finite")
    az = math.atan2(float(u_px) - cx_f, float(focal_px))
    el = math.atan2(cy_f - float(v_px), float(focal_px))
    return float(az), float(el)


def bearing_to_pixel_uv(
    az_rad: float,
    el_rad: float,
    width_px: float,
    height_px: float,
    focal_px: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[float, float]:
    """Project NED boresight bearing `(az_rad, el_rad)` back into Camera RDF pixel `(u_px, v_px)`."""
    if not all(math.isfinite(float(v)) for v in (az_rad, el_rad, width_px, height_px, focal_px)):
        raise ValueError("All bearing/camera arguments must be finite")
    if float(focal_px) <= 0.0 or float(width_px) <= 0.0 or float(height_px) <= 0.0:
        raise ValueError("focal_px, width_px, and height_px must be > 0")
    cx_f = 0.5 * float(width_px) if cx is None else float(cx)
    cy_f = 0.5 * float(height_px) if cy is None else float(cy)
    az_c = max(-1.55, min(1.55, float(az_rad)))
    el_c = max(-1.55, min(1.55, float(el_rad)))
    u_px = cx_f + float(focal_px) * math.tan(az_c)
    v_px = cy_f - float(focal_px) * math.tan(el_c)
    return float(u_px), float(v_px)


@dataclass(frozen=True)
class TrackBeforeConfirmConfig:
    """Configuration for deterministic M-of-N bearing-space Track-Before-Confirm."""

    confirm_m: int = DEFAULT_CONFIRM_M
    confirm_n: int = DEFAULT_CONFIRM_N
    max_coast_frames: int = DEFAULT_MAX_COAST_FRAMES
    weak_score_thr: float = DEFAULT_WEAK_SCORE_THR
    single_frame_score_thr: float = DEFAULT_SINGLE_FRAME_SCORE_THR
    mahalanobis_gate_sq: float = DEFAULT_MAHALANOBIS_GATE_SQ
    tentative_assoc_rad: float = DEFAULT_TENTATIVE_ASSOC_RAD
    max_assoc_rad: float = DEFAULT_MAX_ASSOC_RAD
    nms_radius_rad: float = DEFAULT_NMS_RADIUS_RAD
    init_pos_std_rad: float = DEFAULT_INIT_POS_STD_RAD
    init_vel_std_rad_s: float = DEFAULT_INIT_VEL_STD_RAD_S
    process_accel_std_rad_s2: float = DEFAULT_PROCESS_ACCEL_STD_RAD_S2
    default_eo_meas_std_rad: float = DEFAULT_EO_MEAS_STD_RAD
    default_lwir_meas_std_rad: float = DEFAULT_LWIR_MEAS_STD_RAD
    simplex_tau_ok: float = DEFAULT_SIMPLEX_TAU_OK
    max_tracks: int = DEFAULT_MAX_TRACKS
    min_confirm_score: float = DEFAULT_WEAK_SCORE_THR
    min_modality_health: float = 0.20
    below_horizon_el_margin_rad: float = 0.0
    below_horizon_score_scale: float = 0.50
    reject_static_clutter: bool = False
    static_bearing_rate_rad_s: float = 0.003
    static_min_size_growth_frac: float = 0.05

    def __post_init__(self) -> None:
        if self.confirm_m < 1 or self.confirm_n < self.confirm_m:
            raise ValueError(
                f"Require 1 <= confirm_m <= confirm_n, got M={self.confirm_m}, N={self.confirm_n}"
            )
        if self.max_coast_frames < 1:
            raise ValueError(f"max_coast_frames must be >= 1, got {self.max_coast_frames}")
        if not (math.isfinite(self.weak_score_thr) and 0.0 < self.weak_score_thr <= 1.0):
            raise ValueError(f"weak_score_thr must be in (0, 1], got {self.weak_score_thr}")
        if not (
            math.isfinite(self.single_frame_score_thr)
            and self.weak_score_thr <= self.single_frame_score_thr <= 1.0
        ):
            raise ValueError(
                f"single_frame_score_thr must be in [weak_score_thr, 1], got {self.single_frame_score_thr}"
            )
        if not (math.isfinite(self.nms_radius_rad) and self.nms_radius_rad >= 0.0):
            raise ValueError(f"nms_radius_rad must be finite and >= 0, got {self.nms_radius_rad}")
        for name in (
            "mahalanobis_gate_sq",
            "tentative_assoc_rad",
            "max_assoc_rad",
            "init_pos_std_rad",
            "init_vel_std_rad_s",
            "process_accel_std_rad_s2",
            "default_eo_meas_std_rad",
            "default_lwir_meas_std_rad",
            "simplex_tau_ok",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {val}")
        if self.max_tracks < 1:
            raise ValueError(f"max_tracks must be >= 1, got {self.max_tracks}")
        if not (
            math.isfinite(self.min_confirm_score)
            and self.weak_score_thr <= self.min_confirm_score <= 1.0
        ):
            raise ValueError(
                f"min_confirm_score must be in [weak_score_thr, 1], got {self.min_confirm_score}"
            )
        if not (math.isfinite(self.min_modality_health) and 0.0 <= self.min_modality_health <= 1.0):
            raise ValueError(
                f"min_modality_health must be in [0, 1], got {self.min_modality_health}"
            )
        if not (math.isfinite(self.below_horizon_el_margin_rad) and self.below_horizon_el_margin_rad >= 0.0):
            raise ValueError(
                f"below_horizon_el_margin_rad must be finite and >= 0, got {self.below_horizon_el_margin_rad}"
            )
        if not (math.isfinite(self.below_horizon_score_scale) and 0.0 <= self.below_horizon_score_scale <= 1.0):
            raise ValueError(
                f"below_horizon_score_scale must be in [0, 1], got {self.below_horizon_score_scale}"
            )
        if not (math.isfinite(self.static_bearing_rate_rad_s) and self.static_bearing_rate_rad_s >= 0.0):
            raise ValueError(
                f"static_bearing_rate_rad_s must be finite and >= 0, got {self.static_bearing_rate_rad_s}"
            )
        if not (math.isfinite(self.static_min_size_growth_frac) and self.static_min_size_growth_frac >= 0.0):
            raise ValueError(
                f"static_min_size_growth_frac must be finite and >= 0, got {self.static_min_size_growth_frac}"
            )

    @property
    def max_tentative_coast(self) -> int:
        """Maximum consecutive misses a tentative track can endure and still achieve M-of-N."""
        return (self.confirm_n - self.confirm_m) + 1


@dataclass(frozen=True)
class BearingDetection:
    """Single-frame EO or LWIR detection in Camera RDF / NED bearing space."""

    az_rad: float
    el_rad: float
    score: float
    modality: str = "eo"
    sigma_az_rad: float | None = None
    sigma_el_rad: float | None = None
    size_px: float | None = None
    u_px: float | None = None
    v_px: float | None = None


@dataclass(frozen=True)
class _SanitizedDetection:
    det_idx: int
    az_rad: float
    el_rad: float
    raw_score: float
    eff_score: float
    modality: str
    health: float
    sigma_az_eff: float
    sigma_el_eff: float
    size_px: float | None


@dataclass(frozen=True)
class BearingTrackSnapshot:
    """Immutable deterministic snapshot of a bearing-space Kalman track."""

    track_id: int
    status: TrackStatus
    az_rad: float
    el_rad: float
    az_dot_rad_s: float
    el_dot_rad_s: float
    sigma_az_rad: float
    sigma_el_rad: float
    sigma_az_dot_rad_s: float
    sigma_el_dot_rad_s: float
    score: float
    age_frames: int
    total_hits: int
    window_hits: int
    window_len: int
    coast_frames: int
    eo_hits: int
    lwir_hits: int
    cross_modal_fused: bool
    last_modalities: tuple[str, ...]
    size_px: float | None

    @property
    def is_confirmed(self) -> bool:
        return self.status == TrackStatus.CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "status": self.status.name,
            "az_rad": round(float(self.az_rad), 6),
            "el_rad": round(float(self.el_rad), 6),
            "az_dot_rad_s": round(float(self.az_dot_rad_s), 6),
            "el_dot_rad_s": round(float(self.el_dot_rad_s), 6),
            "sigma_az_rad": round(float(self.sigma_az_rad), 6),
            "sigma_el_rad": round(float(self.sigma_el_rad), 6),
            "score": round(float(self.score), 4),
            "age_frames": int(self.age_frames),
            "total_hits": int(self.total_hits),
            "window_hits": int(self.window_hits),
            "window_len": int(self.window_len),
            "coast_frames": int(self.coast_frames),
            "eo_hits": int(self.eo_hits),
            "lwir_hits": int(self.lwir_hits),
            "cross_modal_fused": bool(self.cross_modal_fused),
            "last_modalities": list(self.last_modalities),
            "size_px": round(float(self.size_px), 3) if self.size_px is not None else None,
        }


@dataclass(frozen=True)
class TbcStepResult:
    """Output of one deterministic `BearingTrackBeforeConfirm.step` tick."""

    frame_idx: int
    dt_valid: bool
    q_eo: float
    q_lwir: float
    simplex_mode: CrossModalSimplexMode
    dual_blind: bool
    detections_received: int
    detections_accepted: int
    detections_rejected_invalid: int
    detections_rejected_health: int
    confirmed_tracks: tuple[BearingTrackSnapshot, ...]
    tentative_tracks: tuple[BearingTrackSnapshot, ...]
    all_tracks: tuple[BearingTrackSnapshot, ...]
    tracks_confirmed_this_step: int
    tracks_deleted_this_step: int


@dataclass(frozen=True)
class TbcBenchmarkSummary:
    """Quantitative BEFORE (single-frame threshold) vs AFTER (M-of-N TBC) benchmark report."""

    scenario_name: str
    num_frames: int
    single_frame_det_frames: int
    single_frame_continuity: float
    single_frame_false_alarms: int
    tbc_confirmed_frames: int
    tbc_continuity: float
    tbc_continuity_lift: float
    tbc_false_confirmed_tracks: int
    tbc_false_confirmed_frames: int
    tbc_confirm_latency_frames: int | None
    single_frame_bearing_rmse_mrad: float | None
    tbc_bearing_rmse_mrad: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "num_frames": int(self.num_frames),
            "single_frame_det_frames": int(self.single_frame_det_frames),
            "single_frame_continuity": round(float(self.single_frame_continuity), 4),
            "single_frame_false_alarms": int(self.single_frame_false_alarms),
            "tbc_confirmed_frames": int(self.tbc_confirmed_frames),
            "tbc_continuity": round(float(self.tbc_continuity), 4),
            "tbc_continuity_lift": round(float(self.tbc_continuity_lift), 4),
            "tbc_false_confirmed_tracks": int(self.tbc_false_confirmed_tracks),
            "tbc_false_confirmed_frames": int(self.tbc_false_confirmed_frames),
            "tbc_confirm_latency_frames": self.tbc_confirm_latency_frames,
            "single_frame_bearing_rmse_mrad": (
                round(float(self.single_frame_bearing_rmse_mrad), 4)
                if self.single_frame_bearing_rmse_mrad is not None
                else None
            ),
            "tbc_bearing_rmse_mrad": (
                round(float(self.tbc_bearing_rmse_mrad), 4)
                if self.tbc_bearing_rmse_mrad is not None
                else None
            ),
        }


def detections_from_d2_peaks(
    peaks: Sequence[dict[str, Any]],
    *,
    width_px: float,
    height_px: float,
    focal_px: float,
    modality: str = "eo",
    offset_u_px: float = 0.0,
    offset_v_px: float = 0.0,
    cx: float | None = None,
    cy: float | None = None,
) -> list[BearingDetection]:
    """Convert `decode_peaks_local_softargmax` dictionaries into `BearingDetection` list."""
    out: list[BearingDetection] = []
    f_px = float(focal_px)
    for pk in peaks:
        if "u_px" not in pk or "v_px" not in pk or "score" not in pk:
            continue
        u = float(pk["u_px"]) + float(offset_u_px)
        v = float(pk["v_px"]) + float(offset_v_px)
        score = float(pk["score"])
        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(score)):
            continue
        az, el = pixel_to_bearing_rad(u, v, width_px, height_px, f_px, cx=cx, cy=cy)
        sigma_az: float | None = None
        sigma_el: float | None = None
        if "var_u" in pk and math.isfinite(float(pk["var_u"])) and float(pk["var_u"]) > 0.0:
            sigma_az = max(1.0e-4, math.sqrt(float(pk["var_u"])) / f_px)
        if "var_v" in pk and math.isfinite(float(pk["var_v"])) and float(pk["var_v"]) > 0.0:
            sigma_el = max(1.0e-4, math.sqrt(float(pk["var_v"])) / f_px)
        size_px = (
            float(pk["s_mom_px"])
            if "s_mom_px" in pk and math.isfinite(float(pk["s_mom_px"]))
            else None
        )
        out.append(
            BearingDetection(
                az_rad=az,
                el_rad=el,
                score=score,
                modality=modality,
                sigma_az_rad=sigma_az,
                sigma_el_rad=sigma_el,
                size_px=size_px,
                u_px=u,
                v_px=v,
            )
        )
    return out


def _angular_nms(
    dets: Sequence[_SanitizedDetection],
    nms_radius_rad: float,
) -> list[_SanitizedDetection]:
    """Deterministic greedy angular Non-Maximum Suppression within a single sensor modality.

    Prevents extended multi-peak targets (e.g., helicopter rotors/fuselage on stride-2 CenterNet
    heatmaps) from birthing duplicate tracks on the same aircraft.
    """
    if len(dets) <= 1 or nms_radius_rad <= 0.0:
        return list(dets)
    ordered = sorted(dets, key=lambda d: (-d.eff_score, d.det_idx))
    kept: list[_SanitizedDetection] = []
    for cand in ordered:
        suppressed = False
        for prev in kept:
            if math.hypot(wrap_pi(cand.az_rad - prev.az_rad), cand.el_rad - prev.el_rad) <= nms_radius_rad:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


def _fuse_pair_internal(
    eo_det: _SanitizedDetection,
    lwir_det: _SanitizedDetection,
) -> tuple[float, float, float, float, float]:
    """Precision-weighted cross-modal EO + LWIR bearing fusion + noisy-OR score."""
    inv_az_e = 1.0 / (eo_det.sigma_az_eff * eo_det.sigma_az_eff)
    inv_az_l = 1.0 / (lwir_det.sigma_az_eff * lwir_det.sigma_az_eff)
    denom_az = inv_az_e + inv_az_l
    # Wrap relative azimuth delta around EO reference to handle ±pi seam safely
    d_az = wrap_pi(lwir_det.az_rad - eo_det.az_rad)
    az_fused = wrap_pi(eo_det.az_rad + (inv_az_l / denom_az) * d_az)
    sigma_az_fused = math.sqrt(1.0 / denom_az)

    inv_el_e = 1.0 / (eo_det.sigma_el_eff * eo_det.sigma_el_eff)
    inv_el_l = 1.0 / (lwir_det.sigma_el_eff * lwir_det.sigma_el_eff)
    denom_el = inv_el_e + inv_el_l
    el_fused = clamp_elevation((inv_el_e * eo_det.el_rad + inv_el_l * lwir_det.el_rad) / denom_el)
    sigma_el_fused = math.sqrt(1.0 / denom_el)

    p_fused = 1.0 - (1.0 - eo_det.eff_score) * (1.0 - lwir_det.eff_score)
    p_fused = max(0.0, min(1.0, p_fused))
    return az_fused, el_fused, sigma_az_fused, sigma_el_fused, p_fused


class _KalmanBearingTrack:
    """Internal 4-state CV Kalman track (`[az, el, az_dot, el_dot]`)."""

    __slots__ = (
        "track_id",
        "status",
        "x",
        "P",
        "score",
        "score_window",
        "age_frames",
        "total_hits",
        "hit_window",
        "coast_frames",
        "eo_hits",
        "lwir_hits",
        "cross_modal_fused",
        "last_modalities",
        "size_px",
        "init_size_px",
    )

    def __init__(
        self,
        track_id: int,
        az_rad: float,
        el_rad: float,
        score: float,
        sigma_az_rad: float,
        sigma_el_rad: float,
        cfg: TrackBeforeConfirmConfig,
        *,
        modalities: tuple[str, ...],
        size_px: float | None = None,
    ) -> None:
        self.track_id = int(track_id)
        self.status = TrackStatus.TENTATIVE
        self.x = np.array([wrap_pi(az_rad), clamp_elevation(el_rad), 0.0, 0.0], dtype=np.float64)
        p_az = max(cfg.init_pos_std_rad, sigma_az_rad) ** 2
        p_el = max(cfg.init_pos_std_rad, sigma_el_rad) ** 2
        p_v = cfg.init_vel_std_rad_s ** 2
        self.P = np.diag([p_az, p_el, p_v, p_v]).astype(np.float64)
        self.score = float(score)
        self.score_window: deque[float] = deque([float(score)], maxlen=cfg.confirm_n)
        self.age_frames = 1
        self.total_hits = 1
        self.hit_window: deque[bool] = deque([True], maxlen=cfg.confirm_n)
        self.coast_frames = 0
        self.eo_hits = 1 if "eo" in modalities else 0
        self.lwir_hits = 1 if "lwir" in modalities else 0
        self.cross_modal_fused = "eo" in modalities and "lwir" in modalities
        self.last_modalities = modalities
        self.size_px = size_px
        self.init_size_px = size_px
        if cfg.confirm_m == 1 and self.score >= cfg.min_confirm_score:
            self.status = TrackStatus.CONFIRMED

    def predict(self, dt_s: float, accel_std: float) -> None:
        dt = float(dt_s)
        f_mat = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        x_pred = f_mat @ self.x
        x_pred[0] = wrap_pi(float(x_pred[0]))
        x_pred[1] = clamp_elevation(float(x_pred[1]))

        q_var = float(accel_std) * float(accel_std)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = 0.25 * dt4 * q_var
        q13 = 0.5 * dt3 * q_var
        q33 = dt2 * q_var
        q_mat = np.array(
            [
                [q11, 0.0, q13, 0.0],
                [0.0, q11, 0.0, q13],
                [q13, 0.0, q33, 0.0],
                [0.0, q13, 0.0, q33],
            ],
            dtype=np.float64,
        )
        p_pred = f_mat @ self.P @ f_mat.T + q_mat
        p_pred = 0.5 * (p_pred + p_pred.T)
        if np.all(np.isfinite(x_pred)) and np.all(np.isfinite(p_pred)):
            self.x = x_pred
            self.P = p_pred

    def mahalanobis_sq(
        self,
        z_az: float,
        z_el: float,
        sigma_az: float,
        sigma_el: float,
    ) -> tuple[float, float] | None:
        """Compute `(d2_mahalanobis, ang_dist_rad)` fail-closed if covariance is non-finite/singular."""
        y0 = wrap_pi(z_az - float(self.x[0]))
        y1 = z_el - float(self.x[1])
        ang_dist = math.hypot(y0, y1)

        s00 = float(self.P[0, 0]) + sigma_az * sigma_az
        s01 = float(self.P[0, 1])
        s10 = float(self.P[1, 0])
        s11 = float(self.P[1, 1]) + sigma_el * sigma_el
        det = s00 * s11 - s01 * s10
        if not (math.isfinite(det) and det > _MIN_DET_S):
            return None
        inv_det = 1.0 / det
        inv00 = s11 * inv_det
        inv01 = -s01 * inv_det
        inv10 = -s10 * inv_det
        inv11 = s00 * inv_det
        d2 = y0 * (inv00 * y0 + inv01 * y1) + y1 * (inv10 * y0 + inv11 * y1)
        if not (math.isfinite(d2) and d2 >= 0.0):
            return None
        return float(d2), float(ang_dist)

    def kalman_update(
        self,
        z_az: float,
        z_el: float,
        sigma_az: float,
        sigma_el: float,
    ) -> bool:
        """Exact Joseph-form 2D bearing measurement update; fails closed if singular/non-finite."""
        y = np.array(
            [wrap_pi(z_az - float(self.x[0])), z_el - float(self.x[1])],
            dtype=np.float64,
        )
        r_mat = np.array(
            [[sigma_az * sigma_az, 0.0], [0.0, sigma_el * sigma_el]],
            dtype=np.float64,
        )
        s00 = float(self.P[0, 0]) + r_mat[0, 0]
        s01 = float(self.P[0, 1])
        s10 = float(self.P[1, 0])
        s11 = float(self.P[1, 1]) + r_mat[1, 1]
        det = s00 * s11 - s01 * s10
        if not (math.isfinite(det) and det > _MIN_DET_S):
            return False

        s_inv = np.array([[s11, -s01], [-s10, s00]], dtype=np.float64) / det
        # H = [I_2 | 0_2] -> P H^T is simply the first 2 columns of P
        p_ht = self.P[:, :2]
        k_gain = p_ht @ s_inv

        x_new = self.x + k_gain @ y
        x_new[0] = wrap_pi(float(x_new[0]))
        x_new[1] = clamp_elevation(float(x_new[1]))

        i_kh = np.eye(4, dtype=np.float64)
        i_kh[:, :2] -= k_gain
        p_new = i_kh @ self.P @ i_kh.T + k_gain @ r_mat @ k_gain.T
        p_new = 0.5 * (p_new + p_new.T)
        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(p_new))):
            return False
        self.x = x_new
        self.P = p_new
        return True

    def snapshot(self) -> BearingTrackSnapshot:
        return BearingTrackSnapshot(
            track_id=self.track_id,
            status=self.status,
            az_rad=float(self.x[0]),
            el_rad=float(self.x[1]),
            az_dot_rad_s=float(self.x[2]),
            el_dot_rad_s=float(self.x[3]),
            sigma_az_rad=math.sqrt(max(0.0, float(self.P[0, 0]))),
            sigma_el_rad=math.sqrt(max(0.0, float(self.P[1, 1]))),
            sigma_az_dot_rad_s=math.sqrt(max(0.0, float(self.P[2, 2]))),
            sigma_el_dot_rad_s=math.sqrt(max(0.0, float(self.P[3, 3]))),
            score=float(self.score),
            age_frames=int(self.age_frames),
            total_hits=int(self.total_hits),
            window_hits=int(sum(self.hit_window)),
            window_len=len(self.hit_window),
            coast_frames=int(self.coast_frames),
            eo_hits=int(self.eo_hits),
            lwir_hits=int(self.lwir_hits),
            cross_modal_fused=bool(self.cross_modal_fused),
            last_modalities=self.last_modalities,
            size_px=self.size_px,
        )


class BearingTrackBeforeConfirm:
    """Deterministic bearing-space CV Kalman tracker with M-of-N (`3-of-5`) Track-Before-Confirm
    and health-weighted EO/LWIR cross-modal track fusion.
    """

    def __init__(self, config: TrackBeforeConfirmConfig | None = None) -> None:
        self.cfg = TrackBeforeConfirmConfig() if config is None else config
        self._tracks: list[_KalmanBearingTrack] = []
        self._next_track_id: int = 1
        self._frame_idx: int = 0

    def reset(self) -> None:
        """Reset all tracks and deterministic ID/frame counters."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_idx = 0

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def active_tracks(self) -> tuple[BearingTrackSnapshot, ...]:
        return tuple(t.snapshot() for t in self._tracks)

    @property
    def confirmed_tracks(self) -> tuple[BearingTrackSnapshot, ...]:
        return tuple(t.snapshot() for t in self._tracks if t.status == TrackStatus.CONFIRMED)

    def _sanitize_detections(
        self,
        detections: Sequence[BearingDetection],
        q_eo: float,
        q_lwir: float,
        *,
        horizon_el_rad: float | None = None,
    ) -> tuple[list[_SanitizedDetection], list[_SanitizedDetection], int, int]:
        eo_valid: list[_SanitizedDetection] = []
        lwir_valid: list[_SanitizedDetection] = []
        rejected_invalid = 0
        rejected_health = 0

        horizon_valid = horizon_el_rad is not None and math.isfinite(float(horizon_el_rad))
        horizon_cutoff = (
            float(horizon_el_rad) - float(self.cfg.below_horizon_el_margin_rad)
            if horizon_valid and horizon_el_rad is not None
            else None
        )

        for idx, det in enumerate(detections):
            mod = det.modality.strip().lower() if isinstance(det.modality, str) else ""
            if mod not in ("eo", "lwir"):
                rejected_invalid += 1
                continue
            if not (
                isinstance(det.az_rad, (int, float, np.floating, np.integer))
                and isinstance(det.el_rad, (int, float, np.floating, np.integer))
                and isinstance(det.score, (int, float, np.floating, np.integer))
            ):
                rejected_invalid += 1
                continue
            az = float(det.az_rad)
            el = float(det.el_rad)
            raw_score = float(det.score)
            if not (
                math.isfinite(az)
                and math.isfinite(el)
                and math.isfinite(raw_score)
                and 0.0 <= raw_score <= 1.0
            ):
                rejected_invalid += 1
                continue

            base_std = (
                self.cfg.default_eo_meas_std_rad
                if mod == "eo"
                else self.cfg.default_lwir_meas_std_rad
            )
            sig_az = base_std if det.sigma_az_rad is None else float(det.sigma_az_rad)
            sig_el = base_std if det.sigma_el_rad is None else float(det.sigma_el_rad)
            if not (math.isfinite(sig_az) and sig_az > 0.0 and math.isfinite(sig_el) and sig_el > 0.0):
                rejected_invalid += 1
                continue

            size_px: float | None = None
            if det.size_px is not None:
                sp = float(det.size_px)
                if math.isfinite(sp) and sp > 0.0:
                    size_px = sp

            q_mod = q_eo if mod == "eo" else q_lwir
            if q_mod <= 0.0 or q_mod < self.cfg.min_modality_health:
                rejected_health += 1
                continue

            el_clamped = clamp_elevation(el)
            eff_score = raw_score * q_mod
            if horizon_cutoff is not None and el_clamped < horizon_cutoff:
                eff_score *= self.cfg.below_horizon_score_scale
            if eff_score < self.cfg.weak_score_thr:
                rejected_health += 1
                continue

            inv_sqrt_q = 1.0 / math.sqrt(max(q_mod, _MIN_HEALTH_SCALE))
            sanitized = _SanitizedDetection(
                det_idx=idx,
                az_rad=wrap_pi(az),
                el_rad=el_clamped,
                raw_score=raw_score,
                eff_score=eff_score,
                modality=mod,
                health=q_mod,
                sigma_az_eff=sig_az * inv_sqrt_q,
                sigma_el_eff=sig_el * inv_sqrt_q,
                size_px=size_px,
            )
            if mod == "eo":
                eo_valid.append(sanitized)
            else:
                lwir_valid.append(sanitized)

        if self.cfg.nms_radius_rad > 0.0:
            eo_valid = _angular_nms(eo_valid, self.cfg.nms_radius_rad)
            lwir_valid = _angular_nms(lwir_valid, self.cfg.nms_radius_rad)

        return eo_valid, lwir_valid, rejected_invalid, rejected_health

    def _associate_modality(
        self,
        dets: Sequence[_SanitizedDetection],
    ) -> tuple[dict[int, _SanitizedDetection], list[_SanitizedDetection]]:
        """Deterministic 1-to-1 global greedy minimum-Mahalanobis association for one modality."""
        if not self._tracks or not dets:
            return {}, list(dets)

        candidates: list[tuple[int, float, int, int]] = []
        for t_idx, trk in enumerate(self._tracks):
            status_prio = 0 if trk.status == TrackStatus.CONFIRMED else 1
            ang_cap = (
                self.cfg.tentative_assoc_rad
                if trk.status == TrackStatus.TENTATIVE and trk.total_hits == 1
                else self.cfg.max_assoc_rad
            )
            for d_idx, det in enumerate(dets):
                gate = trk.mahalanobis_sq(
                    det.az_rad, det.el_rad, det.sigma_az_eff, det.sigma_el_eff
                )
                if gate is None:
                    continue
                d2, ang_dist = gate
                if d2 <= self.cfg.mahalanobis_gate_sq and ang_dist <= ang_cap:
                    candidates.append((status_prio, d2, trk.track_id, d_idx))

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        track_id_to_idx = {t.track_id: idx for idx, t in enumerate(self._tracks)}
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        matches: dict[int, _SanitizedDetection] = {}

        for _prio, _d2, tid, d_idx in candidates:
            t_idx = track_id_to_idx[tid]
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                continue
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)
            matches[t_idx] = dets[d_idx]

        unmatched = [dets[i] for i in range(len(dets)) if i not in assigned_dets]
        return matches, unmatched

    def _can_confirm_track(self, trk: _KalmanBearingTrack) -> bool:
        """Evaluate M-of-N confirmation gate with score consistency & CBDR-safe static-clutter check."""
        if sum(trk.hit_window) < self.cfg.confirm_m:
            return False
        hit_scores = list(trk.score_window)
        mean_hit_score = (sum(hit_scores) / float(len(hit_scores))) if hit_scores else trk.score
        if mean_hit_score < self.cfg.min_confirm_score:
            return False
        if self.cfg.reject_static_clutter:
            bearing_rate = math.hypot(float(trk.x[2]), float(trk.x[3]))
            if bearing_rate < self.cfg.static_bearing_rate_rad_s:
                # Aviation CBDR safety: NEVER suppress a zero-bearing-rate track whose apparent size
                # is growing (head-on closing intruder!). Only suppress flat/non-growing static clutter.
                if (
                    trk.init_size_px is not None
                    and trk.size_px is not None
                    and trk.init_size_px > 0.0
                ):
                    size_growth_frac = (trk.size_px - trk.init_size_px) / trk.init_size_px
                    if size_growth_frac >= self.cfg.static_min_size_growth_frac:
                        return True
                return False
        return True

    def step(
        self,
        detections: Sequence[BearingDetection],
        *,
        dt_s: float = 0.1,
        q_eo: float = 1.0,
        q_lwir: float = 1.0,
        horizon_el_rad: float | None = None,
    ) -> TbcStepResult:
        """Advance tracker by one deterministic frame with EO/LWIR health-gated fusion."""
        self._frame_idx += 1
        qe = sanitize_health_weight(q_eo)
        ql = sanitize_health_weight(q_lwir)
        simplex_mode = select_cross_modal_simplex_mode(qe, ql, self.cfg.simplex_tau_ok)
        dual_blind = qe == 0.0 and ql == 0.0

        dt_ok = (
            isinstance(dt_s, (int, float, np.floating, np.integer))
            and not isinstance(dt_s, bool)
            and math.isfinite(float(dt_s))
            and float(dt_s) > 0.0
        )
        if not dt_ok:
            snaps = tuple(t.snapshot() for t in self._tracks)
            return TbcStepResult(
                frame_idx=self._frame_idx,
                dt_valid=False,
                q_eo=qe,
                q_lwir=ql,
                simplex_mode=simplex_mode,
                dual_blind=dual_blind,
                detections_received=len(detections),
                detections_accepted=0,
                detections_rejected_invalid=len(detections),
                detections_rejected_health=0,
                confirmed_tracks=tuple(s for s in snaps if s.is_confirmed),
                tentative_tracks=tuple(s for s in snaps if not s.is_confirmed),
                all_tracks=snaps,
                tracks_confirmed_this_step=0,
                tracks_deleted_this_step=0,
            )

        dt_f = float(dt_s)

        # 1. Predict all active tracks forward
        for trk in self._tracks:
            trk.predict(dt_f, self.cfg.process_accel_std_rad_s2)

        # 2. Sanitize detections & apply fail-closed health weighting (+ optional horizon clutter gate)
        eo_dets, lwir_dets, rej_invalid, rej_health = self._sanitize_detections(
            detections,
            qe,
            ql,
            horizon_el_rad=horizon_el_rad,
        )
        n_accepted = len(eo_dets) + len(lwir_dets)

        # 3. Per-modality deterministic Mahalanobis association (enables EO + LWIR fusion on same track)
        eo_matches, eo_unmatched = self._associate_modality(eo_dets)
        lwir_matches, lwir_unmatched = self._associate_modality(lwir_dets)

        # 4. Update matched tracks & coast unmatched tracks
        confirmed_this_step = 0
        deleted_this_step = 0
        survivors: list[_KalmanBearingTrack] = []

        for t_idx, trk in enumerate(self._tracks):
            trk.age_frames += 1
            m_eo = eo_matches.get(t_idx)
            m_lw = lwir_matches.get(t_idx)

            if m_eo is not None or m_lw is not None:
                if m_eo is not None and m_lw is not None:
                    z_az, z_el, sig_az, sig_el, p_fused = _fuse_pair_internal(m_eo, m_lw)
                    updated = trk.kalman_update(z_az, z_el, sig_az, sig_el)
                    mods: tuple[str, ...] = ("eo", "lwir")
                    if updated:
                        trk.eo_hits += 1
                        trk.lwir_hits += 1
                        trk.cross_modal_fused = True
                        trk.score = p_fused
                        trk.size_px = m_eo.size_px if m_eo.size_px is not None else m_lw.size_px
                elif m_eo is not None:
                    updated = trk.kalman_update(
                        m_eo.az_rad, m_eo.el_rad, m_eo.sigma_az_eff, m_eo.sigma_el_eff
                    )
                    mods = ("eo",)
                    if updated:
                        trk.eo_hits += 1
                        trk.score = m_eo.eff_score
                        if m_eo.size_px is not None:
                            trk.size_px = m_eo.size_px
                else:
                    assert m_lw is not None
                    updated = trk.kalman_update(
                        m_lw.az_rad, m_lw.el_rad, m_lw.sigma_az_eff, m_lw.sigma_el_eff
                    )
                    mods = ("lwir",)
                    if updated:
                        trk.lwir_hits += 1
                        trk.score = m_lw.eff_score
                        if m_lw.size_px is not None:
                            trk.size_px = m_lw.size_px

                if updated:
                    trk.total_hits += 1
                    trk.coast_frames = 0
                    trk.hit_window.append(True)
                    trk.score_window.append(float(trk.score))
                    if trk.init_size_px is None and trk.size_px is not None:
                        trk.init_size_px = trk.size_px
                    trk.last_modalities = mods
                    if trk.eo_hits > 0 and trk.lwir_hits > 0:
                        trk.cross_modal_fused = True
                    if trk.status == TrackStatus.TENTATIVE and self._can_confirm_track(trk):
                        trk.status = TrackStatus.CONFIRMED
                        confirmed_this_step += 1
                else:
                    trk.coast_frames += 1
                    trk.hit_window.append(False)
                    trk.last_modalities = ()
            else:
                trk.coast_frames += 1
                trk.hit_window.append(False)
                trk.last_modalities = ()

            should_delete = (
                trk.status == TrackStatus.CONFIRMED
                and trk.coast_frames >= self.cfg.max_coast_frames
            ) or (
                trk.status == TrackStatus.TENTATIVE
                and (
                    trk.coast_frames >= self.cfg.max_tentative_coast
                    or (
                        trk.age_frames >= self.cfg.confirm_n
                        and sum(trk.hit_window) < self.cfg.confirm_m
                        and trk.coast_frames >= 1
                    )
                )
            )
            if should_delete:
                deleted_this_step += 1
            else:
                survivors.append(trk)

        self._tracks = survivors

        # 5. Co-birth unmatched EO + LWIR detections that fall within max_assoc_rad, then single-modality births.
        #    Suppress duplicate births inside `nms_radius_rad` of an already-updated track (prevents
        #    secondary rotor/fuselage heatmap peaks from spawning duplicate tracks on the same aircraft).
        used_lwir: set[int] = set()
        eo_ordered = sorted(eo_unmatched, key=lambda d: (-d.eff_score, d.det_idx))
        lwir_ordered = sorted(lwir_unmatched, key=lambda d: (-d.eff_score, d.det_idx))

        def _near_existing_track(az: float, el: float) -> bool:
            if self.cfg.nms_radius_rad <= 0.0:
                return False
            for existing in self._tracks:
                if (
                    math.hypot(wrap_pi(az - float(existing.x[0])), el - float(existing.x[1]))
                    <= self.cfg.nms_radius_rad
                ):
                    return True
            return False

        for eo_d in eo_ordered:
            if len(self._tracks) >= self.cfg.max_tracks:
                break
            if _near_existing_track(eo_d.az_rad, eo_d.el_rad):
                continue
            best_lw_idx: int | None = None
            best_dist = self.cfg.max_assoc_rad
            for j, lw_d in enumerate(lwir_ordered):
                if j in used_lwir:
                    continue
                dist = math.hypot(wrap_pi(lw_d.az_rad - eo_d.az_rad), lw_d.el_rad - eo_d.el_rad)
                if dist <= best_dist:
                    best_dist = dist
                    best_lw_idx = j
            if best_lw_idx is not None:
                used_lwir.add(best_lw_idx)
                lw_d = lwir_ordered[best_lw_idx]
                az_f, el_f, sig_az, sig_el, p_f = _fuse_pair_internal(eo_d, lw_d)
                new_trk = _KalmanBearingTrack(
                    self._next_track_id,
                    az_f,
                    el_f,
                    p_f,
                    sig_az,
                    sig_el,
                    self.cfg,
                    modalities=("eo", "lwir"),
                    size_px=eo_d.size_px if eo_d.size_px is not None else lw_d.size_px,
                )
            else:
                new_trk = _KalmanBearingTrack(
                    self._next_track_id,
                    eo_d.az_rad,
                    eo_d.el_rad,
                    eo_d.eff_score,
                    eo_d.sigma_az_eff,
                    eo_d.sigma_el_eff,
                    self.cfg,
                    modalities=("eo",),
                    size_px=eo_d.size_px,
                )
            self._next_track_id += 1
            if new_trk.status == TrackStatus.CONFIRMED:
                confirmed_this_step += 1
            self._tracks.append(new_trk)

        for j, lw_d in enumerate(lwir_ordered):
            if j in used_lwir:
                continue
            if len(self._tracks) >= self.cfg.max_tracks:
                break
            if _near_existing_track(lw_d.az_rad, lw_d.el_rad):
                continue
            new_trk = _KalmanBearingTrack(
                self._next_track_id,
                lw_d.az_rad,
                lw_d.el_rad,
                lw_d.eff_score,
                lw_d.sigma_az_eff,
                lw_d.sigma_el_eff,
                self.cfg,
                modalities=("lwir",),
                size_px=lw_d.size_px,
            )
            self._next_track_id += 1
            if new_trk.status == TrackStatus.CONFIRMED:
                confirmed_this_step += 1
            self._tracks.append(new_trk)

        # Deterministic snapshot ordering: confirmed first, then ascending track_id
        self._tracks.sort(key=lambda t: (0 if t.status == TrackStatus.CONFIRMED else 1, t.track_id))
        snaps = tuple(t.snapshot() for t in self._tracks)
        return TbcStepResult(
            frame_idx=self._frame_idx,
            dt_valid=True,
            q_eo=qe,
            q_lwir=ql,
            simplex_mode=simplex_mode,
            dual_blind=dual_blind,
            detections_received=len(detections),
            detections_accepted=n_accepted,
            detections_rejected_invalid=rej_invalid,
            detections_rejected_health=rej_health,
            confirmed_tracks=tuple(s for s in snaps if s.is_confirmed),
            tentative_tracks=tuple(s for s in snaps if not s.is_confirmed),
            all_tracks=snaps,
            tracks_confirmed_this_step=confirmed_this_step,
            tracks_deleted_this_step=deleted_this_step,
        )


def compute_12_sector_optical_quality(
    frame: np.ndarray,
    *,
    modality: str = "eo",
) -> list[float]:
    """Compute calibrated 12-sector (3 rows x 4 cols) sky-aware optical quality in `[0.0, 1.0]`.

    Pure-numpy (`0` OpenCV/PyTorch dependency). Unlike texture/Laplacian-based frame sharpness gates
    that wrongly crush featureless clean/overcast sky (`MiG-29` `0.041`, `Hurricane` `0.170`), clean
    mid-tone featureless sky (`15 < mean < 245`) is scored healthy for optical aircraft silhouette
    detection, while true optical degradations fail closed:
      - Constant zero-variance dead/frozen frame (`whole_std < 0.05`) -> `0.0`,
      - Sun-glare / saturation (`>= 250 DN`) -> collapses to `0.0` at `>= 50%`,
      - Lens-cap / crushed black (`<= 5 DN`) -> collapses to `0.0` at `>= 75%` (with LWIR cold-sky
        hot-target preservation when a crisp localized thermal peak `>= 40 DN`, `peak - P50 >= 25 DN`
        exists in a clean sensor frame),
      - Stuck dead-pixel row/column FPN stripes (`>= 3%` flagged, `>= 10%` -> `<= 0.15`),
      - Spatially uncorrelated white/binary sensor noise (`std > 15 DN`, lag-1 autocorr `< 0.25`)
        -> `<= 0.10`.
    """
    mod = modality.strip().lower()
    if mod not in ("eo", "lwir"):
        raise ValueError(f"Unsupported modality {modality!r}; expected 'eo' or 'lwir'")

    arr = np.asarray(frame)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return [0.0] * 12
    if arr.ndim == 2:
        gray = arr.astype(np.float64, copy=False)
    elif arr.shape[2] == 1:
        gray = arr[:, :, 0].astype(np.float64, copy=False)
    elif arr.shape[2] in (3, 4):
        bgr = arr[:, :, :3].astype(np.float64, copy=False)
        gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    else:
        return [0.0] * 12

    if gray.size == 0 or not bool(np.all(np.isfinite(gray))):
        return [0.0] * 12

    def _clamp01_local(v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            return 0.0
        return 1.0 if v >= 1.0 else float(v)

    def _fpn_score_np(g: np.ndarray) -> float:
        gh, gw = g.shape[:2]
        if gh < 6 or gw < 6 or float(np.std(g)) < 1.0:
            return 1.0
        r_med = np.median(g, axis=1)
        r_stk = np.mean(np.abs(g - r_med[:, None]) <= 1.5, axis=1) >= 0.70
        r_up = np.maximum(np.abs(r_med[2:-2] - r_med[1:-3]), np.abs(r_med[2:-2] - r_med[:-4]))
        r_dn = np.maximum(np.abs(r_med[2:-2] - r_med[3:-1]), np.abs(r_med[2:-2] - r_med[4:]))
        f_r = float(np.sum(r_stk[2:-2] & (np.minimum(r_up, r_dn) >= 15.0))) / float(gh)

        c_med = np.median(g, axis=0)
        c_stk = np.mean(np.abs(g - c_med[None, :]) <= 1.5, axis=0) >= 0.70
        c_lt = np.maximum(np.abs(c_med[2:-2] - c_med[1:-3]), np.abs(c_med[2:-2] - c_med[:-4]))
        c_rt = np.maximum(np.abs(c_med[2:-2] - c_med[3:-1]), np.abs(c_med[2:-2] - c_med[4:]))
        f_c = float(np.sum(c_stk[2:-2] & (np.minimum(c_lt, c_rt) >= 15.0))) / float(gw)

        dead_frac = min(1.0, f_r + f_c)
        if dead_frac < 0.03:
            return 1.0
        return _clamp01_local(0.45 * (1.0 - (dead_frac - 0.03) / 0.07))

    def _noise_score_np(g: np.ndarray) -> float:
        gh, gw = g.shape[:2]
        sd = float(np.std(g))
        if gh < 4 or gw < 4 or not math.isfinite(sd) or sd <= 15.0:
            return 1.0
        g0 = g - float(np.mean(g))
        var = float(np.mean(g0 * g0))
        if not math.isfinite(var) or var <= 1e-6:
            return 1.0
        ac_h = float(np.mean(g0[:, :-1] * g0[:, 1:]) / var)
        ac_v = float(np.mean(g0[:-1, :] * g0[1:, :]) / var)
        ac = max(ac_h, ac_v)
        if not math.isfinite(ac):
            return 0.0
        if ac < 0.25:
            return _clamp01_local(0.10 * max(0.0, ac) / 0.25)
        return 1.0

    # Constant zero-variance dead/stuck frame fails closed immediately
    whole_std = float(np.std(gray))
    if not math.isfinite(whole_std) or whole_std < 0.05:
        return [0.0] * 12

    whole_fpn = _fpn_score_np(gray)
    whole_noise = _noise_score_np(gray)

    # Frame-level LWIR cold-sky hot-target presence check (so sectors of a cold clear sky aren't
    # falsely branded lens-capped when a real thermal target is radiating in the FOV)
    lwir_frame_has_hot_target = False
    if mod == "lwir" and whole_fpn >= 0.5 and whole_noise >= 0.5:
        p50_w, p999_w = np.percentile(gray, [50.0, 99.9])
        if gray.shape[0] >= 2 and gray.shape[1] >= 2:
            m2 = float(
                np.max(
                    np.minimum(
                        np.minimum(gray[:-1, :-1], gray[:-1, 1:]),
                        np.minimum(gray[1:, :-1], gray[1:, 1:]),
                    )
                )
            )
        else:
            m2 = float(p999_w)
        pk_w = max(float(p999_w), m2)
        if pk_w >= 40.0 and (pk_w - float(p50_w)) >= 25.0:
            lwir_frame_has_hot_target = True

    def _score_patch(patch: np.ndarray) -> float:
        p_mean = float(np.mean(patch))
        over_r = float(np.mean(patch >= 250.0))
        under_r = float(np.mean(patch <= 5.0))
        glare_s = _clamp01_local(1.0 - max(0.0, over_r - 0.02) / 0.48)
        occl_s = _clamp01_local(1.0 - max(0.0, under_r - 0.10) / 0.65)
        fpn_s = min(whole_fpn, _fpn_score_np(patch))
        noise_s = min(whole_noise, _noise_score_np(patch))

        if mod == "eo":
            low_exp = _clamp01_local((p_mean - 5.0) / 10.0)
            high_exp = _clamp01_local((252.0 - p_mean) / 7.0)
            return _clamp01_local(glare_s * occl_s * low_exp * high_exp * fpn_s * noise_s)

        # LWIR sector: preserve cold clean sky when frame carries localized thermal target signal
        if lwir_frame_has_hot_target:
            occl_s = 1.0
        return _clamp01_local(glare_s * occl_s * fpn_s * noise_s)

    h, w = gray.shape[:2]
    if h < 6 or w < 8:
        q_w = round(_score_patch(gray), 4)
        return [q_w] * 12

    row_edges = np.linspace(0, h, 4, dtype=np.int32)
    col_edges = np.linspace(0, w, 5, dtype=np.int32)
    scores: list[float] = []
    for r in range(3):
        r0, r1 = int(row_edges[r]), int(row_edges[r + 1])
        for c in range(4):
            c0, c1 = int(col_edges[c]), int(col_edges[c + 1])
            scores.append(round(_score_patch(gray[r0:r1, c0:c1]), 4))
    return scores


def simulate_encounter_884_low_snr_sequence(
    *,
    num_frames: int = 60,
    dt_s: float = 0.1,
    clutter_per_frame: int = 4,
    seed: int = 884,
) -> tuple[list[tuple[float, float]], list[list[BearingDetection]]]:
    """Generate a deterministic low-SNR small fixed-wing Encounter `884.0`-class sequence.

    Produces a smooth closing/crossing small fixed-wing bearing trajectory with intermittent weak
    EO detections (`p ~ 0.22–0.38`, ~`12.5%` crossing `0.45`, occasional 1–2 frame fades) embedded
    in random sky clutter false alarms (`p ~ 0.16–0.48`).
    """
    rng = np.random.default_rng(int(seed))
    gt_bearings: list[tuple[float, float]] = []
    frame_dets: list[list[BearingDetection]] = []

    az0, el0 = -0.11, 0.045
    az_vel, el_vel = 0.022, -0.0045  # rad/s smooth crossing small fixed-wing
    az_acc, el_acc = 0.0012, -0.0003

    for k in range(int(num_frames)):
        t = k * float(dt_s)
        gt_az = wrap_pi(az0 + az_vel * t + 0.5 * az_acc * t * t)
        gt_el = clamp_elevation(el0 + el_vel * t + 0.5 * el_acc * t * t)
        gt_bearings.append((gt_az, gt_el))

        dets: list[BearingDetection] = []
        # Intermittent weak target detection (85% weak-hit presence, ~12.5% strong >=0.45 hits)
        fade_drop = (k % 9 == 4) or (k == 23) or (k == 41)
        if not fade_drop:
            # Encounter 884.0 low-SNR profile: mostly p in [0.22, 0.38], occasional >= 0.45 peaks
            if k in (7, 16, 28, 37, 46, 54, 58):
                score = float(rng.uniform(0.46, 0.56))
            else:
                score = float(rng.uniform(0.22, 0.38))
            meas_az = wrap_pi(gt_az + float(rng.normal(0.0, 0.0011)))
            meas_el = clamp_elevation(gt_el + float(rng.normal(0.0, 0.0011)))
            dets.append(
                BearingDetection(
                    az_rad=meas_az,
                    el_rad=meas_el,
                    score=score,
                    modality="eo",
                    sigma_az_rad=0.0012,
                    sigma_el_rad=0.0012,
                    size_px=7.5 + 0.08 * k,
                )
            )

        # Random sky clutter false alarms (birds, cloud edges, specular glints) across FOV
        for _ in range(int(clutter_per_frame)):
            c_az = float(rng.uniform(-0.32, 0.32))
            c_el = float(rng.uniform(-0.14, 0.22))
            # Keep clutter clearly distinct from the true aircraft (> 0.045 rad away)
            if math.hypot(wrap_pi(c_az - gt_az), c_el - gt_el) < 0.045:
                c_az = wrap_pi(c_az + 0.12)
            c_score = float(rng.uniform(0.16, 0.47))
            dets.append(
                BearingDetection(
                    az_rad=c_az,
                    el_rad=c_el,
                    score=c_score,
                    modality="eo",
                )
            )

        frame_dets.append(dets)

    return gt_bearings, frame_dets


def benchmark_tbc_vs_single_frame(
    gt_bearings: Sequence[tuple[float, float]],
    frame_detections: Sequence[Sequence[BearingDetection]],
    *,
    scenario_name: str = "encounter_884_low_snr",
    dt_s: float = 0.1,
    q_eo_seq: Sequence[float] | None = None,
    q_lwir_seq: Sequence[float] | None = None,
    config: TrackBeforeConfirmConfig | None = None,
    match_tol_rad: float = 0.015,
) -> TbcBenchmarkSummary:
    """Benchmark BEFORE (single-frame threshold `>= 0.45`) vs AFTER (M-of-N `3-of-5` TBC)."""
    n_frames = len(gt_bearings)
    if len(frame_detections) != n_frames or n_frames == 0:
        raise ValueError("gt_bearings and frame_detections must be non-empty and equal length")

    cfg = TrackBeforeConfirmConfig() if config is None else config
    tracker = BearingTrackBeforeConfirm(cfg)

    sf_det_frames = 0
    sf_false_alarms = 0
    sf_sq_errs: list[float] = []

    tbc_conf_frames = 0
    tbc_false_conf_frames = 0
    tbc_false_track_ids: set[int] = set()
    tbc_confirm_latency: int | None = None
    tbc_sq_errs: list[float] = []

    for idx in range(n_frames):
        gt_az, gt_el = gt_bearings[idx]
        dets = frame_detections[idx]
        qe = 1.0 if q_eo_seq is None else sanitize_health_weight(q_eo_seq[idx])
        ql = 1.0 if q_lwir_seq is None else sanitize_health_weight(q_lwir_seq[idx])

        # BEFORE: single-frame health-weighted threshold (`eff_score >= single_frame_score_thr`)
        best_sf_err: float | None = None
        for d in dets:
            mod = d.modality.strip().lower()
            qm = qe if mod == "eo" else (ql if mod == "lwir" else 0.0)
            if not (math.isfinite(d.az_rad) and math.isfinite(d.el_rad) and math.isfinite(d.score)):
                continue
            if d.score * qm < cfg.single_frame_score_thr:
                continue
            err = math.hypot(wrap_pi(d.az_rad - gt_az), d.el_rad - gt_el)
            if err <= match_tol_rad:
                if best_sf_err is None or err < best_sf_err:
                    best_sf_err = err
            else:
                sf_false_alarms += 1
        if best_sf_err is not None:
            sf_det_frames += 1
            sf_sq_errs.append(best_sf_err * best_sf_err)

        # AFTER: deterministic M-of-N Track-Before-Confirm
        res = tracker.step(dets, dt_s=dt_s, q_eo=qe, q_lwir=ql)
        best_tbc_err: float | None = None
        for trk in res.confirmed_tracks:
            err = math.hypot(wrap_pi(trk.az_rad - gt_az), trk.el_rad - gt_el)
            if err <= match_tol_rad:
                if best_tbc_err is None or err < best_tbc_err:
                    best_tbc_err = err
            else:
                tbc_false_conf_frames += 1
                tbc_false_track_ids.add(trk.track_id)
        if best_tbc_err is not None:
            tbc_conf_frames += 1
            tbc_sq_errs.append(best_tbc_err * best_tbc_err)
            if tbc_confirm_latency is None:
                tbc_confirm_latency = idx

    sf_cont = sf_det_frames / float(n_frames)
    tbc_cont = tbc_conf_frames / float(n_frames)
    sf_rmse_mrad = (
        math.sqrt(sum(sf_sq_errs) / len(sf_sq_errs)) * 1000.0 if sf_sq_errs else None
    )
    tbc_rmse_mrad = (
        math.sqrt(sum(tbc_sq_errs) / len(tbc_sq_errs)) * 1000.0 if tbc_sq_errs else None
    )
    return TbcBenchmarkSummary(
        scenario_name=scenario_name,
        num_frames=n_frames,
        single_frame_det_frames=sf_det_frames,
        single_frame_continuity=sf_cont,
        single_frame_false_alarms=sf_false_alarms,
        tbc_confirmed_frames=tbc_conf_frames,
        tbc_continuity=tbc_cont,
        tbc_continuity_lift=tbc_cont - sf_cont,
        tbc_false_confirmed_tracks=len(tbc_false_track_ids),
        tbc_false_confirmed_frames=tbc_false_conf_frames,
        tbc_confirm_latency_frames=tbc_confirm_latency,
        single_frame_bearing_rmse_mrad=sf_rmse_mrad,
        tbc_bearing_rmse_mrad=tbc_rmse_mrad,
    )
