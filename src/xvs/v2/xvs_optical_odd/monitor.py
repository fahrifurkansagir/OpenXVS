"""DR-13 5-Mode Deterministic Optical-ODD Health Monitor (Strictly Zero-ML, DO-178C DAL B/C Pattern).

Implements classical physical signal processing across IMX296LLR (10-bit monochrome EO),
Teledyne FLIR Boson+ (14-bit Pre-AGC LWIR), IMX636 (EVS tile slice), SPA solar geometry,
and enclosure hygrothermal sensors with correlated-block M-of-N MLTL persistence
(`r2u2_core`-compatible counters) and a direct adapter to Stage 3 of `AsymmetricSimplexVlmGate`.

HARD AF447 INVARIANT (DR-08 / DR-13):
  `airdata_export_blocked` is permanently locked to `True`. Mode 2 optical window icing
  assesses optical aperture health ONLY and NEVER estimates or exports airspeed, altitude,
  AoA, or airframe ice accretion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
from typing import Mapping

import numpy as np

from xvs.v2.xvs_vlm_gate import OpticalOddReading

EVS_HARD_CAP_EVENTS: int = 65_536
WMO_MOR_LN20: float = math.log(20.0)  # 2.995732273553991 (WMO 5% contrast threshold)


class OpticalOddMode(IntEnum):
    """Five deterministic optical-ODD degradation modes per DR-13 §1.3."""

    RAINDROP_OCCLUSION = 1
    WINDOW_ICING = 2
    SOLAR_GLARE = 3
    FOG_VISIBILITY = 4
    LWIR_BOLOMETER_DRIFT = 5


_ALL_MODES: tuple[OpticalOddMode, ...] = (
    OpticalOddMode.RAINDROP_OCCLUSION,
    OpticalOddMode.WINDOW_ICING,
    OpticalOddMode.SOLAR_GLARE,
    OpticalOddMode.FOG_VISIBILITY,
    OpticalOddMode.LWIR_BOLOMETER_DRIFT,
)


class OpticalOddAction(str, Enum):
    """Deterministic Simplex RTA / sensor-mode routing verdict."""

    NOMINAL = "NOMINAL"
    EVS_LWIR_ELEVATED = "EVS_LWIR_ELEVATED"
    THERMAL_PRIMARY = "THERMAL_PRIMARY"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"


@dataclass(frozen=True)
class MltlPersistenceSpec:
    """Correlated-block M-of-N window + R2U2 C2PO G[0, debounce_ticks-1] debounce."""

    confirm_m: int
    window_n: int
    debounce_ticks: int = 3

    def __post_init__(self) -> None:
        if self.confirm_m < 1 or self.window_n < self.confirm_m:
            raise ValueError(
                f"invalid M-of-N window: confirm_m={self.confirm_m}, window_n={self.window_n}"
            )
        if self.debounce_ticks < 1:
            raise ValueError(f"debounce_ticks must be >= 1, got {self.debounce_ticks}")


def default_mltl_specs() -> dict[OpticalOddMode, MltlPersistenceSpec]:
    """DR-13 §1.3 / §1.4 60 Hz correlated-block M-of-N & G[0,2] persistence specs."""
    return {
        OpticalOddMode.RAINDROP_OCCLUSION: MltlPersistenceSpec(21, 30, 3),
        OpticalOddMode.WINDOW_ICING: MltlPersistenceSpec(65, 90, 3),
        OpticalOddMode.SOLAR_GLARE: MltlPersistenceSpec(22, 30, 3),
        OpticalOddMode.FOG_VISIBILITY: MltlPersistenceSpec(42, 60, 3),
        OpticalOddMode.LWIR_BOLOMETER_DRIFT: MltlPersistenceSpec(45, 60, 3),
    }


@dataclass(frozen=True)
class OpticalOddConfig:
    """Calibrated physical thresholds anchored to IMX296LLR, Boson+ & IMX636 datasheets."""

    # Mode 1: Lens Raindrop / Water-Film / Mud / Bug Occlusion (stride-4 Tenengrad & MLAP)
    raindrop_tile_rows: int = 17
    raindrop_tile_cols: int = 23
    raindrop_tenengrad_ref: float = 1200.0
    raindrop_mlap_ref: float = 40.0
    raindrop_tenengrad_ratio_thresh: float = 0.25
    raindrop_mlap_ratio_thresh: float = 0.25
    raindrop_occluded_tile_frac_thresh: float = 0.12
    raindrop_tile_persist_frames: int = 2

    # Mode 2: Optical Window Icing / Frost / Condensation (AF447 Ban: zero airdata!)
    icing_mtf_ref_eta0: float = 0.40
    icing_mtf_ratio_thresh: float = 0.35
    lwir_netd_dn: float = 1.34  # 20 mK NEDT @ 14.92 mK/DN on 14-bit Boson+ High-Gain
    icing_ir_std_mult: float = 3.5
    icing_dew_margin_c: float = 2.0

    # Mode 3: Direct Solar Veiling Glare / Blooming (SPA-gated)
    glare_sat_dn: int = 1008  # 98.5% of 10-bit scale (EMVA 1288 well sat ~987 DN)
    glare_sat_frac_thresh: float = 0.03
    glare_ring_contrast_ref: float = 80.0
    glare_ring_contrast_ratio_thresh: float = 0.20
    glare_spa_margin_deg: float = 12.0

    # Mode 4: Low-Contrast Fog / Smoke / Visibility Loss (WMO 5% MOR + 3DEP Horizon SNR)
    fog_mor_thresh_m: float = 1609.0  # 1 SM LIFR (beta_ext > 1.862 km^-1)
    fog_horizon_snr_db_thresh: float = 3.0
    fog_ir_healthy_std_mult: float = 6.0
    fog_dcp_airlight_dn: float = 950.0
    fog_nominal_range_km: float = 1.5

    # Mode 5: LWIR Bolometer Drift, Stale FFC & Bad-Pixel Surge
    lwir_ref_median_dn: float = 8192.0
    lwir_median_drift_dn_thresh: float = 650.0
    lwir_min_iqr_dn: float = 12.0
    lwir_fpn_ratio_thresh: float = 0.35
    lwir_bad_pixel_thresh: float = 0.010  # Industrial grade (>=99.0% operability; Pro=0.015)
    lwir_bad_pixel_delta_dn: float = 400.0
    lwir_stale_ffc_s: float = 300.0
    lwir_ffc_temp_delta_c: float = 1.0

    # Correlated-block MLTL persistence (r2u2_core-compatible counters)
    mltl_specs: Mapping[OpticalOddMode, MltlPersistenceSpec] = field(
        default_factory=default_mltl_specs
    )

    def __post_init__(self) -> None:
        for mode in _ALL_MODES:
            if mode not in self.mltl_specs:
                raise ValueError(f"missing MLTL persistence spec for {mode.name}")


@dataclass(frozen=True)
class OpticalOddFrameInput:
    """Single-tick synchronous sensor & auxiliary telemetry bundle for Stage-3 ODD check."""

    eo_frame_u16: np.ndarray
    lwir_frame_u16: np.ndarray
    sun_boresight_deg: float = 90.0
    eo_hfov_deg: float = 60.0
    sun_centroid_uv: tuple[float, float] | None = None
    t_win_c: float = 18.0
    t_amb_c: float = 20.0
    rh_pct: float = 45.0
    boson_dt_ffc_s: float = 60.0
    boson_t_cam_c: float = 25.0
    boson_t_last_ffc_c: float = 24.8
    boson_ffc_freeze: bool = False
    horizon_row_frac: float = 0.50
    evs_tile_counts: np.ndarray | None = None
    attitude_in_odd: bool = True
    beta_ext_km_inv: float | None = None


@dataclass(frozen=True)
class OpticalOddMetrics:
    """Deterministic physical observables extracted across all 5 optical-ODD modes."""

    # Mode 1
    droplet_occlusion_frac: float
    mean_tenengrad: float
    mean_mlap: float
    evs_total_events_capped: int
    # Mode 2
    mtf_hf_ratio: float
    t_dew_frost_c: float
    window_dew_margin_c: float
    lwir_scene_std_dn: float
    # Mode 3
    spa_in_glare_cone: bool
    glare_sat_frac: float
    glare_ring_contrast: float
    # Mode 4
    dcp_floor_dn: float
    horizon_snr_db: float
    estimated_beta_ext_km_inv: float
    estimated_mor_m: float
    fog_contrast: float
    # Mode 5
    lwir_median_dn: float
    lwir_iqr_dn: float
    lwir_fpn_ratio: float
    lwir_bad_pixel_frac: float
    boson_dt_ffc_s: float
    boson_temp_delta_c: float


@dataclass(frozen=True)
class OpticalOddDecision:
    """Deterministic Stage-3 Optical-ODD verdict with R2U2 MLTL counters & VLM adapter."""

    action: OpticalOddAction
    raw_mode_flags: Mapping[OpticalOddMode, bool]
    persisted_mode_flags: Mapping[OpticalOddMode, bool]
    mltl_window_counts: Mapping[OpticalOddMode, int]
    health_bitmask: int
    mask_lwir_frame: bool
    vlm_gate_reject: bool
    ito_heater_cmd: bool
    attitude_in_odd: bool
    metrics: OpticalOddMetrics
    airdata_export_blocked: bool = True

    def __post_init__(self) -> None:
        if not self.airdata_export_blocked:
            raise ValueError(
                "AF447 Visual-Airdata Ban (DR-08/DR-13): airdata_export_blocked MUST remain True"
            )

    def to_vlm_optical_odd_health(self) -> OpticalOddReading:
        """Return a direct `xvs.v2.xvs_vlm_gate.OpticalOddReading` for DR-09 Stage 3.

        Maps persisted/raw optical-ODD breaches directly into `AsymmetricSimplexVlmGate`'s
        classical Stage-3 thresholds so rain, icing, glare, or fog deterministically mask VLM.
        """
        ice_tripped = (
            self.persisted_mode_flags[OpticalOddMode.WINDOW_ICING]
            or self.raw_mode_flags[OpticalOddMode.WINDOW_ICING]
        )
        glare_tripped = self.persisted_mode_flags[OpticalOddMode.SOLAR_GLARE]
        rain_tripped = self.persisted_mode_flags[OpticalOddMode.RAINDROP_OCCLUSION]
        fog_tripped = self.persisted_mode_flags[OpticalOddMode.FOG_VISIBILITY]

        glare_sat = self.metrics.glare_sat_frac
        if glare_tripped and glare_sat <= 0.15:
            glare_sat = 0.25

        droplet_occ = self.metrics.droplet_occlusion_frac
        if (rain_tripped or ice_tripped) and droplet_occ <= 0.12:
            droplet_occ = 0.25

        fog_contrast = self.metrics.fog_contrast
        if (fog_tripped or ice_tripped) and fog_contrast >= 0.35:
            fog_contrast = 0.20

        blur_lap = self.metrics.mean_mlap
        if ice_tripped and blur_lap >= 50.0:
            blur_lap = 10.0

        return OpticalOddReading(
            glare_sat_frac=float(glare_sat),
            droplet_occlusion_frac=float(droplet_occ),
            fog_contrast=float(fog_contrast),
            blur_laplacian=float(blur_lap),
            attitude_in_odd=bool(self.attitude_in_odd and not self.vlm_gate_reject),
        )


def compute_buck_dew_frost_point_c(t_amb_c: float, rh_pct: float) -> float:
    """Arden Buck (1981/1996) saturation vapor pressure inversion for dew/frost point (°C).

    Uses water-vapor constants above 0 °C and ice/frost-point constants at or below 0 °C.
    Strictly for optical aperture condensation/icing ODD monitoring (ZERO airdata export).
    """
    rh_clamped = min(max(float(rh_pct), 0.1), 100.0)
    t_c = float(t_amb_c)
    if t_c >= 0.0:
        b, c, d = 18.678, 257.14, 234.5
    else:
        b, c, d = 23.036, 279.82, 333.7
    gamma = math.log((rh_clamped / 100.0) * math.exp((b - t_c / d) * (t_c / (c + t_c))))
    return (c * gamma) / (b - gamma)


def _stride4_eo(eo: np.ndarray) -> np.ndarray:
    """Return L2-resident stride-4 EO slice when given full-resolution IMX296LLR frame."""
    if eo.shape[0] >= 544 and eo.shape[1] >= 728:
        return eo[::4, ::4].astype(np.float32, copy=False)
    return eo.astype(np.float32, copy=False)


def _compute_tenengrad_and_mlap(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel Tenengrad (Gx^2 + Gy^2) and Modified Laplacian (|Ixx| + |Iyy|)."""
    if img.shape[0] < 3 or img.shape[1] < 3:
        zeros = np.zeros_like(img, dtype=np.float32)
        return zeros, zeros
    # 3x3 Sobel gradients on interior
    tl = img[:-2, :-2]
    tc = img[:-2, 1:-1]
    tr = img[:-2, 2:]
    ml = img[1:-1, :-2]
    mc = img[1:-1, 1:-1]
    mr = img[1:-1, 2:]
    bl = img[2:, :-2]
    bc = img[2:, 1:-1]
    br = img[2:, 2:]

    gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)
    gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)
    tenengrad = (gx * gx + gy * gy) * (1.0 / 16.0)

    ixx = np.abs(2.0 * mc - ml - mr)
    iyy = np.abs(2.0 * mc - tc - bc)
    mlap = ixx + iyy
    return tenengrad, mlap


def _tile_means(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Compute deterministic spatial tile means over an (rows x cols) grid."""
    h, w = arr.shape
    eff_r = max(1, min(rows, h))
    eff_c = max(1, min(cols, w))
    r_edges = np.linspace(0, h, eff_r + 1, dtype=np.int32)
    c_edges = np.linspace(0, w, eff_c + 1, dtype=np.int32)
    out = np.zeros((eff_r, eff_c), dtype=np.float32)
    for r in range(eff_r):
        r0, r1 = int(r_edges[r]), int(r_edges[r + 1])
        for c in range(eff_c):
            c0, c1 = int(c_edges[c]), int(c_edges[c + 1])
            patch = arr[r0:r1, c0:c1]
            out[r, c] = float(np.mean(patch)) if patch.size > 0 else 0.0
    return out


def _compute_mtf_hf_ratio(eo_full: np.ndarray) -> float:
    """Mode 2: 16-line 512-pt RFFT high-frequency MTF roll-off E[0.15,0.40]/E[0.02,0.15]."""
    img = eo_full.astype(np.float32, copy=False)
    h, w = img.shape
    n_fft = 512
    # Sample 8 horizontal + 8 vertical full-resolution scanlines (zero-pad if synthetic < 512)
    lines = np.zeros((16, n_fft), dtype=np.float32)
    h_rows = np.linspace(0, max(h - 1, 0), 8, dtype=np.int32)
    v_cols = np.linspace(0, max(w - 1, 0), 8, dtype=np.int32)

    for idx, r in enumerate(h_rows):
        row = img[int(r), :]
        if row.size >= n_fft:
            start = (row.size - n_fft) // 2
            seg = row[start : start + n_fft]
        else:
            seg = np.resize(row, n_fft)
        lines[idx, :] = seg - float(np.mean(seg))

    for idx, c in enumerate(v_cols):
        col = img[:, int(c)]
        if col.size >= n_fft:
            start = (col.size - n_fft) // 2
            seg = col[start : start + n_fft]
        else:
            seg = np.resize(col, n_fft)
        lines[8 + idx, :] = seg - float(np.mean(seg))

    # Hann window to avoid spectral leakage at boundaries
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(lines * window[None, :], axis=1)
    psd = np.mean(spec.real * spec.real + spec.imag * spec.imag, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    low_mask = (freqs >= 0.02) & (freqs < 0.15)
    high_mask = (freqs >= 0.15) & (freqs <= 0.40)
    e_low = float(np.sum(psd[low_mask]))
    e_high = float(np.sum(psd[high_mask]))
    if e_low <= 1e-9:
        return 0.0
    return e_high / e_low


def _compute_solar_ring_contrast(
    eo_s4: np.ndarray,
    sat_mask: np.ndarray,
    sun_centroid_uv: tuple[float, float] | None,
) -> float:
    """Compute radial ring contrast (std of pixel intensities) around sun centroid."""
    h, w = eo_s4.shape
    if sun_centroid_uv is not None:
        cx = float(sun_centroid_uv[0]) * (w - 1)
        cy = float(sun_centroid_uv[1]) * (h - 1)
    elif np.any(sat_mask):
        ys, xs = np.nonzero(sat_mask)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
    else:
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_inner = max(2.0, 0.05 * min(h, w))
    r_outer = max(r_inner + 3.0, 0.22 * min(h, w))
    ring = (dist >= r_inner) & (dist <= r_outer)
    if not np.any(ring):
        return float(np.std(eo_s4))
    return float(np.std(eo_s4[ring]))


def _compute_fog_metrics(
    eo_full: np.ndarray,
    eo_s4: np.ndarray,
    horizon_row_frac: float,
    cfg: OpticalOddConfig,
    beta_ext_override: float | None,
) -> tuple[float, float, float, float, float]:
    """Mode 4: Monochrome dark-channel lower envelope, 3DEP horizon SNR (dB), and WMO MOR."""
    # 1. Monochrome dark-channel 5th-percentile floor across 8x8 blocks
    h_s4, w_s4 = eo_s4.shape
    bh = max(1, h_s4 // 8)
    bw = max(1, w_s4 // 8)
    trimmed = eo_s4[: bh * 8, : bw * 8]
    if trimmed.size > 0 and bh >= 1 and bw >= 1:
        blocks = trimmed.reshape(bh, 8, bw, 8)
        block_mins = np.min(blocks, axis=(1, 3))
        dcp_floor = float(np.percentile(block_mins, 5.0))
    else:
        dcp_floor = float(np.percentile(eo_s4, 5.0))

    # 2. 3DEP-projected horizon strip vertical Sobel SNR
    img = eo_full.astype(np.float32, copy=False)
    h, w = img.shape
    hor_row = int(round(min(max(horizon_row_frac, 0.1), 0.9) * (h - 1)))
    strip_half = min(16, max(2, h // 8))
    r0 = max(1, hor_row - strip_half)
    r1 = min(h - 1, hor_row + strip_half)
    # Vertical gradient |I[r+1] - I[r-1]| / 2
    gy_strip = np.abs(img[r0 + 1 : r1 + 1, :] - img[r0 - 1 : r1 - 1, :]) * 0.5
    # Peak row-mean vertical edge near horizon vs sky-band noise std
    row_means = np.mean(gy_strip, axis=1) if gy_strip.size > 0 else np.array([0.0])
    hor_edge = float(np.max(row_means))
    sky_end = max(2, r0)
    sky_band = img[:sky_end, :]
    unsat_sky = sky_band[sky_band < float(cfg.glare_sat_dn)]
    sky_sample = unsat_sky if unsat_sky.size >= 16 else sky_band.ravel()
    sky_med = float(np.median(sky_sample))
    sky_mad = float(np.median(np.abs(sky_sample - sky_med)))
    sigma_sky = max(1.4826 * sky_mad, 1.0)
    snr_linear = max(hor_edge / sigma_sky, 1e-6)
    horizon_snr_db = 20.0 * math.log10(snr_linear)

    # 3. WMO 5%-contrast extinction & MOR = 2.996 / beta_ext
    if beta_ext_override is not None:
        beta_ext = max(float(beta_ext_override), 1e-6)
    else:
        # Koschmieder transmission estimate from airlight veil t = clamp(1 - I_dcp / A, 0.005, 0.995)
        veil_ratio = min(max(dcp_floor / max(cfg.fog_dcp_airlight_dn, 1.0), 0.0), 0.995)
        trans_dcp = max(1.0 - veil_ratio, 0.005)
        beta_dcp = -math.log(trans_dcp) / max(cfg.fog_nominal_range_km, 0.1)
        # Blend with low-contrast attenuation if uniform gray fog has low airlight DN
        norm_contrast = min(max(float(np.std(eo_s4)) / 180.0, 0.005), 1.0)
        beta_contrast = -math.log(norm_contrast) / max(cfg.fog_nominal_range_km, 0.1)
        beta_ext = max(beta_dcp, beta_contrast, 1e-4)

    mor_m = (WMO_MOR_LN20 / beta_ext) * 1000.0
    # Normalized contrast metric in [0, 1] for Stage-3 VLM compatibility
    fog_contrast = min(max(float(np.std(eo_s4)) / 150.0, 0.0), 1.0)
    return dcp_floor, horizon_snr_db, beta_ext, mor_m, fog_contrast


def _compute_lwir_mode5_metrics(
    lwir_u16: np.ndarray,
    cfg: OpticalOddConfig,
) -> tuple[float, float, float, float, float]:
    """Mode 5: 14-bit Pre-AGC std, median, IQR, 512-pt RFFT FPN stripe ratio & 3x3 bad pixels."""
    img = lwir_u16.astype(np.float32, copy=False)
    scene_std = float(np.std(img))
    q25, q50, q75 = np.percentile(img, [25.0, 50.0, 75.0])
    median_dn = float(q50)
    iqr_dn = float(q75 - q25)

    # Column & row FPN stripe HF-PSD ratio via 2 x 512-pt RFFT
    col_profile = np.mean(img, axis=0)
    row_profile = np.mean(img, axis=1)
    n_fft = 512
    c_seg = np.resize(col_profile, n_fft) - float(np.mean(col_profile))
    r_seg = np.resize(row_profile, n_fft) - float(np.mean(row_profile))
    win = np.hanning(n_fft).astype(np.float32)
    c_fft = np.fft.rfft(c_seg * win)
    r_fft = np.fft.rfft(r_seg * win)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    hf_mask = freqs >= 0.15
    c_hf_power = float(np.mean((c_fft.real[hf_mask] ** 2 + c_fft.imag[hf_mask] ** 2))) / n_fft
    r_hf_power = float(np.mean((r_fft.real[hf_mask] ** 2 + r_fft.imag[hf_mask] ** 2))) / n_fft
    scene_var = max(scene_std * scene_std, 1.0)
    fpn_ratio = (c_hf_power + r_hf_power) / scene_var

    # 3x3 neighborhood bad-pixel outlier surge fraction
    if img.shape[0] >= 3 and img.shape[1] >= 3:
        nb_mean = (
            img[:-2, :-2]
            + img[:-2, 1:-1]
            + img[:-2, 2:]
            + img[1:-1, :-2]
            + img[1:-1, 2:]
            + img[2:, :-2]
            + img[2:, 1:-1]
            + img[2:, 2:]
        ) * (1.0 / 8.0)
        center = img[1:-1, 1:-1]
        bad_mask = np.abs(center - nb_mean) > cfg.lwir_bad_pixel_delta_dn
        bad_frac = float(np.mean(bad_mask))
    else:
        bad_frac = 0.0

    return scene_std, median_dn, iqr_dn, fpn_ratio, bad_frac


class _MltlCounterState:
    """O(1) circular ring-buffer counter + G[0, debounce_ticks-1] persistence observer."""

    __slots__ = ("spec", "ring", "running_sum", "debounce_streak")

    def __init__(self, spec: MltlPersistenceSpec) -> None:
        self.spec = spec
        self.ring: deque[int] = deque(maxlen=spec.window_n)
        self.running_sum: int = 0
        self.debounce_streak: int = 0

    def reset(self) -> None:
        self.ring.clear()
        self.running_sum = 0
        self.debounce_streak = 0

    def update(self, raw_trip: bool) -> tuple[int, bool]:
        val = 1 if raw_trip else 0
        if len(self.ring) == self.spec.window_n:
            self.running_sum -= self.ring[0]
        self.ring.append(val)
        self.running_sum += val
        if self.running_sum >= self.spec.confirm_m:
            self.debounce_streak += 1
        else:
            self.debounce_streak = 0
        return self.running_sum, self.debounce_streak >= self.spec.debounce_ticks


class DeterministicOpticalOddMonitor:
    """DR-13 Zero-ML 5-Mode Stage-3 Optical-ODD Monitor with R2U2-compatible MLTL counters."""

    def __init__(self, config: OpticalOddConfig | None = None) -> None:
        self._config: OpticalOddConfig = OpticalOddConfig() if config is None else config
        self._mltl: dict[OpticalOddMode, _MltlCounterState] = {
            mode: _MltlCounterState(self._config.mltl_specs[mode])
            for mode in _ALL_MODES
        }
        self._tile_blur_streaks: np.ndarray | None = None

    @property
    def config(self) -> OpticalOddConfig:
        return self._config

    def reset(self) -> None:
        for state in self._mltl.values():
            state.reset()
        self._tile_blur_streaks = None

    def _check_finite_scalars(self, inp: OpticalOddFrameInput) -> bool:
        scalars = (
            inp.sun_boresight_deg,
            inp.eo_hfov_deg,
            inp.t_win_c,
            inp.t_amb_c,
            inp.rh_pct,
            inp.boson_dt_ffc_s,
            inp.boson_t_cam_c,
            inp.boson_t_last_ffc_c,
            inp.horizon_row_frac,
        )
        for val in scalars:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                return False
        if inp.beta_ext_km_inv is not None and not math.isfinite(inp.beta_ext_km_inv):
            return False
        return True

    def evaluate(self, inp: OpticalOddFrameInput) -> OpticalOddDecision:
        """Execute all 5 classical Optical-ODD checks and update MLTL persistence state."""
        cfg = self._config

        # Fail-closed guard for non-finite telemetry or empty sensor arrays
        if (
            not self._check_finite_scalars(inp)
            or inp.eo_frame_u16.ndim != 2
            or inp.lwir_frame_u16.ndim != 2
            or inp.eo_frame_u16.size == 0
            or inp.lwir_frame_u16.size == 0
        ):
            raw_flags = {m: True for m in _ALL_MODES}
            persisted: dict[OpticalOddMode, bool] = {}
            counts: dict[OpticalOddMode, int] = {}
            for m in _ALL_MODES:
                cnt, active = self._mltl[m].update(True)
                counts[m] = cnt
                persisted[m] = active
            zero_metrics = OpticalOddMetrics(
                droplet_occlusion_frac=1.0,
                mean_tenengrad=0.0,
                mean_mlap=0.0,
                evs_total_events_capped=0,
                mtf_hf_ratio=0.0,
                t_dew_frost_c=0.0,
                window_dew_margin_c=0.0,
                lwir_scene_std_dn=0.0,
                spa_in_glare_cone=True,
                glare_sat_frac=1.0,
                glare_ring_contrast=0.0,
                dcp_floor_dn=1023.0,
                horizon_snr_db=0.0,
                estimated_beta_ext_km_inv=100.0,
                estimated_mor_m=0.0,
                fog_contrast=0.0,
                lwir_median_dn=0.0,
                lwir_iqr_dn=0.0,
                lwir_fpn_ratio=1.0,
                lwir_bad_pixel_frac=1.0,
                boson_dt_ffc_s=9999.0,
                boson_temp_delta_c=99.0,
            )
            return OpticalOddDecision(
                action=OpticalOddAction.DEGRADED_FALLBACK,
                raw_mode_flags=raw_flags,
                persisted_mode_flags=persisted,
                mltl_window_counts=counts,
                health_bitmask=0x1F,
                mask_lwir_frame=True,
                vlm_gate_reject=True,
                ito_heater_cmd=True,
                attitude_in_odd=False,
                metrics=zero_metrics,
                airdata_export_blocked=True,
            )

        eo_s4 = _stride4_eo(inp.eo_frame_u16)

        # --- Hard-capped EVS event slice (N_cap = 65,536 events) ---
        evs_total = 0
        if inp.evs_tile_counts is not None and inp.evs_tile_counts.size > 0:
            evs_raw = np.maximum(inp.evs_tile_counts.astype(np.int64, copy=False), 0)
            raw_sum = int(np.sum(evs_raw))
            evs_total = min(raw_sum, EVS_HARD_CAP_EVENTS)

        # --- Mode 1: Lens Raindrop / Water-Film / Mud / Bug Occlusion ---
        # Evaluate spatial sharpness drop strictly across the textured ground/approach band
        # at and below the 3DEP-projected horizon so featureless clear sky never false-trips.
        hor_row_s4 = int(round(min(max(inp.horizon_row_frac, 0.1), 0.85) * eo_s4.shape[0]))
        eo_ground_s4 = eo_s4[hor_row_s4:, :]
        tenengrad_map, mlap_map = _compute_tenengrad_and_mlap(eo_ground_s4)
        mean_tenengrad = float(np.mean(tenengrad_map))
        mean_mlap = float(np.mean(mlap_map))
        ground_tile_rows = max(4, int(round(cfg.raindrop_tile_rows * (1.0 - min(max(inp.horizon_row_frac, 0.1), 0.85)))))
        t_tiles = _tile_means(tenengrad_map, ground_tile_rows, cfg.raindrop_tile_cols)
        m_tiles = _tile_means(mlap_map, ground_tile_rows, cfg.raindrop_tile_cols)
        low_sharp_tiles = (t_tiles < cfg.raindrop_tenengrad_ratio_thresh * cfg.raindrop_tenengrad_ref) & (
            m_tiles < cfg.raindrop_mlap_ratio_thresh * cfg.raindrop_mlap_ref
        )
        if (
            self._tile_blur_streaks is None
            or self._tile_blur_streaks.shape != low_sharp_tiles.shape
        ):
            self._tile_blur_streaks = np.zeros(low_sharp_tiles.shape, dtype=np.int32)
        self._tile_blur_streaks = np.where(
            low_sharp_tiles, self._tile_blur_streaks + 1, 0
        )
        persisted_blur_tiles = self._tile_blur_streaks >= cfg.raindrop_tile_persist_frames
        droplet_occ_frac = float(np.mean(persisted_blur_tiles))
        raw_mode1 = droplet_occ_frac > cfg.raindrop_occluded_tile_frac_thresh

        # --- Mode 5 (shared single-pass LWIR scan feeds Mode 2 & Mode 4 too) ---
        lwir_std, lwir_med, lwir_iqr, lwir_fpn_ratio, lwir_bad_frac = (
            _compute_lwir_mode5_metrics(inp.lwir_frame_u16, cfg)
        )
        boson_temp_delta = abs(float(inp.boson_t_cam_c) - float(inp.boson_t_last_ffc_c))
        lwir_median_drift = abs(lwir_med - cfg.lwir_ref_median_dn)
        # Mode 5 raw trip (suppressed during 1-tick shutter FFC freeze to avoid spurious fallback)
        if inp.boson_ffc_freeze:
            raw_mode5 = False
        else:
            raw_mode5 = (
                (lwir_median_drift > cfg.lwir_median_drift_dn_thresh)
                or (lwir_iqr < cfg.lwir_min_iqr_dn and lwir_std < cfg.icing_ir_std_mult * cfg.lwir_netd_dn)
                or (lwir_fpn_ratio > cfg.lwir_fpn_ratio_thresh)
                or (lwir_bad_frac > cfg.lwir_bad_pixel_thresh)
                or (inp.boson_dt_ffc_s > cfg.lwir_stale_ffc_s)
                or (boson_temp_delta > cfg.lwir_ffc_temp_delta_c)
            )

        # --- Mode 2: Optical Window Icing / Frost / Condensation (HARD AF447 ZERO-AIRDATA BAN) ---
        mtf_hf_ratio = _compute_mtf_hf_ratio(inp.eo_frame_u16)
        t_dew_frost = compute_buck_dew_frost_point_c(inp.t_amb_c, inp.rh_pct)
        dew_margin = float(inp.t_win_c) - t_dew_frost
        mtf_dropped = mtf_hf_ratio < (cfg.icing_mtf_ratio_thresh * cfg.icing_mtf_ref_eta0)
        ir_opaque = lwir_std < (cfg.icing_ir_std_mult * cfg.lwir_netd_dn)
        buck_margin_breached = dew_margin <= cfg.icing_dew_margin_c
        raw_mode2 = bool(mtf_dropped and ir_opaque and buck_margin_breached)

        # --- Mode 3: Direct Solar Veiling Glare / Blooming (SPA-gated) ---
        spa_cone_deg = (float(inp.eo_hfov_deg) * 0.5) + cfg.glare_spa_margin_deg
        spa_in_cone = float(inp.sun_boresight_deg) < spa_cone_deg
        sat_mask = eo_s4 >= float(cfg.glare_sat_dn)
        glare_sat_frac = float(np.mean(sat_mask))
        glare_ring_contrast = _compute_solar_ring_contrast(
            eo_s4, sat_mask, inp.sun_centroid_uv
        )
        raw_mode3 = bool(
            spa_in_cone
            and (
                (glare_sat_frac > cfg.glare_sat_frac_thresh)
                or (glare_ring_contrast < cfg.glare_ring_contrast_ratio_thresh * cfg.glare_ring_contrast_ref)
            )
        )

        # --- Mode 4: Low-Contrast Fog / Smoke / Visibility Loss (WMO MOR + 3DEP Horizon SNR) ---
        dcp_floor, horizon_snr_db, beta_ext, mor_m, fog_contrast = _compute_fog_metrics(
            inp.eo_frame_u16, eo_s4, inp.horizon_row_frac, cfg, inp.beta_ext_km_inv
        )
        raw_mode4 = bool(
            (mor_m < cfg.fog_mor_thresh_m) or (horizon_snr_db < cfg.fog_horizon_snr_db_thresh)
        )

        raw_flags: dict[OpticalOddMode, bool] = {
            OpticalOddMode.RAINDROP_OCCLUSION: raw_mode1,
            OpticalOddMode.WINDOW_ICING: raw_mode2,
            OpticalOddMode.SOLAR_GLARE: raw_mode3,
            OpticalOddMode.FOG_VISIBILITY: raw_mode4,
            OpticalOddMode.LWIR_BOLOMETER_DRIFT: raw_mode5,
        }

        # --- Correlated-Block M-of-N MLTL Persistence Update (r2u2_core C2PO logic) ---
        persisted: dict[OpticalOddMode, bool] = {}
        counts: dict[OpticalOddMode, int] = {}
        bitmask = 0
        for mode in _ALL_MODES:
            cnt, active = self._mltl[mode].update(raw_flags[mode])
            counts[mode] = cnt
            persisted[mode] = active
            if active:
                bitmask |= 1 << (int(mode) - 1)

        eo_occluded = persisted[OpticalOddMode.RAINDROP_OCCLUSION]
        win_iced = persisted[OpticalOddMode.WINDOW_ICING]
        eo_glared = persisted[OpticalOddMode.SOLAR_GLARE]
        eo_fogged = persisted[OpticalOddMode.FOG_VISIBILITY]
        ir_degraded = persisted[OpticalOddMode.LWIR_BOLOMETER_DRIFT]
        p_ir_healthy = bool(
            (lwir_std > cfg.fog_ir_healthy_std_mult * cfg.lwir_netd_dn)
            and not ir_degraded
            and not inp.boson_ffc_freeze
        )

        # Exact DR-13 §1.4 C2PO FTSPEC routing (F0..F3)
        if win_iced or not inp.attitude_in_odd or ((eo_fogged or eo_occluded) and not p_ir_healthy):
            action = OpticalOddAction.DEGRADED_FALLBACK
        elif (eo_fogged or eo_occluded) and p_ir_healthy:
            action = OpticalOddAction.THERMAL_PRIMARY
        elif eo_glared:
            action = OpticalOddAction.EVS_LWIR_ELEVATED
        else:
            action = OpticalOddAction.NOMINAL

        vlm_gate_reject = bool(
            win_iced or eo_glared or eo_occluded or (eo_fogged and not p_ir_healthy) or not inp.attitude_in_odd
        )

        metrics = OpticalOddMetrics(
            droplet_occlusion_frac=droplet_occ_frac,
            mean_tenengrad=mean_tenengrad,
            mean_mlap=mean_mlap,
            evs_total_events_capped=evs_total,
            mtf_hf_ratio=mtf_hf_ratio,
            t_dew_frost_c=t_dew_frost,
            window_dew_margin_c=dew_margin,
            lwir_scene_std_dn=lwir_std,
            spa_in_glare_cone=spa_in_cone,
            glare_sat_frac=glare_sat_frac,
            glare_ring_contrast=glare_ring_contrast,
            dcp_floor_dn=dcp_floor,
            horizon_snr_db=horizon_snr_db,
            estimated_beta_ext_km_inv=beta_ext,
            estimated_mor_m=mor_m,
            fog_contrast=fog_contrast,
            lwir_median_dn=lwir_med,
            lwir_iqr_dn=lwir_iqr,
            lwir_fpn_ratio=lwir_fpn_ratio,
            lwir_bad_pixel_frac=lwir_bad_frac,
            boson_dt_ffc_s=float(inp.boson_dt_ffc_s),
            boson_temp_delta_c=boson_temp_delta,
        )

        return OpticalOddDecision(
            action=action,
            raw_mode_flags=raw_flags,
            persisted_mode_flags=persisted,
            mltl_window_counts=counts,
            health_bitmask=bitmask,
            mask_lwir_frame=bool(inp.boson_ffc_freeze),
            vlm_gate_reject=vlm_gate_reject,
            ito_heater_cmd=bool(win_iced),
            attitude_in_odd=bool(inp.attitude_in_odd),
            metrics=metrics,
            airdata_export_blocked=True,
        )
