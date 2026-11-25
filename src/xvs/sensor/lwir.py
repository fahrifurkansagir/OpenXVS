"""Physics-based 8-14 um Long-Wave Infrared (LWIR) forward sensor model (DR-04 / SP2 §3.3).

Designed to run co-registered with the WP-03 EMVA-1288 EO camera chain (`src/xvs/sensor/`):
  - Exact 5-point Gauss-Legendre Planck blackbody band radiance integration over [8.0, 14.0] um
    (< 1e-5% relative error vs `scipy.integrate.quad` across 200 K..1200 K).
  - Microphysics atmospheric extinction (`clear`, `radiation_fog`, `advection_fog`, `stratus_cloud`,
    `heavy_rain`) separating small-droplet radiation fog (> 10x LWIR-over-EO SNR advantage) from
    large-droplet coastal advection fog and heavy rain (where LWIR degrades honestly below EO).
  - Kasten-Young cold-space sky radiance gradient & Kirchhoff surface leaving radiance.
  - Sub-pixel area-averaged EO -> 640x512 downsampling, diffraction-limited Airy PSF, microbolometer
    thermal inertia (`tau_th = 0.010 s`), NETD/BLIP temporal noise, PRNU, column FPN, and NUC drift.
  - Four explicit thermal fault injectors for WP-07 integrity monitoring:
    `thermal_crossover`, `cloud_washout`, `sun_in_fov_bloom`, and `ffc_freeze`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

# CODATA 2018 exact SI physical constants
H_PLANCK: float = 6.62607015e-34  # J * s
C_LIGHT: float = 2.99792458e8  # m / s
K_BOLTZ: float = 1.380649e-23  # J / K
C1L: float = 2.0 * H_PLANCK * (C_LIGHT**2)  # 1.1910429723971884e-16 W * m^2 / sr
C2: float = (H_PLANCK * C_LIGHT) / K_BOLTZ  # 1.4387768775039338e-2 m * K

LWIR_BAND_MIN_M: float = 8.0e-6
LWIR_BAND_MAX_M: float = 14.0e-6
LWIR_EFF_WAVELENGTH_UM: float = 10.5

# Analytical 5-point Gauss-Legendre nodes & weights on [-1, +1] (full float64 precision)
_SQRT_10_7: float = math.sqrt(10.0 / 7.0)
_X1: float = math.sqrt(5.0 - 2.0 * _SQRT_10_7) / 3.0
_X2: float = math.sqrt(5.0 + 2.0 * _SQRT_10_7) / 3.0
_SQRT_70: float = math.sqrt(70.0)
_W0: float = 128.0 / 225.0
_W1: float = (322.0 + 13.0 * _SQRT_70) / 900.0
_W2: float = (322.0 - 13.0 * _SQRT_70) / 900.0

_GL_HALF_WIDTH: float = 0.5 * (LWIR_BAND_MAX_M - LWIR_BAND_MIN_M)  # 3.0e-6 m
_GL_MIDPOINT: float = 0.5 * (LWIR_BAND_MAX_M + LWIR_BAND_MIN_M)  # 11.0e-6 m

_GL_NODES_UNIT: tuple[float, ...] = (-_X2, -_X1, 0.0, _X1, _X2)
_GL_WEIGHTS_UNIT: tuple[float, ...] = (_W2, _W1, _W0, _W1, _W2)

_GL_LAM_TUPLE: tuple[float, ...] = tuple(
    _GL_MIDPOINT + _GL_HALF_WIDTH * x for x in _GL_NODES_UNIT
)
_GL_W_TUPLE: tuple[float, ...] = tuple(_GL_HALF_WIDTH * w for w in _GL_WEIGHTS_UNIT)

_GL_LAM_F64: torch.Tensor = torch.tensor(_GL_LAM_TUPLE, dtype=torch.float64)
_GL_W_F64: torch.Tensor = torch.tensor(_GL_W_TUPLE, dtype=torch.float64)

# Reference dL_BB/dT at T = 300.0 K over [8, 14] um [W / (m^2 * sr * K)]
DLDT_300K: float = 0.83782018

AtmosphericRegime = Literal[
    "clear",
    "clear_haze",
    "radiation_fog",
    "advection_fog",
    "coastal_advection_fog",
    "stratus_cloud",
    "heavy_rain",
]

DetectorType = Literal["microbolometer", "hgcdte"]

_KAPPA_BY_REGIME: dict[str, float] = {
    "clear": 0.08,
    "clear_haze": 0.08,
    "radiation_fog": 0.32,
    "advection_fog": 0.65,
    "coastal_advection_fog": 0.65,
    "stratus_cloud": 0.95,
    "heavy_rain": 0.08,  # Rain extinction handled explicitly by Marshall-Palmer geometric term
}


@dataclass(frozen=True)
class LwirSensorConfig:
    """WP-03 compatible LWIR sensor configuration."""

    detector_type: DetectorType = "microbolometer"
    netd_mk: float = 40.0
    f_num: float = 1.2
    pitch_um: float = 12.0
    t_sat_k: float = 423.15
    tau_th_s: float = 0.010
    dt_s: float = 1.0 / 30.0
    out_hw: tuple[int, int] = (512, 640)
    prnu_std: float = 0.005
    fpn_col_ratio: float = 0.35
    nuc_drift_k: float = 0.0


@dataclass(frozen=True)
class ThermalFaultConfig:
    """Explicit thermal fault injection controls for WP-07 health monitor testing."""

    thermal_crossover: bool = False
    crossover_residual_k: float = 0.02
    cloud_washout: bool = False
    cloud_washout_beta_km_inv: float = 12.0
    sun_in_fov_bloom: bool = False
    sun_uv_px: tuple[float, float] = (320.0, 128.0)
    sun_disk_radius_px: float = 2.6
    sun_halo_sigma_px: float = 12.0
    sun_direct_rad_w_m2_sr: float = 15556.8
    ffc_freeze: bool = False


def planck_spectral_radiance_scalar(lam_m: float, temp_k: float) -> float:
    """Exact Planck spectral radiance B(lambda, T) [W / (m^2 * sr * m)]."""
    if lam_m <= 0.0 or temp_k <= 0.0:
        raise ValueError(f"lam_m ({lam_m}) and temp_k ({temp_k}) must be strictly positive")
    x = C2 / (lam_m * temp_k)
    return C1L / ((lam_m**5) * math.expm1(x))


def planck_lwir_8_14_scalar(temp_k: float) -> float:
    """Exact 5-point Gauss-Legendre integrated 8-14 um blackbody radiance [W / (m^2 * sr)]."""
    t = min(max(float(temp_k), 150.0), 6000.0)
    total = 0.0
    for lam, w in zip(_GL_LAM_TUPLE, _GL_W_TUPLE):
        total += w * planck_spectral_radiance_scalar(lam, t)
    return total


def planck_lwir_8_14_dldt_scalar(temp_k: float = 300.0) -> float:
    """Analytical temperature derivative dL_BB/dT [W / (m^2 * sr * K)] over [8, 14] um."""
    t = min(max(float(temp_k), 150.0), 6000.0)
    total = 0.0
    for lam, w in zip(_GL_LAM_TUPLE, _GL_W_TUPLE):
        x = C2 / (lam * t)
        ex = math.exp(x)
        exm1 = math.expm1(x)
        db_dt = C1L / (lam**5) * (ex * x / (t * exm1 * exm1))
        total += w * db_dt
    return total


def planck_lwir_8_14(temp_k: torch.Tensor) -> torch.Tensor:
    """Vectorized 5-point Gauss-Legendre 8-14 um blackbody radiance [W / (m^2 * sr)].

    Computes the quadrature in float64 internally to preserve < 1e-5% accuracy
    and casts back to `temp_k.dtype`.
    """
    orig_dtype = temp_k.dtype
    t64 = temp_k.to(dtype=torch.float64).clamp(min=150.0, max=6000.0).unsqueeze(-1)
    lam = _GL_LAM_F64.to(device=t64.device)
    w = _GL_W_F64.to(device=t64.device)
    b_spec = C1L / (lam.pow(5) * torch.expm1(C2 / (lam * t64)))
    out64 = (b_spec * w).sum(dim=-1)
    return out64.to(dtype=orig_dtype)


def eo_extinction_km_inv(
    rvr_km: float,
    regime: AtmosphericRegime = "clear",
    rain_mm_h: float = 0.0,
) -> float:
    """Visible EO Koschmieder / Marshall-Palmer extinction beta_EO [km^-1]."""
    if regime not in _KAPPA_BY_REGIME:
        raise ValueError(f"Unknown atmospheric regime: {regime!r}")
    if regime == "heavy_rain":
        r_eff = rain_mm_h if rain_mm_h > 0.0 else 25.0
        beta_rain = 0.365 * (r_eff**0.63)
        beta_bg = 0.05  # Clear background aerosol
        return beta_bg + beta_rain
    beta_vis = 3.912 / max(float(rvr_km), 0.03)
    beta_rain = 0.365 * (rain_mm_h**0.63) if rain_mm_h > 0.0 else 0.0
    return beta_vis + beta_rain


def lwir_extinction_km_inv(
    rvr_km: float,
    regime: AtmosphericRegime = "radiation_fog",
    rain_mm_h: float = 0.0,
    beta_mol: float = 0.18,
    kappa_override: float | None = None,
) -> float:
    """Microphysics-aware 8-14 um LWIR extinction beta_LWIR [km^-1] (DR-04 §3.2).

    Separates:
      - `clear` / `clear_haze`: low aerosol scattering (kappa = 0.08) + H2O continuum.
      - `radiation_fog`: small droplets (r_eff ~ 3.5 um, kappa = 0.32) -> large LWIR advantage.
      - `advection_fog` / `coastal_advection_fog`: large droplets (r_eff ~ 10 um, kappa = 0.65).
      - `stratus_cloud`: thick liquid water cloud (kappa = 0.95) -> strong LWIR attenuation.
      - `heavy_rain`: geometric-optics raindrop extinction (0.365 * R^0.63) identical for EO/LWIR
        PLUS humid H2O continuum absorption (beta_mol ~ 0.25 km^-1), so tau_LWIR < tau_EO honestly.
    """
    if regime not in _KAPPA_BY_REGIME:
        raise ValueError(f"Unknown atmospheric regime: {regime!r}")
    if regime == "heavy_rain":
        r_eff = rain_mm_h if rain_mm_h > 0.0 else 25.0
        beta_rain = 0.365 * (r_eff**0.63)
        beta_mol_humid = max(float(beta_mol), 0.25)
        return 0.08 * 0.05 + beta_mol_humid + beta_rain
    kappa = _KAPPA_BY_REGIME[regime] if kappa_override is None else float(kappa_override)
    beta_eo_fog = 3.912 / max(float(rvr_km), 0.03)
    beta_rain = 0.365 * (rain_mm_h**0.63) if rain_mm_h > 0.0 else 0.0
    return kappa * beta_eo_fog + float(beta_mol) + beta_rain


def compute_eo_lwir_regime_metrics(
    rvr_km: float,
    range_km: float = 1.0,
    regime: AtmosphericRegime = "radiation_fog",
    rain_mm_h: float = 0.0,
    delta_t_k: float = 12.0,
    emissivity: float = 0.90,
    netd_mk: float = 40.0,
    eo_intrinsic_snr: float = 50.0,
) -> dict[str, float]:
    """Compare EO vs LWIR transmittance and Contrast-to-Noise Ratio (CNR/SNR) at range `range_km`."""
    beta_eo = eo_extinction_km_inv(rvr_km, regime=regime, rain_mm_h=rain_mm_h)
    beta_lwir = lwir_extinction_km_inv(rvr_km, regime=regime, rain_mm_h=rain_mm_h)
    tau_eo = math.exp(-beta_eo * range_km)
    tau_lwir = math.exp(-beta_lwir * range_km)

    # EO contrast decays with Koschmieder airlight factor tau_eo
    snr_eo = eo_intrinsic_snr * tau_eo

    # LWIR CNR: tau_lwir * emissivity * delta_T / NETD_K (wet surface cooling reduces rain delta_T)
    effective_delta_t = delta_t_k * (0.55 if regime == "heavy_rain" else 1.0)
    netd_k = max(netd_mk * 1e-3, 1e-6)
    snr_lwir = tau_lwir * emissivity * effective_delta_t / netd_k
    snr_ratio = snr_lwir / max(snr_eo, 1e-12)
    tau_ratio = tau_lwir / max(tau_eo, 1e-12)
    return {
        "beta_eo_km_inv": beta_eo,
        "beta_lwir_km_inv": beta_lwir,
        "tau_eo": tau_eo,
        "tau_lwir": tau_lwir,
        "tau_lwir_over_eo": tau_ratio,
        "snr_eo": snr_eo,
        "snr_lwir": snr_lwir,
        "snr_lwir_over_eo": snr_ratio,
    }


def inject_thermal_crossover(
    temp_k_hr: torch.Tensor,
    t_air_k: float,
    residual_delta_k: float = 0.02,
) -> torch.Tensor:
    """Dawn/dusk diurnal thermal crossover fault: collapses surface temperature contrast toward `t_air_k`."""
    mean_t = temp_k_hr.mean()
    dev = temp_k_hr - mean_t
    max_dev = dev.abs().max().clamp(min=1e-6)
    scaled_dev = dev * (residual_delta_k / max_dev)
    return torch.full_like(temp_k_hr, float(t_air_k)) + scaled_dev


def inject_sun_in_fov_bloom(
    rad_lwir: torch.Tensor,
    sun_uv_px: tuple[float, float],
    disk_radius_px: float = 2.6,
    halo_sigma_px: float = 12.0,
    sun_direct_rad_w_m2_sr: float = 15556.8,
    l_sat: float | None = None,
) -> torch.Tensor:
    """Inject 5778 K direct solar disk + diffraction scattering bloom halo into LWIR radiance frame."""
    h, w = rad_lwir.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=rad_lwir.device, dtype=rad_lwir.dtype),
        torch.arange(w, device=rad_lwir.device, dtype=rad_lwir.dtype),
        indexing="ij",
    )
    u0, v0 = float(sun_uv_px[0]), float(sun_uv_px[1])
    dist2 = (xx - u0).pow(2) + (yy - v0).pow(2)
    disk_mask = dist2 <= (disk_radius_px**2)
    # Diffraction / optical scattering halo around saturated solar disk
    halo = (0.04 * sun_direct_rad_w_m2_sr) * torch.exp(
        -0.5 * dist2 / max(halo_sigma_px**2, 1.0)
    )
    bloomed = rad_lwir + halo
    bloomed = torch.where(
        disk_mask,
        bloomed + sun_direct_rad_w_m2_sr,
        bloomed,
    )
    if l_sat is not None:
        bloomed = bloomed.clamp(min=0.0, max=float(l_sat))
    return bloomed


def build_airy_psf_kernel_1d(
    f_num: float,
    pitch_um: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Separable 1D Gaussian approximation to 10.5 um diffraction-limited Airy PSF (DR-04 §3.4)."""
    sigma_px = (0.42 * LWIR_EFF_WAVELENGTH_UM * float(f_num)) / max(float(pitch_um), 1e-3)
    k = max(3, int(2 * math.ceil(3.0 * sigma_px) + 1))
    coords = torch.arange(k, device=device, dtype=dtype) - (k // 2)
    g1d = torch.exp(-0.5 * (coords / max(sigma_px, 1e-3)).pow(2))
    return g1d / g1d.sum()


def compute_lwir_agc8(rad_lwir: torch.Tensor) -> torch.Tensor:
    """1st-99th percentile linear AGC stretch to 8-bit [0, 255] for pilot HUD visualization."""
    flat = rad_lwir.reshape(-1).to(dtype=torch.float32)
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    span = (hi - lo).clamp(min=1e-6)
    norm = ((rad_lwir.to(dtype=torch.float32) - lo) / span).clamp(0.0, 1.0)
    return (norm * 255.0).round().to(dtype=torch.uint8)


def build_lwir_two_channel_input(
    rad_lwir: torch.Tensor,
    l_sat: float | None = None,
) -> torch.Tensor:
    """Build 2-channel `[norm_radiance, local_contrast]` tensor `[..., 2, H, W]` for XVS-ThermalNet.

    Matches DR-04 §4.2:
      - Ch 0: 14-bit log-normalized radiance in [0, 1] (clipped safely).
      - Ch 1: local high-pass thermal contrast (`rad - box_blur_7x7(rad)` normalized by robust std).
    """
    if rad_lwir.ndim == 2:
        x = rad_lwir.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    elif rad_lwir.ndim == 3:
        x = rad_lwir.unsqueeze(1)
        squeeze_batch = False
    elif rad_lwir.ndim == 4 and rad_lwir.shape[1] == 1:
        x = rad_lwir
        squeeze_batch = False
    else:
        raise ValueError(f"Expected 2D/3D/4D(C=1) rad_lwir tensor, got shape {tuple(rad_lwir.shape)}")

    sat = planck_lwir_8_14_scalar(423.15) if l_sat is None else float(l_sat)
    ch0 = torch.log1p(x.clamp(min=0.0)) / math.log1p(max(sat, 1.0))

    local_mean = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    hp = x - local_mean
    std = hp.flatten(1).std(dim=1, unbiased=False).clamp(min=1e-3).view(-1, 1, 1, 1)
    ch1 = (hp / std).clamp(-5.0, 5.0)
    out = torch.cat([ch0, ch1], dim=1)
    return out.squeeze(0) if squeeze_batch else out


def compute_wp07_thermal_health(
    rad_lwir: torch.Tensor,
    dn14: torch.Tensor,
    ner: float,
    prev_rad_lwir: torch.Tensor | None = None,
) -> dict[str, float]:
    """Deterministic WP-07 thermal integrity observables & physical health score `q_lwir_phys`."""
    rad_f = rad_lwir.to(dtype=torch.float32)
    gx = rad_f[:, 1:] - rad_f[:, :-1]
    gy = rad_f[1:, :] - rad_f[:-1, :]
    grad_rms = float(torch.sqrt(0.5 * (gx.pow(2).mean() + gy.pow(2).mean())).item())
    dynamic_range_k = float((rad_f.max() - rad_f.min()).item()) / DLDT_300K
    sat_frac = float((dn14 >= 16382.5).to(dtype=torch.float32).mean().item())
    sat_pixels = int((dn14 >= 16382.5).sum().item())

    if prev_rad_lwir is not None:
        diff_var = float((rad_f - prev_rad_lwir.to(dtype=torch.float32)).var(unbiased=False).item())
    else:
        diff_var = float("nan")

    # Deterministic failure checks (DR-04 §3.5)
    crossover_fault = grad_rms < (2.5 * ner)
    washout_fault = dynamic_range_k < 1.5
    bloom_fault = sat_pixels >= 12
    ffc_freeze_fault = (prev_rad_lwir is not None) and (diff_var < 1e-12)

    if crossover_fault or washout_fault or ffc_freeze_fault:
        q_lwir_phys = 0.0
    elif bloom_fault:
        q_lwir_phys = max(0.0, 1.0 - min(1.0, sat_frac * 50.0))
    else:
        q_lwir_phys = 1.0

    return {
        "grad_rms_w_m2_sr": grad_rms,
        "ner_w_m2_sr": float(ner),
        "grad_to_ner_ratio": grad_rms / max(float(ner), 1e-9),
        "dynamic_range_k": dynamic_range_k,
        "saturated_pixels": float(sat_pixels),
        "saturated_fraction": sat_frac,
        "interframe_diff_var": diff_var,
        "crossover_fault": float(crossover_fault),
        "washout_fault": float(washout_fault),
        "bloom_fault": float(bloom_fault),
        "ffc_freeze_fault": float(ffc_freeze_fault),
        "q_lwir_phys": float(q_lwir_phys),
    }


def simulate_lwir_frame(
    temp_k_hr: torch.Tensor,
    emiss_hr: torch.Tensor,
    range_km_hr: torch.Tensor,
    elev_deg_hr: torch.Tensor,
    t_air_k: float,
    rvr_km: float,
    regime: AtmosphericRegime = "radiation_fog",
    rain_mm_h: float = 0.0,
    cloud_cover: float = 0.0,
    sensor_cfg: LwirSensorConfig | None = None,
    fault_cfg: ThermalFaultConfig | None = None,
    prev_bol_rad: torch.Tensor | None = None,
    prev_meas_rad: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    noise_off: bool = False,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """WP-03 co-registered EO-grid -> 640x512 LWIR forward pass (14-bit raw DN, radiance, AGC, WP-07)."""
    cfg = LwirSensorConfig() if sensor_cfg is None else sensor_cfg
    flt = ThermalFaultConfig() if fault_cfg is None else fault_cfg

    device = temp_k_hr.device
    dtype = temp_k_hr.dtype
    l_sat_t = planck_lwir_8_14(torch.tensor(cfg.t_sat_k, device=device, dtype=dtype))
    l_sat = float(l_sat_t.item())
    ner = (cfg.netd_mk * 1e-3) * DLDT_300K

    # Fault 4: Mechanical FFC shutter freeze (identically repeats previous frame -> diff_var == 0)
    if flt.ffc_freeze and prev_meas_rad is not None:
        frozen_rad = prev_meas_rad.clone()
        frozen_dn14 = (frozen_rad / l_sat_t * 16383.0).clamp(0.0, 16383.0).round()
        frozen_bol = prev_bol_rad.clone() if prev_bol_rad is not None else frozen_rad.clone()
        wp07 = compute_wp07_thermal_health(
            frozen_rad, frozen_dn14, ner=ner, prev_rad_lwir=prev_meas_rad
        )
        return {
            "rad_lwir": frozen_rad,
            "dn14": frozen_dn14,
            "agc8": compute_lwir_agc8(frozen_rad),
            "L_bol": frozen_bol,
            "two_channel": build_lwir_two_channel_input(frozen_rad, l_sat=l_sat),
            "wp07_metrics": wp07,
        }

    # Fault 1: Thermal crossover collapses surface temperature contrast toward ambient air temperature
    temp_scene = (
        inject_thermal_crossover(
            temp_k_hr, t_air_k=t_air_k, residual_delta_k=flt.crossover_residual_k
        )
        if flt.thermal_crossover
        else temp_k_hr
    )

    # 1. Cold-space Kasten-Young sky gradient & Kirchhoff surface leaving radiance on high-res EO grid
    el = elev_deg_hr.clamp(min=0.0)
    airmass = (
        1.0 / (torch.sin(torch.deg2rad(el)) + 0.15 * (el + 3.885).pow(-1.253))
    ).clamp(max=38.0)
    eps_clear = 1.0 - (1.0 - 0.30) ** airmass
    eff_cloud = 1.0 if flt.cloud_washout else float(cloud_cover)
    l_air = planck_lwir_8_14(torch.tensor(t_air_k, device=device, dtype=dtype))
    l_sky = ((1.0 - eff_cloud) * eps_clear + eff_cloud * 0.995) * l_air
    l_surf = planck_lwir_8_14(temp_scene)
    l_obj = emiss_hr * l_surf + (1.0 - emiss_hr) * (0.5 * l_sky + 0.5 * l_air)

    # 2. Microphysics Beer-Lambert transmittance + thermal path emission
    if flt.cloud_washout:
        beta = float(flt.cloud_washout_beta_km_inv)
    else:
        beta = lwir_extinction_km_inv(rvr_km, regime=regime, rain_mm_h=rain_mm_h)
    tau = torch.exp(-beta * range_km_hr.clamp(min=0.0))
    if flt.cloud_washout:
        # In optically thick liquid-water cloud, both sky and ground wash out to ambient L_air
        l_at_sensor = torch.where(
            range_km_hr > 50.0,
            l_air.expand_as(l_sky),
            tau * l_obj + (1.0 - tau) * l_air,
        )
    else:
        l_at_sensor = torch.where(
            range_km_hr > 50.0,
            l_sky,
            tau * l_obj + (1.0 - tau) * l_air,
        )

    # 3. Sub-pixel area-averaged downsample to native LWIR grid, then separable Airy PSF convolution
    # Use replicate border padding so uniform scenes have zero artificial edge falloff.
    l_lr = F.interpolate(l_at_sensor[None, None], size=cfg.out_hw, mode="area")
    g1d = build_airy_psf_kernel_1d(cfg.f_num, cfg.pitch_um, device=device, dtype=dtype)
    k = int(g1d.numel())
    pad = k // 2
    l_padded = F.pad(l_lr, (pad, pad, pad, pad), mode="replicate")
    l_opt = F.conv2d(
        F.conv2d(l_padded, g1d[None, None, :, None], padding=0),
        g1d[None, None, None, :],
        padding=0,
    )[0, 0]

    # Fault 3: Sun-in-FOV direct radiance saturation + diffraction scattering halo
    if flt.sun_in_fov_bloom:
        l_opt = inject_sun_in_fov_bloom(
            l_opt,
            sun_uv_px=flt.sun_uv_px,
            disk_radius_px=flt.sun_disk_radius_px,
            halo_sigma_px=flt.sun_halo_sigma_px,
            sun_direct_rad_w_m2_sr=flt.sun_direct_rad_w_m2_sr,
            l_sat=None,  # Clamped after bolometer inertia & readout
        )

    # 4. Microbolometer causal IIR thermal inertia (tau_th = 0.010 s) or HgCdTe instantaneous response
    alpha = (
        1.0
        if cfg.detector_type == "hgcdte"
        else 1.0 - math.exp(-cfg.dt_s / max(cfg.tau_th_s, 1e-6))
    )
    l_bol = l_opt if prev_bol_rad is None else (1.0 - alpha) * prev_bol_rad + alpha * l_opt

    # 5. NETD / BLIP temporal noise + PRNU + column FPN + NUC thermal drift
    if noise_off:
        l_noisy = l_bol
    else:
        if cfg.detector_type == "hgcdte":
            l_300 = planck_lwir_8_14(torch.tensor(300.0, device=device, dtype=dtype))
            z_temp = torch.randn(l_bol.shape, device=device, dtype=dtype, generator=generator)
            temp_noise = z_temp * ner * (l_bol.clamp(min=1.0) / l_300).sqrt()
        else:
            z_temp = torch.randn(l_bol.shape, device=device, dtype=dtype, generator=generator)
            temp_noise = z_temp * ner

        z_prnu = torch.randn(l_bol.shape, device=device, dtype=dtype, generator=generator)
        prnu = z_prnu * cfg.prnu_std
        z_col = torch.randn((1, cfg.out_hw[1]), device=device, dtype=dtype, generator=generator)
        col_fpn = z_col * (cfg.fpn_col_ratio * ner)
        if cfg.nuc_drift_k != 0.0:
            z_nuc = torch.randn(l_bol.shape, device=device, dtype=dtype, generator=generator)
            nuc_err = z_nuc * (cfg.nuc_drift_k * DLDT_300K)
        else:
            nuc_err = torch.zeros_like(l_bol)

        l_noisy = (1.0 + prnu) * l_bol + temp_noise + col_fpn + nuc_err

    rad_clamped = l_noisy.clamp(min=0.0, max=l_sat)
    dn14 = (rad_clamped / l_sat_t * 16383.0).clamp(0.0, 16383.0).round()
    wp07 = compute_wp07_thermal_health(
        rad_clamped, dn14, ner=ner, prev_rad_lwir=prev_meas_rad
    )

    return {
        "rad_lwir": rad_clamped,
        "dn14": dn14,
        "agc8": compute_lwir_agc8(rad_clamped),
        "L_bol": l_bol,
        "two_channel": build_lwir_two_channel_input(rad_clamped, l_sat=l_sat),
        "wp07_metrics": wp07,
    }
