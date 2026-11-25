"""Sensor noise model conforming to EMVA 1288 linear model.

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 5 and §3 item V5.
- Mean photo-electrons: mu_e = L * E
- Temporal noise:
  - photo shot noise: Poisson(mu_e * (1 + PRNU_map))
  - dark signal: Poisson(D * t_exp)
  - read noise: Normal(0, sigma_r^2)
- Fixed pattern noise: DSNU map (e-), PRNU map (relative gain).
  Seeded by SHA-256 of sensor_id.
- Temporal seed: SHA-256 of sensor_id|image_id|frame.
- Saturation at full well FW (e-).
- ADC: DN = clip(round(K*e + black_level), 0, 2^bits - 1).
"""

from __future__ import annotations

import hashlib
import numpy as np


def compute_seed_fpn(sensor_id: str) -> int:
    """Compute 64-bit integer seed for fixed-pattern noise from SHA-256 of sensor_id."""
    digest = hashlib.sha256(sensor_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def compute_seed_temporal(sensor_id: str, image_id: str, frame: int) -> int:
    """Compute 64-bit integer seed for temporal noise from SHA-256 of sensor_id|image_id|frame."""
    key = f"{sensor_id}|{image_id}|{frame}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def generate_fpn_maps(
    sensor_id: str,
    height: int,
    width: int,
    prnu_rms: float,
    dsnu_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed-pattern noise maps (PRNU and DSNU) for a given sensor_id.

    Parameters
    ----------
    sensor_id : str
        Unique identifier of the sensor.
    height, width : int
        Map dimensions.
    prnu_rms : float
        Photo-response non-uniformity RMS (relative gain, dimensionless).
    dsnu_rms : float
        Dark signal non-uniformity RMS in electrons.

    Returns
    -------
    prnu_map : np.ndarray, shape (height, width), float64
        PRNU relative gain variation (mean 0, std prnu_rms).
    dsnu_map : np.ndarray, shape (height, width), float64
        DSNU offset variation in electrons (mean 0, std dsnu_rms).
    """
    seed_fpn = compute_seed_fpn(sensor_id)
    rng = np.random.Generator(np.random.PCG64(seed_fpn))
    prnu_map = rng.normal(0.0, prnu_rms, (height, width))
    dsnu_map = rng.normal(0.0, dsnu_rms, (height, width))
    return prnu_map, dsnu_map


def compute_auto_exposure(
    linear_radiance: np.ndarray,
    full_well_e: float,
    ae_target: float = 0.6,
) -> float:
    """Compute electrons E at L=1 so that the 99th percentile sits at ae_target * full_well.

    Parameters
    ----------
    linear_radiance : np.ndarray
        Linear radiance L in [0, 1].
    full_well_e : float
        Full well capacity in electrons.
    ae_target : float
        Target fraction of full well (default 0.6).

    Returns
    -------
    float
        Electrons E at L = 1.
    """
    p99 = float(np.percentile(linear_radiance, 99.0))
    if p99 > 1e-6:
        e_scale = (ae_target * full_well_e) / p99
    else:
        e_scale = full_well_e
    return float(e_scale)


def apply_sensor_noise(
    linear_radiance: np.ndarray,
    prnu_map: np.ndarray,
    dsnu_map: np.ndarray,
    E: float,
    full_well_e: float,
    sigma_r_e: float,
    dark_current_e_per_s: float,
    exposure_time_s: float,
    k_dn_per_e: float,
    bits: int,
    black_level_dn: float,
    rng_temporal: np.random.Generator,
) -> np.ndarray:
    """Simulate camera sensor detection, noise accumulation, saturation and ADC quantization.

    Parameters
    ----------
    linear_radiance : np.ndarray
        Linear relative radiance in [0, 1].
    prnu_map : np.ndarray
        PRNU map (relative gain).
    dsnu_map : np.ndarray
        DSNU map (electrons).
    E : float
        Number of photo-electrons at L = 1.
    full_well_e : float
        Saturation full well capacity in electrons.
    sigma_r_e : float
        Temporal read noise RMS in electrons.
    dark_current_e_per_s : float
        Dark current D in electrons/second.
    exposure_time_s : float
        Exposure time t_exp in seconds.
    k_dn_per_e : float
        Overall system gain K in DN/e-.
    bits : int
        ADC resolution in bits (e.g. 12).
    black_level_dn : float
        Black level offset in DN.
    rng_temporal : np.random.Generator
        PCG64 generator initialized with temporal seed.

    Returns
    -------
    dn : np.ndarray, dtype uint16
        Output digital numbers in [0, 2^bits - 1].
    """
    rad = np.asarray(linear_radiance, dtype=np.float64)

    # 1. Mean photo-electrons with PRNU gain modulation
    mu_e = rad * E
    mean_photo = np.maximum(0.0, mu_e * (1.0 + prnu_map))
    e_photo = rng_temporal.poisson(mean_photo).astype(np.float64)

    # 2. Dark signal with DSNU spatial offset
    mean_dark = dark_current_e_per_s * exposure_time_s
    e_dark = rng_temporal.poisson(mean_dark, size=rad.shape).astype(np.float64) + dsnu_map

    # 3. Temporal read noise
    e_read = rng_temporal.normal(0.0, sigma_r_e, rad.shape)

    # 4. Total electron accumulation
    e_total = e_photo + e_dark + e_read

    # 5. Saturation at full well FW (electrons)
    e_sat = np.clip(e_total, None, full_well_e)

    # 6. ADC quantization
    max_dn = (1 << bits) - 1
    dn_continuous = k_dn_per_e * e_sat + black_level_dn
    dn = np.clip(np.round(dn_continuous), 0, max_dn).astype(np.uint16)

    return dn
