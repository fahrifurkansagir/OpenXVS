"""Tests for EMVA 1288 simulation and self-consistency analysis (V5).

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V5.
- Photon transfer series: >= 40 exposure levels, >= 10 in [0.80, 1.10].
- Tolerances:
  - K: +-1 %
  - sigma_r: +-3 %
  - saturation capacity: +-3 %
  - DSNU: +-10 %
  - PRNU: +-10 %
  - linearity error: < 0.5 %
"""

from __future__ import annotations

import numpy as np
import pytest

from xvs.sensor.noise import generate_fpn_maps, apply_sensor_noise
from xvs.sensor.emva1288 import analyze_emva1288


def test_v5_emva1288_self_consistency():
    """V5 requirement: EMVA 1288 parameter estimates match configuration within tolerances."""
    H, W = 128, 128
    FW = 10500.0
    sigma_r = 2.4
    D = 5.0
    t_exp = 0.001
    prnu_rms = 0.005
    dsnu_rms = 1.0
    K = 0.35
    bits = 12
    black_level = 64.0

    sensor_id = "test_sensor_v5"
    prnu_map, dsnu_map = generate_fpn_maps(sensor_id, H, W, prnu_rms, dsnu_rms)

    # 45 exposure levels from dark (0.0) to 110% of saturation
    # with 15 levels in [0.80, 1.10]
    levels_low = np.linspace(0.0, 0.78, 30)
    levels_high = np.linspace(0.80, 1.10, 15)
    levels = np.unique(np.concatenate([levels_low, levels_high]))
    assert len(levels) >= 40
    assert np.sum((levels >= 0.80) & (levels <= 1.10)) >= 10

    rng_temp = np.random.Generator(np.random.PCG64(42))
    frame_pairs = []
    for L_val in levels:
        flat = np.full((H, W), L_val, dtype=np.float64)
        f_a = apply_sensor_noise(flat, prnu_map, dsnu_map, FW, FW, sigma_r, D, t_exp, K, bits, black_level, rng_temp)
        f_b = apply_sensor_noise(flat, prnu_map, dsnu_map, FW, FW, sigma_r, D, t_exp, K, bits, black_level, rng_temp)
        frame_pairs.append((f_a, f_b))

    # 16 dark frames and 16 bright frames (at 50% saturation) for FPN analysis
    dark_frames = [
        apply_sensor_noise(np.zeros((H, W)), prnu_map, dsnu_map, FW, FW, sigma_r, D, t_exp, K, bits, black_level, rng_temp)
        for _ in range(16)
    ]
    bright_frames = [
        apply_sensor_noise(np.full((H, W), 0.5), prnu_map, dsnu_map, FW, FW, sigma_r, D, t_exp, K, bits, black_level, rng_temp)
        for _ in range(16)
    ]

    results = analyze_emva1288(levels, frame_pairs, dark_frames, bright_frames, bits=bits)

    # Tolerances verification
    k_err_pct = abs(results["k_est"] - K) / K * 100.0
    sigma_r_err_pct = abs(results["sigma_r_est"] - sigma_r) / sigma_r * 100.0
    sat_cap_err_pct = abs(results["sat_capacity_est"] - FW) / FW * 100.0
    dsnu_err_pct = abs(results["dsnu_est"] - dsnu_rms) / dsnu_rms * 100.0
    prnu_err_pct = abs(results["prnu_est"] - prnu_rms) / prnu_rms * 100.0
    lin_err_pct = results["linearity_error"] * 100.0

    assert k_err_pct <= 1.0, f"K error {k_err_pct:.3f}% exceeds 1.0%"
    assert sigma_r_err_pct <= 3.0, f"sigma_r error {sigma_r_err_pct:.3f}% exceeds 3.0%"
    assert sat_cap_err_pct <= 3.0, f"sat capacity error {sat_cap_err_pct:.3f}% exceeds 3.0%"
    assert dsnu_err_pct <= 10.0, f"DSNU error {dsnu_err_pct:.3f}% exceeds 10.0%"
    assert prnu_err_pct <= 10.0, f"PRNU error {prnu_err_pct:.3f}% exceeds 10.0%"
    assert lin_err_pct < 0.5, f"Linearity error {lin_err_pct:.4f}% exceeds 0.5%"
