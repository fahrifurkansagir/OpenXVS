"""EMVA 1288 (Release 4) standard sensor characterization and self-consistency analysis.

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V5.
- Temporal variance from frame differences.
- Overall system gain K from mean-variance slope over linear range.
- Read noise sigma_r from dark temporal variance minus 1/12 DN^2 quantization term.
- Saturation capacity from the variance maximum.
- DSNU and PRNU from spatial variance of averaged frames corrected for residual temporal noise.
- Linearity error over the linear range.
"""

from __future__ import annotations

import numpy as np


def analyze_emva1288(
    exposure_levels: np.ndarray,
    frame_pairs_dn: list[tuple[np.ndarray, np.ndarray]],
    dark_frames_dn: list[np.ndarray] | None = None,
    bright_frames_dn: list[np.ndarray] | None = None,
    bits: int = 12,
) -> dict[str, float | list[float]]:
    """Perform EMVA 1288 Release 4 analysis on a photon transfer series.

    Parameters
    ----------
    exposure_levels : np.ndarray
        Array of relative exposure levels (e.g. L from 0 to 1.10).
    frame_pairs_dn : list of tuple(ndarray, ndarray)
        Two frames (A, B) per exposure level in DN (digital numbers).
    dark_frames_dn : list of ndarray, optional
        16 dark frames (at zero exposure) for spatial DSNU analysis.
    bright_frames_dn : list of ndarray, optional
        16 bright frames (at ~50% saturation) for spatial PRNU analysis.
    bits : int
        ADC bit depth.

    Returns
    -------
    dict
        Dictionary containing estimated K, sigma_r, saturation capacity,
        DSNU, PRNU, linearity error, and curves.
    """
    num_levels = len(exposure_levels)
    means_y = np.zeros(num_levels, dtype=np.float64)
    vars_y_temp = np.zeros(num_levels, dtype=np.float64)

    for i, (f_a, f_b) in enumerate(frame_pairs_dn):
        a_f = np.asarray(f_a, dtype=np.float64)
        b_f = np.asarray(f_b, dtype=np.float64)

        # Mean digital signal
        means_y[i] = 0.5 * (np.mean(a_f) + np.mean(b_f))
        # Temporal variance from frame difference (FPN cancels completely)
        diff = a_f - b_f
        vars_y_temp[i] = 0.5 * np.var(diff, ddof=1)

    mu_y_dark = means_y[0]

    # 1. Linear range selection: 5% to 70% of saturation signal range
    sat_range = np.max(means_y) - mu_y_dark
    lin_mask = (means_y >= mu_y_dark + 0.05 * sat_range) & (means_y <= mu_y_dark + 0.70 * sat_range)

    # 2. Overall system gain K (DN / e-)
    # In EMVA 1288: var_y_temp = K * (mu_y - mu_y_dark) + var_dark
    signal_above_dark = means_y[lin_mask] - mu_y_dark
    var_linear = vars_y_temp[lin_mask]

    slope_k, intercept = np.polyfit(signal_above_dark, var_linear, 1)
    k_est = float(slope_k)

    # 3. Read noise sigma_r (e-) from dark temporal variance
    # EMVA 1288: sigma_y.dark^2 = K^2 * sigma_r^2 + sigma_quant^2, where sigma_quant^2 = 1/12 DN^2
    var_dark = vars_y_temp[0]
    var_dark_no_quant = max(0.0, var_dark - (1.0 / 12.0))
    sigma_r_est = float(np.sqrt(var_dark_no_quant) / k_est)

    # 4. Saturation capacity (e-) from the variance maximum
    # The maximum of the temporal variance curve marks the onset of saturation
    max_var_idx = int(np.argmax(vars_y_temp))
    sat_capacity_est = float((means_y[max_var_idx] - mu_y_dark) / k_est)

    # 5. Linearity error over the linear range
    # Linear fit of mu_y vs exposure level
    exp_lin = exposure_levels[lin_mask]
    means_lin = means_y[lin_mask]
    p_lin = np.polyfit(exp_lin, means_lin, 1)
    means_predicted = np.polyval(p_lin, exp_lin)
    signal_span = np.max(means_lin) - np.min(means_lin)
    linearity_error = float(np.max(np.abs(means_lin - means_predicted)) / signal_span)

    # 6. Fixed Pattern Noise: DSNU and PRNU
    dsnu_est = None
    prnu_est = None

    if dark_frames_dn is not None and len(dark_frames_dn) >= 2:
        m_dark = len(dark_frames_dn)
        dark_stack = np.stack([f.astype(np.float64) for f in dark_frames_dn], axis=0)
        mean_dark_img = np.mean(dark_stack, axis=0)
        s_spatial_dark2 = float(np.var(mean_dark_img, ddof=1))
        # Correct for residual temporal variance in averaged image: s_y.dark^2 / M
        s_temp_dark2 = 0.5 * float(np.var(dark_stack[0] - dark_stack[1], ddof=1))
        s_dsnu_y2 = max(0.0, s_spatial_dark2 - s_temp_dark2 / m_dark)
        dsnu_est = float(np.sqrt(s_dsnu_y2) / k_est)

    if bright_frames_dn is not None and len(bright_frames_dn) >= 2 and dsnu_est is not None:
        m_bright = len(bright_frames_dn)
        bright_stack = np.stack([f.astype(np.float64) for f in bright_frames_dn], axis=0)
        mean_bright_img = np.mean(bright_stack, axis=0)
        s_spatial_bright2 = float(np.var(mean_bright_img, ddof=1))
        s_temp_bright2 = 0.5 * float(np.var(bright_stack[0] - bright_stack[1], ddof=1))
        s_fpn_bright2 = max(0.0, s_spatial_bright2 - s_temp_bright2 / m_bright)
        s_prnu_y2 = max(0.0, s_fpn_bright2 - s_dsnu_y2)
        mean_signal_delta = float(np.mean(mean_bright_img) - np.mean(dark_stack))
        prnu_est = float(np.sqrt(s_prnu_y2) / mean_signal_delta)

    return {
        "k_est": k_est,
        "sigma_r_est": sigma_r_est,
        "sat_capacity_est": sat_capacity_est,
        "max_var_exposure": float(exposure_levels[max_var_idx]),
        "linearity_error": linearity_error,
        "dsnu_est": dsnu_est,
        "prnu_est": prnu_est,
        "means_y": means_y.tolist(),
        "vars_y_temp": vars_y_temp.tolist(),
        "mu_y_dark": float(mu_y_dark),
    }
