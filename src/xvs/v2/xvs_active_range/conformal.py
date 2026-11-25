"""Distribution-free Split Conformal safe-range calibration (`CP-MSC`) for `XVS-ActiveRange`.

Implements DR-20 §3.3 & Spec §4.1.2:
- One-sided safety nonconformity score `E_i = max(0, (R_hat_i - R_true_i) / max(sigma_R_i, sigma_floor))`
  penalizing hazardous range overestimation (`R_hat > R_true`).
- Finite-sample conformal quantile `q_hat` frozen post-calibration.
- Online lower bound `R_safe_lower = clamp(R_hat - q_hat * max(sigma_R, sigma_floor), R_floor, R_hat)`
  guaranteeing `R_floor <= R_safe_lower <= R_hat` unconditionally.
"""

from __future__ import annotations

import math

import numpy as np

R_FLOOR_M: float = 30.0
R_CEIL_M: float = 6000.0
SIGMA_FLOOR_M: float = 1.0


def compute_conformal_lower_bound(
    r_hat: float,
    sigma_r: float,
    q_hat: float,
    *,
    r_floor_m: float = R_FLOOR_M,
    r_ceil_m: float = R_CEIL_M,
    sigma_floor_m: float = SIGMA_FLOOR_M,
) -> float:
    """Compute guaranteed-safe Split Conformal lower range bound `R_safe_lower` in `[R_floor, R_hat]`."""
    r_floor = float(r_floor_m) if math.isfinite(r_floor_m) and r_floor_m > 0.0 else R_FLOOR_M
    r_ceil = float(r_ceil_m) if math.isfinite(r_ceil_m) and r_ceil_m >= r_floor else R_CEIL_M
    sig_floor = float(sigma_floor_m) if math.isfinite(sigma_floor_m) and sigma_floor_m > 0.0 else SIGMA_FLOOR_M

    if not math.isfinite(r_hat):
        return r_floor
    r_clamped = min(r_ceil, max(r_floor, float(r_hat)))

    if not math.isfinite(sigma_r) or sigma_r < 0.0:
        sig_eff = sig_floor
    else:
        sig_eff = max(sig_floor, float(sigma_r))

    if not math.isfinite(q_hat) or q_hat < 0.0:
        q_eff = 0.0
    else:
        q_eff = float(q_hat)

    lower = r_clamped - q_eff * sig_eff
    if not math.isfinite(lower):
        return r_floor
    return min(r_clamped, max(r_floor, lower))


def fit_one_sided_conformal_quantile(
    r_hat: np.ndarray,
    r_true: np.ndarray,
    sigma_r: np.ndarray,
    alpha: float = 0.05,
    *,
    sigma_floor_m: float = SIGMA_FLOOR_M,
) -> float:
    """Fit frozen finite-sample one-sided Split Conformal quantile `q_hat` (`alpha = 0.05` -> 95% target)."""
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    rh = np.asarray(r_hat, dtype=np.float64).reshape(-1)
    rt = np.asarray(r_true, dtype=np.float64).reshape(-1)
    sr = np.asarray(sigma_r, dtype=np.float64).reshape(-1)

    if rh.size == 0 or rh.size != rt.size or rh.size != sr.size:
        raise ValueError(
            f"Expected non-empty equal-length arrays, got shapes {rh.shape}, {rt.shape}, {sr.shape}"
        )

    valid = np.isfinite(rh) & np.isfinite(rt) & np.isfinite(sr) & (rt > 0.0)
    if not np.any(valid):
        raise ValueError("No finite calibration samples with positive r_true.")

    rh_v = rh[valid]
    rt_v = rt[valid]
    sr_v = np.maximum(sr[valid], float(sigma_floor_m))

    scores = np.maximum(0.0, (rh_v - rt_v) / sr_v)
    scores_sorted = np.sort(scores)
    n_cal = int(scores_sorted.size)
    k_idx = min(n_cal, max(1, int(math.ceil((n_cal + 1) * (1.0 - float(alpha)))))) - 1
    return float(scores_sorted[k_idx])


def evaluate_conformal_coverage(
    r_hat: np.ndarray,
    r_true: np.ndarray,
    sigma_r: np.ndarray,
    q_hat: float,
    *,
    r_floor_m: float = R_FLOOR_M,
    sigma_floor_m: float = SIGMA_FLOOR_M,
) -> dict[str, float]:
    """Evaluate empirical safety coverage `P(R_true >= R_safe_lower)` and hazardous overestimation rate."""
    rh = np.asarray(r_hat, dtype=np.float64).reshape(-1)
    rt = np.asarray(r_true, dtype=np.float64).reshape(-1)
    sr = np.asarray(sigma_r, dtype=np.float64).reshape(-1)

    if rh.size == 0 or rh.size != rt.size or rh.size != sr.size:
        raise ValueError(
            f"Expected non-empty equal-length arrays, got shapes {rh.shape}, {rt.shape}, {sr.shape}"
        )

    valid = np.isfinite(rh) & np.isfinite(rt) & np.isfinite(sr) & (rt > 0.0)
    if not np.any(valid):
        raise ValueError("No finite evaluation samples with positive r_true.")

    rh_v = rh[valid]
    rt_v = rt[valid]
    sr_v = sr[valid]

    lower = np.array(
        [
            compute_conformal_lower_bound(
                float(rh_v[i]),
                float(sr_v[i]),
                float(q_hat),
                r_floor_m=r_floor_m,
                sigma_floor_m=sigma_floor_m,
            )
            for i in range(int(rh_v.size))
        ],
        dtype=np.float64,
    )
    covered = rt_v >= lower
    overestimated = lower > rt_v
    return {
        "n_samples": float(rh_v.size),
        "empirical_coverage": float(np.mean(covered)),
        "overestimation_rate": float(np.mean(overestimated)),
        "mean_r_safe_lower_m": float(np.mean(lower)),
        "mean_safety_margin_m": float(np.mean(rt_v - lower)),
        "median_abs_rel_err_pct": float(np.median(np.abs(rh_v - rt_v) / np.maximum(rt_v, 1.0)) * 100.0),
    }
