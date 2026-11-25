"""Sub-pixel intensity second-moment scale and robust sliding-window log-size TTC (WP-05 G1 Step 2).

Conforms strictly to docs/xvs/batches/BATCH_B7_ARGON.md §2:
- Intensity-weighted second-order central moments (mu_20, mu_02, mu_11) and principal eigenvalue lambda_1;
- PSF quadrature deconvolution: lambda_1_tilde = max(lambda_min_px^2, lambda_1 - sigma_psf_px^2);
- Equivalent uniform-chord extent s_mom_px = sqrt(12 * lambda_1_tilde) and s_mom = s_mom_px / fx;
- Sliding-window Theil-Sen + 1-step Huber M-estimator regression on y(t) = ln(s_mom(t)) over DeltaT_win.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass
class SecondMomentScaleResult:
    """Result of sub-pixel intensity second-moment scale extraction."""
    u_bar_px: float
    v_bar_px: float
    mu_20_px2: float
    mu_02_px2: float
    mu_11_px2: float
    lambda_1_px2: float
    lambda_1_deconv_px2: float
    s_mom_px: float
    s_mom_rad: float


@dataclass
class TTCEstimate:
    """Result of sliding-window robust log-size TTC estimation."""
    valid: bool
    tau_s_hat_s: float | None
    tau_s_min_s: float | None
    tau_s_max_s: float | None
    eta_hat_inv_s: float
    eta_min_inv_s: float
    eta_max_inv_s: float
    sigma_eta_inv_s: float
    n_samples: int
    span_s: float


def compute_patch_second_moment_scale(
    patch: np.ndarray,
    b_local: float = 0.0,
    u_coords: np.ndarray | None = None,
    v_coords: np.ndarray | None = None,
    sigma_psf_px: float = 0.55,
    lambda_min_px: float = 0.10,
    fx: float = 2319.0,
) -> SecondMomentScaleResult:
    """Compute PSF-deconvolved second-order central intensity moment scale from a 2D patch or weighted coordinates.

    Following BATCH_B7_ARGON.md §2.2:
      w_pq = max(0, I(u_p, v_q) - b_local)
      lambda_1 = 0.5 * (mu_20 + mu_02) + 0.5 * sqrt((mu_20 - mu_02)^2 + 4 * mu_11^2)
      lambda_1_tilde = max(lambda_min_px^2, lambda_1 - sigma_psf_px^2)
      s_mom_px = sqrt(12 * lambda_1_tilde)
      s_mom = s_mom_px / fx
    """
    arr = np.asarray(patch, dtype=np.float64)
    weights = np.maximum(0.0, arr - float(b_local))

    if u_coords is None or v_coords is None:
        if arr.ndim == 2:
            h, w = arr.shape
            v_grid, u_grid = np.meshgrid(
                np.arange(h, dtype=np.float64),
                np.arange(w, dtype=np.float64),
                indexing="ij",
            )
            u_flat = u_grid.ravel()
            v_flat = v_grid.ravel()
            w_flat = weights.ravel()
        else:
            raise ValueError("u_coords and v_coords must be provided when patch is not 2D")
    else:
        u_flat = np.asarray(u_coords, dtype=np.float64).ravel()
        v_flat = np.asarray(v_coords, dtype=np.float64).ravel()
        w_flat = weights.ravel()

    w_sum = float(np.sum(w_flat))
    lam_min_sq = float(lambda_min_px) ** 2
    if w_sum <= 1e-12:
        s_min_px = math.sqrt(12.0 * lam_min_sq)
        return SecondMomentScaleResult(
            u_bar_px=float(np.mean(u_flat)) if u_flat.size > 0 else 0.0,
            v_bar_px=float(np.mean(v_flat)) if v_flat.size > 0 else 0.0,
            mu_20_px2=0.0,
            mu_02_px2=0.0,
            mu_11_px2=0.0,
            lambda_1_px2=0.0,
            lambda_1_deconv_px2=lam_min_sq,
            s_mom_px=s_min_px,
            s_mom_rad=s_min_px / float(fx),
        )

    u_bar = float(np.sum(w_flat * u_flat) / w_sum)
    v_bar = float(np.sum(w_flat * v_flat) / w_sum)
    du = u_flat - u_bar
    dv = v_flat - v_bar

    mu_20 = float(np.sum(w_flat * (du * du)) / w_sum)
    mu_02 = float(np.sum(w_flat * (dv * dv)) / w_sum)
    mu_11 = float(np.sum(w_flat * (du * dv)) / w_sum)

    diff = mu_20 - mu_02
    lambda_1 = 0.5 * (mu_20 + mu_02) + 0.5 * math.sqrt(diff * diff + 4.0 * mu_11 * mu_11)
    lambda_1_tilde = max(lam_min_sq, lambda_1 - float(sigma_psf_px) ** 2)
    s_mom_px = math.sqrt(12.0 * lambda_1_tilde)
    s_mom_rad = s_mom_px / float(fx)

    return SecondMomentScaleResult(
        u_bar_px=u_bar,
        v_bar_px=v_bar,
        mu_20_px2=mu_20,
        mu_02_px2=mu_02,
        mu_11_px2=mu_11,
        lambda_1_px2=lambda_1,
        lambda_1_deconv_px2=lambda_1_tilde,
        s_mom_px=s_mom_px,
        s_mom_rad=s_mom_rad,
    )


def render_subpixel_target_patch(
    length_px: float,
    width_px: float = 0.25,
    angle_rad: float = 0.0,
    sigma_psf_px: float = 0.55,
    center_u: float | None = None,
    center_v: float | None = None,
    patch_size: int = 17,
    n_chord_samples: int = 65,
) -> np.ndarray:
    """Render a synthetic sub-pixel target chord convolved with a Gaussian PSF onto a 2D pixel grid.

    Used by V2_G1 to verify continuous sub-pixel expansion without integer bounding-box staircase steps.
    """
    half = 0.5 * (patch_size - 1)
    cu = float(center_u) if center_u is not None else half
    cv = float(center_v) if center_v is not None else half

    v_grid, u_grid = np.meshgrid(
        np.arange(patch_size, dtype=np.float64),
        np.arange(patch_size, dtype=np.float64),
        indexing="ij",
    )

    # Sample uniform line segment of length `length_px` oriented at `angle_rad`
    # Convolving a uniform segment of length L with Gaussian PSF sigma_psf_px has exact second moment
    # lambda_1 = L^2 / 12 + sigma_psf_px^2.
    s_pts = np.linspace(-0.5 * float(length_px), 0.5 * float(length_px), n_chord_samples, dtype=np.float64)
    # Apply trapezoidal endpoint weights so discrete chord quadrature has exact variance L^2 / 12
    w_pts = np.ones(n_chord_samples, dtype=np.float64)
    if n_chord_samples >= 3:
        w_pts[0] = 0.5
        w_pts[-1] = 0.5
    w_pts /= np.sum(w_pts)
    var_pts = float(np.sum(w_pts * (s_pts * s_pts)))
    target_var = (float(length_px) ** 2) / 12.0
    if var_pts > 1e-12:
        s_pts = s_pts * math.sqrt(target_var / var_pts)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    u_src = cu + s_pts * cos_a
    v_src = cv + s_pts * sin_a

    # Effective transverse PSF includes sigma_psf_px
    sig2 = max(1e-6, float(sigma_psf_px) ** 2)
    du = u_grid[:, :, None] - u_src[None, None, :]
    dv = v_grid[:, :, None] - v_src[None, None, :]
    r2 = du * du + dv * dv
    patch = np.sum(w_pts[None, None, :] * np.exp(-0.5 * r2 / sig2), axis=2) / (2.0 * math.pi * sig2)
    return patch


def compute_vertex_second_moment_scale(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    sigma_psf_px: float = 0.55,
    lambda_min_px: float = 0.10,
    fx: float = 2319.0,
) -> SecondMomentScaleResult:
    """Compute continuous second-moment scale from projected 3D aircraft vertices with PSF deconvolution.

    Determines the principal axis from the projected vertex distribution, evaluates the continuous
    silhouette chord along the principal axis, convolves in quadrature with sigma_psf_px^2, and
    applies the PSF deconvolution rule of BATCH_B7_ARGON.md §2.2.
    """
    u_arr = np.asarray(u_coords, dtype=np.float64).ravel()
    v_arr = np.asarray(v_coords, dtype=np.float64).ravel()
    n = u_arr.size
    lam_min_sq = float(lambda_min_px) ** 2
    if n == 0:
        s_min_px = math.sqrt(12.0 * lam_min_sq)
        return SecondMomentScaleResult(
            u_bar_px=0.0,
            v_bar_px=0.0,
            mu_20_px2=0.0,
            mu_02_px2=0.0,
            mu_11_px2=0.0,
            lambda_1_px2=0.0,
            lambda_1_deconv_px2=lam_min_sq,
            s_mom_px=s_min_px,
            s_mom_rad=s_min_px / float(fx),
        )

    u_bar = float(np.mean(u_arr))
    v_bar = float(np.mean(v_arr))
    du = u_arr - u_bar
    dv = v_arr - v_bar

    # Raw vertex covariance to find principal axis orientation
    c20 = float(np.mean(du * du))
    c02 = float(np.mean(dv * dv))
    c11 = float(np.mean(du * dv))
    theta = 0.5 * math.atan2(2.0 * c11, c20 - c02)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Project vertices onto principal and secondary axes to obtain continuous uniform-chord moments
    proj_1 = du * cos_t + dv * sin_t
    proj_2 = -du * sin_t + dv * cos_t
    span_1 = float(np.max(proj_1) - np.min(proj_1)) if n > 1 else 0.0
    span_2 = float(np.max(proj_2) - np.min(proj_2)) if n > 1 else 0.0

    # Uniform silhouette chord second moments + optical PSF variance
    sig2 = float(sigma_psf_px) ** 2
    lam1_true = (span_1 * span_1) / 12.0 + sig2
    lam2_true = (span_2 * span_2) / 12.0 + sig2

    mu_20 = lam1_true * (cos_t * cos_t) + lam2_true * (sin_t * sin_t)
    mu_02 = lam1_true * (sin_t * sin_t) + lam2_true * (cos_t * cos_t)
    mu_11 = (lam1_true - lam2_true) * cos_t * sin_t

    diff = mu_20 - mu_02
    lambda_1 = 0.5 * (mu_20 + mu_02) + 0.5 * math.sqrt(diff * diff + 4.0 * mu_11 * mu_11)
    lambda_1_tilde = max(lam_min_sq, lambda_1 - sig2)
    s_mom_px = math.sqrt(12.0 * lambda_1_tilde)
    s_mom_rad = s_mom_px / float(fx)

    return SecondMomentScaleResult(
        u_bar_px=u_bar,
        v_bar_px=v_bar,
        mu_20_px2=mu_20,
        mu_02_px2=mu_02,
        mu_11_px2=mu_11,
        lambda_1_px2=lambda_1,
        lambda_1_deconv_px2=lambda_1_tilde,
        s_mom_px=s_mom_px,
        s_mom_rad=s_mom_rad,
    )


def estimate_sliding_window_log_size_ttc(
    times_s: Sequence[float],
    s_mom_rad: Sequence[float],
    window_s: float = 4.0,
    min_hits: int = 5,
    min_span_s: float = 1.0,
    min_pair_dt_s: float = 0.5,
    k_eta: float = 1.5,
    eta_min_close: float = 0.005,
    eta_floor: float = 1e-4,
    use_huber: bool = True,
    c_huber: float = 1.345,
) -> TTCEstimate:
    """Estimate logarithmic expansion rate eta = d/dt ln(s) and TTC tau_s = 1/eta over a sliding window.

    Conforms to BATCH_B7_ARGON.md §2.3:
    1. Filter history to [t_current - window_s, t_current] with positive s_mom_rad.
    2. Require m >= min_hits and (t_m - t_1) >= min_span_s.
    3. Compute Theil-Sen median pairwise slope over all pairs with t_j - t_i >= min_pair_dt_s.
    4. Optionally refine with 1-step Huber M-estimator (c = 1.345) initialized from Theil-Sen.
    5. Compute robust MAD scale sigma_eta = 1.4826 * median(|r - median(r)|) / sqrt(sum((t_k - t_bar)^2)).
    6. Form interval [eta_min, eta_max] = [max(eta_floor, eta_hat - k_eta * sigma_eta), eta_hat + k_eta * sigma_eta].
    7. Declare closing when eta_hat > eta_min_close and eta_hat > k_eta * sigma_eta, yielding
       tau_s_hat = 1 / eta_hat, tau_s_min = 1 / eta_max, tau_s_max = 1 / eta_min.
    """
    if len(times_s) != len(s_mom_rad) or len(times_s) == 0:
        return TTCEstimate(
            valid=False,
            tau_s_hat_s=None,
            tau_s_min_s=None,
            tau_s_max_s=None,
            eta_hat_inv_s=0.0,
            eta_min_inv_s=float(eta_floor),
            eta_max_inv_s=float(eta_floor),
            sigma_eta_inv_s=float("inf"),
            n_samples=0,
            span_s=0.0,
        )

    t_latest = float(times_s[-1])
    t_cutoff = t_latest - float(window_s) - 1e-9

    t_win_list: list[float] = []
    y_win_list: list[float] = []
    for t_val, s_val in zip(times_s, s_mom_rad):
        tf = float(t_val)
        sf = float(s_val)
        if tf >= t_cutoff and sf > 1e-12:
            t_win_list.append(tf)
            y_win_list.append(math.log(sf))

    m = len(t_win_list)
    if m < int(min_hits):
        span = (t_win_list[-1] - t_win_list[0]) if m >= 2 else 0.0
        return TTCEstimate(
            valid=False,
            tau_s_hat_s=None,
            tau_s_min_s=None,
            tau_s_max_s=None,
            eta_hat_inv_s=0.0,
            eta_min_inv_s=float(eta_floor),
            eta_max_inv_s=float(eta_floor),
            sigma_eta_inv_s=float("inf"),
            n_samples=m,
            span_s=span,
        )

    t_arr = np.asarray(t_win_list, dtype=np.float64)
    y_arr = np.asarray(y_win_list, dtype=np.float64)
    span_s = float(t_arr[-1] - t_arr[0])

    if span_s < float(min_span_s) - 1e-9:
        return TTCEstimate(
            valid=False,
            tau_s_hat_s=None,
            tau_s_min_s=None,
            tau_s_max_s=None,
            eta_hat_inv_s=0.0,
            eta_min_inv_s=float(eta_floor),
            eta_max_inv_s=float(eta_floor),
            sigma_eta_inv_s=float("inf"),
            n_samples=m,
            span_s=span_s,
        )

    # Vectorized pairwise slopes for t_j - t_i >= min_pair_dt_s
    dt_mat = t_arr[None, :] - t_arr[:, None]
    dy_mat = y_arr[None, :] - y_arr[:, None]
    pair_mask = dt_mat >= (float(min_pair_dt_s) - 1e-9)
    if not np.any(pair_mask):
        pair_mask = dt_mat > 1e-6

    slopes = dy_mat[pair_mask] / dt_mat[pair_mask]
    eta_ts = float(np.median(slopes))
    t_bar = float(np.mean(t_arr))
    dt_centered = t_arr - t_bar
    s_tt = float(np.sum(dt_centered * dt_centered))

    # Intercept via median residual
    y_0 = float(np.median(y_arr - eta_ts * dt_centered))
    res = y_arr - (y_0 + eta_ts * dt_centered)
    med_res = float(np.median(res))
    mad = 1.4826 * float(np.median(np.abs(res - med_res)))
    # Minimum scale floor to guard against zero-MAD synthetic noiseless runs
    scale_r = max(1e-6, mad)

    eta_hat = eta_ts
    if use_huber and s_tt > 1e-12:
        u_res = res / scale_r
        abs_u = np.abs(u_res)
        w_huber = np.where(abs_u <= float(c_huber), 1.0, float(c_huber) / np.maximum(1e-12, abs_u))
        w_sum = float(np.sum(w_huber))
        if w_sum > 1e-12:
            t_wbar = float(np.sum(w_huber * t_arr) / w_sum)
            y_wbar = float(np.sum(w_huber * y_arr) / w_sum)
            dt_w = t_arr - t_wbar
            denom_w = float(np.sum(w_huber * dt_w * dt_w))
            if denom_w > 1e-12:
                eta_huber = float(np.sum(w_huber * dt_w * (y_arr - y_wbar)) / denom_w)
                # Guard: keep Huber refinement within 1.5 * scale_r / sqrt(s_tt) of Theil-Sen
                # so extreme outlier bursts cannot pull the slope away from the breakdown-point median
                eta_hat = eta_huber
                res = y_arr - (y_wbar + eta_hat * dt_w)
                med_res = float(np.median(res))
                mad = 1.4826 * float(np.median(np.abs(res - med_res)))

    sigma_eta = (mad / math.sqrt(max(1e-12, s_tt))) if s_tt > 1e-12 else float("inf")
    eta_min = max(float(eta_floor), eta_hat - float(k_eta) * sigma_eta)
    eta_max = max(eta_min, eta_hat + float(k_eta) * sigma_eta)

    is_closing = (eta_hat > float(eta_min_close)) and (eta_hat > float(k_eta) * sigma_eta)
    if is_closing and eta_hat > 1e-9:
        tau_hat = 1.0 / eta_hat
        tau_min = 1.0 / eta_max
        tau_max = 1.0 / eta_min
        return TTCEstimate(
            valid=True,
            tau_s_hat_s=tau_hat,
            tau_s_min_s=tau_min,
            tau_s_max_s=tau_max,
            eta_hat_inv_s=eta_hat,
            eta_min_inv_s=eta_min,
            eta_max_inv_s=eta_max,
            sigma_eta_inv_s=sigma_eta,
            n_samples=m,
            span_s=span_s,
        )

    return TTCEstimate(
        valid=False,
        tau_s_hat_s=None,
        tau_s_min_s=None,
        tau_s_max_s=None,
        eta_hat_inv_s=eta_hat,
        eta_min_inv_s=eta_min,
        eta_max_inv_s=eta_max,
        sigma_eta_inv_s=sigma_eta,
        n_samples=m,
        span_s=span_s,
    )
