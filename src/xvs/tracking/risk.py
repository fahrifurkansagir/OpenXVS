"""Passive collision risk estimation and alerting rules (WP-05 §3 & Batch B7 Step 1).

Conforms strictly to docs/xvs/WP05_TRACKING.md §3, GEMINI_G-B3.md §1 item 4, and BATCH_B7_ARGON.md §1.
Estimates:
- Legacy G0: Time to contact tau_s = s / s_dot, range interval [L_min, L_max] / s, HMD_est = R * |omega_LOS| * tau_s,
  and alert rules A1 (collision course) and A2 (strict interval inclusion);
- G1 Passive Well-Clear Alert Rule WC1 (RTCA DO-365B / NASA DAIDALUS passive projection):
  * Class-conditional and class-prior quantile range interval [R_min, R_max] from second-moment scale theta = s_mom;
  * Passive miss-distance interval [HMD_min, HMD_max] from (R, omega_LOS, eta);
  * Passive modified-tau interval [tau_mod_min, tau_mod_max] relative to D_THR = 609.6 m;
  * Conservatism-interpolated point/interval estimates HMD_hat(beta_WC) and tau_mod_hat(beta_WC)
    for beta_WC in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xvs.tracking.tracker import Track

# Extreme bounding dimensions per WP-02 §2.1 (span and length min/max in meters)
WP02_CLASS_DIMENSIONS: dict[str, tuple[float, float]] = {
    "C1_UAS": (0.30, 1.20),
    "C2_SEP": (7.00, 12.00),
    "C3_TWIN": (10.00, 18.00),
    "C4_BIZJET": (14.00, 22.00),
    "C5_TRANSPORT": (28.00, 40.00),
    "C6_HELI": (10.00, 14.00),
}

# Union over all classes for blind detector in G0 / G1 (alpha_R = 0.0)
UNION_L_MIN = 0.30   # from C1_UAS
UNION_L_MAX = 40.00  # from C5_TRANSPORT

# ASTM F3442-23 / RTCA DO-365B LoWC horizontal boundary (2000 ft)
LOWC_HMD_LIMIT_M = 609.6


@dataclass
class RiskEstimate:
    """Estimated collision risk metrics and alert status for a track at time t."""
    track_id: str
    t_s: float
    s_rad: float
    s_dot_radps: float
    sigma_s_dot: float
    tau_s: float | None
    omega_los_radps: float
    range_interval_m: tuple[float, float]
    hmd_interval_m: tuple[float, float] | None
    alert_a1_warning: bool
    alert_a1_corrective: bool
    alert_a2_warning: bool
    alert_a2_corrective: bool
    # G1 Passive Well-Clear (WC1) fields
    wc1_warning: bool = False
    wc1_corrective: bool = False
    hmd_min_m: float | None = None
    hmd_max_m: float | None = None
    hmd_hat_m: float | None = None
    tau_mod_min_s: float | None = None
    tau_mod_max_s: float | None = None
    tau_mod_hat_s: float | None = None
    tau_s_hat_s: float | None = None
    eta_hat_inv_s: float | None = None
    eta_min_inv_s: float | None = None
    eta_max_inv_s: float | None = None
    beta_wc: float = 0.5


def _mixture_cdf(length_m: float) -> float:
    """Evaluate equal-weight 6-class uniform mixture CDF F_L(length_m) from sampler_v1.json."""
    total = 0.0
    for l_min, l_max in WP02_CLASS_DIMENSIONS.values():
        if length_m <= l_min:
            c_val = 0.0
        elif length_m >= l_max:
            c_val = 1.0
        else:
            c_val = (length_m - l_min) / (l_max - l_min)
        total += c_val
    return total / float(len(WP02_CLASS_DIMENSIONS))


def _mixture_quantile(q: float) -> float:
    """Invert 6-class mixture CDF for quantile q in [0, 1] via bisection."""
    if q <= 0.0:
        return UNION_L_MIN
    if q >= 1.0:
        return UNION_L_MAX
    lo = UNION_L_MIN
    hi = UNION_L_MAX
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _mixture_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_prior_size_bounds(
    class_id: str | None = None,
    alpha_r: float = 0.0,
) -> tuple[float, float]:
    """Return physical characteristic size bounds [L_min, L_max] in meters.

    - If class_id is known (in WP02_CLASS_DIMENSIONS), returns [L_{c,min}, L_{c,max}].
    - If class_id is None and alpha_r <= 0.0, returns strict union [0.30, 40.0] m.
    - If class_id is None and 0 < alpha_r < 0.5, returns prior-weighted quantile interval
      [L_{alpha_R}, L_{1 - alpha_R}] across the 6-class mixture.
    """
    if class_id is not None and class_id in WP02_CLASS_DIMENSIONS:
        return WP02_CLASS_DIMENSIONS[class_id]
    if alpha_r <= 0.0:
        return (UNION_L_MIN, UNION_L_MAX)
    q_low = max(0.0, min(0.49, float(alpha_r)))
    return (_mixture_quantile(q_low), _mixture_quantile(1.0 - q_low))


def compute_ttc_from_expansion(
    s_rad: float,
    s_dot_radps: float,
    sigma_s_dot: float,
    k_sigma: float = 3.0,
) -> float | None:
    """Compute legacy G0 time to contact from angular expansion rate (WP-05 §3).

    Valid while s_dot > k * sigma_s_dot (with k = 3).
    """
    if s_rad > 0.0 and s_dot_radps > (k_sigma * sigma_s_dot) and s_dot_radps > 1e-9:
        return s_rad / s_dot_radps
    return None


def compute_range_interval(
    s_rad: float,
    class_id: str | None = None,
    sigma_theta_rad: float = 0.0,
    k_theta: float = 1.0,
    theta_floor: float = 1e-6,
    alpha_r: float = 0.0,
) -> tuple[float, float]:
    """Compute range interval [R_min, R_max] from angular size and class size priors (BATCH_B7_ARGON.md §1.3).

    R_min = L_min / max(theta_floor, theta + k_theta * sigma_theta)
    R_max = L_max / max(theta_floor, theta - k_theta * sigma_theta)
    """
    if s_rad <= 1e-12:
        return (0.0, float("inf"))

    l_min, l_max = compute_prior_size_bounds(class_id=class_id, alpha_r=alpha_r)
    if sigma_theta_rad <= 0.0:
        return (l_min / s_rad, l_max / s_rad)

    theta_high = max(float(theta_floor), float(s_rad) + float(k_theta) * float(sigma_theta_rad))
    theta_low = max(float(theta_floor), float(s_rad) - float(k_theta) * float(sigma_theta_rad))
    return (l_min / theta_high, l_max / theta_low)


def compute_hmd_interval(
    range_interval_m: tuple[float, float],
    omega_los_radps: float,
    tau_s: float | None,
) -> tuple[float, float] | None:
    """Compute legacy G0 miss-distance proxy interval: HMD_est = R * |omega_LOS| * tau_s."""
    if tau_s is None or math.isinf(range_interval_m[1]):
        return None

    r_min, r_max = range_interval_m
    factor = omega_los_radps * tau_s
    return (r_min * factor, r_max * factor)


def compute_passive_hmd_interval(
    range_interval_m: tuple[float, float],
    omega_los_radps: float,
    sigma_omega_radps: float = 0.0,
    eta_min_inv_s: float = 1e-4,
    eta_max_inv_s: float = 1e-4,
    k_omega: float = 1.0,
    eta_floor: float = 1e-4,
) -> tuple[float, float] | None:
    """Compute G1 passive miss-distance interval [HMD_min, HMD_max] (BATCH_B7_ARGON.md §1.3 item 3).

    HMD(R, omega, eta) = R * omega / sqrt(eta^2 + omega^2)
    Monotonic bounds over R in [R_min, R_max], omega in [omega_min, omega_max], eta in [eta_min, eta_max]:
      HMD_min = R_min * omega_min / sqrt(eta_max^2 + omega_min^2)
      HMD_max = R_max * omega_max / sqrt(max(eta_floor, eta_min)^2 + omega_max^2)
    """
    r_min, r_max = range_interval_m
    if math.isinf(r_max) or r_max <= 0.0 or eta_max_inv_s <= 0.0:
        return None

    omega = max(0.0, float(omega_los_radps))
    sig_w = max(0.0, float(sigma_omega_radps))
    omega_min = max(0.0, omega - float(k_omega) * sig_w)
    omega_max = omega + float(k_omega) * sig_w

    eta_hi = max(float(eta_floor), float(eta_max_inv_s))
    eta_lo = max(float(eta_floor), float(eta_min_inv_s))

    denom_min = math.hypot(eta_hi, omega_min)
    denom_max = math.hypot(eta_lo, omega_max)

    hmd_min = (r_min * omega_min / denom_min) if denom_min > 1e-12 else 0.0
    hmd_max = (r_max * omega_max / denom_max) if denom_max > 1e-12 else float("inf")
    return (float(hmd_min), float(hmd_max))


def compute_passive_tau_mod_interval(
    range_interval_m: tuple[float, float],
    eta_min_inv_s: float,
    eta_max_inv_s: float,
    d_thr_m: float = LOWC_HMD_LIMIT_M,
    eta_floor: float = 1e-4,
) -> tuple[float, float] | None:
    """Compute G1 passive modified-tau interval [tau_mod_min, tau_mod_max] (BATCH_B7_ARGON.md §1.3 item 4).

    tau_mod(R, eta) = max(0, (1 / eta) * (1 - D_THR^2 / R^2))
    Monotonic bounds over R in [R_min, R_max] and eta in [eta_min, eta_max]:
      tau_mod_min = max(0, (1 / eta_max) * (1 - D_THR^2 / R_min^2))
      tau_mod_max = max(0, (1 / max(eta_floor, eta_min)) * (1 - D_THR^2 / R_max^2))
    """
    r_min, r_max = range_interval_m
    if math.isinf(r_max) or r_max <= 0.0 or eta_max_inv_s <= 0.0:
        return None

    eta_hi = max(float(eta_floor), float(eta_max_inv_s))
    eta_lo = max(float(eta_floor), float(eta_min_inv_s))
    d_sq = float(d_thr_m) ** 2

    if r_min <= float(d_thr_m):
        tau_mod_min = 0.0
    else:
        tau_mod_min = max(0.0, (1.0 / eta_hi) * (1.0 - d_sq / (r_min * r_min)))

    if r_max <= float(d_thr_m):
        tau_mod_max = 0.0
    else:
        tau_mod_max = max(0.0, (1.0 / eta_lo) * (1.0 - d_sq / (r_max * r_max)))

    return (float(tau_mod_min), float(tau_mod_max))


def evaluate_wc1_for_beta(
    hmd_interval_m: tuple[float, float] | None,
    tau_mod_interval_s: tuple[float, float] | None,
    beta_wc: float = 0.5,
    d_thr_m: float = LOWC_HMD_LIMIT_M,
    t_warning_s: float = 25.0,
    t_corrective_s: float = 55.0,
    closing_valid: bool = True,
) -> tuple[bool, bool, float | None, float | None]:
    """Evaluate WC1 warning and corrective alerts for a given conservatism weight beta_wc in [0, 1].

    Returns (wc1_warning, wc1_corrective, hmd_hat_m, tau_mod_hat_s).
    """
    if not closing_valid or hmd_interval_m is None or tau_mod_interval_s is None:
        return (False, False, None, None)

    beta = max(0.0, min(1.0, float(beta_wc)))
    hmd_min, hmd_max = hmd_interval_m
    tau_min, tau_max = tau_mod_interval_s

    hmd_hat = (1.0 - beta) * hmd_min + beta * hmd_max
    tau_mod_hat = (1.0 - beta) * tau_min + beta * tau_max

    hmd_ok = bool(hmd_hat <= float(d_thr_m))
    wc1_warn = bool(hmd_ok and (tau_mod_hat <= float(t_warning_s)))
    wc1_corr = bool(hmd_ok and (tau_mod_hat <= float(t_corrective_s)))
    return (wc1_warn, wc1_corr, float(hmd_hat), float(tau_mod_hat))


def evaluate_track_risk(
    track: Track,
    t_s: float,
    omega_thr_radps: float = 0.005,
    k_sigma: float = 3.0,
    hmd_limit_m: float = LOWC_HMD_LIMIT_M,
    class_id: str | None = None,
    t_warning_s: float = 25.0,
    t_corrective_s: float = 55.0,
    beta_wc: float = 0.5,
    k_theta: float = 1.0,
    k_omega: float = 1.0,
    alpha_r: float = 0.0,
    eta_floor: float = 1e-4,
) -> RiskEstimate:
    """Evaluate passive collision risk and alert rules (legacy A1/A2 and G1 WC1) for an active track."""
    s_rad = track.s
    s_dot = track.s_dot
    sigma_s_dot = track.sigma_s_dot
    omega_los = track.omega_los

    # Legacy G0 TTC and range/HMD intervals
    tau_s = compute_ttc_from_expansion(s_rad, s_dot, sigma_s_dot, k_sigma=k_sigma)
    r_interval_legacy = compute_range_interval(s_rad, class_id=class_id)
    hmd_interval_legacy = compute_hmd_interval(r_interval_legacy, omega_los, tau_s)

    # Alert A1 (collision course): tau_s < T and |omega_LOS| < omega_thr
    a1_warn = False
    a1_corr = False
    if tau_s is not None and omega_los < omega_thr_radps:
        if tau_s < t_warning_s:
            a1_warn = True
        if tau_s < t_corrective_s:
            a1_corr = True

    # Alert A2 (interval): A1 plus HMD_est,max < 609.6 m
    a2_warn = False
    a2_corr = False
    if hmd_interval_legacy is not None and hmd_interval_legacy[1] < hmd_limit_m:
        a2_warn = a1_warn
        a2_corr = a1_corr

    # G1 Passive Well-Clear (WC1) evaluation
    theta_rad = getattr(track, "s_mom", None)
    if theta_rad is None or theta_rad <= 0.0:
        theta_rad = s_rad
    sigma_theta = getattr(track, "sigma_s", 0.0)
    sigma_omega = getattr(track, "sigma_omega", 0.0)

    r_interval_g1 = compute_range_interval(
        theta_rad,
        class_id=class_id,
        sigma_theta_rad=sigma_theta,
        k_theta=k_theta,
        alpha_r=alpha_r,
    )

    ttc_est = getattr(track, "ttc_estimate", None)
    closing_valid = False
    tau_s_hat_s: float | None = None
    eta_hat: float | None = None
    eta_min: float | None = None
    eta_max: float | None = None

    if ttc_est is not None:
        closing_valid = bool(ttc_est.valid)
        tau_s_hat_s = ttc_est.tau_s_hat_s
        eta_hat = ttc_est.eta_hat_inv_s
        eta_min = ttc_est.eta_min_inv_s
        eta_max = ttc_est.eta_max_inv_s
    elif tau_s is not None and tau_s > 1e-6 and s_rad > 1e-12:
        closing_valid = True
        tau_s_hat_s = tau_s
        eta_hat = s_dot / s_rad
        eta_sigma = sigma_s_dot / s_rad
        eta_min = max(eta_floor, eta_hat - 1.5 * eta_sigma)
        eta_max = max(eta_min, eta_hat + 1.5 * eta_sigma)

    hmd_min_m: float | None = None
    hmd_max_m: float | None = None
    hmd_hat_m: float | None = None
    tau_mod_min_s: float | None = None
    tau_mod_max_s: float | None = None
    tau_mod_hat_s: float | None = None
    wc1_warn = False
    wc1_corr = False

    if closing_valid and eta_min is not None and eta_max is not None:
        passive_hmd = compute_passive_hmd_interval(
            r_interval_g1,
            omega_los_radps=omega_los,
            sigma_omega_radps=sigma_omega,
            eta_min_inv_s=eta_min,
            eta_max_inv_s=eta_max,
            k_omega=k_omega,
            eta_floor=eta_floor,
        )
        passive_tau_mod = compute_passive_tau_mod_interval(
            r_interval_g1,
            eta_min_inv_s=eta_min,
            eta_max_inv_s=eta_max,
            d_thr_m=hmd_limit_m,
            eta_floor=eta_floor,
        )
        if passive_hmd is not None and passive_tau_mod is not None:
            hmd_min_m, hmd_max_m = passive_hmd
            tau_mod_min_s, tau_mod_max_s = passive_tau_mod
            wc1_warn, wc1_corr, hmd_hat_m, tau_mod_hat_s = evaluate_wc1_for_beta(
                passive_hmd,
                passive_tau_mod,
                beta_wc=beta_wc,
                d_thr_m=hmd_limit_m,
                t_warning_s=t_warning_s,
                t_corrective_s=t_corrective_s,
                closing_valid=True,
            )

    return RiskEstimate(
        track_id=track.track_id,
        t_s=t_s,
        s_rad=s_rad,
        s_dot_radps=s_dot,
        sigma_s_dot=sigma_s_dot,
        tau_s=tau_s,
        omega_los_radps=omega_los,
        range_interval_m=r_interval_legacy,
        hmd_interval_m=hmd_interval_legacy,
        alert_a1_warning=a1_warn,
        alert_a1_corrective=a1_corr,
        alert_a2_warning=a2_warn,
        alert_a2_corrective=a2_corr,
        wc1_warning=wc1_warn,
        wc1_corrective=wc1_corr,
        hmd_min_m=hmd_min_m,
        hmd_max_m=hmd_max_m,
        hmd_hat_m=hmd_hat_m,
        tau_mod_min_s=tau_mod_min_s,
        tau_mod_max_s=tau_mod_max_s,
        tau_mod_hat_s=tau_mod_hat_s,
        tau_s_hat_s=tau_s_hat_s,
        eta_hat_inv_s=eta_hat,
        eta_min_inv_s=eta_min,
        eta_max_inv_s=eta_max,
        beta_wc=beta_wc,
    )
