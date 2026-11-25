"""Parametric curve fitting for detection probability and range (WP-04 §3, §5, §6).

Implements binomial maximum likelihood estimation (MLE) via Newton-Raphson (IRLS)
for logistic detection functions P_d(s_px) and detection range curves P_d(R).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import numpy as np

from xvs.metrics.aot import clopper_pearson_interval


@dataclass
class LogisticFitResult:
    """Outcome of binomial MLE logistic regression."""
    s50: float
    w: float
    se_s50: float
    ci95_low: float
    ci95_high: float
    converged: bool
    iterations: int


@dataclass
class RangeFitResult:
    """Outcome of detection range curve fitting P_d(R) = 1 / (1 + exp((ln R - ln R50) / b))."""
    r50_m: float
    r90_m: float
    b: float
    r50_ci95: tuple[float, float]
    r90_ci95: tuple[float, float]
    converged: bool


def fit_logistic_mle(
    s_arr: np.ndarray,
    y_arr: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> LogisticFitResult:
    """Fit P_d(s) = 1 / (1 + exp(-(s - s50) / w)) via binomial maximum likelihood.

    Uses Newton-Raphson iteratively reweighted least squares (IRLS).
    Computes exact Fisher information matrix and standard error of s50 via delta method.

    Args:
        s_arr: 1D array of target sizes in pixels
        y_arr: 1D array of binary detection labels (0 or 1)
        max_iter: maximum Newton-Raphson iterations
        tol: parameter convergence tolerance

    Returns:
        LogisticFitResult with s50, w, se_s50, and 95% Wald CI
    """
    s = np.asarray(s_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    n = len(s)
    if n == 0 or np.sum(y) == 0 or np.sum(y) == n:
        # Degenerate cases
        s_mid = float(np.median(s)) if n > 0 else 3.0
        return LogisticFitResult(
            s50=s_mid,
            w=0.7,
            se_s50=float("inf"),
            ci95_low=0.0,
            ci95_high=float("inf"),
            converged=False,
            iterations=0,
        )

    X = np.column_stack([np.ones(n, dtype=np.float64), s])

    # Initial guess: s50 around mean, w ~ 1.0 -> beta1 ~ 1.0, beta0 ~ -s_mean
    s_mean = float(np.mean(s))
    beta = np.array([-s_mean, 1.0], dtype=np.float64)

    converged = False
    H = np.eye(2, dtype=np.float64)

    for it in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        grad = X.T @ (p - y)
        w_diag = np.maximum(p * (1.0 - p), 1e-8)
        H = X.T @ (X * w_diag[:, None])

        try:
            delta = np.linalg.solve(H + 1e-8 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break

        max_d = np.max(np.abs(delta))
        step = min(1.0, 2.0 / max(max_d, 1.0))
        beta -= step * delta
        if max_d < tol:
            converged = True
            break

    b0, b1 = float(beta[0]), float(beta[1])
    b1_safe = b1 if abs(b1) > 1e-12 else (1e-12 if b1 >= 0 else -1e-12)

    s50 = float(-b0 / b1_safe)
    w_param = float(1.0 / b1_safe)

    try:
        cov = np.linalg.inv(H + 1e-9 * np.eye(2))
        grad_s50 = np.array([-1.0 / b1_safe, b0 / (b1_safe ** 2)])
        var_s50 = float(grad_s50.T @ cov @ grad_s50)
        se_s50 = float(np.sqrt(max(1e-12, var_s50)))
    except np.linalg.LinAlgError:
        se_s50 = float("inf")

    ci95_low = float(s50 - 1.96 * se_s50)
    ci95_high = float(s50 + 1.96 * se_s50)

    return LogisticFitResult(
        s50=s50,
        w=w_param,
        se_s50=se_s50,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        converged=converged,
        iterations=it + 1,
    )


def fit_detection_range_mle(
    range_m_arr: np.ndarray,
    y_arr: np.ndarray,
    max_iter: int = 25,
) -> RangeFitResult:
    """Fit detection range curve P_d(R) = 1 / (1 + exp((ln R - ln R50) / b)).

    Log-linear transform: let x = ln(R). Then (x - ln R50)/b = beta1 * x + beta0
    where beta1 = 1/b and beta0 = -ln(R50)/b.
    """
    ranges = np.asarray(range_m_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)

    valid = (ranges > 0.0)
    ranges = ranges[valid]
    y = y[valid]

    if len(ranges) == 0:
        return RangeFitResult(
            r50_m=1000.0,
            r90_m=500.0,
            b=0.5,
            r50_ci95=(0.0, float("inf")),
            r90_ci95=(0.0, float("inf")),
            converged=False,
        )

    log_r = np.log(ranges)
    res = fit_logistic_mle(log_r, 1.0 - y, max_iter=max_iter)
    # Notice: as range increases, probability decreases, so P_d(R) has negative slope.
    # 1 - P_d(R) = 1 / (1 + exp(-(ln R - ln R50)/b)) has positive slope in ln(R).

    ln_r50 = res.s50
    b = max(1e-4, res.w)
    r50 = float(np.exp(ln_r50))

    # P_d(R90) = 0.90 -> exp((ln R90 - ln R50)/b) = (1 - 0.90)/0.90 = 1/9
    # ln R90 = ln R50 + b * ln(1/9) = ln R50 - b * ln(9)
    ln_r90 = ln_r50 - b * math.log(9.0)
    r90 = float(np.exp(ln_r90))

    se = res.se_s50
    if not math.isinf(se):
        r50_low = float(np.exp(ln_r50 - 1.96 * se))
        r50_high = float(np.exp(ln_r50 + 1.96 * se))
        r90_low = float(np.exp(ln_r90 - 1.96 * se))
        r90_high = float(np.exp(ln_r90 + 1.96 * se))
    else:
        r50_low, r50_high = 0.0, float("inf")
        r90_low, r90_high = 0.0, float("inf")

    return RangeFitResult(
        r50_m=r50,
        r90_m=r90,
        b=b,
        r50_ci95=(r50_low, r50_high),
        r90_ci95=(r90_low, r90_high),
        converged=res.converged,
    )


def fit_log_logistic_mle(
    s_arr: np.ndarray,
    y_arr: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> LogisticFitResult:
    """Fit P_d(s) = 1 / (1 + exp(-(ln(s) - ln(s50)) / w_ln)) via binomial maximum likelihood.

    Fits logistic regression on u = ln(s) using Newton-Raphson IRLS.
    Parameter s50 = exp(u50), and Wald CI bounds are exp(u_low), exp(u_high).
    Ensures P_d(s -> 0) -> 0, fixing linear fit divergence at sub-pixel sizes.
    """
    s = np.asarray(s_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    valid = s > 1e-6
    s_v = s[valid]
    y_v = y[valid]
    if len(s_v) == 0:
        return LogisticFitResult(
            s50=3.0,
            w=0.5,
            se_s50=float("inf"),
            ci95_low=0.0,
            ci95_high=float("inf"),
            converged=False,
            iterations=0,
        )
    u_v = np.log(s_v)
    fit_u = fit_logistic_mle(u_v, y_v, max_iter=max_iter, tol=tol)
    if not fit_u.converged or not math.isfinite(fit_u.s50) or fit_u.w <= 0.0:
        return LogisticFitResult(
            s50=float(np.exp(fit_u.s50)) if math.isfinite(fit_u.s50) else float("nan"),
            w=fit_u.w,
            se_s50=float("inf"),
            ci95_low=0.0,
            ci95_high=float("inf"),
            converged=False,
            iterations=fit_u.iterations,
        )

    u50 = fit_u.s50
    w_ln = fit_u.w
    s50 = float(np.exp(u50))
    se_s50 = float(s50 * fit_u.se_s50) if math.isfinite(fit_u.se_s50) else float("inf")
    ci_low = float(np.exp(fit_u.ci95_low)) if math.isfinite(fit_u.ci95_low) else 0.0
    ci_high = float(np.exp(fit_u.ci95_high)) if math.isfinite(fit_u.ci95_high) else float("inf")

    return LogisticFitResult(
        s50=s50,
        w=w_ln,
        se_s50=se_s50,
        ci95_low=ci_low,
        ci95_high=ci_high,
        converged=True,
        iterations=fit_u.iterations,
    )


def evaluate_closure_fit(
    spx_arr: np.ndarray,
    hits_arr: np.ndarray,
    empirical_pd_by_spx: dict[str, Any] | None = None,
    op_k: int = 3,
    op_tau: float = 20.0,
    run_label: str = "WP-04 G0",
    n_encounters: int | None = None,
    fit_mode: str = "linear_logistic",
) -> dict[str, Any]:
    """Fit WP-05 closure curve Pd(s_px) and enforce R1 convergence and CI width rule.

    Supports fit_mode='linear_logistic' (default) and fit_mode='log_logistic'.
    """
    if fit_mode == "log_logistic":
        closure_fit = fit_log_logistic_mle(spx_arr, hits_arr)
    else:
        closure_fit = fit_logistic_mle(spx_arr, hits_arr)

    ci_width = (
        closure_fit.ci95_high - closure_fit.ci95_low
        if (math.isfinite(closure_fit.ci95_high) and math.isfinite(closure_fit.ci95_low))
        else float("inf")
    )
    ci_factor = (
        closure_fit.ci95_high / closure_fit.ci95_low
        if (closure_fit.ci95_low > 0.0 and math.isfinite(closure_fit.ci95_high))
        else float("inf")
    )
    is_valid_fit = (
        closure_fit.converged
        and closure_fit.s50 > 0.0
        and closure_fit.w > 0.0
        and closure_fit.ci95_low > 0.0
        and math.isfinite(closure_fit.ci95_high)
        and ci_width <= 2.0
        and ci_factor < 3.0
    )

    n_instances = len(spx_arr)

    if is_valid_fit:
        return {
            "status": "FITTED",
            "fit_mode": fit_mode,
            "s50_px": closure_fit.s50,
            "w_px": closure_fit.w,
            "w_ln": closure_fit.w if fit_mode == "log_logistic" else None,
            "w": closure_fit.w,
            "ci95_s50": [closure_fit.ci95_low, closure_fit.ci95_high],
            "ci95_width_px": ci_width,
            "ci95_factor": ci_factor,
            "converged": True,
            "fallback_used": False,
            "n_instances": n_instances,
            "n_encounters": n_encounters,
            "op_k": op_k,
            "op_tau": op_tau,
            "provenance": f"WP-04 G0 D0 MLE {fit_mode} fit, {run_label}",
            "empirical_pd_by_spx": empirical_pd_by_spx or {},
        }
    else:
        return {
            "status": "NOT_FITTED",
            "fit_mode": fit_mode,
            "s50_px": None,
            "w_px": None,
            "w_ln": None,
            "w": None,
            "ci95_s50": None,
            "ci95_width_px": None,
            "ci95_factor": None,
            "converged": False,
            "fallback_used": False,
            "n_instances": n_instances,
            "n_encounters": n_encounters,
            "op_k": op_k,
            "op_tau": op_tau,
            "provenance": f"NOT_FITTED ({fit_mode} fit failed or 95% CI width > 2.0 px)",
            "empirical_pd_by_spx": empirical_pd_by_spx or {},
        }


def evaluate_range_fit(
    range_m_arr: np.ndarray,
    hits_arr: np.ndarray,
    cls: str = "",
    bg: str = "",
    n_encounters: int = 0,
    encounters_arr: list[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit detection range curve and enforce R4 identifiability and empirical Pd reporting.

    R4 (F-B9-3): Report R50/R90 only when the fit converged and the 95% Wald CI spans
    less than a factor of 3 (ci95_high / ci95_low < 3.0). Otherwise status is
    'NOT_IDENTIFIABLE' and R50/R90 are reported as 'NOT_IDENTIFIABLE'.
    Always computes empirical Pd by range bin with Clopper-Pearson 95% CIs.
    """
    ranges = np.asarray(range_m_arr, dtype=np.float64)
    hits = np.asarray(hits_arr, dtype=np.float64)
    n_inst = len(ranges)

    # 1. Empirical Pd by range bin
    range_bins_km = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, 10.0),
    ]
    emp_by_range = {}
    enc_np = np.array(encounters_arr) if encounters_arr is not None else None

    for r0, r1 in range_bins_km:
        lbl = f"[{r0:.1f}, {r1:.1f}) km"
        in_bin = (ranges >= r0 * 1000.0) & (ranges < r1 * 1000.0)
        n_b = int(np.sum(in_bin))
        if n_b > 0:
            k_b = int(np.sum(hits[in_bin]))
            ci_b = clopper_pearson_interval(k_b, n_b, 0.95)
            n_enc_b = len(set(enc_np[in_bin])) if enc_np is not None else 0
            emp_by_range[lbl] = {
                "n_instances": n_b,
                "n_encounters": n_enc_b,
                "n_detected": k_b,
                "raw_pd": float(k_b / n_b),
                "raw_pd_ci95": [float(ci_b[0]), float(ci_b[1])],
            }
        else:
            emp_by_range[lbl] = {
                "n_instances": 0,
                "n_encounters": 0,
                "n_detected": 0,
                "raw_pd": 0.0,
                "raw_pd_ci95": [0.0, 0.0],
            }

    in_ge10 = (ranges >= 10000.0)
    n_ge10 = int(np.sum(in_ge10))
    if n_ge10 > 0:
        k_ge10 = int(np.sum(hits[in_ge10]))
        ci_ge10 = clopper_pearson_interval(k_ge10, n_ge10, 0.95)
        n_enc_ge10 = len(set(enc_np[in_ge10])) if enc_np is not None else 0
        emp_by_range[">= 10.0 km"] = {
            "n_instances": n_ge10,
            "n_encounters": n_enc_ge10,
            "n_detected": k_ge10,
            "raw_pd": float(k_ge10 / n_ge10),
            "raw_pd_ci95": [float(ci_ge10[0]), float(ci_ge10[1])],
        }
    else:
        emp_by_range[">= 10.0 km"] = {
            "n_instances": 0,
            "n_encounters": 0,
            "n_detected": 0,
            "raw_pd": 0.0,
            "raw_pd_ci95": [0.0, 0.0],
        }

    # 2. Parametric MLE fit
    if n_inst >= 10:
        fit_res = fit_detection_range_mle(ranges, hits)
        ci_low = fit_res.r50_ci95[0]
        ci_high = fit_res.r50_ci95[1]
        ci_factor = (ci_high / ci_low) if (ci_low > 0.0 and math.isfinite(ci_high)) else float("inf")

        is_identifiable = (
            fit_res.converged
            and fit_res.r50_m > 0.0
            and fit_res.r90_m > 0.0
            and ci_low > 0.0
            and math.isfinite(ci_high)
            and ci_factor < 3.0
        )
    else:
        fit_res = None
        ci_factor = float("inf")
        is_identifiable = False

    if is_identifiable and fit_res is not None:
        return {
            "class": cls,
            "background": bg,
            "n_instances": n_inst,
            "n_encounters": n_encounters,
            "status": "IDENTIFIABLE",
            "converged": True,
            "r50_m": fit_res.r50_m,
            "r90_m": fit_res.r90_m,
            "b": fit_res.b,
            "r50_ci95": [fit_res.r50_ci95[0], fit_res.r50_ci95[1]],
            "r90_ci95": [fit_res.r90_ci95[0], fit_res.r90_ci95[1]],
            "r50_ci_factor": ci_factor,
            "empirical_pd_by_range": emp_by_range,
        }
    else:
        return {
            "class": cls,
            "background": bg,
            "n_instances": n_inst,
            "n_encounters": n_encounters,
            "status": "NOT_IDENTIFIABLE",
            "converged": fit_res.converged if fit_res is not None else False,
            "r50_m": "NOT_IDENTIFIABLE",
            "r90_m": "NOT_IDENTIFIABLE",
            "b": None,
            "r50_ci95": None,
            "r90_ci95": None,
            "r50_ci_factor": ci_factor if (ci_factor < float("inf")) else None,
            "empirical_pd_by_range": emp_by_range,
        }


