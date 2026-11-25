"""Unit tests for WP-04 parametric fitting and Acceptance V4 binomial coverage.

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §6 and §7 (V4).
"""

import math
import numpy as np
import pytest

from xvs.detection import fit


def test_v4_binomial_mle_coverage():
    """Gate G0 Acceptance V4: Binomial MLE Wald 95% coverage across 200 seeded synthetic datasets.

    n = 5,000, s_px ~ U(0, 8), s50 = 3.0, w = 0.7.
    The Wald 95% interval for s50 must cover the truth in 181-197 of the 200 sets (the 99% binomial region).
    """
    n_sets = 200
    n_samples = 5000
    true_s50 = 3.0
    true_w = 0.7

    rng = np.random.RandomState(20260925)
    covered = 0

    for _ in range(n_sets):
        s = rng.uniform(0.0, 8.0, size=n_samples)
        # Logistic probability: P(detect) = 1 / (1 + exp(-(s - s50) / w))
        prob = 1.0 / (1.0 + np.exp(-(s - true_s50) / true_w))
        y = (rng.rand(n_samples) < prob).astype(np.float64)

        res = fit.fit_logistic_mle(s, y)
        assert res.converged is True

        if res.ci95_low <= true_s50 <= res.ci95_high:
            covered += 1

    print(f"\n[V4 Coverage] Covered: {covered} / {n_sets} (Specification allowed range: [181, 197])")
    assert 181 <= covered <= 197, f"Coverage {covered} outside [181, 197]"


def test_detection_range_curve_fit():
    """Verify detection range fitting P_d(R) = 1 / (1 + exp((ln R - ln R50) / b))."""
    rng = np.random.RandomState(42)
    n = 5000
    true_r50 = 2000.0
    true_b = 0.45

    ranges = rng.uniform(300.0, 6000.0, size=n)
    log_r = np.log(ranges)
    log_r50 = np.log(true_r50)

    # Probability decreases with range
    prob = 1.0 / (1.0 + np.exp((log_r - log_r50) / true_b))
    y = (rng.rand(n) < prob).astype(np.float64)

    res = fit.fit_detection_range_mle(ranges, y)
    assert res.converged is True

    # Check fitted parameters within 5%
    assert abs(res.r50_m - true_r50) / true_r50 < 0.05
    assert abs(res.b - true_b) / true_b < 0.05
    assert res.r90_m < res.r50_m
    print(f"\n[Range Fit] R50: true={true_r50:.1f} m, est={res.r50_m:.1f} m, R90={res.r90_m:.1f} m, b={res.b:.3f}")


def test_closure_fit_not_fitted_on_non_converged_or_wide_ci():
    """R1 / F-B9-1: Non-converged or wide CI fits must produce NOT_FITTED and no numbers."""
    # 1. Non-converged case (all zeros, no detections)
    s_deg = np.array([0.5, 1.2, 2.3, 3.4, 4.5, 6.0], dtype=np.float64)
    y_zero = np.zeros_like(s_deg)
    res_zero = fit.evaluate_closure_fit(s_deg, y_zero)
    assert res_zero["status"] == "NOT_FITTED"
    assert res_zero["s50_px"] is None
    assert res_zero["w_px"] is None
    assert res_zero["ci95_s50"] is None
    assert res_zero["converged"] is False
    assert res_zero["fallback_used"] is False

    # 2. Small sample with wide CI (> 2.0 px)
    s_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    y_noisy = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    res_wide = fit.evaluate_closure_fit(s_small, y_noisy)
    assert res_wide["status"] == "NOT_FITTED"
    assert res_wide["s50_px"] is None
    assert res_wide["w_px"] is None
    assert res_wide["ci95_s50"] is None

    # 3. Valid large dataset with converged fit and narrow CI (<= 2.0 px)
    rng = np.random.RandomState(42)
    s_valid = rng.uniform(0.5, 8.0, size=2000)
    prob_valid = 1.0 / (1.0 + np.exp(-(s_valid - 3.2) / 0.8))
    y_valid = (rng.rand(2000) < prob_valid).astype(np.float64)
    res_valid = fit.evaluate_closure_fit(s_valid, y_valid)
    assert res_valid["status"] == "FITTED"
    assert res_valid["s50_px"] is not None
    assert res_valid["w_px"] is not None
    assert res_valid["ci95_s50"] is not None
    assert res_valid["ci95_width_px"] <= 2.0
    assert abs(res_valid["s50_px"] - 3.2) < 0.2
    assert abs(res_valid["w_px"] - 0.8) < 0.2


def test_detection_range_identifiability():
    """R4 / F-B9-3: Range fits spanning >= 3x factor must report NOT_IDENTIFIABLE with empirical Pd."""
    # 1. Unidentifiable case (noisy, small sample -> wide CI > 3x factor)
    ranges_noisy = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 9500.0], dtype=np.float64)
    hits_noisy = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    res_unid = fit.evaluate_range_fit(ranges_noisy, hits_noisy, cls="c2", bg="sky", n_encounters=3)

    assert res_unid["status"] == "NOT_IDENTIFIABLE"
    assert res_unid["r50_m"] == "NOT_IDENTIFIABLE"
    assert res_unid["r90_m"] == "NOT_IDENTIFIABLE"
    assert res_unid["n_instances"] == 10
    assert res_unid["n_encounters"] == 3
    assert "empirical_pd_by_range" in res_unid
    assert len(res_unid["empirical_pd_by_range"]) > 0

    # 2. Identifiable case (clean logistic dataset with narrow CI < 3x factor)
    rng = np.random.RandomState(42)
    n = 2000
    true_r50 = 3000.0
    true_b = 0.5
    ranges_clean = rng.uniform(500.0, 8000.0, size=n)
    log_r = np.log(ranges_clean)
    log_r50 = np.log(true_r50)
    prob_clean = 1.0 / (1.0 + np.exp((log_r - log_r50) / true_b))
    hits_clean = (rng.rand(n) < prob_clean).astype(np.float64)

    res_id = fit.evaluate_range_fit(ranges_clean, hits_clean, cls="c1", bg="sky", n_encounters=30)
    assert res_id["status"] == "IDENTIFIABLE"
    assert isinstance(res_id["r50_m"], float)
    assert isinstance(res_id["r90_m"], float)
    assert res_id["r50_ci_factor"] < 3.0
    assert abs(res_id["r50_m"] - true_r50) / true_r50 < 0.1
    assert "empirical_pd_by_range" in res_id


def test_log_logistic_closure_fit():
    """Verify log-logistic fit on ln(s_px) converges and enforces Pd(0) -> 0."""
    rng = np.random.RandomState(42)
    s_valid = rng.uniform(0.5, 12.0, size=2000)
    # True s50 = 3.0, w_ln = 0.6
    u = np.log(s_valid)
    prob = 1.0 / (1.0 + np.exp(-(u - np.log(3.0)) / 0.6))
    hits = (rng.rand(2000) < prob).astype(np.float64)

    res = fit.evaluate_closure_fit(s_valid, hits, fit_mode="log_logistic")
    assert res["status"] == "FITTED"
    assert res["fit_mode"] == "log_logistic"
    assert abs(res["s50_px"] - 3.0) < 0.25
    assert abs(res["w_ln"] - 0.6) < 0.15
    assert res["ci95_width_px"] <= 2.0


