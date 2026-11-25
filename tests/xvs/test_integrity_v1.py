"""Gate G0 Acceptance V1: Conformal exactness.

Conforms strictly to docs/xvs/WP07_INTEGRITY.md §3 and §5 (V1):
- Use 1,000 seeded synthetic replicates: scores from a known continuous distribution, n_cal = 500, alpha = 0.1.
- Compute each replicate's conditional coverage exactly from the known CDF at q(alpha).
- The mean must lie within +-0.5 % of k/(n+1), with k = ceil((n+1)(1-alpha)).
- The distribution must match Beta(k, n+1-k): a KS test p > 0.01.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats
import pytest

from xvs.integrity.conformal import (
    compute_conformal_quantile,
    compute_conformal_quantile_index,
)


def test_v1_conformal_exactness():
    """Verify finite-sample conformal prediction exactness and Beta distribution property."""
    n_cal = 500
    alpha = 0.1
    replicates = 1000
    seed = 20260927

    k = compute_conformal_quantile_index(n_cal, alpha)
    assert k == math.ceil((n_cal + 1) * (1.0 - alpha))
    assert k == 451

    target_mean = k / (n_cal + 1)  # 451 / 501 ~= 0.9001996

    rng = np.random.default_rng(seed)
    cond_coverages = []

    # Known continuous distribution: Standard Exponential (scale=1.0)
    # CDF: F(x) = 1 - exp(-x) for x >= 0
    for _ in range(replicates):
        scores = rng.standard_exponential(size=n_cal)
        q = compute_conformal_quantile(scores, alpha)
        # Exact conditional coverage from known CDF
        cov = 1.0 - math.exp(-q)
        cond_coverages.append(cov)

    cond_coverages = np.array(cond_coverages, dtype=np.float64)
    empirical_mean = float(np.mean(cond_coverages))
    diff_from_target = abs(empirical_mean - target_mean)

    # 1. Mean must lie within +-0.5% (0.005) of k / (n + 1)
    assert diff_from_target <= 0.005, (
        f"Conformal mean coverage {empirical_mean:.6f} differs from target {target_mean:.6f} "
        f"by {diff_from_target:.6f} (> 0.005)"
    )

    # 2. Distribution must match Beta(k, n + 1 - k) with KS test p > 0.01
    a_param = k
    b_param = n_cal + 1 - k
    ks_res = stats.kstest(cond_coverages, "beta", args=(a_param, b_param))

    assert ks_res.pvalue > 0.01, (
        f"KS test p-value {ks_res.pvalue:.4f} <= 0.01 (statistic={ks_res.statistic:.4f})"
    )


def test_v1_known_distribution_invariance():
    """Verify conformal exactness holds across different continuous distributions (Normal)."""
    n_cal = 500
    alpha = 0.1
    k = compute_conformal_quantile_index(n_cal, alpha)
    target_mean = k / (n_cal + 1)

    rng = np.random.default_rng(42)
    # Standard Normal: CDF is stats.norm.cdf
    scores_reps = rng.standard_normal(size=(500, n_cal))
    q_vals = [compute_conformal_quantile(scores_reps[i], alpha) for i in range(500)]
    cov_vals = [float(stats.norm.cdf(q)) for q in q_vals]

    mean_cov = float(np.mean(cov_vals))
    assert abs(mean_cov - target_mean) <= 0.01
