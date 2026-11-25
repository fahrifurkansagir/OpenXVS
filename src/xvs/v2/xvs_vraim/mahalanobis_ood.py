"""Penultimate-layer tied-covariance Mahalanobis OOD detector with split-conformal calibration (DR-08).

Pure NumPy implementation (zero scikit-learn dependency):
  - `fit(features, labels, shrinkage_lambda=1e-2)` estimates per-class means and Tikhonov-shrunk
    tied covariance matrix `Sigma` + precision `Sigma^-1`.
  - `score(features)` computes min_c (z - mu_c)^T Sigma^-1 (z - mu_c), strictly **fail-closed**
    on `NaN` or `±Inf` rows (returning `+inf`).
  - `calibrate_conformal_threshold(cal_features, alpha=0.05)` computes exact finite-sample
    split-conformal OOD threshold `ceil((n + 1)(1 - alpha)) / n`.
  - `compute_auroc(id_scores, ood_scores)` computes exact Mann-Whitney / rank AUROC.
"""

from __future__ import annotations

import math

import numpy as np


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Exact rank-based AUROC (higher OOD score => positive class = OOD; pure NumPy)."""
    id_s = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_s = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_s.size == 0 or ood_s.size == 0:
        raise ValueError("id_scores and ood_scores must both be non-empty")

    # Replace +inf with a finite value strictly larger than all finite scores for exact ranking
    all_s = np.concatenate([id_s, ood_s])
    finite_vals = all_s[np.isfinite(all_s)]
    max_fin = float(np.max(finite_vals)) if finite_vals.size > 0 else 0.0
    id_clean = np.where(np.isfinite(id_s), id_s, max_fin + 1e6)
    ood_clean = np.where(np.isfinite(ood_s), ood_s, max_fin + 1e6)

    # Exact pairwise Mann-Whitney U statistic (vectorized for typical audit sizes)
    diff = ood_clean[:, None] - id_clean[None, :]
    gt = np.sum(diff > 0.0)
    eq = np.sum(diff == 0.0)
    return float((gt + 0.5 * eq) / float(ood_clean.size * id_clean.size))


class PenultimateMahalanobisOOD:
    """Tied-covariance Mahalanobis OOD scorer over penultimate network activations (DR-08)."""

    def __init__(self, shrinkage_lambda: float = 1e-2) -> None:
        if not (0.0 < shrinkage_lambda < 1.0):
            raise ValueError(f"shrinkage_lambda must be in (0, 1), got {shrinkage_lambda}")
        self.shrinkage_lambda = float(shrinkage_lambda)
        self.class_ids_: np.ndarray | None = None
        self.class_means_: np.ndarray | None = None  # [K, D]
        self.covariance_: np.ndarray | None = None  # [D, D]
        self.precision_: np.ndarray | None = None  # [D, D]
        self.conformal_threshold_: float | None = None
        self.feature_dim_: int = 0

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shrinkage_lambda: float | None = None,
    ) -> PenultimateMahalanobisOOD:
        """Fit class centroids and Tikhonov-shrunk tied covariance matrix."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError(f"Invalid features/labels shapes: {x.shape}, {y.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Training features must be strictly finite")

        lam = self.shrinkage_lambda if shrinkage_lambda is None else float(shrinkage_lambda)
        if not (0.0 < lam < 1.0):
            raise ValueError(f"shrinkage_lambda must be in (0, 1), got {lam}")
        self.shrinkage_lambda = lam

        classes = np.unique(y)
        n, d = x.shape
        means = np.zeros((classes.size, d), dtype=np.float64)
        residuals = np.zeros_like(x)
        for idx, cid in enumerate(classes):
            mask = y == cid
            mu = x[mask].mean(axis=0)
            means[idx] = mu
            residuals[mask] = x[mask] - mu

        cov_emp = (residuals.T @ residuals) / float(n)
        avg_var = max(float(np.trace(cov_emp)) / float(d), 1e-6)
        cov_shrunk = (1.0 - lam) * cov_emp + lam * avg_var * np.eye(d, dtype=np.float64)
        precision = np.linalg.inv(cov_shrunk)

        self.class_ids_ = classes
        self.class_means_ = means
        self.covariance_ = cov_shrunk
        self.precision_ = precision
        self.feature_dim_ = d
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return minimum squared Mahalanobis distance per row; non-finite rows fail-close to +inf."""
        if self.class_means_ is None or self.precision_ is None:
            raise RuntimeError("PenultimateMahalanobisOOD.score called before fit()")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Expected features shape (N, {self.feature_dim_}), got {x.shape}"
            )

        finite_row = np.all(np.isfinite(x), axis=1)
        scores = np.full((x.shape[0],), np.inf, dtype=np.float64)
        if not np.any(finite_row):
            return scores

        x_ok = x[finite_row]
        best = np.full((x_ok.shape[0],), np.inf, dtype=np.float64)
        for mu in self.class_means_:
            diff = x_ok - mu[None, :]
            d2 = np.einsum("ni,ij,nj->n", diff, self.precision_, diff)
            d2 = np.where(np.isfinite(d2) & (d2 >= 0.0), d2, np.inf)
            best = np.minimum(best, d2)
        scores[finite_row] = best
        return scores

    def calibrate_conformal_threshold(
        self,
        cal_features: np.ndarray,
        alpha: float = 0.05,
    ) -> float:
        """Compute exact finite-sample split-conformal OOD threshold `q_{1-alpha}`."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        cal_scores = self.score(cal_features)
        if not np.all(np.isfinite(cal_scores)):
            raise ValueError("Calibration features contain non-finite rows")
        n = cal_scores.size
        if n == 0:
            raise ValueError("Calibration set cannot be empty")
        level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))
        sorted_scores = np.sort(cal_scores)
        rank_idx = min(n - 1, max(0, int(math.ceil(level * n)) - 1))
        thr = float(sorted_scores[rank_idx])
        self.conformal_threshold_ = thr
        return thr

    def is_ood(self, features: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Fail-closed boolean OOD decision (`True` if score > threshold or non-finite)."""
        thr = self.conformal_threshold_ if threshold is None else float(threshold)
        if thr is None or not math.isfinite(thr):
            raise RuntimeError("Conformal threshold must be calibrated or explicitly provided")
        s = self.score(features)
        return (~np.isfinite(s)) | (s > thr)
