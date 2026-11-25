"""Conformal prediction and Protection Level (PL) computation for OpenXVS WP-07.

Implements standard split conformal prediction and Mondrian conformal prediction
partitioned by (cam_set x clean/degraded).
Conforms strictly to docs/xvs/WP07_INTEGRITY.md §3 and §5 (V1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence
import numpy as np


ALPHA_LEVELS = (0.1, 0.01, 0.001)


def compute_conformal_quantile_index(n: int, alpha: float) -> int:
    """Compute 1-based order statistic index k = ceil((n + 1) * (1 - alpha))."""
    if n <= 0:
        return 0
    return math.ceil((n + 1) * (1.0 - alpha))


def compute_conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Compute split conformal quantile q(alpha) for nonconformity scores.

    Uses finite-sample correction index k = ceil((n + 1) * (1 - alpha)).
    Per D28: If k > n, q(alpha) = +inf and PL = inf (unavailable). Never clip to sample max.

    Args:
        scores: Sequence of calibration nonconformity scores.
        alpha: Significance level (e.g. 0.1, 0.01, 0.001).

    Returns:
        Quantile value q(alpha), or +inf if k > n.
    """
    valid_scores = [float(s) for s in scores if not math.isnan(s) and not math.isinf(s)]
    n = len(valid_scores)
    if n == 0:
        return float("inf")

    k = compute_conformal_quantile_index(n, alpha)
    if k > n or k <= 0:
        return float("inf")
    sorted_scores = np.sort(valid_scores)
    # k is 1-indexed, so k - 1 in 0-indexed array
    return float(sorted_scores[k - 1])


def compute_sigma_hat(covariance_rad2: np.ndarray | Sequence[Sequence[float]] | None) -> float:
    """Compute sigma_hat in radians from 2x2 attitude covariance, floored at 1 urad (1e-6 rad)."""
    if covariance_rad2 is None:
        return 1e-6
    cov = np.asarray(covariance_rad2, dtype=np.float64)
    if cov.shape != (2, 2) or np.any(np.isnan(cov)):
        return 1e-6
    tr = float(np.trace(cov))
    if tr <= 0.0 or math.isnan(tr):
        return 1e-6
    return max(1e-6, math.sqrt(tr))


def compute_protection_level_deg(sigma_hat_deg: float, quantile: float) -> float:
    """Compute Protection Level PL(alpha) in degrees = q(alpha) * sigma_hat_deg.

    Per D28: If quantile is inf, PL is inf.
    """
    if math.isnan(quantile) or math.isnan(sigma_hat_deg):
        return float("nan")
    if math.isinf(quantile):
        return float("inf")
    return float(quantile * sigma_hat_deg)


@dataclass
class ConformalModel:
    """Conformal calibration model storing standard and Mondrian quantiles."""
    standard_quantiles: dict[str, float] = field(default_factory=dict)
    # Mondrian quantiles keyed by "cam_set:condition_group" (e.g. "FRBL:degraded") -> {alpha_str: q}
    mondrian_quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    n_cal_standard: int = 0
    n_cal_mondrian: dict[str, int] = field(default_factory=dict)

    # F-B12-5: Systematic error floor and second score quantiles: r = error / (sigma_hat + sigma_sys)
    sigma_sys_rad: dict[str, float] = field(default_factory=dict)
    standard_quantiles_sys: dict[str, float] = field(default_factory=dict)
    mondrian_quantiles_sys: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _clean_q(d: dict[str, float]) -> dict[str, float | None]:
            return {k: (None if (math.isinf(v) or math.isnan(v)) else float(v)) for k, v in d.items()}

        def _clean_mond(d: dict[str, dict[str, float]]) -> dict[str, dict[str, float | None]]:
            return {
                grp: {k: (None if (math.isinf(v) or math.isnan(v)) else float(v)) for k, v in q_dict.items()}
                for grp, q_dict in d.items()
            }

        return {
            "standard_quantiles": _clean_q(self.standard_quantiles),
            "mondrian_quantiles": _clean_mond(self.mondrian_quantiles),
            "n_cal_standard": self.n_cal_standard,
            "n_cal_mondrian": self.n_cal_mondrian,
            "sigma_sys_rad": self.sigma_sys_rad,
            "standard_quantiles_sys": _clean_q(self.standard_quantiles_sys),
            "mondrian_quantiles_sys": _clean_mond(self.mondrian_quantiles_sys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalModel:
        def _restore_q(d: dict[str, Any]) -> dict[str, float]:
            return {k: (float("inf") if v is None else float(v)) for k, v in d.items()}

        def _restore_mond(d: dict[str, Any]) -> dict[str, dict[str, float]]:
            return {
                grp: {k: (float("inf") if v is None else float(v)) for k, v in q_dict.items()}
                for grp, q_dict in d.items()
            }

        return cls(
            standard_quantiles=_restore_q(data.get("standard_quantiles", {})),
            mondrian_quantiles=_restore_mond(data.get("mondrian_quantiles", {})),
            n_cal_standard=data.get("n_cal_standard", 0),  # default-ok: optional fallback default
            n_cal_mondrian=data.get("n_cal_mondrian", {}),
            sigma_sys_rad=data.get("sigma_sys_rad", {}),
            standard_quantiles_sys=_restore_q(data.get("standard_quantiles_sys", {})),
            mondrian_quantiles_sys=_restore_mond(data.get("mondrian_quantiles_sys", {})),
        )

    def get_quantile(
        self,
        alpha: float,
        cam_set: str | None = None,
        condition: str = "degraded",
        mondrian: bool = False,
        is_clean: bool | None = None,
    ) -> float:
        """Retrieve the appropriate conformal quantile."""
        if is_clean is not None:
            condition = "clean" if is_clean else "degraded"
        alpha_key = f"{alpha:.4f}".rstrip("0")
        if mondrian:
            if cam_set is None:
                return float("nan")
            if cam_set == "horizontal":
                cam_set = "FRBL"
            group = "clean" if condition in ("clean", "nominal") else "degraded"
            key = f"{cam_set}:{group}"
            if key in self.mondrian_quantiles and alpha_key in self.mondrian_quantiles[key]:
                val = self.mondrian_quantiles[key][alpha_key]
                return float("inf") if val is None else float(val)
            # D28: Missing Mondrian slice makes frame unavailable; no pooled fallback!
            return float("nan")

        val = self.standard_quantiles.get(alpha_key, float("nan"))
        return float("inf") if val is None else float(val)

    def compute_pl_deg(
        self,
        sigma_hat_rad: float,
        alpha: float,
        cam_set: str | None = None,
        condition: str = "degraded",
        mondrian: bool = False,
        is_clean: bool | None = None,
    ) -> float:
        """Compute protection level in degrees for a given sigma_hat_rad and alpha."""
        q = self.get_quantile(alpha, cam_set=cam_set, condition=condition, mondrian=mondrian, is_clean=is_clean)
        sigma_hat_deg = math.degrees(sigma_hat_rad)
        return compute_protection_level_deg(sigma_hat_deg, q)

    def get_quantile_sys(
        self,
        alpha: float,
        cam_set: str | None = None,
        condition: str = "degraded",
        mondrian: bool = False,
    ) -> float:
        """Retrieve the second score (systematic error aware) conformal quantile."""
        alpha_key = f"{alpha:.4f}".rstrip("0")
        if mondrian:
            if cam_set is None:
                return float("nan")
            if cam_set == "horizontal":
                cam_set = "FRBL"
            group = "clean" if condition in ("clean", "nominal") else "degraded"
            key = f"{cam_set}:{group}"
            if key in self.mondrian_quantiles_sys and alpha_key in self.mondrian_quantiles_sys[key]:
                val = self.mondrian_quantiles_sys[key][alpha_key]
                return float("inf") if val is None else float(val)
            # D28: Missing Mondrian slice makes frame unavailable; no pooled fallback!
            return float("nan")
        val = self.standard_quantiles_sys.get(alpha_key, float("nan"))
        return float("inf") if val is None else float(val)

    def compute_pl_deg_sys(
        self,
        sigma_hat_rad: float,
        alpha: float,
        cam_set: str | None = None,
        condition: str = "degraded",
        mondrian: bool = False,
    ) -> float:
        """Compute protection level in degrees for second score r = error / (sigma_hat + sigma_sys)."""
        q_sys = self.get_quantile_sys(alpha, cam_set=cam_set, condition=condition, mondrian=mondrian)
        s_sys_rad = self.sigma_sys_rad.get(cam_set or "overall", self.sigma_sys_rad.get("overall", 0.0))  # default-ok: optional fallback default
        total_scale_deg = math.degrees(sigma_hat_rad + s_sys_rad)
        return compute_protection_level_deg(total_scale_deg, q_sys)


def fit_sigma_sys(
    alarm_free_cal_records: list[dict[str, Any]],
) -> dict[str, float]:
    """Fit systematic error scale sigma_sys (in radians) per camera set from calibration frames.

    Per D28: Calibrated on degraded frames only.
    sigma_sys = max(0.0, mean(error_rad) - mean(sigma_hat_rad)).
    """
    by_cam: dict[str, tuple[list[float], list[float]]] = {}
    all_err: list[float] = []
    all_sig: list[float] = []

    for r in alarm_free_cal_records:
        if r.get("fault_type", "none") not in ("none", None):
            continue
        err_rad = r.get("total_error_rad")
        if err_rad is None or math.isnan(err_rad):
            err_deg = r.get("total_error_deg")
            if err_deg is not None and not math.isnan(err_deg):
                err_rad = math.radians(err_deg)
            else:
                continue

        sig_rad = r.get("sigma_hat_urad")
        if sig_rad is not None and not math.isnan(sig_rad):
            sig_rad = sig_rad * 1e-6
        else:
            sig_rad = 1e-6

        cset = r.get("cam_set", "overall")
        if cset == "horizontal":
            cset = "FRBL"
        if cset not in by_cam:
            by_cam[cset] = ([], [])
        by_cam[cset][0].append(err_rad)
        by_cam[cset][1].append(sig_rad)
        all_err.append(err_rad)
        all_sig.append(sig_rad)

    out = {}
    if all_err:
        out["overall"] = max(0.0, float(np.mean(all_err) - np.mean(all_sig)))
    else:
        out["overall"] = 0.0

    for cset, (e_list, s_list) in by_cam.items():
        out[cset] = max(0.0, float(np.mean(e_list) - np.mean(s_list)))

    return out


def fit_conformal_model(
    alarm_free_cal_records: list[dict[str, Any]],
    alphas: Sequence[float] = ALPHA_LEVELS,
) -> ConformalModel:
    """Calibrate standard and Mondrian conformal models on alarm-free calibration records.

    Per D28:
    - Pooled ("standard") quantiles use ONLY calibration frames that are degraded,
      fault-free and alarm-free.
    - Clean frames enter ONLY their Mondrian slices.
    - Missing Mondrian slices have NO pooled fallback.
    - If k = ceil((n+1)(1-alpha)) > n, q(alpha) = +inf and PL = inf.

    Args:
        alarm_free_cal_records: Calibration split records where monitor alarm was False.
        alphas: Significance levels to calibrate (default: 0.1, 0.01, 0.001).

    Returns:
        Fitted ConformalModel instance.
    """
    scores_standard: list[float] = []
    scores_mondrian: dict[str, list[float]] = {}
    scores_standard_sys: list[float] = []
    scores_mondrian_sys: dict[str, list[float]] = {}

    sigma_sys_dict = fit_sigma_sys(alarm_free_cal_records)

    for r in alarm_free_cal_records:
        # F-B12-1: Calibrate only on nominal (fault_type == 'none') frames.
        fault_type = r.get("fault_type", "none")
        if fault_type not in ("none", None):
            continue

        score = r.get("nonconformity_score", float("nan"))
        if math.isnan(score) or math.isinf(score):
            continue

        # D28: Condition must be explicit; no fallback to mode.
        condition = r.get("condition")
        if condition is None:
            condition = "degraded"
        is_clean = (condition == "clean")
        is_degraded = not is_clean

        # D28: Pooled standard quantile uses ONLY degraded frames:
        if is_degraded:
            scores_standard.append(score)

        cam_set = r.get("cam_set", "unknown")
        if cam_set == "horizontal":
            cam_set = "FRBL"

        # Clean and degraded frames enter their Mondrian slices
        group = "clean" if is_clean else "degraded"
        key = f"{cam_set}:{group}"
        if key not in scores_mondrian:
            scores_mondrian[key] = []
        scores_mondrian[key].append(score)

        # F-B12-5: Second score r2 = error / (sigma_hat + sigma_sys)
        err_rad = r.get("total_error_rad")
        if err_rad is None or math.isnan(err_rad):
            err_deg = r.get("total_error_deg")
            err_rad = math.radians(err_deg) if (err_deg is not None and not math.isnan(err_deg)) else float("nan")

        sig_rad = r.get("sigma_hat_urad")
        sig_rad = sig_rad * 1e-6 if (sig_rad is not None and not math.isnan(sig_rad)) else 1e-6

        s_sys = sigma_sys_dict.get(cam_set, sigma_sys_dict.get("overall", 0.0))  # default-ok: optional fallback default
        if not math.isnan(err_rad) and (sig_rad + s_sys) > 0.0:
            score2 = err_rad / (sig_rad + s_sys)
            if is_degraded:
                scores_standard_sys.append(score2)
            if key not in scores_mondrian_sys:
                scores_mondrian_sys[key] = []
            scores_mondrian_sys[key].append(score2)

    model = ConformalModel()
    model.n_cal_standard = len(scores_standard)
    model.sigma_sys_rad = sigma_sys_dict

    for alpha in alphas:
        alpha_key = f"{alpha:.4f}".rstrip("0")
        model.standard_quantiles[alpha_key] = compute_conformal_quantile(scores_standard, alpha)
        if scores_standard_sys:
            model.standard_quantiles_sys[alpha_key] = compute_conformal_quantile(scores_standard_sys, alpha)

    for key, sc_list in scores_mondrian.items():
        model.mondrian_quantiles[key] = {}
        model.n_cal_mondrian[key] = len(sc_list)
        for alpha in alphas:
            alpha_key = f"{alpha:.4f}".rstrip("0")
            model.mondrian_quantiles[key][alpha_key] = compute_conformal_quantile(sc_list, alpha)

    for key, sc_list_sys in scores_mondrian_sys.items():
        model.mondrian_quantiles_sys[key] = {}
        for alpha in alphas:
            alpha_key = f"{alpha:.4f}".rstrip("0")
            model.mondrian_quantiles_sys[key][alpha_key] = compute_conformal_quantile(sc_list_sys, alpha)

    return model


def __getattr__(name: str) -> Any:
    if name in (
        "ConformalModelG1",
        "compute_pl_deg_altitude_mondrian",
        "fit_conformal_model_g1",
        "fit_sigma_sys_g1",
    ):
        from xvs.integrity import g1_extensions
        return getattr(g1_extensions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

