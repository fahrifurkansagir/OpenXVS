"""Hybrid Range-Conditioned Conformal-RAIM V-HPL/V-VPL, Mandatory Go-Around Latch & KSFO Anti-Misassociation Gate.

Implements SP2_M2_MODELS_DESIGN.md §3.4 & DR-03:
  - Cat I Alert Limits: VAL = 10.0 m, HAL = 40.0 m.
  - Hybrid Range-Conditioned Conformal-RAIM combines Vision-RAIM worst-case single-feature
    failure-mode slopes `g_max_h`, `g_max_v` with range-conditioned split-conformal margin.
  - Non-resettable `MandatoryGoAroundLatch` ignores `ack_reset` while `approach_active == True`
    and fails closed on any `NaN`/`±Inf`.
  - Parallel-runway/taxiway width & lateral-offset anti-misassociation gate:
      KSFO 28L (61.0 m) vs Taxiway C (22.86 m) and KSFO 28R (228.6 m lateral offset).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

CAT1_VAL_M: float = 10.0
CAT1_HAL_M: float = 40.0

KSFO_28L_WIDTH_M: float = 61.0
KSFO_TAXIWAY_C_WIDTH_M: float = 22.86
KSFO_28R_LATERAL_OFFSET_M: float = 228.6  # 750 ft parallel separation


@dataclass(frozen=True)
class ConformalRaimResult:
    """Hybrid Range-Conditioned Conformal-RAIM protection level output."""

    v_hpl_m: float
    v_vpl_m: float
    cat1_hal_m: float
    cat1_val_m: float
    hpl_within_limit: bool
    vpl_within_limit: bool
    g_max_h: float
    g_max_v: float
    range_m: float


@dataclass(frozen=True)
class RunwayAssociationResult:
    """Parallel-runway / taxiway width & lateral-offset anti-misassociation verdict."""

    accepted: bool
    width_ok: bool
    lateral_ok: bool
    observed_width_m: float
    expected_width_m: float
    lateral_offset_m: float
    reason: str


def compute_hybrid_conformal_raim_pl(
    jacobian_obs: np.ndarray,
    cov_4x4: np.ndarray,
    range_m: float,
    *,
    conformal_quantile_rad: float = 4.0e-4,
    p_bias_fault_sigma: float = 5.33,
    k_ffmd: float = 5.33,
    cat1_hal_m: float = CAT1_HAL_M,
    cat1_val_m: float = CAT1_VAL_M,
) -> ConformalRaimResult:
    """Compute Hybrid Range-Conditioned Conformal-RAIM `V-HPL` & `V-VPL` (fail-closed on NaN)."""
    j_mat = np.asarray(jacobian_obs, dtype=np.float64)
    c_mat = np.asarray(cov_4x4, dtype=np.float64)

    if not (
        np.all(np.isfinite(j_mat))
        and np.all(np.isfinite(c_mat))
        and math.isfinite(range_m)
        and range_m >= 0.0
        and math.isfinite(conformal_quantile_rad)
        and conformal_quantile_rad >= 0.0
    ):
        return ConformalRaimResult(
            v_hpl_m=math.inf,
            v_vpl_m=math.inf,
            cat1_hal_m=cat1_hal_m,
            cat1_val_m=cat1_val_m,
            hpl_within_limit=False,
            vpl_within_limit=False,
            g_max_h=math.inf,
            g_max_v=math.inf,
            range_m=float(range_m) if math.isfinite(range_m) else math.inf,
        )

    # S_proj = Cov @ J^T maps whitened measurement faults to [N, E, D, psi] estimation error
    s_proj = c_mat @ j_mat.T  # (4, M)
    # Parity projection P_par = I - J @ S_proj; residual sensitivity diagonal s_par_i = 1 - h_ii
    h_diag = np.einsum("ij,ji->i", j_mat, s_proj)
    s_par = np.maximum(1.0 - h_diag, 1e-6)

    # Worst-case single-feature failure-mode slopes g_max_h, g_max_v
    horiz_sens = np.hypot(s_proj[0, :], s_proj[1, :])
    vert_sens = np.abs(s_proj[2, :])
    g_max_h = float(np.max(horiz_sens / np.sqrt(s_par))) if s_proj.shape[1] > 0 else 0.0
    g_max_v = float(np.max(vert_sens / np.sqrt(s_par))) if s_proj.shape[1] > 0 else 0.0

    # Fault-free Gaussian position stds from posterior covariance
    sigma_h = math.sqrt(max(0.0, float(c_mat[0, 0] + c_mat[1, 1])))
    sigma_v = math.sqrt(max(0.0, float(c_mat[2, 2])))

    # Range-conditioned split-conformal margin scales linearly with slant range
    q_conf_m = range_m * conformal_quantile_rad

    # Standard Vision-RAIM H0 (fault-free) vs H1 (worst-case single-feature fault) envelope + conformal margin
    k_md = 3.29  # 1e-3 missed-detection quantile under single-feature fault hypothesis H1
    hpl_h0 = k_ffmd * sigma_h
    hpl_h1 = g_max_h * p_bias_fault_sigma + k_md * sigma_h
    vpl_h0 = k_ffmd * sigma_v
    vpl_h1 = g_max_v * p_bias_fault_sigma + k_md * sigma_v

    v_hpl = max(hpl_h0, hpl_h1) + q_conf_m
    v_vpl = max(vpl_h0, vpl_h1) + 0.5 * q_conf_m

    # Strict fail-closed limit checks (`not (pl <= al)` catches NaN/Inf)
    hpl_ok = bool(math.isfinite(v_hpl) and v_hpl <= cat1_hal_m)
    vpl_ok = bool(math.isfinite(v_vpl) and v_vpl <= cat1_val_m)

    return ConformalRaimResult(
        v_hpl_m=float(v_hpl),
        v_vpl_m=float(v_vpl),
        cat1_hal_m=cat1_hal_m,
        cat1_val_m=cat1_val_m,
        hpl_within_limit=hpl_ok,
        vpl_within_limit=vpl_ok,
        g_max_h=g_max_h,
        g_max_v=g_max_v,
        range_m=float(range_m),
    )


def check_runway_association(
    observed_width_m: float,
    lateral_offset_m: float,
    *,
    expected_width_m: float = KSFO_28L_WIDTH_M,
    width_rel_tol: float = 0.12,
    max_lateral_offset_m: float = 35.0,
) -> RunwayAssociationResult:
    """Anti-misassociation gate rejecting Taxiway C (`22.86 m`) and parallel Runway 28R (`228.6 m`)."""
    if not (
        math.isfinite(observed_width_m)
        and math.isfinite(lateral_offset_m)
        and math.isfinite(expected_width_m)
        and expected_width_m > 0.0
    ):
        return RunwayAssociationResult(
            accepted=False,
            width_ok=False,
            lateral_ok=False,
            observed_width_m=float(observed_width_m),
            expected_width_m=float(expected_width_m),
            lateral_offset_m=float(lateral_offset_m),
            reason="NON_FINITE_RUNWAY_OBSERVATION",
        )

    rel_width_err = abs(observed_width_m - expected_width_m) / expected_width_m
    width_ok = bool(rel_width_err <= width_rel_tol)
    lateral_ok = bool(abs(lateral_offset_m) <= max_lateral_offset_m)

    if not width_ok and not lateral_ok:
        reason = "WIDTH_AND_LATERAL_MISMATCH"
    elif not width_ok:
        reason = "TAXIWAY_OR_WIDTH_MISMATCH"
    elif not lateral_ok:
        reason = "PARALLEL_RUNWAY_LATERAL_MISMATCH"
    else:
        reason = "ASSOCIATED_NOMINAL"

    return RunwayAssociationResult(
        accepted=bool(width_ok and lateral_ok),
        width_ok=width_ok,
        lateral_ok=lateral_ok,
        observed_width_m=float(observed_width_m),
        expected_width_m=float(expected_width_m),
        lateral_offset_m=float(lateral_offset_m),
        reason=reason,
    )


class MandatoryGoAroundLatch:
    """Non-resettable Mandatory Go-Around safety latch during active approach (`SP1`/`SP2` §3.4)."""

    def __init__(self, cat1_val_m: float = CAT1_VAL_M, cat1_hal_m: float = CAT1_HAL_M) -> None:
        self.cat1_val_m = float(cat1_val_m)
        self.cat1_hal_m = float(cat1_hal_m)
        self.latched: bool = False

    def step(
        self,
        *,
        approach_active: bool,
        v_vpl_m: float,
        v_hpl_m: float,
        association_ok: bool,
        ack_reset: bool = False,
    ) -> bool:
        """Update latch fail-closed. `ack_reset` is strictly IGNORED while `approach_active`."""
        vpl_fault = not (math.isfinite(v_vpl_m) and v_vpl_m <= self.cat1_val_m)
        hpl_fault = not (math.isfinite(v_hpl_m) and v_hpl_m <= self.cat1_hal_m)
        assoc_fault = not bool(association_ok)

        if approach_active:
            if vpl_fault or hpl_fault or assoc_fault:
                self.latched = True
            # ack_reset is strictly ignored while approach_active is True
            return self.latched

        if ack_reset:
            self.latched = False
        return self.latched
