// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Frozen Split Conformal safe lower range envelope (`CP-MSC`, Spec §4.1.1 / DR-20 §3.3).

use crate::{clamp_f64, R_CEIL_M, R_FLOOR_M, SIGMA_FLOOR_M};

/// Compute the distribution-free Split Conformal lower safe range bound:
///
/// `R_safe_lower = clamp(R_hat - q_hat * max(sigma_R, SIGMA_FLOOR_M), R_FLOOR_M, R_hat)`
///
/// Guarantees `R_FLOOR_M <= R_safe_lower <= clamp(R_hat, R_FLOOR_M, R_CEIL_M)` and finiteness
/// for all finite or non-finite inputs.
pub fn compute_conformal_lower_bound(r_hat: f64, sigma_r: f64, q_hat: f64) -> f64 {
    let r_clamped = if r_hat.is_finite() {
        clamp_f64(r_hat, R_FLOOR_M, R_CEIL_M)
    } else {
        R_FLOOR_M
    };
    let s_eff = if sigma_r.is_finite() && sigma_r > SIGMA_FLOOR_M {
        sigma_r
    } else {
        SIGMA_FLOOR_M
    };
    let q_eff = if q_hat.is_finite() && q_hat > 0.0 {
        q_hat
    } else {
        0.0
    };
    let raw = r_clamped - q_eff * s_eff;
    if !raw.is_finite() {
        return R_FLOOR_M;
    }
    clamp_f64(raw, R_FLOOR_M, r_clamped)
}
