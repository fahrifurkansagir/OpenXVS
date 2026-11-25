//! Fail-closed `DetF64 -> i64` boundary quantizer & float-derived alarm conversion (`SP1_LIVEDECK_DESIGN.md` §4.1.1).
//!
//! Raw Rust `f64 as i64` fails open (`NaN -> 0`, silently passing `PL <= AL`), and raw `stat > thr`
//! fails open (`NaN > thr == false`). Every floating-point value crossing into `xvs-rta-core` MUST
//! pass through this module; any non-finite (`NaN` / `±Inf`) or out-of-range value sets
//! `quantize_fault = true` and substitutes fail-closed sentinels (`i64::MAX`), forcing `P0` at 0 ticks.

use crate::simplex::RtaTickInput;

/// Exact IEEE-754 integer representation ceiling (`2^52 = 4_503_599_627_370_496`).
/// Every integer in `[-SAFE_F64_INT_LIMIT, +SAFE_F64_INT_LIMIT]` is represented exactly in `f64`,
/// eliminating rounding-to-`2^63` saturation bugs on `f64 as i64`.
pub const SAFE_F64_INT_LIMIT: i64 = 1_i64 << 52;

/// Scale factor from radians to SI microradians (`1 rad = 1_000_000 urad`).
pub const RAD_TO_URAD_SCALE: f64 = 1_000_000.0;

/// Scale factor from metres to SI millimetres (`1 m = 1_000 mm`).
pub const M_TO_MM_SCALE: f64 = 1_000.0;

/// Maximum valid quantized angular bound (`100 rad = 100_000_000 urad`).
pub const MAX_VALID_ANGLE_URAD: i64 = 100_000_000;

/// Maximum valid quantized distance bound (`10_000 km = 10_000_000_000 mm`).
pub const MAX_VALID_DIST_MM: i64 = 10_000_000_000;

/// Boundary quantization failure modes. Every fault forces `quantize_fault = true` -> `P0 (InvalidInhibit)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantFault {
    NonFinite = 1,
    NegativeInput = 2,
    OutOfRange = 3,
    InvalidScaleOrBounds = 4,
}

/// Raw floating-point telemetry crossing Zone A / `xvs-math-core` into Zone B (`xvs-rta-core`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RtaFloatTelemetry {
    pub horizon_pl_rad: f64,
    pub horizon_al_rad: f64,
    pub v_vpl_m: f64,
    pub v_val_m: f64,
    pub v_hpl_m: f64,
    pub v_hal_m: f64,
    pub epipolar_max_rad: [f64; 6],
    pub signed_horizon_residual_rad: f64,
    pub runway_width_delta_m: f64,
    pub runway_width_delta_limit_m: f64,
    pub cusum_stat: f64,
    pub cusum_thr: f64,
    pub ood_global_score: f64,
    pub ood_global_thr: f64,
    pub ood_cam_scores: [f64; 6],
    pub ood_cam_thr: f64,
    pub cam_health_mask: u8,
    pub ml_detector_fault: bool,
    pub transient_dropout: bool,
    pub approach_active: bool,
    pub go_around_ack_reset: bool,
}

/// Fail-closed non-negative upper-bound quantizer (`f64 -> i64`).
///
/// - `-0.0` and `+0.0` map cleanly to `Ok(0)`.
/// - `NaN`, signaling `NaN`, `+Inf`, `-Inf` return `Err(QuantFault::NonFinite)`.
/// - Any strictly negative value (`val < 0.0`) returns `Err(QuantFault::NegativeInput)`.
/// - Any finite positive value rounds conservatively UP (`ceil`) to an exact `i64`, and values
///   exceeding `max_valid` return `Err(QuantFault::OutOfRange)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn quantize_nonneg_upper_i64(val: f64, scale: f64, max_valid: i64) -> Result<i64, QuantFault> {
    if !scale.is_finite() || scale <= 0.0 || !(0..=SAFE_F64_INT_LIMIT).contains(&max_valid) {
        return Err(QuantFault::InvalidScaleOrBounds);
    }
    if !val.is_finite() {
        return Err(QuantFault::NonFinite);
    }
    if val == 0.0 {
        return Ok(0);
    }
    if val < 0.0 {
        return Err(QuantFault::NegativeInput);
    }

    let scaled = val * scale;
    if !scaled.is_finite() {
        return Err(QuantFault::NonFinite);
    }
    if scaled < 0.0 {
        return Err(QuantFault::NegativeInput);
    }
    let max_f = max_valid as f64;
    if scaled > max_f {
        return Err(QuantFault::OutOfRange);
    }

    let trunc = scaled as i64;
    let ceil_val = if (trunc as f64) < scaled {
        match trunc.checked_add(1) {
            Some(v) => v,
            None => return Err(QuantFault::OutOfRange),
        }
    } else {
        trunc
    };
    if ceil_val < 0 || ceil_val > max_valid {
        return Err(QuantFault::OutOfRange);
    }
    Ok(ceil_val)
}

/// Fail-closed signed bounded quantizer (`f64 -> i64`) for physical residuals that may legitimately
/// be negative (e.g. `signed_horizon_residual_rad`, `runway_width_delta_m`).
///
/// - `-0.0` and `+0.0` map cleanly to `Ok(0)` when `0` lies in `[min_valid, max_valid]`.
/// - `NaN`, signaling `NaN`, `+Inf`, `-Inf` return `Err(QuantFault::NonFinite)`.
/// - Rounds to nearest integer (ties away from zero) without requiring `std` or `libm`.
#[allow(clippy::arithmetic_side_effects)]
pub fn quantize_signed_bounded_i64(
    val: f64,
    scale: f64,
    min_valid: i64,
    max_valid: i64,
) -> Result<i64, QuantFault> {
    if !scale.is_finite()
        || scale <= 0.0
        || min_valid > max_valid
        || min_valid < -SAFE_F64_INT_LIMIT
        || max_valid > SAFE_F64_INT_LIMIT
    {
        return Err(QuantFault::InvalidScaleOrBounds);
    }
    if !val.is_finite() {
        return Err(QuantFault::NonFinite);
    }
    if val == 0.0 {
        return if min_valid <= 0 && max_valid >= 0 {
            Ok(0)
        } else {
            Err(QuantFault::OutOfRange)
        };
    }

    let scaled = val * scale;
    if !scaled.is_finite() {
        return Err(QuantFault::NonFinite);
    }
    let min_f = min_valid as f64;
    let max_f = max_valid as f64;
    if scaled < min_f || scaled > max_f {
        return Err(QuantFault::OutOfRange);
    }

    let adjusted = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    if !adjusted.is_finite() {
        return Err(QuantFault::OutOfRange);
    }
    let q = adjusted as i64;
    if q < min_valid || q > max_valid {
        return Err(QuantFault::OutOfRange);
    }
    Ok(q)
}

/// Fail-closed float alarm predicate: returns `true` (ALARM) whenever `stat` or `thr` is non-finite
/// OR `!(stat <= thr)`.
#[inline]
pub fn fail_closed_gt(stat: f64, thr: f64) -> (bool, bool) {
    let finite_ok = stat.is_finite() && thr.is_finite();
    let alarm = !finite_ok
        || matches!(
            stat.partial_cmp(&thr),
            None | Some(core::cmp::Ordering::Greater)
        );
    (alarm, !finite_ok)
}

/// Fail-closed symmetric bounded float alarm: returns `true` (ALARM) whenever `val` or `limit` is
/// non-finite, `limit < 0.0`, or `!(-limit <= val && val <= limit)`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn fail_closed_abs_gt(val: f64, limit: f64) -> (bool, bool) {
    let finite_ok = val.is_finite() && limit.is_finite() && limit >= 0.0;
    let in_bounds = finite_ok && (-limit <= val && val <= limit);
    (!in_bounds, !finite_ok)
}

/// Convert `RtaFloatTelemetry` into a pure-integer `RtaTickInput` strictly fail-closed.
///
/// Any non-finite (`NaN`/`±Inf`) or out-of-range scalar OR alarm-driving statistic sets
/// `quantize_fault = true` and substitutes fail-closed sentinels (`i64::MAX`), forcing
/// `SimplexMode::InvalidInhibit (P0)` at 0 ticks.
pub fn build_rta_input_from_f64(t: &RtaFloatTelemetry) -> RtaTickInput {
    let mut quantize_fault = false;

    let horizon_pl_urad = match quantize_nonneg_upper_i64(
        t.horizon_pl_rad,
        RAD_TO_URAD_SCALE,
        MAX_VALID_ANGLE_URAD,
    ) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            i64::MAX
        }
    };
    let horizon_al_urad = match quantize_nonneg_upper_i64(
        t.horizon_al_rad,
        RAD_TO_URAD_SCALE,
        MAX_VALID_ANGLE_URAD,
    ) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            0
        }
    };
    let v_vpl_mm = match quantize_nonneg_upper_i64(t.v_vpl_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            i64::MAX
        }
    };
    let v_val_mm = match quantize_nonneg_upper_i64(t.v_val_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            0
        }
    };
    let v_hpl_mm = match quantize_nonneg_upper_i64(t.v_hpl_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            i64::MAX
        }
    };
    let v_hal_mm = match quantize_nonneg_upper_i64(t.v_hal_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            0
        }
    };

    let mut epipolar_max_urad = [0_i64; 6];
    let mut idx: usize = 0;
    while idx < 6 {
        epipolar_max_urad[idx] = match quantize_nonneg_upper_i64(
            t.epipolar_max_rad[idx],
            RAD_TO_URAD_SCALE,
            MAX_VALID_ANGLE_URAD,
        ) {
            Ok(v) => v,
            Err(_) => {
                quantize_fault = true;
                i64::MAX
            }
        };
        idx = idx.saturating_add(1);
    }

    let signed_horizon_residual_urad = match quantize_signed_bounded_i64(
        t.signed_horizon_residual_rad,
        RAD_TO_URAD_SCALE,
        -MAX_VALID_ANGLE_URAD,
        MAX_VALID_ANGLE_URAD,
    ) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            i64::MAX
        }
    };

    let runway_width_delta_mm = match quantize_signed_bounded_i64(
        t.runway_width_delta_m,
        M_TO_MM_SCALE,
        -MAX_VALID_DIST_MM,
        MAX_VALID_DIST_MM,
    ) {
        Ok(v) => v,
        Err(_) => {
            quantize_fault = true;
            i64::MAX
        }
    };

    // Fail-closed float-derived boolean alarms (`!(stat <= thr)` / `!(lo <= val && val <= hi)`)
    let (runway_width_fault, rw_nonfinite) =
        fail_closed_abs_gt(t.runway_width_delta_m, t.runway_width_delta_limit_m);
    if rw_nonfinite {
        quantize_fault = true;
    }

    let (spoof_alarm, cusum_nonfinite) = fail_closed_gt(t.cusum_stat, t.cusum_thr);
    if cusum_nonfinite {
        quantize_fault = true;
    }

    let (ood_global_exceeded, ood_g_nonfinite) =
        fail_closed_gt(t.ood_global_score, t.ood_global_thr);
    if ood_g_nonfinite {
        quantize_fault = true;
    }

    let mut ood_exceeded_mask: u8 = 0;
    let mut cam_i: usize = 0;
    while cam_i < 6 {
        let (cam_ood, cam_nonfinite) = fail_closed_gt(t.ood_cam_scores[cam_i], t.ood_cam_thr);
        if cam_nonfinite {
            quantize_fault = true;
        }
        if cam_ood {
            ood_exceeded_mask |= 1_u8 << cam_i;
        }
        cam_i = cam_i.saturating_add(1);
    }

    RtaTickInput {
        cam_health_mask: t.cam_health_mask & 0x3F,
        ood_exceeded_mask: ood_exceeded_mask & 0x3F,
        horizon_pl_urad,
        horizon_al_urad,
        v_vpl_mm,
        v_val_mm,
        v_hpl_mm,
        v_hal_mm,
        epipolar_max_urad,
        signed_horizon_residual_urad,
        runway_width_delta_mm,
        spoof_alarm,
        ml_detector_fault: t.ml_detector_fault,
        ood_global_exceeded,
        runway_width_fault,
        transient_dropout: t.transient_dropout,
        approach_active: t.approach_active,
        go_around_ack_reset: t.go_around_ack_reset,
        quantize_fault,
    }
}
