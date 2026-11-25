//! OOD write-gated Guarded Altitude-Mondrian `DtACI` (Adaptive Conformal Inference).
//!
//! Maintains 4 altitude strata with frozen reference baseline `B_ref`, Conformal CUSUM `S_t`,
//! and an OOD write-gate that freezes calibration updates whenever `u_ood >= 1.0`.

use crate::det_f64::clamp;

const NUM_BINS: usize = 4;
const ALPHA_TARGET: f64 = 0.05;
const GAMMA_STEP: f64 = 0.015;
const CUSUM_DRIFT: f64 = 0.03;

#[derive(Debug, Clone, Copy)]
struct MondrianBin {
    alpha_t: f64,
    q_ref_rad: f64,
    q_cur_rad: f64,
    conformal_cusum: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct GuardedDtAci {
    bins: [MondrianBin; NUM_BINS],
    frozen: bool,
    last_cone_rad: f64,
}

impl Default for GuardedDtAci {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardedDtAci {
    pub const fn new() -> Self {
        let base = MondrianBin {
            alpha_t: ALPHA_TARGET,
            q_ref_rad: 0.0018, // ~1.8 mrad frozen reference cone half-angle
            q_cur_rad: 0.0018,
            conformal_cusum: 0.0,
        };
        Self {
            bins: [base; NUM_BINS],
            frozen: false,
            last_cone_rad: 0.0018,
        }
    }

    #[inline]
    fn bin_index(alt_agl_m: f64) -> usize {
        if !alt_agl_m.is_finite() || alt_agl_m < 300.0 {
            0
        } else if alt_agl_m < 1000.0 {
            1
        } else if alt_agl_m < 3000.0 {
            2
        } else {
            3
        }
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    #[inline]
    pub fn alpha(&self, alt_agl_m: f64) -> f64 {
        self.bins[Self::bin_index(alt_agl_m)].alpha_t
    }

    #[inline]
    pub fn conformal_cusum(&self, alt_agl_m: f64) -> f64 {
        self.bins[Self::bin_index(alt_agl_m)].conformal_cusum
    }

    #[inline]
    pub fn last_cone_rad(&self) -> f64 {
        self.last_cone_rad
    }

    /// Update Guarded Altitude-Mondrian `DtACI` with observed bearing nonconformity `score_rad`
    /// and normalized OOD statistic `u_ood`.
    ///
    /// Returns the certified cone half-angle in radians. When `u_ood >= 1.0` (or non-finite),
    /// the write-gate FREEZES `alpha_t` and `q_cur_rad` and returns a conservative inflated cone.
    pub fn update(&mut self, alt_agl_m: f64, score_rad: f64, u_ood: f64) -> f64 {
        let idx = Self::bin_index(alt_agl_m);
        let bin = &mut self.bins[idx];

        let ood_trip = !u_ood.is_finite() || u_ood >= 1.0 || !score_rad.is_finite();
        if ood_trip {
            self.frozen = true;
            let ood_scale = if u_ood.is_finite() {
                clamp(1.0 + 0.5 * (u_ood - 1.0), 1.25, 3.5)
            } else {
                3.5
            };
            let inflated = (bin.q_cur_rad.max(bin.q_ref_rad)) * ood_scale;
            self.last_cone_rad = inflated;
            return inflated;
        }

        self.frozen = false;
        let err_t = if score_rad > bin.q_cur_rad { 1.0 } else { 0.0 };
        bin.alpha_t = clamp(bin.alpha_t + GAMMA_STEP * (ALPHA_TARGET - err_t), 0.005, 0.25);

        // Conformal CUSUM tracking excess miscoverage above target ALPHA_TARGET
        let inc = (err_t - ALPHA_TARGET) - CUSUM_DRIFT;
        bin.conformal_cusum = (bin.conformal_cusum + inc).max(0.0);

        // Conservative quantile mapping from alpha_t anchored to frozen B_ref
        let ratio = ALPHA_TARGET / bin.alpha_t;
        let q_adapt = bin.q_ref_rad * clamp(ratio, 0.75, 2.5);
        bin.q_cur_rad = q_adapt.max(bin.q_ref_rad * 0.85);
        self.last_cone_rad = bin.q_cur_rad;
        self.last_cone_rad
    }
}
