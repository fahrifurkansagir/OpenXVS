// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Dimensionally consistent `m^2/s` discrete-time Universal-Range (`∀ h`) `SC3BF-FIM`
//! micro-swerve selector with compile-time `PHI_CAND` sin/cos/tan tables (`Spec §4.1.1`).

use crate::wire::ActiveRangeMode;
use crate::{
    clamp_f64, ALPHA_CBF, A_MAX_EVADE, CLOSING_RHO_DOT_THRESH, DT_LOOK_S, EVADE_BANK_RAD,
    FOV_LIMIT_RAD, G_MPS2, K_ALLOW, LOCK_REL_SIGMA, NUM_BANKS, R_EVADE_TRIGGER_M, R_FLOOR_M,
    R_SEP_M, TAU_EVADE_TRIGGER_S, TRIGGER_REL_SIGMA, W_ACTIVE,
};

/// Number of deterministic candidate bank angles (`-12..=+12 deg` in `3 deg` steps).
pub const NUM_PHI_CANDS: usize = 9;

/// Candidate bank angles in degrees (`[-12, -9, -6, -3, 0, +3, +6, +9, +12]`).
pub const PHI_CAND_DEG: [f64; NUM_PHI_CANDS] = [
    -12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0,
];

/// Compile-time exact `PHI_CAND_RAD` table.
pub const PHI_CAND_RAD: [f64; NUM_PHI_CANDS] = [
    -0.20943951023931956,
    -0.15707963267948966,
    -0.10471975511965978,
    -0.05235987755982989,
    0.0,
    0.05235987755982989,
    0.10471975511965978,
    0.15707963267948966,
    0.20943951023931956,
];

/// Compile-time exact `sin(PHI_CAND_RAD)` table.
pub const PHI_CAND_SIN: [f64; NUM_PHI_CANDS] = [
    -0.20791169081775934,
    -0.15643446504023087,
    -0.10452846326765347,
    -0.052335956242943835,
    0.0,
    0.052335956242943835,
    0.10452846326765347,
    0.15643446504023087,
    0.20791169081775934,
];

/// Compile-time exact `cos(PHI_CAND_RAD)` table.
pub const PHI_CAND_COS: [f64; NUM_PHI_CANDS] = [
    0.9781476007338057,
    0.9876883405951378,
    0.9945218953682733,
    0.9986295347545738,
    1.0,
    0.9986295347545738,
    0.9945218953682733,
    0.9876883405951378,
    0.9781476007338057,
];

/// Compile-time exact `tan(PHI_CAND_RAD)` table.
pub const PHI_CAND_TAN: [f64; NUM_PHI_CANDS] = [
    -0.21255656167002213,
    -0.15838444032453627,
    -0.10510423526567647,
    -0.05240777928304121,
    0.0,
    0.05240777928304121,
    0.10510423526567647,
    0.15838444032453627,
    0.21255656167002213,
];

/// Precomputed orthonormal NED LOS triad `(u_r, u_alpha, u_beta)` plus `cos(beta)`.
#[derive(Debug, Clone, Copy)]
pub struct LosTriadNed {
    pub u_r: [f64; 3],
    pub u_alpha: [f64; 3],
    pub u_beta: [f64; 3],
    pub cos_beta: f64,
}

impl LosTriadNed {
    pub fn from_angles(alpha_rad: f64, beta_rad: f64) -> Self {
        let sa = libm::sin(alpha_rad);
        let ca = libm::cos(alpha_rad);
        let sb = libm::sin(beta_rad);
        let cb = libm::cos(beta_rad);
        Self {
            u_r: [cb * ca, cb * sa, -sb],
            u_alpha: [-sa, ca, 0.0],
            u_beta: [-sb * ca, -sb * sa, -cb],
            cos_beta: cb,
        }
    }
}

/// Inputs to the pure Kani-verified `select_active_swerve` decision kernel.
#[derive(Debug, Clone, Copy)]
pub struct SwerveInputs {
    pub r_banks: [f64; NUM_BANKS],
    pub weights: [f64; NUM_BANKS],
    pub r_hat_m: f64,
    pub sigma_r_m: f64,
    pub r_safe_lower_m: f64,
    pub rho_dot_inv_s: f64,
    pub alpha_dot: f64,
    pub beta_dot: f64,
    pub alpha_body_rad: f64,
    pub triad: LosTriadNed,
    pub own_fwd_ned: [f64; 3],
    pub own_right_ned: [f64; 3],
    pub tas_mps: f64,
    pub v_stall_mps: f64,
    pub sigma_bearing_rad: f64,
    pub cbdr_probe_active: bool,
    pub was_locked: bool,
    pub preferred_swerve_sign: f64,
}

/// Output of `select_active_swerve`.
#[derive(Debug, Clone, Copy)]
pub struct SwerveDecision {
    pub mode: ActiveRangeMode,
    pub cmd_bank_rad: f64,
    pub min_sc3bf_margin: f64,
    pub fim_gain: f64,
    pub sc3bf_all_safe: bool,
    pub fov_safe: bool,
    pub now_locked: bool,
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    let s = dot3(a, a);
    if s > 0.0 {
        libm::sqrt(s)
    } else {
        0.0
    }
}

/// Compute dimensionally exact (`m^2/s`) SC3BF allowance:
/// `eta(R_h, V_rel) = K_ALLOW * max(0, R_h - R_SEP_M)^2 * A_MAX_EVADE / max(V_rel, 1.0)`.
#[inline]
pub fn sc3bf_eta(r_h: f64, v_rel_norm: f64) -> f64 {
    let excess = if r_h > R_SEP_M { r_h - R_SEP_M } else { 0.0 };
    let v_eff = if v_rel_norm > 1.0 { v_rel_norm } else { 1.0 };
    K_ALLOW * (excess * excess) * A_MAX_EVADE / v_eff
}

/// Compute `cos(theta_c(R_h)) = sqrt(max(0, 1 - (R_SEP_M / max(R_h, R_SEP_M + 1.0))^2))`
/// (with `0.0` when `R_h <= R_SEP_M`).
#[inline]
pub fn sc3bf_cos_theta_c(r_h: f64) -> f64 {
    if r_h <= R_SEP_M {
        return 0.0;
    }
    let denom = if r_h > R_SEP_M + 1.0 {
        r_h
    } else {
        R_SEP_M + 1.0
    };
    let ratio = R_SEP_M / denom;
    let arg = 1.0 - ratio * ratio;
    if arg > 0.0 {
        libm::sqrt(arg)
    } else {
        0.0
    }
}

/// Predicted ownship velocity change `Delta_v_own(phi, DT_LOOK_S)` in NED using exact
/// 2nd/3rd-order polynomial rotation over small lookahead angle `dpsi` (zero runtime trig).
#[inline]
pub fn delta_v_own_ned(
    tan_phi: f64,
    tas_mps: f64,
    own_fwd_ned: [f64; 3],
    own_right_ned: [f64; 3],
) -> [f64; 3] {
    let v_eff = if tas_mps > 1.0 { tas_mps } else { 1.0 };
    let dpsi = clamp_f64((G_MPS2 * tan_phi / v_eff) * DT_LOOK_S, -0.5, 0.5);
    let dpsi2 = dpsi * dpsi;
    // cos(dpsi) - 1 ≈ -dpsi^2/2 + dpsi^4/24; sin(dpsi) ≈ dpsi - dpsi^3/6
    let c_minus_1 = -0.5 * dpsi2 + (dpsi2 * dpsi2) / 24.0;
    let s_approx = dpsi - (dpsi2 * dpsi) / 6.0;
    [
        v_eff * (c_minus_1 * own_fwd_ned[0] + s_approx * own_right_ned[0]),
        v_eff * (c_minus_1 * own_fwd_ned[1] + s_approx * own_right_ned[1]),
        v_eff * (c_minus_1 * own_fwd_ned[2] + s_approx * own_right_ned[2]),
    ]
}

/// Evaluate `h_SC3BF(R_h, phi)` in `m^2/s` for one range hypothesis `R_h` and candidate `tan(phi)`.
pub fn eval_sc3bf_h(inp: &SwerveInputs, r_h: f64, tan_phi: f64) -> f64 {
    let r = if r_h > R_FLOOR_M { r_h } else { R_FLOOR_M };
    let p_rel = [
        r * inp.triad.u_r[0],
        r * inp.triad.u_r[1],
        r * inp.triad.u_r[2],
    ];
    let v_r = r * inp.rho_dot_inv_s;
    let v_a = r * inp.alpha_dot * inp.triad.cos_beta;
    let v_b = r * inp.beta_dot;

    let dv = delta_v_own_ned(tan_phi, inp.tas_mps, inp.own_fwd_ned, inp.own_right_ned);
    let v_rel = [
        v_r * inp.triad.u_r[0] + v_a * inp.triad.u_alpha[0] + v_b * inp.triad.u_beta[0] - dv[0],
        v_r * inp.triad.u_r[1] + v_a * inp.triad.u_alpha[1] + v_b * inp.triad.u_beta[1] - dv[1],
        v_r * inp.triad.u_r[2] + v_a * inp.triad.u_alpha[2] + v_b * inp.triad.u_beta[2] - dv[2],
    ];

    let v_norm = norm3(v_rel);
    let pv = dot3(p_rel, v_rel);
    let cos_tc = sc3bf_cos_theta_c(r);
    let eta = sc3bf_eta(r, v_norm);
    pv + r * v_norm * cos_tc + eta
}

/// Pure algebraic discrete-time SC3BF invariance margin and safety predicate from `(h_0, h_phi)`.
///
/// - When `h_0 >= 0.0` (inside safe set): `margin = h_phi - (1 - ALPHA_CBF * DT_LOOK_S) * h_0`,
///   safe iff `margin >= 0.0`.
/// - When `h_0 < 0.0` (already inside violated collision cone): `margin = h_phi - h_0`,
///   safe iff **strictly healing** `margin > 0.0` (`h_phi > h_0`), so straight flight (`phi = 0.0`,
///   `h_phi == h_0`) is `safe = false` inside a violated collision cone and an empty feasible set
///   genuinely escalates to `ActiveRangeMode::ImmediateEvade`.
#[inline]
pub fn sc3bf_margin_and_safe_from_h(h_0: f64, h_phi: f64) -> (f64, bool) {
    if !h_0.is_finite() || !h_phi.is_finite() {
        return (-1.0e6, false);
    }
    if h_0 >= 0.0 {
        let m = h_phi - (1.0 - ALPHA_CBF * DT_LOOK_S) * h_0;
        (m, m.is_finite() && m >= 0.0)
    } else {
        let m = h_phi - h_0;
        (m, m.is_finite() && m > 0.0)
    }
}

/// Discrete-time SC3BF invariance margin and safety predicate for one range hypothesis `R_h`.
#[inline]
pub fn sc3bf_hypothesis_margin_and_safe(inp: &SwerveInputs, r_h: f64, tan_phi: f64) -> (f64, bool) {
    let h_0 = eval_sc3bf_h(inp, r_h, 0.0);
    let h_phi = eval_sc3bf_h(inp, r_h, tan_phi);
    sc3bf_margin_and_safe_from_h(h_0, h_phi)
}

/// Check camera FOV retention after lookahead `DT_LOOK_S = 1.0 s` (`|alpha_body_pred| <= 47 deg`).
#[inline]
pub fn candidate_fov_safe(inp: &SwerveInputs, tan_phi: f64) -> bool {
    let v_eff = if inp.tas_mps > 1.0 { inp.tas_mps } else { 1.0 };
    let dpsi = clamp_f64((G_MPS2 * tan_phi / v_eff) * DT_LOOK_S, -0.5, 0.5);
    let pred_alpha_body = inp.alpha_body_rad - dpsi + inp.alpha_dot * DT_LOOK_S;
    pred_alpha_body.is_finite() && libm::fabs(pred_alpha_body) <= FOV_LIMIT_RAD
}

/// Compute minimum SC3BF margin across ALL active hypotheses (`w_m >= W_ACTIVE`) union `{r_safe_lower_m}`.
pub fn eval_universal_sc3bf(inp: &SwerveInputs, tan_phi: f64) -> (f64, bool) {
    let mut min_margin = 1.0e300_f64;
    let mut all_safe = true;

    let mut m = 0usize;
    while m < NUM_BANKS {
        if inp.weights[m] >= W_ACTIVE {
            let (marg, safe) = sc3bf_hypothesis_margin_and_safe(inp, inp.r_banks[m], tan_phi);
            if marg < min_margin {
                min_margin = marg;
            }
            if !safe {
                all_safe = false;
            }
        }
        m += 1;
    }

    let (marg_low, safe_low) = sc3bf_hypothesis_margin_and_safe(inp, inp.r_safe_lower_m, tan_phi);
    if marg_low < min_margin {
        min_margin = marg_low;
    }
    if !safe_low {
        all_safe = false;
    }

    if !min_margin.is_finite() {
        min_margin = -1.0e6;
        all_safe = false;
    }
    (min_margin, all_safe)
}

/// Deterministic evade bank direction (`±25 deg` away from bearing; right-of-way FAR 91.113 on head-on).
#[inline]
fn evade_bank_rad(alpha_body_rad: f64) -> f64 {
    if alpha_body_rad > 0.01 {
        -EVADE_BANK_RAD
    } else {
        EVADE_BANK_RAD
    }
}

/// Pure Kani-verified mode/safety/fail-closed decision kernel given pre-evaluated per-candidate
/// universal SC3BF margins, safety flags, FOV flags, and FIM gains.
#[allow(clippy::too_many_arguments)]
pub fn select_active_swerve_from_evals(
    r_hat_m: f64,
    sigma_r_m: f64,
    r_safe_lower_m: f64,
    rho_dot_inv_s: f64,
    alpha_body_rad: f64,
    h_low_zero: f64,
    tas_mps: f64,
    v_stall_mps: f64,
    cbdr_probe_active: bool,
    was_locked: bool,
    preferred_swerve_sign: f64,
    cand_margins: &[f64; NUM_PHI_CANDS],
    cand_all_safe: &[bool; NUM_PHI_CANDS],
    cand_fov_safe: &[bool; NUM_PHI_CANDS],
    cand_fim: &[f64; NUM_PHI_CANDS],
) -> SwerveDecision {
    let r_hat_eff = if r_hat_m.is_finite() && r_hat_m > R_FLOOR_M {
        r_hat_m
    } else {
        R_FLOOR_M
    };
    let sig_valid = sigma_r_m.is_finite() && sigma_r_m >= 0.0;

    // Hysteresis lock state: lock at sigma_r <= 0.08 * r_hat, unlock/trigger at > 0.10 * r_hat
    // (multiplication by positive r_hat_eff is algebraically identical to division with zero f64/f64)
    let now_locked = if sig_valid {
        if was_locked {
            sigma_r_m <= TRIGGER_REL_SIGMA * r_hat_eff
        } else {
            sigma_r_m <= LOCK_REL_SIGMA * r_hat_eff
        }
    } else {
        false
    };

    let is_closing = rho_dot_inv_s < CLOSING_RHO_DOT_THRESH;
    let tau_terminal = rho_dot_inv_s <= -(1.0 / TAU_EVADE_TRIGGER_S);

    let marg0 = if cand_margins[4].is_finite() {
        cand_margins[4]
    } else {
        -1.0e6
    };
    let safe0 = cand_all_safe[4] && cand_margins[4].is_finite() && cand_margins[4] >= 0.0;
    let fov0 = cand_fov_safe[4];

    // Terminal emergency escalation check:
    let range_terminal = r_hat_m <= R_EVADE_TRIGGER_M
        || (r_safe_lower_m <= R_EVADE_TRIGGER_M && sig_valid && sigma_r_m <= 0.35 * r_hat_eff);
    if is_closing && (tau_terminal || range_terminal) && h_low_zero < 0.0 {
        return SwerveDecision {
            mode: ActiveRangeMode::ImmediateEvade,
            cmd_bank_rad: evade_bank_rad(alpha_body_rad),
            min_sc3bf_margin: marg0,
            fim_gain: 0.0,
            sc3bf_all_safe: false,
            fov_safe: fov0,
            now_locked,
        };
    }

    // Envelope guard: V_tas >= 1.25 * V_stall
    let envelope_ok =
        tas_mps.is_finite() && v_stall_mps.is_finite() && tas_mps >= 1.25 * v_stall_mps;

    if !(is_closing || cbdr_probe_active) {
        return SwerveDecision {
            mode: if now_locked {
                ActiveRangeMode::Locked
            } else {
                ActiveRangeMode::NominalHold
            },
            cmd_bank_rad: 0.0,
            min_sc3bf_margin: marg0,
            fim_gain: 0.0,
            sc3bf_all_safe: safe0,
            fov_safe: fov0,
            now_locked,
        };
    }

    if now_locked {
        return SwerveDecision {
            mode: ActiveRangeMode::Locked,
            cmd_bank_rad: 0.0,
            min_sc3bf_margin: marg0,
            fim_gain: 0.0,
            sc3bf_all_safe: safe0,
            fov_safe: fov0,
            now_locked: true,
        };
    }

    if !envelope_ok {
        return SwerveDecision {
            mode: ActiveRangeMode::NominalHold,
            cmd_bank_rad: 0.0,
            min_sc3bf_margin: marg0,
            fim_gain: 0.0,
            sc3bf_all_safe: safe0,
            fov_safe: fov0,
            now_locked: false,
        };
    }

    let lambda_u = 1.0e-2_f64;
    let lambda_margin = 1.0e-4_f64;
    let pref_sign = if preferred_swerve_sign < 0.0 {
        -1.0
    } else {
        1.0
    };

    let mut found_swerve = false;
    let mut best_k = 4usize;
    let mut best_score = -1.0e300_f64;
    let mut best_margin = marg0;
    let mut best_fim = 0.0f64;

    let mut k = 0usize;
    while k < NUM_PHI_CANDS {
        let phi = PHI_CAND_RAD[k];
        let min_marg = cand_margins[k];
        let all_safe = cand_all_safe[k] && min_marg.is_finite() && min_marg >= 0.0;
        let fov_ok = cand_fov_safe[k];
        let fim_gain = cand_fim[k];

        if fov_ok && all_safe && fim_gain.is_finite() {
            let row_tiebreak = if libm::fabs(alpha_body_rad) < 0.07 {
                1.0e-3 * pref_sign * phi
            } else if alpha_body_rad > 0.0 {
                -1.0e-3 * phi
            } else {
                1.0e-3 * phi
            };

            let score = fim_gain - lambda_u * phi * phi + lambda_margin * min_marg + row_tiebreak;
            if score.is_finite() && (!found_swerve || score > best_score) {
                found_swerve = true;
                best_k = k;
                best_score = score;
                best_margin = min_marg;
                best_fim = fim_gain;
            }
        }
        k += 1;
    }

    if !found_swerve {
        return SwerveDecision {
            mode: ActiveRangeMode::ImmediateEvade,
            cmd_bank_rad: evade_bank_rad(alpha_body_rad),
            min_sc3bf_margin: marg0,
            fim_gain: 0.0,
            sc3bf_all_safe: false,
            fov_safe: fov0,
            now_locked: false,
        };
    }

    if best_k == 4 {
        return SwerveDecision {
            mode: ActiveRangeMode::NominalHold,
            cmd_bank_rad: 0.0,
            min_sc3bf_margin: best_margin,
            fim_gain: best_fim,
            sc3bf_all_safe: true,
            fov_safe: true,
            now_locked: false,
        };
    }

    SwerveDecision {
        mode: ActiveRangeMode::ActiveSwerve,
        cmd_bank_rad: PHI_CAND_RAD[best_k],
        min_sc3bf_margin: best_margin,
        fim_gain: best_fim,
        sc3bf_all_safe: true,
        fov_safe: true,
        now_locked: false,
    }
}

/// Pure Universal-Range (`∀ h`) `SC3BF-FIM` micro-swerve selector.
pub fn select_active_swerve(inp: &SwerveInputs) -> SwerveDecision {
    let h_low_zero = eval_sc3bf_h(inp, inp.r_safe_lower_m, 0.0);
    let sig_a = if inp.sigma_bearing_rad > 1.0e-5 {
        inp.sigma_bearing_rad
    } else {
        1.0e-5
    };
    let t_swerve = 2.5_f64;
    let t4_over_12s2 = (t_swerve * t_swerve * t_swerve * t_swerve) / (12.0 * sig_a * sig_a);
    let right_dot_ualpha = dot3(inp.own_right_ned, inp.triad.u_alpha);

    let mut cand_margins = [0.0f64; NUM_PHI_CANDS];
    let mut cand_all_safe = [false; NUM_PHI_CANDS];
    let mut cand_fov_safe = [false; NUM_PHI_CANDS];
    let mut cand_fim = [0.0f64; NUM_PHI_CANDS];

    let mut k = 0usize;
    while k < NUM_PHI_CANDS {
        let tan_phi = PHI_CAND_TAN[k];
        let (min_marg, all_safe) = eval_universal_sc3bf(inp, tan_phi);
        cand_margins[k] = min_marg;
        cand_all_safe[k] = all_safe;
        cand_fov_safe[k] = candidate_fov_safe(inp, tan_phi);

        let a_az_cand = G_MPS2 * tan_phi * right_dot_ualpha;
        let a_az2 = a_az_cand * a_az_cand;
        let mut fim_gain = 0.0f64;
        let mut m = 0usize;
        while m < NUM_BANKS {
            let rm = if inp.r_banks[m] > R_FLOOR_M {
                inp.r_banks[m]
            } else {
                R_FLOOR_M
            };
            fim_gain += inp.weights[m] * (a_az2 / (rm * rm)) * t4_over_12s2;
            m += 1;
        }
        cand_fim[k] = fim_gain;
        k += 1;
    }

    select_active_swerve_from_evals(
        inp.r_hat_m,
        inp.sigma_r_m,
        inp.r_safe_lower_m,
        inp.rho_dot_inv_s,
        inp.alpha_body_rad,
        h_low_zero,
        inp.tas_mps,
        inp.v_stall_mps,
        inp.cbdr_probe_active,
        inp.was_locked,
        inp.preferred_swerve_sign,
        &cand_margins,
        &cand_all_safe,
        &cand_fov_safe,
        &cand_fim,
    )
}
