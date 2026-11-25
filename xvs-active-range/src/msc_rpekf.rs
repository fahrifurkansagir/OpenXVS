// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `M = 6` log-spaced Modified Spherical Coordinates Range-Parameterized EKF (`MSC-RPEKF`).
//!
//! State vector per bank `m in 0..6`:
//!   `x_m = [alpha, beta, alpha_dot, beta_dot, rho, rho_dot]`
//! where strictly in NED:
//!   - `alpha = atan2(p_E, p_N)` (azimuth rad)
//!   - `beta  = atan2(-p_D, hypot(p_N, p_E))` (elevation rad, positive above horizon)
//!   - `rho   = 1 / R` (`m^-1`)
//!   - `rho_dot = R_dot / R = -1 / tau` (`s^-1`, negative when closing)

use crate::{
    clamp_f64, wrap_pi, ActiveRangeConfig, ActiveRangeMeasurement, OwnshipStateNed,
    KOSCH_MAX_LOG_DELTA, KOSCH_MAX_SIZE_PX, NUM_BANKS, R_CEIL_M, R_FLOOR_M, R_INIT_M,
    TAU_MIN_SIZE_PX, TAU_MIN_TRACK_AGE, W_FLOOR,
};
use crate::wire::{FLAG_KOSCH_FUSED, FLAG_TAU_FUSED};

/// Single-bank 6-state MSC EKF.
#[derive(Debug, Clone, Copy)]
pub struct MscBank {
    pub x: [f64; 6],
    pub p: [[f64; 6]; 6],
}

impl MscBank {
    pub fn init(alpha_rad: f64, beta_rad: f64, r_init_m: f64) -> Self {
        let r0 = clamp_f64(r_init_m, R_FLOOR_M, R_CEIL_M);
        let rho0 = 1.0 / r0;
        let x = [alpha_rad, beta_rad, 0.0, 0.0, rho0, 0.0];
        let mut p = [[0.0f64; 6]; 6];
        p[0][0] = 0.002 * 0.002;
        p[1][1] = 0.002 * 0.002;
        p[2][2] = 0.0025 * 0.0025;
        p[3][3] = 0.0025 * 0.0025;
        let sig_rho = 0.28 * rho0;
        p[4][4] = sig_rho * sig_rho;
        p[5][5] = 0.15 * 0.15;
        Self { x, p }
    }

    /// Estimate range `R_m = clamp(1 / rho_m, R_FLOOR_M, R_CEIL_M)` and delta-method variance `Var(R_m)`.
    pub fn range_and_variance(&self) -> (f64, f64) {
        let rho = clamp_f64(self.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
        let r_m = clamp_f64(1.0 / rho, R_FLOOR_M, R_CEIL_M);
        let rho2 = rho * rho;
        let rho4 = rho2 * rho2;
        let raw_var = if self.p[4][4].is_finite() && self.p[4][4] >= 0.0 {
            self.p[4][4] / rho4
        } else {
            (0.28 * r_m) * (0.28 * r_m)
        };
        let max_var = (0.5 * r_m) * (0.5 * r_m);
        let var_r = clamp_f64(raw_var, 1.0, max_var);
        (r_m, var_r)
    }
}

/// Pure algebraic simplex normalization and mixture moment kernel (`Kani #15`).
///
/// Given strictly non-negative bounded unnormalized likelihood weights `unnorm[0..6]` (with at
/// least one positive entry) and bank ranges/variances, projects weights onto the simplex
/// (`w_m >= W_FLOOR`, `sum(w_m) == 1`) and computes mixture mean `R_hat` and total variance `var_tot`.
pub fn normalize_weights_and_var_from_unnorm(
    unnorm: &[f64; NUM_BANKS],
    r_banks: &[f64; NUM_BANKS],
    var_r_banks: &[f64; NUM_BANKS],
) -> ([f64; NUM_BANKS], f64, f64) {
    let mut sum_unnorm = 0.0f64;
    let mut m = 0usize;
    while m < NUM_BANKS {
        let u = if unnorm[m].is_finite() && unnorm[m] > 0.0 {
            unnorm[m]
        } else {
            0.0
        };
        sum_unnorm += u;
        m += 1;
    }

    let scale = 1.0 - (NUM_BANKS as f64) * W_FLOOR;
    let mut w_out = [1.0 / (NUM_BANKS as f64); NUM_BANKS];
    if sum_unnorm.is_finite() && sum_unnorm > 0.0 {
        let mut m = 0usize;
        while m < NUM_BANKS {
            let u = if unnorm[m].is_finite() && unnorm[m] > 0.0 {
                unnorm[m]
            } else {
                0.0
            };
            w_out[m] = scale * (u / sum_unnorm) + W_FLOOR;
            m += 1;
        }
    }

    let mut r_clean = [0.0f64; NUM_BANKS];
    let mut v_clean = [0.0f64; NUM_BANKS];
    let mut r_hat_raw = 0.0f64;
    let mut m = 0usize;
    while m < NUM_BANKS {
        let rm = if r_banks[m].is_finite() {
            clamp_f64(r_banks[m], R_FLOOR_M, R_CEIL_M)
        } else {
            R_INIT_M[m]
        };
        let vm = if var_r_banks[m].is_finite() && var_r_banks[m] >= 0.0 {
            clamp_f64(var_r_banks[m], 0.0, 1.0e8)
        } else {
            (0.28 * rm) * (0.28 * rm)
        };
        r_clean[m] = rm;
        v_clean[m] = vm;
        r_hat_raw += w_out[m] * rm;
        m += 1;
    }

    let r_hat = clamp_f64(r_hat_raw, R_FLOOR_M, R_CEIL_M);
    let mut var_tot = 0.0f64;
    let mut m = 0usize;
    while m < NUM_BANKS {
        let dr = r_clean[m] - r_hat;
        var_tot += w_out[m] * (v_clean[m] + dr * dr);
        m += 1;
    }
    let var_clean = if var_tot.is_finite() && var_tot > 0.0 {
        var_tot
    } else {
        0.0
    };
    (w_out, r_hat, var_clean)
}

/// Pure Kani-verified algebraic kernel normalizing `M = 6` hypothesis weights onto the simplex
/// and computing the Gaussian-mixture mean `R_hat` and total standard deviation `sigma_R`.
pub fn normalize_weights_and_moments(
    log_lik: &[f64; NUM_BANKS],
    r_banks: &[f64; NUM_BANKS],
    var_r_banks: &[f64; NUM_BANKS],
    prev_w: &[f64; NUM_BANKS],
) -> ([f64; NUM_BANKS], f64, f64) {
    // Shift raw log-likelihoods relative to best-fitting bank BEFORE clamping to [-25, 0]
    // so positive Gaussian log-densities (-0.5 * ln(det S) > 0) never clip all banks to 0.0.
    let mut max_raw_ll = -1.0e300_f64;
    let mut m = 0usize;
    while m < NUM_BANKS {
        if log_lik[m].is_finite() && log_lik[m] > max_raw_ll {
            max_raw_ll = log_lik[m];
        }
        m += 1;
    }

    let mut log_w = [0.0f64; NUM_BANKS];
    let mut max_log = -1.0e300_f64;

    let mut m = 0usize;
    while m < NUM_BANKS {
        let wp = if prev_w[m].is_finite() && prev_w[m] >= W_FLOOR {
            if prev_w[m] > 1.0 {
                1.0
            } else {
                prev_w[m]
            }
        } else {
            W_FLOOR
        };
        let ll = if log_lik[m].is_finite() && max_raw_ll > -1.0e299_f64 {
            clamp_f64(log_lik[m] - max_raw_ll, -25.0, 0.0)
        } else {
            -25.0
        };
        let lw = libm::log(wp) + ll;
        log_w[m] = lw;
        if lw > max_log {
            max_log = lw;
        }
        m += 1;
    }

    let mut unnorm = [0.0f64; NUM_BANKS];
    let mut m = 0usize;
    while m < NUM_BANKS {
        let diff = clamp_f64(log_w[m] - max_log, -50.0, 0.0);
        unnorm[m] = libm::exp(diff);
        m += 1;
    }

    let (w_out, r_hat, var_tot) =
        normalize_weights_and_var_from_unnorm(&unnorm, r_banks, var_r_banks);
    let sigma_r = if var_tot > 0.0 {
        libm::sqrt(var_tot)
    } else {
        0.0
    };

    (w_out, r_hat, sigma_r)
}

/// Project ownship acceleration `a_ned` onto the orthonormal LOS triad `(u_r, u_alpha, u_beta)`.
pub fn project_accel_to_los(a_ned: [f64; 3], alpha: f64, beta: f64) -> (f64, f64, f64) {
    let sa = libm::sin(alpha);
    let ca = libm::cos(alpha);
    let sb = libm::sin(beta);
    let cb = libm::cos(beta);

    let a_los = a_ned[0] * (cb * ca) + a_ned[1] * (cb * sa) + a_ned[2] * (-sb);
    let a_az = a_ned[0] * (-sa) + a_ned[1] * ca;
    let a_el = a_ned[0] * (-sb * ca) + a_ned[1] * (-sb * sa) + a_ned[2] * (-cb);
    (a_los, a_az, a_el)
}

fn msc_rhs(x: &[f64; 6], a_ned: [f64; 3]) -> [f64; 6] {
    let alpha = x[0];
    let beta = clamp_f64(x[1], -1.35, 1.35);
    let a_dot = x[2];
    let b_dot = x[3];
    let rho = clamp_f64(x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    let rho_dot = clamp_f64(x[5], -2.0, 2.0);

    let (a_los, a_az, a_el) = project_accel_to_los(a_ned, alpha, beta);
    let sb = libm::sin(beta);
    let cb = libm::cos(beta);
    let cb_safe = if cb > 0.15 { cb } else { 0.15 };
    let tb = sb / cb_safe;

    let a_ddot = -2.0 * a_dot * rho_dot + 2.0 * a_dot * b_dot * tb - (rho / cb_safe) * a_az;
    let b_ddot = -2.0 * b_dot * rho_dot - a_dot * a_dot * sb * cb - rho * a_el;
    let rho_d = -rho * rho_dot;
    let rho_dd = a_dot * a_dot * cb * cb + b_dot * b_dot - rho_dot * rho_dot - rho * a_los;

    [a_dot, b_dot, a_ddot, b_ddot, rho_d, rho_dd]
}

fn msc_jacobian(x: &[f64; 6], a_ned: [f64; 3]) -> [[f64; 6]; 6] {
    let alpha = x[0];
    let beta = clamp_f64(x[1], -1.35, 1.35);
    let a_dot = x[2];
    let b_dot = x[3];
    let rho = clamp_f64(x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    let rho_dot = clamp_f64(x[5], -2.0, 2.0);

    let (a_los, a_az, a_el) = project_accel_to_los(a_ned, alpha, beta);
    let sb = libm::sin(beta);
    let cb = libm::cos(beta);
    let cb_safe = if cb > 0.15 { cb } else { 0.15 };
    let tb = sb / cb_safe;
    let sec2 = 1.0 / (cb_safe * cb_safe);

    let mut j = [[0.0f64; 6]; 6];
    // d(alpha)/dt = alpha_dot
    j[0][2] = 1.0;
    // d(beta)/dt = beta_dot
    j[1][3] = 1.0;
    // d(alpha_dot)/dt
    j[2][1] = 2.0 * a_dot * b_dot * sec2 - rho * a_az * sb * sec2;
    j[2][2] = -2.0 * rho_dot + 2.0 * b_dot * tb;
    j[2][3] = 2.0 * a_dot * tb;
    j[2][4] = -a_az / cb_safe;
    j[2][5] = -2.0 * a_dot;
    // d(beta_dot)/dt
    j[3][1] = -a_dot * a_dot * (cb * cb - sb * sb);
    j[3][2] = -2.0 * a_dot * sb * cb;
    j[3][3] = -2.0 * rho_dot;
    j[3][4] = -a_el;
    j[3][5] = -2.0 * b_dot;
    // d(rho)/dt = -rho * rho_dot
    j[4][4] = -rho_dot;
    j[4][5] = -rho;
    // d(rho_dot)/dt
    j[5][1] = -2.0 * a_dot * a_dot * cb * sb;
    j[5][2] = 2.0 * a_dot * cb * cb;
    j[5][3] = 2.0 * b_dot;
    j[5][4] = -a_los;
    j[5][5] = -2.0 * rho_dot;

    j
}

fn mat6_mul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0f64; 6]; 6];
    let mut i = 0usize;
    while i < 6 {
        let mut k = 0usize;
        while k < 6 {
            let aik = a[i][k];
            if aik != 0.0 {
                let mut j = 0usize;
                while j < 6 {
                    c[i][j] += aik * b[k][j];
                    j += 1;
                }
            }
            k += 1;
        }
        i += 1;
    }
    c
}

fn mat6_mul_bt(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0f64; 6]; 6];
    let mut i = 0usize;
    while i < 6 {
        let mut j = 0usize;
        while j < 6 {
            let mut acc = 0.0f64;
            let mut k = 0usize;
            while k < 6 {
                acc += a[i][k] * b[j][k];
                k += 1;
            }
            c[i][j] = acc;
            j += 1;
        }
        i += 1;
    }
    c
}

/// Propagate one MSC bank forward by `dt_s` using RK2 state integration and 2nd-order `F = I + dt*J + 0.5*dt^2*J^2`.
pub fn propagate_bank(bank: &mut MscBank, a_ned: [f64; 3], dt_s: f64) {
    let dt = clamp_f64(dt_s, 1.0e-3, 0.5);
    let k1 = msc_rhs(&bank.x, a_ned);
    let mut x_mid = [0.0f64; 6];
    let mut i = 0usize;
    while i < 6 {
        x_mid[i] = bank.x[i] + 0.5 * dt * k1[i];
        i += 1;
    }
    x_mid[0] = wrap_pi(x_mid[0]);
    x_mid[1] = clamp_f64(x_mid[1], -1.35, 1.35);
    x_mid[4] = clamp_f64(x_mid[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    x_mid[5] = clamp_f64(x_mid[5], -2.0, 2.0);

    let k2 = msc_rhs(&x_mid, a_ned);
    let mut x_next = [0.0f64; 6];
    let mut i = 0usize;
    while i < 6 {
        x_next[i] = bank.x[i] + dt * k2[i];
        i += 1;
    }
    x_next[0] = wrap_pi(x_next[0]);
    x_next[1] = clamp_f64(x_next[1], -1.35, 1.35);
    x_next[4] = clamp_f64(x_next[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    x_next[5] = clamp_f64(x_next[5], -2.0, 2.0);

    let j = msc_jacobian(&x_mid, a_ned);
    let j2 = mat6_mul(&j, &j);
    let mut f = [[0.0f64; 6]; 6];
    let half_dt2 = 0.5 * dt * dt;
    let mut r = 0usize;
    while r < 6 {
        let mut c = 0usize;
        while c < 6 {
            f[r][c] = (if r == c { 1.0 } else { 0.0 }) + dt * j[r][c] + half_dt2 * j2[r][c];
            c += 1;
        }
        r += 1;
    }

    let fp = mat6_mul(&f, &bank.p);
    let mut p_next = mat6_mul_bt(&fp, &f);

    // Physical process noise Q(dt) (constant-velocity civil cruise intruder model)
    let q_ang_acc = 0.0003 * 0.0003;
    let q_ang_pos = 0.25 * q_ang_acc * dt * dt * dt * dt + 1.0e-10 * dt;
    let q_ang_vel = q_ang_acc * dt * dt + 1.0e-9 * dt;
    let q_rho = (1.0e-6 * dt) * (1.0e-6 * dt);
    let q_rhodot = (0.01 * dt) * (0.01 * dt) + 1.0e-6 * dt;

    p_next[0][0] += q_ang_pos;
    p_next[1][1] += q_ang_pos;
    p_next[2][2] += q_ang_vel;
    p_next[3][3] += q_ang_vel;
    p_next[4][4] += q_rho;
    p_next[5][5] += q_rhodot;

    // Symmetrize & clamp diagonals
    let mut r = 0usize;
    while r < 6 {
        let mut c = r;
        while c < 6 {
            let avg = 0.5 * (p_next[r][c] + p_next[c][r]);
            let clean = if avg.is_finite() { avg } else { 0.0 };
            p_next[r][c] = clean;
            p_next[c][r] = clean;
            c += 1;
        }
        if p_next[r][r] < 1.0e-14 {
            p_next[r][r] = 1.0e-14;
        }
        r += 1;
    }

    bank.x = x_next;
    bank.p = p_next;
}

/// Bearing measurement update `(alpha, beta)` returning log-likelihood `log L_m`.
pub fn update_bank_bearing(
    bank: &mut MscBank,
    alpha_meas: f64,
    beta_meas: f64,
    sigma_bearing_rad: f64,
) -> f64 {
    let r_var = sigma_bearing_rad * sigma_bearing_rad;
    let nu0 = wrap_pi(alpha_meas - bank.x[0]);
    let nu1 = clamp_f64(beta_meas, -1.35, 1.35) - bank.x[1];

    let s00 = bank.p[0][0] + r_var;
    let s01 = bank.p[0][1];
    let s11 = bank.p[1][1] + r_var;
    let det_s = s00 * s11 - s01 * s01;
    if !det_s.is_finite() || det_s <= 1.0e-20 {
        return -25.0;
    }
    let inv_det = 1.0 / det_s;
    let inv00 = s11 * inv_det;
    let inv01 = -s01 * inv_det;
    let inv11 = s00 * inv_det;

    let maha = nu0 * (inv00 * nu0 + inv01 * nu1) + nu1 * (inv01 * nu0 + inv11 * nu1);
    let log_lik = -0.5 * (maha + libm::log(det_s));

    // Kalman gain K = P[:, 0..2] * S^-1  (6x2)
    let mut k0 = [0.0f64; 6];
    let mut k1 = [0.0f64; 6];
    let mut i = 0usize;
    while i < 6 {
        let pi0 = bank.p[i][0];
        let pi1 = bank.p[i][1];
        k0[i] = pi0 * inv00 + pi1 * inv01;
        k1[i] = pi0 * inv01 + pi1 * inv11;
        i += 1;
    }

    let mut i = 0usize;
    while i < 6 {
        bank.x[i] += k0[i] * nu0 + k1[i] * nu1;
        i += 1;
    }
    bank.x[0] = wrap_pi(bank.x[0]);
    bank.x[1] = clamp_f64(bank.x[1], -1.35, 1.35);
    bank.x[4] = clamp_f64(bank.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    bank.x[5] = clamp_f64(bank.x[5], -2.0, 2.0);

    // Joseph form: P+ = (I - K H) P (I - K H)^T + K R K^T
    let mut ikh = [[0.0f64; 6]; 6];
    let mut r = 0usize;
    while r < 6 {
        ikh[r][r] = 1.0;
        ikh[r][0] -= k0[r];
        ikh[r][1] -= k1[r];
        r += 1;
    }
    let tmp = mat6_mul(&ikh, &bank.p);
    let mut p_new = mat6_mul_bt(&tmp, &ikh);
    let mut r = 0usize;
    while r < 6 {
        let mut c = r;
        while c < 6 {
            let krk = r_var * (k0[r] * k0[c] + k1[r] * k1[c]);
            let val = 0.5 * (p_new[r][c] + p_new[c][r]) + krk;
            let clean = if val.is_finite() { val } else { 0.0 };
            p_new[r][c] = clean;
            p_new[c][r] = clean;
            c += 1;
        }
        if p_new[r][r] < 1.0e-14 {
            p_new[r][r] = 1.0e-14;
        }
        r += 1;
    }
    bank.p = p_new;

    if log_lik.is_finite() {
        log_lik
    } else {
        -1.0e6
    }
}

/// Scalar optical expansion update `z_tau = -s_dot / s` directly observing `x[5] = rho_dot`.
pub fn update_bank_tau(bank: &mut MscBank, z_tau: f64, r_tau: f64) {
    let nu = z_tau - bank.x[5];
    let s = bank.p[5][5] + r_tau;
    if !s.is_finite() || s <= 1.0e-16 {
        return;
    }
    let inv_s = 1.0 / s;
    let mut k = [0.0f64; 6];
    let mut i = 0usize;
    while i < 6 {
        k[i] = bank.p[i][5] * inv_s;
        bank.x[i] += k[i] * nu;
        i += 1;
    }
    bank.x[0] = wrap_pi(bank.x[0]);
    bank.x[1] = clamp_f64(bank.x[1], -1.35, 1.35);
    bank.x[4] = clamp_f64(bank.x[4], 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M);
    bank.x[5] = clamp_f64(bank.x[5], -2.0, 2.0);

    // Joseph scalar update
    let mut ikh = [[0.0f64; 6]; 6];
    let mut r = 0usize;
    while r < 6 {
        ikh[r][r] = 1.0;
        ikh[r][5] -= k[r];
        r += 1;
    }
    let tmp = mat6_mul(&ikh, &bank.p);
    let mut p_new = mat6_mul_bt(&tmp, &ikh);
    let mut r = 0usize;
    while r < 6 {
        let mut c = r;
        while c < 6 {
            let val = 0.5 * (p_new[r][c] + p_new[c][r]) + r_tau * k[r] * k[c];
            let clean = if val.is_finite() { val } else { 0.0 };
            p_new[r][c] = clean;
            p_new[c][r] = clean;
            c += 1;
        }
        if p_new[r][r] < 1.0e-14 {
            p_new[r][r] = 1.0e-14;
        }
        r += 1;
    }
    bank.p = p_new;
}

/// Output of one full `M = 6` `MSC-RPEKF` step.
#[derive(Debug, Clone, Copy)]
pub struct MscStepSummary {
    pub r_hat_m: f64,
    pub sigma_r_m: f64,
    pub rho_dot_inv_s: f64,
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub alpha_dot: f64,
    pub beta_dot: f64,
    pub active_bank_idx: u8,
    pub flags: u16,
    pub r_banks: [f64; NUM_BANKS],
    pub weights: [f64; NUM_BANKS],
}

/// Full `M = 6` MSC-RPEKF bank filter state.
#[derive(Debug, Clone, Copy)]
pub struct MscRpekfBank {
    pub initialized: bool,
    pub banks: [MscBank; NUM_BANKS],
    pub weights: [f64; NUM_BANKS],
}

impl MscRpekfBank {
    pub fn new() -> Self {
        let mut banks = [MscBank::init(0.0, 0.0, R_INIT_M[0]); NUM_BANKS];
        let mut m = 0usize;
        while m < NUM_BANKS {
            banks[m] = MscBank::init(0.0, 0.0, R_INIT_M[m]);
            m += 1;
        }
        Self {
            initialized: false,
            banks,
            weights: [1.0 / (NUM_BANKS as f64); NUM_BANKS],
        }
    }

    pub fn update(
        &mut self,
        cfg: &ActiveRangeConfig,
        own: &OwnshipStateNed,
        meas: &ActiveRangeMeasurement,
    ) -> MscStepSummary {
        let alpha_m = wrap_pi(meas.alpha_rad);
        let beta_m = clamp_f64(meas.beta_rad, -1.35, 1.35);
        let dt_s = clamp_f64(meas.dt_s, 1.0e-3, 0.5);

        let mut log_lik = [0.0f64; NUM_BANKS];
        let mut flags: u16 = 0;

        if !self.initialized {
            let mut m = 0usize;
            while m < NUM_BANKS {
                self.banks[m] = MscBank::init(alpha_m, beta_m, cfg.r_init_m[m]);
                self.weights[m] = 1.0 / (NUM_BANKS as f64);
                m += 1;
            }
            self.initialized = true;
        } else {
            let mut m = 0usize;
            while m < NUM_BANKS {
                propagate_bank(&mut self.banks[m], own.a_ned, dt_s);
                log_lik[m] = update_bank_bearing(
                    &mut self.banks[m],
                    alpha_m,
                    beta_m,
                    cfg.sigma_bearing_rad,
                );
                m += 1;
            }
            // Shift raw Gaussian log-densities relative to the best-fitting bank before clamping to [-25, 0]
            let mut ll_max = log_lik[0];
            let mut m = 1usize;
            while m < NUM_BANKS {
                if log_lik[m] > ll_max {
                    ll_max = log_lik[m];
                }
                m += 1;
            }
            let mut m = 0usize;
            while m < NUM_BANKS {
                log_lik[m] = clamp_f64(log_lik[m] - ll_max, -25.0, 0.0);
                m += 1;
            }
        }

        // Scale-cancelling optical expansion `tau`: z_tau = -s_dot / s
        if meas.size_px.is_finite()
            && meas.size_rate_px_s.is_finite()
            && meas.size_px >= TAU_MIN_SIZE_PX
            && meas.track_age >= TAU_MIN_TRACK_AGE
        {
            let z_tau = -meas.size_rate_px_s / meas.size_px;
            if z_tau.is_finite() && libm::fabs(z_tau) <= 1.0 {
                let s2 = meas.size_px * meas.size_px;
                let r_tau = (2.0 * 0.64) / (s2 * 0.25) + 0.015 * 0.015;
                let mut m = 0usize;
                while m < NUM_BANKS {
                    update_bank_tau(&mut self.banks[m], z_tau, r_tau);
                    m += 1;
                }
                flags |= FLAG_TAU_FUSED;
            }
        }

        // Gated weak Koschmieder atmospheric contrast prior (s < 8 px, daytime clear, above horizon, no active parallax)
        let (_a_los, a_az, _a_el) = project_accel_to_los(own.a_ned, alpha_m, beta_m);
        if meas.daytime_clear
            && meas.size_px.is_finite()
            && meas.size_px < KOSCH_MAX_SIZE_PX
            && beta_m > 0.0
            && meas.sky_beta_ext.is_finite()
            && meas.sky_beta_ext >= 5.0e-5
            && meas.sky_beta_ext <= 2.0e-3
            && meas.contrast_ratio.is_finite()
            && meas.contrast_ratio > 0.01
            && meas.contrast_ratio < 0.99
            && libm::fabs(a_az) < 0.25
        {
            let max_delta = KOSCH_MAX_LOG_DELTA * dt_s;
            let mut m = 0usize;
            while m < NUM_BANKS {
                let (r_m, _) = self.banks[m].range_and_variance();
                let c_pred = libm::exp(-meas.sky_beta_ext * r_m);
                if meas.contrast_ratio > c_pred {
                    let diff = (meas.contrast_ratio - c_pred) / 0.25;
                    let delta = clamp_f64(-0.5 * diff * diff * dt_s, -max_delta, 0.0);
                    log_lik[m] += delta;
                }
                m += 1;
            }
            flags |= FLAG_KOSCH_FUSED;
        }

        let mut r_banks = [0.0f64; NUM_BANKS];
        let mut var_r_banks = [0.0f64; NUM_BANKS];
        let mut m = 0usize;
        while m < NUM_BANKS {
            let (rm, vm) = self.banks[m].range_and_variance();
            r_banks[m] = rm;
            var_r_banks[m] = vm;
            m += 1;
        }

        let (w_new, r_hat_m, sigma_r_m) =
            normalize_weights_and_moments(&log_lik, &r_banks, &var_r_banks, &self.weights);
        self.weights = w_new;

        let mut active_idx = 0usize;
        let mut max_w = -1.0f64;
        let mut rho_dot_mix = 0.0f64;
        let mut a_dot_mix = 0.0f64;
        let mut b_dot_mix = 0.0f64;
        let mut alpha_mix_s = 0.0f64;
        let mut alpha_mix_c = 0.0f64;
        let mut beta_mix = 0.0f64;

        let mut m = 0usize;
        while m < NUM_BANKS {
            let wm = self.weights[m];
            if wm > max_w {
                max_w = wm;
                active_idx = m;
            }
            rho_dot_mix += wm * self.banks[m].x[5];
            a_dot_mix += wm * self.banks[m].x[2];
            b_dot_mix += wm * self.banks[m].x[3];
            alpha_mix_s += wm * libm::sin(self.banks[m].x[0]);
            alpha_mix_c += wm * libm::cos(self.banks[m].x[0]);
            beta_mix += wm * self.banks[m].x[1];
            m += 1;
        }

        MscStepSummary {
            r_hat_m,
            sigma_r_m,
            rho_dot_inv_s: clamp_f64(rho_dot_mix, -2.0, 2.0),
            alpha_rad: libm::atan2(alpha_mix_s, alpha_mix_c),
            beta_rad: clamp_f64(beta_mix, -1.35, 1.35),
            alpha_dot: a_dot_mix,
            beta_dot: b_dot_mix,
            active_bank_idx: active_idx as u8,
            flags,
            r_banks,
            weights: self.weights,
        }
    }
}
