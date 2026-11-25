//! Dual-Bank 15-State `SO(3)` Error-State EKF (`ES-EKF`), Exact Augmented Stochastic Cloning,
//! cross-covariance `P_01`, and Bonferroni 3-Layer GNSS Spoofing Guard (`NIS`, 8-arm Cholesky
//! Page's CUSUM with `h_c = 10.530`, and Dual-Bank Solution Separation) with frozen
//! split-conformal maneuver guard.

use crate::det_f64::{
    fabs, mat3_cholesky_lower, mat3_inv, mat3_mul, mat3_mul_vec, mat3_solve_lower,
    max_preserve_nan, Mat3, Vec3,
};
use crate::geo::{euler_to_r, so3_exp, so3_log};
use xvs_rta_core::quantize::fail_closed_gt;

pub const CUSUM_THRESHOLD: f64 = 10.530;
pub const CUSUM_DRIFT_K: f64 = 0.75;
pub const NIS_BONFERRONI_THR: f64 = 20.5;
pub const SS_BONFERRONI_THR: f64 = 20.5;
/// `0.0084 deg` 360-deg horizon sensor 1-sigma accuracy in radians (~146.6 urad).
pub const HORIZON_SIGMA_RAD: f64 = 0.0084 * core::f64::consts::PI / 180.0;

const INV_SQRT3: f64 = 0.577_350_269_189_625_8;
const CUSUM_ARMS: [Vec3; 8] = [
    [INV_SQRT3, INV_SQRT3, INV_SQRT3],
    [INV_SQRT3, INV_SQRT3, -INV_SQRT3],
    [INV_SQRT3, -INV_SQRT3, INV_SQRT3],
    [INV_SQRT3, -INV_SQRT3, -INV_SQRT3],
    [-INV_SQRT3, INV_SQRT3, INV_SQRT3],
    [-INV_SQRT3, INV_SQRT3, -INV_SQRT3],
    [-INV_SQRT3, -INV_SQRT3, INV_SQRT3],
    [-INV_SQRT3, -INV_SQRT3, -INV_SQRT3],
];

type Mat15 = [[f64; 15]; 15];

fn mat15_diag(d: [f64; 15]) -> Mat15 {
    let mut m = [[0.0; 15]; 15];
    for i in 0..15 {
        m[i][i] = d[i];
    }
    m
}

fn propagate_15x15_kinematic(p: &mut Mat15, dt: f64, q_diag: &[f64; 15]) {
    // Error-state transition F = I + A*dt where dp_dot = dv, dv_dot = -R[a]_x dtheta - R db_a, dtheta_dot = -db_g.
    // Fast exact structure update for F P F^T + Q without allocating temporary 15x15 products:
    let mut fp = *p;
    for i in 0..3 {
        for j in 0..15 {
            fp[i][j] = p[i][j] + dt * p[i + 3][j];
            fp[i + 3][j] = p[i + 3][j] - dt * p[i + 9][j];
            fp[i + 6][j] = p[i + 6][j] - dt * p[i + 12][j];
        }
    }
    let mut next = fp;
    for i in 0..15 {
        for j in 0..3 {
            next[i][j] = fp[i][j] + dt * fp[i][j + 3];
            next[i][j + 3] = fp[i][j + 3] - dt * fp[i][j + 9];
            next[i][j + 6] = fp[i][j + 6] - dt * fp[i][j + 12];
        }
        next[i][i] += q_diag[i] * dt;
    }
    *p = next;
}

fn symmetrize_15(p: &mut Mat15) {
    for i in 0..15 {
        for j in (i + 1)..15 {
            let avg = 0.5 * (p[i][j] + p[j][i]);
            p[i][j] = avg;
            p[j][i] = avg;
        }
    }
}

fn extract_block3(p: &Mat15, row0: usize, col0: usize) -> Mat3 {
    [
        [p[row0][col0], p[row0][col0 + 1], p[row0][col0 + 2]],
        [
            p[row0 + 1][col0],
            p[row0 + 1][col0 + 1],
            p[row0 + 1][col0 + 2],
        ],
        [
            p[row0 + 2][col0],
            p[row0 + 2][col0 + 1],
            p[row0 + 2][col0 + 2],
        ],
    ]
}

#[derive(Debug, Clone, Copy)]
pub struct BankState {
    pub p_ned: Vec3,
    pub v_ned: Vec3,
    pub r_ned_body: Mat3,
    pub b_a: Vec3,
    pub b_g: Vec3,
    pub cov: Mat15,
}

impl BankState {
    pub fn new(p_ned: Vec3, v_ned: Vec3) -> Self {
        let cov = mat15_diag([
            1.0, 1.0, 1.0, // pos (m^2)
            0.09, 0.09, 0.09, // vel (m^2/s^2)
            1e-6, 1e-6, 1e-5, // att (rad^2)
            1e-4, 1e-4, 1e-4, // accel bias
            1e-8, 1e-8, 1e-8, // gyro bias
        ]);
        Self {
            p_ned,
            v_ned,
            r_ned_body: euler_to_r(0.0, 0.0, 0.0),
            b_a: [0.0; 3],
            b_g: [0.0; 3],
            cov,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpoofCheckInput {
    pub gnss_pos_ned: Vec3,
    pub gnss_vel_ned: Vec3,
    pub bank_rad: f64,
    pub omega_norm_radps: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SpoofCheckOutput {
    pub nis_stat: f64,
    pub cusum_stat: f64,
    pub ss_stat: f64,
    pub maneuver_guard_scale: f64,
    pub spoof_alarm: bool,
    pub active_bank: u8,
}

/// Dual-Bank 15-state `SO(3)` `ES-EKF` with `Bank 0` (GNSS+IMU+Horizon),
/// `Bank 1` (permanently GNSS-free IMU+Horizon+Stochastic-Cloned VisFix),
/// exact `15x15` cross-covariance `P_01`, and Bonferroni 3-layer spoofing detector.
#[derive(Debug, Clone, Copy)]
pub struct DualBankEsEkf {
    bank0: BankState,
    bank1: BankState,
    p_01: Mat15,
    // Augmented stochastic clone snapshot for delayed VisFix on Bank 1
    clone_active: bool,
    clone_pos_ned: Vec3,
    clone_p_sc: [[f64; 3]; 15],
    clone_p_cc: Mat3,
    // 8-arm Cholesky Page's CUSUM accumulators
    cusum_arms: [f64; 8],
    active_bank: u8,
    spoof_latched: bool,
}

impl DualBankEsEkf {
    pub fn new(p_ned: Vec3, v_ned: Vec3) -> Self {
        let b = BankState::new(p_ned, v_ned);
        Self {
            bank0: b,
            bank1: b,
            p_01: b.cov,
            clone_active: false,
            clone_pos_ned: p_ned,
            clone_p_sc: [[0.0; 3]; 15],
            clone_p_cc: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            cusum_arms: [0.0; 8],
            active_bank: 0,
            spoof_latched: false,
        }
    }

    pub fn bank0_pos_ned(&self) -> Vec3 {
        self.bank0.p_ned
    }

    pub fn bank0_vel_ned(&self) -> Vec3 {
        self.bank0.v_ned
    }

    pub fn bank1_pos_ned(&self) -> Vec3 {
        self.bank1.p_ned
    }

    pub fn bank1_vel_ned(&self) -> Vec3 {
        self.bank1.v_ned
    }

    pub fn active_bank(&self) -> u8 {
        self.active_bank
    }

    pub fn active_pos_ned(&self) -> Vec3 {
        if self.active_bank == 0 {
            self.bank0.p_ned
        } else {
            self.bank1.p_ned
        }
    }

    pub fn active_vel_ned(&self) -> Vec3 {
        if self.active_bank == 0 {
            self.bank0.v_ned
        } else {
            self.bank1.v_ned
        }
    }

    pub fn cross_cov_p01(&self) -> &Mat15 {
        &self.p_01
    }

    /// Propagate BOTH Bank 0 and Bank 1 using shared IMU acceleration/gyro and fuse the
    /// 360-deg horizon attitude measurement (`0.0084 deg` 1-sigma).
    pub fn step_imu_horizon(
        &mut self,
        dt: f64,
        accel_ned: Vec3,
        omega_body: Vec3,
        yaw_meas: f64,
        pitch_meas: f64,
        roll_meas: f64,
    ) {
        let q_diag = [
            0.01, 0.01, 0.01, // pos process noise
            0.04, 0.04, 0.04, // vel process noise
            1e-7, 1e-7, 1e-7, // attitude gyro noise
            1e-6, 1e-6, 1e-6, // accel bias RW
            1e-9, 1e-9, 1e-9, // gyro bias RW
        ];

        for bank in [&mut self.bank0, &mut self.bank1] {
            for k in 0..3 {
                let a_corr = accel_ned[k] - bank.b_a[k];
                bank.p_ned[k] += bank.v_ned[k] * dt + 0.5 * a_corr * dt * dt;
                bank.v_ned[k] += a_corr * dt;
            }
            let dtheta = [
                (omega_body[0] - bank.b_g[0]) * dt,
                (omega_body[1] - bank.b_g[1]) * dt,
                (omega_body[2] - bank.b_g[2]) * dt,
            ];
            bank.r_ned_body = mat3_mul(&bank.r_ned_body, &so3_exp(dtheta));
            propagate_15x15_kinematic(&mut bank.cov, dt, &q_diag);
            symmetrize_15(&mut bank.cov);

            // Fuse 360-deg horizon attitude update on SO(3) (0.0084 deg sigma)
            let r_meas = euler_to_r(yaw_meas, pitch_meas, roll_meas);
            let r_t = crate::det_f64::mat3_transpose(&bank.r_ned_body);
            let err_r = mat3_mul(&r_t, &r_meas);
            let err_vec = so3_log(&err_r);
            let r_var = HORIZON_SIGMA_RAD * HORIZON_SIGMA_RAD;
            let mut corr = [0.0; 3];
            for k in 0..3 {
                let p_ii = bank.cov[6 + k][6 + k];
                let gain = p_ii / (p_ii + r_var);
                corr[k] = gain * err_vec[k];
                bank.cov[6 + k][6 + k] = (1.0 - gain) * p_ii;
            }
            bank.r_ned_body = mat3_mul(&bank.r_ned_body, &so3_exp(corr));
        }

        // Propagate shared cross-covariance P_01 (shared IMU process noise Q)
        propagate_15x15_kinematic(&mut self.p_01, dt, &q_diag);

        // Propagate stochastic clone cross-covariance P_sc with F_1
        if self.clone_active {
            for i in 0..3 {
                for j in 0..3 {
                    self.clone_p_sc[i][j] += dt * self.clone_p_sc[i + 3][j];
                    self.clone_p_sc[i + 3][j] -= dt * self.clone_p_sc[i + 9][j];
                }
            }
        }
    }

    /// Capture an augmented stochastic clone of `Bank 1` pose for delayed Vision Fix (`VisFix`).
    pub fn capture_stochastic_clone(&mut self) {
        self.clone_active = true;
        self.clone_pos_ned = self.bank1.p_ned;
        for i in 0..15 {
            for j in 0..3 {
                self.clone_p_sc[i][j] = self.bank1.cov[i][j];
            }
        }
        self.clone_p_cc = extract_block3(&self.bank1.cov, 0, 0);
    }

    /// Apply delayed Vision Fix measurement to `Bank 1` via Exact Augmented Stochastic Cloning
    /// (`P_sc * (P_cc + R)^{-1}`), leaving `Bank 0` untouched and updating `P_01` consistently.
    pub fn update_visfix_cloning(&mut self, vis_pos_ned: Vec3, sigma_m: f64) {
        if !self.clone_active {
            return;
        }
        let r_var = (sigma_m * sigma_m).max(1e-4);
        let mut s = self.clone_p_cc;
        for (k, row) in s.iter_mut().enumerate() {
            row[k] += r_var;
        }
        let s_inv = match mat3_inv(&s) {
            Some(inv) => inv,
            None => return,
        };
        let innov = [
            vis_pos_ned[0] - self.clone_pos_ned[0],
            vis_pos_ned[1] - self.clone_pos_ned[1],
            vis_pos_ned[2] - self.clone_pos_ned[2],
        ];

        // Kalman gain K_clone (15x3) = P_sc @ S^{-1}
        let mut k_gain = [[0.0; 3]; 15];
        for i in 0..15 {
            for j in 0..3 {
                k_gain[i][j] = self.clone_p_sc[i][0] * s_inv[0][j]
                    + self.clone_p_sc[i][1] * s_inv[1][j]
                    + self.clone_p_sc[i][2] * s_inv[2][j];
            }
        }

        // Update Bank 1 state (pos, vel, accel bias)
        for k in 0..3 {
            let dp = k_gain[k][0] * innov[0] + k_gain[k][1] * innov[1] + k_gain[k][2] * innov[2];
            let dv = k_gain[k + 3][0] * innov[0]
                + k_gain[k + 3][1] * innov[1]
                + k_gain[k + 3][2] * innov[2];
            self.bank1.p_ned[k] += dp;
            self.bank1.v_ned[k] += dv;
        }

        // Update Bank 1 covariance: P_1 <- P_1 - K_clone @ P_sc^T
        for i in 0..15 {
            for j in 0..15 {
                let sub = k_gain[i][0] * self.clone_p_sc[j][0]
                    + k_gain[i][1] * self.clone_p_sc[j][1]
                    + k_gain[i][2] * self.clone_p_sc[j][2];
                self.bank1.cov[i][j] -= sub;
            }
            if self.bank1.cov[i][i] < 1e-9 {
                self.bank1.cov[i][i] = 1e-9;
            }
        }
        symmetrize_15(&mut self.bank1.cov);

        // Update P_01 consistently: P_01 <- P_01 * (I - K_clone H_clone)^T
        let old_p01 = self.p_01;
        for i in 0..15 {
            for j in 0..15 {
                let corr = old_p01[i][0] * k_gain[j][0]
                    + old_p01[i][1] * k_gain[j][1]
                    + old_p01[i][2] * k_gain[j][2];
                self.p_01[i][j] -= corr;
            }
        }
        self.clone_active = false;
    }

    /// Run Bonferroni 3-layer GNSS spoofing check (`NIS ∨ 8-arm Cholesky Page's CUSUM ∨ Dual-Bank SS`)
    /// with frozen split-conformal maneuver guard, and update `Bank 0` ONLY while unspoofed.
    /// `Bank 1` NEVER ingests GNSS measurements.
    ///
    /// Fail-closed non-finite guard (`D16-M1-B1`): if ANY input or intermediate statistic is
    /// `NaN`/`±Inf`, Bank 0 assimilation is skipped (never contaminating `bank0.v_ned`), Bank 1
    /// is selected, and `f64::NAN` is preserved outward so `xvs-rta-core::build_rta_input_from_f64`
    /// trips `quantize_fault = true` -> `INVALID_INHIBIT (P0)` at 0 ticks.
    pub fn update_gnss_and_check_spoof(&mut self, inp: &SpoofCheckInput) -> SpoofCheckOutput {
        let inputs_finite = inp.gnss_pos_ned.iter().all(|v| v.is_finite())
            && inp.gnss_vel_ned.iter().all(|v| v.is_finite())
            && inp.bank_rad.is_finite()
            && inp.omega_norm_radps.is_finite();
        if !inputs_finite {
            self.spoof_latched = true;
            self.active_bank = 1;
            return SpoofCheckOutput {
                nis_stat: f64::NAN,
                cusum_stat: f64::NAN,
                ss_stat: f64::NAN,
                maneuver_guard_scale: f64::NAN,
                spoof_alarm: true,
                active_bank: 1,
            };
        }

        // Frozen split-conformal maneuver guard calibrated for coordinated up to 30 deg bank turns
        let bank_norm = fabs(inp.bank_rad) / (30.0 * core::f64::consts::PI / 180.0);
        let maneuver_guard_scale =
            1.0 + 4.5 * bank_norm * bank_norm + 16.0 * inp.omega_norm_radps * inp.omega_norm_radps;

        let innov_v = [
            inp.gnss_vel_ned[0] - self.bank0.v_ned[0],
            inp.gnss_vel_ned[1] - self.bank0.v_ned[1],
            inp.gnss_vel_ned[2] - self.bank0.v_ned[2],
        ];
        let r_gnss_v = 0.09; // (0.30 m/s)^2 nominal GNSS velocity noise
        let p0_vv = extract_block3(&self.bank0.cov, 3, 3);
        let mut s_v = p0_vv;
        for (k, row) in s_v.iter_mut().enumerate() {
            row[k] = (row[k] + r_gnss_v) * maneuver_guard_scale;
            if row[k] < 0.04 {
                row[k] = 0.04;
            }
        }

        // Layer 1: NIS (3-DoF velocity Mahalanobis squared) — NaN-preserving
        let nis_stat = match mat3_inv(&s_v) {
            Some(s_inv) => {
                let sv_r = mat3_mul_vec(&s_inv, innov_v);
                max_preserve_nan(
                    innov_v[0] * sv_r[0] + innov_v[1] * sv_r[1] + innov_v[2] * sv_r[2],
                    0.0,
                )
            }
            None => f64::NAN,
        };

        // Layer 2: 8-arm Cholesky Page's CUSUM (h_c = 10.530) — NaN-preserving
        let z_white = match mat3_cholesky_lower(&s_v) {
            Some(l) => mat3_solve_lower(&l, innov_v),
            None => [f64::NAN, f64::NAN, f64::NAN],
        };
        let mut max_cusum = 0.0_f64;
        for (arm_idx, dir) in CUSUM_ARMS.iter().enumerate() {
            let proj = dir[0] * z_white[0] + dir[1] * z_white[1] + dir[2] * z_white[2];
            let next_g = max_preserve_nan(self.cusum_arms[arm_idx] + proj - CUSUM_DRIFT_K, 0.0);
            if next_g.is_finite() {
                self.cusum_arms[arm_idx] = next_g;
            }
            max_cusum = max_preserve_nan(max_cusum, next_g);
        }

        // Apply GNSS update to Bank 0 ONLY while unspoofed (NEVER Bank 1); once spoofed, reject
        // spoofed GNSS assimilation so CUSUM stays elevated throughout the active spoofing attack.
        let nis_pre_alarm = fail_closed_gt(nis_stat, NIS_BONFERRONI_THR).0;
        let cusum_pre_alarm = fail_closed_gt(max_cusum, CUSUM_THRESHOLD).0;
        if !self.spoof_latched && !nis_pre_alarm && !cusum_pre_alarm {
            let mut s_raw = p0_vv;
            for (k, row) in s_raw.iter_mut().enumerate() {
                row[k] += r_gnss_v;
            }
            if let Some(s_raw_inv) = mat3_inv(&s_raw) {
                let mut k_v = [[0.0; 3]; 15];
                for i in 0..15 {
                    for j in 0..3 {
                        k_v[i][j] = self.bank0.cov[i][3] * s_raw_inv[0][j]
                            + self.bank0.cov[i][4] * s_raw_inv[1][j]
                            + self.bank0.cov[i][5] * s_raw_inv[2][j];
                    }
                }
                for k in 0..3 {
                    let dv = k_v[k + 3][0] * innov_v[0]
                        + k_v[k + 3][1] * innov_v[1]
                        + k_v[k + 3][2] * innov_v[2];
                    self.bank0.v_ned[k] += dv;
                    // Mild GNSS position pull on Bank 0
                    self.bank0.p_ned[k] += 0.15 * (inp.gnss_pos_ned[k] - self.bank0.p_ned[k]);
                }
                let old_p0 = self.bank0.cov;
                for i in 0..15 {
                    for j in 0..15 {
                        let sub = k_v[i][0] * old_p0[3][j]
                            + k_v[i][1] * old_p0[4][j]
                            + k_v[i][2] * old_p0[5][j];
                        self.bank0.cov[i][j] -= sub;
                    }
                    if self.bank0.cov[i][i] < 1e-9 {
                        self.bank0.cov[i][i] = 1e-9;
                    }
                }
                symmetrize_15(&mut self.bank0.cov);

                // P_01 <- (I - K_0 H_0) P_01
                let old_p01 = self.p_01;
                for i in 0..15 {
                    for j in 0..15 {
                        let sub = k_v[i][0] * old_p01[3][j]
                            + k_v[i][1] * old_p01[4][j]
                            + k_v[i][2] * old_p01[5][j];
                        self.p_01[i][j] -= sub;
                    }
                }
            }
        }

        // Layer 3: Dual-Bank Solution Separation (Bank 0 vs permanently GNSS-free Bank 1)
        let dv_ss = [
            self.bank0.v_ned[0] - self.bank1.v_ned[0],
            self.bank0.v_ned[1] - self.bank1.v_ned[1],
            self.bank0.v_ned[2] - self.bank1.v_ned[2],
        ];
        let p0_v = extract_block3(&self.bank0.cov, 3, 3);
        let p1_v = extract_block3(&self.bank1.cov, 3, 3);
        let p01_v = extract_block3(&self.p_01, 3, 3);
        let mut sigma_ss = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                sigma_ss[i][j] = p0_v[i][j] + p1_v[i][j] - p01_v[i][j] - p01_v[j][i];
            }
            let floor = 0.36 * maneuver_guard_scale;
            sigma_ss[i][i] = (sigma_ss[i][i] * maneuver_guard_scale).max(floor);
        }
        let ss_stat = match mat3_inv(&sigma_ss) {
            Some(inv) => {
                let tmp = mat3_mul_vec(&inv, dv_ss);
                max_preserve_nan(dv_ss[0] * tmp[0] + dv_ss[1] * tmp[1] + dv_ss[2] * tmp[2], 0.0)
            }
            None => f64::NAN,
        };

        let tripped = fail_closed_gt(nis_stat, NIS_BONFERRONI_THR).0
            || fail_closed_gt(max_cusum, CUSUM_THRESHOLD).0
            || fail_closed_gt(ss_stat, SS_BONFERRONI_THR).0;
        if tripped {
            self.spoof_latched = true;
            self.active_bank = 1;
        }

        SpoofCheckOutput {
            nis_stat,
            cusum_stat: max_cusum,
            ss_stat,
            maneuver_guard_scale,
            spoof_alarm: self.spoof_latched,
            active_bank: self.active_bank,
        }
    }
}
