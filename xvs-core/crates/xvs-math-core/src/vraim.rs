//! Multi-Camera Vision-RAIM (`V-RAIM`):
//! - 6-cam pairwise overlap epipolar residual & 360-deg horizon leave-one-out isolation (`F,R,B,L,U,D`)
//! - Runway keypoint Jacobian worst-case failure-mode slopes `g_max,i` -> `V-HPL` / `V-VPL`
//! - KSFO `28L` (`61 m`) vs `Taxiway C` (`23 m`) parallel width consistency gate.

use crate::det_f64::{fabs, hypot, mat3_inv, max_preserve_nan, min_preserve_nan, sqrt};
use crate::rig::XVS6_CUBE_V1;

/// KSFO Runway 28L nominal width in metres (`61.0 m` = 200 ft).
pub const KSFO_28L_WIDTH_M: f64 = 61.0;
/// KSFO parallel Taxiway C nominal width in metres (`23.0 m` = 75 ft).
pub const KSFO_TAXIWAY_C_WIDTH_M: f64 = 23.0;
/// Runway width gate tolerance (`±12.0 m`, cleanly separating 61 m runway from 23 m taxiway).
pub const RUNWAY_WIDTH_TOL_M: f64 = 12.0;

/// Single-camera epipolar fault threshold (`6109 urad` ≈ `0.35 deg`, matching `v2_m1_odd.yaml`).
pub const EPIPOLAR_FAULT_RAD: f64 = 0.006_109;

/// All 12 adjacent face overlap edges of the `XVS6-CUBE-v1` rig (`F=0, R=1, B=2, L=3, U=4, D=5`).
const CUBE_OVERLAP_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (0, 3),
    (0, 4),
    (0, 5),
    (2, 1),
    (2, 3),
    (2, 4),
    (2, 5),
    (1, 4),
    (1, 5),
    (3, 4),
    (3, 5),
];

#[derive(Debug, Clone, Copy)]
pub struct HorizonIsolationResult {
    pub epipolar_max_rad: [f64; 6],
    pub isolated_cam_idx: Option<usize>,
    pub healthy_cam_mask: u8,
    pub fused_attitude_err_rad: f64,
    pub horizon_pl_rad: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RunwayVraimResult {
    pub v_vpl_m: f64,
    pub v_hpl_m: f64,
    pub max_slope_v: f64,
    pub max_slope_h: f64,
    pub parity_stat: f64,
    pub width_delta_m: f64,
    pub width_fault: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct VisionRaimEvaluator {
    /// Per-camera horizon sensor 1-sigma noise (rad).
    pub sigma_cam_rad: f64,
    /// Runway keypoint pixel measurement 1-sigma noise (px).
    pub sigma_kp_px: f64,
    /// Integrity risk inflation factor `K_md` (10^-7 integrity allocation).
    pub k_md: f64,
    /// Nominal parity detection threshold `T_det` (normalized chi-square sqrt).
    pub t_det: f64,
}

impl Default for VisionRaimEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionRaimEvaluator {
    pub const fn new() -> Self {
        Self {
            sigma_cam_rad: 0.000_146_6, // 0.0084 deg
            sigma_kp_px: 0.25,          // sub-pixel runway edge-intersection accuracy
            k_md: 5.33,
            t_det: 3.5,
        }
    }

    /// Evaluate 6-cam pairwise overlap epipolar residuals and leave-one-out 360-deg horizon
    /// isolation across `F(0), R(1), B(2), L(3), U(4), D(5)`.
    pub fn evaluate_6cam_horizon_epipolar(
        &self,
        cam_biases_rad: &[f64; 6],
        input_health_mask: u8,
    ) -> HorizonIsolationResult {
        let mut mask = input_health_mask & 0x3F;

        // Fail-closed non-finite camera telemetry guard (D16-M1-B1): never swallow NaN/Inf.
        if cam_biases_rad.iter().any(|b| !b.is_finite()) {
            let mut epipolar_max_rad = [0.0; 6];
            let mut isolated_cam_idx = None;
            for (i, &b) in cam_biases_rad.iter().enumerate() {
                if !b.is_finite() {
                    epipolar_max_rad[i] = f64::NAN;
                    mask &= !(1 << i);
                    if isolated_cam_idx.is_none() {
                        isolated_cam_idx = Some(i);
                    }
                }
            }
            return HorizonIsolationResult {
                epipolar_max_rad,
                isolated_cam_idx,
                healthy_cam_mask: mask,
                fused_attitude_err_rad: f64::NAN,
                horizon_pl_rad: f64::NAN,
            };
        }

        // Compute leave-one-out consensus mean and RMS disagreement for each candidate excluded cam
        let mut best_excl: Option<usize> = None;
        let mut best_loo_rms = f64::INFINITY;
        let mut full_rms = 0.0;
        let mut full_edges = 0usize;

        for &(i, j) in &CUBE_OVERLAP_EDGES {
            if ((mask >> i) & 1) == 1 && ((mask >> j) & 1) == 1 {
                let d = cam_biases_rad[i] - cam_biases_rad[j];
                full_rms += d * d;
                full_edges += 1;
            }
        }
        full_rms = if full_edges > 0 {
            sqrt(full_rms / (full_edges as f64))
        } else {
            0.0
        };

        if full_rms > EPIPOLAR_FAULT_RAD * 0.25 {
            for excl in 0..6 {
                if ((mask >> excl) & 1) == 0 {
                    continue;
                }
                let mut sum_sq = 0.0;
                let mut cnt = 0usize;
                for &(i, j) in &CUBE_OVERLAP_EDGES {
                    if i == excl || j == excl {
                        continue;
                    }
                    if ((mask >> i) & 1) == 1 && ((mask >> j) & 1) == 1 {
                        let d = cam_biases_rad[i] - cam_biases_rad[j];
                        sum_sq += d * d;
                        cnt += 1;
                    }
                }
                let rms = if cnt > 0 {
                    sqrt(sum_sq / (cnt as f64))
                } else {
                    f64::INFINITY
                };
                if rms < best_loo_rms {
                    best_loo_rms = rms;
                    best_excl = Some(excl);
                }
            }
        }

        // Compute consensus attitude from the best subset (or full set if no single fault dominates)
        let mut consensus_mask = mask;
        if let Some(excl) = best_excl {
            if best_loo_rms < full_rms * 0.35 {
                consensus_mask &= !(1 << excl);
            }
        }
        let mut cons_sum = 0.0;
        let mut cons_cnt = 0usize;
        for (i, &b) in cam_biases_rad.iter().enumerate() {
            if ((consensus_mask >> i) & 1) == 1 {
                cons_sum += b;
                cons_cnt += 1;
            }
        }
        let cons_mean = if cons_cnt > 0 {
            cons_sum / (cons_cnt as f64)
        } else {
            0.0
        };

        // Per-camera epipolar discrepancy relative to healthy overlap neighbors
        let mut epipolar_max_rad = [0.0; 6];
        let mut isolated_cam_idx = None;
        for i in 0..6 {
            let mut diffs = [0.0; 4];
            let mut n_adj = 0usize;
            for &(a, b) in &CUBE_OVERLAP_EDGES {
                let other = if a == i {
                    b
                } else if b == i {
                    a
                } else {
                    continue;
                };
                if ((consensus_mask >> other) & 1) == 1 && n_adj < 4 {
                    diffs[n_adj] = fabs(cam_biases_rad[i] - cam_biases_rad[other]);
                    n_adj += 1;
                }
            }
            // Conservative max residual against consensus healthy neighbors
            let mut max_d = fabs(cam_biases_rad[i] - cons_mean);
            for d in diffs.iter().take(n_adj) {
                if *d > max_d {
                    max_d = *d;
                }
            }
            epipolar_max_rad[i] = max_d;
            if max_d > EPIPOLAR_FAULT_RAD && ((mask >> i) & 1) == 1 {
                mask &= !(1 << i);
                if isolated_cam_idx.is_none() {
                    isolated_cam_idx = Some(i);
                }
            }
        }

        // Fused attitude residual over remaining healthy cameras
        let mut rem_sum = 0.0;
        let mut rem_cnt = 0usize;
        for (i, &b) in cam_biases_rad.iter().enumerate() {
            if ((mask >> i) & 1) == 1 {
                rem_sum += b;
                rem_cnt += 1;
            }
        }
        let fused_attitude_err_rad = if rem_cnt > 0 {
            fabs(rem_sum / (rem_cnt as f64))
        } else {
            0.05
        };

        // Horizon Protection Level (PL) scales inversely with sqrt(rem_cnt)
        let horizon_pl_rad = if rem_cnt >= 4 {
            let sigma_fused = self.sigma_cam_rad / sqrt(rem_cnt as f64);
            let loo_term = if best_loo_rms.is_finite() {
                min_preserve_nan(best_loo_rms, 0.001)
            } else {
                0.0
            };
            self.k_md * sigma_fused + loo_term + 0.0008
        } else {
            0.025 // > 8727 urad AL when < 4 cams healthy
        };

        HorizonIsolationResult {
            epipolar_max_rad,
            isolated_cam_idx,
            healthy_cam_mask: mask,
            fused_attitude_err_rad,
            horizon_pl_rad,
        }
    }

    /// Evaluate Runway Vision-RAIM `V-HPL` / `V-VPL` using worst-case failure-mode slopes `g_max,i`
    /// over the 4-corner (`8x3`) runway keypoint Jacobian + KSFO `28L` (`61 m`) width gate.
    pub fn evaluate_runway_approach(
        &self,
        corner_bias_px: f64,
        observed_width_m: f64,
        slant_range_m: f64,
    ) -> RunwayVraimResult {
        // Fail-closed non-finite runway telemetry guard (D16-M1-B1): propagate NaN to RTA quantizer.
        if !corner_bias_px.is_finite()
            || !observed_width_m.is_finite()
            || !slant_range_m.is_finite()
        {
            return RunwayVraimResult {
                v_vpl_m: f64::NAN,
                v_hpl_m: f64::NAN,
                max_slope_v: f64::NAN,
                max_slope_h: f64::NAN,
                parity_stat: f64::NAN,
                width_delta_m: if observed_width_m.is_finite() {
                    observed_width_m - KSFO_28L_WIDTH_M
                } else {
                    f64::NAN
                },
                width_fault: true,
            };
        }

        let range = slant_range_m.clamp(300.0, 6000.0);
        // Forward EFVS camera (`XVS6_CUBE_V1`, fx = fy = 429.619 px/rad)
        let fx = XVS6_CUBE_V1.intrinsics.fx;
        let fy = XVS6_CUBE_V1.intrinsics.fy;
        let half_w = KSFO_28L_WIDTH_M * 0.5;
        let rwy_len = 1600.0; // visible runway keypoint depth baseline
        let alt_z = range * 0.052_407_78; // tan(3 deg) glideslope height

        // 4 runway keypoints in forward camera frame: [TL, TR, FL, FR]
        // (x_cam = right/East, y_cam = down/Down, z_cam = forward/North)
        let kps = [
            [-half_w, alt_z, range],
            [half_w, alt_z, range],
            [-half_w, alt_z, range + rwy_len],
            [half_w, alt_z, range + rwy_len],
        ];

        // Build 8x3 Jacobian H mapping aircraft position error [dN, dE, dD] -> pixel residual [du, dv]
        // Relative point in cam frame moves by [-dE, -dD, -dN] when aircraft moves by [dN, dE, dD].
        let mut h = [[0.0; 3]; 8];
        for (idx, kp) in kps.iter().enumerate() {
            let x = kp[0];
            let y = kp[1];
            let z = kp[2];
            let inv_z = 1.0 / z;
            let inv_z2 = inv_z * inv_z;
            // u = fx * x / z + cx  => du/dN = +fx * x / z^2, du/dE = -fx / z, du/dD = 0
            h[2 * idx][0] = fx * x * inv_z2;
            h[2 * idx][1] = -fx * inv_z;
            h[2 * idx][2] = 0.0;
            // v = fy * y / z + cy  => dv/dN = +fy * y / z^2, dv/dE = 0, dv/dD = -fy / z
            h[2 * idx + 1][0] = fy * y * inv_z2;
            h[2 * idx + 1][1] = 0.0;
            h[2 * idx + 1][2] = -fy * inv_z;
        }

        let w_inv = 1.0 / (self.sigma_kp_px * self.sigma_kp_px);
        // Normal matrix A = H^T W H (3x3) plus weak INS along-track prior (2.5 m 1-sigma)
        let mut a = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let mut sum = 0.0;
                for row in &h {
                    sum += row[r] * row[c] * w_inv;
                }
                a[r][c] = sum;
            }
            a[r][r] += 1e-6;
        }
        a[0][0] += 1.0 / (2.5 * 2.5);

        let cov_pos = mat3_inv(&a).unwrap_or([
            [25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
            [0.0, 0.0, 25.0],
        ]);

        // Weighted least-squares estimator S = cov_pos @ H^T @ W (3x8)
        let mut s_mat = [[0.0; 8]; 3];
        for r in 0..3 {
            for i in 0..8 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += cov_pos[r][k] * h[i][k] * w_inv;
                }
                s_mat[r][i] = sum;
            }
        }

        // Parity matrix P = I_8 - H @ S (8x8) -> diagonal P_ii gives fault observability
        let mut max_slope_v = 0.0_f64;
        let mut max_slope_h = 0.0_f64;
        for i in 0..8 {
            let hs_ii = h[i][0] * s_mat[0][i] + h[i][1] * s_mat[1][i] + h[i][2] * s_mat[2][i];
            let p_ii = (1.0 - hs_ii).max(1e-4);
            let denom = sqrt(p_ii) / self.sigma_kp_px;
            let gv = fabs(s_mat[2][i]) / denom.max(1e-9);
            let gh = hypot(s_mat[0][i], s_mat[1][i]) / denom.max(1e-9);
            if gv > max_slope_v {
                max_slope_v = gv;
            }
            if gh > max_slope_h {
                max_slope_h = gh;
            }
        }

        // Inject observed threshold corner bias on keypoint TL vertical pixel (index 1)
        let mut y_meas = [0.0; 8];
        y_meas[1] = corner_bias_px;
        let mut x_hat = [0.0; 3];
        for r in 0..3 {
            for i in 0..8 {
                x_hat[r] += s_mat[r][i] * y_meas[i];
            }
        }
        let mut sse = 0.0;
        for i in 0..8 {
            let pred = h[i][0] * x_hat[0] + h[i][1] * x_hat[1] + h[i][2] * x_hat[2];
            let resid = y_meas[i] - pred;
            sse += resid * resid * w_inv;
        }
        let parity_stat = sqrt(max_preserve_nan(sse, 0.0));

        let sigma_v = sqrt(max_preserve_nan(cov_pos[2][2], 0.0));
        let sigma_h = sqrt(max_preserve_nan(cov_pos[0][0] + cov_pos[1][1], 0.0));

        // Conservative Vision-RAIM V-VPL and V-HPL (m) — NaN-preserving
        let fault_scale = max_preserve_nan(parity_stat, self.t_det * 0.25);
        let v_vpl_m = self.k_md * sigma_v + max_slope_v * fault_scale;
        let v_hpl_m = self.k_md * sigma_h + max_slope_h * fault_scale;

        let width_delta_m = observed_width_m - KSFO_28L_WIDTH_M;
        let width_fault =
            !observed_width_m.is_finite() || fabs(width_delta_m) > RUNWAY_WIDTH_TOL_M;

        RunwayVraimResult {
            v_vpl_m,
            v_hpl_m,
            max_slope_v,
            max_slope_h,
            parity_stat,
            width_delta_m,
            width_fault,
        }
    }
}
