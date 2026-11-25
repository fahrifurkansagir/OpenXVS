//! Classical `D0` matched-filter fallback, passive `WC1` Well-Clear alerting (DO-365B),
//! and Slack-Augmented 3D Fixed-Wing `ECBF-QP` (`<= 16` iterations, zero alloc) with
//! deterministic max-rate escape fallback (`ecbf_infeasible = true`).

use crate::det_f64::{atan2, clamp, cos, fabs, hypot, sin, tan, Vec3};
use crate::flight6dof::{GRAVITY_MPS2, MAX_BANK_RAD, MAX_CLIMB_MPS, MIN_CLIMB_MPS};

pub const MAX_INTRUDERS: usize = 8;
pub const MAX_QP_ITERS: usize = 16;

/// RTCA DO-365B Loss of Well-Clear (LoWC) horizontal & vertical thresholds.
pub const LOWC_HMD_M: f64 = 1219.2; // 4000 ft
pub const LOWC_ZTHR_M: f64 = 137.16; // 450 ft
pub const LOWC_TAU_MOD_S: f64 = 35.0;

/// NMAC thresholds.
pub const NMAC_H_M: f64 = 152.4; // 500 ft
pub const NMAC_V_M: f64 = 30.48; // 100 ft

/// Slack penalty `rho = 1e4` for always-feasible slack-augmented ECBF-QP.
const RHO_SLACK: f64 = 1.0e4;
/// Critical slack threshold above which hard safety margin is compromised -> escape fallback.
const DELTA_CRIT: f64 = 0.04;

#[derive(Debug, Clone, Copy, Default)]
pub struct IntruderStateNed {
    pub pos_ned: Vec3,
    pub vel_ned: Vec3,
    pub valid: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Wc1AlertStatus {
    pub alert_active: bool,
    pub min_tau_mod_s: f64,
    pub min_h_dist_m: f64,
    pub min_v_dist_m: f64,
    pub closest_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct EcbfQpSolution {
    pub bank_cmd_rad: f64,
    pub climb_rate_cmd_mps: f64,
    pub slack: f64,
    pub iterations: u8,
    pub ecbf_infeasible: bool,
    pub wc1_alert: bool,
    pub min_ttc_s: f64,
}

/// Classical `D0` matched-filter fallback SNR metric (deterministic, zero ML dependency).
#[inline]
pub fn d0_classical_matched_filter_snr(contrast: f64, range_m: f64) -> f64 {
    if !contrast.is_finite() || !range_m.is_finite() || range_m <= 1.0 {
        return 0.0;
    }
    let r_km = (range_m * 1e-3).max(0.1);
    (contrast.max(0.0) * 18.0) / (r_km * r_km).max(0.25)
}

/// Passive `WC1` Well-Clear alerting (DO-365B modified tau / HMD / ZTHR).
pub fn evaluate_wc1_alerts(
    p_own: Vec3,
    v_own: Vec3,
    intruders: &[IntruderStateNed],
) -> Wc1AlertStatus {
    let mut alert_active = false;
    let mut min_tau = f64::INFINITY;
    let mut min_h = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    let mut closest_idx = None;

    for (idx, intr) in intruders.iter().enumerate() {
        if !intr.valid {
            continue;
        }
        let rx = p_own[0] - intr.pos_ned[0];
        let ry = p_own[1] - intr.pos_ned[1];
        let rz = p_own[2] - intr.pos_ned[2];
        let vx = v_own[0] - intr.vel_ned[0];
        let vy = v_own[1] - intr.vel_ned[1];

        let d_h = hypot(rx, ry);
        let d_v = fabs(rz);
        if d_h < min_h {
            min_h = d_h;
            min_v = d_v;
            closest_idx = Some(idx);
        }

        let r_dot_v = rx * vx + ry * vy;
        let tau_mod = if d_h <= LOWC_HMD_M {
            0.0
        } else if r_dot_v < -1e-3 {
            -(d_h * d_h - LOWC_HMD_M * LOWC_HMD_M) / r_dot_v
        } else {
            f64::INFINITY
        };

        if tau_mod < min_tau {
            min_tau = tau_mod;
        }
        if tau_mod <= LOWC_TAU_MOD_S && d_v <= LOWC_ZTHR_M * 1.5 {
            alert_active = true;
        }
    }

    Wc1AlertStatus {
        alert_active,
        min_tau_mod_s: min_tau,
        min_h_dist_m: min_h,
        min_v_dist_m: min_v,
        closest_idx,
    }
}

/// Slack-Augmented 3D Fixed-Wing Exponential Control Barrier Function QP (`ECBF-QP`).
#[derive(Debug, Clone, Copy)]
pub struct EcbfQpSolver {
    /// Horizontal protective ellipsoid semi-axis (m), padded beyond DO-365B 4000 ft LoWC.
    pub r_h_m: f64,
    /// Vertical protective ellipsoid semi-axis (m), padded beyond DO-365B 450 ft LoWC.
    pub r_v_m: f64,
    /// ECBF gain `k0` (`h_ddot + k1*h_dot + k0*h >= -delta`).
    pub k0: f64,
    /// ECBF gain `k1`.
    pub k1: f64,
    /// Max vertical acceleration authority (m/s^2).
    pub max_vert_accel_mps2: f64,
}

impl Default for EcbfQpSolver {
    fn default() -> Self {
        Self {
            r_h_m: 1850.0,
            r_v_m: 240.0,
            k0: 0.012,
            k1: 0.28,
            max_vert_accel_mps2: 4.0,
        }
    }
}

impl EcbfQpSolver {
    /// Solve slack-augmented 3D fixed-wing ECBF-QP in `<= 16` bounded iterations (zero alloc).
    ///
    /// If simultaneous threats saturate `±30°` bank / climb bounds (`slack > DELTA_CRIT`),
    /// executes deterministic max-rate escape bank/climb away from minimum-TTC threat and
    /// sets `ecbf_infeasible = true`.
    pub fn solve(
        &self,
        p_own: Vec3,
        v_own: Vec3,
        nom_bank_rad: f64,
        nom_climb_mps: f64,
        intruders: &[IntruderStateNed],
    ) -> EcbfQpSolution {
        let wc1 = evaluate_wc1_alerts(p_own, v_own, intruders);
        let max_a_lat = GRAVITY_MPS2 * tan(MAX_BANK_RAD); // ~5.6618 m/s^2
        let max_a_vert = self.max_vert_accel_mps2;

        let hdg = atan2(v_own[1], v_own[0]);
        // Horizontal unit vector pointing right of current ground track (NED)
        let e_lat = [-sin(hdg), cos(hdg)];
        let cur_climb = -v_own[2];

        let u_nom = [
            clamp(GRAVITY_MPS2 * tan(clamp(nom_bank_rad, -MAX_BANK_RAD, MAX_BANK_RAD)), -max_a_lat, max_a_lat),
            clamp((nom_climb_mps - cur_climb) / 1.2, -max_a_vert, max_a_vert),
        ];

        // Build up to MAX_INTRUDERS linearized second-order ECBF constraints: a_i · u + delta >= b_i
        let mut a_con = [[0.0; 2]; MAX_INTRUDERS];
        let mut b_con = [0.0; MAX_INTRUDERS];
        let mut n_con = 0usize;

        let mut min_ttc = f64::INFINITY;
        let mut min_ttc_rel = [1.0, 0.0, 0.0];

        let inv_rh2 = 1.0 / (self.r_h_m * self.r_h_m);
        let inv_rv2 = 1.0 / (self.r_v_m * self.r_v_m);

        for intr in intruders.iter().take(MAX_INTRUDERS) {
            if !intr.valid {
                continue;
            }
            let r = [
                p_own[0] - intr.pos_ned[0],
                p_own[1] - intr.pos_ned[1],
                p_own[2] - intr.pos_ned[2],
            ];
            let v = [
                v_own[0] - intr.vel_ned[0],
                v_own[1] - intr.vel_ned[1],
                v_own[2] - intr.vel_ned[2],
            ];

            let r_norm2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let r_dot_v = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
            let v_norm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let ttc = if r_dot_v < -1e-3 && v_norm2 > 1e-3 {
                (-r_dot_v / v_norm2).max(0.01)
            } else if r_norm2 < self.r_h_m * self.r_h_m {
                0.5
            } else {
                f64::INFINITY
            };
            if ttc < min_ttc {
                min_ttc = ttc;
                min_ttc_rel = r;
            }

            let h = (r[0] * r[0] + r[1] * r[1]) * inv_rh2 + (r[2] * r[2]) * inv_rv2 - 1.0;
            let h_dot = 2.0 * (r[0] * v[0] + r[1] * v[1]) * inv_rh2 + 2.0 * r[2] * v[2] * inv_rv2;
            // Inside the protective barrier (h < 0), escalate restoring poles so simultaneous
            // adversarial threats expose true actuator infeasibility cleanly via slack > DELTA_CRIT.
            let (k0_eff, k1_eff) = if h < 0.0 {
                (0.65, 1.6)
            } else {
                (self.k0, self.k1)
            };
            let f_drift = 2.0 * (v[0] * v[0] + v[1] * v[1]) * inv_rh2
                + 2.0 * v[2] * v[2] * inv_rv2
                + k1_eff * h_dot
                + k0_eff * h;

            // Only include active or closing constraints within 25 km
            if r_norm2 > 25_000.0 * 25_000.0 || (h > 8.0 && h_dot >= 0.0) {
                continue;
            }

            // Lateral & vertical ECBF control gradients (CPA lookahead + right-of-way symmetry break)
            let t_look = if r_dot_v < 0.0 && ttc.is_finite() {
                clamp(ttc * 0.45, 0.0, 12.0)
            } else {
                0.0
            };
            let rx_eff = r[0] + t_look * v[0];
            let ry_eff = r[1] + t_look * v[1];
            let mut g_lat = 2.0 * (rx_eff * e_lat[0] + ry_eff * e_lat[1]) * inv_rh2;
            if fabs(g_lat) < 0.15 / self.r_h_m && h_dot < 0.0 {
                g_lat = 0.45 / self.r_h_m; // 14 CFR 91.113 right turn tie-breaker
            }
            // Note: climb_accel = -a_D => d(h_ddot)/d(u_vert) = -2 * r_D / R_v^2
            let rz_eff = r[2] + t_look * v[2];
            let mut g_vert = -2.0 * rz_eff * inv_rv2;
            if fabs(g_vert) < 0.08 / self.r_v_m && h < 3.0 && h_dot < 0.0 {
                g_vert = 0.35 / self.r_v_m; // climb separation preference when co-altitude
            }

            a_con[n_con] = [g_lat, g_vert];
            b_con[n_con] = -f_drift;
            n_con += 1;
        }

        if n_con == 0 {
            return EcbfQpSolution {
                bank_cmd_rad: clamp(nom_bank_rad, -MAX_BANK_RAD, MAX_BANK_RAD),
                climb_rate_cmd_mps: clamp(nom_climb_mps, MIN_CLIMB_MPS, MAX_CLIMB_MPS),
                slack: 0.0,
                iterations: 0,
                ecbf_infeasible: false,
                wc1_alert: wc1.alert_active,
                min_ttc_s: min_ttc,
            };
        }

        // Bounded projected dual / active-set solver (<= MAX_QP_ITERS = 16)
        let mut lambda = [0.0; MAX_INTRUDERS];
        let mut u = u_nom;
        let mut slack = 0.0_f64;
        let mut iters_used: u8 = 0;

        for iter in 0..MAX_QP_ITERS {
            iters_used = (iter + 1) as u8;
            let mut max_viol = 0.0_f64;

            for i in 0..n_con {
                let lhs = a_con[i][0] * u[0] + a_con[i][1] * u[1];
                let viol = b_con[i] - lhs;
                let a_norm2 = a_con[i][0] * a_con[i][0] + a_con[i][1] * a_con[i][1] + 1.0 / RHO_SLACK;
                let step = viol / a_norm2.max(1e-12);
                lambda[i] = (lambda[i] + step).max(0.0);
                if viol > max_viol {
                    max_viol = viol;
                }
            }

            // Primal update projected onto fixed-wing actuator box [-u_max, +u_max]
            let mut u_next = u_nom;
            for i in 0..n_con {
                u_next[0] += lambda[i] * a_con[i][0];
                u_next[1] += lambda[i] * a_con[i][1];
            }
            u[0] = clamp(u_next[0], -max_a_lat, max_a_lat);
            u[1] = clamp(u_next[1], -max_a_vert, max_a_vert);

            // Recompute actual slack required by box-clamped control
            slack = 0.0;
            for i in 0..n_con {
                let resid = b_con[i] - (a_con[i][0] * u[0] + a_con[i][1] * u[1]);
                if resid > slack {
                    slack = resid;
                }
            }

            if slack <= 1e-6 && max_viol <= 1e-6 {
                break;
            }
        }

        // Review Focus #3: Explicit infeasibility check & deterministic max-rate escape fallback
        if !u[0].is_finite() || !u[1].is_finite() || !slack.is_finite() || slack > DELTA_CRIT {
            // Escape deterministically away from minimum-TTC threat
            let lat_proj = min_ttc_rel[0] * e_lat[0] + min_ttc_rel[1] * e_lat[1];
            let esc_bank = if lat_proj < 0.0 {
                -MAX_BANK_RAD
            } else {
                MAX_BANK_RAD
            };
            // If ownship is above threat (r_D <= 0), climb max; else descend max
            let esc_climb = if min_ttc_rel[2] <= 0.0 {
                MAX_CLIMB_MPS
            } else {
                MIN_CLIMB_MPS
            };
            return EcbfQpSolution {
                bank_cmd_rad: esc_bank,
                climb_rate_cmd_mps: esc_climb,
                slack: if slack.is_finite() { slack } else { 1.0 },
                iterations: iters_used,
                ecbf_infeasible: true,
                wc1_alert: true,
                min_ttc_s: min_ttc,
            };
        }

        let bank_cmd_rad = clamp(atan2(u[0], GRAVITY_MPS2), -MAX_BANK_RAD, MAX_BANK_RAD);
        let climb_rate_cmd_mps = clamp(cur_climb + u[1] * 1.2, MIN_CLIMB_MPS, MAX_CLIMB_MPS);
        EcbfQpSolution {
            bank_cmd_rad,
            climb_rate_cmd_mps,
            slack,
            iterations: iters_used,
            ecbf_infeasible: false,
            wc1_alert: wc1.alert_active,
            min_ttc_s: min_ttc,
        }
    }
}
