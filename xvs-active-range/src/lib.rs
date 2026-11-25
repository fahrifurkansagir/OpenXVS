// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OpenXVS v3.0 `XVS-ActiveRange` standalone `#![no_std]` `#![forbid(unsafe_code)]` crate.
//!
//! Unifies:
//! 1. `M = 6` log-spaced Modified Spherical Coordinates Range-Parameterized EKF (`MSC-RPEKF`)
//! 2. Scale-cancelling optical expansion (`tau = s / s_dot`) + gated Koschmieder prior
//! 3. Distribution-free Split Conformal lower safe range bound (`CP-MSC`)
//! 4. Dimensionally exact (`m^2/s`) Universal-Range (`∀ h`) discrete-time `SC3BF-FIM` micro-swerve
//! 5. Exact 64-byte little-endian IEEE-802.3 CRC32 wire codec (`ActiveRangeTelemetry`)
//!
//! All coordinates strictly follow OpenXVS NED (`North, East, Down`). Zero internal ENU.

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

pub mod conformal;
#[cfg(kani)]
mod kani_proofs;
pub mod msc_rpekf;
pub mod sc3bf_fim;
pub mod wire;

pub use conformal::compute_conformal_lower_bound;
pub use msc_rpekf::{normalize_weights_and_moments, MscBank, MscRpekfBank, MscStepSummary};
pub use sc3bf_fim::{
    candidate_fov_safe, eval_sc3bf_h, eval_universal_sc3bf, select_active_swerve, LosTriadNed,
    SwerveDecision, SwerveInputs, NUM_PHI_CANDS, PHI_CAND_COS, PHI_CAND_DEG, PHI_CAND_RAD,
    PHI_CAND_SIN, PHI_CAND_TAN,
};
pub use wire::{
    decode_payload, decode_wire, encode_payload, encode_wire, ieee_crc32, ActiveRangeMode,
    ActiveRangeTelemetry, ACTIVE_RANGE_PAYLOAD_BYTES, ACTIVE_RANGE_WIRE_BYTES, FLAG_FOV_SAFE,
    FLAG_KOSCH_FUSED, FLAG_SC3BF_ALL_SAFE, FLAG_TAU_FUSED,
};

/// Locked numerical constants (Spec §4.1.1 / DR-20 §3).
pub const NUM_BANKS: usize = 6;
pub const R_INIT_M: [f64; NUM_BANKS] = [200.0, 348.2202, 606.2866, 1055.6063, 1837.9174, 3200.0];
pub const R_FLOOR_M: f64 = 30.0;
pub const R_CEIL_M: f64 = 6000.0;
pub const W_FLOOR: f64 = 1.0e-4;
pub const W_ACTIVE: f64 = 0.02;
pub const SIGMA_FLOOR_M: f64 = 1.0;

pub const G_MPS2: f64 = 9.80665;
pub const R_SEP_M: f64 = 152.4;
pub const EVADE_BANK_DEG: f64 = 25.0;
pub const EVADE_BANK_RAD: f64 = 0.4363323129985824;
pub const A_MAX_EVADE: f64 = 4.572915995845716;
pub const K_ALLOW: f64 = 0.5;
pub const ALPHA_CBF: f64 = 0.35;
pub const DT_LOOK_S: f64 = 1.0;

pub const R_EVADE_TRIGGER_M: f64 = 220.0;
pub const TAU_EVADE_TRIGGER_S: f64 = 4.5;

pub const FOV_MAX_DEG: f64 = 55.0;
pub const FOV_MARGIN_DEG: f64 = 8.0;
pub const FOV_LIMIT_RAD: f64 = (FOV_MAX_DEG - FOV_MARGIN_DEG) * (core::f64::consts::PI / 180.0);

pub const TAU_MIN_SIZE_PX: f64 = 6.0;
pub const TAU_MIN_TRACK_AGE: u32 = 5;
pub const KOSCH_MAX_SIZE_PX: f64 = 8.0;
pub const KOSCH_MAX_LOG_DELTA: f64 = 0.15;

pub const TRIGGER_REL_SIGMA: f64 = 0.10;
pub const LOCK_REL_SIGMA: f64 = 0.08;
pub const CLOSING_RHO_DOT_THRESH: f64 = -0.015;

#[inline]
pub fn clamp_f64(val: f64, lo: f64, hi: f64) -> f64 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

#[inline]
pub fn wrap_pi(angle_rad: f64) -> f64 {
    if !angle_rad.is_finite() {
        return 0.0;
    }
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut a = libm::fmod(angle_rad + core::f64::consts::PI, two_pi);
    if a < 0.0 {
        a += two_pi;
    }
    a - core::f64::consts::PI
}

/// Static configuration for `ActiveRangeController`.
#[derive(Debug, Clone, Copy)]
pub struct ActiveRangeConfig {
    pub r_init_m: [f64; NUM_BANKS],
    pub sigma_bearing_rad: f64,
    pub q_hat_conformal: f64,
}

impl Default for ActiveRangeConfig {
    fn default() -> Self {
        Self {
            r_init_m: R_INIT_M,
            sigma_bearing_rad: 0.00055,
            q_hat_conformal: 1.65,
        }
    }
}

/// Ownship kinematic state strictly in NED coordinates.
#[derive(Debug, Clone, Copy)]
pub struct OwnshipStateNed {
    pub v_ned: [f64; 3],
    pub a_ned: [f64; 3],
    pub heading_rad: f64,
    pub tas_mps: f64,
    pub v_stall_mps: f64,
}

/// Monocular detection measurement for one tracked intruder.
#[derive(Debug, Clone, Copy)]
pub struct ActiveRangeMeasurement {
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub size_px: f64,
    pub size_rate_px_s: f64,
    pub contrast_ratio: f64,
    pub sky_beta_ext: f64,
    pub daytime_clear: bool,
    pub track_age: u32,
    pub dt_s: f64,
}

/// Top-level stateful `XVS-ActiveRange` controller.
#[derive(Debug, Clone, Copy)]
pub struct ActiveRangeController {
    cfg: ActiveRangeConfig,
    filter: MscRpekfBank,
    seq: u32,
    swerve_ticks: u32,
    was_locked: bool,
    last_alpha_rad: f64,
    last_beta_rad: f64,
}

impl ActiveRangeController {
    pub fn new(cfg: ActiveRangeConfig) -> Self {
        Self {
            cfg,
            filter: MscRpekfBank::new(),
            seq: 0,
            swerve_ticks: 0,
            was_locked: false,
            last_alpha_rad: 0.0,
            last_beta_rad: 0.0,
        }
    }

    pub fn config(&self) -> ActiveRangeConfig {
        self.cfg
    }

    /// Execute one deterministic `XVS-ActiveRange` update and emit a CRC32-signed 64-byte telemetry packet.
    ///
    /// Fail-safe sanitization: non-finite (`NaN`/`Inf`) or `dt_s <= 0` inputs never panic and never
    /// leak `NaN`/`Inf` into weights or telemetry.
    pub fn step(
        &mut self,
        own: &OwnshipStateNed,
        meas: &ActiveRangeMeasurement,
    ) -> ActiveRangeTelemetry {
        self.seq = self.seq.wrapping_add(1);

        let own_safe = sanitize_ownship(own);
        let inputs_valid = meas.alpha_rad.is_finite()
            && meas.beta_rad.is_finite()
            && meas.dt_s.is_finite()
            && meas.dt_s > 0.0;

        if !inputs_valid {
            // Fail-safe hold on invalid measurement or dt <= 0: keep filter intact and return bounded finite telemetry
            let (r_hat_m, sigma_r_m, r_banks, active_idx, rho_dot) = self.current_moments();
            let r_safe_lower_m =
                compute_conformal_lower_bound(r_hat_m, sigma_r_m, self.cfg.q_hat_conformal);
            let mut weights_f32 = [0.0f32; NUM_BANKS];
            let mut m = 0usize;
            while m < NUM_BANKS {
                weights_f32[m] = self.filter.weights[m] as f32;
                m += 1;
            }
            let _ = r_banks;
            let mut pkt = ActiveRangeTelemetry {
                seq: self.seq,
                mode: ActiveRangeMode::NominalHold as u8,
                active_bank_idx: active_idx,
                flags: 0,
                r_hat_m: r_hat_m as f32,
                sigma_r_m: sigma_r_m as f32,
                r_safe_lower_m: r_safe_lower_m as f32,
                rho_dot_inv_s: rho_dot as f32,
                cmd_bank_rad: 0.0f32,
                min_sc3bf_margin: 0.0f32,
                fim_gain: 0.0f32,
                weights: weights_f32,
                crc32: 0,
            };
            let wire = encode_wire(&pkt);
            pkt.crc32 = u32::from_le_bytes([wire[60], wire[61], wire[62], wire[63]]);
            return pkt;
        }

        let summary = self.filter.update(&self.cfg, &own_safe, meas);
        self.last_alpha_rad = summary.alpha_rad;
        self.last_beta_rad = summary.beta_rad;

        let r_safe_lower_m = compute_conformal_lower_bound(
            summary.r_hat_m,
            summary.sigma_r_m,
            self.cfg.q_hat_conformal,
        );

        let triad = LosTriadNed::from_angles(summary.alpha_rad, summary.beta_rad);
        let sh = libm::sin(own_safe.heading_rad);
        let ch = libm::cos(own_safe.heading_rad);
        let own_fwd_ned = [ch, sh, 0.0];
        let own_right_ned = [-sh, ch, 0.0];
        let alpha_body_rad = wrap_pi(summary.alpha_rad - own_safe.heading_rad);

        // Allow CBDR ambiguity probe for small distant dots (s < 6 px) ONLY when bearing is steady
        // AND apparent size is strictly growing (never fires on zero-closure s_dot == 0 targets).
        let cbdr_probe_active = meas.track_age >= TAU_MIN_TRACK_AGE
            && meas.size_px.is_finite()
            && meas.size_px < TAU_MIN_SIZE_PX
            && meas.size_rate_px_s.is_finite()
            && meas.size_rate_px_s > 0.0
            && libm::fabs(summary.alpha_dot) < 0.004
            && libm::fabs(summary.beta_dot) < 0.004;

        let preferred_swerve_sign = if ((self.swerve_ticks / 12) % 2) == 0 {
            1.0
        } else {
            -1.0
        };

        let swerve_inp = SwerveInputs {
            r_banks: summary.r_banks,
            weights: summary.weights,
            r_hat_m: summary.r_hat_m,
            sigma_r_m: summary.sigma_r_m,
            r_safe_lower_m,
            rho_dot_inv_s: summary.rho_dot_inv_s,
            alpha_dot: summary.alpha_dot,
            beta_dot: summary.beta_dot,
            alpha_body_rad,
            triad,
            own_fwd_ned,
            own_right_ned,
            tas_mps: own_safe.tas_mps,
            v_stall_mps: own_safe.v_stall_mps,
            sigma_bearing_rad: self.cfg.sigma_bearing_rad,
            cbdr_probe_active,
            was_locked: self.was_locked,
            preferred_swerve_sign,
        };

        let decision = select_active_swerve(&swerve_inp);
        self.was_locked = decision.now_locked;
        if decision.mode == ActiveRangeMode::ActiveSwerve {
            self.swerve_ticks = self.swerve_ticks.wrapping_add(1);
        } else {
            self.swerve_ticks = 0;
        }

        let mut flags = summary.flags;
        if decision.sc3bf_all_safe {
            flags |= FLAG_SC3BF_ALL_SAFE;
        }
        if decision.fov_safe {
            flags |= FLAG_FOV_SAFE;
        }

        let mut weights_f32 = [0.0f32; NUM_BANKS];
        let mut m = 0usize;
        while m < NUM_BANKS {
            weights_f32[m] = summary.weights[m] as f32;
            m += 1;
        }

        let mut pkt = ActiveRangeTelemetry {
            seq: self.seq,
            mode: decision.mode as u8,
            active_bank_idx: summary.active_bank_idx,
            flags,
            r_hat_m: summary.r_hat_m as f32,
            sigma_r_m: summary.sigma_r_m as f32,
            r_safe_lower_m: r_safe_lower_m as f32,
            rho_dot_inv_s: summary.rho_dot_inv_s as f32,
            cmd_bank_rad: decision.cmd_bank_rad as f32,
            min_sc3bf_margin: decision.min_sc3bf_margin as f32,
            fim_gain: decision.fim_gain as f32,
            weights: weights_f32,
            crc32: 0,
        };
        let wire = encode_wire(&pkt);
        pkt.crc32 = u32::from_le_bytes([wire[60], wire[61], wire[62], wire[63]]);
        pkt
    }

    fn current_moments(&self) -> (f64, f64, [f64; NUM_BANKS], u8, f64) {
        let mut r_banks = [0.0f64; NUM_BANKS];
        let mut v_banks = [0.0f64; NUM_BANKS];
        let mut r_hat = 0.0f64;
        let mut rho_dot = 0.0f64;
        let mut active_idx = 0usize;
        let mut max_w = -1.0f64;
        let mut m = 0usize;
        while m < NUM_BANKS {
            let (rm, vm) = self.filter.banks[m].range_and_variance();
            r_banks[m] = rm;
            v_banks[m] = vm;
            let wm = self.filter.weights[m];
            if wm > max_w {
                max_w = wm;
                active_idx = m;
            }
            r_hat += wm * rm;
            rho_dot += wm * self.filter.banks[m].x[5];
            m += 1;
        }
        let r_hat = clamp_f64(r_hat, R_FLOOR_M, R_CEIL_M);
        let mut var_tot = 0.0f64;
        let mut m = 0usize;
        while m < NUM_BANKS {
            let dr = r_banks[m] - r_hat;
            var_tot += self.filter.weights[m] * (v_banks[m] + dr * dr);
            m += 1;
        }
        let sigma_r = if var_tot.is_finite() && var_tot > 0.0 {
            libm::sqrt(var_tot)
        } else {
            0.0
        };
        (
            r_hat,
            sigma_r,
            r_banks,
            active_idx as u8,
            clamp_f64(rho_dot, -2.0, 2.0),
        )
    }
}

fn sanitize_ownship(own: &OwnshipStateNed) -> OwnshipStateNed {
    let clean = |v: f64, fallback: f64| -> f64 { if v.is_finite() { v } else { fallback } };
    OwnshipStateNed {
        v_ned: [
            clean(own.v_ned[0], 0.0),
            clean(own.v_ned[1], 0.0),
            clean(own.v_ned[2], 0.0),
        ],
        a_ned: [
            clamp_f64(clean(own.a_ned[0], 0.0), -25.0, 25.0),
            clamp_f64(clean(own.a_ned[1], 0.0), -25.0, 25.0),
            clamp_f64(clean(own.a_ned[2], 0.0), -25.0, 25.0),
        ],
        heading_rad: wrap_pi(clean(own.heading_rad, 0.0)),
        tas_mps: clamp_f64(clean(own.tas_mps, 25.0), 1.0, 250.0),
        v_stall_mps: clamp_f64(clean(own.v_stall_mps, 14.5), 1.0, 100.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_64b_roundtrip_and_crc_tamper() {
        assert_eq!(core::mem::size_of::<ActiveRangeTelemetry>(), 64);
        assert_eq!(ieee_crc32(b"123456789"), 0xCBF4_3926);

        let mut pkt = ActiveRangeTelemetry {
            seq: 42,
            mode: ActiveRangeMode::ActiveSwerve as u8,
            active_bank_idx: 3,
            flags: FLAG_TAU_FUSED | FLAG_SC3BF_ALL_SAFE | FLAG_FOV_SAFE,
            r_hat_m: 987.5,
            sigma_r_m: 41.25,
            r_safe_lower_m: 919.4,
            rho_dot_inv_s: -0.082,
            cmd_bank_rad: PHI_CAND_RAD[8] as f32,
            min_sc3bf_margin: 124.5,
            fim_gain: 18.75,
            weights: [0.0001, 0.0001, 0.0497, 0.9499, 0.0001, 0.0001],
            crc32: 0,
        };
        let wire = encode_wire(&pkt);
        assert_eq!(wire.len(), 64);
        pkt.crc32 = u32::from_le_bytes([wire[60], wire[61], wire[62], wire[63]]);

        let decoded = decode_wire(&wire).expect("valid wire frame must decode");
        assert_eq!(decoded, pkt);

        // Single-bit flip in EVERY byte position must be rejected
        for idx in 0..64 {
            let mut tampered = wire;
            tampered[idx] ^= 0x01;
            assert!(
                decode_wire(&tampered).is_none(),
                "bit flip at byte {idx} was not rejected"
            );
        }
    }

    #[test]
    fn test_cbdr_tick1_allows_active_swerve_and_collapses_range() {
        let cfg = ActiveRangeConfig::default();
        let mut ctrl = ActiveRangeController::new(cfg);

        // Pure head-on CBDR: ownship at origin flying North at 35 m/s, unknown 10m intruder at 1000m North flying South at 45 m/s
        let dt = 0.1_f64;
        let focal_px = 2690.0_f64;
        let wingspan_m = 10.0_f64;
        let mut own_p = [0.0_f64, 0.0, -500.0];
        let mut own_hdg = 0.0_f64;
        let own_tas = 35.0_f64;
        let mut int_p = [1000.0_f64, 0.0, -500.0];
        let int_v = [-45.0_f64, 0.0, 0.0];

        let mut own_a = [0.0_f64; 3];
        let mut last_telem = None;
        let mut prev_s = focal_px * wingspan_m / 1000.0;

        for step_idx in 0..31 {
            let rel = [int_p[0] - own_p[0], int_p[1] - own_p[1], int_p[2] - own_p[2]];
            let r_true = libm::sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
            let horiz = libm::hypot(rel[0], rel[1]);
            let alpha = libm::atan2(rel[1], rel[0]);
            let beta = libm::atan2(-rel[2], horiz);
            let s_now = focal_px * wingspan_m / r_true;
            let s_rate = if step_idx == 0 {
                // True analytical initial growth rate s_dot = s * (V_c / R_true)
                s_now * (80.0 / r_true)
            } else {
                (s_now - prev_s) / dt
            };
            prev_s = s_now;

            let own = OwnshipStateNed {
                v_ned: [own_tas * libm::cos(own_hdg), own_tas * libm::sin(own_hdg), 0.0],
                a_ned: own_a,
                heading_rad: own_hdg,
                tas_mps: own_tas,
                v_stall_mps: 14.5,
            };
            let meas = ActiveRangeMeasurement {
                alpha_rad: alpha,
                beta_rad: beta,
                size_px: s_now,
                size_rate_px_s: s_rate,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: 5 + step_idx as u32,
                dt_s: dt,
            };

            let telem = ctrl.step(&own, &meas);
            if step_idx == 0 {
                assert_eq!(
                    telem.mode,
                    ActiveRangeMode::ActiveSwerve as u8,
                    "tick 1 on pure head-on CBDR must permit ActiveSwerve, got {:?}",
                    telem
                );
                assert!(
                    (telem.cmd_bank_rad as f64).abs() <= PHI_CAND_RAD[8] + 1e-6,
                    "active swerve bank must stay <= 12 deg"
                );
                assert!(telem.min_sc3bf_margin >= 0.0);
                assert_ne!(telem.flags & FLAG_SC3BF_ALL_SAFE, 0);
                assert_ne!(telem.flags & FLAG_FOV_SAFE, 0);
            }

            // Advance 6-DoF coordinated turn kinematics with commanded bank
            let phi = telem.cmd_bank_rad as f64;
            let a_lat = G_MPS2 * libm::tan(phi);
            let psi_dot = a_lat / own_tas;
            let right = [-libm::sin(own_hdg), libm::cos(own_hdg), 0.0];
            own_a = [a_lat * right[0], a_lat * right[1], 0.0];
            let hdg_mid = own_hdg + 0.5 * psi_dot * dt;
            own_p[0] += own_tas * libm::cos(hdg_mid) * dt;
            own_p[1] += own_tas * libm::sin(hdg_mid) * dt;
            own_hdg = wrap_pi(own_hdg + psi_dot * dt);
            int_p[0] += int_v[0] * dt;
            int_p[1] += int_v[1] * dt;

            last_telem = Some((telem, r_true));
        }

        let (final_telem, final_r_true) = last_telem.unwrap();
        let rel_err = ((final_telem.r_hat_m as f64) - final_r_true).abs() / final_r_true;
        assert!(
            rel_err < 0.08,
            "3.0 s active micro-swerve must collapse range error < 8%: r_hat={:.1}, r_true={:.1}, err={:.2}%",
            final_telem.r_hat_m,
            final_r_true,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_off_axis_intruder_rejects_turn_toward_threat() {
        // Review Focus #2: Intruder at alpha = +5 deg, R = 260 m, closing at 45 m/s (tau = 5.78 s)
        let cfg = ActiveRangeConfig::default();
        let mut ctrl = ActiveRangeController::new(cfg);

        let alpha_5deg = 5.0_f64.to_radians();
        let own = OwnshipStateNed {
            v_ned: [35.0, 0.0, 0.0],
            a_ned: [0.0, 0.0, 0.0],
            heading_rad: 0.0,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let r_true = 260.0_f64;
        let s_px = 2690.0 * 6.0 / r_true; // ~62 px
        let s_rate = s_px * (45.0 / r_true); // rho_dot = -45/260 = -0.173 s^-1 (tau = 5.78 s)
        let meas = ActiveRangeMeasurement {
            alpha_rad: alpha_5deg,
            beta_rad: 0.0,
            size_px: s_px,
            size_rate_px_s: s_rate,
            contrast_ratio: 0.8,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: 6,
            dt_s: 0.1,
        };

        let telem = ctrl.step(&own, &meas);
        assert!(
            telem.cmd_bank_rad <= 0.0,
            "SC3BF must NEVER command positive bank toward +5 deg intruder; got bank={:.3} rad, mode={}",
            telem.cmd_bank_rad,
            telem.mode
        );
        assert!(
            telem.mode == ActiveRangeMode::ActiveSwerve as u8
                || telem.mode == ActiveRangeMode::ImmediateEvade as u8,
            "closing off-axis threat must command safe away swerve or ImmediateEvade; got mode={}",
            telem.mode
        );

        // Also directly check eval_universal_sc3bf rejects +12 deg (PHI_CAND_TAN[8])
        let triad = LosTriadNed::from_angles(alpha_5deg, 0.0);
        let sw_inp = SwerveInputs {
            r_banks: R_INIT_M,
            weights: [1.0 / 6.0; NUM_BANKS],
            r_hat_m: 1208.0,
            sigma_r_m: 1000.0,
            r_safe_lower_m: 260.0,
            rho_dot_inv_s: -45.0 / 260.0,
            alpha_dot: 0.0,
            beta_dot: 0.0,
            alpha_body_rad: alpha_5deg,
            triad,
            own_fwd_ned: [1.0, 0.0, 0.0],
            own_right_ned: [0.0, 1.0, 0.0],
            tas_mps: 35.0,
            v_stall_mps: 14.5,
            sigma_bearing_rad: 0.001063,
            cbdr_probe_active: false,
            was_locked: false,
            preferred_swerve_sign: 1.0,
        };
        let (_m_pos, safe_pos) = eval_universal_sc3bf(&sw_inp, PHI_CAND_TAN[8]);
        assert!(!safe_pos, "+12 deg turn toward +5 deg threat MUST fail universal SC3BF");
        let (m_neg, safe_neg) = eval_universal_sc3bf(&sw_inp, PHI_CAND_TAN[0]);
        assert!(safe_neg && m_neg >= 0.0, "-12 deg turn away from +5 deg threat MUST pass universal SC3BF");
    }

    #[test]
    fn test_nonfinite_and_zero_dt_inputs_fail_safe() {
        // Review Focus #1: NaN, Inf, dt=0.0, dt=-1.0 must never panic or emit non-finite fields
        let mut ctrl = ActiveRangeController::new(ActiveRangeConfig::default());
        let bad_cases = [
            (f64::NAN, 0.0, 0.1),
            (0.0, f64::INFINITY, 0.1),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (f64::NEG_INFINITY, f64::NAN, f64::NAN),
        ];

        for (alpha, beta, dt_s) in bad_cases {
            let own = OwnshipStateNed {
                v_ned: [f64::NAN, 0.0, 0.0],
                a_ned: [f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                heading_rad: f64::NAN,
                tas_mps: f64::NAN,
                v_stall_mps: f64::INFINITY,
            };
            let meas = ActiveRangeMeasurement {
                alpha_rad: alpha,
                beta_rad: beta,
                size_px: f64::NAN,
                size_rate_px_s: f64::INFINITY,
                contrast_ratio: f64::NAN,
                sky_beta_ext: f64::NAN,
                daytime_clear: true,
                track_age: 10,
                dt_s,
            };

            let t = ctrl.step(&own, &meas);
            assert!(t.r_hat_m.is_finite());
            assert!(t.r_hat_m >= R_FLOOR_M as f32 && t.r_hat_m <= R_CEIL_M as f32);
            assert!(t.sigma_r_m.is_finite() && t.sigma_r_m >= 0.0);
            assert!(t.r_safe_lower_m.is_finite() && t.r_safe_lower_m >= R_FLOOR_M as f32);
            assert!(t.r_safe_lower_m <= t.r_hat_m);
            assert!(t.cmd_bank_rad.is_finite());
            assert!(t.min_sc3bf_margin.is_finite());
            assert!(t.fim_gain.is_finite());

            let sum_w: f32 = t.weights.iter().sum();
            assert!((sum_w - 1.0).abs() <= 1.0e-4, "weights sum={sum_w}");
            assert!(decode_wire(&encode_wire(&t)).is_some());
        }
    }

    #[test]
    fn test_counterexample_in_cone_near_fov_edge_escalates_to_immediate_evade() {
        // D16-P1-M1 regression: intruder at alpha_body = +46.8 deg (near 47 deg FOV edge),
        // r_hat = 600 m, sigma_r = 300 m, r_safe_lower = 155 m (h_0(155 m) < 0, inside collision
        // cone), rho_dot = -0.19 s^-1 (tau = 5.26 s > 4.5 s). Left turns violate 47 deg FOV,
        // right turns worsen SC3BF, and straight flight (phi = 0.0, h_phi == h_0) MUST be
        // unsafe inside a violated cone -> empty feasible set escalates to ImmediateEvade (±25 deg).
        let alpha_rad = 46.8_f64.to_radians();
        let triad = LosTriadNed::from_angles(alpha_rad, 0.0);
        let inp = SwerveInputs {
            r_banks: R_INIT_M,
            weights: [1.0 / 6.0; NUM_BANKS],
            r_hat_m: 600.0,
            sigma_r_m: 300.0,
            r_safe_lower_m: 155.0,
            rho_dot_inv_s: -0.19,
            alpha_dot: 0.0,
            beta_dot: 0.0,
            alpha_body_rad: alpha_rad,
            triad,
            own_fwd_ned: [1.0, 0.0, 0.0],
            own_right_ned: [0.0, 1.0, 0.0],
            tas_mps: 35.0,
            v_stall_mps: 14.5,
            sigma_bearing_rad: 0.00055,
            cbdr_probe_active: false,
            was_locked: false,
            preferred_swerve_sign: 1.0,
        };

        let h0_low = crate::sc3bf_fim::eval_sc3bf_h(&inp, 155.0, 0.0);
        assert!(h0_low < 0.0, "expected violated collision cone h_0 < 0, got {h0_low}");

        let (m0, safe0) = eval_universal_sc3bf(&inp, 0.0);
        assert!(
            !safe0,
            "straight flight phi=0.0 MUST be unsafe inside violated collision cone (m0={m0})"
        );

        let dec = crate::sc3bf_fim::select_active_swerve(&inp);
        assert_eq!(
            dec.mode,
            ActiveRangeMode::ImmediateEvade,
            "empty feasible set inside violated cone near FOV edge must escalate to ImmediateEvade, got {:?}",
            dec
        );
        assert!(
            dec.cmd_bank_rad.abs() > 0.4,
            "ImmediateEvade must command ±25 deg (|bank| > 0.4 rad), got {}",
            dec.cmd_bank_rad
        );
        assert!(!dec.sc3bf_all_safe);
    }
}
