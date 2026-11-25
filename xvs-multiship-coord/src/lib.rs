// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0
//! OpenXVS V5.0 Additive `#![no_std]` `#![forbid(unsafe_code)]` Multi-Ship Coordination Layer.
//!
//! Wraps the unmodified frozen OpenXVS crates (`xvs-active-range`, `xvs-math-core::mesh_daa`,
//! `xvs-rta-core`) from above to close two verified formal/empirical findings from `DR-22`:
//!
//! 1. **Finding F1 (`swerve_coord`):** Breaks asynchronous `12`-tick `preferred_swerve_sign`
//!    observability cancellation (`det(FIM_rel) = 0`, `3.79x` unconservative range overestimate)
//!    via FAR 91.113(e) non-reversing right-turn precedence (`+1`) + deterministic aircraft-ID
//!    phase staggering over fresh `XVS-MeshDAA` links, matching `USE_COORD_FIX = TRUE` in
//!    `formal/tla/OpenXvsMultiShipRta.tla`, plus a Ring-0 `P0` inhibit authority arbiter.
//! 2. **Finding F4 (`mesh_guard`):** Wraps frozen `MeshDaaFuser` with a `200 ms` monotonic
//!    staleness/future-skew/replay gate and a `500 ms` sticky triple-ray ghost-suppression
//!    hysteresis window eliminating `3 <-> 2` peer-dropout coplanar ghost re-admission.
//!
//! Strictly civil/dual-use aviation Detect-and-Avoid (DAA) and Run-Time Assurance (`14 CFR Part
//! 91/107/108/135`, `EASA SORA 2.5`, `ASTM F3269-21`, `RTCA DO-365A/B`). All local geometry is
//! strictly NED (`North, East, Down`) / ECEF; zero internal ENU, zero heap allocations, zero
//! `unsafe`, and zero modifications to any frozen crate.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod clock_sync;
#[cfg(kani)]
mod kani_proofs;
pub mod mesh_guard;
pub mod swerve_coord;

pub use clock_sync::{
    ClockSyncRejectReason, ClockSyncSampleOutcome, PeerClockEstimate, PeerClockOffsetEstimator,
    TwoWaySyncExchange, MAX_ABS_CLOCK_OFFSET_US, MAX_OFFSET_UNCERT_US, MAX_REL_TCXO_DRIFT_PPM,
    MAX_SLEW_STEP_US, MAX_SYNC_PEERS, MAX_SYNC_RTT_US, MIN_SYNC_SAMPLES, SLEW_RATE_DIVISOR,
    SYNC_WINDOW_SIZE,
};
pub use mesh_guard::{
    GuardedMeshFusionResult, StatefulMeshDaaGuard, MAX_FUTURE_SKEW_US, MAX_GUARD_OBS,
    MAX_MESH_AGE_US, MAX_TRACKED_PEERS, TRIPLE_RAY_HYSTERESIS_US,
};
pub use swerve_coord::{
    CoordinatedActiveRange, PeerCoordHint, Ring0AuthorityGate, SwerveCoordRole,
    HEAD_ON_ALPHA_BAND_RAD, PEER_LINK_FRESH_US, STAGGER_SLOT_TICKS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use xvs_active_range::{
        decode_wire, encode_wire, wrap_pi, ActiveRangeConfig, ActiveRangeController,
        ActiveRangeMeasurement, ActiveRangeMode, OwnshipStateNed, G_MPS2,
    };
    use xvs_math_core::mesh_daa::{MeshDaaFuser, OpenXvsBearingTrack};

    struct Ship6Dof {
        p: [f64; 3],
        hdg: f64,
        tas: f64,
        a: [f64; 3],
    }

    impl Ship6Dof {
        fn own_state(&self) -> OwnshipStateNed {
            OwnshipStateNed {
                v_ned: [self.tas * self.hdg.cos(), self.tas * self.hdg.sin(), 0.0],
                a_ned: self.a,
                heading_rad: self.hdg,
                tas_mps: self.tas,
                v_stall_mps: 14.5,
            }
        }

        fn advance(&mut self, bank_rad: f64, dt: f64) {
            let a_lat = G_MPS2 * bank_rad.tan();
            let psi_dot = a_lat / self.tas;
            let right = [-self.hdg.sin(), self.hdg.cos(), 0.0];
            self.a = [a_lat * right[0], a_lat * right[1], 0.0];
            let hdg_mid = self.hdg + 0.5 * psi_dot * dt;
            self.p[0] += self.tas * hdg_mid.cos() * dt;
            self.p[1] += self.tas * hdg_mid.sin() * dt;
            self.hdg = wrap_pi(self.hdg + psi_dot * dt);
        }
    }

    fn measure_rel(
        own: &Ship6Dof,
        intr: &Ship6Dof,
        prev_s: &mut Option<f64>,
        track_age: u32,
        dt: f64,
    ) -> (ActiveRangeMeasurement, f64) {
        let rel = [
            intr.p[0] - own.p[0],
            intr.p[1] - own.p[1],
            intr.p[2] - own.p[2],
        ];
        let r_true = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2])
            .sqrt()
            .max(15.0);
        let horiz = rel[0].hypot(rel[1]).max(1.0);
        let alpha = wrap_pi(rel[1].atan2(rel[0]));
        let beta = (-rel[2]).atan2(horiz);
        let s_now = 2690.0 * 10.0 / r_true;
        let s_rate = match *prev_s {
            None => {
                let v_own = [own.tas * own.hdg.cos(), own.tas * own.hdg.sin(), 0.0];
                let v_int = [intr.tas * intr.hdg.cos(), intr.tas * intr.hdg.sin(), 0.0];
                let v_rel = [v_int[0] - v_own[0], v_int[1] - v_own[1], 0.0];
                let r_dot = (rel[0] * v_rel[0] + rel[1] * v_rel[1]) / r_true;
                -s_now * (r_dot / r_true)
            }
            Some(ps) => (s_now - ps) / dt,
        };
        *prev_s = Some(s_now);
        (
            ActiveRangeMeasurement {
                alpha_rad: alpha,
                beta_rad: beta,
                size_px: s_now,
                size_rate_px_s: s_rate,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age,
                dt_s: dt,
            },
            r_true,
        )
    }

    fn bank_sign(b: f32) -> i32 {
        if b > 1.0e-6 {
            1
        } else if b < -1.0e-6 {
            -1
        } else {
            0
        }
    }

    #[test]
    fn test_f1_eliminates_async12_379x_overestimate_optical_only() {
        let dt = 0.1_f64;

        // 1. Reproduce baseline DR-22 Finding F1 on unmodified frozen ActiveRangeController
        let (base_peak_r_hat, base_r_true_at_peak, base_peak_ratio, base_cancelled) = {
            let mut ar2 = ActiveRangeController::new(ActiveRangeConfig::default());
            let prime_own = OwnshipStateNed {
                v_ned: [-35.0, 0.0, 0.0],
                a_ned: [0.0; 3],
                heading_rad: core::f64::consts::PI,
                tas_mps: 35.0,
                v_stall_mps: 14.5,
            };
            let prime_meas = ActiveRangeMeasurement {
                alpha_rad: core::f64::consts::PI,
                beta_rad: 0.0,
                size_px: 15.0,
                size_rate_px_s: 0.75,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: 6,
                dt_s: dt,
            };
            for _ in 0..12 {
                let p = ar2.step(&prime_own, &prime_meas);
                assert_eq!(p.mode, ActiveRangeMode::ActiveSwerve as u8);
            }

            let mut s1 = Ship6Dof {
                p: [0.0, 0.0, -500.0],
                hdg: 0.0,
                tas: 52.0,
                a: [0.0; 3],
            };
            let mut s2 = Ship6Dof {
                p: [1200.0, 0.0, -500.0],
                hdg: core::f64::consts::PI,
                tas: 52.0,
                a: [0.0; 3],
            };
            let mut ar1 = ActiveRangeController::new(ActiveRangeConfig::default());
            let mut ps1 = None;
            let mut ps2 = None;
            let mut peak_ratio = 0.0_f64;
            let mut peak_r_hat = 0.0_f64;
            let mut r_true_at_peak = 0.0_f64;
            let mut cancelled = 0u32;

            for t in 0..30u32 {
                let (m1, r_true) = measure_rel(&s1, &s2, &mut ps1, 6 + t, dt);
                let (m2, _) = measure_rel(&s2, &s1, &mut ps2, 18 + t, dt);
                let t1 = ar1.step(&s1.own_state(), &m1);
                let t2 = ar2.step(&s2.own_state(), &m2);
                if bank_sign(t1.cmd_bank_rad) != 0
                    && bank_sign(t1.cmd_bank_rad) == -bank_sign(t2.cmd_bank_rad)
                {
                    cancelled += 1;
                }
                let ratio = (t1.r_hat_m as f64) / r_true;
                if ratio > peak_ratio {
                    peak_ratio = ratio;
                    peak_r_hat = t1.r_hat_m as f64;
                    r_true_at_peak = r_true;
                }
                s1.advance(t1.cmd_bank_rad as f64, dt);
                s2.advance(t2.cmd_bank_rad as f64, dt);
            }
            (peak_r_hat, r_true_at_peak, peak_ratio, cancelled)
        };
        assert_eq!(base_cancelled, 30);
        assert!(base_peak_ratio > 3.5, "baseline must reproduce F1 3.79x overestimate");

        // 2. Run EXACT same 12-tick asynchronous offset with CoordinatedActiveRange (Optical-Only Tier B:
        //    NO MeshDAA peer link, pure FAR 91.113 non-reversing +1 right-turn precedence!)
        let (fix_peak_r_hat, fix_final_r_hat, fix_final_r_true, fix_peak_ratio, fix_cancelled) = {
            let mut c2 = CoordinatedActiveRange::new(2, ActiveRangeConfig::default());
            let prime_own = OwnshipStateNed {
                v_ned: [-35.0, 0.0, 0.0],
                a_ned: [0.0; 3],
                heading_rad: core::f64::consts::PI,
                tas_mps: 35.0,
                v_stall_mps: 14.5,
            };
            let prime_meas = ActiveRangeMeasurement {
                alpha_rad: core::f64::consts::PI,
                beta_rad: 0.0,
                size_px: 15.0,
                size_rate_px_s: 0.75,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: 6,
                dt_s: dt,
            };
            for _ in 0..12 {
                let p = c2.step(&prime_own, &prime_meas, None, Ring0AuthorityGate::NOMINAL);
                assert_eq!(p.mode, ActiveRangeMode::ActiveSwerve as u8);
            }
            // Reset Ship 2 filter moments as encounter starts while KEEPING its 12-tick swerve phase offset!
            c2.reset_filter_keep_phase();

            let mut s1 = Ship6Dof {
                p: [0.0, 0.0, -500.0],
                hdg: 0.0,
                tas: 52.0,
                a: [0.0; 3],
            };
            let mut s2 = Ship6Dof {
                p: [1200.0, 0.0, -500.0],
                hdg: core::f64::consts::PI,
                tas: 52.0,
                a: [0.0; 3],
            };
            let mut c1 = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
            let mut ps1 = None;
            let mut ps2 = None;
            let mut peak_ratio = 0.0_f64;
            let mut peak_r_hat = 0.0_f64;
            let mut final_r_hat = 0.0_f64;
            let mut final_r_true = 0.0_f64;
            let mut cancelled = 0u32;

            for t in 0..30u32 {
                let (m1, r_true) = measure_rel(&s1, &s2, &mut ps1, 6 + t, dt);
                let (m2, _) = measure_rel(&s2, &s1, &mut ps2, 18 + t, dt);
                let t1 = c1.step(&s1.own_state(), &m1, None, Ring0AuthorityGate::NOMINAL);
                let t2 = c2.step(&s2.own_state(), &m2, None, Ring0AuthorityGate::NOMINAL);
                assert!(decode_wire(&encode_wire(&t1)).is_some());
                assert!(decode_wire(&encode_wire(&t2)).is_some());
                if bank_sign(t1.cmd_bank_rad) != 0
                    && bank_sign(t1.cmd_bank_rad) == -bank_sign(t2.cmd_bank_rad)
                {
                    cancelled += 1;
                }
                let ratio = (t1.r_hat_m as f64) / r_true;
                if ratio > peak_ratio {
                    peak_ratio = ratio;
                    peak_r_hat = t1.r_hat_m as f64;
                }
                if t == 29 {
                    final_r_hat = t1.r_hat_m as f64;
                    final_r_true = r_true;
                }
                s1.advance(t1.cmd_bank_rad as f64, dt);
                s2.advance(t2.cmd_bank_rad as f64, dt);
            }
            (peak_r_hat, final_r_hat, final_r_true, peak_ratio, cancelled)
        };

        println!(
            "F1_OPTICAL_ONLY: baseline_peak={base_peak_r_hat:.2}m (r_true={base_r_true_at_peak:.2}m, ratio={base_peak_ratio:.4}x, cancelled={base_cancelled}/30) -> \
             fixed_peak={fix_peak_r_hat:.2}m (peak_ratio={fix_peak_ratio:.4}x, final_r_hat={fix_final_r_hat:.2}m vs r_true={fix_final_r_true:.2}m, cancelled={fix_cancelled}/30)"
        );

        assert_eq!(fix_cancelled, 0, "FAR 91.113 coordination must eliminate all cancelled ticks");
        assert!(
            fix_peak_ratio < 1.15,
            "Coordinated peak R_hat/R_true must never blow up unconservatively: got {fix_peak_ratio:.4}x"
        );
        assert!(
            fix_final_r_hat <= fix_final_r_true,
            "Optical-only FAR 91.113 head-on R_hat must remain strictly conservative: {fix_final_r_hat:.2} vs {fix_final_r_true:.2}"
        );
    }

    #[test]
    fn test_f1_meshdaa_aircraft_id_stagger_converges_accurately() {
        let dt = 0.1_f64;
        let mut s1 = Ship6Dof {
            p: [0.0, 0.0, -500.0],
            hdg: 0.0,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut s2 = Ship6Dof {
            p: [1200.0, 0.0, -500.0],
            hdg: core::f64::consts::PI,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut c1 = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let mut c2 = CoordinatedActiveRange::new(2, ActiveRangeConfig::default());
        let mut ps1 = None;
        let mut ps2 = None;

        let mut cancelled = 0u32;
        let mut s1_r_hat_t20 = 0.0_f64;
        let mut s1_r_true_t20 = 0.0_f64;
        let mut s2_r_hat_t39 = 0.0_f64;
        let mut s2_r_true_t39 = 0.0_f64;

        for t in 0..40u32 {
            let now_us = 1_700_000_000_000 + (t as u64) * 100_000;
            let (m1, r_true) = measure_rel(&s1, &s2, &mut ps1, 6 + t, dt);
            let (m2, _) = measure_rel(&s2, &s1, &mut ps2, 6 + t, dt);

            let h1 = PeerCoordHint {
                peer_sender_id: 2,
                peer_ts_utc_us: now_us - 40_000, // 40 ms fresh MeshDAA frame
                now_utc_us: now_us,
                peer_a_ned: Some(s2.a),
            };
            let h2 = PeerCoordHint {
                peer_sender_id: 1,
                peer_ts_utc_us: now_us - 40_000,
                now_utc_us: now_us,
                peer_a_ned: Some(s1.a),
            };

            let t1 = c1.step(&s1.own_state(), &m1, Some(h1), Ring0AuthorityGate::NOMINAL);
            let t2 = c2.step(&s2.own_state(), &m2, Some(h2), Ring0AuthorityGate::NOMINAL);

            if bank_sign(t1.cmd_bank_rad) != 0
                && bank_sign(t1.cmd_bank_rad) == -bank_sign(t2.cmd_bank_rad)
            {
                cancelled += 1;
            }
            if t == 19 {
                s1_r_hat_t20 = t1.r_hat_m as f64;
                s1_r_true_t20 = r_true;
            }
            if t == 39 {
                s2_r_hat_t39 = t2.r_hat_m as f64;
                s2_r_true_t39 = r_true;
            }
            s1.advance(t1.cmd_bank_rad as f64, dt);
            s2.advance(t2.cmd_bank_rad as f64, dt);
        }

        let err1 = (s1_r_hat_t20 - s1_r_true_t20).abs() / s1_r_true_t20;
        let err2 = (s2_r_hat_t39 - s2_r_true_t39).abs() / s2_r_true_t39;
        println!(
            "F1_MESHDAA_STAGGER: Ship1@2.0s r_hat={s1_r_hat_t20:.2}m vs r_true={s1_r_true_t20:.2}m (err={:.2}%), \
             Ship2@4.0s r_hat={s2_r_hat_t39:.2}m vs r_true={s2_r_true_t39:.2}m (err={:.2}%), cancelled={cancelled}/40",
            err1 * 100.0,
            err2 * 100.0
        );
        assert_eq!(cancelled, 0);
        assert!(err1 < 0.12, "Ship 1 staggered probe error too high: {:.2}%", err1 * 100.0);
        assert!(err2 < 0.20, "Ship 2 staggered probe error too high: {:.2}%", err2 * 100.0);
    }

    #[test]
    fn test_f3_ring0_p0_arbiter_overrides_immediate_evade() {
        let mut coord = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let own = OwnshipStateNed {
            v_ned: [35.0, 0.0, 0.0],
            a_ned: [0.0; 3],
            heading_rad: 0.0,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let close_meas = ActiveRangeMeasurement {
            alpha_rad: 0.0,
            beta_rad: 0.0,
            size_px: 140.0,
            size_rate_px_s: 45.0, // tau = 3.1 s < 4.5 s
            contrast_ratio: 0.9,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: 15,
            dt_s: 0.1,
        };

        // Under P0 RTA inhibit, bank MUST be strictly zeroed
        let t_p0 = coord.step(
            &own,
            &close_meas,
            None,
            Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: true,
                hal_out_mode_wire: 4,
            },
        );
        assert_eq!(t_p0.cmd_bank_rad, 0.0);
        assert_eq!(t_p0.mode, ActiveRangeMode::NominalHold as u8);

        // Under HAL-P0 monotonic tightening, bank MUST also be strictly zeroed
        let t_hal_p0 = coord.step(
            &own,
            &close_meas,
            None,
            Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: false,
                hal_out_mode_wire: 0,
            },
        );
        assert_eq!(t_hal_p0.cmd_bank_rad, 0.0);
        assert_eq!(t_hal_p0.mode, ActiveRangeMode::NominalHold as u8);
    }

    #[test]
    fn test_f4_meshdaa_200ms_staleness_and_sticky_triple_ray_zero_ghosts() {
        let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
        let p_s1 = [base_ecef[0], 0.0, 0.0];
        let p_s2 = [base_ecef[0], 1000.0, 0.0];
        let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
        let t1 = [base_ecef[0] + 500.0, 200.0, 0.0];
        let t2 = [base_ecef[0] + 500.0, 800.0, 0.0];

        let mk_obs = |sender: u8, tid: u16, origin: [f64; 3], target: [f64; 3], ts_us: u64| {
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let raw = OpenXvsBearingTrack::from_ecef_ray(
                ts_us, origin, dir, [0.0; 3], 1500, 100, tid, sender, 0,
            );
            OpenXvsBearingTrack::from_le_bytes(&raw.to_le_bytes())
        };

        let mut raw_fuser = MeshDaaFuser::new();
        let mut guard = StatefulMeshDaaGuard::new();

        let mut raw_3ship_ghosts_rej = 0u32;
        let mut raw_2ship_ghosts_admitted = 0u32;
        let mut guard_3ship_ghosts_rej = 0u32;
        let mut guard_2ship_ghosts_admitted = 0u32;
        let mut guard_2ship_sticky_suppressed = 0u32;
        let mut guard_2ship_true_tracks_kept = 0u32;

        // 50 cycles of 3-ship nominal tick followed by 200 ms peer #3 network dropout (2-ship tick)
        for cycle in 0..50u64 {
            let ts_3ship = 1_700_000_000_000 + cycle * 400_000;
            let obs_3 = [
                mk_obs(1, 1, p_s1, t1, ts_3ship),
                mk_obs(1, 2, p_s1, t2, ts_3ship),
                mk_obs(2, 1, p_s2, t1, ts_3ship),
                mk_obs(2, 2, p_s2, t2, ts_3ship),
                mk_obs(3, 1, p_s3, t1, ts_3ship),
                mk_obs(3, 2, p_s3, t2, ts_3ship),
            ];
            let r3 = raw_fuser.fuse(&obs_3);
            raw_3ship_ghosts_rej += r3.ghosts_rejected;

            let g3 = guard.fuse_guarded(&obs_3, ts_3ship + 20_000);
            guard_3ship_ghosts_rej += g3.ghosts_rejected_3ship;
            assert_eq!(g3.num_fused, 2);

            // 200 ms later: peer #3 drops out; only peers 1 and 2 arrive fresh!
            let ts_2ship = ts_3ship + 200_000;
            let obs_2 = [
                mk_obs(1, 1, p_s1, t1, ts_2ship),
                mk_obs(1, 2, p_s1, t2, ts_2ship),
                mk_obs(2, 1, p_s2, t1, ts_2ship),
                mk_obs(2, 2, p_s2, t2, ts_2ship),
            ];
            let r2 = raw_fuser.fuse(&obs_2);
            if r2.num_fused > 2 {
                raw_2ship_ghosts_admitted += (r2.num_fused - 2) as u32;
            }

            let g2 = guard.fuse_guarded(&obs_2, ts_2ship + 20_000);
            assert!(g2.sticky_triple_ray_active);
            guard_2ship_sticky_suppressed += g2.sticky_ghosts_suppressed;
            guard_2ship_true_tracks_kept += g2.num_fused.min(2) as u32;
            if g2.num_fused > 2 {
                guard_2ship_ghosts_admitted += (g2.num_fused - 2) as u32;
            }
        }

        println!(
            "F4_MESHDAA_GUARD: raw_3ship_rej={raw_3ship_ghosts_rej} raw_2ship_ghosts_admitted={raw_2ship_ghosts_admitted} -> \
             guard_3ship_rej={guard_3ship_ghosts_rej} guard_2ship_ghosts_admitted={guard_2ship_ghosts_admitted} \
             guard_sticky_suppressed={guard_2ship_sticky_suppressed} true_tracks_kept={guard_2ship_true_tracks_kept}/100"
        );
        assert_eq!(raw_2ship_ghosts_admitted, 100);
        assert_eq!(guard_2ship_ghosts_admitted, 0, "Sticky triple-ray must admit 0/100 ghosts!");
        assert_eq!(guard_2ship_sticky_suppressed, 100);
        assert_eq!(guard_2ship_true_tracks_kept, 100);

        // Verify > 200 ms staleness gate, > 50 ms future skew gate, and monotonic replay rejection
        let now_us = 1_700_050_000_000u64;
        let stale_obs = [
            mk_obs(1, 1, p_s1, t1, now_us - 250_000), // 250 ms stale -> REJECT
            mk_obs(2, 1, p_s2, t1, now_us + 90_000),  // 90 ms future -> REJECT
            mk_obs(3, 1, p_s3, t1, now_us - 50_000),  // fresh
        ];
        let g_stale = guard.fuse_guarded(&stale_obs, now_us);
        assert_eq!(g_stale.stale_obs_rejected, 2);
        assert_eq!(g_stale.fresh_sender_count, 1);

        // Replay older timestamp for sender 3 -> REJECT
        let replay_obs = [mk_obs(3, 1, p_s3, t1, now_us - 100_000)];
        let g_rep = guard.fuse_guarded(&replay_obs, now_us + 10_000);
        assert_eq!(g_rep.replay_obs_rejected, 1);
        assert_eq!(g_rep.fresh_sender_count, 0);
    }

    #[test]
    fn test_nonfinite_inputs_fail_safe_and_64b_wire_crc_valid() {
        let mut coord = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let bad_own = OwnshipStateNed {
            v_ned: [f64::NAN, 0.0, 0.0],
            a_ned: [f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            heading_rad: f64::NAN,
            tas_mps: f64::NAN,
            v_stall_mps: f64::INFINITY,
        };
        let bad_meas = ActiveRangeMeasurement {
            alpha_rad: f64::NAN,
            beta_rad: 0.0,
            size_px: f64::NAN,
            size_rate_px_s: f64::INFINITY,
            contrast_ratio: f64::NAN,
            sky_beta_ext: f64::NAN,
            daytime_clear: true,
            track_age: 10,
            dt_s: 0.0,
        };
        let pkt = coord.step(&bad_own, &bad_meas, None, Ring0AuthorityGate::NOMINAL);
        assert_eq!(pkt.mode, ActiveRangeMode::NominalHold as u8);
        assert_eq!(pkt.cmd_bank_rad, 0.0);
        assert!(pkt.r_hat_m.is_finite());
        assert!(decode_wire(&encode_wire(&pkt)).is_some());
    }

    #[test]
    fn test_sort5_network_exhaustive_all_120_permutations() {
        let mut base = [10i64, 20, 30, 40, 50];
        // Generate all 5! = 120 permutations via Heap's algorithm
        let mut c = [0usize; 5];
        let mut checked = 0usize;
        let check = |arr: [i64; 5]| {
            let s = clock_sync::sort5_i64(arr);
            assert_eq!(s, [10, 20, 30, 40, 50]);
            let u = [
                arr[0] as u64,
                arr[1] as u64,
                arr[2] as u64,
                arr[3] as u64,
                arr[4] as u64,
            ];
            assert_eq!(clock_sync::sort5_u64(u), [10, 20, 30, 40, 50]);
        };
        check(base);
        checked += 1;
        let mut i = 0usize;
        while i < 5 {
            if c[i] < i {
                if i.is_multiple_of(2) {
                    base.swap(0, i);
                } else {
                    base.swap(c[i], i);
                }
                check(base);
                checked += 1;
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        assert_eq!(checked, 120);
    }

    #[test]
    fn test_clock_sync_gnss_denied_60min_tcxo_drift_and_fail_closed_tier_b() {
        let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
        let p_s1 = [base_ecef[0], 0.0, 0.0];
        let p_s2 = [base_ecef[0], 1000.0, 0.0];
        let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
        let t1 = [base_ecef[0] + 500.0, 200.0, 0.0];
        let t2 = [base_ecef[0] + 500.0, 800.0, 0.0];

        let mk_obs = |sender: u8, tid: u16, origin: [f64; 3], target: [f64; 3], ts_us: u64| {
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let raw = OpenXvsBearingTrack::from_ecef_ray(
                ts_us, origin, dir, [0.0; 3], 1500, 100, tid, sender, 0,
            );
            OpenXvsBearingTrack::from_le_bytes(&raw.to_le_bytes())
        };

        let mk_exchange = |sender: u8, t1_local: u64, offset_us: i64, d_fwd: u64, turn: u64, d_rev: u64| {
            let t2_peer = ((t1_local as i128) + (offset_us as i128) + (d_fwd as i128)) as u64;
            let t3_peer = t2_peer + turn;
            let t4_local = t1_local + d_fwd + turn + d_rev;
            TwoWaySyncExchange {
                sender_id: sender,
                t1_local_tx_us: t1_local,
                t2_peer_rx_us: t2_peer,
                t3_peer_tx_us: t3_peer,
                t4_local_rx_us: t4_local,
            }
        };

        // 1. Reproduce Audit Failure Mode 7: 60-min GNSS-denied TCXO drift (+120 ms on peer 2, -110 ms on peer 3)
        //    Legacy unsynced guard drops peer 2 (future skew > 50 ms)!
        let now_base: u64 = 1_700_000_000_000;
        let drift_p2: i64 = 120_000; // +120 ms TCXO drift
        let drift_p3: i64 = -110_000; // -110 ms TCXO drift
        let mut legacy_guard = StatefulMeshDaaGuard::new();
        let obs_drifted = [
            mk_obs(1, 1, p_s1, t1, now_base - 15_000),
            mk_obs(1, 2, p_s1, t2, now_base - 15_000),
            mk_obs(2, 1, p_s2, t1, (now_base - 15_000).wrapping_add_signed(drift_p2)),
            mk_obs(2, 2, p_s2, t2, (now_base - 15_000).wrapping_add_signed(drift_p2)),
            mk_obs(3, 1, p_s3, t1, (now_base - 15_000).wrapping_add_signed(drift_p3)),
            mk_obs(3, 2, p_s3, t2, (now_base - 15_000).wrapping_add_signed(drift_p3)),
        ];
        let g_legacy = legacy_guard.fuse_guarded(&obs_drifted, now_base);
        assert_eq!(g_legacy.stale_obs_rejected, 2, "Legacy guard must drop +120 ms drifted peer 2");
        assert_eq!(g_legacy.fresh_sender_count, 2);

        // 2. Calibrate PeerClockOffsetEstimator via 5 two-way RF exchanges (including 1 wild outlier!)
        let mut sync = PeerClockOffsetEstimator::new();
        // Also verify >150 ms RTT is rejected outright
        let bad_rtt = sync.record_exchange(mk_exchange(2, now_base - 900_000, drift_p2, 80_000, 5_000, 80_000));
        assert!(!bad_rtt.accepted);
        assert_eq!(bad_rtt.reject_reason, Some(ClockSyncRejectReason::RttExceeded));

        for k in 0..5u64 {
            let t1_k = now_base - 1_200_000 + k * 200_000;
            // Inject 1 massive multipath outlier on k==2 (still <= 150 ms RTT)
            let ex2 = if k == 2 {
                mk_exchange(2, t1_k, drift_p2 + 300_000, 60_000, 5_000, 60_000)
            } else {
                mk_exchange(2, t1_k, drift_p2, 5_000 + k * 200, 2_000, 5_000)
            };
            let o2 = sync.record_exchange(ex2);
            assert!(o2.accepted);

            let ex3 = if k == 3 {
                mk_exchange(3, t1_k, drift_p3 - 250_000, 65_000, 5_000, 65_000)
            } else {
                mk_exchange(3, t1_k, drift_p3, 6_000, 2_000, 6_000 + k * 200)
            };
            let o3 = sync.record_exchange(ex3);
            assert!(o3.accepted);
        }

        let est2 = sync.estimate_peer(2, now_base);
        let est3 = sync.estimate_peer(3, now_base);
        assert!(est2.synchronized && est2.offset_uncert_us <= MAX_OFFSET_UNCERT_US);
        assert!(est3.synchronized && est3.offset_uncert_us <= MAX_OFFSET_UNCERT_US);
        assert!((est2.offset_us - drift_p2).abs() <= 1_000);
        assert!((est3.offset_us - drift_p3).abs() <= 1_000);

        // 3. Under GNSS-denied 60-min drift: 0 false rejections, 2/2 true tracks confirmed, 0 ghosts, NO fallback
        let mut guard = StatefulMeshDaaGuard::new();
        let g_synced = guard.fuse_guarded_with_clock_sync(&obs_drifted, now_base, 1, true, &sync);
        assert_eq!(g_synced.stale_obs_rejected, 0);
        assert_eq!(g_synced.replay_obs_rejected, 0);
        assert_eq!(g_synced.uncalibrated_obs_rejected, 0);
        assert_eq!(g_synced.fresh_sender_count, 3);
        assert_eq!(g_synced.num_fused, 2);
        assert!(!g_synced.fallback_to_tier_b_optical);

        // 4. Genuine >200 ms stale packet and replay packet are STILL rejected cleanly!
        let now_next = now_base + 100_000;
        let stale_and_replay = [
            // Genuine 250 ms stale packet from drifted peer 2
            mk_obs(2, 1, p_s2, t1, (now_next - 250_000).wrapping_add_signed(drift_p2)),
            // Replay older raw timestamp from drifted peer 3 (even though aligned within 200 ms!)
            mk_obs(3, 1, p_s3, t1, (now_base - 30_000).wrapping_add_signed(drift_p3)),
        ];
        let g_attack = guard.fuse_guarded_with_clock_sync(&stale_and_replay, now_next, 1, true, &sync);
        assert_eq!(g_attack.stale_obs_rejected, 1);
        assert_eq!(g_attack.replay_obs_rejected, 1);
        assert_eq!(g_attack.fresh_sender_count, 0);

        // 5. 3-ship dropout > 500 ms (TRIPLE_RAY_HYSTERESIS_US): NEVER admits unguarded 2-ship ghosts,
        //    signals fallback_to_tier_b_optical == true!
        let now_drop = now_base + TRIPLE_RAY_HYSTERESIS_US + 50_000;
        let obs_2ship_expired = [
            mk_obs(1, 1, p_s1, t1, now_drop - 10_000),
            mk_obs(1, 2, p_s1, t2, now_drop - 10_000),
            mk_obs(2, 1, p_s2, t1, (now_drop - 10_000).wrapping_add_signed(drift_p2)),
            mk_obs(2, 2, p_s2, t2, (now_drop - 10_000).wrapping_add_signed(drift_p2)),
        ];
        let g_exp = guard.fuse_guarded_with_clock_sync(&obs_2ship_expired, now_drop, 1, true, &sync);
        assert_eq!(g_exp.fresh_sender_count, 2);
        assert_eq!(g_exp.num_fused, 0, "Must NEVER admit unguarded 2-ship coplanar ghosts after 500 ms");
        assert!(g_exp.sticky_ghosts_suppressed >= 2);
        assert!(g_exp.fallback_to_tier_b_optical);

        // 6. Uncalibrated peer during GNSS-denied forces fallback_to_tier_b_optical == true & 0 ghosts
        sync.invalidate_peer(3);
        let obs_uncal = [
            mk_obs(1, 1, p_s1, t1, now_drop + 100_000),
            mk_obs(1, 2, p_s1, t2, now_drop + 100_000),
            mk_obs(2, 1, p_s2, t1, (now_drop + 100_000).wrapping_add_signed(drift_p2)),
            mk_obs(2, 2, p_s2, t2, (now_drop + 100_000).wrapping_add_signed(drift_p2)),
            mk_obs(3, 1, p_s3, t1, (now_drop + 100_000).wrapping_add_signed(drift_p3)),
        ];
        let g_uncal = guard.fuse_guarded_with_clock_sync(&obs_uncal, now_drop + 110_000, 1, true, &sync);
        assert_eq!(g_uncal.uncalibrated_obs_rejected, 1);
        assert_eq!(g_uncal.num_fused, 0);
        assert!(g_uncal.fallback_to_tier_b_optical);

        // 7. Bounded offset slew cannot re-accept previously rejected replay packet!
        let mut sync_slew = PeerClockOffsetEstimator::new();
        let mut guard_slew = StatefulMeshDaaGuard::new();
        for k in 0..5u64 {
            let t1_k = now_base + k * 100_000;
            assert!(sync_slew.record_exchange(mk_exchange(2, t1_k, 120_000, 5_000, 2_000, 5_000)).accepted);
        }
        let t_obs1 = now_base + 500_000;
        let raw_p1 = (t_obs1 - 10_000).wrapping_add_signed(120_000);
        let raw_old = raw_p1 - 3_000; // 3 ms older raw timestamp
        let g1 = guard_slew.fuse_guarded_with_clock_sync(&[mk_obs(2, 1, p_s2, t1, raw_p1)], t_obs1, 1, true, &sync_slew);
        assert_eq!(g1.fresh_sender_count, 1);
        let g_rep_before = guard_slew.fuse_guarded_with_clock_sync(&[mk_obs(2, 1, p_s2, t1, raw_old)], t_obs1 + 5_000, 1, true, &sync_slew);
        assert_eq!(g_rep_before.replay_obs_rejected, 1);
        // Slew offset downward by -5 ms (120 ms -> 115 ms, targeting 110 ms across 3x 80 ms syncs)
        // so aligned_old jumps FORWARD by +2 ms past aligned_p1 while staying synchronized & in freshness window!
        for k in 0..3u64 {
            let t1_k = t_obs1 + 10_000 + k * 80_000;
            assert!(
                sync_slew
                    .record_exchange(mk_exchange(2, t1_k, 110_000, 5_000, 2_000, 5_000))
                    .accepted
            );
        }
        let t_after_slew = t_obs1 + 190_000;
        let est_after = sync_slew.estimate_peer(2, t_after_slew);
        assert!(est_after.synchronized);
        assert_eq!(est_after.offset_us, 115_000, "Must have slewed downward by -5 ms");
        // Verify aligned_old > aligned_p1 and still inside [now - 200 ms, now + 50 ms]
        let g_rep_after = guard_slew.fuse_guarded_with_clock_sync(
            &[mk_obs(2, 1, p_s2, t1, raw_old)],
            t_after_slew,
            1,
            true,
            &sync_slew,
        );
        assert_eq!(g_rep_after.fresh_sender_count, 0);
        assert_eq!(
            g_rep_after.replay_obs_rejected, 1,
            "Downward offset slew must NEVER allow previously rejected replay timestamp to pass!"
        );
        assert_eq!(g_rep_after.uncalibrated_obs_rejected, 0);
        assert_eq!(g_rep_after.stale_obs_rejected, 0);
    }
}
