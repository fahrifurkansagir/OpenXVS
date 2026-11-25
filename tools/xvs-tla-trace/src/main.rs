// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Standalone Zero-Modification Rust <-> TLA+ NDJSON Trace Emitter (`tools/xvs-tla-trace`).
//!
//! Executes >= 12,000 deterministic adversarial ticks directly against the FOUR FROZEN
//! `#![no_std]` OpenXVS safety crates (`xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`,
//! `xvs-os-hal`) plus `xvs-math-core::mesh_daa`, emits a canonical NDJSON execution trace
//! for `formal/tla/OpenXvsTraceRefinement.tla` (`tlc2.TLC`), and records empirical multi-ship
//! livelock / starvation / ghost-oscillation metrics for DR-22 & DASC 2027.
//!
//! Strictly civil/dual-use aviation safety (Part 91/107/108/135, EASA SORA 2.5, ED-324).

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use xvs_active_range::{
    wrap_pi, ActiveRangeConfig, ActiveRangeController, ActiveRangeMeasurement, ActiveRangeMode,
    ActiveRangeTelemetry, OwnshipStateNed, FLAG_FOV_SAFE, FLAG_SC3BF_ALL_SAFE, G_MPS2,
    LOCK_REL_SIGMA, PHI_CAND_RAD, R_FLOOR_M, TRIGGER_REL_SIGMA,
};
use xvs_math_core::mesh_daa::{MeshDaaFuser, OpenXvsBearingTrack};
use xvs_multiship_coord::{
    CoordinatedActiveRange, PeerCoordHint, Ring0AuthorityGate, StatefulMeshDaaGuard,
    SwerveCoordRole, HEAD_ON_ALPHA_BAND_RAD,
};
use xvs_odd_vlm::{
    AsymmetricSimplexVlmGate, OpticalOddMonitor, OpticalOddSensors, VlmHazard, VlmWirePacket,
    UNUSED_ZONE_ID,
};
use xvs_os_hal::{
    CivilMissionClass, PluginLifecycle, PluginSupervisorState, Ring1AdvisoryPacket,
    SimplexMode as HalSimplexMode, DEFAULT_MAX_HEARTBEAT_MISS_MS, RING1_ADVISORY_MAGIC_VER,
};
use xvs_rta_core::{
    RtaTickInput, RtaTickOutput, SimplexMode, SimplexRta, HORIZON_AL_URAD, V_HAL_MM, V_VAL_MM,
};

#[derive(Debug, Clone)]
struct TraceRecord {
    step: u64,
    scenario: &'static str,
    tick: u32,
    ship_id: u8,
    rta_mode: u8,
    rta_raw_target: u8,
    rta_consec_drop: u32,
    rta_win_drop_20: u8,
    rta_rec_ticks: u32,
    rta_dwell_ticks: u32,
    rta_go_around: bool,
    rta_approach_active: bool,
    rta_go_around_ack: bool,
    rta_transient_drop: bool,
    rta_inhibit_turns: bool,
    rta_blank_hud: bool,
    rta_p0_healthy: bool,
    ar_mode: u8,
    ar_was_locked: bool,
    ar_swerve_ticks: u32,
    ar_pref_sign: i32,
    ar_sigma_bucket: &'static str,
    ar_sc3bf_all_safe: bool,
    ar_fov_safe: bool,
    ar_bank_within_12deg: bool,
    ar_bank_sign: i32,
    vlm_status: u8,
    vlm_act_auth: bool,
    vlm_cert_mask: u8,
    vlm_final_mask: u8,
    odd_airdata_blocked: bool,
    hal_lifecycle: u8,
    hal_base_mode_wire: u8,
    hal_out_mode_wire: u8,
    hal_act_auth: bool,
    arbiter_bank_sign: i32,
    coord_enabled: bool,
    coord_head_on: bool,
    coord_role: &'static str,
    mesh_guard_ghosts_admitted: u32,
    mesh_guard_sticky_active: bool,
    mesh_guard_stale_rejected: u32,
    mesh_guard_replay_rejected: u32,
}

impl TraceRecord {
    fn write_json_line<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "{{\"step\":{},\"scenario\":\"{}\",\"tick\":{},\"ship_id\":{},\
            \"rta_mode\":{},\"rta_raw_target\":{},\"rta_consec_drop\":{},\"rta_win_drop_20\":{},\
            \"rta_rec_ticks\":{},\"rta_dwell_ticks\":{},\"rta_go_around\":{},\
            \"rta_approach_active\":{},\"rta_go_around_ack\":{},\"rta_transient_drop\":{},\
            \"rta_inhibit_turns\":{},\"rta_blank_hud\":{},\"rta_p0_healthy\":{},\
            \"ar_mode\":{},\"ar_was_locked\":{},\"ar_swerve_ticks\":{},\"ar_pref_sign\":{},\
            \"ar_sigma_bucket\":\"{}\",\"ar_sc3bf_all_safe\":{},\"ar_fov_safe\":{},\
            \"ar_bank_within_12deg\":{},\"ar_bank_sign\":{},\
            \"vlm_status\":{},\"vlm_act_auth\":{},\"vlm_cert_mask\":{},\"vlm_final_mask\":{},\
            \"odd_airdata_blocked\":{},\
            \"hal_lifecycle\":{},\"hal_base_mode_wire\":{},\"hal_out_mode_wire\":{},\
            \"hal_act_auth\":{},\"arbiter_bank_sign\":{},\
            \"coord_enabled\":{},\"coord_head_on\":{},\"coord_role\":\"{}\",\
            \"mesh_guard_ghosts_admitted\":{},\"mesh_guard_sticky_active\":{},\
            \"mesh_guard_stale_rejected\":{},\"mesh_guard_replay_rejected\":{}}}",
            self.step,
            self.scenario,
            self.tick,
            self.ship_id,
            self.rta_mode,
            self.rta_raw_target,
            self.rta_consec_drop,
            self.rta_win_drop_20,
            self.rta_rec_ticks,
            self.rta_dwell_ticks,
            self.rta_go_around,
            self.rta_approach_active,
            self.rta_go_around_ack,
            self.rta_transient_drop,
            self.rta_inhibit_turns,
            self.rta_blank_hud,
            self.rta_p0_healthy,
            self.ar_mode,
            self.ar_was_locked,
            self.ar_swerve_ticks,
            self.ar_pref_sign,
            self.ar_sigma_bucket,
            self.ar_sc3bf_all_safe,
            self.ar_fov_safe,
            self.ar_bank_within_12deg,
            self.ar_bank_sign,
            self.vlm_status,
            self.vlm_act_auth,
            self.vlm_cert_mask,
            self.vlm_final_mask,
            self.odd_airdata_blocked,
            self.hal_lifecycle,
            self.hal_base_mode_wire,
            self.hal_out_mode_wire,
            self.hal_act_auth,
            self.arbiter_bank_sign,
            self.coord_enabled,
            self.coord_head_on,
            self.coord_role,
            self.mesh_guard_ghosts_admitted,
            self.mesh_guard_sticky_active,
            self.mesh_guard_stale_rejected,
            self.mesh_guard_replay_rejected,
        )
    }
}

/// Empirical findings collected directly from executing the frozen Rust crates.
#[derive(Debug, Default)]
struct EmpiricalFindings {
    total_ticks: u64,
    scenario_counts: Vec<(&'static str, u64)>,
    // Finding F1: Reciprocal head-on synchronous vs 12-tick asynchronous phase-offset ActiveRange
    headon_sync_ship1_final_r_hat_m: f64,
    headon_sync_ship1_final_r_true_m: f64,
    headon_sync_ship1_rel_err: f64,
    headon_async12_ship1_peak_r_hat_m: f64,
    headon_async12_ship1_r_true_at_peak_m: f64,
    headon_async12_overestimate_ratio: f64,
    headon_async12_cancelled_ticks: u32,
    // Finding F2: Intermittent dropout starvation & chattering immunity in xvs-rta-core
    starvation_5pct_ticks_tested: u32,
    starvation_5pct_p1_to_p2_transitions: u32,
    starvation_5pct_recovered_to_p4: bool,
    starvation_5pct_tripped_p0: bool,
    sawtooth_21tick_p4_cycles: u32,
    exact_p0_to_p1_ticks_after_dropout11: u32,
    exact_p0_to_p1_ticks_after_horizon_fault: u32,
    exact_p1_to_p4_ticks: u32,
    exact_p2_to_p1_dwell_observed: u32,
    // Finding F3: Cross-layer authority & absorbing PluginOffline
    p0_overrode_immediate_evade_ticks: u32,
    plugin_offline_post_latch_attempts: u32,
    plugin_offline_stayed_latched: bool,
    go_around_approach_reset_attempts_blocked: u32,
    // Finding F4: 3-ship XVS-MeshDAA ghost gate under 3 <-> 2 peer drop oscillation
    meshdaa_3ship_ghosts_rejected: u32,
    meshdaa_2ship_dropout_ghosts_admitted: u32,
    meshdaa_3ship_true_tracks_confirmed: u32,
    // Coordinated xvs-multiship-coord (F1 + F4 + F3) verification metrics
    frozen_ticks: u64,
    coordinated_ticks: u64,
    coord_optical_async12_cancelled_ticks: u32,
    coord_optical_async12_peak_ratio: f64,
    coord_optical_async12_final_r_hat_m: f64,
    coord_optical_async12_final_r_true_m: f64,
    coord_meshdaa_stagger_cancelled_ticks: u32,
    coord_meshdaa_yield_ticks: u32,
    coord_mesh_guard_3ship_ghosts_rejected: u32,
    coord_mesh_guard_2ship_ghosts_admitted: u32,
    coord_mesh_guard_sticky_suppressed: u32,
    coord_mesh_guard_stale_rejected: u32,
    coord_mesh_guard_replay_rejected: u32,
    coord_p0_inhibited_ticks: u32,
    coord_headon_non_reversing_swerve_ticks: u32,
}

#[inline]
fn rta_mode_to_hal(mode: SimplexMode) -> HalSimplexMode {
    match mode {
        SimplexMode::NominalMl => HalSimplexMode::NominalMl,
        SimplexMode::Isolated5Cam => HalSimplexMode::Isolated5Cam,
        SimplexMode::DegradedCoast => HalSimplexMode::DegradedCoast,
        SimplexMode::FallbackD0Safe => HalSimplexMode::FallbackD0Safe,
        SimplexMode::InvalidInhibit => HalSimplexMode::InvalidInhibit,
    }
}

#[inline]
fn sign_i32(v: f32) -> i32 {
    if v > 1.0e-6 {
        1
    } else if v < -1.0e-6 {
        -1
    } else {
        0
    }
}

#[inline]
fn arbiter_bank(rta_out: &RtaTickOutput, hal_out_wire: u8, ar_telem: &ActiveRangeTelemetry) -> i32 {
    if rta_out.inhibit_avoidance_turns || hal_out_wire == 0 {
        0
    } else if ar_telem.mode == ActiveRangeMode::ImmediateEvade as u8 {
        2 * sign_i32(ar_telem.cmd_bank_rad)
    } else if ar_telem.mode == ActiveRangeMode::ActiveSwerve as u8 {
        sign_i32(ar_telem.cmd_bank_rad)
    } else {
        0
    }
}

/// Shadow observer for private ActiveRangeController fields (`swerve_ticks`, `was_locked`)
/// verified strictly against the public telemetry output on every tick.
struct ActiveRangeObserver {
    swerve_ticks: u32,
    was_locked: bool,
}

impl ActiveRangeObserver {
    fn new() -> Self {
        Self {
            swerve_ticks: 0,
            was_locked: false,
        }
    }

    fn observe(&mut self, telem: &ActiveRangeTelemetry) -> (i32, u32, bool, &'static str) {
        let pref_sign = if (self.swerve_ticks / 12).is_multiple_of(2) {
            1
        } else {
            -1
        };

        let r_hat = (telem.r_hat_m as f64).max(R_FLOOR_M);
        let sig = telem.sigma_r_m as f64;
        let lock_thr = LOCK_REL_SIGMA * r_hat;
        let trig_thr = TRIGGER_REL_SIGMA * r_hat;

        // Classify sigma bucket and update was_locked faithfully to sc3bf_fim.rs:337-345.
        // Guard rare f64->f32 boundary rounding using the authoritative mode verdict.
        let mut now_locked = if sig.is_finite() && sig >= 0.0 {
            if self.was_locked {
                sig <= trig_thr
            } else {
                sig <= lock_thr
            }
        } else {
            false
        };
        let mut bucket = if sig <= lock_thr {
            "LOCKED"
        } else if sig > trig_thr {
            "HIGH_SIGMA"
        } else {
            "HYST_BAND"
        };

        if telem.mode == ActiveRangeMode::Locked as u8 {
            if !now_locked {
                // Only permitted within f32 quantization margin at exact threshold
                assert!(
                    (sig / r_hat - LOCK_REL_SIGMA).abs() < 1e-4
                        || (sig / r_hat - TRIGGER_REL_SIGMA).abs() < 1e-4,
                    "Unexpected lock mismatch: sig={sig}, r_hat={r_hat}"
                );
                now_locked = true;
                bucket = "LOCKED";
            }
        } else if (telem.mode == ActiveRangeMode::ActiveSwerve as u8
            || telem.mode == ActiveRangeMode::NominalHold as u8)
            && now_locked
        {
            assert!(
                (sig / r_hat - LOCK_REL_SIGMA).abs() < 1e-4
                    || (sig / r_hat - TRIGGER_REL_SIGMA).abs() < 1e-4,
                "Unexpected unlock mismatch: sig={sig}, r_hat={r_hat}"
            );
            now_locked = false;
            bucket = "HIGH_SIGMA";
        }

        if telem.mode == ActiveRangeMode::ActiveSwerve as u8 {
            self.swerve_ticks = self.swerve_ticks.wrapping_add(1);
        } else {
            self.swerve_ticks = 0;
        }
        self.was_locked = now_locked;
        (pref_sign, self.swerve_ticks, now_locked, bucket)
    }

    fn observe_coordinated(
        &mut self,
        telem: &ActiveRangeTelemetry,
        coord_swerve_ticks: u32,
        head_on: bool,
        role: SwerveCoordRole,
    ) -> (i32, u32, bool, &'static str) {
        let pref_sign = if head_on || (self.swerve_ticks / 12).is_multiple_of(2) {
            1
        } else {
            -1
        };

        let r_hat = (telem.r_hat_m as f64).max(R_FLOOR_M);
        let sig = telem.sigma_r_m as f64;
        let lock_thr = LOCK_REL_SIGMA * r_hat;
        let trig_thr = TRIGGER_REL_SIGMA * r_hat;

        let mut now_locked = if sig.is_finite() && sig >= 0.0 {
            if self.was_locked {
                sig <= trig_thr
            } else {
                sig <= lock_thr
            }
        } else {
            false
        };
        let mut bucket = if sig <= lock_thr {
            "LOCKED"
        } else if sig > trig_thr {
            "HIGH_SIGMA"
        } else {
            "HYST_BAND"
        };

        let overridden_to_hold = matches!(
            role,
            SwerveCoordRole::InhibitedByP0 | SwerveCoordRole::YieldHoldForPeerProbe
        );
        if !overridden_to_hold {
            if telem.mode == ActiveRangeMode::Locked as u8 {
                if !now_locked {
                    assert!(
                        (sig / r_hat - LOCK_REL_SIGMA).abs() < 1e-4
                            || (sig / r_hat - TRIGGER_REL_SIGMA).abs() < 1e-4,
                        "Unexpected coordinated lock mismatch: sig={sig}, r_hat={r_hat}"
                    );
                    now_locked = true;
                    bucket = "LOCKED";
                }
            } else if (telem.mode == ActiveRangeMode::ActiveSwerve as u8
                || telem.mode == ActiveRangeMode::NominalHold as u8)
                && now_locked
            {
                assert!(
                    (sig / r_hat - LOCK_REL_SIGMA).abs() < 1e-4
                        || (sig / r_hat - TRIGGER_REL_SIGMA).abs() < 1e-4,
                    "Unexpected coordinated unlock mismatch: sig={sig}, r_hat={r_hat}"
                );
                now_locked = false;
                bucket = "HIGH_SIGMA";
            }
        }

        if telem.mode == ActiveRangeMode::ActiveSwerve as u8 {
            self.swerve_ticks = self.swerve_ticks.wrapping_add(1);
        } else {
            self.swerve_ticks = 0;
        }
        assert_eq!(
            self.swerve_ticks, coord_swerve_ticks,
            "CoordinatedActiveRange swerve_ticks diverged from TLA+ transition rule"
        );
        self.was_locked = now_locked;
        (pref_sign, self.swerve_ticks, now_locked, bucket)
    }
}

#[inline]
fn coord_role_str(role: SwerveCoordRole) -> &'static str {
    match role {
        SwerveCoordRole::InhibitedByP0 => "InhibitedByP0",
        SwerveCoordRole::YieldHoldForPeerProbe => "YieldHoldForPeerProbe",
        SwerveCoordRole::CooperativeRightProbe => "CooperativeRightProbe",
        SwerveCoordRole::OpticalFar91113Right => "OpticalFar91113Right",
    }
}

fn nominal_odd_sensors() -> OpticalOddSensors {
    OpticalOddSensors {
        droplet_occ_permille: 10,
        blur_mlap_centi: 14_000,
        mtf_hf_ratio_permille: 450,
        lwir_std_centidn: 2100,
        dew_margin_centic: 700,
        sun_boresight_cdeg: 9500,
        eo_hfov_cdeg: 6000,
        glare_sat_permille: 2,
        glare_ring_ratio_permille: 850,
        mor_m: 10_000,
        horizon_snr_centidb: 900,
        lwir_median_drift_dn: 20,
        lwir_fpn_ratio_permille: 40,
        lwir_bad_px_ppm: 100,
        boson_dt_ffc_s: 30,
        boson_temp_delta_centic: 15,
        boson_ffc_freeze: false,
        attitude_in_odd: true,
    }
}

fn make_ring1_pkt(seq_no: u64, ts_ns: u64, odd_tighten_mask: u32) -> Ring1AdvisoryPacket {
    Ring1AdvisoryPacket {
        magic_ver: RING1_ADVISORY_MAGIC_VER,
        civil_mission_class: CivilMissionClass::SearchAndRescue as u16,
        advisory_flags: 0x0001,
        plugin_id_hash: 0x0AE5_1122_3344_5566,
        seq_no,
        timestamp_mono_ns: ts_ns,
        odd_tighten_mask,
        roi_lat_e7: 376_117_120,
        roi_lon_e7: -1_223_583_492,
        roi_alt_mm: 150_000,
        confidence_q15: 28_000,
        reserved_zero: [0u8; 10],
        ieee_crc32: 0,
    }
    .with_valid_crc()
}

fn make_vlm_pkt(seq: u32, ts_ms: u64, hazard: VlmHazard, conf_valid: u8) -> VlmWirePacket {
    VlmWirePacket {
        seq,
        capture_ts_ms: ts_ms,
        hazard_enum: hazard as u8,
        azimuth_sector: 3,
        ranked_zone_ids: [2, 1, 0, UNUSED_ZONE_ID],
        conformal_valid: conf_valid,
        aleatoric_flag: 0,
        epistemic_flag: 0,
        crc32: 0,
    }
}

struct ShipKinematics {
    p: [f64; 3],
    hdg: f64,
    tas: f64,
    a: [f64; 3],
}

impl ShipKinematics {
    fn ownship_state(&self, v_stall: f64) -> OwnshipStateNed {
        OwnshipStateNed {
            v_ned: [self.tas * self.hdg.cos(), self.tas * self.hdg.sin(), 0.0],
            a_ned: self.a,
            heading_rad: self.hdg,
            tas_mps: self.tas,
            v_stall_mps: v_stall,
        }
    }

    fn advance(&mut self, cmd_bank_rad: f64, dt: f64) {
        let a_lat = G_MPS2 * cmd_bank_rad.tan();
        let psi_dot = a_lat / self.tas;
        let right = [-self.hdg.sin(), self.hdg.cos(), 0.0];
        self.a = [a_lat * right[0], a_lat * right[1], 0.0];
        let hdg_mid = self.hdg + 0.5 * psi_dot * dt;
        self.p[0] += self.tas * hdg_mid.cos() * dt;
        self.p[1] += self.tas * hdg_mid.sin() * dt;
        self.hdg = wrap_pi(self.hdg + psi_dot * dt);
    }
}

fn measure_intruder(
    own: &ShipKinematics,
    intr: &ShipKinematics,
    prev_s: &mut Option<f64>,
    track_age: u32,
    dt: f64,
    bearing_noise_rad: f64,
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
    let alpha = wrap_pi(rel[1].atan2(rel[0]) + bearing_noise_rad);
    let beta = (-rel[2]).atan2(horiz);
    let focal_px = 2690.0_f64;
    let wingspan_m = 10.0_f64;
    let s_now = focal_px * wingspan_m / r_true;
    let s_rate = match *prev_s {
        None => {
            // Exact initial analytical optical expansion rate
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

#[allow(clippy::too_many_arguments)]
fn build_record(
    step: u64,
    scenario: &'static str,
    tick: u32,
    ship_id: u8,
    rta_inp: &RtaTickInput,
    rta_out: &RtaTickOutput,
    ar_telem: &ActiveRangeTelemetry,
    ar_obs: &mut ActiveRangeObserver,
    vlm_status: u8,
    vlm_act_auth: bool,
    vlm_cert_mask: u8,
    vlm_final_mask: u8,
    odd_airdata_blocked: bool,
    hal_lifecycle: u8,
    hal_base_mode_wire: u8,
    hal_out_mode_wire: u8,
    hal_act_auth: bool,
) -> TraceRecord {
    let (ar_pref_sign, ar_swerve_ticks, ar_was_locked, ar_sigma_bucket) = ar_obs.observe(ar_telem);
    let p0_healthy = matches!(
        rta_out.raw_target,
        SimplexMode::FallbackD0Safe | SimplexMode::Isolated5Cam | SimplexMode::NominalMl
    ) && !rta_inp.transient_dropout;
    let ar_bank_within_12 = (ar_telem.cmd_bank_rad as f64).abs() <= PHI_CAND_RAD[8] + 1.0e-5;
    let ar_sc3bf_all_safe = (ar_telem.flags & FLAG_SC3BF_ALL_SAFE) != 0;
    let ar_fov_safe = (ar_telem.flags & FLAG_FOV_SAFE) != 0;
    let arb_sign = arbiter_bank(rta_out, hal_out_mode_wire, ar_telem);

    TraceRecord {
        step,
        scenario,
        tick,
        ship_id,
        rta_mode: rta_out.mode as u8,
        rta_raw_target: rta_out.raw_target as u8,
        rta_consec_drop: rta_out.consec_dropout_ticks,
        rta_win_drop_20: rta_out.window_dropout_20,
        rta_rec_ticks: rta_out.recovery_ticks,
        rta_dwell_ticks: rta_out.dwell_ticks,
        rta_go_around: rta_out.mandatory_go_around,
        rta_approach_active: rta_inp.approach_active,
        rta_go_around_ack: rta_inp.go_around_ack_reset,
        rta_transient_drop: rta_inp.transient_dropout,
        rta_inhibit_turns: rta_out.inhibit_avoidance_turns,
        rta_blank_hud: rta_out.blank_conformal_hud,
        rta_p0_healthy: p0_healthy,
        ar_mode: ar_telem.mode,
        ar_was_locked,
        ar_swerve_ticks,
        ar_pref_sign,
        ar_sigma_bucket,
        ar_sc3bf_all_safe,
        ar_fov_safe,
        ar_bank_within_12deg: ar_bank_within_12,
        ar_bank_sign: sign_i32(ar_telem.cmd_bank_rad),
        vlm_status,
        vlm_act_auth,
        vlm_cert_mask,
        vlm_final_mask,
        odd_airdata_blocked,
        hal_lifecycle,
        hal_base_mode_wire,
        hal_out_mode_wire,
        hal_act_auth,
        arbiter_bank_sign: arb_sign,
        coord_enabled: false,
        coord_head_on: false,
        coord_role: "FrozenUncoord",
        mesh_guard_ghosts_admitted: 0,
        mesh_guard_sticky_active: false,
        mesh_guard_stale_rejected: 0,
        mesh_guard_replay_rejected: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_coord_record(
    step: u64,
    scenario: &'static str,
    tick: u32,
    ship_id: u8,
    rta_inp: &RtaTickInput,
    rta_out: &RtaTickOutput,
    ar_telem: &ActiveRangeTelemetry,
    ar_obs: &mut ActiveRangeObserver,
    coord_swerve_ticks: u32,
    coord_head_on: bool,
    coord_role: SwerveCoordRole,
    vlm_status: u8,
    vlm_act_auth: bool,
    vlm_cert_mask: u8,
    vlm_final_mask: u8,
    odd_airdata_blocked: bool,
    hal_lifecycle: u8,
    hal_base_mode_wire: u8,
    hal_out_mode_wire: u8,
    hal_act_auth: bool,
    mesh_guard_ghosts_admitted: u32,
    mesh_guard_sticky_active: bool,
    mesh_guard_stale_rejected: u32,
    mesh_guard_replay_rejected: u32,
) -> TraceRecord {
    let (ar_pref_sign, ar_swerve_ticks, ar_was_locked, ar_sigma_bucket) =
        ar_obs.observe_coordinated(ar_telem, coord_swerve_ticks, coord_head_on, coord_role);
    let p0_healthy = matches!(
        rta_out.raw_target,
        SimplexMode::FallbackD0Safe | SimplexMode::Isolated5Cam | SimplexMode::NominalMl
    ) && !rta_inp.transient_dropout;
    let ar_bank_within_12 = (ar_telem.cmd_bank_rad as f64).abs() <= PHI_CAND_RAD[8] + 1.0e-5;
    let ar_sc3bf_all_safe = (ar_telem.flags & FLAG_SC3BF_ALL_SAFE) != 0;
    let ar_fov_safe = (ar_telem.flags & FLAG_FOV_SAFE) != 0;
    let arb_sign = arbiter_bank(rta_out, hal_out_mode_wire, ar_telem);

    TraceRecord {
        step,
        scenario,
        tick,
        ship_id,
        rta_mode: rta_out.mode as u8,
        rta_raw_target: rta_out.raw_target as u8,
        rta_consec_drop: rta_out.consec_dropout_ticks,
        rta_win_drop_20: rta_out.window_dropout_20,
        rta_rec_ticks: rta_out.recovery_ticks,
        rta_dwell_ticks: rta_out.dwell_ticks,
        rta_go_around: rta_out.mandatory_go_around,
        rta_approach_active: rta_inp.approach_active,
        rta_go_around_ack: rta_inp.go_around_ack_reset,
        rta_transient_drop: rta_inp.transient_dropout,
        rta_inhibit_turns: rta_out.inhibit_avoidance_turns,
        rta_blank_hud: rta_out.blank_conformal_hud,
        rta_p0_healthy: p0_healthy,
        ar_mode: ar_telem.mode,
        ar_was_locked,
        ar_swerve_ticks,
        ar_pref_sign,
        ar_sigma_bucket,
        ar_sc3bf_all_safe,
        ar_fov_safe,
        ar_bank_within_12deg: ar_bank_within_12,
        ar_bank_sign: sign_i32(ar_telem.cmd_bank_rad),
        vlm_status,
        vlm_act_auth,
        vlm_cert_mask,
        vlm_final_mask,
        odd_airdata_blocked,
        hal_lifecycle,
        hal_base_mode_wire,
        hal_out_mode_wire,
        hal_act_auth,
        arbiter_bank_sign: arb_sign,
        coord_enabled: true,
        coord_head_on,
        coord_role: coord_role_str(coord_role),
        mesh_guard_ghosts_admitted,
        mesh_guard_sticky_active,
        mesh_guard_stale_rejected,
        mesh_guard_replay_rejected,
    }
}

/// Scenario 1 (2,000 ticks): 2-Ship Reciprocal Head-On CBDR (Synchronous vs 12-Tick Async Phase Offset)
fn run_scenario_1_reciprocal_headon(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let dt = 0.1_f64;

    // Empirical Physics Sub-experiment A: Synchronous 2-ship head-on (both start ActiveSwerve at t=0)
    {
        let mut s1 = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: 0.0,
            tas: 35.0,
            a: [0.0; 3],
        };
        let mut s2 = ShipKinematics {
            p: [1200.0, 0.0, -500.0],
            hdg: std::f64::consts::PI,
            tas: 35.0,
            a: [0.0; 3],
        };
        let mut ar1 = ActiveRangeController::new(ActiveRangeConfig::default());
        let mut ar2 = ActiveRangeController::new(ActiveRangeConfig::default());
        let mut ps1 = None;
        let mut ps2 = None;
        for t in 0..35u32 {
            let (m1, r_true) = measure_intruder(&s1, &s2, &mut ps1, 6 + t, dt, 0.0);
            let (m2, _) = measure_intruder(&s2, &s1, &mut ps2, 6 + t, dt, 0.0);
            let t1 = ar1.step(&s1.ownship_state(14.5), &m1);
            let t2 = ar2.step(&s2.ownship_state(14.5), &m2);
            if t == 34 {
                findings.headon_sync_ship1_final_r_hat_m = t1.r_hat_m as f64;
                findings.headon_sync_ship1_final_r_true_m = r_true;
                findings.headon_sync_ship1_rel_err =
                    ((t1.r_hat_m as f64) - r_true).abs() / r_true;
            }
            s1.advance(t1.cmd_bank_rad as f64, dt);
            s2.advance(t2.cmd_bank_rad as f64, dt);
        }
    }

    // Empirical Physics Sub-experiment B: Asynchronous 12-tick phase offset head-on CBDR!
    // Ship 2 was already in ActiveSwerve for 12 ticks (pref_sign = -1 -> banks LEFT = +East),
    // while Ship 1 starts ActiveSwerve at tick 0 (pref_sign = +1 -> banks RIGHT = +East)!
    // Both accelerate EAST simultaneously -> a_rel_E = a_2 - a_1 ≈ 0 -> MSC-RPEKF (a_int=0)
    // experiences severe observability cancellation & unconservative far-range overestimate!
    {
        let mut ar2 = ActiveRangeController::new(ActiveRangeConfig::default());
        // Prime Ship 2 through 12 straight head-on swerve ticks so internal swerve_ticks == 12
        let prime_own = OwnshipStateNed {
            v_ned: [-35.0, 0.0, 0.0],
            a_ned: [0.0; 3],
            heading_rad: std::f64::consts::PI,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let prime_meas = ActiveRangeMeasurement {
            alpha_rad: std::f64::consts::PI,
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

        // Now run head-on encounter at fast GA cruise (52 m/s ~ 101 kt, keeping |alpha_body| < 0.07)
        let mut s1 = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: 0.0,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut s2 = ShipKinematics {
            p: [1200.0, 0.0, -500.0],
            hdg: std::f64::consts::PI,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut ar1 = ActiveRangeController::new(ActiveRangeConfig::default());
        let mut ps1 = None;
        let mut ps2 = None;
        let mut peak_ratio = 0.0_f64;
        for t in 0..30u32 {
            let (m1, r_true) = measure_intruder(&s1, &s2, &mut ps1, 6 + t, dt, 0.0);
            let (m2, _) = measure_intruder(&s2, &s1, &mut ps2, 18 + t, dt, 0.0);
            let t1 = ar1.step(&s1.ownship_state(14.5), &m1);
            let t2 = ar2.step(&s2.ownship_state(14.5), &m2);
            // Opposite body-bank signs head-on => SAME inertial East acceleration!
            if sign_i32(t1.cmd_bank_rad) != 0
                && sign_i32(t1.cmd_bank_rad) == -sign_i32(t2.cmd_bank_rad)
            {
                findings.headon_async12_cancelled_ticks += 1;
            }
            let ratio = (t1.r_hat_m as f64) / r_true;
            if ratio > peak_ratio {
                peak_ratio = ratio;
                findings.headon_async12_ship1_peak_r_hat_m = t1.r_hat_m as f64;
                findings.headon_async12_ship1_r_true_at_peak_m = r_true;
                findings.headon_async12_overestimate_ratio = ratio;
            }
            s1.advance(t1.cmd_bank_rad as f64, dt);
            s2.advance(t2.cmd_bank_rad as f64, dt);
        }
    }

    // Emit 2,000 full-stack NDJSON ticks (1,000 for Ship 1, 1,000 for Ship 2)
    let mut emitted = 0u64;
    for ship_id in 1u8..=2u8 {
        let mut rta = SimplexRta::new();
        let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
        let mut ar_obs = ActiveRangeObserver::new();
        let mut odd = OpticalOddMonitor::new_canonical_60hz();
        let mut vlm = AsymmetricSimplexVlmGate::new(200);
        let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

        let mut own = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: if ship_id == 1 { 0.0 } else { std::f64::consts::PI },
            tas: 42.0,
            a: [0.0; 3],
        };
        let mut intr = ShipKinematics {
            p: [if ship_id == 1 { 1500.0 } else { -1500.0 }, 0.0, -500.0],
            hdg: if ship_id == 1 { std::f64::consts::PI } else { 0.0 },
            tas: 42.0,
            a: [0.0; 3],
        };
        let mut prev_s = None;

        for tick in 0..1000u32 {
            // Re-seed encounter geometry every 100 ticks to exercise repeated CBDR acquisitions
            if tick > 0 && tick % 100 == 0 {
                own.p = [0.0, 0.0, -500.0];
                own.hdg = if ship_id == 1 { 0.0 } else { std::f64::consts::PI };
                own.a = [0.0; 3];
                intr.p = [if ship_id == 1 { 1500.0 } else { -1500.0 }, 0.0, -500.0];
                intr.hdg = if ship_id == 1 { std::f64::consts::PI } else { 0.0 };
                prev_s = None;
            }
            let rta_inp = RtaTickInput::default();
            let rta_out = rta.step(&rta_inp);
            let (meas, _) = measure_intruder(&own, &intr, &mut prev_s, 6 + (tick % 100), dt, 0.0);
            let ar_telem = ar.step(&own.ownship_state(14.5), &meas);
            let odd_out = odd.evaluate(&nominal_odd_sensors());
            let vlm_pkt = make_vlm_pkt(tick + 1, (tick as u64) * 100 + 50, VlmHazard::None, 1);
            let vlm_dec = vlm.evaluate_wire(
                Some(&vlm_pkt.encode()),
                (tick as u64) * 100 + 80,
                !odd_out.vlm_gate_reject,
                0x04,
                0x0007,
            );
            let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, 0);
            let hal_dec = hal.ingest_wire(
                &hal_pkt.to_le_bytes(),
                (tick as u64) * 100_000_000 + 10_000_000,
                rta_mode_to_hal(rta_out.mode),
                0,
            );

            let step_idx = (records.len() + 1) as u64;
            records.push(build_record(
                step_idx,
                "reciprocal_headon_cbdr",
                tick,
                ship_id,
                &rta_inp,
                &rta_out,
                &ar_telem,
                &mut ar_obs,
                vlm_dec.status as u8,
                vlm_dec.actuator_authority,
                vlm_dec.certified_hazards_mask,
                vlm_dec.final_hazards_mask,
                odd_out.airdata_export_blocked,
                hal_dec.lifecycle as u8,
                rta_out.mode as u8,
                hal_dec.out_mode.to_rta_wire_u8(),
                hal_dec.actuator_authority,
            ));
            own.advance(ar_telem.cmd_bank_rad as f64, dt);
            intr.advance(0.0, dt);
            emitted += 1;
        }
    }
    findings
        .scenario_counts
        .push(("reciprocal_headon_cbdr", emitted));
}

/// Scenario 2 (2,100 ticks): 3-Ship Symmetric Converging Encounter + XVS-MeshDAA Ghost Gate
fn run_scenario_2_threeship_meshdaa(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    // Exercise REAL xvs_math_core::mesh_daa::MeshDaaFuser under 3-ship vs 2-ship peer drop!
    // Coplanar geometry: Ship 1 at (0,0,0), Ship 2 at (0,1000,0), Ship 3 at (1000,500,0).
    // Two real coplanar intruders at T1=(500,0,0) and T2=(500,1000,0):
    //   - Pairwise rays from Ship 1 & Ship 2 intersect at T1, T2 AND two coplanar GHOSTS!
    //   - With Ship 3 online (n_senders=3), triple-ray epipolar gate rejects coplanar ghosts!
    //   - When Ship 3 suffers a 150-250ms network drop (n_senders=2), ghosts are ADMITTED!
    let mut fuser = MeshDaaFuser::new();
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
        // Verify exact 52-byte wire round-trip
        OpenXvsBearingTrack::from_le_bytes(&raw.to_le_bytes())
    };

    for cycle in 0..50u64 {
        let ts = 1_700_000_000_000 + cycle * 100_000;
        let obs_3ship = [
            mk_obs(1, 1, p_s1, t1, ts),
            mk_obs(1, 2, p_s1, t2, ts),
            mk_obs(2, 1, p_s2, t1, ts),
            mk_obs(2, 2, p_s2, t2, ts),
            mk_obs(3, 1, p_s3, t1, ts),
            mk_obs(3, 2, p_s3, t2, ts),
        ];
        let res3 = fuser.fuse(&obs_3ship);
        findings.meshdaa_3ship_ghosts_rejected += res3.ghosts_rejected;
        findings.meshdaa_3ship_true_tracks_confirmed += res3
            .tracks
            .iter()
            .take(res3.num_fused)
            .filter(|tr| tr.triple_ray_confirmed)
            .count() as u32;

        // Simulate 150-250ms network drop of Ship 3 -> only 2 senders arrive!
        let obs_2ship = &obs_3ship[..4];
        let res2 = fuser.fuse(obs_2ship);
        // True targets = 2; any extra fused tracks > 2 are coplanar ghosts admitted when Ship 3 drops!
        if res2.num_fused > 2 {
            findings.meshdaa_2ship_dropout_ghosts_admitted += (res2.num_fused - 2) as u32;
        }
    }

    // Emit 2,100 ticks (3 ships x 700 ticks)
    let dt = 0.1_f64;
    let mut emitted = 0u64;
    for ship_id in 1u8..=3u8 {
        let mut rta = SimplexRta::new();
        let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
        let mut ar_obs = ActiveRangeObserver::new();
        let mut odd = OpticalOddMonitor::new_canonical_60hz();
        let mut vlm = AsymmetricSimplexVlmGate::new(200);
        let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

        let angle = (ship_id as f64 - 1.0) * (2.0 * std::f64::consts::PI / 3.0);
        let mut own = ShipKinematics {
            p: [-900.0 * angle.cos(), -900.0 * angle.sin(), -500.0],
            hdg: angle,
            tas: 36.0,
            a: [0.0; 3],
        };
        let peer_ang = angle + (2.0 * std::f64::consts::PI / 3.0);
        let mut intr = ShipKinematics {
            p: [-900.0 * peer_ang.cos(), -900.0 * peer_ang.sin(), -500.0],
            hdg: peer_ang,
            tas: 36.0,
            a: [0.0; 3],
        };
        let mut prev_s = None;

        for tick in 0..700u32 {
            if tick > 0 && tick % 140 == 0 {
                own.p = [-900.0 * angle.cos(), -900.0 * angle.sin(), -500.0];
                own.hdg = angle;
                own.a = [0.0; 3];
                intr.p = [-900.0 * peer_ang.cos(), -900.0 * peer_ang.sin(), -500.0];
                intr.hdg = peer_ang;
                prev_s = None;
            }
            let mut rta_inp = RtaTickInput::default();
            // Inject occasional single-camera epipolar degradation (P3 Isolated5Cam)
            if tick % 140 >= 40 && tick % 140 < 45 {
                rta_inp.epipolar_max_urad[0] = 7000;
            }
            let rta_out = rta.step(&rta_inp);
            let (meas, _) = measure_intruder(&own, &intr, &mut prev_s, 6 + (tick % 140), dt, 0.0);
            let ar_telem = ar.step(&own.ownship_state(14.5), &meas);
            let odd_out = odd.evaluate(&nominal_odd_sensors());
            let vlm_pkt = make_vlm_pkt(
                tick + 1,
                (tick as u64) * 100 + 20,
                if tick % 50 == 0 {
                    VlmHazard::BirdFlock
                } else {
                    VlmHazard::None
                },
                1,
            );
            let vlm_dec = vlm.evaluate_wire(
                Some(&vlm_pkt.encode()),
                (tick as u64) * 100 + 60,
                !odd_out.vlm_gate_reject,
                0x02,
                0x000F,
            );
            let hal_mask = if tick % 100 == 25 { 0x0000_0003 } else { 0 };
            let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, hal_mask);
            let hal_dec = hal.ingest_wire(
                &hal_pkt.to_le_bytes(),
                (tick as u64) * 100_000_000 + 15_000_000,
                rta_mode_to_hal(rta_out.mode),
                0,
            );

            let step_idx = (records.len() + 1) as u64;
            records.push(build_record(
                step_idx,
                "threeship_symmetric_meshdaa",
                tick,
                ship_id,
                &rta_inp,
                &rta_out,
                &ar_telem,
                &mut ar_obs,
                vlm_dec.status as u8,
                vlm_dec.actuator_authority,
                vlm_dec.certified_hazards_mask,
                vlm_dec.final_hazards_mask,
                odd_out.airdata_export_blocked,
                hal_dec.lifecycle as u8,
                rta_out.mode as u8,
                hal_dec.out_mode.to_rta_wire_u8(),
                hal_dec.actuator_authority,
            ));
            own.advance(ar_telem.cmd_bank_rad as f64, dt);
            intr.advance(0.0, dt);
            emitted += 1;
        }
    }
    findings
        .scenario_counts
        .push(("threeship_symmetric_meshdaa", emitted));
}

/// Scenario 3 (2,000 ticks): Intermittent Dropout Duty-Cycle Attack (5% starvation, 21-tick sawtooth,
/// 10/20 max burst, 11/20 P0 trip, >10 consecutive P0 trip, exact 30+20=50 tick staged recovery)
fn run_scenario_3_dropout_duty_cycle(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let mut rta = SimplexRta::new();
    let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
    let mut ar_obs = ActiveRangeObserver::new();
    let mut odd = OpticalOddMonitor::new_canonical_60hz();
    let mut vlm = AsymmetricSimplexVlmGate::new(200);
    let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

    let own = OwnshipStateNed {
        v_ned: [35.0, 0.0, 0.0],
        a_ned: [0.0; 3],
        heading_rad: 0.0,
        tas_mps: 35.0,
        v_stall_mps: 14.5,
    };
    let meas = ActiveRangeMeasurement {
        alpha_rad: 0.02,
        beta_rad: 0.0,
        size_px: 14.0,
        size_rate_px_s: 0.5,
        contrast_ratio: 0.7,
        sky_beta_ext: 2.0e-4,
        daytime_clear: true,
        track_age: 10,
        dt_s: 0.1,
    };

    let mut prev_mode = SimplexMode::NominalMl;
    let mut p0_rec_counter = 0u32;
    let mut p0_horizon_counter = 0u32;
    let mut p1_dwell_counter = 0u32;
    let mut tracking_p0_recovery = false;
    let mut tracking_p0_horizon = false;
    let mut tracking_p1_recovery = false;

    for tick in 0..2000u32 {
        let mut rta_inp = RtaTickInput::default();

        if tick < 400 {
            // Phase A (ticks 0..399): 5% duty-cycle starvation attack!
            // 1 dropout every 20 ticks (tick 0, 20, 40, ...): P4->P2->P1(dwell=1..19)->reset!
            rta_inp.transient_dropout = tick % 20 == 0;
        } else if tick < 450 {
            // Clean recovery to P4
        } else if tick < 870 {
            // Phase B (ticks 450..869): 21-tick periodic sawtooth (P4->P2->P1->P4!)
            rta_inp.transient_dropout = (tick - 450) % 21 == 0;
        } else if tick < 950 {
            // Phase C (ticks 870..949): Alternating 10/20 max-tolerated burst (50% duty cycle)
            rta_inp.transient_dropout = tick % 2 == 0;
        } else if tick < 1000 {
            // Clean gap
        } else if tick < 1011 {
            // Phase D (ticks 1000..1010): 11 consecutive dropouts -> trips P0 on tick 1010!
            rta_inp.transient_dropout = true;
        } else if tick < 1100 {
            // Exact 30-tick P0->P1 + 20-tick P1->P4 staged recovery measurement!
            if tick == 1011 {
                tracking_p0_recovery = true;
                p0_rec_counter = 0;
            }
        } else if tick < 1140 {
            // Phase E (ticks 1100..1139): Intermittent 11/20 duty-cycle breach (2 on, 1 off)
            rta_inp.transient_dropout = (tick - 1100) % 3 != 2;
        } else if tick < 1260 {
            // Clean recovery from P0
        } else if tick < 1300 {
            // Phase F: 5-camera isolation P3 + 20-tick P3->P4 climb
            if tick < 1270 {
                rta_inp.cam_health_mask = 0x1F; // 5 cameras
            }
        } else if tick < 1400 {
            // Phase G: 3-camera (<4) H[0,2] violation -> P0, then recovery
            if tick < 1310 {
                rta_inp.cam_health_mask = 0x07; // 3 cameras
            }
        } else if tick < 1600 {
            // Phase H: Horizon PL > AL fault -> P0 (10 ticks), then clean 30-tick P0->P1 recovery, then spoof_alarm -> P1
            if tick < 1410 {
                rta_inp.horizon_pl_urad = HORIZON_AL_URAD + 500;
            } else if tick == 1410 {
                tracking_p0_horizon = true;
                p0_horizon_counter = 0;
            } else if (1460..1490).contains(&tick) {
                rta_inp.spoof_alarm = true;
            }
        } else {
            // Phase I: Clean steady-state nominal operation + periodic single transient drop
            rta_inp.transient_dropout = tick == 1650 || tick == 1800;
        }

        let rta_out = rta.step(&rta_inp);

        // Collect empirical findings for Finding F2
        if (1..400).contains(&tick) {
            findings.starvation_5pct_ticks_tested += 1;
            if prev_mode == SimplexMode::FallbackD0Safe && rta_out.mode == SimplexMode::DegradedCoast
            {
                findings.starvation_5pct_p1_to_p2_transitions += 1;
            }
            if rta_out.mode == SimplexMode::NominalMl {
                findings.starvation_5pct_recovered_to_p4 = true;
            }
            if rta_out.mode == SimplexMode::InvalidInhibit {
                findings.starvation_5pct_tripped_p0 = true;
            }
            if prev_mode == SimplexMode::DegradedCoast
                && rta_out.mode == SimplexMode::FallbackD0Safe
                && findings.exact_p2_to_p1_dwell_observed == 0
            {
                findings.exact_p2_to_p1_dwell_observed = rta_out.dwell_ticks;
            }
        }
        if (451..870).contains(&tick)
            && prev_mode == SimplexMode::FallbackD0Safe
            && rta_out.mode == SimplexMode::NominalMl
        {
            findings.sawtooth_21tick_p4_cycles += 1;
        }
        if tracking_p0_recovery {
            p0_rec_counter += 1;
            if rta_out.mode == SimplexMode::FallbackD0Safe {
                findings.exact_p0_to_p1_ticks_after_dropout11 = p0_rec_counter;
                tracking_p0_recovery = false;
                tracking_p1_recovery = true;
                p1_dwell_counter = 0;
            }
        } else if tracking_p1_recovery {
            p1_dwell_counter += 1;
            if rta_out.mode == SimplexMode::NominalMl {
                findings.exact_p1_to_p4_ticks = p1_dwell_counter;
                tracking_p1_recovery = false;
            }
        }
        if tracking_p0_horizon {
            p0_horizon_counter += 1;
            if rta_out.mode == SimplexMode::FallbackD0Safe {
                findings.exact_p0_to_p1_ticks_after_horizon_fault = p0_horizon_counter;
                tracking_p0_horizon = false;
            }
        }
        prev_mode = rta_out.mode;

        let ar_telem = ar.step(&own, &meas);
        let odd_out = odd.evaluate(&nominal_odd_sensors());
        let vlm_pkt = make_vlm_pkt(tick + 1, (tick as u64) * 100 + 10, VlmHazard::None, 1);
        let vlm_dec = vlm.evaluate_wire(
            Some(&vlm_pkt.encode()),
            (tick as u64) * 100 + 40,
            !odd_out.vlm_gate_reject,
            0x01,
            0x0003,
        );
        let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, 0);
        let hal_dec = hal.ingest_wire(
            &hal_pkt.to_le_bytes(),
            (tick as u64) * 100_000_000 + 10_000_000,
            rta_mode_to_hal(rta_out.mode),
            0,
        );

        let step_idx = (records.len() + 1) as u64;
        records.push(build_record(
            step_idx,
            "intermittent_dropout_10_20_attack",
            tick,
            1,
            &rta_inp,
            &rta_out,
            &ar_telem,
            &mut ar_obs,
            vlm_dec.status as u8,
            vlm_dec.actuator_authority,
            vlm_dec.certified_hazards_mask,
            vlm_dec.final_hazards_mask,
            odd_out.airdata_export_blocked,
            hal_dec.lifecycle as u8,
            rta_out.mode as u8,
            hal_dec.out_mode.to_rta_wire_u8(),
            hal_dec.actuator_authority,
        ));
    }
    findings
        .scenario_counts
        .push(("intermittent_dropout_10_20_attack", 2000));
}

/// Scenario 4 (2,000 ticks): ActiveRange 0.08/0.10 Sigma Hysteresis Noise, Stall Guard & ImmediateEvade
fn run_scenario_4_sigma_hysteresis(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let mut rta = SimplexRta::new();
    let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
    let mut ar_obs = ActiveRangeObserver::new();
    let mut odd = OpticalOddMonitor::new_canonical_60hz();
    let mut vlm = AsymmetricSimplexVlmGate::new(200);
    let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

    let dt = 0.1_f64;
    let mut own = ShipKinematics {
        p: [0.0, 0.0, -500.0],
        hdg: 0.0,
        tas: 35.0,
        a: [0.0; 3],
    };
    let mut intr = ShipKinematics {
        p: [1000.0, 0.0, -500.0],
        hdg: std::f64::consts::PI,
        tas: 45.0,
        a: [0.0; 3],
    };
    let mut prev_s = None;

    // Deterministic LCG PRNG for reproducible bearing noise across 0.08/0.10 boundary
    let mut rng: u64 = 0x517C_C1B7_2722_0A95;
    let mut next_noise = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((rng >> 33) as f64) / ((1u64 << 31) as f64) - 0.5;
        u * 0.0012 // +/- 0.6 mrad bearing jitter
    };

    for tick in 0..2000u32 {
        if tick > 0 && tick % 120 == 0 {
            own.p = [0.0, 0.0, -500.0];
            own.hdg = 0.0;
            own.a = [0.0; 3];
            own.tas = if (tick / 120) % 4 == 2 { 16.0 } else { 35.0 }; // Exercises < 1.25*v_stall guard!
            intr.p = [
                if (tick / 120) % 5 == 3 { 200.0 } else { 1000.0 },
                0.0,
                -500.0,
            ];
            intr.hdg = std::f64::consts::PI;
            prev_s = None;
        }
        let rta_inp = RtaTickInput::default();
        let rta_out = rta.step(&rta_inp);

        let noise = next_noise();
        let (meas, _) = measure_intruder(&own, &intr, &mut prev_s, 6 + (tick % 120), dt, noise);
        let ar_telem = ar.step(&own.ownship_state(14.5), &meas);
        let odd_out = odd.evaluate(&nominal_odd_sensors());
        let vlm_pkt = make_vlm_pkt(tick + 1, (tick as u64) * 100 + 25, VlmHazard::None, 1);
        let vlm_dec = vlm.evaluate_wire(
            Some(&vlm_pkt.encode()),
            (tick as u64) * 100 + 55,
            !odd_out.vlm_gate_reject,
            0x00,
            0x00FF,
        );
        let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, 0);
        let hal_dec = hal.ingest_wire(
            &hal_pkt.to_le_bytes(),
            (tick as u64) * 100_000_000 + 5_000_000,
            rta_mode_to_hal(rta_out.mode),
            0,
        );

        let step_idx = (records.len() + 1) as u64;
        records.push(build_record(
            step_idx,
            "sigma_hysteresis_008_010_noise",
            tick,
            1,
            &rta_inp,
            &rta_out,
            &ar_telem,
            &mut ar_obs,
            vlm_dec.status as u8,
            vlm_dec.actuator_authority,
            vlm_dec.certified_hazards_mask,
            vlm_dec.final_hazards_mask,
            odd_out.airdata_export_blocked,
            hal_dec.lifecycle as u8,
            rta_out.mode as u8,
            hal_dec.out_mode.to_rta_wire_u8(),
            hal_dec.actuator_authority,
        ));
        own.advance(ar_telem.cmd_bank_rad as f64, dt);
        intr.advance(0.0, dt);
    }
    findings
        .scenario_counts
        .push(("sigma_hysteresis_008_010_noise", 2000));
}

/// Scenario 5 (2,000 ticks): Permanent PluginOffline Mid-Flight Latch + 6-Stage VLM Gate Faults
fn run_scenario_5_plugin_offline_and_vlm(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let mut rta = SimplexRta::new();
    let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
    let mut ar_obs = ActiveRangeObserver::new();
    let mut odd = OpticalOddMonitor::new_canonical_60hz();
    let mut vlm = AsymmetricSimplexVlmGate::new(200);
    let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

    let own = OwnshipStateNed {
        v_ned: [35.0, 0.0, 0.0],
        a_ned: [0.0; 3],
        heading_rad: 0.0,
        tas_mps: 35.0,
        v_stall_mps: 14.5,
    };
    let meas = ActiveRangeMeasurement {
        alpha_rad: 0.01,
        beta_rad: 0.0,
        size_px: 12.0,
        size_rate_px_s: 0.4,
        contrast_ratio: 0.7,
        sky_beta_ext: 2.0e-4,
        daytime_clear: true,
        track_age: 10,
        dt_s: 0.1,
    };

    findings.plugin_offline_stayed_latched = true;
    for tick in 0..2000u32 {
        let rta_inp = RtaTickInput::default();
        let rta_out = rta.step(&rta_inp);
        let ar_telem = ar.step(&own, &meas);

        // Exercise Optical-ODD Fog (Mode 4) + Rain (Mode 1) + Icing (Mode 2) + Glare (Mode 3)
        let mut s = nominal_odd_sensors();
        if (100..220).contains(&tick) {
            s.mor_m = 900; // Mode 4 fog with healthy LWIR -> ThermalPrimary
        } else if (250..380).contains(&tick) {
            s.mtf_hf_ratio_permille = 80;
            s.lwir_std_centidn = 300;
            s.dew_margin_centic = 100; // Mode 2 icing -> DegradedFallback + vlm_gate_reject
        }
        let odd_out = odd.evaluate(&s);

        // Exercise all 6 VLM gate statuses (CRC fail, stale watchdog, OpticalOddMasked, ConformalAbstain, DebouncePending, NominalAdvisory)
        let cert_mask = if tick % 3 == 0 { 0x05 } else { 0x02 };
        let vlm_dec = if tick % 17 == 0 {
            // Corrupt CRC wire packet
            let mut raw = make_vlm_pkt(tick + 1, (tick as u64) * 100, VlmHazard::BirdFlock, 1).encode();
            raw[12] ^= 0xFF;
            vlm.evaluate_wire(Some(&raw), (tick as u64) * 100 + 20, !odd_out.vlm_gate_reject, cert_mask, 0x000F)
        } else if tick % 29 == 0 {
            // Stale (>200 ms) watchdog timeout -> VlmOffline
            let raw = make_vlm_pkt(tick + 1, (tick as u64) * 100, VlmHazard::RunwayIncursion, 1).encode();
            vlm.evaluate_wire(Some(&raw), (tick as u64) * 100 + 500, !odd_out.vlm_gate_reject, cert_mask, 0x000F)
        } else {
            let conf = if tick % 11 == 0 { 0 } else { 1 };
            let haz = if tick % 5 < 3 {
                VlmHazard::RunwayIncursion
            } else {
                VlmHazard::None
            };
            let raw = make_vlm_pkt(tick + 1, (tick as u64) * 100, haz, conf).encode();
            vlm.evaluate_wire(Some(&raw), (tick as u64) * 100 + 30, !odd_out.vlm_gate_reject, cert_mask, 0x000F)
        };

        // Exercise Ring-1 monotonic tightening (ticks 0..399), then mid-flight CRC tamper at tick 400!
        // Ticks 401..1999 send 1,599 perfectly valid packets proving PluginOffline NEVER unlatches!
        let hal_mask = if tick < 100 {
            0
        } else if tick < 200 {
            0x0000_0001 // Isolated5Cam
        } else if tick < 300 {
            0x0000_0010 // FallbackD0Safe
        } else if tick < 400 {
            0x0000_0100 // InvalidInhibit (P0 tightening!)
        } else {
            0x0000_00F0
        };
        let mut hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, hal_mask);
        if tick == 400 {
            hal_pkt.ieee_crc32 ^= 0x0000_0001; // Single-bit mid-flight CRC corruption!
        }
        let hal_dec = hal.ingest_advisory(
            &hal_pkt,
            (tick as u64) * 100_000_000 + 10_000_000,
            rta_mode_to_hal(rta_out.mode),
            0x0000_0001,
        );
        if tick > 400 {
            findings.plugin_offline_post_latch_attempts += 1;
            if hal_dec.lifecycle != PluginLifecycle::PluginOffline {
                findings.plugin_offline_stayed_latched = false;
            }
        }

        let step_idx = (records.len() + 1) as u64;
        records.push(build_record(
            step_idx,
            "plugin_offline_and_vlm_faults",
            tick,
            1,
            &rta_inp,
            &rta_out,
            &ar_telem,
            &mut ar_obs,
            vlm_dec.status as u8,
            vlm_dec.actuator_authority,
            vlm_dec.certified_hazards_mask,
            vlm_dec.final_hazards_mask,
            odd_out.airdata_export_blocked,
            hal_dec.lifecycle as u8,
            rta_out.mode as u8,
            hal_dec.out_mode.to_rta_wire_u8(),
            hal_dec.actuator_authority,
        ));
    }
    findings
        .scenario_counts
        .push(("plugin_offline_and_vlm_faults", 2000));
}

/// Scenario 6 (1,900 ticks): Mandatory Go-Around Latch + Simultaneous ImmediateEvade vs P0 Arbiter
fn run_scenario_6_go_around_and_p0_arbiter(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let mut rta = SimplexRta::new();
    let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
    let mut ar_obs = ActiveRangeObserver::new();
    let mut odd = OpticalOddMonitor::new_canonical_60hz();
    let mut vlm = AsymmetricSimplexVlmGate::new(200);
    let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

    let own = OwnshipStateNed {
        v_ned: [35.0, 0.0, 0.0],
        a_ned: [0.0; 3],
        heading_rad: 0.0,
        tas_mps: 35.0,
        v_stall_mps: 14.5,
    };
    // Close terminal threat forcing ActiveRange into ImmediateEvade (±25 deg) while P0 is latched!
    let close_meas = ActiveRangeMeasurement {
        alpha_rad: 0.0,
        beta_rad: 0.0,
        size_px: 140.0,
        size_rate_px_s: 45.0, // tau = 3.1 s < 4.5 s!
        contrast_ratio: 0.9,
        sky_beta_ext: 2.0e-4,
        daytime_clear: true,
        track_age: 15,
        dt_s: 0.1,
    };

    for tick in 0..1900u32 {
        let mut rta_inp = RtaTickInput::default();
        let phase = tick % 250;
        if phase == 10 {
            // Trip mandatory go-around via VPL/HPL/runway/quantize fault during approach!
            rta_inp.approach_active = true;
            match (tick / 250) % 4 {
                0 => rta_inp.v_vpl_mm = V_VAL_MM + 250,
                1 => rta_inp.v_hpl_mm = V_HAL_MM + 500,
                2 => rta_inp.runway_width_fault = true,
                _ => rta_inp.quantize_fault = true,
            }
        } else if (11..80).contains(&phase) {
            // Faults cleared AND adversary repeatedly asserts go_around_ack_reset=true while approach_active=true!
            rta_inp.approach_active = true;
            rta_inp.go_around_ack_reset = true;
        } else if phase == 80 {
            // Pilot executes go-around climb (approach_active=false) and acknowledges reset!
            rta_inp.approach_active = false;
            rta_inp.go_around_ack_reset = true;
        }
        // Ticks 81..249: Full 30+20=50 tick P0->P1->P4 recovery!

        let rta_out = rta.step(&rta_inp);
        if (11..80).contains(&phase) {
            assert!(rta_out.mandatory_go_around);
            assert_eq!(rta_out.mode, SimplexMode::InvalidInhibit);
            findings.go_around_approach_reset_attempts_blocked += 1;
        }

        let ar_telem = ar.step(&own, &close_meas);
        let odd_out = odd.evaluate(&nominal_odd_sensors());
        let vlm_pkt = make_vlm_pkt(tick + 1, (tick as u64) * 100 + 10, VlmHazard::RunwayIncursion, 1);
        let vlm_dec = vlm.evaluate_wire(
            Some(&vlm_pkt.encode()),
            (tick as u64) * 100 + 30,
            !odd_out.vlm_gate_reject,
            0x01,
            0x0007,
        );
        let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, 0);
        let hal_dec = hal.ingest_advisory(
            &hal_pkt,
            (tick as u64) * 100_000_000 + 10_000_000,
            rta_mode_to_hal(rta_out.mode),
            0,
        );

        let arb = arbiter_bank(&rta_out, hal_dec.out_mode.to_rta_wire_u8(), &ar_telem);
        if rta_out.inhibit_avoidance_turns && ar_telem.mode == ActiveRangeMode::ImmediateEvade as u8 {
            assert_eq!(arb, 0);
            findings.p0_overrode_immediate_evade_ticks += 1;
        }

        let step_idx = (records.len() + 1) as u64;
        records.push(build_record(
            step_idx,
            "mandatory_go_around_and_p0_arbiter",
            tick,
            1,
            &rta_inp,
            &rta_out,
            &ar_telem,
            &mut ar_obs,
            vlm_dec.status as u8,
            vlm_dec.actuator_authority,
            vlm_dec.certified_hazards_mask,
            vlm_dec.final_hazards_mask,
            odd_out.airdata_export_blocked,
            hal_dec.lifecycle as u8,
            rta_out.mode as u8,
            hal_dec.out_mode.to_rta_wire_u8(),
            hal_dec.actuator_authority,
        ));
    }
    findings
        .scenario_counts
        .push(("mandatory_go_around_and_p0_arbiter", 1900));
}

/// Scenario 7 (2,000 ticks): Coordinated Multi-Ship F1 + F4 + F3 (`xvs-multiship-coord`)
/// Drives real `CoordinatedActiveRange` (Tier B optical-only FAR 91.113 + Tier A MeshDAA
/// aircraft-ID stagger), `StatefulMeshDaaGuard` (200 ms staleness/replay + 500 ms sticky
/// triple-ray ghost hysteresis), and `Ring0AuthorityGate` (RTA P0 / HAL-P0 dominance),
/// verifying behavioral refinement against `OpenXvsMultiShipRta.tla` (`USE_COORD_FIX = TRUE`).
fn run_scenario_7_coordinated_multiship_f1_f4_f3(
    records: &mut Vec<TraceRecord>,
    findings: &mut EmpiricalFindings,
) {
    let dt = 0.1_f64;

    // 7A. Empirical Check: Coordinated Optical-Only (Tier B) async-12 head-on eliminates 3.79x overestimate
    {
        let mut c2 = CoordinatedActiveRange::new(2, ActiveRangeConfig::default());
        let prime_own = OwnshipStateNed {
            v_ned: [-35.0, 0.0, 0.0],
            a_ned: [0.0; 3],
            heading_rad: std::f64::consts::PI,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let prime_meas = ActiveRangeMeasurement {
            alpha_rad: std::f64::consts::PI,
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
        c2.reset_filter_keep_phase();

        let mut s1 = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: 0.0,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut s2 = ShipKinematics {
            p: [1200.0, 0.0, -500.0],
            hdg: std::f64::consts::PI,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut c1 = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let mut ps1 = None;
        let mut ps2 = None;
        let mut peak_ratio = 0.0_f64;
        for t in 0..30u32 {
            let (m1, r_true) = measure_intruder(&s1, &s2, &mut ps1, 6 + t, dt, 0.0);
            let (m2, _) = measure_intruder(&s2, &s1, &mut ps2, 18 + t, dt, 0.0);
            let t1 = c1.step(&s1.ownship_state(14.5), &m1, None, Ring0AuthorityGate::NOMINAL);
            let t2 = c2.step(&s2.ownship_state(14.5), &m2, None, Ring0AuthorityGate::NOMINAL);
            if sign_i32(t1.cmd_bank_rad) != 0 && sign_i32(t1.cmd_bank_rad) == -sign_i32(t2.cmd_bank_rad) {
                findings.coord_optical_async12_cancelled_ticks += 1;
            }
            let ratio = (t1.r_hat_m as f64) / r_true;
            if ratio > peak_ratio {
                peak_ratio = ratio;
                findings.coord_optical_async12_peak_ratio = ratio;
            }
            if t == 29 {
                findings.coord_optical_async12_final_r_hat_m = t1.r_hat_m as f64;
                findings.coord_optical_async12_final_r_true_m = r_true;
            }
            s1.advance(t1.cmd_bank_rad as f64, dt);
            s2.advance(t2.cmd_bank_rad as f64, dt);
        }
    }

    // 7B. Empirical Check: Coordinated Tier A MeshDAA aircraft-ID phase staggering
    {
        let mut s1 = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: 0.0,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut s2 = ShipKinematics {
            p: [1200.0, 0.0, -500.0],
            hdg: std::f64::consts::PI,
            tas: 52.0,
            a: [0.0; 3],
        };
        let mut c1 = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let mut c2 = CoordinatedActiveRange::new(2, ActiveRangeConfig::default());
        let mut ps1 = None;
        let mut ps2 = None;
        for t in 0..40u32 {
            let now_us = 1_700_000_000_000 + (t as u64) * 100_000;
            let (m1, _) = measure_intruder(&s1, &s2, &mut ps1, 6 + t, dt, 0.0);
            let (m2, _) = measure_intruder(&s2, &s1, &mut ps2, 6 + t, dt, 0.0);
            let h1 = PeerCoordHint {
                peer_sender_id: 2,
                peer_ts_utc_us: now_us - 40_000,
                now_utc_us: now_us,
                peer_a_ned: Some(s2.a),
            };
            let h2 = PeerCoordHint {
                peer_sender_id: 1,
                peer_ts_utc_us: now_us - 40_000,
                now_utc_us: now_us,
                peer_a_ned: Some(s1.a),
            };
            let t1 = c1.step(&s1.ownship_state(14.5), &m1, Some(h1), Ring0AuthorityGate::NOMINAL);
            let t2 = c2.step(&s2.ownship_state(14.5), &m2, Some(h2), Ring0AuthorityGate::NOMINAL);
            if sign_i32(t1.cmd_bank_rad) != 0 && sign_i32(t1.cmd_bank_rad) == -sign_i32(t2.cmd_bank_rad) {
                findings.coord_meshdaa_stagger_cancelled_ticks += 1;
            }
            if c1.last_role() == SwerveCoordRole::YieldHoldForPeerProbe
                || c2.last_role() == SwerveCoordRole::YieldHoldForPeerProbe
            {
                findings.coord_meshdaa_yield_ticks += 1;
            }
            s1.advance(t1.cmd_bank_rad as f64, dt);
            s2.advance(t2.cmd_bank_rad as f64, dt);
        }
    }

    // 7C. Empirical Check: StatefulMeshDaaGuard 50-cycle 3<->2 dropout + staleness/replay gate
    let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
    let p_s1 = [base_ecef[0], 0.0, 0.0];
    let p_s2 = [base_ecef[0], 1000.0, 0.0];
    let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
    let t1_pos = [base_ecef[0] + 500.0, 200.0, 0.0];
    let t2_pos = [base_ecef[0] + 500.0, 800.0, 0.0];

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

    {
        let mut guard = StatefulMeshDaaGuard::new();
        for cycle in 0..50u64 {
            let ts_3 = 1_700_000_000_000 + cycle * 400_000;
            let obs_3 = [
                mk_obs(1, 1, p_s1, t1_pos, ts_3),
                mk_obs(1, 2, p_s1, t2_pos, ts_3),
                mk_obs(2, 1, p_s2, t1_pos, ts_3),
                mk_obs(2, 2, p_s2, t2_pos, ts_3),
                mk_obs(3, 1, p_s3, t1_pos, ts_3),
                mk_obs(3, 2, p_s3, t2_pos, ts_3),
            ];
            let g3 = guard.fuse_guarded(&obs_3, ts_3 + 20_000);
            findings.coord_mesh_guard_3ship_ghosts_rejected += g3.ghosts_rejected_3ship;

            let ts_2 = ts_3 + 200_000;
            let obs_2 = [
                mk_obs(1, 1, p_s1, t1_pos, ts_2),
                mk_obs(1, 2, p_s1, t2_pos, ts_2),
                mk_obs(2, 1, p_s2, t1_pos, ts_2),
                mk_obs(2, 2, p_s2, t2_pos, ts_2),
            ];
            let g2 = guard.fuse_guarded(&obs_2, ts_2 + 20_000);
            assert!(g2.sticky_triple_ray_active);
            findings.coord_mesh_guard_sticky_suppressed += g2.sticky_ghosts_suppressed;
            if g2.num_fused > 2 {
                findings.coord_mesh_guard_2ship_ghosts_admitted += (g2.num_fused - 2) as u32;
            }
        }
        let now_chk = 1_700_050_000_000u64;
        let stale_obs = [
            mk_obs(1, 1, p_s1, t1_pos, now_chk - 250_000),
            mk_obs(2, 1, p_s2, t1_pos, now_chk + 90_000),
            mk_obs(3, 1, p_s3, t1_pos, now_chk - 50_000),
        ];
        let g_st = guard.fuse_guarded(&stale_obs, now_chk);
        findings.coord_mesh_guard_stale_rejected += g_st.stale_obs_rejected;
        let rep_obs = [mk_obs(3, 1, p_s3, t1_pos, now_chk - 100_000)];
        let g_rp = guard.fuse_guarded(&rep_obs, now_chk + 10_000);
        findings.coord_mesh_guard_replay_rejected += g_rp.replay_obs_rejected;
    }

    // 7D. Emit 2,000 coordinated full-stack NDJSON ticks (1,000 Ship 1 + 1,000 Ship 2)
    let mut emitted = 0u64;
    for ship_id in 1u8..=2u8 {
        let peer_id = if ship_id == 1 { 2 } else { 1 };
        let mut rta = SimplexRta::new();
        let mut coord = CoordinatedActiveRange::new(ship_id, ActiveRangeConfig::default());
        let mut ar_obs = ActiveRangeObserver::new();
        let mut mesh_guard = StatefulMeshDaaGuard::new();
        let mut odd = OpticalOddMonitor::new_canonical_60hz();
        let mut vlm = AsymmetricSimplexVlmGate::new(200);
        let mut hal = PluginSupervisorState::new(DEFAULT_MAX_HEARTBEAT_MISS_MS);

        let init_hdg = if ship_id == 1 { 0.0 } else { std::f64::consts::PI };
        let intr_hdg = if ship_id == 1 { std::f64::consts::PI } else { 0.0 };
        let intr_x = if ship_id == 1 { 1500.0 } else { -1500.0 };

        let mut own = ShipKinematics {
            p: [0.0, 0.0, -500.0],
            hdg: init_hdg,
            tas: 80.0,
            a: [0.0; 3],
        };
        let mut intr = ShipKinematics {
            p: [intr_x, 0.0, -500.0],
            hdg: intr_hdg,
            tas: 42.0,
            a: [0.0; 3],
        };
        let mut prev_s = None;

        for tick in 0..1000u32 {
            let phase = tick % 200;
            // Re-seed head-on encounter cleanly so |alpha_body| remains strictly inside head-on band
            // (reset_filter_keep_phase clears stale EKF alpha_dot/range while preserving swerve_ticks phase!)
            if matches!(phase, 0 | 25 | 50 | 70 | 90 | 110 | 155) {
                own.p = [0.0, 0.0, -500.0];
                own.hdg = init_hdg;
                own.a = [0.0; 3];
                own.tas = 80.0;
                intr.p = [intr_x, 0.0, -500.0];
                intr.hdg = intr_hdg;
                prev_s = None;
                coord.reset_filter_keep_phase();
            }

            let mut rta_inp = RtaTickInput::default();
            if (110..120).contains(&phase) {
                // Inject RTA P0 fault during close terminal threat window
                rta_inp.horizon_pl_urad = HORIZON_AL_URAD + 500;
            }
            let rta_out = rta.step(&rta_inp);

            let odd_out = odd.evaluate(&nominal_odd_sensors());
            let vlm_pkt = make_vlm_pkt(tick + 1, (tick as u64) * 100 + 20, VlmHazard::None, 1);
            let vlm_dec = vlm.evaluate_wire(
                Some(&vlm_pkt.encode()),
                (tick as u64) * 100 + 50,
                !odd_out.vlm_gate_reject,
                0x02,
                0x0007,
            );

            // Inject HAL-P0 monotonic tightening at phase 150..155 right after RTA P0->P1 recovery
            let hal_mask = if (150..155).contains(&phase) {
                0x0000_0100
            } else {
                0
            };
            let hal_pkt = make_ring1_pkt((tick + 1) as u64, (tick as u64) * 100_000_000, hal_mask);
            let hal_dec = hal.ingest_wire(
                &hal_pkt.to_le_bytes(),
                (tick as u64) * 100_000_000 + 10_000_000,
                rta_mode_to_hal(rta_out.mode),
                0,
            );
            let hal_out_wire = hal_dec.out_mode.to_rta_wire_u8();

            let authority = Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: rta_out.inhibit_avoidance_turns,
                hal_out_mode_wire: hal_out_wire,
            };

            let now_us = 1_700_000_000_000 + (tick as u64) * 100_000;
            let meas = if (110..155).contains(&phase) {
                // Close terminal threat forcing ImmediateEvade if not inhibited by Ring-0 P0 / HAL-P0
                ActiveRangeMeasurement {
                    alpha_rad: own.hdg,
                    beta_rad: 0.0,
                    size_px: 140.0,
                    size_rate_px_s: 45.0,
                    contrast_ratio: 0.9,
                    sky_beta_ext: 2.0e-4,
                    daytime_clear: true,
                    track_age: 15,
                    dt_s: dt,
                }
            } else {
                let (m, _) = measure_intruder(&own, &intr, &mut prev_s, 6 + (phase % 25), dt, 0.0);
                m
            };

            // Tier A fresh cooperative PeerCoordHint during phase 50..110; Tier B optical-only otherwise
            let peer_hint = if (50..110).contains(&phase) {
                Some(PeerCoordHint {
                    peer_sender_id: peer_id,
                    peer_ts_utc_us: now_us - 40_000,
                    now_utc_us: now_us,
                    peer_a_ned: Some([0.0; 3]),
                })
            } else {
                None
            };

            let alpha_body_meas = wrap_pi(meas.alpha_rad - own.hdg);
            let head_on = alpha_body_meas.abs() < HEAD_ON_ALPHA_BAND_RAD;

            let ar_telem = coord.step(&own.ownship_state(14.5), &meas, peer_hint, authority);
            let coord_role = coord.last_role();

            if authority.inhibits_turns() {
                assert_eq!(ar_telem.cmd_bank_rad, 0.0);
                assert_eq!(ar_telem.mode, ActiveRangeMode::NominalHold as u8);
                assert_eq!(coord_role, SwerveCoordRole::InhibitedByP0);
                findings.coord_p0_inhibited_ticks += 1;
            }
            if head_on && ar_telem.mode == ActiveRangeMode::ActiveSwerve as u8 {
                assert!(
                    ar_telem.cmd_bank_rad > 0.0,
                    "Coordinated head-on ActiveSwerve must strictly preserve FAR 91.113 +1 right turn"
                );
                findings.coord_headon_non_reversing_swerve_ticks += 1;
            }

            // Step real StatefulMeshDaaGuard every tick across 3-ship, 2-ship dropout, stale & replay
            let ts_fresh = now_us - 20_000;
            let guard_res = match tick % 4 {
                0 => {
                    let obs_3 = [
                        mk_obs(1, 1, p_s1, t1_pos, ts_fresh),
                        mk_obs(1, 2, p_s1, t2_pos, ts_fresh),
                        mk_obs(2, 1, p_s2, t1_pos, ts_fresh),
                        mk_obs(2, 2, p_s2, t2_pos, ts_fresh),
                        mk_obs(3, 1, p_s3, t1_pos, ts_fresh),
                        mk_obs(3, 2, p_s3, t2_pos, ts_fresh),
                    ];
                    mesh_guard.fuse_guarded(&obs_3, now_us)
                }
                1 => {
                    let obs_2 = [
                        mk_obs(1, 1, p_s1, t1_pos, ts_fresh),
                        mk_obs(1, 2, p_s1, t2_pos, ts_fresh),
                        mk_obs(2, 1, p_s2, t1_pos, ts_fresh),
                        mk_obs(2, 2, p_s2, t2_pos, ts_fresh),
                    ];
                    mesh_guard.fuse_guarded(&obs_2, now_us)
                }
                2 => {
                    // 2-ship dropout + 250 ms stale & 90 ms future skew attacks
                    let obs_2_stale = [
                        mk_obs(1, 1, p_s1, t1_pos, ts_fresh),
                        mk_obs(1, 2, p_s1, t2_pos, ts_fresh),
                        mk_obs(2, 1, p_s2, t1_pos, ts_fresh),
                        mk_obs(2, 2, p_s2, t2_pos, ts_fresh),
                        mk_obs(3, 1, p_s3, t1_pos, now_us - 250_000),
                        mk_obs(3, 2, p_s3, t2_pos, now_us + 90_000),
                    ];
                    mesh_guard.fuse_guarded(&obs_2_stale, now_us)
                }
                _ => {
                    // 2-ship dropout + non-monotonic replay from sender 1 (150 ms old < 120 ms last_ts)
                    let obs_2_rep = [
                        mk_obs(1, 1, p_s1, t1_pos, now_us - 150_000),
                        mk_obs(1, 1, p_s1, t1_pos, ts_fresh),
                        mk_obs(1, 2, p_s1, t2_pos, ts_fresh),
                        mk_obs(2, 1, p_s2, t1_pos, ts_fresh),
                        mk_obs(2, 2, p_s2, t2_pos, ts_fresh),
                    ];
                    mesh_guard.fuse_guarded(&obs_2_rep, now_us)
                }
            };
            let ghosts_admitted = guard_res.num_fused.saturating_sub(2) as u32;
            assert_eq!(
                ghosts_admitted, 0,
                "StatefulMeshDaaGuard must admit 0 coplanar ghosts on every coordinated tick"
            );

            let step_idx = (records.len() + 1) as u64;
            records.push(build_coord_record(
                step_idx,
                "coordinated_multiship_f1_f4_f3",
                tick,
                ship_id,
                &rta_inp,
                &rta_out,
                &ar_telem,
                &mut ar_obs,
                coord.swerve_ticks(),
                head_on,
                coord_role,
                vlm_dec.status as u8,
                vlm_dec.actuator_authority,
                vlm_dec.certified_hazards_mask,
                vlm_dec.final_hazards_mask,
                odd_out.airdata_export_blocked,
                hal_dec.lifecycle as u8,
                rta_out.mode as u8,
                hal_out_wire,
                hal_dec.actuator_authority,
                ghosts_admitted,
                guard_res.sticky_triple_ray_active,
                guard_res.stale_obs_rejected,
                guard_res.replay_obs_rejected,
            ));

            if !(110..155).contains(&phase) {
                own.advance(ar_telem.cmd_bank_rad as f64, dt);
                intr.advance(0.0, dt);
            }
            emitted += 1;
        }
    }

    findings.coordinated_ticks = emitted;
    findings
        .scenario_counts
        .push(("coordinated_multiship_f1_f4_f3", emitted));
}

fn generate_all_traces() -> (Vec<TraceRecord>, EmpiricalFindings) {
    let mut records = Vec::with_capacity(14_000);
    let mut findings = EmpiricalFindings::default();

    run_scenario_1_reciprocal_headon(&mut records, &mut findings);
    run_scenario_2_threeship_meshdaa(&mut records, &mut findings);
    run_scenario_3_dropout_duty_cycle(&mut records, &mut findings);
    run_scenario_4_sigma_hysteresis(&mut records, &mut findings);
    run_scenario_5_plugin_offline_and_vlm(&mut records, &mut findings);
    run_scenario_6_go_around_and_p0_arbiter(&mut records, &mut findings);
    findings.frozen_ticks = records.len() as u64;

    run_scenario_7_coordinated_multiship_f1_f4_f3(&mut records, &mut findings);

    findings.total_ticks = records.len() as u64;
    (records, findings)
}

fn write_findings_json(path: &PathBuf, f: &EmpiricalFindings) -> std::io::Result<()> {
    let mut sc_json = String::from("{");
    for (i, (name, cnt)) in f.scenario_counts.iter().enumerate() {
        if i > 0 {
            sc_json.push(',');
        }
        sc_json.push_str(&format!("\"{name}\":{cnt}"));
    }
    sc_json.push('}');

    let json = format!(
        "{{\n\
        \x20 \"schema\": \"xvs.tla_rust_empirical_findings/1\",\n\
        \x20 \"total_ticks\": {},\n\
        \x20 \"scenario_counts\": {},\n\
        \x20 \"f1_reciprocal_headon\": {{\n\
        \x20   \"sync_ship1_final_r_hat_m\": {:.4},\n\
        \x20   \"sync_ship1_final_r_true_m\": {:.4},\n\
        \x20   \"sync_ship1_rel_err\": {:.6},\n\
        \x20   \"async12_ship1_peak_r_hat_m\": {:.4},\n\
        \x20   \"async12_ship1_r_true_at_peak_m\": {:.4},\n\
        \x20   \"async12_overestimate_ratio\": {:.4},\n\
        \x20   \"async12_cancelled_ticks\": {}\n\
        \x20 }},\n\
        \x20 \"f2_dropout_starvation_and_chattering\": {{\n\
        \x20   \"starvation_5pct_ticks_tested\": {},\n\
        \x20   \"starvation_5pct_p1_to_p2_transitions\": {},\n\
        \x20   \"starvation_5pct_recovered_to_p4\": {},\n\
        \x20   \"starvation_5pct_tripped_p0\": {},\n\
        \x20   \"sawtooth_21tick_p4_cycles\": {},\n\
        \x20   \"exact_p0_to_p1_ticks_after_dropout11\": {},\n\
        \x20   \"exact_p0_to_p1_ticks_after_horizon_fault\": {},\n\
        \x20   \"exact_p1_to_p4_ticks\": {},\n\
        \x20   \"exact_p2_to_p1_dwell_observed\": {}\n\
        \x20 }},\n\
        \x20 \"f3_authority_and_latches\": {{\n\
        \x20   \"p0_overrode_immediate_evade_ticks\": {},\n\
        \x20   \"plugin_offline_post_latch_attempts\": {},\n\
        \x20   \"plugin_offline_stayed_latched\": {},\n\
        \x20   \"go_around_approach_reset_attempts_blocked\": {}\n\
        \x20 }},\n\
        \x20 \"f4_meshdaa_ghost_oscillation\": {{\n\
        \x20   \"meshdaa_3ship_ghosts_rejected\": {},\n\
        \x20   \"meshdaa_2ship_dropout_ghosts_admitted\": {},\n\
        \x20   \"meshdaa_3ship_true_tracks_confirmed\": {}\n\
        \x20 }},\n\
        \x20 \"coordinated_multiship_verification\": {{\n\
        \x20   \"frozen_ticks\": {},\n\
        \x20   \"coordinated_ticks\": {},\n\
        \x20   \"f1_optical_async12_cancelled_ticks\": {},\n\
        \x20   \"f1_optical_async12_peak_ratio\": {:.4},\n\
        \x20   \"f1_optical_async12_final_r_hat_m\": {:.4},\n\
        \x20   \"f1_optical_async12_final_r_true_m\": {:.4},\n\
        \x20   \"f1_meshdaa_stagger_cancelled_ticks\": {},\n\
        \x20   \"f1_meshdaa_yield_ticks\": {},\n\
        \x20   \"f1_headon_non_reversing_swerve_ticks\": {},\n\
        \x20   \"f4_mesh_guard_3ship_ghosts_rejected\": {},\n\
        \x20   \"f4_mesh_guard_2ship_ghosts_admitted\": {},\n\
        \x20   \"f4_mesh_guard_sticky_suppressed\": {},\n\
        \x20   \"f4_mesh_guard_stale_rejected\": {},\n\
        \x20   \"f4_mesh_guard_replay_rejected\": {},\n\
        \x20   \"f3_p0_inhibited_ticks\": {}\n\
        \x20 }}\n\
        }}\n",
        f.total_ticks,
        sc_json,
        f.headon_sync_ship1_final_r_hat_m,
        f.headon_sync_ship1_final_r_true_m,
        f.headon_sync_ship1_rel_err,
        f.headon_async12_ship1_peak_r_hat_m,
        f.headon_async12_ship1_r_true_at_peak_m,
        f.headon_async12_overestimate_ratio,
        f.headon_async12_cancelled_ticks,
        f.starvation_5pct_ticks_tested,
        f.starvation_5pct_p1_to_p2_transitions,
        f.starvation_5pct_recovered_to_p4,
        f.starvation_5pct_tripped_p0,
        f.sawtooth_21tick_p4_cycles,
        f.exact_p0_to_p1_ticks_after_dropout11,
        f.exact_p0_to_p1_ticks_after_horizon_fault,
        f.exact_p1_to_p4_ticks,
        f.exact_p2_to_p1_dwell_observed,
        f.p0_overrode_immediate_evade_ticks,
        f.plugin_offline_post_latch_attempts,
        f.plugin_offline_stayed_latched,
        f.go_around_approach_reset_attempts_blocked,
        f.meshdaa_3ship_ghosts_rejected,
        f.meshdaa_2ship_dropout_ghosts_admitted,
        f.meshdaa_3ship_true_tracks_confirmed,
        f.frozen_ticks,
        f.coordinated_ticks,
        f.coord_optical_async12_cancelled_ticks,
        f.coord_optical_async12_peak_ratio,
        f.coord_optical_async12_final_r_hat_m,
        f.coord_optical_async12_final_r_true_m,
        f.coord_meshdaa_stagger_cancelled_ticks,
        f.coord_meshdaa_yield_ticks,
        f.coord_headon_non_reversing_swerve_ticks,
        f.coord_mesh_guard_3ship_ghosts_rejected,
        f.coord_mesh_guard_2ship_ghosts_admitted,
        f.coord_mesh_guard_sticky_suppressed,
        f.coord_mesh_guard_stale_rejected,
        f.coord_mesh_guard_replay_rejected,
        f.coord_p0_inhibited_ticks,
    );
    fs::write(path, json)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut trace_out = PathBuf::from("../../outputs/formal_tla/rust_trace.ndjson");
    let mut findings_out = PathBuf::from("../../outputs/formal_tla/rust_adversarial_findings.json");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                trace_out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--findings" if i + 1 < args.len() => {
                findings_out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    if let Some(p) = trace_out.parent() {
        fs::create_dir_all(p)?;
    }
    if let Some(p) = findings_out.parent() {
        fs::create_dir_all(p)?;
    }

    let (records, findings) = generate_all_traces();
    assert!(
        records.len() >= 10_000,
        "Trace must contain >= 10,000 ticks, got {}",
        records.len()
    );

    let file = File::create(&trace_out)?;
    let mut writer = BufWriter::new(file);
    for rec in &records {
        rec.write_json_line(&mut writer)?;
    }
    writer.flush()?;

    write_findings_json(&findings_out, &findings)?;

    println!(
        "xvs-tla-trace: emitted {} ticks across {} scenarios -> {} (findings -> {})",
        findings.total_ticks,
        findings.scenario_counts.len(),
        trace_out.display(),
        findings_out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generates_14000_deterministic_ticks_and_confirms_empirical_and_coord_findings() {
        let (recs1, f1) = generate_all_traces();
        let (recs2, _f2) = generate_all_traces();
        assert_eq!(f1.frozen_ticks, 12_000);
        assert_eq!(f1.coordinated_ticks, 2_000);
        assert_eq!(recs1.len(), 14_000);
        assert_eq!(recs1.len(), recs2.len());

        // Byte-exact determinism check across independent runs
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        for r in &recs1 {
            r.write_json_line(&mut b1).unwrap();
        }
        for r in &recs2 {
            r.write_json_line(&mut b2).unwrap();
        }
        assert_eq!(b1, b2, "Trace emission must be 100% deterministic");

        // Verify F2 exact frozen Rust constants & starvation / chattering immunity
        assert_eq!(f1.starvation_5pct_p1_to_p2_transitions, 0);
        assert!(!f1.starvation_5pct_recovered_to_p4);
        assert!(!f1.starvation_5pct_tripped_p0);
        assert!(f1.sawtooth_21tick_p4_cycles >= 15);
        assert_eq!(f1.exact_p0_to_p1_ticks_after_dropout11, 39);
        assert_eq!(f1.exact_p0_to_p1_ticks_after_horizon_fault, 30);
        assert_eq!(f1.exact_p1_to_p4_ticks, 20);
        assert_eq!(f1.exact_p2_to_p1_dwell_observed, 1);

        // Verify F3 latches & authority precedence
        assert!(f1.p0_overrode_immediate_evade_ticks > 100);
        assert_eq!(f1.plugin_offline_post_latch_attempts, 1599);
        assert!(f1.plugin_offline_stayed_latched);
        assert!(f1.go_around_approach_reset_attempts_blocked > 400);

        // Verify F4 3-ship MeshDAA ghost gate & 2-ship dropout oscillation
        assert!(f1.meshdaa_3ship_ghosts_rejected > 0);
        assert!(f1.meshdaa_3ship_true_tracks_confirmed > 0);
        assert!(f1.meshdaa_2ship_dropout_ghosts_admitted > 0);

        // Verify F1 asynchronous 12-tick phase offset triggers opposite-bank inertial cancellation
        assert!(f1.headon_async12_cancelled_ticks > 0);

        // Verify coordinated xvs-multiship-coord closures (F1 + F4 + F3)
        assert_eq!(f1.coord_optical_async12_cancelled_ticks, 0);
        assert!(f1.coord_optical_async12_peak_ratio < 1.15);
        assert!(f1.coord_optical_async12_final_r_hat_m <= f1.coord_optical_async12_final_r_true_m);
        assert_eq!(f1.coord_meshdaa_stagger_cancelled_ticks, 0);
        assert!(f1.coord_meshdaa_yield_ticks > 0);
        assert!(f1.coord_headon_non_reversing_swerve_ticks > 100);
        assert_eq!(f1.coord_mesh_guard_3ship_ghosts_rejected, 100);
        assert_eq!(f1.coord_mesh_guard_2ship_ghosts_admitted, 0);
        assert_eq!(f1.coord_mesh_guard_sticky_suppressed, 100);
        assert_eq!(f1.coord_mesh_guard_stale_rejected, 2);
        assert_eq!(f1.coord_mesh_guard_replay_rejected, 1);
        assert!(f1.coord_p0_inhibited_ticks >= 400);
    }
}
