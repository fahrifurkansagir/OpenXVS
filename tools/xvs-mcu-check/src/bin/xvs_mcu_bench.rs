// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Bare-Metal ARM Cortex-M7 (`thumbv7em-none-eabi`) `10,000`-Tick Real Emulated Execution Proof.
//!
//! Strictly `#![no_std]`, `#![no_main]`, zero heap allocations (`0` `std`/`alloc`/`malloc`/`free`),
//! strictly NED/ECEF/RDF (`0` internal ENU), `NON_FLIGHT_SIM_ONLY = true`,
//! `ACTUATOR_AUTHORITY = false`.
//!
//! Exercises ALL FIVE OpenXVS Ring-0 crates (`xvs-rta-core`, `xvs-odd-vlm`, `xvs-os-hal`,
//! `xvs-active-range`, `xvs-multiship-coord`, plus transitive `xvs-math-core` and `libm`)
//! across 10,000 deterministic ticks spanning:
//!   - Phase 0 (`0..2500`): Nominal flight, cooperative MeshDAA, VLM debounce/conformal checks.
//!   - Phase 1 (`2500..5000`): Degraded Optical-ODD (rain/glare), 5-cam/4-cam/transient-coast RTA,
//!     and Ring-1 monotonic pessimistic tightening.
//!   - Phase 2 (`5000..7500`): Hard faults, P0 `InvalidInhibit` F3 turn dominance, staged P0->P1
//!     recovery, Mode-2 icing, non-finite NaN/Inf sanitization, VLM CRC fault, and permanent
//!     `PluginOffline` mid-flight latching.
//!   - Phase 3 (`7500..10000`): GNSS-denied `+120 ms` TCXO clock offset (+20 ppm continuous drift),
//!     uncalibrated fail-closed Tier-B fallback, calibrated two-way RF `PeerClockOffsetEstimator`
//!     lock with median-of-5 outlier rejection, `RttExceeded` rejection, and replay rejection.

#![no_std]
#![no_main]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::fmt::Write;
use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};

use xvs_active_range::{
    decode_wire, encode_wire, ActiveRangeConfig, ActiveRangeController, ActiveRangeMeasurement,
    OwnshipStateNed, EVADE_BANK_RAD,
};
use xvs_math_core::mesh_daa::OpenXvsBearingTrack;
use xvs_mcu_check::{
    openxvs_ring0_mcu_struct_bytes, openxvs_ring0_mcu_tick, Ring0McuInput, ACTUATOR_AUTHORITY,
    NON_FLIGHT_SIM_ONLY,
};
use xvs_multiship_coord::{
    ClockSyncRejectReason, CoordinatedActiveRange, PeerClockOffsetEstimator, PeerCoordHint,
    Ring0AuthorityGate, StatefulMeshDaaGuard, SwerveCoordRole, TwoWaySyncExchange,
    MAX_OFFSET_UNCERT_US,
};
use xvs_odd_vlm::{
    AsymmetricSimplexVlmGate, GateStatus, OddAction, OpticalOddMonitor, OpticalOddSensors,
    VlmHazard, VlmWirePacket, UNUSED_ZONE_ID,
};
use xvs_os_hal::{
    verify_firmware_and_ekf_gate, AirframeClass, AutopilotStack, AvoidanceVector,
    CivilMissionClass, HalError, ManifestRejectReason, PluginHealth48B, PluginLifecycle,
    PluginManifestPolicy, PluginSupervisorState, Ring1AdvisoryPacket,
    SimplexMode as HalSimplexMode, SupervisorFault, RING1_ADVISORY_MAGIC_VER,
};
use xvs_rta_core::{
    build_rta_input_from_f64, RtaFloatTelemetry, SimplexMode as RtaSimplexMode, SimplexRta,
};

const TOTAL_TICKS: u32 = 10_000;
const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv_u64(mut hash: u64, value: u64) -> u64 {
    let bytes = value.to_le_bytes();
    let mut i = 0usize;
    while i < 8 {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV1A_PRIME);
        i += 1;
    }
    hash
}

#[inline]
const fn rta_to_hal_mode(mode: RtaSimplexMode) -> HalSimplexMode {
    match mode {
        RtaSimplexMode::NominalMl => HalSimplexMode::NominalMl,
        RtaSimplexMode::Isolated5Cam => HalSimplexMode::Isolated5Cam,
        RtaSimplexMode::DegradedCoast => HalSimplexMode::DegradedCoast,
        RtaSimplexMode::FallbackD0Safe => HalSimplexMode::FallbackD0Safe,
        RtaSimplexMode::InvalidInhibit => HalSimplexMode::InvalidInhibit,
    }
}

struct BenchSummary {
    ticks_executed: u32,
    ticks_passed: u32,
    invariants_checked: u64,
    invariant_violations: u32,
    rta_mode_transitions: u32,
    odd_action_transitions: u32,
    rta_mode_counts: [u32; 5],
    odd_action_counts: [u32; 4],
    p0_inhibit_verified_ticks: u32,
    ar_crc_verified_ticks: u32,
    vlm_crc_reject_ticks: u32,
    hal_offline_latched_ticks: u32,
    clock_sync_tier_b_fallback_ticks: u32,
    clock_sync_locked_ticks: u32,
    clock_sync_rtt_rejects: u32,
    clock_sync_replay_rejects: u32,
    clock_sync_outlier_spikes_survived: u32,
    max_clock_sync_residual_us: u64,
    final_clock_sync_true_offset_us: i64,
    final_clock_sync_est_offset_us: i64,
    final_clock_sync_residual_us: u64,
    final_clock_sync_uncert_us: u64,
    ring0_struct_bytes_total: u32,
    checksum_fnv1a64: u64,
}

#[inline(never)]
fn run_10k_benchmark(out: &mut Option<hio::HostStream>) -> BenchSummary {
    let mut summary = BenchSummary {
        ticks_executed: 0,
        ticks_passed: 0,
        invariants_checked: 0,
        invariant_violations: 0,
        rta_mode_transitions: 0,
        odd_action_transitions: 0,
        rta_mode_counts: [0; 5],
        odd_action_counts: [0; 4],
        p0_inhibit_verified_ticks: 0,
        ar_crc_verified_ticks: 0,
        vlm_crc_reject_ticks: 0,
        hal_offline_latched_ticks: 0,
        clock_sync_tier_b_fallback_ticks: 0,
        clock_sync_locked_ticks: 0,
        clock_sync_rtt_rejects: 0,
        clock_sync_replay_rejects: 0,
        clock_sync_outlier_spikes_survived: 0,
        max_clock_sync_residual_us: 0,
        final_clock_sync_true_offset_us: 0,
        final_clock_sync_est_offset_us: 0,
        final_clock_sync_residual_us: 0,
        final_clock_sync_uncert_us: 0,
        ring0_struct_bytes_total: openxvs_ring0_mcu_struct_bytes(),
        checksum_fnv1a64: FNV1A_OFFSET_BASIS,
    };

    let check = |cond: bool, checked: &mut u64, viols: &mut u32| {
        *checked = checked.saturating_add(1);
        if !cond {
            *viols = viols.saturating_add(1);
        }
    };

    // Compile-time + boot-time sanity checks on static constants & library C-ABI entrypoint
    check(
        NON_FLIGHT_SIM_ONLY && !ACTUATOR_AUTHORITY,
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );
    let lib_probe_inp = Ring0McuInput {
        own_aircraft_id: 1,
        cam_health_mask: 0x3F,
        approach_active: 0,
        boson_ffc_freeze: 0,
        gnss_denied: 1,
        now_ms: 10_000,
        now_mono_ns: 10_000_000_000,
        peer_clock_offset_us: 120_000,
        alpha_rad: 0.01,
        beta_rad: 0.0,
        size_px: 15.0,
        size_rate_px_s: 0.75,
        tas_mps: 35.0,
        horizon_pl_rad: 0.004,
        v_vpl_m: 2.0,
        v_hpl_m: 5.0,
        droplet_occ_permille: 0,
        ring1_odd_tighten_mask: 0,
    };
    let lib_probe_out = openxvs_ring0_mcu_tick(&lib_probe_inp);
    check(
        lib_probe_out.actuator_authority == 0
            && lib_probe_out.hal_fw_gate_ok == 1
            && lib_probe_out.active_range_wire_crc_ok == 1
            && lib_probe_out.mesh_num_fused == 1
            && lib_probe_out.clock_sync_fallback_tier_b == 0
            && lib_probe_out.stack_struct_bytes_total == summary.ring0_struct_bytes_total,
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );

    // Verify static HAL manifest, firmware gate, airframe lock, and PluginHealth48B codec
    let valid_policy = PluginManifestPolicy {
        mission_class_raw: CivilMissionClass::SearchAndRescue as u16,
        non_itar_civil_only: true,
        ed25519_sig_verified: true,
        actuator_authority: false,
        max_advisory_bytes: 64,
        max_heartbeat_miss_ms: 200,
        swap_max_bytes: 0,
        orin_requests_gpu: false,
        requests_rw_ring0_topic: false,
        network_enabled: false,
        memory_max_mb: 128,
        pids_max: 4,
    };
    let bad_policy = PluginManifestPolicy {
        actuator_authority: true,
        ..valid_policy
    };
    check(
        valid_policy.validate() == Ok(CivilMissionClass::SearchAndRescue)
            && bad_policy.validate() == Err(ManifestRejectReason::ActuatorAuthorityForbidden),
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );
    check(
        verify_firmware_and_ekf_gate(
            AutopilotStack::ArduPlane {
                major: 4,
                minor: 7,
                patch: 2,
            },
            true,
        )
        .is_ok()
            && verify_firmware_and_ekf_gate(
                AutopilotStack::ArduPlane {
                    major: 4,
                    minor: 7,
                    patch: 1,
                },
                true,
            ) == Err(HalError::FirmwareGateFailed),
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );
    let valid_escape = AvoidanceVector {
        target_course_cdeg: 9000,
        target_alt_mm: 120_000,
        target_eas_cm_s: 2500,
        is_velocity_only: false,
        is_multicopter_only: false,
        pitch_direct_cdeg: None,
        throttle_direct_permille: None,
    };
    let invalid_fw_escape = AvoidanceVector {
        is_velocity_only: true,
        ..valid_escape
    };
    check(
        valid_escape
            .validate_for_airframe(AirframeClass::FixedWing)
            .is_ok()
            && invalid_fw_escape.validate_for_airframe(AirframeClass::FixedWing)
                == Err(HalError::InvalidAirframeCmd),
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );
    let health_pkt = PluginHealth48B {
        plugin_id_hash: 0x0123_4567_89AB_CDEF,
        timestamp_mono_us: 10_000_000,
        last_seq_no: 1,
        ieee_crc32: 0,
        missed_heartbeats: 0,
        rss_mb: 48,
        cpu_permille: 120,
        lifecycle_state: PluginLifecycle::Healthy as u8,
        civil_mission_class: CivilMissionClass::SearchAndRescue as u8,
        reserved_zero: [0u8; 12],
    }
    .with_valid_crc();
    let health_wire = health_pkt.to_le_bytes();
    check(
        PluginHealth48B::from_le_bytes_verified(&health_wire) == Ok(health_pkt),
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );

    // Persistent Ring-0 stateful controllers across all 10,000 ticks (zero heap)
    let mut rta = SimplexRta::new();
    let mut odd_mon = OpticalOddMonitor::new_canonical_60hz();
    let mut vlm_gate = AsymmetricSimplexVlmGate::new(500);
    let mut sup_nominal = PluginSupervisorState::new(200);
    let mut sup_fault_latch = PluginSupervisorState::new(200);
    let mut ar_ctrl = ActiveRangeController::new(ActiveRangeConfig::default());
    let mut coord_ctrl = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
    let mut clock_sync = PeerClockOffsetEstimator::new();
    let mut mesh_guard = StatefulMeshDaaGuard::new();

    let mut prev_rta_mode = rta.mode();
    let mut prev_odd_action = OddAction::Nominal;
    let asym_table: [i64; 5] = [-120, -60, 0, 60, 120];
    let base_x = 6_378_137.0_f64;

    let mut tick: u32 = 0;
    while tick < TOTAL_TICKS {
        let viols_before = summary.invariant_violations;
        let now_ms: u64 = 10_000u64 + (tick as u64) * 100u64;
        let now_mono_ns: u64 = now_ms * 1_000_000u64;
        let now_local_us: u64 = now_ms * 1_000u64;

        // -----------------------------------------------------------------
        // 1. xvs-rta-core: 5-Mode ASTM F3269-21 Simplex RTA + MLTL
        // -----------------------------------------------------------------
        let cam_health_mask: u8 = if (2500..3200).contains(&tick) {
            0x1F // 5 cams -> Isolated5Cam (P3)
        } else if (3200..3900).contains(&tick) {
            0x0F // 4 cams -> FallbackD0Safe (P1)
        } else if (5000..5700).contains(&tick) {
            0x03 // 2 cams < MIN_HEALTHY_CAMS -> InvalidInhibit (P0)
        } else {
            0x3F // 6 healthy cams
        };
        let transient_dropout = (4000..4005).contains(&tick); // 5 ticks < 10 -> DegradedCoast (P2)
        let in_hard_fault = (5000..5700).contains(&tick);
        let float_tel = RtaFloatTelemetry {
            horizon_pl_rad: if in_hard_fault { 0.04 } else { 0.004 },
            horizon_al_rad: 0.017_453_3,
            v_vpl_m: if in_hard_fault { 15.0 } else { 2.0 },
            v_val_m: 10.0,
            v_hpl_m: if in_hard_fault { 45.0 } else { 5.0 },
            v_hal_m: 30.0,
            epipolar_max_rad: [0.0002; 6],
            signed_horizon_residual_rad: 0.0,
            runway_width_delta_m: 0.0,
            runway_width_delta_limit_m: 5.0,
            cusum_stat: 0.4,
            cusum_thr: 5.0,
            ood_global_score: 0.1,
            ood_global_thr: 0.7,
            ood_cam_scores: [0.1; 6],
            ood_cam_thr: 0.7,
            cam_health_mask,
            ml_detector_fault: false,
            transient_dropout,
            approach_active: in_hard_fault,
            go_around_ack_reset: (5700..5710).contains(&tick),
        };
        let rta_inp = build_rta_input_from_f64(&float_tel);
        let rta_out = rta.step(&rta_inp);

        if rta_out.mode != prev_rta_mode {
            // Hard ASTM F3269-21 staged recovery invariant: P0 NEVER jumps directly to P2/P3/P4
            if prev_rta_mode == RtaSimplexMode::InvalidInhibit {
                check(
                    rta_out.mode == RtaSimplexMode::FallbackD0Safe,
                    &mut summary.invariants_checked,
                    &mut summary.invariant_violations,
                );
            }
            summary.rta_mode_transitions = summary.rta_mode_transitions.saturating_add(1);
            prev_rta_mode = rta_out.mode;
        }
        let mode_idx = (rta_out.mode as usize).min(4);
        summary.rta_mode_counts[mode_idx] = summary.rta_mode_counts[mode_idx].saturating_add(1);

        if in_hard_fault {
            check(
                rta_out.mode == RtaSimplexMode::InvalidInhibit
                    && rta_out.inhibit_avoidance_turns
                    && rta_out.blank_conformal_hud
                    && rta_out.mandatory_go_around,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        }
        if tick == 4002 {
            check(
                rta_out.mode == RtaSimplexMode::DegradedCoast,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        }

        // -----------------------------------------------------------------
        // 2. xvs-odd-vlm: DR-13 5-Mode Optical-ODD + DR-09 6-Stage VLM Gate
        // -----------------------------------------------------------------
        let rain_phase = (2500..3200).contains(&tick);
        let glare_phase = (3200..3900).contains(&tick);
        let icing_phase = (5000..5700).contains(&tick);
        let ffc_freeze = tick == 5500 || tick == 7600;
        let odd_sensors = OpticalOddSensors {
            droplet_occ_permille: if rain_phase { 220 } else { 0 },
            blur_mlap_centi: 9500,
            mtf_hf_ratio_permille: if icing_phase { 90 } else { 350 },
            lwir_std_centidn: if icing_phase { 300 } else { 1200 },
            dew_margin_centic: if icing_phase { 100 } else { 600 },
            sun_boresight_cdeg: if glare_phase { 2500 } else { 7500 },
            eo_hfov_cdeg: 6000,
            glare_sat_permille: if glare_phase { 80 } else { 5 },
            glare_ring_ratio_permille: if glare_phase { 150 } else { 600 },
            mor_m: 8000,
            horizon_snr_centidb: 1200,
            lwir_median_drift_dn: 40,
            lwir_fpn_ratio_permille: 80,
            lwir_bad_px_ppm: 500,
            boson_dt_ffc_s: 30,
            boson_temp_delta_centic: 20,
            boson_ffc_freeze: ffc_freeze,
            attitude_in_odd: true,
        };
        let odd_out = odd_mon.evaluate(&odd_sensors);

        if odd_out.action != prev_odd_action {
            summary.odd_action_transitions = summary.odd_action_transitions.saturating_add(1);
            prev_odd_action = odd_out.action;
        }
        let odd_idx = (odd_out.action as usize).min(3);
        summary.odd_action_counts[odd_idx] = summary.odd_action_counts[odd_idx].saturating_add(1);

        // AF447 hard invariant + Boson FFC mask check
        check(
            odd_out.airdata_export_blocked && (!ffc_freeze || odd_out.mask_lwir_frame),
            &mut summary.invariants_checked,
            &mut summary.invariant_violations,
        );

        let corrupt_vlm_crc = (5300..5350).contains(&tick);
        let conformal_abstain = (200..210).contains(&tick);
        let vlm_hazard = if (100..110).contains(&tick) {
            VlmHazard::BirdFlock as u8
        } else {
            VlmHazard::None as u8
        };
        let vlm_pkt = VlmWirePacket {
            seq: tick + 1,
            capture_ts_ms: now_ms - 20,
            hazard_enum: vlm_hazard,
            azimuth_sector: (tick % 12) as u8,
            ranked_zone_ids: [0, 1, UNUSED_ZONE_ID, UNUSED_ZONE_ID],
            conformal_valid: u8::from(!conformal_abstain),
            aleatoric_flag: 0,
            epistemic_flag: u8::from(conformal_abstain),
            crc32: 0,
        };
        let mut vlm_wire = vlm_pkt.encode();
        if corrupt_vlm_crc {
            vlm_wire[31] ^= 0xFF;
        }
        let cert_hazards_mask: u8 = if tick.is_multiple_of(250) {
            VlmHazard::RunwayIncursion.bit()
        } else {
            0
        };
        let vlm_dec = vlm_gate.evaluate_wire(
            Some(&vlm_wire),
            now_ms,
            !odd_out.vlm_gate_reject,
            cert_hazards_mask,
            0b0011,
        );

        check(
            !vlm_dec.actuator_authority
                && (vlm_dec.final_hazards_mask & cert_hazards_mask) == cert_hazards_mask,
            &mut summary.invariants_checked,
            &mut summary.invariant_violations,
        );
        if corrupt_vlm_crc {
            check(
                vlm_dec.status == GateStatus::SchemaOrCrcFail && !vlm_dec.vlm_used,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            summary.vlm_crc_reject_ticks = summary.vlm_crc_reject_ticks.saturating_add(1);
        } else if odd_out.vlm_gate_reject {
            check(
                vlm_dec.status == GateStatus::OpticalOddMasked && !vlm_dec.vlm_used,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        } else if conformal_abstain {
            check(
                vlm_dec.status == GateStatus::ConformalAbstain && !vlm_dec.vlm_used,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        }

        // -----------------------------------------------------------------
        // 3. xvs-os-hal: Ring-1 Monotonic Pessimistic Supervisor + Crash Latch
        // -----------------------------------------------------------------
        let base_hal_mode = rta_to_hal_mode(rta_out.mode);
        let ring1_tighten_mask: u32 = if (3200..3900).contains(&tick) {
            0x0000_0010 // FallbackD0Safe floor
        } else if (3000..3200).contains(&tick) {
            0x0000_0001 // Isolated5Cam floor
        } else {
            0
        };
        let ring1_pkt = Ring1AdvisoryPacket {
            magic_ver: RING1_ADVISORY_MAGIC_VER,
            civil_mission_class: CivilMissionClass::SearchAndRescue as u16,
            advisory_flags: 0,
            plugin_id_hash: 0x0123_4567_89AB_CDEF,
            seq_no: (tick as u64) + 1,
            timestamp_mono_ns: now_mono_ns - 10_000_000,
            odd_tighten_mask: ring1_tighten_mask,
            roi_lat_e7: 374_220_000,
            roi_lon_e7: -1_220_840_000,
            roi_alt_mm: 120_000,
            confidence_q15: 28_000,
            reserved_zero: [0u8; 10],
            ieee_crc32: 0,
        }
        .with_valid_crc();
        let ring1_wire = ring1_pkt.to_le_bytes();
        let sup_dec = sup_nominal.ingest_wire(&ring1_wire, now_mono_ns, base_hal_mode, 0);

        check(
            sup_dec.fault == SupervisorFault::None
                && !sup_dec.actuator_authority
                && (sup_dec.out_mode as u8) >= (base_hal_mode as u8)
                && sup_dec.out_mode.to_rta_wire_u8() <= (rta_out.mode as u8),
            &mut summary.invariants_checked,
            &mut summary.invariant_violations,
        );

        let mut fault_wire = ring1_wire;
        if tick == 6500 {
            fault_wire[60] ^= 0x01; // Inject CRC mismatch at tick 6500
        }
        let sup_fault_dec = sup_fault_latch.ingest_wire(&fault_wire, now_mono_ns, base_hal_mode, 0);
        if tick < 6500 {
            check(
                sup_fault_dec.fault == SupervisorFault::None && !sup_fault_dec.actuator_authority,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        } else if tick == 6500 {
            check(
                sup_fault_dec.lifecycle == PluginLifecycle::PluginOffline
                    && sup_fault_dec.fault == SupervisorFault::Crc32Mismatch
                    && !sup_fault_dec.actuator_authority,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            summary.hal_offline_latched_ticks = summary.hal_offline_latched_ticks.saturating_add(1);
        } else {
            check(
                sup_fault_dec.lifecycle == PluginLifecycle::PluginOffline
                    && sup_fault_dec.fault == SupervisorFault::AlreadyLatchedOffline
                    && sup_fault_dec.out_mode == base_hal_mode
                    && !sup_fault_dec.actuator_authority,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            summary.hal_offline_latched_ticks = summary.hal_offline_latched_ticks.saturating_add(1);
        }

        // -----------------------------------------------------------------
        // 4. xvs-active-range: M=6 MSC-RPEKF + CP-MSC + SC3BF-FIM + 64B CRC32
        // -----------------------------------------------------------------
        let own = OwnshipStateNed {
            v_ned: [35.0, 0.0, 0.0],
            a_ned: [0.0, 0.0, 0.0],
            heading_rad: 0.0,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let inject_nonfinite = (5200..5250).contains(&tick);
        let meas = if inject_nonfinite {
            ActiveRangeMeasurement {
                alpha_rad: f64::NAN,
                beta_rad: f64::INFINITY,
                size_px: f64::NAN,
                size_rate_px_s: f64::NEG_INFINITY,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: 6 + (tick % 60),
                dt_s: 0.1,
            }
        } else {
            ActiveRangeMeasurement {
                alpha_rad: 0.0015 * (((tick % 20) as f64) - 10.0),
                beta_rad: 0.0004 * (((tick % 10) as f64) - 5.0),
                size_px: 14.0 + 0.08 * ((tick % 50) as f64),
                size_rate_px_s: 0.70 + 0.01 * ((tick % 20) as f64),
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: 6 + (tick % 60),
                dt_s: 0.1,
            }
        };
        let ar_pkt = ar_ctrl.step(&own, &meas);
        let ar_wire = encode_wire(&ar_pkt);
        let ar_crc_ok = decode_wire(&ar_wire) == Some(ar_pkt);
        check(
            ar_crc_ok
                && ar_pkt.r_hat_m.is_finite()
                && ar_pkt.r_safe_lower_m.is_finite()
                && ar_pkt.cmd_bank_rad.is_finite()
                && ar_pkt.r_safe_lower_m >= 0.0
                && ar_pkt.r_safe_lower_m <= ar_pkt.r_hat_m + 1.0e-3,
            &mut summary.invariants_checked,
            &mut summary.invariant_violations,
        );
        if ar_crc_ok {
            summary.ar_crc_verified_ticks = summary.ar_crc_verified_ticks.saturating_add(1);
        }

        // -----------------------------------------------------------------
        // 5. xvs-multiship-coord: F1 + F3 P0 Dominance + F4 MeshGuard + +120 ms TCXO Clock Sync
        // -----------------------------------------------------------------
        let gate = Ring0AuthorityGate {
            rta_inhibit_avoidance_turns: rta_out.inhibit_avoidance_turns,
            hal_out_mode_wire: sup_dec.out_mode.to_rta_wire_u8(),
        };
        let peer_hint = if tick < 2500 {
            Some(PeerCoordHint {
                peer_sender_id: 2,
                peer_ts_utc_us: now_local_us - 10_000,
                now_utc_us: now_local_us,
                peer_a_ned: None,
            })
        } else {
            None
        };
        let coord_pkt = coord_ctrl.step(&own, &meas, peer_hint, gate);
        let coord_role = coord_ctrl.last_role();
        check(
            coord_pkt.cmd_bank_rad.is_finite()
                && coord_pkt.cmd_bank_rad.abs() <= (EVADE_BANK_RAD as f32) + 1.0e-5,
            &mut summary.invariants_checked,
            &mut summary.invariant_violations,
        );
        if gate.inhibits_turns() {
            check(
                coord_role == SwerveCoordRole::InhibitedByP0 && coord_pkt.cmd_bank_rad == 0.0,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            summary.p0_inhibit_verified_ticks = summary.p0_inhibit_verified_ticks.saturating_add(1);
        }

        // GNSS-denied +120 ms TCXO offset & +20 ppm continuous drift profile in Phase 3
        let gnss_denied = tick >= 7500;
        let true_offset_us: i64 = if tick < 7500 {
            0
        } else if tick < 7800 {
            120_000 // +120 ms uncalibrated TCXO offset -> MUST fail closed to Tier-B
        } else {
            120_000 + ((tick - 7800) as i64) * 2 // +120 ms + 20 ppm drift (+2 us / 100 ms tick)
        };

        if tick >= 7800 {
            let is_outlier_tick = (7820..9600).contains(&tick) && tick.is_multiple_of(100);
            let asym_us = if is_outlier_tick {
                15_000 // +15 ms single-sample RF multipath spike (rejected by median-of-5)
            } else {
                asym_table[(tick as usize) % 5]
            };
            let t1 = now_local_us - 4_000;
            let fwd_plus_off = ((1_500_i64 + asym_us) + true_offset_us) as u64;
            let t2 = t1 + fwd_plus_off;
            let t3 = t2 + 500;
            let t4 = t1 + 3_500; // exact rtt = 3_000 us -> rtt/2 = 1_500 us
            let out2 = clock_sync.record_exchange(TwoWaySyncExchange {
                sender_id: 2,
                t1_local_tx_us: t1,
                t2_peer_rx_us: t2,
                t3_peer_tx_us: t3,
                t4_local_rx_us: t4,
            });
            let out3 = clock_sync.record_exchange(TwoWaySyncExchange {
                sender_id: 3,
                t1_local_tx_us: t1,
                t2_peer_rx_us: t2,
                t3_peer_tx_us: t3,
                t4_local_rx_us: t4,
            });
            check(
                out2.accepted && out3.accepted,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            if is_outlier_tick {
                let residual_after_spike = (out2.estimate.offset_us - true_offset_us).unsigned_abs();
                check(
                    out2.estimate.synchronized && residual_after_spike <= 2_000,
                    &mut summary.invariants_checked,
                    &mut summary.invariant_violations,
                );
                summary.clock_sync_outlier_spikes_survived =
                    summary.clock_sync_outlier_spikes_survived.saturating_add(1);
            }

            // Phase 3c adversarial RTT-exceeded (>150 ms) sync exchange rejection probe
            if tick >= 9600 && tick.is_multiple_of(20) {
                let bad_rtt = clock_sync.record_exchange(TwoWaySyncExchange {
                    sender_id: 2,
                    t1_local_tx_us: now_local_us - 200_000,
                    t2_peer_rx_us: now_local_us - 100_000 + (true_offset_us as u64),
                    t3_peer_tx_us: now_local_us - 99_500 + (true_offset_us as u64),
                    t4_local_rx_us: now_local_us - 19_500, // rtt = 180_000 us > 150_000 us
                });
                check(
                    !bad_rtt.accepted
                        && bad_rtt.reject_reason == Some(ClockSyncRejectReason::RttExceeded),
                    &mut summary.invariants_checked,
                    &mut summary.invariant_violations,
                );
                summary.clock_sync_rtt_rejects = summary.clock_sync_rtt_rejects.saturating_add(1);
            }
        }

        let obs_local_ts_us = now_local_us - 10_000;
        let peer_obs_ts_us = obs_local_ts_us + (true_offset_us as u64);
        let obs_own = OpenXvsBearingTrack::from_ecef_ray(
            obs_local_ts_us,
            [base_x, 0.0, 0.0],
            [500.0, 200.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            1,
            0,
        );
        let obs_p2 = OpenXvsBearingTrack::from_ecef_ray(
            peer_obs_ts_us,
            [base_x, 1000.0, 0.0],
            [500.0, -800.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            2,
            0,
        );
        let obs_p3 = OpenXvsBearingTrack::from_ecef_ray(
            peer_obs_ts_us,
            [base_x + 1000.0, 500.0, 0.0],
            [-500.0, -300.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            3,
            0,
        );

        let inject_replay = tick >= 9600 && tick.is_multiple_of(25);
        let mesh_res = if inject_replay {
            // 160 ms old (<= 200 ms MAX_MESH_AGE_US, passes staleness gate) yet strictly older
            // than the previous committed tick watermark (110 ms old) -> MUST hit RejectReplay!
            let obs_replay = OpenXvsBearingTrack::from_ecef_ray(
                peer_obs_ts_us - 150_000,
                [base_x, 1000.0, 0.0],
                [500.0, -800.0, 0.0],
                [0.0; 3],
                1500,
                100,
                1,
                2,
                0,
            );
            let obs4 = [obs_own, obs_p2, obs_p3, obs_replay];
            mesh_guard.fuse_guarded_with_clock_sync(&obs4, now_local_us, 1, gnss_denied, &clock_sync)
        } else {
            let obs3 = [obs_own, obs_p2, obs_p3];
            mesh_guard.fuse_guarded_with_clock_sync(&obs3, now_local_us, 1, gnss_denied, &clock_sync)
        };

        let est2 = clock_sync.estimate_peer(2, now_local_us);
        if tick < 7500 {
            check(
                mesh_res.num_fused == 1
                    && !mesh_res.fallback_to_tier_b_optical
                    && mesh_res.stale_obs_rejected == 0
                    && mesh_res.replay_obs_rejected == 0,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
        } else if tick < 7802 {
            // Uncalibrated (<3 samples) under GNSS-denied +120 ms offset MUST fail closed to Tier-B
            check(
                mesh_res.fallback_to_tier_b_optical
                    && mesh_res.uncalibrated_obs_rejected == 2
                    && mesh_res.num_fused == 0,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            summary.clock_sync_tier_b_fallback_ticks =
                summary.clock_sync_tier_b_fallback_ticks.saturating_add(1);
        } else {
            // Calibrated GNSS-denied +120 ms + 20 ppm drift: MUST stay locked, residual <= 2 ms, fuse cleanly
            let est3 = clock_sync.estimate_peer(3, now_local_us);
            let res_us = (est2.offset_us - true_offset_us).unsigned_abs();
            if res_us > summary.max_clock_sync_residual_us {
                summary.max_clock_sync_residual_us = res_us;
            }
            summary.final_clock_sync_true_offset_us = true_offset_us;
            summary.final_clock_sync_est_offset_us = est2.offset_us;
            summary.final_clock_sync_residual_us = res_us;
            summary.final_clock_sync_uncert_us = est2.offset_uncert_us;

            check(
                est2.synchronized
                    && est3.synchronized
                    && est2.offset_uncert_us <= MAX_OFFSET_UNCERT_US
                    && res_us <= 2_000
                    && mesh_res.num_fused == 1
                    && !mesh_res.fallback_to_tier_b_optical
                    && mesh_res.stale_obs_rejected == 0,
                &mut summary.invariants_checked,
                &mut summary.invariant_violations,
            );
            if inject_replay {
                let replay_caught = mesh_res.replay_obs_rejected == 1;
                check(
                    replay_caught,
                    &mut summary.invariants_checked,
                    &mut summary.invariant_violations,
                );
                if replay_caught {
                    summary.clock_sync_replay_rejects =
                        summary.clock_sync_replay_rejects.saturating_add(1);
                }
            } else {
                check(
                    mesh_res.replay_obs_rejected == 0,
                    &mut summary.invariants_checked,
                    &mut summary.invariant_violations,
                );
            }
            summary.clock_sync_locked_ticks = summary.clock_sync_locked_ticks.saturating_add(1);
        }

        // Deterministic 64-bit FNV-1a checksum across ALL 5 Ring-0 crate outputs
        let packed_modes: u64 = (rta_out.mode as u64)
            | ((odd_out.action as u64) << 8)
            | ((vlm_dec.status as u64) << 16)
            | ((sup_dec.out_mode.to_rta_wire_u8() as u64) << 24)
            | ((ar_pkt.mode as u64) << 32)
            | ((coord_role as u64) << 40)
            | ((mesh_res.num_fused as u64) << 48)
            | ((u64::from(mesh_res.fallback_to_tier_b_optical)) << 56);
        summary.checksum_fnv1a64 = fnv_u64(summary.checksum_fnv1a64, packed_modes);
        summary.checksum_fnv1a64 = fnv_u64(
            summary.checksum_fnv1a64,
            ((ar_pkt.r_safe_lower_m.to_bits() as u64) << 32)
                | (coord_pkt.cmd_bank_rad.to_bits() as u64),
        );
        summary.checksum_fnv1a64 = fnv_u64(summary.checksum_fnv1a64, est2.offset_us as u64);

        summary.ticks_executed = summary.ticks_executed.saturating_add(1);
        if summary.invariant_violations == viols_before {
            summary.ticks_passed = summary.ticks_passed.saturating_add(1);
        }
        tick += 1;
        if tick == 2500 || tick == 5000 || tick == 7500 || tick == TOTAL_TICKS {
            if let Some(ref mut s) = *out {
                let _ = writeln!(
                    s,
                    "PHASE_PROGRESS tick={}/{} passed={} viols={} rta_mode={} odd_action={} checksum=0x{:016x}",
                    tick,
                    TOTAL_TICKS,
                    summary.ticks_passed,
                    summary.invariant_violations,
                    rta_out.mode as u8,
                    odd_out.action as u8,
                    summary.checksum_fnv1a64
                );
            }
        }
    }

    // Post-run coverage invariants: confirm ALL 5 RTA modes and ALL 4 Optical-ODD actions exercised
    check(
        summary.rta_mode_counts[0] > 0
            && summary.rta_mode_counts[1] > 0
            && summary.rta_mode_counts[2] > 0
            && summary.rta_mode_counts[3] > 0
            && summary.rta_mode_counts[4] > 0,
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );
    check(
        summary.odd_action_counts[0] > 0
            && summary.odd_action_counts[1] > 0
            && summary.odd_action_counts[2] > 0
            && summary.odd_action_counts[3] > 0,
        &mut summary.invariants_checked,
        &mut summary.invariant_violations,
    );

    summary
}

#[entry]
fn main() -> ! {
    let mut stdout = hio::hstdout().ok();
    let s = run_10k_benchmark(&mut stdout);
    let passed = s.ticks_executed == TOTAL_TICKS
        && s.ticks_passed == TOTAL_TICKS
        && s.invariant_violations == 0;

    if let Some(mut out) = stdout {
        let _ = writeln!(out, "OPENXVS_MCU_BENCH target=thumbv7em-none-eabi cpu=cortex-m7 machine=mps2-an500");
        let _ = writeln!(
            out,
            "SCOPE non_flight_sim_only={} actuator_authority={} emulation_only=true physical_stm32_hw=false",
            NON_FLIGHT_SIM_ONLY, ACTUATOR_AUTHORITY
        );
        let _ = writeln!(
            out,
            "TICKS executed={}/{} passed={}/{} invariants_checked={} violations={}",
            s.ticks_executed,
            TOTAL_TICKS,
            s.ticks_passed,
            TOTAL_TICKS,
            s.invariants_checked,
            s.invariant_violations
        );
        let _ = writeln!(
            out,
            "RTA_MODES transitions={} p0={} p1={} p2={} p3={} p4={} p0_inhibit_verified={}",
            s.rta_mode_transitions,
            s.rta_mode_counts[0],
            s.rta_mode_counts[1],
            s.rta_mode_counts[2],
            s.rta_mode_counts[3],
            s.rta_mode_counts[4],
            s.p0_inhibit_verified_ticks
        );
        let _ = writeln!(
            out,
            "ODD_VLM_HAL odd_transitions={} nominal={} evs_lwir={} thermal={} degraded={} ar_crc_ok={} vlm_crc_rej={} hal_offline_latched={}",
            s.odd_action_transitions,
            s.odd_action_counts[0],
            s.odd_action_counts[1],
            s.odd_action_counts[2],
            s.odd_action_counts[3],
            s.ar_crc_verified_ticks,
            s.vlm_crc_reject_ticks,
            s.hal_offline_latched_ticks
        );
        let _ = writeln!(
            out,
            "CLOCK_SYNC_120MS tier_b_fallback={} locked={} max_residual_us={} final_true_offset_us={} final_est_offset_us={} final_residual_us={} final_uncert_us={} outlier_survived={} rtt_rej={} replay_rej={}",
            s.clock_sync_tier_b_fallback_ticks,
            s.clock_sync_locked_ticks,
            s.max_clock_sync_residual_us,
            s.final_clock_sync_true_offset_us,
            s.final_clock_sync_est_offset_us,
            s.final_clock_sync_residual_us,
            s.final_clock_sync_uncert_us,
            s.clock_sync_outlier_spikes_survived,
            s.clock_sync_rtt_rejects,
            s.clock_sync_replay_rejects
        );
        let _ = writeln!(
            out,
            "FOOTPRINT ring0_struct_bytes={} checksum_fnv1a64=0x{:016x} verdict={}",
            s.ring0_struct_bytes_total,
            s.checksum_fnv1a64,
            if passed { "PASS" } else { "FAIL" }
        );
    }

    if passed {
        debug::exit(debug::EXIT_SUCCESS);
    } else {
        debug::exit(debug::EXIT_FAILURE);
    }
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    debug::exit(debug::EXIT_FAILURE);
    loop {
        core::hint::spin_loop();
    }
}
