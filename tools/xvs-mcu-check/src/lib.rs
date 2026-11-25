// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Bare-Metal ARM Cortex-M7 (`thumbv7em-none-eabi`) Ring-0 Compilation Proof.
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocation (`0` `alloc` / `std`),
//! `NON_FLIGHT_SIM_ONLY = true`, `ACTUATOR_AUTHORITY = false`, strictly NED/ECEF/RDF.
//! Executes a single deterministic Ring-0 tick across ALL FIVE OpenXVS Ring-0 crates:
//!   1. `xvs-rta-core` (`SimplexRta` + `build_rta_input_from_f64`)
//!   2. `xvs-odd-vlm` (`OpticalOddMonitor` + `AsymmetricSimplexVlmGate`)
//!   3. `xvs-os-hal` (`PluginManifestPolicy` + `PluginSupervisorState` + `verify_firmware_and_ekf_gate`)
//!   4. `xvs-active-range` (`ActiveRangeController` + 64B IEEE-802.3 CRC32 wire codec)
//!   5. `xvs-multiship-coord` (`CoordinatedActiveRange` F1/F3 + `StatefulMeshDaaGuard` F4)

#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::mem::size_of;

use xvs_active_range::{
    decode_wire, encode_wire, ActiveRangeConfig, ActiveRangeController, ActiveRangeMeasurement,
    OwnshipStateNed,
};
use xvs_math_core::mesh_daa::OpenXvsBearingTrack;
use xvs_multiship_coord::{
    CoordinatedActiveRange, PeerClockOffsetEstimator, Ring0AuthorityGate, StatefulMeshDaaGuard,
    TwoWaySyncExchange,
};
use xvs_odd_vlm::{
    AsymmetricSimplexVlmGate, OpticalOddMonitor, OpticalOddSensors, VlmHazard, VlmWirePacket,
    UNUSED_ZONE_ID,
};
use xvs_os_hal::{
    verify_firmware_and_ekf_gate, AirframeClass, AutopilotStack, AvoidanceVector,
    CivilMissionClass, PluginManifestPolicy, PluginSupervisorState, Ring1AdvisoryPacket,
    SimplexMode as HalSimplexMode, RING1_ADVISORY_MAGIC_VER,
};
use xvs_rta_core::{build_rta_input_from_f64, RtaFloatTelemetry, SimplexRta};

pub const NON_FLIGHT_SIM_ONLY: bool = true;
pub const ACTUATOR_AUTHORITY: bool = false;

const _: () = {
    assert!(NON_FLIGHT_SIM_ONLY);
    assert!(!ACTUATOR_AUTHORITY);
    assert!(!AsymmetricSimplexVlmGate::ACTUATOR_AUTHORITY);
    assert!(!PluginSupervisorState::ACTUATOR_AUTHORITY);
    assert!(OpticalOddMonitor::<30, 90, 30, 60, 60>::AIRDATA_EXPORT_BLOCKED);
};

/// FFI-safe `#[repr(C)]` Ring-0 MCU tick input vector (NED / ECEF / SI units).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ring0McuInput {
    pub own_aircraft_id: u8,
    pub cam_health_mask: u8,
    pub approach_active: u8,
    pub boson_ffc_freeze: u8,
    pub gnss_denied: u8,
    pub now_ms: u64,
    pub now_mono_ns: u64,
    pub peer_clock_offset_us: i64,
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub size_px: f64,
    pub size_rate_px_s: f64,
    pub tas_mps: f64,
    pub horizon_pl_rad: f64,
    pub v_vpl_m: f64,
    pub v_hpl_m: f64,
    pub droplet_occ_permille: u16,
    pub ring1_odd_tighten_mask: u32,
}

/// FFI-safe `#[repr(C)]` Ring-0 MCU tick output summary across all 5 Ring-0 crates.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ring0McuOutput {
    pub rta_mode_u8: u8,
    pub odd_action_u8: u8,
    pub vlm_gate_status_u8: u8,
    pub hal_out_mode_u8: u8,
    pub hal_fw_gate_ok: u8,
    pub active_range_wire_crc_ok: u8,
    pub multiship_role_u8: u8,
    pub mesh_num_fused: u8,
    pub clock_sync_fallback_tier_b: u8,
    pub actuator_authority: u8,
    pub r_safe_lower_m: f32,
    pub coord_cmd_bank_rad: f32,
    pub stack_struct_bytes_total: u32,
}

/// Compile-time exact stack footprint (bytes) of all five Ring-0 state structs.
#[inline(never)]
pub extern "C" fn openxvs_ring0_mcu_struct_bytes() -> u32 {
    (size_of::<SimplexRta>()
        + size_of::<OpticalOddMonitor<30, 90, 30, 60, 60>>()
        + size_of::<AsymmetricSimplexVlmGate>()
        + size_of::<PluginSupervisorState>()
        + size_of::<ActiveRangeController>()
        + size_of::<CoordinatedActiveRange>()
        + size_of::<StatefulMeshDaaGuard>()
        + size_of::<PeerClockOffsetEstimator>()) as u32
}

/// Execute one real deterministic zero-heap Ring-0 tick across ALL FIVE bare-metal Rust crates:
/// `xvs-rta-core`, `xvs-odd-vlm`, `xvs-os-hal`, `xvs-active-range`, and `xvs-multiship-coord`.
///
/// Note: Rust 1.100 classifies `#[no_mangle]` as an `unsafe` attribute under `#![forbid(unsafe_code)]`;
/// this crate keeps strict `#![forbid(unsafe_code)]` at crate root and exports the C-ABI entry
/// `pub extern "C" fn openxvs_ring0_mcu_tick` as a bare-metal `thumbv7em-none-eabi` ELF32-ARM `rlib`/`.o`.
#[inline(never)]
pub extern "C" fn openxvs_ring0_mcu_tick(inp: &Ring0McuInput) -> Ring0McuOutput {
    // 1. xvs-rta-core: ASTM F3269-21 Simplex RTA + fail-closed SI i64 quantizer + past-time MLTL
    let mut rta = SimplexRta::new();
    let float_tel = RtaFloatTelemetry {
        horizon_pl_rad: inp.horizon_pl_rad,
        horizon_al_rad: 0.017_453_3,
        v_vpl_m: inp.v_vpl_m,
        v_val_m: 10.0,
        v_hpl_m: inp.v_hpl_m,
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
        cam_health_mask: inp.cam_health_mask,
        ml_detector_fault: false,
        transient_dropout: false,
        approach_active: inp.approach_active != 0,
        go_around_ack_reset: false,
    };
    let rta_inp = build_rta_input_from_f64(&float_tel);
    let rta_out = rta.step(&rta_inp);

    // 2. xvs-odd-vlm: DR-13 5-Mode Optical-ODD MLTL Monitor + DR-09 6-Stage VLM Verification Gate
    let mut odd_mon = OpticalOddMonitor::new_canonical_60hz();
    let odd_sensors = OpticalOddSensors {
        droplet_occ_permille: inp.droplet_occ_permille,
        blur_mlap_centi: 9500,
        mtf_hf_ratio_permille: 350,
        lwir_std_centidn: 1200,
        dew_margin_centic: 600,
        sun_boresight_cdeg: 7500,
        eo_hfov_cdeg: 6000,
        glare_sat_permille: 5,
        glare_ring_ratio_permille: 600,
        mor_m: 8000,
        horizon_snr_centidb: 1200,
        lwir_median_drift_dn: 40,
        lwir_fpn_ratio_permille: 80,
        lwir_bad_px_ppm: 500,
        boson_dt_ffc_s: 30,
        boson_temp_delta_centic: 20,
        boson_ffc_freeze: inp.boson_ffc_freeze != 0,
        attitude_in_odd: true,
    };
    let odd_out = odd_mon.evaluate(&odd_sensors);

    let mut vlm_gate = AsymmetricSimplexVlmGate::new(500);
    let vlm_pkt = VlmWirePacket {
        seq: 1,
        capture_ts_ms: inp.now_ms.saturating_sub(20),
        hazard_enum: VlmHazard::None as u8,
        azimuth_sector: 0,
        ranked_zone_ids: [0, 1, UNUSED_ZONE_ID, UNUSED_ZONE_ID],
        conformal_valid: 1,
        aleatoric_flag: 0,
        epistemic_flag: 0,
        crc32: 0,
    };
    let vlm_wire = vlm_pkt.encode();
    let vlm_dec = vlm_gate.evaluate_wire(
        Some(&vlm_wire),
        inp.now_ms,
        !odd_out.vlm_gate_reject,
        0,
        0b0011,
    );

    // 3. xvs-os-hal: Ring-1 Manifest Policy + Monotonic Pessimistic Supervisor + Firmware/EKF Gate
    let policy = PluginManifestPolicy {
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
    let policy_ok = policy.validate().is_ok();
    let fw_gate_ok = verify_firmware_and_ekf_gate(
        AutopilotStack::ArduPlane {
            major: 4,
            minor: 7,
            patch: 2,
        },
        policy_ok,
    )
    .is_ok();
    let escape_vec = AvoidanceVector {
        target_course_cdeg: 9000,
        target_alt_mm: 120_000,
        target_eas_cm_s: 2500,
        is_velocity_only: false,
        is_multicopter_only: false,
        pitch_direct_cdeg: None,
        throttle_direct_permille: None,
    };
    let hal_airframe_ok = escape_vec.validate_for_airframe(AirframeClass::FixedWing).is_ok();

    let mut sup = PluginSupervisorState::new(200);
    let ring1_pkt = Ring1AdvisoryPacket {
        magic_ver: RING1_ADVISORY_MAGIC_VER,
        civil_mission_class: CivilMissionClass::SearchAndRescue as u16,
        advisory_flags: 0,
        plugin_id_hash: 0x0123_4567_89AB_CDEF,
        seq_no: 1,
        timestamp_mono_ns: inp.now_mono_ns.saturating_sub(10_000_000),
        odd_tighten_mask: inp.ring1_odd_tighten_mask,
        roi_lat_e7: 374_220_000,
        roi_lon_e7: -1_220_840_000,
        roi_alt_mm: 120_000,
        confidence_q15: 28_000,
        reserved_zero: [0u8; 10],
        ieee_crc32: 0,
    }
    .with_valid_crc();
    let ring1_wire = ring1_pkt.to_le_bytes();
    let sup_dec = sup.ingest_wire(
        &ring1_wire,
        inp.now_mono_ns,
        HalSimplexMode::NominalMl,
        0,
    );

    // 4. xvs-active-range: M=6 MSC-RPEKF + CP-MSC + SC3BF-FIM + 64B IEEE-802.3 CRC32 wire codec
    let own = OwnshipStateNed {
        v_ned: [inp.tas_mps, 0.0, 0.0],
        a_ned: [0.0, 0.0, 0.0],
        heading_rad: 0.0,
        tas_mps: inp.tas_mps,
        v_stall_mps: 14.5,
    };
    let meas = ActiveRangeMeasurement {
        alpha_rad: inp.alpha_rad,
        beta_rad: inp.beta_rad,
        size_px: inp.size_px,
        size_rate_px_s: inp.size_rate_px_s,
        contrast_ratio: 0.65,
        sky_beta_ext: 2.0e-4,
        daytime_clear: true,
        track_age: 6,
        dt_s: 0.1,
    };
    let mut ar = ActiveRangeController::new(ActiveRangeConfig::default());
    let ar_pkt = ar.step(&own, &meas);
    let ar_wire = encode_wire(&ar_pkt);
    let ar_crc_ok = decode_wire(&ar_wire) == Some(ar_pkt);

    // 5. xvs-multiship-coord: F1 FAR 91.113 + F3 Ring-0 P0 Arbiter + F4 StatefulMeshDaaGuard + GNSS-denied PeerClockOffsetEstimator
    let mut coord = CoordinatedActiveRange::new(inp.own_aircraft_id, ActiveRangeConfig::default());
    let gate = Ring0AuthorityGate {
        rta_inhibit_avoidance_turns: rta_out.inhibit_avoidance_turns,
        hal_out_mode_wire: sup_dec.out_mode.to_rta_wire_u8(),
    };
    let coord_pkt = coord.step(&own, &meas, None, gate);
    let coord_role = coord.last_role();

    let ts_us = inp.now_ms.saturating_mul(1_000);
    let mut clock_sync = PeerClockOffsetEstimator::new();
    let offset_signed = inp.peer_clock_offset_us;
    let peer_shift = |local_t: u64| -> u64 {
        if offset_signed >= 0 {
            local_t.saturating_add(offset_signed as u64)
        } else {
            local_t.saturating_sub(offset_signed.unsigned_abs())
        }
    };
    let mut s_idx = 0u64;
    while s_idx < 3 {
        let t1 = ts_us.saturating_sub((3 - s_idx).saturating_mul(50_000));
        let t2 = peer_shift(t1.saturating_add(1_500));
        let t3 = t2.saturating_add(500);
        let t4 = t1.saturating_add(3_500);
        let _ = clock_sync.record_exchange(TwoWaySyncExchange {
            sender_id: 2,
            t1_local_tx_us: t1,
            t2_peer_rx_us: t2,
            t3_peer_tx_us: t3,
            t4_local_rx_us: t4,
        });
        let _ = clock_sync.record_exchange(TwoWaySyncExchange {
            sender_id: 3,
            t1_local_tx_us: t1,
            t2_peer_rx_us: t2,
            t3_peer_tx_us: t3,
            t4_local_rx_us: t4,
        });
        s_idx += 1;
    }

    let mut mesh_guard = StatefulMeshDaaGuard::new();
    let base_x = 6_378_137.0_f64;
    let peer_ts_us = peer_shift(ts_us);
    let obs = [
        OpenXvsBearingTrack::from_ecef_ray(
            ts_us,
            [base_x, 0.0, 0.0],
            [500.0, 200.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            inp.own_aircraft_id,
            0,
        ),
        OpenXvsBearingTrack::from_ecef_ray(
            peer_ts_us,
            [base_x, 1000.0, 0.0],
            [500.0, -800.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            2,
            0,
        ),
        OpenXvsBearingTrack::from_ecef_ray(
            peer_ts_us,
            [base_x + 1000.0, 500.0, 0.0],
            [-500.0, -300.0, 0.0],
            [0.0; 3],
            1500,
            100,
            1,
            3,
            0,
        ),
    ];
    let mesh_res = mesh_guard.fuse_guarded_with_clock_sync(
        &obs,
        ts_us.saturating_add(10_000),
        inp.own_aircraft_id,
        inp.gnss_denied != 0,
        &clock_sync,
    );

    let any_actuator = ACTUATOR_AUTHORITY || vlm_dec.actuator_authority || sup_dec.actuator_authority;

    Ring0McuOutput {
        rta_mode_u8: rta_out.mode as u8,
        odd_action_u8: odd_out.action as u8,
        vlm_gate_status_u8: vlm_dec.status as u8,
        hal_out_mode_u8: sup_dec.out_mode.to_rta_wire_u8(),
        hal_fw_gate_ok: u8::from(fw_gate_ok && hal_airframe_ok),
        active_range_wire_crc_ok: u8::from(ar_crc_ok),
        multiship_role_u8: coord_role as u8,
        mesh_num_fused: mesh_res.num_fused as u8,
        clock_sync_fallback_tier_b: u8::from(mesh_res.fallback_to_tier_b_optical),
        actuator_authority: u8::from(any_actuator),
        r_safe_lower_m: ar_pkt.r_safe_lower_m,
        coord_cmd_bank_rad: coord_pkt.cmd_bank_rad,
        stack_struct_bytes_total: openxvs_ring0_mcu_struct_bytes(),
    }
}

/// Per-crate compile-time Ring-0 struct stack sizes (bytes, zero heap).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ring0StructSizes {
    pub xvs_rta_core_simplex_rta_bytes: u32,
    pub xvs_odd_vlm_optical_odd_monitor_bytes: u32,
    pub xvs_odd_vlm_vlm_gate_bytes: u32,
    pub xvs_os_hal_plugin_supervisor_bytes: u32,
    pub xvs_active_range_controller_bytes: u32,
    pub xvs_multiship_coord_active_range_bytes: u32,
    pub xvs_multiship_coord_mesh_guard_bytes: u32,
    pub xvs_multiship_coord_clock_sync_bytes: u32,
}

#[inline(never)]
pub extern "C" fn openxvs_ring0_mcu_struct_breakdown() -> Ring0StructSizes {
    Ring0StructSizes {
        xvs_rta_core_simplex_rta_bytes: size_of::<SimplexRta>() as u32,
        xvs_odd_vlm_optical_odd_monitor_bytes: size_of::<OpticalOddMonitor<30, 90, 30, 60, 60>>()
            as u32,
        xvs_odd_vlm_vlm_gate_bytes: size_of::<AsymmetricSimplexVlmGate>() as u32,
        xvs_os_hal_plugin_supervisor_bytes: size_of::<PluginSupervisorState>() as u32,
        xvs_active_range_controller_bytes: size_of::<ActiveRangeController>() as u32,
        xvs_multiship_coord_active_range_bytes: size_of::<CoordinatedActiveRange>() as u32,
        xvs_multiship_coord_mesh_guard_bytes: size_of::<StatefulMeshDaaGuard>() as u32,
        xvs_multiship_coord_clock_sync_bytes: size_of::<PeerClockOffsetEstimator>() as u32,
    }
}
