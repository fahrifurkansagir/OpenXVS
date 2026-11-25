//! OpenXVS v2.0 C11 FFI (`cdylib` / `staticlib`) exposing `xvs-rta-core` and `xvs-math-core`
//! with strictly `#[repr(C)]` POD types for Python 3.14 `ctypes` (`src/xvs/core_ffi.py`) and
//! future M3 real-time C/C++ avionics targets.

#![allow(clippy::missing_safety_doc)]

use core::slice;
use xvs_math_core::codecs::{
    encode_gdl90_heartbeat, encode_gdl90_ownship, encode_gdl90_ownship_geo_alt,
    encode_gdl90_traffic, encode_mavlink_v2_adsb_246, encode_mavlink_v2_bearing_track,
    Gdl90TargetParams, MavlinkAdsb246Params,
};
use xvs_math_core::daa_ecbf::{EcbfQpSolver, IntruderStateNed, MAX_INTRUDERS};
use xvs_math_core::es_ekf::{DualBankEsEkf, SpoofCheckInput};
use xvs_math_core::geo::{ecef_to_geodetic, euler_zyx_to_quat, geodetic_to_ecef, r_ecef_ned};
use xvs_math_core::mesh_daa::{fci_fuse_3d, MeshDaaFuser, OpenXvsBearingTrack, MAX_FUSED_TRACKS};
use xvs_math_core::rig::{project_ecef_to_cam, TRAFFIC_RING_V1, XVS6_CUBE_V1};
use xvs_math_core::sim_engine::{FaultPerturbations, LiveDeckEngine, MissionId, TelemetrySnapshot};
use xvs_math_core::vraim::VisionRaimEvaluator;
use xvs_rta_core::{build_rta_input_from_f64, RtaFloatTelemetry, RtaTickInput, SimplexRta};

/// Opaque handle to a stateful `SimplexRta` instance.
pub struct XvsRtaHandle {
    inner: SimplexRta,
}

/// Opaque handle to a stateful `LiveDeckEngine` instance.
pub struct XvsSimHandle {
    inner: LiveDeckEngine,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsRtaFloatTelemetry {
    pub horizon_pl_rad: f64,
    pub horizon_al_rad: f64,
    pub v_vpl_m: f64,
    pub v_val_m: f64,
    pub v_hpl_m: f64,
    pub v_hal_m: f64,
    pub epipolar_max_rad: [f64; 6],
    pub signed_horizon_residual_rad: f64,
    pub runway_width_delta_m: f64,
    pub runway_width_delta_limit_m: f64,
    pub cusum_stat: f64,
    pub cusum_thr: f64,
    pub ood_global_score: f64,
    pub ood_global_thr: f64,
    pub ood_cam_scores: [f64; 6],
    pub ood_cam_thr: f64,
    pub cam_health_mask: u8,
    pub ml_detector_fault: u8,
    pub transient_dropout: u8,
    pub approach_active: u8,
    pub go_around_ack_reset: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsRtaTickInput {
    pub cam_health_mask: u8,
    pub ood_exceeded_mask: u8,
    pub horizon_pl_urad: i64,
    pub horizon_al_urad: i64,
    pub v_vpl_mm: i64,
    pub v_val_mm: i64,
    pub v_hpl_mm: i64,
    pub v_hal_mm: i64,
    pub epipolar_max_urad: [i64; 6],
    pub signed_horizon_residual_urad: i64,
    pub runway_width_delta_mm: i64,
    pub spoof_alarm: u8,
    pub ml_detector_fault: u8,
    pub ood_global_exceeded: u8,
    pub runway_width_fault: u8,
    pub transient_dropout: u8,
    pub approach_active: u8,
    pub go_around_ack_reset: u8,
    pub quantize_fault: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsRtaTickOutput {
    pub mode: u8,
    pub raw_target: u8,
    pub healthy_cam_mask: u8,
    pub active_healthy_cams: u8,
    pub consec_dropout_ticks: u32,
    pub window_dropout_20: u8,
    pub recovery_ticks: u32,
    pub dwell_ticks: u32,
    pub mandatory_go_around: u8,
    pub mltl_critical_violation: u8,
    pub blank_conformal_hud: u8,
    pub inhibit_avoidance_turns: u8,
    pub tick_index: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsFaultPerturbations {
    pub cam_r_bias_rad: f64,
    pub gnss_spoof_vel_ned_mps: [f64; 3],
    pub runway_corner_bias_px: f64,
    pub runway_observed_width_m: f64,
    pub fog_contrast: f64,
    pub ood_global_score: f64,
    pub manual_bank_cmd_rad: f64,
    pub manual_climb_cmd_mps: f64,
    pub lwir_efvs_enabled: u8,
    pub transient_dropout: u8,
    pub ml_detector_fault: u8,
    pub go_around_ack_reset: u8,
    pub peer3_active: u8,
    pub manual_override: u8,
    pub foreflight_sim_prefix: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsTelemetrySnapshot {
    pub time_s: f64,
    pub tick_count: u64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_m: f64,
    pub p_ecef: [f64; 3],
    pub v_ned: [f64; 3],
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub roll_rad: f64,
    pub q_ned_body: [f64; 4],
    pub tas_mps: f64,
    pub epipolar_max_urad: [f64; 6],
    pub fused_attitude_err_urad: f64,
    pub horizon_pl_urad: f64,
    pub v_vpl_mm: f64,
    pub v_hpl_mm: f64,
    pub cusum_stat: f64,
    pub nis_stat: f64,
    pub ss_stat: f64,
    pub bank_vel_delta_mps: f64,
    pub ecbf_bank_cmd_rad: f64,
    pub ecbf_climb_cmd_mps: f64,
    pub ecbf_slack: f64,
    pub dtaci_cone_urad: f64,
    pub intruder_rel_ned: [[f64; 3]; 8],
    pub intruder_vel_ned: [[f64; 3]; 8],
    pub intruder_count: u32,
    pub ghosts_rejected: u32,
    pub mesh_fused_count: u32,
    pub mission_id: u8,
    pub simplex_mode: u8,
    pub raw_target: u8,
    pub healthy_cam_mask: u8,
    pub active_healthy_cams: u8,
    pub active_nav_bank: u8,
    pub mandatory_go_around: u8,
    pub blank_conformal_hud: u8,
    pub inhibit_avoidance_turns: u8,
    pub ecbf_infeasible: u8,
    pub wc1_alert: u8,
    pub dtaci_frozen: u8,
    pub approach_active: u8,
    pub lwir_efvs_enabled: u8,
    pub _pad: [u8; 2],
}

impl From<&TelemetrySnapshot> for XvsTelemetrySnapshot {
    fn from(s: &TelemetrySnapshot) -> Self {
        Self {
            time_s: s.time_s,
            tick_count: s.tick_count,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            h_m: s.h_m,
            p_ecef: s.p_ecef,
            v_ned: s.v_ned,
            yaw_rad: s.yaw_rad,
            pitch_rad: s.pitch_rad,
            roll_rad: s.roll_rad,
            q_ned_body: s.q_ned_body,
            tas_mps: s.tas_mps,
            epipolar_max_urad: s.epipolar_max_urad,
            fused_attitude_err_urad: s.fused_attitude_err_urad,
            horizon_pl_urad: s.horizon_pl_urad,
            v_vpl_mm: s.v_vpl_mm,
            v_hpl_mm: s.v_hpl_mm,
            cusum_stat: s.cusum_stat,
            nis_stat: s.nis_stat,
            ss_stat: s.ss_stat,
            bank_vel_delta_mps: s.bank_vel_delta_mps,
            ecbf_bank_cmd_rad: s.ecbf_bank_cmd_rad,
            ecbf_climb_cmd_mps: s.ecbf_climb_cmd_mps,
            ecbf_slack: s.ecbf_slack,
            dtaci_cone_urad: s.dtaci_cone_urad,
            intruder_rel_ned: s.intruder_rel_ned,
            intruder_vel_ned: s.intruder_vel_ned,
            intruder_count: s.intruder_count,
            ghosts_rejected: s.ghosts_rejected,
            mesh_fused_count: s.mesh_fused_count,
            mission_id: s.mission_id,
            simplex_mode: s.simplex_mode,
            raw_target: s.raw_target,
            healthy_cam_mask: s.healthy_cam_mask,
            active_healthy_cams: s.active_healthy_cams,
            active_nav_bank: s.active_nav_bank,
            mandatory_go_around: s.mandatory_go_around,
            blank_conformal_hud: s.blank_conformal_hud,
            inhibit_avoidance_turns: s.inhibit_avoidance_turns,
            ecbf_infeasible: s.ecbf_infeasible,
            wc1_alert: s.wc1_alert,
            dtaci_frozen: s.dtaci_frozen,
            approach_active: s.approach_active,
            lwir_efvs_enabled: s.lwir_efvs_enabled,
            _pad: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsSpoofEvalResult {
    pub nis_stat: f64,
    pub cusum_stat: f64,
    pub ss_stat: f64,
    pub maneuver_guard_scale: f64,
    pub spoof_alarm: u8,
    pub active_bank: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsVraimEvalResult {
    pub epipolar_max_rad: [f64; 6],
    pub isolated_cam_idx: i32,
    pub healthy_cam_mask: u8,
    pub fused_attitude_err_rad: f64,
    pub horizon_pl_rad: f64,
    pub v_vpl_m: f64,
    pub v_hpl_m: f64,
    pub max_slope_v: f64,
    pub max_slope_h: f64,
    pub parity_stat: f64,
    pub width_delta_m: f64,
    pub width_fault: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsBearingTrack {
    pub time_utc_us: u64,
    pub own_ecef_cm: [i32; 3],
    pub bearing_ecef_q15: [i16; 3],
    pub bearing_rate_urad_s: [i16; 3],
    pub cone_half_angle_urad: u16,
    pub own_pos_std_cm: u16,
    pub track_id: u16,
    pub sender_id: u8,
    pub flags: u8,
    pub cov_diag_packed: [u16; 6],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsMeshDaaEvalResult {
    pub num_fused: u32,
    pub ghosts_rejected: u32,
    pub fused_pos_ecef: [[f64; 3]; 8],
    pub triple_ray_confirmed: [u8; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsIntruderNed {
    pub pos_ned: [f64; 3],
    pub vel_ned: [f64; 3],
    pub valid: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsEcbfQpResult {
    pub bank_cmd_rad: f64,
    pub climb_rate_cmd_mps: f64,
    pub slack: f64,
    pub iterations: u8,
    pub ecbf_infeasible: u8,
    pub wc1_alert: u8,
    pub min_ttc_s: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsGdl90Target {
    pub alert_status: u8,
    pub address_type: u8,
    pub participant_address: u32,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_ft: f64,
    pub altitude_valid: u8,
    pub misc_indicators: u8,
    pub nic: u8,
    pub nacp: u8,
    pub horiz_vel_kt: f64,
    pub horiz_vel_valid: u8,
    pub vert_vel_fpm: f64,
    pub vert_vel_valid: u8,
    pub track_heading_deg: f64,
    pub emitter_category: u8,
    pub callsign: [u8; 8],
    pub emergency_code: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XvsMavlinkAdsb246 {
    pub icao_address: u32,
    pub lat_deg_e7: i32,
    pub lon_deg_e7: i32,
    pub altitude_mm: i32,
    pub heading_cdeg: u16,
    pub hor_velocity_cms: u16,
    pub ver_velocity_cms: i16,
    pub flags: u16,
    pub squawk: u16,
    pub altitude_type: u8,
    pub callsign: [u8; 9],
    pub emitter_type: u8,
    pub tslc: u8,
    pub seq: u8,
    pub sysid: u8,
    pub compid: u8,
}

// ============================================================================
// Simplex RTA C-ABI
// ============================================================================

#[no_mangle]
pub extern "C" fn xvs_rta_create() -> *mut XvsRtaHandle {
    Box::into_raw(Box::new(XvsRtaHandle {
        inner: SimplexRta::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn xvs_rta_destroy(handle: *mut XvsRtaHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn xvs_rta_build_input_from_f64(
    telemetry: *const XvsRtaFloatTelemetry,
    out_input: *mut XvsRtaTickInput,
) -> i32 {
    if telemetry.is_null() || out_input.is_null() {
        return -1;
    }
    let t = &*telemetry;
    let inp = build_rta_input_from_f64(&RtaFloatTelemetry {
        horizon_pl_rad: t.horizon_pl_rad,
        horizon_al_rad: t.horizon_al_rad,
        v_vpl_m: t.v_vpl_m,
        v_val_m: t.v_val_m,
        v_hpl_m: t.v_hpl_m,
        v_hal_m: t.v_hal_m,
        epipolar_max_rad: t.epipolar_max_rad,
        signed_horizon_residual_rad: t.signed_horizon_residual_rad,
        runway_width_delta_m: t.runway_width_delta_m,
        runway_width_delta_limit_m: t.runway_width_delta_limit_m,
        cusum_stat: t.cusum_stat,
        cusum_thr: t.cusum_thr,
        ood_global_score: t.ood_global_score,
        ood_global_thr: t.ood_global_thr,
        ood_cam_scores: t.ood_cam_scores,
        ood_cam_thr: t.ood_cam_thr,
        cam_health_mask: t.cam_health_mask,
        ml_detector_fault: t.ml_detector_fault != 0,
        transient_dropout: t.transient_dropout != 0,
        approach_active: t.approach_active != 0,
        go_around_ack_reset: t.go_around_ack_reset != 0,
    });
    *out_input = XvsRtaTickInput {
        cam_health_mask: inp.cam_health_mask,
        ood_exceeded_mask: inp.ood_exceeded_mask,
        horizon_pl_urad: inp.horizon_pl_urad,
        horizon_al_urad: inp.horizon_al_urad,
        v_vpl_mm: inp.v_vpl_mm,
        v_val_mm: inp.v_val_mm,
        v_hpl_mm: inp.v_hpl_mm,
        v_hal_mm: inp.v_hal_mm,
        epipolar_max_urad: inp.epipolar_max_urad,
        signed_horizon_residual_urad: inp.signed_horizon_residual_urad,
        runway_width_delta_mm: inp.runway_width_delta_mm,
        spoof_alarm: u8::from(inp.spoof_alarm),
        ml_detector_fault: u8::from(inp.ml_detector_fault),
        ood_global_exceeded: u8::from(inp.ood_global_exceeded),
        runway_width_fault: u8::from(inp.runway_width_fault),
        transient_dropout: u8::from(inp.transient_dropout),
        approach_active: u8::from(inp.approach_active),
        go_around_ack_reset: u8::from(inp.go_around_ack_reset),
        quantize_fault: u8::from(inp.quantize_fault),
    };
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_rta_step(
    handle: *mut XvsRtaHandle,
    input: *const XvsRtaTickInput,
    out: *mut XvsRtaTickOutput,
) -> i32 {
    if handle.is_null() || input.is_null() || out.is_null() {
        return -1;
    }
    let inp = &*input;
    let res = (*handle).inner.step(&RtaTickInput {
        cam_health_mask: inp.cam_health_mask,
        ood_exceeded_mask: inp.ood_exceeded_mask,
        horizon_pl_urad: inp.horizon_pl_urad,
        horizon_al_urad: inp.horizon_al_urad,
        v_vpl_mm: inp.v_vpl_mm,
        v_val_mm: inp.v_val_mm,
        v_hpl_mm: inp.v_hpl_mm,
        v_hal_mm: inp.v_hal_mm,
        epipolar_max_urad: inp.epipolar_max_urad,
        signed_horizon_residual_urad: inp.signed_horizon_residual_urad,
        runway_width_delta_mm: inp.runway_width_delta_mm,
        spoof_alarm: inp.spoof_alarm != 0,
        ml_detector_fault: inp.ml_detector_fault != 0,
        ood_global_exceeded: inp.ood_global_exceeded != 0,
        runway_width_fault: inp.runway_width_fault != 0,
        transient_dropout: inp.transient_dropout != 0,
        approach_active: inp.approach_active != 0,
        go_around_ack_reset: inp.go_around_ack_reset != 0,
        quantize_fault: inp.quantize_fault != 0,
    });
    *out = XvsRtaTickOutput {
        mode: res.mode as u8,
        raw_target: res.raw_target as u8,
        healthy_cam_mask: res.healthy_cam_mask,
        active_healthy_cams: res.active_healthy_cams,
        consec_dropout_ticks: res.consec_dropout_ticks,
        window_dropout_20: res.window_dropout_20,
        recovery_ticks: res.recovery_ticks,
        dwell_ticks: res.dwell_ticks,
        mandatory_go_around: u8::from(res.mandatory_go_around),
        mltl_critical_violation: u8::from(res.mltl_critical_violation),
        blank_conformal_hud: u8::from(res.blank_conformal_hud),
        inhibit_avoidance_turns: u8::from(res.inhibit_avoidance_turns),
        tick_index: res.tick_index,
    };
    0
}

// ============================================================================
// LiveDeckEngine Simulation C-ABI
// ============================================================================

#[no_mangle]
pub extern "C" fn xvs_sim_create(seed: u64) -> *mut XvsSimHandle {
    Box::into_raw(Box::new(XvsSimHandle {
        inner: LiveDeckEngine::new(seed),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_destroy(handle: *mut XvsSimHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_reset_mission(
    handle: *mut XvsSimHandle,
    mission_id: u8,
    seed: u64,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    (*handle)
        .inner
        .reset_mission(MissionId::from_u8(mission_id), seed);
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_set_faults(
    handle: *mut XvsSimHandle,
    faults: *const XvsFaultPerturbations,
) -> i32 {
    if handle.is_null() || faults.is_null() {
        return -1;
    }
    let f = &*faults;
    (*handle).inner.set_faults(FaultPerturbations {
        cam_r_bias_rad: f.cam_r_bias_rad,
        gnss_spoof_vel_ned_mps: f.gnss_spoof_vel_ned_mps,
        runway_corner_bias_px: f.runway_corner_bias_px,
        runway_observed_width_m: f.runway_observed_width_m,
        fog_contrast: f.fog_contrast,
        ood_global_score: f.ood_global_score,
        manual_bank_cmd_rad: f.manual_bank_cmd_rad,
        manual_climb_cmd_mps: f.manual_climb_cmd_mps,
        lwir_efvs_enabled: f.lwir_efvs_enabled != 0,
        transient_dropout: f.transient_dropout != 0,
        ml_detector_fault: f.ml_detector_fault != 0,
        go_around_ack_reset: f.go_around_ack_reset != 0,
        peer3_active: f.peer3_active != 0,
        manual_override: f.manual_override != 0,
        foreflight_sim_prefix: f.foreflight_sim_prefix != 0,
    });
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_step(handle: *mut XvsSimHandle, rta_ticks: u32) -> i32 {
    if handle.is_null() {
        return -1;
    }
    for _ in 0..rta_ticks {
        (*handle).inner.step_rta_tick();
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_step_100hz(handle: *mut XvsSimHandle, substeps: u32) -> i32 {
    if handle.is_null() {
        return -1;
    }
    for _ in 0..substeps {
        (*handle).inner.step_100hz();
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_snapshot(
    handle: *const XvsSimHandle,
    out: *mut XvsTelemetrySnapshot,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return -1;
    }
    *out = XvsTelemetrySnapshot::from((*handle).inner.snapshot());
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_sim_drain_tx_packets(
    handle: *mut XvsSimHandle,
    out_buf: *mut u8,
    out_cap: usize,
    out_proto: *mut u8,
) -> i32 {
    if handle.is_null() || out_buf.is_null() || out_proto.is_null() {
        return -1;
    }
    let buf = slice::from_raw_parts_mut(out_buf, out_cap);
    match (*handle).inner.drain_tx_packet(buf) {
        Some((proto, n)) => {
            *out_proto = proto;
            n as i32
        }
        None => 0,
    }
}

// ============================================================================
// Geodesy, Rig & Math Submodule Parity Exports
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn xvs_geo_geodetic_to_ecef(
    lat_deg: f64,
    lon_deg: f64,
    h_m: f64,
    out_ecef: *mut f64,
) -> i32 {
    if out_ecef.is_null() {
        return -1;
    }
    let e = geodetic_to_ecef(lat_deg, lon_deg, h_m);
    let out = slice::from_raw_parts_mut(out_ecef, 3);
    out.copy_from_slice(&e);
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_geo_ecef_to_geodetic(
    x: f64,
    y: f64,
    z: f64,
    out_lat: *mut f64,
    out_lon: *mut f64,
    out_h: *mut f64,
) -> i32 {
    if out_lat.is_null() || out_lon.is_null() || out_h.is_null() {
        return -1;
    }
    let (la, lo, hm) = ecef_to_geodetic(x, y, z);
    *out_lat = la;
    *out_lon = lo;
    *out_h = hm;
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_geo_r_ecef_ned(
    lat_deg: f64,
    lon_deg: f64,
    out_mat9: *mut f64,
) -> i32 {
    if out_mat9.is_null() {
        return -1;
    }
    let r = r_ecef_ned(lat_deg, lon_deg);
    let out = slice::from_raw_parts_mut(out_mat9, 9);
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = r[i][j];
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_geo_euler_zyx_to_quat(
    yaw_rad: f64,
    pitch_rad: f64,
    roll_rad: f64,
    out_q4: *mut f64,
) -> i32 {
    if out_q4.is_null() {
        return -1;
    }
    let q = euler_zyx_to_quat(yaw_rad, pitch_rad, roll_rad);
    let out = slice::from_raw_parts_mut(out_q4, 4);
    out.copy_from_slice(&q);
    0
}

/// Project ECEF point to pixel `(u, v)`. `rig_kind == 0` -> `XVS6_CUBE_V1`, `1` -> `TRAFFIC_RING_V1`.
/// Returns `1` if visible (`z_cam > 0`), `0` if behind camera (`z_cam <= 0`), `-1` on null ptr.
#[no_mangle]
pub unsafe extern "C" fn xvs_rig_project_ecef_to_cam(
    rig_kind: u8,
    p_ecef3: *const f64,
    p_cam3: *const f64,
    r_ecef_cam9: *const f64,
    out_u: *mut f64,
    out_v: *mut f64,
) -> i32 {
    if p_ecef3.is_null()
        || p_cam3.is_null()
        || r_ecef_cam9.is_null()
        || out_u.is_null()
        || out_v.is_null()
    {
        return -1;
    }
    let pe = slice::from_raw_parts(p_ecef3, 3);
    let pc = slice::from_raw_parts(p_cam3, 3);
    let rm = slice::from_raw_parts(r_ecef_cam9, 9);
    let r_mat = [
        [rm[0], rm[1], rm[2]],
        [rm[3], rm[4], rm[5]],
        [rm[6], rm[7], rm[8]],
    ];
    let intr = if rig_kind == 1 {
        &TRAFFIC_RING_V1.intrinsics
    } else {
        &XVS6_CUBE_V1.intrinsics
    };
    match project_ecef_to_cam(intr, [pe[0], pe[1], pe[2]], [pc[0], pc[1], pc[2]], &r_mat) {
        Some((u, v)) => {
            *out_u = u;
            *out_v = v;
            1
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn xvs_es_ekf_step_cusum_eval(
    p_init3: *const f64,
    v_init3: *const f64,
    n_steps: usize,
    dt: f64,
    accel_ned_flat: *const f64,
    gnss_vel_flat: *const f64,
    bank_rad_arr: *const f64,
    omega_arr: *const f64,
    out_res: *mut XvsSpoofEvalResult,
) -> i32 {
    if p_init3.is_null()
        || v_init3.is_null()
        || accel_ned_flat.is_null()
        || gnss_vel_flat.is_null()
        || bank_rad_arr.is_null()
        || omega_arr.is_null()
        || out_res.is_null()
        || n_steps == 0
    {
        return -1;
    }
    let pi = slice::from_raw_parts(p_init3, 3);
    let vi = slice::from_raw_parts(v_init3, 3);
    let accs = slice::from_raw_parts(accel_ned_flat, n_steps * 3);
    let gvs = slice::from_raw_parts(gnss_vel_flat, n_steps * 3);
    let banks = slice::from_raw_parts(bank_rad_arr, n_steps);
    let omegas = slice::from_raw_parts(omega_arr, n_steps);

    let mut ekf = DualBankEsEkf::new([pi[0], pi[1], pi[2]], [vi[0], vi[1], vi[2]]);
    let mut last = None;
    for s in 0..n_steps {
        let a = [accs[3 * s], accs[3 * s + 1], accs[3 * s + 2]];
        let gv = [gvs[3 * s], gvs[3 * s + 1], gvs[3 * s + 2]];
        ekf.step_imu_horizon(dt, a, [0.0, 0.0, omegas[s]], 0.0, 0.0, banks[s]);
        last = Some(ekf.update_gnss_and_check_spoof(&SpoofCheckInput {
            gnss_pos_ned: ekf.bank0_pos_ned(),
            gnss_vel_ned: gv,
            bank_rad: banks[s],
            omega_norm_radps: omegas[s],
        }));
    }
    if let Some(r) = last {
        *out_res = XvsSpoofEvalResult {
            nis_stat: r.nis_stat,
            cusum_stat: r.cusum_stat,
            ss_stat: r.ss_stat,
            maneuver_guard_scale: r.maneuver_guard_scale,
            spoof_alarm: u8::from(r.spoof_alarm),
            active_bank: r.active_bank,
        };
        0
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn xvs_vraim_eval(
    cam_biases6: *const f64,
    cam_mask: u8,
    corner_bias_px: f64,
    obs_width_m: f64,
    slant_range_m: f64,
    out_res: *mut XvsVraimEvalResult,
) -> i32 {
    if cam_biases6.is_null() || out_res.is_null() {
        return -1;
    }
    let cb = slice::from_raw_parts(cam_biases6, 6);
    let biases = [cb[0], cb[1], cb[2], cb[3], cb[4], cb[5]];
    let eval = VisionRaimEvaluator::new();
    let iso = eval.evaluate_6cam_horizon_epipolar(&biases, cam_mask);
    let rwy = eval.evaluate_runway_approach(corner_bias_px, obs_width_m, slant_range_m);
    *out_res = XvsVraimEvalResult {
        epipolar_max_rad: iso.epipolar_max_rad,
        isolated_cam_idx: iso.isolated_cam_idx.map(|i| i as i32).unwrap_or(-1),
        healthy_cam_mask: iso.healthy_cam_mask,
        fused_attitude_err_rad: iso.fused_attitude_err_rad,
        horizon_pl_rad: iso.horizon_pl_rad,
        v_vpl_m: rwy.v_vpl_m,
        v_hpl_m: rwy.v_hpl_m,
        max_slope_v: rwy.max_slope_v,
        max_slope_h: rwy.max_slope_h,
        parity_stat: rwy.parity_stat,
        width_delta_m: rwy.width_delta_m,
        width_fault: u8::from(rwy.width_fault),
    };
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_mesh_daa_eval(
    tracks: *const XvsBearingTrack,
    n_tracks: usize,
    out_res: *mut XvsMeshDaaEvalResult,
) -> i32 {
    if tracks.is_null() || out_res.is_null() || n_tracks > 32 {
        return -1;
    }
    let raw = slice::from_raw_parts(tracks, n_tracks);
    const ZERO_TRACK: OpenXvsBearingTrack = OpenXvsBearingTrack {
        time_utc_us: 0,
        own_ecef_cm: [0; 3],
        bearing_ecef_q15: [0; 3],
        bearing_rate_urad_s: [0; 3],
        cone_half_angle_urad: 0,
        own_pos_std_cm: 0,
        track_id: 0,
        sender_id: 0,
        flags: 0,
        cov_diag_packed: [0; 6],
    };
    let mut conv = [ZERO_TRACK; 32];
    for (dst, t) in conv.iter_mut().zip(raw.iter()) {
        *dst = OpenXvsBearingTrack {
            time_utc_us: t.time_utc_us,
            own_ecef_cm: t.own_ecef_cm,
            bearing_ecef_q15: t.bearing_ecef_q15,
            bearing_rate_urad_s: t.bearing_rate_urad_s,
            cone_half_angle_urad: t.cone_half_angle_urad,
            own_pos_std_cm: t.own_pos_std_cm,
            track_id: t.track_id,
            sender_id: t.sender_id,
            flags: t.flags,
            cov_diag_packed: t.cov_diag_packed,
        };
    }
    let mut fuser = MeshDaaFuser::new();
    let res = fuser.fuse(&conv[..n_tracks]);
    let mut pos = [[0.0; 3]; MAX_FUSED_TRACKS];
    let mut conf = [0u8; MAX_FUSED_TRACKS];
    for i in 0..res.num_fused.min(MAX_FUSED_TRACKS) {
        pos[i] = res.tracks[i].pos_ecef;
        conf[i] = u8::from(res.tracks[i].triple_ray_confirmed);
    }
    *out_res = XvsMeshDaaEvalResult {
        num_fused: res.num_fused as u32,
        ghosts_rejected: res.ghosts_rejected,
        fused_pos_ecef: pos,
        triple_ray_confirmed: conf,
    };
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_fci_fuse_3d(
    xa3: *const f64,
    pa9: *const f64,
    xb3: *const f64,
    pb9: *const f64,
    out_x3: *mut f64,
    out_p9: *mut f64,
) -> i32 {
    if xa3.is_null()
        || pa9.is_null()
        || xb3.is_null()
        || pb9.is_null()
        || out_x3.is_null()
        || out_p9.is_null()
    {
        return -1;
    }
    let xa = slice::from_raw_parts(xa3, 3);
    let pa = slice::from_raw_parts(pa9, 9);
    let xb = slice::from_raw_parts(xb3, 3);
    let pb = slice::from_raw_parts(pb9, 9);
    let pa_m = [
        [pa[0], pa[1], pa[2]],
        [pa[3], pa[4], pa[5]],
        [pa[6], pa[7], pa[8]],
    ];
    let pb_m = [
        [pb[0], pb[1], pb[2]],
        [pb[3], pb[4], pb[5]],
        [pb[6], pb[7], pb[8]],
    ];
    let (xf, pf) = fci_fuse_3d([xa[0], xa[1], xa[2]], &pa_m, [xb[0], xb[1], xb[2]], &pb_m);
    let ox = slice::from_raw_parts_mut(out_x3, 3);
    ox.copy_from_slice(&xf);
    let op = slice::from_raw_parts_mut(out_p9, 9);
    for i in 0..3 {
        for j in 0..3 {
            op[3 * i + j] = pf[i][j];
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn xvs_ecbf_qp_solve(
    p_own3: *const f64,
    v_own3: *const f64,
    nom_bank_rad: f64,
    nom_climb_mps: f64,
    intruders: *const XvsIntruderNed,
    n_intruders: usize,
    out_res: *mut XvsEcbfQpResult,
) -> i32 {
    if p_own3.is_null() || v_own3.is_null() || out_res.is_null() {
        return -1;
    }
    let po = slice::from_raw_parts(p_own3, 3);
    let vo = slice::from_raw_parts(v_own3, 3);
    let mut list = [IntruderStateNed::default(); MAX_INTRUDERS];
    let cnt = n_intruders.min(MAX_INTRUDERS);
    if cnt > 0 && !intruders.is_null() {
        let raw = slice::from_raw_parts(intruders, cnt);
        for i in 0..cnt {
            list[i] = IntruderStateNed {
                pos_ned: raw[i].pos_ned,
                vel_ned: raw[i].vel_ned,
                valid: raw[i].valid != 0,
            };
        }
    }
    let solver = EcbfQpSolver::default();
    let sol = solver.solve(
        [po[0], po[1], po[2]],
        [vo[0], vo[1], vo[2]],
        nom_bank_rad,
        nom_climb_mps,
        &list[..cnt],
    );
    *out_res = XvsEcbfQpResult {
        bank_cmd_rad: sol.bank_cmd_rad,
        climb_rate_cmd_mps: sol.climb_rate_cmd_mps,
        slack: sol.slack,
        iterations: sol.iterations,
        ecbf_infeasible: u8::from(sol.ecbf_infeasible),
        wc1_alert: u8::from(sol.wc1_alert),
        min_ttc_s: sol.min_ttc_s,
    };
    0
}

// ============================================================================
// GDL-90 & MAVLink v2 Codec C-ABI
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_gdl90_heartbeat(
    utc_time_s: u32,
    gps_pos_valid: u8,
    maint_reqd: u8,
    utc_ok: u8,
    uplink_count: u8,
    basic_and_long_count: u16,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if out_buf.is_null() || out_cap < 24 {
        return -1;
    }
    let buf = slice::from_raw_parts_mut(out_buf, out_cap);
    encode_gdl90_heartbeat(
        utc_time_s,
        gps_pos_valid != 0,
        maint_reqd != 0,
        utc_ok != 0,
        uplink_count,
        basic_and_long_count,
        buf,
    ) as i32
}

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_gdl90_ownship_geo_alt(
    alt_ft: f64,
    vfom_m: f64,
    vertical_warning: u8,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if out_buf.is_null() || out_cap < 20 {
        return -1;
    }
    let buf = slice::from_raw_parts_mut(out_buf, out_cap);
    encode_gdl90_ownship_geo_alt(alt_ft, vfom_m, vertical_warning != 0, buf) as i32
}

fn target_from_c(t: &XvsGdl90Target) -> Gdl90TargetParams {
    Gdl90TargetParams {
        alert_status: t.alert_status,
        address_type: t.address_type,
        participant_address: t.participant_address,
        lat_deg: t.lat_deg,
        lon_deg: t.lon_deg,
        altitude_ft: if t.altitude_valid != 0 {
            Some(t.altitude_ft)
        } else {
            None
        },
        misc_indicators: t.misc_indicators,
        nic: t.nic,
        nacp: t.nacp,
        horiz_vel_kt: if t.horiz_vel_valid != 0 {
            Some(t.horiz_vel_kt)
        } else {
            None
        },
        vert_vel_fpm: if t.vert_vel_valid != 0 {
            Some(t.vert_vel_fpm)
        } else {
            None
        },
        track_heading_deg: t.track_heading_deg,
        emitter_category: t.emitter_category,
        callsign: t.callsign,
        emergency_code: t.emergency_code,
    }
}

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_gdl90_ownship(
    target: *const XvsGdl90Target,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if target.is_null() || out_buf.is_null() || out_cap < 64 {
        return -1;
    }
    let buf = slice::from_raw_parts_mut(out_buf, out_cap);
    encode_gdl90_ownship(&target_from_c(&*target), buf) as i32
}

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_gdl90_traffic(
    target: *const XvsGdl90Target,
    sim_prefix: u8,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if target.is_null() || out_buf.is_null() || out_cap < 64 {
        return -1;
    }
    let buf = slice::from_raw_parts_mut(out_buf, out_cap);
    encode_gdl90_traffic(&target_from_c(&*target), sim_prefix != 0, buf) as i32
}

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_mavlink_adsb_246(
    params: *const XvsMavlinkAdsb246,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if params.is_null() || out_buf.is_null() || out_cap < 50 {
        return -1;
    }
    let p = &*params;
    let frame = encode_mavlink_v2_adsb_246(&MavlinkAdsb246Params {
        icao_address: p.icao_address,
        lat_deg_e7: p.lat_deg_e7,
        lon_deg_e7: p.lon_deg_e7,
        altitude_mm: p.altitude_mm,
        heading_cdeg: p.heading_cdeg,
        hor_velocity_cms: p.hor_velocity_cms,
        ver_velocity_cms: p.ver_velocity_cms,
        flags: p.flags,
        squawk: p.squawk,
        altitude_type: p.altitude_type,
        callsign: p.callsign,
        emitter_type: p.emitter_type,
        tslc: p.tslc,
        seq: p.seq,
        sysid: p.sysid,
        compid: p.compid,
    });
    let out = slice::from_raw_parts_mut(out_buf, 50);
    out.copy_from_slice(&frame);
    50
}

#[no_mangle]
pub unsafe extern "C" fn xvs_encode_mavlink_bearing_track(
    track: *const XvsBearingTrack,
    seq: u8,
    sysid: u8,
    compid: u8,
    link_id: u8,
    timestamp_48: u64,
    secret_key32: *const u8,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if track.is_null() || secret_key32.is_null() || out_buf.is_null() || out_cap < 77 {
        return -1;
    }
    let t = &*track;
    let mut key = [0u8; 32];
    key.copy_from_slice(slice::from_raw_parts(secret_key32, 32));
    let frame = encode_mavlink_v2_bearing_track(
        &OpenXvsBearingTrack {
            time_utc_us: t.time_utc_us,
            own_ecef_cm: t.own_ecef_cm,
            bearing_ecef_q15: t.bearing_ecef_q15,
            bearing_rate_urad_s: t.bearing_rate_urad_s,
            cone_half_angle_urad: t.cone_half_angle_urad,
            own_pos_std_cm: t.own_pos_std_cm,
            track_id: t.track_id,
            sender_id: t.sender_id,
            flags: t.flags,
            cov_diag_packed: t.cov_diag_packed,
        },
        seq,
        sysid,
        compid,
        link_id,
        timestamp_48,
        &key,
    );
    let out = slice::from_raw_parts_mut(out_buf, 77);
    out.copy_from_slice(&frame);
    77
}
