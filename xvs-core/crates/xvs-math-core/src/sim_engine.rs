//! Unified `#![no_std]` `LiveDeckEngine`: 100 Hz 6-DoF physics + 10 Hz Dual-Bank ES-EKF,
//! Vision-RAIM, MeshDAA, 3D ECBF-QP, Guarded DtACI -> `xvs_rta_core::build_rta_input_from_f64`
//! -> `SimplexRta::step` -> GDL-90 & MAVLink v2 TX ring + flat `#[repr(C)]` `TelemetrySnapshot`.
//!
//! Strict Anti-Scripting Rule: `FaultPerturbations` injects physical/sensor faults ONLY and
//! NEVER sets Simplex modes, alarms, or Go-Around flags directly.

use crate::codecs::{
    encode_gdl90_heartbeat, encode_gdl90_ownship, encode_gdl90_ownship_geo_alt,
    encode_gdl90_traffic, encode_mavlink_v2_adsb_246, encode_mavlink_v2_bearing_track,
    Gdl90TargetParams, MavlinkAdsb246Params,
};
use crate::daa_ecbf::{EcbfQpSolver, IntruderStateNed, MAX_INTRUDERS};
use crate::det_f64::{atan2, hypot, mat3_mul_vec, norm3};
use crate::dtaci::GuardedDtAci;
use crate::es_ekf::{DualBankEsEkf, SpoofCheckInput, CUSUM_THRESHOLD};
use crate::flight6dof::{FlightControlCmd, FlightModel6Dof};
use crate::geo::{ecef_to_geodetic, geodetic_to_ecef, r_ecef_ned, rad_to_deg};
use crate::mesh_daa::{MeshDaaFuser, OpenXvsBearingTrack, FLAG_GNSS_DENIED, FLAG_SIMULATION};
use crate::vraim::{VisionRaimEvaluator, KSFO_28L_WIDTH_M, RUNWAY_WIDTH_TOL_M};
use xvs_rta_core::{build_rta_input_from_f64, RtaFloatTelemetry, SimplexMode, SimplexRta};

pub const TX_RING_CAPACITY: usize = 32;
pub const TX_MAX_PACKET_LEN: usize = 128;

const DEFAULT_MAVLINK_SIGN_KEY: [u8; 32] = [
    0x4F, 0x50, 0x45, 0x4E, 0x58, 0x56, 0x53, 0x5F, 0x56, 0x32, 0x5F, 0x4D, 0x31, 0x5F, 0x53, 0x49,
    0x54, 0x4C, 0x5F, 0x54, 0x45, 0x53, 0x54, 0x5F, 0x4B, 0x45, 0x59, 0x5F, 0x32, 0x30, 0x32, 0x36,
];

/// Seeded mission presets (`SP1_LIVEDECK_DESIGN.md` §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MissionId {
    FreeFlight = 0,
    M1CamRDrift = 1,
    M2GpsSpoof = 2,
    M3MeshDaa = 3,
    M4KsfoApproach = 4,
}

impl MissionId {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::M1CamRDrift,
            2 => Self::M2GpsSpoof,
            3 => Self::M3MeshDaa,
            4 => Self::M4KsfoApproach,
            _ => Self::FreeFlight,
        }
    }
}

/// Strictly physical/sensor perturbation inputs ONLY (never sets Simplex modes/alarms directly).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FaultPerturbations {
    pub cam_r_bias_rad: f64,
    pub gnss_spoof_vel_ned_mps: [f64; 3],
    pub runway_corner_bias_px: f64,
    pub runway_observed_width_m: f64,
    pub fog_contrast: f64,
    pub ood_global_score: f64,
    pub manual_bank_cmd_rad: f64,
    pub manual_climb_cmd_mps: f64,
    pub lwir_efvs_enabled: bool,
    pub transient_dropout: bool,
    pub ml_detector_fault: bool,
    pub go_around_ack_reset: bool,
    pub peer3_active: bool,
    pub manual_override: bool,
    pub foreflight_sim_prefix: bool,
}

impl Default for FaultPerturbations {
    fn default() -> Self {
        Self {
            cam_r_bias_rad: 0.0,
            gnss_spoof_vel_ned_mps: [0.0; 3],
            runway_corner_bias_px: 0.0,
            runway_observed_width_m: KSFO_28L_WIDTH_M,
            fog_contrast: 1.0,
            ood_global_score: 0.15,
            manual_bank_cmd_rad: 0.0,
            manual_climb_cmd_mps: 0.0,
            lwir_efvs_enabled: false,
            transient_dropout: false,
            ml_detector_fault: false,
            go_around_ack_reset: false,
            peer3_active: false,
            manual_override: false,
            foreflight_sim_prefix: true,
        }
    }
}

/// Flat `#[repr(C)]` POD telemetry snapshot for zero-copy `xvs-ffi` & `xvs-wasm` views.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetrySnapshot {
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
    pub intruder_rel_ned: [[f64; 3]; MAX_INTRUDERS],
    pub intruder_vel_ned: [[f64; 3]; MAX_INTRUDERS],
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

impl Default for TelemetrySnapshot {
    fn default() -> Self {
        Self {
            time_s: 0.0,
            tick_count: 0,
            lat_deg: 37.6119,
            lon_deg: -122.3574,
            h_m: 600.0,
            p_ecef: [0.0; 3],
            v_ned: [0.0; 3],
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            roll_rad: 0.0,
            q_ned_body: [1.0, 0.0, 0.0, 0.0],
            tas_mps: 90.0,
            epipolar_max_urad: [0.0; 6],
            fused_attitude_err_urad: 0.0,
            horizon_pl_urad: 0.0,
            v_vpl_mm: 0.0,
            v_hpl_mm: 0.0,
            cusum_stat: 0.0,
            nis_stat: 0.0,
            ss_stat: 0.0,
            bank_vel_delta_mps: 0.0,
            ecbf_bank_cmd_rad: 0.0,
            ecbf_climb_cmd_mps: 0.0,
            ecbf_slack: 0.0,
            dtaci_cone_urad: 1800.0,
            intruder_rel_ned: [[0.0; 3]; MAX_INTRUDERS],
            intruder_vel_ned: [[0.0; 3]; MAX_INTRUDERS],
            intruder_count: 0,
            ghosts_rejected: 0,
            mesh_fused_count: 0,
            mission_id: 0,
            simplex_mode: SimplexMode::NominalMl as u8,
            raw_target: SimplexMode::NominalMl as u8,
            healthy_cam_mask: 0x3F,
            active_healthy_cams: 6,
            active_nav_bank: 0,
            mandatory_go_around: 0,
            blank_conformal_hud: 0,
            inhibit_avoidance_turns: 0,
            ecbf_infeasible: 0,
            wc1_alert: 0,
            dtaci_frozen: 0,
            approach_active: 0,
            lwir_efvs_enabled: 0,
            _pad: [0; 2],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TxSlot {
    proto: u8, // 0x01 = GDL-90, 0x02 = MAVLink v2
    len: usize,
    data: [u8; TX_MAX_PACKET_LEN],
}

impl Default for TxSlot {
    fn default() -> Self {
        Self {
            proto: 0,
            len: 0,
            data: [0; TX_MAX_PACKET_LEN],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TxPacketRing {
    slots: [TxSlot; TX_RING_CAPACITY],
    head: usize,
    tail: usize,
    count: usize,
}

impl Default for TxPacketRing {
    fn default() -> Self {
        Self::new()
    }
}

impl TxPacketRing {
    pub const fn new() -> Self {
        Self {
            slots: [TxSlot {
                proto: 0,
                len: 0,
                data: [0; TX_MAX_PACKET_LEN],
            }; TX_RING_CAPACITY],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, proto: u8, payload: &[u8]) {
        if payload.len() > TX_MAX_PACKET_LEN {
            return;
        }
        let slot = &mut self.slots[self.tail];
        slot.proto = proto;
        slot.len = payload.len();
        slot.data[..payload.len()].copy_from_slice(payload);
        self.tail = (self.tail + 1) % TX_RING_CAPACITY;
        if self.count == TX_RING_CAPACITY {
            self.head = (self.head + 1) % TX_RING_CAPACITY;
        } else {
            self.count += 1;
        }
    }

    pub fn pop(&mut self, out_buf: &mut [u8]) -> Option<(u8, usize)> {
        if self.count == 0 {
            return None;
        }
        let slot = &self.slots[self.head];
        let n = slot.len.min(out_buf.len());
        out_buf[..n].copy_from_slice(&slot.data[..n]);
        let proto = slot.proto;
        self.head = (self.head + 1) % TX_RING_CAPACITY;
        self.count -= 1;
        Some((proto, n))
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

pub struct LiveDeckEngine {
    mission: MissionId,
    flight: FlightModel6Dof,
    ekf: DualBankEsEkf,
    vraim: VisionRaimEvaluator,
    mesh: MeshDaaFuser,
    ecbf: EcbfQpSolver,
    dtaci: GuardedDtAci,
    rta: SimplexRta,
    faults: FaultPerturbations,
    intruders: [IntruderStateNed; MAX_INTRUDERS],
    n_intruders: usize,
    approach_active: bool,
    runway_slant_range_m: f64,
    substep_100hz: u32,
    rta_tick_count: u64,
    prev_tick_v_ned: [f64; 3],
    delayed_visfix_pos_ned: [f64; 3],
    mav_seq: u8,
    cmd: FlightControlCmd,
    snap: TelemetrySnapshot,
    tx_ring: TxPacketRing,
}

impl LiveDeckEngine {
    pub fn new(seed: u64) -> Self {
        let mut eng = Self {
            mission: MissionId::FreeFlight,
            flight: FlightModel6Dof::new(37.65, -122.32, 600.0, 0.0, 90.0, 0.0, seed),
            ekf: DualBankEsEkf::new([0.0; 3], [90.0, 0.0, 0.0]),
            vraim: VisionRaimEvaluator::new(),
            mesh: MeshDaaFuser::new(),
            ecbf: EcbfQpSolver::default(),
            dtaci: GuardedDtAci::new(),
            rta: SimplexRta::new(),
            faults: FaultPerturbations::default(),
            intruders: [IntruderStateNed::default(); MAX_INTRUDERS],
            n_intruders: 0,
            approach_active: false,
            runway_slant_range_m: 3000.0,
            substep_100hz: 0,
            rta_tick_count: 0,
            prev_tick_v_ned: [90.0, 0.0, 0.0],
            delayed_visfix_pos_ned: [0.0; 3],
            mav_seq: 0,
            cmd: FlightControlCmd::default(),
            snap: TelemetrySnapshot::default(),
            tx_ring: TxPacketRing::new(),
        };
        eng.reset_mission(MissionId::FreeFlight, seed);
        eng
    }

    pub fn reset_mission(&mut self, mission: MissionId, seed: u64) {
        self.mission = mission;
        self.faults = FaultPerturbations::default();
        self.intruders = [IntruderStateNed::default(); MAX_INTRUDERS];
        self.n_intruders = 0;
        self.approach_active = false;
        self.runway_slant_range_m = 3000.0;
        self.substep_100hz = 0;
        self.rta_tick_count = 0;
        self.mav_seq = 0;
        self.rta = SimplexRta::new();
        self.dtaci = GuardedDtAci::new();
        self.tx_ring = TxPacketRing::new();

        match mission {
            MissionId::M1CamRDrift | MissionId::FreeFlight => {
                self.flight = FlightModel6Dof::new(37.65, -122.32, 600.0, 0.0, 90.0, 0.0, seed);
                self.cmd = FlightControlCmd {
                    bank_cmd_rad: 0.0,
                    climb_rate_cmd_mps: 0.0,
                    speed_cmd_mps: 90.0,
                    go_around_active: false,
                };
            }
            MissionId::M2GpsSpoof => {
                self.flight = FlightModel6Dof::new(37.55, -122.52, 1200.0, 0.0, 90.0, 0.0, seed);
                self.cmd = FlightControlCmd {
                    bank_cmd_rad: 0.0,
                    climb_rate_cmd_mps: 0.0,
                    speed_cmd_mps: 90.0,
                    go_around_active: false,
                };
            }
            MissionId::M3MeshDaa => {
                self.flight = FlightModel6Dof::new(37.62, -122.36, 1000.0, 0.0, 90.0, 0.0, seed);
                self.cmd = FlightControlCmd {
                    bank_cmd_rad: 0.0,
                    climb_rate_cmd_mps: 0.0,
                    speed_cmd_mps: 90.0,
                    go_around_active: false,
                };
                self.intruders[0] = IntruderStateNed {
                    pos_ned: [4200.0, 120.0, 0.0],
                    vel_ned: [-85.0, 0.0, 0.0],
                    valid: true,
                };
                self.intruders[1] = IntruderStateNed {
                    pos_ned: [3600.0, 2200.0, 0.0],
                    vel_ned: [-30.0, -75.0, 0.0],
                    valid: true,
                };
                self.intruders[2] = IntruderStateNed {
                    pos_ned: [3900.0, -2400.0, 40.0],
                    vel_ned: [-35.0, 80.0, -1.0],
                    valid: true,
                };
                self.n_intruders = 3;
            }
            MissionId::M4KsfoApproach => {
                // 3 deg final approach to KSFO 28L (37.6119 N, -122.3574 W, h = -28.3 m WGS-84)
                let yaw_28l = 284.0_f64.to_radians();
                self.flight =
                    FlightModel6Dof::new(37.6050, -122.3250, 130.0, yaw_28l, 72.0, -3.77, seed);
                self.cmd = FlightControlCmd {
                    bank_cmd_rad: 0.0,
                    climb_rate_cmd_mps: -3.77,
                    speed_cmd_mps: 72.0,
                    go_around_active: false,
                };
                self.approach_active = true;
                self.runway_slant_range_m = 2800.0;
            }
        }

        self.ekf = DualBankEsEkf::new(self.flight.state.p_ned_rel, self.flight.state.v_ned);
        self.prev_tick_v_ned = self.flight.state.v_ned;
        self.delayed_visfix_pos_ned = self.flight.state.p_ned_rel;
        self.refresh_snapshot_geometry();
    }

    pub fn set_faults(&mut self, faults: FaultPerturbations) {
        self.faults = faults;
    }

    pub fn snapshot(&self) -> &TelemetrySnapshot {
        &self.snap
    }

    pub fn drain_tx_packet(&mut self, out_buf: &mut [u8]) -> Option<(u8, usize)> {
        self.tx_ring.pop(out_buf)
    }

    pub fn tx_queue_len(&self) -> usize {
        self.tx_ring.len()
    }

    /// Advance 1 physics step (`0.01 s`, 100 Hz); executes the 10 Hz integrity/DAA/RTA cycle
    /// every 10th substep.
    pub fn step_100hz(&mut self) {
        self.flight.step_rk4(&self.cmd);
        for intr in self.intruders.iter_mut().take(self.n_intruders) {
            if intr.valid {
                for k in 0..3 {
                    intr.pos_ned[k] += intr.vel_ned[k] * 0.01;
                }
            }
        }
        if self.approach_active && !self.cmd.go_around_active {
            self.runway_slant_range_m =
                (self.runway_slant_range_m - self.flight.state.tas_mps * 0.01).max(650.0);
        }
        self.substep_100hz += 1;
        if self.substep_100hz >= 10 {
            self.substep_100hz = 0;
            self.run_10hz_integrity_and_rta();
        } else {
            self.refresh_snapshot_geometry();
        }
    }

    /// Advance exactly one 10 Hz RTA tick (`0.10 s` = 10 physics substeps).
    pub fn step_rta_tick(&mut self) {
        for _ in 0..10 {
            self.step_100hz();
        }
    }

    fn run_10hz_integrity_and_rta(&mut self) {
        self.rta_tick_count += 1;

        // 1. Dual-Bank 15-state ES-EKF + Bonferroni 3-layer spoofing check
        // Exact strapdown delta-V average acceleration over the 0.10 s tick (no 100Hz aliasing)
        let avg_a_ned = [
            (self.flight.state.v_ned[0] - self.prev_tick_v_ned[0]) / 0.1,
            (self.flight.state.v_ned[1] - self.prev_tick_v_ned[1]) / 0.1,
            (self.flight.state.v_ned[2] - self.prev_tick_v_ned[2]) / 0.1,
        ];
        self.prev_tick_v_ned = self.flight.state.v_ned;
        self.ekf.step_imu_horizon(
            0.1,
            avg_a_ned,
            self.flight.state.omega_body,
            self.flight.state.yaw_rad,
            self.flight.state.pitch_rad,
            self.flight.state.roll_rad,
        );
        if self.rta_tick_count.is_multiple_of(5) {
            self.ekf.capture_stochastic_clone();
            self.delayed_visfix_pos_ned = self.flight.state.p_ned_rel;
        } else if self.rta_tick_count >= 7 && self.rta_tick_count % 5 == 2 {
            self.ekf.update_visfix_cloning(self.delayed_visfix_pos_ned, 0.5);
        }

        let gnss_v = [
            self.flight.state.v_ned[0] + self.faults.gnss_spoof_vel_ned_mps[0],
            self.flight.state.v_ned[1] + self.faults.gnss_spoof_vel_ned_mps[1],
            self.flight.state.v_ned[2] + self.faults.gnss_spoof_vel_ned_mps[2],
        ];
        let spoof_res = self.ekf.update_gnss_and_check_spoof(&SpoofCheckInput {
            gnss_pos_ned: self.flight.state.p_ned_rel,
            gnss_vel_ned: gnss_v,
            bank_rad: self.flight.state.roll_rad,
            omega_norm_radps: self.flight.omega_norm(),
        });

        // 2. Multi-Cam Vision-RAIM (6-cam epipolar/horizon + KSFO 28L runway approach)
        let cam_biases = [0.0, self.faults.cam_r_bias_rad, 0.0, 0.0, 0.0, 0.0];
        let iso = self.vraim.evaluate_6cam_horizon_epipolar(&cam_biases, 0x3F);

        let rwy = if self.approach_active
            || !self.faults.runway_corner_bias_px.is_finite()
            || !self.faults.runway_observed_width_m.is_finite()
        {
            // Fog slightly degrades keypoint precision unless LWIR EFVS is active, without
            // triggering false Go-Around prior to an actual runway threshold bias fault.
            // Preserve non-finite values strictly (D16-M1-B1).
            let eff_bias = if !self.faults.fog_contrast.is_finite()
                || !self.faults.runway_corner_bias_px.is_finite()
            {
                f64::NAN
            } else if self.faults.fog_contrast < 0.35 && !self.faults.lwir_efvs_enabled {
                self.faults.runway_corner_bias_px + 0.25
            } else {
                self.faults.runway_corner_bias_px
            };
            self.vraim.evaluate_runway_approach(
                eff_bias,
                self.faults.runway_observed_width_m,
                self.runway_slant_range_m,
            )
        } else {
            self.vraim
                .evaluate_runway_approach(0.0, KSFO_28L_WIDTH_M, 2000.0)
        };

        // 3. Guarded Altitude-Mondrian DtACI
        let cone_rad =
            self.dtaci
                .update(self.flight.state.h_m.max(0.0), 0.0012, self.faults.ood_global_score);

        // 4. XVS-MeshDAA 3-ship triple-ray ghost gate (Mission M3 or when intruders present)
        let mut ghosts_rejected = 0u32;
        let mut mesh_fused_count = 0u32;
        if self.n_intruders >= 2 {
            let r_en = r_ecef_ned(self.flight.state.lat_deg, self.flight.state.lon_deg);
            let p_own = self.flight.state.p_ecef;
            let to_ecef = |rel_ned: [f64; 3]| -> [f64; 3] {
                let d = mat3_mul_vec(&r_en, rel_ned);
                [p_own[0] + d[0], p_own[1] + d[1], p_own[2] + d[2]]
            };
            let s1 = to_ecef([0.0, -1000.0, 0.0]);
            let s2 = to_ecef([0.0, 1000.0, 0.0]);
            let s3 = to_ecef([-800.0, 0.0, -600.0]);
            let t1 = to_ecef([2000.0, -400.0, 0.0]);
            let t2 = to_ecef([2000.0, 400.0, 0.0]);
            let cone_urad = (cone_rad * 1e6).clamp(100.0, 65000.0) as u16;
            let flags = FLAG_SIMULATION
                | if spoof_res.spoof_alarm {
                    FLAG_GNSS_DENIED
                } else {
                    0
                };
            let mk = |o: [f64; 3], tgt: [f64; 3], sender: u8, tid: u16| {
                let d = [tgt[0] - o[0], tgt[1] - o[1], tgt[2] - o[2]];
                let n = norm3(d).max(1.0);
                OpenXvsBearingTrack::from_ecef_ray(
                    (self.flight.state.time_s * 1e6) as u64,
                    o,
                    [d[0] / n, d[1] / n, d[2] / n],
                    [0.0; 3],
                    cone_urad,
                    50,
                    tid,
                    sender,
                    flags,
                )
            };
            let mut obs = [mk(s1, t1, 1, 1); 6];
            obs[0] = mk(s1, t1, 1, 1);
            obs[1] = mk(s1, t2, 1, 2);
            obs[2] = mk(s2, t1, 2, 3);
            obs[3] = mk(s2, t2, 2, 4);
            let n_obs = if self.faults.peer3_active {
                obs[4] = mk(s3, t1, 3, 5);
                obs[5] = mk(s3, t2, 3, 6);
                6
            } else {
                4
            };
            let mres = self.mesh.fuse(&obs[..n_obs]);
            ghosts_rejected = mres.ghosts_rejected;
            mesh_fused_count = mres.num_fused as u32;

            // Emit signed #52001 OPENXVS_BEARING_TRACK MAVLink v2 frame into TX ring
            let signed = encode_mavlink_v2_bearing_track(
                &obs[0],
                self.next_mav_seq(),
                1,
                156,
                1,
                self.rta_tick_count * 10_000,
                &DEFAULT_MAVLINK_SIGN_KEY,
            );
            self.tx_ring.push(0x02, &signed);
        }

        // 5. 3D Fixed-Wing ECBF-QP avoidance solver
        let nom_bank = if self.faults.manual_override {
            self.faults.manual_bank_cmd_rad
        } else {
            0.0
        };
        let nom_climb = if self.faults.manual_override {
            self.faults.manual_climb_cmd_mps
        } else if self.approach_active {
            -3.77
        } else {
            0.0
        };
        let ecbf_sol = self.ecbf.solve(
            self.flight.state.p_ned_rel,
            self.flight.state.v_ned,
            nom_bank,
            nom_climb,
            &self.intruders[..self.n_intruders],
        );

        // 6. Strictly route ALL float scalars & float-derived alarms through xvs_rta_core::build_rta_input_from_f64
        // Fail-closed non-finite preservation (D16-M1-B1): NEVER replace a NaN/Inf spoof stat with
        // a finite constant (which would downgrade P0 quantize_fault to P1 FallbackD0Safe).
        let any_spoof_stat_nonfinite = !spoof_res.nis_stat.is_finite()
            || !spoof_res.cusum_stat.is_finite()
            || !spoof_res.ss_stat.is_finite();
        let rta_cusum_stat = if any_spoof_stat_nonfinite {
            f64::NAN
        } else if spoof_res.spoof_alarm {
            CUSUM_THRESHOLD + 1.0
        } else {
            spoof_res.cusum_stat
        };
        let rta_in = build_rta_input_from_f64(&RtaFloatTelemetry {
            horizon_pl_rad: iso.horizon_pl_rad,
            horizon_al_rad: 0.008_727,
            v_vpl_m: rwy.v_vpl_m,
            v_val_m: 10.0,
            v_hpl_m: rwy.v_hpl_m,
            v_hal_m: 40.0,
            epipolar_max_rad: iso.epipolar_max_rad,
            signed_horizon_residual_rad: iso.fused_attitude_err_rad,
            runway_width_delta_m: rwy.width_delta_m,
            runway_width_delta_limit_m: RUNWAY_WIDTH_TOL_M,
            cusum_stat: rta_cusum_stat,
            cusum_thr: CUSUM_THRESHOLD,
            ood_global_score: self.faults.ood_global_score,
            ood_global_thr: 1.0,
            ood_cam_scores: [0.15; 6],
            ood_cam_thr: 1.0,
            cam_health_mask: iso.healthy_cam_mask,
            ml_detector_fault: self.faults.ml_detector_fault || ecbf_sol.ecbf_infeasible,
            transient_dropout: self.faults.transient_dropout,
            approach_active: self.approach_active,
            go_around_ack_reset: self.faults.go_around_ack_reset,
        });
        let rta_out = self.rta.step(&rta_in);

        // Update autopilot command respecting Simplex RTA inhibit / Go-Around latch
        self.cmd.go_around_active = rta_out.mandatory_go_around;
        if rta_out.mandatory_go_around {
            self.cmd.bank_cmd_rad = 0.0;
            self.cmd.climb_rate_cmd_mps = 7.5;
            self.cmd.speed_cmd_mps = 95.0;
        } else if rta_out.inhibit_avoidance_turns {
            self.cmd.bank_cmd_rad = nom_bank;
            self.cmd.climb_rate_cmd_mps = nom_climb;
        } else {
            self.cmd.bank_cmd_rad = ecbf_sol.bank_cmd_rad;
            self.cmd.climb_rate_cmd_mps = ecbf_sol.climb_rate_cmd_mps;
        }

        // 7. Emit GDL-90 (#0, #10, #11, #20) and MAVLink v2 (#246) into TX ring
        self.emit_avionics_packets(ecbf_sol.wc1_alert);

        // 8. Pack flat TelemetrySnapshot
        self.refresh_snapshot_geometry();
        for i in 0..6 {
            self.snap.epipolar_max_urad[i] = iso.epipolar_max_rad[i] * 1e6;
        }
        self.snap.fused_attitude_err_urad = iso.fused_attitude_err_rad * 1e6;
        self.snap.horizon_pl_urad = iso.horizon_pl_rad * 1e6;
        self.snap.v_vpl_mm = rwy.v_vpl_m * 1e3;
        self.snap.v_hpl_mm = rwy.v_hpl_m * 1e3;
        self.snap.cusum_stat = spoof_res.cusum_stat;
        self.snap.nis_stat = spoof_res.nis_stat;
        self.snap.ss_stat = spoof_res.ss_stat;
        let b0v = self.ekf.bank0_vel_ned();
        let b1v = self.ekf.bank1_vel_ned();
        self.snap.bank_vel_delta_mps = norm3([b0v[0] - b1v[0], b0v[1] - b1v[1], b0v[2] - b1v[2]]);
        self.snap.ecbf_bank_cmd_rad = ecbf_sol.bank_cmd_rad;
        self.snap.ecbf_climb_cmd_mps = ecbf_sol.climb_rate_cmd_mps;
        self.snap.ecbf_slack = ecbf_sol.slack;
        self.snap.dtaci_cone_urad = cone_rad * 1e6;
        self.snap.ghosts_rejected = ghosts_rejected;
        self.snap.mesh_fused_count = mesh_fused_count;
        self.snap.mission_id = self.mission as u8;
        self.snap.simplex_mode = rta_out.mode as u8;
        self.snap.raw_target = rta_out.raw_target as u8;
        self.snap.healthy_cam_mask = rta_out.healthy_cam_mask;
        self.snap.active_healthy_cams = rta_out.active_healthy_cams;
        self.snap.active_nav_bank = spoof_res.active_bank;
        self.snap.mandatory_go_around = u8::from(rta_out.mandatory_go_around);
        self.snap.blank_conformal_hud = u8::from(rta_out.blank_conformal_hud);
        self.snap.inhibit_avoidance_turns = u8::from(rta_out.inhibit_avoidance_turns);
        self.snap.ecbf_infeasible = u8::from(ecbf_sol.ecbf_infeasible);
        self.snap.wc1_alert = u8::from(ecbf_sol.wc1_alert);
        self.snap.dtaci_frozen = u8::from(self.dtaci.is_frozen());
        self.snap.approach_active = u8::from(self.approach_active);
        self.snap.lwir_efvs_enabled = u8::from(self.faults.lwir_efvs_enabled);
    }

    fn refresh_snapshot_geometry(&mut self) {
        self.snap.time_s = self.flight.state.time_s;
        self.snap.tick_count = self.rta_tick_count;
        self.snap.lat_deg = self.flight.state.lat_deg;
        self.snap.lon_deg = self.flight.state.lon_deg;
        self.snap.h_m = self.flight.state.h_m;
        self.snap.p_ecef = self.flight.state.p_ecef;
        self.snap.v_ned = self.flight.state.v_ned;
        self.snap.yaw_rad = self.flight.state.yaw_rad;
        self.snap.pitch_rad = self.flight.state.pitch_rad;
        self.snap.roll_rad = self.flight.state.roll_rad;
        self.snap.q_ned_body = self.flight.state.q_ned_body;
        self.snap.tas_mps = self.flight.state.tas_mps;
        self.snap.intruder_count = self.n_intruders as u32;
        for i in 0..MAX_INTRUDERS {
            if i < self.n_intruders && self.intruders[i].valid {
                self.snap.intruder_rel_ned[i] = [
                    self.intruders[i].pos_ned[0] - self.flight.state.p_ned_rel[0],
                    self.intruders[i].pos_ned[1] - self.flight.state.p_ned_rel[1],
                    self.intruders[i].pos_ned[2] - self.flight.state.p_ned_rel[2],
                ];
                self.snap.intruder_vel_ned[i] = self.intruders[i].vel_ned;
            } else {
                self.snap.intruder_rel_ned[i] = [0.0; 3];
                self.snap.intruder_vel_ned[i] = [0.0; 3];
            }
        }
    }

    fn next_mav_seq(&mut self) -> u8 {
        let s = self.mav_seq;
        self.mav_seq = self.mav_seq.wrapping_add(1);
        s
    }

    fn emit_avionics_packets(&mut self, wc1_alert: bool) {
        let mut buf = [0u8; 96];
        let alt_ft = self.flight.state.h_m * 3.280_839_895;
        let h_kt = hypot(self.flight.state.v_ned[0], self.flight.state.v_ned[1]) * 1.943_844_49;
        let v_fpm = -self.flight.state.v_ned[2] * 196.850_394;
        let trk_deg = rad_to_deg(atan2(self.flight.state.v_ned[1], self.flight.state.v_ned[0]));

        // GDL-90 Heartbeat (#0)
        let n_hb = encode_gdl90_heartbeat(
            self.flight.state.time_s as u32,
            self.ekf.active_bank() == 0,
            false,
            true,
            0,
            self.n_intruders as u16,
            &mut buf,
        );
        self.tx_ring.push(0x01, &buf[..n_hb]);

        // GDL-90 Ownship (#10) & Geometric Altitude (#11)
        let own_p = Gdl90TargetParams {
            alert_status: 0,
            address_type: 0,
            participant_address: 0xA1B2C3,
            lat_deg: self.flight.state.lat_deg,
            lon_deg: self.flight.state.lon_deg,
            altitude_ft: Some(alt_ft),
            misc_indicators: 0x09,
            nic: 8,
            nacp: 8,
            horiz_vel_kt: Some(h_kt),
            vert_vel_fpm: Some(v_fpm),
            track_heading_deg: trk_deg,
            emitter_category: 1,
            callsign: *b"XVSOWN1 ",
            emergency_code: 0,
        };
        let n_own = encode_gdl90_ownship(&own_p, &mut buf);
        self.tx_ring.push(0x01, &buf[..n_own]);
        let n_geo = encode_gdl90_ownship_geo_alt(alt_ft, 3.0, false, &mut buf);
        self.tx_ring.push(0x01, &buf[..n_geo]);

        // Emit traffic reports (GDL-90 #20 + MAVLink #246) for first intruder or ownship beacon
        let r_en = r_ecef_ned(self.flight.state.lat_deg, self.flight.state.lon_deg);
        let base_ecef = geodetic_to_ecef(
            self.flight.state.origin_lat_deg,
            self.flight.state.origin_lon_deg,
            self.flight.state.origin_h_m,
        );
        let (t_lat, t_lon, t_alt_m, t_vn, t_ve, t_vd) = if self.n_intruders > 0 {
            let intr = self.intruders[0];
            let d_ecef = mat3_mul_vec(&r_en, intr.pos_ned);
            let (la, lo, hm) = ecef_to_geodetic(
                base_ecef[0] + d_ecef[0],
                base_ecef[1] + d_ecef[1],
                base_ecef[2] + d_ecef[2],
            );
            (la, lo, hm, intr.vel_ned[0], intr.vel_ned[1], intr.vel_ned[2])
        } else {
            (
                self.flight.state.lat_deg,
                self.flight.state.lon_deg,
                self.flight.state.h_m,
                self.flight.state.v_ned[0],
                self.flight.state.v_ned[1],
                self.flight.state.v_ned[2],
            )
        };

        let trf_p = Gdl90TargetParams {
            alert_status: u8::from(wc1_alert),
            address_type: 0,
            participant_address: 0xF00001,
            lat_deg: t_lat,
            lon_deg: t_lon,
            altitude_ft: Some(t_alt_m * 3.280_839_895),
            misc_indicators: 0x09,
            nic: 6,
            nacp: 6,
            horiz_vel_kt: Some(hypot(t_vn, t_ve) * 1.943_844_49),
            vert_vel_fpm: Some(-t_vd * 196.850_394),
            track_heading_deg: rad_to_deg(atan2(t_ve, t_vn)),
            emitter_category: 1,
            callsign: *b"XVS0001 ",
            emergency_code: 0,
        };
        let n_trf = encode_gdl90_traffic(&trf_p, self.faults.foreflight_sim_prefix, &mut buf);
        self.tx_ring.push(0x01, &buf[..n_trf]);

        let hdg_cdeg = {
            let mut d = rad_to_deg(atan2(t_ve, t_vn)) % 360.0;
            if d < 0.0 {
                d += 360.0;
            }
            (d * 100.0) as u16
        };
        let adsb = MavlinkAdsb246Params {
            icao_address: 0xF00001,
            lat_deg_e7: (t_lat * 1e7) as i32,
            lon_deg_e7: (t_lon * 1e7) as i32,
            altitude_mm: (t_alt_m * 1000.0) as i32,
            heading_cdeg: hdg_cdeg,
            hor_velocity_cms: (hypot(t_vn, t_ve) * 100.0).clamp(0.0, 65535.0) as u16,
            ver_velocity_cms: (-t_vd * 100.0).clamp(-32768.0, 32767.0) as i16,
            flags: 0x00FF,
            squawk: if wc1_alert { 7700 } else { 1200 },
            altitude_type: 1,
            callsign: *b"XVS0001\0\0",
            emitter_type: 14,
            tslc: 0,
            seq: self.next_mav_seq(),
            sysid: 1,
            compid: 156,
        };
        let mav_frame = encode_mavlink_v2_adsb_246(&adsb);
        self.tx_ring.push(0x02, &mav_frame);
    }
}
