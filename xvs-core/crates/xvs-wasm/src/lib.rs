//! OpenXVS v2.0 WebAssembly Live Deck simulation engine (`<= 180 KB` Brotli).
//!
//! Zero per-frame JavaScript heap allocations: exposes flat linear-memory pointers
//! `snapshot_f64_ptr()`, `snapshot_f32_ptr()`, and `tx_ring_ptr()`.

use wasm_bindgen::prelude::*;
use xvs_math_core::sim_engine::{FaultPerturbations, LiveDeckEngine, MissionId};

pub const SNAPSHOT_F64_LEN: usize = 96;
pub const TX_RING_BYTES: usize = 4096;

#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub struct WasmLiveDeckSim {
    engine: LiveDeckEngine,
    snap_f64: [f64; SNAPSHOT_F64_LEN],
    snap_f32: [f32; SNAPSHOT_F64_LEN],
    tx_ring: [u8; TX_RING_BYTES],
    tx_ring_len: usize,
}

#[wasm_bindgen]
impl WasmLiveDeckSim {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Self {
        let mut sim = Self {
            engine: LiveDeckEngine::new(seed as u64),
            snap_f64: [0.0; SNAPSHOT_F64_LEN],
            snap_f32: [0.0; SNAPSHOT_F64_LEN],
            tx_ring: [0; TX_RING_BYTES],
            tx_ring_len: 0,
        };
        sim.sync_snapshot();
        sim
    }

    pub fn reset_mission(&mut self, mission_id: u8, seed: u32) {
        self.engine
            .reset_mission(MissionId::from_u8(mission_id), seed as u64);
        self.tx_ring_len = 0;
        self.sync_snapshot();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_faults(
        &mut self,
        cam_r_bias_rad: f64,
        spoof_vn_mps: f64,
        spoof_ve_mps: f64,
        spoof_vd_mps: f64,
        runway_corner_bias_px: f64,
        runway_observed_width_m: f64,
        fog_contrast: f64,
        ood_global_score: f64,
        lwir_efvs_enabled: bool,
        transient_dropout: bool,
        ml_detector_fault: bool,
        go_around_ack_reset: bool,
        peer3_active: bool,
        manual_override: bool,
        manual_bank_cmd_rad: f64,
        manual_climb_cmd_mps: f64,
        foreflight_sim_prefix: bool,
    ) {
        self.engine.set_faults(FaultPerturbations {
            cam_r_bias_rad,
            gnss_spoof_vel_ned_mps: [spoof_vn_mps, spoof_ve_mps, spoof_vd_mps],
            runway_corner_bias_px,
            runway_observed_width_m,
            fog_contrast,
            ood_global_score,
            manual_bank_cmd_rad,
            manual_climb_cmd_mps,
            lwir_efvs_enabled,
            transient_dropout,
            ml_detector_fault,
            go_around_ack_reset,
            peer3_active,
            manual_override,
            foreflight_sim_prefix,
        });
    }

    /// Advance `dt_ticks` physics substeps (`100 Hz`, `0.01 s` each) and update zero-copy views.
    pub fn step_100hz(&mut self, dt_ticks: u32) {
        for _ in 0..dt_ticks {
            self.engine.step_100hz();
        }
        self.sync_snapshot();
        self.drain_tx_into_ring();
    }

    /// Advance exactly one 10 Hz RTA tick (`10` physics substeps = `0.10 s`).
    pub fn step_rta_tick(&mut self) {
        self.step_100hz(10);
    }

    pub fn snapshot_f64_ptr(&self) -> *const f64 {
        self.snap_f64.as_ptr()
    }

    pub fn snapshot_f64_len(&self) -> usize {
        SNAPSHOT_F64_LEN
    }

    pub fn snapshot_f32_ptr(&self) -> *const f32 {
        self.snap_f32.as_ptr()
    }

    pub fn snapshot_f32_len(&self) -> usize {
        SNAPSHOT_F64_LEN
    }

    pub fn tx_ring_ptr(&self) -> *const u8 {
        self.tx_ring.as_ptr()
    }

    pub fn tx_ring_len(&self) -> usize {
        self.tx_ring_len
    }

    pub fn clear_tx_ring(&mut self) {
        self.tx_ring_len = 0;
    }

    /// Drain queued GDL-90 (`proto=0x01`) and MAVLink v2 (`proto=0x02`) packets into the flat
    /// `tx_ring` buffer using `[proto: u8, len_lo: u8, len_hi: u8, payload...]` framing.
    pub fn drain_tx_into_ring(&mut self) -> usize {
        let mut pkt = [0u8; 128];
        while let Some((proto, n)) = self.engine.drain_tx_packet(&mut pkt) {
            if self.tx_ring_len + 3 + n > TX_RING_BYTES {
                break;
            }
            let p = self.tx_ring_len;
            self.tx_ring[p] = proto;
            self.tx_ring[p + 1] = (n & 0xFF) as u8;
            self.tx_ring[p + 2] = ((n >> 8) & 0xFF) as u8;
            self.tx_ring[p + 3..p + 3 + n].copy_from_slice(&pkt[..n]);
            self.tx_ring_len += 3 + n;
        }
        self.tx_ring_len
    }

    fn sync_snapshot(&mut self) {
        let s = self.engine.snapshot();
        let f = &mut self.snap_f64;
        f[0] = s.time_s;
        f[1] = s.tick_count as f64;
        f[2] = s.lat_deg;
        f[3] = s.lon_deg;
        f[4] = s.h_m;
        f[5] = s.p_ecef[0];
        f[6] = s.p_ecef[1];
        f[7] = s.p_ecef[2];
        f[8] = s.v_ned[0];
        f[9] = s.v_ned[1];
        f[10] = s.v_ned[2];
        f[11] = s.yaw_rad;
        f[12] = s.pitch_rad;
        f[13] = s.roll_rad;
        f[14] = s.q_ned_body[0];
        f[15] = s.q_ned_body[1];
        f[16] = s.q_ned_body[2];
        f[17] = s.q_ned_body[3];
        f[18] = s.tas_mps;
        f[19..25].copy_from_slice(&s.epipolar_max_urad);
        f[25] = s.fused_attitude_err_urad;
        f[26] = s.horizon_pl_urad;
        f[27] = s.v_vpl_mm;
        f[28] = s.v_hpl_mm;
        f[29] = s.cusum_stat;
        f[30] = s.nis_stat;
        f[31] = s.ss_stat;
        f[32] = s.bank_vel_delta_mps;
        f[33] = s.ecbf_bank_cmd_rad;
        f[34] = s.ecbf_climb_cmd_mps;
        f[35] = s.ecbf_slack;
        f[36] = s.dtaci_cone_urad;
        f[37] = s.mission_id as f64;
        f[38] = s.simplex_mode as f64;
        f[39] = s.raw_target as f64;
        f[40] = s.healthy_cam_mask as f64;
        f[41] = s.active_healthy_cams as f64;
        f[42] = s.active_nav_bank as f64;
        f[43] = s.mandatory_go_around as f64;
        f[44] = s.blank_conformal_hud as f64;
        f[45] = s.inhibit_avoidance_turns as f64;
        f[46] = s.ecbf_infeasible as f64;
        f[47] = s.wc1_alert as f64;
        f[48] = s.dtaci_frozen as f64;
        f[49] = s.approach_active as f64;
        f[50] = s.lwir_efvs_enabled as f64;
        f[51] = s.ghosts_rejected as f64;
        f[52] = s.mesh_fused_count as f64;
        f[53] = s.intruder_count as f64;
        for i in 0..8 {
            f[54 + 3 * i] = s.intruder_rel_ned[i][0];
            f[54 + 3 * i + 1] = s.intruder_rel_ned[i][1];
            f[54 + 3 * i + 2] = s.intruder_rel_ned[i][2];
        }
        for i in 0..6 {
            f[78 + 3 * i] = s.intruder_vel_ned[i][0];
            f[78 + 3 * i + 1] = s.intruder_vel_ned[i][1];
            f[78 + 3 * i + 2] = s.intruder_vel_ned[i][2];
        }
        for i in 0..SNAPSHOT_F64_LEN {
            self.snap_f32[i] = f[i] as f32;
        }
    }
}
