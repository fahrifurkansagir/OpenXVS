//! Fixed-step 100 Hz (`dt = 0.01 s`) RK4 civil twin-jet 6-DoF flight dynamics model.
//!
//! Pure `#![no_std]` `DetF64` SI units (`m`, `s`, `rad`), NED velocity, FRD body frame,
//! `±30°` bank limit, climb/descent limits, `SplitMix64` seeded turbulence, and Go-Around climb-out.

use crate::det_f64::{clamp, cos, fabs, hypot, mat3_mul_vec, sin, tan, SplitMix64, Vec3};
use crate::geo::{
    ecef_to_geodetic, euler_to_r, euler_zyx_to_quat, geodetic_to_ecef, r_ecef_ned,
};

pub const GRAVITY_MPS2: f64 = 9.80665;
pub const MAX_BANK_RAD: f64 = 30.0 * core::f64::consts::PI / 180.0;
pub const MIN_CLIMB_MPS: f64 = -8.0;
pub const MAX_CLIMB_MPS: f64 = 10.0;
pub const MIN_TAS_MPS: f64 = 55.0;
pub const MAX_TAS_MPS: f64 = 140.0;
pub const PHYSICS_DT_S: f64 = 0.01;

#[derive(Debug, Clone, Copy)]
pub struct FlightControlCmd {
    pub bank_cmd_rad: f64,
    pub climb_rate_cmd_mps: f64,
    pub speed_cmd_mps: f64,
    pub go_around_active: bool,
}

impl Default for FlightControlCmd {
    fn default() -> Self {
        Self {
            bank_cmd_rad: 0.0,
            climb_rate_cmd_mps: 0.0,
            speed_cmd_mps: 90.0,
            go_around_active: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlightState {
    pub time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_m: f64,
    pub p_ecef: Vec3,
    pub p_ned_rel: Vec3,
    pub v_ned: Vec3,
    pub a_ned: Vec3,
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub roll_rad: f64,
    pub q_ned_body: [f64; 4],
    pub omega_body: Vec3,
    pub tas_mps: f64,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub origin_h_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct FlightModel6Dof {
    pub state: FlightState,
    rng: SplitMix64,
    pub turbulence_sigma_mps2: f64,
}

impl FlightModel6Dof {
    pub fn new(
        lat_deg: f64,
        lon_deg: f64,
        h_m: f64,
        yaw_rad: f64,
        tas_mps: f64,
        climb_rate_mps: f64,
        seed: u64,
    ) -> Self {
        let tas = clamp(tas_mps, MIN_TAS_MPS, MAX_TAS_MPS);
        let climb = clamp(climb_rate_mps, MIN_CLIMB_MPS, MAX_CLIMB_MPS);
        let v_h = hypot(tas, 0.0).max(1.0);
        let pitch_rad = clamp(climb / v_h, -0.25, 0.25);
        let v_ned = [tas * cos(yaw_rad), tas * sin(yaw_rad), -climb];
        let p_ecef = geodetic_to_ecef(lat_deg, lon_deg, h_m);
        let q_ned_body = euler_zyx_to_quat(yaw_rad, pitch_rad, 0.0);
        Self {
            state: FlightState {
                time_s: 0.0,
                lat_deg,
                lon_deg,
                h_m,
                p_ecef,
                p_ned_rel: [0.0, 0.0, 0.0],
                v_ned,
                a_ned: [0.0, 0.0, 0.0],
                yaw_rad,
                pitch_rad,
                roll_rad: 0.0,
                q_ned_body,
                omega_body: [0.0, 0.0, 0.0],
                tas_mps: tas,
                origin_lat_deg: lat_deg,
                origin_lon_deg: lon_deg,
                origin_h_m: h_m,
            },
            rng: SplitMix64::new(seed),
            turbulence_sigma_mps2: 0.08,
        }
    }

    /// Integrate one fixed `dt = 0.01 s` RK4 physics step.
    pub fn step_rk4(&mut self, cmd: &FlightControlCmd) {
        let dt = PHYSICS_DT_S;
        let (bank_tgt, climb_tgt, spd_tgt) = if cmd.go_around_active {
            // Mandatory Go-Around climb-out: wings level, +7.5 m/s climb, 95 m/s airspeed
            (0.0, 7.5, 95.0)
        } else {
            (
                clamp(cmd.bank_cmd_rad, -MAX_BANK_RAD, MAX_BANK_RAD),
                clamp(cmd.climb_rate_cmd_mps, MIN_CLIMB_MPS, MAX_CLIMB_MPS),
                clamp(cmd.speed_cmd_mps, MIN_TAS_MPS, MAX_TAS_MPS),
            )
        };

        let (g0, g1) = self.rng.next_normal_pair();
        let turb_tas = g0 * self.turbulence_sigma_mps2;
        let turb_pitch = g1 * self.turbulence_sigma_mps2 * 0.002;

        // Compact 7-element kinematic/dynamic state vector:
        // [pn, pe, pd, yaw, pitch, roll, tas]
        let y0 = [
            self.state.p_ned_rel[0],
            self.state.p_ned_rel[1],
            self.state.p_ned_rel[2],
            self.state.yaw_rad,
            self.state.pitch_rad,
            self.state.roll_rad,
            self.state.tas_mps,
        ];

        let deriv = |y: [f64; 7]| -> [f64; 7] {
            let yaw = y[3];
            let pitch = y[4];
            let roll = clamp(y[5], -MAX_BANK_RAD, MAX_BANK_RAD);
            let tas = clamp(y[6], MIN_TAS_MPS, MAX_TAS_MPS);

            let pitch_tgt = clamp(climb_tgt / tas, -0.22, 0.25);
            let roll_dot = (bank_tgt - roll) / 0.45;
            let pitch_dot = (pitch_tgt - pitch) / 0.55 + turb_pitch;
            let yaw_dot = (GRAVITY_MPS2 / tas) * tan(roll);
            let tas_dot = (spd_tgt - tas) / 2.5 + turb_tas;

            let v_n = tas * cos(pitch) * cos(yaw);
            let v_e = tas * cos(pitch) * sin(yaw);
            let v_d = -tas * sin(pitch);
            [v_n, v_e, v_d, yaw_dot, pitch_dot, roll_dot, tas_dot]
        };

        let k1 = deriv(y0);
        let mut yt = [0.0; 7];
        for i in 0..7 {
            yt[i] = y0[i] + 0.5 * dt * k1[i];
        }
        let k2 = deriv(yt);
        for i in 0..7 {
            yt[i] = y0[i] + 0.5 * dt * k2[i];
        }
        let k3 = deriv(yt);
        for i in 0..7 {
            yt[i] = y0[i] + dt * k3[i];
        }
        let k4 = deriv(yt);

        let mut y1 = [0.0; 7];
        for i in 0..7 {
            y1[i] = y0[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        // Wrap yaw cleanly to [-pi, pi]
        let mut yaw = y1[3];
        while yaw > core::f64::consts::PI {
            yaw -= 2.0 * core::f64::consts::PI;
        }
        while yaw < -core::f64::consts::PI {
            yaw += 2.0 * core::f64::consts::PI;
        }
        let pitch = clamp(y1[4], -0.35, 0.35);
        let roll = clamp(y1[5], -MAX_BANK_RAD, MAX_BANK_RAD);
        let tas = clamp(y1[6], MIN_TAS_MPS, MAX_TAS_MPS);

        let v_prev = self.state.v_ned;
        let v_new = [
            tas * cos(pitch) * cos(yaw),
            tas * cos(pitch) * sin(yaw),
            -tas * sin(pitch),
        ];
        let a_ned = [
            (v_new[0] - v_prev[0]) / dt,
            (v_new[1] - v_prev[1]) / dt,
            (v_new[2] - v_prev[2]) / dt,
        ];

        // Body angular rate approximation from Euler rates
        let roll_dot = k1[5];
        let pitch_dot = k1[4];
        let yaw_dot = k1[3];
        let p_body = roll_dot - yaw_dot * sin(pitch);
        let q_body = pitch_dot * cos(roll) + yaw_dot * sin(roll) * cos(pitch);
        let r_body = -pitch_dot * sin(roll) + yaw_dot * cos(roll) * cos(pitch);

        // Exact WGS-84 ECEF & Geodetic update from local NED step
        let dp_ned = [y1[0] - y0[0], y1[1] - y0[1], y1[2] - y0[2]];
        let r_en = r_ecef_ned(self.state.lat_deg, self.state.lon_deg);
        let dp_ecef = mat3_mul_vec(&r_en, dp_ned);
        let p_ecef = [
            self.state.p_ecef[0] + dp_ecef[0],
            self.state.p_ecef[1] + dp_ecef[1],
            self.state.p_ecef[2] + dp_ecef[2],
        ];
        let (lat_deg, lon_deg, h_m) = ecef_to_geodetic(p_ecef[0], p_ecef[1], p_ecef[2]);

        self.state.time_s += dt;
        self.state.lat_deg = lat_deg;
        self.state.lon_deg = lon_deg;
        self.state.h_m = h_m;
        self.state.p_ecef = p_ecef;
        self.state.p_ned_rel = [y1[0], y1[1], y1[2]];
        self.state.v_ned = v_new;
        self.state.a_ned = a_ned;
        self.state.yaw_rad = yaw;
        self.state.pitch_rad = pitch;
        self.state.roll_rad = roll;
        self.state.q_ned_body = euler_zyx_to_quat(yaw, pitch, roll);
        self.state.omega_body = [p_body, q_body, r_body];
        self.state.tas_mps = tas;
    }

    pub fn r_ned_body(&self) -> crate::det_f64::Mat3 {
        euler_to_r(self.state.yaw_rad, self.state.pitch_rad, self.state.roll_rad)
    }

    pub fn omega_norm(&self) -> f64 {
        let w = self.state.omega_body;
        hypot(hypot(w[0], w[1]), fabs(w[2]))
    }
}
