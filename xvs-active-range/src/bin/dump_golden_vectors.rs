// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Emits `tests/golden/active_range_vectors.json` (10 deterministic multi-step trajectories
//! with exact inputs, `ActiveRangeTelemetry` outputs, and 64-byte `wire_hex` + `crc32`)
//! locking bit-exact Rust <-> Python <-> TypeScript parity.

#[cfg(not(feature = "std"))]
fn main() {}

#[cfg(feature = "std")]
mod gen {
    use std::env;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::Path;

    use xvs_active_range::{
        encode_wire, wrap_pi, ActiveRangeConfig, ActiveRangeController, ActiveRangeMeasurement,
        ActiveRangeTelemetry, OwnshipStateNed, G_MPS2,
    };

struct StepRecord {
    own: OwnshipStateNed,
    meas: ActiveRangeMeasurement,
    out: ActiveRangeTelemetry,
    wire_hex: String,
}

struct Trajectory {
    name: &'static str,
    description: &'static str,
    steps: Vec<StepRecord>,
}

fn hex_64(bytes: &[u8; 64]) -> String {
    let mut s = String::with_capacity(128);
    for b in bytes {
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn sim_closed_loop(
    name: &'static str,
    description: &'static str,
    cfg: ActiveRangeConfig,
    int_p0: [f64; 3],
    int_v: [f64; 3],
    wingspan_m: f64,
    own_tas: f64,
    n_steps: usize,
    start_track_age: u32,
    daytime_clear: bool,
    contrast_ratio: f64,
) -> Trajectory {
    let mut ctrl = ActiveRangeController::new(cfg);
    let dt = 0.1_f64;
    let focal_px = 2690.0_f64;
    let mut own_p = [0.0_f64, 0.0, -500.0];
    let mut own_hdg = 0.0_f64;
    let mut own_a = [0.0_f64; 3];
    let mut int_p = int_p0;
    let mut prev_s = 0.0_f64;
    let mut steps = Vec::with_capacity(n_steps);

    for idx in 0..n_steps {
        let rel = [
            int_p[0] - own_p[0],
            int_p[1] - own_p[1],
            int_p[2] - own_p[2],
        ];
        let r_true = libm::sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
        let horiz = libm::hypot(rel[0], rel[1]);
        let alpha = libm::atan2(rel[1], rel[0]);
        let beta = libm::atan2(-rel[2], horiz);
        let s_now = focal_px * wingspan_m / r_true.max(1.0);
        let own_v = [own_tas * libm::cos(own_hdg), own_tas * libm::sin(own_hdg), 0.0];
        let v_rel = [int_v[0] - own_v[0], int_v[1] - own_v[1], int_v[2] - own_v[2]];
        let r_dot = (rel[0] * v_rel[0] + rel[1] * v_rel[1] + rel[2] * v_rel[2]) / r_true.max(1.0);
        let s_rate = if idx == 0 {
            -s_now * (r_dot / r_true.max(1.0))
        } else {
            (s_now - prev_s) / dt
        };
        prev_s = s_now;

        let own = OwnshipStateNed {
            v_ned: own_v,
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
            contrast_ratio,
            sky_beta_ext: 2.0e-4,
            daytime_clear,
            track_age: start_track_age + idx as u32,
            dt_s: dt,
        };

        let out = ctrl.step(&own, &meas);
        let wire = encode_wire(&out);
        steps.push(StepRecord {
            own,
            meas,
            out,
            wire_hex: hex_64(&wire),
        });

        let phi = out.cmd_bank_rad as f64;
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
        int_p[2] += int_v[2] * dt;
    }

    Trajectory {
        name,
        description,
        steps,
    }
}

fn fmt_f64(v: f64) -> String {
    format!("{:.15e}", v)
}

fn fmt_f32(v: f32) -> String {
    format!("{:.9e}", v)
}

    pub fn run() {
    let out_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/golden/active_range_vectors.json".to_string());
    let cfg = ActiveRangeConfig::default();

    let trajectories = vec![
        sim_closed_loop(
            "cbdr_headon_1000m",
            "Pure head-on CBDR at 1000m, 80 m/s closure, unknown 10m wingspan (31 ticks = 3.0s)",
            cfg,
            [1000.0, 0.0, -500.0],
            [-45.0, 0.0, 0.0],
            10.0,
            35.0,
            31,
            5,
            true,
            0.65,
        ),
        sim_closed_loop(
            "cbdr_headon_500m_small_uav",
            "Head-on CBDR small 2.0m UAV at 500m, 65 m/s closure",
            cfg,
            [500.0, 0.0, -510.0],
            [-30.0, 0.0, 0.0],
            2.0,
            35.0,
            25,
            5,
            true,
            0.78,
        ),
        sim_closed_loop(
            "cbdr_headon_1800m_regional",
            "Head-on CBDR large 28m aircraft at 1800m, 100 m/s closure",
            cfg,
            [1800.0, 0.0, -540.0],
            [-65.0, 0.0, 0.0],
            28.0,
            35.0,
            31,
            5,
            true,
            0.48,
        ),
        sim_closed_loop(
            "off_axis_plus5deg_260m",
            "Off-axis +5 deg close-in threat at 260m rejecting positive bank toward threat",
            cfg,
            [260.0 * libm::cos(5.0_f64.to_radians()), 260.0 * libm::sin(5.0_f64.to_radians()), -500.0],
            [-10.0, 0.0, 0.0],
            6.0,
            35.0,
            10,
            6,
            true,
            0.82,
        ),
        sim_closed_loop(
            "off_axis_minus7deg_650m",
            "Off-axis -7 deg closing threat at 650m banking right away from intruder",
            cfg,
            [650.0 * libm::cos(-7.0_f64.to_radians()), 650.0 * libm::sin(-7.0_f64.to_radians()), -520.0],
            [-35.0, 3.0, 0.0],
            11.0,
            35.0,
            20,
            5,
            true,
            0.70,
        ),
        sim_closed_loop(
            "zero_closure_crossing",
            "Zero-closure perpendicular crossing traffic (no false ActiveSwerve)",
            cfg,
            [900.0, 0.0, -520.0],
            [35.0, 50.0, 0.0],
            11.0,
            35.0,
            15,
            5,
            true,
            0.60,
        ),
        sim_closed_loop(
            "sub_6px_distant_koschmieder",
            "Sub-6px distant dot above horizon with daytime Koschmieder prior",
            cfg,
            [2400.0, 0.0, -680.0],
            [-55.0, 0.0, 0.0],
            3.0,
            35.0,
            15,
            5,
            true,
            0.72,
        ),
        sim_closed_loop(
            "terminal_immediate_evade_190m",
            "Terminal close-in threat (<220m, tau < 4.5s) triggering ImmediateEvade +/-25 deg",
            cfg,
            [190.0, 12.0, -500.0],
            [-40.0, -2.0, 0.0],
            8.0,
            35.0,
            6,
            6,
            true,
            0.88,
        ),
        sim_closed_loop(
            "elevated_descending_cbdr_1200m",
            "3D elevated descending intruder (+4 deg elevation) at 1200m",
            cfg,
            [1200.0, 0.0, -585.0],
            [-50.0, 0.0, 4.0],
            12.0,
            35.0,
            25,
            5,
            true,
            0.55,
        ),
        sim_closed_loop(
            "low_airspeed_envelope_guard",
            "Low airspeed below 1.25*V_stall envelope guard holding NominalHold",
            cfg,
            [900.0, 0.0, -500.0],
            [-40.0, 0.0, 0.0],
            10.0,
            17.0,
            8,
            5,
            false,
            0.50,
        ),
    ];

    let mut json = String::new();
    json.push_str("{\n");
    let _ = writeln!(json, "  \"schema\": \"xvs.v3.active_range_golden_vectors/1\",");
    let _ = writeln!(json, "  \"config\": {{");
    let _ = writeln!(
        json,
        "    \"r_init_m\": [{}, {}, {}, {}, {}, {}],",
        fmt_f64(cfg.r_init_m[0]),
        fmt_f64(cfg.r_init_m[1]),
        fmt_f64(cfg.r_init_m[2]),
        fmt_f64(cfg.r_init_m[3]),
        fmt_f64(cfg.r_init_m[4]),
        fmt_f64(cfg.r_init_m[5])
    );
    let _ = writeln!(json, "    \"sigma_bearing_rad\": {},", fmt_f64(cfg.sigma_bearing_rad));
    let _ = writeln!(json, "    \"q_hat_conformal\": {}", fmt_f64(cfg.q_hat_conformal));
    json.push_str("  },\n");
    json.push_str("  \"trajectories\": [\n");

    for (ti, traj) in trajectories.iter().enumerate() {
        json.push_str("    {\n");
        let _ = writeln!(json, "      \"name\": \"{}\",", traj.name);
        let _ = writeln!(json, "      \"description\": \"{}\",", traj.description);
        json.push_str("      \"steps\": [\n");
        for (si, st) in traj.steps.iter().enumerate() {
            json.push_str("        {\n");
            let _ = writeln!(
                json,
                "          \"own\": {{\"v_ned\": [{}, {}, {}], \"a_ned\": [{}, {}, {}], \"heading_rad\": {}, \"tas_mps\": {}, \"v_stall_mps\": {}}},",
                fmt_f64(st.own.v_ned[0]),
                fmt_f64(st.own.v_ned[1]),
                fmt_f64(st.own.v_ned[2]),
                fmt_f64(st.own.a_ned[0]),
                fmt_f64(st.own.a_ned[1]),
                fmt_f64(st.own.a_ned[2]),
                fmt_f64(st.own.heading_rad),
                fmt_f64(st.own.tas_mps),
                fmt_f64(st.own.v_stall_mps),
            );
            let _ = writeln!(
                json,
                "          \"meas\": {{\"alpha_rad\": {}, \"beta_rad\": {}, \"size_px\": {}, \"size_rate_px_s\": {}, \"contrast_ratio\": {}, \"sky_beta_ext\": {}, \"daytime_clear\": {}, \"track_age\": {}, \"dt_s\": {}}},",
                fmt_f64(st.meas.alpha_rad),
                fmt_f64(st.meas.beta_rad),
                fmt_f64(st.meas.size_px),
                fmt_f64(st.meas.size_rate_px_s),
                fmt_f64(st.meas.contrast_ratio),
                fmt_f64(st.meas.sky_beta_ext),
                if st.meas.daytime_clear { "true" } else { "false" },
                st.meas.track_age,
                fmt_f64(st.meas.dt_s),
            );
            let _ = writeln!(
                json,
                "          \"out\": {{\"seq\": {}, \"mode\": {}, \"active_bank_idx\": {}, \"flags\": {}, \"r_hat_m\": {}, \"sigma_r_m\": {}, \"r_safe_lower_m\": {}, \"rho_dot_inv_s\": {}, \"cmd_bank_rad\": {}, \"min_sc3bf_margin\": {}, \"fim_gain\": {}, \"weights\": [{}, {}, {}, {}, {}, {}], \"crc32\": {}}},",
                st.out.seq,
                st.out.mode,
                st.out.active_bank_idx,
                st.out.flags,
                fmt_f32(st.out.r_hat_m),
                fmt_f32(st.out.sigma_r_m),
                fmt_f32(st.out.r_safe_lower_m),
                fmt_f32(st.out.rho_dot_inv_s),
                fmt_f32(st.out.cmd_bank_rad),
                fmt_f32(st.out.min_sc3bf_margin),
                fmt_f32(st.out.fim_gain),
                fmt_f32(st.out.weights[0]),
                fmt_f32(st.out.weights[1]),
                fmt_f32(st.out.weights[2]),
                fmt_f32(st.out.weights[3]),
                fmt_f32(st.out.weights[4]),
                fmt_f32(st.out.weights[5]),
                st.out.crc32,
            );
            let _ = writeln!(json, "          \"wire_hex\": \"{}\"", st.wire_hex);
            if si + 1 < traj.steps.len() {
                json.push_str("        },\n");
            } else {
                json.push_str("        }\n");
            }
        }
        json.push_str("      ]\n");
        if ti + 1 < trajectories.len() {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    let dest = Path::new(&out_path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("create golden directory");
    }
    fs::write(dest, json.as_bytes()).expect("write golden vectors json");
    println!(
        "Wrote {} trajectories ({} bytes) to {}",
        trajectories.len(),
        json.len(),
        dest.display()
    );
    }
}

#[cfg(feature = "std")]
fn main() {
    gen::run();
}
