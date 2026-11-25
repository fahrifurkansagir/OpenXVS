use xvs_math_core::codecs::{
    compute_gdl90_crc, decode_gdl90_frame, decode_mavlink_v2_adsb_246,
    decode_mavlink_v2_bearing_track, encode_gdl90_heartbeat, encode_gdl90_ownship,
    encode_gdl90_ownship_geo_alt, encode_gdl90_traffic, encode_mavlink_v2_adsb_246,
    encode_mavlink_v2_bearing_track, frame_gdl90_message, Gdl90TargetParams,
    MavlinkAdsb246Params, GDL90_ICD_HEARTBEAT_CRC_U16, GDL90_ICD_HEARTBEAT_FRAMED,
    GDL90_ICD_HEARTBEAT_UNSTUFFED, MAVLINK_BEARING_TRACK_FRAME_LEN,
};
use xvs_math_core::daa_ecbf::{EcbfQpSolver, IntruderStateNed};
use xvs_math_core::det_f64::{fabs, norm3, sanitize_nan};
use xvs_math_core::dtaci::GuardedDtAci;
use xvs_math_core::es_ekf::{DualBankEsEkf, SpoofCheckInput};
use xvs_math_core::geo::{
    ecef_to_geodetic, euler_to_r, euler_zyx_to_quat, geodetic_to_ecef, mat3_mul_vec,
    quat_to_r, r_ecef_ned, r_to_euler, r_to_quat, so3_exp, so3_log,
};
use xvs_math_core::mesh_daa::{
    fci_fuse_3d, MeshDaaFuser, OpenXvsBearingTrack, OPENXVS_BEARING_TRACK_PAYLOAD_LEN,
};
use xvs_math_core::rig::{
    camera_pose_ecef, project_ecef_to_cam, TRAFFIC_RING_V1, XVS6_CUBE_V1,
};
use xvs_math_core::sim_engine::{FaultPerturbations, LiveDeckEngine, MissionId};
use xvs_math_core::vraim::{VisionRaimEvaluator, KSFO_28L_WIDTH_M, KSFO_TAXIWAY_C_WIDTH_M};
use xvs_rta_core::{build_rta_input_from_f64, RtaFloatTelemetry, SimplexMode, SimplexRta};

#[test]
fn test_wgs84_roundtrip_and_conventions_sign_tests() {
    // Canonical NaN sanitizer
    assert!(sanitize_nan(f64::NAN).to_bits() == 0x7FF8_0000_0000_0000);

    // Bowring round-trip < 1e-6 m across KSFO, equator, southern hemisphere, and polar
    let pts = [
        (37.6119, -122.3574, -28.3),
        (0.0, 0.0, 1200.0),
        (-33.8688, 151.2093, 540.5),
        (89.9999, 45.0, 3000.0),
        (-89.9999, -120.0, 250.0),
    ];
    for (lat, lon, h) in pts {
        let ecef = geodetic_to_ecef(lat, lon, h);
        let (lat2, lon2, h2) = ecef_to_geodetic(ecef[0], ecef[1], ecef[2]);
        let ecef2 = geodetic_to_ecef(lat2, lon2, h2);
        let err_m = norm3([ecef[0] - ecef2[0], ecef[1] - ecef2[1], ecef[2] - ecef2[2]]);
        assert!(err_m < 1e-6, "WGS-84 round-trip error {err_m} m >= 1e-6 m");
    }

    // CONVENTIONS.md §4 sign tests:
    // 1. psi = +90 deg -> body +x (forward) points East [0, 1, 0] in NED
    let r_yaw90 = euler_to_r(core::f64::consts::FRAC_PI_2, 0.0, 0.0);
    let fwd_east = mat3_mul_vec(&r_yaw90, [1.0, 0.0, 0.0]);
    assert!(fabs(fwd_east[0]) < 1e-12);
    assert!(fabs(fwd_east[1] - 1.0) < 1e-12);
    assert!(fabs(fwd_east[2]) < 1e-12);

    // 2. theta > 0 -> nose up -> body +x has negative Down component in NED
    let r_pitch = euler_to_r(0.0, 0.2, 0.0);
    let fwd_up = mat3_mul_vec(&r_pitch, [1.0, 0.0, 0.0]);
    assert!(fwd_up[2] < -0.19);

    // 3. phi > 0 -> right wing down -> body +y has positive Down component in NED
    let r_roll = euler_to_r(0.0, 0.0, 0.3);
    let right_down = mat3_mul_vec(&r_roll, [0.0, 1.0, 0.0]);
    assert!(right_down[2] > 0.29);

    // Quaternion canonical w >= 0 and round-trip < 1e-12
    let q = euler_zyx_to_quat(1.1, -0.15, 0.42);
    assert!(q[0] >= 0.0);
    let r_q = quat_to_r(q);
    let r_e = euler_to_r(1.1, -0.15, 0.42);
    for i in 0..3 {
        for j in 0..3 {
            assert!(fabs(r_q[i][j] - r_e[i][j]) < 1e-12);
        }
    }
    let (y2, p2, r2) = r_to_euler(&r_q);
    assert!(fabs(y2 - 1.1) < 1e-12 && fabs(p2 + 0.15) < 1e-12 && fabs(r2 - 0.42) < 1e-12);
    let q2 = r_to_quat(&r_e);
    assert!(q2[0] >= 0.0);

    // SO(3) Exp / Log round-trip
    let omega = [0.03, -0.05, 0.12];
    let r_exp = so3_exp(omega);
    let omega_back = so3_log(&r_exp);
    assert!(norm3([omega[0] - omega_back[0], omega[1] - omega_back[1], omega[2] - omega_back[2]]) < 1e-12);

    // Rig spec constants & behind-camera check (Review Focus #3)
    assert_eq!(XVS6_CUBE_V1.intrinsics.width, 1024);
    assert!(fabs(XVS6_CUBE_V1.intrinsics.fx - 429.6190111627674) < 1e-12);
    assert_eq!(TRAFFIC_RING_V1.intrinsics.width, 2448);
    assert_eq!(TRAFFIC_RING_V1.intrinsics.height, 2048);
    assert!(fabs(TRAFFIC_RING_V1.intrinsics.fx - 2319.0) < 1e-12);

    let p_own = geodetic_to_ecef(37.6119, -122.3574, 600.0);
    let r_en = r_ecef_ned(37.6119, -122.3574);
    let (p_cam, r_ec) = camera_pose_ecef(p_own, &r_en, &XVS6_CUBE_V1.cameras[0]);
    // Point 100 m strictly in front of camera F (+z_cam)
    let p_front = [
        p_cam[0] + 100.0 * r_ec[0][2],
        p_cam[1] + 100.0 * r_ec[1][2],
        p_cam[2] + 100.0 * r_ec[2][2],
    ];
    let uv_cam = XVS6_CUBE_V1.intrinsics.project_cam([0.0, 0.0, 100.0]).expect("cam front");
    assert!(fabs(uv_cam.0 - 511.5) < 1e-12 && fabs(uv_cam.1 - 511.5) < 1e-12);
    let uv = project_ecef_to_cam(&XVS6_CUBE_V1.intrinsics, p_front, p_cam, &r_ec).expect("front point");
    assert!(fabs(uv.0 - 511.5) < 1e-7 && fabs(uv.1 - 511.5) < 1e-7);

    // Behind-camera (z_cam <= 0) MUST return None (Review Focus #3)
    let p_behind = [
        p_cam[0] - 50.0 * r_ec[0][2],
        p_cam[1] - 50.0 * r_ec[1][2],
        p_cam[2] - 50.0 * r_ec[2][2],
    ];
    assert!(project_ecef_to_cam(&XVS6_CUBE_V1.intrinsics, p_behind, p_cam, &r_ec).is_none());
    // Exactly z_cam == 0.0 MUST also return None
    assert!(project_ecef_to_cam(&XVS6_CUBE_V1.intrinsics, p_cam, p_cam, &r_ec).is_none());
}

#[test]
fn test_dual_bank_es_ekf_spoof_ramp_vs_nominal_maneuver() {
    // 1. Coordinated 30 deg bank turn with nominal GNSS -> 0 false spoof alarms
    let mut ekf_turn = DualBankEsEkf::new([0.0, 0.0, -1200.0], [90.0, 0.0, 0.0]);
    let bank_30 = 30.0_f64.to_radians();
    let g = 9.80665;
    let speed = 90.0;
    let yaw_rate = (g / speed) * xvs_math_core::det_f64::tan(bank_30);
    let mut yaw = 0.0;
    let mut p_true = [0.0, 0.0, -1200.0];
    let mut v_prev = [speed, 0.0, 0.0];
    for tick in 0..60 {
        let dt = 0.1;
        yaw += yaw_rate * dt;
        let v_true = [
            speed * xvs_math_core::det_f64::cos(yaw),
            speed * xvs_math_core::det_f64::sin(yaw),
            0.0,
        ];
        let a_ned = [
            (v_true[0] - v_prev[0]) / dt,
            (v_true[1] - v_prev[1]) / dt,
            0.0,
        ];
        for k in 0..3 {
            p_true[k] += 0.5 * (v_prev[k] + v_true[k]) * dt;
        }
        v_prev = v_true;
        ekf_turn.step_imu_horizon(dt, a_ned, [0.0, 0.0, yaw_rate], yaw, 0.0, bank_30);
        // Nominal GNSS matches true kinematics (small realistic noise)
        let gnss_v = [v_true[0] + 0.04, v_true[1] - 0.03, 0.02];
        let res = ekf_turn.update_gnss_and_check_spoof(&SpoofCheckInput {
            gnss_pos_ned: p_true,
            gnss_vel_ned: gnss_v,
            bank_rad: bank_30,
            omega_norm_radps: fabs(yaw_rate),
        });
        assert!(
            !res.spoof_alarm,
            "False spoof alarm during 30 deg turn at tick {tick}: cusum={} nis={} ss={}",
            res.cusum_stat, res.nis_stat, res.ss_stat
        );
        if tick % 5 == 0 {
            ekf_turn.capture_stochastic_clone();
        }
        if tick % 5 == 2 {
            ekf_turn.update_visfix_cloning(p_true, 0.5);
        }
    }

    // 2. +2.5 m/s East GNSS velocity ramp -> trips within <= 10 ticks (1.0 s) & fails over to Bank 1
    let mut ekf_spoof = DualBankEsEkf::new([0.0, 0.0, -1200.0], [90.0, 0.0, 0.0]);
    for _ in 0..10 {
        ekf_spoof.step_imu_horizon(0.1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0.0, 0.0);
        let r = ekf_spoof.update_gnss_and_check_spoof(&SpoofCheckInput {
            gnss_pos_ned: ekf_spoof.bank0_pos_ned(),
            gnss_vel_ned: [90.0, 0.0, 0.0],
            bank_rad: 0.0,
            omega_norm_radps: 0.0,
        });
        assert!(!r.spoof_alarm);
    }

    let mut trip_tick = None;
    for t in 1..=10 {
        ekf_spoof.step_imu_horizon(0.1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0.0, 0.0);
        let spoof_ve = 2.5 * (t as f64) * 0.5; // +2.5 m/s ramp reaching full bias quickly
        let res = ekf_spoof.update_gnss_and_check_spoof(&SpoofCheckInput {
            gnss_pos_ned: ekf_spoof.bank0_pos_ned(),
            gnss_vel_ned: [90.0, spoof_ve.min(2.5), 0.0],
            bank_rad: 0.0,
            omega_norm_radps: 0.0,
        });
        if res.spoof_alarm && trip_tick.is_none() {
            trip_tick = Some(t);
        }
    }
    let tripped = trip_tick.expect("Bonferroni 3-layer spoof guard MUST trip within <= 10 ticks");
    assert!(tripped <= 10, "Tripped too late at tick {tripped}");
    assert_eq!(ekf_spoof.active_bank(), 1, "Must fail over to uncontaminated Bank 1");
    // Bank 1 East velocity stays uncontaminated near 0.0 m/s
    assert!(fabs(ekf_spoof.active_vel_ned()[1]) < 0.25);
}

#[test]
fn test_vraim_cam_r_isolation_and_runway_corner_go_around() {
    let vraim = VisionRaimEvaluator::new();

    // 1. Cam-R (bit 1) +0.020944 rad (+1.2 deg) extrinsic drift
    let cam_biases = [0.0, 0.020944, 0.0, 0.0, 0.0, 0.0];
    let iso = vraim.evaluate_6cam_horizon_epipolar(&cam_biases, 0x3F);
    assert_eq!(iso.isolated_cam_idx, Some(1), "Must isolate Camera R (index 1)");
    assert_eq!(iso.healthy_cam_mask, 0b111101, "Cam R bit 1 must be cleared");
    assert!(iso.epipolar_max_rad[1] > 0.006109);
    assert!(
        iso.fused_attitude_err_rad < 175e-6,
        "Remaining 5-cam fused residual {} rad >= 175 urad (0.01 deg)",
        iso.fused_attitude_err_rad
    );

    // 2. KSFO 28L (61 m) passes width check; Taxiway C (23 m) trips width check
    let nom_rwy = vraim.evaluate_runway_approach(0.0, KSFO_28L_WIDTH_M, 1800.0);
    assert!(!nom_rwy.width_fault);
    assert!(nom_rwy.v_vpl_m < 10.0);
    assert!(nom_rwy.v_hpl_m < 40.0);

    let twy_rwy = vraim.evaluate_runway_approach(0.0, KSFO_TAXIWAY_C_WIDTH_M, 1800.0);
    assert!(twy_rwy.width_fault, "Taxiway C (23 m) MUST fail KSFO 28L (61 m) width gate");

    // 3. +18 px threshold corner bias drives v_vpl_m > 10.0 m -> latched mandatory_go_around
    let bias_rwy = vraim.evaluate_runway_approach(18.0, KSFO_28L_WIDTH_M, 1200.0);
    assert!(
        bias_rwy.v_vpl_m > 10.0,
        "Expected v_vpl_m > 10.0 m under +18 px bias, got {}",
        bias_rwy.v_vpl_m
    );
    let rta_in = build_rta_input_from_f64(&RtaFloatTelemetry {
        horizon_pl_rad: iso.horizon_pl_rad,
        horizon_al_rad: 0.008727,
        v_vpl_m: bias_rwy.v_vpl_m,
        v_val_m: 10.0,
        v_hpl_m: bias_rwy.v_hpl_m,
        v_hal_m: 40.0,
        epipolar_max_rad: [0.0005; 6],
        signed_horizon_residual_rad: 0.0,
        runway_width_delta_m: bias_rwy.width_delta_m,
        runway_width_delta_limit_m: 12.0,
        cusum_stat: 0.0,
        cusum_thr: 10.530,
        ood_global_score: 0.1,
        ood_global_thr: 1.0,
        ood_cam_scores: [0.1; 6],
        ood_cam_thr: 1.0,
        cam_health_mask: 0x3F,
        ml_detector_fault: false,
        transient_dropout: false,
        approach_active: true,
        go_around_ack_reset: false,
    });
    assert!(rta_in.v_vpl_mm > 10_000);
    let mut rta = SimplexRta::new();
    let out = rta.step(&rta_in);
    assert!(out.mandatory_go_around);
    assert_eq!(out.mode, SimplexMode::InvalidInhibit);
}

#[test]
fn test_mesh_daa_triple_ray_ghost_rejection_and_ecbf_qp_infeasibility_escape() {
    // 1. Exact 52-byte OPENXVS_BEARING_TRACK round-trip
    let p_own = geodetic_to_ecef(37.62, -122.36, 1000.0);
    let trk = OpenXvsBearingTrack::from_ecef_ray(
        1_732_100_000_000_000,
        p_own,
        [0.6, 0.8, 0.0],
        [0.001, -0.002, 0.0],
        500,
        25,
        7,
        1,
        0x01,
    );
    let raw = trk.to_le_bytes();
    assert_eq!(raw.len(), OPENXVS_BEARING_TRACK_PAYLOAD_LEN);
    let trk2 = OpenXvsBearingTrack::from_le_bytes(&raw);
    assert_eq!(trk, trk2);

    // 2. 2-ship shared-corridor ghost rejected when 3rd peer joins
    // Two ships at y = -1000 and y = +1000 in local horizontal plane (ECEF centered) observing
    // two coplanar targets T1=(2000, -400, 0) and T2=(2000, +400, 0) create 2 real + 2 ghost intersections!
    let base = geodetic_to_ecef(37.62, -122.36, 1000.0);
    let r_en = r_ecef_ned(37.62, -122.36);
    let to_ecef = |ned: [f64; 3]| -> [f64; 3] {
        let d = mat3_mul_vec(&r_en, ned);
        [base[0] + d[0], base[1] + d[1], base[2] + d[2]]
    };
    let s1 = to_ecef([0.0, -1000.0, 0.0]);
    let s2 = to_ecef([0.0, 1000.0, 0.0]);
    let s3 = to_ecef([-800.0, 0.0, -600.0]); // 3rd peer out-of-plane breaks coplanar ghost ambiguity!
    let t1 = to_ecef([2000.0, -400.0, 0.0]);
    let t2 = to_ecef([2000.0, 400.0, 0.0]);

    let make_ray = |o: [f64; 3], tgt: [f64; 3], sender: u8, tid: u16| -> OpenXvsBearingTrack {
        let d = [tgt[0] - o[0], tgt[1] - o[1], tgt[2] - o[2]];
        let n = norm3(d);
        OpenXvsBearingTrack::from_ecef_ray(
            1000,
            o,
            [d[0] / n, d[1] / n, d[2] / n],
            [0.0; 3],
            2000,
            50,
            tid,
            sender,
            0x01,
        )
    };

    let obs_2ship = [
        make_ray(s1, t1, 1, 1),
        make_ray(s1, t2, 1, 2),
        make_ray(s2, t1, 2, 3),
        make_ray(s2, t2, 2, 4),
    ];
    let mut fuser = MeshDaaFuser::new();
    let res2 = fuser.fuse(&obs_2ship);
    assert!(res2.num_fused >= 3, "2-ship coplanar geometry produces ghosts ({})", res2.num_fused);

    let obs_3ship = [
        obs_2ship[0],
        obs_2ship[1],
        obs_2ship[2],
        obs_2ship[3],
        make_ray(s3, t1, 3, 5),
        make_ray(s3, t2, 3, 6),
    ];
    let res3 = fuser.fuse(&obs_3ship);
    assert!(res3.ghosts_rejected >= 2, "3rd peer MUST reject >= 2 coplanar ghosts, got {}", res3.ghosts_rejected);
    assert_eq!(res3.num_fused, 2, "Exactly 2 real targets should survive triple-ray consensus");

    // Verify FCI produces conservative finite covariance
    let pa = [[100.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 200.0]];
    let pb = [[300.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 200.0]];
    let (xf, pf) = fci_fuse_3d([10.0, 20.0, 30.0], &pa, [12.0, 18.0, 31.0], &pb);
    assert!(xf[0] > 10.0 && xf[0] < 12.0);
    assert!(pf[0][0] > 0.0 && pf[0][0] < 300.0);

    // 3. ECBF-QP resolves 3 converging intruders with 0 NMAC / 0 LoWC
    let solver = EcbfQpSolver::default();
    let mut p_own = [0.0, 0.0, -1000.0];
    let mut v_own = [90.0, 0.0, 0.0];
    let mut ints = [
        IntruderStateNed { pos_ned: [4200.0, 80.0, -1000.0], vel_ned: [-85.0, 0.0, 0.0], valid: true },
        IntruderStateNed { pos_ned: [3600.0, 2200.0, -1000.0], vel_ned: [-30.0, -75.0, 0.0], valid: true },
        IntruderStateNed { pos_ned: [3900.0, -2400.0, -960.0], vel_ned: [-35.0, 80.0, -1.0], valid: true },
    ];
    let mut min_h = f64::INFINITY;
    let mut nmac_or_lowc = false;
    let dt = 0.1;
    let g = 9.80665;
    for _ in 0..300 {
        let sol = solver.solve(p_own, v_own, 0.0, 0.0, &ints);
        assert!(sol.bank_cmd_rad.is_finite() && sol.climb_rate_cmd_mps.is_finite());
        assert!(fabs(sol.bank_cmd_rad) <= 30.0_f64.to_radians() + 1e-9);
        // Kinematic update of ownship & intruders
        let spd = xvs_math_core::det_f64::hypot(v_own[0], v_own[1]).max(60.0);
        let hdg = xvs_math_core::det_f64::atan2(v_own[1], v_own[0]);
        let hdg_next = hdg + (g / spd) * xvs_math_core::det_f64::tan(sol.bank_cmd_rad) * dt;
        v_own[0] = spd * xvs_math_core::det_f64::cos(hdg_next);
        v_own[1] = spd * xvs_math_core::det_f64::sin(hdg_next);
        v_own[2] = -sol.climb_rate_cmd_mps;
        for k in 0..3 {
            p_own[k] += v_own[k] * dt;
        }
        for intr in &mut ints {
            for k in 0..3 {
                intr.pos_ned[k] += intr.vel_ned[k] * dt;
            }
            let dh = xvs_math_core::det_f64::hypot(p_own[0] - intr.pos_ned[0], p_own[1] - intr.pos_ned[1]);
            let dv = fabs(p_own[2] - intr.pos_ned[2]);
            if dh < min_h {
                min_h = dh;
            }
            if dh < 1219.2 && dv < 137.16 {
                nmac_or_lowc = true;
            }
        }
    }
    assert!(!nmac_or_lowc, "ECBF-QP MUST achieve 0 NMAC / 0 LoWC on 3 converging intruders (min_h={min_h:.1} m)");

    // 4. Review Focus #3: Adversarial head-on + bilateral sandwich saturating +/-30 deg bank
    // triggers slack + deterministic max-rate escape with finite controls and ecbf_infeasible == true
    let sandwich = [
        IntruderStateNed { pos_ned: [220.0, 0.0, -1000.0], vel_ned: [-100.0, 0.0, 0.0], valid: true },
        IntruderStateNed { pos_ned: [120.0, -120.0, -1000.0], vel_ned: [-20.0, 95.0, 0.0], valid: true },
        IntruderStateNed { pos_ned: [120.0, 120.0, -1000.0], vel_ned: [-20.0, -95.0, 0.0], valid: true },
        IntruderStateNed { pos_ned: [140.0, 0.0, -1030.0], vel_ned: [-40.0, 0.0, 15.0], valid: true },
        IntruderStateNed { pos_ned: [140.0, 0.0, -970.0], vel_ned: [-40.0, 0.0, -15.0], valid: true },
    ];
    let esc = solver.solve([0.0, 0.0, -1000.0], [95.0, 0.0, 0.0], 0.0, 0.0, &sandwich);
    assert!(esc.ecbf_infeasible, "Adversarial bilateral+vertical sandwich MUST set ecbf_infeasible = true");
    assert!(esc.bank_cmd_rad.is_finite() && esc.climb_rate_cmd_mps.is_finite());
    assert!(fabs(esc.bank_cmd_rad) <= 30.0_f64.to_radians() + 1e-9);
}

#[test]
fn test_gdl90_and_mavlink_v2_wire_frames() {
    // 1. Official FAA GDL-90 ICD §2.2.3 Heartbeat test vector
    assert_eq!(compute_gdl90_crc(&GDL90_ICD_HEARTBEAT_UNSTUFFED), GDL90_ICD_HEARTBEAT_CRC_U16);
    let mut framed_buf = [0u8; 64];
    let n_icd = frame_gdl90_message(&GDL90_ICD_HEARTBEAT_UNSTUFFED, &mut framed_buf);
    assert_eq!(&framed_buf[..n_icd], &GDL90_ICD_HEARTBEAT_FRAMED);

    // 2. Byte-stuffing with 0x7E and 0x7D inside payload
    let tricky = [0x00, 0x7E, 0x7D, 0x5E, 0x5D, 0x12, 0x34];
    let n_tricky = frame_gdl90_message(&tricky, &mut framed_buf);
    let mut decoded_buf = [0u8; 64];
    let n_dec = decode_gdl90_frame(&framed_buf[..n_tricky], &mut decoded_buf).expect("valid frame");
    assert_eq!(&decoded_buf[..n_dec], &tricky);

    // 3. Heartbeat, Ownship, Geo Alt, and Traffic Report with sim_prefix guard
    let n_hb = encode_gdl90_heartbeat(43200, true, false, true, 0, 0, &mut framed_buf);
    assert!(decode_gdl90_frame(&framed_buf[..n_hb], &mut decoded_buf).is_ok());
    let n_geo = encode_gdl90_ownship_geo_alt(3500.0, 3.0, false, &mut framed_buf);
    assert!(decode_gdl90_frame(&framed_buf[..n_geo], &mut decoded_buf).is_ok());

    let tgt = Gdl90TargetParams {
        alert_status: 1,
        address_type: 0,
        participant_address: 0xF00001,
        lat_deg: 37.6119,
        lon_deg: -122.3574,
        altitude_ft: Some(2500.0),
        misc_indicators: 0x09,
        nic: 8,
        nacp: 8,
        horiz_vel_kt: Some(180.0),
        vert_vel_fpm: Some(-500.0),
        track_heading_deg: 280.0,
        emitter_category: 1,
        callsign: *b"XVS0001 ",
        emergency_code: 0,
    };
    let n_own = encode_gdl90_ownship(&tgt, &mut framed_buf);
    assert!(decode_gdl90_frame(&framed_buf[..n_own], &mut decoded_buf).is_ok());
    let n_trf = encode_gdl90_traffic(&tgt, true, &mut framed_buf);
    let n_trf_dec = decode_gdl90_frame(&framed_buf[..n_trf], &mut decoded_buf).expect("valid traffic");
    assert_eq!(n_trf_dec, 28);
    assert_eq!(&decoded_buf[19..22], b"SIM", "sim_prefix MUST enforce SIM callsign prefix");

    // 4. MAVLink v2 #246 ADSB_VEHICLE (50 bytes, CRC_EXTRA=184)
    let adsb = MavlinkAdsb246Params {
        icao_address: 0xF00007,
        lat_deg_e7: 376_119_000,
        lon_deg_e7: -1_223_574_000,
        altitude_mm: 600_000,
        heading_cdeg: 28_000,
        hor_velocity_cms: 9_000,
        ver_velocity_cms: -250,
        flags: 0x00FF,
        squawk: 1200,
        altitude_type: 1,
        callsign: *b"XVS0007\0\0",
        emitter_type: 14,
        tslc: 0,
        seq: 5,
        sysid: 1,
        compid: 156,
    };
    let frame_246 = encode_mavlink_v2_adsb_246(&adsb);
    assert_eq!(frame_246.len(), 50);
    let dec_246 = decode_mavlink_v2_adsb_246(&frame_246).expect("valid MAVLink #246");
    assert_eq!(dec_246.icao_address, adsb.icao_address);
    assert_eq!(dec_246.lat_deg_e7, adsb.lat_deg_e7);
    assert_eq!(dec_246.squawk, 1200);

    // 5. Exact 77-byte signed MAVLink v2 #52001 OPENXVS_BEARING_TRACK round-trip
    let key = [0xA5u8; 32];
    let p_own = geodetic_to_ecef(37.62, -122.36, 1000.0);
    let trk = OpenXvsBearingTrack::from_ecef_ray(
        1_732_100_000_123_456,
        p_own,
        [0.0, 1.0, 0.0],
        [0.0005, 0.0, -0.0002],
        450,
        30,
        42,
        2,
        0x03,
    );
    let signed = encode_mavlink_v2_bearing_track(&trk, 11, 1, 156, 3, 987_654_321, &key);
    assert_eq!(signed.len(), MAVLINK_BEARING_TRACK_FRAME_LEN);
    assert_eq!(signed.len(), 77);
    let (dec_trk, link_id, ts48) =
        decode_mavlink_v2_bearing_track(&signed, &key).expect("valid signed #52001");
    assert_eq!(dec_trk, trk);
    assert_eq!(link_id, 3);
    assert_eq!(ts48, 987_654_321);

    // Tampered byte MUST fail signature / CRC verification
    let mut bad = signed;
    bad[20] ^= 0x01;
    assert!(decode_mavlink_v2_bearing_track(&bad, &key).is_err());
}

#[test]
fn test_guarded_dtaci_ood_write_gate_freeze() {
    let mut dtaci = GuardedDtAci::new();
    let base_alpha = dtaci.alpha(500.0);
    let cone_nom = dtaci.update(500.0, 0.002, 0.3);
    assert!(!dtaci.is_frozen());
    assert!(cone_nom > 0.0);
    let adapted_alpha = dtaci.alpha(500.0);
    assert!(fabs(adapted_alpha - base_alpha) > 0.0);

    // Under OOD (u_ood >= 1.0), write gate MUST freeze alpha and inflate cone conservatively
    let cone_ood = dtaci.update(500.0, 0.050, 1.5);
    assert!(dtaci.is_frozen());
    assert_eq!(dtaci.alpha(500.0), adapted_alpha, "OOD write gate MUST freeze alpha");
    assert!(cone_ood >= cone_nom);
}

#[test]
fn test_livedeck_engine_emergent_missions_m1_to_m4() {
    let mut eng = LiveDeckEngine::new(20261120);

    // Mission M1: Cam-R +0.020944 rad (+1.2 deg) drift -> emergent ISOLATED_5CAM within <= 2 ticks
    eng.reset_mission(MissionId::M1CamRDrift, 20261120);
    for _ in 0..10 {
        eng.step_rta_tick();
    }
    assert_eq!(eng.snapshot().simplex_mode, SimplexMode::NominalMl as u8);
    let f1 = FaultPerturbations {
        cam_r_bias_rad: 0.020944,
        ..Default::default()
    };
    eng.set_faults(f1);
    eng.step_rta_tick();
    eng.step_rta_tick();
    let s1 = eng.snapshot();
    assert_eq!(s1.simplex_mode, SimplexMode::Isolated5Cam as u8);
    assert_eq!(s1.healthy_cam_mask, 0b111101);
    assert!(s1.fused_attitude_err_urad < 175.0);

    // Mission M2: +2.5 m/s East GNSS velocity ramp -> emergent FALLBACK_D0_SAFE within <= 10 ticks
    eng.reset_mission(MissionId::M2GpsSpoof, 20261120);
    for _ in 0..10 {
        eng.step_rta_tick();
    }
    let f2 = FaultPerturbations {
        gnss_spoof_vel_ned_mps: [0.0, 2.5, 0.0],
        ..Default::default()
    };
    eng.set_faults(f2);
    let mut tripped = false;
    for _ in 0..10 {
        eng.step_rta_tick();
        if eng.snapshot().simplex_mode == SimplexMode::FallbackD0Safe as u8
            && eng.snapshot().active_nav_bank == 1
        {
            tripped = true;
            break;
        }
    }
    assert!(tripped, "M2 MUST trip to FALLBACK_D0_SAFE + Bank 1 within <= 10 ticks");

    // Mission M3: 3-ship ghost rejection & ECBF-QP avoidance
    eng.reset_mission(MissionId::M3MeshDaa, 20261120);
    for _ in 0..5 {
        eng.step_rta_tick();
    }
    let f3 = FaultPerturbations {
        peer3_active: true,
        ..Default::default()
    };
    eng.set_faults(f3);
    for _ in 0..20 {
        eng.step_rta_tick();
    }
    assert!(eng.snapshot().ghosts_rejected >= 1);

    // Mission M4: KSFO 28L +18 px corner bias -> emergent MANDATORY GO-AROUND + climb-out
    eng.reset_mission(MissionId::M4KsfoApproach, 20261120);
    for _ in 0..5 {
        eng.step_rta_tick();
    }
    let f4 = FaultPerturbations {
        runway_corner_bias_px: 18.0,
        fog_contrast: 0.22,
        lwir_efvs_enabled: true,
        ..Default::default()
    };
    eng.set_faults(f4);
    eng.step_rta_tick();
    let s4 = eng.snapshot();
    assert_eq!(s4.mandatory_go_around, 1);
    assert_eq!(s4.simplex_mode, SimplexMode::InvalidInhibit as u8);
    assert_eq!(s4.blank_conformal_hud, 1);

    // Verify TX ring drained real GDL-90 and MAVLink v2 packets
    let mut pkt = [0u8; 128];
    let mut gdl_cnt = 0usize;
    let mut mav_cnt = 0usize;
    while let Some((proto, len)) = eng.drain_tx_packet(&mut pkt) {
        if proto == 0x01 && len >= 5 {
            gdl_cnt += 1;
        } else if proto == 0x02 && (len == 50 || len == 77) {
            mav_cnt += 1;
        }
    }
    assert!(gdl_cnt > 0 && mav_cnt > 0, "Expected non-empty GDL-90 and MAVLink TX ring");
}

#[test]
fn test_d16_m1_b1_nan_fail_closed_propagation_all_channels() {
    // 1. Direct ES-EKF NaN GNSS velocity -> skips Bank 0 assimilation (v_ned finite), selects Bank 1, returns NaN stats
    let mut ekf = DualBankEsEkf::new([0.0, 0.0, -1000.0], [90.0, 0.0, 0.0]);
    ekf.step_imu_horizon(0.1, [0.0; 3], [0.0; 3], 0.0, 0.0, 0.0);
    let res = ekf.update_gnss_and_check_spoof(&SpoofCheckInput {
        gnss_pos_ned: [9.0, 0.0, -1000.0],
        gnss_vel_ned: [90.0, f64::NAN, 0.0],
        bank_rad: 0.0,
        omega_norm_radps: 0.0,
    });
    assert!(res.spoof_alarm);
    assert_eq!(res.active_bank, 1);
    assert!(res.nis_stat.is_nan() && res.cusum_stat.is_nan() && res.ss_stat.is_nan());
    assert!(ekf.bank0_vel_ned().iter().all(|v| v.is_finite()), "Bank 0 must NOT assimilate NaN");

    // 2. LiveDeckEngine: gnss_spoof_vel_ned_mps[1] = NaN -> P0 (INVALID_INHIBIT) at 0 ticks
    let mut eng = LiveDeckEngine::new(20261120);
    eng.reset_mission(MissionId::M2GpsSpoof, 20261120);
    eng.step_rta_tick();
    assert_eq!(eng.snapshot().simplex_mode, SimplexMode::NominalMl as u8);
    eng.set_faults(FaultPerturbations {
        gnss_spoof_vel_ned_mps: [0.0, f64::NAN, 0.0],
        ..Default::default()
    });
    eng.step_rta_tick();
    let s_gnss = eng.snapshot();
    assert_eq!(s_gnss.simplex_mode, SimplexMode::InvalidInhibit as u8);
    assert_eq!(s_gnss.active_nav_bank, 1);
    assert_eq!(s_gnss.blank_conformal_hud, 1);
    assert!(s_gnss.v_ned.iter().all(|v| v.is_finite()));

    // 3. LiveDeckEngine: runway_corner_bias_px = NaN during M4 approach -> P0 + mandatory_go_around at 0 ticks
    eng.reset_mission(MissionId::M4KsfoApproach, 20261120);
    eng.step_rta_tick();
    eng.set_faults(FaultPerturbations {
        runway_corner_bias_px: f64::NAN,
        ..Default::default()
    });
    eng.step_rta_tick();
    let s_rwy = eng.snapshot();
    assert_eq!(s_rwy.simplex_mode, SimplexMode::InvalidInhibit as u8);
    assert_eq!(s_rwy.mandatory_go_around, 1);
    assert_eq!(s_rwy.blank_conformal_hud, 1);

    // 4. LiveDeckEngine: cam_r_bias_rad = NaN -> P0 at 0 ticks
    eng.reset_mission(MissionId::M1CamRDrift, 20261120);
    eng.step_rta_tick();
    eng.set_faults(FaultPerturbations {
        cam_r_bias_rad: f64::NAN,
        ..Default::default()
    });
    eng.step_rta_tick();
    assert_eq!(eng.snapshot().simplex_mode, SimplexMode::InvalidInhibit as u8);

    // 5. LiveDeckEngine: ood_global_score = NaN -> P0 at 0 ticks
    eng.reset_mission(MissionId::M1CamRDrift, 20261120);
    eng.step_rta_tick();
    eng.set_faults(FaultPerturbations {
        ood_global_score: f64::NAN,
        ..Default::default()
    });
    eng.step_rta_tick();
    assert_eq!(eng.snapshot().simplex_mode, SimplexMode::InvalidInhibit as u8);
}
