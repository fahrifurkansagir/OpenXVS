"""Tests for DR-15 Tier-0A/0B exact geodetic ADS-B-to-pixel auto-labeler & compliance locks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from xvs.v2.xvs_realsky import (
    AirframeDimensionsM,
    CameraIntrinsics,
    RealSkyComplianceError,
    assert_adsb_provider_allowed,
    assert_thermal_sensor_sanctions_clean,
    ecef_to_geodetic,
    estimate_airframe_box_prior_px,
    geodetic_to_ecef,
    geodetic_to_local_ned,
    interpolate_cubic_hermite_ned,
    local_ned_to_ecef,
    ned_velocity_from_adsb,
    project_ned_to_pixel,
    propagate_causal_turn_ned,
    reconcile_ellipsoidal_altitude_m,
    sanitize_realsky_label_record,
)
from xvs.v2.xvs_realsky.autolabel import (
    apply_sky_elevation_mask,
    compute_sky_elevation_mask,
    ed25519_public_key_from_seed,
    ed25519_sign,
    ed25519_verify,
    load_camera_calibration,
    sign_manifest_sha256,
    snap_target_contrast_centroid,
    verify_manifest_signature,
)


def test_vertical_datum_qnh_cold_temp_geoid18_eliminates_baro_trough_pixel_bias() -> None:
    """Verify ISA QNH + ICAO cold-temp + GEOID18 reconciliation vs ~80 m weather-trough bias (144-205 px @ 1 km)."""
    # Prefer geom_alt_ft directly when GVA >= 1
    h_geom_hae = reconcile_ellipsoidal_altitude_m(baro_alt_ft=3500.0, geom_alt_ft=3400.0, gva=2)
    assert h_geom_hae == pytest.approx(3400.0 * 0.3048, abs=1e-6)

    h_geom_msl = reconcile_ellipsoidal_altitude_m(
        baro_alt_ft=3500.0,
        geom_alt_ft=3400.0,
        gva=1,
        geom_is_orthometric_msl=True,
        geoid_undulation_m=-32.63,
    )
    assert h_geom_msl == pytest.approx(3400.0 * 0.3048 - 32.63, abs=1e-6)

    # Routine low-pressure weather trough (QNH = 1003.7 hPa -> ~80 m vertical bias below standard 1013.25)
    baro_ft = 3280.84  # 1000.0 m pressure altitude
    h_naive_m = baro_ft * 0.3048
    h_corr_m = reconcile_ellipsoidal_altitude_m(
        baro_alt_ft=baro_ft,
        geom_alt_ft=None,
        gva=0,
        qnh_hpa=1003.7,
        t_amb_k=288.15,
        station_elev_m=0.0,
        geoid_undulation_m=0.0,
    )
    delta_z_m = abs(h_naive_m - h_corr_m)
    assert 75.0 <= delta_z_m <= 85.0

    # At 1 km slant range: 4K phone (f=2560 px) -> ~205 px; IMX296 6.2mm (f=1797 px) -> ~144 px
    shift_phone_px = (delta_z_m / 1000.0) * 2560.0
    shift_imx_px = (delta_z_m / 1000.0) * 1797.0
    assert 190.0 <= shift_phone_px <= 220.0
    assert 135.0 <= shift_imx_px <= 155.0

    # Cold-temperature contraction below ISA lowers true MSL further above station
    h_cold_m = reconcile_ellipsoidal_altitude_m(
        baro_alt_ft=baro_ft,
        qnh_hpa=1003.7,
        t_amb_k=268.15,  # -5 C
        station_elev_m=10.0,
        geoid_undulation_m=-32.63,
    )
    assert h_cold_m < h_corr_m - 60.0


def test_wgs84_geodetic_ecef_local_ned_roundtrip_submillimeter() -> None:
    """Verify WGS-84 geodetic -> ECEF -> local NED -> ECEF -> geodetic roundtrip < 1e-4 m."""
    ksfo_lat, ksfo_lon, ksfo_h = 37.61356, -122.35714, -28.67
    tgt_lat, tgt_lon, tgt_h = 37.62850, -122.33900, 480.0

    ecef_t = geodetic_to_ecef(tgt_lat, tgt_lon, tgt_h)
    lat_rt, lon_rt, h_rt = ecef_to_geodetic(ecef_t)
    assert lat_rt == pytest.approx(tgt_lat, abs=1e-9)
    assert lon_rt == pytest.approx(tgt_lon, abs=1e-9)
    assert h_rt == pytest.approx(tgt_h, abs=1e-4)

    p_ned = geodetic_to_local_ned(
        tgt_lat,
        tgt_lon,
        tgt_h,
        origin_lat_deg=ksfo_lat,
        origin_lon_deg=ksfo_lon,
        origin_alt_ellip_m=ksfo_h,
    )
    # Target is North-East and higher (negative Down) than KSFO origin
    assert p_ned[0] > 1500.0
    assert p_ned[1] > 1500.0
    assert p_ned[2] < -450.0

    ecef_back = local_ned_to_ecef(
        p_ned,
        origin_lat_deg=ksfo_lat,
        origin_lon_deg=ksfo_lon,
        origin_alt_ellip_m=ksfo_h,
    )
    assert float(np.linalg.norm(ecef_back - ecef_t)) < 1e-4


def test_cubic_hermite_vs_causal_turning_aircraft_error_reduction() -> None:
    """Verify DR-15 Step 3: a_lat=2.5 m/s^2, 1 km, 3 s poll, f=2560 px -> causal ~28.8 px vs Hermite < 1.0 px (>25x)."""
    v_g = 60.0  # m/s GA cruise
    a_lat = 2.5  # m/s^2 standard-rate turn
    omega = a_lat / v_g  # rad/s
    radius_m = v_g / omega  # 1440 m turn radius
    dt_span = 3.0  # 3 s Tier-0A public API poll
    f_px = 2560.0  # 4K 1x smartphone focal length
    range_m = 1000.0

    def _circle_pos_vel(t: float) -> tuple[np.ndarray, np.ndarray]:
        theta = omega * t
        p = np.array([radius_m * math.sin(theta), radius_m * (1.0 - math.cos(theta)), -300.0], dtype=np.float64)
        v = np.array([v_g * math.cos(theta), v_g * math.sin(theta), 0.0], dtype=np.float64)
        return p, v

    p0, v0 = _circle_pos_vel(0.0)
    p1, v1 = _circle_pos_vel(dt_span)

    # 1. Causal straight-line extrapolation at t=3.0 s (turn rate unmeasured on 3 s API poll)
    p_causal_end = propagate_causal_turn_ned(
        p0,
        ground_speed_mps=v_g,
        track_rad=0.0,
        vertical_rate_mps=0.0,
        turn_rate_rad_s=0.0,
        dt_s=dt_span,
    )
    causal_err_m = float(np.linalg.norm(p_causal_end - p1))
    causal_err_px = (causal_err_m / range_m) * f_px

    # 2. Offline non-causal Cubic Hermite spline evaluated at worst-case midpoint t=1.5 s
    p_true_mid, _ = _circle_pos_vel(0.5 * dt_span)
    p_hermite_mid = interpolate_cubic_hermite_ned(
        p0, v0, p1, v1, t_mid_s=0.5 * dt_span, t_k_s=0.0, t_k1_s=dt_span
    )
    hermite_err_m = float(np.linalg.norm(p_hermite_mid - p_true_mid))
    hermite_err_px = (hermite_err_m / range_m) * f_px

    assert 27.0 <= causal_err_px <= 31.0
    assert hermite_err_px < 1.0
    assert (causal_err_px / max(hermite_err_px, 1e-9)) > 25.0

    # Also verify Tier-0B live causal turn propagation WITH known turn_rate tracks circle tightly over dt=0.5 s
    p_true_05, _ = _circle_pos_vel(0.5)
    p_turn_05 = propagate_causal_turn_ned(
        p0,
        ground_speed_mps=v_g,
        track_rad=0.0,
        turn_rate_rad_s=omega,
        dt_s=0.5,
    )
    assert float(np.linalg.norm(p_turn_05 - p_true_05)) < 0.05

    v_check = ned_velocity_from_adsb(ground_speed_kt=100.0, track_deg=90.0, vertical_rate_fpm=500.0)
    assert v_check[0] == pytest.approx(0.0, abs=1e-9)
    assert v_check[1] == pytest.approx(51.4444, abs=1e-4)
    assert v_check[2] == pytest.approx(-2.54, abs=1e-4)


def test_brown_conrady_projection_behind_camera_rejection_and_box_prior() -> None:
    """Verify Brown-Conrady distortion, Z_c <= 0 behind-camera rejection, and physical box span prior."""
    intr = CameraIntrinsics(
        fx=1800.0,
        fy=1800.0,
        cx=720.0,
        cy=540.0,
        k1=-0.12,
        k2=0.03,
        p1=0.001,
        p2=-0.001,
        k3=0.0,
        width_px=1440,
        height_px=1080,
    )
    # Camera looking due North: X_cam=+East, Y_cam=+Down, Z_cam=+North
    r_ned_to_cam = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Behind camera (N = -500 m -> Z_c = -500 <= 0) must return None
    assert project_ned_to_pixel([-500.0, 10.0, -100.0], r_ned_to_cam, intr) is None
    assert project_ned_to_pixel([0.0, 10.0, -100.0], r_ned_to_cam, intr) is None

    # In-front aircraft (N=1000, E=100, D=-200 -> 11.3 deg elevation)
    proj = project_ned_to_pixel([1000.0, 100.0, -200.0], r_ned_to_cam, intr)
    assert proj is not None
    assert proj.in_image_bounds is True
    assert proj.u_px > intr.cx
    assert proj.v_px < intr.cy
    assert proj.elevation_deg > 10.0

    # Barrel distortion (k1 < 0) pulls off-axis pixel strictly closer to principal point than pinhole
    intr_pinhole = CameraIntrinsics(fx=1800.0, fy=1800.0, cx=720.0, cy=540.0, width_px=1440, height_px=1080)
    proj_pinhole = project_ned_to_pixel([1000.0, 100.0, -200.0], r_ned_to_cam, intr_pinhole)
    assert proj_pinhole is not None
    r_dist = math.hypot(proj.u_px - intr.cx, proj.v_px - intr.cy)
    r_pin = math.hypot(proj_pinhole.u_px - intr.cx, proj_pinhole.v_px - intr.cy)
    assert r_dist < r_pin

    # Cessna 172 head-on box prior @ 1 km on IMX296 (f=1800 px): W=11.0 m -> 19.8 px
    c172 = AirframeDimensionsM(wingspan_m=11.0, length_m=8.28, tail_height_m=2.72)
    w_px, h_px = estimate_airframe_box_prior_px(
        slant_range_m=1000.0,
        intrinsics=intr,
        dimensions=c172,
        aspect_angle_rad=0.0,
    )
    assert w_px == pytest.approx(19.8, abs=0.1)
    assert h_px == pytest.approx(4.896, abs=0.1)


def test_legal_privacy_odbl_icao_stripping_and_ofac_sanctions_fail_closed() -> None:
    """Verify DR-15 fail-closed blocks: airplanes.live, non-commercial APIs, ODbL, <5 deg mask, InfiRay OFAC."""
    # airplanes.live hard-blocked (HTTP 403 closed Aug 2026)
    with pytest.raises(RealSkyComplianceError, match="airplanes.live"):
        assert_adsb_provider_allowed("airplanes.live", commercial_release=False, strip_upstream_json=True)

    # OpenSky & adsb.fi blocked on commercial releases
    for nc_prov in ("opensky", "adsb.fi"):
        with pytest.raises(RealSkyComplianceError):
            assert_adsb_provider_allowed(nc_prov, commercial_release=True, strip_upstream_json=True)
        # Permitted only when commercial_release=False (academic/personal eval)
        assert_adsb_provider_allowed(nc_prov, commercial_release=False, strip_upstream_json=True)

    # adsb.lol (ODbL 1.0) & flyitalyadsb (CC BY-SA 4.0) require strip_upstream_json=True
    for sa_prov in ("adsb.lol", "flyitalyadsb"):
        with pytest.raises(RealSkyComplianceError, match="strip_upstream_json"):
            assert_adsb_provider_allowed(sa_prov, commercial_release=True, strip_upstream_json=False)
        assert assert_adsb_provider_allowed(sa_prov, commercial_release=True, strip_upstream_json=True) == "TRANSIENT_CUE_STRIPPED_OK"

    # Local 1090ES RF allowed natively
    assert assert_adsb_provider_allowed("local_rf_1090es", commercial_release=True) == "LOCAL_RF_NATIVE_CC_BY_4"

    # OFAC SDN sanctions hard-block on Yantai IRay / InfiRay LWIR cores
    for bad_sensor in ("InfiRay Tiny1-C", "Yantai IRay P2 Pro", "Xinfrared P2S"):
        with pytest.raises(RealSkyComplianceError, match="OFAC SDN"):
            assert_thermal_sensor_sanctions_clean(bad_sensor)
    assert assert_thermal_sensor_sanctions_clean("Teledyne FLIR Boson+ 640") == "SANCTIONS_CLEAN"
    assert assert_thermal_sensor_sanctions_clean("Shanghai iRay Medical") == "SHANGHAI_IRAY_MEDICAL_NON_AFFILIATED"

    # Sky-only privacy mask (< 5.0 deg rejected) + PII/ICAO24/upstream JSON stripping
    raw_label = {
        "frame_idx": 42,
        "u_px": 1920.5,
        "v_px": 410.2,
        "w_px": 18.4,
        "h_px": 6.1,
        "slant_range_m": 1850.0,
        "icao24": "a1b2c3",
        "callsign": "UAL123",
        "tail_number": "N12345",
        "raw_adsb_json": {"hex": "a1b2c3"},
    }
    with pytest.raises(RealSkyComplianceError, match="5.0 deg"):
        sanitize_realsky_label_record(
            raw_label,
            provider="adsb.lol",
            ephemeral_track_id="trk_0001",
            aerodynamic_class="NARROWBODY_JET",
            elevation_mask_deg=3.5,
        )

    clean = sanitize_realsky_label_record(
        raw_label,
        provider="adsb.lol",
        ephemeral_track_id="trk_0001",
        aerodynamic_class="NARROWBODY_JET",
        elevation_mask_deg=5.0,
        commercial_release=True,
        strip_upstream_json=True,
    )
    assert clean["license"] == "CC-BY-4.0"
    assert clean["pii_stripped"] is True
    assert clean["upstream_adsb_stripped"] is True
    assert clean["stripped_field_count"] == 4
    for forbidden in ("icao24", "callsign", "tail_number", "raw_adsb_json"):
        assert forbidden not in clean


def test_load_camera_calibration_valid_and_fail_closed(tmp_path: Path) -> None:
    """Verify flat and OpenCV-nested camera_calib.json loading + strict finite/positive validation."""
    flat_path = tmp_path / "calib_flat.json"
    flat_path.write_text(
        json.dumps(
            {
                "fx": 2540.5,
                "fy": 2542.2,
                "cx": 1918.4,
                "cy": 1079.1,
                "k1": -0.082,
                "k2": 0.021,
                "p1": 0.0004,
                "p2": -0.0007,
                "k3": -0.003,
                "width": 3840,
                "height": 2160,
            }
        ),
        encoding="utf-8",
    )
    intr_flat = load_camera_calibration(flat_path)
    assert intr_flat.fx == pytest.approx(2540.5)
    assert intr_flat.fy == pytest.approx(2542.2)
    assert intr_flat.cx == pytest.approx(1918.4)
    assert intr_flat.cy == pytest.approx(1079.1)
    assert intr_flat.k1 == pytest.approx(-0.082)
    assert intr_flat.k2 == pytest.approx(0.021)
    assert intr_flat.p1 == pytest.approx(0.0004)
    assert intr_flat.p2 == pytest.approx(-0.0007)
    assert intr_flat.k3 == pytest.approx(-0.003)
    assert intr_flat.width_px == 3840
    assert intr_flat.height_px == 2160

    nested_path = tmp_path / "calib_opencv.json"
    nested_path.write_text(
        json.dumps(
            {
                "camera_matrix": [[1800.0, 0.0, 720.0], [0.0, 1805.0, 540.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [[-0.11, 0.03, 0.001, -0.002, 0.005]],
                "image_width": 1440,
                "image_height": 1080,
            }
        ),
        encoding="utf-8",
    )
    intr_nested = load_camera_calibration(nested_path)
    assert intr_nested.fx == pytest.approx(1800.0)
    assert intr_nested.fy == pytest.approx(1805.0)
    assert intr_nested.k3 == pytest.approx(0.005)
    assert intr_nested.width_px == 1440
    assert intr_nested.height_px == 1080

    # Fail-closed checks: non-positive fx, non-finite k1, negative height, principal point out of bounds, bool
    bad_cases = [
        {"fx": 0.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        {"fx": -500.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "k1": float("nan"), "width": 640, "height": 480},
        {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": -480},
        {"fx": 500.0, "fy": 500.0, "cx": 700.0, "cy": 240.0, "width": 640, "height": 480},
        {"fx": True, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
        {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640.5, "height": 480},
    ]
    for idx, payload in enumerate(bad_cases):
        bad_file = tmp_path / f"bad_{idx}.json"
        bad_file.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        with pytest.raises(ValueError):
            load_camera_calibration(bad_file)


def test_apply_sky_elevation_mask_zeros_below_5_deg_with_brown_conrady_distortion() -> None:
    """Verify deterministic >= +5.0 deg pre-encode sky mask zeroes every below-horizon pixel (`max == 0`)."""
    calib = CameraIntrinsics(
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
        k1=-0.10,
        k2=0.025,
        p1=0.001,
        p2=-0.001,
        k3=-0.004,
        width_px=640,
        height_px=480,
    )
    # Camera facing North pitched up 10 deg -> bottom portion of frame is below +5 deg horizon
    pitch = math.radians(10.0)
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)], dtype=np.float64)
    x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    r_ned_cam = np.vstack([x_axis, y_axis, z_axis])

    # Reject < 5.0 deg mask fail-closed
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    with pytest.raises(RealSkyComplianceError, match="5.0"):
        apply_sky_elevation_mask(frame, calib, r_ned_cam, min_elev_deg=4.9)

    sky_mask = compute_sky_elevation_mask(calib, r_ned_cam, min_elev_deg=5.0)
    assert sky_mask.shape == (480, 640)
    assert np.any(sky_mask)
    assert np.any(~sky_mask)

    masked = apply_sky_elevation_mask(frame, calib, r_ned_cam, min_elev_deg=5.0)
    # Caller buffer must not be mutated
    assert int(frame.min()) == 210
    # Every pixel below +5 deg must be strictly 0 (`max == 0`)
    assert int(masked[~sky_mask].max()) == 0
    # Every sky pixel >= +5 deg must be preserved untouched
    assert int(masked[sky_mask].min()) == 210

    # Cross-check exact 3D NED projection consistency against project_ned_to_pixel
    # Target at +3.5 deg elevation -> MUST land on a masked (0) pixel
    p_low = np.array([1000.0, 40.0, -1000.0 * math.tan(math.radians(3.5))], dtype=np.float64)
    proj_low = project_ned_to_pixel(p_low, r_ned_cam, calib)
    assert proj_low is not None and proj_low.in_image_bounds
    assert int(masked[int(round(proj_low.v_px)), int(round(proj_low.u_px))].max()) == 0

    # Target at +15.0 deg elevation -> MUST land on an unmasked sky pixel (210)
    p_high = np.array([1000.0, -80.0, -1000.0 * math.tan(math.radians(15.0))], dtype=np.float64)
    proj_high = project_ned_to_pixel(p_high, r_ned_cam, calib)
    assert proj_high is not None and proj_high.in_image_bounds
    assert int(masked[int(round(proj_high.v_px)), int(round(proj_high.u_px))].min()) == 210


def test_contrast_centroid_snap_locks_bright_and_dark_dots_and_rejects_noise() -> None:
    """Verify dual-polarity top-hat/bottom-hat snap (< 1.5 px error) and honest `snapped=False` on noise/mask edge."""
    rng = np.random.default_rng(12345)
    h, w = 240, 320
    grad = np.linspace(130.0, 175.0, h, dtype=np.float32)[:, np.newaxis]

    # 1. Bright sunlit aircraft dot shifted (+12.0, -8.0) px from prior inside 3-sigma window
    prior_u, prior_v = 160.0, 110.0
    true_u_b, true_v_b = 172.0, 102.0
    sky_b = np.clip(grad + rng.normal(0.0, 4.0, (h, w)).astype(np.float32), 0.0, 255.0)
    yy, xx = np.mgrid[0:h, 0:w]
    sky_b = np.clip(
        sky_b + 55.0 * np.exp(-((xx - true_u_b) ** 2 + (yy - true_v_b) ** 2) / (2.0 * 1.4 * 1.4)),
        0.0,
        255.0,
    )
    frame_b = np.repeat(sky_b.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
    res_b = snap_target_contrast_centroid(frame_b, prior_u, prior_v, sigma_u_px=15.0, sigma_v_px=15.0, min_snr=4.5)
    assert res_b.snapped is True
    assert res_b.polarity == "BRIGHT_TOPHAT"
    assert res_b.snr >= 4.5
    assert math.hypot(res_b.u_px - true_u_b, res_b.v_px - true_v_b) < 1.5

    # 2. Dark silhouette aircraft dot shifted (-10.5, +11.5) px from prior inside 3-sigma window
    true_u_d, true_v_d = 149.5, 121.5
    sky_d = np.clip(grad + 35.0 + rng.normal(0.0, 4.0, (h, w)).astype(np.float32), 0.0, 255.0)
    sky_d = np.clip(
        sky_d - 60.0 * np.exp(-((xx - true_u_d) ** 2 + (yy - true_v_d) ** 2) / (2.0 * 1.5 * 1.5)),
        0.0,
        255.0,
    )
    frame_d = np.repeat(sky_d.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
    res_d = snap_target_contrast_centroid(frame_d, prior_u, prior_v, sigma_u_px=15.0, sigma_v_px=15.0, min_snr=4.5)
    assert res_d.snapped is True
    assert res_d.polarity == "DARK_BOTTOMHAT"
    assert res_d.snr >= 4.5
    assert math.hypot(res_d.u_px - true_u_d, res_d.v_px - true_v_d) < 1.5

    # 3. Pure gradient + Gaussian noise across 10 seeds: MUST NEVER hallucinate a snap!
    for seed in range(10):
        r_local = np.random.default_rng(700 + seed)
        sky_n = np.clip(grad + r_local.normal(0.0, 5.0, (h, w)).astype(np.float32), 0.0, 255.0)
        frame_n = np.repeat(sky_n.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        res_n = snap_target_contrast_centroid(frame_n, prior_u, prior_v, sigma_u_px=15.0, sigma_v_px=15.0, min_snr=4.5)
        assert res_n.snapped is False
        assert res_n.polarity == "NONE"
        assert res_n.u_px == pytest.approx(prior_u)
        assert res_n.v_px == pytest.approx(prior_v)
        assert res_n.du_px == 0.0 and res_n.dv_px == 0.0

    # 4. Bright decoy strictly OUTSIDE 3-sigma (at 4.2*sigma = 63 px away) must be ignored
    sky_far = np.full((h, w), 150, dtype=np.uint8)
    sky_far[108:113, 222:227] = 250  # du = +64 px > 3*15 = 45 px
    res_far = snap_target_contrast_centroid(sky_far, prior_u, prior_v, sigma_u_px=15.0, sigma_v_px=15.0, min_snr=4.5)
    assert res_far.snapped is False

    # 5. Artificial >= +5 deg sky-mask cut-to-black step edge inside search window must NOT trigger false snap
    sky_edge = np.full((h, w, 3), 185, dtype=np.uint8)
    edge_mask = np.ones((h, w), dtype=bool)
    edge_mask[125:, :] = False
    sky_edge[~edge_mask] = 0
    res_edge = snap_target_contrast_centroid(
        sky_edge, prior_u, prior_v, sigma_u_px=15.0, sigma_v_px=15.0, min_snr=4.5, sky_mask=edge_mask
    )
    assert res_edge.snapped is False


def test_ed25519_rfc8032_test_vectors_and_manifest_tamper_rejection() -> None:
    """Verify RFC 8032 §7.1 official Ed25519 vectors bit-exact + manifest signature tamper rejection."""
    vectors = [
        # RFC 8032 Section 7.1 TEST 1 (empty message)
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "",
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        ),
        # RFC 8032 Section 7.1 TEST 2 (1-byte message 0x72)
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "72",
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
            "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        ),
        # RFC 8032 Section 7.1 TEST 3 (2-byte message 0xaf82)
        (
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            "af82",
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
            "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
        ),
    ]
    for seed_hex, expected_pub_hex, msg_hex, expected_sig_hex in vectors:
        seed = bytes.fromhex(seed_hex)
        msg = bytes.fromhex(msg_hex)
        pub = ed25519_public_key_from_seed(seed)
        sig = ed25519_sign(seed, msg)
        assert pub.hex() == expected_pub_hex
        assert sig.hex() == expected_sig_hex
        assert ed25519_verify(pub, msg, sig) is True
        # Flip 1 bit in signature -> MUST fail
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        assert ed25519_verify(pub, msg, bytes(bad_sig)) is False
        # Wrong message -> MUST fail
        assert ed25519_verify(pub, msg + b"\x00", sig) is False

    # Manifest sign + verify + tamper rejection
    manifest_body = {
        "schema": "openxvs.realsky.shard_manifest/1",
        "license": "CDLA-Permissive-2.0",
        "record_count": 7,
    }
    digest = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    shard_manifest = dict(manifest_body, manifest_sha256=digest)
    sig_doc = sign_manifest_sha256(digest, private_seed=bytes.fromhex(vectors[0][0]))
    assert sig_doc["algorithm"] == "Ed25519"
    assert len(sig_doc["public_key_hex"]) == 64
    assert len(sig_doc["signature_hex"]) == 128
    assert verify_manifest_signature(shard_manifest, sig_doc) is True

    # Tampering manifest body -> MUST reject
    tampered_body = dict(shard_manifest, record_count=8)
    assert verify_manifest_signature(tampered_body, sig_doc) is False

    # Tampering signature hex -> MUST reject
    tampered_sig = dict(sig_doc, signature_hex=("00" if sig_doc["signature_hex"][:2] != "00" else "ff") + sig_doc["signature_hex"][2:])
    assert verify_manifest_signature(shard_manifest, tampered_sig) is False

    # Swapping public key -> MUST reject
    swapped_pub = dict(sig_doc, public_key_hex=vectors[1][1])
    assert verify_manifest_signature(shard_manifest, swapped_pub) is False
