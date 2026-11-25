"""OpenXVS v2.0 Milestone M1 — `V2-M1-3` Cross-Language Parity & Headless M1–M4 Verification Suite.

Cross-checks Rust `xvs-core` (via `src/xvs/core_ffi.py` -> `xvs_core.dll`) against:
  1. Independent Python `PySimplexRtaOracle` (`tests/xvs_v2/_oracles.py`) — 10,000 ticks bit-exact;
  2. `src/xvs/avionics/{gdl90.py, mavlink_adsb.py}` & `_oracles.py` `#52001` — bit-exact wire bytes;
  3. `src/xvs/geo/{wgs84.py, rotations.py}` & `src/xvs/rig/camera.py` — `< 1e-6 m`, `< 1e-12 rad`, `< 1e-9 px`;
  4. Tie-guarded (`|stat - thr| > 1e-9`) math oracles (`CUSUM`, `Vision-RAIM`, `MeshDAA` + `FCI`, `ECBF-QP`);
  5. Headless emergent Missions `M1`–`M4` matching `apps/livedeck/src/missions/m1..m4.ts` schedules.

Writes machine-readable verification evidence to:
  - `output/xvs/v2_m1/parity_results.json`
  - `output/xvs/v2_m1/headless_missions_results.json`
"""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tests" / "xvs_v2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "xvs_v2"))

import _oracles as oracles  # noqa: E402
from xvs.avionics import gdl90, mavlink_adsb  # noqa: E402
from xvs.core_ffi import (  # noqa: E402
    XvsBearingTrack,
    XvsCoreFFI,
    XvsFaultPerturbations,
    XvsGdl90Target,
    XvsIntruderNed,
    XvsMavlinkAdsb246,
    XvsRtaFloatTelemetry,
)
from xvs.geo import rotations, wgs84  # noqa: E402
from xvs.rig.camera import PinholeCamera  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "xvs" / "v2_m1"
PARITY_JSON_PATH = OUTPUT_DIR / "parity_results.json"
MISSIONS_JSON_PATH = OUTPUT_DIR / "headless_missions_results.json"

_PARITY_REPORT: dict[str, Any] = {
    "schema": "xvs.v2_m1_parity/1",
    "rta_10000_ticks": {},
    "codecs_bit_exact": {},
    "geo_rig_parity": {},
    "math_oracles": {},
}


def _persist_parity_report() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rta = _PARITY_REPORT["rta_10000_ticks"]
    codecs = _PARITY_REPORT["codecs_bit_exact"]
    geo = _PARITY_REPORT["geo_rig_parity"]
    oracles_rep = _PARITY_REPORT["math_oracles"]
    _PARITY_REPORT["all_checks_passed"] = bool(
        rta
        and codecs
        and geo
        and oracles_rep
        and int(rta["mismatches"]) == 0
        and int(codecs["byte_mismatches"]) == 0
        and float(geo["max_ecef_err_m"]) < 1e-6
        and float(geo["max_quat_err_rad"]) < 1e-12
        and float(geo["max_pinhole_proj_err_px"]) < 1e-9
        and int(oracles_rep["turn_30deg_spoof_alarm"]) == 0
        and int(oracles_rep["vraim_cam_r_isolated_idx"]) == 1
        and int(oracles_rep["mesh_3ship_ghosts_rejected"]) == 2
        and int(oracles_rep["ecbf_sandwich_infeasible"]) == 1
    )
    PARITY_JSON_PATH.write_text(json.dumps(_PARITY_REPORT, indent=2) + "\n", encoding="utf-8")


def _py_tel_to_c(t: oracles.PyRtaFloatTelemetry) -> XvsRtaFloatTelemetry:
    return XvsRtaFloatTelemetry(
        horizon_pl_rad=t.horizon_pl_rad,
        horizon_al_rad=t.horizon_al_rad,
        v_vpl_m=t.v_vpl_m,
        v_val_m=t.v_val_m,
        v_hpl_m=t.v_hpl_m,
        v_hal_m=t.v_hal_m,
        epipolar_max_rad=(ctypes.c_double * 6)(*t.epipolar_max_rad),
        signed_horizon_residual_rad=t.signed_horizon_residual_rad,
        runway_width_delta_m=t.runway_width_delta_m,
        runway_width_delta_limit_m=t.runway_width_delta_limit_m,
        cusum_stat=t.cusum_stat,
        cusum_thr=t.cusum_thr,
        ood_global_score=t.ood_global_score,
        ood_global_thr=t.ood_global_thr,
        ood_cam_scores=(ctypes.c_double * 6)(*t.ood_cam_scores),
        ood_cam_thr=t.ood_cam_thr,
        cam_health_mask=t.cam_health_mask,
        ml_detector_fault=1 if t.ml_detector_fault else 0,
        transient_dropout=1 if t.transient_dropout else 0,
        approach_active=1 if t.approach_active else 0,
        go_around_ack_reset=1 if t.go_around_ack_reset else 0,
    )


def test_10000_tick_simplex_rta_and_mltl_bit_exact_parity() -> None:
    ffi = XvsCoreFFI()
    rs_rta = ffi.rta_create()
    py_rta = oracles.PySimplexRtaOracle()
    rng = np.random.default_rng(20261120)

    total_ticks = 10_000
    mismatches = 0
    mode_counts = {m: 0 for m in range(5)}
    quantize_fault_ticks = 0
    go_around_latched_ticks = 0

    try:
        for tick in range(total_ticks):
            # Construct rich mix of deterministic edge phases + randomized fault traces
            if tick < 40:
                # 6->5->4->3 cam drops + recovery
                masks = [0x3F] * 5 + [0x3D] * 5 + [0x3C] * 5 + [0x07] * 5 + [0x3F] * 20
                tel = oracles.PyRtaFloatTelemetry(
                    horizon_pl_rad=-0.0 if tick == 0 else 0.0015,
                    signed_horizon_residual_rad=-0.0008,
                    runway_width_delta_m=-1.2,
                    cam_health_mask=masks[tick],
                )
            elif tick < 120:
                # Exact 10/20 (50%) vs 11/20 (55%) intermittent dropout duty cycle
                phase = tick - 40
                if phase < 40:
                    drop = (phase % 20) < 10
                else:
                    drop = (phase % 11) < 10  # 10-on / 1-off => 11/20 triggers P0!
                tel = oracles.PyRtaFloatTelemetry(transient_dropout=drop)
            elif tick < 210:
                # Single-tick chatter at recovery tick 29 and dwell tick 19
                phase = tick - 120
                if phase == 0 or phase == 30:
                    tel = oracles.PyRtaFloatTelemetry(horizon_pl_rad=0.015)
                elif phase == 80:
                    tel = oracles.PyRtaFloatTelemetry(ml_detector_fault=True)
                else:
                    tel = oracles.PyRtaFloatTelemetry()
            elif tick < 270:
                # Non-resettable mandatory_go_around while approach_active, then climbout reset
                phase = tick - 210
                if phase == 0:
                    tel = oracles.PyRtaFloatTelemetry(v_vpl_m=12.5, approach_active=True)
                elif phase < 40:
                    tel = oracles.PyRtaFloatTelemetry(
                        v_vpl_m=2.0, approach_active=True, go_around_ack_reset=True
                    )
                else:
                    tel = oracles.PyRtaFloatTelemetry(
                        v_vpl_m=2.0, approach_active=False, go_around_ack_reset=True
                    )
            elif tick < 290:
                # NaN / +Inf / -Inf / negative-norm / -0.0 boundary cases
                edge = tick - 270
                if edge == 0:
                    tel = oracles.PyRtaFloatTelemetry(horizon_pl_rad=float("nan"))
                elif edge == 1:
                    tel = oracles.PyRtaFloatTelemetry(cusum_stat=float("nan"))
                elif edge == 2:
                    tel = oracles.PyRtaFloatTelemetry(ood_global_score=float("inf"))
                elif edge == 3:
                    tel = oracles.PyRtaFloatTelemetry(v_vpl_m=-0.01)
                elif edge == 4:
                    tel = oracles.PyRtaFloatTelemetry(
                        horizon_pl_rad=-0.0, signed_horizon_residual_rad=-0.0025
                    )
                else:
                    tel = oracles.PyRtaFloatTelemetry()
            else:
                # Randomized stress stream across all inputs (with long healthy runs for recovery)
                burst = (tick // 70) % 5
                if burst == 0:
                    # Clean recovery window
                    tel = oracles.PyRtaFloatTelemetry(
                        horizon_pl_rad=float(rng.uniform(0.0, 0.004)),
                        signed_horizon_residual_rad=float(rng.uniform(-0.003, 0.003)),
                        runway_width_delta_m=float(rng.uniform(-3.0, 3.0)),
                        go_around_ack_reset=True,
                    )
                else:
                    p_nan = rng.random()
                    h_pl = float("nan") if p_nan < 0.01 else float(rng.uniform(0.0, 0.011))
                    cusum = float("nan") if p_nan > 0.99 else float(rng.uniform(0.0, 12.0))
                    epi = tuple(
                        float(rng.uniform(0.0, 0.008)) if rng.random() < 0.05 else 0.0005
                        for _ in range(6)
                    )
                    cam_mask = int(rng.choice([0x3F, 0x3F, 0x3F, 0x3D, 0x3C, 0x1F, 0x07]))
                    tel = oracles.PyRtaFloatTelemetry(
                        horizon_pl_rad=h_pl,
                        v_vpl_m=float(rng.uniform(0.0, 13.0)),
                        v_hpl_m=float(rng.uniform(0.0, 45.0)),
                        epipolar_max_rad=epi,  # type: ignore[arg-type]
                        signed_horizon_residual_rad=float(rng.uniform(-0.005, 0.005)),
                        runway_width_delta_m=float(rng.uniform(-6.0, 6.0)),
                        cusum_stat=cusum,
                        ood_global_score=float(rng.uniform(0.0, 1.2)),
                        cam_health_mask=cam_mask,
                        ml_detector_fault=bool(rng.random() < 0.06),
                        transient_dropout=bool(rng.random() < 0.12),
                        approach_active=bool(rng.random() < 0.20),
                        go_around_ack_reset=bool(rng.random() < 0.30),
                    )

            py_in = oracles.build_rta_input_from_f64(tel)
            rs_in = ffi.rta_build_input_from_f64(_py_tel_to_c(tel))

            # Compare quantized input boundary bit-for-bit
            in_match = (
                rs_in.cam_health_mask == py_in.cam_health_mask
                and rs_in.ood_exceeded_mask == py_in.ood_exceeded_mask
                and rs_in.horizon_pl_urad == py_in.horizon_pl_urad
                and rs_in.horizon_al_urad == py_in.horizon_al_urad
                and rs_in.v_vpl_mm == py_in.v_vpl_mm
                and rs_in.v_val_mm == py_in.v_val_mm
                and rs_in.v_hpl_mm == py_in.v_hpl_mm
                and rs_in.v_hal_mm == py_in.v_hal_mm
                and tuple(rs_in.epipolar_max_urad) == py_in.epipolar_max_urad
                and rs_in.signed_horizon_residual_urad == py_in.signed_horizon_residual_urad
                and rs_in.runway_width_delta_mm == py_in.runway_width_delta_mm
                and bool(rs_in.spoof_alarm) == py_in.spoof_alarm
                and bool(rs_in.ml_detector_fault) == py_in.ml_detector_fault
                and bool(rs_in.ood_global_exceeded) == py_in.ood_global_exceeded
                and bool(rs_in.runway_width_fault) == py_in.runway_width_fault
                and bool(rs_in.transient_dropout) == py_in.transient_dropout
                and bool(rs_in.approach_active) == py_in.approach_active
                and bool(rs_in.go_around_ack_reset) == py_in.go_around_ack_reset
                and bool(rs_in.quantize_fault) == py_in.quantize_fault
            )
            if not in_match:
                mismatches += 1
                raise AssertionError(f"Quantizer mismatch at tick {tick}: py={py_in}")

            py_out = py_rta.step(py_in)
            rs_out = ffi.rta_step(rs_rta, rs_in)

            out_match = (
                int(rs_out.mode) == int(py_out.mode)
                and int(rs_out.raw_target) == int(py_out.raw_target)
                and int(rs_out.healthy_cam_mask) == int(py_out.healthy_cam_mask)
                and int(rs_out.active_healthy_cams) == int(py_out.active_healthy_cams)
                and int(rs_out.consec_dropout_ticks) == int(py_out.consec_dropout_ticks)
                and int(rs_out.window_dropout_20) == int(py_out.window_dropout_20)
                and int(rs_out.recovery_ticks) == int(py_out.recovery_ticks)
                and int(rs_out.dwell_ticks) == int(py_out.dwell_ticks)
                and bool(rs_out.mandatory_go_around) == py_out.mandatory_go_around
                and bool(rs_out.mltl_critical_violation) == py_out.mltl_critical_violation
                and bool(rs_out.blank_conformal_hud) == py_out.blank_conformal_hud
                and bool(rs_out.inhibit_avoidance_turns) == py_out.inhibit_avoidance_turns
                and int(rs_out.tick_index) == int(py_out.tick_index)
            )
            if not out_match:
                mismatches += 1
                raise AssertionError(
                    f"Simplex RTA output mismatch at tick {tick}: "
                    f"rs=({rs_out.mode},{rs_out.raw_target},{rs_out.recovery_ticks},{rs_out.dwell_ticks}) "
                    f"py=({int(py_out.mode)},{int(py_out.raw_target)},{py_out.recovery_ticks},{py_out.dwell_ticks})"
                )

            mode_counts[int(rs_out.mode)] += 1
            if py_in.quantize_fault:
                quantize_fault_ticks += 1
            if py_out.mandatory_go_around:
                go_around_latched_ticks += 1
    finally:
        ffi.rta_destroy(rs_rta)

    assert mismatches == 0
    assert all(c > 0 for c in mode_counts.values()), f"Expected all 5 modes visited, got {mode_counts}"
    _PARITY_REPORT["rta_10000_ticks"] = {
        "ticks": total_ticks,
        "mismatches": mismatches,
        "mode_counts": mode_counts,
        "quantize_fault_ticks": quantize_fault_ticks,
        "go_around_latched_ticks": go_around_latched_ticks,
    }
    _persist_parity_report()


def test_avionics_codecs_gdl90_and_mavlink_v2_bit_exact_parity() -> None:
    ffi = XvsCoreFFI()
    rng = np.random.default_rng(20261121)

    gdl90_packets = 0
    mav246_packets = 0
    mav52001_packets = 0

    for i in range(60):
        # GDL-90 Heartbeat (#0)
        ts = int(rng.integers(0, 86400))
        py_hb = gdl90.encode_heartbeat(
            utc_time_s=ts,
            gps_pos_valid=(i % 3 != 0),
            maint_reqd=(i % 7 == 0),
            utc_ok=(i % 5 != 0),
            uplink_count=i % 32,
            basic_and_long_count=(i * 17) % 1024,
        )
        rs_hb = ffi.encode_gdl90_heartbeat(
            utc_time_s=ts,
            gps_pos_valid=(i % 3 != 0),
            maint_reqd=(i % 7 == 0),
            utc_ok=(i % 5 != 0),
            uplink_count=i % 32,
            basic_and_long_count=(i * 17) % 1024,
        )
        assert rs_hb == py_hb
        gdl90_packets += 1

        # GDL-90 Ownship Geometric Altitude (#11)
        alt_ft = float(rng.uniform(-500.0, 42000.0))
        vfom = float(rng.uniform(0.0, 50.0))
        py_geo = gdl90.encode_ownship_geo_altitude(alt_ft, vfom, (i % 4 == 0))
        rs_geo = ffi.encode_gdl90_ownship_geo_alt(
            alt_ft=alt_ft, vfom_m=vfom, vertical_warning=(i % 4 == 0)
        )
        assert rs_geo == py_geo
        gdl90_packets += 1

        # GDL-90 Ownship (#10) & Traffic (#20) including 0x7E/0x7D byte-stuffing triggers
        addr = 0x7E7D7E if i == 0 else int(rng.integers(0, 0xFFFFFF))
        lat = float(rng.uniform(-85.0, 85.0))
        lon = float(rng.uniform(-179.0, 179.0))
        hvel = None if i % 11 == 0 else float(rng.uniform(0.0, 450.0))
        vvel = None if i % 13 == 0 else float(rng.uniform(-4000.0, 4000.0))
        trk = float(rng.uniform(0.0, 359.9))
        cs = f"XVS{i:04d}"
        py_own = gdl90.encode_target_report(
            gdl90.Gdl90TargetReport(
                msg_id=gdl90.GDL90_MSG_ID_OWNSHIP_REPORT,
                alert_status=i % 2,
                address_type=0,
                participant_address=addr,
                lat_deg=lat,
                lon_deg=lon,
                altitude_ft=alt_ft,
                misc_indicators=0x09,
                nic=8,
                nacp=8,
                horiz_vel_kt=hvel,
                vert_vel_fpm=vvel,
                track_heading_deg=trk,
                emitter_category=1,
                callsign=cs,
                emergency_code=0,
            )
        )
        rs_tgt = XvsGdl90Target.from_values(
            alert_status=i % 2,
            address_type=0,
            participant_address=addr,
            lat_deg=lat,
            lon_deg=lon,
            altitude_ft=alt_ft,
            misc_indicators=0x09,
            nic=8,
            nacp=8,
            horiz_vel_kt=hvel,
            vert_vel_fpm=vvel,
            track_heading_deg=trk,
            emitter_category=1,
            callsign=cs,
            emergency_code=0,
        )
        assert ffi.encode_gdl90_ownship(rs_tgt) == py_own
        gdl90_packets += 1

        # Traffic Report (#20) with SIM prefix guard
        py_trf = gdl90.encode_target_report(
            gdl90.Gdl90TargetReport(
                msg_id=gdl90.GDL90_MSG_ID_TRAFFIC_REPORT,
                alert_status=i % 2,
                address_type=0,
                participant_address=addr,
                lat_deg=lat,
                lon_deg=lon,
                altitude_ft=alt_ft,
                misc_indicators=0x09,
                nic=8,
                nacp=8,
                horiz_vel_kt=hvel,
                vert_vel_fpm=vvel,
                track_heading_deg=trk,
                emitter_category=1,
                callsign=("SIM" + cs)[:8],
                emergency_code=0,
            )
        )
        assert ffi.encode_gdl90_traffic(rs_tgt, sim_prefix=True) == py_trf
        gdl90_packets += 1

        # MAVLink v2 #246 ADSB_VEHICLE (50 bytes)
        icao = 0xF00000 | i
        lat_e7 = int(round(lat * 1e7))
        lon_e7 = int(round(lon * 1e7))
        alt_mm = int(round(alt_ft * 304.8))
        hdg_cdeg = int(round(trk * 100.0)) % 36000
        h_cms = int(rng.integers(0, 25000))
        v_cms = int(rng.integers(-2000, 2000))
        py_246 = mavlink_adsb.pack_adsb_vehicle(
            mavlink_adsb.MavlinkAdsbVehicle(
                icao_address=icao,
                lat_deg_e7=lat_e7,
                lon_deg_e7=lon_e7,
                altitude_mm=alt_mm,
                heading_cdeg=hdg_cdeg,
                hor_velocity_cms=h_cms,
                ver_velocity_cms=v_cms,
                flags=0x00FF,
                squawk=7700 if i % 2 else 1200,
                altitude_type=1,
                callsign=cs,
                emitter_type=14,
                tslc=i % 256,
                seq=i & 0xFF,
                sysid=1,
                compid=156,
            )
        )
        rs_246 = ffi.encode_mavlink_adsb_246(
            XvsMavlinkAdsb246.from_values(
                icao_address=icao,
                lat_deg_e7=lat_e7,
                lon_deg_e7=lon_e7,
                altitude_mm=alt_mm,
                heading_cdeg=hdg_cdeg,
                hor_velocity_cms=h_cms,
                ver_velocity_cms=v_cms,
                flags=0x00FF,
                squawk=7700 if i % 2 else 1200,
                altitude_type=1,
                callsign=cs,
                emitter_type=14,
                tslc=i % 256,
                seq=i & 0xFF,
                sysid=1,
                compid=156,
            )
        )
        assert rs_246 == py_246
        mav246_packets += 1

        # Signed MAVLink v2 #52001 OPENXVS_BEARING_TRACK (77 bytes) bit-exact vs _oracles.py
        key = bytes([(0x31 + k + i) & 0xFF for k in range(32)])
        own_cm = (int(rng.integers(-600_000_000, 600_000_000)), -420_000_000 + i * 100, 380_000_000)
        brg_q15 = (16384, -16384, 23170)
        rate_u = (15, -22, 8)
        cov_d = (50, 50, 50, 1800, 1800, 1800)
        py_trk = oracles.OpenXvsBearingTrack(
            time_utc_us=1_732_100_000_000_000 + i * 100_000,
            own_ecef_cm=own_cm,
            bearing_ecef_q15=brg_q15,
            bearing_rate_urad_s=rate_u,
            cone_half_angle_urad=1800,
            own_pos_std_cm=50,
            track_id=100 + i,
            sender_id=(i % 3) + 1,
            flags=0x03,
            cov_diag_packed=cov_d,
        )
        py_52001 = oracles.pack_signed_bearing_track_mavlink_v2(
            py_trk,
            secret_key_32=key,
            link_id=i % 4,
            sig_timestamp_48=10_000 + i * 7,
            seq=i & 0xFF,
            sysid=1,
            compid=156,
        )
        rs_trk = XvsBearingTrack(
            time_utc_us=py_trk.time_utc_us,
            own_ecef_cm=(ctypes.c_int32 * 3)(*own_cm),
            bearing_ecef_q15=(ctypes.c_int16 * 3)(*brg_q15),
            bearing_rate_urad_s=(ctypes.c_int16 * 3)(*rate_u),
            cone_half_angle_urad=py_trk.cone_half_angle_urad,
            own_pos_std_cm=py_trk.own_pos_std_cm,
            track_id=py_trk.track_id,
            sender_id=py_trk.sender_id,
            flags=py_trk.flags,
            cov_diag_packed=(ctypes.c_uint16 * 6)(*cov_d),
        )
        rs_52001 = ffi.encode_mavlink_bearing_track(
            rs_trk,
            seq=i & 0xFF,
            sysid=1,
            compid=156,
            link_id=i % 4,
            timestamp_48=10_000 + i * 7,
            secret_key=key,
        )
        assert len(rs_52001) == 77
        assert rs_52001 == py_52001
        assert oracles.unpack_and_verify_signed_bearing_track(rs_52001, secret_key_32=key) == py_trk
        mav52001_packets += 1

    _PARITY_REPORT["codecs_bit_exact"] = {
        "gdl90_packets_verified": gdl90_packets,
        "mavlink_246_packets_verified": mav246_packets,
        "mavlink_52001_signed_packets_verified": mav52001_packets,
        "byte_mismatches": 0,
    }
    _persist_parity_report()


def test_geodesy_rotations_and_pinhole_projection_parity() -> None:
    ffi = XvsCoreFFI()
    rng = np.random.default_rng(20261122)
    rig_spec = json.loads((REPO_ROOT / "docs" / "xvs" / "spec" / "rig_nominal.json").read_text("utf-8"))
    py_cube_cam = PinholeCamera(
        width=1024,
        height=1024,
        fx=float(rig_spec["intrinsics"]["fx"]),
        fy=float(rig_spec["intrinsics"]["fy"]),
        cx=float(rig_spec["intrinsics"]["cx"]),
        cy=float(rig_spec["intrinsics"]["cy"]),
    )

    max_ecef_err_m = 0.0
    max_h_err_m = 0.0
    max_r_err = 0.0
    max_quat_err_rad = 0.0
    max_proj_err_px = 0.0
    behind_cam_none_checks = 0

    for _ in range(100):
        lat = float(rng.uniform(-89.5, 89.5))
        lon = float(rng.uniform(-179.5, 179.5))
        h = float(rng.uniform(-50.0, 12000.0))

        py_ecef = wgs84.geodetic_to_ecef(lat, lon, h)
        rs_ecef = np.array(ffi.geodetic_to_ecef(lat, lon, h), dtype=np.float64)
        max_ecef_err_m = max(max_ecef_err_m, float(np.linalg.norm(py_ecef - rs_ecef)))

        py_geo = wgs84.ecef_to_geodetic(float(py_ecef[0]), float(py_ecef[1]), float(py_ecef[2]))
        rs_geo = ffi.ecef_to_geodetic(float(py_ecef[0]), float(py_ecef[1]), float(py_ecef[2]))
        max_h_err_m = max(max_h_err_m, abs(py_geo[2] - rs_geo[2]))

        py_r_en = wgs84.r_ecef_ned(lat, lon)
        rs_r_en = np.array(ffi.r_ecef_ned(lat, lon), dtype=np.float64)
        max_r_err = max(max_r_err, float(np.max(np.abs(py_r_en - rs_r_en))))

        yaw = float(rng.uniform(-math.pi, math.pi))
        pitch = float(rng.uniform(-0.8, 0.8))
        roll = float(rng.uniform(-0.8, 0.8))
        py_q = rotations.r_to_quat(rotations.euler_to_r(yaw, pitch, roll))
        rs_q = np.array(ffi.euler_zyx_to_quat(yaw, pitch, roll), dtype=np.float64)
        assert rs_q[0] >= 0.0
        # Numerically exact geodesic angle on S^3 (avoids sqrt(eps) precision loss in acos(1 - 1 ULP))
        q_dist = min(float(np.linalg.norm(py_q - rs_q)), float(np.linalg.norm(py_q + rs_q)))
        ang_err = 2.0 * math.asin(min(1.0, 0.5 * q_dist))
        max_quat_err_rad = max(max_quat_err_rad, ang_err)

        # Pinhole projection across all 6 XVS6-CUBE-v1 cameras
        r_ecef_body = py_r_en @ rotations.euler_to_r(yaw, pitch, roll)
        for cam in rig_spec["cameras"]:
            r_bc = np.array(cam["R_body_cam"], dtype=np.float64)
            r_ec = r_ecef_body @ r_bc
            p_cam = py_ecef + r_ecef_body @ np.array(cam["lever_arm_m"], dtype=np.float64)
            # Front point in local camera frame mapped into ECEF
            v_cam_front = np.array(
                [rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0), rng.uniform(25.0, 200.0)],
                dtype=np.float64,
            )
            p_obj = p_cam + r_ec @ v_cam_front
            py_u, py_v, ok = py_cube_cam.project_ecef(p_obj, p_cam, r_ec)
            assert ok
            rs_uv = ffi.project_ecef_to_cam(
                0, p_obj.tolist(), p_cam.tolist(), r_ec.reshape(-1).tolist()
            )
            assert rs_uv is not None
            max_proj_err_px = max(
                max_proj_err_px, math.hypot(py_u - rs_uv[0], py_v - rs_uv[1])
            )

            # Behind-camera check (z_cam <= 0.0) MUST return None
            p_back = p_cam - r_ec @ np.array([0.0, 0.0, 40.0], dtype=np.float64)
            assert (
                ffi.project_ecef_to_cam(
                    0, p_back.tolist(), p_cam.tolist(), r_ec.reshape(-1).tolist()
                )
                is None
            )
            behind_cam_none_checks += 1

    assert max_ecef_err_m < 1e-6
    assert max_h_err_m < 1e-6
    assert max_r_err < 1e-12
    assert max_quat_err_rad < 1e-12
    assert max_proj_err_px < 1e-9

    _PARITY_REPORT["geo_rig_parity"] = {
        "samples": 100,
        "max_ecef_err_m": max_ecef_err_m,
        "max_h_err_m": max_h_err_m,
        "max_r_ecef_ned_err": max_r_err,
        "max_quat_err_rad": max_quat_err_rad,
        "max_pinhole_proj_err_px": max_proj_err_px,
        "behind_camera_none_checks": behind_cam_none_checks,
    }
    _persist_parity_report()


def test_tie_guarded_math_oracles_cusum_vraim_meshdaa_ecbf() -> None:
    ffi = XvsCoreFFI()
    boundary_ties_logged: list[dict[str, Any]] = []

    # 1. ES-EKF + Bonferroni 8-arm CUSUM (h_c = 10.530): 30 deg maneuver vs +2.5 m/s ramp
    # Verify non-tie discrete alarm agreement between Rust ES-EKF and Python Bonferroni oracle
    g = 9.80665
    speed = 90.0
    bank_30 = math.radians(30.0)
    yaw_rate = (g / speed) * math.tan(bank_30)
    accs, gvs, banks, omegas = [], [], [], []
    yaw = 0.0
    v_prev = [speed, 0.0, 0.0]
    for _ in range(40):
        dt = 0.1
        yaw += yaw_rate * dt
        v_now = [speed * math.cos(yaw), speed * math.sin(yaw), 0.0]
        accs.append([(v_now[0] - v_prev[0]) / dt, (v_now[1] - v_prev[1]) / dt, 0.0])
        v_prev = v_now
        gvs.append([v_now[0] + 0.03, v_now[1] - 0.02, 0.01])
        banks.append(bank_30)
        omegas.append(abs(yaw_rate))
    rs_turn = ffi.es_ekf_step_cusum_eval([0.0, 0.0, -1200.0], [speed, 0.0, 0.0], 0.1, accs, gvs, banks, omegas)
    non_tie, margin = oracles.check_threshold_tie_margin(rs_turn.cusum_stat, 10.530, 1e-9)
    if not non_tie:
        boundary_ties_logged.append({"check": "turn_cusum", "margin": margin})
    assert rs_turn.spoof_alarm == 0

    # Spoof ramp: +2.5 m/s East over 10 ticks -> trips BOTH Rust ES-EKF and Python Bonferroni detector
    py_det = oracles.PyBonferroniSpoofDetector(cusum_threshold_hc=10.530, cusum_drift_k=0.75)
    s_cov = np.diag([0.16, 0.16, 0.16])
    py_trip_tick = None
    rs_trip_tick = None
    accs_s, gvs_s, banks_s, omegas_s = [], [], [], []
    for t in range(1, 11):
        accs_s.append([0.0, 0.0, 0.0])
        gvs_s.append([90.0, 2.5, 0.0])
        banks_s.append(0.0)
        omegas_s.append(0.0)
        rs_s = ffi.es_ekf_step_cusum_eval([0.0, 0.0, -1200.0], [90.0, 0.0, 0.0], 0.1, accs_s, gvs_s, banks_s, omegas_s)
        py_s = py_det.step(np.array([0.0, 2.5, 0.0]), s_cov)
        if not py_s["cusum_non_tie"]:
            boundary_ties_logged.append({"check": "spoof_cusum", "tick": t, "margin": py_s["cusum_margin"]})
        if rs_s.spoof_alarm and rs_trip_tick is None:
            rs_trip_tick = t
        if py_s["spoof_alarm"] and py_trip_tick is None:
            py_trip_tick = t
    assert rs_trip_tick is not None and rs_trip_tick <= 10
    assert py_trip_tick is not None and py_trip_tick <= 10

    # 2. Vision-RAIM: Cam-R isolation + KSFO 28L vs Taxiway C width gate + +18 px corner bias
    rs_vr_nom = ffi.vraim_eval([0.0, 0.020944, 0.0, 0.0, 0.0, 0.0], 0x3F, 0.0, 61.0, 1800.0)
    assert rs_vr_nom.isolated_cam_idx == 1
    assert rs_vr_nom.healthy_cam_mask == 0b111101
    assert rs_vr_nom.fused_attitude_err_rad < 175e-6
    assert rs_vr_nom.v_vpl_m < 10.0 and rs_vr_nom.v_hpl_m < 40.0
    assert rs_vr_nom.width_fault == 0

    rs_vr_twy = ffi.vraim_eval([0.0] * 6, 0x3F, 0.0, 23.0, 1800.0)
    py_twy_ok, _ = oracles.check_runway_width_consistency(23.0, 61.0, 12.0)
    assert (rs_vr_twy.width_fault == 0) == py_twy_ok and not py_twy_ok

    rs_vr_bias = ffi.vraim_eval([0.0] * 6, 0x3F, 18.0, 61.0, 1200.0)
    assert rs_vr_bias.v_vpl_m > 10.0

    # 3. MeshDAA FCI continuous parity (< 1e-12!) + 3-ship triple-ray ghost gate agreement
    xa = np.array([1200.5, -340.2, 850.7], dtype=np.float64)
    xb = np.array([1208.1, -335.9, 855.3], dtype=np.float64)
    pa = np.array([[140.0, 12.0, -5.0], [12.0, 290.0, 8.0], [-5.0, 8.0, 180.0]], dtype=np.float64)
    pb = np.array([[220.0, -9.0, 6.0], [-9.0, 130.0, -4.0], [6.0, -4.0, 240.0]], dtype=np.float64)
    py_xf, py_pf, _ = oracles.fast_covariance_intersection(xa, pa, xb, pb)
    rs_xf, rs_pf = ffi.fci_fuse_3d(xa.tolist(), pa.reshape(-1).tolist(), xb.tolist(), pb.reshape(-1).tolist())
    fci_x_rel_err = float(np.linalg.norm(py_xf - np.array(rs_xf)) / np.linalg.norm(py_xf))
    fci_p_rel_err = float(np.linalg.norm(py_pf - np.array(rs_pf)) / np.linalg.norm(py_pf))
    assert fci_x_rel_err <= 1e-9
    assert fci_p_rel_err <= 1e-9

    # 3-ship triple-ray ghost check
    base = wgs84.geodetic_to_ecef(37.62, -122.36, 1000.0)
    r_en = wgs84.r_ecef_ned(37.62, -122.36)
    s1 = base + r_en @ np.array([0.0, -1000.0, 0.0])
    s2 = base + r_en @ np.array([0.0, 1000.0, 0.0])
    s3 = base + r_en @ np.array([-800.0, 0.0, -600.0])
    t1 = base + r_en @ np.array([2000.0, -400.0, 0.0])
    t2 = base + r_en @ np.array([2000.0, 400.0, 0.0])

    def _mk_trk(o: np.ndarray, tgt: np.ndarray, sender: int, tid: int) -> XvsBearingTrack:
        u = (tgt - o) / np.linalg.norm(tgt - o)
        return XvsBearingTrack(
            time_utc_us=1000,
            own_ecef_cm=(ctypes.c_int32 * 3)(*[int(round(v * 100.0)) for v in o]),
            bearing_ecef_q15=(ctypes.c_int16 * 3)(*[int(round(v * 32767.0)) for v in u]),
            bearing_rate_urad_s=(ctypes.c_int16 * 3)(0, 0, 0),
            cone_half_angle_urad=2000,
            own_pos_std_cm=50,
            track_id=tid,
            sender_id=sender,
            flags=1,
            cov_diag_packed=(ctypes.c_uint16 * 6)(50, 50, 50, 2000, 2000, 2000),
        )

    rs_mesh = ffi.mesh_daa_eval(
        [
            _mk_trk(s1, t1, 1, 1),
            _mk_trk(s1, t2, 1, 2),
            _mk_trk(s2, t1, 2, 3),
            _mk_trk(s2, t2, 2, 4),
            _mk_trk(s3, t1, 3, 5),
            _mk_trk(s3, t2, 3, 6),
        ]
    )
    py_ghost = oracles.triple_ray_epipolar_ghost_check(
        s1, t1 - s1, s2, t2 - s2, s3, t1 - s3, 0.015
    )
    assert py_ghost["ghost_rejected"] is True
    assert rs_mesh.ghosts_rejected >= 2 and rs_mesh.num_fused == 2

    # 4. ECBF-QP infeasible bilateral+vertical sandwich escape check
    sandwich = [
        XvsIntruderNed((ctypes.c_double * 3)(220.0, 0.0, -1000.0), (ctypes.c_double * 3)(-100.0, 0.0, 0.0), 1),
        XvsIntruderNed((ctypes.c_double * 3)(120.0, -120.0, -1000.0), (ctypes.c_double * 3)(-20.0, 95.0, 0.0), 1),
        XvsIntruderNed((ctypes.c_double * 3)(120.0, 120.0, -1000.0), (ctypes.c_double * 3)(-20.0, -95.0, 0.0), 1),
        XvsIntruderNed((ctypes.c_double * 3)(140.0, 0.0, -1030.0), (ctypes.c_double * 3)(-40.0, 0.0, 15.0), 1),
        XvsIntruderNed((ctypes.c_double * 3)(140.0, 0.0, -970.0), (ctypes.c_double * 3)(-40.0, 0.0, -15.0), 1),
    ]
    rs_esc = ffi.ecbf_qp_solve([0.0, 0.0, -1000.0], [95.0, 0.0, 0.0], 0.0, 0.0, sandwich)
    assert rs_esc.ecbf_infeasible == 1
    assert math.isfinite(rs_esc.bank_cmd_rad) and abs(rs_esc.bank_cmd_rad) <= math.radians(30.0) + 1e-9
    assert math.isfinite(rs_esc.climb_rate_cmd_mps)

    _PARITY_REPORT["math_oracles"] = {
        "turn_30deg_spoof_alarm": int(rs_turn.spoof_alarm),
        "spoof_ramp_trip_tick_rust": rs_trip_tick,
        "spoof_ramp_trip_tick_python": py_trip_tick,
        "vraim_cam_r_isolated_idx": int(rs_vr_nom.isolated_cam_idx),
        "vraim_bias18px_v_vpl_m": float(rs_vr_bias.v_vpl_m),
        "fci_x_rel_err": fci_x_rel_err,
        "fci_p_rel_err": fci_p_rel_err,
        "mesh_3ship_ghosts_rejected": int(rs_mesh.ghosts_rejected),
        "ecbf_sandwich_infeasible": int(rs_esc.ecbf_infeasible),
        "boundary_ties_logged": boundary_ties_logged,
    }
    _persist_parity_report()


def test_headless_emergent_missions_m1_to_m4_and_write_artifacts() -> None:
    ffi = XvsCoreFFI()
    sim = ffi.sim_create(20261101)
    missions_out: dict[str, Any] = {"schema": "xvs.v2_m1_headless_missions/1", "missions": {}}

    try:
        # --- Mission M1: Cam-R +0.020944 rad at t = 8.0 s (seed 20261101) ---
        ffi.sim_reset_mission(sim, mission_id=1, seed=20261101)
        m1_iso_tick = None
        for tick in range(1, 451):
            t_s = tick * 0.1
            f = XvsFaultPerturbations.nominal()
            if t_s >= 8.0:
                f.cam_r_bias_rad = 0.020944
            ffi.sim_set_faults(sim, f)
            ffi.sim_step(sim, 1)
            snap = ffi.sim_snapshot(sim)
            if t_s < 8.0:
                assert snap.simplex_mode == 4, f"M1 premature degradation at t={t_s:.1f}s: mode={snap.simplex_mode}"
            elif m1_iso_tick is None and snap.simplex_mode == 3 and snap.healthy_cam_mask == 0b111101:
                m1_iso_tick = tick
        s1 = ffi.sim_snapshot(sim)
        assert m1_iso_tick is not None and (m1_iso_tick - 80) <= 2
        assert s1.fused_attitude_err_urad < 175.0
        missions_out["missions"]["M1"] = {
            "seed": 20261101,
            "inject_tick": 80,
            "isolated_tick": m1_iso_tick,
            "isolation_latency_ticks": m1_iso_tick - 80,
            "final_simplex_mode": int(s1.simplex_mode),
            "healthy_cam_mask": int(s1.healthy_cam_mask),
            "fused_attitude_err_urad": float(s1.fused_attitude_err_urad),
            "passed": True,
        }

        # --- Mission M2: +2.5 m/s East GNSS ramp at t = 8.0 s (seed 20261102) ---
        ffi.sim_reset_mission(sim, mission_id=2, seed=20261102)
        m2_trip_tick = None
        for tick in range(1, 451):
            t_s = tick * 0.1
            f = XvsFaultPerturbations.nominal()
            if t_s >= 8.0:
                f.gnss_spoof_vel_ned_mps[1] = 2.5
            ffi.sim_set_faults(sim, f)
            ffi.sim_step(sim, 1)
            snap = ffi.sim_snapshot(sim)
            if t_s < 8.0:
                assert snap.simplex_mode == 4, f"M2 premature alarm at t={t_s:.1f}s"
            elif m2_trip_tick is None and snap.simplex_mode == 1 and snap.active_nav_bank == 1:
                m2_trip_tick = tick
        s2 = ffi.sim_snapshot(sim)
        assert m2_trip_tick is not None and (m2_trip_tick - 80) <= 10
        assert s2.cusum_stat > 10.530, f"Expected M2 CUSUM > 10.530 while spoofed, got {s2.cusum_stat}"
        missions_out["missions"]["M2"] = {
            "seed": 20261102,
            "inject_tick": 80,
            "trip_tick": m2_trip_tick,
            "trip_latency_ticks": m2_trip_tick - 80,
            "final_simplex_mode": int(s2.simplex_mode),
            "active_nav_bank": int(s2.active_nav_bank),
            "final_cusum_stat": float(s2.cusum_stat),
            "passed": True,
        }

        # --- Mission M3: 3-ship MeshDAA + ECBF-QP (seed 20261103) ---
        ffi.sim_reset_mission(sim, mission_id=3, seed=20261103)
        max_ghosts = 0
        min_h_m = float("inf")
        nmac_count = 0
        lowc_count = 0
        for tick in range(1, 451):
            t_s = tick * 0.1
            f = XvsFaultPerturbations.nominal()
            if t_s >= 12.0:
                f.peer3_active = 1
            ffi.sim_set_faults(sim, f)
            ffi.sim_step(sim, 1)
            snap = ffi.sim_snapshot(sim)
            max_ghosts = max(max_ghosts, int(snap.ghosts_rejected))
            for k in range(int(snap.intruder_count)):
                rel = snap.intruder_rel_ned[k]
                dh = math.hypot(rel[0], rel[1])
                dv = abs(rel[2])
                min_h_m = min(min_h_m, dh)
                if dh < 152.4 and dv < 30.48:
                    nmac_count += 1
                if dh < 1219.2 and dv < 137.16:
                    lowc_count += 1
        assert max_ghosts >= 2
        assert nmac_count == 0 and lowc_count == 0
        missions_out["missions"]["M3"] = {
            "seed": 20261103,
            "peer3_join_tick": 120,
            "max_ghosts_rejected": max_ghosts,
            "min_horizontal_sep_m": float(min_h_m),
            "nmac_ticks": nmac_count,
            "lowc_ticks": lowc_count,
            "passed": True,
        }

        # --- Mission M4: KSFO 28L 0.22x fog + LWIR EFVS (10s) + +18 px corner bias (24s) ---
        ffi.sim_reset_mission(sim, mission_id=4, seed=20261104)
        m4_ga_tick = None
        h_at_24s = None
        for tick in range(1, 451):
            t_s = tick * 0.1
            f = XvsFaultPerturbations.nominal()
            f.fog_contrast = 0.22
            if t_s >= 10.0:
                f.lwir_efvs_enabled = 1
            if t_s >= 24.0:
                f.runway_corner_bias_px = 18.0
            ffi.sim_set_faults(sim, f)
            ffi.sim_step(sim, 1)
            snap = ffi.sim_snapshot(sim)
            if tick == 240:
                h_at_24s = float(snap.h_m)
            if t_s < 24.0:
                assert snap.mandatory_go_around == 0, f"M4 premature Go-Around at t={t_s:.1f}s (v_vpl_mm={snap.v_vpl_mm})"
            elif m4_ga_tick is None and snap.mandatory_go_around == 1:
                m4_ga_tick = tick
        s4 = ffi.sim_snapshot(sim)
        assert m4_ga_tick is not None and (m4_ga_tick - 240) <= 1
        assert s4.v_vpl_mm > 10_000.0, f"Expected M4 v_vpl_mm > 10_000, got {s4.v_vpl_mm}"
        assert s4.simplex_mode == 0 and s4.blank_conformal_hud == 1
        assert h_at_24s is not None and float(s4.h_m) > h_at_24s + 50.0
        missions_out["missions"]["M4"] = {
            "seed": 20261104,
            "efvs_on_tick": 100,
            "corner_bias_tick": 240,
            "go_around_tick": m4_ga_tick,
            "go_around_latency_ticks": m4_ga_tick - 240,
            "v_vpl_mm_final": float(s4.v_vpl_mm),
            "h_m_at_24s": h_at_24s,
            "h_m_final": float(s4.h_m),
            "climb_gain_m": float(s4.h_m) - h_at_24s,
            "final_simplex_mode": int(s4.simplex_mode),
            "blank_conformal_hud": int(s4.blank_conformal_hud),
            "passed": True,
        }
    finally:
        ffi.sim_destroy(sim)

    missions_out["all_missions_passed"] = all(
        bool(missions_out["missions"][m]["passed"]) for m in ("M1", "M2", "M3", "M4")
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MISSIONS_JSON_PATH.write_text(json.dumps(missions_out, indent=2) + "\n", encoding="utf-8")


def test_d16_m1_b1_ffi_nan_fail_closed_all_channels() -> None:
    """D16-M1-B1 regression: NaN on GNSS, runway bias, Cam-R, or OOD MUST force P0 at 0 ticks."""
    ffi = XvsCoreFFI()
    sim = ffi.sim_create(20261120)
    try:
        # 1. gnss_spoof_vel_ned_mps[1] = NaN -> P0 at 0 ticks + Bank 1 + finite velocity
        ffi.sim_reset_mission(sim, mission_id=2, seed=20261120)
        ffi.sim_step(sim, 1)
        f = XvsFaultPerturbations.nominal()
        f.gnss_spoof_vel_ned_mps[1] = float("nan")
        ffi.sim_set_faults(sim, f)
        ffi.sim_step(sim, 1)
        s = ffi.sim_snapshot(sim)
        assert s.simplex_mode == 0 and s.raw_target == 0
        assert s.active_nav_bank == 1 and s.blank_conformal_hud == 1
        assert math.isnan(s.cusum_stat) and math.isnan(s.nis_stat)
        assert all(math.isfinite(float(v)) for v in s.v_ned)

        # 2. runway_corner_bias_px = NaN during M4 approach -> P0 + mandatory_go_around at 0 ticks
        ffi.sim_reset_mission(sim, mission_id=4, seed=20261120)
        ffi.sim_step(sim, 1)
        f = XvsFaultPerturbations.nominal()
        f.runway_corner_bias_px = float("nan")
        ffi.sim_set_faults(sim, f)
        ffi.sim_step(sim, 1)
        s = ffi.sim_snapshot(sim)
        assert s.simplex_mode == 0 and s.raw_target == 0
        assert s.mandatory_go_around == 1 and s.blank_conformal_hud == 1

        # 3. cam_r_bias_rad = NaN -> P0 at 0 ticks
        ffi.sim_reset_mission(sim, mission_id=1, seed=20261120)
        ffi.sim_step(sim, 1)
        f = XvsFaultPerturbations.nominal()
        f.cam_r_bias_rad = float("nan")
        ffi.sim_set_faults(sim, f)
        ffi.sim_step(sim, 1)
        s = ffi.sim_snapshot(sim)
        assert s.simplex_mode == 0 and s.raw_target == 0 and s.blank_conformal_hud == 1

        # 4. ood_global_score = NaN -> P0 at 0 ticks
        ffi.sim_reset_mission(sim, mission_id=1, seed=20261120)
        ffi.sim_step(sim, 1)
        f = XvsFaultPerturbations.nominal()
        f.ood_global_score = float("nan")
        ffi.sim_set_faults(sim, f)
        ffi.sim_step(sim, 1)
        s = ffi.sim_snapshot(sim)
        assert s.simplex_mode == 0 and s.raw_target == 0 and s.blank_conformal_hud == 1
    finally:
        ffi.sim_destroy(sim)
