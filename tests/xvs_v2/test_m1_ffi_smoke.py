"""Task 3 smoke tests for xvs-ffi (C-ABI + Python ctypes), xvs-wasm Brotli budget, and xvs_core.h."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WASM_BG_PATH = REPO_ROOT / "apps" / "livedeck" / "src" / "wasm" / "xvs_wasm_bg.wasm"
HEADER_PATH = REPO_ROOT / "xvs-core" / "crates" / "xvs-ffi" / "include" / "xvs_core.h"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics import gdl90, mavlink_adsb  # noqa: E402
from xvs.core_ffi import (  # noqa: E402
    XvsCoreFFI,
    XvsFaultPerturbations,
    XvsGdl90Target,
    XvsMavlinkAdsb246,
)


def _brotli_size_bytes(path: Path) -> int:
    try:
        import brotli  # type: ignore

        return len(brotli.compress(path.read_bytes(), quality=11))
    except ImportError:
        out = subprocess.check_output(
            [
                "node",
                "-e",
                (
                    "const fs=require('fs'),z=require('zlib');"
                    "const b=fs.readFileSync(process.argv[1]);"
                    "const c=z.brotliCompressSync(b,{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}});"
                    "console.log(c.length);"
                ),
                str(path),
            ],
            text=True,
        )
        return int(out.strip())


def test_ffi_header_and_wasm_brotli_budget() -> None:
    assert HEADER_PATH.is_file(), f"Missing generated C header: {HEADER_PATH}"
    header_text = HEADER_PATH.read_text(encoding="utf-8")
    for sym in (
        "xvs_rta_step",
        "xvs_sim_create",
        "xvs_sim_snapshot",
        "xvs_geo_geodetic_to_ecef",
        "xvs_encode_gdl90_heartbeat",
        "xvs_encode_mavlink_bearing_track",
    ):
        assert sym in header_text, f"Expected symbol {sym} in {HEADER_PATH}"

    assert WASM_BG_PATH.is_file(), f"Missing wasm-bindgen output: {WASM_BG_PATH}"
    brotli_bytes = _brotli_size_bytes(WASM_BG_PATH)
    assert brotli_bytes <= 184_320, f"xvs_wasm_bg.wasm Brotli {brotli_bytes} B > 184,320 B (180 KB)"


def test_ffi_emergent_missions_m1_and_m2_and_m4() -> None:
    try:
        ffi = XvsCoreFFI()
    except FileNotFoundError as exc:
        pytest.skip(f"xvs-ffi cdylib not built; run cargo build -p xvs-ffi --release ({exc})")
    sim = ffi.sim_create(20261120)
    try:
        # Mission M1: Cam-R +0.020944 rad (+1.2 deg) drift -> ISOLATED_5CAM (mode == 3)
        ffi.sim_reset_mission(sim, mission_id=1, seed=20261120)
        ffi.sim_step(sim, rta_ticks=10)
        snap0 = ffi.sim_snapshot(sim)
        assert snap0.simplex_mode == 4  # NominalMl

        f1 = XvsFaultPerturbations.nominal()
        f1.cam_r_bias_rad = 0.020944
        ffi.sim_set_faults(sim, f1)
        ffi.sim_step(sim, rta_ticks=2)
        snap1 = ffi.sim_snapshot(sim)
        assert snap1.simplex_mode == 3  # Isolated5Cam
        assert snap1.healthy_cam_mask == 0b111101
        assert snap1.fused_attitude_err_urad < 175.0

        # Mission M2: +2.5 m/s East GNSS ramp -> FALLBACK_D0_SAFE (mode == 1) + Bank 1
        ffi.sim_reset_mission(sim, mission_id=2, seed=20261120)
        ffi.sim_step(sim, rta_ticks=10)
        f2 = XvsFaultPerturbations.nominal()
        f2.gnss_spoof_vel_ned_mps[1] = 2.5
        ffi.sim_set_faults(sim, f2)
        tripped = False
        for _ in range(10):
            ffi.sim_step(sim, rta_ticks=1)
            snap2 = ffi.sim_snapshot(sim)
            if snap2.simplex_mode == 1 and snap2.active_nav_bank == 1:
                tripped = True
                break
        assert tripped, "M2 must trip to FALLBACK_D0_SAFE + Bank 1 within <= 10 ticks"

        # Mission M4: KSFO 28L +18 px corner bias -> MANDATORY GO-AROUND + INVALID_INHIBIT
        ffi.sim_reset_mission(sim, mission_id=4, seed=20261120)
        ffi.sim_step(sim, rta_ticks=5)
        f4 = XvsFaultPerturbations.nominal()
        f4.runway_corner_bias_px = 18.0
        f4.fog_contrast = 0.22
        f4.lwir_efvs_enabled = 1
        ffi.sim_set_faults(sim, f4)
        ffi.sim_step(sim, rta_ticks=1)
        snap4 = ffi.sim_snapshot(sim)
        assert snap4.mandatory_go_around == 1
        assert snap4.simplex_mode == 0  # InvalidInhibit
        assert snap4.blank_conformal_hud == 1

        # Drain live TX packets
        pkts = ffi.sim_drain_all_tx_packets(sim)
        assert any(proto == 0x01 for proto, _ in pkts)
        assert any(proto == 0x02 for proto, _ in pkts)
    finally:
        ffi.sim_destroy(sim)


def test_ffi_gdl90_and_mavlink_bit_exact_vs_python() -> None:
    try:
        ffi = XvsCoreFFI()
    except FileNotFoundError as exc:
        pytest.skip(f"xvs-ffi cdylib not built; run cargo build -p xvs-ffi --release ({exc})")

    # 1. GDL-90 Heartbeat bit-exact vs src/xvs/avionics/gdl90.py
    py_hb = gdl90.encode_heartbeat(
        utc_time_s=43200,
        gps_pos_valid=True,
        maint_reqd=False,
        utc_ok=True,
        uplink_count=3,
        basic_and_long_count=17,
    )
    rs_hb = ffi.encode_gdl90_heartbeat(
        utc_time_s=43200,
        gps_pos_valid=True,
        maint_reqd=False,
        utc_ok=True,
        uplink_count=3,
        basic_and_long_count=17,
    )
    assert rs_hb == py_hb

    # 2. GDL-90 Ownship Geometric Altitude bit-exact
    py_geo = gdl90.encode_ownship_geo_altitude(alt_ft=3505.0, vfom_m=4.0, vertical_warning=False)
    rs_geo = ffi.encode_gdl90_ownship_geo_alt(alt_ft=3505.0, vfom_m=4.0, vertical_warning=False)
    assert rs_geo == py_geo

    # 3. GDL-90 Ownship Report (#10) & Traffic Report (#20) bit-exact
    py_rep = gdl90.Gdl90TargetReport(
        msg_id=gdl90.GDL90_MSG_ID_OWNSHIP_REPORT,
        alert_status=0,
        address_type=0,
        participant_address=0xA1B2C3,
        lat_deg=37.6119,
        lon_deg=-122.3574,
        altitude_ft=2500.0,
        misc_indicators=0x09,
        nic=8,
        nacp=8,
        horiz_vel_kt=180.0,
        vert_vel_fpm=-512.0,
        track_heading_deg=284.0,
        emitter_category=1,
        callsign="XVSOWN1",
        emergency_code=0,
    )
    py_own = gdl90.encode_target_report(py_rep)
    rs_tgt = XvsGdl90Target.from_values(
        alert_status=0,
        address_type=0,
        participant_address=0xA1B2C3,
        lat_deg=37.6119,
        lon_deg=-122.3574,
        altitude_ft=2500.0,
        misc_indicators=0x09,
        nic=8,
        nacp=8,
        horiz_vel_kt=180.0,
        vert_vel_fpm=-512.0,
        track_heading_deg=284.0,
        emitter_category=1,
        callsign="XVSOWN1",
        emergency_code=0,
    )
    rs_own = ffi.encode_gdl90_ownship(rs_tgt)
    assert rs_own == py_own

    # 4. MAVLink v2 #246 ADSB_VEHICLE bit-exact vs src/xvs/avionics/mavlink_adsb.py
    py_adsb = mavlink_adsb.pack_adsb_vehicle(
        mavlink_adsb.MavlinkAdsbVehicle(
            icao_address=0xF00007,
            lat_deg_e7=376_119_000,
            lon_deg_e7=-1_223_574_000,
            altitude_mm=600_000,
            heading_cdeg=28_400,
            hor_velocity_cms=9_000,
            ver_velocity_cms=-377,
            flags=0x00FF,
            squawk=1200,
            altitude_type=1,
            callsign="XVS0007",
            emitter_type=14,
            tslc=0,
            seq=7,
            sysid=1,
            compid=156,
        )
    )
    rs_adsb = ffi.encode_mavlink_adsb_246(
        XvsMavlinkAdsb246.from_values(
            icao_address=0xF00007,
            lat_deg_e7=376_119_000,
            lon_deg_e7=-1_223_574_000,
            altitude_mm=600_000,
            heading_cdeg=28_400,
            hor_velocity_cms=9_000,
            ver_velocity_cms=-377,
            flags=0x00FF,
            squawk=1200,
            altitude_type=1,
            callsign="XVS0007",
            emitter_type=14,
            tslc=0,
            seq=7,
            sysid=1,
            compid=156,
        )
    )
    assert rs_adsb == py_adsb

    # 5. WGS-84 & pinhole projection parity sanity
    ecef = ffi.geodetic_to_ecef(37.6119, -122.3574, -28.3)
    lat2, lon2, h2 = ffi.ecef_to_geodetic(*ecef)
    assert abs(lat2 - 37.6119) < 1e-9
    assert abs(lon2 - (-122.3574)) < 1e-9
    assert abs(h2 - (-28.3)) < 1e-6
    q = ffi.euler_zyx_to_quat(1.1, -0.15, 0.42)
    assert q[0] >= 0.0 and abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-12
