"""Comprehensive verification suite for Wave 1 deliverables:
  1. Clean-room Zero-GPL PX4 MAVLink 2.0 dialect (`#102 VISION_POSITION_ESTIMATE` & `#331 ODOMETRY`)
  2. Simultaneous compound-fault live WSL2 `ArduPlane V4.7.3` SIL tape & subsystem integration
  3. Upgraded Full-Stack GitHub Actions CI workflow (`.github/workflows/xvs_core_ci.yml`)
  4. Jetson Orin Day-1 TensorRT script (`docker/edge-infer/build_trt_engines.sh`)
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pytest

from xvs.v2.sensor_health import SensorHealthMonitor
from xvs.v2.studio_bridge.hil_bridge import (
    MAVLINK_V2_STX,
    NonFlightSimOnlyError,
    ShadowModeTelemetryLogger,
    verify_shadow_log_chain,
)
from xvs.v2.studio_bridge.px4_mavlink import (
    COV21_LEN,
    CRC_EXTRA_ODOMETRY,
    CRC_EXTRA_VISION_POSITION_ESTIMATE,
    MAV_ESTIMATOR_TYPE_VIO,
    MAV_FRAME_BODY_FRD,
    MAV_FRAME_LOCAL_ENU,
    MAV_FRAME_LOCAL_NED,
    MSGID_ODOMETRY,
    MSGID_VISION_POSITION_ESTIMATE,
    Px4FrameViolationError,
    Px4MavlinkUdpSender,
    Px4Odometry,
    Px4VisionPositionEstimate,
    assert_zero_pymavlink,
    decode_px4_frame,
    encode_odometry_frame,
    encode_px4_frame,
    encode_vision_position_estimate_frame,
    euler_frd_to_quaternion_ned,
    extract_mavlink_v2_frames,
    make_upper_tri_6x6_covariance,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MavlinkCrcError,
    MavlinkFrameError,
    compute_crc_extra,
    x25_crc16,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PX4_MODULE_PATH = REPO_ROOT / "src" / "xvs" / "v2" / "studio_bridge" / "px4_mavlink.py"
COMPOUND_SIL_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "run_compound_fault_sil_gate.py"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "xvs_core_ci.yml"
TRT_SCRIPT_PATH = REPO_ROOT / "docker" / "edge-infer" / "build_trt_engines.sh"
EDGE_RELEASE_EXE = REPO_ROOT / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe"
LIVE_COMPOUND_DIR = REPO_ROOT / "outputs" / "sitl_proof" / "v112_compound_fault"
LIVE_COMPOUND_TAPE = LIVE_COMPOUND_DIR / "compound_mission_tape.ndjson"
LIVE_COMPOUND_REPORT = LIVE_COMPOUND_DIR / "compound_fault_sil_report.json"


def _find_bash_executable() -> str | None:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    return shutil.which("bash")


def _build_raw_mav2_frame(msgid: int, payload: bytes, crc_extra: int, *, seq: int = 0) -> bytes:
    hdr = struct.pack(
        "<BBBBBBBBB",
        len(payload),
        0,
        0,
        seq & 0xFF,
        1,
        197,
        msgid & 0xFF,
        (msgid >> 8) & 0xFF,
        (msgid >> 16) & 0xFF,
    )
    crc = x25_crc16(hdr + payload + bytes([crc_extra & 0xFF]))
    return bytes([MAVLINK_V2_STX]) + hdr + payload + struct.pack("<H", crc)


# ==============================================================================
# 1. Clean-Room Zero-GPL PX4 MAVLink 2.0 (`#102` & `#331`) Tests
# ==============================================================================


def test_px4_crc_extra_mathematical_derivation_and_zero_pymavlink() -> None:
    """Verify canonical MAVLink common.xml CRC_EXTRA bytes (#102 -> 158, #331 -> 91) and zero pymavlink."""
    assert_zero_pymavlink()
    assert "pymavlink" not in sys.modules

    vpe_crc = compute_crc_extra(
        "VISION_POSITION_ESTIMATE",
        (
            ("uint64_t", "usec", 0),
            ("float", "x", 0),
            ("float", "y", 0),
            ("float", "z", 0),
            ("float", "roll", 0),
            ("float", "pitch", 0),
            ("float", "yaw", 0),
        ),
    )
    odom_crc = compute_crc_extra(
        "ODOMETRY",
        (
            ("uint64_t", "time_usec", 0),
            ("uint8_t", "frame_id", 0),
            ("uint8_t", "child_frame_id", 0),
            ("float", "x", 0),
            ("float", "y", 0),
            ("float", "z", 0),
            ("float", "q", 4),
            ("float", "vx", 0),
            ("float", "vy", 0),
            ("float", "vz", 0),
            ("float", "rollspeed", 0),
            ("float", "pitchspeed", 0),
            ("float", "yawspeed", 0),
            ("float", "pose_covariance", 21),
            ("float", "velocity_covariance", 21),
        ),
    )
    assert vpe_crc == 158
    assert odom_crc == 91
    assert CRC_EXTRA_VISION_POSITION_ESTIMATE == 158
    assert CRC_EXTRA_ODOMETRY == 91

    # Static AST audit across both new Python files: zero `pymavlink` imports!
    for py_path in (PX4_MODULE_PATH, COMPOUND_SIL_SCRIPT):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "pymavlink", f"GPL pymavlink imported in {py_path}"
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] != "pymavlink", f"GPL pymavlink imported in {py_path}"


def test_px4_vision_position_estimate_102_roundtrip_signing_and_tamper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #102 VISION_POSITION_ESTIMATE NED/FRD wire round-trip, truncation, SHA-256 signing & tamper reject."""
    monkeypatch.delenv("NON_FLIGHT_SIM_ONLY", raising=False)
    monkeypatch.delenv("OPENXVS_NON_FLIGHT_SIM_ONLY", raising=False)

    cov = make_upper_tri_6x6_covariance(0.25, 0.30, 0.45, 0.002, 0.003, 0.01)
    wire = encode_vision_position_estimate_frame(
        usec=1_732_700_123_456,
        x_north_m=142.5,
        y_east_m=-38.25,
        z_down_m=-115.75,
        roll_rad=math.radians(6.5),
        pitch_rad=math.radians(-2.25),
        yaw_rad=math.radians(298.0),
        covariance=cov,
        reset_counter=3,
        seq=19,
        sysid=1,
        compid=197,
    )
    assert wire[0] == MAVLINK_V2_STX
    dec = decode_px4_frame(wire)
    assert dec.msgid == MSGID_VISION_POSITION_ESTIMATE
    assert dec.crc_extra == 158
    assert dec.seq == 19
    assert dec.sysid == 1 and dec.compid == 197
    assert isinstance(dec.message, Px4VisionPositionEstimate)
    assert dec.message.usec == 1_732_700_123_456
    assert dec.message.x_north_m == pytest.approx(142.5, abs=1e-4)
    assert dec.message.y_east_m == pytest.approx(-38.25, abs=1e-4)
    assert dec.message.z_down_m == pytest.approx(-115.75, abs=1e-4)
    assert dec.message.roll_rad == pytest.approx(math.radians(6.5), abs=1e-5)
    assert dec.message.pitch_rad == pytest.approx(math.radians(-2.25), abs=1e-5)
    assert dec.message.yaw_rad == pytest.approx(math.radians(298.0), abs=1e-5)
    assert dec.message.reset_counter == 3
    assert len(dec.message.covariance) == COV21_LEN
    assert dec.message.covariance[0] == pytest.approx(0.25, abs=1e-6)
    assert dec.message.covariance[20] == pytest.approx(0.01, abs=1e-6)

    # Trailing-zero truncation shrinks frame when reset_counter=0 and tail covariances are 0
    wire_trunc = encode_vision_position_estimate_frame(
        usec=12345,
        x_north_m=10.0,
        y_east_m=20.0,
        z_down_m=-50.0,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        reset_counter=0,
    )
    assert len(wire_trunc) < 12 + 117
    dec_trunc = decode_px4_frame(wire_trunc)
    assert isinstance(dec_trunc.message, Px4VisionPositionEstimate)
    assert dec_trunc.message.z_down_m == pytest.approx(-50.0, abs=1e-4)

    # 1-bit tamper rejected fail-closed by X.25 CRC-16
    corrupted = bytearray(wire)
    corrupted[12] ^= 0x01
    with pytest.raises(MavlinkCrcError):
        decode_px4_frame(bytes(corrupted))

    # Signed MAVLink 2.0 round-trip + signature tamper rejection
    key = bytes(range(32))
    signed_wire = encode_px4_frame(
        dec.message, seq=20, secret_key=key, link_id=2, timestamp_48=987_654_321
    )
    dec_sig = decode_px4_frame(signed_wire, secret_key=key, require_signature=True)
    assert dec_sig.signed is True and dec_sig.signature_verified is True
    assert dec_sig.link_id == 2 and dec_sig.timestamp_48 == 987_654_321

    bad_sig = bytearray(signed_wire)
    bad_sig[-1] ^= 0x80
    with pytest.raises(MavlinkCrcError):
        decode_px4_frame(bytes(bad_sig), secret_key=key, require_signature=True)


def test_px4_odometry_331_roundtrip_and_strict_ned_frd_enu_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #331 ODOMETRY strictly permits LOCAL_NED(1)+BODY_FRD(12) and rejects ENU/unknown fail-closed."""
    monkeypatch.delenv("NON_FLIGHT_SIM_ONLY", raising=False)
    monkeypatch.delenv("OPENXVS_NON_FLIGHT_SIM_ONLY", raising=False)

    q_ned = euler_frd_to_quaternion_ned(math.radians(12.0), math.radians(-3.0), math.radians(298.0))
    pose_cov = make_upper_tri_6x6_covariance(0.4, 0.4, 0.8, 0.005, 0.005, 0.02)
    vel_cov = make_upper_tri_6x6_covariance(0.1, 0.1, 0.2, 0.001, 0.001, 0.004)

    wire = encode_odometry_frame(
        time_usec=1_732_700_999_000,
        x_north_m=512.0,
        y_east_m=-210.5,
        z_down_m=-180.25,
        q=q_ned,
        vx_mps=22.5,
        vy_mps=-4.0,
        vz_mps=0.35,
        rollspeed_rad_s=0.04,
        pitchspeed_rad_s=-0.01,
        yawspeed_rad_s=0.06,
        pose_covariance=pose_cov,
        velocity_covariance=vel_cov,
        frame_id=MAV_FRAME_LOCAL_NED,
        child_frame_id=MAV_FRAME_BODY_FRD,
        reset_counter=2,
        estimator_type=MAV_ESTIMATOR_TYPE_VIO,
        quality=94,
        seq=41,
    )
    dec = decode_px4_frame(wire)
    assert dec.msgid == MSGID_ODOMETRY
    assert dec.crc_extra == 91
    assert isinstance(dec.message, Px4Odometry)
    assert dec.message.frame_id == MAV_FRAME_LOCAL_NED
    assert dec.message.child_frame_id == MAV_FRAME_BODY_FRD
    assert dec.message.x_north_m == pytest.approx(512.0, abs=1e-4)
    assert dec.message.z_down_m == pytest.approx(-180.25, abs=1e-4)
    assert dec.message.vx_mps == pytest.approx(22.5, abs=1e-4)
    assert dec.message.quality == 94
    assert dec.message.estimator_type == MAV_ESTIMATOR_TYPE_VIO
    for got, exp in zip(dec.message.q, q_ned):
        assert got == pytest.approx(exp, abs=1e-5)

    # Hard-reject ENU (MAV_FRAME_LOCAL_ENU = 4) and every non-NED / non-FRD frame on construction!
    for bad_frame, bad_child in (
        (MAV_FRAME_LOCAL_ENU, MAV_FRAME_BODY_FRD),
        (MAV_FRAME_LOCAL_NED, MAV_FRAME_LOCAL_ENU),
        (0, MAV_FRAME_BODY_FRD),
        (MAV_FRAME_LOCAL_NED, 8),  # BODY_NED deprecated
        (MAV_FRAME_LOCAL_NED, 21),  # LOCAL_FRD != BODY_FRD
        (99, 99),
    ):
        with pytest.raises(Px4FrameViolationError):
            Px4Odometry(
                time_usec=100,
                x_north_m=0.0,
                y_east_m=0.0,
                z_down_m=-10.0,
                q=(1.0, 0.0, 0.0, 0.0),
                vx_mps=20.0,
                vy_mps=0.0,
                vz_mps=0.0,
                frame_id=bad_frame,
                child_frame_id=bad_child,
            )

    # Crafted valid-CRC wire packet with forged ENU frame_id=4 MUST ALSO be rejected fail-closed on decode!
    forged_payload = struct.pack(
        "<Q55fBBBBb",
        1000,
        *([0.0, 0.0, -50.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 48),
        MAV_FRAME_LOCAL_ENU,
        MAV_FRAME_BODY_FRD,
        0,
        MAV_ESTIMATOR_TYPE_VIO,
        100,
    )
    forged_wire = _build_raw_mav2_frame(MSGID_ODOMETRY, forged_payload, CRC_EXTRA_ODOMETRY)
    with pytest.raises(Px4FrameViolationError):
        decode_px4_frame(forged_wire)

    # Non-unit / zero quaternion rejected fail-closed
    with pytest.raises(ValueError):
        Px4Odometry(
            time_usec=100,
            x_north_m=0.0,
            y_east_m=0.0,
            z_down_m=-10.0,
            q=(0.0, 0.0, 0.0, 0.0),
            vx_mps=20.0,
            vy_mps=0.0,
            vz_mps=0.0,
        )


def test_px4_nan_inf_bool_and_non_flight_sim_only_lockouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify NaN/Inf/bool fail-closed rejection (encode + decode) and NON_FLIGHT_SIM_ONLY=1 lockout."""
    # 1. NaN / Inf / negative variance rejected on #102 and #331
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            Px4VisionPositionEstimate(
                usec=1, x_north_m=bad_val, y_east_m=0.0, z_down_m=-10.0, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0
            )
        with pytest.raises(ValueError):
            Px4Odometry(
                time_usec=1,
                x_north_m=0.0,
                y_east_m=0.0,
                z_down_m=-10.0,
                q=(1.0, 0.0, 0.0, 0.0),
                vx_mps=bad_val,
                vy_mps=0.0,
                vz_mps=0.0,
            )

    bad_cov = list((0.0,) * COV21_LEN)
    bad_cov[6] = -0.1  # negative diagonal variance
    with pytest.raises(ValueError):
        Px4VisionPositionEstimate(
            usec=1,
            x_north_m=0.0,
            y_east_m=0.0,
            z_down_m=-10.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
            covariance=tuple(bad_cov),
        )

    # Wire-forged #102 containing IEEE-754 NaN with valid CRC MUST fail closed on decode!
    nan_vpe_payload = struct.pack(
        "<Q6f21fB", 1000, float("nan"), 0.0, -50.0, 0.0, 0.0, 0.0, *([0.0] * 21), 0
    )
    nan_vpe_wire = _build_raw_mav2_frame(
        MSGID_VISION_POSITION_ESTIMATE, nan_vpe_payload, CRC_EXTRA_VISION_POSITION_ESTIMATE
    )
    with pytest.raises(ValueError):
        decode_px4_frame(nan_vpe_wire)

    # 2. NON_FLIGHT_SIM_ONLY=1 and OPENXVS_NON_FLIGHT_SIM_ONLY=1 lockouts block ALL outbound encoding & UDP TX
    vpe_msg = Px4VisionPositionEstimate(
        usec=42, x_north_m=1.0, y_east_m=2.0, z_down_m=-30.0, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.5
    )
    odom_msg = Px4Odometry(
        time_usec=42,
        x_north_m=1.0,
        y_east_m=2.0,
        z_down_m=-30.0,
        q=(1.0, 0.0, 0.0, 0.0),
        vx_mps=20.0,
        vy_mps=0.0,
        vz_mps=0.0,
    )
    # Capture a valid wire frame while explicitly unlocked for listen-only RX decode verification
    valid_wire = encode_px4_frame(vpe_msg, non_flight_sim_only=False)

    for env_key in ("NON_FLIGHT_SIM_ONLY", "OPENXVS_NON_FLIGHT_SIM_ONLY"):
        monkeypatch.delenv("NON_FLIGHT_SIM_ONLY", raising=False)
        monkeypatch.delenv("OPENXVS_NON_FLIGHT_SIM_ONLY", raising=False)
        monkeypatch.setenv(env_key, "1")

        with pytest.raises(NonFlightSimOnlyError):
            encode_px4_frame(vpe_msg)
        with pytest.raises(NonFlightSimOnlyError):
            encode_px4_frame(odom_msg)
        with pytest.raises(NonFlightSimOnlyError):
            with Px4MavlinkUdpSender(target_host="127.0.0.1", target_port=14540) as sender:
                sender.send(vpe_msg)

        # Listen-only RX decode remains safe under NON_FLIGHT_SIM_ONLY=1
        rx = decode_px4_frame(valid_wire)
        assert rx.msgid == MSGID_VISION_POSITION_ESTIMATE

    # 3. Localhost UDP loopback send/recv when explicitly unlocked (non_flight_sim_only=False)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx_sock.bind(("127.0.0.1", 0))
        rx_sock.settimeout(1.5)
        rx_port = int(rx_sock.getsockname()[1])
        with Px4MavlinkUdpSender(
            target_host="127.0.0.1", target_port=rx_port, non_flight_sim_only=False
        ) as sender:
            sender.send(vpe_msg)
            sender.send(odom_msg)
        p1, _ = rx_sock.recvfrom(512)
        p2, _ = rx_sock.recvfrom(512)
        frames, rem = extract_mavlink_v2_frames(p1 + p2)
        assert rem == b"" and len(frames) == 2
        assert decode_px4_frame(frames[0]).msgid == MSGID_VISION_POSITION_ESTIMATE
        assert decode_px4_frame(frames[1]).msgid == MSGID_ODOMETRY
    finally:
        rx_sock.close()

    # Unknown msgid rejected fail-closed
    bad_mid = _build_raw_mav2_frame(84, b"\x00", 143)
    with pytest.raises(MavlinkFrameError):
        decode_px4_frame(bad_mid)


# ==============================================================================
# 2. Compound-Fault SIL Tape & Subsystem Stress Verification
# ==============================================================================


def test_compound_fault_subsystems_and_synthetic_tape_tamper_detection(tmp_path: Path) -> None:
    """Self-contained CI-safe compound-fault check: 90 deg coarse heading + LWIR freeze + hash-chained tape."""
    ch_mod = pytest.importorskip("xvs.v2.xvs_nav.coarse_heading")
    s2_mod = pytest.importorskip("xvs.v2.xvs_nav.stage2_xfeat")
    COLD_START_4X90_HYPOTHESES_DEG = ch_mod.COLD_START_4X90_HYPOTHESES_DEG
    match_with_coarse_heading_sweep = ch_mod.match_with_coarse_heading_sweep
    XFeatOnnxRunner = s2_mod.XFeatOnnxRunner

    # 1. 90-deg heading perturbation rescue via coarse_heading.py (fails naive 0 deg, rescued by 4x90)
    rng = np.random.default_rng(20261127)
    scene = np.full((384, 384, 3), 108, dtype=np.uint8)
    for _ in range(95):
        x0, y0 = int(rng.integers(18, 342)), int(rng.integers(18, 342))
        bw, bh = int(rng.integers(8, 28)), int(rng.integers(8, 28))
        col = tuple(int(c) for c in rng.integers(25, 245, size=3))
        cv2.rectangle(scene, (x0, y0), (x0 + bw, y0 + bh), col, -1)
        cv2.rectangle(scene, (x0, y0), (x0 + bw, y0 + bh), (18, 18, 18), 1)
    cv2.line(scene, (10, 178), (374, 198), (235, 235, 225), 5)
    cv2.line(scene, (188, 10), (212, 374), (220, 226, 238), 4)
    map_crop = scene[64:320, 64:320].copy()
    q90 = cv2.warpAffine(scene, cv2.getRotationMatrix2D((192.0, 192.0), 90.0, 1.0), (384, 384))[64:320, 64:320]

    xrunner = XFeatOnnxRunner(None, max_keypoints=256, ort_threads=1)
    km, dm, _ = xrunner.extract(map_crop)
    res_naive = match_with_coarse_heading_sweep(q90, km, dm, xfeat_runner=xrunner, angles_deg=(0.0,), img_hw_m=(256, 256))
    res_sweep = match_with_coarse_heading_sweep(
        q90, km, dm, xfeat_runner=xrunner, angles_deg=COLD_START_4X90_HYPOTHESES_DEG, img_hw_m=(256, 256)
    )
    assert res_naive.n_geom_inliers < 12
    assert res_sweep.n_geom_inliers >= 12 and res_sweep.affine_orig_to_map is not None

    # 2. Simultaneous LWIR shutter freeze -> q_lwir_effective == 0.00 fail-closed
    mon = SensorHealthMonitor(lwir_mad_threshold=0.25, lwir_max_freeze_ticks=3)
    eo = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(240):
        if y < 120:
            eo[y, :, :] = (195 - y // 3, 175 - y // 4, 150 - y // 4)
        else:
            eo[y, :, :] = (65 + (y % 17), 90 + (y % 13), 75 + (y % 11))
    cv2.rectangle(eo, (90, 125), (230, 225), (58, 62, 68), -1)
    for x in range(100, 225, 16):
        cv2.line(eo, (x, 128), (x, 222), (225, 230, 235), 2, cv2.LINE_AA)
    for _ in range(30):
        px, py = int(rng.integers(15, 305)), int(rng.integers(15, 225))
        rad = int(rng.integers(3, 10))
        col = tuple(int(c) for c in rng.integers(30, 220, size=3))
        cv2.circle(eo, (px, py), rad, col, 2, cv2.LINE_AA)
    lwir = np.full((128, 160), 80, dtype=np.uint8)
    cv2.rectangle(lwir, (25, 30), (65, 80), 190, -1)
    cv2.circle(lwir, (110, 60), 14, 225, -1)

    for k in range(4):
        rep = mon.assess_thermal_pair(eo, lwir.copy(), frame_ts_ms=1000.0 + k * 100.0, imu_ts_ms=996.0 + k * 100.0)
    assert rep["lwir_nuc_frozen"] is True
    assert rep["q_lwir_effective"] == 0.0
    assert rep["thermal_mode"] == "LWIR_DEGRADED_EO_PRIMARY"

    # 3. Hash-chained tape verification + 1-byte tamper rejection
    tape_path = tmp_path / "synthetic_compound_tape.ndjson"
    with ShadowModeTelemetryLogger(tape_path, trial_phase="bench_shadow_replay", hardware_tx_pin_removed=True) as log:
        for i in range(4):
            log.record_event(
                timestamp_mono_us=(i + 1) * 1_000_000,
                event_type="compound_4fault_epoch",
                direction="rx",
                payload={
                    "gnss_ne_m": [float(i * 20), float(i * 5)],
                    "vis_ne_m": [float(i * 18), 0.0],
                    "baro_down_m": -100.0,
                    "gnss_down_m": -100.0,
                    "q_lwir_effective": 0.0 if i >= 2 else 0.95,
                    "coarse_heading_inliers": int(res_sweep.n_geom_inliers),
                    "approach_active": i == 3,
                    "v_vpl_m": 15.0 if i == 3 else 5.0,
                },
            )
    ver = verify_shadow_log_chain(tape_path)
    assert ver.valid is True and ver.record_count == 4
    assert ver.tx_inhibited_all is True and ver.actuator_authority_all_false is True

    lines = tape_path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("compound_4fault_epoch", "compound_4fault_epocX", 1)
    tampered = tmp_path / "synthetic_compound_tampered.ndjson"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    assert verify_shadow_log_chain(tampered).valid is False


def test_live_compound_fault_sil_tape_and_report_if_present() -> None:
    """Verify live WSL2 ArduPlane 4.7.3 compound-fault mission tape & Rust --replay when present on disk."""
    if not (LIVE_COMPOUND_TAPE.is_file() and LIVE_COMPOUND_REPORT.is_file()):
        pytest.skip("Live WSL2 ArduPlane compound-fault artifacts are gitignored / not generated on this runner")

    rep = json.loads(LIVE_COMPOUND_REPORT.read_text(encoding="utf-8"))
    assert rep["schema"] == "openxvs.v112.compound_fault_sil_gate/1"
    assert rep["simulator_backend"] == "WSL2_ARDUPLANE_OFFICIAL_SITL"
    assert rep["pymavlink_imported"] is False
    assert rep["strictly_ned_zero_enu"] is True
    assert rep["all_pass"] is True
    assert all(rep["checks"].values())

    p1 = rep["phase1_simultaneous_compound_faults"]
    assert p1["simultaneous_4fault_epoch_observed"] is True
    assert p1["coarse_heading_90deg"]["naive_0deg_inliers"] < 12
    assert p1["coarse_heading_90deg"]["sweep_max_inliers"] >= 12
    assert p1["lwir_shutter_freeze"]["min_q_lwir_effective"] == 0.0
    assert p1["daa_under_compound_faults"]["min_vfr_hud_ias_mps"] >= 17.0
    assert p1["daa_under_compound_faults"]["wire_msg84_seen"] == 0
    assert rep["phase4_teardown_hygiene"]["leftover_wsl_arduplane_pids"] == []

    py_ver = verify_shadow_log_chain(LIVE_COMPOUND_TAPE)
    assert py_ver.valid is True and py_ver.record_count >= 12
    assert py_ver.tx_inhibited_all is True and py_ver.actuator_authority_all_false is True

    if EDGE_RELEASE_EXE.is_file():
        rs = subprocess.run(
            [str(EDGE_RELEASE_EXE), "--replay", str(LIVE_COMPOUND_TAPE)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert rs.returncode == 0
        rs_doc = json.loads(rs.stdout)
        assert rs_doc["valid"] is True
        assert rs_doc["head_hash"] == py_ver.head_hash
        assert rs_doc["record_count"] == py_ver.record_count
        assert rs_doc["cusum_alarms"] > 0
        assert rs_doc["rta_mandatory_go_around_count"] > 0

        # 1-byte tamper must exit rc=1 on Rust --replay
        orig_lines = LIVE_COMPOUND_TAPE.read_text(encoding="utf-8").splitlines()
        orig_lines[2] = orig_lines[2].replace("phase0_crosswind_baseline", "phase0_crosswind_baselinX", 1)
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8", newline="\n") as tf:
            tf.write("\n".join(orig_lines) + "\n")
            tpath = Path(tf.name)
        try:
            rs_t = subprocess.run(
                [str(EDGE_RELEASE_EXE), "--replay", str(tpath)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert rs_t.returncode == 1
            assert json.loads(rs_t.stdout)["valid"] is False
        finally:
            tpath.unlink(missing_ok=True)


# ==============================================================================
# 3. Full-Stack GitHub Actions CI YAML Validation
# ==============================================================================


def test_fullstack_ci_workflow_yaml_and_coverage() -> None:
    """Validate `.github/workflows/xvs_core_ci.yml` syntax and full-stack 26/26 Kani + edge-infer + Studio + pytest."""
    assert CI_WORKFLOW_PATH.is_file()
    raw_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    # Validate YAML syntax via PyYAML (if installed in active env)
    try:
        import yaml  # type: ignore[import-untyped]

        parsed = yaml.safe_load(raw_text)
        assert isinstance(parsed, dict)
        assert "jobs" in parsed and isinstance(parsed["jobs"], dict)
        assert set(parsed["jobs"].keys()) >= {"rust-core-and-kani", "edge-infer-rust", "studio-frontend"}
        # Note: PyYAML 1.1 parses bare `on:` as boolean True
        triggers = parsed.get("on") if "on" in parsed else parsed.get(True)
        assert isinstance(triggers, dict)
        for trig_name in ("push", "pull_request"):
            paths = triggers[trig_name]["paths"]
            for req_p in (
                "xvs-multiship-coord/**",
                "docker/edge-infer/**",
                "apps/studio/**",
                "src/xvs/**",
                "tests/**",
            ):
                assert req_p in paths, f"Missing CI path trigger {req_p!r} under {trig_name}"
    except ImportError:
        pass

    # Structural & command assertions across all 3 jobs
    for required_token in (
        "xvs-multiship-coord/Cargo.toml",
        "--target thumbv7em-none-eabi",
        "cargo kani --manifest-path xvs-multiship-coord/Cargo.toml",
        "26/26 Kani",
        "edge-infer-rust:",
        "cargo clippy --manifest-path docker/edge-infer/Cargo.toml --all-targets --locked -- -D warnings",
        "cargo test --manifest-path docker/edge-infer/Cargo.toml --locked",
        "--replay",
        "samples/mission_flight_tape.ndjson",
        "studio-frontend:",
        "apps/studio",
        'node-version: "20"',
        "npm ci",
        "npx tsc --noEmit",
        "npm test",
        "npm run build",
        "pytest -q",
    ):
        assert required_token in raw_text, f"Missing expected CI workflow clause: {required_token!r}"


# ==============================================================================
# 4. Jetson Orin Day-1 TensorRT Builder Script Static & Syntax Checks
# ==============================================================================


def test_jetson_orin_build_trt_engines_script_static_and_bash_syntax() -> None:
    """Verify `docker/edge-infer/build_trt_engines.sh` LF hygiene, bash -n syntax, 6 models, v1.1+extended & SHA-256."""
    assert TRT_SCRIPT_PATH.is_file()
    raw_bytes = TRT_SCRIPT_PATH.read_bytes()
    assert b"\r" not in raw_bytes, "build_trt_engines.sh MUST use Unix LF line endings (no CRLF)"

    text = raw_bytes.decode("utf-8")
    for required_token in (
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "NON_FLIGHT_SIM_ONLY",
        "NVIDIA TensorRT SLA",
        "trtexec",
        "--fp16",
        "--int8",
        "--calib=",
        "SHA256_MISMATCH_REJECTED",
        "xvs_d2_web_fp16.onnx",
        "xvs_d2_v53_fp16.onnx",
        "x_t:1x6x256x256,h_prev_warped:1x24x64x64",
        "x_t:1x6x640x640,h_prev_warped:1x24x160x160",
        "xvs_land_v52_fp16.onnx",
        "xvs_land_v53_fp16.onnx",
        "xvs_thermal_v52_fp16.onnx",
        "xvs_thermal_v53_fp16.onnx",
        "1.13MiB",
        "xvs_nav_v5_fazb_fp16.onnx",
        "xfeat_320x256_fp16.onnx",
        "lighterglue_k256_fp16.onnx",
        "--percentile=50,95,99",
        "nvidia-smi",
        "tegrastats",
        "trap cleanup EXIT INT TERM",
    ):
        assert required_token in text, f"Missing required clause in build_trt_engines.sh: {required_token!r}"

    bash_exe = _find_bash_executable()
    if bash_exe is not None:
        proc = subprocess.run([bash_exe, "-n", str(TRT_SCRIPT_PATH)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"
