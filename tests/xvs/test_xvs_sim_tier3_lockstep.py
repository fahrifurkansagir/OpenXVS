"""Tests for DR-12 clean-room Apache-2.0 480 Hz Tier-3 lockstep clock & SIM_JSON UDP bridge."""

from __future__ import annotations

import json
import math
import struct

import pytest

from xvs.v2.xvs_sim_tier3 import (
    LWIR_RATE_HZ,
    SERVO_PACKET_16_MAGIC,
    SERVO_PACKET_16_SIZE,
    SERVO_PACKET_32_MAGIC,
    SERVO_PACKET_32_SIZE,
    SIM_RATE_HZ,
    VOX_TAU_S,
    MasterLockstepClock,
    SimJsonPhysicsState,
    SimJsonProtocolError,
    SimJsonServoPacket,
    Tier3SafetyLockError,
    VoxBolometerIIRFilter,
    run_tier3_udp_lockstep_loopback,
    validate_arduplane_version,
    validate_gz_sim_version,
)


def test_servo_packet_16_and_32_roundtrip_and_bad_magic_rejection() -> None:
    """Verify ArduPilot SIM_JSON.h binary 16-ch (40 B) and 32-ch (72 B) servo codecs."""
    pwm16 = tuple(1100 + 25 * i for i in range(16))
    pkt16 = SimJsonServoPacket(
        magic=SERVO_PACKET_16_MAGIC,
        frame_rate=SIM_RATE_HZ,
        frame_count=42,
        pwm=pwm16,
    )
    raw16 = pkt16.pack()
    assert len(raw16) == SERVO_PACKET_16_SIZE == 40
    assert SimJsonServoPacket.unpack(raw16) == pkt16

    pwm32 = tuple(1000 + 15 * i for i in range(32))
    pkt32 = SimJsonServoPacket(
        magic=SERVO_PACKET_32_MAGIC,
        frame_rate=SIM_RATE_HZ,
        frame_count=99999,
        pwm=pwm32,
    )
    raw32 = pkt32.pack()
    assert len(raw32) == SERVO_PACKET_32_SIZE == 72
    assert SimJsonServoPacket.unpack(raw32) == pkt32

    # Bad magic in 40 B and 72 B buffers must raise SimJsonProtocolError
    bad40 = struct.pack("<HHI16H", 12345, 480, 1, *pwm16)
    bad72 = struct.pack("<HHI32H", 54321, 480, 1, *pwm32)
    with pytest.raises(SimJsonProtocolError, match="magic"):
        SimJsonServoPacket.unpack(bad40)
    with pytest.raises(SimJsonProtocolError, match="magic"):
        SimJsonServoPacket.unpack(bad72)

    # Wrong byte length must fail closed
    with pytest.raises(SimJsonProtocolError, match="byte length"):
        SimJsonServoPacket.unpack(raw16[:-2])


def test_master_lockstep_clock_480hz_exact_divisors_and_zero_drift_over_10s() -> None:
    """Verify exact integer tick divisors and zero EO/LWIR phase drift over 4,800 ticks (10.0 s)."""
    with pytest.raises(Tier3SafetyLockError, match="480 Hz"):
        MasterLockstepClock(rate_hz=400)

    clock = MasterLockstepClock()
    prev_ts = 0.0
    plant_count = 0
    evis_count = 0
    lwir_count = 0
    eo_count = 0
    last_lwir_ts = -1.0

    for k in range(1, 4801):
        tick = clock.advance()
        assert tick.tick_index == k
        assert tick.timestamp_s > prev_ts
        assert tick.timestamp_s == pytest.approx(k / 480.0, abs=1e-12)
        prev_ts = tick.timestamp_s

        if tick.plant_due:
            plant_count += 1
        if tick.evis_due:
            evis_count += 1
        if tick.lwir_due:
            lwir_count += 1
            last_lwir_ts = tick.timestamp_s
        if tick.eo_autonomy_due:
            eo_count += 1
            # Every EO+Autonomy tick MUST be mathematically co-stamped with LWIR (zero drift)
            assert tick.lwir_due is True
            assert tick.timestamp_s == last_lwir_ts

    assert clock.timestamp_s == pytest.approx(10.0, abs=1e-12)
    assert plant_count == 4800
    assert evis_count == 2400
    assert lwir_count == 600
    assert eo_count == 300


def test_issue_29113_airspeed_wind_and_lockstep_flag_guards() -> None:
    """Reject missing airspeed/velocity_wind (#29113) and forbidden no_lockstep/no_time_sync flags."""
    valid = SimJsonPhysicsState(
        timestamp=1.0 / 480.0,
        imu_gyro=(0.0, 0.0, 0.0),
        imu_accel_body=(0.0, 0.0, -9.80665),
        position=(0.0, 0.0, -100.0),
        attitude=(0.0, -0.05, 5.19),
        velocity=(17.0, -32.0, 1.8),
        airspeed=36.2,
        velocity_wind=(3.0, -1.5, 0.0),
    )
    wire = valid.encode_json_bytes()
    assert wire.endswith(b"\n")
    assert SimJsonPhysicsState.decode_json_bytes(wire) == valid

    base_dict = valid.to_payload_dict()

    # Missing airspeed -> blocked by Issue #29113 guard
    no_air = dict(base_dict)
    del no_air["airspeed"]
    with pytest.raises(Tier3SafetyLockError, match="29113"):
        SimJsonPhysicsState.decode_json_bytes((json.dumps(no_air) + "\n").encode("utf-8"))

    # Missing velocity_wind -> blocked by Issue #29113 guard
    no_wind = dict(base_dict)
    del no_wind["velocity_wind"]
    with pytest.raises(Tier3SafetyLockError, match="29113"):
        SimJsonPhysicsState.decode_json_bytes((json.dumps(no_wind) + "\n").encode("utf-8"))

    # Forbidden no_lockstep=True or no_time_sync=True rejected in constructor AND decoder
    with pytest.raises(Tier3SafetyLockError, match="no_lockstep"):
        SimJsonPhysicsState(
            timestamp=0.01,
            imu_gyro=(0.0, 0.0, 0.0),
            imu_accel_body=(0.0, 0.0, -9.81),
            position=(0.0, 0.0, -100.0),
            attitude=(0.0, 0.0, 0.0),
            velocity=(20.0, 0.0, 0.0),
            airspeed=20.0,
            velocity_wind=(0.0, 0.0, 0.0),
            no_lockstep=True,
        )
    with pytest.raises(Tier3SafetyLockError, match="no_time_sync"):
        SimJsonPhysicsState(
            timestamp=0.01,
            imu_gyro=(0.0, 0.0, 0.0),
            imu_accel_body=(0.0, 0.0, -9.81),
            position=(0.0, 0.0, -100.0),
            attitude=(0.0, 0.0, 0.0),
            velocity=(20.0, 0.0, 0.0),
            airspeed=20.0,
            velocity_wind=(0.0, 0.0, 0.0),
            no_time_sync=True,
        )

    for bad_flag in ("no_lockstep", "no_time_sync"):
        bad_payload = dict(base_dict)
        bad_payload[bad_flag] = True
        with pytest.raises(Tier3SafetyLockError, match=bad_flag):
            SimJsonPhysicsState.decode_json_bytes((json.dumps(bad_payload) + "\n").encode("utf-8"))


def test_arduplane_and_gz_sim_version_guards() -> None:
    """Enforce ArduPlane >= 4.7.3 (#32506 + #29113) and gz-sim >= 8.15.0 (#3695)."""
    for bad_ap in ((4, 6, 3), (4, 7, 0), (4, 7, 2)):
        with pytest.raises(Tier3SafetyLockError, match="4.7.3"):
            validate_arduplane_version(bad_ap)
    validate_arduplane_version((4, 7, 3))
    validate_arduplane_version((4, 8, 0))

    for bad_gz in ((8, 0, 0), (8, 14, 9)):
        with pytest.raises(Tier3SafetyLockError, match="8.15.0"):
            validate_gz_sim_version(bad_gz)
    validate_gz_sim_version((8, 15, 0))
    validate_gz_sim_version((9, 0, 0))


def test_vox_bolometer_8ms_iir_step_response() -> None:
    """Verify FLIR Boson+ 8.0 ms VOx IIR filter alpha (~0.8755 @ 60 Hz) & exponential convergence."""
    filt = VoxBolometerIIRFilter(tau_s=VOX_TAU_S, dt_s=1.0 / LWIR_RATE_HZ)
    expected_alpha = 1.0 - math.exp(-(1.0 / 60.0) / 0.008)
    assert filt.alpha == pytest.approx(expected_alpha, rel=1e-9)
    assert 0.875 < filt.alpha < 0.876

    # Initialize at baseline 0.0, then apply unit step 1.0
    assert filt.step(0.0) == pytest.approx(0.0)
    y1 = filt.step(1.0)
    assert y1 == pytest.approx(expected_alpha, abs=1e-9)
    y2 = filt.step(1.0)
    assert y2 == pytest.approx(1.0 - (1.0 - expected_alpha) ** 2, abs=1e-9)
    y5 = y2
    for _ in range(3):
        y5 = filt.step(1.0)
    assert y5 == pytest.approx(1.0 - (1.0 - expected_alpha) ** 5, abs=1e-9)
    assert y5 > 0.9999


def test_real_udp_loopback_480_ticks_and_duplicate_frame_drain() -> None:
    """Run real local 127.0.0.1 UDP closed-loop lockstep (480 ticks = 1.0 s) + duplicate drain."""
    summary_16 = run_tier3_udp_lockstep_loopback(
        num_ticks=480,
        use_servo_32=False,
        inject_duplicate_frame_at=120,
    )
    assert summary_16["sim_rate_hz"] == 480
    assert summary_16["plant_steps"] == 480
    assert summary_16["final_timestamp_s"] == pytest.approx(1.0, abs=1e-9)
    assert summary_16["evis_ticks"] == 240
    assert summary_16["lwir_ticks"] == 60
    assert summary_16["eo_autonomy_ticks"] == 30
    assert summary_16["costamped_eo_lwir_matches"] == 30
    assert summary_16["max_phase_drift_s"] == 0.0
    assert summary_16["duplicate_packets_drained"] == 1
    assert summary_16["sitl_received_count"] == 480
    assert summary_16["timestamps_strictly_monotonic"] is True
    assert summary_16["all_states_have_airspeed_and_wind"] is True
    assert 41.99 < float(summary_16["vox_final_radiance"]) <= 42.0

    # Also verify 32-channel servo_packet_32 path over real UDP loopback
    summary_32 = run_tier3_udp_lockstep_loopback(
        num_ticks=48,
        use_servo_32=True,
        inject_duplicate_frame_at=16,
    )
    assert summary_32["plant_steps"] == 48
    assert summary_32["lwir_ticks"] == 6
    assert summary_32["eo_autonomy_ticks"] == 3
    assert summary_32["costamped_eo_lwir_matches"] == 3
    assert summary_32["duplicate_packets_drained"] == 1
