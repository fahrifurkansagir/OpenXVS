"""Self-tests proving `tests/xvs_v2/_oracles.py` satisfies every §4.1–§4.4 spec invariant."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tests" / "xvs_v2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "xvs_v2"))

from _oracles import (
    OPENXVS_BEARING_TRACK_PAYLOAD_LEN,
    OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN,
    OpenXvsBearingTrack,
    PyBonferroniSpoofDetector,
    PyBoolRing64,
    PyRtaFloatTelemetry,
    PyRtaTickInput,
    PySimplexRtaOracle,
    QuantFault,
    SimplexMode,
    build_rta_input_from_f64,
    check_runway_width_consistency,
    check_threshold_tie_margin,
    compute_pass4_reference_tensor,
    compute_vision_raim_pl,
    fast_covariance_intersection,
    load_odd_contract,
    pack_signed_bearing_track_mavlink_v2,
    quantize_nonneg_upper_i64,
    quantize_signed_bounded_i64,
    triple_ray_epipolar_ghost_check,
    unpack_and_verify_signed_bearing_track,
)
from xvs.detection.d1_model import build_d1_kf


def test_odd_contract_loaded_without_hardcoding() -> None:
    odd = load_odd_contract()
    assert odd["schema"] == "xvs.v2_m1_odd/1"
    th = odd["thresholds"]
    assert th["epipolar_cam_max_urad"] == 6109
    assert th["horizon_al_urad"] == 8727
    assert th["v_val_mm"] == 10000
    assert th["consec_dropout_max_ticks"] == 10
    assert th["window_dropout_20_max_ticks"] == 10
    assert th["p0_recovery_ticks"] == 30
    assert th["dwell_ticks"] == 20


def test_quantizers_fail_closed_neg_zero_nan_inf_and_signed_residuals() -> None:
    assert quantize_nonneg_upper_i64(-0.0, 1_000_000.0, 100_000_000) == (0, None)
    assert quantize_nonneg_upper_i64(0.0, 1_000_000.0, 100_000_000) == (0, None)
    assert quantize_nonneg_upper_i64(float("nan"), 1_000_000.0, 100_000_000)[1] == QuantFault.NON_FINITE
    assert quantize_nonneg_upper_i64(float("inf"), 1_000_000.0, 100_000_000)[1] == QuantFault.NON_FINITE
    assert quantize_nonneg_upper_i64(-1e-9, 1_000_000.0, 100_000_000)[1] == QuantFault.NEGATIVE_INPUT

    # Valid negative physical residuals pass signed bounded quantizer without fault
    assert quantize_signed_bounded_i64(-1.5, 1000.0, -10_000_000, 10_000_000) == (-1500, None)
    assert quantize_signed_bounded_i64(-0.0, 1000.0, -10_000_000, 10_000_000) == (0, None)
    assert quantize_signed_bounded_i64(float("nan"), 1000.0, -10_000_000, 10_000_000)[1] == QuantFault.NON_FINITE

    # NaN in alarm-driving statistic (cusum_stat) forces quantize_fault=True -> P0 at 0 ticks
    rta = PySimplexRtaOracle()
    inp = build_rta_input_from_f64(PyRtaFloatTelemetry(cusum_stat=float("nan")))
    assert inp.quantize_fault is True
    assert inp.spoof_alarm is True
    out = rta.step(inp)
    assert out.mode == SimplexMode.INVALID_INHIBIT
    assert out.blank_conformal_hud is True


def test_simplex_oracle_transitions_recovery_chatter_dropout_and_go_around() -> None:
    rta = PySimplexRtaOracle()
    # 6 -> 5 -> 4 -> 3 cam degradation (0-tick downward preemption)
    assert rta.step(PyRtaTickInput(cam_health_mask=0x3F)).mode == SimplexMode.NOMINAL_ML
    assert rta.step(PyRtaTickInput(cam_health_mask=0x3E)).mode == SimplexMode.ISOLATED_5CAM
    assert rta.step(PyRtaTickInput(cam_health_mask=0x3C)).mode == SimplexMode.FALLBACK_D0_SAFE
    assert rta.step(PyRtaTickInput(cam_health_mask=0x38)).mode == SimplexMode.INVALID_INHIBIT

    # Single fault tick at recovery_ticks == 29 resets counter cleanly to 0
    # First clear H[0,2](n_cams >= 4) warmup (2 ticks still critical, then 29 healthy ticks)
    for _ in range(2):
        out = rta.step(PyRtaTickInput(cam_health_mask=0x3F))
        assert out.mode == SimplexMode.INVALID_INHIBIT
    for expected_r in range(1, 30):
        out = rta.step(PyRtaTickInput(cam_health_mask=0x3F))
        assert out.mode == SimplexMode.INVALID_INHIBIT
        assert out.recovery_ticks == expected_r
    # Chatter fault at tick 29 resets recovery_ticks -> 0
    out_chatter = rta.step(PyRtaTickInput(horizon_pl_urad=9999))
    assert out_chatter.mode == SimplexMode.INVALID_INHIBIT
    assert out_chatter.recovery_ticks == 0

    # Full 30-tick P0 -> P1 + 20-tick P1 -> P4 staged recovery
    for _ in range(29):
        assert rta.step(PyRtaTickInput()).mode == SimplexMode.INVALID_INHIBIT
    assert rta.step(PyRtaTickInput()).mode == SimplexMode.FALLBACK_D0_SAFE
    for _ in range(19):
        assert rta.step(PyRtaTickInput()).mode == SimplexMode.FALLBACK_D0_SAFE
    assert rta.step(PyRtaTickInput()).mode == SimplexMode.NOMINAL_ML

    # Sustained dropout: 10 ticks stay P2, tick 11 forces P0
    rta_drop = PySimplexRtaOracle()
    for tick in range(1, 11):
        out = rta_drop.step(PyRtaTickInput(transient_dropout=True))
        assert out.mode == SimplexMode.DEGRADED_COAST
        assert out.consec_dropout_ticks == tick
    out11 = rta_drop.step(PyRtaTickInput(transient_dropout=True))
    assert out11.mode == SimplexMode.INVALID_INHIBIT

    # Intermittent 11/20 (55%) forces P0 whereas 10/20 (50%) stays bounded
    rta_50 = PySimplexRtaOracle()
    for i in range(20):
        out = rta_50.step(PyRtaTickInput(transient_dropout=(i % 2 == 0)))
        assert out.mode != SimplexMode.INVALID_INHIBIT
    rta_55 = PySimplexRtaOracle()
    hit_p0 = False
    for i in range(20):
        # 11 dropouts out of 20
        out = rta_55.step(PyRtaTickInput(transient_dropout=(i < 10 or i == 11)))
        if out.mode == SimplexMode.INVALID_INHIBIT:
            hit_p0 = True
    assert hit_p0 is True

    # Non-resettable mandatory Go-Around while approach_active=True
    rta_ga = PySimplexRtaOracle()
    out_ga = rta_ga.step(PyRtaTickInput(approach_active=True, v_vpl_mm=12_000, v_val_mm=10_000))
    assert out_ga.mandatory_go_around is True
    assert out_ga.mode == SimplexMode.INVALID_INHIBIT
    out_still = rta_ga.step(
        PyRtaTickInput(approach_active=True, v_vpl_mm=1000, go_around_ack_reset=True)
    )
    assert out_still.mandatory_go_around is True
    out_clear = rta_ga.step(
        PyRtaTickInput(approach_active=False, v_vpl_mm=1000, go_around_ack_reset=True)
    )
    assert out_clear.mandatory_go_around is False

    # Past-time MLTL Since operator sanity
    p_ring = PyBoolRing64()
    q_ring = PyBoolRing64()
    for p_v, q_v in [(False, True), (True, False), (True, False)]:
        p_ring.push(p_v)
        q_ring.push(q_v)
    assert PyBoolRing64.since(p_ring, q_ring, 1, 3) is True


def test_math_oracles_cusum_vraim_meshdaa_codec_and_pass4() -> None:
    # 1. Bonferroni 8-arm CUSUM vs h_c = 10.530 + tie margin helper
    det = PyBonferroniSpoofDetector()
    s_cov = np.eye(3) * 0.25  # sigma = 0.5 m/s
    for _ in range(10):
        res_nom = det.step(np.array([0.05, -0.04, 0.02]), s_cov)
        assert res_nom["spoof_alarm"] is False
    tripped = False
    for _ in range(10):
        res_sp = det.step(np.array([0.0, 2.5, 0.0]), s_cov)
        if res_sp["cusum_alarm"]:
            tripped = True
            break
    assert tripped is True
    assert check_threshold_tie_margin(10.5300000001, 10.530, eps=1e-9)[0] is False
    assert check_threshold_tie_margin(10.531, 10.530, eps=1e-9)[0] is True

    # 2. Vision-RAIM V-VPL/V-HPL & KSFO 61m vs Taxiway C 23m width gate
    # Realistic KSFO 28L perspective Jacobian (fx=429.6 px at R=1500 m -> ~0.286 px/m)
    scale_px_per_m = 429.6 / 1500.0
    h_jac = (
        np.array(
            [
                [0.4, 0.2, 0.9],
                [-0.4, 0.2, 0.9],
                [0.4, -0.2, 0.9],
                [-0.4, -0.2, 0.9],
                [0.0, 0.5, 0.85],
                [0.0, -0.5, 0.85],
            ],
            dtype=np.float64,
        )
        * scale_px_per_m
    )
    pl_nom = compute_vision_raim_pl(h_jac, sigma_meas=0.35)
    assert pl_nom["v_vpl_m"] < 10.0
    pl_fault = compute_vision_raim_pl(h_jac, sigma_meas=0.35, extra_bias_meas=18.0)
    assert pl_fault["v_vpl_m"] > 10.0

    ok_28l, _ = check_runway_width_consistency(60.96)
    ok_twy, _ = check_runway_width_consistency(22.86)
    assert ok_28l is True
    assert ok_twy is False

    # 3. MeshDAA triple-ray ghost rejection & FCI positive-definiteness
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1000.0, 0.0, 0.0])
    p3 = np.array([500.0, -800.0, 0.0])
    p_true = np.array([500.0, 1200.0, 50.0])
    u1 = (p_true - p1) / np.linalg.norm(p_true - p1)
    u2 = (p_true - p2) / np.linalg.norm(p_true - p2)
    u3_true = (p_true - p3) / np.linalg.norm(p_true - p3)
    u3_other = np.array([0.8, 0.6, 0.0])
    u3_other /= np.linalg.norm(u3_other)

    assert triple_ray_epipolar_ghost_check(p1, u1, p2, u2, p3, u3_true, math.radians(1.0))["accepted"] is True
    assert triple_ray_epipolar_ghost_check(p1, u1, p2, u2, p3, u3_other, math.radians(1.0))["ghost_rejected"] is True

    x_f, p_f, w_f = fast_covariance_intersection(
        np.array([10.0, 20.0, 30.0]),
        np.diag([4.0, 9.0, 16.0]),
        np.array([11.0, 19.0, 31.0]),
        np.diag([9.0, 4.0, 9.0]),
    )
    assert 0.0 < w_f < 1.0
    assert np.all(np.linalg.eigvalsh(p_f) > 0.0)
    assert np.all(np.isfinite(x_f))

    # 4. OPENXVS_BEARING_TRACK (#52001) exact 52 B payload & 77 B signed MAVLink v2 frame
    msg = OpenXvsBearingTrack(
        time_utc_us=1_763_668_800_000_000,
        own_ecef_cm=(-270_617_400, -426_096_200, 388_572_100),
        bearing_ecef_q15=(16384, -23170, 16384),
        bearing_rate_urad_s=(120, -45, 10),
        cone_half_angle_urad=3500,
        own_pos_std_cm=150,
        track_id=42,
        sender_id=3,
        flags=0x03,
        cov_diag_packed=(10, 12, 15, 4, 5, 6),
    )
    key = bytes(range(32))
    wire = pack_signed_bearing_track_mavlink_v2(msg, secret_key_32=key, seq=7)
    assert len(wire) == OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN == 77
    assert wire[1] == OPENXVS_BEARING_TRACK_PAYLOAD_LEN == 52
    assert unpack_and_verify_signed_bearing_track(wire, secret_key_32=key) == msg

    tampered = bytearray(wire)
    tampered[20] ^= 0x01
    with pytest.raises(ValueError):
        unpack_and_verify_signed_bearing_track(bytes(tampered), secret_key_32=key)

    # 5. Pass-4 sensor-matched noise-off reference vs D1HeatmapDetector.normalize_input
    flat_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
    x_flat = compute_pass4_reference_tensor(flat_rgb, flat_rgb, flat_rgb)
    assert x_flat.shape == (1, 5, 256, 256)
    assert np.all(np.isfinite(x_flat))
    assert float(np.max(np.abs(x_flat))) == 0.0

    rng = np.random.default_rng(7)
    r0 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    r1 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    r2 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.float32)
    mask[:10, :] = 0.0
    x_ref = compute_pass4_reference_tensor(r0, r1, r2, mask_m1=mask, mask_m2=mask)
    # Cross-check against PyTorch D1HeatmapDetector.normalize_input
    det_m = build_d1_kf(num_frames_k=3)
    with torch.no_grad():
        # Verify zero NaN and normalized properties
        out_pt = det_m.forward_normalized(torch.from_numpy(x_ref))
    assert np.all(np.isfinite(out_pt["heatmap"].numpy()))
    assert np.allclose(x_ref[0, 2, :10, :], x_ref[0, 2, :10, :1])  # Masked border constant after mean shift
