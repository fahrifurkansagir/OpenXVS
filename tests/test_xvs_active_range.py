"""Tests for `XVS-ActiveRange` Python mirror, Split Conformal, Review Focus #3 gating, and Tier B CBDR MC."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.v2.xvs_active_range import (
    FLAG_TAU_FUSED,
    NUM_BANKS,
    R_CEIL_M,
    R_FLOOR_M,
    WIRE_SIZE_BYTES,
    ActiveRangeConfig,
    ActiveRangeController,
    ActiveRangeMeasurement,
    ActiveRangeMode,
    OwnshipStateNed,
    R_INIT_M,
    compute_conformal_lower_bound,
    decode_wire,
    encode_wire,
    evaluate_conformal_coverage,
    fit_one_sided_conformal_quantile,
    select_active_swerve,
)
from xvs.v2.xvs_active_range.cbdr_benchmark import run_cbdr_monte_carlo

GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "active_range_vectors.json"


def _parse_golden_config(raw_cfg: dict[str, object] | None) -> ActiveRangeConfig:
    if raw_cfg is None:
        return ActiveRangeConfig()
    r_init = tuple(float(v) for v in raw_cfg["r_init_m"])  # type: ignore[union-attr]
    return ActiveRangeConfig(
        r_init_m=(r_init[0], r_init[1], r_init[2], r_init[3], r_init[4], r_init[5]),
        sigma_bearing_rad=float(raw_cfg["sigma_bearing_rad"]),  # type: ignore[arg-type]
        q_hat_conformal=float(raw_cfg["q_hat_conformal"]),  # type: ignore[arg-type]
    )


def test_python_matches_rust_golden_vectors() -> None:
    """Lock Python `ActiveRangeController` bit-for-bit against Rust `tests/golden/active_range_vectors.json`."""
    assert GOLDEN_PATH.is_file(), f"Missing Rust golden vectors at {GOLDEN_PATH}"
    doc = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    top_cfg = doc["config"] if "config" in doc else None
    trajectories = doc["trajectories"]
    assert len(trajectories) >= 1

    max_float_err = 0.0
    total_steps = 0

    for traj in trajectories:
        cfg = _parse_golden_config(traj["config"] if "config" in traj else top_cfg)
        ctrl = ActiveRangeController(cfg)
        for step_rec in traj["steps"]:
            own_raw = step_rec["own"]
            meas_raw = step_rec["meas"]
            exp = step_rec["out"]
            wire_hex_exp = str(step_rec["wire_hex"]).lower()

            own = OwnshipStateNed(
                v_ned=(float(own_raw["v_ned"][0]), float(own_raw["v_ned"][1]), float(own_raw["v_ned"][2])),
                a_ned=(float(own_raw["a_ned"][0]), float(own_raw["a_ned"][1]), float(own_raw["a_ned"][2])),
                heading_rad=float(own_raw["heading_rad"]),
                tas_mps=float(own_raw["tas_mps"]),
                v_stall_mps=float(own_raw["v_stall_mps"]),
            )
            meas = ActiveRangeMeasurement(
                alpha_rad=float(meas_raw["alpha_rad"]),
                beta_rad=float(meas_raw["beta_rad"]),
                size_px=float(meas_raw["size_px"]),
                size_rate_px_s=float(meas_raw["size_rate_px_s"]),
                contrast_ratio=float(meas_raw["contrast_ratio"]),
                sky_beta_ext=float(meas_raw["sky_beta_ext"]),
                daytime_clear=bool(meas_raw["daytime_clear"]),
                track_age=int(meas_raw["track_age"]),
                dt_s=float(meas_raw["dt_s"]),
            )
            tel = ctrl.step(own, meas)
            wire = encode_wire(tel)
            assert len(wire) == WIRE_SIZE_BYTES

            assert tel.seq == int(exp["seq"])
            assert int(tel.mode) == int(exp["mode"]), (
                f"Mode mismatch in {traj['name']} step {tel.seq}: py={tel.mode} rust={exp['mode']}"
            )
            assert tel.active_bank_idx == int(exp["active_bank_idx"]), (
                f"Bank idx mismatch in {traj['name']} step {tel.seq}: py={tel.active_bank_idx} rust={exp['active_bank_idx']}"
            )
            assert tel.flags == int(exp["flags"]), (
                f"Flags mismatch in {traj['name']} step {tel.seq}: py={tel.flags:#x} rust={int(exp['flags']):#x}"
            )

            float_pairs = (
                ("r_hat_m", tel.r_hat_m, float(exp["r_hat_m"])),
                ("sigma_r_m", tel.sigma_r_m, float(exp["sigma_r_m"])),
                ("r_safe_lower_m", tel.r_safe_lower_m, float(exp["r_safe_lower_m"])),
                ("rho_dot_inv_s", tel.rho_dot_inv_s, float(exp["rho_dot_inv_s"])),
                ("cmd_bank_rad", tel.cmd_bank_rad, float(exp["cmd_bank_rad"])),
                ("min_sc3bf_margin", tel.min_sc3bf_margin, float(exp["min_sc3bf_margin"])),
                ("fim_gain", tel.fim_gain, float(exp["fim_gain"])),
            )
            for fname, py_v, rs_v in float_pairs:
                err = abs(py_v - rs_v)
                max_float_err = max(max_float_err, err)
                assert err < 1.0e-5, (
                    f"{traj['name']} step {tel.seq} {fname}: py={py_v} vs rust={rs_v} (err={err})"
                )

            for m in range(NUM_BANKS):
                w_err = abs(tel.weights[m] - float(exp["weights"][m]))
                max_float_err = max(max_float_err, w_err)
                assert w_err < 1.0e-5, (
                    f"{traj['name']} step {tel.seq} weight[{m}]: py={tel.weights[m]} vs rust={exp['weights'][m]}"
                )

            assert tel.crc32 == int(exp["crc32"]), (
                f"CRC32 mismatch in {traj['name']} step {tel.seq}: py={tel.crc32:#010x} rust={int(exp['crc32']):#010x}"
            )
            assert wire.hex() == wire_hex_exp

            decoded = decode_wire(wire)
            assert decoded is not None
            assert decoded.crc32 == tel.crc32
            total_steps += 1

    assert total_steps >= 10
    assert max_float_err < 1.0e-5


def test_zero_closure_and_sub_6px_dot_gating() -> None:
    """Review Focus #3: sub-6px dot gates tau off cleanly; zero-closure crossing target never triggers ActiveSwerve."""
    ctrl = ActiveRangeController(ActiveRangeConfig())
    own = OwnshipStateNed(
        v_ned=(40.0, 0.0, 0.0),
        a_ned=(0.0, 0.0, 0.0),
        heading_rad=0.0,
        tas_mps=40.0,
        v_stall_mps=18.0,
    )

    # 1. Distant sub-6px dot (s = 4.0 px, s_dot = 0.0) across 8 frames
    for age in range(1, 9):
        tel = ctrl.step(
            own,
            ActiveRangeMeasurement(
                alpha_rad=0.05,
                beta_rad=0.02,
                size_px=4.0,
                size_rate_px_s=0.0,
                track_age=age,
                dt_s=0.1,
            ),
        )
        assert (tel.flags & FLAG_TAU_FUSED) == 0, "tau must NOT fuse when size_px < 6.0"
        assert math.isfinite(tel.r_hat_m) and R_FLOOR_M <= tel.r_hat_m <= R_CEIL_M
        assert math.isfinite(tel.rho_dot_inv_s)
        assert tel.mode == ActiveRangeMode.NOMINAL_HOLD

    # 2. Larger zero-closure crossing target (s = 12.0 px, s_dot = 0.0) -> tau fuses 0 without div-by-zero, stays NOMINAL_HOLD
    ctrl_cross = ActiveRangeController(ActiveRangeConfig())
    for age in range(1, 10):
        tel = ctrl_cross.step(
            own,
            ActiveRangeMeasurement(
                alpha_rad=0.05 + 0.0005 * age,
                beta_rad=0.02,
                size_px=12.0,
                size_rate_px_s=0.0,
                track_age=4 + age,
                dt_s=0.1,
            ),
        )
        assert (tel.flags & FLAG_TAU_FUSED) != 0, "tau should cleanly fuse z_tau=0 when s >= 6 px and age >= 5"
        assert math.isfinite(tel.r_hat_m) and math.isfinite(tel.rho_dot_inv_s)
        assert abs(tel.rho_dot_inv_s) < 0.05
        assert tel.mode == ActiveRangeMode.NOMINAL_HOLD
        assert tel.cmd_bank_rad == 0.0


def test_split_conformal_quantile_achieves_finite_sample_coverage() -> None:
    """Synthetic heavy-tailed overestimation residuals achieve >= 93% finite-sample coverage for alpha = 0.05."""
    rng = np.random.default_rng(20261121)
    n_cal, n_test = 400, 400
    r_true_cal = rng.uniform(200.0, 2000.0, size=n_cal)
    sigma_cal = rng.uniform(30.0, 120.0, size=n_cal)
    # Heavy-tailed Student-t(3) overestimation errors
    noise_cal = rng.standard_t(df=3.0, size=n_cal) * sigma_cal
    r_hat_cal = np.clip(r_true_cal + noise_cal, R_FLOOR_M, R_CEIL_M)

    q_hat = fit_one_sided_conformal_quantile(r_hat_cal, r_true_cal, sigma_cal, alpha=0.05)
    assert math.isfinite(q_hat) and q_hat > 0.0

    r_true_test = rng.uniform(200.0, 2000.0, size=n_test)
    sigma_test = rng.uniform(30.0, 120.0, size=n_test)
    r_hat_test = np.clip(r_true_test + rng.standard_t(df=3.0, size=n_test) * sigma_test, R_FLOOR_M, R_CEIL_M)

    cov = evaluate_conformal_coverage(r_hat_test, r_true_test, sigma_test, q_hat)
    assert cov["empirical_coverage"] >= 0.93
    for i in range(n_test):
        low = compute_conformal_lower_bound(float(r_hat_test[i]), float(sigma_test[i]), q_hat)
        assert R_FLOOR_M <= low <= float(r_hat_test[i])


def test_wire_64b_roundtrip_and_crc_tamper() -> None:
    """Verify exact 64-byte wire encoding/decoding and single-bit CRC32 tamper rejection."""
    ctrl = ActiveRangeController()
    own = OwnshipStateNed(v_ned=(40.0, 0.0, 0.0), a_ned=(0.0, 0.0, 0.0), heading_rad=0.0, tas_mps=40.0)
    tel = ctrl.step(own, ActiveRangeMeasurement(alpha_rad=0.0, beta_rad=0.0, size_px=10.0, size_rate_px_s=0.5, track_age=6, dt_s=0.1))
    raw = encode_wire(tel)
    assert len(raw) == WIRE_SIZE_BYTES
    decoded = decode_wire(raw)
    assert decoded is not None
    assert decoded.seq == tel.seq
    assert decoded.mode == tel.mode
    assert decoded.crc32 == tel.crc32

    tampered = bytearray(raw)
    tampered[12] ^= 0x01
    assert decode_wire(bytes(tampered)) is None


def test_counterexample_in_cone_near_fov_edge_escalates_to_immediate_evade() -> None:
    """D16-P1-M1 regression: violated collision cone near FOV edge must reject phi=0 and escalate to ImmediateEvade."""
    alpha_rad = math.radians(46.8)
    dec = select_active_swerve(
        r_banks=R_INIT_M,
        weights=(1.0 / 6.0,) * NUM_BANKS,
        r_hat_m=600.0,
        sigma_r_m=300.0,
        r_safe_lower_m=155.0,
        rho_dot_inv_s=-0.19,
        alpha_dot=0.0,
        beta_dot=0.0,
        alpha_body_rad=alpha_rad,
        alpha_rad=alpha_rad,
        beta_rad=0.0,
        heading_rad=0.0,
        tas_mps=35.0,
        v_stall_mps=14.5,
        sigma_bearing_rad=0.00055,
        cbdr_probe_active=False,
        was_locked=False,
        preferred_swerve_sign=1.0,
    )
    assert dec.mode == ActiveRangeMode.IMMEDIATE_EVADE
    assert abs(dec.cmd_bank_rad) > 0.4
    assert dec.cmd_bank_rad < 0.0  # turn away from starboard (+46.8 deg) intruder
    assert dec.sc3bf_all_safe is False


def test_cbdr_monte_carlo_small_n_beats_baselines() -> None:
    """Fast CI gate (`N = 20`) verifying Tier B 6-DoF CBDR range collapse and safety invariants."""
    res = run_cbdr_monte_carlo(n_trials=20, seed=20261121)
    xvs = res["methods"]["xvs_active_range"]
    straight = res["methods"]["straight_passive"]
    assert float(xvs["rmse_pct"]) < 10.0
    assert float(straight["rmse_pct"]) > 50.0
    assert int(xvs["collisions"]) == 0
    assert int(xvs["fov_drops"]) == 0
    assert float(xvs["sc3bf_invariance_rate"]) == 1.0
