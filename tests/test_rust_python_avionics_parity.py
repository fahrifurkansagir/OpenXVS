"""Faz 1 Cross-Verification: Pure-Rust vs Python Avionics Parity (< 1e-6) & Edge-Infer CLI Proof.

Verifies bit-near (< 1e-6) parity between Python reference implementations and Pure-Rust
`openxvs-edge-infer.exe` (`nav_integrity.rs`, byte-identical mirror of `apps/studio/src-tauri/src/nav_integrity.rs`):
  1. Byte-for-byte identity between Studio and Edge-Infer `nav_integrity.rs`.
  2. `SequenceVprFilter`: nominal W=7 (Bayesian & Viterbi), aliasing/flat rejection, all-zero blind,
     constant 0.50, low-signal, NaN/Inf fail-closed, post-blind recovery, and 6-stage aliasing gate
     -> max abs diff < 1e-6 and 100% exact boolean/index/reason-code match.
  3. `DualBankEsEkfCusum`: F1..F5 correlated multipath (0 FA @ 1 Hz & 10 Hz), 0.05..5.0 m/s
     boiled-frog ramps (100% det < 30 m HAL), 5..100 m steps, vertical Down baro-turbulence /
     ramps / steps / NaN fail-closed, and 15-epoch hysteresis reintegration
     -> max abs diff < 1e-6 and 100% exact alarm-tick match.
  4. `openxvs-edge-infer.exe --replay`: Pure-Rust SHA-256 chain verification & telemetry replay of
     Python `ShadowModeTelemetryLogger` NDJSON logs + non-zero exit on ANY 1-byte tamper / forged TX.
  5. `openxvs-edge-infer.exe` zero-arg `--self-check` backward compatibility & `--soak` sanity check.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.v2.studio_bridge.hil_bridge import (  # noqa: E402
    SHADOW_GENESIS_HASH,
    ShadowModeTelemetryLogger,
    verify_shadow_log_chain,
)
from xvs.v2.xvs_nav import (  # noqa: E402
    DualBankConfig,
    DualBankEsEkfCusum,
    SequenceVprConfig,
    SequenceVprFilter,
    VisualAliasingThresholds,
    VisualFixMetrics,
)

STUDIO_NAV_INTEGRITY_RS = REPO_ROOT / "apps" / "studio" / "src-tauri" / "src" / "nav_integrity.rs"
EDGE_NAV_INTEGRITY_RS = REPO_ROOT / "docker" / "edge-infer" / "src" / "nav_integrity.rs"
EDGE_RELEASE_EXE = REPO_ROOT / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe"
EDGE_DEBUG_EXE = REPO_ROOT / "docker" / "edge-infer" / "target" / "debug" / "openxvs-edge-infer.exe"

PARITY_TOL: float = 1e-6


def _resolve_edge_exe() -> Path:
    if EDGE_RELEASE_EXE.is_file():
        return EDGE_RELEASE_EXE
    if EDGE_DEBUG_EXE.is_file():
        return EDGE_DEBUG_EXE
    pytest.fail(
        f"openxvs-edge-infer.exe not built at {EDGE_RELEASE_EXE} or {EDGE_DEBUG_EXE}; "
        "build with cargo build --manifest-path docker/edge-infer/Cargo.toml --release --locked"
    )


def _enc_f64(val: float) -> Any:
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Inf" if val > 0.0 else "-Inf"
    return float(val)


def _dec_f64(val: Any) -> float:
    if isinstance(val, str):
        if val == "NaN":
            return float("nan")
        if val in ("Inf", "+Inf", "Infinity"):
            return float("inf")
        if val in ("-Inf", "-Infinity"):
            return float("-inf")
        raise ValueError(f"Unexpected string float token: {val!r}")
    return float(val)


def _compare_f64(py_val: float, rs_val: float, label: str, max_diff_box: list[float]) -> None:
    if math.isnan(py_val) or math.isnan(rs_val):
        assert math.isnan(py_val) and math.isnan(rs_val), f"{label}: NaN mismatch py={py_val} vs rs={rs_val}"
        return
    if math.isinf(py_val) or math.isinf(rs_val):
        assert py_val == rs_val, f"{label}: Inf mismatch py={py_val} vs rs={rs_val}"
        return
    diff = abs(py_val - rs_val)
    if diff > max_diff_box[0]:
        max_diff_box[0] = diff
    assert diff < PARITY_TOL, f"{label}: |py={py_val:.12g} - rs={rs_val:.12g}| = {diff:.3e} >= {PARITY_TOL}"


def _run_parity_eval(payload: dict[str, Any], tmp_path: Path, name: str) -> dict[str, Any]:
    exe = _resolve_edge_exe()
    trace_path = tmp_path / f"{name}.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        [str(exe), "--parity-eval", str(trace_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"--parity-eval failed (rc={proc.returncode}): {proc.stderr}\n{proc.stdout}"
    return json.loads(proc.stdout)


def _make_grid_ned(rows: int, cols: int, spacing_m: float = 150.0) -> tuple[np.ndarray, list[int]]:
    coords = [[-r * spacing_m, c * spacing_m] for r in range(rows) for c in range(cols)]
    path: list[int] = []
    for r in range(rows):
        col_iter = range(cols) if (r % 2 == 0) else range(cols - 1, -1, -1)
        for c in col_iter:
            path.append(r * cols + c)
    return np.asarray(coords, dtype=np.float64), path


def test_studio_and_edge_nav_integrity_rs_are_byte_identical() -> None:
    assert STUDIO_NAV_INTEGRITY_RS.is_file(), f"Missing {STUDIO_NAV_INTEGRITY_RS}"
    assert EDGE_NAV_INTEGRITY_RS.is_file(), f"Missing {EDGE_NAV_INTEGRITY_RS}"
    assert STUDIO_NAV_INTEGRITY_RS.read_bytes() == EDGE_NAV_INTEGRITY_RS.read_bytes(), (
        "apps/studio/src-tauri/src/nav_integrity.rs and docker/edge-infer/src/nav_integrity.rs drifted!"
    )


def test_sequence_vpr_python_rust_parity_under_1e6(tmp_path: Path) -> None:
    coords_8x8, path_8x8 = _make_grid_ned(8, 8, spacing_m=153.4)
    rng = np.random.default_rng(20261124)

    # Case 1 & 2: Nominal noisy multi-year W=7 trajectory (Bayesian + Viterbi)
    n_tiles = coords_8x8.shape[0]
    traj_steps: list[dict[str, Any]] = []
    for t, gt_idx in enumerate(path_8x8[:32]):
        sims = rng.normal(0.18, 0.07, size=n_tiles)
        sims[gt_idx] = float(rng.normal(0.46, 0.06))
        if rng.random() < 0.50:
            dist_id = int(rng.choice([i for i in range(n_tiles) if i != gt_idx]))
            sims[dist_id] = sims[gt_idx] + float(rng.uniform(0.03, 0.12))
        sims = np.clip(sims, -0.95, 0.99)
        if t == 0:
            delta = np.zeros(2, dtype=np.float64)
        else:
            delta = (coords_8x8[gt_idx] - coords_8x8[path_8x8[t - 1]]) + rng.normal(0.0, 12.0, size=2)
        traj_steps.append(
            {
                "similarities": [_enc_f64(x) for x in sims.tolist()],
                "delta_ne_odom_m": [_enc_f64(delta[0]), _enc_f64(delta[1])],
                "odom_sigma_m": 25.0,
            }
        )

    # Case 3: Edge cases — W=7 prior -> all_zero -> constant 0.50 -> low-signal -> NaN -> Inf odom -> bad sigma ->
    # aliased flat -> post-blind recovery -> reset -> 6-stage VisualFixMetrics pass + HPL>HAL integrity alert
    coords_4x4, path_4x4 = _make_grid_ned(4, 4, spacing_m=150.0)
    edge_steps: list[dict[str, Any]] = []
    for i in range(7):
        cur = path_4x4[i]
        delta = coords_4x4[cur] - coords_4x4[path_4x4[i - 1]] if i > 0 else np.zeros(2)
        sims = np.full(16, 0.15, dtype=np.float64)
        sims[cur] = 0.88
        edge_steps.append(
            {
                "similarities": sims.tolist(),
                "delta_ne_odom_m": delta.tolist(),
                "odom_sigma_m": 20.0,
            }
        )
    delta_next = (coords_4x4[path_4x4[7]] - coords_4x4[path_4x4[6]]).tolist()
    # all_zero
    edge_steps.append({"similarities": [0.0] * 16, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    # constant 0.50
    edge_steps.append({"similarities": [0.50] * 16, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    # low signal non-flat
    edge_steps.append(
        {"similarities": np.linspace(0.0, 0.03, 16).tolist(), "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0}
    )
    # NaN similarity
    nan_sims = [0.15] * 16
    nan_sims[3] = "NaN"  # type: ignore[assignment]
    edge_steps.append({"similarities": nan_sims, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    # Inf odom delta
    good_s = [0.15] * 16
    good_s[path_4x4[7]] = 0.86
    edge_steps.append({"similarities": good_s, "delta_ne_odom_m": ["Inf", 0.0], "odom_sigma_m": 20.0})
    # Non-positive sigma
    edge_steps.append({"similarities": good_s, "delta_ne_odom_m": delta_next, "odom_sigma_m": 0.0})
    # Post-blind recovery (genuine frame admits path[7])
    edge_steps.append({"similarities": good_s, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    # Reset + aliased nearly-flat frame + 6-stage VisualFixMetrics pass & fail
    edge_steps.append({"reset": True})
    aliased = [0.40] * 16
    aliased[0] = 0.41
    edge_steps.append({"similarities": aliased, "delta_ne_odom_m": [0.0, 0.0], "odom_sigma_m": 20.0})
    edge_steps.append({"reset": True})
    sharp = [0.15] * 16
    sharp[0] = 0.88
    good_vfm = {
        "n_inliers": 72,
        "n_eff": 7.5,
        "h_norm": 0.82,
        "kappa_geom": 2.5,
        "kappa_fim": 2.1,
        "psr": 1.0,
        "chi2_mc": 1.4,
        "q_vis": 2.0,
        "hpl_nav_m": 18.5,
        "hal_nav_m": 30.0,
    }
    bad_hpl_vfm = {**good_vfm, "hpl_nav_m": 42.0}
    edge_steps.append(
        {
            "similarities": sharp,
            "delta_ne_odom_m": [0.0, 0.0],
            "odom_sigma_m": 20.0,
            "visual_fix_metrics": good_vfm,
            "aliasing_thresholds": {"min_psr_delta": 0.05},
        }
    )
    edge_steps.append(
        {
            "similarities": sharp,
            "delta_ne_odom_m": [0.0, 0.0],
            "odom_sigma_m": 20.0,
            "visual_fix_metrics": bad_hpl_vfm,
        }
    )

    batch_req = {
        "kind": "batch",
        "cases": {
            "bayesian_w7": {
                "kind": "sequence_vpr",
                "gallery_ne_m": coords_8x8.tolist(),
                "config": {
                    "window_size": 7,
                    "mode": "bayesian",
                    "min_psr_delta": 0.05,
                    "min_posterior_prob": 0.30,
                    "min_posterior_margin": 0.08,
                },
                "steps": traj_steps,
            },
            "viterbi_w7": {
                "kind": "sequence_vpr",
                "gallery_ne_m": coords_8x8.tolist(),
                "config": {
                    "window_size": 7,
                    "mode": "viterbi",
                    "min_psr_delta": 0.05,
                    "min_posterior_prob": 0.30,
                    "min_posterior_margin": 0.08,
                },
                "steps": traj_steps,
            },
            "edge_blind_aliasing_recovery": {
                "kind": "sequence_vpr",
                "gallery_ne_m": coords_4x4.tolist(),
                "config": {
                    "window_size": 7,
                    "mode": "bayesian",
                    "min_psr_cal": 1.15,
                    "min_psr_delta": 0.05,
                    "min_posterior_prob": 0.30,
                    "min_posterior_margin": 0.08,
                },
                "steps": edge_steps,
            },
        },
    }

    rs_batch = _run_parity_eval(batch_req, tmp_path, "seq_vpr_parity")["results"]
    max_diff = [0.0]
    total_steps_checked = 0

    for case_name, spec in batch_req["cases"].items():
        py_cfg = SequenceVprConfig(**spec["config"])
        py_flt = SequenceVprFilter(np.asarray(spec["gallery_ne_m"], dtype=np.float64), config=py_cfg)
        rs_outs = rs_batch[case_name]["outputs"]
        out_i = 0
        for step in spec["steps"]:
            if step.get("reset"):
                py_flt.reset()
                continue
            sims = np.asarray([_dec_f64(x) for x in step["similarities"]], dtype=np.float64)
            delta = np.asarray([_dec_f64(x) for x in step["delta_ne_odom_m"]], dtype=np.float64)
            sigma = _dec_f64(step["odom_sigma_m"])
            vfm = VisualFixMetrics(**step["visual_fix_metrics"]) if "visual_fix_metrics" in step else None
            thr = VisualAliasingThresholds(**step["aliasing_thresholds"]) if "aliasing_thresholds" in step else None
            py_dec = py_flt.step(sims, delta, odom_sigma_m=sigma, visual_fix_metrics=vfm, aliasing_thresholds=thr)
            rs_dec = rs_outs[out_i]
            out_i += 1
            total_steps_checked += 1

            prefix = f"{case_name}[{out_i - 1}]"
            assert py_dec.frame_idx == rs_dec["frame_idx"], prefix
            assert py_dec.window_len == rs_dec["window_len"], prefix
            assert py_dec.raw_top1_idx == rs_dec["raw_top1_idx"], prefix
            assert py_dec.seq_top1_idx == rs_dec["seq_top1_idx"], prefix
            assert py_dec.final_top1_idx == rs_dec["final_top1_idx"], prefix
            py_top5 = list(py_dec.final_top5_indices)
            rs_top5 = list(rs_dec["final_top5_indices"])
            assert len(py_top5) == len(rs_top5) and len(set(rs_top5)) == len(rs_top5), prefix
            # Exact index match for every non-tied rank; allow unstable numpy quicksort permutation ONLY on exact ties (< 1e-12)
            for rank, (p_idx, r_idx) in enumerate(zip(py_top5, rs_top5)):
                if p_idx != r_idx:
                    p_sc = float(py_dec.fused_scores[p_idx])
                    r_sc = float(py_dec.fused_scores[r_idx])
                    assert math.isfinite(p_sc) and math.isfinite(r_sc) and abs(p_sc - r_sc) < 1e-12, (
                        f"{prefix} final_top5[{rank}] mismatch on non-tied scores: py={p_idx}({p_sc}) vs rs={r_idx}({r_sc})"
                    )
            assert list(py_dec.viterbi_path_indices) == rs_dec["viterbi_path_indices"], prefix
            assert py_dec.admitted is rs_dec["admitted"], prefix
            assert py_dec.psr_result.admitted is rs_dec["psr_admitted"], prefix
            assert list(py_dec.reason_codes) == rs_dec["reason_codes"], (
                f"{prefix} reason_codes mismatch: py={py_dec.reason_codes} vs rs={rs_dec['reason_codes']}"
            )
            _compare_f64(py_dec.posterior_prob_top1, _dec_f64(rs_dec["posterior_prob_top1"]), f"{prefix}.p_top1", max_diff)
            _compare_f64(
                py_dec.posterior_margin_non_adj,
                _dec_f64(rs_dec["posterior_margin_non_adj"]),
                f"{prefix}.p_margin",
                max_diff,
            )
            _compare_f64(py_dec.psr_result.psr_cal, _dec_f64(rs_dec["psr_cal"]), f"{prefix}.psr_cal", max_diff)
            _compare_f64(py_dec.psr_result.margin, _dec_f64(rs_dec["psr_margin"]), f"{prefix}.psr_margin", max_diff)

        assert out_i == len(rs_outs)

    assert total_steps_checked >= 80
    assert max_diff[0] < PARITY_TOL, f"SequenceVprFilter max abs diff {max_diff[0]:.3e} >= {PARITY_TOL}"
    print(f"[PARITY-SEQ-VPR] steps={total_steps_checked} max_abs_diff={max_diff[0]:.3e} (< 1e-6 PASS)")


def test_dual_bank_cusum_python_rust_parity_under_1e6(tmp_path: Path) -> None:
    cases: dict[str, dict[str, Any]] = {}

    # 1. F1..F5 Correlated GNSS multipath clean turbulent flights at BOTH 1 Hz and 10 Hz -> 0 FA!
    f_scenarios = (
        ("f1_clean", 40.0, 3.0, 0.01, 0.0, 1.0, 1.5, 0.0, 3.5),
        ("f2_mp15", 35.0, 2.5, 0.01, 1.5, 12.0, 1.0, 0.0, 3.5),
        ("f3_mp25", 40.0, 3.0, 0.01, 2.5, 15.0, 1.0, 0.0, 4.0),
        ("f4_mp40", 45.0, 3.5, 0.01, 4.0, 25.0, 1.2, 0.0, 5.0),
        ("f5_vdrop20", 45.0, 3.0, 0.01, 3.0, 20.0, 1.0, 0.20, 4.5),
    )
    for hz in (1.0, 10.0):
        phys_dt = 0.1
        sub = max(1, int(round(10.0 / hz)))
        dur_s = 120.0 if hz == 1.0 else 40.0  # covers warmup + >=8s orthogonal-eig + persist window at 1Hz & 10Hz
        n_epochs = int(round(dur_s * hz))
        for fi, (tag, bank_deg, gust_std, imu_sc, mp_sig, mp_tau, gw, vdrop, b_turb) in enumerate(f_scenarios):
            rng = np.random.default_rng(20261500 + fi * 131)
            speed = 40.0
            true_p = np.zeros(2, dtype=np.float64)
            true_v = np.array([speed, 0.0], dtype=np.float64)
            mp = np.zeros(2, dtype=np.float64)
            phi_mp = math.exp(-phys_dt / max(mp_tau, 0.1))
            sig_mp = mp_sig * math.sqrt(max(0.0, 1.0 - phi_mp * phi_mp))
            gust = np.zeros(2, dtype=np.float64)
            phi_g = math.exp(-phys_dt / 1.5)
            sig_g = gust_std * math.sqrt(max(0.0, 1.0 - phi_g * phi_g))
            baro = 0.0
            phi_b = math.exp(-phys_dt / 8.0)
            steps = []
            for ep in range(n_epochs):
                a_sum = np.zeros(2, dtype=np.float64)
                for s in range(sub):
                    t = (ep * sub + s + 1) * phys_dt
                    bank = math.radians(bank_deg) * math.sin(2.0 * math.pi * t / 60.0)
                    vh = true_v / max(float(np.linalg.norm(true_v)), 1.0)
                    perp = np.array([-vh[1], vh[0]], dtype=np.float64)
                    gust = phi_g * gust + rng.normal(0.0, sig_g, 2)
                    a_true = (9.80665 * math.tan(bank)) * perp + gust - 0.05 * (true_v - speed * vh)
                    true_p = true_p + true_v * phys_dt + 0.5 * a_true * phys_dt * phys_dt
                    true_v = true_v + a_true * phys_dt
                    a_sum += a_true * (1.0 + imu_sc) + rng.normal(0.0, 0.35, 2)
                    mp = phi_mp * mp + rng.normal(0.0, sig_mp, 2)
                    baro = phi_b * baro + rng.normal(0.0, b_turb * math.sqrt(1.0 - phi_b * phi_b))
                yg = true_p + mp + rng.normal(0.0, gw, 2)
                vis_ok = bool(rng.random() >= vdrop)
                yv = (true_p + rng.normal(0.0, 2.8, 2)).tolist() if vis_ok else None
                steps.append(
                    {
                        "ins_accel_ne_mps2": (a_sum / float(sub)).tolist(),
                        "baro_down_m": -300.0 + baro,
                        "gnss_ne_m": yg.tolist(),
                        "vis_ne_m": yv,
                        "vis_gate_admitted": vis_ok,
                    }
                )
            cases[f"mp_{tag}_{int(hz)}hz"] = {
                "kind": "dual_bank_cusum",
                "config": {"sample_rate_hz": hz},
                "reset": {"init_ned_m": [0.0, 0.0, -300.0], "init_vel_ne_mps": [speed, 0.0]},
                "steps": steps,
                "_expect_zero_fa": True,
            }

    # 2. Boiled-frog horizontal ramps (0.05..5.0 m/s @ 22.5 deg) at 1 Hz & 10 Hz (< 30 m HAL)
    u225 = np.array([math.cos(math.radians(22.5)), math.sin(math.radians(22.5))], dtype=np.float64)
    for hz, rates in ((1.0, (0.05, 0.25, 0.5, 2.0, 5.0)), (10.0, (0.5, 2.0, 5.0))):
        dt = 1.0 / hz
        for rate in rates:
            rng = np.random.default_rng(1000 + int(rate * 100))
            t_onset = 40.0
            n_steps = int((t_onset + max(100.0, 22.0 / rate)) * hz)
            steps = []
            true_ps = []
            for k in range(n_steps):
                t = (k + 1) * dt
                tp = np.array([15.0 * t, 5.0 * t], dtype=np.float64)
                true_ps.append(tp)
                spoof = max(0.0, t - t_onset) * rate * u225
                steps.append(
                    {
                        "baro_down_m": -100.0,
                        "gnss_ne_m": (tp + spoof + rng.normal(0.0, 0.75, 2)).tolist(),
                        "vis_ne_m": (tp + rng.normal(0.0, 1.6, 2)).tolist(),
                        "vis_gate_admitted": True,
                    }
                )
            cases[f"ramp_{rate:.2f}mps_{int(hz)}hz"] = {
                "kind": "dual_bank_cusum",
                "config": {"sample_rate_hz": hz},
                "reset": {"init_ned_m": [0.0, 0.0, -100.0], "init_vel_ne_mps": [15.0, 5.0]},
                "steps": steps,
                "_expect_alarm_after_idx": int(t_onset * hz),
                "_true_ps": true_ps,
            }

    # 3. Horizontal steps (10 m, 20 m, 100 m) + hysteresis reintegration
    for mag in (10.0, 20.0, 100.0):
        rng = np.random.default_rng(2000 + int(mag * 10))
        steps = []
        for k in range(120):
            t = float(k + 1)
            tp = np.array([15.0 * t, 5.0 * t], dtype=np.float64)
            sb = np.array([mag if 40.0 <= t < 65.0 else 0.0, 0.0], dtype=np.float64)
            steps.append(
                {
                    "baro_down_m": -100.0,
                    "gnss_ne_m": (tp + sb + rng.normal(0.0, 0.75, 2)).tolist(),
                    "vis_ne_m": (tp + rng.normal(0.0, 1.6, 2)).tolist(),
                    "vis_gate_admitted": True,
                }
            )
        cases[f"step_{int(mag)}m_1hz"] = {
            "kind": "dual_bank_cusum",
            "config": {"sample_rate_hz": 1.0},
            "reset": {"init_ned_m": [0.0, 0.0, -100.0], "init_vel_ne_mps": [15.0, 5.0]},
            "steps": steps,
            "_expect_alarm_after_idx": 39,
        }

    # 4. Vertical Down: correlated baro-turbulence (0 FA) + +1.0 m/s ramp + -30 m step + NaN fail-closed
    for hz in (1.0, 10.0):
        dt = 1.0 / hz
        rng = np.random.default_rng(3100 + int(hz))
        b_baro, b_mp_d = 0.0, 0.0
        a_b, a_m = math.exp(-dt / 8.0), math.exp(-dt / 12.0)
        steps_turb = []
        n_turb = int((120.0 if hz == 1.0 else 40.0) * hz)
        for k in range(n_turb):
            t = (k + 1) * dt
            b_baro = a_b * b_baro + math.sqrt(1.0 - a_b * a_b) * 5.0 * rng.normal()
            b_mp_d = a_m * b_mp_d + math.sqrt(1.0 - a_m * a_m) * 3.5 * rng.normal()
            steps_turb.append(
                {
                    "baro_down_m": -120.0 + b_baro + rng.normal(0.0, 0.4),
                    "gnss_ne_m": [15.0 * t + rng.normal(0.0, 0.75), rng.normal(0.0, 0.75)],
                    "vis_ne_m": [15.0 * t + rng.normal(0.0, 1.6), rng.normal(0.0, 1.6)],
                    "gnss_down_m": -120.0 + b_mp_d + rng.normal(0.0, 1.2),
                }
            )
        cases[f"vert_turb_{int(hz)}hz"] = {
            "kind": "dual_bank_cusum",
            "config": {"sample_rate_hz": hz},
            "reset": {"init_ned_m": [0.0, 0.0, -120.0], "init_vel_ne_mps": [15.0, 0.0]},
            "steps": steps_turb,
            "_expect_zero_vert_fa": True,
        }

        for sign, mode, val in ((1.0, "ramp", 1.0), (-1.0, "step", 30.0)):
            rng_v = np.random.default_rng(4200)
            steps_v = []
            for k in range(int(90.0 * hz)):
                t = (k + 1) * dt
                spoof_d = sign * ((max(0.0, t - 25.0) * val) if mode == "ramp" else (val if t >= 25.0 else 0.0))
                steps_v.append(
                    {
                        "baro_down_m": -120.0 + rng_v.normal(0.0, 0.5),
                        "gnss_ne_m": [15.0 * t + rng_v.normal(0.0, 0.75), rng_v.normal(0.0, 0.75)],
                        "vis_ne_m": [15.0 * t + rng_v.normal(0.0, 1.6), rng_v.normal(0.0, 1.6)],
                        "gnss_down_m": -120.0 + spoof_d + rng_v.normal(0.0, 1.2),
                    }
                )
            cases[f"vert_{mode}_{int(hz)}hz"] = {
                "kind": "dual_bank_cusum",
                "config": {"sample_rate_hz": hz},
                "reset": {"init_ned_m": [0.0, 0.0, -120.0], "init_vel_ne_mps": [15.0, 0.0]},
                "steps": steps_v,
                "_expect_vert_alarm_after_idx": int(25.0 * hz) - 1,
            }

    cases["vert_nan_fail_closed"] = {
        "kind": "dual_bank_cusum",
        "config": {"sample_rate_hz": 1.0},
        "steps": [
            {
                "baro_down_m": -100.0,
                "gnss_ne_m": [0.0, 0.0],
                "vis_ne_m": [0.0, 0.0],
                "gnss_down_m": "NaN",
            }
        ],
        "_expect_vert_alarm_after_idx": 0,
    }

    # Strip private test-assertion keys before sending JSON to Rust
    wire_cases = {
        name: {k: v for k, v in spec.items() if not k.startswith("_")}
        for name, spec in cases.items()
    }
    rs_batch = _run_parity_eval({"kind": "batch", "cases": wire_cases}, tmp_path, "cusum_parity")["results"]

    max_diff = [0.0]
    total_epochs_checked = 0

    for name, spec in cases.items():
        py_cfg = DualBankConfig(**spec["config"])
        py_flt = DualBankEsEkfCusum(cfg=py_cfg)
        if "reset" in spec:
            py_flt.reset(
                init_ned_m=spec["reset"]["init_ned_m"],
                init_vel_ne_mps=spec["reset"]["init_vel_ne_mps"],
            )
        rs_outs = rs_batch[name]["outputs"]
        assert len(rs_outs) == len(spec["steps"]), f"{name}: step count mismatch"

        first_alarm_idx: int | None = None
        first_vert_alarm_idx: int | None = None
        bank_a_err_at_alarm = float("inf")

        for idx, step in enumerate(spec["steps"]):
            gnss_down = _dec_f64(step["gnss_down_m"]) if "gnss_down_m" in step and step["gnss_down_m"] is not None else None
            py_out = py_flt.step(
                ins_accel_ne_mps2=step.get("ins_accel_ne_mps2", (0.0, 0.0)),
                baro_down_m=float(step.get("baro_down_m", 0.0)),
                gnss_ne_m=step.get("gnss_ne_m"),
                vis_ne_m=step.get("vis_ne_m"),
                vis_gate_admitted=bool(step.get("vis_gate_admitted", True)),
                gnss_down_m=gnss_down,
            )
            rs_out = rs_outs[idx]
            total_epochs_checked += 1

            prefix = f"{name}[{idx}]"
            assert py_out.epoch_idx == rs_out["epoch_idx"], prefix
            assert py_out.active_bank == rs_out["active_bank"], prefix
            assert py_out.gnss_quarantined is rs_out["gnss_quarantined"], prefix
            assert py_out.quarantine_triggered_this_epoch is rs_out["quarantine_triggered_this_epoch"], prefix
            assert py_out.reintegrated_this_epoch is rs_out["reintegrated_this_epoch"], prefix
            assert py_out.spoof_alarm is rs_out["spoof_alarm"], prefix
            assert py_out.vertical_spoof_alarm is rs_out["vertical_spoof_alarm"], prefix
            assert py_out.nominal_streak == rs_out["nominal_streak"], prefix

            _compare_f64(py_out.max_cusum_stat, _dec_f64(rs_out["max_cusum_stat"]), f"{prefix}.cusum_stat", max_diff)
            _compare_f64(
                py_out.vertical_cusum_stat,
                _dec_f64(rs_out["vertical_cusum_stat"]),
                f"{prefix}.vert_cusum",
                max_diff,
            )
            for ax in range(2):
                _compare_f64(
                    float(py_out.bank_a_p_ne_m[ax]),
                    _dec_f64(rs_out["bank_a_p_ne_m"][ax]),
                    f"{prefix}.bank_a[{ax}]",
                    max_diff,
                )
                _compare_f64(
                    float(py_out.bank_b_p_ne_m[ax]),
                    _dec_f64(rs_out["bank_b_p_ne_m"][ax]),
                    f"{prefix}.bank_b[{ax}]",
                    max_diff,
                )

            if py_out.spoof_alarm and first_alarm_idx is None:
                first_alarm_idx = idx
                if "_true_ps" in spec:
                    bank_a_err_at_alarm = float(np.linalg.norm(py_out.bank_a_p_ne_m - spec["_true_ps"][idx]))
            if py_out.vertical_spoof_alarm and first_vert_alarm_idx is None:
                first_vert_alarm_idx = idx

        if spec.get("_expect_zero_fa"):
            assert first_alarm_idx is None, f"{name}: unexpected false alarm at epoch {first_alarm_idx}"
        if spec.get("_expect_zero_vert_fa"):
            assert first_vert_alarm_idx is None, f"{name}: unexpected vertical FA at epoch {first_vert_alarm_idx}"
        if "_expect_alarm_after_idx" in spec:
            assert first_alarm_idx is not None and first_alarm_idx >= spec["_expect_alarm_after_idx"], (
                f"{name}: expected alarm >= {spec['_expect_alarm_after_idx']}, got {first_alarm_idx}"
            )
            if "_true_ps" in spec:
                assert bank_a_err_at_alarm < 30.0, f"{name}: Bank-A error {bank_a_err_at_alarm:.2f} m >= 30 m HAL"
        if "_expect_vert_alarm_after_idx" in spec:
            assert first_vert_alarm_idx is not None and first_vert_alarm_idx >= spec["_expect_vert_alarm_after_idx"], (
                f"{name}: expected vertical alarm >= {spec['_expect_vert_alarm_after_idx']}, got {first_vert_alarm_idx}"
            )

    assert total_epochs_checked >= 5000
    assert max_diff[0] < PARITY_TOL, f"DualBankEsEkfCusum max abs diff {max_diff[0]:.3e} >= {PARITY_TOL}"
    print(f"[PARITY-DUAL-BANK-CUSUM] epochs={total_epochs_checked} max_abs_diff={max_diff[0]:.3e} (< 1e-6 PASS)")


def test_edge_infer_cli_replay_python_shadow_logger_and_tamper_rejection(tmp_path: Path) -> None:
    exe = _resolve_edge_exe()
    log_path = tmp_path / "phase3_shadow_replay.ndjson"

    # Write a real Python ShadowModeTelemetryLogger log with em-dash watermark, sci/fixed floats, and spoof ramp
    with ShadowModeTelemetryLogger(
        log_path,
        trial_phase="phase3_university_shadow_tx_pin_removed",
        hardware_tx_pin_removed=True,
    ) as logger:
        for k in range(25):
            spoof_off = 0.0 if k < 5 else 2.5 * float(k - 4)
            logger.record_event(
                timestamp_mono_us=(k + 1) * 1_000_000,
                event_type="rx_mavlink2",
                payload={
                    "frame_idx": k,
                    "tiny_val": 1.0e-5,
                    "down_m": -180.0 - 0.25 * k,
                    "gnss_ne_m": [15.0 * k + spoof_off, 0.42],
                    "vis_ne_m": [15.0 * k, 0.42],
                    "gnss_down_m": -180.0 - 0.25 * k,
                },
                direction="rx",
            )

    py_ver = verify_shadow_log_chain(log_path)
    assert py_ver.valid is True

    # 1. Valid log: Pure-Rust `--replay` verifies exact SHA-256 chain & detects the spoof ramp during replay
    proc_ok = subprocess.run(
        [str(exe), "--replay", str(log_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_ok.returncode == 0, f"--replay failed on valid log: {proc_ok.stderr}\n{proc_ok.stdout}"
    rs_rep = json.loads(proc_ok.stdout)
    assert rs_rep["valid"] is True
    assert rs_rep["record_count"] == py_ver.record_count == 25
    assert rs_rep["head_hash"] == py_ver.head_hash
    assert rs_rep["tx_inhibited_all"] is True
    assert rs_rep["actuator_authority_all_false"] is True
    assert rs_rep["total_outbound_mavlink_frames"] == 0
    assert rs_rep["monotonic_timestamps_ok"] is True
    assert rs_rep["cusum_epochs_replayed"] == 25
    assert rs_rep["cusum_alarms"] > 0
    assert rs_rep["cusum_quarantined_final"] is True
    assert rs_rep["rta_ticks_replayed"] == 25

    orig_lines = log_path.read_text(encoding="utf-8").splitlines()

    # 2. 1-byte payload tamper -> MUST exit non-zero!
    tamper_1b = tmp_path / "tamper_1byte.ndjson"
    mod = list(orig_lines)
    mod[3] = mod[3].replace("0.42", "0.43", 1)
    tamper_1b.write_text("\n".join(mod) + "\n", encoding="utf-8")
    proc_bad = subprocess.run([str(exe), "--replay", str(tamper_1b)], capture_output=True, text=True, check=False)
    assert proc_bad.returncode != 0, "1-byte payload tamper MUST exit non-zero!"
    bad_rep = json.loads(proc_bad.stdout)
    assert bad_rep["valid"] is False
    assert bad_rep["first_failed_line"] == 4

    # 3. Forged tx_inhibited=False with re-hashed chain -> MUST exit non-zero!
    forged_path = tmp_path / "forged_tx.ndjson"
    rehashed: list[str] = []
    prev_h = SHADOW_GENESIS_HASH
    for idx, line in enumerate(orig_lines[:5]):
        rec = json.loads(line)
        rec["prev_hash"] = prev_h
        if idx == 2:
            rec["tx_inhibited"] = False
            rec["outbound_mavlink_frames_emitted"] = 1
        body = {k: v for k, v in rec.items() if k != "entry_hash"}
        new_h = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        rec["entry_hash"] = new_h
        prev_h = new_h
        rehashed.append(json.dumps(rec, sort_keys=True, separators=(",", ":")))
    forged_path.write_text("\n".join(rehashed) + "\n", encoding="utf-8")
    proc_forged = subprocess.run([str(exe), "--replay", str(forged_path)], capture_output=True, text=True, check=False)
    assert proc_forged.returncode != 0, "Forged TX record MUST exit non-zero!"
    forged_rep = json.loads(proc_forged.stdout)
    assert forged_rep["valid"] is False
    assert forged_rep["tx_inhibited_all"] is False
    assert forged_rep["total_outbound_mavlink_frames"] == 1


def test_edge_infer_cli_default_self_check_and_short_soak() -> None:
    exe = _resolve_edge_exe()

    # 1. Zero-arg default backward-compatible --self-check JSON
    proc_sc = subprocess.run(
        [str(exe)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_sc.returncode == 0, f"default self-check failed: {proc_sc.stderr}"
    sc = json.loads(proc_sc.stdout)
    assert sc["non_flight_sim_only"] is True
    assert sc["actuator_authority"] is False
    assert sc["rta_mode"] == 4
    assert sc["active_range_wire_crc_ok"] is True

    # 2. Short 2 s thermal-safe paced --soak check
    proc_soak = subprocess.run(
        [str(exe), "--soak", "--seconds", "2", "--hz", "5"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_soak.returncode == 0, f"--soak failed: {proc_soak.stderr}\n{proc_soak.stdout}"
    soak = json.loads(proc_soak.stdout)
    assert soak["mode"] == "soak"
    assert soak["pass"] is True
    assert soak["non_flight_sim_only"] is True
    assert soak["actuator_authority"] is False
    assert soak["fast_ticks"] >= 5
    assert soak["slow_ticks"] >= 1
    assert soak["non_finite_count"] == 0
    assert soak["p95_ms"] >= 0.0 and soak["max_ms"] >= soak["p95_ms"]
    assert soak["rss_end_mib"] > 0.0
