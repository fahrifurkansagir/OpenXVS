"""Tests for Wave-2 Runtime Integration (Commit H2): Sequence-VPR, 3-of-5 TBC, dual 5/6-output XVS-Land, and v5.2 loader pins."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

import xvs.v2.studio_bridge.inference_engine as ie_mod
from xvs.v2.studio_bridge import (
    DEFAULT_VRAM_CAP_MB,
    DEFAULT_WAVE_C_LAND_DIR,
    DEFAULT_WAVE_C_THERMAL_DIR,
    EXPECTED_D2_ONNX_SHA256,
    EXPECTED_WAVE_C_LAND_ONNX_SHA256,
    EXPECTED_WAVE_C_THERMAL_ONNX_SHA256,
    StudioInferenceEngine,
    load_studio_models,
)
from xvs.v2.studio_bridge.inference_engine import validate_keyline_sigmas_px
from xvs.v2.studio_bridge.models_loader import (
    DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
    DEFAULT_V52_LAND_DIR,
    DEFAULT_V52_THERMAL_DIR,
    EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256,
    EXPECTED_V52_LAND_FP16_ONNX_SHA256,
    EXPECTED_V52_THERMAL_FP16_ONNX_SHA256,
    XVS_LAND_V52_FP16_ONNX_FILENAME,
    XVS_LAND_V52_PARAM_COUNT,
    XVS_NAV_V5_FAZB_ONNX_FILENAME,
    XVS_NAV_V52_FAZC_FP16_ONNX_FILENAME,
    XVS_THERMAL_V52_FP16_ONNX_FILENAME,
    compute_file_sha256,
)
from xvs.v2.xvs_d2.track_before_confirm import BearingDetection
from xvs.v2.xvs_land.model import XvsLandV52Model
from xvs.v2.xvs_land.pnpl_ekf import (
    build_ksfo_28l_reference_features,
    compute_hesse_line_from_points,
    project_runway_points_ned,
)
from xvs.v2.xvs_land.train_g4 import export_xvs_land_v52_onnx
from xvs.v2.xvs_nav.sequence_vpr import SequenceVprConfig, SequenceVprFilter


def _make_textured_frame(w: int = 320, h: int = 240, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(35, 220, size=(h, w, 3), dtype=np.uint8)
    for k in range(0, w, 24):
        cv2.line(img, (k, 0), (w - 1 - k, h - 1), (235, 240, 245), 1, cv2.LINE_AA)
    for r in range(16, h, 28):
        cv2.circle(img, ((r * 3) % (w - 20) + 10, r), 6, (25, 45, 210), -1, cv2.LINE_AA)
    return img


class _FakeOrtNodeMeta:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


class _FakeLandOrtSession:
    """Deterministic fake 5-output or 6-output ONNX session for unit testing `infer_runway_land`."""

    def __init__(
        self,
        *,
        kpts_xy: np.ndarray,
        keylines_hesse: np.ndarray,
        cov_2x2: np.ndarray,
        keyline_sigma_px: np.ndarray | None,
        roi_hw: tuple[int, int] = (256, 256),
    ) -> None:
        self._kpts = np.asarray(kpts_xy, dtype=np.float32)[np.newaxis, :, :]
        self._kl = np.asarray(keylines_hesse, dtype=np.float32)[np.newaxis, :, :]
        self._cov = np.asarray(cov_2x2, dtype=np.float32)[np.newaxis, :, :, :]
        self._kl_sig = (
            None
            if keyline_sigma_px is None
            else np.asarray(keyline_sigma_px, dtype=np.float32)[np.newaxis, :]
        )
        self._roi_hw = roi_hw
        self.last_fetch_names: list[str] = []

    def get_inputs(self) -> list[_FakeOrtNodeMeta]:
        return [_FakeOrtNodeMeta("roi_crop", [1, 3, self._roi_hw[0], self._roi_hw[1]])]

    def get_outputs(self) -> list[_FakeOrtNodeMeta]:
        outs = [
            _FakeOrtNodeMeta("keypoints_xy", [1, 12, 2]),
            _FakeOrtNodeMeta("keylines_hesse", [1, 5, 3]),
            _FakeOrtNodeMeta("cov_2x2", [1, 12, 2, 2]),
            _FakeOrtNodeMeta("logits_x", [1, 12, self._roi_hw[1] * 2]),
            _FakeOrtNodeMeta("logits_y", [1, 12, self._roi_hw[0] * 2]),
        ]
        if self._kl_sig is not None:
            outs.append(_FakeOrtNodeMeta("keyline_sigma_px", [1, 5]))
        return outs

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert "roi_crop" in inputs
        assert inputs["roi_crop"].shape == (1, 3, self._roi_hw[0], self._roi_hw[1])
        table: dict[str, np.ndarray] = {
            "keypoints_xy": self._kpts,
            "keylines_hesse": self._kl,
            "cov_2x2": self._cov,
        }
        if self._kl_sig is not None:
            table["keyline_sigma_px"] = self._kl_sig
        names = list(table.keys()) if output_names is None else list(output_names)
        self.last_fetch_names = names
        return [table[n] for n in names]


@pytest.fixture(scope="module")
def cpu_studio_engine() -> StudioInferenceEngine:
    bundle = load_studio_models(device_pref="cpu", vram_cap_mb=DEFAULT_VRAM_CAP_MB, smoke_fast=False)
    return StudioInferenceEngine(bundle)


def test_sequence_vpr_multiframe_wiring_reset_and_fail_closed(
    cpu_studio_engine: StudioInferenceEngine,
) -> None:
    """Verify SequenceVprFilter (W=7) live wiring, NED odometry transitions, reset, and 0-HMI fail-closed."""
    cpu_studio_engine.reset_temporal_state()
    ortho = cpu_studio_engine._load_ksfo_ortho()
    crop = ortho[120:376, 120:376].copy()

    # 1. Multi-frame Sequence-VPR over a 10-tile NED corridor (step 200 m North per frame, W=7 cap)
    corridor_ne_m = np.array([[float(i) * 200.0, 0.0] for i in range(10)], dtype=np.float64)
    seq_filter = SequenceVprFilter(
        corridor_ne_m,
        config=SequenceVprConfig(window_size=7, non_adj_radius_m=300.0, min_psr_cal=1.15, min_psr_delta=0.04),
    )

    for step_idx in range(9):
        sims = np.full((10,), 0.20, dtype=np.float64)
        sims[step_idx] = 0.92
        delta_ne = (0.0, 0.0) if step_idx == 0 else (200.0, 0.0)
        out = cpu_studio_engine.infer_crossview_nav(
            crop,
            ref_bgr=crop,
            ref_cache_key="v52_seq_vpr_crop",
            seq_vpr_filter=seq_filter,
            seq_vpr_similarities=sims,
            delta_ne_odom_m=delta_ne,
            odom_sigma_m=20.0,
            return_jpeg=False,
        )
        json.dumps(out, allow_nan=False)
        assert out["seq_vpr_window"] == min(step_idx + 1, 7)
        assert out["seq_vpr_window_max"] == 7
        assert out["seq_vpr_admitted"] is True
        assert out["seq_vpr_posterior_prob"] > 0.90
        assert out["seq_vpr"]["final_top1_idx"] == step_idx
        assert out["seq_vpr"]["admitted"] is True

    # 2. Fail-closed on NaN odometry & non-positive sigma WITHOUT polluting the 7-frame window
    out_nan_odom = cpu_studio_engine.infer_crossview_nav(
        crop,
        ref_bgr=crop,
        ref_cache_key="v52_seq_vpr_crop",
        seq_vpr_filter=seq_filter,
        seq_vpr_similarities=np.full((10,), 0.25, dtype=np.float64),
        delta_ne_odom_m=(float("nan"), 0.0),
        odom_sigma_m=20.0,
        return_jpeg=False,
    )
    json.dumps(out_nan_odom, allow_nan=False)
    assert out_nan_odom["seq_vpr_admitted"] is False
    assert out_nan_odom["seq_vpr_posterior_prob"] == 0.0
    assert "NONFINITE_OR_INVALID_ODOM_DELTA" in out_nan_odom["seq_vpr"]["reason_codes"]
    assert out_nan_odom["seq_vpr_window"] == 7

    out_bad_sigma = cpu_studio_engine.infer_crossview_nav(
        crop,
        ref_bgr=crop,
        ref_cache_key="v52_seq_vpr_crop",
        seq_vpr_filter=seq_filter,
        seq_vpr_similarities=np.full((10,), 0.25, dtype=np.float64),
        delta_ne_odom_m=(200.0, 0.0),
        odom_sigma_m=0.0,
        return_jpeg=False,
    )
    json.dumps(out_bad_sigma, allow_nan=False)
    assert out_bad_sigma["seq_vpr_admitted"] is False
    assert "NONFINITE_OR_NONPOSITIVE_ODOM_SIGMA" in out_bad_sigma["seq_vpr"]["reason_codes"]

    # 3. Fail-closed aliasing rejection on blank uniform image (0 inliers -> 0 HMI)
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    out_blank = cpu_studio_engine.infer_crossview_nav(
        blank,
        ref_bgr=blank,
        return_jpeg=False,
    )
    json.dumps(out_blank, allow_nan=False)
    assert out_blank["aliasing_gate"]["admitted"] is False
    assert out_blank["seq_vpr_admitted"] is False

    # 4. Stateful engine filter growth + clean episode/scenario reset
    cpu_studio_engine.reset_temporal_state()
    w_seq: list[int] = []
    for _ in range(3):
        res_s = cpu_studio_engine.infer_crossview_nav(crop, return_jpeg=False)
        w_seq.append(int(res_s["seq_vpr_window"]))
    assert w_seq == [1, 2, 3]

    cpu_studio_engine.reset_temporal_state()
    res_after_reset = cpu_studio_engine.infer_crossview_nav(crop, return_jpeg=False)
    assert res_after_reset["seq_vpr_window"] == 1

    # Demo-tick scenario switch resets the demo Sequence-VPR sliding window cleanly
    cpu_studio_engine.reset_temporal_state()
    t0 = cpu_studio_engine.run_demo_tick(scenario="cbdr", frame_idx=0, include_images=False)
    t1 = cpu_studio_engine.run_demo_tick(scenario="cbdr", frame_idx=1, include_images=False)
    t_switch = cpu_studio_engine.run_demo_tick(scenario="gps_spoof", frame_idx=2, include_images=False)
    assert t0["channels"]["crossview_nav"]["seq_vpr_window"] == 1
    assert t1["channels"]["crossview_nav"]["seq_vpr_window"] == 2
    assert t_switch["channels"]["crossview_nav"]["seq_vpr_window"] == 1

    # analyze_frame_fast exposes v5.2 Sequence-VPR metrics
    fast_nav = cpu_studio_engine.analyze_frame_fast(crop, channel="nav", return_jpeg=False)
    assert "seq_vpr_window" in fast_nav["metrics"]
    assert "seq_vpr_admitted" in fast_nav["metrics"]
    assert "seq_vpr_posterior_prob" in fast_nav["metrics"]


def test_bearing_tbc_3_of_5_live_wiring_clutter_rejection_and_reset() -> None:
    """Verify 3-of-5 BearingTrackBeforeConfirm live wiring, clutter rejection, coast drop, and telemetry fields."""
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)
    sky_frame = _make_textured_frame(w=320, h=240, seed=77)

    # 1. Transient random sky clutter across distant bearings MUST NOT confirm
    clutter_bearings = [
        (0.25, 0.10),
        (-0.22, -0.08),
        (0.18, -0.12),
        (-0.30, 0.14),
        (0.05, 0.20),
    ]
    for az, el in clutter_bearings:
        res_clutter = engine.infer_eo_sky(
            sky_frame,
            tbc_detections=[BearingDetection(az_rad=az, el_rad=el, score=0.35, modality="eo")],
            return_jpeg=False,
        )
        json.dumps(res_clutter, allow_nan=False)
        assert res_clutter["bearing_track_confirmed"] is False
        assert res_clutter["bearing_tbc_confirmed_tracks"] == 0
        assert res_clutter["bearing_track_hits_in_window"] <= 1
        assert res_clutter["sensor_health"]["bearing_track_confirmed"] is False

    # 2. Clean reset + low-SNR consistent target trajectory (hit, hit, miss, hit -> 3-of-5 confirm!)
    engine.reset_temporal_state()
    snap_idle = engine.sensor_health_snapshot()
    assert snap_idle["bearing_track_confirmed"] is False
    assert snap_idle["bearing_track_hits_in_window"] == 0
    assert snap_idle["bearing_tbc_confirmed_tracks"] == 0

    traj_steps: list[list[BearingDetection]] = [
        [BearingDetection(az_rad=-0.100, el_rad=0.040, score=0.26, modality="eo")],
        [BearingDetection(az_rad=-0.098, el_rad=0.0405, score=0.28, modality="eo")],
        [],  # transient 1-frame fade
        [BearingDetection(az_rad=-0.094, el_rad=0.0415, score=0.27, modality="eo")],
    ]
    results = [
        engine.infer_eo_sky(sky_frame, dt_s=0.1, tbc_detections=dets, return_jpeg=False)
        for dets in traj_steps
    ]
    assert results[0]["bearing_track_confirmed"] is False
    assert results[0]["bearing_track_hits_in_window"] == 1
    assert results[1]["bearing_track_confirmed"] is False
    assert results[1]["bearing_track_hits_in_window"] == 2
    assert results[2]["bearing_track_confirmed"] is False
    assert results[2]["bearing_track_hits_in_window"] == 2
    # 3rd hit within 5-frame window confirms!
    assert results[3]["bearing_track_confirmed"] is True
    assert results[3]["bearing_track_hits_in_window"] == 3
    assert results[3]["bearing_tbc_confirmed_tracks"] == 1
    assert results[3]["bearing_tbc"]["tracks_confirmed_this_step"] == 1
    assert results[3]["sensor_health"]["bearing_track_confirmed"] is True
    assert results[3]["sensor_health"]["bearing_track_hits_in_window"] == 3
    assert results[3]["sensor_health"]["bearing_tbc_confirmed_tracks"] == 1

    # Engine-level sensor_health_snapshot matches live TBC state (1:1 parity with v52_telemetry.ts)
    snap_conf = engine.sensor_health_snapshot()
    assert snap_conf["bearing_track_confirmed"] is True
    assert snap_conf["bearing_track_hits_in_window"] == 3
    assert snap_conf["bearing_tbc_confirmed_tracks"] == 1

    # 3. 5 consecutive misses coast cleanly then drop the confirmed track
    for coast_i in range(4):
        c_res = engine.infer_eo_sky(sky_frame, dt_s=0.1, tbc_detections=[], return_jpeg=False)
        assert c_res["bearing_track_confirmed"] is True, f"Coast frame {coast_i + 1} dropped too early"
    drop_res = engine.infer_eo_sky(sky_frame, dt_s=0.1, tbc_detections=[], return_jpeg=False)
    assert drop_res["bearing_track_confirmed"] is False
    assert drop_res["bearing_tbc_confirmed_tracks"] == 0
    assert drop_res["bearing_tbc"]["tracks_deleted_this_step"] >= 1

    # 4. analyze_frame_fast exposes TBC keys cleanly
    fast_d2 = engine.analyze_frame_fast(sky_frame, channel="d2", return_jpeg=False)
    for key in ("bearing_track_confirmed", "bearing_track_hits_in_window", "bearing_tbc_confirmed_tracks", "bearing_tbc"):
        assert key in fast_d2["metrics"]


def test_xvs_land_dual_5_and_6_output_pnpl_forwarding_and_fail_closed(
    cpu_studio_engine: StudioInferenceEngine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify 5-output (explicit keyline_sigmas_px=None), 6-output v5.2 ONNX promotion, & fail-closed validation."""
    frame = _make_textured_frame(w=320, h=240, seed=101)

    # 1a. Promoted default 6-output v5.2 ONNX (`xvs_land_v52_fp16.onnx`, 384x384) forwards keyline_sigmas_px
    recorded_sigmas: list[Any] = []
    real_solve = ie_mod.solve_4dof_map_pnpl

    def _spy_solve(*args: Any, **kwargs: Any):
        assert "keyline_sigmas_px" in kwargs, "infer_runway_land must pass keyline_sigmas_px explicitly"
        recorded_sigmas.append(kwargs["keyline_sigmas_px"])
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(ie_mod, "solve_4dof_map_pnpl", _spy_solve)
    out_default = cpu_studio_engine.infer_runway_land(frame, return_jpeg=False)
    json.dumps(out_default, allow_nan=False)
    if (DEFAULT_V52_LAND_DIR / XVS_LAND_V52_FP16_ONNX_FILENAME).is_file():
        assert out_default["onnx_output_contract"] == "v52_6output"
        assert out_default["roi_hw"] == [384, 384]
        assert isinstance(out_default["keyline_sigma_px"], list) and len(out_default["keyline_sigma_px"]) == 5
        assert out_default["pnpl"]["used_keyline_sigmas_px"] is True
        assert out_default["pnpl"]["huber_tukey_irls"] is True
        assert isinstance(recorded_sigmas[-1], np.ndarray) and recorded_sigmas[-1].shape == (5,)

    # 1b. Backward-compatible 5-output v4.5/v5.1 ONNX -> explicitly passes keyline_sigmas_px=None
    bundle_v51 = load_studio_models(
        device_pref="cpu",
        v52_land_dir=tmp_path / "absent_v52_land",
        prefer_onnx=True,
        smoke_fast=False,
    )
    engine_v51 = StudioInferenceEngine(bundle_v51)
    out_v51 = engine_v51.infer_runway_land(frame, return_jpeg=False)
    json.dumps(out_v51, allow_nan=False)
    assert out_v51["onnx_output_contract"] == "v51_5output"
    assert out_v51["keyline_sigma_px"] is None
    assert out_v51["mean_keyline_sigma_px"] is None
    assert out_v51["pnpl"]["used_keyline_sigmas_px"] is False
    assert out_v51["pnpl"]["huber_tukey_irls"] is True
    assert recorded_sigmas[-1] is None

    # 2. Real exported 6-output XvsLandV52Model FP16 ONNX promoted via opt-in SHA-256 pin
    torch.manual_seed(20261124)
    v52_model = XvsLandV52Model(roi_width=256, roi_height=256, split_ratio=2.0).eval()
    exp_dir = tmp_path / "land_v52_export"
    exp_rep = export_xvs_land_v52_onnx(
        v52_model,
        exp_dir,
        roi_size=256,
        split_ratio=2.0,
        opset=17,
        export_fp16=True,
        bench_warmup=1,
        bench_iters=2,
    )
    v52_fp16_path = Path(str(exp_rep["fp16_onnx_path"]))
    assert v52_fp16_path.name == XVS_LAND_V52_FP16_ONNX_FILENAME
    v52_sha = compute_file_sha256(v52_fp16_path)

    bundle_v52 = load_studio_models(
        device_pref="cpu",
        v52_land_dir=exp_dir,
        v52_land_expected_sha256=v52_sha,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_v52.meta["xvs_land"].tier == "v52_land_fp16_onnx"
    assert bundle_v52.meta["xvs_land"].backend == "onnxruntime"
    assert bundle_v52.meta["xvs_land"].dtype == "fp16"
    assert bundle_v52.meta["xvs_land"].sha256 == v52_sha
    assert bundle_v52.meta["xvs_land"].tensor_count == 7  # 1 input + 6 outputs
    assert bundle_v52.meta["xvs_land"].param_count == XVS_LAND_V52_PARAM_COUNT
    assert bundle_v52.land_has_keyline_sigma is True

    engine_v52 = StudioInferenceEngine(bundle_v52)
    out_v52 = engine_v52.infer_runway_land(frame, return_jpeg=True)
    json.dumps(out_v52, allow_nan=False)
    assert out_v52["onnx_output_contract"] == "v52_6output"
    assert isinstance(out_v52["keyline_sigma_px"], list) and len(out_v52["keyline_sigma_px"]) == 5
    assert all(math.isfinite(s) and s > 0.0 for s in out_v52["keyline_sigma_px"])
    assert out_v52["pnpl"]["used_keyline_sigmas_px"] is True
    assert isinstance(recorded_sigmas[-1], np.ndarray) and recorded_sigmas[-1].shape == (5,)
    assert np.allclose(recorded_sigmas[-1], np.asarray(out_v52["keyline_sigma_px"]), atol=1e-3)

    # 3. Synthetic 6-output geometry test: per-line keyline_sigma_px + Huber-Tukey IRLS recovers exact NED pose
    pts_ned, kl_pairs = build_ksfo_28l_reference_features()
    hdg = math.radians(297.7)
    pitch = math.radians(-3.0)
    roll = math.radians(0.5)
    fx, fy, cx, cy = 420.0, 420.0, 128.0, 128.0
    true_pos = np.array([-850.0 * math.cos(hdg), -850.0 * math.sin(hdg), -52.0], dtype=np.float64)
    uv_true, _ = project_runway_points_ned(pts_ned, true_pos, hdg, pitch, roll, fx, fy, cx, cy)
    lines_true = np.vstack([compute_hesse_line_from_points(uv_true[i0], uv_true[i1]) for i0, i1 in kl_pairs])
    covs_tight = np.broadcast_to(np.eye(2, dtype=np.float64) * 0.25, (12, 2, 2)).copy()

    fake_6_sess = _FakeLandOrtSession(
        kpts_xy=uv_true,
        keylines_hesse=lines_true,
        cov_2x2=covs_tight,
        keyline_sigma_px=np.array([1.0, 1.0, 1.5, 2.0, 25.0], dtype=np.float32),
        roi_hw=(256, 256),
    )
    bundle_fake = load_studio_models(device_pref="cpu", smoke_fast=True)
    bundle_fake.land_session = fake_6_sess  # type: ignore[assignment]
    eng_fake = StudioInferenceEngine(bundle_fake)
    out_fake = eng_fake.infer_runway_land(
        frame,
        return_jpeg=False,
        pitch_rad=pitch,
        roll_rad=roll,
        init_pos_ned=true_pos + np.array([25.0, -15.0, 6.0]),
        init_yaw_rad=hdg + math.radians(1.5),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pts_ned=pts_ned,
        keyline_pairs=kl_pairs,
    )
    assert "keyline_sigma_px" in fake_6_sess.last_fetch_names
    assert out_fake["onnx_output_contract"] == "v52_6output"
    est_pos = np.asarray(out_fake["pnpl"]["pos_ned_m"], dtype=np.float64)
    assert float(np.linalg.norm(est_pos - true_pos)) < 2.0

    # 4. Fail-closed rejection on NaN, +Inf, -Inf, 0.0, negative, and wrong-length keyline_sigma_px
    assert validate_keyline_sigmas_px(None) is None
    for bad in (
        [1.0, float("nan"), 1.5, 2.0, 2.0],
        [1.0, float("inf"), 1.5, 2.0, 2.0],
        [1.0, float("-inf"), 1.5, 2.0, 2.0],
        [1.0, 0.0, 1.5, 2.0, 2.0],
        [1.0, -0.5, 1.5, 2.0, 2.0],
        [1.0, 2.0, 3.0],
    ):
        with pytest.raises(ValueError):
            validate_keyline_sigmas_px(bad)

    bad_6_sess = _FakeLandOrtSession(
        kpts_xy=uv_true,
        keylines_hesse=lines_true,
        cov_2x2=covs_tight,
        keyline_sigma_px=np.array([1.0, float("nan"), 1.5, 2.0, 2.0], dtype=np.float32),
        roi_hw=(256, 256),
    )
    bundle_fake.land_session = bad_6_sess  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Invalid keyline_sigma_px"):
        eng_fake.infer_runway_land(frame, return_jpeg=False)


def test_models_loader_v52_opt_in_fail_closed_preserves_v51_pinned_onnx(tmp_path: Path) -> None:
    """Verify Wave 3A promoted v5.2 Land+Thermal + held Faz-B Nav, and fail-closed fallback on rogue v5.2 files."""
    # 1. Default loader loads promoted v5.2 Land & Thermal ONNX + held v5.1 Faz-B Nav ONNX cleanly
    b_default = load_studio_models(device_pref="cpu", prefer_onnx=True, smoke_fast=False)
    assert b_default.meta["xvs_d2"].sha256 == EXPECTED_D2_ONNX_SHA256
    if (DEFAULT_V52_LAND_DIR / XVS_LAND_V52_FP16_ONNX_FILENAME).is_file():
        assert b_default.meta["xvs_land"].tier == "v52_land_fp16_onnx"
        assert b_default.meta["xvs_land"].sha256 == EXPECTED_V52_LAND_FP16_ONNX_SHA256
        assert b_default.land_has_keyline_sigma is True
    elif (DEFAULT_WAVE_C_LAND_DIR / "xvs_land_v45.onnx").is_file():
        assert b_default.meta["xvs_land"].tier == "wave_c_v45_onnx"
        assert b_default.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_ONNX_SHA256
        assert b_default.land_has_keyline_sigma is False
    if (DEFAULT_V52_THERMAL_DIR / XVS_THERMAL_V52_FP16_ONNX_FILENAME).is_file():
        assert b_default.meta["xvs_thermal"].tier == "v52_thermal_fp16_onnx"
        assert b_default.meta["xvs_thermal"].sha256 == EXPECTED_V52_THERMAL_FP16_ONNX_SHA256
    elif (DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_v45_fp32.onnx").is_file():
        assert b_default.meta["xvs_thermal"].tier == "wave_c_v45_onnx"
        assert b_default.meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_ONNX_SHA256
    if (DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_ONNX_FILENAME).is_file():
        assert b_default.meta["xvs_nav"].tier == "v5_nav_fazb_onnx"
        assert b_default.meta["xvs_nav"].sha256 == EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256

    # 2. Rogue/tampered v5.2 files placed in custom v5.2 dirs MUST fail closed and fall back to pinned Wave-C / Faz-B
    rogue_dir = tmp_path / "rogue_v52"
    rogue_dir.mkdir()
    (rogue_dir / XVS_LAND_V52_FP16_ONNX_FILENAME).write_bytes(b"rogue-unpinned-v52-land")
    (rogue_dir / XVS_THERMAL_V52_FP16_ONNX_FILENAME).write_bytes(b"rogue-unpinned-v52-thermal")
    (rogue_dir / XVS_NAV_V52_FAZC_FP16_ONNX_FILENAME).write_bytes(b"rogue-unpinned-v52-nav")

    b_fallback = load_studio_models(
        device_pref="cpu",
        v52_land_dir=rogue_dir,
        v52_thermal_dir=rogue_dir,
        v52_nav_dir=rogue_dir,
        prefer_onnx=True,
        smoke_fast=False,
    )
    if (DEFAULT_WAVE_C_LAND_DIR / "xvs_land_v45.onnx").is_file():
        assert b_fallback.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_ONNX_SHA256
        assert b_fallback.meta["xvs_land"].tier == "wave_c_v45_onnx"
    if (DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_v45_fp32.onnx").is_file():
        assert b_fallback.meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_ONNX_SHA256
        assert b_fallback.meta["xvs_thermal"].tier == "wave_c_v45_onnx"
    assert b_fallback.meta["xvs_nav"].sha256 == b_default.meta["xvs_nav"].sha256
    assert b_fallback.meta["xvs_nav"].tier == b_default.meta["xvs_nav"].tier
