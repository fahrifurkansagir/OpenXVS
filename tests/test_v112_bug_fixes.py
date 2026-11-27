"""Regression tests for all 7 Wave-1 Python runtime/safety/thermal bugs fixed in OpenXVS v1.1.2.

Covers:
  - BUG-01: `StudioInferenceEngine` wires pinned `bundle.nav_lighterglue_onnx_path` into
            `LighterGlueFallbackRunner` (with graceful Sinkhorn fallback when file missing).
  - BUG-02: `infer_crossview_nav` passes `img_hw_m=(h_ref, w_ref)` to `LighterGlueFallbackRunner.match`
            so `224x224` map keypoints are normalized cleanly on 720p/1080p query frames.
  - BUG-03: `infer_thermal_dve` assesses native-resolution `lwir_bgr` & `eo_bgr` in `SensorHealthMonitor`
            before `INTER_AREA` downscaling so 720p white-noise & 1080p 1-px hot FPN stripes fail closed.
  - BUG-04: `coarse_heading.py` & `sequence_vpr.py` accept & forward `color_order` (`"rgb"` default,
            `"bgr"` for OpenCV BGR frames) into `XFeatOnnxRunner.extract(...)`.
  - BUG-05: `D2TrackerState` defaults TBC `reject_static_clutter=True` (honoring overrides), and
            HD tile probes run unified `horizon_tiling` preserving ConvGRU `c1..c5` & `h_prev` history.
  - BUG-06: CPU ORT sessions in `models_loader.py` & `stage2_xfeat.py` clamp `intra_op_num_threads`
            to `<= OMP_NUM_THREADS` when explicitly set.
  - BUG-07: `StudioHilBridgeManager` skips re-evaluating `DatumAlignmentGuard.gate_ekf3_switch_latch`
            and preserves `last_datum_decision` once `EKF3=2` is already latched (no spurious
            `PRE_SWITCH_VERT_INNOV_EXCEEDED` during DAA climbs).
"""

from __future__ import annotations

from pathlib import Path
import socket
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from xvs.v2.studio_bridge import hil_bridge as hb_mod
from xvs.v2.studio_bridge import inference_engine as ie_mod
from xvs.v2.studio_bridge import models_loader as ml_mod
from xvs.v2.studio_bridge.hil_bridge import (
    DATUM_REASON_OK,
    DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED,
    KSFO_28L_GEOID18_UNDULATION_M,
    DatumAlignmentGuard,
    StudioHilBridgeManager,
    msl_to_wgs84_hae,
)
from xvs.v2.studio_bridge.inference_engine import D2TrackerState, StudioInferenceEngine
from xvs.v2.studio_bridge.models_loader import (
    DEFAULT_D2_ONNX,
    DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
    EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256,
    EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256,
    XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME,
    XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME,
    load_studio_models,
)
from xvs.v2.xvs_d2.track_before_confirm import BearingDetection, TrackBeforeConfirmConfig
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MSGID_COMMAND_LONG,
    MavCommandLong,
    MavGlobalPositionInt,
    decode_frame,
    encode_frame,
)
try:
    from xvs.v2.xvs_nav import stage2_xfeat as s2_mod
    from xvs.v2.xvs_nav.coarse_heading import (
        COLD_START_4X90_HYPOTHESES_DEG,
        extract_coarse_heading_query_bank,
        match_with_coarse_heading_sweep,
    )
    from xvs.v2.xvs_nav.sequence_vpr import XFeatLighterGlueReranker
    from xvs.v2.xvs_nav.stage2_xfeat import LighterGlueFallbackRunner, XFeatOnnxRunner
    _HAS_XVS_NAV = True
except ImportError:  # Clean-room snapshot with --exclude-nav-source
    s2_mod = None  # type: ignore[assignment]
    COLD_START_4X90_HYPOTHESES_DEG = (0.0, 90.0, 180.0, 270.0)  # type: ignore[assignment]
    extract_coarse_heading_query_bank = None  # type: ignore[assignment]
    match_with_coarse_heading_sweep = None  # type: ignore[assignment]
    XFeatLighterGlueReranker = None  # type: ignore[assignment,misc]
    LighterGlueFallbackRunner = None  # type: ignore[assignment,misc]
    XFeatOnnxRunner = None  # type: ignore[assignment,misc]
    _HAS_XVS_NAV = False
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
)


def _make_textured_bgr(h: int, w: int, seed: int = 17) -> np.ndarray:
    """Create a rich, high-contrast natural-like BGR scene with distinct color channels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    b = np.clip(45 + (xx * 130) // max(1, w) + (yy % 29), 15, 240).astype(np.uint8)
    g = np.clip(70 + (yy * 110) // max(1, h) + ((xx * 3) % 23), 15, 240).astype(np.uint8)
    r = np.clip(180 - (xx * 120) // max(1, w) + ((xx + yy) % 19), 15, 240).astype(np.uint8)
    img = np.stack([b, g, r], axis=-1)
    for _ in range(45):
        x0 = int(rng.integers(12, max(13, w - 36)))
        y0 = int(rng.integers(12, max(13, h - 36)))
        bw = int(rng.integers(10, 34))
        bh = int(rng.integers(10, 34))
        col = tuple(int(c) for c in rng.integers(25, 235, size=3))
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), col, -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (20, 20, 20), 1)
    cv2.line(img, (10, h // 3), (w - 10, (2 * h) // 3), (235, 210, 60), 3, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 2), max(12, min(h, w) // 10), (40, 195, 230), 2, cv2.LINE_AA)
    return img


def _alloc_free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


# ---- BUG-01: StudioInferenceEngine wires bundle.nav_lighterglue_onnx_path ----


def test_bug01_lighterglue_onnx_wired_into_studio_engine_with_graceful_fallback(tmp_path: Path) -> None:
    """BUG-01: Pinned LighterGlue ONNX loads on StudioInferenceEngine; missing file falls back cleanly."""
    if not _HAS_XVS_NAV:
        pytest.skip("xvs_nav source excluded in clean-room snapshot")
    lg_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME
    if not lg_path.is_file():
        pytest.skip("Faz-B LighterGlue FP16 ONNX absent locally")

    # 1. Smoke bundle with pinned LighterGlue ONNX wired -> backend == "lighterglue_onnx"
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    bundle.nav_lighterglue_onnx_path = lg_path
    bundle.nav_lighterglue_sha256 = EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256
    engine = StudioInferenceEngine(bundle)
    assert engine._nav_lighterglue is not None
    assert engine._nav_lighterglue.backend == "lighterglue_onnx"
    assert engine._nav_lighterglue.onnx_sha256 == EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256

    sample = _make_textured_bgr(256, 320, seed=101)
    nav_res = engine.infer_crossview_nav(sample, ref_bgr=sample, return_jpeg=False)
    assert nav_res["stage2_xfeat"]["lighterglue_backend"] == "lighterglue_onnx"
    assert nav_res["stage2_xfeat"]["lighterglue_onnx_sha256"] == EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256

    # 2. Missing ONNX path -> graceful Sinkhorn fallback (no crash)
    bundle_missing = load_studio_models(device_pref="cpu", smoke_fast=True)
    bundle_missing.nav_lighterglue_onnx_path = tmp_path / "nonexistent_lighterglue.onnx"
    bundle_missing.nav_lighterglue_sha256 = EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256
    engine_fb = StudioInferenceEngine(bundle_missing)
    assert engine_fb._nav_lighterglue is not None
    assert engine_fb._nav_lighterglue.backend == "reference_lighterglue_sinkhorn_compat"


# ---- BUG-02: infer_crossview_nav forwards img_hw_m=(h_ref, w_ref) to LighterGlue ----


def test_bug02_infer_crossview_nav_forwards_img_hw_m_for_mismatched_resolutions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG-02: 720p query vs 224x224 ref must pass img_hw=(720,1280) and img_hw_m=(224,224)."""
    if not _HAS_XVS_NAV:
        pytest.skip("xvs_nav source excluded in clean-room snapshot")
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    lg_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME
    xf_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME
    if lg_path.is_file() and xf_path.is_file():
        bundle.nav_lighterglue_onnx_path = lg_path
        bundle.nav_lighterglue_sha256 = EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256
        bundle.nav_xfeat_onnx_path = xf_path
        bundle.nav_xfeat_sha256 = EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256

    engine = StudioInferenceEngine(bundle)
    assert engine._nav_lighterglue is not None

    recorded_calls: list[dict[str, Any]] = []
    orig_match = engine._nav_lighterglue.match

    def _spy_match(*args: Any, **kwargs: Any):
        recorded_calls.append({"img_hw": kwargs.get("img_hw"), "img_hw_m": kwargs.get("img_hw_m")})
        return orig_match(*args, **kwargs)

    # Force Tier-2B trigger so LighterGlue.match is guaranteed to execute on both calls
    monkeypatch.setattr(
        ie_mod,
        "match_mnn",
        lambda dq, dr, **kw: (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(engine._nav_lighterglue, "match", _spy_match)

    query_720p = _make_textured_bgr(720, 1280, seed=202)
    ref_224 = cv2.resize(query_720p, (224, 224), interpolation=cv2.INTER_AREA)

    # Case A: explicit 224x224 ref_bgr against 720p query
    res_explicit = engine.infer_crossview_nav(query_720p, ref_bgr=ref_224, return_jpeg=False)
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["img_hw"] == (720, 1280)
    assert recorded_calls[0]["img_hw_m"] == (224, 224)
    assert res_explicit["stage2_xfeat"]["tier2b_triggered"] is True

    # Case B: default KSFO 224x224 gallery (ref_bgr=None) against 720p query
    res_gallery = engine.infer_crossview_nav(query_720p, ref_bgr=None, return_jpeg=False)
    assert len(recorded_calls) == 2
    assert recorded_calls[1]["img_hw"] == (720, 1280)
    assert recorded_calls[1]["img_hw_m"] == (224, 224)
    assert res_gallery["stage2_xfeat"]["tier2b_triggered"] is True


# ---- BUG-03 [HIGH SAFETY]: infer_thermal_dve native-resolution SensorHealthMonitor ----


def test_bug03_infer_thermal_dve_catches_720p_white_noise_and_1080p_1px_fpn() -> None:
    """BUG-03: Native-res SensorHealthMonitor fails closed on 720p white-noise & 1080p 1-px FPN."""
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)

    clean_eo_720 = _make_textured_bgr(720, 1280, seed=301)
    clean_lwir_720 = _make_textured_bgr(720, 1280, seed=302)

    # 1. Baseline healthy 720p EO + LWIR passes cleanly
    sh_nom = engine.new_sensor_health_monitor()
    res_nom = engine.infer_thermal_dve(
        clean_eo_720,
        lwir_bgr=clean_lwir_720,
        q_eo=0.90,
        q_lwir=0.95,
        sensor_health_monitor=sh_nom,
        enforce_sensor_health_gate=True,
        return_jpeg=False,
    )
    assert res_nom["q_eo"] >= 0.50
    assert res_nom["q_lwir"] >= 0.50
    assert res_nom["sensor_health"]["reasons"] == []
    assert res_nom["simplex_mode"] == "NOMINAL_FUSION"

    # 2. 720p uncorrelated white-noise LWIR MUST drop q_lwir -> ~0.00 & flag LWIR_SENSOR_NOISE_FAULT
    rng = np.random.default_rng(42)
    wn_720 = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    sh_wn = engine.new_sensor_health_monitor()
    res_wn = engine.infer_thermal_dve(
        clean_eo_720,
        lwir_bgr=wn_720,
        q_eo=0.90,
        q_lwir=0.95,
        sensor_health_monitor=sh_wn,
        enforce_sensor_health_gate=True,
        return_jpeg=False,
    )
    assert res_wn["q_lwir"] <= 0.01
    assert res_wn["sensor_health"]["q_lwir_effective"] <= 0.01
    assert "LWIR_SENSOR_NOISE_FAULT" in res_wn["sensor_health"]["reasons"]
    assert res_wn["simplex_mode"] in ("LWIR_DEGRADED_EO_PRIMARY", "DUAL_DVE_SVS_FALLBACK")

    # 3. 1080p 1-px hot FPN stripes (every 8th row = 255) MUST drop q_lwir == 0.0 & flag LWIR_DEAD_PIXEL_FPN_FAULT
    clean_eo_1080 = _make_textured_bgr(1080, 1920, seed=303)
    fpn_lwir_1080 = _make_textured_bgr(1080, 1920, seed=304)
    fpn_lwir_1080[8:-8:8, :, :] = 255
    sh_fpn = engine.new_sensor_health_monitor()
    res_fpn = engine.infer_thermal_dve(
        clean_eo_1080,
        lwir_bgr=fpn_lwir_1080,
        q_eo=0.90,
        q_lwir=0.95,
        sensor_health_monitor=sh_fpn,
        enforce_sensor_health_gate=True,
        return_jpeg=False,
    )
    assert res_fpn["q_lwir"] == 0.0
    assert res_fpn["sensor_health"]["q_lwir_effective"] == 0.0
    assert "LWIR_DEAD_PIXEL_FPN_FAULT" in res_fpn["sensor_health"]["reasons"]
    assert res_fpn["simplex_mode"] in ("LWIR_DEGRADED_EO_PRIMARY", "DUAL_DVE_SVS_FALLBACK")

    # 4. Native 1080p EO with 1-px FPN stripes MUST also fail closed on EO while keeping clean LWIR primary
    fpn_eo_1080 = _make_textured_bgr(1080, 1920, seed=305)
    fpn_eo_1080[8:-8:8, :, :] = 255
    clean_lwir_1080 = _make_textured_bgr(1080, 1920, seed=306)
    sh_eo_fpn = engine.new_sensor_health_monitor()
    res_eo_fpn = engine.infer_thermal_dve(
        fpn_eo_1080,
        lwir_bgr=clean_lwir_1080,
        q_eo=0.90,
        q_lwir=0.95,
        sensor_health_monitor=sh_eo_fpn,
        enforce_sensor_health_gate=True,
        return_jpeg=False,
    )
    assert res_eo_fpn["q_eo"] == 0.0
    assert "EO_DEAD_PIXEL_FPN_FAULT" in res_eo_fpn["sensor_health"]["reasons"]
    assert res_eo_fpn["simplex_mode"] == "EO_DEGRADED_LWIR_PRIMARY"


# ---- BUG-04: color_order forwarding in coarse_heading.py & sequence_vpr.py ----


def test_bug04_coarse_heading_and_sequence_vpr_forward_color_order() -> None:
    """BUG-04: Forward optional color_order ('rgb' default, 'bgr' for OpenCV) to xfeat_runner.extract."""
    if not _HAS_XVS_NAV:
        pytest.skip("xvs_nav source excluded in clean-room snapshot")
    class _SpyXFeatRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract(self, image_lum: np.ndarray, *, color_order: str = "rgb"):
            self.calls.append(color_order)
            kpts = np.array([[40.0, 40.0], [180.0, 40.0], [40.0, 180.0], [180.0, 180.0]], dtype=np.float32)
            descs = np.zeros((4, 64), dtype=np.float32)
            for i in range(4):
                descs[i, i] = 1.0
            scores = np.ones((4,), dtype=np.float32)
            return kpts, descs, scores

    img = _make_textured_bgr(256, 256, seed=401)

    # 1. coarse_heading defaults to "rgb" and forwards explicit "bgr"
    spy_default = _SpyXFeatRunner()
    extract_coarse_heading_query_bank(img, spy_default, angles_deg=(0.0, 90.0))  # type: ignore[arg-type]
    assert spy_default.calls == ["rgb", "rgb"]

    spy_bgr = _SpyXFeatRunner()
    km, dm, _ = spy_bgr.extract(img, color_order="bgr")
    spy_bgr.calls.clear()
    match_with_coarse_heading_sweep(
        img,
        km,
        dm,
        xfeat_runner=spy_bgr,  # type: ignore[arg-type]
        angles_deg=(0.0, 90.0),
        color_order="bgr",
    )
    assert spy_bgr.calls == ["bgr", "bgr"]

    with pytest.raises(ValueError, match="Unsupported color_order"):
        extract_coarse_heading_query_bank(img, spy_bgr, angles_deg=(0.0,), color_order="yuv")  # type: ignore[arg-type]

    # 2. XFeatLighterGlueReranker forwards color_order to gallery loader, query extract, and coarse sweep
    spy_rr = _SpyXFeatRunner()
    rr = XFeatLighterGlueReranker(
        xfeat_runner=spy_rr,  # type: ignore[arg-type]
        gallery_image_loader=lambda _idx: img,
        img_hw=(256, 256),
        color_order="bgr",
    )
    rr.rerank(img, candidate_indices=[0], candidate_seq_scores=[0.8])
    assert spy_rr.calls == ["bgr", "bgr"]  # query ndarray + gallery tile 0

    spy_rr_dict = _SpyXFeatRunner()
    rr_dict = XFeatLighterGlueReranker(
        xfeat_runner=spy_rr_dict,  # type: ignore[arg-type]
        gallery_raw_features={0: (km, dm)},
        img_hw=(256, 256),
        color_order="rgb",
    )
    rr_dict.rerank({"image": img, "color_order": "bgr"}, candidate_indices=[0], candidate_seq_scores=[0.8])
    assert spy_rr_dict.calls == ["bgr"]

    spy_rr_sweep = _SpyXFeatRunner()
    rr_sweep = XFeatLighterGlueReranker(
        xfeat_runner=spy_rr_sweep,  # type: ignore[arg-type]
        gallery_image_loader=lambda _idx: img,
        img_hw=(256, 256),
        coarse_heading_angles_deg=(0.0, 90.0),
        color_order="bgr",
    )
    rr_sweep.rerank({"image": img}, candidate_indices=[0], candidate_seq_scores=[0.8])
    assert spy_rr_sweep.calls == ["bgr", "bgr", "bgr"]  # 2 sweep rotations + gallery tile 0

    # 3. Real 3-ch XFeat ONNX bit-exact parity: BGR(color_order="bgr") == RGB(color_order="rgb")
    xf_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME
    if xf_path.is_file():
        runner = XFeatOnnxRunner(xf_path, expected_sha256=EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256, max_keypoints=256)
        assert runner.input_channels == 3
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bank_bgr = extract_coarse_heading_query_bank(img, runner, angles_deg=COLD_START_4X90_HYPOTHESES_DEG, color_order="bgr")
        bank_rgb = extract_coarse_heading_query_bank(img_rgb, runner, angles_deg=COLD_START_4X90_HYPOTHESES_DEG, color_order="rgb")
        assert len(bank_bgr) == len(bank_rgb) == 4
        for (_, kb, db, _, _), (_, kr, dr, _, _) in zip(bank_bgr, bank_rgb):
            assert np.allclose(kb, kr, atol=1e-5)
            assert np.allclose(db, dr, atol=1e-5)


# ---- BUG-05: D2TrackerState reject_static_clutter=True + stateful HD tile probes ----


def test_bug05_d2_tbc_static_clutter_default_and_stateful_tile_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG-05: TBC rejects static clutter by default (overridable) and HD tiles keep ConvGRU history."""
    # 1. D2TrackerState defaults reject_static_clutter=True and honors explicit overrides
    st_default = D2TrackerState()
    assert st_default.bearing_tbc.cfg.reject_static_clutter is True

    st_off_flag = D2TrackerState(reject_static_clutter=False)
    assert st_off_flag.bearing_tbc.cfg.reject_static_clutter is False

    custom_cfg = TrackBeforeConfirmConfig(confirm_m=2, confirm_n=4, reject_static_clutter=False)
    st_custom = D2TrackerState(tbc_config=custom_cfg)
    assert st_custom.bearing_tbc.cfg.reject_static_clutter is False
    assert st_custom.bearing_tbc.cfg.confirm_m == 2

    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)
    assert engine._demo_d2_state.bearing_tbc.cfg.reject_static_clutter is True
    assert engine._fast_d2_state.bearing_tbc.cfg.reject_static_clutter is True
    assert engine.new_d2_tracker_state().bearing_tbc.cfg.reject_static_clutter is True
    assert engine.new_d2_tracker_state(reject_static_clutter=False).bearing_tbc.cfg.reject_static_clutter is False

    # Static non-growing clutter at fixed bearing MUST NOT confirm under default reject_static_clutter=True
    sky = _make_textured_bgr(240, 320, seed=501)
    iso_static = engine.new_d2_tracker_state()
    for _ in range(5):
        res_s = engine.infer_eo_sky(
            sky,
            dt_s=0.1,
            tracker_state=iso_static,
            tbc_detections=[BearingDetection(az_rad=0.05, el_rad=0.02, score=0.40, modality="eo", size_px=10.0)],
            return_jpeg=False,
        )
    assert res_s["bearing_track_confirmed"] is False

    # CBDR head-on closing intruder (zero bearing rate, growing size_px) MUST still confirm!
    iso_cbdr = engine.new_d2_tracker_state()
    for step_i, sz in enumerate((10.0, 11.2, 12.6, 14.0)):
        res_c = engine.infer_eo_sky(
            sky,
            dt_s=0.1,
            tracker_state=iso_cbdr,
            tbc_detections=[BearingDetection(az_rad=0.05, el_rad=0.02, score=0.40, modality="eo", size_px=sz)],
            return_jpeg=False,
        )
    assert res_c["bearing_track_confirmed"] is True

    # 2. HD 720p multi-crop fallback on 256x256 D2 invokes horizon_tiling & preserves c1..c5 + h_prev!
    if engine.bundle.d2_session is None:
        pytest.skip("XVS-D2 ONNX session absent locally")

    real_run = engine.bundle.d2_session.run
    per_frame_tile_inputs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    current_frame_calls: list[tuple[np.ndarray, np.ndarray]] = []
    horizon_calls = 0
    real_extract_tiles = ie_mod.extract_horizon_tiles

    def _spy_extract_tiles(*args: Any, **kwargs: Any):
        nonlocal horizon_calls
        horizon_calls += 1
        return real_extract_tiles(*args, **kwargs)

    def _spy_d2_run(output_names: Any, feed: dict[str, np.ndarray]):
        current_frame_calls.append(
            (np.asarray(feed["x_t"], dtype=np.float32).copy(), np.asarray(feed["h_prev_warped"], dtype=np.float32).copy())
        )
        return real_run(output_names, feed)

    monkeypatch.setattr(ie_mod, "extract_horizon_tiles", _spy_extract_tiles)
    monkeypatch.setattr(engine.bundle.d2_session, "run", _spy_d2_run)

    iso_hd = engine.new_d2_tracker_state()
    # Run 6 consecutive 720p frames with a tiny moving dot and high score_thr=0.999 so tile fallback always executes
    for f_idx in range(6):
        current_frame_calls = []
        hd = np.full((720, 1280, 3), (185, 165, 145), dtype=np.uint8)
        # Mild texture + moving dot inside the horizon band
        cv2.line(hd, (0, 360), (1279, 360), (140, 150, 130), 2)
        cv2.circle(hd, (500 + f_idx * 12, 340), 5, (30, 32, 36), -1, cv2.LINE_AA)
        engine.infer_eo_sky(
            hd,
            score_thr=0.999,
            tracker_state=iso_hd,
            multi_crop_fallback=True,
            return_jpeg=False,
        )
        # Call 0 is full-frame pass; calls 1..K are horizon tile probes
        assert len(current_frame_calls) >= 2
        per_frame_tile_inputs.append(current_frame_calls[1:])

    assert horizon_calls == 6  # horizon_tiling.py executed on active 256x256 D2 for all 6 frames!

    # Frame 0: cold-start tile probes have zeros for c1..c5 and h_prev
    x0_tile0, h0_tile0 = per_frame_tile_inputs[0][0]
    assert float(np.max(np.abs(x0_tile0[0, 1:]))) == 0.0
    assert float(np.max(np.abs(h0_tile0))) == 0.0

    # Frame 1: tile probes preserve 1-frame temporal history (c1 non-zero, h_prev non-zero!)
    x1_tile0, h1_tile0 = per_frame_tile_inputs[1][0]
    assert float(np.max(np.abs(x1_tile0[0, 1]))) > 0.0  # c1 populated
    assert float(np.max(np.abs(h1_tile0))) > 0.0  # h_prev populated from frame 0 tile hidden state

    # Frame 5: tile probes preserve full 5-frame temporal history (c1, c2, c5 > 0, h_prev > 0!)
    x5_tile0, h5_tile0 = per_frame_tile_inputs[5][0]
    assert float(np.max(np.abs(x5_tile0[0, 1]))) > 0.0  # c1
    assert float(np.max(np.abs(x5_tile0[0, 2]))) > 0.0  # c2 (t-5)
    assert float(np.mean(x5_tile0[0, 5])) > 0.5  # c5 = r1 * r5 overlap mask
    assert float(np.max(np.abs(h5_tile0))) > 0.0  # h_prev


# ---- BUG-06 [LOW THERMAL LOCK]: CPU ORT sessions clamp intra_op_num_threads <= OMP_NUM_THREADS ----


def test_bug06_cpu_ort_sessions_respect_omp_num_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BUG-06: models_loader._create_ort_session, XFeatOnnxRunner, and LighterGlueFallbackRunner honor OMP_NUM_THREADS."""
    # 1. Pure helper unit tests across valid/invalid/unset OMP_NUM_THREADS
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert ml_mod._resolve_cpu_ort_threads(default_cap=6) == 1
    if _HAS_XVS_NAV and s2_mod is not None:
        assert s2_mod._resolve_cpu_ort_threads(6) == 1
        assert s2_mod._resolve_cpu_ort_threads(4) == 1

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert ml_mod._resolve_cpu_ort_threads(default_cap=6) <= 2
    if _HAS_XVS_NAV and s2_mod is not None:
        assert s2_mod._resolve_cpu_ort_threads(6) == 2
        assert s2_mod._resolve_cpu_ort_threads(1) == 1

    monkeypatch.setenv("OMP_NUM_THREADS", "invalid")
    assert 1 <= ml_mod._resolve_cpu_ort_threads(default_cap=6) <= 6
    if _HAS_XVS_NAV and s2_mod is not None:
        assert s2_mod._resolve_cpu_ort_threads(6) == 6

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert 1 <= ml_mod._resolve_cpu_ort_threads(default_cap=6) <= 6
    if _HAS_XVS_NAV and s2_mod is not None:
        assert s2_mod._resolve_cpu_ort_threads(6) == 6

    # 2. Live SessionOptions capture on ORT session creation under OMP_NUM_THREADS=1
    captured_threads: list[int] = []
    real_ort_session = ort.InferenceSession

    def _spy_ort_session(path_or_bytes: Any, sess_options: Any = None, providers: Any = None, **kwargs: Any):
        if sess_options is not None:
            captured_threads.append(int(sess_options.intra_op_num_threads))
        if isinstance(path_or_bytes, str) and path_or_bytes.endswith("dummy_cleanroom_probe.onnx"):
            return type("_DummyOrtSession", (), {"get_providers": lambda self: ["CPUExecutionProvider"]})()
        return real_ort_session(path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(ml_mod.ort, "InferenceSession", _spy_ort_session)
    if _HAS_XVS_NAV and s2_mod is not None:
        monkeypatch.setattr(s2_mod.ort, "InferenceSession", _spy_ort_session)

    if DEFAULT_D2_ONNX.is_file():
        ml_mod._create_ort_session(DEFAULT_D2_ONNX, "cpu")
        assert captured_threads[-1] == 1
    else:
        dummy_onnx = tmp_path / "dummy_cleanroom_probe.onnx"
        dummy_onnx.write_bytes(b"")
        ml_mod._create_ort_session(dummy_onnx, "cpu")
        assert captured_threads[-1] == 1

    if _HAS_XVS_NAV and XFeatOnnxRunner is not None and LighterGlueFallbackRunner is not None:
        xf_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME
        if xf_path.is_file():
            xr = XFeatOnnxRunner(xf_path, expected_sha256=EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256, ort_threads=6)
            assert xr.ort_threads == 1
            assert captured_threads[-1] == 1

        lg_path = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME
        if lg_path.is_file():
            lr = LighterGlueFallbackRunner(lg_path, expected_sha256=EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256, ort_threads=6)
            assert lr.ort_threads == 1
            assert captured_threads[-1] == 1

    assert len(captured_threads) >= 1
    assert all(t == 1 for t in captured_threads)


# ---- BUG-07 [MEDIUM COMPOUND SIL]: Do not re-gate DatumAlignmentGuard when EKF3=2 already latched ----


def test_bug07_hil_bridge_skips_datum_regating_once_ekf3_vision_latched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG-07: Once EKF3=2 is latched, DAA climbs must NOT trigger spurious PRE_SWITCH_VERT_INNOV_EXCEEDED."""
    guard = DatumAlignmentGuard(
        geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
        max_pre_switch_vert_innov_m=3.0,
    )
    gate_calls = 0
    orig_gate = guard.gate_ekf3_switch_latch

    def _spy_gate(*args: Any, **kwargs: Any):
        nonlocal gate_calls
        gate_calls += 1
        return orig_gate(*args, **kwargs)

    monkeypatch.setattr(guard, "gate_ekf3_switch_latch", _spy_gate)

    mgr = StudioHilBridgeManager(non_flight_sim_only=False, datum_guard=guard)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send_gpi(seq: int, boot_ms: int, alt_msl_m: float) -> list[Any]:
        frame = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=boot_ms,
                lat_e7=376_117_200 + seq * 100,
                lon_e7=-1_223_583_670 - seq * 100,
                alt_mm=int(round(alt_msl_m * 1000.0)),
                relative_alt_mm=int(round(max(50.0, alt_msl_m - 4.0) * 1000.0)),
                vx_cms=1500,
                vy_cms=-2200,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=seq,
            sysid=1,
            compid=1,
        )
        inj.sendto(frame, ("127.0.0.1", rx_port))
        out_msgs: list[Any] = []
        deadline = time.perf_counter() + 0.35
        while time.perf_counter() < deadline:
            try:
                pkt, _ = tx_rx.recvfrom(512)
                out_msgs.append(decode_frame(pkt))
            except socket.timeout:
                if out_msgs:
                    break
        return out_msgs

    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(0.08)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_port=tx_port, tx_command_mode="set_attitude_target")
        cruise_msl = 180.0
        cruise_hae = msl_to_wgs84_hae(cruise_msl, KSFO_28L_GEOID18_UNDULATION_M)
        mgr.update_visual_altitude(
            visual_alt_m=cruise_hae + 0.25,
            visual_datum="WGS84_HAE",
            geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
        )
        mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True, min_tx_interval_s=0.0)

        # 1. Transition EKF3: 1 -> 2 is gated once and succeeds cleanly
        msgs_sw = _send_gpi(seq=1, boot_ms=1000, alt_msl_m=cruise_msl)
        assert any(
            isinstance(m.message, MavCommandLong)
            and m.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
            and abs(m.message.param1 - float(EKF3_SOURCE_VISION_OPTICAL)) < 1e-3
            for m in msgs_sw
        )
        assert gate_calls == 1
        st_sw = mgr.status()
        assert st_sw["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL
        assert st_sw["last_datum_decision"]["allowed"] is True
        assert st_sw["last_datum_decision"]["reason_code"] == DATUM_REASON_OK

        # 2. Simulate DAA climbs (+15 m -> +25 m MSL) while EKF3=2 is ALREADY latched, without updating visual alt
        for step_idx, climb_msl in enumerate((188.0, 195.0, 205.0), start=2):
            _send_gpi(seq=step_idx, boot_ms=1000 + step_idx * 100, alt_msl_m=climb_msl)

        assert gate_calls == 1  # MUST NOT re-evaluate gate_ekf3_switch_latch while EKF3=2 already latched!
        st_climb = mgr.status()
        assert st_climb["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL
        assert st_climb["datum_inhibited_switches"] == 0
        assert st_climb["last_datum_decision"]["allowed"] is True
        assert st_climb["last_datum_decision"]["reason_code"] == DATUM_REASON_OK
        assert st_climb["last_datum_decision"]["reason_code"] != DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED

        # 3. Clear spoof & verify >=2.0s dwell reversion 2 -> 1 STILL works normally
        mgr.set_autonomy_triggers(gnss_spoof_alarm=False, gnss_healthy=True, min_tx_interval_s=0.0)
        _send_gpi(seq=10, boot_ms=2000, alt_msl_m=205.0)
        assert mgr.status()["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL  # < 2.0 s dwell
        time.sleep(2.15)
        msgs_rev = _send_gpi(seq=11, boot_ms=4200, alt_msl_m=205.0)
        assert any(
            isinstance(m.message, MavCommandLong)
            and m.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
            and abs(m.message.param1 - float(EKF3_SOURCE_GNSS)) < 1e-3
            for m in msgs_rev
        )
        assert mgr.status()["ekf3_active_source"] == EKF3_SOURCE_GNSS
        assert gate_calls == 1  # Reversion to GNSS=1 did not spuriously re-gate

        # 4. A NEW spoof attempt from EKF3=1 at 205 m vs stale 180 m visual alt MUST re-engage the datum gate & inhibit!
        mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True, min_tx_interval_s=0.0)
        msgs_re_inhibit = _send_gpi(seq=12, boot_ms=4300, alt_msl_m=205.0)
        assert not any(
            isinstance(m.message, MavCommandLong) and m.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
            for m in msgs_re_inhibit
        )
        assert gate_calls == 2
        st_inh = mgr.status()
        assert st_inh["ekf3_active_source"] == EKF3_SOURCE_GNSS
        assert st_inh["datum_inhibited_switches"] == 1
        assert st_inh["last_datum_decision"]["allowed"] is False
        assert st_inh["last_datum_decision"]["reason_code"] == DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()
