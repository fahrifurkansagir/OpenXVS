"""Integration and real-checkpoint tests for OpenXVS Studio v4.0 live 5/5 inference bridge."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import shutil
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from xvs.v2.studio_bridge import (
    DEFAULT_VRAM_CAP_MB,
    DEFAULT_WAVE_C_LAND_DIR,
    DEFAULT_WAVE_C_THERMAL_DIR,
    EXPECTED_D2_ONNX_SHA256,
    EXPECTED_WAVE_C_LAND_ONNX_SHA256,
    EXPECTED_WAVE_C_LAND_PT_SHA256,
    EXPECTED_WAVE_C_THERMAL_ONNX_SHA256,
    EXPECTED_WAVE_C_THERMAL_PT_SHA256,
    PinnedArtifactCandidate,
    StudioInferenceEngine,
    create_studio_http_server,
    load_studio_models,
    resolve_pinned_candidate,
)
from xvs.v2.studio_bridge.models_loader import (
    DEFAULT_D2_ONNX,
    DEFAULT_V5_NAV_FAZB_HARVEST_DIR,
    DEFAULT_V5_NAV_HARVEST_DIR,
    DEFAULT_V52_LAND_DIR,
    DEFAULT_V52_THERMAL_DIR,
    EXPECTED_L4_CKPT_MEMBERS,
    EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256,
    EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256,
    EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256,
    EXPECTED_V5_NAV_FP16_ONNX_SHA256,
    EXPECTED_V5_XFEAT_ONNX_SHA256,
    EXPECTED_V52_LAND_FP16_ONNX_SHA256,
    EXPECTED_V52_THERMAL_FP16_ONNX_SHA256,
    V52_LAND_DEFAULT_STUDENT_T_NU,
    V52_LAND_DEFAULT_STUDENT_T_SCALE,
    WAVE_C_LAND_DEFAULT_STUDENT_T_NU,
    WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE,
    XVS_LAND_PARAM_COUNT,
    XVS_LAND_V52_FP16_ONNX_FILENAME,
    XVS_LAND_V52_PARAM_COUNT,
    XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME,
    XVS_NAV_V5_FAZB_ONNX_FILENAME,
    XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME,
    XVS_NAV_V5_ONNX_FILENAME,
    XVS_NAV_V5_XFEAT_ONNX_FILENAME,
    XVS_THERMAL_PARAM_COUNT,
    XVS_THERMAL_V52_FP16_ONNX_FILENAME,
    XVS_THERMAL_V52_PARAM_COUNT,
    _create_ort_session,
    compute_file_sha256,
    get_process_vram_mb,
)


def _make_sample_bgr(w: int = 320, h: int = 240) -> np.ndarray:
    img = np.full((h, w, 3), (180, 155, 130), dtype=np.uint8)
    cv2.rectangle(img, (90, 130), (230, 225), (62, 68, 74), -1)
    cv2.line(img, (160, 130), (160, 225), (235, 240, 245), 2, cv2.LINE_AA)
    cv2.circle(img, (158, 72), 6, (30, 35, 42), -1, cv2.LINE_AA)
    return img


def _assert_valid_b64_jpeg(b64_str: str) -> None:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 3


@pytest.fixture(scope="module")
def real_studio_engine() -> StudioInferenceEngine:
    bundle = load_studio_models(device_pref="auto", vram_cap_mb=DEFAULT_VRAM_CAP_MB, smoke_fast=False)
    return StudioInferenceEngine(bundle)


def test_all_real_checkpoints_load_and_verify_sha256(real_studio_engine: StudioInferenceEngine) -> None:
    meta = real_studio_engine.bundle.meta
    for key in ("xvs_d2", "xvs_active_range", "xvs_land", "xvs_thermal", "xvs_nav"):
        assert key in meta
        assert meta[key].loaded is True
        assert meta[key].real_weights is True

    # Exact SHA-256 verification for XVS-D2 ONNX + Wave-C / V5.0 / NB2 pinned checkpoints (D16-W19-m4)
    assert meta["xvs_d2"].sha256 == EXPECTED_D2_ONNX_SHA256

    v5_nav_fazb_onnx = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_ONNX_FILENAME
    v5_nav_onnx = DEFAULT_V5_NAV_HARVEST_DIR / XVS_NAV_V5_ONNX_FILENAME
    if v5_nav_fazb_onnx.is_file():
        assert meta["xvs_nav"].sha256 == EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256
        assert meta["xvs_nav"].backend == "onnxruntime"
        assert meta["xvs_nav"].tier == "v5_nav_fazb_onnx"
        assert meta["xvs_nav"].dtype == "fp16"
        assert real_studio_engine.bundle.nav_session is not None
        assert real_studio_engine.bundle.nav_pca_whitening is None
    elif v5_nav_onnx.is_file():
        assert meta["xvs_nav"].sha256 == EXPECTED_V5_NAV_FP16_ONNX_SHA256
        assert meta["xvs_nav"].backend == "onnxruntime"
        assert meta["xvs_nav"].tier == "v5_nav_onnx"
        assert meta["xvs_nav"].dtype == "fp16"
        assert real_studio_engine.bundle.nav_session is not None
        assert real_studio_engine.bundle.nav_pca_whitening is None
    else:
        assert meta["xvs_nav"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_nav"][1]
        assert meta["xvs_nav"].backend == "pytorch"
        assert meta["xvs_nav"].tier == "nb2_l4_pt"
        assert meta["xvs_nav"].tensor_count == 176

    v5_fazb_xfeat = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME
    v5_xfeat_onnx = DEFAULT_V5_NAV_HARVEST_DIR / XVS_NAV_V5_XFEAT_ONNX_FILENAME
    if v5_fazb_xfeat.is_file():
        assert real_studio_engine.bundle.nav_xfeat_onnx_path is not None
        assert real_studio_engine.bundle.nav_xfeat_sha256 == EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256
        assert real_studio_engine._nav_xfeat.backend == "xfeat_onnx"
    elif v5_xfeat_onnx.is_file():
        assert real_studio_engine.bundle.nav_xfeat_onnx_path is not None
        assert real_studio_engine.bundle.nav_xfeat_sha256 == EXPECTED_V5_XFEAT_ONNX_SHA256
        assert real_studio_engine._nav_xfeat.backend == "xfeat_onnx"

    v5_fazb_lg = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME
    if v5_fazb_lg.is_file():
        assert real_studio_engine.bundle.nav_lighterglue_onnx_path is not None
        assert real_studio_engine.bundle.nav_lighterglue_sha256 == EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256

    v52_land_onnx = DEFAULT_V52_LAND_DIR / XVS_LAND_V52_FP16_ONNX_FILENAME
    wave_c_land_onnx = DEFAULT_WAVE_C_LAND_DIR / "xvs_land_v45.onnx"
    if v52_land_onnx.is_file():
        assert meta["xvs_land"].sha256 == EXPECTED_V52_LAND_FP16_ONNX_SHA256
        assert meta["xvs_land"].backend == "onnxruntime"
        assert meta["xvs_land"].tier == "v52_land_fp16_onnx"
        assert meta["xvs_land"].dtype == "fp16"
        assert meta["xvs_land"].tensor_count == 7
        assert meta["xvs_land"].param_count == XVS_LAND_V52_PARAM_COUNT
        assert real_studio_engine.bundle.land_session is not None
        assert real_studio_engine.bundle.land_has_keyline_sigma is True
        assert real_studio_engine.bundle.land_calibrator.nu == pytest.approx(V52_LAND_DEFAULT_STUDENT_T_NU)
        assert real_studio_engine.bundle.land_calibrator.scale == pytest.approx(
            V52_LAND_DEFAULT_STUDENT_T_SCALE, rel=1e-5
        )
    elif wave_c_land_onnx.is_file():
        assert meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_ONNX_SHA256
        assert meta["xvs_land"].backend == "onnxruntime"
        assert meta["xvs_land"].tier == "wave_c_v45_onnx"
        assert meta["xvs_land"].tensor_count == 6
        assert meta["xvs_land"].param_count == XVS_LAND_PARAM_COUNT
        assert real_studio_engine.bundle.land_session is not None
        assert real_studio_engine.bundle.land_calibrator.nu == pytest.approx(WAVE_C_LAND_DEFAULT_STUDENT_T_NU)
        assert real_studio_engine.bundle.land_calibrator.scale == pytest.approx(
            WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE, rel=1e-5
        )
    else:
        assert meta["xvs_land"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_land"][1]
        assert meta["xvs_land"].backend == "pytorch"
        assert meta["xvs_land"].tensor_count == 75

    v52_thermal_onnx = DEFAULT_V52_THERMAL_DIR / XVS_THERMAL_V52_FP16_ONNX_FILENAME
    wave_c_thermal_onnx = DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_v45_fp32.onnx"
    if v52_thermal_onnx.is_file():
        assert meta["xvs_thermal"].sha256 == EXPECTED_V52_THERMAL_FP16_ONNX_SHA256
        assert meta["xvs_thermal"].backend == "onnxruntime"
        assert meta["xvs_thermal"].tier == "v52_thermal_fp16_onnx"
        assert meta["xvs_thermal"].dtype == "fp16"
        assert meta["xvs_thermal"].tensor_count == 10
        assert meta["xvs_thermal"].param_count == XVS_THERMAL_V52_PARAM_COUNT
        assert real_studio_engine.bundle.thermal_session is not None
    elif wave_c_thermal_onnx.is_file():
        assert meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_ONNX_SHA256
        assert meta["xvs_thermal"].backend == "onnxruntime"
        assert meta["xvs_thermal"].tier == "wave_c_v45_onnx"
        assert meta["xvs_thermal"].tensor_count == 10
        assert meta["xvs_thermal"].param_count == XVS_THERMAL_PARAM_COUNT
        assert real_studio_engine.bundle.thermal_session is not None
    else:
        assert meta["xvs_thermal"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_thermal"][1]
        assert meta["xvs_thermal"].backend == "pytorch"
        assert meta["xvs_thermal"].tensor_count == 82

    assert get_process_vram_mb() < DEFAULT_VRAM_CAP_MB


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def test_land_wave_c_sha256_tamper_and_fallback_chain(tmp_path: Path) -> None:
    wave_c_land_onnx_src = DEFAULT_WAVE_C_LAND_DIR / "xvs_land_v45.onnx"
    wave_c_land_pt_src = DEFAULT_WAVE_C_LAND_DIR / "xvs_land_final.pt"

    # 1. Tampered Wave-C ONNX + valid Wave-C .pt -> rejects ONNX, loads verified Wave-C .pt
    if wave_c_land_pt_src.is_file():
        case1 = tmp_path / "case1_land_tampered_onnx"
        case1.mkdir()
        (case1 / "xvs_land_v45.onnx").write_bytes(b"tampered-wave-c-land-onnx")
        _link_or_copy(wave_c_land_pt_src, case1 / "xvs_land_final.pt")
        b_pt = load_studio_models(
            device_pref="cpu",
            wave_c_land_dir=case1,
            prefer_onnx=True,
            smoke_fast=False,
        )
        assert b_pt.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_PT_SHA256
        assert b_pt.meta["xvs_land"].backend == "pytorch"
        assert b_pt.meta["xvs_land"].tier == "wave_c_v45_pt"
        assert b_pt.land_session is None
        assert b_pt.land_model is not None
        assert b_pt.land_calibrator.scale == pytest.approx(WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE, rel=1e-5)

    # 2. Valid Wave-C ONNX WITHOUT companion .pt -> retains exact calibrated Student-t parameters
    if wave_c_land_onnx_src.is_file():
        case2 = tmp_path / "case2_land_onnx_only"
        case2.mkdir()
        _link_or_copy(wave_c_land_onnx_src, case2 / "xvs_land_v45.onnx")
        b_onnx_only = load_studio_models(
            device_pref="cpu",
            wave_c_land_dir=case2,
            prefer_onnx=True,
            smoke_fast=False,
        )
        assert b_onnx_only.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_ONNX_SHA256
        assert b_onnx_only.meta["xvs_land"].backend == "onnxruntime"
        assert b_onnx_only.land_calibrator.nu == pytest.approx(WAVE_C_LAND_DEFAULT_STUDENT_T_NU)
        assert b_onnx_only.land_calibrator.scale == pytest.approx(
            WAVE_C_LAND_DEFAULT_STUDENT_T_SCALE, rel=1e-5
        )

    # 3. Tampered Wave-C ONNX + tampered Wave-C .pt -> falls back safely to verified NB2 .pt
    case3 = tmp_path / "case3_land_both_tampered"
    case3.mkdir()
    (case3 / "xvs_land_v45.onnx").write_bytes(b"corrupted-land-onnx")
    (case3 / "xvs_land_final.pt").write_bytes(b"corrupted-land-pt")
    b_nb2 = load_studio_models(
        device_pref="cpu",
        wave_c_land_dir=case3,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert b_nb2.meta["xvs_land"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_land"][1]
    assert b_nb2.meta["xvs_land"].backend == "pytorch"
    assert b_nb2.meta["xvs_land"].tier == "nb2_l4_pt"
    assert b_nb2.land_session is None
    assert b_nb2.land_model is not None

    # 4. Missing Wave-C land directory -> falls back safely to verified NB2 .pt
    b_missing = load_studio_models(
        device_pref="cpu",
        wave_c_land_dir=tmp_path / "nonexistent_wave_c_land",
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert b_missing.meta["xvs_land"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_land"][1]
    assert b_missing.meta["xvs_land"].tier == "nb2_l4_pt"


def test_land_wave_c_onnx_vs_pytorch_parity_and_horizon_framing(
    real_studio_engine: StudioInferenceEngine,
) -> None:
    wave_c_onnx = DEFAULT_WAVE_C_LAND_DIR / "xvs_land_v45.onnx"
    wave_c_pt = DEFAULT_WAVE_C_LAND_DIR / "xvs_land_final.pt"
    if not (wave_c_onnx.is_file() and wave_c_pt.is_file()):
        pytest.skip("Wave-C land artifacts absent locally")

    bundle_ort_v45 = load_studio_models(
        device_pref="cpu",
        v52_land_expected_sha256="",
        v52_land_fp32_expected_sha256="",
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_ort_v45.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_ONNX_SHA256
    assert bundle_ort_v45.meta["xvs_land"].backend == "onnxruntime"
    engine_ort_v45 = StudioInferenceEngine(bundle_ort_v45)

    bundle_pt = load_studio_models(device_pref="cpu", prefer_onnx=False, smoke_fast=False)
    assert bundle_pt.meta["xvs_land"].sha256 == EXPECTED_WAVE_C_LAND_PT_SHA256
    assert bundle_pt.meta["xvs_land"].backend == "pytorch"
    engine_pt = StudioInferenceEngine(bundle_pt)

    frame = _make_sample_bgr(w=640, h=360)
    rbox = (290.0, 140.0, 350.0, 180.0)
    for box in (None, rbox):
        out_ort = engine_ort_v45.infer_runway_land(frame, roi_box_px=box, return_jpeg=False)
        out_pt = engine_pt.infer_runway_land(frame, roi_box_px=box, return_jpeg=False)
        kp_ort = np.asarray(out_ort["keypoints_xy_256"], dtype=np.float64)
        kp_pt = np.asarray(out_pt["keypoints_xy_256"], dtype=np.float64)
        assert float(np.max(np.abs(kp_ort - kp_pt))) <= 0.05
        assert abs(out_ort["mean_keypoint_sigma_px"] - out_pt["mean_keypoint_sigma_px"]) <= 0.02
        assert abs(out_ort["vpl_m"] - out_pt["vpl_m"]) <= 0.2
        assert abs(out_ort["hpl_m"] - out_pt["hpl_m"]) <= 0.5

    # Verify horizon-preserving framing on real USGS NAIP KSFO-28L studio_media sequence:
    # (a) Wave-C v4.5 ONNX backward-compat bounds AND (b) promoted v5.2 6-output ONNX sharper localization + 0 HMI
    manifest_path = real_studio_engine.studio_media_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        land_frames = manifest["channels"]["runway_land"]["frames"]
        v45_errs: list[float] = []
        v45_vpls: list[float] = []
        v45_hpls: list[float] = []
        v52_errs: list[float] = []
        for rec in land_frames:
            img = cv2.imread(str(real_studio_engine.studio_media_dir / rec["file"]), cv2.IMREAD_COLOR)
            assert img is not None
            h_in, w_in = img.shape[:2]
            rbox_f = tuple(float(v) for v in rec["runway_bbox_px"])
            ox, oy, rw, rh = StudioInferenceEngine._compute_land_roi_origin(
                w_in, h_in, rbox_f, horizon_preserving=True
            )
            assert oy == 0 and rh == h_in  # Full vertical sky+horizon preserved
            gt_full = np.asarray(rec["gt_keypoints_12_px"], dtype=np.float64)

            out_v45 = engine_ort_v45.infer_runway_land(img, roi_box_px=rbox_f, return_jpeg=False)
            pred_v45_256 = np.asarray(out_v45["keypoints_xy_256"], dtype=np.float64)
            pred_v45_full = np.column_stack(
                [ox + pred_v45_256[:, 0] / 256.0 * rw, oy + pred_v45_256[:, 1] / 256.0 * rh]
            )
            v45_errs.append(float(np.mean(np.linalg.norm(pred_v45_full - gt_full, axis=1))))
            v45_vpls.append(float(out_v45["vpl_m"]))
            v45_hpls.append(float(out_v45["hpl_m"]))
            assert out_v45["hmi_count"] == 0

            out_v52 = real_studio_engine.infer_runway_land(img, roi_box_px=rbox_f, return_jpeg=False)
            pred_v52_256 = np.asarray(out_v52["keypoints_xy_256"], dtype=np.float64)
            pred_v52_full = np.column_stack(
                [ox + pred_v52_256[:, 0] / 256.0 * rw, oy + pred_v52_256[:, 1] / 256.0 * rh]
            )
            v52_errs.append(float(np.mean(np.linalg.norm(pred_v52_full - gt_full, axis=1))))
            assert out_v52["onnx_output_contract"] == "v52_6output"
            assert out_v52["roi_hw"] == [384, 384]
            assert out_v52["hmi_count"] == 0
            assert math.isfinite(out_v52["vpl_m"]) and out_v52["vpl_m"] > 0.0
            assert math.isfinite(out_v52["hpl_m"]) and out_v52["hpl_m"] > 0.0
        assert float(np.mean(v45_errs)) < 21.0
        assert float(np.mean(v45_vpls)) < 20.0
        assert float(np.mean(v45_hpls)) < 52.0
        assert float(np.mean(v52_errs)) < 15.0


def test_thermal_wave_c_sha256_tamper_and_fallback_chain(tmp_path: Path) -> None:
    # 0. Unpinned candidate (empty expected_sha256) is fail-closed skipped when verify_sha256=True
    unpinned_file = tmp_path / "unpinned.onnx"
    unpinned_file.write_bytes(b"unpinned-model-bytes")
    assert (
        resolve_pinned_candidate(
            [
                PinnedArtifactCandidate(
                    tier="wave_c_v45_onnx",
                    backend="onnxruntime",
                    path=unpinned_file,
                    expected_sha256="",
                )
            ],
            verify_sha256=True,
        )
        is None
    )

    # 1. Tampered Wave-C ONNX + valid Wave-C .pt -> rejects ONNX, loads verified Wave-C .pt
    wave_c_pt_src = DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_final.pt"
    if wave_c_pt_src.is_file():
        case1 = tmp_path / "case1_tampered_onnx"
        case1.mkdir()
        (case1 / "xvs_thermal_v45_fp32.onnx").write_bytes(b"tampered-wave-c-onnx-bytes")
        shutil.copy2(wave_c_pt_src, case1 / "xvs_thermal_final.pt")
        b_pt = load_studio_models(
            device_pref="cpu",
            wave_c_thermal_dir=case1,
            prefer_onnx=True,
            smoke_fast=False,
        )
        assert b_pt.meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_PT_SHA256
        assert b_pt.meta["xvs_thermal"].backend == "pytorch"
        assert b_pt.meta["xvs_thermal"].tier == "wave_c_v45_pt"
        assert b_pt.thermal_session is None
        assert b_pt.thermal_model is not None

    # 2. Tampered Wave-C ONNX + tampered Wave-C .pt -> falls back safely to verified NB2 .pt
    case2 = tmp_path / "case2_both_tampered"
    case2.mkdir()
    (case2 / "xvs_thermal_v45_fp32.onnx").write_bytes(b"corrupted-onnx")
    (case2 / "xvs_thermal_final.pt").write_bytes(b"corrupted-pt")
    b_nb2 = load_studio_models(
        device_pref="cpu",
        wave_c_thermal_dir=case2,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert b_nb2.meta["xvs_thermal"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_thermal"][1]
    assert b_nb2.meta["xvs_thermal"].backend == "pytorch"
    assert b_nb2.meta["xvs_thermal"].tier == "nb2_l4_pt"
    assert b_nb2.thermal_session is None
    assert b_nb2.thermal_model is not None

    # 3. Missing Wave-C directory -> falls back safely to verified NB2 .pt
    b_missing = load_studio_models(
        device_pref="cpu",
        wave_c_thermal_dir=tmp_path / "nonexistent_wave_c",
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert b_missing.meta["xvs_thermal"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_thermal"][1]
    assert b_missing.meta["xvs_thermal"].tier == "nb2_l4_pt"


def test_thermal_wave_c_onnx_vs_pytorch_parity_and_dual_blind_exact(
    real_studio_engine: StudioInferenceEngine,
) -> None:
    wave_c_onnx = DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_v45_fp32.onnx"
    wave_c_pt = DEFAULT_WAVE_C_THERMAL_DIR / "xvs_thermal_final.pt"
    if not (wave_c_onnx.is_file() and wave_c_pt.is_file()):
        pytest.skip("Wave-C thermal artifacts absent locally")

    bundle_ort_v45 = load_studio_models(
        device_pref="cpu",
        v52_thermal_expected_sha256="",
        v52_thermal_fp32_expected_sha256="",
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_ort_v45.meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_ONNX_SHA256
    assert bundle_ort_v45.meta["xvs_thermal"].backend == "onnxruntime"
    engine_ort_v45 = StudioInferenceEngine(bundle_ort_v45)

    bundle_pt = load_studio_models(device_pref="cpu", prefer_onnx=False, smoke_fast=False)
    assert bundle_pt.meta["xvs_thermal"].sha256 == EXPECTED_WAVE_C_THERMAL_PT_SHA256
    assert bundle_pt.meta["xvs_thermal"].backend == "pytorch"
    engine_pt = StudioInferenceEngine(bundle_pt)

    frame = _make_sample_bgr()
    for q_eo, q_lwir, fog in ((0.90, 0.96, 0.0), (0.08, 0.96, 0.92), (0.0, 0.0, 0.0)):
        out_ort = engine_ort_v45.infer_thermal_dve(
            frame, q_eo=q_eo, q_lwir=q_lwir, fog_density=fog, return_jpeg=False
        )
        out_pt = engine_pt.infer_thermal_dve(
            frame, q_eo=q_eo, q_lwir=q_lwir, fog_density=fog, return_jpeg=False
        )
        assert out_ort["simplex_mode"] == out_pt["simplex_mode"]
        assert abs(out_ort["mean_vacuity"] - out_pt["mean_vacuity"]) <= 1.0e-3
        assert abs(out_ort["max_heatmap_prob"] - out_pt["max_heatmap_prob"]) <= 1.0e-3
        out_v52 = real_studio_engine.infer_thermal_dve(
            frame, q_eo=q_eo, q_lwir=q_lwir, fog_density=fog, return_jpeg=False
        )
        assert out_v52["simplex_mode"] == out_pt["simplex_mode"]
        if q_eo == 0.0 and q_lwir == 0.0:
            assert out_ort["mean_vacuity"] == pytest.approx(1.0, abs=1e-6)
            assert out_pt["mean_vacuity"] == pytest.approx(1.0, abs=1e-6)
            assert out_v52["mean_vacuity"] == pytest.approx(1.0, abs=1e-6)
            assert out_ort["simplex_mode"] == "DUAL_DVE_SVS_FALLBACK"
            assert out_v52["simplex_mode"] == "DUAL_DVE_SVS_FALLBACK"


def test_thermal_dve_studio_media_wave_c_localization(
    real_studio_engine: StudioInferenceEngine,
) -> None:
    tier = real_studio_engine.bundle.meta["xvs_thermal"].tier
    if tier not in ("v52_thermal_fp16_onnx", "v52_thermal_fp32_onnx", "wave_c_v45_onnx"):
        pytest.skip("Thermal ONNX not active")
    manifest_path = real_studio_engine.studio_media_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("data/studio_media/manifest.json absent")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    th_frames = manifest["channels"]["thermal_dve"]["frames"]
    mode_matches = 0
    active_hits = 0
    active_total = 0
    dual_vacs: list[float] = []

    for rec in th_frames:
        eo_bgr = cv2.imread(str(real_studio_engine.studio_media_dir / rec["eo_file"]), cv2.IMREAD_COLOR)
        lw_bgr = cv2.imread(str(real_studio_engine.studio_media_dir / rec["lwir_file"]), cv2.IMREAD_COLOR)
        assert eo_bgr is not None and lw_bgr is not None
        out = real_studio_engine.infer_thermal_dve(
            eo_bgr,
            lwir_bgr=lw_bgr,
            q_eo=float(rec["q_eo_expected"]),
            q_lwir=float(rec["q_lwir_expected"]),
            return_jpeg=False,
        )
        if out["simplex_mode"] == rec["expected_simplex_mode"]:
            mode_matches += 1
        if rec["regime"] == "stratus_cloud":
            dual_vacs.append(float(out["mean_vacuity"]))
        else:
            active_total += 1
            px_g, py_g = out["peak_grid_xy"]
            px_img = (float(px_g) + 0.5) / 80.0 * float(rec["width"])
            py_img = (float(py_g) + 0.5) / 64.0 * float(rec["height"])
            bx1, by1, bx2, by2 = rec["aircraft_box_px"]
            if bx1 <= px_img <= bx2 and by1 <= py_img <= by2:
                active_hits += 1

    assert mode_matches == len(th_frames)
    assert len(dual_vacs) == 5 and all(abs(v - 1.0) <= 1e-6 for v in dual_vacs)
    assert active_total == 19
    assert active_hits >= (15 if tier.startswith("v52_") else 19)


def test_d2_onnx_sha256_mismatch_fails_closed(tmp_path) -> None:
    bad_onnx = tmp_path / "tampered_d2.onnx"
    bad_onnx.write_bytes(b"corrupted-onnx-bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_studio_models(device_pref="cpu", d2_onnx_path=bad_onnx, smoke_fast=True, verify_d2_sha256=True)


def test_real_forward_passes_all_4_channels_finite(real_studio_engine: StudioInferenceEngine) -> None:
    frame = _make_sample_bgr()

    # 1. EO Sky (XVS-D2 ONNX + XVS-ActiveRange)
    eo = real_studio_engine.infer_eo_sky(frame, return_jpeg=True)
    assert math.isfinite(eo["latency_ms"]) and eo["latency_ms"] >= 0.0
    assert math.isfinite(eo["max_prob"]) and 0.0 <= eo["max_prob"] <= 1.0
    ar = eo["active_range"]
    assert math.isfinite(ar["r_hat_m"]) and ar["r_hat_m"] >= 30.0
    assert math.isfinite(ar["r_safe_lower_m"]) and ar["r_safe_lower_m"] <= ar["r_hat_m"]
    _assert_valid_b64_jpeg(eo["image_b64"])

    # 2. Runway Land (XVS-Land 12-keypoint SimCC + 5 Hesse keylines + Cholesky GNLL)
    land = real_studio_engine.infer_runway_land(frame, return_jpeg=True)
    assert math.isfinite(land["latency_ms"]) and land["latency_ms"] >= 0.0
    assert len(land["keypoints_xy_256"]) == 12
    assert len(land["keylines_hesse"]) == 5
    for x, y in land["keypoints_xy_256"]:
        assert math.isfinite(x) and math.isfinite(y)
    assert math.isfinite(land["hpl_m"]) and land["hpl_m"] > 0.0
    assert math.isfinite(land["vpl_m"]) and land["vpl_m"] > 0.0
    _assert_valid_b64_jpeg(land["image_b64"])

    # 3. Thermal DVE (XVS-ThermalNet: fog penetration + exact dual-blind vacuity == 1.0)
    th_fog = real_studio_engine.infer_thermal_dve(frame, q_eo=0.08, q_lwir=0.96, fog_density=0.92, return_jpeg=True)
    assert math.isfinite(th_fog["mean_vacuity"]) and 0.0 <= th_fog["mean_vacuity"] <= 1.0
    assert th_fog["simplex_mode"] == "EO_DEGRADED_LWIR_PRIMARY"
    _assert_valid_b64_jpeg(th_fog["image_b64"])

    th_dual = real_studio_engine.infer_thermal_dve(frame, q_eo=0.0, q_lwir=0.0, return_jpeg=False)
    assert th_dual["mean_vacuity"] == pytest.approx(1.0, abs=1e-4)
    assert th_dual["simplex_mode"] == "DUAL_DVE_SVS_FALLBACK"

    # 4. Cross-View Nav (XVS-Nav DINOv2-B/14 + BoQ 512-D + PCA whitening + spoof guard)
    nav_ok = real_studio_engine.infer_crossview_nav(frame, ref_bgr=frame, gps_offset_m=3.5, return_jpeg=True)
    assert math.isfinite(nav_ok["cosine_sim"]) and nav_ok["cosine_sim"] > 0.99
    assert nav_ok["spoof_detected"] is False
    _assert_valid_b64_jpeg(nav_ok["image_b64"])

    nav_spoof = real_studio_engine.infer_crossview_nav(frame, ref_bgr=frame, gps_offset_m=185.0, return_jpeg=False)
    assert nav_spoof["spoof_detected"] is True
    assert nav_spoof["nav_mode"] == "VISUAL_PRIMARY_SPOOF_ISOLATED"

    assert get_process_vram_mb() < DEFAULT_VRAM_CAP_MB


def test_http_server_endpoints_health_demo_tick_and_analyze_image(
    real_studio_engine: StudioInferenceEngine,
) -> None:
    server = create_studio_http_server(real_studio_engine, host="127.0.0.1", port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. GET /api/health with strict CORS allowlist check (D16-W19-m2)
        req_h = urllib.request.Request(
            f"{base_url}/api/health",
            headers={"Origin": "http://127.0.0.1:4174"},
        )
        with urllib.request.urlopen(req_h, timeout=15.0) as resp:
            assert resp.status == 200
            assert resp.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:4174"
            health = json.loads(resp.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["kani_proofs"] == "18/18"
        assert "gpu_name" in health and "device" in health
        assert health["vram_mb"] < DEFAULT_VRAM_CAP_MB
        assert all(health["models"][k]["real_weights"] for k in ("xvs_d2", "xvs_land", "xvs_thermal", "xvs_nav"))

        # Untrusted origin MUST NOT receive Access-Control-Allow-Origin header
        req_evil = urllib.request.Request(
            f"{base_url}/api/health",
            headers={"Origin": "https://evil.example"},
        )
        with urllib.request.urlopen(req_evil, timeout=15.0) as resp_evil:
            assert resp_evil.status == 200
            assert resp_evil.headers["Access-Control-Allow-Origin"] is None

        # 2. GET /api/demo_tick across CBDR, thermal_fog, and gps_spoof + LRU warm hit (< 10 ms)
        real_studio_engine.clear_demo_cache()
        for scen in ("cbdr", "thermal_fog", "gps_spoof"):
            with urllib.request.urlopen(
                f"{base_url}/api/demo_tick?scenario={scen}&frame=0&images=1",
                timeout=30.0,
            ) as resp:
                assert resp.status == 200
                tick = json.loads(resp.read().decode("utf-8"))
            assert tick["scenario"] == scen
            assert tick["cached"] is False
            assert set(tick["channels"].keys()) == {"eo_sky", "runway_land", "thermal_dve", "crossview_nav"}
            for ch_data in tick["channels"].values():
                assert math.isfinite(ch_data["latency_ms"])
                assert isinstance(ch_data["summary"], str) and len(ch_data["summary"]) > 0
                assert isinstance(ch_data["checkpoint_sha256"], str) and len(ch_data["checkpoint_sha256"]) == 64
                assert ch_data["annotated_jpeg_b64"].startswith("data:image/jpeg;base64,")
                _assert_valid_b64_jpeg(ch_data["image_b64"])
            if scen == "thermal_fog":
                assert tick["channels"]["thermal_dve"]["simplex_mode"] == "EO_DEGRADED_LWIR_PRIMARY"
            if scen == "gps_spoof":
                assert tick["channels"]["crossview_nav"]["spoof_detected"] is True

        # Cached repeat tick must hit bounded 128-entry LRU in < 10 ms
        with urllib.request.urlopen(
            f"{base_url}/api/demo_tick?scenario=cbdr&frame=0&images=1",
            timeout=10.0,
        ) as resp:
            assert resp.status == 200
            tick_cached = json.loads(resp.read().decode("utf-8"))
        assert tick_cached["cached"] is True
        assert tick_cached["total_latency_ms"] < 10.0

        # 3. POST /api/analyze_image for all 4 channels with raw JPEG bytes
        ok, enc = cv2.imencode(".jpg", _make_sample_bgr())
        assert ok
        jpeg_bytes = enc.tobytes()
        for ch in ("d2", "land", "thermal", "nav"):
            req_p = urllib.request.Request(
                f"{base_url}/api/analyze_image?channel={ch}",
                data=jpeg_bytes,
                method="POST",
                headers={"Content-Type": "image/jpeg"},
            )
            with urllib.request.urlopen(req_p, timeout=30.0) as resp:
                assert resp.status == 200
                out = json.loads(resp.read().decode("utf-8"))
            assert math.isfinite(out["latency_ms"])
            _assert_valid_b64_jpeg(out["image_b64"])

        # 4. Invalid channel fails closed with 400
        req_bad = urllib.request.Request(
            f"{base_url}/api/analyze_image?channel=unknown_sensor",
            data=jpeg_bytes,
            method="POST",
            headers={"Content-Type": "image/jpeg"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad, timeout=10.0)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_smoke_fast_mode_loads_without_heavy_weights() -> None:
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)
    tick = engine.run_demo_tick(scenario="forcedland", frame_idx=1, include_images=False)
    assert tick["scenario"] == "forcedland"
    assert set(tick["channels"].keys()) == {"eo_sky", "runway_land", "thermal_dve", "crossview_nav"}


def test_ort_session_configures_dynamic_block_base_4(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    recorded_entries: list[tuple[str, str]] = []
    real_session_options = ort.SessionOptions

    class _RecordingSessionOptions(real_session_options):  # type: ignore[misc,valid-type]
        def add_session_config_entry(self, key: str, value: str) -> None:
            recorded_entries.append((str(key), str(value)))
            super().add_session_config_entry(key, value)

    monkeypatch.setattr(ort, "SessionOptions", _RecordingSessionOptions)
    sess, dev = _create_ort_session(DEFAULT_D2_ONNX, "cpu")
    assert sess is not None
    assert dev == "cpu"
    assert ("session.intra_op.allow_spinning", "0") in recorded_entries
    assert ("session.dynamic_block_base", "4") in recorded_entries


def test_nav_v5_onnx_candidate_chain_sha256_fail_closed_and_inference(tmp_path: Path) -> None:
    from xvs.v2.xvs_nav.v5_student import XvsNavV5Student, export_nav_v5_onnx

    export_dir = tmp_path / "v5_export"
    student = XvsNavV5Student(use_timm=False).eval()
    exp_res = export_nav_v5_onnx(
        export_dir,
        model=student,
        use_timm=False,
        bench_warmup=1,
        bench_iters=2,
    )
    fp16_onnx = Path(exp_res["fp16_onnx_path"])
    assert fp16_onnx.is_file()
    actual_sha = compute_file_sha256(fp16_onnx)

    # 1. Valid V5.0 ONNX + matching xvs_nav_v5_manifest.json -> promotes to v5_nav_onnx (0 external whitening)
    valid_dir = tmp_path / "harvest_valid"
    valid_dir.mkdir()
    shutil.copy2(fp16_onnx, valid_dir / XVS_NAV_V5_ONNX_FILENAME)
    (valid_dir / "xvs_nav_v5_manifest.json").write_text(
        json.dumps({"files": {XVS_NAV_V5_ONNX_FILENAME: {"sha256": actual_sha}}}),
        encoding="utf-8",
    )
    bundle_v5 = load_studio_models(
        device_pref="cpu",
        v5_nav_dir=valid_dir,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_v5.meta["xvs_nav"].tier == "v5_nav_onnx"
    assert bundle_v5.meta["xvs_nav"].backend == "onnxruntime"
    assert bundle_v5.meta["xvs_nav"].dtype == "fp16"
    assert bundle_v5.meta["xvs_nav"].sha256 == actual_sha
    assert bundle_v5.nav_session is not None
    assert bundle_v5.nav_model is None
    assert bundle_v5.nav_pca_whitening is None

    engine_v5 = StudioInferenceEngine(bundle_v5)
    frame = _make_sample_bgr()
    out_ok = engine_v5.infer_crossview_nav(frame, ref_bgr=frame, gps_offset_m=3.5, return_jpeg=True)
    assert out_ok["backend"] == "onnxruntime"
    assert out_ok["tier"] == "v5_nav_onnx"
    assert out_ok["checkpoint_sha256"] == actual_sha
    assert out_ok["cosine_sim"] > 0.999
    assert out_ok["spoof_detected"] is False
    assert out_ok["hpl_vis_m"] <= 30.0
    assert out_ok["stage2_xfeat"]["synthetic_lines_used"] is False
    _assert_valid_b64_jpeg(out_ok["image_b64"])
    json.dumps(out_ok, allow_nan=False)

    out_spoof = engine_v5.infer_crossview_nav(frame, ref_bgr=frame, gps_offset_m=185.0, return_jpeg=False)
    assert out_spoof["spoof_detected"] is True
    assert out_spoof["nav_mode"] == "VISUAL_PRIMARY_SPOOF_ISOLATED"
    assert out_spoof["dual_bank_cusum"]["active_bank"] == "bank_b_visual"
    assert out_spoof["dual_bank_cusum"]["max_cusum_stat"] >= 17.619

    # 2. Tampered V5.0 ONNX against manifest SHA-256 -> fails closed and falls back cleanly to pinned NB2 .pt
    tampered_dir = tmp_path / "harvest_tampered"
    tampered_dir.mkdir()
    (tampered_dir / XVS_NAV_V5_ONNX_FILENAME).write_bytes(b"tampered-v5-nav-onnx-bytes")
    (tampered_dir / "xvs_nav_v5_manifest.json").write_text(
        json.dumps({"files": {XVS_NAV_V5_ONNX_FILENAME: {"sha256": actual_sha}}}),
        encoding="utf-8",
    )
    bundle_fb = load_studio_models(
        device_pref="cpu",
        v5_nav_dir=tampered_dir,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_fb.meta["xvs_nav"].tier == "nb2_l4_pt"
    assert bundle_fb.meta["xvs_nav"].backend == "pytorch"
    assert bundle_fb.meta["xvs_nav"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_nav"][1]
    assert bundle_fb.nav_session is None
    assert bundle_fb.nav_model is not None

    # 3. Malformed manifest JSON alongside valid ONNX -> fails closed to NB2 .pt
    bad_man_dir = tmp_path / "harvest_bad_manifest"
    bad_man_dir.mkdir()
    shutil.copy2(fp16_onnx, bad_man_dir / XVS_NAV_V5_ONNX_FILENAME)
    (bad_man_dir / "xvs_nav_v5_manifest.json").write_text("{corrupt json", encoding="utf-8")
    bundle_bad_man = load_studio_models(
        device_pref="cpu",
        v5_nav_dir=bad_man_dir,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert bundle_bad_man.meta["xvs_nav"].tier == "nb2_l4_pt"
    assert bundle_bad_man.meta["xvs_nav"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_nav"][1]
    assert bundle_bad_man.nav_session is None


def test_nav_v5_aliasing_gate_cusum_stealth_ramp_and_no_synthetic_lines(
    real_studio_engine: StudioInferenceEngine,
) -> None:
    # 1. Textured KSFO ortho self-match: full 6-stage aliasing gate + calibrated HPL <= 30m + Stage-2 XFeat lines
    ortho = real_studio_engine._load_ksfo_ortho()
    crop = ortho[120:376, 120:376].copy()
    out_nom = real_studio_engine.infer_crossview_nav(crop, ref_bgr=crop, gps_offset_m=3.5, return_jpeg=True)
    json.dumps(out_nom, allow_nan=False)
    assert out_nom["cosine_sim"] > 0.99
    assert out_nom["hal_nav_m"] == pytest.approx(30.0)
    assert out_nom["k_h0"] == pytest.approx(5.678)
    assert out_nom["gamma_conf"] == pytest.approx(1.264)
    assert out_nom["hpl_vis_m"] <= 30.0
    assert out_nom["hpl_within_hal"] is True
    assert out_nom["psr_cal"] >= 1.15
    assert out_nom["aliasing_gate"]["admitted"] is True
    assert out_nom["aliasing_gate"]["integrity_alert"] is False
    assert set(out_nom["aliasing_gate"]["stage_pass"].keys()) == {
        "stage_1_support",
        "stage_2_entropy",
        "stage_3_conditioning",
        "stage_4_psr",
        "stage_5_multicam",
        "stage_6_innovation",
        "integrity_hpl_nav",
    }
    assert out_nom["dual_bank_cusum"]["num_arms"] == 8
    assert out_nom["dual_bank_cusum"]["k_d"] == pytest.approx(0.50)
    assert out_nom["dual_bank_cusum"]["h_threshold"] == pytest.approx(17.619)
    assert out_nom["dual_bank_cusum"]["gnss_quarantined"] is False
    assert out_nom["stage2_xfeat"]["synthetic_lines_used"] is False
    assert out_nom["stage2_xfeat"]["cache_budget_bytes_per_tile"] == 32768
    assert out_nom["stage2_xfeat"]["tier2b_trigger_min_inliers"] == 30
    assert out_nom["stage2_xfeat"]["correspondence_source"] == "stage2_xfeat"
    assert out_nom["stage2_xfeat"]["correspondences_drawn"] > 0
    if (DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME).is_file():
        assert out_nom["stage2_xfeat"]["xfeat_backend"] == "xfeat_onnx"
        assert out_nom["stage2_xfeat"]["xfeat_onnx_sha256"] == EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256
    elif (DEFAULT_V5_NAV_HARVEST_DIR / XVS_NAV_V5_XFEAT_ONNX_FILENAME).is_file():
        assert out_nom["stage2_xfeat"]["xfeat_backend"] == "xfeat_onnx"
        assert out_nom["stage2_xfeat"]["xfeat_onnx_sha256"] == EXPECTED_V5_XFEAT_ONNX_SHA256

    # 2. Blank uniform image: MUST NOT fabricate 5 synthetic lines, MUST reject aliasing gate, strict JSON safe
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    out_blank = real_studio_engine.infer_crossview_nav(blank, ref_bgr=blank, gps_offset_m=0.0, return_jpeg=True)
    json.dumps(out_blank, allow_nan=False)
    assert out_blank["stage2_xfeat"]["synthetic_lines_used"] is False
    assert out_blank["stage2_xfeat"]["correspondences_drawn"] == 0
    assert out_blank["stage2_xfeat"]["correspondence_source"] == "none"
    assert out_blank["aliasing_gate"]["admitted"] is False

    # 3. Multi-epoch 1.0 m/s stealth GNSS ramp with stateful local Dual-Bank CUSUM:
    #    MUST quarantine Bank-A before drift reaches 30.0 m HAL (0 HMI)
    cusum_filter = real_studio_engine.new_nav_cusum_filter()
    quarantine_epoch: int | None = None
    for epoch in range(1, 30):
        step_out = real_studio_engine.infer_crossview_nav(
            crop,
            ref_bgr=crop,
            ref_cache_key="test_ramp_crop",
            gps_offset_m=float(epoch) * 1.0,
            cusum_filter=cusum_filter,
            return_jpeg=False,
        )
        if epoch == 1:
            assert step_out["spoof_detected"] is False
            assert step_out["dual_bank_cusum"]["active_bank"] == "bank_a_gnss"
        if step_out["spoof_detected"]:
            quarantine_epoch = epoch
            assert step_out["nav_mode"] == "VISUAL_PRIMARY_SPOOF_ISOLATED"
            assert step_out["dual_bank_cusum"]["active_bank"] == "bank_b_visual"
            assert step_out["dual_bank_cusum"]["max_cusum_stat"] >= 17.619
            break
    assert quarantine_epoch is not None and quarantine_epoch < 25


def test_nav_v5_fazb_triad_promotion_tamper_fail_closed_and_fallback_chain(tmp_path: Path) -> None:
    from xvs.v2.xvs_nav.stage2_xfeat import LighterGlueFallbackRunner, XFeatOnnxRunner

    fazb_s1 = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_ONNX_FILENAME
    fazb_xf = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME
    fazb_lg = DEFAULT_V5_NAV_FAZB_HARVEST_DIR / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME
    if not (fazb_s1.is_file() and fazb_xf.is_file() and fazb_lg.is_file()):
        pytest.skip("Faz-B FP16 ONNX triad absent locally")

    # 1. Default loader promotes verified Faz-B FP16 triad (S1 + XFeat FP16 + LighterGlue FP16)
    b_fazb = load_studio_models(device_pref="cpu", prefer_onnx=True, smoke_fast=False)
    assert b_fazb.meta["xvs_nav"].tier == "v5_nav_fazb_onnx"
    assert b_fazb.meta["xvs_nav"].backend == "onnxruntime"
    assert b_fazb.meta["xvs_nav"].sha256 == EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256
    assert b_fazb.nav_session is not None
    assert b_fazb.nav_xfeat_onnx_path == fazb_xf
    assert b_fazb.nav_xfeat_sha256 == EXPECTED_V5_FAZB_XFEAT_FP16_ONNX_SHA256
    assert b_fazb.nav_lighterglue_onnx_path == fazb_lg
    assert b_fazb.nav_lighterglue_sha256 == EXPECTED_V5_FAZB_LIGHTERGLUE_FP16_ONNX_SHA256

    # Real ORT inference across all 3 Faz-B triad models
    eng_fazb = StudioInferenceEngine(b_fazb)
    ortho = eng_fazb._load_ksfo_ortho()
    c0 = ortho[120:376, 120:440].copy()
    c1 = ortho[136:392, 136:456].copy()
    nav_res = eng_fazb.infer_crossview_nav(c0, ref_bgr=c0, gps_offset_m=2.0, return_jpeg=False)
    assert nav_res["tier"] == "v5_nav_fazb_onnx"
    assert nav_res["checkpoint_sha256"] == EXPECTED_V5_NAV_FAZB_FP16_ONNX_SHA256
    assert nav_res["cosine_sim"] > 0.999

    xrunner = XFeatOnnxRunner(b_fazb.nav_xfeat_onnx_path, expected_sha256=b_fazb.nav_xfeat_sha256, max_keypoints=256)
    lgrunner = LighterGlueFallbackRunner(
        b_fazb.nav_lighterglue_onnx_path,
        expected_sha256=b_fazb.nav_lighterglue_sha256,
    )
    assert xrunner.backend == "xfeat_onnx"
    assert lgrunner.backend == "lighterglue_onnx"
    k0, d0, _ = xrunner.extract(c0)
    k1, d1, _ = xrunner.extract(c1)
    qi, mi, sc = lgrunner.match(k0, d0, k1, d1, img_hw=(256, 320))
    assert qi.size >= 50
    assert np.all(sc >= 0.20)
    med_shift = np.median(k0[qi] - k1[mi], axis=0)
    assert abs(float(med_shift[0]) - 16.0) <= 2.0
    assert abs(float(med_shift[1]) - 16.0) <= 2.0

    # 2. Tampered Faz-B triad -> fails closed and falls back cleanly to verified Faz-A ONNX
    tampered_fazb = tmp_path / "tampered_fazb"
    tampered_fazb.mkdir()
    (tampered_fazb / XVS_NAV_V5_FAZB_ONNX_FILENAME).write_bytes(b"corrupted-fazb-s1")
    (tampered_fazb / XVS_NAV_V5_FAZB_XFEAT_ONNX_FILENAME).write_bytes(b"corrupted-fazb-xfeat")
    (tampered_fazb / XVS_NAV_V5_FAZB_LIGHTERGLUE_ONNX_FILENAME).write_bytes(b"corrupted-fazb-lg")
    if (DEFAULT_V5_NAV_HARVEST_DIR / XVS_NAV_V5_ONNX_FILENAME).is_file():
        b_faza_fb = load_studio_models(
            device_pref="cpu",
            v5_nav_fazb_dir=tampered_fazb,
            v5_nav_dir=DEFAULT_V5_NAV_HARVEST_DIR,
            prefer_onnx=True,
            smoke_fast=False,
        )
        assert b_faza_fb.meta["xvs_nav"].tier == "v5_nav_onnx"
        assert b_faza_fb.meta["xvs_nav"].sha256 == EXPECTED_V5_NAV_FP16_ONNX_SHA256
        assert b_faza_fb.nav_xfeat_sha256 == EXPECTED_V5_XFEAT_ONNX_SHA256
        assert b_faza_fb.nav_lighterglue_onnx_path is None
        assert b_faza_fb.nav_lighterglue_sha256 == ""

    # 3. Both Faz-B and Faz-A tampered -> fails closed and falls back to verified NB2 .pt
    tampered_faza = tmp_path / "tampered_faza"
    tampered_faza.mkdir()
    (tampered_faza / XVS_NAV_V5_ONNX_FILENAME).write_bytes(b"corrupted-faza-s1")
    (tampered_faza / XVS_NAV_V5_XFEAT_ONNX_FILENAME).write_bytes(b"corrupted-faza-xfeat")
    b_nb2_fb = load_studio_models(
        device_pref="cpu",
        v5_nav_fazb_dir=tampered_fazb,
        v5_nav_dir=tampered_faza,
        prefer_onnx=True,
        smoke_fast=False,
    )
    assert b_nb2_fb.meta["xvs_nav"].tier == "nb2_l4_pt"
    assert b_nb2_fb.meta["xvs_nav"].sha256 == EXPECTED_L4_CKPT_MEMBERS["xvs_nav"][1]
    assert b_nb2_fb.nav_session is None
    assert b_nb2_fb.nav_xfeat_onnx_path is None
    assert b_nb2_fb.nav_lighterglue_onnx_path is None
