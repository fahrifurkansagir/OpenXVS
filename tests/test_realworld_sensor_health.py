"""Comprehensive unit & integration tests for real-world avionics sensor health & calibration."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from xvs.v2.sensor_health import (
    DEFAULT_LWIR_NUC_MAD_THRESHOLD,
    DEFAULT_MAX_CAMERA_IMU_SKEW_MS,
    LwirNucFreezeMonitor,
    PinholeIntrinsics,
    SensorHealthMonitor,
    ThermalSimplexMode,
    TimestampSkewGuard,
    distort_keypoints_pinhole,
    evaluate_frame_optical_quality,
    select_thermal_simplex_mode_fail_closed,
    undistort_frame_pinhole,
    undistort_keypoints_pinhole,
)
from xvs.v2.studio_bridge import StudioInferenceEngine, load_studio_models
from xvs.v2.xvs_thermal.model import (
    ThermalSimplexMode as TorchThermalSimplexMode,
    select_thermal_simplex_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SENSOR_HEALTH_PATH = REPO_ROOT / "src" / "xvs" / "v2" / "sensor_health.py"
STUDIO_MEDIA_DIR = REPO_ROOT / "data" / "studio_media"


def _make_textured_scene(seed: int = 7, w: int = 320, h: int = 240) -> np.ndarray:
    """Create a deterministic, crisp multi-edge BGR scene representing clear daytime imagery."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient + ground texture + crisp geometric structure
    for y in range(h):
        if y < h // 2:
            img[y, :, :] = (195 - y // 3, 175 - y // 4, 150 - y // 4)
        else:
            img[y, :, :] = (65 + (y % 17), 90 + (y % 13), 75 + (y % 11))
    # Runway stripes & crisp structural features
    cv2.rectangle(img, (90, 125), (230, 225), (58, 62, 68), -1)
    for x in range(100, 225, 16):
        cv2.line(img, (x, 128), (x, 222), (225, 230, 235), 2, cv2.LINE_AA)
    for k in range(30):
        px = int(rng.integers(15, w - 15))
        py = int(rng.integers(15, h - 15))
        rad = int(rng.integers(3, 10))
        col = tuple(int(c) for c in rng.integers(30, 220, size=3))
        cv2.circle(img, (px, py), rad, col, 2, cv2.LINE_AA)
    return img


def _make_lwir_scene(seed: int = 11, shift_x: int = 0, w: int = 160, h: int = 128) -> np.ndarray:
    """Create a deterministic 2D uint8 thermal microbolometer frame with warm targets."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 78, dtype=np.uint8)
    noise = rng.integers(-6, 7, size=(h, w))
    frame = np.clip(base.astype(np.int16) + noise, 20, 220).astype(np.uint8)
    cv2.rectangle(frame, (20 + shift_x, 30), (60 + shift_x, 80), 175, -1)
    cv2.circle(frame, (105 + shift_x, 54), 12, 225, -1, cv2.LINE_AA)
    cv2.line(frame, (10, 100), (150, 100), 145, 2)
    return frame


def test_sensor_health_module_is_dependency_light() -> None:
    """Verify `sensor_health.py` imports ONLY Python stdlib + `numpy` + `cv2` (zero torch/scipy)."""
    tree = ast.parse(SENSOR_HEALTH_PATH.read_text(encoding="utf-8"), filename=str(SENSOR_HEALTH_PATH))
    forbidden_roots = {"torch", "scipy", "sklearn", "PIL", "onnxruntime", "pymavlink"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(forbidden_roots), (
        f"Forbidden heavy imports found in sensor_health.py: {imported_roots & forbidden_roots}"
    )
    assert "numpy" in imported_roots and "cv2" in imported_roots


def test_lwir_nuc_freeze_detection_by_ticks_duration_and_recovery() -> None:
    # 1. Tick-count trigger (`MAD < 0.25` for >= 3 consecutive ticks)
    mon = LwirNucFreezeMonitor(mad_threshold=DEFAULT_LWIR_NUC_MAD_THRESHOLD, max_freeze_ticks=3, max_freeze_duration_ms=250.0)
    f_static = _make_lwir_scene(seed=42, shift_x=0)

    s0 = mon.update(f_static, timestamp_ms=1000.0)
    assert s0.frozen is False and s0.freeze_ticks == 0 and s0.q_lwir_gate == 1.0 and s0.reason == "INITIAL_FRAME"

    s1 = mon.update(f_static.copy(), timestamp_ms=1050.0)
    assert s1.frozen is False and s1.freeze_ticks == 1 and s1.mad is not None and s1.mad == pytest.approx(0.0)
    assert s1.reason == "STATIC_WATCH" and s1.q_lwir_gate == 1.0

    # Even tiny sub-threshold sensor drift (MAD ~ 0.10 < 0.25) still counts as frozen shutter
    f_tiny_drift = f_static.astype(np.float32) + 0.10
    s2 = mon.update(f_tiny_drift, timestamp_ms=1100.0)
    assert s2.frozen is False and s2.freeze_ticks == 2 and s2.mad is not None and s2.mad < 0.25

    s3 = mon.update(f_tiny_drift.copy(), timestamp_ms=1150.0)
    assert s3.frozen is True
    assert s3.freeze_ticks == 3
    assert s3.q_lwir_gate == 0.0
    assert s3.reason == "LWIR_NUC_SHUTTER_FROZEN"
    assert mon.is_frozen is True

    # 2. Recovery once live thermal motion resumes (MAD >= 0.25)
    f_live = _make_lwir_scene(seed=43, shift_x=12)
    s4 = mon.update(f_live, timestamp_ms=1200.0)
    assert s4.frozen is False
    assert s4.freeze_ticks == 0
    assert s4.freeze_duration_ms == 0.0
    assert s4.mad is not None and s4.mad >= 0.25
    assert s4.q_lwir_gate == 1.0
    assert s4.reason == "OK"

    # 3. Duration trigger (> 250 ms static even with only 1-2 ticks on slow LWIR stream)
    mon_dur = LwirNucFreezeMonitor(mad_threshold=0.25, max_freeze_ticks=5, max_freeze_duration_ms=250.0)
    mon_dur.update(f_static, timestamp_ms=0.0)
    sd1 = mon_dur.update(f_static.copy(), timestamp_ms=250.0)
    assert sd1.frozen is False and sd1.freeze_duration_ms == pytest.approx(250.0)
    sd2 = mon_dur.update(f_static.copy(), timestamp_ms=260.0)
    assert sd2.frozen is True and sd2.freeze_duration_ms == pytest.approx(260.0) and sd2.q_lwir_gate == 0.0

    # 4. Hysteretic recovery (`recovery_ticks=2`)
    mon_hyst = LwirNucFreezeMonitor(max_freeze_ticks=2, recovery_ticks=2)
    mon_hyst.update(f_static)
    mon_hyst.update(f_static.copy())
    assert mon_hyst.update(f_static.copy()).frozen is True
    r1 = mon_hyst.update(_make_lwir_scene(seed=50, shift_x=6))
    assert r1.frozen is True and r1.q_lwir_gate == 0.0
    r2 = mon_hyst.update(_make_lwir_scene(seed=51, shift_x=14))
    assert r2.frozen is False and r2.q_lwir_gate == 1.0

    # 5. Non-finite / empty frame fails closed immediately
    bad_nan = f_static.astype(np.float32)
    bad_nan[10, 10] = float("nan")
    snan = mon.update(bad_nan, timestamp_ms=1300.0)
    assert snan.frozen is True and snan.q_lwir_gate == 0.0 and snan.reason == "NON_FINITE_OR_EMPTY_LWIR_FRAME"


def test_optical_quality_gate_clear_blur_sun_glare_lens_cap_and_real_media() -> None:
    clean = _make_textured_scene(seed=7)
    q_clean = evaluate_frame_optical_quality(clean, modality="eo")
    assert q_clean.q_effective >= 0.75
    assert q_clean.degraded is False
    assert q_clean.flags == ()
    assert q_clean.over_exposed_ratio < 0.02
    assert q_clean.under_exposed_ratio < 0.10

    # 1. Heavy vibration / motion blur (1D linear motion kernel + defocus)
    ksize = 25
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / float(ksize)
    blurred = cv2.filter2D(clean, -1, kernel)
    blurred = cv2.GaussianBlur(blurred, (21, 21), 6.0)
    q_blur = evaluate_frame_optical_quality(blurred, modality="eo")
    assert q_blur.laplacian_var < q_clean.laplacian_var * 0.05
    assert q_blur.q_effective < 0.50
    assert q_blur.degraded is True
    assert "MOTION_OR_FOCUS_BLUR" in q_blur.flags

    # 2. Sun-glare / over-exposure saturation (`>= 250`)
    glared = clean.copy()
    glared[:140, :, :] = 255  # ~58% saturated whiteout
    q_glare = evaluate_frame_optical_quality(glared, modality="eo")
    assert q_glare.over_exposed_ratio >= 0.50
    assert q_glare.glare_score == 0.0
    assert q_glare.q_effective == 0.0
    assert q_glare.degraded is True
    assert "SUN_GLARE_OVEREXPOSED" in q_glare.flags

    # Partial sun glare (~35% blown out) still degrades below 0.50 even with crisp edges elsewhere
    partial_glare = clean.copy()
    partial_glare[:85, :, :] = 252
    q_partial = evaluate_frame_optical_quality(partial_glare, modality="eo")
    assert 0.30 <= q_partial.over_exposed_ratio <= 0.40
    assert q_partial.q_effective < 0.50
    assert "SUN_GLARE_OVEREXPOSED" in q_partial.flags

    # 3. Lens-cap occlusion / severe under-exposure (`<= 5`)
    lens_cap = np.full((240, 320, 3), 2, dtype=np.uint8)
    q_cap = evaluate_frame_optical_quality(lens_cap, modality="eo")
    assert q_cap.under_exposed_ratio == pytest.approx(1.0)
    assert q_cap.q_effective == 0.0
    assert q_cap.degraded is True
    assert "LENS_CAP_OR_UNDEREXPOSED" in q_cap.flags

    # 4. Flat low-contrast fog haze
    flat_haze = np.full((240, 320, 3), 185, dtype=np.uint8)
    q_haze = evaluate_frame_optical_quality(flat_haze, modality="eo")
    assert q_haze.dynamic_range_p95_p05 == pytest.approx(0.0)
    assert q_haze.q_effective == 0.0
    assert "LOW_CONTRAST_DYNAMIC_RANGE" in q_haze.flags

    # 5. Non-finite / empty frame fails closed
    q_empty = evaluate_frame_optical_quality(np.empty((0, 0), dtype=np.uint8), modality="eo")
    assert q_empty.q_effective == 0.0 and q_empty.degraded is True

    # 6. Ground-truth validation on real `data/studio_media/` frames when present
    man_path = STUDIO_MEDIA_DIR / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        # Real clear sky, runway, and aerial frames MUST score healthy (>= 0.55)
        for ch, key in (("eo_sky", "file"), ("runway_land", "file"), ("crossview_nav", "aerial_file")):
            f0 = man["channels"][ch]["frames"][0]
            im = cv2.imread(str(STUDIO_MEDIA_DIR / f0[key]), cv2.IMREAD_COLOR)
            assert im is not None
            q_real = evaluate_frame_optical_quality(im, modality="eo")
            assert q_real.q_effective >= 0.55, f"Real clear {ch} unexpectedly degraded: {q_real}"
            assert q_real.degraded is False

        # Real radiation_fog EO degrades (< 0.50) while paired LWIR stays healthy (>= 0.80)
        fog_recs = [r for r in man["channels"]["thermal_dve"]["frames"] if r["regime"] == "radiation_fog"]
        assert fog_recs
        eo_fog = cv2.imread(str(STUDIO_MEDIA_DIR / fog_recs[0]["eo_file"]), cv2.IMREAD_COLOR)
        lw_fog = cv2.imread(str(STUDIO_MEDIA_DIR / fog_recs[0]["lwir_file"]), cv2.IMREAD_COLOR)
        assert eo_fog is not None and lw_fog is not None
        q_eo_fog = evaluate_frame_optical_quality(eo_fog, modality="eo")
        q_lw_fog = evaluate_frame_optical_quality(lw_fog, modality="lwir")
        assert q_eo_fog.q_effective < 0.50 and q_eo_fog.degraded is True
        assert q_lw_fog.q_effective >= 0.80 and q_lw_fog.degraded is False


def test_timestamp_skew_guard_monotonicity_and_imu_skew_fail_closed() -> None:
    guard = TimestampSkewGuard(max_skew_ms=DEFAULT_MAX_CAMERA_IMU_SKEW_MS)

    # First valid frame + exact boundary 200.0 ms skew passes
    st0 = guard.check(frame_ts_ms=1000.0, imu_ts_ms=800.0)
    assert st0.valid is True and st0.monotonic_ok is True and st0.skew_ok is True
    assert st0.skew_ms == pytest.approx(200.0) and st0.q_time_gate == 1.0

    # Monotonic step with small 15 ms skew passes
    st1 = guard.check(frame_ts_ms=1100.0, imu_ts_ms=1085.0)
    assert st1.valid is True and st1.dt_ms == pytest.approx(100.0) and st1.skew_ms == pytest.approx(15.0)

    # Duplicate timestamp (dt == 0) rejected fail-closed
    st_dup = guard.check(frame_ts_ms=1100.0, imu_ts_ms=1100.0)
    assert st_dup.valid is False and st_dup.monotonic_ok is False and st_dup.q_time_gate == 0.0
    assert "NON_MONOTONIC_TIMESTAMP" in st_dup.reasons

    # Backward jump (dt < 0) rejected fail-closed AND does NOT rewind reference clock
    st_back = guard.check(frame_ts_ms=1050.0, imu_ts_ms=1050.0)
    assert st_back.valid is False and st_back.monotonic_ok is False and st_back.dt_ms == pytest.approx(-50.0)
    assert guard.last_frame_ts_ms == pytest.approx(1100.0)

    # Next frame at 1090.0 is still <= 1100.0 -> still rejected!
    assert guard.check(frame_ts_ms=1090.0, imu_ts_ms=1090.0).valid is False

    # Excessive positive camera-IMU skew (> 200 ms) rejected fail-closed
    st_skew_pos = guard.check(frame_ts_ms=1200.0, imu_ts_ms=995.0)
    assert st_skew_pos.valid is False and st_skew_pos.monotonic_ok is True and st_skew_pos.skew_ok is False
    assert st_skew_pos.skew_ms == pytest.approx(205.0) and st_skew_pos.q_time_gate == 0.0
    assert "CAMERA_IMU_SKEW_EXCEEDED" in st_skew_pos.reasons

    # Excessive negative camera-IMU skew (future IMU > 200 ms ahead) rejected fail-closed
    st_skew_neg = guard.check(frame_ts_ms=1300.0, imu_ts_ms=1550.0)
    assert st_skew_neg.valid is False and st_skew_neg.skew_ok is False
    assert st_skew_neg.signed_skew_ms == pytest.approx(-250.0) and st_skew_neg.skew_ms == pytest.approx(250.0)

    # Non-finite or negative timestamps rejected fail-closed
    for bad_ts in (float("nan"), float("inf"), float("-inf"), -10.0):
        st_bad = guard.check(frame_ts_ms=bad_ts, imu_ts_ms=1300.0)
        assert st_bad.valid is False and st_bad.q_time_gate == 0.0
    st_bad_imu = guard.check(frame_ts_ms=1400.0, imu_ts_ms=float("nan"))
    assert st_bad_imu.valid is False and st_bad_imu.skew_ok is False and st_bad_imu.q_time_gate == 0.0


def test_brown_conrady_pinhole_undistortion_roundtrip_and_cv2_parity() -> None:
    intr = PinholeIntrinsics(
        fx=680.0,
        fy=682.5,
        cx=640.0,
        cy=360.0,
        k1=-0.22,
        k2=0.06,
        p1=0.0018,
        p2=-0.0014,
        k3=-0.012,
    )
    # Dense 25x25 grid spanning a 1280x720 wide-angle camera frame including corners
    us = np.linspace(20.0, 1260.0, 25)
    vs = np.linspace(20.0, 700.0, 25)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    pts_true = np.column_stack([uu.ravel(), vv.ravel()])

    pts_dist = distort_keypoints_pinhole(pts_true, intr)
    # Confirm significant non-trivial distortion displacement (> 20 px at corners)
    corner_disp = float(np.max(np.linalg.norm(pts_dist - pts_true, axis=1)))
    assert corner_disp > 20.0

    # Round-trip recovery MUST be << 0.05 px (Newton-Raphson achieves < 1e-6 px!)
    pts_rec = undistort_keypoints_pinhole(pts_dist, intr)
    err_px = np.linalg.norm(pts_rec - pts_true, axis=1)
    max_err_px = float(np.max(err_px))
    assert max_err_px < 1.0e-6, f"Round-trip undistortion error {max_err_px:.3e} px exceeded 1e-6 px"
    assert max_err_px < 0.05

    # Normalized Camera RDF rays match `(u - cx)/fx, (v - cy)/fy`
    rays = undistort_keypoints_pinhole(pts_dist, intr, return_normalized=True)
    expected_rays = np.column_stack([(pts_true[:, 0] - intr.cx) / intr.fx, (pts_true[:, 1] - intr.cy) / intr.fy])
    assert float(np.max(np.abs(rays - expected_rays))) < 1.0e-8

    # Cross-check against OpenCV `cv2.undistortPoints`
    cv_out = cv2.undistortPoints(
        pts_dist.reshape(-1, 1, 2),
        intr.camera_matrix,
        intr.dist_coeffs,
        P=intr.camera_matrix,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-12),
    ).reshape(-1, 2)
    cv_diff = float(np.max(np.linalg.norm(pts_rec - cv_out, axis=1)))
    assert cv_diff < 1.0e-4

    # Single point (2,) and empty (0, 2) edge cases + full frame rectification
    pt_single = undistort_keypoints_pinhole(pts_dist[0], intr)
    assert pt_single.shape == (2,) and float(np.linalg.norm(pt_single - pts_true[0])) < 1.0e-6
    assert undistort_keypoints_pinhole(np.zeros((0, 2)), intr).shape == (0, 2)
    rect_img = undistort_frame_pinhole(_make_textured_scene(), intr)
    assert rect_img.shape == (240, 320, 3)

    # Invalid intrinsics / non-finite keypoints fail closed
    with pytest.raises(ValueError):
        PinholeIntrinsics(fx=0.0, fy=500.0, cx=320.0, cy=240.0)
    with pytest.raises(ValueError):
        undistort_keypoints_pinhole(np.array([[100.0, float("nan")]]), intr)


def test_thermal_simplex_mode_coupling_parity_and_fail_closed() -> None:
    # 1:1 parity with `xvs.v2.xvs_thermal.model.select_thermal_simplex_mode`
    cases = [
        (0.90, 0.95, 0.15, ThermalSimplexMode.NOMINAL_FUSION),
        (0.10, 0.95, 0.20, ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY),
        (0.90, 0.00, 0.20, ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY),
        (0.10, 0.00, 0.20, ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK),
        (0.95, 0.95, 0.85, ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK),
        (float("nan"), 0.95, 0.20, ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY),
        (0.95, float("nan"), 0.20, ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY),
        (0.95, 0.95, float("nan"), ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK),
        (-0.1, 1.5, 0.10, ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK),
    ]
    for q_eo, q_lwir, vac, expected in cases:
        m_fast = select_thermal_simplex_mode_fail_closed(q_eo, q_lwir, vac)
        m_torch = select_thermal_simplex_mode(q_eo, q_lwir, vac)
        assert m_fast == expected
        assert int(m_fast) == int(m_torch)
        assert m_fast.name == TorchThermalSimplexMode(int(m_torch)).name

    # SensorHealthMonitor end-to-end coupling:
    #   (a) Healthy EO + Frozen LWIR -> LWIR_DEGRADED_EO_PRIMARY (q_lwir_effective == 0.0)
    #   (b) Lens-capped EO + Frozen LWIR -> DUAL_DVE_SVS_FALLBACK (both == 0.0)
    #   (c) Camera-IMU skew > 200 ms -> DUAL_DVE_SVS_FALLBACK (both gated to 0.0)
    mon = SensorHealthMonitor()
    eo_ok = _make_textured_scene(seed=9)
    lw_static = _make_lwir_scene(seed=15, shift_x=0)
    for idx in range(4):
        rep = mon.assess_thermal_pair(
            eo_ok,
            lw_static.copy(),
            q_eo_nominal=0.95,
            q_lwir_nominal=0.95,
            vacuity=0.15,
            frame_ts_ms=1000.0 + idx * 100.0,
            imu_ts_ms=995.0 + idx * 100.0,
        )
    assert rep["lwir_nuc_frozen"] is True
    assert rep["q_lwir_effective"] == 0.0
    assert rep["q_eo_effective"] >= 0.50
    assert rep["thermal_mode"] == "LWIR_DEGRADED_EO_PRIMARY"

    lens_cap = np.zeros_like(eo_ok)
    rep_dual = mon.assess_thermal_pair(
        lens_cap,
        lw_static.copy(),
        q_eo_nominal=0.95,
        q_lwir_nominal=0.95,
        vacuity=0.15,
        frame_ts_ms=1500.0,
        imu_ts_ms=1495.0,
    )
    assert rep_dual["q_eo_effective"] == 0.0
    assert rep_dual["q_lwir_effective"] == 0.0
    assert rep_dual["thermal_mode"] == "DUAL_DVE_SVS_FALLBACK"

    # Unfreeze LWIR with live frame, then inject >200 ms camera-IMU skew -> fail closed!
    rep_skew = mon.assess_thermal_pair(
        eo_ok,
        _make_lwir_scene(seed=19, shift_x=15),
        q_eo_nominal=0.95,
        q_lwir_nominal=0.95,
        vacuity=0.15,
        frame_ts_ms=2000.0,
        imu_ts_ms=1700.0,
    )
    assert rep_skew["timestamp_valid"] is False
    assert rep_skew["timestamp_skew_ms"] == pytest.approx(300.0)
    assert rep_skew["q_eo_effective"] == 0.0
    assert rep_skew["q_lwir_effective"] == 0.0
    assert rep_skew["thermal_mode"] == "DUAL_DVE_SVS_FALLBACK"


def test_studio_inference_engine_wires_sensor_health_non_breaking_and_enforced() -> None:
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)
    eo_frame = _make_textured_scene(seed=21)
    lw_static = _make_lwir_scene(seed=22, shift_x=0)
    lw_bgr = cv2.cvtColor(lw_static, cv2.COLOR_GRAY2BGR)

    # 1. EO Sky channel emits strict JSON-safe `sensor_health`
    res_eo = engine.infer_eo_sky(
        eo_frame, return_jpeg=False, frame_ts_ms=100.0, imu_ts_ms=95.0
    )
    assert "sensor_health" in res_eo
    assert res_eo["sensor_health"]["q_eo_effective"] >= 0.50
    assert res_eo["sensor_health"]["timestamp_valid"] is True
    json.dumps(res_eo, allow_nan=False)

    # 2. Cross-view Nav channel emits `sensor_health` + supports Brown-Conrady `camera_intrinsics`
    import importlib.util
    if importlib.util.find_spec("xvs.v2.xvs_nav") is not None:
        intr = PinholeIntrinsics(fx=320.0, fy=320.0, cx=160.0, cy=120.0, k1=-0.12, k2=0.02, p1=0.001, p2=-0.001)
        res_nav = engine.infer_crossview_nav(
            eo_frame,
            ref_bgr=eo_frame,
            return_jpeg=False,
            frame_ts_ms=200.0,
            imu_ts_ms=190.0,
            camera_intrinsics=intr,
        )
        assert "sensor_health" in res_nav
        assert res_nav["sensor_health"]["camera_rectified"] is True
        assert res_nav["sensor_health"]["timestamp_valid"] is True
        json.dumps(res_nav, allow_nan=False)

    # 3. Thermal DVE channel with `enforce_sensor_health_gate=True`: frozen LWIR forces fail-closed!
    iso_sh = engine.new_sensor_health_monitor()
    last_th = {}
    for step in range(4):
        last_th = engine.infer_thermal_dve(
            eo_frame,
            lwir_bgr=lw_bgr,
            q_eo=0.95,
            q_lwir=0.95,
            return_jpeg=False,
            frame_ts_ms=1000.0 + step * 100.0,
            imu_ts_ms=995.0 + step * 100.0,
            sensor_health_monitor=iso_sh,
            enforce_sensor_health_gate=True,
        )
    assert last_th["sensor_health"]["lwir_nuc_frozen"] is True
    assert last_th["sensor_health"]["gate_enforced"] is True
    assert last_th["q_lwir"] == 0.0
    assert last_th["w_lwir_eff"] == pytest.approx(0.0, abs=1e-6)
    assert last_th["simplex_mode"] in ("LWIR_DEGRADED_EO_PRIMARY", "DUAL_DVE_SVS_FALLBACK")
    json.dumps(last_th, allow_nan=False)

    # 4. analyze_frame_fast forwards `sensor_health` for d2, thermal (and nav when present)
    fast_channels = ("d2", "thermal", "nav") if importlib.util.find_spec("xvs.v2.xvs_nav") is not None else ("d2", "thermal")
    for ch in fast_channels:
        fast_out = engine.analyze_frame_fast(eo_frame, channel=ch, return_jpeg=False)
        assert "sensor_health" in fast_out
        json.dumps(fast_out, allow_nan=False)


def test_lwir_cold_sky_hot_target_rescue_and_true_dead_frame_fail_closed() -> None:
    """Campaign 1D-2 fix: clean cold LWIR sky with compact hot aircraft MUST NOT be blind-killed,
    while true lens-cap/all-black, all-white, +40% clip, and 60% saturation MUST fail closed."""
    rng = np.random.default_rng(1420)
    # 5 cold-sky + compact hot-aircraft LWIR probes spanning background 3..40 DN and P95-P05 = 3..13 DN
    probes = [
        (3, 1, 210, 4),   # freezing night sky (bg ~ 3 DN) + 4x4 hot jet exhaust
        (8, 2, 185, 5),   # cold high-alt sky (bg ~ 8 DN) + 5x5 warm fuselage
        (16, 3, 165, 6),  # clear winter sky (bg ~ 16 DN) + 6x6 aircraft
        (28, 3, 195, 4),  # twilight sky (bg ~ 28 DN) + 4x4 aircraft
        (40, 4, 220, 5),  # overcast cold sky (bg ~ 40 DN) + 5x5 aircraft
    ]
    for bg_dn, noise_dn, target_dn, box_sz in probes:
        frame = np.clip(
            bg_dn + rng.integers(-noise_dn, noise_dn + 1, size=(128, 160)),
            0,
            255,
        ).astype(np.uint8)
        frame[60 : 60 + box_sz, 78 : 78 + box_sz] = target_dn
        q = evaluate_frame_optical_quality(frame, modality="lwir")
        assert q.q_effective >= 0.85, f"Cold-sky LWIR hot-target probe killed: bg={bg_dn}, q={q}"
        assert q.degraded is False
        assert "LENS_CAP_OR_UNDEREXPOSED" not in q.flags
        assert "LOW_CONTRAST_DYNAMIC_RANGE" not in q.flags

    # Single-pixel hot-pixel impulse on dead black lens-cap MUST still fail closed (2x2 min-filter rejects it)
    dead_with_1px = np.zeros((128, 160), dtype=np.uint8)
    dead_with_1px[64, 80] = 250
    q_1px = evaluate_frame_optical_quality(dead_with_1px, modality="lwir")
    assert q_1px.q_effective <= 0.15 and q_1px.degraded is True

    # True dead frames: all-zero black, all-255 white, +40% top clip, and 60% saturation MUST fail closed
    q_zero = evaluate_frame_optical_quality(np.zeros((128, 160), dtype=np.uint8), modality="lwir")
    assert q_zero.q_effective == 0.0 and q_zero.degraded is True
    q_white = evaluate_frame_optical_quality(np.full((128, 160), 255, dtype=np.uint8), modality="lwir")
    assert q_white.q_effective == 0.0 and q_white.degraded is True

    clipped_40 = _make_lwir_scene(seed=77).copy()
    clipped_40[:55, :] = 255  # >40% saturated white clip
    q_clip40 = evaluate_frame_optical_quality(clipped_40, modality="lwir")
    assert q_clip40.q_effective <= 0.20 and q_clip40.degraded is True
    assert "SUN_GLARE_OVEREXPOSED" in q_clip40.flags

    sat_60 = _make_lwir_scene(seed=78).copy()
    sat_60[:78, :] = 254  # >60% saturation
    q_sat60 = evaluate_frame_optical_quality(sat_60, modality="lwir")
    assert q_sat60.q_effective == 0.0 and q_sat60.degraded is True


def test_dead_pixel_row_col_fpn_and_white_noise_laplacian_bypass_blocked() -> None:
    """Campaign 3 fix: stuck dead-pixel row/col FPN grids and uncorrelated white/binary noise
    MUST NOT bypass SensorHealthMonitor via inflated Laplacian variance (both EO and LWIR)."""
    rng = np.random.default_rng(3030)
    clean_eo = _make_textured_scene(seed=17, w=320, h=240)
    clean_lw = _make_lwir_scene(seed=18, w=160, h=128)

    # 1. Clean frames have zero FPN and high spatial autocorrelation
    q_eo_ok = evaluate_frame_optical_quality(clean_eo, modality="eo")
    q_lw_ok = evaluate_frame_optical_quality(clean_lw, modality="lwir")
    assert q_eo_ok.dead_line_fraction < 0.03 and q_eo_ok.spatial_autocorr > 0.60
    assert q_lw_ok.dead_line_fraction < 0.03 and q_lw_ok.spatial_autocorr > 0.60

    # 2. 5% and 15% dead-pixel stuck row/col stripes (both 0 and 255) on EO & LWIR
    for pct, max_q in ((0.05, 0.45), (0.15, 0.15)):
        eo_fpn = clean_eo.copy()
        h, w = eo_fpn.shape[:2]
        n_rows = max(1, int(round(h * pct)))
        n_cols = max(1, int(round(w * pct)))
        row_ids = np.linspace(4, h - 5, n_rows, dtype=int)
        col_ids = np.linspace(4, w - 5, n_cols, dtype=int)
        for idx_i, r in enumerate(row_ids):
            eo_fpn[r, :, :] = 0 if (idx_i % 2 == 0) else 255
        for idx_i, c in enumerate(col_ids):
            eo_fpn[:, c, :] = 255 if (idx_i % 2 == 0) else 0
        q_ef = evaluate_frame_optical_quality(eo_fpn, modality="eo")
        assert "DEAD_PIXEL_FPN_FAULT" in q_ef.flags
        assert q_ef.dead_line_fraction >= 0.03
        assert q_ef.degraded is True
        assert q_ef.q_effective <= max_q, f"EO {pct*100:.0f}% FPN bypassed gate: {q_ef}"

        lw_fpn = clean_lw.copy()
        lh, lw_w = lw_fpn.shape[:2]
        for idx_i, r in enumerate(np.linspace(4, lh - 5, max(1, int(round(lh * pct))), dtype=int)):
            lw_fpn[r, :] = 0 if (idx_i % 2 == 0) else 255
        for idx_i, c in enumerate(np.linspace(4, lw_w - 5, max(1, int(round(lw_w * pct))), dtype=int)):
            lw_fpn[:, c] = 255 if (idx_i % 2 == 0) else 0
        q_lf = evaluate_frame_optical_quality(lw_fpn, modality="lwir")
        assert "DEAD_PIXEL_FPN_FAULT" in q_lf.flags
        assert q_lf.degraded is True
        assert q_lf.q_effective <= max_q, f"LWIR {pct*100:.0f}% FPN bypassed gate: {q_lf}"

    # 3. Uniform white noise & binary 0/255 impulse noise MUST fail closed (q <= 0.10) on EO & LWIR
    uniform_noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    binary_noise = (rng.integers(0, 2, size=(240, 320), dtype=np.uint8) * 255).astype(np.uint8)
    for mod in ("eo", "lwir"):
        for noise_img in (uniform_noise, binary_noise):
            q_n = evaluate_frame_optical_quality(noise_img, modality=mod)
            assert "SENSOR_NOISE_FAULT" in q_n.flags
            assert q_n.spatial_autocorr < 0.25
            assert q_n.degraded is True
            assert q_n.q_effective <= 0.10, f"White noise bypassed {mod} gate: {q_n}"
