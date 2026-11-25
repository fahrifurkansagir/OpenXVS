"""Comprehensive unit, fail-closed, cross-modal, and real-media benchmark tests for
`xvs.v2.xvs_d2.track_before_confirm` (OpenXVS v5.2 M-of-N Track-Before-Confirm).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from xvs.v2.sensor_health import LwirNucFreezeMonitor, evaluate_frame_optical_quality
from xvs.v2.xvs_d2 import (
    DEFAULT_CONFIRM_M,
    DEFAULT_CONFIRM_N,
    DEFAULT_MAX_COAST_FRAMES,
    DEFAULT_SINGLE_FRAME_SCORE_THR,
    DEFAULT_WEAK_SCORE_THR,
    BearingDetection,
    BearingTrackBeforeConfirm,
    CrossModalSimplexMode,
    TrackBeforeConfirmConfig,
    TrackStatus,
    bearing_to_pixel_uv,
    benchmark_tbc_vs_single_frame,
    compute_12_sector_optical_quality,
    detections_from_d2_peaks,
    pixel_to_bearing_rad,
    sanitize_health_weight,
    select_cross_modal_simplex_mode,
    simulate_encounter_884_low_snr_sequence,
    wrap_pi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TBC_MODULE_PATH = REPO_ROOT / "src" / "xvs" / "v2" / "xvs_d2" / "track_before_confirm.py"
STUDIO_MEDIA_DIR = REPO_ROOT / "data" / "studio_media"
D2_ONNX_PATH = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"


def test_tbc_module_is_dependency_light_and_properly_exported() -> None:
    """Verify `track_before_confirm.py` imports ONLY stdlib + `numpy` (zero torch/scipy/cv2/GPL)."""
    tree = ast.parse(TBC_MODULE_PATH.read_text(encoding="utf-8"), filename=str(TBC_MODULE_PATH))
    forbidden_roots = {"torch", "scipy", "cv2", "sklearn", "PIL", "onnxruntime", "pymavlink"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(forbidden_roots), (
        f"Forbidden imports in track_before_confirm.py: {imported_roots & forbidden_roots}"
    )
    assert "numpy" in imported_roots
    assert DEFAULT_CONFIRM_M == 3
    assert DEFAULT_CONFIRM_N == 5
    assert DEFAULT_MAX_COAST_FRAMES == 5
    assert DEFAULT_WEAK_SCORE_THR == pytest.approx(0.15)
    assert DEFAULT_SINGLE_FRAME_SCORE_THR == pytest.approx(0.45)


def test_bearing_projection_camera_rdf_ned_roundtrip_and_wrap_pi() -> None:
    w, h, f = 512.0, 384.0, 2690.0
    # Center pixel maps strictly to (0, 0)
    az0, el0 = pixel_to_bearing_rad(256.0, 192.0, w, h, f)
    assert az0 == pytest.approx(0.0, abs=1e-12)
    assert el0 == pytest.approx(0.0, abs=1e-12)

    # Right of center -> +az (NED East), above center -> +el (-Down)
    az_ru, el_ru = pixel_to_bearing_rad(356.0, 92.0, w, h, f)
    assert az_ru > 0.0 and el_ru > 0.0
    u_rt, v_rt = bearing_to_pixel_uv(az_ru, el_ru, w, h, f)
    assert u_rt == pytest.approx(356.0, abs=1e-11)
    assert v_rt == pytest.approx(92.0, abs=1e-11)

    # Wrap-pi seam check
    assert wrap_pi(math.pi + 0.05) == pytest.approx(-math.pi + 0.05, abs=1e-12)
    assert wrap_pi(-math.pi - 0.05) == pytest.approx(math.pi - 0.05, abs=1e-12)
    assert wrap_pi(float("nan")) == 0.0

    # Adapter from decode_peaks_local_softargmax dicts
    dets = detections_from_d2_peaks(
        [{"u_px": 100.0, "v_px": 80.0, "score": 0.31, "var_u": 4.0, "var_v": 9.0, "s_mom_px": 12.0}],
        width_px=w,
        height_px=h,
        focal_px=f,
        offset_u_px=128.0,
        offset_v_px=64.0,
    )
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(0.31)
    assert dets[0].sigma_az_rad == pytest.approx(2.0 / f)
    assert dets[0].sigma_el_rad == pytest.approx(3.0 / f)
    assert dets[0].size_px == pytest.approx(12.0)


def test_m_of_n_3_of_5_confirmation_5_frame_coast_delete_and_determinism() -> None:
    cfg = TrackBeforeConfirmConfig(confirm_m=3, confirm_n=5, max_coast_frames=5, weak_score_thr=0.15)

    # 1. Exact 3-of-5 intermittent weak hits: [hit, miss, hit, miss, hit] -> confirms on frame 5!
    trk = BearingTrackBeforeConfirm(cfg)
    az0, el0 = -0.05, 0.02
    vaz, vel = 0.015, -0.005
    pattern = [True, False, True, False, True]
    for k, has_hit in enumerate(pattern):
        t = k * 0.1
        dets = (
            [BearingDetection(az_rad=az0 + vaz * t, el_rad=el0 + vel * t, score=0.21, modality="eo")]
            if has_hit
            else []
        )
        res = trk.step(dets, dt_s=0.1, q_eo=1.0, q_lwir=1.0)
        if k < 4:
            assert len(res.confirmed_tracks) == 0
            assert len(res.tentative_tracks) == 1
        else:
            assert res.tracks_confirmed_this_step == 1
            assert len(res.confirmed_tracks) == 1
            c = res.confirmed_tracks[0]
            assert c.status == TrackStatus.CONFIRMED
            assert c.window_hits == 3 and c.window_len == 5
            assert c.az_dot_rad_s == pytest.approx(vaz, abs=0.008)
            assert c.el_dot_rad_s == pytest.approx(vel, abs=0.008)

    # 2. Confirmed track coasts smoothly for 4 misses and deletes cleanly on the 5th consecutive miss
    for miss_idx in range(1, 6):
        res_coast = trk.step([], dt_s=0.1, q_eo=1.0, q_lwir=1.0)
        if miss_idx < 5:
            assert len(res_coast.confirmed_tracks) == 1
            assert res_coast.confirmed_tracks[0].coast_frames == miss_idx
            assert res_coast.tracks_deleted_this_step == 0
        else:
            assert len(res_coast.confirmed_tracks) == 0
            assert len(res_coast.all_tracks) == 0
            assert res_coast.tracks_deleted_this_step == 1

    # 3. 2-of-5 weak hits must NEVER confirm
    trk_2of5 = BearingTrackBeforeConfirm(cfg)
    for k, has_hit in enumerate([True, False, False, True, False, False, False]):
        dets = [BearingDetection(az_rad=0.01 * k * 0.1, el_rad=0.0, score=0.25)] if has_hit else []
        res = trk_2of5.step(dets, dt_s=0.1)
        assert len(res.confirmed_tracks) == 0

    # 4. Azimuth ±pi wrap continuity across seam
    trk_seam = BearingTrackBeforeConfirm(cfg)
    for k in range(5):
        az = wrap_pi(math.pi - 0.003 + 0.002 * k)
        res_s = trk_seam.step([BearingDetection(az_rad=az, el_rad=0.01, score=0.24)], dt_s=0.1)
    assert len(res_s.confirmed_tracks) == 1
    assert res_s.confirmed_tracks[0].track_id == 1
    assert res_s.confirmed_tracks[0].az_dot_rad_s == pytest.approx(0.02, abs=0.008)


def test_clutter_stress_zero_false_confirmed_tracks() -> None:
    """120 frames x 6 random sky clutter false alarms/frame (720 total clutter hits) -> 0 false confirmed tracks."""
    rng = np.random.default_rng(2026052)
    tracker = BearingTrackBeforeConfirm()
    total_clutter = 0
    total_confirmed_false = 0

    for _ in range(120):
        dets = [
            BearingDetection(
                az_rad=float(rng.uniform(-0.45, 0.45)),
                el_rad=float(rng.uniform(-0.18, 0.28)),
                score=float(rng.uniform(0.16, 0.44)),
                modality="eo",
            )
            for _ in range(6)
        ]
        total_clutter += len(dets)
        res = tracker.step(dets, dt_s=0.1, q_eo=1.0, q_lwir=0.0)
        total_confirmed_false += len(res.confirmed_tracks)

    assert total_clutter == 720
    assert total_confirmed_false == 0, f"Expected 0 false confirmed tracks, got {total_confirmed_false}"


def test_eo_lwir_cross_modal_fusion_health_weights_and_seamless_dve_handoff() -> None:
    """Verify EO+LWIR fusion -> radiation fog EO blackout -> LWIR NUC freeze -> dual-blind fail-closed."""
    tracker = BearingTrackBeforeConfirm()
    az0, el0 = 0.03, 0.04
    vaz, vel = -0.018, 0.004

    confirmed_tid: int | None = None

    # Phase 1 (frames 0..5): Clear sky NOMINAL_FUSION — both EO and LWIR detect same target
    for k in range(6):
        t = k * 0.1
        gt_az, gt_el = az0 + vaz * t, el0 + vel * t
        dets = [
            BearingDetection(az_rad=gt_az + 0.0003, el_rad=gt_el - 0.0002, score=0.28, modality="eo"),
            BearingDetection(az_rad=gt_az - 0.0009, el_rad=gt_el + 0.0007, score=0.32, modality="lwir"),
        ]
        res = tracker.step(dets, dt_s=0.1, q_eo=0.95, q_lwir=0.92)
        assert res.simplex_mode == CrossModalSimplexMode.NOMINAL_FUSION
        assert res.dual_blind is False
        assert len(res.all_tracks) == 1, "Co-located EO + LWIR must fuse into 1 track, never split!"
        if k >= 2:
            assert len(res.confirmed_tracks) == 1
            trk = res.confirmed_tracks[0]
            confirmed_tid = trk.track_id
            assert trk.cross_modal_fused is True
            assert trk.last_modalities == ("eo", "lwir")
            # Noisy-OR fused score exceeds each individual modality effective score
            assert trk.score > max(0.28 * 0.95, 0.32 * 0.92)
            assert trk.sigma_az_rad < 0.0012

    assert confirmed_tid is not None

    # Phase 2 (frames 6..11): Dense radiation fog (q_eo = 0.02, q_lwir = 0.98)
    # Corrupted EO glare false alarm MUST be blocked by q_eo; LWIR keeps exact same track_id alive!
    for k in range(6, 12):
        t = k * 0.1
        gt_az, gt_el = az0 + vaz * t, el0 + vel * t
        dets = [
            BearingDetection(az_rad=gt_az + 0.012, el_rad=gt_el - 0.012, score=0.99, modality="eo"),
            BearingDetection(az_rad=gt_az + 0.0005, el_rad=gt_el - 0.0005, score=0.34, modality="lwir"),
        ]
        res = tracker.step(dets, dt_s=0.1, q_eo=0.02, q_lwir=0.98)
        assert res.simplex_mode == CrossModalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
        assert res.detections_rejected_health == 1
        assert len(res.confirmed_tracks) == 1
        assert res.confirmed_tracks[0].track_id == confirmed_tid
        assert res.confirmed_tracks[0].last_modalities == ("lwir",)

    # Phase 3 (frames 12..17): LWIR NUC shutter freeze (q_eo = 0.92, q_lwir = 0.0)
    # Stuck LWIR detection MUST be strictly rejected; EO seamlessly retains exact same track_id!
    for k in range(12, 18):
        t = k * 0.1
        gt_az, gt_el = az0 + vaz * t, el0 + vel * t
        dets = [
            BearingDetection(az_rad=gt_az, el_rad=gt_el, score=0.30, modality="eo"),
            BearingDetection(az_rad=-0.20, el_rad=0.10, score=0.99, modality="lwir"),
        ]
        res = tracker.step(dets, dt_s=0.1, q_eo=0.92, q_lwir=0.0)
        assert res.simplex_mode == CrossModalSimplexMode.LWIR_DEGRADED_EO_PRIMARY
        assert res.detections_rejected_health == 1
        assert len(res.confirmed_tracks) == 1
        assert res.confirmed_tracks[0].track_id == confirmed_tid
        assert res.confirmed_tracks[0].last_modalities == ("eo",)

    # Phase 4 (frames 18..22): Dual-blind (q_eo = 0.0, q_lwir = 0.0) -> zero hits accepted, deletes on tick 5
    for m in range(5):
        res_db = tracker.step(
            [
                BearingDetection(az_rad=0.0, el_rad=0.0, score=0.99, modality="eo"),
                BearingDetection(az_rad=0.0, el_rad=0.0, score=0.99, modality="lwir"),
            ],
            dt_s=0.1,
            q_eo=0.0,
            q_lwir=0.0,
        )
        assert res_db.dual_blind is True
        assert res_db.simplex_mode == CrossModalSimplexMode.DUAL_DVE_SVS_FALLBACK
        assert res_db.detections_accepted == 0
        if m < 4:
            assert len(res_db.confirmed_tracks) == 1
        else:
            assert len(res_db.confirmed_tracks) == 0
            assert res_db.tracks_deleted_this_step == 1


def test_strict_fail_closed_on_nan_inf_and_invalid_inputs() -> None:
    for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 1.01):
        assert sanitize_health_weight(bad) == 0.0
    assert (
        select_cross_modal_simplex_mode(float("nan"), float("inf"))
        == CrossModalSimplexMode.DUAL_DVE_SVS_FALLBACK
    )

    tracker = BearingTrackBeforeConfirm()
    # NaN/Inf health weights must drop ALL detections fail-closed
    res_bad_q = tracker.step(
        [
            BearingDetection(az_rad=0.0, el_rad=0.0, score=0.95, modality="eo"),
            BearingDetection(az_rad=0.0, el_rad=0.0, score=0.95, modality="lwir"),
        ],
        dt_s=0.1,
        q_eo=float("nan"),
        q_lwir=float("inf"),
    )
    assert res_bad_q.dual_blind is True
    assert res_bad_q.q_eo == 0.0 and res_bad_q.q_lwir == 0.0
    assert res_bad_q.detections_accepted == 0
    assert len(res_bad_q.all_tracks) == 0

    # Non-finite detections and invalid modalities are rejected without crashing
    res_bad_dets = tracker.step(
        [
            BearingDetection(az_rad=float("nan"), el_rad=0.0, score=0.9, modality="eo"),
            BearingDetection(az_rad=0.0, el_rad=float("inf"), score=0.9, modality="eo"),
            BearingDetection(az_rad=0.0, el_rad=0.0, score=float("nan"), modality="eo"),
            BearingDetection(az_rad=0.0, el_rad=0.0, score=1.5, modality="eo"),
            BearingDetection(az_rad=0.0, el_rad=0.0, score=0.5, modality="radar"),
            BearingDetection(az_rad=0.0, el_rad=0.0, score=0.5, modality="eo", sigma_az_rad=-1.0),
            BearingDetection(az_rad=0.02, el_rad=0.01, score=0.35, modality="eo"),
        ],
        dt_s=0.1,
        q_eo=1.0,
        q_lwir=1.0,
    )
    assert res_bad_dets.detections_rejected_invalid == 6
    assert res_bad_dets.detections_accepted == 1
    assert len(res_bad_dets.tentative_tracks) == 1

    # Invalid dt_s fails closed without poisoning existing track state
    for bad_dt in (0.0, -0.1, float("nan"), float("inf")):
        res_dt = tracker.step(
            [BearingDetection(az_rad=0.02, el_rad=0.01, score=0.8, modality="eo")],
            dt_s=bad_dt,
            q_eo=1.0,
            q_lwir=1.0,
        )
        assert res_dt.dt_valid is False
        assert res_dt.detections_accepted == 0
        assert all(math.isfinite(t.az_rad) and math.isfinite(t.el_rad) for t in res_dt.all_tracks)


def test_benchmark_encounter_884_low_snr_continuity_lift_and_zero_false_tracks() -> None:
    """BEFORE (single-frame >= 0.45) vs AFTER (M-of-N 3-of-5 TBC >= 0.15) on Encounter 884.0-class
    low-SNR small fixed-wing trajectory with dense random sky clutter."""
    gt_bearings, frame_dets = simulate_encounter_884_low_snr_sequence(
        num_frames=60, dt_s=0.1, clutter_per_frame=4, seed=884
    )
    summary = benchmark_tbc_vs_single_frame(
        gt_bearings,
        frame_dets,
        scenario_name="encounter_884_class_low_snr",
        dt_s=0.1,
    )
    print(f"\nBENCH_ENC884: {json.dumps(summary.to_dict())}")

    # Single-frame threshold mirrors Encounter 884.0's ~12% low-SNR dropout + clutter false alarms
    assert summary.single_frame_continuity <= 0.20
    assert summary.single_frame_false_alarms > 0

    # M-of-N TBC delivers dramatic continuity lift (>= +75 pp) with ZERO false confirmed tracks!
    assert summary.tbc_continuity >= 0.90
    assert summary.tbc_continuity_lift >= 0.75
    assert summary.tbc_false_confirmed_tracks == 0
    assert summary.tbc_false_confirmed_frames == 0
    assert summary.tbc_confirm_latency_frames is not None and summary.tbc_confirm_latency_frames <= 4
    assert summary.tbc_bearing_rmse_mrad is not None and summary.tbc_bearing_rmse_mrad <= 1.6


def _extract_d2_peaks_from_onnx(
    sess: ort.InferenceSession,
    in_dtypes: dict[str, np.dtype[Any]],
    enc_frames: list[dict[str, Any]],
    *,
    weak_thr: float = 0.15,
    max_peaks: int = 12,
) -> tuple[list[tuple[float, float]], list[list[BearingDetection]]]:
    """Run real `xvs_d2_web_fp16.onnx` ConvGRU over a real `studio_media/eo_sky` encounter and
    convert all local-maxima peaks (`>= weak_thr`) into full-frame Camera RDF / NED bearings."""
    h_prev = np.zeros((1, 24, 64, 64), dtype=np.float32)
    hist: list[tuple[np.ndarray, int, int]] = []
    gt_bearings: list[tuple[float, float]] = []
    frame_dets: list[list[BearingDetection]] = []

    full_w, full_h = 2448.0, 2048.0

    def _warp(g: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
        mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        w = cv2.warpAffine(
            g, mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        mk = (
            cv2.warpAffine(
                np.ones((256, 256), dtype=np.float32),
                mat,
                (256, 256),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            > 0.5
        ).astype(np.float32)
        norm_w = np.clip((w - float(w.mean())) / max(float(w.std()), 8.0), -5.0, 5.0).astype(np.float32)
        return norm_w * mk, mk

    for f in enc_frames:
        bgr = cv2.imread(str(STUDIO_MEDIA_DIR / f["file"]), cv2.IMREAD_COLOR)
        assert bgr is not None
        ox, oy = int(f["d2_subcrop_origin_px"][0]), int(f["d2_subcrop_origin_px"][1])
        cx0, cy0 = int(f["crop_origin_full_px"][0]), int(f["crop_origin_full_px"][1])
        focal_px = float(f["focal_px"])

        gt = f["gt_box_px"]
        gt_u_full = cx0 + 0.5 * (float(gt[0]) + float(gt[2]))
        gt_v_full = cy0 + 0.5 * (float(gt[1]) + float(gt[3]))
        gt_bearings.append(pixel_to_bearing_rad(gt_u_full, gt_v_full, full_w, full_h, focal_px))

        crop = cv2.cvtColor(bgr[oy : oy + 256, ox : ox + 256], cv2.COLOR_BGR2GRAY).astype(np.float32)
        c0 = np.clip((crop - float(crop.mean())) / max(float(crop.std()), 8.0), -5.0, 5.0).astype(np.float32)
        gmc_x, gmc_y = cx0 + ox, cy0 + oy

        if len(hist) >= 1:
            g1, x1, y1 = hist[-1]
            dx1, dy1 = float(x1 - gmc_x), float(y1 - gmc_y)
            c1, r1 = _warp(g1, dx1, dy1)
        else:
            dx1, dy1 = 0.0, 0.0
            c1 = np.zeros((256, 256), dtype=np.float32)
            r1 = np.zeros((256, 256), dtype=np.float32)
        if len(hist) >= 5:
            g5, x5, y5 = hist[-5]
            c2, r5 = _warp(g5, float(x5 - gmc_x), float(y5 - gmc_y))
        else:
            c2 = np.zeros((256, 256), dtype=np.float32)
            r5 = np.zeros((256, 256), dtype=np.float32)

        x_t = np.stack([c0, c1, c2, (c0 - c1) * r1, (c0 - c2) * r5, r1 * r5], axis=0)[np.newaxis]
        if abs(dx1) > 1e-4 or abs(dy1) > 1e-4:
            mh = np.array([[1.0, 0.0, dx1 / 4.0], [0.0, 1.0, dy1 / 4.0]], dtype=np.float32)
            h_warped = np.empty_like(h_prev)
            for ch in range(24):
                h_warped[0, ch] = cv2.warpAffine(
                    h_prev[0, ch], mh, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
        else:
            h_warped = h_prev

        hist.append((crop.copy(), gmc_x, gmc_y))
        if len(hist) > 8:
            hist.pop(0)

        outs = sess.run(
            None,
            {
                "x_t": x_t.astype(in_dtypes["x_t"]),
                "h_prev_warped": h_warped.astype(in_dtypes["h_prev_warped"]),
            },
        )
        hm = outs[0][0, 0].astype(np.float32)
        off = outs[1][0].astype(np.float32)
        pmk = outs[4][0, 0].astype(np.float32)
        h_prev = outs[5].astype(np.float32)

        probs = 1.0 / (1.0 + np.exp(-np.clip(hm, -40.0, 40.0)))
        ys, xs = np.where((probs >= weak_thr) & (pmk > 0.5))
        scores = probs[ys, xs]
        order = np.argsort(-scores)[:max_peaks]

        dets: list[BearingDetection] = []
        for idx in order:
            cy_i, cx_i = int(ys[idx]), int(xs[idx])
            sc = float(probs[cy_i, cx_i])
            u_crop = (float(cx_i) + float(off[0, cy_i, cx_i]) + 0.5) * 2.0
            v_crop = (float(cy_i) + float(off[1, cy_i, cx_i]) + 0.5) * 2.0
            u_full = gmc_x + u_crop
            v_full = gmc_y + v_crop
            az, el = pixel_to_bearing_rad(u_full, v_full, full_w, full_h, focal_px)
            dets.append(
                BearingDetection(
                    az_rad=az,
                    el_rad=el,
                    score=sc,
                    modality="eo",
                    u_px=u_full,
                    v_px=v_full,
                )
            )
        frame_dets.append(dets)

    return gt_bearings, frame_dets


def test_benchmark_real_studio_media_eo_sky_and_thermal_dve() -> None:
    """Evaluate M-of-N TBC + EO/LWIR cross-modal fusion on real `data/studio_media/` frames."""
    manifest_path = STUDIO_MEDIA_DIR / "manifest.json"
    assert manifest_path.is_file(), f"Missing real studio_media manifest: {manifest_path}"
    assert D2_ONNX_PATH.is_file(), f"Missing XVS-D2 ONNX: {D2_ONNX_PATH}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Real `data/studio_media/eo_sky/` (60 real Amazon AOT frames across 3 encounters)
    sess = ort.InferenceSession(str(D2_ONNX_PATH), providers=["CPUExecutionProvider"])
    in_dtypes = {
        inp.name: (np.float16 if "float16" in inp.type else np.float32)
        for inp in sess.get_inputs()
    }
    eo_frames = manifest["channels"]["eo_sky"]["frames"]
    total_sf_fa = 0
    total_tbc_fa_frames = 0
    total_tbc_conf = 0
    total_frames = 0

    # Real `eo_sky` encounters are close-range helicopters (46-308 m CPA, up to 0.71 rad/s bearing rate)
    eo_sky_cpa_cfg = TrackBeforeConfirmConfig(
        confirm_m=3,
        confirm_n=5,
        max_coast_frames=5,
        weak_score_thr=0.15,
        single_frame_score_thr=0.45,
        init_vel_std_rad_s=0.35,
        process_accel_std_rad_s2=1.2,
        tentative_assoc_rad=0.03,
        max_assoc_rad=0.08,
        nms_radius_rad=0.015,
    )

    total_low_snr_sf = 0
    total_low_snr_tbc = 0
    for enc_id in sorted(set(f["encounter_id"] for f in eo_frames)):
        sub = [f for f in eo_frames if f["encounter_id"] == enc_id]
        gt_b, dets_b = _extract_d2_peaks_from_onnx(sess, in_dtypes, sub)
        q_eo_seq = [
            evaluate_frame_optical_quality(
                cv2.imread(str(STUDIO_MEDIA_DIR / f["file"]), cv2.IMREAD_COLOR), modality="eo"
            ).q_effective
            for f in sub
        ]
        res = benchmark_tbc_vs_single_frame(
            gt_b,
            dets_b,
            scenario_name=f"real_studio_eo_sky_enc_{enc_id}",
            dt_s=0.1,
            q_eo_seq=q_eo_seq,
            config=eo_sky_cpa_cfg,
            match_tol_rad=0.020,
        )
        print(f"BENCH_REAL_EO_SKY[{enc_id}]: {json.dumps(res.to_dict())}")
        assert res.tbc_continuity >= 0.85, f"Low TBC continuity on real enc {enc_id}: {res}"
        total_sf_fa += res.single_frame_false_alarms
        total_tbc_fa_frames += res.tbc_false_confirmed_frames
        total_tbc_conf += res.tbc_confirmed_frames
        total_frames += res.num_frames

        # Also evaluate low-SNR haze-attenuated scores (0.45x on real ONNX detections)
        low_snr_dets = [
            [
                BearingDetection(
                    az_rad=d.az_rad,
                    el_rad=d.el_rad,
                    score=d.score * 0.45,
                    modality=d.modality,
                    sigma_az_rad=d.sigma_az_rad,
                    sigma_el_rad=d.sigma_el_rad,
                    size_px=d.size_px,
                    u_px=d.u_px,
                    v_px=d.v_px,
                )
                for d in frame_d
            ]
            for frame_d in dets_b
        ]
        res_low = benchmark_tbc_vs_single_frame(
            gt_b,
            low_snr_dets,
            scenario_name=f"real_studio_eo_sky_low_snr_{enc_id}",
            dt_s=0.1,
            q_eo_seq=q_eo_seq,
            config=eo_sky_cpa_cfg,
            match_tol_rad=0.020,
        )
        print(f"BENCH_REAL_EO_SKY_LOW_SNR[{enc_id}]: {json.dumps(res_low.to_dict())}")
        total_low_snr_sf += res_low.single_frame_det_frames
        total_low_snr_tbc += res_low.tbc_confirmed_frames

    # Across all 60 real Amazon AOT frames, TBC sustains >= 90% continuity, slashes clutter false alarms,
    # and lifts low-SNR continuity dramatically over single-frame 0.45 thresholding!
    assert total_frames == 60
    assert total_tbc_conf / float(total_frames) >= 0.88
    assert total_tbc_fa_frames < total_sf_fa
    assert total_low_snr_tbc > total_low_snr_sf + 30

    # 2. Real `data/studio_media/thermal_dve/` (24 real paired EO + LWIR frames:
    #    clear -> radiation_fog -> stratus_cloud) with real optical quality + NUC freeze monitor!
    th_frames = manifest["channels"]["thermal_dve"]["frames"]
    assert len(th_frames) == 24
    nuc_mon = LwirNucFreezeMonitor()
    tracker = BearingTrackBeforeConfirm()
    rng = np.random.default_rng(77)

    focal_dve = 640.0
    clear_fog_ids: set[int] = set()
    regimes_confirmed: dict[str, int] = {}
    simplex_modes_seen: set[CrossModalSimplexMode] = set()
    false_confirmed_dve = 0

    for idx, f in enumerate(th_frames):
        eo_img = cv2.imread(str(STUDIO_MEDIA_DIR / f["eo_file"]), cv2.IMREAD_COLOR)
        lw_img = cv2.imread(str(STUDIO_MEDIA_DIR / f["lwir_file"]), cv2.IMREAD_COLOR)
        assert eo_img is not None and lw_img is not None
        h_i, w_i = eo_img.shape[:2]

        q_eo = evaluate_frame_optical_quality(eo_img, modality="eo").q_effective
        q_lw_opt = evaluate_frame_optical_quality(lw_img, modality="lwir").q_effective
        nuc_st = nuc_mon.update(lw_img, timestamp_ms=idx * 100.0)
        q_lwir = q_lw_opt * nuc_st.q_lwir_gate

        box = f["aircraft_box_px"]
        u_gt = 0.5 * (float(box[0]) + float(box[2]))
        v_gt = 0.5 * (float(box[1]) + float(box[3]))
        gt_az, gt_el = pixel_to_bearing_rad(u_gt, v_gt, float(w_i), float(h_i), focal_dve)

        # Measure real local contrast around the aircraft box in EO and LWIR pixels
        x1, y1, x2, y2 = [int(v) for v in box]
        eo_gray = cv2.cvtColor(eo_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lw_gray = cv2.cvtColor(lw_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        eo_roi_std = float(eo_gray[max(0, y1 - 8) : min(h_i, y2 + 8), max(0, x1 - 8) : min(w_i, x2 + 8)].std())
        lw_roi_std = float(lw_gray[max(0, y1 - 8) : min(h_i, y2 + 8), max(0, x1 - 8) : min(w_i, x2 + 8)].std())
        eo_score = min(0.85, max(0.05, eo_roi_std / 35.0))
        lw_score = min(0.85, max(0.20, lw_roi_std / 35.0))

        dets: list[BearingDetection] = [
            BearingDetection(
                az_rad=gt_az + float(rng.normal(0.0, 0.001)),
                el_rad=gt_el + float(rng.normal(0.0, 0.001)),
                score=eo_score,
                modality="eo",
            ),
            BearingDetection(
                az_rad=gt_az + float(rng.normal(0.0, 0.002)),
                el_rad=gt_el + float(rng.normal(0.0, 0.002)),
                score=lw_score,
                modality="lwir",
            ),
            # Random EO fog-glare clutter blip per frame
            BearingDetection(
                az_rad=float(rng.uniform(-0.4, 0.4)),
                el_rad=float(rng.uniform(-0.2, 0.2)),
                score=float(rng.uniform(0.18, 0.42)),
                modality="eo",
            ),
        ]

        step_res = tracker.step(dets, dt_s=0.1, q_eo=q_eo, q_lwir=q_lwir)
        simplex_modes_seen.add(step_res.simplex_mode)
        matched_this_frame = False
        for trk in step_res.confirmed_tracks:
            err = math.hypot(wrap_pi(trk.az_rad - gt_az), trk.el_rad - gt_el)
            if err <= 0.025:
                matched_this_frame = True
                if f["regime"] in ("clear", "radiation_fog"):
                    clear_fog_ids.add(trk.track_id)
            else:
                false_confirmed_dve += 1
        if matched_this_frame:
            regimes_confirmed[f["regime"]] = regimes_confirmed.get(f["regime"], 0) + 1

    print(
        f"BENCH_REAL_THERMAL_DVE: clear_fog_ids={sorted(clear_fog_ids)}, "
        f"regimes_confirmed={regimes_confirmed}, modes={[m.name for m in simplex_modes_seen]}, "
        f"false_confirmed={false_confirmed_dve}"
    )
    assert false_confirmed_dve == 0
    assert len(clear_fog_ids) == 1, "Single continuous track ID must bridge clear -> radiation_fog!"
    assert CrossModalSimplexMode.NOMINAL_FUSION in simplex_modes_seen
    assert CrossModalSimplexMode.EO_DEGRADED_LWIR_PRIMARY in simplex_modes_seen
    assert regimes_confirmed.get("clear", 0) == 4
    assert regimes_confirmed.get("radiation_fog", 0) == 13
    assert regimes_confirmed.get("stratus_cloud", 0) == 5


def test_compute_12_sector_optical_quality_overcast_sky_and_true_degradations() -> None:
    """Campaign 1B-2 fix: clean featureless/overcast EO sky MUST score >= 0.70 (NOT blind-killed),
    while glare, all-black, constant flat-dead frames, dead-pixel FPN, and white noise score <= 0.15."""
    rng = np.random.default_rng(1812)

    # 1. Clean featureless overcast sky (MiG-29 / Hurricane class: smooth sky gradient + tiny aircraft silhouette)
    sky_grad = np.linspace(150.0, 190.0, 240, dtype=np.float64)[:, None] * np.ones((1, 320), dtype=np.float64)
    sky_overcast = np.clip(sky_grad + rng.normal(0.0, 0.8, size=(240, 320)), 0, 255).astype(np.uint8)
    sky_overcast[115:121, 155:165] = 42  # small dark aircraft silhouette
    sky_bgr = np.repeat(sky_overcast[:, :, None], 3, axis=2)
    q_sky = compute_12_sector_optical_quality(sky_bgr, modality="eo")
    assert len(q_sky) == 12
    assert float(np.mean(q_sky)) >= 0.90 and min(q_sky) >= 0.70, f"Overcast sky crushed: {q_sky}"

    # 2. True optical failures MUST fail closed (<= 0.15)
    all_black = np.zeros((240, 320, 3), dtype=np.uint8)
    all_white = np.full((240, 320, 3), 255, dtype=np.uint8)
    flat_dead = np.full((240, 320, 3), 128, dtype=np.uint8)
    white_noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    fpn_15 = sky_bgr.copy()
    for i, r in enumerate(np.linspace(4, 235, 36, dtype=int)):
        fpn_15[r, :, :] = 0 if (i % 2 == 0) else 255
    for i, c in enumerate(np.linspace(4, 315, 48, dtype=int)):
        fpn_15[:, c, :] = 255 if (i % 2 == 0) else 0

    for label, bad_img in (
        ("all_black", all_black),
        ("all_white", all_white),
        ("flat_dead", flat_dead),
        ("white_noise", white_noise),
        ("fpn_15", fpn_15),
    ):
        qs = compute_12_sector_optical_quality(bad_img, modality="eo")
        assert max(qs) <= 0.15, f"{label} bypassed 12-sector quality gate: {qs}"


def test_tbc_noise_health_gate_below_horizon_attenuation_and_cbdr_safe_static_clutter_gate() -> None:
    """Campaign 1B-1 fix: verify low-health noise/FPN rejection, below-horizon score attenuation,
    and opt-in CBDR-safe static-clutter gating (suppresses non-growing stationary ground clutter
    while PRESERVING zero-bearing-rate closing CBDR aircraft whose apparent size grows)."""
    # 1. Hard low-health gate (q_eo < min_modality_health=0.20) drops even score=0.99 detections
    trk_health = BearingTrackBeforeConfirm(TrackBeforeConfirmConfig(min_modality_health=0.20))
    for _ in range(5):
        res_h = trk_health.step(
            [BearingDetection(az_rad=0.05, el_rad=0.04, score=0.99, modality="eo")],
            dt_s=0.1,
            q_eo=0.10,
            q_lwir=0.0,
        )
        assert res_h.detections_accepted == 0
        assert res_h.detections_rejected_health == 1
        assert len(res_h.confirmed_tracks) == 0

    # 2. Below-horizon clutter attenuation (when horizon_el_rad is provided)
    trk_hor = BearingTrackBeforeConfirm(
        TrackBeforeConfirmConfig(
            weak_score_thr=0.15,
            min_confirm_score=0.25,
            below_horizon_score_scale=0.50,
            below_horizon_el_margin_rad=0.01,
        )
    )
    for _ in range(5):
        res_hor = trk_hor.step(
            [
                # Ground clutter below horizon (0.36 * 0.50 = 0.18 < min_confirm_score 0.25 -> blocked!)
                BearingDetection(az_rad=-0.10, el_rad=-0.05, score=0.36, modality="eo"),
                # Sky aircraft above horizon (0.36 >= 0.25 -> confirms!)
                BearingDetection(az_rad=0.10, el_rad=0.06, score=0.36, modality="eo"),
            ],
            dt_s=0.1,
            q_eo=1.0,
            q_lwir=0.0,
            horizon_el_rad=0.0,
        )
    assert len(res_hor.confirmed_tracks) == 1
    assert res_hor.confirmed_tracks[0].el_rad > 0.0

    # 3. CBDR-safe static clutter gate (`reject_static_clutter=True`):
    #    - Stationary non-growing background feature (constant bearing + constant size_px=8.0) -> rejected!
    #    - True CBDR closing intruder (EXACT zero bearing rate, growing size_px 8.0 -> 10.4) -> CONFIRMED!
    trk_cbdr = BearingTrackBeforeConfirm(
        TrackBeforeConfirmConfig(
            confirm_m=3,
            confirm_n=5,
            reject_static_clutter=True,
            static_bearing_rate_rad_s=0.003,
            static_min_size_growth_frac=0.05,
        )
    )
    for k in range(5):
        res_c = trk_cbdr.step(
            [
                # Static rock/cloud edge: zero bearing motion & zero apparent angular growth
                BearingDetection(az_rad=-0.15, el_rad=0.02, score=0.42, modality="eo", size_px=8.0),
                # Real CBDR closing aircraft: ZERO bearing rate, apparent size grows steadily!
                BearingDetection(
                    az_rad=0.12,
                    el_rad=0.03,
                    score=0.42,
                    modality="eo",
                    size_px=8.0 + 0.6 * k,
                ),
            ],
            dt_s=0.1,
            q_eo=1.0,
            q_lwir=0.0,
        )
    assert len(res_c.confirmed_tracks) == 1
    assert res_c.confirmed_tracks[0].az_rad == pytest.approx(0.12, abs=0.01)
    assert abs(res_c.confirmed_tracks[0].az_dot_rad_s) < 0.003
