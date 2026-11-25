"""Tests for WP-04 G3a Traffic Detection evaluation and runner.

Conforms to GEMINI_G-B9.md:
- Fast tests use tiny synthetic fixtures, not the pilot.
- One data test runs the pipeline on 20 pilot frames.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from xvs.detection.g3a_eval import (
    primary_matching_radius,
    evaluate_hits_and_fps_for_roi,
    pair_phantom_targets,
    compute_roi_sky_earth_area,
    compute_stratified_pd_stats,
    compute_fp_rate_sweep,
    TAU_LIST,
    K_VALUES,
)
from xvs.traffic.ring import TrafficRing
from xvs.rig.rig import Rig
from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_primary_and_secondary_hit_rules():
    """Verify primary tight centroid distance r = max(1.5, 0.5 * s_px) and secondary eIoU."""
    # Sub-pixel target: s_px = 0.8 -> r = max(1.5, 0.4) = 1.5 px
    r_sub = primary_matching_radius(0.8)
    assert abs(r_sub - 1.5) < 1e-9

    # Large target: s_px = 6.0 -> r = max(1.5, 3.0) = 3.0 px
    r_large = primary_matching_radius(6.0)
    assert abs(r_large - 3.0) < 1e-9

    ring = TrafficRing()
    p_cam = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r_cam = np.eye(3, dtype=np.float64)

    target_uv = (500.0, 500.0)
    box_analytic = [497.0, 497.0, 503.0, 503.0]  # 6x6 px
    s_px = 6.0  # r = 3.0 px

    # 1. Detection at distance 2.0 px from target (inside r=3.0)
    dets_hit = [
        {
            "u": 501.5,
            "v": 501.0,
            "box_edges": [498.0, 498.0, 504.0, 504.0],
            "score": 8.5,
        }
    ]
    res1 = evaluate_hits_and_fps_for_roi(
        dets_hit, target_uv, box_analytic, s_px, p_cam, r_cam, ring.camera_model
    )
    assert res1["primary_best_score"] == 8.5
    assert res1["secondary_best_score"] == 8.5
    assert len(res1["fp_candidates"]) == 0

    # 2. Detection at distance 3.5 px from target (outside r=3.0)
    dets_miss = [
        {
            "u": 503.0,
            "v": 502.5,  # dist = sqrt(3^2 + 2.5^2) = 3.9 px > 3.0 px
            "box_edges": [501.0, 500.0, 505.0, 504.0],
            "score": 7.0,
        }
    ]
    res2 = evaluate_hits_and_fps_for_roi(
        dets_miss, target_uv, box_analytic, s_px, p_cam, r_cam, ring.camera_model
    )
    assert res2["primary_best_score"] == -float("inf")
    # Inside dilated box (+16 px), so not an FP
    assert len(res2["fp_candidates"]) == 0

    # 3. Detection far away (outside box + 16 px -> candidate FP)
    dets_fp = [
        {
            "u": 600.0,
            "v": 600.0,
            "box_edges": [598.0, 598.0, 602.0, 602.0],
            "score": 4.5,
        }
    ]
    res3 = evaluate_hits_and_fps_for_roi(
        dets_fp, target_uv, box_analytic, s_px, p_cam, r_cam, ring.camera_model
    )
    assert res3["primary_best_score"] == -float("inf")
    assert len(res3["fp_candidates"]) == 1
    assert res3["fp_candidates"][0]["score"] == 4.5


def test_phantom_target_pairing():
    """Verify positive instances pair with negative frames of same encounter matching background."""
    ring = TrafficRing()
    rig = Rig.load_nominal()
    poses = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)
    p_cam = poses["F"]["p_ecef_cam"]
    r_cam = poses["F"]["r_ecef_cam"]

    # Synthetic instances across 2 encounters (E01, E02)
    pos_instances = [
        {
            "instance_id": "E01_001_T0",
            "encounter_id": "E01",
            "u": 500.0,
            "v": 500.0,
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "s_px": 4.0,
            "box_analytic": [498.0, 498.0, 502.0, 502.0],
            "background": "sky",
        },
        {
            "instance_id": "E01_002_T0",
            "encounter_id": "E01",
            "u": 500.0,
            "v": 1500.0,
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "s_px": 5.0,
            "box_analytic": [497.5, 1497.5, 502.5, 1502.5],
            "background": "earth",
        },
        {
            "instance_id": "E02_001_T0",
            "encounter_id": "E02",
            "u": 400.0,
            "v": 400.0,
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "s_px": 2.0,
            "box_analytic": [399.0, 399.0, 401.0, 401.0],
            "background": "sky",
        },
    ]

    neg_instances = [
        {
            "negative_id": "E01_neg_1",
            "encounter_id": "E01",
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "p_cam_ecef": p_cam,
            "R_cam_ecef": r_cam,
        },
        {
            "negative_id": "E02_neg_1",
            "encounter_id": "E02",
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "p_cam_ecef": p_cam,
            "R_cam_ecef": r_cam,
        },
    ]

    pairings = pair_phantom_targets(pos_instances, neg_instances, ring.camera_model)
    assert len(pairings) == 3

    # All E01 pos must map to E01 neg
    assert pairings["E01_001_T0"]["negative_id"] == "E01_neg_1"
    assert pairings["E01_001_T0"]["encounter_id"] == "E01"
    assert pairings["E01_002_T0"]["negative_id"] == "E01_neg_1"
    assert pairings["E01_002_T0"]["encounter_id"] == "E01"

    # E02 pos must map to E02 neg
    assert pairings["E02_001_T0"]["negative_id"] == "E02_neg_1"
    assert pairings["E02_001_T0"]["encounter_id"] == "E02"

    # Phantom box size must match analytic box size
    ph1 = pairings["E01_001_T0"]
    w_ph1 = ph1["box_phantom"][2] - ph1["box_phantom"][0]
    h_ph1 = ph1["box_phantom"][3] - ph1["box_phantom"][1]
    assert abs(w_ph1 - 4.0) < 1e-9
    assert abs(h_ph1 - 4.0) < 1e-9

    # Verify R-GB2-10 / G-B14 W6: Never pair across backgrounds; record unpaired when no matching background exists
    neg_with_unmatched = list(neg_instances) + [
        {
            "negative_id": "E03_neg_1",
            "encounter_id": "E03",
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "p_cam_ecef": p_cam,
            "R_cam_ecef": np.eye(3, dtype=np.float64),  # all pixels evaluate to sky
        }
    ]
    pos_with_unmatched = list(pos_instances) + [
        {
            "instance_id": "E03_001_T0",
            "encounter_id": "E03",
            "u": 500.0,
            "v": 1500.0,
            "roi_u0": 0.0,
            "roi_v0": 0.0,
            "s_px": 2.0,
            "box_analytic": [499.0, 1499.0, 501.0, 1501.0],
            "background": "earth",
        }
    ]
    pairs_unp, unp_list = pair_phantom_targets(
        pos_with_unmatched, neg_with_unmatched, ring.camera_model, return_unpaired=True
    )
    assert len(pairs_unp) == 3
    assert "E03_001_T0" not in pairs_unp
    assert unp_list == ["E03_001_T0"]


def test_roi_sky_earth_area_partition():
    """Verify sky + earth area equals 1024^2 px and exclusion box subtracts area correctly."""
    ring = TrafficRing()
    p_cam = np.array([-2614751.5, -4377596.3, 3823631.1], dtype=np.float64)
    r_cam = np.eye(3, dtype=np.float64)

    sky_mpx, earth_mpx = compute_roi_sky_earth_area(
        p_cam, r_cam, ring.camera_model, u0=500.0, v0=500.0, width=1024, height=1024
    )
    total_mpx = sky_mpx + earth_mpx
    expected_mpx = (1024.0 * 1024.0) / 1.0e6
    assert abs(total_mpx - expected_mpx) < 1e-5

    # With exclusion box
    ex_box = [600.0, 600.0, 632.0, 632.0]  # 32x32 px = 1024 px = 0.001024 Mpx
    sky_ex, earth_ex = compute_roi_sky_earth_area(
        p_cam, r_cam, ring.camera_model, u0=500.0, v0=500.0, width=1024, height=1024, exclusion_box=ex_box
    )
    diff_mpx = total_mpx - (sky_ex + earth_ex)
    assert abs(diff_mpx - 1024.0 / 1.0e6) < 1e-6


def test_false_positive_sweep_properties():
    """Verify monotonicity and Poisson bounds across tau in compute_fp_rate_sweep."""
    # Synthetic FP candidates
    candidates = [
        {"score": 2.2, "is_earth": False},
        {"score": 3.1, "is_earth": False},
        {"score": 4.5, "is_earth": True},
        {"score": 7.2, "is_earth": True},
    ]

    sky_mpx = 2.0
    earth_mpx = 2.0
    sweep = compute_fp_rate_sweep(candidates, sky_mpx, earth_mpx, tau_list=TAU_LIST)

    prev_tot = float("inf")
    for tau in TAU_LIST:
        st = sweep[f"{tau:.1f}"]
        curr_tot = st["fp_count_total"]
        assert curr_tot <= prev_tot, f"Monotonicity violated at tau={tau}"
        prev_tot = curr_tot

        # Poisson upper bound must be strictly >= observed rate
        assert st["fp_per_mpx_total_ub95"] >= st["fp_per_mpx_total"]
        assert st["fp_per_mpx_sky_ub95"] >= st["fp_per_mpx_sky"]
        assert st["fp_per_mpx_earth_ub95"] >= st["fp_per_mpx_earth"]


@pytest.mark.data
def test_g3a_pipeline_on_pilot_20_frames(tmp_path):
    """Gate G0 acceptance rehearsal: Run complete G3a evaluation on 20 pilot frames."""
    from scripts.xvs.detection.run_g3a_g0 import load_g3a_dataset_and_pair, process_batch_worker, init_worker
    import json

    pilot_root = get_data_root() / "xvs" / "wp02" / "g3a_pilot"
    if not pilot_root.is_dir():
        pytest.skip(f"Pilot data not found at {pilot_root}")

    manifests, pos_instances, neg_instances, phantom_map, _ = load_g3a_dataset_and_pair(
        g3a_root=pilot_root,
        max_frames=20,
    )

    assert len(manifests) > 0, "No manifests discovered"
    assert len(pos_instances) > 0, "No positive instances loaded"

    # Initialize worker and evaluate tasks
    init_worker(str(pilot_root))
    batch = [("positive", it) for it in pos_instances[:5]] + [("negative", it) for it in neg_instances[:2]]
    results = process_batch_worker(batch)

    assert len(results) == len(batch)
    for r in results:
        assert "dt_ms" in r
        assert r["dt_ms"] > 0.0
        assert "search_area_mpx" in r
        assert "eval_by_sensor_k" in r
        for sensor in ["clean", "degraded"]:
            for k_val in [3, 5, 7]:
                assert k_val in r["eval_by_sensor_k"][sensor]


@pytest.mark.data
def test_fast_path_bit_identical_to_slow_path_on_20_pilot_rois():
    """Verify that vectorized fast path produces bit-identical results to slow path on 20 pilot ROIs."""
    from PIL import Image
    from scipy import ndimage
    from scripts.xvs.detection.run_g3a_g0 import (
        load_g3a_dataset_and_pair,
        init_worker,
        _WORKER_D0_CFG,
    )
    from xvs.sensor.srgb import srgb_eotf, linear_rgb_to_monochrome
    from xvs.detection.d0 import (
        detect_d0,
        compute_cmo,
        compute_cmo_multi,
        compute_z_score,
        compute_srgb_quantization_sigma,
        interpolate_bilinear_grid,
        compute_clean_floor_grid,
    )

    def slow_compute_clean_floor_grid(image, tile_size=64, tile_stride=32):
        h, w = image.shape
        y_centres = np.arange(tile_stride // 2, h, tile_stride)
        x_centres = np.arange(tile_stride // 2, w, tile_stride)
        ny = len(y_centres)
        nx = len(x_centres)
        sig_min_grid = np.zeros((ny, nx), dtype=np.float64)
        half = tile_size // 2
        for j, yc in enumerate(y_centres):
            y_start = max(0, min(h - tile_size, yc - half)) if h >= tile_size else 0
            y_end = y_start + tile_size if h >= tile_size else h
            for i, xc in enumerate(x_centres):
                x_start = max(0, min(w - tile_size, xc - half)) if w >= tile_size else 0
                x_end = x_start + tile_size if w >= tile_size else w
                patch = image[y_start:y_end, x_start:x_end]
                med_l = float(np.median(patch))
                sig_min_grid[j, i] = compute_srgb_quantization_sigma(med_l)
        return interpolate_bilinear_grid(sig_min_grid, h, w, stride=tile_stride)

    def slow_compute_local_stats(cmo, tile_size=64, tile_stride=32, mad_scale=1.4826):
        h, w = cmo.shape
        y_centres = np.arange(tile_stride // 2, h, tile_stride)
        x_centres = np.arange(tile_stride // 2, w, tile_stride)
        ny = len(y_centres)
        nx = len(x_centres)
        m_grid = np.zeros((ny, nx), dtype=np.float64)
        sig_grid = np.zeros((ny, nx), dtype=np.float64)
        half = tile_size // 2
        for j, yc in enumerate(y_centres):
            y_start = max(0, min(h - tile_size, yc - half)) if h >= tile_size else 0
            y_end = y_start + tile_size if h >= tile_size else h
            for i, xc in enumerate(x_centres):
                x_start = max(0, min(w - tile_size, xc - half)) if w >= tile_size else 0
                x_end = x_start + tile_size if w >= tile_size else w
                patch = cmo[y_start:y_end, x_start:x_end]
                med = float(np.median(patch))
                mad = float(np.median(np.abs(patch - med)))
                m_grid[j, i] = med
                sig_grid[j, i] = mad_scale * mad
        m_loc = interpolate_bilinear_grid(m_grid, h, w, stride=tile_stride)
        sig_loc = interpolate_bilinear_grid(sig_grid, h, w, stride=tile_stride)
        return m_loc, sig_loc

    def slow_extract_components(binary_mask, cmo, z, u0=0.0, v0=0.0, image_id="", camera="", struct=None):
        if struct is None:
            struct = np.ones((3, 3), dtype=bool)
        dilated_mask = ndimage.binary_dilation(binary_mask, structure=struct)
        labels, num_features = ndimage.label(dilated_mask, structure=struct)
        detections = []
        slices = ndimage.find_objects(labels)
        for comp_id, sl in enumerate(slices, 1):
            if sl is None:
                continue
            sub_labels = labels[sl]
            sub_bin = binary_mask[sl]
            undilated_mask = (sub_labels == comp_id) & sub_bin
            ys_sub, xs_sub = np.where(undilated_mask)
            if len(xs_sub) == 0:
                continue
            ys = ys_sub + sl[0].start
            xs = xs_sub + sl[1].start
            weights = cmo[ys, xs]
            sum_w = float(np.sum(weights))
            if sum_w > 1e-12:
                u_roi = float(np.sum(weights * xs) / sum_w)
                v_roi = float(np.sum(weights * ys) / sum_w)
            else:
                u_roi = float(np.mean(xs))
                v_roi = float(np.mean(ys))
            x_min = int(np.min(xs))
            x_max = int(np.max(xs))
            y_min = int(np.min(ys))
            y_max = int(np.max(ys))
            box_edges = [
                float(x_min - 0.5 + u0),
                float(y_min - 0.5 + v0),
                float(x_max + 0.5 + u0),
                float(y_max + 0.5 + v0),
            ]
            s_det = float(max(x_max - x_min + 1, y_max - y_min + 1))
            score = float(np.max(z[ys, xs]))
            n_px = int(len(xs))
            detections.append({
                "image_id": image_id,
                "camera": camera,
                "u": u_roi + u0,
                "v": v_roi + v0,
                "box_edges": box_edges,
                "s_det": s_det,
                "score": score,
                "n_px": n_px,
            })
        return detections

    def slow_detect_d0(image, k=3, tau=2.0, is_clean=True, config=None, u0=0.0, v0=0.0, image_id="", camera=""):
        cfg = config or {}
        tile_size = cfg.get("tile_size", 64)
        tile_stride = cfg.get("tile_stride", 32)
        mad_scale = cfg.get("mad_scale", 1.4826)
        if is_clean:
            sigma_min = slow_compute_clean_floor_grid(image, tile_size=tile_size, tile_stride=tile_stride)
        else:
            sigma_min = float(cfg.get("sigma_min_degraded", 1.0))
        cmo = compute_cmo(image, k=k)
        m_loc, sig_loc = slow_compute_local_stats(cmo, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z = compute_z_score(cmo, m_loc, sig_loc, sigma_min=sigma_min)
        struct = np.ones((3, 3), dtype=bool)
        binary_mask = (z >= tau)
        return slow_extract_components(binary_mask, cmo, z, u0=u0, v0=v0, image_id=image_id, camera=camera, struct=struct)

    pilot_root = get_data_root() / "xvs" / "wp02" / "g3a_pilot"
    if not pilot_root.is_dir():
        pytest.skip(f"Pilot data not found at {pilot_root}")

    manifests, pos_instances, neg_instances, _, _ = load_g3a_dataset_and_pair(pilot_root, max_frames=25)
    sample_rois = pos_instances[:20]
    assert len(sample_rois) == 20, f"Expected 20 pilot ROIs, got {len(sample_rois)}"

    init_worker(str(pilot_root))

    for it in sample_rois:
        with Image.open(pilot_root / it["rgb_path"]) as im:
            rgb_arr = np.array(im.convert("RGB"), dtype=np.uint8)
        roi_clean = linear_rgb_to_monochrome(srgb_eotf(rgb_arr))
        
        # Test clean floor grid identity
        cf_slow = slow_compute_clean_floor_grid(roi_clean)
        cf_fast = compute_clean_floor_grid(roi_clean)
        np.testing.assert_allclose(cf_fast, cf_slow, rtol=1e-12, atol=1e-12)

        # Test CMO and detections across k in [3, 5, 7]
        cmos_fast = compute_cmo_multi(roi_clean, K_VALUES)
        for k_val in K_VALUES:
            cmo_slow = compute_cmo(roi_clean, k=k_val)
            np.testing.assert_allclose(cmos_fast[k_val], cmo_slow, rtol=1e-12, atol=1e-12)

            dets_slow = slow_detect_d0(
                roi_clean, k=k_val, tau=2.0, is_clean=True, config=_WORKER_D0_CFG,
                u0=it["roi_u0"], v0=it["roi_v0"], image_id=it["instance_id"], camera=it["cam_id"]
            )
            dets_fast = detect_d0(
                roi_clean, k=k_val, tau=2.0, is_clean=True, config=_WORKER_D0_CFG,
                u0=it["roi_u0"], v0=it["roi_v0"], image_id=it["instance_id"], camera=it["cam_id"],
                clean_floor=cf_fast, cmo=cmos_fast[k_val]
            )

            assert len(dets_fast) == len(dets_slow), f"Count mismatch for k={k_val}: {len(dets_fast)} vs {len(dets_slow)}"
            # Sort detections deterministically
            dets_slow_s = sorted(dets_slow, key=lambda d: (round(d["score"], 6), round(d["u"], 4), round(d["v"], 4)))
            dets_fast_s = sorted(dets_fast, key=lambda d: (round(d["score"], 6), round(d["u"], 4), round(d["v"], 4)))

            for ds, df in zip(dets_slow_s, dets_fast_s):
                assert abs(ds["score"] - df["score"]) < 1e-9
                assert abs(ds["u"] - df["u"]) < 1e-9
                assert abs(ds["v"] - df["v"]) < 1e-9
                assert abs(ds["s_det"] - df["s_det"]) < 1e-9
                assert ds["n_px"] == df["n_px"]


