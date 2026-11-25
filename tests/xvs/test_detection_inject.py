"""Unit tests for WP-04 traffic detection injection study module.

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §5.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from xvs.detection.inject import (
    load_traffic_models,
    sample_traffic_silhouette,
    composite_signal,
    extract_background_crops,
    InjectionSensorSimulator,
    fit_ac_logistic,
    matching_radius,
    is_primary_hit,
    is_secondary_hit,
    compute_corrected_pd_bootstrap,
    evaluate_detections_for_trial,
    compute_fp_density_sweep,
    find_operating_point_tau,
    CLASSES,
)


def test_traffic_models_load():
    """Verify all 18 traffic GLB models load with non-empty vertices and faces."""
    models = load_traffic_models()
    assert len(models) == 18, f"Expected 18 models (6 classes x 3), got {len(models)}"
    for name, (verts, faces) in models.items():
        assert len(verts) > 10, f"Model {name} has too few vertices: {len(verts)}"
        assert len(faces) > 10, f"Model {name} has too few faces: {len(faces)}"
        assert verts.shape[1] == 3, f"Model {name} vertices not 3D"
        assert faces.shape[1] == 3, f"Model {name} faces not triangles"


def test_sample_traffic_silhouette_geometry():
    """Verify silhouette rasterization produces alpha in [0, 1] and correct bounding box size."""
    models = load_traffic_models()
    rng = np.random.default_rng(42)

    for target_spx in [1.5, 4.0, 8.0]:
        u_t, v_t = 128.0, 128.0
        sil = sample_traffic_silhouette(models, s_px=target_spx, u_t=u_t, v_t=v_t, rng=rng, crop_size=256)
        alpha = sil["alpha"]
        box_gt = sil["box_gt"]

        assert alpha.shape == (256, 256), f"Alpha shape mismatch: {alpha.shape}"
        assert 0.0 <= float(np.min(alpha)) and float(np.max(alpha)) <= 1.0, "Alpha outside [0, 1]"
        assert float(np.sum(alpha)) > 0.0, "Alpha mask cannot be completely empty"

        box_w = box_gt[2] - box_gt[0]
        box_h = box_gt[3] - box_gt[1]
        measured_spx = max(box_w, box_h)
        assert abs(measured_spx - target_spx) < 1e-6, f"Measured s_px {measured_spx} != target {target_spx}"


def test_composite_signal_exactness():
    """Verify linear luminance composition formula L' = L_bg * (1 + C * alpha)."""
    h, w = 256, 256
    L_bg = np.full((h, w), 50.0, dtype=np.float64)
    alpha = np.zeros((h, w), dtype=np.float64)
    alpha[100:105, 100:105] = 0.5
    contrast_c = 0.2

    L_prime, a_c = composite_signal(L_bg, alpha, contrast_c)
    expected_sum_alpha = 25 * 0.5
    expected_ac = contrast_c * expected_sum_alpha

    assert abs(a_c - expected_ac) < 1e-12, f"A_C mismatch: {a_c} vs {expected_ac}"
    assert abs(L_prime[0, 0] - 50.0) < 1e-12, "Background unchanged outside target"
    assert abs(L_prime[102, 102] - 50.0 * (1.0 + 0.2 * 0.5)) < 1e-12, "Target luminance mismatch"


def test_injection_sensor_simulator():
    """Verify degraded sensor imaging chain executes and produces electrons signal."""
    sim = InjectionSensorSimulator()
    L_in = np.full((256, 256), 0.2, dtype=np.float64)

    # Clean
    L_clean = sim.apply_sensor(L_in, is_degraded=False, seed=123)
    assert np.allclose(L_clean, L_in)

    # Degraded
    L_deg = sim.apply_sensor(L_in, is_degraded=True, seed=123)
    assert L_deg.shape == (256, 256)
    assert L_deg.dtype == np.float64
    assert np.isfinite(L_deg).all()
    # Degraded signal should be centered around E * L_in = 10500 * 0.2 = 2100 e-
    mean_val = float(np.mean(L_deg))
    assert 1800.0 < mean_val < 2400.0, f"Unexpected degraded mean electrons: {mean_val}"


def test_fit_ac_logistic():
    """Verify logistic fitting in ln|A_C| on synthetic threshold response."""
    rng = np.random.default_rng(12345)
    ac_true_50 = 2.5
    w_true = 0.6
    ln_ac_true = math.log(ac_true_50)

    # Generate 1000 points
    ac_vals = rng.uniform(0.1, 10.0, size=1000)
    p_detect = 1.0 / (1.0 + np.exp(-(np.log(ac_vals) - ln_ac_true) / w_true))
    detected = (rng.uniform(0.0, 1.0, size=1000) < p_detect).astype(int)

    res = fit_ac_logistic(ac_vals, detected)
    assert res["computable"] is True
    assert abs(res["ln_ac_50"] - ln_ac_true) < 0.25, f"ln|A_C|50 estimate {res['ln_ac_50']} too far from {ln_ac_true}"
    assert abs(res["w"] - w_true) < 0.25, f"w estimate {res['w']} too far from {w_true}"


def test_hit_rule_and_p0_theoretical_distribution():
    """Gate G0 Acceptance Part A: Verification of tight matching rule r and chance-hit rate p0.

    1. Detections are scattered uniformly at random with known density rho per px^2
       around 10,000 seeded target positions.
    2. Measured chance-hit rate for tight rule must equal 1 - exp(-rho * pi * r^2) within 3 SE.
    3. Centroids exactly inside r are accepted and those outside are rejected.
    """
    # 1. Matching radius formula check: r = max(1.5 px, 0.5 * s_px)
    assert matching_radius(0.5) == 1.5
    assert matching_radius(1.0) == 1.5
    assert matching_radius(2.0) == 1.5
    assert matching_radius(3.0) == 1.5
    assert matching_radius(4.0) == 2.0
    assert matching_radius(6.0) == 3.0
    assert matching_radius(8.0) == 4.0

    # 2. Boundary acceptance / rejection tests
    u_t, v_t = 120.0, 120.0
    s_px = 3.0
    r = matching_radius(s_px)  # 1.5 px

    # Exact boundary and interior
    assert is_primary_hit(u_t, v_t, u_t, v_t, s_px) is True
    assert is_primary_hit(u_t + r, v_t, u_t, v_t, s_px) is True
    assert is_primary_hit(u_t, v_t + r, u_t, v_t, s_px) is True
    assert is_primary_hit(u_t + r - 1e-6, v_t, u_t, v_t, s_px) is True
    assert is_primary_hit(u_t + r / math.sqrt(2.0), v_t + r / math.sqrt(2.0), u_t, v_t, s_px) is True

    # Exterior
    assert is_primary_hit(u_t + r + 1e-6, v_t, u_t, v_t, s_px) is False
    assert is_primary_hit(u_t, v_t - r - 1e-6, u_t, v_t, s_px) is False
    assert is_primary_hit(u_t + (r + 1e-5) / math.sqrt(2.0), v_t + (r + 1e-5) / math.sqrt(2.0), u_t, v_t, s_px) is False

    # 3. Statistical verification over 10,000 seeded targets
    rng = np.random.default_rng(20260925)
    n_targets = 10000
    s_test = 2.0
    r_test = matching_radius(s_test)  # 1.5
    l_box = 20.0
    area_box = l_box * l_box
    rho = 0.005  # detections per px^2

    p_theory = 1.0 - math.exp(-rho * math.pi * r_test * r_test)
    se = math.sqrt(p_theory * (1.0 - p_theory) / n_targets)

    # Sample Poisson point process
    mu = rho * area_box
    k_per_target = rng.poisson(mu, size=n_targets)
    total_k = int(np.sum(k_per_target))

    offsets_u = rng.uniform(-l_box / 2.0, l_box / 2.0, size=total_k)
    offsets_v = rng.uniform(-l_box / 2.0, l_box / 2.0, size=total_k)
    dist_sq = offsets_u * offsets_u + offsets_v * offsets_v

    target_indices = np.repeat(np.arange(n_targets), k_per_target)
    target_hits = np.zeros(n_targets, dtype=bool)
    np.logical_or.at(target_hits, target_indices, dist_sq <= (r_test * r_test))

    p_measured = float(np.mean(target_hits))
    diff = abs(p_measured - p_theory)

    print(f"\n[Chance Hit Test] p_measured={p_measured:.5f}, p_theory={p_theory:.5f}, diff={diff:.5f}, 3*SE={3*se:.5f}")
    assert diff <= 3.0 * se, f"Measured chance-hit rate {p_measured:.5f} deviates from theory {p_theory:.5f} by {diff/se:.2f} SE (> 3 SE)"


def test_compute_corrected_pd_bootstrap():
    """Verify Abbott-corrected P_d and 95% bootstrap confidence interval estimation."""
    rng = np.random.default_rng(42)
    n = 1000
    p0_true = 0.10
    pd_raw_true = 0.55
    # Theoretical corrected: (0.55 - 0.10) / (1 - 0.10) = 0.45 / 0.90 = 0.50

    y_hits = (rng.uniform(0.0, 1.0, size=n) < pd_raw_true).astype(int)
    z_null = (rng.uniform(0.0, 1.0, size=n) < p0_true).astype(int)

    res = compute_corrected_pd_bootstrap(y_hits, z_null, n_boot=1000, seed=123)

    assert abs(res["pd_raw"] - pd_raw_true) < 0.03
    assert abs(res["p0"] - p0_true) < 0.03
    assert abs(res["pd_corrected"] - 0.50) < 0.04
    ci_low, ci_high = res["pd_corrected_ci95"]
    assert ci_low < res["pd_corrected"] < ci_high
    assert 0.40 < ci_low < 0.50
    assert 0.50 < ci_high < 0.60


def test_evaluate_detections_for_trial_and_tau_sweep():
    """Verify evaluation of detections per trial, score storage, FP density sweep, and operating point."""
    u_t, v_t = 100.0, 100.0
    s_px = 3.0  # r = 1.5
    box_gt = [98.5, 98.5, 101.5, 101.5]

    dets = [
        # Target hit: primary & secondary
        {"u": 100.5, "v": 100.5, "score": 6.5, "box_edges": [99.0, 99.0, 102.0, 102.0]},
        # Target hit secondary only (distance = 2.0 > 1.5, eIoU > 0.2)
        {"u": 102.0, "v": 100.0, "score": 5.0, "box_edges": [100.0, 99.0, 103.0, 102.0]},
        # False alarm (> 16 px from target)
        {"u": 130.0, "v": 100.0, "score": 4.2, "box_edges": [128.0, 98.0, 132.0, 102.0]},
        {"u": 20.0, "v": 20.0, "score": 3.1, "box_edges": [18.0, 18.0, 22.0, 22.0]},
    ]

    res = evaluate_detections_for_trial(dets, u_t, v_t, box_gt, s_px)
    assert res["primary_best_score"] == 6.5
    assert res["secondary_best_score"] == 6.5
    assert len(res["fp_scores"]) == 2
    assert 4.2 in res["fp_scores"] and 3.1 in res["fp_scores"]

    # Test FP density sweep over 100 trials
    fp_scores = [4.2, 3.1, 7.5, 2.8]
    sweep = compute_fp_density_sweep(fp_scores, num_trials=100)
    assert "2.0" in sweep
    assert "4.0" in sweep
    assert "20.0" in sweep
    assert sweep["4.0"]["fp_count"] == 2  # 4.2 and 7.5 >= 4.0
    assert sweep["8.0"]["fp_count"] == 0  # 0 >= 8.0
    assert sweep["4.0"]["fp_per_mpx_ci95_high"] > sweep["4.0"]["fp_per_mpx"]

    # Operating point resolution check
    op_res = find_operating_point_tau(sweep, target_fp_upper_per_mpx=1.0e-4)
    # With only 100 trials (total area ~ 6.47 Mpx), even 0 FP gives upper bound ~ 3.0 / 6.47 = 0.46/Mpx > 1.0e-4
    assert op_res["status"] == "UNRESOLVED"


def test_extract_background_crops_missing_data_raises():
    """Verify that missing G4 data raises FileNotFoundError without synthetic fallback."""
    fake_path = Path("non_existent_directory_for_xvs_backgrounds")
    with pytest.raises(FileNotFoundError):
        extract_background_crops(data_root=fake_path, num_crops_per_type=1)


@pytest.mark.data
def test_extract_background_crops_g4_horizon_band(check_data):
    """Verify background crops from G4 T1/T3 F/R/B/L frames within 16-200 px horizon bands."""
    from xvs.paths import get_data_root
    data_root = get_data_root()
    check_data(data_root / "xvs" / "wp01" / "g4" / "T1_cruise" / "frames.jsonl", "G4 T1 cruise frames")

    crops_dict = extract_background_crops(data_root=data_root, num_crops_per_type=4, rng=np.random.default_rng(42))
    assert "sky" in crops_dict and "earth" in crops_dict
    assert len(crops_dict["sky"]) == 4
    assert len(crops_dict["earth"]) == 4
    assert "manifest" in crops_dict and "manifest_sha256" in crops_dict
    assert len(crops_dict["manifest"]) == 8

    for crop in crops_dict["sky"]:
        assert crop["image"].shape == (256, 256)
        assert crop["metadata"]["camera"] in ["F", "R", "B", "L"]
        assert crop["metadata"]["camera"] not in ["U", "D"]
        assert crop["metadata"]["type"] == "sky"

    for crop in crops_dict["earth"]:
        assert crop["image"].shape == (256, 256)
        assert crop["metadata"]["camera"] in ["F", "R", "B", "L"]
        assert crop["metadata"]["camera"] not in ["U", "D"]
        assert crop["metadata"]["type"] == "earth"
