"""Unit tests for WP-04 evaluation exactness (V2) and contrast estimation exactness (V3).

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §3, §4, and §7.
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from xvs.metrics.aot import (
    GTObject,
    GTFrame,
    Detection,
    DetectionFrame,
    load_gt_from_g2,
    dilate_box,
)
from xvs.detection import evaluate, contrast
from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v2_evaluation_exactness_synthetic():
    """Gate G0 Acceptance V2 (Hermetic): Verify oracle, empty, and shifted detector properties."""
    # Create 20 synthetic frames across 3 classes, multiple ranges, and horizons
    gt_frames = []
    classes = ["cessna_172", "b737_800", "general_glider"]
    ranges = [400.0, 1500.0, 4500.0, 8000.0]
    horizons = ["above", "below"]

    rng = np.random.RandomState(42)

    for i in range(20):
        cls_id = classes[i % len(classes)]
        rng_m = ranges[i % len(ranges)]
        hz = horizons[i % len(horizons)]
        u_c = float(rng.uniform(200.0, 800.0))
        v_c = float(rng.uniform(200.0, 800.0))
        w = float(rng.uniform(4.0, 15.0))
        h = float(rng.uniform(3.0, 12.0))
        box = [u_c - w / 2.0, v_c - h / 2.0, u_c + w / 2.0, v_c + h / 2.0]

        obj = GTObject(
            object_id=f"target_{i}",
            class_id=cls_id,
            range_m=rng_m,
            box_edges=box,
            horizon=hz,
        )
        gt_frames.append(
            GTFrame(
                frame_id=f"frame_{i:04d}",
                t_s=float(i) * 0.1,
                sequence_id=f"enc_{i % 5}",
                camera_id="F",
                image_wh=(1024, 1024),
                objects=[obj],
            )
        )

    # 1. Oracle detections: exact GT boxes
    oracle_det_frames = [
        DetectionFrame(
            frame_id=f.frame_id,
            camera_id=f.camera_id,
            detections=[Detection(box=list(obj.box_edges), score=1.0) for obj in f.objects],
        )
        for f in gt_frames
    ]
    res_oracle = evaluate.evaluate_detections(gt_frames, oracle_det_frames)
    assert res_oracle["overall_pd"] == 1.0, "Oracle must yield P_d = 1.0"
    assert res_oracle["total_fp"] == 0, "Oracle must yield 0 FP"
    for cls_name, stat in res_oracle["by_class"].items():
        assert stat["pd"] == 1.0, f"Class {cls_name} oracle P_d != 1.0"
    for rng_name, stat in res_oracle["by_range"].items():
        assert stat["pd"] == 1.0, f"Range {rng_name} oracle P_d != 1.0"

    # 2. Empty detector: 0 detections
    empty_det_frames = [
        DetectionFrame(frame_id=f.frame_id, camera_id=f.camera_id, detections=[])
        for f in gt_frames
    ]
    res_empty = evaluate.evaluate_detections(gt_frames, empty_det_frames)
    assert res_empty["overall_pd"] == 0.0, "Empty detector must yield P_d = 0.0"
    assert res_empty["total_fp"] == 0, "Empty detector must yield 0 FP"

    # 3. Shifted detector: shift in u by dilated width + 2 px
    shifted_det_frames = []
    for f in gt_frames:
        shifted_dets = []
        for obj in f.objects:
            dilated = dilate_box(obj.box_edges, min_area=100.0)
            dil_w = dilated[2] - dilated[0]
            shift = dil_w + 2.0
            s_box = [
                obj.box_edges[0] + shift,
                obj.box_edges[1],
                obj.box_edges[2] + shift,
                obj.box_edges[3],
            ]
            shifted_dets.append(Detection(box=s_box, score=1.0))
        shifted_det_frames.append(
            DetectionFrame(frame_id=f.frame_id, camera_id=f.camera_id, detections=shifted_dets)
        )

    res_shifted = evaluate.evaluate_detections(gt_frames, shifted_det_frames)
    assert res_shifted["overall_pd"] == 0.0, "Shifted detector must yield P_d = 0.0"
    assert res_shifted["total_fp"] == len(gt_frames), "Every single-intruder shifted box must be an FP"


@pytest.mark.data
def test_v2_evaluation_exactness_eval12(check_data):
    """Gate G0 Acceptance V2 on real E-VAL-12 dataset."""
    data_root = get_data_root()
    g2_json = check_data(data_root / "xvs" / "wp02" / "g2" / "results_ref.json", "E-VAL-12 results JSON")
    with open(g2_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_frames = load_gt_from_g2(data, use_mask_box=False)
    # Filter to frames with active GT
    gt_active = [f for f in gt_frames if len(f.objects) > 0]
    assert len(gt_active) > 0

    # 1. Oracle detections
    oracle_frames = [
        DetectionFrame(
            frame_id=f.frame_id,
            camera_id=f.camera_id,
            detections=[Detection(box=list(o.box_edges), score=1.0) for o in f.objects],
        )
        for f in gt_active
    ]
    res_oracle = evaluate.evaluate_detections(gt_active, oracle_frames)
    assert res_oracle["overall_pd"] == 1.0
    assert res_oracle["total_fp"] == 0

    # 2. Empty detector
    empty_frames = [
        DetectionFrame(frame_id=f.frame_id, camera_id=f.camera_id, detections=[])
        for f in gt_active
    ]
    res_empty = evaluate.evaluate_detections(gt_active, empty_frames)
    assert res_empty["overall_pd"] == 0.0
    assert res_empty["total_fp"] == 0

    # 3. Shifted detector
    single_intruder_frames = [f for f in gt_active if len(f.objects) == 1]
    shifted_frames = []
    for f in single_intruder_frames:
        o = f.objects[0]
        dil = dilate_box(o.box_edges, min_area=100.0)
        dil_w = dil[2] - dil[0]
        shift = dil_w + 2.0
        s_box = [o.box_edges[0] + shift, o.box_edges[1], o.box_edges[2] + shift, o.box_edges[3]]
        shifted_frames.append(
            DetectionFrame(
                frame_id=f.frame_id,
                camera_id=f.camera_id,
                detections=[Detection(box=s_box, score=1.0)],
            )
        )
    res_shifted = evaluate.evaluate_detections(single_intruder_frames, shifted_frames)
    assert res_shifted["overall_pd"] == 0.0
    assert res_shifted["total_fp"] == len(single_intruder_frames)


def test_v3_contrast_exactness_flat_background():
    """Gate G0 Acceptance V3: Exact integrated contrast recovery on flat background."""
    h, w = 128, 128
    bg_level = 100.0
    contrast_c = 0.25

    # Synthetic sub-pixel alpha coverage mask
    alpha = np.zeros((h, w), dtype=np.float64)
    # 3x3 blob with fractions inside
    alpha[60, 60] = 0.8
    alpha[60, 61] = 1.0
    alpha[61, 60] = 0.9
    alpha[61, 61] = 0.5
    true_sum_alpha = float(np.sum(alpha))
    expected_ac = contrast_c * true_sum_alpha

    # Injected image: L' = L_bg * (1 + C * alpha)
    img = bg_level * (1.0 + contrast_c * alpha)

    # Boolean amodal mask covering support of alpha
    amodal_mask = (alpha > 0.0)

    # Measure target signal
    res = contrast.estimate_target_signal(
        image=img,
        amodal_mask=amodal_mask,
        center_uv=(60.5, 60.5),
        mad_scale=1.4826,
    )

    rel_error = abs(res.a_c - expected_ac) / max(1e-12, abs(expected_ac))
    print(f"\n[V3 Contrast Exactness] Expected A_C = {expected_ac:.6f}, Measured = {res.a_c:.6f}, Rel Error = {rel_error:.2e}")
    assert rel_error < 1e-9, f"A_C relative error {rel_error:.2e} exceeds 1e-9 threshold"
    assert abs(res.l_bar_bg - bg_level) < 1e-12, "Background level must match exact flat level"
    assert abs(res.sigma_bg) < 1e-12, "Background noise must be zero on flat image"
