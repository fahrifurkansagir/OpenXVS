"""Tests for AOT traffic perception and encounter tracking metrics library.

Verifies acceptance requirements §1 of docs/xvs/batches/GEMINI_G-B1.md:
1. eIoU dilation hand-calculated cases (smaller/larger than 100 px², non-square).
2. Synthetic sequences with known answers for AFDR, FPPI, EDR, HFAR, and don't-care cases.
3. Director-verified benchmark interval values from R1 digest §3 & §4.1.
4. End-to-end smoke test on G2 results_ref.json with perfect and shifted detections.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import scipy.stats as stats

from xvs.metrics.aot import (
    AOTConfig,
    Detection,
    DetectionFrame,
    GTFrame,
    GTObject,
    clopper_pearson_interval,
    compute_eiou,
    compute_iou,
    dilate_box,
    evaluate_aot,
    load_gt_from_g2,
    load_gt_from_g3a,
    poisson_rate_upper_bound,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test 1: eIoU Dilation Cases
# ---------------------------------------------------------------------------

def test_dilate_box_smaller_than_min_area():
    """Box smaller than 100 px² dilated about its centre keeping aspect ratio."""
    # Box: 4 px wide, 8 px high -> Area = 32 px², aspect ratio w/h = 0.5
    # Center = (12.0, 24.0)
    # Scale = sqrt(100 / 32) = sqrt(3.125)
    # w' = 4 * sqrt(3.125) = sqrt(50) ~= 7.0710678
    # h' = 8 * sqrt(3.125) = sqrt(200) ~= 14.1421356
    box = [10.0, 20.0, 14.0, 28.0]
    dilated = dilate_box(box, min_area=100.0)

    u_min, v_min, u_max, v_max = dilated
    w = u_max - u_min
    h = v_max - v_min
    area = w * h
    u_c = (u_min + u_max) / 2.0
    v_c = (v_min + v_max) / 2.0

    assert pytest.approx(area, rel=1e-6) == 100.0
    assert pytest.approx(w / h, rel=1e-6) == 0.5
    assert pytest.approx(u_c, rel=1e-6) == 12.0
    assert pytest.approx(v_c, rel=1e-6) == 24.0
    assert pytest.approx(w, rel=1e-6) == math.sqrt(50.0)
    assert pytest.approx(h, rel=1e-6) == math.sqrt(200.0)


def test_dilate_box_larger_and_equal_min_area():
    """Boxes at or above 100 px² remain unchanged."""
    # Exactly 100 px² (10x10)
    b_exact = [0.0, 0.0, 10.0, 10.0]
    assert dilate_box(b_exact, min_area=100.0) == b_exact

    # Larger square box (20x20 = 400 px²)
    b_large_sq = [10.0, 10.0, 30.0, 30.0]
    assert dilate_box(b_large_sq, min_area=100.0) == b_large_sq

    # Larger non-square box (100x10 = 1000 px²)
    b_large_rect = [50.0, 50.0, 150.0, 60.0]
    assert dilate_box(b_large_rect, min_area=100.0) == b_large_rect


def test_dilate_box_degenerate_point():
    """Point or line boxes expand gracefully to 100 px² preserving center."""
    point = [50.0, 50.0, 50.0, 50.0]
    dilated = dilate_box(point, min_area=100.0)
    assert pytest.approx((dilated[2] - dilated[0]) * (dilated[3] - dilated[1]), rel=1e-6) == 100.0
    assert pytest.approx((dilated[0] + dilated[2]) / 2.0, rel=1e-6) == 50.0
    assert pytest.approx((dilated[1] + dilated[3]) / 2.0, rel=1e-6) == 50.0


def test_eiou_dilation_allows_small_target_match():
    """Verify that eIoU prevents false miss for small targets with minor shift."""
    # GT box: 5x5 px at (10, 10) to (15, 15), area = 25 px²
    # Standard IoU with 3 px shift would be 4 / 46 = 0.087 < 0.2
    # Dilated to 10x10: GT dilated is [7.5, 7.5, 17.5, 17.5]
    # Det shifted by 3 px: [13, 13, 18, 18], dilated to 10x10: [10.5, 10.5, 20.5, 20.5]
    # Intersection = 7x7 = 49, Union = 100 + 100 - 49 = 151 -> eIoU = 49 / 151 ~= 0.3245 > 0.2
    gt = [10.0, 10.0, 15.0, 15.0]
    det = [13.0, 13.0, 18.0, 18.0]

    raw_iou = compute_iou(gt, det)
    assert raw_iou < 0.2

    eiou = compute_eiou(gt, det, min_area=100.0)
    assert pytest.approx(eiou, rel=1e-4) == 49.0 / 151.0
    assert eiou > 0.2  # Matches under eIoU!


# ---------------------------------------------------------------------------
# Test 2: Synthetic Sequences with Hand-Known Answers
# ---------------------------------------------------------------------------

def test_synthetic_sequence_complete_aot_metrics():
    """Synthetic sequence with known AFDR, FPPI, EDR, HFAR and don't-care cases."""
    # Setup: 41 frames at 10 Hz (t = 0.0 to 4.0 s, dt = 0.1 s). Flight duration = 4.0 s.
    frames: list[GTFrame] = []
    det_frames: list[DetectionFrame] = []

    # Objects:
    # 1. obj_valid: C1_UAS, starts at 400 m, closes to 200 m (min=200<=330, max=400<=700). Valid encounter!
    #    Matched by trk_1 for all t in [0.0, 3.5] s (duration 3.5 s >= 3.0 s).
    #    Comes under 300 m at t = 2.0 s.
    #    Detection duration 3.0 s reached at t = 3.0 s.
    #    Does it satisfy detection criteria? Yes: reached within first 3.0 s of encounter (t_detect = 3.0 <= t_start + 3.0 = 3.0).
    # 2. obj_distant: C2_SEP, constant 1200 m (> 700 m).
    #    Not detected. Counts as don't care, NOT a miss for AFDR.
    # 3. obj_far_matched: C3_TWIN, constant 1500 m (> 700 m).
    #    Matched by trk_far. Counts as don't care, NOT a false positive.
    # 4. A false detection with track_id = 'trk_fa' at frames 0..4 (5 frames).
    #    Counts as 1 false alarm for HFAR.
    # 5. Untracked false detections (track_id = None) at frames 10 and 11 (2 frames).
    #    Counts as 2 false alarms for HFAR.
    # 6. A detection in the don't-care band (eIoU = 0.1 against obj_valid) at frame 20.
    #    Counts as don't care, NOT a false positive.

    for i in range(41):
        t = round(i * 0.1, 2)
        # obj_valid closes from 400 m at t=0 to 200 m at t=4
        r_valid = 400.0 - 50.0 * t

        gt_objects = [
            GTObject(
                object_id="obj_valid",
                class_id="C1_UAS",
                range_m=r_valid,
                box_edges=[100.0, 100.0, 120.0, 120.0],
                horizon="above",
            ),
            GTObject(
                object_id="obj_distant",
                class_id="C2_SEP",
                range_m=1200.0,
                box_edges=[500.0, 500.0, 510.0, 510.0],
                horizon="below",
            ),
            GTObject(
                object_id="obj_far_matched",
                class_id="C3_TWIN",
                range_m=1500.0,
                box_edges=[800.0, 800.0, 810.0, 810.0],
                horizon="unknown",
            ),
        ]

        frames.append(
            GTFrame(
                frame_id=f"f_{i}",
                t_s=t,
                sequence_id="seq_1",
                camera_id="F",
                image_wh=(1024, 1024),
                objects=gt_objects,
            )
        )

        dets: list[Detection] = []
        # Detection for obj_valid up to t = 3.5 s
        if t <= 3.5:
            dets.append(
                Detection(
                    box=[100.0, 100.0, 120.0, 120.0],
                    score=0.9,
                    track_id="trk_1",
                )
            )

        # Detection for obj_far_matched (range 1500 m > 700 m)
        dets.append(
            Detection(
                box=[800.0, 800.0, 810.0, 810.0],
                score=0.85,
                track_id="trk_far",
            )
        )

        # False detection with track_id = 'trk_fa' for first 5 frames
        if i < 5:
            dets.append(
                Detection(
                    box=[300.0, 300.0, 310.0, 310.0],
                    score=0.7,
                    track_id="trk_fa",
                )
            )

        # Untracked false detections at frame 10 and 11
        if i in (10, 11):
            dets.append(
                Detection(
                    box=[400.0, 400.0, 410.0, 410.0],
                    score=0.6,
                    track_id=None,
                )
            )

        # Don't-care band detection at frame 20 (shifted partially to give eIoU ~ 0.1 with obj_valid)
        # GT is [100, 100, 120, 120] (20x20). Center (110, 110).
        # Box [115, 115, 135, 135] has intersection [115, 115, 120, 120] = 5x5 = 25.
        # Union = 400 + 400 - 25 = 775 -> IoU = 25 / 775 = 0.0322 in [0.02, 0.2]
        if i == 20:
            dets.append(
                Detection(
                    box=[115.0, 115.0, 135.0, 135.0],
                    score=0.4,  # lower score so it doesn't preempt trk_1
                    track_id="trk_dc",
                )
            )

        det_frames.append(
            DetectionFrame(
                frame_id=f"f_{i}",
                camera_id="F",
                detections=dets,
            )
        )

    summary = evaluate_aot(frames, det_frames)

    # 1. Frame level verification:
    # Total images: 41
    assert summary.total_images == 41
    # Only obj_valid is within 700 m (41 instances). obj_distant and obj_far_matched are > 700 m.
    assert summary.gt_eval_objects == 41
    # obj_valid is matched for frames 0..35 (36 matches). Frames 36..40 are misses (5 misses).
    assert summary.matched_gt_objects == 36
    assert pytest.approx(summary.afdr, rel=1e-5) == 36 / 41
    # False positives: 5 from trk_fa + 2 untracked = 7 false positives.
    # trk_far matches >700m GT -> don't care. trk_dc has eIoU=0.0322 -> don't care.
    assert summary.false_positives == 7
    assert pytest.approx(summary.fppi, rel=1e-5) == 7 / 41

    # 2. Encounter level verification:
    # obj_valid: 1 encounter, valid=True (min=200<=330, max=400<=700). Detected=True.
    assert summary.valid_encounters == 1
    assert summary.detected_valid_encounters == 1
    assert summary.edr == 1.0

    # 3. HFAR verification:
    # Flight hours: 4.0 s / 3600.0 h
    expected_hours = 4.0 / 3600.0
    assert pytest.approx(summary.total_flight_hours, rel=1e-6) == expected_hours
    # False alarms: 1 track ('trk_fa') + 2 untracked = 3 false alarms
    assert summary.false_alarms == 3
    assert pytest.approx(summary.hfar, rel=1e-5) == 3.0 / expected_hours


# ---------------------------------------------------------------------------
# Test 3: Director-Verified R1 Digest Statistical Benchmark Numbers
# ---------------------------------------------------------------------------

def test_director_verified_clopper_pearson_98_of_103():
    """Clopper-Pearson for 98/103: [0.890, 0.984] to 3 decimals."""
    low, high = clopper_pearson_interval(98, 103, confidence=0.95)
    assert round(low, 3) == 0.890
    assert round(high, 3) == 0.984


def test_director_verified_poisson_hfar_winner():
    """Flight hours = 1/0.152285 h. With 2 false alarms the 95% upper bound is 0.96/h."""
    flight_hours = 1.0 / 0.152285
    upper = poisson_rate_upper_bound(2, flight_hours, confidence=0.95)
    assert round(upper, 2) == 0.96


def test_director_verified_zero_false_alarms_hours_needed():
    """With zero false alarms, showing HFAR <= 0.5/h at 95% needs >= 6.0 h."""
    # We find exposure T where poisson_rate_upper_bound(0, T, 0.95) <= 0.5
    # For k=0: mu_upper = 0.5 * chi2.ppf(0.95, 2) = -ln(0.05) ~= 2.99573
    # T = 2.99573 / 0.5 = 5.99146 h -> 6.0 h (1 decimal)
    mu_0 = 0.5 * float(stats.chi2.ppf(0.95, 2))
    hours_needed = mu_0 / 0.5
    assert round(hours_needed, 1) == 6.0
    # And at T = 6.0 h, the upper bound is <= 0.50
    assert poisson_rate_upper_bound(0, 6.0, confidence=0.95) < 0.50


# ---------------------------------------------------------------------------
# Test 4: End-to-End Smoke Test on G2 results_ref.json
# ---------------------------------------------------------------------------

from xvs.paths import get_data_root


@pytest.mark.data
def test_g2_results_ref_smoke_test(check_data, tmp_path):
    """End-to-end smoke test on G2 results_ref.json:

    - Perfect detections: AFDR = 1.0, FPPI = 0.0
    - Shifted by 50 px: AFDR = 0.0, FPPI > 0.0
    - Summaries saved as JSON under output/xvs/metrics_smoke/
    """
    data_root = get_data_root()
    candidate_paths = [
        data_root / "xvs" / "wp02" / "g2_b1" / "results_ref.json",
        data_root / "xvs" / "wp02" / "g2" / "results_ref.json",
    ]

    ref_json_path: Path | None = None
    for cp in candidate_paths:
        if cp.is_file():
            ref_json_path = cp
            break

    if ref_json_path is None:
        ref_json_path = check_data(data_root / "xvs" / "wp02" / "g2" / "results_ref.json", "G2 results_ref.json")

    with open(ref_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_frames = load_gt_from_g2(data, use_mask_box=False)
    assert len(gt_frames) > 0, "No GT frames loaded from results_ref.json"

    # 1. Build perfect detections equal to the GT boxes
    perfect_det_frames: list[DetectionFrame] = []
    for gf in gt_frames:
        dets = [
            Detection(
                box=list(obj.box_edges),
                score=1.0,
                track_id=obj.object_id,
            )
            for obj in gf.objects
        ]
        perfect_det_frames.append(
            DetectionFrame(
                frame_id=gf.frame_id,
                camera_id=gf.camera_id,
                detections=dets,
            )
        )

    summary_perfect = evaluate_aot(gt_frames, perfect_det_frames)
    assert summary_perfect.gt_eval_objects > 0
    assert summary_perfect.afdr == 1.0, f"Expected AFDR=1.0, got {summary_perfect.afdr}"
    assert summary_perfect.false_positives == 0, f"Expected 0 FPs, got {summary_perfect.false_positives}"
    assert summary_perfect.fppi == 0.0

    # 2. Shift every box by 50 px
    shifted_det_frames: list[DetectionFrame] = []
    for gf in gt_frames:
        dets = [
            Detection(
                box=[
                    obj.box_edges[0] + 50.0,
                    obj.box_edges[1] + 50.0,
                    obj.box_edges[2] + 50.0,
                    obj.box_edges[3] + 50.0,
                ],
                score=0.9,
                track_id=f"shift_{obj.object_id}",
            )
            for obj in gf.objects
        ]
        shifted_det_frames.append(
            DetectionFrame(
                frame_id=gf.frame_id,
                camera_id=gf.camera_id,
                detections=dets,
            )
        )

    summary_shifted = evaluate_aot(gt_frames, shifted_det_frames)
    assert summary_shifted.afdr == 0.0, f"Expected AFDR=0.0, got {summary_shifted.afdr}"
    assert summary_shifted.false_positives > 0, "Expected FP > 0 for 50px shifted boxes"
    assert summary_shifted.fppi > 0.0

    # 3. Save both summaries under tmp_path / metrics_smoke/
    out_dir = tmp_path / "metrics_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "perfect_summary.json", "w", encoding="utf-8") as f:
        f.write(summary_perfect.to_json())

    with open(out_dir / "shifted_summary.json", "w", encoding="utf-8") as f:
        f.write(summary_shifted.to_json())

    assert (out_dir / "perfect_summary.json").is_file()
    assert (out_dir / "shifted_summary.json").is_file()


# ---------------------------------------------------------------------------
# Additional Deep Acceptance & Edge-Case Tests
# ---------------------------------------------------------------------------

def test_encounter_closing_from_long_range():
    """Verify encounter entering from > 700 m (e.g. 1000 m down to 100 m) is a valid detected encounter."""
    frames = []
    det_frames = []
    # 61 frames at 10 Hz (6.0 s total).
    # Intruder closes from 1000 m to 100 m at 150 m/s:
    # At t=0.0..2.0 s (i=0..20): range 1000..700 m (> 700 m, outside eval range).
    # At t=2.0..6.0 s (i=20..60): range 700..100 m (inside encounter, duration 4.0 s >= 3.0 s).
    # Closest distance is 100 m <= 330 m. Maximum distance in encounter is 700 m <= 700 m.
    # Tracker acquires target at t=2.0 s and tracks continuously with trk_1.
    # Comes under 300 m at t=4.67 s (i~47). 3 s tracking achieved at t=5.0 s (within first 3 s of encounter: 2.0 + 3.0 = 5.0 s).
    for i in range(61):
        t = round(i * 0.1, 2)
        r = 1000.0 - 15.0 * i
        frames.append(
            GTFrame(
                frame_id=f"f_{i}",
                t_s=t,
                sequence_id="seq_closing",
                camera_id="F",
                image_wh=(1024, 1024),
                objects=[GTObject("obj_close", "C1_UAS", r, [100.0, 100.0, 120.0, 120.0], "above")],
            )
        )
        det_frames.append(
            DetectionFrame(
                frame_id=f"f_{i}",
                camera_id="F",
                detections=[Detection([100.0, 100.0, 120.0, 120.0], 0.95, "trk_1")],
            )
        )

    summary = evaluate_aot(frames, det_frames)
    assert summary.valid_encounters == 1, f"Expected 1 valid encounter, got {summary.valid_encounters}"
    assert summary.detected_valid_encounters == 1
    assert summary.edr == 1.0


def test_multi_sequence_false_alarm_tracking():
    """Verify identical track IDs across different sequences are counted as distinct false alarms."""
    # Sequence A has false alarm track 'trk_1'
    f1 = [GTFrame("f_0", 0.0, "seq_A", "F", (1024, 1024), [])]
    d1 = [DetectionFrame("f_0", "F", [Detection([10.0, 10.0, 20.0, 20.0], 0.9, "trk_1")])]

    # Sequence B has false alarm track 'trk_1' (tracker restarted in new sequence)
    f2 = [GTFrame("f_1", 0.0, "seq_B", "F", (1024, 1024), [])]
    d2 = [DetectionFrame("f_1", "F", [Detection([10.0, 10.0, 20.0, 20.0], 0.9, "trk_1")])]

    summary = evaluate_aot(f1 + f2, d1 + d2)
    assert summary.false_alarms == 2, f"Expected 2 false alarms across 2 sequences, got {summary.false_alarms}"


def test_detection_on_empty_negative_frames():
    """Verify detections on negative images omitted from GT frames are scored as false positives."""
    # GT list contains only frame f_0 (which has no GT objects)
    # Detector also ran on negative image f_1 and emitted a false positive
    f = [GTFrame("f_0", 0.0, "seq_A", "F", (1024, 1024), [])]
    d = [
        DetectionFrame("f_0", "F", []),
        DetectionFrame("f_1", "F", [Detection([50.0, 50.0, 70.0, 70.0], 0.8, "trk_fp")]),
    ]

    summary = evaluate_aot(f, d)
    assert summary.total_images == 2
    assert summary.false_positives == 1
    assert pytest.approx(summary.fppi, rel=1e-5) == 0.5


def test_dilate_box_and_eiou_inverted_coords():
    """Verify robustness when box coordinates are inverted (u_min > u_max)."""
    # Box [30, 30, 10, 10]: width 20, height 20 -> area 400 (larger than min_area)
    dilated = dilate_box([30.0, 30.0, 10.0, 10.0], min_area=100.0)
    assert dilated == [10.0, 10.0, 30.0, 30.0]

    # IoU between inverted box and standard box
    iou = compute_iou([30.0, 30.0, 10.0, 10.0], [10.0, 10.0, 30.0, 30.0])
    assert pytest.approx(iou, rel=1e-6) == 1.0


def test_load_gt_from_g3a_empty_traffic_and_custom_cameras():
    """Verify G3a loader with empty traffic list and custom camera list preserves all frames."""
    records = [
        {"frame_id": "001", "t_s": 0.0, "sequence_id": "s1", "traffic": []},
        {
            "frame_id": "002",
            "t_s": 0.1,
            "sequence_id": "s1",
            "traffic": [
                {
                    "id": "obj_1",
                    "class": "C1_UAS",
                    "range_m": 500.0,
                    "cams": {
                        "F": {
                            "in_image": True,
                            "box_analytic": [10.0, 10.0, 20.0, 20.0],
                            "limb_dv_px": 5.0,
                        }
                    },
                }
            ],
        },
    ]

    # Without explicit cameras: frame 001 gets default "F", frame 002 gets "F"
    gt_frames = load_gt_from_g3a(records)
    assert len(gt_frames) == 2
    assert len(gt_frames[0].objects) == 0
    assert len(gt_frames[1].objects) == 1

    # With explicit cameras: both frames get all 6 cameras
    cams_6 = ["F", "R", "B", "L", "U", "D"]
    gt_frames_multi = load_gt_from_g3a(records, cameras=cams_6)
    assert len(gt_frames_multi) == 12  # 2 frames * 6 cameras

