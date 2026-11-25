"""Acceptance tests for visual QA contact sheets and dataset statistics.

Conforms to docs/xvs/batches/GEMINI_G-B1.md §2:
- Fast test with tiny synthetic fixture checking crop and overlay coordinates
  sitting on the mask within 1 px after ROI offset.
- Reader and statistics execution on G2 outputs of both backends (ref & hw).
- G3a schema parser test.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts.xvs.qa.make_contact_sheets import render_single_tile
from scripts.xvs.qa.qa_stats import compute_qa_statistics
from scripts.xvs.qa.reader import (
    QATileItem,
    load_qa_items_from_g2,
    load_qa_items_from_g3a,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


import tempfile


def test_synthetic_fixture_crop_and_overlay_alignment():
    """Synthetic fixture: overlay must sit on the mask to within 1 px after ROI offset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 1. Create a 1024x1024 synthetic mask with an intruder at image coords (50..54, 60..64)
        mask_w, mask_h = 1024, 1024
        mask_arr = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mask_arr[60:65, 50:55] = 255  # 5x5 px box at [50, 60, 54, 64]

        mask_path = tmp_path / "synthetic_mask_A.png"
        Image.fromarray(mask_arr, mode="L").save(mask_path)

    # 2. Assume camera ROI offset (u0, v0) = (300, 200)
    u0, v0 = 300, 200
    # Full image coordinates:
    # box_px = [350, 260, 354, 264]
    # Center = (352.0, 262.0)
    center_u = float(u0 + 52)
    center_v = float(v0 + 62)
    box_an = [float(u0 + 50), float(v0 + 60), float(u0 + 54), float(v0 + 64)]
    box_px = [u0 + 50, v0 + 60, u0 + 54, v0 + 64]

    item = QATileItem(
        instant_id="SYN_TEST_001",
        camera_id="T0",
        class_id="C1_UAS",
        range_m=1000.0,
        background="sky",
        visibility=1.0,
        mask_px=25,
        center_u=center_u,
        center_v=center_v,
        roi=(u0, v0),
        box_analytic=box_an,
        box_px=box_px,
        mask_path=mask_path,
        limb_dv_px=-50.0,
    )

    # 3. Render tile
    tile = render_single_tile(item)
    tile_arr = np.array(tile)

    # In the 128x128 crop:
    # Center in mask coords is (52, 62).
    # Crop top-left in mask coords is (52 - 64, 62 - 64) = (-12, -2).
    # The mask pixels [50, 60, 54, 64] in the crop are at:
    # x in [50 - (-12), 54 - (-12)] = [62, 66]
    # y in [60 - (-2), 64 - (-2)] = [62, 66]
    # In the upscaled 384x384 image (3x):
    # Mask pixels appear at x in [62 * 3, 66 * 3 + 2] = [186, 200]
    # y in [62 * 3, 66 * 3 + 2] = [186, 200]

    # 4. Verification:
    # Overlay analytic box (green: R < 50, G > 200, B < 50)
    rendered_crop = tile_arr[:384, :384]
    green_y, green_x = np.where(
        (rendered_crop[:, :, 0] < 50) & (rendered_crop[:, :, 1] > 200) & (rendered_crop[:, :, 2] < 50)
    )

    assert len(green_x) > 0, "Rendered tile contains no visible green analytic box pixels"
    measured_box_x1 = green_x.min()
    measured_box_x2 = green_x.max()
    measured_box_y1 = green_y.min()
    measured_box_y2 = green_y.max()

    # Theoretical coordinates of analytic box overlay mapped to tile:
    # u_crop = (box_an[0] - u0) - u_crop_min = (350 - 300) - (-12) = 62.0 -> 62.0 * 3 = 186.0
    u_crop_min = round(center_u - u0) - 64
    v_crop_min = round(center_v - v0) - 64
    expected_x1 = ((box_an[0] - u0) - u_crop_min) * 3.0
    expected_y1 = ((box_an[1] - v0) - v_crop_min) * 3.0
    expected_x2 = ((box_an[2] - u0) - u_crop_min) * 3.0
    expected_y2 = ((box_an[3] - v0) - v_crop_min) * 3.0

    # Requirement: The overlay must sit on the mask to within 1 px after the ROI offset
    assert abs(measured_box_x1 - expected_x1) <= 1.0, f"X1 mismatch: measured={measured_box_x1}, expected={expected_x1}"
    assert abs(measured_box_y1 - expected_y1) <= 1.0, f"Y1 mismatch: measured={measured_box_y1}, expected={expected_y1}"
    assert abs(measured_box_x2 - expected_x2) <= 1.0, f"X2 mismatch: measured={measured_box_x2}, expected={expected_x2}"
    assert abs(measured_box_y2 - expected_y2) <= 1.0, f"Y2 mismatch: measured={measured_box_y2}, expected={expected_y2}"


def test_g3a_schema_reader_synthetic():
    """Test G3a WP02 §5 reader parses schema lines accurately."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames_jsonl = tmp_path / "frames.jsonl"
        sample_frame = {
            "frame_id": "000042",
            "t_s": 4.2,
            "sequence_id": "enc_001",
            "traffic": [
                {
                    "id": "E0007",
                    "model_id": "C2_SEP_0002",
                    "class": "C2_SEP",
                    "range_m": 850.5,
                    "cams": {
                        "F": {
                            "u": 512.4,
                            "v": 490.2,
                            "z_m": 850.0,
                            "in_image": True,
                            "box_analytic": [508.0, 486.0, 516.0, 494.0],
                            "limb_dv_px": 15.2,
                            "background": "earth",
                            "amodal_px": 32,
                            "visibility": 0.85,
                            "box_amodal_px": [508, 486, 516, 494],
                        },
                        "T0": {
                            "u": 1100.0,
                            "v": 900.0,
                            "in_image": False,
                        }
                    }
                }
            ]
        }

        with open(frames_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_frame) + "\n")

        items = load_qa_items_from_g3a(tmp_path)
        assert len(items) == 1
        it = items[0]
        assert it.instant_id == "000042_E0007"
        assert it.camera_id == "F"
        assert it.class_id == "C2_SEP"
        assert it.range_m == 850.5
        assert it.background == "earth"
        assert it.visibility == 0.85
        assert it.mask_px == 32
        assert it.box_analytic == [508.0, 486.0, 516.0, 494.0]
        assert it.box_px == [508.0, 486.0, 516.0, 494.0]



from xvs.paths import get_data_root, REPO_ROOT


@pytest.mark.data
def test_g2_dataset_loader_and_stats_both_backends(check_data):
    """Verify loader and stats computation work on G2 outputs of ref and hw backends."""
    data_root = get_data_root()
    candidates = [
        data_root / "xvs" / "wp02" / "g2_b1",
        data_root / "xvs" / "wp02" / "g2",
    ]
    g2_root = None
    for cand in candidates:
        if cand.is_dir():
            g2_root = cand
            break

    if g2_root is None:
        check_data(data_root / "xvs" / "wp02" / "g2", "G2 output directory")

    # Check ref backend
    items_ref = load_qa_items_from_g2(g2_root, backend="ref")
    assert len(items_ref) > 0, "No items loaded for ref backend"
    stats_ref = compute_qa_statistics(items_ref)
    assert stats_ref["total_items"] == len(items_ref)
    assert len(stats_ref["breakdown"]) > 0

    # Check hw backend
    items_hw = load_qa_items_from_g2(g2_root, backend="hw")
    assert len(items_hw) > 0, "No items loaded for hw backend"
    stats_hw = compute_qa_statistics(items_hw)
    assert stats_hw["total_items"] == len(items_hw)
    assert len(stats_hw["breakdown"]) > 0
