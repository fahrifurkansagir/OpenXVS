"""Tests for end-to-end imaging chain, configuration loading and determinism (V6).

Spec reference: docs/xvs/WP03_SENSOR.md §1, §2 and §3 item V6.
- The same seed gives byte-identical raw and display outputs.
- A different frame gives different temporal noise but the same fixed-pattern maps.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest

from xvs.sensor.chain import SensorConfig, CameraSimulator, save_raw_png, save_display_png


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "xvs" / "sensors"


def test_config_loading():
    """Verify loading ring_mono_v0 and rig_mono_v0 configs."""
    ring_cfg_path = CONFIG_DIR / "ring_mono_v0.json"
    rig_cfg_path = CONFIG_DIR / "rig_mono_v0.json"

    assert ring_cfg_path.exists(), f"Missing {ring_cfg_path}"
    assert rig_cfg_path.exists(), f"Missing {rig_cfg_path}"

    ring_cfg = SensorConfig.from_file(ring_cfg_path)
    rig_cfg = SensorConfig.from_file(rig_cfg_path)

    assert ring_cfg.image_width == 2448
    assert ring_cfg.image_height == 2048
    assert ring_cfg.fx == 2319.0
    assert ring_cfg.k1 == -0.05
    assert ring_cfg.k_dn_per_e == 0.35

    assert rig_cfg.image_width == 1024
    assert rig_cfg.image_height == 1024
    assert rig_cfg.k1 == -0.10


def test_v6_determinism_byte_identical():
    """V6 requirement: The same seed gives byte-identical raw and display outputs."""
    cfg = SensorConfig.from_file(CONFIG_DIR / "rig_mono_v0.json")
    # Use smaller test image for speed
    cfg.image_width = 128
    cfg.image_height = 128
    cfg.cx = 63.5
    cfg.cy = 63.5

    sim1 = CameraSimulator(cfg, sensor_id="CAM_TEST_01")
    sim2 = CameraSimulator(cfg, sensor_id="CAM_TEST_01")

    # Synthetic render input
    rng = np.random.Generator(np.random.PCG64(100))
    test_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    res1 = sim1.process_frame(test_img, image_id="img_001", frame_idx=5)
    res2 = sim2.process_frame(test_img, image_id="img_001", frame_idx=5)

    # Array exact equality
    np.testing.assert_array_equal(res1["raw"], res2["raw"])
    np.testing.assert_array_equal(res1["display"], res2["display"])

    # PNG byte-identical equality using BytesIO
    buf_raw1 = io.BytesIO()
    buf_raw2 = io.BytesIO()
    from PIL import Image
    Image.fromarray(res1["raw"]).save(buf_raw1, format="PNG")
    Image.fromarray(res2["raw"]).save(buf_raw2, format="PNG")
    assert buf_raw1.getvalue() == buf_raw2.getvalue()

    buf_disp1 = io.BytesIO()
    buf_disp2 = io.BytesIO()
    Image.fromarray(res1["display"], mode="L").save(buf_disp1, format="PNG")
    Image.fromarray(res2["display"], mode="L").save(buf_disp2, format="PNG")
    assert buf_disp1.getvalue() == buf_disp2.getvalue()


def test_v6_different_frame_temporal_vs_fixed_pattern():
    """V6 requirement: A different frame gives different temporal noise but the same fixed-pattern maps."""
    cfg = SensorConfig.from_file(CONFIG_DIR / "rig_mono_v0.json")
    cfg.image_width = 128
    cfg.image_height = 128
    cfg.cx = 63.5
    cfg.cy = 63.5

    sim = CameraSimulator(cfg, sensor_id="CAM_TEST_02")
    sim_ref = CameraSimulator(cfg, sensor_id="CAM_TEST_02")
    initial_prnu = np.copy(sim.prnu_map)
    initial_dsnu = np.copy(sim.dsnu_map)

    test_img = np.full((128, 128), 128, dtype=np.uint8)

    res_f1 = sim.process_frame(test_img, image_id="img_001", frame_idx=1)
    res_f2 = sim.process_frame(test_img, image_id="img_001", frame_idx=2)

    # Fixed pattern noise maps must remain preserved and match reference instance
    np.testing.assert_array_equal(sim.prnu_map, initial_prnu)
    np.testing.assert_array_equal(sim.dsnu_map, initial_dsnu)
    np.testing.assert_array_equal(sim.prnu_map, sim_ref.prnu_map)
    np.testing.assert_array_equal(sim.dsnu_map, sim_ref.dsnu_map)

    # Outputs must differ due to temporal noise
    diff_raw = np.abs(res_f1["raw"].astype(int) - res_f2["raw"].astype(int))
    assert np.max(diff_raw) > 0, "Frames with different frame indices should have distinct noise"


def test_different_sensor_id_changes_fpn():
    """Different sensor_id gives distinct fixed-pattern noise maps."""
    cfg = SensorConfig.from_file(CONFIG_DIR / "rig_mono_v0.json")
    cfg.image_width = 64
    cfg.image_height = 64
    sim_a = CameraSimulator(cfg, sensor_id="CAM_A")
    sim_b = CameraSimulator(cfg, sensor_id="CAM_B")

    assert not np.array_equal(sim_a.prnu_map, sim_b.prnu_map)
    assert not np.array_equal(sim_a.dsnu_map, sim_b.dsnu_map)


def test_mismatched_image_dimensions_rejected():
    """Verify CameraSimulator raises ValueError if image shape does not match config."""
    cfg = SensorConfig.from_file(CONFIG_DIR / "rig_mono_v0.json")
    cfg.image_width = 128
    cfg.image_height = 128
    sim = CameraSimulator(cfg, sensor_id="CAM_TEST_DIM")

    bad_img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="do not match sensor config"):
        sim.process_frame(bad_img, image_id="bad", frame_idx=0)
