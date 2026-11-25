"""Render suite tests for XVSRenderer, camera direct assignment, and reprojection (Gate G2).

Marked strictly with @pytest.mark.render to keep the fast suite decoupled.
Conforms to docs/xvs/WP01_SIM_RIG.md v1.2 §2.4, §2.5, §3 (G2), §4.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import socket
import threading
import numpy as np
import pytest

from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from scripts.xvs.validate_markers import (
    build_g2_scene_specs,
    generate_scene_markers,
    evaluate_scene,
    THRESHOLDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CESIUM_BUILD = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
PAGE_DIR = REPO_ROOT / "sim" / "cesium" / "page"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "usgs_imagery"
G2_DIR = REPO_ROOT / "output" / "xvs" / "wp01" / "g2"


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def xvs_server():
    """Start threaded XVSServer fixture for render tests."""
    port = get_free_port()
    log_dir = REPO_ROOT / "output" / "xvs" / "test_logs"
    server = XVSServer(
        server_address=("127.0.0.1", port),
        cesium_build_dir=CESIUM_BUILD,
        page_dir=PAGE_DIR,
        cache_dir=CACHE_DIR,
        log_dir=log_dir,
        offline=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.shutdown()
    server.server_close()


def test_launch_args_exhaustive_per_backend():
    """Verify XVSRenderer launch arguments equal exhaustive §2.5 list with no extra flags (C19)."""
    r_ref = XVSRenderer(port=8000, backend="ref")
    assert r_ref.get_browser_args() == [
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
    ]

    r_hw = XVSRenderer(port=8000, backend="hw")
    assert r_hw.get_browser_args() == [
        "--use-angle=d3d11",
        "--enable-gpu",
        "--ignore-gpu-blocklist",
    ]


@pytest.mark.render
def test_renderer_all_four_profiles_match_json(xvs_server):
    """Verify that all 4 render profiles initialize and return resolved settings equal to render_profiles.json (C15)."""
    server, port = xvs_server
    profiles_cfg_path = REPO_ROOT / "configs" / "xvs" / "render_profiles.json"
    with open(profiles_cfg_path, "r", encoding="utf-8") as f:
        profiles_expected = json.load(f)["profiles"]

    with XVSRenderer(port=port, backend="ref", server=server) as renderer:
        for profile, expected in profiles_expected.items():
            res = renderer.set_profile(profile)
            assert res["success"] is True
            assert res["profile"] == profile
            assert res["msaa_samples"] == expected["msaa_samples"]
            assert res["sky_box"] == expected["sky_box"]
            assert res["sky_atmosphere"] == expected["sky_atmosphere"]
            assert res["sun_moon"] == expected["sun_moon"]
            assert res["globe_show"] == expected["globe_show"]
            assert res["ground_atmosphere"] == expected["ground_atmosphere"]
            assert res["lighting"] == expected["lighting"]
            assert res["fog"] == expected["fog"]
            assert res["fxaa"] == expected["fxaa"]
            assert res["hdr"] == expected["hdr"]
            assert res["maximum_screen_space_error"] == expected["maximum_screen_space_error"]
            if expected["background_color"] == "sky":
                assert res["background_color"] == "sky"
            else:
                np.testing.assert_allclose(res["background_color"], expected["background_color"], atol=1e-5)
            np.testing.assert_allclose(res["base_color"], expected["base_color"], atol=1e-5)
            assert res["drawing_buffer"] == [1024, 1024]


@pytest.mark.render
def test_drawing_buffer_guard_fires_on_wrong_canvas_size(xvs_server):
    """Verify drawing-buffer / canvas client size guard throws when container size or CSS is broken (C12)."""
    server, port = xvs_server
    with XVSRenderer(port=port, backend="ref", server=server) as renderer:
        page = renderer.get_page()

        # 1. Break container width
        page.evaluate("document.getElementById('cesiumContainer').style.width = '500px'")
        with pytest.raises(Exception) as exc_info:
            renderer.set_profile("mask_earth")
        assert "Canvas / drawing buffer size guard failed" in str(exc_info.value)

        # Restore container width
        page.evaluate("document.getElementById('cesiumContainer').style.width = '1024px'")
        res = renderer.set_profile("mask_earth")
        assert res["success"] is True
        assert res["drawing_buffer"] == [1024, 1024]

        # 2. Break CSS (disable widgets.css, reproducing the original sandbox failure)
        page.evaluate("""() => {
            const links = document.getElementsByTagName('link');
            for (let l of links) {
                if (l.href.includes('widgets.css')) l.disabled = true;
            }
        }""")
        with pytest.raises(Exception) as exc_info_css:
            renderer.set_profile("mask_earth")
        assert "Canvas / drawing buffer size guard failed" in str(exc_info_css.value)

        # Restore page and verify normal initialization succeeds
        renderer._load_cesium_page(page)
        res_restored = renderer.set_profile("mask_earth")
        assert res_restored["success"] is True
        assert res_restored["drawing_buffer"] == [1024, 1024]


@pytest.mark.render
def test_direct_camera_vectors_and_intrinsics(xvs_server):
    """Verify direct camera assignment in Cesium matches requested pose via atan2 on both readbacks (C13)."""
    server, port = xvs_server
    rig = Rig.load_nominal().make_true(20260922)
    # Test near-vertical pose (regime where setView fails)
    poses = rig.compute_poses(37.62, -122.37, 1000.0, math.radians(40.0), math.radians(0.5), 0.0)
    cam_u = poses["U"]

    with XVSRenderer(port=port, backend="ref", server=server) as renderer:
        renderer.set_profile("markers")
        readback_set, ms_set = renderer.set_camera(
            position=cam_u["position"],
            direction=cam_u["direction"],
            up=cam_u["up"],
            fov_rad=cam_u["fov_rad"],
            aspect_ratio=cam_u["aspect_ratio"],
        )

        from scripts.xvs.validate_markers import angle_between_vectors

        # Check readback after set
        pos_err_set = np.linalg.norm(np.array(readback_set["position"]) - cam_u["position"])
        dir_err_set = angle_between_vectors(np.array(readback_set["direction"]), cam_u["direction"])
        up_err_set = angle_between_vectors(np.array(readback_set["up"]), cam_u["up"])

        assert pos_err_set <= THRESHOLDS["m3_pos_err_m"], f"Set pos error {pos_err_set} m exceeds 1 mm"
        assert dir_err_set <= THRESHOLDS["m3_dir_err_deg"], f"Set dir error {dir_err_set} deg exceeds 1e-6 deg"
        assert up_err_set <= THRESHOLDS["m3_up_err_deg"], f"Set up error {up_err_set} deg exceeds 1e-6 deg"

        # Check readback after render
        render_res = renderer.render_frame()
        readback_render = render_res.readback
        pos_err_render = np.linalg.norm(np.array(readback_render["position"]) - cam_u["position"])
        dir_err_render = angle_between_vectors(np.array(readback_render["direction"]), cam_u["direction"])
        up_err_render = angle_between_vectors(np.array(readback_render["up"]), cam_u["up"])

        assert pos_err_render <= THRESHOLDS["m3_pos_err_m"], f"Render pos error {pos_err_render} m exceeds 1 mm"
        assert dir_err_render <= THRESHOLDS["m3_dir_err_deg"], f"Render dir error {dir_err_render} deg exceeds 1e-6 deg"
        assert up_err_render <= THRESHOLDS["m3_up_err_deg"], f"Render up error {up_err_render} deg exceeds 1e-6 deg"
        assert render_res.drawing_buffer == (1024, 1024)
        assert len(render_res.rgb_sha256) == 64


@pytest.mark.render
def test_marker_projection_m2_exact(xvs_server):
    """Verify M2: worldToWindowCoordinates - 0.5 vs pinhole projection is <= 1e-3 px."""
    server, port = xvs_server
    rig = Rig.load_nominal().make_true(20260922)
    poses = rig.compute_poses(47.40, 8.50, 600.0, math.radians(30.0), 0.0, 0.0)
    cam_f = poses["F"]

    markers = generate_scene_markers(rig, cam_f, seed=20260923, n_markers=40)

    with XVSRenderer(port=port, backend="ref", server=server) as renderer:
        renderer.set_profile("markers")
        renderer.set_camera(
            position=cam_f["position"],
            direction=cam_f["direction"],
            up=cam_f["up"],
            fov_rad=cam_f["fov_rad"],
            aspect_ratio=cam_f["aspect_ratio"],
        )
        renderer.set_markers(markers)
        projections = renderer.get_marker_projections(markers)

        assert len(projections) == 40
        max_err = 0.0
        for m, proj in zip(markers, projections):
            assert proj is not None, f"Marker {m['id']} projection failed"
            du = abs(proj["u"] - m["u_gt"])
            dv = abs(proj["v"] - m["v_gt"])
            err = max(du, dv)
            if err > max_err:
                max_err = err

        assert max_err <= THRESHOLDS["m2_max_diff"], f"M2 max error {max_err} px exceeds 1e-3 px"


@pytest.mark.render
def test_marker_evaluation_ref_passes_all_thresholds(xvs_server):
    """Execute live M1, M2, M3 evaluation on ref backend and verify all thresholds pass."""
    import tempfile
    server, port = xvs_server
    rig = Rig.load_nominal().make_true(20260922)

    # Test two representative scenes: standard P1_F and near-vertical NV_pitch0.0_U
    scenes = [
        {
            "scene_id": "test_P1_F",
            "group": "standard",
            "pose_name": "P1",
            "camera_id": "F",
            "lat_deg": 47.40,
            "lon_deg": 8.50,
            "h_m": 600.0,
            "yaw_deg": 30.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        {
            "scene_id": "test_NV_U",
            "group": "near_vertical",
            "pose_name": "NV_pitch0.0",
            "camera_id": "U",
            "lat_deg": 37.62,
            "lon_deg": -122.37,
            "h_m": 1000.0,
            "yaw_deg": 40.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with XVSRenderer(port=port, backend="ref", server=server) as renderer:
            renderer.set_profile("markers")
            for sc in scenes:
                res = evaluate_scene(
                    renderer=renderer,
                    rig=rig,
                    scene_spec=sc,
                    out_backend_dir=tmp_path,
                    seed=20260923,
                )

                assert res["m1_pass"] is True, f"M1 failed on {sc['scene_id']}: {res}"
                assert res["m2_pass"] is True, f"M2 failed on {sc['scene_id']}: {res}"
                assert res["m3_pass"] is True, f"M3 failed on {sc['scene_id']}: {res}"
                assert res["all_pass"] is True
                assert res["m1_detected"] == 40
                assert res["m1_abs_mean_du"] <= THRESHOLDS["m1_mean_du"]
                assert res["m1_abs_mean_dv"] <= THRESHOLDS["m1_mean_dv"]
                assert res["m1_median_e"] <= THRESHOLDS["m1_median_e"]
                assert res["m1_max_e"] <= THRESHOLDS["m1_max_e"]


@pytest.mark.render
def test_marker_evaluation_hw_runs(xvs_server):
    """Execute live M1-M3 evaluation on hw (D3D11) backend and verify execution and diagnostics."""
    import tempfile
    server, port = xvs_server
    rig = Rig.load_nominal().make_true(20260922)

    scene = {
        "scene_id": "test_hw_P1_F",
        "group": "standard",
        "pose_name": "P1",
        "camera_id": "F",
        "lat_deg": 47.40,
        "lon_deg": 8.50,
        "h_m": 600.0,
        "yaw_deg": 30.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with XVSRenderer(port=port, backend="hw", server=server) as renderer:
            diag = renderer.get_webgl_diagnostics()
            assert diag["success"] is True
            assert diag["isWebGL2"] is True
            renderer.set_profile("markers")
            res = evaluate_scene(
                renderer=renderer,
                rig=rig,
                scene_spec=scene,
                out_backend_dir=tmp_path,
                seed=20260923,
            )
            assert res["m1_detected"] == 40
            assert res["m2_pass"] is True
            assert res["m3_pass"] is True
            assert res["all_pass"] is True
