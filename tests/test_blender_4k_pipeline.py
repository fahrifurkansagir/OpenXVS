"""Offline verification tests for Aşama 2 4K film trajectory export & Blender 4.x scene script."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "export_4k_film_trajectories.py"
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "blender_openxvs_4k_scene.py"
STAGE_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "build_v45_colab_stages.py"
BLENDER_NOTEBOOK = REPO_ROOT / "tools" / "colab" / "openxvs_v4_5_blender_4k_l4.ipynb"


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_export_4k_film_trajectories_all_4_chapters(tmp_path: Path) -> None:
    mod = _load_module_from_path("export_4k_film_trajectories_test", EXPORT_SCRIPT)
    out_json = tmp_path / "trajectories.json"
    doc = mod.export_all_film_trajectories(out_path=out_json, fps=30)

    assert out_json.is_file()
    assert doc["fps"] == 30
    assert doc["total_frames"] == 1320
    expected_chapters = {
        "ch1_cbdr_activerange": 360,
        "ch2_gps_spoof_nav": 300,
        "ch3_thermal_fog_landing": 300,
        "ch4_forcedland_35kt_ksfo": 360,
    }
    assert set(doc["chapters"].keys()) == set(expected_chapters.keys())

    for cid, exp_n in expected_chapters.items():
        ch = doc["chapters"][cid]
        assert ch["frame_count"] == exp_n
        assert len(ch["frames"]) == exp_n
        for fr in ch["frames"]:
            own = fr["ownship_ned"]
            for k in ("north_m", "east_m", "down_m", "heading_rad", "pitch_rad", "roll_rad", "tas_mps"):
                assert math.isfinite(float(own[k]))

    # Chapter 1: Real ActiveRangeController CBDR lock & safety assertions
    s1 = doc["chapters"]["ch1_cbdr_activerange"]["summary"]
    assert s1["entered_active_swerve"] is True
    assert s1["reached_locked"] is True
    assert s1["initial_sigma_r_m"] > 800.0
    assert s1["locked_sigma_r_m"] < 35.0
    assert s1["locked_rel_err_pct"] < 8.0
    assert s1["min_sc3bf_margin"] >= 0.0
    assert s1["min_separation_m"] >= 152.4

    # Chapter 2: Stealth GPS spoof isolation
    ch2_frames = doc["chapters"]["ch2_gps_spoof_nav"]["frames"]
    assert ch2_frames[0]["telemetry"]["spoof_detected"] is False
    assert ch2_frames[-1]["telemetry"]["spoof_detected"] is True
    assert ch2_frames[-1]["telemetry"]["gps_offset_m"] >= 200.0

    # Chapter 3: Coastal advection fog triggers LWIR primary Simplex mode
    ch3_frames = doc["chapters"]["ch3_thermal_fog_landing"]["frames"]
    assert ch3_frames[0]["telemetry"]["simplex_mode"] == "NOMINAL_FUSION"
    assert ch3_frames[-1]["telemetry"]["simplex_mode"] == "EO_DEGRADED_LWIR_PRIMARY"

    # Chapter 4: Real DR-18 ForcedLand 35 kt crosswind reachability & trochoid waypoints
    ch4 = doc["chapters"]["ch4_forcedland_35kt_ksfo"]
    assert ch4["summary"]["flz_reachable"] is True
    assert ch4["summary"]["energy_margin_m"] > 0.0
    assert len(ch4["corridor_waypoints_ned"]) == 41


def test_blender_scene_ast_and_pure_python_helpers(tmp_path: Path) -> None:
    # 1. Strict AST syntax verification
    src_text = BLENDER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src_text, filename=str(BLENDER_SCRIPT))
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_and_render_chapter" in fn_names
    assert "ned_to_blender_xyz" in fn_names
    assert "compute_cinematic_camera_pose" in fn_names

    # 2. Pure-Python coordinate transform & right-handed orientation checks
    bmod = _load_module_from_path("blender_openxvs_4k_scene_test", BLENDER_SCRIPT)
    bx, by, bz = bmod.ned_to_blender_xyz(100.0, 250.0, -320.0)
    assert (bx, by, bz) == (250.0, 100.0, 320.0)
    with pytest.raises(ValueError):
        bmod.ned_to_blender_xyz(float("nan"), 0.0, 0.0)

    # Attitude check: heading=0 -> rot_z=0; heading=+pi/2 (East) -> rot_z=-pi/2 (CW around +Z from +Y to +X)
    rx, ry, rz = bmod.ned_euler_to_blender_xyz_rad(math.pi / 2.0, 0.1, -0.2)
    assert rx == pytest.approx(0.1)
    assert ry == pytest.approx(-0.2)
    assert rz == pytest.approx(-math.pi / 2.0)

    # Resolution parser & Blender `--` CLI argument separator
    assert bmod.parse_resolution("3840x2160") == (3840, 2160)
    assert bmod.parse_resolution("1920x1080") == (1920, 1080)
    with pytest.raises(ValueError):
        bmod.parse_resolution("invalid")
    cli = bmod.parse_cli_args(["-b", "-P", "scene.py", "--", "--chapter", "ch4_forcedland_35kt_ksfo", "--samples", "16"])
    assert cli.chapter == "ch4_forcedland_35kt_ksfo"
    assert cli.samples == 16

    # 3. Real XVS3DEP1 terrain binary decoding
    pack = bmod.parse_xvs3dep_pack()
    assert pack["width"] == 256 and pack["height"] == 256
    assert pack["extent_m"] == pytest.approx(10000.0)
    heights = pack["heights_msl_m"]
    assert isinstance(heights, np.ndarray) and heights.shape == (256, 256)
    assert bool(np.all(np.isfinite(heights)))
    assert float(heights.max()) > 100.0  # San Bruno Mountain / Marin hills present

    # 4. Dynamic cinematic camera framing across all 4 chapters
    emod = _load_module_from_path("export_4k_film_trajectories_cam", EXPORT_SCRIPT)
    doc = emod.export_all_film_trajectories(out_path=tmp_path / "traj_cam.json", fps=30)
    for cid, ch in doc["chapters"].items():
        for idx in (0, len(ch["frames"]) // 2, len(ch["frames"]) - 1):
            t_norm = float(idx) / float(len(ch["frames"]) - 1)
            pose = bmod.compute_cinematic_camera_pose(cid, ch["frames"][idx], t_norm)
            loc = pose["location"]
            tgt = pose["look_at"]
            assert all(math.isfinite(v) for v in (*loc, *tgt))
            baseline = math.dist(loc, tgt)
            assert baseline > 5.0, f"Camera too close to target in {cid} frame {idx}: {baseline}"
            assert 20.0 <= float(pose["lens_mm"]) <= 85.0


def test_yxz_attitude_math_wnw_bank_vs_pitch() -> None:
    """BUG-1 regression: verify Blender `'YXZ'` order banks wings on WNW heading instead of pitching nose."""
    bmod = _load_module_from_path("blender_openxvs_4k_scene_yxz", BLENDER_SCRIPT)
    assert bmod.AIRCRAFT_ROTATION_MODE == "YXZ"

    src_text = BLENDER_SCRIPT.read_text(encoding="utf-8")
    assert 'rotation_mode = "ZXY"' not in src_text
    assert "rotation_mode = 'ZXY'" not in src_text
    assert src_text.count("rotation_mode = AIRCRAFT_ROTATION_MODE") == 2

    wnw_hdg = math.radians(298.0)
    nose_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_wing_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    left_wing_local = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    # Case A: ch1 FAR 91.113 right bank (+24 deg) on WNW 298 deg heading -> wings bank, nose stays level
    roll_rad = math.radians(24.0)
    rx, ry, rz = bmod.ned_euler_to_blender_xyz_rad(wnw_hdg, 0.0, roll_rad)
    m_bank = bmod.blender_euler_yxz_to_matrix(rx, ry, rz)
    assert np.allclose(m_bank.T @ m_bank, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(m_bank)) == pytest.approx(1.0, abs=1e-12)

    nose_world = m_bank @ nose_local
    right_world = m_bank @ right_wing_local
    left_world = m_bank @ left_wing_local
    assert float(nose_world[2]) == pytest.approx(0.0, abs=1e-12)
    assert float(nose_world[0]) == pytest.approx(math.sin(wnw_hdg), abs=1e-12)
    assert float(nose_world[1]) == pytest.approx(math.cos(wnw_hdg), abs=1e-12)
    assert float(right_world[2]) == pytest.approx(-math.sin(roll_rad), abs=1e-12)
    assert float(left_world[2]) == pytest.approx(math.sin(roll_rad), abs=1e-12)

    # Case B: ch4 steep glide pitch (-21 deg) on WNW 298 deg heading -> nose pitches down, wings stay level
    pitch_rad = math.radians(-21.0)
    rx2, ry2, rz2 = bmod.ned_euler_to_blender_xyz_rad(wnw_hdg, pitch_rad, 0.0)
    m_pitch = bmod.blender_euler_yxz_to_matrix(rx2, ry2, rz2)
    nose_pitch = m_pitch @ nose_local
    right_pitch = m_pitch @ right_wing_local
    assert float(nose_pitch[2]) == pytest.approx(math.sin(pitch_rad), abs=1e-12)
    assert float(right_pitch[2]) == pytest.approx(0.0, abs=1e-12)


def test_heading_unwrap_continuity_across_ch1_and_all_chapters(tmp_path: Path) -> None:
    """BUG-4 regression: `unwrap_headings_rad` eliminates `-2*pi` wrap jump across ch1 frame 0->1."""
    bmod = _load_module_from_path("blender_openxvs_4k_scene_unwrap", BLENDER_SCRIPT)
    emod = _load_module_from_path("export_4k_film_trajectories_unwrap", EXPORT_SCRIPT)

    # Direct synthetic wrap test matching ch1 frame 0->1 (5.20108 -> -1.08210)
    unwrapped = bmod.unwrap_headings_rad([5.20108, -1.08210, -1.08210])
    assert abs(unwrapped[1] - unwrapped[0]) < 1e-4
    assert abs(unwrapped[2] - unwrapped[1]) < 1e-9

    doc = emod.export_all_film_trajectories(out_path=tmp_path / "traj_unwrap.json", fps=30)
    ch1_raw = [float(f["ownship_ned"]["heading_rad"]) for f in doc["chapters"]["ch1_cbdr_activerange"]["frames"]]
    assert abs(ch1_raw[1] - ch1_raw[0]) > 6.0, "Expected raw ch1 frame 0->1 wrap before unwrapping"
    for cid, ch in doc["chapters"].items():
        raw_hdgs = [float(f["ownship_ned"]["heading_rad"]) for f in ch["frames"]]
        cont_hdgs = bmod.unwrap_headings_rad(raw_hdgs)
        diffs = [abs(cont_hdgs[i] - cont_hdgs[i - 1]) for i in range(1, len(cont_hdgs))]
        assert max(diffs) < 0.15, f"Heading discontinuity remained in {cid}: max diff={max(diffs)}"

    # Camera Euler continuity + ch4 engine-out stopped spinners
    src_text = BLENDER_SCRIPT.read_text(encoding="utf-8")
    assert 'to_euler("XYZ", prev_cam_euler)' in src_text
    assert bmod.spinner_angular_step_rad("ch4_forcedland_35kt_ksfo") == 0.0
    assert bmod.spinner_angular_step_rad("ch1_cbdr_activerange") > 1.0


def test_ksfo_ortho_georeference_inner_core2k_blend_and_runway_markings() -> None:
    """BUG-2 & visual polish regression: 1:1 inner_v1 + exact 2x2 km core2k UV footprint + runway markings."""
    bmod = _load_module_from_path("blender_openxvs_4k_scene_ortho", BLENDER_SCRIPT)
    inner_pack = bmod.parse_xvs3dep_pack(ring_name="inner_ksfo")
    outer_pack = bmod.parse_xvs3dep_pack(ring_name="outer_sf_bay")
    assert outer_pack["ring_name"] == "outer_sf_bay"
    assert outer_pack["extent_m"] == pytest.approx(40000.0)

    fp = bmod.compute_core2k_uv_footprint_in_inner(inner_pack)
    # 2 km core inside 10 km inner ring MUST span ~0.20 in U and V (NOT 1.0 = 5x stretch)
    assert fp["u_span"] == pytest.approx(0.20, abs=0.005)
    assert fp["v_span"] == pytest.approx(0.20, abs=0.005)
    assert 0.50 < fp["u_min"] < 0.53 and 0.70 < fp["u_max"] < 0.73
    assert 0.35 < fp["v_min"] < 0.38 and 0.55 < fp["v_max"] < 0.58

    # Core2k center maps to (0.5, 0.5) with full blend weight 1.0
    u_mid = 0.5 * (fp["u_min"] + fp["u_max"])
    v_mid = 0.5 * (fp["v_min"] + fp["v_max"])
    uc_mid, vc_mid = bmod.inner_uv_to_core2k_uv(u_mid, v_mid, fp)
    assert uc_mid == pytest.approx(0.5, abs=1e-9)
    assert vc_mid == pytest.approx(0.5, abs=1e-9)
    assert bmod.core2k_feather_weight(uc_mid, vc_mid) == pytest.approx(1.0)

    # KSFO 28L threshold (E=1505.37, N=-786.28) lands squarely inside core2k with full feather weight
    tx, ty, _ = bmod.ned_to_blender_xyz(*bmod.KSFO_28L_THR_NED)
    u_thr = tx / inner_pack["extent_m"] + 0.5
    v_thr = ty / inner_pack["extent_m"] + 0.5
    uc_thr, vc_thr = bmod.inner_uv_to_core2k_uv(u_thr, v_thr, fp)
    assert 0.60 < uc_thr < 0.75
    assert 0.20 < vc_thr < 0.38
    assert bmod.core2k_feather_weight(uc_thr, vc_thr) == pytest.approx(1.0)

    # Far corner (San Bruno Mtn) outside core2k has zero core2k leakage -> 100% inner_v1
    uc_far, vc_far = bmod.inner_uv_to_core2k_uv(0.10, 0.85, fp)
    assert bmod.core2k_feather_weight(uc_far, vc_far) == 0.0

    # Procedural FAA Part 139 runway markings contrast cleanly against asphalt
    rwy_rgba = bmod.generate_runway_markings_rgba(width=256, height=2048)
    assert rwy_rgba.shape == (2048, 256, 4)
    assert rwy_rgba.dtype == np.float32
    assert float(rwy_rgba.min()) >= 0.0 and float(rwy_rgba.max()) <= 1.0
    # Piano-key threshold & centerline dashes are bright white (> 0.85) while asphalt shoulder is dark (< 0.30)
    assert float(rwy_rgba[30, :, 0].max()) > 0.85
    assert float(rwy_rgba[1024, 128, 0]) > 0.85 or float(rwy_rgba[1000, 128, 0]) > 0.85
    assert float(rwy_rgba[1024, 60, 0]) < 0.30


def test_gpu_failfast_persistent_data_volumetrics_and_visual_polish(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-3 & BUG-5 regression: GPU-only device selection, fail-fast, OIDN GPU, clouds Geometry.Position, holograms."""
    bmod = _load_module_from_path("blender_openxvs_4k_scene_gpu", BLENDER_SCRIPT)

    # CLI flag parity including new --inner-ortho-jpg, --outer-ortho-jpg, --require-gpu
    cli = bmod.parse_cli_args(
        [
            "--",
            "--chapter",
            "ch1_cbdr_activerange",
            "--inner-ortho-jpg",
            "apps/livedeck/public/assets/ksfo_ortho_inner_v1.jpg",
            "--outer-ortho-jpg",
            "apps/livedeck/public/assets/ksfo_ortho_outer_v1.jpg",
            "--require-gpu",
        ]
    )
    assert cli.require_gpu is True
    assert cli.inner_ortho_jpg.name == "ksfo_ortho_inner_v1.jpg"
    assert cli.outer_ortho_jpg.name == "ksfo_ortho_outer_v1.jpg"

    # Source contract checks for BUG-3 clouds & visual polish
    src_text = BLENDER_SCRIPT.read_text(encoding="utf-8")
    assert 'ShaderNodeNewGeometry' in src_text
    assert 'geom.outputs["Position"]' in src_text
    assert "use_persistent_data = True" in src_text
    assert "denoising_use_gpu" in src_text
    assert "volume_bounces = 1" in src_text
    assert "volume_step_rate = 2.0" in src_text
    assert "volume_max_steps = 128" in src_text
    assert "visible_shadow = False" in src_text
    assert "subdivisions=4, radius=152.4" in src_text
    assert "z_offset_m=-4.0" in src_text

    # Mock `bpy` to verify `_configure_cycles_gpu` disables CPU drag & enforces `--require-gpu` fail-fast
    class _FakeDev:
        def __init__(self, dtype: str, name: str) -> None:
            self.type = dtype
            self.name = name
            self.use = True

    def _make_fake_env(devices: list[_FakeDev]) -> tuple[Any, Any]:
        prefs = types.SimpleNamespace(
            compute_device_type="NONE",
            devices=devices,
            get_devices=lambda: None,
        )
        fake_bpy = types.ModuleType("bpy")
        fake_bpy.context = types.SimpleNamespace(  # type: ignore[attr-defined]
            preferences=types.SimpleNamespace(addons={"cycles": types.SimpleNamespace(preferences=prefs)})
        )
        scene = types.SimpleNamespace(
            render=types.SimpleNamespace(
                engine="",
                use_persistent_data=False,
                use_motion_blur=False,
                motion_blur_shutter=0.0,
                resolution_x=0,
                resolution_y=0,
                resolution_percentage=0,
                fps=0,
                image_settings=types.SimpleNamespace(file_format="", color_mode=""),
            ),
            cycles=types.SimpleNamespace(
                samples=0,
                use_adaptive_sampling=False,
                adaptive_threshold=0.0,
                use_denoising=False,
                denoiser="",
                denoising_use_gpu=False,
                max_bounces=0,
                diffuse_bounces=0,
                glossy_bounces=0,
                volume_bounces=0,
                volume_step_rate=0.0,
                volume_max_steps=0,
                transparent_max_bounces=0,
                device="",
            ),
            view_settings=types.SimpleNamespace(view_transform="", look="", exposure=0.0),
        )
        return fake_bpy, scene

    # 1. Mixed GPU + CPU -> GPU enabled, CPU explicitly disabled, persistent_data + GPU OIDN active
    gpu_dev = _FakeDev("OPTIX", "NVIDIA L4")
    cpu_dev = _FakeDev("CPU", "Intel Xeon")
    fake_bpy_gpu, scene_gpu = _make_fake_env([gpu_dev, cpu_dev])
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy_gpu)
    info = bmod._configure_cycles_gpu(scene_gpu, 3840, 2160, 48, 30, require_gpu=True)
    assert info["device"] == "GPU"
    assert info["backend"] == "OPTIX"
    assert gpu_dev.use is True
    assert cpu_dev.use is False
    assert scene_gpu.render.use_persistent_data is True
    assert scene_gpu.cycles.denoising_use_gpu is True
    assert scene_gpu.cycles.volume_bounces == 1
    assert scene_gpu.cycles.volume_step_rate == pytest.approx(2.0)
    assert scene_gpu.cycles.volume_max_steps == 128

    # 2. CPU-only with require_gpu=True -> MUST fail fast (never burn 50+ hrs silently on CPU)
    fake_bpy_cpu, scene_cpu = _make_fake_env([_FakeDev("CPU", "Intel Xeon")])
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy_cpu)
    with pytest.raises(RuntimeError, match="--require-gpu"):
        bmod._configure_cycles_gpu(scene_cpu, 3840, 2160, 48, 30, require_gpu=True)


def test_blender_4k_notebook_crash_telemetry_resume_and_stage_ortho_tiers(tmp_path: Path) -> None:
    """BUG-2 & BUG-5 regression: notebook crash logs, resume, ffmpeg hardening, and 3-tier ortho stage."""
    if not BLENDER_NOTEBOOK.is_file():
        pytest.skip(f"Clean-room checkout omits optional notebook: {BLENDER_NOTEBOOK.name}")
    nb = json.loads(BLENDER_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 2
    cell0, cell1 = code_cells

    # AST validity of both cells
    ast.parse(cell0, filename="cell0")
    ast.parse(cell1, filename="cell1")

    # Cell 0: urllib3-safe seekable zstd shim + fonts-dejavu-core + headless Blender lib fallback
    assert "import urllib3.response" in cell0
    assert cell0.index("import urllib3.response") < cell0.index("sys.modules['compression.zstd']")
    assert "zstd_mod.ZstdError" in cell0
    assert "dctx.copy_stream(" in cell0
    assert "fonts-dejavu-core" in cell0
    assert "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" in cell0
    assert "libxrender1" in cell0

    # Cell 1: crash-safe flushed+fsynced error logs, run_status heartbeat, per-chapter resume, ffmpeg fixes, unassign
    for token in (
        "traceback.print_exc()",
        "blender_error.log",
        "run_status.json",
        "os.fsync(",
        "os.sync()",
        "_try_resume_chapter",
        "_sync_file_to_drive",
        "--require-gpu",
        "--inner-ortho-jpg",
        "--outer-ortho-jpg",
        "ksfo_ortho_inner_v1.jpg",
        "ksfo_ortho_outer_v1.jpg",
        "ksfo_ortho_core2k_v2.jpg",
        "'-start_number', '1'",
        "'-frames:v', str(expected_frames)",
        "fontfile=",
        "REAL AOT PROOF",
        "safe_colab_unassign(AUTO_UNASSIGN_ON_EXIT)",
    ):
        assert token in cell1, f"Missing Wave E notebook hardening token: {token!r}"
    assert cell1.rindex("os.sync()") < cell1.rindex("safe_colab_unassign(AUTO_UNASSIGN_ON_EXIT)")

    # Stage builder packs all 3 ortho tiers into blender stage ONLY (never touches openxvs_v45_stage.tar.zst)
    smod = _load_module_from_path("build_v45_colab_stages_test", STAGE_SCRIPT)
    man = smod.build_v45_blender_4k_stage(out_dir=tmp_path, include_aot_mp4=False)
    packed_paths = {rec["path"] for rec in man["files"]}
    for req in (
        "apps/livedeck/public/assets/ksfo_ortho_core2k_v2.jpg",
        "apps/livedeck/public/assets/ksfo_ortho_inner_v1.jpg",
        "apps/livedeck/public/assets/ksfo_ortho_outer_v1.jpg",
        "apps/livedeck/public/assets/ksfo_ortho_v1.manifest.json",
        "scripts/xvs/blender_openxvs_4k_scene.py",
    ):
        assert req in packed_paths, f"Missing required blender stage asset: {req}"
    assert not (tmp_path / "openxvs_v45_stage.tar.zst").exists()
