"""Tests for Colab Notebooks 2 & 3 hardening, XVS-World SSIM+Adam VRAM probe & chunked eval, and XVS-D2 harvester."""

from __future__ import annotations

import ast
from compression import zstd
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "xvs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import harvest_colab_g4_d2_release as harvest_mod
from xvs.v2.colab_g4_utils import snapshot_eager_state
from xvs.v2.xvs_world.config import ksfo_28l_default_config
import xvs.v2.xvs_world.train_g4 as world_mod
from xvs.v2.xvs_world.train_g4 import (
    GaussianScene3D,
    _eval_views_chunked,
    _probe_world_vram_batch,
    build_ksfo_approach_cameras,
    render_gaussians_pytorch,
)

NB01_PATH = REPO_ROOT / "tools" / "colab" / "v2_m2_01_xvs_d2_aot_g4.ipynb"
NB02_PATH = REPO_ROOT / "tools" / "colab" / "v2_m2_02_xvs_nav_land_thermal_g4.ipynb"
NB03_PATH = REPO_ROOT / "tools" / "colab" / "v2_m2_03_xvs_world_1km_3dgs_g4.ipynb"
WEB_ONNX_PATH = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"
LIVE_REGISTRY_PATH = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "registry.json"


def test_colab_notebooks_2_and_3_high_ram_metadata_ast_and_vram_cleanup() -> None:
    """Verify Notebooks 2 (G4 parallel/L4 seq) & 3 (L4) set machine_shape='hm', parse via ast, and NB2 frees VRAM."""
    expected_gpu_types = {NB02_PATH: "G4", NB03_PATH: "L4"}
    for nb_path, expected_gpu in expected_gpu_types.items():
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert int(nb["nbformat_minor"]) >= 5
        assert nb["metadata"]["colab"]["gpuType"] == expected_gpu
        assert nb["metadata"]["colab"]["machine_shape"] == "hm"
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
                ast.parse(src, filename=f"{nb_path.name}#cell{idx}")

    nb2 = json.loads(NB02_PATH.read_text(encoding="utf-8"))
    nb2_all = "\n".join("".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code")
    assert "PARALLEL_3_MODELS = True" in nb2_all
    assert "TOTAL_VRAM_GIB >= 45.0" in nb2_all
    assert "set_per_process_memory_fraction" in nb2_all
    cell3 = "".join(nb2["cells"][3]["source"])
    assert "gc.collect()" in cell3
    assert "torch._dynamo.reset()" in cell3
    assert "torch.cuda.empty_cache()" in cell3
    # Ensure cleanup executes between Nav -> Land and Land -> Thermal call sites
    idx_nav = cell3.index("run_xvs_nav_g4_training(")
    idx_land = cell3.index("run_xvs_land_g4_training(")
    idx_therm = cell3.index("run_xvs_thermal_g4_training(")
    assert "_release_gpu_between_models()" in cell3[idx_nav:idx_land]
    assert "_release_gpu_between_models()" in cell3[idx_land:idx_therm]


def test_xvs_world_ssim_adam_vram_probe_restores_weights_and_eval_chunked_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify XVS-World probe runs full L1+SSIM+Adam with >=12 GiB headroom & restores weights; chunked eval exact."""
    world_cfg = ksfo_28l_default_config()
    device = torch.device("cpu")

    rng = np.random.default_rng(20261121)
    means = torch.tensor(rng.uniform(-50.0, 50.0, size=(32, 3)), dtype=torch.float32)
    albedo = torch.tensor(rng.uniform(0.15, 0.85, size=(32, 3)), dtype=torch.float32)

    gt_scene = GaussianScene3D(means, albedo, init_scale_m=6.0, color_perturbation_std=0.0, seed=1).to(device)
    scene = GaussianScene3D(means, albedo, init_scale_m=6.0, color_perturbation_std=0.2, seed=2).to(device)
    views = build_ksfo_approach_cameras(num_views=6, width=32, height=24)

    with torch.no_grad():
        gt_rgbs = [
            render_gaussians_pytorch(
                gt_scene.means, gt_scene.scales(), gt_scene.opacities(), gt_scene.rgb_colors(), v
            )[0].detach()
            for v in views
        ]

    ssim_calls = 0
    orig_ssim = world_mod._compute_ssim_loss

    def _spy_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nonlocal ssim_calls
        ssim_calls += 1
        return orig_ssim(pred, target)

    monkeypatch.setattr(world_mod, "_compute_ssim_loss", _spy_ssim)

    snap_before = snapshot_eager_state(scene)
    probe_report = _probe_world_vram_batch(
        scene,
        views,
        gt_rgbs,
        world_cfg,
        use_gsplat=False,
        probe_candidates=[1, 2],
        device=device,
        target_frac=0.86,
        vram_headroom_gib=8.0,
    )

    assert ssim_calls == 2
    assert float(probe_report["safety_headroom_gib"]) >= 12.0
    assert int(probe_report["selected_batch_size"]) == 2
    for key, orig in snap_before.items():
        assert torch.equal(scene.state_dict()[key], orig), f"Scene weight drift on {key}"
    for param in scene.parameters():
        assert param.grad is None

    # Chunked eval (chunk_size=2) must match unchunked eval (chunk_size=6) accurately
    l1_chunked, psnr_chunked = _eval_views_chunked(
        scene, views, gt_rgbs, world_cfg, use_gsplat=False, chunk_size=2
    )
    l1_full, psnr_full = _eval_views_chunked(
        scene, views, gt_rgbs, world_cfg, use_gsplat=False, chunk_size=6
    )
    assert l1_chunked == pytest.approx(l1_full, abs=1e-6)
    assert psnr_chunked == pytest.approx(psnr_full, abs=1e-5)


def _write_tar_zst(archive_path: Path, files: dict[str, bytes]) -> None:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:") as tf:
        for rel_name, payload in files.items():
            info = tarfile.TarInfo(name=rel_name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    archive_path.write_bytes(zstd.compress(tar_buf.getvalue()))


def test_harvest_colab_g4_d2_release_full_partial_and_traversal_guard(tmp_path: Path) -> None:
    """Verify full 4-tier harvest + Live Deck hot-swap, partial teacher-only harvest, and traversal block."""
    assert WEB_ONNX_PATH.is_file(), f"Missing ONNX model: {WEB_ONNX_PATH}"
    assert LIVE_REGISTRY_PATH.is_file(), f"Missing Live Deck registry: {LIVE_REGISTRY_PATH}"

    onnx_bytes = WEB_ONNX_PATH.read_bytes()
    dummy_pt_buf = io.BytesIO()
    torch.save({"w": torch.tensor([1.0, 2.0])}, dummy_pt_buf)
    dummy_pt_bytes = dummy_pt_buf.getvalue()

    env_rep = {"gpu_name": "NVIDIA RTX PRO 6000 Blackwell", "total_vram_gib": 94.97, "is_blackwell_sm120": True}
    eval_metrics = {
        "teacher": {
            "pd_at_fpi_0p1": 0.91,
            "pd_at_fpi_0p5": 0.95,
            "pd_at_fpi_1p0": 0.97,
            "ap50": 0.93,
            "centroid_rmse_px": 1.12,
            "above_horizon_pd": 0.96,
            "below_horizon_pd": 0.89,
        },
        "web": {
            "pd_at_fpi_0p1": 0.86,
            "pd_at_fpi_0p5": 0.91,
            "pd_at_fpi_1p0": 0.94,
            "ap50": 0.89,
            "centroid_rmse_px": 1.45,
            "above_horizon_pd": 0.93,
            "below_horizon_pd": 0.84,
        },
    }
    train_summary = {
        "schema": "xvs.v2_m2.xvs_d2_g4_train_summary/1",
        "artifact_tag": "COLAB_G4_GPU_ARTIFACT",
        "env_report": env_rep,
        "eval_metrics": eval_metrics,
        "pass": True,
    }

    # 1. Full 4-tier archive + isolated Live Deck hot-swap
    full_arc = tmp_path / "v2_m2_01_xvs_d2_aot_g4_results_stage.tar.zst"
    _write_tar_zst(
        full_arc,
        {
            "checkpoints/xvs_d2_teacher.pt": dummy_pt_bytes,
            "checkpoints/xvs_d2_edge.pt": dummy_pt_bytes,
            "checkpoints/xvs_d2_edge_slim.pt": dummy_pt_bytes,
            "checkpoints/xvs_d2_web.pt": dummy_pt_bytes,
            "onnx/xvs_d2_web_fp16.onnx": onnx_bytes,
            "training_summary.json": json.dumps(train_summary).encode("utf-8"),
            "eval_metrics.json": json.dumps(eval_metrics).encode("utf-8"),
            "blackwell_env_report.json": json.dumps(env_rep).encode("utf-8"),
        },
    )

    iso_models = tmp_path / "iso_livedeck_models"
    iso_models.mkdir()
    shutil.copy2(LIVE_REGISTRY_PATH, iso_models / "registry.json")

    report_md = tmp_path / "harvest_full" / "release_report.md"
    res_full = harvest_mod.harvest_colab_g4_d2_release(
        archive_path=full_arc,
        out_dir=tmp_path / "harvest_full",
        install_livedeck=True,
        livedeck_models_dir=iso_models,
        report_md=report_md,
    )
    assert res_full["completeness"] == "FULL_4_TIER_COMPLETE"
    assert set(res_full["checkpoints"].keys()) == {"teacher", "edge", "edge_slim", "web"}
    assert res_full["onnx_audit"]["pass"] is True
    assert res_full["onnx_audit"]["forbidden_ops"] == []
    assert res_full["onnx_audit"]["all_outputs_finite"] is True
    assert res_full["livedeck_install"]["registry_updated"] is True
    assert (iso_models / "xvs_d2_web_fp16.onnx").is_file()
    assert report_md.is_file()
    md_text = report_md.read_text(encoding="utf-8")
    assert "[COLAB_G4_GPU_ARTIFACT]" in md_text
    assert "FULL_4_TIER_COMPLETE" in md_text
    assert "0.9300" in md_text

    # 2. Partial teacher-only archive gracefully summarizes without crashing
    partial_arc = tmp_path / "partial_teacher.tar.zst"
    _write_tar_zst(
        partial_arc,
        {"checkpoints/xvs_d2_teacher.pt": dummy_pt_bytes},
    )
    res_partial = harvest_mod.harvest_colab_g4_d2_release(
        archive_path=partial_arc,
        out_dir=tmp_path / "harvest_partial",
        install_livedeck=False,
    )
    assert res_partial["completeness"] == "PARTIAL_TEACHER_ONLY"
    assert set(res_partial["checkpoints"].keys()) == {"teacher"}
    assert Path(res_partial["report_md"]).is_file()

    # 3. Path traversal member is strictly rejected fail-closed
    evil_arc = tmp_path / "evil_traversal.tar.zst"
    _write_tar_zst(evil_arc, {"../escape_pwn.txt": b"nope"})
    with pytest.raises(ValueError, match="traversal"):
        harvest_mod.safe_extract_tar_zst(evil_arc, tmp_path / "evil_out")


def test_colab_zstd_shim_urllib3_safety_and_nb3_gsplat_no_source_build() -> None:
    """Verify all 3 Colab G4 notebooks guard Python 3.13 urllib3/ZstdError and NB3 never builds gsplat from git."""
    for nb_path in (NB01_PATH, NB02_PATH, NB03_PATH):
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        code_cells: list[str] = []
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
                ast.parse(src, filename=f"{nb_path.name}#cell{idx}")
                code_cells.append(src)
        joined = "\n".join(code_cells)
        assert "def ensure_colab_compression_zstd_shim(" in joined, f"Missing shim fn in {nb_path.name}"
        idx_urllib3 = joined.index("import urllib3.response")
        idx_reg = joined.index("sys.modules['compression.zstd']")
        assert idx_urllib3 < idx_reg, f"urllib3.response must be pre-imported before shim registration in {nb_path.name}"
        assert ".ZstdError = _zstd_compat.ZstdError" in joined, f"Missing ZstdError binding in {nb_path.name}"

    nb3 = json.loads(NB03_PATH.read_text(encoding="utf-8"))
    nb3_code = "\n".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else str(c["source"]))
        for c in nb3["cells"]
        if c["cell_type"] == "code"
    )
    assert "git+https://" not in nb3_code
    assert "--no-build-isolation" not in nb3_code
    assert "INSTALL_GSPLAT = False" in nb3_code
    assert "--only-binary=:all:" in nb3_code
    assert "timeout=90" in nb3_code
    assert "timeout=120" in nb3_code
    assert "MAX_STEPS = min(MAX_STEPS, 500)" in nb3_code
    assert "GRID_RES = min(GRID_RES, 96)" in nb3_code
    assert "IMAGE_WIDTH = 512" in nb3_code
    assert "IMAGE_HEIGHT = 384" in nb3_code
    assert "NUM_VIEWS = 24" in nb3_code
    assert "GRID_RES = 128" in nb3_code


def test_xvs_world_checkpointed_backward_parity_and_bundled_ksfo_ortho_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify gradient-checkpointed render_gaussians_pytorch backward grads + <1e-6 parity & bundled KSFO ortho."""
    import v2_m2_stage as stage_mod

    assert "apps/livedeck/public/assets/ksfo_ortho_inner_v1.jpg" in stage_mod.G4_FULL_STAGE_PATHS
    assert "apps/livedeck/public/assets/ksfo_ortho_v1.manifest.json" in stage_mod.G4_FULL_STAGE_PATHS

    # 1. Gradient-checkpointed backward pass produces finite non-zero grads & matches no_grad forward < 1e-6
    rng = np.random.default_rng(20261121)
    means = torch.tensor(rng.uniform(-45.0, 45.0, size=(160, 3)), dtype=torch.float32)
    albedo = torch.tensor(rng.uniform(0.15, 0.85, size=(160, 3)), dtype=torch.float32)
    scene = GaussianScene3D(means, albedo, init_scale_m=18.0, color_perturbation_std=0.1, seed=7)
    view = build_ksfo_approach_cameras(num_views=4, width=64, height=48)[0]

    with torch.no_grad():
        rgb_ref, depth_ref = render_gaussians_pytorch(
            scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), view
        )

    rgb_ckpt, depth_ckpt = render_gaussians_pytorch(
        scene.means, scene.scales(), scene.opacities(), scene.rgb_colors(), view
    )
    assert float((rgb_ckpt.detach() - rgb_ref).abs().max().item()) < 1e-6
    assert float((depth_ckpt.detach() - depth_ref).abs().max().item()) < 1e-5

    loss = rgb_ckpt.mean()
    loss.backward()
    for param_name in ("means", "log_scales", "logit_opacities", "logit_colors"):
        grad = getattr(scene, param_name).grad
        assert grad is not None, f"Missing gradient on {param_name}"
        assert bool(torch.isfinite(grad).all().item()), f"Non-finite gradient on {param_name}"
        assert float(grad.abs().sum().item()) > 0.0, f"Zero gradient on {param_name}"

    # 2. Bundled public-domain USDA/USGS KSFO ortho fallback triggers when NAIP STAC fails/returns None
    monkeypatch.setattr(
        world_mod,
        "_fetch_optional_naip_albedo",
        lambda cfg, grid_res: (None, "procedural_ksfo_fallback_stac_error:SimulatedTimeout"),
    )
    patch = world_mod.load_ksfo_3dep_patch_ned(
        ksfo_28l_default_config(),
        grid_res=16,
        use_naip_stac=True,
    )
    assert patch["albedo_source"] == "bundled_usda_usgs_ksfo_ortho_v1"
    albedo_arr = patch["albedo_rgb"]
    assert albedo_arr.shape == (16, 16, 3)
    assert bool(np.all(np.isfinite(albedo_arr)))
    assert float(albedo_arr.std()) > 0.01


