"""Offline verification tests for OpenXVS v4.5 real-world trainers (`XVS-ThermalNet` & `XVS-Land`) and Colab L4 notebook."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from xvs.v2.xvs_land.train_g4 import (
    XvsLandG4Config,
    run_xvs_land_g4_training,
)
from xvs.v2.xvs_thermal.train_g4 import (
    XvsThermalG4Config,
    _safe_extract_zip,
    discover_real_thermal_samples,
    run_xvs_thermal_g4_training,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LARD_MANIFEST = REPO_ROOT / "data" / "reference" / "lard_real_airports_manifest.json"
RUNWAYS_CSV = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"
NOTEBOOK_PATH = REPO_ROOT / "tools" / "colab" / "openxvs_v4_5_realworld_finetune_l4.ipynb"


def _write_gray_png(path: Path, hw: tuple[int, int] = (48, 64), *, hot_box: tuple[int, int, int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(hw, 64, dtype=np.uint8)
    if hot_box is not None:
        x1, y1, x2, y2 = hot_box
        arr[y1:y2, x1:x2] = 240
    Image.fromarray(arr, mode="L").save(path)


def _create_mock_nested_thermal_cache(cache_dir: Path) -> None:
    """Create nested `.zip` archives mimicking real IRSTD-1k (`MIT`) and HIT-UAV (`CC-BY-4.0` YOLO + COCO JSON)."""
    staging = cache_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    # 1. Nested IRSTD-1k zip (inside `irstd1k/IRSTD1k_nested.zip`)
    irstd_zip = cache_dir / "irstd1k" / "IRSTD1k_nested.zip"
    irstd_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(irstd_zip, "w") as zf:
        for i in range(4):
            img_p = staging / f"irstd_img_{i}.png"
            msk_p = staging / f"irstd_msk_{i}.png"
            _write_gray_png(img_p, (48, 64), hot_box=(20 + i, 14 + i, 28 + i, 22 + i))
            _write_gray_png(msk_p, (48, 64), hot_box=(20 + i, 14 + i, 28 + i, 22 + i))
            zf.write(img_p, f"IRSTD-1k/IRSTD1k_Img/XDU{i}.png")
            zf.write(msk_p, f"IRSTD-1k/IRSTD1k_Label/XDU{i}.png")

    # 2. Nested HIT-UAV zip with BOTH `dataset/images/train` (YOLO) and `normal_json/train.json` (COCO)
    hituav_zip = cache_dir / "hit_uav" / "HIT_UAV_nested.zip"
    hituav_zip.parent.mkdir(parents=True, exist_ok=True)
    coco_images = []
    coco_anns = []
    with zipfile.ZipFile(hituav_zip, "w") as zf:
        for i in range(4):
            img_p = staging / f"hit_img_{i}.jpg"
            _write_gray_png(img_p, (48, 64), hot_box=(10 + 3 * i, 12, 22 + 3 * i, 24))
            zf.write(img_p, f"HIT-UAV/dataset/images/train/hit_{i:03d}.jpg")
            if i < 2:
                # YOLO label format
                zf.writestr(f"HIT-UAV/dataset/labels/train/hit_{i:03d}.txt", "0 0.30 0.35 0.18 0.22\n")
            else:
                # COCO normal_json format (for samples without .txt)
                coco_images.append({"id": i, "file_name": f"hit_{i:03d}.jpg", "width": 64, "height": 48})
                coco_anns.append({"id": i, "image_id": i, "bbox": [10 + 3 * i, 12, 12, 12]})
        zf.writestr(
            "HIT-UAV/dataset/normal_json/train.json",
            json.dumps({"images": coco_images, "annotations": coco_anns}),
        )

    # Clean up staging files so discovery MUST unpack the nested zips
    for p in sorted(staging.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    staging.rmdir()


def test_nested_zip_discovery_loads_both_irstd1k_and_hituav(tmp_path: Path) -> None:
    """`discover_real_thermal_samples` recursively unpacks nested zips and loads both IRSTD-1k & HIT-UAV."""
    cache_dir = tmp_path / "thermal_cache"
    _create_mock_nested_thermal_cache(cache_dir)

    samples = discover_real_thermal_samples(cache_dir, (32, 40), max_samples=16)
    assert len(samples) == 8
    ds_ids = {s.dataset_id for s in samples}
    assert ds_ids == {"irstd1k", "hit_uav"}
    for s in samples:
        assert tuple(s.lwir_2ch.shape) == (2, 32, 40)
        assert tuple(s.mask.shape) == (1, 32, 40)
        assert torch.isfinite(s.lwir_2ch).all()
        assert float(s.mask.sum().item()) > 0.0, f"Expected non-empty target mask for {s.dataset_id}/{s.sample_id}"


def test_zip_slip_blocked_in_nested_and_top_level_archives(tmp_path: Path) -> None:
    """Zip-Slip path traversal (`../evil.txt` and absolute paths) MUST fail closed with ValueError."""
    evil_zip = tmp_path / "evil_traversal.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    with pytest.raises(ValueError, match="Zip-Slip"):
        _safe_extract_zip(evil_zip, tmp_path / "dest_a")

    abs_zip = tmp_path / "evil_abs.zip"
    with zipfile.ZipFile(abs_zip, "w") as zf:
        zf.writestr("/etc/passwd", "root")
    with pytest.raises(ValueError, match="Zip-Slip"):
        _safe_extract_zip(abs_zip, tmp_path / "dest_b")

    # Nested Zip-Slip inside cache_dir must also fail closed during discovery
    nested_cache = tmp_path / "nested_evil_cache"
    nested_cache.mkdir()
    nested_evil = nested_cache / "payload.zip"
    with zipfile.ZipFile(nested_evil, "w") as zf:
        zf.writestr("ok/../../escape.txt", "bad")
    with pytest.raises(ValueError, match="Zip-Slip"):
        discover_real_thermal_samples(nested_cache, (32, 40), max_samples=8)


def test_require_real_thermal_fails_closed_on_empty_cache(tmp_path: Path) -> None:
    """`--require-real-thermal` raises RuntimeError when `< min_real_thermal_samples` are discovered."""
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    cfg = XvsThermalG4Config(
        out_dir=tmp_path / "out_fail",
        data_cache_dir=empty_cache,
        smoke_local=True,
        require_real_thermal=True,
        min_real_thermal_samples=4,
        run_vraim=False,
    )
    with pytest.raises(RuntimeError, match="require_real_thermal"):
        run_xvs_thermal_g4_training(cfg)


def test_xvs_thermal_smoke_local_with_real_cache_and_onnx_parity(tmp_path: Path) -> None:
    """Upgraded XVS-ThermalNet loads >= 50% real thermal mix, keeps dual_blind vacuity=1.0, and exports verified ONNX."""
    cache_dir = tmp_path / "thermal_cache"
    _create_mock_nested_thermal_cache(cache_dir)
    out_dir = tmp_path / "thermal_out"

    cfg = XvsThermalG4Config(
        out_dir=out_dir,
        data_cache_dir=cache_dir,
        smoke_local=True,
        require_real_thermal=True,
        min_real_thermal_samples=6,
        export_onnx=True,
        run_vraim=True,
        seed=20261122,
    )
    rep = run_xvs_thermal_g4_training(cfg)
    assert rep["pass"] is True
    assert int(rep["real_thermal_samples_loaded"]) == 8
    assert float(rep["real_item_fraction"]) >= 0.50
    assert rep["four_mode_eval"]["dual_blind_vacuity_exact_one"] is True
    assert rep["four_mode_eval"]["four_mode_simplex_pass"] is True
    assert rep["vraim_easa_rlm_07_11_pass"] is True

    # Verify finite checkpoint tensors
    ckpt = torch.load(out_dir / "xvs_thermal_final.pt", map_location="cpu", weights_only=False)
    for k, v in ckpt["model_state_dict"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"Non-finite thermal tensor: {k}"

    # Verify ONNX export + ORT numerical parity < 1e-3
    onnx_meta = rep["onnx_export"]
    assert onnx_meta is not None
    assert Path(str(onnx_meta["path"])).is_file()
    assert onnx_meta["ort_verified"] is True
    assert float(onnx_meta["max_abs_diff"]) < 1.0e-3


def test_xvs_land_real_world_scale_smoke_local_and_onnx_parity(tmp_path: Path) -> None:
    """Upgraded XVS-Land (`--real-world-scale`) preserves BC-G4-M2 LARD guard, 0 HMI, and verified ONNX parity."""
    out_dir = tmp_path / "land_out"
    cfg = XvsLandG4Config(
        out_dir=out_dir,
        lard_manifest=LARD_MANIFEST,
        runways_csv=RUNWAYS_CSV,
        smoke_local=True,
        real_world_scale=True,
        use_naip_stac=False,
        export_onnx=True,
        seed=20261122,
    )
    rep = run_xvs_land_g4_training(cfg)
    assert rep["pass"] is True
    assert rep["real_world_scale"] is True
    assert rep["lard_guard"]["manifest_verified"] is True
    assert rep["lard_guard"]["ksfo_heldout"] is True
    assert int(rep["lard_guard"]["lard_real_count"]) >= 36
    assert rep["eval_metrics"]["ksfo_anti_misassociation_pass"] is True
    assert int(rep["eval_metrics"]["hmi_count"]) == 0

    # Verify finite checkpoint tensors
    ckpt = torch.load(out_dir / "xvs_land_final.pt", map_location="cpu", weights_only=False)
    for k, v in ckpt["model_state_dict"].items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"Non-finite land tensor: {k}"

    # Verify ONNX export + ORT numerical parity < 1e-3
    onnx_meta = rep["onnx_export"]
    assert onnx_meta is not None
    assert Path(str(onnx_meta["path"])).is_file()
    assert onnx_meta["ort_verified"] is True
    assert float(onnx_meta["max_abs_diff"]) < 1.0e-3


def test_colab_l4_v45_notebook_json_and_isolation_contract() -> None:
    """Validate `openxvs_v4_5_realworld_finetune_l4.ipynb` JSON structure, L4 metadata, and isolation locks."""
    assert NOTEBOOK_PATH.is_file()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert int(nb["nbformat"]) == 4
    assert nb["metadata"]["colab"]["gpuType"] == "L4"
    full_src = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "v45_realworld_land_thermal_l4" in full_src
    assert "/content/drive/MyDrive/xvs_v45_l4_runs" in full_src
    assert "require_real_thermal" in full_src
    assert "real_world_scale=True" in full_src
    assert "use_naip_stac=USE_NAIP_STAC" in full_src
    assert "safe_colab_unassign(AUTO_UNASSIGN_ON_EXIT)" in full_src
    assert "verify_results_archive" in full_src


def test_blender_4k_colab_l4_notebook_cli_parity_and_isolation() -> None:
    """Validate `openxvs_v4_5_blender_4k_l4.ipynb` JSON, isolation locks, and 1:1 CLI parity with `blender_openxvs_4k_scene.py`."""
    import sys

    nb_path = REPO_ROOT / "tools" / "colab" / "openxvs_v4_5_blender_4k_l4.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert int(nb["nbformat"]) == 4
    assert nb["metadata"]["colab"]["gpuType"] == "L4"
    full_src = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "v45_blender_4k_keynote_l4" in full_src
    assert "/content/drive/MyDrive/xvs_v45_l4_runs" in full_src
    assert "safe_colab_unassign(AUTO_UNASSIGN_ON_EXIT)" in full_src
    assert "openxvs_v4_5_keynote_4k.mp4" in full_src
    assert "openxvs_v4_5_keynote_1080p.mp4" in full_src
    assert "ksfo_3dep_geoid18_v1.bin" in full_src
    assert "ksfo_ortho_core2k_v2.jpg" in full_src
    for ch_id in (
        "ch1_cbdr_activerange",
        "ch2_gps_spoof_nav",
        "ch3_thermal_fog_landing",
        "ch4_forcedland_35kt_ksfo",
    ):
        assert ch_id in full_src

    # Verify exact CLI argument parity by feeding the notebook's exact flags into `blender_openxvs_4k_scene.parse_cli_args`
    scripts_xvs = str(REPO_ROOT / "scripts" / "xvs")
    if scripts_xvs not in sys.path:
        sys.path.insert(0, scripts_xvs)
    import blender_openxvs_4k_scene as bscene  # type: ignore[import-not-found]

    parsed = bscene.parse_cli_args(
        [
            "--",
            "--trajectories",
            "outputs/film_4k/trajectories.json",
            "--out-dir",
            "/content/out_4k",
            "--chapter",
            "ch1_cbdr_activerange",
            "--res",
            "3840x2160",
            "--samples",
            "48",
            "--fps",
            "30",
            "--terrain-bin",
            "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin",
            "--ortho-jpg",
            "apps/livedeck/public/assets/ksfo_ortho_core2k_v2.jpg",
        ]
    )
    assert parsed.res == "3840x2160"
    assert int(parsed.samples) == 48
    assert int(parsed.fps) == 30
    assert parsed.chapter == "ch1_cbdr_activerange"


def test_build_v45_colab_stages_creates_verified_archives_and_manifests(tmp_path: Path) -> None:
    """`build_v45_colab_stages.py` builds both `.tar.zst` archives with verified SHA-256 manifests and required paths."""
    from compression import zstd
    import hashlib
    import sys
    import tarfile

    scripts_xvs = str(REPO_ROOT / "scripts" / "xvs")
    if scripts_xvs not in sys.path:
        sys.path.insert(0, scripts_xvs)
    import build_v45_colab_stages as stage_mod  # type: ignore[import-not-found]

    res = stage_mod.build_all_v45_colab_stages(out_dir=tmp_path, include_aot_mp4=False)
    for key, req_members in (
        (
            "finetune_stage",
            (
                "src/xvs/v2/xvs_thermal/train_g4.py",
                "src/xvs/v2/xvs_land/train_g4.py",
                "scripts/xvs/v2_m2_stage.py",
                "data/reference/lard_real_airports_manifest.json",
                "data/reference/xvs_land_airport_splits.json",
                "data/reference/ourairports/runways.csv",
            ),
        ),
        (
            "blender_4k_stage",
            (
                "scripts/xvs/blender_openxvs_4k_scene.py",
                "outputs/film_4k/trajectories.json",
                "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin",
                "apps/livedeck/public/assets/ksfo_ortho_core2k_v2.jpg",
                "src/xvs/v2/colab_g4_utils.py",
            ),
        ),
    ):
        man = res[key]
        arch_p = tmp_path / str(man["archive_filename"])
        assert arch_p.is_file()
        assert hashlib.sha256(arch_p.read_bytes()).hexdigest() == str(man["archive_sha256"])
        with zstd.open(arch_p, "rb") as zf, tarfile.open(fileobj=zf, mode="r:*") as tf:
            names = set(tf.getnames())
        for req in req_members:
            assert req in names, f"Missing {req!r} in {arch_p.name}"


def test_both_v45_colab_notebooks_imports_and_config_kwargs_exist() -> None:
    """AST-parse BOTH v4.5 Colab L4 notebooks verifying all repo imports exist and config kwargs match dataclasses."""
    import ast
    from dataclasses import fields
    import importlib
    import sys

    scripts_xvs = str(REPO_ROOT / "scripts" / "xvs")
    if scripts_xvs not in sys.path:
        sys.path.insert(0, scripts_xvs)

    config_field_map = {
        "XvsThermalG4Config": {f.name for f in fields(XvsThermalG4Config)},
        "XvsLandG4Config": {f.name for f in fields(XvsLandG4Config)},
    }
    nb_paths = (
        REPO_ROOT / "tools" / "colab" / "openxvs_v4_5_realworld_finetune_l4.ipynb",
        REPO_ROOT / "tools" / "colab" / "openxvs_v4_5_blender_4k_l4.ipynb",
    )
    checked_imports = 0
    checked_configs = 0
    for nb_p in nb_paths:
        nb = json.loads(nb_p.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            tree = ast.parse(src, filename=nb_p.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod_name = node.module
                    if mod_name == "v2_m2_stage" or mod_name.startswith("xvs."):
                        mod = importlib.import_module(mod_name)
                        for alias in node.names:
                            assert hasattr(mod, alias.name), (
                                f"{nb_p.name} imports missing symbol {alias.name!r} from {mod_name!r}"
                            )
                            checked_imports += 1
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in config_field_map:
                        valid_fields = config_field_map[node.func.id]
                        for kw in node.keywords:
                            assert kw.arg is not None and kw.arg in valid_fields, (
                                f"{nb_p.name} passes unknown kwarg {kw.arg!r} to {node.func.id}"
                            )
                        checked_configs += 1
    assert checked_imports >= 6
    assert checked_configs == 2


def test_colab_l4_v45_notebook_cell2_xvs_land_config_contract_and_error_guards(tmp_path: Path) -> None:
    """Assert Wave C Part 2 toggles, crash-safe Drive error logging, and exact Cell 2 XvsLandG4Config contract."""
    import ast
    import inspect

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 2
    cell1_src, cell2_src = code_cells

    # 1. Wave C Part 2 execution flags & urllib3-safe seekable zstd shim in Cell 1
    assert "RUN_XVS_THERMAL = False" in cell1_src
    assert "RUN_XVS_LAND = True" in cell1_src
    assert "import urllib3.response" in cell1_src
    assert "zstd_mod.ZstdError = _zstd_compat.ZstdError" in cell1_src
    assert cell1_src.index("import urllib3.response") < cell1_src.index("sys.modules['compression.zstd']")
    assert "_zstd_compat.open(" not in cell1_src, "Non-seekable _zstd_compat.open breaks verify_results_archive mode='r:'"
    assert "dctx.copy_stream(" in cell1_src

    # 2. Crash-safe flushed+fsynced error logs, heartbeat, direct Drive export & isolated unassign in Cell 2
    for token in (
        "traceback.print_exc()",
        "land_error.log",
        "thermal_error.log",
        "run_status.json",
        "os.fsync(",
        "xvs_land_final.pt",
        "xvs_land_v45.onnx",
        "xvs_land_g4_metrics.json",
        "safe_colab_unassign(AUTO_UNASSIGN_ON_EXIT)",
    ):
        assert token in cell2_src, f"Missing required Cell 2 contract token: {token!r}"

    # 3. Extract exact XvsLandG4Config(...) call AST from Cell 2, bind signature, and run local smoke proof
    tree2 = ast.parse(cell2_src, filename=NOTEBOOK_PATH.name)
    land_calls = [
        node
        for node in ast.walk(tree2)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "XvsLandG4Config"
    ]
    assert len(land_calls) == 1
    land_call = land_calls[0]
    assert len(land_call.args) == 0
    kw_names = [kw.arg for kw in land_call.keywords]
    assert all(isinstance(k, str) for k in kw_names)
    expected_kw_set = {
        "out_dir",
        "lard_manifest",
        "runways_csv",
        "drive_ckpt_dir",
        "smoke_local",
        "real_world_scale",
        "use_naip_stac",
        "export_onnx",
        "stage2a_epochs",
        "stage2b_epochs",
        "steps_per_epoch",
        "num_train_samples",
        "vram_headroom_gib",
        "seed",
    }
    assert set(kw_names) == expected_kw_set

    # Resolve literal kwargs directly from Cell 2 AST and bind against XvsLandG4Config signature
    env_bindings = {
        "land_out": tmp_path / "land_out",
        "LARD_ICAO_MANIFEST": LARD_MANIFEST,
        "land_drive": tmp_path / "drive_ckpts" / "xvs_land",
        "USE_NAIP_STAC": True,
        "SEED": 20261122,
    }
    resolved_kwargs: dict[str, object] = {}
    for kw in land_call.keywords:
        assert kw.arg is not None
        if kw.arg == "runways_csv":
            resolved_kwargs[kw.arg] = RUNWAYS_CSV
        elif isinstance(kw.value, ast.Name) and kw.value.id in env_bindings:
            resolved_kwargs[kw.arg] = env_bindings[kw.value.id]
        else:
            resolved_kwargs[kw.arg] = ast.literal_eval(kw.value)

    assert resolved_kwargs["smoke_local"] is False
    assert resolved_kwargs["real_world_scale"] is True
    assert resolved_kwargs["use_naip_stac"] is True
    assert resolved_kwargs["export_onnx"] is True
    assert resolved_kwargs["stage2a_epochs"] == 24
    assert resolved_kwargs["stage2b_epochs"] == 10
    assert resolved_kwargs["steps_per_epoch"] == 200
    assert resolved_kwargs["num_train_samples"] == 2400
    assert float(resolved_kwargs["vram_headroom_gib"]) == pytest.approx(4.0)

    bound = inspect.signature(XvsLandG4Config).bind(**resolved_kwargs)
    bound.apply_defaults()
    # Instantiate exact Cell 2 production config (proving full runtime constructor compatibility)
    prod_cfg = XvsLandG4Config(**resolved_kwargs)  # type: ignore[arg-type]
    assert prod_cfg.real_world_scale is True
    assert prod_cfg.num_train_samples == 2400

    # Execute local CPU smoke run with the exact same kwarg set (only toggling smoke_local=True)
    smoke_kwargs = dict(resolved_kwargs)
    smoke_kwargs["smoke_local"] = True
    smoke_rep = run_xvs_land_g4_training(XvsLandG4Config(**smoke_kwargs))  # type: ignore[arg-type]
    assert smoke_rep["pass"] is True
    assert smoke_rep["lard_guard"]["manifest_verified"] is True
    assert smoke_rep["lard_guard"]["ksfo_heldout"] is True
    assert int(smoke_rep["eval_metrics"]["hmi_count"]) == 0
    assert (tmp_path / "land_out" / "xvs_land_final.pt").is_file()
    assert (tmp_path / "land_out" / "xvs_land_v45.onnx").is_file()
    assert (tmp_path / "land_out" / "xvs_land_g4_metrics.json").is_file()


def test_colab_l4_v45_zstd_shim_prevents_urllib3_crash_and_supports_seekable_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the notebook's exact Cell 1 zstd shim under simulated Python 3.12: urllib3 + verify_results_archive PASS."""
    import ast
    import builtins
    from compression import zstd as stdlib_zstd
    import importlib
    import importlib.machinery
    import os
    import sys
    import tempfile
    import types

    scripts_xvs = str(REPO_ROOT / "scripts" / "xvs")
    if scripts_xvs not in sys.path:
        sys.path.insert(0, scripts_xvs)
    import v2_m2_stage as v2_m2_stage_mod  # type: ignore[import-not-found]

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell1_src = "".join(nb["cells"][1]["source"])
    tree1 = ast.parse(cell1_src, filename=NOTEBOOK_PATH.name)
    fn_nodes = [
        n for n in tree1.body if isinstance(n, ast.FunctionDef) and n.name == "ensure_colab_compression_zstd_shim"
    ]
    assert len(fn_nodes) == 1
    fn_src = ast.get_source_segment(cell1_src, fn_nodes[0])
    assert fn_src is not None

    class _FakeZstdError(Exception):
        pass

    class _FakeCompressor:
        def __init__(self, level: int = 3) -> None:
            self.level = int(level)

        def copy_stream(self, reader, writer) -> tuple[int, int]:
            raw = reader.read()
            comp = stdlib_zstd.compress(raw)
            writer.write(comp)
            return len(raw), len(comp)

    class _FakeDecompressor:
        def copy_stream(self, reader, writer) -> tuple[int, int]:
            comp = reader.read()
            raw = stdlib_zstd.decompress(comp)
            writer.write(raw)
            return len(comp), len(raw)

    fake_zstandard = types.ModuleType("zstandard")
    fake_zstandard.ZstdCompressor = _FakeCompressor  # type: ignore[attr-defined]
    fake_zstandard.ZstdDecompressor = _FakeDecompressor  # type: ignore[attr-defined]
    fake_zstandard.ZstdError = _FakeZstdError  # type: ignore[attr-defined]

    def _forbid_direct_open(*_a, **_kw):
        raise AssertionError("Non-seekable zstandard.open must NOT be used by shim")

    fake_zstandard.open = _forbid_direct_open  # type: ignore[attr-defined]

    orig_import = builtins.__import__

    def _sim_py312_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (name == "compression" or name.startswith("compression.")) and "compression.zstd" not in sys.modules:
            raise ImportError("simulated Python 3.12 without stdlib compression.zstd")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "zstandard", fake_zstandard)
    monkeypatch.delitem(sys.modules, "compression", raising=False)
    monkeypatch.delitem(sys.modules, "compression.zstd", raising=False)
    monkeypatch.delitem(sys.modules, "urllib3.response", raising=False)
    monkeypatch.setattr(builtins, "__import__", _sim_py312_import)

    ns: dict[str, object] = {
        "Path": Path,
        "os": os,
        "sys": sys,
        "tempfile": tempfile,
        "types": types,
        "importlib": importlib,
    }
    exec(compile(fn_src, NOTEBOOK_PATH.name, "exec"), ns)
    ns["ensure_colab_compression_zstd_shim"]()  # type: ignore[operator]

    # Restore real __import__ now that shim is installed in sys.modules
    monkeypatch.setattr(builtins, "__import__", orig_import)

    shimmed_zstd = sys.modules["compression.zstd"]
    assert hasattr(shimmed_zstd, "ZstdError")
    assert shimmed_zstd.ZstdError is _FakeZstdError

    # 1a. Pre-imported urllib3.response saw HAS_ZSTD=False cleanly (never binds to fake zstd)
    urllib3_resp_mod = sys.modules["urllib3.response"]
    assert hasattr(urllib3_resp_mod, "BaseHTTPResponse")
    assert getattr(urllib3_resp_mod, "HAS_ZSTD", None) is False

    # 1b. Secondary defense: even reloading urllib3.response AFTER shim install must NOT raise AttributeError
    importlib.reload(urllib3_resp_mod)
    assert hasattr(urllib3_resp_mod, "BaseHTTPResponse")

    # 2. Full build_stage_archive + verify_results_archive (mode="r:") round-trip via shimmed compression.zstd
    src_root = tmp_path / "shim_roundtrip"
    src_root.mkdir()
    (src_root / "metrics.json").write_text(json.dumps({"ok": True, "hmi_count": 0}) + "\n", encoding="utf-8")
    torch.save({"w": torch.tensor([1.0, 2.0])}, src_root / "xvs_land_final.pt")

    monkeypatch.setattr(v2_m2_stage_mod, "zstd", shimmed_zstd)
    packed = v2_m2_stage_mod.build_stage_archive(
        repo_root=src_root,
        out_dir=tmp_path / "archives",
        job_id="v45_shim_verify_test",
        include_paths=["metrics.json", "xvs_land_final.pt"],
    )
    ver = v2_m2_stage_mod.verify_results_archive(
        Path(packed["archive_path"]),
        expected_sha256=packed["archive_sha256"],
        check_tensors=True,
    )
    assert ver["pass"] is True
    assert ver["checkpoints_checked"] == 1
    assert ver["non_finite_tensor_count"] == 0


def test_xvs_land_naip_stac_optional_import_non_importerror_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-ImportError failure in optional STAC deps degrades gracefully to [] instead of aborting training."""
    import builtins
    from xvs.v2.xvs_land.train_g4 import RunwaySpec, fetch_conus_airport_naip_textures

    orig_import = builtins.__import__

    def _broken_stac_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "planetary_computer":
            raise AttributeError("module 'compression.zstd' has no attribute 'ZstdError'")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken_stac_import)
    rwys = [RunwaySpec(icao="KDEN", length_m=3657.6, width_m=45.7, heading_deg=170.0, le_lat_deg=39.85, le_lon_deg=-104.67)]
    res = fetch_conus_airport_naip_textures(rwys, "https://planetarycomputer.microsoft.com/api/stac/v1")
    assert res == []
