"""Gate `G4R-4` verification: `XVS-World` 1x1 km KSFO 3DEP Trainer, Niantic SPZ v2, Previews & `stage-g4` CLI.

Verifies `SP2B_G4_TRAINERS_DESIGN.md` §3.3 & §4 (`BC-G4-B1`, `BC-G4-B3`, `BC-G4-M1`, `BC-G4-m1`):
1. Full `--smoke-local` multi-view 3DGS optimization on real `apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin`
   via `load_ksfo_3dep_mipmap`, asserting loss decrease, PSNR improvement, `VISUAL_STRESS_ONLY`,
   Drive checkpoint mirror + resume round-trip, Niantic SPZ v2 (`0x5053474E` `"NGSP"`, `version == 2`)
   round-trip header check, and 3 distinct RGB PNG visual-stress previews.
2. `scripts/xvs/v2_m2_stage.py` `stage-g4` deterministic `.tar.zst` archive + `verify_results_archive` +
   strict zero `"zstandard"` substring lock.
3. Upgraded `tools/colab/v2_m2_03_xvs_world_1km_3dgs_g4.ipynb` `nbformat 4.5`, pure `ast.parse()`,
   `importlib` + `inspect.signature` bindings, Blackwell `sm_120`, `compression.zstd` shim, and hygiene.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import types
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.xvs.v2_m2_stage import (  # noqa: E402
    G4_FULL_STAGE_PATHS,
    build_g4_full_stage_archive,
    main as stage_cli_main,
    verify_results_archive,
)
from xvs.v2.xvs_world.config import ksfo_28l_default_config  # noqa: E402
from xvs.v2.xvs_world.train_g4 import (  # noqa: E402
    SPZ_MAGIC_ASCII,
    SPZ_MAGIC_U32,
    SPZ_VERSION,
    GaussianScene3D,
    _densify_and_prune_splats,
    _fetch_optional_naip_albedo,
    read_spz_v2_header,
    run_xvs_world_g4_training,
    write_spz_v2,
)

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "university_1652",
    "sues_200",
    "lard_synthetic",
    "convnext_v2_nc",
    "dfine_obj365",
    "rf_detr_xl",
    "rf_detr_2xl",
    "lard_v2_yolo_agpl",
    "msfs",
    "xplane",
    "arcgis.com",
    "arcgisonline.com",
    "aerial_uav_thermal_inferred_unified_kaggle",
    "thermal_uav_scc_loc",
)


def test_xvs_world_smoke_local_real_ksfo_3dep_spz_v2_and_previews(tmp_path: Path) -> None:
    run_dir = tmp_path / "xvs_world_run"
    drive_dir = tmp_path / "drive_mirror"

    res = run_xvs_world_g4_training(
        out_dir=run_dir,
        drive_dir=drive_dir,
        smoke_local=True,
    )
    assert res["pass"] is True, f"XVS-World smoke-local failed: {res}"
    assert res["gt_tier"] == "VISUAL_STRESS_ONLY"
    assert res["wp01_certification_eligible"] is False
    assert res["ksfo_3dep_terrain_mode"] == "usgs_3dep_geoid18_ellipsoidal"
    assert res["ksfo_3dep_mipmap_levels"] == 9
    assert res["num_splats"] == 16 * 16
    assert res["num_splats"] <= res["max_splats_cap"]
    assert res["loss_final"] < res["loss_init"], (
        f"Expected loss drop, got init={res['loss_init']} final={res['loss_final']}"
    )
    assert res["psnr_final_db"] > res["psnr_init_db"], (
        f"Expected PSNR improvement, got init={res['psnr_init_db']} final={res['psnr_final_db']}"
    )
    assert res["checkpoint_resumed"] is False

    # Verify atomic checkpoints exist locally AND mirrored to Drive
    local_ckpts = sorted((run_dir / "checkpoints").glob("xvs_world_step_*.pt"))
    drive_ckpts = sorted(drive_dir.glob("xvs_world_step_*.pt"))
    assert len(local_ckpts) >= 1 and len(drive_ckpts) >= 1

    # Verify Niantic SPZ v2 binary round-trip header (`BC-G4-m1`)
    spz_path = run_dir / "ksfo_28l_1km_visual_stress.spz"
    assert spz_path.is_file()
    hdr = read_spz_v2_header(spz_path)
    assert hdr["magic_u32"] == SPZ_MAGIC_U32
    assert hdr["magic_ascii"] == SPZ_MAGIC_ASCII == "NGSP"
    assert hdr["version"] == SPZ_VERSION == 2
    assert hdr["num_points"] == res["num_splats"]
    assert hdr["uncompressed_bytes"] == 16 + res["num_splats"] * 19
    assert 0 < hdr["compressed_bytes"] < hdr["uncompressed_bytes"]
    assert hdr["spz_gb"] <= 0.35

    # Negative SPZ check: non-finite Gaussian mean rejected fail-closed
    bad_means = np.zeros((4, 3), dtype=np.float32)
    bad_means[1, 2] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        write_spz_v2(
            tmp_path / "bad.spz",
            means_ned=bad_means,
            log_scales=np.zeros((4, 3), dtype=np.float32),
            quats_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
            logit_opacities=np.zeros((4,), dtype=np.float32),
            rgb_01=np.full((4, 3), 0.5, dtype=np.float32),
        )

    # Verify 3 real visual-stress RGB PNG previews
    preview_dir = run_dir / "previews"
    arrays: dict[str, np.ndarray] = {}
    for fname in ("preview_clear.png", "preview_radiation_fog.png", "preview_dusk_glare.png"):
        fpath = preview_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 100
        with Image.open(fpath) as im:
            assert im.mode == "RGB"
            assert im.size == (48, 36)
            arrays[fname] = np.asarray(im, dtype=np.float32)
    assert not np.array_equal(arrays["preview_clear.png"], arrays["preview_radiation_fog.png"])
    assert not np.array_equal(arrays["preview_clear.png"], arrays["preview_dusk_glare.png"])

    # Verify checkpoint resume path cleanly resumes from latest step (+2 steps)
    res2 = run_xvs_world_g4_training(
        out_dir=run_dir,
        drive_dir=drive_dir,
        smoke_local=True,
        max_steps=res["steps_completed"] + 2,
    )
    assert res2["checkpoint_resumed"] is True
    assert res2["steps_completed"] == res["steps_completed"] + 2
    assert res2["pass"] is True

    # D16-G4-m2 regression: resuming an ALREADY COMPLETED run (start_step == max_steps)
    # must preserve initial baseline loss_init/psnr_init and still report pass=True!
    res_completed = run_xvs_world_g4_training(
        out_dir=run_dir,
        drive_dir=drive_dir,
        smoke_local=True,
        max_steps=res2["steps_completed"],
    )
    assert res_completed["checkpoint_resumed"] is True
    assert res_completed["loss_final"] < res_completed["loss_init"]
    assert res_completed["psnr_final_db"] > res_completed["psnr_init_db"]
    assert res_completed["pass"] is True


def test_naip_utm_crs_reprojection_and_bounded_mcmc_densification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `D16-G4-M2` (EPSG:4326 -> UTM EPSG:26910 reprojection + boundless=True) & `D16-G4-m2` cap."""
    calls: dict[str, Any] = {"transform_bounds_args": None, "from_bounds_args": None, "read_kwargs": None}

    class _FakeAsset:
        href = "https://example.invalid/ksfo_naip_utm10n.tif"

    class _FakeItem:
        id = "ca_m_3712221_se_10_060_20220519"
        assets = {"image": _FakeAsset()}

    class _FakeSearch:
        def items(self):
            return [_FakeItem()]

    class _FakeClient:
        def search(self, **kwargs):
            return _FakeSearch()

    class _FakeSrc:
        crs = "EPSG:26910"
        transform = "FAKE_UTM_AFFINE"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def read(self, bands, **kwargs):
            calls["read_kwargs"] = kwargs
            h, w = kwargs["out_shape"][1], kwargs["out_shape"][2]
            return np.full((3, h, w), 128, dtype=np.uint8)

    fake_pystac = types.ModuleType("pystac_client")
    fake_pystac.Client = types.SimpleNamespace(open=lambda url, modifier=None: _FakeClient())  # type: ignore[attr-defined]
    fake_pc = types.ModuleType("planetary_computer")
    fake_pc.sign_inplace = lambda x: x  # type: ignore[attr-defined]

    fake_rasterio = types.ModuleType("rasterio")
    fake_rasterio.open = lambda href: _FakeSrc()  # type: ignore[attr-defined]
    fake_warp = types.ModuleType("rasterio.warp")

    def _fake_transform_bounds(src_crs, dst_crs, left, bottom, right, top):
        calls["transform_bounds_args"] = (src_crs, dst_crs, left, bottom, right, top)
        return (556000.0, 4162500.0, 557000.0, 4163500.0)

    fake_warp.transform_bounds = _fake_transform_bounds  # type: ignore[attr-defined]
    fake_windows = types.ModuleType("rasterio.windows")

    def _fake_from_bounds(left, bottom, right, top, transform=None):
        calls["from_bounds_args"] = (left, bottom, right, top, transform)
        return "FAKE_WINDOW"

    fake_windows.from_bounds = _fake_from_bounds  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pystac_client", fake_pystac)
    monkeypatch.setitem(sys.modules, "planetary_computer", fake_pc)
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)
    monkeypatch.setitem(sys.modules, "rasterio.warp", fake_warp)
    monkeypatch.setitem(sys.modules, "rasterio.windows", fake_windows)

    albedo, src_tag = _fetch_optional_naip_albedo(ksfo_28l_default_config(), grid_res=16)
    assert albedo is not None and albedo.shape == (16, 16, 3)
    assert src_tag == "usda_naip_stac:ca_m_3712221_se_10_060_20220519"
    assert calls["transform_bounds_args"][0] == "EPSG:4326"
    assert calls["transform_bounds_args"][1] == "EPSG:26910"
    assert calls["from_bounds_args"] == (556000.0, 4162500.0, 557000.0, 4163500.0, "FAKE_UTM_AFFINE")
    assert calls["read_kwargs"]["boundless"] is True

    # Bounded MCMC densification cap check (`D16-G4-m2`)
    scene = GaussianScene3D(torch.zeros((32, 3)), torch.full((32, 3), 0.5))
    grads = torch.linspace(0.1, 3.2, 32)
    stats = _densify_and_prune_splats(scene, max_splats=35, grad_norms=grads, split_fraction=0.5)
    assert stats["splats_after"] <= 35
    assert scene.num_splats == stats["splats_after"]
    assert scene.num_splats == 35


def test_stage_g4_full_archive_determinism_and_verification(tmp_path: Path) -> None:
    # BC-G4-B3 hard lock: zero "zstandard" substring anywhere in scripts/xvs/v2_m2_stage.py
    stage_src = (REPO_ROOT / "scripts" / "xvs" / "v2_m2_stage.py").read_text(encoding="utf-8")
    assert "zstandard" not in stage_src

    assert "data/reference/ourairports/runways.csv" in G4_FULL_STAGE_PATHS
    assert "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin" in G4_FULL_STAGE_PATHS
    assert "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.manifest.json" in G4_FULL_STAGE_PATHS

    s1 = build_g4_full_stage_archive(REPO_ROOT, tmp_path / "stage1", job_id="openxvs_v2_m2_g4")
    s2 = build_g4_full_stage_archive(REPO_ROOT, tmp_path / "stage2", job_id="openxvs_v2_m2_g4")
    assert s1["archive_sha256"] == s2["archive_sha256"]
    assert s1["archive_bytes"] == s2["archive_bytes"]
    assert s1["file_count"] >= 40

    manifest = json.loads(Path(s1["manifest_path"]).read_text(encoding="utf-8"))
    packed_paths = {entry["path"] for entry in manifest["files"]}
    for required_rel in (
        "src/xvs/v2/xvs_world/train_g4.py",
        "src/xvs/v2/colab_g4_utils.py",
        "scripts/xvs/v2_m2_stage.py",
        "docs/xvs/spec/data_allowlist.json",
        "data/reference/ourairports/runways.csv",
        "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.bin",
        "apps/livedeck/public/assets/ksfo_3dep_geoid18_v1.manifest.json",
    ):
        assert required_rel in packed_paths, f"Missing {required_rel} in stage-g4 archive"

    verified = verify_results_archive(
        Path(s1["archive_path"]),
        expected_sha256=s1["archive_sha256"],
        check_tensors=True,
    )
    assert verified["pass"] is True
    assert verified["member_count"] == s1["file_count"]

    # Verify CLI subcommand `stage-g4`
    cli_out = tmp_path / "cli_stage"
    rc = stage_cli_main(["stage-g4", "--out-dir", str(cli_out), "--job-id", "openxvs_v2_m2_g4"])
    assert rc == 0
    assert (cli_out / "openxvs_v2_m2_g4_stage.tar.zst").is_file()


def test_v2_m2_03_notebook_ast_signatures_and_no_enu() -> None:
    nb_path = REPO_ROOT / "tools" / "colab" / "v2_m2_03_xvs_world_1km_3dgs_g4.ipynb"
    assert nb_path.is_file()
    raw = nb_path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert nb["nbformat"] == 4 and int(nb["nbformat_minor"]) >= 5
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["language_info"]["version"] == "3.12"

    lower = raw.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in lower, f"Forbidden substring {bad!r} found in {nb_path.name}"

    imported_symbols: dict[str, Any] = {}
    code_cells: list[str] = []
    verified_calls = 0

    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
        if cell["cell_type"] != "code":
            continue
        for ln in src.splitlines():
            assert not ln.lstrip().startswith(("!", "%")), f"IPython magic forbidden in cell {idx}: {ln!r}"
        tree = ast.parse(src, filename=f"v2_m2_03#cell{idx}")
        code_cells.append(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("xvs.", "scripts.xvs.")):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"Missing {alias.name} in {node.module}"
                    imported_symbols[alias.asname or alias.name] = getattr(mod, alias.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported_symbols:
                target = imported_symbols[node.func.id]
                if callable(target):
                    sig = inspect.signature(target)
                    sig.bind(
                        *[object() for _ in node.args],
                        **{str(kw.arg): object() for kw in node.keywords if kw.arg is not None},
                    )
                    verified_calls += 1

    joined = "\n".join(code_cells)
    assert len(code_cells) >= 3
    assert len(imported_symbols) >= 8
    assert verified_calls >= 8
    for tok in (
        "COLAB_G4_GPU_ARTIFACT",
        "from compression import zstd",
        "except ImportError",
        "import zstandard",
        "types.ModuleType('compression.zstd')",
        "comp_pkg.__path__ = []",
        "ModuleSpec('compression'",
        "ModuleSpec('compression.zstd'",
        "SpooledTemporaryFile",
        "if exc_type is not None:",
        "find_spec('onnx')",
        "/content/openxvs_v2_m2_g4_stage.tar.zst",
        "xvs_v2_g4_runs/openxvs_v2_m2_g4_stage.tar.zst",
        "FileNotFoundError",
        "USE_DRIVE",
        "AUTO_UNASSIGN_ON_EXIT",
        "safe_colab_unassign",
        "--only-binary=:all:",
        "INSTALL_GSPLAT",
        "pytorch_cuda_fallback",
        "except Exception as pack_exc:",
        "finally:",
    ):
        assert tok in joined, f"Missing required token {tok!r} in v2_m2_03 notebook"

    # Zero internal ENU identifiers in train_g4.py AST (`CONVENTIONS.md` §3)
    train_src = (REPO_ROOT / "src" / "xvs" / "v2" / "xvs_world" / "train_g4.py").read_text(encoding="utf-8")
    train_ast = ast.parse(train_src, filename="train_g4.py")
    enu_rx = re.compile(r"(?:^|_)enu(?:$|_)", flags=re.IGNORECASE)
    enu_hits: list[str] = []
    for node in ast.walk(train_ast):
        if isinstance(node, ast.Name) and enu_rx.search(node.id):
            enu_hits.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and enu_rx.search(node.name):
            enu_hits.append(node.name)
        elif isinstance(node, ast.arg) and enu_rx.search(node.arg):
            enu_hits.append(node.arg)
        elif isinstance(node, ast.Attribute) and enu_rx.search(node.attr):
            enu_hits.append(node.attr)
    assert enu_hits == [], f"Internal ENU identifier(s) found in train_g4.py: {enu_hits}"
