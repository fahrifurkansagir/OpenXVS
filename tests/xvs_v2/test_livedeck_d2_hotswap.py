"""Verification tests for Live Deck `XVS-D2` model registration, G4 `96/128` headroom & post-Colab hot-swap CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from scripts.xvs.install_colab_g4_d2_onnx import install_colab_g4_d2_onnx
from scripts.xvs.v2_m2_stage import build_stage_archive
from xvs.v2.xvs_d2.export_onnx import audit_onnx_forbidden_ops

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVEDECK_MODELS_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "models"
SEED_ONNX_PATH = LIVEDECK_MODELS_DIR / "xvs_d2_web_fp16.onnx"
REGISTRY_PATH = LIVEDECK_MODELS_DIR / "registry.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_livedeck_registry_and_seeded_xvs_d2_onnx_sha256_match() -> None:
    assert SEED_ONNX_PATH.is_file(), f"Missing Live Deck ONNX seed: {SEED_ONNX_PATH}"
    assert REGISTRY_PATH.is_file(), f"Missing Live Deck registry: {REGISTRY_PATH}"

    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert reg["active_model_id"] == "d1_kf_k3_g1"

    d2_entries = [m for m in reg["models"] if m["id"] == "xvs_d2_web_fp16"]
    assert len(d2_entries) == 1
    d2 = d2_entries[0]
    assert d2["status"] == "local_v2_m2_architecture_seed_awaiting_colab_g4_weights"
    assert [i["name"] for i in d2["inputs"]] == ["x_t", "h_prev_warped"]
    assert d2["inputs"][0]["shape"] == [1, 6, 256, 256]
    assert d2["inputs"][1]["shape"] == [1, 24, 64, 64]
    assert [o["name"] for o in d2["outputs"]] == [
        "hm_logits",
        "offset",
        "log_scale",
        "log_var",
        "peak_mask",
        "h_next",
    ]

    fp16 = d2["artifacts"]["fp16"]
    assert fp16["path"] == "models/xvs_d2_web_fp16.onnx"
    actual_bytes = int(SEED_ONNX_PATH.stat().st_size)
    actual_sha = _sha256_file(SEED_ONNX_PATH)
    assert int(fp16["bytes"]) == actual_bytes
    assert str(fp16["sha256"]) == actual_sha
    assert actual_bytes <= 1_048_576
    assert audit_onnx_forbidden_ops(SEED_ONNX_PATH) == []


def test_install_colab_g4_d2_onnx_hotswap_round_trip_and_nan_rejection(tmp_path: Path) -> None:
    # Prepare isolated Live Deck models dir copy
    isolated_models = tmp_path / "livedeck_models"
    isolated_models.mkdir()
    shutil.copy2(REGISTRY_PATH, isolated_models / "registry.json")
    shutil.copy2(SEED_ONNX_PATH, isolated_models / "xvs_d2_web_fp16.onnx")

    # Build a clean mock Colab G4 results archive
    results_src = tmp_path / "g4_results_src"
    (results_src / "onnx").mkdir(parents=True)
    (results_src / "checkpoints").mkdir(parents=True)
    shutil.copy2(SEED_ONNX_PATH, results_src / "onnx" / "xvs_d2_web_fp16.onnx")
    torch.save({"w": torch.tensor([1.0, 2.0, 3.0])}, results_src / "checkpoints" / "xvs_d2_web.pt")
    eval_metrics = {
        "teacher": {
            "pd_at_fpi_0p1": 0.91,
            "pd_at_fpi_0p5": 0.95,
            "pd_at_fpi_1p0": 0.97,
            "ap50": 0.93,
            "centroid_rmse_px": 1.42,
            "above_horizon_pd": 0.96,
            "below_horizon_pd": 0.89,
        },
        "web": {
            "pd_at_fpi_0p1": 0.87,
            "pd_at_fpi_0p5": 0.92,
            "pd_at_fpi_1p0": 0.95,
            "ap50": 0.89,
            "centroid_rmse_px": 1.85,
            "above_horizon_pd": 0.93,
            "below_horizon_pd": 0.84,
        },
    }
    (results_src / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2) + "\n", encoding="utf-8")

    packed = build_stage_archive(
        repo_root=results_src,
        out_dir=tmp_path / "archives",
        job_id="v2_m2_01_xvs_d2_aot_g4_results",
        include_paths=["onnx", "checkpoints", "eval_metrics.json"],
    )

    # 1. Dry-run verify (install=False) leaves registry untouched
    dry_report = install_colab_g4_d2_onnx(
        Path(packed["archive_path"]),
        expected_sha256=packed["archive_sha256"],
        install=False,
        livedeck_models_dir=isolated_models,
    )
    assert dry_report["pass"] is True
    assert dry_report["installed"] is False
    assert dry_report["forbidden_ops"] == []
    assert dry_report["eval_metrics"]["web"]["ap50"] == 0.89
    reg_before = json.loads((isolated_models / "registry.json").read_text(encoding="utf-8"))
    d2_before = [m for m in reg_before["models"] if m["id"] == "xvs_d2_web_fp16"][0]
    assert d2_before["status"] == "local_v2_m2_architecture_seed_awaiting_colab_g4_weights"

    # 2. Live install updates ONNX + registry.json to trained_colab_g4_blackwell
    live_report = install_colab_g4_d2_onnx(
        Path(packed["archive_path"]),
        expected_sha256=packed["archive_sha256"],
        install=True,
        livedeck_models_dir=isolated_models,
    )
    assert live_report["pass"] is True
    assert live_report["installed"] is True
    assert live_report["registry_updated"] is True
    reg_after = json.loads((isolated_models / "registry.json").read_text(encoding="utf-8"))
    d2_after = [m for m in reg_after["models"] if m["id"] == "xvs_d2_web_fp16"][0]
    assert d2_after["status"] == "trained_colab_g4_blackwell"
    assert d2_after["artifacts"]["fp16"]["sha256"] == _sha256_file(isolated_models / "xvs_d2_web_fp16.onnx")
    assert d2_after["artifacts"]["fp16"]["bytes"] == int((isolated_models / "xvs_d2_web_fp16.onnx").stat().st_size)
    assert d2_after["trained_eval_metrics"]["web"]["ap50"] == 0.89

    # 3. Negative control: NaN tensor in checkpoint MUST fail closed before installing
    bad_src = tmp_path / "g4_bad_src"
    (bad_src / "onnx").mkdir(parents=True)
    (bad_src / "checkpoints").mkdir(parents=True)
    shutil.copy2(SEED_ONNX_PATH, bad_src / "onnx" / "xvs_d2_web_fp16.onnx")
    torch.save({"w": torch.tensor([1.0, float("nan")])}, bad_src / "checkpoints" / "xvs_d2_web.pt")
    bad_packed = build_stage_archive(
        repo_root=bad_src,
        out_dir=tmp_path / "bad_archives",
        job_id="bad_nan_results",
        include_paths=["onnx", "checkpoints"],
    )
    with pytest.raises(ValueError, match="verification FAILED|Non-finite"):
        install_colab_g4_d2_onnx(
            Path(bad_packed["archive_path"]),
            install=True,
            livedeck_models_dir=isolated_models,
        )


def test_xvs_d2_train_g4_has_96_128_batch_headroom() -> None:
    src = (REPO_ROOT / "src" / "xvs" / "v2" / "xvs_d2" / "train_g4.py").read_text(encoding="utf-8")
    assert "stage1_candidates = [4, 8, 16, 24, 32, 48, 64, 96, 128]" in src
    assert "stage4_candidates = [8, 16, 24, 32, 48, 64, 96, 128]" in src
