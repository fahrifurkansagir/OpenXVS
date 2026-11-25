"""Fast unit tests for tools/xvs/verify_copy_sha256.py using tmp_path fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "xvs" / "verify_copy_sha256.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("verify_copy_sha256", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_copy_sha256"] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_synthetic_output_tree(src_root: Path) -> dict[str, str]:
    """Create a minimal synthetic output tree with valid manifests and JSONL records."""
    wp07_dir = src_root / "xvs" / "wp07" / "g0"
    wp05_dir = src_root / "xvs" / "wp05" / "g0_lambda_sweep_v2"
    wp07_dir.mkdir(parents=True, exist_ok=True)
    wp05_dir.mkdir(parents=True, exist_ok=True)

    wp07_payloads = {
        "calibration": (
            "dataset_calibration.jsonl",
            b'{"sample_id": "cal_01"}\n',
        ),
        "validation": (
            "dataset_validation.jsonl",
            b'{"sample_id": "val_01"}\n',
        ),
        "test": (
            "dataset_test.jsonl",
            b'{"sample_id": "tst_01"}\n',
        ),
        "fault_log": (
            "fault_log.json",
            b'{"faults": []}\n',
        ),
        "pilot": (
            "pilot_projection.json",
            b'{"projected_s": 12.0}\n',
        ),
    }

    files_map: dict[str, str] = {}
    sha_map: dict[str, str] = {}
    for key, (fname, raw_bytes) in wp07_payloads.items():
        fpath = wp07_dir / fname
        fpath.write_bytes(raw_bytes)
        files_map[key] = f"output\\xvs\\wp07\\g0\\{fname}"
        sha_map[key] = _sha256_bytes(raw_bytes)

    (wp07_dir / "results.json").write_text('{"status": "ok"}\n', encoding="utf-8")
    (wp07_dir / "data_manifest.json").write_text(
        json.dumps({"files": files_map, "file_sha256": sha_map}, indent=2),
        encoding="utf-8",
    )

    rec_v2_bytes = b'{"encounter_id": "E0001", "seed": 101}\n'
    rec_pers_bytes = b'{"encounter_id": "E0001", "seed": 101, "model": "primary_1hz_hold"}\n'
    (wp05_dir / "per_run_records.jsonl").write_bytes(rec_v2_bytes)
    (wp05_dir / "per_run_records_persistent.jsonl").write_bytes(rec_pers_bytes)

    sha_v2 = _sha256_bytes(rec_v2_bytes)
    sha_pers = _sha256_bytes(rec_pers_bytes)

    (wp05_dir / "lambda_sweep_v2_results.json").write_text(
        json.dumps(
            {
                "per_run_records_file": "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records.jsonl",
                "per_run_records_sha256": sha_v2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (wp05_dir / "lambda_sweep_persistent_clutter.json").write_text(
        json.dumps(
            {
                "per_run_records_file": "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records_persistent.jsonl",
                "per_run_records_sha256": sha_pers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        **sha_map,
        "per_run_records_v2": sha_v2,
        "per_run_records_persistent": sha_pers,
    }


def test_dry_run_reports_matches_and_never_copies(tmp_path: Path) -> None:
    mod = _load_tool_module()
    src_root = tmp_path / "src_output"
    dst_root = tmp_path / "dst_output"
    _build_synthetic_output_tree(src_root)

    # Run via CLI without --execute (default dry-run)
    rc = mod.main(["--src-root", str(src_root), "--dst-root", str(dst_root)])
    assert rc == 0
    assert not dst_root.exists()

    summary = mod.verify_and_copy(src_root=src_root, dst_root=dst_root, execute=False)
    assert summary["mode"] == "DRY_RUN"
    assert summary["all_verified"] is True
    assert summary["manifest_pinned_files_checked"] == 7
    assert summary["copied_count"] == 0
    assert summary["would_copy_count"] == summary["total_files_checked"]
    assert not dst_root.exists()


def test_execute_copies_verified_files_and_never_deletes(tmp_path: Path) -> None:
    mod = _load_tool_module()
    src_root = tmp_path / "src_output"
    dst_root = tmp_path / "dst_output"
    _build_synthetic_output_tree(src_root)

    # Pre-populate an unrelated existing file in dst_root to verify nothing is ever deleted
    preexisting = dst_root / "xvs" / "wp07" / "g0" / "keep_me_existing.txt"
    preexisting.parent.mkdir(parents=True, exist_ok=True)
    preexisting.write_text("do not delete", encoding="utf-8")

    rc = mod.main(
        ["--src-root", str(src_root), "--dst-root", str(dst_root), "--execute"]
    )
    assert rc == 0
    assert preexisting.is_file()
    assert preexisting.read_text(encoding="utf-8") == "do not delete"
    assert (dst_root / "xvs" / "wp07" / "g0" / "dataset_test.jsonl").is_file()
    assert (
        dst_root / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records.jsonl"
    ).is_file()

    # Second run should find all files ALREADY_IDENTICAL and copy 0 files
    summary_second = mod.verify_and_copy(
        src_root=src_root, dst_root=dst_root, execute=False
    )
    assert summary_second["all_verified"] is True
    assert summary_second["would_copy_count"] == 0
    assert all(f["action"] == "ALREADY_IDENTICAL" for f in summary_second["files"])


def test_hash_mismatch_blocks_copy_even_with_execute(tmp_path: Path) -> None:
    mod = _load_tool_module()
    src_root = tmp_path / "src_output"
    dst_root = tmp_path / "dst_output"
    _build_synthetic_output_tree(src_root)

    # Tamper with per_run_records.jsonl so its SHA-256 no longer matches lambda_sweep_v2_results.json
    (
        src_root / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records.jsonl"
    ).write_bytes(b"corrupted content\n")

    rc = mod.main(
        ["--src-root", str(src_root), "--dst-root", str(dst_root), "--execute"]
    )
    assert rc == 1
    assert not dst_root.exists()
