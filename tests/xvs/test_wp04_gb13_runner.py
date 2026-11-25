"""Unit tests for WP-04 G3a runner G-B13 enhancements (P0-2).

Tests:
1. --g3a-root resolution across repo and main directories.
2. File SHA-256 calculation for RELEASE.json.
3. Recording release_json_sha256 in closure config and metadata outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from scripts.xvs.detection.run_g3a_g0 import (
    compute_file_sha256,
    resolve_g3a_root,
    write_closure_config,
)


def test_compute_file_sha256(tmp_path: Path):
    """compute_file_sha256 must match standard hashlib sha256."""
    test_file = tmp_path / "RELEASE.json"
    content = b'{"version": "0.1", "status": "APPROVED", "frames": 49812}'
    test_file.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()
    actual_sha = compute_file_sha256(test_file)
    assert actual_sha == expected_sha


def test_resolve_g3a_root(tmp_path: Path):
    """resolve_g3a_root resolves existing directories and raises FileNotFoundError on missing."""
    d = tmp_path / "g3a_v01"
    d.mkdir()

    resolved = resolve_g3a_root(d)
    assert resolved == d.resolve()

    with pytest.raises(FileNotFoundError):
        resolve_g3a_root(tmp_path / "nonexistent_dir_12345")


def test_write_closure_config_records_release_sha256(tmp_path: Path):
    """write_closure_config must record release_json_sha256 and provenance when provided."""
    base_cfg = {
        "fppi_lambda": 0.001,
        "s50_px": 3.0,
        "w_px": 0.7,
        "provenances": {},
    }
    base_cfg_path = tmp_path / "base_g0.json"
    base_cfg_path.write_text(json.dumps(base_cfg), encoding="utf-8")

    out_cfg_path = tmp_path / "out_measured.json"
    mock_closure = {
        "status": "FITTED",
        "s50_px": 2.85,
        "w_px": 0.45,
        "w_ln": 0.52,
        "provenance": "WP-04 G3a log-logistic",
    }
    test_sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

    # 1. With release_sha256
    write_closure_config(
        mock_closure,
        measured_lambda=0.00045,
        op_k=3,
        op_tau=20.0,
        out_config_path=out_cfg_path,
        base_config_path=base_cfg_path,
        release_sha256=test_sha,
    )

    with open(out_cfg_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["release_json_sha256"] == test_sha
    assert "release_json_sha256" in res["provenances"]
    assert test_sha in res["provenances"]["release_json_sha256"]
    assert res["s50_px"] == 2.85
    assert res["w_ln"] == 0.52
    assert res["fppi_lambda"] == 0.00045

    # 2. Without release_sha256
    out_cfg_path_no_sha = tmp_path / "out_no_sha.json"
    write_closure_config(
        mock_closure,
        measured_lambda=0.00045,
        op_k=3,
        op_tau=20.0,
        out_config_path=out_cfg_path_no_sha,
        base_config_path=base_cfg_path,
        release_sha256=None,
    )

    with open(out_cfg_path_no_sha, "r", encoding="utf-8") as f:
        res_no_sha = json.load(f)

    assert "release_json_sha256" not in res_no_sha
