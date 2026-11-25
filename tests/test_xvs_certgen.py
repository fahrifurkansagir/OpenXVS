"""Tests for `scripts/xvs/xvs_certgen.py` v3.0 (`18/18` Kani proofs + `xvs-active-range` + `V3_M1` evidence)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTGEN_PATH = REPO_ROOT / "scripts" / "xvs" / "xvs_certgen.py"


def _load_certgen():
    spec = importlib.util.spec_from_file_location("xvs_certgen_v3_test", CERTGEN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xvs_certgen_v3_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_xvs_certgen_registers_active_range_v3_m1_and_18_kani_proofs(tmp_path: Path) -> None:
    """Task 8B gate: `xvs_certgen` verifies `xvs-active-range`, `V3_M1` reports, and `18/18` Kani proofs."""
    certgen = _load_certgen()
    out_file = tmp_path / "xvs_certgen_v3_bundle.json"
    rc = certgen.main(["--check", "--out", str(out_file)])
    assert rc == 0
    assert out_file.is_file()

    bundle = certgen.synthesize_assurance_bundle()
    assert bundle["status"] == "PASS"
    assert bundle["errors"] == []

    # Preserve G5_GAP & G2_TLA_DEFERRED fail-closed boundaries untouched
    assert "G5_GAP" in bundle["gsn_source"]["undeveloped_nodes"]
    assert "G2_TLA_DEFERRED" in bundle["gsn_source"]["undeveloped_nodes"]

    # Verify required v3.0 active-range & V3_M1 artifacts are SHA-256 verified on disk
    live_by_path = {entry["path"]: entry for entry in bundle["live_artifact_ledger"]}
    for rel_path in (
        "xvs-active-range/src/lib.rs",
        "xvs-active-range/src/kani_proofs.rs",
        "tests/golden/active_range_vectors.json",
        "docs/xvs/reports/V3_M1_active_range_benchmark.md",
        "docs/xvs/reports/V3_M1_aot_realsky_proof.md",
    ):
        assert rel_path in live_by_path, f"Missing required v3.0 artifact {rel_path!r} in live_artifact_ledger"
        entry = live_by_path[rel_path]
        assert entry["required"] is True
        assert entry["present"] is True
        assert isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
        assert int(entry["size_bytes"]) > 0

    # Verify 18/18 Kani formal proof tally across all 4 Rust crates
    assert bundle["kani_proofs_total"] == 18
    kani_by_crate = {row["crate"]: row for row in bundle["kani_proof_ledger"]}
    assert kani_by_crate["xvs-rta-core"]["proof_count"] == 5
    assert kani_by_crate["xvs-odd-vlm"]["proof_count"] == 5
    assert kani_by_crate["xvs-os-hal"]["proof_count"] == 4
    assert kani_by_crate["xvs-active-range"]["proof_count"] == 4
    for row in bundle["kani_proof_ledger"]:
        assert row["matches_expected"] is True
        assert isinstance(row["sha256"], str) and len(row["sha256"]) == 64
