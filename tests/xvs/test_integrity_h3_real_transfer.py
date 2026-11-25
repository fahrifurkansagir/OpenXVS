"""Verification tests V1_B13 .. V6_B13 for Batch B13 (WP-06/WP-07 Synthetic-to-Real Transfer Study, Hypothesis H3)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.integrity.real_transfer import (
    load_icehorizon_sequence_annotations,
    sample_stratified_frames,
    verify_icehorizon_allowlist,
)

RESULTS_PATH = REPO_ROOT / "output" / "xvs" / "wp07" / "h3_real_transfer" / "results.json"
MANIFEST_PATH = REPO_ROOT / "output" / "xvs" / "wp07" / "h3_real_transfer" / "run_manifest.json"
CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "integrity" / "h3_real_transfer.json"
ALLOWLIST_PATH = REPO_ROOT / "docs" / "xvs" / "spec" / "data_allowlist.json"


def _load_results() -> dict:
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_v1_b13_allowlist_and_dataset_inventory() -> None:
    """V1_B13: Verify IceHorizon allowlist provenance (CC-BY 4.0, evaluation-only) and 29 sequences (580 frames)."""
    info = verify_icehorizon_allowlist(ALLOWLIST_PATH, "IceHorizon")
    assert info["status"] == "ALLOWLIST_VERIFIED_EVALUATION_ONLY"
    assert "CC-BY 4.0" in info["rule"]

    res = _load_results()
    ds = res["dataset_summary"]
    assert ds["total_sequences"] == 29
    assert ds["drone_sequences"] == 5
    assert ds["ship_sequences"] == 24
    assert ds["sampled_frames_total"] == 580
    assert res["verification_matrix"]["V1_B13_allowlist_and_29_sequences"] == "PASS"


def test_v2_b13_sequence_disjoint_split_hygiene() -> None:
    """V2_B13: Verify strict sequence-disjoint split (zero shared video sequences between real_cal and real_test)."""
    res = _load_results()
    sp = res["h3_conformal_transfer"]["split_summary"]
    cal_seqs = set(sp["cal_sequences"])
    test_seqs = set(sp["test_sequences"])
    assert len(cal_seqs & test_seqs) == 0
    assert sp["sequence_overlap_count"] == 0
    assert sp["n_cal_sequences"] == 14
    assert sp["n_test_sequences"] == 15
    assert sp["n_cal_frames"] == 280
    assert sp["n_test_frames"] == 300
    assert sp["tuned_on_splits"] == ["real_cal"]
    assert sp["evaluated_on_split"] == "real_test"
    assert res["verification_matrix"]["V2_B13_sequence_disjoint_split_hygiene"] == "PASS"


def test_v3_b13_g1_vs_g0_real_horizon_improvement() -> None:
    """V3_B13: Verify G1 multi-scale + smooth pseudo-Huber IRLS outperforms G0 on real IceHorizon frames."""
    res = _load_results()
    ov = res["g0_vs_g1_benchmark"]["overall"]
    g0 = ov["g0"]
    g1 = ov["g1"]
    assert g1["convergence_rate"] >= g0["convergence_rate"]
    assert g1["hazy_convergence_rate"] > g0["hazy_convergence_rate"]
    assert g1["median_total_error_deg"] <= g0["median_total_error_deg"]
    assert g1["p90_total_error_deg"] < g0["p90_total_error_deg"]
    assert res["verification_matrix"]["V3_B13_g1_vs_g0_real_horizon_improvement"] == "PASS"


def test_v4_b13_hypothesis_h3_synthetic_gap_measured() -> None:
    """V4_B13: Verify Charter Hypothesis H3 — zero-shot synthetic conformal bounds under-cover on real imagery."""
    res = _load_results()
    by_alpha = res["h3_conformal_transfer"]["by_alpha"]
    for a_key in ("0.1", "0.05", "0.01"):
        syn = by_alpha[a_key]["zero_shot_synthetic"]
        assert syn["coverage_gap"] < -0.05
        assert syn["transfers_without_recalibration"] is False
    assert res["verification_matrix"]["V4_B13_hypothesis_h3_synthetic_gap_measured"] == "PASS"


def test_v5_b13_real_conformal_recalibration_restores_coverage() -> None:
    """V5_B13: Verify sequence-disjoint real conformal recalibration restores empirical coverage on real_test."""
    res = _load_results()
    h3 = res["h3_conformal_transfer"]
    assert h3["hypothesis_h3_confirmed"] is True
    for a_key in ("0.1", "0.05", "0.01"):
        mond = h3["by_alpha"][a_key]["real_recalibrated_mondrian"]
        assert mond["pass"] is True
        assert mond["empirical_coverage"] > h3["by_alpha"][a_key]["zero_shot_synthetic"]["empirical_coverage"]
    assert res["verification_matrix"]["V5_B13_real_conformal_recalibration_coverage"] == "PASS"


def test_v6_b13_clean_provenance_and_figures() -> None:
    """V6_B13: Verify run_manifest.json code hashes match and diagnostic figures exist on disk."""
    res = _load_results()
    man = _load_manifest()
    assert man["code_hashes_match"] is True
    assert man["code_hashes_start"] == man["code_hashes_end"]
    assert res["overall_gate_verdict"] == "PASS"
    for fig_rel in res["figures"].values():
        fig_path = REPO_ROOT / fig_rel
        assert fig_path.is_file() and fig_path.stat().st_size > 1024

    # Spot-check deterministic stratified sampling helper
    anns = load_icehorizon_sequence_annotations(REPO_ROOT / "icehorizon_data" / "Annotations" / "drone1.xml")
    sampled = sample_stratified_frames(anns, frames_per_sequence=20)
    assert len(sampled) == 20
