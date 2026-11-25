"""Unit and End-to-End Verification Tests for Batch B14: Cross-WP Gate G1 System Harness.

Verifies `V1_B14..V6_B14` (`docs/xvs/batches/BATCH_B14_G1_SYSTEM_HARNESS.md`):
  - V1_B14: 50-encounter CRN `D0` vs `D1` closed-loop sensitivity + zero-edit `--d1-closure-config`
            drop-in override check with monotonic sub-pixel (`C1_UAS`) confirmation-range gain.
  - V2_B14 & V3_B14: Real WP-07 G1 (`MonitorThresholdsG1` + altitude-Mondrian `ConformalModelG1`)
            sensor-to-autopilot state transitions (`VALID` -> `F_isolated` -> `INVALID_ALERT` -> `VALID`)
            on WP-01 G4 CesiumJS sequences with `0` end-to-end HMI frames.
  - V4_B14: Static AST + runtime `PoisonedTruthTrack` sentinel truth-isolation audit.
  - V5_B14: Single-process (`workers == 1`), wall-time `<= 240 s`, clean manifest (`code_hashes_match == True`,
            `git_status == ""`), comparative PNG verification, and repeated-run byte-identity.
  - V6_B14: Zero-typing report provenance and `tmp_path` regeneration for `WP08_B14_G1_system_harness_report.md`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics.g1_system_harness import (
    evaluate_wp07_g1_end_to_end_transitions,
    file_sha256,
    run_d0_vs_d1_closed_loop_sensitivity,
)
from xvs.report_provenance import find_denylisted_config_keys, verify_report_provenance

sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs" / "avionics"))
from run_g1_end_to_end_system_check import (  # noqa: E402
    PoisonedTruthTrack,
    evaluate_b14_truth_isolation,
)

B14_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "avionics" / "b14_system_spec.json"
B14_OUT_DIR = REPO_ROOT / "output" / "xvs" / "wp08" / "b14_system_harness"
B14_RESULTS_PATH = B14_OUT_DIR / "results.json"
B14_MANIFEST_PATH = B14_OUT_DIR / "run_manifest.json"
B14_PNG_PATH = B14_OUT_DIR / "g1_d0_vs_d1_system_leadtime_hmd.png"
B14_REPORT_PATH = (
    REPO_ROOT / "docs" / "xvs" / "reports" / "WP08_B14_G1_system_harness_report.md"
)


def test_v1_b14_d0_vs_d1_closed_loop_sensitivity_and_override(tmp_path: Path) -> None:
    """V1_B14: 50-encounter D0 vs D1 closed-loop sensitivity and zero-edit `--d1-closure-config` override."""
    spec = json.loads(B14_SPEC_PATH.read_text(encoding="utf-8"))
    assert find_denylisted_config_keys(spec) == []

    assert B14_RESULTS_PATH.is_file(), f"Missing B14 results JSON: {B14_RESULTS_PATH}"
    res = json.loads(B14_RESULTS_PATH.read_text(encoding="utf-8"))
    v1 = res["v1_d0_vs_d1_sensitivity"]

    assert v1["encounters_count"] == spec["benchmark_encounters_count"]
    d0_p = v1["detector_closures"]["D0"]
    d1_p = v1["detector_closures"]["D1"]
    assert d1_p["s50_px"] < d0_p["s50_px"]
    assert d1_p["effective_persistent_lambda"] <= spec["wp04_lambda_req_g1"]

    g_all = v1["paired_gains"]["all_50_d1_vs_d0_b050"]
    g_uas = v1["paired_gains"]["c1_uas_d1_vs_d0_b050"]
    m_d1 = v1["modes_all_50"]["d1_g1_wc1_b050"]
    assert g_all["delta_confirm_range_3d_mean_m"] > 0.0
    assert g_uas["delta_confirm_range_3d_mean_m"] > 200.0
    assert g_uas["delta_lead_time_mean_s"] > 1.0
    assert m_d1["nmac_resolution_rate_pct"] >= spec["min_nmac_resolution_pct"]

    # Zero-edit drop-in Colab A100 refresh check (`--d1-closure-config` / `d1_closure_path` override)
    d0_raw = json.loads(
        (REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json").read_text(
            encoding="utf-8"
        )
    )
    synth_d1_path = tmp_path / "colab_a100_g1_measured_d1.json"
    synth_d1_path.write_text(
        json.dumps(
            {
                **d0_raw,
                "schema": "xvs.tracking.g1_measured_d1/1",
                "s50_px": 0.85,
                "w_ln": 0.35,
                "lambda_ub": 0.0001,
                "persistence_3_fraction": 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    override_res = run_d0_vs_d1_closed_loop_sensitivity(
        d1_closure_path=synth_d1_path,
        max_encounters=6,
    )
    assert override_res["encounters_count"] == 6
    assert override_res["detector_closures"]["D1"]["s50_px"] == pytest.approx(0.85)
    assert override_res["detector_closures"]["D1"]["lambda_ub"] == pytest.approx(0.0001)
    assert (
        override_res["paired_gains"]["all_50_d1_vs_d0_b050"]["delta_confirm_range_3d_mean_m"]
        >= 0.0
    )


def test_v2_v3_b14_real_wp07_g1_transitions_and_zero_hmi() -> None:
    """V2_B14 & V3_B14: Real WP-07 G1 -> HUD -> Bridge -> SITL state transitions and 0 end-to-end HMI."""
    spec = json.loads(B14_SPEC_PATH.read_text(encoding="utf-8"))
    live = evaluate_wp07_g1_end_to_end_transitions(spec=spec)

    assert live["sequences_evaluated_count"] == 2
    assert live["total_loop_frames"] == 16
    assert live["valid_exact_state_count"] == live["valid_frames_count"] == 8
    assert live["f_isolated_succeeded_count"] == live["f_isolated_frames_count"] == 4
    assert live["f_isolated_correct_cam_count"] == 4
    assert live["invalid_alert_detected_count"] == live["invalid_alert_frames_count"] == 4
    assert live["invalid_alert_hud_blanked_count"] == 4
    assert live["invalid_alert_inhibited_count"] == 4
    assert live["invalid_alert_false_turns_count"] == 0
    assert live["ttd_invalid_frames"] <= spec["max_ttd_invalid_frames"]
    assert live["end_to_end_hmi_al_050"] == 0
    assert live["end_to_end_hmi_al_100"] == 0
    assert live["wp07_test_split_reference"]["wp07_total_hmi_al_050"] == 0


def test_v4_b14_strict_truth_isolation() -> None:
    """V4_B14: Static AST + runtime PoisonedTruthTrack sentinel verification."""
    audit = evaluate_b14_truth_isolation()
    assert audit["isolated"] is True
    assert audit["forbidden_imports_count"] == 0
    assert audit["forbidden_attr_refs_count"] == 0
    assert audit["runtime_sentinel_violations"] == 0
    assert audit["total_violations"] == 0

    sentinel = PoisonedTruthTrack()
    with pytest.raises(AssertionError, match="Truth-isolation violation"):
        _ = sentinel.true_range_m


def test_v5_b14_manifest_single_process_and_plot() -> None:
    """V5_B14: Clean manifest, single-process runtime <= 240 s, determinism, and 3-panel PNG artifact."""
    spec = json.loads(B14_SPEC_PATH.read_text(encoding="utf-8"))
    assert B14_MANIFEST_PATH.is_file(), f"Missing B14 manifest: {B14_MANIFEST_PATH}"
    assert B14_RESULTS_PATH.is_file(), f"Missing B14 results: {B14_RESULTS_PATH}"
    assert B14_PNG_PATH.is_file(), f"Missing B14 PNG plot: {B14_PNG_PATH}"

    man = json.loads(B14_MANIFEST_PATH.read_text(encoding="utf-8"))
    res = json.loads(B14_RESULTS_PATH.read_text(encoding="utf-8"))

    assert man["workers"] == spec["required_workers"] == 1
    assert man["code_hashes_match"] is True
    assert man["git_status"] == ""
    assert float(man["elapsed_wall_s"]) <= float(spec["max_wall_time_s"])

    v5 = res["v5_determinism"]
    assert v5["json_byte_identical"] is True
    assert v5["png_byte_identical"] is True

    with Image.open(B14_PNG_PATH) as im:
        assert im.size == (1680, 640)
        assert im.mode == "RGB"
    assert file_sha256(B14_PNG_PATH) == res["plot"]["sha256"]
    assert (
        man["outputs_sha256"][
            "output/xvs/wp08/b14_system_harness/g1_d0_vs_d1_system_leadtime_hmd.png"
        ]
        == res["plot"]["sha256"]
    )


def test_v6_b14_report_provenance_and_regeneration(tmp_path: Path) -> None:
    """V6_B14: Committed B14 report passes verify_report_provenance and regenerates cleanly into tmp_path."""
    assert B14_REPORT_PATH.is_file(), f"Missing B14 report: {B14_REPORT_PATH}"
    assert verify_report_provenance(B14_REPORT_PATH) == []

    tmp_report = tmp_path / "WP08_B14_G1_system_harness_report.md"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'};{REPO_ROOT}"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "xvs" / "avionics" / "report_tables_g1_system_summary.py"),
            "--out",
            str(tmp_report),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"Report regeneration failed:\n{proc.stdout}\n{proc.stderr}"
    assert tmp_report.is_file()
    assert verify_report_provenance(tmp_report) == []
