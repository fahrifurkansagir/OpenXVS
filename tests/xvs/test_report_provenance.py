"""Tests for G-B14 W5 / Step 2: Report provenance tracking, verification, and negative controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pytest

from xvs.report_provenance import (
    ProvenanceTracker,
    find_denylisted_config_keys,
    register_json_source,
    rv,
    save_provenance,
    verify_report_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs" / "xvs"
ALL_CONFIG_REL_PATHS = [
    p.relative_to(REPO_ROOT).as_posix() for p in sorted(CONFIGS_DIR.rglob("*.json"))
]


def _normalize_report_for_byte_compare(text: str) -> str:
    """Mask out dynamic pytest timing (e.g. 'in 1.23s') and rootdir lines in §4 Test output."""
    lines = []
    for line in text.splitlines():
        if line.startswith("rootdir:"):
            lines.append("rootdir: <NORMALIZED>")
            continue
        line = re.sub(
            r"^(=+\s*)(.*?)in \d+\.\d+s(?:\s*\([^)]*\))?(\s*=+)?$",
            lambda m: f"=== {m.group(2)}in <TIME>s ===" if m.group(1) else f"{m.group(2)}in <TIME>s",
            line,
        )
        lines.append(line)
    return "\n".join(lines)


def test_rv_rejects_bare_literals(tmp_path: Path) -> None:
    rep_file = tmp_path / "reject_report.md"
    with ProvenanceTracker(rep_file):
        with pytest.raises(TypeError, match="numeric/boolean literals"):
            rv(12.34, "rate", ".2f")
        with pytest.raises(TypeError, match="numeric/boolean literals"):
            rv(100, "count", "d")
        with pytest.raises(TypeError, match="numeric/boolean literals"):
            rv(True, "verdict", "verdict")
        with pytest.raises(TypeError, match="string literals"):
            rv("PASS", "verdict", "verdict")


def test_no_literal_rv_calls_in_generators() -> None:
    rx = re.compile(r"rv\(\s*-?[0-9]")
    scripts_dir = REPO_ROOT / "scripts" / "xvs"
    matches: list[str] = []
    for py_file in sorted(scripts_dir.rglob("*.py")):
        txt = py_file.read_text(encoding="utf-8")
        for m in rx.finditer(txt):
            matches.append(f"{py_file.relative_to(REPO_ROOT)}:{m.group(0)}")
    assert not matches, f"Found literal rv() calls in generators: {matches}"


def test_config_denylist_helper_catches_all_denylisted_patterns() -> None:
    sample_bad_config = {
        "meas_read_noise_e": 2.4,
        "foo_meas_bar": 1.0,
        "item_1_verdict": "PASS",
        "verdicts": {"v1": "PASS"},
        "val_pos_count_meas": 450,
        "nees_in_bounds": 94.4,
        "nis_avg": 3.04,
        "v1_pd": 1,
        "my_v7_agreement": 100,
        "solver_vs_edge": {"p50": 0.01},
        "max_hmd_bound_m": 495.5,
        "false_det_8cam": 221,
        "deg_p50_deg": 0.4,
        "fa_per_mpix_mean": 1.2,
        "cam_fa_rate": 0.5,
        "ratio_min": 5.3,
        "err_ratio_val": 1.1,
        "v0_v01_comparison": {},
        "hmi_05_faulty_k_sum": 129,
        "provisional_lambda_req": 0.5,
        # Allowed simulator inputs:
        "spec_filter_bound_pct": 90,
        "fa_size_min_px": 1.5,
        "max_leave_one_out_deg_reason": "threshold",
    }
    violations = find_denylisted_config_keys(sample_bad_config)
    expected = {
        "meas_read_noise_e",
        "foo_meas_bar",
        "item_1_verdict",
        "verdicts",
        "val_pos_count_meas",
        "nees_in_bounds",
        "nis_avg",
        "v1_pd",
        "my_v7_agreement",
        "solver_vs_edge",
        "max_hmd_bound_m",
        "false_det_8cam",
        "deg_p50_deg",
        "fa_per_mpix_mean",
        "cam_fa_rate",
        "ratio_min",
        "err_ratio_val",
        "v0_v01_comparison",
        "hmi_05_faulty_k_sum",
        "provisional_lambda_req",
    }
    assert set(violations) == expected


@pytest.mark.parametrize("config_rel_path", ALL_CONFIG_REL_PATHS)
def test_configs_contain_no_denylisted_result_keys(config_rel_path: str) -> None:
    """Fail if a config file under configs/xvs/ contains any key in the result-like denylist (F3)."""
    cfg_path = REPO_ROOT / config_rel_path
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    violations = find_denylisted_config_keys(data)
    assert not violations, f"Config {config_rel_path} contains denylisted result/verdict keys: {violations}"


def test_rv_helper_and_tracker(tmp_path: Path) -> None:
    rep_file = tmp_path / "test_report.md"
    src_obj = register_json_source({"rate": 12.3456, "count": 42, "ok": True}, "configs/xvs/test.json")
    with ProvenanceTracker(rep_file):
        val1 = rv(src_obj, "rate", ".2f")
        val2 = rv(src_obj, "count", "d")
        verdict = rv(src_obj, "ok", "verdict")
        assert val1 == "12.35"
        assert val2 == "42"
        assert verdict == "PASS"
        rep_file.write_text(f"# Test\nRate: {val1}, Count: {val2}, Verdict: **{verdict}**\n", encoding="utf-8")

    save_provenance(rep_file)
    prov_file = tmp_path / "test_report.md.provenance.json"
    assert prov_file.is_file()
    data = json.loads(prov_file.read_text(encoding="utf-8"))
    assert "12.35" in data["numbers"]
    assert "42" in data["numbers"]
    assert len(data["number_positions"]) == 2
    assert len(data["verdict_positions"]) == 1
    assert verify_report_provenance(rep_file) == []


def test_provenance_negative_control(tmp_path: Path) -> None:
    rep_file = tmp_path / "sample_report.md"
    src_obj = register_json_source({"metric": 99.5}, "configs/xvs/test.json")

    with ProvenanceTracker(rep_file):
        val = rv(src_obj, "metric", ".1f")
        rep_file.write_text(f"# Test Report\n\nMetric was {val} %.\n", encoding="utf-8")
    save_provenance(rep_file)

    # Overwrite report with extra typed numbers that were never recorded
    rep_file.write_text(
        f"# Test Report\n\nMetric was {val} % but unrecorded speed was 42.77 m/s and unrecorded rate was 18.5 %.\n",
        encoding="utf-8",
    )

    unprov = verify_report_provenance(rep_file)
    assert any("42.77" in u for u in unprov), f"Expected 42.77 to be detected as unprovenanced, got: {unprov}"
    assert any("18.5" in u for u in unprov), f"Expected 18.5 to be detected as unprovenanced, got: {unprov}"


def test_provenance_negative_control_integer(tmp_path: Path) -> None:
    rep_file = tmp_path / "sample_report_int.md"
    src_obj = register_json_source({"metric": 100}, "configs/xvs/test.json")
    with ProvenanceTracker(rep_file):
        val = rv(src_obj, "metric", "d")
        rep_file.write_text(f"# Test Report\n\nMetric was {val}.\n", encoding="utf-8")
    save_provenance(rep_file)

    rep_file.write_text(
        f"# Test Report\n\nMetric was {val} but unrecorded count was 98765 instances.\n",
        encoding="utf-8",
    )

    unprov = verify_report_provenance(rep_file)
    assert any("98765" in u for u in unprov), f"Expected 98765 to be detected as unprovenanced, got: {unprov}"


@pytest.mark.parametrize(
    "report_name",
    [
        "WP03_G0_report.md",
        "WP04_G0_report.md",
        "WP04_G1_report.md",
        "WP05_G0_report.md",
        "WP05_G1_report.md",
        "WP06_G0_report.md",
        "WP07_G0_report.md",
        "WP07_G1_report.md",
        "WP07_H3_real_transfer_report.md",
        "WP08_G0_report.md",
        "WP08_B10_tier0_report.md",
        "WP08_G1_report.md",
        "WP08_B14_G1_system_harness_report.md",
    ],
)
def test_all_committed_reports_pass_provenance(report_name: str) -> None:
    rep_path = REPO_ROOT / "docs" / "xvs" / "reports" / report_name
    assert rep_path.is_file(), f"Report file not found: {rep_path}"
    prov_path = rep_path.parent / f"{report_name}.provenance.json"
    assert prov_path.is_file(), f"Provenance file not found: {prov_path}"

    unprov = verify_report_provenance(rep_path)
    assert not unprov, f"Report {report_name} has unprovenanced tokens: {unprov[:20]}"


@pytest.mark.parametrize(
    "report_name,script_rel,out_flag",
    [
        ("WP03_G0_report.md", "scripts/xvs/sensor/report_tables_wp03.py", "--out-report"),
        ("WP04_G0_report.md", "scripts/xvs/detection/report_tables_wp04.py", "--out"),
        ("WP04_G1_report.md", "scripts/xvs/detection/report_tables_wp04_g1.py", "--out"),
        ("WP05_G0_report.md", "scripts/xvs/tracking/report_tables_wp05.py", "--out"),
        ("WP05_G1_report.md", "scripts/xvs/tracking/report_tables_wp05_g1.py", "--out"),
        ("WP06_G0_report.md", "scripts/xvs/horizon_att/report_tables_wp06.py", "--out"),
        ("WP07_G0_report.md", "scripts/xvs/integrity/report_tables_wp07.py", "--report-path"),
        ("WP07_G1_report.md", "scripts/xvs/integrity/report_tables_wp07_g1.py", "--report-path"),
        ("WP07_H3_real_transfer_report.md", "scripts/xvs/integrity/report_tables_wp07_h3_real_transfer.py", "--report-path"),
        ("WP08_G0_report.md", "scripts/xvs/display/report_tables_wp08.py", "--out"),
        ("WP08_B10_tier0_report.md", "scripts/xvs/avionics/report_tables_tier0_b10.py", "--out"),
        ("WP08_G1_report.md", "scripts/xvs/display/report_tables_wp08_g1.py", "--out"),
        ("WP08_B14_G1_system_harness_report.md", "scripts/xvs/avionics/report_tables_g1_system_summary.py", "--out"),
    ],
)
def test_regenerate_report_into_tmp_path_passes_provenance(
    tmp_path: Path, report_name: str, script_rel: str, out_flag: str
) -> None:
    """Verify regenerating each of the reports into tmp_path passes provenance and matches committed report."""
    out_file = tmp_path / report_name
    script = REPO_ROOT / script_rel
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'};{REPO_ROOT}"
    if "XVS_DATA_ROOT" not in env:
        worktree_local_reports = {
            "WP08_B10_tier0_report.md",
            "WP08_G1_report.md",
            "WP08_B14_G1_system_harness_report.md",
        }
        common_git = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        common_out: Path | None = None
        if common_git.returncode == 0 and common_git.stdout.strip():
            cand_common = (REPO_ROOT / common_git.stdout.strip()).resolve().parent / "output"
            if cand_common.is_dir():
                common_out = cand_common

        if report_name not in worktree_local_reports and common_out is not None:
            env["XVS_DATA_ROOT"] = str(common_out)
        elif (REPO_ROOT / "output").is_dir():
            env["XVS_DATA_ROOT"] = str(REPO_ROOT / "output")

    cmd = [sys.executable, str(script), out_flag, str(out_file)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    assert res.returncode == 0, f"Generator {script_rel} failed with {res.returncode}:\n{res.stdout}\n{res.stderr}"

    assert out_file.is_file()
    prov_file = tmp_path / f"{report_name}.provenance.json"
    assert prov_file.is_file()

    unprov = verify_report_provenance(out_file)
    assert not unprov, f"Regenerated {report_name} has unprovenanced tokens: {unprov[:20]}"

    committed_file = REPO_ROOT / "docs" / "xvs" / "reports" / report_name
    gen_norm = _normalize_report_for_byte_compare(out_file.read_text(encoding="utf-8"))
    com_norm = _normalize_report_for_byte_compare(committed_file.read_text(encoding="utf-8"))
    assert gen_norm == com_norm, f"Regenerated {report_name} differs from committed report"


@pytest.mark.parametrize(
    "report_name,old_str,new_str,label",
    [
        ("WP05_G0_report.md", "661/1035 (63.86%", "662/1035 (63.96%", "swap to another provenanced value"),
        ("WP05_G0_report.md", "661/1035 (63.86%", "123/1035 (11.88%", "typed value that may be absent"),
        ("WP05_G0_report.md", ", FAIL) |", ", PASS) |", "flip crit-3 verdict FAIL -> PASS"),
        (
            "WP07_G0_report.md",
            "| 0.50deg | Standard | **OVERALL** | **overall** | 129 |",
            "| 0.50deg | Standard | **OVERALL** | **overall** | 0 |",
            "HMI k=129 -> 0",
        ),
        ("WP07_G0_report.md", "186 | 24300 | 0.765%", "123 | 24300 | 0.506%", "FAR 186 -> 123"),
    ],
)
def test_r08_hand_edits_caught(tmp_path: Path, report_name: str, old_str: str, new_str: str, label: str) -> None:
    """Verify all 5 r08 hand edits to committed reports are caught by verify_report_provenance."""
    reports_dir = REPO_ROOT / "docs" / "xvs" / "reports"
    src_rep = reports_dir / report_name
    src_prov = reports_dir / f"{report_name}.provenance.json"
    dst_rep = tmp_path / report_name
    dst_prov = tmp_path / f"{report_name}.provenance.json"

    shutil.copy(src_rep, dst_rep)
    shutil.copy(src_prov, dst_prov)

    assert verify_report_provenance(dst_rep) == []

    txt = dst_rep.read_text(encoding="utf-8")
    assert old_str in txt, f"Target string {old_str!r} not found in {report_name} ({label})"
    dst_rep.write_text(txt.replace(old_str, new_str, 1), encoding="utf-8")

    unprov = verify_report_provenance(dst_rep)
    assert bool(unprov), f"Expected r08 edit ({label}) to be caught by verify_report_provenance, got empty list"
