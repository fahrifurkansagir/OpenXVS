"""OpenXVS v1.1.0 Verification Suite: Single-Command OEM Verification Gate & Fail-Closed Tamper Proofs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "run_oem_verification_gate.py"


@pytest.fixture(scope="module")
def oem_gate_mod():
    spec = importlib.util.spec_from_file_location("openxvs_oem_gate_test_mod", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openxvs_oem_gate_test_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_oem_verification_gate_cli_all_offline_gates_pass(tmp_path: Path) -> None:
    report_path = tmp_path / "oem_gate_report.json"
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["RAYON_NUM_THREADS"] = "2"
    env["CARGO_BUILD_JOBS"] = "2"

    proc = subprocess.run(
        [sys.executable, str(GATE_SCRIPT), "--json-out", str(report_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"OEM gate failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    assert "OVERALL VERDICT: PASS (6/6 gates passed" in proc.stdout
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "openxvs.v110.oem_verification_gate/1"
    assert report["overall_pass"] is True
    assert report["passed_count"] == 6
    assert report["total_count"] == 6
    assert report["non_flight_sim_only"] is True
    assert report["actuator_authority"] is False
    assert report["pymavlink_imported"] is False
    assert report["elapsed_s"] < 30.0

    by_id = {g["gate_id"]: g for g in report["gates"]}
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6"):
        assert by_id[gid]["passed"] is True, f"{gid} failed: {by_id[gid]}"

    # Spot-check critical invariants across gates
    assert by_id["G1"]["details"]["pymavlink_imported"] is False
    assert by_id["G1"]["details"]["frozen_paths_dirty"] == []
    assert by_id["G2"]["details"]["models_verified_count"] == 6
    assert abs(by_id["G2"]["details"]["dual_blind_vacuity"] - 1.0) <= 1e-6
    assert by_id["G3"]["details"]["tamper_exact_byte_diff"] == 1
    assert by_id["G3"]["details"]["tamper_rust_rc"] != 0
    assert by_id["G4"]["details"]["max_abs_diff"] < 1e-6
    assert by_id["G4"]["details"]["bool_mismatches"] == 0
    assert by_id["G5"]["details"]["leak_hits"] == {}
    assert by_id["G5"]["details"]["non_flight_watermark_present"] is True
    assert by_id["G6"]["details"]["physical_stm32_hardware_verified"] is False
    assert by_id["G6"]["details"]["static_ram_bytes"] == 0
    assert by_id["G6"]["details"]["ticks_passed"] == 10000


def test_gate3_fallback_generates_shadow_tape_when_missing(oem_gate_mod, tmp_path: Path) -> None:
    missing_tape = tmp_path / "does_not_exist.ndjson"
    res = oem_gate_mod.run_gate3_flight_recorder_replay(repo_root=REPO_ROOT, tape_path=missing_tape)
    assert res.passed is True, res.details
    assert res.details["tape_source"] == "generated_deterministic_shadow_tape"
    assert res.details["rust_record_count"] == 24
    assert res.details["tamper_exact_byte_diff"] == 1
    assert res.details["tamper_rust_rc"] != 0


def test_gate3_corrupted_input_tape_fails_closed(oem_gate_mod, tmp_path: Path) -> None:
    tape = tmp_path / "corrupt_input.ndjson"
    oem_gate_mod._generate_deterministic_shadow_tape(tape)
    lines = tape.read_text(encoding="utf-8").splitlines()
    lines[1] = oem_gate_mod._tamper_one_byte_in_payload(lines[1])
    tape.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = oem_gate_mod.run_gate3_flight_recorder_replay(repo_root=REPO_ROOT, tape_path=tape)
    assert res.passed is False
    assert res.details["py_valid"] is False
    assert res.details["rust_replay_rc"] != 0


def test_gate5_studio_binary_leak_and_missing_watermark_fail_closed(oem_gate_mod, tmp_path: Path) -> None:
    # 1. Binary leaking a Windows user-home path in UTF-16LE must FAIL closed
    leaky_exe = tmp_path / "leaky_studio.exe"
    leaky_path = ("C:\\" + "Users\\" + "reviewer\\build").encode("utf-16-le")
    leaky_exe.write_bytes(b"MZ\x90\x00NON_FLIGHT_SIM_ONLY\x00" + leaky_path)
    res_leak = oem_gate_mod.run_gate5_studio_binary_hygiene(repo_root=REPO_ROOT, studio_exe=leaky_exe)
    assert res_leak.passed is False
    assert "windows_user_home_path" in res_leak.details["leak_hits"]

    # 2. Binary missing NON_FLIGHT_SIM_ONLY watermark must FAIL closed
    no_wm_exe = tmp_path / "no_watermark_studio.exe"
    no_wm_exe.write_bytes(b"MZ\x90\x00clean_binary_without_watermark\x00")
    res_wm = oem_gate_mod.run_gate5_studio_binary_hygiene(repo_root=REPO_ROOT, studio_exe=no_wm_exe)
    assert res_wm.passed is False
    assert res_wm.details["non_flight_watermark_present"] is False


def test_gate6_cortex_m7_ledger_falsified_hw_or_violations_fail_closed(oem_gate_mod, tmp_path: Path) -> None:
    base_doc = json.loads(
        (REPO_ROOT / "outputs" / "formal_tla" / "cortex_m7_emu_exec_proof.json").read_text(encoding="utf-8")
    )

    # 1. Falsifying physical_stm32_hardware_verified=True must FAIL closed
    bad_hw = json.loads(json.dumps(base_doc))
    bad_hw["scope_flags"]["physical_stm32_hardware_verified"] = True
    p_hw = tmp_path / "bad_hw.json"
    p_hw.write_text(json.dumps(bad_hw), encoding="utf-8")
    res_hw = oem_gate_mod.run_gate6_cortex_m7_proof_ledger(repo_root=REPO_ROOT, m7_proof_path=p_hw)
    assert res_hw.passed is False

    # 2. Any invariant violation must FAIL closed
    bad_inv = json.loads(json.dumps(base_doc))
    bad_inv["execution_results"]["invariant_violations"] = 1
    p_inv = tmp_path / "bad_inv.json"
    p_inv.write_text(json.dumps(bad_inv), encoding="utf-8")
    res_inv = oem_gate_mod.run_gate6_cortex_m7_proof_ledger(repo_root=REPO_ROOT, m7_proof_path=p_inv)
    assert res_inv.passed is False


def test_v110_docs_and_gate_script_zero_host_paths_and_zero_dollars() -> None:
    host_re = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/Users/[A-Za-z0-9._-]+/", re.IGNORECASE)
    for rel in (
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/VERIFICATION.md",
        "docs/LIMITATIONS.md",
    ):
        txt = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "$" not in txt, f"Unescaped '$' found in {rel}"
        assert host_re.search(txt) is None, f"Host path leak found in {rel}"

    gate_txt = GATE_SCRIPT.read_text(encoding="utf-8")
    assert host_re.search(gate_txt) is None, "Host path leak found in run_oem_verification_gate.py"
