"""Gate `V2-M3L-1` & `V2-M3L-2` verification: WSL2 `xvs-rt` C++20 Build, Zero-Alloc Audit,
Cross-OS Linux↔Windows FFI Parity (`<= 1e-12`) & 500-Encounter `XVS-MeshDAA` Benchmark.

Invokes WSL2 `make run-all` in `xvs-rt/`, verifies `v1_xvs_rt_evidence.json` and
`v2_meshdaa_500enc_evidence.json`, and enforces `SP3_M3_XVS_RT_BENCH_DESIGN.md` §§3.1 & 4.
"""

from __future__ import annotations

from ctypes import c_double
import json
import math
from pathlib import Path
import subprocess

import pytest

from xvs.core_ffi import XvsCoreFFI, XvsIntruderNed

REPO_ROOT = Path(__file__).resolve().parents[2]
XVS_RT_DIR = REPO_ROOT / "xvs-rt"
DAIDALUS_DIR = REPO_ROOT / "third_party" / "daidalus_boundary"
OUT_M3_DIR = REPO_ROOT / "output" / "xvs" / "v2_m3"
V1_EVIDENCE_PATH = OUT_M3_DIR / "v1_xvs_rt_evidence.json"
V2_EVIDENCE_PATH = OUT_M3_DIR / "v2_meshdaa_500enc_evidence.json"


def _to_wsl_path(win_path: Path) -> str:
    resolved = win_path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _run_wsl_make_run_all(out_dir: Path) -> subprocess.CompletedProcess[str]:
    wsl_rt_dir = _to_wsl_path(XVS_RT_DIR)
    wsl_out_dir = _to_wsl_path(out_dir)
    return subprocess.run(
        [
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "bash",
            "-lc",
            f"cd '{wsl_rt_dir}' && make run-all OUT_M3_DIR='{wsl_out_dir}'",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


@pytest.fixture(scope="module")
def isolated_xvs_rt_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run WSL2 `make run-all` into an isolated tmp_path so GSN-pinned output/xvs/v2_m3/ is never clobbered."""
    out_dir = tmp_path_factory.mktemp("v2_m3_xvs_rt")
    proc = _run_wsl_make_run_all(out_dir)
    return proc, out_dir


def test_daidalus_boundary_isolation_and_zero_internal_enu_in_xvs_rt() -> None:
    notice = (DAIDALUS_DIR / "NOTICE").read_text(encoding="utf-8")
    assert "NOSA 1.3" in notice
    assert "ZERO upstream NASA NOSA source files are bundled" in notice

    # Confirm zero ENU identifiers anywhere inside xvs-rt/ (NED <-> ENU is strictly isolated in bridge.cpp)
    for src_file in list(XVS_RT_DIR.rglob("*.hpp")) + list(XVS_RT_DIR.rglob("*.cpp")):
        text = src_file.read_text(encoding="utf-8")
        assert "ned_to_enu" not in text.lower(), f"Forbidden ENU conversion in {src_file}"
        assert "vec3enu" not in text.lower(), f"Forbidden ENU struct in {src_file}"


def test_wsl2_xvs_rt_build_zero_alloc_o0_o2_and_cross_os_ffi_parity(
    isolated_xvs_rt_run: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    proc, out_dir = isolated_xvs_rt_run
    v1_evidence_path = out_dir / "v1_xvs_rt_evidence.json"
    assert proc.returncode == 0, (
        f"WSL2 make run-all failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "[PASS] test_xvs_rt" in proc.stdout

    assert v1_evidence_path.is_file(), f"Missing {v1_evidence_path}"
    v1 = json.loads(v1_evidence_path.read_text(encoding="utf-8"))
    assert v1["schema"] == "xvs.v2_m3.v1_xvs_rt_evidence/1"
    assert v1["gpu_wcet_status"] == "PENDING_NATIVE_LINUX_GPU_WCET"
    assert v1["ecbf_solver_provenance"] == "STANDALONE_ACTIVE_SET_SOLVER_DAQP_ABI_READY"
    assert v1["daidalus_adapter_mode"] == "STANDALONE_DO365_BOUNDARY_EMULATOR_READY_FOR_UPSTREAM_NOSA"

    # Both -O0 and -O2 zero-alloc audits + positive self-tests MUST pass
    assert int(v1["o0_selftest_alloc_count"]) >= 1
    assert int(v1["o0_hotloop_alloc_count"]) == 0
    assert int(v1["o0_hotloop_alloc_bytes"]) == 0
    assert v1["o0_audit_pass"] is True

    assert int(v1["o2_selftest_alloc_count"]) >= 1
    assert int(v1["o2_selftest_alloc_bytes"]) >= 64
    assert int(v1["o2_hotloop_ticks"]) == 10000
    assert int(v1["o2_hotloop_alloc_count"]) == 0
    assert int(v1["o2_hotloop_alloc_bytes"]) == 0
    assert v1["o2_audit_pass"] is True
    assert int(v1["mavlink_bearing_track_wire_bytes"]) == 77

    # Cross-OS <= 1e-12 FFI parity check: Linux libxvs_core.a vs Windows xvs_core.dll
    ffi = XvsCoreFFI()
    probes = v1["linux_ffi_ecbf_probes"]
    assert len(probes) == 4
    max_diff = 0.0
    for row in probes:
        intr = XvsIntruderNed(
            pos_ned=(c_double * 3)(*row["intr_pos"]),
            vel_ned=(c_double * 3)(*row["intr_vel"]),
            valid=1,
        )
        win_res = ffi.ecbf_qp_solve(
            row["p_own"],
            row["v_own"],
            float(row["nom_bank"]),
            float(row["nom_climb"]),
            [intr],
        )
        d_bank = abs(float(win_res.bank_cmd_rad) - float(row["bank_cmd_rad"]))
        d_climb = abs(float(win_res.climb_rate_cmd_mps) - float(row["climb_rate_cmd_mps"]))
        d_slack = abs(float(win_res.slack) - float(row["slack"]))
        max_diff = max(max_diff, d_bank, d_climb, d_slack)
        assert int(win_res.ecbf_infeasible) == int(row["ecbf_infeasible"])
        assert int(win_res.wc1_alert) == int(row["wc1_alert"])

    assert max_diff <= 1e-12, f"Cross-OS Linux vs Windows xvs_ecbf_qp_solve diff {max_diff:.3e} > 1e-12"

    # Record verified cross-OS parity into isolated tmp_path v1_xvs_rt_evidence.json (never clobber repo output/)
    v1["cross_os_ffi_probes_checked"] = len(probes)
    v1["cross_os_ffi_max_diff"] = max_diff
    v1["cross_os_ffi_parity_pass"] = True
    v1["pass"] = bool(v1["o0_audit_pass"] and v1["o2_audit_pass"] and max_diff <= 1e-12)
    v1_evidence_path.write_text(json.dumps(v1, indent=2) + "\n", encoding="utf-8")
    assert v1["pass"] is True


def test_meshdaa_500_encounter_multi_intruder_benchmark(
    isolated_xvs_rt_run: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    _, out_dir = isolated_xvs_rt_run
    v2_evidence_path = out_dir / "v2_meshdaa_500enc_evidence.json"
    assert v2_evidence_path.is_file(), f"Missing {v2_evidence_path}"
    v2 = json.loads(v2_evidence_path.read_text(encoding="utf-8"))
    assert v2["schema"] == "xvs.v2_m3.v2_meshdaa_500enc_evidence/1"
    assert int(v2["encounters"]) == 500
    assert int(v2["simultaneous_intruders_per_encounter"]) >= 2
    assert int(v2["mavlink_52001_packet_bytes"]) == 77

    # 3-ship MUST achieve 0 NMAC, beat 1-ship on P50 & P95 range error, >= 10x ghost suppression, < 125 kbps
    assert int(v2["three_ship"]["nmac_count"]) == 0
    assert float(v2["three_ship"]["p50_range_err_m"]) < float(v2["one_ship"]["p50_range_err_m"])
    assert float(v2["three_ship"]["p95_range_err_m"]) < float(v2["one_ship"]["p95_range_err_m"])

    g2 = int(v2["ghosts_2ship"])
    g3 = int(v2["ghosts_3ship"])
    expected_ratio = float(g2) / float(max(g3, 1))
    ratio = float(v2["ghost_suppression_ratio"])
    assert math.isfinite(ratio) and abs(ratio - expected_ratio) < 1e-3
    assert ratio >= 10.0, f"Ghost suppression ratio {ratio:.2f}x < 10.0x (g2={g2}, g3={g3})"

    bw = float(v2["bandwidth_kbps_3ship_per_node"])
    assert 0.0 < bw < 125.0, f"Bandwidth {bw:.2f} kbps outside (0, 125)"
    assert v2["pass"] is True
