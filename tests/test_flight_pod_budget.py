"""Wave 2 Track 2 Tests: Flight-Pod SWaP, Bare-Metal MCU & Telemetry Bandwidth Auditor.

Validates `scripts/xvs/verify_flight_pod_budget.py` against real on-disk OpenXVS artifacts:
  - `outputs/formal_tla/ring0_thumbv7em_mcu_proof.json` (`188,166 B` = `8.9725% <= 15%` of 2 MiB Flash, `0 B` static RAM)
  - All 6 active SHA-256-pinned ONNX perception weights on disk
  - All 4 `docker/*.Dockerfile` profiles (`rpi5`, `orin_nano_super_jp62`, `orin_nx_jp62`, `orin_nx_jp72`)
  - All 3 deployment profiles (Tier-0B Bench, Tier-1 EO Shadow Pod, Tier-2 Dual EO+LWIR Pod)
  - Strict evidence-tagging, zero REL1 host path leaks, and fail-closed rejection on tampering/over-budget.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "xvs"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_flight_pod_budget import (  # noqa: E402
    ACTIVE_SIX_ONNX_SLOTS,
    DEFAULT_MCU_PROOF_REL,
    EVIDENCE_UNVERIFIED_FLIGHT_HW,
    EVIDENCE_UNVERIFIED_VENDOR_SPEC,
    EVIDENCE_VERIFIED_FORMAL_OR_SIM,
    EVIDENCE_VERIFIED_ON_SILICON,
    REPORT_SCHEMA,
    VALID_EVIDENCE_TAGS,
    audit_dockerfile_profile,
    audit_ring0_mcu_proof,
    audit_uart_telemetry_budget,
    main as budget_cli_main,
    run_flight_pod_budget_audit,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (  # noqa: E402
    MAX_SIGNED_UART_BYTES_PER_SEC,
    MavlinkStreamSpec,
)


def _walk_evidence_tags(node: object, out: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k.endswith("evidence_tag"):
                assert isinstance(v, str)
                out.add(v)
            else:
                _walk_evidence_tags(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_evidence_tags(item, out)


def test_real_on_disk_flight_pod_budget_audit_all_three_profiles_pass() -> None:
    """Run deterministic audit over real on-disk artifacts and verify all 3 profiles & evidence tags."""
    report = run_flight_pod_budget_audit(repo_root=REPO_ROOT)

    assert report["schema"] == REPORT_SCHEMA
    assert report["overall_pass"] is True
    assert report["non_flight_sim_only"] is True
    assert report["actuator_authority"] is False
    assert report["gpl_free"] is True
    assert "NED" in report["coordinate_frame"]

    # Strict JSON finite serialization & zero REL1 host path leaks
    raw_json = json.dumps(report, indent=2, allow_nan=False)
    for forbidden in sorted(
        {
            str(REPO_ROOT),
            REPO_ROOT.as_posix(),
            str(Path.home()),
            Path.home().as_posix(),
        }
    ):
        assert re.search(re.escape(forbidden), raw_json, re.IGNORECASE) is None, (
            f"REL1 host path leak found in flight_pod_budget_report: {forbidden!r}"
        )

    # All evidence tags strictly belong to canonical 4-tag taxonomy (and all 4 are represented)
    tags: set[str] = set()
    _walk_evidence_tags(report, tags)
    assert tags.issubset(VALID_EVIDENCE_TAGS)
    assert EVIDENCE_VERIFIED_ON_SILICON in tags
    assert EVIDENCE_VERIFIED_FORMAL_OR_SIM in tags
    assert EVIDENCE_UNVERIFIED_FLIGHT_HW in tags
    assert EVIDENCE_UNVERIFIED_VENDOR_SPEC in tags

    # Real on-disk 6 ONNX weights verified (promoted v5.2 Land + Thermal + Faz-B Nav Triad)
    onnx = report["on_disk_artifacts"]["active_6_onnx_weights"]
    assert onnx["pass"] is True
    assert onnx["active_model_count"] == 6
    assert onnx["verified_model_count"] == 6
    assert 25_000_000 <= onnx["total_onnx_bytes"] <= 35_000_000
    assert onnx["evidence_tag"] == EVIDENCE_VERIFIED_ON_SILICON
    models_by_slot = {m["slot_id"]: m for m in onnx["models"]}
    assert models_by_slot["xvs_land"]["rel_path"].endswith("xvs_land_v52_fp16.onnx")
    assert (
        models_by_slot["xvs_land"]["sha256"]
        == "834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799"
    )
    assert models_by_slot["xvs_thermal"]["rel_path"].endswith("xvs_thermal_v52_fp16.onnx")
    assert (
        models_by_slot["xvs_thermal"]["sha256"]
        == "bc6ceac6252f48dc52305c4e7955cb7841bc5adc7c191a59ce9b0e36ef10fcac"
    )
    assert models_by_slot["xvs_nav_s1"]["rel_path"].endswith("xvs_nav_v5_fazb_fp16.onnx")
    assert (
        models_by_slot["xvs_nav_s1"]["sha256"]
        == "e48f0ed5a234dcf1661031b80db5028d8c5b926b7b54e06575b4ed20bd6fc46f"
    )

    # Real on-disk STM32H7 Ring-0 MCU proof verified
    mcu = report["on_disk_artifacts"]["ring0_thumbv7em_mcu_proof"]
    assert mcu["pass"] is True
    assert mcu["flash_bytes"] == 188_166
    assert mcu["flash_utilization_pct"] == pytest.approx(8.9725, abs=1e-3)
    assert mcu["flash_within_15pct_cap"] is True
    assert mcu["static_ram_bytes"] == 0
    assert mcu["zero_static_ram_verified"] is True
    assert mcu["zero_std_and_zero_alloc_verified"] is True

    # ---- Profile 1: Tier-0B Bench (RPi 5 8GB) ----
    t0 = report["profiles"]["tier_0b_bench_rpi5_8gb"]
    assert t0["pass"] is True
    assert t0["classification"] == "NOT_FLIGHT_CAPABLE"
    assert t0["flight_capable"] is False
    assert t0["swap_budget"]["mass_within_300g_cap"] is True
    assert t0["swap_budget"]["mass_max_g"] <= 300.0
    assert t0["swap_budget"]["power_within_15w_cap"] is True
    assert t0["swap_budget"]["power_max_w"] <= 15.0
    assert t0["compliance"]["non_flight_sim_only_locked"] is True
    assert t0["compliance"]["no_alpine_musl"] is True

    # ---- Profile 2: Tier-1 EO-Only Shadow Pod (Orin Nano Super 8GB + STM32H7) ----
    t1 = report["profiles"]["tier_1_eo_shadow_orin_nano_super_8gb"]
    assert t1["pass"] is True
    assert t1["compute"]["compute_tops"] == 67
    assert t1["ram_budget_8gb"]["app_ram_cap_mib"] == pytest.approx(6144.0)
    assert t1["ram_budget_8gb"]["trt_workspace_cap_mib"] == pytest.approx(1024.0)
    assert t1["ram_budget_8gb"]["fits_within_6144_mib_cap"] is True
    assert t1["ram_budget_8gb"]["projected_6_onnx_plus_workspace_mib"] <= 6144.0
    assert t1["ram_budget_8gb"]["vlm_2b_policy"]["status"] == "EXCLUDED_8GB_CAP"
    assert t1["ram_budget_8gb"]["vlm_2b_policy"]["admitted"] is False
    assert t1["ram_budget_8gb"]["vlm_2b_policy"]["projected_combined_with_vlm_2b_mib"] > 6144.0
    assert t1["ring0_mcu_budget"]["flash_within_15pct_cap"] is True
    assert t1["ring0_mcu_budget"]["flash_bytes"] == 188_166
    assert t1["ring0_mcu_budget"]["static_ram_bytes"] == 0
    assert t1["compliance"]["non_flight_sim_only_locked"] is True
    assert t1["compliance"]["nvidia_tensorrt_sla_sec_1_2_and_2_ok"] is True

    # ---- Profile 3: Tier-2 Dual EO+LWIR Pod (Orin NX 16GB + STM32H7/R5 + <=9 Hz Boson+ 320) ----
    t2 = report["profiles"]["tier_2_dual_eo_lwir_orin_nx_16gb"]
    assert t2["pass"] is True
    assert t2["swap_margin_audit"]["mass_min_g"] == pytest.approx(255.0)
    assert t2["swap_margin_audit"]["mass_max_g"] == pytest.approx(350.0)
    assert t2["swap_margin_audit"]["mass_margin_verdict"] == "MARGINAL_BARE_CORE_REQUIRED"
    assert t2["swap_margin_audit"]["power_rail_10w_nvpmodel_w"] <= 15.0
    assert t2["swap_margin_audit"]["power_rail_15w_nvpmodel_w"] > 15.0
    assert t2["swap_margin_audit"]["nvpmodel_10w_mandatory_for_sub_15w"] is True
    assert t2["lwir_export_audit"]["frame_rate_hz"] <= 9.0
    assert t2["lwir_export_audit"]["compliant_le_9hz"] is True
    assert t2["uart_921k_20pct_audit"]["cap_bytes_per_sec"] == pytest.approx(MAX_SIGNED_UART_BYTES_PER_SEC)
    assert t2["uart_921k_20pct_audit"]["signed_bidir_bytes_per_sec"] <= MAX_SIGNED_UART_BYTES_PER_SEC
    assert t2["uart_921k_20pct_audit"]["signed_bidir_utilization_pct"] < 20.0
    assert t2["uart_921k_20pct_audit"]["shadow_rx_only_bytes_per_sec"] < MAX_SIGNED_UART_BYTES_PER_SEC
    assert t2["uart_921k_20pct_audit"]["within_20pct_cap"] is True
    assert t2["compliance"]["non_flight_sim_only_locked"] is True
    assert t2["compliance"]["actuator_authority"] is False
    assert t2["compliance"]["nvidia_tensorrt_sla_sec_1_2_and_2_ok"] is True


def test_cli_generates_json_report_and_exits_zero(tmp_path: Path) -> None:
    """CLI entrypoint writes valid machine-readable JSON report and returns exit code 0."""
    out_file = tmp_path / "flight_pod_budget_report.json"
    rc = budget_cli_main(["--repo-root", str(REPO_ROOT), "--out", str(out_file)])
    assert rc == 0
    assert out_file.is_file()
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    assert doc["overall_pass"] is True
    assert set(doc["profiles"].keys()) == {
        "tier_0b_bench_rpi5_8gb",
        "tier_1_eo_shadow_orin_nano_super_8gb",
        "tier_2_dual_eo_lwir_orin_nx_16gb",
    }

    # Subprocess smoke test matching exact command invocation
    out_sub = tmp_path / "sub_report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "verify_flight_pod_budget.py"), "--out", str(out_sub)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "overall_pass=True" in proc.stdout
    assert "EXCLUDED_8GB_CAP" in proc.stdout
    assert out_sub.is_file()


def test_auditor_fails_closed_on_mcu_overflow_bad_dockerfile_uart_saturation_and_tampered_onnx(
    tmp_path: Path,
) -> None:
    """Fail-closed negative tests for MCU flash/RAM cap, TRT SLA, UART saturation, & tampered ONNX."""
    # 1. MCU proof exceeding 15% Flash OR non-zero static RAM fails closed
    real_mcu = json.loads((REPO_ROOT / DEFAULT_MCU_PROOF_REL).read_text(encoding="utf-8"))
    bad_flash_path = tmp_path / "bad_flash_mcu.json"
    bad_flash = json.loads(json.dumps(real_mcu))
    bad_flash["aggregate_elf32_arm_totals_bytes"]["text_bytes"] = 350_000
    bad_flash["aggregate_elf32_arm_totals_bytes"]["total_flash_text_rodata_data_bytes"] = 362_374
    bad_flash_path.write_text(json.dumps(bad_flash), encoding="utf-8")
    res_flash = audit_ring0_mcu_proof(bad_flash_path, repo_root=tmp_path)
    assert res_flash["pass"] is False
    assert res_flash["flash_within_15pct_cap"] is False

    bad_ram_path = tmp_path / "bad_ram_mcu.json"
    bad_ram = json.loads(json.dumps(real_mcu))
    bad_ram["aggregate_elf32_arm_totals_bytes"]["bss_bytes"] = 128
    bad_ram["aggregate_elf32_arm_totals_bytes"]["total_static_ram_data_bss_bytes"] = 128
    bad_ram_path.write_text(json.dumps(bad_ram), encoding="utf-8")
    res_ram = audit_ring0_mcu_proof(bad_ram_path, repo_root=tmp_path)
    assert res_ram["pass"] is False
    assert res_ram["zero_static_ram_verified"] is False

    # 2. Dockerfile missing NON_FLIGHT_SIM_ONLY=1 or NVIDIA TRT SLA Sec. 1.2/2 fails closed
    bad_df = tmp_path / "orin_nx_bad.Dockerfile"
    bad_df.write_text("FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime\nENV FOO=1\n", encoding="utf-8")
    res_df = audit_dockerfile_profile(bad_df, repo_root=tmp_path, requires_nvidia_trt_sla=True)
    assert res_df["pass"] is False
    assert res_df["non_flight_sim_only_locked"] is False
    assert res_df["nvidia_trt_sla_sec_1_2_and_2_ok"] is False

    # 3. Alpine musl FROM in rpi5 Dockerfile fails closed
    bad_rpi = tmp_path / "rpi5_alpine.Dockerfile"
    bad_rpi.write_text(
        "FROM --platform=linux/arm64 alpine:3.20\nENV NON_FLIGHT_SIM_ONLY=1 OPENXVS_NON_FLIGHT_SIM_ONLY=1\n",
        encoding="utf-8",
    )
    res_rpi = audit_dockerfile_profile(
        bad_rpi,
        repo_root=tmp_path,
        requires_nvidia_trt_sla=False,
        forbid_alpine_musl=True,
    )
    assert res_rpi["pass"] is False
    assert res_rpi["no_alpine_musl"] is False

    # 4. Saturated UART (> 20% = 18,432 B/s) fails closed
    saturated_streams = (
        MavlinkStreamSpec("ODOMETRY", 331, "outbound", 232, 80.0),  # 80 Hz * 257 B = 20,560 B/s > 18,432 B/s
    )
    res_uart = audit_uart_telemetry_budget(repo_root=REPO_ROOT, signed_streams=saturated_streams)
    assert res_uart["pass"] is False
    assert res_uart["signed_bidirectional"]["within_budget"] is False

    # 5. Tampered ONNX model fails closed in full audit & CLI returns exit code 1
    fake_root = tmp_path / "tampered_repo"
    (fake_root / "outputs" / "formal_tla").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / DEFAULT_MCU_PROOF_REL, fake_root / DEFAULT_MCU_PROOF_REL)
    shutil.copytree(REPO_ROOT / "docker", fake_root / "docker")
    # Put tampered dummy bytes for the primary candidate of every ONNX slot
    for slot in ACTIVE_SIX_ONNX_SLOTS:
        dst = fake_root / slot.candidates[0].rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"tampered-onnx-payload")

    bad_out = tmp_path / "tampered_report.json"
    rc_bad = budget_cli_main(["--repo-root", str(fake_root), "--out", str(bad_out)])
    assert rc_bad == 1
    bad_doc = json.loads(bad_out.read_text(encoding="utf-8"))
    assert bad_doc["overall_pass"] is False
    assert bad_doc["on_disk_artifacts"]["active_6_onnx_weights"]["pass"] is False
