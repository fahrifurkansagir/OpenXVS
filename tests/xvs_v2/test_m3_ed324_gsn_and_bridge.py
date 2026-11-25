"""OpenXVS v2.0 Milestone M3 — `V2-M3L-4` (`ED-324` GSN v3) & `V2-M3L-5` (ArduPlane Bridge & Compliance) Tests.

Verifies:
1. `ED-324` (`ARP6983`) GSN v3 topology, DAG acyclicity, single root `G0`, `BC-M3-M1` (`100%` `Sn*`
   bound to real on-disk SHA-256 files and `0` `Sn*` marked `undeveloped`), `BC-M3-M2` (strict
   acyclic lower-stage evidence lock), `G5_GAP` & `G2_TLA_DEFERRED` undeveloped Goals with `0`
   fake solutions, Mermaid dashed-red styling, WSL2 `yaml 6.0.3` parse, and adversarial rejections.
2. ArduPlane fixed-wing MAVLink bridge: hard rejection of `#84 SET_POSITION_TARGET_LOCAL_NED`,
   non-finite rejection & bank/airspeed clamping on `COMMAND_INT` / `SET_ATTITUDE_TARGET`,
   `Ekf3SourceSwitchLatch` (`42007`, `>= 2.0 s` dwell, non-monotonic/NaN rejection), and
   `100/100` local UDP loopback signed `77 B` `#52001` frames + 1-bit tamper rejection.
3. Top-level `COMPLIANCE_AND_DUAL_USE.md` structural, civil-scope, license, deferral-tag, and
   zero-overclaim lint.

Writes `output/xvs/v2_m3/v5_bridge_compliance_evidence.json` (and `assurance/*` +
`output/xvs/v2_m3/v4_ed324_gsn_evidence.json` whenever Stage-1 & Stage-2 outputs exist on disk).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.v2.ed324_gsn import (  # noqa: E402
    OPENXVS_V2_SOLUTION_PATHS,
    EvidenceBinding,
    GsnGraph,
    GsnNode,
    GsnNodeType,
    build_openxvs_v2_gsn_graph,
    find_banned_overclaims,
    generate_assurance_bundle,
    validate_gsn_graph,
    validate_yaml_via_wsl2,
)
from xvs.v2.xvs_rt_bridge import (  # noqa: E402
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    DEFAULT_MAX_BANK_RAD,
    DEFAULT_V_MIN_MPS,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    MAVLINK_MSG_COMMAND_INT,
    MAVLINK_MSG_SET_ATTITUDE_TARGET,
    MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED,
    SITL_DEFER_TAG,
    Ekf3SourceSwitchLatch,
    build_command_int_avoidance,
    build_set_attitude_target,
    run_bearing_track_loopback,
    validate_fixed_wing_avoidance_msgid,
)

V2_M3_OUT_DIR = REPO_ROOT / "output" / "xvs" / "v2_m3"
V4_GSN_EVIDENCE_PATH = V2_M3_OUT_DIR / "v4_ed324_gsn_evidence.json"
V5_BRIDGE_EVIDENCE_PATH = V2_M3_OUT_DIR / "v5_bridge_compliance_evidence.json"


def _all_real_gsn_upstream_files_exist() -> bool:
    for rels in OPENXVS_V2_SOLUTION_PATHS.values():
        for rel in rels:
            if not (REPO_ROOT / rel).is_file():
                return False
    return True


def _prepare_fixture_repo(tmp_path: Path) -> Path:
    """Create a minimal fixture repo root combining real M1/M2 files with Stage-1/2 fixtures."""
    root = tmp_path / "fixture_repo"
    root.mkdir(parents=True, exist_ok=True)
    for rels in OPENXVS_V2_SOLUTION_PATHS.values():
        for rel in rels:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = REPO_ROOT / rel
            if src.is_file():
                shutil.copy2(src, dst)
            else:
                dst.write_text(
                    json.dumps({"fixture_for": rel, "status": "UNIT_TEST_FIXTURE"}, indent=2) + "\n",
                    encoding="utf-8",
                )
    return root


def test_ed324_gsn_dag_wsl2_yaml_and_negative_controls(tmp_path: Path) -> None:
    missing_upstream = [
        rel
        for rels in OPENXVS_V2_SOLUTION_PATHS.values()
        for rel in rels
        if not (REPO_ROOT / rel).is_file()
    ]
    assert not missing_upstream, (
        f"Stage-1/2/M1/M2 real upstream evidence files missing (zero synthetic fallback allowed): {missing_upstream}"
    )

    target_root = REPO_ROOT
    target_assurance = tmp_path / "assurance"

    # 1. Generate & verify byte-exact determinism across two consecutive runs into isolated tmp_path
    #    (never mutates frozen repo `assurance/openxvs_v2.gsn.yaml`!)
    res1 = generate_assurance_bundle(target_root, out_assurance_dir=target_assurance)
    res2 = generate_assurance_bundle(target_root, out_assurance_dir=target_assurance)
    assert res1["sha256"] == res2["sha256"], "Assurance bundle emission must be byte-exact deterministic"
    frozen_gsn = REPO_ROOT / "assurance" / "openxvs_v2.gsn.yaml"
    assert Path(res1["files"]["gsn_yaml"]).read_bytes() == frozen_gsn.read_bytes(), (
        "Generated GSN YAML drifted from frozen assurance/openxvs_v2.gsn.yaml"
    )

    val = res1["validation"]
    assert val.valid, f"Expected valid GSN graph, got errors: {val.errors}"
    assert val.node_count == 18
    assert val.solution_count == 8
    assert val.undeveloped_goal_ids == ("G2_TLA_DEFERRED", "G5_GAP")
    assert val.banned_overclaim_hits == 0
    assert val.evidence_bindings_verified == sum(len(v) for v in OPENXVS_V2_SOLUTION_PATHS.values())

    # 2. Verify Mermaid contains dashed-red styling applied to BOTH G5_GAP and G2_TLA_DEFERRED
    mmd_text = Path(res1["files"]["gsn_mermaid"]).read_text(encoding="utf-8")
    assert "stroke-dasharray" in mmd_text and "#c62828" in mmd_text
    assert "class G5_GAP undeveloped;" in mmd_text
    assert "class G2_TLA_DEFERRED undeveloped;" in mmd_text

    # 3. Cross-validate emitted YAML with WSL2 Ubuntu python3 yaml 6.0.3 (yaml.safe_load)
    yaml_path = Path(res1["files"]["gsn_yaml"])
    wsl_yaml = validate_yaml_via_wsl2(yaml_path, expected_node_ids=set(res1["graph"].nodes.keys()))
    assert wsl_yaml["node_count"] == 18
    assert wsl_yaml["undeveloped"] == ["G2_TLA_DEFERRED", "G5_GAP"]
    assert len(wsl_yaml["solutions"]) == 8

    # 4. Adversarial negative controls (BC-M3-M1 & BC-M3-M2) in isolated fixture copy
    neg_root = _prepare_fixture_repo(tmp_path)
    base_graph = build_openxvs_v2_gsn_graph(neg_root)

    # 4a. Reject Solution Sn* marked undeveloped=True
    nodes_bad_sn = dict(base_graph.nodes)
    nodes_bad_sn["Sn1_MHVIT"] = replace(nodes_bad_sn["Sn1_MHVIT"], undeveloped=True)
    v_bad_sn = validate_gsn_graph(replace(base_graph, nodes=nodes_bad_sn), neg_root)
    assert not v_bad_sn.valid and any("BC-M3-M1" in e for e in v_bad_sn.errors)

    # 4b. Reject Solution SHA-256 mismatch
    nodes_bad_sha = dict(base_graph.nodes)
    orig_b = nodes_bad_sha["Sn5_KANI"].evidence_files[0]
    nodes_bad_sha["Sn5_KANI"] = replace(
        nodes_bad_sha["Sn5_KANI"],
        evidence_files=(EvidenceBinding(orig_b.rel_path, "0" * 64, orig_b.size_bytes),),
    )
    v_bad_sha = validate_gsn_graph(replace(base_graph, nodes=nodes_bad_sha), neg_root)
    assert not v_bad_sha.valid and any("SHA-256 mismatch" in e for e in v_bad_sha.errors)

    # 4c. Reject circular Stage-4 / assurance evidence reference (BC-M3-M2)
    fake_stage4 = neg_root / "output" / "xvs" / "v2_m3" / "run_manifest.json"
    fake_stage4.parent.mkdir(parents=True, exist_ok=True)
    fake_stage4.write_text("{}\n", encoding="utf-8")
    nodes_circ = dict(base_graph.nodes)
    nodes_circ["Sn9_PROVENANCE"] = replace(
        nodes_circ["Sn9_PROVENANCE"],
        evidence_files=(
            *nodes_circ["Sn9_PROVENANCE"].evidence_files,
            EvidenceBinding("output/xvs/v2_m3/run_manifest.json", "0" * 64, 3),
        ),
    )
    v_circ = validate_gsn_graph(replace(base_graph, nodes=nodes_circ), neg_root)
    assert not v_circ.valid and any("BC-M3-M2" in e for e in v_circ.errors)

    # 4d. Reject undeveloped Goal with fake children/solutions
    nodes_fake_gap = dict(base_graph.nodes)
    nodes_fake_gap["G5_GAP"] = replace(nodes_fake_gap["G5_GAP"], supported_by=("Sn5_KANI",))
    v_fake_gap = validate_gsn_graph(replace(base_graph, nodes=nodes_fake_gap), neg_root)
    assert not v_fake_gap.valid and any("0 supportedBy" in e for e in v_fake_gap.errors)

    # 4e. Reject cycle in GSN DAG
    nodes_cycle = dict(base_graph.nodes)
    nodes_cycle["G1"] = replace(nodes_cycle["G1"], supported_by=(*nodes_cycle["G1"].supported_by, "G0"))
    v_cycle = validate_gsn_graph(replace(base_graph, nodes=nodes_cycle), neg_root)
    assert not v_cycle.valid and any("Cycle detected" in e for e in v_cycle.errors)

    # 4f. Reject banned regulatory overclaims
    for bad_phrase in ("This is Certifiable AI", "Full DO-178C ML proof", "Provides zero RF emissions"):
        nodes_overclaim = dict(base_graph.nodes)
        nodes_overclaim["G0"] = replace(nodes_overclaim["G0"], text=bad_phrase)
        v_oc = validate_gsn_graph(replace(base_graph, nodes=nodes_overclaim), neg_root)
        assert not v_oc.valid and v_oc.banned_overclaim_hits >= 1

    # 5. Persist isolated `v4_ed324_gsn_evidence.json` into tmp_path (never clobber repo output/)
    out_ev4_path = tmp_path / "v4_ed324_gsn_evidence.json"
    ev4 = {
        "schema": "xvs.v2_m3.v4_ed324_gsn_evidence/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "real_upstream_bound": True,
        "node_count": val.node_count,
        "solution_count": val.solution_count,
        "undeveloped_goal_ids": list(val.undeveloped_goal_ids),
        "evidence_bindings_verified": val.evidence_bindings_verified,
        "banned_overclaim_hits": val.banned_overclaim_hits,
        "wsl2_yaml_version": wsl_yaml["yaml_version"],
        "wsl2_yaml_node_count": wsl_yaml["node_count"],
        "deterministic_regen_verified": True,
        "negative_controls_passed": 6,
        "assurance_sha256": res1["sha256"],
        "pass": True,
    }
    out_ev4_path.write_text(json.dumps(ev4, indent=2) + "\n", encoding="utf-8")
    assert out_ev4_path.is_file()


def test_arduplane_bridge_no_msg84_clamps_nonfinite_and_ekf3_latch() -> None:
    # 1. Hard-reject MAVLink #84 (DR-06:171, 193)
    with pytest.raises(ValueError, match="#84"):
        validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED)
    assert validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_COMMAND_INT) == MAVLINK_MSG_COMMAND_INT
    assert validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_SET_ATTITUDE_TARGET) == MAVLINK_MSG_SET_ATTITUDE_TARGET

    # 2. COMMAND_INT clamping & non-finite rejection
    cmd_clamped = build_command_int_avoidance(
        lat_deg=37.6188,
        lon_deg=-122.3750,
        alt_m=300.0,
        airspeed_mps=10.0,  # below V_min=18.0
        bank_cmd_rad=math.radians(48.0),  # above 30 deg
    )
    assert cmd_clamped.bank_clamped is True
    assert cmd_clamped.airspeed_clamped is True
    assert abs(cmd_clamped.bank_cmd_rad - DEFAULT_MAX_BANK_RAD) < 1e-12
    assert abs(cmd_clamped.airspeed_mps - DEFAULT_V_MIN_MPS) < 1e-12
    assert len(cmd_clamped.payload_bytes) == 35

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            build_command_int_avoidance(lat_deg=bad, lon_deg=-122.3, alt_m=100.0, airspeed_mps=25.0)
        with pytest.raises(ValueError, match="Non-finite"):
            build_command_int_avoidance(lat_deg=37.6, lon_deg=-122.3, alt_m=100.0, airspeed_mps=bad)
        with pytest.raises(ValueError, match="Non-finite"):
            build_command_int_avoidance(lat_deg=37.6, lon_deg=-122.3, alt_m=100.0, airspeed_mps=25.0, bank_cmd_rad=bad)

    # 3. SET_ATTITUDE_TARGET clamping, Hamilton w>=0, & non-finite rejection
    att = build_set_attitude_target(
        roll_rad=math.radians(-55.0),
        pitch_rad=math.radians(30.0),
        yaw_rad=math.radians(280.0),
        thrust_norm=0.75,
        airspeed_mps=12.0,
    )
    assert att.bank_clamped is True
    assert att.airspeed_clamped is True
    assert abs(att.roll_rad + DEFAULT_MAX_BANK_RAD) < 1e-12
    assert abs(att.airspeed_mps - DEFAULT_V_MIN_MPS) < 1e-12
    q = att.q_ned_body
    assert q[0] >= 0.0
    assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-12
    assert len(att.payload_bytes) == 39

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            build_set_attitude_target(roll_rad=bad, pitch_rad=0.0, yaw_rad=0.0, thrust_norm=0.5, airspeed_mps=25.0)
        with pytest.raises(ValueError, match="Non-finite"):
            build_set_attitude_target(roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0, thrust_norm=bad, airspeed_mps=25.0)

    # 4. EKF3 42007 source-switch latch (>= 2.0 s dwell + non-finite/non-monotonic rejection)
    with pytest.raises(ValueError, match=">= 2.0"):
        Ekf3SourceSwitchLatch(min_gnss_revert_dwell_s=1.5)

    latch = Ekf3SourceSwitchLatch(min_gnss_revert_dwell_s=2.0)
    s0 = latch.step(timestamp_s=0.0, gnss_healthy=True, spoof_alarm=False)
    assert s0.active_source == EKF3_SOURCE_GNSS and s0.mav_cmd_id is None

    # Immediate 0-tick switch to Vision on spoof alarm
    s1 = latch.step(timestamp_s=1.0, gnss_healthy=True, spoof_alarm=True)
    assert s1.active_source == EKF3_SOURCE_VISION_OPTICAL
    assert s1.mav_cmd_id == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
    assert s1.commanded_source == EKF3_SOURCE_VISION_OPTICAL

    # Healthy again at t=2.0 -> must stay in Vision until dwell >= 2.0 s (t >= 4.0)
    s2 = latch.step(timestamp_s=2.0, gnss_healthy=True, spoof_alarm=False)
    assert s2.active_source == EKF3_SOURCE_VISION_OPTICAL and s2.mav_cmd_id is None
    s3 = latch.step(timestamp_s=3.9, gnss_healthy=True, spoof_alarm=False)
    assert s3.active_source == EKF3_SOURCE_VISION_OPTICAL and s3.mav_cmd_id is None
    s4 = latch.step(timestamp_s=4.0, gnss_healthy=True, spoof_alarm=False)
    assert s4.active_source == EKF3_SOURCE_GNSS
    assert s4.mav_cmd_id == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
    assert s4.commanded_source == EKF3_SOURCE_GNSS

    # Non-monotonic & non-finite timestamp rejection (BC-M3-m6)
    with pytest.raises(ValueError, match="Non-monotonic"):
        latch.step(timestamp_s=3.95, gnss_healthy=True, spoof_alarm=False)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Non-finite"):
            latch.step(timestamp_s=bad, gnss_healthy=True, spoof_alarm=False)


def test_arduplane_udp_loopback_and_compliance_lint_emit_v5_evidence(tmp_path: Path) -> None:
    # 1. Real 127.0.0.1 UDP loopback of 100 signed 77B #52001 frames + 1-bit tamper rejection
    loopback = run_bearing_track_loopback(n_frames=100)
    assert loopback["pass"] is True
    assert loopback["frame_size_bytes"] == 77
    assert loopback["frames_sent"] == 100
    assert loopback["frames_received"] == 100
    assert loopback["byte_exact_matches"] == 100
    assert loopback["crc_sig_verified"] == 100
    assert loopback["tamper_cases_tested"] == 3
    assert loopback["tamper_cases_rejected"] == 3
    assert loopback["sitl_scope"] == SITL_DEFER_TAG

    # 2. Top-level COMPLIANCE_AND_DUAL_USE.md structural & scope lint
    comp_path = REPO_ROOT / "COMPLIANCE_AND_DUAL_USE.md"
    assert comp_path.is_file(), "COMPLIANCE_AND_DUAL_USE.md missing at repo root"
    comp_text = comp_path.read_text(encoding="utf-8")
    overclaims = find_banned_overclaims(comp_text)
    assert overclaims == [], f"COMPLIANCE_AND_DUAL_USE.md contains banned overclaim(s): {overclaims}"

    required_tokens = [
        "Charter §2",
        "weapons",
        "fire-control",
        "targeting",
        "electronic warfare",
        "zero active radar emissions",
        "D16",
        "DO-178C Table A-7",
        "ED-324",
        "ARP6983",
        "EASA NPA 2025-07",
        "QA-01",
        "G5_GAP",
        "DO-254",
        "DO-330",
        "ARP4754B",
        "RTOS",
        "NASA DAIDALUS v2.0.4",
        "NOSA 1.3",
        "DAQP",
        "CDLA-Permissive-1.0",
        "Copernicus",
        "IRSTD-1k",
        "HIT-UAV",
        "BSD-3-Clause",
        "Research Permission (Eval-Only, Never Redistribute",
        "Charter §6",
        "[PENDING_COLAB_G4_TRAINING]",
        "[PENDING_NATIVE_LINUX_GPU_WCET]",
        "[LOCAL_LOOPBACK_SITL_EMULATOR; FULL_ARDUPILOT_BINARY_DEFERRED_TO_PHASE_4]",
        "[PREWIRED_PHASE4_PENDING_CAPTURE]",
    ]
    comp_lower = comp_text.lower()
    missing = [tok for tok in required_tokens if tok.lower() not in comp_lower]
    assert missing == [], f"COMPLIANCE_AND_DUAL_USE.md missing required governance tokens: {missing}"

    # D16-M3-M1 regression guard: LARD Real Airports MUST NOT be labeled MIT
    lard_lines = [ln for ln in comp_text.splitlines() if "lard" in ln.lower() and "real airports" in ln.lower()]
    assert len(lard_lines) >= 1, "Expected LARD Real Airports row in COMPLIANCE_AND_DUAL_USE.md"
    for ln in lard_lines:
        assert "mit" not in ln.lower(), f"D16-M3-M1 regression: LARD Real Airports row must not say MIT: {ln}"
        assert "research permission (eval-only, never redistribute" in ln.lower(), (
            f"LARD Real Airports row must specify Research Permission (Eval-Only, Never Redistribute...): {ln}"
        )

    # 3. Emit isolated `v5_bridge_compliance_evidence.json` into tmp_path (never clobber repo output/)
    out_ev5_path = tmp_path / "v5_bridge_compliance_evidence.json"
    ev5: dict[str, Any] = {
        "schema": "xvs.v2_m3.v5_bridge_compliance_evidence/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "arduplane_bridge": {
            "msg84_hard_rejected": True,
            "command_int_clamps_and_nonfinite_verified": True,
            "set_attitude_target_clamps_and_nonfinite_verified": True,
            "ekf3_42007_dwell_2s_and_nonmonotonic_verified": True,
            "udp_loopback": loopback,
            "pass": True,
        },
        "compliance_doc": {
            "path": "COMPLIANCE_AND_DUAL_USE.md",
            "banned_overclaim_hits": len(overclaims),
            "required_tokens_checked": len(required_tokens),
            "required_tokens_missing": len(missing),
            "pass": True,
        },
        "pass": True,
    }
    out_ev5_path.write_text(json.dumps(ev5, indent=2) + "\n", encoding="utf-8")
    assert out_ev5_path.is_file()
