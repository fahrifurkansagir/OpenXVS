"""Tests for OpenXVS V5.0 Formal TLA+ Specs, Rust <-> TLA+ NDJSON Trace Refinement,
`Sn4_TLA` Evidence Schema, and Non-Breaking GSN v3 Proposal (`DR-22`).

Verifies:
  1. All TLA+ specs & configs exist under `formal/tla/` with exact frozen Rust constants.
  2. Standalone Rust trace emitter (`tools/xvs-tla-trace`) passes `cargo test` cleanly.
  3. `outputs/formal_tla/sn4_tla_evidence.json` satisfies the `xvs.sn4_tla_evidence/1`
     schema, SHA-256 self-integrity, `status == "PASS"`, exhaustive `N=2` safety +
     temporal liveness (`liveness_verified == True`, `zero_deadlock_verified == True`),
     reproduced uncoordinated livelock witness, `N=3` symmetry-reduced safety, and
     `rust_ticks_validated >= 10000` with `trace_violations == 0`.
  4. Honest compliance boundaries (`do178c_certified == False`, `do330_tool_qualified == False`,
     `g5_gap_preserved == True`, `frozen_crates_modified == False`).
  5. Negative control: injecting a 1-tick transition violation into a truncated NDJSON
     trace causes real `tlc2.TLC` (`OpenXvsTraceRefinement.tla`) to REJECT the trace!
  6. Non-breaking proposal `assurance/proposals/openxvs_v3_sn4_tla_proposal.gsn.yaml`
     parses cleanly via `xvs_certgen.parse_gsn_yaml_stdlib`, replaces `G2_TLA_DEFERRED`
     with `Sn4_TLA`, preserves `G5_GAP` (`undeveloped: true`), contains ZERO forbidden
     regulatory over-claims, and leaves frozen `assurance/openxvs_v2.gsn.yaml` intact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_TLA_DIR = REPO_ROOT / "formal" / "tla"
EVIDENCE_PATH = REPO_ROOT / "outputs" / "formal_tla" / "sn4_tla_evidence.json"
TRACE_PATH = REPO_ROOT / "outputs" / "formal_tla" / "rust_trace.ndjson"
PROPOSAL_GSN_PATH = REPO_ROOT / "assurance" / "proposals" / "openxvs_v3_sn4_tla_proposal.gsn.yaml"
FROZEN_GSN_PATH = REPO_ROOT / "assurance" / "openxvs_v2.gsn.yaml"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "run_tla_verification.py"
CERTGEN_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "xvs_certgen.py"
TRACE_CRATE_MANIFEST = REPO_ROOT / "tools" / "xvs-tla-trace" / "Cargo.toml"
COORD_CRATE_MANIFEST = REPO_ROOT / "xvs-multiship-coord" / "Cargo.toml"
COORD_KANI_SRC = REPO_ROOT / "xvs-multiship-coord" / "src" / "kani_proofs.rs"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_certgen():
    spec = importlib.util.spec_from_file_location("xvs_certgen_tla_test", CERTGEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xvs_certgen_tla_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tla_specs_and_exact_frozen_constants_present() -> None:
    """All formal TLA+ modules/configs exist and encode the exact frozen Rust constants."""
    required_files = [
        FORMAL_TLA_DIR / "OpenXvsMultiShipRta.tla",
        FORMAL_TLA_DIR / "OpenXvsMultiShipRta.cfg",
        FORMAL_TLA_DIR / "OpenXvsMultiShipRta_UncoordBug.cfg",
        FORMAL_TLA_DIR / "OpenXvsMultiShipRta_N3.cfg",
        FORMAL_TLA_DIR / "OpenXvsTraceRefinement.tla",
        FORMAL_TLA_DIR / "OpenXvsTraceRefinement.cfg",
        FORMAL_TLA_DIR / "README.md",
    ]
    for path in required_files:
        assert path.is_file(), f"Missing required formal artifact: {path}"
        assert path.stat().st_size > 100

    trace_spec = (FORMAL_TLA_DIR / "OpenXvsTraceRefinement.tla").read_text(encoding="utf-8")
    for exact_const in (
        "P0_RECOVERY_TICKS           == 30",
        "DWELL_TICKS                 == 20",
        "CONSEC_DROPOUT_MAX_TICKS    == 10",
        "WINDOW_DROPOUT_20_MAX_TICKS == 10",
        "SWERVE_FLIP_PERIOD          == 12",
        "TraceFullyAccepted",
    ):
        assert exact_const in trace_spec, f"Missing exact frozen constant {exact_const!r}"

    multiship_spec = (FORMAL_TLA_DIR / "OpenXvsMultiShipRta.tla").read_text(encoding="utf-8")
    for prop in (
        "SafetyInv_P0InhibitsTurns",
        "SafetyInv_ZeroVlmActuatorAuth",
        "SafetyInv_PessimisticUnion",
        "Prop_PluginOfflineAbsorbing",
        "Prop_P0ExitsOnlyToP1",
        "Prop_P1NeverClimbsToP2",
        "Live_RtaRecoveryToNominal",
        "Live_MultiShipResolution",
        "ApalacheConstInitN2",
        "ApalacheConstInitN3",
        "AllSafetyInvariants",
        "AllActionSafetyInvariants",
    ):
        assert prop in multiship_spec, f"Missing required formal property {prop!r}"

    # All 8 Kani CBMC harnesses exist on xvs-multiship-coord (4 v5.1 + 4 v5.2 GNSS-denied clock-sync)
    assert COORD_KANI_SRC.is_file(), f"Missing Kani proofs source: {COORD_KANI_SRC}"
    kani_src = COORD_KANI_SRC.read_text(encoding="utf-8")
    for harness in (
        "kani_f1_optical_only_non_reversing_far91113",
        "kani_f1_cooperative_peer_stagger_asymmetry",
        "kani_f4_mesh_staleness_replay_and_sticky_ghost",
        "kani_f3_ring0_p0_zeroes_all_avoidance",
        "kani_clock_sync_overflow_and_bounds_free",
        "kani_clock_sync_gnss_denied_drift_no_false_reject",
        "kani_clock_sync_uncalibrated_forces_tier_b_fallback",
        "kani_clock_sync_slew_preserves_monotonic_replay_guard",
    ):
        assert f"fn {harness}(" in kani_src, f"Missing Kani harness {harness!r}"


def test_rust_tla_trace_emitter_and_multiship_coord_cargo_test_pass() -> None:
    """Standalone `tools/xvs-tla-trace` and `xvs-multiship-coord` crates pass `cargo test` cleanly."""
    proc = subprocess.run(
        ["cargo", "test", "--manifest-path", str(TRACE_CRATE_MANIFEST)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"xvs-tla-trace cargo test failed:\n{proc.stdout}\n{proc.stderr}"
    assert "1 passed; 0 failed" in proc.stdout

    proc_coord = subprocess.run(
        ["cargo", "test", "--manifest-path", str(COORD_CRATE_MANIFEST)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert (
        proc_coord.returncode == 0
    ), f"xvs-multiship-coord cargo test failed:\n{proc_coord.stdout}\n{proc_coord.stderr}"
    assert "7 passed; 0 failed" in proc_coord.stdout


def test_sn4_tla_evidence_json_schema_hashes_and_proof_verdicts() -> None:
    """`outputs/formal_tla/sn4_tla_evidence.json` binds real TLC proofs with exact SHA-256 digests."""
    assert EVIDENCE_PATH.is_file(), f"Missing {EVIDENCE_PATH}; run scripts/xvs/run_tla_verification.py"
    doc = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert doc["schema"] == "xvs.sn4_tla_evidence/1"
    assert doc["status"] == "PASS"

    # Recompute deterministic evidence_sha256 over canonical payload
    recorded_sha = doc["evidence_sha256"]
    payload_copy = {k: v for k, v in doc.items() if k != "evidence_sha256"}
    canonical = json.dumps(payload_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == recorded_sha

    # Honest compliance & anti-overclaim lock
    cb = doc["compliance_boundaries"]
    assert cb["civil_dual_use_only"] is True
    assert cb["do178c_certified"] is False
    assert cb["do330_tool_qualified"] is False
    assert cb["g5_gap_preserved"] is True
    assert cb["frozen_crates_modified"] is False

    # Permissive MIT / Apache-2.0 toolchain lock
    tc = doc["toolchain"]
    assert tc["tla2tools_license"] == "MIT"
    assert tc["community_modules_license"] == "MIT"
    assert tc["apalache_license"] == "Apache-2.0"
    assert tc["kani_license"] == "MIT OR Apache-2.0"
    assert len(tc["tla2tools_sha256"]) == 64
    assert len(tc["apalache_jar_sha256"]) == 64

    # Exhaustive N=2 safety + temporal liveness + negative witness + N=3 symmetry safety
    mc = doc["multiship_model_check"]
    assert mc["liveness_verified"] is True
    assert mc["zero_deadlock_verified"] is True
    assert mc["states_generated"] > 100_000
    assert mc["distinct_states"] > 10_000
    assert mc["spec_sha256"] == _sha256_file(REPO_ROOT / mc["spec_path"])
    assert mc["cfg_sha256"] == _sha256_file(REPO_ROOT / mc["cfg_path"])
    assert mc["n2_uncoord_livelock_witness"]["temporal_livelock_reproduced"] is True
    assert mc["n3_result"] is not None
    assert mc["n3_result"]["model_passed"] is True
    assert mc["n3_result"]["exit_code"] == 0

    # Apache-2.0 Apalache Z3 symbolic verification (Snowcat typecheck + unbounded inductive + BMC)
    ap = doc["apalache_symbolic_check"]
    assert ap["all_passed"] is True
    assert ap["typecheck_passed"] is True
    assert ap["spec_sha256"] == _sha256_file(REPO_ROOT / ap["spec_path"])
    assert ap["n2_inductive_state"]["unbounded_safety_proved"] is True
    assert ap["n2_inductive_action"]["unbounded_action_safety_proved"] is True
    assert ap["n3_inductive_state"]["unbounded_safety_proved"] is True
    assert ap["n3_inductive_action"]["unbounded_action_safety_proved"] is True
    assert ap["n2_bmc"]["passed"] is True and ap["n2_bmc"]["length"] > 6
    assert ap["n3_bmc"]["passed"] is True and ap["n3_bmc"]["length"] > 6
    assert ap["closes_tlc_n3_depth6_limitation"] is True

    # WSL2 Kani CBMC formal proofs on xvs-multiship-coord (8/8 -> 26/26 OpenXVS total)
    kn = doc["multiship_coord_kani"]
    assert kn["all_verified"] is True
    assert kn["proofs_verified"] == 8
    assert kn["proofs_total"] == 8
    assert kn["frozen_crates_kani_proofs"] == 18
    assert kn["openxvs_total_kani_proofs"] == 26
    assert kn["cbmc_checks_failed"] == 0
    assert kn["cbmc_checks_verified"] == 5228
    assert kn["log_path"] == "outputs/formal_tla/kani_multiship_coord_v52.log"
    assert kn["do330_tool_qualified"] is False
    assert kn["source_sha256"] == _sha256_file(REPO_ROOT / kn["source_path"])
    assert kn["log_sha256"] == _sha256_file(REPO_ROOT / kn["log_path"])
    assert len(kn["harnesses"]) == 8
    assert all(h["verified"] is True and h["failed_checks"] == 0 for h in kn["harnesses"].values())

    # Mechanical Rust <-> TLA+ NDJSON trace refinement (12,000 frozen + 2,000 coordinated, 0 violations)
    tr = doc["trace_refinement"]
    assert tr["frozen_crate_ticks_validated"] == 12_000
    assert tr["coordinated_multiship_ticks_validated"] >= 2_000
    assert tr["rust_ticks_validated"] >= 14_000
    assert tr["rust_ticks_validated"] == tr["rust_ticks_emitted"]
    assert tr["trace_violations"] == 0
    assert tr["trace_fully_accepted"] is True
    assert tr["coordinated_crate_verified"] == "xvs-multiship-coord"
    assert tr["spec_sha256"] == _sha256_file(REPO_ROOT / tr["spec_path"])
    assert tr["cfg_sha256"] == _sha256_file(REPO_ROOT / tr["cfg_path"])
    assert tr["trace_sha256"] == _sha256_file(REPO_ROOT / tr["trace_path"])


def test_tlc_trace_refinement_rejects_deliberate_one_tick_mutation(tmp_path: Path) -> None:
    """Negative control: mutating 1 field on 1 tick MUST cause real TLC to reject the trace."""
    assert TRACE_PATH.is_file()
    lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()[:80]
    assert len(lines) == 80

    # Baseline clean 80-tick slice MUST PASS real TLC
    clean_trace = tmp_path / "clean_80.ndjson"
    clean_trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc_ok = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--trace-only", "--trace-path", str(clean_trace)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc_ok.returncode == 0, f"Clean 80-tick slice failed TLC:\n{proc_ok.stdout}\n{proc_ok.stderr}"

    # Corrupt tick index 25: illegally claim actuator_authority=true in VLM or wrong rta_mode jump
    corrupted_lines = list(lines)
    rec = json.loads(corrupted_lines[25])
    rec["vlm_act_auth"] = True
    corrupted_lines[25] = json.dumps(rec, separators=(",", ":"))
    bad_trace = tmp_path / "corrupt_80.ndjson"
    bad_trace.write_text("\n".join(corrupted_lines) + "\n", encoding="utf-8")

    proc_bad = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--trace-only", "--trace-path", str(bad_trace)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc_bad.returncode != 0, "TLC MUST reject a 1-tick violation in the Rust NDJSON trace!"

    # Coordinated scenario negative control: take ticks 12,000..12,080 (xvs-multiship-coord F1+F4+F3),
    # verify clean slice passes TLC, then inject a head-on F1 sign reversal (ar_bank_sign = -1) -> MUST REJECT!
    all_lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(all_lines) >= 14_000
    coord_lines = all_lines[12_000:12_080]
    clean_coord = tmp_path / "clean_coord_80.ndjson"
    clean_coord.write_text("\n".join(coord_lines) + "\n", encoding="utf-8")
    proc_coord_ok = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--trace-only", "--trace-path", str(clean_coord)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (
        proc_coord_ok.returncode == 0
    ), f"Clean coordinated 80-tick slice failed TLC:\n{proc_coord_ok.stdout}\n{proc_coord_ok.stderr}"

    bad_coord_lines = list(coord_lines)
    mutated = False
    for idx, raw_line in enumerate(bad_coord_lines):
        rec_c = json.loads(raw_line)
        if rec_c["coord_enabled"] and rec_c["coord_head_on"] and rec_c["ar_mode"] == 1:
            rec_c["ar_bank_sign"] = -1
            bad_coord_lines[idx] = json.dumps(rec_c, separators=(",", ":"))
            mutated = True
            break
    assert mutated, "Expected at least one coordinated head-on ActiveSwerve tick in slice"
    bad_coord = tmp_path / "corrupt_coord_80.ndjson"
    bad_coord.write_text("\n".join(bad_coord_lines) + "\n", encoding="utf-8")
    proc_coord_bad = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--trace-only", "--trace-path", str(bad_coord)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (
        proc_coord_bad.returncode != 0
    ), "TLC MUST reject a 1-tick coordinated F1 head-on swerve sign reversal!"


def test_gsn_v3_sn4_tla_proposal_valid_and_frozen_v2_untouched() -> None:
    """Proposal replaces G2_TLA_DEFERRED -> Sn4_TLA, preserves G5_GAP, and leaves v2 frozen."""
    certgen = _load_certgen()

    # 1. Frozen v2 GSN must remain 100% untouched & passing xvs_certgen
    v2_bundle = certgen.synthesize_assurance_bundle(gsn_path=FROZEN_GSN_PATH, repo_root=REPO_ROOT)
    assert v2_bundle["status"] == "PASS", f"Frozen v2 GSN broken: {v2_bundle['errors']}"
    assert "G2_TLA_DEFERRED" in v2_bundle["gsn_source"]["undeveloped_nodes"]
    assert "G5_GAP" in v2_bundle["gsn_source"]["undeveloped_nodes"]

    # 2. Proposal v3 GSN parses cleanly, has acyclic DAG, zero dangling refs, zero forbidden claims
    assert PROPOSAL_GSN_PATH.is_file()
    prop_text = PROPOSAL_GSN_PATH.read_text(encoding="utf-8")
    nodes = certgen.parse_gsn_yaml_stdlib(prop_text)

    assert "G2" in nodes
    assert "Sn4_TLA" in nodes
    assert "G2_TLA_DEFERRED" not in nodes
    assert "Sn4_TLA" in nodes["G2"]["supportedBy"]
    assert "G2_TLA_DEFERRED" not in nodes["G2"]["supportedBy"]

    # G5_GAP stays permanently undeveloped with zero supportedBy children
    assert "G5_GAP" in nodes
    assert nodes["G5_GAP"].get("undeveloped") is True
    assert not nodes["G5_GAP"].get("supportedBy")

    # Dangling refs & acyclicity check
    for nid, body in nodes.items():
        for rel in ("supportedBy", "inContextOf"):
            for target in body.get(rel, []):
                assert target in nodes, f"Dangling GSN proposal ref: {nid}.{rel} -> {target}"
    assert certgen._check_dag_cycles(nodes) == []

    # Forbidden regulatory over-claim check across all proposal nodes
    for nid, body in nodes.items():
        low = str(body.get("text", "")).lower()
        for phrase in certgen.FORBIDDEN_CERT_CLAIMS:
            assert phrase not in low, f"Forbidden claim {phrase!r} in proposal node {nid}"

    # Verify SHA-256 & byte-size digests of all Sn4_TLA & Sn5_KANI evidenceFiles match disk byte-for-byte
    sn4_files = nodes["Sn4_TLA"]["evidenceFiles"]
    assert len(sn4_files) >= 3
    for entry in sn4_files:
        fpath = REPO_ROOT / entry["path"]
        assert fpath.is_file(), f"Missing Sn4_TLA evidence file: {entry['path']}"
        assert _sha256_file(fpath) == entry["sha256"]
        assert fpath.stat().st_size == entry["size_bytes"]

    sn5_files = nodes["Sn5_KANI"]["evidenceFiles"]
    assert len(sn5_files) >= 3
    for entry in sn5_files:
        fpath = REPO_ROOT / entry["path"]
        assert fpath.is_file(), f"Missing Sn5_KANI evidence file: {entry['path']}"
        assert _sha256_file(fpath) == entry["sha256"]
        assert fpath.stat().st_size == entry["size_bytes"]
