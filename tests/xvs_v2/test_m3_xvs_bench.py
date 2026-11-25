"""OpenXVS v2.0 M3 Stage-2 XVS-Bench verification suite (V2-M3L-3, SP3 §3.2)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil

import pytest

from xvs.v2.xvs_bench import (
    DEFAULT_LOCAL_TEST_SEED32,
    M1_CHAIN_ARTIFACTS,
    M2_CHAIN_ARTIFACTS,
    PHASE4_STATUS_PREWIRED,
    SIGNING_KEY_SCOPE_LOCAL_TEST,
    STAGE1_M3_ARTIFACTS,
    LiveSkyFrameGateInput,
    assert_no_secret_material,
    assert_path_allowed_in_stage2_chain,
    build_croissant_jsonld,
    build_prewired_phase4_slot_manifest,
    build_signed_hash_chain,
    canonical_json_bytes,
    cross_verify_with_wsl2_cryptography,
    ed25519_public_key,
    ed25519_sign,
    ed25519_verify,
    run_xvs_bench_stage2,
    validate_croissant_with_wsl2_jsonschema,
    validate_live_sky_gate,
    verify_rfc8032_test_vectors,
    verify_signed_hash_chain,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ed25519_rfc8032_official_vectors_and_tamper() -> None:
    res = verify_rfc8032_test_vectors()
    assert res["vectors_total"] == 3
    assert res["vectors_passed"] == 3
    assert res["all_pass"] is True

    seed = DEFAULT_LOCAL_TEST_SEED32
    pk = ed25519_public_key(seed)
    msg = b"openxvs-v2-m3-bench-determinism-check"
    sig = ed25519_sign(seed, msg)
    assert ed25519_verify(pk, msg, sig) is True

    # 1-bit tamper in signature, message, or public key must fail closed.
    bad_sig = bytearray(sig)
    bad_sig[0] ^= 0x01
    assert ed25519_verify(pk, msg, bytes(bad_sig)) is False

    bad_msg = msg + b"!"
    assert ed25519_verify(pk, bad_msg, sig) is False

    bad_pk = bytearray(pk)
    bad_pk[0] ^= 0x01
    assert ed25519_verify(bytes(bad_pk), msg, sig) is False

    # Invalid lengths fail closed.
    assert ed25519_verify(pk[:-1], msg, sig) is False
    assert ed25519_verify(pk, msg, sig[:-1]) is False


def test_ed25519_bidirectional_wsl2_cryptography_cross_verify() -> None:
    msg = b"OpenXVS-Bench-v2.0-M3-Ed25519-CrossVerify"
    res = cross_verify_with_wsl2_cryptography(DEFAULT_LOCAL_TEST_SEED32, msg)
    assert res["pk_match"] is True
    assert res["pure_sig_verified_by_crypto"] is True
    assert res["crypto_sig_verified_by_pure"] is True
    assert res["bit_exact_signature_match"] is True
    assert res["pass"] is True
    assert isinstance(res["cryptography_version"], str) and len(res["cryptography_version"]) > 0


@pytest.mark.parametrize(
    "allowed_rel",
    [
        "output/xvs/v2_m1/v1_cargo_evidence.json",
        "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
        "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
    ],
)
def test_stage2_dag_guard_accepts_upstream_artifacts(allowed_rel: str) -> None:
    assert_path_allowed_in_stage2_chain(allowed_rel)


@pytest.mark.parametrize(
    "forbidden_rel",
    [
        "assurance/openxvs_v2.gsn.yaml",
        "output/xvs/v2_m3/v4_ed324_gsn_evidence.json",
        "output/xvs/v2_m3/v3_xvs_bench_evidence.json",
        "output/xvs/v2_m3/xvs_bench_chain.json",
        "output/xvs/v2_m3/xvs_bench_croissant.json",
        "output/xvs/v2_m3/run_manifest.json",
        "docs/xvs/reports/V2_M3_local_core_report.md",
        "../outside.json",
        "/abs/path.json",
    ],
)
def test_stage2_dag_guard_rejects_cycles_and_downstream(forbidden_rel: str) -> None:
    with pytest.raises(ValueError):
        assert_path_allowed_in_stage2_chain(forbidden_rel)


def test_hash_chain_determinism_zero_secrets_and_tamper_rejection(tmp_path: Path) -> None:
    # Create isolated upstream files matching the strict acyclic Stage-2 allowlist.
    rel_paths = [
        "output/xvs/v2_m1/v1_cargo_evidence.json",
        "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
        "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
    ]
    for idx, rel in enumerate(rel_paths):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"artifact_idx": idx, "rel": rel}) + "\n", encoding="utf-8")

    chain1 = build_signed_hash_chain(
        tmp_path,
        rel_paths,
        DEFAULT_LOCAL_TEST_SEED32,
        signing_key_scope=SIGNING_KEY_SCOPE_LOCAL_TEST,
        metadata={"benchmark": "OpenXVS-Bench-v2.0"},
    )
    chain2 = build_signed_hash_chain(
        tmp_path,
        rel_paths,
        DEFAULT_LOCAL_TEST_SEED32,
        signing_key_scope=SIGNING_KEY_SCOPE_LOCAL_TEST,
        metadata={"benchmark": "OpenXVS-Bench-v2.0"},
    )

    # Dual-run exact determinism.
    assert chain1["root_digest_sha256"] == chain2["root_digest_sha256"]
    assert chain1["signature_hex"] == chain2["signature_hex"]
    assert canonical_json_bytes(chain1) == canonical_json_bytes(chain2)
    assert chain1["signing_key_scope"] == "LOCAL_DETERMINISTIC_TEST_SEED_NON_PRODUCTION"

    # Zero private key / seed serialization (BC-M3-m5).
    assert_no_secret_material(chain1)
    serialized = json.dumps(chain1)
    assert DEFAULT_LOCAL_TEST_SEED32.hex() not in serialized
    assert "private_key" not in serialized.lower()
    assert "secret_key" not in serialized.lower()

    ok, errs = verify_signed_hash_chain(chain1, repo_root=tmp_path)
    assert ok is True and errs == []

    # 1-byte tamper in entry file_sha256 must break BOTH chain digest AND signature.
    tampered_entry = copy.deepcopy(chain1)
    orig_hex = tampered_entry["entries"][0]["file_sha256"]
    flipped_char = "0" if orig_hex[0] != "0" else "1"
    tampered_entry["entries"][0]["file_sha256"] = flipped_char + orig_hex[1:]
    ok_e, errs_e = verify_signed_hash_chain(tampered_entry)
    assert ok_e is False
    assert any("ENTRY_SHA256_TAMPERED" in e for e in errs_e)
    assert any("ROOT_DIGEST" in e or "SIGNATURE" in e for e in errs_e)

    # 1-byte tamper in root_digest_sha256 must break verification.
    tampered_root = copy.deepcopy(chain1)
    r_hex = tampered_root["root_digest_sha256"]
    tampered_root["root_digest_sha256"] = ("0" if r_hex[0] != "0" else "1") + r_hex[1:]
    ok_r, errs_r = verify_signed_hash_chain(tampered_root)
    assert ok_r is False
    assert any("SIGNATURE" in e or "ROOT_DIGEST" in e for e in errs_r)

    # 1-byte tamper in signature_hex must break verification.
    tampered_sig = copy.deepcopy(chain1)
    s_hex = tampered_sig["signature_hex"]
    tampered_sig["signature_hex"] = ("0" if s_hex[0] != "0" else "1") + s_hex[1:]
    ok_s, errs_s = verify_signed_hash_chain(tampered_sig)
    assert ok_s is False
    assert any("SIGNATURE" in e for e in errs_s)

    # Injecting a forbidden secret field must fail closed.
    leaked_secret = copy.deepcopy(chain1)
    leaked_secret["private_key_hex"] = "00" * 32
    ok_sec, errs_sec = verify_signed_hash_chain(leaked_secret)
    assert ok_sec is False
    assert any("FORBIDDEN_SECRET_KEYS_SERIALIZED" in e for e in errs_sec)

    # 1-byte tamper of an upstream artifact on disk must be caught when repo_root is supplied.
    disk_file = tmp_path / rel_paths[1]
    disk_file.write_text(disk_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    ok_disk, errs_disk = verify_signed_hash_chain(chain1, repo_root=tmp_path)
    assert ok_disk is False
    assert any("SHA256" in e.upper() or "DISK" in e.upper() for e in errs_disk)


def test_phase4_live_sky_slot_manifest_and_fail_closed_gates() -> None:
    manifest = build_prewired_phase4_slot_manifest()
    assert manifest["split"] == "live_sky_imx296_adsb"
    assert manifest["status"] == PHASE4_STATUS_PREWIRED == "PREWIRED_PHASE4_PENDING_CAPTURE"
    assert "Sony IMX296" in manifest["hardware_spec"]["camera"]
    assert "1456x1088" in manifest["hardware_spec"]["camera"]
    assert "16mm" in manifest["hardware_spec"]["optics"]
    assert "FlightAware Pro Stick Plus" in manifest["hardware_spec"]["sdr"]
    assert any("RTL-SDR Blog V4" in s for s in manifest["hardware_spec"]["excluded_eol_hardware"])
    assert "ICM-42688-P" in manifest["hardware_spec"]["imu"]
    assert "1PPS" in manifest["hardware_spec"]["timebase"]

    # Nominal frame passes.
    nominal = LiveSkyFrameGateInput(
        mode_s_tc19_decoded=True,
        nic=8,
        nacp=10,
        boresight_rms_deg=0.08,
        pps_synced=True,
        exposure_latency_ms=1.2,
    )
    ok_res = validate_live_sky_gate(nominal)
    assert ok_res.admitted is True
    assert ok_res.failed_checks == ()

    # Every single hard gate & non-finite value must fail closed.
    bad_cases: list[tuple[LiveSkyFrameGateInput, str]] = [
        (
            LiveSkyFrameGateInput(False, 8, 10, 0.08, True, 1.2),
            "MODE_S_TC19_NOT_DECODED",
        ),
        (
            LiveSkyFrameGateInput(True, 6, 10, 0.08, True, 1.2),
            "NIC_BELOW_MIN",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 8, 0.08, True, 1.2),
            "NACP_BELOW_MIN",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, 0.16, True, 1.2),
            "BORESIGHT_RMS_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, -0.01, True, 1.2),
            "BORESIGHT_RMS_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, 0.08, False, 1.2),
            "PPS_SYNC_MISSING",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, 0.08, True, 6.0),
            "EXPOSURE_LATENCY_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, math.nan, True, 1.2),
            "BORESIGHT_RMS_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, math.inf, True, 1.2),
            "BORESIGHT_RMS_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, -math.inf, True, 1.2),
            "BORESIGHT_RMS_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, 0.08, True, math.nan),
            "EXPOSURE_LATENCY_FAIL",
        ),
        (
            LiveSkyFrameGateInput(True, 8, 10, 0.08, True, math.inf),
            "EXPOSURE_LATENCY_FAIL",
        ),
    ]
    for frame, expected_tag in bad_cases:
        res = validate_live_sky_gate(frame)
        assert res.admitted is False
        assert any(
            expected_tag in r for r in res.failed_checks
        ), f"Expected {expected_tag} in {res.failed_checks}"


def test_croissant_jsonld_wsl2_jsonschema_validation_and_negative_rejection() -> None:
    phase4_slot = build_prewired_phase4_slot_manifest()
    doc = build_croissant_jsonld(
        REPO_ROOT,
        list(M1_CHAIN_ARTIFACTS) + list(M2_CHAIN_ARTIFACTS),
        phase4_slot,
    )
    res = validate_croissant_with_wsl2_jsonschema(doc)
    assert res["valid"] is True, f"Croissant schema errors: {res}"
    assert res["errors"] == []
    assert doc["@type"] == "sc:Dataset"
    assert len(doc["recordSet"]) == 5

    # Negative check: dropping mandatory distribution must fail schema validation.
    broken = copy.deepcopy(doc)
    del broken["distribution"]
    bad_res = validate_croissant_with_wsl2_jsonschema(broken)
    assert bad_res["valid"] is False
    assert len(bad_res["errors"]) >= 1


def test_stage2_runner_require_stage1_and_synthetic_stage1_closure(tmp_path: Path) -> None:
    # Copy real M1 & M2 artifacts into isolated tmp_path to verify require_stage1=True enforcement
    # and full 16-entry Stage-1-bound closure without mutating the real repo.
    for rel in list(M1_CHAIN_ARTIFACTS) + list(M2_CHAIN_ARTIFACTS):
        src = REPO_ROOT / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Missing Stage-1 must fail closed when require_stage1=True.
    with pytest.raises(FileNotFoundError):
        run_xvs_bench_stage2(tmp_path, require_stage1=True, run_wsl2_cross_checks=False)

    # Populate synthetic Stage-1 v1 & v2 files and verify full 16-entry bound closure.
    v1_path = tmp_path / STAGE1_M3_ARTIFACTS[0]
    v2_path = tmp_path / STAGE1_M3_ARTIFACTS[1]
    v1_path.parent.mkdir(parents=True, exist_ok=True)
    v1_path.write_text(
        json.dumps({"schema": "xvs.v2_m3.v1_xvs_rt_evidence/1", "wsl2_smoke_pass": True}) + "\n",
        encoding="utf-8",
    )
    v2_path.write_text(
        json.dumps({"schema": "xvs.v2_m3.v2_meshdaa_500enc_evidence/1", "encounters": 500, "nmac_count": 0})
        + "\n",
        encoding="utf-8",
    )

    ev = run_xvs_bench_stage2(tmp_path, require_stage1=True, run_wsl2_cross_checks=False)
    assert ev["stage1_m3_bound"] is True
    assert ev["hash_chain_entry_count"] == len(M1_CHAIN_ARTIFACTS) + len(M2_CHAIN_ARTIFACTS) + 2
    assert ev["dual_run_deterministic_match"] is True
    assert ev["chain_verified_ok"] is True
    assert ev["private_key_serialized"] is False
    assert ev["tracks"]["Track_A"]["stage1_meshdaa_bound"] is True
    assert ev["tracks"]["Track_D"]["stage1_xvs_rt_bound"] is True
    assert ev["pass"] is True


def test_stage2_runner_real_repo_emits_verified_m3_artifacts(tmp_path: Path) -> None:
    # Copy real M1/M2/Stage-1 upstream files into isolated tmp_path so pytest never clobbers
    # GSN-pinned `output/xvs/v2_m3/xvs_bench_chain.json` or V5.0 Croissant 1.1+RAI metadata.
    for rel in list(M1_CHAIN_ARTIFACTS) + list(M2_CHAIN_ARTIFACTS) + list(STAGE1_M3_ARTIFACTS):
        src = REPO_ROOT / rel
        assert src.is_file(), f"Missing upstream M1/M2/Stage-1 artifact: {rel}"
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    ev = run_xvs_bench_stage2(
        tmp_path,
        require_stage1=True,
        run_wsl2_cross_checks=True,
    )
    assert ev["pass"] is True
    assert ev["stage1_m3_bound"] is True
    assert ev["hash_chain_entry_count"] == 16
    assert ev["tracks"]["Track_A"]["stage1_meshdaa_bound"] is True
    assert ev["tracks"]["Track_D"]["stage1_xvs_rt_bound"] is True
    assert ev["rfc8032_vectors_passed"] == 3
    assert ev["rfc8032_vectors_total"] == 3
    assert ev["wsl2_cryptography_cross_verify"]["pass"] is True
    assert ev["wsl2_jsonschema_croissant"]["valid"] is True
    assert ev["dual_run_deterministic_match"] is True
    assert ev["chain_verified_ok"] is True
    assert ev["private_key_serialized"] is False
    assert ev["signing_key_scope"] == "LOCAL_DETERMINISTIC_TEST_SEED_NON_PRODUCTION"

    # Honest deferred tags must remain intact.
    assert ev["tracks"]["Track_A"]["aot_real_data_map"] == "[PENDING_COLAB_G4_TRAINING]"
    assert ev["tracks"]["Track_B"]["corridor_50km_colab_ate_m"] == "[PENDING_COLAB_G4_TRAINING]"
    assert ev["tracks"]["Track_C"]["lard_h3_colab_hmi_rate"] == "[PENDING_COLAB_G4_TRAINING]"
    assert ev["tracks"]["Track_D"]["native_linux_gpu_p99_wcet_ms"] == "[PENDING_NATIVE_LINUX_GPU_WCET]"
    assert ev["tracks"]["Track_D"]["kani_harnesses_passed"] == 5
    assert ev["tracks"]["Track_D"]["kani_harnesses_failed"] == 0

    # Verify all 3 persisted Stage-2 output JSONs exist in isolated tmp_path and pass verification.
    out_dir = tmp_path / "output" / "xvs" / "v2_m3"
    chain_path = out_dir / "xvs_bench_chain.json"
    croissant_path = out_dir / "xvs_bench_croissant.json"
    evidence_path = out_dir / "v3_xvs_bench_evidence.json"
    assert chain_path.is_file()
    assert croissant_path.is_file()
    assert evidence_path.is_file()

    chain_doc = json.loads(chain_path.read_text(encoding="utf-8"))
    ok, errs = verify_signed_hash_chain(chain_doc, repo_root=tmp_path)
    assert ok is True and errs == []
    assert DEFAULT_LOCAL_TEST_SEED32.hex() not in chain_path.read_text(encoding="utf-8")
