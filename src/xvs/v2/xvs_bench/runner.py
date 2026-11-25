"""Stage-2 XVS-Bench 4-track closed-loop harness & dual-run deterministic closure runner (SP3 §3.2).

Reads ONLY real M1 (`output/xvs/v2_m1/*`), M2 (`output/xvs/v2_m2/*`), and Stage-1 M3
(`output/xvs/v2_m3/{v1_xvs_rt_evidence.json, v2_meshdaa_500enc_evidence.json}`) artifacts
per the strict acyclic 4-stage DAG (`BC-M3-M2`). Never hashes `assurance/*`, Stage 3, or Stage 4.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .croissant import build_croissant_jsonld, validate_croissant_with_wsl2_jsonschema
from .ed25519_pure import (
    cross_verify_with_wsl2_cryptography,
    verify_rfc8032_test_vectors,
)
from .hash_chain import (
    SIGNING_KEY_SCOPE_LOCAL_TEST,
    _scan_for_forbidden_secret_keys,
    build_signed_hash_chain,
    canonical_json_bytes,
    verify_signed_hash_chain,
)
from .phase4_slot import (
    PHASE4_STATUS_PREWIRED,
    build_prewired_phase4_slot_manifest,
)

# Deterministic non-production test seed (NEVER serialized into any output JSON!)
DEFAULT_LOCAL_TEST_SEED32: bytes = hashlib.sha256(
    b"openxvs-v2-m3-local-deterministic-test-seed-non-production"
).digest()

M1_CHAIN_ARTIFACTS: tuple[str, ...] = (
    "output/xvs/v2_m1/kani_proofs.log",
    "output/xvs/v2_m1/v1_cargo_evidence.json",
    "output/xvs/v2_m1/parity_results.json",
    "output/xvs/v2_m1/headless_missions_results.json",
    "output/xvs/v2_m1/d1_onnx_parity.json",
    "output/xvs/v2_m1/v5_bundle_budget.json",
    "output/xvs/v2_m1/v6_bridge_evidence.json",
    "output/xvs/v2_m1/e2e_results.json",
)

M2_CHAIN_ARTIFACTS: tuple[str, ...] = (
    "output/xvs/v2_m2/v1_allowlist_evidence.json",
    "output/xvs/v2_m2/v2_xvs_d2_evidence.json",
    "output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json",
    "output/xvs/v2_m2/v4_wp01b_land_evidence.json",
    "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
    "output/xvs/v2_m2/v5_colab_staging_evidence.json",
)

STAGE1_M3_ARTIFACTS: tuple[str, ...] = (
    "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
    "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_kani_summary(kani_log_path: Path) -> tuple[int, int, int]:
    """Parse `(passed, failed, total)` from `kani_proofs.log` fail-closed."""
    text = kani_log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Complete\s*-\s*(\d+)\s+successfully verified harnesses,\s*(\d+)\s+failures,\s*(\d+)\s+total\.",
        text,
    )
    if not matches:
        raise ValueError(f"Could not parse Kani harness summary from {kani_log_path}")
    passed, failed, total = matches[-1]
    return int(passed), int(failed), int(total)


def _assemble_four_tracks(
    repo_root: Path,
    stage1_v1: dict[str, Any] | None,
    stage1_v2: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble all 4 closed-loop civil aviation tracks from real M1/M2/Stage-1 M3 producer JSONs."""
    m1_dir = repo_root / "output" / "xvs" / "v2_m1"
    m2_dir = repo_root / "output" / "xvs" / "v2_m2"

    m1_cargo = _read_json(m1_dir / "v1_cargo_evidence.json")
    m1_missions = _read_json(m1_dir / "headless_missions_results.json")
    m1_e2e = _read_json(m1_dir / "e2e_results.json")
    m2_d2 = _read_json(m2_dir / "v2_xvs_d2_evidence.json")
    m2_land = _read_json(m2_dir / "v4_wp01b_land_evidence.json")
    m2_nav_vraim = _read_json(m2_dir / "v5_nav_vraim_evidence.json")
    kani_passed, kani_failed, kani_total = _parse_kani_summary(m1_dir / "kani_proofs.log")

    stage1_real = (stage1_v1 is not None) and (stage1_v2 is not None)

    track_a: dict[str, Any] = {
        "track_id": "Track_A_DAA_AOT_MeshDAA",
        "xvs_d2_web_fp16_bytes": int(m2_d2["onnx_export"]["fp16_onnx_bytes"]),
        "xvs_d2_web_fp16_ort_max_diff": float(m2_d2["onnx_export"]["fp16_max_abs_diff"]),
        "aot_real_data_map": "[PENDING_COLAB_G4_TRAINING]",
        "stage1_meshdaa_bound": stage1_real,
        "meshdaa_500enc_summary": (
            stage1_v2 if stage1_v2 is not None else {"status": "[PENDING_STAGE1_XVS_RT_EXECUTION]"}
        ),
    }

    track_b: dict[str, Any] = {
        "track_id": "Track_B_Nav_AntiSpoofing",
        "rectification_max_roundtrip_err_px": float(
            m2_nav_vraim["xvs_nav"]["rectification_max_roundtrip_err_px"]
        ),
        "internal_enu_hits": int(m2_nav_vraim["xvs_nav"]["internal_enu_hits"]),
        "nonfinite_aliasing_cases_blocked": int(
            m2_nav_vraim["xvs_nav"]["nonfinite_adversarial_cases_blocked"]
        ),
        "m1_m2_spoof_trip_latency_ticks": int(m1_missions["missions"]["M2"]["trip_latency_ticks"]),
        "m1_m2_final_cusum_stat": float(m1_missions["missions"]["M2"]["final_cusum_stat"]),
        "corridor_50km_colab_ate_m": "[PENDING_COLAB_G4_TRAINING]",
    }

    track_c: dict[str, Any] = {
        "track_id": "Track_C_Land_VRAIM",
        "wp01b_ray_dem_max_err_m": float(m2_land["wp01b"]["curved_ray_dem_max_vert_err_m"]),
        "taws_nominal_false_alerts": int(m2_land["wp01b"]["taws_nominal_ksfo_28l_false_alerts"]),
        "vraim_short_final_vvpl_m": float(m2_land["xvs_land"]["conformal_raim_v_vpl_m"]),
        "lard_real_icao_overlap": int(m2_land["xvs_land"]["lard_train_heldout_overlap_count"]),
        "m1_m4_go_around_climb_gain_m": float(
            m1_e2e["pass_a_webgpu"]["missions"]["M4"]["climb_gain_m"]
        ),
        "lard_h3_colab_hmi_rate": "[PENDING_COLAB_G4_TRAINING]",
    }

    track_d: dict[str, Any] = {
        "track_id": "Track_D_xvs_rt_ODD_Assurance",
        "kani_harnesses_passed": kani_passed,
        "kani_harnesses_failed": kani_failed,
        "kani_harnesses_total": kani_total,
        "cargo_release_tests_passed": int(m1_cargo["cargo_test_workspace_release_passed"]),
        "rta_proptest_cases": int(m1_cargo["rta_proptest_cases"]),
        "rta_unsafe_code_count": int(m1_cargo["rta_unsafe_code_count"]),
        "easa_rlm_07_11_all_pass": bool(m2_nav_vraim["xvs_vraim"]["easa_rlm_07_11_all_pass"]),
        "stage1_xvs_rt_bound": stage1_real,
        "xvs_rt_wsl2_summary": (
            stage1_v1 if stage1_v1 is not None else {"status": "[PENDING_STAGE1_XVS_RT_EXECUTION]"}
        ),
        "native_linux_gpu_p99_wcet_ms": "[PENDING_NATIVE_LINUX_GPU_WCET]",
    }

    return {
        "Track_A": track_a,
        "Track_B": track_b,
        "Track_C": track_c,
        "Track_D": track_d,
    }


def run_xvs_bench_stage2(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    require_stage1: bool = True,
    signing_seed32: bytes = DEFAULT_LOCAL_TEST_SEED32,
    run_wsl2_cross_checks: bool = True,
) -> dict[str, Any]:
    """Execute full Stage-2 XVS-Bench dual-run closure and persist Croissant, Chain & Evidence JSONs."""
    out_dir = (repo_root / "output" / "xvs" / "v2_m3") if output_dir is None else output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_v1_path = repo_root / STAGE1_M3_ARTIFACTS[0]
    stage1_v2_path = repo_root / STAGE1_M3_ARTIFACTS[1]
    has_stage1 = stage1_v1_path.is_file() and stage1_v2_path.is_file()
    if require_stage1 and not has_stage1:
        raise FileNotFoundError(
            f"Stage-1 M3 artifacts required for Stage-2 XVS-Bench closure: "
            f"{stage1_v1_path} (exists={stage1_v1_path.is_file()}), "
            f"{stage1_v2_path} (exists={stage1_v2_path.is_file()})"
        )

    stage1_v1 = _read_json(stage1_v1_path) if has_stage1 else None
    stage1_v2 = _read_json(stage1_v2_path) if has_stage1 else None

    artifact_list = list(M1_CHAIN_ARTIFACTS) + list(M2_CHAIN_ARTIFACTS)
    if has_stage1:
        artifact_list.extend(STAGE1_M3_ARTIFACTS)

    phase4_slot = build_prewired_phase4_slot_manifest()
    tracks = _assemble_four_tracks(repo_root, stage1_v1, stage1_v2)

    chain_meta = {
        "benchmark_name": "OpenXVS-Bench v2.0",
        "tracks_count": 4,
        "stage1_m3_bound": has_stage1,
        "phase4_slot_status": PHASE4_STATUS_PREWIRED,
    }

    # Dual independent deterministic closure runs (DR-08:52, 107)
    chain_run1 = build_signed_hash_chain(
        repo_root,
        artifact_list,
        signing_seed32,
        signing_key_scope=SIGNING_KEY_SCOPE_LOCAL_TEST,
        metadata=chain_meta,
    )
    chain_run2 = build_signed_hash_chain(
        repo_root,
        artifact_list,
        signing_seed32,
        signing_key_scope=SIGNING_KEY_SCOPE_LOCAL_TEST,
        metadata=chain_meta,
    )
    dual_run_exact = (
        chain_run1["root_digest_sha256"] == chain_run2["root_digest_sha256"]
        and chain_run1["signature_hex"] == chain_run2["signature_hex"]
        and canonical_json_bytes(chain_run1) == canonical_json_bytes(chain_run2)
    )
    chain_valid, chain_errors = verify_signed_hash_chain(chain_run1, repo_root=repo_root)
    if not dual_run_exact or not chain_valid:
        raise RuntimeError(
            f"XVS-Bench closure verification failed: dual_run_exact={dual_run_exact}, errors={chain_errors}"
        )

    croissant_doc = build_croissant_jsonld(repo_root, artifact_list, phase4_slot)
    rfc8032_res = verify_rfc8032_test_vectors()

    if run_wsl2_cross_checks:
        wsl_ed25519 = cross_verify_with_wsl2_cryptography(
            signing_seed32,
            chain_run1["root_digest_sha256"].encode("ascii"),
        )
        wsl_jsonschema = validate_croissant_with_wsl2_jsonschema(croissant_doc)
    else:
        wsl_ed25519 = {"pass": False, "skipped": True}
        wsl_jsonschema = {"valid": False, "skipped": True}

    chain_path = out_dir / "xvs_bench_chain.json"
    croissant_path = out_dir / "xvs_bench_croissant.json"
    evidence_path = out_dir / "v3_xvs_bench_evidence.json"

    chain_text = json.dumps(chain_run1, indent=2) + "\n"
    chain_path.write_text(chain_text, encoding="utf-8")
    croissant_path.write_text(json.dumps(croissant_doc, indent=2) + "\n", encoding="utf-8")

    private_key_serialized = bool(_scan_for_forbidden_secret_keys(chain_run1)) or (
        signing_seed32.hex() in chain_text
    )

    evidence_doc: dict[str, Any] = {
        "schema": "xvs.v2_m3.v3_xvs_bench_evidence/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_m3_bound": has_stage1,
        "signing_key_scope": SIGNING_KEY_SCOPE_LOCAL_TEST,
        "rfc8032_vectors_passed": int(rfc8032_res["vectors_passed"]),
        "rfc8032_vectors_total": int(rfc8032_res["vectors_total"]),
        "wsl2_cryptography_cross_verify": wsl_ed25519,
        "wsl2_jsonschema_croissant": wsl_jsonschema,
        "hash_chain_entry_count": int(chain_run1["entry_count"]),
        "hash_chain_root_digest_sha256": str(chain_run1["root_digest_sha256"]),
        "hash_chain_public_key_hex": str(chain_run1["public_key_hex"]),
        "dual_run_deterministic_match": bool(dual_run_exact),
        "chain_verified_ok": bool(chain_valid),
        "private_key_serialized": private_key_serialized,
        "phase4_slot": phase4_slot,
        "tracks": tracks,
        "pass": bool(
            rfc8032_res["all_pass"]
            and dual_run_exact
            and chain_valid
            and (not private_key_serialized)
            and (not run_wsl2_cross_checks or (wsl_ed25519["pass"] and wsl_jsonschema["valid"]))
        ),
    }
    evidence_path.write_text(json.dumps(evidence_doc, indent=2) + "\n", encoding="utf-8")
    return evidence_doc
