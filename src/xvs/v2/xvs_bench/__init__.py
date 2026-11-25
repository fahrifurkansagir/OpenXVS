"""XVS-Bench: Cryptographically attested 4-track closed-loop aviation autonomy benchmark (DR-08 / SP3 §3.2)."""

from .croissant import (
    CROISSANT_1_0_JSON_SCHEMA,
    build_croissant_jsonld,
    validate_croissant_with_wsl2_jsonschema,
)
from .ed25519_pure import (
    RFC8032_TEST_VECTORS,
    cross_verify_with_wsl2_cryptography,
    ed25519_public_key,
    ed25519_sign,
    ed25519_verify,
    verify_rfc8032_test_vectors,
)
from .hash_chain import (
    GENESIS_PREV_SHA256,
    SIGNING_KEY_SCOPE_LOCAL_TEST,
    assert_no_secret_material,
    assert_path_allowed_in_stage2_chain,
    build_signed_hash_chain,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_signed_hash_chain,
)
from .phase4_slot import (
    PHASE4_SPLIT_NAME,
    PHASE4_STATUS_PREWIRED,
    LiveSkyFrameGateInput,
    LiveSkyGateDecision,
    LiveSkyGateThresholds,
    LiveSkyHardwareSpec,
    build_prewired_phase4_slot_manifest,
    validate_live_sky_gate,
)
from .runner import (
    DEFAULT_LOCAL_TEST_SEED32,
    M1_CHAIN_ARTIFACTS,
    M2_CHAIN_ARTIFACTS,
    STAGE1_M3_ARTIFACTS,
    run_xvs_bench_stage2,
)

__all__ = [
    "CROISSANT_1_0_JSON_SCHEMA",
    "DEFAULT_LOCAL_TEST_SEED32",
    "GENESIS_PREV_SHA256",
    "LiveSkyFrameGateInput",
    "LiveSkyGateDecision",
    "LiveSkyGateThresholds",
    "LiveSkyHardwareSpec",
    "M1_CHAIN_ARTIFACTS",
    "M2_CHAIN_ARTIFACTS",
    "PHASE4_SPLIT_NAME",
    "PHASE4_STATUS_PREWIRED",
    "RFC8032_TEST_VECTORS",
    "SIGNING_KEY_SCOPE_LOCAL_TEST",
    "STAGE1_M3_ARTIFACTS",
    "assert_no_secret_material",
    "assert_path_allowed_in_stage2_chain",
    "build_croissant_jsonld",
    "build_prewired_phase4_slot_manifest",
    "build_signed_hash_chain",
    "canonical_json_bytes",
    "cross_verify_with_wsl2_cryptography",
    "ed25519_public_key",
    "ed25519_sign",
    "ed25519_verify",
    "run_xvs_bench_stage2",
    "sha256_bytes",
    "sha256_file",
    "validate_croissant_with_wsl2_jsonschema",
    "validate_live_sky_gate",
    "verify_rfc8032_test_vectors",
    "verify_signed_hash_chain",
]
