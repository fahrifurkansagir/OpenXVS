"""Chained SHA-256 + Ed25519 hash-chain ledger for XVS-Bench (DR-08 / SP3 §2.1 & §3.2).

Enforces:
  - Strict acyclic Stage-2 DAG (`BC-M3-M2`): ONLY M1, M2, and Stage-1 M3 artifacts may enter
    the hash chain; `assurance/*`, Stage-3, Stage-4, and Stage-2 self-outputs are hard-rejected.
  - Zero private-key serialization (`BC-M3-m5`): ONLY `public_key_hex` and `signature_hex` are
    stored alongside `"signing_key_scope": "LOCAL_DETERMINISTIC_TEST_SEED_NON_PRODUCTION"`.
  - Dual-run deterministic closure (`Run 1 == Run 2` byte/digest exact) & 1-byte tamper detection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .ed25519_pure import ed25519_public_key, ed25519_sign, ed25519_verify

GENESIS_PREV_SHA256: str = "0" * 64
SIGNING_KEY_SCOPE_LOCAL_TEST: str = "LOCAL_DETERMINISTIC_TEST_SEED_NON_PRODUCTION"

_FORBIDDEN_STAGE2_EXACT: frozenset[str] = frozenset(
    {
        "output/xvs/v2_m3/xvs_bench_chain.json",
        "output/xvs/v2_m3/xvs_bench_croissant.json",
        "output/xvs/v2_m3/v3_xvs_bench_evidence.json",
        "output/xvs/v2_m3/v4_ed324_gsn_evidence.json",
        "output/xvs/v2_m3/v5_bridge_compliance_evidence.json",
        "output/xvs/v2_m3/v6_baseline_evidence.json",
        "output/xvs/v2_m3/run_manifest.json",
        "docs/xvs/reports/V2_M3_local_core_report.md",
        "docs/xvs/reports/V2_M3_local_core_report.md.provenance.json",
    }
)

_FORBIDDEN_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "private_key",
        "private_key_hex",
        "secret_key",
        "secret_key_hex",
        "signing_seed",
        "signing_seed_hex",
        "seed_hex",
    }
)


def canonical_json_bytes(obj: Any) -> bytes:
    """RFC 8785-style canonical deterministic UTF-8 JSON bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


_ALLOWED_STAGE2_STAGE1_M3: frozenset[str] = frozenset(
    {
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
        "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
    }
)


def assert_path_allowed_in_stage2_chain(rel_posix_path: str) -> None:
    """Enforce SP3 §2.1 strict acyclic 4-stage DAG (BC-M3-M2)."""
    raw = rel_posix_path.replace("\\", "/")
    if raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError(
            f"Acyclic DAG violation (BC-M3-M2): absolute path forbidden in Stage-2 chain: {rel_posix_path!r}"
        )
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(
            f"Acyclic DAG violation (BC-M3-M2): path traversal forbidden in Stage-2 chain: {rel_posix_path!r}"
        )
    norm = "/".join(parts)
    if norm.startswith("assurance/") or norm == "assurance":
        raise ValueError(
            f"Acyclic DAG violation (BC-M3-M2): Stage-2 XVS-Bench MUST NOT hash Stage-3 path {norm!r}"
        )
    if norm in _FORBIDDEN_STAGE2_EXACT:
        raise ValueError(
            f"Acyclic DAG violation (BC-M3-M2): Stage-2 XVS-Bench MUST NOT hash downstream/self path {norm!r}"
        )
    if not (
        norm.startswith("output/xvs/v2_m1/")
        or norm.startswith("output/xvs/v2_m2/")
        or norm in _ALLOWED_STAGE2_STAGE1_M3
    ):
        raise ValueError(
            f"Acyclic DAG violation (BC-M3-M2): path outside M1/M2/Stage-1-M3 allowlist: {norm!r}"
        )


def _scan_for_forbidden_secret_keys(obj: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in _FORBIDDEN_SECRET_KEYS:
                hits.append(f"{path}.{k}")
            hits.extend(_scan_for_forbidden_secret_keys(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_scan_for_forbidden_secret_keys(v, f"{path}[{i}]"))
    return hits


def assert_no_secret_material(obj: Any) -> None:
    """Fail-closed check ensuring zero private keys or signing seeds appear in serialized objects."""
    hits = _scan_for_forbidden_secret_keys(obj)
    if hits:
        raise ValueError(f"Forbidden secret key material detected (BC-M3-m5): {hits}")


def build_signed_hash_chain(
    repo_root: Path,
    artifact_rel_paths: Sequence[str],
    signing_seed32: bytes,
    *,
    signing_key_scope: str = SIGNING_KEY_SCOPE_LOCAL_TEST,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic SHA-256 hash chain over real on-disk artifacts & sign root with Ed25519.

    NEVER stores `signing_seed32` or any private-key material in the returned document.
    """
    if len(signing_seed32) != 32:
        raise ValueError(f"signing_seed32 must be 32 bytes, got {len(signing_seed32)}")
    if not artifact_rel_paths:
        raise ValueError("artifact_rel_paths must be non-empty")

    norm_paths = sorted(set(p.replace("\\", "/").lstrip("./") for p in artifact_rel_paths))
    entries: list[dict[str, Any]] = []
    prev_hash = GENESIS_PREV_SHA256

    for idx, rel in enumerate(norm_paths):
        assert_path_allowed_in_stage2_chain(rel)
        abs_path = repo_root / rel
        if not abs_path.is_file():
            raise FileNotFoundError(f"Missing artifact for XVS-Bench hash chain: {abs_path}")
        file_sha = sha256_file(abs_path)
        size_bytes = int(abs_path.stat().st_size)
        link_payload = {
            "index": idx,
            "path": rel,
            "bytes": size_bytes,
            "file_sha256": file_sha,
            "prev_sha256": prev_hash,
        }
        entry_sha = sha256_bytes(canonical_json_bytes(link_payload))
        entries.append(
            {
                **link_payload,
                "entry_sha256": entry_sha,
            }
        )
        prev_hash = entry_sha

    meta_block = {} if metadata is None else metadata
    root_payload = {
        "schema": "xvs.v2_m3.xvs_bench_hash_chain/1",
        "signing_key_scope": signing_key_scope,
        "genesis_prev_sha256": GENESIS_PREV_SHA256,
        "entry_count": len(entries),
        "tail_entry_sha256": prev_hash,
        "entries": entries,
        "metadata": meta_block,
    }
    root_digest = sha256_bytes(canonical_json_bytes(root_payload))
    pub_bytes = ed25519_public_key(signing_seed32)
    sig_bytes = ed25519_sign(signing_seed32, root_digest.encode("ascii"))

    chain_doc: dict[str, Any] = {
        **root_payload,
        "root_digest_sha256": root_digest,
        "signature_algorithm": "Ed25519-RFC8032",
        "public_key_hex": pub_bytes.hex(),
        "signature_hex": sig_bytes.hex(),
    }
    return chain_doc


def verify_signed_hash_chain(
    chain_doc: dict[str, Any],
    repo_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Verify hash-chain linkage, Ed25519 signature, zero secret leakage, and optional disk SHA-256s."""
    errors: list[str] = []

    secret_hits = _scan_for_forbidden_secret_keys(chain_doc)
    if secret_hits:
        errors.append(f"FORBIDDEN_SECRET_KEYS_SERIALIZED: {secret_hits}")

    required = (
        "schema",
        "signing_key_scope",
        "genesis_prev_sha256",
        "entry_count",
        "tail_entry_sha256",
        "entries",
        "metadata",
        "root_digest_sha256",
        "public_key_hex",
        "signature_hex",
    )
    for k in required:
        if k not in chain_doc:
            errors.append(f"MISSING_REQUIRED_FIELD: {k}")
    if errors:
        return False, errors

    entries = chain_doc["entries"]
    if not isinstance(entries, list) or len(entries) != int(chain_doc["entry_count"]):
        errors.append("ENTRY_COUNT_MISMATCH")
        return False, errors

    prev_hash = GENESIS_PREV_SHA256
    if chain_doc["genesis_prev_sha256"] != GENESIS_PREV_SHA256:
        errors.append("INVALID_GENESIS_PREV_SHA256")

    for idx, entry in enumerate(entries):
        if int(entry["index"]) != idx:
            errors.append(f"INDEX_MISMATCH_AT_{idx}")
        if entry["prev_sha256"] != prev_hash:
            errors.append(f"PREV_SHA256_BROKEN_AT_{idx}")
        try:
            assert_path_allowed_in_stage2_chain(str(entry["path"]))
        except ValueError as exc:
            errors.append(f"ACYCLIC_DAG_VIOLATION_AT_{idx}: {exc}")

        recomputed_entry = sha256_bytes(
            canonical_json_bytes(
                {
                    "index": entry["index"],
                    "path": entry["path"],
                    "bytes": entry["bytes"],
                    "file_sha256": entry["file_sha256"],
                    "prev_sha256": entry["prev_sha256"],
                }
            )
        )
        if recomputed_entry != entry["entry_sha256"]:
            errors.append(f"ENTRY_SHA256_TAMPERED_AT_{idx}")

        if repo_root is not None:
            disk_path = repo_root / str(entry["path"])
            if not disk_path.is_file():
                errors.append(f"MISSING_DISK_FILE: {entry['path']}")
            else:
                disk_sha = sha256_file(disk_path)
                if disk_sha != entry["file_sha256"]:
                    errors.append(f"DISK_SHA256_MISMATCH: {entry['path']}")
        prev_hash = str(entry["entry_sha256"])

    if prev_hash != chain_doc["tail_entry_sha256"]:
        errors.append("TAIL_ENTRY_SHA256_MISMATCH")

    root_payload = {
        "schema": chain_doc["schema"],
        "signing_key_scope": chain_doc["signing_key_scope"],
        "genesis_prev_sha256": chain_doc["genesis_prev_sha256"],
        "entry_count": chain_doc["entry_count"],
        "tail_entry_sha256": chain_doc["tail_entry_sha256"],
        "entries": chain_doc["entries"],
        "metadata": chain_doc["metadata"],
    }
    expected_root = sha256_bytes(canonical_json_bytes(root_payload))
    if expected_root != chain_doc["root_digest_sha256"]:
        errors.append("ROOT_DIGEST_SHA256_MISMATCH")

    try:
        pub = bytes.fromhex(str(chain_doc["public_key_hex"]))
        sig = bytes.fromhex(str(chain_doc["signature_hex"]))
    except ValueError:
        errors.append("MALFORMED_HEX_KEY_OR_SIGNATURE")
        return False, errors

    if not ed25519_verify(pub, str(chain_doc["root_digest_sha256"]).encode("ascii"), sig):
        errors.append("ED25519_SIGNATURE_INVALID")

    return (len(errors) == 0), errors
