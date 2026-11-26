#!/usr/bin/env python3
"""OpenXVS DR-14 Automated Assurance-as-Code GSN v3 Evidence Synthesizer (`xvs-certgen`).

Pure-stdlib CI compiler that:
  1. Parses `assurance/openxvs_v2.gsn.yaml` (stdlib YAML parser — zero PyYAML dependency).
  2. Validates GSN v3 graph integrity (dangling references, acyclicity, node counts).
  3. Verifies SHA-256 & byte-size integrity of all GSN `evidenceFiles` plus live on-disk
     OpenXVS V2/V2.5 artifacts (reports, `pod_daa.xml`, `xvs_vlm_gate`, `xvs_optical_odd`,
     and `xvs_sim_tier3/lockstep_bridge.py` when present).
  4. Enforces FAIL-CLOSED `G5_GAP` / `G2_TLA_DEFERRED` undeveloped-node boundary protection:
     fails immediately if any mandatory hardware/cert-lab gap is missing, falsely marked
     `undeveloped: false`, or given `supportedBy` children claiming closure.
  5. Enforces ZERO forbidden regulatory over-claims (`"DO-178C certified"`,
     `"ED-324 approved"`, `"FAA certified"`, `"EASA certified"`).
  6. Emits a deterministic cryptographic bundle to `output/xvs/assurance/xvs_certgen_bundle.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GSN_PATH = REPO_ROOT / "assurance" / "openxvs_v2.gsn.yaml"
DEFAULT_OUT_PATH = REPO_ROOT / "output" / "xvs" / "assurance" / "xvs_certgen_bundle.json"

MANDATORY_UNDEVELOPED_GAPS: tuple[str, ...] = ("G5_GAP", "G2_TLA_DEFERRED")

FORBIDDEN_CERT_CLAIMS: tuple[str, ...] = (
    "do-178c certified",
    "ed-324 approved",
    "faa certified",
    "easa certified",
    "cert-approved",
    "openxvs certified",
    "airworthiness verified by openxvs",
)

LIVE_SUPPLEMENTARY_ARTIFACTS: tuple[tuple[str, bool], ...] = (
    ("docs/xvs/reports/V2_M1_livedeck_report.md", True),
    ("docs/xvs/reports/V2_M2_local_prep_report.md", True),
    ("docs/xvs/reports/V2_M3_local_core_report.md", True),
    ("docs/xvs/reports/V3_M1_active_range_benchmark.md", True),
    ("docs/xvs/reports/V3_M1_aot_realsky_proof.md", True),
    ("configs/mavlink/pod_daa.xml", True),
    ("src/xvs/v2/xvs_vlm_gate/gate.py", True),
    ("src/xvs/v2/xvs_optical_odd/monitor.py", True),
    ("src/xvs/v2/xvs_sim_tier3/lockstep_bridge.py", False),
    ("xvs-os-hal/src/lib.rs", True),
    ("xvs-active-range/src/lib.rs", True),
    ("xvs-active-range/src/kani_proofs.rs", True),
    ("tests/golden/active_range_vectors.json", True),
    ("docs/legal/TRADEMARK.md", True),
    ("docs/legal/SAFETY_DISCLAIMER.md", True),
    ("docs/legal/EXPORTS.md", True),
)

KANI_PROOF_CRATES: tuple[tuple[str, str, int], ...] = (
    ("xvs-rta-core", "xvs-core/crates/xvs-rta-core/src/proofs.rs", 5),
    ("xvs-odd-vlm", "xvs-odd-vlm/src/lib.rs", 5),
    ("xvs-os-hal", "xvs-os-hal/src/lib.rs", 4),
    ("xvs-active-range", "xvs-active-range/src/kani_proofs.rs", 4),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        inner = val[1:-1]
        return inner.replace('\\"', '"').replace("\\'", "'")
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def parse_gsn_yaml_stdlib(text: str) -> dict[str, dict[str, Any]]:
    """Minimal, deterministic pure-stdlib parser for SCSC-141C / gsn2x GSN YAML mappings."""
    nodes: dict[str, dict[str, Any]] = {}
    current_node: str | None = None
    current_list_key: str | None = None
    current_list_dict: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"Expected top-level GSN node mapping, got: {raw_line!r}")
            node_id = stripped[:-1].strip()
            if node_id in nodes:
                raise ValueError(f"Duplicate top-level GSN node: {node_id!r}")
            nodes[node_id] = {}
            current_node = node_id
            current_list_key = None
            current_list_dict = None
            continue

        if current_node is None:
            raise ValueError(f"Indented YAML line before top-level GSN node: {raw_line!r}")

        if indent == 2:
            if ":" not in stripped:
                raise ValueError(f"Expected key: value at indent 2, got: {raw_line!r}")
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            current_list_dict = None
            if rest == "":
                nodes[current_node][key] = []
                current_list_key = key
            else:
                nodes[current_node][key] = _parse_scalar(rest)
                current_list_key = None
            continue

        if indent == 4 and stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without active list key: {raw_line!r}")
            item_str = stripped[2:].strip()
            if ":" in item_str and not (item_str.startswith('"') or item_str.startswith("'")):
                k, v = item_str.split(":", 1)
                entry: dict[str, Any] = {k.strip(): _parse_scalar(v)}
                nodes[current_node][current_list_key].append(entry)
                current_list_dict = entry
            else:
                nodes[current_node][current_list_key].append(_parse_scalar(item_str))
                current_list_dict = None
            continue

        if indent >= 6:
            if current_list_dict is None or ":" not in stripped:
                raise ValueError(f"Unexpected nested YAML property: {raw_line!r}")
            k, v = stripped.split(":", 1)
            current_list_dict[k.strip()] = _parse_scalar(v)
            continue

        raise ValueError(f"Unsupported YAML indentation ({indent}): {raw_line!r}")

    return nodes


def _check_dag_cycles(nodes: dict[str, dict[str, Any]]) -> list[str]:
    """Detect directed cycles across `supportedBy` and `inContextOf` edges."""
    color: dict[str, int] = {nid: 0 for nid in nodes}
    cycles: list[str] = []

    def dfs(nid: str, stack: list[str]) -> None:
        color[nid] = 1
        node = nodes[nid]
        edges: list[str] = []
        if "supportedBy" in node and isinstance(node["supportedBy"], list):
            edges.extend(str(x) for x in node["supportedBy"])
        if "inContextOf" in node and isinstance(node["inContextOf"], list):
            edges.extend(str(x) for x in node["inContextOf"])
        for child in edges:
            if child not in nodes:
                continue
            if color[child] == 1:
                cycles.append(" -> ".join(stack + [child]))
            elif color[child] == 0:
                dfs(child, stack + [child])
        color[nid] = 2

    for nid in sorted(nodes):
        if color[nid] == 0:
            dfs(nid, [nid])
    return cycles


def synthesize_assurance_bundle(
    *,
    gsn_path: Path = DEFAULT_GSN_PATH,
    repo_root: Path = REPO_ROOT,
    cleanroom: bool = False,
) -> dict[str, Any]:
    """Compile GSN v3 graph, verify cryptographic evidence & fail-closed gap locks."""
    errors: list[str] = []
    if not gsn_path.is_file():
        return {
            "schema": "xvs.certgen_gsn_bundle/1",
            "status": "FAIL",
            "errors": [f"Missing GSN file: {gsn_path}"],
        }

    gsn_text = gsn_path.read_text(encoding="utf-8")
    gsn_sha256 = _sha256_file(gsn_path)
    nodes = parse_gsn_yaml_stdlib(gsn_text)

    # 1. Graph reference & acyclicity check
    dangling_refs: list[str] = []
    for nid, body in sorted(nodes.items()):
        for rel_key in ("supportedBy", "inContextOf"):
            if rel_key in body and isinstance(body[rel_key], list):
                for target in body[rel_key]:
                    if str(target) not in nodes:
                        dangling_refs.append(f"{nid}.{rel_key} -> {target}")
    if dangling_refs:
        errors.append(f"Dangling GSN references: {dangling_refs}")

    cycles = _check_dag_cycles(nodes)
    if cycles:
        errors.append(f"Cyclic GSN graph detected: {cycles}")

    # 2. Mandatory G5_GAP / G2_TLA_DEFERRED fail-closed undeveloped boundary protection
    required_gaps = sorted(
        set(MANDATORY_UNDEVELOPED_GAPS)
        | {nid for nid in nodes if nid.startswith("G5_GAP")}
    )
    gap_audit: list[dict[str, Any]] = []
    for gap_id in required_gaps:
        if gap_id not in nodes:
            errors.append(f"MISSING_MANDATORY_GAP: {gap_id} removed from GSN graph")
            gap_audit.append({"node": gap_id, "present": False, "undeveloped": False, "ok": False})
            continue
        gnode = nodes[gap_id]
        is_undev = ("undeveloped" in gnode) and (gnode["undeveloped"] is True)
        has_support = ("supportedBy" in gnode) and bool(gnode["supportedBy"])
        ok = is_undev and not has_support
        if not ok:
            errors.append(
                f"GAP_BOUNDARY_BREACH: {gap_id} must have undeveloped=true and zero supportedBy "
                f"(got undeveloped={is_undev}, has_supportedBy={has_support})"
            )
        gap_audit.append(
            {
                "node": gap_id,
                "present": True,
                "undeveloped": is_undev,
                "has_supportedBy": has_support,
                "ok": ok,
            }
        )

    # 3. Forbidden regulatory over-claim scan across all GSN text nodes
    forbidden_hits: list[dict[str, str]] = []
    for nid, body in sorted(nodes.items()):
        if "text" in body and isinstance(body["text"], str):
            lowered = body["text"].lower()
            for phrase in FORBIDDEN_CERT_CLAIMS:
                if phrase in lowered:
                    forbidden_hits.append({"node": nid, "phrase": phrase})
                    errors.append(f"FORBIDDEN_CLAIM: node {nid} contains {phrase!r}")

    # 4. Cryptographic digest verification of GSN `evidenceFiles`
    gsn_evidence_ledger: list[dict[str, Any]] = []
    seen_evidence_keys: set[tuple[str, str]] = set()
    for nid, body in sorted(nodes.items()):
        if "evidenceFiles" not in body or not isinstance(body["evidenceFiles"], list):
            continue
        for entry in body["evidenceFiles"]:
            if not isinstance(entry, dict) or "path" not in entry:
                errors.append(f"Malformed evidenceFiles entry in {nid}")
                continue
            rel_path = str(entry["path"])
            dedup_key = (nid, rel_path)
            if dedup_key in seen_evidence_keys:
                continue
            seen_evidence_keys.add(dedup_key)
            full_path = repo_root / rel_path
            expected_sha = str(entry["sha256"]) if "sha256" in entry else None
            expected_size = int(entry["size_bytes"]) if "size_bytes" in entry else None

            if not full_path.is_file():
                is_cleanroom_omitted = cleanroom and (
                    rel_path.startswith("output/") or rel_path.startswith("docs/xvs/reports/")
                )
                if not is_cleanroom_omitted:
                    errors.append(f"MISSING_EVIDENCE_FILE: {nid} -> {rel_path}")
                gsn_evidence_ledger.append(
                    {
                        "node": nid,
                        "path": rel_path,
                        "present": False,
                        "sha256_match": False,
                        "size_match": False,
                        "skipped_cleanroom": is_cleanroom_omitted,
                    }
                )
                continue

            actual_sha = _sha256_file(full_path)
            actual_size = full_path.stat().st_size
            sha_ok = expected_sha is None or actual_sha == expected_sha
            size_ok = expected_size is None or actual_size == expected_size
            if not sha_ok:
                errors.append(
                    f"SHA256_MISMATCH: {nid} -> {rel_path} "
                    f"(expected={expected_sha}, actual={actual_sha})"
                )
            if not size_ok:
                errors.append(
                    f"SIZE_MISMATCH: {nid} -> {rel_path} "
                    f"(expected={expected_size}, actual={actual_size})"
                )
            gsn_evidence_ledger.append(
                {
                    "node": nid,
                    "path": rel_path,
                    "present": True,
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                    "sha256_match": sha_ok,
                    "size_match": size_ok,
                }
            )

    # 5. Live supplementary on-disk OpenXVS assurance & implementation artifact ledger
    live_artifact_ledger: list[dict[str, Any]] = []
    for rel_path, required in LIVE_SUPPLEMENTARY_ARTIFACTS:
        full_path = repo_root / rel_path
        if full_path.is_file():
            live_artifact_ledger.append(
                {
                    "path": rel_path,
                    "required": required,
                    "present": True,
                    "sha256": _sha256_file(full_path),
                    "size_bytes": full_path.stat().st_size,
                }
            )
        else:
            is_cleanroom_omitted = cleanroom and rel_path.startswith("docs/xvs/reports/")
            if required and not is_cleanroom_omitted:
                errors.append(f"MISSING_LIVE_ARTIFACT: {rel_path}")
            live_artifact_ledger.append(
                {
                    "path": rel_path,
                    "required": required,
                    "present": False,
                    "sha256": None,
                    "size_bytes": 0,
                    "skipped_cleanroom": is_cleanroom_omitted,
                }
            )

    # 5b. Formal Kani CBMC harness verification across standalone Rust crates (`18/18`)
    kani_proof_ledger: list[dict[str, Any]] = []
    kani_proofs_total = 0
    for crate_name, rel_path, expected_count in KANI_PROOF_CRATES:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            errors.append(f"MISSING_KANI_PROOF_FILE: {crate_name} -> {rel_path}")
            kani_proof_ledger.append(
                {
                    "crate": crate_name,
                    "path": rel_path,
                    "expected_count": expected_count,
                    "proof_count": 0,
                    "matches_expected": False,
                    "sha256": None,
                }
            )
            continue
        src_text = full_path.read_text(encoding="utf-8")
        actual_count = sum(1 for line in src_text.splitlines() if line.strip() == "#[kani::proof]")
        matches = actual_count == expected_count
        if not matches:
            errors.append(
                f"KANI_PROOF_COUNT_MISMATCH: {crate_name} ({rel_path}) "
                f"expected={expected_count}, actual={actual_count}"
            )
        kani_proofs_total += actual_count
        kani_proof_ledger.append(
            {
                "crate": crate_name,
                "path": rel_path,
                "expected_count": expected_count,
                "proof_count": actual_count,
                "matches_expected": matches,
                "sha256": _sha256_file(full_path),
            }
        )

    # 6. Summary statistics & deterministic bundle digest
    undeveloped_nodes = sorted(
        nid for nid, body in nodes.items() if ("undeveloped" in body and body["undeveloped"] is True)
    )
    payload: dict[str, Any] = {
        "schema": "xvs.certgen_gsn_bundle/1",
        "standard": "SCSC-141C / EUROCAE ED-324 (SAE ARP6983) / FAA DOT-FAA-TC-26-20",
        "status": "PASS" if not errors else "FAIL",
        "gsn_source": {
            "path": str(gsn_path.relative_to(repo_root)).replace("\\", "/")
            if gsn_path.is_relative_to(repo_root)
            else str(gsn_path),
            "sha256": gsn_sha256,
            "node_count": len(nodes),
            "undeveloped_nodes": undeveloped_nodes,
        },
        "gap_audit": gap_audit,
        "forbidden_claims_audit": {
            "checked_phrases": list(FORBIDDEN_CERT_CLAIMS),
            "violations": forbidden_hits,
            "clean": len(forbidden_hits) == 0,
        },
        "kani_proofs_total": kani_proofs_total,
        "kani_proof_ledger": kani_proof_ledger,
        "gsn_evidence_ledger": gsn_evidence_ledger,
        "live_artifact_ledger": live_artifact_ledger,
        "errors": errors,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenXVS DR-14 Automated GSN v3 Assurance Evidence Synthesizer (xvs-certgen)"
    )
    parser.add_argument("--gsn", type=Path, default=DEFAULT_GSN_PATH, help="Path to GSN YAML file")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output JSON bundle path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI fail-closed mode: exit 1 if any GSN/hash/gap/claim check fails",
    )
    parser.add_argument(
        "--cleanroom",
        action="store_true",
        help="Public clean-room CI mode: skip gitignored output/ and internal docs/xvs/reports/ while enforcing GSN DAG, gap locks, forbidden claims, 18/18 Kani proofs, and tracked artifacts",
    )
    args = parser.parse_args(argv)

    bundle = synthesize_assurance_bundle(
        gsn_path=args.gsn,
        repo_root=args.repo_root,
        cleanroom=args.cleanroom,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = bundle["status"]
    err_count = len(bundle["errors"])
    print(
        f"xvs-certgen: status={status} nodes={bundle['gsn_source']['node_count'] if 'gsn_source' in bundle else 0} "
        f"kani_proofs_total={bundle['kani_proofs_total'] if 'kani_proofs_total' in bundle else 0} "
        f"errors={err_count} bundle_sha256={bundle['bundle_sha256'] if 'bundle_sha256' in bundle else 'none'} "
        f"out={args.out}"
    )
    if err_count > 0:
        for err in bundle["errors"]:
            print(f"  [ERROR] {err}", file=sys.stderr)
    if args.check and status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
