"""Verify and optionally copy G-B14 output artifacts with SHA-256 checks.

Dry-run by default: reports what it would copy and whether SHA-256 digests match
`output/xvs/wp07/g0/data_manifest.json` (`file_sha256`) and
`output/xvs/wp05/g0_lambda_sweep_v2/{lambda_sweep_v2_results,lambda_sweep_persistent_clutter}.json`
(`per_run_records_sha256`).

Safety invariants:
1. Never copies or modifies anything unless `--execute` is explicitly passed.
2. Performs all manifest and source SHA-256 checks before copying a single byte;
   if any hash check fails, zero files are copied even with `--execute`.
3. Never deletes any file or directory in either source or destination.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_ROOT = REPO_ROOT / "output"
DEFAULT_DST_ROOT = REPO_ROOT / "output"

SUBTREES = (
    Path("xvs/wp07/g0"),
    Path("xvs/wp05/g0_lambda_sweep_v2"),
)


@dataclass
class FileCheckRecord:
    """Verification and copy status for one artifact file."""

    rel_path: str
    src_path: str
    dst_path: str
    src_exists: bool
    src_sha256: str | None
    expected_sha256: str | None
    manifest_source: str | None
    hash_matches_manifest: bool
    dst_exists_before: bool
    dst_sha256_before: str | None
    action: str
    dst_sha256_after: str | None = None
    copied_verified: bool | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_output_relpath(raw_path: str) -> Path:
    """Normalize a manifest path like 'output\\xvs\\wp07\\g0\\foo.jsonl' to 'xvs/wp07/g0/foo.jsonl'."""
    norm = raw_path.replace("\\", "/").lstrip("/")
    if norm.startswith("output/"):
        norm = norm[len("output/") :]
    return Path(norm)


def collect_expected_hashes(src_root: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Load expected SHA-256 digests from WP-07 data_manifest.json and WP-05 sweep JSONs.

    Returns:
        expected_map: mapping from POSIX relative path (under output/) to
            (expected_sha256_hex, manifest_citation).
        errors: list of human-readable error strings if any manifest is missing or malformed.
    """
    expected_map: dict[str, tuple[str, str]] = {}
    errors: list[str] = []

    # 1. WP-07 G0 data_manifest.json
    wp07_manifest_path = src_root / "xvs" / "wp07" / "g0" / "data_manifest.json"
    if not wp07_manifest_path.is_file():
        errors.append(f"Missing WP-07 manifest: {wp07_manifest_path}")
    else:
        try:
            wp07_manifest = json.loads(wp07_manifest_path.read_text(encoding="utf-8"))
            files_dict = wp07_manifest.get("files", {})
            sha_dict = wp07_manifest.get("file_sha256", {})
            if not isinstance(files_dict, dict) or not isinstance(sha_dict, dict) or not sha_dict:
                errors.append(
                    f"Malformed 'files' or 'file_sha256' in {wp07_manifest_path}"
                )
            else:
                for key, expected_sha in sha_dict.items():
                    raw_rel = files_dict.get(key)
                    if not raw_rel or not isinstance(expected_sha, str):
                        errors.append(
                            f"Manifest key '{key}' missing file path or sha256 in {wp07_manifest_path}"
                        )
                        continue
                    rel_posix = _normalize_output_relpath(raw_rel).as_posix()
                    expected_map[rel_posix] = (
                        expected_sha.lower(),
                        f"xvs/wp07/g0/data_manifest.json:file_sha256.{key}",
                    )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed reading {wp07_manifest_path}: {exc}")

    # 2. WP-05 G0 lambda_sweep_v2_results.json and lambda_sweep_persistent_clutter.json
    wp05_specs = (
        ("xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json", "per_run_records.jsonl"),
        (
            "xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_persistent_clutter.json",
            "per_run_records_persistent.jsonl",
        ),
    )
    for rel_json, default_records_name in wp05_specs:
        json_path = src_root / Path(rel_json)
        if not json_path.is_file():
            errors.append(f"Missing WP-05 results JSON: {json_path}")
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            expected_sha = payload.get("per_run_records_sha256")
            raw_records_file = payload.get(
                "per_run_records_file",
                f"output/xvs/wp05/g0_lambda_sweep_v2/{default_records_name}",
            )
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                errors.append(
                    f"Missing or invalid 'per_run_records_sha256' in {json_path}"
                )
                continue
            rel_posix = _normalize_output_relpath(str(raw_records_file)).as_posix()
            expected_map[rel_posix] = (
                expected_sha.lower(),
                f"{rel_json}:per_run_records_sha256",
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed reading {json_path}: {exc}")

    return expected_map, errors


def verify_and_copy(
    src_root: Path = DEFAULT_SRC_ROOT,
    dst_root: Path = DEFAULT_DST_ROOT,
    execute: bool = False,
) -> dict[str, Any]:
    """Verify SHA-256 hashes of G-B14 output trees and optionally copy them.

    Never deletes any file. Never writes to `dst_root` when `execute=False` or when any
    verification check fails.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    expected_map, errors = collect_expected_hashes(src_root)

    # Collect all candidate relative file paths across the target subtrees
    rel_paths_set: set[str] = set(expected_map.keys())
    for subtree in SUBTREES:
        subtree_dir = src_root / subtree
        if not subtree_dir.is_dir():
            errors.append(f"Missing source subtree directory: {subtree_dir}")
            continue
        for p in sorted(subtree_dir.rglob("*")):
            if p.is_file():
                rel_paths_set.add(p.relative_to(src_root).as_posix())

    records: list[FileCheckRecord] = []
    for rel_posix in sorted(rel_paths_set):
        src_file = src_root / Path(rel_posix)
        dst_file = dst_root / Path(rel_posix)
        exp_entry = expected_map.get(rel_posix)
        expected_sha = exp_entry[0] if exp_entry else None
        manifest_source = exp_entry[1] if exp_entry else None

        src_exists = src_file.is_file()
        src_sha = sha256_file(src_file) if src_exists else None

        if not src_exists:
            hash_matches = False
            errors.append(f"Missing source file: {src_file}")
        elif expected_sha is not None and src_sha != expected_sha:
            hash_matches = False
            errors.append(
                f"SHA-256 mismatch for {rel_posix}: computed={src_sha}, "
                f"expected={expected_sha} ({manifest_source})"
            )
        else:
            hash_matches = True

        dst_exists_before = dst_file.is_file()
        dst_sha_before = sha256_file(dst_file) if dst_exists_before else None

        if not hash_matches:
            action = "BLOCKED_HASH_MISMATCH"
        elif dst_exists_before and dst_sha_before == src_sha:
            action = "ALREADY_IDENTICAL"
        else:
            action = "WOULD_COPY"

        records.append(
            FileCheckRecord(
                rel_path=rel_posix,
                src_path=str(src_file),
                dst_path=str(dst_file),
                src_exists=src_exists,
                src_sha256=src_sha,
                expected_sha256=expected_sha,
                manifest_source=manifest_source,
                hash_matches_manifest=hash_matches,
                dst_exists_before=dst_exists_before,
                dst_sha256_before=dst_sha_before,
                action=action,
            )
        )

    all_verified = len(errors) == 0 and all(r.hash_matches_manifest for r in records)

    # Copy phase: runs ONLY if execute=True AND all_verified=True
    copied_count = 0
    if execute and all_verified:
        for rec in records:
            src_file = Path(rec.src_path)
            dst_file = Path(rec.dst_path)
            if rec.action == "WOULD_COPY":
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied_count += 1
                rec.action = "COPIED"
            dst_sha_after = sha256_file(dst_file)
            rec.dst_sha256_after = dst_sha_after
            rec.copied_verified = (dst_sha_after == rec.src_sha256) and (
                rec.expected_sha256 is None or dst_sha_after == rec.expected_sha256
            )
            if not rec.copied_verified:
                errors.append(
                    f"Post-copy SHA-256 verification failed for {rec.dst_path}"
                )
        all_verified = len(errors) == 0

    return {
        "mode": "EXECUTE" if execute else "DRY_RUN",
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "all_verified": all_verified,
        "manifest_pinned_files_checked": len(expected_map),
        "total_files_checked": len(records),
        "would_copy_count": sum(
            1 for r in records if r.action in ("WOULD_COPY", "COPIED")
        ),
        "copied_count": copied_count,
        "errors": errors,
        "files": [asdict(r) for r in records],
    }


def _print_report(summary: dict[str, Any]) -> None:
    mode = summary["mode"]
    print(f"=== verify_copy_sha256 [{mode}] ===")
    print(f"Source root      : {summary['src_root']}")
    print(f"Destination root : {summary['dst_root']}")
    print(
        f"Files checked    : {summary['total_files_checked']} "
        f"(manifest-pinned: {summary['manifest_pinned_files_checked']})"
    )
    print("-" * 80)
    for item in summary["files"]:
        pin_tag = (
            f" [PINNED: {item['manifest_source']}]"
            if item["manifest_source"]
            else " [AUX]"
        )
        status = "MATCH" if item["hash_matches_manifest"] else "MISMATCH"
        print(
            f"[{status}] [{item['action']}] {item['rel_path']}{pin_tag}\n"
            f"       src_sha256={item['src_sha256']}"
        )
        if item["expected_sha256"] is not None:
            print(f"       exp_sha256={item['expected_sha256']}")
    print("-" * 80)
    if summary["errors"]:
        print("ERRORS:")
        for err in summary["errors"]:
            print(f"  - {err}")
    print(
        f"Summary: mode={mode}, all_verified={summary['all_verified']}, "
        f"would_copy={summary['would_copy_count']}, copied={summary['copied_count']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify SHA-256 digests of WP-07 G0 and WP-05 G0 lambda_sweep_v2 outputs "
            "against data_manifest.json and per_run_records_sha256. Dry-run by default."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help="Source output root directory (default: <repo>/output).",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help="Destination output root directory (default: <repo>/output).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy verified files into --dst-root. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary to stdout.",
    )
    args = parser.parse_args(argv)

    summary = verify_and_copy(
        src_root=args.src_root,
        dst_root=args.dst_root,
        execute=args.execute,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)

    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
