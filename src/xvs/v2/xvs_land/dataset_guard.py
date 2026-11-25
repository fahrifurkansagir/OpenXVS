"""Fail-closed LARD real-airport manifest & KSFO held-out split guard (`BC-M2-M3`).

Enforces SP2_M2_MODELS_DESIGN.md §3.4:
  - Requires a verified `lard_real_airports_manifest` containing >= 36 unique real LARD ICAO codes
    (never fabricated or placeholder entries!) before ANY training split is cleared.
  - Enforces `"KSFO"` strictly in `heldout_icaos` and NEVER in `train_icaos`.
  - Programmatically asserts 0% ICAO overlap between `train_icaos` and `heldout_icaos`, AND 0%
    overlap between `train_icaos` and `lard_real_airports` (LARD real split is eval-only `H3`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

MIN_LARD_REAL_AIRPORTS: int = 36
REQUIRED_HELDOUT_AIRPORTS: frozenset[str] = frozenset({"KSFO"})
ICAO_PATTERN = re.compile(r"^[A-Z0-9]{4}$")
FORBIDDEN_PLACEHOLDER_PREFIXES: tuple[str, ...] = ("FAKE", "MOCK", "TEST", "XXXX", "ZZZZ")


@dataclass(frozen=True)
class DatasetSplitVerification:
    """Verified zero-leakage airport split summary."""

    manifest_verified: bool
    lard_real_count: int
    train_count: int
    heldout_count: int
    ksfo_heldout: bool
    train_heldout_overlap: tuple[str, ...]
    train_lard_real_overlap: tuple[str, ...]


def _normalize_icaos(icaos: Sequence[str], label: str) -> set[str]:
    out: set[str] = set()
    for raw in icaos:
        if not isinstance(raw, str):
            raise ValueError(f"{label} contains non-string ICAO entry: {raw!r}")
        code = raw.strip().upper()
        if not ICAO_PATTERN.match(code):
            raise ValueError(f"{label} contains invalid 4-char ICAO code: {raw!r}")
        if any(code.startswith(pfx) for pfx in FORBIDDEN_PLACEHOLDER_PREFIXES):
            raise PermissionError(f"{label} contains forbidden fabricated/placeholder ICAO: {code!r}")
        out.add(code)
    return out


def validate_lard_real_airports_manifest(
    manifest: Path | str | Mapping[str, Any] | None,
    train_icaos: Sequence[str],
    heldout_icaos: Sequence[str],
) -> DatasetSplitVerification:
    """Fail-closed validation of LARD real-airport manifest (`>= 36` unique ICAOs) & zero split leakage."""
    if manifest is None:
        raise PermissionError(
            "Fail-closed BC-M2-M3 guard: missing lard_real_airports_manifest (cannot clear training split)."
        )

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if not path.is_file():
            raise PermissionError(
                f"Fail-closed BC-M2-M3 guard: lard_real_airports_manifest file not found at {path}."
            )
        doc: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    else:
        doc = manifest

    if "verified" not in doc or doc["verified"] is not True:
        raise PermissionError(
            "Fail-closed BC-M2-M3 guard: lard_real_airports_manifest must explicitly set verified=true."
        )
    if "source" not in doc or not isinstance(doc["source"], str) or not doc["source"].strip():
        raise PermissionError("Fail-closed BC-M2-M3 guard: manifest missing provenance 'source'.")
    src_lower = doc["source"].strip().lower()
    if "fabricat" in src_lower or "placeholder" in src_lower or "dummy" in src_lower:
        raise PermissionError(
            f"Fail-closed BC-M2-M3 guard: fabricated manifest source rejected ({doc['source']!r})."
        )
    if "airports" not in doc or not isinstance(doc["airports"], (list, tuple)):
        raise PermissionError("Fail-closed BC-M2-M3 guard: manifest missing 'airports' list.")

    lard_set = _normalize_icaos(doc["airports"], "lard_real_airports_manifest")
    if len(lard_set) < MIN_LARD_REAL_AIRPORTS:
        raise PermissionError(
            f"Fail-closed BC-M2-M3 guard: LARD real manifest has only {len(lard_set)} unique ICAOs "
            f"(minimum required >= {MIN_LARD_REAL_AIRPORTS})."
        )

    train_set = _normalize_icaos(train_icaos, "train_icaos")
    heldout_set = _normalize_icaos(heldout_icaos, "heldout_icaos")
    if len(train_set) == 0:
        raise ValueError("train_icaos must be non-empty")

    for req in REQUIRED_HELDOUT_AIRPORTS:
        if req not in heldout_set:
            raise PermissionError(f"Fail-closed guard: required held-out airport {req!r} missing from heldout_icaos.")
        if req in train_set:
            raise PermissionError(f"Fail-closed guard: held-out airport {req!r} leaked into train_icaos.")

    overlap_th = tuple(sorted(train_set.intersection(heldout_set)))
    if overlap_th:
        raise PermissionError(
            f"Fail-closed guard: {len(overlap_th)} ICAO(s) overlap between train and heldout: {overlap_th}"
        )

    overlap_lard = tuple(sorted(train_set.intersection(lard_set)))
    if overlap_lard:
        raise PermissionError(
            f"Fail-closed guard: {len(overlap_lard)} eval-only LARD real ICAO(s) leaked into train: {overlap_lard}"
        )

    return DatasetSplitVerification(
        manifest_verified=True,
        lard_real_count=len(lard_set),
        train_count=len(train_set),
        heldout_count=len(heldout_set),
        ksfo_heldout=("KSFO" in heldout_set and "KSFO" not in train_set),
        train_heldout_overlap=overlap_th,
        train_lard_real_overlap=overlap_lard,
    )
