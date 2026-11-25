"""Fetch & cross-verify real `deel-ai/LARD` ICAO exclusion manifest & disjoint CONUS training splits.

Pure Python stdlib metadata-only synchronization script (ZERO LARD imagery downloaded or redistributed):
  1. Downloads public ICAO metadata JSON `https://raw.githubusercontent.com/deel-ai/LARD/main/data/runways_database.json`.
  2. Cross-references every ICAO key against local Public Domain `data/reference/ourairports/runways.csv`
     to verify genuine real-world 4-character ICAO airport codes (`>= 36` unique ICAOs).
  3. Builds disjoint CONUS paved, lighted (`>= 1,000 m` / `>= 3,281 ft`) training & held-out splits
     (`KSFO` strictly in `heldout_icaos`, `0%` overlap with LARD or `heldout_icaos`).
  4. Validates the generated manifests fail-closed via `validate_lard_real_airports_manifest(...)` before writing:
     - `data/reference/lard_real_airports_manifest.json`
     - `data/reference/xvs_land_airport_splits.json`
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.xvs_land.dataset_guard import (
    MIN_LARD_REAL_AIRPORTS,
    validate_lard_real_airports_manifest,
)

LARD_RUNWAYS_DB_URL: str = (
    "https://raw.githubusercontent.com/deel-ai/LARD/main/data/runways_database.json"
)
DEFAULT_OURAIRPORTS_RUNWAYS_CSV: Path = (
    REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"
)
DEFAULT_LARD_MANIFEST_OUT: Path = (
    REPO_ROOT / "data" / "reference" / "lard_real_airports_manifest.json"
)
DEFAULT_SPLITS_OUT: Path = (
    REPO_ROOT / "data" / "reference" / "xvs_land_airport_splits.json"
)
ICAO_RE = re.compile(r"^[A-Z0-9]{4}$")
PAVED_SURFACE_TOKENS: tuple[str, ...] = ("ASP", "CON", "BIT", "PEM")
MIN_RUNWAY_LENGTH_M: float = 1000.0
FT_TO_M: float = 0.3048
PINNED_HELDOUT_CONUS: tuple[str, ...] = ("KSFO", "KOAK", "KSJC")


def fetch_lard_runways_metadata(
    url: str = LARD_RUNWAYS_DB_URL,
    *,
    timeout_s: float = 30.0,
) -> tuple[bytes, str, dict[str, Any]]:
    """Fetch public `deel-ai/LARD` `runways_database.json` metadata (zero imagery)."""
    req = urllib.request.Request(url, headers={"User-Agent": "OpenXVS-v2-MetadataSync/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw_bytes = resp.read()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    parsed = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(parsed, dict) or len(parsed) == 0:
        raise ValueError(f"Expected non-empty dict in {url}, got {type(parsed).__name__}")
    return raw_bytes, digest, parsed


def load_ourairports_catalog(
    runways_csv: Path,
) -> tuple[set[str], list[str], str]:
    """Load all 4-char ICAO codes and paved lighted CONUS (`>= 1000 m`) `K*` airports from OurAirports."""
    if not runways_csv.is_file():
        raise FileNotFoundError(f"OurAirports runways.csv not found: {runways_csv}")

    csv_sha256 = hashlib.sha256(runways_csv.read_bytes()).hexdigest()
    all_icaos: set[str] = set()
    conus_eligible: set[str] = set()

    with runways_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ident = str(row["airport_ident"]).strip().upper()
            if not ICAO_RE.match(ident):
                continue
            all_icaos.add(ident)

            if not ident.startswith("K"):
                continue
            if str(row["closed"]).strip() == "1":
                continue
            if str(row["lighted"]).strip() != "1":
                continue
            surf = str(row["surface"]).strip().upper()
            if not any(tok in surf for tok in PAVED_SURFACE_TOKENS):
                continue
            len_raw = str(row["length_ft"]).strip()
            if not len_raw:
                continue
            try:
                length_m = float(len_raw) * FT_TO_M
            except ValueError:
                continue
            if math.isfinite(length_m) and length_m >= MIN_RUNWAY_LENGTH_M:
                conus_eligible.add(ident)

    return all_icaos, sorted(conus_eligible), csv_sha256


def build_and_write_lard_manifests(
    *,
    lard_url: str = LARD_RUNWAYS_DB_URL,
    runways_csv: Path = DEFAULT_OURAIRPORTS_RUNWAYS_CSV,
    lard_manifest_out: Path = DEFAULT_LARD_MANIFEST_OUT,
    splits_out: Path = DEFAULT_SPLITS_OUT,
    heldout_stride: int = 20,
) -> dict[str, Any]:
    """Fetch LARD ICAO metadata, cross-check OurAirports, verify fail-closed guard, and write JSONs."""
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    raw_bytes, source_sha256, lard_db = fetch_lard_runways_metadata(lard_url)
    all_oa_icaos, conus_eligible, oa_sha256 = load_ourairports_catalog(runways_csv)

    raw_lard_icaos = sorted(
        {str(k).strip().upper() for k in lard_db.keys() if ICAO_RE.match(str(k).strip().upper())}
    )
    verified_lard_icaos = [code for code in raw_lard_icaos if code in all_oa_icaos]
    unmatched_lard_icaos = [code for code in raw_lard_icaos if code not in all_oa_icaos]
    if len(verified_lard_icaos) < MIN_LARD_REAL_AIRPORTS:
        raise ValueError(
            f"Only {len(verified_lard_icaos)} OurAirports-verified LARD ICAOs found "
            f"(minimum required >= {MIN_LARD_REAL_AIRPORTS})"
        )

    # Conservative exclusion: exclude ALL raw LARD ICAOs (both matched & unmatched) + pinned Bay Area held-out
    all_lard_set = set(raw_lard_icaos)
    pinned_set = set(PINNED_HELDOUT_CONUS)
    disjoint_conus = [code for code in conus_eligible if code not in all_lard_set and code not in pinned_set]

    heldout_extra = [
        code for idx, code in enumerate(disjoint_conus) if (idx % max(2, int(heldout_stride))) == 0
    ]
    heldout_set = pinned_set | set(heldout_extra)
    train_icaos = sorted(code for code in disjoint_conus if code not in heldout_set)
    heldout_icaos = sorted(heldout_set)

    manifest_doc: dict[str, Any] = {
        "schema": "xvs.v2_m2.lard_real_airports_manifest/1",
        "verified": True,
        "source": (
            "deel-ai/LARD public runways_database.json "
            "(metadata-only ICAO exclusion list; zero LARD imagery)"
        ),
        "source_url": lard_url,
        "retrieved_utc": retrieved_utc,
        "source_bytes": len(raw_bytes),
        "source_sha256": source_sha256,
        "cross_reference_catalog": "data/reference/ourairports/runways.csv",
        "cross_reference_sha256": oa_sha256,
        "raw_lard_icao_count": len(raw_lard_icaos),
        "verified_icao_count": len(verified_lard_icaos),
        "unmatched_in_ourairports": unmatched_lard_icaos,
        "airports": verified_lard_icaos,
    }

    # Fail-closed verification BEFORE writing to disk
    ver = validate_lard_real_airports_manifest(
        manifest_doc,
        train_icaos=train_icaos,
        heldout_icaos=heldout_icaos,
    )

    splits_doc: dict[str, Any] = {
        "schema": "xvs.v2_m2.xvs_land_airport_splits/1",
        "generated_utc": retrieved_utc,
        "lard_manifest_path": "data/reference/lard_real_airports_manifest.json",
        "lard_source_sha256": source_sha256,
        "ourairports_runways_csv": "data/reference/ourairports/runways.csv",
        "ourairports_runways_sha256": oa_sha256,
        "filter_criteria": {
            "conus_prefix": "K",
            "paved_surface_tokens": list(PAVED_SURFACE_TOKENS),
            "lighted_required": True,
            "open_required": True,
            "min_runway_length_m": MIN_RUNWAY_LENGTH_M,
            "excluded_lard_icaos_count": len(all_lard_set),
            "pinned_heldout_icaos": list(PINNED_HELDOUT_CONUS),
        },
        "verification": {
            "manifest_verified": ver.manifest_verified,
            "lard_real_count": ver.lard_real_count,
            "train_count": ver.train_count,
            "heldout_count": ver.heldout_count,
            "ksfo_heldout": ver.ksfo_heldout,
            "train_heldout_overlap_count": len(ver.train_heldout_overlap),
            "train_lard_real_overlap_count": len(ver.train_lard_real_overlap),
        },
        "train_icaos": train_icaos,
        "heldout_icaos": heldout_icaos,
    }

    lard_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)
    lard_manifest_out.write_text(
        json.dumps(manifest_doc, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    splits_out.write_text(
        json.dumps(splits_doc, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Re-verify directly from disk files to confirm round-trip integrity
    disk_ver = validate_lard_real_airports_manifest(
        lard_manifest_out,
        train_icaos=splits_doc["train_icaos"],
        heldout_icaos=splits_doc["heldout_icaos"],
    )
    return {
        "lard_manifest_path": str(lard_manifest_out),
        "splits_path": str(splits_out),
        "source_sha256": source_sha256,
        "ourairports_sha256": oa_sha256,
        "lard_real_count": disk_ver.lard_real_count,
        "train_count": disk_ver.train_count,
        "heldout_count": disk_ver.heldout_count,
        "ksfo_heldout": disk_ver.ksfo_heldout,
        "pass": bool(
            disk_ver.manifest_verified
            and disk_ver.ksfo_heldout
            and len(disk_ver.train_heldout_overlap) == 0
            and len(disk_ver.train_lard_real_overlap) == 0
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync public deel-ai/LARD ICAO metadata exclusion list & CONUS splits"
    )
    p.add_argument("--lard-url", type=str, default=LARD_RUNWAYS_DB_URL)
    p.add_argument("--runways-csv", type=Path, default=DEFAULT_OURAIRPORTS_RUNWAYS_CSV)
    p.add_argument("--lard-manifest-out", type=Path, default=DEFAULT_LARD_MANIFEST_OUT)
    p.add_argument("--splits-out", type=Path, default=DEFAULT_SPLITS_OUT)
    p.add_argument("--heldout-stride", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(argv)
    summary = build_and_write_lard_manifests(
        lard_url=ns.lard_url,
        runways_csv=ns.runways_csv,
        lard_manifest_out=ns.lard_manifest_out,
        splits_out=ns.splits_out,
        heldout_stride=int(ns.heldout_stride),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
