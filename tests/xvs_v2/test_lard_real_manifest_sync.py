"""Verify real `deel-ai/LARD` ICAO exclusion manifest & disjoint CONUS airport splits for XVS-Land."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from xvs.v2.xvs_land.dataset_guard import (
    MIN_LARD_REAL_AIRPORTS,
    validate_lard_real_airports_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LARD_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "lard_real_airports_manifest.json"
SPLITS_PATH = REPO_ROOT / "data" / "reference" / "xvs_land_airport_splits.json"
RUNWAYS_CSV_PATH = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"


def test_real_lard_manifest_and_conus_splits_verified_zero_leakage() -> None:
    assert LARD_MANIFEST_PATH.is_file(), f"Missing {LARD_MANIFEST_PATH}"
    assert SPLITS_PATH.is_file(), f"Missing {SPLITS_PATH}"

    manifest_doc = json.loads(LARD_MANIFEST_PATH.read_text(encoding="utf-8"))
    splits_doc = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    # Provenance & metadata-only checks (zero LARD imagery, real GitHub URL + SHA-256)
    assert manifest_doc["verified"] is True
    assert "deel-ai/LARD" in manifest_doc["source"]
    assert manifest_doc["source_url"].startswith("https://raw.githubusercontent.com/deel-ai/LARD/")
    assert len(str(manifest_doc["source_sha256"])) == 64
    assert int(manifest_doc["verified_icao_count"]) == len(manifest_doc["airports"])

    train_icaos = splits_doc["train_icaos"]
    heldout_icaos = splits_doc["heldout_icaos"]

    # Fail-closed guard verification
    ver = validate_lard_real_airports_manifest(
        LARD_MANIFEST_PATH,
        train_icaos=train_icaos,
        heldout_icaos=heldout_icaos,
    )
    assert ver.manifest_verified is True
    assert ver.lard_real_count >= MIN_LARD_REAL_AIRPORTS
    assert ver.ksfo_heldout is True
    assert ver.train_count >= 100
    assert ver.heldout_count >= 3
    assert len(ver.train_heldout_overlap) == 0
    assert len(ver.train_lard_real_overlap) == 0
    assert "KSFO" in heldout_icaos
    assert "KSFO" not in train_icaos

    # Cross-check that every verified LARD ICAO and every training ICAO exists in OurAirports runways.csv
    with RUNWAYS_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        oa_icaos = {str(row["airport_ident"]).strip().upper() for row in csv.DictReader(f)}

    for code in manifest_doc["airports"]:
        assert code in oa_icaos, f"LARD manifest ICAO {code!r} missing from OurAirports runways.csv"
    for code in train_icaos:
        assert code in oa_icaos, f"Training ICAO {code!r} missing from OurAirports runways.csv"
        assert code.startswith("K"), f"Training ICAO {code!r} is not CONUS K*"
