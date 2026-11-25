"""Fast unit tests for OurAirports loader and KSFO 28L landing threshold calculations.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.2, §3 (G1) and Findings C5, C7.
"""

import json
from pathlib import Path
import pytest

from xvs.sim import airports


def test_ksfo_28l_threshold_parsed_and_displaced():
    """The 28L threshold is parsed and the displacement applied; the parsed row and the SHA-256 are logged."""
    thr = airports.load_ksfo_28l_threshold()
    assert thr.airport_ident == "KSFO"
    assert thr.runway_ident == "28L"
    assert thr.heading_deg == 298.0
    assert thr.h_m == 0.0  # Ellipsoid mode
    assert abs(thr.displaced_threshold_m - 91.44) < 1e-6

    # Verify exact Director reference coordinates:
    # THR lat 37.61210677733923, lon -122.35928141744020
    assert abs(thr.lat_deg - 37.61210677733923) <= 1e-9
    assert abs(thr.lon_deg - (-122.35928141744020)) <= 1e-9

    # Source row contents
    assert thr.source_row["airport_ident"] == "KSFO"
    assert thr.source_row["he_ident"] == "28L"
    assert thr.file_sha256 == "97f30f038be33a1c93c190b9a90c46fabbe16e36a86f5decdac62259613411ae"


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reference_metadata_sidecar():
    """Verify data/reference/ourairports/runways.csv.meta.json exists and is valid."""
    meta_path = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.meta.json"
    assert meta_path.is_file(), "runways.csv.meta.json should exist"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["id"] == "ourairports"
    assert data["sha256"] == "97f30f038be33a1c93c190b9a90c46fabbe16e36a86f5decdac62259613411ae"
    assert data["bytes"] == 3965119


def test_crlf_normalization_produces_identical_hash(tmp_path: Path):
    """Verify that a CRLF copy in tmp_path produces the exact same LF hash as the reference file (F-B5-7)."""
    src_csv = airports.DEFAULT_RUNWAYS_CSV
    raw_bytes = src_csv.read_bytes()
    # Ensure CRLF line endings
    crlf_bytes = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    crlf_csv = tmp_path / "runways_crlf.csv"
    crlf_csv.write_bytes(crlf_bytes)

    thr_crlf = airports.load_ksfo_28l_threshold(csv_path=crlf_csv)
    thr_orig = airports.load_ksfo_28l_threshold(csv_path=src_csv)

    assert thr_crlf.file_sha256 == thr_orig.file_sha256
    assert thr_crlf.file_sha256 == "97f30f038be33a1c93c190b9a90c46fabbe16e36a86f5decdac62259613411ae"
    assert thr_crlf.lat_deg == thr_orig.lat_deg
    assert thr_crlf.lon_deg == thr_orig.lon_deg
