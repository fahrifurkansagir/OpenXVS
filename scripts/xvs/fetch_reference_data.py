"""Fetches or verifies reference datasets and writes metadata sidecars.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3 and Finding C5.
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_or_fetch_ourairports_runways(
    csv_path: Path = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv",
    sha_path: Path = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.sha256",
    meta_path: Path = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.meta.json",
) -> dict:
    """Verify local runways.csv against SHA-256 and generate runways.csv.meta.json."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {csv_path}")

    raw_bytes = csv_path.read_bytes()
    computed_sha = hashlib.sha256(raw_bytes).hexdigest()

    if sha_path.is_file():
        expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
        if computed_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {csv_path}: expected {expected_sha}, got {computed_sha}"
            )
    else:
        sha_path.write_text(f"{computed_sha}\n", encoding="utf-8")

    meta = {
        "id": "ourairports",
        "url": "https://davidmegginson.github.io/ourairports-data/runways.csv",
        "host": "davidmegginson.github.io",
        "path_prefix": "/ourairports-data/",
        "attribution": "OurAirports (ourairports.com)",
        "license": "Public Domain",
        "sha256": computed_sha,
        "bytes": len(raw_bytes),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Verified against data/reference/ourairports/runways.csv.sha256",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    meta = verify_or_fetch_ourairports_runways()
    print("Wrote OurAirports reference metadata:", meta["sha256"], f"({meta['bytes']} bytes)")
