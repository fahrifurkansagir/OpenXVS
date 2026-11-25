"""OurAirports data loader and runway landing threshold calculations.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md §2.2.
"""

from __future__ import annotations
import csv
import functools
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from xvs.geo import wgs84


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNWAYS_CSV = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"
DEFAULT_RUNWAYS_SHA = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.sha256"


@dataclass
class RunwayThreshold:
    airport_ident: str
    runway_ident: str
    lat_deg: float
    lon_deg: float
    h_m: float
    heading_deg: float
    displaced_threshold_m: float
    source_row: dict[str, str]
    file_sha256: str


@functools.lru_cache(maxsize=8)
def load_ksfo_28l_threshold(
    csv_path: str | Path = DEFAULT_RUNWAYS_CSV,
    sha_path: str | Path = DEFAULT_RUNWAYS_SHA,
) -> RunwayThreshold:
    """Load and compute landing threshold (THR) for KSFO runway 28L.

    THR = runway end moved he_displaced_threshold_ft * 0.3048 m along he_heading_degT.
    Ellipsoid mode: h_THR = 0.0.

    File integrity is verified using LF-normalized SHA-256 (converting CRLF to LF)
    to guarantee identical hashes regardless of git checkout core.autocrlf settings.
    """
    csv_file = Path(csv_path)
    with open(csv_file, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n")
        file_sha256 = hashlib.sha256(content).hexdigest()

    target_row: dict[str, str] | None = None
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("airport_ident") == "KSFO" and row.get("he_ident") == "28L":
                target_row = row
                break

    if target_row is None:
        raise ValueError("Could not find KSFO runway 28L in runways.csv")

    he_lat = float(target_row["he_latitude_deg"])
    he_lon = float(target_row["he_longitude_deg"])
    heading_deg = float(target_row["he_heading_degT"])
    displaced_ft_str = target_row.get("he_displaced_threshold_ft", "")
    displaced_ft = float(displaced_ft_str) if displaced_ft_str else 0.0
    displaced_m = displaced_ft * 0.3048

    # Offset in tangent (NED) plane at runway high end via exact tangent-plane construction
    heading_rad = math.radians(heading_deg)
    d_north = displaced_m * math.cos(heading_rad)
    d_east = displaced_m * math.sin(heading_rad)

    lat_thr, lon_thr, _ = wgs84.ned_offset_to_geodetic(he_lat, he_lon, d_north, d_east)
    h_thr = 0.0  # Ellipsoid mode (CONVENTIONS §2)

    return RunwayThreshold(
        airport_ident="KSFO",
        runway_ident="28L",
        lat_deg=lat_thr,
        lon_deg=lon_thr,
        h_m=h_thr,
        heading_deg=heading_deg,
        displaced_threshold_m=displaced_m,
        source_row=target_row,
        file_sha256=file_sha256,
    )
