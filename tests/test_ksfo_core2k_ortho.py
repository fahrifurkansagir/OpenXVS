"""Unit tests for the 2048x2048 KSFO 2x2 km core orthoimagery builder (~0.98 m/texel)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_XVS = REPO_ROOT / "scripts" / "xvs"
if str(SCRIPTS_XVS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_XVS))

from build_ksfo_naip_ortho_pack import (  # noqa: E402
    CORE2K_EXTENT_KM,
    CORE2K_MAX_BYTES,
    CORE2K_SIZE_PX,
    KSFO_28L_LAT,
    KSFO_28L_LON,
    build_ksfo_core2k_ortho,
    latlon_to_equirect_uv,
)


def test_ksfo_core2k_offline_fallback(tmp_path: Path) -> None:
    """Offline procedural fallback generates a valid 2048x2048 < 1.2 MB KSFO core ortho."""
    out_path = tmp_path / "ksfo_ortho_core2k_v2.jpg"
    meta = build_ksfo_core2k_ortho(out_path, offline_procedural=True)

    assert out_path.is_file()
    raw_bytes = out_path.read_bytes()
    assert len(raw_bytes) < 1_200_000
    assert len(raw_bytes) <= CORE2K_MAX_BYTES
    assert meta["provenance"] == "procedural_fallback"
    assert meta["width"] == CORE2K_SIZE_PX == 2048
    assert meta["height"] == CORE2K_SIZE_PX == 2048
    assert meta["bytes"] == len(raw_bytes)
    assert meta["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert meta["extent_km"] == CORE2K_EXTENT_KM == 2.0
    assert float(meta["meters_per_texel"]) < 1.0

    with Image.open(out_path) as img:
        assert img.size == (2048, 2048)
        assert img.mode == "RGB"
        arr = np.asarray(img, dtype=np.float32)

    # Ensure procedural fallback renders structured runway/water contrast (not blank)
    assert float(arr.std()) > 12.0

    # Verify KSFO 28L threshold lies comfortably inside the 8% blend interior of the 2x2 km box
    u_28l, v_28l = latlon_to_equirect_uv(
        KSFO_28L_LAT,
        KSFO_28L_LON,
        lat_min=float(meta["lat_min"]),
        lat_max=float(meta["lat_max"]),
        lon_min=float(meta["lon_min"]),
        lon_max=float(meta["lon_max"]),
    )
    assert 0.12 <= u_28l <= 0.88
    assert 0.12 <= v_28l <= 0.88


def test_ksfo_core2k_network_error_falls_back_cleanly(tmp_path: Path, monkeypatch) -> None:
    """If live USGS tile fetch raises a network error, builder degrades to procedural_fallback."""
    import build_ksfo_naip_ortho_pack as mod

    def _fail_fetch(**_kwargs):
        raise OSError("Simulated offline network timeout")

    monkeypatch.setattr(mod, "fetch_reprojected_ring_rgb", _fail_fetch)

    out_path = tmp_path / "core_fallback.jpg"
    meta = build_ksfo_core2k_ortho(out_path, offline_procedural=False)
    assert out_path.is_file()
    assert meta["provenance"] == "procedural_fallback"
    assert meta["width"] == 2048
    assert meta["height"] == 2048
    assert int(meta["bytes"]) < 1_200_000
