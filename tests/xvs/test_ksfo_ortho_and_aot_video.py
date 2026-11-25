"""Tests for KSFO public-domain USGS orthoimagery pack & XVS-D2 ONNX demo renderer."""

from __future__ import annotations

from compression import zstd
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_XVS = REPO_ROOT / "scripts" / "xvs"
for p in (str(SRC_ROOT), str(SCRIPTS_XVS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_ksfo_naip_ortho_pack as ortho_mod
import render_aot_d2_demo_video as video_mod
from xvs.sim.allowlist import URLClassifier

ASSETS_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "assets"
TERRAIN_MANIFEST_PATH = ASSETS_DIR / "ksfo_3dep_geoid18_v1.manifest.json"
ORTHO_MANIFEST_PATH = ASSETS_DIR / "ksfo_ortho_v1.manifest.json"
WEB_ONNX_PATH = REPO_ROOT / "apps" / "livedeck" / "public" / "models" / "xvs_d2_web_fp16.onnx"


def test_equirect_uv_orientation_matches_pass2_shader() -> None:
    """Verify row=0 -> lat_max (North), row=H-1 -> lat_min (South), col=0 -> lon_min (West)."""
    lat_min, lat_max = 37.5738, 37.6638
    lon_min, lon_max = -122.4321, -122.3188
    w, h = 1024, 1024

    nw_lat, nw_lon = ortho_mod.equirect_pixel_to_latlon(
        0, 0, width=w, height=h, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    se_lat, se_lon = ortho_mod.equirect_pixel_to_latlon(
        h - 1, w - 1, width=w, height=h, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    assert nw_lat == pytest.approx(lat_max, abs=1e-9)
    assert nw_lon == pytest.approx(lon_min, abs=1e-9)
    assert se_lat == pytest.approx(lat_min, abs=1e-9)
    assert se_lon == pytest.approx(lon_max, abs=1e-9)

    u_nw, v_nw = ortho_mod.latlon_to_equirect_uv(
        nw_lat, nw_lon, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    u_se, v_se = ortho_mod.latlon_to_equirect_uv(
        se_lat, se_lon, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    assert (u_nw, v_nw) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert (u_se, v_se) == pytest.approx((1.0, 1.0), abs=1e-9)


def test_mercator_to_equirect_reprojection_preserves_cardinal_directions() -> None:
    """Synthetic quadrant mosaic confirms North=top (y=0), South=bottom, West=left, East=right."""
    lat_min, lat_max = 37.573814430866676, 37.66379676913333
    lon_min, lon_max = -122.43207461537688, -122.31875878462313
    zoom = 13

    x_min, x_max, y_min, y_max = ortho_mod.tile_bounds_for_ring(
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, zoom=zoom
    )
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    mosaic_h = rows * ortho_mod.TILE_SIZE_PX
    mosaic_w = cols * ortho_mod.TILE_SIZE_PX

    # Encode North-South gradient in R channel (high=North/top, low=South/bottom)
    # and West-East gradient in G channel (low=West/left, high=East/right).
    yy = np.linspace(240.0, 20.0, mosaic_h, dtype=np.float32)[:, np.newaxis]
    xx = np.linspace(20.0, 240.0, mosaic_w, dtype=np.float32)[np.newaxis, :]
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    mosaic[:, :, 0] = np.broadcast_to(yy, (mosaic_h, mosaic_w)).astype(np.uint8)
    mosaic[:, :, 1] = np.broadcast_to(xx, (mosaic_h, mosaic_w)).astype(np.uint8)
    mosaic[:, :, 2] = 128

    out = ortho_mod.reproject_mercator_mosaic_to_equirect(
        mosaic,
        x_min=x_min,
        y_min=y_min,
        zoom=zoom,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        out_width=128,
        out_height=128,
    )
    assert out.shape == (128, 128, 3)
    # Top row (North) must have strictly higher R than bottom row (South)
    assert float(out[0, :, 0].mean()) > float(out[-1, :, 0].mean()) + 20.0
    # Right col (East) must have strictly higher G than left col (West)
    assert float(out[:, -1, 1].mean()) > float(out[:, 0, 1].mean()) + 20.0


def test_allowlist_enforcement_allows_usgs_and_blocks_forbidden_hosts() -> None:
    """URLClassifier must allow USGSImageryOnly tiles and fail-closed on non-allowlisted URLs."""
    classifier = URLClassifier()
    usgs_url = ortho_mod.build_usgs_tile_url(13, 3170, 1310)
    assert ortho_mod.enforce_allowlist(usgs_url, classifier) == "ALLOWED"

    for bad_url in (
        "https://mt0.google.com/vt/lyrs=s&x=0&y=0&z=0",
        "https://example.invalid/tiles/13/3170/1310.jpg",
    ):
        with pytest.raises(PermissionError):
            ortho_mod.enforce_allowlist(bad_url, classifier)


def test_generated_ksfo_ortho_manifest_and_textures_are_valid() -> None:
    """Verify live USGS KSFO ortho JPEGs match manifest SHA-256s, budgets, bounds, and RGB variance."""
    assert ORTHO_MANIFEST_PATH.is_file(), f"Missing ortho manifest: {ORTHO_MANIFEST_PATH}"
    assert TERRAIN_MANIFEST_PATH.is_file(), f"Missing terrain manifest: {TERRAIN_MANIFEST_PATH}"

    ortho = json.loads(ORTHO_MANIFEST_PATH.read_text(encoding="utf-8"))
    terrain = json.loads(TERRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert ortho["schema"] == "xvs.terrain.ksfo_ortho_manifest/1"
    assert ortho["license"] == "U.S. Public Domain"
    assert "USGS The National Map: Orthoimagery" in str(ortho["attribution"])
    assert ortho["aligned_terrain_pack_sha256"] == terrain["pack_sha256"]

    terrain_rings = {str(r["name"]): r for r in terrain["rings"]}
    ortho_rings = {str(r["name"]): r for r in ortho["rings"]}
    assert set(ortho_rings.keys()) == {"inner_ksfo", "outer_sf_bay"}

    budgets = {"inner_ksfo": 350_000, "outer_sf_bay": 250_000}
    for name, max_budget in budgets.items():
        entry = ortho_rings[name]
        t_ring = terrain_rings[name]
        img_path = ASSETS_DIR / str(entry["filename"])
        assert img_path.is_file(), f"Missing ortho JPEG: {img_path}"

        raw_bytes = img_path.read_bytes()
        assert len(raw_bytes) == int(entry["bytes"])
        assert len(raw_bytes) <= max_budget
        assert hashlib.sha256(raw_bytes).hexdigest() == str(entry["sha256"])

        assert float(entry["lat_min"]) == pytest.approx(float(t_ring["lat_min"]), abs=1e-9)
        assert float(entry["lat_max"]) == pytest.approx(float(t_ring["lat_max"]), abs=1e-9)
        assert float(entry["lon_min"]) == pytest.approx(float(t_ring["lon_min"]), abs=1e-9)
        assert float(entry["lon_max"]) == pytest.approx(float(t_ring["lon_max"]), abs=1e-9)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        assert bgr is not None
        assert bgr.shape == (1024, 1024, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert float(rgb.std(axis=(0, 1)).max()) > 15.0

        # Confirm zero large pure-black no-data cutouts remain
        black_mask = ortho_mod.detect_nodata_black_mask(rgb, min_area_px=256)
        assert int((black_mask > 0).sum()) == 0

        # Confirm KSFO 28L threshold maps cleanly inside [0, 1]^2
        uv_28l = entry["ksfo_28l_thr_uv"]
        assert 0.1 < float(uv_28l["u"]) < 0.9
        assert 0.1 < float(uv_28l["v"]) < 0.9


def test_safe_results_archive_extraction_blocks_path_traversal(tmp_path: Path) -> None:
    """Verify `.tar.zst` extractor rejects traversal paths and selects safe `xvs_d2_web_fp16.onnx`."""
    assert WEB_ONNX_PATH.is_file(), f"Missing ONNX model: {WEB_ONNX_PATH}"
    onnx_bytes = WEB_ONNX_PATH.read_bytes()

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:") as tf:
        evil = tarfile.TarInfo(name="../escape_evil.onnx")
        evil.size = 4
        tf.addfile(evil, io.BytesIO(b"EVIL"))

        good = tarfile.TarInfo(name="exports/xvs_d2_web_fp16.onnx")
        good.size = len(onnx_bytes)
        tf.addfile(good, io.BytesIO(onnx_bytes))

    archive_path = tmp_path / "mock_g4_results.tar.zst"
    archive_path.write_bytes(zstd.compress(tar_buf.getvalue()))

    extract_dir = tmp_path / "extracted"
    extracted_path = video_mod.extract_onnx_from_results_archive(archive_path, extract_dir)
    assert extracted_path.name == "xvs_d2_web_fp16.onnx"
    assert extracted_path.parent == extract_dir
    assert extracted_path.read_bytes() == onnx_bytes
    assert not (tmp_path / "escape_evil.onnx").exists()


def test_decode_xvs_d2_outputs_recovers_injected_peak() -> None:
    """Confirm stride-2 local 3x3 soft-argmax decoder recovers injected target center & size."""
    hm = np.full((1, 1, 128, 128), -10.0, dtype=np.float32)
    off = np.zeros((1, 2, 128, 128), dtype=np.float32)
    lsc = np.zeros((1, 2, 128, 128), dtype=np.float32)
    lvr = np.zeros((1, 3, 128, 128), dtype=np.float32)
    pmk = np.zeros((1, 1, 128, 128), dtype=np.float32)

    cy, cx = 40, 62
    hm[0, 0, cy, cx] = 5.0
    pmk[0, 0, cy, cx] = 1.0
    lsc[0, 0, cy, cx] = np.log(12.0)  # s_mom = 12 px
    lsc[0, 1, cy, cx] = 0.0           # square aspect ratio

    boxes, (_tu, _tv, top_p) = video_mod.decode_xvs_d2_outputs_np(
        hm, off, lsc, lvr, pmk, score_thr=0.5
    )
    assert len(boxes) == 1
    assert top_p > 0.99
    box = boxes[0]
    assert 0.5 * (box.x1 + box.x2) == pytest.approx((cx + 0.5) * 2.0, abs=0.25)
    assert 0.5 * (box.y1 + box.y2) == pytest.approx((cy + 0.5) * 2.0, abs=0.25)
    assert (box.x2 - box.x1) == pytest.approx(12.0, abs=0.2)


def test_smoke_synthetic_onnx_video_and_contact_sheet_end_to_end(tmp_path: Path) -> None:
    """Run genuine ONNXRuntime ConvGRU loop end-to-end and verify MP4 + 2x3 PNG contact sheet."""
    assert WEB_ONNX_PATH.is_file(), f"Missing ONNX model: {WEB_ONNX_PATH}"
    out_mp4 = tmp_path / "smoke.mp4"
    out_png = tmp_path / "smoke_contact.png"

    summary = video_mod.run_demo_video(
        onnx_path=WEB_ONNX_PATH,
        smoke_synthetic=True,
        out_mp4=out_mp4,
        out_contact_sheet=out_png,
        num_synthetic_frames=12,
        fps=12.0,
    )
    assert summary["frames_rendered"] == 12
    assert summary["input_hw"] == [256, 256]
    assert summary["hidden_shape"] == [1, 24, 64, 64]
    assert out_mp4.is_file() and out_mp4.stat().st_size > 10_000
    assert out_png.is_file() and out_png.stat().st_size > 10_000

    # Verify MP4 decodes cleanly with OpenCV VideoCapture
    cap = cv2.VideoCapture(str(out_mp4))
    try:
        assert cap.isOpened()
        decoded_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (720, 1280, 3)
            decoded_count += 1
        assert decoded_count == 12
    finally:
        cap.release()

    # Verify 2x3 keyframe contact sheet dimensions and non-blank content
    sheet = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
    assert sheet is not None
    assert sheet.shape == (720, 1920, 3)
    assert float(sheet.std()) > 20.0
