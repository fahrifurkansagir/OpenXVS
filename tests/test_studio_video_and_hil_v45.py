"""Wave 20 (OpenXVS v4.5) real-world video analyzer, fast camera frame, and security tests."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
import pytest

from xvs.v2.studio_bridge import (
    DEFAULT_VRAM_CAP_MB,
    EXPECTED_D2_ONNX_SHA256,
    StudioInferenceEngine,
    create_studio_http_server,
    load_studio_models,
)
from xvs.v2.studio_bridge.video_analyzer import _render_hud_overlay


def _write_synthetic_sky_mp4(path: Path, n_frames: int = 12, w: int = 640, h: int = 360) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), (205, 182, 154), dtype=np.uint8)
        cx = 220 + i * 14
        cy = 120 + (i % 3) * 3
        cv2.ellipse(frame, (cx, cy), (14, 5), 0.0, 0.0, 360.0, (28, 32, 38), -1, cv2.LINE_AA)
        cv2.line(frame, (cx - 16, cy), (cx + 16, cy), (22, 24, 28), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    assert path.is_file() and path.stat().st_size > 1000
    return path


def _assert_valid_data_uri_jpeg(data_uri: str) -> None:
    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.ndim == 3 and img.shape[2] == 3


def _poll_video_job_until_done(base_url: str, job_id: str, timeout_s: float = 45.0) -> dict:
    deadline = time.perf_counter() + timeout_s
    last_doc: dict = {}
    while time.perf_counter() < deadline:
        with urllib.request.urlopen(f"{base_url}/api/video_job/{job_id}", timeout=15.0) as resp:
            assert resp.status == 200
            last_doc = json.loads(resp.read().decode("utf-8"))
        if last_doc["status"] in ("done", "error"):
            return last_doc
        time.sleep(0.15)
    raise AssertionError(f"Video job {job_id} did not finish within {timeout_s}s; last={last_doc}")


@pytest.fixture(scope="module")
def v45_live_server():
    bundle = load_studio_models(device_pref="auto", vram_cap_mb=DEFAULT_VRAM_CAP_MB, smoke_fast=False)
    engine = StudioInferenceEngine(bundle)
    server = create_studio_http_server(engine, host="127.0.0.1", port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", engine
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)


def test_analyze_frame_fast_all_4_channels_contract(v45_live_server) -> None:
    base_url, _ = v45_live_server
    frame = np.full((360, 640, 3), (185, 165, 140), dtype=np.uint8)
    cv2.rectangle(frame, (180, 210), (460, 340), (65, 70, 76), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    jpeg_bytes = buf.tobytes()

    for ch in ("d2", "land", "thermal", "nav"):
        req = urllib.request.Request(
            f"{base_url}/api/analyze_frame_fast?channel={ch}",
            data=jpeg_bytes,
            method="POST",
            headers={"Content-Type": "image/jpeg", "Origin": "http://127.0.0.1:4174"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            assert resp.status == 200
            assert resp.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:4174"
            doc = json.loads(resp.read().decode("utf-8"))
        assert doc["channel"] == ch
        assert doc["real_weights"] is True
        assert math.isfinite(doc["latency_ms"]) and doc["latency_ms"] >= 0.0
        assert math.isfinite(doc["confidence"])
        assert isinstance(doc["checkpoint_sha256"], str) and len(doc["checkpoint_sha256"]) == 64
        assert isinstance(doc["metrics"], dict) and len(doc["metrics"]) > 0
        _assert_valid_data_uri_jpeg(doc["annotated_jpeg_b64"])


def test_hd_multi_crop_fallback_does_not_corrupt_demo_state(v45_live_server) -> None:
    _, engine = v45_live_server
    hd_frame = np.full((720, 1280, 3), (195, 175, 155), dtype=np.uint8)
    cv2.circle(hd_frame, (640, 210), 8, (25, 28, 34), -1, cv2.LINE_AA)

    before_age = engine._demo_d2_state.active_range_age
    iso_state = engine.new_d2_tracker_state()
    res = engine.infer_eo_sky(
        hd_frame,
        return_jpeg=True,
        tracker_state=iso_state,
        multi_crop_fallback=True,
    )
    assert engine._demo_d2_state.active_range_age == before_age
    assert res["checkpoint_sha256"] == EXPECTED_D2_ONNX_SHA256
    assert math.isfinite(res["active_range"]["r_safe_lower_m"])
    _assert_valid_data_uri_jpeg(res["annotated_jpeg_b64"])


def test_upload_mp4_video_job_end_to_end_and_download(v45_live_server, tmp_path: Path) -> None:
    base_url, _ = v45_live_server
    mp4_in = _write_synthetic_sky_mp4(tmp_path / "sample_sky.mp4", n_frames=10)
    raw_bytes = mp4_in.read_bytes()

    req = urllib.request.Request(
        f"{base_url}/api/analyze_video?channel=d2&sample_fps=10&max_frames=10",
        data=raw_bytes,
        method="POST",
        headers={"Content-Type": "video/mp4"},
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        assert resp.status == 202
        start_doc = json.loads(resp.read().decode("utf-8"))
    job_id = start_doc["job_id"]
    assert len(job_id) == 12
    assert start_doc["channel"] == "d2"

    final_doc = _poll_video_job_until_done(base_url, job_id)
    assert final_doc["status"] == "done", f"Unexpected error: {final_doc}"
    assert final_doc["progress_pct"] == 100.0
    _assert_valid_data_uri_jpeg(final_doc["latest_preview_b64"])

    summary = final_doc["summary"]
    assert summary["frames_processed"] == 10
    assert 0.0 <= summary["detection_rate"] <= 1.0
    assert summary["mean_latency_ms"] > 0.0
    assert summary["checkpoint_sha256"] == EXPECTED_D2_ONNX_SHA256
    assert summary["output_mp4_bytes"] > 1000
    assert summary["download_url"] == f"/api/video_job/{job_id}/download"

    # Download annotated MP4 and verify readable via cv2.VideoCapture
    dl_path = tmp_path / "downloaded_annotated.mp4"
    with urllib.request.urlopen(f"{base_url}{summary['download_url']}", timeout=20.0) as resp_dl:
        assert resp_dl.status == 200
        assert resp_dl.headers["Content-Type"] == "video/mp4"
        mp4_data = resp_dl.read()
    assert len(mp4_data) == summary["output_mp4_bytes"]
    dl_path.write_bytes(mp4_data)

    cap = cv2.VideoCapture(str(dl_path))
    assert cap.isOpened()
    ok, first_f = cap.read()
    cap.release()
    assert ok and first_f is not None and first_f.ndim == 3

    # Verify HTTP Range 206 partial content works for Chrome <video> seeking
    req_rng = urllib.request.Request(
        f"{base_url}{summary['download_url']}",
        headers={"Range": "bytes=0-1023"},
    )
    with urllib.request.urlopen(req_rng, timeout=15.0) as resp_rng:
        assert resp_rng.status == 206
        assert len(resp_rng.read()) == 1024


def test_builtin_aot_realsky_video_job_end_to_end(v45_live_server, tmp_path: Path) -> None:
    base_url, _ = v45_live_server
    req = urllib.request.Request(
        f"{base_url}/api/analyze_builtin_video?channel=d2&clip=aot_realsky&sample_fps=10&max_frames=12",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        assert resp.status == 202
        doc = json.loads(resp.read().decode("utf-8"))
    assert doc["clip"] == "aot_realsky"
    job_id = doc["job_id"]

    final_doc = _poll_video_job_until_done(base_url, job_id)
    assert final_doc["status"] == "done", f"Unexpected error: {final_doc}"
    summary = final_doc["summary"]
    assert summary["frames_processed"] == 12
    assert summary["output_mp4_bytes"] > 2000
    assert summary["checkpoint_sha256"] == EXPECTED_D2_ONNX_SHA256

    # Verify RAW real-sky resolution (512x384 from data/studio_media/eo_sky, NOT 1280x720 prebaked dashboard!)
    dl_path = tmp_path / "aot_raw_annotated.mp4"
    with urllib.request.urlopen(f"{base_url}{summary['download_url']}", timeout=20.0) as resp_dl:
        dl_path.write_bytes(resp_dl.read())
    cap = cv2.VideoCapture(str(dl_path))
    assert cap.isOpened()
    ok, f0 = cap.read()
    cap.release()
    assert ok and f0 is not None
    assert f0.shape[1] <= 640 and f0.shape[0] <= 480

    # Also verify optional 2nd built-in raw clip: runway_approach on XVS-Land
    req_rwy = urllib.request.Request(
        f"{base_url}/api/analyze_builtin_video?channel=land&clip=runway_approach&sample_fps=10&max_frames=8",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(req_rwy, timeout=20.0) as resp_r:
        assert resp_r.status == 202
        rwy_doc = json.loads(resp_r.read().decode("utf-8"))
    rwy_final = _poll_video_job_until_done(base_url, rwy_doc["job_id"])
    assert rwy_final["status"] == "done"
    assert rwy_final["summary"]["frames_processed"] == 8
    assert rwy_final["summary"]["output_mp4_bytes"] > 2000


def test_invalid_job_id_path_traversal_fails_closed(v45_live_server) -> None:
    base_url, _ = v45_live_server
    for bad_path in (
        "/api/video_job/..%2F..%2Fetc%2Fpasswd",
        "/api/video_job/not_a_valid_id",
        "/api/video_job/../../etc/passwd/download",
        "/api/video_job/deadbeef1234",
    ):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}{bad_path}", timeout=10.0)
        assert exc_info.value.code in (400, 404)


def test_flightgear_protocol_xml_valid() -> None:
    xml_path = Path(__file__).resolve().parents[1] / "configs" / "flightgear" / "openxvs_fg_protocol.xml"
    assert xml_path.is_file()
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    assert root.tag == "PropertyList"
    out_chunks = root.findall("./generic/output/chunk")
    in_chunks = root.findall("./generic/input/chunk")
    assert len(out_chunks) == 10
    assert len(in_chunks) == 4


def test_hil_loopback_and_live_udp_connect(v45_live_server) -> None:
    base_url, _ = v45_live_server

    # 1. 1-Click Loopback Self-Test (`POST /api/hil/loopback_test`)
    req_lb = urllib.request.Request(f"{base_url}/api/hil/loopback_test", data=b"", method="POST")
    with urllib.request.urlopen(req_lb, timeout=20.0) as resp:
        assert resp.status == 200
        lb = json.loads(resp.read().decode("utf-8"))
    assert lb["pass"] is True
    assert lb["gpl_free"] is True
    assert lb["tamper_rejected"] is True
    assert lb["mavlink_52001_packets_verified"] == 100
    assert len(lb["checks"]) == 4
    assert all(c["pass"] is True for c in lb["checks"])

    # 2. Live 127.0.0.1 FlightGear UDP listener bind -> real UDP packet RX -> disconnect to idle
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    free_rx_port = int(probe.getsockname()[1])
    probe.close()

    req_conn = urllib.request.Request(
        f"{base_url}/api/hil/connect",
        data=json.dumps({"mode": "flightgear_udp", "rx_port": free_rx_port, "tx_port": 5501}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_conn, timeout=10.0) as resp:
        assert resp.status == 200
        st_conn = json.loads(resp.read().decode("utf-8"))
    assert st_conn["mode"] == "flightgear_udp"
    assert st_conn["bound_host"] == "127.0.0.1"
    assert st_conn["rx_port"] == free_rx_port

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx.sendto(b"37.6117200,-122.3583670,150.000,298.000,-3.000,2.500,66.000,15.200,-29.800,1.600\n", ("127.0.0.1", free_rx_port))
        time.sleep(0.15)
    finally:
        tx.close()

    with urllib.request.urlopen(f"{base_url}/api/hil/status", timeout=10.0) as resp:
        assert resp.status == 200
        st_after = json.loads(resp.read().decode("utf-8"))
    assert st_after["rx_packets"] >= 1
    assert st_after["last_telemetry_ned"] is not None
    assert st_after["last_telemetry_ned"]["down_m"] == pytest.approx(-150.0)

    req_idle = urllib.request.Request(
        f"{base_url}/api/hil/connect",
        data=json.dumps({"mode": "idle"}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_idle, timeout=10.0) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8"))["mode"] == "idle"


def test_export_buyer_assurance_pack_zip(v45_live_server) -> None:
    base_url, _ = v45_live_server
    with urllib.request.urlopen(f"{base_url}/api/export_buyer_pack", timeout=30.0) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "application/zip"
        raw_zip = resp.read()
    assert len(raw_zip) > 2000

    expected_members = {
        "01_executive_summary.json",
        "02_formal_certification_certgen.json",
        "03_hil_loopback_proof.json",
        "04_latest_video_analysis.json",
        "05_flightgear_openxvs_protocol.xml",
        "06_FIELD_TRIAL_PROTOCOL.md",
    }
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        names = set(zf.namelist())
        assert expected_members.issubset(names)

        exec_doc = json.loads(zf.read("01_executive_summary.json").decode("utf-8"))
        assert exec_doc["hil_loopback_pass"] is True
        assert "G5_GAP" in exec_doc["certification_boundary_disclosure"]["explicit_undeveloped_gaps"]

        cert_doc = json.loads(zf.read("02_formal_certification_certgen.json").decode("utf-8"))
        assert cert_doc["kani_proofs_total"] == 18

        hil_doc = json.loads(zf.read("03_hil_loopback_proof.json").decode("utf-8"))
        assert hil_doc["pass"] is True and hil_doc["mavlink_52001_packets_verified"] == 100

        vid_doc = json.loads(zf.read("04_latest_video_analysis.json").decode("utf-8"))
        assert "summary" in vid_doc

        fg_xml = zf.read("05_flightgear_openxvs_protocol.xml").decode("utf-8")
        assert ET.fromstring(fg_xml).tag == "PropertyList"

        md_text = zf.read("06_FIELD_TRIAL_PROTOCOL.md").decode("utf-8")
        assert "Tier-0A" in md_text and "Tier-0B" in md_text and "Phase-4" in md_text
        for forbidden in ("do-178c certified", "faa certified", "easa certified"):
            assert forbidden not in md_text.lower()


def test_video_hud_overlay_keeps_bottom_64px_clear_of_controls() -> None:
    """Footer badge (y=52) & thermal vacuity bar (y=68..84) stay in top safe zone, leaving bottom 64 px clear."""
    h, w = 360, 640
    base = np.full((h, w, 3), 128, dtype=np.uint8)

    cases: list[tuple[str, dict]] = [
        (
            "d2",
            {
                "latency_ms": 18.4,
                "metrics": {
                    "detected": False,
                    "detection": None,
                    "active_range": {
                        "mode": "ACTIVE_SWERVE",
                        "r_safe_lower_m": 420.0,
                        "sigma_r_m": 62.0,
                    },
                },
            },
        ),
        (
            "land",
            {
                "latency_ms": 14.1,
                "metrics": {
                    "keypoints_xy_256": [(96.0 + i * 4.0, 100.0 + (i % 3) * 10.0) for i in range(12)],
                    "glideslope_deg": 3.0,
                    "hpl_m": 12.0,
                    "vpl_m": 6.0,
                },
            },
        ),
        (
            "thermal",
            {
                "latency_ms": 22.7,
                "metrics": {
                    "mean_vacuity": 0.42,
                    "simplex_mode": "P4_NOMINAL",
                },
            },
        ),
        (
            "thermal",
            {
                "latency_ms": 23.1,
                "metrics": {
                    "mean_vacuity": 0.85,
                    "simplex_mode": "P1_FALLBACK",
                },
            },
        ),
        (
            "nav",
            {
                "latency_ms": 19.5,
                "metrics": {
                    "spoof_detected": False,
                    "nav_mode": "VISUAL_LOCK",
                    "cosine_sim": 0.88,
                    "hpl_vis_m": 14.0,
                },
            },
        ),
    ]

    for ch, res in cases:
        vis = _render_hud_overlay(
            base,
            channel=ch,
            result=res,
            frame_idx=3,
            total_frames=30,
            timestamp_s=0.30,
            ckpt_sha8="2c9578f3",
        )
        # Top banner around (12, 26) and footer badge around (12, 52) must both render in top safe zone
        assert not np.array_equal(vis[8:36, :], base[8:36, :]), f"{ch}: top banner missing from top zone"
        assert not np.array_equal(vis[36:64, :], base[36:64, :]), f"{ch}: footer badge missing from (12, 52) safe zone"

        if ch == "thermal":
            # Thermal vacuity bar sits at top-left y=68..84, x=12..176 (clear of top banner & footer)
            assert not np.array_equal(vis[68:85, 12:177], base[68:85, 12:177]), "thermal vacuity bar missing at y=68..84"

        # Bottom 64 px must remain 100% clear of HUD badges/bars so HTML5 <video controls> never occludes telemetry
        assert np.array_equal(vis[h - 64 :, :], base[h - 64 :, :]), (
            f"{ch}: bottom 64 px contaminated by HUD overlay"
        )

