"""Unit tests for DR-15/DR-19 `$0` Real-Sky Window Capture CLI (`scripts/xvs/realsky_window_capture.py`)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "realsky_window_capture.py"

_SPEC = importlib.util.spec_from_file_location("realsky_window_capture", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["realsky_window_capture"] = _MOD
_SPEC.loader.exec_module(_MOD)

from xvs.v2.xvs_realsky.autolabel import (  # noqa: E402
    CameraIntrinsics,
    RealSkyComplianceError,
    compute_sky_elevation_mask,
    geodetic_to_local_ned,
    interpolate_cubic_hermite_ned,
    ned_velocity_from_adsb,
    project_ned_to_pixel,
    reconcile_ellipsoidal_altitude_m,
    verify_manifest_signature,
)


def test_smoke_offline_cli_emits_croissant_1_1_rai_and_strips_all_pii_and_adsb(tmp_path: Path) -> None:
    """End-to-end `--smoke-offline` emits Croissant 1.1 + RAI 1.0, SHA-256 manifest, zero PII leak."""
    rc = _MOD.main(["--smoke-offline", "--out-dir", str(tmp_path)])
    assert rc == 0

    labels_path = tmp_path / "labels.jsonl"
    croissant_path = tmp_path / "croissant.json"
    manifest_path = tmp_path / "shard_manifest.json"
    assert labels_path.is_file()
    assert croissant_path.is_file()
    assert manifest_path.is_file()

    # 1. Verify sanitized labels: high-elevation airliner kept, low-elevation decoy masked (>= +5 deg)
    labels_raw = labels_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in labels_raw.splitlines() if line.strip()]
    assert len(records) == 4  # 4 frames for high-altitude airliner; low-altitude decoy rejected
    for rec in records:
        assert rec["schema"] == "openxvs.realsky.label/1"
        assert rec["license"] == "CC-BY-4.0"
        assert rec["pii_stripped"] is True
        assert rec["upstream_adsb_stripped"] is True
        assert rec["elevation_mask_deg"] >= 5.0
        assert rec["ephemeral_track_id"].startswith("trk_")
        assert rec["aerodynamic_class"] == "NARROWBODY_JET"

    # Zero ICAO24 / callsign / tail_number / raw ADS-B JSON leakage anywhere in emitted files
    for secret in ("A12345", "UAL999", "N12345", "B54321", "N54321", "raw_adsb_json", "icao24", "callsign"):
        assert secret not in labels_raw

    # 2. Verify Croissant 1.1 + RAI 1.0 JSON-LD conformance
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    _MOD.validate_realsky_croissant_manifest(croissant)
    assert croissant["dct:conformsTo"] == [
        "http://mlcommons.org/croissant/1.1",
        "http://mlcommons.org/croissant/RAI/1.0",
    ]
    assert croissant["license"] == "CDLA-Permissive-2.0"
    assert croissant["rai:hasSyntheticData"] is False
    assert "civil aviation" in croissant["rai:dataUseCases"].lower()
    assert "+5.0 deg" in croissant["rai:personalSensitiveInformation"]

    # 3. Verify deterministic SHA-256 shard manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 4
    assert manifest["rejected_below_sky_mask"] == 4
    assert manifest["upstream_adsb_stripped"] is True
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["files"]) == 2


def test_polite_polling_provider_guards_and_exponential_429_backoff() -> None:
    """Enforce DR-15 polite poll floors, HTTP 429 exponential backoff, and blocked providers."""
    # Valid polite configs pass
    assert _MOD.validate_polite_poll_config("adsb.lol", 3.5, strip_upstream_json=True)[0] == "adsb.lol"
    assert _MOD.validate_polite_poll_config("flyitalyadsb", 2.5, strip_upstream_json=True)[0] == "flyitalyadsb"

    # Impolite fast polling rejected fail-closed
    with pytest.raises(RealSkyComplianceError, match="3.5 s"):
        _MOD.validate_polite_poll_config("adsb.lol", 2.0, strip_upstream_json=True)
    with pytest.raises(RealSkyComplianceError, match="2.5 s"):
        _MOD.validate_polite_poll_config("flyitalyadsb", 1.5, strip_upstream_json=True)

    # Blocked providers: airplanes.live (HTTP 403), commercial opensky & adsb.fi, unstripped adsb.lol
    for blocked in ("airplanes.live", "opensky", "adsb.fi"):
        with pytest.raises(RealSkyComplianceError):
            _MOD.validate_polite_poll_config(blocked, 5.0, commercial_release=True, strip_upstream_json=True)
    with pytest.raises(RealSkyComplianceError, match="strip_upstream_json"):
        _MOD.validate_polite_poll_config("adsb.lol", 4.0, commercial_release=True, strip_upstream_json=False)

    # Exponential 429 backoff progression
    assert _MOD.compute_backoff_delay_s(3.5, 0) == pytest.approx(3.5)
    assert _MOD.compute_backoff_delay_s(3.5, 1) == pytest.approx(7.0)
    assert _MOD.compute_backoff_delay_s(3.5, 2) == pytest.approx(14.0)
    assert _MOD.compute_backoff_delay_s(3.5, 5, max_backoff_s=60.0) == pytest.approx(60.0)


def test_privacy_elevation_mask_below_5_deg_rejected_fail_closed(tmp_path: Path) -> None:
    """Elevation mask < +5.0 deg must fail closed to prevent capturing ground homes/faces/plates."""
    with pytest.raises(RealSkyComplianceError, match="5.0"):
        _MOD.main(["--smoke-offline", "--elevation-mask-deg", "4.9", "--out-dir", str(tmp_path)])


def test_metar_qnh_cache_enforces_15min_ttl_and_avoids_refetch() -> None:
    """AviationWeather.gov METAR cache enforces >= 15 min TTL and serves cached QNH within TTL."""
    with pytest.raises(ValueError, match="900"):
        _MOD.MetarQnhCache(ttl_s=60.0)

    now = [1000.0]
    calls = [0]

    def fake_fetch(icao: str) -> dict[str, object]:
        calls[0] += 1
        return {"icaoId": icao, "altim": 1004.5, "temp": 11.0, "elev": 4.0}

    cache = _MOD.MetarQnhCache(ttl_s=900.0, clock_fn=lambda: now[0], fetch_fn=fake_fetch)
    obs1 = cache.get_observation("KSFO")
    assert calls[0] == 1
    assert obs1.qnh_hpa == pytest.approx(1004.5)
    assert obs1.t_amb_k == pytest.approx(284.15)
    assert obs1.source == "AVIATIONWEATHER_METAR"

    # Second call 300 s later must hit cache (0 new HTTP fetches)
    now[0] += 300.0
    obs2 = cache.get_observation("KSFO")
    assert calls[0] == 1
    assert obs2 == obs1

    # After TTL (>= 900 s), refreshes cleanly
    now[0] += 650.0
    _ = cache.get_observation("KSFO")
    assert calls[0] == 2


def test_croissant_quarantine_guard_rejects_lard_lardon_pass_msfs_xplane_infiray() -> None:
    """Verify DR-15/DR-19 quarantine guard rejects tainted datasets/simulators and sanctioned sensors."""
    for bad_name in ("LARD 2.0", "LARDON", "PASS", "MSFS 2024", "X-Plane 12", "OpenCelliD", "InfiRay"):
        with pytest.raises(RealSkyComplianceError):
            _MOD.build_realsky_croissant_manifest(
                dataset_name=bad_name,
                version="1.0.0",
                license_spdx="CDLA-Permissive-2.0",
                labels_file_name="labels.jsonl",
                labels_sha256="0" * 64,
                labels_size_bytes=128,
                record_count=1,
                elevation_mask_deg=5.0,
                provider_used="adsb.lol",
            )


def _write_synthetic_realsky_clip_and_adsb(
    tmp_path: Path,
    *,
    paint_targets: bool = True,
    clock_offset_ms: float = 50.0,
) -> dict[str, object]:
    """Create a synthetic Tier-0A smartphone .mp4 + camera_calib.json + recorded ADS-B JSONL."""
    site_lat = 37.6188
    site_lon = -122.3750
    site_alt_m = -28.0
    pitch_up_deg = 12.0
    azimuth_deg = 0.0
    w_px, h_px = 640, 480
    src_fps = 10.0
    n_frames = 32  # 0.0 s .. 3.1 s
    start_utc_iso = "2026-11-22T18:00:00.000Z"
    start_epoch_s = _MOD.parse_utc_iso8601_to_epoch_s(start_utc_iso)

    calib_dict = {
        "fx": 520.0,
        "fy": 520.0,
        "cx": 320.0,
        "cy": 240.0,
        "k1": -0.06,
        "k2": 0.012,
        "p1": 0.0004,
        "p2": -0.0005,
        "k3": 0.0,
        "width": w_px,
        "height": h_px,
    }
    calib_path = tmp_path / "camera_calib.json"
    calib_path.write_text(json.dumps(calib_dict, indent=2), encoding="utf-8")
    intr = CameraIntrinsics(
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
        k1=-0.06,
        k2=0.012,
        p1=0.0004,
        p2=-0.0005,
        k3=0.0,
        width_px=w_px,
        height_px=h_px,
    )
    r_ned_to_cam = _MOD.build_sky_tilted_r_ned_to_cam(azimuth_deg=azimuth_deg, pitch_up_deg=pitch_up_deg)
    sky_mask = compute_sky_elevation_mask(intr, r_ned_to_cam, min_elev_deg=5.0)

    # Two high-elevation civil aircraft (bright & dark) + one low-elevation ground-skimming decoy (< +5 deg)
    t0 = start_epoch_s
    t1 = start_epoch_s + 3.5
    adsb_snapshots_raw = [
        {
            "rx_time_s": t0,
            "ac": [
                {
                    "hex": "A77777",
                    "flight": "SWA777 ",
                    "r": "N777SW",
                    "lat": site_lat + 0.0120,
                    "lon": site_lon - 0.0020,
                    "alt_baro": 1700.0,
                    "alt_geom": 1650.0,
                    "gva": 2,
                    "gs": 130.0,
                    "track": 88.0,
                    "baro_rate": 60.0,
                    "category": "A3",
                },
                {
                    "hex": "C88888",
                    "flight": "DAL888 ",
                    "r": "N888DL",
                    "lat": site_lat + 0.0145,
                    "lon": site_lon + 0.0024,
                    "alt_baro": 1500.0,
                    "alt_geom": 1460.0,
                    "gva": 2,
                    "gs": 120.0,
                    "track": 272.0,
                    "baro_rate": -40.0,
                    "category": "A2",
                },
                {
                    "hex": "D99999",
                    "flight": "LOW999 ",
                    "r": "N999LO",
                    "lat": site_lat + 0.0120,
                    "lon": site_lon + 0.0002,
                    "alt_baro": 85.0,
                    "alt_geom": None,
                    "gva": 0,
                    "gs": 90.0,
                    "track": 90.0,
                    "baro_rate": 0.0,
                    "category": "A1",
                },
            ],
        },
        {
            "rx_time_s": t1,
            "ac": [
                {
                    "hex": "A77777",
                    "flight": "SWA777 ",
                    "r": "N777SW",
                    "lat": site_lat + 0.0121,
                    "lon": site_lon + 0.0006,
                    "alt_baro": 1708.0,
                    "alt_geom": 1658.0,
                    "gva": 2,
                    "gs": 130.0,
                    "track": 93.0,
                    "baro_rate": 60.0,
                    "category": "A3",
                },
                {
                    "hex": "C88888",
                    "flight": "DAL888 ",
                    "r": "N888DL",
                    "lat": site_lat + 0.0144,
                    "lon": site_lon - 0.0001,
                    "alt_baro": 1495.0,
                    "alt_geom": 1455.0,
                    "gva": 2,
                    "gs": 120.0,
                    "track": 267.0,
                    "baro_rate": -40.0,
                    "category": "A2",
                },
                {
                    "hex": "D99999",
                    "flight": "LOW999 ",
                    "r": "N999LO",
                    "lat": site_lat + 0.0120,
                    "lon": site_lon + 0.0018,
                    "alt_baro": 85.0,
                    "alt_geom": None,
                    "gva": 0,
                    "gs": 90.0,
                    "track": 90.0,
                    "baro_rate": 0.0,
                    "category": "A1",
                },
            ],
        },
    ]
    adsb_path = tmp_path / "recorded_adsb.jsonl"
    adsb_path.write_text(
        "".join(json.dumps(row) + "\n" for row in adsb_snapshots_raw),
        encoding="utf-8",
    )

    # Precompute exact Hermite NED endpoints to paint shifted aircraft dots in video frames
    track_specs = [
        ("A77777", 0, (+8.0, -5.0), +85.0),  # Bright sunlit fuselage shifted (+8, -5) px
        ("C88888", 1, (-7.0, +6.0), -90.0),  # Dark silhouette shifted (-7, +6) px
    ]
    endpoints: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float]] = {}
    for hex_id, idx_ac, shift_uv, amp in track_specs:
        ac0 = adsb_snapshots_raw[0]["ac"][idx_ac]
        ac1 = adsb_snapshots_raw[1]["ac"][idx_ac]
        h0 = reconcile_ellipsoidal_altitude_m(baro_alt_ft=ac0["alt_baro"], geom_alt_ft=ac0["alt_geom"], gva=ac0["gva"])
        h1 = reconcile_ellipsoidal_altitude_m(baro_alt_ft=ac1["alt_baro"], geom_alt_ft=ac1["alt_geom"], gva=ac1["gva"])
        p0 = geodetic_to_local_ned(ac0["lat"], ac0["lon"], h0, origin_lat_deg=site_lat, origin_lon_deg=site_lon, origin_alt_ellip_m=site_alt_m)
        p1 = geodetic_to_local_ned(ac1["lat"], ac1["lon"], h1, origin_lat_deg=site_lat, origin_lon_deg=site_lon, origin_alt_ellip_m=site_alt_m)
        v0 = ned_velocity_from_adsb(ground_speed_kt=ac0["gs"], track_deg=ac0["track"], vertical_rate_fpm=ac0["baro_rate"])
        v1 = ned_velocity_from_adsb(ground_speed_kt=ac1["gs"], track_deg=ac1["track"], vertical_rate_fpm=ac1["baro_rate"])
        endpoints[hex_id] = (p0, v0, p1, v1, shift_uv, amp)

    video_path = tmp_path / "phone_clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w_px, h_px))
    assert writer.isOpened()

    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:h_px, 0:w_px]
    expected_true_dots: dict[tuple[int, str], tuple[float, float]] = {}

    # Sample indices for sample_fps=2.0 on 10 FPS video -> raw frames 0, 5, 10, 15, 20, 25, 30
    sampled_raw_to_idx = {raw_i: s_i for s_i, raw_i in enumerate(range(0, n_frames, 5))}

    for raw_i in range(n_frames):
        pts_s = float(raw_i) / src_fps
        t_utc = start_epoch_s + pts_s + (clock_offset_ms / 1000.0)
        # Clean sky gradient + gentle camera read noise
        sky_luma = (
            155.0
            + 20.0 * (yy.astype(np.float32) / float(h_px))
            + rng.normal(0.0, 2.5, (h_px, w_px)).astype(np.float32)
        )
        frame = np.stack(
            [
                np.clip(sky_luma + 20.0, 0.0, 255.0),
                np.clip(sky_luma + 5.0, 0.0, 255.0),
                np.clip(sky_luma - 15.0, 0.0, 255.0),
            ],
            axis=2,
        )
        # Paint bright fake "buildings/cars/plates" strictly BELOW +5 deg horizon to prove pre-encode obliteration!
        frame[~sky_mask] = np.array([235.0, 210.0, 195.0], dtype=np.float32)

        if paint_targets:
            for hex_id, (p0, v0, p1, v1, (du_s, dv_s), amp) in endpoints.items():
                p_ned = interpolate_cubic_hermite_ned(p0, v0, p1, v1, t_mid_s=t_utc, t_k_s=t0, t_k1_s=t1)
                proj = project_ned_to_pixel(p_ned, r_ned_to_cam, intr)
                assert proj is not None and proj.in_image_bounds and proj.elevation_deg >= 5.0
                tu = proj.u_px + du_s
                tv = proj.v_px + dv_s
                blob = amp * np.exp(-((xx - tu) ** 2 + (yy - tv) ** 2) / (2.0 * 2.1 * 2.1))
                frame = np.clip(frame + blob[:, :, np.newaxis], 0.0, 255.0)
                if raw_i in sampled_raw_to_idx:
                    expected_true_dots[(sampled_raw_to_idx[raw_i], hex_id)] = (tu, tv)

        writer.write(frame.astype(np.uint8))
    writer.release()

    return {
        "video_path": video_path,
        "calib_path": calib_path,
        "adsb_path": adsb_path,
        "start_utc_iso": start_utc_iso,
        "start_epoch_s": start_epoch_s,
        "clock_offset_ms": clock_offset_ms,
        "intr": intr,
        "r_ned_to_cam": r_ned_to_cam,
        "sky_mask": sky_mask,
        "expected_true_dots": expected_true_dots,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "site_alt_m": site_alt_m,
        "pitch_up_deg": pitch_up_deg,
    }


def test_video_ingest_utc_alignment_sky_mask_snap_croissant_ed25519_and_xvs_d2_eval(tmp_path: Path) -> None:
    """Full Tier-0A end-to-end proof: .mp4 ingest, UTC sync, >=+5 deg mask, snap, Croissant, Ed25519, XVS-D2."""
    env = _write_synthetic_realsky_clip_and_adsb(tmp_path, paint_targets=True, clock_offset_ms=50.0)
    out_dir = tmp_path / "shard_out"

    rc = _MOD.main(
        [
            "--video-path",
            str(env["video_path"]),
            "--video-start-utc",
            str(env["start_utc_iso"]),
            "--clock-offset-ms",
            "50.0",
            "--sample-fps",
            "2.0",
            "--camera-calib",
            str(env["calib_path"]),
            "--adsb-jsonl",
            str(env["adsb_path"]),
            "--lat",
            str(env["site_lat"]),
            "--lon",
            str(env["site_lon"]),
            "--alt-m",
            str(env["site_alt_m"]),
            "--pitch-up-deg",
            str(env["pitch_up_deg"]),
            "--elevation-mask-deg",
            "5.0",
            "--eval-xvs-d2",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    labels_path = out_dir / "labels.jsonl"
    frames_tar_path = out_dir / "frames_shard.tar"
    croissant_path = out_dir / "croissant.json"
    manifest_path = out_dir / "shard_manifest.json"
    sig_path = out_dir / "manifest_sig.json"
    eval_path = out_dir / "xvs_d2_eval.json"
    for p in (labels_path, frames_tar_path, croissant_path, manifest_path, sig_path, eval_path):
        assert p.is_file(), f"Missing expected artifact: {p.name}"

    # 1. Verify exact UTC alignment, visual snap accuracy (< 1.5 px), low-elev rejection, and zero PII leak
    labels_raw = labels_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in labels_raw.splitlines() if line.strip()]
    # 7 sampled frames (0.0, 0.5, ..., 3.0 s) x 2 sky tracks = 14 records; 7 low-elev decoy hits rejected
    assert len(records) == 14
    for secret in ("A77777", "C88888", "D99999", "SWA777", "DAL888", "LOW999", "N777SW", "N888DL", "N999LO"):
        assert secret not in labels_raw

    start_epoch = float(env["start_epoch_s"])
    expected_dots = env["expected_true_dots"]
    trk_bright = _MOD._ephemeral_track_id("openxvs-realsky-v1-session", "A77777")
    trk_dark = _MOD._ephemeral_track_id("openxvs-realsky-v1-session", "C88888")

    pol_seen: set[str] = set()
    for rec in records:
        f_idx = int(rec["frame_idx"])
        expected_utc = _MOD.epoch_s_to_utc_iso8601(start_epoch + 0.5 * f_idx + 0.050)
        assert rec["frame_utc_iso"] == expected_utc
        assert rec["frame_file"] == f"frames/frame_{f_idx:06d}.png"
        assert rec["visual_snapped"] is True
        assert rec["visual_snap_snr"] >= 4.5
        pol_seen.add(rec["visual_snap_polarity"])
        hex_id = "A77777" if rec["ephemeral_track_id"] == trk_bright else "C88888"
        tu, tv = expected_dots[(f_idx, hex_id)]
        err_px = math.hypot(rec["u_px"] - tu, rec["v_px"] - tv)
        assert err_px < 1.5, f"Snap error {err_px:.2f} px >= 1.5 px on frame {f_idx} track {hex_id}"
    assert pol_seen == {"BRIGHT_TOPHAT", "DARK_BOTTOMHAT"}

    # 2. Verify >= +5 deg pre-encode sky mask zeroed EVERY below-horizon pixel (`max == 0`) on disk
    sky_mask = env["sky_mask"]
    assert np.any(~sky_mask)
    with tarfile.open(frames_tar_path, mode="r") as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)
        assert len(members) == 7
        for member in members:
            raw_img = tf.extractfile(member)
            assert raw_img is not None
            decoded = cv2.imdecode(np.frombuffer(raw_img.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None
            assert int(decoded[~sky_mask].max()) == 0
            assert int(decoded[sky_mask].max()) > 100

    # 3. Verify Croissant 1.1 + RAI 1.0 references frames_shard.tar with matching SHA-256 & byte size
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    _MOD.validate_realsky_croissant_manifest(croissant)
    tar_bytes = frames_tar_path.read_bytes()
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()
    dist_by_id = {d["@id"]: d for d in croissant["distribution"]}
    assert "realsky_frames_tar" in dist_by_id
    assert "realsky_masked_frames" in dist_by_id
    assert dist_by_id["realsky_frames_tar"]["sha256"] == tar_sha
    assert dist_by_id["realsky_frames_tar"]["contentSize"] == f"{len(tar_bytes)} B"

    # 4. Verify Ed25519 detached signature round-trip + tamper rejection
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sig_doc = json.loads(sig_path.read_text(encoding="utf-8"))
    assert manifest["exported_frame_count"] == 7
    assert manifest["visual_snap_locked"] == 14
    assert manifest["visual_snap_fallback"] == 0
    assert manifest["rejected_below_sky_mask"] == 7
    assert verify_manifest_signature(manifest, sig_doc) is True
    assert verify_manifest_signature(dict(manifest, record_count=999), sig_doc) is False

    # 5. Verify real pinned XVS-D2 FP16 ONNX zero-shot evaluation executed cleanly
    eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
    assert eval_doc["schema"] == "openxvs.realsky.xvs_d2_eval/1"
    assert eval_doc["model_sha256"].startswith("2c9578f3")
    assert eval_doc["frames_evaluated"] == 7
    assert eval_doc["total_gt_targets"] == 14
    assert len(eval_doc["per_frame"]) == 7
    assert 0.0 <= eval_doc["precision"] <= 1.0
    assert 0.0 <= eval_doc["recall"] <= 1.0
    assert 0.0 <= eval_doc["f1"] <= 1.0
    assert eval_doc["mean_latency_ms"] > 0.0


def test_video_ingest_pure_noise_falls_back_honestly_without_hallucinating_snaps(tmp_path: Path) -> None:
    """When sky contains only noise (no visible aircraft), contrast snap MUST fall back (`snapped=False`)."""
    env = _write_synthetic_realsky_clip_and_adsb(tmp_path, paint_targets=False, clock_offset_ms=0.0)
    out_dir = tmp_path / "noise_out"

    rc = _MOD.main(
        [
            "--video-path",
            str(env["video_path"]),
            "--video-start-utc",
            str(env["start_utc_iso"]),
            "--sample-fps",
            "2.0",
            "--camera-calib",
            str(env["calib_path"]),
            "--adsb-jsonl",
            str(env["adsb_path"]),
            "--lat",
            str(env["site_lat"]),
            "--lon",
            str(env["site_lon"]),
            "--alt-m",
            str(env["site_alt_m"]),
            "--pitch-up-deg",
            str(env["pitch_up_deg"]),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "shard_manifest.json").read_text(encoding="utf-8"))
    sig_doc = json.loads((out_dir / "manifest_sig.json").read_text(encoding="utf-8"))
    assert verify_manifest_signature(manifest, sig_doc) is True
    assert manifest["visual_snap_locked"] == 0
    assert manifest["visual_snap_fallback"] == 14

    records = [json.loads(line) for line in (out_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 14
    for rec in records:
        assert rec["visual_snapped"] is False
        assert rec["visual_snap_polarity"] == "NONE"
        assert rec["visual_snap_du_px"] == 0.0
        assert rec["visual_snap_dv_px"] == 0.0


def test_video_cli_fail_closed_utc_and_flag_guards(tmp_path: Path) -> None:
    """Reject naive timestamps, missing --video-start-utc, and --eval-xvs-d2 without --video-path."""
    with pytest.raises(ValueError, match="timezone"):
        _MOD.parse_utc_iso8601_to_epoch_s("2026-11-22T18:00:00")

    dummy_mp4 = tmp_path / "dummy.mp4"
    dummy_mp4.write_bytes(b"")
    with pytest.raises(SystemExit):
        _MOD.main(["--smoke-offline", "--video-path", str(dummy_mp4), "--out-dir", str(tmp_path / "o1")])

    with pytest.raises(SystemExit):
        _MOD.main(["--smoke-offline", "--eval-xvs-d2", "--out-dir", str(tmp_path / "o2")])
