"""TDD tests for Milestone M1 Task 4: Gate-G1 D1-KF ONNX export & USGS 3DEP + GEOID18 KSFO pack."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "xvs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs"))

from xvs.detection.d1_model import D1HeatmapDetector, build_d1_kf
from xvs.sim.allowlist import URLClassifier

CKPT_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "checkpoints" / "d1_kf_k3_synth.pt"
MODELS_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "models"
FP32_ONNX_PATH = MODELS_DIR / "d1_kf_k3_fp32.onnx"
FP16_ONNX_PATH = MODELS_DIR / "d1_kf_k3_fp16.onnx"
REGISTRY_PATH = MODELS_DIR / "registry.json"
PARITY_JSON_PATH = REPO_ROOT / "output" / "xvs" / "v2_m1" / "d1_onnx_parity.json"

ASSETS_DIR = REPO_ROOT / "apps" / "livedeck" / "public" / "assets"
KSFO_BIN_PATH = ASSETS_DIR / "ksfo_3dep_geoid18_v1.bin"
KSFO_MANIFEST_PATH = ASSETS_DIR / "ksfo_3dep_geoid18_v1.manifest.json"
KSFO_SOURCES_PATH = REPO_ROOT / "output" / "xvs" / "v2_m1" / "ksfo_3dep_data_sources.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _brotli_size_bytes(path: Path) -> int:
    js = (
        "const fs=require('fs'),z=require('zlib');"
        "const b=fs.readFileSync(process.argv[1]);"
        "const c=z.brotliCompressSync(b,{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}});"
        "process.stdout.write(String(c.length));"
    )
    proc = subprocess.run(
        ["node", "-e", js, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout.strip())


def _load_gate_g1_detector() -> D1HeatmapDetector:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = build_d1_kf(
        num_frames_k=int(ckpt["num_frames_k"]),
        ablation_mode=str(ckpt["ablation_mode"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device="cpu", dtype=torch.float32)
    return model


def _peak_centroid_uv(hm: np.ndarray, offset: np.ndarray) -> tuple[float, float]:
    """Compute sub-pixel peak centroid (u, v) on interior [2:-2, 2:-2]."""
    h = hm[0, 0].copy()
    h[:2, :] = -1.0
    h[-2:, :] = -1.0
    h[:, :2] = -1.0
    h[:, -2:] = -1.0
    flat_idx = int(np.argmax(h))
    vy, ux = divmod(flat_idx, h.shape[1])
    du = float(offset[0, 0, vy, ux])
    dv = float(offset[0, 1, vy, ux])
    return float(ux) + du, float(vy) + dv


def _parse_ksfo_pack(path: Path) -> tuple[dict, list[np.ndarray]]:
    raw = path.read_bytes()
    assert len(raw) >= 16
    magic, version, header_len = struct.unpack_from("<8sII", raw, 0)
    assert magic == b"XVS3DEP1"
    assert version == 1
    header = json.loads(raw[16 : 16 + header_len].decode("utf-8"))
    rings: list[np.ndarray] = []
    for r in header["rings"]:
        off = int(r["byte_offset"])
        nbytes = int(r["byte_length"])
        w = int(r["width"])
        h = int(r["height"])
        assert nbytes == w * h * 2
        arr = np.frombuffer(raw[off : off + nbytes], dtype="<u2").reshape(h, w)
        rings.append(arr)
    return header, rings


def test_d1_onnx_fp32_parity_and_no_double_norm() -> None:
    """FP32 ONNX wraps forward_normalized ONLY and matches PyTorch < 1e-5 on synthetic, G3a & flat tiles."""
    assert FP32_ONNX_PATH.is_file(), f"Missing FP32 ONNX model: {FP32_ONNX_PATH}"
    onnx_model = onnx.load(str(FP32_ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    import export_d1_onnx  # noqa: E402

    model = _load_gate_g1_detector()
    sess = ort.InferenceSession(str(FP32_ONNX_PATH), providers=["CPUExecutionProvider"])

    in_meta = sess.get_inputs()
    out_meta = sess.get_outputs()
    assert [i.name for i in in_meta] == ["x_norm"]
    assert list(in_meta[0].shape) == [1, 5, 256, 256]
    assert [o.name for o in out_meta] == ["heatmap", "offset", "log_scale"]
    assert [list(o.shape) for o in out_meta] == [
        [1, 1, 256, 256],
        [1, 2, 256, 256],
        [1, 2, 256, 256],
    ]

    fixtures = export_d1_onnx.build_parity_fixtures(model)
    assert "synthetic_point_target" in fixtures
    assert "flat_zero_contrast" in fixtures
    assert any(k.startswith("g3a_recorded") for k in fixtures)

    for name, x_norm_np in fixtures.items():
        assert x_norm_np.shape == (1, 5, 256, 256)
        assert x_norm_np.dtype == np.float32
        with torch.no_grad():
            pt_out = model.forward_normalized(torch.from_numpy(x_norm_np))
        ort_hm, ort_off, ort_lsc = sess.run(None, {"x_norm": x_norm_np})

        for arr in (ort_hm, ort_off, ort_lsc):
            assert np.all(np.isfinite(arr)), f"Non-finite FP32 ONNX output on {name}"

        diff_hm = float(np.max(np.abs(ort_hm - pt_out["heatmap"].numpy())))
        diff_off = float(np.max(np.abs(ort_off - pt_out["offset"].numpy())))
        diff_lsc = float(np.max(np.abs(ort_lsc - pt_out["log_scale"].numpy())))
        assert max(diff_hm, diff_off, diff_lsc) < 1e-5, (
            f"FP32 diff too large on {name}: hm={diff_hm}, off={diff_off}, lsc={diff_lsc}"
        )

    # Confirm forward_normalized ONLY (no double-normalization): an un-centered/scaled input
    # must match forward_normalized(x) and differ from forward(x) = forward_normalized(normalize_input(x)).
    x_probe = fixtures["synthetic_point_target"] * 3.5 + 1.25
    with torch.no_grad():
        pt_norm_only = model.forward_normalized(torch.from_numpy(x_probe))["heatmap"].numpy()
        pt_double = model.forward(torch.from_numpy(x_probe))["heatmap"].numpy()
    ort_probe_hm = sess.run(["heatmap"], {"x_norm": x_probe})[0]
    assert float(np.max(np.abs(ort_probe_hm - pt_norm_only))) < 1e-5
    assert float(np.max(np.abs(ort_probe_hm - pt_double))) > 1e-3


def test_d1_onnx_fp16_size_parity_and_centroid() -> None:
    """FP16 ONNX is <= 512,000 B, matches forward_normalized < 2e-3, and peak centroid < 0.05 px."""
    assert FP16_ONNX_PATH.is_file(), f"Missing FP16 ONNX model: {FP16_ONNX_PATH}"
    size_bytes = FP16_ONNX_PATH.stat().st_size
    assert size_bytes <= 512_000, f"FP16 ONNX too large: {size_bytes} > 512000 bytes"

    onnx_model = onnx.load(str(FP16_ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    import export_d1_onnx  # noqa: E402

    model = _load_gate_g1_detector()
    sess = ort.InferenceSession(str(FP16_ONNX_PATH), providers=["CPUExecutionProvider"])
    in_dtype = sess.get_inputs()[0].type
    fixtures = export_d1_onnx.build_parity_fixtures(model)

    for name, x_norm_np in fixtures.items():
        with torch.no_grad():
            pt_out = model.forward_normalized(torch.from_numpy(x_norm_np))
        feed_arr = x_norm_np.astype(np.float16) if "float16" in in_dtype else x_norm_np
        ort_hm, ort_off, ort_lsc = sess.run(None, {"x_norm": feed_arr})
        ort_hm_f32 = ort_hm.astype(np.float32)
        ort_off_f32 = ort_off.astype(np.float32)
        ort_lsc_f32 = ort_lsc.astype(np.float32)

        for arr in (ort_hm_f32, ort_off_f32, ort_lsc_f32):
            assert np.all(np.isfinite(arr)), f"Non-finite FP16 ONNX output on {name}"

        diff_hm = float(np.max(np.abs(ort_hm_f32 - pt_out["heatmap"].numpy())))
        diff_off = float(np.max(np.abs(ort_off_f32 - pt_out["offset"].numpy())))
        diff_lsc = float(np.max(np.abs(ort_lsc_f32 - pt_out["log_scale"].numpy())))
        assert max(diff_hm, diff_off, diff_lsc) < 2e-3, (
            f"FP16 diff too large on {name}: hm={diff_hm}, off={diff_off}, lsc={diff_lsc}"
        )

        if name != "flat_zero_contrast":
            pt_u, pt_v = _peak_centroid_uv(
                pt_out["heatmap"].numpy(), pt_out["offset"].numpy()
            )
            ort_u, ort_v = _peak_centroid_uv(ort_hm_f32, ort_off_f32)
            dist_px = math.hypot(ort_u - pt_u, ort_v - pt_v)
            assert dist_px < 0.05, (
                f"FP16 centroid diff {dist_px:.4f} px >= 0.05 px on {name}"
            )


def test_model_registry_and_parity_manifest() -> None:
    """registry.json pins SHA-256s for active d1_kf_k3_g1 + reserved xvs_d2_web_v2; parity JSON matches."""
    assert REGISTRY_PATH.is_file(), f"Missing registry.json: {REGISTRY_PATH}"
    assert PARITY_JSON_PATH.is_file(), f"Missing parity JSON: {PARITY_JSON_PATH}"

    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models_by_id = {m["id"]: m for m in reg["models"]}
    assert reg["active_model_id"] == "d1_kf_k3_g1"
    assert "d1_kf_k3_g1" in models_by_id
    assert "xvs_d2_web_v2" in models_by_id
    assert models_by_id["xvs_d2_web_v2"]["status"] == "reserved_m2"

    active = models_by_id["d1_kf_k3_g1"]
    assert active["preprocessing"]["forward_normalized_only"] is True
    assert active["artifacts"]["fp32"]["sha256"] == _sha256_file(FP32_ONNX_PATH)
    assert active["artifacts"]["fp16"]["sha256"] == _sha256_file(FP16_ONNX_PATH)
    assert active["artifacts"]["fp16"]["bytes"] <= 512_000

    parity = json.loads(PARITY_JSON_PATH.read_text(encoding="utf-8"))
    assert parity["all_checks_passed"] is True
    assert parity["fp32_max_abs_diff"] < 1e-5
    assert parity["fp16_max_abs_diff"] < 2e-3
    assert parity["fp16_max_centroid_diff_px"] < 0.05
    assert parity["fp32_onnx_sha256"] == _sha256_file(FP32_ONNX_PATH)
    assert parity["fp16_onnx_sha256"] == _sha256_file(FP16_ONNX_PATH)


def test_ksfo_3dep_geoid18_pack_budget_datum_and_allowlist() -> None:
    """KSFO 3DEP+GEOID18 pack is <= 300 KiB Brotli, ellipsoidal h in [-35, -22] m, allowlist-clean."""
    assert KSFO_BIN_PATH.is_file(), f"Missing KSFO terrain pack: {KSFO_BIN_PATH}"
    assert KSFO_MANIFEST_PATH.is_file(), f"Missing KSFO manifest: {KSFO_MANIFEST_PATH}"
    assert KSFO_SOURCES_PATH.is_file(), f"Missing KSFO data_sources: {KSFO_SOURCES_PATH}"

    br_bytes = _brotli_size_bytes(KSFO_BIN_PATH)
    assert br_bytes <= 307_200, f"KSFO pack Brotli size {br_bytes} > 307200 bytes"

    header, rings = _parse_ksfo_pack(KSFO_BIN_PATH)
    assert header["terrain_mode"] == "usgs_3dep_geoid18_ellipsoidal"
    assert header["geoid_model"] == "GEOID18"
    assert header["datum"] == "WGS84_ELLIPSOIDAL"

    h_28l = float(header["ksfo_28l_thr_wgs84_h_m"])
    assert -35.0 <= h_28l <= -22.0, (
        f"KSFO 28L ellipsoidal height {h_28l:.3f} m outside [-35.0, -22.0] m"
    )
    assert len(rings) == 2
    assert header["rings"][0]["extent_km"] == pytest.approx(40.0)
    assert header["rings"][1]["extent_km"] == pytest.approx(10.0)

    # Verify dequantized elevation span shows real SF Bay relief (bay water near -32 m, hills > 100 m)
    q = header["quantization"]
    outer_m = float(q["h_min_m"]) + rings[0].astype(np.float64) * float(q["h_step_m"])
    assert float(np.min(outer_m)) < -20.0
    assert float(np.max(outer_m)) > 100.0

    rwy_ids = {r["id"]: r for r in header["runways"]}
    assert {"28L", "28R", "Taxiway C"}.issubset(rwy_ids.keys())
    assert 55.0 <= float(rwy_ids["28L"]["width_m"]) <= 66.0
    assert 18.0 <= float(rwy_ids["Taxiway C"]["width_m"]) <= 28.0

    manifest = json.loads(KSFO_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["terrain_mode"] == "usgs_3dep_geoid18_ellipsoidal"
    assert manifest["pack_sha256"] == _sha256_file(KSFO_BIN_PATH)
    assert manifest["brotli_bytes"] <= 307_200

    clf = URLClassifier()
    sources = json.loads(KSFO_SOURCES_PATH.read_text(encoding="utf-8"))
    assert len(sources["sources"]) >= 3
    seen_ids = set()
    for src in sources["sources"]:
        verdict, reason, entry = clf.classify_upstream_url(src["url"])
        assert verdict == "ALLOWED", f"Disallowed source URL {src['url']}: {reason}"
        assert entry is not None
        seen_ids.add(entry["id"])
    assert {"usgs_3dep", "noaa_geoid18", "ourairports"}.issubset(seen_ids)


def test_ksfo_procedural_ci_mode_explicit_and_allowlist_guard(tmp_path: Path) -> None:
    """Explicit --terrain-mode procedural-ci emits 'procedural_ci' header; allowlist blocks forbidden hosts."""
    out_proc = tmp_path / "ksfo_proc_ci.bin"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "xvs" / "build_ksfo_3dep_pack.py"),
            "--terrain-mode",
            "procedural-ci",
            "--out",
            str(out_proc),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"procedural-ci failed:\n{proc.stdout}\n{proc.stderr}"
    assert out_proc.is_file()
    header, rings = _parse_ksfo_pack(out_proc)
    assert header["terrain_mode"] == "procedural_ci"
    assert header["terrain_mode"] != "usgs_3dep_geoid18_ellipsoidal"
    assert len(rings) == 2
    assert _brotli_size_bytes(out_proc) <= 307_200

    import build_ksfo_3dep_pack  # noqa: E402

    clf = URLClassifier()
    with pytest.raises(PermissionError):
        build_ksfo_3dep_pack.allowlisted_fetch(
            "https://tile.googleapis.com/v1/3dtiles/root.json",
            tmp_path / "blocked.bin",
            clf,
            [],
        )
    with pytest.raises(PermissionError):
        build_ksfo_3dep_pack.allowlisted_fetch(
            "https://api.cesium.com/v1/assets/1/endpoint",
            tmp_path / "blocked2.bin",
            clf,
            [],
        )
