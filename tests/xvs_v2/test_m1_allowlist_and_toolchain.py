"""Task 0 verification: Allowlist v1.4 (NOAA GEOID18) and verified Windows + WSL2 toolchain manifest."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = REPO_ROOT / "docs" / "xvs" / "spec" / "data_allowlist.json"
TOOLCHAIN_MANIFEST_PATH = REPO_ROOT / "output" / "xvs" / "v2_m1" / "toolchain_manifest.json"


def test_allowlist_v14_clears_noaa_geoid18() -> None:
    doc = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert doc["version"].startswith(("1.4", "2.0", "2.1")) and "1.4" in doc["version"], (
        f"Expected allowlist v1.4+ (preserving 1.4 history), got {doc['version']!r}"
    )

    ids = {entry["id"]: entry for entry in doc["upstream_allowed"]}
    assert "noaa_geoid18" in ids, "Missing noaa_geoid18 entry in upstream_allowed"

    geoid = ids["noaa_geoid18"]
    assert "Public Domain" in geoid["license"]

    hosts = geoid["host"] if isinstance(geoid["host"], list) else [geoid["host"]]
    prefixes = geoid["path_prefix"] if isinstance(geoid["path_prefix"], list) else [geoid["path_prefix"]]
    assert "cdn.proj.org" in hosts
    assert "geodesy.noaa.gov" in hosts
    assert any("/us_noaa_g2018u0.tif" in p for p in prefixes)
    assert any("/PCPROD/GEOID18/" in p for p in prefixes)


def test_toolchain_manifest_records_all_verified_versions() -> None:
    assert TOOLCHAIN_MANIFEST_PATH.is_file(), f"Missing toolchain manifest: {TOOLCHAIN_MANIFEST_PATH}"
    manifest = json.loads(TOOLCHAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema"] == "xvs.v2_m1_toolchain_manifest/1"

    win_tools = manifest["windows"]
    for key in (
        "rustc",
        "cargo",
        "rustup_default_toolchain",
        "wasm_bindgen",
        "naga",
        "cbindgen",
        "cargo_deny",
        "node",
        "npm",
    ):
        assert isinstance(win_tools.get(key), str) and win_tools[key].strip(), f"Missing/empty windows.{key}"

    targets = set(win_tools.get("rustup_targets_installed", []))
    assert "wasm32-unknown-unknown" in targets
    assert "thumbv7em-none-eabi" in targets
    assert win_tools.get("cdylib_and_test_smoke_pass") is True

    wsl_tools = manifest["wsl2_ubuntu"]
    for key in ("rustc", "cargo", "cargo_kani"):
        assert isinstance(wsl_tools.get(key), str) and wsl_tools[key].strip(), f"Missing/empty wsl2_ubuntu.{key}"

    ort = manifest["onnxruntime_web"]
    assert isinstance(ort.get("version"), str) and ort["version"].strip()
    assert len(ort.get("non_jsep_webgpu_wasm_files", [])) >= 1

    mav = manifest["mavlink_bearing_track_id_check"]
    assert mav["selected_id"] == 52001
    assert mav["collision_detected"] is False
    assert "common.xml" in mav["checked_definitions"]
    assert "ardupilotmega.xml" in mav["checked_definitions"]
