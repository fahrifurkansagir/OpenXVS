"""Track 2 (Part A) Verification Suite:
  1. Edge ARM64 OCI Dockerfiles (`rpi5`, `orin_nx_jp62`, `orin_nx_jp72`) & native ARM64 GitHub Actions CI
  2. Croissant 1.1 + RAI (`CDLA-Permissive-2.0`) generator & `--double-blind` scrubbing
  3. `OPENXVS_NON_FLIGHT_SIM_ONLY=1` fail-closed lockout in `hil_bridge.py` & `server.py` (EU PLD Art. 4(7))
  4. OpenXVS Studio v4.5 REL1/REL2/REL3/REL4 compliance (0 path leaks, 0 TLS/AES crates, immutable watermark)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "xvs"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_croissant_rai import (  # noqa: E402
    CDLA_PERMISSIVE_2_0_URL,
    CROISSANT_1_1_URI,
    CROISSANT_RAI_1_0_URI,
    DOUBLE_BLIND_FORBIDDEN_PATTERNS,
    NON_FLIGHT_WATERMARK as CROISSANT_WATERMARK,
    REQUIRED_RAI_KEYS,
    build_croissant_rai_metadata,
    validate_croissant_rai_document,
)
from xvs.v2.studio_bridge import (  # noqa: E402
    StudioInferenceEngine,
    create_studio_http_server,
    load_studio_models,
)
from xvs.v2.studio_bridge.hil_bridge import (  # noqa: E402
    NON_FLIGHT_WATERMARK,
    NonFlightSimOnlyError,
    StudioHilBridgeManager,
    is_non_flight_sim_only,
)


# ---- 1. Edge ARM64 OCI Dockerfiles & GitHub Actions Workflow ----


def test_rpi5_dockerfile_glibc_no_alpine_and_non_flight_lock() -> None:
    path = REPO_ROOT / "docker" / "rpi5.Dockerfile"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")

    # Strictly NO Alpine musl base image
    assert not re.search(r"^\s*FROM\b[^\n]*alpine", text, re.IGNORECASE | re.MULTILINE)
    assert "cgr.dev/chainguard/wolfi-base" in text
    assert "gcr.io/distroless/cc-debian12:nonroot" in text
    assert "NON_FLIGHT_SIM_ONLY=1" in text
    assert "OPENXVS_NON_FLIGHT_SIM_ONLY=1" in text
    assert "ORT_DYLIB_PATH=" in text
    assert "openxvs-edge-infer" in text
    assert "--remap-path-prefix=" in text


def test_orin_nx_jp62_and_jp72_dockerfiles_and_nvidia_sla_eula() -> None:
    jp62 = (REPO_ROOT / "docker" / "orin_nx_jp62.Dockerfile").read_text(encoding="utf-8")
    jp72 = (REPO_ROOT / "docker" / "orin_nx_jp72.Dockerfile").read_text(encoding="utf-8")

    assert "nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime" in jp62
    assert "nvcr.io/nvidia/cuda:13." in jp72 and "runtime-ubuntu24.04" in jp72

    for name, content in (("jp62", jp62), ("jp72", jp72)):
        assert "NON_FLIGHT_SIM_ONLY=1" in content, f"{name} missing NON_FLIGHT_SIM_ONLY=1"
        assert "TensorRT" in content and "1.2" in content and "Critical Applications" in content, (
            f"{name} missing NVIDIA TensorRT SLA Sec. 1.2 & 2 Critical Applications EULA notice"
        )
        assert "/etc/openxvs/NVIDIA_TENSORRT_EULA_NOTICE.txt" in content, (
            f"{name} missing runtime EULA notice file"
        )
        assert NON_FLIGHT_WATERMARK in content, f"{name} missing canonical watermark"


def test_docker_readme_and_arm64_ci_workflow_private_first_cosign() -> None:
    readme = (REPO_ROOT / "docker" / "README.md").read_text(encoding="utf-8")
    assert "ubuntu-24.04-arm" in readme
    assert "--tlog-upload=false" in readme
    assert "UNVERIFIED" in readme
    assert "TensorRT SLA" in readme

    wf_path = REPO_ROOT / ".github" / "workflows" / "xvs_edge_arm64_oci.yml"
    assert wf_path.is_file()
    wf = wf_path.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04-arm" in wf
    assert "anchore/sbom-action" in wf
    assert "spdx-json" in wf
    assert "cosign sign --key" in wf
    assert "--tlog-upload=false" in wf
    assert "NON_FLIGHT_SIM_ONLY" in wf
    assert "docker/rpi5.Dockerfile" in wf
    assert "docker/orin_nx_jp62.Dockerfile" in wf
    assert "docker/orin_nx_jp72.Dockerfile" in wf


# ---- 2. Croissant 1.1 + RAI (`CDLA-Permissive-2.0`) & Double-Blind Scrubbing ----


def test_canonical_croissant_1_1_rai_metadata_on_disk() -> None:
    croissant_path = REPO_ROOT / "output" / "xvs" / "v2_m3" / "xvs_bench_croissant.json"
    if not croissant_path.is_file():
        pytest.skip("Clean-room checkout omits generated output/xvs/v2_m3/xvs_bench_croissant.json")
    assert croissant_path.is_file()
    doc = json.loads(croissant_path.read_text(encoding="utf-8"))

    validate_croissant_rai_document(doc, double_blind=False)
    assert CROISSANT_1_1_URI in doc["conformsTo"]
    assert CROISSANT_RAI_1_0_URI in doc["conformsTo"]
    assert doc["license"] == CDLA_PERMISSIVE_2_0_URL
    assert "apache.org" not in str(doc["license"]).lower()
    assert len(doc["distribution"]) == 16
    assert len(doc["recordSet"]) == 5
    for k in REQUIRED_RAI_KEYS:
        assert k in doc, f"Missing RAI key {k}"
    assert CROISSANT_WATERMARK in doc["rai:dataLimitations"]


def test_generate_croissant_rai_double_blind_cli_scrubs_all_identifiers(tmp_path: Path) -> None:
    out_path = tmp_path / "neurips_2027_double_blind_croissant.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "xvs" / "generate_croissant_rai.py"),
            "--double-blind",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}\n{proc.stdout}"
    assert out_path.is_file()

    raw_text = out_path.read_text(encoding="utf-8")
    doc = json.loads(raw_text)
    validate_croissant_rai_document(doc, double_blind=True)
    assert doc["license"] == CDLA_PERMISSIVE_2_0_URL
    assert "Anonymous Authors" in doc["citeAs"]

    for pat in DOUBLE_BLIND_FORBIDDEN_PATTERNS:
        assert pat.search(raw_text) is None, f"Double-blind leak matched {pat.pattern!r}"

    # Verify library API matches CLI invariants
    doc_lib = build_croissant_rai_metadata(REPO_ROOT, double_blind=True)
    validate_croissant_rai_document(doc_lib, double_blind=True)


# ---- 3. OPENXVS_NON_FLIGHT_SIM_ONLY=1 Fail-Closed Lockout in Studio Bridge ----


def test_non_flight_env_parser_fail_closed_and_dev_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENXVS_NON_FLIGHT_SIM_ONLY", raising=False)
    monkeypatch.delenv("NON_FLIGHT_SIM_ONLY", raising=False)
    assert is_non_flight_sim_only() is False

    for dev_val in ("0", "false", "FALSE", "no", "off", " 0 "):
        monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", dev_val)
        assert is_non_flight_sim_only() is False

    # Fail-closed on "1", "true", empty string, or any unrecognized value
    for lock_val in ("1", "true", "yes", "on", "", "   ", "maybe", "enabled"):
        monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", lock_val)
        assert is_non_flight_sim_only() is True, f"Expected fail-closed True for {lock_val!r}"


def test_hil_manager_and_http_server_enforce_non_flight_sim_only_lockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", "1")
    mgr = StudioHilBridgeManager()
    st = mgr.status()
    assert st["non_flight_sim_only"] is True
    assert st["tx_enabled"] is False
    assert st["actuator_authority"] is False
    assert st["watermark"] == NON_FLIGHT_WATERMARK

    for mode in ("flightgear_udp", "mavlink2_udp"):
        with pytest.raises(NonFlightSimOnlyError, match="Art. 4\\(7\\)"):
            mgr.connect(mode=mode)
        assert mgr.status()["tx_enabled"] is False

    with pytest.raises(NonFlightSimOnlyError):
        mgr.set_autonomy_triggers(tx_enabled=True)

    # HTTP server enforcement (smoke_fast engine keeps test < 1s)
    bundle = load_studio_models(device_pref="cpu", smoke_fast=True)
    engine = StudioInferenceEngine(bundle)
    server = create_studio_http_server(engine, host="127.0.0.1", port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. GET /api/health stamped with non_flight_sim_only=True + canonical watermark
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=10.0) as resp:
            assert resp.status == 200
            assert resp.headers["X-OpenXVS-Non-Flight-Sim-Only"] == "1"
            health = json.loads(resp.read().decode("utf-8"))
        assert health["non_flight_sim_only"] is True
        assert health["actuator_authority"] is False
        assert health["watermark"] == NON_FLIGHT_WATERMARK

        # 2. GET /api/demo_tick stamped with non_flight_sim_only=True + watermark (when nav source present)
        import importlib.util
        if importlib.util.find_spec("xvs.v2.xvs_nav") is not None:
            with urllib.request.urlopen(
                f"{base_url}/api/demo_tick?scenario=cbdr&frame=0&images=0", timeout=15.0
            ) as resp:
                assert resp.status == 200
                tick = json.loads(resp.read().decode("utf-8"))
            assert tick["non_flight_sim_only"] is True
            assert tick["watermark"] == NON_FLIGHT_WATERMARK

        # 3. POST /api/hil/connect MUST return HTTP 403 with EU PLD Art. 4(7) lockout
        for mode in ("flightgear_udp", "mavlink2_udp"):
            req = urllib.request.Request(
                f"{base_url}/api/hil/connect",
                data=json.dumps({"mode": mode, "rx_port": 5500, "tx_port": 5501}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=10.0)
            assert exc_info.value.code == 403
            err_body = json.loads(exc_info.value.read().decode("utf-8"))
            assert err_body["code"] == "NON_FLIGHT_SIM_ONLY_LOCKOUT"
            assert "4(7)" in err_body["error"]
            assert err_body["tx_enabled"] is False
            assert err_body["actuator_authority"] is False
            assert err_body["non_flight_sim_only"] is True
            assert err_body["watermark"] == NON_FLIGHT_WATERMARK

        # 4. When explicitly disabled in dev fixture (`0`), `/api/hil/connect` idle succeeds cleanly
        monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", "0")
        req_idle = urllib.request.Request(
            f"{base_url}/api/hil/connect",
            data=json.dumps({"mode": "idle"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_idle, timeout=10.0) as resp:
            assert resp.status == 200
            idle_doc = json.loads(resp.read().decode("utf-8"))
        assert idle_doc["mode"] == "idle"
        assert idle_doc["non_flight_sim_only"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


# ---- 4. OpenXVS Studio v4.5 REL1/REL2/REL3/REL4 Hardening Verification ----


def test_studio_rel1_rel3_rel4_cargo_ui_and_release_binary() -> None:
    cargo_toml = (REPO_ROOT / "apps" / "studio" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    assert "[profile.release]" in cargo_toml
    assert "strip = true" in cargo_toml
    assert "debug = 0" in cargo_toml
    assert "lto = true" in cargo_toml
    assert 'panic = "abort"' in cargo_toml
    assert 'default = ["non_flight_sim_only"]' in cargo_toml
    assert "load-dynamic" in cargo_toml
    assert "default-features = false" in cargo_toml
    assert "xvs-active-range" in cargo_toml
    assert "xvs-rta-core" in cargo_toml

    # 5D002 private-first guardrail: 0 TLS/AES confidentiality crypto crates in Cargo.lock
    cargo_lock = (REPO_ROOT / "apps" / "studio" / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
    banned_crypto = (
        "rustls",
        "openssl",
        "openssl-sys",
        "native-tls",
        "schannel",
        "aes",
        "aes-gcm",
        "chacha20",
        "quinn",
        "ring",
        "aws-lc-rs",
        "hyper-tls",
        "hyper-rustls",
        "ureq",
    )
    for crate in banned_crypto:
        assert not re.search(rf'^name = "{re.escape(crate)}"$', cargo_lock, re.MULTILINE), (
            f"5D002 violation: forbidden crate {crate!r} found in apps/studio/src-tauri/Cargo.lock"
        )

    # Immutable UI watermark & native Tauri IPC checks
    index_html = (REPO_ROOT / "apps" / "studio" / "index.html").read_text(encoding="utf-8")
    assert "studio-non-flight-watermark" in index_html
    assert "ZERO ACTUATION" in index_html

    bridge_ts = (REPO_ROOT / "apps" / "studio" / "src" / "ui" / "bridge_client.ts").read_text(encoding="utf-8")
    assert NON_FLIGHT_WATERMARK in bridge_ts
    assert "native_health" in bridge_ts
    assert "native_demo_tick" in bridge_ts
    assert "native_analyze_image" in bridge_ts
    assert "native_hil_connect" in bridge_ts

    # REL1 release binary zero-path-leak check (runs whenever release binary is present)
    exe_path = REPO_ROOT / "apps" / "studio" / "src-tauri" / "target" / "release" / "openxvs-studio.exe"
    if exe_path.is_file():
        raw = exe_path.read_bytes()
        ascii_txt = raw.decode("latin-1")
        utf16_txt = raw.decode("utf-16-le", errors="ignore")
        for forbidden in ("<OPENXVS_WORKSPACE>", "<USER_HOME>", "<OPENXVS_WORKSPACE>"):
            assert re.search(re.escape(forbidden), ascii_txt, re.IGNORECASE) is None, (
                f"REL1 ASCII path leak in openxvs-studio.exe: {forbidden!r}"
            )
            assert re.search(re.escape(forbidden), utf16_txt, re.IGNORECASE) is None, (
                f"REL1 UTF-16LE path leak in openxvs-studio.exe: {forbidden!r}"
            )
