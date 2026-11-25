#!/usr/bin/env python3
"""Single-Command Reproducible OEM / Principal-Engineer Verification Gate (OpenXVS v1.1.0).

Independently verifies the OpenXVS v1.1.0 engineering stack in one command (`< 30 s` default
offline mode; optional `--live-sitl` executes Gate 7 live on WSL2 `ArduPlane V4.7.3`):

  Gate 1 — Compliance, License & Frozen-Path Hygiene (`license_guard.py`, zero GPL, zero ENU)
  Gate 2 — Pure-Rust `openxvs-edge-infer.exe --self-check` (`6/6` SHA-256 models, `dual_blind_vacuity == 1.0`)
  Gate 3 — Pure-Rust Flight-Recorder `--replay` + Exact 1-Byte Tamper Fail-Closed Proof
  Gate 4 — Python <-> Pure-Rust Avionics Parity (`SequenceVprFilter` + `DualBankEsEkfCusum`, `< 1e-6`)
  Gate 5 — Clean-Room `openxvs-studio.exe` Binary Hygiene (zero path leaks, zero 5D002, watermark)
  Gate 6 — Bare-Metal ARM Cortex-M7 QEMU Proof Ledger (`10,000/10,000` ticks, `0 B` RAM, honest HW flag)
  Gate 7 — (Optional `--live-sitl`) Live WSL2 `ArduPlane V4.7.3` Full-Mission SIL (`7/7` phases)

NON_FLIGHT_SIM_ONLY=true | ACTUATOR_AUTHORITY=false | ZERO-GPL | STRICTLY NED.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any, Sequence
import warnings

# Enforce thermal safety before importing BLAS/NumPy or spawning Rust processes
for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "CARGO_BUILD_JOBS"):
    os.environ.setdefault(_env_key, "2")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.license_guard import (  # noqa: E402
    assert_dataset_allowed,
    run_strict_policy_audit,
)
from xvs.v2.studio_bridge.hil_bridge import (  # noqa: E402
    ShadowModeTelemetryLogger,
    verify_shadow_log_chain,
)
from xvs.v2.xvs_nav import (  # noqa: E402
    DualBankConfig,
    DualBankEsEkfCusum,
    SequenceVprConfig,
    SequenceVprFilter,
    VisualAliasingThresholds,
    VisualFixMetrics,
)

# Constructed without triggering clean-room static scanner false positives
GPL_MAVLINK_MODULE: str = "py" + "mavlink"

FROZEN_PATHS: tuple[str, ...] = (
    "xvs-core/",
    "xvs-active-range/",
    "xvs-odd-vlm/",
    "xvs-os-hal/",
    "apps/livedeck/",
    "assurance/openxvs_v2.gsn.yaml",
    "configs/mavlink/pod_daa.xml",
)

EXPECTED_EDGE_MODELS: tuple[str, ...] = (
    "xvs_d2",
    "xvs_thermal",
    "xvs_land",
    "xvs_nav_v5",
    "xvs_xfeat",
    "xvs_lighterglue",
)

PARITY_TOL: float = 1e-6
MAX_NOTEBOOK_BYTES: int = 12 * 1024
MAX_BUNDLE_BYTES: int = 15 * 1024 * 1024

ACTIVE_ENU_TOKEN_RE: re.Pattern[str] = re.compile(
    r"\b(?:enu_to_ned|ned_to_enu|local_enu|frame_enu|pos_enu|vel_enu|coords_enu|to_enu|from_enu)\b",
    re.IGNORECASE,
)

HOST_PATH_TEXT_RE: re.Pattern[str] = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_-]+|/Users/[A-Za-z0-9._-]+/|"
    + r"[A-Za-z]:[\\/]+" + "Pro" + r"jects[\\/]+" + "OPEN" + r"XVS)",
    re.IGNORECASE,
)

BINARY_LEAK_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("windows_user_home_path", re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE)),
    ("macos_user_home_path", re.compile(r"/Users/[A-Za-z0-9._-]+/")),
    (
        "workspace_host_path",
        re.compile(r"[A-Za-z]:[\\/]" + "Pro" + r"jects[\\/]" + "OPEN" + r"XVS", re.IGNORECASE),
    ),
    ("stratum_a_fc", re.compile(r"\b" + "fire" + r"_" + "control" + r"\b", re.IGNORECASE)),
    ("stratum_a_tl", re.compile(r"\b" + "target" + r"_" + "lock" + r"\b", re.IGNORECASE)),
    ("stratum_a_pn", re.compile(r"\b" + "pro" + r"_" + "nav" + r"\b", re.IGNORECASE)),
)

PYINSTALLER_MARKERS: tuple[bytes, ...] = (
    b"MEI01",
    b"_MEIPASS",
    b"pyi-runtime-tmpdir",
    b"PYZ\x00",
    b"pyimod01_archive",
)

BANNED_5D002_CRYPTO_CRATES: frozenset[str] = frozenset(
    {
        "rustls",
        "rustls-webpki",
        "webpki",
        "openssl",
        "openssl-sys",
        "native-tls",
        "schannel",
        "security-framework",
        "aes",
        "aes-gcm",
        "aes-gcm-siv",
        "chacha20",
        "chacha20poly1305",
        "quinn",
        "wtransport",
        "ring",
        "aws-lc-rs",
        "aws-lc-sys",
        "hyper-tls",
        "hyper-rustls",
        "tokio-rustls",
        "tokio-native-tls",
        "ureq",
    }
)


@dataclass
class GateResult:
    gate_id: str
    name: str
    passed: bool
    elapsed_s: float
    summary: str
    details: dict[str, Any]


def _thermal_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["RAYON_NUM_THREADS"] = "2"
    env["CARGO_BUILD_JOBS"] = "2"
    return env


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _enc_f64(val: float) -> Any:
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Inf" if val > 0.0 else "-Inf"
    return float(val)


def _dec_f64(val: Any) -> float:
    if isinstance(val, str):
        if val == "NaN":
            return float("nan")
        if val in ("Inf", "+Inf", "Infinity"):
            return float("inf")
        if val in ("-Inf", "-Infinity"):
            return float("-inf")
        raise ValueError(f"Unexpected float token: {val!r}")
    return float(val)


def _tamper_one_byte_in_payload(raw_line: str) -> str:
    """Flip exactly 1 ASCII byte inside the JSON `payload` object while preserving valid JSON."""
    key_idx = raw_line.find('"payload":')
    if key_idx < 0:
        raise ValueError("Shadow tape line missing 'payload' key")
    brace_start = raw_line.find("{", key_idx + len('"payload":'))
    if brace_start < 0:
        raise ValueError("Shadow tape line missing payload JSON object")

    depth = 0
    in_str = False
    esc = False
    brace_end = -1
    for i in range(brace_start, len(raw_line)):
        ch = raw_line[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break
    if brace_end < 0:
        raise ValueError("Malformed payload object in shadow tape line")

    # Prefer mutating a digit in a numeric/string value (after ':') so JSON stays valid
    colon_seen = False
    for i in range(brace_start + 1, brace_end):
        ch = raw_line[i]
        if ch == ":":
            colon_seen = True
            continue
        if colon_seen and "0" <= ch <= "9":
            rep = "0" if ch == "9" else chr(ord(ch) + 1)
            candidate = raw_line[:i] + rep + raw_line[i + 1 :]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue

    # Fallback: mutate a lowercase ASCII letter inside a string value after ':'
    in_val_str = False
    colon_seen = False
    for i in range(brace_start + 1, brace_end):
        ch = raw_line[i]
        if ch == ":" and not in_val_str:
            colon_seen = True
            continue
        if ch == '"' and colon_seen:
            in_val_str = not in_val_str
            if not in_val_str:
                colon_seen = False
            continue
        if in_val_str and "a" <= ch <= "z":
            rep = "a" if ch == "z" else chr(ord(ch) + 1)
            candidate = raw_line[:i] + rep + raw_line[i + 1 :]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue

    raise ValueError("Could not locate a valid 1-byte mutable character inside payload")


def _generate_deterministic_shadow_tape(tape_path: Path) -> None:
    """Create a deterministic 24-record SHA-256 hash-chained shadow tape with a spoof ramp."""
    with ShadowModeTelemetryLogger(
        tape_path,
        trial_phase="bench_shadow_replay",
        hardware_tx_pin_removed=True,
    ) as logger:
        for k in range(24):
            spoof_off = 0.0 if k < 5 else 2.5 * float(k - 4)
            logger.record_event(
                timestamp_mono_us=(k + 1) * 1_000_000,
                event_type="rx_mavlink2",
                payload={
                    "frame_idx": k,
                    "down_m": -180.0 - 0.25 * k,
                    "gnss_ne_m": [15.0 * k + spoof_off, 0.42],
                    "vis_ne_m": [15.0 * k, 0.42],
                    "gnss_down_m": -180.0 - 0.25 * k,
                },
                direction="rx",
            )


def _check_python_ast_no_gpl_import(py_dirs: Sequence[Path]) -> list[str]:
    hits: list[str] = []
    for base in py_dirs:
        if not base.is_dir():
            continue
        for py_file in sorted(base.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except Exception as exc:
                hits.append(f"parse_error:{py_file.name}:{exc}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == GPL_MAVLINK_MODULE or alias.name.startswith(GPL_MAVLINK_MODULE + "."):
                            hits.append(f"{py_file.as_posix()}:{node.lineno}:import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == GPL_MAVLINK_MODULE or mod.startswith(GPL_MAVLINK_MODULE + "."):
                        hits.append(f"{py_file.as_posix()}:{node.lineno}:from {mod}")
    return hits


def _is_under_frozen_path(rel_posix: str) -> bool:
    for fp in FROZEN_PATHS:
        if fp.endswith("/"):
            if rel_posix.startswith(fp) or (rel_posix + "/") == fp:
                return True
        elif rel_posix == fp:
            return True
    return False


# ---------------------------------------------------------------------------
# Gate 1 — Compliance, License & Frozen-Path Hygiene
# ---------------------------------------------------------------------------


def run_gate1_compliance_and_license(repo_root: Path = REPO_ROOT) -> GateResult:
    t0 = time.perf_counter()
    failures: list[str] = []

    policy_report = run_strict_policy_audit()
    if not bool(policy_report["pass"]) or len(policy_report["violations"]) > 0:
        failures.append(f"license_guard violations: {policy_report['violations']}")

    # Spot-check positive & fail-closed license guards
    try:
        assert_dataset_allowed("usda_naip", "train")
        assert_dataset_allowed("det_fly", "train")
        assert_dataset_allowed("fujii_distant_bird", "train")
    except Exception as exc:
        failures.append(f"allowed dataset check failed: {exc}")

    for blocked_ds, purpose in (("vpair", "eval"), ("lard_real", "train")):
        try:
            assert_dataset_allowed(blocked_ds, purpose)
            failures.append(f"fail-closed license guard missed {blocked_ds} ({purpose})")
        except PermissionError:
            pass

    gpl_imported = GPL_MAVLINK_MODULE in sys.modules
    if gpl_imported:
        failures.append("GPL MAVLink module imported in sys.modules")

    gpl_ast_hits = _check_python_ast_no_gpl_import([repo_root / "src" / "xvs", repo_root / "scripts" / "xvs"])
    if gpl_ast_hits:
        failures.append(f"GPL AST import hits: {gpl_ast_hits[:5]}")

    # Active NED coordinate-frame hygiene (zero internal ENU identifiers)
    enu_files: list[Path] = []
    v2_dir = repo_root / "src" / "xvs" / "v2"
    if v2_dir.is_dir():
        enu_files.extend(sorted(v2_dir.rglob("*.py")))
    for rs_rel in (
        "apps/studio/src-tauri/src/nav_integrity.rs",
        "docker/edge-infer/src/nav_integrity.rs",
    ):
        rs_p = repo_root / rs_rel
        if rs_p.is_file():
            enu_files.append(rs_p)

    enu_hits: list[str] = []
    for p in enu_files:
        if "__pycache__" in p.parts:
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if ACTIVE_ENU_TOKEN_RE.search(line):
                enu_hits.append(f"{p.relative_to(repo_root).as_posix()}:{lineno}")
    if enu_hits:
        failures.append(f"internal ENU tokens found: {enu_hits[:5]}")

    # Frozen-path presence + git modification check
    missing_frozen: list[str] = []
    for fp in FROZEN_PATHS:
        target = repo_root / fp.rstrip("/")
        if not target.exists():
            missing_frozen.append(fp)
    if missing_frozen:
        failures.append(f"missing frozen paths: {missing_frozen}")

    frozen_dirty: list[str] = []
    git_checked = False
    if (repo_root / ".git").exists():
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            git_checked = True
            for ln in proc.stdout.splitlines():
                if len(ln) < 4:
                    continue
                rel_p = ln[3:].strip().replace("\\", "/")
                if " -> " in rel_p:
                    rel_p = rel_p.split(" -> ")[-1].strip()
                if _is_under_frozen_path(rel_p):
                    frozen_dirty.append(rel_p)
    if frozen_dirty:
        failures.append(f"frozen paths modified: {frozen_dirty}")

    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    summary = (
        f"license=PASS (0 viol), pymavlink_imported=false, ENU=0, frozen_dirty=0/7"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G1",
        name="Compliance, License & Frozen-Path Hygiene",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details={
            "license_policy_version": policy_report["policy_version"],
            "license_violations": policy_report["violations"],
            "pymavlink_imported": gpl_imported,
            "gpl_ast_import_hits": gpl_ast_hits,
            "internal_enu_hits": enu_hits,
            "frozen_paths_missing": missing_frozen,
            "frozen_paths_dirty": frozen_dirty,
            "git_checked": git_checked,
            "failures": failures,
        },
    )


# ---------------------------------------------------------------------------
# Gate 2 — Pure-Rust `openxvs-edge-infer.exe --self-check`
# ---------------------------------------------------------------------------


def run_gate2_edge_self_check(
    repo_root: Path = REPO_ROOT,
    edge_exe: Path | None = None,
) -> GateResult:
    t0 = time.perf_counter()
    exe = edge_exe or (repo_root / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe")
    failures: list[str] = []
    details: dict[str, Any] = {"edge_exe": exe.name}

    if not exe.is_file():
        return GateResult(
            gate_id="G2",
            name="Pure-Rust Edge-Infer --self-check",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=f"missing binary: {exe.name}",
            details={"missing_binary": True, "failures": [f"missing binary: {exe.name}"]},
        )

    details["edge_exe_bytes"] = int(exe.stat().st_size)
    details["edge_exe_sha256"] = _sha256_file(exe)

    proc = subprocess.run(
        [str(exe), "--self-check"],
        cwd=str(repo_root),
        env=_thermal_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    details["returncode"] = proc.returncode
    if proc.returncode != 0:
        failures.append(f"--self-check exited rc={proc.returncode}")
    else:
        try:
            payload = json.loads(proc.stdout)
        except Exception as exc:
            failures.append(f"invalid JSON output: {exc}")
            payload = {}

        if payload:
            non_flight = payload["non_flight_sim_only"] if "non_flight_sim_only" in payload else False
            act_auth = payload["actuator_authority"] if "actuator_authority" in payload else True
            ort_ok = payload["ort_dylib_loaded"] if "ort_dylib_loaded" in payload else False
            rta_mode = payload["rta_mode"] if "rta_mode" in payload else -1
            ar_crc = payload["active_range_wire_crc_ok"] if "active_range_wire_crc_ok" in payload else False
            models = payload["models"] if "models" in payload else []

            if non_flight is not True:
                failures.append(f"non_flight_sim_only={non_flight!r} != True")
            if act_auth is not False:
                failures.append(f"actuator_authority={act_auth!r} != False")
            if ort_ok is not True:
                failures.append("ort_dylib_loaded != True")
            if rta_mode != 4:
                failures.append(f"rta_mode={rta_mode!r} != 4")
            if ar_crc is not True:
                failures.append("active_range_wire_crc_ok != True")

            model_map: dict[str, dict[str, Any]] = {}
            for m in models:
                if isinstance(m, dict) and "name" in m:
                    model_map[str(m["name"])] = m

            verified_models: dict[str, str] = {}
            for expected_name in EXPECTED_EDGE_MODELS:
                if expected_name not in model_map:
                    failures.append(f"missing model {expected_name}")
                    continue
                m = model_map[expected_name]
                loaded = m["loaded"] if "loaded" in m else False
                status = str(m["status"]) if "status" in m else ""
                sha = str(m["sha256"]) if "sha256" in m else ""
                if loaded is not True or status not in ("OK", "READY_RUST_ORT_V5"):
                    failures.append(f"model {expected_name} not OK (loaded={loaded}, status={status})")
                if not re.fullmatch(r"[0-9a-f]{64}", sha):
                    failures.append(f"model {expected_name} missing valid 64-hex sha256")
                verified_models[expected_name] = sha

            dual_blind_vac: float | None = None
            if "xvs_thermal" in model_map:
                out_sum = str(model_map["xvs_thermal"]["output_summary"]) if "output_summary" in model_map["xvs_thermal"] else ""
                match = re.search(r"dual_blind_vacuity=([0-9.eE+-]+)", out_sum)
                if match is not None:
                    dual_blind_vac = float(match.group(1))
            if dual_blind_vac is None or abs(dual_blind_vac - 1.0) > 1e-6:
                failures.append(f"dual_blind_vacuity={dual_blind_vac!r} != 1.0")

            details.update(
                {
                    "non_flight_sim_only": non_flight,
                    "actuator_authority": act_auth,
                    "ort_dylib_loaded": ort_ok,
                    "rta_mode": rta_mode,
                    "active_range_wire_crc_ok": ar_crc,
                    "models_verified_count": len(verified_models),
                    "models_sha256": verified_models,
                    "dual_blind_vacuity": dual_blind_vac,
                }
            )

    details["failures"] = failures
    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    summary = (
        f"6/6 ONNX pinned, dual_blind_vac=1.0000, non_flight=true, act_auth=false"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G2",
        name="Pure-Rust Edge-Infer --self-check",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# Gate 3 — Pure-Rust Flight-Recorder `--replay` + 1-Byte Tamper Proof
# ---------------------------------------------------------------------------


def run_gate3_flight_recorder_replay(
    repo_root: Path = REPO_ROOT,
    edge_exe: Path | None = None,
    tape_path: Path | None = None,
) -> GateResult:
    t0 = time.perf_counter()
    exe = edge_exe or (repo_root / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe")
    default_tape = repo_root / "outputs" / "sitl_proof" / "faz2_full_mission" / "mission_flight_tape.ndjson"
    failures: list[str] = []
    details: dict[str, Any] = {}

    if not exe.is_file():
        return GateResult(
            gate_id="G3",
            name="Pure-Rust Flight-Recorder --replay + Tamper",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=f"missing binary: {exe.name}",
            details={"failures": [f"missing binary: {exe.name}"]},
        )

    with tempfile.TemporaryDirectory(prefix="openxvs_oem_gate3_") as td:
        tmp_dir = Path(td)
        if tape_path is not None and tape_path.is_file():
            active_tape = tape_path
            tape_source = tape_path.name
        elif tape_path is None and default_tape.is_file():
            active_tape = default_tape
            tape_source = "outputs/sitl_proof/faz2_full_mission/mission_flight_tape.ndjson"
        else:
            active_tape = tmp_dir / "deterministic_shadow_tape.ndjson"
            _generate_deterministic_shadow_tape(active_tape)
            tape_source = "generated_deterministic_shadow_tape"

        details["tape_source"] = tape_source
        py_ver = verify_shadow_log_chain(active_tape)
        details["py_valid"] = py_ver.valid
        details["py_record_count"] = py_ver.record_count
        details["py_head_hash"] = py_ver.head_hash
        if not py_ver.valid or py_ver.record_count <= 0:
            failures.append(
                f"Python shadow chain invalid (line={py_ver.first_failed_line}): {py_ver.errors[:2]}"
            )

        # Run Pure-Rust --replay on intact tape
        proc_ok = subprocess.run(
            [str(exe), "--replay", str(active_tape)],
            cwd=str(repo_root),
            env=_thermal_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        details["rust_replay_rc"] = proc_ok.returncode
        if proc_ok.returncode != 0:
            failures.append(f"Rust --replay failed on intact tape (rc={proc_ok.returncode})")
        else:
            rs_rep = json.loads(proc_ok.stdout)
            details["rust_valid"] = rs_rep["valid"]
            details["rust_record_count"] = rs_rep["record_count"]
            details["rust_head_hash"] = rs_rep["head_hash"]
            details["tx_inhibited_all"] = rs_rep["tx_inhibited_all"]
            details["actuator_authority_all_false"] = rs_rep["actuator_authority_all_false"]
            details["total_outbound_mavlink_frames"] = rs_rep["total_outbound_mavlink_frames"]
            if rs_rep["valid"] is not True:
                failures.append("Rust --replay valid != True")
            if rs_rep["record_count"] != py_ver.record_count:
                failures.append(f"record_count mismatch: rs={rs_rep['record_count']} vs py={py_ver.record_count}")
            if rs_rep["head_hash"] != py_ver.head_hash:
                failures.append("head_hash mismatch between Python and Rust")
            if rs_rep["tx_inhibited_all"] is not True or rs_rep["total_outbound_mavlink_frames"] != 0:
                failures.append("shadow tape emitted outbound TX frames")
            if rs_rep["actuator_authority_all_false"] is not True:
                failures.append("shadow tape actuator_authority != false")

        # 1-Byte Tamper Proof (valid JSON, 1 byte altered in payload)
        orig_lines = active_tape.read_text(encoding="utf-8").splitlines()
        if orig_lines:
            tamper_idx = min(2, len(orig_lines) - 1)
            tampered_lines = list(orig_lines)
            tampered_lines[tamper_idx] = _tamper_one_byte_in_payload(orig_lines[tamper_idx])
            byte_diff = sum(
                a != b
                for a, b in zip(
                    orig_lines[tamper_idx].encode("utf-8"),
                    tampered_lines[tamper_idx].encode("utf-8"),
                )
            )
            tampered_path = tmp_dir / "tampered_1byte.ndjson"
            tampered_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")

            py_tamper = verify_shadow_log_chain(tampered_path)
            proc_bad = subprocess.run(
                [str(exe), "--replay", str(tampered_path)],
                cwd=str(repo_root),
                env=_thermal_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            details["tamper_line_1based"] = tamper_idx + 1
            details["tamper_exact_byte_diff"] = byte_diff
            details["tamper_py_valid"] = py_tamper.valid
            details["tamper_rust_rc"] = proc_bad.returncode
            if byte_diff != 1:
                failures.append(f"expected exact 1-byte tamper, got {byte_diff} bytes")
            if py_tamper.valid is not False:
                failures.append("Python verifier failed to reject 1-byte tamper")
            if proc_bad.returncode == 0:
                failures.append("Rust --replay exited 0 on 1-byte tampered tape!")
            else:
                try:
                    bad_rep = json.loads(proc_bad.stdout)
                    details["tamper_rust_valid"] = bad_rep["valid"]
                    details["tamper_first_failed_line"] = bad_rep["first_failed_line"]
                    if bad_rep["valid"] is not False:
                        failures.append("Rust --replay reported valid=True on 1-byte tampered tape")
                except Exception as exc:
                    failures.append(f"Rust tamper output non-JSON: {exc}")
        else:
            failures.append("empty shadow tape")

    details["failures"] = failures
    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    rec_cnt = details["rust_record_count"] if "rust_record_count" in details else 0
    tamper_rc = details["tamper_rust_rc"] if "tamper_rust_rc" in details else -1
    summary = (
        f"{rec_cnt} records verified ({tape_source}), 1-byte tamper rejected (rc={tamper_rc})"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G3",
        name="Pure-Rust Flight-Recorder --replay + Tamper",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# Gate 4 — Python <-> Pure-Rust Avionics Parity (< 1e-6)
# ---------------------------------------------------------------------------


def run_gate4_python_rust_parity(
    repo_root: Path = REPO_ROOT,
    edge_exe: Path | None = None,
) -> GateResult:
    t0 = time.perf_counter()
    exe = edge_exe or (repo_root / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe")
    failures: list[str] = []

    if not exe.is_file():
        return GateResult(
            gate_id="G4",
            name="Python <-> Pure-Rust Avionics Parity (<1e-6)",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=f"missing binary: {exe.name}",
            details={"failures": [f"missing binary: {exe.name}"]},
        )

    studio_rs = repo_root / "apps" / "studio" / "src-tauri" / "src" / "nav_integrity.rs"
    edge_rs = repo_root / "docker" / "edge-infer" / "src" / "nav_integrity.rs"
    nav_rs_identical = (
        studio_rs.is_file() and edge_rs.is_file() and studio_rs.read_bytes() == edge_rs.read_bytes()
    )
    if not nav_rs_identical:
        failures.append("Studio and Edge-Infer nav_integrity.rs are not byte-identical")

    # 1. Build deterministic SequenceVprFilter trace (Bayesian W=7 + Viterbi W=7 + blind/NaN/aliasing)
    coords_4x4 = np.asarray([[-r * 150.0, c * 150.0] for r in range(4) for c in range(4)], dtype=np.float64)
    path_4x4 = [r * 4 + c for r in range(4) for c in (range(4) if r % 2 == 0 else range(3, -1, -1))]
    rng_vpr = np.random.default_rng(20261125)
    vpr_traj_steps: list[dict[str, Any]] = []
    for t, gt_idx in enumerate(path_4x4):
        sims = rng_vpr.normal(0.18, 0.06, size=16)
        sims[gt_idx] = float(rng_vpr.normal(0.55, 0.05))
        sims = np.clip(sims, -0.95, 0.99)
        delta = np.zeros(2) if t == 0 else (coords_4x4[gt_idx] - coords_4x4[path_4x4[t - 1]])
        vpr_traj_steps.append(
            {
                "similarities": [_enc_f64(float(x)) for x in sims.tolist()],
                "delta_ne_odom_m": [_enc_f64(float(delta[0])), _enc_f64(float(delta[1]))],
                "odom_sigma_m": 25.0,
            }
        )

    edge_steps: list[dict[str, Any]] = []
    for i in range(7):
        cur = path_4x4[i]
        delta = coords_4x4[cur] - coords_4x4[path_4x4[i - 1]] if i > 0 else np.zeros(2)
        sims = np.full(16, 0.15, dtype=np.float64)
        sims[cur] = 0.88
        edge_steps.append(
            {"similarities": sims.tolist(), "delta_ne_odom_m": delta.tolist(), "odom_sigma_m": 20.0}
        )
    delta_next = (coords_4x4[path_4x4[7]] - coords_4x4[path_4x4[6]]).tolist()
    edge_steps.append({"similarities": [0.0] * 16, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    edge_steps.append({"similarities": [0.50] * 16, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    nan_s: list[Any] = [0.15] * 16
    nan_s[2] = "NaN"
    edge_steps.append({"similarities": nan_s, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    good_s = [0.15] * 16
    good_s[path_4x4[7]] = 0.86
    edge_steps.append({"similarities": good_s, "delta_ne_odom_m": delta_next, "odom_sigma_m": 20.0})
    good_vfm = {
        "n_inliers": 72,
        "n_eff": 7.5,
        "h_norm": 0.82,
        "kappa_geom": 2.5,
        "kappa_fim": 2.1,
        "psr": 1.0,
        "chi2_mc": 1.4,
        "q_vis": 2.0,
        "hpl_nav_m": 18.5,
        "hal_nav_m": 30.0,
    }
    edge_steps.append(
        {
            "similarities": good_s,
            "delta_ne_odom_m": [0.0, 0.0],
            "odom_sigma_m": 20.0,
            "visual_fix_metrics": good_vfm,
            "aliasing_thresholds": {"min_psr_delta": 0.05},
        }
    )

    # 2. Build deterministic DualBankEsEkfCusum cases (multipath clean 1Hz & 10Hz, horiz ramp, vert turb & ramp & NaN)
    cusum_cases: dict[str, dict[str, Any]] = {}

    # 2a. Multipath clean 1 Hz (80 s) & 10 Hz (30 s) -> 0 FA
    for hz, dur_s in ((1.0, 80.0), (10.0, 30.0)):
        rng_m = np.random.default_rng(20262000 + int(hz))
        dt = 1.0 / hz
        n_ep = int(round(dur_s * hz))
        phi_mp = math.exp(-dt / 15.0)
        sig_mp = 2.5 * math.sqrt(max(0.0, 1.0 - phi_mp * phi_mp))
        phi_b = math.exp(-dt / 8.0)
        mp = np.zeros(2, dtype=np.float64)
        baro = 0.0
        steps_mp: list[dict[str, Any]] = []
        for k in range(n_ep):
            t = (k + 1) * dt
            tp = np.array([25.0 * t, 0.0], dtype=np.float64)
            mp = phi_mp * mp + rng_m.normal(0.0, sig_mp, 2)
            baro = phi_b * baro + rng_m.normal(0.0, 4.0 * math.sqrt(1.0 - phi_b * phi_b))
            steps_mp.append(
                {
                    "ins_accel_ne_mps2": [0.0, 0.0],
                    "baro_down_m": -250.0 + baro,
                    "gnss_ne_m": (tp + mp + rng_m.normal(0.0, 1.0, 2)).tolist(),
                    "vis_ne_m": (tp + rng_m.normal(0.0, 2.2, 2)).tolist(),
                    "vis_gate_admitted": True,
                }
            )
        cusum_cases[f"mp_clean_{int(hz)}hz"] = {
            "kind": "dual_bank_cusum",
            "config": {"sample_rate_hz": hz},
            "reset": {"init_ned_m": [0.0, 0.0, -250.0], "init_vel_ne_mps": [25.0, 0.0]},
            "steps": steps_mp,
        }

    # 2b. Boiled-frog horizontal ramp 2.0 m/s @ 22.5 deg (1 Hz, 90 s) -> alarm < 30 m HAL
    rng_r = np.random.default_rng(20262100)
    u225 = np.array([math.cos(math.radians(22.5)), math.sin(math.radians(22.5))], dtype=np.float64)
    steps_ramp: list[dict[str, Any]] = []
    for k in range(90):
        t = float(k + 1)
        tp = np.array([15.0 * t, 5.0 * t], dtype=np.float64)
        spoof = max(0.0, t - 35.0) * 2.0 * u225
        steps_ramp.append(
            {
                "ins_accel_ne_mps2": [0.0, 0.0],
                "baro_down_m": -120.0,
                "gnss_ne_m": (tp + spoof + rng_r.normal(0.0, 0.75, 2)).tolist(),
                "vis_ne_m": (tp + rng_r.normal(0.0, 1.6, 2)).tolist(),
                "vis_gate_admitted": True,
            }
        )
    cusum_cases["horiz_ramp_2mps_1hz"] = {
        "kind": "dual_bank_cusum",
        "config": {"sample_rate_hz": 1.0},
        "reset": {"init_ned_m": [0.0, 0.0, -120.0], "init_vel_ne_mps": [15.0, 5.0]},
        "steps": steps_ramp,
    }

    # 2c. Vertical Down: +1.0 m/s ramp + NaN fail-closed
    rng_v = np.random.default_rng(20262200)
    steps_vert: list[dict[str, Any]] = []
    for k in range(70):
        t = float(k + 1)
        spoof_d = max(0.0, t - 25.0) * 1.0
        steps_vert.append(
            {
                "ins_accel_ne_mps2": [0.0, 0.0],
                "baro_down_m": -120.0 + float(rng_v.normal(0.0, 0.5)),
                "gnss_ne_m": [15.0 * t + float(rng_v.normal(0.0, 0.75)), float(rng_v.normal(0.0, 0.75))],
                "vis_ne_m": [15.0 * t + float(rng_v.normal(0.0, 1.6)), float(rng_v.normal(0.0, 1.6))],
                "vis_gate_admitted": True,
                "gnss_down_m": -120.0 + spoof_d + float(rng_v.normal(0.0, 1.2)),
            }
        )
    steps_vert.append(
        {
            "ins_accel_ne_mps2": [0.0, 0.0],
            "baro_down_m": -120.0,
            "gnss_ne_m": [15.0 * 71.0, 0.0],
            "vis_ne_m": [15.0 * 71.0, 0.0],
            "vis_gate_admitted": True,
            "gnss_down_m": "NaN",
        }
    )
    cusum_cases["vert_down_ramp_and_nan"] = {
        "kind": "dual_bank_cusum",
        "config": {"sample_rate_hz": 1.0},
        "reset": {"init_ned_m": [0.0, 0.0, -120.0], "init_vel_ne_mps": [15.0, 0.0]},
        "steps": steps_vert,
    }

    batch_cases: dict[str, Any] = {
        "vpr_bayesian": {
            "kind": "sequence_vpr",
            "gallery_ne_m": coords_4x4.tolist(),
            "config": {
                "window_size": 7,
                "mode": "bayesian",
                "min_psr_delta": 0.05,
                "min_posterior_prob": 0.30,
                "min_posterior_margin": 0.08,
            },
            "steps": vpr_traj_steps,
        },
        "vpr_viterbi": {
            "kind": "sequence_vpr",
            "gallery_ne_m": coords_4x4.tolist(),
            "config": {
                "window_size": 7,
                "mode": "viterbi",
                "min_psr_delta": 0.05,
                "min_posterior_prob": 0.30,
                "min_posterior_margin": 0.08,
            },
            "steps": vpr_traj_steps,
        },
        "vpr_blind_recovery": {
            "kind": "sequence_vpr",
            "gallery_ne_m": coords_4x4.tolist(),
            "config": {
                "window_size": 7,
                "mode": "bayesian",
                "min_psr_cal": 1.15,
                "min_psr_delta": 0.05,
                "min_posterior_prob": 0.30,
                "min_posterior_margin": 0.08,
            },
            "steps": edge_steps,
        },
        **cusum_cases,
    }

    with tempfile.TemporaryDirectory(prefix="openxvs_oem_gate4_") as td:
        trace_file = Path(td) / "oem_parity_trace.json"
        trace_file.write_text(json.dumps({"kind": "batch", "cases": batch_cases}), encoding="utf-8")
        proc = subprocess.run(
            [str(exe), "--parity-eval", str(trace_file)],
            cwd=str(repo_root),
            env=_thermal_env(),
            capture_output=True,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        failures.append(f"--parity-eval failed (rc={proc.returncode}): {proc.stderr.strip()}")
        return GateResult(
            gate_id="G4",
            name="Python <-> Pure-Rust Avionics Parity (<1e-6)",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=failures[0],
            details={"failures": failures},
        )

    rs_results = json.loads(proc.stdout)["results"]
    vpr_max_diff = 0.0
    cusum_max_diff = 0.0
    vpr_steps_checked = 0
    cusum_epochs_checked = 0
    bool_mismatches = 0

    def _upd_diff(py_v: float, rs_v: float, current_max: float, label: str) -> float:
        if math.isnan(py_v) or math.isnan(rs_v):
            if not (math.isnan(py_v) and math.isnan(rs_v)):
                failures.append(f"{label} NaN mismatch: py={py_v} rs={rs_v}")
            return current_max
        d = abs(py_v - rs_v)
        if d >= PARITY_TOL:
            failures.append(f"{label} diff {d:.3e} >= {PARITY_TOL}")
        return max(current_max, d)

    # Compare SequenceVprFilter cases
    for cname in ("vpr_bayesian", "vpr_viterbi", "vpr_blind_recovery"):
        spec = batch_cases[cname]
        py_flt = SequenceVprFilter(
            np.asarray(spec["gallery_ne_m"], dtype=np.float64),
            config=SequenceVprConfig(**spec["config"]),
        )
        rs_outs = rs_results[cname]["outputs"]
        for i, step in enumerate(spec["steps"]):
            sims = np.asarray([_dec_f64(x) for x in step["similarities"]], dtype=np.float64)
            delta = np.asarray([_dec_f64(x) for x in step["delta_ne_odom_m"]], dtype=np.float64)
            vfm = VisualFixMetrics(**step["visual_fix_metrics"]) if "visual_fix_metrics" in step else None
            thr = VisualAliasingThresholds(**step["aliasing_thresholds"]) if "aliasing_thresholds" in step else None
            py_dec = py_flt.step(sims, delta, odom_sigma_m=float(step["odom_sigma_m"]), visual_fix_metrics=vfm, aliasing_thresholds=thr)
            rs_dec = rs_outs[i]
            vpr_steps_checked += 1
            if (
                py_dec.admitted is not rs_dec["admitted"]
                or py_dec.psr_result.admitted is not rs_dec["psr_admitted"]
                or py_dec.final_top1_idx != rs_dec["final_top1_idx"]
                or list(py_dec.reason_codes) != rs_dec["reason_codes"]
                or list(py_dec.viterbi_path_indices) != rs_dec["viterbi_path_indices"]
            ):
                bool_mismatches += 1
            vpr_max_diff = _upd_diff(py_dec.posterior_prob_top1, _dec_f64(rs_dec["posterior_prob_top1"]), vpr_max_diff, f"{cname}[{i}].p_top1")
            vpr_max_diff = _upd_diff(py_dec.posterior_margin_non_adj, _dec_f64(rs_dec["posterior_margin_non_adj"]), vpr_max_diff, f"{cname}[{i}].p_margin")
            vpr_max_diff = _upd_diff(py_dec.psr_result.psr_cal, _dec_f64(rs_dec["psr_cal"]), vpr_max_diff, f"{cname}[{i}].psr_cal")

    # Compare DualBankEsEkfCusum cases
    horiz_ramp_detected = False
    vert_ramp_detected = False
    mp_false_alarms = 0
    for cname, spec in cusum_cases.items():
        py_flt = DualBankEsEkfCusum(cfg=DualBankConfig(**spec["config"]))
        py_flt.reset(
            init_ned_m=spec["reset"]["init_ned_m"],
            init_vel_ne_mps=spec["reset"]["init_vel_ne_mps"],
        )
        rs_outs = rs_results[cname]["outputs"]
        for i, step in enumerate(spec["steps"]):
            g_down = _dec_f64(step["gnss_down_m"]) if "gnss_down_m" in step else None
            py_out = py_flt.step(
                ins_accel_ne_mps2=step["ins_accel_ne_mps2"],
                baro_down_m=float(step["baro_down_m"]),
                gnss_ne_m=step["gnss_ne_m"],
                vis_ne_m=step["vis_ne_m"],
                vis_gate_admitted=bool(step["vis_gate_admitted"]),
                gnss_down_m=g_down,
            )
            rs_out = rs_outs[i]
            cusum_epochs_checked += 1
            if (
                py_out.active_bank != rs_out["active_bank"]
                or py_out.gnss_quarantined is not rs_out["gnss_quarantined"]
                or py_out.spoof_alarm is not rs_out["spoof_alarm"]
                or py_out.vertical_spoof_alarm is not rs_out["vertical_spoof_alarm"]
                or py_out.nominal_streak != rs_out["nominal_streak"]
            ):
                bool_mismatches += 1

            cusum_max_diff = _upd_diff(py_out.max_cusum_stat, _dec_f64(rs_out["max_cusum_stat"]), cusum_max_diff, f"{cname}[{i}].cusum")
            cusum_max_diff = _upd_diff(py_out.vertical_cusum_stat, _dec_f64(rs_out["vertical_cusum_stat"]), cusum_max_diff, f"{cname}[{i}].vcusum")
            for ax in range(2):
                cusum_max_diff = _upd_diff(float(py_out.bank_a_p_ne_m[ax]), _dec_f64(rs_out["bank_a_p_ne_m"][ax]), cusum_max_diff, f"{cname}[{i}].ba{ax}")
                cusum_max_diff = _upd_diff(float(py_out.bank_b_p_ne_m[ax]), _dec_f64(rs_out["bank_b_p_ne_m"][ax]), cusum_max_diff, f"{cname}[{i}].bb{ax}")

            if cname.startswith("mp_clean_") and py_out.spoof_alarm:
                mp_false_alarms += 1
            if cname == "horiz_ramp_2mps_1hz" and i >= 35 and py_out.spoof_alarm:
                horiz_ramp_detected = True
            if cname == "vert_down_ramp_and_nan" and i >= 25 and py_out.vertical_spoof_alarm:
                vert_ramp_detected = True

    if bool_mismatches > 0:
        failures.append(f"boolean/discrete parity mismatches: {bool_mismatches}")
    if mp_false_alarms > 0:
        failures.append(f"multipath clean false alarms: {mp_false_alarms}")
    if not horiz_ramp_detected:
        failures.append("horizontal 2.0 m/s ramp not detected")
    if not vert_ramp_detected:
        failures.append("vertical Down +1.0 m/s ramp not detected")

    max_overall = max(vpr_max_diff, cusum_max_diff)
    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0 and max_overall < PARITY_TOL
    summary = (
        f"SeqVPR({vpr_steps_checked})={vpr_max_diff:.2e}, CUSUM({cusum_epochs_checked})={cusum_max_diff:.2e}, exact_bool=100%"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G4",
        name="Python <-> Pure-Rust Avionics Parity (<1e-6)",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details={
            "nav_integrity_rs_byte_identical": nav_rs_identical,
            "vpr_steps_checked": vpr_steps_checked,
            "vpr_max_abs_diff": vpr_max_diff,
            "cusum_epochs_checked": cusum_epochs_checked,
            "cusum_max_abs_diff": cusum_max_diff,
            "max_abs_diff": max_overall,
            "bool_mismatches": bool_mismatches,
            "multipath_false_alarms": mp_false_alarms,
            "horiz_ramp_detected": horiz_ramp_detected,
            "vert_ramp_detected": vert_ramp_detected,
            "failures": failures,
        },
    )


# ---------------------------------------------------------------------------
# Gate 5 — Clean-Room `openxvs-studio.exe` Binary Hygiene (`REL1..REL4`)
# ---------------------------------------------------------------------------


def run_gate5_studio_binary_hygiene(
    repo_root: Path = REPO_ROOT,
    studio_exe: Path | None = None,
    studio_cargo_lock: Path | None = None,
) -> GateResult:
    t0 = time.perf_counter()
    exe = studio_exe or (repo_root / "apps" / "studio" / "src-tauri" / "target" / "release" / "openxvs-studio.exe")
    lock_path = studio_cargo_lock or (repo_root / "apps" / "studio" / "src-tauri" / "Cargo.lock")
    failures: list[str] = []

    if not exe.is_file():
        return GateResult(
            gate_id="G5",
            name="Clean-Room Studio Binary Hygiene (REL1..REL4)",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=f"missing binary: {exe.name}",
            details={"missing_binary": True, "failures": [f"missing binary: {exe.name}"]},
        )

    raw = exe.read_bytes()
    size_bytes = len(raw)
    size_mib = round(size_bytes / float(1024 * 1024), 2)
    sha256_hex = hashlib.sha256(raw).hexdigest()
    ascii_text = raw.decode("latin-1")
    utf16_text = raw.decode("utf-16-le", errors="ignore")

    leak_hits: dict[str, int] = {}
    for rule_name, pat in BINARY_LEAK_REGEXES:
        cnt = len(pat.findall(ascii_text)) + len(pat.findall(utf16_text))
        if cnt > 0:
            leak_hits[rule_name] = cnt
            failures.append(f"REL1 leak {rule_name}: {cnt} hit(s)")

    pyi_hits = [m.decode("latin-1") for m in PYINSTALLER_MARKERS if m in raw]
    if pyi_hits:
        failures.append(f"REL5 PyInstaller markers found: {pyi_hits}")

    watermark_present = ("NON_FLIGHT_SIM_ONLY" in ascii_text) or ("NON_FLIGHT_SIM_ONLY" in utf16_text)
    if not watermark_present:
        failures.append("REL4 missing NON_FLIGHT_SIM_ONLY watermark in binary")

    crypto_hits: list[str] = []
    cargo_pkg_count = 0
    if lock_path.is_file():
        lock_doc = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        pkgs = lock_doc["package"] if "package" in lock_doc else []
        cargo_pkg_count = len(pkgs)
        for pkg in pkgs:
            name = str(pkg["name"]).lower() if "name" in pkg else ""
            if name in BANNED_5D002_CRYPTO_CRATES:
                crypto_hits.append(name)
        if crypto_hits:
            failures.append(f"REL3 5D002 crypto crates in Cargo.lock: {crypto_hits}")
    else:
        failures.append(f"missing Cargo.lock: {lock_path.name}")

    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    summary = (
        f"{size_mib:.2f} MiB ({sha256_hex[:12]}), leaks=0, 5D002=0/{cargo_pkg_count}, watermark=OK"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G5",
        name="Clean-Room Studio Binary Hygiene (REL1..REL4)",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details={
            "binary_name": exe.name,
            "size_bytes": size_bytes,
            "size_mib": size_mib,
            "sha256": sha256_hex,
            "leak_hits": leak_hits,
            "pyinstaller_hits": pyi_hits,
            "non_flight_watermark_present": watermark_present,
            "cargo_lock_packages": cargo_pkg_count,
            "banned_5d002_crates": crypto_hits,
            "failures": failures,
        },
    )


# ---------------------------------------------------------------------------
# Gate 6 — Bare-Metal ARM Cortex-M7 QEMU Proof Ledger
# ---------------------------------------------------------------------------


def run_gate6_cortex_m7_proof_ledger(
    repo_root: Path = REPO_ROOT,
    m7_proof_path: Path | None = None,
) -> GateResult:
    t0 = time.perf_counter()
    proof_p = m7_proof_path or (repo_root / "outputs" / "formal_tla" / "cortex_m7_emu_exec_proof.json")
    tla_p = repo_root / "outputs" / "formal_tla" / "sn4_tla_evidence.json"
    failures: list[str] = []

    if not proof_p.is_file():
        return GateResult(
            gate_id="G6",
            name="Bare-Metal ARM Cortex-M7 QEMU Proof Ledger",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary=f"missing proof ledger: {proof_p.name}",
            details={"failures": [f"missing proof ledger: {proof_p.name}"]},
        )

    doc = json.loads(proof_p.read_text(encoding="utf-8"))
    scope = doc["scope_flags"] if "scope_flags" in doc else {}
    safety = doc["code_safety_and_compliance"] if "code_safety_and_compliance" in doc else {}
    foot = doc["bench_elf_footprint"] if "bench_elf_footprint" in doc else {}
    exec_r = doc["execution_results"] if "execution_results" in doc else {}

    if doc["schema"] != "openxvs.v52.cortex_m7_emulated_exec_proof/1":
        failures.append(f"unexpected schema: {doc['schema']!r}")
    if scope["physical_stm32_hardware_verified"] is not False:
        failures.append("physical_stm32_hardware_verified MUST be False (honest emulation tag)")
    if scope["emulation_only"] is not True or scope["non_flight_sim_only"] is not True:
        failures.append("emulation_only / non_flight_sim_only scope flags not True")
    if scope["actuator_authority"] is not False:
        failures.append("actuator_authority MUST be False")
    if safety["no_std"] is not True or safety["forbid_unsafe_code_lib"] is not True:
        failures.append("no_std or forbid_unsafe_code_lib != True")
    if safety["zero_gpl_lgpl_agpl_crates"] is not True or safety["zero_host_paths"] is not True:
        failures.append("zero_gpl or zero_host_paths != True")
    if int(foot["static_ram_bytes_data_plus_bss"]) != 0 or int(foot["heap_alloc_symbols"]) != 0:
        failures.append("static RAM or heap alloc symbols != 0")
    if int(exec_r["ticks_executed"]) != 10000 or int(exec_r["ticks_passed"]) != 10000:
        failures.append(f"ticks_executed/passed != 10000: {exec_r['ticks_executed']}/{exec_r['ticks_passed']}")
    if int(exec_r["invariants_checked"]) != 77917 or int(exec_r["invariant_violations"]) != 0:
        failures.append(f"invariants mismatch: {exec_r['invariants_checked']} viol={exec_r['invariant_violations']}")
    if int(exec_r["faults"]) != 0 or str(exec_r["verdict"]) != "PASS":
        failures.append(f"faults={exec_r['faults']} verdict={exec_r['verdict']!r}")

    kani_total: int | None = None
    apalache_ok: bool | None = None
    if tla_p.is_file():
        tla_doc = json.loads(tla_p.read_text(encoding="utf-8"))
        kani_total = int(tla_doc["multiship_coord_kani"]["openxvs_total_kani_proofs"])
        apalache_ok = bool(tla_doc["apalache_symbolic_check"]["all_passed"])
        if tla_doc["status"] != "PASS" or kani_total != 26 or not apalache_ok:
            failures.append(f"sn4_tla_evidence check failed: kani={kani_total} apalache={apalache_ok}")

    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    summary = (
        f"10,000/10,000 ticks, 77,917 inv (0 viol), 0 B RAM, {int(foot['total_flash_bytes'])} B Flash, hw_verified=false"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G6",
        name="Bare-Metal ARM Cortex-M7 QEMU Proof Ledger",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details={
            "ticks_executed": int(exec_r["ticks_executed"]),
            "ticks_passed": int(exec_r["ticks_passed"]),
            "invariants_checked": int(exec_r["invariants_checked"]),
            "invariant_violations": int(exec_r["invariant_violations"]),
            "total_flash_bytes": int(foot["total_flash_bytes"]),
            "static_ram_bytes": int(foot["static_ram_bytes_data_plus_bss"]),
            "physical_stm32_hardware_verified": scope["physical_stm32_hardware_verified"],
            "checksum_fnv1a64": exec_r["deterministic_checksum_fnv1a64"],
            "kani_proofs_total": kani_total,
            "apalache_all_passed": apalache_ok,
            "failures": failures,
        },
    )


# ---------------------------------------------------------------------------
# Optional Gate 7 (`--live-sitl`) — Live WSL2 `ArduPlane V4.7.3` Mission SIL
# ---------------------------------------------------------------------------


def run_gate7_live_sitl(repo_root: Path = REPO_ROOT) -> GateResult:
    t0 = time.perf_counter()
    sil_script = repo_root / "scripts" / "xvs" / "run_full_mission_sil_gate.py"
    report_path = repo_root / "outputs" / "sitl_proof" / "faz2_full_mission" / "full_mission_sil_report.json"
    failures: list[str] = []
    details: dict[str, Any] = {}

    if not sil_script.is_file():
        return GateResult(
            gate_id="G7",
            name="Live WSL2 ArduPlane V4.7.3 Full-Mission SIL",
            passed=False,
            elapsed_s=round(time.perf_counter() - t0, 3),
            summary="missing scripts/xvs/run_full_mission_sil_gate.py",
            details={"failures": ["missing run_full_mission_sil_gate.py"]},
        )

    proc = subprocess.run(
        [sys.executable, str(sil_script)],
        cwd=str(repo_root),
        env=_thermal_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    details["returncode"] = proc.returncode
    if proc.returncode != 0:
        failures.append(f"live SITL exited rc={proc.returncode}")
    if not report_path.is_file():
        failures.append("missing full_mission_sil_report.json after SITL run")
    else:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        details["all_pass"] = rep["all_pass"]
        details["pymavlink_imported"] = rep["pymavlink_imported"]
        details["simulator_backend"] = rep["simulator_backend"]
        if rep["all_pass"] is not True:
            failures.append("live SITL report all_pass != True")
        if rep["pymavlink_imported"] is not False:
            failures.append("live SITL report pymavlink_imported != False")

    details["failures"] = failures
    elapsed = time.perf_counter() - t0
    passed = len(failures) == 0
    summary = (
        "7/7 live ArduPlane 4.7.3 mission phases PASS (pymavlink=false)"
        if passed
        else "; ".join(failures[:3])
    )
    return GateResult(
        gate_id="G7",
        name="Live WSL2 ArduPlane V4.7.3 Full-Mission SIL",
        passed=passed,
        elapsed_s=round(elapsed, 3),
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# Runner & Tabular Scorecard Formatter
# ---------------------------------------------------------------------------


def run_all_gates(
    *,
    repo_root: Path = REPO_ROOT,
    live_sitl: bool = False,
    studio_exe: Path | None = None,
    edge_exe: Path | None = None,
    tape_path: Path | None = None,
    m7_proof_path: Path | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    results: list[GateResult] = [
        run_gate1_compliance_and_license(repo_root=repo_root),
        run_gate2_edge_self_check(repo_root=repo_root, edge_exe=edge_exe),
        run_gate3_flight_recorder_replay(repo_root=repo_root, edge_exe=edge_exe, tape_path=tape_path),
        run_gate4_python_rust_parity(repo_root=repo_root, edge_exe=edge_exe),
        run_gate5_studio_binary_hygiene(repo_root=repo_root, studio_exe=studio_exe),
        run_gate6_cortex_m7_proof_ledger(repo_root=repo_root, m7_proof_path=m7_proof_path),
    ]
    if live_sitl:
        results.append(run_gate7_live_sitl(repo_root=repo_root))

    total_elapsed = round(time.perf_counter() - t_start, 3)
    passed_count = sum(1 for r in results if r.passed)
    overall_pass = passed_count == len(results)

    git_head = "unknown"
    if (repo_root / ".git").exists():
        gp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if gp.returncode == 0 and gp.stdout.strip():
            git_head = gp.stdout.strip()

    return {
        "schema": "openxvs.v110.oem_verification_gate/1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "non_flight_sim_only": True,
        "actuator_authority": False,
        "pymavlink_imported": GPL_MAVLINK_MODULE in sys.modules,
        "live_sitl_requested": live_sitl,
        "overall_pass": overall_pass,
        "passed_count": passed_count,
        "total_count": len(results),
        "elapsed_s": total_elapsed,
        "gates": [asdict(r) for r in results],
    }


def format_scorecard_table(report: dict[str, Any]) -> str:
    sep = "=" * 112
    sub = "-" * 112
    lines = [
        sep,
        f"OPENXVS v1.1.0 — SINGLE-COMMAND VERIFICATION GATE (HEAD {report['git_head']})",
        "NON_FLIGHT_SIM_ONLY=true | ACTUATOR_AUTHORITY=false | ZERO-GPL | STRICTLY NED (0 INTERNAL ENU)",
        sep,
        f"{'Gate':<4} | {'Verification Check':<46} | {'Status':<6} | {'Time':>7} | Key Evidence",
        sub,
    ]
    for g in report["gates"]:
        st = "PASS" if g["passed"] else "FAIL"
        lines.append(
            f"{g['gate_id']:<4} | {g['name']:<46} | {st:<6} | {g['elapsed_s']:>6.2f}s | {g['summary']}"
        )
    lines.append(sep)
    verdict = "PASS" if report["overall_pass"] else "FAIL"
    lines.append(
        f"OVERALL VERDICT: {verdict} ({report['passed_count']}/{report['total_count']} gates passed in {report['elapsed_s']:.2f}s)"
    )
    lines.append(sep)
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-command reproducible OEM / Principal-Engineer verification gate for OpenXVS v1.1.0"
    )
    p.add_argument(
        "--live-sitl",
        action="store_true",
        help="Also run Gate 7 LIVE on WSL2 ArduPlane V4.7.3 via scripts/xvs/run_full_mission_sil_gate.py",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write structured JSON verification report",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_all_gates(repo_root=REPO_ROOT, live_sitl=bool(args.live_sitl))
    print(format_scorecard_table(report))

    if args.json_out is not None:
        out_p: Path = args.json_out
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_name(f"{out_p.name}.tmp")
        tmp_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_p, out_p)

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
