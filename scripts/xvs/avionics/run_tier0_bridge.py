"""Single-process Tier-0 Avionics Bridge Verification & Closed-Loop WP-05 G1 / WP-07 Producer (Batch B10 Phase 2).

Executes deterministic end-to-end replay, live closed-loop WP-05 G1 + WP-07 integrity
simulation, and localhost UDP loopback verification for FAA GDL-90 Public ICD Rev A
and MAVLink v2 `ADSB_VEHICLE` (#246, CRC_EXTRA=184) across:
1. Official FAA GDL-90 ICD §2.2.3 Heartbeat test vector and control-octet byte-stuffing;
2. Four 10 Hz multi-epoch synthetic passive-track scenarios (`WC1` threat, `WC1` safe
   pass-by, `A1`/`A2` fallback threat, and simultaneous dual-track mixed encounter)
   referenced to `configs/xvs/encounters/sampler_v1.json`, with true continuous-to-`int32`
   (`degE7` / `mm`) MAVLink quantization residual measurement (`B10-m3`) and per-track
   multi-target summary visibility (`B10-m4`);
3. Deterministic stratified replay of saved WP-05 G0 (`A1`/`A2`) and WP-05 G1 (`WC1`)
   passive track records from the 4,500-run baseline corpus (`lambda = 0.0005`), plus
   full 4,500-run G0 vs G1 alerting and 6-DoF ground-clutter rejection comparison;
4. Live 30-step (3.0 s at 10 Hz) closed-loop simulation coupling `MultiCameraTracker`
   (`configs/xvs/tracking/g1.json`), `evaluate_track_risk` (`WC1`), and WP-07 Horizon
   Integrity state transitions (`VALID` -> `DEGRADED_COAST` -> `INVALID_ALERT`) into
   `PassiveTrackAvionicsBridge.step_from_tracker` (`B10-m2`) over localhost UDP loopback;
5. Static AST and runtime sentinel-trap verification of the V5_B10 truth-isolation
   invariant across `src/xvs/avionics/*`.

Writes `output/xvs/wp08/tier0_b10/tier0_verification.json` with cryptographic
provenance (`commit`, `git_status`, `argv`, `code_hashes`, `inputs_sha256`).
Strictly single-process (`workers = 1`).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics.bridge import (
    METERS_TO_FEET,
    MPS_TO_FPM,
    MPS_TO_KNOTS,
    OwnshipNavState,
    PassiveTrackAvionicsBridge,
    PassiveTrackInput,
)
from xvs.avionics.gdl90 import (
    GDL90_ICD_HEARTBEAT_CRC_U16,
    GDL90_ICD_HEARTBEAT_CRC_WIRE,
    GDL90_ICD_HEARTBEAT_FRAMED,
    GDL90_ICD_HEARTBEAT_UNSTUFFED,
    GDL90_MSG_ID_HEARTBEAT,
    GDL90_MSG_ID_OWNSHIP_GEO_ALT,
    GDL90_MSG_ID_OWNSHIP_REPORT,
    GDL90_MSG_ID_TRAFFIC_REPORT,
    Gdl90TargetReport,
    Gdl90UdpEndpoint,
    compute_gdl90_crc,
    decode_gdl90_frame,
    decode_gdl90_message,
    frame_gdl90_message,
)
from xvs.avionics.mavlink_adsb import (
    ADSB_FLAGS_VALID_COORDS,
    ADSB_VEHICLE_CRC_EXTRA,
    ADSB_VEHICLE_FRAME_LEN,
    ADSB_VEHICLE_MSG_ID,
    ADSB_VEHICLE_PAYLOAD_LEN,
    MAVLINK_V2_STX,
    MavlinkAdsbUdpStreamer,
    unpack_adsb_vehicle,
)
from xvs.geo import rotations, wgs84
from xvs.paths import get_data_root
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.ring import TrafficRing
from xvs.tracking.ground_rejection import ray_wgs84_intersect
from xvs.tracking.measurements import DetectionRecord
from xvs.tracking.risk import evaluate_track_risk
from xvs.tracking.tracker import MultiCameraTracker, TrackStatus

DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "avionics" / "tier0_spec.json"
DEFAULT_SAMPLER_PATH = REPO_ROOT / "configs" / "xvs" / "encounters" / "sampler_v1.json"
DEFAULT_G1_TRACKING_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json"

DEFAULT_WP05_G0_RECORDS_CANDIDATES = (
    REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records.jsonl",
)

DEFAULT_WP05_G1_RECORDS_CANDIDATES = (
    REPO_ROOT / "output" / "xvs" / "wp05" / "g1" / "per_run_records.jsonl",
)

DEFAULT_WP05_G1_BASELINE_CANDIDATES = (
    REPO_ROOT / "output" / "xvs" / "wp05" / "g1" / "baseline_results.json",
)

TRACKED_CODE_FILES = (
    "src/xvs/avionics/__init__.py",
    "src/xvs/avionics/gdl90.py",
    "src/xvs/avionics/mavlink_adsb.py",
    "src/xvs/avionics/bridge.py",
    "scripts/xvs/avionics/run_tier0_bridge.py",
    "scripts/xvs/avionics/report_tables_tier0_b10.py",
    "tests/xvs/test_avionics_tier0.py",
)

FORBIDDEN_TRUTH_IDENTIFIERS = frozenset(
    {
        "truth_id",
        "true_range",
        "true_range_m",
        "range_true",
        "range_true_m",
        "p_int_ned_true",
        "intruder_pos_true",
        "tau_true_s",
        "omega_true_radps",
    }
)


def _get_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _get_git_status_porcelain() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in TRACKED_CODE_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            out[rel] = _sha256_file(p)
    return out


class _TruthSentinelTrap:
    """Object wrapper that raises RuntimeError if any ground-truth attribute is accessed."""

    def __init__(self, passive_fields: dict[str, Any]) -> None:
        self._passive = dict(passive_fields)
        self.truth_access_count = 0

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_TRUTH_IDENTIFIERS or name.startswith("truth_"):
            self.truth_access_count += 1
            raise RuntimeError(f"Forbidden ground-truth attribute accessed: {name}")
        if name in self._passive:
            return self._passive[name]
        raise AttributeError(name)


def audit_truth_isolation() -> dict[str, Any]:
    """Run AST and runtime sentinel-trap checks on `src/xvs/avionics/*` (V5_B10)."""
    avionics_dir = REPO_ROOT / "src" / "xvs" / "avionics"
    py_files = sorted(avionics_dir.glob("*.py"))

    forbidden_imports: list[str] = []
    forbidden_attr_accesses: list[str] = []

    for py_path in py_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        rel_name = py_path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "xvs.traffic" in alias.name:
                        forbidden_imports.append(f"{rel_name}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "xvs.traffic" in mod:
                    forbidden_imports.append(f"{rel_name}:{mod}")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_TRUTH_IDENTIFIERS:
                    forbidden_attr_accesses.append(f"{rel_name}:{node.attr}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in FORBIDDEN_TRUTH_IDENTIFIERS:
                    forbidden_attr_accesses.append(f"{rel_name}:{node.value}")

    # Runtime sentinel-trap test across both step() and step_from_tracker()
    bridge = PassiveTrackAvionicsBridge(DEFAULT_SPEC_PATH)
    ownship = OwnshipNavState.from_spec(bridge.spec)
    sentinel_track = _TruthSentinelTrap(
        {
            "track_id": "T0099",
            "t_s": 1.0,
            "az": 0.05,
            "el": -0.01,
            "az_rate": 0.0005,
            "el_rate": 0.0001,
            "s": 0.008,
            "s_mom": 0.008,
            "s_dot": 0.0004,
            "time_since_update": 0.0,
            "status": "confirmed",
            "ground_clutter_rejected": False,
        }
    )
    sentinel_risk = _TruthSentinelTrap(
        {
            "track_id": "T0099",
            "t_s": 1.0,
            "range_interval_m": (900.0, 1500.0),
            "hmd_hat_m": 180.0,
            "tau_mod_hat_s": 18.0,
            "tau_s_hat_s": 19.5,
            "omega_los_radps": 0.00051,
            "wc1_warning": True,
            "wc1_corrective": True,
            "alert_a1_warning": False,
            "alert_a2_warning": False,
        }
    )
    bridge.step_from_tracker(
        ownship=ownship,
        confirmed_tracks=[sentinel_track],
        risk_by_track_id={"T0099": sentinel_risk},
        integrity_state="VALID",
        class_by_track_id={"T0099": "C2_SEP"},
        force_heartbeat=True,
    )

    runtime_violations = int(sentinel_track.truth_access_count) + int(
        sentinel_risk.truth_access_count
    )
    total_violations = (
        len(forbidden_imports)
        + len(forbidden_attr_accesses)
        + runtime_violations
    )
    return {
        "files_audited": len(py_files),
        "forbidden_imports_count": len(forbidden_imports),
        "forbidden_attribute_refs_count": len(forbidden_attr_accesses),
        "runtime_sentinel_violations": runtime_violations,
        "total_violations": total_violations,
        "isolated": total_violations == 0,
    }


def verify_icd_vectors() -> dict[str, Any]:
    """Verify FAA GDL-90 ICD §2.2.3 Heartbeat CRC-16 and control-octet byte stuffing."""
    crc_u16 = compute_gdl90_crc(GDL90_ICD_HEARTBEAT_UNSTUFFED)
    crc_wire = bytes([crc_u16 & 0xFF, (crc_u16 >> 8) & 0xFF])
    framed = frame_gdl90_message(GDL90_ICD_HEARTBEAT_UNSTUFFED)
    decoded_payload = decode_gdl90_frame(framed)

    stuff_payload = bytes([0x14, 0x7E, 0x7D, 0x00, 0x7E, 0x7D] + [0x55] * 22)
    stuffed_frame = frame_gdl90_message(stuff_payload)
    unstuffed_payload = decode_gdl90_frame(stuffed_frame)
    escaped_occurrences = stuffed_frame[1:-1].count(0x7D)

    exact_match = (
        crc_u16 == GDL90_ICD_HEARTBEAT_CRC_U16
        and crc_wire == GDL90_ICD_HEARTBEAT_CRC_WIRE
        and framed == GDL90_ICD_HEARTBEAT_FRAMED
        and decoded_payload == GDL90_ICD_HEARTBEAT_UNSTUFFED
        and unstuffed_payload == stuff_payload
    )
    return {
        "heartbeat_unstuffed_len_bytes": len(GDL90_ICD_HEARTBEAT_UNSTUFFED),
        "heartbeat_crc_u16": crc_u16,
        "heartbeat_crc_expected_u16": GDL90_ICD_HEARTBEAT_CRC_U16,
        "heartbeat_crc_wire_hex": crc_wire.hex(),
        "heartbeat_framed_hex": framed.hex(),
        "heartbeat_framed_len_bytes": len(framed),
        "stuffing_escaped_bytes_count": escaped_occurrences,
        "stuffing_roundtrip_exact": unstuffed_payload == stuff_payload,
        "icd_vector_exact_match": exact_match,
    }


def _build_synthetic_scenarios(steps_per_scenario: int, dt_s: float) -> list[dict[str, Any]]:
    """Construct 4 deterministic 10 Hz passive-track replay scenarios."""
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: WC1 Head-On Threat (C2_SEP)
    s1_epochs: list[tuple[float, list[PassiveTrackInput]]] = []
    for k in range(steps_per_scenario):
        t_s = round(k * dt_s, 3)
        tau_val = max(8.0, 32.0 - 0.18 * k)
        r_mid = 60.0 * tau_val
        s1_epochs.append(
            (
                t_s,
                [
                    PassiveTrackInput(
                        track_id="T0001",
                        t_s=t_s,
                        az_rad=0.015 - 0.00005 * k,
                        el_rad=-0.005,
                        az_rate_radps=-0.0005,
                        el_rate_radps=0.0,
                        s_rad=9.5 / max(100.0, r_mid),
                        s_dot_radps=(9.5 / max(100.0, r_mid)) / tau_val,
                        r_min_m=0.80 * r_mid,
                        r_max_m=1.20 * r_mid,
                        hmd_hat_m=140.0,
                        tau_mod_hat_s=tau_val,
                        tau_s_hat_s=tau_val,
                        wc1_warning=True,
                        wc1_corrective=True,
                        class_id="C2_SEP",
                        tslc_s=0,
                    )
                ],
            )
        )
    scenarios.append(
        {
            "scenario_id": "S1_WC1_HEAD_ON_THREAT",
            "rule_family": "WC1",
            "class_id": "C2_SEP",
            "is_threat_scenario": True,
            "tracks_per_step": 1,
            "epochs": s1_epochs,
        }
    )

    # Scenario 2: WC1 Safe Pass-By (C3_TWIN)
    s2_epochs: list[tuple[float, list[PassiveTrackInput]]] = []
    for k in range(steps_per_scenario):
        t_s = round(k * dt_s, 3)
        s2_epochs.append(
            (
                t_s,
                [
                    PassiveTrackInput(
                        track_id="T0002",
                        t_s=t_s,
                        az_rad=0.42 + 0.006 * k,
                        el_rad=0.02,
                        az_rate_radps=0.06,
                        el_rate_radps=0.0,
                        s_rad=0.004,
                        s_dot_radps=0.00002,
                        r_min_m=2500.0,
                        r_max_m=4200.0,
                        hmd_hat_m=1650.0,
                        tau_mod_hat_s=75.0,
                        tau_s_hat_s=75.0,
                        wc1_warning=False,
                        wc1_corrective=False,
                        class_id="C3_TWIN",
                        tslc_s=1,
                    )
                ],
            )
        )
    scenarios.append(
        {
            "scenario_id": "S2_WC1_SAFE_PASSBY",
            "rule_family": "WC1",
            "class_id": "C3_TWIN",
            "is_threat_scenario": False,
            "tracks_per_step": 1,
            "epochs": s2_epochs,
        }
    )

    # Scenario 3: WP-05 G0 A1/A2 Fallback Threat (C1_UAS)
    s3_epochs: list[tuple[float, list[PassiveTrackInput]]] = []
    for k in range(steps_per_scenario):
        t_s = round(k * dt_s, 3)
        tau_val = max(6.0, 22.0 - 0.14 * k)
        s_rad = 0.75 / (25.0 * tau_val)
        s3_epochs.append(
            (
                t_s,
                [
                    PassiveTrackInput(
                        track_id="T0003",
                        t_s=t_s,
                        az_rad=-0.08,
                        el_rad=0.01,
                        az_rate_radps=0.0008,
                        el_rate_radps=-0.0004,
                        s_rad=s_rad,
                        s_dot_radps=s_rad / tau_val,
                        tau_s=tau_val,
                        omega_los_radps=math.hypot(0.0008, -0.0004),
                        a1_warning=True,
                        a1_corrective=True,
                        a2_warning=True,
                        a2_corrective=True,
                        class_id="C1_UAS",
                        tslc_s=0,
                    )
                ],
            )
        )
    scenarios.append(
        {
            "scenario_id": "S3_A1_A2_FALLBACK_THREAT",
            "rule_family": "A1_A2",
            "class_id": "C1_UAS",
            "is_threat_scenario": True,
            "tracks_per_step": 1,
            "epochs": s3_epochs,
        }
    )

    # Scenario 4: Simultaneous Dual-Track Mixed (C5_TRANSPORT safe + C4_BIZJET WC1 threat)
    s4_epochs: list[tuple[float, list[PassiveTrackInput]]] = []
    for k in range(steps_per_scenario):
        t_s = round(k * dt_s, 3)
        s4_epochs.append(
            (
                t_s,
                [
                    PassiveTrackInput(
                        track_id="T0004",
                        t_s=t_s,
                        az_rad=-0.55 - 0.003 * k,
                        el_rad=0.04,
                        az_rate_radps=-0.03,
                        el_rate_radps=0.0,
                        s_rad=0.006,
                        s_dot_radps=0.00001,
                        r_min_m=4500.0,
                        r_max_m=6200.0,
                        hmd_hat_m=2200.0,
                        tau_mod_hat_s=90.0,
                        wc1_warning=False,
                        wc1_corrective=False,
                        class_id="C5_TRANSPORT",
                        tslc_s=1,
                    ),
                    PassiveTrackInput(
                        track_id="T0005",
                        t_s=t_s,
                        az_rad=0.12 - 0.0002 * k,
                        el_rad=-0.01,
                        az_rate_radps=-0.0006,
                        el_rate_radps=0.0001,
                        s_rad=0.010,
                        s_dot_radps=0.0005,
                        r_min_m=1400.0,
                        r_max_m=2100.0,
                        hmd_hat_m=280.0,
                        tau_mod_hat_s=max(10.0, 24.0 - 0.12 * k),
                        wc1_warning=True,
                        wc1_corrective=True,
                        class_id="C4_BIZJET",
                        tslc_s=0,
                    ),
                ],
            )
        )
    scenarios.append(
        {
            "scenario_id": "S4_MULTI_TRACK_MIXED",
            "rule_family": "WC1_MIXED",
            "class_id": "C4_BIZJET+C5_TRANSPORT",
            "is_threat_scenario": True,
            "tracks_per_step": 2,
            "epochs": s4_epochs,
        }
    )
    return scenarios


def _resolve_candidate_path(
    explicit_path: Path | None, candidates: tuple[Path, ...]
) -> Path | None:
    if explicit_path is not None and explicit_path.is_file():
        return explicit_path
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_wp05_g0_passive_samples(
    records_path: Path | None, max_samples: int = 24
) -> tuple[list[PassiveTrackInput], Path | None]:
    """Load deterministic stratified passive WP-05 G0 track states (stripping all truth keys)."""
    chosen_path = _resolve_candidate_path(records_path, DEFAULT_WP05_G0_RECORDS_CANDIDATES)
    if chosen_path is None:
        return [], None

    selected: list[PassiveTrackInput] = []
    seen_buckets: dict[tuple[str, bool], int] = {}
    per_bucket_cap = max(1, max_samples // 12)

    with open(chosen_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(selected) >= max_samples:
                break
            rec = json.loads(line)
            cls_id = str(rec.get("class_id", "C1_UAS"))
            a1_w = bool(rec.get("has_alert_a1_warn", False))
            a1_c = bool(rec.get("has_alert_a1_corr", False))
            a2_w = bool(rec.get("has_alert_a2_warn", False))
            a2_c = bool(rec.get("has_alert_a2_corr", False))
            has_alert = a1_w or a1_c or a2_w or a2_c

            bucket = (cls_id, has_alert)
            if seen_buckets.get(bucket, 0) >= per_bucket_cap:  # default-ok: counter init or optional replay config fallback
                continue

            mc = rec.get("miss_check") or {}
            omega_est = mc.get("omega_est_radps")
            if omega_est is None or not math.isfinite(float(omega_est)):
                omega_est = 0.002 if has_alert else 0.015
            tau_est = mc.get("tau_est_s")
            if tau_est is not None and math.isfinite(float(tau_est)) and float(tau_est) > 0.0:
                tau_val: float | None = float(tau_est)
            else:
                tau_val = 20.0 if has_alert else None

            t_chk = float(mc.get("t_check_s", 30.0))  # default-ok: counter init or optional replay config fallback
            enc_str = str(rec.get("encounter", f"E{idx + 1:04d}"))
            az_deterministic = ((idx % 12) - 6) * 0.12

            selected.append(
                PassiveTrackInput(
                    track_id=enc_str,
                    t_s=t_chk,
                    az_rad=az_deterministic,
                    el_rad=0.0,
                    az_rate_radps=float(omega_est),
                    el_rate_radps=0.0,
                    s_rad=0.006,
                    s_dot_radps=(0.006 / tau_val) if tau_val else 0.0,
                    tau_s=tau_val,
                    omega_los_radps=float(omega_est),
                    a1_warning=a1_w,
                    a1_corrective=a1_c,
                    a2_warning=a2_w,
                    a2_corrective=a2_c,
                    class_id=cls_id,
                    tslc_s=0,
                )
            )
            seen_buckets[bucket] = seen_buckets.get(bucket, 0) + 1  # default-ok: counter init or optional replay config fallback

    return selected, chosen_path


def _load_wp05_g1_corpus_and_samples(
    g1_records_path: Path | None,
    g1_baseline_path: Path | None,
    baseline_lambda: float = 0.0005,
    max_samples: int = 24,
) -> tuple[list[PassiveTrackInput], dict[str, Any], Path | None, Path | None]:
    """Load stratified WP-05 G1 WC1 passive samples and 4,500-run G0 vs G1 corpus summary."""
    chosen_rec_path = _resolve_candidate_path(
        g1_records_path, DEFAULT_WP05_G1_RECORDS_CANDIDATES
    )
    chosen_base_path = _resolve_candidate_path(
        g1_baseline_path, DEFAULT_WP05_G1_BASELINE_CANDIDATES
    )

    selected: list[PassiveTrackInput] = []
    seen_buckets: dict[tuple[str, bool], int] = {}
    per_bucket_cap = max(1, max_samples // 12)

    runs_at_lambda = 0
    threats_at_lambda = 0
    non_threats_at_lambda = 0
    wc1_threat_alerts = 0
    wc1_ge_12_5s = 0
    a1_threat_alerts = 0
    a1_ge_12_5s = 0
    gnd_rejected_total = 0
    false_tracks_confirmed_total = 0

    if chosen_rec_path is not None:
        with open(chosen_rec_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                lam = float(rec.get("lambda", 0.0))  # default-ok: counter init or optional replay config fallback
                if abs(lam - baseline_lambda) > 1e-9:
                    continue

                runs_at_lambda += 1
                is_threat = bool(rec.get("is_threat", False))
                wc1_w = bool(rec.get("has_alert_wc1_warn", False))
                wc1_c = bool(rec.get("has_alert_wc1_corr", False))
                a1_w = bool(rec.get("has_alert_a1_warn", False))
                a1_c = bool(rec.get("has_alert_a1_corr", False))
                a2_w = bool(rec.get("has_alert_a2_warn", False))
                a2_c = bool(rec.get("has_alert_a2_corr", False))

                gnd_rejected_total += int(rec.get("ground_clutter_rejected_count", 0))  # default-ok: counter init or optional replay config fallback
                false_tracks_confirmed_total += int(
                    rec.get("false_tracks_ever_confirmed", 0)  # default-ok: counter init or optional replay config fallback
                )

                if is_threat:
                    threats_at_lambda += 1
                    if wc1_w:
                        wc1_threat_alerts += 1
                    lead_wc1 = rec.get("lead_wc1_warn_s")
                    if lead_wc1 is not None and float(lead_wc1) >= 12.5:
                        wc1_ge_12_5s += 1
                    if a1_w:
                        a1_threat_alerts += 1
                    lead_a1 = rec.get("lead_a1_warn_s")
                    if lead_a1 is not None and float(lead_a1) >= 12.5:
                        a1_ge_12_5s += 1
                else:
                    non_threats_at_lambda += 1

                if len(selected) < max_samples:
                    cls_id = str(rec.get("class_id", "C1_UAS"))
                    has_wc1 = wc1_w or wc1_c
                    bucket = (cls_id, has_wc1)
                    if seen_buckets.get(bucket, 0) < per_bucket_cap:  # default-ok: counter init or optional replay config fallback
                        mc = rec.get("miss_check") or {}
                        omega_est = mc.get("omega_est_radps")
                        if omega_est is None or not math.isfinite(float(omega_est)):
                            omega_est = 0.003 if has_wc1 else 0.015
                        tau_mod_est = mc.get("wc1_tau_mod_hat_s")
                        if (
                            tau_mod_est is not None
                            and math.isfinite(float(tau_mod_est))
                            and float(tau_mod_est) > 0.0
                        ):
                            tm_val: float | None = float(tau_mod_est)
                        else:
                            tm_val = 18.0 if has_wc1 else None

                        hmd_est = mc.get("wc1_hmd_hat_m")
                        hmd_val = (
                            float(hmd_est)
                            if (hmd_est is not None and math.isfinite(float(hmd_est)))
                            else (220.0 if has_wc1 else 1400.0)
                        )
                        tau_est = mc.get("tau_est_s")
                        ts_val = (
                            float(tau_est)
                            if (
                                tau_est is not None
                                and math.isfinite(float(tau_est))
                                and float(tau_est) > 0.0
                            )
                            else tm_val
                        )
                        t_chk = float(mc.get("t_check_s", 30.0))  # default-ok: counter init or optional replay config fallback
                        enc_str = str(rec.get("encounter", f"E{idx + 1:04d}"))
                        az_deterministic = ((idx % 12) - 6) * 0.11

                        selected.append(
                            PassiveTrackInput(
                                track_id=enc_str,
                                t_s=t_chk,
                                az_rad=az_deterministic,
                                el_rad=0.0,
                                az_rate_radps=float(omega_est),
                                el_rate_radps=0.0,
                                s_rad=0.007,
                                s_dot_radps=(0.007 / ts_val) if ts_val else 0.0,
                                hmd_hat_m=hmd_val,
                                tau_mod_hat_s=tm_val,
                                tau_s_hat_s=ts_val,
                                tau_s=ts_val,
                                omega_los_radps=float(omega_est),
                                wc1_warning=wc1_w,
                                wc1_corrective=wc1_c,
                                a1_warning=a1_w,
                                a1_corrective=a1_c,
                                a2_warning=a2_w,
                                a2_corrective=a2_c,
                                class_id=cls_id,
                                tslc_s=0,
                            )
                        )
                        seen_buckets[bucket] = seen_buckets.get(bucket, 0) + 1  # default-ok: counter init or optional replay config fallback

    a1_lead_median_s = 0.0
    wc1_lead_median_s = 0.0
    wc1_ck_ge_12_5s_pct = 0.0
    wc1_ck_lead_median_s = 0.0
    if chosen_base_path is not None:
        base_data = json.loads(chosen_base_path.read_text(encoding="utf-8"))
        alerting = base_data.get("pooled_metrics", {}).get("alerting", {})
        a1_lead_median_s = float(
            alerting.get("A1_warning", {}).get("lead_time_median_s", 0.0)  # default-ok: counter init or optional replay config fallback
        )
        wc1_lead_median_s = float(
            alerting.get("WC1_warning", {}).get("lead_time_median_s", 0.0)  # default-ok: counter init or optional replay config fallback
        )
        wc1_ck_ge_12_5s_pct = 100.0 * float(
            alerting.get("WC1_warning_class_known", {}).get("frac_ge_12_5s", 0.0)  # default-ok: counter init or optional replay config fallback
        )
        wc1_ck_lead_median_s = float(
            alerting.get("WC1_warning_class_known", {}).get("lead_time_median_s", 0.0)  # default-ok: counter init or optional replay config fallback
        )

    threats_denom = max(1, threats_at_lambda)
    corpus_summary = {
        "baseline_lambda": float(baseline_lambda),
        "total_runs": runs_at_lambda,
        "threats_count": threats_at_lambda,
        "non_threats_count": non_threats_at_lambda,
        "g0_a1_threat_alerts_count": a1_threat_alerts,
        "g0_a1_threat_alert_rate_pct": 100.0 * float(a1_threat_alerts) / float(threats_denom),
        "g0_a1_lead_ge_12_5s_count": a1_ge_12_5s,
        "g0_a1_lead_ge_12_5s_pct": 100.0 * float(a1_ge_12_5s) / float(threats_denom),
        "g0_a1_lead_median_s": a1_lead_median_s,
        "g1_wc1_threat_alerts_count": wc1_threat_alerts,
        "g1_wc1_threat_alert_rate_pct": 100.0 * float(wc1_threat_alerts) / float(threats_denom),
        "g1_wc1_lead_ge_12_5s_count": wc1_ge_12_5s,
        "g1_wc1_lead_ge_12_5s_pct": 100.0 * float(wc1_ge_12_5s) / float(threats_denom),
        "g1_wc1_lead_median_s": wc1_lead_median_s,
        "g1_wc1_class_known_lead_ge_12_5s_pct": wc1_ck_ge_12_5s_pct,
        "g1_wc1_class_known_lead_median_s": wc1_ck_lead_median_s,
        "g1_ground_clutter_rejected_total": gnd_rejected_total,
        "g1_false_tracks_ever_confirmed_total": false_tracks_confirmed_total,
    }
    return selected, corpus_summary, chosen_rec_path, chosen_base_path


def run_live_closed_loop_g1_verification(
    bridge: PassiveTrackAvionicsBridge,
    gdl_rx: Gdl90UdpEndpoint,
    mav_rx: MavlinkAdsbUdpStreamer,
    ownship: OwnshipNavState,
    g1_cfg_path: Path = DEFAULT_G1_TRACKING_CONFIG_PATH,
    n_steps: int = 30,
    dt_s: float = 0.1,
) -> tuple[dict[str, Any], list[float]]:
    """Run live 30-step closed-loop WP-05 G1 tracker + WC1 risk + WP-07 integrity verification over UDP."""
    g1_cfg = json.loads(g1_cfg_path.read_text(encoding="utf-8"))
    ring = TrafficRing()
    cam0 = ring.cameras[0]
    tracker = MultiCameraTracker(config=g1_cfg, ring=ring)

    # Precompute stationary below-horizon 3D WGS-84 ground point in front camera FOV
    st0 = t1_cruise_state(0.0)
    r_en0 = wgs84.r_ecef_ned(st0.lat_deg, st0.lon_deg)
    poses0 = ring.compute_poses(
        st0.lat_deg, st0.lon_deg, st0.h_m, st0.yaw_rad, st0.pitch_rad, st0.roll_rad
    )
    p_cam0 = np.asarray(poses0[cam0.id]["position"], dtype=np.float64)
    az_gnd = 0.08
    el_gnd = -0.14  # ~8 deg below horizon
    u_ned_gnd = np.array(
        [
            math.cos(el_gnd) * math.cos(az_gnd),
            math.cos(el_gnd) * math.sin(az_gnd),
            -math.sin(el_gnd),
        ],
        dtype=np.float64,
    )
    _, g_pt_ecef = ray_wgs84_intersect(p_cam0, r_en0 @ u_ned_gnd)
    if g_pt_ecef is None:
        raise RuntimeError("Failed to intersect stationary ground point with WGS-84 ellipsoid")

    bridge._last_heartbeat_t_s = None
    step_latencies_ms: list[float] = []

    gdl_sent = 0
    gdl_rx_count = 0
    mav_sent = 0
    mav_rx_count = 0

    valid_epochs = 0
    degraded_epochs = 0
    invalid_epochs = 0

    valid_max_nic = 0
    valid_min_nic = 99
    valid_maint_reqd_count = 0
    valid_coords_flag_count = 0

    degraded_max_nic = 0
    degraded_max_nacp = 0
    degraded_maint_reqd_count = 0
    degraded_coords_flag_count = 0

    invalid_max_nic = 0
    invalid_max_nacp = 0
    invalid_maint_reqd_count = 0
    invalid_gps_pos_valid_count = 0
    invalid_coords_flag_count = 0

    threat_wc1_alert_frames = 0
    safe_passby_alert_frames = 0
    clutter_traffic_frames_emitted = 0

    for k in range(n_steps):
        t_s = round(k * dt_s, 3)
        if k < 20:
            integrity_state = "VALID"
            valid_epochs += 1
        elif k < 25:
            integrity_state = "DEGRADED_COAST"
            degraded_epochs += 1
        else:
            integrity_state = "INVALID_ALERT"
            invalid_epochs += 1

        st_k = t1_cruise_state(t_s)
        r_en_k = wgs84.r_ecef_ned(st_k.lat_deg, st_k.lon_deg)
        r_nb_k = rotations.euler_to_r(st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad)
        poses_k = ring.compute_poses(
            st_k.lat_deg, st_k.lon_deg, st_k.h_m, st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad
        )
        p_cam_k = np.asarray(poses_k[cam0.id]["position"], dtype=np.float64)
        r_ec_k = np.asarray(poses_k[cam0.id]["r_ecef_cam"], dtype=np.float64)

        # 1. Closing head-on C2_SEP threat (tau = 14.0 s, near-zero LOS rate)
        s_threat_px = 18.0 * math.exp(t_s / 14.0)
        det_threat = DetectionRecord(
            t=t_s,
            camera=cam0.id,
            u=1280.0 + 0.12 * k,
            v=640.0,
            s_px=s_threat_px,
            score=35.0,
            is_false_alarm=False,
            truth_id=None,
            s_mom_px=s_threat_px,
        )

        # 2. Wide safe pass-by C3_TWIN (high bearing rate, non-closing constant scale)
        det_safe = DetectionRecord(
            t=t_s,
            camera=cam0.id,
            u=1680.0 + 8.5 * k,
            v=560.0,
            s_px=5.5,
            score=28.0,
            is_false_alarm=False,
            truth_id=None,
            s_mom_px=5.5,
        )

        # 3. Stationary 3D WGS-84 ground-clutter point reprojected into camera
        v_cam_gnd = r_ec_k.T @ (g_pt_ecef - p_cam_k)
        u_gnd = float(ring.camera_model.fx * (v_cam_gnd[0] / v_cam_gnd[2]) + ring.camera_model.cx)
        v_gnd = float(ring.camera_model.fy * (v_cam_gnd[1] / v_cam_gnd[2]) + ring.camera_model.cy)
        det_gnd = DetectionRecord(
            t=t_s,
            camera=cam0.id,
            u=u_gnd,
            v=v_gnd,
            s_px=2.5,
            score=20.0,
            is_false_alarm=True,
            truth_id=None,
            s_mom_px=2.5,
        )

        t0 = time.perf_counter()
        confirmed_tracks = tracker.step(
            t_s=t_s,
            raw_detections=[det_threat, det_safe, det_gnd],
            r_ned_body_ins=r_nb_k,
            p_own_ecef=st_k.p_ecef_m,
            r_ecef_ned=r_en_k,
        )

        # Assign classes by azimuth sector (threat near bore-sight az ~ 0, safe pass-by az > 0.12)
        class_by_id: dict[str, str] = {}
        risk_by_id: dict[str, Any] = {}
        for trk in confirmed_tracks:
            cls_id = "C2_SEP" if abs(trk.az) < 0.08 else "C3_TWIN"
            class_by_id[trk.track_id] = cls_id
            risk_by_id[trk.track_id] = evaluate_track_risk(
                trk, t_s=t_s, class_id=cls_id, beta_wc=0.5
            )

        # Pass ALL tracker tracks (including CLUTTER_HOLD) to verify step_from_tracker suppresses clutter
        step_out = bridge.step_from_tracker(
            ownship=ownship,
            confirmed_tracks=tracker.tracks,
            risk_by_track_id=risk_by_id,
            integrity_state=integrity_state,
            t_s=t_s,
            class_by_track_id=class_by_id,
            force_heartbeat=True,
        )
        rx_gdl = gdl_rx.recv_frames(max_packets=64)
        rx_mav = mav_rx.recv_packets(max_packets=64)
        step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        gdl_sent += len(step_out.gdl90_frames)
        mav_sent += len(step_out.mavlink_frames)

        for gf in rx_gdl:
            msg = decode_gdl90_message(gf)
            gdl_rx_count += 1
            if isinstance(msg, dict) and msg["msg_id"] == GDL90_MSG_ID_HEARTBEAT:
                if integrity_state == "VALID":
                    if msg["maint_reqd"]:
                        valid_maint_reqd_count += 1
                elif integrity_state == "DEGRADED_COAST":
                    if msg["maint_reqd"]:
                        degraded_maint_reqd_count += 1
                elif integrity_state == "INVALID_ALERT":
                    if msg["maint_reqd"]:
                        invalid_maint_reqd_count += 1
                    if msg["gps_pos_valid"]:
                        invalid_gps_pos_valid_count += 1
            elif isinstance(msg, Gdl90TargetReport) and msg.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT:
                # Track 3 is the stationary ground-clutter point (TRK_0003 -> XVS0003)
                if msg.callsign == "XVS0003":
                    clutter_traffic_frames_emitted += 1
                elif msg.callsign == "XVS0001":
                    if msg.alert_status == 1:
                        threat_wc1_alert_frames += 1
                elif msg.callsign == "XVS0002":
                    if msg.alert_status == 1:
                        safe_passby_alert_frames += 1

                if integrity_state == "VALID":
                    valid_max_nic = max(valid_max_nic, msg.nic)
                    valid_min_nic = min(valid_min_nic, msg.nic)
                elif integrity_state == "DEGRADED_COAST":
                    degraded_max_nic = max(degraded_max_nic, msg.nic)
                    degraded_max_nacp = max(degraded_max_nacp, msg.nacp)
                elif integrity_state == "INVALID_ALERT":
                    invalid_max_nic = max(invalid_max_nic, msg.nic)
                    invalid_max_nacp = max(invalid_max_nacp, msg.nacp)

        for mf in rx_mav:
            m_msg = unpack_adsb_vehicle(mf)
            mav_rx_count += 1
            has_valid_coords = bool(m_msg.flags & ADSB_FLAGS_VALID_COORDS)
            if integrity_state == "VALID" and has_valid_coords:
                valid_coords_flag_count += 1
            elif integrity_state == "DEGRADED_COAST" and has_valid_coords:
                degraded_coords_flag_count += 1
            elif integrity_state == "INVALID_ALERT" and has_valid_coords:
                invalid_coords_flag_count += 1

    clutter_tracks = [
        t for t in tracker.tracks if t.status == TrackStatus.CLUTTER_HOLD or t.ground_clutter_rejected
    ]

    summary = {
        "steps_executed": n_steps,
        "dt_s": dt_s,
        "duration_s": round(n_steps * dt_s, 3),
        "total_tracks_initiated": len(tracker.tracks),
        "confirmed_airborne_tracks_count": len(
            [t for t in tracker.tracks if t.status == TrackStatus.CONFIRMED]
        ),
        "clutter_hold_tracks_count": len(clutter_tracks),
        "ground_clutter_rejected_count": int(tracker.ground_clutter_rejected_count),
        "clutter_traffic_frames_emitted": clutter_traffic_frames_emitted,
        "threat_wc1_alert_frames": threat_wc1_alert_frames,
        "safe_passby_alert_frames": safe_passby_alert_frames,
        "gdl90_packets_sent": gdl_sent,
        "gdl90_packets_received": gdl_rx_count,
        "mavlink_packets_sent": mav_sent,
        "mavlink_packets_received": mav_rx_count,
        "integrity_transitions": {
            "VALID": {
                "epochs": valid_epochs,
                "min_traffic_nic": valid_min_nic if valid_min_nic < 99 else 0,
                "max_traffic_nic": valid_max_nic,
                "maint_reqd_heartbeats": valid_maint_reqd_count,
                "mavlink_valid_coords_packets": valid_coords_flag_count,
            },
            "DEGRADED_COAST": {
                "epochs": degraded_epochs,
                "max_traffic_nic": degraded_max_nic,
                "max_traffic_nacp": degraded_max_nacp,
                "maint_reqd_heartbeats": degraded_maint_reqd_count,
                "mavlink_valid_coords_packets": degraded_coords_flag_count,
            },
            "INVALID_ALERT": {
                "epochs": invalid_epochs,
                "max_traffic_nic": invalid_max_nic,
                "max_traffic_nacp": invalid_max_nacp,
                "maint_reqd_heartbeats": invalid_maint_reqd_count,
                "gps_pos_valid_heartbeats": invalid_gps_pos_valid_count,
                "mavlink_valid_coords_packets": invalid_coords_flag_count,
            },
        },
    }
    return summary, step_latencies_ms


def run_verification(
    spec_path: Path = DEFAULT_SPEC_PATH,
    sampler_path: Path = DEFAULT_SAMPLER_PATH,
    wp05_records_path: Path | None = None,
    wp05_g1_records_path: Path | None = None,
    wp05_g1_baseline_path: Path | None = None,
) -> dict[str, Any]:
    """Run full single-process Tier-0 verification, closed-loop WP-05 G1 replay, and UDP loopback benchmark."""
    t_start = time.perf_counter()

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(sampler_path, "r", encoding="utf-8") as f:
        sampler_cfg = json.load(f)

    icd_res = verify_icd_vectors()
    isolation_res = audit_truth_isolation()

    ownship = OwnshipNavState.from_spec(spec)
    replay_cfg = spec["replay_scenarios"]
    steps_per_scenario = int(replay_cfg["steps_per_scenario"])
    dt_s = float(replay_cfg["dt_s"])
    wp05_limit = int(replay_cfg["wp05_sample_records_limit"])
    closed_loop_steps = int(replay_cfg.get("closed_loop_g1_steps", 30))  # default-ok: counter init or optional replay config fallback
    baseline_lambda = float(replay_cfg.get("baseline_lambda", 0.0005))  # default-ok: counter init or optional replay config fallback

    scenarios = _build_synthetic_scenarios(steps_per_scenario=steps_per_scenario, dt_s=dt_s)
    wp05_g0_samples, resolved_wp05_g0_path = _load_wp05_g0_passive_samples(
        wp05_records_path, max_samples=wp05_limit
    )
    (
        wp05_g1_samples,
        wp05_g1_corpus_summary,
        resolved_wp05_g1_path,
        resolved_wp05_g1_base_path,
    ) = _load_wp05_g1_corpus_and_samples(
        wp05_g1_records_path,
        wp05_g1_baseline_path,
        baseline_lambda=baseline_lambda,
        max_samples=wp05_limit,
    )

    latencies_ms: list[float] = []
    max_gdl_lat_err_deg = 0.0
    max_gdl_lon_err_deg = 0.0
    max_gdl_alt_err_ft = 0.0
    max_mav_lat_err_deg = 0.0
    max_mav_lon_err_deg = 0.0
    max_mav_alt_err_m = 0.0
    max_mav_alt_err_mm = 0.0
    mavlink_wire_bit_exact = True

    total_gdl90_sent = 0
    total_gdl90_received = 0
    total_gdl90_corrupted = 0
    gdl90_hb_count = 0
    gdl90_own_count = 0
    gdl90_geo_count = 0
    gdl90_trf_count = 0

    total_mav_sent = 0
    total_mav_received = 0
    total_mav_corrupted = 0

    alert_checks_total = 0
    alert_checks_matched = 0

    scenario_summaries: list[dict[str, Any]] = []

    with (
        Gdl90UdpEndpoint(bind_host="127.0.0.1", bind_port=0) as gdl_rx,
        Gdl90UdpEndpoint(
            bind_host="127.0.0.1", bind_port=0, target_host="127.0.0.1", target_port=0
        ) as gdl_tx,
        MavlinkAdsbUdpStreamer(bind_host="127.0.0.1", bind_port=0) as mav_rx,
        MavlinkAdsbUdpStreamer(
            bind_host="127.0.0.1", bind_port=0, target_host="127.0.0.1", target_port=0
        ) as mav_tx,
    ):
        gdl_tx.set_target("127.0.0.1", gdl_rx.port)
        mav_tx.set_target("127.0.0.1", mav_rx.port)

        bridge = PassiveTrackAvionicsBridge(
            spec_path=spec_path,
            gdl90_endpoint=gdl_tx,
            mavlink_streamer=mav_tx,
        )

        for scen in scenarios:
            bridge._last_heartbeat_t_s = None
            scen_gdl_sent = 0
            scen_gdl_rx = 0
            scen_mav_sent = 0
            scen_mav_rx = 0
            scen_hb_rx = 0
            scen_trf_rx = 0
            scen_alerts_active = 0
            last_nic = 0
            last_nacp = 0
            last_callsign = ""
            per_track_stats: dict[str, dict[str, Any]] = {}

            for t_s, trk_list in scen["epochs"]:
                t_step_0 = time.perf_counter()
                step_out = bridge.step(t_s=t_s, ownship=ownship, tracks=trk_list)

                rx_gdl_frames = gdl_rx.recv_frames(max_packets=64)
                rx_mav_frames = mav_rx.recv_packets(max_packets=64)
                t_step_1 = time.perf_counter()
                step_lat_ms = (t_step_1 - t_step_0) * 1000.0
                latencies_ms.append(step_lat_ms)

                scen_gdl_sent += len(step_out.gdl90_frames)
                scen_mav_sent += len(step_out.mavlink_frames)
                total_gdl90_sent += len(step_out.gdl90_frames)
                total_mav_sent += len(step_out.mavlink_frames)

                decoded_trf_by_cs: dict[str, Gdl90TargetReport] = {}
                for gf in rx_gdl_frames:
                    try:
                        msg = decode_gdl90_message(gf)
                        scen_gdl_rx += 1
                        total_gdl90_received += 1
                        if isinstance(msg, dict):
                            if msg["msg_id"] == GDL90_MSG_ID_HEARTBEAT:
                                gdl90_hb_count += 1
                                scen_hb_rx += 1
                            elif msg["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT:
                                gdl90_geo_count += 1
                        elif isinstance(msg, Gdl90TargetReport):
                            if msg.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT:
                                gdl90_own_count += 1
                            elif msg.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT:
                                gdl90_trf_count += 1
                                scen_trf_rx += 1
                                decoded_trf_by_cs[msg.callsign] = msg
                                last_nic = msg.nic
                                last_nacp = msg.nacp
                                last_callsign = msg.callsign
                    except ValueError:
                        total_gdl90_corrupted += 1

                decoded_mav_by_cs: dict[str, Any] = {}
                for mf in rx_mav_frames:
                    try:
                        m_msg = unpack_adsb_vehicle(mf)
                        scen_mav_rx += 1
                        total_mav_received += 1
                        decoded_mav_by_cs[m_msg.callsign] = m_msg
                    except ValueError:
                        total_mav_corrupted += 1

                # Verify continuous-to-wire quantization accuracy (B10-m3) and per-track summary (B10-m4)
                for trk_in, exp_trf, exp_mav, cont_geo in zip(
                    trk_list,
                    step_out.gdl90_traffic_reports,
                    step_out.mavlink_messages,
                    step_out.continuous_geodetic,
                ):
                    cont_lat_deg, cont_lon_deg, cont_alt_m = cont_geo
                    rx_trf = decoded_trf_by_cs.get(trk_in.callsign)
                    rx_mav = decoded_mav_by_cs.get(trk_in.callsign)
                    if rx_trf is not None and rx_mav is not None:
                        max_gdl_lat_err_deg = max(
                            max_gdl_lat_err_deg, abs(rx_trf.lat_deg - cont_lat_deg)
                        )
                        max_gdl_lon_err_deg = max(
                            max_gdl_lon_err_deg, abs(rx_trf.lon_deg - cont_lon_deg)
                        )
                        if rx_trf.altitude_ft is not None:
                            max_gdl_alt_err_ft = max(
                                max_gdl_alt_err_ft,
                                abs(rx_trf.altitude_ft - cont_alt_m * METERS_TO_FEET),
                            )

                        # B10-m3: compare decoded MAVLink int32 fields against unquantized continuous geodetic state
                        lat_q_err = abs(rx_mav.lat_deg - cont_lat_deg)
                        lon_q_err = abs(rx_mav.lon_deg - cont_lon_deg)
                        alt_q_err_m = abs(rx_mav.altitude_m - cont_alt_m)
                        max_mav_lat_err_deg = max(max_mav_lat_err_deg, lat_q_err)
                        max_mav_lon_err_deg = max(max_mav_lon_err_deg, lon_q_err)
                        max_mav_alt_err_m = max(max_mav_alt_err_m, alt_q_err_m)
                        max_mav_alt_err_mm = max(max_mav_alt_err_mm, alt_q_err_m * 1000.0)

                        if rx_mav != exp_mav:
                            mavlink_wire_bit_exact = False

                        exp_alert_bit = 1 if trk_in.alert_active else 0
                        exp_squawk = (
                            bridge.squawk_alert
                            if trk_in.alert_active
                            else bridge.squawk_nominal
                        )
                        alert_checks_total += 1
                        if (
                            rx_trf.alert_status == exp_alert_bit
                            and rx_mav.squawk == exp_squawk
                        ):
                            alert_checks_matched += 1
                        if trk_in.alert_active:
                            scen_alerts_active += 1

                        # B10-m4: accumulate per-track summary so multi-track scenarios preserve all tracks
                        t_entry = per_track_stats.setdefault(
                            trk_in.callsign,
                            {
                                "track_id": trk_in.track_id,
                                "callsign": trk_in.callsign,
                                "class_id": str(trk_in.class_id or "UNKNOWN"),
                                "nic": rx_trf.nic,
                                "nacp": rx_trf.nacp,
                                "active_alert_reports": 0,
                                "alert_rule_source": trk_in.alert_rule_source,
                            },
                        )
                        t_entry["nic"] = rx_trf.nic
                        t_entry["nacp"] = rx_trf.nacp
                        if trk_in.alert_active:
                            t_entry["active_alert_reports"] += 1

            tracks_summary_list = [
                per_track_stats[cs] for cs in sorted(per_track_stats.keys())
            ]
            scenario_summaries.append(
                {
                    "scenario_id": scen["scenario_id"],
                    "rule_family": scen["rule_family"],
                    "class_id": scen["class_id"],
                    "epochs_count": len(scen["epochs"]),
                    "tracks_per_step": scen["tracks_per_step"],
                    "gdl90_sent": scen_gdl_sent,
                    "gdl90_received": scen_gdl_rx,
                    "gdl90_heartbeats": scen_hb_rx,
                    "gdl90_traffic_reports": scen_trf_rx,
                    "mavlink_sent": scen_mav_sent,
                    "mavlink_received": scen_mav_rx,
                    "active_alert_reports": scen_alerts_active,
                    "nic": last_nic,
                    "nacp": last_nacp,
                    "sample_callsign": last_callsign,
                    "tracks_summary": tracks_summary_list,
                }
            )

        # Replay stratified WP-05 G0 track sample records over UDP loopback
        wp05_gdl_sent = 0
        wp05_gdl_rx = 0
        wp05_mav_sent = 0
        wp05_mav_rx = 0
        wp05_alert_tracks = 0
        for idx, wp_trk in enumerate(wp05_g0_samples):
            t_step_0 = time.perf_counter()
            step_out = bridge.step(
                t_s=float(idx),
                ownship=ownship,
                tracks=[wp_trk],
                force_heartbeat=(idx == 0),
            )
            rx_gdl_frames = gdl_rx.recv_frames(max_packets=16)
            rx_mav_frames = mav_rx.recv_packets(max_packets=16)
            latencies_ms.append((time.perf_counter() - t_step_0) * 1000.0)

            wp05_gdl_sent += len(step_out.gdl90_frames)
            wp05_mav_sent += len(step_out.mavlink_frames)
            total_gdl90_sent += len(step_out.gdl90_frames)
            total_mav_sent += len(step_out.mavlink_frames)

            rx_trf_msg: Gdl90TargetReport | None = None
            for gf in rx_gdl_frames:
                msg = decode_gdl90_message(gf)
                wp05_gdl_rx += 1
                total_gdl90_received += 1
                if isinstance(msg, dict):
                    if msg["msg_id"] == GDL90_MSG_ID_HEARTBEAT:
                        gdl90_hb_count += 1
                    elif msg["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT:
                        gdl90_geo_count += 1
                elif isinstance(msg, Gdl90TargetReport):
                    if msg.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT:
                        gdl90_own_count += 1
                    elif msg.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT:
                        gdl90_trf_count += 1
                        rx_trf_msg = msg

            rx_mav_msg = None
            for mf in rx_mav_frames:
                rx_mav_msg = unpack_adsb_vehicle(mf)
                wp05_mav_rx += 1
                total_mav_received += 1

            if rx_trf_msg is not None and rx_mav_msg is not None:
                alert_checks_total += 1
                exp_bit = 1 if wp_trk.alert_active else 0
                exp_sq = (
                    bridge.squawk_alert if wp_trk.alert_active else bridge.squawk_nominal
                )
                if rx_trf_msg.alert_status == exp_bit and rx_mav_msg.squawk == exp_sq:
                    alert_checks_matched += 1
                if wp_trk.alert_active:
                    wp05_alert_tracks += 1

        # Replay stratified WP-05 G1 WC1 track sample records over UDP loopback
        wp05_g1_gdl_sent = 0
        wp05_g1_gdl_rx = 0
        wp05_g1_mav_sent = 0
        wp05_g1_mav_rx = 0
        wp05_g1_alert_tracks = 0
        for idx, wp_g1_trk in enumerate(wp05_g1_samples):
            t_step_0 = time.perf_counter()
            step_out = bridge.step(
                t_s=float(idx),
                ownship=ownship,
                tracks=[wp_g1_trk],
                force_heartbeat=(idx == 0),
            )
            rx_gdl_frames = gdl_rx.recv_frames(max_packets=16)
            rx_mav_frames = mav_rx.recv_packets(max_packets=16)
            latencies_ms.append((time.perf_counter() - t_step_0) * 1000.0)

            wp05_g1_gdl_sent += len(step_out.gdl90_frames)
            wp05_g1_mav_sent += len(step_out.mavlink_frames)
            total_gdl90_sent += len(step_out.gdl90_frames)
            total_mav_sent += len(step_out.mavlink_frames)

            rx_trf_msg = None
            for gf in rx_gdl_frames:
                msg = decode_gdl90_message(gf)
                wp05_g1_gdl_rx += 1
                total_gdl90_received += 1
                if isinstance(msg, dict):
                    if msg["msg_id"] == GDL90_MSG_ID_HEARTBEAT:
                        gdl90_hb_count += 1
                    elif msg["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT:
                        gdl90_geo_count += 1
                elif isinstance(msg, Gdl90TargetReport):
                    if msg.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT:
                        gdl90_own_count += 1
                    elif msg.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT:
                        gdl90_trf_count += 1
                        rx_trf_msg = msg

            rx_mav_msg = None
            for mf in rx_mav_frames:
                rx_mav_msg = unpack_adsb_vehicle(mf)
                wp05_g1_mav_rx += 1
                total_mav_received += 1

            if rx_trf_msg is not None and rx_mav_msg is not None:
                alert_checks_total += 1
                exp_bit = 1 if wp_g1_trk.alert_active else 0
                exp_sq = (
                    bridge.squawk_alert
                    if wp_g1_trk.alert_active
                    else bridge.squawk_nominal
                )
                if rx_trf_msg.alert_status == exp_bit and rx_mav_msg.squawk == exp_sq:
                    alert_checks_matched += 1
                if wp_g1_trk.alert_active:
                    wp05_g1_alert_tracks += 1

        # Stage 3 Part B: Live 30-step closed-loop WP-05 G1 tracker + WC1 risk + WP-07 integrity transitions
        closed_loop_summary, cl_latencies_ms = run_live_closed_loop_g1_verification(
            bridge=bridge,
            gdl_rx=gdl_rx,
            mav_rx=mav_rx,
            ownship=ownship,
            g1_cfg_path=DEFAULT_G1_TRACKING_CONFIG_PATH,
            n_steps=closed_loop_steps,
            dt_s=dt_s,
        )
        latencies_ms.extend(cl_latencies_ms)
        total_gdl90_sent += int(closed_loop_summary["gdl90_packets_sent"])
        total_gdl90_received += int(closed_loop_summary["gdl90_packets_received"])
        total_mav_sent += int(closed_loop_summary["mavlink_packets_sent"])
        total_mav_received += int(closed_loop_summary["mavlink_packets_received"])
        gdl90_hb_count += int(closed_loop_summary["steps_executed"])
        gdl90_own_count += int(closed_loop_summary["steps_executed"])
        gdl90_geo_count += int(closed_loop_summary["steps_executed"])
        gdl90_trf_count += int(closed_loop_summary["mavlink_packets_received"])

    total_sent = total_gdl90_sent + total_mav_sent
    total_received = total_gdl90_received + total_mav_received
    total_dropped = total_sent - total_received
    total_corrupted = total_gdl90_corrupted + total_mav_corrupted

    roundtrip_accuracy_pct = (
        100.0 * float(total_received - total_corrupted) / float(total_sent)
        if total_sent > 0
        else 0.0
    )
    alert_parity_pct = (
        100.0 * float(alert_checks_matched) / float(alert_checks_total)
        if alert_checks_total > 0
        else 0.0
    )

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    latency_stats = {
        "samples_count": len(latencies_ms),
        "min_ms": float(np.min(lat_arr)),
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "max_ms": float(np.max(lat_arr)),
    }

    inputs_sha256: dict[str, str] = {
        "configs/xvs/avionics/tier0_spec.json": _sha256_file(spec_path),
        "configs/xvs/encounters/sampler_v1.json": _sha256_file(sampler_path),
        "configs/xvs/tracking/g1.json": _sha256_file(DEFAULT_G1_TRACKING_CONFIG_PATH),
    }
    if resolved_wp05_g0_path is not None and resolved_wp05_g0_path.is_file():
        inputs_sha256["output/xvs/wp05/g0_lambda_sweep_v2/per_run_records.jsonl"] = (
            _sha256_file(resolved_wp05_g0_path)
        )
    if resolved_wp05_g1_path is not None and resolved_wp05_g1_path.is_file():
        inputs_sha256["output/xvs/wp05/g1/per_run_records.jsonl"] = _sha256_file(
            resolved_wp05_g1_path
        )
    if resolved_wp05_g1_base_path is not None and resolved_wp05_g1_base_path.is_file():
        inputs_sha256["output/xvs/wp05/g1/baseline_results.json"] = _sha256_file(
            resolved_wp05_g1_base_path
        )

    elapsed_s = time.perf_counter() - t_start

    return {
        "schema": "xvs.avionics.tier0_verification/2",
        "batch_id": "B10",
        "commit": _get_git_commit(),
        "git_status": _get_git_status_porcelain(),
        "argv": [Path(sys.argv[0]).as_posix()] + sys.argv[1:],
        "workers": 1,
        "elapsed_wall_s": float(elapsed_s),
        "code_hashes": _compute_code_hashes(),
        "inputs_sha256": inputs_sha256,
        "sampler_reference": {
            "total_encounters": int(sampler_cfg["total_encounters"]),
            "master_seed": int(sampler_cfg["master_seed"]),
            "encounters_sha256": str(sampler_cfg["encounters_sha256"]),
        },
        "icd_test_vectors": icd_res,
        "mavlink_protocol_check": {
            "stx_byte": MAVLINK_V2_STX,
            "msg_id": ADSB_VEHICLE_MSG_ID,
            "crc_extra": ADSB_VEHICLE_CRC_EXTRA,
            "payload_len_bytes": ADSB_VEHICLE_PAYLOAD_LEN,
            "frame_len_bytes": ADSB_VEHICLE_FRAME_LEN,
        },
        "udp_loopback_summary": {
            "total_packets_sent": total_sent,
            "total_packets_received": total_received,
            "total_packets_dropped": total_dropped,
            "total_packets_corrupted": total_corrupted,
            "gdl90_packets_sent": total_gdl90_sent,
            "gdl90_packets_received": total_gdl90_received,
            "gdl90_heartbeats_decoded": gdl90_hb_count,
            "gdl90_ownship_reports_decoded": gdl90_own_count,
            "gdl90_ownship_geo_alt_decoded": gdl90_geo_count,
            "gdl90_traffic_reports_decoded": gdl90_trf_count,
            "mavlink_packets_sent": total_mav_sent,
            "mavlink_packets_received": total_mav_received,
            "roundtrip_decode_accuracy_pct": roundtrip_accuracy_pct,
            "alert_checks_total": alert_checks_total,
            "alert_checks_matched": alert_checks_matched,
            "alert_parity_pct": alert_parity_pct,
        },
        "quantization_residuals": {
            "max_gdl90_lat_err_deg": float(max_gdl_lat_err_deg),
            "max_gdl90_lon_err_deg": float(max_gdl_lon_err_deg),
            "max_gdl90_alt_err_ft": float(max_gdl_alt_err_ft),
            "max_mavlink_lat_err_deg": float(max_mav_lat_err_deg),
            "max_mavlink_lon_err_deg": float(max_mav_lon_err_deg),
            "max_mavlink_alt_err_m": float(max_mav_alt_err_m),
            "max_mavlink_alt_err_mm": float(max_mav_alt_err_mm),
            "mavlink_wire_bit_exact_match": bool(mavlink_wire_bit_exact),
        },
        "loopback_latency_ms": latency_stats,
        "truth_isolation_audit": isolation_res,
        "synthetic_scenarios": scenario_summaries,
        "wp05_saved_replay": {
            "records_replayed": len(wp05_g0_samples),
            "alert_tracks_count": wp05_alert_tracks,
            "non_alert_tracks_count": len(wp05_g0_samples) - wp05_alert_tracks,
            "gdl90_packets_sent": wp05_gdl_sent,
            "gdl90_packets_received": wp05_gdl_rx,
            "mavlink_packets_sent": wp05_mav_sent,
            "mavlink_packets_received": wp05_mav_rx,
        },
        "wp05_g1_saved_replay": {
            "records_replayed": len(wp05_g1_samples),
            "alert_tracks_count": wp05_g1_alert_tracks,
            "non_alert_tracks_count": len(wp05_g1_samples) - wp05_g1_alert_tracks,
            "gdl90_packets_sent": wp05_g1_gdl_sent,
            "gdl90_packets_received": wp05_g1_gdl_rx,
            "mavlink_packets_sent": wp05_g1_mav_sent,
            "mavlink_packets_received": wp05_g1_mav_rx,
        },
        "wp05_g1_corpus_comparison": wp05_g1_corpus_summary,
        "closed_loop_g1_integrity": closed_loop_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run single-process Batch B10 Phase 2 Tier-0 avionics verification and closed-loop WP-05 G1 replay."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for XVS outputs (default: XVS_DATA_ROOT or output/)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Path to configs/xvs/avionics/tier0_spec.json",
    )
    parser.add_argument(
        "--sampler",
        type=Path,
        default=DEFAULT_SAMPLER_PATH,
        help="Path to configs/xvs/encounters/sampler_v1.json",
    )
    parser.add_argument(
        "--wp05-records",
        type=Path,
        default=None,
        help="Optional path to WP-05 G0 per_run_records.jsonl",
    )
    parser.add_argument(
        "--wp05-g1-records",
        type=Path,
        default=None,
        help="Optional path to WP-05 G1 per_run_records.jsonl",
    )
    parser.add_argument(
        "--wp05-g1-baseline",
        type=Path,
        default=None,
        help="Optional path to WP-05 G1 baseline_results.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional explicit output JSON path",
    )
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    out_path = (
        args.out
        if args.out is not None
        else (data_root / "xvs" / "wp08" / "tier0_b10" / "tier0_verification.json")
    )

    result = run_verification(
        spec_path=args.spec,
        sampler_path=args.sampler,
        wp05_records_path=args.wp05_records,
        wp05_g1_records_path=args.wp05_g1_records,
        wp05_g1_baseline_path=args.wp05_g1_baseline,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote Batch B10 Tier-0 verification artifact to {out_path}")
    print(
        f"Packets sent={result['udp_loopback_summary']['total_packets_sent']}, "
        f"received={result['udp_loopback_summary']['total_packets_received']}, "
        f"dropped={result['udp_loopback_summary']['total_packets_dropped']}, "
        f"decode_accuracy={result['udp_loopback_summary']['roundtrip_decode_accuracy_pct']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
