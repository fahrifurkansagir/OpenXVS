"""Croissant 1.0 JSON-LD metadata generator & WSL2 `jsonschema` validator for XVS-Bench (DR-08 / SP3 §3.2)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .hash_chain import sha256_file

CROISSANT_1_0_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OpenXVS XVS-Bench Croissant 1.0 JSON-LD Schema",
    "type": "object",
    "required": [
        "@context",
        "@type",
        "name",
        "description",
        "version",
        "license",
        "citeAs",
        "distribution",
        "recordSet",
        "openxvs:civilScope",
        "openxvs:phase4LiveSkySlot",
    ],
    "properties": {
        "@context": {
            "type": "object",
            "required": ["@vocab", "cr", "sc", "dct", "openxvs"],
            "properties": {
                "@vocab": {"const": "https://schema.org/"},
                "cr": {"const": "http://mlcommons.org/croissant/"},
                "sc": {"const": "https://schema.org/"},
                "dct": {"const": "http://purl.org/dc/terms/"},
                "openxvs": {"type": "string"},
            },
        },
        "@type": {"const": "sc:Dataset"},
        "conformsTo": {"const": "http://mlcommons.org/croissant/1.0"},
        "name": {"type": "string", "minLength": 3},
        "description": {"type": "string", "minLength": 10},
        "version": {"type": "string", "minLength": 3},
        "license": {"type": "string", "minLength": 3},
        "citeAs": {"type": "string", "minLength": 10},
        "distribution": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["@type", "@id", "name", "contentUrl", "encodingFormat", "sha256"],
                "properties": {
                    "@type": {"const": "cr:FileObject"},
                    "@id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "contentUrl": {"type": "string", "minLength": 1},
                    "encodingFormat": {"type": "string", "minLength": 3},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
        },
        "recordSet": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "required": ["@type", "@id", "name", "description", "field"],
                "properties": {
                    "@type": {"const": "cr:RecordSet"},
                    "@id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 5},
                    "field": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["@type", "@id", "name", "dataType"],
                            "properties": {
                                "@type": {"const": "cr:Field"},
                                "@id": {"type": "string"},
                                "name": {"type": "string"},
                                "dataType": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "openxvs:civilScope": {
            "type": "object",
            "required": ["civil_and_dual_use_only", "weapons_or_targeting_excluded", "regulatory_disclaimer"],
            "properties": {
                "civil_and_dual_use_only": {"const": True},
                "weapons_or_targeting_excluded": {"const": True},
                "regulatory_disclaimer": {"type": "string", "minLength": 10},
            },
        },
        "openxvs:phase4LiveSkySlot": {
            "type": "object",
            "required": ["split", "status", "hardware_spec", "adsb_gates"],
            "properties": {
                "split": {"const": "live_sky_imx296_adsb"},
                "status": {"const": "PREWIRED_PHASE4_PENDING_CAPTURE"},
            },
        },
    },
}


def _encoding_format_for(rel_path: str) -> str:
    if rel_path.endswith(".json"):
        return "application/json"
    if rel_path.endswith(".log") or rel_path.endswith(".txt"):
        return "text/plain"
    if rel_path.endswith(".md"):
        return "text/markdown"
    return "application/octet-stream"


def _make_field(rs_id: str, name: str, data_type: str, desc: str) -> dict[str, Any]:
    return {
        "@type": "cr:Field",
        "@id": f"{rs_id}/{name}",
        "name": name,
        "dataType": data_type,
        "description": desc,
    }


def build_croissant_jsonld(
    repo_root: Path,
    tracked_rel_paths: Sequence[str],
    phase4_slot: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic Croissant 1.0 JSON-LD document binding real disk SHA-256 FileObjects."""
    distribution: list[dict[str, Any]] = []
    for idx, rel in enumerate(sorted(set(p.replace("\\", "/").lstrip("./") for p in tracked_rel_paths))):
        abs_p = repo_root / rel
        if not abs_p.is_file():
            raise FileNotFoundError(f"Missing Croissant distribution artifact: {abs_p}")
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"file_{idx:03d}_{Path(rel).stem}",
                "name": Path(rel).name,
                "contentUrl": rel,
                "encodingFormat": _encoding_format_for(rel),
                "contentSize": f"{int(abs_p.stat().st_size)} B",
                "sha256": sha256_file(abs_p),
            }
        )

    record_sets: list[dict[str, Any]] = [
        {
            "@type": "cr:RecordSet",
            "@id": "track_a_daa_meshdaa",
            "name": "track_a_daa_meshdaa",
            "description": "Track A: Civil DAA / AOT real-data transfer & 1/2/3-ship XVS-MeshDAA closed-loop safety metrics.",
            "field": [
                _make_field("track_a_daa_meshdaa", "nmac_count_3ship", "sc:Integer", "3-ship NMAC count across 500 encounters"),
                _make_field("track_a_daa_meshdaa", "lowc_count_3ship", "sc:Integer", "3-ship Loss of Well Clear count"),
                _make_field("track_a_daa_meshdaa", "ghost_suppression_ratio", "sc:Float", "Finite 2-ship vs 3-ship triple-ray ghost suppression ratio"),
                _make_field("track_a_daa_meshdaa", "bandwidth_kbps_52001", "sc:Float", "Measured 77 B signed MAVLink v2 #52001 bandwidth"),
                _make_field("track_a_daa_meshdaa", "aot_colab_g4_map", "sc:Text", "Deferred Colab G4 training metric tag"),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "track_b_nav_antispoof",
            "name": "track_b_nav_antispoof",
            "description": "Track B: XVS-Nav GNSS-denied visual navigation, 6-stage aliasing gate & Bonferroni CUSUM anti-spoofing.",
            "field": [
                _make_field("track_b_nav_antispoof", "rectification_max_err_px", "sc:Float", "CONVENTIONS.md ECEF/NED/CAM homography round-trip error"),
                _make_field("track_b_nav_antispoof", "aliasing_nonfinite_blocked", "sc:Integer", "Adversarial NaN/Inf inputs fail-closed across 6 stages"),
                _make_field("track_b_nav_antispoof", "cusum_spoof_trip_latency_ticks", "sc:Integer", "10 Hz ticks from +2.5 m/s East spoof onset to Bank 1 failover"),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "track_c_land_vraim",
            "name": "track_c_land_vraim",
            "description": "Track C: XVS-Land + WP-01b 3DEP TAWS + XVS-VRAIM Cat I V-HPL/V-VPL & LARD real H3 isolation.",
            "field": [
                _make_field("track_c_land_vraim", "vvpl_cat1_short_final_m", "sc:Float", "Vision-RAIM vertical protection level vs 10.0 m Cat I VAL"),
                _make_field("track_c_land_vraim", "taws_nominal_false_alerts", "sc:Integer", "TSO-C151d Class B nominal KSFO 28L glide false alert count"),
                _make_field("track_c_land_vraim", "lard_real_icao_overlap", "sc:Integer", "Train vs LARD real 36-airport ICAO overlap (must be 0)"),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "track_d_xvs_rt_odd",
            "name": "track_d_xvs_rt_odd",
            "description": "Track D: xvs-rt C++20 zero-alloc hot loop (-O0/-O2), 5/5 Kani BMC proofs & EASA R-LM-07..11 ODD audits.",
            "field": [
                _make_field("track_d_xvs_rt_odd", "hot_loop_alloc_bytes_o0", "sc:Integer", "Wrapped malloc bytes during 10,000-tick -O0 audit"),
                _make_field("track_d_xvs_rt_odd", "hot_loop_alloc_bytes_o2", "sc:Integer", "Wrapped malloc bytes during 10,000-tick -O2 audit"),
                _make_field("track_d_xvs_rt_odd", "kani_harnesses_passed", "sc:Integer", "WSL2 real cargo kani passed harness count"),
                _make_field("track_d_xvs_rt_odd", "native_linux_gpu_wcet_ms", "sc:Text", "Deferred native Linux GPU TensorRT WCET tag"),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "live_sky_imx296_adsb",
            "name": "live_sky_imx296_adsb",
            "description": "Pre-wired Phase 4 Live-Sky IMX296 + 16mm + FlightAware/ADSBexchange 1090 MHz ADS-B capture slot (PREWIRED_PHASE4_PENDING_CAPTURE).",
            "field": [
                _make_field("live_sky_imx296_adsb", "mode_s_tc19_decoded", "sc:Boolean", "Mode S Type Code 19 geometric-baro offset decoded"),
                _make_field("live_sky_imx296_adsb", "nic", "sc:Integer", "ADS-B Navigation Integrity Category (gate >= 7)"),
                _make_field("live_sky_imx296_adsb", "nacp", "sc:Integer", "ADS-B Navigation Accuracy Category Position (gate >= 9)"),
                _make_field("live_sky_imx296_adsb", "boresight_rms_deg", "sc:Float", "Camera-to-NED boresight calibration RMS (gate <= 0.15 deg)"),
                _make_field("live_sky_imx296_adsb", "pps_synced", "sc:Boolean", "GPS 1PPS hardware XTR exposure synchronization lock"),
            ],
        },
    ]

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/",
            "dct": "http://purl.org/dc/terms/",
            "openxvs": "https://openxvs.org/schema/v2/",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "OpenXVS-Bench v2.0 (Local Core Milestone M3)",
        "description": (
            "Cryptographically attested 4-track closed-loop civil aviation autonomy benchmark "
            "covering DAA/MeshDAA, GNSS-denied visual navigation, vision-based precision landing "
            "with Vision-RAIM, and deterministic C++20/Rust runtime assurance."
        ),
        "version": "2.0.0-m3-local",
        "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "citeAs": (
            "OpenXVS Project (2026). XVS-Bench v2.0: Cryptographically Attested Closed-Loop "
            "Benchmark for Formally Assured Civil Aerial Autonomy."
        ),
        "distribution": distribution,
        "recordSet": record_sets,
        "openxvs:civilScope": {
            "civil_and_dual_use_only": True,
            "weapons_or_targeting_excluded": True,
            "regulatory_disclaimer": (
                "Architected for IDAL C learned perception bounded by a deterministic ASTM F3269-21 "
                "Simplex safety envelope whose discrete switching, quantization, and past-time MLTL "
                "ODD logic (xvs-rta-core) is bounded-model-checked via Kani."
            ),
        },
        "openxvs:phase4LiveSkySlot": phase4_slot,
    }


def validate_croissant_with_wsl2_jsonschema(croissant_doc: dict[str, Any]) -> dict[str, Any]:
    """Validate Croissant JSON-LD document against `CROISSANT_1_0_JSON_SCHEMA` via WSL2 `jsonschema`."""
    wsl_code = (
        "import importlib.metadata, json, sys, jsonschema; "
        "payload = json.loads(sys.stdin.read()); "
        "v = jsonschema.Draft202012Validator(payload['schema']); "
        "errs = [e.message for e in v.iter_errors(payload['doc'])]; "
        "print(json.dumps({'jsonschema_version': importlib.metadata.version('jsonschema'), "
        "'valid': len(errs) == 0, 'errors': errs}))"
    )
    req = json.dumps({"doc": croissant_doc, "schema": CROISSANT_1_0_JSON_SCHEMA})
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "python3", "-c", wsl_code],
        input=req,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    res = json.loads(proc.stdout.strip())
    return {
        "jsonschema_version": str(res["jsonschema_version"]),
        "valid": bool(res["valid"]),
        "errors": list(res["errors"]),
    }
