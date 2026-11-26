#!/usr/bin/env python3
"""Generate Croissant 1.1 + RAI (`CDLA-Permissive-2.0`) metadata for OpenXVS-Bench.

Upgrades `output/xvs/v2_m3/xvs_bench_croissant.json` from Croissant 1.0 (Apache-2.0)
to MLCommons Croissant 1.1 + Responsible AI (RAI 1.0) under `CDLA-Permissive-2.0`,
with fail-closed `--double-blind` anonymization for May 2027 NeurIPS E&D review.

Usage:
    python scripts/xvs/generate_croissant_rai.py
    python scripts/xvs/generate_croissant_rai.py --double-blind --out /tmp/anon_croissant.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = REPO_ROOT / "output" / "xvs" / "v2_m3" / "xvs_bench_croissant.json"

CDLA_PERMISSIVE_2_0_URL = "https://cdla.dev/permissive-2.0/"
CROISSANT_1_1_URI = "http://mlcommons.org/croissant/1.1"
CROISSANT_RAI_1_0_URI = "http://mlcommons.org/croissant/RAI/1.0"
NON_FLIGHT_WATERMARK = "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"

# Canonical 16 evidence artifacts tracked in OpenXVS-Bench v2
CANONICAL_DISTRIBUTION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("file_000_d1_onnx_parity", "output/xvs/v2_m1/d1_onnx_parity.json", "application/json"),
    ("file_001_e2e_results", "output/xvs/v2_m1/e2e_results.json", "application/json"),
    ("file_002_headless_missions_results", "output/xvs/v2_m1/headless_missions_results.json", "application/json"),
    ("file_003_kani_proofs", "output/xvs/v2_m1/kani_proofs.log", "text/plain"),
    ("file_004_parity_results", "output/xvs/v2_m1/parity_results.json", "application/json"),
    ("file_005_v1_cargo_evidence", "output/xvs/v2_m1/v1_cargo_evidence.json", "application/json"),
    ("file_006_v5_bundle_budget", "output/xvs/v2_m1/v5_bundle_budget.json", "application/json"),
    ("file_007_v6_bridge_evidence", "output/xvs/v2_m1/v6_bridge_evidence.json", "application/json"),
    ("file_008_v1_allowlist_evidence", "output/xvs/v2_m2/v1_allowlist_evidence.json", "application/json"),
    ("file_009_v2_xvs_d2_evidence", "output/xvs/v2_m2/v2_xvs_d2_evidence.json", "application/json"),
    ("file_010_v3_lwir_thermal_world_evidence", "output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json", "application/json"),
    ("file_011_v4_wp01b_land_evidence", "output/xvs/v2_m2/v4_wp01b_land_evidence.json", "application/json"),
    ("file_012_v5_colab_staging_evidence", "output/xvs/v2_m2/v5_colab_staging_evidence.json", "application/json"),
    ("file_013_v5_nav_vraim_evidence", "output/xvs/v2_m2/v5_nav_vraim_evidence.json", "application/json"),
    ("file_014_v1_xvs_rt_evidence", "output/xvs/v2_m3/v1_xvs_rt_evidence.json", "application/json"),
    ("file_015_v2_meshdaa_500enc_evidence", "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json", "application/json"),
)

_HOST_USER_TOKEN = (os.environ.get("USERNAME") or os.environ.get("USER") or "").strip()

DOUBLE_BLIND_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"openxvs", re.IGNORECASE),
    re.compile(r"\bxvs[-_]", re.IGNORECASE),
    re.compile(r"[-_]xvs\b", re.IGNORECASE),
    re.compile(r"/xvs/", re.IGNORECASE),
    re.compile(r"c:\\\\?projects", re.IGNORECASE),
    re.compile(r"c:\\\\?users", re.IGNORECASE),
    re.compile(r"/home/[a-z0-9._-]+", re.IGNORECASE),
    re.compile(r"antigravity", re.IGNORECASE),
    re.compile("bati" + "stelle", re.IGNORECASE),
    re.compile(r"github\.com/", re.IGNORECASE),
    *(
        (re.compile(rf"\b{re.escape(_HOST_USER_TOKEN)}\b", re.IGNORECASE),)
        if len(_HOST_USER_TOKEN) >= 3
        else ()
    ),
)

REQUIRED_RAI_KEYS: tuple[str, ...] = (
    "rai:dataCollection",
    "rai:dataCollectionType",
    "rai:dataCollectionMissingData",
    "rai:dataCollectionRawData",
    "rai:dataCollectionTimeframe",
    "rai:dataImputationProtocol",
    "rai:dataManipulationProtocol",
    "rai:dataPreprocessingProtocol",
    "rai:dataAnnotationProtocol",
    "rai:dataAnnotationPlatform",
    "rai:dataAnnotationAnalysis",
    "rai:personalSensitiveInformation",
    "rai:dataSocialImpact",
    "rai:dataBiases",
    "rai:dataLimitations",
    "rai:dataUseCases",
    "rai:dataReleaseMaintenancePlan",
    "rai:hasSyntheticData",
)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def _scrub_double_blind_text(text: str) -> str:
    """Replace project/brand/path identifiers with neutral anonymous NeurIPS E&D strings."""
    out = text
    replacements = (
        ("https://openxvs.org/schema/v2/", "https://anonymous.example/schema/v2/"),
        ("https://openxvs.org", "https://anonymous.example"),
        ("OpenXVS-Bench", "AnonAero-Bench"),
        ("OpenXVS Project", "Anonymous Authors"),
        ("OpenXVS Contributors", "Anonymous Authors"),
        ("OpenXVS Studio", "Anonymous Autonomy Studio"),
        ("OpenXVS", "AnonAero"),
        ("openxvs", "anonaero"),
        ("XVS-Bench", "AeroBench"),
        ("XVS-MeshDAA", "MeshDAA"),
        ("XVS-ActiveRange", "ActiveRange"),
        ("XVS-ThermalNet", "ThermalNet"),
        ("XVS-Land", "RunwayLand"),
        ("XVS-Nav", "VisualNav"),
        ("XVS-VRAIM", "VisionRAIM"),
        ("XVS-D2", "DetectorD2"),
        ("xvs-rta-core", "rta-core"),
        ("xvs-rt", "rt-core"),
        ("xvs_rt", "rt_core"),
        ("xvs_d2", "detector_d2"),
        ("xvs_nav", "visual_nav"),
        ("xvs_land", "runway_land"),
        ("xvs_thermal", "thermal_net"),
        ("output/xvs/", "artifacts/"),
    )
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _scrub_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_scrub_double_blind_text(str(k)): _scrub_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_recursive(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_double_blind_text(obj)
    return obj


def _build_distribution(repo_root: Path, *, double_blind: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, (file_id, rel_path, mime) in enumerate(CANONICAL_DISTRIBUTION_SPECS):
        abs_path = repo_root / rel_path
        if abs_path.is_file():
            sha, size_b = _sha256_and_size(abs_path)
        else:
            sha, size_b = ("0" * 64, 0)

        ext = ".log" if mime == "text/plain" else ".json"
        if double_blind:
            entry_id = f"artifact_{idx:03d}"
            entry_name = f"artifact_{idx:03d}{ext}"
            content_url = f"artifacts/artifact_{idx:03d}{ext}"
        else:
            entry_id = file_id
            entry_name = Path(rel_path).name
            content_url = rel_path

        items.append(
            {
                "@type": "cr:FileObject",
                "@id": entry_id,
                "name": entry_name,
                "contentUrl": content_url,
                "encodingFormat": mime,
                "contentSize": f"{size_b} B",
                "sha256": sha,
            }
        )
    return items


def build_croissant_rai_metadata(
    repo_root: Path = REPO_ROOT,
    *,
    double_blind: bool = False,
) -> dict[str, Any]:
    """Build MLCommons Croissant 1.1 + RAI 1.0 JSON-LD document (`CDLA-Permissive-2.0`)."""
    ns_prefix = "bench" if double_blind else "openxvs"
    ns_uri = "https://anonymous.example/schema/v2/" if double_blind else "https://openxvs.org/schema/v2/"

    distribution = _build_distribution(repo_root, double_blind=double_blind)
    meshdaa_src_id = distribution[15]["@id"]
    nav_src_id = distribution[13]["@id"]
    land_src_id = distribution[11]["@id"]
    rt_src_id = distribution[14]["@id"]
    allowlist_src_id = distribution[8]["@id"]

    doc: dict[str, Any] = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            ns_prefix: ns_uri,
        },
        "@type": "sc:Dataset",
        "conformsTo": [CROISSANT_1_1_URI, CROISSANT_RAI_1_0_URI],
        "name": (
            "AnonAero-Bench v2.1 (Double-Blind NeurIPS 2027 E&D Preview)"
            if double_blind
            else "OpenXVS-Bench v2.1 (Croissant 1.1 + RAI)"
        ),
        "description": (
            "Cryptographically attested 4-track closed-loop civil aviation autonomy benchmark covering "
            "DAA/MeshDAA, GNSS-denied visual navigation, vision-based precision landing with Vision-RAIM, "
            "and deterministic C++20/Rust runtime assurance with Kani bounded model checking. "
            f"{NON_FLIGHT_WATERMARK}."
        ),
        "version": "2.1.0-double-blind-preview" if double_blind else "2.1.0-rai",
        "datePublished": "2026-11-23",
        "isLiveDataset": False,
        "license": CDLA_PERMISSIVE_2_0_URL,
        "url": "https://anonymous.example/benchmark" if double_blind else "https://openxvs.org/benchmark",
        "creator": {
            "@type": "sc:Organization",
            "name": "Anonymous Authors" if double_blind else "OpenXVS Project Contributors",
        },
        "keywords": [
            "civil-aviation",
            "detect-and-avoid",
            "gnss-denied-navigation",
            "vision-based-landing",
            "runtime-assurance",
            "formal-verification",
            "non-flight-benchmark",
        ],
        "citeAs": (
            "Anonymous Authors (2027). Cryptographically Attested Closed-Loop Benchmark for "
            "Formally Assured Civil Aerial Autonomy. Submitted for double-blind review."
            if double_blind
            else (
                "OpenXVS Project (2026). XVS-Bench v2.1: Cryptographically Attested Closed-Loop "
                "Benchmark for Formally Assured Civil Aerial Autonomy."
            )
        ),
        # ---- MLCommons Croissant RAI 1.0 Responsible AI Metadata ----
        "rai:dataCollection": (
            "Aggregated from strictly civil public aviation perception corpora (Airborne Object Tracking, "
            "LARD 36-airport zero-overlap runway split, USGS 3DEP DEM + NAIP orthophotos, civil LWIR "
            "datasets) combined with deterministic 6-DoF NED closed-loop simulation and Kani 0.68.0 "
            "bounded-model-checking proof artifacts. Strictly excludes weapons, targeting, and military datasets."
        ),
        "rai:dataCollectionType": [
            "Direct Measurement",
            "Simulation",
            "Formal Verification",
        ],
        "rai:dataCollectionMissingData": (
            "Honest gap disclosure: (1) live_sky_imx296_adsb Phase-4 capture slot is pre-wired "
            "(PREWIRED_PHASE4_PENDING_CAPTURE, 0 live frames captured to date); (2) physical flight-test "
            "assurance evidence (G5_GAP) remains explicitly undeveloped; (3) edge silicon latencies on "
            "Raspberry Pi 5 and Jetson Orin NX remain UNVERIFIED until physical hardware measurement; "
            "(4) private hold-out edge-case evaluation split is withheld to prevent benchmark contamination."
        ),
        "rai:dataCollectionRawData": (
            "Public civil aviation imagery and elevation grids under permissive/public-domain licenses "
            "(USGS 3DEP/NAIP public domain, LARD MIT, AOT civil benchmark) plus deterministic synthetic "
            "atmospheric fog/LWIR physics forward models and Rust/C++20 execution logs."
        ),
        "rai:dataCollectionTimeframe": "2025-01/2026-11",
        "rai:dataImputationProtocol": (
            "Zero statistical imputation. Non-finite (NaN/Inf) sensor, bearing, or covariance inputs "
            "trigger deterministic fail-closed safety invariants in the Simplex RTA and ActiveRange cores."
        ),
        "rai:dataManipulationProtocol": (
            "Strict North-East-Down (NED) coordinate normalization (zero internal ENU), zero train/test "
            "airport ICAO overlap (LARD 36-airport strict split), and SHA-256 content-addressed evidence pinning."
        ),
        "rai:dataPreprocessingProtocol": (
            "EO small-target crops scene-normalized to 256x256 float32; LWIR encoded into 2-channel "
            "[log1p radiance, 7x7 high-pass local contrast] at 128x160; runway approach frames resized with "
            "horizon-preserving 256x256 framing for 12-keypoint SimCC and Hesse keylines."
        ),
        "rai:dataAnnotationProtocol": (
            "Automated geometric projection from 3DEP/NED runway coordinates, calibrated SimCC keypoint labels, "
            "and automated Kani CBMC proof verification (18/18 harnesses across 4 no_std Rust crates). "
            "No crowd-worker personal annotations."
        ),
        "rai:dataAnnotationPlatform": (
            "Deterministic offline Python/Rust simulation pipeline, USGS/LARD/AOT geometric metadata, "
            "and Kani 0.68.0 CBMC formal verification harness."
        ),
        "rai:dataAnnotationAnalysis": (
            "Split-conformal safe range bounds (CP-MSC), Student-t heavy-tailed covariance calibration "
            "(nu=6.0, scale=4.214477), Dirichlet vacuity OOD monitoring (exact dual-blind vacuity=1.0), "
            "and formal safety-envelope proofs."
        ),
        "rai:personalSensitiveInformation": (
            "None. Contains zero personal data, zero human face/biometric identifiers, and zero "
            "aircraft registration/owner PII."
        ),
        "rai:dataSocialImpact": (
            "Intended to advance civil unmanned aviation safety (mid-air collision avoidance, GNSS-spoofing "
            "resilience, and degraded-visual-environment landing for civil sub-25 kg BVLOS inspection, "
            "search-and-rescue, wildfire monitoring, and medical logistics). Explicitly prohibits weapons, "
            "targeting, and flight actuation in evaluation artifacts."
        ),
        "rai:dataBiases": (
            "Geographic concentration in North American (California NAIP/KSFO) and European civil airports "
            "(LARD); daytime clear-sky bias in AOT real-sky sequences; seasonal foliage/snow shifts require "
            "multi-season cross-view validation."
        ),
        "rai:dataLimitations": (
            f"{NON_FLIGHT_WATERMARK}. Not certified to DO-178C, DO-178C/ED-324, ASTM F3442-26, or EASA SORA 2.5 "
            "operational approval. Provided for desktop/bench simulation and academic reproducibility only "
            "under EU PLD Directive (EU) 2024/2853 Art. 4(7) non-flight evaluation boundaries."
        ),
        "rai:dataUseCases": (
            "Reproducible academic benchmarking of civil DAA perception, GNSS-denied visual localization, "
            "vision-based precision approach integrity, and formally verified ASTM F3269-21 Simplex RTA."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Maintained under CDLA-Permissive-2.0 with SHA-256 content-addressed manifests, staged "
            "open-core release cadence (formal assurance core followed by full benchmark & weights "
            "post-review), and a documented private hold-out split to guard against leaderboard overfitting."
        ),
        "rai:hasSyntheticData": True,
        "distribution": distribution,
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "track_a_daa_meshdaa",
                "name": "track_a_daa_meshdaa",
                "description": "Track A: Civil DAA / AOT real-data transfer & 1/2/3-ship XVS-MeshDAA closed-loop safety metrics.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "track_a_daa_meshdaa/nmac_count_3ship",
                        "name": "nmac_count_3ship",
                        "dataType": "sc:Integer",
                        "description": "3-ship NMAC count across 500 encounters",
                        "source": {
                            "fileObject": {"@id": meshdaa_src_id},
                            "extract": {"jsonPath": "$.nmac_count_3ship"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_a_daa_meshdaa/lowc_count_3ship",
                        "name": "lowc_count_3ship",
                        "dataType": "sc:Integer",
                        "description": "3-ship Loss of Well Clear count",
                        "source": {
                            "fileObject": {"@id": meshdaa_src_id},
                            "extract": {"jsonPath": "$.lowc_count_3ship"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_a_daa_meshdaa/ghost_suppression_ratio",
                        "name": "ghost_suppression_ratio",
                        "dataType": "sc:Float",
                        "description": "Finite 2-ship vs 3-ship triple-ray ghost suppression ratio",
                        "source": {
                            "fileObject": {"@id": meshdaa_src_id},
                            "extract": {"jsonPath": "$.ghost_suppression_ratio"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_a_daa_meshdaa/bandwidth_kbps_52001",
                        "name": "bandwidth_kbps_52001",
                        "dataType": "sc:Float",
                        "description": "Measured 77 B signed MAVLink v2 #52001 bandwidth",
                        "source": {
                            "fileObject": {"@id": meshdaa_src_id},
                            "extract": {"jsonPath": "$.bandwidth_kbps_52001"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_a_daa_meshdaa/aot_colab_g4_map",
                        "name": "aot_colab_g4_map",
                        "dataType": "sc:Text",
                        "description": "Deferred Colab G4 training metric tag",
                        "source": {
                            "fileObject": {"@id": allowlist_src_id},
                            "extract": {"jsonPath": "$.aot_colab_g4_map"},
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "track_b_nav_antispoof",
                "name": "track_b_nav_antispoof",
                "description": "Track B: XVS-Nav GNSS-denied visual navigation, 6-stage aliasing gate & Bonferroni CUSUM anti-spoofing.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "track_b_nav_antispoof/rectification_max_err_px",
                        "name": "rectification_max_err_px",
                        "dataType": "sc:Float",
                        "description": "ECEF/NED/CAM homography round-trip error",
                        "source": {
                            "fileObject": {"@id": nav_src_id},
                            "extract": {"jsonPath": "$.rectification_max_err_px"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_b_nav_antispoof/aliasing_nonfinite_blocked",
                        "name": "aliasing_nonfinite_blocked",
                        "dataType": "sc:Integer",
                        "description": "Adversarial NaN/Inf inputs fail-closed across 6 stages",
                        "source": {
                            "fileObject": {"@id": nav_src_id},
                            "extract": {"jsonPath": "$.aliasing_nonfinite_blocked"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_b_nav_antispoof/cusum_spoof_trip_latency_ticks",
                        "name": "cusum_spoof_trip_latency_ticks",
                        "dataType": "sc:Integer",
                        "description": "10 Hz ticks from +2.5 m/s East spoof onset to Bank 1 failover",
                        "source": {
                            "fileObject": {"@id": nav_src_id},
                            "extract": {"jsonPath": "$.cusum_spoof_trip_latency_ticks"},
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "track_c_land_vraim",
                "name": "track_c_land_vraim",
                "description": "Track C: XVS-Land + WP-01b 3DEP TAWS + XVS-VRAIM Cat I V-HPL/V-VPL & LARD real H3 isolation.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "track_c_land_vraim/vvpl_cat1_short_final_m",
                        "name": "vvpl_cat1_short_final_m",
                        "dataType": "sc:Float",
                        "description": "Vision-RAIM vertical protection level vs 10.0 m Cat I VAL",
                        "source": {
                            "fileObject": {"@id": land_src_id},
                            "extract": {"jsonPath": "$.vvpl_cat1_short_final_m"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_c_land_vraim/taws_nominal_false_alerts",
                        "name": "taws_nominal_false_alerts",
                        "dataType": "sc:Integer",
                        "description": "TSO-C151d Class B nominal KSFO 28L glide false alert count",
                        "source": {
                            "fileObject": {"@id": land_src_id},
                            "extract": {"jsonPath": "$.taws_nominal_false_alerts"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_c_land_vraim/lard_real_icao_overlap",
                        "name": "lard_real_icao_overlap",
                        "dataType": "sc:Integer",
                        "description": "Train vs LARD real 36-airport ICAO overlap (must be 0)",
                        "source": {
                            "fileObject": {"@id": land_src_id},
                            "extract": {"jsonPath": "$.lard_real_icao_overlap"},
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "track_d_rt_odd",
                "name": "track_d_rt_odd",
                "description": "Track D: C++20 zero-alloc hot loop (-O0/-O2), 18/18 Kani BMC proofs & EASA R-LM-07..11 ODD audits.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "track_d_rt_odd/hot_loop_alloc_bytes_o0",
                        "name": "hot_loop_alloc_bytes_o0",
                        "dataType": "sc:Integer",
                        "description": "Wrapped malloc bytes during 10,000-tick -O0 audit",
                        "source": {
                            "fileObject": {"@id": rt_src_id},
                            "extract": {"jsonPath": "$.hot_loop_alloc_bytes_o0"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_d_rt_odd/hot_loop_alloc_bytes_o2",
                        "name": "hot_loop_alloc_bytes_o2",
                        "dataType": "sc:Integer",
                        "description": "Wrapped malloc bytes during 10,000-tick -O2 audit",
                        "source": {
                            "fileObject": {"@id": rt_src_id},
                            "extract": {"jsonPath": "$.hot_loop_alloc_bytes_o2"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_d_rt_odd/kani_harnesses_passed",
                        "name": "kani_harnesses_passed",
                        "dataType": "sc:Integer",
                        "description": "Real cargo kani passed harness count (18/18 across 4 crates)",
                        "source": {
                            "fileObject": {"@id": rt_src_id},
                            "extract": {"jsonPath": "$.kani_harnesses_passed"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "track_d_rt_odd/native_linux_gpu_wcet_ms",
                        "name": "native_linux_gpu_wcet_ms",
                        "dataType": "sc:Text",
                        "description": "Deferred native Linux GPU TensorRT WCET tag",
                        "source": {
                            "fileObject": {"@id": rt_src_id},
                            "extract": {"jsonPath": "$.native_linux_gpu_wcet_ms"},
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "live_sky_imx296_adsb",
                "name": "live_sky_imx296_adsb",
                "description": "Pre-wired Phase 4 Live-Sky IMX296 + 16mm + FlightAware/ADSBexchange 1090 MHz ADS-B capture slot (PREWIRED_PHASE4_PENDING_CAPTURE).",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "live_sky_imx296_adsb/mode_s_tc19_decoded",
                        "name": "mode_s_tc19_decoded",
                        "dataType": "sc:Boolean",
                        "description": "Mode S Type Code 19 geometric-baro offset decoded",
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "live_sky_imx296_adsb/nic",
                        "name": "nic",
                        "dataType": "sc:Integer",
                        "description": "ADS-B Navigation Integrity Category (gate >= 7)",
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "live_sky_imx296_adsb/nacp",
                        "name": "nacp",
                        "dataType": "sc:Integer",
                        "description": "ADS-B Navigation Accuracy Category Position (gate >= 9)",
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "live_sky_imx296_adsb/boresight_rms_deg",
                        "name": "boresight_rms_deg",
                        "dataType": "sc:Float",
                        "description": "Camera-to-NED boresight calibration RMS (gate <= 0.15 deg)",
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "live_sky_imx296_adsb/pps_synced",
                        "name": "pps_synced",
                        "dataType": "sc:Boolean",
                        "description": "GPS 1PPS hardware XTR exposure synchronization lock",
                    },
                ],
            },
        ],
        f"{ns_prefix}:civilScope": {
            "civil_and_dual_use_only": True,
            "weapons_or_targeting_excluded": True,
            "non_flight_sim_only": True,
            "actuator_authority": False,
            "watermark": NON_FLIGHT_WATERMARK,
            "regulatory_disclaimer": (
                "Architected for IDAL C learned perception bounded by a deterministic ASTM F3269-21 "
                "Simplex safety envelope whose discrete switching, quantization, and past-time MLTL "
                "ODD logic is bounded-model-checked via Kani (18/18 harnesses)."
            ),
        },
        f"{ns_prefix}:phase4LiveSkySlot": {
            "split": "live_sky_imx296_adsb",
            "status": "PREWIRED_PHASE4_PENDING_CAPTURE",
            "hardware_spec": {
                "camera": "Sony IMX296 1456x1088 global-shutter (3.45 um)",
                "optics": "16mm C-mount telephoto lens (17.8x13.4 deg FOV, 0.216 mrad/px)",
                "sdr": "FlightAware Pro Stick Plus / ADSBexchange 1090 MHz filtered SDR",
                "excluded_eol_hardware": [
                    "RTL-SDR Blog V4 (EOL 2026-05-10)",
                ],
                "imu": "TDK InvenSense ICM-42688-P 6-DoF SPI MEMS IMU",
                "timebase": "u-blox NEO-M8N 1PPS + Raspberry Pi Pico XTR exposure trigger",
            },
            "adsb_gates": {
                "mode_s_tc19_decoded_required": True,
                "nic_min": 7,
                "nacp_min": 9,
                "boresight_rms_deg_max": 0.15,
                "pps_synced_required": True,
                "max_exposure_latency_ms": 5.0,
            },
            "target_airliner_captures_phase4": 50,
            "captured_frames_count": 0,
        },
    }

    if double_blind:
        doc = _scrub_recursive(doc)

    validate_croissant_rai_document(doc, double_blind=double_blind)
    return doc


def validate_croissant_rai_document(doc: dict[str, Any], *, double_blind: bool = False) -> None:
    """Validate Croissant 1.1 + RAI schema invariants and double-blind scrubbing."""
    if doc.get("@type") != "sc:Dataset":
        raise ValueError(f"Expected @type == 'sc:Dataset', got {doc.get('@type')!r}")

    conforms = doc.get("conformsTo")
    conforms_list = conforms if isinstance(conforms, list) else [conforms]
    if CROISSANT_1_1_URI not in conforms_list:
        raise ValueError(f"Missing Croissant 1.1 conformance URI {CROISSANT_1_1_URI!r}")
    if CROISSANT_RAI_1_0_URI not in conforms_list:
        raise ValueError(f"Missing Croissant RAI 1.0 conformance URI {CROISSANT_RAI_1_0_URI!r}")

    if doc.get("license") != CDLA_PERMISSIVE_2_0_URL:
        raise ValueError(f"Expected CDLA-Permissive-2.0 license URL, got {doc.get('license')!r}")

    ctx = doc.get("@context")
    if not isinstance(ctx, dict) or ctx.get("rai") != "http://mlcommons.org/croissant/RAI/":
        raise ValueError("Missing MLCommons RAI @context namespace binding")

    for k in REQUIRED_RAI_KEYS:
        if k not in doc:
            raise ValueError(f"Missing required Croissant RAI field: {k}")

    dist = doc.get("distribution")
    if not isinstance(dist, list) or len(dist) < 16:
        raise ValueError("Expected >= 16 cr:FileObject entries in distribution")
    dist_ids = {str(item["@id"]) for item in dist}

    rsets = doc.get("recordSet")
    if not isinstance(rsets, list) or len(rsets) < 5:
        raise ValueError("Expected >= 5 cr:RecordSet entries")
    for rs in rsets:
        for fld in rs.get("field", []):
            src = fld.get("source")
            if isinstance(src, dict) and "fileObject" in src:
                ref_id = src["fileObject"].get("@id")
                if ref_id not in dist_ids:
                    raise ValueError(f"Dangling FileObject reference {ref_id!r} in field {fld.get('@id')!r}")

    if double_blind:
        serialized = json.dumps(doc, ensure_ascii=False)
        for pat in DOUBLE_BLIND_FORBIDDEN_PATTERNS:
            m = pat.search(serialized)
            if m is not None:
                raise ValueError(
                    f"Double-blind leak detected matching {pat.pattern!r}: {m.group(0)!r}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MLCommons Croissant 1.1 + RAI (CDLA-Permissive-2.0) benchmark metadata."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output JSON-LD path (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for SHA-256 verification of distribution files",
    )
    parser.add_argument(
        "--double-blind",
        action="store_true",
        help="Scrub OpenXVS/author/host identifiers for NeurIPS 2027 E&D double-blind review",
    )
    args = parser.parse_args()

    doc = build_croissant_rai_metadata(args.repo_root, double_blind=bool(args.double_blind))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mode_lbl = "double-blind" if args.double_blind else "canonical"
    print(
        f"[croissant-1.1+rai] wrote {args.out} ({mode_lbl}, license=CDLA-Permissive-2.0, "
        f"files={len(doc['distribution'])}, recordSets={len(doc['recordSet'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
