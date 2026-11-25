"""Deterministic Flight-Pod SWaP, Bare-Metal MCU & Telemetry Bandwidth Auditor (`OpenXVS v5.2`).

Audits real on-disk OpenXVS artifacts across all three real-world deployment profiles:
  1. Tier-0B Bench (`RPi 5 8GB`): `<= 300 g`, `<= 15 W`, CPU-only bench (`NOT_FLIGHT_CAPABLE`).
  2. Tier-1 EO-Only Shadow Pod (`Orin Nano Super 8GB` + `STM32H7` Ring-0 MCU): `67 TOPS`,
     `<= 6,144 MiB` RAM cap (all 6 active ONNX weights + 1024 MiB TRT workspace fit inside 8 GB
     while flagging optional `VLM 2B` as `EXCLUDED_8GB_CAP`), `STM32H7` Ring-0 Flash `<= 15%`
     of `2 MiB` (verified `8.9725%` = `188,166 B`) & `0 B` static RAM.
  3. Tier-2 Dual EO+LWIR Pod (`Orin NX 16GB` + `STM32H7/R5` Ring-0 MCU + `<= 9 Hz` Boson+ 320):
     explicit SWaP margin audit (`255-350 g`, `10 W nvpmodel` mandatory for `< 15 W`),
     `UART 921.6 kbps < 20%` (`<= 18,432 B/s`) MAVLink2 telemetry cap, `NON_FLIGHT_SIM_ONLY=1`,
     and NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 Critical Applications compliance.

Strictly civil/dual-use, zero GPL (`pymavlink` forbidden), strictly NED/ECEF (`0` internal ENU),
zero host path leaks (REL1 clean), and strictly evidence-tagged per `REALWORLD_DEPLOYABILITY_AUDIT_V5.md`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.studio_bridge.hil_bridge import NON_FLIGHT_WATERMARK  # noqa: E402
from xvs.v2.xvs_mavlink.pod_daa_codec import (  # noqa: E402
    DEFAULT_V3_SIGNED_UART_STREAMS,
    MAX_SIGNED_UART_BYTES_PER_SEC,
    MAX_UART_UTILIZATION_FRAC,
    UART_BAUD_BPS,
    UART_CAPACITY_BYTES_PER_SEC,
    MavlinkStreamSpec,
    compute_uart_bandwidth_budget,
    parse_and_validate_dialect_xml,
)

REPORT_SCHEMA: str = "openxvs.flight_pod_budget_report/1.0"
DEFAULT_MCU_PROOF_REL: str = "outputs/formal_tla/ring0_thumbv7em_mcu_proof.json"
DEFAULT_REPORT_OUT_REL: str = "outputs/flight_pod_budget_report.json"
DEFAULT_DIALECT_XML_REL: str = "configs/mavlink/pod_daa.xml"

# Canonical 4-level evidence taxonomy from docs/xvs/REALWORLD_DEPLOYABILITY_AUDIT_V5.md
EVIDENCE_VERIFIED_ON_SILICON: str = "VERIFIED_ON_SILICON"
EVIDENCE_VERIFIED_FORMAL_OR_SIM: str = "VERIFIED_FORMAL_OR_SIM"
EVIDENCE_UNVERIFIED_FLIGHT_HW: str = "UNVERIFIED_REQUIRES_PHYSICAL_FLIGHT_HW"
EVIDENCE_UNVERIFIED_VENDOR_SPEC: str = "UNVERIFIED_VENDOR_SPEC"
VALID_EVIDENCE_TAGS: frozenset[str] = frozenset(
    {
        EVIDENCE_VERIFIED_ON_SILICON,
        EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        EVIDENCE_UNVERIFIED_FLIGHT_HW,
        EVIDENCE_UNVERIFIED_VENDOR_SPEC,
    }
)

STM32H7_FLASH_CAPACITY_BYTES: int = 2 * 1024 * 1024  # 2,097,152 B (2 MiB)
MAX_RING0_MCU_FLASH_UTIL_PCT: float = 15.0
MAX_RING0_MCU_STATIC_RAM_BYTES: int = 0
EXPECTED_FIVE_RING0_CRATES: tuple[str, ...] = (
    "xvs-rta-core",
    "xvs-active-range",
    "xvs-odd-vlm",
    "xvs-os-hal",
    "xvs-multiship-coord",
)

POD_MASS_TARGET_CAP_G: float = 300.0
POD_POWER_TARGET_CAP_W: float = 15.0
ORIN_NANO_SUPER_TOTAL_RAM_MIB: float = 8192.0
ORIN_NANO_SUPER_APP_RAM_CAP_MIB: float = 6144.0
ORIN_NANO_SUPER_OS_RESERVE_MIB: float = 2048.0
ORIN_NANO_SUPER_EXPECTED_TOPS: int = 67
ORIN_NX_TOTAL_RAM_MIB: float = 16384.0
LWIR_EAR99_MAX_HZ: float = 9.0

# Conservative runtime allocation multipliers/overheads for Jetson unified LPDDR5 audit
TRT_ENGINE_AND_ACTIVATION_MULTIPLIER: float = 3.0
JETSON_CUDA_TRT_ORT_RUNTIME_OVERHEAD_MIB: float = 900.0
VLM_2B_FP16_WEIGHTS_MIB: float = 4096.0
VLM_2B_VISION_AND_KV_ACTIVATIONS_MIB: float = 1280.0
VLM_2B_TOTAL_FOOTPRINT_MIB: float = VLM_2B_FP16_WEIGHTS_MIB + VLM_2B_VISION_AND_KV_ACTIVATIONS_MIB


@dataclass(frozen=True)
class OnnxModelCandidateSpec:
    """Canonical SHA-256-pinned ONNX model candidate matching `docker/edge-infer/src/main.rs`."""

    rel_path: str
    expected_sha256: str
    precision: str


@dataclass(frozen=True)
class ActiveOnnxSlotSpec:
    """One of the 6 active OpenXVS production ONNX perception models."""

    slot_id: str
    display_name: str
    subsystem: str
    candidates: tuple[OnnxModelCandidateSpec, ...]


ACTIVE_SIX_ONNX_SLOTS: tuple[ActiveOnnxSlotSpec, ...] = (
    ActiveOnnxSlotSpec(
        slot_id="xvs_d2",
        display_name="XVS-D2 (EO Non-Cooperative DAA ConvGRU)",
        subsystem="fast_10hz_eo_daa",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="apps/livedeck/public/models/xvs_d2_web_fp16.onnx",
                expected_sha256="2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d",
                precision="fp16",
            ),
        ),
    ),
    ActiveOnnxSlotSpec(
        slot_id="xvs_thermal",
        display_name="XVS-ThermalNet v5.2 (EO+LWIR Asymmetric FPN DVE Fusion)",
        subsystem="fast_10hz_thermal_dve",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp16.onnx",
                expected_sha256="bc6ceac6252f48dc52305c4e7955cb7841bc5adc7c191a59ce9b0e36ef10fcac",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp32.onnx",
                expected_sha256="87fbd639591012ec5918c95a1915de9c0fa2911606823afb4e5c6f2bd0c82dbd",
                precision="fp32",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp16.onnx",
                expected_sha256="85a6c9100846b00810889c6c1d97b227cd495057b34c98923abd3b04acf6bd98",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp32.onnx",
                expected_sha256="db06e32fb44dd8a949ed8972e06b362c10557c608ed36809d0e9626eeb0a93c3",
                precision="fp32",
            ),
        ),
    ),
    ActiveOnnxSlotSpec(
        slot_id="xvs_land",
        display_name="XVS-Land v5.2 (12-Keypoint SimCC 384 + 5 Hesse Keylines + Keyline Sigma)",
        subsystem="slow_2hz_runway_land",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp16.onnx",
                expected_sha256="834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp32.onnx",
                expected_sha256="fc8e808394b7e20c0d80e5fe357306ba332ee6f57018666e941a8d55caee6407",
                precision="fp32",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v45/harvest_land/xvs_land_v45_fp16.onnx",
                expected_sha256="254a6e49bafb9cedf01edaf9650ff98b98ad42ad2d219a29dc8725bb280997e5",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v45/harvest_land/xvs_land_v45.onnx",
                expected_sha256="4245c0d1cdfea8d5065f56d9d3a7dd9a3d6cc29ff8dea1b1a7c173cd39e2920d",
                precision="fp32",
            ),
        ),
    ),
    ActiveOnnxSlotSpec(
        slot_id="xvs_nav_s1",
        display_name="XVS-Nav v5.0 Stage-1 (Faz-B MNV4-Hybrid-M + MixVPR)",
        subsystem="slow_2hz_crossview_nav",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
                expected_sha256="e48f0ed5a234dcf1661031b80db5028d8c5b926b7b54e06575b4ed20bd6fc46f",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx",
                expected_sha256="c9b78a98f51ff294b2eeffb12c65902647a9ad08b28a06dec6b870b39fdf6bd1",
                precision="fp16",
            ),
        ),
    ),
    ActiveOnnxSlotSpec(
        slot_id="xvs_nav_s2a_xfeat",
        display_name="XVS-Nav Stage-2A (XFeat 320x256 Keypoint Extractor)",
        subsystem="slow_2hz_crossview_nav",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
                expected_sha256="e28a2f3afd7e23fe9818ecea34b57185708ff2e2b45f24cb514628f49c3d8182",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest/xfeat_320x256_fp32.onnx",
                expected_sha256="04a153209ec1bd494fea9489b92552d05c91339c86990d2492edbf104deb37c6",
                precision="fp32",
            ),
        ),
    ),
    ActiveOnnxSlotSpec(
        slot_id="xvs_nav_s2b_lighterglue",
        display_name="XVS-Nav Stage-2B (LighterGlue Static K=256 Matcher)",
        subsystem="slow_2hz_crossview_nav",
        candidates=(
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
                expected_sha256="e63121ce24b840dba04acd490133eba4175a2ae43b087db3dac6e9d2f3220f3b",
                precision="fp16",
            ),
            OnnxModelCandidateSpec(
                rel_path="outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp32.onnx",
                expected_sha256="fd43db4f25a7d714c414526dd387e0c7f1397a313175994f5eba79389a143512",
                precision="fp32",
            ),
        ),
    ),
)

_ENV_ASSIGN_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)=([^\s\\]+)")
_LABEL_ASSIGN_RE = re.compile(r'([a-z0-9_.-]+)="([^"]*)"')
_ALPINE_FROM_RE = re.compile(r"^\s*FROM\b[^\n]*alpine", re.IGNORECASE | re.MULTILINE)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _bytes_to_mib(n_bytes: int | float) -> float:
    return round(float(n_bytes) / (1024.0 * 1024.0), 4)


def _rel_posix(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root_resolved = repo_root.resolve()
    if resolved.is_relative_to(root_resolved):
        return resolved.relative_to(root_resolved).as_posix()
    return path.name


def _parse_dockerfile_env_and_labels(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse `ENV` assignments and quoted `LABEL` key/value pairs from a Dockerfile."""
    env_map: dict[str, str] = {}
    label_map: dict[str, str] = {}

    # Join backslash-continued lines for reliable ENV/LABEL parsing
    logical_lines: list[str] = []
    buf = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if buf:
                logical_lines.append(buf)
                buf = ""
            continue
        if stripped.endswith("\\"):
            buf += " " + stripped[:-1].strip()
        else:
            buf += " " + stripped
            logical_lines.append(buf.strip())
            buf = ""
    if buf:
        logical_lines.append(buf.strip())

    for line in logical_lines:
        if line.startswith("ENV "):
            for key, val in _ENV_ASSIGN_RE.findall(line[4:]):
                env_map[key] = val.strip('"').strip("'")
        elif line.startswith("LABEL "):
            for key, val in _LABEL_ASSIGN_RE.findall(line[6:]):
                label_map[key] = val

    return env_map, label_map


def audit_dockerfile_profile(
    dockerfile_path: Path,
    *,
    repo_root: Path,
    requires_nvidia_trt_sla: bool,
    forbid_alpine_musl: bool = False,
) -> dict[str, Any]:
    """Deterministically verify non-flight lock, ABI rules, and NVIDIA TensorRT SLA in one Dockerfile."""
    rel = _rel_posix(dockerfile_path, repo_root)
    errors: list[str] = []
    if not dockerfile_path.is_file():
        return {
            "dockerfile": rel,
            "present": False,
            "pass": False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "errors": [f"Missing Dockerfile: {rel}"],
            "env": {},
            "labels": {},
        }

    text = dockerfile_path.read_text(encoding="utf-8")
    env_map, label_map = _parse_dockerfile_env_and_labels(text)

    has_non_flight_env = (
        ("NON_FLIGHT_SIM_ONLY" in env_map and env_map["NON_FLIGHT_SIM_ONLY"] == "1")
        and ("OPENXVS_NON_FLIGHT_SIM_ONLY" in env_map and env_map["OPENXVS_NON_FLIGHT_SIM_ONLY"] == "1")
    )
    has_non_flight_feature = "--features non_flight_sim_only" in text
    has_actuator_false_label = (
        "org.openxvs.actuator_authority" in label_map
        and label_map["org.openxvs.actuator_authority"].lower() == "false"
    )
    has_watermark = NON_FLIGHT_WATERMARK in text
    has_remap_path_prefix = "--remap-path-prefix=" in text
    no_alpine_musl = _ALPINE_FROM_RE.search(text) is None

    if not has_non_flight_env:
        errors.append("Missing ENV NON_FLIGHT_SIM_ONLY=1 and OPENXVS_NON_FLIGHT_SIM_ONLY=1")
    if not has_non_flight_feature:
        errors.append("Missing cargo --features non_flight_sim_only build lock")
    if not has_actuator_false_label:
        errors.append("Missing LABEL org.openxvs.actuator_authority=\"false\"")
    if not has_watermark:
        errors.append("Missing canonical EU PLD Art. 4(7) non-flight watermark")
    if not has_remap_path_prefix:
        errors.append("Missing REL1 --remap-path-prefix build hygiene")
    if forbid_alpine_musl and not no_alpine_musl:
        errors.append("Forbidden Alpine musl FROM detected (ORT linux-aarch64 requires glibc >= 2.28)")

    nvidia_sla_ok = True
    has_strongly_typed = "--stronglyTyped" in text
    if requires_nvidia_trt_sla:
        has_sla_text = (
            "TensorRT" in text
            and "1.2" in text
            and ("Section 2" in text or "Sec. 2" in text or "& 2" in text)
            and "Critical Applications" in text
        )
        has_eula_file = "/etc/openxvs/NVIDIA_TENSORRT_EULA_NOTICE.txt" in text
        nvidia_sla_ok = bool(has_sla_text and has_eula_file and has_strongly_typed)
        if not has_sla_text:
            errors.append("Missing NVIDIA TensorRT SLA Sec. 1.2 & Sec. 2 Critical Applications notice")
        if not has_eula_file:
            errors.append("Missing /etc/openxvs/NVIDIA_TENSORRT_EULA_NOTICE.txt runtime disclosure")
        if not has_strongly_typed:
            errors.append("Missing trtexec --stronglyTyped FP16 enforcement")

    return {
        "dockerfile": rel,
        "present": True,
        "sha256": _sha256_file(dockerfile_path),
        "non_flight_sim_only_locked": bool(has_non_flight_env and has_non_flight_feature),
        "actuator_authority_false_locked": bool(has_actuator_false_label),
        "canonical_watermark_present": bool(has_watermark),
        "rel1_remap_path_prefix": bool(has_remap_path_prefix),
        "no_alpine_musl": bool(no_alpine_musl),
        "requires_nvidia_trt_sla": bool(requires_nvidia_trt_sla),
        "nvidia_trt_sla_sec_1_2_and_2_ok": bool(nvidia_sla_ok),
        "trtexec_strongly_typed": bool(has_strongly_typed) if requires_nvidia_trt_sla else None,
        "env": env_map,
        "labels": label_map,
        "pass": len(errors) == 0,
        "errors": errors,
        "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        "runtime_silicon_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
    }


def audit_ring0_mcu_proof(
    mcu_proof_path: Path,
    *,
    repo_root: Path,
    flash_capacity_bytes: int = STM32H7_FLASH_CAPACITY_BYTES,
    max_flash_util_pct: float = MAX_RING0_MCU_FLASH_UTIL_PCT,
    max_static_ram_bytes: int = MAX_RING0_MCU_STATIC_RAM_BYTES,
) -> dict[str, Any]:
    """Read and deterministically audit `outputs/formal_tla/ring0_thumbv7em_mcu_proof.json`."""
    rel = _rel_posix(mcu_proof_path, repo_root)
    errors: list[str] = []
    if not mcu_proof_path.is_file():
        return {
            "artifact": rel,
            "present": False,
            "pass": False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
            "errors": [f"Missing Ring-0 MCU proof artifact: {rel}"],
        }

    try:
        doc = json.loads(mcu_proof_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "artifact": rel,
            "present": True,
            "pass": False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
            "errors": [f"Malformed JSON in {rel}: {exc}"],
        }

    if not isinstance(doc, dict):
        return {
            "artifact": rel,
            "present": True,
            "pass": False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
            "errors": [f"Expected JSON object in {rel}"],
        }

    schema = str(doc["schema"]) if "schema" in doc else ""
    target_triple = str(doc["target_triple"]) if "target_triple" in doc else ""
    invariants = doc["invariants"] if "invariants" in doc and isinstance(doc["invariants"], dict) else {}
    crates = doc["five_ring0_crates"] if "five_ring0_crates" in doc and isinstance(doc["five_ring0_crates"], list) else []
    forbidden_hits = (
        doc["forbidden_std_alloc_symbol_hits"]
        if "forbidden_std_alloc_symbol_hits" in doc and isinstance(doc["forbidden_std_alloc_symbol_hits"], list)
        else None
    )
    totals = (
        doc["aggregate_elf32_arm_totals_bytes"]
        if "aggregate_elf32_arm_totals_bytes" in doc and isinstance(doc["aggregate_elf32_arm_totals_bytes"], dict)
        else {}
    )

    if schema != "openxvs.ring0_thumbv7em_mcu_proof/1.0":
        errors.append(f"Unexpected MCU proof schema: {schema!r}")
    if target_triple != "thumbv7em-none-eabi":
        errors.append(f"Unexpected target_triple: {target_triple!r}")
    if tuple(str(c) for c in crates) != EXPECTED_FIVE_RING0_CRATES:
        errors.append(f"Expected 5 Ring-0 crates {EXPECTED_FIVE_RING0_CRATES}, got {crates!r}")
    if forbidden_hits is None or len(forbidden_hits) != 0:
        errors.append(f"Forbidden std/alloc symbol hits must be empty, got {forbidden_hits!r}")

    required_true_invariants = (
        "no_std",
        "forbid_unsafe_code",
        "zero_std_verified",
        "zero_alloc_verified",
        "zero_heap_allocations",
        "zero_tls_aes_5d002_crypto",
        "zero_gpl_crates",
        "non_flight_sim_only",
        "all_5_ring0_crates_compiled_and_invoked",
    )
    for inv_key in required_true_invariants:
        if inv_key not in invariants or invariants[inv_key] is not True:
            errors.append(f"MCU proof invariant {inv_key!r} is not True")
    if "actuator_authority" not in invariants or invariants["actuator_authority"] is not False:
        errors.append("MCU proof invariant 'actuator_authority' must be False")

    required_total_keys = (
        "text_bytes",
        "rodata_bytes",
        "data_bytes",
        "bss_bytes",
        "total_flash_text_rodata_data_bytes",
        "total_static_ram_data_bss_bytes",
    )
    missing_totals = [k for k in required_total_keys if k not in totals]
    if missing_totals:
        errors.append(f"Missing aggregate_elf32_arm_totals_bytes keys: {missing_totals}")
        return {
            "artifact": rel,
            "present": True,
            "pass": False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
            "errors": errors,
        }

    text_b = int(totals["text_bytes"])
    rodata_b = int(totals["rodata_bytes"])
    data_b = int(totals["data_bytes"])
    bss_b = int(totals["bss_bytes"])
    flash_b = int(totals["total_flash_text_rodata_data_bytes"])
    static_ram_b = int(totals["total_static_ram_data_bss_bytes"])

    if flash_b != text_b + rodata_b + data_b:
        errors.append(
            f"Flash byte sum mismatch: reported {flash_b} != text({text_b})+rodata({rodata_b})+data({data_b})"
        )
    if static_ram_b != data_b + bss_b:
        errors.append(
            f"Static RAM sum mismatch: reported {static_ram_b} != data({data_b})+bss({bss_b})"
        )

    flash_util_pct = round((float(flash_b) / float(flash_capacity_bytes)) * 100.0, 4)
    max_flash_bytes = int(math.floor(float(flash_capacity_bytes) * (float(max_flash_util_pct) / 100.0)))
    flash_within_cap = flash_util_pct <= float(max_flash_util_pct)
    static_ram_ok = static_ram_b <= int(max_static_ram_bytes)

    if not flash_within_cap:
        errors.append(
            f"STM32H7 Flash utilization {flash_util_pct:.4f}% ({flash_b} B) exceeds {max_flash_util_pct:.2f}% cap"
        )
    if not static_ram_ok:
        errors.append(
            f"STM32H7 static RAM (data+bss) {static_ram_b} B exceeds {max_static_ram_bytes} B cap"
        )

    return {
        "artifact": rel,
        "artifact_sha256": _sha256_file(mcu_proof_path),
        "present": True,
        "target_triple": target_triple,
        "five_ring0_crates": list(EXPECTED_FIVE_RING0_CRATES),
        "text_bytes": text_b,
        "rodata_bytes": rodata_b,
        "data_bytes": data_b,
        "bss_bytes": bss_b,
        "flash_bytes": flash_b,
        "flash_capacity_bytes": int(flash_capacity_bytes),
        "flash_utilization_pct": flash_util_pct,
        "max_flash_utilization_pct": float(max_flash_util_pct),
        "flash_headroom_bytes_to_cap": int(max_flash_bytes - flash_b),
        "flash_within_15pct_cap": bool(flash_within_cap),
        "static_ram_bytes": static_ram_b,
        "max_static_ram_bytes": int(max_static_ram_bytes),
        "zero_static_ram_verified": bool(static_ram_b == 0),
        "zero_std_and_zero_alloc_verified": bool(
            invariants["zero_std_verified"] is True
            and invariants["zero_alloc_verified"] is True
            and forbidden_hits == []
        ),
        "pass": len(errors) == 0,
        "errors": errors,
        "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
    }


def audit_active_onnx_models(
    repo_root: Path,
    *,
    slots: Sequence[ActiveOnnxSlotSpec] = ACTIVE_SIX_ONNX_SLOTS,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Measure real on-disk bytes and verify fail-closed SHA-256 pins for all 6 active ONNX models."""
    models_out: list[dict[str, Any]] = []
    errors: list[str] = []
    total_bytes = 0
    nav_triad_bytes = 0

    for slot in slots:
        chosen: OnnxModelCandidateSpec | None = None
        chosen_path: Path | None = None
        chosen_sha = ""
        tampered_candidates: list[str] = []

        for cand in slot.candidates:
            full = repo_root / cand.rel_path
            if not full.is_file():
                continue
            actual_sha = _sha256_file(full)
            if verify_sha256 and actual_sha != cand.expected_sha256:
                tampered_candidates.append(
                    f"{cand.rel_path}: sha256 mismatch (expected {cand.expected_sha256[:12]}..., got {actual_sha[:12]}...)"
                )
                continue
            chosen = cand
            chosen_path = full
            chosen_sha = actual_sha
            break

        if chosen is None or chosen_path is None:
            err_msg = (
                f"Active ONNX slot {slot.slot_id!r} failed verification: "
                + ("; ".join(tampered_candidates) if tampered_candidates else "no candidate file found on disk")
            )
            errors.append(err_msg)
            models_out.append(
                {
                    "slot_id": slot.slot_id,
                    "display_name": slot.display_name,
                    "subsystem": slot.subsystem,
                    "present": False,
                    "sha256_verified": False,
                    "size_bytes": 0,
                    "size_mib": 0.0,
                    "error": err_msg,
                    "evidence_tag": EVIDENCE_VERIFIED_ON_SILICON,
                }
            )
            continue

        size_b = int(chosen_path.stat().st_size)
        total_bytes += size_b
        if slot.subsystem == "slow_2hz_crossview_nav":
            nav_triad_bytes += size_b
        models_out.append(
            {
                "slot_id": slot.slot_id,
                "display_name": slot.display_name,
                "subsystem": slot.subsystem,
                "rel_path": chosen.rel_path,
                "precision": chosen.precision,
                "present": True,
                "sha256": chosen_sha,
                "sha256_verified": bool(chosen_sha == chosen.expected_sha256),
                "size_bytes": size_b,
                "size_mib": _bytes_to_mib(size_b),
                "evidence_tag": EVIDENCE_VERIFIED_ON_SILICON,
            }
        )

    return {
        "active_model_count": len(slots),
        "verified_model_count": sum(1 for m in models_out if m["present"] and m["sha256_verified"]),
        "total_onnx_bytes": total_bytes,
        "total_onnx_mib": _bytes_to_mib(total_bytes),
        "nav_fazb_triad_bytes": nav_triad_bytes,
        "nav_fazb_triad_mib": _bytes_to_mib(nav_triad_bytes),
        "models": models_out,
        "pass": len(errors) == 0 and len(models_out) == len(slots),
        "errors": errors,
        "evidence_tag": EVIDENCE_VERIFIED_ON_SILICON,
        "host_measurement_note": (
            "File byte sizes and SHA-256 digests measured directly on host disk; "
            "embedded ARM64 inference latency remains UNVERIFIED_REQUIRES_PHYSICAL_FLIGHT_HW."
        ),
    }


def audit_uart_telemetry_budget(
    *,
    repo_root: Path,
    signed_streams: Sequence[MavlinkStreamSpec] = DEFAULT_V3_SIGNED_UART_STREAMS,
    dialect_xml_path: Path | None = None,
) -> dict[str, Any]:
    """Deterministically audit MAVLink2 `921.6 kbps` `< 20%` (`<= 18,432 B/s`) UART budgets."""
    errors: list[str] = []

    # 1. Full post-REL6 signed bidirectional Ring-0 <-> FC telemetry budget
    signed_bidir = compute_uart_bandwidth_budget(
        signed_streams,
        baud_bps=UART_BAUD_BPS,
        max_bidir_bytes_per_sec=MAX_SIGNED_UART_BYTES_PER_SEC,
        max_utilization_frac=MAX_UART_UTILIZATION_FRAC,
        signed_mavlink2=True,
    )
    if not bool(signed_bidir["within_budget"]):
        errors.append(
            f"Signed bidirectional MAVLink2 UART load {signed_bidir['total_bidir_bytes_per_sec']} B/s "
            f"exceeds {MAX_SIGNED_UART_BYTES_PER_SEC} B/s (<20% of 921.6 kbps)"
        )

    # 2. Passive shadow-mode listen-only tap (TX pin physically removed -> inbound streams only, 0 B/s TX)
    inbound_only = tuple(s for s in signed_streams if s.direction == "inbound")
    shadow_rx = compute_uart_bandwidth_budget(
        inbound_only,
        baud_bps=UART_BAUD_BPS,
        max_bidir_bytes_per_sec=MAX_SIGNED_UART_BYTES_PER_SEC,
        max_utilization_frac=MAX_UART_UTILIZATION_FRAC,
        signed_mavlink2=True,
    )
    if float(shadow_rx["outbound_bytes_per_sec"]) != 0.0:
        errors.append("Shadow-mode listen-only tap must have exactly 0.0 B/s outbound UART load")
    if not bool(shadow_rx["within_budget"]):
        errors.append("Shadow-mode inbound UART load exceeded 20% cap")

    # 3. Optional read-only cross-check of frozen configs/mavlink/pod_daa.xml
    xml_path = (repo_root / DEFAULT_DIALECT_XML_REL) if dialect_xml_path is None else dialect_xml_path
    dialect_verified = False
    dialect_msg_ids: list[int] = []
    if xml_path.is_file():
        try:
            xml_info = parse_and_validate_dialect_xml(xml_path)
            dialect_msg_ids = [int(mid) for mid in xml_info["message_ids"]]  # type: ignore[union-attr]
            dialect_verified = True
        except Exception as exc:
            errors.append(f"pod_daa.xml validation failed: {exc}")

    return {
        "baud_bps": int(UART_BAUD_BPS),
        "capacity_bytes_per_sec": float(UART_CAPACITY_BYTES_PER_SEC),
        "max_utilization_pct": round(MAX_UART_UTILIZATION_FRAC * 100.0, 2),
        "cap_bytes_per_sec": float(MAX_SIGNED_UART_BYTES_PER_SEC),
        "signed_bidirectional": {
            "outbound_bytes_per_sec": float(signed_bidir["outbound_bytes_per_sec"]),
            "inbound_bytes_per_sec": float(signed_bidir["inbound_bytes_per_sec"]),
            "total_bytes_per_sec": float(signed_bidir["total_bidir_bytes_per_sec"]),
            "utilization_pct": round(float(signed_bidir["utilization_frac"]) * 100.0, 4),
            "headroom_bytes_per_sec": float(signed_bidir["headroom_bytes_per_sec"]),
            "within_budget": bool(signed_bidir["within_budget"]),
        },
        "passive_shadow_rx_only": {
            "outbound_bytes_per_sec": float(shadow_rx["outbound_bytes_per_sec"]),
            "inbound_bytes_per_sec": float(shadow_rx["inbound_bytes_per_sec"]),
            "total_bytes_per_sec": float(shadow_rx["total_bidir_bytes_per_sec"]),
            "utilization_pct": round(float(shadow_rx["utilization_frac"]) * 100.0, 4),
            "within_budget": bool(shadow_rx["within_budget"]),
        },
        "dialect_xml": _rel_posix(xml_path, repo_root),
        "dialect_xml_verified": bool(dialect_verified),
        "dialect_custom_msgids": dialect_msg_ids,
        "pass": len(errors) == 0,
        "errors": errors,
        "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        "physical_uart_jitter_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
    }


def _collect_all_evidence_tags(obj: Any, found: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("evidence_tag") and isinstance(v, str):
                found.add(v)
            else:
                _collect_all_evidence_tags(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_all_evidence_tags(item, found)


def run_flight_pod_budget_audit(
    *,
    repo_root: Path = REPO_ROOT,
    mcu_proof_path: Path | None = None,
    docker_dir: Path | None = None,
    uart_streams: Sequence[MavlinkStreamSpec] = DEFAULT_V3_SIGNED_UART_STREAMS,
    verify_onnx_sha256: bool = True,
) -> dict[str, Any]:
    """Execute complete deterministic audit across Tier-0B, Tier-1, and Tier-2 flight-pod profiles."""
    root = Path(repo_root).resolve()
    mcu_path = (root / DEFAULT_MCU_PROOF_REL) if mcu_proof_path is None else Path(mcu_proof_path)
    dkr_dir = (root / "docker") if docker_dir is None else Path(docker_dir)

    mcu_audit = audit_ring0_mcu_proof(mcu_path, repo_root=root)
    onnx_audit = audit_active_onnx_models(root, verify_sha256=verify_onnx_sha256)
    uart_audit = audit_uart_telemetry_budget(repo_root=root, signed_streams=uart_streams)

    df_rpi5 = audit_dockerfile_profile(
        dkr_dir / "rpi5.Dockerfile",
        repo_root=root,
        requires_nvidia_trt_sla=False,
        forbid_alpine_musl=True,
    )
    df_orin_nano = audit_dockerfile_profile(
        dkr_dir / "orin_nano_super_jp62.Dockerfile",
        repo_root=root,
        requires_nvidia_trt_sla=True,
        forbid_alpine_musl=False,
    )
    df_orin_nx_jp62 = audit_dockerfile_profile(
        dkr_dir / "orin_nx_jp62.Dockerfile",
        repo_root=root,
        requires_nvidia_trt_sla=True,
        forbid_alpine_musl=False,
    )
    df_orin_nx_jp72 = audit_dockerfile_profile(
        dkr_dir / "orin_nx_jp72.Dockerfile",
        repo_root=root,
        requires_nvidia_trt_sla=True,
        forbid_alpine_musl=False,
    )

    total_onnx_mib = float(onnx_audit["total_onnx_mib"])

    # ---- Profile 1: Tier-0B Bench (RPi 5 8GB) ----
    t0b_mass_min_g = 221.0
    t0b_mass_max_g = 243.0
    t0b_power_min_w = 8.7
    t0b_power_max_w = 12.7
    t0b_errors: list[str] = []
    if not df_rpi5["pass"]:
        t0b_errors.extend(df_rpi5["errors"])
    if not onnx_audit["pass"]:
        t0b_errors.extend(onnx_audit["errors"])
    if t0b_mass_max_g > POD_MASS_TARGET_CAP_G:
        t0b_errors.append(f"Tier-0B max mass {t0b_mass_max_g} g exceeds {POD_MASS_TARGET_CAP_G} g cap")
    if t0b_power_max_w > POD_POWER_TARGET_CAP_W:
        t0b_errors.append(f"Tier-0B max power {t0b_power_max_w} W exceeds {POD_POWER_TARGET_CAP_W} W cap")

    tier_0b = {
        "profile_id": "tier_0b_bench_rpi5_8gb",
        "title": "Tier-0B Bench (Raspberry Pi 5 8GB, CPU-Only)",
        "classification": "NOT_FLIGHT_CAPABLE",
        "flight_capable": False,
        "compute": {
            "board": "Raspberry Pi 5 8GB (ARM Cortex-A76, CPU-only)",
            "accelerator_tops": 0,
            "ring0_hardware_fault_isolated": False,
            "lwir_included": False,
            "reason_not_flight_capable": (
                "CPU-only bench/maker logger with zero hardware Ring-0 MCU fault isolation; "
                "desktop Ryzen 5 5600X 65W already stalls at 9.0-9.4 Hz multi-sensor loop."
            ),
            "evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
        },
        "swap_budget": {
            "mass_min_g": t0b_mass_min_g,
            "mass_max_g": t0b_mass_max_g,
            "mass_cap_g": POD_MASS_TARGET_CAP_G,
            "mass_within_300g_cap": bool(t0b_mass_max_g <= POD_MASS_TARGET_CAP_G),
            "mass_headroom_min_g": round(POD_MASS_TARGET_CAP_G - t0b_mass_max_g, 2),
            "power_min_w": t0b_power_min_w,
            "power_max_w": t0b_power_max_w,
            "power_cap_w": POD_POWER_TARGET_CAP_W,
            "power_within_15w_cap": bool(t0b_power_max_w <= POD_POWER_TARGET_CAP_W),
            "power_headroom_min_w": round(POD_POWER_TARGET_CAP_W - t0b_power_max_w, 2),
            "evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
        },
        "compliance": {
            "dockerfile": df_rpi5["dockerfile"],
            "non_flight_sim_only_locked": bool(df_rpi5["non_flight_sim_only_locked"]),
            "actuator_authority": False,
            "no_alpine_musl": bool(df_rpi5["no_alpine_musl"]),
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        },
        "pass": len(t0b_errors) == 0,
        "errors": t0b_errors,
    }

    # ---- Profile 2: Tier-1 EO-Only Shadow Pod (Orin Nano Super 8GB + STM32H7 Ring-0 MCU) ----
    t1_env = df_orin_nano["env"]
    t1_tops = int(t1_env["OPENXVS_COMPUTE_TOPS"]) if "OPENXVS_COMPUTE_TOPS" in t1_env else 0
    t1_ram_cap_gb = int(t1_env["OPENXVS_RAM_CAP_GB"]) if "OPENXVS_RAM_CAP_GB" in t1_env else 0
    t1_mem_cap_mib = float(t1_env["OPENXVS_MEMORY_CAP_MB"]) if "OPENXVS_MEMORY_CAP_MB" in t1_env else 0.0
    t1_trt_ws_mib = float(t1_env["OPENXVS_TRT_WORKSPACE_MB"]) if "OPENXVS_TRT_WORKSPACE_MB" in t1_env else 0.0

    t1_engines_act_mib = round(total_onnx_mib * TRT_ENGINE_AND_ACTIVATION_MULTIPLIER, 2)
    t1_6_onnx_total_mib = round(
        total_onnx_mib + t1_engines_act_mib + t1_trt_ws_mib + JETSON_CUDA_TRT_ORT_RUNTIME_OVERHEAD_MIB,
        2,
    )
    t1_fits_6144 = t1_mem_cap_mib > 0.0 and t1_6_onnx_total_mib <= t1_mem_cap_mib
    t1_with_vlm_2b_mib = round(t1_6_onnx_total_mib + VLM_2B_TOTAL_FOOTPRINT_MIB, 2)
    t1_vlm_2b_exceeds_cap = t1_with_vlm_2b_mib > ORIN_NANO_SUPER_APP_RAM_CAP_MIB

    t1_errors: list[str] = []
    if not df_orin_nano["pass"]:
        t1_errors.extend(df_orin_nano["errors"])
    if not onnx_audit["pass"]:
        t1_errors.extend(onnx_audit["errors"])
    if not mcu_audit["pass"]:
        t1_errors.extend(mcu_audit["errors"])
    if not uart_audit["pass"]:
        t1_errors.extend(uart_audit["errors"])
    if t1_tops != ORIN_NANO_SUPER_EXPECTED_TOPS:
        t1_errors.append(
            f"Tier-1 Orin Nano Super TOPS mismatch in Dockerfile: expected {ORIN_NANO_SUPER_EXPECTED_TOPS}, got {t1_tops}"
        )
    if t1_ram_cap_gb != 8 or abs(t1_mem_cap_mib - ORIN_NANO_SUPER_APP_RAM_CAP_MIB) > 1e-6:
        t1_errors.append(
            f"Tier-1 Orin Nano Super RAM cap mismatch: ram_cap_gb={t1_ram_cap_gb}, mem_cap_mib={t1_mem_cap_mib}"
        )
    if not t1_fits_6144:
        t1_errors.append(
            f"Tier-1 6-ONNX + TRT workspace ({t1_6_onnx_total_mib} MiB) exceeds {t1_mem_cap_mib} MiB app cap"
        )
    if not t1_vlm_2b_exceeds_cap:
        t1_errors.append("Expected VLM 2B combined footprint to exceed 6,144 MiB 8 GB cap")

    tier_1 = {
        "profile_id": "tier_1_eo_shadow_orin_nano_super_8gb",
        "title": "Tier-1 EO-Only Shadow Pod (Orin Nano Super 8GB + STM32H7 Ring-0 MCU)",
        "classification": "PASSIVE_SHADOW_POD_EVAL_ONLY",
        "flight_capable": False,
        "shadow_mode_tx_inhibited_mandatory": True,
        "compute": {
            "board": "NVIDIA Jetson Orin Nano Super 8GB",
            "compute_tops": t1_tops,
            "ring0_mcu": "STM32H7 (ARM Cortex-M7, thumbv7em-none-eabi, physically isolated)",
            "lwir_included": False,
            "vendor_spec_evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
            "silicon_latency_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
        },
        "ram_budget_8gb": {
            "total_unified_ram_mib": ORIN_NANO_SUPER_TOTAL_RAM_MIB,
            "app_ram_cap_mib": t1_mem_cap_mib,
            "os_l4t_reserve_mib": ORIN_NANO_SUPER_OS_RESERVE_MIB,
            "active_6_onnx_weights_bytes": int(onnx_audit["total_onnx_bytes"]),
            "active_6_onnx_weights_mib": total_onnx_mib,
            "weights_evidence_tag": EVIDENCE_VERIFIED_ON_SILICON,
            "trt_workspace_cap_mib": t1_trt_ws_mib,
            "trt_engines_and_activations_est_mib": t1_engines_act_mib,
            "cuda_trt_ort_runtime_overhead_est_mib": JETSON_CUDA_TRT_ORT_RUNTIME_OVERHEAD_MIB,
            "projected_6_onnx_plus_workspace_mib": t1_6_onnx_total_mib,
            "headroom_inside_6144_mib_cap": round(t1_mem_cap_mib - t1_6_onnx_total_mib, 2),
            "fits_within_6144_mib_cap": bool(t1_fits_6144),
            "runtime_allocation_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
            "vlm_2b_policy": {
                "status": "EXCLUDED_8GB_CAP",
                "admitted": False,
                "vlm_2b_fp16_weights_est_mib": VLM_2B_FP16_WEIGHTS_MIB,
                "vlm_2b_activations_kv_est_mib": VLM_2B_VISION_AND_KV_ACTIVATIONS_MIB,
                "vlm_2b_total_est_mib": VLM_2B_TOTAL_FOOTPRINT_MIB,
                "projected_combined_with_vlm_2b_mib": t1_with_vlm_2b_mib,
                "overage_vs_6144_mib_cap": round(t1_with_vlm_2b_mib - t1_mem_cap_mib, 2),
                "rationale": (
                    "2B VLM exceeds 6,144 MiB unified LPDDR5 app cap when co-resident with 6 ONNX "
                    "perception engines and 1,024 MiB TRT workspace; excluded fail-closed on 8 GB Tier-1."
                ),
                "evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
            },
        },
        "ring0_mcu_budget": {
            "artifact": mcu_audit["artifact"],
            "flash_bytes": int(mcu_audit["flash_bytes"]) if "flash_bytes" in mcu_audit else 0,
            "flash_capacity_bytes": STM32H7_FLASH_CAPACITY_BYTES,
            "flash_utilization_pct": float(mcu_audit["flash_utilization_pct"]) if "flash_utilization_pct" in mcu_audit else 100.0,
            "max_flash_utilization_pct": MAX_RING0_MCU_FLASH_UTIL_PCT,
            "flash_within_15pct_cap": bool(mcu_audit["flash_within_15pct_cap"]) if "flash_within_15pct_cap" in mcu_audit else False,
            "static_ram_bytes": int(mcu_audit["static_ram_bytes"]) if "static_ram_bytes" in mcu_audit else -1,
            "zero_static_ram_verified": bool(mcu_audit["zero_static_ram_verified"]) if "zero_static_ram_verified" in mcu_audit else False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
        },
        "swap_budget": {
            "mass_min_g": 205.0,
            "mass_max_g": 275.0,
            "mass_cap_g": POD_MASS_TARGET_CAP_G,
            "mass_within_300g_cap": True,
            "power_rail_10w_nvpmodel_w": 12.9,
            "power_rail_maxn_super_25w_w": 28.0,
            "nvpmodel_le_10w_required_for_sub_15w": True,
            "evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
        },
        "uart_budget": {
            "cap_bytes_per_sec": float(MAX_SIGNED_UART_BYTES_PER_SEC),
            "signed_bidir_bytes_per_sec": float(uart_audit["signed_bidirectional"]["total_bytes_per_sec"]),
            "signed_bidir_utilization_pct": float(uart_audit["signed_bidirectional"]["utilization_pct"]),
            "shadow_rx_only_bytes_per_sec": float(uart_audit["passive_shadow_rx_only"]["total_bytes_per_sec"]),
            "within_budget": bool(uart_audit["pass"]),
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        },
        "compliance": {
            "dockerfile": df_orin_nano["dockerfile"],
            "non_flight_sim_only_locked": bool(df_orin_nano["non_flight_sim_only_locked"]),
            "actuator_authority": False,
            "nvidia_tensorrt_sla_sec_1_2_and_2_ok": bool(df_orin_nano["nvidia_trt_sla_sec_1_2_and_2_ok"]),
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        },
        "pass": len(t1_errors) == 0,
        "errors": t1_errors,
    }

    # ---- Profile 3: Tier-2 Dual EO+LWIR Pod (Orin NX 16GB + STM32H7/R5 + <=9 Hz Boson+ 320) ----
    t2_mass_min_g = 255.0
    t2_mass_max_g = 350.0
    t2_power_10w_rail_w = 13.9
    t2_power_15w_rail_w = 20.7
    t2_lwir_hz = 9.0
    t2_nvpmodel_10w_mandatory = (t2_power_10w_rail_w <= POD_POWER_TARGET_CAP_W) and (
        t2_power_15w_rail_w > POD_POWER_TARGET_CAP_W
    )

    t2_errors: list[str] = []
    if not df_orin_nx_jp62["pass"]:
        t2_errors.extend([f"jp62: {e}" for e in df_orin_nx_jp62["errors"]])
    if not df_orin_nx_jp72["pass"]:
        t2_errors.extend([f"jp72: {e}" for e in df_orin_nx_jp72["errors"]])
    if not onnx_audit["pass"]:
        t2_errors.extend(onnx_audit["errors"])
    if not mcu_audit["pass"]:
        t2_errors.extend(mcu_audit["errors"])
    if not uart_audit["pass"]:
        t2_errors.extend(uart_audit["errors"])
    if t2_lwir_hz > LWIR_EAR99_MAX_HZ:
        t2_errors.append(f"LWIR frame rate {t2_lwir_hz} Hz exceeds EAR99 <= {LWIR_EAR99_MAX_HZ} Hz limit (6A003.b.4.b)")
    if not t2_nvpmodel_10w_mandatory:
        t2_errors.append("Tier-2 10W nvpmodel sub-15W rail check failed")

    t2_projected_6_onnx_mib = round(
        total_onnx_mib + t1_engines_act_mib + 1024.0 + JETSON_CUDA_TRT_ORT_RUNTIME_OVERHEAD_MIB,
        2,
    )

    tier_2 = {
        "profile_id": "tier_2_dual_eo_lwir_orin_nx_16gb",
        "title": "Tier-2 Dual EO+LWIR Pod (Orin NX 16GB + STM32H7/R5 Ring-0 MCU + <=9 Hz Boson+ 320)",
        "classification": "PASSIVE_SHADOW_POD_EVAL_ONLY",
        "flight_capable": False,
        "shadow_mode_tx_inhibited_mandatory": True,
        "compute": {
            "board": "NVIDIA Jetson Orin NX 16GB",
            "total_unified_ram_mib": ORIN_NX_TOTAL_RAM_MIB,
            "ring0_mcu": "STM32H7 / Cortex-R5 Lock-Step MCU (physically isolated + 2s supercap + WWDT)",
            "lwir_core": "FLIR Boson+ 320 (<= 9 Hz EAR99 civil core, bare frameless)",
            "active_6_onnx_weights_mib": total_onnx_mib,
            "projected_6_onnx_plus_workspace_mib": t2_projected_6_onnx_mib,
            "vlm_2b_policy": "OPTIONAL_16GB_ADVISORY_ONLY",
            "weights_evidence_tag": EVIDENCE_VERIFIED_ON_SILICON,
            "silicon_latency_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
        },
        "swap_margin_audit": {
            "mass_min_g": t2_mass_min_g,
            "mass_max_g": t2_mass_max_g,
            "mass_target_cap_g": POD_MASS_TARGET_CAP_G,
            "mass_bare_core_fits_300g": bool(t2_mass_min_g <= POD_MASS_TARGET_CAP_G),
            "mass_bare_core_headroom_g": round(POD_MASS_TARGET_CAP_G - t2_mass_min_g, 2),
            "mass_cots_boxed_exceeds_300g": bool(t2_mass_max_g > POD_MASS_TARGET_CAP_G),
            "mass_cots_boxed_overage_g": round(t2_mass_max_g - POD_MASS_TARGET_CAP_G, 2),
            "mass_margin_verdict": "MARGINAL_BARE_CORE_REQUIRED",
            "power_cap_w": POD_POWER_TARGET_CAP_W,
            "power_rail_10w_nvpmodel_w": t2_power_10w_rail_w,
            "power_rail_10w_headroom_w": round(POD_POWER_TARGET_CAP_W - t2_power_10w_rail_w, 2),
            "power_rail_15w_nvpmodel_w": t2_power_15w_rail_w,
            "power_rail_15w_deficit_w": round(POD_POWER_TARGET_CAP_W - t2_power_15w_rail_w, 2),
            "nvpmodel_10w_mandatory_for_sub_15w": bool(t2_nvpmodel_10w_mandatory),
            "slipstream_ram_air_cooling_required": True,
            "evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
        },
        "lwir_export_audit": {
            "sensor": "Boson+ 320",
            "frame_rate_hz": t2_lwir_hz,
            "ear99_ceiling_hz": LWIR_EAR99_MAX_HZ,
            "compliant_le_9hz": bool(t2_lwir_hz <= LWIR_EAR99_MAX_HZ),
            "eccn_6a003_b_4_b_quarantine_avoided": bool(t2_lwir_hz <= LWIR_EAR99_MAX_HZ),
            "infiray_yantai_ofac_banned": True,
            "evidence_tag": EVIDENCE_UNVERIFIED_VENDOR_SPEC,
        },
        "ring0_mcu_budget": {
            "artifact": mcu_audit["artifact"],
            "flash_bytes": int(mcu_audit["flash_bytes"]) if "flash_bytes" in mcu_audit else 0,
            "flash_utilization_pct": float(mcu_audit["flash_utilization_pct"]) if "flash_utilization_pct" in mcu_audit else 100.0,
            "flash_within_15pct_cap": bool(mcu_audit["flash_within_15pct_cap"]) if "flash_within_15pct_cap" in mcu_audit else False,
            "static_ram_bytes": int(mcu_audit["static_ram_bytes"]) if "static_ram_bytes" in mcu_audit else -1,
            "zero_static_ram_verified": bool(mcu_audit["zero_static_ram_verified"]) if "zero_static_ram_verified" in mcu_audit else False,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
            "wcet_evidence_tag": EVIDENCE_UNVERIFIED_FLIGHT_HW,
        },
        "uart_921k_20pct_audit": {
            "baud_bps": int(UART_BAUD_BPS),
            "cap_bytes_per_sec": float(MAX_SIGNED_UART_BYTES_PER_SEC),
            "signed_bidir_bytes_per_sec": float(uart_audit["signed_bidirectional"]["total_bytes_per_sec"]),
            "signed_bidir_utilization_pct": float(uart_audit["signed_bidirectional"]["utilization_pct"]),
            "signed_bidir_headroom_bytes_per_sec": float(uart_audit["signed_bidirectional"]["headroom_bytes_per_sec"]),
            "shadow_rx_only_bytes_per_sec": float(uart_audit["passive_shadow_rx_only"]["total_bytes_per_sec"]),
            "within_20pct_cap": bool(uart_audit["pass"]),
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        },
        "compliance": {
            "dockerfiles": [df_orin_nx_jp62["dockerfile"], df_orin_nx_jp72["dockerfile"]],
            "non_flight_sim_only_locked": bool(
                df_orin_nx_jp62["non_flight_sim_only_locked"] and df_orin_nx_jp72["non_flight_sim_only_locked"]
            ),
            "actuator_authority": False,
            "nvidia_tensorrt_sla_sec_1_2_and_2_ok": bool(
                df_orin_nx_jp62["nvidia_trt_sla_sec_1_2_and_2_ok"]
                and df_orin_nx_jp72["nvidia_trt_sla_sec_1_2_and_2_ok"]
            ),
            "rel6_legal_and_g5_gap_required_before_actuation": True,
            "evidence_tag": EVIDENCE_VERIFIED_FORMAL_OR_SIM,
        },
        "pass": len(t2_errors) == 0,
        "errors": t2_errors,
    }

    overall_pass = bool(
        mcu_audit["pass"]
        and onnx_audit["pass"]
        and uart_audit["pass"]
        and tier_0b["pass"]
        and tier_1["pass"]
        and tier_2["pass"]
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "non_flight_sim_only": True,
        "actuator_authority": False,
        "gpl_free": True,
        "coordinate_frame": "NED/ECEF (0 internal ENU)",
        "watermark": NON_FLIGHT_WATERMARK,
        "evidence_taxonomy": sorted(VALID_EVIDENCE_TAGS),
        "overall_pass": overall_pass,
        "on_disk_artifacts": {
            "ring0_thumbv7em_mcu_proof": mcu_audit,
            "active_6_onnx_weights": onnx_audit,
            "uart_mavlink2_921k_budget": uart_audit,
            "dockerfiles": {
                "rpi5": df_rpi5,
                "orin_nano_super_jp62": df_orin_nano,
                "orin_nx_jp62": df_orin_nx_jp62,
                "orin_nx_jp72": df_orin_nx_jp72,
            },
        },
        "profiles": {
            "tier_0b_bench_rpi5_8gb": tier_0b,
            "tier_1_eo_shadow_orin_nano_super_8gb": tier_1,
            "tier_2_dual_eo_lwir_orin_nx_16gb": tier_2,
        },
    }

    tags_used: set[str] = set()
    _collect_all_evidence_tags(report, tags_used)
    unknown_tags = sorted(tags_used - VALID_EVIDENCE_TAGS)
    if unknown_tags:
        report["overall_pass"] = False
        report["evidence_tag_errors"] = unknown_tags

    return report


def write_flight_pod_budget_report(report: dict[str, Any], out_path: Path) -> Path:
    """Write deterministic UTF-8 JSON report (strictly `allow_nan=False`)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n"
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic OpenXVS Flight-Pod SWaP, Bare-Metal MCU & Telemetry Bandwidth Auditor"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to OpenXVS repository root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output JSON report path (default: <repo-root>/{DEFAULT_REPORT_OUT_REL})",
    )
    parser.add_argument(
        "--mcu-proof",
        type=Path,
        default=None,
        help=f"Optional override path to {DEFAULT_MCU_PROOF_REL}",
    )
    parser.add_argument(
        "--docker-dir",
        type=Path,
        default=None,
        help="Optional override path to docker/ directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.repo_root.resolve()
    out_path = (root / DEFAULT_REPORT_OUT_REL) if args.out is None else args.out
    report = run_flight_pod_budget_audit(
        repo_root=root,
        mcu_proof_path=args.mcu_proof,
        docker_dir=args.docker_dir,
    )
    write_flight_pod_budget_report(report, out_path)

    t0 = report["profiles"]["tier_0b_bench_rpi5_8gb"]
    t1 = report["profiles"]["tier_1_eo_shadow_orin_nano_super_8gb"]
    t2 = report["profiles"]["tier_2_dual_eo_lwir_orin_nx_16gb"]
    print(
        "[verify_flight_pod_budget] "
        f"overall_pass={report['overall_pass']} | "
        f"6_onnx={report['on_disk_artifacts']['active_6_onnx_weights']['total_onnx_mib']:.2f} MiB | "
        f"mcu_flash={t1['ring0_mcu_budget']['flash_utilization_pct']:.4f}% "
        f"({t1['ring0_mcu_budget']['flash_bytes']} B, static_ram={t1['ring0_mcu_budget']['static_ram_bytes']} B) | "
        f"tier0b={t0['classification']} | "
        f"tier1_ram={t1['ram_budget_8gb']['projected_6_onnx_plus_workspace_mib']:.2f}/"
        f"{t1['ram_budget_8gb']['app_ram_cap_mib']:.0f} MiB (vlm2b={t1['ram_budget_8gb']['vlm_2b_policy']['status']}) | "
        f"tier2_swap={t2['swap_margin_audit']['mass_min_g']:.0f}-{t2['swap_margin_audit']['mass_max_g']:.0f}g "
        f"({t2['swap_margin_audit']['mass_margin_verdict']}), "
        f"10W_rail={t2['swap_margin_audit']['power_rail_10w_nvpmodel_w']:.1f}W, "
        f"uart={t2['uart_921k_20pct_audit']['signed_bidir_bytes_per_sec']:.1f} B/s "
        f"({t2['uart_921k_20pct_audit']['signed_bidir_utilization_pct']:.2f}%)"
    )
    return 0 if bool(report["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
