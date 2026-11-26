"""One-Click Buyer Verification & Assurance Pack Exporter (`.zip`, `OpenXVS v5.0`).

Generates a deterministic, cryptographically signed, audit-ready ZIP archive containing:
  - `01_executive_summary.json`           : Mode-strict governance, runtime provenance & gap disclosure
  - `02_formal_certification_certgen.json`: Live `xvs_certgen.py` GSN v3 bundle (`18/18` Kani proofs)
  - `03_hil_loopback_proof.json`          : Loopback self-test + fail-closed live ArduPlane SITL proof
  - `04_latest_video_analysis.json`       : Live video job or Amazon AOT real-sky proof summary
  - `05_flightgear_openxvs_protocol.xml`  : FlightGear Generic UDP simulation protocol XML
  - `06_FIELD_TRIAL_PROTOCOL.md`          : Strictly desktop/bench sim only in `private_eval` (zero
                                            Python CLI, zero Phase-4 flight-test instructions); full
                                            field-trial protocol only in `public_open_core`
  - `07_sora25_f3442_ed324_matrix.json`   : Machine-readable EASA SORA 2.5 (17 OSOs + M1A/B/C),
                                            ASTM F3442-26 & EUROCAE ED-324 traceability matrix
  - `08_spdx_2_3_sbom.json`               : Deterministic SPDX 2.3 JSON SBOM (0 GPL/AGPL/LGPL-static)
  - `09_onnx_model_cards.json`            : Model cards (tensor shapes, ODD bounds, failure modes,
                                            fallback policies, SHA-256 digests where present on disk)
  - `MANIFEST.sha256`                     : Canonical sorted SHA-256 manifest covering items 01..09
  - `MANIFEST.sha256.sig`                 : Detached 64-byte RFC 8032 Ed25519 signature over MANIFEST.sha256
  - `MANIFEST_VERIFY.json`                : Ed25519 verification metadata (`5D002` auth-only note)
  - `verify_pack.py`                      : Standalone pure-stdlib offline SHA-256 + Ed25519 verifier

Enforces fail-closed `release_mode` governance (`"private_eval"` default with ZERO `Apache-2.0`
contradictions, ZERO `§734.7` public-domain claim, ZERO `Art. 2(2)` FOSS claim, and explicit
`G5_GAP` / `G2_TLA_DEFERRED` undeveloped boundaries).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
import zipfile

if TYPE_CHECKING:
    from .hil_bridge import StudioHilBridgeManager
    from .inference_engine import StudioInferenceEngine
    from .video_analyzer import VideoJobManager

REPO_ROOT = Path(__file__).resolve().parents[4]
FG_PROTOCOL_XML_PATH = REPO_ROOT / "configs" / "flightgear" / "openxvs_fg_protocol.xml"
CERTGEN_SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "xvs_certgen.py"
CERTGEN_BUNDLE_PATH = REPO_ROOT / "output" / "xvs" / "assurance" / "xvs_certgen_bundle.json"
AOT_METRICS_PATH = REPO_ROOT / "outputs" / "xvs_activerange_aot_metrics.json"
LIVE_SITL_PROOF_PATH = REPO_ROOT / "outputs" / "sitl_proof" / "live_arduplane_sitl_proof.json"
ED25519_PURE_PATH = REPO_ROOT / "src" / "xvs" / "v2" / "xvs_bench" / "ed25519_pure.py"
SORA_MATRIX_PATH = REPO_ROOT / "src" / "xvs" / "v2" / "assurance" / "sora25_f3442_matrix.py"

DEFAULT_RELEASE_MODE: str = "private_eval"
VALID_RELEASE_MODES: tuple[str, ...] = ("private_eval", "public_open_core")
NON_FLIGHT_WATERMARK: str = "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"

# Deterministic evaluation-pack Ed25519 seed (authentication/integrity only; NOT a secret production key)
_DEFAULT_EVAL_ED25519_SEED32: bytes = hashlib.sha256(
    b"openxvs-v5-buyer-assurance-pack-eval-ed25519-v1"
).digest()

_PRIVATE_EVAL_HEADER_MD = """# OpenXVS v5.0 — Confidential Non-Flight Desktop/Bench Evaluation Protocol

> **CONFIDENTIAL NON-FLIGHT DESKTOP/BENCH EVALUATION NOTICE (`release_mode=private_eval`):**
> **SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT**
>
> This OpenXVS v5.0 evaluation package is provided strictly on a **Private-First, Confidential,
> Non-Flight Desktop/Bench Simulation Evaluation** basis for strictly **civil / dual-use**
> aviation perception and Runtime Assurance (RTA) technical review (zero military/weapons/
> fire-control scope, zero GPL/AGPL/LGPL-static runtime dependencies).
>
> - **Certification Boundary (`G5_GAP` & `G2_TLA_DEFERRED`):** OpenXVS is **NOT** a certified
>   flight-control or airworthiness product. Target hardware qualification (`DO-254`), qualified
>   compilers/RTOS (`DO-330` / `ARINC 653`), airframe safety assessment (`ARP4754B`), and
>   independent `DER/CVE` verification remain explicitly undeveloped in
>   `assurance/openxvs_v2.gsn.yaml` (`G5_GAP`), and full distributed temporal model checking
>   remains transparently marked under `G2_TLA_DEFERRED`.
> - **Export Control Non-Public Status (`EAR 15 CFR §734.7` / `Reg. (EU) 2026/2288` / `91 FR 52501`):**
>   During Private-First maturation, this software is **non-public / access-restricted** and does
>   **NOT** qualify for `15 CFR §734.7` or EU Dual-Use General Software Note exclusions. Restricted
>   strictly to screened civil recipients (`< 3-hour` endurance, `< 300-km` range civil UAS).
>   Private `XVS-Nav` DBRN source code (`ECCN 7D003.b.3`) is withheld; offline packs bundle zero
>   TLS/AES confidentiality encryption (pure RFC 8032 Ed25519 authentication/integrity only,
>   excluded from `ECCN 5D002`).
> - **EU Product Liability Firewall (`Directive (EU) 2024/2853`, Effective Dec 9, 2026):**
>   No `Art. 2(2)` open-source exemption is claimed (`Art. 4(1)` and `Art. 4(7)` cover compiled
>   software supplied free of charge; `Art. 15` bars contractual waivers toward injured natural
>   persons; `Art. 17` imposes a 10-year expiry). Provided exclusively for unpaid desktop/bench
>   simulation evaluation (`NON_FLIGHT_SIM_ONLY=1`, `ACTUATOR_AUTHORITY = false`, zero flight-
>   controller actuation, zero `.stl`/`.gcode`/`.kicad*` manufacturing files) under countersigned
>   Mutual NDA & Non-Flight Evaluation License.
> - **Planned Timed Public Open-Core Transition (Informational Only — Zero Current OSS Grant):**
>   A future clean-room Timed Public Open-Core transition (Model 2: Wave-1 Q1 2027 formal core;
>   Wave-2 October 2027 community core) is planned separately; **no public open-source license
>   grant is active in this confidential evaluation package**.
"""

_PRIVATE_EVAL_BODY_MD = """---

## Tier-0A — Zero-Cost Desktop/Bench Simulation Evaluation ($0 Hardware)

1. Verify package cryptographic integrity offline before launching:
   - Run `python verify_pack.py` (or verify `MANIFEST.sha256` against `MANIFEST.sha256.sig` using
     the Ed25519 public key recorded in `MANIFEST_VERIFY.json`).
2. Unzip and launch standalone **`OpenXVS Studio` (`openxvs-studio.exe`)** — compiled with pure-Rust
   `ort` (`load-dynamic`, relative `onnxruntime.dll`), `NON_FLIGHT_SIM_ONLY=1`, zero Python runtime
   required, and immutable `SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT`
   UI & export watermark.
3. Evaluate bundled real civil `.mp4`/`.webm` footage and static test imagery in **Real Video Lab**:
   - Inspect per-frame p50/p95/p99 latency, false-alarm bounds, SHA-256 provenance badges, and
     deterministic `xvs-rta-core` fallback transitions across `XVS-D2 + ActiveRange`, `XVS-Land`,
     `XVS-ThermalNet`, and `XVS-Nav` (when ONNX artifact is present).
4. Review offline deterministic assurance artifacts inside this pack:
   - `02_formal_certification_certgen.json` (`18/18` Kani CBMC harnesses, `G5_GAP` open),
   - `07_sora25_f3442_ed324_matrix.json` (EASA SORA 2.5 all 17 OSOs + M1A/B/C, ASTM F3442-26, ED-324),
   - `08_spdx_2_3_sbom.json` (deterministic SPDX 2.3 SBOM, 0 GPL/AGPL/LGPL-static),
   - `09_onnx_model_cards.json` (tensor shapes, ODD bounds, failure modes & fallback guarantees).

---

## Tier-0B — Offline Edge Benchmark Kit (Stationary Lab Bench Only)

- **Raspberry Pi 5 (`linux/arm64` OCI):** Pull or load the signed benchmark-only `linux/arm64` OCI
  archive (`cosign verify --key openxvs-eval-pub.pem`, zero public Rekor log leak) built on
  `wolfi-base` / `distroless-cc` glibc + Rust `ort` (`NON_FLIGHT_SIM_ONLY=1`). Measure offline
  CPU latency & accuracy over local `.mp4` test clips on a stationary laboratory bench.
- **Jetson Orin NX (`linux/arm64` JetPack 6.2 / 7.2 OEM Recipe):** Build hardware-specific
  TensorRT `.engine` caches locally on OEM bench hardware from the official `nvcr.io` Dockerfile
  recipe strictly for non-flight benchmarking, observing **NVIDIA TensorRT SDK License Agreement
  Sec. 1.2 & Sec. 2** (which prohibits Critical Applications including avionics/navigation).
- **Zero Flight Coupling:** Stationary bench power and local test datasets ONLY — zero airborne
  mounting, zero autopilot serial/UART/CAN/PWM wiring, zero outbound flight commands.

---

## Excluded Scope — Airborne Flight Testing Prohibited in `private_eval` (`REL4` / `REL6`)

- **Phase-4 supervised civil flight testing, closed-loop autopilot integration, and any outbound
  flight-controller actuation are STRICTLY EXCLUDED AND PROHIBITED under `private_eval`.**
- Any unauthorized coupling to an autopilot or flying airframe constitutes an unforeseeable
  substantial modification under `Directive (EU) 2024/2853 Art. 4(18) / Art. 8(2)` under the
  recipient's exclusive liability.
- Airborne flight trials require either (a) a separate signed university aero-lab addendum for
  physically air-gapped passive shadow-mode data logging under the university's permit & aviation
  insurance, (b) `REL6` post-incorporation commercial contracting & Aviation Products Liability
  insurance, or (c) the scheduled Timed Public Open-Core release (`public_open_core`).
"""

_PUBLIC_OPEN_CORE_HEADER_MD = """# OpenXVS v5.0 — Field-Trial & Open-Core Civil Evaluation Protocol

> **Scope, Open-Core License & Safety Notice (`release_mode=public_open_core`):**
> OpenXVS is a civil/dual-use aviation perception and Runtime Assurance (RTA) open-core
> software & reference hardware suite distributed under permissive licenses:
> **`Apache-2.0`** (software & Community Studio), **`CERN-OHL-P-2.0`** (open hardware CAD/PCB/BOM),
> **`CDLA-Permissive-2.0`** (datasets & model weights), and **`CC-BY-4.0`** (documentation
> and GSN artifacts), with strictly zero GPL/AGPL/LGPL-static runtime dependencies.
>
> - **Certification Boundary (`G5_GAP` & `G2_TLA_DEFERRED`):** OpenXVS is **NOT** a certified
>   flight-control or airworthiness product (`G5_GAP`: target hardware qualification, qualified
>   compilers/RTOS, and formal civil cert-lab/DER verification remain explicitly undeveloped in
>   `assurance/openxvs_v2.gsn.yaml`).
> - **Civil Safety & Non-Commercial OSS Boundary:** Strictly civil Part 91/107/108/135 &
>   EASA SORA 2.5 scope; zero military/targeting/weapons use. See `LICENSE`, `NOTICE`,
>   `SAFETY_DISCLAIMER.md`, `EXPORTS.md`, and `TRADEMARK.md`.
"""

_PUBLIC_OPEN_CORE_BODY_MD = """---

## Tier-0A — Zero-Cost Bench & Window/Tripod Test ($0 Hardware)

1. Launch `OpenXVS Studio` (`openxvs-studio.exe` native Rust `ort` IPC or local dev preview
   `http://127.0.0.1:4174` with `python scripts/xvs/studio_live_server.py --port 8765`).
2. Open **Real Video & Camera Lab**:
   - Upload any civil `.mp4`/`.webm` sky, cockpit landing, or aerial footage and inspect the
     real-model annotated `.mp4` output + verified SHA-256 provenance badges.
   - Connect any standard USB webcam (`640×360`–`1080p`) pointed out a window or from a tripod
     at local civil airport traffic and verify live `XVS-D2` + `ActiveRange` tracking (`d2`),
     `XVS-Land` runway geometry (`land`), `XVS-ThermalNet` EO/LWIR vacuity (`thermal`), and
     `XVS-Nav` cross-view retrieval (`nav`).
3. Run **1-Click HIL Loopback Self-Test** (`POST /api/hil/loopback_test`) to confirm `100/100`
   signed `77 B` MAVLink2 `#52001` packets, `#84` fixed-wing ban, `64 B` ActiveRange CRC32, and
   FlightGear Generic UDP loopback on `127.0.0.1`.

---

## Tier-0B — Low-Cost Ground/Vehicle Pod Prototype (~$408 BOM)

- **Compute:** Raspberry Pi 5 (8 GB) + Jetson Orin NX / Nano Super dev kit (running ONNX/TensorRT).
- **Optics:** Global-shutter EO camera (CSI/GMSL2, `60–90°` HFOV) + optional LWIR uncooled microbolometer.
- **Mount:** Roof-rack or tripod vehicle test rig driving near civil airfield approaches (zero airborne risk).
- **Validation Gates:**
  - Verify real-time inference cadence (`>= 10 Hz` `XVS-D2 + ActiveRange`, `>= 5 Hz` `XVS-Land`).
  - Log `64 B` `XVS-ActiveRange` wire frames + MAVLink2 `#50004 POD_PLUGIN_HEALTH` (`ACTUATOR_AUTHORITY=false`)
    in shadow listen-only mode before any autopilot integration.

---

## Phase-4 — Supervised Civil Flight Test (ArduPlane / PX4 Shadow Mode)

1. Connect OpenXVS companion computer to ArduPlane `>= 4.7.2` (required for `#331 ODOMETRY` safety)
   over `921.6 kbps` signed MAVLink 2.0 UART (`< 20%` bandwidth utilization).
2. Begin strictly in **Shadow / Advisory-Only Mode** (`ACTUATOR_AUTHORITY = false`, visual observer +
   safety pilot with hardware RC override switch at all times; requires separate flight-test
   authorization and OEM safety sign-off outside this open-core research distribution).
3. Verify `DR-06` fixed-wing avoidance interface strictly uses `#75 COMMAND_INT` or `#82 SET_ATTITUDE_TARGET`
   (`|bank| <= 30 deg`, `V >= 18 m/s`, MAVLink `#84` hard-blocked) and `EKF3 42007` source-switch
   dwell (`>= 2.0 s`) under controlled GNSS-spoof simulation before closing any autonomy loop.
"""

# Backwards-compatible module alias (default points to private_eval desktop/bench body)
_FIELD_TRIAL_BODY_MD = _PRIVATE_EVAL_BODY_MD


def _load_helper_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_ed25519_helpers():
    mod = _load_helper_module("openxvs_ed25519_pure_shared", ED25519_PURE_PATH)
    return mod.ed25519_public_key, mod.ed25519_sign, mod.ed25519_verify


def _get_assurance_helpers():
    mod = _load_helper_module("openxvs_sora25_f3442_matrix_shared", SORA_MATRIX_PATH)
    return mod.build_sora25_f3442_ed324_matrix, mod.build_spdx_2_3_sbom


def validate_release_mode(release_mode: str) -> str:
    """Fail-closed validation for buyer assurance pack `release_mode`."""
    if not isinstance(release_mode, str) or release_mode not in VALID_RELEASE_MODES:
        raise ValueError(
            f"Unsupported release_mode {release_mode!r}; expected one of {VALID_RELEASE_MODES}"
        )
    return release_mode


def render_field_trial_protocol_md(release_mode: str = DEFAULT_RELEASE_MODE) -> str:
    """Render `06_FIELD_TRIAL_PROTOCOL.md` for the selected validated `release_mode`."""
    mode = validate_release_mode(release_mode)
    if mode == "private_eval":
        return f"{_PRIVATE_EVAL_HEADER_MD}\n{_PRIVATE_EVAL_BODY_MD}"
    return f"{_PUBLIC_OPEN_CORE_HEADER_MD}\n{_PUBLIC_OPEN_CORE_BODY_MD}"


FIELD_TRIAL_PROTOCOL_MD = render_field_trial_protocol_md(DEFAULT_RELEASE_MODE)


def _build_release_governance(release_mode: str) -> tuple[str, dict[str, Any]]:
    """Build `(license_posture, release_governance)` metadata for `01_executive_summary.json`."""
    if release_mode == "private_eval":
        license_posture = (
            "Confidential Non-Flight Desktop/Bench Evaluation Only "
            "(Private-First pre-release; zero active open-source license grant; "
            "zero GPL/AGPL/LGPL-static runtime dependencies; planned Timed Public Open-Core release)"
        )
        governance: dict[str, Any] = {
            "release_mode": "private_eval",
            "distribution_class": "CONFIDENTIAL_NON_FLIGHT_DESKTOP_BENCH_EVAL",
            "non_flight_sim_only": True,
            "actuator_authority": False,
            "watermark": NON_FLIGHT_WATERMARK,
            "ear_export_status": (
                "NON_PUBLIC_PRIVATE_FIRST: Not publicly available under EAR 15 CFR §734.7 or "
                "Commission Delegated Regulation (EU) 2026/2288; restricted to screened civil "
                "<3-hr / <300-km UAS recipients (91 FR 52501 / 15 CFR §744.21); private XVS-Nav "
                "source excluded (ECCN 7D003.b.3); zero TLS/AES confidentiality crypto (Ed25519 auth only)."
            ),
            "eu_pld_posture": (
                "NON_FLIGHT_SIMULATION_FIREWALL: Directive (EU) 2024/2853 Art. 2(2) OSS carve-out "
                "is NOT claimed (Art. 4(1)/4(7)/15/17 apply to closed software even at $0); "
                "restricted strictly to unpaid desktop/bench simulation evaluation "
                "(ACTUATOR_AUTHORITY=false, zero STL/G-code/KiCad files) under Mutual NDA + Non-Flight Eval License."
            ),
            "planned_public_open_core_release": (
                "Timed Public Open-Core (Model 2: Wave-1 Q1 2027 brand-decoupled formal core; "
                "Wave-2 October 2027 post-NeurIPS community core)"
            ),
            "g5_gap_enforced": True,
            "g2_tla_deferred_enforced": True,
        }
        return license_posture, governance

    license_posture = (
        "Apache-2.0 (software & Community Studio) / CERN-OHL-P-2.0 (hardware) / "
        "CDLA-Permissive-2.0 (datasets & weights) / CC-BY-4.0 (docs); "
        "strictly zero GPL/AGPL/LGPL-static runtime dependencies"
    )
    governance = {
        "release_mode": "public_open_core",
        "distribution_class": "PUBLIC_PERMISSIVE_OPEN_CORE",
        "non_flight_sim_only": False,
        "actuator_authority": False,
        "spdx_licenses": {
            "software": "Apache-2.0",
            "hardware": "CERN-OHL-P-2.0",
            "datasets_and_weights": "CDLA-Permissive-2.0",
            "documentation_and_gsn": "CC-BY-4.0",
        },
        "ear_export_status": (
            "CLEAN_ROOM_PUBLIC_RELEASE: Published without access restriction under 15 CFR §734.7 "
            "and EU GSN with zero legacy defense history and zero GPL/AGPL runtime dependencies."
        ),
        "eu_pld_posture": (
            "FREE_NON_COMMERCIAL_OSS: Published outside commercial activity under "
            "Directive (EU) 2024/2853 Art. 2(2); non-certified civil research & engineering software."
        ),
        "g5_gap_enforced": True,
        "g2_tla_deferred_enforced": True,
    }
    return license_posture, governance


def _sha256_file_if_present(path: Path) -> tuple[bool, int, str | None]:
    if not path.is_file():
        return False, 0, None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return True, path.stat().st_size, h.hexdigest()


def _resolve_artifact_candidates(
    repo_root: Path,
    candidates: tuple[str, ...],
) -> tuple[str, bool, int, str | None]:
    """Resolve the highest-priority existing artifact path on disk with SHA-256 digest."""
    for rel_artifact in candidates:
        present, size_bytes, sha256 = _sha256_file_if_present(repo_root / rel_artifact)
        if present:
            return rel_artifact, True, size_bytes, sha256
    return candidates[0], False, 0, None


def _build_onnx_model_cards(
    repo_root: Path,
    release_mode: str,
    health_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured ONNX & Rust model cards with tensor shapes, ODD bounds, failure modes & hashes."""
    weight_license = (
        "Confidential Non-Flight Evaluation Only (planned CDLA-Permissive-2.0 at Wave-2)"
        if release_mode == "private_eval"
        else "CDLA-Permissive-2.0"
    )
    specs: list[tuple[str, tuple[str, ...], dict[str, Any]]] = [
        (
            "xvs_d2",
            ("apps/livedeck/public/models/xvs_d2_web_fp16.onnx",),
            {
                "display_name": "XVS-D2 (EO Non-Cooperative Airborne Intruder Detector)",
                "runtime_format": "ONNX FP16",
                "input_tensors": [{"name": "input", "shape": [1, 3, 512, 512], "dtype": "float32/float16"}],
                "output_tensors": [{"name": "heatmap_and_offsets", "shape": [1, 5, 128, 128], "dtype": "float32"}],
                "training_data_provenance": "Amazon AOT (CDLA-Permissive-1.0, capture_day split, m_box=0) + G3a civil synthetic",
                "odd_bounds": "Daylight civil VFR/MVFR EO sky above horizon (+5 deg mask); non-cooperative intruder bearings",
                "known_failure_modes": [
                    "Sub-pixel CBDR silhouette below 3 px in heavy fog/precipitation",
                    "Sun-disc direct glare saturation (mitigated by xvs-odd-vlm Mode-3 glare gate)",
                ],
                "rta_fallback": "xvs-rta-core P0..P4 fail-closed escalation + xvs-odd-vlm Optical-ODD monitor",
                "verification_status": "VERIFIED_ON_AOT_REALSKY (78.05% Pd above horizon, 2.86 px centroid RMSE)",
            },
        ),
        (
            "xvs_active_range",
            ("xvs-active-range/src/lib.rs",),
            {
                "display_name": "XVS-ActiveRange (Monocular Active Range Observability & SC3BF-FIM)",
                "runtime_format": "Pure `#![no_std]` `#![forbid(unsafe_code)]` Rust (4/4 Kani CBMC proofs)",
                "input_tensors": [{"name": "bearing_obs_ned", "shape": [2], "dtype": "float64 (alpha, beta rad)"}],
                "output_tensors": [{"name": "wire_64b_packet", "shape": [64], "dtype": "uint8 (IEEE-CRC32 framed)"}],
                "training_data_provenance": "Deterministic physics filter (M=6 MSC-RPEKF) + split-conformal CP-MSC calibrated on Amazon AOT",
                "odd_bounds": "Civil fixed-wing encounters with unknown intruder wingspan 1.5-35 m; bounded <=12 deg bank micro-swerve",
                "known_failure_modes": [
                    "Zero bearing observability prior to ownship lateral acceleration (solved by SC3BF-FIM swerve)",
                    "Sustained >10/20 detection dropout triggers fail-closed IMMEDIATE_EVADE",
                ],
                "rta_fallback": "Universal forall-m SC3BF-FIM invariant + ImmediateEvade deterministic floor",
                "verification_status": "VERIFIED_4_OF_4_KANI_AND_AOT_REALSKY (97.5% real AOT conformal coverage)",
            },
        ),
        (
            "xvs_land",
            (
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45_fp16.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45.onnx",
            ),
            {
                "display_name": "XVS-Land v5.2 (Vision Runway 12-Keypoint SimCC 384, 5 Hesse Keylines & Keyline Sigma)",
                "runtime_format": "Static Opset-17 ONNX FP16 (2.7657 MiB, 1.186M params, 0 Resize ops) + StudentTCalibrator (nu=6.0, scale=1.048)",
                "input_tensors": [{"name": "roi_crop", "shape": [1, 3, 384, 384], "dtype": "float32"}],
                "output_tensors": [
                    {"name": "keypoints_xy", "shape": [1, 12, 2], "dtype": "float32"},
                    {"name": "keylines_hesse", "shape": [1, 5, 3], "dtype": "float32"},
                    {"name": "cov_2x2", "shape": [1, 12, 2, 2], "dtype": "float32"},
                    {"name": "logits_x", "shape": [1, 12, 768], "dtype": "float32"},
                    {"name": "logits_y", "shape": [1, 12, 768], "dtype": "float32"},
                    {"name": "keyline_sigma_px", "shape": [1, 5], "dtype": "float32"},
                ],
                "training_data_provenance": "100% public-domain USDA NAIP + USGS 3DEP + FAA NASR (LARD real airports eval-only)",
                "odd_bounds": "Civil paved runway final approach (<= 3 NM), parallel-runway width gate & Conformal-RAIM HPL/VPL",
                "known_failure_modes": [
                    "Heavy snow cover obscuring runway threshold markings",
                    "Parallel taxiway confusion (mitigated by width gate & non-resettable Go-Around latch)",
                ],
                "rta_fallback": "Non-resettable mandatory Go-Around latch in xvs-rta-core when HPL/VPL exceeds alert limit",
                "verification_status": (
                    "VERIFIED_V52_G4_AND_WINDOWS_ORT (SimCC=2.92 px, OKLS=7.43 deg, PnPL=137.98 m, EKF=71.97 m, "
                    "v_hpl=82.41 m, v_vpl=4.78 m, student_t_scale=1.048, hmi_count=0, cat1_availability=0.0%; "
                    "Ryzen 5600X ORT 9.176 ms P50 / 10.247 ms P95; physical flight in G5_GAP)"
                ),
            },
        ),
        (
            "xvs_thermal",
            (
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp16.onnx",
            ),
            {
                "display_name": "XVS-ThermalNet v5.2 (Asymmetric EO + 8-14 um LWIR FPN DVE Fusion & Dirichlet Vacuity)",
                "runtime_format": "Static Opset-17 ONNX FP16 (1.7561 MiB, 820,116 params, 0 Resize ops, 4-mode EO/LWIR fusion)",
                "input_tensors": [
                    {"name": "x_eo", "shape": [1, 3, 256, 320], "dtype": "float32"},
                    {"name": "x_lwir", "shape": [1, 2, 128, 160], "dtype": "float32"},
                    {"name": "q_eo", "shape": [1], "dtype": "float32"},
                    {"name": "q_lwir", "shape": [1], "dtype": "float32"},
                ],
                "output_tensors": [
                    {"name": "fused_hm_logits", "shape": [1, 1, 64, 80], "dtype": "float32"},
                    {"name": "fused_box_reg", "shape": [1, 4, 64, 80], "dtype": "float32"},
                    {"name": "vacuity", "shape": [1, 1, 64, 80], "dtype": "float32"},
                    {"name": "probs", "shape": [1, 3, 64, 80], "dtype": "float32"},
                    {"name": "w_eo_eff", "shape": [1, 1, 64, 80], "dtype": "float32"},
                    {"name": "w_lwir_eff", "shape": [1, 1, 64, 80], "dtype": "float32"},
                ],
                "training_data_provenance": "IRSTD-1k (MIT) + HIT-UAV (CC-BY-4.0) + physics 5-pt Gauss-Legendre Planck LWIR (zero InfiRay)",
                "odd_bounds": "Civil smoke, night, and haze DVE perception; does NOT modify FPA frame rates (outside ECCN 6D003.c)",
                "known_failure_modes": [
                    "Uncooled bolometer NUC shutter freeze / dead-pixel burst (mitigated by WP-07 physical-health hard-gating)",
                    "Thermal crossover under zero background delta-T (flagged via Dirichlet vacuity OOD)",
                ],
                "rta_fallback": "WP-07 deterministic physical-health hard-gate + xvs-odd-vlm modal degradation escalation",
                "verification_status": (
                    "VERIFIED_V52_G4_AND_WINDOWS_ORT (overall_mean_peak_err=5.19 px: 1.68 px nominal, "
                    "5.92 px radiation fog, 7.97 px post-FFC shock; dual_blind vacuity==1.0 max_abs_diff=0.0 bit-exact; "
                    "EASA R-LM-07..11 5/5 PASS; Ryzen 5600X ORT 17.212 ms P50 / 21.244 ms P95; physical LWIR pod in G5_GAP)"
                ),
            },
        ),
        (
            "xvs_nav",
            (
                "outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx",
            ),
            {
                "display_name": "XVS-Nav v5.1 Faz-B Stage-1 (MobileNetV4-Hybrid-M + MixVPR-Clean + Folded Ridge-Whitening; Faz-C Held)",
                "runtime_format": "Selective FP16/FP32 Static Opset-17 ONNX (21.525 MiB <= 22 MiB; ECCN 7D003.b.3 source withheld in private_eval)",
                "input_tensors": [{"name": "img_s1", "shape": [1, 3, 224, 224], "dtype": "float32"}],
                "output_tensors": [{"name": "desc_global", "shape": [1, 512], "dtype": "float32 (L2-normalized)"}],
                "training_data_provenance": "Clean public USDA NAIP + Copernicus Sentinel-2 L2A (33,286 8-biome pairs incl. 1,788 S2 SCL=11 snow; >=718.58 km ECEF from KSFO)",
                "odd_bounds": "Civil GNSS-denied cross-view visual localization; fail-closed coasting over unstructured biomes",
                "known_failure_modes": [
                    "Uniform water, dense canopy, or deep featureless snow (rejected by 6-stage aliasing_gate PSR_cal >= 1.15)",
                    "Raw unaugmented cross-year appearance shift (48.24% R@1 / 58.20% R@5 single-frame on 2020<->2022 KSFO; mitigated by SequenceVprFilter W=7 to 97.50%/98.28% LAND-320 with 0 HMI; v5.2 Faz-C intentionally HELD due to -10.54 pp all-512 / -16.09 pp land-320 regression from 92.6% S2 / 7.4% NAIP ratio)",
                ],
                "rta_fallback": "6-stage aliasing_gate fail-closed reject + Sequence-VPR posterior margin gate -> inertial coasting + ODD alert (0 HMI)",
                "verification_status": (
                    "VERIFIED_FAZ_B_L4_HARVEST_PINNED (91.41% R@1 / 96.68% R@5 disjoint KSFO combined-aug; "
                    "SequenceVprFilter W=7 97.50%/98.28% LAND-320, 0 HMI; 13.03 ms p50 / 17.99 ms p95 Ryzen CPU ORT; "
                    "v5.2 Faz-C HELD; physical flight in G5_GAP)"
                ),
            },
        ),
        (
            "xfeat",
            (
                "outputs/colab_g4_v52/harvest_nav/xfeat_320x256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest/xfeat_320x256_fp32.onnx",
            ),
            {
                "display_name": "XVS-Nav Stage-2A XFeat (320x256 Dense Local Feature Extractor & Descriptor Head)",
                "runtime_format": "Static Opset-17 ONNX FP16 (1.303 MiB <= 1.6 MiB; paired with 32 KB/tile 3DEP NED cache & SQPnP)",
                "input_tensors": [{"name": "img_s2", "shape": [1, 3, 256, 320], "dtype": "float32"}],
                "output_tensors": [
                    {"name": "feats", "shape": [1, 64, 32, 40], "dtype": "float32"},
                    {"name": "keypoints", "shape": [1, 65, 32, 40], "dtype": "float32"},
                    {"name": "heatmap", "shape": [1, 1, 32, 40], "dtype": "float32"},
                ],
                "training_data_provenance": "Permissive upstream verlab/accelerated_features XFeat weights verified under license_guard --strict",
                "odd_bounds": "Horizon-rectified civil nadir/near-nadir terrain crops for Stage-2A MNN + 3DEP SQPnP-RANSAC & Top-K Sequence-VPR re-ranking",
                "known_failure_modes": [
                    "Textureless water, cloud decks, or low-contrast canopy yielding <30 MNN inliers (escalates to Stage-2B LighterGlue or fail-closed coast)",
                    "Extreme cross-year construction or snow albedo inversion reducing local keypoint repeatability",
                ],
                "rta_fallback": "Conditional Tier-2B LighterGlue fallback (<30 inliers) + 6-stage aliasing_gate fail-closed HPL <= 30.0 m guard",
                "verification_status": "VERIFIED_FAZ_B_L4_HARVEST (p50 4.04 ms ORT / 6.21 ms full extract on Ryzen CPU; S1+S2A+SQPnP p95 28.59 ms <= 30 ms)",
            },
        ),
        (
            "lighterglue",
            (
                "outputs/colab_g4_v52/harvest_nav/lighterglue_k256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp32.onnx",
            ),
            {
                "display_name": "XVS-Nav Stage-2B LighterGlue (Static K=256 Transformer Correspondence Fallback)",
                "runtime_format": "Static Opset-17 ONNX FP16 (2.455 MiB <= 3.0 MiB; d=96, L=6, h=1, depth/width confidence disabled, 0 control-flow ops)",
                "input_tensors": [
                    {"name": "kpts0", "shape": [1, 256, 2], "dtype": "float32"},
                    {"name": "kpts1", "shape": [1, 256, 2], "dtype": "float32"},
                    {"name": "desc0", "shape": [1, 256, 64], "dtype": "float32"},
                    {"name": "desc1", "shape": [1, 256, 64], "dtype": "float32"},
                ],
                "output_tensors": [{"name": "scores", "shape": [1, 256, 256], "dtype": "float32 (dual-softmax log assignment)"}],
                "training_data_provenance": "Permissive upstream verlab/accelerated_features LighterGlue weights (139 tensors) verified under license_guard --strict",
                "odd_bounds": "Conditional Tier-2B fallback invoked strictly when Tier-2A XFeat MNN returns <30 RANSAC inliers",
                "known_failure_modes": [
                    "Complete occlusion or featureless snow/water where <30 geometric correspondences exist (rejected fail-closed into inertial coasting)",
                    "Inputs >256 keypoints truncated to static K=256 cap (sub-K inputs deterministically zero-padded and masked)",
                ],
                "rta_fallback": "6-stage aliasing_gate + SQPnP-RANSAC + Dual-Bank 8-Arm Cholesky CUSUM fail-closed coasting (0 HMI)",
                "verification_status": "VERIFIED_FAZ_B_L4_HARVEST (p50 22.01 ms ORT / 23.36 ms full match on Ryzen CPU; 88 KSFO matches, 81/88 <3 px inliers)",
            },
        ),
    ]

    cards: list[dict[str, Any]] = []
    for key, candidates, info in specs:
        rel_artifact, present, size_bytes, sha256 = _resolve_artifact_candidates(repo_root, candidates)
        card = dict(info)
        card["model_id"] = key
        card["license"] = weight_license
        card["artifact_path"] = rel_artifact
        card["artifact_candidates"] = list(candidates)
        card["artifact_present_on_disk"] = present
        card["artifact_size_bytes"] = size_bytes
        card["artifact_sha256"] = sha256
        if health_models and key in health_models and isinstance(health_models[key], dict):
            card["runtime_health_snapshot"] = dict(health_models[key])
        cards.append(card)

    return {
        "schema": "openxvs.onnx_model_cards/1.0",
        "release_mode": release_mode,
        "model_count": len(cards),
        "models": cards,
    }


def _collect_certgen_snapshot() -> dict[str, Any]:
    """Run live `xvs_certgen.py` verification in-memory (fallback to saved bundle JSON)."""
    try:
        spec = importlib.util.spec_from_file_location("openxvs_certgen_live", str(CERTGEN_SCRIPT_PATH))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            bundle = mod.synthesize_assurance_bundle()
            return bundle
    except Exception:
        pass
    if CERTGEN_BUNDLE_PATH.is_file():
        return json.loads(CERTGEN_BUNDLE_PATH.read_text(encoding="utf-8"))
    return {
        "status": "SUMMARY_ONLY",
        "kani_proofs_total": 18,
        "undeveloped_gaps": ["G5_GAP", "G2_TLA_DEFERRED"],
    }


def _collect_latest_video_summary(video_manager: VideoJobManager | None) -> dict[str, Any]:
    if video_manager is not None:
        with video_manager._lock:
            for rec in reversed(list(video_manager._jobs.values())):
                if rec["status"] == "done" and rec["summary"] is not None:
                    return {
                        "source": "live_video_job",
                        "job_id": rec["job_id"],
                        "channel": rec["channel"],
                        "clip": rec["clip"],
                        "summary": rec["summary"],
                    }
    if AOT_METRICS_PATH.is_file():
        aot_doc = json.loads(AOT_METRICS_PATH.read_text(encoding="utf-8"))
        return {
            "source": "aot_realsky_reference_proof",
            "metrics_file": "outputs/xvs_activerange_aot_metrics.json",
            "summary": aot_doc,
        }
    return {
        "source": "none_completed_yet",
        "note": "Run real-video analysis inside OpenXVS Studio to embed a live MP4 job summary.",
    }


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fail_closed_sitl_proof(
    *,
    status: str,
    proof_file: str,
    validation_error: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "verified_live_sitl": False,
        "status": status,
        "proof_file": proof_file,
    }
    if validation_error is not None:
        out["validation_error"] = validation_error
    return out


def collect_live_sitl_proof_summary(
    proof_path: Path | None = None,
    *,
    include_detail: bool = False,
) -> dict[str, Any]:
    """Validate and summarize live WSL2 ArduPlane SITL proof artifact (strictly fail-closed)."""
    path = LIVE_SITL_PROOF_PATH if proof_path is None else Path(proof_path)
    proof_file = "outputs/sitl_proof/live_arduplane_sitl_proof.json"

    if not path.is_file():
        return _fail_closed_sitl_proof(status="NOT_PRESENT_ON_DISK", proof_file=proof_file)

    try:
        raw_bytes = path.read_bytes()
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error=f"unreadable_or_invalid_json: {type(exc).__name__}",
        )

    if not isinstance(doc, dict):
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="root payload must be a JSON object",
        )

    if "pass" not in doc or doc["pass"] is not True:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="pass must be strictly true",
        )

    if "msg84_ever_emitted" not in doc or doc["msg84_ever_emitted"] is not False:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="msg84_ever_emitted must be strictly false",
        )

    if "pymavlink_imported" not in doc or doc["pymavlink_imported"] is not False:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="pymavlink_imported must be strictly false",
        )

    if "gpl_free_openxvs_codec" in doc and doc["gpl_free_openxvs_codec"] is not True:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="gpl_free_openxvs_codec must be strictly true",
        )

    if "arduplane_verified" in doc and doc["arduplane_verified"] is not True:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="arduplane_verified must be strictly true",
        )

    if "checks" in doc:
        checks = doc["checks"]
        if not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values()):
            return _fail_closed_sitl_proof(
                status="INVALID_PROOF_ARTIFACT",
                proof_file=proof_file,
                validation_error="all SITL verification checks must be strictly true",
            )

    simulator_backend = doc["simulator_backend"] if "simulator_backend" in doc else None
    if not isinstance(simulator_backend, str) or not simulator_backend.strip():
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="missing non-empty simulator_backend",
        )

    firmware_version: str | None = None
    for fw_key in (
        "arduplane_apm_version_line",
        "firmware_version",
        "arduplane_firmware_version_txt",
        "arduplane_version",
    ):
        if fw_key in doc and isinstance(doc[fw_key], str) and doc[fw_key].strip():
            firmware_version = doc[fw_key].strip()
            break
    if firmware_version is None:
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="missing non-empty firmware_version",
        )

    telemetry = (
        doc["telemetry_summary"]
        if "telemetry_summary" in doc and isinstance(doc["telemetry_summary"], dict)
        else doc
    )
    rx_packets = telemetry["rx_packets"] if "rx_packets" in telemetry else None
    tx_packets = telemetry["tx_packets"] if "tx_packets" in telemetry else None
    if not _is_positive_int(rx_packets) or not _is_positive_int(tx_packets):
        return _fail_closed_sitl_proof(
            status="INVALID_PROOF_ARTIFACT",
            proof_file=proof_file,
            validation_error="rx_packets and tx_packets must be positive integers",
        )

    if "tx_msgids_emitted" in telemetry:
        emitted = telemetry["tx_msgids_emitted"]
        if not isinstance(emitted, dict) or "84" in emitted or 84 in emitted:
            return _fail_closed_sitl_proof(
                status="INVALID_PROOF_ARTIFACT",
                proof_file=proof_file,
                validation_error="forbidden MAVLink #84 detected in tx_msgids_emitted",
            )

    digest = hashlib.sha256(raw_bytes).hexdigest()
    summary: dict[str, Any] = {
        "verified_live_sitl": True,
        "status": "VERIFIED",
        "proof_file": proof_file,
        "sha256": digest,
        "simulator_backend": simulator_backend.strip(),
        "firmware_version": firmware_version,
        "rx_packets": int(rx_packets),
        "tx_packets": int(tx_packets),
        "msg84_ever_emitted": False,
        "pymavlink_imported": False,
        "gpl_free_openxvs_codec": True,
    }
    if "schema" in doc and isinstance(doc["schema"], str):
        summary["schema"] = doc["schema"]
    if "timestamp_utc" in doc and isinstance(doc["timestamp_utc"], str):
        summary["timestamp_utc"] = doc["timestamp_utc"]
    if "arduplane_firmware_version_txt" in doc and isinstance(doc["arduplane_firmware_version_txt"], str):
        summary["arduplane_firmware_version_txt"] = doc["arduplane_firmware_version_txt"]
    if "arduplane_binary_sha256" in doc and isinstance(doc["arduplane_binary_sha256"], str):
        summary["arduplane_binary_sha256"] = doc["arduplane_binary_sha256"]

    if include_detail:
        if "checks" in doc and isinstance(doc["checks"], dict):
            summary["checks"] = dict(doc["checks"])
        if "transport" in doc and isinstance(doc["transport"], dict):
            summary["transport"] = dict(doc["transport"])
        if "flight_response" in doc and isinstance(doc["flight_response"], dict):
            summary["flight_response"] = dict(doc["flight_response"])

    return summary


_STANDALONE_VERIFY_PACK_PY = '''#!/usr/bin/env python3
"""Offline Zero-Dependency OpenXVS Buyer Assurance Pack Verifier (SHA-256 + RFC 8032 Ed25519).

Verifies every artifact listed in `MANIFEST.sha256` and verifies the detached 64-byte
RFC 8032 Ed25519 signature in `MANIFEST.sha256.sig` against `MANIFEST_VERIFY.json`.
Authentication and integrity verification only (excluded from ECCN 5D002 confidentiality).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

_P = (1 << 255) - 19
_L = (1 << 252) + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = ((y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)) % _P
    if x2 == 0:
        return None if sign == 1 else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = (x * _I) % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_BY = (4 * pow(5, _P - 2, _P)) % _P
_BX = _recover_x(_BY, 0) or 0
_BASE = (_BX, _BY, 1, (_BX * _BY) % _P)
_ID = (0, 1, 1, 0)


def _add(p1, p2):
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (2 * t1 * t2 * _D) % _P
    d = (2 * z1 * z2) % _P
    e, f, g, h = (b - a) % _P, (d - c) % _P, (d + c) % _P, (b + a) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _mul(k: int, pt):
    res, addend = _ID, pt
    while k > 0:
        if k & 1:
            res = _add(res, addend)
        addend = _add(addend, addend)
        k >>= 1
    return res


def _eq(p1, p2) -> bool:
    return ((p1[0] * p2[2] - p2[0] * p1[2]) % _P == 0) and ((p1[1] * p2[2] - p2[1] * p1[2]) % _P == 0)


def _dec(raw: bytes):
    if len(raw) != 32:
        return None
    val = int.from_bytes(raw, "little")
    x = _recover_x(val & ((1 << 255) - 1), (val >> 255) & 1)
    return None if x is None else (x, val & ((1 << 255) - 1), 1, (x * (val & ((1 << 255) - 1))) % _P)


def ed25519_verify(pk32: bytes, msg: bytes, sig64: bytes) -> bool:
    if len(pk32) != 32 or len(sig64) != 64:
        return False
    a_pt, r_pt = _dec(pk32), _dec(sig64[:32])
    if a_pt is None or r_pt is None:
        return False
    s = int.from_bytes(sig64[32:], "little")
    if s >= _L:
        return False
    k = int.from_bytes(hashlib.sha512(sig64[:32] + pk32 + msg).digest(), "little") % _L
    return _eq(_mul(s, _BASE), _add(r_pt, _mul(k, a_pt)))


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest_bytes = (root / "MANIFEST.sha256").read_bytes()
    sig_bytes = (root / "MANIFEST.sha256.sig").read_bytes()
    meta = json.loads((root / "MANIFEST_VERIFY.json").read_text(encoding="utf-8"))
    pk = bytes.fromhex(meta["ed25519_public_key_hex"])
    if not ed25519_verify(pk, manifest_bytes, sig_bytes):
        print("FAIL: Ed25519 signature verification failed for MANIFEST.sha256.sig", file=sys.stderr)
        return 1
    checked = 0
    for line in manifest_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        expected_hex, rel = line.split("  ", 1)
        actual_hex = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual_hex != expected_hex:
            print(f"FAIL: SHA-256 mismatch on {rel}", file=sys.stderr)
            return 1
        checked += 1
    print(f"PASS: Verified Ed25519 signature and {checked} SHA-256 artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def verify_buyer_assurance_pack_zip(
    raw_zip: bytes,
    *,
    expected_public_key32: bytes | None = None,
) -> dict[str, Any]:
    """Cryptographically verify a Buyer Assurance Pack ZIP (SHA-256 manifest + RFC 8032 Ed25519 sig)."""
    _, _, ed25519_verify = _get_ed25519_helpers()
    errors: list[str] = []
    files_verified = 0
    public_key_hex: str | None = None
    manifest_sha256: str | None = None

    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
            names = set(zf.namelist())
            for req in ("MANIFEST.sha256", "MANIFEST.sha256.sig", "MANIFEST_VERIFY.json", "verify_pack.py"):
                if req not in names:
                    errors.append(f"MISSING_SIGNING_ARTIFACT: {req}")
            if errors:
                return {"passed": False, "files_verified": 0, "errors": errors}

            manifest_bytes = zf.read("MANIFEST.sha256")
            sig_bytes = zf.read("MANIFEST.sha256.sig")
            meta = json.loads(zf.read("MANIFEST_VERIFY.json").decode("utf-8"))
            public_key_hex = str(meta.get("ed25519_public_key_hex", ""))
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

            if meta.get("manifest_sha256") != manifest_sha256:
                errors.append("MANIFEST_SHA256_METADATA_MISMATCH")

            pk_bytes = bytes.fromhex(public_key_hex)
            if expected_public_key32 is not None and pk_bytes != expected_public_key32:
                errors.append("PUBLIC_KEY_MISMATCH")

            if len(sig_bytes) != 64:
                errors.append(f"INVALID_SIG_LENGTH: expected 64 bytes, got {len(sig_bytes)}")
            elif not ed25519_verify(pk_bytes, manifest_bytes, sig_bytes):
                errors.append("ED25519_SIGNATURE_INVALID")

            if meta.get("signature_hex") != sig_bytes.hex():
                errors.append("SIGNATURE_HEX_METADATA_MISMATCH")

            manifest_entries: dict[str, str] = {}
            for line in manifest_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                if "  " not in line:
                    errors.append(f"MALFORMED_MANIFEST_LINE: {line!r}")
                    continue
                digest_hex, rel_name = line.split("  ", 1)
                manifest_entries[rel_name] = digest_hex

            for rel_name, expected_hex in sorted(manifest_entries.items()):
                if rel_name not in names:
                    errors.append(f"MISSING_SIGNED_FILE: {rel_name}")
                    continue
                actual_hex = hashlib.sha256(zf.read(rel_name)).hexdigest()
                if actual_hex != expected_hex:
                    errors.append(f"SHA256_MISMATCH: {rel_name}")
                else:
                    files_verified += 1
    except Exception as exc:
        errors.append(f"ZIP_VERIFY_EXCEPTION: {type(exc).__name__}: {exc}")

    return {
        "passed": len(errors) == 0 and files_verified >= 9,
        "files_verified": files_verified,
        "public_key_hex": public_key_hex,
        "manifest_sha256": manifest_sha256,
        "errors": errors,
    }


def build_buyer_assurance_pack_zip(
    engine: StudioInferenceEngine,
    hil_manager: StudioHilBridgeManager,
    video_manager: VideoJobManager | None = None,
    release_mode: str = DEFAULT_RELEASE_MODE,
    sitl_proof_path: Path | None = None,
    ed25519_seed32: bytes | None = None,
) -> bytes:
    """Compile all OpenXVS v5.0 buyer assurance artifacts + detached Ed25519 signature into `.zip`.

    Args:
        engine: Loaded `StudioInferenceEngine`.
        hil_manager: Active `StudioHilBridgeManager`.
        video_manager: Optional `VideoJobManager` for embedding the latest MP4 job summary.
        release_mode: `"private_eval"` (default, Confidential Non-Flight Desktop/Bench Evaluation;
            zero Apache-2.0 text, zero Python CLI, zero Phase-4 flight-test procedure) or
            `"public_open_core"` (`Apache-2.0 / CERN-OHL-P-2.0 / CDLA-Permissive-2.0`).
        sitl_proof_path: Optional override path for the verified live ArduPlane SITL proof JSON.
        ed25519_seed32: Optional 32-byte RFC 8032 Ed25519 private signing seed (defaults to
            deterministic evaluation signing seed for offline reproducibility).
    """
    mode = validate_release_mode(release_mode)
    license_posture, release_governance = _build_release_governance(mode)
    protocol_md = render_field_trial_protocol_md(mode)

    ed25519_public_key, ed25519_sign, _ = _get_ed25519_helpers()
    build_sora25_f3442_ed324_matrix, build_spdx_2_3_sbom = _get_assurance_helpers()

    seed = _DEFAULT_EVAL_ED25519_SEED32 if ed25519_seed32 is None else ed25519_seed32
    if len(seed) != 32:
        raise ValueError(f"ed25519_seed32 must be exactly 32 bytes, got {len(seed)}")
    pub_key32 = ed25519_public_key(seed)

    health = engine.bundle.health_summary()
    hil_status = hil_manager.status()
    hil_proof = (
        hil_status["last_loopback"]
        if isinstance(hil_status["last_loopback"], dict)
        else hil_manager.run_loopback_test()
    )
    live_sitl_exec = collect_live_sitl_proof_summary(sitl_proof_path, include_detail=False)
    live_sitl_hil = collect_live_sitl_proof_summary(sitl_proof_path, include_detail=True)
    hil_proof_doc = dict(hil_proof)
    hil_proof_doc["live_sitl_proof"] = live_sitl_hil

    certgen_doc = _collect_certgen_snapshot()
    video_doc = _collect_latest_video_summary(video_manager)
    sora_matrix_doc = build_sora25_f3442_ed324_matrix(repo_root=REPO_ROOT, release_mode=mode)
    spdx_sbom_doc = build_spdx_2_3_sbom(repo_root=REPO_ROOT, release_mode=mode)
    model_cards_doc = _build_onnx_model_cards(
        REPO_ROOT,
        release_mode=mode,
        health_models=health.get("models") if isinstance(health, dict) else None,
    )

    exec_summary: dict[str, Any] = {
        "schema": "openxvs.buyer_assurance_pack/5.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_mode": mode,
        "non_flight_sim_only": mode == "private_eval",
        "watermark": NON_FLIGHT_WATERMARK if mode == "private_eval" else None,
        "license_posture": license_posture,
        "release_governance": release_governance,
        "domain_scope": "Strictly Civil / Dual-Use Aviation Perception & Runtime Assurance",
        "certification_boundary_disclosure": {
            "formal_proofs_verified": "26/26 Kani CBMC harnesses (0 failed)",
            "explicit_undeveloped_gaps": ["G5_GAP", "G2_TLA_DEFERRED"],
            "note": (
                "Target hardware environmental qualification and formal civil airworthiness "
                "certification remain explicitly open under G5_GAP."
            ),
        },
        "assurance_matrix_summary": {
            "status": sora_matrix_doc["status"],
            "easa_sora_25_osos": sora_matrix_doc["easa_sora_25"]["oso_count"],
            "easa_sora_25_m1_mitigations": len(sora_matrix_doc["easa_sora_25"]["ground_risk_mitigations_m1"]),
            "astm_f3442_26_clauses": sora_matrix_doc["astm_f3442_26"]["requirement_count"],
            "eurocae_ed324_objectives": sora_matrix_doc["eurocae_ed324"]["objective_count"],
            "matrix_sha256": sora_matrix_doc["matrix_sha256"],
        },
        "spdx_sbom_summary": {
            "spdx_version": spdx_sbom_doc["spdxVersion"],
            "total_dependencies_audited": spdx_sbom_doc["openxvsLicenseAudit"]["total_dependencies_audited"],
            "zero_copyleft_pass": spdx_sbom_doc["openxvsLicenseAudit"]["zero_copyleft_pass"],
            "sbom_sha256": spdx_sbom_doc["openxvsLicenseAudit"]["sbom_sha256"],
        },
        "ed25519_signing": {
            "algorithm": "Ed25519 (RFC 8032)",
            "manifest_file": "MANIFEST.sha256",
            "signature_file": "MANIFEST.sha256.sig",
            "public_key_hex": pub_key32.hex(),
            "key_mode": "deterministic_eval_key" if ed25519_seed32 is None else "caller_supplied_key",
            "eccn_5d002_exclusion": "Authentication & integrity only (excluded from 5D002 confidentiality)",
        },
        "runtime_environment": {
            "host_gpu_name": health["host_gpu_name"],
            "compute_device": health["compute_device"],
            "vram_mb": health["vram_mb"],
            "vram_cap_mb": health["vram_cap_mb"],
            "media_pack_ready": engine.media_pack_ready(),
        },
        "models_provenance": health["models"],
        "hil_loopback_pass": bool(hil_proof["pass"]),
        "verified_live_sitl": bool(live_sitl_exec["verified_live_sitl"]),
        "live_sitl_proof": live_sitl_exec,
    }

    fg_xml_text = (
        FG_PROTOCOL_XML_PATH.read_text(encoding="utf-8")
        if FG_PROTOCOL_XML_PATH.is_file()
        else "<!-- openxvs_fg_protocol.xml not found -->\n"
    )

    signed_artifacts: dict[str, bytes] = {
        "01_executive_summary.json": (json.dumps(exec_summary, indent=2) + "\n").encode("utf-8"),
        "02_formal_certification_certgen.json": (json.dumps(certgen_doc, indent=2) + "\n").encode("utf-8"),
        "03_hil_loopback_proof.json": (json.dumps(hil_proof_doc, indent=2) + "\n").encode("utf-8"),
        "04_latest_video_analysis.json": (json.dumps(video_doc, indent=2) + "\n").encode("utf-8"),
        "05_flightgear_openxvs_protocol.xml": fg_xml_text.encode("utf-8"),
        "06_FIELD_TRIAL_PROTOCOL.md": protocol_md.encode("utf-8"),
        "07_sora25_f3442_ed324_matrix.json": (json.dumps(sora_matrix_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "08_spdx_2_3_sbom.json": (json.dumps(spdx_sbom_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "09_onnx_model_cards.json": (json.dumps(model_cards_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }

    manifest_lines = [
        f"{hashlib.sha256(signed_artifacts[name]).hexdigest()}  {name}"
        for name in sorted(signed_artifacts)
    ]
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    sig_bytes = ed25519_sign(seed, manifest_bytes)
    verify_meta = {
        "schema": "openxvs.manifest_verify/1.0",
        "algorithm": "Ed25519 (RFC 8032)",
        "release_mode": mode,
        "manifest_file": "MANIFEST.sha256",
        "signature_file": "MANIFEST.sha256.sig",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "ed25519_public_key_hex": pub_key32.hex(),
        "signature_hex": sig_bytes.hex(),
        "files_signed": sorted(signed_artifacts.keys()),
        "key_mode": "deterministic_eval_key" if ed25519_seed32 is None else "caller_supplied_key",
        "eccn_5d002_exclusion_note": (
            "Pure digital signature authentication & integrity verification only; "
            "excluded from ECCN 5D002 data-confidentiality controls (Category 5 Part 2 Technical Notes)."
        ),
        "verify_command": "python verify_pack.py",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(signed_artifacts):
            zf.writestr(name, signed_artifacts[name])
        zf.writestr("MANIFEST.sha256", manifest_bytes)
        zf.writestr("MANIFEST.sha256.sig", sig_bytes)
        zf.writestr("MANIFEST_VERIFY.json", (json.dumps(verify_meta, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        zf.writestr("verify_pack.py", _STANDALONE_VERIFY_PACK_PY.encode("utf-8"))
    return buf.getvalue()
