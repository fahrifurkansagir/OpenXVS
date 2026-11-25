"""DR-09 Edge VLM Semantic Safety Monitor Runner, License Guard & MAVLink #50003 Bridge.

Strictly civil/dual-use unmanned aviation safety peripheral per:
- OpenXVS DR-09 (Asymmetric Simplex VLM Semantic Safety Monitor, IDAL D/E isolation)
- OpenXVS DR-13 (5-Mode Zero-ML Optical-ODD Health Monitor, AF447 zero-airdata lock)
- EASA Concept Paper Proposed Issue 03 (Table 2: IDAL D ceiling for Large OTS VLMs;
  Objective RA-08 aleatoric vs epistemic uncertainty separation; frozen weights in flight)
- FAA DOT/FAA/TC-26/20 (Overarching Properties: Intent, Correctness, Innocuity)
- ASTM F3269-21 §1.2 (VLM strictly outside certified Safety Monitor; zero actuator authority)

Wire parity:
- 32-byte IEEE-802.3 CRC32 `VlmAdvisoryPacket` matches frozen Rust crate `xvs-odd-vlm` bit-for-bit.
- MAVLink 2.0 `#50003 POD_VLM_SEMANTIC_VETO` matches frozen `configs/mavlink/pod_daa.xml` 1:1.
- Strictly zero GPL dependencies, zero `pymavlink`, zero ENU coordinates, zero unannotated `.get()`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import math
import re
import socket
import time
from typing import TYPE_CHECKING, AbstractSet, Any, Mapping, Protocol, Sequence
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

from xvs.v2.xvs_mavlink.pod_daa_codec import (
    PodVlmSemanticVeto,
    encode_frame,
)

if TYPE_CHECKING:
    from xvs.v2.xvs_optical_odd.monitor import (
        DeterministicOpticalOddMonitor,
        OpticalOddConfig,
        OpticalOddDecision,
    )

from .gate import (
    NUM_AZIMUTH_SECTORS,
    NUM_RANKED_ZONES,
    PACKET_SIZE,
    UNUSED_ZONE_ID,
    AsymmetricSimplexVlmGate,
    GateDecision,
    GateStatus,
    OpticalOddReading,
    OpticalOddThresholds,
    VlmAdvisoryPacket,
    VlmGateConfig,
    VlmHazard,
)

# ---------------------------------------------------------------------------
# Canonical DR-09 Constants (1:1 Parity with Frozen `xvs-odd-vlm` & `pod_daa.xml`)
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS: int = 8
MAX_FLZ_ID: int = 16
WATCHDOG_MAX_AGE_MS: int = 200
MAX_HTTP_TIMEOUT_MS: int = 199
UNKNOWN_AZIMUTH_SECTOR: int = 255
MAX_RAW_OUTPUT_CHARS: int = 120

HTTP_BACKEND_LABEL: str = "OPENAI_COMPATIBLE_LOCAL_HTTP"
OFFLINE_STUB_BACKEND_LABEL: str = "OFFLINE_CONFORMAL_BENCH_STUB"
OFFLINE_STUB_PROVENANCE_NOTICE: str = (
    "OFFLINE_CONFORMAL_BENCH_STUB: deterministic calibration and fault-injection stub "
    "(NOT a neural VLM; used strictly for deterministic CI and gate verification when no "
    "local OpenAI-compatible VLM server is attached)."
)

GATE_STATUS_TO_MAVLINK_VERDICT: Mapping[GateStatus, int] = {
    GateStatus.NOMINAL_ADVISORY: 0,
    GateStatus.VLM_OFFLINE: 1,
    GateStatus.SCHEMA_OR_CRC_FAIL: 2,
    GateStatus.OPTICAL_ODD_MASKED: 3,
    GateStatus.CONFORMAL_ABSTAIN: 4,
    GateStatus.DEBOUNCE_PENDING: 5,
}

VLM_HAZARD_BIT: Mapping[VlmHazard, int] = {
    VlmHazard.NONE: 0,
    VlmHazard.RUNWAY_INCURSION: 1 << 0,
    VlmHazard.BIRD_FLOCK: 1 << 1,
    VlmHazard.CONVECTIVE_VIRGA: 1 << 2,
    VlmHazard.CRANE_PROXY: 1 << 3,
    VlmHazard.FLZ_HAZARD: 1 << 4,
}


def vlm_hazard_bit(hazard: VlmHazard | int) -> int:
    """Return canonical `xvs-odd-vlm` hazard bit (`1 << (h - 1)` for `1..5`, else `0`)."""
    return VLM_HAZARD_BIT[VlmHazard(int(hazard))]


def active_vetoes_to_mask(accepted_vlm_vetoes: AbstractSet[str]) -> int:
    """Convert `GateDecision.accepted_vlm_vetoes` strings into `xvs-odd-vlm` u8 hazard bitmask."""
    mask = 0
    for hazard, bit in VLM_HAZARD_BIT.items():
        if bit != 0 and f"VLM:{hazard.name}" in accepted_vlm_vetoes:
            mask |= bit
    return mask


# ---------------------------------------------------------------------------
# Strict DR-09 License Guard (Permissive Apache-2.0 / MIT Allowlist + Hard Bans)
# ---------------------------------------------------------------------------

ALLOWED_SPDX_LICENSES: frozenset[str] = frozenset({"Apache-2.0", "MIT"})


@dataclass(frozen=True)
class AllowedVlmSpec:
    """Verified permissive (Apache-2.0 / MIT) edge VLM specification per DR-09."""

    canonical_name: str
    spdx_license: str
    max_prompt_tokens: int
    default_visual_tokens: int = 280
    allow_family_suffix: bool = False


PERMITTED_VLM_SPECS: tuple[AllowedVlmSpec, ...] = (
    AllowedVlmSpec("Qwen3-VL-2B-Instruct", "Apache-2.0", 512, 256, allow_family_suffix=False),
    AllowedVlmSpec("Qwen3-VL-4B-Instruct", "Apache-2.0", 2048, 280, allow_family_suffix=False),
    AllowedVlmSpec("Gemma-4-E2B-it", "Apache-2.0", 512, 140, allow_family_suffix=False),
    AllowedVlmSpec("Gemma-4-E4B-it", "Apache-2.0", 1024, 280, allow_family_suffix=False),
    AllowedVlmSpec("Qwen3.5-2B", "Apache-2.0", 1024, 256, allow_family_suffix=True),
    AllowedVlmSpec("InternVL3", "Apache-2.0", 1024, 256, allow_family_suffix=True),
    AllowedVlmSpec("SmolVLM2", "Apache-2.0", 512, 81, allow_family_suffix=True),
)

ALLOWED_QUANT_OR_RUNTIME_SUFFIX_RE = re.compile(
    r"^(?:[-_](?:instruct|it|nvfp4|awq|int4|w4a8|w4a16|fp16|bf16|fp8|gguf|q4[_a-z0-9]*|q8[_a-z0-9]*|qat|trt))*$",
    re.IGNORECASE,
)

ALLOWED_FAMILY_VARIANT_SUFFIX_RE = re.compile(
    r"^(?:[-_](?:256m|500m|1b|2b|2\.2b|4b|8b|video|instruct|it|nvfp4|awq|int4|w4a8|w4a16|fp16|bf16|fp8|gguf|q4[_a-z0-9]*|q8[_a-z0-9]*|qat|trt))*$",
    re.IGNORECASE,
)

_BANNED_MODEL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Gemma-3 is banned by DR-09 (custom Gemma Terms of Use §3.1/§3.2 pass-through & remote restriction)",
        re.compile(r"gemma[-_.\s]?3\b", re.IGNORECASE),
    ),
    (
        "PaliGemma-2 is banned by DR-09 (custom Gemma Terms of Use §3.1/§3.2)",
        re.compile(r"paligemma(?:[-_.\s]?2)?\b", re.IGNORECASE),
    ),
    (
        "Qwen2.5-VL-3B is banned by DR-09 (non-commercial Qwen Research License)",
        re.compile(r"qwen2[._]?5[-_]?vl[-_]?(?:3b|72b)\b", re.IGNORECASE),
    ),
    (
        "Cosmos-Reason2 is banned by DR-09 (custom NVIDIA Open Model License, not Apache-2.0/MIT)",
        re.compile(r"cosmos[-_.\s]?reason(?:2)?\b", re.IGNORECASE),
    ),
    (
        "GPL/AGPL/LGPL copyleft models are banned by DR-09",
        re.compile(r"(?:^|[-_/\s.])(?:a?gpl|lgpl)(?:[-_.\s0-9]|$)", re.IGNORECASE),
    ),
    (
        "Non-commercial (NC) models are banned by DR-09",
        re.compile(r"(?:^|[-_/\s.])(?:nc|non[-_]?commercial|cc[-_]?by[-_]?nc|research[-_]?only)(?:[-_.\s0-9]|$)", re.IGNORECASE),
    ),
    (
        "Thinking checkpoints are banned in the real-time <200 ms DR-09 safety loop (enable_thinking=False)",
        re.compile(r"(?:^|[-_/\s.])thinking(?:[-_.\s]|$)", re.IGNORECASE),
    ),
)

_BANNED_LICENSE_RE = re.compile(
    r"(?:gpl|agpl|lgpl|non[-_]?commercial|cc[-_]?by[-_]?nc|cc[-_]?by[-_]?sa|\bnc\b|gemma|research|custom|proprietary)",
    re.IGNORECASE,
)


def _normalize_spdx(license_spdx: str) -> str:
    raw = license_spdx.strip()
    low = raw.lower()
    if low in ("apache-2.0", "apache 2.0", "apache2", "apache-2"):
        return "Apache-2.0"
    if low == "mit":
        return "MIT"
    return raw


def validate_vlm_model_license(
    model_id: str,
    license_spdx: str | None = None,
) -> AllowedVlmSpec:
    """Enforce DR-09 strict permissive VLM allowlist and hard-reject banned models/licenses.

    Allowlist ONLY permissive Apache-2.0 / MIT models:
      - Qwen3-VL-2B-Instruct
      - Qwen3-VL-4B-Instruct
      - Gemma-4-E2B-it
      - Gemma-4-E4B-it
      - Qwen3.5-2B
      - InternVL3
      - SmolVLM2

    Hard-rejects Gemma-3, PaliGemma-2, Qwen2.5-VL-3B, Cosmos-Reason2, Thinking variants,
    GPL/AGPL/NC licenses, and any unlisted model with `ValueError`.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("DR-09 license guard: model_id must be a non-empty string")
    cleaned = model_id.strip()

    if license_spdx is not None:
        if not isinstance(license_spdx, str) or not license_spdx.strip():
            raise ValueError("DR-09 license guard: license_spdx must be a non-empty string when provided")
        if _BANNED_LICENSE_RE.search(license_spdx):
            raise ValueError(
                f"DR-09 license guard REJECTED banned license {license_spdx!r} for model {cleaned!r}; "
                "only permissive Apache-2.0 or MIT licenses are allowed"
            )
        norm_lic = _normalize_spdx(license_spdx)
        if norm_lic not in ALLOWED_SPDX_LICENSES:
            raise ValueError(
                f"DR-09 license guard REJECTED non-permissive license {license_spdx!r}; "
                f"allowed={sorted(ALLOWED_SPDX_LICENSES)}"
            )

    for reason, pattern in _BANNED_MODEL_RULES:
        if pattern.search(cleaned):
            raise ValueError(f"DR-09 license guard REJECTED {cleaned!r}: {reason}")

    # Strip optional HuggingFace namespace (e.g. "Qwen/Qwen3-VL-2B-Instruct", "google/Gemma-4-E2B-it")
    short_name = cleaned.rsplit("/", maxsplit=1)[-1]
    short_low = short_name.lower()

    for spec in PERMITTED_VLM_SPECS:
        canon_low = spec.canonical_name.lower()
        if short_low == canon_low:
            return spec
        if short_low.startswith(canon_low):
            tail = short_name[len(spec.canonical_name) :]
            suffix_re = (
                ALLOWED_FAMILY_VARIANT_SUFFIX_RE
                if spec.allow_family_suffix
                else ALLOWED_QUANT_OR_RUNTIME_SUFFIX_RE
            )
            if tail and suffix_re.match(tail):
                return spec

    allowed_names = ", ".join(s.canonical_name for s in PERMITTED_VLM_SPECS)
    raise ValueError(
        f"DR-09 license guard REJECTED unlisted model {cleaned!r}. "
        f"Permitted Apache-2.0/MIT allowlist: {allowed_names}"
    )


# ---------------------------------------------------------------------------
# Static <= 8-Token JSON / Trie Grammar Parser (Zero Prose, Zero Absolute Scores)
# ---------------------------------------------------------------------------

class VlmGrammarError(ValueError):
    """Raised when VLM output violates the DR-09 static <= 8-token constrained grammar."""


_FORBIDDEN_SCORE_OR_PROSE_RE = re.compile(
    r"(?:<think>|</think>|```|\bscore\b|\bscores\b|\bflz_score|\bsafety_score|\bconfidence_score|\brating\b)",
    re.IGNORECASE,
)

_ALLOWED_COMPACT_JSON_KEYS: frozenset[str] = frozenset({"h", "a", "z"})
_ALLOWED_VERBOSE_JSON_KEYS: frozenset[str] = frozenset(
    {"hazard_enum", "azimuth_sector", "ranked_zone_ids"}
)


def _require_strict_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VlmGrammarError(
            f"DR-09 grammar REJECTED non-integer {name}={value!r} (floats/booleans/strings forbidden)"
        )
    return int(value)


def _validate_reachable_flz_mask(mask: int) -> int:
    if isinstance(mask, bool) or not isinstance(mask, int) or not (0 <= mask <= 0xFFFF):
        raise ValueError(f"reachable_flz_mask must be uint16 (0..0xFFFF), got {mask!r}")
    return int(mask)


@dataclass(frozen=True)
class ParsedVlmOutput:
    """Validated structured advisory parsed strictly from <= 8 constrained tokens."""

    hazard: VlmHazard
    azimuth_sector: int
    ranked_zone_ids: tuple[int, int, int, int]
    token_count: int
    wire_format: str


class StaticVlmTrieGrammar:
    """Deterministic static Trie / compact JSON parser enforcing DR-09 <= 8-token bounds.

    Enforces:
      - Strictly <= `max_tokens` (<= 8) output tokens.
      - `hazard_enum` strictly in `VlmHazard` (`0..5`).
      - `azimuth_sector` strictly in `0..11`.
      - `ranked_zone_ids` strictly 4x unique `uint8` subset of `reachable_flz_mask`
        (padded with `0xFF = UNUSED_ZONE_ID`).
      - Hard-rejects free-form prose, `<think>` tags, floats, and absolute FLZ scores.
    """

    def __init__(
        self,
        reachable_flz_mask: int,
        *,
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> None:
        self._reachable_flz_mask = _validate_reachable_flz_mask(reachable_flz_mask)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not (2 <= max_tokens <= MAX_OUTPUT_TOKENS):
            raise ValueError(f"max_tokens must be in 2..{MAX_OUTPUT_TOKENS}, got {max_tokens!r}")
        self._max_tokens = int(max_tokens)

    @property
    def reachable_flz_mask(self) -> int:
        return self._reachable_flz_mask

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def allowed_zone_ids(self) -> tuple[int, ...]:
        """Return sorted reachable zone IDs (`0..15`) permitted by `reachable_flz_mask`."""
        return tuple(z for z in range(MAX_FLZ_ID) if (self._reachable_flz_mask & (1 << z)) != 0)

    def format_trie(
        self,
        hazard: VlmHazard | int,
        azimuth_sector: int,
        ranked_zone_ids: Sequence[int] = (),
    ) -> str:
        """Format a valid advisory as a canonical <= 6-token static Trie wire string."""
        haz = VlmHazard(int(hazard))
        az = _require_strict_int("azimuth_sector", azimuth_sector)
        if not (0 <= az < NUM_AZIMUTH_SECTORS):
            raise VlmGrammarError(f"azimuth_sector out of range 0..11: {az}")
        zones = self._validate_and_pad_zones(ranked_zone_ids)
        active = [z for z in zones if z != UNUSED_ZONE_ID]
        parts = [f"H{int(haz)}", f"A{az}"] + [f"Z{z}" for z in active]
        return " ".join(parts)

    def format_compact_json(
        self,
        hazard: VlmHazard | int,
        azimuth_sector: int,
        ranked_zone_ids: Sequence[int] = (),
    ) -> str:
        """Format a valid advisory as a minimal compact JSON object (`h`, `a`, `z`)."""
        haz = VlmHazard(int(hazard))
        az = _require_strict_int("azimuth_sector", azimuth_sector)
        if not (0 <= az < NUM_AZIMUTH_SECTORS):
            raise VlmGrammarError(f"azimuth_sector out of range 0..11: {az}")
        zones = self._validate_and_pad_zones(ranked_zone_ids)
        active = [z for z in zones if z != UNUSED_ZONE_ID]
        return json.dumps({"h": int(haz), "a": az, "z": active}, separators=(",", ":"))

    def _validate_and_pad_zones(self, raw_zones: Sequence[Any]) -> tuple[int, int, int, int]:
        if len(raw_zones) > NUM_RANKED_ZONES:
            raise VlmGrammarError(
                f"ranked_zone_ids exceeds {NUM_RANKED_ZONES} entries: {len(raw_zones)}"
            )
        seen: set[int] = set()
        accepted: list[int] = []
        for idx, z_val in enumerate(raw_zones):
            zid = _require_strict_int(f"ranked_zone_ids[{idx}]", z_val)
            if zid == UNUSED_ZONE_ID:
                continue
            if not (0 <= zid < MAX_FLZ_ID):
                raise VlmGrammarError(
                    f"ranked_zone_id {zid} outside valid uint8 FLZ ID range 0..{MAX_FLZ_ID - 1}"
                )
            if (self._reachable_flz_mask & (1 << zid)) == 0:
                raise VlmGrammarError(
                    f"ranked_zone_id {zid} outside reachable_flz_mask 0x{self._reachable_flz_mask:04X}"
                )
            if zid in seen:
                raise VlmGrammarError(f"duplicate ranked_zone_id {zid} rejected by static grammar")
            seen.add(zid)
            accepted.append(zid)

        while len(accepted) < NUM_RANKED_ZONES:
            accepted.append(UNUSED_ZONE_ID)
        return (accepted[0], accepted[1], accepted[2], accepted[3])

    def parse(self, raw_text: str) -> ParsedVlmOutput:
        """Parse and validate constrained VLM output against the static <= 8-token grammar."""
        if not isinstance(raw_text, str):
            raise VlmGrammarError(f"VLM output must be str, got {type(raw_text).__name__}")
        stripped = raw_text.strip()
        if not stripped:
            raise VlmGrammarError("empty VLM output rejected")
        if "\n" in stripped or "\r" in stripped:
            raise VlmGrammarError("multi-line VLM prose output rejected by <= 8-token grammar")
        if len(stripped) > MAX_RAW_OUTPUT_CHARS:
            raise VlmGrammarError(
                f"VLM output length {len(stripped)} exceeds static cap {MAX_RAW_OUTPUT_CHARS} chars"
            )
        if _FORBIDDEN_SCORE_OR_PROSE_RE.search(stripped):
            raise VlmGrammarError(
                "VLM output contains forbidden thinking/score/prose marker"
            )

        if stripped.startswith("{"):
            return self._parse_compact_json(stripped)
        return self._parse_trie_tokens(stripped)

    def _parse_compact_json(self, stripped: str) -> ParsedVlmOutput:
        if not stripped.endswith("}"):
            raise VlmGrammarError("trailing prose after JSON object rejected")
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise VlmGrammarError(f"malformed JSON in constrained VLM output: {exc}") from exc

        if not isinstance(obj, dict):
            raise VlmGrammarError("constrained JSON root must be a flat object")

        keys = set(obj.keys())
        if keys <= _ALLOWED_COMPACT_JSON_KEYS and {"h", "a"} <= keys:
            raw_h = obj["h"]
            raw_a = obj["a"]
            raw_z = obj["z"] if "z" in obj else []
        elif keys <= _ALLOWED_VERBOSE_JSON_KEYS and {"hazard_enum", "azimuth_sector"} <= keys:
            raw_h = obj["hazard_enum"]
            raw_a = obj["azimuth_sector"]
            raw_z = obj["ranked_zone_ids"] if "ranked_zone_ids" in obj else []
        else:
            raise VlmGrammarError(
                f"disallowed or missing JSON keys {sorted(keys)} (absolute FLZ scores & extra keys forbidden)"
            )

        if not isinstance(raw_z, list):
            raise VlmGrammarError("JSON ranked zone field must be a list of <= 4 integers")

        haz_int = _require_strict_int("hazard_enum", raw_h)
        az_int = _require_strict_int("azimuth_sector", raw_a)
        if haz_int not in VLM_HAZARD_BIT:
            raise VlmGrammarError(f"hazard_enum {haz_int} outside 0..5")
        if not (0 <= az_int < NUM_AZIMUTH_SECTORS):
            raise VlmGrammarError(f"azimuth_sector {az_int} outside 0..11")

        zones = self._validate_and_pad_zones(raw_z)
        token_count = 2 + len(raw_z)
        if token_count > self._max_tokens:
            raise VlmGrammarError(
                f"grammar token count {token_count} exceeds max_tokens={self._max_tokens}"
            )
        return ParsedVlmOutput(
            hazard=VlmHazard(haz_int),
            azimuth_sector=az_int,
            ranked_zone_ids=zones,
            token_count=token_count,
            wire_format="compact_json",
        )

    def _parse_trie_tokens(self, stripped: str) -> ParsedVlmOutput:
        # Reject any floating-point literal (signals absolute FLZ score leakage)
        if "." in stripped:
            raise VlmGrammarError("floating-point scalar rejected (absolute FLZ scores forbidden)")

        raw_tokens = [tok for tok in re.split(r"[\s,]+", stripped) if tok]
        if len(raw_tokens) < 2:
            raise VlmGrammarError(
                f"expected at least 2 Trie tokens (hazard, azimuth), got {len(raw_tokens)}"
            )
        if len(raw_tokens) > self._max_tokens:
            raise VlmGrammarError(
                f"Trie token count {len(raw_tokens)} exceeds max_tokens={self._max_tokens}"
            )

        t0 = raw_tokens[0]
        t1 = raw_tokens[1]
        z_tokens = raw_tokens[2:]

        if len(z_tokens) > NUM_RANKED_ZONES:
            raise VlmGrammarError(
                f"Trie zone token count {len(z_tokens)} exceeds {NUM_RANKED_ZONES}"
            )

        # Tagged Trie form: H<0-5> A<0-11> [Z<0-15|255> ...]
        if t0[0] in ("H", "h"):
            if not re.fullmatch(r"[Hh]\d+", t0):
                raise VlmGrammarError(f"invalid tagged hazard token {t0!r}")
            if not re.fullmatch(r"[Aa]\d+", t1):
                raise VlmGrammarError(f"invalid tagged azimuth token {t1!r}")
            haz_int = int(t0[1:])
            az_int = int(t1[1:])
            parsed_zones: list[int] = []
            for zt in z_tokens:
                if not re.fullmatch(r"[Zz]\d+", zt):
                    raise VlmGrammarError(f"invalid tagged zone token {zt!r}")
                parsed_zones.append(int(zt[1:]))
        else:
            # Bare numeric Trie tuple: <0-5> <0-11> [<0-15|255> ...]
            if not t0.isdigit() or not t1.isdigit():
                raise VlmGrammarError(
                    f"free-form prose or invalid Trie token rejected: {raw_tokens!r}"
                )
            haz_int = int(t0)
            az_int = int(t1)
            parsed_zones = []
            for zt in z_tokens:
                if not zt.isdigit():
                    raise VlmGrammarError(f"non-integer Trie zone token rejected: {zt!r}")
                parsed_zones.append(int(zt))

        if haz_int not in VLM_HAZARD_BIT:
            raise VlmGrammarError(f"hazard_enum {haz_int} outside 0..5")
        if not (0 <= az_int < NUM_AZIMUTH_SECTORS):
            raise VlmGrammarError(f"azimuth_sector {az_int} outside 0..11")

        zones = self._validate_and_pad_zones(parsed_zones)
        return ParsedVlmOutput(
            hazard=VlmHazard(haz_int),
            azimuth_sector=az_int,
            ranked_zone_ids=zones,
            token_count=len(raw_tokens),
            wire_format="trie",
        )


def parse_vlm_constrained_output(
    raw_text: str,
    reachable_flz_mask: int,
    *,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> ParsedVlmOutput:
    """Convenience entry point for DR-09 static <= 8-token grammar parsing."""
    return StaticVlmTrieGrammar(reachable_flz_mask, max_tokens=max_tokens).parse(raw_text)


# ---------------------------------------------------------------------------
# Single-Slot Drop-Stale Freshness Guard (Zero Frame Queueing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferedVlmFrame:
    """Single latest frame awaiting asynchronous VLM evaluation."""

    bgr_frame: np.ndarray
    capture_ts_ms: int
    reachable_flz_mask: int
    scenario_hint: str = ""
    lwir_bgr_frame: np.ndarray | None = None


class SingleSlotDropStaleBuffer:
    """DR-09 zero-queue single-slot drop-stale ring buffer (capacity == 1, <= 200 ms)."""

    CAPACITY: int = 1

    def __init__(self, max_age_ms: int = WATCHDOG_MAX_AGE_MS) -> None:
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or not (0 <= max_age_ms <= WATCHDOG_MAX_AGE_MS):
            raise ValueError(
                f"max_age_ms must be in 0..{WATCHDOG_MAX_AGE_MS}, got {max_age_ms!r}"
            )
        self._max_age_ms = int(max_age_ms)
        self._slot: BufferedVlmFrame | None = None
        self._dropped_stale_count: int = 0
        self._expired_stale_count: int = 0

    @property
    def max_age_ms(self) -> int:
        return self._max_age_ms

    @property
    def dropped_stale_count(self) -> int:
        return self._dropped_stale_count

    @property
    def expired_stale_count(self) -> int:
        return self._expired_stale_count

    @property
    def has_pending(self) -> bool:
        return self._slot is not None

    def submit(self, frame: BufferedVlmFrame) -> bool:
        """Store `frame` in the single slot, evicting any older unconsumed frame.

        Returns True iff an older unconsumed frame was dropped.
        """
        evicted = self._slot is not None
        if evicted:
            self._dropped_stale_count += 1
        self._slot = frame
        return evicted

    def take_fresh(self, now_ms: int) -> BufferedVlmFrame | None:
        """Consume the single slot iff `0 <= now_ms - capture_ts_ms <= max_age_ms`."""
        slot = self._slot
        self._slot = None
        if slot is None:
            return None
        age_ms = int(now_ms) - int(slot.capture_ts_ms)
        if age_ms < 0 or age_ms > self._max_age_ms:
            self._expired_stale_count += 1
            return None
        return slot


# ---------------------------------------------------------------------------
# Frozen Split-Conformal Abstention + BCEA Contrastive Grounding Calibrator
# ---------------------------------------------------------------------------

def default_blank_image_priors() -> dict[VlmHazard, float]:
    """Offline-precomputed blank-image prompt priors P(y | I_blank) for BCEA grounding."""
    return {
        VlmHazard.NONE: 0.50,
        VlmHazard.RUNWAY_INCURSION: 0.12,
        VlmHazard.BIRD_FLOCK: 0.10,
        VlmHazard.CONVECTIVE_VIRGA: 0.11,
        VlmHazard.CRANE_PROXY: 0.10,
        VlmHazard.FLZ_HAZARD: 0.12,
    }


@dataclass(frozen=True)
class FrozenConformalCalibration:
    """Immutable offline split-conformal + BCEA + EASA RA-08 calibration parameters.

    Strictly frozen in flight (zero online weight updates or adaptive quantile drift)
    per EASA Concept Paper Proposed Issue 03 and FAA AI Safety Assurance Roadmap.
    """

    alpha: float = 0.05
    q_hat: float = 0.25
    bcea_min_delta: float = 0.15
    aleatoric_entropy_threshold: float = 0.45
    epistemic_ood_threshold: float = 0.35
    blank_image_priors: Mapping[VlmHazard, float] = field(
        default_factory=default_blank_image_priors
    )
    weights_frozen: bool = True
    online_weight_updates_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.weights_frozen:
            raise ValueError("DR-09 regulatory lock: conformal & VLM weights MUST remain frozen in flight")
        if self.online_weight_updates_allowed:
            raise ValueError("DR-09 regulatory lock: online weight updates are strictly forbidden in flight")
        if not (0.0 < self.alpha < 1.0) or not math.isfinite(self.alpha):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if not (0.0 <= self.q_hat <= 1.0) or not math.isfinite(self.q_hat):
            raise ValueError(f"q_hat must be in [0, 1], got {self.q_hat!r}")
        if not (0.0 <= self.bcea_min_delta <= 1.0) or not math.isfinite(self.bcea_min_delta):
            raise ValueError(f"bcea_min_delta must be in [0, 1], got {self.bcea_min_delta!r}")
        for haz in VlmHazard:
            if haz not in self.blank_image_priors:
                raise ValueError(f"missing offline blank_image_prior for {haz.name}")
            prior = float(self.blank_image_priors[haz])
            if not (0.0 <= prior <= 1.0) or not math.isfinite(prior):
                raise ValueError(f"invalid blank_image_prior for {haz.name}: {prior!r}")


@dataclass(frozen=True)
class ConformalCalibrationVerdict:
    """Split-conformal + BCEA + EASA RA-08 aleatoric/epistemic calibration output."""

    conformal_valid_bit: int
    aleatoric_flag: int
    epistemic_flag: int
    nonconformity_score: float
    bcea_visual_lift: float
    abstain_reason: str
    weights_frozen: bool = True


class FrozenConformalBceaCalibrator:
    """Deterministic frozen-weight split-conformal & BCEA grounding gate wrapper."""

    def __init__(self, calibration: FrozenConformalCalibration | None = None) -> None:
        self._cal = FrozenConformalCalibration() if calibration is None else calibration

    @property
    def calibration(self) -> FrozenConformalCalibration:
        return self._cal

    def calibrate(
        self,
        *,
        hazard: VlmHazard,
        p_image: float,
        token_entropy: float,
        ood_distance: float,
        optical_odd_degraded: bool = False,
    ) -> ConformalCalibrationVerdict:
        """Compute `conformal_valid_bit`, `aleatoric_flag`, and `epistemic_flag` fail-closed."""
        for val in (p_image, token_entropy, ood_distance):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                return ConformalCalibrationVerdict(
                    conformal_valid_bit=0,
                    aleatoric_flag=1,
                    epistemic_flag=1,
                    nonconformity_score=1.0,
                    bcea_visual_lift=0.0,
                    abstain_reason="non-finite calibration score",
                    weights_frozen=True,
                )

        p_clamped = min(max(float(p_image), 0.0), 1.0)
        ent_clamped = min(max(float(token_entropy), 0.0), 1.0)
        ood_clamped = min(max(float(ood_distance), 0.0), 1.0)

        nonconformity = 1.0 - p_clamped
        p_blank = float(self._cal.blank_image_priors[hazard])
        bcea_lift = p_clamped - p_blank

        aleatoric_flag = (
            1
            if (ent_clamped >= self._cal.aleatoric_entropy_threshold or optical_odd_degraded)
            else 0
        )
        bcea_under_grounded = (hazard != VlmHazard.NONE) and (
            bcea_lift < self._cal.bcea_min_delta
        )
        epistemic_flag = (
            1
            if (ood_clamped >= self._cal.epistemic_ood_threshold or bcea_under_grounded)
            else 0
        )

        reasons: list[str] = []
        if nonconformity > self._cal.q_hat:
            reasons.append(f"nonconformity={nonconformity:.3f}>q_hat={self._cal.q_hat:.3f}")
        if bcea_under_grounded:
            reasons.append(f"bcea_lift={bcea_lift:.3f}<min={self._cal.bcea_min_delta:.3f}")
        if aleatoric_flag:
            reasons.append("ra08_aleatoric")
        if epistemic_flag:
            reasons.append("ra08_epistemic")

        valid_bit = 1 if not reasons else 0
        return ConformalCalibrationVerdict(
            conformal_valid_bit=valid_bit,
            aleatoric_flag=aleatoric_flag,
            epistemic_flag=epistemic_flag,
            nonconformity_score=round(nonconformity, 6),
            bcea_visual_lift=round(bcea_lift, 6),
            abstain_reason="ok" if valid_bit == 1 else "; ".join(reasons),
            weights_frozen=True,
        )


# ---------------------------------------------------------------------------
# Backends Behind One Interface (Local HTTP VLM vs Honest Offline Calibration Stub)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VlmInferenceRequest:
    """Single-frame request handed to a VLM backend."""

    bgr_frame: np.ndarray
    capture_ts_ms: int
    reachable_flz_mask: int
    scenario_hint: str = ""
    roi_bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class VlmBackendResponse:
    """Structured raw backend response prior to grammar parsing and conformal gating."""

    raw_text: str
    p_image: float
    token_entropy: float
    ood_distance: float
    latency_ms: float
    backend_label: str
    model_id: str
    spdx_license: str
    is_neural_vlm: bool
    enable_thinking: bool
    http_status: int | None = None


class VlmBackendProtocol(Protocol):
    """Unified interface shared by live local HTTP VLM servers and the offline stub."""

    @property
    def backend_label(self) -> str: ...

    @property
    def is_neural_vlm(self) -> bool: ...

    @property
    def model_id(self) -> str: ...

    @property
    def model_spec(self) -> AllowedVlmSpec: ...

    @property
    def enable_thinking(self) -> bool: ...

    @property
    def max_output_tokens(self) -> int: ...

    def infer(self, req: VlmInferenceRequest) -> VlmBackendResponse | None: ...


def _encode_bgr_to_compact_jpeg_b64(bgr_frame: np.ndarray, max_side: int = 512) -> str:
    """Resize BGR/gray frame to <= `max_side` and return base64-encoded JPEG string."""
    if bgr_frame.ndim not in (2, 3) or bgr_frame.size == 0:
        raise ValueError(f"bgr_frame must be non-empty 2D or 3D ndarray, got shape {bgr_frame.shape}")
    h, w = bgr_frame.shape[:2]
    img = bgr_frame
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed for VLM request frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _extract_http_confidence(choice: Mapping[str, Any], body: Mapping[str, Any]) -> tuple[float, float, float]:
    """Extract calibrated `(p_image, token_entropy, ood_distance)` from HTTP response metadata."""
    # Optional explicit calibration block from TensorRT Edge-LLM / custom server
    if "xvs_calibration" in body and isinstance(body["xvs_calibration"], Mapping):
        cal = body["xvs_calibration"]
        if "p_image" in cal and "token_entropy" in cal and "ood_distance" in cal:
            return (
                float(cal["p_image"]),
                float(cal["token_entropy"]),
                float(cal["ood_distance"]),
            )

    # Standard OpenAI logprobs extraction
    if "logprobs" in choice and isinstance(choice["logprobs"], Mapping):
        lp = choice["logprobs"]
        if "content" in lp and isinstance(lp["content"], list) and lp["content"]:
            logprobs: list[float] = []
            for item in lp["content"]:
                if isinstance(item, Mapping) and "logprob" in item:
                    val = float(item["logprob"])
                    if math.isfinite(val):
                        logprobs.append(val)
            if logprobs:
                mean_lp = sum(logprobs) / float(len(logprobs))
                p_img = min(max(math.exp(min(mean_lp, 0.0)), 0.0), 1.0)
                entropy = min(max(-mean_lp / 3.0, 0.0), 1.0)
                ood = min(max(1.0 - p_img, 0.0), 1.0)
                return p_img, entropy, ood

    # Fail-conservative when server omits logprobs: trigger conformal abstention rather than over-trusting
    return 0.50, 0.50, 0.50


class OpenAICompatibleVlmBackend:
    """Live local HTTP VLM adapter for `llama-server` / `vLLM` / `TensorRT Edge-LLM`.

    Strictly enforces:
      - DR-09 Apache-2.0 / MIT model allowlist (rejects Gemma-3, PaliGemma-2, Qwen2.5-VL-3B, etc.).
      - `enable_thinking=False` in config and every outbound HTTP JSON payload.
      - `max_tokens <= 8`, `temperature = 0.0`, strict `< 200 ms` timeout fail-closed (`None`).
      - Loopback endpoint (`127.0.0.1` / `localhost` / `::1`) by default (no accidental cloud egress).
    """

    def __init__(
        self,
        *,
        model_id: str = "Qwen3-VL-2B-Instruct",
        base_url: str = "http://127.0.0.1:8080/v1",
        timeout_ms: int = 180,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        enable_thinking: bool = False,
        license_spdx: str | None = None,
        api_key: str = "local-openxvs",
        max_image_side: int = 512,
        allow_non_loopback: bool = False,
    ) -> None:
        self._spec = validate_vlm_model_license(model_id, license_spdx)
        if enable_thinking is not False:
            raise ValueError("DR-09 runtime lock: enable_thinking MUST be False")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or not (1 <= max_output_tokens <= MAX_OUTPUT_TOKENS):
            raise ValueError(
                f"DR-09 runtime lock: max_output_tokens must be in 1..{MAX_OUTPUT_TOKENS}, got {max_output_tokens!r}"
            )
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not (1 <= timeout_ms <= MAX_HTTP_TIMEOUT_MS):
            raise ValueError(
                f"DR-09 runtime lock: timeout_ms must be strictly < 200 ms (1..{MAX_HTTP_TIMEOUT_MS}), got {timeout_ms!r}"
            )

        parsed_url = urllib.parse.urlparse(base_url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"base_url must use http/https, got {base_url!r}")
        host = (parsed_url.hostname or "").lower()
        if not allow_non_loopback and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"OpenAICompatibleVlmBackend defaults to local loopback only (127.0.0.1/localhost), got host={host!r}"
            )

        self._model_id = model_id.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_ms = int(timeout_ms)
        self._max_output_tokens = int(max_output_tokens)
        self._enable_thinking = False
        self._api_key = api_key
        self._max_image_side = int(max_image_side)

    @property
    def backend_label(self) -> str:
        return HTTP_BACKEND_LABEL

    @property
    def is_neural_vlm(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_spec(self) -> AllowedVlmSpec:
        return self._spec

    @property
    def enable_thinking(self) -> bool:
        return self._enable_thinking

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    def build_request_payload(self, req: VlmInferenceRequest) -> dict[str, Any]:
        """Build deterministic OpenAI `/chat/completions` JSON payload with DR-09 locks."""
        mask = _validate_reachable_flz_mask(req.reachable_flz_mask)
        allowed_zones = [z for z in range(MAX_FLZ_ID) if (mask & (1 << z)) != 0]
        b64_jpeg = _encode_bgr_to_compact_jpeg_b64(req.bgr_frame, max_side=self._max_image_side)
        mime_prefix = "data:" + "image/jpeg" + ";base64,"
        image_uri = mime_prefix + b64_jpeg

        system_prompt = (
            "Civil aviation semantic hazard monitor (IDAL-D advisory only, zero actuator authority). "
            "Emit ONLY <= 8 constrained tokens in format 'H<0-5> A<0-11> Z<id>...' or compact JSON "
            '{"h":0..5,"a":0..11,"z":[...]}. '
            "Hazards: 0=NONE, 1=RUNWAY_INCURSION, 2=BIRD_FLOCK, 3=CONVECTIVE_VIRGA, 4=CRANE_PROXY, 5=FLZ_HAZARD. "
            "Relative FLZ ranking ONLY from reachable_flz_ids; NEVER emit prose, thoughts, or absolute scores."
        )
        user_prompt = (
            f"phase={req.scenario_hint or 'nominal'} "
            f"reachable_flz_ids={allowed_zones} "
            "Respond strictly in <= 8 tokens."
        )
        return {
            "model": self._model_id,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.0,
            "max_tokens": self._max_output_tokens,
            "logprobs": True,
            "top_logprobs": 4,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                },
            ],
        }

    def infer(self, req: VlmInferenceRequest) -> VlmBackendResponse | None:
        """Execute bounded `< 200 ms` local HTTP inference; return `None` fail-closed on any fault."""
        t0 = time.perf_counter()
        payload = self.build_request_payload(req)
        raw_body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self._base_url}/chat/completions"
        http_req = urllib.request.Request(
            endpoint,
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        timeout_s = float(self._timeout_ms) / 1000.0
        try:
            with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
                status_code = int(resp.status)
                resp_bytes = resp.read(65_536)
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if status_code != 200 or elapsed_ms >= float(WATCHDOG_MAX_AGE_MS):
            return None

        try:
            decoded = json.loads(resp_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(decoded, Mapping):
            return None
        choices = decoded["choices"] if "choices" in decoded else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return None
        choice0 = choices[0]
        message = choice0["message"] if "message" in choice0 else None
        if not isinstance(message, Mapping) or "content" not in message:
            return None
        content = message["content"]
        if not isinstance(content, str):
            return None

        p_image, token_entropy, ood_distance = _extract_http_confidence(choice0, decoded)
        return VlmBackendResponse(
            raw_text=content.strip(),
            p_image=p_image,
            token_entropy=token_entropy,
            ood_distance=ood_distance,
            latency_ms=round(elapsed_ms, 3),
            backend_label=HTTP_BACKEND_LABEL,
            model_id=self._model_id,
            spdx_license=self._spec.spdx_license,
            is_neural_vlm=True,
            enable_thinking=False,
            http_status=status_code,
        )


@dataclass(frozen=True)
class OfflineStubProposal:
    """Explicit deterministic proposal for `DeterministicOfflineCalibrationBackend`."""

    raw_text: str | None = None
    hazard: VlmHazard = VlmHazard.NONE
    azimuth_sector: int = 0
    ranked_zone_ids: tuple[int, ...] = ()
    p_image: float = 0.92
    token_entropy: float = 0.12
    ood_distance: float = 0.08
    simulate_offline: bool = False
    wire_format: str = "trie"


class DeterministicOfflineCalibrationBackend:
    """Deterministic calibration & fault-injection stub (`OFFLINE_CONFORMAL_BENCH_STUB`).

    NEVER pretends to be a neural VLM (`is_neural_vlm == False`,
    `backend_label == "OFFLINE_CONFORMAL_BENCH_STUB"`). Used strictly for deterministic
    CI, timing, conformal calibration, and fault-injection verification when no live
    local VLM HTTP server is attached.
    """

    BACKEND_LABEL: str = OFFLINE_STUB_BACKEND_LABEL
    IS_NEURAL_VLM: bool = False
    PROVENANCE_NOTICE: str = OFFLINE_STUB_PROVENANCE_NOTICE

    def __init__(
        self,
        *,
        model_id: str = "Qwen3-VL-2B-Instruct",
        license_spdx: str | None = None,
        enable_thinking: bool = False,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        scripted_proposals: Sequence[OfflineStubProposal] | None = None,
    ) -> None:
        self._spec = validate_vlm_model_license(model_id, license_spdx)
        if enable_thinking is not False:
            raise ValueError("DR-09 runtime lock: enable_thinking MUST be False")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or not (1 <= max_output_tokens <= MAX_OUTPUT_TOKENS):
            raise ValueError(
                f"DR-09 runtime lock: max_output_tokens must be in 1..{MAX_OUTPUT_TOKENS}, got {max_output_tokens!r}"
            )
        self._model_id = model_id.strip()
        self._enable_thinking = False
        self._max_output_tokens = int(max_output_tokens)
        self._scripted: list[OfflineStubProposal] = (
            list(scripted_proposals) if scripted_proposals is not None else []
        )
        self._cursor: int = 0

    @property
    def backend_label(self) -> str:
        return OFFLINE_STUB_BACKEND_LABEL

    @property
    def is_neural_vlm(self) -> bool:
        return False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_spec(self) -> AllowedVlmSpec:
        return self._spec

    @property
    def enable_thinking(self) -> bool:
        return self._enable_thinking

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    def enqueue(self, proposal: OfflineStubProposal) -> None:
        """Append a scripted proposal for deterministic test/fault-injection sequences."""
        self._scripted.append(proposal)

    def _derive_deterministic_pixel_proposal(self, req: VlmInferenceRequest) -> OfflineStubProposal:
        """Compute deterministic, image-conditioned calibration output (explicitly non-neural)."""
        gray = (
            cv2.cvtColor(req.bgr_frame, cv2.COLOR_BGR2GRAY)
            if req.bgr_frame.ndim == 3
            else req.bgr_frame
        )
        h, w = gray.shape[:2]
        # Deterministic sector from column centroid of highest horizontal gradient energy
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        col_energy = np.mean(gx, axis=0) if gx.size > 0 else np.zeros((max(w - 1, 1),), dtype=np.float32)
        peak_col = int(np.argmax(col_energy)) if col_energy.size > 0 else 0
        azimuth_sector = min(
            NUM_AZIMUTH_SECTORS - 1,
            int((peak_col * NUM_AZIMUTH_SECTORS) // max(w, 1)),
        )
        contrast = float(np.std(gray.astype(np.float32))) / 255.0

        allowed = [z for z in range(MAX_FLZ_ID) if (req.reachable_flz_mask & (1 << z)) != 0]
        # Deterministic relative permutation driven by image mean parity (no absolute scores!)
        if len(allowed) >= 2 and (int(np.mean(gray)) % 2 == 1):
            ranked = tuple(reversed(allowed[:NUM_RANKED_ZONES]))
        else:
            ranked = tuple(allowed[:NUM_RANKED_ZONES])

        hint = req.scenario_hint.lower()
        if "runway" in hint:
            hazard = VlmHazard.RUNWAY_INCURSION
        elif "sky" in hint or "bird" in hint:
            hazard = VlmHazard.BIRD_FLOCK
        elif "fog" in hint or "thermal" in hint or "dve" in hint:
            hazard = VlmHazard.CRANE_PROXY
        elif "flz" in hint:
            hazard = VlmHazard.FLZ_HAZARD
        else:
            hazard = VlmHazard.NONE

        p_image = 0.91 if contrast >= 0.12 else 0.62
        entropy = 0.14 if contrast >= 0.12 else 0.55
        ood_dist = 0.10 if contrast >= 0.12 else 0.42
        return OfflineStubProposal(
            hazard=hazard,
            azimuth_sector=azimuth_sector,
            ranked_zone_ids=ranked,
            p_image=p_image,
            token_entropy=entropy,
            ood_distance=ood_dist,
        )

    def infer(self, req: VlmInferenceRequest) -> VlmBackendResponse | None:
        t0 = time.perf_counter()
        if self._cursor < len(self._scripted):
            prop = self._scripted[self._cursor]
            self._cursor += 1
        else:
            prop = self._derive_deterministic_pixel_proposal(req)

        if prop.simulate_offline:
            return None

        if prop.raw_text is not None:
            raw_text = prop.raw_text
        else:
            grammar = StaticVlmTrieGrammar(req.reachable_flz_mask, max_tokens=self._max_output_tokens)
            if prop.wire_format == "compact_json":
                raw_text = grammar.format_compact_json(
                    prop.hazard, prop.azimuth_sector, prop.ranked_zone_ids
                )
            else:
                raw_text = grammar.format_trie(
                    prop.hazard, prop.azimuth_sector, prop.ranked_zone_ids
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return VlmBackendResponse(
            raw_text=raw_text,
            p_image=prop.p_image,
            token_entropy=prop.token_entropy,
            ood_distance=prop.ood_distance,
            latency_ms=round(elapsed_ms, 3),
            backend_label=OFFLINE_STUB_BACKEND_LABEL,
            model_id=self._model_id,
            spdx_license=self._spec.spdx_license,
            is_neural_vlm=False,
            enable_thinking=False,
            http_status=None,
        )


# ---------------------------------------------------------------------------
# Live `xvs_optical_odd` Bridge Over Real BGR / Paired LWIR Frames
# ---------------------------------------------------------------------------

def bgr_to_eo_u16(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert 8-bit BGR or monochrome frame into 10-bit monochrome EO `uint16` (`0..1023`)."""
    if bgr_frame.ndim == 3:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    elif bgr_frame.ndim == 2:
        gray = bgr_frame
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {bgr_frame.shape}")
    return np.round(gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)


def bgr_to_lwir_u16(
    lwir_bgr_frame: np.ndarray,
    *,
    ref_median_dn: float = 8192.0,
    gain_dn_per_luma: float = 12.0,
) -> np.ndarray:
    """Convert 8-bit LWIR frame into 14-bit Pre-AGC-centered Boson+ `uint16` (`0..16383`)."""
    if lwir_bgr_frame.ndim == 3:
        gray = cv2.cvtColor(lwir_bgr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif lwir_bgr_frame.ndim == 2:
        gray = lwir_bgr_frame.astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D LWIR image array, got shape {lwir_bgr_frame.shape}")
    centered = float(ref_median_dn) + (gray - float(np.median(gray))) * float(gain_dn_per_luma)
    return np.clip(centered, 0.0, 16383.0).astype(np.uint16)


def make_nominal_reference_lwir_u16(height: int = 128, width: int = 160) -> np.ndarray:
    """Healthy 14-bit Boson+ reference baseline for EO-only channels without a paired LWIR stream."""
    yy, xx = np.indices((height, width), dtype=np.float32)
    scene = 8192.0 + 180.0 * np.sin(yy * 0.08) + 140.0 * np.cos(xx * 0.07)
    return np.clip(scene, 0.0, 16383.0).astype(np.uint16)


def studio_media_optical_odd_config(
    *,
    horizon_visible: bool = True,
    fast_mltl: bool = True,
) -> OpticalOddConfig:
    """Calibrated `OpticalOddConfig` for 8-bit->10-bit compressed `data/studio_media/` frames.

    Preserves canonical WMO 1609 m fog MOR threshold, 14-bit Boson+ bad-pixel/FPN thresholds,
    SPA solar glare geometry, and the hard AF447 zero-airdata lock.
    """
    from xvs.v2.xvs_optical_odd.monitor import (
        MltlPersistenceSpec,
        OpticalOddConfig,
        OpticalOddMode,
    )

    specs = (
        {m: MltlPersistenceSpec(confirm_m=2, window_n=3, debounce_ticks=1) for m in OpticalOddMode}
        if fast_mltl
        else None
    )
    kwargs: dict[str, Any] = {
        "raindrop_tenengrad_ref": 8.0,
        "raindrop_mlap_ref": 6.0,
        "raindrop_tile_persist_frames": 2,
        "fog_horizon_snr_db_thresh": 3.0 if horizon_visible else -25.0,
    }
    if specs is not None:
        kwargs["mltl_specs"] = specs
    return OpticalOddConfig(**kwargs)


def studio_media_vlm_gate_config(*, max_age_ms: int = WATCHDOG_MAX_AGE_MS) -> VlmGateConfig:
    """Calibrated `VlmGateConfig` for compressed real `data/studio_media/` BGR frames."""
    return VlmGateConfig(
        max_age_ms=max_age_ms,
        optical_odd=OpticalOddThresholds(
            max_glare_sat_frac=0.15,
            max_droplet_occlusion_frac=0.12,
            min_fog_contrast=0.35,
            min_blur_laplacian=4.5,
        ),
    )


class OpticalOddLiveBridge:
    """Direct bridge from real BGR (+ optional LWIR) frames to DR-13 `DeterministicOpticalOddMonitor`."""

    def __init__(self, config: OpticalOddConfig | None = None) -> None:
        from xvs.v2.xvs_optical_odd.monitor import DeterministicOpticalOddMonitor

        self._monitor = DeterministicOpticalOddMonitor(config)
        self._ref_lwir = make_nominal_reference_lwir_u16()

    @property
    def monitor(self) -> DeterministicOpticalOddMonitor:
        return self._monitor

    def reset(self) -> None:
        self._monitor.reset()

    def evaluate_bgr(
        self,
        eo_bgr: np.ndarray,
        *,
        lwir_bgr: np.ndarray | None = None,
        rvr_km: float | None = None,
        beta_ext_km_inv: float | None = None,
        sun_boresight_deg: float = 95.0,
        eo_hfov_deg: float = 60.0,
        t_win_c: float = 20.0,
        t_amb_c: float = 20.0,
        rh_pct: float = 45.0,
        boson_dt_ffc_s: float = 45.0,
        boson_t_cam_c: float = 25.0,
        boson_t_last_ffc_c: float = 24.8,
        boson_ffc_freeze: bool = False,
        horizon_row_frac: float = 0.50,
        attitude_in_odd: bool = True,
    ) -> tuple[OpticalOddDecision, OpticalOddReading]:
        """Compute live 5-mode zero-ML Optical-ODD verdict and Stage-3 `OpticalOddReading`."""
        from xvs.v2.xvs_optical_odd.monitor import WMO_MOR_LN20, OpticalOddFrameInput

        eo_u16 = bgr_to_eo_u16(eo_bgr)
        lwir_u16 = bgr_to_lwir_u16(lwir_bgr) if lwir_bgr is not None else self._ref_lwir

        eff_beta = beta_ext_km_inv
        if eff_beta is None and rvr_km is not None:
            eff_beta = WMO_MOR_LN20 / max(float(rvr_km), 0.05)

        frame_input = OpticalOddFrameInput(
            eo_frame_u16=eo_u16,
            lwir_frame_u16=lwir_u16,
            sun_boresight_deg=sun_boresight_deg,
            eo_hfov_deg=eo_hfov_deg,
            t_win_c=t_win_c,
            t_amb_c=t_amb_c,
            rh_pct=rh_pct,
            boson_dt_ffc_s=boson_dt_ffc_s,
            boson_t_cam_c=boson_t_cam_c,
            boson_t_last_ffc_c=boson_t_last_ffc_c,
            boson_ffc_freeze=boson_ffc_freeze,
            horizon_row_frac=horizon_row_frac,
            attitude_in_odd=attitude_in_odd,
            beta_ext_km_inv=eff_beta,
        )
        decision = self._monitor.evaluate(frame_input)
        assert decision.airdata_export_blocked is True
        return decision, decision.to_vlm_optical_odd_health()


# ---------------------------------------------------------------------------
# MAVLink 2.0 `#50003 POD_VLM_SEMANTIC_VETO` Frame Builder
# ---------------------------------------------------------------------------

def _pad_flz_ids_4(ranked_flz_ids: Sequence[int]) -> tuple[int, int, int, int]:
    out = [int(z) & 0xFF for z in ranked_flz_ids[:NUM_RANKED_ZONES]]
    while len(out) < NUM_RANKED_ZONES:
        out.append(UNUSED_ZONE_ID)
    return (out[0], out[1], out[2], out[3])


def build_pod_vlm_semantic_veto_frame(
    gate_decision: GateDecision,
    *,
    seq: int,
    capture_ts_ms: int,
    packet: VlmAdvisoryPacket | None = None,
    mav_seq: int = 0,
    sysid: int = 1,
    compid: int = 191,
) -> tuple[PodVlmSemanticVeto, bytes]:
    """Build and encode a verified MAVLink 2.0 `#50003 POD_VLM_SEMANTIC_VETO` binary frame."""
    verdict_code = GATE_STATUS_TO_MAVLINK_VERDICT[gate_decision.status]
    if packet is not None and gate_decision.status not in (
        GateStatus.VLM_OFFLINE,
        GateStatus.SCHEMA_OR_CRC_FAIL,
    ):
        hazard_enum = int(packet.hazard_enum)
        azimuth_sector = int(packet.azimuth_sector)
        conformal_valid = int(packet.conformal_valid_bit)
    else:
        hazard_enum = int(VlmHazard.NONE)
        azimuth_sector = UNKNOWN_AZIMUTH_SECTOR
        conformal_valid = 0

    msg = PodVlmSemanticVeto(
        seq=int(seq) & 0xFFFFFFFF,
        capture_ts_ms=int(capture_ts_ms) & 0xFFFFFFFF,
        hazard_enum=hazard_enum,
        azimuth_sector=azimuth_sector,
        conformal_valid=conformal_valid,
        gate_verdict=verdict_code,
        ranked_flz_ids=_pad_flz_ids_4(gate_decision.ranked_flz_ids),
    )
    frame_bytes = encode_frame(msg, seq=mav_seq & 0xFF, sysid=sysid, compid=compid)
    return msg, frame_bytes


# ---------------------------------------------------------------------------
# End-to-End DR-09 Edge VLM Semantic Safety Monitor Runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VlmRunnerConfig:
    """Top-level DR-09 runner lock configuration."""

    max_age_ms: int = WATCHDOG_MAX_AGE_MS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    enable_thinking: bool = False
    actuator_authority: bool = False
    allow_absolute_flz_scores: bool = False
    mav_sysid: int = 1
    mav_compid: int = 191

    def __post_init__(self) -> None:
        if self.enable_thinking is not False:
            raise ValueError("DR-09 runtime lock: enable_thinking MUST be False")
        if self.actuator_authority is not False:
            raise ValueError("ASTM F3269-21 §1.2 lock: VLM actuator_authority MUST be False")
        if self.allow_absolute_flz_scores is not False:
            raise ValueError("DR-09 lock: absolute scalar FLZ scores are strictly forbidden")
        if not (0 <= self.max_age_ms <= WATCHDOG_MAX_AGE_MS):
            raise ValueError(f"max_age_ms must be in 0..{WATCHDOG_MAX_AGE_MS}, got {self.max_age_ms}")
        if not (1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS):
            raise ValueError(
                f"max_output_tokens must be in 1..{MAX_OUTPUT_TOKENS}, got {self.max_output_tokens}"
            )


@dataclass(frozen=True)
class VlmMonitorStepResult:
    """Complete auditable output of one DR-09 VLM safety monitor evaluation cycle."""

    seq: int
    capture_ts_ms: int
    evaluated_now_ms: int
    backend_label: str
    is_neural_vlm: bool
    model_id: str
    spdx_license: str
    enable_thinking: bool
    optical_odd_decision: OpticalOddDecision
    optical_odd_reading: OpticalOddReading
    parsed_output: ParsedVlmOutput | None
    conformal_verdict: ConformalCalibrationVerdict | None
    advisory_packet: VlmAdvisoryPacket | None
    wire_packet_bytes: bytes | None
    gate_decision: GateDecision
    mavlink_message: PodVlmSemanticVeto
    mavlink_frame_bytes: bytes
    cert_hazards_mask: int
    active_vlm_mask: int
    final_hazards_mask: int
    actuator_authority: bool
    pipeline_latency_ms: float
    grammar_error: str | None = None


class EdgeVlmSafetyMonitorRunner:
    """DR-09 End-to-End Edge VLM Semantic Safety Monitor Pipeline.

    Wires:
      1. License-validated VLM backend (`OpenAICompatibleVlmBackend` or `DeterministicOfflineCalibrationBackend`),
      2. Single-slot drop-stale freshness buffer (`<= 200 ms`) + monotonic sequence counter,
      3. Static `<= 8`-token JSON/Trie grammar parser (rejects prose & absolute FLZ scores),
      4. Frozen split-conformal + BCEA contrastive grounding calibrator (EASA RA-08),
      5. Exact 32-byte IEEE-CRC32 `VlmAdvisoryPacket` packing (1:1 `xvs-odd-vlm` wire parity),
      6. Live zero-ML 5-mode `xvs_optical_odd` Stage-3 health bridge (`airdata_export_blocked=True`),
      7. 6-Stage `AsymmetricSimplexVlmGate` (`actuator_authority=False`, pessimistic union),
      8. Verified MAVLink 2.0 `#50003 POD_VLM_SEMANTIC_VETO` frame emission.
    """

    def __init__(
        self,
        *,
        backend: VlmBackendProtocol,
        runner_config: VlmRunnerConfig | None = None,
        gate_config: VlmGateConfig | None = None,
        optical_odd_config: OpticalOddConfig | None = None,
        conformal_calibration: FrozenConformalCalibration | None = None,
    ) -> None:
        self._runner_cfg = VlmRunnerConfig() if runner_config is None else runner_config
        # Re-verify license & runtime locks at runner construction (defense in depth)
        self._model_spec = validate_vlm_model_license(backend.model_id)
        if backend.enable_thinking is not False:
            raise ValueError("DR-09 runtime lock: backend.enable_thinking MUST be False")
        if backend.max_output_tokens > MAX_OUTPUT_TOKENS:
            raise ValueError(
                f"DR-09 runtime lock: backend.max_output_tokens={backend.max_output_tokens} > {MAX_OUTPUT_TOKENS}"
            )

        self._backend = backend
        eff_gate_cfg = (
            VlmGateConfig(max_age_ms=self._runner_cfg.max_age_ms)
            if gate_config is None
            else gate_config
        )
        self._gate = AsymmetricSimplexVlmGate(eff_gate_cfg)
        self._odd_bridge = OpticalOddLiveBridge(optical_odd_config)
        self._calibrator = FrozenConformalBceaCalibrator(conformal_calibration)
        self._buffer = SingleSlotDropStaleBuffer(max_age_ms=self._runner_cfg.max_age_ms)
        self._seq: int = 0
        self._mav_seq: int = 0

    @property
    def backend(self) -> VlmBackendProtocol:
        return self._backend

    @property
    def gate(self) -> AsymmetricSimplexVlmGate:
        return self._gate

    @property
    def optical_odd_bridge(self) -> OpticalOddLiveBridge:
        return self._odd_bridge

    @property
    def calibrator(self) -> FrozenConformalBceaCalibrator:
        return self._calibrator

    @property
    def buffer(self) -> SingleSlotDropStaleBuffer:
        return self._buffer

    @property
    def seq(self) -> int:
        return self._seq

    def reset(self) -> None:
        """Reset monotonic sequence, gate debounce windows, and Optical-ODD MLTL rings."""
        self._seq = 0
        self._mav_seq = 0
        self._gate.reset()
        self._odd_bridge.reset()
        self._buffer = SingleSlotDropStaleBuffer(max_age_ms=self._runner_cfg.max_age_ms)

    def submit_frame(self, frame: BufferedVlmFrame) -> bool:
        """Submit latest frame to the single-slot drop-stale buffer (evicts stale unconsumed slot)."""
        return self._buffer.submit(frame)

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        if self._seq == 0:
            self._seq = 1
        return self._seq

    def _next_mav_seq(self) -> int:
        val = self._mav_seq & 0xFF
        self._mav_seq = (self._mav_seq + 1) & 0xFF
        return val

    def process_frame(
        self,
        *,
        eo_bgr: np.ndarray,
        capture_ts_ms: int,
        now_ms: int,
        reachable_flz_mask: int,
        lwir_bgr: np.ndarray | None = None,
        certified_geometric_hazards: AbstractSet[str] = frozenset(),
        cert_hazards_mask: int = 0,
        hazardous_flz_ids: AbstractSet[int] = frozenset(),
        scenario_hint: str = "",
        rvr_km: float | None = None,
        beta_ext_km_inv: float | None = None,
        horizon_row_frac: float = 0.50,
        sun_boresight_deg: float = 95.0,
        t_win_c: float = 20.0,
        t_amb_c: float = 20.0,
        rh_pct: float = 45.0,
        attitude_in_odd: bool = True,
        raw_vlm_absolute_scores: Mapping[int, float] | None = None,
        corrupt_wire_byte_index: int | None = None,
    ) -> VlmMonitorStepResult:
        """Run one complete DR-09 6-stage evaluation over a real BGR (+ optional LWIR) frame."""
        t0 = time.perf_counter()
        mask = _validate_reachable_flz_mask(reachable_flz_mask)
        candidate_flz_ids = tuple(z for z in range(MAX_FLZ_ID) if (mask & (1 << z)) != 0)
        cert_mask = int(cert_hazards_mask) & 0xFF

        # 1. Always run zero-ML 5-mode Optical-ODD monitor first (Stage 3 independent ground truth)
        odd_decision, odd_reading = self._odd_bridge.evaluate_bgr(
            eo_bgr,
            lwir_bgr=lwir_bgr,
            rvr_km=rvr_km,
            beta_ext_km_inv=beta_ext_km_inv,
            sun_boresight_deg=sun_boresight_deg,
            t_win_c=t_win_c,
            t_amb_c=t_amb_c,
            rh_pct=rh_pct,
            horizon_row_frac=horizon_row_frac,
            attitude_in_odd=attitude_in_odd,
        )

        # 2. Single-slot drop-stale freshness guard (<= 200 ms)
        self._buffer.submit(
            BufferedVlmFrame(
                bgr_frame=eo_bgr,
                capture_ts_ms=int(capture_ts_ms),
                reachable_flz_mask=mask,
                scenario_hint=scenario_hint,
                lwir_bgr_frame=lwir_bgr,
            )
        )
        fresh_slot = self._buffer.take_fresh(int(now_ms))

        parsed_output: ParsedVlmOutput | None = None
        conformal_verdict: ConformalCalibrationVerdict | None = None
        advisory_packet: VlmAdvisoryPacket | None = None
        wire_bytes: bytes | None = None
        grammar_error: str | None = None
        packet_input: bytes | None = None
        emitted_seq = self._seq

        if fresh_slot is None:
            # Watchdog expired or future capture timestamp -> Stage 1 VLM_OFFLINE
            packet_input = None
        else:
            req = VlmInferenceRequest(
                bgr_frame=fresh_slot.bgr_frame,
                capture_ts_ms=fresh_slot.capture_ts_ms,
                reachable_flz_mask=mask,
                scenario_hint=scenario_hint,
            )
            backend_resp = self._backend.infer(req)
            if backend_resp is None:
                # Local HTTP timeout / offline / >200 ms overrun -> fail-closed Stage 1 VLM_OFFLINE
                packet_input = None
            else:
                grammar = StaticVlmTrieGrammar(
                    mask, max_tokens=self._runner_cfg.max_output_tokens
                )
                try:
                    parsed_output = grammar.parse(backend_resp.raw_text)
                except VlmGrammarError as exc:
                    grammar_error = str(exc)
                    # Pass empty buffer into Stage 2 so gate deterministically emits SCHEMA_OR_CRC_FAIL
                    packet_input = b""
                else:
                    conformal_verdict = self._calibrator.calibrate(
                        hazard=parsed_output.hazard,
                        p_image=backend_resp.p_image,
                        token_entropy=backend_resp.token_entropy,
                        ood_distance=backend_resp.ood_distance,
                        optical_odd_degraded=bool(odd_decision.vlm_gate_reject),
                    )
                    emitted_seq = self._next_seq()
                    advisory_packet = VlmAdvisoryPacket(
                        seq=emitted_seq,
                        capture_ts_ms=int(capture_ts_ms),
                        hazard_enum=int(parsed_output.hazard),
                        azimuth_sector=int(parsed_output.azimuth_sector),
                        ranked_zone_ids=parsed_output.ranked_zone_ids,
                        conformal_valid_bit=int(conformal_verdict.conformal_valid_bit),
                        aleatoric_flag=int(conformal_verdict.aleatoric_flag),
                        epistemic_flag=int(conformal_verdict.epistemic_flag),
                    )
                    wire_bytes = advisory_packet.pack()
                    assert len(wire_bytes) == PACKET_SIZE
                    if corrupt_wire_byte_index is not None:
                        corrupted = bytearray(wire_bytes)
                        corrupted[int(corrupt_wire_byte_index) % PACKET_SIZE] ^= 0x01
                        wire_bytes = bytes(corrupted)
                    packet_input = wire_bytes

        # 3. 6-Stage DO-178C DAL B/C Deterministic Verification Gate
        gate_decision = self._gate.evaluate(
            packet_or_bytes=packet_input,
            now_ms=int(now_ms),
            optical_odd=odd_reading,
            certified_geometric_hazards=certified_geometric_hazards,
            candidate_flz_ids=candidate_flz_ids,
            hazardous_flz_ids=hazardous_flz_ids,
            raw_vlm_absolute_scores=raw_vlm_absolute_scores,
        )
        assert gate_decision.actuator_authority is False

        # 4. Pessimistic union bitmask (matches frozen `xvs-odd-vlm` invariant: cert | active)
        active_vlm_mask = active_vetoes_to_mask(gate_decision.accepted_vlm_vetoes)
        final_hazards_mask = cert_mask | active_vlm_mask

        # 5. Emit verified MAVLink 2.0 `#50003 POD_VLM_SEMANTIC_VETO` frame
        mav_msg, mav_bytes = build_pod_vlm_semantic_veto_frame(
            gate_decision,
            seq=emitted_seq,
            capture_ts_ms=int(capture_ts_ms),
            packet=advisory_packet,
            mav_seq=self._next_mav_seq(),
            sysid=self._runner_cfg.mav_sysid,
            compid=self._runner_cfg.mav_compid,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return VlmMonitorStepResult(
            seq=emitted_seq,
            capture_ts_ms=int(capture_ts_ms),
            evaluated_now_ms=int(now_ms),
            backend_label=self._backend.backend_label,
            is_neural_vlm=self._backend.is_neural_vlm,
            model_id=self._backend.model_id,
            spdx_license=self._model_spec.spdx_license,
            enable_thinking=False,
            optical_odd_decision=odd_decision,
            optical_odd_reading=odd_reading,
            parsed_output=parsed_output,
            conformal_verdict=conformal_verdict,
            advisory_packet=advisory_packet,
            wire_packet_bytes=wire_bytes,
            gate_decision=gate_decision,
            mavlink_message=mav_msg,
            mavlink_frame_bytes=mav_bytes,
            cert_hazards_mask=cert_mask,
            active_vlm_mask=active_vlm_mask,
            final_hazards_mask=final_hazards_mask,
            actuator_authority=False,
            pipeline_latency_ms=round(elapsed_ms, 3),
            grammar_error=grammar_error,
        )
