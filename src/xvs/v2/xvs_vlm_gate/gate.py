"""DR-09 Asymmetric Simplex 6-Stage Deterministic VLM Verification Gate (IDAL D/E Isolation).

Strictly civil/dual-use unmanned aviation safety peripheral per:
- OpenXVS DR-09 §4 (Asymmetric Simplex Architecture & 6-Stage DO-178C DAL B/C Gate)
- EASA Proposed Issue 03 (Jun 3, 2026, Table 2: IDAL D ceiling for Large OTS Models / VLMs;
  Objective RA-08 aleatoric vs epistemic uncertainty separation)
- FAA DOT/FAA/TC-26/20 (Jul 2026: Overarching Properties — Intent, Correctness, Innocuity)
- ASTM F3269-21 §1.2 (VLM strictly outside the certified Safety Monitor; zero actuator authority)

Zero third-party dependencies, zero heap-unbounded buffers, zero unannotated `.get()` calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
import struct
from typing import AbstractSet, Mapping, Sequence
import zlib

PACKET_MAGIC: bytes = b"XV"
PACKET_VERSION: int = 1
PACKET_SIZE: int = 32
BODY_SIZE: int = 28
NUM_AZIMUTH_SECTORS: int = 12
NUM_RANKED_ZONES: int = 4
UNUSED_ZONE_ID: int = 0xFF

# Little-endian 32-byte fixed wire layout (C / Rust #[repr(C)] compatible):
#   2s  magic ("XV")
#   B   version (1)
#   B   reserved (0)
#   I   seq (uint32)
#   Q   capture_ts_ms (uint64)
#   B   hazard_enum (uint8)
#   B   azimuth_sector (uint8, 0..11)
#   4B  ranked_zone_ids (4 x uint8, 0xFF = unused)
#   B   conformal_valid_bit (0 or 1)
#   B   aleatoric_flag (0 or 1)
#   B   epistemic_flag (0 or 1)
#   3B  pad (0, 0, 0)
#   I   crc32 (IEEE 802.3 CRC32 over preceding 28 bytes)
_BODY_STRUCT = struct.Struct("<2sBBIQBB4BBBB3B")
_CRC_STRUCT = struct.Struct("<I")
assert _BODY_STRUCT.size == BODY_SIZE
assert _BODY_STRUCT.size + _CRC_STRUCT.size == PACKET_SIZE


class VlmHazard(IntEnum):
    """Fixed semantic hazard vocabulary emitted by the IDAL-D/E VLM peripheral."""

    NONE = 0
    RUNWAY_INCURSION = 1
    BIRD_FLOCK = 2
    CONVECTIVE_VIRGA = 3
    CRANE_PROXY = 4
    FLZ_HAZARD = 5


_VALID_HAZARD_VALUES: frozenset[int] = frozenset(int(h) for h in VlmHazard)
_TRACKED_HAZARDS: tuple[VlmHazard, ...] = (
    VlmHazard.RUNWAY_INCURSION,
    VlmHazard.BIRD_FLOCK,
    VlmHazard.CONVECTIVE_VIRGA,
    VlmHazard.CRANE_PROXY,
    VlmHazard.FLZ_HAZARD,
)


class GateStatus(str, Enum):
    """Deterministic outcome status of the 6-stage verification gate."""

    NOMINAL_ADVISORY = "NOMINAL_ADVISORY"
    VLM_OFFLINE = "VLM_OFFLINE"
    SCHEMA_OR_CRC_FAIL = "SCHEMA_OR_CRC_FAIL"
    OPTICAL_ODD_MASKED = "OPTICAL_ODD_MASKED"
    CONFORMAL_ABSTAIN = "CONFORMAL_ABSTAIN"
    DEBOUNCE_PENDING = "DEBOUNCE_PENDING"


class UncertaintySource(str, Enum):
    """EASA Objective RA-08 aleatoric vs epistemic uncertainty separation."""

    NONE = "NONE"
    ALEATORIC = "ALEATORIC"
    EPISTEMIC = "EPISTEMIC"
    BOTH = "BOTH"


def _compute_crc32(body: bytes) -> int:
    return zlib.crc32(body) & 0xFFFFFFFF


@dataclass(frozen=True)
class VlmAdvisoryPacket:
    """Fixed 32-byte deterministic advisory packet from the IDAL-D/E VLM peripheral."""

    seq: int
    capture_ts_ms: int
    hazard_enum: int
    azimuth_sector: int
    ranked_zone_ids: tuple[int, int, int, int] = (
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
    )
    conformal_valid_bit: int = 1
    aleatoric_flag: int = 0
    epistemic_flag: int = 0
    crc32: int | None = None

    def _pack_body(self) -> bytes:
        if len(self.ranked_zone_ids) != NUM_RANKED_ZONES:
            raise ValueError(
                f"ranked_zone_ids must have exactly {NUM_RANKED_ZONES} entries, "
                f"got {len(self.ranked_zone_ids)}"
            )
        z0, z1, z2, z3 = self.ranked_zone_ids
        return _BODY_STRUCT.pack(
            PACKET_MAGIC,
            PACKET_VERSION,
            0,
            int(self.seq),
            int(self.capture_ts_ms),
            int(self.hazard_enum),
            int(self.azimuth_sector),
            int(z0),
            int(z1),
            int(z2),
            int(z3),
            int(self.conformal_valid_bit),
            int(self.aleatoric_flag),
            int(self.epistemic_flag),
            0,
            0,
            0,
        )

    def computed_crc32(self) -> int:
        """Return IEEE CRC32 over the 28-byte packet body."""
        return _compute_crc32(self._pack_body())

    def pack(self) -> bytes:
        """Serialize into the exact 32-byte wire frame with IEEE CRC32."""
        body = self._pack_body()
        crc = _compute_crc32(body) if self.crc32 is None else int(self.crc32)
        return body + _CRC_STRUCT.pack(crc)

    @classmethod
    def unpack(cls, raw: object) -> PacketUnpackResult:
        """Deterministically decode a 32-byte wire buffer without raising uncaught exceptions.

        Verifies buffer type, exact length, framing magic/version, zero reserved/pad bytes,
        and IEEE CRC32. Returns `SCHEMA_OR_CRC_FAIL` on any corruption or truncation.
        """
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail="buffer must be bytes-like",
            )
        buf = bytes(raw)
        if len(buf) != PACKET_SIZE:
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail=f"invalid packet length {len(buf)} (expected {PACKET_SIZE})",
            )

        body = buf[:BODY_SIZE]
        (wire_crc,) = _CRC_STRUCT.unpack_from(buf, BODY_SIZE)
        expected_crc = _compute_crc32(body)
        if wire_crc != expected_crc:
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail=f"crc32 mismatch: wire=0x{wire_crc:08X} expected=0x{expected_crc:08X}",
            )

        (
            magic,
            version,
            reserved,
            seq,
            capture_ts_ms,
            hazard_enum,
            azimuth_sector,
            z0,
            z1,
            z2,
            z3,
            conformal_valid_bit,
            aleatoric_flag,
            epistemic_flag,
            p0,
            p1,
            p2,
        ) = _BODY_STRUCT.unpack(body)

        if magic != PACKET_MAGIC:
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail="invalid packet magic",
            )
        if version != PACKET_VERSION:
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail=f"unsupported packet version {version}",
            )
        if reserved != 0 or p0 != 0 or p1 != 0 or p2 != 0:
            return PacketUnpackResult(
                ok=False,
                packet=None,
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                detail="non-zero reserved or padding bytes",
            )

        packet = cls(
            seq=seq,
            capture_ts_ms=capture_ts_ms,
            hazard_enum=hazard_enum,
            azimuth_sector=azimuth_sector,
            ranked_zone_ids=(z0, z1, z2, z3),
            conformal_valid_bit=conformal_valid_bit,
            aleatoric_flag=aleatoric_flag,
            epistemic_flag=epistemic_flag,
            crc32=wire_crc,
        )
        return PacketUnpackResult(
            ok=True,
            packet=packet,
            status=GateStatus.NOMINAL_ADVISORY,
            detail="ok",
        )


@dataclass(frozen=True)
class PacketUnpackResult:
    """Result of wire-packet deserialization and CRC32 check."""

    ok: bool
    packet: VlmAdvisoryPacket | None
    status: GateStatus
    detail: str


@dataclass(frozen=True)
class OpticalOddReading:
    """Independent classical optical-ODD & sensor-health metrics (Stage 3)."""

    glare_sat_frac: float = 0.0
    droplet_occlusion_frac: float = 0.0
    fog_contrast: float = 1.0
    blur_laplacian: float = 200.0
    attitude_in_odd: bool = True


@dataclass(frozen=True)
class OpticalOddThresholds:
    """Deterministic Stage-3 optical ODD thresholds preventing rain/fog/glare ghosting."""

    max_glare_sat_frac: float = 0.15
    max_droplet_occlusion_frac: float = 0.12
    min_fog_contrast: float = 0.35
    min_blur_laplacian: float = 50.0


@dataclass(frozen=True)
class DebounceSpec:
    """Stage-5 M-of-N confirmation & asymmetric consecutive-clear release specification."""

    confirm_m: int
    window_n: int
    release_clear_frames: int

    def __post_init__(self) -> None:
        if self.confirm_m < 1 or self.window_n < self.confirm_m:
            raise ValueError(
                f"invalid M-of-N spec: confirm_m={self.confirm_m}, window_n={self.window_n}"
            )
        if self.release_clear_frames < 1:
            raise ValueError(
                f"release_clear_frames must be >= 1, got {self.release_clear_frames}"
            )


def default_debounce_specs() -> dict[VlmHazard, DebounceSpec]:
    """DR-09 default per-hazard M-of-N & asymmetric release hysteresis table."""
    return {
        VlmHazard.RUNWAY_INCURSION: DebounceSpec(
            confirm_m=2, window_n=3, release_clear_frames=4
        ),
        VlmHazard.BIRD_FLOCK: DebounceSpec(
            confirm_m=3, window_n=5, release_clear_frames=5
        ),
        VlmHazard.CONVECTIVE_VIRGA: DebounceSpec(
            confirm_m=3, window_n=5, release_clear_frames=5
        ),
        VlmHazard.CRANE_PROXY: DebounceSpec(
            confirm_m=3, window_n=5, release_clear_frames=5
        ),
        VlmHazard.FLZ_HAZARD: DebounceSpec(
            confirm_m=3, window_n=5, release_clear_frames=5
        ),
    }


@dataclass(frozen=True)
class VlmGateConfig:
    """Configuration for the 6-stage DO-178C DAL B/C VLM verification gate."""

    max_age_ms: int = 200
    optical_odd: OpticalOddThresholds = field(default_factory=OpticalOddThresholds)
    debounce_specs: Mapping[VlmHazard, DebounceSpec] = field(
        default_factory=default_debounce_specs
    )
    allow_absolute_flz_scores: bool = False

    def __post_init__(self) -> None:
        if self.max_age_ms < 0:
            raise ValueError(f"max_age_ms must be >= 0, got {self.max_age_ms}")
        if self.allow_absolute_flz_scores:
            raise ValueError(
                "DR-09 Ranking-vs-Scoring decoupling forbids absolute scalar FLZ safety scores"
            )
        for hazard in _TRACKED_HAZARDS:
            if hazard not in self.debounce_specs:
                raise ValueError(f"missing debounce spec for {hazard.name}")


@dataclass(frozen=True)
class GateDecision:
    """Deterministic output emitted to the certified Simplex RTA / cockpit display."""

    status: GateStatus
    stage_reached: int
    vlm_online: bool
    vlm_used: bool
    actuator_authority: bool
    certified_geometric_hazards: frozenset[str]
    accepted_vlm_vetoes: frozenset[str]
    final_hazards: frozenset[str]
    ranked_flz_ids: tuple[int, ...]
    uncertainty_source: UncertaintySource
    absolute_scores_ignored: bool
    detail: str


class _HazardDebounceState:
    """Bounded per-hazard ring window + asymmetric consecutive-clear hysteresis state."""

    __slots__ = ("spec", "window", "latched", "clear_streak")

    def __init__(self, spec: DebounceSpec) -> None:
        self.spec = spec
        self.window: deque[bool] = deque(maxlen=spec.window_n)
        self.latched: bool = False
        self.clear_streak: int = 0

    def reset(self) -> None:
        self.window.clear()
        self.latched = False
        self.clear_streak = 0

    def update(self, hit: bool) -> bool:
        self.window.append(hit)
        if not self.latched:
            if sum(self.window) >= self.spec.confirm_m:
                self.latched = True
                self.clear_streak = 0
        else:
            if hit:
                self.clear_streak = 0
            else:
                self.clear_streak += 1
                if self.clear_streak >= self.spec.release_clear_frames:
                    self.latched = False
                    self.window.clear()
                    self.clear_streak = 0
        return self.latched


def _validate_packet_schema(packet: VlmAdvisoryPacket) -> str | None:
    """Stage-2 static schema, range & CRC32 integrity check."""
    if not isinstance(packet.seq, int) or isinstance(packet.seq, bool) or not (0 <= packet.seq <= 0xFFFFFFFF):
        return f"seq out of uint32 range: {packet.seq!r}"
    if (
        not isinstance(packet.capture_ts_ms, int)
        or isinstance(packet.capture_ts_ms, bool)
        or not (0 <= packet.capture_ts_ms <= 0xFFFFFFFFFFFFFFFF)
    ):
        return f"capture_ts_ms out of uint64 range: {packet.capture_ts_ms!r}"
    if (
        not isinstance(packet.hazard_enum, int)
        or isinstance(packet.hazard_enum, bool)
        or int(packet.hazard_enum) not in _VALID_HAZARD_VALUES
    ):
        return f"invalid hazard_enum: {packet.hazard_enum!r}"
    if (
        not isinstance(packet.azimuth_sector, int)
        or isinstance(packet.azimuth_sector, bool)
        or not (0 <= packet.azimuth_sector < NUM_AZIMUTH_SECTORS)
    ):
        return f"azimuth_sector out of range 0..11: {packet.azimuth_sector!r}"
    if not isinstance(packet.ranked_zone_ids, (tuple, list)) or len(packet.ranked_zone_ids) != NUM_RANKED_ZONES:
        return f"ranked_zone_ids must be a {NUM_RANKED_ZONES}-element sequence: {packet.ranked_zone_ids!r}"
    for z in packet.ranked_zone_ids:
        if not isinstance(z, int) or isinstance(z, bool) or not (0 <= z <= 0xFF):
            return f"ranked_zone_id out of uint8 range: {z!r}"
    if packet.conformal_valid_bit not in (0, 1) or isinstance(packet.conformal_valid_bit, bool):
        return f"conformal_valid_bit must be 0 or 1: {packet.conformal_valid_bit!r}"
    if packet.aleatoric_flag not in (0, 1) or isinstance(packet.aleatoric_flag, bool):
        return f"aleatoric_flag must be 0 or 1: {packet.aleatoric_flag!r}"
    if packet.epistemic_flag not in (0, 1) or isinstance(packet.epistemic_flag, bool):
        return f"epistemic_flag must be 0 or 1: {packet.epistemic_flag!r}"
    if packet.crc32 is not None:
        expected = packet.computed_crc32()
        if packet.crc32 != expected:
            return f"crc32 mismatch: packet=0x{packet.crc32:08X} expected=0x{expected:08X}"
    return None


def _check_optical_odd(
    reading: OpticalOddReading, thresholds: OpticalOddThresholds
) -> str | None:
    """Stage-3 deterministic classical optical-ODD health check."""
    numeric_fields = (
        ("glare_sat_frac", reading.glare_sat_frac),
        ("droplet_occlusion_frac", reading.droplet_occlusion_frac),
        ("fog_contrast", reading.fog_contrast),
        ("blur_laplacian", reading.blur_laplacian),
    )
    for name, value in numeric_fields:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return f"non-finite optical metric {name}={value!r}"
    if not reading.attitude_in_odd:
        return "attitude outside optical ODD"
    if reading.glare_sat_frac < 0.0 or reading.glare_sat_frac > thresholds.max_glare_sat_frac:
        return (
            f"glare_sat_frac={reading.glare_sat_frac:.3f} exceeds "
            f"{thresholds.max_glare_sat_frac:.3f}"
        )
    if (
        reading.droplet_occlusion_frac < 0.0
        or reading.droplet_occlusion_frac > thresholds.max_droplet_occlusion_frac
    ):
        return (
            f"droplet_occlusion_frac={reading.droplet_occlusion_frac:.3f} exceeds "
            f"{thresholds.max_droplet_occlusion_frac:.3f}"
        )
    if reading.fog_contrast < thresholds.min_fog_contrast:
        return (
            f"fog_contrast={reading.fog_contrast:.3f} below "
            f"{thresholds.min_fog_contrast:.3f}"
        )
    if reading.blur_laplacian < thresholds.min_blur_laplacian:
        return (
            f"blur_laplacian={reading.blur_laplacian:.1f} below "
            f"{thresholds.min_blur_laplacian:.1f}"
        )
    return None


def _classify_uncertainty(aleatoric_flag: int, epistemic_flag: int) -> UncertaintySource:
    if aleatoric_flag and epistemic_flag:
        return UncertaintySource.BOTH
    if aleatoric_flag:
        return UncertaintySource.ALEATORIC
    if epistemic_flag:
        return UncertaintySource.EPISTEMIC
    return UncertaintySource.NONE


def _filter_eligible_flz(
    candidate_flz_ids: Sequence[int],
    hazardous_flz_ids: AbstractSet[int],
) -> tuple[int, ...]:
    """Return unique pre-verified reachable/flat FLZ IDs excluding hazardous/unused IDs."""
    seen: set[int] = set()
    eligible: list[int] = []
    for zid in candidate_flz_ids:
        if zid == UNUSED_ZONE_ID or zid in hazardous_flz_ids or zid in seen:
            continue
        seen.add(zid)
        eligible.append(zid)
    return tuple(eligible)


def _rank_flz_relative_only(
    eligible_flz_ids: tuple[int, ...],
    vlm_ranked_zone_ids: tuple[int, int, int, int],
) -> tuple[int, ...]:
    """Stage-6 relative-only FLZ re-ordering strictly within pre-verified eligible candidates."""
    if not eligible_flz_ids:
        return ()
    eligible_set = set(eligible_flz_ids)
    ordered: list[int] = []
    seen: set[int] = set()
    for zid in vlm_ranked_zone_ids:
        if zid == UNUSED_ZONE_ID or zid not in eligible_set or zid in seen:
            continue
        seen.add(zid)
        ordered.append(zid)
    for zid in eligible_flz_ids:
        if zid not in seen:
            ordered.append(zid)
    return tuple(ordered)


class AsymmetricSimplexVlmGate:
    """6-Stage Deterministic Verification Gate isolating an IDAL-D/E VLM peripheral.

    Enforces:
      - Stage 1: Watchdog heartbeat, monotonic sequence number & freshness (<= max_age_ms)
      - Stage 2: Static fixed-size schema, enum/range & IEEE CRC32 integrity
      - Stage 3: Independent classical optical-ODD & sensor-health masking
      - Stage 4: Conformal abstention & strict Monotonic Pessimistic Union (veto-only)
      - Stage 5: Per-hazard bounded M-of-N debounce & asymmetric release hysteresis
      - Stage 6: Kinematic feasibility & relative-only FLZ re-ordering (no absolute scores)
    """

    def __init__(self, config: VlmGateConfig | None = None) -> None:
        self._config: VlmGateConfig = VlmGateConfig() if config is None else config
        self._last_seq: int | None = None
        self._debounce: dict[VlmHazard, _HazardDebounceState] = {
            hazard: _HazardDebounceState(self._config.debounce_specs[hazard])
            for hazard in _TRACKED_HAZARDS
        }

    @property
    def config(self) -> VlmGateConfig:
        return self._config

    @property
    def last_seq(self) -> int | None:
        return self._last_seq

    def reset(self) -> None:
        """Reset watchdog sequence tracking and all bounded debounce windows."""
        self._last_seq = None
        for state in self._debounce.values():
            state.reset()

    def _active_latched_vetoes(self) -> frozenset[str]:
        return frozenset(
            f"VLM:{hazard.name}"
            for hazard, state in self._debounce.items()
            if state.latched
        )

    def _clear_all_debounce(self) -> None:
        for state in self._debounce.values():
            state.reset()

    def evaluate(
        self,
        *,
        packet_or_bytes: VlmAdvisoryPacket | bytes | bytearray | memoryview | None,
        now_ms: int,
        optical_odd: OpticalOddReading,
        certified_geometric_hazards: AbstractSet[str] = frozenset(),
        candidate_flz_ids: Sequence[int] = (),
        hazardous_flz_ids: AbstractSet[int] = frozenset(),
        raw_vlm_absolute_scores: Mapping[int, float] | None = None,
    ) -> GateDecision:
        """Execute all 6 sequential verification stages fail-closed.

        Any `raw_vlm_absolute_scores` passed by a caller are explicitly ignored per DR-09
        Ranking-vs-Scoring decoupling (`absolute_scores_ignored=True` when supplied).
        """
        cert_hazards = frozenset(certified_geometric_hazards)
        eligible_flz = _filter_eligible_flz(candidate_flz_ids, hazardous_flz_ids)
        scores_ignored = raw_vlm_absolute_scores is not None

        # Stage 1 pre-check: missing packet (GPU hang / link timeout) -> VLM_OFFLINE
        if packet_or_bytes is None:
            self._clear_all_debounce()
            return GateDecision(
                status=GateStatus.VLM_OFFLINE,
                stage_reached=1,
                vlm_online=False,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=frozenset(),
                final_hazards=cert_hazards,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=UncertaintySource.NONE,
                absolute_scores_ignored=scores_ignored,
                detail="missing packet (watchdog timeout)",
            )

        # Stage 2 wire unpack & CRC32 / direct-object schema integrity (precedes trusting header fields)
        if isinstance(packet_or_bytes, VlmAdvisoryPacket):
            packet = packet_or_bytes
            schema_pre_err = _validate_packet_schema(packet)
            if schema_pre_err is not None:
                active = self._active_latched_vetoes()
                return GateDecision(
                    status=GateStatus.SCHEMA_OR_CRC_FAIL,
                    stage_reached=2,
                    vlm_online=self._last_seq is not None,
                    vlm_used=False,
                    actuator_authority=False,
                    certified_geometric_hazards=cert_hazards,
                    accepted_vlm_vetoes=active,
                    final_hazards=cert_hazards | active,
                    ranked_flz_ids=eligible_flz,
                    uncertainty_source=UncertaintySource.NONE,
                    absolute_scores_ignored=scores_ignored,
                    detail=schema_pre_err,
                )
        else:
            unpacked = VlmAdvisoryPacket.unpack(packet_or_bytes)
            if not unpacked.ok or unpacked.packet is None:
                active = self._active_latched_vetoes()
                return GateDecision(
                    status=GateStatus.SCHEMA_OR_CRC_FAIL,
                    stage_reached=2,
                    vlm_online=self._last_seq is not None,
                    vlm_used=False,
                    actuator_authority=False,
                    certified_geometric_hazards=cert_hazards,
                    accepted_vlm_vetoes=active,
                    final_hazards=cert_hazards | active,
                    ranked_flz_ids=eligible_flz,
                    uncertainty_source=UncertaintySource.NONE,
                    absolute_scores_ignored=scores_ignored,
                    detail=unpacked.detail,
                )
            packet = unpacked.packet

        # Stage 1: Watchdog freshness (0 <= age <= max_age_ms) & strictly monotonic seq
        age_ms = now_ms - packet.capture_ts_ms
        if age_ms < 0 or age_ms > self._config.max_age_ms:
            self._clear_all_debounce()
            return GateDecision(
                status=GateStatus.VLM_OFFLINE,
                stage_reached=1,
                vlm_online=False,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=frozenset(),
                final_hazards=cert_hazards,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=UncertaintySource.NONE,
                absolute_scores_ignored=scores_ignored,
                detail=f"stale or future packet age_ms={age_ms} (limit 0..{self._config.max_age_ms})",
            )
        if self._last_seq is not None and packet.seq <= self._last_seq:
            self._clear_all_debounce()
            return GateDecision(
                status=GateStatus.VLM_OFFLINE,
                stage_reached=1,
                vlm_online=False,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=frozenset(),
                final_hazards=cert_hazards,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=UncertaintySource.NONE,
                absolute_scores_ignored=scores_ignored,
                detail=f"non-monotonic seq={packet.seq} <= last_seq={self._last_seq}",
            )

        # Stage 2: Static schema & range validation
        schema_err = _validate_packet_schema(packet)
        if schema_err is not None:
            active = self._active_latched_vetoes()
            return GateDecision(
                status=GateStatus.SCHEMA_OR_CRC_FAIL,
                stage_reached=2,
                vlm_online=self._last_seq is not None,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=active,
                final_hazards=cert_hazards | active,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=UncertaintySource.NONE,
                absolute_scores_ignored=scores_ignored,
                detail=schema_err,
            )

        # Commit monotonic sequence only after Stages 1 & 2 succeed
        self._last_seq = packet.seq
        uncertainty = _classify_uncertainty(packet.aleatoric_flag, packet.epistemic_flag)

        # Stage 3: Deterministic classical Optical-ODD & sensor-health mask
        odd_err = _check_optical_odd(optical_odd, self._config.optical_odd)
        if odd_err is not None:
            self._clear_all_debounce()
            return GateDecision(
                status=GateStatus.OPTICAL_ODD_MASKED,
                stage_reached=3,
                vlm_online=True,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=frozenset(),
                final_hazards=cert_hazards,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=uncertainty,
                absolute_scores_ignored=scores_ignored,
                detail=odd_err,
            )

        # Stage 4: Conformal abstention (NEVER treated as hazard clear) + RA-08 separation
        if packet.conformal_valid_bit == 0:
            active = self._active_latched_vetoes()
            return GateDecision(
                status=GateStatus.CONFORMAL_ABSTAIN,
                stage_reached=4,
                vlm_online=True,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=active,
                final_hazards=cert_hazards | active,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=uncertainty,
                absolute_scores_ignored=scores_ignored,
                detail=f"conformal_valid_bit=0 ({uncertainty.value})",
            )

        # Stage 5: Spatiotemporal M-of-N debounce & asymmetric release hysteresis
        reported_hazard = VlmHazard(packet.hazard_enum)
        for hazard in _TRACKED_HAZARDS:
            self._debounce[hazard].update(reported_hazard == hazard)

        active_vetoes = self._active_latched_vetoes()
        # Strict Monotonic Pessimistic Union: VLM can ONLY add vetoes, NEVER clear geometric hazards
        final_hazards = cert_hazards | active_vetoes

        if reported_hazard != VlmHazard.NONE and not self._debounce[reported_hazard].latched:
            state = self._debounce[reported_hazard]
            return GateDecision(
                status=GateStatus.DEBOUNCE_PENDING,
                stage_reached=5,
                vlm_online=True,
                vlm_used=False,
                actuator_authority=False,
                certified_geometric_hazards=cert_hazards,
                accepted_vlm_vetoes=active_vetoes,
                final_hazards=final_hazards,
                ranked_flz_ids=eligible_flz,
                uncertainty_source=uncertainty,
                absolute_scores_ignored=scores_ignored,
                detail=(
                    f"{reported_hazard.name} pending M-of-N "
                    f"({sum(state.window)}/{state.spec.confirm_m} in window {state.spec.window_n})"
                ),
            )

        # Stage 6: Kinematic feasibility & relative-only FLZ re-ordering (zero absolute scores)
        ranked_flz = _rank_flz_relative_only(eligible_flz, packet.ranked_zone_ids)
        return GateDecision(
            status=GateStatus.NOMINAL_ADVISORY,
            stage_reached=6,
            vlm_online=True,
            vlm_used=True,
            actuator_authority=False,
            certified_geometric_hazards=cert_hazards,
            accepted_vlm_vetoes=active_vetoes,
            final_hazards=final_hazards,
            ranked_flz_ids=ranked_flz,
            uncertainty_source=uncertainty,
            absolute_scores_ignored=scores_ignored,
            detail=(
                f"accepted sector={packet.azimuth_sector} "
                f"hazard={reported_hazard.name} active_vetoes={len(active_vetoes)}"
            ),
        )
