"""HIL Fault-Injection & Verification Suite for DR-09 6-Stage VLM Verification Gate."""

from __future__ import annotations

import math

import pytest

from xvs.v2.xvs_vlm_gate import (
    PACKET_SIZE,
    UNUSED_ZONE_ID,
    AsymmetricSimplexVlmGate,
    GateStatus,
    OpticalOddReading,
    UncertaintySource,
    VlmAdvisoryPacket,
    VlmGateConfig,
    VlmHazard,
)


def _nominal_odd() -> OpticalOddReading:
    return OpticalOddReading(
        glare_sat_frac=0.02,
        droplet_occlusion_frac=0.01,
        fog_contrast=0.82,
        blur_laplacian=180.0,
        attitude_in_odd=True,
    )


def _make_packet(
    *,
    seq: int,
    ts_ms: int,
    hazard: int = int(VlmHazard.NONE),
    azimuth_sector: int = 3,
    ranked_zone_ids: tuple[int, int, int, int] = (
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
    ),
    conformal_valid_bit: int = 1,
    aleatoric_flag: int = 0,
    epistemic_flag: int = 0,
) -> bytes:
    return VlmAdvisoryPacket(
        seq=seq,
        capture_ts_ms=ts_ms,
        hazard_enum=hazard,
        azimuth_sector=azimuth_sector,
        ranked_zone_ids=ranked_zone_ids,
        conformal_valid_bit=conformal_valid_bit,
        aleatoric_flag=aleatoric_flag,
        epistemic_flag=epistemic_flag,
    ).pack()


# ---------------------------------------------------------------------------
# Wire Packet Pack / Unpack & Stage 2 (Schema, Range, CRC32)
# ---------------------------------------------------------------------------


def test_wire_packet_roundtrip_and_fixed_size() -> None:
    pkt = VlmAdvisoryPacket(
        seq=42,
        capture_ts_ms=1_700_000_000_123,
        hazard_enum=int(VlmHazard.RUNWAY_INCURSION),
        azimuth_sector=11,
        ranked_zone_ids=(7, 2, 5, UNUSED_ZONE_ID),
        conformal_valid_bit=1,
        aleatoric_flag=0,
        epistemic_flag=1,
    )
    raw = pkt.pack()
    assert len(raw) == PACKET_SIZE
    assert 32 <= len(raw) <= 40

    res = VlmAdvisoryPacket.unpack(raw)
    assert res.ok is True
    assert res.status == GateStatus.NOMINAL_ADVISORY
    assert res.packet is not None
    assert res.packet.seq == 42
    assert res.packet.capture_ts_ms == 1_700_000_000_123
    assert res.packet.hazard_enum == int(VlmHazard.RUNWAY_INCURSION)
    assert res.packet.azimuth_sector == 11
    assert res.packet.ranked_zone_ids == (7, 2, 5, UNUSED_ZONE_ID)
    assert res.packet.conformal_valid_bit == 1
    assert res.packet.aleatoric_flag == 0
    assert res.packet.epistemic_flag == 1
    assert res.packet.crc32 == pkt.computed_crc32()


@pytest.mark.parametrize(
    "corrupt_input",
    [
        b"",
        b"XV\x01\x00truncated",
        b"\x00" * 31,
        b"\x00" * 33,
        b"\xFF" * 32,
        "not-bytes",
        12345,
    ],
)
def test_unpack_never_throws_on_truncated_or_corrupt_buffers(corrupt_input: object) -> None:
    res = VlmAdvisoryPacket.unpack(corrupt_input)
    assert res.ok is False
    assert res.packet is None
    assert res.status == GateStatus.SCHEMA_OR_CRC_FAIL


def test_stage2_flipped_crc_bit_and_out_of_range_fields_rejected() -> None:
    gate = AsymmetricSimplexVlmGate()
    cert = frozenset({"ADSB:N12345"})

    # 1. Single-bit flip in wire payload -> CRC32 fails cleanly
    raw = bytearray(_make_packet(seq=1, ts_ms=1000, hazard=int(VlmHazard.CRANE_PROXY)))
    raw[10] ^= 0x01
    dec = gate.evaluate(
        packet_or_bytes=bytes(raw),
        now_ms=1050,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert dec.status == GateStatus.SCHEMA_OR_CRC_FAIL
    assert dec.stage_reached == 2
    assert dec.final_hazards == cert
    assert dec.actuator_authority is False
    assert gate.last_seq is None

    # 2. Valid CRC32 but invalid hazard_enum=99 -> Stage 2 rejects
    bad_enum = _make_packet(seq=1, ts_ms=1000, hazard=99)
    dec2 = gate.evaluate(
        packet_or_bytes=bad_enum,
        now_ms=1050,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert dec2.status == GateStatus.SCHEMA_OR_CRC_FAIL
    assert dec2.stage_reached == 2
    assert dec2.final_hazards == cert

    # 3. Valid CRC32 but azimuth_sector=12 (out of 0..11) -> Stage 2 rejects
    bad_az = _make_packet(seq=1, ts_ms=1000, hazard=int(VlmHazard.NONE), azimuth_sector=12)
    dec3 = gate.evaluate(
        packet_or_bytes=bad_az,
        now_ms=1050,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert dec3.status == GateStatus.SCHEMA_OR_CRC_FAIL
    assert dec3.stage_reached == 2
    assert dec3.final_hazards == cert

    # 4. Direct malformed VlmAdvisoryPacket object (out-of-range seq / bad crc32 / non-binary flag)
    bad_obj = VlmAdvisoryPacket(
        seq=-1,
        capture_ts_ms=1000,
        hazard_enum=int(VlmHazard.NONE),
        azimuth_sector=0,
        conformal_valid_bit=2,
        crc32=0xDEADBEEF,
    )
    dec4 = gate.evaluate(
        packet_or_bytes=bad_obj,
        now_ms=1050,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert dec4.status == GateStatus.SCHEMA_OR_CRC_FAIL
    assert dec4.stage_reached == 2
    assert dec4.final_hazards == cert

    # 5. Direct malformed VlmAdvisoryPacket with crc32=None and non-int capture_ts_ms / seq / zones
    for bad_direct in (
        VlmAdvisoryPacket(
            seq="bad",  # type: ignore[arg-type]
            capture_ts_ms="1000",  # type: ignore[arg-type]
            hazard_enum=int(VlmHazard.NONE),
            azimuth_sector=0,
            crc32=None,
        ),
        VlmAdvisoryPacket(
            seq=1,
            capture_ts_ms=None,  # type: ignore[arg-type]
            hazard_enum=int(VlmHazard.NONE),
            azimuth_sector=0,
            crc32=None,
        ),
        VlmAdvisoryPacket(
            seq=1,
            capture_ts_ms=1000,
            hazard_enum=int(VlmHazard.NONE),
            azimuth_sector=0,
            ranked_zone_ids=None,  # type: ignore[arg-type]
            crc32=None,
        ),
    ):
        dec5 = gate.evaluate(
            packet_or_bytes=bad_direct,
            now_ms=1050,
            optical_odd=_nominal_odd(),
            certified_geometric_hazards=cert,
        )
        assert dec5.status == GateStatus.SCHEMA_OR_CRC_FAIL
        assert dec5.stage_reached == 2
        assert dec5.final_hazards == cert
        assert dec5.actuator_authority is False


# ---------------------------------------------------------------------------
# Stage 1: Watchdog Heartbeat, Freshness & Monotonic Sequence
# ---------------------------------------------------------------------------


def test_stage1_gpu_hang_500ms_timeout_and_non_monotonic_seq() -> None:
    gate = AsymmetricSimplexVlmGate(VlmGateConfig(max_age_ms=200))
    cert = frozenset({"TAWS:TERRAIN_PULL_UP"})

    # Nominal fresh packet establishes seq=10
    d0 = gate.evaluate(
        packet_or_bytes=_make_packet(seq=10, ts_ms=1000),
        now_ms=1120,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert d0.status == GateStatus.NOMINAL_ADVISORY
    assert d0.vlm_online is True
    assert d0.final_hazards == cert

    # 500 ms GPU hang (> 200 ms max_age_ms) -> VLM_OFFLINE, zero perturbation to TAWS
    d_hang = gate.evaluate(
        packet_or_bytes=_make_packet(seq=11, ts_ms=1000, hazard=int(VlmHazard.RUNWAY_INCURSION)),
        now_ms=1500,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert d_hang.status == GateStatus.VLM_OFFLINE
    assert d_hang.stage_reached == 1
    assert d_hang.vlm_online is False
    assert d_hang.accepted_vlm_vetoes == frozenset()
    assert d_hang.final_hazards == cert

    # Missing packet (None) -> VLM_OFFLINE
    d_none = gate.evaluate(
        packet_or_bytes=None,
        now_ms=1600,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert d_none.status == GateStatus.VLM_OFFLINE
    assert d_none.final_hazards == cert

    # Replayed / out-of-order seq=10 (<= last_seq=10) -> VLM_OFFLINE
    d_replay = gate.evaluate(
        packet_or_bytes=_make_packet(seq=10, ts_ms=1550),
        now_ms=1600,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert d_replay.status == GateStatus.VLM_OFFLINE
    assert d_replay.stage_reached == 1

    # Future timestamp (now_ms < capture_ts_ms) -> VLM_OFFLINE
    d_future = gate.evaluate(
        packet_or_bytes=_make_packet(seq=12, ts_ms=1700),
        now_ms=1600,
        optical_odd=_nominal_odd(),
        certified_geometric_hazards=cert,
    )
    assert d_future.status == GateStatus.VLM_OFFLINE
    assert d_future.stage_reached == 1


# ---------------------------------------------------------------------------
# Stage 3: Deterministic Optical-ODD & Sensor-Health Monitor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_odd",
    [
        OpticalOddReading(glare_sat_frac=0.25, droplet_occlusion_frac=0.0, fog_contrast=0.8, blur_laplacian=150.0, attitude_in_odd=True),
        OpticalOddReading(glare_sat_frac=0.0, droplet_occlusion_frac=0.20, fog_contrast=0.8, blur_laplacian=150.0, attitude_in_odd=True),
        OpticalOddReading(glare_sat_frac=0.0, droplet_occlusion_frac=0.0, fog_contrast=0.20, blur_laplacian=150.0, attitude_in_odd=True),
        OpticalOddReading(glare_sat_frac=0.0, droplet_occlusion_frac=0.0, fog_contrast=0.8, blur_laplacian=20.0, attitude_in_odd=True),
        OpticalOddReading(glare_sat_frac=0.0, droplet_occlusion_frac=0.0, fog_contrast=0.8, blur_laplacian=150.0, attitude_in_odd=False),
        OpticalOddReading(glare_sat_frac=math.nan, droplet_occlusion_frac=0.0, fog_contrast=0.8, blur_laplacian=150.0, attitude_in_odd=True),
    ],
)
def test_stage3_optical_odd_fog_glare_droplet_masks_vlm(bad_odd: OpticalOddReading) -> None:
    gate = AsymmetricSimplexVlmGate()
    cert = frozenset({"DAIDALUS:INTRUDER_01"})

    # Even repeated crane hazards during rain/fog/glare MUST be masked (MK30 rain-ghosting guard)
    for idx in range(5):
        ts = 2000 + idx * 200
        dec = gate.evaluate(
            packet_or_bytes=_make_packet(seq=idx + 1, ts_ms=ts, hazard=int(VlmHazard.CRANE_PROXY)),
            now_ms=ts + 40,
            optical_odd=bad_odd,
            certified_geometric_hazards=cert,
            candidate_flz_ids=(10, 20),
        )
        assert dec.status == GateStatus.OPTICAL_ODD_MASKED
        assert dec.stage_reached == 3
        assert dec.vlm_used is False
        assert dec.accepted_vlm_vetoes == frozenset()
        assert dec.final_hazards == cert
        assert dec.ranked_flz_ids == (10, 20)


# ---------------------------------------------------------------------------
# Stage 4: Conformal Abstention & Strict Monotonic Pessimistic Union
# ---------------------------------------------------------------------------


def test_stage4_monotonic_pessimistic_union_blocks_false_clears() -> None:
    gate = AsymmetricSimplexVlmGate()
    cert = frozenset({"ADSB:TRAFFIC_ALERT", "TAWS:FLTA_WARNING"})

    # Adversarial silent 0-hazard (NONE) false clears from VLM cannot relax ADS-B / TAWS hazards
    for i in range(6):
        ts = 3000 + i * 200
        dec = gate.evaluate(
            packet_or_bytes=_make_packet(seq=i + 1, ts_ms=ts, hazard=int(VlmHazard.NONE)),
            now_ms=ts + 50,
            optical_odd=_nominal_odd(),
            certified_geometric_hazards=cert,
        )
        assert dec.status == GateStatus.NOMINAL_ADVISORY
        assert cert.issubset(dec.final_hazards)
        assert dec.final_hazards == cert
        assert dec.actuator_authority is False


def test_stage4_conformal_abstain_never_clears_latched_hazard_and_separates_ra08() -> None:
    gate = AsymmetricSimplexVlmGate()
    cert = frozenset({"ADSB:N999"})

    # Latch RUNWAY_INCURSION via 2-of-3 confirmed frames
    for i in range(2):
        ts = 4000 + i * 200
        d = gate.evaluate(
            packet_or_bytes=_make_packet(seq=i + 1, ts_ms=ts, hazard=int(VlmHazard.RUNWAY_INCURSION)),
            now_ms=ts + 30,
            optical_odd=_nominal_odd(),
            certified_geometric_hazards=cert,
        )
    assert d.status == GateStatus.NOMINAL_ADVISORY
    assert "VLM:RUNWAY_INCURSION" in d.accepted_vlm_vetoes
    assert d.final_hazards == cert | {"VLM:RUNWAY_INCURSION"}

    # Send 10 consecutive conformal_valid_bit=0 abstentions reporting NONE:
    # MUST NOT clear latched RUNWAY_INCURSION, and MUST report RA-08 uncertainty source.
    for j in range(10):
        ts = 4400 + j * 200
        aleatoric = 1 if j % 2 == 0 else 0
        epistemic = 1 if j % 3 == 0 else 0
        dec = gate.evaluate(
            packet_or_bytes=_make_packet(
                seq=3 + j,
                ts_ms=ts,
                hazard=int(VlmHazard.NONE),
                conformal_valid_bit=0,
                aleatoric_flag=aleatoric,
                epistemic_flag=epistemic,
            ),
            now_ms=ts + 30,
            optical_odd=_nominal_odd(),
            certified_geometric_hazards=cert,
        )
        assert dec.status == GateStatus.CONFORMAL_ABSTAIN
        assert dec.stage_reached == 4
        assert "VLM:RUNWAY_INCURSION" in dec.accepted_vlm_vetoes
        assert dec.final_hazards == cert | {"VLM:RUNWAY_INCURSION"}
        if aleatoric and epistemic:
            assert dec.uncertainty_source == UncertaintySource.BOTH
        elif aleatoric:
            assert dec.uncertainty_source == UncertaintySource.ALEATORIC
        elif epistemic:
            assert dec.uncertainty_source == UncertaintySource.EPISTEMIC
        else:
            assert dec.uncertainty_source == UncertaintySource.NONE


# ---------------------------------------------------------------------------
# Stage 5: M-of-N Debounce & Asymmetric Release Hysteresis
# ---------------------------------------------------------------------------


def test_stage5_single_frame_hallucination_blocked_and_asymmetric_release() -> None:
    gate = AsymmetricSimplexVlmGate()

    # 1. Single-frame hallucinated RUNWAY_INCURSION -> DEBOUNCE_PENDING (blocked!)
    d1 = gate.evaluate(
        packet_or_bytes=_make_packet(seq=1, ts_ms=5000, hazard=int(VlmHazard.RUNWAY_INCURSION)),
        now_ms=5050,
        optical_odd=_nominal_odd(),
    )
    assert d1.status == GateStatus.DEBOUNCE_PENDING
    assert d1.stage_reached == 5
    assert d1.accepted_vlm_vetoes == frozenset()
    assert d1.final_hazards == frozenset()

    # 2. Followed by NONE -> still zero VLM vetoes
    d2 = gate.evaluate(
        packet_or_bytes=_make_packet(seq=2, ts_ms=5200, hazard=int(VlmHazard.NONE)),
        now_ms=5250,
        optical_odd=_nominal_odd(),
    )
    assert d2.status == GateStatus.NOMINAL_ADVISORY
    assert d2.accepted_vlm_vetoes == frozenset()

    # 3. Second hit within 3-frame window (2-of-3!) -> RUNWAY_INCURSION confirmed
    d3 = gate.evaluate(
        packet_or_bytes=_make_packet(seq=3, ts_ms=5400, hazard=int(VlmHazard.RUNWAY_INCURSION)),
        now_ms=5450,
        optical_odd=_nominal_odd(),
    )
    assert d3.status == GateStatus.NOMINAL_ADVISORY
    assert d3.accepted_vlm_vetoes == frozenset({"VLM:RUNWAY_INCURSION"})

    # 4. Asymmetric release requires 4 consecutive clear frames; flicker at frame 3 resets streak!
    for k in range(3):
        ts = 5600 + k * 200
        dc = gate.evaluate(
            packet_or_bytes=_make_packet(seq=4 + k, ts_ms=ts, hazard=int(VlmHazard.NONE)),
            now_ms=ts + 40,
            optical_odd=_nominal_odd(),
        )
        assert dc.accepted_vlm_vetoes == frozenset({"VLM:RUNWAY_INCURSION"})

    # Flicker hit resets clear streak to 0
    df = gate.evaluate(
        packet_or_bytes=_make_packet(seq=7, ts_ms=6200, hazard=int(VlmHazard.RUNWAY_INCURSION)),
        now_ms=6240,
        optical_odd=_nominal_odd(),
    )
    assert df.accepted_vlm_vetoes == frozenset({"VLM:RUNWAY_INCURSION"})

    # Now 3 clear frames still hold latch; 4th consecutive clear releases cleanly
    for k in range(3):
        ts = 6400 + k * 200
        dc = gate.evaluate(
            packet_or_bytes=_make_packet(seq=8 + k, ts_ms=ts, hazard=int(VlmHazard.NONE)),
            now_ms=ts + 40,
            optical_odd=_nominal_odd(),
        )
        assert dc.accepted_vlm_vetoes == frozenset({"VLM:RUNWAY_INCURSION"})

    d_rel = gate.evaluate(
        packet_or_bytes=_make_packet(seq=11, ts_ms=7000, hazard=int(VlmHazard.NONE)),
        now_ms=7040,
        optical_odd=_nominal_odd(),
    )
    assert d_rel.status == GateStatus.NOMINAL_ADVISORY
    assert d_rel.accepted_vlm_vetoes == frozenset()


def test_stage5_crane_proxy_requires_3_of_5_confirmation() -> None:
    gate = AsymmetricSimplexVlmGate()
    seq = 1
    ts = 8000
    pattern = [
        VlmHazard.CRANE_PROXY,  # 1/3 -> pending
        VlmHazard.NONE,         # 1/3
        VlmHazard.CRANE_PROXY,  # 2/3 -> pending
        VlmHazard.NONE,         # 2/3
        VlmHazard.CRANE_PROXY,  # 3/5 -> CONFIRMED!
    ]
    decisions = []
    for haz in pattern:
        dec = gate.evaluate(
            packet_or_bytes=_make_packet(seq=seq, ts_ms=ts, hazard=int(haz)),
            now_ms=ts + 30,
            optical_odd=_nominal_odd(),
        )
        decisions.append(dec)
        seq += 1
        ts += 200

    assert decisions[0].status == GateStatus.DEBOUNCE_PENDING
    assert decisions[2].status == GateStatus.DEBOUNCE_PENDING
    assert decisions[2].accepted_vlm_vetoes == frozenset()
    assert decisions[4].status == GateStatus.NOMINAL_ADVISORY
    assert decisions[4].accepted_vlm_vetoes == frozenset({"VLM:CRANE_PROXY"})


# ---------------------------------------------------------------------------
# Stage 6: Kinematic Feasibility & Relative-Only FLZ Ranking
# ---------------------------------------------------------------------------


def test_stage6_relative_only_flz_ranking_drops_unreachable_and_ignores_scores() -> None:
    # Config explicitly forbids absolute scalar VLM safety scores
    with pytest.raises(ValueError, match="forbids absolute scalar"):
        VlmGateConfig(allow_absolute_flz_scores=True)

    gate = AsymmetricSimplexVlmGate()
    # Pre-verified kinematically reachable flat candidates: (1, 2, 3, 4); zone 2 is geometrically hazardous
    # VLM ranks: (99 [unreachable!], 2 [hazardous!], 4, 1) -> Stage 6 must yield (4, 1, 3)
    d1 = gate.evaluate(
        packet_or_bytes=_make_packet(
            seq=1,
            ts_ms=9000,
            hazard=int(VlmHazard.NONE),
            ranked_zone_ids=(99, 2, 4, 1),
        ),
        now_ms=9040,
        optical_odd=_nominal_odd(),
        candidate_flz_ids=(1, 2, 3, 4),
        hazardous_flz_ids=frozenset({2}),
        raw_vlm_absolute_scores={3: 0.999, 1: 0.95, 4: 0.01},
    )
    assert d1.status == GateStatus.NOMINAL_ADVISORY
    assert d1.stage_reached == 6
    assert d1.ranked_flz_ids == (4, 1, 3)
    assert d1.absolute_scores_ignored is True
    assert 99 not in d1.ranked_flz_ids
    assert 2 not in d1.ranked_flz_ids

    # Flipping absolute scores completely produces identical relative ranking
    d2 = gate.evaluate(
        packet_or_bytes=_make_packet(
            seq=2,
            ts_ms=9200,
            hazard=int(VlmHazard.NONE),
            ranked_zone_ids=(99, 2, 4, 1),
        ),
        now_ms=9240,
        optical_odd=_nominal_odd(),
        candidate_flz_ids=(1, 2, 3, 4),
        hazardous_flz_ids=frozenset({2}),
        raw_vlm_absolute_scores={3: 0.001, 1: 0.05, 4: 0.99},
    )
    assert d2.ranked_flz_ids == d1.ranked_flz_ids == (4, 1, 3)
    assert d2.absolute_scores_ignored is True
