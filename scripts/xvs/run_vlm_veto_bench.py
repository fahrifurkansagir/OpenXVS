#!/usr/bin/env python3
"""DR-09 Edge VLM Semantic Safety Monitor & 6-Stage Veto Benchmark (`OpenXVS v5.0`).

Executes:
  1. Bit-exact 32-byte IEEE-802.3 CRC32 wire parity check against frozen Rust `xvs-odd-vlm`.
  2. Live zero-ML 5-mode `xvs_optical_odd` + 6-stage `xvs_vlm_gate` + MAVLink 2.0
     `#50003 POD_VLM_SEMANTIC_VETO` wire round-trip benchmark across real
     `data/studio_media/{runway_land,thermal_dve,eo_sky}` BGR (+ paired LWIR) frames.
  3. All 6 mandatory DR-09 fault-injection proofs:
       - FI-1: Stale watchdog (> 200 ms) & missing packet -> Stage 1 `VLM_OFFLINE`
       - FI-2: Corrupted IEEE-CRC32 wire bit flip -> Stage 2 `SCHEMA_OR_CRC_FAIL`
       - FI-3: Real `thermal_dve` fog frame Optical-ODD mask -> Stage 3 `OPTICAL_ODD_MASKED`
       - FI-4: `conformal_valid_bit=0` abstain -> Stage 4 `CONFORMAL_ABSTAIN` (never clears latch)
       - FI-5: Forbidden `Gemma-3` (+ PaliGemma-2 / Qwen2.5-VL-3B / Cosmos-Reason2 / GPL / NC) license ban
       - FI-6: `actuator_authority=False` & pessimistic-union non-relaxation + absolute score ban

Honest provenance:
  - By default uses `DeterministicOfflineCalibrationBackend`, explicitly stamped
    `OFFLINE_CONFORMAL_BENCH_STUB` (`is_neural_vlm=false`) — NEVER pretends to be a neural VLM.
  - Passing `--vlm-url http://127.0.0.1:<port>/v1` attaches `OpenAICompatibleVlmBackend`
    (with strict `< 200 ms` fail-closed timeout and `enable_thinking=False`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MSGID_POD_VLM_SEMANTIC_VETO,
    PodVlmSemanticVeto,
    decode_frame,
)
from xvs.v2.xvs_vlm_gate import (
    GATE_STATUS_TO_MAVLINK_VERDICT,
    OFFLINE_STUB_BACKEND_LABEL,
    OFFLINE_STUB_PROVENANCE_NOTICE,
    PACKET_SIZE,
    UNUSED_ZONE_ID,
    DeterministicOfflineCalibrationBackend,
    EdgeVlmSafetyMonitorRunner,
    FrozenConformalCalibration,
    GateStatus,
    OfflineStubProposal,
    OpenAICompatibleVlmBackend,
    VlmAdvisoryPacket,
    VlmGateConfig,
    VlmHazard,
    VlmRunnerConfig,
    studio_media_optical_odd_config,
    studio_media_vlm_gate_config,
    validate_vlm_model_license,
    vlm_hazard_bit,
)

DEFAULT_MEDIA_DIR = REPO_ROOT / "data" / "studio_media"
DEFAULT_OUT_PATH = REPO_ROOT / "outputs" / "vlm_veto_bench" / "vlm_veto_bench_evidence.json"

# Exact golden 32-byte vector from frozen Rust `xvs-odd-vlm/src/lib.rs`
RUST_XVS_ODD_VLM_GOLDEN_BYTES = bytes(
    [
        88, 86, 1, 0, 42, 0, 0, 0, 123, 104, 229, 207, 139, 1, 0, 0,
        1, 11, 7, 2, 5, 255, 1, 0, 1, 0, 0, 0, 173, 124, 70, 216,
    ]
)
RUST_XVS_ODD_VLM_GOLDEN_CRC32 = 0xD8467CAD


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Failed to read real studio_media frame: {path}")
    return img


def _verify_mavlink_50003_roundtrip(
    frame_bytes: bytes,
    expected_msg: PodVlmSemanticVeto,
) -> bool:
    decoded = decode_frame(frame_bytes)
    if decoded.msgid != MSGID_POD_VLM_SEMANTIC_VETO:
        return False
    if not isinstance(decoded.message, PodVlmSemanticVeto):
        return False
    return decoded.message == expected_msg


def _verify_rust_wire_parity() -> dict[str, Any]:
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
    packed = pkt.pack()
    unpacked = VlmAdvisoryPacket.unpack(packed)
    bit_exact = (
        len(packed) == PACKET_SIZE
        and packed == RUST_XVS_ODD_VLM_GOLDEN_BYTES
        and pkt.computed_crc32() == RUST_XVS_ODD_VLM_GOLDEN_CRC32
        and unpacked.ok
        and unpacked.packet is not None
        and unpacked.packet.crc32 == RUST_XVS_ODD_VLM_GOLDEN_CRC32
    )
    return {
        "packet_size_bytes": len(packed),
        "golden_crc32_hex": f"0x{pkt.computed_crc32():08x}",
        "bit_exact_match": bool(bit_exact),
    }


def _make_backend(
    *,
    model_id: str,
    vlm_url: str | None,
    timeout_ms: int,
    scripted: list[OfflineStubProposal] | None = None,
) -> OpenAICompatibleVlmBackend | DeterministicOfflineCalibrationBackend:
    if vlm_url:
        return OpenAICompatibleVlmBackend(
            model_id=model_id,
            base_url=vlm_url,
            timeout_ms=timeout_ms,
            enable_thinking=False,
        )
    return DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        enable_thinking=False,
        scripted_proposals=scripted,
    )


def _run_channel_benchmark(
    *,
    channel: str,
    media_dir: Path,
    manifest: dict[str, Any],
    model_id: str,
    vlm_url: str | None,
    timeout_ms: int,
) -> tuple[dict[str, Any], list[float], set[int]]:
    frames_meta: list[dict[str, Any]] = manifest["channels"][channel]["frames"]
    horizon_visible = channel == "thermal_dve"
    odd_cfg = studio_media_optical_odd_config(horizon_visible=horizon_visible, fast_mltl=True)
    gate_cfg = studio_media_vlm_gate_config(max_age_ms=200)
    backend = _make_backend(model_id=model_id, vlm_url=vlm_url, timeout_ms=timeout_ms)
    runner = EdgeVlmSafetyMonitorRunner(
        backend=backend,
        runner_config=VlmRunnerConfig(),
        gate_config=gate_cfg,
        optical_odd_config=odd_cfg,
    )

    cert_hazards_by_channel = {
        "runway_land": (frozenset({"TAWS:GLIDESLOPE_MONITOR"}), 0b0010_0000),
        "thermal_dve": (frozenset({"LWIR:THERMAL_TRACK_01"}), 0b0100_0000),
        "eo_sky": (frozenset({"DAIDALUS:CBDR_WATCH"}), 0b1000_0000),
    }
    cert_set, cert_mask = cert_hazards_by_channel[channel]
    reachable_mask = 0b0000_0000_0000_1111  # Zones 0, 1, 2, 3 reachable
    hazardous_flz = frozenset({1})  # Zone 1 geometrically hazardous -> excluded by Stage 6

    status_counts: dict[str, int] = {}
    odd_action_counts: dict[str, int] = {}
    latencies_ms: list[float] = []
    stages_seen: set[int] = set()
    mav_ok_count = 0
    wire_ok_count = 0
    af447_ok_all = True
    actuator_false_all = True
    pessimistic_union_all = True
    fog_masked_frames = 0
    confirmed_veto_frames = 0

    base_ts_ms = 10_000
    for idx, meta in enumerate(frames_meta):
        capture_ts = base_ts_ms + idx * 200
        now_ts = capture_ts + 45

        if channel == "thermal_dve":
            eo_bgr = _load_bgr(media_dir / meta["eo_file"])
            lwir_bgr = _load_bgr(media_dir / meta["lwir_file"])
            rvr_km = float(meta["rvr_km"])
            horizon_frac = 0.50
        elif channel == "runway_land":
            eo_bgr = _load_bgr(media_dir / meta["file"])
            lwir_bgr = None
            rvr_km = 12.0
            horizon_frac = float(meta["runway_bbox_px"][1]) / float(meta["height"])
        else:
            eo_bgr = _load_bgr(media_dir / meta["file"])
            lwir_bgr = None
            rvr_km = 12.0
            horizon_frac = 0.50

        res = runner.process_frame(
            eo_bgr=eo_bgr,
            lwir_bgr=lwir_bgr,
            capture_ts_ms=capture_ts,
            now_ms=now_ts,
            reachable_flz_mask=reachable_mask,
            certified_geometric_hazards=cert_set,
            cert_hazards_mask=cert_mask,
            hazardous_flz_ids=hazardous_flz,
            scenario_hint=channel,
            rvr_km=rvr_km,
            horizon_row_frac=horizon_frac,
        )

        st_name = res.gate_decision.status.value
        status_counts[st_name] = (status_counts[st_name] if st_name in status_counts else 0) + 1
        act_name = res.optical_odd_decision.action.value
        odd_action_counts[act_name] = (
            odd_action_counts[act_name] if act_name in odd_action_counts else 0
        ) + 1
        latencies_ms.append(res.pipeline_latency_ms)
        stages_seen.add(res.gate_decision.stage_reached)

        if _verify_mavlink_50003_roundtrip(res.mavlink_frame_bytes, res.mavlink_message):
            mav_ok_count += 1
        if res.wire_packet_bytes is not None:
            unpacked = VlmAdvisoryPacket.unpack(res.wire_packet_bytes)
            if unpacked.ok and len(res.wire_packet_bytes) == PACKET_SIZE:
                wire_ok_count += 1

        if not res.optical_odd_decision.airdata_export_blocked:
            af447_ok_all = False
        if res.actuator_authority or res.gate_decision.actuator_authority:
            actuator_false_all = False
        if not cert_set.issubset(res.gate_decision.final_hazards):
            pessimistic_union_all = False
        if (res.final_hazards_mask & cert_mask) != cert_mask:
            pessimistic_union_all = False
        if res.gate_decision.status == GateStatus.OPTICAL_ODD_MASKED:
            fog_masked_frames += 1
        if res.gate_decision.accepted_vlm_vetoes:
            confirmed_veto_frames += 1

    channel_summary = {
        "channel": channel,
        "frames_evaluated": len(frames_meta),
        "backend_label": backend.backend_label,
        "is_neural_vlm": backend.is_neural_vlm,
        "gate_status_counts": status_counts,
        "optical_odd_action_counts": odd_action_counts,
        "stages_reached": sorted(stages_seen),
        "confirmed_veto_frames": confirmed_veto_frames,
        "optical_odd_masked_frames": fog_masked_frames,
        "wire_32b_packets_verified": wire_ok_count,
        "mavlink_50003_roundtrip_verified": mav_ok_count,
        "af447_airdata_export_blocked_all": af447_ok_all,
        "actuator_authority_false_all": actuator_false_all,
        "pessimistic_union_preserved_all": pessimistic_union_all,
    }
    return channel_summary, latencies_ms, stages_seen


def _run_fault_injections(
    *,
    media_dir: Path,
    manifest: dict[str, Any],
    model_id: str,
) -> tuple[dict[str, Any], set[int], int]:
    """Execute all 6 mandatory DR-09 fault-injection proofs deterministically."""
    stages_seen: set[int] = set()
    mav_verified = 0

    runway_meta0 = manifest["channels"]["runway_land"]["frames"][0]
    runway_f0 = _load_bgr(media_dir / runway_meta0["file"])
    runway_hor_frac = float(runway_meta0["runway_bbox_px"][1]) / float(runway_meta0["height"])
    fog_meta = manifest["channels"]["thermal_dve"]["frames"][10]
    fog_eo = _load_bgr(media_dir / fog_meta["eo_file"])
    fog_lwir = _load_bgr(media_dir / fog_meta["lwir_file"])
    fog_rvr = float(fog_meta["rvr_km"])

    cert_set = frozenset({"ADSB:N12345", "TAWS:FLTA_WARNING"})
    cert_mask = 0b0110_0000
    reachable_mask = 0b0000_0000_0000_1111

    # --- FI-1: Stale Watchdog (> 200 ms) ---
    be1 = DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        scripted_proposals=[
            OfflineStubProposal(hazard=VlmHazard.RUNWAY_INCURSION, azimuth_sector=2, ranked_zone_ids=(2, 0))
        ],
    )
    runner1 = EdgeVlmSafetyMonitorRunner(
        backend=be1,
        optical_odd_config=studio_media_optical_odd_config(horizon_visible=False),
        gate_config=studio_media_vlm_gate_config(),
    )
    res1 = runner1.process_frame(
        eo_bgr=runway_f0,
        capture_ts_ms=20_000,
        now_ms=20_250,  # 250 ms > 200 ms watchdog limit
        reachable_flz_mask=reachable_mask,
        certified_geometric_hazards=cert_set,
        cert_hazards_mask=cert_mask,
        rvr_km=12.0,
    )
    mav1_ok = _verify_mavlink_50003_roundtrip(res1.mavlink_frame_bytes, res1.mavlink_message)
    if mav1_ok:
        mav_verified += 1
    stages_seen.add(res1.gate_decision.stage_reached)
    fi1_pass = (
        res1.gate_decision.status == GateStatus.VLM_OFFLINE
        and res1.gate_decision.stage_reached == 1
        and res1.gate_decision.vlm_online is False
        and res1.gate_decision.accepted_vlm_vetoes == frozenset()
        and res1.gate_decision.final_hazards == cert_set
        and res1.actuator_authority is False
        and res1.mavlink_message.gate_verdict == GATE_STATUS_TO_MAVLINK_VERDICT[GateStatus.VLM_OFFLINE]
        and mav1_ok
    )

    # --- FI-2: Corrupted IEEE-CRC32 Wire Bit Flip ---
    be2 = DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        scripted_proposals=[
            OfflineStubProposal(hazard=VlmHazard.CRANE_PROXY, azimuth_sector=4, ranked_zone_ids=(3, 1))
        ],
    )
    runner2 = EdgeVlmSafetyMonitorRunner(
        backend=be2,
        optical_odd_config=studio_media_optical_odd_config(horizon_visible=False),
        gate_config=studio_media_vlm_gate_config(),
    )
    res2 = runner2.process_frame(
        eo_bgr=runway_f0,
        capture_ts_ms=30_000,
        now_ms=30_040,
        reachable_flz_mask=reachable_mask,
        certified_geometric_hazards=cert_set,
        cert_hazards_mask=cert_mask,
        rvr_km=12.0,
        corrupt_wire_byte_index=10,
    )
    mav2_ok = _verify_mavlink_50003_roundtrip(res2.mavlink_frame_bytes, res2.mavlink_message)
    if mav2_ok:
        mav_verified += 1
    stages_seen.add(res2.gate_decision.stage_reached)
    fi2_pass = (
        res2.gate_decision.status == GateStatus.SCHEMA_OR_CRC_FAIL
        and res2.gate_decision.stage_reached == 2
        and res2.gate_decision.vlm_used is False
        and res2.gate_decision.final_hazards == cert_set
        and res2.actuator_authority is False
        and res2.mavlink_message.gate_verdict == GATE_STATUS_TO_MAVLINK_VERDICT[GateStatus.SCHEMA_OR_CRC_FAIL]
        and mav2_ok
    )

    # --- FI-3: Optical-ODD Fog-Masked Abstain (Real thermal_dve radiation_fog frame) ---
    be3 = DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        scripted_proposals=[
            OfflineStubProposal(hazard=VlmHazard.CRANE_PROXY, azimuth_sector=5, ranked_zone_ids=(2, 0))
            for _ in range(3)
        ],
    )
    runner3 = EdgeVlmSafetyMonitorRunner(
        backend=be3,
        optical_odd_config=studio_media_optical_odd_config(horizon_visible=True, fast_mltl=True),
        gate_config=studio_media_vlm_gate_config(),
    )
    res3 = None
    for k in range(3):
        ts = 40_000 + k * 200
        res3 = runner3.process_frame(
            eo_bgr=fog_eo,
            lwir_bgr=fog_lwir,
            capture_ts_ms=ts,
            now_ms=ts + 35,
            reachable_flz_mask=reachable_mask,
            certified_geometric_hazards=cert_set,
            cert_hazards_mask=cert_mask,
            rvr_km=fog_rvr,
        )
    assert res3 is not None
    mav3_ok = _verify_mavlink_50003_roundtrip(res3.mavlink_frame_bytes, res3.mavlink_message)
    if mav3_ok:
        mav_verified += 1
    stages_seen.add(res3.gate_decision.stage_reached)
    fi3_pass = (
        res3.gate_decision.status == GateStatus.OPTICAL_ODD_MASKED
        and res3.gate_decision.stage_reached == 3
        and res3.gate_decision.vlm_used is False
        and res3.optical_odd_decision.airdata_export_blocked is True
        and res3.optical_odd_decision.metrics.estimated_mor_m < 1609.0
        and res3.gate_decision.final_hazards == cert_set
        and res3.actuator_authority is False
        and res3.mavlink_message.gate_verdict == GATE_STATUS_TO_MAVLINK_VERDICT[GateStatus.OPTICAL_ODD_MASKED]
        and mav3_ok
    )

    # --- FI-4: Conformal Abstention (conformal_valid_bit == 0 NEVER clears latched hazard) ---
    be4 = DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        scripted_proposals=[
            OfflineStubProposal(hazard=VlmHazard.RUNWAY_INCURSION, azimuth_sector=1, p_image=0.95, token_entropy=0.10, ood_distance=0.05),
            OfflineStubProposal(hazard=VlmHazard.RUNWAY_INCURSION, azimuth_sector=1, p_image=0.95, token_entropy=0.10, ood_distance=0.05),
            # Low grounding / high epistemic OOD -> conformal_valid_bit=0 reporting NONE
            OfflineStubProposal(hazard=VlmHazard.NONE, azimuth_sector=1, p_image=0.52, token_entropy=0.20, ood_distance=0.72),
        ],
    )
    runner4 = EdgeVlmSafetyMonitorRunner(
        backend=be4,
        optical_odd_config=studio_media_optical_odd_config(horizon_visible=False),
        gate_config=studio_media_vlm_gate_config(),
    )
    r4_steps = [
        runner4.process_frame(
            eo_bgr=runway_f0,
            capture_ts_ms=50_000 + k * 200,
            now_ms=50_040 + k * 200,
            reachable_flz_mask=reachable_mask,
            certified_geometric_hazards=cert_set,
            cert_hazards_mask=cert_mask,
            rvr_km=12.0,
            horizon_row_frac=runway_hor_frac,
        )
        for k in range(3)
    ]
    res4 = r4_steps[-1]
    mav4_ok = _verify_mavlink_50003_roundtrip(res4.mavlink_frame_bytes, res4.mavlink_message)
    if mav4_ok:
        mav_verified += 1
    stages_seen.add(res4.gate_decision.stage_reached)
    fi4_pass = (
        r4_steps[0].gate_decision.status == GateStatus.DEBOUNCE_PENDING
        and r4_steps[1].gate_decision.status == GateStatus.NOMINAL_ADVISORY
        and "VLM:RUNWAY_INCURSION" in r4_steps[1].gate_decision.accepted_vlm_vetoes
        and res4.gate_decision.status == GateStatus.CONFORMAL_ABSTAIN
        and res4.gate_decision.stage_reached == 4
        and res4.advisory_packet is not None
        and res4.advisory_packet.conformal_valid_bit == 0
        and res4.advisory_packet.epistemic_flag == 1
        and "VLM:RUNWAY_INCURSION" in res4.gate_decision.accepted_vlm_vetoes
        and res4.final_hazards_mask == (cert_mask | vlm_hazard_bit(VlmHazard.RUNWAY_INCURSION))
        and res4.actuator_authority is False
        and res4.mavlink_message.gate_verdict == GATE_STATUS_TO_MAVLINK_VERDICT[GateStatus.CONFORMAL_ABSTAIN]
        and mav4_ok
    )

    # --- FI-5: Forbidden Gemma-3 (+ PaliGemma-2 / Qwen2.5-VL-3B / Cosmos-Reason2 / GPL / NC) Ban ---
    banned_probes = [
        ("google/Gemma-3-4B-it", None),
        ("google/PaliGemma-2-3B", None),
        ("Qwen/Qwen2.5-VL-3B-Instruct", None),
        ("nvidia/Cosmos-Reason2-8B", None),
        ("Qwen3-VL-2B-Instruct", "GPL-3.0"),
        ("Qwen3-VL-2B-Instruct", "AGPL-3.0"),
        ("Qwen3-VL-2B-Instruct", "CC-BY-NC-4.0"),
    ]
    rejected_all = True
    rejected_details: list[str] = []
    for bad_model, bad_lic in banned_probes:
        try:
            validate_vlm_model_license(bad_model, bad_lic)
            rejected_all = False
        except ValueError as exc:
            rejected_details.append(f"{bad_model}[{bad_lic or 'auto'}]: {exc}")

    # --- FI-6: Actuator Authority & Non-Relaxation Invariant ---
    locks_enforced = True
    for bad_factory in (
        lambda: VlmRunnerConfig(actuator_authority=True),
        lambda: VlmRunnerConfig(enable_thinking=True),
        lambda: VlmGateConfig(allow_absolute_flz_scores=True),
        lambda: FrozenConformalCalibration(weights_frozen=False),
        lambda: FrozenConformalCalibration(online_weight_updates_allowed=True),
    ):
        try:
            bad_factory()
            locks_enforced = False
        except ValueError:
            pass

    be6 = DeterministicOfflineCalibrationBackend(
        model_id=model_id,
        scripted_proposals=[
            OfflineStubProposal(hazard=VlmHazard.NONE, azimuth_sector=0, ranked_zone_ids=(3, 2, 0))
        ],
    )
    runner6 = EdgeVlmSafetyMonitorRunner(
        backend=be6,
        optical_odd_config=studio_media_optical_odd_config(horizon_visible=False),
        gate_config=studio_media_vlm_gate_config(),
    )
    res6 = runner6.process_frame(
        eo_bgr=runway_f0,
        capture_ts_ms=60_000,
        now_ms=60_030,
        reachable_flz_mask=reachable_mask,
        certified_geometric_hazards=cert_set,
        cert_hazards_mask=cert_mask,
        hazardous_flz_ids=frozenset({2}),
        raw_vlm_absolute_scores={0: 0.99, 2: 0.98, 3: 0.01},
        rvr_km=12.0,
        horizon_row_frac=runway_hor_frac,
    )
    mav6_ok = _verify_mavlink_50003_roundtrip(res6.mavlink_frame_bytes, res6.mavlink_message)
    if mav6_ok:
        mav_verified += 1
    stages_seen.add(res6.gate_decision.stage_reached)
    fi6_pass = (
        locks_enforced
        and res6.actuator_authority is False
        and res6.gate_decision.actuator_authority is False
        and cert_set.issubset(res6.gate_decision.final_hazards)
        and (res6.final_hazards_mask & cert_mask) == cert_mask
        and res6.gate_decision.absolute_scores_ignored is True
        and 2 not in res6.gate_decision.ranked_flz_ids
        and res6.gate_decision.ranked_flz_ids == (3, 0, 1)
        and mav6_ok
    )

    fault_report = {
        "fi_1_stale_watchdog_gt_200ms": {
            "pass": bool(fi1_pass),
            "status": res1.gate_decision.status.value,
            "stage_reached": res1.gate_decision.stage_reached,
            "actuator_authority": res1.actuator_authority,
            "mavlink_verdict": res1.mavlink_message.gate_verdict,
        },
        "fi_2_corrupted_crc32": {
            "pass": bool(fi2_pass),
            "status": res2.gate_decision.status.value,
            "stage_reached": res2.gate_decision.stage_reached,
            "actuator_authority": res2.actuator_authority,
            "mavlink_verdict": res2.mavlink_message.gate_verdict,
        },
        "fi_3_optical_odd_fog_masked_abstain": {
            "pass": bool(fi3_pass),
            "status": res3.gate_decision.status.value,
            "stage_reached": res3.gate_decision.stage_reached,
            "optical_odd_action": res3.optical_odd_decision.action.value,
            "estimated_mor_m": round(res3.optical_odd_decision.metrics.estimated_mor_m, 2),
            "airdata_export_blocked": res3.optical_odd_decision.airdata_export_blocked,
            "actuator_authority": res3.actuator_authority,
            "mavlink_verdict": res3.mavlink_message.gate_verdict,
        },
        "fi_4_conformal_valid_zero_abstain": {
            "pass": bool(fi4_pass),
            "status": res4.gate_decision.status.value,
            "stage_reached": res4.gate_decision.stage_reached,
            "latched_veto_preserved": "VLM:RUNWAY_INCURSION" in res4.gate_decision.accepted_vlm_vetoes,
            "uncertainty_source": res4.gate_decision.uncertainty_source.value,
            "actuator_authority": res4.actuator_authority,
            "mavlink_verdict": res4.mavlink_message.gate_verdict,
        },
        "fi_5_forbidden_gemma3_license_rejection": {
            "pass": bool(rejected_all),
            "probes_rejected": len(rejected_details),
            "probes_total": len(banned_probes),
        },
        "fi_6_actuator_authority_non_relaxation_invariant": {
            "pass": bool(fi6_pass),
            "config_locks_enforced": bool(locks_enforced),
            "actuator_authority": res6.actuator_authority,
            "absolute_scores_ignored": res6.gate_decision.absolute_scores_ignored,
            "ranked_flz_ids": list(res6.gate_decision.ranked_flz_ids),
            "mavlink_verdict": res6.mavlink_message.gate_verdict,
        },
    }
    return fault_report, stages_seen, mav_verified


def run_benchmark(
    *,
    media_dir: Path = DEFAULT_MEDIA_DIR,
    model_id: str = "Qwen3-VL-2B-Instruct",
    vlm_url: str | None = None,
    timeout_ms: int = 180,
) -> dict[str, Any]:
    """Execute the full DR-09 6-stage VLM veto benchmark and return structured evidence dict."""
    spec = validate_vlm_model_license(model_id)
    manifest_path = media_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing studio_media manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rust_parity = _verify_rust_wire_parity()

    channels_out: dict[str, Any] = {}
    all_latencies: list[float] = []
    all_stages: set[int] = set()
    for ch in ("runway_land", "thermal_dve", "eo_sky"):
        ch_res, ch_lat, ch_stages = _run_channel_benchmark(
            channel=ch,
            media_dir=media_dir,
            manifest=manifest,
            model_id=model_id,
            vlm_url=vlm_url,
            timeout_ms=timeout_ms,
        )
        channels_out[ch] = ch_res
        all_latencies.extend(ch_lat)
        all_stages |= ch_stages

    fault_report, fi_stages, fi_mav_ok = _run_fault_injections(
        media_dir=media_dir,
        manifest=manifest,
        model_id=model_id,
    )
    all_stages |= fi_stages

    total_frames = sum(int(c["frames_evaluated"]) for c in channels_out.values())
    total_wire_ok = sum(int(c["wire_32b_packets_verified"]) for c in channels_out.values())
    total_mav_ok = sum(int(c["mavlink_50003_roundtrip_verified"]) for c in channels_out.values()) + fi_mav_ok
    total_mav_emitted = total_frames + 5  # 5 packet-emitting fault injections (FI-1..4, FI-6)
    af447_all = all(bool(c["af447_airdata_export_blocked_all"]) for c in channels_out.values())
    actuator_false_all = (
        all(bool(c["actuator_authority_false_all"]) for c in channels_out.values())
        and all(
            (not bool(v["actuator_authority"]))
            for v in fault_report.values()
            if "actuator_authority" in v
        )
    )
    pessimistic_all = all(bool(c["pessimistic_union_preserved_all"]) for c in channels_out.values())
    fi_passed = sum(1 for v in fault_report.values() if bool(v["pass"]))
    fi_total = len(fault_report)

    lat_arr = np.asarray(all_latencies, dtype=np.float64)
    p50 = round(float(np.percentile(lat_arr, 50.0)), 3) if lat_arr.size else 0.0
    p95 = round(float(np.percentile(lat_arr, 95.0)), 3) if lat_arr.size else 0.0
    max_lat = round(float(np.max(lat_arr)), 3) if lat_arr.size else 0.0

    overall_pass = (
        bool(rust_parity["bit_exact_match"])
        and sorted(all_stages) == [1, 2, 3, 4, 5, 6]
        and fi_passed == fi_total
        and af447_all
        and actuator_false_all
        and pessimistic_all
        and total_mav_ok == total_mav_emitted
        and channels_out["thermal_dve"]["optical_odd_masked_frames"] >= 10
        and channels_out["runway_land"]["confirmed_veto_frames"] >= 1
    )

    is_live_http = bool(vlm_url)
    evidence: dict[str, Any] = {
        "schema": "xvs.dr09_vlm_veto_bench/1",
        "scope": "strictly_civil_dual_use_aviation_safety",
        "coordinate_frame": "NED",
        "model_id": model_id,
        "spdx_license": spec.spdx_license,
        "enable_thinking": False,
        "max_output_tokens": 8,
        "weights_frozen_in_flight": True,
        "actuator_authority": False,
        "backend_label": "OPENAI_COMPATIBLE_LOCAL_HTTP" if is_live_http else OFFLINE_STUB_BACKEND_LABEL,
        "is_neural_vlm": is_live_http,
        "backend_provenance_notice": (
            f"Live local OpenAI-compatible VLM server at {vlm_url}"
            if is_live_http
            else OFFLINE_STUB_PROVENANCE_NOTICE
        ),
        "wire_parity_rust_xvs_odd_vlm": rust_parity,
        "real_media_channels": channels_out,
        "fault_injections": fault_report,
        "summary": {
            "total_frames_evaluated": total_frames,
            "stages_exercised": sorted(all_stages),
            "wire_32b_packets_verified": total_wire_ok,
            "mavlink_50003_frames_emitted": total_mav_emitted,
            "mavlink_50003_roundtrip_verified": total_mav_ok,
            "optical_odd_af447_airdata_blocked_all": af447_all,
            "actuator_authority_false_all": actuator_false_all,
            "pessimistic_union_preserved_all": pessimistic_all,
            "fault_injections_passed": fi_passed,
            "fault_injections_total": fi_total,
            "latency_ms": {"p50": p50, "p95": p95, "max": max_lat},
            "overall_pass": bool(overall_pass),
        },
    }
    canonical_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence["evidence_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenXVS DR-09 Edge VLM Semantic Safety Monitor & 6-Stage Veto Benchmark"
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=DEFAULT_MEDIA_DIR,
        help="Path to real data/studio_media directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output path for deterministic JSON evidence report",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen3-VL-2B-Instruct",
        help="Permitted Apache-2.0/MIT DR-09 model identifier",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default=None,
        help="Optional local loopback OpenAI-compatible VLM server URL (e.g. http://127.0.0.1:8080/v1)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=180,
        help="Strict <200 ms HTTP timeout when --vlm-url is supplied",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if overall_pass is False",
    )
    args = parser.parse_args(argv)

    evidence = run_benchmark(
        media_dir=args.media_dir,
        model_id=args.model_id,
        vlm_url=args.vlm_url,
        timeout_ms=args.timeout_ms,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    summary = evidence["summary"]
    print(
        f"[DR-09 VLM Veto Bench] overall_pass={summary['overall_pass']} "
        f"backend={evidence['backend_label']} (is_neural_vlm={evidence['is_neural_vlm']}) "
        f"frames={summary['total_frames_evaluated']} stages={summary['stages_exercised']} "
        f"fault_injections={summary['fault_injections_passed']}/{summary['fault_injections_total']} "
        f"mav50003={summary['mavlink_50003_roundtrip_verified']}/{summary['mavlink_50003_frames_emitted']} "
        f"p50={summary['latency_ms']['p50']}ms max={summary['latency_ms']['max']}ms "
        f"out={args.out}"
    )
    if args.check and not summary["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
