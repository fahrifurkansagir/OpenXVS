"""Verification & HIL Fault-Injection Suite for DR-09 Edge VLM Safety Monitor Runner & Veto Bench."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MSGID_POD_VLM_SEMANTIC_VETO,
    PodVlmSemanticVeto,
    decode_frame,
)
from xvs.v2.xvs_optical_odd import OpticalOddAction
from xvs.v2.xvs_vlm_gate import (
    GATE_STATUS_TO_MAVLINK_VERDICT,
    HTTP_BACKEND_LABEL,
    OFFLINE_STUB_BACKEND_LABEL,
    PACKET_SIZE,
    UNKNOWN_AZIMUTH_SECTOR,
    UNUSED_ZONE_ID,
    BufferedVlmFrame,
    DeterministicOfflineCalibrationBackend,
    EdgeVlmSafetyMonitorRunner,
    FrozenConformalBceaCalibrator,
    FrozenConformalCalibration,
    GateDecision,
    GateStatus,
    OfflineStubProposal,
    OpenAICompatibleVlmBackend,
    OpticalOddLiveBridge,
    SingleSlotDropStaleBuffer,
    StaticVlmTrieGrammar,
    UncertaintySource,
    VlmAdvisoryPacket,
    VlmGateConfig,
    VlmGrammarError,
    VlmHazard,
    VlmInferenceRequest,
    VlmRunnerConfig,
    build_pod_vlm_semantic_veto_frame,
    parse_vlm_constrained_output,
    studio_media_optical_odd_config,
    studio_media_vlm_gate_config,
    validate_vlm_model_license,
    vlm_hazard_bit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO_ROOT / "data" / "studio_media"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "xvs" / "run_vlm_veto_bench.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("run_vlm_veto_bench", BENCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_vlm_veto_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_studio_manifest() -> dict[str, Any]:
    return json.loads((MEDIA_DIR / "manifest.json").read_text(encoding="utf-8"))


def _synthetic_sharp_bgr(h: int = 240, w: int = 320) -> np.ndarray:
    yy, xx = np.indices((h, w), dtype=np.float32)
    img = np.zeros((h, w), dtype=np.float32)
    sky = yy < (h // 2)
    img[sky] = 175.0 + 2.0 * np.sin(xx[sky] * 0.25)
    terrain = ~sky
    img[terrain] = 90.0 + 35.0 * np.sin(xx[terrain] * 1.4) + 30.0 * np.cos(yy[terrain] * 1.6)
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 1. Strict DR-09 License Guard (Apache-2.0 / MIT Allowlist + Hard Bans)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected_canon",
    [
        ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
        ("Qwen/Qwen3-VL-2B-Instruct-NVFP4", "Qwen3-VL-2B-Instruct"),
        ("Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct"),
        ("Gemma-4-E2B-it", "Gemma-4-E2B-it"),
        ("google/Gemma-4-E2B-it-GGUF", "Gemma-4-E2B-it"),
        ("Gemma-4-E4B-it", "Gemma-4-E4B-it"),
        ("Qwen3.5-2B", "Qwen3.5-2B"),
        ("Qwen/Qwen3.5-2B-Instruct", "Qwen3.5-2B"),
        ("InternVL3", "InternVL3"),
        ("OpenGVLab/InternVL3-2B", "InternVL3"),
        ("SmolVLM2", "SmolVLM2"),
        ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "SmolVLM2"),
    ],
)
def test_dr09_license_allowlist_accepts_permissive_models(model_id: str, expected_canon: str) -> None:
    spec = validate_vlm_model_license(model_id, "Apache-2.0")
    assert spec.canonical_name == expected_canon
    assert spec.spdx_license in ("Apache-2.0", "MIT")
    # MIT license annotation also allowed
    spec_mit = validate_vlm_model_license(model_id, "MIT")
    assert spec_mit.canonical_name == expected_canon


@pytest.mark.parametrize(
    "bad_model,bad_license",
    [
        ("Gemma-3", None),
        ("google/gemma-3-4b-it", None),
        ("PaliGemma-2", None),
        ("google/paligemma-2-3b-pt-896", None),
        ("Qwen2.5-VL-3B", None),
        ("Qwen/Qwen2.5-VL-3B-Instruct", None),
        ("Cosmos-Reason2", None),
        ("nvidia/Cosmos-Reason2-8B", None),
        ("Qwen3-VL-2B-Thinking", None),
        ("SomeVLM-GPL", None),
        ("SomeVLM-AGPL", None),
        ("SomeVLM-NC", None),
        ("UnknownProprietaryVLM-7B", None),
        ("Qwen3-VL-2B-Instruct", "GPL-3.0"),
        ("Qwen3-VL-2B-Instruct", "AGPL-3.0"),
        ("Qwen3-VL-2B-Instruct", "CC-BY-NC-4.0"),
        ("Gemma-4-E2B-it", "Gemma-Terms-of-Use"),
        ("", None),
    ],
)
def test_dr09_license_guard_hard_rejects_banned_models_and_licenses(
    bad_model: str, bad_license: str | None
) -> None:
    with pytest.raises(ValueError):
        validate_vlm_model_license(bad_model, bad_license)
    with pytest.raises(ValueError):
        DeterministicOfflineCalibrationBackend(model_id=bad_model, license_spdx=bad_license)
    with pytest.raises(ValueError):
        OpenAICompatibleVlmBackend(model_id=bad_model, license_spdx=bad_license)


# ---------------------------------------------------------------------------
# 2. DR-09 Runtime Locks (enable_thinking=False, <=8 tokens, <200 ms, frozen)
# ---------------------------------------------------------------------------


def test_dr09_runtime_locks_enforced() -> None:
    with pytest.raises(ValueError, match="enable_thinking"):
        VlmRunnerConfig(enable_thinking=True)
    with pytest.raises(ValueError, match="actuator_authority"):
        VlmRunnerConfig(actuator_authority=True)
    with pytest.raises(ValueError, match="absolute"):
        VlmRunnerConfig(allow_absolute_flz_scores=True)
    with pytest.raises(ValueError, match="max_output_tokens"):
        VlmRunnerConfig(max_output_tokens=9)
    with pytest.raises(ValueError, match="max_age_ms"):
        VlmRunnerConfig(max_age_ms=250)

    with pytest.raises(ValueError, match="enable_thinking"):
        OpenAICompatibleVlmBackend(model_id="Qwen3-VL-2B-Instruct", enable_thinking=True)
    with pytest.raises(ValueError, match="timeout_ms"):
        OpenAICompatibleVlmBackend(model_id="Qwen3-VL-2B-Instruct", timeout_ms=200)
    with pytest.raises(ValueError, match="max_output_tokens"):
        OpenAICompatibleVlmBackend(model_id="Qwen3-VL-2B-Instruct", max_output_tokens=16)
    with pytest.raises(ValueError, match="loopback"):
        OpenAICompatibleVlmBackend(
            model_id="Qwen3-VL-2B-Instruct",
            base_url="https://api.example.com/v1",
        )

    with pytest.raises(ValueError, match="frozen"):
        FrozenConformalCalibration(weights_frozen=False)
    with pytest.raises(ValueError, match="online weight updates"):
        FrozenConformalCalibration(online_weight_updates_allowed=True)

    cal = FrozenConformalCalibration()
    with pytest.raises(FrozenInstanceError):
        cal.q_hat = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Static <= 8-Token JSON / Trie Grammar Parser
# ---------------------------------------------------------------------------


def test_static_trie_and_compact_json_parser_valid_cases() -> None:
    mask = 0b0000_0000_0000_1111  # zones 0, 1, 2, 3 reachable
    grammar = StaticVlmTrieGrammar(mask)
    assert grammar.allowed_zone_ids() == (0, 1, 2, 3)

    # Tagged Trie string (4 tokens <= 8)
    p1 = grammar.parse("H1 A3 Z2 Z0")
    assert p1.hazard == VlmHazard.RUNWAY_INCURSION
    assert p1.azimuth_sector == 3
    assert p1.ranked_zone_ids == (2, 0, UNUSED_ZONE_ID, UNUSED_ZONE_ID)
    assert p1.token_count == 4
    assert p1.wire_format == "trie"

    # Bare numeric Trie tuple (6 tokens <= 8)
    p2 = parse_vlm_constrained_output("4 11 3 1 0 2", mask)
    assert p2.hazard == VlmHazard.CRANE_PROXY
    assert p2.azimuth_sector == 11
    assert p2.ranked_zone_ids == (3, 1, 0, 2)
    assert p2.token_count == 6

    # Compact JSON (<= 8 tokens)
    p3 = grammar.parse('{"h":2,"a":7,"z":[1,3]}')
    assert p3.hazard == VlmHazard.BIRD_FLOCK
    assert p3.azimuth_sector == 7
    assert p3.ranked_zone_ids == (1, 3, UNUSED_ZONE_ID, UNUSED_ZONE_ID)
    assert p3.wire_format == "compact_json"

    # Canonical JSON field names with empty zones
    p4 = grammar.parse('{"hazard_enum":0,"azimuth_sector":0,"ranked_zone_ids":[]}')
    assert p4.hazard == VlmHazard.NONE
    assert p4.ranked_zone_ids == (UNUSED_ZONE_ID, UNUSED_ZONE_ID, UNUSED_ZONE_ID, UNUSED_ZONE_ID)

    # Round-trip formatters
    assert grammar.parse(grammar.format_trie(VlmHazard.FLZ_HAZARD, 5, (3, 0))).ranked_zone_ids == (
        3,
        0,
        UNUSED_ZONE_ID,
        UNUSED_ZONE_ID,
    )
    assert grammar.parse(grammar.format_compact_json(VlmHazard.CONVECTIVE_VIRGA, 9, (2,))).hazard == (
        VlmHazard.CONVECTIVE_VIRGA
    )


@pytest.mark.parametrize(
    "bad_output",
    [
        "",
        "There is a vehicle on runway 28L ahead.",
        "<think>analyzing runway</think>H1 A3 Z2",
        "H1 A3 Z2\nExplanation: ground tug detected",
        '{"h":1,"a":3,"z":[2,0],"scores":{"2":0.95}}',
        '{"h":1,"a":3,"z":[2,0],"safety_score":8.5}',
        '{"h":1,"a":3,"z":[2,0.85]}',
        "1 3 0.92 0.14",
        "H1 A3 Z2 Z2",  # duplicate zone ID
        "H1 A3 Z9",  # unreachable zone ID (outside 0b1111)
        "H6 A3 Z0",  # invalid VlmHazard (0..5 only)
        "H1 A12 Z0",  # invalid azimuth_sector (0..11 only)
        "H1 A3 Z0 Z1 Z2 Z3 Z0",  # > 4 zones
        "1 2 0 1 2 3 4 5 6",  # > 8 tokens
        '{"h":true,"a":3,"z":[0]}',  # boolean instead of int
        '{"h":1,"a":3,"z":[0]} trailing prose',
    ],
)
def test_static_grammar_rejects_prose_thinking_scores_and_overflow(bad_output: str) -> None:
    mask = 0b0000_0000_0000_1111
    with pytest.raises(VlmGrammarError):
        parse_vlm_constrained_output(bad_output, mask)


# ---------------------------------------------------------------------------
# 4. Single-Slot Drop-Stale Buffer & Frozen Conformal + BCEA Calibrator
# ---------------------------------------------------------------------------


def test_single_slot_drop_stale_buffer_never_queues_and_enforces_200ms() -> None:
    buf = SingleSlotDropStaleBuffer(max_age_ms=200)
    img = _synthetic_sharp_bgr(64, 64)

    assert buf.submit(BufferedVlmFrame(img, capture_ts_ms=1000, reachable_flz_mask=0x0F)) is False
    # Submitting a newer frame evicts the older unconsumed slot (zero queueing!)
    assert buf.submit(BufferedVlmFrame(img, capture_ts_ms=1100, reachable_flz_mask=0x0F)) is True
    assert buf.dropped_stale_count == 1

    fresh = buf.take_fresh(now_ms=1250)  # age = 150 ms <= 200 ms
    assert fresh is not None
    assert fresh.capture_ts_ms == 1100
    assert buf.take_fresh(now_ms=1260) is None

    # Stale (> 200 ms) is dropped fail-closed
    buf.submit(BufferedVlmFrame(img, capture_ts_ms=2000, reachable_flz_mask=0x0F))
    assert buf.take_fresh(now_ms=2201) is None
    assert buf.expired_stale_count == 1

    # Future capture timestamp is dropped fail-closed
    buf.submit(BufferedVlmFrame(img, capture_ts_ms=3500, reachable_flz_mask=0x0F))
    assert buf.take_fresh(now_ms=3400) is None
    assert buf.expired_stale_count == 2


def test_frozen_conformal_bcea_calibrator_separates_ra08_and_suppresses_priors() -> None:
    cal = FrozenConformalBceaCalibrator()

    # Well-grounded hazard passes conformal + BCEA
    ok = cal.calibrate(
        hazard=VlmHazard.RUNWAY_INCURSION,
        p_image=0.92,
        token_entropy=0.10,
        ood_distance=0.08,
    )
    assert ok.conformal_valid_bit == 1
    assert ok.aleatoric_flag == 0
    assert ok.epistemic_flag == 0
    assert ok.weights_frozen is True
    assert ok.bcea_visual_lift > 0.70

    # Language-prior hallucination (p_image too close to blank prior) -> epistemic abstain
    halluc = cal.calibrate(
        hazard=VlmHazard.RUNWAY_INCURSION,
        p_image=0.22,  # blank prior = 0.12 -> lift = 0.10 < 0.15
        token_entropy=0.10,
        ood_distance=0.10,
    )
    assert halluc.conformal_valid_bit == 0
    assert halluc.epistemic_flag == 1
    assert halluc.aleatoric_flag == 0

    # High aleatoric entropy -> aleatoric abstain
    ale = cal.calibrate(
        hazard=VlmHazard.BIRD_FLOCK,
        p_image=0.90,
        token_entropy=0.70,
        ood_distance=0.05,
    )
    assert ale.conformal_valid_bit == 0
    assert ale.aleatoric_flag == 1

    # Non-finite input fails closed
    nan_v = cal.calibrate(
        hazard=VlmHazard.CRANE_PROXY,
        p_image=math.nan,
        token_entropy=0.1,
        ood_distance=0.1,
    )
    assert nan_v.conformal_valid_bit == 0
    assert nan_v.aleatoric_flag == 1
    assert nan_v.epistemic_flag == 1


# ---------------------------------------------------------------------------
# 5. 32-Byte Rust `xvs-odd-vlm` Wire Parity & MAVLink `#50003` Round-Trip
# ---------------------------------------------------------------------------


def test_32b_ieee_crc32_rust_parity_and_mavlink_50003_all_verdicts() -> None:
    expected_rust_bytes = bytes(
        [
            88, 86, 1, 0, 42, 0, 0, 0, 123, 104, 229, 207, 139, 1, 0, 0,
            1, 11, 7, 2, 5, 255, 1, 0, 1, 0, 0, 0, 173, 124, 70, 216,
        ]
    )
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
    assert len(raw) == PACKET_SIZE == 32
    assert raw == expected_rust_bytes
    assert pkt.computed_crc32() == 0xD8467CAD

    # Round-trip MAVLink 2.0 #50003 across all 6 GateStatus values
    for status, expected_code in GATE_STATUS_TO_MAVLINK_VERDICT.items():
        dec = GateDecision(
            status=status,
            stage_reached=6 if status == GateStatus.NOMINAL_ADVISORY else 1,
            vlm_online=status != GateStatus.VLM_OFFLINE,
            vlm_used=status == GateStatus.NOMINAL_ADVISORY,
            actuator_authority=False,
            certified_geometric_hazards=frozenset(),
            accepted_vlm_vetoes=frozenset(),
            final_hazards=frozenset(),
            ranked_flz_ids=(2, 0),
            uncertainty_source=UncertaintySource.NONE,
            absolute_scores_ignored=False,
            detail="roundtrip",
        )
        msg, frame_bytes = build_pod_vlm_semantic_veto_frame(
            dec, seq=42, capture_ts_ms=123456, packet=pkt, mav_seq=expected_code
        )
        decoded = decode_frame(frame_bytes)
        assert decoded.msgid == MSGID_POD_VLM_SEMANTIC_VETO
        assert isinstance(decoded.message, PodVlmSemanticVeto)
        assert decoded.message == msg
        assert decoded.message.gate_verdict == expected_code
        assert decoded.message.ranked_flz_ids == (2, 0, UNUSED_ZONE_ID, UNUSED_ZONE_ID)
        if status in (GateStatus.VLM_OFFLINE, GateStatus.SCHEMA_OR_CRC_FAIL):
            assert decoded.message.azimuth_sector == UNKNOWN_AZIMUTH_SECTOR
            assert decoded.message.conformal_valid == 0


# ---------------------------------------------------------------------------
# 6. Live `OpticalOddReading` Integration on Real `data/studio_media/` Frames
# ---------------------------------------------------------------------------


def test_live_optical_odd_bridge_on_real_studio_media_frames() -> None:
    manifest = _load_studio_manifest()

    # 1. Real runway_land clear frame -> NOMINAL, AF447 locked, Stage 6 reachable
    r_meta = manifest["channels"]["runway_land"]["frames"][0]
    r_bgr = cv2.imread(str(MEDIA_DIR / r_meta["file"]), cv2.IMREAD_COLOR)
    assert r_bgr is not None
    r_hor = float(r_meta["runway_bbox_px"][1]) / float(r_meta["height"])
    bridge_r = OpticalOddLiveBridge(studio_media_optical_odd_config(horizon_visible=False))
    odd_dec_r, odd_read_r = bridge_r.evaluate_bgr(r_bgr, rvr_km=12.0, horizon_row_frac=r_hor)
    assert odd_dec_r.airdata_export_blocked is True
    assert odd_dec_r.action == OpticalOddAction.NOMINAL
    assert odd_read_r.fog_contrast >= 0.35

    # 2. Real thermal_dve radiation_fog paired EO+LWIR -> THERMAL_PRIMARY & Stage-3 VLM mask
    bridge_fog = OpticalOddLiveBridge(studio_media_optical_odd_config(horizon_visible=True, fast_mltl=True))
    fog_dec = None
    fog_read = None
    for idx in (6, 7, 8):
        f_meta = manifest["channels"]["thermal_dve"]["frames"][idx]
        eo_bgr = cv2.imread(str(MEDIA_DIR / f_meta["eo_file"]), cv2.IMREAD_COLOR)
        lw_bgr = cv2.imread(str(MEDIA_DIR / f_meta["lwir_file"]), cv2.IMREAD_COLOR)
        assert eo_bgr is not None and lw_bgr is not None
        fog_dec, fog_read = bridge_fog.evaluate_bgr(eo_bgr, lwir_bgr=lw_bgr, rvr_km=float(f_meta["rvr_km"]))
    assert fog_dec is not None and fog_read is not None
    assert fog_dec.airdata_export_blocked is True
    assert fog_dec.action == OpticalOddAction.THERMAL_PRIMARY
    assert fog_dec.metrics.estimated_mor_m < 1609.0
    assert fog_read.fog_contrast < 0.35

    # 3. Real eo_sky frame -> live OpticalOddReading & AF447 locked
    s_meta = manifest["channels"]["eo_sky"]["frames"][0]
    s_bgr = cv2.imread(str(MEDIA_DIR / s_meta["file"]), cv2.IMREAD_COLOR)
    assert s_bgr is not None
    bridge_s = OpticalOddLiveBridge(studio_media_optical_odd_config(horizon_visible=False))
    s_dec, s_read = bridge_s.evaluate_bgr(s_bgr, rvr_km=12.0)
    assert s_dec.airdata_export_blocked is True
    assert s_dec.action == OpticalOddAction.NOMINAL
    assert s_read.attitude_in_odd is True


# ---------------------------------------------------------------------------
# 7. Mock Local HTTP `OpenAICompatibleVlmBackend` (Nominal, Timeout, Prose Fault)
# ---------------------------------------------------------------------------


def test_openai_compatible_http_backend_nominal_timeout_fail_closed_and_prose_reject() -> None:
    captured_payloads: list[dict[str, Any]] = []
    server_mode: dict[str, Any] = {"mode": "nominal", "delay_s": 0.0}

    class _MockVlmHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"]) if "Content-Length" in self.headers else 0
            raw = self.rfile.read(length)
            captured_payloads.append(json.loads(raw.decode("utf-8")))
            delay = float(server_mode["delay_s"])
            if delay > 0.0:
                time.sleep(delay)

            mode = server_mode["mode"]
            if mode == "nominal":
                content = "H1 A4 Z2 Z0"
            else:
                content = "There is definitely a tug on the runway, safety_score=0.12"

            resp_obj = {
                "id": "chatcmpl-openxvs-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "logprobs": {
                            "content": [
                                {"token": "H1", "logprob": -0.05},
                                {"token": "A4", "logprob": -0.04},
                            ]
                        },
                    }
                ],
                "xvs_calibration": {
                    "p_image": 0.94,
                    "token_entropy": 0.09,
                    "ood_distance": 0.06,
                },
            }
            encoded = json.dumps(resp_obj).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockVlmHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}/v1"
        backend = OpenAICompatibleVlmBackend(
            model_id="Gemma-4-E2B-it",
            base_url=base_url,
            timeout_ms=120,  # strict < 200 ms
            enable_thinking=False,
        )
        assert backend.backend_label == HTTP_BACKEND_LABEL
        assert backend.is_neural_vlm is True

        runner = EdgeVlmSafetyMonitorRunner(
            backend=backend,
            optical_odd_config=studio_media_optical_odd_config(horizon_visible=False),
            gate_config=studio_media_vlm_gate_config(),
        )
        img = _synthetic_sharp_bgr()
        cert = frozenset({"TAWS:FLTA"})
        cert_mask = 0b0010_0000
        reach_mask = 0b0000_0000_0000_0111  # zones 0, 1, 2

        # 1. Two nominal HTTP calls confirm 2-of-3 RUNWAY_INCURSION at Stage 6
        r1 = runner.process_frame(
            eo_bgr=img,
            capture_ts_ms=10_000,
            now_ms=10_040,
            reachable_flz_mask=reach_mask,
            certified_geometric_hazards=cert,
            cert_hazards_mask=cert_mask,
            rvr_km=12.0,
        )
        r2 = runner.process_frame(
            eo_bgr=img,
            capture_ts_ms=10_200,
            now_ms=10_240,
            reachable_flz_mask=reach_mask,
            certified_geometric_hazards=cert,
            cert_hazards_mask=cert_mask,
            rvr_km=12.0,
        )
        assert r1.gate_decision.status == GateStatus.DEBOUNCE_PENDING
        assert r2.gate_decision.status == GateStatus.NOMINAL_ADVISORY
        assert r2.is_neural_vlm is True
        assert r2.actuator_authority is False
        assert "VLM:RUNWAY_INCURSION" in r2.gate_decision.accepted_vlm_vetoes
        assert r2.final_hazards_mask == (cert_mask | vlm_hazard_bit(VlmHazard.RUNWAY_INCURSION))
        assert r2.wire_packet_bytes is not None and len(r2.wire_packet_bytes) == PACKET_SIZE
        decoded_mav = decode_frame(r2.mavlink_frame_bytes)
        assert decoded_mav.msgid == MSGID_POD_VLM_SEMANTIC_VETO

        # Verify DR-09 payload locks actually arrived over HTTP
        assert len(captured_payloads) == 2
        payload0 = captured_payloads[0]
        assert payload0["enable_thinking"] is False
        assert payload0["chat_template_kwargs"]["enable_thinking"] is False
        assert payload0["max_tokens"] <= 8
        assert payload0["temperature"] == 0.0

        # 2. Prose / absolute-score HTTP response -> grammar rejected -> Stage 2 SCHEMA_OR_CRC_FAIL
        server_mode["mode"] = "prose"
        server_mode["delay_s"] = 0.0
        r_prose = runner.process_frame(
            eo_bgr=img,
            capture_ts_ms=10_400,
            now_ms=10_440,
            reachable_flz_mask=reach_mask,
            certified_geometric_hazards=cert,
            cert_hazards_mask=cert_mask,
            rvr_km=12.0,
        )
        assert r_prose.gate_decision.status == GateStatus.SCHEMA_OR_CRC_FAIL
        assert r_prose.gate_decision.stage_reached == 2
        assert r_prose.grammar_error is not None
        assert r_prose.actuator_authority is False
        assert cert.issubset(r_prose.gate_decision.final_hazards)

        # 3. Timeout (> 120 ms) fails closed -> Stage 1 VLM_OFFLINE, cert hazards intact
        server_mode["mode"] = "nominal"
        server_mode["delay_s"] = 0.35
        r_to = runner.process_frame(
            eo_bgr=img,
            capture_ts_ms=10_600,
            now_ms=10_640,
            reachable_flz_mask=reach_mask,
            certified_geometric_hazards=cert,
            cert_hazards_mask=cert_mask,
            rvr_km=12.0,
        )
        assert r_to.gate_decision.status == GateStatus.VLM_OFFLINE
        assert r_to.gate_decision.stage_reached == 1
        assert r_to.gate_decision.vlm_online is False
        assert r_to.gate_decision.final_hazards == cert
        assert r_to.actuator_authority is False
        assert r_to.mavlink_message.gate_verdict == GATE_STATUS_TO_MAVLINK_VERDICT[GateStatus.VLM_OFFLINE]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# 8. End-to-End `scripts/xvs/run_vlm_veto_bench.py` JSON Proof Verification
# ---------------------------------------------------------------------------


def test_run_vlm_veto_bench_script_produces_valid_6_stage_and_6_fault_proof(tmp_path: Path) -> None:
    bench = _load_bench_module()
    out_path = tmp_path / "vlm_veto_bench_evidence.json"
    rc = bench.main(["--media-dir", str(MEDIA_DIR), "--out", str(out_path), "--check"])
    assert rc == 0
    assert out_path.is_file()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema"] == "xvs.dr09_vlm_veto_bench/1"
    assert data["coordinate_frame"] == "NED"
    assert data["enable_thinking"] is False
    assert data["weights_frozen_in_flight"] is True
    assert data["actuator_authority"] is False
    assert data["backend_label"] == OFFLINE_STUB_BACKEND_LABEL
    assert data["is_neural_vlm"] is False
    assert data["wire_parity_rust_xvs_odd_vlm"]["bit_exact_match"] is True

    # Verify SHA-256 self-consistency
    recorded_sha = data["evidence_sha256"]
    body_without_sha = {k: v for k, v in data.items() if k != "evidence_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body_without_sha, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded_sha == expected_sha

    summary = data["summary"]
    assert summary["overall_pass"] is True
    assert summary["stages_exercised"] == [1, 2, 3, 4, 5, 6]
    assert summary["fault_injections_passed"] == 6
    assert summary["fault_injections_total"] == 6
    assert summary["optical_odd_af447_airdata_blocked_all"] is True
    assert summary["actuator_authority_false_all"] is True
    assert summary["pessimistic_union_preserved_all"] is True
    assert summary["mavlink_50003_roundtrip_verified"] == summary["mavlink_50003_frames_emitted"]

    # Verify all 6 mandatory fault-injection cases passed individually
    fi = data["fault_injections"]
    for key in (
        "fi_1_stale_watchdog_gt_200ms",
        "fi_2_corrupted_crc32",
        "fi_3_optical_odd_fog_masked_abstain",
        "fi_4_conformal_valid_zero_abstain",
        "fi_5_forbidden_gemma3_license_rejection",
        "fi_6_actuator_authority_non_relaxation_invariant",
    ):
        assert fi[key]["pass"] is True, f"Fault injection {key} failed: {fi[key]}"
