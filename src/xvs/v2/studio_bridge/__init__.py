"""OpenXVS Studio v4.5 local 5/5 real-model inference, camera, video & zero-GPL HIL bridge."""

from __future__ import annotations

from .buyer_pack import build_buyer_assurance_pack_zip
from .hil_bridge import (
    FlightGearTelemetryNed,
    StaleTelemetryError,
    StudioHilBridgeManager,
    encode_flightgear_control_csv,
    parse_flightgear_telemetry_csv,
    run_hil_loopback_self_test,
)
from .inference_engine import D2TrackerState, StudioInferenceEngine
from .models_loader import (
    DEFAULT_VRAM_CAP_MB,
    DEFAULT_WAVE_C_LAND_DIR,
    DEFAULT_WAVE_C_THERMAL_DIR,
    EXPECTED_D2_ONNX_SHA256,
    EXPECTED_WAVE_C_LAND_ONNX_SHA256,
    EXPECTED_WAVE_C_LAND_PT_SHA256,
    EXPECTED_WAVE_C_THERMAL_ONNX_SHA256,
    EXPECTED_WAVE_C_THERMAL_PT_SHA256,
    DinoV2ViTBaseBoQ,
    ModelCheckpointMeta,
    PinnedArtifactCandidate,
    StudioModelsBundle,
    extract_l4_checkpoints,
    load_studio_models,
    resolve_pinned_candidate,
)
from .server import StudioBridgeHandler, create_studio_http_server
from .video_analyzer import (
    MAX_VIDEO_UPLOAD_BYTES,
    VALID_VIDEO_CHANNELS,
    VideoJobManager,
    is_valid_job_id,
)

__all__ = [
    "DEFAULT_VRAM_CAP_MB",
    "DEFAULT_WAVE_C_LAND_DIR",
    "DEFAULT_WAVE_C_THERMAL_DIR",
    "EXPECTED_D2_ONNX_SHA256",
    "EXPECTED_WAVE_C_LAND_ONNX_SHA256",
    "EXPECTED_WAVE_C_LAND_PT_SHA256",
    "EXPECTED_WAVE_C_THERMAL_ONNX_SHA256",
    "EXPECTED_WAVE_C_THERMAL_PT_SHA256",
    "MAX_VIDEO_UPLOAD_BYTES",
    "VALID_VIDEO_CHANNELS",
    "D2TrackerState",
    "DinoV2ViTBaseBoQ",
    "FlightGearTelemetryNed",
    "ModelCheckpointMeta",
    "PinnedArtifactCandidate",
    "StaleTelemetryError",
    "StudioBridgeHandler",
    "StudioHilBridgeManager",
    "StudioInferenceEngine",
    "StudioModelsBundle",
    "VideoJobManager",
    "build_buyer_assurance_pack_zip",
    "create_studio_http_server",
    "encode_flightgear_control_csv",
    "extract_l4_checkpoints",
    "is_valid_job_id",
    "load_studio_models",
    "parse_flightgear_telemetry_csv",
    "resolve_pinned_candidate",
    "run_hil_loopback_self_test",
]
