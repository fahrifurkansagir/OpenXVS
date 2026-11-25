"""Pre-wired Phase 4 `live_sky_imx296_adsb` benchmark slot & fail-closed gate validator (DR-08 / SP3 §3.2).

Encodes the late-2026 supply-chain-audited hardware specification (`DR-08:56-69`):
  - Sony IMX296 1456x1088 global-shutter camera + 16 mm C-mount telephoto lens
  - FlightAware Pro Stick Plus / ADSBexchange 1090 MHz filtered SDR (replaces EOL RTL-SDR Blog V4)
  - TDK InvenSense ICM-42688-P 6-DoF MEMS IMU
  - u-blox NEO-M8N GNSS 1PPS + Raspberry Pi Pico hardware XTR exposure trigger
  - Strict fail-closed ADS-B-to-pixel projection gates (`mode_s_tc19_decoded == True`,
    `nic >= 7`, `nacp >= 9`, `boresight_rms_deg <= 0.15`, `pps_synced == True`, fail-closed on `NaN`/`±Inf`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

PHASE4_SPLIT_NAME: str = "live_sky_imx296_adsb"
PHASE4_STATUS_PREWIRED: str = "PREWIRED_PHASE4_PENDING_CAPTURE"


@dataclass(frozen=True)
class LiveSkyHardwareSpec:
    """Late-2026 audited Phase 4 Live-Sky Micro-Pod hardware specification (DR-08 §Candidate 6)."""

    camera: str = "Sony IMX296 1456x1088 global-shutter (3.45 um)"
    optics: str = "16mm C-mount telephoto lens (17.8x13.4 deg FOV, 0.216 mrad/px)"
    sdr: str = "FlightAware Pro Stick Plus / ADSBexchange 1090 MHz filtered SDR"
    excluded_eol_hardware: tuple[str, ...] = ("RTL-SDR Blog V4 (EOL 2026-05-10)",)
    imu: str = "TDK InvenSense ICM-42688-P 6-DoF SPI MEMS IMU"
    timebase: str = "u-blox NEO-M8N 1PPS + Raspberry Pi Pico XTR exposure trigger"


@dataclass(frozen=True)
class LiveSkyGateThresholds:
    """Fail-closed Phase 4 ADS-B-to-pixel error-budget gate thresholds (DR-08:69, 148)."""

    nic_min: int = 7
    nacp_min: int = 9
    boresight_rms_deg_max: float = 0.15
    max_exposure_latency_ms: float = 5.0


@dataclass(frozen=True)
class LiveSkyFrameGateInput:
    """Per-frame live-sky capture metadata screened before admitting into XVS-Bench."""

    mode_s_tc19_decoded: bool
    nic: int
    nacp: int
    boresight_rms_deg: float
    pps_synced: bool
    exposure_latency_ms: float = 0.5


@dataclass(frozen=True)
class LiveSkyGateDecision:
    """Fail-closed admission verdict for a Phase 4 live-sky frame."""

    admitted: bool
    failed_checks: tuple[str, ...]


def validate_live_sky_gate(
    inp: LiveSkyFrameGateInput,
    thresholds: LiveSkyGateThresholds | None = None,
) -> LiveSkyGateDecision:
    """Strictly validate Phase 4 live-sky frame metadata (fail-closed on NaN/±Inf)."""
    thr = LiveSkyGateThresholds() if thresholds is None else thresholds
    failed: list[str] = []

    if inp.mode_s_tc19_decoded is not True:
        failed.append("MODE_S_TC19_NOT_DECODED")
    if not isinstance(inp.nic, int) or isinstance(inp.nic, bool) or inp.nic < thr.nic_min:
        failed.append(f"NIC_BELOW_MIN(nic={inp.nic}<{thr.nic_min})")
    if not isinstance(inp.nacp, int) or isinstance(inp.nacp, bool) or inp.nacp < thr.nacp_min:
        failed.append(f"NACP_BELOW_MIN(nacp={inp.nacp}<{thr.nacp_min})")
    if not (
        math.isfinite(float(inp.boresight_rms_deg))
        and (0.0 <= float(inp.boresight_rms_deg) <= thr.boresight_rms_deg_max)
    ):
        failed.append(
            f"BORESIGHT_RMS_FAIL(deg={inp.boresight_rms_deg}>{thr.boresight_rms_deg_max})"
        )
    if inp.pps_synced is not True:
        failed.append("PPS_SYNC_MISSING")
    if not (
        math.isfinite(float(inp.exposure_latency_ms))
        and (0.0 <= float(inp.exposure_latency_ms) <= thr.max_exposure_latency_ms)
    ):
        failed.append(
            f"EXPOSURE_LATENCY_FAIL(ms={inp.exposure_latency_ms}>{thr.max_exposure_latency_ms})"
        )

    return LiveSkyGateDecision(admitted=(len(failed) == 0), failed_checks=tuple(failed))


def build_prewired_phase4_slot_manifest() -> dict[str, Any]:
    """Emit canonical pre-wired Phase 4 `live_sky_imx296_adsb` manifest block."""
    hw = LiveSkyHardwareSpec()
    thr = LiveSkyGateThresholds()
    return {
        "split": PHASE4_SPLIT_NAME,
        "status": PHASE4_STATUS_PREWIRED,
        "hardware_spec": asdict(hw),
        "adsb_gates": {
            "mode_s_tc19_decoded_required": True,
            "nic_min": thr.nic_min,
            "nacp_min": thr.nacp_min,
            "boresight_rms_deg_max": thr.boresight_rms_deg_max,
            "pps_synced_required": True,
            "max_exposure_latency_ms": thr.max_exposure_latency_ms,
        },
        "target_airliner_captures_phase4": 50,
        "captured_frames_count": 0,
    }
