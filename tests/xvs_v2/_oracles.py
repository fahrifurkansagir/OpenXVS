"""Independent pure-Python reference oracles for OpenXVS v2.0 Milestone M1 (`V2-M1-3` & `V2-M1-4`).

Implements clean-room Python reference models from `docs/xvs/SP1_LIVEDECK_DESIGN.md` (§4.1–§4.4)
and `configs/xvs/odd/v2_m1_odd.yaml` (zero hardcoded ODD thresholds) for differential parity
verification against Rust `xvs-core` (`core_ffi.py`) and browser WebGPU Pass-4 (`pass4_ai_roi.wgsl`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ODD_YAML_PATH = REPO_ROOT / "configs" / "xvs" / "odd" / "v2_m1_odd.yaml"

SAFE_F64_INT_LIMIT = 1 << 52
RAD_TO_URAD_SCALE = 1_000_000.0
M_TO_MM_SCALE = 1_000.0
MAX_VALID_ANGLE_URAD = 100_000_000
MAX_VALID_DIST_MM = 10_000_000_000
I64_MAX = (1 << 63) - 1

# MAVLink v2 OPENXVS_BEARING_TRACK (#52001) wire constants (SP1_LIVEDECK_DESIGN.md §4.2)
OPENXVS_BEARING_TRACK_MSG_ID = 52001
OPENXVS_BEARING_TRACK_CRC_EXTRA = 73
OPENXVS_BEARING_TRACK_PAYLOAD_LEN = 52
OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN = 77
MAVLINK_V2_STX = 0xFD
MAVLINK_IFLAG_SIGNED = 0x01
_BEARING_TRACK_STRUCT = struct.Struct("<Q3i3h3hHHHB B6H".replace(" ", ""))
assert _BEARING_TRACK_STRUCT.size == OPENXVS_BEARING_TRACK_PAYLOAD_LEN


# =============================================================================
# 1. YAML ODD Contract Loader (Zero Hardcoded Thresholds)
# =============================================================================


def load_odd_contract(yaml_path: Path = DEFAULT_ODD_YAML_PATH) -> dict[str, Any]:
    """Parse `configs/xvs/odd/v2_m1_odd.yaml` without requiring third-party YAML deps."""
    text = yaml_path.read_text(encoding="utf-8")
    root: dict[str, Any] = {"thresholds": {}, "rules": {}}
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1].strip()
            root.setdefault(section, {})
            continue
        if ":" not in stripped:
            continue
        key, val_s = [p.strip() for p in stripped.split(":", 1)]
        if val_s.startswith('"') and val_s.endswith('"'):
            val: Any = val_s[1:-1]
        else:
            try:
                val = int(val_s)
            except ValueError:
                val = val_s
        if indent == 0:
            section = None
            root[key] = val
        elif section is not None:
            root[section][key] = val
    return root


# =============================================================================
# 2. PySimplexRtaOracle — Exact §4.1 Fail-Closed Quantizer, MLTL & Simplex RTA
# =============================================================================


class SimplexMode(IntEnum):
    INVALID_INHIBIT = 0
    FALLBACK_D0_SAFE = 1
    DEGRADED_COAST = 2
    ISOLATED_5CAM = 3
    NOMINAL_ML = 4


class QuantFault(IntEnum):
    NON_FINITE = 1
    NEGATIVE_INPUT = 2
    OUT_OF_RANGE = 3
    INVALID_SCALE_OR_BOUNDS = 4


def quantize_nonneg_upper_i64(val: float, scale: float, max_valid: int) -> tuple[int | None, QuantFault | None]:
    """Fail-closed non-negative upper-bound quantizer matching `xvs-rta-core/src/quantize.rs`."""
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or max_valid < 0
        or max_valid > SAFE_F64_INT_LIMIT
    ):
        return None, QuantFault.INVALID_SCALE_OR_BOUNDS
    if not math.isfinite(val):
        return None, QuantFault.NON_FINITE
    if val == 0.0:  # Handles both +0.0 and -0.0 cleanly
        return 0, None
    if val < 0.0:
        return None, QuantFault.NEGATIVE_INPUT

    scaled = float(val) * float(scale)
    if not math.isfinite(scaled):
        return None, QuantFault.NON_FINITE
    if scaled < 0.0:
        return None, QuantFault.NEGATIVE_INPUT
    if scaled > float(max_valid):
        return None, QuantFault.OUT_OF_RANGE

    trunc = int(scaled)
    ceil_val = trunc + 1 if float(trunc) < scaled else trunc
    if ceil_val < 0 or ceil_val > max_valid:
        return None, QuantFault.OUT_OF_RANGE
    return ceil_val, None


def quantize_signed_bounded_i64(
    val: float, scale: float, min_valid: int, max_valid: int
) -> tuple[int | None, QuantFault | None]:
    """Fail-closed signed bounded quantizer (ties away from zero) matching `quantize.rs`."""
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or min_valid > max_valid
        or min_valid < -SAFE_F64_INT_LIMIT
        or max_valid > SAFE_F64_INT_LIMIT
    ):
        return None, QuantFault.INVALID_SCALE_OR_BOUNDS
    if not math.isfinite(val):
        return None, QuantFault.NON_FINITE
    if val == 0.0:
        if min_valid <= 0 <= max_valid:
            return 0, None
        return None, QuantFault.OUT_OF_RANGE

    scaled = float(val) * float(scale)
    if not math.isfinite(scaled):
        return None, QuantFault.NON_FINITE
    if scaled < float(min_valid) or scaled > float(max_valid):
        return None, QuantFault.OUT_OF_RANGE

    adjusted = scaled + 0.5 if scaled >= 0.0 else scaled - 0.5
    if not math.isfinite(adjusted):
        return None, QuantFault.OUT_OF_RANGE
    q = int(adjusted)
    if q < min_valid or q > max_valid:
        return None, QuantFault.OUT_OF_RANGE
    return q, None


def fail_closed_gt(stat: float, thr: float) -> tuple[bool, bool]:
    """Returns `(alarm, non_finite_fault)` strictly fail-closed via `!(stat <= thr)`."""
    finite_ok = math.isfinite(stat) and math.isfinite(thr)
    alarm = (not finite_ok) or (not (stat <= thr))
    return alarm, (not finite_ok)


def fail_closed_abs_gt(val: float, limit: float) -> tuple[bool, bool]:
    """Returns `(alarm, non_finite_fault)` strictly fail-closed via `!(-limit <= val <= limit)`."""
    finite_ok = math.isfinite(val) and math.isfinite(limit) and (limit >= 0.0)
    in_bounds = finite_ok and (-limit <= val <= limit)
    return (not in_bounds), (not finite_ok)


@dataclass(frozen=True)
class PyRtaFloatTelemetry:
    horizon_pl_rad: float = 0.0
    horizon_al_rad: float = 0.008727
    v_vpl_m: float = 0.0
    v_val_m: float = 10.0
    v_hpl_m: float = 0.0
    v_hal_m: float = 40.0
    epipolar_max_rad: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    signed_horizon_residual_rad: float = 0.0
    runway_width_delta_m: float = 0.0
    runway_width_delta_limit_m: float = 5.0
    cusum_stat: float = 0.0
    cusum_thr: float = 10.530
    ood_global_score: float = 0.0
    ood_global_thr: float = 1.0
    ood_cam_scores: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ood_cam_thr: float = 1.0
    cam_health_mask: int = 0x3F
    ml_detector_fault: bool = False
    transient_dropout: bool = False
    approach_active: bool = False
    go_around_ack_reset: bool = False


@dataclass(frozen=True)
class PyRtaTickInput:
    cam_health_mask: int = 0x3F
    ood_exceeded_mask: int = 0x00
    horizon_pl_urad: int = 0
    horizon_al_urad: int = 8727
    v_vpl_mm: int = 0
    v_val_mm: int = 10_000
    v_hpl_mm: int = 0
    v_hal_mm: int = 40_000
    epipolar_max_urad: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    signed_horizon_residual_urad: int = 0
    runway_width_delta_mm: int = 0
    spoof_alarm: bool = False
    ml_detector_fault: bool = False
    ood_global_exceeded: bool = False
    runway_width_fault: bool = False
    transient_dropout: bool = False
    approach_active: bool = False
    go_around_ack_reset: bool = False
    quantize_fault: bool = False


@dataclass(frozen=True)
class PyRtaTickOutput:
    mode: SimplexMode
    raw_target: SimplexMode
    healthy_cam_mask: int
    active_healthy_cams: int
    consec_dropout_ticks: int
    window_dropout_20: int
    recovery_ticks: int
    dwell_ticks: int
    mandatory_go_around: bool
    mltl_critical_violation: bool
    blank_conformal_hud: bool
    inhibit_avoidance_turns: bool
    tick_index: int


def build_rta_input_from_f64(t: PyRtaFloatTelemetry) -> PyRtaTickInput:
    """Pure-Python fail-closed boundary conversion matching `xvs-rta-core::build_rta_input_from_f64`."""
    q_fault = False

    def _q_nn(val: float, scale: float, max_v: int, sentinel: int) -> int:
        nonlocal q_fault
        res, err = quantize_nonneg_upper_i64(val, scale, max_v)
        if err is not None or res is None:
            q_fault = True
            return sentinel
        return res

    def _q_sb(val: float, scale: float, min_v: int, max_v: int) -> int:
        nonlocal q_fault
        res, err = quantize_signed_bounded_i64(val, scale, min_v, max_v)
        if err is not None or res is None:
            q_fault = True
            return I64_MAX
        return res

    horizon_pl_urad = _q_nn(t.horizon_pl_rad, RAD_TO_URAD_SCALE, MAX_VALID_ANGLE_URAD, I64_MAX)
    horizon_al_urad = _q_nn(t.horizon_al_rad, RAD_TO_URAD_SCALE, MAX_VALID_ANGLE_URAD, 0)
    v_vpl_mm = _q_nn(t.v_vpl_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM, I64_MAX)
    v_val_mm = _q_nn(t.v_val_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM, 0)
    v_hpl_mm = _q_nn(t.v_hpl_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM, I64_MAX)
    v_hal_mm = _q_nn(t.v_hal_m, M_TO_MM_SCALE, MAX_VALID_DIST_MM, 0)

    epi_list = [
        _q_nn(t.epipolar_max_rad[i], RAD_TO_URAD_SCALE, MAX_VALID_ANGLE_URAD, I64_MAX)
        for i in range(6)
    ]
    signed_hor = _q_sb(
        t.signed_horizon_residual_rad,
        RAD_TO_URAD_SCALE,
        -MAX_VALID_ANGLE_URAD,
        MAX_VALID_ANGLE_URAD,
    )
    rw_delta = _q_sb(
        t.runway_width_delta_m, M_TO_MM_SCALE, -MAX_VALID_DIST_MM, MAX_VALID_DIST_MM
    )

    rw_fault, rw_nf = fail_closed_abs_gt(t.runway_width_delta_m, t.runway_width_delta_limit_m)
    if rw_nf:
        q_fault = True
    spoof_alarm, c_nf = fail_closed_gt(t.cusum_stat, t.cusum_thr)
    if c_nf:
        q_fault = True
    ood_glob, og_nf = fail_closed_gt(t.ood_global_score, t.ood_global_thr)
    if og_nf:
        q_fault = True

    ood_mask = 0
    for i in range(6):
        cam_ood, cam_nf = fail_closed_gt(t.ood_cam_scores[i], t.ood_cam_thr)
        if cam_nf:
            q_fault = True
        if cam_ood:
            ood_mask |= 1 << i

    return PyRtaTickInput(
        cam_health_mask=int(t.cam_health_mask) & 0x3F,
        ood_exceeded_mask=ood_mask & 0x3F,
        horizon_pl_urad=horizon_pl_urad,
        horizon_al_urad=horizon_al_urad,
        v_vpl_mm=v_vpl_mm,
        v_val_mm=v_val_mm,
        v_hpl_mm=v_hpl_mm,
        v_hal_mm=v_hal_mm,
        epipolar_max_urad=(
            epi_list[0],
            epi_list[1],
            epi_list[2],
            epi_list[3],
            epi_list[4],
            epi_list[5],
        ),
        signed_horizon_residual_urad=signed_hor,
        runway_width_delta_mm=rw_delta,
        spoof_alarm=spoof_alarm,
        ml_detector_fault=bool(t.ml_detector_fault),
        ood_global_exceeded=ood_glob,
        runway_width_fault=rw_fault,
        transient_dropout=bool(t.transient_dropout),
        approach_active=bool(t.approach_active),
        go_around_ack_reset=bool(t.go_around_ack_reset),
        quantize_fault=q_fault,
    )


class PyBoolRing64:
    """Fixed-capacity boolean ring buffer implementing past-time MLTL `H`, `O`, `S`."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = int(capacity)
        self._buf: list[bool] = [False] * self.capacity
        self._head: int = 0
        self._len: int = 0

    def push(self, val: bool) -> None:
        self._buf[self._head] = bool(val)
        self._head = (self._head + 1) % self.capacity
        self._len += 1

    @property
    def available(self) -> int:
        return min(self._len, self.capacity)

    def get_ago(self, k: int) -> bool | None:
        if k < 0 or k >= self.available:
            return None
        idx = (self._head - 1 - k) % self.capacity
        return self._buf[idx]

    def historically(self, n: int, require_warmup: bool = False) -> bool:
        avail = self.available
        if avail == 0:
            return False
        if require_warmup and avail <= n:
            return False
        max_k = min(n, avail - 1)
        return all(bool(self.get_ago(k)) for k in range(max_k + 1))

    def once(self, n: int) -> bool:
        avail = self.available
        if avail == 0:
            return False
        max_k = min(n, avail - 1)
        return any(bool(self.get_ago(k)) for k in range(max_k + 1))

    @staticmethod
    def since(p: PyBoolRing64, q: PyBoolRing64, a: int, b: int) -> bool:
        if a > b:
            return False
        avail = min(p.available, q.available)
        if avail == 0 or a >= avail:
            return False
        max_k = min(b, avail - 1)
        p_all = True
        for k in range(max_k + 1):
            if k >= a and p_all and q.get_ago(k) is True:
                return True
            if p.get_ago(k) is not True:
                p_all = False
                break
        return False

    def count_true_last(self, window: int) -> int:
        limit = min(window, self.available)
        return sum(1 for k in range(limit) if self.get_ago(k) is True)


class PySimplexRtaOracle:
    """Clean-room Python reference oracle for `xvs-rta-core` Simplex RTA + Past-Time MLTL."""

    def __init__(self, odd_yaml_path: Path = DEFAULT_ODD_YAML_PATH) -> None:
        doc = load_odd_contract(odd_yaml_path)
        th = doc["thresholds"]
        cap = int(doc["ring_buffer_capacity"])
        self.epipolar_cam_max_urad = int(th["epipolar_cam_max_urad"])
        self.horizon_al_urad = int(th["horizon_al_urad"])
        self.v_val_mm = int(th["v_val_mm"])
        self.v_hal_mm = int(th["v_hal_mm"])
        self.consec_dropout_max_ticks = int(th["consec_dropout_max_ticks"])
        self.window_dropout_len_ticks = int(th["window_dropout_len_ticks"])
        self.window_dropout_20_max_ticks = int(th["window_dropout_20_max_ticks"])
        self.p0_recovery_ticks = int(th["p0_recovery_ticks"])
        self.dwell_ticks_req = int(th["dwell_ticks"])
        self.min_healthy_cams = int(th["min_healthy_cams"])
        self.cam_count_history_window = int(th["cam_count_history_window"])
        self.epi_once_window = int(th["epi_once_window"])
        self.spoof_once_window = int(th["spoof_once_window"])

        self.mode: SimplexMode = SimplexMode.NOMINAL_ML
        self.consec_dropout_ticks: int = 0
        self.recovery_ticks: int = 0
        self.dwell_ticks: int = 0
        self.mandatory_go_around: bool = False
        self.tick_index: int = 0

        self._cam_ge_4_ring = PyBoolRing64(cap)
        self._epi_rings = [PyBoolRing64(cap) for _ in range(6)]
        self._spoof_ring = PyBoolRing64(cap)
        self._dropout_ring = PyBoolRing64(cap)

    def step(self, inp: PyRtaTickInput) -> PyRtaTickOutput:
        # 1. Past-time MLTL rules
        mltl_cam_fault_mask = 0
        for i in range(6):
            self._epi_rings[i].push(inp.epipolar_max_urad[i] > self.epipolar_cam_max_urad)
            if self._epi_rings[i].once(self.epi_once_window):
                mltl_cam_fault_mask |= 1 << i

        healthy_mask = (inp.cam_health_mask & (~inp.ood_exceeded_mask) & (~mltl_cam_fault_mask)) & 0x3F
        n_cams = int(healthy_mask.bit_count())

        self._cam_ge_4_ring.push(n_cams >= self.min_healthy_cams)
        r_cam_count_ok = self._cam_ge_4_ring.historically(self.cam_count_history_window, False)

        self._spoof_ring.push(bool(inp.spoof_alarm))
        spoof_tripped = self._spoof_ring.once(self.spoof_once_window)

        self._dropout_ring.push(bool(inp.transient_dropout))
        window_dropout_20 = self._dropout_ring.count_true_last(self.window_dropout_len_ticks)

        mltl_critical = (not r_cam_count_ok) or (
            window_dropout_20 > self.window_dropout_20_max_ticks
        )

        # 2. Mode-independent dropout duration & non-resettable Go-Around latch
        next_consec = self.consec_dropout_ticks + 1 if inp.transient_dropout else 0
        go_around = self.mandatory_go_around
        if inp.approach_active:
            if (
                inp.quantize_fault
                or inp.v_vpl_mm > inp.v_val_mm
                or inp.v_hpl_mm > inp.v_hal_mm
                or inp.runway_width_fault
            ):
                go_around = True
        elif inp.go_around_ack_reset:
            go_around = False

        # 3. Total raw_target function
        if (
            inp.quantize_fault
            or n_cams < self.min_healthy_cams
            or inp.horizon_pl_urad > inp.horizon_al_urad
            or go_around
            or mltl_critical
            or next_consec > self.consec_dropout_max_ticks
            or window_dropout_20 > self.window_dropout_20_max_ticks
        ):
            raw_target = SimplexMode.INVALID_INHIBIT
        elif (
            inp.spoof_alarm
            or spoof_tripped
            or inp.ml_detector_fault
            or inp.ood_global_exceeded
            or n_cams == 4
        ):
            raw_target = SimplexMode.FALLBACK_D0_SAFE
        elif inp.transient_dropout:
            raw_target = SimplexMode.DEGRADED_COAST
        elif n_cams == 5:
            raw_target = SimplexMode.ISOLATED_5CAM
        else:
            raw_target = SimplexMode.NOMINAL_ML

        # 4. P0 healthy-recovery predicate
        is_p0_healthy = (
            raw_target
            in (SimplexMode.FALLBACK_D0_SAFE, SimplexMode.ISOLATED_5CAM, SimplexMode.NOMINAL_ML)
            and not inp.transient_dropout
        )

        # 5. Total transition table
        if int(raw_target) < int(self.mode):
            next_mode, next_rec, next_dwell = raw_target, 0, 0
        elif self.mode == SimplexMode.INVALID_INHIBIT:
            if not is_p0_healthy:
                next_mode, next_rec, next_dwell = SimplexMode.INVALID_INHIBIT, 0, 0
            else:
                r = self.recovery_ticks + 1
                if r >= self.p0_recovery_ticks:
                    next_mode, next_rec, next_dwell = SimplexMode.FALLBACK_D0_SAFE, 0, 0
                else:
                    next_mode, next_rec, next_dwell = SimplexMode.INVALID_INHIBIT, r, 0
        elif self.mode == SimplexMode.FALLBACK_D0_SAFE:
            if raw_target in (SimplexMode.FALLBACK_D0_SAFE, SimplexMode.DEGRADED_COAST):
                next_mode, next_rec, next_dwell = SimplexMode.FALLBACK_D0_SAFE, 0, 0
            else:
                d = self.dwell_ticks + 1
                if d >= self.dwell_ticks_req:
                    next_mode, next_rec, next_dwell = raw_target, 0, 0
                else:
                    next_mode, next_rec, next_dwell = SimplexMode.FALLBACK_D0_SAFE, 0, d
        elif self.mode == SimplexMode.DEGRADED_COAST:
            if raw_target == SimplexMode.DEGRADED_COAST:
                next_mode, next_rec, next_dwell = SimplexMode.DEGRADED_COAST, 0, 0
            else:
                next_mode, next_rec, next_dwell = SimplexMode.FALLBACK_D0_SAFE, 0, 1
        elif self.mode == SimplexMode.ISOLATED_5CAM:
            if raw_target == SimplexMode.ISOLATED_5CAM:
                next_mode, next_rec, next_dwell = SimplexMode.ISOLATED_5CAM, 0, 0
            else:
                d = self.dwell_ticks + 1
                if d >= self.dwell_ticks_req:
                    next_mode, next_rec, next_dwell = SimplexMode.NOMINAL_ML, 0, 0
                else:
                    next_mode, next_rec, next_dwell = SimplexMode.ISOLATED_5CAM, 0, d
        else:
            next_mode, next_rec, next_dwell = SimplexMode.NOMINAL_ML, 0, 0

        self.mode = next_mode
        self.consec_dropout_ticks = next_consec
        self.recovery_ticks = next_rec
        self.dwell_ticks = next_dwell
        self.mandatory_go_around = go_around
        self.tick_index += 1

        is_p0 = next_mode == SimplexMode.INVALID_INHIBIT
        return PyRtaTickOutput(
            mode=next_mode,
            raw_target=raw_target,
            healthy_cam_mask=healthy_mask,
            active_healthy_cams=n_cams,
            consec_dropout_ticks=next_consec,
            window_dropout_20=window_dropout_20,
            recovery_ticks=next_rec,
            dwell_ticks=next_dwell,
            mandatory_go_around=go_around,
            mltl_critical_violation=mltl_critical,
            blank_conformal_hud=is_p0,
            inhibit_avoidance_turns=is_p0,
            tick_index=self.tick_index,
        )


# =============================================================================
# 3. PyMathOracles — Tie-Margin Helper, CUSUM/SS, Vision-RAIM, MeshDAA, Pass-4
# =============================================================================


def check_threshold_tie_margin(stat: float, thr: float, eps: float = 1e-9) -> tuple[bool, float]:
    """Return `(is_non_tie, abs_margin)` for `V2-M1-3` discrete boundary comparison."""
    margin = abs(float(stat) - float(thr))
    return (margin > float(eps)), margin


_INV_SQRT3 = 1.0 / math.sqrt(3.0)
OCTAHEDRAL_8_ARMS = np.array(
    [
        [+1.0, +1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=np.float64,
) * _INV_SQRT3


@dataclass
class PyBonferroniSpoofDetector:
    """Reference Bonferroni 3-layer GNSS spoofing detector: NIS v 8-arm Cholesky CUSUM v Dual-Bank SS."""

    cusum_threshold_hc: float = 10.530
    cusum_drift_k: float = 0.5
    nis_threshold: float = 16.266  # chi2_3(0.999)
    ss_threshold: float = 16.266
    maneuver_conformal_margin: float = 0.0
    arms: np.ndarray = field(default_factory=lambda: OCTAHEDRAL_8_ARMS.copy())
    cusum_states: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))

    def step(
        self,
        innov: np.ndarray,
        s_cov: np.ndarray,
        *,
        bank0_state: np.ndarray | None = None,
        bank1_state: np.ndarray | None = None,
        p_0: np.ndarray | None = None,
        p_1: np.ndarray | None = None,
        p_01: np.ndarray | None = None,
        maneuver_active: bool = False,
    ) -> dict[str, Any]:
        nu = np.asarray(innov, dtype=np.float64).reshape(3)
        s_mat = np.asarray(s_cov, dtype=np.float64).reshape(3, 3)
        chol_l = np.linalg.cholesky(s_mat)
        z = np.linalg.solve(chol_l, nu)

        nis = float(np.dot(z, z))
        eff_nis_thr = self.nis_threshold + (self.maneuver_conformal_margin if maneuver_active else 0.0)
        nis_alarm = not (nis <= eff_nis_thr)

        proj = self.arms @ z
        self.cusum_states = np.maximum(0.0, self.cusum_states + proj - self.cusum_drift_k)
        cusum_max = float(np.max(self.cusum_states))
        cusum_alarm = not (cusum_max <= self.cusum_threshold_hc)

        ss_stat = 0.0
        ss_alarm = False
        if (
            bank0_state is not None
            and bank1_state is not None
            and p_0 is not None
            and p_1 is not None
        ):
            diff = np.asarray(bank0_state, dtype=np.float64) - np.asarray(bank1_state, dtype=np.float64)
            p_cross = np.zeros_like(p_0) if p_01 is None else np.asarray(p_01, dtype=np.float64)
            p_ss = np.asarray(p_0, dtype=np.float64) + np.asarray(p_1, dtype=np.float64) - p_cross - p_cross.T
            p_ss = 0.5 * (p_ss + p_ss.T) + 1e-12 * np.eye(p_ss.shape[0])
            ss_stat = float(diff.T @ np.linalg.solve(p_ss, diff))
            ss_alarm = not (ss_stat <= self.ss_threshold)

        non_tie_cusum, cusum_margin = check_threshold_tie_margin(cusum_max, self.cusum_threshold_hc)
        return {
            "nis": nis,
            "nis_alarm": nis_alarm,
            "cusum_max": cusum_max,
            "cusum_alarm": cusum_alarm,
            "cusum_non_tie": non_tie_cusum,
            "cusum_margin": cusum_margin,
            "ss_stat": ss_stat,
            "ss_alarm": ss_alarm,
            "spoof_alarm": bool(nis_alarm or cusum_alarm or ss_alarm),
        }


def compute_vision_raim_pl(
    h_jac: np.ndarray,
    sigma_meas: float,
    *,
    k_md: float = 5.33,
    t_detect: float = 4.5,
    extra_bias_meas: float = 0.0,
) -> dict[str, float]:
    """Compute Vision-RAIM V-VPL & V-HPL via worst-case single-feature failure-mode slopes `g_max,i`.

    Assumes state ordering `[north_m, east_m, down_m]` (NED per CONVENTIONS.md).
    """
    h = np.asarray(h_jac, dtype=np.float64)
    m, n = h.shape
    if m <= n or n < 3:
        raise ValueError(f"Vision-RAIM requires redundant observations (m > n >= 3), got {h.shape}")
    w_inv = float(sigma_meas) ** 2
    cov = np.linalg.inv(h.T @ h) * w_inv
    s_mat = np.linalg.solve(h.T @ h, h.T)  # (n, m)
    hat = h @ s_mat  # (m, m)

    g_v_max = 0.0
    g_h_max = 0.0
    for i in range(m):
        denom = max(1e-9, 1.0 - float(hat[i, i]))
        sqrt_d = math.sqrt(denom)
        g_v = abs(float(s_mat[2, i])) / sqrt_d
        g_h = math.hypot(float(s_mat[0, i]), float(s_mat[1, i])) / sqrt_d
        if g_v > g_v_max:
            g_v_max = g_v
        if g_h > g_h_max:
            g_h_max = g_h

    sigma_v = math.sqrt(max(0.0, float(cov[2, 2])))
    sigma_h = math.sqrt(max(0.0, float(cov[0, 0] + cov[1, 1])))
    bias_term = float(t_detect) * float(sigma_meas) + abs(float(extra_bias_meas))
    v_vpl_m = float(k_md) * sigma_v + g_v_max * bias_term
    v_hpl_m = float(k_md) * sigma_h + g_h_max * bias_term
    return {
        "v_vpl_m": v_vpl_m,
        "v_hpl_m": v_hpl_m,
        "g_v_max": g_v_max,
        "g_h_max": g_h_max,
        "sigma_v_m": sigma_v,
        "sigma_h_m": sigma_h,
    }


def check_runway_width_consistency(
    observed_width_m: float,
    expected_runway_width_m: float = 60.96,
    max_abs_delta_m: float = 12.0,
) -> tuple[bool, float]:
    """KSFO 28L (60.96 m) vs parallel Taxiway C (22.86 m) width consistency gate."""
    delta_m = float(observed_width_m) - float(expected_runway_width_m)
    alarm, _ = fail_closed_abs_gt(delta_m, max_abs_delta_m)
    return (not alarm), delta_m


def two_ray_closest_approach_ecef(
    p1: np.ndarray, u1: np.ndarray, p2: np.ndarray, u2: np.ndarray
) -> dict[str, Any]:
    """Compute exact 3D closest point of approach between two ECEF bearing rays."""
    p1_v = np.asarray(p1, dtype=np.float64).reshape(3)
    p2_v = np.asarray(p2, dtype=np.float64).reshape(3)
    u1_v = np.asarray(u1, dtype=np.float64).reshape(3)
    u2_v = np.asarray(u2, dtype=np.float64).reshape(3)
    u1_v = u1_v / max(1e-15, float(np.linalg.norm(u1_v)))
    u2_v = u2_v / max(1e-15, float(np.linalg.norm(u2_v)))

    w0 = p1_v - p2_v
    b = float(np.dot(u1_v, u2_v))
    d = float(np.dot(u1_v, w0))
    e = float(np.dot(u2_v, w0))
    denom = 1.0 - b * b
    if denom < 1e-12:
        return {"valid": False, "skew_dist_m": float("inf"), "p_mid": np.zeros(3), "r1_m": 0.0, "r2_m": 0.0}
    r1 = (b * e - d) / denom
    r2 = (e - b * d) / denom
    q1 = p1_v + r1 * u1_v
    q2 = p2_v + r2 * u2_v
    return {
        "valid": bool(r1 > 0.0 and r2 > 0.0),
        "skew_dist_m": float(np.linalg.norm(q1 - q2)),
        "p_mid": 0.5 * (q1 + q2),
        "r1_m": float(r1),
        "r2_m": float(r2),
    }


def triple_ray_epipolar_ghost_check(
    p1: np.ndarray,
    u1: np.ndarray,
    p2: np.ndarray,
    u2: np.ndarray,
    p3: np.ndarray,
    u3: np.ndarray,
    cone_half_angle_rad: float,
) -> dict[str, Any]:
    """Verify a 2-ship candidate point against a 3rd cooperative peer's ECEF bearing ray."""
    tri = two_ray_closest_approach_ecef(p1, u1, p2, u2)
    if not tri["valid"]:
        return {"accepted": False, "ghost_rejected": True, "ray3_angle_err_rad": float("inf")}
    vec3 = tri["p_mid"] - np.asarray(p3, dtype=np.float64).reshape(3)
    r3 = float(np.linalg.norm(vec3))
    if r3 < 1.0:
        return {"accepted": False, "ghost_rejected": True, "ray3_angle_err_rad": float("inf")}
    u3_hat = np.asarray(u3, dtype=np.float64).reshape(3)
    u3_hat = u3_hat / max(1e-15, float(np.linalg.norm(u3_hat)))
    cos_ang = max(-1.0, min(1.0, float(np.dot(vec3 / r3, u3_hat))))
    ang_err = math.acos(cos_ang)
    accepted = bool(ang_err <= float(cone_half_angle_rad))
    return {
        "accepted": accepted,
        "ghost_rejected": (not accepted),
        "ray3_angle_err_rad": ang_err,
        "candidate_ecef": tri["p_mid"],
    }


def fast_covariance_intersection(
    x1: np.ndarray, p1: np.ndarray, x2: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fast Covariance Intersection (FCI) with closed-form trace-inverse weight `omega`."""
    x1_v = np.asarray(x1, dtype=np.float64)
    x2_v = np.asarray(x2, dtype=np.float64)
    p1_m = np.asarray(p1, dtype=np.float64)
    p2_m = np.asarray(p2, dtype=np.float64)
    inv1 = np.linalg.inv(p1_m)
    inv2 = np.linalg.inv(p2_m)
    tr1 = float(np.trace(inv1))
    tr2 = float(np.trace(inv2))
    omega = tr1 / max(1e-15, tr1 + tr2)
    p_fused = np.linalg.inv(omega * inv1 + (1.0 - omega) * inv2)
    p_fused = 0.5 * (p_fused + p_fused.T)
    x_fused = p_fused @ (omega * (inv1 @ x1_v) + (1.0 - omega) * (inv2 @ x2_v))
    return x_fused, p_fused, float(omega)


def _mavlink_x25_crc(header_no_stx: bytes, payload: bytes, crc_extra: int) -> int:
    crc = 0xFFFF
    for b in header_no_stx + payload + bytes([crc_extra & 0xFF]):
        tmp = (b ^ (crc & 0xFF)) & 0xFF
        tmp = (tmp ^ ((tmp << 4) & 0xFF)) & 0xFF
        crc = (((crc >> 8) & 0xFF) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc & 0xFFFF


@dataclass(frozen=True)
class OpenXvsBearingTrack:
    """52-byte ECEF payload for private MAVLink v2 message #52001 (`SP1_LIVEDECK_DESIGN.md` §4.2)."""

    time_utc_us: int
    own_ecef_cm: tuple[int, int, int]
    bearing_ecef_q15: tuple[int, int, int]
    bearing_rate_urad_s: tuple[int, int, int]
    cone_half_angle_urad: int
    own_pos_std_cm: int
    track_id: int
    sender_id: int
    flags: int
    cov_diag_packed: tuple[int, int, int, int, int, int]


def pack_bearing_track_payload(msg: OpenXvsBearingTrack) -> bytes:
    payload = _BEARING_TRACK_STRUCT.pack(
        int(msg.time_utc_us),
        int(msg.own_ecef_cm[0]),
        int(msg.own_ecef_cm[1]),
        int(msg.own_ecef_cm[2]),
        int(msg.bearing_ecef_q15[0]),
        int(msg.bearing_ecef_q15[1]),
        int(msg.bearing_ecef_q15[2]),
        int(msg.bearing_rate_urad_s[0]),
        int(msg.bearing_rate_urad_s[1]),
        int(msg.bearing_rate_urad_s[2]),
        int(msg.cone_half_angle_urad),
        int(msg.own_pos_std_cm),
        int(msg.track_id),
        int(msg.sender_id),
        int(msg.flags),
        *[int(v) for v in msg.cov_diag_packed],
    )
    assert len(payload) == OPENXVS_BEARING_TRACK_PAYLOAD_LEN
    return payload


def unpack_bearing_track_payload(payload: bytes) -> OpenXvsBearingTrack:
    if len(payload) != OPENXVS_BEARING_TRACK_PAYLOAD_LEN:
        raise ValueError(f"Expected 52-byte payload, got {len(payload)}")
    vals = _BEARING_TRACK_STRUCT.unpack(payload)
    return OpenXvsBearingTrack(
        time_utc_us=vals[0],
        own_ecef_cm=(vals[1], vals[2], vals[3]),
        bearing_ecef_q15=(vals[4], vals[5], vals[6]),
        bearing_rate_urad_s=(vals[7], vals[8], vals[9]),
        cone_half_angle_urad=vals[10],
        own_pos_std_cm=vals[11],
        track_id=vals[12],
        sender_id=vals[13],
        flags=vals[14],
        cov_diag_packed=(vals[15], vals[16], vals[17], vals[18], vals[19], vals[20]),
    )


def pack_signed_bearing_track_mavlink_v2(
    msg: OpenXvsBearingTrack,
    *,
    secret_key_32: bytes,
    link_id: int = 0,
    sig_timestamp_48: int = 1,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 156,
) -> bytes:
    """Pack exact 77-byte signed MAVLink v2 frame (10B header + 52B payload + 2B CRC + 13B SHA-256 sig)."""
    if len(secret_key_32) != 32:
        raise ValueError("MAVLink v2 signing key must be 32 bytes")
    payload = pack_bearing_track_payload(msg)
    header = bytes(
        [
            MAVLINK_V2_STX,
            OPENXVS_BEARING_TRACK_PAYLOAD_LEN,
            MAVLINK_IFLAG_SIGNED,
            0,
            seq & 0xFF,
            sysid & 0xFF,
            compid & 0xFF,
            OPENXVS_BEARING_TRACK_MSG_ID & 0xFF,
            (OPENXVS_BEARING_TRACK_MSG_ID >> 8) & 0xFF,
            (OPENXVS_BEARING_TRACK_MSG_ID >> 16) & 0xFF,
        ]
    )
    crc = _mavlink_x25_crc(header[1:], payload, OPENXVS_BEARING_TRACK_CRC_EXTRA)
    crc_bytes = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    sig_hdr = bytes([link_id & 0xFF]) + int(sig_timestamp_48).to_bytes(6, "little")
    sig_tag = hashlib.sha256(secret_key_32 + header + payload + crc_bytes + sig_hdr).digest()[:6]
    frame = header + payload + crc_bytes + sig_hdr + sig_tag
    assert len(frame) == OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN
    return frame


def unpack_and_verify_signed_bearing_track(
    frame: bytes, *, secret_key_32: bytes
) -> OpenXvsBearingTrack:
    """Verify exact 77-byte MAVLink v2 CRC and SHA-256 signature and decode 52-byte payload."""
    if len(frame) != OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN:
        raise ValueError(f"Expected 77-byte signed frame, got {len(frame)}")
    if frame[0] != MAVLINK_V2_STX or (frame[2] & MAVLINK_IFLAG_SIGNED) == 0:
        raise ValueError("Invalid MAVLink v2 signed STX or incompat_flags")
    msgid = frame[7] | (frame[8] << 8) | (frame[9] << 16)
    if msgid != OPENXVS_BEARING_TRACK_MSG_ID:
        raise ValueError(f"Unexpected msgid {msgid}, expected {OPENXVS_BEARING_TRACK_MSG_ID}")

    header = frame[:10]
    payload = frame[10:62]
    crc_bytes = frame[62:64]
    sig_hdr = frame[64:71]
    sig_rx = frame[71:77]

    rx_crc = crc_bytes[0] | (crc_bytes[1] << 8)
    calc_crc = _mavlink_x25_crc(header[1:], payload, OPENXVS_BEARING_TRACK_CRC_EXTRA)
    if rx_crc != calc_crc:
        raise ValueError(f"MAVLink X.25 CRC mismatch: rx=0x{rx_crc:04X}, calc=0x{calc_crc:04X}")

    sig_calc = hashlib.sha256(secret_key_32 + header + payload + crc_bytes + sig_hdr).digest()[:6]
    if sig_rx != sig_calc:
        raise ValueError("MAVLink v2 SHA-256 signature verification failed")

    return unpack_bearing_track_payload(payload)


def srgb_u8_to_electron_luma(rgb_u8: np.ndarray, full_well_e: float = 10500.0) -> np.ndarray:
    """Exact Pass-4 sRGB EOTF -> Rec.709 linear luma -> `* full_well_e` (`pack.py:354-356`)."""
    s = np.asarray(rgb_u8, dtype=np.float32) / np.float32(255.0)
    lin = np.where(
        s <= np.float32(0.04045),
        s / np.float32(12.92),
        np.power((s + np.float32(0.055)) / np.float32(1.055), np.float32(2.4)),
    )
    luma = (
        np.float32(0.2126) * lin[:, :, 0]
        + np.float32(0.7152) * lin[:, :, 1]
        + np.float32(0.0722) * lin[:, :, 2]
    )
    return (luma * np.float32(full_well_e)).astype(np.float32)


def compute_pass4_reference_tensor(
    rgb_t: np.ndarray,
    rgb_tm1_warped: np.ndarray,
    rgb_tm2_warped: np.ndarray,
    mask_m1: np.ndarray | None = None,
    mask_m2: np.ndarray | None = None,
    *,
    full_well_e: float = 10500.0,
) -> np.ndarray:
    """Deterministic noise-off Pass-4 reference matching `pack.py` + `warp.py` + `d1_model.py:normalize_input`."""
    i_t = srgb_u8_to_electron_luma(rgb_t, full_well_e=full_well_e)
    i_1 = srgb_u8_to_electron_luma(rgb_tm1_warped, full_well_e=full_well_e)
    i_2 = srgb_u8_to_electron_luma(rgb_tm2_warped, full_well_e=full_well_e)

    m1 = np.ones_like(i_t, dtype=np.float32) if mask_m1 is None else np.asarray(mask_m1, dtype=np.float32)
    m2 = np.ones_like(i_t, dtype=np.float32) if mask_m2 is None else np.asarray(mask_m2, dtype=np.float32)

    d1 = m1 * (i_t - i_1)
    d2 = m2 * (i_t - i_2)
    stack = np.stack([i_t, i_1, d1, i_2, d2], axis=0)[np.newaxis, :, :, :].astype(np.float32)

    # Exact `D1HeatmapDetector.normalize_input` (d1_model.py:285-289)
    mean = stack.mean(axis=(-2, -1), keepdims=True)
    std0 = np.maximum(np.std(stack[:, :1], axis=(-2, -1), ddof=0, keepdims=True), np.float32(1.0))
    x_norm = np.clip((stack - mean) / std0, np.float32(-20.0), np.float32(20.0))
    return x_norm.astype(np.float32)
