"""Bit-exact `f64` Python reference mirror of `#![no_std]` Rust `xvs-active-range` (Spec §4.1.1–§4.1.2).

Line-for-line port of `xvs-active-range/src/{lib,msc_rpekf,conformal,sc3bf_fim,wire}.rs`:
1. `M = 6` log-spaced Modified Spherical Coordinates Range-Parameterized EKF (`MSC-RPEKF`) in NED
   with RK2 propagation, 2nd-order `F = I + dt*J + 0.5*dt^2*J^2`, and Joseph covariance updates.
2. Scale-cancelling optical expansion (`z_tau = -s_dot / s`) + gated daytime Koschmieder prior.
3. Frozen Split Conformal safe lower range bound (`CP-MSC`).
4. Dimensionally exact (`m^2/s`) universal-range `∀ h ∈ active_banks ∪ {R_safe_lower}` discrete-time
   `SC3BF-FIM` micro-swerve selector using exact compile-time `PHI_CAND` tables.
5. Exact `64-byte` (`[u8; 64]`) LE wire codec with IEEE-802.3 CRC32 over bytes `0..60`.

Strictly NED coordinates (`North, East, Down`), zero internal ENU, zero unannotated `.get()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import struct
import zlib

import numpy as np

from xvs.v2.xvs_active_range.conformal import (
    R_CEIL_M,
    R_FLOOR_M,
    SIGMA_FLOOR_M,
    compute_conformal_lower_bound,
)

# ---- Locked Spec §4.1.1 Constants (Bit-Exact with `xvs-active-range/src/lib.rs`) ----
NUM_BANKS: int = 6
R_INIT_M: tuple[float, float, float, float, float, float] = (
    200.0,
    348.2202,
    606.2866,
    1055.6063,
    1837.9174,
    3200.0,
)
W_FLOOR: float = 1.0e-4
W_ACTIVE: float = 0.02

G_ACCEL_MPS2: float = 9.80665
R_SEP_M: float = 152.4
EVADE_BANK_DEG: float = 25.0
EVADE_BANK_RAD: float = 0.4363323129985824
A_MAX_EVADE_MPS2: float = 4.572915995845716
K_ALLOW: float = 0.5
ALPHA_CBF: float = 0.35
DT_LOOK_S: float = 1.0
R_EVADE_TRIGGER_M: float = 220.0
TAU_EVADE_TRIGGER_S: float = 4.5

PHI_CAND_DEG: tuple[float, ...] = (-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0)
PHI_CAND_RAD: tuple[float, ...] = (
    -0.20943951023931956,
    -0.15707963267948966,
    -0.10471975511965978,
    -0.05235987755982989,
    0.0,
    0.05235987755982989,
    0.10471975511965978,
    0.15707963267948966,
    0.20943951023931956,
)
PHI_CAND_SIN: tuple[float, ...] = (
    -0.20791169081775934,
    -0.15643446504023087,
    -0.10452846326765347,
    -0.052335956242943835,
    0.0,
    0.052335956242943835,
    0.10452846326765347,
    0.15643446504023087,
    0.20791169081775934,
)
PHI_CAND_COS: tuple[float, ...] = (
    0.9781476007338057,
    0.9876883405951378,
    0.9945218953682733,
    0.9986295347545738,
    1.0,
    0.9986295347545738,
    0.9945218953682733,
    0.9876883405951378,
    0.9781476007338057,
)
PHI_CAND_TAN: tuple[float, ...] = (
    -0.21255656167002213,
    -0.15838444032453627,
    -0.10510423526567647,
    -0.05240777928304121,
    0.0,
    0.05240777928304121,
    0.10510423526567647,
    0.15838444032453627,
    0.21255656167002213,
)

FOV_MAX_DEG: float = 55.0
FOV_MARGIN_DEG: float = 8.0
FOV_LIMIT_RAD: float = (FOV_MAX_DEG - FOV_MARGIN_DEG) * (math.pi / 180.0)

TAU_MIN_SIZE_PX: float = 6.0
TAU_MIN_TRACK_AGE: int = 5
KOSCH_MAX_SIZE_PX: float = 8.0
KOSCH_MAX_LOG_DELTA: float = 0.15

TRIGGER_REL_SIGMA: float = 0.10
LOCK_REL_SIGMA: float = 0.08
CLOSING_RHO_DOT_THRESH: float = -0.015

FLAG_TAU_FUSED: int = 1 << 0
FLAG_KOSCH_FUSED: int = 1 << 1
FLAG_SC3BF_ALL_SAFE: int = 1 << 2
FLAG_FOV_SAFE: int = 1 << 3

WIRE_STRUCT = struct.Struct("<IBBH7f6fI")
WIRE_SIZE_BYTES: int = 64
_F32_STRUCT = struct.Struct("<f")


def _f32(val: float) -> float:
    """Exact IEEE-754 single-precision roundtrip matching Rust `as f32`."""
    return float(_F32_STRUCT.unpack(_F32_STRUCT.pack(float(val)))[0])


def _clamp_f64(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def _wrap_pi(angle_rad: float) -> float:
    """Bit-exact port of Rust `wrap_pi` using `math.fmod`."""
    if not math.isfinite(angle_rad):
        return 0.0
    two_pi = 2.0 * math.pi
    a = math.fmod(angle_rad + math.pi, two_pi)
    if a < 0.0:
        a += two_pi
    return a - math.pi


class ActiveRangeMode(IntEnum):
    """Mode enum matching Rust `#[repr(u8)] ActiveRangeMode`."""

    NOMINAL_HOLD = 0
    ACTIVE_SWERVE = 1
    LOCKED = 2
    IMMEDIATE_EVADE = 3


@dataclass(frozen=True)
class ActiveRangeConfig:
    """Static configuration matching Rust `ActiveRangeConfig`."""

    r_init_m: tuple[float, float, float, float, float, float] = R_INIT_M
    sigma_bearing_rad: float = 0.00055
    q_hat_conformal: float = 1.65


@dataclass(frozen=True)
class OwnshipStateNed:
    """Ownship kinematic state strictly in NED (`North, East, Down`)."""

    v_ned: tuple[float, float, float]
    a_ned: tuple[float, float, float]
    heading_rad: float
    tas_mps: float
    v_stall_mps: float = 14.5


@dataclass(frozen=True)
class ActiveRangeMeasurement:
    """Single monocular intruder observation at step `t`."""

    alpha_rad: float
    beta_rad: float
    size_px: float
    size_rate_px_s: float
    contrast_ratio: float = 0.5
    sky_beta_ext: float = 0.0
    daytime_clear: bool = False
    track_age: int = 1
    dt_s: float = 0.1


@dataclass(frozen=True)
class ActiveRangeTelemetry:
    """Exact 64-byte wire-compatible telemetry packet."""

    seq: int
    mode: ActiveRangeMode
    active_bank_idx: int
    flags: int
    r_hat_m: float
    sigma_r_m: float
    r_safe_lower_m: float
    rho_dot_inv_s: float
    cmd_bank_rad: float
    min_sc3bf_margin: float
    fim_gain: float
    weights: tuple[float, float, float, float, float, float]
    crc32: int = 0


def encode_wire(telemetry: ActiveRangeTelemetry) -> bytes:
    """Encode `ActiveRangeTelemetry` into exact 64-byte LE frame with IEEE-802.3 CRC32 over `0..60`."""
    if len(telemetry.weights) != NUM_BANKS:
        raise ValueError(f"Expected {NUM_BANKS} weights, got {len(telemetry.weights)}")
    payload_60 = struct.pack(
        "<IBBH7f6f",
        int(telemetry.seq) & 0xFFFFFFFF,
        int(telemetry.mode) & 0xFF,
        int(telemetry.active_bank_idx) & 0xFF,
        int(telemetry.flags) & 0xFFFF,
        float(telemetry.r_hat_m),
        float(telemetry.sigma_r_m),
        float(telemetry.r_safe_lower_m),
        float(telemetry.rho_dot_inv_s),
        float(telemetry.cmd_bank_rad),
        float(telemetry.min_sc3bf_margin),
        float(telemetry.fim_gain),
        float(telemetry.weights[0]),
        float(telemetry.weights[1]),
        float(telemetry.weights[2]),
        float(telemetry.weights[3]),
        float(telemetry.weights[4]),
        float(telemetry.weights[5]),
    )
    crc = zlib.crc32(payload_60) & 0xFFFFFFFF
    return payload_60 + struct.pack("<I", crc)


def decode_wire(raw: bytes) -> ActiveRangeTelemetry | None:
    """Decode and validate exact 64-byte wire frame (CRC32, mode, active_bank_idx, finiteness)."""
    if len(raw) != WIRE_SIZE_BYTES:
        return None
    expected_crc = zlib.crc32(raw[:60]) & 0xFFFFFFFF
    fields = WIRE_STRUCT.unpack(raw)
    seq = int(fields[0])
    mode_raw = int(fields[1])
    active_bank_idx = int(fields[2])
    flags = int(fields[3])
    floats_7 = fields[4:11]
    weights_6 = fields[11:17]
    crc = int(fields[17])
    if crc != expected_crc:
        return None
    if mode_raw not in (0, 1, 2, 3) or active_bank_idx >= NUM_BANKS:
        return None
    if not all(math.isfinite(float(v)) for v in floats_7) or not all(math.isfinite(float(w)) for w in weights_6):
        return None
    return ActiveRangeTelemetry(
        seq=seq,
        mode=ActiveRangeMode(mode_raw),
        active_bank_idx=active_bank_idx,
        flags=flags,
        r_hat_m=float(floats_7[0]),
        sigma_r_m=float(floats_7[1]),
        r_safe_lower_m=float(floats_7[2]),
        rho_dot_inv_s=float(floats_7[3]),
        cmd_bank_rad=float(floats_7[4]),
        min_sc3bf_margin=float(floats_7[5]),
        fim_gain=float(floats_7[6]),
        weights=(
            float(weights_6[0]),
            float(weights_6[1]),
            float(weights_6[2]),
            float(weights_6[3]),
            float(weights_6[4]),
            float(weights_6[5]),
        ),
        crc32=crc,
    )


def normalize_weights_and_moments(
    log_lik: tuple[float, ...] | list[float],
    r_banks: tuple[float, ...] | list[float],
    var_r_banks: tuple[float, ...] | list[float],
    prev_w: tuple[float, ...] | list[float],
) -> tuple[tuple[float, float, float, float, float, float], float, float]:
    """Bit-exact port of Rust `normalize_weights_and_moments` (`Kani #15`)."""
    max_raw_ll = -1.0e300
    for m in range(NUM_BANKS):
        ll_v = float(log_lik[m])
        if math.isfinite(ll_v) and ll_v > max_raw_ll:
            max_raw_ll = ll_v

    log_w = [0.0] * NUM_BANKS
    max_log = -1.0e300
    for m in range(NUM_BANKS):
        pw = float(prev_w[m])
        if math.isfinite(pw) and pw >= W_FLOOR:
            wp = 1.0 if pw > 1.0 else pw
        else:
            wp = W_FLOOR
        ll_raw = float(log_lik[m])
        if math.isfinite(ll_raw) and max_raw_ll > -1.0e299:
            ll = _clamp_f64(ll_raw - max_raw_ll, -25.0, 0.0)
        else:
            ll = -25.0
        lw = math.log(wp) + ll
        log_w[m] = lw
        if lw > max_log:
            max_log = lw

    unnorm = [0.0] * NUM_BANKS
    sum_unnorm = 0.0
    for m in range(NUM_BANKS):
        diff = _clamp_f64(log_w[m] - max_log, -50.0, 0.0)
        u = math.exp(diff)
        unnorm[m] = u
        sum_unnorm += u

    scale = 1.0 - float(NUM_BANKS) * W_FLOOR
    w_out = [1.0 / float(NUM_BANKS)] * NUM_BANKS
    if math.isfinite(sum_unnorm) and sum_unnorm > 0.0:
        for m in range(NUM_BANKS):
            w_out[m] = scale * (unnorm[m] / sum_unnorm) + W_FLOOR

    r_clean = [0.0] * NUM_BANKS
    v_clean = [0.0] * NUM_BANKS
    r_hat_raw = 0.0
    for m in range(NUM_BANKS):
        rm_raw = float(r_banks[m])
        rm = _clamp_f64(rm_raw, R_FLOOR_M, R_CEIL_M) if math.isfinite(rm_raw) else R_INIT_M[m]
        vm_raw = float(var_r_banks[m])
        vm = _clamp_f64(vm_raw, 0.0, 1.0e8) if (math.isfinite(vm_raw) and vm_raw >= 0.0) else ((0.28 * rm) ** 2)
        r_clean[m] = rm
        v_clean[m] = vm
        r_hat_raw += w_out[m] * rm

    r_hat = _clamp_f64(r_hat_raw, R_FLOOR_M, R_CEIL_M)
    var_tot = 0.0
    for m in range(NUM_BANKS):
        dr = r_clean[m] - r_hat
        var_tot += w_out[m] * (v_clean[m] + dr * dr)
    sigma_r = math.sqrt(var_tot) if (math.isfinite(var_tot) and var_tot > 0.0) else 0.0
    return (w_out[0], w_out[1], w_out[2], w_out[3], w_out[4], w_out[5]), r_hat, sigma_r


def _project_accel_to_los(a_ned: tuple[float, float, float], alpha: float, beta: float) -> tuple[float, float, float]:
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    sb = math.sin(beta)
    cb = math.cos(beta)
    a_los = a_ned[0] * (cb * ca) + a_ned[1] * (cb * sa) + a_ned[2] * (-sb)
    a_az = a_ned[0] * (-sa) + a_ned[1] * ca
    a_el = a_ned[0] * (-sb * ca) + a_ned[1] * (-sb * sa) + a_ned[2] * (-cb)
    return a_los, a_az, a_el


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm3(a: tuple[float, float, float]) -> float:
    s = _dot3(a, a)
    return math.sqrt(s) if s > 0.0 else 0.0


@dataclass(frozen=True)
class _LosTriadNed:
    u_r: tuple[float, float, float]
    u_alpha: tuple[float, float, float]
    u_beta: tuple[float, float, float]
    cos_beta: float

    @staticmethod
    def from_angles(alpha_rad: float, beta_rad: float) -> _LosTriadNed:
        sa = math.sin(alpha_rad)
        ca = math.cos(alpha_rad)
        sb = math.sin(beta_rad)
        cb = math.cos(beta_rad)
        return _LosTriadNed(
            u_r=(cb * ca, cb * sa, -sb),
            u_alpha=(-sa, ca, 0.0),
            u_beta=(-sb * ca, -sb * sa, -cb),
            cos_beta=cb,
        )


def _sc3bf_eta(r_h: float, v_rel_norm: float) -> float:
    excess = (r_h - R_SEP_M) if r_h > R_SEP_M else 0.0
    v_eff = v_rel_norm if v_rel_norm > 1.0 else 1.0
    return K_ALLOW * (excess * excess) * A_MAX_EVADE_MPS2 / v_eff


def _sc3bf_cos_theta_c(r_h: float) -> float:
    if r_h <= R_SEP_M:
        return 0.0
    denom = r_h if r_h > (R_SEP_M + 1.0) else (R_SEP_M + 1.0)
    ratio = R_SEP_M / denom
    arg = 1.0 - ratio * ratio
    return math.sqrt(arg) if arg > 0.0 else 0.0


def _delta_v_own_ned(
    tan_phi: float,
    tas_mps: float,
    own_fwd_ned: tuple[float, float, float],
    own_right_ned: tuple[float, float, float],
) -> tuple[float, float, float]:
    v_eff = tas_mps if tas_mps > 1.0 else 1.0
    dpsi = _clamp_f64((G_ACCEL_MPS2 * tan_phi / v_eff) * DT_LOOK_S, -0.5, 0.5)
    dpsi2 = dpsi * dpsi
    c_minus_1 = -0.5 * dpsi2 + (dpsi2 * dpsi2) / 24.0
    s_approx = dpsi - (dpsi2 * dpsi) / 6.0
    return (
        v_eff * (c_minus_1 * own_fwd_ned[0] + s_approx * own_right_ned[0]),
        v_eff * (c_minus_1 * own_fwd_ned[1] + s_approx * own_right_ned[1]),
        v_eff * (c_minus_1 * own_fwd_ned[2] + s_approx * own_right_ned[2]),
    )


@dataclass(frozen=True)
class SwerveDecision:
    """Decision emitted by universal-range `SC3BF-FIM` selector."""

    mode: ActiveRangeMode
    cmd_bank_rad: float
    min_sc3bf_margin: float
    fim_gain: float
    sc3bf_all_safe: bool
    fov_safe: bool
    now_locked: bool


def select_active_swerve(
    *,
    r_banks: tuple[float, ...],
    weights: tuple[float, ...],
    r_hat_m: float,
    sigma_r_m: float,
    r_safe_lower_m: float,
    rho_dot_inv_s: float,
    alpha_dot: float,
    beta_dot: float,
    alpha_body_rad: float,
    alpha_rad: float,
    beta_rad: float,
    heading_rad: float,
    tas_mps: float,
    v_stall_mps: float,
    sigma_bearing_rad: float,
    cbdr_probe_active: bool,
    was_locked: bool,
    preferred_swerve_sign: float = 1.0,
) -> SwerveDecision:
    """Bit-exact port of Rust `select_active_swerve` (`Kani #16`)."""
    triad = _LosTriadNed.from_angles(alpha_rad, beta_rad)
    sh = math.sin(heading_rad)
    ch = math.cos(heading_rad)
    own_fwd_ned = (ch, sh, 0.0)
    own_right_ned = (-sh, ch, 0.0)

    def _eval_h(r_h: float, tan_phi: float) -> float:
        r = r_h if r_h > R_FLOOR_M else R_FLOOR_M
        p_rel = (r * triad.u_r[0], r * triad.u_r[1], r * triad.u_r[2])
        v_r = r * rho_dot_inv_s
        v_a = r * alpha_dot * triad.cos_beta
        v_b = r * beta_dot
        dv = _delta_v_own_ned(tan_phi, tas_mps, own_fwd_ned, own_right_ned)
        v_rel = (
            v_r * triad.u_r[0] + v_a * triad.u_alpha[0] + v_b * triad.u_beta[0] - dv[0],
            v_r * triad.u_r[1] + v_a * triad.u_alpha[1] + v_b * triad.u_beta[1] - dv[1],
            v_r * triad.u_r[2] + v_a * triad.u_alpha[2] + v_b * triad.u_beta[2] - dv[2],
        )
        v_norm = _norm3(v_rel)
        pv = _dot3(p_rel, v_rel)
        return pv + r * v_norm * _sc3bf_cos_theta_c(r) + _sc3bf_eta(r, v_norm)

    def _hyp_margin_safe(r_h: float, tan_phi: float) -> tuple[float, bool]:
        h_0 = _eval_h(r_h, 0.0)
        h_phi = _eval_h(r_h, tan_phi)
        if h_0 >= 0.0:
            margin = h_phi - (1.0 - ALPHA_CBF * DT_LOOK_S) * h_0
            return margin, (math.isfinite(margin) and margin >= 0.0)
        margin = h_phi - h_0
        return margin, (math.isfinite(margin) and margin > 0.0)

    def _fov_safe(tan_phi: float) -> bool:
        v_eff = tas_mps if tas_mps > 1.0 else 1.0
        dpsi = _clamp_f64((G_ACCEL_MPS2 * tan_phi / v_eff) * DT_LOOK_S, -0.5, 0.5)
        pred_ab = alpha_body_rad - dpsi + alpha_dot * DT_LOOK_S
        return math.isfinite(pred_ab) and abs(pred_ab) <= FOV_LIMIT_RAD

    def _universal(tan_phi: float) -> tuple[float, bool]:
        min_m = 1.0e300
        all_ok = True
        for m in range(NUM_BANKS):
            if weights[m] >= W_ACTIVE:
                mg, ok = _hyp_margin_safe(r_banks[m], tan_phi)
                if mg < min_m:
                    min_m = mg
                if not ok:
                    all_ok = False
        mg_low, ok_low = _hyp_margin_safe(r_safe_lower_m, tan_phi)
        if mg_low < min_m:
            min_m = mg_low
        if not ok_low:
            all_ok = False
        if not math.isfinite(min_m):
            return -1.0e6, False
        return min_m, all_ok

    evade_bank = -EVADE_BANK_RAD if alpha_body_rad > 0.01 else EVADE_BANK_RAD

    r_hat_eff = r_hat_m if r_hat_m > R_FLOOR_M else R_FLOOR_M
    rel_sigma = (sigma_r_m / r_hat_eff) if (math.isfinite(sigma_r_m) and sigma_r_m >= 0.0) else 1.0
    now_locked = (rel_sigma <= TRIGGER_REL_SIGMA) if was_locked else (rel_sigma <= LOCK_REL_SIGMA)

    is_closing = rho_dot_inv_s < CLOSING_RHO_DOT_THRESH
    tau_terminal = rho_dot_inv_s <= -(1.0 / TAU_EVADE_TRIGGER_S)
    h_low_zero = _eval_h(r_safe_lower_m, 0.0)
    range_terminal = (r_hat_m <= R_EVADE_TRIGGER_M) or (r_safe_lower_m <= R_EVADE_TRIGGER_M and rel_sigma <= 0.35)

    if is_closing and (tau_terminal or range_terminal) and h_low_zero < 0.0:
        marg0, _ = _universal(0.0)
        return SwerveDecision(
            mode=ActiveRangeMode.IMMEDIATE_EVADE,
            cmd_bank_rad=evade_bank,
            min_sc3bf_margin=marg0,
            fim_gain=0.0,
            sc3bf_all_safe=False,
            fov_safe=_fov_safe(0.0),
            now_locked=now_locked,
        )

    envelope_ok = math.isfinite(tas_mps) and math.isfinite(v_stall_mps) and (tas_mps >= 1.25 * v_stall_mps)
    marg0, safe0 = _universal(0.0)
    fov0 = _fov_safe(0.0)

    if not (is_closing or cbdr_probe_active):
        return SwerveDecision(
            mode=ActiveRangeMode.LOCKED if now_locked else ActiveRangeMode.NOMINAL_HOLD,
            cmd_bank_rad=0.0,
            min_sc3bf_margin=marg0,
            fim_gain=0.0,
            sc3bf_all_safe=safe0,
            fov_safe=fov0,
            now_locked=now_locked,
        )

    if now_locked:
        return SwerveDecision(
            mode=ActiveRangeMode.LOCKED,
            cmd_bank_rad=0.0,
            min_sc3bf_margin=marg0,
            fim_gain=0.0,
            sc3bf_all_safe=safe0,
            fov_safe=fov0,
            now_locked=True,
        )

    if not envelope_ok:
        return SwerveDecision(
            mode=ActiveRangeMode.NOMINAL_HOLD,
            cmd_bank_rad=0.0,
            min_sc3bf_margin=marg0,
            fim_gain=0.0,
            sc3bf_all_safe=safe0,
            fov_safe=fov0,
            now_locked=False,
        )

    sig_a = sigma_bearing_rad if sigma_bearing_rad > 1.0e-5 else 1.0e-5
    t_swerve = 2.5
    t4_over_12s2 = (t_swerve ** 4) / (12.0 * sig_a * sig_a)
    lambda_u = 1.0e-2
    lambda_margin = 1.0e-4

    found_swerve = False
    best_k = 4
    best_score = -1.0e300
    best_margin = marg0
    best_fim = 0.0
    right_dot_ualpha = _dot3(own_right_ned, triad.u_alpha)

    for k in range(len(PHI_CAND_RAD)):
        phi = PHI_CAND_RAD[k]
        tan_phi = PHI_CAND_TAN[k]
        fov_ok = _fov_safe(tan_phi)
        min_marg, all_safe = _universal(tan_phi)
        if fov_ok and all_safe:
            a_az_cand = G_ACCEL_MPS2 * tan_phi * right_dot_ualpha
            a_az2 = a_az_cand * a_az_cand
            fim_gain = 0.0
            for m in range(NUM_BANKS):
                rm = r_banks[m] if r_banks[m] > R_FLOOR_M else R_FLOOR_M
                fim_gain += weights[m] * (a_az2 / (rm * rm)) * t4_over_12s2
            pref_sign = -1.0 if preferred_swerve_sign < 0.0 else 1.0
            if abs(alpha_body_rad) < 0.07:
                row_tiebreak = 1.0e-3 * pref_sign * phi
            elif alpha_body_rad > 0.0:
                row_tiebreak = -1.0e-3 * phi
            else:
                row_tiebreak = 1.0e-3 * phi
            score = fim_gain - lambda_u * phi * phi + lambda_margin * min_marg + row_tiebreak
            if math.isfinite(score) and ((not found_swerve) or score > best_score):
                found_swerve = True
                best_k = k
                best_score = score
                best_margin = min_marg
                best_fim = fim_gain

    if not found_swerve:
        return SwerveDecision(
            mode=ActiveRangeMode.IMMEDIATE_EVADE,
            cmd_bank_rad=evade_bank,
            min_sc3bf_margin=marg0,
            fim_gain=0.0,
            sc3bf_all_safe=False,
            fov_safe=fov0,
            now_locked=False,
        )

    if best_k == 4:
        return SwerveDecision(
            mode=ActiveRangeMode.NOMINAL_HOLD,
            cmd_bank_rad=0.0,
            min_sc3bf_margin=best_margin,
            fim_gain=best_fim,
            sc3bf_all_safe=True,
            fov_safe=True,
            now_locked=False,
        )

    return SwerveDecision(
        mode=ActiveRangeMode.ACTIVE_SWERVE,
        cmd_bank_rad=PHI_CAND_RAD[best_k],
        min_sc3bf_margin=best_margin,
        fim_gain=best_fim,
        sc3bf_all_safe=True,
        fov_safe=True,
        now_locked=False,
    )


class _MscBank:
    """Bit-exact port of Rust `MscBank` (`msc_rpekf.rs`)."""

    def __init__(self, alpha_rad: float, beta_rad: float, r_init_m: float) -> None:
        r0 = _clamp_f64(r_init_m, R_FLOOR_M, R_CEIL_M)
        rho0 = 1.0 / r0
        self.x = np.array([alpha_rad, beta_rad, 0.0, 0.0, rho0, 0.0], dtype=np.float64)
        self.p = np.zeros((6, 6), dtype=np.float64)
        self.p[0, 0] = 0.002 * 0.002
        self.p[1, 1] = 0.002 * 0.002
        self.p[2, 2] = 0.0025 * 0.0025
        self.p[3, 3] = 0.0025 * 0.0025
        sig_rho = 0.28 * rho0
        self.p[4, 4] = sig_rho * sig_rho
        self.p[5, 5] = 0.15 * 0.15

    def range_and_variance(self) -> tuple[float, float]:
        rho = _clamp_f64(float(self.x[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        r_m = _clamp_f64(1.0 / rho, R_FLOOR_M, R_CEIL_M)
        rho2 = rho * rho
        rho4 = rho2 * rho2
        p44 = float(self.p[4, 4])
        if math.isfinite(p44) and p44 >= 0.0:
            raw_var = p44 / rho4
        else:
            raw_var = (0.28 * r_m) ** 2
        max_var = (0.5 * r_m) ** 2
        var_r = _clamp_f64(raw_var, 1.0, max_var)
        return r_m, var_r

    @staticmethod
    def _rhs(x: np.ndarray, a_ned: tuple[float, float, float]) -> np.ndarray:
        alpha = float(x[0])
        beta = _clamp_f64(float(x[1]), -1.35, 1.35)
        a_dot = float(x[2])
        b_dot = float(x[3])
        rho = _clamp_f64(float(x[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        rho_dot = _clamp_f64(float(x[5]), -2.0, 2.0)

        a_los, a_az, a_el = _project_accel_to_los(a_ned, alpha, beta)
        sb = math.sin(beta)
        cb = math.cos(beta)
        cb_safe = cb if cb > 0.15 else 0.15
        tb = sb / cb_safe

        a_ddot = -2.0 * a_dot * rho_dot + 2.0 * a_dot * b_dot * tb - (rho / cb_safe) * a_az
        b_ddot = -2.0 * b_dot * rho_dot - a_dot * a_dot * sb * cb - rho * a_el
        rho_d = -rho * rho_dot
        rho_dd = a_dot * a_dot * cb * cb + b_dot * b_dot - rho_dot * rho_dot - rho * a_los
        return np.array([a_dot, b_dot, a_ddot, b_ddot, rho_d, rho_dd], dtype=np.float64)

    @staticmethod
    def _jacobian(x: np.ndarray, a_ned: tuple[float, float, float]) -> np.ndarray:
        alpha = float(x[0])
        beta = _clamp_f64(float(x[1]), -1.35, 1.35)
        a_dot = float(x[2])
        b_dot = float(x[3])
        rho = _clamp_f64(float(x[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        rho_dot = _clamp_f64(float(x[5]), -2.0, 2.0)

        a_los, a_az, a_el = _project_accel_to_los(a_ned, alpha, beta)
        sb = math.sin(beta)
        cb = math.cos(beta)
        cb_safe = cb if cb > 0.15 else 0.15
        tb = sb / cb_safe
        sec2 = 1.0 / (cb_safe * cb_safe)

        j = np.zeros((6, 6), dtype=np.float64)
        j[0, 2] = 1.0
        j[1, 3] = 1.0
        j[2, 1] = 2.0 * a_dot * b_dot * sec2 - rho * a_az * sb * sec2
        j[2, 2] = -2.0 * rho_dot + 2.0 * b_dot * tb
        j[2, 3] = 2.0 * a_dot * tb
        j[2, 4] = -a_az / cb_safe
        j[2, 5] = -2.0 * a_dot
        j[3, 1] = -a_dot * a_dot * (cb * cb - sb * sb)
        j[3, 2] = -2.0 * a_dot * sb * cb
        j[3, 3] = -2.0 * rho_dot
        j[3, 4] = -a_el
        j[3, 5] = -2.0 * b_dot
        j[4, 4] = -rho_dot
        j[4, 5] = -rho
        j[5, 1] = -2.0 * a_dot * a_dot * cb * sb
        j[5, 2] = 2.0 * a_dot * cb * cb
        j[5, 3] = 2.0 * b_dot
        j[5, 4] = -a_los
        j[5, 5] = -2.0 * rho_dot
        return j

    def propagate(self, a_ned: tuple[float, float, float], dt_s: float) -> None:
        dt = _clamp_f64(dt_s, 1.0e-3, 0.5)
        k1 = self._rhs(self.x, a_ned)
        x_mid = self.x + 0.5 * dt * k1
        x_mid[0] = _wrap_pi(float(x_mid[0]))
        x_mid[1] = _clamp_f64(float(x_mid[1]), -1.35, 1.35)
        x_mid[4] = _clamp_f64(float(x_mid[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        x_mid[5] = _clamp_f64(float(x_mid[5]), -2.0, 2.0)

        k2 = self._rhs(x_mid, a_ned)
        x_next = self.x + dt * k2
        x_next[0] = _wrap_pi(float(x_next[0]))
        x_next[1] = _clamp_f64(float(x_next[1]), -1.35, 1.35)
        x_next[4] = _clamp_f64(float(x_next[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        x_next[5] = _clamp_f64(float(x_next[5]), -2.0, 2.0)

        j = self._jacobian(x_mid, a_ned)
        j2 = j @ j
        f_mat = np.eye(6, dtype=np.float64) + dt * j + (0.5 * dt * dt) * j2
        p_next = (f_mat @ self.p) @ f_mat.T

        q_ang_acc = 0.0003 * 0.0003
        q_ang_pos = 0.25 * q_ang_acc * (dt ** 4) + 1.0e-10 * dt
        q_ang_vel = q_ang_acc * (dt ** 2) + 1.0e-9 * dt
        q_rho = (1.0e-6 * dt) ** 2
        q_rhodot = (0.01 * dt) ** 2 + 1.0e-6 * dt

        p_next[0, 0] += q_ang_pos
        p_next[1, 1] += q_ang_pos
        p_next[2, 2] += q_ang_vel
        p_next[3, 3] += q_ang_vel
        p_next[4, 4] += q_rho
        p_next[5, 5] += q_rhodot

        p_next = 0.5 * (p_next + p_next.T)
        p_next = np.where(np.isfinite(p_next), p_next, 0.0)
        for r in range(6):
            if p_next[r, r] < 1.0e-14:
                p_next[r, r] = 1.0e-14

        self.x = x_next
        self.p = p_next

    def update_bearing(self, alpha_meas: float, beta_meas: float, sigma_bearing_rad: float) -> float:
        r_var = sigma_bearing_rad * sigma_bearing_rad
        nu0 = _wrap_pi(alpha_meas - float(self.x[0]))
        nu1 = _clamp_f64(beta_meas, -1.35, 1.35) - float(self.x[1])

        s00 = float(self.p[0, 0]) + r_var
        s01 = float(self.p[0, 1])
        s11 = float(self.p[1, 1]) + r_var
        det_s = s00 * s11 - s01 * s01
        if (not math.isfinite(det_s)) or det_s <= 1.0e-20:
            return -25.0
        inv_det = 1.0 / det_s
        inv00 = s11 * inv_det
        inv01 = -s01 * inv_det
        inv11 = s00 * inv_det

        maha = nu0 * (inv00 * nu0 + inv01 * nu1) + nu1 * (inv01 * nu0 + inv11 * nu1)
        log_lik = -0.5 * (maha + math.log(det_s))

        k0 = self.p[:, 0] * inv00 + self.p[:, 1] * inv01
        k1 = self.p[:, 0] * inv01 + self.p[:, 1] * inv11
        self.x = self.x + k0 * nu0 + k1 * nu1
        self.x[0] = _wrap_pi(float(self.x[0]))
        self.x[1] = _clamp_f64(float(self.x[1]), -1.35, 1.35)
        self.x[4] = _clamp_f64(float(self.x[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        self.x[5] = _clamp_f64(float(self.x[5]), -2.0, 2.0)

        ikh = np.eye(6, dtype=np.float64)
        ikh[:, 0] -= k0
        ikh[:, 1] -= k1
        p_new = (ikh @ self.p) @ ikh.T
        krk = r_var * (np.outer(k0, k0) + np.outer(k1, k1))
        p_new = 0.5 * (p_new + p_new.T) + krk
        p_new = np.where(np.isfinite(p_new), p_new, 0.0)
        for r in range(6):
            if p_new[r, r] < 1.0e-14:
                p_new[r, r] = 1.0e-14
        self.p = p_new
        return log_lik if math.isfinite(log_lik) else -1.0e6

    def update_tau(self, z_tau: float, r_tau: float) -> None:
        nu = z_tau - float(self.x[5])
        s = float(self.p[5, 5]) + r_tau
        if (not math.isfinite(s)) or s <= 1.0e-16:
            return
        k = self.p[:, 5] / s
        self.x = self.x + k * nu
        self.x[0] = _wrap_pi(float(self.x[0]))
        self.x[1] = _clamp_f64(float(self.x[1]), -1.35, 1.35)
        self.x[4] = _clamp_f64(float(self.x[4]), 1.0 / R_CEIL_M, 1.0 / R_FLOOR_M)
        self.x[5] = _clamp_f64(float(self.x[5]), -2.0, 2.0)

        ikh = np.eye(6, dtype=np.float64)
        ikh[:, 5] -= k
        p_new = (ikh @ self.p) @ ikh.T
        p_new = 0.5 * (p_new + p_new.T) + r_tau * np.outer(k, k)
        p_new = np.where(np.isfinite(p_new), p_new, 0.0)
        for r in range(6):
            if p_new[r, r] < 1.0e-14:
                p_new[r, r] = 1.0e-14
        self.p = p_new


def _sanitize_ownship(own: OwnshipStateNed) -> OwnshipStateNed:
    def _clean(val: float, fallback: float) -> float:
        return float(val) if math.isfinite(val) else fallback

    return OwnshipStateNed(
        v_ned=(_clean(own.v_ned[0], 0.0), _clean(own.v_ned[1], 0.0), _clean(own.v_ned[2], 0.0)),
        a_ned=(
            _clamp_f64(_clean(own.a_ned[0], 0.0), -25.0, 25.0),
            _clamp_f64(_clean(own.a_ned[1], 0.0), -25.0, 25.0),
            _clamp_f64(_clean(own.a_ned[2], 0.0), -25.0, 25.0),
        ),
        heading_rad=_wrap_pi(_clean(own.heading_rad, 0.0)),
        tas_mps=_clamp_f64(_clean(own.tas_mps, 25.0), 1.0, 250.0),
        v_stall_mps=_clamp_f64(_clean(own.v_stall_mps, 14.5), 1.0, 100.0),
    )


class ActiveRangeController:
    """Bit-exact Python mirror of Rust `ActiveRangeController` (`xvs-active-range/src/lib.rs`)."""

    def __init__(self, cfg: ActiveRangeConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ActiveRangeConfig()
        self._initialized: bool = False
        self._banks = [_MscBank(0.0, 0.0, self.cfg.r_init_m[m]) for m in range(NUM_BANKS)]
        self._weights: tuple[float, float, float, float, float, float] = (
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        )
        self._seq: int = 0
        self._swerve_ticks: int = 0
        self._was_locked: bool = False
        self._last_alpha_rad: float = 0.0
        self._last_beta_rad: float = 0.0

    def _current_moments(self) -> tuple[float, float, list[float], int, float]:
        r_banks = [0.0] * NUM_BANKS
        v_banks = [0.0] * NUM_BANKS
        r_hat = 0.0
        rho_dot = 0.0
        active_idx = 0
        max_w = -1.0
        for m in range(NUM_BANKS):
            rm, vm = self._banks[m].range_and_variance()
            r_banks[m] = rm
            v_banks[m] = vm
            wm = self._weights[m]
            if wm > max_w:
                max_w = wm
                active_idx = m
            r_hat += wm * rm
            rho_dot += wm * float(self._banks[m].x[5])
        r_hat = _clamp_f64(r_hat, R_FLOOR_M, R_CEIL_M)
        var_tot = 0.0
        for m in range(NUM_BANKS):
            dr = r_banks[m] - r_hat
            var_tot += self._weights[m] * (v_banks[m] + dr * dr)
        sigma_r = math.sqrt(var_tot) if (math.isfinite(var_tot) and var_tot > 0.0) else 0.0
        return r_hat, sigma_r, r_banks, active_idx, _clamp_f64(rho_dot, -2.0, 2.0)

    def _pack_telemetry(
        self,
        *,
        mode: ActiveRangeMode,
        active_bank_idx: int,
        flags: int,
        r_hat_m: float,
        sigma_r_m: float,
        r_safe_lower_m: float,
        rho_dot_inv_s: float,
        cmd_bank_rad: float,
        min_sc3bf_margin: float,
        fim_gain: float,
        weights: tuple[float, float, float, float, float, float],
    ) -> ActiveRangeTelemetry:
        w32 = (
            _f32(weights[0]),
            _f32(weights[1]),
            _f32(weights[2]),
            _f32(weights[3]),
            _f32(weights[4]),
            _f32(weights[5]),
        )
        pkt = ActiveRangeTelemetry(
            seq=self._seq,
            mode=mode,
            active_bank_idx=active_bank_idx,
            flags=flags,
            r_hat_m=_f32(r_hat_m),
            sigma_r_m=_f32(sigma_r_m),
            r_safe_lower_m=_f32(r_safe_lower_m),
            rho_dot_inv_s=_f32(rho_dot_inv_s),
            cmd_bank_rad=_f32(cmd_bank_rad),
            min_sc3bf_margin=_f32(min_sc3bf_margin),
            fim_gain=_f32(fim_gain),
            weights=w32,
            crc32=0,
        )
        wire = encode_wire(pkt)
        crc = int(struct.unpack_from("<I", wire, 60)[0])
        return ActiveRangeTelemetry(
            seq=pkt.seq,
            mode=pkt.mode,
            active_bank_idx=pkt.active_bank_idx,
            flags=pkt.flags,
            r_hat_m=pkt.r_hat_m,
            sigma_r_m=pkt.sigma_r_m,
            r_safe_lower_m=pkt.r_safe_lower_m,
            rho_dot_inv_s=pkt.rho_dot_inv_s,
            cmd_bank_rad=pkt.cmd_bank_rad,
            min_sc3bf_margin=pkt.min_sc3bf_margin,
            fim_gain=pkt.fim_gain,
            weights=pkt.weights,
            crc32=crc,
        )

    def step(
        self,
        own: OwnshipStateNed,
        meas: ActiveRangeMeasurement,
    ) -> ActiveRangeTelemetry:
        """Advance MSC-RPEKF + tau + Koschmieder + CP-MSC + SC3BF-FIM by one deterministic step."""
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        own_safe = _sanitize_ownship(own)

        inputs_valid = (
            math.isfinite(meas.alpha_rad)
            and math.isfinite(meas.beta_rad)
            and math.isfinite(meas.dt_s)
            and meas.dt_s > 0.0
        )
        if not inputs_valid:
            r_hat_m, sigma_r_m, _r_banks, active_idx, rho_dot = self._current_moments()
            r_safe = compute_conformal_lower_bound(r_hat_m, sigma_r_m, self.cfg.q_hat_conformal)
            return self._pack_telemetry(
                mode=ActiveRangeMode.NOMINAL_HOLD,
                active_bank_idx=active_idx,
                flags=0,
                r_hat_m=r_hat_m,
                sigma_r_m=sigma_r_m,
                r_safe_lower_m=r_safe,
                rho_dot_inv_s=rho_dot,
                cmd_bank_rad=0.0,
                min_sc3bf_margin=0.0,
                fim_gain=0.0,
                weights=self._weights,
            )

        alpha_m = _wrap_pi(float(meas.alpha_rad))
        beta_m = _clamp_f64(float(meas.beta_rad), -1.35, 1.35)
        dt_s = _clamp_f64(float(meas.dt_s), 1.0e-3, 0.5)

        log_lik = [0.0] * NUM_BANKS
        flags = 0

        if not self._initialized:
            for m in range(NUM_BANKS):
                self._banks[m] = _MscBank(alpha_m, beta_m, self.cfg.r_init_m[m])
            self._weights = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
            self._initialized = True
        else:
            for m in range(NUM_BANKS):
                self._banks[m].propagate(own_safe.a_ned, dt_s)
                log_lik[m] = self._banks[m].update_bearing(alpha_m, beta_m, self.cfg.sigma_bearing_rad)
            ll_max = max(log_lik)
            for m in range(NUM_BANKS):
                log_lik[m] = _clamp_f64(log_lik[m] - ll_max, -25.0, 0.0)

        # Scale-cancelling optical expansion `tau`: z_tau = -s_dot / s
        if (
            math.isfinite(meas.size_px)
            and math.isfinite(meas.size_rate_px_s)
            and meas.size_px >= TAU_MIN_SIZE_PX
            and int(meas.track_age) >= TAU_MIN_TRACK_AGE
        ):
            z_tau = -float(meas.size_rate_px_s) / float(meas.size_px)
            if math.isfinite(z_tau) and abs(z_tau) <= 1.0:
                s2 = meas.size_px * meas.size_px
                r_tau = (2.0 * 0.64) / (s2 * 0.25) + 0.015 * 0.015
                for m in range(NUM_BANKS):
                    self._banks[m].update_tau(z_tau, r_tau)
                flags |= FLAG_TAU_FUSED

        # Gated weak Koschmieder atmospheric contrast prior
        _a_los, a_az, _a_el = _project_accel_to_los(own_safe.a_ned, alpha_m, beta_m)
        if (
            meas.daytime_clear
            and math.isfinite(meas.size_px)
            and meas.size_px < KOSCH_MAX_SIZE_PX
            and beta_m > 0.0
            and math.isfinite(meas.sky_beta_ext)
            and (5.0e-5 <= meas.sky_beta_ext <= 2.0e-3)
            and math.isfinite(meas.contrast_ratio)
            and (0.01 < meas.contrast_ratio < 0.99)
            and abs(a_az) < 0.25
        ):
            max_delta = KOSCH_MAX_LOG_DELTA * dt_s
            for m in range(NUM_BANKS):
                r_m, _ = self._banks[m].range_and_variance()
                c_pred = math.exp(-meas.sky_beta_ext * r_m)
                if meas.contrast_ratio > c_pred:
                    diff = (meas.contrast_ratio - c_pred) / 0.25
                    delta = _clamp_f64(-0.5 * diff * diff * dt_s, -max_delta, 0.0)
                    log_lik[m] += delta
            flags |= FLAG_KOSCH_FUSED

        r_banks_l = [0.0] * NUM_BANKS
        var_r_l = [0.0] * NUM_BANKS
        for m in range(NUM_BANKS):
            rm, vm = self._banks[m].range_and_variance()
            r_banks_l[m] = rm
            var_r_l[m] = vm

        w_new, r_hat_m, sigma_r_m = normalize_weights_and_moments(log_lik, r_banks_l, var_r_l, self._weights)
        self._weights = w_new

        active_idx = 0
        max_w = -1.0
        rho_dot_mix = 0.0
        a_dot_mix = 0.0
        b_dot_mix = 0.0
        alpha_mix_s = 0.0
        alpha_mix_c = 0.0
        beta_mix = 0.0
        for m in range(NUM_BANKS):
            wm = w_new[m]
            if wm > max_w:
                max_w = wm
                active_idx = m
            rho_dot_mix += wm * float(self._banks[m].x[5])
            a_dot_mix += wm * float(self._banks[m].x[2])
            b_dot_mix += wm * float(self._banks[m].x[3])
            alpha_mix_s += wm * math.sin(float(self._banks[m].x[0]))
            alpha_mix_c += wm * math.cos(float(self._banks[m].x[0]))
            beta_mix += wm * float(self._banks[m].x[1])

        rho_dot_inv_s = _clamp_f64(rho_dot_mix, -2.0, 2.0)
        alpha_mix = math.atan2(alpha_mix_s, alpha_mix_c)
        beta_mix = _clamp_f64(beta_mix, -1.35, 1.35)
        self._last_alpha_rad = alpha_mix
        self._last_beta_rad = beta_mix

        r_safe_lower_m = compute_conformal_lower_bound(r_hat_m, sigma_r_m, self.cfg.q_hat_conformal)
        alpha_body_rad = _wrap_pi(alpha_mix - own_safe.heading_rad)

        cbdr_probe_active = (
            int(meas.track_age) >= TAU_MIN_TRACK_AGE
            and math.isfinite(meas.size_px)
            and meas.size_px < TAU_MIN_SIZE_PX
            and math.isfinite(meas.size_rate_px_s)
            and meas.size_rate_px_s > 0.0
            and abs(a_dot_mix) < 0.004
            and abs(b_dot_mix) < 0.004
        )

        preferred_swerve_sign = 1.0 if ((self._swerve_ticks // 12) % 2) == 0 else -1.0

        decision = select_active_swerve(
            r_banks=tuple(r_banks_l),
            weights=w_new,
            r_hat_m=r_hat_m,
            sigma_r_m=sigma_r_m,
            r_safe_lower_m=r_safe_lower_m,
            rho_dot_inv_s=rho_dot_inv_s,
            alpha_dot=a_dot_mix,
            beta_dot=b_dot_mix,
            alpha_body_rad=alpha_body_rad,
            alpha_rad=alpha_mix,
            beta_rad=beta_mix,
            heading_rad=own_safe.heading_rad,
            tas_mps=own_safe.tas_mps,
            v_stall_mps=own_safe.v_stall_mps,
            sigma_bearing_rad=self.cfg.sigma_bearing_rad,
            cbdr_probe_active=cbdr_probe_active,
            was_locked=self._was_locked,
            preferred_swerve_sign=preferred_swerve_sign,
        )
        self._was_locked = decision.now_locked
        if decision.mode == ActiveRangeMode.ACTIVE_SWERVE:
            self._swerve_ticks = (self._swerve_ticks + 1) & 0xFFFFFFFF
        else:
            self._swerve_ticks = 0

        if decision.sc3bf_all_safe:
            flags |= FLAG_SC3BF_ALL_SAFE
        if decision.fov_safe:
            flags |= FLAG_FOV_SAFE

        return self._pack_telemetry(
            mode=decision.mode,
            active_bank_idx=active_idx,
            flags=flags,
            r_hat_m=r_hat_m,
            sigma_r_m=sigma_r_m,
            r_safe_lower_m=r_safe_lower_m,
            rho_dot_inv_s=rho_dot_inv_s,
            cmd_bank_rad=decision.cmd_bank_rad,
            min_sc3bf_margin=decision.min_sc3bf_margin,
            fim_gain=decision.fim_gain,
            weights=w_new,
        )
