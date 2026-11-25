"""DR-10 Neuromorphic IMX636 Event-Camera + EO Asymmetric Fusion & FRIES Rotor Discriminator.

Clean-room Apache-2.0 / MIT implementation grounded in OpenXVS DR-10 §1–§3:
  1. Physical optics & CBDR Paradox solver (`IMX636` 1280x720, 4.86 um pitch, 12 mm lens,
     IFOV = 0.405 mrad) + rigid 35–50 mm co-located baseline parallax (`H_inf`, zero -3 dB
     beam-splitter loss) + asymmetric EO-anchor vs 1 kHz EVS fast-track router.
  2. IMX636 Hardware Event Signal Processing (ESP) guard: EVT 3.0 bus ceilings
     (`EVK4=100 Meps`, `Triton2 2.5GigE=138 Meps`, `IDS XCP-E=200 Meps`), ERC register
     `cd_event_count = target_rate_eps // 5000`, and Selective Narrow-Notch AFK enforcement
     (strictly rejecting broadband 50–500 Hz AFK that erases 66.7–450 Hz propeller BPF).
  3. FRIES propeller-harmonic spectral discriminator (`66.7–450 Hz` civil rotor BPF band via
     windowed `np.fft.rfft` + parabolic peak interpolation, `< 3%` frequency accuracy).

Zero GPL (`DAGr`/`Tonic`/`IEBCS`), zero AGPL (`YOLO`), zero `v2e`, zero `SAST`, zero `.get()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

# IMX636 physical sensor defaults (DR-10 §1)
IMX636_WIDTH_PX: int = 1280
IMX636_HEIGHT_PX: int = 720
IMX636_PIXEL_PITCH_M: float = 4.86e-6
IMX636_DEFAULT_FOCAL_M: float = 12.0e-3
IMX636_DEFAULT_IFOV_RAD: float = IMX636_PIXEL_PITCH_M / IMX636_DEFAULT_FOCAL_M  # 0.000405 rad

KNOTS_TO_MPS: float = 0.5144444444444445

# Civil UAS / GA propeller blade-pass frequency (BPF) band: 2,000–9,000 RPM (2–3 blades)
CIVIL_ROTOR_BPF_MIN_HZ: float = 66.7
CIVIL_ROTOR_BPF_MAX_HZ: float = 450.0

# IMX636 hardware Anti-Flicker (AFK) range & selective narrow-notch ceiling
IMX636_AFK_HW_MIN_HZ: float = 50.0
IMX636_AFK_HW_MAX_HZ: float = 500.0
DEFAULT_MAX_AFK_NOTCH_WIDTH_HZ: float = 25.0
ERC_DIVISOR: int = 5000


class SensorAnchorMode(str, Enum):
    """Asymmetric DAA primary tracking modality selected by CBDR & crossing kinematics."""

    EO_ANCHOR = "EO_ANCHOR"
    EVS_FAST_TRACK = "EVS_FAST_TRACK"


class Imx636Interface(str, Enum):
    """IMX636 camera interface models and their EVT 3.0 sustained bus ceilings (eps)."""

    EVK4_USB3 = "EVK4_USB3"
    TRITON2_2_5GIGE = "TRITON2_2_5GIGE"
    IDS_XCP_E_USB3 = "IDS_XCP_E_USB3"


_EVT3_BUS_CEILING_EPS: dict[Imx636Interface, int] = {
    Imx636Interface.EVK4_USB3: 100_000_000,
    Imx636Interface.TRITON2_2_5GIGE: 138_000_000,
    Imx636Interface.IDS_XCP_E_USB3: 200_000_000,
}


class RotorClass(str, Enum):
    """FRIES spectral classification verdict for civil DAA clutter rejection."""

    ROTORCRAFT_OR_PROP_AIRCRAFT = "ROTORCRAFT_OR_PROP_AIRCRAFT"
    NON_ROTATING_CLUTTER = "NON_ROTATING_CLUTTER"


@dataclass(frozen=True)
class CbdrExpansionResult:
    """Physical CBDR edge-expansion kinematics for a target of width W closing at v_close."""

    edge_expansion_px_s: float
    seconds_per_px: float
    apparent_span_px: float
    ttc_s: float
    blind_gap_10hz_m: float
    blind_gap_1khz_m: float


@dataclass(frozen=True)
class AsymmetricAnchorDecision:
    """Routing decision between Global-Shutter EO Anchor and 1 kHz IMX636 EVS Fast-Track."""

    mode: SensorAnchorMode
    cbdr: CbdrExpansionResult
    parallax_px: float
    pure_rotational_homography_valid: bool
    reason: str


@dataclass(frozen=True)
class AfkNotchBand:
    """Selective narrow-notch Anti-Flicker (AFK) filter band in Hz."""

    low_hz: float
    high_hz: float
    label: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low_hz) and math.isfinite(self.high_hz)):
            raise ValueError("AFK notch frequencies must be finite")
        if self.low_hz >= self.high_hz:
            raise ValueError(
                f"AFK notch low_hz ({self.low_hz}) must be < high_hz ({self.high_hz})"
            )

    @property
    def width_hz(self) -> float:
        return self.high_hz - self.low_hz


@dataclass(frozen=True)
class EspConfigResult:
    """Validated IMX636 hardware ESP register configuration (ERC + selective AFK)."""

    interface: Imx636Interface
    bus_ceiling_eps: int
    target_rate_eps: int
    erc_cd_event_count: int
    afk_enabled: bool
    afk_notches: tuple[AfkNotchBand, ...]
    accepted: bool
    reason: str


@dataclass(frozen=True)
class FriesSpectrumResult:
    """FRIES propeller-harmonic spectral discrimination output."""

    classification: RotorClass
    estimated_bpf_hz: float
    peak_snr_db: float
    in_band_energy_ratio: float
    harmonic_2x_confirmed: bool
    sample_rate_hz: float
    reason: str


# ---------------------------------------------------------------------------
# 1. Physical Optics, CBDR Paradox & Co-Located Baseline Parallax
# ---------------------------------------------------------------------------


def compute_cbdr_edge_expansion_px_s(
    *,
    target_width_m: float,
    closing_speed_mps: float,
    range_m: float,
    ifov_rad: float = IMX636_DEFAULT_IFOV_RAD,
) -> CbdrExpansionResult:
    """Compute exact CBDR single-edge pixel expansion speed:

        v_edge = (W * v_close) / (2 * IFOV * R^2)   [px/s]

    For a Cessna 172 (W = 11.0 m) closing head-on at 250 kt (128.611 m/s) at R = 3,000 m
    on a 12 mm IMX636 (IFOV = 0.405 mrad), v_edge = 0.1941 px/s (5.15 s/px), proving
    pure EVS blindness to long-range CBDR intruders.
    """
    if (
        not all(math.isfinite(x) for x in (target_width_m, closing_speed_mps, range_m, ifov_rad))
        or target_width_m <= 0.0
        or closing_speed_mps < 0.0
        or range_m <= 0.0
        or ifov_rad <= 0.0
    ):
        raise ValueError("target_width_m, range_m, ifov_rad must be > 0 and closing_speed_mps >= 0")

    apparent_span_px = target_width_m / (ifov_rad * range_m)
    v_edge = (target_width_m * closing_speed_mps) / (2.0 * ifov_rad * range_m * range_m)
    seconds_per_px = math.inf if v_edge <= 0.0 else (1.0 / v_edge)
    ttc_s = math.inf if closing_speed_mps <= 0.0 else (range_m / closing_speed_mps)
    return CbdrExpansionResult(
        edge_expansion_px_s=v_edge,
        seconds_per_px=seconds_per_px,
        apparent_span_px=apparent_span_px,
        ttc_s=ttc_s,
        blind_gap_10hz_m=closing_speed_mps * 0.100,
        blind_gap_1khz_m=closing_speed_mps * 0.001,
    )


def compute_colocated_parallax_px(
    *,
    baseline_m: float,
    range_m: float,
    focal_length_m: float = IMX636_DEFAULT_FOCAL_M,
    pixel_pitch_m: float = IMX636_PIXEL_PITCH_M,
) -> float:
    """Compute rigid co-located EO+EVS stereo parallax in pixels:

        delta_px = (B * f) / (p * R)

    With 35–50 mm rigid baselines on 12 mm IMX636, parallax is < 0.5 px beyond 250 m
    and < 0.25 px beyond 500 m, enabling pure rotational homography H_inf with 0% light loss.
    """
    if (
        not all(math.isfinite(x) for x in (baseline_m, range_m, focal_length_m, pixel_pitch_m))
        or baseline_m < 0.0
        or range_m <= 0.0
        or focal_length_m <= 0.0
        or pixel_pitch_m <= 0.0
    ):
        raise ValueError("invalid optical parameters for parallax calculation")
    return (baseline_m * focal_length_m) / (pixel_pitch_m * range_m)


def select_asymmetric_sensor_anchor(
    *,
    range_m: float,
    closing_speed_mps: float,
    bearing_rate_deg_s: float,
    target_width_m: float = 11.0,
    baseline_m: float = 0.045,
    focal_length_m: float = IMX636_DEFAULT_FOCAL_M,
    pixel_pitch_m: float = IMX636_PIXEL_PITCH_M,
    terminal_range_m: float = 1000.0,
    crossing_bearing_rate_deg_s: float = 0.5,
    evs_min_edge_rate_px_s: float = 2.0,
) -> AsymmetricAnchorDecision:
    """Route intruders between long-range Global-Shutter EO Anchor and 1 kHz EVS Fast-Track."""
    if not math.isfinite(bearing_rate_deg_s):
        raise ValueError("bearing_rate_deg_s must be finite")
    ifov_rad = pixel_pitch_m / focal_length_m
    cbdr = compute_cbdr_edge_expansion_px_s(
        target_width_m=target_width_m,
        closing_speed_mps=closing_speed_mps,
        range_m=range_m,
        ifov_rad=ifov_rad,
    )
    parallax = compute_colocated_parallax_px(
        baseline_m=baseline_m,
        range_m=range_m,
        focal_length_m=focal_length_m,
        pixel_pitch_m=pixel_pitch_m,
    )
    h_inf_ok = parallax < 0.5

    if range_m < terminal_range_m:
        return AsymmetricAnchorDecision(
            mode=SensorAnchorMode.EVS_FAST_TRACK,
            cbdr=cbdr,
            parallax_px=parallax,
            pure_rotational_homography_valid=h_inf_ok,
            reason=f"terminal range {range_m:.1f} m < {terminal_range_m:.1f} m -> 1 kHz EVS",
        )
    if abs(bearing_rate_deg_s) > crossing_bearing_rate_deg_s:
        return AsymmetricAnchorDecision(
            mode=SensorAnchorMode.EVS_FAST_TRACK,
            cbdr=cbdr,
            parallax_px=parallax,
            pure_rotational_homography_valid=h_inf_ok,
            reason=(
                f"crossing/maneuvering |bearing_rate|={abs(bearing_rate_deg_s):.2f} deg/s > "
                f"{crossing_bearing_rate_deg_s:.2f} deg/s -> 1 kHz EVS"
            ),
        )
    if cbdr.edge_expansion_px_s >= evs_min_edge_rate_px_s:
        return AsymmetricAnchorDecision(
            mode=SensorAnchorMode.EVS_FAST_TRACK,
            cbdr=cbdr,
            parallax_px=parallax,
            pure_rotational_homography_valid=h_inf_ok,
            reason=(
                f"rapid edge expansion {cbdr.edge_expansion_px_s:.2f} px/s >= "
                f"{evs_min_edge_rate_px_s:.2f} px/s -> 1 kHz EVS"
            ),
        )
    return AsymmetricAnchorDecision(
        mode=SensorAnchorMode.EO_ANCHOR,
        cbdr=cbdr,
        parallax_px=parallax,
        pure_rotational_homography_valid=h_inf_ok,
        reason=(
            f"long-range CBDR (R={range_m:.0f} m, v_edge={cbdr.edge_expansion_px_s:.3f} px/s, "
            f"{cbdr.seconds_per_px:.2f} s/px) -> Global-Shutter EO Anchor"
        ),
    )


# ---------------------------------------------------------------------------
# 2. IMX636 Hardware ESP Guard (EVT 3.0 Bus Ceilings, ERC & Narrow-Notch AFK)
# ---------------------------------------------------------------------------


class Imx636EspController:
    """Hardware Event Signal Processing (ESP) guard for Sony IMX636 civil DAA."""

    def __init__(
        self,
        interface: Imx636Interface = Imx636Interface.TRITON2_2_5GIGE,
        max_notch_width_hz: float = DEFAULT_MAX_AFK_NOTCH_WIDTH_HZ,
        max_total_notch_span_hz: float = 50.0,
    ) -> None:
        if interface not in _EVT3_BUS_CEILING_EPS:
            raise ValueError(f"unsupported IMX636 interface: {interface!r}")
        if max_notch_width_hz <= 0.0 or max_total_notch_span_hz <= 0.0:
            raise ValueError("notch width ceilings must be positive")
        self._interface = interface
        self._bus_ceiling_eps = _EVT3_BUS_CEILING_EPS[interface]
        self._max_notch_width_hz = max_notch_width_hz
        self._max_total_notch_span_hz = max_total_notch_span_hz

    @property
    def interface(self) -> Imx636Interface:
        return self._interface

    @property
    def bus_ceiling_eps(self) -> int:
        return self._bus_ceiling_eps

    @staticmethod
    def compute_erc_cd_event_count(target_rate_eps: int) -> int:
        """Exact IMX636 ERC register mapping: cd_event_count = target_rate_eps // 5000."""
        if not isinstance(target_rate_eps, int) or isinstance(target_rate_eps, bool) or target_rate_eps <= 0:
            raise ValueError(f"target_rate_eps must be a positive int, got {target_rate_eps!r}")
        return target_rate_eps // ERC_DIVISOR

    def validate_afk_notches(self, notches: Sequence[AfkNotchBand]) -> tuple[bool, str]:
        """Enforce DR-10 Selective Narrow-Notch AFK rule (strictly rejects broadband 50–500 Hz)."""
        if not notches:
            return True, "AFK disabled (full 66.7–450 Hz propeller BPF preserved)"

        total_overlap_bpf_hz = 0.0
        for notch in notches:
            if notch.low_hz < IMX636_AFK_HW_MIN_HZ or notch.high_hz > IMX636_AFK_HW_MAX_HZ:
                return (
                    False,
                    f"AFK notch [{notch.low_hz}, {notch.high_hz}] Hz outside IMX636 HW "
                    f"[{IMX636_AFK_HW_MIN_HZ}, {IMX636_AFK_HW_MAX_HZ}] Hz",
                )
            if notch.width_hz > self._max_notch_width_hz:
                return (
                    False,
                    f"REJECTED broadband AFK [{notch.low_hz:.1f}..{notch.high_hz:.1f} Hz] "
                    f"(width {notch.width_hz:.1f} Hz > {self._max_notch_width_hz:.1f} Hz limit): "
                    f"erases 66.7–450 Hz civil propeller BPF signatures",
                )
            ov_lo = max(notch.low_hz, CIVIL_ROTOR_BPF_MIN_HZ)
            ov_hi = min(notch.high_hz, CIVIL_ROTOR_BPF_MAX_HZ)
            if ov_hi > ov_lo:
                total_overlap_bpf_hz += ov_hi - ov_lo

        if total_overlap_bpf_hz > self._max_total_notch_span_hz:
            return (
                False,
                f"REJECTED cumulative AFK overlap {total_overlap_bpf_hz:.1f} Hz > "
                f"{self._max_total_notch_span_hz:.1f} Hz inside 66.7–450 Hz rotor BPF band",
            )
        return True, f"accepted {len(notches)} selective narrow-notch AFK band(s)"

    def configure(
        self,
        *,
        target_rate_eps: int,
        afk_notches: Sequence[AfkNotchBand] = (),
        strict: bool = True,
    ) -> EspConfigResult:
        """Validate EVT 3.0 bus ceiling, compute ERC register, and enforce selective AFK."""
        if not isinstance(target_rate_eps, int) or isinstance(target_rate_eps, bool) or target_rate_eps <= 0:
            raise ValueError(f"target_rate_eps must be a positive int, got {target_rate_eps!r}")

        if target_rate_eps > self._bus_ceiling_eps:
            msg = (
                f"target_rate_eps={target_rate_eps} exceeds {self._interface.value} "
                f"EVT 3.0 bus ceiling {self._bus_ceiling_eps} eps"
            )
            if strict:
                raise ValueError(msg)
            clamped = self._bus_ceiling_eps
            return EspConfigResult(
                interface=self._interface,
                bus_ceiling_eps=self._bus_ceiling_eps,
                target_rate_eps=clamped,
                erc_cd_event_count=self.compute_erc_cd_event_count(clamped),
                afk_enabled=False,
                afk_notches=(),
                accepted=False,
                reason=msg,
            )

        afk_ok, afk_reason = self.validate_afk_notches(afk_notches)
        if not afk_ok:
            if strict:
                raise ValueError(afk_reason)
            return EspConfigResult(
                interface=self._interface,
                bus_ceiling_eps=self._bus_ceiling_eps,
                target_rate_eps=target_rate_eps,
                erc_cd_event_count=self.compute_erc_cd_event_count(target_rate_eps),
                afk_enabled=False,
                afk_notches=(),
                accepted=False,
                reason=afk_reason,
            )

        return EspConfigResult(
            interface=self._interface,
            bus_ceiling_eps=self._bus_ceiling_eps,
            target_rate_eps=target_rate_eps,
            erc_cd_event_count=self.compute_erc_cd_event_count(target_rate_eps),
            afk_enabled=len(afk_notches) > 0,
            afk_notches=tuple(afk_notches),
            accepted=True,
            reason=afk_reason,
        )


# ---------------------------------------------------------------------------
# 3. FRIES Propeller-Harmonic Spectral Discriminator (66.7–450 Hz Civil BPF)
# ---------------------------------------------------------------------------


def estimate_rotor_bpf_hz(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    bpf_min_hz: float = CIVIL_ROTOR_BPF_MIN_HZ,
    bpf_max_hz: float = CIVIL_ROTOR_BPF_MAX_HZ,
    excluded_notches: Sequence[AfkNotchBand] = (),
) -> tuple[float, float, float, bool]:
    """Estimate dominant rotor BPF (Hz), peak SNR (dB), in-band energy ratio, and 2x harmonic flag.

    Uses Hann-windowed zero-padded `np.fft.rfft` with parabolic bin interpolation on log-power
    to achieve well under 1% frequency accuracy (< 3% DR-10 budget) at >= 1,000 Hz sampling.
    """
    if not math.isfinite(sample_rate_hz) or sample_rate_hz < 1000.0:
        raise ValueError(
            f"FRIES requires sample_rate_hz >= 1000.0 Hz (Nyquist >= 500 Hz), got {sample_rate_hz!r}"
        )
    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size < 32:
        raise ValueError(f"signal too short for spectral analysis (len={arr.size} < 32)")
    if not np.all(np.isfinite(arr)):
        return 0.0, 0.0, 0.0, False

    centered = arr - float(np.mean(arr))
    ac_energy = float(np.sum(centered * centered))
    if ac_energy <= 1e-12:
        return 0.0, 0.0, 0.0, False

    n = centered.size
    window = np.hanning(n)
    # Zero-pad to at least 4096 bins for sub-0.25 Hz frequency grid precision
    n_fft = max(4096, 1 << (n - 1).bit_length())
    spec = np.fft.rfft(centered * window, n=n_fft)
    power = spec.real * spec.real + spec.imag * spec.imag
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

    nyquist = 0.5 * sample_rate_hz
    eff_max_hz = min(bpf_max_hz, nyquist - 1.0)
    band_mask = (freqs >= bpf_min_hz) & (freqs <= eff_max_hz)
    for notch in excluded_notches:
        band_mask &= ~((freqs >= notch.low_hz) & (freqs <= notch.high_hz))

    if not np.any(band_mask):
        return 0.0, 0.0, 0.0, False

    ac_mask = freqs >= 1.0
    total_ac_power = float(np.sum(power[ac_mask]))
    in_band_power = float(np.sum(power[band_mask]))
    in_band_energy_ratio = in_band_power / max(total_ac_power, 1e-18)

    band_indices = np.nonzero(band_mask)[0]
    band_powers = power[band_indices]
    local_argmax = int(np.argmax(band_powers))
    peak_idx = int(band_indices[local_argmax])
    peak_power = float(power[peak_idx])

    # Parabolic peak refinement on log-power for sub-bin accuracy
    df = sample_rate_hz / float(n_fft)
    if 0 < peak_idx < (power.size - 1):
        p_l = math.log(max(float(power[peak_idx - 1]), 1e-30))
        p_c = math.log(max(peak_power, 1e-30))
        p_r = math.log(max(float(power[peak_idx + 1]), 1e-30))
        denom = p_l - 2.0 * p_c + p_r
        delta = 0.5 * (p_l - p_r) / denom if abs(denom) > 1e-12 else 0.0
        delta = min(max(delta, -0.5), 0.5)
    else:
        delta = 0.0
    est_freq_hz = (peak_idx + delta) * df

    # Robust median noise floor inside the rotor BPF band (excluding +/- 5 Hz main lobe)
    lobe_excl = np.abs(freqs[band_indices] - est_freq_hz) > 5.0
    noise_bins = band_powers[lobe_excl] if np.any(lobe_excl) else band_powers
    noise_floor = max(float(np.median(noise_bins)), 1e-18)
    peak_snr_db = 10.0 * math.log10(max(peak_power / noise_floor, 1e-12))

    # Check whether 2nd harmonic (2 * est_freq_hz) is also prominent above noise floor
    harmonic_2x_confirmed = False
    f2 = 2.0 * est_freq_hz
    if f2 <= (nyquist - 5.0):
        h2_mask = np.abs(freqs - f2) <= max(3.0, 0.03 * f2)
        if np.any(h2_mask):
            h2_power = float(np.max(power[h2_mask]))
            h2_snr_db = 10.0 * math.log10(max(h2_power / noise_floor, 1e-12))
            harmonic_2x_confirmed = h2_snr_db >= 10.0

    return est_freq_hz, peak_snr_db, in_band_energy_ratio, harmonic_2x_confirmed


class FriesRotorDiscriminator:
    """DR-10 FRIES propeller-harmonic discriminator separating aircraft/UAS from biological/cloud clutter."""

    def __init__(
        self,
        *,
        min_peak_snr_db: float = 15.0,
        min_in_band_energy_ratio: float = 0.25,
        bpf_min_hz: float = CIVIL_ROTOR_BPF_MIN_HZ,
        bpf_max_hz: float = CIVIL_ROTOR_BPF_MAX_HZ,
        excluded_notches: Sequence[AfkNotchBand] = (),
    ) -> None:
        self._min_peak_snr_db = min_peak_snr_db
        self._min_in_band_energy_ratio = min_in_band_energy_ratio
        self._bpf_min_hz = bpf_min_hz
        self._bpf_max_hz = bpf_max_hz
        self._excluded_notches = tuple(excluded_notches)

    def classify(
        self,
        roi_event_series: np.ndarray,
        *,
        sample_rate_hz: float = 1000.0,
    ) -> FriesSpectrumResult:
        """Classify an ROI event-count / polarity time-series at >= 1,000 Hz."""
        est_hz, snr_db, band_ratio, h2_ok = estimate_rotor_bpf_hz(
            roi_event_series,
            sample_rate_hz=sample_rate_hz,
            bpf_min_hz=self._bpf_min_hz,
            bpf_max_hz=self._bpf_max_hz,
            excluded_notches=self._excluded_notches,
        )
        if (
            est_hz >= self._bpf_min_hz
            and est_hz <= self._bpf_max_hz
            and snr_db >= self._min_peak_snr_db
            and band_ratio >= self._min_in_band_energy_ratio
        ):
            return FriesSpectrumResult(
                classification=RotorClass.ROTORCRAFT_OR_PROP_AIRCRAFT,
                estimated_bpf_hz=est_hz,
                peak_snr_db=snr_db,
                in_band_energy_ratio=band_ratio,
                harmonic_2x_confirmed=h2_ok,
                sample_rate_hz=sample_rate_hz,
                reason=(
                    f"rotor BPF lock {est_hz:.1f} Hz "
                    f"(SNR={snr_db:.1f} dB, in_band={band_ratio:.2f}, 2x={h2_ok})"
                ),
            )
        return FriesSpectrumResult(
            classification=RotorClass.NON_ROTATING_CLUTTER,
            estimated_bpf_hz=est_hz,
            peak_snr_db=snr_db,
            in_band_energy_ratio=band_ratio,
            harmonic_2x_confirmed=h2_ok,
            sample_rate_hz=sample_rate_hz,
            reason=(
                f"clutter rejected (peak={est_hz:.1f} Hz, "
                f"SNR={snr_db:.1f} dB < {self._min_peak_snr_db:.1f} or "
                f"in_band={band_ratio:.2f} < {self._min_in_band_energy_ratio:.2f})"
            ),
        )
