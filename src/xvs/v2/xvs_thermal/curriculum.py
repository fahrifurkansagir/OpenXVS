"""Seeded DR-04:205 40/20/20/10/10 EO+LWIR physical fault curriculum sampler (SP2 §3.3 / BC-M2-m1).

Exact bucket weights aligned with DR-04 §4.2 item 5:
  - 40%: `nominal`           (paired healthy EO + LWIR)
  - 20%: `eo_degraded`       (fog / glare / night -> q_eo = 0.0, q_lwir = 1.0)
  - 20%: `lwir_degraded`     (thermal_crossover / cloud_washout / ffc_freeze -> q_eo = 1.0, q_lwir = 0.0)
  - 10%: `partial_local`     (sun_in_fov_bloom / local_cloud -> partial localized degradation)
  - 10%: `simultaneous_dve`  (dense liquid cloud / coastal fog + night -> q_eo = 0.0, q_lwir = 0.0, vacuity -> 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from xvs.sensor.lwir import AtmosphericRegime, ThermalFaultConfig

CurriculumBucket = Literal[
    "nominal",
    "eo_degraded",
    "lwir_degraded",
    "partial_local",
    "simultaneous_dve",
]

CURRICULUM_BUCKETS: tuple[CurriculumBucket, ...] = (
    "nominal",
    "eo_degraded",
    "lwir_degraded",
    "partial_local",
    "simultaneous_dve",
)

CURRICULUM_PROBS: dict[CurriculumBucket, float] = {
    "nominal": 0.40,
    "eo_degraded": 0.20,
    "lwir_degraded": 0.20,
    "partial_local": 0.10,
    "simultaneous_dve": 0.10,
}

_EO_FAULT_SUBMODES: tuple[str, ...] = ("radiation_fog", "solar_glare", "pitch_black_night")
_LWIR_FAULT_SUBMODES: tuple[str, ...] = ("thermal_crossover", "cloud_washout", "ffc_freeze")
_PARTIAL_SUBMODES: tuple[str, ...] = ("sun_in_fov_bloom", "local_cloud_clutter")
_DUAL_DVE_SUBMODES: tuple[str, ...] = ("in_stratus_cloud", "coastal_advection_fog_night")


@dataclass(frozen=True)
class CurriculumSample:
    """Single sampled physical DVE fault scenario for XVS-ThermalNet training."""

    bucket: CurriculumBucket
    submode: str
    regime: AtmosphericRegime
    rvr_km: float
    q_eo: float
    q_lwir: float
    target_vacuity: float
    lwir_fault: ThermalFaultConfig


class FaultCurriculumSampler:
    """Deterministic seeded sampler for the DR-04:205 40/20/20/10/10 fault curriculum."""

    def __init__(self, seed: int = 20261121) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._probs = np.array([CURRICULUM_PROBS[b] for b in CURRICULUM_BUCKETS], dtype=np.float64)
        if abs(float(self._probs.sum()) - 1.0) > 1e-12:
            raise ValueError(f"Curriculum probabilities must sum to 1.0, got {self._probs.sum()}")

    def sample_one(self) -> CurriculumSample:
        idx = int(self._rng.choice(len(CURRICULUM_BUCKETS), p=self._probs))
        bucket = CURRICULUM_BUCKETS[idx]

        if bucket == "nominal":
            rvr_km = float(self._rng.uniform(5.0, 20.0))
            return CurriculumSample(
                bucket="nominal",
                submode="clear_nominal",
                regime="clear",
                rvr_km=rvr_km,
                q_eo=1.0,
                q_lwir=1.0,
                target_vacuity=0.0,
                lwir_fault=ThermalFaultConfig(),
            )

        if bucket == "eo_degraded":
            sub = _EO_FAULT_SUBMODES[int(self._rng.integers(0, len(_EO_FAULT_SUBMODES)))]
            regime: AtmosphericRegime = "radiation_fog" if sub == "radiation_fog" else "clear"
            rvr_km = float(self._rng.uniform(0.35, 0.85)) if sub == "radiation_fog" else 10.0
            return CurriculumSample(
                bucket="eo_degraded",
                submode=sub,
                regime=regime,
                rvr_km=rvr_km,
                q_eo=0.0,
                q_lwir=1.0,
                target_vacuity=0.0,
                lwir_fault=ThermalFaultConfig(),
            )

        if bucket == "lwir_degraded":
            sub = _LWIR_FAULT_SUBMODES[int(self._rng.integers(0, len(_LWIR_FAULT_SUBMODES)))]
            if sub == "thermal_crossover":
                flt = ThermalFaultConfig(thermal_crossover=True, crossover_residual_k=0.02)
                regime = "clear"
                rvr_km = 10.0
            elif sub == "cloud_washout":
                flt = ThermalFaultConfig(cloud_washout=True)
                regime = "stratus_cloud"
                rvr_km = 0.25
            else:
                flt = ThermalFaultConfig(ffc_freeze=True)
                regime = "clear"
                rvr_km = 10.0
            return CurriculumSample(
                bucket="lwir_degraded",
                submode=sub,
                regime=regime,
                rvr_km=rvr_km,
                q_eo=1.0,
                q_lwir=0.0,
                target_vacuity=0.0,
                lwir_fault=flt,
            )

        if bucket == "partial_local":
            sub = _PARTIAL_SUBMODES[int(self._rng.integers(0, len(_PARTIAL_SUBMODES)))]
            if sub == "sun_in_fov_bloom":
                u = float(self._rng.uniform(80.0, 560.0))
                v = float(self._rng.uniform(40.0, 220.0))
                flt = ThermalFaultConfig(sun_in_fov_bloom=True, sun_uv_px=(u, v))
                q_eo = float(self._rng.uniform(0.5, 0.9))
                q_lwir = float(self._rng.uniform(0.35, 0.75))
            else:
                flt = ThermalFaultConfig()
                q_eo = float(self._rng.uniform(0.55, 0.9))
                q_lwir = float(self._rng.uniform(0.55, 0.9))
            return CurriculumSample(
                bucket="partial_local",
                submode=sub,
                regime="clear",
                rvr_km=float(self._rng.uniform(2.0, 8.0)),
                q_eo=q_eo,
                q_lwir=q_lwir,
                target_vacuity=0.25,
                lwir_fault=flt,
            )

        # simultaneous_dve (10%): both EO and LWIR blind -> vacuity target 1.0
        sub = _DUAL_DVE_SUBMODES[int(self._rng.integers(0, len(_DUAL_DVE_SUBMODES)))]
        if sub == "in_stratus_cloud":
            flt = ThermalFaultConfig(cloud_washout=True)
            regime = "stratus_cloud"
            rvr_km = float(self._rng.uniform(0.08, 0.25))
        else:
            flt = ThermalFaultConfig(cloud_washout=True, cloud_washout_beta_km_inv=9.0)
            regime = "advection_fog"
            rvr_km = float(self._rng.uniform(0.15, 0.30))
        return CurriculumSample(
            bucket="simultaneous_dve",
            submode=sub,
            regime=regime,
            rvr_km=rvr_km,
            q_eo=0.0,
            q_lwir=0.0,
            target_vacuity=1.0,
            lwir_fault=flt,
        )

    def sample_batch(self, n: int) -> list[CurriculumSample]:
        if n <= 0:
            raise ValueError(f"Batch size n must be positive, got {n}")
        return [self.sample_one() for _ in range(n)]


def summarize_curriculum_distribution(samples: Sequence[CurriculumSample]) -> dict[str, float]:
    """Compute empirical fraction of each bucket across sampled scenarios."""
    total = len(samples)
    if total == 0:
        raise ValueError("Cannot summarize empty sample sequence")
    counts: dict[str, int] = {b: 0 for b in CURRICULUM_BUCKETS}
    for s in samples:
        counts[s.bucket] += 1
    return {b: counts[b] / float(total) for b in CURRICULUM_BUCKETS}
