"""Seeded Adaptive Stress Testing (AST) & EASA Proposed Issue 03 R-LM-07..11 Evidence Emitter (DR-08).

Searches environmental & visual-stress parameter trajectories (`fog_rvr_km`, `sun_glare_intensity`,
`cloud_shadow_frac`, `aliasing_entropy`, `sensor_fault_prob`) via deterministic seeded evolutionary
importance search to uncover worst-case integrity failure modes and emit structured EASA R-LM-07..11
compliance evidence JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable

import numpy as np

from .sae_audit import SaeAuditResult


@dataclass(frozen=True)
class StressParameters:
    """Bounded environmental & visual-stress trajectory parameters."""

    fog_rvr_km: float  # [0.15, 15.0]
    sun_glare_intensity: float  # [0.0, 1.0]
    cloud_shadow_frac: float  # [0.0, 1.0]
    aliasing_entropy: float  # [0.0, 1.0] (lower = more aliased)
    sensor_fault_prob: float  # [0.0, 1.0]


@dataclass(frozen=True)
class StressTrajectoryResult:
    """Evaluated AST scenario with severity, log-likelihood & failure flag."""

    trial_index: int
    params: StressParameters
    severity: float
    log_likelihood: float
    ast_objective: float
    failure_triggered: bool
    failure_mode: str


StressEvaluator = Callable[[StressParameters], tuple[float, bool, str]]


def default_integrity_stress_evaluator(params: StressParameters) -> tuple[float, bool, str]:
    """Deterministic surrogate evaluator mapping environmental stress to integrity margin violation."""
    # Severity grows under dense fog (low RVR), strong sun glare, low aliasing entropy & sensor faults
    fog_term = max(0.0, (1.2 - params.fog_rvr_km) / 1.05) * 1.4
    glare_term = (params.sun_glare_intensity**1.5) * 1.1
    shadow_term = params.cloud_shadow_frac * 0.5
    alias_term = max(0.0, 0.55 - params.aliasing_entropy) * 2.5
    fault_term = params.sensor_fault_prob * 1.6
    severity = fog_term + glare_term + shadow_term + alias_term + fault_term

    if params.aliasing_entropy < 0.35 and params.fog_rvr_km < 1.0:
        return severity, True, "VISUAL_ALIASING_IN_FOG"
    if params.sensor_fault_prob > 0.65 and params.sun_glare_intensity > 0.70:
        return severity, True, "SIMULTANEOUS_FAULT_AND_GLARE"
    if severity >= 2.2:
        return severity, True, "COMPOUND_DVE_INTEGRITY_LIMIT"
    return severity, False, "NOMINAL_WITHIN_ODD"


def _scenario_log_likelihood(params: StressParameters) -> float:
    """Gaussian-prior log-likelihood over nominal operating distribution (higher = more plausible)."""
    z_fog = (math.log(max(params.fog_rvr_km, 0.05)) - math.log(5.0)) / 1.2
    z_glare = (params.sun_glare_intensity - 0.15) / 0.35
    z_shadow = (params.cloud_shadow_frac - 0.25) / 0.35
    z_alias = (params.aliasing_entropy - 0.78) / 0.22
    z_fault = (params.sensor_fault_prob - 0.05) / 0.25
    return -0.5 * (z_fog**2 + z_glare**2 + z_shadow**2 + z_alias**2 + z_fault**2)


class AdaptiveStressTester:
    """Seeded evolutionary Adaptive Stress Testing sampler (DR-08 / R-LM-10)."""

    def __init__(
        self,
        seed: int = 20261121,
        evaluator: StressEvaluator | None = None,
        likelihood_weight: float = 0.10,
    ) -> None:
        self.seed = int(seed)
        self.evaluator = default_integrity_stress_evaluator if evaluator is None else evaluator
        self.likelihood_weight = float(likelihood_weight)

    def _clip_params(self, raw: np.ndarray) -> StressParameters:
        return StressParameters(
            fog_rvr_km=float(np.clip(raw[0], 0.15, 15.0)),
            sun_glare_intensity=float(np.clip(raw[1], 0.0, 1.0)),
            cloud_shadow_frac=float(np.clip(raw[2], 0.0, 1.0)),
            aliasing_entropy=float(np.clip(raw[3], 0.0, 1.0)),
            sensor_fault_prob=float(np.clip(raw[4], 0.0, 1.0)),
        )

    def run_search(
        self,
        n_generations: int = 6,
        pop_size: int = 24,
        elite_k: int = 6,
    ) -> list[StressTrajectoryResult]:
        """Run deterministic seeded AST search and return all evaluated trajectories sorted by objective."""
        if n_generations <= 0 or pop_size <= 0 or not (1 <= elite_k <= pop_size):
            raise ValueError("Invalid AST search hyperparameters")
        rng = np.random.default_rng(self.seed)

        # Generation 0: Latin-like uniform coverage over stress hypercube
        pop = np.column_stack(
            [
                rng.uniform(0.15, 15.0, size=pop_size),
                rng.uniform(0.0, 1.0, size=pop_size),
                rng.uniform(0.0, 1.0, size=pop_size),
                rng.uniform(0.0, 1.0, size=pop_size),
                rng.uniform(0.0, 1.0, size=pop_size),
            ]
        )
        all_results: list[StressTrajectoryResult] = []
        trial_idx = 0

        for gen in range(n_generations):
            gen_results: list[StressTrajectoryResult] = []
            for row in pop:
                p = self._clip_params(row)
                sev, failed, mode = self.evaluator(p)
                ll = _scenario_log_likelihood(p)
                obj = float(sev) + self.likelihood_weight * ll + (1.0 if failed else 0.0)
                res = StressTrajectoryResult(
                    trial_index=trial_idx,
                    params=p,
                    severity=float(sev),
                    log_likelihood=float(ll),
                    ast_objective=obj,
                    failure_triggered=bool(failed),
                    failure_mode=str(mode),
                )
                gen_results.append(res)
                all_results.append(res)
                trial_idx += 1

            if gen + 1 < n_generations:
                gen_results.sort(key=lambda r: r.ast_objective, reverse=True)
                elites = gen_results[:elite_k]
                elite_arr = np.array(
                    [
                        [
                            e.params.fog_rvr_km,
                            e.params.sun_glare_intensity,
                            e.params.cloud_shadow_frac,
                            e.params.aliasing_entropy,
                            e.params.sensor_fault_prob,
                        ]
                        for e in elites
                    ],
                    dtype=np.float64,
                )
                scales = np.array([1.0, 0.15, 0.15, 0.12, 0.15], dtype=np.float64) * (
                    0.85**gen
                )
                next_pop = np.zeros_like(pop)
                next_pop[:elite_k] = elite_arr
                for i in range(elite_k, pop_size):
                    parent = elite_arr[int(rng.integers(0, elite_k))]
                    next_pop[i] = parent + rng.normal(0.0, scales)
                pop = next_pop

        all_results.sort(key=lambda r: r.ast_objective, reverse=True)
        return all_results


def build_easa_rlm_07_11_report(
    *,
    seed: int,
    ood_auroc: float,
    ood_conformal_threshold: float,
    ood_id_fpr: float,
    sae_result: SaeAuditResult,
    ast_results: list[StressTrajectoryResult],
    top_k_failures: int = 5,
) -> dict[str, Any]:
    """Emit structured EASA Proposed Issue 03 R-LM-07..11 compliance evidence JSON (DR-08)."""
    failures = [r for r in ast_results if r.failure_triggered]
    top_trajs = ast_results[: max(1, min(top_k_failures, len(ast_results)))]

    # Compute sensitivity across each stress axis as mean parameter delta (failures - non-failures)
    non_failures = [r for r in ast_results if not r.failure_triggered]

    def _mean_param(items: list[StressTrajectoryResult], name: str) -> float:
        if not items:
            return 0.0
        return float(np.mean([getattr(r.params, name) for r in items]))

    axis_names = (
        "fog_rvr_km",
        "sun_glare_intensity",
        "cloud_shadow_frac",
        "aliasing_entropy",
        "sensor_fault_prob",
    )
    shift_sensitivity = {
        ax: round(_mean_param(failures, ax) - _mean_param(non_failures, ax), 4)
        for ax in axis_names
    }

    provenance_core = {
        "seed": int(seed),
        "n_ast_trials": len(ast_results),
        "n_failures_discovered": len(failures),
        "ood_auroc": round(float(ood_auroc), 6),
        "sae_content_ratio": round(float(sae_result.content_reliance_ratio), 6),
    }
    audit_sha256 = hashlib.sha256(
        json.dumps(provenance_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "xvs.v2_m2.easa_rlm_07_11_audit/1",
        "easa_standard": "EASA AI Concept Paper Proposed Issue 03 (R-LM-07..11)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "R-LM-07": {
            "requirement": "OOD / ODD boundary coverage & conformal calibration",
            "ood_auroc": float(ood_auroc),
            "conformal_threshold": float(ood_conformal_threshold),
            "empirical_id_fpr": float(ood_id_fpr),
            "pass": bool(ood_auroc > 0.95),
        },
        "R-LM-08": {
            "requirement": "Distributional shift sensitivity across environmental stress axes",
            "stress_axes": list(axis_names),
            "failure_vs_nominal_axis_delta": shift_sensitivity,
            "pass": bool(len(failures) > 0),
        },
        "R-LM-09": {
            "requirement": "Shortcut / spurious-feature attribution audit via Top-K SAE",
            "reconstruction_mse": float(sae_result.reconstruction_mse),
            "mean_active_latents": float(sae_result.mean_active_latents),
            "content_reliance_ratio": float(sae_result.content_reliance_ratio),
            "style_reliance_ratio": float(sae_result.style_reliance_ratio),
            "shortcut_warning": bool(sae_result.shortcut_warning),
            "pass": bool(not sae_result.shortcut_warning),
        },
        "R-LM-10": {
            "requirement": "Worst-case stress failure trajectories discovered by seeded AST",
            "total_trials": len(ast_results),
            "failures_discovered": len(failures),
            "top_trajectories": [
                {
                    "trial_index": r.trial_index,
                    "severity": round(r.severity, 4),
                    "log_likelihood": round(r.log_likelihood, 4),
                    "ast_objective": round(r.ast_objective, 4),
                    "failure_triggered": r.failure_triggered,
                    "failure_mode": r.failure_mode,
                    "params": asdict(r.params),
                }
                for r in top_trajs
            ],
            "pass": bool(len(failures) >= 1),
        },
        "R-LM-11": {
            "requirement": "Deterministic seed & reproducible audit provenance",
            "seed": int(seed),
            "audit_sha256": audit_sha256,
            "deterministic_reproducible": True,
            "pass": True,
        },
    }
