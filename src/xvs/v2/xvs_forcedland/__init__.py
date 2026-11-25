# Copyright 2026 OpenXVS Contributors
# SPDX-License-Identifier: Apache-2.0
"""XVS-ForcedLand (`DR-18`): Deterministic Glide, 3DEP FLZ, 6-Stage VLM Gate, and Passive LEO-Doppler SoOP Guard."""

from __future__ import annotations

from xvs.v2.xvs_forcedland.planner import (
    G_STANDARD_MPS2,
    MAX_NOMINAL_BANK_DEG,
    POLAR_PRESETS,
    RHO_SL_KG_M3,
    AircraftGlideState,
    BankedTurnSchedule,
    BestGlideParams,
    FixedWingPolar,
    FlzCandidate,
    FlzEvaluationResult,
    ForcedLandDecision,
    ForcedLandPolicyError,
    ForcedLandVlmGate,
    LeoOmmNisResult,
    MacCreadySolution,
    VlmRankMsg,
    abl_wind_profile,
    banked_turn_schedule,
    best_glide_params,
    evaluate_flz_candidates,
    maccready_speed_to_fly,
    validate_leo_omm_and_nis,
)

__all__ = [
    "G_STANDARD_MPS2",
    "MAX_NOMINAL_BANK_DEG",
    "POLAR_PRESETS",
    "RHO_SL_KG_M3",
    "AircraftGlideState",
    "BankedTurnSchedule",
    "BestGlideParams",
    "FixedWingPolar",
    "FlzCandidate",
    "FlzEvaluationResult",
    "ForcedLandDecision",
    "ForcedLandPolicyError",
    "ForcedLandVlmGate",
    "LeoOmmNisResult",
    "MacCreadySolution",
    "VlmRankMsg",
    "abl_wind_profile",
    "banked_turn_schedule",
    "best_glide_params",
    "evaluate_flz_candidates",
    "maccready_speed_to_fly",
    "validate_leo_omm_and_nis",
]
