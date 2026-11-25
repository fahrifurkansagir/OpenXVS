# Copyright 2026 OpenXVS Contributors
# SPDX-License-Identifier: Apache-2.0
"""XVS-Fleet (`DR-17`): Adaptive MeshDAA RF Governor, ASTM F3548/F3269 Anti-Oscillation Interlock, and 1-to-Many EASA HCD Triage Queue."""

from __future__ import annotations

from xvs.v2.xvs_fleet.governor import (
    MAX_ACTIVE_FOCUS_CARDS,
    MAX_FLEET_PER_OPERATOR,
    MAX_MEC_ZENOH_PEERS_5KM,
    AlertSeverity,
    FleetDatalinkPlan,
    FleetPolicyError,
    FocusCard,
    HcdAlert,
    HcdTriageQueue,
    HcdTriageSnapshot,
    InterlockState,
    InterlockStepResult,
    StrategicTacticalInterlock,
    compute_fleet_datalink_plan,
    compute_tactical_evasion_buffer_m,
    parse_operator_fleet_ratio,
)

__all__ = [
    "MAX_ACTIVE_FOCUS_CARDS",
    "MAX_FLEET_PER_OPERATOR",
    "MAX_MEC_ZENOH_PEERS_5KM",
    "AlertSeverity",
    "FleetDatalinkPlan",
    "FleetPolicyError",
    "FocusCard",
    "HcdAlert",
    "HcdTriageQueue",
    "HcdTriageSnapshot",
    "InterlockState",
    "InterlockStepResult",
    "StrategicTacticalInterlock",
    "compute_fleet_datalink_plan",
    "compute_tactical_evasion_buffer_m",
    "parse_operator_fleet_ratio",
]
