# Copyright 2026 OpenXVS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DR-17 Civil Fleet MeshDAA RF Governor, ASTM F3548/F3269 Interlock, and EASA HCD Triage Queue.

Strictly civil-only, Apache-2.0 clean-room implementation. Zero GPL dependencies,
zero unannotated `.get()` calls, strictly NED/horizontal scalar distances.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

MAX_MEC_ZENOH_PEERS_5KM: int = 12
MAX_SECTOR_FLEET_SIZE: int = 50
MAX_EU_SX1262_CLUSTER_SIZE: int = 25
MAX_FLEET_PER_OPERATOR: int = 20
MAX_ACTIVE_FOCUS_CARDS: int = 2

_BANNED_RF_TOKENS: tuple[str, ...] = (
    "lora",
    "sf7",
    "sf8",
    "sf9",
    "sf10",
    "sf11",
    "sf12",
    "expresslrs",
    "elrs",
    "meshtastic",
    "starlink_roam",
    "starlink-roam",
    "starlink roam",
    "consumer_starlink",
    "direct_to_cell",
    "direct-to-cell",
    "direct to cell",
    "d2c",
)

_ALLOWED_TRANSPORTS: frozenset[str] = frozenset(
    {
        "dual_sim_5g_lte_mec_zenoh",
        "sx1262_gfsk_gps_tdma",
    }
)


class FleetPolicyError(PermissionError):
    """Fail-closed policy rejection for non-compliant RF links or unsafe HCD ratios."""


@dataclass(frozen=True)
class FleetDatalinkPlan:
    """Computed MAVLink `#52001` MeshDAA transport schedule and dynamic ECBF-QP barrier."""

    transport: str
    region: str
    fleet_size: int
    geo_culled_peers_5km: int
    nominal_rate_hz: float
    burst_rate_hz: float
    active_publish_rate_hz: float
    active_daa_burst: bool
    phy_bitrate_kbps: float
    duty_cycle_cap_pct: float
    dt_packet_s: float
    ecbf_base_r_min_m: float
    ecbf_barrier_expansion_m: float
    ecbf_effective_r_min_m: float


class InterlockState(str, Enum):
    """Unified ASTM F3548-21 <-> ASTM F3269-21 anti-oscillation interlock states."""

    NOMINAL_STRATEGIC = "NOMINAL_STRATEGIC"
    TACTICAL_OVERRIDE_FREEZE = "TACTICAL_OVERRIDE_FREEZE"
    HYSTERESIS_RECOVERY = "HYSTERESIS_RECOVERY"
    CONTINGENT = "CONTINGENT"


@dataclass(frozen=True)
class InterlockStepResult:
    """Single-tick deterministic output of `StrategicTacticalInterlock`."""

    state: InterlockState
    strategic_freeze: bool
    strategic_cmd_accepted: bool
    dss_operational_intent_state: str
    omit_key_ovn: bool
    replan_unlocked: bool
    clear_dwell_s: float
    nonconforming_elapsed_s: float


class AlertSeverity(IntEnum):
    """EASA HCD / NASA Ames 4-tier operator triage severity (lower integer = higher priority)."""

    P0_EMERGENCY = 0
    P1_HIGH = 1
    P2_ADVISORY = 2
    P3_NOMINAL = 3


@dataclass(frozen=True)
class HcdAlert:
    """Incoming fleet alert routed through the 1-to-Many GCS HCD triage queue."""

    alert_id: str
    aircraft_id: str
    severity: AlertSeverity
    timestamp_s: float
    summary: str
    region_cluster_id: str | None = None


@dataclass(frozen=True)
class FocusCard:
    """Operator-visible Focus Card strictly capped at `<= 2` active simultaneously."""

    card_id: str
    severity: AlertSeverity
    aircraft_ids: tuple[str, ...]
    timestamp_s: float
    summary: str
    coalesced: bool = False


@dataclass(frozen=True)
class HcdTriageSnapshot:
    """Deterministic HCD triage queue output enforcing `<= 2` Focus Cards & peer offload."""

    fleet_size: int
    max_fleet_per_operator: int
    active_focus_cards: tuple[FocusCard, ...]
    preempted_overflow_cards: tuple[FocusCard, ...]
    deferred_alerts: tuple[HcdAlert, ...]
    offload_required: bool
    offload_recommended_ac_ids: tuple[str, ...]
    p2_aural_chime_suppressed: bool


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compute_fleet_datalink_plan(
    *,
    fleet_size: int,
    region: str = "US",
    transport: str = "dual_sim_5g_lte_mec_zenoh",
    min_peer_distance_m: float = 5000.0,
    peers_within_5km: int | None = None,
    r_min_m: float = 15.0,
    v_max_mps: float = 50.0,
) -> FleetDatalinkPlan:
    """Compute adaptive `#52001` MeshDAA publish rate and dynamic ECBF-QP containment barrier.

    Enforces DR-17 hard locks:
      - Fail-closed rejection of LoRa CSS (`SF7..12`), ExpressLRS, Meshtastic,
        consumer Starlink Roam airborne hacks, and Direct-to-Cell (`D2C`).
      - Primary Dual-SIM 5G SA / 4G LTE + Regional MEC Zenoh (`k <= 12` peers within `5 km`,
        `10 / 5 / 5 Hz` nominal for `<= 10 / 25 / 50` AC, `10-20 Hz` burst `< 2 km`).
      - Fallback `SX1262` GFSK GPS-TDMA lifeboat (`US 200 kbps` or `EU g3 100 kbps <= 25 AC`).
      - Dynamic ECBF-QP barrier expansion `r_eff = r_min + v_max * dt_packet`.
    """
    transport_norm = transport.strip().lower()
    for banned in _BANNED_RF_TOKENS:
        if banned in transport_norm:
            raise FleetPolicyError(
                f"Transport '{transport}' matches DR-17 banned RF technology '{banned}'"
            )
    if transport_norm not in _ALLOWED_TRANSPORTS:
        raise FleetPolicyError(
            f"Unsupported transport '{transport}'; only Dual-SIM 5G/LTE MEC Zenoh "
            "or SX1262 GFSK GPS-TDMA lifeboat are permitted"
        )

    region_norm = region.strip().upper()
    if region_norm not in ("US", "EU"):
        raise ValueError(f"region must be 'US' or 'EU', got {region!r}")

    if not _is_exact_int(fleet_size) or fleet_size <= 0:
        raise ValueError("fleet_size must be a positive integer")
    if fleet_size > MAX_SECTOR_FLEET_SIZE:
        raise FleetPolicyError(
            f"fleet_size={fleet_size} exceeds single-sector cap ({MAX_SECTOR_FLEET_SIZE} AC); "
            "shard across regional MEC sectors"
        )
    if min_peer_distance_m < 0.0 or not math.isfinite(min_peer_distance_m):
        raise ValueError("min_peer_distance_m must be finite and non-negative")
    if r_min_m <= 0.0 or v_max_mps < 0.0:
        raise ValueError("r_min_m must be positive and v_max_mps non-negative")

    raw_peers = (fleet_size - 1) if peers_within_5km is None else int(peers_within_5km)
    if raw_peers < 0:
        raise ValueError("peers_within_5km must be non-negative")

    active_daa_burst = min_peer_distance_m < 2000.0

    if transport_norm == "dual_sim_5g_lte_mec_zenoh":
        culled_peers = min(raw_peers, MAX_MEC_ZENOH_PEERS_5KM)
        if fleet_size <= 10:
            nominal_hz = 10.0
            burst_hz = 20.0
        elif fleet_size <= 25:
            nominal_hz = 5.0
            burst_hz = 10.0
        else:
            nominal_hz = 5.0
            burst_hz = 10.0
        active_hz = burst_hz if active_daa_burst else nominal_hz
        phy_kbps = 10000.0
        duty_cap_pct = 100.0
    else:
        # SX1262 GFSK GPS-TDMA lifeboat
        culled_peers = min(raw_peers, fleet_size - 1)
        if region_norm == "EU":
            if fleet_size > MAX_EU_SX1262_CLUSTER_SIZE:
                raise FleetPolicyError(
                    f"EU ETSI g3 100 kbps SX1262 lifeboat strictly caps single RF cluster at "
                    f"<= {MAX_EU_SX1262_CLUSTER_SIZE} AC (requested {fleet_size}); spatial sub-clustering required"
                )
            nominal_hz = 5.0 if fleet_size <= 10 else 2.0
            burst_hz = nominal_hz
            active_hz = nominal_hz
            phy_kbps = 100.0
            duty_cap_pct = 10.0
        else:
            if fleet_size <= 10:
                nominal_hz = 10.0
            elif fleet_size <= 25:
                nominal_hz = 5.0
            else:
                nominal_hz = 1.0
            burst_hz = nominal_hz
            active_hz = nominal_hz
            phy_kbps = 200.0
            duty_cap_pct = 100.0

    dt_packet_s = 1.0 / active_hz
    expansion_m = v_max_mps * dt_packet_s
    effective_r_min_m = r_min_m + expansion_m

    return FleetDatalinkPlan(
        transport=transport_norm,
        region=region_norm,
        fleet_size=fleet_size,
        geo_culled_peers_5km=culled_peers,
        nominal_rate_hz=nominal_hz,
        burst_rate_hz=burst_hz,
        active_publish_rate_hz=active_hz,
        active_daa_burst=active_daa_burst,
        phy_bitrate_kbps=phy_kbps,
        duty_cycle_cap_pct=duty_cap_pct,
        dt_packet_s=dt_packet_s,
        ecbf_base_r_min_m=r_min_m,
        ecbf_barrier_expansion_m=expansion_m,
        ecbf_effective_r_min_m=effective_r_min_m,
    )


def compute_tactical_evasion_buffer_m(
    *,
    lateral_daa_envelope_m: float,
    r_min_m: float,
    v_max_mps: float,
    telemetry_age_s: float,
) -> float:
    """Two-Tier Volume4D Tactical Evasion Buffer padding (`DR-17` §2)."""
    if (
        lateral_daa_envelope_m < 0.0
        or r_min_m < 0.0
        or v_max_mps < 0.0
        or telemetry_age_s < 0.0
    ):
        raise ValueError("All tactical evasion buffer parameters must be non-negative")
    return lateral_daa_envelope_m + r_min_m + v_max_mps * telemetry_age_s


class StrategicTacticalInterlock:
    """ASTM F3548-21 <-> ASTM F3269-21 anti-oscillation sovereignty interlock (`DR-17` §2).

    Enforces:
      - Unconditional tactical RTA sovereignty (`strategic_freeze=True`) during any
        DAIDALUS / ECBF-QP override (incoming strategic waypoints are dropped).
      - Omission of `key` OVN array (`omit_key_ovn=True`) on nonconforming DSS broadcast
        for zero-blockage airspace sharing.
      - Dual spatial (`D > D_dwc + 1.25 * D_hyst` and `h(x) > h_reentry > 0`) AND
        temporal (`dwell >= 8.0 s`) hysteresis release before unlocking strategic replanning.
      - Irreversible escalation to `CONTINGENT` after `contingent_timeout_s` (default 30 s).
    """

    def __init__(
        self,
        *,
        d_dwc_m: float = 150.0,
        d_hyst_m: float = 40.0,
        h_reentry: float = 1.0,
        dwell_required_s: float = 8.0,
        contingent_timeout_s: float = 30.0,
    ) -> None:
        if d_dwc_m <= 0.0 or d_hyst_m <= 0.0 or h_reentry <= 0.0:
            raise ValueError("d_dwc_m, d_hyst_m, and h_reentry must be strictly positive")
        if dwell_required_s <= 0.0 or contingent_timeout_s <= dwell_required_s:
            raise ValueError("contingent_timeout_s must exceed positive dwell_required_s")
        self._d_dwc_m = float(d_dwc_m)
        self._d_hyst_m = float(d_hyst_m)
        self._h_reentry = float(h_reentry)
        self._dwell_required_s = float(dwell_required_s)
        self._contingent_timeout_s = float(contingent_timeout_s)
        self._state = InterlockState.NOMINAL_STRATEGIC
        self._clear_dwell_s = 0.0
        self._nonconforming_elapsed_s = 0.0

    @property
    def state(self) -> InterlockState:
        return self._state

    @property
    def spatial_release_distance_m(self) -> float:
        return self._d_dwc_m + 1.25 * self._d_hyst_m

    def step(
        self,
        *,
        dt_s: float,
        tactical_override_active: bool,
        min_intruder_dist_m: float,
        min_ecbf_h: float,
        incoming_strategic_waypoint: bool = False,
    ) -> InterlockStepResult:
        if dt_s < 0.0 or not math.isfinite(dt_s):
            raise ValueError("dt_s must be finite and non-negative")

        if self._state == InterlockState.CONTINGENT:
            self._nonconforming_elapsed_s += dt_s
            return InterlockStepResult(
                state=InterlockState.CONTINGENT,
                strategic_freeze=True,
                strategic_cmd_accepted=False,
                dss_operational_intent_state="CONTINGENT",
                omit_key_ovn=True,
                replan_unlocked=False,
                clear_dwell_s=0.0,
                nonconforming_elapsed_s=self._nonconforming_elapsed_s,
            )

        spatial_clear = (
            min_intruder_dist_m > self.spatial_release_distance_m
            and min_ecbf_h > self._h_reentry
        )

        if tactical_override_active:
            self._clear_dwell_s = 0.0
            self._nonconforming_elapsed_s += dt_s
            if self._nonconforming_elapsed_s >= self._contingent_timeout_s:
                self._state = InterlockState.CONTINGENT
                return InterlockStepResult(
                    state=InterlockState.CONTINGENT,
                    strategic_freeze=True,
                    strategic_cmd_accepted=False,
                    dss_operational_intent_state="CONTINGENT",
                    omit_key_ovn=True,
                    replan_unlocked=False,
                    clear_dwell_s=0.0,
                    nonconforming_elapsed_s=self._nonconforming_elapsed_s,
                )
            self._state = InterlockState.TACTICAL_OVERRIDE_FREEZE
            return InterlockStepResult(
                state=InterlockState.TACTICAL_OVERRIDE_FREEZE,
                strategic_freeze=True,
                strategic_cmd_accepted=False,
                dss_operational_intent_state="NONCONFORMING",
                omit_key_ovn=True,
                replan_unlocked=False,
                clear_dwell_s=0.0,
                nonconforming_elapsed_s=self._nonconforming_elapsed_s,
            )

        if self._state == InterlockState.NOMINAL_STRATEGIC:
            return InterlockStepResult(
                state=InterlockState.NOMINAL_STRATEGIC,
                strategic_freeze=False,
                strategic_cmd_accepted=bool(incoming_strategic_waypoint),
                dss_operational_intent_state="ACCEPTED",
                omit_key_ovn=False,
                replan_unlocked=False,
                clear_dwell_s=0.0,
                nonconforming_elapsed_s=0.0,
            )

        # Currently in TACTICAL_OVERRIDE_FREEZE or HYSTERESIS_RECOVERY
        self._nonconforming_elapsed_s += dt_s
        if spatial_clear:
            self._clear_dwell_s += dt_s
            if self._clear_dwell_s >= self._dwell_required_s:
                self._state = InterlockState.NOMINAL_STRATEGIC
                self._clear_dwell_s = 0.0
                self._nonconforming_elapsed_s = 0.0
                return InterlockStepResult(
                    state=InterlockState.NOMINAL_STRATEGIC,
                    strategic_freeze=False,
                    strategic_cmd_accepted=bool(incoming_strategic_waypoint),
                    dss_operational_intent_state="ACCEPTED",
                    omit_key_ovn=False,
                    replan_unlocked=True,
                    clear_dwell_s=self._dwell_required_s,
                    nonconforming_elapsed_s=0.0,
                )
            if self._nonconforming_elapsed_s >= self._contingent_timeout_s:
                self._state = InterlockState.CONTINGENT
                return InterlockStepResult(
                    state=InterlockState.CONTINGENT,
                    strategic_freeze=True,
                    strategic_cmd_accepted=False,
                    dss_operational_intent_state="CONTINGENT",
                    omit_key_ovn=True,
                    replan_unlocked=False,
                    clear_dwell_s=self._clear_dwell_s,
                    nonconforming_elapsed_s=self._nonconforming_elapsed_s,
                )
            self._state = InterlockState.HYSTERESIS_RECOVERY
            return InterlockStepResult(
                state=InterlockState.HYSTERESIS_RECOVERY,
                strategic_freeze=True,
                strategic_cmd_accepted=False,
                dss_operational_intent_state="NONCONFORMING",
                omit_key_ovn=True,
                replan_unlocked=False,
                clear_dwell_s=self._clear_dwell_s,
                nonconforming_elapsed_s=self._nonconforming_elapsed_s,
            )

        # Not spatially clear yet -> reset temporal dwell timer, remain frozen
        self._clear_dwell_s = 0.0
        if self._nonconforming_elapsed_s >= self._contingent_timeout_s:
            self._state = InterlockState.CONTINGENT
            return InterlockStepResult(
                state=InterlockState.CONTINGENT,
                strategic_freeze=True,
                strategic_cmd_accepted=False,
                dss_operational_intent_state="CONTINGENT",
                omit_key_ovn=True,
                replan_unlocked=False,
                clear_dwell_s=0.0,
                nonconforming_elapsed_s=self._nonconforming_elapsed_s,
            )
        self._state = InterlockState.TACTICAL_OVERRIDE_FREEZE
        return InterlockStepResult(
            state=InterlockState.TACTICAL_OVERRIDE_FREEZE,
            strategic_freeze=True,
            strategic_cmd_accepted=False,
            dss_operational_intent_state="NONCONFORMING",
            omit_key_ovn=True,
            replan_unlocked=False,
            clear_dwell_s=0.0,
            nonconforming_elapsed_s=self._nonconforming_elapsed_s,
        )


def parse_operator_fleet_ratio(ratio_spec: str) -> tuple[int, int]:
    """Parse an `'operators:aircraft'` ratio string (e.g. `'1:10'`) and enforce `<= 20` AC/operator."""
    parts = ratio_spec.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid operator ratio '{ratio_spec}'; expected '<operators>:<aircraft>'")
    operators = int(parts[0])
    aircraft = int(parts[1])
    if operators <= 0 or aircraft <= 0:
        raise ValueError("operators and aircraft counts must be positive")
    if (aircraft / operators) > float(MAX_FLEET_PER_OPERATOR):
        raise FleetPolicyError(
            f"Operator ratio '{ratio_spec}' ({aircraft / operators:.1f} AC/op) exceeds DR-17 / EASA HCD "
            f"hard ceiling of <= {MAX_FLEET_PER_OPERATOR} aircraft per operator"
        )
    return operators, aircraft


class HcdTriageQueue:
    """1-to-Many EASA HCD / NASA Ames operator triage queue (`DR-17` §3).

    Enforces:
      - `max_fleet_per_operator <= 20` (fail-closed rejection of `"1:100"` / `> 20` AC).
      - Hard cap of `<= 2` simultaneous active Focus Cards (`P0 + P1`).
      - Deterministic severity preemption (`P0` preempts `P1`) + vehicle offload signaling
        when `> 2` simultaneous urgent cards occur.
      - Regional coalescing of `>= 2` `P1` alerts sharing a `region_cluster_id` (`P0` never coalesced).
      - Automatic deferral of `P2/P3` alerts and P2 aural chime suppression while Focus Cards are active.
    """

    def __init__(
        self,
        *,
        fleet_size: int | str,
        max_fleet_per_operator: int = MAX_FLEET_PER_OPERATOR,
        max_active_focus_cards: int = MAX_ACTIVE_FOCUS_CARDS,
    ) -> None:
        if not _is_exact_int(max_fleet_per_operator) or not (
            1 <= max_fleet_per_operator <= MAX_FLEET_PER_OPERATOR
        ):
            raise FleetPolicyError(
                f"max_fleet_per_operator must be in 1..={MAX_FLEET_PER_OPERATOR}, got {max_fleet_per_operator!r}"
            )
        if not _is_exact_int(max_active_focus_cards) or not (
            1 <= max_active_focus_cards <= MAX_ACTIVE_FOCUS_CARDS
        ):
            raise FleetPolicyError(
                f"max_active_focus_cards must be in 1..={MAX_ACTIVE_FOCUS_CARDS}, got {max_active_focus_cards!r}"
            )

        if isinstance(fleet_size, str):
            operators, aircraft = parse_operator_fleet_ratio(fleet_size)
            effective_per_op = math.ceil(aircraft / operators)
            resolved_fleet = aircraft
        elif _is_exact_int(fleet_size) and fleet_size > 0:
            effective_per_op = fleet_size
            resolved_fleet = fleet_size
        else:
            raise ValueError(f"fleet_size must be positive int or 'm:N' ratio, got {fleet_size!r}")

        if effective_per_op > max_fleet_per_operator:
            raise FleetPolicyError(
                f"Fleet load {effective_per_op} AC/operator exceeds ceiling {max_fleet_per_operator} "
                "(DR-17 forbids unvalidated 1:100 supervision)"
            )

        self._fleet_size = resolved_fleet
        self._max_fleet_per_operator = max_fleet_per_operator
        self._max_active_focus_cards = max_active_focus_cards

    @property
    def fleet_size(self) -> int:
        return self._fleet_size

    @property
    def max_fleet_per_operator(self) -> int:
        return self._max_fleet_per_operator

    @property
    def max_active_focus_cards(self) -> int:
        return self._max_active_focus_cards

    def triage(self, alerts: Sequence[HcdAlert]) -> HcdTriageSnapshot:
        """Deterministically rank, coalesce, cap (`<= 2`), preempt, and signal peer offload."""
        urgent_p0: list[HcdAlert] = []
        urgent_p1: list[HcdAlert] = []
        low_alerts: list[HcdAlert] = []

        for alert in alerts:
            if alert.severity == AlertSeverity.P0_EMERGENCY:
                urgent_p0.append(alert)
            elif alert.severity == AlertSeverity.P1_HIGH:
                urgent_p1.append(alert)
            else:
                low_alerts.append(alert)

        candidate_cards: list[FocusCard] = []

        # P0 emergencies are NEVER coalesced so every life-safety anomaly has a dedicated card.
        for alert in urgent_p0:
            candidate_cards.append(
                FocusCard(
                    card_id=alert.alert_id,
                    severity=alert.severity,
                    aircraft_ids=(alert.aircraft_id,),
                    timestamp_s=alert.timestamp_s,
                    summary=alert.summary,
                    coalesced=False,
                )
            )

        # Group P1 alerts by region_cluster_id for regional coalescing (>= 2 alerts in same region)
        p1_clusters: dict[str, list[HcdAlert]] = {}
        p1_unclustered: list[HcdAlert] = []
        for alert in urgent_p1:
            if alert.region_cluster_id is not None and alert.region_cluster_id != "":
                cid = alert.region_cluster_id
                if cid in p1_clusters:
                    p1_clusters[cid].append(alert)
                else:
                    p1_clusters[cid] = [alert]
            else:
                p1_unclustered.append(alert)

        for alert in p1_unclustered:
            candidate_cards.append(
                FocusCard(
                    card_id=alert.alert_id,
                    severity=alert.severity,
                    aircraft_ids=(alert.aircraft_id,),
                    timestamp_s=alert.timestamp_s,
                    summary=alert.summary,
                    coalesced=False,
                )
            )

        for cid in sorted(p1_clusters):
            group = sorted(p1_clusters[cid], key=lambda item: (item.timestamp_s, item.alert_id))
            if len(group) == 1:
                only = group[0]
                candidate_cards.append(
                    FocusCard(
                        card_id=only.alert_id,
                        severity=only.severity,
                        aircraft_ids=(only.aircraft_id,),
                        timestamp_s=only.timestamp_s,
                        summary=only.summary,
                        coalesced=False,
                    )
                )
            else:
                ac_ids = tuple(sorted({item.aircraft_id for item in group}))
                candidate_cards.append(
                    FocusCard(
                        card_id=f"CLUSTER:{cid}",
                        severity=AlertSeverity.P1_HIGH,
                        aircraft_ids=ac_ids,
                        timestamp_s=group[0].timestamp_s,
                        summary=f"Regional P1 cluster '{cid}' ({len(ac_ids)} aircraft)",
                        coalesced=True,
                    )
                )

        candidate_cards.sort(key=lambda card: (int(card.severity), card.timestamp_s, card.card_id))

        active_cards = tuple(candidate_cards[: self._max_active_focus_cards])
        overflow_cards = tuple(candidate_cards[self._max_active_focus_cards :])

        offload_ac_list: list[str] = []
        for card in overflow_cards:
            for ac_id in card.aircraft_ids:
                if ac_id not in offload_ac_list:
                    offload_ac_list.append(ac_id)

        has_active_urgent = len(active_cards) > 0
        deferred = tuple(
            sorted(low_alerts, key=lambda item: (int(item.severity), item.timestamp_s, item.alert_id))
        ) if has_active_urgent else ()

        return HcdTriageSnapshot(
            fleet_size=self._fleet_size,
            max_fleet_per_operator=self._max_fleet_per_operator,
            active_focus_cards=active_cards,
            preempted_overflow_cards=overflow_cards,
            deferred_alerts=deferred,
            offload_required=len(overflow_cards) > 0,
            offload_recommended_ac_ids=tuple(offload_ac_list),
            p2_aural_chime_suppressed=has_active_urgent,
        )
