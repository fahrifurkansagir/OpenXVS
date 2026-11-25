----------------------- MODULE OpenXvsTraceRefinement -----------------------
(***************************************************************************)
(* OpenXVS V5.0 Mechanical Rust <-> TLA+ NDJSON Trace Refinement Spec.     *)
(*                                                                         *)
(* Follows the Cirstea–Kuppe–Loillier–Merz TLC Trace Validation method     *)
(* (arXiv:2404.16075, SoSyM 2026) using MIT-licensed CommunityModules      *)
(* Json!ndJsonDeserialize + IOUtils!IOEnv.                                 *)
(*                                                                         *)
(* ZERO timer scaling, ZERO crate modifications: validates 10,000+ real    *)
(* adversarial execution ticks emitted by the frozen #![no_std] Rust       *)
(* crates using their EXACT frozen production constants:                   *)
(*   - xvs-rta-core:     P0_RECOVERY_TICKS=30, DWELL_TICKS=20,             *)
(*                       CONSEC_DROPOUT_MAX_TICKS=10,                      *)
(*                       WINDOW_DROPOUT_20_MAX_TICKS=10,                   *)
(*                       P2 -> P1 (dwell_ticks = 1), 0-tick preemption     *)
(*   - xvs-active-range: SWERVE_FLIP_PERIOD=12, 0.08/0.10 sigma hysteresis *)
(*                       Universal SC3BF + 47 deg FOV + <= 12 deg swerve   *)
(*   - xvs-odd-vlm:      ACTUATOR_AUTHORITY=FALSE, AIRDATA_EXPORT_BLOCKED, *)
(*                       monotonic pessimistic union (cert | active_vlm)   *)
(*   - xvs-os-hal:       permanent absorbing PluginOffline latch,          *)
(*                       monotonic tightening (out_wire <= base_wire),     *)
(*                       ACTUATOR_AUTHORITY=FALSE                          *)
(*   - Ring-0 Arbiter:   P0 / HAL-P0 strictly zeroes avoidance bank cmd    *)
(***************************************************************************)
EXTENDS Integers, Sequences, TLC, Json, IOUtils

\* Exact frozen Rust constants (xvs-core/crates/xvs-rta-core/src/odd_generated.rs
\* and xvs-active-range/src/lib.rs — ZERO scaling!)
P0_RECOVERY_TICKS           == 30
DWELL_TICKS                 == 20
CONSEC_DROPOUT_MAX_TICKS    == 10
WINDOW_DROPOUT_20_MAX_TICKS == 10
SWERVE_FLIP_PERIOD          == 12

TracePath ==
    IF "TRACE_PATH" \in DOMAIN IOEnv
    THEN IOEnv.TRACE_PATH
    ELSE "outputs/formal_tla/rust_trace.ndjson"

Trace == ndJsonDeserialize(TracePath)

VARIABLES
    l,                  \* Current 1-based NDJSON trace index in 1..Len(Trace)
    rta_mode,           \* xvs-rta-core SimplexMode wire u8 (0..4)
    rta_consec_drop,    \* xvs-rta-core consec_dropout_ticks
    rta_rec_ticks,      \* xvs-rta-core recovery_ticks
    rta_dwell_ticks,    \* xvs-rta-core dwell_ticks
    rta_go_around,      \* xvs-rta-core mandatory_go_around latch
    rta_inhibit_turns,  \* xvs-rta-core inhibit_avoidance_turns
    rta_blank_hud,      \* xvs-rta-core blank_conformal_hud
    ar_mode,            \* xvs-active-range ActiveRangeMode u8 (0..3)
    ar_was_locked,      \* xvs-active-range hysteresis lock memory
    ar_swerve_ticks,    \* xvs-active-range swerve_ticks counter
    vlm_act_auth,       \* xvs-odd-vlm ACTUATOR_AUTHORITY (always FALSE)
    vlm_cert_mask,      \* xvs-odd-vlm certified_hazards_mask (0..255)
    vlm_final_mask,     \* xvs-odd-vlm final_hazards_mask (0..255)
    odd_airdata_blocked,\* xvs-odd-vlm OpticalOddMonitor AF447 invariant (always TRUE)
    hal_lifecycle,      \* xvs-os-hal PluginLifecycle u8 (0..3; 3=PluginOffline)
    hal_base_mode_wire, \* xvs-os-hal base_mode in RTA wire ordering (0..4)
    hal_out_mode_wire,  \* xvs-os-hal out_mode in RTA wire ordering (0..4)
    hal_act_auth,       \* xvs-os-hal ACTUATOR_AUTHORITY (always FALSE)
    arbiter_bank_sign   \* Ring-0 Arbiter effective bank command sign (-2..2)

vars == << l, rta_mode, rta_consec_drop, rta_rec_ticks, rta_dwell_ticks,
           rta_go_around, rta_inhibit_turns, rta_blank_hud,
           ar_mode, ar_was_locked, ar_swerve_ticks,
           vlm_act_auth, vlm_cert_mask, vlm_final_mask, odd_airdata_blocked,
           hal_lifecycle, hal_base_mode_wire, hal_out_mode_wire, hal_act_auth,
           arbiter_bank_sign >>

-----------------------------------------------------------------------------
(* Pure Bitwise Subset Helper for u8 Hazard Masks *)

BitSet(mask, b) == ((mask \div (2^b)) % 2) = 1
BitmaskSubset(sub, sup) == \A b \in 0..7 : BitSet(sub, b) => BitSet(sup, b)

-----------------------------------------------------------------------------
(* State-Level Consistency Predicate Checked on Every Single Rust Tick *)

ValidRecordInvariants(rec) ==
    /\ rec.rta_mode \in 0..4
    /\ rec.rta_raw_target \in 0..4
    /\ rec.rta_consec_drop >= 0
    /\ rec.rta_win_drop_20 \in 0..20
    /\ rec.rta_rec_ticks \in 0..(P0_RECOVERY_TICKS - 1)
    /\ rec.rta_dwell_ticks \in 0..(DWELL_TICKS - 1)
    /\ rec.rta_go_around \in BOOLEAN
    /\ rec.rta_inhibit_turns = (rec.rta_mode = 0)
    /\ rec.rta_blank_hud = (rec.rta_mode = 0)
    /\ (rec.rta_go_around => rec.rta_raw_target = 0 /\ rec.rta_mode = 0)
    /\ (rec.rta_consec_drop > CONSEC_DROPOUT_MAX_TICKS => rec.rta_raw_target = 0 /\ rec.rta_mode = 0)
    /\ (rec.rta_win_drop_20 > WINDOW_DROPOUT_20_MAX_TICKS => rec.rta_raw_target = 0 /\ rec.rta_mode = 0)
    /\ (rec.rta_mode = 2 => (rec.rta_consec_drop >= 1
                             /\ rec.rta_consec_drop <= CONSEC_DROPOUT_MAX_TICKS
                             /\ rec.rta_win_drop_20 <= WINDOW_DROPOUT_20_MAX_TICKS))
    /\ rec.rta_p0_healthy = ((rec.rta_raw_target \in {1, 3, 4}) /\ ~rec.rta_transient_drop)
    \* xvs-active-range invariants
    /\ rec.ar_mode \in 0..3
    /\ rec.ar_was_locked \in BOOLEAN
    /\ rec.ar_swerve_ticks >= 0
    /\ rec.ar_pref_sign \in {-1, 1}
    /\ rec.ar_sigma_bucket \in {"HIGH_SIGMA", "HYST_BAND", "LOCKED"}
    /\ (rec.ar_sigma_bucket = "LOCKED" => rec.ar_was_locked)
    /\ (rec.ar_sigma_bucket = "HIGH_SIGMA" => ~rec.ar_was_locked)
    /\ (rec.ar_mode = 1 => (rec.ar_sc3bf_all_safe /\ rec.ar_fov_safe
                            /\ rec.ar_bank_within_12deg /\ rec.ar_bank_sign # 0))
    /\ (rec.ar_mode \in {0, 2} => rec.ar_bank_sign = 0)
    /\ (rec.ar_mode = 3 => rec.ar_bank_sign # 0)
    \* xvs-odd-vlm invariants
    /\ rec.vlm_status \in 0..5
    /\ rec.vlm_act_auth = FALSE
    /\ rec.odd_airdata_blocked = TRUE
    /\ rec.vlm_cert_mask \in 0..255
    /\ rec.vlm_final_mask \in 0..255
    /\ BitmaskSubset(rec.vlm_cert_mask, rec.vlm_final_mask)
    \* xvs-os-hal invariants
    /\ rec.hal_lifecycle \in 0..3
    /\ rec.hal_base_mode_wire \in 0..4
    /\ rec.hal_out_mode_wire \in 0..4
    /\ rec.hal_act_auth = FALSE
    /\ rec.hal_out_mode_wire <= rec.hal_base_mode_wire
    /\ (rec.hal_lifecycle = 3 => rec.hal_out_mode_wire = rec.hal_base_mode_wire)
    \* Ring-0 Cross-Layer Authority Precedence: P0 or HAL-P0 strictly zeroes avoidance bank
    /\ ((rec.rta_inhibit_turns \/ rec.hal_out_mode_wire = 0) => rec.arbiter_bank_sign = 0)
    \* xvs-multiship-coord (F1 + F4 + F3) coordinated refinement invariants (USE_COORD_FIX = TRUE)
    /\ rec.coord_enabled \in BOOLEAN
    /\ rec.coord_head_on \in BOOLEAN
    /\ rec.coord_role \in {"FrozenUncoord", "OpticalFar91113Right",
                           "CooperativeRightProbe", "YieldHoldForPeerProbe",
                           "InhibitedByP0"}
    /\ rec.mesh_guard_ghosts_admitted >= 0
    /\ rec.mesh_guard_sticky_active \in BOOLEAN
    /\ rec.mesh_guard_stale_rejected >= 0
    /\ rec.mesh_guard_replay_rejected >= 0
    /\ (rec.coord_enabled =>
            /\ rec.coord_role # "FrozenUncoord"
            /\ (rec.coord_head_on => rec.ar_pref_sign = 1)
            /\ ((rec.coord_head_on /\ rec.ar_mode = 1) => rec.ar_bank_sign = 1)
            /\ (rec.coord_role = "YieldHoldForPeerProbe" => (rec.ar_mode = 0 /\ rec.ar_bank_sign = 0))
            /\ ((rec.rta_inhibit_turns \/ rec.hal_out_mode_wire = 0) =>
                    (rec.coord_role = "InhibitedByP0" /\ rec.ar_mode = 0
                     /\ rec.ar_bank_sign = 0 /\ rec.arbiter_bank_sign = 0))
            /\ rec.mesh_guard_ghosts_admitted = 0)
    /\ (~rec.coord_enabled => rec.coord_role = "FrozenUncoord")

-----------------------------------------------------------------------------
(* Exact Frozen Rust Transition Relation Between Consecutive Ticks (cur -> nxt) *)

ExactRtaTransition(cur, nxt) ==
    /\ nxt.rta_consec_drop = IF nxt.rta_transient_drop THEN cur.rta_consec_drop + 1 ELSE 0
    /\ \* Non-resettable mandatory_go_around while approach_active
       (cur.rta_go_around /\ nxt.rta_approach_active) => nxt.rta_go_around
    /\ (~nxt.rta_approach_active /\ nxt.rta_go_around_ack) => ~nxt.rta_go_around
    /\ (~nxt.rta_approach_active /\ ~nxt.rta_go_around_ack) => (nxt.rta_go_around = cur.rta_go_around)
    /\ \* Total Next-State & Counter Transition Table (xvs-rta-core/src/simplex.rs:231-289)
       IF nxt.rta_raw_target < cur.rta_mode THEN
           \* 0-tick downward preemption
           /\ nxt.rta_mode = nxt.rta_raw_target
           /\ nxt.rta_rec_ticks = 0
           /\ nxt.rta_dwell_ticks = 0
       ELSE IF cur.rta_mode = 0 THEN
           IF ~nxt.rta_p0_healthy THEN
               /\ nxt.rta_mode = 0
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE IF cur.rta_rec_ticks + 1 >= P0_RECOVERY_TICKS THEN
               /\ nxt.rta_mode = 1
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE
               /\ nxt.rta_mode = 0
               /\ nxt.rta_rec_ticks = cur.rta_rec_ticks + 1
               /\ nxt.rta_dwell_ticks = 0
       ELSE IF cur.rta_mode = 1 THEN
           IF nxt.rta_raw_target \in {1, 2} THEN
               \* P1 NEVER climbs upward into P2 (DegradedCoast)
               /\ nxt.rta_mode = 1
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE IF cur.rta_dwell_ticks + 1 >= DWELL_TICKS THEN
               /\ nxt.rta_mode = nxt.rta_raw_target
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE
               /\ nxt.rta_mode = 1
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = cur.rta_dwell_ticks + 1
       ELSE IF cur.rta_mode = 2 THEN
           IF nxt.rta_raw_target = 2 THEN
               /\ nxt.rta_mode = 2
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE
               \* Exact frozen Rust P2 -> P1 with dwell_ticks = 1
               /\ nxt.rta_mode = 1
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 1
       ELSE IF cur.rta_mode = 3 THEN
           IF nxt.rta_raw_target = 3 THEN
               /\ nxt.rta_mode = 3
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE IF cur.rta_dwell_ticks + 1 >= DWELL_TICKS THEN
               /\ nxt.rta_mode = 4
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = 0
           ELSE
               /\ nxt.rta_mode = 3
               /\ nxt.rta_rec_ticks = 0
               /\ nxt.rta_dwell_ticks = cur.rta_dwell_ticks + 1
       ELSE
           /\ nxt.rta_mode = 4
           /\ nxt.rta_rec_ticks = 0
           /\ nxt.rta_dwell_ticks = 0

ExactActiveRangeTransition(cur, nxt) ==
    /\ IF nxt.coord_enabled /\ nxt.coord_head_on THEN
           \* xvs-multiship-coord FAR 91.113 non-reversing right-turn precedence (USE_COORD_FIX = TRUE)
           nxt.ar_pref_sign = 1
       ELSE
           \* Frozen xvs-active-range/src/lib.rs:242 preferred_swerve_sign evaluated on pre-step swerve_ticks
           nxt.ar_pref_sign = IF ((cur.ar_swerve_ticks \div SWERVE_FLIP_PERIOD) % 2) = 0 THEN 1 ELSE -1
    /\ nxt.ar_swerve_ticks = IF nxt.ar_mode = 1 THEN cur.ar_swerve_ticks + 1 ELSE 0
    /\ \* 0.08 / 0.10 hysteresis lock retention inside HYST_BAND
       (nxt.ar_sigma_bucket = "HYST_BAND" => nxt.ar_was_locked = cur.ar_was_locked)

ExactHalTransition(cur, nxt) ==
    \* Permanent mid-flight absorbing PluginOffline latch (xvs-os-hal/src/lib.rs:459)
    (cur.hal_lifecycle = 3) => (nxt.hal_lifecycle = 3)

ExactRustStep(cur, nxt) ==
    /\ ValidRecordInvariants(nxt)
    /\ IF nxt.tick = 0 THEN
           \* Clean scenario boundary (independent scenario reset)
           TRUE
       ELSE
           /\ nxt.scenario = cur.scenario
           /\ nxt.ship_id = cur.ship_id
           /\ nxt.tick = cur.tick + 1
           /\ ExactRtaTransition(cur, nxt)
           /\ ExactActiveRangeTransition(cur, nxt)
           /\ ExactHalTransition(cur, nxt)

-----------------------------------------------------------------------------
(* Trace-Driven State Machine *)

BindRecord(rec) ==
    /\ rta_mode'            = rec.rta_mode
    /\ rta_consec_drop'     = rec.rta_consec_drop
    /\ rta_rec_ticks'       = rec.rta_rec_ticks
    /\ rta_dwell_ticks'     = rec.rta_dwell_ticks
    /\ rta_go_around'       = rec.rta_go_around
    /\ rta_inhibit_turns'   = rec.rta_inhibit_turns
    /\ rta_blank_hud'       = rec.rta_blank_hud
    /\ ar_mode'             = rec.ar_mode
    /\ ar_was_locked'       = rec.ar_was_locked
    /\ ar_swerve_ticks'     = rec.ar_swerve_ticks
    /\ vlm_act_auth'        = rec.vlm_act_auth
    /\ vlm_cert_mask'       = rec.vlm_cert_mask
    /\ vlm_final_mask'      = rec.vlm_final_mask
    /\ odd_airdata_blocked' = rec.odd_airdata_blocked
    /\ hal_lifecycle'       = rec.hal_lifecycle
    /\ hal_base_mode_wire'  = rec.hal_base_mode_wire
    /\ hal_out_mode_wire'   = rec.hal_out_mode_wire
    /\ hal_act_auth'        = rec.hal_act_auth
    /\ arbiter_bank_sign'   = rec.arbiter_bank_sign

TraceInit ==
    /\ Len(Trace) >= 1
    /\ ValidRecordInvariants(Trace[1])
    /\ l = 1
    /\ rta_mode            = Trace[1].rta_mode
    /\ rta_consec_drop     = Trace[1].rta_consec_drop
    /\ rta_rec_ticks       = Trace[1].rta_rec_ticks
    /\ rta_dwell_ticks     = Trace[1].rta_dwell_ticks
    /\ rta_go_around       = Trace[1].rta_go_around
    /\ rta_inhibit_turns   = Trace[1].rta_inhibit_turns
    /\ rta_blank_hud       = Trace[1].rta_blank_hud
    /\ ar_mode             = Trace[1].ar_mode
    /\ ar_was_locked       = Trace[1].ar_was_locked
    /\ ar_swerve_ticks     = Trace[1].ar_swerve_ticks
    /\ vlm_act_auth        = Trace[1].vlm_act_auth
    /\ vlm_cert_mask       = Trace[1].vlm_cert_mask
    /\ vlm_final_mask      = Trace[1].vlm_final_mask
    /\ odd_airdata_blocked = Trace[1].odd_airdata_blocked
    /\ hal_lifecycle       = Trace[1].hal_lifecycle
    /\ hal_base_mode_wire  = Trace[1].hal_base_mode_wire
    /\ hal_out_mode_wire   = Trace[1].hal_out_mode_wire
    /\ hal_act_auth        = Trace[1].hal_act_auth
    /\ arbiter_bank_sign   = Trace[1].arbiter_bank_sign

AdvanceTrace ==
    /\ l < Len(Trace)
    /\ ExactRustStep(Trace[l], Trace[l + 1])
    /\ l' = l + 1
    /\ BindRecord(Trace[l + 1])

TraceDone ==
    /\ l = Len(Trace)
    /\ UNCHANGED vars

TraceNext == AdvanceTrace \/ TraceDone

TraceSpec ==
    /\ TraceInit
    /\ [][TraceNext]_vars
    /\ WF_vars(AdvanceTrace)

-----------------------------------------------------------------------------
(* Trace Invariants & Completeness Acceptance Condition *)

TraceTypeAndSafetyInv ==
    /\ l \in 1..Len(Trace)
    /\ rta_mode \in 0..4
    /\ rta_inhibit_turns = (rta_mode = 0)
    /\ rta_blank_hud = (rta_mode = 0)
    /\ vlm_act_auth = FALSE
    /\ odd_airdata_blocked = TRUE
    /\ hal_act_auth = FALSE
    /\ hal_out_mode_wire <= hal_base_mode_wire
    /\ BitmaskSubset(vlm_cert_mask, vlm_final_mask)
    /\ ((rta_inhibit_turns \/ hal_out_mode_wire = 0) => arbiter_bank_sign = 0)

\* Kuppe Trace-Acceptance Postcondition:
\* Every single Rust trace tick 1..Len(Trace) MUST be accepted without premature deadlock!
TraceFullyAccepted ==
    TLCGet("stats").distinct = Len(Trace)

=============================================================================
