------------------------- MODULE OpenXvsMultiShipRta -------------------------
(***************************************************************************)
(* OpenXVS V5.0 Formal Multi-Ship Simplex RTA, ActiveRange, Optical-ODD/   *)
(* VLM, OS-HAL, and Lossy/Delayed/Byzantine XVS-MeshDAA Specification.     *)
(*                                                                         *)
(* Strictly civil/dual-use aviation Detect-and-Avoid (DAA) and Run-Time    *)
(* Assurance (ASTM F3269-21, EUROCAE ED-324 / SAE ARP6983, RTCA DO-365A/B). *)
(* Zero military/weapons/targeting scope.                                  *)
(*                                                                         *)
(* Sound compositional predicate/interval abstraction of all 4 frozen      *)
(* #![no_std] Rust crates + XVS-MeshDAA #52001 air-to-air channel:         *)
(*   1. xvs-rta-core     (P0..P4, 0-tick preemption, P0->P1, P2->P1 d=1,   *)
(*                        non-resettable mandatory_go_around, dropouts)    *)
(*   2. xvs-active-range (NominalHold/ActiveSwerve/Locked/ImmediateEvade,  *)
(*                        0.08/0.10 sigma hysteresis, 12-tick swerve flip  *)
(*                        vs FAR 91.113 / aircraft-ID symmetry breaking)   *)
(*   3. xvs-odd-vlm      (pessimistic union cert \/ vlm, ACTUATOR_AUTH=F)  *)
(*   4. xvs-os-hal       (absorbing PluginOffline latch, max severity)     *)
(*   5. XVS-MeshDAA      (NOMINAL / DELAYED_LOSS / BYZANTINE_PEER medium)  *)
(*                                                                         *)
(* Timer Scaling Bisimulation Justification:                               *)
(*   Exact frozen constants (30, 20, 10, 10, 12) are verified directly     *)
(*   against 12,000+ real Rust ticks in OpenXvsTraceRefinement.tla.        *)
(*   Here, scaled positive thresholds (T_P0_REC >= 2, T_DWELL >= 2,        *)
(*   T_CONSEC_DROP >= 1, T_SWERVE_FLIP >= 1) preserve every strict dwell   *)
(*   barrier, coast timeout, P2->P1(dwell=1) step, and parity flip while   *)
(*   enabling fast (< 15 s) exhaustive finite-N TLC safety + liveness!     *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    \* @type: Set(Int);
    Ships,              \* Finite set of civil aircraft IDs, e.g. {1, 2} or {1, 2, 3}
    \* @type: Int;
    T_P0_REC,           \* Scaled P0 -> P1 recovery ticks (2 ~ 30 in frozen Rust)
    \* @type: Int;
    T_DWELL,            \* Scaled P1 -> P3/P4 & P3 -> P4 dwell ticks (2 ~ 20 in frozen Rust)
    \* @type: Int;
    T_CONSEC_DROP,      \* Scaled P2 coast cap (1 ~ 10 in frozen Rust; > cap => P0)
    \* @type: Int;
    T_SWERVE_FLIP,      \* Scaled preferred_swerve_sign half-period (1 ~ 12 in frozen Rust)
    \* @type: Bool;
    USE_COORD_FIX       \* FALSE: raw frozen 12-tick flip (reveals reciprocal livelock!)
                        \* TRUE:  v5.0 FAR 91.113 right-of-way symmetry breaking

ASSUME /\ IsFiniteSet(Ships)
       /\ Cardinality(Ships) >= 2
       /\ T_P0_REC >= 2
       /\ T_DWELL >= 2
       /\ T_CONSEC_DROP >= 1
       /\ T_SWERVE_FLIP >= 1
       /\ USE_COORD_FIX \in BOOLEAN

VARIABLES
    \* --- Per-ship xvs-rta-core state ---
    \* @type: Int -> Int;
    rtaMode,            \* 0=P0(InvalidInhibit), 1=P1(FallbackD0Safe), 2=P2(DegradedCoast),
                        \* 3=P3(Isolated5Cam), 4=P4(NominalMl)
    \* @type: Int -> Int;
    recTicks,           \* Consecutive P0 healthy ticks in 0..(T_P0_REC - 1)
    \* @type: Int -> Int;
    dwellTicks,         \* Consecutive P1/P3 dwell ticks in 0..(T_DWELL - 1)
    \* @type: Int -> Int;
    consecDrop,         \* Consecutive transient dropout ticks in 0..T_CONSEC_DROP
    \* @type: Int -> Bool;
    goAround,           \* Non-resettable mandatory_go_around latch
    \* @type: Int -> Bool;
    approachActive,     \* Active approach latch guard (TRUE while on final approach fault)

    \* --- Per-ship xvs-active-range state ---
    \* @type: Int -> Str;
    arMode,             \* {"NominalHold", "ActiveSwerve", "Locked", "ImmediateEvade"}
    \* @type: Int -> Str;
    sigmaBucket,        \* {"HIGH_SIGMA", "HYST_BAND", "LOCKED"} (0.08 / 0.10 hysteresis)
    \* @type: Int -> Bool;
    wasLocked,          \* Hysteresis lock memory
    \* @type: Int -> Int;
    swerveTicks,        \* ActiveSwerve counter mod (2 * T_SWERVE_FLIP)
    \* @type: Int -> Int;
    cmdBankSign,        \* ActiveRange requested bank sign in {-1, 0, 1}

    \* --- Per-ship xvs-os-hal Ring-1 supervisor ---
    \* @type: Int -> Str;
    pluginLife,         \* {"Healthy", "Degraded", "PluginOffline"}
    \* @type: Int -> Int;
    halTightenWire,     \* Ring-1 tightening floor in RTA wire ordering (4=none, 0=P0)

    \* --- Encounter-wide xvs-odd-vlm & XVS-MeshDAA #52001 channel ---
    \* @type: Bool;
    vlmCertHazard,      \* Certified geometric hazard bit
    \* @type: Bool;
    vlmFinalHazard,     \* Monotonic pessimistic union (vlmCertHazard \/ accepted_vlm)
    \* @type: Bool;
    vlmActuatorAuth,    \* Compile-time constant FALSE (ASTM F3269-21 §1.2)
    \* @type: Str;
    relRange,           \* {"CRITICAL", "CLOSING", "SAFE"}
    \* @type: Str;
    meshChan,           \* {"NOMINAL", "DELAYED_LOSS", "BYZANTINE_PEER"}
    \* @type: Str;
    envMode             \* {"ADVERSARIAL", "FAIR_RECOVER"}

vars == << rtaMode, recTicks, dwellTicks, consecDrop, goAround, approachActive,
           arMode, sigmaBucket, wasLocked, swerveTicks, cmdBankSign,
           pluginLife, halTightenWire,
           vlmCertHazard, vlmFinalHazard, vlmActuatorAuth,
           relRange, meshChan, envMode >>

-----------------------------------------------------------------------------
(* Pure Derived Operators & Symmetry Set (for Safety-Only N=3 Model Check) *)

ArModes       == {"NominalHold", "ActiveSwerve", "Locked", "ImmediateEvade"}
SigmaBuckets  == {"HIGH_SIGMA", "HYST_BAND", "LOCKED"}
PluginStates  == {"Healthy", "Degraded", "PluginOffline"}
RangeBuckets  == {"CRITICAL", "CLOSING", "SAFE"}
MeshStates    == {"NOMINAL", "DELAYED_LOSS", "BYZANTINE_PEER"}
EnvModes      == {"ADVERSARIAL", "FAIR_RECOVER"}

\* Permutation symmetry set (sound for safety invariants; NEVER used with liveness!)
Perms == Permutations(Ships)

\* Bounded BFS depth constraint used ONLY by OpenXvsMultiShipRta_N3.cfg (3-ship CI check)
DepthBoundN3 == TLCGet("level") <= 6

\* Apalache symbolic Z3 model-checking constant initializers (N=2 and N=3, USE_COORD_FIX=TRUE)
ApalacheConstInitN2 ==
    /\ Ships = {1, 2}
    /\ T_P0_REC = 2
    /\ T_DWELL = 2
    /\ T_CONSEC_DROP = 1
    /\ T_SWERVE_FLIP = 1
    /\ USE_COORD_FIX = TRUE

ApalacheConstInitN3 ==
    /\ Ships = {1, 2, 3}
    /\ T_P0_REC = 2
    /\ T_DWELL = 2
    /\ T_CONSEC_DROP = 1
    /\ T_SWERVE_FLIP = 1
    /\ USE_COORD_FIX = TRUE

Min2(a, b) == IF a <= b THEN a ELSE b

\* xvs-os-hal Monotonic Pessimistic Union in xvs-rta-core wire ordering:
\* Ring-1 can ONLY tighten (lower RTA wire mode = higher severity), NEVER relax.
\* When PluginOffline, Ring-1 output is dropped and EffectiveMode(s) = rtaMode[s].
EffectiveMode(s) ==
    IF pluginLife[s] = "Degraded"
    THEN Min2(rtaMode[s], halTightenWire[s])
    ELSE rtaMode[s]

InhibitTurns(s) == EffectiveMode(s) = 0

\* Ring-0 Cross-Layer Authority Precedence Lattice:
\* P0 (Inhibit) > ImmediateEvade (±2) > ActiveSwerve (±1) > NominalHold/Locked (0)
ActuatorBankCmd(s) ==
    IF InhibitTurns(s) THEN 0
    ELSE IF arMode[s] = "ImmediateEvade" THEN 2 * cmdBankSign[s]
    ELSE IF arMode[s] = "ActiveSwerve" THEN cmdBankSign[s]
    ELSE 0

\* Frozen xvs-active-range/src/lib.rs:242 preferred_swerve_sign:
RawPreferredSwerveSign(ticks) ==
    IF ((ticks \div T_SWERVE_FLIP) % 2) = 0 THEN 1 ELSE -1

\* Coordinated FAR 91.113 right-of-way precedence vs raw alternating flip:
ChosenSwerveSign(curTicks) ==
    IF USE_COORD_FIX THEN 1
    ELSE RawPreferredSwerveSign(curTicks)

\* Relative lateral observability condition det(FIM_rel) > 0:
\* Head-on Ship 1 (+N, right=+E) vs Ship 2 (-N, right=-E):
\*   Same body bank (+1, +1) -> inertial (+E, -E) -> a_rel = +2 != 0 (strong FIM + port-to-port)
\*   Opposite body bank (+1, -1) -> inertial (+E, +E) -> a_rel = 0 (CANCELLATION!)
\* @type: (Int, Int -> Int) => Bool;
RelativeObsActive(s, nextActCmd) ==
    /\ nextActCmd[s] # 0
    /\ IF USE_COORD_FIX
       THEN TRUE
       ELSE \E peer \in Ships \ {s} : (nextActCmd[s] + nextActCmd[peer]) # 0

-----------------------------------------------------------------------------
(* Type Invariant *)

TypeOK ==
    /\ rtaMode         \in [Ships -> 0..4]
    /\ recTicks        \in [Ships -> 0..(T_P0_REC - 1)]
    /\ dwellTicks      \in [Ships -> 0..(T_DWELL - 1)]
    /\ consecDrop      \in [Ships -> 0..T_CONSEC_DROP]
    /\ goAround        \in [Ships -> BOOLEAN]
    /\ approachActive  \in [Ships -> BOOLEAN]
    /\ arMode          \in [Ships -> ArModes]
    /\ sigmaBucket     \in [Ships -> SigmaBuckets]
    /\ wasLocked       \in [Ships -> BOOLEAN]
    /\ swerveTicks     \in [Ships -> 0..(2 * T_SWERVE_FLIP - 1)]
    /\ cmdBankSign     \in [Ships -> {-1, 0, 1}]
    /\ pluginLife      \in [Ships -> PluginStates]
    /\ halTightenWire  \in [Ships -> {0, 4}]
    /\ vlmCertHazard   \in BOOLEAN
    /\ vlmFinalHazard  \in BOOLEAN
    /\ vlmActuatorAuth \in BOOLEAN
    /\ relRange        \in RangeBuckets
    /\ meshChan        \in MeshStates
    /\ envMode         \in EnvModes

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ rtaMode         = [s \in Ships |-> 4]
    /\ recTicks        = [s \in Ships |-> 0]
    /\ dwellTicks      = [s \in Ships |-> 0]
    /\ consecDrop      = [s \in Ships |-> 0]
    /\ goAround        = [s \in Ships |-> FALSE]
    /\ approachActive  = [s \in Ships |-> FALSE]
    /\ arMode          = [s \in Ships |-> "NominalHold"]
    /\ sigmaBucket     = [s \in Ships |-> "HIGH_SIGMA"]
    /\ wasLocked       = [s \in Ships |-> FALSE]
    /\ swerveTicks     = [s \in Ships |-> 0]
    /\ cmdBankSign     = [s \in Ships |-> 0]
    /\ pluginLife      = [s \in Ships |-> "Healthy"]
    /\ halTightenWire  = [s \in Ships |-> 4]
    /\ vlmCertHazard   = FALSE
    /\ vlmFinalHazard  = FALSE
    /\ vlmActuatorAuth = FALSE
    /\ relRange        = "CLOSING"
    /\ meshChan        = "NOMINAL"
    /\ envMode         = "ADVERSARIAL"

-----------------------------------------------------------------------------
(* Exact xvs-rta-core Single-Ship Step Helper (matches simplex.rs 1:1) *)

RtaStepTuple(curMode, curRec, curDwell, curDrop, curGA,
             rawTarget, transientDrop, appActive, gaAck, appFault) ==
    LET nextDropRaw == IF transientDrop THEN curDrop + 1 ELSE 0
        nextGA ==
            IF appActive THEN (curGA \/ appFault)
            ELSE IF gaAck THEN FALSE
            ELSE curGA
        effTarget ==
            IF nextGA \/ nextDropRaw > T_CONSEC_DROP
            THEN 0
            ELSE rawTarget
        isP0Healthy == (effTarget \in {1, 3, 4}) /\ ~transientDrop
        storedDrop(m) == IF m = 2 THEN nextDropRaw ELSE 0
    IN
    IF effTarget < curMode THEN
        [mode |-> effTarget, rec |-> 0, dwell |-> 0,
         drop |-> storedDrop(effTarget), ga |-> nextGA]
    ELSE IF curMode = 0 THEN
        IF ~isP0Healthy THEN
            [mode |-> 0, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE IF curRec + 1 >= T_P0_REC THEN
            [mode |-> 1, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE
            [mode |-> 0, rec |-> curRec + 1, dwell |-> 0, drop |-> 0, ga |-> nextGA]
    ELSE IF curMode = 1 THEN
        IF effTarget \in {1, 2} THEN
            [mode |-> 1, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE IF curDwell + 1 >= T_DWELL THEN
            [mode |-> effTarget, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE
            [mode |-> 1, rec |-> 0, dwell |-> curDwell + 1, drop |-> 0, ga |-> nextGA]
    ELSE IF curMode = 2 THEN
        IF effTarget = 2 THEN
            [mode |-> 2, rec |-> 0, dwell |-> 0, drop |-> nextDropRaw, ga |-> nextGA]
        ELSE
            \* Exact frozen Rust simplex.rs:271: P2 exits to P1 with dwell_ticks = 1!
            [mode |-> 1, rec |-> 0, dwell |-> 1, drop |-> 0, ga |-> nextGA]
    ELSE IF curMode = 3 THEN
        IF effTarget = 3 THEN
            [mode |-> 3, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE IF curDwell + 1 >= T_DWELL THEN
            [mode |-> 4, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]
        ELSE
            [mode |-> 3, rec |-> 0, dwell |-> curDwell + 1, drop |-> 0, ga |-> nextGA]
    ELSE
        [mode |-> 4, rec |-> 0, dwell |-> 0, drop |-> 0, ga |-> nextGA]

-----------------------------------------------------------------------------
(* Exact xvs-active-range Hysteresis & Decision Helper *)

NextLockState(curWasLocked, nextSigma) ==
    IF nextSigma = "LOCKED" THEN TRUE
    ELSE IF nextSigma = "HIGH_SIGMA" THEN FALSE
    ELSE curWasLocked

ArStepTuple(curWasLocked, curSwerveTicks, nextSigma, curRange) ==
    LET nowLocked == NextLockState(curWasLocked, nextSigma)
        isClosing == curRange \in {"CRITICAL", "CLOSING"}
    IN
    IF curRange = "CRITICAL" THEN
        [mode |-> "ImmediateEvade", locked |-> nowLocked, ticks |-> 0, bank |-> 1]
    ELSE IF ~isClosing THEN
        [mode |-> IF nowLocked THEN "Locked" ELSE "NominalHold",
         locked |-> nowLocked, ticks |-> 0, bank |-> 0]
    ELSE IF nowLocked THEN
        [mode |-> "Locked", locked |-> TRUE, ticks |-> 0, bank |-> 0]
    ELSE
        LET sign == ChosenSwerveSign(curSwerveTicks)
            \* Under USE_COORD_FIX=TRUE, FAR 91.113 holds +1 without alternating flip counter
            nxtTicks == IF USE_COORD_FIX THEN 0 ELSE (curSwerveTicks + 1) % (2 * T_SWERVE_FLIP)
        IN [mode |-> "ActiveSwerve", locked |-> FALSE, ticks |-> nxtTicks, bank |-> sign]

-----------------------------------------------------------------------------
(* Orthogonal Factored Adversarial Actions *)

\* A1. Per-ship xvs-rta-core fault injection & recovery stepping
AdvRtaStep(s) ==
    /\ envMode = "ADVERSARIAL"
    /\ \E evt \in {"P0_FAULT", "P1_FAULT", "DROP_P2", "P3_5CAM", "NOMINAL_P4",
                   "GA_TRIP", "GA_ACK_ON_APP", "GA_ACK_CLEAR"} :
        LET rawT == CASE evt = "P0_FAULT"      -> 0
                      [] evt = "P1_FAULT"      -> 1
                      [] evt = "DROP_P2"       -> 2
                      [] evt = "P3_5CAM"       -> 3
                      [] OTHER                 -> 4
            tDrop == (evt = "DROP_P2")
            appAct == (evt \in {"GA_TRIP", "GA_ACK_ON_APP"})
            appFlt == (evt = "GA_TRIP")
            gaAck  == (evt \in {"GA_ACK_ON_APP", "GA_ACK_CLEAR"})
            rta == RtaStepTuple(rtaMode[s], recTicks[s], dwellTicks[s], consecDrop[s],
                                goAround[s], rawT, tDrop, appAct, gaAck, appFlt)
        IN
        /\ rtaMode'        = [rtaMode EXCEPT ![s] = rta.mode]
        /\ recTicks'       = [recTicks EXCEPT ![s] = rta.rec]
        /\ dwellTicks'     = [dwellTicks EXCEPT ![s] = rta.dwell]
        /\ consecDrop'     = [consecDrop EXCEPT ![s] = rta.drop]
        /\ goAround'       = [goAround EXCEPT ![s] = rta.ga]
        /\ approachActive' = [approachActive EXCEPT ![s] = (rta.ga /\ appAct)]
        /\ UNCHANGED << arMode, sigmaBucket, wasLocked, swerveTicks, cmdBankSign,
                        pluginLife, halTightenWire,
                        vlmCertHazard, vlmFinalHazard, vlmActuatorAuth,
                        relRange, meshChan, envMode >>

\* A2. Per-ship xvs-active-range asynchronous stepping (exercises phase offsets,
\*     0.08/0.10 hysteresis band retention, and ImmediateEvade escalation)
AdvActiveRangeStep(s) ==
    /\ envMode = "ADVERSARIAL"
    /\ LET allowedSig ==
            IF relRange = "CRITICAL" THEN {"HIGH_SIGMA"}
            ELSE IF relRange = "SAFE" THEN {"HIGH_SIGMA", "LOCKED"}
            ELSE SigmaBuckets
       IN
       \E nextSigma \in allowedSig :
        LET effSigma == IF sigmaBucket[s] = "LOCKED" /\ meshChan = "BYZANTINE_PEER"
                        THEN "LOCKED" ELSE nextSigma
            ar == ArStepTuple(wasLocked[s], swerveTicks[s], effSigma, relRange)
        IN
        /\ arMode'      = [arMode EXCEPT ![s] = ar.mode]
        /\ sigmaBucket' = [sigmaBucket EXCEPT ![s] = effSigma]
        /\ wasLocked'   = [wasLocked EXCEPT ![s] = ar.locked]
        /\ swerveTicks' = [swerveTicks EXCEPT ![s] = ar.ticks]
        /\ cmdBankSign' = [cmdBankSign EXCEPT ![s] = ar.bank]
        /\ UNCHANGED << rtaMode, recTicks, dwellTicks, consecDrop, goAround, approachActive,
                        pluginLife, halTightenWire,
                        vlmCertHazard, vlmFinalHazard, vlmActuatorAuth,
                        relRange, meshChan, envMode >>

\* A3. Per-ship xvs-os-hal Ring-1 supervisor fault & monotonic tightening
AdvHalStep(s) ==
    /\ envMode = "ADVERSARIAL"
    /\ \E halEvt \in {"HEALTHY", "TIGHTEN_P0", "FAULT_OFFLINE"} :
        LET nextLife ==
                IF pluginLife[s] = "PluginOffline" THEN "PluginOffline"
                ELSE IF halEvt = "FAULT_OFFLINE" THEN "PluginOffline"
                ELSE IF halEvt = "TIGHTEN_P0" THEN "Degraded"
                ELSE "Healthy"
            nextTight == IF nextLife = "Degraded" THEN 0 ELSE 4
        IN
        /\ pluginLife'     = [pluginLife EXCEPT ![s] = nextLife]
        /\ halTightenWire' = [halTightenWire EXCEPT ![s] = nextTight]
        /\ UNCHANGED << rtaMode, recTicks, dwellTicks, consecDrop, goAround, approachActive,
                        arMode, sigmaBucket, wasLocked, swerveTicks, cmdBankSign,
                        vlmCertHazard, vlmFinalHazard, vlmActuatorAuth,
                        relRange, meshChan, envMode >>

\* A4. Encounter-wide xvs-odd-vlm pessimistic union, MeshDAA channel & range geometry
\*     Exercises all 3 RangeBuckets, all 3 MeshStates, and all 3 legal (cert, final) VLM pairs.
AdvEnvChannelStep ==
    /\ envMode = "ADVERSARIAL"
    /\ \E regime \in {1, 2, 3} :
        LET nextRange == IF regime = 1 THEN "SAFE"
                         ELSE IF regime = 2 THEN "CLOSING"
                         ELSE "CRITICAL"
            nextMesh  == IF regime = 1 THEN "NOMINAL"
                         ELSE IF regime = 2 THEN "DELAYED_LOSS"
                         ELSE "BYZANTINE_PEER"
            certHaz   == (regime = 3)
            vlmHaz    == (regime = 2)
        IN
        /\ relRange'        = nextRange
        /\ meshChan'        = nextMesh
        /\ vlmCertHazard'   = certHaz
        /\ vlmFinalHazard'  = (certHaz \/ vlmHaz)
        /\ vlmActuatorAuth' = FALSE
        /\ UNCHANGED << rtaMode, recTicks, dwellTicks, consecDrop, goAround, approachActive,
                        arMode, sigmaBucket, wasLocked, swerveTicks, cmdBankSign,
                        pluginLife, halTightenWire, envMode >>

\* A5. Transition into Fair-Recovery / Fault-Free Environment (<>[] AllHealthy hypothesis)
StabilizeEnvironment ==
    /\ envMode = "ADVERSARIAL"
    /\ envMode' = "FAIR_RECOVER"
    /\ meshChan' = "NOMINAL"
    /\ vlmCertHazard' = FALSE
    /\ vlmFinalHazard' = FALSE
    /\ UNCHANGED << rtaMode, recTicks, dwellTicks, consecDrop, goAround, approachActive,
                    arMode, sigmaBucket, wasLocked, swerveTicks, cmdBankSign,
                    pluginLife, halTightenWire, vlmActuatorAuth, relRange >>

\* A6. Synchronous 10 Hz Multi-Ship Fair-Recovery Tick
FairRecoveryTick ==
    /\ envMode = "FAIR_RECOVER"
    /\ LET rtaNext == [s \in Ships |->
                RtaStepTuple(rtaMode[s], recTicks[s], dwellTicks[s], consecDrop[s],
                             goAround[s], 4, FALSE, FALSE, TRUE, FALSE)]
           lifeNext == [s \in Ships |->
                IF pluginLife[s] = "PluginOffline" THEN "PluginOffline" ELSE "Healthy"]
           tightNext == [s \in Ships |-> 4]
           arPre == [s \in Ships |->
                ArStepTuple(wasLocked[s], swerveTicks[s], sigmaBucket[s], relRange)]
           actCmd == [s \in Ships |->
                IF rtaNext[s].mode = 0 THEN 0
                ELSE IF arPre[s].mode = "ImmediateEvade" THEN 2 * arPre[s].bank
                ELSE IF arPre[s].mode = "ActiveSwerve" THEN arPre[s].bank
                ELSE 0]
           sigmaNext == [s \in Ships |->
                IF sigmaBucket[s] = "LOCKED" THEN "LOCKED"
                ELSE IF arPre[s].mode = "ActiveSwerve" /\ RelativeObsActive(s, actCmd) THEN
                    IF sigmaBucket[s] = "HIGH_SIGMA" THEN "HYST_BAND" ELSE "LOCKED"
                ELSE sigmaBucket[s]]
           arFinal == [s \in Ships |->
                ArStepTuple(wasLocked[s], swerveTicks[s], sigmaNext[s], relRange)]
           rangeNext ==
                IF relRange = "SAFE" THEN "SAFE"
                ELSE IF \E s \in Ships : (arFinal[s].mode = "ImmediateEvade" /\ rtaNext[s].mode > 0)
                THEN "SAFE"
                ELSE IF \A s \in Ships : arFinal[s].mode = "Locked"
                THEN "SAFE"
                ELSE relRange
       IN
       /\ rtaMode'         = [s \in Ships |-> rtaNext[s].mode]
       /\ recTicks'        = [s \in Ships |-> rtaNext[s].rec]
       /\ dwellTicks'      = [s \in Ships |-> rtaNext[s].dwell]
       /\ consecDrop'      = [s \in Ships |-> rtaNext[s].drop]
       /\ goAround'        = [s \in Ships |-> rtaNext[s].ga]
       /\ approachActive'  = [s \in Ships |-> FALSE]
       /\ arMode'          = [s \in Ships |-> arFinal[s].mode]
       /\ sigmaBucket'     = sigmaNext
       /\ wasLocked'       = [s \in Ships |-> arFinal[s].locked]
       /\ swerveTicks'     = [s \in Ships |-> arFinal[s].ticks]
       /\ cmdBankSign'     = [s \in Ships |-> arFinal[s].bank]
       /\ pluginLife'      = lifeNext
       /\ halTightenWire'  = tightNext
       /\ vlmCertHazard'   = FALSE
       /\ vlmFinalHazard'  = FALSE
       /\ vlmActuatorAuth' = FALSE
       /\ relRange'        = rangeNext
       /\ meshChan'        = "NOMINAL"
       /\ UNCHANGED envMode

Next ==
    \/ \E s \in Ships : AdvRtaStep(s)
    \/ \E s \in Ships : AdvActiveRangeStep(s)
    \/ \E s \in Ships : AdvHalStep(s)
    \/ AdvEnvChannelStep
    \/ StabilizeEnvironment
    \/ FairRecoveryTick

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(FairRecoveryTick)

-----------------------------------------------------------------------------
(* State Safety Invariants *)

SafetyInv_P0InhibitsTurns ==
    \A s \in Ships :
        (rtaMode[s] = 0 \/ EffectiveMode(s) = 0)
            => (InhibitTurns(s) /\ ActuatorBankCmd(s) = 0)

SafetyInv_ZeroVlmActuatorAuth == ~vlmActuatorAuth

SafetyInv_PessimisticUnion ==
    /\ (vlmCertHazard => vlmFinalHazard)
    /\ \A s \in Ships :
        /\ (EffectiveMode(s) <= rtaMode[s])
        /\ (pluginLife[s] = "PluginOffline" => EffectiveMode(s) = rtaMode[s])

SafetyInv_GoAroundForcesP0 ==
    \A s \in Ships :
        goAround[s] => (rtaMode[s] = 0 /\ EffectiveMode(s) = 0 /\ InhibitTurns(s))

SafetyInv_CoastBounded ==
    \A s \in Ships :
        (rtaMode[s] = 2) => (consecDrop[s] >= 1 /\ consecDrop[s] <= T_CONSEC_DROP)

SafetyInv_SigmaHysteresis ==
    \A s \in Ships :
        /\ (sigmaBucket[s] = "LOCKED" => wasLocked[s])
        /\ (sigmaBucket[s] = "HIGH_SIGMA" => ~wasLocked[s])

SafetyInv_HoldAndLockZeroBank ==
    \A s \in Ships :
        (arMode[s] \in {"NominalHold", "Locked"}) => (cmdBankSign[s] = 0)

SafetyInv_TimerDiscipline ==
    \A s \in Ships :
        /\ (rtaMode[s] # 0 => recTicks[s] = 0)
        /\ (rtaMode[s] \notin {1, 3} => dwellTicks[s] = 0)
        /\ (rtaMode[s] # 2 => consecDrop[s] = 0)

\* Conjunction of all 9 state safety invariants (used for Apalache symbolic BMC & inductive checks)
AllSafetyInvariants ==
    /\ TypeOK
    /\ SafetyInv_P0InhibitsTurns
    /\ SafetyInv_ZeroVlmActuatorAuth
    /\ SafetyInv_PessimisticUnion
    /\ SafetyInv_GoAroundForcesP0
    /\ SafetyInv_CoastBounded
    /\ SafetyInv_SigmaHysteresis
    /\ SafetyInv_HoldAndLockZeroBank
    /\ SafetyInv_TimerDiscipline

-----------------------------------------------------------------------------
(* Action-Level Temporal Safety Properties *)

Prop_PluginOfflineAbsorbing ==
    [][\A s \in Ships : pluginLife[s] = "PluginOffline" => pluginLife'[s] = "PluginOffline"]_vars

Prop_P0ExitsOnlyToP1 ==
    [][\A s \in Ships : rtaMode[s] = 0 => rtaMode'[s] \in {0, 1}]_vars

Prop_P1NeverClimbsToP2 ==
    [][\A s \in Ships : rtaMode[s] = 1 => rtaMode'[s] # 2]_vars

Prop_GoAroundNonResettable ==
    [][\A s \in Ships : (goAround[s] /\ approachActive'[s]) => goAround'[s]]_vars

\* Action-level safety invariant conjunction (checked inductively by Apalache Z3 for N=2 & N=3)
AllActionSafetyInvariants ==
    /\ \A s \in Ships : pluginLife[s] = "PluginOffline" => pluginLife'[s] = "PluginOffline"
    /\ \A s \in Ships : rtaMode[s] = 0 => rtaMode'[s] \in {0, 1}
    /\ \A s \in Ships : rtaMode[s] = 1 => rtaMode'[s] # 2
    /\ \A s \in Ships : (goAround[s] /\ approachActive'[s]) => goAround'[s]

-----------------------------------------------------------------------------
(* Temporal Liveness Properties *)

Live_RtaRecoveryToNominal ==
    \A s \in Ships :
        (envMode = "FAIR_RECOVER" /\ rtaMode[s] < 4) ~> (rtaMode[s] = 4)

Live_MultiShipResolution ==
    (envMode = "FAIR_RECOVER") ~>
        ((\A s \in Ships : arMode[s] = "Locked") \/ relRange = "SAFE")

=============================================================================
