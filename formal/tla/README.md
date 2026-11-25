# OpenXVS Formal TLA+ Specifications & Rust Trace Refinement (`formal/tla/`)

Strictly civil/dual-use aviation Detect-and-Avoid (DAA) and Run-Time Assurance (RTA) formal specifications for **OpenXVS V5.0** (`ASTM F3269-21`, `EUROCAE ED-324 / SAE ARP6983`, `RTCA DO-365A/B`, `DO-333`).

> **Scope & Certification Disclaimer:** Neither `TLC`, `Apalache`, nor `Kani` is RTCA DO-330 TQL-4/5 qualified. All hardware, compiler, RTOS, airframe ARP4754B, and human DER/CVE certification gaps remain permanently isolated in `G5_GAP` (`assurance/openxvs_v2.gsn.yaml`).

## Files

| File | Purpose | Constants |
| :--- | :--- | :--- |
| `OpenXvsMultiShipRta.tla` | Multi-ship (`N=2..3`) sound abstraction of `xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`, Ring-0 arbiter, and lossy/delayed/Byzantine `XVS-MeshDAA` with Snowcat `@type` annotations and Apalache `AllSafetyInvariants` / `AllActionSafetyInvariants`. | Scaled bisimulation timers (`T_P0_REC=2`, `T_DWELL=2`, `T_CONSEC_DROP=1`, `T_SWERVE_FLIP=1`). |
| `OpenXvsMultiShipRta.cfg` | Exhaustive `N=2` safety + action-safety + temporal liveness proof (`USE_COORD_FIX = TRUE`). | `Ships = {1, 2}`, `USE_COORD_FIX = TRUE`. |
| `OpenXvsMultiShipRta_UncoordBug.cfg` | Negative witness proving asynchronous `12`-tick `preferred_swerve_sign` flip causes reciprocal observability-cancellation livelock without coordination. | `Ships = {1, 2}`, `USE_COORD_FIX = FALSE`. |
| `OpenXvsMultiShipRta_N3.cfg` | 3-ship (`N=3`) symmetry-reduced bounded-depth TLC safety check (`depth <= 6`, superseded to unbounded depth by Apalache Z3 inductive proofs). | `Ships = {s1, s2, s3}`, `USE_COORD_FIX = TRUE`. |
| `OpenXvsTraceRefinement.tla` | Cirstea–Kuppe–Loillier–Merz (`arXiv:2404.16075`) NDJSON trace-refinement checker validating `12,000` frozen-crate ticks + `2,000` coordinated `xvs-multiship-coord` (`F1 + F4 + F3`) ticks (`14,000` total). | **EXACT frozen Rust constants** (`P0_RECOVERY_TICKS=30`, `DWELL_TICKS=20`, `CONSEC_DROPOUT_MAX_TICKS=10`, `WINDOW_DROPOUT_20_MAX_TICKS=10`, `SWERVE_FLIP_PERIOD=12`). |
| `OpenXvsTraceRefinement.cfg` | TLC config enforcing `TraceTypeAndSafetyInv` on every tick and `POSTCONDITION TraceFullyAccepted` (`distinct == Len(Trace)`). | Reads `TRACE_PATH` env var (default `outputs/formal_tla/rust_trace.ndjson`). |

## One-Command Automated Reproduction

```powershell
python scripts/xvs/run_tla_verification.py
```

This script:
1. Builds and executes the standalone external Rust crate `tools/xvs-tla-trace/` (`cargo run --release`) to generate `14,000` adversarial ticks (`12,000` across 6 frozen-crate scenarios + `2,000` coordinated `xvs-multiship-coord` F1+F4+F3 ticks).
2. Locates Java 21+ (or downloads portable Temurin OpenJDK 21 JRE + MIT `tla2tools.jar` + MIT `CommunityModules-deps.jar` + Apache-2.0 `Apalache v0.63.0` into gitignored `output/formal_tla/.tools_cache/`).
3. Executes real `tlc2.TLC` on `OpenXvsMultiShipRta.tla` (`N=2`, uncoordinated witness, `N=3`) and `OpenXvsTraceRefinement.tla` (`14,000 / 14,000` ticks, `0` violations).
4. Executes real Apache-2.0 `apalache-mc` Z3 Snowcat typecheck, unbounded 1-inductive state & action safety proofs (`N=2` & `N=3`), and symbolic Z3 BMC (`N=2` & `N=3`, closing TLC's `N=3 depth <= 6` limit).
5. Binds the `4/4` WSL2 Kani CBMC proofs on `xvs-multiship-coord` (`18 + 4 = 22/22` OpenXVS total Kani proofs) with exact SHA-256 digests into `outputs/formal_tla/sn4_tla_evidence.json`.

## Direct Manual CLI Invocations

### 1. TLC Multi-Ship Safety & Liveness (`MIT tla2tools.jar`)

```bash
java -XX:+UseParallelGC -Xmx4g -cp output/formal_tla/.tools_cache/tla2tools.jar \
  tlc2.TLC -workers auto -cleanup \
  -config formal/tla/OpenXvsMultiShipRta.cfg \
  formal/tla/OpenXvsMultiShipRta.tla
```

### 2. TLC Mechanical Rust NDJSON Trace Validation (`MIT tla2tools.jar` + `CommunityModules-deps.jar`)

```bash
TRACE_PATH=outputs/formal_tla/rust_trace.ndjson \
java -XX:+UseParallelGC -Xmx4g \
  -Dtlc2.tool.queue.IStateQueue=StateDeque \
  -cp "output/formal_tla/.tools_cache/tla2tools.jar:output/formal_tla/.tools_cache/CommunityModules-deps.jar" \
  tlc2.TLC -workers 1 -cleanup -nowarning \
  -config formal/tla/OpenXvsTraceRefinement.cfg \
  formal/tla/OpenXvsTraceRefinement.tla
```

### 3. Apalache Symbolic Z3 Model Checking & Unbounded Inductive Proofs (`Apache-2.0 apalache-mc v0.63.0`)

```bash
apalache-mc typecheck formal/tla/OpenXvsMultiShipRta.tla
# Unbounded (∀ k >= 0) 1-inductive state & action safety proofs for N=2 and N=3:
apalache-mc check --cinit=ApalacheConstInitN3 --init=AllSafetyInvariants --next=Next \
  --inv=AllSafetyInvariants --length=1 formal/tla/OpenXvsMultiShipRta.tla
apalache-mc check --cinit=ApalacheConstInitN3 --init=AllSafetyInvariants --next=Next \
  --inv=AllActionSafetyInvariants --length=1 formal/tla/OpenXvsMultiShipRta.tla
# Reachable symbolic Z3 BMC (closing TLC's N=3 depth <= 6 limit):
apalache-mc check --cinit=ApalacheConstInitN3 --init=Init --next=Next \
  --inv=AllSafetyInvariants --length=8 formal/tla/OpenXvsMultiShipRta.tla
```
