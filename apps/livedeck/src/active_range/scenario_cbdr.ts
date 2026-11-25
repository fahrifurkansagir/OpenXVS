/**
 * Closed-Loop Unknown-Size Head-On CBDR Scenario Controller for OpenXVS v3.0 Observatory.
 *
 * Drives ownship's real Rust 6-DoF WASM RK4 physics (`WasmLiveDeckSim`) cleanly via
 * `set_faults(..., manual_override=true, manual_bank_cmd_rad=out.cmdBankRad, ...)` against
 * a deterministic unknown-wingspan CBDR intruder — leaving frozen `xvs-core/` and `M1..M4`
 * Anti-Scripting Lock 100% untouched.
 *
 * Emits `ActiveRangeOverlayState` for Pass 3's collapsing 3D CRLB/Conformal range cone
 * + `SC3BF` barrier sheath, and a live `IntruderTrack` for 3D aircraft rendering.
 */

import {
  DEG_TO_RAD,
  geodeticToEcef,
  mat3MulVec,
  mat3Transpose,
  rEcefNed,
  type Vec3,
} from "../rhi/coords.ts";
import type { ActiveRangeOverlayState } from "../passes/pass3_airframe_traffic.ts";
import type { IntruderTrack } from "../passes/types.ts";
import {
  ActiveRangeControllerTs,
  ActiveRangeModeTs,
  G_ACCEL_MPS2,
  R_SEP_M,
  type ActiveRangeTelemetryTs,
} from "./active_range_ts.ts";

export interface CbdrScenarioOptions {
  readonly initialRangeM?: number;
  readonly closingSpeedMps?: number;
  readonly intruderWingspanM?: number;
  readonly ownshipTasMps?: number;
  readonly focalPx?: number;
}

export interface CbdrWasmStepInput {
  readonly simTimeS: number;
  readonly dtS: number;
  readonly latRad: number;
  readonly lonRad: number;
  readonly altWgs84M: number;
  readonly pEcefBody: Vec3;
  readonly vNedMps: Vec3;
  readonly yawRad: number;
  readonly rollRad: number;
  readonly tasMps: number;
}

export interface CbdrStepOutput {
  readonly telemetry: ActiveRangeTelemetryTs;
  readonly manualOverride: boolean;
  readonly manualBankCmdRad: number;
  readonly trueRangeM: number;
  readonly intruderTrack: IntruderTrack;
  readonly overlayState: ActiveRangeOverlayState;
}

export class CbdrScenarioController {
  private readonly controller: ActiveRangeControllerTs;
  private readonly initialRangeM: number;
  private readonly closingSpeedMps: number;
  private readonly intruderWingspanM: number;
  private readonly focalPx: number;

  // Deterministic ECEF intruder state anchored in fixed local inertial NED frame
  private anchored = false;
  private anchorREcefNed: ReturnType<typeof rEcefNed> | null = null;
  private anchorRNedEcef: ReturnType<typeof rEcefNed> | null = null;
  private prevOwnPosEcef: Vec3 | null = null;
  private prevOwnVelNed: Vec3 | null = null;
  private intrPosEcef: Vec3 = [0, 0, 0];
  private intrVelEcef: Vec3 = [0, 0, 0];
  private trackAge = 4;
  private prevSizePx = 0.0;
  private lastTelemetry: ActiveRangeTelemetryTs | null = null;

  // Autonomous local-NED kinematic state (for unit testing without WASM)
  private autoOwnPos: [number, number, number] = [0, 0, 0];
  private autoOwnHdgRad = 0.0;
  private autoOwnTasMps: number;
  private autoIntrPos: [number, number, number];
  private autoIntrVel: [number, number, number];
  private autoBankRad = 0.0;

  constructor(opts: CbdrScenarioOptions = {}) {
    this.initialRangeM = opts.initialRangeM ?? 1400.0;
    this.closingSpeedMps = opts.closingSpeedMps ?? 80.0;
    this.intruderWingspanM = opts.intruderWingspanM ?? 9.5;
    this.autoOwnTasMps = opts.ownshipTasMps ?? 45.0;
    this.focalPx = opts.focalPx ?? 2690.0;
    this.controller = new ActiveRangeControllerTs({
      sigmaBearingRad: 0.00055,
      qHatConformal: 0.925,
    });
    const intrSpeed = Math.max(15.0, this.closingSpeedMps - this.autoOwnTasMps);
    this.autoIntrPos = [this.initialRangeM, 0.0, 0.0];
    this.autoIntrVel = [-intrSpeed, 0.0, 0.0];
  }

  reset(): void {
    this.controller.reset();
    this.anchored = false;
    this.anchorREcefNed = null;
    this.anchorRNedEcef = null;
    this.prevOwnPosEcef = null;
    this.prevOwnVelNed = null;
    this.trackAge = 4;
    this.prevSizePx = 0.0;
    this.lastTelemetry = null;
    this.autoOwnPos = [0, 0, 0];
    this.autoOwnHdgRad = 0.0;
    this.autoBankRad = 0.0;
    const intrSpeed = Math.max(15.0, this.closingSpeedMps - this.autoOwnTasMps);
    this.autoIntrPos = [this.initialRangeM, 0.0, 0.0];
    this.autoIntrVel = [-intrSpeed, 0.0, 0.0];
  }

  getLastTelemetry(): ActiveRangeTelemetryTs | null {
    return this.lastTelemetry;
  }

  /** Step live CBDR intruder against real Rust 6-DoF WASM ownship telemetry. */
  stepWithWasmOwnship(inp: CbdrWasmStepInput): CbdrStepOutput {
    const dtS = Math.max(1e-3, Math.min(0.25, inp.dtS));

    // Anchor head-on intruder dead ahead of ownship in a fixed local inertial NED frame
    if (!this.anchored || !this.anchorREcefNed || !this.anchorRNedEcef) {
      const rEcefFromNed = rEcefNed(inp.latRad, inp.lonRad);
      const rNedFromEcef = mat3Transpose(rEcefFromNed);
      this.anchorREcefNed = rEcefFromNed;
      this.anchorRNedEcef = rNedFromEcef;

      const fwdN = Math.cos(inp.yawRad);
      const fwdE = Math.sin(inp.yawRad);
      const intrRelNed: Vec3 = [fwdN * this.initialRangeM, fwdE * this.initialRangeM, -14.0];
      const intrSpeed = Math.max(18.0, this.closingSpeedMps - Math.max(25.0, inp.tasMps));
      const intrVelNed: Vec3 = [-fwdN * intrSpeed, -fwdE * intrSpeed, 0.0];
      const dEcef = mat3MulVec(rEcefFromNed, intrRelNed);
      this.intrPosEcef = [
        inp.pEcefBody[0] + dEcef[0],
        inp.pEcefBody[1] + dEcef[1],
        inp.pEcefBody[2] + dEcef[2],
      ];
      this.intrVelEcef = mat3MulVec(rEcefFromNed, intrVelNed);
      this.prevOwnPosEcef = [inp.pEcefBody[0], inp.pEcefBody[1], inp.pEcefBody[2]];
      this.prevOwnVelNed = [inp.vNedMps[0], inp.vNedMps[1], inp.vNedMps[2]];
      this.anchored = true;
    } else {
      this.intrPosEcef = [
        this.intrPosEcef[0] + this.intrVelEcef[0] * dtS,
        this.intrPosEcef[1] + this.intrVelEcef[1] * dtS,
        this.intrPosEcef[2] + this.intrVelEcef[2] * dtS,
      ];
    }

    const rNedFromEcef = this.anchorRNedEcef;
    const relEcef: Vec3 = [
      this.intrPosEcef[0] - inp.pEcefBody[0],
      this.intrPosEcef[1] - inp.pEcefBody[1],
      this.intrPosEcef[2] - inp.pEcefBody[2],
    ];
    const relNed = mat3MulVec(rNedFromEcef, relEcef);
    const trueRangeM = Math.hypot(relNed[0], relNed[1], relNed[2]);

    // Loop cleanly if intruder passes behind ownship during long interactive viewing
    if (trueRangeM < 190.0 || trueRangeM > 5200.0) {
      this.reset();
      return this.stepWithWasmOwnship(inp);
    }

    const horiz = Math.max(1e-3, Math.hypot(relNed[0], relNed[1]));
    const alphaRad = Math.atan2(relNed[1], relNed[0]);
    const betaRad = Math.atan2(-relNed[2], horiz);

    const sizePx = (this.focalPx * this.intruderWingspanM) / Math.max(trueRangeM, 10.0);
    const sizeRatePxS =
      this.prevSizePx > 0 ? (sizePx - this.prevSizePx) / dtS : (sizePx * this.closingSpeedMps) / trueRangeM;
    this.prevSizePx = sizePx;
    this.trackAge++;

    // Exact inertial ownship velocity & lateral acceleration matching WASM's true ECEF displacement
    let currVelNed: Vec3 = [inp.vNedMps[0], inp.vNedMps[1], inp.vNedMps[2]];
    let aNed: readonly [number, number, number];
    if (this.prevOwnPosEcef && this.prevOwnVelNed && this.trackAge > 5) {
      const dispEcef: Vec3 = [
        inp.pEcefBody[0] - this.prevOwnPosEcef[0],
        inp.pEcefBody[1] - this.prevOwnPosEcef[1],
        inp.pEcefBody[2] - this.prevOwnPosEcef[2],
      ];
      const dispNed = mat3MulVec(rNedFromEcef, dispEcef);
      const dispNorm = Math.hypot(dispNed[0], dispNed[1], dispNed[2]);
      if (dispNorm > 0.1) {
        currVelNed = [dispNed[0] / dtS, dispNed[1] / dtS, dispNed[2] / dtS];
      }
      aNed = [
        Math.max(-12.0, Math.min(12.0, (currVelNed[0] - this.prevOwnVelNed[0]) / dtS)),
        Math.max(-12.0, Math.min(12.0, (currVelNed[1] - this.prevOwnVelNed[1]) / dtS)),
        Math.max(-12.0, Math.min(12.0, (currVelNed[2] - this.prevOwnVelNed[2]) / dtS)),
      ];
    } else {
      const aLat = G_ACCEL_MPS2 * Math.tan(Math.max(-0.45, Math.min(0.45, inp.rollRad)));
      aNed = [-Math.sin(inp.yawRad) * aLat, Math.cos(inp.yawRad) * aLat, 0.0];
    }
    this.prevOwnPosEcef = [inp.pEcefBody[0], inp.pEcefBody[1], inp.pEcefBody[2]];
    this.prevOwnVelNed = currVelNed;

    const telemetry = this.controller.step(
      {
        vNed: currVelNed,
        aNed,
        headingRad: inp.yawRad,
        tasMps: Math.max(25.0, inp.tasMps),
        vStallMps: 14.5,
      },
      {
        alphaRad,
        betaRad,
        sizePx,
        sizeRatePxS,
        contrastRatio: 0.65,
        skyBetaExt: 2.0e-4,
        daytimeClear: true,
        trackAge: this.trackAge,
        dtS,
      },
    );
    this.lastTelemetry = telemetry;

    const intrVelNed = mat3MulVec(rNedFromEcef, this.intrVelEcef);
    const vRelN = intrVelNed[0] - inp.vNedMps[0];
    const vRelE = intrVelNed[1] - inp.vNedMps[1];
    const vRelD = intrVelNed[2] - inp.vNedMps[2];
    const closingMps = -(relNed[0] * vRelN + relNed[1] * vRelE + relNed[2] * vRelD) / Math.max(trueRangeM, 1.0);
    const ttcS = closingMps > 0.5 ? Math.min(999, trueRangeM / closingMps) : 999;

    const alertTier: 0 | 1 | 2 | 3 =
      telemetry.mode === ActiveRangeModeTs.ImmediateEvade || trueRangeM < 450
        ? 3
        : telemetry.mode === ActiveRangeModeTs.ActiveSwerve || trueRangeM < 1100
          ? 2
          : 1;

    const intruderTrack: IntruderTrack = {
      id: "CBDR-1",
      posEcef: this.intrPosEcef,
      velEcef: this.intrVelEcef,
      rangeM: trueRangeM,
      ttcS,
      alertTier,
      wcvRadiusM: R_SEP_M,
    };

    const overlayState: ActiveRangeOverlayState = {
      ownshipEcef: inp.pEcefBody,
      targetPosEcef: this.intrPosEcef,
      rHatM: telemetry.rHatM,
      sigmaRM: telemetry.sigmaRM,
      rSafeLowerM: telemetry.rSafeLowerM,
      minSc3bfMargin: telemetry.minSc3bfMargin,
      sc3bfRadiusM: R_SEP_M,
      mode: telemetry.mode,
    };

    return {
      telemetry,
      manualOverride: telemetry.mode === ActiveRangeModeTs.ActiveSwerve || telemetry.mode === ActiveRangeModeTs.ImmediateEvade,
      manualBankCmdRad: telemetry.cmdBankRad,
      trueRangeM,
      intruderTrack,
      overlayState,
    };
  }

  /** Pure closed-loop local-NED step (used by Vitest without needing WASM runtime). */
  stepAutonomous(dtS = 0.1): CbdrStepOutput {
    const dt = Math.max(1e-3, Math.min(0.25, dtS));
    const v = this.autoOwnTasMps;
    const aLat = G_ACCEL_MPS2 * Math.tan(this.autoBankRad);
    const psiDot = aLat / v;
    const hdgMid = this.autoOwnHdgRad + 0.5 * psiDot * dt;
    const vNed: readonly [number, number, number] = [v * Math.cos(this.autoOwnHdgRad), v * Math.sin(this.autoOwnHdgRad), 0.0];
    const aNed: readonly [number, number, number] = [-aLat * Math.sin(this.autoOwnHdgRad), aLat * Math.cos(this.autoOwnHdgRad), 0.0];

    const relN = this.autoIntrPos[0] - this.autoOwnPos[0];
    const relE = this.autoIntrPos[1] - this.autoOwnPos[1];
    const relD = this.autoIntrPos[2] - this.autoOwnPos[2];
    const trueRangeM = Math.hypot(relN, relE, relD);
    const horiz = Math.max(1e-3, Math.hypot(relN, relE));
    const alphaRad = Math.atan2(relE, relN);
    const betaRad = Math.atan2(-relD, horiz);
    const sizePx = (this.focalPx * this.intruderWingspanM) / Math.max(trueRangeM, 10.0);
    const sizeRatePxS =
      this.prevSizePx > 0 ? (sizePx - this.prevSizePx) / dt : (sizePx * this.closingSpeedMps) / trueRangeM;
    this.prevSizePx = sizePx;
    this.trackAge++;

    const telemetry = this.controller.step(
      { vNed, aNed, headingRad: this.autoOwnHdgRad, tasMps: v, vStallMps: 14.5 },
      {
        alphaRad,
        betaRad,
        sizePx,
        sizeRatePxS,
        contrastRatio: 0.65,
        skyBetaExt: 2.0e-4,
        daytimeClear: true,
        trackAge: this.trackAge,
        dtS: dt,
      },
    );
    this.lastTelemetry = telemetry;

    // Advance 6-DoF coordinated-turn kinematics for next tick
    this.autoOwnPos = [
      this.autoOwnPos[0] + v * Math.cos(hdgMid) * dt,
      this.autoOwnPos[1] + v * Math.sin(hdgMid) * dt,
      this.autoOwnPos[2],
    ];
    this.autoOwnHdgRad += psiDot * dt;
    this.autoIntrPos = [
      this.autoIntrPos[0] + this.autoIntrVel[0] * dt,
      this.autoIntrPos[1] + this.autoIntrVel[1] * dt,
      this.autoIntrPos[2] + this.autoIntrVel[2] * dt,
    ];
    this.autoBankRad = telemetry.cmdBankRad;

    const pOwnEcef = geodeticToEcef(37.61356 * DEG_TO_RAD, -122.35714 * DEG_TO_RAD, 300.0);
    const rNed = rEcefNed(37.61356 * DEG_TO_RAD, -122.35714 * DEG_TO_RAD);
    const dEcef = mat3MulVec(rNed, [relN, relE, relD]);
    const pIntrEcef: Vec3 = [pOwnEcef[0] + dEcef[0], pOwnEcef[1] + dEcef[1], pOwnEcef[2] + dEcef[2]];

    return {
      telemetry,
      manualOverride: telemetry.mode === ActiveRangeModeTs.ActiveSwerve || telemetry.mode === ActiveRangeModeTs.ImmediateEvade,
      manualBankCmdRad: telemetry.cmdBankRad,
      trueRangeM,
      intruderTrack: {
        id: "CBDR-1",
        posEcef: pIntrEcef,
        rangeM: trueRangeM,
        ttcS: Math.min(999, trueRangeM / Math.max(this.closingSpeedMps, 1.0)),
        alertTier: 2,
        wcvRadiusM: R_SEP_M,
      },
      overlayState: {
        ownshipEcef: pOwnEcef,
        targetPosEcef: pIntrEcef,
        rHatM: telemetry.rHatM,
        sigmaRM: telemetry.sigmaRM,
        rSafeLowerM: telemetry.rSafeLowerM,
        minSc3bfMargin: telemetry.minSc3bfMargin,
        sc3bfRadiusM: R_SEP_M,
        mode: telemetry.mode,
      },
    };
  }
}
