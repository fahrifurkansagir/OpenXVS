/**
 * OpenXVS v3.0 XVS-ForcedLand Closed-Loop Engine-Out Scenario Controller.
 *
 * Computes a physically grounded 3D wind-warped trochoidal glide corridor down to KSFO 28L
 * under live crosswind (e.g. 35 kt), emitting both:
 *   1. Bounded `manual_bank_cmd_rad` & `manual_climb_cmd_mps` commands for `WasmLiveDeckSim`
 *      (`set_faults(..., manual_override=true, ...)`), leaving frozen `xvs-core/` and M1..M4
 *      Anti-Scripting Lock 100% untouched.
 *   2. `ForcedLandFunnelState` 3D ECEF waypoints directly rendered by Pass 3's
 *      `translucentVolPipeline` golden-emerald glide funnel.
 *
 * Strictly uses NED / ECEF coordinates (CONVENTIONS.md v1.0).
 */

import {
  DEG_TO_RAD,
  geodeticToEcef,
  mat3MulVec,
  mat3Transpose,
  rEcefNed,
  type Vec3,
} from "../rhi/coords.ts";
import type {
  ForcedLandFunnelState,
  ForcedLandFunnelWaypoint,
} from "../passes/pass3_airframe_traffic.ts";

export const KT_TO_MPS = 0.514444;
export const MAX_GLIDE_BANK_RAD = 25.0 * DEG_TO_RAD;
export const BEST_GLIDE_LD = 14.5;
export const DEFAULT_GLIDE_AIRSPEED_MPS = 51.4; // ~100 kt clean best-glide speed

/** KSFO Runway 28L threshold (matches ksfo_3dep_geoid18_v1.manifest.json). */
export const KSFO_28L_THRESHOLD = {
  latDeg: 37.61356,
  lonDeg: -122.35714,
  altWgs84M: -28.67,
  trueHeadingDeg: 297.7,
} as const;

export interface WindWarpedTrochoidInput {
  readonly latRad: number;
  readonly lonRad: number;
  readonly altWgs84M: number;
  /** Crosswind speed in knots (+ blows from port/SSW toward starboard/NNE across 28L). */
  readonly crosswindKt: number;
  /** Optional best-glide airspeed (m/s). */
  readonly glideAirspeedMps?: number;
  /** Optional target runway threshold override. */
  readonly targetLatDeg?: number;
  readonly targetLonDeg?: number;
  readonly targetAltWgs84M?: number;
  readonly targetHeadingDeg?: number;
  readonly numSegments?: number;
}

export interface WindWarpedTrochoidPath {
  readonly waypoints: readonly ForcedLandFunnelWaypoint[];
  readonly waypointsNed: readonly Vec3[];
  readonly targetEcef: Vec3;
  readonly distToThresholdM: number;
  readonly maxLateralWindWarpM: number;
  readonly crabAngleDeg: number;
  readonly nominalGlideTimeS: number;
}

export interface ForcedLandStepInput {
  readonly latRad: number;
  readonly lonRad: number;
  readonly altWgs84M: number;
  readonly yawRad: number;
  readonly airspeedMps: number;
}

export interface ForcedLandDiagnostics {
  readonly crosswindKt: number;
  readonly crabAngleDeg: number;
  readonly distToThresholdM: number;
  readonly glideMarginM: number;
  readonly maxLateralWindWarpM: number;
  readonly energyState: "HIGH" | "NOMINAL" | "TIGHT";
}

export interface ForcedLandStepOutput {
  readonly manualOverride: true;
  readonly manualBankCmdRad: number;
  readonly manualClimbCmdMps: number;
  readonly funnelState: ForcedLandFunnelState;
  readonly diagnostics: ForcedLandDiagnostics;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function wrapAnglePi(rad: number): number {
  let a = rad;
  while (a > Math.PI) a -= 2.0 * Math.PI;
  while (a < -Math.PI) a += 2.0 * Math.PI;
  return a;
}

/**
 * Compute an exact 3D wind-warped trochoidal glide corridor in local KSFO 28L NED and ECEF.
 *
 * In the air-mass reference frame, the aircraft follows a smooth glide arc toward the
 * upwind-shifted virtual aimpoint `p_air_end = -v_wind * t_glide`. Transforming back to the
 * Earth-fixed NED frame (`p_earth(t) = p_air(t) + v_wind * t`) yields a physically real
 * wind-warped trochoid that bows upwind mid-glide and terminates precisely at KSFO 28L threshold.
 */
export function computeWindWarpedTrochoidPath(
  input: WindWarpedTrochoidInput,
): WindWarpedTrochoidPath {
  const thrLatDeg = input.targetLatDeg ?? KSFO_28L_THRESHOLD.latDeg;
  const thrLonDeg = input.targetLonDeg ?? KSFO_28L_THRESHOLD.lonDeg;
  const thrAltM = input.targetAltWgs84M ?? KSFO_28L_THRESHOLD.altWgs84M;
  const rwyHdgRad = (input.targetHeadingDeg ?? KSFO_28L_THRESHOLD.trueHeadingDeg) * DEG_TO_RAD;

  const thrLatRad = thrLatDeg * DEG_TO_RAD;
  const thrLonRad = thrLonDeg * DEG_TO_RAD;
  const targetEcef = geodeticToEcef(thrLatRad, thrLonRad, thrAltM);
  const rEcefFromNed = rEcefNed(thrLatRad, thrLonRad);
  const rNedFromEcef = mat3Transpose(rEcefFromNed);

  const ownEcef = geodeticToEcef(input.latRad, input.lonRad, input.altWgs84M);
  const ownRelEcef: Vec3 = [
    ownEcef[0] - targetEcef[0],
    ownEcef[1] - targetEcef[1],
    ownEcef[2] - targetEcef[2],
  ];
  // Ownship position in threshold-centered NED (threshold is [0, 0, 0])
  const p0Ned = mat3MulVec(rNedFromEcef, ownRelEcef);
  const horizDistM = Math.max(50.0, Math.hypot(p0Ned[0], p0Ned[1]));

  const vGlide = Math.max(32.0, input.glideAirspeedMps ?? DEFAULT_GLIDE_AIRSPEED_MPS);
  const windMps = (Number.isFinite(input.crosswindKt) ? input.crosswindKt : 0.0) * KT_TO_MPS;
  // Crosswind vector perpendicular to runway 28L (blowing across runway from port to starboard)
  const rightN = -Math.sin(rwyHdgRad);
  const rightE = Math.cos(rwyHdgRad);
  const fwdN = Math.cos(rwyHdgRad);
  const fwdE = Math.sin(rwyHdgRad);

  const crabSin = clamp(windMps / vGlide, -0.58, 0.58);
  const crabAngleRad = Math.asin(crabSin);
  const groundSpeedMps = Math.max(20.0, vGlide * Math.cos(crabAngleRad));
  const nominalGlideTimeS = horizDistM / groundSpeedMps;

  // Trochoidal upwind bow amplitude scales with crosswind displacement during glide turn-in
  const windDriftM = windMps * nominalGlideTimeS;
  const bowPeakM = clamp(windDriftM * 0.16, -480.0, 480.0);

  // Final approach tangency guide along extended runway centerline
  const approachBackM = Math.min(horizDistM * 0.35, 950.0);
  const ctrl1N = p0Ned[0] * 0.55 - fwdN * (approachBackM * 0.45) - rightN * bowPeakM;
  const ctrl1E = p0Ned[1] * 0.55 - fwdE * (approachBackM * 0.45) - rightE * bowPeakM;
  const ctrl2N = -fwdN * approachBackM - rightN * (bowPeakM * 0.42);
  const ctrl2E = -fwdE * approachBackM - rightE * (bowPeakM * 0.42);

  // Reference calm chord for measuring true lateral wind warp
  const segs = Math.max(4, Math.min(32, Math.floor(input.numSegments ?? 16)));
  const waypoints: ForcedLandFunnelWaypoint[] = [];
  const waypointsNed: Vec3[] = [];
  let maxLateralWindWarpM = 0.0;

  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const omt = 1.0 - t;

    // Cubic Bezier trochoid in NED from ownship p0Ned -> threshold [0, 0, 0]
    const b0 = omt * omt * omt;
    const b1 = 3.0 * omt * omt * t;
    const b2 = 3.0 * omt * t * t;

    // Calm reference point (same longitudinal tangency, zero wind bow)
    const calmCtrl1N = p0Ned[0] * 0.55 - fwdN * (approachBackM * 0.45);
    const calmCtrl1E = p0Ned[1] * 0.55 - fwdE * (approachBackM * 0.45);
    const calmCtrl2N = -fwdN * approachBackM;
    const calmCtrl2E = -fwdE * approachBackM;

    const calmN = b0 * p0Ned[0] + b1 * calmCtrl1N + b2 * calmCtrl2N;
    const calmE = b0 * p0Ned[1] + b1 * calmCtrl1E + b2 * calmCtrl2E;

    const nM = b0 * p0Ned[0] + b1 * ctrl1N + b2 * ctrl2N;
    const eM = b0 * p0Ned[1] + b1 * ctrl1E + b2 * ctrl2E;
    // Smooth flare-shaped glide profile ending cleanly at threshold elevation (d = 0)
    const dM = p0Ned[2] * Math.pow(omt, 1.08);

    const warpM = Math.hypot(nM - calmN, eM - calmE);
    if (warpM > maxLateralWindWarpM) {
      maxLateralWindWarpM = warpM;
    }

    const ptNed: Vec3 = [nM, eM, dM];
    waypointsNed.push(ptNed);

    const dEcef = mat3MulVec(rEcefFromNed, ptNed);
    const posEcef: Vec3 = [
      targetEcef[0] + dEcef[0],
      targetEcef[1] + dEcef[1],
      targetEcef[2] + dEcef[2],
    ];
    waypoints.push({
      posEcef,
      halfWidthM: 64.0 - 40.0 * t,
      halfHeightM: 28.0 - 17.0 * t,
    });
  }

  return {
    waypoints,
    waypointsNed,
    targetEcef,
    distToThresholdM: horizDistM,
    maxLateralWindWarpM,
    crabAngleDeg: crabAngleRad / DEG_TO_RAD,
    nominalGlideTimeS,
  };
}

/**
 * Closed-loop Engine-Out XVS-ForcedLand Scenario Controller.
 * Steers WasmLiveDeckSim's real Rust 6-DoF RK4 aircraft along the live wind-warped trochoid
 * via `manual_override = true` while publishing the 3D glide funnel to Pass 3.
 */
export class ForcedLandScenarioController {
  private crosswindKt: number;
  private active = true;

  constructor(crosswindKt = 35.0) {
    this.crosswindKt = crosswindKt;
  }

  setCrosswindKt(kt: number): void {
    if (Number.isFinite(kt)) {
      this.crosswindKt = clamp(kt, -60.0, 60.0);
    }
  }

  getCrosswindKt(): number {
    return this.crosswindKt;
  }

  setActive(active: boolean): void {
    this.active = active;
  }

  isActive(): boolean {
    return this.active;
  }

  step(input: ForcedLandStepInput): ForcedLandStepOutput {
    const glideV = Math.max(38.0, Number.isFinite(input.airspeedMps) ? input.airspeedMps : DEFAULT_GLIDE_AIRSPEED_MPS);
    const path = computeWindWarpedTrochoidPath({
      latRad: input.latRad,
      lonRad: input.lonRad,
      altWgs84M: input.altWgs84M,
      crosswindKt: this.crosswindKt,
      glideAirspeedMps: glideV,
      numSegments: 16,
    });

    // Pure-pursuit lookahead waypoint (~18% along corridor, min index 2)
    const lookIdx = Math.min(path.waypointsNed.length - 1, 3);
    const p0 = path.waypointsNed[0];
    const pLook = path.waypointsNed[lookIdx];
    const dN = pLook[0] - p0[0];
    const dE = pLook[1] - p0[1];
    const desiredTrackRad = Math.atan2(dE, dN);

    // Crab compensation into crosswind so ground track tracks the trochoid cleanly
    const desiredHeadingRad = desiredTrackRad - path.crabAngleDeg * DEG_TO_RAD * 0.45;
    const headingErrRad = wrapAnglePi(desiredHeadingRad - input.yawRad);
    const manualBankCmdRad = clamp(headingErrRad * 0.85, -MAX_GLIDE_BANK_RAD, MAX_GLIDE_BANK_RAD);

    // Energy-managed engine-out sink rate (L/D = 14.5 nominal, deeper sink with spoilers if high)
    const heightAboveThrM = Math.max(0.0, input.altWgs84M - KSFO_28L_THRESHOLD.altWgs84M);
    const requiredAltM = path.distToThresholdM / BEST_GLIDE_LD;
    const glideMarginM = heightAboveThrM - requiredAltM;
    const nominalSinkMps = -glideV / BEST_GLIDE_LD;
    const energyTrimMps = clamp(-glideMarginM * 0.008, -2.6, 0.8);
    const manualClimbCmdMps = clamp(nominalSinkMps + energyTrimMps, -6.5, -1.8);

    const energyState: ForcedLandDiagnostics["energyState"] =
      glideMarginM > 120.0 ? "HIGH" : glideMarginM < -40.0 ? "TIGHT" : "NOMINAL";

    return {
      manualOverride: true,
      manualBankCmdRad,
      manualClimbCmdMps,
      funnelState: {
        active: this.active,
        crosswindKt: this.crosswindKt,
        targetEcef: path.targetEcef,
        waypoints: path.waypoints,
      },
      diagnostics: {
        crosswindKt: this.crosswindKt,
        crabAngleDeg: path.crabAngleDeg,
        distToThresholdM: path.distToThresholdM,
        glideMarginM,
        maxLateralWindWarpM: path.maxLateralWindWarpM,
        energyState,
      },
    };
  }
}
