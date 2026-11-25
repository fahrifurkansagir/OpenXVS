/**
 * Closed-Loop 6-DoF Studio Scenario Engine (`OpenXVS Studio v5.0`).
 *
 * Drives BOTH:
 *   - Autopilot demo scenarios (`cbdr`, `forcedland`, `thermal_fog`, `gps_spoof`) powered by
 *     bit-exact `ActiveRangeControllerTs` + 3D wind-warped trochoid math.
 *   - Flyable 6-DoF Manual Flight (`WASD`/Arrows/Gamepad) with live Buyer A/B `Kalkan AÇIK / KAPALI`
 *     (`shieldEnabled`) override & KSFO 28L touchdown grading.
 *   - Dynamic real-world weather & atmosphere state (`timeOfDayH`, `visibilityKm`, `crosswindKt`,
 *     `cloudCover`, `rainIntensity`).
 *
 * World Coordinates (Three.js right-handed Y-up meters centered at KSFO 28L threshold):
 *   +X = East (m), +Y = Up above 28L runway datum (m), -Z = North (m).
 */

import {
  ActiveRangeControllerTs,
  ActiveRangeModeTs,
  FLAG_SC3BF_ALL_SAFE,
  G_ACCEL_MPS2,
  R_SEP_M,
  type ActiveRangeTelemetryTs,
} from "./active_range_ts.ts";
import { KSFO_28L_ORIGIN, sampleKsfoElevationM, type KsfoTerrainPack } from "./ksfo_terrain_pack.ts";
import {
  stepManualFlight6DoF,
  type FlightControlMode,
  type PilotInputState,
  type ShieldReason,
  type TouchdownReport,
} from "./manual_flight_controller.ts";

export type StudioScenarioId = "cbdr" | "forcedland" | "thermal_fog" | "gps_spoof";

export interface WeatherState {
  readonly timeOfDayH: number; // 5.0..22.0 (default 16.5 Golden Hour Daylight)
  readonly visibilityKm: number; // 0.5..25.0
  readonly crosswindKt: number; // 0..45
  readonly cloudCover: number; // 0..1
  readonly rainIntensity: number; // 0..1
}

export const DEFAULT_WEATHER_STATE: WeatherState = {
  timeOfDayH: 16.5,
  visibilityKm: 18.0,
  crosswindKt: 12.0,
  cloudCover: 0.42,
  rainIntensity: 0.0,
};

export interface Vec3Enu {
  readonly x: number; // East (m)
  readonly y: number; // Up above 28L datum (m)
  readonly z: number; // South = -North (m)
}

export interface AircraftKinematics {
  readonly pos: Vec3Enu;
  readonly vel: Vec3Enu;
  readonly headingRad: number; // NED clockwise from North (rad)
  readonly pitchRad: number; // +nose up
  readonly rollRad: number; // +right wing down
  readonly tasMps: number;
  readonly throttle: number;
  readonly engineOutRight: boolean;
}

export interface TrochoidWaypointEnu {
  readonly pos: Vec3Enu;
  readonly halfWidthM: number;
  readonly halfHeightM: number;
}

export interface StudioSimSnapshot {
  readonly scenario: StudioScenarioId;
  readonly controlMode: FlightControlMode;
  readonly shieldEnabled: boolean;
  readonly shieldOverrideActive: boolean;
  readonly shieldReason: ShieldReason;
  readonly envelopeBreach: boolean;
  readonly breachReason: string | null;
  readonly touchdown: TouchdownReport;
  readonly yokePitch: number;
  readonly yokeRoll: number;
  readonly simTimeS: number;
  readonly ownship: AircraftKinematics;
  readonly intruder: AircraftKinematics;
  readonly intruderVisible: boolean;
  readonly activeRange: ActiveRangeTelemetryTs;
  readonly trueRangeM: number;
  readonly closingMps: number;
  readonly ttcS: number;
  readonly sc3bfSafe: boolean;
  readonly forcedLandWaypoints: readonly TrochoidWaypointEnu[];
  readonly forcedLandCrosswindKt: number;
  readonly forcedLandCrabDeg: number;
  readonly weather: WeatherState;
  readonly fogDensity: number;
  readonly spoofGhostOffset: Vec3Enu;
  readonly spoofAlarm: boolean;
  readonly thermalVacuity: number;
  readonly navMatchConfidence: number;
}

const DEG_TO_RAD = Math.PI / 180.0;
const RWY_28L_HDG_RAD = KSFO_28L_ORIGIN.trueHeadingDeg * DEG_TO_RAD;
const KT_TO_MPS = 0.514444;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function wrapPi(rad: number): number {
  let a = rad;
  while (a > Math.PI) a -= 2.0 * Math.PI;
  while (a < -Math.PI) a += 2.0 * Math.PI;
  return a;
}

export function computeTrochoidWaypointsEnu(
  ownPos: Vec3Enu,
  crosswindKt: number,
  glideAirspeedMps = 52.0,
  numSegments = 20,
): {
  readonly waypoints: readonly TrochoidWaypointEnu[];
  readonly crabAngleDeg: number;
  readonly maxLateralWarpM: number;
} {
  const fwdE = Math.sin(RWY_28L_HDG_RAD);
  const fwdN = Math.cos(RWY_28L_HDG_RAD);
  const rightE = Math.cos(RWY_28L_HDG_RAD);
  const rightN = -Math.sin(RWY_28L_HDG_RAD);

  const p0E = ownPos.x;
  const p0N = -ownPos.z;
  const p0Up = Math.max(12.0, ownPos.y);
  const horizDistM = Math.max(80.0, Math.hypot(p0E, p0N));

  const windMps = crosswindKt * KT_TO_MPS;
  const crabRad = Math.asin(clamp(windMps / Math.max(32.0, glideAirspeedMps), -0.58, 0.58));
  const groundSpeedMps = Math.max(22.0, glideAirspeedMps * Math.cos(crabRad));
  const glideTimeS = horizDistM / groundSpeedMps;
  const bowPeakM = clamp(windMps * glideTimeS * 0.22, -460.0, 460.0);

  const approachBackM = Math.min(horizDistM * 0.36, 920.0);
  const calmC1E = p0E * 0.58 - fwdE * (approachBackM * 0.42);
  const calmC1N = p0N * 0.58 - fwdN * (approachBackM * 0.42);
  const calmC2E = -fwdE * approachBackM;
  const calmC2N = -fwdN * approachBackM;

  const c1E = calmC1E - rightE * bowPeakM;
  const c1N = calmC1N - rightN * bowPeakM;
  const c2E = calmC2E - rightE * (bowPeakM * 0.44);
  const c2N = calmC2N - rightN * (bowPeakM * 0.44);

  const waypoints: TrochoidWaypointEnu[] = [];
  let maxLateralWarpM = 0.0;

  for (let i = 0; i <= numSegments; i++) {
    const t = i / numSegments;
    const omt = 1.0 - t;
    const b0 = omt * omt * omt;
    const b1 = 3.0 * omt * omt * t;
    const b2 = 3.0 * omt * t * t;

    const eM = b0 * p0E + b1 * c1E + b2 * c2E;
    const nM = b0 * p0N + b1 * c1N + b2 * c2N;
    const cE = b0 * p0E + b1 * calmC1E + b2 * calmC2E;
    const cN = b0 * p0N + b1 * calmC1N + b2 * calmC2N;
    const warp = Math.hypot(eM - cE, nM - cN);
    if (warp > maxLateralWarpM) maxLateralWarpM = warp;

    const yM = 2.0 + (p0Up - 2.0) * Math.pow(omt, 1.12);
    waypoints.push({
      pos: { x: eM, y: yM, z: -nM },
      halfWidthM: 54.0 - 34.0 * t,
      halfHeightM: 22.0 - 14.0 * t,
    });
  }

  return {
    waypoints,
    crabAngleDeg: crabRad / DEG_TO_RAD,
    maxLateralWarpM,
  };
}

export class StudioScenarioSim {
  private scenario: StudioScenarioId = "cbdr";
  private controlMode: FlightControlMode = "autopilot";
  private shieldEnabled = true;
  private weather: WeatherState = { ...DEFAULT_WEATHER_STATE };
  private terrainPack: KsfoTerrainPack | null = null;

  private simTimeS = 0.0;
  private readonly activeRangeCtrl: ActiveRangeControllerTs;
  private lastArTel: ActiveRangeTelemetryTs | null = null;
  private arAccumS = 0.0;
  private trackAge = 4;
  private prevSizePx = 0.0;

  // Ownship local ENU kinematics
  private ownPos: [number, number, number] = [0, 0, 0];
  private ownHdgRad = RWY_28L_HDG_RAD;
  private ownPitchRad = -0.04;
  private ownRollRad = 0.0;
  private ownTasMps = 54.0;
  private ownClimbMps = -2.4;
  private ownThrottle = 0.64;
  private engineOutRight = false;

  // Intruder helicopter local ENU kinematics
  private intrPos: [number, number, number] = [0, 0, 0];
  private intrVel: [number, number, number] = [0, 0, 0];
  private intrHdgRad = wrapPi(RWY_28L_HDG_RAD + Math.PI);

  private lastWaypoints: readonly TrochoidWaypointEnu[] = [];
  private lastCrabDeg = 0.0;

  // Manual flight feedback
  private shieldOverrideActive = false;
  private shieldReason: ShieldReason = "NONE";
  private envelopeBreach = false;
  private breachReason: string | null = null;
  private touchdown: TouchdownReport = {
    landed: false,
    sinkRateFpm: 0,
    centerlineOffsetM: 0,
    onRunway: false,
    grade: "NONE",
  };
  private yokePitch = 0;
  private yokeRoll = 0;

  constructor(initialScenario: StudioScenarioId = "cbdr") {
    this.activeRangeCtrl = new ActiveRangeControllerTs({
      sigmaBearingRad: 0.00055,
      qHatConformal: 0.925,
    });
    this.setScenario(initialScenario);
  }

  setTerrainPack(pack: KsfoTerrainPack | null): void {
    this.terrainPack = pack;
  }

  getScenario(): StudioScenarioId {
    return this.scenario;
  }

  getControlMode(): FlightControlMode {
    return this.controlMode;
  }

  setControlMode(mode: FlightControlMode): void {
    this.controlMode = mode;
    if (mode === "autopilot") {
      this.shieldOverrideActive = false;
      this.shieldReason = "NONE";
      this.envelopeBreach = false;
      this.breachReason = null;
    }
  }

  isShieldEnabled(): boolean {
    return this.shieldEnabled;
  }

  setShieldEnabled(enabled: boolean): void {
    this.shieldEnabled = enabled;
  }

  toggleEngineOut(): boolean {
    this.engineOutRight = !this.engineOutRight;
    return this.engineOutRight;
  }

  getWeather(): WeatherState {
    return this.weather;
  }

  setWeather(partial: Partial<WeatherState>): WeatherState {
    this.weather = {
      timeOfDayH: clamp(partial.timeOfDayH ?? this.weather.timeOfDayH, 5.0, 22.0),
      visibilityKm: clamp(partial.visibilityKm ?? this.weather.visibilityKm, 0.5, 25.0),
      crosswindKt: clamp(partial.crosswindKt ?? this.weather.crosswindKt, 0.0, 45.0),
      cloudCover: clamp(partial.cloudCover ?? this.weather.cloudCover, 0.0, 1.0),
      rainIntensity: clamp(partial.rainIntensity ?? this.weather.rainIntensity, 0.0, 1.0),
    };
    return this.weather;
  }

  setScenario(next: StudioScenarioId): void {
    this.scenario = next;
    // Apply sensible scenario weather defaults while preserving user's chosen timeOfDayH
    if (next === "forcedland") {
      this.weather = { ...this.weather, crosswindKt: 35.0, visibilityKm: 16.0, rainIntensity: 0.0 };
    } else if (next === "thermal_fog") {
      this.weather = { ...this.weather, visibilityKm: 1.4, cloudCover: 0.9, rainIntensity: 0.35 };
    } else {
      this.weather = { ...this.weather, visibilityKm: 18.0, crosswindKt: 12.0, rainIntensity: 0.0 };
    }
    this.resetScenarioState();
  }

  resetApproach(): void {
    this.resetScenarioState();
  }

  private resetScenarioState(): void {
    this.simTimeS = 0.0;
    this.arAccumS = 0.0;
    this.trackAge = 4;
    this.prevSizePx = 0.0;
    this.activeRangeCtrl.reset();
    this.lastArTel = null;
    this.shieldOverrideActive = false;
    this.shieldReason = "NONE";
    this.envelopeBreach = false;
    this.breachReason = null;
    this.touchdown = {
      landed: false,
      sinkRateFpm: 0,
      centerlineOffsetM: 0,
      onRunway: false,
      grade: "NONE",
    };

    const fwdE = Math.sin(RWY_28L_HDG_RAD);
    const fwdN = Math.cos(RWY_28L_HDG_RAD);
    const rightE = Math.cos(RWY_28L_HDG_RAD);
    const rightN = -Math.sin(RWY_28L_HDG_RAD);

    if (this.scenario === "cbdr") {
      const backM = 2550.0;
      this.ownPos = [-fwdE * backM, 235.0, fwdN * backM];
      this.ownHdgRad = RWY_28L_HDG_RAD;
      this.ownPitchRad = -0.025;
      this.ownRollRad = 0.0;
      this.ownTasMps = 52.0;
      this.ownClimbMps = -1.6;
      this.ownThrottle = 0.64;
      this.engineOutRight = false;

      const intrAheadM = 1280.0;
      this.intrPos = [
        this.ownPos[0] + fwdE * intrAheadM,
        this.ownPos[1] + 8.0,
        this.ownPos[2] - fwdN * intrAheadM,
      ];
      const intrSpeedMps = 34.0;
      this.intrVel = [-fwdE * intrSpeedMps, 0.0, fwdN * intrSpeedMps];
      this.intrHdgRad = wrapPi(RWY_28L_HDG_RAD + Math.PI);
    } else if (this.scenario === "forcedland") {
      const backM = 2900.0;
      const offsetRightM = -520.0;
      this.ownPos = [
        -fwdE * backM + rightE * offsetRightM,
        365.0,
        fwdN * backM - rightN * offsetRightM,
      ];
      this.ownHdgRad = wrapPi(RWY_28L_HDG_RAD - 0.14);
      this.ownPitchRad = -0.065;
      this.ownRollRad = 0.12;
      this.ownTasMps = 51.4;
      this.ownClimbMps = -3.6;
      this.ownThrottle = 0.25;
      this.engineOutRight = true;
      this.intrPos = [this.ownPos[0] + fwdE * 2800.0, 290.0, this.ownPos[2] - fwdN * 2800.0];
      this.intrVel = [-fwdE * 28.0, 0.0, fwdN * 28.0];
      this.intrHdgRad = wrapPi(RWY_28L_HDG_RAD + Math.PI);
    } else if (this.scenario === "thermal_fog") {
      const backM = 1850.0;
      this.ownPos = [-fwdE * backM, 145.0, fwdN * backM];
      this.ownHdgRad = RWY_28L_HDG_RAD;
      this.ownPitchRad = -0.05;
      this.ownRollRad = 0.0;
      this.ownTasMps = 48.0;
      this.ownClimbMps = -2.8;
      this.ownThrottle = 0.56;
      this.engineOutRight = false;
      this.intrPos = [
        this.ownPos[0] + fwdE * 980.0 + rightE * 340.0,
        175.0,
        this.ownPos[2] - fwdN * 980.0 - rightN * 340.0,
      ];
      this.intrVel = [-fwdE * 26.0, 0.0, fwdN * 26.0];
      this.intrHdgRad = wrapPi(RWY_28L_HDG_RAD + Math.PI);
    } else {
      const backM = 2300.0;
      this.ownPos = [-fwdE * backM, 260.0, fwdN * backM];
      this.ownHdgRad = RWY_28L_HDG_RAD;
      this.ownPitchRad = -0.02;
      this.ownRollRad = 0.0;
      this.ownTasMps = 55.0;
      this.ownClimbMps = -1.2;
      this.ownThrottle = 0.66;
      this.engineOutRight = false;
      this.intrPos = [
        this.ownPos[0] + fwdE * 1450.0 + rightE * 260.0,
        285.0,
        this.ownPos[2] - fwdN * 1450.0 - rightN * 260.0,
      ];
      this.intrVel = [-fwdE * 30.0, 0.0, fwdN * 30.0];
      this.intrHdgRad = wrapPi(RWY_28L_HDG_RAD + Math.PI);
    }

    this.stepActiveRange(0.1);
    const troch = computeTrochoidWaypointsEnu(
      { x: this.ownPos[0], y: this.ownPos[1], z: this.ownPos[2] },
      this.weather.crosswindKt,
    );
    this.lastWaypoints = troch.waypoints;
    this.lastCrabDeg = troch.crabAngleDeg;
  }

  private stepActiveRange(dtS: number): void {
    const relN = -this.intrPos[2] - -this.ownPos[2];
    const relE = this.intrPos[0] - this.ownPos[0];
    const relD = -this.intrPos[1] - -this.ownPos[1];
    const trueRangeM = Math.max(20.0, Math.hypot(relN, relE, relD));
    const horiz = Math.max(1.0, Math.hypot(relN, relE));
    const alphaRad = Math.atan2(relE, relN);
    const betaRad = Math.atan2(-relD, horiz);

    const focalPx = 2690.0;
    const intrWingspanM = 10.6;
    const sizePx = (focalPx * intrWingspanM) / trueRangeM;
    const sizeRatePxS =
      this.prevSizePx > 0.0 ? (sizePx - this.prevSizePx) / dtS : (sizePx * 82.0) / trueRangeM;
    this.prevSizePx = sizePx;
    this.trackAge++;

    const vN = this.ownTasMps * Math.cos(this.ownHdgRad);
    const vE = this.ownTasMps * Math.sin(this.ownHdgRad);
    const vD = -this.ownClimbMps;
    const aLat = G_ACCEL_MPS2 * Math.tan(clamp(this.ownRollRad, -0.45, 0.45));
    const aN = -Math.sin(this.ownHdgRad) * aLat;
    const aE = Math.cos(this.ownHdgRad) * aLat;

    this.lastArTel = this.activeRangeCtrl.step(
      {
        vNed: [vN, vE, vD],
        aNed: [aN, aE, 0.0],
        headingRad: this.ownHdgRad,
        tasMps: this.ownTasMps,
        vStallMps: 14.5,
      },
      {
        alphaRad,
        betaRad,
        sizePx,
        sizeRatePxS,
        contrastRatio: this.weather.visibilityKm < 4.0 ? 0.22 : 0.72,
        skyBetaExt: this.weather.visibilityKm < 4.0 ? 1.8e-3 : 2.0e-4,
        daytimeClear: this.weather.visibilityKm >= 4.0,
        trackAge: this.trackAge,
        dtS,
      },
    );
  }

  step(dtS: number, pilotInput?: PilotInputState): StudioSimSnapshot {
    const dt = clamp(dtS, 1e-4, 0.05);
    this.simTimeS += dt;
    this.arAccumS += dt;

    if (this.arAccumS >= 0.1) {
      this.stepActiveRange(0.1);
      this.arAccumS -= 0.1;
    }

    // Recompute ForcedLand corridor whenever in forcedland or engine-out
    if (this.scenario === "forcedland" || this.engineOutRight || this.scenario === "thermal_fog") {
      const troch = computeTrochoidWaypointsEnu(
        { x: this.ownPos[0], y: this.ownPos[1], z: this.ownPos[2] },
        this.weather.crosswindKt,
        this.ownTasMps,
        20,
      );
      this.lastWaypoints = troch.waypoints;
      this.lastCrabDeg = troch.crabAngleDeg;
    }

    if (this.controlMode === "manual") {
      const terrainElevM = sampleKsfoElevationM(this.terrainPack, this.ownPos[0], -this.ownPos[2]);
      const intrVisible = this.scenario === "cbdr" || this.scenario === "gps_spoof";
      const lookWp =
        this.lastWaypoints.length > 3
          ? this.lastWaypoints[3].pos
          : { x: 0, y: 2.0, z: 0 };

      const res = stepManualFlight6DoF({
        prev: {
          posX: this.ownPos[0],
          posY: this.ownPos[1],
          posZ: this.ownPos[2],
          headingRad: this.ownHdgRad,
          pitchRad: this.ownPitchRad,
          rollRad: this.ownRollRad,
          tasMps: this.ownTasMps,
          climbMps: this.ownClimbMps,
          throttle: this.ownThrottle,
          engineOutRight: this.engineOutRight,
        },
        input: pilotInput ?? { pitch: 0, roll: 0, yaw: 0, throttleDelta: 0 },
        dtS: dt,
        simTimeS: this.simTimeS,
        crosswindKt: this.weather.crosswindKt,
        terrainElevM,
        intruderPos: intrVisible
          ? { x: this.intrPos[0], y: this.intrPos[1], z: this.intrPos[2] }
          : null,
        intruderVel: intrVisible
          ? { x: this.intrVel[0], y: this.intrVel[1], z: this.intrVel[2] }
          : null,
        shieldEnabled: this.shieldEnabled,
        forcedLandTargetPos: lookWp,
      });

      this.ownPos = [res.state.posX, res.state.posY, res.state.posZ];
      this.ownHdgRad = res.state.headingRad;
      this.ownPitchRad = res.state.pitchRad;
      this.ownRollRad = res.state.rollRad;
      this.ownTasMps = res.state.tasMps;
      this.ownClimbMps = res.state.climbMps;
      this.ownThrottle = res.state.throttle;
      this.shieldOverrideActive = res.shieldOverrideActive;
      this.shieldReason = res.shieldReason;
      this.envelopeBreach = res.envelopeBreach;
      this.breachReason = res.breachReason;
      this.touchdown = res.touchdown;
      this.yokePitch = res.yokePitch;
      this.yokeRoll = res.yokeRoll;

      this.intrPos = [
        this.intrPos[0] + this.intrVel[0] * dt,
        this.intrPos[1] + this.intrVel[1] * dt,
        this.intrPos[2] + this.intrVel[2] * dt,
      ];
      return this.getSnapshot();
    }

    // Autopilot demo mode
    let cmdBankRad = 0.0;
    let cmdClimbMps = this.ownClimbMps;

    if (this.scenario === "cbdr") {
      const tel = this.lastArTel!;
      cmdBankRad =
        tel.mode === ActiveRangeModeTs.ActiveSwerve || tel.mode === ActiveRangeModeTs.ImmediateEvade
          ? tel.cmdBankRad
          : wrapPi(RWY_28L_HDG_RAD - this.ownHdgRad) * 0.45;
      cmdClimbMps = -1.4;
    } else if (this.scenario === "forcedland") {
      const look = this.lastWaypoints[Math.min(3, this.lastWaypoints.length - 1)].pos;
      const dE = look.x - this.ownPos[0];
      const dN = -look.z - -this.ownPos[2];
      const desiredTrack = Math.atan2(dE, dN);
      const desiredHdg = desiredTrack - this.lastCrabDeg * DEG_TO_RAD * 0.42;
      cmdBankRad = clamp(wrapPi(desiredHdg - this.ownHdgRad) * 0.85, -0.4, 0.4);
      cmdClimbMps = -3.6;
    } else {
      cmdBankRad = clamp(wrapPi(RWY_28L_HDG_RAD - this.ownHdgRad) * 0.55, -0.22, 0.22);
      cmdClimbMps = this.scenario === "thermal_fog" ? -2.6 : -1.2;
    }

    this.ownRollRad += (clamp(cmdBankRad, -0.45, 0.45) - this.ownRollRad) * Math.min(1.0, dt * 3.8);
    this.ownClimbMps += (cmdClimbMps - this.ownClimbMps) * Math.min(1.0, dt * 2.5);
    const aLat = G_ACCEL_MPS2 * Math.tan(this.ownRollRad);
    this.ownHdgRad = wrapPi(this.ownHdgRad + (aLat / Math.max(25.0, this.ownTasMps)) * dt);
    this.ownPitchRad = clamp(
      Math.asin(clamp(this.ownClimbMps / this.ownTasMps, -0.25, 0.25)),
      -0.18,
      0.15,
    );
    this.yokeRoll = clamp(this.ownRollRad / 0.45, -1, 1);
    this.yokePitch = clamp(this.ownPitchRad / 0.2, -1, 1);

    const vE = this.ownTasMps * Math.sin(this.ownHdgRad);
    const vN = this.ownTasMps * Math.cos(this.ownHdgRad);
    this.ownPos = [
      this.ownPos[0] + vE * dt,
      Math.max(18.0, this.ownPos[1] + this.ownClimbMps * dt),
      this.ownPos[2] - vN * dt,
    ];

    this.intrPos = [
      this.intrPos[0] + this.intrVel[0] * dt,
      this.intrPos[1] + this.intrVel[1] * dt,
      this.intrPos[2] + this.intrVel[2] * dt,
    ];

    const distToThrM = Math.hypot(this.ownPos[0], this.ownPos[2]);
    const relIntrE = this.intrPos[0] - this.ownPos[0];
    const relIntrN = -this.intrPos[2] - -this.ownPos[2];
    const alongTrackIntr = relIntrE * Math.sin(this.ownHdgRad) + relIntrN * Math.cos(this.ownHdgRad);

    if (
      distToThrM < 320.0 ||
      this.simTimeS > 24.0 ||
      (this.scenario === "cbdr" && alongTrackIntr < 140.0)
    ) {
      this.resetScenarioState();
    }

    return this.getSnapshot();
  }

  getSnapshot(): StudioSimSnapshot {
    const ar = this.lastArTel!;
    const relX = this.intrPos[0] - this.ownPos[0];
    const relY = this.intrPos[1] - this.ownPos[1];
    const relZ = this.intrPos[2] - this.ownPos[2];
    const trueRangeM = Math.max(1.0, Math.hypot(relX, relY, relZ));
    const ownVel: Vec3Enu = {
      x: this.ownTasMps * Math.sin(this.ownHdgRad),
      y: this.ownClimbMps,
      z: -this.ownTasMps * Math.cos(this.ownHdgRad),
    };
    const vRelX = this.intrVel[0] - ownVel.x;
    const vRelY = this.intrVel[1] - ownVel.y;
    const vRelZ = this.intrVel[2] - ownVel.z;
    const closingMps = -(relX * vRelX + relY * vRelY + relZ * vRelZ) / trueRangeM;
    const ttcS = closingMps > 0.5 ? Math.min(999.0, trueRangeM / closingMps) : 999.0;

    const spoofRamp = this.scenario === "gps_spoof" ? Math.min(1.0, 0.35 + this.simTimeS * 0.08) : 0.0;
    const spoofGhostOffset: Vec3Enu = {
      x: 220.0 * spoofRamp,
      y: 26.0 * spoofRamp,
      z: -165.0 * spoofRamp,
    };

    // Map meteorological visibilityKm (0.5..25 km) + rainIntensity to Beer-Lambert fog extinction
    const visM = Math.max(500.0, this.weather.visibilityKm * 1000.0);
    const fogDensity = clamp(0.95 / visM + this.weather.rainIntensity * 0.00018, 0.000035, 0.0014);

    return {
      scenario: this.scenario,
      controlMode: this.controlMode,
      shieldEnabled: this.shieldEnabled,
      shieldOverrideActive: this.shieldOverrideActive,
      shieldReason: this.shieldReason,
      envelopeBreach: this.envelopeBreach,
      breachReason: this.breachReason,
      touchdown: this.touchdown,
      yokePitch: this.yokePitch,
      yokeRoll: this.yokeRoll,
      simTimeS: this.simTimeS,
      ownship: {
        pos: { x: this.ownPos[0], y: this.ownPos[1], z: this.ownPos[2] },
        vel: ownVel,
        headingRad: this.ownHdgRad,
        pitchRad: this.ownPitchRad,
        rollRad: this.ownRollRad,
        tasMps: this.ownTasMps,
        throttle: this.ownThrottle,
        engineOutRight: this.engineOutRight,
      },
      intruder: {
        pos: { x: this.intrPos[0], y: this.intrPos[1], z: this.intrPos[2] },
        vel: { x: this.intrVel[0], y: this.intrVel[1], z: this.intrVel[2] },
        headingRad: this.intrHdgRad,
        pitchRad: 0.03,
        rollRad: 0.0,
        tasMps: Math.hypot(this.intrVel[0], this.intrVel[2]),
        throttle: 0.72,
        engineOutRight: false,
      },
      intruderVisible: this.scenario === "cbdr" || this.scenario === "gps_spoof",
      activeRange: ar,
      trueRangeM,
      closingMps,
      ttcS,
      sc3bfSafe: (ar.flags & FLAG_SC3BF_ALL_SAFE) !== 0 && !this.envelopeBreach,
      forcedLandWaypoints: this.lastWaypoints,
      forcedLandCrosswindKt: this.weather.crosswindKt,
      forcedLandCrabDeg: this.lastCrabDeg,
      weather: this.weather,
      fogDensity,
      spoofGhostOffset,
      spoofAlarm: this.scenario === "gps_spoof",
      thermalVacuity: this.weather.visibilityKm < 4.0 ? 0.164 : 0.12,
      navMatchConfidence: this.scenario === "gps_spoof" ? 0.945 : 0.962,
    };
  }
}

export { ActiveRangeModeTs, R_SEP_M };
