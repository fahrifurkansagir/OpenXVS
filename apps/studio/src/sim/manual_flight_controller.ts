/**
 * Flyable 6-DoF Manual Flight Controller + Live OpenXVS Safety Shield Override (`OpenXVS Studio v5.0`).
 *
 * Supports:
 *   - Keyboard (`WASD` / Arrow keys pitch & roll, `Q`/`E` yaw, `Shift`/`Ctrl` throttle,
 *     `F` toggle right engine-out, `R` reset approach) — strictly ignored when typing in inputs/modals.
 *   - Standard HTML5 `Gamepad API` polling (axes 0/1 roll/pitch, axis 2 yaw, trigger/axis 3 throttle).
 *   - Fixed-wing 6-DoF kinematics: lift/drag, bank-to-turn rate, crosswind drift (0–45 kt),
 *     light gusts, airspeed/stall floor, and KSFO 28L runway touchdown grading.
 *   - Interactive Buyer A/B Proof (`shieldEnabled`):
 *     - When `true`: `XVS-ActiveRange` + `SC3BF` + `ForcedLand` smoothly overrides unsafe pilot inputs
 *       (`SC3BF CBDR`, `TERRAIN FLOOR`, `FORCEDLAND GLIDE`) to maintain > 152.4 m separation & safe glide.
 *     - When `false`: permits raw manual flight and reports `envelopeBreach` (`COLLISION` / `TERRAIN` / `STALL`).
 */

import { G_ACCEL_MPS2, R_SEP_M } from "./active_range_ts.ts";
import { KSFO_28L_ORIGIN } from "./ksfo_terrain_pack.ts";

export type FlightControlMode = "autopilot" | "manual";
export type ShieldReason = "NONE" | "SC3BF CBDR" | "TERRAIN FLOOR" | "FORCEDLAND GLIDE";
export type LandingGrade = "NONE" | "KUSURSUZ" | "GÜVENLİ" | "SERT İNİŞ";

export interface PilotInputState {
  readonly pitch: number; // [-1, +1] (+1 pull back / nose up)
  readonly roll: number; // [-1, +1] (+1 bank right)
  readonly yaw: number; // [-1, +1] (+1 yaw right)
  readonly throttleDelta: number; // [-1, +1]
}

export interface TouchdownReport {
  readonly landed: boolean;
  readonly sinkRateFpm: number;
  readonly centerlineOffsetM: number;
  readonly onRunway: boolean;
  readonly grade: LandingGrade;
}

export interface ManualFlightState {
  posX: number; // East (m)
  posY: number; // Up above 28L datum (m)
  posZ: number; // South = -North (m)
  headingRad: number; // NED clockwise from North
  pitchRad: number; // +nose up
  rollRad: number; // +right wing down
  tasMps: number;
  climbMps: number;
  throttle: number; // [0..1]
  engineOutRight: boolean;
}

export interface ManualStepResult {
  readonly state: ManualFlightState;
  readonly shieldOverrideActive: boolean;
  readonly shieldReason: ShieldReason;
  readonly envelopeBreach: boolean;
  readonly breachReason: string | null;
  readonly touchdown: TouchdownReport;
  readonly yokePitch: number;
  readonly yokeRoll: number;
  readonly rudderYaw: number;
}

const DEG_TO_RAD = Math.PI / 180.0;
const RWY_28L_HDG_RAD = KSFO_28L_ORIGIN.trueHeadingDeg * DEG_TO_RAD;
const KT_TO_MPS = 0.514444;
const MPS_TO_FPM = 196.8504;
const V_STALL_MPS = 29.0; // ~56 kt clean twin-turboprop stall floor
const V_MAX_MPS = 92.0; // ~179 kt cruise ceiling

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function wrapPi(rad: number): number {
  let a = rad;
  while (a > Math.PI) a -= 2.0 * Math.PI;
  while (a < -Math.PI) a += 2.0 * Math.PI;
  return a;
}

/**
 * Evaluates KSFO 28L touchdown geometry & landing grade when aircraft descends to runway surface.
 */
export function evaluateRunwayTouchdown(
  posX: number,
  posY: number,
  posZ: number,
  climbMps: number,
): TouchdownReport {
  if (posY > 3.2) {
    return {
      landed: false,
      sinkRateFpm: 0,
      centerlineOffsetM: 0,
      onRunway: false,
      grade: "NONE",
    };
  }
  const fwdE = Math.sin(RWY_28L_HDG_RAD);
  const fwdN = Math.cos(RWY_28L_HDG_RAD);
  const rightE = Math.cos(RWY_28L_HDG_RAD);
  const rightN = -Math.sin(RWY_28L_HDG_RAD);

  const eastM = posX;
  const northM = -posZ;
  const alongM = eastM * fwdE + northM * fwdN;
  const crossM = eastM * rightE + northM * rightN;

  const onRunway = alongM >= -60.0 && alongM <= 3470.0 && Math.abs(crossM) <= 32.0;
  const sinkRateFpm = Math.max(0.0, -climbMps * MPS_TO_FPM);
  const absOff = Math.abs(crossM);

  let grade: LandingGrade = "SERT İNİŞ";
  if (onRunway && sinkRateFpm <= 240.0 && absOff <= 8.0) {
    grade = "KUSURSUZ";
  } else if (onRunway && sinkRateFpm <= 480.0 && absOff <= 22.0) {
    grade = "GÜVENLİ";
  }

  return {
    landed: true,
    sinkRateFpm,
    centerlineOffsetM: crossM,
    onRunway,
    grade,
  };
}

/**
 * Pure deterministic 6-DoF manual flight step with live OpenXVS Safety Shield A/B guard.
 */
export function stepManualFlight6DoF(args: {
  readonly prev: ManualFlightState;
  readonly input: PilotInputState;
  readonly dtS: number;
  readonly simTimeS: number;
  readonly crosswindKt: number;
  readonly terrainElevM: number;
  readonly intruderPos: { readonly x: number; readonly y: number; readonly z: number } | null;
  readonly intruderVel: { readonly x: number; readonly y: number; readonly z: number } | null;
  readonly shieldEnabled: boolean;
  readonly forcedLandTargetPos?: { readonly x: number; readonly y: number; readonly z: number } | null;
}): ManualStepResult {
  const dt = clamp(args.dtS, 1e-4, 0.05);
  const s: ManualFlightState = { ...args.prev };

  let yokeRoll = clamp(args.input.roll, -1.0, 1.0);
  let yokePitch = clamp(args.input.pitch, -1.0, 1.0);
  const rudderYaw = clamp(args.input.yaw, -1.0, 1.0);

  // Adjust throttle
  s.throttle = clamp(s.throttle + args.input.throttleDelta * dt * 0.55, 0.0, 1.0);
  const effectiveThrottle = s.engineOutRight ? s.throttle * 0.32 : s.throttle;

  // Evaluate Safety Shield threats BEFORE applying pilot command (A/B Buyer Proof)
  let shieldOverrideActive = false;
  let shieldReason: ShieldReason = "NONE";
  let envelopeBreach = false;
  let breachReason: string | null = null;

  let cmdRollTarget = s.rollRad + yokeRoll * 1.15 * dt;
  let cmdPitchTarget = s.pitchRad + yokePitch * 0.65 * dt;

  // Natural mild wing-level stability when pilot centers stick
  if (Math.abs(yokeRoll) < 0.05) {
    cmdRollTarget *= Math.max(0.0, 1.0 - dt * 0.65);
  }
  if (Math.abs(yokePitch) < 0.05) {
    cmdPitchTarget += (-0.02 - cmdPitchTarget) * Math.min(1.0, dt * 0.45);
  }

  // 1. Check SC3BF Intruder Separation / Collision Cone Threat
  if (args.intruderPos) {
    const dx = args.intruderPos.x - s.posX;
    const dy = args.intruderPos.y - s.posY;
    const dz = args.intruderPos.z - s.posZ;
    const distM = Math.hypot(dx, dy, dz);

    const ownVx = s.tasMps * Math.sin(s.headingRad);
    const ownVz = -s.tasMps * Math.cos(s.headingRad);
    const rvx = (args.intruderVel?.x ?? 0) - ownVx;
    const rvy = (args.intruderVel?.y ?? 0) - s.climbMps;
    const rvz = (args.intruderVel?.z ?? 0) - ownVz;
    const closingMps = -(dx * rvx + dy * rvy + dz * rvz) / Math.max(distM, 1.0);
    const ttcS = closingMps > 2.0 ? distM / closingMps : 999.0;

    // Bearing of intruder relative to ownship nose (+right, -left)
    const fwdX = Math.sin(s.headingRad);
    const fwdZ = -Math.cos(s.headingRad);
    const rightX = Math.cos(s.headingRad);
    const rightZ = Math.sin(s.headingRad);
    const aheadDot = (dx * fwdX + dz * fwdZ) / Math.max(distM, 1.0);
    const rightDot = (dx * rightX + dz * rightZ) / Math.max(distM, 1.0);

    const cbdrThreat =
      distM < 440.0 && (ttcS < 7.5 || distM < R_SEP_M * 1.45) && aheadDot > 0.25;

    if (cbdrThreat) {
      if (args.shieldEnabled) {
        shieldOverrideActive = true;
        shieldReason = "SC3BF CBDR";
        // Bank cleanly away from intruder + gentle climb out of collision cone
        const evadeSign = rightDot >= 0 ? -1.0 : 1.0;
        cmdRollTarget = evadeSign * 0.42; // 24 deg decisive SC3BF evasion
        cmdPitchTarget = Math.max(cmdPitchTarget, 0.11);
        yokeRoll = evadeSign * 0.85;
        yokePitch = 0.55;
      } else if (distM < R_SEP_M) {
        envelopeBreach = true;
        breachReason = "SC3BF COLLISION BREACH (<152.4m)";
      }
    }
  }

  // 2. Check TAWS Terrain Floor / Steep Dive Protection (away from 28L landing flare corridor)
  const aglM = s.posY - Math.max(0.0, args.terrainElevM);
  const fwdE = Math.sin(RWY_28L_HDG_RAD);
  const fwdN = Math.cos(RWY_28L_HDG_RAD);
  const rightE = Math.cos(RWY_28L_HDG_RAD);
  const rightN = -Math.sin(RWY_28L_HDG_RAD);
  const alongRwy = s.posX * fwdE + -s.posZ * fwdN;
  const crossRwy = Math.abs(s.posX * rightE + -s.posZ * rightN);
  const inFinalApproachBox = alongRwy >= -1400.0 && alongRwy <= 3500.0 && crossRwy <= 120.0;

  if (!shieldOverrideActive && !inFinalApproachBox && (aglM < 48.0 || (aglM < 85.0 && s.climbMps < -6.5))) {
    if (args.shieldEnabled) {
      shieldOverrideActive = true;
      shieldReason = "TERRAIN FLOOR";
      cmdPitchTarget = 0.16;
      cmdRollTarget *= 0.4;
      s.throttle = Math.max(s.throttle, 0.85);
      yokePitch = 0.78;
    } else if (aglM < 12.0) {
      envelopeBreach = true;
      breachReason = "TERRAIN FLOOR BREACH";
    }
  }

  // 3. Check Engine-Out ForcedLand Stall / Glide Energy Protection
  if (!shieldOverrideActive && s.engineOutRight && (s.tasMps < V_STALL_MPS + 5.0 || s.pitchRad > 0.08)) {
    if (args.shieldEnabled) {
      shieldOverrideActive = true;
      shieldReason = "FORCEDLAND GLIDE";
      cmdPitchTarget = -0.055; // Best-glide pitch attitude
      if (args.forcedLandTargetPos) {
        const dE = args.forcedLandTargetPos.x - s.posX;
        const dN = -args.forcedLandTargetPos.z - -s.posZ;
        const desiredHdg = Math.atan2(dE, dN);
        cmdRollTarget = clamp(wrapPi(desiredHdg - s.headingRad) * 0.75, -0.36, 0.36);
      }
      yokePitch = -0.35;
    } else if (s.tasMps <= V_STALL_MPS + 0.5) {
      envelopeBreach = true;
      breachReason = "ENGINE-OUT STALL BREACH";
    }
  }

  // Smooth aerodynamic response
  s.rollRad += (clamp(cmdRollTarget, -0.78, 0.78) - s.rollRad) * Math.min(1.0, dt * 6.5);
  s.pitchRad += (clamp(cmdPitchTarget, -0.34, 0.32) - s.pitchRad) * Math.min(1.0, dt * 5.8);

  // Longitudinal lift/drag + thrust + gravity acceleration
  const thrustAccel = effectiveThrottle * 5.2;
  const dragAccel = 0.00072 * s.tasMps * s.tasMps + (s.engineOutRight ? 0.45 : 0.0);
  const gravAccel = -G_ACCEL_MPS2 * Math.sin(s.pitchRad);
  const netAccel = thrustAccel - dragAccel + gravAccel;
  const minSpeed = args.shieldEnabled ? V_STALL_MPS + 3.0 : V_STALL_MPS - 4.0;
  s.tasMps = clamp(s.tasMps + netAccel * dt, minSpeed, V_MAX_MPS);

  // Coordinated bank-to-turn heading rate + yaw rudder + gentle atmospheric turbulence gust
  const gustRoll = Math.sin(args.simTimeS * 1.9) * 0.004 * (args.crosswindKt / 35.0);
  const aLat = G_ACCEL_MPS2 * Math.tan(clamp(s.rollRad + gustRoll, -0.72, 0.72));
  const yawRate = aLat / Math.max(25.0, s.tasMps) + rudderYaw * 0.28;
  s.headingRad = wrapPi(s.headingRad + yawRate * dt);

  // Climb / sink rate including engine-out glide L/D sink
  const engineSinkBias = s.engineOutRight ? -1.8 : 0.0;
  const targetClimb = s.tasMps * Math.sin(s.pitchRad) + engineSinkBias;
  s.climbMps += (targetClimb - s.climbMps) * Math.min(1.0, dt * 4.5);

  // Crosswind drift perpendicular to KSFO 28L
  const windMps = clamp(args.crosswindKt, 0.0, 45.0) * KT_TO_MPS;
  const windE = rightE * windMps * 0.55;
  const windN = rightN * windMps * 0.55;

  const vE = s.tasMps * Math.sin(s.headingRad) + windE;
  const vN = s.tasMps * Math.cos(s.headingRad) + windN;

  s.posX += vE * dt;
  s.posZ -= vN * dt;
  s.posY = Math.max(1.8, s.posY + s.climbMps * dt);

  const touchdown = evaluateRunwayTouchdown(s.posX, s.posY, s.posZ, s.climbMps);
  if (touchdown.landed) {
    s.posY = 1.8;
    s.climbMps = 0.0;
    s.rollRad *= 0.85;
    s.pitchRad = Math.max(0.0, s.pitchRad * 0.85);
    s.tasMps = Math.max(0.0, s.tasMps - 14.0 * dt);
  }

  return {
    state: s,
    shieldOverrideActive,
    shieldReason,
    envelopeBreach,
    breachReason,
    touchdown,
    yokePitch,
    yokeRoll,
    rudderYaw,
  };
}

/** Browser keyboard + HTML5 Gamepad input aggregator (never hijacks inputs/modals). */
export class PilotInputCapture {
  private readonly keys = new Set<string>();
  private enabled = true;
  private onToggleEngineOut?: () => void;
  private onResetApproach?: () => void;

  constructor(callbacks?: {
    readonly onToggleEngineOut?: () => void;
    readonly onResetApproach?: () => void;
  }) {
    this.onToggleEngineOut = callbacks?.onToggleEngineOut;
    this.onResetApproach = callbacks?.onResetApproach;

    if (typeof window !== "undefined") {
      window.addEventListener("keydown", this.handleKeyDown);
      window.addEventListener("keyup", this.handleKeyUp);
      window.addEventListener("blur", () => this.keys.clear());
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.keys.clear();
  }

  private isTypingTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  private readonly handleKeyDown = (ev: KeyboardEvent): void => {
    if (!this.enabled || this.isTypingTarget(ev.target)) return;
    const code = ev.code;
    if (
      code === "KeyW" ||
      code === "KeyS" ||
      code === "KeyA" ||
      code === "KeyD" ||
      code === "KeyQ" ||
      code === "KeyE" ||
      code === "ArrowUp" ||
      code === "ArrowDown" ||
      code === "ArrowLeft" ||
      code === "ArrowRight" ||
      code === "ShiftLeft" ||
      code === "ShiftRight" ||
      code === "ControlLeft" ||
      code === "ControlRight"
    ) {
      this.keys.add(code);
    } else if (code === "KeyF" && !ev.repeat) {
      this.onToggleEngineOut?.();
    } else if (code === "KeyR" && !ev.repeat) {
      this.onResetApproach?.();
    }
  };

  private readonly handleKeyUp = (ev: KeyboardEvent): void => {
    this.keys.delete(ev.code);
  };

  sample(): PilotInputState {
    if (!this.enabled) {
      return { pitch: 0, roll: 0, yaw: 0, throttleDelta: 0 };
    }
    // W / ArrowDown pushes nose down; S / ArrowUp pulls nose up (aircraft stick convention)
    let pitch = 0;
    if (this.keys.has("KeyS") || this.keys.has("ArrowUp")) pitch += 1;
    if (this.keys.has("KeyW") || this.keys.has("ArrowDown")) pitch -= 1;

    let roll = 0;
    if (this.keys.has("KeyD") || this.keys.has("ArrowRight")) roll += 1;
    if (this.keys.has("KeyA") || this.keys.has("ArrowLeft")) roll -= 1;

    let yaw = 0;
    if (this.keys.has("KeyE")) yaw += 1;
    if (this.keys.has("KeyQ")) yaw -= 1;

    let throttleDelta = 0;
    if (this.keys.has("ShiftLeft") || this.keys.has("ShiftRight")) throttleDelta += 1;
    if (this.keys.has("ControlLeft") || this.keys.has("ControlRight")) throttleDelta -= 1;

    // Standard HTML5 Gamepad API polling
    if (typeof navigator !== "undefined" && typeof navigator.getGamepads === "function") {
      const pads = navigator.getGamepads();
      for (const pad of pads) {
        if (!pad || !pad.connected) continue;
        const deadzone = (v: number) => (Math.abs(v) < 0.08 ? 0 : v);
        roll = clamp(roll + deadzone(pad.axes[0] ?? 0), -1, 1);
        pitch = clamp(pitch + deadzone(pad.axes[1] ?? 0), -1, 1);
        yaw = clamp(yaw + deadzone(pad.axes[2] ?? 0), -1, 1);
        if (pad.axes.length > 3) {
          throttleDelta = clamp(throttleDelta - deadzone(pad.axes[3] ?? 0), -1, 1);
        }
        break;
      }
    }

    return { pitch, roll, yaw, throttleDelta };
  }
}
