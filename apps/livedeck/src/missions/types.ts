/**
 * Pure physical/sensor perturbation types for OpenXVS Live Deck Red-Team Missions M1..M4.
 *
 * Anti-Scripting Lock (SP1_LIVEDECK_DESIGN.md §4.5 / M6):
 *   Missions ONLY specify initial seeded scenario parameters and time-stamped physical/sensor
 *   perturbation inputs. Simplex modes, alarms, isolations, and Go-Around latches MUST emerge
 *   solely from `xvs-wasm` (`xvs-rta-core` + `xvs-math-core`).
 */

export type MissionId = "M1" | "M2" | "M3" | "M4";

export type DetectorMode = "D0" | "D1_KF";

export interface PhysicalPerturbationState {
  readonly camRBiasRad: number;
  readonly gnssSpoofVelNedMps: readonly [number, number, number];
  readonly thirdPeerActive: boolean;
  readonly detectorMode: DetectorMode;
  readonly fogContrast: number;
  readonly lwirEfvsEnabled: boolean;
  readonly runwayCornerBiasPx: number;
}

export type PhysicalPerturbationPatch = Partial<PhysicalPerturbationState>;

export const ALLOWED_PERTURBATION_KEYS = [
  "camRBiasRad",
  "gnssSpoofVelNedMps",
  "thirdPeerActive",
  "detectorMode",
  "fogContrast",
  "lwirEfvsEnabled",
  "runwayCornerBiasPx",
] as const;

export interface TimedPerturbationEvent {
  readonly timeS: number;
  readonly label: string;
  readonly patch: PhysicalPerturbationPatch;
}

export interface MissionDefinition {
  readonly id: MissionId;
  readonly scenarioIndex: number;
  readonly name: string;
  readonly summary: string;
  readonly seed: number;
  readonly durationS: number;
  readonly initialAltAglM: number;
  readonly referenceRunwayWgs84HM?: number;
  readonly domainShiftNotice?: string;
  readonly initialPerturbations: PhysicalPerturbationState;
  readonly events: readonly TimedPerturbationEvent[];
}

export function createNominalPerturbations(): PhysicalPerturbationState {
  return {
    camRBiasRad: 0.0,
    gnssSpoofVelNedMps: [0.0, 0.0, 0.0],
    thirdPeerActive: false,
    detectorMode: "D1_KF",
    fogContrast: 1.0,
    lwirEfvsEnabled: false,
    runwayCornerBiasPx: 0.0,
  };
}
