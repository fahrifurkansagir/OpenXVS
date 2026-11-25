import { MISSION_M1_CAM_DRIFT } from "./m1_cam_drift.ts";
import { MISSION_M2_GPS_SPOOF } from "./m2_gps_spoof.ts";
import { MISSION_M3_MESH_DAA } from "./m3_mesh_daa.ts";
import { MISSION_M4_KSFO_AUTOLAND } from "./m4_ksfo_autoland.ts";
import type {
  MissionDefinition,
  MissionId,
  PhysicalPerturbationState,
} from "./types.ts";

export * from "./types.ts";
export {
  MISSION_M1_CAM_DRIFT,
  MISSION_M2_GPS_SPOOF,
  MISSION_M3_MESH_DAA,
  MISSION_M4_KSFO_AUTOLAND,
};

export const ALL_MISSIONS: readonly MissionDefinition[] = [
  MISSION_M1_CAM_DRIFT,
  MISSION_M2_GPS_SPOOF,
  MISSION_M3_MESH_DAA,
  MISSION_M4_KSFO_AUTOLAND,
];

export function getMissionById(id: MissionId): MissionDefinition {
  const found = ALL_MISSIONS.find((m) => m.id === id);
  if (!found) {
    throw new Error(`Unknown mission ID: ${id}`);
  }
  return found;
}

/**
 * Evaluate the cumulative physical perturbation state of a mission at simulation time `simTimeS`.
 * Pure deterministic function: applies `initialPerturbations` followed by all events with `timeS <= simTimeS`.
 */
export function evaluateMissionPerturbations(
  mission: MissionDefinition,
  simTimeS: number,
): PhysicalPerturbationState {
  let state: PhysicalPerturbationState = {
    camRBiasRad: mission.initialPerturbations.camRBiasRad,
    gnssSpoofVelNedMps: [
      mission.initialPerturbations.gnssSpoofVelNedMps[0],
      mission.initialPerturbations.gnssSpoofVelNedMps[1],
      mission.initialPerturbations.gnssSpoofVelNedMps[2],
    ],
    thirdPeerActive: mission.initialPerturbations.thirdPeerActive,
    detectorMode: mission.initialPerturbations.detectorMode,
    fogContrast: mission.initialPerturbations.fogContrast,
    lwirEfvsEnabled: mission.initialPerturbations.lwirEfvsEnabled,
    runwayCornerBiasPx: mission.initialPerturbations.runwayCornerBiasPx,
  };

  for (const ev of mission.events) {
    if (ev.timeS <= simTimeS) {
      const p = ev.patch;
      state = {
        camRBiasRad: p.camRBiasRad ?? state.camRBiasRad,
        gnssSpoofVelNedMps: p.gnssSpoofVelNedMps
          ? [p.gnssSpoofVelNedMps[0], p.gnssSpoofVelNedMps[1], p.gnssSpoofVelNedMps[2]]
          : state.gnssSpoofVelNedMps,
        thirdPeerActive: p.thirdPeerActive ?? state.thirdPeerActive,
        detectorMode: p.detectorMode ?? state.detectorMode,
        fogContrast: p.fogContrast ?? state.fogContrast,
        lwirEfvsEnabled: p.lwirEfvsEnabled ?? state.lwirEfvsEnabled,
        runwayCornerBiasPx: p.runwayCornerBiasPx ?? state.runwayCornerBiasPx,
      };
    }
  }
  return state;
}
